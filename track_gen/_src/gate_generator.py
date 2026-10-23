"""Top-level facade for native batched gate sequence generation.

``GateGenerator`` mirrors the fixed-batch contract of ``TrackGenerator``: construction
plans and allocates a gate :class:`track_gen.pipeline.Program` once (and runs one real
eager generation), while each generation refreshes the per-env effective seeds (rng
seeds + epoch) and writes into the same ``GateSequence`` instance every call. The
Program is built from ``legacy.to_gate_pipeline_spec(config)`` (roadmap PR11: the
generator's anchors -> order -> [normalize] -> sphere relax -> [arc length ->
elevation] -> lift -> tangents -> frames -> validity, byte-identical to 0.2.0.dev3's
``warp_gate._run_gate_pipeline``), or from any gate ``PipelineSpec``
(:func:`track_gen.pipeline.presets.gates`).

Execution follows :class:`track_gen.exec.GraphRunner`, exactly like ``TrackGenerator``:
record into a caller's capture, self-graph replay on CUDA, eager on CPU. Seeding too:
``seeds`` (a :class:`track_gen.seeds.SeedState`) binds the rng's seeds by pointer and
owns a device-side epoch that ``generate(advance=True, mask=...)`` advances.
"""

import numbers
import types
from typing import TYPE_CHECKING

import warp as wp

from ..exec.context import LaunchContext
from ..exec.runner import _check_default_stream, ensure_not_capturing
from ..seeds import SeedState, _is_rng
# isinstance checks use this alias: tests monkeypatch the module's SeedState name.
from ..seeds import SeedState as _SeedStateType
from .runtime import _snapshot_config
from .types import GateGenConfig, GateSequence

if TYPE_CHECKING:  # pragma: no cover - track_gen.plugin stays lazy at import time
    from ..pipeline.spec import PipelineSpec
    from ..plugin.params import ParamTable

__all__ = ["GateGenConfig", "GateGenerator", "GateSequence"]


class GateGenerator:
    """Facade for fixed-batch native gate sequence generation.

    Mirrors the fixed-batch contract of :class:`~track_gen.TrackGenerator`: construction
    plans a gate :class:`track_gen.pipeline.Program` and allocates its output and
    scratch buffers once, and every generation refreshes the per-env seed buffer then
    writes results into the same :class:`~track_gen.GateSequence` instance (stable
    ``.ptr`` pointers). Use ``GateSequence.clone()`` when an independent snapshot is
    needed.

    Construction runs one real eager generation (warm-up), so it must happen outside any
    CUDA graph capture. Afterwards ``generate()`` / :meth:`enqueue` can be recorded into a
    caller's CUDA graph; outside a capture ``generate()`` replays the generator's own graph
    (:attr:`graph`) on CUDA. A pipeline with HOST stages (``capturable=False`` /
    ``CaptureClass.HOST``) runs them eagerly every call and replays the rest from
    per-segment graphs on CUDA (and refuses to be recorded). On the Warp ``cpu`` device
    the pipeline runs eagerly.

    Per-env seeding lives in :attr:`seeds` (a :class:`~track_gen.seeds.SeedState`:
    the rng's seeds bound by pointer plus a device-side epoch); ``generate(advance=True,
    mask=...)`` advances the epoch on device to regenerate selected envs, and
    ``GateSequence.generation`` reports the epoch each env was built at.

    ``GateGenerator(spec)`` runs any gate ``track_gen.pipeline.PipelineSpec`` (e.g.
    :func:`track_gen.pipeline.presets.gates`, also with a track generator's stations);
    :attr:`program` is the built pipeline.
    """

    def __init__(self, config, rng=None, *, seeds=None,
                 stream: "wp.Stream | None" = None) -> None:
        """
        Args:
            config: A ``GateGenConfig`` (the default stage list,
                ``track_gen.pipeline.legacy.to_gate_pipeline_spec(config)``), or a gate
                ``track_gen.pipeline.PipelineSpec`` (``BatchSpec.max_gates`` set, e.g.
                from ``track_gen.pipeline.presets.gates``) run as given.
                ``config.generator`` must be a registered gate generator;
                ``config.gate_ordering`` a registered order the generator supports
                (``None``, the default: the generator's default order, ``"ccw"`` for
                the built-ins);
                ``config.z_profile`` a registered elevation stage; and
                ``config.max_gates`` must satisfy the generator's capacity. The
                generator keeps a private copy (a ``wp.Device`` value is shared, not
                copied): mutating ``config`` afterwards has no effect.
            rng: Optional ``PerEnvSeededRNG`` with one seed per configured env, on the
                generator's device. Its ``seeds_warp`` array is bound by pointer as
                :attr:`seeds`: the CURRENT seeds are read on every generation (also by
                every replay). The generator never writes them; ``rng.set_seeds_warp``
                does, in place. ``None`` (and no ``seeds``): owned seeds ``0 .. E-1``.
            seeds: Instead of ``rng``: anything :class:`~track_gen.seeds.SeedState`
                accepts (an int base seed, an ``[E]`` int32 ``wp.array`` bound by
                pointer, a ``PerEnvSeededRNG``) or a ``SeedState``.
            stream: optional default ``wp.Stream`` for calls that pass no ``stream=``
                and run outside an ambient ``LaunchContext`` scope (see
                :mod:`track_gen.exec`).

        Raises:
            ValueError: if both ``rng`` and ``seeds`` are given.
            TypeError: if ``config`` is neither a ``GateGenConfig`` nor a
                ``PipelineSpec``.
            ValueError: if ``config`` is a track pipeline spec (``max_gates`` unset).
            ValueError: if the number of seeds in ``rng`` does not match
                ``config.num_envs``.
            ValueError: if ``rng.seeds_warp`` is not an ``[E]`` int32 array on
                ``config.device`` (it is bound by pointer, so the rng must live on
                the generator's device).
            ValueError: if ``config.generator`` is not a registered gate generator
                (see ``track_gen.plugin.registry.available("gate_generator")``).
            track_gen.plugin.PluginLoadError: if ``config.generator`` (or the order /
                elevation it names) is provided by a third-party plugin that failed to
                load (quarantined).
            ValueError: if a generator-specific rule of the selected gate generator
                fails (its ``Params.validate``), ``config.generator_params`` has unknown
                keys or conflicts with the flat fields, or a value is outside its lo/hi.
            TypeError: if ``config.generator_params`` is not the gate generator's
                Params class (or a mapping of its fields).
            ValueError: if ``config.gate_ordering`` is not a registered order, or not
                one the selected gate generator supports; if ``config.z_profile`` is not
                a registered elevation stage.
            ValueError: if ``config.min_gates`` exceeds the maximum gate count the
                generator can produce, or if ``config.max_gates`` is too small for
                the generator's required capacity.
            track_gen.pipeline.PlanError: an invalid ``PipelineSpec``.
            track_gen.exec.CaptureError: if the resolved stream or Warp's current
                stream is capturing (construction allocates and runs a real warm-up
                pass); raised before anything is allocated.
        """
        if rng is not None and seeds is not None:
            raise ValueError("pass rng= or seeds=, not both")
        from ..pipeline import legacy as _legacy
        from ..pipeline import planner as _planner
        from ..pipeline.spec import PipelineSpec
        from ..plugin import registry
        from .generator_binding import bind_generator

        if isinstance(config, PipelineSpec):
            spec = config
            _planner._check_spec_shape(spec)
            if spec.batch.max_gates is None:
                raise ValueError("GateGenerator runs a gate pipeline (BatchSpec.max_gates "
                                 "set, e.g. presets.gates); this spec is a track pipeline: "
                                 "use TrackGenerator")
            E_spec = int(spec.batch.num_envs)
        elif isinstance(config, GateGenConfig):
            spec = None
            E_spec = int(config.num_envs)
        else:
            raise TypeError(f"GateGenerator expects a GateGenConfig or a "
                            f"track_gen.pipeline.PipelineSpec, got {type(config).__name__}")
        # Catch an env-count mismatch at construction (clear message) rather than as an
        # opaque wp.copy size error on the first generation.
        if rng is not None and int(rng.seeds_warp.shape[0]) != E_spec:
            field = "spec.batch.num_envs" if spec is not None else "config.num_envs"
            raise ValueError(
                f"rng has {int(rng.seeds_warp.shape[0])} seeds but {field}="
                f"{E_spec!r}; construct the rng with num_envs={E_spec!r}."
            )

        # Resolve the gate generator and its Params (generator_params, or from_legacy of
        # the flat fields) and run its validation, for the selected generator only;
        # then the default stage list (the ordering / capacity / z-profile checks read
        # the resolved run config, so max_gates sees generator_params).
        if spec is not None:
            binding = _planner._bind(spec, registry)
            self._user_config = spec
            # Always a GateGenConfig (a stations source binds a TrackGenConfig).
            self._config = _legacy.gate_generator_config(spec, registry=registry)
        else:
            binding = bind_generator("gate_generator", config, registry=registry)
            spec = _legacy._to_gate_pipeline_spec(binding.run_config, binding,
                                                  registry=registry)
            # _user_config is the config as passed (reported by `config`); _config the
            # resolved run config.
            self._user_config = binding.user_config
            self._config = binding.run_config
        self._generator_spec = binding.pipeline_spec
        # The generator binding's run config: a GateGenConfig for an anchors / legacy
        # generator, the centerline generator's TrackGenConfig for a stations source.
        self._generator_config = binding.run_config
        source = rng if rng is not None else seeds
        self._rng = source if _is_rng(source) else None

        E = int(spec.batch.num_envs)
        dev = str(spec.batch.device)
        # Refuse a capture BEFORE allocating (see TrackGenerator).
        _check_default_stream(dev, stream)
        ensure_not_capturing(dev, default_stream=stream, label=binding.name)
        # Per-env (seed, epoch): the rng's seeds_warp bound by pointer (never written
        # by the generator) plus an owned epoch (see track_gen.seeds). Its methods
        # default to this generator's stream, so they stay ordered with it.
        seed_state = source if isinstance(source, _SeedStateType) else \
            SeedState(E, dev, source, default_stream=stream)

        program = _planner._build(spec, binding=binding, seeds=seed_state, stream=stream,
                                  prepare=False, owner=self, registry=registry)
        self._program = program
        self._gate_sequence: GateSequence = program.result
        self._seed_buf: wp.array = program.seed
        self._runner = program.runner
        self._seed_state: SeedState = seed_state
        # Pipelines with HOST stages (a host-driven generator or stage) run those
        # eagerly (the rest from per-segment graphs on CUDA) and refuse to be recorded
        # into a caller's graph.
        self._capturable = program.capturable
        # Runtime parameters: the generator's, and every stage with a DYNAMIC field
        # (live [E] arrays, stable pointers), keyed by stage id.
        from ..plugin.params import _new_param_table

        table = program.params
        self._params = _new_param_table({sid: table[sid] for sid in table
                                         if sid == "generator" or len(table[sid])})
        self._inputs = program.inputs
        self._regen_mask: wp.array = program.inputs["regen_mask"]
        # Debug-only: per-env [E] int32 counter of frame fallbacks (near-vertical or
        # degenerate tangents) accumulated during the last generation: the frames
        # stage's scratch (None when the spec replaced it). Not part of the public
        # GateSequence contract.
        self.frame_fallbacks: "wp.array | None" = \
            program._scratch.get("gate_frames", {}).get("fallbacks")
        # The generator's raw output, by name for track_gen.testing (the raw output
        # check): the anchors ([E*A] vec2f), or the gates a legacy spec wrote.
        self._raw_anchors: wp.array = program._channel_or_none("anchors.xy")
        if self._raw_anchors is None:
            self._raw_anchors = program.tap("gates.xy", after="generator")

        # One real eager generation outside any capture (loads every module the hot
        # path uses; leaves the batch for the rng's current seeds in the result), then
        # a host wait on its stream: the result is complete when construction returns.
        program.prepare()

    # -- properties ---------------------------------------------------------------
    @property
    def config(self) -> "GateGenConfig | PipelineSpec":
        """A copy of the configuration this generator was built with, as passed (a
        ``wp.Device`` value is shared, not copied), with a mapping
        ``generator_params`` normalised to its Params instance; for a generator built
        from a ``PipelineSpec``, that spec (immutable). ``GateGenerator(gen.config,
        rng)`` builds an identical generator, with the construction-time (initial)
        values of any DYNAMIC parameters. A later ``params[...].set(...)`` is not
        reflected here: the current values are the live arrays,
        ``gen.params[stage_id][field]``.

        Read-only in effect: mutating the returned object (or the object passed to
        the constructor) does not change this generator.
        """
        from ..pipeline.spec import PipelineSpec

        if isinstance(self._user_config, PipelineSpec):
            return self._user_config
        return _snapshot_config(self._user_config)

    @property
    def program(self):
        """The :class:`track_gen.pipeline.Program` this generator runs (preview): its
        taps (:meth:`~track_gen.pipeline.Program.tap`), fail bits and ``on_invalid``."""
        return self._program

    @property
    def params(self) -> "ParamTable":
        """Runtime parameters, keyed by stage id: ``"generator"``, plus every pipeline
        stage with a DYNAMIC field. ``params["generator"].params`` is the resolved
        Params instance (``None`` for a legacy gate generator without a Params class).
        The built-in generators' and stages' fields are all static (``set`` raises
        ``KeyError``); every stage's resolved params are ``program.params``."""
        return self._params

    @property
    def num_envs(self) -> int:
        """``E``, the batch size (whether built from a config or a spec)."""
        return int(self._program.layout.num_envs)

    @property
    def device(self) -> str:
        """The Warp device the generator runs on (e.g. ``"cuda:0"``)."""
        return str(self._program.layout.device)

    @property
    def max_gates(self) -> int:
        """``max_gates``, the per-env gate capacity (the ``GateSequence``'s stride)."""
        return int(self._program.layout.stride)

    @property
    def graph(self) -> "wp.Graph | None":
        """The generator's own CUDA graph for plain ``generate()`` (no ``advance``):
        captured on the first such CUDA call made outside any capture, replayed by
        every later one. ``None`` on CPU, for a pipeline with HOST stages, and before
        that first call. ``generate(advance=True, mask=m)`` uses one more cached
        graph per ``(advance, m.ptr)``."""
        return self._runner.graph

    @property
    def seeds(self) -> SeedState:
        """The per-env :class:`~track_gen.seeds.SeedState`: the rng's ``seeds_warp``
        bound by pointer (never written by the generator) plus the device-side
        ``epoch`` that ``generate(advance=True, ...)`` advances. Its buffers are baked
        into every recorded graph, so it is fixed for the generator's lifetime. Its
        methods run on the constructor's ``stream=`` when a call passes none."""
        return self._seed_state

    @property
    def inputs(self) -> "types.MappingProxyType":
        """Persistent device inputs (stable pointers, allocated at construction):

        - ``"regen_mask"``: ``[E]`` int32, zero-initialised. Write it from your own
          kernels and pass it as ``generate(advance=True,
          mask=gen.inputs["regen_mask"])``; its pointer never changes, so the
          generator's self-graph for it is captured once.
        """
        return self._inputs

    @property
    def _graph(self) -> "wp.Graph | None":
        """Read-only alias of :attr:`graph` (backward compatibility)."""
        return self._runner.graph

    # -- enqueue body -------------------------------------------------------------
    def _enqueue(self, ctx: LaunchContext, *, advance: bool = False,
                 mask: "wp.array | None" = None) -> None:
        """Emit one whole gate generation on ``ctx``: [epoch advance] -> effective
        seeds -> the pipeline -> ``generation <- epoch``
        (:meth:`track_gen.pipeline.Program.enqueue`).

        The ONLY generation body. Zero allocation and host-sync free; the seed
        prologue is part of it, so a recorded graph reads the live seeds and epochs
        on every replay (and, with ``advance``, advances them on every replay). With
        a ``mask`` the unmasked envs' ``GateSequence`` rows end the generation bit for
        bit as they were (see TrackGenerator._enqueue).
        """
        self._program.enqueue(ctx, advance=advance, mask=mask)

    def _body(self, advance: bool, mask) -> "tuple[object, object]":
        """``(body, self-graph key)`` of one call; validates ``advance``/``mask``
        host-side, before any launch (see TrackGenerator._body)."""
        return self._program._body(advance, mask)

    def _run(self, ctx: "LaunchContext | None" = None) -> None:
        """Backward-compatible alias: run :meth:`_enqueue` on ``ctx`` (default:
        ``LaunchContext.current(device)``, Warp's current stream)."""
        self._enqueue(ctx or LaunchContext.current(self.device))

    # -- public calls -------------------------------------------------------------
    def generate(self, num_or_ids=None, *, stream=None, sync: bool = True,
                 advance: bool = False, mask: "wp.array | None" = None) -> GateSequence:
        """Generate gates for the fixed configured batch.

        ``GateGenerator`` always operates on exactly ``config.num_envs`` environments.
        Passing an integer is accepted only when it matches that batch size. Explicit
        environment-id sequences are rejected because selected-env execution is outside
        the fixed-batch graph-capturable contract.

        Execution (stream resolved per :mod:`track_gen.exec`):

        - inside a capture on the resolved stream: RECORDED into the caller's graph
          (seed refresh included; no sync, no self-capture, no ``wp.capture_launch``);
        - outside a capture on CUDA: replay of :attr:`graph` (captured on the first
          such call) on the resolved stream;
        - on the Warp ``cpu`` device: eager; a pipeline with HOST stages runs them
          eagerly and the rest from per-segment graphs on CUDA.

        Writes results into ``self._gate_sequence`` in place and returns the SAME
        instance every call (stable ``.ptr`` pointers). Use ``GateSequence.clone()``
        to obtain an independent snapshot.

        Determinism and masked regeneration follow :meth:`TrackGenerator.generate`:
        env ``e`` is keyed by ``(seeds.seeds[e], seeds.epoch[e])``; ``advance=True``
        bumps the epochs (of the envs with ``mask[e] != 0``) on device first; the
        fixed-shape pipeline recomputes every env, then the Program's commit restores
        every unmasked env's ``GateSequence`` rows bit for bit (see
        :mod:`track_gen.stages.commit`; a reseed or epoch reset reaches an env at its
        next masked generation, or every env at the next one without a mask).

        Args:
            num_or_ids: Optional. Either ``None`` or the integer batch size. When an
                integer is provided, it must equal ``config.num_envs``. Explicit
                environment-id sequences are not supported.
            stream: optional ``wp.Stream`` (or ``"torch"``) to run on.
            sync: block the host until the generation finished on the resolved stream
                (default ``True``). Never blocks while that stream is capturing, nor
                while torch is capturing in the calling thread. ``sync=False`` returns
                immediately; results are stream-ordered.
            advance: first advance the epoch on device (default ``False``).
            mask: optional ``[E]`` int32 advance mask on the generator's device (read
                on device); requires ``advance=True``; ``None`` advances every env.
                Pass a stable buffer (e.g. ``inputs["regen_mask"]``): each distinct
                pointer captures and caches one more self-graph.

        Returns:
            ``self._gate_sequence`` — the same :class:`~track_gen.GateSequence` instance
            every call (stable pointers); ``GateSequence.generation`` holds the per-env
            epoch it was built at.

        Raises:
            TypeError: if ``num_or_ids`` is not an ``int`` (i.e. an explicit
                environment-id sequence was passed).
            ValueError: if an integer ``num_or_ids`` differs from ``config.num_envs``;
                if ``mask`` is invalid or given without ``advance=True``.
            track_gen.exec.NotCapturableError: recording a pipeline with HOST stages (a
                ``capturable=False`` generator) into a capture (raised before any
                launch, naming them).
            track_gen.exec.LegacyPluginStreamError: recording a legacy plugin
                generator on a stream that is not Warp's current stream.
        """
        if num_or_ids is not None:
            if isinstance(num_or_ids, bool) or not isinstance(num_or_ids, numbers.Integral):
                raise TypeError(
                    "GateGenerator.generate() does not accept explicit environment ids; "
                    "construct a generator with the desired fixed num_envs instead."
                )
            if int(num_or_ids) != self.num_envs:
                raise ValueError(
                    f"GateGenerator is fixed-batch for {self.num_envs} envs; "
                    f"got num_or_ids={num_or_ids}. Construct a new GateGenerator "
                    f"with num_envs={num_or_ids} instead."
                )
        body, key = self._body(advance, mask)
        self._runner.run(body, stream=stream, sync=sync, key=key)
        return self._gate_sequence

    def enqueue(self, *, stream=None, advance: bool = False,
                mask: "wp.array | None" = None) -> GateSequence:
        """Enqueue one generation without self-graphing or blocking.

        Inside a capture on the resolved stream the generation is recorded into the
        caller's graph; otherwise it runs eagerly on the resolved stream. Never
        captures :attr:`graph`, never host-syncs (a HOST stage's own ``ctx.host()``
        waits on the stream aside; a BOUNDED stage runs its full iteration bound when
        called eagerly on CUDA). ``advance`` / ``mask`` as in :meth:`generate`.

        Returns:
            ``self._gate_sequence`` (results are stream-ordered).

        Raises:
            ValueError: an invalid ``mask`` (see :meth:`generate`).
            track_gen.exec.NotCapturableError: recording a pipeline with HOST stages (a
                ``capturable=False`` generator) into a capture (raised before any
                launch, naming them).
        """
        body, _ = self._body(advance, mask)
        self._runner.run(body, stream=stream, policy="eager")
        return self._gate_sequence
