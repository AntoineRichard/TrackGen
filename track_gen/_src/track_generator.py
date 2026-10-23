"""Top-level facade for the batched track generator.

Runs the pure-Warp track pipeline (generation -> resample -> relax -> inflate, all
NVIDIA Warp kernels) and returns a fully-populated :class:`Track`. The pipeline is a
:class:`track_gen.pipeline.Program` built from ``legacy.to_pipeline_spec(config)``
(roadmap PR10a): the ordered built-in stages, byte-identical to the 0.2.0.dev2
composition ``warp_pipeline._run_pipeline``. The
public dataclasses ``TrackGenConfig`` and ``Track`` live in the dependency-free leaf
module ``types.py``; this facade re-exports ``TrackGenerator`` as the package's
top-level entry point.

TrackGenerator pre-allocates a single :class:`Track` instance in ``__init__`` and
returns the SAME instance from every ``generate()`` call (stable ``.ptr`` pointers).
The generator operates on a fixed configured batch (``config.num_envs``) and the same
persistent ``Track`` is reused across calls; callers that need a snapshot must use
``Track.clone()``.

Execution (see :class:`track_gen.exec.GraphRunner`): construction runs ONE real eager
generation (``prepare``: loads every kernel the hot path uses, and leaves the batch for
the rng's current seeds in the result). ``generate()`` then, per call:

- inside a capture on the resolved stream (Warp ``ScopedCapture``,
  ``capture_begin(external=True)``, ``torch.cuda.graph``): RECORDS the pipeline into the
  caller's graph (effective-seed prologue included, so each replay reads the live seeds
  and epochs);
- otherwise on CUDA: captures the pipeline once into :attr:`TrackGenerator.graph` on a
  private stream and replays it on the call's stream (one more cached graph per
  ``(advance, mask)`` variant, see :meth:`TrackGenerator.generate`);
- on the Warp ``cpu`` device: runs eagerly.

A pipeline with HOST stages (a ``capturable=False`` generator, or a stage of
``CaptureClass.HOST``) runs its HOST segments eagerly and replays the rest from per-segment
graphs on CUDA, and refuses a caller's capture (``NotCapturableError`` naming them).

Seeds (:mod:`track_gen.seeds`): the generator keys env ``e`` by ``(seed[e], epoch[e])``.
``seeds`` binds the rng's ``seeds_warp`` by pointer (reseeding the rng keeps working);
``generate(advance=True, mask=m)`` bumps the epoch of the masked envs on device first,
so a replayed caller graph produces a NEW batch for them on every replay.
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
from .types import Track, TrackGenConfig

if TYPE_CHECKING:  # pragma: no cover - track_gen.plugin stays lazy at import time
    from ..pipeline.spec import PipelineSpec
    from ..plugin.params import ParamTable

__all__ = [
    "Track",
    "TrackGenConfig",
    "TrackGenerator",
]


class TrackGenerator:
    """Top-level facade: run the pure-Warp track-generation pipeline and return a
    :class:`Track`.

    Generation, resample, relaxation and inflation are all expressed as NVIDIA Warp
    kernels (a :class:`track_gen.pipeline.Program` of the default stage list,
    ``track_gen.pipeline.legacy.to_pipeline_spec(config)``), runnable on the Warp
    ``cpu`` and ``cuda`` devices.
    ``config.generator`` must be a registered generator (see
    ``track_gen.plugin.registry.available("generator")``).

    The generator is fixed-batch: it always operates on exactly ``config.num_envs``
    environments. The output :class:`Track` is pre-allocated once in ``__init__`` (by
    the pipeline planner) and reused across calls: ``generate()`` always returns
    the same ``self._track`` instance, writing new results into the same wp.array buffers
    in place. This ensures stable ``.ptr`` pointers so downstream CUDA graph consumers can
    bake the device addresses once. Use ``Track.clone()`` to obtain an independent snapshot.

    Construction runs one real eager generation (warm-up: every kernel the hot path uses
    is loaded, and the result already holds the batch for the rng's current seeds), so
    it must happen outside any CUDA graph capture. Afterwards ``generate()`` /
    :meth:`enqueue` can be recorded into a caller's CUDA graph (Warp ``ScopedCapture``,
    ``capture_begin(external=True)``, ``torch.cuda.graph``); outside a capture,
    ``generate()`` replays the generator's own graph (:attr:`graph`) on CUDA and runs
    eagerly on the Warp ``cpu`` device.

    Per-env seeding lives in :attr:`seeds` (a :class:`~track_gen.seeds.SeedState`:
    the rng's seeds bound by pointer plus a device-side epoch); ``generate(advance=True,
    mask=...)`` advances the epoch on device to regenerate selected envs, and
    ``Track.generation`` reports the epoch each env was built at.

    ``TrackGenerator(spec)`` runs any ``track_gen.pipeline.PipelineSpec`` (e.g.
    :func:`track_gen.pipeline.presets.track` with plugin relax / width / elevation /
    validity stages); :attr:`program` is the built pipeline.

    ``relax_solver`` and ``z_profile`` name any registered relax / elevation stage;
    ``smooth_finish=True`` is not supported (construction raises ``ValueError``).
    """

    def __init__(self, config, rng=None, *, seeds=None,
                 stream: "wp.Stream | None" = None,
                 out: "Track | None" = None) -> None:
        """
        Args:
            config: A ``TrackGenConfig`` (the default stage list,
                ``track_gen.pipeline.legacy.to_pipeline_spec(config)``), or a
                ``track_gen.pipeline.PipelineSpec`` (e.g. from
                ``track_gen.pipeline.presets.track``) run as given.
                ``config.generator`` / the spec's generator must be a registered
                generator (see ``track_gen.plugin.registry.available("generator")``).
                The generator keeps a private copy (a ``wp.Device`` value is shared,
                not copied): mutating ``config`` afterwards has no effect.
            rng: Optional ``PerEnvSeededRNG`` on the generator's device; its per-env
                seed values seed the pipeline's built-in Warp RNG (one base seed per
                env). Its ``seeds_warp`` array is bound by pointer as :attr:`seeds`
                (a :class:`~track_gen.seeds.SeedState` that also owns the per-env
                epoch): the CURRENT seeds are read on every generation (also by every
                graph replay). The generator never writes them;
                ``rng.set_seeds_warp`` does, in place. ``None`` (and no ``seeds``):
                owned seeds ``0 .. E-1``.
            seeds: Instead of ``rng``: anything :class:`~track_gen.seeds.SeedState`
                accepts (an int base seed, an ``[E]`` int32 ``wp.array`` bound by
                pointer, a ``PerEnvSeededRNG``) or a ``SeedState``.
            stream: optional default ``wp.Stream`` for calls that pass no ``stream=``
                and run outside an ambient ``LaunchContext`` scope (see
                :mod:`track_gen.exec`).
            out: optional caller-owned ``Track`` the pipeline writes INTO (roadmap
                PR13b): the generator's result IS ``out`` (stable caller pointers, so a
                caller graph built over it keeps working). Every field is checked
                against the pipeline's layout (``generation`` / ``fail_mask`` may be
                ``None``, and ``half_width``). A view of a larger result
                (``out=big.view(e0, e1)``) makes a mixed batch: several generators
                writing disjoint env ranges of one ``Track`` (overlapping a live
                generator's result raises ``PlanError``: release a generator
                before building another into its storage); with ``seeds`` / ``rng``
                left ``None`` (or an int) env ``e`` of the view is seeded with
                ``base + e0 + e``, as in one batch. A seed array, a
                ``PerEnvSeededRNG`` (``PerEnvSeededRNG(seeds=0, num_envs=4)`` holds
                seeds ``0 .. 3`` for EVERY view it is given to) or a ``SeedState`` is
                bound as given, with no offset: give each generator of a mixed batch
                its own seeds (e.g. slices of one seed array). An ``out.attributes``
                key is bound when this pipeline exports it; on a view, any other key
                is filled once with its padding (NaN / ``-1``) and never written (on
                a whole result it raises ``ValueError``). The construction-time
                warm-up writes into ``out``.

        Raises:
            ValueError: if both ``rng`` and ``seeds`` are given.
            TypeError: if ``config`` is neither a ``TrackGenConfig`` nor a
                ``PipelineSpec``.
            ValueError: if ``config`` is a gate pipeline spec (``BatchSpec.max_gates``
                set: use ``GateGenerator``).
            ValueError: if the number of seeds in ``rng`` does not match
                ``config.num_envs``.
            ValueError: if ``config.generator`` is not a registered generator.
            track_gen.plugin.PluginLoadError: if ``config.generator`` (or a stage the
                spec / ``relax_solver`` / ``z_profile`` names) is provided by a
                third-party plugin that failed to load (quarantined).
            ValueError: if a generator-specific rule of the selected generator fails
                (its ``Params.validate``: e.g. ``voronoi_control_points >= 6``, the
                ``repulsive_*`` rules), ``config.generator_params`` has unknown keys
                or conflicts with the flat fields, or a value is outside its lo/hi.
            TypeError: if ``config.generator_params`` is not the generator's Params
                class (or a mapping of its fields).
            ValueError: if ``rng.seeds_warp`` is not an ``[E]`` int32 array on
                ``config.device`` (it is bound by pointer, so the rng must live on
                the generator's device).
            ValueError: if ``config.relax_solver`` is not a registered relax stage
                (the oracle-only ``"energy"`` / ``"tp_sobolev"`` included), if
                ``config.z_profile`` is not a registered elevation stage, or if
                ``config.smooth_finish`` is ``True``.
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
        from ..plugin.params import _new_param_table
        from .generator_binding import _StageGenerator, bind_generator

        if isinstance(config, PipelineSpec):
            spec = config
            _planner._check_spec_shape(spec)
            _planner._check_topology(spec, registry)
            if spec.batch.max_gates is not None:
                raise ValueError("TrackGenerator runs a track pipeline; this spec is a gate "
                                 "pipeline (BatchSpec.max_gates set): use GateGenerator")
            E_spec = int(spec.batch.num_envs)
        elif isinstance(config, TrackGenConfig):
            spec = None
            E_spec = int(config.num_envs)
        else:
            raise TypeError(f"TrackGenerator expects a TrackGenConfig or a "
                            f"track_gen.pipeline.PipelineSpec, got {type(config).__name__}")
        # Catch an env-count mismatch at construction (clear message) rather than as an
        # opaque wp.copy size error on the first generation.
        if rng is not None and int(rng.seeds_warp.shape[0]) != E_spec:
            field = "spec.batch.num_envs" if spec is not None else "config.num_envs"
            raise ValueError(
                f"rng has {int(rng.seeds_warp.shape[0])} seeds but {field}="
                f"{E_spec!r}; construct the rng with num_envs={E_spec!r}."
            )

        # Resolve the generator and its Params (generator_params, or from_legacy of the
        # flat fields) and run the generator-specific validation, for the selected
        # generator only. Host-only: nothing is allocated yet.
        if spec is not None:
            binding = bind_generator("generator", _legacy.generator_config(spec),
                                     registry=registry)
            self._user_config = spec
        else:
            binding = bind_generator("generator", config, registry=registry)
            # The default stage list (raises the relax_solver / z_profile /
            # smooth_finish errors).
            spec = _legacy._to_pipeline_spec(binding.run_config, binding)
            # _user_config is the config as passed (reported by `config`).
            self._user_config = binding.user_config
        self._generator_spec = binding.pipeline_spec
        # Snapshots: the pipeline bakes host scalars from the config into its launches
        # (and into any recorded graph), so later mutation of the caller's object must
        # not change what this generator emits. _config is what the pipeline reads (the
        # resolved built-in Params written back onto the flat fields).
        self._config = binding.run_config
        config = self._config
        source = rng if rng is not None else seeds
        self._rng = source if _is_rng(source) else None
        # out= a view of a larger result: default / int seeds are offset by the view's
        # first env (a mixed batch is seeded like one batch, roadmap PR13b).
        source = _planner._offset_seeds(source, out)

        E = int(config.num_envs)
        dev = str(config.device)
        # Refuse a capture BEFORE allocating: the allocations and zero-fills below run
        # on Warp's current stream and would otherwise land in the caller's graph.
        _check_default_stream(dev, stream)
        ensure_not_capturing(dev, default_stream=stream, label=config.generator)

        # Per-env (seed, epoch): the rng's seeds_warp bound by pointer (read live at
        # every generation, never written by the generator) plus an owned epoch. Its
        # methods default to this generator's stream, so they stay ordered with it.
        seed_state = source if isinstance(source, _SeedStateType) else \
            SeedState(E, dev, source, default_stream=stream)

        # Plan and allocate the pipeline once: the persistent output Track, the [E]
        # effective-seed buffer (written from (seeds, epoch) at the start of every
        # generation, inside the graph), every channel and stage scratch. All are
        # reused across every generation; the hot path never allocates.
        program = _planner._build(spec, binding=binding, seeds=seed_state, stream=stream,
                                  prepare=False, owner=self, registry=registry, out=out,
                                  register=False)
        self._track: Track = program.result
        # The PR9 scratch holder, as a view aliasing the program's buffers.
        self._scratch = _legacy._scratch_view(program)
        self._seed_buf: wp.array = program.seed
        self._runner = program.runner
        self._seed_state: SeedState = seed_state
        # Pipelines with HOST stages (a host-driven generator or stage) run those
        # eagerly (the rest from per-segment graphs on CUDA) and refuse to be recorded
        # into a caller's graph.
        self._capturable = program.capturable

        # Runtime parameters: the generator's, and every stage with a DYNAMIC field
        # (live [E] arrays, stable pointers), keyed by stage id.
        table = program.params
        self._params = _new_param_table({sid: table[sid] for sid in table
                                         if sid == "generator" or len(table[sid])})
        self._inputs = program.inputs
        self._regen_mask: wp.array = program.inputs["regen_mask"]
        # Latch (live -> work) of the dynamic params / inputs (the program runs it after
        # the effective-seed prologue of every generation); None for legacy generators.
        stage = self._generator_spec if isinstance(self._generator_spec, _StageGenerator) \
            else None
        self._latch = stage.latch if stage is not None and stage._pairs else None
        self._program = program

        # One real eager generation outside any capture (loads every module the hot
        # path uses; leaves the batch for the rng's current seeds in self._track), then
        # a host wait on its stream: the result is complete when construction returns.
        program.prepare()
        # Only now does this generator hold its result storage (the out= overlap
        # registry): a construction that raised leaves nothing to refuse a retry.
        _planner._register_result(program, program.result)

    # -- properties ---------------------------------------------------------------
    @property
    def config(self) -> "TrackGenConfig | PipelineSpec":
        """A copy of the configuration this generator was built with, as passed (a
        ``wp.Device`` value is shared, not copied), with a mapping
        ``generator_params`` normalised to its Params instance; for a generator built
        from a ``PipelineSpec``, that spec (immutable). ``TrackGenerator(gen.config,
        rng)`` builds an identical generator, with the construction-time (initial)
        values of any DYNAMIC parameters. A later ``params[...].set(...)`` is not
        reflected here: the current values are the live arrays,
        ``gen.params[stage_id][field]``.

        Read-only in effect: mutating the returned object (or the object passed to
        the constructor) does not change this generator. Build a new generator to
        change the configuration; dynamic parameters change through :attr:`params`.
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
        stage with a DYNAMIC field (e.g. ``"relax"`` for a relax plugin with one).

        ``params["generator"].params`` is the resolved Params instance (``None`` for a
        legacy generator without a Params class). The DYNAMIC fields of a Stage map to
        live ``[E]`` device arrays with stable pointers (``params["generator"]["amp"]``):
        write them from your own kernels (also inside your graph) or with
        ``params[stage_id].set("amp", value, stream=...)``; the next generation latches
        them for the envs it regenerates, with no re-capture. The built-in generators'
        and stages' fields are static (``set`` raises ``KeyError``), except
        ``width.from_channel``'s ``scale`` / ``lo`` / ``hi`` and a ``randomize.uniform``
        stage's ``lo`` / ``hi``; every stage's resolved params are ``program.params``. A field a ``randomize.*`` stage draws
        (``presets.track(randomize=...)``) is an unlatched array owned by that stage:
        written every generation (``params[stage][field]`` observes the draws), ``set``
        raises (set the randomizer's ``lo`` / ``hi`` instead).
        """
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
    def n_max(self) -> int:
        """``N_max``, the per-env point capacity (the ``Track``'s stride)."""
        return int(self._program.layout.stride)

    @property
    def graph(self) -> "wp.Graph | None":
        """The generator's own CUDA graph for plain ``generate()`` (no ``advance``):
        captured on the first such CUDA call made outside any capture, replayed by
        every later one. ``None`` on CPU, for a pipeline with HOST stages (its
        capturable segments get one graph each), and before that first call. Calls recorded into a caller's capture never create it.
        ``generate(advance=True, mask=m)`` uses one more cached graph per
        ``(advance, m.ptr)``."""
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
          kernels (e.g. inside your graph) and pass it as
          ``generate(advance=True, mask=gen.inputs["regen_mask"])``; its pointer
          never changes, so the generator's self-graph for it is captured once.
        - every per-env input a Stage generator declared
          (``StageBuilder.input(name, ...)``): a live ``[E]`` array, latched at each
          generation for the envs it regenerates, like a dynamic parameter;
        - every ``per="raw"`` input (``StageBuilder.input(..., per="raw")``, e.g. the
          ``centerline.*`` inputs of ``generator/external``): an ``[E * raw]`` array
          the generation READS (unlatched: write it on the generation's stream, or
          order the streams with events). :attr:`input_info` tells them apart.
        """
        return self._inputs

    @property
    def input_info(self) -> "types.MappingProxyType":
        """``name -> track_gen.pipeline.InputInfo`` of every entry of :attr:`inputs`
        (``program.input_info``, roadmap PR13b): its dtype, ``per`` (``"env"`` /
        ``"raw"``), length, declaring stage, and whether it is latched at the start of a
        generation or read during it."""
        return self._program.input_info

    @property
    def _graph(self) -> "wp.Graph | None":
        """Read-only alias of :attr:`graph` (backward compatibility)."""
        return self._runner.graph

    # -- enqueue body -------------------------------------------------------------
    def _enqueue(self, ctx: LaunchContext, *, advance: bool = False,
                 mask: "wp.array | None" = None) -> None:
        """Emit one whole generation on ``ctx``: [epoch advance] -> effective seeds
        -> the pipeline -> ``generation <- epoch`` (:meth:`track_gen.pipeline.Program.
        enqueue`).

        The ONLY generation body (prepare, eager, self-graph capture and caller
        recording all run it). Zero allocation and host-sync free; the seed prologue
        is part of it, so a recorded graph reads the live seeds and epochs on every
        replay (and, with ``advance``, advances the epochs on every replay).

        With a ``mask``, the prologue and the ``generation`` write cover the masked
        envs only: every other env keeps the effective seed its current track was
        built from, so the fixed-shape pipeline recomputes it to identical values
        even if its seed or epoch was changed since (an all-zero mask is an exact
        no-op).

        Args:
            advance: bump ``epoch[e]`` first (every env, or where ``mask[e] != 0``).
            mask: optional ``[E]`` int32 mask (validated by the caller).
        """
        self._program.enqueue(ctx, advance=advance, mask=mask)

    def _body(self, advance: bool, mask) -> "tuple[object, object]":
        """``(body, self-graph key)`` of one call; validates ``advance``/``mask``
        host-side, before any launch."""
        return self._program._body(advance, mask)

    def _run(self, ctx: "LaunchContext | None" = None) -> None:
        """Backward-compatible alias: run :meth:`_enqueue` on ``ctx`` (default:
        ``LaunchContext.current(device)``, Warp's current stream)."""
        self._enqueue(ctx or LaunchContext.current(str(self._config.device)))

    # -- public calls -------------------------------------------------------------
    def _check_num_or_ids(self, num_or_ids) -> None:
        if num_or_ids is None:
            return
        # Accept any integer scalar (incl. numpy ints from array shapes); reject bool
        # (an int subclass) and non-integers, which are treated as env-id sequences.
        if isinstance(num_or_ids, bool) or not isinstance(num_or_ids, numbers.Integral):
            raise TypeError(
                "TrackGenerator.generate() does not accept explicit environment ids; "
                "construct a generator with the desired fixed num_envs instead."
            )
        if int(num_or_ids) != self._config.num_envs:
            raise ValueError(
                f"TrackGenerator is fixed-batch for {self._config.num_envs} envs; "
                f"got num_or_ids={num_or_ids}. "
                f"Construct a new TrackGenerator with num_envs={num_or_ids} instead."
            )

    def generate(self, num_or_ids=None, *, stream=None, sync: bool = True,
                 advance: bool = False, mask: "wp.array | None" = None) -> Track:
        """Generate a batch of tracks for the fixed configured batch.

        This generator is fixed-batch: it always operates on exactly ``config.num_envs``
        environments. If integer ``num_or_ids`` is provided it is validated against the
        configured batch size and a ``ValueError`` is raised if they disagree (so existing
        call sites passing ``E == num_envs`` continue to work unchanged). Explicit
        environment-id sequences are rejected with ``TypeError`` because selected-env
        execution is not part of the fixed-batch graph-captured contract.

        Execution depends on where the call runs (the stream is resolved per
        :mod:`track_gen.exec`: ``stream=`` > ambient scope > constructor ``stream=`` >
        an enclosing ``torch.cuda.graph`` capture stream > Warp's current stream):

        - inside a capture on the resolved stream: the generation (seed refresh
          included) is RECORDED into the caller's graph — no sync, no self-capture,
          no ``wp.capture_launch``. Each replay of the caller's graph regenerates from
          the rng's seeds at replay time;
        - outside a capture on CUDA: the generator's own graph (:attr:`graph`,
          captured on the first such call) is replayed on the resolved stream;
        - on the Warp ``cpu`` device: eager;
        - a pipeline with HOST stages (``capturable=False`` generator, ``CaptureClass.HOST``
          stage): the HOST stages run eagerly, the rest replays from per-segment graphs
          on CUDA.

        Writes results into ``self._track`` in place and returns the SAME instance every
        call (stable ``.ptr`` pointers). Use ``Track.clone()`` to obtain an independent copy.

        Determinism: env ``e`` is keyed by ``(seeds.seeds[e], seeds.epoch[e])`` (see
        :mod:`track_gen.seeds`); every generation re-reads both on device, so repeated
        calls with unchanged seeds and epochs return the IDENTICAL batch. To vary the
        batch, either reseed the rng (e.g. ``rng.set_seeds_warp(new_seeds, None)``) or
        pass ``advance=True``, which bumps the epochs on device (capturable: a caller
        graph containing ``generate(advance=True)`` yields a new batch per replay).
        Reseeding the rng keeps the epochs; ``seeds.reset_epoch()`` restarts them at 0
        (then the batch equals a fresh generator's for those seeds).

        Masked regeneration (``advance=True, mask=m``): envs with ``m[e] != 0`` advance
        their epoch and get the track of their new ``(seed, epoch)``. Every other env's
        ``Track`` rows (and ``generation``) end the call bit for bit as they were, by
        construction (the pipeline is fixed-shape and recomputes every env, then the
        Program's commit restores the unmasked envs; see
        :mod:`track_gen.stages.commit`). A reseed or ``seeds.reset_epoch()`` therefore
        reaches an env at its next masked generation, or every env at the next
        generation without a mask. Within the call the unmasked rows are overwritten
        by the recomputation and then restored, both on the generation's stream: read
        the ``Track`` on that stream (or after an event recorded on it) once the call
        is enqueued; a reader on another stream with no such dependency may see the
        recomputed rows of an unmasked env.

        Args:
            num_or_ids: Optional. Either ``None`` or the integer batch size. When an
                integer is provided, it must equal ``config.num_envs``. Explicit
                environment-id sequences are not supported by this fixed-batch,
                graph-capturable facade.
            stream: optional ``wp.Stream`` (or ``"torch"``) to run on.
            sync: block the host until the generation finished on the resolved stream
                (default ``True``). Never blocks while that stream is capturing, nor
                while torch is capturing in the calling thread. ``sync=False`` returns
                immediately; results are stream-ordered (ready for later work on the
                same stream).
            advance: first advance the epoch (``seeds.advance(mask)``) on device, so
                the selected envs get a new track. Default ``False``.
            mask: optional ``[E]`` int32 array on the generator's device selecting the
                envs to advance (``mask[e] != 0``); read on device at execution (and
                replay) time. Requires ``advance=True``. ``None`` advances every env.
                Pass a STABLE buffer (e.g. ``inputs["regen_mask"]``): outside a
                capture each distinct mask pointer captures, and caches, one more
                self-graph.

        Returns:
            ``self._track`` — the same :class:`Track` instance every call (stable
            pointers); ``Track.generation`` holds the per-env epoch it was built at.

        Raises:
            TypeError: if ``num_or_ids`` is an explicit environment-id sequence.
            ValueError: if an integer ``num_or_ids`` differs from ``config.num_envs``;
                if ``mask`` is not an ``[E]`` int32 array on the generator's device,
                or is given without ``advance=True``.
            track_gen.exec.NotCapturableError: recording a pipeline with HOST stages
                (a ``capturable=False`` generator) into a capture, or running it under
                a CPU graph capture (raised before any launch, naming them).
            track_gen.exec.LegacyPluginStreamError: recording a legacy plugin
                generator on a stream that is not Warp's current stream.
        """
        self._check_num_or_ids(num_or_ids)
        body, key = self._body(advance, mask)
        self._runner.run(body, stream=stream, sync=sync, key=key)
        return self._track

    def enqueue(self, *, stream=None, advance: bool = False,
                mask: "wp.array | None" = None) -> Track:
        """Enqueue one generation without self-graphing or blocking.

        Inside a capture on the resolved stream the generation is recorded into the
        caller's graph (as :meth:`generate`); otherwise it runs eagerly on the resolved
        stream. Never captures :attr:`graph`, never host-syncs (a HOST stage's own
        ``ctx.host()`` waits on the stream aside; an iterative, BOUNDED stage runs its
        full iteration bound when called eagerly on CUDA; its self-graph skips converged
        windows on device). ``advance`` / ``mask`` as in :meth:`generate`.

        Returns:
            ``self._track`` (results are stream-ordered).

        Raises:
            ValueError: an invalid ``mask`` (see :meth:`generate`).
            track_gen.exec.NotCapturableError: recording a pipeline with HOST stages
                into a capture (raised before any launch).
        """
        body, _ = self._body(advance, mask)
        self._runner.run(body, stream=stream, policy="eager")
        return self._track
