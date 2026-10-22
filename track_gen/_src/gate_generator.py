"""Top-level facade for native batched gate sequence generation.

``GateGenerator`` mirrors the fixed-batch contract of ``TrackGenerator``: construction
owns persistent output and scratch buffers (and runs one real eager generation), while
each generation refreshes the per-env effective seeds (rng seeds + epoch) and writes into
the same ``GateSequence`` instance every call. Native gate generators provide centerline-generator
anchors through the registry; ``warp_gate`` owns the common ordering, collision, frame,
and validity pipeline.

Execution follows :class:`track_gen.exec.GraphRunner`, exactly like ``TrackGenerator``:
record into a caller's capture, self-graph replay on CUDA, eager on CPU. Seeding too:
``seeds`` (a :class:`track_gen.seeds.SeedState`) binds the rng's seeds by pointer and
owns a device-side epoch that ``generate(advance=True, mask=...)`` advances.
"""

import functools
import types

import warp as wp

from ..exec.context import LaunchContext
from ..exec.runner import GraphRunner
from ..seeds import SeedState
from .runtime import _snapshot_config
from .types import GateGenConfig, GateSequence

__all__ = ["GateGenConfig", "GateGenerator", "GateSequence"]


class GateGenerator:
    """Facade for fixed-batch native gate sequence generation.

    Mirrors the fixed-batch contract of :class:`~track_gen.TrackGenerator`: construction
    allocates persistent output and scratch buffers once (via ``warp_gate._gate_warp_alloc``),
    and every generation refreshes the per-env seed buffer then writes results into the
    same :class:`~track_gen.GateSequence` instance (stable ``.ptr`` pointers).
    Use ``GateSequence.clone()`` when an independent snapshot is needed.

    Construction runs one real eager generation (warm-up), so it must happen outside any
    CUDA graph capture. Afterwards ``generate()`` / :meth:`enqueue` can be recorded into a
    caller's CUDA graph; outside a capture ``generate()`` replays the generator's own graph
    (:attr:`graph`) on CUDA, unless the generator registers ``capturable=False``, in which
    case it runs eagerly on CUDA every call (and refuses to be recorded). On the Warp
    ``cpu`` device the pipeline runs eagerly.

    Per-env seeding lives in :attr:`seeds` (a :class:`~track_gen.seeds.SeedState`:
    the rng's seeds bound by pointer plus a device-side epoch); ``generate(advance=True,
    mask=...)`` advances the epoch on device to regenerate selected envs, and
    ``GateSequence.generation`` reports the epoch each env was built at.
    """

    def __init__(self, config: GateGenConfig, rng, *, stream: "wp.Stream | None" = None) -> None:
        """
        Args:
            config: The gate generation configuration. ``config.generator`` must be a
                registered gate generator; ``config.gate_ordering`` must be supported
                by that generator; and ``config.max_gates`` must satisfy the
                generator's capacity requirements. The generator keeps a private copy
                (a ``wp.Device`` value is shared, not copied): mutating ``config``
                afterwards has no effect.
            rng: A ``PerEnvSeededRNG`` instance with one seed per configured env, on
                the generator's device. Its ``seeds_warp`` array is bound by pointer as
                :attr:`seeds`: the CURRENT seeds are read on every generation (also by
                every replay). The generator never writes them; ``rng.set_seeds_warp``
                does, in place.
            stream: optional default ``wp.Stream`` for calls that pass no ``stream=``
                and run outside an ambient ``LaunchContext`` scope (see
                :mod:`track_gen.exec`).

        Raises:
            ValueError: if ``rng`` is ``None``.
            ValueError: if the number of seeds in ``rng`` does not match
                ``config.num_envs``.
            ValueError: if ``rng.seeds_warp`` is not an ``[E]`` int32 array on
                ``config.device`` (it is bound by pointer, so the rng must live on
                the generator's device).
            ValueError: if ``config.generator`` is not a registered gate generator
                (see ``track_gen.plugin.registry.available("gate_generator")``).
            track_gen.plugin.PluginLoadError: if ``config.generator`` is provided by
                a third-party plugin that failed to load (quarantined).
            ValueError: if ``config.gate_ordering`` is not supported by the selected
                gate generator.
            ValueError: if ``config.min_gates`` exceeds the maximum gate count the
                generator can produce, or if ``config.max_gates`` is too small for
                the generator's required capacity.
            track_gen.exec.CaptureError: if the resolved stream or Warp's current
                stream is capturing (construction allocates and runs a real warm-up
                pass); raised before anything is allocated.
        """
        if rng is None:
            raise ValueError("A random number generator must be provided.")
        # Catch an env-count mismatch at construction (clear message) rather than as an
        # opaque wp.copy size error on the first generation.
        if int(rng.seeds_warp.shape[0]) != int(config.num_envs):
            raise ValueError(
                f"rng has {int(rng.seeds_warp.shape[0])} seeds but config.num_envs="
                f"{config.num_envs!r}; construct the rng with num_envs={config.num_envs!r}."
            )

        from ..plugin import registry

        generator_spec = registry.get("gate_generator", config.generator)
        if config.gate_ordering not in generator_spec.supported_orderings:
            raise ValueError(
                f"gate generator {config.generator!r} does not support "
                f"gate_ordering {config.gate_ordering!r}; supported: "
                f"{sorted(generator_spec.supported_orderings)}"
            )

        required_max_gates = int(generator_spec.max_gates(config))
        if required_max_gates < int(config.min_gates):
            raise ValueError(
                f"GateGenConfig.min_gates={config.min_gates!r} is too large for "
                f"gate generator {config.generator!r}; generator can produce at most "
                f"{required_max_gates} gates."
            )
        if required_max_gates > int(config.max_gates):
            raise ValueError(
                f"GateGenConfig.max_gates={config.max_gates!r} is too small for "
                f"gate generator {config.generator!r}; required max_gates "
                f"{required_max_gates}."
            )

        # Snapshot (see TrackGenerator): host scalars from the config are baked into
        # the launches and any recorded graph.
        self._config = _snapshot_config(config)
        config = self._config
        self._rng = rng
        self._generator_spec = generator_spec
        # Non-capturable generators (host-driven loops / per-call allocation) run eagerly
        # on CUDA every call and refuse to be recorded into a caller's graph.
        self._capturable = generator_spec.capturable

        self._runner = GraphRunner(
            str(config.device),
            capturable=self._capturable,
            legacy_stream=not getattr(generator_spec, "accepts_ctx", False),
            default_stream=stream,
            owner=self,
            name=config.generator,
        )
        # Refuse a capture BEFORE allocating (see TrackGenerator).
        self._runner.ensure_not_capturing()

        from . import warp_gate

        self._gate_sequence: GateSequence
        self._gate_sequence, self._scratch = warp_gate._gate_warp_alloc(
            config, generator_spec=generator_spec
        )
        # Debug-only: per-env [E] int32 counter of frame fallbacks (near-vertical
        # or degenerate tangents) accumulated during the last generation. Reset
        # and rewritten each call; not part of the public GateSequence contract.
        # scratch is (gen_scratch, pos2, cum, perim, z, fallbacks) — fallbacks
        # is at index 5 (moved from 4 when the z-profile perim slot was added).
        self.frame_fallbacks: wp.array = self._scratch[5]

        E = int(config.num_envs)
        dev = str(config.device)
        # Per-env (seed, epoch): the rng's seeds_warp bound by pointer (never written
        # by the generator) plus an owned epoch (see track_gen.seeds). Its methods
        # default to this generator's stream, so they stay ordered with it.
        self._seed_state: SeedState = SeedState(E, dev, rng, default_stream=stream)
        self._seed_buf: wp.array = wp.empty(E, dtype=wp.int32, device=dev)
        # Persistent device input (see `inputs`): a stable [E] int32 mask buffer.
        self._regen_mask: wp.array = wp.zeros(E, dtype=wp.int32, device=dev)
        self._inputs = types.MappingProxyType({"regen_mask": self._regen_mask})

        # One real eager generation outside any capture (loads every module the hot
        # path uses; leaves the batch for the rng's current seeds in the result), then
        # a host wait on its stream: the result is complete when construction returns.
        self._runner.prepare(self._enqueue)

    # -- properties ---------------------------------------------------------------
    @property
    def config(self) -> GateGenConfig:
        """A copy of the configuration this generator was built with (a
        ``wp.Device`` value is shared, not copied).

        Read-only in effect: mutating the returned object (or the object passed to
        the constructor) does not change this generator.
        """
        return _snapshot_config(self._config)

    @property
    def graph(self) -> "wp.Graph | None":
        """The generator's own CUDA graph for plain ``generate()`` (no ``advance``):
        captured on the first such CUDA call made outside any capture, replayed by
        every later one. ``None`` on CPU, for a non-capturable generator, and before
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
        seeds -> pipeline -> ``generation <- epoch``.

        The ONLY generation body. Zero allocation and host-sync free; the seed
        prologue is part of it, so a recorded graph reads the live seeds and epochs
        on every replay (and, with ``advance``, advances them on every replay). With
        a ``mask`` the prologue and the ``generation`` write cover the masked envs
        only (see TrackGenerator._enqueue).
        """
        from . import warp_gate

        if advance:
            self.seeds.advance(mask, ctx=ctx)
        self.seeds.effective(self._seed_buf, mask, ctx=ctx)
        warp_gate._run_gate_pipeline(
            self._config,
            self._seed_buf,
            out=self._gate_sequence,
            scratch=self._scratch,
            generator_spec=self._generator_spec,
            ctx=ctx,
        )
        if self._gate_sequence.generation is not None:
            self.seeds._write_generation(self._gate_sequence.generation, mask, ctx)

    def _body(self, advance: bool, mask) -> "tuple[object, object]":
        """``(body, self-graph key)`` of one call; validates ``advance``/``mask``
        host-side, before any launch (see TrackGenerator._body)."""
        if mask is not None:
            if not advance:
                raise ValueError(
                    "mask= selects the envs whose epoch advances; it requires "
                    "advance=True (without it every env is regenerated identically)")
            self.seeds.check_mask(mask)
        if not advance:
            return self._enqueue, None
        key = (True, 0 if mask is None else int(mask.ptr))
        return functools.partial(self._enqueue, advance=True, mask=mask), key

    def _run(self, ctx: "LaunchContext | None" = None) -> None:
        """Backward-compatible alias: run :meth:`_enqueue` on ``ctx`` (default:
        ``LaunchContext.current(device)``, Warp's current stream)."""
        self._enqueue(ctx or LaunchContext.current(str(self._config.device)))

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
        - on the Warp ``cpu`` device, or for a ``capturable=False`` generator: eager.

        Writes results into ``self._gate_sequence`` in place and returns the SAME
        instance every call (stable ``.ptr`` pointers). Use ``GateSequence.clone()``
        to obtain an independent snapshot.

        Determinism and masked regeneration follow :meth:`TrackGenerator.generate`:
        env ``e`` is keyed by ``(seeds.seeds[e], seeds.epoch[e])``; ``advance=True``
        bumps the epochs (of the envs with ``mask[e] != 0``) on device first; the
        fixed-shape pipeline still recomputes every env, and the unmasked ones keep
        the effective seed their current gates were built from, so they are
        recomputed to identical values (a reseed or epoch reset reaches an env at its
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
            track_gen.exec.NotCapturableError: recording a ``capturable=False``
                generator into a capture (raised before any launch).
            track_gen.exec.LegacyPluginStreamError: recording a legacy plugin
                generator on a stream that is not Warp's current stream.
        """
        if num_or_ids is not None:
            if not isinstance(num_or_ids, int):
                raise TypeError(
                    "GateGenerator.generate() does not accept explicit environment ids; "
                    "construct a generator with the desired fixed num_envs instead."
                )
            if num_or_ids != self._config.num_envs:
                raise ValueError(
                    f"GateGenerator is fixed-batch for {self._config.num_envs} envs; "
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
        captures :attr:`graph`, never host-syncs. ``advance`` / ``mask`` as in
        :meth:`generate`.

        Returns:
            ``self._gate_sequence`` (results are stream-ordered).

        Raises:
            ValueError: an invalid ``mask`` (see :meth:`generate`).
            track_gen.exec.NotCapturableError: recording a ``capturable=False``
                generator into a capture (raised before any launch).
        """
        body, _ = self._body(advance, mask)
        self._runner.run(body, stream=stream, policy="eager")
        return self._gate_sequence
