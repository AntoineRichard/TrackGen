"""Top-level facade for the batched track generator.

Runs the pure-Warp pipeline (``_run_pipeline``: generation -> resample -> relax ->
inflate, all NVIDIA Warp kernels) and returns a fully-populated :class:`Track`. The
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
- on the Warp ``cpu`` device (or for a ``capturable=False`` generator): runs eagerly.

Seeds (:mod:`track_gen.seeds`): the generator keys env ``e`` by ``(seed[e], epoch[e])``.
``seeds`` binds the rng's ``seeds_warp`` by pointer (reseeding the rng keeps working);
``generate(advance=True, mask=m)`` bumps the epoch of the masked envs on device first,
so a replayed caller graph produces a NEW batch for them on every replay.
"""

import functools
import numbers
import types

import warp as wp

from ..exec.context import LaunchContext
from ..exec.runner import GraphRunner
from ..seeds import SeedState
from .runtime import _snapshot_config
from .types import Track, TrackGenConfig

__all__ = [
    "Track",
    "TrackGenConfig",
    "TrackGenerator",
]


class TrackGenerator:
    """Top-level facade: run the pure-Warp track-generation pipeline and return a
    :class:`Track`.

    Generation, resample, relaxation and inflation are all expressed as NVIDIA Warp
    kernels (via ``_run_pipeline``), runnable on the Warp ``cpu`` and ``cuda`` devices.
    ``config.generator`` must be a registered generator (see
    ``track_gen.plugin.registry.available("generator")``).

    The generator is fixed-batch: it always operates on exactly ``config.num_envs``
    environments. The output :class:`Track` is pre-allocated once in ``__init__`` (via
    ``_inflate_warp_alloc``) and reused across calls: ``generate()`` always returns
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

    Only ``relax_solver="xpbd"`` and ``smooth_finish=False`` are supported; construction
    raises ``ValueError`` for other combinations.
    """

    def __init__(self, config: TrackGenConfig, rng, *, stream: "wp.Stream | None" = None) -> None:
        """
        Args:
            config: The pipeline configuration. ``config.generator`` must be a
                registered generator (see
                ``track_gen.plugin.registry.available("generator")``). The
                generator keeps a private copy (a ``wp.Device`` value is shared, not
                copied): mutating ``config`` afterwards has no effect.
            rng: A ``PerEnvSeededRNG`` instance on the generator's device; its
                per-env seed values seed the pipeline's built-in Warp RNG (one base
                seed per env). Its ``seeds_warp`` array is bound by pointer as
                :attr:`seeds` (a :class:`~track_gen.seeds.SeedState` that also owns
                the per-env epoch): the CURRENT seeds are read on every generation
                (also by every graph replay). The generator never writes them;
                ``rng.set_seeds_warp`` does, in place.
            stream: optional default ``wp.Stream`` for calls that pass no ``stream=``
                and run outside an ambient ``LaunchContext`` scope (see
                :mod:`track_gen.exec`).

        Raises:
            ValueError: if ``rng`` is ``None``.
            ValueError: if the number of seeds in ``rng`` does not match
                ``config.num_envs``.
            ValueError: if ``config.generator`` is not a registered generator.
            track_gen.plugin.PluginLoadError: if ``config.generator`` is provided by
                a third-party plugin that failed to load (quarantined).
            ValueError: if ``rng.seeds_warp`` is not an ``[E]`` int32 array on
                ``config.device`` (it is bound by pointer, so the rng must live on
                the generator's device).
            ValueError: if ``config.relax_solver`` is not ``"xpbd"``, or if
                ``config.smooth_finish`` is ``True``.
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
        self._generator_spec = registry.get("generator", config.generator)
        if config.relax_solver != "xpbd":
            raise ValueError(
                f"TrackGenerator only supports relax_solver='xpbd'; "
                f"got {config.relax_solver!r}."
            )
        if config.smooth_finish:
            raise ValueError(
                "TrackGenerator does not support smooth_finish=True; "
                "set smooth_finish=False."
            )
        # Snapshot: the pipeline bakes host scalars from the config into its launches
        # (and into any recorded graph), so later mutation of the caller's object must
        # not change what this generator emits.
        self._config = _snapshot_config(config)
        config = self._config
        self._rng = rng
        # Non-capturable generators (host-driven loops / per-call allocation) run eagerly
        # on CUDA every call and refuse to be recorded into a caller's graph.
        self._capturable = self._generator_spec.capturable

        E = int(config.num_envs)
        dev = str(config.device)
        self._runner = GraphRunner(
            dev,
            capturable=self._capturable,
            legacy_stream=not getattr(self._generator_spec, "accepts_ctx", False),
            default_stream=stream,
            owner=self,
            name=config.generator,
        )
        # Refuse a capture BEFORE allocating: the allocations and zero-fills below run
        # on Warp's current stream and would otherwise land in the caller's graph.
        self._runner.ensure_not_capturing()

        # Pre-allocate the persistent output Track buffers, scratch, and seed buffer
        # (one allocation per generator); all are reused across every generation.
        from . import warp_pipeline
        self._track: Track
        self._track, self._scratch = warp_pipeline._inflate_warp_alloc(
            config, generator_spec=self._generator_spec,
        )

        # Per-env (seed, epoch): the rng's seeds_warp bound by pointer (read live at
        # every generation, never written by the generator) plus an owned epoch. Its
        # methods default to this generator's stream, so they stay ordered with it.
        self._seed_state: SeedState = SeedState(E, dev, rng, default_stream=stream)

        # Pre-allocate the [E] int32 effective-seed buffer on the pipeline device. It is
        # written from (seeds, epoch) at the start of every generation (inside the
        # graph); no allocation occurs in the hot path.
        self._seed_buf: wp.array = wp.empty(E, dtype=wp.int32, device=dev)
        # Persistent device input (see `inputs`): a stable [E] int32 mask buffer.
        self._regen_mask: wp.array = wp.zeros(E, dtype=wp.int32, device=dev)
        self._inputs = types.MappingProxyType({"regen_mask": self._regen_mask})

        # One real eager generation outside any capture (loads every module the hot
        # path uses; leaves the batch for the rng's current seeds in self._track), then
        # a host wait on its stream: the result is complete when construction returns.
        self._runner.prepare(self._enqueue)

    # -- properties ---------------------------------------------------------------
    @property
    def config(self) -> TrackGenConfig:
        """A copy of the configuration this generator was built with (a
        ``wp.Device`` value is shared, not copied).

        Read-only in effect: mutating the returned object (or the object passed to
        the constructor) does not change this generator. Build a new generator to
        change the configuration.
        """
        return _snapshot_config(self._config)

    @property
    def graph(self) -> "wp.Graph | None":
        """The generator's own CUDA graph for plain ``generate()`` (no ``advance``):
        captured on the first such CUDA call made outside any capture, replayed by
        every later one. ``None`` on CPU, for a non-capturable generator, and before
        that first call. Calls recorded into a caller's capture never create it.
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
        """
        return self._inputs

    @property
    def _graph(self) -> "wp.Graph | None":
        """Read-only alias of :attr:`graph` (backward compatibility)."""
        return self._runner.graph

    # -- enqueue body -------------------------------------------------------------
    def _enqueue(self, ctx: LaunchContext, *, advance: bool = False,
                 mask: "wp.array | None" = None) -> None:
        """Emit one whole generation on ``ctx``: [epoch advance] -> effective seeds
        -> the pipeline -> ``generation <- epoch``.

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
        from . import warp_pipeline
        if advance:
            self.seeds.advance(mask, ctx=ctx)
        self.seeds.effective(self._seed_buf, mask, ctx=ctx)
        warp_pipeline._run_pipeline(
            self._config, self._seed_buf,
            out=self._track, scratch=self._scratch,
            generator_spec=self._generator_spec,
            ctx=ctx,
        )
        if self._track.generation is not None:
            self.seeds._write_generation(self._track.generation, mask, ctx)

    def _body(self, advance: bool, mask) -> "tuple[object, object]":
        """``(body, self-graph key)`` of one call; validates ``advance``/``mask``
        host-side, before any launch."""
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
        - on the Warp ``cpu`` device, or for a ``capturable=False`` generator: eager.

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
        their epoch and get the track of their new ``(seed, epoch)``. Every other env
        keeps the effective seed its current track was built from: the pipeline is
        fixed-shape and still recomputes EVERY env, so the unmasked ones are recomputed
        to identical values, and their ``generation`` is not written. A reseed or
        ``seeds.reset_epoch()`` therefore reaches an env at its next masked generation,
        or every env at the next generation without a mask. A masked commit that skips
        the recompute is a later roadmap step (PR10).

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
            track_gen.exec.NotCapturableError: recording a ``capturable=False``
                generator into a capture (raised before any launch).
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
        stream. Never captures :attr:`graph`, never host-syncs. ``advance`` / ``mask``
        as in :meth:`generate`.

        Returns:
            ``self._track`` (results are stream-ordered).

        Raises:
            ValueError: an invalid ``mask`` (see :meth:`generate`).
            track_gen.exec.NotCapturableError: recording a ``capturable=False``
                generator into a capture (raised before any launch).
        """
        body, _ = self._body(advance, mask)
        self._runner.run(body, stream=stream, policy="eager")
        return self._track
