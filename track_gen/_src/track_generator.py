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

On a CUDA device, the first ``generate()`` call warms up the Warp kernels then captures
the whole pipeline into a native ``wp.Graph`` (via ``wp.ScopedCapture``). Every subsequent
call replays the graph (``wp.capture_launch``) with new seeds written into the pre-allocated
seed buffer in place. On the Warp ``cpu`` device the pipeline runs eagerly every call.
"""

import warp as wp

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
    ``generator_registry.available()``).

    The generator is fixed-batch: it always operates on exactly ``config.num_envs``
    environments. The output :class:`Track` is pre-allocated once in ``__init__`` (via
    ``_inflate_warp_alloc``) and reused across calls: ``generate()`` always returns
    the same ``self._track`` instance, writing new results into the same wp.array buffers
    in place. This ensures stable ``.ptr`` pointers so downstream CUDA graph consumers can
    bake the device addresses once. Use ``Track.clone()`` to obtain an independent snapshot.

    On a CUDA device the pipeline is auto-captured into a ``wp.Graph`` on the first
    ``generate()`` call and replayed on every subsequent call. On the Warp ``cpu``
    device the pipeline runs eagerly.

    Only ``relax_solver="xpbd"`` and ``smooth_finish=False`` are supported; construction
    raises ``AssertionError`` for other combinations.
    """

    def __init__(self, config: TrackGenConfig, rng) -> None:
        """Args:
        config: The pipeline configuration. ``config.generator`` must be a registered
            generator (see ``generator_registry.available()``).
        rng: A ``PerEnvSeededRNG`` instance; its per-env seed values seed the pipeline's
            built-in Warp RNG (one base seed per env).
        """
        if rng is None:
            raise ValueError("A random number generator must be provided.")
        from . import generator_registry
        self._generator_spec = generator_registry.get(config.generator)
        assert config.relax_solver == "xpbd", (
            f"TrackGenerator only supports relax_solver='xpbd'; "
            f"got {config.relax_solver!r}."
        )
        assert not config.smooth_finish, (
            "TrackGenerator does not support smooth_finish=True; "
            "set smooth_finish=False."
        )
        self._config = config
        self._rng = rng

        # Pre-allocate the persistent output Track buffers, scratch, and seed buffer
        # (one allocation per generator); all are reused across every generate() call.
        from . import warp_pipeline
        self._track: Track
        self._track, self._scratch = warp_pipeline._inflate_warp_alloc(
            config, generator_spec=self._generator_spec,
        )

        # Pre-allocate the [E] int32 seed buffer on the pipeline device. Seeds are
        # written in place before each run (wp.copy); no allocation occurs in the hot path.
        E = int(config.num_envs)
        dev = str(config.device)
        self._seed_buf: wp.array = wp.empty(E, dtype=wp.int32, device=dev)

        # _graph is None until the first cuda generate() call, which captures it.
        self._graph: "wp.Graph | None" = None

    def _run(self) -> None:
        """Execute the owned pipeline once into self._track / self._scratch off self._seed_buf.

        Zero-alloc: all buffers are pre-allocated in __init__. Safe to call during
        wp.ScopedCapture (no host syncs, no allocations inside the capture region).
        """
        from . import warp_pipeline
        warp_pipeline._run_pipeline(
            self._config, self._seed_buf,
            out=self._track, scratch=self._scratch,
            generator_spec=self._generator_spec,
        )

    def generate(self, num_or_ids=None) -> Track:
        """Generate a batch of tracks for the fixed configured batch.

        This generator is fixed-batch: it always operates on exactly ``config.num_envs``
        environments. If integer ``num_or_ids`` is provided it is validated against the
        configured batch size and a ``ValueError`` is raised if they disagree (so existing
        call sites passing ``E == num_envs`` continue to work unchanged). Explicit
        environment-id sequences are rejected with ``TypeError`` because selected-env
        execution is not part of the fixed-batch graph-captured contract.

        On the first CUDA call: warms up kernel loading then captures the pipeline into a
        ``wp.Graph`` (via ``wp.ScopedCapture``), then immediately replays it. On subsequent
        CUDA calls: writes new seeds into the seed buffer in place and replays the graph.
        On ``cpu``: runs ``_run()`` eagerly every call.

        Writes results into ``self._track`` in place and returns the SAME instance every
        call (stable ``.ptr`` pointers). Use ``Track.clone()`` to obtain an independent copy.

        Determinism: ``generate()`` re-copies the rng's CURRENT seeds each call, so repeated
        calls with an unchanged rng return the IDENTICAL batch. To vary the batch between
        calls, reseed the rng first (e.g. ``rng.set_seeds_warp(new_seeds, None)``).

        Args:
            num_or_ids: Optional. Either ``None`` or the integer batch size. When an
                integer is provided, it must equal ``config.num_envs``. Explicit
                environment-id sequences are not supported by this fixed-batch,
                graph-capturable facade.

        Returns:
            ``self._track`` — the same :class:`Track` instance every call (stable pointers).

        Raises:
            TypeError: if ``num_or_ids`` is an explicit environment-id sequence.
            ValueError: if an integer ``num_or_ids`` differs from ``config.num_envs``.
        """
        if num_or_ids is not None:
            if not isinstance(num_or_ids, int):
                raise TypeError(
                    "TrackGenerator.generate() does not accept explicit environment ids; "
                    "construct a generator with the desired fixed num_envs instead."
                )
            if num_or_ids != self._config.num_envs:
                raise ValueError(
                    f"TrackGenerator is fixed-batch for {self._config.num_envs} envs; "
                    f"got num_or_ids={num_or_ids}. "
                    f"Construct a new TrackGenerator with num_envs={num_or_ids} instead."
                )
        from . import warp_pipeline

        # Refresh the seed buffer in place from the rng's CURRENT seeds (zero allocation:
        # wp.copy). The rng holds fixed seeds unless reseeded, so back-to-back generate()
        # calls are deterministic; reseed the rng to vary the batch. seeds_warp is [E] int32.
        wp.copy(self._seed_buf, self._rng.seeds_warp)

        dev = str(self._config.device)
        _is_cuda = "cuda" in dev

        if _is_cuda:
            if self._graph is None:
                # First CUDA call: warm up (loads kernels/modules), then capture.
                # _CAPTURING suppresses host-blocking syncs during warmup + capture.
                warp_pipeline._CAPTURING = True
                try:
                    for _ in range(3):
                        self._run()
                    wp.synchronize()  # OUTSIDE capture: ensure warmup finished

                    with wp.ScopedCapture(device=dev) as cap:
                        self._run()
                    self._graph = cap.graph
                finally:
                    warp_pipeline._CAPTURING = False

            # Replay the captured graph (seed buffer already updated above).
            wp.capture_launch(self._graph)
            wp.synchronize()
        else:
            # CPU eager path — no graph capture.
            self._run()

        return self._track
