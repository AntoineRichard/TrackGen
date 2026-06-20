"""Top-level facade for the batched track generator.

Runs the pure-Warp pipeline (``warp_pipeline.generate_tracks_warp``: generation ->
resample -> relax -> inflate, all NVIDIA Warp kernels) and returns a fully-populated
:class:`Track`. The public dataclasses ``TrackGenConfig`` and ``Track`` live in the
dependency-free leaf module ``types.py``; this facade re-exports ``TrackGenerator``
as the package's top-level entry point.

TrackGenerator pre-allocates a single :class:`Track` instance in ``__init__`` and
returns the SAME instance from every ``generate()`` call (stable ``.ptr`` pointers).
Callers that need a snapshot must clone the individual fields.

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
    kernels (``warp_pipeline.generate_tracks_warp``), runnable on the Warp ``cpu`` and
    ``cuda`` devices. Only the ``bezier`` generator is supported on this path (the Fourier
    generator was not ported to Warp).

    The output :class:`Track` is pre-allocated once in ``__init__`` (via
    ``_inflate_warp_alloc``) and reused across calls: ``generate()`` always returns
    ``self._track``, writing new results into the same wp.array buffers in place. This
    ensures stable ``.ptr`` pointers so downstream CUDA graph consumers can bake the
    device addresses once.

    On a CUDA device the pipeline is auto-captured into a ``wp.Graph`` on the first
    ``generate()`` call and replayed on every subsequent call. On the Warp ``cpu``
    device the pipeline runs eagerly.
    """

    def __init__(self, config: TrackGenConfig, rng) -> None:
        """Args:
        config: The pipeline configuration. ``config.generator`` must be ``"bezier"``.
        rng: A ``PerEnvSeededRNG`` instance; its per-env seed values seed the pipeline's
            built-in Warp RNG (one base seed per env).
        """
        if rng is None:
            raise ValueError("A random number generator must be provided.")
        if config.generator != "bezier":
            raise ValueError(
                f"The pure-Warp pipeline supports generator='bezier' only; "
                f"got {config.generator!r}."
            )
        self._config = config
        self._rng = rng

        # Pre-allocate the persistent output Track buffers, scratch, and seed buffer
        # (one allocation per generator); all are reused across every generate() call.
        from . import warp_pipeline
        self._track: Track
        self._track, self._scratch = warp_pipeline._inflate_warp_alloc(config)

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
        )

    def generate(self, num_or_ids) -> Track:
        """Generate a batch of tracks via the pure-Warp pipeline.

        On the first CUDA call: warms up kernel loading then captures the pipeline into a
        ``wp.Graph`` (via ``wp.ScopedCapture``), then immediately replays it. On subsequent
        CUDA calls: writes new seeds into the seed buffer in place and replays the graph.
        On ``cpu``: runs ``_run()`` eagerly every call.

        Writes results into ``self._track`` in place and returns the SAME instance every
        call (stable ``.ptr`` pointers). Callers that need a snapshot must clone fields.

        Args:
            num_or_ids: Either an ``int`` number of tracks (ids ``0..n-1``) or a
                1D tensor of explicit environment ids.

        Returns:
            ``self._track`` — the same :class:`Track` instance every call (stable pointers).
        """
        from . import warp_pipeline

        # Refresh the seed buffer in place from the rng (zero allocation: wp.copy).
        # rng.seeds_warp is a wp.array [num_envs] int32. We write all envs; num_or_ids
        # determines which env ids are active but the seed buffer covers all envs by design
        # (the pipeline uses seeds_warp[e] for env e; extra envs just produce irrelevant output).
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
