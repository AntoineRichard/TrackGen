"""Top-level facade for the batched track generator.

Runs the pure-Warp pipeline (``warp_pipeline.generate_tracks_warp``: generation ->
resample -> relax -> inflate, all NVIDIA Warp kernels) and returns a fully-populated
:class:`Track`. The public dataclasses ``TrackGenConfig`` and ``Track`` live in the
dependency-free leaf module ``types.py``; this facade re-exports ``TrackGenerator``
as the package's top-level entry point.

TrackGenerator pre-allocates a single :class:`Track` instance in ``__init__`` and
returns the SAME instance from every ``generate()`` call (stable ``.ptr`` pointers).
Callers that need a snapshot must clone the individual fields.
"""

import torch
from torch import Tensor

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
    ``cuda`` devices with torch only as the array container. Only the ``bezier``
    generator is supported on this path (the Fourier generator was not ported to Warp).

    The output :class:`Track` is pre-allocated once in ``__init__`` (via
    ``_inflate_warp_alloc``) and reused across calls: ``generate()`` always returns
    ``self._track``, writing new results into the same wp.array buffers in place. This
    ensures stable ``.ptr`` pointers so downstream CUDA graph consumers can bake the
    device addresses once.
    """

    def __init__(self, config: TrackGenConfig, rng) -> None:
        """Args:
        config: The pipeline configuration. ``config.generator`` must be ``"bezier"``.
        rng: A ``PerEnvSeededRNG`` instance; its per-env seed values seed the pipeline's
            built-in Warp RNG (one base seed per env). The legacy host-side RNG state
            machine is not used by the Warp pipeline.
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

        # Pre-allocate the persistent output Track buffers and offset scratch (one
        # allocation per generator); both are reused across every generate() call.
        from . import warp_pipeline
        self._track: Track
        self._track, self._scratch = warp_pipeline._inflate_warp_alloc(config)

    def _resolve_ids(self, num_or_ids) -> Tensor:
        """Map an int count to ids ``0..n-1``; pass a tensor of ids through."""
        if isinstance(num_or_ids, int):
            return torch.arange(num_or_ids, device=self._config.device)
        return num_or_ids

    def _seeds_for(self, ids: Tensor) -> Tensor:
        """Per-env base seeds for the Warp RNG: the rng's seed value for each env id.

        Reproduces the legacy per-env seeding (a scalar rng seed -> all envs share it;
        a per-env seed tensor -> distinct seeds) without driving the retired host-side
        RNG state machine: the Warp pipeline reseeds ``wp.rand_init`` per (env, attempt)
        from these values.
        """
        import warp as wp

        seeds_all = wp.to_torch(self._rng.seeds_warp)  # [num_envs] int32 (rng device)
        return seeds_all.to(self._config.device)[ids.long()]

    def generate(self, num_or_ids) -> Track:
        """Generate a batch of tracks via the pure-Warp pipeline.

        Writes results into ``self._track`` in place and returns the SAME instance every
        call (stable ``.ptr`` pointers). Callers that need a snapshot must clone fields.

        Args:
            num_or_ids: Either an ``int`` number of tracks (ids ``0..n-1``) or a
                1D tensor of explicit environment ids.

        Returns:
            ``self._track`` — the same :class:`Track` instance every call (stable pointers).
        """
        from . import warp_pipeline

        ids = self._resolve_ids(num_or_ids)
        seeds = self._seeds_for(ids)
        warp_pipeline.generate_tracks_warp(self._config, seeds, out=self._track,
                                           scratch=self._scratch)
        return self._track
