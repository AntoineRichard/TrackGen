# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level facade for the batched track generator.

Runs the pure-Warp pipeline (``warp_pipeline.generate_tracks_warp``: generation ->
resample -> relax -> inflate, all NVIDIA Warp kernels) and returns a fully-populated
:class:`Track`. The public dataclasses ``TrackGenConfig`` and ``Track`` live in the
dependency-free leaf module ``types.py`` and are re-exported here for backward
compatibility.
"""

import warnings

import torch
from torch import Tensor

from .types import Track, TrackGenConfig

__all__ = [
    "Track",
    "TrackGenConfig",
    "TrackGenerator",
    "generate_tracks",
]


class TrackGenerator:
    """Top-level facade: run the pure-Warp track-generation pipeline and return a
    :class:`Track`.

    Generation, resample, relaxation and inflation are all expressed as NVIDIA Warp
    kernels (``warp_pipeline.generate_tracks_warp``), runnable on the Warp ``cpu`` and
    ``cuda`` devices with torch only as the array container. Only the ``bezier``
    generator is supported on this path (the Fourier generator was not ported to Warp).
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

        Args:
            num_or_ids: Either an ``int`` number of tracks (ids ``0..n-1``) or a
                1D tensor of explicit environment ids.

        Returns:
            A fully-populated :class:`Track`.
        """
        from . import warp_pipeline

        ids = self._resolve_ids(num_or_ids)
        seeds = self._seeds_for(ids)
        return warp_pipeline.generate_tracks_warp(self._config, seeds)


def generate_tracks(num_tracks: int, config: TrackGenConfig | None = None, rng=None) -> Tensor:
    """Deprecated backward-compatibility shim for the old centerline-only API.

    The legacy ``TrackGenerator.generate_tracks`` returned only centerline data.
    This shim runs the full pipeline and returns just the centerline points,
    shaped ``[num_tracks, N, 2]``, so existing callers keep working.

    .. deprecated::
        Use ``TrackGenerator(config, rng).generate(num_tracks).center`` instead.

    Args:
        num_tracks: Number of tracks to generate.
        config: Pipeline configuration. If ``None``, a default
            :class:`TrackGenConfig` with ``num_envs = num_tracks`` is used.
        rng: A ``PerEnvSeededRNG`` instance.

    Returns:
        The centerline points, shape ``[num_tracks, N, 2]``.
    """
    warnings.warn(
        "generate_tracks() is deprecated; use "
        "TrackGenerator(config, rng).generate(num_tracks).center instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if config is None:
        config = TrackGenConfig(num_envs=num_tracks)
    generator = TrackGenerator(config, rng)
    track = generator.generate(num_tracks)
    return track.center
