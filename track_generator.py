# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level facade for the batched track generator.

Wires the configured centerline generator (Bezier or Fourier) to the inflation
stage and returns a fully-populated :class:`Track`. The public dataclasses
``TrackGenConfig`` and ``Track`` live in the dependency-free leaf module
``types.py`` and are re-exported here for backward compatibility.
"""

import warnings

import torch
from torch import Tensor

from . import PerEnvSeededRNG  # noqa: F401  (re-export; matches legacy import surface)
from .types import Track, TrackGenConfig
from .generators import (
    BezierCenterlineGenerator,
    Centerline,
    FourierCenterlineGenerator,
)
from .inflation import inflate

__all__ = [
    "Track",
    "TrackGenConfig",
    "TrackGenerator",
    "generate_tracks",
]


class TrackGenerator:
    """Top-level facade: build the configured centerline generator, run it,
    inflate the result, and return a :class:`Track`.
    """

    _GENERATORS = {
        "bezier": BezierCenterlineGenerator,
        "fourier": FourierCenterlineGenerator,
    }

    def __init__(self, config: TrackGenConfig, rng) -> None:
        """Args:
        config: The pipeline configuration.
        rng: A ``PerEnvSeededRNG`` instance for per-env reproducible sampling.
        """
        if rng is None:
            raise ValueError("A random number generator must be provided.")
        self._config = config
        self._rng = rng

        generator_cls = self._GENERATORS.get(config.generator)
        if generator_cls is None:
            raise ValueError(
                f"Unknown generator '{config.generator}'. "
                f"Expected one of {sorted(self._GENERATORS)}."
            )
        self._generator = generator_cls(config, rng)

    def _resolve_ids(self, num_or_ids) -> Tensor:
        """Map an int count to ids ``0..n-1``; pass a tensor of ids through."""
        if isinstance(num_or_ids, int):
            return torch.arange(num_or_ids, device=self._config.device)
        return num_or_ids

    def generate(self, num_or_ids) -> Track:
        """Generate a batch of tracks.

        Args:
            num_or_ids: Either an ``int`` number of tracks (ids ``0..n-1``) or a
                1D tensor of explicit environment ids.

        Returns:
            A fully-populated :class:`Track`.
        """
        ids = self._resolve_ids(num_or_ids)
        centerline: Centerline = self._generator.generate(ids)
        return inflate(centerline, self._config)
