# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""track_gen — GPU-batched race-track generator.

The public API is the Warp pipeline plus its result types. NVIDIA Warp is
imported lazily (only ``PerEnvSeededRNG`` needs it at import time), so
``import track_gen`` works in a Warp-free environment.
"""

__version__ = "0.1.0"

from .types import Track, TrackGenConfig
from .track_generator import TrackGenerator
from .warp_pipeline import generate_tracks_warp, generate_tracks_warp_graph
import sys as _sys; _sys.modules[__name__].__dict__.pop("warp_pipeline", None); del _sys


def __getattr__(name):
    # rng_utils imports NVIDIA Warp at module load; defer it so `import track_gen`
    # stays Warp-free.
    if name == "PerEnvSeededRNG":
        from .rng_utils import PerEnvSeededRNG
        return PerEnvSeededRNG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrackGenerator",
    "generate_tracks_warp",
    "generate_tracks_warp_graph",
    "TrackGenConfig",
    "Track",
    "PerEnvSeededRNG",
    "__version__",
]
