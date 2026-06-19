# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""track_gen — GPU-batched race-track generator.

The public API is the Warp pipeline plus its result types. The heavy Warp
implementation lives in the private ``track_gen._src`` subpackage; this module
re-exports the supported surface. NVIDIA Warp is imported lazily (only
``PerEnvSeededRNG`` needs it at import time), so ``import track_gen`` works in a
Warp-free environment.
"""

from ._version import __version__
from ._src.types import Track, TrackGenConfig
from ._src.track_generator import TrackGenerator
from ._src.warp_pipeline import generate_tracks_warp, generate_tracks_warp_graph


def __getattr__(name):
    # _src.rng_utils imports NVIDIA Warp at module load; defer it so
    # `import track_gen` stays Warp-free.
    if name == "PerEnvSeededRNG":
        from ._src.rng_utils import PerEnvSeededRNG
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
