"""track_gen — GPU-batched race-track generator.

The public API is the Warp pipeline plus its result types. The heavy Warp
implementation lives in the private ``track_gen._src`` subpackage; this module
re-exports the supported surface. NVIDIA Warp (``warp-lang``) is a required core
dependency — this is a Warp-first library, so the surface is imported eagerly.
"""

from ._version import __version__
from ._src.types import Track, TrackGenConfig
from ._src.track_generator import TrackGenerator
from ._src.rng_utils import PerEnvSeededRNG
from ._src.warp_pipeline import generate_tracks_warp

__all__ = [
    "TrackGenerator",
    "generate_tracks_warp",
    "TrackGenConfig",
    "Track",
    "PerEnvSeededRNG",
    "__version__",
]
