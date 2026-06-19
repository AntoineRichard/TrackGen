"""track_gen — GPU-batched race-track generator.

The public API is the Warp pipeline plus its result types. The heavy Warp
implementation lives in the private ``track_gen._src`` subpackage; this module
re-exports the supported surface. NVIDIA Warp is imported lazily: ``import
track_gen`` is Warp-free, and Warp loads only when a Warp entry point
(``generate_tracks_warp``, ``generate_tracks_warp_graph``, ``PerEnvSeededRNG``,
or ``TrackGenerator.generate``) is actually used.
"""

from ._version import __version__
from ._src.types import Track, TrackGenConfig
from ._src.track_generator import TrackGenerator


def __getattr__(name):
    # These names pull in NVIDIA Warp at import time, so resolve them lazily to
    # keep ``import track_gen`` Warp-free (Warp is an optional extra).
    if name == "PerEnvSeededRNG":
        from ._src.rng_utils import PerEnvSeededRNG
        return PerEnvSeededRNG
    if name in ("generate_tracks_warp", "generate_tracks_warp_graph"):
        from ._src import warp_pipeline
        return getattr(warp_pipeline, name)
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
