"""track_gen — GPU-batched race-track generator.

The public API is the Warp pipeline plus its result types. The heavy Warp
implementation lives in the private ``track_gen._src`` subpackage; this module
re-exports the supported surface. NVIDIA Warp (``warp-lang``) is a required core
dependency — this is a Warp-first library, so the surface is imported eagerly.
"""

from ._version import __version__
from ._src.types import GateGenConfig, GateSequence, Track, TrackGenConfig
from ._src.track_generator import TrackGenerator
from ._src.gate_generator import GateGenerator
from ._src.rng_utils import PerEnvSeededRNG
from . import collision
from . import props
from . import checkpoints
from . import progress
from . import course

__all__ = [
    "TrackGenerator",
    "TrackGenConfig",
    "Track",
    "GateGenerator",
    "GateGenConfig",
    "GateSequence",
    "PerEnvSeededRNG",
    "collision",
    "props",
    "checkpoints",
    "progress",
    "course",
    "__version__",
]
