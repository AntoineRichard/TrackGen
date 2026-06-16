"""track_gen — GPU-batched race-track generator.

Public API is grown incrementally as modules land. Geometry primitives and the
public dataclasses / generators are re-exported here for convenience once they
exist.
"""

__version__ = "0.1.0"

from .rng_utils import PerEnvSeededRNG  # noqa: F401
from . import geometry  # noqa: F401

__all__ = ["PerEnvSeededRNG", "geometry"]
