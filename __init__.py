"""track_gen — GPU-batched race-track generator.

Public API is grown incrementally as modules land. Geometry primitives are
re-exported here for convenience.
"""

__version__ = "0.1.0"

from .rng_utils import PerEnvSeededRNG
from . import geometry
from .geometry import (
    arc_length_resample,
    ccw_sort,
    menger_curvature,
    nearest_nonadjacent_distance,
    polygon_area,
    safe_normalize,
    segment_directions,
    tangents_normals,
    turning_number,
    vertex_tangents,
)

__all__ = [
    "PerEnvSeededRNG",
    "geometry",
    "safe_normalize",
    "polygon_area",
    "ccw_sort",
    "segment_directions",
    "vertex_tangents",
    "turning_number",
    "menger_curvature",
    "tangents_normals",
    "arc_length_resample",
    "nearest_nonadjacent_distance",
]
