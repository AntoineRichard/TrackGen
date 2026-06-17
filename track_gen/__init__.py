"""track_gen — GPU-batched race-track generator."""

__version__ = "0.1.0"

from .rng_utils import PerEnvSeededRNG
from . import geometry
from . import relaxation
from .relaxation import relax
from .geometry import (
    arc_length_resample,
    ccw_sort,
    curvature_radius_min,
    menger_curvature,
    nearest_nonadjacent_distance,
    polygon_area,
    safe_normalize,
    segment_directions,
    self_intersections,
    separation_min,
    tangents_normals,
    thickness,
    turning_number,
    vertex_tangents,
)
from .types import Track, TrackGenConfig
from .generators import (
    BezierCenterlineGenerator,
    Centerline,
    CenterlineGenerator,
    FourierCenterlineGenerator,
)
from .track_generator import TrackGenerator, generate_tracks

__all__ = [
    "PerEnvSeededRNG",
    "geometry",
    "relaxation",
    "relax",
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
    "thickness",
    "self_intersections",
    "separation_min",
    "curvature_radius_min",
    "Track",
    "TrackGenConfig",
    "Centerline",
    "CenterlineGenerator",
    "BezierCenterlineGenerator",
    "FourierCenterlineGenerator",
    "TrackGenerator",
    "generate_tracks",
]
