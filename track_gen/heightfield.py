"""Public heightfield API: per-env road height grids for external physics solvers.

``HeightFieldBaker`` bakes each env's road surface (from a bound ``Track``
batch) into a square :class:`HeightField` height grid, for consumers that want
a grid rather than a polyline. Normally driven indirectly through
``track_gen.course.Course`` (set ``CourseConfig.heightfield_resolution``,
track mode only) which builds and re-bakes it on every ``generate()`` and
exposes it as ``course.heightfield``.
"""
from ._src.heightfield import HeightField, HeightFieldBaker

__all__ = ["HeightField", "HeightFieldBaker"]
