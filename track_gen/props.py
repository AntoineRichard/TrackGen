"""Public boundary prop-sampling API: instancing poses along track boundaries.

``PropSampler`` resamples the inner or outer boundary at a set spacing into a
:class:`PropSet` of per-prop poses (position, tangent, yaw, length) for
rendering-only instancing — cone lines (``mode="points"``) or wall pieces
(``mode="segments"``). The complement of ``track_gen.collision``: these props
never collide; use ``CollisionChecker`` for out-of-bounds queries.
"""
from ._src.props import PropSampler, PropSet

__all__ = ["PropSampler", "PropSet"]
