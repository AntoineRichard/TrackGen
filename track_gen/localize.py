"""Public track-frame localization API: (s, n) projection + speed hints.

``TrackLocalizer`` projects one point per env onto the bound ``Track``'s
centerline and returns a :class:`TrackFrame`: arc length ``s``, signed
lateral offset ``n`` (positive to the RIGHT of the direction of travel;
which boundary that is depends on the loop's generator-dependent winding),
and the nearest segment index — the Frenet-style frame
racing controllers and reward shapers consume every sim step. An optional
warm start (``warm_window=``) narrows each scan to a window around the
previous result; ``reset(mask)`` drops that memory after regenerations and
teleports.

``curvature()`` and ``speed_profile()`` are per-generation companions:
per-point signed centerline curvature and the curvature-limited target
speed (steady-state lateral limit, then forward-acceleration and
backward-braking passes over the closed loop).
"""
from ._src.localize import TrackFrame, TrackLocalizer, curvature, speed_profile

__all__ = ["TrackFrame", "TrackLocalizer", "curvature", "speed_profile"]
