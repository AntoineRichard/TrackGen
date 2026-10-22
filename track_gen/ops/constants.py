"""Shared numeric conventions of the built-in generators.

``TARGET_EXTENT`` is the longest bounding-box side (per unit ``config.scale``) that the
``polar``, ``voronoi`` and ``checkpoint`` generators (and their gate generators) centre
and scale their output to. ``bezier`` and ``hull`` land near it by construction of their
sampling grid (a longest side of about 1.0 to 1.8 at scale 1, not centred), and
``repulsive`` is larger (about 2.2). The downstream defaults (``half_width``,
``spacing``, ``N_max``, relaxation bands) are tuned for coordinates of that size, so a
plugin generator that wants those defaults to behave the same should end with
``ops.normalize_extent(ctx, out_centerline, N, ops.constants.TARGET_EXTENT * config.scale)``.
"""
from __future__ import annotations

__all__ = ["TARGET_EXTENT"]

#: Longest bbox side of a built-in centerline at ``config.scale == 1`` (the bezier
#: baseline's typical extent).
TARGET_EXTENT = 1.44
