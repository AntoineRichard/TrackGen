"""Shared numeric conventions of the built-in generators.

``TARGET_EXTENT`` is the longest bounding-box side (per unit ``config.scale``) that the
``polar``, ``voronoi`` and ``checkpoint`` generators (and their gate generators) centre
and scale their output to. ``bezier`` and ``hull`` land near it by construction of their
sampling grid (a longest side of about 1.0 to 1.8 at scale 1, not centred), and
``repulsive`` is larger (about 2.2). The downstream defaults (``half_width``,
``spacing``, ``N_max``, relaxation bands) are tuned for coordinates of that size, so a
plugin generator that wants those defaults to behave the same should end with
``ops.normalize_extent(ctx, out_centerline, N, ops.constants.TARGET_EXTENT * config.scale)``.

``SIZE_BOUNDS`` is the declared range of the per-env ``size`` multiplier every built-in
generator except ``external`` (and ``normalize.extent``) declares, and the range
``ops.sample_grid_points(size=)`` / ``ops.normalize_extent(size=)`` clamp each env's
value to (by comparisons: a NaN passes). It is the convention value a plugin generator
imports for its own ``size`` field::

    import dataclasses

    from track_gen import plugin
    from track_gen.ops.constants import SIZE_BOUNDS

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class MyParams(plugin.Params):
        size: float = plugin.dynamic(1.0, lo=SIZE_BOUNDS[0], hi=SIZE_BOUNDS[1])

A legality range: the useful range is set by ``n_max`` (truncation), ``num_points``, the
absolute half-width and the utilities' capacities (the sizing rule in the tuning guide).
"""
from __future__ import annotations

__all__ = ["SIZE_BOUNDS", "TARGET_EXTENT"]

#: Longest bbox side of a built-in centerline at ``config.scale == 1`` (the bezier
#: baseline's typical extent).
TARGET_EXTENT = 1.44

#: ``(lo, hi)`` of the per-env ``size`` multiplier: the declared bounds
#: of every built-in ``size`` field and the clamp of the ``size=`` arrays of
#: ``ops.sample_grid_points`` / ``ops.normalize_extent``.
SIZE_BOUNDS = (0.1, 10.0)

#: The declared lower bound of the dynamic corner count ``min_num_points`` (private).
_MIN_COUNT_LO = 2
