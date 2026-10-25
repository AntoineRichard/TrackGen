"""Independent float64 SciPy hull reference for actual finished road cells.

This deliberately shares no production clearance predicates. The union of
four-vertex convex cell hulls conservatively covers both ruled/triangulated
road surfaces, including folded cells; the whole loop hole stays unoccupied.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def minimum_band_gaps(outer, inner, centers, radii, domain_radius):
    outer, inner, centers, radii = [np.asarray(x, dtype=np.float64)
                                     for x in (outer, inner, centers, radii)]
    if not (np.isfinite(outer).all() and np.isfinite(inner).all()):
        return -np.inf, -np.inf
    if not np.isfinite(radii).all() or np.any(radii < 0.):
        return -np.inf, -np.inf
    used = radii > 0
    centers, radii = centers[used], radii[used]
    if not (np.isfinite(centers).all() and np.isfinite(radii).all()
            and np.isfinite(domain_radius)):
        return -np.inf, -np.inf
    wall_gap = domain_radius - max(np.linalg.norm(outer, axis=1).max(),
                                   np.linalg.norm(inner, axis=1).max())
    gap = np.full(len(radii), np.inf)
    for i in range(len(outer)):
        j = (i + 1) % len(outer)
        points = np.array([outer[i], outer[j], inner[j], inner[i]])
        try:
            hull = ConvexHull(points)
            vertices = points[hull.vertices]
            inside = (centers @ hull.equations[:, :2].T + hull.equations[:, 2]
                      <= 0.).all(axis=1)
        except QhullError:
            # Degenerate cells reduce to their segment hull (or a point).
            vectors = points - points[0]
            cross = (vectors[:, None, 0] * vectors[None, :, 1]
                     - vectors[:, None, 1] * vectors[None, :, 0])
            if np.any(cross != 0.):
                raise  # a failed two-dimensional hull is not a clearance proof
            vertices = np.unique(points, axis=0)
            inside = np.zeros(len(centers), dtype=bool)
        delta = np.roll(vertices, -1, axis=0) - vertices
        delta2 = np.sum(delta * delta, axis=1)
        t = np.clip(np.sum((centers[:, None, :] - vertices) * delta, axis=2)
                    / np.maximum(delta2, np.finfo(float).tiny), 0., 1.)
        distances = np.linalg.norm(vertices - centers[:, None, :] + t[:, :, None] * delta,
                                   axis=2).min(axis=1)
        distances[inside] = 0.
        gap = np.minimum(gap, distances - radii)
    return gap.min(initial=np.inf), wall_gap
