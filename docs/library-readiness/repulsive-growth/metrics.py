"""Portable geometry diagnostics for the internal repulsive-growth investigation."""
from __future__ import annotations

import hashlib

import numpy as np


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def array_record(array):
    """Hash the unmodified source bytes; dtype/shape disambiguate byte interpretation."""
    array = np.ascontiguousarray(array)
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256_bytes": hashlib.sha256(array.tobytes()).hexdigest()}


def circles_from_rings(points, weights, n_wall=96, n_disc=12):
    """Recover existing discs; weights stay nonzero even after force deactivation."""
    rings = np.asarray(points)[n_wall:].reshape(-1, n_disc, 2)
    masses = np.asarray(weights)[n_wall:].reshape(-1, n_disc)
    exists = np.isfinite(rings).all((1, 2)) & (masses != 0).all(1)
    rings = rings[exists]
    # Retain the original float32 reconstruction before float64 distance calculations.
    centers = rings.mean(1)
    radii = np.linalg.norm(rings - centers[:, None], axis=-1).mean(1)
    return centers.astype(float), radii.astype(float), np.flatnonzero(exists)


def point_segment_distances(points, starts, ends):
    points, starts, ends = (np.asarray(x, dtype=float) for x in (points, starts, ends))
    delta = ends - starts
    numerator = ((points[:, None] - starts[None]) * delta[None]).sum(-1)
    fraction = np.clip(numerator / np.maximum((delta * delta).sum(-1), 1e-30), 0, 1)
    nearest = starts[None] + fraction[..., None] * delta[None]
    return np.linalg.norm(points[:, None] - nearest, axis=-1)


def curve_metrics(xy, centers, radii, domain_radius):
    """All closed-polyline segments, not just vertices, participate in disc checks."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError("diagnostic curve needs at least three finite real points")
    edge = np.roll(xy, -1, axis=0) - xy
    lengths = np.linalg.norm(edge, axis=1)
    previous = np.roll(edge, 1, axis=0)
    turns = np.abs(np.arctan2(cross2(previous, edge), (previous * edge).sum(-1)))
    degrees = np.degrees(turns)
    curvature = turns / np.maximum(.5 * (lengths + np.roll(lengths, 1)), 1e-30)
    result = {
        "perimeter": float(lengths.sum()),
        "area": float(abs(cross2(xy, np.roll(xy, -1, axis=0)).sum()) / 2),
        "turn_max_deg": float(degrees.max()),
        "turn_p95_deg": float(np.quantile(degrees, .95)),
        "turn_rms_deg": float(np.mean(degrees ** 2) ** .5),
        "turn_over30": int((degrees > 30).sum()),
        "kappa_max": float(curvature.max()),
        "kappa_p95": float(np.quantile(curvature, .95)),
        "edge_cv": float(lengths.std() / lengths.mean()),
        "edge_min": float(lengths.min()), "edge_max": float(lengths.max()),
        "wall_center_gap": float(domain_radius - np.linalg.norm(xy, axis=-1).max()),
    }
    if len(centers):
        nodes = np.linalg.norm(xy[None] - centers[:, None], axis=-1) - radii[:, None]
        segments = point_segment_distances(centers, xy, np.roll(xy, -1, axis=0)) - radii[:, None]
        result.update(disc_node_gap=float(nodes.min()), disc_seg_gap=float(segments.min()),
                      disc_hits=int((segments.min(1) < 0).sum()),
                      node_hits=int((nodes.min(1) < 0).sum()))
    else:
        result.update(disc_node_gap=None, disc_seg_gap=None, disc_hits=0, node_hits=0)
    return result


def road_disc_gaps(inner, outer, centers, radii):
    """Test the rendered quads as two triangles; negative values indicate overlap.

    If a disc center is inside a quad, distance is zero and the reported value is
    -radius. This is an intersection indicator, not signed penetration depth.
    """
    inner, outer = np.asarray(inner, dtype=float), np.asarray(outer, dtype=float)
    quads = np.stack((inner, outer, np.roll(outer, -1, axis=0),
                      np.roll(inner, -1, axis=0)), axis=1)
    distance = np.full(len(centers), np.inf)
    for i in range(4):
        distance = np.minimum(distance, point_segment_distances(
            centers, quads[:, i], quads[:, (i + 1) % 4]).min(1))
    for indices in ((0, 1, 2), (0, 2, 3)):
        triangle = quads[:, indices, :]
        signs = np.stack([cross2(triangle[:, (i + 1) % 3] - triangle[:, i],
                                  centers[:, None] - triangle[None, :, i])
                          for i in range(3)], axis=-1)
        nondegenerate = abs(cross2(triangle[:, 1] - triangle[:, 0],
                                  triangle[:, 2] - triangle[:, 0])) > 1e-15
        inside = ((signs >= -1e-12).all(-1) | (signs <= 1e-12).all(-1))
        distance[(inside & nondegenerate[None]).any(1)] = 0
    return (distance - radii).tolist()


def tp_wall_energy(xy, wall_points, domain_radius, alpha=3., beta=6., eps=1e-4):
    """Float64 evaluation of source TP+wall energy after inner-force deactivation."""
    xy, wall_points = np.asarray(xy, dtype=float), np.asarray(wall_points, dtype=float)
    forward = np.roll(xy, -1, axis=0) - xy
    backward = xy - np.roll(xy, 1, axis=0)
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-8)
    dual = .5 * (np.linalg.norm(forward, axis=1) + np.linalg.norm(backward, axis=1))
    difference = xy[None] - xy[:, None]
    distance2 = (difference * difference).sum(-1)
    wedge = cross2(difference, tangent[:, None])
    indices = np.arange(len(xy))
    offset = abs(indices[:, None] - indices[None])
    nonlocal_pair = np.minimum(offset, len(xy) - offset) > 2
    tp = (((abs(wedge) + eps) ** alpha / (distance2 + eps * eps) ** (beta / 2)) *
          dual[:, None] * dual[None] * nonlocal_pair).sum()
    obstacle_distance2 = ((xy[:, None] - wall_points[None]) ** 2).sum(-1)
    wall_mass = 2 * np.pi * domain_radius / len(wall_points)
    wall = (wall_mass * (obstacle_distance2 + 1e-8) ** (-(beta - alpha) / 2)).sum()
    return {"tp": float(tp), "wall": float(wall), "sum": float(tp + wall)}


def late_motion(arrays, start=177, stop=192):
    """Consecutive unresampled source-array movement in the baseline's late window."""
    cosines, maximum_steps = [], []
    for iteration in range(start, stop + 1):
        current, previous, older = (arrays[f"xy_{i}"]
                                    for i in (iteration, iteration - 1, iteration - 2))
        delta, old_delta = current - previous, previous - older
        denominator = np.linalg.norm(delta) * np.linalg.norm(old_delta)
        if denominator > 1e-15:
            cosines.append(float((delta * old_delta).sum() / denominator))
            maximum_steps.append(float(np.linalg.norm(delta, axis=1).max()))
    return {"start": start, "stop": stop, "movement_pairs": len(cosines),
            "mean_consecutive_cosine": float(np.mean(cosines)),
            "mean_max_vertex_step": float(np.mean(maximum_steps))}
