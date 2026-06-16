"""Pure batched-torch geometry primitives.

Device-agnostic and dependency-light: torch only (NO warp import), so the whole
module is unit-testable on CPU. Batch dimension is E (num_envs); shapes are
documented per function in [brackets].
"""

import torch  # noqa: F401


def safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors along the last axis; zero vectors stay finite (zero).

    Args:
        v: Tensor [..., D]. Vectors live along the final dimension.
        eps: Floor for the norm so a zero/near-zero vector yields zero, not NaN.

    Returns:
        Tensor of the same shape as ``v`` with unit-length vectors; the zero
        vector maps to the zero vector.
    """
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / norm.clamp_min(eps)


def polygon_area(points: torch.Tensor) -> torch.Tensor:
    """Signed shoelace area of each closed polygon in the batch.

    Args:
        points: Tensor [E, P, 2]. Each env's P vertices in order; the polygon is
            implicitly closed (last vertex connects to first).

    Returns:
        Tensor [E]. Positive for counter-clockwise vertex order, negative for
        clockwise.
    """
    x = points[..., 0]
    y = points[..., 1]
    x_next = torch.roll(x, shifts=-1, dims=1)
    y_next = torch.roll(y, shifts=-1, dims=1)
    cross = x * y_next - x_next * y
    return 0.5 * cross.sum(dim=1)


def segment_directions(points: torch.Tensor, closed: bool = True) -> torch.Tensor:
    """Unit direction of each edge i -> i+1.

    Args:
        points: Tensor [E, P, 2].
        closed: If True, the last edge wraps from the final vertex back to the
            first. If False, that final wrap slot is set to zero.

    Returns:
        Tensor [E, P, 2] of unit edge directions; zero vectors (degenerate or
        the open-chain wrap slot) stay finite (zero).
    """
    points_next = torch.roll(points, shifts=-1, dims=1)
    deltas = points_next - points
    dirs = safe_normalize(deltas)
    if not closed:
        dirs = dirs.clone()
        dirs[:, -1, :] = 0.0
    return dirs


def vertex_tangents(points: torch.Tensor, p: float) -> torch.Tensor:
    """Blended unit tangent at each vertex from its two incident edge dirs.

    Vector-space tangent blend (replaces the old atan2 angle blend). At vertex i,
    u_out is the direction of edge i -> i+1 and u_in is the direction of edge
    i-1 -> i; the tangent is safe_normalize(p * u_out + (1 - p) * u_in).

    Args:
        points: Tensor [E, P, 2], closed loop.
        p: Blend weight in [0, 1]. p=1 -> pure out-edge, p=0 -> pure in-edge,
            p=0.5 -> bisector.

    Returns:
        Tensor [E, P, 2] of unit tangents.
    """
    u_out = segment_directions(points, closed=True)
    u_in = torch.roll(u_out, shifts=1, dims=1)
    blended = p * u_out + (1.0 - p) * u_in
    return safe_normalize(blended)


def ccw_sort(points: torch.Tensor) -> torch.Tensor:
    """Order each env's points angularly around their centroid.

    Ported from the original ``TrackGenerator.ccw_sort`` to preserve behavior,
    including its ``atan2(dx, dy)`` argument order. Reordering points by angle
    around the centroid yields a simple (non-self-intersecting) polygon.

    Args:
        points: Tensor [E, P, 2].

    Returns:
        Tensor [E, P, 2], the same points reordered along the P axis.
    """
    mean = torch.mean(points, dim=1)
    dist = points - mean.unsqueeze(1)
    angles = torch.arctan2(dist[:, :, 0], dist[:, :, 1])
    ids = torch.argsort(angles, dim=1)
    points = torch.gather(points, 1, ids.unsqueeze(-1).expand(-1, -1, points.size(2)))
    return points
