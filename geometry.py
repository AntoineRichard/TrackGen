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


def turning_number(points: torch.Tensor) -> torch.Tensor:
    """Signed total turning of a closed polygon, in radians.

    +/-2*pi for a simple loop (sign = orientation); ~0 for a figure-eight whose
    lobes wind in opposite directions. Used as a cheap O(P) self-intersection
    gate.

    Args:
        points: Tensor [E, P, 2], closed loop.

    Returns:
        Tensor [E].
    """
    dirs = segment_directions(points, closed=True)
    theta = torch.atan2(dirs[..., 1], dirs[..., 0])
    dtheta = theta - torch.roll(theta, shifts=1, dims=1)
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))  # wrap into (-pi, pi]
    return dtheta.sum(dim=1)


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


def menger_curvature(points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Non-negative Menger curvature at each point on a closed loop.

    For the triple (i-1, i, i+1): kappa = 4 * |triangle area| / (|a||b||c|),
    where a, b, c are the triangle's side lengths. Tends to 1/r on a radius-r
    circle; ~0 on a straight line. The denominator is clamped by eps so
    coincident points yield 0 rather than NaN.

    Args:
        points: Tensor [E, N, 2], closed loop.
        eps: Denominator floor guarding divide-by-zero.

    Returns:
        Tensor [E, N], kappa >= 0.
    """
    p_prev = torch.roll(points, shifts=1, dims=1)
    p_curr = points
    p_next = torch.roll(points, shifts=-1, dims=1)

    a = p_curr - p_prev
    b = p_next - p_curr
    c = p_next - p_prev

    len_a = torch.linalg.norm(a, dim=-1)
    len_b = torch.linalg.norm(b, dim=-1)
    len_c = torch.linalg.norm(c, dim=-1)

    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]  # 2D cross product
    area = 0.5 * cross.abs()

    denom = (len_a * len_b * len_c).clamp_min(eps)
    return 4.0 * area / denom


def tangents_normals(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit central-difference tangents and their left-normals on a closed loop.

    Tangent at point i uses the central difference points[i+1] - points[i-1]
    (wrapping on the closed loop), then safe_normalize. The left-normal is the
    90-degree CCW rotation: Nrm = stack(-T_y, T_x). Orthonormal by construction.

    Args:
        points: Tensor [E, N, 2], closed loop.

    Returns:
        (T, Nrm), each Tensor [E, N, 2]. ||T|| = 1 and T . Nrm = 0 everywhere.
    """
    p_next = torch.roll(points, shifts=-1, dims=1)
    p_prev = torch.roll(points, shifts=1, dims=1)
    T = safe_normalize(p_next - p_prev)
    Nrm = torch.stack([-T[..., 1], T[..., 0]], dim=-1)
    return T, Nrm


def _resample_one(real: torch.Tensor, num: int | None, spacing: float | None) -> torch.Tensor:
    """Resample a single closed loop of real points at arc-length targets.

    Args:
        real: Tensor [R, 2] of the real (valid, non-NaN) loop points.
        num: If given, produce exactly ``num`` arc-uniform points.
        spacing: If given, produce points every ``spacing`` arc length.

    Returns:
        Tensor [K, 2] of resampled points. If R < 2, returns a NaN-filled row of
        the target width (K = num in fixed mode, 0 in spacing mode) so an
        unconverged / all-NaN env never indexes into an empty tensor.
    """
    if real.shape[0] < 2:
        # Degenerate env: emit NaN of the target width (fixed) or empty (spacing).
        k = num if num is not None else 0
        return torch.full((k, 2), float("nan"), dtype=real.dtype, device=real.device)

    # Close the loop: append the first point so the wrap segment is included.
    closed = torch.cat([real, real[:1]], dim=0)  # [R+1, 2]
    seg = closed[1:] - closed[:-1]  # [R, 2]
    seg_len = torch.linalg.norm(seg, dim=-1)  # [R]
    s = torch.cat(
        [torch.zeros(1, dtype=real.dtype, device=real.device), torch.cumsum(seg_len, dim=0)]
    )  # [R+1]
    total = s[-1]

    if num is not None:
        targets = torch.arange(num, dtype=real.dtype, device=real.device) * (total / num)
    else:
        k = int(torch.floor(total / spacing).item()) + 1
        targets = torch.arange(k, dtype=real.dtype, device=real.device) * spacing
        targets = targets[targets < total]

    idx = torch.searchsorted(s[1:], targets, right=False)  # [K] in [0, R-1]
    idx = idx.clamp(max=seg_len.shape[0] - 1)
    s0 = s[idx]
    seg_l = seg_len[idx].clamp_min(1e-12)
    frac = ((targets - s0) / seg_l).clamp(0.0, 1.0).unsqueeze(-1)  # [K, 1]
    p0 = closed[idx]
    p1 = closed[idx + 1]
    return p0 + frac * (p1 - p0)


def arc_length_resample(
    points: torch.Tensor,
    num: int | None = None,
    spacing: float | None = None,
    valid_mask: torch.Tensor | None = None,
    n_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arc-length-uniform resampling of a batch of closed loops.

    Invalid points (valid_mask False, when given) and any point with a NaN
    coordinate are dropped before measuring arc length; the loop is closed by
    appending the wrap segment back to the first real point.

    Exactly one of ``num`` / ``spacing`` must be given:
      - num: N = num and count = num for every env (fixed mode). Envs with < 2
        real points yield a NaN row and count 0.
      - spacing: constant arc-length spacing; real count varies per env. Output
        is padded to ``n_max`` (when given; falls back to the batch-max count
        otherwise) with NaN, and the real count is returned per env. The caller
        (inflation) passes ``n_max=config.N_max``.

    Args:
        points: Tensor [E, M, 2].
        num: Fixed output point count.
        spacing: Constant arc-length spacing.
        valid_mask: Optional Tensor [E, M] bool; False marks padding/invalid.
        n_max: Padded output width for spacing mode.

    Returns:
        (resampled [E, N, 2], count [E] int).
    """
    if (num is None) == (spacing is None):
        raise ValueError("Provide exactly one of `num` or `spacing`.")

    E, M, _ = points.shape
    device = points.device

    per_env = []
    counts = []
    for e in range(E):
        pe = points[e]  # [M, 2]
        keep = torch.isfinite(pe).all(dim=-1)
        if valid_mask is not None:
            keep = keep & valid_mask[e].bool()
        real = pe[keep]
        out_e = _resample_one(real, num, spacing)
        per_env.append(out_e)
        # R < 2 guard: degenerate env gets count 0 regardless of output tensor size.
        if real.shape[0] < 2:
            counts.append(0)
        else:
            counts.append(out_e.shape[0])

    if num is not None:
        width = num
    elif n_max is not None:
        assert max(counts) <= n_max, f"spacing produced {max(counts)} > n_max={n_max}"
        width = n_max
    else:
        width = max(counts) if counts else 0

    resampled = torch.full((E, width, 2), float("nan"), dtype=points.dtype, device=device)
    for e in range(E):
        k = min(counts[e], width)
        if k > 0:
            resampled[e, :k] = per_env[e][:k]

    count = torch.tensor(counts, dtype=torch.long, device=device)
    return resampled, count
