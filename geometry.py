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
