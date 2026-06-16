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
