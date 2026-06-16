"""Pure batched-torch geometry primitives.

Device-agnostic and dependency-light: torch only (NO warp import), so the whole
module is unit-testable on CPU. Batch dimension is E (num_envs); shapes are
documented per function in [brackets].
"""

import torch  # noqa: F401
