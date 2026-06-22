"""Render a 5x5 grid for the K=2+clip config (DEV ONLY) -> viz/out/checkpoint_k2clip_grid.png.

Eyeball check: do clipped tracks look like real tracks (clip preserves shape) or are there
artifacts? Compare against viz/out/checkpoint_proto_grid_k8.png. Red = still self-crossing.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from track_gen._experimental import checkpoint_proto as cp  # noqa: E402
from benchmarks.track_metrics import compactness, self_intersects, straight_fraction  # noqa: E402

cfg = cp.DEFAULTS.copy()
os.makedirs(os.path.join(_ROOT, "viz", "out"), exist_ok=True)

fig, axes = plt.subplots(5, 5, figsize=(15, 15))
for i, ax in enumerate(axes.flat):
    p, _ = cp.generate_centerline_clip(i, cfg, K=2)
    cpoly = np.vstack([p, p[0]])
    crossing = self_intersects(p)
    ax.plot(cpoly[:, 0], cpoly[:, 1], "r-" if crossing else "b-", lw=0.9)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"s={i} c={compactness(p):.2f} st={straight_fraction(p):.2f}", fontsize=7)
plt.suptitle("Checkpoint Proto #5 — K=2 best-of-K + single-crossing CLIP "
             f"(closure={cfg['closure']})", fontsize=12)
plt.tight_layout()
out = os.path.join(_ROOT, "viz", "out", "checkpoint_k2clip_grid.png")
plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
print(f"saved {out}")
