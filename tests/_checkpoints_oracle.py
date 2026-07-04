"""Independent numpy reference for centerline checkpoint sampling."""
from __future__ import annotations

import numpy as np


def sample_checkpoints(center, inner, outer, spacing, max_cp):
    """Mirror of CheckpointSampler semantics on one env's real polylines."""
    center = np.asarray(center, np.float64)
    inner = np.asarray(inner, np.float64)
    outer = np.asarray(outer, np.float64)
    m = len(center)
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    perim = float(cum[-1])
    n = int(np.clip(round(perim / spacing), 3, max_cp))
    step = perim / n
    s = np.arange(n) * step
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, m - 1)
    t = ((s - cum[idx]) / np.maximum(seg[idx], 1e-12))[:, None]
    j = (idx + 1) % m
    d = center[j] - center[idx]
    return {
        "position": center[idx] + d * t,
        "left": inner[idx] + (inner[j] - inner[idx]) * t,
        "right": outer[idx] + (outer[j] - outer[idx]) * t,
        "tangent": d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None],
        "n": n, "step": step,
    }
