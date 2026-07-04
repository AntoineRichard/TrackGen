"""Independent numpy reference for boundary prop sampling (tests only)."""
from __future__ import annotations

import numpy as np


def sample_boundary(poly, spacing, max_props):
    """Points-mode reference: closed-polyline arc resample with snap rule.

    Returns (positions [n,2], tangents [n,2], n, step, truncated).
    """
    poly = np.asarray(poly, dtype=np.float64)
    seg = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])  # cum[m] == perimeter
    perim = float(cum[-1])
    n = int(np.clip(round(perim / spacing), 3, max_props))
    truncated = int(round(perim / spacing) > max_props)
    step = perim / n
    s = np.arange(n) * step
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(poly) - 1)
    t = (s - cum[idx]) / np.maximum(seg[idx], 1e-12)
    p0 = poly[idx]
    p1 = poly[(idx + 1) % len(poly)]
    pos = p0 + (p1 - p0) * t[:, None]
    d = p1 - p0
    tang = d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
    return pos, tang, n, step, truncated
