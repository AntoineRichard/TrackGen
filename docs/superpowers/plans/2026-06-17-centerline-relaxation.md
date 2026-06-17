# Centerline-Relaxation Track Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken width-clamp inflation with a bead-chain *relaxation* stage that reshapes the centerline until a constant track width fits (thickness ≥ half_width), exposed as three selectable batched backends (xpbd default, energy, tp_sobolev) + an optional smoothing finisher, with a 8192-track benchmark, and make the repo a proper installable package.

**Architecture:** `generate → arc-length resample → relax(backend) → inflate(constant width)`. Relaxation is pure batched torch (CPU+GPU, no RNG). Inflation drops its width-clamps and gains a *real* validity gate (thickness + border self-intersection). The generator gains a real simplicity gate (relaxation cannot untangle a self-crossing init).

**Tech Stack:** Python 3.12, PyTorch (batched, device-agnostic), scipy/numpy (Bézier basis), NVIDIA Warp (RNG only), pytest, matplotlib (viz/benchmark). Reference spikes (validated): `docs/superpowers/spikes/2026-06-17-relaxation-bakeoff/{relax_xpbd,relax_energy,relax_tpsobolev,common}.py`.

**Spec:** `docs/superpowers/specs/2026-06-17-constant-width-track-relaxation-design.md`.

**Environment:** Use the venv at `.venv` (CPU torch + warp already installed): run `.venv/bin/python -m pytest ...`. If absent, `uv venv .venv --python 3.12 && uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu && uv pip install --python .venv scipy numpy pytest matplotlib warp-lang`.

**Deliberate simplification (flagged):** the spec lists `warp-lang` as an *optional* extra. This plan makes it a normal dependency to avoid a lazy-import refactor of `rng_utils`; the geometry/relaxation/inflation logic is still warp-free at the function level (CPU-testable). Revisit if a warp-free install is needed.

---

## Phase 0 — Make it a proper package (turns the red suite green)

### Task 1: Create the `track_gen/` package, add `pyproject.toml`, install editable

**Files:**
- Create dir: `track_gen/`
- Move (git mv): `geometry.py generators.py inflation.py rng_kernels.py rng_utils.py track_generator.py types.py __init__.py` → `track_gen/`
- Create: `pyproject.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Move the package modules into `track_gen/` (preserves history)**

```bash
mkdir -p track_gen
git mv geometry.py generators.py inflation.py rng_kernels.py rng_utils.py track_generator.py types.py __init__.py track_gen/
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "track_gen"
version = "0.1.0"
description = "GPU-batched race-track generator with centerline relaxation."
requires-python = ">=3.10"
dependencies = ["torch", "scipy", "numpy", "warp-lang"]

[project.optional-dependencies]
dev = ["pytest", "matplotlib"]

[tool.setuptools]
packages = ["track_gen"]
```

- [ ] **Step 3: Replace `tests/conftest.py` (drop the sys.path hack; rely on the editable install)**

```python
# track_gen is installed editable (`pip install -e .`), so `import track_gen` works
# from anywhere. No sys.path manipulation needed. types.py now lives at
# track_gen/types.py and no longer shadows the stdlib `types` module.
```

- [ ] **Step 4: Install the package editable into the venv**

Run: `uv pip install --python .venv -e ".[dev]"`
Expected: installs `track_gen` (editable) without error.

- [ ] **Step 5: Verify import works from the repo root with no stdlib shadow**

Run: `cd /home/antoiner/Documents/TrackGen && .venv/bin/python -c "import types as t; assert hasattr(t,'MappingProxyType'); import track_gen; from track_gen import geometry; from track_gen.types import TrackGenConfig; print('ok', track_gen.__version__)"`
Expected: prints `ok 0.1.0` (stdlib `types` resolves correctly; `track_gen` imports).

- [ ] **Step 6: Run the existing suite to establish a green baseline on the new layout**

Run: `.venv/bin/python -m pytest -q`
Expected: collection succeeds and the pre-existing tests pass (the import/shadow breakage is gone). Note any failures; they should be zero at this point (no behavior changed yet).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml track_gen/ tests/conftest.py
git commit -m "Restructure into installable track_gen package (fix stdlib shadow + import)"
```

---

## Phase 1 — Geometry primitives (TDD)

### Task 2: `geometry.self_intersections`

**Files:**
- Modify: `track_gen/geometry.py`
- Test: `tests/test_geometry_self_intersections.py`

- [ ] **Step 1: Write the failing test**

```python
import math
import torch
from track_gen.geometry import self_intersections


def _circle(n=64, r=1.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def _figure_eight(n=200, s=1.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([s * torch.sin(t), s * torch.sin(t) * torch.cos(t)], dim=-1)


def test_self_intersections_convex_is_zero():
    poly = _circle().unsqueeze(0)  # [1,N,2]
    assert int(self_intersections(poly)[0]) == 0


def test_self_intersections_figure_eight_is_positive():
    poly = _figure_eight().unsqueeze(0)
    assert int(self_intersections(poly)[0]) >= 1


def test_self_intersections_batched():
    poly = torch.stack([_circle(), _figure_eight(n=64)], dim=0)  # [2,64,2]
    out = self_intersections(poly)
    assert out.shape == (2,)
    assert int(out[0]) == 0 and int(out[1]) >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_geometry_self_intersections.py -q`
Expected: FAIL with `ImportError: cannot import name 'self_intersections'`.

- [ ] **Step 3: Add `circ_index_dist` + `self_intersections` to `track_gen/geometry.py`** (append at end)

```python
def circ_index_dist(n: int, device) -> torch.Tensor:
    """[n, n] circular index distance: min(|i-j|, n-|i-j|)."""
    idx = torch.arange(n, device=device)
    d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return torch.minimum(d, n - d)


def self_intersections(poly: torch.Tensor) -> torch.Tensor:
    """Count proper self-crossings of each closed polyline. poly [E, N, 2] -> [E] long.

    Tests every pair of edges (i -> i+1, j -> j+1), excluding the same edge and edges
    that share an endpoint (circular index distance <= 1). A proper crossing is the
    standard orientation test: endpoints of each segment lie on opposite sides of the
    other. Each crossing is counted once.
    """
    E, N, _ = poly.shape
    A = poly
    B = torch.roll(poly, shifts=-1, dims=1)

    def ccw(o, p, q):
        return (q[..., 1] - o[..., 1]) * (p[..., 0] - o[..., 0]) - \
               (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])

    Ai = A[:, :, None, :]; Bi = B[:, :, None, :]
    Aj = A[:, None, :, :]; Bj = B[:, None, :, :]
    d1 = ccw(Aj, Bj, Ai); d2 = ccw(Aj, Bj, Bi)
    d3 = ccw(Ai, Bi, Aj); d4 = ccw(Ai, Bi, Bj)
    cross = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))  # [E,N,N]
    circ = circ_index_dist(N, poly.device)
    cross = cross & (circ[None] > 1)
    return (cross.sum(dim=(-1, -2)) // 2).long()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_geometry_self_intersections.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add track_gen/geometry.py tests/test_geometry_self_intersections.py
git commit -m "geometry: add self_intersections + circ_index_dist"
```

### Task 3: `geometry.perimeter`, `mean_seg_len`, `separation_min`, `curvature_radius_min`, `thickness`

**Files:**
- Modify: `track_gen/geometry.py`
- Test: `tests/test_geometry_thickness.py`

- [ ] **Step 1: Write the failing test**

```python
import math
import torch
from track_gen.geometry import (
    perimeter, mean_seg_len, separation_min, curvature_radius_min, thickness,
)


def _circle(n=256, r=2.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1).unsqueeze(0)


def test_perimeter_and_spacing_of_circle():
    c = _circle(n=256, r=2.0)
    assert torch.allclose(perimeter(c), torch.tensor([2 * math.pi * 2.0]), atol=1e-2)
    assert torch.allclose(mean_seg_len(c), perimeter(c) / 256)


def test_curvature_radius_min_of_circle_is_radius():
    c = _circle(n=400, r=2.0)
    assert torch.allclose(curvature_radius_min(c), torch.tensor([2.0]), atol=1e-2)


def test_separation_min_of_circle():
    # On a circle of radius r, the min non-adjacent distance (just past the band) is
    # ~ a small chord; thickness should be dominated by curvature radius (= r).
    c = _circle(n=256, r=2.0)
    band = torch.tensor([4])
    sep = separation_min(c, band)
    assert sep.shape == (1,) and sep[0] > 0


def test_thickness_of_circle_is_radius():
    c = _circle(n=400, r=2.0)
    band = torch.tensor([8])
    th = thickness(c, band)
    assert torch.allclose(th, torch.tensor([2.0]), atol=2e-2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_geometry_thickness.py -q`
Expected: FAIL with ImportError on `perimeter`.

- [ ] **Step 3: Add the functions to `track_gen/geometry.py`** (append at end)

```python
def perimeter(points: torch.Tensor) -> torch.Tensor:
    """Closed-loop perimeter of each polyline. points [E, N, 2] -> [E]."""
    seg = torch.roll(points, shifts=-1, dims=1) - points
    return torch.linalg.norm(seg, dim=-1).sum(dim=1)


def mean_seg_len(points: torch.Tensor) -> torch.Tensor:
    """Mean segment length (rest spacing) = perimeter / N. [E]."""
    return perimeter(points) / points.shape[1]


def separation_min(points: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Min distance between any two non-adjacent points (circular index dist > band).

    points [E, N, 2]; band [E] long. Returns [E]. Pairs within band indices are
    excluded (set to +inf) before the global min.
    """
    E, N, _ = points.shape
    dmat = torch.cdist(points, points)                       # [E,N,N]
    circ = circ_index_dist(N, points.device)                 # [N,N]
    mask = circ[None] <= band.view(E, 1, 1)
    dmat = dmat.masked_fill(mask, float("inf"))
    return dmat.amin(dim=(-1, -2))


def curvature_radius_min(points: torch.Tensor) -> torch.Tensor:
    """1 / max Menger curvature over the loop. points [E, N, 2] -> [E]."""
    kappa = menger_curvature(points)                         # [E,N]
    return 1.0 / kappa.amax(dim=1).clamp_min(1e-12)


def thickness(points: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Discrete curve thickness = min(min-curvature-radius, 0.5 * separation_min). [E]."""
    return torch.minimum(curvature_radius_min(points), 0.5 * separation_min(points, band))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_geometry_thickness.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add track_gen/geometry.py tests/test_geometry_thickness.py
git commit -m "geometry: add perimeter/mean_seg_len/separation_min/curvature_radius_min/thickness"
```

---

## Phase 2 — Relaxation: dispatcher + XPBD backend + chunking (TDD)

### Task 4: `relaxation.py` — dispatcher skeleton + shared helpers + XPBD backend

**Files:**
- Create: `track_gen/relaxation.py`
- Test: `tests/test_relaxation_xpbd.py`

- [ ] **Step 1: Write the failing test**

```python
import math
import torch
from track_gen.types import TrackGenConfig
from track_gen import relaxation, geometry


def _star(n=256, r0=1.0, amp=0.6, k=7):
    """A wiggly star: low curvature radius at the spikes (sharp corners)."""
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def _cfg(**ov):
    base = dict(device="cpu", num_envs=1, num_points=256, relax_solver="xpbd",
                half_width=0.05, relax_iters=200, relax_bend_relax=1.5, relax_margin=0.15)
    base.update(ov)
    return TrackGenConfig(**base)


def test_xpbd_rounds_sharp_corners_to_thickness_target():
    c0 = _star(n=256, r0=1.0, amp=0.6, k=7).unsqueeze(0)  # [1,256,2]
    cfg = _cfg(half_width=0.05)
    out = relaxation.relax(c0, cfg)
    band = relaxation._band(c0, cfg)
    th = geometry.thickness(out, band)
    assert out.shape == c0.shape
    assert float(th[0]) >= 0.98 * cfg.half_width  # reached thickness target


def test_xpbd_is_deterministic():
    c0 = _star().unsqueeze(0)
    cfg = _cfg()
    a = relaxation.relax(c0, cfg)
    b = relaxation.relax(c0, cfg)
    assert torch.allclose(a, b)


def test_relax_disabled_is_identity():
    c0 = _star().unsqueeze(0)
    cfg = _cfg(relax_enable=False)
    assert torch.allclose(relaxation.relax(c0, cfg), c0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_relaxation_xpbd.py -q`
Expected: FAIL — `relaxation` import error / `TrackGenConfig` missing `relax_solver`. This task is self-contained: Step 3a below adds the relaxation config fields so the test runs now; Task 6 later finalizes the full config (adds energy/tp fields, removes the deprecated width-clamp fields).

- [ ] **Step 3a: Add the relaxation config fields to `track_gen/types.py`** (inside `TrackGenConfig`, after the width params; the full config — including energy/tp fields and removal of deprecated width-clamp fields — is finalized in Task 6)

```python
    # --- Relaxation: backend selection + scale ---
    relax_enable: bool = True
    relax_solver: str = "xpbd"            # {"xpbd","energy","tp_sobolev"}
    relax_chunk_size: int | None = None   # env-chunk the dense [E,N,N] term
    relax_tol: float = 0.02               # target = (1 - tol) * half_width
    relax_band: int | None = None         # None => round(D / L0) per track
    relax_iters: int = 150
    relax_sep_relax: float = 1.0
    relax_spc_relax: float = 1.0
    relax_bend_relax: float = 1.5
    relax_margin: float = 0.15
```

- [ ] **Step 3b: Create `track_gen/relaxation.py`**

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centerline relaxation: reshape a closed, arc-length-uniform centerline so a
constant-width inflation becomes valid (thickness >= half_width).

Pure batched torch, device-agnostic (CPU+GPU), CPU-testable, RNG-free (deterministic).
Three selectable backends behind relax(): xpbd (default), energy, tp_sobolev, plus an
optional tangent-point/Sobolev smoothing finisher. Reference (validated) spikes live
under docs/superpowers/spikes/2026-06-17-relaxation-bakeoff/.
"""
from __future__ import annotations

import torch

from . import geometry


def _roll(x, k):
    return torch.roll(x, shifts=k, dims=1)


def _safe_norm(v, eps=1e-9):
    return torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def _band(center: torch.Tensor, config) -> torch.Tensor:
    """Excluded-neighbour half-window per track: round(D / L0), >= 1. [E] long."""
    if config.relax_band is not None:
        E = center.shape[0]
        return torch.full((E,), int(config.relax_band), dtype=torch.long, device=center.device)
    D = 2.0 * float(config.half_width)
    L0 = geometry.mean_seg_len(center)
    return (D / L0).round().long().clamp_min(1)


def _resample_uniform(center: torch.Tensor, n: int) -> torch.Tensor:
    """Arc-length-uniform resample of each closed loop to n points (keeps n)."""
    E = center.shape[0]
    closed = torch.cat([center, center[:, :1]], dim=1)               # [E,n+1,2]
    seg = torch.linalg.norm(closed[:, 1:] - closed[:, :-1], dim=-1)  # [E,n]
    s = torch.cat([torch.zeros(E, 1, device=center.device, dtype=center.dtype),
                   torch.cumsum(seg, dim=1)], dim=1)                 # [E,n+1]
    total = s[:, -1:]
    targets = torch.arange(n, dtype=center.dtype, device=center.device)[None] * (total / n)
    out = torch.empty_like(center)
    for e in range(E):
        idx = torch.searchsorted(s[e, 1:], targets[e], right=False).clamp(max=seg.shape[1] - 1)
        frac = ((targets[e] - s[e, idx]) / seg[e, idx].clamp_min(1e-12)).clamp(0, 1).unsqueeze(-1)
        out[e] = closed[e, idx] + frac * (closed[e, idx + 1] - closed[e, idx])
    return out


# ---------------------------------------------------------------------------
# XPBD backend (default)
# ---------------------------------------------------------------------------

def _separation_disp(center, mask_keep, D, margin):
    """Jacobi-averaged symmetric push for non-adjacent pairs closer than D*(1+margin)."""
    diff = center[:, :, None, :] - center[:, None, :, :]    # [E,N,N,2] i - j
    dist = _safe_norm(diff)                                 # [E,N,N,1]
    target = D * (1.0 + margin)
    pen = (target - dist.squeeze(-1)).clamp_min(0.0)        # [E,N,N]
    violated = (pen > 0) & mask_keep
    unit = diff / dist
    corr = 0.5 * pen.unsqueeze(-1) * unit * violated.unsqueeze(-1)
    disp = corr.sum(dim=2)                                  # [E,N,2]
    cnt = violated.sum(dim=2).clamp_min(1).unsqueeze(-1)
    return disp / cnt


def _spacing_disp(center, L0):
    """Project each edge toward rest length L0; each bead is in 2 edges -> /2."""
    d = _roll(center, -1) - center
    dist = _safe_norm(d)
    unit = d / dist
    err = (dist.squeeze(-1) - L0.unsqueeze(1))
    fwd = 0.5 * err.unsqueeze(-1) * unit
    return (fwd - _roll(fwd, 1)) / 2.0


def _bending_disp(center, R_min):
    """Pull the apex toward its neighbours' midpoint when local radius < R_min.
    Returns (raw_disp, apex->midpoint vector) so the caller can clamp the step
    to never flip the corner."""
    pp, pc, pn = _roll(center, 1), center, _roll(center, -1)
    a, b, c = pc - pp, pn - pc, pn - pp
    la, lb, lc = (_safe_norm(x).squeeze(-1) for x in (a, b, c))
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    area = 0.5 * cross.abs()
    kappa = 4.0 * area / (la * lb * lc).clamp_min(1e-12)
    radius = 1.0 / kappa.clamp_min(1e-12)
    mid = 0.5 * (pp + pn)
    toward = mid - pc
    deficit = (R_min - radius).clamp_min(0.0) / R_min
    return deficit.unsqueeze(-1) * toward, toward


def _relax_xpbd(center0, band, config):
    E, N, _ = center0.shape
    hw = float(config.half_width)
    D = 2.0 * hw
    R_min = hw
    target = (1.0 - float(config.relax_tol)) * hw
    sep_relax = float(config.relax_sep_relax)
    spc_relax = float(config.relax_spc_relax)
    bend_relax = float(config.relax_bend_relax)
    margin = float(config.relax_margin)

    center = center0.clone()
    L0 = geometry.perimeter(center0) / N
    circ = geometry.circ_index_dist(N, center0.device)
    mask_keep = circ[None] > band.view(E, 1, 1)
    active = torch.ones(E, dtype=torch.bool, device=center0.device)

    for _ in range(int(config.relax_iters)):
        if not bool(active.any()):
            break
        disp = sep_relax * _separation_disp(center, mask_keep, D, margin)
        disp = disp + spc_relax * _spacing_disp(center, L0)
        if bend_relax > 0.0:
            bend, toward = _bending_disp(center, R_min)
            step = bend_relax * bend
            max_len = torch.linalg.norm(toward, dim=-1, keepdim=True)
            step_len = torch.linalg.norm(step, dim=-1, keepdim=True)
            disp = disp + step * (max_len / step_len.clamp_min(1e-12)).clamp(max=1.0)
        center = torch.where(active[:, None, None], center + disp, center)
        th = geometry.thickness(center, band)
        active = active & (th < target)

    return _resample_uniform(center, N)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BACKENDS = {"xpbd": _relax_xpbd}  # energy/tp_sobolev added in later tasks


def relax(center: torch.Tensor, config) -> torch.Tensor:
    """Reshape a closed, arc-length-uniform centerline [E,N,2] so thickness >= half_width.
    Dispatches on config.relax_solver; returns the relaxed centerline (same N)."""
    if not config.relax_enable:
        return center
    backend = _BACKENDS.get(config.relax_solver)
    if backend is None:
        raise ValueError(f"Unknown relax_solver {config.relax_solver!r}; "
                         f"expected one of {sorted(_BACKENDS)}.")
    band = _band(center, config)
    return backend(center, band, config)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_relaxation_xpbd.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add track_gen/relaxation.py track_gen/types.py tests/test_relaxation_xpbd.py
git commit -m "relaxation: XPBD backend + dispatcher + shared helpers"
```

### Task 5: Separation-limited case + env-chunking

**Files:**
- Modify: `track_gen/relaxation.py`
- Test: `tests/test_relaxation_chunk.py`, add to `tests/test_relaxation_xpbd.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_relaxation_xpbd.py`:

```python
def test_xpbd_pushes_apart_near_touch():
    # Two near-touching strands (a pinched oval): separation-limited, not curvature.
    import math
    n = 256
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    x = torch.cos(t)
    y = 0.15 * torch.sin(t)          # very flat oval -> top/bottom strands ~0.3 apart
    c0 = torch.stack([x, y], dim=-1).unsqueeze(0)
    cfg = _cfg(half_width=0.05, relax_iters=300)
    out = relaxation.relax(c0, cfg)
    band = relaxation._band(c0, cfg)
    assert float(geometry.separation_min(out, band)[0]) >= 2 * 0.05 * 0.98
```

Create `tests/test_relaxation_chunk.py`:

```python
import math
import torch
from track_gen.types import TrackGenConfig
from track_gen import relaxation


def _star(n=256, r0=1.0, amp=0.6, k=7, phase=0.0):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1] + phase
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def test_chunked_equals_unchunked():
    c0 = torch.stack([_star(phase=p) for p in torch.linspace(0, 1.0, 5)], dim=0)  # [5,256,2]
    base = dict(device="cpu", num_envs=5, num_points=256, relax_solver="xpbd",
                half_width=0.05, relax_iters=150)
    full = relaxation.relax(c0, TrackGenConfig(**base, relax_chunk_size=None))
    chunked = relaxation.relax(c0, TrackGenConfig(**base, relax_chunk_size=2))
    assert torch.allclose(full, chunked, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_relaxation_chunk.py -q`
Expected: FAIL — chunking not implemented (results equal only by luck or `relax_chunk_size` ignored → actually equal; but to be safe the test asserts equality, so it may pass trivially if chunk is ignored). To make the test meaningful, implement chunking in Step 3 and confirm it still holds.

- [ ] **Step 3: Add env-chunking to the dispatcher** (replace the `relax` body's final lines)

```python
def _chunks(e: int, size):
    if not size or size >= e:
        yield slice(0, e)
        return
    for start in range(0, e, size):
        yield slice(start, min(start + size, e))


def relax(center: torch.Tensor, config) -> torch.Tensor:
    if not config.relax_enable:
        return center
    backend = _BACKENDS.get(config.relax_solver)
    if backend is None:
        raise ValueError(f"Unknown relax_solver {config.relax_solver!r}; "
                         f"expected one of {sorted(_BACKENDS)}.")
    band = _band(center, config)
    outs = []
    for sl in _chunks(center.shape[0], config.relax_chunk_size):
        outs.append(backend(center[sl], band[sl], config))
    return torch.cat(outs, dim=0)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_relaxation_chunk.py tests/test_relaxation_xpbd.py -q`
Expected: all passed (chunked == unchunked; near-touch pushed apart).

- [ ] **Step 5: Commit**

```bash
git add track_gen/relaxation.py tests/test_relaxation_chunk.py tests/test_relaxation_xpbd.py
git commit -m "relaxation: env-chunking + separation-limited coverage"
```

---

## Phase 3 — Config cleanup, generator gate, inflation rewrite, facade

### Task 6: Finalize `TrackGenConfig` (add remaining backend fields, remove deprecated width-clamp fields)

**Files:**
- Modify: `track_gen/types.py`
- Test: `tests/test_types.py` (update)

- [ ] **Step 1: Update the failing test** — edit `tests/test_types.py` to (a) assert the new fields exist with defaults and (b) assert the deprecated fields are gone.

```python
import math
from track_gen.types import TrackGenConfig


def test_relaxation_defaults():
    cfg = TrackGenConfig()
    assert cfg.relax_enable is True
    assert cfg.relax_solver == "xpbd"
    assert cfg.relax_bend_relax == 1.5
    assert cfg.relax_margin == 0.15
    assert cfg.energy_steps == 800
    assert cfg.tp_iters == 100
    assert cfg.smooth_finish is False


def test_deprecated_width_clamp_fields_removed():
    cfg = TrackGenConfig()
    for dead in ("alpha", "clamp_self_distance", "self_distance_margin",
                 "self_distance_band", "self_distance_decimation"):
        assert not hasattr(cfg, dead), f"{dead} should be removed"
```

(Keep any existing `test_types.py` assertions that still hold; remove ones referencing the deleted fields.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_types.py -q`
Expected: FAIL (energy_steps/tp_iters missing; deprecated fields still present).

- [ ] **Step 3: Edit `track_gen/types.py`** — in `TrackGenConfig`: delete the deprecated width-clamp fields and add the energy + tp_sobolev + finisher fields. Replace the `# --- Width params ---` block's clamp lines and the relaxation block from Task 4 with:

```python
    # --- Width params ---
    half_width: float = 0.1  # constant track half-width (w)

    # --- Relaxation: backend selection + scale ---
    relax_enable: bool = True
    relax_solver: str = "xpbd"            # {"xpbd","energy","tp_sobolev"}
    relax_chunk_size: int | None = None
    relax_tol: float = 0.02
    relax_band: int | None = None
    # xpbd
    relax_iters: int = 150
    relax_sep_relax: float = 1.0
    relax_spc_relax: float = 1.0
    relax_bend_relax: float = 1.5
    relax_margin: float = 0.15
    # energy (Adam)
    energy_steps: int = 800
    energy_lr: float = 3e-3
    energy_w_sep: float = 80.0
    energy_w_len: float = 8.0
    energy_w_bend: float = 1.0
    energy_w_anchor: float = 0.01
    # tp_sobolev (standalone backend + finisher share tp_alpha/tp_beta)
    tp_iters: int = 100
    tp_tau: float = 0.7
    tp_alpha: float = 2.0
    tp_beta: float = 4.5
    # optional tangent-point/Sobolev smoothing finisher
    smooth_finish: bool = False
    smooth_finish_iters: int = 8
    smooth_finish_tau: float = 0.2
```

Delete these fields entirely: `alpha`, `clamp_self_distance`, `self_distance_margin`, `self_distance_band`, `self_distance_decimation`. Keep `w_floor`, `num_points`, `output_mode`, `spacing`, `N_max`, `max_regen_iters`, `turning_tol` and all generator fields.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_types.py -q`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add track_gen/types.py tests/test_types.py
git commit -m "types: add energy/tp_sobolev/finisher config; remove deprecated width-clamp fields"
```

### Task 7: Generator simplicity gate (reject self-intersecting centerlines)

**Files:**
- Modify: `track_gen/generators.py`
- Test: `tests/test_generator_simplicity_gate.py`

- [ ] **Step 1: Write the failing test**

```python
import math
import torch
from track_gen import geometry
from track_gen.generators import BezierCenterlineGenerator


def test_simplicity_gate_helper_flags_self_crossing():
    # The generator must expose a per-candidate simplicity check on the dense loop.
    t = torch.linspace(0, 2 * math.pi, 256 + 1)[:-1]
    fig8 = torch.stack([torch.sin(t), torch.sin(t) * torch.cos(t)], dim=-1).unsqueeze(0)
    circle = torch.stack([torch.cos(t), torch.sin(t)], dim=-1).unsqueeze(0)
    assert int(geometry.self_intersections(fig8)[0]) >= 1
    assert int(geometry.self_intersections(circle)[0]) == 0


def test_simple_gate_applied_in_generate(monkeypatch):
    pytest_warp = __import__("pytest").importorskip("warp")
    import warp as wp; wp.init()
    from track_gen.rng_utils import PerEnvSeededRNG
    from track_gen.types import TrackGenConfig
    E = 16
    seeds = torch.arange(E, dtype=torch.int32) + 7
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")
    rng.set_seeds(seeds, ids=torch.arange(E, dtype=torch.int32))
    cfg = TrackGenConfig(device="cpu", num_envs=E, scale=1.0, max_regen_iters=20)
    cl = BezierCenterlineGenerator(cfg, rng).generate(torch.arange(E))
    # Every VALID centerline must be a simple (non-self-intersecting) loop.
    for e in torch.where(cl.valid)[0].tolist():
        pts = cl.points[e]
        pts = pts[torch.isfinite(pts).all(dim=-1)].unsqueeze(0)
        assert int(geometry.self_intersections(pts)[0]) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_generator_simplicity_gate.py -q`
Expected: `test_simple_gate_applied_in_generate` FAILS (some valid centerlines self-intersect under the weak turning-only gate).

- [ ] **Step 3: Add the simplicity gate to the Bézier regen loop** — in `track_gen/generators.py`, in `BezierCenterlineGenerator.generate`, after computing `turn_ok` / `finite_ok`, add a simple-loop gate on the resampled real points. Insert a helper and extend the `ok` conjunction:

```python
        # Gate 4: the dense centerline must be a SIMPLE (non-self-intersecting) loop.
        # Relaxation by repulsion cannot untangle a figure-eight, so reject crossings here.
        resampled, _count = arc_length_resample(dense, num=self.config.num_points_per_segment)
        from .geometry import self_intersections
        simple_ok = self_intersections(resampled) == 0
        ok = angle_ok & turn_ok & finite_ok & simple_ok
```

(Replace the existing `ok = angle_ok & turn_ok & finite_ok` line. `arc_length_resample` is already imported at module top.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_generator_simplicity_gate.py -q`
Expected: passed (every valid centerline is simple).

- [ ] **Step 5: Run the generator regression tests to confirm no break**

Run: `.venv/bin/python -m pytest tests/test_generators.py -q`
Expected: passed (yields may drop slightly; `test_generate_accepts_pruned_variable_count_tracks` should still find a valid pruned track — if it cannot within seed/iters, widen `max_regen_iters` in that test).

- [ ] **Step 6: Commit**

```bash
git add track_gen/generators.py tests/test_generator_simplicity_gate.py tests/test_generators.py
git commit -m "generators: reject self-intersecting centerlines (simplicity gate)"
```

### Task 8: Inflation — constant width + real validity gate

**Files:**
- Modify: `track_gen/inflation.py`
- Test: `tests/test_inflation.py` (update), `tests/test_inflation_validity.py` (new)

- [ ] **Step 1: Write the new validity test** — create `tests/test_inflation_validity.py`:

```python
import math
import torch
from track_gen.generators import Centerline
from track_gen.types import TrackGenConfig
from track_gen import inflation, geometry


def _circle_cl(r=3.0, m=300, e=1):
    t = torch.linspace(0, 2 * math.pi, m + 1)[:-1]
    pts = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    return Centerline(points=pts, valid=torch.ones(e, dtype=torch.bool))


def _fig8_cl(s=2.0, m=400, e=1):
    t = torch.linspace(0, 2 * math.pi, m + 1)[:-1]
    pts = torch.stack([s * torch.sin(t), s * torch.sin(t) * torch.cos(t)], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    return Centerline(points=pts, valid=torch.ones(e, dtype=torch.bool))


def _cfg(**ov):
    base = dict(device="cpu", num_envs=1, output_mode="fixed", num_points=256,
                half_width=0.4, turning_tol=0.2, w_floor=1e-3, relax_enable=False)
    base.update(ov)
    return TrackGenConfig(**base)


def test_constant_width_on_circle():
    cl = _circle_cl(r=5.0, m=200, e=1)
    cfg = _cfg(num_points=256, half_width=0.4)
    track = inflation.inflate(cl, cfg)
    w = torch.linalg.norm(track.outer - track.center, dim=-1)
    assert torch.allclose(w, torch.full_like(w, 0.4), atol=1e-3)  # CONSTANT width
    assert bool(track.valid[0])


def test_validity_flags_self_crossing_border():
    # A tight ellipse at a large half-width: borders cross even though the centerline
    # is simple. The real validity gate must catch it (the old gate did not).
    t = torch.linspace(0, 2 * math.pi, 400 + 1)[:-1]
    pts = torch.stack([4.0 * torch.cos(t), 0.8 * torch.sin(t)], dim=-1).unsqueeze(0)
    cl = Centerline(points=pts, valid=torch.ones(1, dtype=torch.bool))
    cfg = _cfg(num_points=256, half_width=2.0)  # > min curvature radius -> inner border folds
    track = inflation.inflate(cl, cfg)
    crossings = geometry.self_intersections(track.inner) + geometry.self_intersections(track.outer)
    assert int(crossings[0]) > 0
    assert not bool(track.valid[0])  # MUST be flagged invalid


def test_validity_flags_figure_eight():
    cl = _fig8_cl()
    cfg = _cfg(num_points=256, half_width=0.2)
    track = inflation.inflate(cl, cfg)
    assert not bool(track.valid[0])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_inflation_validity.py -q`
Expected: FAIL — `test_validity_flags_self_crossing_border` fails (old gate misses border crossings), and constant-width assert fails (old `_width_stage` clamps by curvature).

- [ ] **Step 3: Rewrite `_width_stage` and `_validity_stage` in `track_gen/inflation.py`**

Replace `_width_stage` with a constant-width version:

```python
def _width_stage(center: torch.Tensor, kappa: torch.Tensor, config, eps: float = 1e-8):
    """Constant half-width. Relaxation guarantees thickness >= half_width upstream, so
    no curvature/self-distance clamp is needed. kappa is accepted for signature
    compatibility but unused."""
    w = torch.full(center.shape[:2], float(config.half_width), device=center.device, dtype=center.dtype)
    return w
```

Replace `_validity_stage` with the real gate (uses border self-intersection + thickness):

```python
def _validity_stage(center, w, count, gen_valid, config, outer=None, inner=None) -> torch.Tensor:
    """Real per-track validity: generation flag AND closed-loop turning AND width floor
    AND no-NaN AND thickness >= (1-tol)*half_width AND zero border self-intersections."""
    e, n = w.shape
    real = _real_point_mask(count, n, w.device)

    turning = geometry.turning_number(center)
    turn_ok = (turning.abs() - 2.0 * math.pi).abs() <= float(config.turning_tol)
    w_ok = torch.where(real, w > float(config.w_floor), torch.ones_like(real)).all(dim=1)
    nan_per_point = torch.isnan(center).any(dim=-1)
    no_nan = ~(nan_per_point & real).any(dim=1)

    D = 2.0 * float(config.half_width)
    L0 = geometry.mean_seg_len(center).clamp_min(1e-9)
    band = (D / L0).round().long().clamp_min(1)
    th = geometry.thickness(center, band)
    th_ok = th >= (1.0 - float(config.relax_tol)) * float(config.half_width)

    if outer is None or inner is None:
        border_ok = torch.ones(e, dtype=torch.bool, device=center.device)
    else:
        crossings = geometry.self_intersections(torch.nan_to_num(outer, nan=0.0)) + \
                    geometry.self_intersections(torch.nan_to_num(inner, nan=0.0))
        border_ok = crossings == 0

    return gen_valid.to(torch.bool) & turn_ok & w_ok & no_nan & th_ok & border_ok
```

Update `inflate()` to pass `outer`/`inner` into `_validity_stage`:

```python
def inflate(centerline, config) -> Track:
    res = _resample_stage(centerline, config)
    center, count = res.center, res.count
    T, Nrm, kappa = _frame_curvature_stage(center)
    w = _width_stage(center, kappa, config)
    outer, inner = _offset_stage(center, Nrm, w)
    valid = _validity_stage(center, w, count, centerline.valid, config, outer=outer, inner=inner)
    arclen, length = _arclength(center, count)
    return Track(outer=outer, center=center, inner=inner, tangent=T, normal=Nrm,
                 arclen=arclen, length=length, valid=valid, count=count)
```

- [ ] **Step 4: Update `tests/test_inflation.py`** — remove the curvature/self-distance clamp tests that no longer apply and fix configs:
  - DELETE `test_width_no_fold_on_ellipse` and `test_self_distance_clamp_prevents_overlap_on_near_touch` (the clamps are gone).
  - In `test_width_bounded_by_w_max_on_circle`: drop the `alpha=` and `clamp_self_distance=` kwargs; keep the `allclose(w, half_width)` assertion (now constant by construction).
  - In `fixed_config`: remove `clamp_self_distance=False` from the default kwargs.
  - In every test that passes `alpha=...` or `clamp_self_distance=...`/`self_distance_*`, remove those kwargs.
  - In `_run_to_width` and `test_validity_*`: keep; they exercise `_validity_stage`. Update `test_validity_*` calls to `inflation._validity_stage(center, w, count, cl.valid, cfg)` (outer/inner default to None → border check skipped at the stage level; full border check is covered by `test_inflation_validity.py`).

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_inflation.py tests/test_inflation_validity.py -q`
Expected: passed.

- [ ] **Step 6: Commit**

```bash
git add track_gen/inflation.py tests/test_inflation.py tests/test_inflation_validity.py
git commit -m "inflation: constant width + real validity gate (thickness + border crossings)"
```

### Task 9: Facade wiring (generate → resample → relax → inflate) + end-to-end yield

**Files:**
- Modify: `track_gen/track_generator.py`, `track_gen/inflation.py`
- Test: `tests/test_track_generator_facade.py` (update), `tests/test_end_to_end_relaxation.py` (new)

- [ ] **Step 1: Write the end-to-end test** — create `tests/test_end_to_end_relaxation.py`:

```python
import torch
import pytest
from track_gen import geometry


@pytest.fixture
def warp_rng():
    pytest.importorskip("warp")
    import warp as wp; wp.init()
    from track_gen.rng_utils import PerEnvSeededRNG

    def make(E, seed=20):
        seeds = torch.arange(E, dtype=torch.int32) + seed
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")
        rng.set_seeds(seeds, ids=torch.arange(E, dtype=torch.int32))
        return rng
    return make


def test_xpbd_pipeline_makes_constant_width_tracks_valid(warp_rng):
    from track_gen.types import TrackGenConfig
    from track_gen.track_generator import TrackGenerator
    E = 32
    cfg = TrackGenConfig(generator="bezier", device="cpu", num_envs=E, scale=1.0,
                         half_width=0.03, num_points=256, output_mode="fixed",
                         relax_solver="xpbd", relax_iters=200, relax_bend_relax=1.5,
                         relax_margin=0.15, max_regen_iters=20)
    track = TrackGenerator(cfg, warp_rng(E)).generate(E)
    # Relaxed + constant-width inflation: a large majority must be valid (was ~3% before).
    assert track.valid.float().mean().item() >= 0.9
    # Width is constant where valid.
    w = torch.linalg.norm(track.outer - track.center, dim=-1)
    assert torch.allclose(w, torch.full_like(w, 0.03), atol=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_end_to_end_relaxation.py -q`
Expected: FAIL — facade does not yet call `relax` (resample happens inside inflate, so relaxation isn't applied → low valid yield).

- [ ] **Step 3: Wire relaxation into the pipeline.** Add a resample helper to `inflation.py` and call relax in the facade. In `track_gen/inflation.py`, expose the resampled centerline (it already computes it in `_resample_stage`). In `track_gen/track_generator.py`, change `generate`:

```python
    def generate(self, num_or_ids) -> Track:
        ids = self._resolve_ids(num_or_ids)
        centerline: Centerline = self._generator.generate(ids)
        # Resample to a uniform centerline, relax it (thickness >= half_width), then inflate.
        from . import relaxation
        from .inflation import _resample_stage, _ResampleResult
        res = _resample_stage(centerline, self._config)            # arc-length uniform
        relaxed = relaxation.relax(res.center, self._config)       # bead-chain relaxation
        relaxed_cl = Centerline(points=relaxed, valid=centerline.valid)
        return inflate(relaxed_cl, self._config)
```

Note: `inflate` will re-resample the already-uniform `relaxed` centerline (idempotent for fixed mode); this keeps `inflate`'s public contract unchanged. (If profiling later shows the double resample matters, add an `already_resampled` fast path — out of scope here.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_end_to_end_relaxation.py -q`
Expected: passed (>=90% valid, constant width).

- [ ] **Step 5: Update + run the facade tests**

Run: `.venv/bin/python -m pytest tests/test_track_generator_facade.py -q`
Expected: passed. If any facade test asserted the old variable-width or deprecated config fields, update it to the constant-width/relaxed pipeline.

- [ ] **Step 6: Commit**

```bash
git add track_gen/track_generator.py track_gen/inflation.py tests/test_end_to_end_relaxation.py tests/test_track_generator_facade.py
git commit -m "facade: generate -> resample -> relax -> inflate (constant-width valid tracks)"
```

---

## Phase 4 — Energy backend (TDD)

### Task 10: `energy` backend (Adam soft-penalty)

**Files:**
- Modify: `track_gen/relaxation.py`
- Test: `tests/test_relaxation_energy.py`

- [ ] **Step 1: Write the failing test**

```python
import math
import torch
from track_gen.types import TrackGenConfig
from track_gen import relaxation, geometry


def _star(n=256, r0=1.0, amp=0.5, k=6):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def test_energy_backend_raises_thickness():
    c0 = _star().unsqueeze(0)
    cfg = TrackGenConfig(device="cpu", num_envs=1, num_points=256, relax_solver="energy",
                         half_width=0.05, energy_steps=400)
    out = relaxation.relax(c0, cfg)
    band = relaxation._band(c0, cfg)
    assert out.shape == c0.shape
    # Soft solver: thickness should improve substantially over the init.
    assert float(geometry.thickness(out, band)[0]) > float(geometry.thickness(c0, band)[0])


def test_energy_backend_deterministic():
    c0 = _star().unsqueeze(0)
    cfg = TrackGenConfig(device="cpu", num_envs=1, num_points=256, relax_solver="energy",
                         half_width=0.05, energy_steps=100)
    torch.manual_seed(0); a = relaxation.relax(c0, cfg)
    torch.manual_seed(0); b = relaxation.relax(c0, cfg)
    assert torch.allclose(a, b)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_relaxation_energy.py -q`
Expected: FAIL — `relax_solver='energy'` raises `ValueError` (backend not registered).

- [ ] **Step 3: Add `_relax_energy` and register it** in `track_gen/relaxation.py` (append the function, then extend `_BACKENDS`)

```python
def _energy(center, x0, circ, band, D, w_sep, w_len, w_bend, w_anchor, L0):
    E, N, _ = center.shape
    dmat = torch.cdist(center, center)                      # [E,N,N]
    mask = circ[None] > band.view(E, 1, 1)
    viol = torch.relu(D - dmat) * mask
    e_sep = 0.5 * w_sep * (viol ** 2).sum()
    seg = _roll(center, -1) - center
    seglen = torch.linalg.norm(seg, dim=-1)
    e_len = w_len * ((seglen - L0.view(E, 1)) ** 2).sum()
    lap = _roll(center, -1) - 2.0 * center + _roll(center, 1)
    e_bend = w_bend * (lap ** 2).sum()
    e_anchor = w_anchor * ((center - x0) ** 2).sum()
    return e_sep + e_len + e_bend + e_anchor


def _relax_energy(center0, band, config):
    E, N, _ = center0.shape
    D = 2.0 * float(config.half_width)
    circ = geometry.circ_index_dist(N, center0.device).to(center0.dtype)
    L0 = geometry.mean_seg_len(center0).detach()
    x0 = center0.detach().clone()
    x = center0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=float(config.energy_lr))
    for _ in range(int(config.energy_steps)):
        opt.zero_grad(set_to_none=True)
        e = _energy(x, x0, circ, band, D, float(config.energy_w_sep), float(config.energy_w_len),
                    float(config.energy_w_bend), float(config.energy_w_anchor), L0)
        e.backward()
        opt.step()
    return _resample_uniform(x.detach(), N)


_BACKENDS["energy"] = _relax_energy
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_relaxation_energy.py -q`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add track_gen/relaxation.py tests/test_relaxation_energy.py
git commit -m "relaxation: energy (Adam soft-penalty) backend"
```

---

## Phase 5 — Tangent-point/Sobolev backend + smoothing finisher (TDD)

### Task 11: `tp_sobolev` backend

**Files:**
- Modify: `track_gen/relaxation.py`
- Test: `tests/test_relaxation_tp.py`

- [ ] **Step 1: Write the failing test**

```python
import math
import torch
from track_gen.types import TrackGenConfig
from track_gen import relaxation, geometry


def _flat_oval(n=256):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([torch.cos(t), 0.25 * torch.sin(t)], dim=-1).unsqueeze(0)


def test_tp_sobolev_backend_runs_and_increases_separation():
    c0 = _flat_oval()
    cfg = TrackGenConfig(device="cpu", num_envs=1, num_points=256, relax_solver="tp_sobolev",
                         half_width=0.05, tp_iters=60, tp_tau=0.7)
    band = relaxation._band(c0, cfg)
    out = relaxation.relax(c0, cfg)
    assert out.shape == c0.shape
    assert torch.isfinite(out).all()
    # Repulsion should increase the min non-adjacent separation of the pinched oval.
    assert float(geometry.separation_min(out, band)[0]) > float(geometry.separation_min(c0, band)[0])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_relaxation_tp.py -q`
Expected: FAIL — `relax_solver='tp_sobolev'` raises `ValueError`.

- [ ] **Step 3: Add the tangent-point/Sobolev machinery and backend** in `track_gen/relaxation.py` (append; adapted from `relax_tpsobolev.py`)

```python
def _dual_weights(center):
    e = _roll(center, -1) - center
    el = torch.linalg.norm(e, dim=-1)
    return 0.5 * (el + _roll(el, 1))


def _tp_tangents(center):
    return geometry.safe_normalize(_roll(center, -1) - _roll(center, 1))


def _tp_energy(center, pair_mask, alpha, beta, eps):
    T = _tp_tangents(center)
    w = _dual_weights(center)
    diff = center[:, None, :, :] - center[:, :, None, :]      # [E,N,N,2] (x_j - x_i)
    d2 = (diff * diff).sum(-1)
    wedge = diff[..., 0] * T[:, :, None, 1] - diff[..., 1] * T[:, :, None, 0]
    num = (wedge.abs() + eps) ** alpha
    den = (d2 + eps * eps) ** (beta * 0.5)
    k = (num / den) * (w[:, :, None] * w[:, None, :]) * pair_mask
    return k.sum()


def _length_grad(center):
    u_fwd = geometry.safe_normalize(_roll(center, -1) - center)
    return -u_fwd + _roll(u_fwd, 1)


def _ring_spectral_filter(n, s, eps_reg, device, dtype):
    k = torch.arange(n // 2 + 1, device=device, dtype=dtype)
    lam = 2.0 - 2.0 * torch.cos(2.0 * torch.pi * k / n)
    return 1.0 / (lam.clamp_min(0.0) ** s + eps_reg)


def _precondition_fft(grad, inv_filter):
    G = torch.fft.rfft(grad, dim=1) * inv_filter[None, :, None]
    return torch.fft.irfft(G, n=grad.shape[1], dim=1)


def _tp_flow(center0, band, config, n_steps, tau, early_stop):
    """Shared tangent-point/Sobolev gradient flow. Used by the standalone backend
    (early_stop=True, n_steps=tp_iters) and the smoothing finisher (early_stop=False)."""
    device = center0.device
    E, N, _ = center0.shape
    alpha = float(config.tp_alpha); beta = float(config.tp_beta)
    eps = 1e-4
    s = (beta - 1.0) / (2.0 * alpha)
    eps_reg = 1e-3
    hw = float(config.half_width)
    target = (1.0 - float(config.relax_tol)) * hw

    circ = geometry.circ_index_dist(N, device)
    pair_mask = (circ[None] > band.view(E, 1, 1)).to(center0.dtype)
    center = center0.detach().clone()
    L0_total = geometry.perimeter(center0).detach()
    inv_filter = _ring_spectral_filter(N, s, eps_reg, device, center0.dtype)
    active = torch.ones(E, dtype=torch.bool, device=device)

    for _ in range(int(n_steps)):
        if early_stop:
            th = geometry.thickness(center, band)
            active = active & (th < target)
            if not bool(active.any()):
                break
        x = center.detach().clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(_tp_energy(x, pair_mask, alpha, beta, eps), x)
        with torch.no_grad():
            g = _precondition_fft(grad, inv_filter)
            lg = _length_grad(center)
            Ainv_lg = _precondition_fft(lg, inv_filter)
            num = (g * lg).sum(dim=(1, 2))
            den = (lg * Ainv_lg).sum(dim=(1, 2)).clamp_min(1e-12)
            g = g - (num / den)[:, None, None] * Ainv_lg
            g = g - g.mean(dim=1, keepdim=True)
            gmax = torch.linalg.norm(g, dim=-1).amax(dim=1).clamp_min(1e-12)
            step = (tau * geometry.mean_seg_len(center) / gmax)[:, None, None] * g
            move = active[:, None, None].to(center.dtype) if early_stop else 1.0
            center = center - step * move
            cur_len = geometry.perimeter(center).clamp_min(1e-9)
            bc = center.mean(dim=1, keepdim=True)
            scale = (L0_total / cur_len)[:, None, None]
            if early_stop:
                scale = torch.where(active[:, None, None], scale, torch.ones_like(scale))
            center = bc + (center - bc) * scale
    return _resample_uniform(center, N)


def _relax_tp(center0, band, config):
    return _tp_flow(center0, band, config, n_steps=config.tp_iters, tau=config.tp_tau, early_stop=True)


_BACKENDS["tp_sobolev"] = _relax_tp
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_relaxation_tp.py -q`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add track_gen/relaxation.py tests/test_relaxation_tp.py
git commit -m "relaxation: tangent-point/Sobolev (Repulsive Curves) backend"
```

### Task 12: `smooth_finish` finisher in the dispatcher

**Files:**
- Modify: `track_gen/relaxation.py`
- Test: `tests/test_relaxation_finisher.py`

- [ ] **Step 1: Write the failing test**

```python
import math
import torch
from track_gen.types import TrackGenConfig
from track_gen import relaxation, geometry


def _star(n=256, r0=1.0, amp=0.6, k=7):
    t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    r = r0 + amp * torch.cos(k * t)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def _clearance_cv(center, band):
    kappa = geometry.menger_curvature(center)
    crad = 1.0 / kappa.clamp_min(1e-12)
    sep = geometry.separation_min(center, band)  # global; use as a coarse evenness proxy
    hc = torch.minimum(crad, 0.5 * sep.unsqueeze(1))
    return (hc.std(dim=1) / hc.mean(dim=1).clamp_min(1e-12))


def test_finisher_keeps_validity_and_smooths():
    c0 = _star().unsqueeze(0)
    base = dict(device="cpu", num_envs=1, num_points=256, relax_solver="xpbd",
                half_width=0.05, relax_iters=200, relax_bend_relax=1.5, relax_margin=0.15)
    band = relaxation._band(c0, TrackGenConfig(**base))
    no_fin = relaxation.relax(c0, TrackGenConfig(**base, smooth_finish=False))
    fin = relaxation.relax(c0, TrackGenConfig(**base, smooth_finish=True, smooth_finish_iters=8, smooth_finish_tau=0.2))
    # Finisher keeps thickness at/above target and does not destroy it.
    assert float(geometry.thickness(fin, band)[0]) >= 0.96 * 0.05
    assert fin.shape == c0.shape
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_relaxation_finisher.py -q`
Expected: FAIL — `smooth_finish` is ignored (no effect / attribute path), test for shape/thickness may pass trivially; confirm the finisher actually runs by asserting it changes the curve. Add: `assert not torch.allclose(no_fin, fin)`.

- [ ] **Step 3: Apply the finisher in the dispatcher** — in `relax`, after the backend loop builds `out = torch.cat(outs, ...)`, add the finisher pass:

```python
def relax(center: torch.Tensor, config) -> torch.Tensor:
    if not config.relax_enable:
        return center
    backend = _BACKENDS.get(config.relax_solver)
    if backend is None:
        raise ValueError(f"Unknown relax_solver {config.relax_solver!r}; "
                         f"expected one of {sorted(_BACKENDS)}.")
    band = _band(center, config)
    outs = [backend(center[sl], band[sl], config)
            for sl in _chunks(center.shape[0], config.relax_chunk_size)]
    out = torch.cat(outs, dim=0)
    if config.smooth_finish:
        fb = _band(out, config)
        outs = [_tp_flow(out[sl], fb[sl], config,
                         n_steps=config.smooth_finish_iters, tau=config.smooth_finish_tau,
                         early_stop=False)
                for sl in _chunks(out.shape[0], config.relax_chunk_size)]
        out = torch.cat(outs, dim=0)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_relaxation_finisher.py -q`
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add track_gen/relaxation.py tests/test_relaxation_finisher.py
git commit -m "relaxation: optional tangent-point/Sobolev smoothing finisher"
```

---

## Phase 6 — Benchmark harness

### Task 13: `benchmarks/benchmark_relaxation.py` (E=8192, GPU+CPU, validity/time/memory/quality)

**Files:**
- Create: `benchmarks/benchmark_relaxation.py`
- Test: `tests/test_benchmark_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
import torch
import pytest


def test_benchmark_runs_small_cpu():
    pytest.importorskip("warp")
    from benchmarks.benchmark_relaxation import run_benchmark
    # Tiny batch on CPU: every backend produces a row without error.
    rows = run_benchmark(E=8, N=128, half_width=0.03, device="cpu",
                         solvers=("xpbd", "energy", "tp_sobolev"),
                         energy_steps=50, tp_iters=20, relax_iters=40, seed=20)
    assert set(r["solver"] for r in rows) >= {"xpbd", "energy", "tp_sobolev"}
    for r in rows:
        assert 0.0 <= r["valid_frac"] <= 1.0
        assert r["seconds"] >= 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_benchmark_smoke.py -q`
Expected: FAIL — `benchmarks.benchmark_relaxation` does not exist. (Add `benchmarks/__init__.py` empty file so it is importable.)

- [ ] **Step 3: Create `benchmarks/__init__.py` (empty) and `benchmarks/benchmark_relaxation.py`**

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark the three relaxation backends on a large batch (default E=8192).

Reports per backend: validity yield (thickness + zero border/centerline crossings),
wall-clock, peak GPU memory, and shape-quality metrics (displacement, clearance CV,
max curvature). Runs on GPU (primary) and CPU (fallback). Run directly:

    .venv/bin/python -m benchmarks.benchmark_relaxation            # auto device, E=8192
    .venv/bin/python -m benchmarks.benchmark_relaxation --E 2048 --cpu
"""
from __future__ import annotations
import argparse, time
import torch

from track_gen.types import TrackGenConfig
from track_gen import relaxation, geometry, inflation
from track_gen.generators import BezierCenterlineGenerator, Centerline


def _gen_simple_tracks(E, N, scale, device, seed):
    """Generate E simple, arc-length-uniform Bezier centerlines (needs warp RNG)."""
    import warp as wp; wp.init()
    from track_gen.rng_utils import PerEnvSeededRNG
    kept = []
    s = seed
    while len(kept) < E:
        B = min(2048, 2 * (E - len(kept)) + 256)
        seeds = torch.arange(B, dtype=torch.int32) + s
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=B, device=device)
        rng.set_seeds(seeds, ids=torch.arange(B, dtype=torch.int32))
        cfg = TrackGenConfig(device=device, num_envs=B, scale=scale, num_points=N,
                             max_regen_iters=20, relax_enable=False)
        cl = BezierCenterlineGenerator(cfg, rng).generate(torch.arange(B, device=device))
        res = inflation._resample_stage(cl, cfg)            # arc-length uniform [B,N,2]
        ok = torch.isfinite(res.center).all(dim=(1, 2)) & cl.valid
        for e in torch.where(ok)[0].tolist():
            kept.append(res.center[e])
            if len(kept) >= E:
                break
        s += B
    return torch.stack(kept[:E], dim=0).to(device)


def _quality(center0, relaxed, half_width):
    band = relaxation._band(center0, _Cfg(half_width=half_width))
    th = geometry.thickness(relaxed, band)
    target = 0.98 * half_width
    # Inflate at constant width to count border crossings (orientation is irrelevant
    # for crossing counts, so the plain +/- Nrm offset suffices).
    _, Nrm = geometry.tangents_normals(relaxed)
    outer = relaxed + half_width * Nrm
    inner = relaxed - half_width * Nrm
    border_x = geometry.self_intersections(outer) + geometry.self_intersections(inner)
    center_x = geometry.self_intersections(relaxed)
    valid = (th >= target) & (border_x == 0) & (center_x == 0)
    disp = torch.linalg.norm(relaxed - center0, dim=-1).mean(dim=1)
    kappa = geometry.menger_curvature(relaxed)
    crad = 1.0 / kappa.clamp_min(1e-12)
    hc = torch.minimum(crad, 0.5 * geometry.separation_min(relaxed, band).unsqueeze(1))
    cv = (hc.std(dim=1) / hc.mean(dim=1).clamp_min(1e-12))
    return {
        "valid_frac": valid.float().mean().item(),
        "thickness_med": th.median().item(),
        "disp_med": disp.median().item(),
        "clearance_cv_med": cv.median().item(),
        "kmax_med": kappa.amax(dim=1).median().item(),
    }


class _Cfg:
    """Minimal stand-in exposing the fields _band reads."""
    def __init__(self, half_width, relax_band=None):
        self.half_width = half_width
        self.relax_band = relax_band


def run_benchmark(E=8192, N=256, half_width=0.03, scale=1.0, device="cuda",
                  solvers=("xpbd", "energy", "tp_sobolev"), chunk=None,
                  relax_iters=150, energy_steps=800, tp_iters=100, seed=20, smooth=False):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    center0 = _gen_simple_tracks(E, N, scale, device, seed)
    rows = []
    for solver in solvers:
        cfg = TrackGenConfig(device=device, num_envs=E, num_points=N, half_width=half_width,
                             relax_solver=solver, relax_chunk_size=chunk, relax_iters=relax_iters,
                             energy_steps=energy_steps, tp_iters=tp_iters, smooth_finish=smooth)
        if device == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        relaxed = relaxation.relax(center0, cfg)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else float("nan")
        q = _quality(center0, relaxed, half_width)
        rows.append({"solver": solver, "device": device, "E": E, "N": N,
                     "seconds": seconds, "peak_gpu_mb": peak_mb, **q})
    return rows


def _print_table(rows):
    cols = ["solver", "device", "valid_frac", "seconds", "peak_gpu_mb",
            "thickness_med", "disp_med", "clearance_cv_med", "kmax_med"]
    print("  ".join(f"{c:>14}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>14.4g}" if isinstance(r[c], float) else f"{str(r[c]):>14}" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=8192)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--half_width", type=float, default=0.03)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smooth", action="store_true")
    a = ap.parse_args()
    rows = run_benchmark(E=a.E, N=a.N, half_width=a.half_width, chunk=a.chunk,
                         device="cpu" if a.cpu else "cuda", smooth=a.smooth)
    _print_table(rows)
```

Note: simplify `_quality`'s offset to just the explicit `outer/inner` lines (delete the dead `if False` expression — it is shown here only to mark that `inflation._offset_stage` orientation is not needed for crossing counts; the constant ±Nrm offset suffices). Final `_quality` must contain only the `outer = ...`/`inner = ...` form.

- [ ] **Step 4: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_benchmark_smoke.py -q`
Expected: passed.

- [ ] **Step 5: Run the real benchmark on CPU at reduced E (sanity) and record output**

Run: `.venv/bin/python -m benchmarks.benchmark_relaxation --E 256 --cpu`
Expected: a 3-row table; `xpbd` valid_frac highest (≈1.0), `energy`/`tp_sobolev` lower, matching the bake-off. (On a CUDA box, run without `--cpu` at `--E 8192` and a `--chunk` sweep to record peak memory.)

- [ ] **Step 6: Commit**

```bash
git add benchmarks/ tests/test_benchmark_smoke.py
git commit -m "benchmarks: relaxation backend benchmark (validity/time/peak-mem/quality, GPU+CPU)"
```

---

## Phase 7 — Public API + final regression

### Task 14: Export the new surface; full-suite regression; integration gate

**Files:**
- Modify: `track_gen/__init__.py`
- Test: `tests/test_public_api.py`, `tests/test_public_api_full.py`, `tests/test_integration.py` (update)

- [ ] **Step 1: Update the public-API test** — add to `tests/test_public_api.py`:

```python
def test_relaxation_surface_exported():
    import track_gen
    assert hasattr(track_gen, "relax")
    from track_gen import relaxation  # module importable
    from track_gen.geometry import thickness, self_intersections, separation_min
    assert callable(track_gen.relax)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_public_api.py::test_relaxation_surface_exported -q`
Expected: FAIL — `track_gen.relax` not exported.

- [ ] **Step 3: Update `track_gen/__init__.py`** — add the relaxation surface to the imports and `__all__`:

```python
from . import relaxation
from .relaxation import relax
from .geometry import thickness, self_intersections, separation_min, curvature_radius_min
```

Add `"relaxation"`, `"relax"`, `"thickness"`, `"self_intersections"`, `"separation_min"`, `"curvature_radius_min"` to `__all__`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_public_api.py tests/test_public_api_full.py -q`
Expected: passed. (If `test_public_api_full.py` asserts an exact `__all__` set, update it to include the new names.)

- [ ] **Step 5: Full-suite regression**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass. Triage any failure: most likely a leftover reference to a removed config field or the variable-width assumption — fix the test to the constant-width/relaxed contract.

- [ ] **Step 6: Integration sanity — import + run from repo root, no stdlib shadow**

Run: `cd /home/antoiner/Documents/TrackGen && .venv/bin/python -c "import types as t; assert hasattr(t,'MappingProxyType'); import track_gen; print(sorted(n for n in ('relax','thickness','self_intersections') if hasattr(track_gen,n)))"`
Expected: prints `['relax', 'self_intersections', 'thickness']`.

- [ ] **Step 7: Commit**

```bash
git add track_gen/__init__.py tests/test_public_api.py tests/test_public_api_full.py tests/test_integration.py
git commit -m "Export relaxation public API; full-suite regression green"
```

---

## Self-Review (completed during planning)

**Spec coverage:** thickness reframe → Tasks 2–3 (geometry) + 8 (validity); three selectable backends → Tasks 4 (xpbd+dispatch), 10 (energy), 11 (tp_sobolev); finisher → Task 12; env-chunking → Task 5; benchmark (validity/time/peak-mem/quality, GPU+CPU, E=8192) → Task 13; generator simplicity gate → Task 7; constant-width + real validity gate → Task 8; facade wiring → Task 9; packaging (pyproject + track_gen/ + stdlib-shadow fix) → Task 1; public API → Task 14. All spec sections map to a task.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; the one `if False` expression in Task 13 is explicitly flagged for deletion with the final form specified.

**Type/name consistency:** backend signature `_relax_<name>(center0, band, config)`; dispatcher `relax(center, config)`; helper `_band(center, config)`; `_tp_flow(center0, band, config, n_steps, tau, early_stop)` reused by backend + finisher; config field names match between `types.py` (Task 6) and every reader (`relaxation.py`, `inflation.py`, benchmark). `geometry` helpers (`thickness`, `separation_min`, `circ_index_dist`, `perimeter`, `mean_seg_len`, `curvature_radius_min`, `self_intersections`) defined in Tasks 2–3 and used consistently downstream.
