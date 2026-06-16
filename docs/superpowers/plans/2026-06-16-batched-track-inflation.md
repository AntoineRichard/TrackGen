# Batched Track Inflation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ("- [ ]") syntax for tracking.

**Goal:** Replace the legacy recursive track generator with a batched, per-env-seeded pipeline that turns dense centerlines (Bezier or Fourier) into index-aligned outer/center/inner track boundaries with curvature-clamped width, robust validity gating, and reproducible GPU-resident sampling.

**Architecture:** A FLAT `track_gen/` package with a six-file split — `geometry.py` (pure torch primitives), `types.py` (TrackGenConfig + Track leaf), `generators.py` (Bezier + Fourier behind one CenterlineGenerator interface), `inflation.py` (shared resample/width/offset stage), `track_generator.py` (facade + compat shim) — alongside the untouched `rng_kernels.py` / `rng_utils.py`. The package IS the existing directory `/home/antoine/Documents/track_gen/` (no nesting, no file moves); new modules are created directly beside the existing files and use relative imports. The leaf `types.py` carries the warp-free dataclasses so `inflation.py` and the facade can share `Track`/`TrackGenConfig` without a circular import, and CPU-only tests never drag in NVIDIA Warp.

**Tech Stack:** Python, PyTorch, NVIDIA Warp (per-env seeded RNG), scipy, pytest

---

### Task 1: Project + test scaffolding (git bootstrap, flat package)

This task captures the pre-existing files in git (the directory is not yet a repo), establishes the test directory with a `conftest.py` that puts the repo's parent on `sys.path` so `import track_gen` resolves, creates `__init__.py` re-exporting the public API, and adds the (initially empty) `geometry.py`. Because the codebase already ships top-level modules (`track_generator.py`, `rng_utils.py`, `rng_kernels.py`) that use intra-package relative imports (e.g. `from . import PerEnvSeededRNG`, `from .rng_kernels import ...`), the package IS the existing directory — new modules are created FLAT beside the existing files, import each other relatively (`from .geometry import ...`), and tests import absolutely (`from track_gen.geometry import ...`).

Throughout, `E` is the number of environments (the batch dimension) and shapes are written in `[brackets]`. Geometry functions are pure (no side effects, no global state) and device-agnostic — they run on CPU with PyTorch alone, no NVIDIA Warp, no GPU.

Before starting, confirm PyTorch and pytest are importable:

```bash
cd /home/antoine/Documents/track_gen
python -c "import torch; import pytest; print('torch', torch.__version__, 'pytest', pytest.__version__)"
```

All `pytest` commands below run as `cd /home/antoine/Documents/track_gen && python -m pytest tests -v`. All `git` commands run from `/home/antoine/Documents/track_gen` with relative paths.

**Files:**
- Create: `/home/antoine/Documents/track_gen/__init__.py`
- Create: `/home/antoine/Documents/track_gen/tests/conftest.py`
- Create: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_scaffolding.py`

- [ ] **Step 1 — Bootstrap git, then write the FAILING test.** First initialize the repo and capture the pre-existing files so every later task can commit (the `git init` is idempotent — skip it if `.git` already exists):

```bash
cd /home/antoine/Documents/track_gen
git init
git add -A
git commit -m "baseline: existing RNG and track generator

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Then create `/home/antoine/Documents/track_gen/tests/conftest.py` with EXACTLY this content (puts the directory that CONTAINS `track_gen/` on `sys.path` so `import track_gen` resolves to this flat package):

```python
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
```

Create `/home/antoine/Documents/track_gen/tests/test_scaffolding.py`:

```python
"""Scaffolding tests: prove the package and its test dir import cleanly."""


def test_package_imports():
    import track_gen

    assert hasattr(track_gen, "__version__")
    assert isinstance(track_gen.__version__, str)


def test_geometry_module_imports():
    from track_gen import geometry

    assert geometry is not None
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'track_gen'` (the package `__init__.py` and `geometry.py` do not exist yet), so both tests error out.

- [ ] **Step 3 — Write the minimal implementation.** Create `/home/antoine/Documents/track_gen/__init__.py`:

```python
"""track_gen — GPU-batched race-track generator.

Public API is grown incrementally as modules land. Geometry primitives and the
public dataclasses / generators are re-exported here for convenience once they
exist.
"""

__version__ = "0.1.0"

from .rng_utils import PerEnvSeededRNG  # noqa: F401
from . import geometry  # noqa: F401

__all__ = ["PerEnvSeededRNG", "geometry"]
```

Create `/home/antoine/Documents/track_gen/geometry.py` with only the module docstring and the torch import for now (primitives are added in later tasks):

```python
"""Pure batched-torch geometry primitives.

Device-agnostic and dependency-light: torch only (NO warp import), so the whole
module is unit-testable on CPU. Batch dimension is E (num_envs); shapes are
documented per function in [brackets].
"""

import torch  # noqa: F401
```

Note: re-exporting `PerEnvSeededRNG` here means importing `track_gen` pulls in `rng_utils` (which imports warp). That is fine for the scaffolding test (warp present in the dev env) and for the facade. CPU-only inflation/geometry tests import the leaf modules directly (`from track_gen.geometry import ...`, `from track_gen.types import ...`), never the package root, so they do not drag in warp — see the inflation tasks.

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests -v
```

Expected: `2 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add __init__.py tests/conftest.py geometry.py tests/test_scaffolding.py
git commit -m "Add track_gen package + test scaffolding

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `safe_normalize`

`safe_normalize(v, eps=1e-8)` returns unit vectors along the last axis of `v`, preserving the input shape. The key robustness property: a zero (or near-zero) vector must produce a **finite** result (we return the zero vector itself, not NaN/Inf). We achieve this by dividing by the norm clamped to a minimum of `eps`.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_safe_normalize.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_safe_normalize.py`:

```python
import torch

from track_gen.geometry import safe_normalize


def test_unit_vectors_have_norm_one():
    v = torch.tensor([[[3.0, 4.0], [0.0, 2.0], [-5.0, 0.0]]])  # [1, 3, 2]
    out = safe_normalize(v)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    assert torch.allclose(out[0, 0], torch.tensor([0.6, 0.8]), atol=1e-6)


def test_zero_vector_stays_finite_and_zero():
    v = torch.zeros((1, 1, 2))
    out = safe_normalize(v)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_shape_is_preserved():
    v = torch.randn((4, 7, 2))
    out = safe_normalize(v)
    assert out.shape == v.shape
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_safe_normalize.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'safe_normalize' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py`:

```python
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
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_safe_normalize.py -v
```

Expected: `3 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_safe_normalize.py
git commit -m "Add geometry.safe_normalize

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `polygon_area`

`polygon_area(points[E,P,2]) -> [E]` returns the **signed** shoelace area of each polygon in the batch. Sign convention: counter-clockwise (CCW) ordering gives positive area, clockwise (CW) gives negative. The shoelace formula is `0.5 * Σ_i (x_i * y_{i+1} − x_{i+1} * y_i)` with the index wrapping (the last vertex connects back to the first). We compute the wrap with `torch.roll(points, shifts=-1, dims=1)`.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_polygon_area.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_polygon_area.py`:

```python
import torch

from track_gen.geometry import polygon_area


def test_unit_square_ccw_is_plus_one():
    sq = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    area = polygon_area(sq)
    assert area.shape == (1,)
    assert torch.allclose(area, torch.tensor([1.0]), atol=1e-6)


def test_unit_square_cw_is_minus_one():
    sq = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]])
    area = polygon_area(sq)
    assert torch.allclose(area, torch.tensor([-1.0]), atol=1e-6)


def test_batched_mixed_orientation():
    ccw = [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]]  # area +6
    cw = [[0.0, 0.0], [0.0, 3.0], [2.0, 3.0], [2.0, 0.0]]  # area -6
    pts = torch.tensor([ccw, cw])
    area = polygon_area(pts)
    assert torch.allclose(area, torch.tensor([6.0, -6.0]), atol=1e-6)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_polygon_area.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'polygon_area' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py`:

```python
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
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_polygon_area.py -v
```

Expected: `3 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_polygon_area.py
git commit -m "Add geometry.polygon_area (signed shoelace)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `ccw_sort`

`ccw_sort(points[E,P,2]) -> points[E,P,2]` reorders each env's vertices by their angle around the batch centroid, producing a non-self-intersecting (simple) polygon ordering. This is ported directly from the existing `TrackGenerator.ccw_sort` in `track_generator.py` — **preserve its exact behavior**, including its specific (and slightly unconventional) `arctan2(dx, dy)` argument order, so generated tracks remain consistent with the original lineage. We keep the batch dimension and add no host syncs.

Note on the test: because the original uses `atan2(dx, dy)` rather than `atan2(dy, dx)`, do not over-specify the exact starting vertex or direction. Instead assert the load-bearing property the design relies on: after sorting, the polygon is simple, hence `|polygon_area| > 0` and equals the area of the convex test shape.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_ccw_sort.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_ccw_sort.py`:

```python
import torch

from track_gen.geometry import ccw_sort, polygon_area


def test_scramble_is_reordered_to_a_simple_polygon():
    scrambled = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    out = ccw_sort(scrambled)
    assert out.shape == scrambled.shape
    assert torch.isclose(polygon_area(out).abs(), torch.tensor([1.0]), atol=1e-6)


def test_sorted_output_has_monotone_angles_around_centroid():
    pts = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, -1.0]]]
    )
    out = ccw_sort(pts)
    centroid = out.mean(dim=1, keepdim=True)
    d = out - centroid
    ang = torch.atan2(d[..., 0], d[..., 1])  # reproduce the ported convention
    diffs = ang[0, 1:] - ang[0, :-1]
    assert (diffs >= -1e-6).all()


def test_output_is_a_permutation_of_the_input():
    pts = torch.tensor([[[0.3, 0.9], [-0.5, 0.2], [0.7, -0.4], [0.1, 0.6]]])
    out = ccw_sort(pts)
    assert torch.allclose(out.sum(dim=1), pts.sum(dim=1), atol=1e-6)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_ccw_sort.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'ccw_sort' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py` (ported verbatim from `track_generator.py`, keeping the `arctan2(dx, dy)` argument order):

```python
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
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_ccw_sort.py -v
```

Expected: `3 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_ccw_sort.py
git commit -m "Port geometry.ccw_sort from track_generator (batched)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `segment_directions`

`segment_directions(points[E,P,2], closed=True) -> dirs[E,P,2]` returns the unit direction vector of edge `i -> i+1` for each vertex `i`. When `closed=True`, the last edge wraps from the final vertex back to the first (so `dirs` has the same `P` length as `points`). We form `points_next = roll(points, -1)` minus `points`, then `safe_normalize`. When `closed=False`, the final slot (which would be the wrap edge) is set to zero so it carries no spurious direction.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_segment_directions.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_segment_directions.py`:

```python
import torch

from track_gen.geometry import segment_directions


def test_unit_square_edges_are_axis_aligned_unit_dirs():
    sq = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    dirs = segment_directions(sq, closed=True)
    assert dirs.shape == sq.shape
    expected = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]
    )
    assert torch.allclose(dirs, expected, atol=1e-6)
    norms = torch.linalg.norm(dirs, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_open_chain_last_dir_is_zero():
    pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    dirs = segment_directions(pts, closed=False)
    assert torch.allclose(dirs[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(dirs[0, 1], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(dirs[0, 2], torch.tensor([0.0, 0.0]), atol=1e-6)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_segment_directions.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'segment_directions' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py`:

```python
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
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_segment_directions.py -v
```

Expected: `2 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_segment_directions.py
git commit -m "Add geometry.segment_directions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `vertex_tangents`

`vertex_tangents(points[E,P,2], p) -> tangents[E,P,2]` computes, at each vertex `i`, a blended unit tangent from the two incident edge directions. `u_out` is the direction of edge `i -> i+1`; `u_in` is the direction of the previous edge `i-1 -> i`. The tangent is `safe_normalize(p * u_out + (1 - p) * u_in)`. This is the vector-space replacement for the old atan2-angle blend and its `+π` wraparound hacks — there is no angle arithmetic, so no wraparound discontinuity. `u_out = segment_directions(points)`; `u_in = roll(u_out, +1)` (the previous vertex's out-edge is this vertex's in-edge).

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_vertex_tangents.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_vertex_tangents.py`:

```python
import math

import torch

from track_gen.geometry import vertex_tangents


def _regular_polygon(n: int, radius: float = 1.0) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)  # [1, n, 2]


def test_regular_polygon_tangents_are_unit_length():
    pts = _regular_polygon(6)
    t = vertex_tangents(pts, p=0.5)
    assert t.shape == pts.shape
    norms = torch.linalg.norm(t, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_p_half_is_symmetric_between_in_and_out_edges():
    pts = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float64
    )
    t = vertex_tangents(pts, p=0.5)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert torch.allclose(
        t[0, 1], torch.tensor([inv_sqrt2, inv_sqrt2], dtype=torch.float64), atol=1e-6
    )


def test_p_extremes_recover_pure_edge_directions():
    pts = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float64
    )
    t_out = vertex_tangents(pts, p=1.0)  # pure out-edge at vertex 1 -> +y
    t_in = vertex_tangents(pts, p=0.0)  # pure in-edge at vertex 1 -> +x
    assert torch.allclose(t_out[0, 1], torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(t_in[0, 1], torch.tensor([1.0, 0.0], dtype=torch.float64), atol=1e-6)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_vertex_tangents.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'vertex_tangents' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py`:

```python
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
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_vertex_tangents.py -v
```

Expected: `3 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_vertex_tangents.py
git commit -m "Add geometry.vertex_tangents (vector-space blend)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `turning_number`

`turning_number(points[E,P,2]) -> [E]` returns the signed total turning of the closed polygon, in radians. For a simple loop this is `+2π` (CCW) or `−2π` (CW); for a figure-eight (one CW lobe + one CCW lobe) the turns cancel to `~0`. This is the cheap O(P) self-intersection gate used by the generators.

Algorithm: take edge directions `dirs = segment_directions(points)`, their angles `theta = atan2(dir_y, dir_x)`, then the per-vertex turn `dtheta = theta - roll(theta, +1)` **wrapped into (−π, π]**, and sum over the loop. Wrapping is `atan2(sin(dtheta), cos(dtheta))`.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_turning_number.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_turning_number.py`:

```python
import math

import torch

from track_gen.geometry import turning_number


def _regular_polygon(n: int, radius: float = 1.0) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_convex_ccw_polygon_is_plus_two_pi():
    pts = _regular_polygon(8)  # CCW by construction (increasing angle)
    tn = turning_number(pts)
    assert tn.shape == (1,)
    assert torch.isclose(tn, torch.tensor([2.0 * math.pi], dtype=torch.float64), atol=1e-4)


def test_convex_cw_polygon_is_minus_two_pi():
    pts = _regular_polygon(8).flip(dims=[1])  # reverse -> clockwise
    tn = turning_number(pts)
    assert torch.isclose(tn, torch.tensor([-2.0 * math.pi], dtype=torch.float64), atol=1e-4)


def test_figure_eight_turns_cancel_to_zero():
    pts = torch.tensor(
        [[[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]], dtype=torch.float64
    )
    tn = turning_number(pts)
    assert torch.isclose(tn, torch.tensor([0.0], dtype=torch.float64), atol=1e-4)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_turning_number.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'turning_number' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py`:

```python
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
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_turning_number.py -v
```

Expected: `3 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_turning_number.py
git commit -m "Add geometry.turning_number (self-intersection gate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `menger_curvature`

`menger_curvature(points[E,N,2]) -> kappa[E,N]` returns the (non-negative) Menger curvature at each point, computed from the triple of points `(i-1, i, i+1)` on the closed loop. The Menger curvature of three points is `kappa = 4 * |triangle area| / (|a| * |b| * |c|)`, where `a, b, c` are the three side lengths. For a circle of radius `r` this tends to `1/r`; for collinear points the triangle area is zero so `kappa = 0`. We clamp the denominator with `eps` to avoid divide-by-zero on coincident points.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_menger_curvature.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_menger_curvature.py`:

```python
import math

import torch

from track_gen.geometry import menger_curvature


def _circle(n: int, radius: float) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_circle_curvature_is_one_over_r():
    r = 2.5
    pts = _circle(256, r)
    kappa = menger_curvature(pts)
    assert kappa.shape == (1, 256)
    expected = torch.full_like(kappa, 1.0 / r)
    assert torch.allclose(kappa, expected, atol=1e-3)
    assert (kappa >= 0).all()


def test_straight_line_curvature_is_zero():
    xs = torch.linspace(0.0, 9.0, 10, dtype=torch.float64)
    pts = torch.stack([xs, torch.zeros_like(xs)], dim=-1).unsqueeze(0)
    kappa = menger_curvature(pts)
    assert torch.allclose(kappa[0, 1:-1], torch.zeros(8, dtype=torch.float64), atol=1e-6)
    assert (kappa >= 0).all()


def test_coincident_points_do_not_produce_nan():
    pts = torch.zeros((1, 5, 2), dtype=torch.float64)
    kappa = menger_curvature(pts)
    assert torch.isfinite(kappa).all()
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_menger_curvature.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'menger_curvature' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py`:

```python
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
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_menger_curvature.py -v
```

Expected: `3 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_menger_curvature.py
git commit -m "Add geometry.menger_curvature

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `tangents_normals`

`tangents_normals(points[E,N,2]) -> (T[E,N,2], Nrm[E,N,2])` returns, for each point on the closed loop, a unit central-difference tangent `T` and its left-normal `Nrm`. The tangent at point `i` uses the central difference `points[i+1] - points[i-1]` (with index wrapping on the closed loop), then `safe_normalize`. The left-normal is the 90° CCW rotation of the tangent: `Nrm = stack(-T_y, T_x)`. By construction `‖T‖ = 1` and `T · Nrm = 0` everywhere.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_tangents_normals.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_tangents_normals.py`:

```python
import math

import torch

from track_gen.geometry import tangents_normals


def _circle(n: int, radius: float) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_tangent_is_unit_everywhere():
    pts = _circle(64, 3.0)
    T, Nrm = tangents_normals(pts)
    assert T.shape == pts.shape
    assert Nrm.shape == pts.shape
    norms = torch.linalg.norm(T, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_tangent_and_normal_are_orthogonal_everywhere():
    pts = _circle(64, 3.0)
    T, Nrm = tangents_normals(pts)
    dot = (T * Nrm).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-6)


def test_normal_is_unit_and_is_left_rotation_of_tangent():
    pts = _circle(32, 1.0)
    T, Nrm = tangents_normals(pts)
    nrm_norms = torch.linalg.norm(Nrm, dim=-1)
    assert torch.allclose(nrm_norms, torch.ones_like(nrm_norms), atol=1e-6)
    assert torch.allclose(Nrm[..., 0], -T[..., 1], atol=1e-6)
    assert torch.allclose(Nrm[..., 1], T[..., 0], atol=1e-6)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_tangents_normals.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'tangents_normals' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py`:

```python
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
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_tangents_normals.py -v
```

Expected: `3 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_tangents_normals.py
git commit -m "Add geometry.tangents_normals (central diff + left normal)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: `arc_length_resample` (masked, fixed + constant-spacing, N_max-padded, R<2 guard)

`arc_length_resample(points[E,M,2], num=None, spacing=None, valid_mask=None, n_max=None) -> (resampled[E,N,2], count[E])` re-samples each env's **closed** loop at arc-length-uniform positions. This is the core of fixed-`N`, index-aligned output.

Per env:
1. Drop invalid points — a point is invalid if `valid_mask` is `False` for it (when `valid_mask` is given) or if either coordinate is NaN. Only the *real* (kept) points define the loop. We operate per-env in a loop (E is the batch of independent tracks; this loop is over envs, not over points, and is fine for the test scales).
2. **R<2 guard (finding):** if an env has fewer than 2 real points (e.g. an unconverged, all-NaN generator output), emit a NaN-filled row of the target width and `count 0` instead of indexing into an empty tensor. This keeps a batch that mixes valid and all-NaN envs from crashing the whole pipeline with an `IndexError`.
3. Build cumulative arc length along the closed loop: lengths of consecutive real-point segments **including the closing wrap segment** from the last real point back to the first. `s` starts at 0; `L` is the total perimeter.
4. Choose targets:
   - `num` given → `N = num`, `count = num`, targets `= arange(num) * L / num`.
   - `spacing` given → targets `= arange(0, L, spacing)`; the per-env real count is `len(targets)`. **N_max threading (finding):** when `spacing` is used, the batch output is padded to `n_max` (passed by the caller as `config.N_max`); `max(counts)` must be `<= n_max` (asserted). When `n_max` is `None` in spacing mode the function falls back to batch-max, but the inflation caller always passes `n_max=config.N_max`.
5. For each target distance, `searchsorted` into `s` to find the bracketing segment, then linearly interpolate between the two bracketing real points.

Exactly one of `num` / `spacing` must be provided.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_arc_length_resample.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_arc_length_resample.py`:

```python
import math

import torch

from track_gen.geometry import arc_length_resample


def _circle(n: int, radius: float) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_circle_resample_is_arc_uniform_and_on_the_circle():
    r = 2.0
    pts = _circle(37, r)  # uneven count to force genuine interpolation
    out, count = arc_length_resample(pts, num=120)
    assert out.shape == (1, 120, 2)
    assert count.shape == (1,)
    assert int(count[0]) == 120
    radii = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(radii, torch.full_like(radii, r), atol=1e-2)
    step = torch.linalg.norm(out[:, 1:] - out[:, :-1], dim=-1)
    assert (step.std() / step.mean()) < 1e-2


def test_nan_padded_input_is_handled():
    real = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    nan_pad = torch.full((3, 2), float("nan"), dtype=torch.float64)
    pts = torch.cat([real, nan_pad], dim=0).unsqueeze(0)  # [1, 7, 2]
    out, count = arc_length_resample(pts, num=40)
    assert out.shape == (1, 40, 2)
    assert torch.isfinite(out).all()
    assert int(count[0]) == 40
    assert (out >= -1e-6).all() and (out <= 1.0 + 1e-6).all()


def test_constant_spacing_pads_to_n_max_with_nan():
    # Two circles of different perimeter -> different real counts -> padding to n_max.
    small = _circle(60, 1.0)
    big = _circle(60, 3.0)
    pts = torch.cat([small, big], dim=0)  # [2, 60, 2]
    out, count = arc_length_resample(pts, spacing=0.25, n_max=128)
    assert out.shape == (2, 128, 2)  # padded to n_max, not batch-max
    assert int(count[0]) < int(count[1])
    assert int(count[0]) <= 128 and int(count[1]) <= 128
    c0 = int(count[0])
    if c0 < 128:
        assert torch.isnan(out[0, c0:]).all()
    real0 = out[0, :c0]
    radii0 = torch.linalg.norm(real0, dim=-1)
    assert torch.allclose(radii0, torch.ones_like(radii0), atol=2e-2)


def test_fewer_than_two_real_points_yields_nan_row_count_zero():
    # One valid env, one all-NaN env: the all-NaN env must not crash, returns NaN row + count 0.
    good = _circle(40, 1.0)[0]  # [40, 2]
    bad = torch.full((40, 2), float("nan"), dtype=torch.float64)
    pts = torch.stack([good, bad], dim=0)  # [2, 40, 2]
    out, count = arc_length_resample(pts, num=32)
    assert out.shape == (2, 32, 2)
    assert int(count[0]) == 32
    assert int(count[1]) == 0  # all-NaN env: zero real points
    assert torch.isfinite(out[0]).all()
    assert torch.isnan(out[1]).all()
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_arc_length_resample.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'arc_length_resample' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py`:

```python
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
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_arc_length_resample.py -v
```

Expected: `4 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_arc_length_resample.py
git commit -m "Add geometry.arc_length_resample (masked, fixed + constant-spacing, n_max, R<2 guard)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: `nearest_nonadjacent_distance`

`nearest_nonadjacent_distance(points[E,N,2], band, decimation=None) -> d[E,N]` returns, for each point `i`, the minimum Euclidean distance to any **non-adjacent** point — where "adjacent" means within `±band` index positions on the closed loop (with wraparound). This drives the optional self-distance width clamp. Adjacent points (including `i` itself, since `band ≥ 0`) are masked out by setting their pairwise distance to `+inf` before taking the per-point min.

Algorithm:
1. Optionally **decimate** to `decimation` evenly-spaced points along the loop, compute the distance there, then map `d` back to `N` indices (cheaper for large `N`). When `decimation is None`, work at full resolution.
2. Compute the full pairwise distance matrix with `torch.cdist` → `[E, P, P]`.
3. Build a boolean mask of the `±band` index window **with wraparound**: circular index distance `min(|i−j|, P − |i−j|)`; mask where `≤ band`. Set masked entries to `+inf`.
4. Take `min` over the last axis → `[E, P]`. If decimated, map back to `N`.

Note on the decimation test tolerance (finding): decimation is an intentional approximation; the decimated estimate systematically overestimates (larger chords) so the cross-check tolerance is `atol=0.2`, not `0.1`.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/geometry.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_geometry_nearest_nonadjacent.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_geometry_nearest_nonadjacent.py`:

```python
import math

import torch

from track_gen.geometry import nearest_nonadjacent_distance


def _circle(n: int, radius: float) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_circle_min_nonadjacent_distance_is_positive():
    pts = _circle(64, 1.0)
    d = nearest_nonadjacent_distance(pts, band=2)
    assert d.shape == (1, 64)
    assert torch.isfinite(d).all()
    assert (d > 0).all()


def test_immediate_neighbors_are_masked_not_self():
    pts = _circle(32, 1.0)
    d_band1 = nearest_nonadjacent_distance(pts, band=1)
    two_step = 2.0 * 1.0 * math.sin(2.0 * math.pi / 32 * 2 / 2)
    assert torch.allclose(d_band1, torch.full_like(d_band1, two_step), atol=1e-3)
    assert (d_band1 > 1e-6).all()


def test_decimation_path_runs_and_matches_full_resolution_shape():
    pts = _circle(80, 2.0)
    d_full = nearest_nonadjacent_distance(pts, band=2)
    d_dec = nearest_nonadjacent_distance(pts, band=2, decimation=40)
    assert d_dec.shape == d_full.shape == (1, 80)
    assert torch.isfinite(d_dec).all()
    assert (d_dec > 0).all()
    # Decimation is an intentional approximation: ballpark agreement only.
    assert torch.allclose(d_dec, d_full, atol=0.2)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_nearest_nonadjacent.py -v
```

Expected: FAIL at import — `ImportError: cannot import name 'nearest_nonadjacent_distance' from 'track_gen.geometry'`.

- [ ] **Step 3 — Write the minimal implementation.** Append to `/home/antoine/Documents/track_gen/geometry.py`:

```python
def _circular_band_mask(p: int, band: int, device, dtype) -> torch.Tensor:
    """Boolean [P, P] mask: True where circular index distance <= band."""
    idx = torch.arange(p, device=device)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    circ = torch.minimum(diff, p - diff)
    return circ <= band


def nearest_nonadjacent_distance(
    points: torch.Tensor,
    band: int,
    decimation: int | None = None,
) -> torch.Tensor:
    """Min distance from each point to any non-adjacent point on a closed loop.

    "Adjacent" = within +/-band index positions on the loop (with wraparound),
    which also excludes the point itself. Masked pairs are set to +inf before
    the per-point min. Optionally decimate to `decimation` evenly-spaced points
    for speed, then map the result back to N indices.

    Args:
        points: Tensor [E, N, 2], closed loop.
        band: Half-width (in indices) of the excluded neighbor window.
        decimation: If given, compute on this many evenly-spaced points and map
            back to N.

    Returns:
        Tensor [E, N], min non-adjacent distance per point.
    """
    E, N, _ = points.shape
    device = points.device
    dtype = points.dtype

    if decimation is not None and decimation < N:
        sel = torch.linspace(0, N - 1, decimation, device=device).round().long()
        work = points[:, sel, :]  # [E, P, 2]
        dec_band = max(1, int(round(band * decimation / N)))
    else:
        sel = None
        work = points
        dec_band = band

    P = work.shape[1]
    dmat = torch.cdist(work, work)  # [E, P, P]
    mask = _circular_band_mask(P, dec_band, device, dtype)  # [P, P]
    dmat = dmat.masked_fill(mask.unsqueeze(0), float("inf"))
    d_work = dmat.min(dim=-1).values  # [E, P]

    if sel is None:
        return d_work

    full_idx = torch.arange(N, device=device)
    nearest = torch.bucketize(full_idx, sel.clamp(max=N - 1))
    nearest = nearest.clamp(max=P - 1)
    return d_work[:, nearest]
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_geometry_nearest_nonadjacent.py -v
```

Expected: `3 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add geometry.py tests/test_geometry_nearest_nonadjacent.py
git commit -m "Add geometry.nearest_nonadjacent_distance (circular band mask)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Re-export geometry primitives + full geometry-suite green check

Grow the public API in `__init__.py` to re-export all geometry primitives (so callers can `from track_gen import safe_normalize`), and confirm the entire geometry test suite passes together.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/__init__.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_public_api.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_public_api.py`:

```python
import track_gen


def test_geometry_primitives_are_reexported():
    names = [
        "safe_normalize",
        "polygon_area",
        "ccw_sort",
        "segment_directions",
        "vertex_tangents",
        "turning_number",
        "menger_curvature",
        "tangents_normals",
        "arc_length_resample",
        "nearest_nonadjacent_distance",
    ]
    for name in names:
        assert hasattr(track_gen, name), f"track_gen.{name} is not exported"
        assert callable(getattr(track_gen, name))
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_public_api.py -v
```

Expected: FAIL — `AssertionError: track_gen.safe_normalize is not exported`.

- [ ] **Step 3 — Write the minimal implementation.** Replace the body of `/home/antoine/Documents/track_gen/__init__.py` with:

```python
"""track_gen — GPU-batched race-track generator.

Public API is grown incrementally as modules land. Geometry primitives are
re-exported here for convenience.
"""

__version__ = "0.1.0"

from .rng_utils import PerEnvSeededRNG
from . import geometry
from .geometry import (
    arc_length_resample,
    ccw_sort,
    menger_curvature,
    nearest_nonadjacent_distance,
    polygon_area,
    safe_normalize,
    segment_directions,
    tangents_normals,
    turning_number,
    vertex_tangents,
)

__all__ = [
    "PerEnvSeededRNG",
    "geometry",
    "safe_normalize",
    "polygon_area",
    "ccw_sort",
    "segment_directions",
    "vertex_tangents",
    "turning_number",
    "menger_curvature",
    "tangents_normals",
    "arc_length_resample",
    "nearest_nonadjacent_distance",
]
```

- [ ] **Step 4 — Run it, expect PASS.** First the targeted test, then the entire suite so far:

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_public_api.py -v
python -m pytest tests -v
```

Expected: `test_public_api.py` shows `1 passed`; the full run reports all geometry tests passing (`test_geometry_*.py`, `test_scaffolding.py`, `test_public_api.py`) with `0 failed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add __init__.py tests/test_public_api.py
git commit -m "Re-export geometry primitives from track_gen public API

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: `types.py` leaf module — `TrackGenConfig` + `Track` dataclasses

This is the dependency-free LEAF that breaks the import cycle. `types.py` imports nothing from the package (only `math`, `dataclasses`, `torch`), so `inflation.py` and the facade both import `Track`/`TrackGenConfig` from here without triggering a circular import, and CPU-only inflation tests can import them without dragging in warp. `TrackGenConfig` carries every field from spec section 3.2 with sensible defaults so `TrackGenConfig()` instantiates with zero arguments. **Field names are reconciled to the single shared contract** consumed at runtime by generators and inflation (finding): the Fourier decay exponent is `decay_p` (not `p`), there is a `num_centerline_samples` (Fourier sample count) and `w_floor` (validity width floor).

**Files:**
- Create: `/home/antoine/Documents/track_gen/types.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_types.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_types.py`:

```python
import math

import torch

from track_gen.types import Track, TrackGenConfig


def test_config_defaults_instantiate():
    cfg = TrackGenConfig()
    # generator + batching
    assert cfg.generator == "bezier"
    assert cfg.num_envs == 1
    assert cfg.device == "cpu"
    # Bezier params
    assert cfg.min_num_points == 9
    assert cfg.max_num_points == 13
    assert cfg.num_points_per_segment == 30
    assert cfg.min_point_distance == 0.05
    assert math.isclose(cfg.min_angle, (12.5 / 180) * math.pi)
    assert cfg.rad == 0.2
    assert cfg.edgy == 0.0
    assert cfg.scale == 1.0
    # Fourier params (reconciled names)
    assert cfg.num_harmonics == 5
    assert cfg.decay_p == 2
    assert cfg.amplitude == 1.0
    assert cfg.num_centerline_samples == 256
    # Width params
    assert cfg.half_width == 0.1
    assert math.isclose(cfg.alpha, 0.9)
    assert cfg.clamp_self_distance is False
    assert cfg.self_distance_margin == 0.0
    assert cfg.self_distance_band == 8
    assert cfg.self_distance_decimation == 64
    # Output params
    assert cfg.num_points == 256
    assert cfg.output_mode == "fixed"
    assert cfg.spacing == 0.1
    assert cfg.N_max == 256
    # Robustness params
    assert cfg.max_regen_iters == 10
    assert cfg.turning_tol == 0.1
    assert cfg.w_floor == 1e-3


def test_config_overrides_round_trip():
    cfg = TrackGenConfig(
        generator="fourier",
        num_envs=32,
        num_points=128,
        half_width=0.25,
        alpha=0.8,
        clamp_self_distance=True,
        output_mode="constant_spacing",
        spacing=0.05,
        N_max=512,
        decay_p=3,
        num_centerline_samples=512,
        w_floor=1e-2,
    )
    assert cfg.generator == "fourier"
    assert cfg.num_envs == 32
    assert cfg.num_points == 128
    assert cfg.clamp_self_distance is True
    assert cfg.output_mode == "constant_spacing"
    assert cfg.spacing == 0.05
    assert cfg.N_max == 512
    assert cfg.decay_p == 3
    assert cfg.num_centerline_samples == 512
    assert cfg.w_floor == 1e-2


def test_track_construct_from_tensors_field_shapes():
    E, N = 4, 16
    track = Track(
        outer=torch.zeros(E, N, 2),
        center=torch.zeros(E, N, 2),
        inner=torch.zeros(E, N, 2),
        tangent=torch.zeros(E, N, 2),
        normal=torch.zeros(E, N, 2),
        arclen=torch.zeros(E, N),
        length=torch.zeros(E),
        valid=torch.ones(E, dtype=torch.bool),
        count=torch.full((E,), N, dtype=torch.long),
    )
    for arr in (track.outer, track.center, track.inner, track.tangent, track.normal):
        assert arr.shape == (E, N, 2)
    assert track.arclen.shape == (E, N)
    assert track.length.shape == (E,)
    assert track.valid.shape == (E,)
    assert track.valid.dtype == torch.bool
    assert track.count.shape == (E,)


def test_types_module_has_no_intra_package_imports():
    # The leaf must not import generators/inflation/track_generator/rng_utils.
    import track_gen.types as t

    src = open(t.__file__).read()
    for forbidden in ("from .generators", "from .inflation", "from .track_generator", "from .rng_utils", "import warp"):
        assert forbidden not in src, f"types.py must not contain '{forbidden}'"
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_types.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'track_gen.types'`.

- [ ] **Step 3 — Write the minimal implementation.** Create `/home/antoine/Documents/track_gen/types.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-free leaf dataclasses shared across the pipeline.

This module imports NOTHING from the rest of the package (no generators, no
inflation, no track_generator, no rng_utils, no warp). It is the shared home for
``TrackGenConfig`` and ``Track`` so that ``inflation.py`` and the facade can both
import them without a circular import, and so CPU-only tests never drag in NVIDIA
Warp.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TrackGenConfig:
    """Single configuration object passed to every stage of the pipeline.

    Fields mirror design spec section 3.2. ``rad``, ``edgy`` and ``half_width``
    are scalars for now (per-env sampling of their ranges is intentionally
    deferred — see the "Deferred (YAGNI)" note at the end of the plan).
    """

    # --- Generator selection + batching ---
    generator: str = "bezier"  # one of {"bezier", "fourier"}
    device: str = "cpu"
    num_envs: int = 1

    # --- Bezier params ---
    min_num_points: int = 9
    max_num_points: int = 13
    num_points_per_segment: int = 30
    min_point_distance: float = 0.05
    min_angle: float = (12.5 / 180) * math.pi
    rad: float = 0.2
    edgy: float = 0.0
    scale: float = 1.0

    # --- Fourier params ---
    num_harmonics: int = 5  # K
    decay_p: int = 2  # decay exponent: amplitude ~ amp / k**decay_p
    amplitude: float = 1.0
    num_centerline_samples: int = 256  # Fourier dense sample count (M_max)

    # --- Width params ---
    half_width: float = 0.1  # w_max
    alpha: float = 0.9  # curvature safety fraction; w * kappa <= alpha < 1
    clamp_self_distance: bool = False
    self_distance_margin: float = 0.0
    self_distance_band: int = 8
    self_distance_decimation: int = 64

    # --- Output params ---
    num_points: int = 256  # N
    output_mode: str = "fixed"  # one of {"fixed", "constant_spacing"}
    spacing: float = 0.1
    N_max: int = 256

    # --- Robustness params ---
    max_regen_iters: int = 10
    turning_tol: float = 0.1
    w_floor: float = 1e-3  # validity: every real point must have w > w_floor


@dataclass
class Track:
    """Final batched result of the track generation pipeline.

    All boundary arrays are index-aligned: ``outer[i]``, ``center[i]`` and
    ``inner[i]`` share a single cross-section normal. Half-width is not stored;
    recover it as ``torch.linalg.norm(outer - center, dim=-1)``.
    """

    outer: Tensor  # [E, N, 2]
    center: Tensor  # [E, N, 2]
    inner: Tensor  # [E, N, 2]
    tangent: Tensor  # [E, N, 2] unit tangent along centerline
    normal: Tensor  # [E, N, 2] unit left-normal along centerline
    arclen: Tensor  # [E, N] cumulative arc length
    length: Tensor  # [E] total length per track
    valid: Tensor  # [E] bool validity mask
    count: Tensor  # [E] int real point count (== N in fixed mode)
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_types.py -v
```

Expected: `4 passed`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add types.py tests/test_types.py
git commit -m "Add types.py leaf: TrackGenConfig + Track dataclasses (breaks import cycle)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: `Centerline` dataclass + `CenterlineGenerator` interface

This task creates `generators.py` with the `Centerline` intermediate dataclass and the abstract `CenterlineGenerator` with `generate(ids) -> Centerline`. `Centerline` STAYS in `generators.py` (per the contract); only `Track`/`TrackGenConfig` live in `types.py`. This task is pure torch — no warp, importable anywhere.

`Centerline(points[E,M_max,2], valid[E] bool)` holds a closed, ordered, dense centerline batch; shorter tracks are NaN-padded to `M_max`, and `valid` is False where a generator gave up for an env.

**Files:**
- Create: `/home/antoine/Documents/track_gen/generators.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_generators.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_generators.py`:

```python
import torch
import pytest

from track_gen.generators import Centerline, CenterlineGenerator


def test_centerline_holds_tensors():
    E, M_max = 4, 7
    points = torch.zeros((E, M_max, 2))
    valid = torch.ones((E,), dtype=torch.bool)
    cl = Centerline(points=points, valid=valid)
    assert cl.points.shape == (E, M_max, 2)
    assert cl.valid.shape == (E,)
    assert cl.valid.dtype == torch.bool


def test_fake_generator_satisfies_protocol():
    class FakeGen(CenterlineGenerator):
        def generate(self, ids):
            E = len(ids)
            return Centerline(
                points=torch.zeros((E, 5, 2)),
                valid=torch.ones((E,), dtype=torch.bool),
            )

    gen = FakeGen()
    assert isinstance(gen, CenterlineGenerator)
    ids = torch.arange(3)
    out = gen.generate(ids)
    assert isinstance(out, Centerline)
    assert out.points.shape == (3, 5, 2)
    assert out.valid.tolist() == [True, True, True]


def test_abstract_generator_cannot_instantiate():
    with pytest.raises(TypeError):
        CenterlineGenerator()
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -v
```

Expected: `ModuleNotFoundError: No module named 'track_gen.generators'`, all three tests error out.

- [ ] **Step 3 — Write minimal implementation.** Create `/home/antoine/Documents/track_gen/generators.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch


@dataclass
class Centerline:
    """A closed, ordered, dense centerline batch.

    Attributes:
        points: [E, M_max, 2] closed dense samples; shorter tracks NaN-padded to M_max.
        valid: [E] bool generation-time validity (False if a generator gave up for an env).
    """

    points: torch.Tensor
    valid: torch.Tensor


class CenterlineGenerator(abc.ABC):
    """Interface every centerline generator implements.

    inflation.inflate consumes only a Centerline and never knows which generator ran.
    """

    @abc.abstractmethod
    def generate(self, ids: torch.Tensor) -> Centerline:
        """Generate one Centerline per env id in `ids`.

        Args:
            ids: [E] int tensor of environment ids to generate for.

        Returns:
            Centerline with points [E, M_max, 2] and valid [E].
        """
        raise NotImplementedError
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -v
```

Expected: 3 passed.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add generators.py tests/test_generators.py
git commit -m "Add Centerline dataclass and CenterlineGenerator interface

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: `BezierCenterlineGenerator.__init__` (bernstein basis + derived params)

The constructor takes `(config, rng)`, stores them, and precomputes the four cubic-Bernstein basis vectors of length `num_points_per_segment` plus the two derived scalars from the old code: `num_cells = int(1 / (2 * min_point_distance))` and `self.p = atan(edgy)/pi + 0.5`. No warp needed (Bernstein precompute is pure scipy+torch). The test builds a `SimpleNamespace` config.

**Note (finding):** `self.p` is the edgy-based blend weight in `[0, 1]` used by `vertex_tangents`. It is DISTINCT from `config.decay_p` (the Fourier decay exponent). The Bezier `SimpleNamespace` config in these tests deliberately has NO `decay_p`/`p` field, which guarantees later tasks read `self.p`, not the config.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/generators.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_generators.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_generators.py`:

```python
import math
import types

from track_gen.generators import BezierCenterlineGenerator


def _bezier_config(**overrides):
    cfg = types.SimpleNamespace(
        min_num_points=9,
        max_num_points=13,
        num_points_per_segment=30,
        min_point_distance=0.05,
        min_angle=(12.5 / 180) * math.pi,
        rad=0.2,
        edgy=0.0,
        scale=1.0,
        device="cpu",
        num_envs=4,
        max_regen_iters=20,
        turning_tol=0.35,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_bezier_init_sets_derived_params():
    cfg = _bezier_config(min_point_distance=0.05, edgy=0.0)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    # num_cells = int(1 / (2 * 0.05)) = int(10.0) = 10
    assert gen.num_cells == 10
    # p = atan(0)/pi + 0.5 = 0.5
    assert gen.p == pytest.approx(0.5)


def test_bezier_init_basis_shapes():
    cfg = _bezier_config(num_points_per_segment=30)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    for basis in (gen.bernstein_0, gen.bernstein_1, gen.bernstein_2, gen.bernstein_3):
        assert basis.shape == (30,)
        assert basis.dtype == torch.float32
    total = gen.bernstein_0 + gen.bernstein_1 + gen.bernstein_2 + gen.bernstein_3
    assert torch.allclose(total, torch.ones(30), atol=1e-5)


def test_bezier_init_p_increases_with_edgy():
    low = BezierCenterlineGenerator(_bezier_config(edgy=0.0), rng=None).p
    high = BezierCenterlineGenerator(_bezier_config(edgy=5.0), rng=None).p
    assert high > low
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k bezier_init -v
```

Expected: FAIL with `ImportError: cannot import name 'BezierCenterlineGenerator'`.

- [ ] **Step 3 — Write minimal implementation.** Add to the top of `generators.py` (with the other imports):

```python
import math

import numpy as np
from scipy.special import binom
```

Then append:

```python
def bernstein(n: int, k: int, t: np.ndarray) -> np.ndarray:
    """The k-th Bernstein basis polynomial of degree n at t (ported from track_generator.py)."""
    return binom(n, k) * t**k * (1.0 - t) ** (n - k)


class BezierCenterlineGenerator(CenterlineGenerator):
    """Closed-Bezier centerline generator (the repaired ccw_sort / get_bezier_curve pipeline)."""

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.device = config.device

        # p maps edginess into [0, 1]; it weights the outgoing vs incoming edge direction.
        # This is the vertex_tangents blend weight, NOT the Fourier decay exponent.
        self.p = math.atan(config.edgy) / math.pi + 0.5
        # Number of grid cells per axis; smaller min_point_distance => finer grid.
        self.num_cells = int(1.0 / (config.min_point_distance * 2))

        # Precompute the four cubic (degree-3) Bernstein basis vectors over a uniform t grid.
        t = np.linspace(0.0, 1.0, num=config.num_points_per_segment)
        self.bernstein_0 = torch.tensor(bernstein(3, 0, t), device=self.device, dtype=torch.float32)
        self.bernstein_1 = torch.tensor(bernstein(3, 1, t), device=self.device, dtype=torch.float32)
        self.bernstein_2 = torch.tensor(bernstein(3, 2, t), device=self.device, dtype=torch.float32)
        self.bernstein_3 = torch.tensor(bernstein(3, 3, t), device=self.device, dtype=torch.float32)

    def generate(self, ids: torch.Tensor) -> Centerline:
        raise NotImplementedError  # filled in by later tasks
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k bezier_init -v
```

Expected: 3 passed.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add generators.py tests/test_generators.py
git commit -m "Add BezierCenterlineGenerator.__init__ with bernstein basis precompute

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: On-GPU cell sampling via top-k (replaces numpy unique-int)

A method `_sample_corner_points(ids) -> [E, max_num_points, 2]`: draw `u = rng.sample_uniform_torch(0, 1, (num_cells**2,), ids)`; the indices of the `max_num_points` largest values (`u.topk(max_num_points).indices`) are a uniform subset without replacement — fully device-resident, per-env seeded. Convert each cell index to grid `x = idx % num_cells`, `y = idx // num_cells`, add per-corner uniform noise in `[-0.5, 0.5)`, multiply by `min_point_distance * 2`, then by `scale`. **Warp-guarded** (uses the RNG).

`rng.sample_uniform_torch(low, high, shape, ids)` returns `[len(ids), *shape]`. The `_make_rng` helper builds seeds as `arange(num_envs) + seed`, so env `e` has seed `seed + e`.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/generators.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_generators.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_generators.py`:

```python
def _make_rng(num_envs, seed=1234, device="cpu"):
    import warp as wp  # noqa: F401  (governed by importorskip in each test)
    from track_gen.rng_utils import PerEnvSeededRNG

    seeds = torch.arange(num_envs, dtype=torch.int32) + seed
    return PerEnvSeededRNG(seeds=seeds, num_envs=num_envs, device=device)


def test_cell_sampling_shape():
    pytest.importorskip("warp")
    E = 4
    cfg = _bezier_config(num_envs=E, device="cpu")
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E))
    ids = torch.arange(E)
    pts = gen._sample_corner_points(ids)
    assert pts.shape == (E, cfg.max_num_points, 2)
    assert torch.isfinite(pts).all()


def test_cell_sampling_reproducible():
    pytest.importorskip("warp")
    E = 4
    cfg = _bezier_config(num_envs=E, device="cpu")
    ids = torch.arange(E)
    gen_a = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=7))
    gen_b = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=7))
    pts_a = gen_a._sample_corner_points(ids)
    pts_b = gen_b._sample_corner_points(ids)
    assert torch.allclose(pts_a, pts_b)


def test_cell_indices_distinct_per_env():
    pytest.importorskip("warp")
    E = 4
    cfg = _bezier_config(num_envs=E, device="cpu")
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E))
    ids = torch.arange(E)
    idxs = gen._sample_cell_indices(ids)  # [E, max_num_points] cell ids
    assert idxs.shape == (E, cfg.max_num_points)
    for e in range(E):
        assert len(torch.unique(idxs[e])) == cfg.max_num_points  # no duplicate cells


def test_cell_sampling_env_independence():
    pytest.importorskip("warp")
    E = 3
    cfg = _bezier_config(num_envs=E, device="cpu")
    ids = torch.arange(E)
    base = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=100))
    base_pts = base._sample_corner_points(ids)
    import warp as wp  # noqa: F401
    from track_gen.rng_utils import PerEnvSeededRNG

    seeds = torch.tensor([100, 101, 999], dtype=torch.int32)
    rng2 = PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")
    gen2 = BezierCenterlineGenerator(cfg, rng=rng2)
    pts2 = gen2._sample_corner_points(ids)
    assert torch.allclose(base_pts[0], pts2[0])
    assert torch.allclose(base_pts[1], pts2[1])
    assert not torch.allclose(base_pts[2], pts2[2])
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k cell -v
```

Expected: FAIL with `AttributeError: ... has no attribute '_sample_cell_indices'` (and `_sample_corner_points`), OR SKIP if warp unavailable.

- [ ] **Step 3 — Write minimal implementation.** Add these two methods to `BezierCenterlineGenerator` (above `generate`):

```python
    def _sample_cell_indices(self, ids: torch.Tensor) -> torch.Tensor:
        """Per-env uniform subset (without replacement) of grid cell indices.

        Draws num_cells**2 i.i.d. uniforms per env; the indices of the max_num_points
        largest are a uniform k-subset without replacement (top-k trick). Device-resident,
        per-env seeded -- replaces the old numpy rng.choice host-sync path.

        Returns:
            [E, max_num_points] long tensor of cell indices in [0, num_cells**2).
        """
        n = self.num_cells * self.num_cells
        u = self.rng.sample_uniform_torch(0.0, 1.0, (n,), ids=ids)  # [E, n]
        cell_idxs = u.topk(self.config.max_num_points, dim=1).indices  # [E, max_num_points]
        return cell_idxs.long()

    def _sample_corner_points(self, ids: torch.Tensor) -> torch.Tensor:
        """Sample max_num_points corner points in scaled grid coordinates.

        Returns:
            [E, max_num_points, 2] float tensor.
        """
        cell_idxs = self._sample_cell_indices(ids)  # [E, max_num_points]
        x = (cell_idxs % self.num_cells).float()
        y = (cell_idxs // self.num_cells).float()
        # Per-corner uniform noise in [-0.5, 0.5) makes the discrete grid continuous.
        noise = self.rng.sample_uniform_torch(-0.5, 0.5, (self.config.max_num_points, 2), ids=ids)
        xy = torch.stack([x, y], dim=2) * (self.config.min_point_distance * 2.0) + noise
        return xy * self.config.scale
```

- [ ] **Step 4 — Run it, expect PASS (or SKIP without warp).**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k cell -v
```

Expected with warp: 4 passed. Without warp: all SKIPPED.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add generators.py tests/test_generators.py
git commit -m "Add on-GPU top-k cell sampling for Bezier corners

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 17: Variable corner count with NaN padding (no prune-collapse)

A method `_prune_corners(points, ids) -> (points[E,max_num_points,2], count[E])`: sample a per-env corner count uniformly in `[min_num_points, max_num_points]`; corners beyond that count are set to a **NaN sentinel** — NEVER collapsed onto the first point (the old bug that created zero-length segments). Real (kept) corners are the first `count[e]` rows after `ccw_sort`; trailing rows become `NaN`. **Warp-guarded.**

**Files:**
- Modify: `/home/antoine/Documents/track_gen/generators.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_generators.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_generators.py`:

```python
def test_prune_corners_shape_and_count():
    pytest.importorskip("warp")
    E = 6
    cfg = _bezier_config(num_envs=E, device="cpu", min_num_points=9, max_num_points=13)
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E))
    ids = torch.arange(E)
    raw = gen._sample_corner_points(ids)
    pruned, count = gen._prune_corners(raw, ids)
    assert pruned.shape == (E, cfg.max_num_points, 2)
    assert count.shape == (E,)
    assert (count >= cfg.min_num_points).all()
    assert (count <= cfg.max_num_points).all()


def test_prune_corners_pads_with_nan():
    pytest.importorskip("warp")
    E = 6
    cfg = _bezier_config(num_envs=E, device="cpu", min_num_points=4, max_num_points=13)
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=42))
    ids = torch.arange(E)
    raw = gen._sample_corner_points(ids)
    pruned, count = gen._prune_corners(raw, ids)
    for e in range(E):
        c = int(count[e])
        assert torch.isfinite(pruned[e, :c]).all()
        if c < cfg.max_num_points:
            assert torch.isnan(pruned[e, c:]).all()
        finite_rows = torch.isfinite(pruned[e]).all(dim=1).sum().item()
        assert finite_rows == c


def test_prune_corners_reproducible():
    pytest.importorskip("warp")
    E = 5
    cfg = _bezier_config(num_envs=E, device="cpu")
    ids = torch.arange(E)
    a = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=3))
    b = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=3))
    pa, ca = a._prune_corners(a._sample_corner_points(ids), ids)
    pb, cb = b._prune_corners(b._sample_corner_points(ids), ids)
    assert torch.equal(ca, cb)
    assert torch.equal(torch.isnan(pa), torch.isnan(pb))
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k prune -v
```

Expected: FAIL with `AttributeError: ... has no attribute '_prune_corners'`, or SKIP without warp.

- [ ] **Step 3 — Write minimal implementation.** Add `ccw_sort` to the geometry import at the top of `generators.py`:

```python
from .geometry import ccw_sort
```

Add the method to `BezierCenterlineGenerator` (above `generate`):

```python
    def _prune_corners(self, points: torch.Tensor, ids: torch.Tensor):
        """ccw-sort corners, then NaN-pad a per-env random tail to vary the corner count.

        Args:
            points: [E, max_num_points, 2] raw sampled corners.
            ids: [E] env ids (for per-env reproducible count sampling).

        Returns:
            (pruned [E, max_num_points, 2], count [E] long) where rows >= count are NaN.
        """
        E, P, _ = points.shape
        points = ccw_sort(points)  # disjoint angular wedges -> simple polygon

        # Per-env corner count in [min_num_points, max_num_points] (inclusive).
        # sample_integer_torch samples in [low, high); high = max+1 for an inclusive upper bound.
        count = self.rng.sample_integer_torch(
            self.config.min_num_points,
            self.config.max_num_points + 1,
            (1,),
            ids=ids,
        ).view(E).long()
        count = count.clamp(max=P)

        row_idx = torch.arange(P, device=points.device).unsqueeze(0).expand(E, P)
        keep = row_idx < count.unsqueeze(1)  # [E, P] bool
        nan = torch.full_like(points, float("nan"))
        pruned = torch.where(keep.unsqueeze(-1), points, nan)
        return pruned, count
```

- [ ] **Step 4 — Run it, expect PASS (or SKIP).**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k prune -v
```

Expected with warp: 3 passed. Without warp: SKIPPED.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add generators.py tests/test_generators.py
git commit -m "Add NaN-padded variable corner count (no prune-collapse)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 18: Centerline assembly (vector tangents + cubic Bézier dense polyline)

A method `_assemble_centerline(corners[E,P,2]) -> dense[E, M_max, 2]` that:
1. computes per-vertex unit tangents with `geometry.vertex_tangents(corners, self.p)` — **using `self.p` (the edgy-based blend weight), NOT `config.p`/`config.decay_p`** (finding);
2. for each consecutive corner pair `(i, i+1)` (wrapping `P-1 -> 0`) builds a cubic Bézier whose inner handles lie at distance `rad * chord` along the corner tangents, sampled at `num_points_per_segment` points via the precomputed Bernstein basis;
3. concatenates all `P` segments into a closed dense polyline of length `M_max = P * num_points_per_segment`.

Pruned (NaN) corners make their incident segments NaN, which propagate downstream. **Pure torch** (no RNG) — testable with a hand-built corner polygon and **no warp**.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/generators.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_generators.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_generators.py`:

```python
def _square_corners(E=2):
    sq = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return sq.unsqueeze(0).expand(E, 4, 2).contiguous()


def test_assemble_centerline_shape():
    cfg = _bezier_config(num_points_per_segment=30)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    corners = _square_corners(E=3)  # P=4
    dense = gen._assemble_centerline(corners)
    assert dense.shape == (3, 4 * 30, 2)


def test_assemble_centerline_is_closed_loop():
    cfg = _bezier_config(num_points_per_segment=30, rad=0.2, edgy=0.0)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    corners = _square_corners(E=1)
    dense = gen._assemble_centerline(corners)
    assert torch.isfinite(dense).all()
    gap = torch.linalg.norm(dense[0, -1] - dense[0, 0])
    seg_step = torch.linalg.norm(dense[0, 1] - dense[0, 0])
    assert gap <= 3.0 * seg_step + 1e-4


def test_assemble_centerline_nan_corner_propagates():
    cfg = _bezier_config(num_points_per_segment=30)
    gen = BezierCenterlineGenerator(cfg, rng=None)
    # Use a hexagon: vertex_tangents makes a pruned corner poison its tangent
    # plus its two neighbours' tangents (4 consecutive cubic segments). With a
    # 4-corner square that is ALL segments, so nothing finite survives; with
    # >=6 corners at least one fully-finite segment remains, which is what lets
    # us assert that the NaN propagates *locally* rather than destroying the
    # whole dense polyline.
    ang = torch.arange(6, dtype=torch.float32) * (2.0 * torch.pi / 6.0)
    corners = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1).unsqueeze(0)  # [1, 6, 2]
    corners[0, 2] = float("nan")  # prune the 3rd corner
    dense = gen._assemble_centerline(corners)
    assert torch.isnan(dense[0]).any()
    assert torch.isfinite(dense[0]).any()
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k assemble -v
```

Expected: FAIL with `AttributeError: ... has no attribute '_assemble_centerline'`.

- [ ] **Step 3 — Write minimal implementation.** Add `vertex_tangents` to the geometry import line in `generators.py`:

```python
from .geometry import ccw_sort, vertex_tangents
```

Add these methods to `BezierCenterlineGenerator` (above `generate`):

```python
    def _cubic_bezier(self, p0, p1, p2, p3):
        """Evaluate a batched cubic Bezier with the precomputed Bernstein basis.

        Args:
            p0, p1, p2, p3: each [E, 2] control points.

        Returns:
            [E, num_points_per_segment, 2] dense samples.
        """
        curve = (
            torch.einsum("s,ed->esd", self.bernstein_0, p0)
            + torch.einsum("s,ed->esd", self.bernstein_1, p1)
            + torch.einsum("s,ed->esd", self.bernstein_2, p2)
            + torch.einsum("s,ed->esd", self.bernstein_3, p3)
        )
        return curve

    def _segment(self, c0, c1, t0, t1):
        """Cubic Bezier from corner c0 (tangent t0) to corner c1 (tangent t1).

        Inner handles sit at distance rad * chord along the corner tangents.
        """
        chord = torch.linalg.norm(c1 - c0, dim=1, keepdim=True)  # [E, 1]
        handle = self.config.rad * chord
        p1 = c0 + t0 * handle  # leave c0 along its tangent
        p2 = c1 - t1 * handle  # arrive at c1 along its tangent
        return self._cubic_bezier(c0, p1, p2, c1)

    def _assemble_centerline(self, corners: torch.Tensor) -> torch.Tensor:
        """Build the closed dense centerline from ccw-ordered (possibly NaN-padded) corners.

        Args:
            corners: [E, P, 2]; NaN rows are pruned corners.

        Returns:
            [E, P * num_points_per_segment, 2] closed dense polyline (NaN where pruned).
        """
        P = corners.shape[1]
        # Use the derived edgy-based blend weight self.p, NOT config.decay_p.
        tangents = vertex_tangents(corners, self.p)  # [E, P, 2] unit, NaN at pruned

        segments = []
        for i in range(P):
            j = (i + 1) % P  # wrap the last corner back to the first
            seg = self._segment(corners[:, i], corners[:, j], tangents[:, i], tangents[:, j])
            segments.append(seg)
        return torch.cat(segments, dim=1)
```

Note: NaN propagation is automatic — a NaN corner makes its tangent NaN (via `vertex_tangents`/`safe_normalize`), so both incident segments contain NaN samples.

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k assemble -v
```

Expected: 3 passed.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add generators.py tests/test_generators.py
git commit -m "Assemble Bezier centerline via vector tangents (self.p) and cubic segments

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 19: Bounded iterative regeneration + `generate` (min-angle + turning-number gate, NaN-aware)

Wire the full Bézier `generate(ids) -> Centerline`: a **bounded `while` loop** (NOT the old unbounded recursion) that, each iteration, regenerates only the envs failing the checks:
1. the **clamped-`arccos` min-angle** test on the control polygon (NaN corners yield angle `0` via `nan_to_num`, so they fail);
2. the **turning-number gate**: `|turning_number(real centerline)| ≈ 2π` within `config.turning_tol`.

**Two findings applied here:**
- **`torch.all` tuple-dim crash:** replace `torch.isfinite(dense).all(dim=(1, 2))` (raises `TypeError` on torch 2.1.2) with the flattened reduction `torch.isfinite(dense).reshape(dense.shape[0], -1).all(dim=1)`.
- **NaN-aware turning/finiteness gates:** the variable-corner-count feature is self-defeating if `turning_number`/finiteness run on the NaN-padded `dense`, because a legitimately-pruned env (count < max) always contains NaN segments and can never pass. Evaluate the turning-number and finiteness gates per env over the env's **real (non-NaN) compacted** centerline — the same masked path inflation uses — via a helper `_real_turning_and_finite(dense)`. A pruned-but-otherwise-good track is then accepted.

The loop is capped at `config.max_regen_iters`; envs still failing get `valid = False` with NaN points. **Warp-guarded** (except `_corner_angles`, which is pure).

**Files:**
- Modify: `/home/antoine/Documents/track_gen/generators.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_generators.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_generators.py`:

```python
def test_corner_angles_clamped_no_nan():
    cfg = _bezier_config()
    gen = BezierCenterlineGenerator(cfg, rng=None)
    corners = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])  # repeated corner
    ang = gen._corner_angles(corners)
    assert ang.shape == (1, 4)
    assert torch.isfinite(ang).all()


def test_generate_returns_centerline():
    pytest.importorskip("warp")
    E = 8
    cfg = _bezier_config(num_envs=E, device="cpu")
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=11))
    ids = torch.arange(E)
    cl = gen.generate(ids)
    from track_gen.generators import Centerline

    assert isinstance(cl, Centerline)
    M_max = cfg.max_num_points * cfg.num_points_per_segment
    assert cl.points.shape == (E, M_max, 2)
    assert cl.valid.shape == (E,)
    assert cl.valid.dtype == torch.bool


def test_generate_reproducible():
    pytest.importorskip("warp")
    E = 6
    cfg = _bezier_config(num_envs=E, device="cpu")
    ids = torch.arange(E)
    a = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=5)).generate(ids)
    b = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=5)).generate(ids)
    assert torch.equal(torch.isnan(a.points), torch.isnan(b.points))
    fin = torch.isfinite(a.points)
    assert torch.allclose(a.points[fin], b.points[fin])
    assert torch.equal(a.valid, b.valid)


def test_generate_pathological_flags_invalid_without_hang():
    pytest.importorskip("warp")
    E = 4
    cfg = _bezier_config(num_envs=E, device="cpu", min_angle=3.10, max_regen_iters=3)
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=1))
    ids = torch.arange(E)
    cl = gen.generate(ids)  # must return, not hang
    assert cl.valid.shape == (E,)
    assert (~cl.valid).all()


def test_generate_accepts_pruned_variable_count_tracks():
    pytest.importorskip("warp")
    # With a wide [min,max] count window, many envs draw < max corners. The
    # NaN-aware gates must still accept geometrically-good pruned tracks, so
    # not every valid env has exactly max_num_points corners.
    E = 32
    cfg = _bezier_config(num_envs=E, device="cpu", min_num_points=6, max_num_points=13)
    gen = BezierCenterlineGenerator(cfg, rng=_make_rng(E, seed=99))
    ids = torch.arange(E)
    cl = gen.generate(ids)
    # At least one valid env exists and at least one valid env has a NaN tail
    # (i.e. a pruned, variable-count track was accepted).
    valid_idx = torch.where(cl.valid)[0]
    assert valid_idx.numel() > 0
    has_nan_tail = torch.tensor(
        [bool(torch.isnan(cl.points[e]).any()) for e in valid_idx.tolist()]
    )
    assert has_nan_tail.any(), "no pruned variable-count track was accepted"
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k "corner_angles or generate_" -v
```

Expected: FAIL — `_corner_angles` missing; `generate` still raises `NotImplementedError`.

- [ ] **Step 3 — Write minimal implementation.** Add `turning_number`, `safe_normalize`, and the resample helper to the geometry import:

```python
from .geometry import arc_length_resample, ccw_sort, safe_normalize, turning_number, vertex_tangents
```

Add `_corner_angles`, `_real_turning_and_finite`, and replace the placeholder `generate` in `BezierCenterlineGenerator`:

```python
    def _corner_angles(self, corners: torch.Tensor) -> torch.Tensor:
        """Interior angle at each corner via clamped arccos (NaN-safe).

        Args:
            corners: [E, P, 2] (may contain NaN pruned rows).

        Returns:
            [E, P] angles in radians; degenerate/NaN corners -> 0.0 (always fail).
        """
        eps = 1e-7
        prev = torch.roll(corners, 1, dims=1)
        nxt = torch.roll(corners, -1, dims=1)
        u_in = safe_normalize(corners - prev)
        u_out = safe_normalize(nxt - corners)
        cos_turn = (u_in * u_out).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        angle = math.pi - torch.arccos(cos_turn)  # interior angle
        return torch.nan_to_num(angle, nan=0.0)

    def _real_turning_and_finite(self, dense: torch.Tensor):
        """Per-env turning number + finiteness computed over REAL (non-NaN) points only.

        The NaN-padded dense buffer would otherwise poison turning_number for any
        pruned (variable-count) env. We compact each env to a fixed-N real
        centerline via arc_length_resample (which drops NaN points), then gate on
        that. An env with < 2 real points yields turn = nan (fails) and finite = False.

        Args:
            dense: [n, M_max, 2] candidate centerlines (may contain NaN).

        Returns:
            (turn [n], finite_ok [n] bool).
        """
        n = dense.shape[0]
        # Resample onto a fixed-N real loop (NaN dropped); count[e] == 0 for all-NaN env.
        resampled, count = arc_length_resample(dense, num=self.config.num_points_per_segment)
        turn = turning_number(resampled)  # [n]; nan where the loop is degenerate/NaN
        finite_ok = (count >= 2) & torch.isfinite(turn)
        return turn, finite_ok

    def _generate_batch(self, ids: torch.Tensor):
        """One full draw for the given ids: corners -> prune -> dense centerline + control corners."""
        raw = self._sample_corner_points(ids)
        pruned, _count = self._prune_corners(raw, ids)
        dense = self._assemble_centerline(pruned)
        return dense, pruned

    def generate(self, ids: torch.Tensor) -> Centerline:
        E = len(ids)
        M_max = self.config.max_num_points * self.config.num_points_per_segment

        points = torch.full((E, M_max, 2), float("nan"), device=self.device)
        valid = torch.zeros((E,), dtype=torch.bool, device=self.device)
        pending = torch.arange(E, device=self.device)  # local rows still needing a good draw

        for _ in range(self.config.max_regen_iters):
            if pending.numel() == 0:
                break
            sub_ids = ids[pending]
            dense, corners = self._generate_batch(sub_ids)

            # Gate 1: every interior corner angle exceeds min_angle (NaN corners -> 0 -> fail).
            angles = self._corner_angles(corners)  # [n, P]
            angle_ok = (angles > self.config.min_angle).all(dim=1)
            # Gates 2 & 3: turning number ~ 2*pi AND finite, evaluated on REAL points only.
            turn, finite_ok = self._real_turning_and_finite(dense)
            turn_ok = (turn.abs() - 2.0 * math.pi).abs() <= self.config.turning_tol
            ok = angle_ok & turn_ok & finite_ok

            good = pending[ok]
            points[good] = dense[ok]
            valid[good] = True
            pending = pending[~ok]

        return Centerline(points=points, valid=valid)
```

Note: the loop only re-draws `pending` rows, so it never hangs — at most `max_regen_iters` iterations. Each iteration advances the per-env RNG stream, giving fresh geometry. Unconverged envs keep `valid=False` with NaN points; inflation's R<2 guard handles those rows.

- [ ] **Step 4 — Run it, expect PASS (or SKIP for warp-guarded ones).**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k "corner_angles or generate_" -v
```

Expected: `test_corner_angles_clamped_no_nan` PASSES (no warp); the four warp-guarded tests PASS with warp / SKIP without. None hang.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add generators.py tests/test_generators.py
git commit -m "Add bounded Bezier regen loop with NaN-aware min-angle + turning gates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 20: `FourierCenterlineGenerator` (guaranteed-smooth alternative)

A second generator behind the same interface. Sample harmonic coefficients `a_k, b_k ∈ ℝ²` with `a_k, b_k ~ N(0, (amplitude / k**decay_p)²)` for `k = 1..K`; evaluate `c(t) = Σ_k a_k cos(k t) + b_k sin(k t)` on a dense `t` grid in `[0, 2π)`; mean-center; scale the bounding box so its larger side equals `config.scale`; run the turning-number gate as a cheap safety net for rare low-`K` self-crossing. `M_max = config.num_centerline_samples`, no NaN padding. **Warp-guarded.**

**Finding applied (sample_normal_torch):** do NOT pass a float mean with a tensor std (the warp `normal()` dispatch raises `ValueError`), and the tensorized normal kernel only honors a per-env scalar std anyway. Instead sample standard normals with float args (`rng.sample_normal_torch(0.0, 1.0, (K, 2), ids=ids)` → `[E, K, 2]`) and scale in torch by the precomputed per-harmonic decay `self.std_k.view(1, K, 1)`. The generator reads `config.decay_p` and `config.num_centerline_samples` (NOT `p`/`num_centerline_samples` mismatches).

**Files:**
- Modify: `/home/antoine/Documents/track_gen/generators.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_generators.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_generators.py`:

```python
from track_gen.generators import FourierCenterlineGenerator


def _fourier_config(**overrides):
    cfg = types.SimpleNamespace(
        num_harmonics=3,
        decay_p=2.0,
        amplitude=1.0,
        scale=10.0,
        num_centerline_samples=256,
        device="cpu",
        num_envs=4,
        turning_tol=0.5,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_fourier_generate_shape_and_closed():
    pytest.importorskip("warp")
    E = 4
    cfg = _fourier_config(num_envs=E, num_centerline_samples=256)
    gen = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=21))
    ids = torch.arange(E)
    cl = gen.generate(ids)
    assert cl.points.shape == (E, 256, 2)
    assert torch.isfinite(cl.points).all()
    for e in range(E):
        gap = torch.linalg.norm(cl.points[e, -1] - cl.points[e, 0])
        step = torch.linalg.norm(cl.points[e, 1] - cl.points[e, 0])
        assert gap <= 3.0 * step + 1e-4


def test_fourier_mean_centered_and_scaled():
    pytest.importorskip("warp")
    E = 3
    cfg = _fourier_config(num_envs=E, scale=10.0)
    gen = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=8))
    cl = gen.generate(torch.arange(E))
    for e in range(E):
        center = cl.points[e].mean(dim=0)
        assert torch.allclose(center, torch.zeros(2), atol=1e-4)
        bbox = cl.points[e].amax(dim=0) - cl.points[e].amin(dim=0)
        assert bbox.amax().item() == pytest.approx(10.0, abs=1e-3)


def test_fourier_reproducible_and_independent():
    pytest.importorskip("warp")
    E = 4
    cfg = _fourier_config(num_envs=E)
    ids = torch.arange(E)
    a = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=2)).generate(ids)
    b = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=2)).generate(ids)
    assert torch.allclose(a.points, b.points)
    import warp as wp  # noqa: F401
    from track_gen.rng_utils import PerEnvSeededRNG

    seeds = torch.tensor([2, 3, 4, 999], dtype=torch.int32)
    c = FourierCenterlineGenerator(cfg, rng=PerEnvSeededRNG(seeds=seeds, num_envs=E, device="cpu")).generate(ids)
    assert torch.allclose(a.points[0], c.points[0])
    assert not torch.allclose(a.points[3], c.points[3])


def test_fourier_low_k_turning_is_loop():
    pytest.importorskip("warp")
    from track_gen.geometry import turning_number

    E = 4
    cfg = _fourier_config(num_envs=E, num_harmonics=1, decay_p=2.0)
    cl = FourierCenterlineGenerator(cfg, rng=_make_rng(E, seed=14)).generate(torch.arange(E))
    turn = turning_number(cl.points)
    assert torch.allclose(turn.abs(), torch.full((E,), 2.0 * math.pi), atol=cfg.turning_tol)
    assert cl.valid.dtype == torch.bool
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k fourier -v
```

Expected: FAIL with `ImportError: cannot import name 'FourierCenterlineGenerator'`.

- [ ] **Step 3 — Write minimal implementation.** Append the class to `/home/antoine/Documents/track_gen/generators.py`:

```python
class FourierCenterlineGenerator(CenterlineGenerator):
    """Truncated-Fourier centerline generator: smooth-by-construction closed curves."""

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.device = config.device
        self.K = config.num_harmonics
        self.M = config.num_centerline_samples

        # Dense parameter grid over [0, 2*pi) (endpoint excluded so the loop closes cleanly).
        t = torch.linspace(0.0, 2.0 * math.pi, self.M + 1, device=self.device)[:-1]  # [M]
        self.t = t
        k = torch.arange(1, self.K + 1, device=self.device, dtype=torch.float32)  # [K]
        self.cos_kt = torch.cos(k.unsqueeze(1) * t.unsqueeze(0))  # [K, M]
        self.sin_kt = torch.sin(k.unsqueeze(1) * t.unsqueeze(0))  # [K, M]
        # Per-harmonic std: amplitude / k**decay_p.
        self.std_k = config.amplitude / (k**config.decay_p)  # [K]

    def generate(self, ids: torch.Tensor) -> Centerline:
        E = len(ids)
        # Sample standard normals (float args), then scale by the per-harmonic decay in torch.
        # NOTE: do NOT pass a tensor std into sample_normal_torch (warp dispatch rejects
        # float-mean / tensor-std, and only honors a per-env scalar std).
        a = self.rng.sample_normal_torch(0.0, 1.0, (self.K, 2), ids=ids)  # [E, K, 2]
        b = self.rng.sample_normal_torch(0.0, 1.0, (self.K, 2), ids=ids)  # [E, K, 2]
        a = a * self.std_k.view(1, self.K, 1)
        b = b * self.std_k.view(1, self.K, 1)

        # c(t) = sum_k a_k cos(k t) + b_k sin(k t); c0 omitted (cancels under mean-centering).
        curve = torch.einsum("ekd,km->emd", a, self.cos_kt) + torch.einsum("ekd,km->emd", b, self.sin_kt)

        curve = curve - curve.mean(dim=1, keepdim=True)
        bbox = curve.amax(dim=1) - curve.amin(dim=1)  # [E, 2]
        longest = bbox.amax(dim=1, keepdim=True).clamp_min(1e-8)  # [E, 1]
        curve = curve * (self.config.scale / longest).unsqueeze(1)

        # valid via the turning-number safety net for rare low-K crossings.
        turn = turning_number(curve)  # [E]
        valid = (turn.abs() - 2.0 * math.pi).abs() <= self.config.turning_tol

        return Centerline(points=curve, valid=valid)
```

- [ ] **Step 4 — Run it, expect PASS (or SKIP without warp).**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -k fourier -v
```

Expected with warp: 4 passed. Without warp: SKIPPED.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add generators.py tests/test_generators.py
git commit -m "Add FourierCenterlineGenerator (torch-scaled std, turning-number gate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 21: Full generators-module regression run

Confirm the whole `generators.py` test file passes together (catches cross-task interaction and import ordering), both with and without warp present.

**Finding applied:** the sentinel `test_module_exposes_both_generators` IS the new test; it should pass immediately because Tasks 14-20 already define every symbol. Add it, run the full file as the regression gate.

**Files:**
- Test: `/home/antoine/Documents/track_gen/tests/test_generators.py`

- [ ] **Step 1 — Write the test.** Append `test_module_exposes_both_generators` to `/home/antoine/Documents/track_gen/tests/test_generators.py`:

```python
def test_module_exposes_both_generators():
    from track_gen.generators import (
        BezierCenterlineGenerator,
        Centerline,
        CenterlineGenerator,
        FourierCenterlineGenerator,
    )

    assert issubclass(BezierCenterlineGenerator, CenterlineGenerator)
    assert issubclass(FourierCenterlineGenerator, CenterlineGenerator)
    assert Centerline.__dataclass_fields__.keys() == {"points", "valid"}
```

- [ ] **Step 2 — Run it.** This sentinel passes immediately (the symbols already exist from Tasks 14-20); it would only fail if a regression broke an import. Run the whole file as the regression gate:

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -v
```

- [ ] **Step 3 — No implementation change needed.** `Centerline`, both generators, and the interface already exist.

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generators.py -v
```

Expected: every test in the file PASSES (warp-dependent ones SKIP cleanly when warp/GPU unavailable). No hangs, no NaN-comparison failures.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add tests/test_generators.py
git commit -m "Add generators module regression sentinel test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 22: Inflation resample stage — masked arc-length resampling of the centerline

This begins `inflation.py`: `inflate(centerline, config) -> Track`, the stage that turns a dense centerline into three index-aligned closed polylines (outer/center/inner) plus per-point frame, curvature-clamped half-width, arc-length metadata, and a per-track validity flag. It is pure batched torch, runs on CPU, and depends only on `geometry.py` plus the `Centerline` (from `generators.py`) and `Track`/`TrackGenConfig` (from the leaf `types.py`) dataclasses.

**Crucially, every inflation test builds its own `Centerline` input directly as tensors (circle / ellipse / near-touch / self-crossing) — no generator and no Warp are needed.** Inflation tests import `Track`/`TrackGenConfig` from `track_gen.types` (the warp-free leaf) and `Centerline` from `track_gen.generators`. `generators.py` imports only `geometry` + scipy/numpy at module top (the warp-touching RNG is only referenced inside generator `__init__`/`generate`, never at import), so importing `Centerline` does not drag in warp.

This task's first stage returns a small intermediate namedtuple `_ResampleResult(center, count)` so the resample stage is testable in isolation; later tasks replace the return with a full `Track`. It honors `config.output_mode`: `"fixed"` → `num=config.num_points` (so `N = num_points`, `count == N`); `"constant_spacing"` → `spacing=config.spacing` with `n_max=config.N_max` (variable reals padded to `config.N_max`).

**Files:**
- Create: `/home/antoine/Documents/track_gen/inflation.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_inflation.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_inflation.py`:

```python
import math

import pytest
import torch

from track_gen.generators import Centerline
from track_gen.types import Track, TrackGenConfig
from track_gen import inflation


def make_circle_centerline(radius=2.0, m=200, e=1, center=(0.0, 0.0), device="cpu"):
    """Build a Centerline whose points lie exactly on a circle (closed, no NaN padding)."""
    theta = torch.linspace(0, 2 * math.pi, m + 1, device=device)[:-1]  # drop duplicate endpoint
    x = center[0] + radius * torch.cos(theta)
    y = center[1] + radius * torch.sin(theta)
    pts = torch.stack([x, y], dim=-1)  # [m, 2]
    pts = pts.unsqueeze(0).expand(e, m, 2).contiguous()  # [e, m, 2]
    valid = torch.ones(e, dtype=torch.bool, device=device)
    return Centerline(points=pts, valid=valid)


def fixed_config(num_points=128, device="cpu", **overrides):
    """A TrackGenConfig in fixed output mode with self-distance clamp OFF by default."""
    kwargs = dict(
        device=device,
        num_envs=1,
        output_mode="fixed",
        num_points=num_points,
        clamp_self_distance=False,
    )
    kwargs.update(overrides)
    return TrackGenConfig(**kwargs)


def test_resample_stage_circle_is_arc_uniform_and_on_circle():
    radius = 2.0
    cl = make_circle_centerline(radius=radius, m=200, e=3)
    cfg = fixed_config(num_points=128, num_envs=3)

    res = inflation._resample_stage(cl, cfg)

    assert res.center.shape == (3, 128, 2)
    assert torch.equal(res.count, torch.full((3,), 128, dtype=res.count.dtype))
    r = torch.linalg.norm(res.center, dim=-1)  # [E, N]
    assert torch.allclose(r, torch.full_like(r, radius), atol=1e-3)
    seg = torch.linalg.norm(torch.diff(res.center, dim=1, append=res.center[:, :1]), dim=-1)
    assert seg.std(dim=1).max().item() < 1e-3
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py -v
```

Expected: collection/import error or `AttributeError: module 'track_gen.inflation' has no attribute '_resample_stage'`.

- [ ] **Step 3 — Write the minimal implementation.** Create `/home/antoine/Documents/track_gen/inflation.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inflation stage: turn a dense Centerline into a Track (outer/center/inner + metadata).

Pure batched torch, device-agnostic, CPU-testable. Depends only on geometry.py and
the warp-free leaf dataclasses in types.py (Track, TrackGenConfig). It must NOT import
from track_generator (that would create a circular import); Track comes from .types.
"""

import math
from collections import namedtuple

import torch

from . import geometry
from .types import Track, TrackGenConfig  # noqa: F401  (TrackGenConfig used for typing)

# Intermediate result of the resample stage; replaced by a full Track once inflate() is complete.
_ResampleResult = namedtuple("_ResampleResult", ["center", "count"])


def _valid_mask_from_points(points: torch.Tensor) -> torch.Tensor:
    """A point is valid iff neither of its two coordinates is NaN. Returns [E, M] bool."""
    return ~torch.isnan(points).any(dim=-1)


def _resample_stage(centerline, config) -> _ResampleResult:
    """Masked arc-length resample of the centerline per config.output_mode.

    fixed mode            -> num = config.num_points, count == num.
    constant_spacing mode -> spacing = config.spacing, padded to config.N_max with NaN.
    """
    points = centerline.points  # [E, M_max, 2]
    valid_mask = _valid_mask_from_points(points)  # [E, M_max]

    if config.output_mode == "fixed":
        resampled, count = geometry.arc_length_resample(
            points, num=config.num_points, valid_mask=valid_mask
        )
    elif config.output_mode == "constant_spacing":
        resampled, count = geometry.arc_length_resample(
            points, spacing=config.spacing, valid_mask=valid_mask, n_max=config.N_max
        )
    else:
        raise ValueError(f"Unknown output_mode: {config.output_mode!r}")

    return _ResampleResult(center=resampled, count=count)
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py -v
```

Expected: `test_resample_stage_circle_is_arc_uniform_and_on_circle PASSED`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add inflation.py tests/test_inflation.py
git commit -m "inflation: resample stage (masked arc-length resample, fixed/constant_spacing)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 23: Frame + curvature stage — tangents, normals, and Menger curvature

Add a `_frame_curvature_stage(center)` that returns `(T, Nrm, kappa)` by delegating to `geometry.tangents_normals` (unit central-difference tangent `T`, left-normal `Nrm`) and `geometry.menger_curvature` (non-negative). Test that the frame is orthonormal and that a radius-`r` circle has `kappa ≈ 1/r`.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/inflation.py`
- Modify: `/home/antoine/Documents/track_gen/tests/test_inflation.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_inflation.py`:

```python
def test_frame_curvature_orthonormal_and_circle_kappa():
    radius = 2.0
    cl = make_circle_centerline(radius=radius, m=200, e=2)
    cfg = fixed_config(num_points=256, num_envs=2)

    res = inflation._resample_stage(cl, cfg)
    T, Nrm, kappa = inflation._frame_curvature_stage(res.center)

    t_norm = torch.linalg.norm(T, dim=-1)  # [E, N]
    assert torch.allclose(t_norm, torch.ones_like(t_norm), atol=1e-4)
    n_norm = torch.linalg.norm(Nrm, dim=-1)
    assert torch.allclose(n_norm, torch.ones_like(n_norm), atol=1e-4)
    dot = (T * Nrm).sum(dim=-1)  # [E, N]
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-4)
    assert torch.allclose(kappa, torch.full_like(kappa, 1.0 / radius), atol=1e-2)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py::test_frame_curvature_orthonormal_and_circle_kappa -v
```

Expected: `AttributeError: module 'track_gen.inflation' has no attribute '_frame_curvature_stage'`.

- [ ] **Step 3 — Write the minimal implementation.** Add to `/home/antoine/Documents/track_gen/inflation.py` (after `_resample_stage`):

```python
def _frame_curvature_stage(center: torch.Tensor):
    """Compute the per-point frame and curvature on the resampled centerline.

    Returns:
        T:     [E, N, 2] unit tangent (central difference).
        Nrm:   [E, N, 2] unit left-normal, Nrm = (-T_y, T_x).
        kappa: [E, N]    non-negative Menger curvature.
    """
    T, Nrm = geometry.tangents_normals(center)
    kappa = geometry.menger_curvature(center)
    return T, Nrm, kappa
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py::test_frame_curvature_orthonormal_and_circle_kappa -v
```

Expected: `PASSED`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add inflation.py tests/test_inflation.py
git commit -m "inflation: frame + curvature stage (tangents_normals, menger_curvature)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 24: Width rule stage — curvature clamp + optional self-distance clamp

Add `_width_stage(center, kappa, config)` returning per-point half-width `w: [E, N]`:
1. `w_curv = where(kappa > eps, alpha / kappa, w_max)`, then `clamp_max(w_max)`.
2. If `config.clamp_self_distance`: `d = geometry.nearest_nonadjacent_distance(center, band, decimation)`; `w_self = 0.5 * (d - self_distance_margin)`; `w = minimum(w_curv, w_self)`.
3. Finally `clamp_min(0)`.

`w_max = config.half_width`, `alpha = config.alpha`. Tests: `w <= w_max` everywhere; on an ellipse `w * kappa < alpha < 1` everywhere (no fold); with the self-distance clamp ON, the half-width never exceeds half the local self-distance.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/inflation.py`
- Modify: `/home/antoine/Documents/track_gen/tests/test_inflation.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_inflation.py`:

```python
def make_ellipse_centerline(a=4.0, b=1.5, m=300, e=1, device="cpu"):
    """Closed ellipse Centerline; high curvature at the ends of the major axis."""
    theta = torch.linspace(0, 2 * math.pi, m + 1, device=device)[:-1]
    x = a * torch.cos(theta)
    y = b * torch.sin(theta)
    pts = torch.stack([x, y], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    valid = torch.ones(e, dtype=torch.bool, device=device)
    return Centerline(points=pts, valid=valid)


def make_near_touch_centerline(m=400, e=1, gap=0.2, device="cpu"):
    """A peanut/dumbbell loop whose two lobes pass within ~gap of each other."""
    t = torch.linspace(0, 2 * math.pi, m + 1, device=device)[:-1]
    r = 3.0 + 2.0 * torch.cos(2 * t)  # waist where cos(2t) is most negative
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    squeeze = gap + 0.5 * (x / x.abs().max())**2
    y = y * squeeze / (squeeze.abs().max())
    pts = torch.stack([x, y], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    valid = torch.ones(e, dtype=torch.bool, device=device)
    return Centerline(points=pts, valid=valid)


def test_width_bounded_by_w_max_on_circle():
    cl = make_circle_centerline(radius=5.0, m=200, e=1)
    cfg = fixed_config(num_points=256, num_envs=1, half_width=0.4, alpha=0.9,
                       clamp_self_distance=False)
    res = inflation._resample_stage(cl, cfg)
    _, _, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)
    assert w.shape == (1, 256)
    assert torch.all(w <= cfg.half_width + 1e-6)
    assert torch.allclose(w, torch.full_like(w, cfg.half_width), atol=1e-3)


def test_width_no_fold_on_ellipse():
    cl = make_ellipse_centerline(a=4.0, b=1.0, m=400, e=1)
    cfg = fixed_config(num_points=400, num_envs=1, half_width=2.0, alpha=0.9,
                       clamp_self_distance=False)
    res = inflation._resample_stage(cl, cfg)
    _, _, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)
    assert torch.all(w * kappa < cfg.alpha + 1e-4)
    assert torch.all(w * kappa < 1.0)


def test_self_distance_clamp_prevents_overlap_on_near_touch():
    cl = make_near_touch_centerline(m=600, e=1, gap=0.3)
    cfg = fixed_config(
        num_points=600, num_envs=1, half_width=1.0, alpha=0.9,
        clamp_self_distance=True, self_distance_margin=0.02,
        self_distance_band=8, self_distance_decimation=64,
    )
    res = inflation._resample_stage(cl, cfg)
    _, Nrm, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)
    outer = res.center + w.unsqueeze(-1) * Nrm
    inner = res.center - w.unsqueeze(-1) * Nrm
    d = geometry.nearest_nonadjacent_distance(
        res.center, cfg.self_distance_band, cfg.self_distance_decimation
    )
    assert torch.all(w <= 0.5 * d + 1e-5)
    assert torch.all(w >= 0.0)
    assert torch.isfinite(outer).all() and torch.isfinite(inner).all()
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py -k width -v
```

Expected: `AttributeError: module 'track_gen.inflation' has no attribute '_width_stage'`.

- [ ] **Step 3 — Write the minimal implementation.** Add to `/home/antoine/Documents/track_gen/inflation.py`:

```python
def _width_stage(center: torch.Tensor, kappa: torch.Tensor, config, eps: float = 1e-8):
    """Per-point half-width via curvature clamp + optional self-distance clamp.

    Args:
        center: [E, N, 2] resampled centerline.
        kappa:  [E, N]    non-negative curvature.
        config: TrackGenConfig (half_width, alpha, clamp_self_distance,
                self_distance_margin, self_distance_band, self_distance_decimation).
    Returns:
        w: [E, N] non-negative half-width.
    """
    w_max = float(config.half_width)
    alpha = float(config.alpha)

    w_curv = torch.where(
        kappa > eps,
        alpha / kappa.clamp_min(eps),
        torch.full_like(kappa, w_max),
    )
    w = w_curv.clamp_max(w_max)

    if config.clamp_self_distance:
        d = geometry.nearest_nonadjacent_distance(
            center, config.self_distance_band, config.self_distance_decimation
        )  # [E, N]
        w_self = 0.5 * (d - float(config.self_distance_margin))
        w = torch.minimum(w, w_self)

    return w.clamp_min(0.0)
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py -k width -v
```

Expected: all three width tests `PASSED`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add inflation.py tests/test_inflation.py
git commit -m "inflation: width stage (curvature clamp + optional self-distance clamp)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 25: Offset + orientation stage — assign outer/inner by larger absolute area

Add `_offset_stage(center, Nrm, w)` that builds the two candidate offset polylines `a = center + w*Nrm` and `b = center - w*Nrm`, then assigns `outer` to whichever of `{a, b}` has the **larger** `|polygon_area|` and `inner` to the smaller — per env, vectorized.

**Finding applied (NaN-safe area for constant_spacing):** `polygon_area` over a centerline containing NaN padding returns NaN, which would make the area comparison unreliable in constant_spacing mode. Compute the area on a NaN-zeroed copy of each candidate (`torch.nan_to_num(..., nan=0.0)`) before comparing magnitudes, so padded slots cannot poison the per-env area. (Fixed-mode tracks have no padding, so this is a no-op there.)

Test on a circle: `|area(outer)| > |area(center)| > |area(inner)|`.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/inflation.py`
- Modify: `/home/antoine/Documents/track_gen/tests/test_inflation.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_inflation.py`:

```python
def test_offset_orientation_outer_bigger_inner_smaller():
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=4)
    cfg = fixed_config(num_points=256, num_envs=4, half_width=0.5, alpha=0.9,
                       clamp_self_distance=False)
    res = inflation._resample_stage(cl, cfg)
    _, Nrm, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)

    outer, inner = inflation._offset_stage(res.center, Nrm, w)

    assert outer.shape == res.center.shape
    assert inner.shape == res.center.shape

    a_outer = geometry.polygon_area(outer).abs()   # [E]
    a_center = geometry.polygon_area(res.center).abs()
    a_inner = geometry.polygon_area(inner).abs()

    assert torch.all(a_outer > a_center)
    assert torch.all(a_center > a_inner)
    w_scalar = cfg.half_width
    assert torch.allclose(a_outer, torch.full_like(a_outer, math.pi * (radius + w_scalar) ** 2), atol=1e-1)
    assert torch.allclose(a_inner, torch.full_like(a_inner, math.pi * (radius - w_scalar) ** 2), atol=1e-1)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py::test_offset_orientation_outer_bigger_inner_smaller -v
```

Expected: `AttributeError: module 'track_gen.inflation' has no attribute '_offset_stage'`.

- [ ] **Step 3 — Write the minimal implementation.** Add to `/home/antoine/Documents/track_gen/inflation.py`:

```python
def _offset_stage(center: torch.Tensor, Nrm: torch.Tensor, w: torch.Tensor):
    """Offset the centerline by +/- w along the left-normal and assign outer/inner.

    outer = the candidate with the LARGER |polygon_area|; inner = the smaller.
    Robust to loop orientation. Areas are computed on NaN-zeroed copies so
    constant_spacing padding (NaN slots) cannot poison the per-env area.

    Args:
        center: [E, N, 2]
        Nrm:    [E, N, 2] unit left-normal
        w:      [E, N]    half-width
    Returns:
        outer: [E, N, 2], inner: [E, N, 2]
    """
    wn = w.unsqueeze(-1) * Nrm  # [E, N, 2]
    a = center + wn
    b = center - wn

    area_a = geometry.polygon_area(torch.nan_to_num(a, nan=0.0)).abs()  # [E]
    area_b = geometry.polygon_area(torch.nan_to_num(b, nan=0.0)).abs()  # [E]

    a_is_outer = (area_a >= area_b).view(-1, 1, 1)  # [E, 1, 1] for broadcasting
    outer = torch.where(a_is_outer, a, b)
    inner = torch.where(a_is_outer, b, a)
    return outer, inner
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py::test_offset_orientation_outer_bigger_inner_smaller -v
```

Expected: `PASSED`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add inflation.py tests/test_inflation.py
git commit -m "inflation: offset stage (outer/inner by larger |polygon_area|, NaN-safe)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 26: Validity stage — turning number, width floor, NaN, and generation validity

Add `_validity_stage(center, w, count, gen_valid, config)` returning `valid: [E] bool`. A track is valid iff ALL of:
1. `gen_valid` (the centerline's own `valid` flag), AND
2. `|turning_number(center)| ≈ 2π` within `config.turning_tol`, AND
3. `w > config.w_floor` at every **real** point (slot index `< count`), AND
4. no NaN among the real points of `center`.

In fixed mode every slot is real (`count == N`). The masking still works for `constant_spacing` mode where trailing slots are padding. `config.w_floor` is read here (now a real `TrackGenConfig` field). Tests: a self-crossing centerline → `valid False`; a clean circle → `valid True`; a forced-False generation flag propagates.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/inflation.py`
- Modify: `/home/antoine/Documents/track_gen/tests/test_inflation.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_inflation.py`:

```python
def make_figure_eight_centerline(scale=2.0, m=400, e=1, device="cpu"):
    """A self-crossing figure-eight (lemniscate): turning number ~ 0, not +/- 2pi."""
    t = torch.linspace(0, 2 * math.pi, m + 1, device=device)[:-1]
    x = scale * torch.sin(t)
    y = scale * torch.sin(t) * torch.cos(t)
    pts = torch.stack([x, y], dim=-1).unsqueeze(0).expand(e, m, 2).contiguous()
    valid = torch.ones(e, dtype=torch.bool, device=device)
    return Centerline(points=pts, valid=valid)


def _run_to_width(cl, cfg):
    res = inflation._resample_stage(cl, cfg)
    _, Nrm, kappa = inflation._frame_curvature_stage(res.center)
    w = inflation._width_stage(res.center, kappa, cfg)
    return res.center, Nrm, w, res.count


def test_validity_true_for_clean_circle():
    cl = make_circle_centerline(radius=3.0, m=300, e=2)
    cfg = fixed_config(num_points=256, num_envs=2, half_width=0.4, alpha=0.9,
                       clamp_self_distance=False, turning_tol=0.2, w_floor=1e-3)
    center, _, w, count = _run_to_width(cl, cfg)
    valid = inflation._validity_stage(center, w, count, cl.valid, cfg)
    assert valid.dtype == torch.bool
    assert valid.shape == (2,)
    assert torch.all(valid)


def test_validity_false_for_self_crossing():
    cl = make_figure_eight_centerline(scale=2.0, m=400, e=1)
    cfg = fixed_config(num_points=256, num_envs=1, half_width=0.2, alpha=0.9,
                       clamp_self_distance=False, turning_tol=0.2, w_floor=1e-3)
    center, _, w, count = _run_to_width(cl, cfg)
    valid = inflation._validity_stage(center, w, count, cl.valid, cfg)
    assert not bool(valid[0])


def test_validity_respects_gen_valid_flag():
    cl = make_circle_centerline(radius=3.0, m=300, e=2)
    cl.valid[1] = False
    cfg = fixed_config(num_points=256, num_envs=2, half_width=0.4, alpha=0.9,
                       clamp_self_distance=False, turning_tol=0.2, w_floor=1e-3)
    center, _, w, count = _run_to_width(cl, cfg)
    valid = inflation._validity_stage(center, w, count, cl.valid, cfg)
    assert bool(valid[0]) is True
    assert bool(valid[1]) is False
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py -k validity -v
```

Expected: `AttributeError: module 'track_gen.inflation' has no attribute '_validity_stage'`.

- [ ] **Step 3 — Write the minimal implementation.** Add to `/home/antoine/Documents/track_gen/inflation.py`:

```python
def _real_point_mask(count: torch.Tensor, n: int, device) -> torch.Tensor:
    """[E, N] bool mask: slot j is real iff j < count[env]. Fixed mode -> all True."""
    idx = torch.arange(n, device=device).unsqueeze(0)  # [1, N]
    return idx < count.unsqueeze(1)  # [E, N]


def _validity_stage(center, w, count, gen_valid, config) -> torch.Tensor:
    """Per-track validity: generation flag AND closed-loop turning AND width floor AND no-NaN.

    Args:
        center:    [E, N, 2]
        w:         [E, N]
        count:     [E]   number of real points per env.
        gen_valid: [E]   bool generation-time validity.
        config:    TrackGenConfig (turning_tol, w_floor).
    Returns:
        valid: [E] bool.
    """
    e, n = w.shape
    real = _real_point_mask(count, n, w.device)  # [E, N]

    turning = geometry.turning_number(center)  # [E]
    turn_ok = (turning.abs() - 2.0 * math.pi).abs() <= float(config.turning_tol)

    w_ok = torch.where(real, w > float(config.w_floor), torch.ones_like(real)).all(dim=1)

    nan_per_point = torch.isnan(center).any(dim=-1)  # [E, N]
    nan_real = (nan_per_point & real).any(dim=1)  # [E]
    no_nan = ~nan_real

    return gen_valid.to(torch.bool) & turn_ok & w_ok & no_nan
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py -k validity -v
```

Expected: all three validity tests `PASSED`.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add inflation.py tests/test_inflation.py
git commit -m "inflation: validity stage (gen flag + turning + width floor + no-NaN)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 27: Assemble — full `inflate()` returning a `Track` (both output modes)

Wire all stages into the public `inflate(centerline, config) -> Track`. Compute cumulative `arclen: [E, N]` and total `length: [E]` from the resampled (closed-loop) centerline, attach `tangent`/`normal`, `outer`/`center`/`inner`, `count`, and `valid`.

**Finding applied (closing wrap segment in constant_spacing mode):** the per-segment `real & roll(real,-1)` mask is False at index `count-1` (the next slot is padding), so the segment from the last real point back to point 0 would be dropped — under-reporting `length` and leaving `arclen` short by the closing segment. Add the explicit closing wrap segment: for each env gather its last real point (index `count-1`) and its first point (index 0), and add that segment's length into `length`. This matches the closed-loop definition and the fixed-mode result (where the wrap is already included because `real[0]` wraps at index `N-1`).

Both output modes: **fixed** (dense `N`, `count == N`, no padding) and **constant_spacing** (variable real count `≤ N_max`, trailing NaN-padded slots; `count` carries the real length).

**Files:**
- Modify: `/home/antoine/Documents/track_gen/inflation.py`
- Modify: `/home/antoine/Documents/track_gen/tests/test_inflation.py`

- [ ] **Step 1 — Write the FAILING test.** Append to `/home/antoine/Documents/track_gen/tests/test_inflation.py`:

```python
def test_inflate_fixed_mode_full_track():
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=3)
    cfg = fixed_config(num_points=128, num_envs=3, half_width=0.4, alpha=0.9,
                       clamp_self_distance=False, turning_tol=0.2, w_floor=1e-3)

    track = inflation.inflate(cl, cfg)

    assert isinstance(track, Track)
    for arr in (track.outer, track.center, track.inner, track.tangent, track.normal):
        assert arr.shape == (3, 128, 2)
    assert track.arclen.shape == (3, 128)
    assert track.length.shape == (3,)
    assert track.valid.shape == (3,)
    assert track.count.shape == (3,)

    assert torch.equal(track.count, torch.full((3,), 128, dtype=track.count.dtype))
    assert torch.all(track.valid)
    assert torch.isfinite(track.center).all()
    assert torch.isfinite(track.outer).all()
    assert torch.isfinite(track.inner).all()

    assert torch.allclose(track.arclen[:, 0], torch.zeros(3), atol=1e-6)
    assert torch.all(track.arclen[:, 1:] - track.arclen[:, :-1] >= -1e-6)
    assert torch.allclose(track.length, torch.full((3,), 2 * math.pi * radius), atol=1e-1)


def test_inflate_constant_spacing_mode_padding_and_wrap_length():
    radius = 3.0
    cl = make_circle_centerline(radius=radius, m=300, e=1)
    # Circumference ~ 18.85; spacing 0.5 -> ~38 real points, padded to N_max=128.
    cfg = TrackGenConfig(
        device="cpu", num_envs=1,
        output_mode="constant_spacing", spacing=0.5, N_max=128,
        half_width=0.3, alpha=0.9, clamp_self_distance=False,
        turning_tol=0.2, w_floor=1e-3,
    )

    track = inflation.inflate(cl, cfg)

    assert track.center.shape == (1, 128, 2)
    c = int(track.count[0].item())
    assert 0 < c <= 128
    assert torch.isfinite(track.center[0, :c]).all()
    if c < 128:
        assert torch.isnan(track.center[0, c:]).all()
    assert abs(c - round(2 * math.pi * radius / 0.5)) <= 2
    # The closing wrap segment must be included -> total length ~ circumference,
    # not short by one segment.
    assert torch.allclose(track.length, torch.full((1,), 2 * math.pi * radius), atol=0.6)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py -k inflate -v
```

Expected: `AttributeError: module 'track_gen.inflation' has no attribute 'inflate'`.

- [ ] **Step 3 — Write the minimal implementation.** Add the public `inflate` plus an arclen helper to `/home/antoine/Documents/track_gen/inflation.py` (`Track` is already imported from `.types`):

```python
def _arclength(center: torch.Tensor, count: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cumulative arc length [E, N] (0 at index 0) and closed-loop total length [E].

    Uses only real points: padded (NaN) slots contribute zero-length segments. The
    closing wrap segment (last real point -> point 0) is added explicitly so the
    total length matches the closed-loop definition in both output modes.
    """
    e, n, _ = center.shape
    real = _real_point_mask(count, n, center.device)  # [E, N]

    nxt = torch.roll(center, shifts=-1, dims=1)
    seg = nxt - center  # [E, N, 2]
    seg_len = torch.linalg.norm(seg, dim=-1)  # [E, N]
    real_next = torch.roll(real, shifts=-1, dims=1)
    seg_real = real & real_next  # [E, N]
    seg_len = torch.where(seg_real, seg_len, torch.zeros_like(seg_len))

    cum = torch.cumsum(seg_len, dim=1)  # length at i is sum of seg[0..i]
    arclen = torch.zeros_like(cum)
    arclen[:, 1:] = cum[:, :-1]

    # Closing wrap segment: last real point (index count-1) -> first point (index 0).
    # In fixed mode this is already captured by seg at index N-1; in constant_spacing
    # mode it is NOT (the next slot after count-1 is padding), so add it explicitly.
    last_idx = (count - 1).clamp_min(0)  # [E]
    first_pt = center[:, 0, :]  # [E, 2]
    last_pt = center[torch.arange(e, device=center.device), last_idx]  # [E, 2]
    wrap_already_counted = real_next.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # [E] bool
    wrap_len = torch.linalg.norm(first_pt - last_pt, dim=-1)  # [E]
    # Add the wrap only when it was NOT already counted as a real segment and count>=2.
    add_wrap = (~wrap_already_counted) & (count >= 2)
    wrap_contrib = torch.where(add_wrap, wrap_len, torch.zeros_like(wrap_len))

    length = seg_len.sum(dim=1) + wrap_contrib
    return arclen, length


def inflate(centerline, config) -> Track:
    """Inflate a dense Centerline into a Track (outer/center/inner + frame + metadata).

    Stages: resample -> frame+curvature -> width -> offset -> validity -> assemble.
    """
    res = _resample_stage(centerline, config)
    center, count = res.center, res.count

    T, Nrm, kappa = _frame_curvature_stage(center)
    w = _width_stage(center, kappa, config)
    outer, inner = _offset_stage(center, Nrm, w)
    valid = _validity_stage(center, w, count, centerline.valid, config)
    arclen, length = _arclength(center, count)

    return Track(
        outer=outer,
        center=center,
        inner=inner,
        tangent=T,
        normal=Nrm,
        arclen=arclen,
        length=length,
        valid=valid,
        count=count,
    )
```

- [ ] **Step 4 — Run it, expect PASS, then run the whole module.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_inflation.py -k inflate -v
python -m pytest tests -v
```

Expected: both `inflate` tests `PASSED`, and the full suite so far is green. Constant_spacing `outer`/`inner` inherit the centerline's NaN padding past `count` — that is expected.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add inflation.py tests/test_inflation.py
git commit -m "inflation: assemble full inflate() -> Track (arclen + closing-wrap length, both modes)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 28: Facade — `TrackGenerator(config, rng)` with `generate(num_or_ids) -> Track`

This REPLACES the existing `track_generator.py` with the new facade. It imports the warp-free dataclasses from `.types`, the generators from `.generators`, and `inflate` from `.inflation`, then defines `TrackGenerator` plus (next task) the `generate_tracks` compat shim. **It must NOT redefine `TrackGenConfig`/`Track`** — those now live in `types.py`; the facade re-exports them. Import ordering is safe: `.types` is a leaf, `.inflation` imports only `.types`+`.geometry`, so `from .inflation import inflate` at the top of `track_generator.py` cannot trigger a circular import (the old cycle is gone).

`PerEnvSeededRNG` is imported from the package top-level (`from . import PerEnvSeededRNG`), exactly as the legacy module did. The facade constructs the configured generator (`"bezier"`→`BezierCenterlineGenerator`, `"fourier"`→`FourierCenterlineGenerator`) at init, and `generate(num_or_ids)` runs that generator then `inflate()`. `num_or_ids` may be an `int` (mapped to ids `0..n-1`) or a tensor of explicit env ids. This task's test exercises the real warp-backed RNG and generators, so it is `pytest.importorskip("warp")`-guarded.

**Files:**
- Replace: `/home/antoine/Documents/track_gen/track_generator.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_track_generator_facade.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_track_generator_facade.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

pytest.importorskip("warp")

from track_gen import PerEnvSeededRNG
from track_gen.track_generator import Track, TrackGenConfig, TrackGenerator


def _make_rng(num_envs, device="cpu"):
    return PerEnvSeededRNG(seeds=0, num_envs=num_envs, device=device)


def test_bezier_path_returns_track_with_aligned_boundaries():
    E, N = 4, 64
    cfg = TrackGenConfig(generator="bezier", num_envs=E, num_points=N, device="cpu")
    rng = _make_rng(E)
    gen = TrackGenerator(cfg, rng)

    track = gen.generate(E)

    assert isinstance(track, Track)
    assert track.outer.shape == (E, N, 2)
    assert track.center.shape == (E, N, 2)
    assert track.inner.shape == (E, N, 2)
    assert track.valid.shape == (E,)
    assert track.valid.dtype == torch.bool


def test_fourier_generator_is_routed():
    E, N = 4, 64
    cfg = TrackGenConfig(generator="fourier", num_envs=E, num_points=N, device="cpu")
    rng = _make_rng(E)
    gen = TrackGenerator(cfg, rng)

    from track_gen.generators import FourierCenterlineGenerator

    assert isinstance(gen._generator, FourierCenterlineGenerator)

    track = gen.generate(E)
    assert isinstance(track, Track)
    assert track.center.shape == (E, N, 2)


def test_unknown_generator_raises():
    cfg = TrackGenConfig(generator="spline", num_envs=2, device="cpu")
    rng = _make_rng(2)
    with pytest.raises(ValueError):
        TrackGenerator(cfg, rng)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_track_generator_facade.py -v
```

Expected: with warp present, `ImportError: cannot import name 'TrackGenConfig' from 'track_gen.track_generator'` (the legacy module does not yet define the new facade / re-export the leaf types); SKIPPED if warp unavailable.

- [ ] **Step 3 — Write the minimal implementation.** REPLACE the entire contents of `/home/antoine/Documents/track_gen/track_generator.py` with the new facade:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level facade for the batched track generator.

Wires the configured centerline generator (Bezier or Fourier) to the inflation
stage and returns a fully-populated :class:`Track`. The public dataclasses
``TrackGenConfig`` and ``Track`` live in the dependency-free leaf module
``types.py`` and are re-exported here for backward compatibility.
"""

import warnings

import torch
from torch import Tensor

from . import PerEnvSeededRNG  # noqa: F401  (re-export; matches legacy import surface)
from .types import Track, TrackGenConfig
from .generators import (
    BezierCenterlineGenerator,
    Centerline,
    FourierCenterlineGenerator,
)
from .inflation import inflate

__all__ = [
    "Track",
    "TrackGenConfig",
    "TrackGenerator",
    "generate_tracks",
]


class TrackGenerator:
    """Top-level facade: build the configured centerline generator, run it,
    inflate the result, and return a :class:`Track`.
    """

    _GENERATORS = {
        "bezier": BezierCenterlineGenerator,
        "fourier": FourierCenterlineGenerator,
    }

    def __init__(self, config: TrackGenConfig, rng) -> None:
        """Args:
        config: The pipeline configuration.
        rng: A ``PerEnvSeededRNG`` instance for per-env reproducible sampling.
        """
        if rng is None:
            raise ValueError("A random number generator must be provided.")
        self._config = config
        self._rng = rng

        generator_cls = self._GENERATORS.get(config.generator)
        if generator_cls is None:
            raise ValueError(
                f"Unknown generator '{config.generator}'. "
                f"Expected one of {sorted(self._GENERATORS)}."
            )
        self._generator = generator_cls(config, rng)

    def _resolve_ids(self, num_or_ids) -> Tensor:
        """Map an int count to ids ``0..n-1``; pass a tensor of ids through."""
        if isinstance(num_or_ids, int):
            return torch.arange(num_or_ids, device=self._config.device)
        return num_or_ids

    def generate(self, num_or_ids) -> Track:
        """Generate a batch of tracks.

        Args:
            num_or_ids: Either an ``int`` number of tracks (ids ``0..n-1``) or a
                1D tensor of explicit environment ids.

        Returns:
            A fully-populated :class:`Track`.
        """
        ids = self._resolve_ids(num_or_ids)
        centerline: Centerline = self._generator.generate(ids)
        return inflate(centerline, self._config)
```

Note: `from . import PerEnvSeededRNG` re-imports the symbol already bound by the package `__init__`. Because `types`, `generators`, `inflation` are all imported AFTER (or are leaves not importing back into `track_generator`), there is no circular import. The `generate_tracks` shim is added in the next task.

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_track_generator_facade.py -v
```

Expected: three tests `PASSED` with warp available; cleanly `SKIPPED` otherwise.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add track_generator.py tests/test_track_generator_facade.py
git commit -m "Replace track_generator with facade routing to bezier/fourier + inflate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 29: `generate_tracks(num_tracks)` backward-compat shim

Provide a module-level `generate_tracks(num_tracks, config=None, rng=None)` shim that runs the new pipeline and returns centerline-only data shaped like the old API: a `[num_tracks, N, 2]` tensor of centerline points. It is documented as deprecated and emits a `DeprecationWarning`. Warp-guarded.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/track_generator.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_generate_tracks_compat.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_generate_tracks_compat.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

pytest.importorskip("warp")

from track_gen import PerEnvSeededRNG
from track_gen.track_generator import TrackGenConfig, generate_tracks


def test_compat_shim_returns_centerline_shaped_tensor():
    E, N = 5, 48
    cfg = TrackGenConfig(generator="bezier", num_envs=E, num_points=N, device="cpu")
    rng = PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")

    centerline = generate_tracks(E, config=cfg, rng=rng)

    assert isinstance(centerline, torch.Tensor)
    assert centerline.shape == (E, N, 2)


def test_compat_shim_emits_deprecation_warning():
    E, N = 3, 32
    cfg = TrackGenConfig(generator="bezier", num_envs=E, num_points=N, device="cpu")
    rng = PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")

    with pytest.warns(DeprecationWarning):
        generate_tracks(E, config=cfg, rng=rng)
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generate_tracks_compat.py -v
```

Expected: FAIL with `ImportError: cannot import name 'generate_tracks' from 'track_gen.track_generator'`, or SKIPPED if warp unavailable.

- [ ] **Step 3 — Write the minimal implementation.** Append the module-level shim to the end of `/home/antoine/Documents/track_gen/track_generator.py`:

```python
def generate_tracks(num_tracks: int, config: TrackGenConfig | None = None, rng=None) -> Tensor:
    """Deprecated backward-compatibility shim for the old centerline-only API.

    The legacy ``TrackGenerator.generate_tracks`` returned only centerline data.
    This shim runs the full pipeline and returns just the centerline points,
    shaped ``[num_tracks, N, 2]``, so existing callers keep working.

    .. deprecated::
        Use ``TrackGenerator(config, rng).generate(num_tracks).center`` instead.

    Args:
        num_tracks: Number of tracks to generate.
        config: Pipeline configuration. If ``None``, a default
            :class:`TrackGenConfig` with ``num_envs = num_tracks`` is used.
        rng: A ``PerEnvSeededRNG`` instance.

    Returns:
        The centerline points, shape ``[num_tracks, N, 2]``.
    """
    warnings.warn(
        "generate_tracks() is deprecated; use "
        "TrackGenerator(config, rng).generate(num_tracks).center instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if config is None:
        config = TrackGenConfig(num_envs=num_tracks)
    generator = TrackGenerator(config, rng)
    track = generator.generate(num_tracks)
    return track.center
```

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_generate_tracks_compat.py -v
```

Expected: both tests `PASSED` with warp available; cleanly `SKIPPED` otherwise.

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add track_generator.py tests/test_generate_tracks_compat.py
git commit -m "Add deprecated generate_tracks compat shim returning centerline points

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 30: Re-export public API from the package (`TrackGenerator`, dataclasses, generators)

Grow `__init__.py` to re-export the full public surface so callers can `from track_gen import TrackGenerator, TrackGenConfig, Track, Centerline`. This is the final wiring of the package's public API.

**Files:**
- Modify: `/home/antoine/Documents/track_gen/__init__.py`
- Test: `/home/antoine/Documents/track_gen/tests/test_public_api_full.py`

- [ ] **Step 1 — Write the FAILING test.** Create `/home/antoine/Documents/track_gen/tests/test_public_api_full.py`:

```python
def test_full_public_api_is_reexported():
    import track_gen

    for name in ("PerEnvSeededRNG", "TrackGenerator", "TrackGenConfig", "Track", "Centerline"):
        assert hasattr(track_gen, name), f"track_gen.{name} is not exported"
```

- [ ] **Step 2 — Run it, expect FAIL.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_public_api_full.py -v
```

Expected: FAIL — `AssertionError: track_gen.TrackGenerator is not exported`.

- [ ] **Step 3 — Write the minimal implementation.** Replace the body of `/home/antoine/Documents/track_gen/__init__.py` with:

```python
"""track_gen — GPU-batched race-track generator."""

__version__ = "0.1.0"

from .rng_utils import PerEnvSeededRNG
from . import geometry
from .geometry import (
    arc_length_resample,
    ccw_sort,
    menger_curvature,
    nearest_nonadjacent_distance,
    polygon_area,
    safe_normalize,
    segment_directions,
    tangents_normals,
    turning_number,
    vertex_tangents,
)
from .types import Track, TrackGenConfig
from .generators import (
    BezierCenterlineGenerator,
    Centerline,
    CenterlineGenerator,
    FourierCenterlineGenerator,
)
from .track_generator import TrackGenerator, generate_tracks

__all__ = [
    "PerEnvSeededRNG",
    "geometry",
    "safe_normalize",
    "polygon_area",
    "ccw_sort",
    "segment_directions",
    "vertex_tangents",
    "turning_number",
    "menger_curvature",
    "tangents_normals",
    "arc_length_resample",
    "nearest_nonadjacent_distance",
    "Track",
    "TrackGenConfig",
    "Centerline",
    "CenterlineGenerator",
    "BezierCenterlineGenerator",
    "FourierCenterlineGenerator",
    "TrackGenerator",
    "generate_tracks",
]
```

Note: this `__init__` imports `track_generator`, which imports `generators`/`inflation`/`types`. Because `types` is a leaf and `inflation` imports only `types`+`geometry`, the import graph is acyclic. `import track_gen` therefore succeeds with no circular import (verified in the integration task).

- [ ] **Step 4 — Run it, expect PASS.**

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_public_api_full.py -v
```

Expected: `1 passed` (with warp present, since the package root pulls in the RNG; SKIP-free because the test itself does not need warp at runtime — but importing the package does load `rng_utils`. If warp is unavailable on the dev box, this single test errors at import; that is acceptable for the dev environment which has warp. The CPU-only guarantee applies to the geometry/inflation/types tests, which import leaves directly.)

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add __init__.py tests/test_public_api_full.py
git commit -m "Re-export full public API (TrackGenerator, dataclasses, generators)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 31: Final integration / regression — no circular import + full suite green

This is the LAST task. It proves the whole package imports with no circular import, runs the entire test suite, and confirms the CPU/warp split behaves: all CPU tests pass, and warp-guarded tests SKIP cleanly when warp is unavailable (and pass when it is). Then commit.

**Files:**
- Test: `/home/antoine/Documents/track_gen/tests/test_integration.py`

- [ ] **Step 1 — Write the integration test.** Create `/home/antoine/Documents/track_gen/tests/test_integration.py`:

```python
import importlib

import torch


def test_package_imports_without_circular_import():
    # A clean import of the package must not raise (no circular import between
    # track_generator <-> inflation <-> types).
    import track_gen

    importlib.reload(track_gen)
    assert hasattr(track_gen, "TrackGenerator")
    assert hasattr(track_gen, "Centerline")
    assert hasattr(track_gen, "Track")
    assert hasattr(track_gen, "TrackGenConfig")


def test_inflate_runs_on_a_synthetic_centerline_without_warp():
    # End-to-end inflation on a hand-built circle, importing only warp-free leaves.
    import math

    from track_gen.generators import Centerline
    from track_gen.types import TrackGenConfig
    from track_gen import inflation

    theta = torch.linspace(0, 2 * math.pi, 201)[:-1]
    pts = torch.stack([2.0 * torch.cos(theta), 2.0 * torch.sin(theta)], dim=-1).unsqueeze(0)
    cl = Centerline(points=pts, valid=torch.ones(1, dtype=torch.bool))
    cfg = TrackGenConfig(num_envs=1, num_points=128, output_mode="fixed", clamp_self_distance=False)

    track = inflation.inflate(cl, cfg)
    assert track.center.shape == (1, 128, 2)
    assert bool(track.valid[0])
```

- [ ] **Step 2 — Run the no-circular-import smoke check + the full suite.** First prove the bare import works, then run everything:

```bash
cd /home/antoine/Documents/track_gen
python -c "import track_gen; print('import OK:', bool(track_gen.TrackGenerator))"
python -m pytest tests -v
```

Expected:
- `python -c "import track_gen"` prints `import OK: True` with NO `ImportError`/`cannot import name`/circular-import traceback.
- `python -m pytest tests -v` reports every test passing or skipped, `0 failed`. With warp installed (the dev environment), all generator/facade/compat tests PASS. On a machine WITHOUT warp, those warp-guarded tests show `SKIPPED` (reason: `could not import 'warp'`) while every geometry/types/inflation CPU test still PASSES. Confirm the summary line shows `0 failed` and the only non-passing entries (if any) are `skipped`.

- [ ] **Step 3 — No implementation change needed.** This task is a verification gate. If `import track_gen` raises a circular-import or any test fails, fix the offending module per the contract (the leaf `types.py` must hold `Track`/`TrackGenConfig`; `inflation.py` must import them from `.types`, never from `.track_generator`) and re-run until green.

- [ ] **Step 4 — Confirm the CPU-only subset in isolation (optional but recommended).** To demonstrate the no-warp guarantee explicitly, run only the leaf-importing tests; these must pass even if warp were absent:

```bash
cd /home/antoine/Documents/track_gen
python -m pytest tests/test_types.py tests/test_inflation.py tests/test_geometry_safe_normalize.py tests/test_geometry_polygon_area.py tests/test_geometry_ccw_sort.py tests/test_geometry_segment_directions.py tests/test_geometry_vertex_tangents.py tests/test_geometry_turning_number.py tests/test_geometry_menger_curvature.py tests/test_geometry_tangents_normals.py tests/test_geometry_arc_length_resample.py tests/test_geometry_nearest_nonadjacent.py -v
```

Expected: all PASS, `0 failed`, `0 skipped` (none of these import warp).

- [ ] **Step 5 — Commit.**

```bash
cd /home/antoine/Documents/track_gen
git add tests/test_integration.py
git commit -m "Add final integration test: no circular import + full-suite regression gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Deferred (YAGNI)

Per-env sampling of the `rad`, `edgy`, and `half_width` ranges is intentionally deferred. For now these stay scalars on `TrackGenConfig` (a single value applied to every env). If per-env variety in handle tightness, edginess, or track width is needed later, promote each to an optional `(low, high)` range sampled with `PerEnvSeededRNG` inside the relevant generator/inflation stage — the plumbing (per-env seeded RNG, `ids`-threaded sampling) already exists, so this is an additive change with no rework of the current scalar path.
