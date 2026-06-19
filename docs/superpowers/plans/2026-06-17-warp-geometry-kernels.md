# Warp Geometry Kernels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `self_intersections`, `separation_min`, `curvature_radius_min`, and `thickness` from `track_gen/geometry.py` into pure-Warp kernels in `track_gen/warp_pipeline.py`, verified `torch.equal` (integers) or `allclose(atol=1e-4)` (floats) against the torch oracle on both CPU and CUDA Warp devices.

**Architecture:** Each heavy computation uses ONE thread per env (`e = tid`; dim = E), looping over point pairs inside the kernel. This avoids atomics for `self_intersections` (local counter, write once), and lets per-env `band` values be read per-thread. Warp helper `@wp.func` functions handle orientation tests and distance math. All kernels follow the existing `warp_pipeline.py` convention: `wp.from_torch(...contiguous(), dtype=...)`, `wp.launch(device=str(dev))`, `_sync(dev)`.

**Tech Stack:** NVIDIA Warp 1.14 (`wp.kernel`, `wp.func`, `wp.vec2f`, `wp.length`, `wp.abs`, `wp.min`, `wp.max`, `wp.from_torch`), PyTorch (I/O + oracle), pytest. Branch: `feat/pure-warp-pipeline` (do not create a new branch, commit to this one).

**Key files:**
- Modify: `track_gen/warp_pipeline.py` — add all new kernels and wrappers inside `if _HAVE_WARP:` guard
- Create: `tests/test_warp_geom_gates.py` — gate tests matching oracle

---

## Task 1: Create the failing test file

**Files:**
- Create: `tests/test_warp_geom_gates.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_warp_geom_gates.py` with exactly this content:

```python
import math, pytest, torch
pytest.importorskip("warp")
from track_gen import warp_pipeline as wpl
from track_gen import geometry
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

def _circle(n=256,r=2.0,dev="cpu"):
    t=torch.linspace(0,2*math.pi,n+1,device=dev)[:-1]
    return torch.stack([r*torch.cos(t),r*torch.sin(t)],-1).unsqueeze(0)
def _fig8(n=256,s=1.0,dev="cpu"):
    t=torch.linspace(0,2*math.pi,n+1,device=dev)[:-1]
    return torch.stack([s*torch.sin(t),s*torch.sin(t)*torch.cos(t)],-1).unsqueeze(0)

@pytest.mark.parametrize("dev", DEVS)
def test_self_intersections_matches(dev):
    poly=torch.cat([_circle(64,1.0,dev), _fig8(64,1.0,dev)],0)
    got=wpl.self_intersections(poly); ref=geometry.self_intersections(poly)
    assert torch.equal(got.cpu(), ref.cpu())

@pytest.mark.parametrize("dev", DEVS)
def test_thickness_matches(dev):
    torch.manual_seed(0)
    c=(torch.randn(6,256,2,device=dev)*0.7)
    band=torch.randint(2,10,(6,),device=dev)
    assert torch.allclose(wpl.thickness(c,band), geometry.thickness(c,band), atol=1e-4)

@pytest.mark.parametrize("dev", DEVS)
def test_thickness_circle(dev):
    c=_circle(400,2.0,dev); band=torch.tensor([400//2-2],device=dev)
    assert torch.allclose(wpl.thickness(c,band), torch.tensor([2.0],device=dev), atol=2e-2)
```

- [ ] **Step 2: Run the tests and verify they fail with NameError / AttributeError (functions absent)**

```bash
cd /home/antoiner/Documents/TrackGen && .venv/bin/python -m pytest tests/test_warp_geom_gates.py -q 2>&1 | head -30
```

Expected: FAIL — `AttributeError: module 'track_gen.warp_pipeline' has no attribute 'self_intersections'`

---

## Task 2: Implement `_self_intersections_k` kernel and `self_intersections` wrapper

**Files:**
- Modify: `track_gen/warp_pipeline.py`

**Algorithm:** One thread per env (`tid = e`). Inner loops: `i` in `[0, N)`, `j` in `[i+1, N)`. Skip pair if `min(|i-j|, N-|i-j|) <= 1`. Orientation test using a helper `@wp.func _ccw(o, p, q) -> float`. Proper crossing if `((d1>0)!=(d2>0)) and ((d3>0)!=(d4>0))`. Count locally, write `out[e]`.

**Orientation test formula:** `_ccw(o, p, q) = (q.y - o.y)*(p.x - o.x) - (p.y - o.y)*(q.x - o.x)`

**Segment naming for pair (i, j):**
- `Ai = poly[e, i]`, `Bi = poly[e, (i+1)%N]` (first segment endpoints)
- `Aj = poly[e, j]`, `Bj = poly[e, (j+1)%N]` (second segment endpoints)
- `d1 = _ccw(Aj, Bj, Ai)`, `d2 = _ccw(Aj, Bj, Bi)` — Ai, Bi vs segment j
- `d3 = _ccw(Ai, Bi, Aj)`, `d4 = _ccw(Ai, Bi, Bj)` — Aj, Bj vs segment i

- [ ] **Step 1: Add `_ccw` helper and `_self_intersections_k` kernel inside the `if _HAVE_WARP:` block in `warp_pipeline.py`**

Add after the last kernel definition (`_frame_k`) and before the `offset()` wrapper function. The `if _HAVE_WARP:` block is already open — just add more kernel definitions inside it:

```python
    @wp.func
    def _ccw(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
        # Returns (q.y-o.y)*(p.x-o.x) - (p.y-o.y)*(q.x-o.x)
        return (qy - oy) * (px - ox) - (py - oy) * (qx - ox)

    @wp.kernel
    def _self_intersections_k(
        poly: wp.array2d(dtype=wp.vec2f),  # [E*N] flat, indexed as poly[e*N+i]
        N: int,
        out: wp.array(dtype=wp.int32),
    ):
        # One thread per env e. Loops all unique pairs (i,j) with j > i,
        # skips if circular index distance <= 1, counts proper crossings.
        e = wp.tid()
        count = int(0)
        for i in range(N):
            for j in range(i + 1, N):
                # Circular index distance
                diff = j - i
                circ_dist = wp.min(diff, N - diff)
                if circ_dist <= 1:
                    continue
                Ai = poly[e * N + i]
                Bi = poly[e * N + (i + 1) % N]
                Aj = poly[e * N + j]
                Bj = poly[e * N + (j + 1) % N]
                d1 = _ccw(Aj[0], Aj[1], Bj[0], Bj[1], Ai[0], Ai[1])
                d2 = _ccw(Aj[0], Aj[1], Bj[0], Bj[1], Bi[0], Bi[1])
                d3 = _ccw(Ai[0], Ai[1], Bi[0], Bi[1], Aj[0], Aj[1])
                d4 = _ccw(Ai[0], Ai[1], Bi[0], Bi[1], Bj[0], Bj[1])
                seg_ij = (d1 > 0.0) != (d2 > 0.0)
                seg_ji = (d3 > 0.0) != (d4 > 0.0)
                if seg_ij and seg_ji:
                    count = count + 1
        out[e] = count
```

- [ ] **Step 2: Add `self_intersections` wrapper function after the `frame_curvature` wrapper**

```python
def self_intersections(poly: torch.Tensor) -> torch.Tensor:
    """Count proper self-crossings of each closed polyline. poly [E, N, 2] -> [E] long.

    Matches geometry.self_intersections exactly (torch.equal). Pure Warp (cpu+cuda).
    One thread per env; O(N^2) loop over edge pairs inside the kernel.
    """
    _init()
    E, N, _ = poly.shape
    dev = str(poly.device)

    # Reshape to [E*N, 2] then view as wp.array2d indexed by flat index e*N+i.
    # We pass as a 1D array of vec2f and index manually.
    flat = poly.reshape(E * N, 2).contiguous()
    wp_poly = wp.from_torch(flat, dtype=wp.vec2f)

    out_t = torch.zeros(E, device=poly.device, dtype=torch.int32)
    wp_out = wp.from_torch(out_t, dtype=wp.int32)

    wp.launch(_self_intersections_k, dim=E,
              inputs=[wp_poly, N, wp_out],
              device=dev)
    _sync(poly.device)
    return out_t.long()
```

Note: `wp.array2d` in the kernel signature requires the input to actually be 2D. Since we're using a 1D array of `wp.vec2f` and indexing by `e*N+i`, keep the kernel signature as `wp.array(dtype=wp.vec2f)` (not `wp.array2d`). Revise:

The kernel should use `wp.array(dtype=wp.vec2f)` (1D flat array, indexed as `poly[e*N+i]`):

```python
    @wp.kernel
    def _self_intersections_k(
        poly: wp.array(dtype=wp.vec2f),
        N: int,
        out: wp.array(dtype=wp.int32),
    ):
        e = wp.tid()
        count = int(0)
        for i in range(N):
            for j in range(i + 1, N):
                diff = j - i
                circ_dist = wp.min(diff, N - diff)
                if circ_dist <= 1:
                    continue
                Ai = poly[e * N + i]
                Bi = poly[e * N + (i + 1) % N]
                Aj = poly[e * N + j]
                Bj = poly[e * N + (j + 1) % N]
                d1 = _ccw(Aj[0], Aj[1], Bj[0], Bj[1], Ai[0], Ai[1])
                d2 = _ccw(Aj[0], Aj[1], Bj[0], Bj[1], Bi[0], Bi[1])
                d3 = _ccw(Ai[0], Ai[1], Bi[0], Bi[1], Aj[0], Aj[1])
                d4 = _ccw(Ai[0], Ai[1], Bi[0], Bi[1], Bj[0], Bj[1])
                seg_ij = (d1 > 0.0) != (d2 > 0.0)
                seg_ji = (d3 > 0.0) != (d4 > 0.0)
                if seg_ij and seg_ji:
                    count = count + 1
        out[e] = count
```

- [ ] **Step 3: Run `test_self_intersections_matches` and verify it passes**

```bash
cd /home/antoiner/Documents/TrackGen && .venv/bin/python -m pytest tests/test_warp_geom_gates.py::test_self_intersections_matches -v 2>&1 | tail -20
```

Expected: PASS for all parametrized devices.

---

## Task 3: Implement `_thickness_k` kernel and wrappers

**Files:**
- Modify: `track_gen/warp_pipeline.py`

**Algorithm (one thread per env, e = tid):**
1. `sep_min`: initialize to `1e30`. Loop `i` in `[0, N)`, `j` in `[i+1, N)`. Skip if `min(|i-j|, N-|i-j|) <= band[e]`. Compute `d = wp.length(points[e*N+i] - points[e*N+j])`. Update `sep_min = wp.min(sep_min, d)`. (Note: `j > i` only counts half the pairs, but since distance is symmetric, `dmat.amin` in torch also sees both (i,j) and (j,i) — we must use ALL pairs, not just j>i, OR recognize that min over j>i equals min over all pairs. The min is the same, so j>i is fine.)
2. `rad_min`: compute max Menger curvature over all i, then `rad_min = 1/kappa_max`. For each i, triple is `(i-1, i, i+1)` with wrap. Menger: `a = pts[i] - pts[i-1]`, `b = pts[i+1] - pts[i]`, `c = pts[i+1] - pts[i-1]`. `cross = a[0]*b[1] - a[1]*b[0]`. `area = 0.5*abs(cross)`. `denom = max(len_a*len_b*len_c, 1e-12)`. `kappa = 4*area/denom`.
3. `out[e] = min(rad_min, 0.5 * sep_min)`.

- [ ] **Step 1: Add `_thickness_k` kernel inside the `if _HAVE_WARP:` block**

```python
    @wp.kernel
    def _thickness_k(
        pts: wp.array(dtype=wp.vec2f),
        band: wp.array(dtype=wp.int32),
        N: int,
        out: wp.array(dtype=wp.float32),
    ):
        # One thread per env e.
        e = wp.tid()
        b = band[e]

        # --- sep_min: min dist over pairs with circ_dist > band ---
        sep_min = float(1.0e30)
        for i in range(N):
            for j in range(i + 1, N):
                diff = j - i
                circ_dist = wp.min(diff, N - diff)
                if circ_dist > b:
                    pi = pts[e * N + i]
                    pj = pts[e * N + j]
                    d = wp.length(pi - pj)
                    sep_min = wp.min(sep_min, d)

        # --- rad_min: 1 / max Menger curvature ---
        kappa_max = float(0.0)
        for i in range(N):
            xp = pts[e * N + (i + N - 1) % N]
            xc = pts[e * N + i]
            xn = pts[e * N + (i + 1) % N]
            a = xc - xp
            bb = xn - xc
            cc = xn - xp
            cross = a[0] * bb[1] - a[1] * bb[0]
            area = 0.5 * wp.abs(cross)
            denom = wp.max(wp.length(a) * wp.length(bb) * wp.length(cc), float(1.0e-12))
            kappa = 4.0 * area / denom
            kappa_max = wp.max(kappa_max, kappa)
        rad_min = 1.0 / wp.max(kappa_max, float(1.0e-12))

        out[e] = wp.min(rad_min, 0.5 * sep_min)
```

- [ ] **Step 2: Add `thickness`, `separation_min`, and `curvature_radius_min` wrappers**

Add after the `self_intersections` wrapper:

```python
def thickness(points: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Discrete curve thickness = min(curvature_radius_min, 0.5*separation_min).

    points [E, N, 2] float32; band [E] int (per-env exclusion window). Returns [E] float32.
    Matches geometry.thickness to allclose(atol=1e-4). Pure Warp (cpu+cuda).
    """
    _init()
    E, N, _ = points.shape
    dev = str(points.device)

    flat = points.reshape(E * N, 2).contiguous()
    wp_pts = wp.from_torch(flat, dtype=wp.vec2f)
    band_i32 = band.to(torch.int32).contiguous()
    wp_band = wp.from_torch(band_i32, dtype=wp.int32)

    out_t = torch.empty(E, device=points.device, dtype=torch.float32)
    wp_out = wp.from_torch(out_t, dtype=wp.float32)

    wp.launch(_thickness_k, dim=E,
              inputs=[wp_pts, wp_band, N, wp_out],
              device=dev)
    _sync(points.device)
    return out_t


def separation_min(points: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Min Euclidean distance over pairs with circ-index-dist > band. [E]."""
    _init()
    E, N, _ = points.shape
    dev = str(points.device)

    flat = points.reshape(E * N, 2).contiguous()
    wp_pts = wp.from_torch(flat, dtype=wp.vec2f)
    band_i32 = band.to(torch.int32).contiguous()
    wp_band = wp.from_torch(band_i32, dtype=wp.int32)

    out_t = torch.empty(E, device=points.device, dtype=torch.float32)
    wp_out = wp.from_torch(out_t, dtype=wp.float32)

    wp.launch(_sep_min_k, dim=E,
              inputs=[wp_pts, wp_band, N, wp_out],
              device=dev)
    _sync(points.device)
    return out_t


def curvature_radius_min(points: torch.Tensor) -> torch.Tensor:
    """1 / max Menger curvature over the loop. points [E, N, 2] -> [E]."""
    _init()
    E, N, _ = points.shape
    dev = str(points.device)

    flat = points.reshape(E * N, 2).contiguous()
    wp_pts = wp.from_torch(flat, dtype=wp.vec2f)

    out_t = torch.empty(E, device=points.device, dtype=torch.float32)
    wp_out = wp.from_torch(out_t, dtype=wp.float32)

    wp.launch(_curvrad_min_k, dim=E,
              inputs=[wp_pts, N, wp_out],
              device=dev)
    _sync(points.device)
    return out_t
```

Also add two small helper kernels for `separation_min` and `curvature_radius_min` inside `if _HAVE_WARP:`:

```python
    @wp.kernel
    def _sep_min_k(
        pts: wp.array(dtype=wp.vec2f),
        band: wp.array(dtype=wp.int32),
        N: int,
        out: wp.array(dtype=wp.float32),
    ):
        e = wp.tid()
        b = band[e]
        sep_min = float(1.0e30)
        for i in range(N):
            for j in range(i + 1, N):
                diff = j - i
                circ_dist = wp.min(diff, N - diff)
                if circ_dist > b:
                    pi = pts[e * N + i]
                    pj = pts[e * N + j]
                    d = wp.length(pi - pj)
                    sep_min = wp.min(sep_min, d)
        out[e] = sep_min

    @wp.kernel
    def _curvrad_min_k(
        pts: wp.array(dtype=wp.vec2f),
        N: int,
        out: wp.array(dtype=wp.float32),
    ):
        e = wp.tid()
        kappa_max = float(0.0)
        for i in range(N):
            xp = pts[e * N + (i + N - 1) % N]
            xc = pts[e * N + i]
            xn = pts[e * N + (i + 1) % N]
            a = xc - xp
            bb = xn - xc
            cc = xn - xp
            cross = a[0] * bb[1] - a[1] * bb[0]
            area = 0.5 * wp.abs(cross)
            denom = wp.max(wp.length(a) * wp.length(bb) * wp.length(cc), float(1.0e-12))
            kappa = 4.0 * area / denom
            kappa_max = wp.max(kappa_max, kappa)
        out[e] = 1.0 / wp.max(kappa_max, float(1.0e-12))
```

- [ ] **Step 3: Run all gate tests**

```bash
cd /home/antoiner/Documents/TrackGen && .venv/bin/python -m pytest tests/test_warp_geom_gates.py -v 2>&1 | tail -30
```

Expected: all tests pass on CPU (and CUDA if available).

---

## Task 4: Debug any failures

**Potential issues and fixes:**

1. **`wp.min` with int args**: In Warp kernels, `wp.min` may need explicit type. If `N - diff` is negative (impossible since j > i, so diff = j-i <= N-1, N-diff >= 1), not an issue. But ensure `diff` and `N - diff` are both typed as `int`. Use `wp.min(diff, N - diff)` — these are Python `int` inside warp kernels (wp.int32 arithmetic).

2. **`bool` XOR in Warp**: `(d1 > 0.0) != (d2 > 0.0)` — Warp supports `!=` on bools returned from comparisons. If this fails, use: `seg_ij = int(d1 > 0.0) + int(d2 > 0.0) == 1` (XOR as "exactly one is positive").

3. **`sep_min` when no valid pairs exist** (band >= N//2): `1e30` is correct — `geometry.separation_min` returns `+inf` for these. But `torch.allclose` with `+inf` values: two `+inf` values ARE equal. However `1e30 != inf`. If band is large enough that no pair satisfies `circ_dist > band`, the Warp version returns `1e30` while the torch version returns `inf`. Fix: after the kernel, replace any `>= 1e29` values with `float("inf")`:

```python
    out_t[out_t >= 1e29] = float("inf")
```

Apply this fix in `thickness` wrapper too (for the `sep_min` component), OR initialize `sep_min = float("inf")` in Warp (use a very large value). Actually `float("inf")` in Warp kernel context: use `float(1.0e38)` or check if `wp.inf` exists. In Warp 1.14, you can use `wp.float32(float('inf'))` — but safest is `float(3.4028235e+38)` (max float32). Then in the wrapper, no fixup needed since `thickness` takes `min(rad_min, 0.5*sep_min)` and `0.5*1e38` >> any curvature radius, so it doesn't affect thickness. But `separation_min` wrapper should return actual `inf`:

```python
    out_t[out_t >= 1e29] = float("inf")
```

4. **`_thickness_k` must also apply the `1e30` → `inf` fixup if tested directly**: The test `test_thickness_matches` uses `geometry.thickness` as oracle. When `sep_min = inf` and `rad_min = some_value`, `min(rad_min, 0.5*inf) = rad_min`. With `sep_min = 1e30`, `0.5*1e30 = 5e29 >> rad_min`, so `min(rad_min, 5e29) = rad_min`. This is numerically correct for `thickness`. **No fixup needed for `thickness`.**

5. **`allclose` failure on `test_thickness_circle`**: Circle with r=2 → Menger kappa = 1/r = 0.5 → `curvature_radius_min = 2.0`. `sep_min` with band = N//2-2 ≈ 198 on N=400: pairs with circ_dist > 198, i.e. pairs near opposite side of circle. Opposite points are distance ~2r=4 apart. `0.5 * 4 = 2.0`. So thickness = min(2.0, 2.0) = 2.0. Should match. If fails, check the `N//2 - 2` band vs exact formula.

---

## Task 5: Run full test suite and commit

**Files:** No new files.

- [ ] **Step 1: Run full test suite**

```bash
cd /home/antoiner/Documents/TrackGen && .venv/bin/python -m pytest -q 2>&1 | tail -15
```

Expected: gate tests pass; no regressions in existing tests. Record the total pass count.

- [ ] **Step 2: Commit**

```bash
cd /home/antoiner/Documents/TrackGen && git add track_gen/warp_pipeline.py tests/test_warp_geom_gates.py && git commit -m "warp_pipeline: self_intersections + thickness kernels == torch oracle"
```

Expected: commit succeeds on `feat/pure-warp-pipeline` branch.

---

## Self-Review

**Spec coverage:**
- `self_intersections(poly[E,N,2]) -> [E] long` ✓ — Task 2
- `separation_min(points, band) -> [E]` ✓ — Task 3 (`_sep_min_k` + wrapper)
- `curvature_radius_min(points) -> [E]` ✓ — Task 3 (`_curvrad_min_k` + wrapper)
- `thickness(points, band) -> [E]` ✓ — Task 3 (`_thickness_k` + wrapper)
- Tests on cpu + cuda ✓ — Task 1 (DEVS = ["cpu"] + cuda if available)
- `torch.equal` for `self_intersections` ✓
- `allclose(atol=1e-4)` for `thickness` ✓
- `atol=2e-2` for circle thickness ✓
- Commit on `feat/pure-warp-pipeline` ✓ — Task 5

**Placeholder scan:** No TBDs or vague steps.

**Type consistency:**
- `_self_intersections_k` takes `wp.array(dtype=wp.vec2f)` → wrapper passes `wp.from_torch(flat, dtype=wp.vec2f)` ✓
- `_thickness_k` takes `band: wp.array(dtype=wp.int32)` → wrapper does `.to(torch.int32)` ✓
- `separation_min` wrapper references `_sep_min_k` which is defined ✓
- `curvature_radius_min` wrapper references `_curvrad_min_k` which is defined ✓
