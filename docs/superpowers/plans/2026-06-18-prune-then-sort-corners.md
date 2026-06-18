# Prune-then-sort Corner Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate winding-0 figure-eight generation failures by angle-sorting the corners actually used (the first `count`) about *their own* centroid, instead of sorting all `P` candidates and truncating to a mis-centered angular wedge.

**Architecture:** A count-aware sort applied identically in the Warp runtime (`ccw_sort`/`_ccw_sort_k`, used by `generate_centerline_warp`) and the torch oracle (`geometry.ccw_sort_count`, used by `generators._prune_corners`). `ccw_sort(points, count=None)` stays byte-identical to today when `count is None`, so existing parity is preserved; the count-aware path NaN-pads rows `>= count`. `assemble` is unchanged (it already NaN-prunes `i >= count`).

**Tech Stack:** Python, PyTorch (oracle + array container), NVIDIA Warp (`@wp.kernel`/`wp.launch`), pytest.

**Context for the implementer:** The figure-eight bug: the corner pipeline does sort-then-prune in both the runtime and the oracle. `ccw_sort` orders all `P = max_num_points` corners by polar angle about the centroid of all `P`; a separately-sampled `count` then keeps only the first `count`. When `count < P` those are the `count` smallest-angle corners — a partial angular wedge ordered about the wrong centroid, whose long Bézier closing chord can overshoot into a self-crossing winding-0 loop. Fix = prune-then-sort: sort the kept subset about *its* centroid. Spec: `docs/superpowers/specs/2026-06-18-prune-then-sort-corners-design.md`.

**Conventions:** Run Python via `.venv/bin/python`. Tests run on the Warp `cpu` device (no GPU needed); cuda assertions are guarded by `torch.cuda.is_available()`. GPG signing is on; if a commit hangs on pinentry, append `--no-gpg-sign`. Out of scope: the equispace hard-guarantee variant, deep self-X (`rad` overshoot), coarser-spacing, the angle-gate skip (separate branch).

---

## File Structure

- `track_gen/geometry.py` — **add** `ccw_sort_count(points, count)` (torch oracle helper, beside `ccw_sort`).
- `track_gen/warp_pipeline.py` — **modify** `_ccw_sort_k` (count-aware) and the `ccw_sort` wrapper (optional `count` arg); **modify** `generate_centerline_warp` (sample count before sorting).
- `track_gen/generators.py` — **modify** `_prune_corners` (prune-then-sort via `ccw_sort_count`).
- `tests/test_geometry_ccw_sort.py` — **add** a `ccw_sort_count` unit test.
- `tests/test_warp_ccw_sort.py` — **add** a count-aware Warp-vs-oracle parity test.
- `tests/test_warp_corner_ordering.py` — **create**: behaviour test (figure-8 elimination + per-attempt yield direction).

---

### Task 1: Oracle helper `geometry.ccw_sort_count`

**Files:**
- Modify: `track_gen/geometry.py` (add a function after `ccw_sort`)
- Test: `tests/test_geometry_ccw_sort.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_geometry_ccw_sort.py`:

```python
import math
import torch
from track_gen import geometry


def test_ccw_sort_count_sorts_kept_and_nans_tail():
    # P=6 corners, count=4: first 4 sorted about THEIR centroid; last 2 -> NaN.
    pts = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [5., 5.], [6., 6.]]])  # [1,6,2]
    count = torch.tensor([4])
    out = geometry.ccw_sort_count(pts, count)

    assert out.shape == (1, 6, 2)
    assert torch.isnan(out[0, 4:]).all()       # pruned tail
    assert torch.isfinite(out[0, :4]).all()    # kept rows finite

    # kept rows are a permutation of the first 4 inputs
    kept_in = pts[0, :4].sort(dim=0).values
    kept_out = out[0, :4].sort(dim=0).values
    assert torch.allclose(kept_out, kept_in)

    # kept rows are angularly monotone about their own centroid
    c = out[0, :4].mean(dim=0)
    d = out[0, :4] - c
    ang = torch.arctan2(d[:, 0], d[:, 1])
    assert (ang[1:] - ang[:-1] >= -1e-6).all()


def test_ccw_sort_count_full_count_matches_ccw_sort():
    # count == P sorts everything; equals plain ccw_sort (no pruned tail).
    pts = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]])  # [1,4,2]
    count = torch.tensor([4])
    out = geometry.ccw_sort_count(pts, count)
    ref = geometry.ccw_sort(pts)
    assert torch.equal(out, ref)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_geometry_ccw_sort.py::test_ccw_sort_count_sorts_kept_and_nans_tail -v`
Expected: FAIL with `AttributeError: module 'track_gen.geometry' has no attribute 'ccw_sort_count'`.

- [ ] **Step 3: Implement `ccw_sort_count`**

Add to `track_gen/geometry.py` immediately after the existing `ccw_sort`:

```python
def ccw_sort_count(points: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Angle-sort each env's FIRST ``count`` corners about THEIR OWN centroid; NaN the rest.

    Prune-then-sort. Unlike :func:`ccw_sort` (which sorts all ``P`` corners about the
    all-``P`` centroid), this sorts only the corners that will be used (rows ``0..count-1``)
    about the centroid of that subset, so the kept polygon is angularly monotone about its
    own centre (winding +-1). Rows ``>= count`` are returned as NaN (the pruned tail). Same
    ``atan2(dx, dy)`` (X-first) key as :func:`ccw_sort`.

    Args:
        points: [E, P, 2] raw corners.
        count:  [E] integer per-env kept-corner count, each in [1, P].

    Returns:
        [E, P, 2]: rows [0, count) = kept corners sorted by angle about their centroid;
        rows [count, P) = NaN.
    """
    E, P, _ = points.shape
    count = count.to(points.device)
    row = torch.arange(P, device=points.device).unsqueeze(0)                 # [1, P]
    kept = row < count.unsqueeze(1)                                          # [E, P] bool
    nan = torch.full_like(points, float("nan"))
    masked = torch.where(kept.unsqueeze(-1), points, nan)                    # pruned rows -> NaN
    centroid = torch.nansum(masked, dim=1) / count.clamp(min=1).unsqueeze(-1).to(points.dtype)
    d = masked - centroid.unsqueeze(1)                                       # NaN on pruned rows
    angles = torch.arctan2(d[:, :, 0], d[:, :, 1])                           # X first
    key = torch.where(kept, angles, torch.full_like(angles, float("inf")))   # pruned -> tail
    ids = torch.argsort(key, dim=1, stable=True)
    return torch.gather(masked, 1, ids.unsqueeze(-1).expand(-1, -1, points.size(2)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_geometry_ccw_sort.py -v`
Expected: PASS (both new tests and all pre-existing tests in the file).

- [ ] **Step 5: Commit**

```bash
git add track_gen/geometry.py tests/test_geometry_ccw_sort.py
git commit -m "feat(geometry): add ccw_sort_count (prune-then-sort corner helper)"
```

---

### Task 2: Count-aware Warp `ccw_sort`

**Files:**
- Modify: `track_gen/warp_pipeline.py` (`_ccw_sort_k` ~line 734; `ccw_sort` wrapper ~line 1896)
- Test: `tests/test_warp_ccw_sort.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_warp_ccw_sort.py` (the file already imports `wpl`, `geometry`, `math`, `torch`, `DEVS`):

```python
def _count_env(P, count, cx, cy, r, scramble, dev):
    """First `count` points on a jittered circle (well-separated angles, scrambled),
    then `P-count` far-away padding points that ccw_sort_count must drop to NaN."""
    base = torch.arange(count, dtype=torch.float32)
    ang = base * (2.0 * math.pi / count) + 0.07 * torch.sin(base * 1.3)
    rad = r * (1.0 + 0.13 * torch.cos(base * 0.9))
    circle = torch.stack([cx + rad * torch.cos(ang), cy + rad * torch.sin(ang)], dim=-1)
    circle = circle[torch.tensor(scramble, dtype=torch.long)]
    j = torch.arange(P - count, dtype=torch.float32)
    pad = torch.stack([cx + 20.0 + j, cy + 20.0 + j], dim=-1)
    return torch.cat([circle, pad], dim=0).to(dev)            # [P, 2]


@pytest.mark.parametrize("dev", DEVS)
def test_ccw_sort_count_matches_oracle(dev):
    P = 11
    pts = torch.stack([
        _count_env(P, count=11, cx=0.0, cy=0.0, r=2.0, scramble=[3, 7, 0, 10, 4, 1, 9, 2, 6, 8, 5], dev=dev),
        _count_env(P, count=7, cx=5.0, cy=-3.0, r=1.0, scramble=[6, 0, 4, 2, 5, 1, 3], dev=dev),
        _count_env(P, count=5, cx=-4.0, cy=8.0, r=3.5, scramble=[2, 0, 4, 1, 3], dev=dev),
    ], dim=0)
    count = torch.tensor([11, 7, 5], device=dev)

    got = wpl.ccw_sort(pts, count)
    ref = geometry.ccw_sort_count(pts, count)
    for e in range(pts.shape[0]):
        c = int(count[e])
        # kept rows are a pure permutation (no coord arithmetic) -> byte-exact
        assert torch.equal(got[e, :c].cpu(), ref[e, :c].cpu())
        assert torch.isnan(got[e, c:]).all()
        assert torch.isnan(ref[e, c:]).all()


@pytest.mark.parametrize("dev", DEVS)
def test_ccw_sort_no_count_unchanged(dev):
    # count=None keeps the legacy all-P behaviour byte-for-byte.
    pts = _make_batch(dev)
    assert torch.equal(wpl.ccw_sort(pts).cpu(), wpl.ccw_sort(pts, count=None).cpu())
    assert torch.equal(wpl.ccw_sort(pts).cpu(), geometry.ccw_sort(pts).cpu())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_warp_ccw_sort.py::test_ccw_sort_count_matches_oracle -v`
Expected: FAIL — `ccw_sort()` takes 1 positional arg / `TypeError` (no `count` parameter yet).

- [ ] **Step 3a: Make `_ccw_sort_k` count-aware**

Replace the body of `_ccw_sort_k` (warp_pipeline.py ~734-769) with this (note the new `count` array parameter; all loops stay `range(P)` and guard on `m`, so no dynamic-range constructs):

```python
    def _ccw_sort_k(
        points: wp.array(dtype=wp.vec2f),
        P: int,
        count: wp.array(dtype=wp.int32),
        keys: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.vec2f),
    ):
        # One thread per env e. Orders this env's FIRST m = count[e] corners ascending by
        # the centroid-relative angle key = atan2(dx, dy) (X FIRST), about the centroid of
        # those m corners; rows [m, P) are written NaN (the pruned tail). m == P reproduces
        # the legacy all-P sort with no NaN tail. The insertion sort reads only slots behind
        # its write frontier, so the uninitialised keys/out scratch is never consumed.
        e = wp.tid()
        base = e * P
        m = count[e]
        if m < 1:
            m = 1
        if m > P:
            m = P

        # Centroid over the first m corners (float64 to match torch.mean closely).
        sx = wp.float64(0.0)
        sy = wp.float64(0.0)
        for i in range(P):
            if i < m:
                p = points[base + i]
                sx = sx + wp.float64(p[0])
                sy = sy + wp.float64(p[1])
        cx = wp.float32(sx / wp.float64(m))
        cy = wp.float32(sy / wp.float64(m))

        for c in range(P):
            if c < m:
                p = points[base + c]
                key = wp.atan2(p[0] - cx, p[1] - cy)   # X first!
                j = c - 1
                while j >= 0 and keys[base + j] > key:
                    keys[base + j + 1] = keys[base + j]
                    out[base + j + 1] = out[base + j]
                    j = j - 1
                keys[base + j + 1] = key
                out[base + j + 1] = p
            else:
                out[base + c] = wp.vec2f(wp.nan, wp.nan)
```

- [ ] **Step 3b: Make the `ccw_sort` wrapper accept `count`**

Replace the `ccw_sort` wrapper body (warp_pipeline.py ~1896-1913) so it builds a count array (defaulting to all-`P`) and passes it to the kernel:

```python
def ccw_sort(points: torch.Tensor, count: torch.Tensor | None = None) -> torch.Tensor:
    """Angle-sort each env's corners about their centroid (X-first atan2 key).

    With ``count=None`` (default) sorts all P corners about the all-P centroid -- the legacy
    behaviour, byte-identical to :func:`track_gen.geometry.ccw_sort`. With a per-env ``count``
    it prune-then-sorts: only the first ``count[e]`` corners are sorted (about THEIR centroid)
    and rows ``>= count`` are returned NaN, matching :func:`track_gen.geometry.ccw_sort_count`.
    Pure Warp (cpu+cuda); one thread per env, in-place insertion sort.
    """
    _init()
    E, P, _ = points.shape
    dev = str(points.device)
    flat = E * P

    pf = wp.from_torch(points.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    # keys_t / out_t are intentionally uninitialised: the insertion sort only ever reads
    # slots strictly behind its write frontier, so no garbage is consumed; pruned rows
    # ([count, P)) are written NaN by the kernel.
    keys_t = torch.empty(flat, device=points.device, dtype=torch.float32)
    out_t = torch.empty(flat, 2, device=points.device, dtype=torch.float32)
    if count is None:
        count_t = torch.full((E,), P, device=points.device, dtype=torch.int32)
    else:
        count_t = count.to(device=points.device, dtype=torch.int32).contiguous()
    wp.launch(_ccw_sort_k, dim=E,
              inputs=[pf, P, wp.from_torch(count_t, dtype=wp.int32),
                      wp.from_torch(keys_t, dtype=wp.float32),
                      wp.from_torch(out_t, dtype=wp.vec2f)],
              device=dev)
    _sync(points.device)
    return out_t.view(E, P, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_warp_ccw_sort.py -v`
Expected: PASS — the two new tests AND the pre-existing `test_ccw_sort_matches_oracle` / `test_ccw_sort_keys_monotone` (count=None path is byte-identical to before).

- [ ] **Step 5: Commit**

```bash
git add track_gen/warp_pipeline.py tests/test_warp_ccw_sort.py
git commit -m "feat(warp): count-aware ccw_sort (prune-then-sort; count=None unchanged)"
```

---

### Task 3: Oracle `_prune_corners` → prune-then-sort

**Files:**
- Modify: `track_gen/generators.py` (`_prune_corners`, ~line 106-133)
- Test: `tests/test_generators.py` (existing tests must stay green)

- [ ] **Step 1: Run the existing prune tests to confirm the baseline is green**

Run: `.venv/bin/python -m pytest tests/test_generators.py -k prune -v`
Expected: PASS (`test_prune_corners_shape_and_count`, `test_prune_corners_pads_with_nan`, `test_prune_corners_reproducible`). These assert shape, count range, NaN-tail, and reproducibility — none assert the ordering, so they must remain green after the change.

- [ ] **Step 2: Reorder `_prune_corners` to prune-then-sort**

Replace the body of `_prune_corners` (generators.py 106-133) with:

```python
    def _prune_corners(self, points: torch.Tensor, ids: torch.Tensor):
        """Sample a per-env corner count, then prune-then-sort: angle-sort the first
        ``count`` corners about THEIR OWN centroid (rows >= count are NaN).

        Sorting the kept subset about its own centre yields an angularly-monotone (winding
        +-1) polygon, avoiding the figure-eight that sort-then-prune produced (the kept
        corners were a mis-centered angular wedge with a long Bezier closing chord).

        Args:
            points: [E, max_num_points, 2] raw sampled corners.
            ids: [E] env ids (for per-env reproducible count sampling).

        Returns:
            (pruned [E, max_num_points, 2], count [E] long) where rows >= count are NaN.
        """
        E, P, _ = points.shape

        # Per-env corner count in [min_num_points, max_num_points] (inclusive).
        # sample_integer_torch samples in [low, high); high = max+1 for an inclusive upper bound.
        count = self.rng.sample_integer_torch(
            self.config.min_num_points,
            self.config.max_num_points + 1,
            (1,),
            ids=ids,
        ).view(E).long()
        count = count.clamp(max=P)

        pruned = ccw_sort_count(points, count)  # sort first `count` about own centroid; NaN tail
        return pruned, count
```

- [ ] **Step 3: Import `ccw_sort_count`**

In `track_gen/generators.py`, the line ~16 currently imports from `.geometry`. Add `ccw_sort_count` to that import:

```python
from .geometry import arc_length_resample, ccw_sort, ccw_sort_count, safe_normalize, self_intersections, turning_number, vertex_tangents
```

(`ccw_sort` is still imported even though `_prune_corners` no longer calls it directly, in case other methods use it; leave it.)

- [ ] **Step 4: Run the generator tests to verify they still pass**

Run: `.venv/bin/python -m pytest tests/test_generators.py -v`
Expected: PASS (all). The NaN-tail / count-range / reproducibility assertions hold; the ordering changed but no test pins it.

- [ ] **Step 5: Commit**

```bash
git add track_gen/generators.py
git commit -m "feat(oracle): _prune_corners prune-then-sort via ccw_sort_count"
```

---

### Task 4: Warp `generate_centerline_warp` → prune-then-sort

**Files:**
- Modify: `track_gen/warp_pipeline.py` (`generate_centerline_warp`, ~line 1100-1102)
- Test: `tests/test_warp_generate.py` (existing tests must stay green)

- [ ] **Step 1: Reorder the corner pipeline inside the regen loop**

In `generate_centerline_warp`, replace these three lines (~1100-1102):

```python
        corners = ccw_sort(corner_sample(seeds, k, config))   # [E, P, 2]
        count = corner_count_sample(seeds, k, config)         # [E]
        dense = assemble(corners, count, config)              # [E, M, 2] (NaN-pruned)
```

with (sample count first, then prune-then-sort):

```python
        count = corner_count_sample(seeds, k, config)              # [E]
        corners = ccw_sort(corner_sample(seeds, k, config), count) # [E, P, 2] prune-then-sort
        dense = assemble(corners, count, config)                   # [E, M, 2] (NaN-pruned)
```

(`assemble` is unchanged: it NaN-prunes `i >= count` internally, consistent with the now-NaN tail.)

- [ ] **Step 2: Run the generate tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_warp_generate.py -v`
Expected: PASS. `test_generate_centerline_warp` asserts `yield_rate >= 0.95` (the change only improves yield), simplicity, finiteness, and reproducibility — all hold.

- [ ] **Step 3: Commit**

```bash
git add track_gen/warp_pipeline.py
git commit -m "feat(warp): generate_centerline_warp prune-then-sort (sample count before ccw_sort)"
```

---

### Task 5: Behaviour test — figure-8 elimination + yield direction

**Files:**
- Create: `tests/test_warp_corner_ordering.py`

- [ ] **Step 1: Write the behaviour test**

Create `tests/test_warp_corner_ordering.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Behaviour: prune-then-sort eliminates winding-0 figure-eights at the fat-band regime.

The old sort-then-prune ordering produced ~3.5% figure-eights (winding != +-1) on a single
generation attempt; prune-then-sort drops that to ~0.1%. We assert the figure-8 rate is well
under 1% and record the per-attempt accept rate.
"""
import math

import pytest
import torch

pytest.importorskip("warp")

from track_gen import warp_pipeline as wpl  # noqa: E402
from track_gen.types import TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVS)
def test_prune_then_sort_eliminates_figure_eights(dev):
    E = 1024
    cfg = TrackGenConfig(num_envs=E, num_points=256, half_width=0.5, scale=10.0,
                         output_mode="constant_spacing", spacing=0.30, N_max=384, device=dev)
    seeds = torch.arange(E, dtype=torch.int32, device=dev)

    # Reproduce generate_centerline_warp's attempt-0 corner pipeline (prune-then-sort).
    count = wpl.corner_count_sample(seeds, 0, cfg)
    corners = wpl.ccw_sort(wpl.corner_sample(seeds, 0, cfg), count)
    dense = wpl.assemble(corners, count, cfg)
    rs30, _ = wpl.arc_length_resample_warp(dense, int(cfg.num_points_per_segment))
    turn = wpl.turning_number(rs30)
    turn_ok = (turn.abs() - 2.0 * math.pi).abs() <= float(cfg.turning_tol)

    fig8_rate = 1.0 - turn_ok.float().mean().item()
    assert fig8_rate < 0.01, f"figure-8 rate {fig8_rate:.4f} not < 1% (old sort-then-prune ~3.5%)"

    # Per-attempt accept should clear the old single-attempt baseline (~0.51).
    accept = wpl.gates(corners, dense, count, cfg)
    assert accept.float().mean().item() > 0.55
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_warp_corner_ordering.py -v`
Expected: PASS (figure-8 rate ~0.1% < 1%; accept rate > 0.55).

- [ ] **Step 3: Sanity-check the failure mode is real (optional, manual)**

Temporarily revert Task 4's reorder locally (sort-then-prune) and rerun this test; it should FAIL with `fig8_rate ~0.035`. Re-apply the reorder. (Don't commit the revert.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_warp_corner_ordering.py
git commit -m "test: prune-then-sort eliminates figure-eights at fat-band regime"
```

---

### Task 6: Full suite + per-attempt yield measurement

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all). No pre-existing test should regress; the new tests pass.

- [ ] **Step 2: Measure the per-attempt yield improvement (record in the commit message)**

Run:

```bash
.venv/bin/python - <<'PY'
import torch, warp as wp; wp.init()
from track_gen import warp_pipeline as wpl
from track_gen.types import TrackGenConfig
dev = "cuda" if torch.cuda.is_available() else "cpu"
E = 2048
cfg = TrackGenConfig(num_envs=E, num_points=256, half_width=0.5, scale=10.0,
                     output_mode="constant_spacing", spacing=0.30, N_max=384,
                     max_regen_iters=1, relax_iters=150, device=dev)
seeds = torch.arange(E, dtype=torch.int32, device=dev)
y = wpl.generate_tracks_warp(cfg, seeds).valid.float().mean().item()
print(f"per-attempt (regen=1) valid yield: {y:.3f}  (baseline ~0.53)")
PY
```

Expected: yield noticeably above the ~0.53 baseline (target ~0.73). Record the number.

- [ ] **Step 3: Finish the branch**

Use **superpowers:finishing-a-development-branch** to verify tests, then present merge / PR / keep / discard options. (Branch: `prune-then-sort-corners`.)

---

## Self-Review

**Spec coverage:**
- Count-aware sort applied to Warp (`_ccw_sort_k`/`ccw_sort`) → Task 2. ✓
- Matching reorder in oracle (`_prune_corners`) → Task 3. ✓
- Reorder Warp `generate_centerline_warp` → Task 4. ✓
- Parity preserved (count-aware Warp == oracle `ccw_sort_count`; count=None byte-identical) → Tasks 1, 2. ✓
- New winding/figure-8 behaviour test → Task 5. ✓
- Per-attempt yield measurement → Task 6. ✓
- Scope boundaries (no equispace, no deep self-X, no angle-gate) → respected (not implemented). ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every test step shows the assertion and the run command with expected result.

**Type consistency:** `ccw_sort_count(points, count)` (Task 1) is the exact name imported (Task 3) and compared against (Task 2). `ccw_sort(points, count=None)` signature (Task 2) matches its call in `generate_centerline_warp` (Task 4) and the tests. `_ccw_sort_k(points, P, count, keys, out)` parameter order matches its `wp.launch` inputs in the wrapper.
