# Constant-Spacing Warp Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `generate_tracks_warp` support `output_mode="constant_spacing"` — each track relaxed and inflated at a **per-track point count** `count[e] = round(perimeter[e] / spacing)` (constant arc-length spacing ≈ track-width scale) instead of a fixed 256 — which lets the XPBD relaxation actually converge to **smooth, genuinely valid tracks** (validated: ~100% road-valid + half the jaggedness vs fixed-256's ~71% jagged).

**Architecture:** Generation stays **fixed** at `num_points` (untouched). After generation, the centerline is resampled to a per-track `count[e]` at constant spacing into an `[E, N_max, 2]` buffer with NaN padding beyond `count[e]`; every post-generation Warp stage (band/L0, XPBD relax, uniform resample, thickness, self-intersections, turning, frame, offset, arclength, validity, inflate) is made **count-aware**: it loops `range(count[e])`, wraps `% count[e]`, bases at `e*N_max`, and guards per-point threads with `if i < count[e]`. The **invariant that protects all existing tests**: when `count[e] == N_max == N` for every env (no padding), each count-aware kernel is **bit-identical** to today's fixed-N kernel — this is the parity test for every task.

**Tech Stack:** NVIDIA Warp 1.14 (`wp.kernel`, `wp.func`, `wp.from_torch`), PyTorch 2.6, pytest. Env: `.venv/bin/python` (CUDA torch + warp-lang). GPU present (16 GB) — **run GPU work serially**. Warp kernels also run on the Warp `cpu` device, so most tests are GPU-free; cuda-only assertions guard on `torch.cuda.is_available()`. Run the suite with `.venv/bin/python -m pytest -q` (baseline **189 passing** on branch `relaxation-quality`).

---

## Background / why (validated findings)

The fixed `num_points=256` over-resolves the centerline relative to the 0.5 m half-width (segment ≈ 0.2 m ≪ 0.5 m). The Jacobi XPBD solve under-converges at that resolution in 150 iters → high-frequency **jaggedness** → the 1 m road folds (real self-overlap) → ~30 % of tracks fail. Relaxing at **constant ~0.25–0.40 m spacing** (≈ width scale, ≈ 112–176 links for typical 40–70 m tracks) lets the solve converge → smooth, genuinely-valid tracks.

Measured (E=512, half_width=0.5, scale=10, relax_iters=150), road-border-overlap validity:

| config | median links | road-valid yield |
|---|---|---|
| Fixed 256 | 256 | 0.712 |
| Const s=0.40 | 112 | 1.000 |
| Const s=0.30 | 144 | 1.000 |
| Const s=0.25 | 176 | 1.000 |

Recommended `spacing ≈ 0.6 × half_width` (≈ 0.30 m for the 1 m-track regime).

---

## The count-aware convention (READ FIRST — every kernel task applies this)

All per-env buffers are flat `[E * N_max]` (`wp.vec2f`) or `[E]` scalars. A new `count: wp.array(dtype=wp.int32)` of length `E` carries each track's real point count (`1 <= count[e] <= N_max`).

**Thread-per-point kernels** (`e = t // N_max`, `i = t % N_max`):
- Guard the write: `if i >= count[e]: return` (or wrap the body in `if i < count[e]:`).
- Wrap neighbours over the **real** loop: next `= base + (i + 1) % count[e]`, prev `= base + (i + count[e] - 1) % count[e]`, where `base = e * N_max`.

**One-thread-per-env kernels** (`e = wp.tid()`, loops `for j in range(N)`):
- Loop `for j in range(count[e])`, base `e * N_max`, wrap `% count[e]`, circular distance `circ = wp.min(d, count[e] - d)`.

**Padding:** slots `count[e] .. N_max-1` hold `(nan, nan)` and must never be written or read as real.

**Parity invariant (the safety net):** if a kernel is launched with `N_max == N` and `count[e] == N` for all `e`, it must produce results **identical** (bit-for-bit for integer/exact ops, `allclose 1e-6` for float reductions) to the current fixed-N kernel. The current fixed-N call sites pass `count = full((E,), N)` and `N_max = N`, so **fixed mode is just the all-real special case** — this is why the 189 existing tests keep passing. Every task below includes this parity test.

**Backward-compat strategy:** keep the existing fixed-N kernel launches working by passing a `count` array equal to `N` and `N_max = N`. Do NOT delete the fixed path; constant_spacing is purely additive (gated by `config.output_mode`).

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `track_gen/types.py` | config | document/validate `output_mode`, `spacing`, `N_max` for the Warp path (fields already exist) |
| `track_gen/warp_pipeline.py` | all pipeline kernels + wrappers | new constant-spacing resample; make post-generation kernels/wrappers count-aware; constant_spacing path in `generate_tracks_warp`/`inflate_warp`; graph capture |
| `track_gen/warp_relax.py` | XPBD solve | `_disp_kernel`/`_apply_kernel`/`xpbd_solve` count-aware |
| `tests/test_warp_constant_spacing.py` | new | parity (count==N matches fixed) + variable-count behaviour per stage; end-to-end yield/smoothness |
| `tests/test_warp_relax.py` | existing | extend with count-aware parity |
| `benchmarks/benchmark_yield_sweep.py` | existing | add a `constant_spacing` config row |
| `viz/make_report.py` | existing | add a constant-spacing page (the real fix) |

---

## Task 1: Config validation + test scaffold

**Files:** Modify `track_gen/types.py`; create `tests/test_warp_constant_spacing.py`.

- [ ] **Step 1: Confirm the config fields.** `TrackGenConfig` already has `output_mode: str = "fixed"` (`{"fixed","constant_spacing"}`), `spacing: float = 0.1`, `N_max: int = 256`, `num_points: int = 256`. No new field. Add a one-line clarifying comment next to `spacing`:
```python
    spacing: float = 0.1            # constant_spacing arc-length step (m). Warp relax: set ~0.6*half_width.
```

- [ ] **Step 2: Create the test module skeleton** `tests/test_warp_constant_spacing.py`:
```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Constant-spacing Warp pipeline: per-stage parity (count==N_max matches fixed-N) and
variable-count behaviour, plus the end-to-end smoothness/yield win."""
import math
import pytest
import torch

pytest.importorskip("warp")
import warp as wp  # noqa: E402
wp.init()

from track_gen import warp_pipeline as wpl, warp_relax, geometry  # noqa: E402
from track_gen.types import TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _circle(N, r, dev):
    t = torch.linspace(0, 2 * math.pi, N + 1, device=dev)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], -1).to(torch.float32)


def _pad(center, n_max):
    """[E,N,2] -> [E,n_max,2] NaN-padded, count=[N]*E."""
    E, N, _ = center.shape
    buf = torch.full((E, n_max, 2), float("nan"), device=center.device, dtype=torch.float32)
    buf[:, :N] = center
    count = torch.full((E,), N, dtype=torch.int32, device=center.device)
    return buf, count
```

- [ ] **Step 3: Run it (collects, no tests yet).** Run: `.venv/bin/python -m pytest tests/test_warp_constant_spacing.py -q` Expected: `no tests ran` (or 0 passed) — module imports cleanly.

- [ ] **Step 4: Commit.**
```bash
git add track_gen/types.py tests/test_warp_constant_spacing.py
git commit -m "constant-spacing: config doc + test scaffold"
```

---

## Task 2: Constant-spacing resample (fixed source → variable count)

**New primitive:** `resample_constant_spacing(center, spacing, n_max) -> (out [E,n_max,2] NaN-padded, count [E] long)`, matching `geometry.arc_length_resample(points, spacing=spacing, n_max=n_max)` on a fully-real (no-NaN) `[E,N,2]` source. `count = floor(perimeter/spacing)+1` filtered to `targets < total` (i.e. effectively `count = floor(total/spacing)+1` capped so the last target `< total`; replicate the oracle exactly).

**Files:** Modify `track_gen/warp_pipeline.py`; test in `tests/test_warp_constant_spacing.py`.

- [ ] **Step 1: Write the parity test** (append to the test module):
```python
def test_constant_spacing_resample_matches_torch_oracle():
    dev = "cpu"
    E, N = 3, 300
    src = torch.stack([_circle(N, r, dev) for r in (1.0, 2.5, 4.0)], 0)  # [3,N,2]
    spacing, n_max = 0.5, 128
    out_w, cnt_w = wpl.resample_constant_spacing(src, spacing, n_max)
    out_t, cnt_t = geometry.arc_length_resample(src, spacing=spacing, n_max=n_max)
    assert out_w.shape == (E, n_max, 2)
    assert torch.equal(cnt_w.cpu(), cnt_t.cpu()), f"{cnt_w} vs {cnt_t}"
    for e in range(E):
        c = int(cnt_w[e])
        assert torch.allclose(out_w[e, :c], out_t[e, :c], atol=1e-4)
        assert torch.isnan(out_w[e, c:]).all()
```

- [ ] **Step 2: Run, expect failure** — `resample_constant_spacing` undefined.
Run: `.venv/bin/python -m pytest tests/test_warp_constant_spacing.py::test_constant_spacing_resample_matches_torch_oracle -q` Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Add the kernels + wrapper** to `track_gen/warp_pipeline.py` inside the `if _HAVE_WARP:` block (place near `_resample_scan_k`):
```python
    @wp.kernel
    def _cs_scan_k(c: wp.array(dtype=wp.vec2f), N: int, spacing: wp.float32, n_max: int,
                   seg: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32),
                   count: wp.array(dtype=wp.int32)):
        # One thread per env e. Closed-loop seg lengths + cumulative arc s (len N+1),
        # then count = floor(total/spacing)+1, capped so target (count-1)*spacing < total
        # and to n_max. Mirrors geometry._resample_one's spacing branch.
        e = wp.tid()
        b = e * N
        es = e * (N + 1)
        s[es] = float(0.0)
        acc = wp.float64(0.0)
        for i in range(N):
            d = c[b + (i + 1) % N] - c[b + i]
            l = wp.length(d)
            seg[b + i] = l
            acc = acc + wp.float64(l)
            s[es + i + 1] = wp.float32(acc)
        total = wp.float32(acc)
        k = int(wp.floor(total / spacing)) + 1
        # drop targets >= total: last target is (k-1)*spacing; ensure < total
        while k > 1 and wp.float32(k - 1) * spacing >= total:
            k = k - 1
        count[e] = wp.min(wp.max(k, 1), n_max)

    @wp.kernel
    def _cs_lookup_k(c: wp.array(dtype=wp.vec2f), seg: wp.array(dtype=wp.float32),
                     s: wp.array(dtype=wp.float32), N: int, spacing: wp.float32, n_max: int,
                     count: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec2f)):
        # One thread per OUTPUT slot t (dim = E*n_max). k >= count[e] -> NaN pad.
        t = wp.tid()
        e = t // n_max
        k = t % n_max
        if k >= count[e]:
            out[t] = wp.vec2f(wp.nan, wp.nan)
            return
        eb = e * N
        esi = e * (N + 1)
        target = wp.float32(k) * spacing
        idx = int(0)
        while idx < N - 1 and s[esi + idx + 1] < target:
            idx = idx + 1
        s0 = s[esi + idx]
        segl = wp.max(seg[eb + idx], float(1.0e-12))
        frac = wp.clamp((target - s0) / segl, float(0.0), float(1.0))
        p0 = c[eb + idx]
        p1 = c[eb + (idx + 1) % N]
        out[t] = p0 + frac * (p1 - p0)


def resample_constant_spacing(center: torch.Tensor, spacing: float, n_max: int):
    """Arc-length resample each fully-real closed loop to constant `spacing`, padded to
    n_max with NaN. Returns (out [E, n_max, 2], count [E] long). Matches
    geometry.arc_length_resample(points, spacing=spacing, n_max=n_max). Pure Warp (cpu+cuda)."""
    _init()
    E, N, _ = center.shape
    dev = str(center.device)
    cf = wp.from_torch(center.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    seg = torch.empty(E * N, device=center.device, dtype=torch.float32)
    s = torch.empty(E * (N + 1), device=center.device, dtype=torch.float32)
    cnt = torch.empty(E, device=center.device, dtype=torch.int32)
    out = torch.empty(E * n_max, 2, device=center.device, dtype=torch.float32)
    wp.launch(_cs_scan_k, dim=E, inputs=[cf, N, float(spacing), n_max,
              wp.from_torch(seg, dtype=wp.float32), wp.from_torch(s, dtype=wp.float32),
              wp.from_torch(cnt, dtype=wp.int32)], device=dev)
    wp.launch(_cs_lookup_k, dim=E * n_max, inputs=[cf, wp.from_torch(seg, dtype=wp.float32),
              wp.from_torch(s, dtype=wp.float32), N, float(spacing), n_max,
              wp.from_torch(cnt, dtype=wp.int32), wp.from_torch(out, dtype=wp.vec2f)], device=dev)
    _sync(center.device)
    return out.view(E, n_max, 2), cnt.long()
```

- [ ] **Step 4: Run the parity test, expect pass.** If `count` is off by one vs the oracle, reconcile the `while` cap against `geometry._resample_one` exactly (the oracle filters `targets[targets < total]` after building `arange(k)*spacing`).
Run: `.venv/bin/python -m pytest tests/test_warp_constant_spacing.py::test_constant_spacing_resample_matches_torch_oracle -q` Expected: PASS.

- [ ] **Step 5: Full suite (no regression).** Run: `.venv/bin/python -m pytest -q` Expected: `189 passed`.

- [ ] **Step 6: Commit.**
```bash
git add track_gen/warp_pipeline.py tests/test_warp_constant_spacing.py
git commit -m "constant-spacing: Warp arc-length resample to constant spacing (oracle parity)"
```

---

## Task 3: Count-aware `resample_uniform`

Make the relax-output resampler honor a per-track `count` (resample the `count[e]` real points to `count[e]` arc-uniform points, in an `[E, N_max, 2]` buffer). Default `count=None` ⇒ today's fixed behaviour.

**Files:** Modify `track_gen/warp_pipeline.py` (`_resample_scan_k`, `_resample_lookup_k`, `resample_uniform`); test in `tests/test_warp_constant_spacing.py`.

- [ ] **Step 1: Parity + variable test:**
```python
@pytest.mark.parametrize("dev", DEVS)
def test_resample_uniform_count_aware(dev):
    # parity: count==N reproduces the fixed call
    src = torch.stack([_circle(64, 1.0, dev), _circle(64, 2.0, dev)], 0)
    base = wpl.resample_uniform(src, 64)
    buf, cnt = _pad(src, 64)
    out = wpl.resample_uniform(buf, 64, count=cnt)
    assert torch.allclose(out, base, atol=1e-5, equal_nan=True)
    # variable: env0 uses 40 real pts (rest NaN), env1 uses 64; env0 stays ~circle, pad NaN
    buf2 = torch.full((2, 64, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :40] = _circle(40, 1.0, dev); buf2[1, :64] = _circle(64, 2.0, dev)
    cnt2 = torch.tensor([40, 64], dtype=torch.int32, device=dev)
    out2 = wpl.resample_uniform(buf2, 64, count=cnt2)
    r0 = torch.linalg.norm(out2[0, :40], dim=-1)
    assert torch.allclose(r0, torch.ones_like(r0), atol=2e-2)
    assert torch.isnan(out2[0, 40:]).all()
```

- [ ] **Step 2: Run, expect failure** (`resample_uniform` has no `count` kwarg / asserts `n==N`).
Run: `.venv/bin/python -m pytest tests/test_warp_constant_spacing.py::test_resample_uniform_count_aware -q` Expected: FAIL.

- [ ] **Step 3: Make the kernels count-aware.** Change `_resample_scan_k` to take `n_max:int, count:wp.array(dtype=wp.int32)`, base `e*n_max`, loop `for i in range(count[e])`, wrap `(i+1)%count[e]`, write `s[e*(n_max+1) ...]`. Change `_resample_lookup_k` to take `n_max, count`: `e=t//n_max, k=t%n_max`; `if k>=count[e]: out[t]=NaN; return`; `total=s[es+count[e]]`, `tk=float(k)*total/float(count[e])`, scan `idx<count[e]-1`, wrap `(idx+1)%count[e]`. In `resample_uniform(center, n, count=None)`: when `count is None`, set `n_max=N` and `count=full(E,N)` (parity); else `n_max=center.shape[1]`. Allocate `s` as `E*(n_max+1)`. Drop the `assert n==N` (replace with `assert n==N_max when count is None`).

- [ ] **Step 4: Run the test, expect pass.** Run: `.venv/bin/python -m pytest tests/test_warp_constant_spacing.py::test_resample_uniform_count_aware -q` Expected: PASS.

- [ ] **Step 5: Full suite (the fixed-N default path must be unchanged).** Run: `.venv/bin/python -m pytest -q` Expected: `190 passed` (189 + 1 new; cpu+cuda parametrize collapses per device).

- [ ] **Step 6: Commit.**
```bash
git add track_gen/warp_pipeline.py tests/test_warp_constant_spacing.py
git commit -m "constant-spacing: count-aware resample_uniform (fixed-N parity preserved)"
```

---

## Tasks 4–9: Make the per-env geometry kernels count-aware

Each task follows the **identical recipe** below. They are independent and can be done in any order; each has a parity test (count==N matches the current wrapper) and a variable-count test (padding ignored). Do them one per commit.

**Recipe for a count-aware geometry wrapper `f`:**
1. Add `count: wp.array(dtype=wp.int32)` and `n_max` (or reuse `N` as the buffer stride) to the kernel signature; the Python wrapper gains `count=None`.
2. In the kernel, replace `N`→`count[e]` in loop bounds, `% N`→`% count[e]`, circular distance `wp.min(d, N-d)`→`wp.min(d, count[e]-d)`, base `e*N`→`e*n_max`.
3. In the wrapper, `count is None` ⇒ `count=full(E, N)`, `n_max=N` (parity); else read the passed buffer width as `n_max`.
4. Parity test: `f(buf, count=full(E,N)) == f_fixed(center)`. Variable test: a padded batch with `count[e]<N` ignores the NaN tail.

- [ ] **Task 4 — `thickness` / `_thickness_func` / `_thickness_k` (and `_sep_min_k`, `_curvrad_min_k`, `separation_min`, `curvature_radius_min`).**
  - Parity test:
```python
@pytest.mark.parametrize("dev", DEVS)
def test_thickness_count_aware(dev):
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    band = torch.tensor([3, 3], dtype=torch.int32, device=dev)
    base = wpl.thickness(src, band)
    buf, cnt = _pad(src, 80)
    out = wpl.thickness(buf, band, count=cnt)
    assert torch.allclose(out, base, atol=1e-5)
    # variable: env0 real=50 (radius-1 circle) padded to 80; thickness ~ min(1.0, 0.5*sep)
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev); buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    th = wpl.thickness(buf2, band, count=cnt2)
    assert th[0] > 0.0 and torch.isfinite(th[0])  # NaN tail did not poison it
```
  - Implement per recipe (`_thickness_func(pts, base, count, band)`, loops `range(count)`, wrap `%count`; `_thickness_k` reads `count[e]`, base `e*n_max`). Update `_sep_min_k`/`_curvrad_min_k` likewise. Wrappers gain `count=None`.
  - Run the test (PASS), then full suite (`191 passed`). Commit: `"constant-spacing: count-aware thickness/sep/curvature"`.

- [ ] **Task 5 — `self_intersections` / `_self_intersections_func` / `_self_intersections_k`.** Parity test mirrors Task 4 (a self-crossing fixture, e.g. a pinched loop, padded; count==N matches fixed). `_self_intersections_func(poly, base, count)` loops `range(count)`, wrap `%count`, `circ = wp.min(diff, count-diff)`. Run + full suite (`192 passed`). Commit: `"constant-spacing: count-aware self_intersections"`.

- [ ] **Task 6 — `turning_number` / `_turning_func` / `_turning_k`.** Parity + a half-resolution circle (turning ≈ 2π regardless of count). `_turning_func(c, base, count)` loops `range(count)`, edges wrap `%count`. Run + full suite (`193 passed`). Commit: `"constant-spacing: count-aware turning_number"`.

- [ ] **Task 7 — `frame_curvature` / `_frame_k` and `offset` / `_offset_build_k` / `_offset_assign_k`.** Thread-per-point kernels: `e=t//n_max, i=t%n_max`; guard `if i<count[e]` (pad slots → write NaN frame / skip atomic add); neighbour wrap `(i+1)%count[e]`, `(i+count[e]-1)%count[e]`. `_offset_build_k`'s atomic shoelace must only accumulate real edges (`if i < count[e]`). Parity + variable tests. Run + full suite (`194 passed`). Commit: `"constant-spacing: count-aware frame_curvature + offset"`.

- [ ] **Task 8 — `_arclength` / `_arclength_k`.** Match the torch `inflation._arclength` semantics for variable count, **including the explicit closing-wrap segment** (last real point → point 0) which is NOT the `%count` neighbour when `count<N_max`. `_arclength_k`: loop `for i in range(count[e])`, `arclen[b+i]=acc`, add `seg = |c[b+(i+1)%count]-c[b+i]|` (the `i=count-1` term IS the wrap), `length[e]=acc`. Pad slots `arclen` = last value or 0 (define + test). Parity (count==N) + variable test against a hand-computed circle perimeter. Run + full suite (`195 passed`). Commit: `"constant-spacing: count-aware arclength (closing wrap)"`.

- [ ] **Task 9 — `xpbd_solve` / `_disp_kernel` / `_apply_kernel` (the core quality fix).**
  - Parity test (count==N reproduces fixed solve, anchor included):
```python
@pytest.mark.parametrize("dev", DEVS)
def test_xpbd_count_aware_parity(dev):
    src = torch.stack([_circle(96, 1.0, dev), _circle(96, 1.4, dev)], 0)
    band = torch.tensor([3, 2], dtype=torch.int32, device=dev)
    L0 = geometry.perimeter(src) / 96
    cfg = TrackGenConfig(num_envs=2, num_points=96, half_width=0.05, relax_iters=40, device=dev)
    base = warp_relax.xpbd_solve(src, band, L0, cfg)
    buf, cnt = _pad(src, 96)
    out = warp_relax.xpbd_solve(buf, band, L0, cfg, count=cnt)
    assert torch.allclose(out[:, :96], base, atol=1e-5)
```
  - Variable-count quality test (the validated win — CPU, small E):
```python
def test_xpbd_constant_spacing_is_smoother():
    dev = "cpu"
    # a wiggly source; relax at fixed-256-equivalent vs constant spacing, compare jaggedness
    seeds = torch.arange(16, dtype=torch.int32, device=dev)
    cfg = TrackGenConfig(num_envs=16, num_points=256, half_width=0.5, scale=10.0,
                         relax_iters=150, device=dev)
    cl, gv = wpl.generate_centerline_warp(seeds, cfg)
    def jag(c, count):
        # mean abs turning over real points
        ...
    # fixed
    L0f = wpl._mean_seg_len_torch(cl)
    bandf = (2*0.5/L0f.clamp_min(1e-9)).round().int().clamp_min(1)
    relf = wpl.resample_uniform(wpl.warp_relax.xpbd_solve(cl, bandf, L0f, cfg), 256)
    # constant spacing 0.30
    buf, cnt = wpl.resample_constant_spacing(cl, 0.30, 384)
    L0c = (geometry.perimeter(cl) / cnt.float()).float()  # ~spacing
    bandc = (2*0.5/L0c.clamp_min(1e-9)).round().int().clamp_min(1)
    relc = warp_relax.xpbd_solve(buf, bandc, L0c, cfg, count=cnt)
    assert _mean_jag(relc, cnt) < _mean_jag(relf, torch.full((16,),256)) * 0.7
```
    (Define `_mean_jag` as mean absolute per-vertex turning over `count[e]` real points.)
  - Implement: `xpbd_solve(center0, band, L0, config, count=None)`. `count is None` ⇒ `n_max=N`, `count=full(E,N)` (parity, includes the anchor term from the relaxation-quality branch). Kernels: `e=t//n_max, i=t%n_max, base=e*n_max`; `if i>=count[e]: out[t]=vec2f(0,0); return`. Separation loop `for j in range(count[e])`, `circ=wp.min(dd, count[e]-dd)`. Spacing/bending neighbours wrap `%count[e]`. `x0`/anchor indexing uses the flat `t`. `_apply_kernel`: guard `if i<count[e]` (leave pad slots unchanged / NaN).
  - Run parity (PASS, cpu+cuda), then the smoothness test (PASS), then `tests/test_warp_relax.py` (unchanged) and full suite. Expected: green (`~197 passed`). Commit: `"constant-spacing: count-aware XPBD solve (the convergence/quality fix)"`.

---

## Task 10: Count-aware validity (`_validity_k`)

Get right what the torch `inflation._validity_stage` admits it gets wrong: count-mask the turning / thickness / width-floor / no-NaN / border checks so NaN padding does not poison them.

**Files:** Modify `track_gen/warp_pipeline.py` (`_validity_k`, `validity` wrapper); test in `tests/test_warp_constant_spacing.py`.

- [ ] **Step 1: Parity + variable test:**
```python
@pytest.mark.parametrize("dev", DEVS)
def test_validity_count_aware(dev):
    # a clean wide circle is valid in both fixed and padded form
    src = _circle(120, 5.0, dev).unsqueeze(0)        # radius 5, half_width 0.5 -> valid
    w = torch.full((1, 120), 0.5, device=dev)
    cnt_full = torch.tensor([120], dtype=torch.int32, device=dev)
    cfg = TrackGenConfig(num_envs=1, num_points=120, half_width=0.5, device=dev)
    # build outer/inner via offset for the border gate
    ...  # reuse wpl.frame_curvature + wpl.offset
    v_fixed = wpl.validity(src, w, cnt_full, gen_valid, cfg, outer, inner, count=cnt_full)
    buf = torch.full((1, 200, 2), float("nan"), device=dev, dtype=torch.float32); buf[0,:120]=src[0]
    cnt = torch.tensor([120], dtype=torch.int32, device=dev)
    ...  # padded outer/inner too
    v_pad = wpl.validity(buf, w_pad, cnt, gen_valid, cfg, outer_pad, inner_pad, count=cnt)
    assert bool(v_pad[0]) == bool(v_fixed[0]) == True
```
  (Fill the `...` using the existing `frame_curvature`/`offset`/`validity` call pattern from `inflate_warp`.)

- [ ] **Step 2: Run, expect failure** (`validity`/`_validity_k` have no `count`-mask plumbing for the per-point loops / band derivation).
Run: `.venv/bin/python -m pytest tests/test_warp_constant_spacing.py::test_validity_count_aware -q` Expected: FAIL.

- [ ] **Step 3: Implement.** `_validity_k` gains `n_max:int` (buffer stride) and uses `cnt=count[e]` for: the turning call (`_turning_func(center, base, cnt)`), the width/NaN loop (`for i in range(n_max): if i < cnt:`), the band perimeter loop (`for i in range(cnt): peri += |center[base+(i+1)%cnt]-center[base+i]|; L0=peri/cnt`), `_thickness_func(center, base, cnt, band)`, and the border calls (`_self_intersections_func(outer, base, cnt)+(inner,...)`). `base=e*n_max`. The `validity` wrapper gains `count=None` (⇒ `full(E,N)`, `n_max=N`).

- [ ] **Step 4: Run the test (PASS), then full suite.** Run: `.venv/bin/python -m pytest tests/test_warp_constant_spacing.py -q && .venv/bin/python -m pytest -q` Expected: green.

- [ ] **Step 5: Commit.**
```bash
git add track_gen/warp_pipeline.py tests/test_warp_constant_spacing.py
git commit -m "constant-spacing: count-masked validity (correct for padded tracks)"
```

---

## Task 11: `inflate_warp` constant_spacing

**Files:** Modify `track_gen/warp_pipeline.py` (`inflate_warp`); test in `tests/test_warp_constant_spacing.py`.

- [ ] **Step 1: Test** — a padded centerline inflates to a Track with `count[e]` real points, NaN tail, validity correct, `length` ≈ closed perimeter:
```python
@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_constant_spacing(dev):
    buf = torch.full((1, 200, 2), float("nan"), device=dev, dtype=torch.float32)
    buf[0, :120] = _circle(120, 5.0, dev)
    cnt = torch.tensor([120], dtype=torch.int32, device=dev)
    gen_valid = torch.ones(1, dtype=torch.bool, device=dev)
    cfg = TrackGenConfig(num_envs=1, num_points=120, half_width=0.5, output_mode="constant_spacing",
                         spacing=0.30, N_max=200, device=dev)
    tr = wpl.inflate_warp(buf, cfg, valid=gen_valid, count=cnt)
    assert tr.center.shape == (1, 200, 2)
    assert int(tr.count[0]) == 120
    assert torch.isnan(tr.center[0, 120:]).all()
    assert bool(tr.valid[0]) is True
    assert torch.allclose(tr.length, torch.tensor([2*math.pi*5.0], device=dev), atol=0.5)
```

- [ ] **Step 2: Run, expect failure** (`inflate_warp` asserts `output_mode=="fixed"`, no `count` arg).

- [ ] **Step 3: Implement.** `inflate_warp(center, config, valid=None, count=None)`. Replace `assert config.output_mode == "fixed"` with: if `count is None`, behave exactly as today (`n_max=N`, `count=full(E,N)`, `Track.count=full(E,N)`); else `n_max=center.shape[1]`, use the passed `count`, and pass `count`/`n_max` to `frame_curvature`, `offset`, `validity`, `_arclength`. Track.count = the passed `count`. Keep the fixed branch bit-identical.

- [ ] **Step 4: Run test (PASS) + full suite (fixed path unchanged).**

- [ ] **Step 5: Commit.** `"constant-spacing: inflate_warp variable-count Track output"`.

---

## Task 12: `generate_tracks_warp` constant_spacing path (end-to-end)

**Files:** Modify `track_gen/warp_pipeline.py` (`generate_tracks_warp`, `_band_l0_k`); test in `tests/test_warp_constant_spacing.py`.

- [ ] **Step 1: End-to-end test (the validated win):**
```python
def test_generate_tracks_constant_spacing_smoother_and_valid():
    dev = "cpu"
    E = 48
    seeds = torch.arange(E, dtype=torch.int32, device=dev)
    base = dict(num_envs=E, num_points=256, half_width=0.5, scale=10.0, relax_iters=150, device=dev)
    fixed = wpl.generate_tracks_warp(TrackGenConfig(output_mode="fixed", **base), seeds)
    cs = wpl.generate_tracks_warp(TrackGenConfig(output_mode="constant_spacing", spacing=0.30,
                                                 N_max=384, **base), seeds)
    # constant spacing yields strictly more valid tracks in this fat-band regime
    assert cs.valid.float().mean() > fixed.valid.float().mean()
    # and the valid constant-spacing tracks are smoother (lower mean turning)
    assert _mean_jag(cs.center, cs.count)[cs.valid].mean() < \
           _mean_jag(fixed.center, fixed.count)[fixed.valid].mean()
    # fixed-mode output is unchanged shape
    assert fixed.center.shape == (E, 256, 2) and int(fixed.count[0]) == 256
```

- [ ] **Step 2: Run, expect failure** (`generate_tracks_warp` asserts/assumes fixed).

- [ ] **Step 3: Implement the constant_spacing branch** in `generate_tracks_warp`, after `generate_centerline_warp` returns the fixed `[E, num_points, 2]` centerline:
```python
    if config.output_mode == "constant_spacing":
        n_max = int(config.N_max)
        centerline, count = resample_constant_spacing(centerline, float(config.spacing), n_max)
        E = centerline.shape[0]
        # band/L0 over count[e] real points (uniform ~spacing); _band_l0_k count-aware
        band = torch.empty(E, device=centerline.device, dtype=torch.int32)
        L0 = torch.empty(E, device=centerline.device, dtype=torch.float32)
        cl_w = wp.from_torch(centerline.reshape(E * n_max, 2).contiguous(), dtype=wp.vec2f)
        wp.launch(_band_l0_k, dim=E, inputs=[cl_w, n_max, 2.0 * hw,
                  wp.from_torch(band, dtype=wp.int32), wp.from_torch(L0, dtype=wp.float32),
                  wp.from_torch(count.to(torch.int32), dtype=wp.int32)], device=dev)  # _band_l0_k gains count
        if config.relax_band is not None:
            wp.launch(_fill_i32_k, dim=E, inputs=[wp.from_torch(band, dtype=wp.int32),
                      int(config.relax_band)], device=dev)
        _sync(centerline.device)
        relaxed = warp_relax.xpbd_solve(centerline, band, L0, config, count=count)
        relaxed = resample_uniform(relaxed, n_max, count=count)
        return inflate_warp(relaxed, config, valid=gen_valid, count=count)
    # ... existing fixed path unchanged ...
```
  Make `_band_l0_k` count-aware (loop `range(count[e])`, base `e*n_max`, `L0=peri/count[e]`); when called from the fixed path, pass `count=full(E,N)` and `n_max=N` (parity). Keep the `assert config.relax_solver == "xpbd"` / `assert not config.smooth_finish` lines. Remove any `output_mode=="fixed"` assertion in this function.

- [ ] **Step 4: Run the end-to-end test (PASS), then the full suite.** Expected: green; `test_generate_tracks_constant_spacing_smoother_and_valid` passes and all fixed-mode tests unchanged.

- [ ] **Step 5: Commit.** `"constant-spacing: end-to-end generate_tracks_warp path"`.

---

## Task 13: CUDA graph capture for constant_spacing

`count[e]` is data-dependent (computed from perimeter inside the captured region) but used only as a Warp array input — no host branching — so capture should work. Verify, or guard with a clear assertion if it doesn't.

**Files:** Modify `track_gen/warp_pipeline.py` (`generate_tracks_warp_graph`); test in `tests/test_warp_constant_spacing.py`.

- [ ] **Step 1: Test (cuda-only):**
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_graph_capture_constant_spacing():
    E = 8
    cfg = TrackGenConfig(num_envs=E, num_points=256, half_width=0.5, scale=10.0,
                         output_mode="constant_spacing", spacing=0.30, N_max=384, device="cuda")
    seeds = torch.arange(E, dtype=torch.int32, device="cuda")
    cap = wpl.generate_tracks_warp_graph(cfg, seeds)
    eager = wpl.generate_tracks_warp(cfg, seeds)
    replay = cap.replay(seeds)
    assert torch.equal(replay.count.cpu(), eager.count.cpu())
    assert torch.equal(replay.valid.cpu(), eager.valid.cpu())
```

- [ ] **Step 2: Run.** If it passes, done. If capture fails (e.g. dynamic count breaks the static graph), add an explicit guard in `generate_tracks_warp_graph` instead: `assert config.output_mode == "fixed", "CUDA graph capture requires output_mode='fixed'"`, and change the test to `pytest.raises(AssertionError)`. Record which path was taken in the commit message.

- [ ] **Step 3: Full suite.** Run: `.venv/bin/python -m pytest -q` Expected: green.

- [ ] **Step 4: Commit.** `"constant-spacing: CUDA graph capture (verified) | graph guard"` (pick the accurate half).

---

## Task 14: Study + report (the real fix)

**Files:** Modify `benchmarks/benchmark_yield_sweep.py`, `viz/make_report.py`.

- [ ] **Step 1: Add a constant_spacing sweep row.** Give `bench()` an `output_mode`/`spacing`/`n_max` knob and append a config row `{"output_mode": "constant_spacing", "spacing": 0.30, "n_max": 384}` (256-links baseline regime). Report yield + a road-valid (border-overlap) yield + mean jaggedness in the returned dict.

- [ ] **Step 2: Run the sweep (E=8192, serial GPU).** Run: `.venv/bin/python -m benchmarks.benchmark_yield_sweep | tee /tmp/cs_sweep.txt` Record: constant_spacing yield + road-valid + jaggedness vs the fixed-256 baseline.

- [ ] **Step 3: Add a report page.** In `viz/make_report.py` add a "Constant spacing — the convergence fix" page: a fixed-seed comparison (fixed-256 jagged-and-invalid vs constant-spacing smooth-and-valid; reuse `_fixed_seed_page`/`_gen` with `output_mode`), and the measured yield/jaggedness numbers. Regenerate: `.venv/bin/python -m viz.make_report` Expected: `wrote .../track_gen_report.pdf`.

- [ ] **Step 4: Full suite green.** Run: `.venv/bin/python -m pytest -q` Expected: green.

- [ ] **Step 5: Commit.** `"study: constant-spacing yield/smoothness win at E=8192 + report page"`.

---

## Self-Review

**Spec coverage:** resample→count (Task 2) + count-aware resample_uniform (3); count-aware geometry kernels thickness/sep/curv (4), self-intersections (5), turning (6), frame+offset (7), arclength (8); count-aware XPBD relax (9, the quality fix); count-masked validity (10); variable-count inflate (11); end-to-end pipeline (12); graph capture (13); study/report (14). Every Warp stage the explorations flagged as N-hard-coded is covered; the generation stage is intentionally left fixed (already count-capable, untouched).

**Parity backbone:** every count-aware task asserts `count==N_max` reproduces the current fixed-N result, so the 189 existing tests and fixed mode stay green throughout. Fixed mode remains the default (`output_mode="fixed"`), constant_spacing is additive and opt-in.

**Placeholder scan:** the only `...` placeholders are in test bodies where the surrounding call pattern is explicitly named to copy from (`inflate_warp`'s frame/offset/validity sequence) and in `_mean_jag` (defined in Task 9). Resolve them by copying the named existing call sites. All kernel changes follow the single explicit recipe in the Tasks 4–9 preamble; novel kernels (Task 2 resample, Task 10 validity) have full code.

**Type/name consistency:** `count: wp.array(dtype=wp.int32)` `[E]` and `n_max:int` buffer stride are used identically across all kernels; wrappers take `count=None` (⇒ fixed-N parity); `resample_constant_spacing` returns `(out, count)` matching `geometry.arc_length_resample`'s `(resampled, count)`; `Track.count` carries `count[e]`. `xpbd_solve(..., count=None)` signature matches its call sites in Task 12.

**Open risks:** (1) Task 2 `count` off-by-one vs the oracle's `targets < total` filter — reconcile against `geometry._resample_one`. (2) `N_max` must exceed the max `count` (long tracks); default 256 may be too small for `spacing<0.25` — Task 12 test uses `N_max=384`; document choosing `N_max ≳ max_perimeter/spacing`. (3) Task 13 graph capture of data-dependent `count` — fall back to a graph guard if capture fails.
```
