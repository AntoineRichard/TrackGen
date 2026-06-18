> **⚠️ Superseded — not in the live code.** The `relax_anchor` and `relax_tp_finish` features in this plan were implemented then **removed**: the anchor lowered yield with no deformation benefit, and the TP-Sobolev finisher circularized tangled tracks. The real root cause — fixed-resolution / slow-Jacobi **under-convergence** — is addressed by [`2026-06-18-constant-spacing-warp.md`](2026-06-18-constant-spacing-warp.md). The removed implementation is preserved on git branch `anchor-tp-finisher`. Kept as a point-in-time record.

# Relaxation Quality (TP-Sobolev Untangle Hybrid + Anti-Creep Anchor) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two **opt-in, default-off** relaxation enhancements to the track generator: an anti-creep position **anchor** in the XPBD solve, and a **TP-Sobolev self-avoiding finisher** that untangles the looped centerlines XPBD can't fix.

**Architecture:** The anchor is a per-sweep pull toward each bead's pre-relax position, added to BOTH the Warp kernel (`warp_relax`) and its torch oracle (`relaxation._relax_xpbd`) so their verified numerical parity holds. The TP finisher reuses the existing validated `relaxation._tp_flow` self-avoiding solver, wired into `generate_tracks_warp` after the XPBD solve, applied to all envs or only the post-XPBD failures. Both default off → the pure-Warp pipeline and its CUDA-graph path are unchanged.

**Tech Stack:** NVIDIA Warp 1.14 (`wp.kernel`, `wp.from_torch`), PyTorch 2.6, pytest. Spec: `docs/superpowers/specs/2026-06-18-relaxation-quality-design.md`. Env: `.venv/bin/python` (CUDA torch + warp-lang); GPU present (16 GB) — **run GPU work serially**. Warp kernels also run on the Warp `cpu` device, so most tests are GPU-free; cuda-only assertions guard on `torch.cuda.is_available()`.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `track_gen/types.py` | config dataclass | add `relax_tp_finish: str = "off"`, `relax_anchor: float = 0.0` |
| `track_gen/warp_relax.py` | fused Warp XPBD solve | anchor term in `_disp_kernel` + `xpbd_solve` |
| `track_gen/relaxation.py` | torch relaxation backends | anchor in `_relax_xpbd` (parity); `initial_active` arg in `_tp_flow` |
| `track_gen/warp_pipeline.py` | pipeline facade | wire TP finisher into `generate_tracks_warp`; guard `generate_tracks_warp_graph` |
| `tests/test_relaxation_anchor.py` | new | anchor: Warp==torch parity + deformation-reduction |
| `tests/test_tp_untangle.py` | new | `_tp_flow` reduces self-intersections; pipeline tp_finish raises yield |
| `tests/test_warp_graph.py` | existing | graph rejects `relax_tp_finish != "off"` |
| `benchmarks/benchmark_yield_sweep.py` | existing | add `relax_anchor` + `relax_tp_finish` configs |
| `viz/make_report.py` | existing | add anchor + TP-hybrid pages |

Run the full suite with `.venv/bin/python -m pytest -q`. Baseline is **179 passing**.

---

## Task 1: Config fields

**Files:** Modify `track_gen/types.py`.

- [ ] **Step 1: Add the two fields** to the `TrackGenConfig` dataclass, in the relaxation section (next to `relax_margin`):

```python
    relax_margin: float = 0.15
    relax_anchor: float = 0.0             # XPBD anti-creep: per-sweep pull toward pre-relax x0
    relax_tp_finish: str = "off"          # TP-Sobolev untangle finisher: {"off","all","failures"}
```

- [ ] **Step 2: Verify import + defaults** — Run:
`.venv/bin/python -c "from track_gen.types import TrackGenConfig as C; c=C(); print(c.relax_anchor, c.relax_tp_finish)"`
Expected: `0.0 off`

- [ ] **Step 3: Run the full suite to confirm no regression.**
Run: `.venv/bin/python -m pytest -q`  Expected: `179 passed`.

- [ ] **Step 4: Commit.**
```bash
git add track_gen/types.py
git commit -m "config: add relax_anchor + relax_tp_finish (default off)"
```

---

## Task 2: Anti-creep anchor (Warp kernel + torch parity)

**Files:**
- Modify `track_gen/warp_relax.py` (`_disp_kernel`, `xpbd_solve`)
- Modify `track_gen/relaxation.py` (`_relax_xpbd`)
- Test: `tests/test_relaxation_anchor.py` (create)

- [ ] **Step 1: Write the failing parity + deformation test.** Create `tests/test_relaxation_anchor.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Anti-creep anchor: Warp XPBD == torch XPBD with the anchor on, and the anchor reduces
deformation from the pre-relax shape."""
import pytest
import torch

pytest.importorskip("warp")
import warp as wp  # noqa: E402

wp.init()

from track_gen import warp_relax, relaxation, geometry  # noqa: E402
from track_gen.types import TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _band_L0(center, hw):
    band = (2.0 * hw / geometry.mean_seg_len(center)).round().long().clamp_min(1)
    L0 = geometry.perimeter(center) / center.shape[1]
    return band, L0


def _circle_batch(E, N, dev):
    import math
    t = torch.linspace(0, 2 * math.pi, N + 1, device=dev)[:-1]
    base = torch.stack([torch.cos(t), torch.sin(t)], -1)
    radii = torch.linspace(0.6, 1.4, E, device=dev).view(E, 1, 1)
    return (base.unsqueeze(0) * radii).to(torch.float32)


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("anchor", [0.0, 0.5])
def test_xpbd_anchor_warp_matches_torch(dev, anchor):
    E, N, hw = 4, 64, 0.05
    center0 = _circle_batch(E, N, dev)
    band, L0 = _band_L0(center0, hw)
    # Warp path (un-resampled).
    cfg = TrackGenConfig(num_envs=E, num_points=N, half_width=hw, relax_iters=40,
                         relax_anchor=anchor, device=dev)
    warp_out = warp_relax.xpbd_solve(center0, band, L0, cfg)
    # Torch reference: force the pure-torch loop (relax_use_warp=False) and read its
    # pre-resample result via the same sweep. We replicate _relax_xpbd's loop here so we
    # compare the SOLVE (not the trailing resample).
    D = 2.0 * hw
    margin = float(cfg.relax_margin)
    R_min = hw * (1.0 + margin)
    circ = geometry.circ_index_dist(N, center0.device)
    mask_keep = circ[None] > band.view(E, 1, 1)
    c = center0.clone()
    for _ in range(40):
        disp = relaxation._separation_disp(c, mask_keep, D, margin)
        disp = disp + relaxation._spacing_disp(c, L0)
        bend, toward = relaxation._bending_disp(c, R_min)
        step = 1.5 * bend
        max_len = torch.linalg.norm(toward, dim=-1, keepdim=True)
        step_len = torch.linalg.norm(step, dim=-1, keepdim=True)
        disp = disp + step * (max_len / step_len.clamp_min(1e-12)).clamp(max=1.0)
        disp = disp + anchor * (center0 - c)
        c = c + disp
    assert torch.allclose(warp_out, c, atol=1e-4), \
        f"{dev} anchor={anchor}: warp vs torch max err {(warp_out - c).abs().max().item():.2e}"


@pytest.mark.parametrize("dev", DEVS)
def test_anchor_reduces_deformation(dev):
    E, N, hw = 8, 64, 0.08
    center0 = _circle_batch(E, N, dev)
    band, L0 = _band_L0(center0, hw)

    def relaxed_drift(anchor):
        cfg = TrackGenConfig(num_envs=E, num_points=N, half_width=hw, relax_iters=120,
                             relax_anchor=anchor, device=dev)
        out = warp_relax.xpbd_solve(center0, band, L0, cfg)
        return torch.linalg.norm(out - center0, dim=-1).mean().item()

    assert relaxed_drift(0.5) < relaxed_drift(0.0), "anchor should reduce drift from x0"
```

- [ ] **Step 2: Run, expect failures** — `_disp_kernel` has no `x0`/`anchor` args yet, and `_relax_xpbd`/`xpbd_solve` ignore `relax_anchor`, so the `anchor=0.5` parity case and the drift case fail.
Run: `.venv/bin/python -m pytest tests/test_relaxation_anchor.py -q`  Expected: FAIL (`anchor=0.5` mismatch / drift not reduced).

- [ ] **Step 3: Add the anchor to the Warp kernel.** In `track_gen/warp_relax.py`, change `_disp_kernel`'s signature to add `x0` and `anchor` (before `out`), and add the anchor term to the final write:

```python
    @wp.kernel
    def _disp_kernel(center: wp.array(dtype=wp.vec2f), band: wp.array(dtype=wp.int32),
                     L0: wp.array(dtype=wp.float32), N: int, target: wp.float32, R_min: wp.float32,
                     sr: wp.float32, pr: wp.float32, br: wp.float32,
                     x0: wp.array(dtype=wp.vec2f), anchor: wp.float32,
                     out: wp.array(dtype=wp.vec2f)):
        # ... existing sep / spc / bending body unchanged ...
        # final line was: out[t] = sr * sep + pr * spc + bscale * toward
        out[t] = sr * sep + pr * spc + bscale * toward + anchor * (x0[t] - xi)
```
(`xi = center[t]` is already defined near the top of the kernel.)

- [ ] **Step 4: Pass `x0`/`anchor` from `xpbd_solve`.** In `track_gen/warp_relax.py::xpbd_solve`, after `cb`/`db` are created, add a frozen `x0` copy and read the anchor, and add both to the `_disp_kernel` launch inputs (before `dw`):

```python
    anchor = float(getattr(config, "relax_anchor", 0.0))
    x0b = center0.reshape(E * N, 2).contiguous().clone()   # frozen pre-relax positions
    x0w = wp.from_torch(x0b, dtype=wp.vec2f)
    # ... existing cw/dw/bw/lw setup ...
    for _ in range(int(config.relax_iters)):
        wp.launch(_disp_kernel, dim=E * N,
                  inputs=[cw, bw, lw, N, target, R_min, sr, pr, br, x0w, anchor, dw], device=dev)
        wp.launch(_apply_kernel, dim=E * N, inputs=[cw, dw], device=dev)
```
(Leave the existing `wp.synchronize()` / `_CAPTURING` guard and the `return cb.view(E, N, 2)` unchanged.)

- [ ] **Step 5: Add the anchor to the torch oracle.** In `track_gen/relaxation.py::_relax_xpbd`, read the anchor and add the term inside the torch loop just before `center = center + disp`:

```python
    anchor = float(getattr(config, "relax_anchor", 0.0))
    # ... existing setup; the torch loop ...
    for _ in range(int(config.relax_iters)):
        disp = sep_relax * _separation_disp(center, mask_keep, D, margin)
        disp = disp + spc_relax * _spacing_disp(center, L0)
        if bend_relax > 0.0:
            bend, toward = _bending_disp(center, R_min)
            step = bend_relax * bend
            max_len = torch.linalg.norm(toward, dim=-1, keepdim=True)
            step_len = torch.linalg.norm(step, dim=-1, keepdim=True)
            disp = disp + step * (max_len / step_len.clamp_min(1e-12)).clamp(max=1.0)
        disp = disp + anchor * (center0 - center)   # <-- anchor (matches the Warp kernel)
        center = center + disp
```

- [ ] **Step 6: Run the anchor test (cpu+cuda), expect pass.**
Run: `.venv/bin/python -m pytest tests/test_relaxation_anchor.py -q`  Expected: PASS (4 parity + N drift cases).

- [ ] **Step 7: Run the existing Warp-relax parity + full suite (anchor=0 must be unchanged).**
Run: `.venv/bin/python -m pytest tests/test_warp_relax.py -q && .venv/bin/python -m pytest -q`
Expected: `test_warp_relax` green; full suite `181 passed` (179 + the 2 new test functions' cases collapse per parametrization — confirm 0 failures).

- [ ] **Step 8: Commit.**
```bash
git add track_gen/warp_relax.py track_gen/relaxation.py tests/test_relaxation_anchor.py
git commit -m "relax: add anti-creep anchor (relax_anchor) to Warp XPBD + torch parity"
```

---

## Task 3: `_tp_flow` initial-active mask + untangle unit test

**Files:**
- Modify `track_gen/relaxation.py` (`_tp_flow`)
- Test: `tests/test_tp_untangle.py` (create)

- [ ] **Step 1: Write the failing untangle test.** Create `tests/test_tp_untangle.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""The TP-Sobolev self-avoiding solver reduces self-intersections, and its initial-active
mask leaves non-selected envs untouched."""
import math

import pytest
import torch

pytest.importorskip("warp")
import warp as wp  # noqa: E402

wp.init()

from track_gen import relaxation, geometry  # noqa: E402
from track_gen.types import TrackGenConfig  # noqa: E402


def _lemniscate(N, dev):
    # A self-crossing figure-eight (1 proper self-intersection) for the untangler to fix.
    t = torch.linspace(0, 2 * math.pi, N + 1, device=dev)[:-1]
    return torch.stack([torch.sin(t), torch.sin(t) * torch.cos(t)], -1).to(torch.float32)


def test_tp_flow_reduces_self_intersections():
    dev = "cpu"
    N = 128
    fig8 = _lemniscate(N, dev).unsqueeze(0)            # [1, N, 2]
    cfg = TrackGenConfig(num_envs=1, num_points=N, half_width=0.05, device=dev)
    band = (2.0 * 0.05 / geometry.mean_seg_len(fig8)).round().long().clamp_min(1)
    before = int(geometry.self_intersections(fig8)[0])
    assert before > 0
    out = relaxation._tp_flow(fig8, band, cfg, n_steps=60, tau=0.5, early_stop=False)
    after = int(geometry.self_intersections(out)[0])
    assert after < before, f"tp_flow should reduce crossings: {before} -> {after}"


def test_tp_flow_initial_active_freezes_unselected():
    dev = "cpu"
    N = 64
    t = torch.linspace(0, 2 * math.pi, N + 1, device=dev)[:-1]
    circle = torch.stack([torch.cos(t), torch.sin(t)], -1).to(torch.float32)
    batch = torch.stack([circle, circle * 1.3], 0)     # [2, N, 2]
    cfg = TrackGenConfig(num_envs=2, num_points=N, half_width=0.05, device=dev)
    band = (2.0 * 0.05 / geometry.mean_seg_len(batch)).round().long().clamp_min(1)
    active0 = torch.tensor([True, False], device=dev)   # only env 0 active
    out = relaxation._tp_flow(batch, band, cfg, n_steps=20, tau=0.3, early_stop=False,
                              initial_active=active0)
    # env 1 (inactive) is unchanged up to the trailing resample (which is ~identity on a
    # uniform circle); env 0 (active) moved.
    assert torch.allclose(out[1], relaxation._resample_uniform(batch[1:2], N)[0], atol=1e-4)
    assert not torch.allclose(out[0], batch[0], atol=1e-3)
```

- [ ] **Step 2: Run, expect failure** — `_tp_flow` does not accept `initial_active` yet.
Run: `.venv/bin/python -m pytest tests/test_tp_untangle.py -q`  Expected: FAIL (`unexpected keyword argument 'initial_active'`).

- [ ] **Step 3: Add the `initial_active` arg to `_tp_flow`.** In `track_gen/relaxation.py`, change the signature and the `active` initialization:

```python
def _tp_flow(center0, band, config, n_steps, tau, early_stop, initial_active=None):
    # ... existing setup ...
    # the line was: active = torch.ones(E, dtype=torch.bool, device=device)
    active = (torch.ones(E, dtype=torch.bool, device=device)
              if initial_active is None else initial_active.clone())
    # ... rest of the loop unchanged; when early_stop, active &= (th < target) as before.
```
When `early_stop=False`, `active` stays as `initial_active` for the whole flow (the `move = active[...] if early_stop else 1.0` line uses `1.0` when `early_stop=False`, so for the test we keep early_stop=False but still want masking) — **also** change that `move` line so the initial-active mask is honored even without early-stop:

```python
            move = active[:, None, None].to(center.dtype)   # honor active mask regardless of early_stop
            center = center - step * move
```
(Previously `move` was `1.0` when `early_stop=False`. Using the `active` mask always is correct: with `initial_active=None` it is all-True, so the default whole-batch finisher behavior is unchanged. Verify the existing `test_relaxation_finisher.py` still passes in Step 5.)

- [ ] **Step 4: Run the untangle test, expect pass.**
Run: `.venv/bin/python -m pytest tests/test_tp_untangle.py -q`  Expected: PASS.

- [ ] **Step 5: Run the existing tp / finisher tests + full suite (default behavior unchanged).**
Run: `.venv/bin/python -m pytest tests/test_relaxation_tp.py tests/test_relaxation_finisher.py -q && .venv/bin/python -m pytest -q`
Expected: all green, no regressions.

- [ ] **Step 6: Commit.**
```bash
git add track_gen/relaxation.py tests/test_tp_untangle.py
git commit -m "relax: _tp_flow honors an initial-active mask (failures-only untangling)"
```

---

## Task 4: Wire the TP finisher into the pipeline + graph guard

**Files:**
- Modify `track_gen/warp_pipeline.py` (`generate_tracks_warp`, `generate_tracks_warp_graph`)
- Test: `tests/test_tp_untangle.py` (extend), `tests/test_warp_graph.py` (extend)

- [ ] **Step 1: Write the failing pipeline + graph-guard tests.** Append to `tests/test_tp_untangle.py`:

```python
from track_gen import warp_pipeline as wpl  # noqa: E402


def test_tp_finish_raises_yield_on_fat_band():
    # 1m/20m fat-band regime: XPBD leaves ~1/3 invalid (looped). The TP finisher on
    # failures should untangle some -> strictly higher yield. Small E + modest iters (cpu).
    dev = "cpu"
    E = 24
    seeds = torch.arange(E, dtype=torch.int32, device=dev)
    base = dict(num_envs=E, num_points=128, half_width=0.5, scale=10.0,
                relax_iters=60, max_regen_iters=6, device=dev)
    y_off = wpl.generate_tracks_warp(TrackGenConfig(relax_tp_finish="off", **base),
                                     seeds).valid.float().mean().item()
    y_tp = wpl.generate_tracks_warp(TrackGenConfig(relax_tp_finish="failures", tp_iters=40,
                                                   **base), seeds).valid.float().mean().item()
    assert y_tp >= y_off, f"tp finisher should not lower yield: off={y_off} tp={y_tp}"
    # It should help at least sometimes; assert a non-trivial floor so a no-op wiring fails.
    assert y_tp > y_off or y_off >= 0.99


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_graph_rejects_tp_finish():
    E = 8
    cfg = TrackGenConfig(num_envs=E, num_points=128, half_width=0.5, scale=10.0,
                         relax_tp_finish="failures", device="cuda")
    seeds = torch.arange(E, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError):
        wpl.generate_tracks_warp_graph(cfg, seeds)
```
(Note: `test_tp_finish_raises_yield_on_fat_band` is mildly stochastic; if it proves flaky in CI, raise `E` to 64 and `max_regen_iters` to 10. Keep the `>=` assertion as the hard invariant.)

- [ ] **Step 2: Run, expect failure** — `generate_tracks_warp` currently `assert`s on / ignores `relax_tp_finish` and never calls the finisher; the graph fn has no `relax_tp_finish` guard.
Run: `.venv/bin/python -m pytest tests/test_tp_untangle.py::test_tp_finish_raises_yield_on_fat_band -q`  Expected: FAIL.

- [ ] **Step 3: Wire the finisher into `generate_tracks_warp`.** In `track_gen/warp_pipeline.py`, just before the final `return inflate_warp(relaxed, config, valid=gen_valid)`, insert:

```python
    if config.relax_tp_finish != "off":
        # Opt-in TP-Sobolev untangle finisher (torch; NOT graph-capturable in this mode).
        from . import relaxation
        tp_band = (2.0 * hw / _mean_seg_len_torch(relaxed).clamp_min(1e-9)).round().long().clamp_min(1)
        if config.relax_tp_finish == "failures":
            active0 = thickness(relaxed, tp_band) < (1.0 - float(config.relax_tol)) * hw
        else:  # "all"
            active0 = torch.ones(E, dtype=torch.bool, device=relaxed.device)
        relaxed = relaxation._tp_flow(relaxed, tp_band, config, n_steps=int(config.tp_iters),
                                      tau=float(config.tp_tau), early_stop=True,
                                      initial_active=active0)
    return inflate_warp(relaxed, config, valid=gen_valid)
```
(`_tp_flow` returns an `[E, N, 2]` resampled centerline, so no extra `resample_uniform` is needed. `thickness` and `_mean_seg_len_torch` are existing `warp_pipeline` symbols. Keep the existing `assert config.relax_solver == "xpbd"` and `assert not config.smooth_finish` lines as they are.)

- [ ] **Step 4: Add the graph guard.** In `track_gen/warp_pipeline.py::generate_tracks_warp_graph`, after the existing `assert seeds_template.is_cuda` line, add:

```python
    assert getattr(config, "relax_tp_finish", "off") == "off", \
        "CUDA graph capture requires the pure-Warp path; relax_tp_finish must be 'off'"
```

- [ ] **Step 5: Run the pipeline + graph tests, expect pass.**
Run: `.venv/bin/python -m pytest tests/test_tp_untangle.py tests/test_warp_graph.py -q`  Expected: PASS (cuda graph guard runs only if a GPU is present).

- [ ] **Step 6: Run the full suite (default `relax_tp_finish="off"` unchanged).**
Run: `.venv/bin/python -m pytest -q`  Expected: green, no regressions.

- [ ] **Step 7: Commit.**
```bash
git add track_gen/warp_pipeline.py tests/test_tp_untangle.py tests/test_warp_graph.py
git commit -m "pipeline: opt-in TP-Sobolev untangle finisher (relax_tp_finish) + graph guard"
```

---

## Task 5: Study sweep + report, and recommend a `relax_anchor` default

**Files:** Modify `benchmarks/benchmark_yield_sweep.py`, `viz/make_report.py`.

- [ ] **Step 1: Extend the sweep with anchor + tp_finish rows.** In `benchmarks/benchmark_yield_sweep.py`, give `bench()` two more kwargs and add config rows. Update the `bench` signature and the `TrackGenConfig(...)` call to pass them, and append to `CONFIGS`:

```python
def bench(links=256, iters=150, regen=10, sr=1.0, pr=1.0, br=1.5, margin=0.15,
          anchor=0.0, tp_finish="off", seed=0):
    cfg = TrackGenConfig(
        num_envs=E, num_points=links, half_width=HALF_WIDTH, scale=SCALE,
        relax_iters=iters, max_regen_iters=regen, relax_sep_relax=sr, relax_spc_relax=pr,
        relax_bend_relax=br, relax_margin=margin, relax_anchor=anchor,
        relax_tp_finish=tp_finish, device=DEVICE,
    )
    # ... rest unchanged; add anchor/tp_finish to the returned dict for the table ...
```
Append to `CONFIGS` (after the existing rows):
```python
    # anti-creep anchor sweep (256 links, iters=150)
    {"anchor": 0.05}, {"anchor": 0.15}, {"anchor": 0.3}, {"anchor": 0.6},
    # TP-Sobolev untangle finisher (256 links, iters=150)
    {"tp_finish": "failures"}, {"tp_finish": "all"},
```
Also add `anchor` and `tp_finish` to the `ROW`/`TABLE` print lines.

- [ ] **Step 2: Run the sweep (E=8192, serial; ~several minutes — the tp rows are slow torch).**
Run: `.venv/bin/python -m benchmarks.benchmark_yield_sweep | tee /tmp/relax_quality_sweep.txt`
Expected: a table including the anchor rows (yield vs drift) and the tp_finish rows (yield vs the 0.684 baseline). Record the numbers.

- [ ] **Step 3: Pick the recommended `relax_anchor`** = the largest anchor whose yield is within ~0.02 of the `anchor=0` baseline (so deformation drops without a yield hit). Note it in the report and in your final summary. **Do not change the default in `types.py`.**

- [ ] **Step 4: Add report pages.** In `viz/make_report.py`, add the measured anchor + tp_finish numbers to the data block, add a quantitative panel (yield vs anchor, with a drift overlay; yield: off vs tp-failures vs tp-all), and a fixed-seed page comparing a looped track XPBD-only vs +TP-finish (it should untangle). Reuse the existing `_fixed_seed_page` / `_gen` helpers (pass `relax_tp_finish` through `_gen`). Regenerate:
Run: `.venv/bin/python -m viz.make_report`  Expected: `wrote .../track_gen_report.pdf` (now with anchor + TP pages).

- [ ] **Step 5: Full suite green.**
Run: `.venv/bin/python -m pytest -q`  Expected: green.

- [ ] **Step 6: Commit.**
```bash
git add benchmarks/benchmark_yield_sweep.py viz/make_report.py viz/out/track_gen_report.pdf
git commit -m "study: sweep relax_anchor + TP-finish at E=8192; report pages + recommended anchor"
```

---

## Self-Review

**Spec coverage:** config knobs → Task 1; anchor (warp+torch parity, default 0) → Task 2; `_tp_flow` initial-active → Task 3; TP finisher wiring + scope (all/failures) + graph guard → Task 4; verification (parity test, deformation test, untangle test, pipeline-yield test) → Tasks 2–4; study/report + recommended anchor default → Task 5. All spec §3–§7 items mapped. Default-off invariants (graph capture, bit-identical XPBD at anchor=0) are asserted in Tasks 2/4.

**Placeholder scan:** No TBD/TODO; every code step shows the actual change. The one stochastic test (`test_tp_finish_raises_yield_on_fat_band`) has its hard invariant (`y_tp >= y_off`) plus a flakiness fallback noted.

**Type/name consistency:** `relax_anchor`/`relax_tp_finish` used identically in types/warp_relax/relaxation/warp_pipeline/benchmark; `_tp_flow(..., initial_active=...)` signature matches its caller in Task 4; `_disp_kernel`'s new `(x0, anchor, out)` arg order matches the `xpbd_solve` launch inputs; the anchor term `anchor*(x0 - x_current)` is identical in the Warp kernel and the torch loop (parity).

**Note on the `move` change (Task 3 Step 3):** honoring the `active` mask regardless of `early_stop` is behavior-preserving for `initial_active=None` (all-True); Step 5 re-runs `test_relaxation_finisher.py` to confirm.
