# Track-Gen Parameter Explorer (Gradio UI) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser-based Gradio tool to interactively visualize how track-generation parameters affect the output — sliders/toggles drive the real `generate_tracks_warp`, showing a live track grid + valid-yield/quality stats.

**Architecture:** A thin Gradio shell over a pure, UI-free core. `build_config(params)→TrackGenConfig` and `render_grid(params)→(matplotlib Figure, stats dict)` are the testable core (import only `track_gen` + matplotlib); `build_app()→gr.Blocks` wires controls to `render_grid`. `gradio` is an optional extra so the core (and the test suite) work without it.

**Tech Stack:** Gradio (optional extra), matplotlib (Agg), PyTorch + NVIDIA Warp (the existing pipeline), pytest. Env: `.venv/bin/python`. Spec: `docs/superpowers/specs/2026-06-18-param-explorer-ui-design.md`. Baseline suite: **205 passing**.

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `viz/param_explorer.py` | `build_config`, `render_grid` (core) + `build_app`, `main` (Gradio shell) | create |
| `viz/plot_tracks.py` | `draw_track(ax, track, e)` reused by `render_grid` | reuse, no change |
| `pyproject.toml` | add `ui = ["gradio"]` optional extra | modify |
| `tests/test_param_explorer.py` | core unit tests (cpu) + gradio app-builds smoke | create |
| `README.md` | "Parameter explorer" launch note | modify |

**Params dict (single source of truth, used by `build_config` and `render_grid`):** keys
`half_width, scale, min_num_points, max_num_points, rad, edgy, output_mode, num_points,
spacing, n_max, relax_iters, max_regen_iters, relax_sep_relax, relax_spc_relax,
relax_bend_relax, relax_margin, grid_n, seed`. Grid is square: `E = grid_n**2`.

---

## Task 1: `build_config` — params → TrackGenConfig (with clamping)

**Files:** Create `viz/param_explorer.py`; create `tests/test_param_explorer.py`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_param_explorer.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Parameter-explorer core (build_config + render_grid) on CPU, plus a gradio app-builds smoke."""
import pytest

from viz import param_explorer as px


def _params(**over):
    p = dict(half_width=0.5, scale=10.0, min_num_points=9, max_num_points=13, rad=0.2, edgy=0.0,
             output_mode="fixed", num_points=256, spacing=0.30, n_max=384,
             relax_iters=40, max_regen_iters=3, relax_sep_relax=1.0, relax_spc_relax=1.0,
             relax_bend_relax=1.5, relax_margin=0.15, grid_n=3, seed=0)
    p.update(over)
    return p


def test_build_config_maps_and_clamps():
    cfg = px.build_config(_params(min_num_points=15, max_num_points=8))
    # corners clamped so min <= max
    assert cfg.min_num_points <= cfg.max_num_points
    assert cfg.num_envs == 9                      # grid_n**2
    assert cfg.output_mode == "fixed"
    assert abs(cfg.half_width - 0.5) < 1e-9 and abs(cfg.scale - 10.0) < 1e-9
    cfg2 = px.build_config(_params(output_mode="constant_spacing", spacing=0.3, n_max=384))
    assert cfg2.output_mode == "constant_spacing" and cfg2.N_max == 384
```

- [ ] **Step 2: Run, expect failure** (module/function missing).
Run: `.venv/bin/python -m pytest tests/test_param_explorer.py::test_build_config_maps_and_clamps -q`  Expected: FAIL (ImportError / AttributeError).

- [ ] **Step 3: Create `viz/param_explorer.py` with the header + `build_config`:**

```python
#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Interactive Gradio explorer for track-generation parameters.

A thin Gradio shell over a pure core (`build_config` + `render_grid`) that drives the real
pure-Warp pipeline (`track_gen.warp_pipeline.generate_tracks_warp`) and renders a grid of
tracks + yield/quality stats. Launch:  `.venv/bin/python -m viz.param_explorer`
(needs the `ui` extra: `pip install -e ".[ui]"`).
"""
from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot

import matplotlib.pyplot as plt
import torch

import warp as wp

from track_gen.types import TrackGenConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_config(p: dict) -> TrackGenConfig:
    """Map a params dict to a TrackGenConfig, clamping degenerate inputs."""
    lo = min(int(p["min_num_points"]), int(p["max_num_points"]))
    hi = max(int(p["min_num_points"]), int(p["max_num_points"]))
    grid_n = int(p["grid_n"])
    return TrackGenConfig(
        num_envs=grid_n * grid_n,
        num_points=int(p["num_points"]),
        half_width=float(p["half_width"]),
        scale=float(p["scale"]),
        min_num_points=lo,
        max_num_points=hi,
        rad=float(p["rad"]),
        edgy=float(p["edgy"]),
        output_mode=str(p["output_mode"]),
        spacing=float(p["spacing"]),
        N_max=int(p["n_max"]),
        relax_iters=int(p["relax_iters"]),
        max_regen_iters=int(p["max_regen_iters"]),
        relax_sep_relax=float(p["relax_sep_relax"]),
        relax_spc_relax=float(p["relax_spc_relax"]),
        relax_bend_relax=float(p["relax_bend_relax"]),
        relax_margin=float(p["relax_margin"]),
        device=DEVICE,
    )
```

- [ ] **Step 4: Run, expect pass.**
Run: `.venv/bin/python -m pytest tests/test_param_explorer.py::test_build_config_maps_and_clamps -q`  Expected: PASS.

- [ ] **Step 5: Commit.**
```bash
git add viz/param_explorer.py tests/test_param_explorer.py
git commit -m "viz(explorer): build_config — params -> TrackGenConfig with clamping"
```

---

## Task 2: `render_grid` — generate + draw + stats

**Files:** Modify `viz/param_explorer.py`; modify `tests/test_param_explorer.py`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_param_explorer.py`:

```python
import matplotlib.figure


def test_render_grid_fixed():
    fig, stats = px.render_grid(_params(grid_n=3, output_mode="fixed", num_points=128))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 9                       # grid_n**2 cells
    assert 0.0 <= stats["yield"] <= 1.0
    assert stats["count"] == 128                    # fixed mode: count == num_points
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_render_grid_constant_spacing_runs():
    fig, stats = px.render_grid(_params(grid_n=3, output_mode="constant_spacing",
                                        spacing=0.30, n_max=384))
    assert 0.0 <= stats["yield"] <= 1.0
    assert stats["count"] >= 1                       # variable per-track count (mean over valid)
    import matplotlib.pyplot as plt
    plt.close(fig)
```

- [ ] **Step 2: Run, expect failure** (`render_grid` missing).
Run: `.venv/bin/python -m pytest tests/test_param_explorer.py -q`  Expected: the two new tests FAIL (AttributeError).

- [ ] **Step 3: Add `render_grid` (and stats) to `viz/param_explorer.py`:**

```python
from track_gen import warp_pipeline as wpl
from viz.plot_tracks import draw_track


def _stats(track, output_mode: str, num_points: int) -> dict:
    """Aggregate readout over the batch (means taken over valid tracks)."""
    valid = track.valid
    n = int(valid.sum())
    if n == 0:
        return {"yield": 0.0, "n_valid": 0, "mean_len": float("nan"),
                "mean_thickness": float("nan"), "count": (num_points if output_mode == "fixed" else 0)}
    # count-aware thickness (works in both modes): band = round(2*hw/(len/count)).clamp_min(1)
    hw = float(torch.linalg.norm(track.outer[:, 0] - track.center[:, 0], dim=-1).median())
    cnt = track.count.clamp_min(1)
    band = (2.0 * hw / (track.length / cnt.float()).clamp_min(1e-9)).round().to(torch.int32).clamp_min(1)
    th = wpl.thickness(track.center, band, count=track.count.to(torch.int32))
    return {
        "yield": float(valid.float().mean()),
        "n_valid": n,
        "mean_len": float(track.length[valid].mean()),
        "mean_thickness": float(th[valid].mean()),
        "count": (num_points if output_mode == "fixed" else float(track.count[valid].float().mean())),
    }


def render_grid(p: dict):
    """Generate grid_n**2 tracks for the given params; return (matplotlib Figure, stats dict).
    Any pipeline error is caught and returned as a small error figure + an 'error' stat."""
    wp.init()
    grid_n = int(p["grid_n"])
    try:
        cfg = build_config(p)
        seeds = (torch.arange(grid_n * grid_n, dtype=torch.int32) + int(p["seed"])).to(DEVICE)
        track = wpl.generate_tracks_warp(cfg, seeds)
        fig, axes = plt.subplots(grid_n, grid_n, figsize=(2.1 * grid_n, 2.1 * grid_n))
        axes = axes.flatten() if grid_n > 1 else [axes]
        for k, ax in enumerate(axes):
            draw_track(ax, track, k)
        st = _stats(track, str(p["output_mode"]), int(p["num_points"]))
        fig.suptitle(f"{grid_n}x{grid_n}  ·  valid {st['yield'] * 100:.0f}%", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig, st
    except Exception as exc:  # never crash the UI
        fig = plt.figure(figsize=(5, 3))
        fig.text(0.5, 0.5, f"error: {exc}", ha="center", va="center", fontsize=9, color="red", wrap=True)
        return fig, {"error": str(exc), "yield": 0.0, "n_valid": 0, "mean_len": float("nan"),
                     "mean_thickness": float("nan"), "count": 0}
```

- [ ] **Step 4: Run, expect pass.**
Run: `.venv/bin/python -m pytest tests/test_param_explorer.py -q`  Expected: PASS (build_config + both render tests).

- [ ] **Step 5: Full suite (no regression — the explorer isn't imported elsewhere).**
Run: `.venv/bin/python -m pytest -q`  Expected: green (`205` + the 3 new explorer tests).

- [ ] **Step 6: Commit.**
```bash
git add viz/param_explorer.py tests/test_param_explorer.py
git commit -m "viz(explorer): render_grid — generate batch, draw grid, yield/quality stats"
```

---

## Task 3: optional `ui` extra + README launch note

**Files:** Modify `pyproject.toml`; modify `README.md`.

- [ ] **Step 1: Add the optional extra.** In `pyproject.toml`, under `[project.optional-dependencies]` (next to `warp`/`dev`), add:

```toml
ui = ["gradio"]
```

- [ ] **Step 2: Verify it parses.**
Run: `.venv/bin/python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"`  Expected: `ok`.

- [ ] **Step 3: Add the README note.** In `README.md`, after the "## Development" section, add:

```markdown
## Parameter explorer (UI)

An interactive Gradio app to see how each parameter affects generation — sliders for the
regime / shape / resolution / relaxation knobs, a live track grid, and the valid-yield stat.

```bash
.venv/bin/pip install -e ".[ui]"     # adds gradio
.venv/bin/python -m viz.param_explorer   # opens a local URL
```
```

- [ ] **Step 4: Commit.**
```bash
git add pyproject.toml README.md
git commit -m "viz(explorer): add optional [ui] extra (gradio) + README launch note"
```

---

## Task 4: `build_app` (Gradio Blocks) + `main` + smoke test

**Files:** Modify `viz/param_explorer.py`; modify `tests/test_param_explorer.py`.

- [ ] **Step 1: Write the app-builds smoke test.** Append to `tests/test_param_explorer.py`:

```python
def test_build_app_smoke():
    gr = pytest.importorskip("gradio")            # skip if the ui extra isn't installed
    app = px.build_app()
    assert isinstance(app, gr.Blocks)
```

- [ ] **Step 2: Run.** If gradio is absent it SKIPS; if present it FAILS (no `build_app`).
Run: `.venv/bin/python -m pytest tests/test_param_explorer.py::test_build_app_smoke -q`  Expected: SKIP (no gradio) or FAIL (gradio present, `build_app` missing).

- [ ] **Step 3: Add `build_app` + `main` to `viz/param_explorer.py`:**

```python
def _collect(*vals) -> dict:
    keys = ["half_width", "scale", "min_num_points", "max_num_points", "rad", "edgy",
            "output_mode", "num_points", "spacing", "n_max", "relax_iters", "max_regen_iters",
            "relax_sep_relax", "relax_spc_relax", "relax_bend_relax", "relax_margin",
            "grid_n", "seed"]
    return dict(zip(keys, vals))


def _stats_md(st: dict) -> str:
    if "error" in st:
        return f"**error:** {st['error']}"
    return (f"**valid yield: {st['yield'] * 100:.0f}%**  ·  {st['n_valid']} valid  ·  "
            f"mean length {st['mean_len']:.1f} m  ·  mean thickness {st['mean_thickness']:.3f} m  ·  "
            f"mean count {st['count']:.0f}")


def build_app():
    """Build the Gradio Blocks UI (does not launch). Requires the `ui` extra."""
    import gradio as gr

    with gr.Blocks(title="Track-gen parameter explorer") as app:
        gr.Markdown("## Track-gen parameter explorer")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Regime")
                half_width = gr.Slider(0.05, 1.0, value=0.5, step=0.01, label="half_width (m)")
                scale = gr.Slider(1.0, 20.0, value=10.0, step=0.5, label="scale (box)")
                gr.Markdown("### Shape")
                min_np = gr.Slider(5, 20, value=9, step=1, label="min corners")
                max_np = gr.Slider(5, 20, value=13, step=1, label="max corners")
                rad = gr.Slider(0.0, 0.5, value=0.2, step=0.01, label="rad (roundness)")
                edgy = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="edgy")
                gr.Markdown("### Resolution & mode")
                output_mode = gr.Radio(["fixed", "constant_spacing"], value="fixed", label="output_mode")
                num_points = gr.Slider(64, 512, value=256, step=8, label="num_points (links)")
                spacing = gr.Slider(0.1, 1.0, value=0.30, step=0.02, label="spacing (m)", visible=False)
                n_max = gr.Slider(128, 512, value=384, step=8, label="N_max", visible=False)
                gr.Markdown("### Relaxation")
                relax_iters = gr.Slider(0, 600, value=150, step=10, label="relax_iters")
                max_regen = gr.Slider(1, 20, value=10, step=1, label="max_regen_iters")
                sep = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="sep factor")
                spc = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="spc factor")
                bend = gr.Slider(0.0, 2.0, value=1.5, step=0.1, label="bend factor")
                margin = gr.Slider(0.0, 0.5, value=0.15, step=0.01, label="relax_margin")
                gr.Markdown("### Batch")
                grid_n = gr.Dropdown([3, 4, 5, 6], value=4, label="grid (n x n)")
                seed = gr.Number(value=0, precision=0, label="seed")
                with gr.Row():
                    reroll = gr.Button("reroll seed")
                    generate = gr.Button("Generate", variant="primary")
                auto = gr.Checkbox(value=True, label="auto-update")
            with gr.Column(scale=2):
                stats = gr.Markdown("")
                plot = gr.Plot()

        controls = [half_width, scale, min_np, max_np, rad, edgy, output_mode, num_points,
                    spacing, n_max, relax_iters, max_regen, sep, spc, bend, margin, grid_n, seed]

        def _run(*vals):
            fig, st = render_grid(_collect(*vals))
            return fig, _stats_md(st)

        # mode-aware visibility
        def _toggle(mode):
            fixed = mode == "fixed"
            return gr.update(visible=fixed), gr.update(visible=not fixed), gr.update(visible=not fixed)
        output_mode.change(_toggle, output_mode, [num_points, spacing, n_max])

        # explicit Generate
        generate.click(_run, controls, [plot, stats])
        # reroll: bump seed then regenerate
        reroll.click(lambda s: int(s) + 1, seed, seed).then(_run, controls, [plot, stats])

        # auto-update: regenerate when any control settles (only if 'auto' is on)
        def _maybe(*vals):
            *rest, auto_on = vals
            if not auto_on:
                return gr.update(), gr.update()
            return _run(*rest)
        for c in controls:
            ev = c.release if hasattr(c, "release") else c.change
            ev(_maybe, controls + [auto], [plot, stats])

        app.load(_run, controls, [plot, stats])   # initial render
    return app


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test.** If gradio is installed it should PASS; otherwise SKIP.
Run: `.venv/bin/pip install -e ".[ui]" && .venv/bin/python -m pytest tests/test_param_explorer.py::test_build_app_smoke -q`  Expected: PASS. (If the installed gradio API rejects any kwarg above, adjust to that version — e.g. `gr.Plot()`/event signatures — keeping behaviour identical.)

- [ ] **Step 5: Full suite (with and without behaviour change to the core).**
Run: `.venv/bin/python -m pytest -q`  Expected: green (core tests + the now-passing smoke test).

- [ ] **Step 6: Manual launch check (optional, not in CI).**
Run: `.venv/bin/python -m viz.param_explorer`  Expected: prints a local URL; the page shows the controls + an initial 4×4 grid; toggling `output_mode` swaps `num_points` ↔ `spacing`/`N_max`; dragging a slider (auto-update on) re-renders.

- [ ] **Step 7: Commit.**
```bash
git add viz/param_explorer.py tests/test_param_explorer.py
git commit -m "viz(explorer): Gradio Blocks app (mode-aware controls, live + Generate, reroll)"
```

---

## Self-Review

**Spec coverage:** Gradio shell + pure core (Task 1 `build_config`, Task 2 `render_grid`) → all spec §Architecture; the curated control set + ranges/defaults → Task 4 `build_app`; mode-aware visibility → Task 4 `_toggle`; live/Generate/reroll → Task 4 events; stats (yield/len/thickness/count) → Task 2 `_stats`; optional `ui` extra → Task 3; README → Task 3; tests (core on cpu + gradio smoke skip-if-absent) → Tasks 1/2/4. All spec sections mapped.

**Placeholder scan:** no TBD/TODO; every code step shows the actual code. The one adaptivity note (Task 4 Step 4 "adjust to the installed gradio version") is a real caveat, not a placeholder — the `gr.Blocks`/`Slider`/`Radio`/`Plot` API used is stable across recent gradio, but the implementer verifies against the pinned version.

**Type/name consistency:** the params-dict keys are identical in `build_config`, `render_grid`, `_collect`, and the `controls` order (Task 4) ↔ `_collect` keys (Task 4) ↔ `_params` test helper (Task 1) — all 18 keys in the same order. `render_grid`/`build_config` take a single `p`/`params` dict; `_stats` returns the keys `_stats_md` reads (`yield`, `n_valid`, `mean_len`, `mean_thickness`, `count`, optional `error`). `wpl.thickness(points, band, count=)` matches the count-aware signature shipped in the constant-spacing work.

**Risk note:** `_collect`'s key order MUST match the `controls` list order — both are spelled out in Task 4; the implementer must keep them aligned (the `controls` list has 18 entries matching the 18 `_collect` keys; `auto` is appended only for the `_maybe` handler, not in `controls`).
