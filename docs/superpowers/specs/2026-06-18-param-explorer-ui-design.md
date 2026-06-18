# Track-Gen Parameter Explorer (Gradio UI) — Design

**Goal:** A nice, browser-based tool to interactively visualize how track-generation parameters impact the output — drag sliders / flip toggles, see a live grid of generated tracks plus the valid-yield and quality stats update.

**Non-goals:** Not a hosted/multi-user service (local launch only). No new pipeline logic — it drives the existing `generate_tracks_warp`. No persistence/export beyond what the browser/matplotlib give for free.

---

## Architecture

A thin Gradio shell over a **pure, UI-free core**, so the logic is testable without a browser:

- **`viz/param_explorer.py`** — contains:
  - `build_config(params) -> TrackGenConfig` — maps control values to a config (clamps degenerate inputs, e.g. `min_num_points = min(min, max)`).
  - `render_grid(params) -> (matplotlib.figure.Figure, stats: dict)` — builds the config, `seeds = arange(rows*cols) + seed`, runs `track_gen.warp_pipeline.generate_tracks_warp(config, seeds)`, draws an `rows×cols` grid reusing **`viz.plot_tracks.draw_track`** (consistent styling), and computes `stats`. This is the unit-tested core.
  - `build_app() -> gradio.Blocks` — the UI shell wiring controls → `render_grid` → (image, stats). `main()` launches it.
- **Device:** auto — `cuda` if `torch.cuda.is_available()` else `cpu`.
- **Dependency:** `gradio` as an **optional extra** in `pyproject.toml` (`[project.optional-dependencies] ui = ["gradio"]`); the pipeline core never imports it. `render_grid`/`build_config` import only `track_gen` (+ matplotlib), so they're testable without gradio.

## Controls (curated, grouped) — with ranges & defaults

| group | control | type | range | default |
|---|---|---|---|---|
| Regime | `half_width` | slider | 0.05–1.0 | 0.5 |
| Regime | `scale` | slider | 1–20 | 10 |
| Shape | `min_num_points` | int slider | 5–20 | 9 |
| Shape | `max_num_points` | int slider | 5–20 | 13 |
| Shape | `rad` | slider | 0.0–0.5 | 0.2 |
| Shape | `edgy` | slider | 0.0–1.0 | 0.0 |
| Res/mode | `output_mode` | radio | fixed / constant_spacing | constant_spacing |
| Res/mode | `num_points` (fixed) | int slider | 64–512 | 256 |
| Res/mode | `spacing` (constant) | slider | 0.1–1.0 | 0.30 |
| Res/mode | `N_max` (constant) | int slider | 128–512 | 384 |
| Relax | `relax_iters` | int slider | 0–600 | 150 |
| Relax | `max_regen_iters` | int slider | 1–20 | 10 |
| Relax | `relax_sep_relax` | slider | 0–2 | 1.0 |
| Relax | `relax_spc_relax` | slider | 0–2 | 1.0 |
| Relax | `relax_bend_relax` | slider | 0–2 | 1.5 |
| Relax | `relax_margin` | slider | 0–0.5 | 0.15 |
| Batch | grid size | dropdown | 3×3 / 4×4 / 5×5 / 6×6 | 4×4 |
| Batch | `seed` | int + **reroll** btn | 0–10⁶ | 0 |

The `output_mode` radio toggles visibility: `num_points` shown for fixed; `spacing` + `N_max` shown for constant_spacing.

## Stats readout

Computed from the returned `Track` over the batch:
- **valid yield** = `valid.float().mean()` (as %).
- **mean length** (over valid), **mean thickness** (over valid, via the per-track thickness), and in constant_spacing mode **mean count** (`count[valid].float().mean()`).
Rendered as a small text/markdown panel above the grid.

## Layout

Left: a controls column grouped (Regime / Shape / Resolution & mode / Relaxation / Batch) with a **Generate** button and an **auto-update** checkbox (default on). Right: the stats panel above an `rows×cols` track-grid image. (See the ASCII sketch in the brainstorming thread.)

## Interaction & data flow

- **auto-update on** (default): control changes trigger `render_grid` (Gradio debounces release events; a 16–36-track batch is ~0.1–0.25 s warm). The **Generate** button is the explicit trigger and the path for heavier settings (large grid × high `relax_iters` × big `num_points`).
- **reroll** bumps `seed` by 1 and regenerates.
- output_mode change re-shows the relevant resolution controls and regenerates.

## Error handling

- `build_config` clamps degenerate inputs (`min_num_points ← min(min,max)`, etc.).
- Invalid / NaN tracks render as red-titled "INVALID" cells (already how `draw_track` behaves), so failures are visible, not crashes.
- `render_grid` wraps the pipeline call in try/except; on error it returns a small "error" figure + a message string so the UI shows the error instead of dying.

## Testing

`tests/test_param_explorer.py` (CPU, CI-safe, GPU-free):
- `render_grid` with default params returns a `matplotlib.figure.Figure` and a stats dict with `0 ≤ yield ≤ 1`; the grid has exactly `rows*cols` axes.
- constant_spacing params path runs and reports a `mean count`; fixed path reports `count == num_points`.
- `build_config` clamps `min_num_points > max_num_points` (assert `cfg.min_num_points ≤ cfg.max_num_points`).
- App-builds smoke test: `build_app()` returns a `gradio.Blocks` without launching — **skipped via `importorskip("gradio")`** so the suite passes without the `ui` extra.

The Gradio event wiring itself is not unit-tested (thin shell); correctness lives in the tested `render_grid`/`build_config` core.

## File structure

| file | responsibility | change |
|---|---|---|
| `viz/param_explorer.py` | `build_config`, `render_grid`, `build_app`, `main` | create |
| `viz/plot_tracks.py` | `draw_track` reused by `render_grid` | reuse (no change) |
| `pyproject.toml` | `ui = ["gradio"]` optional extra | modify |
| `tests/test_param_explorer.py` | core unit tests + app-builds smoke | create |
| `README.md` | "Parameter explorer" launch note | modify |

### Batch size & pagination

A `batch_size` control (dropdown: 256 / 1024 / 2048 / 4096 / 8192, default 2048) sets how many
tracks are generated per run. Stats (yield, mean length, mean thickness, mean count) are computed
over the **full batch** — not just the visible page — for honest statistics at large N. The grid
shows a `grid_n×grid_n` page of the cached `Track`; ◀ prev / next ▶ buttons re-draw slices of
the cached object without regenerating. Only Generate / reroll / param changes trigger a new batch.

## Success criteria

- `pip install -e ".[ui]"` then `python -m viz.param_explorer` opens a browser page where changing any listed parameter re-renders the track grid and updates the yield/quality stats live (or on Generate).
- Both `output_mode`s work; the mode-specific controls show/hide correctly.
- Degenerate inputs and invalid tracks never crash the app.
- The full test suite stays green with and without `gradio` installed (core tests run; the gradio smoke test skips when absent).
