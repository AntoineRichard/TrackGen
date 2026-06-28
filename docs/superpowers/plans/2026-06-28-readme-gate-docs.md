# README Gate Documentation + Collision-Phase Imagery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the four documented gaps in the README's gate section and add a deterministic before/after image that proves the phase-2 gate collision solve runs.

**Architecture:** One self-contained gate renderer added to `viz/render_readme_assets.py` (public-API-only, CPU, fixed seeds) produces `docs/assets/readme-gate-strip.png` as a 2×5 raw-vs-solved figure; the rest is README prose — a `GateSequence` result table, `gate_width`/`min_gates`/`max_gates` semantics, and a Gates-explorer-tab block.

**Tech Stack:** Python ≥ 3.10, NVIDIA Warp (`warp-lang`), numpy, matplotlib + torch (dev extra, via `wp.to_torch`), pytest.

## Global Constraints

- The gate renderer MUST be self-contained: import only from the public `track_gen` package plus `matplotlib`/`numpy`/`warp` (+ `wp.to_torch`). Do NOT import from `viz.param_explorer`.
- Renderer runs on the Warp `cpu` device with fixed seeds — deterministic, no `cuda`, no randomness outside seeded RNG.
- No changes to `track_gen` core or to the explorer UI (`viz/param_explorer.py`). Docs + the new renderer only.
- Match the existing `viz/render_readme_assets.py` style and reuse its `_style_axis` / `_finite_rows` helpers.
- New committed asset path is exactly `docs/assets/readme-gate-strip.png`.
- `gate_width` is the FULL gate opening (`left/right = center ± 0.5*gate_width*normal`); default `0.0` ⇒ point gates.
- Registered gate generators: `bezier` (default), `checkpoint`, `hull`, `polar`, `voronoi`.

---

## File Structure

- `viz/render_readme_assets.py` — **Modify.** Add gate constants, gate-drawing helpers, `render_gate_assets()`, and wire it into `render_readme_assets()`.
- `tests/test_readme_assets.py` — **Create.** One slow smoke test that the gate renderer writes a non-trivial PNG.
- `docs/assets/readme-gate-strip.png` — **Create (generated).** Committed binary asset.
- `README.md` — **Modify.** Gate section (result table + aliasing + `gate_width`/`min_gates` prose + embedded image) and Parameter-explorer section (Gates tab block).

---

### Task 1: Gate asset renderer + committed image

**Files:**
- Modify: `viz/render_readme_assets.py`
- Test: `tests/test_readme_assets.py` (create)
- Create (generated): `docs/assets/readme-gate-strip.png`

**Interfaces:**
- Consumes: public API `track_gen.GateGenConfig`, `track_gen.GateGenerator`, `track_gen.PerEnvSeededRNG`; existing module helpers `_style_axis`, `OUT_DIR`.
- Produces: `render_gate_assets(output_dir: Path = OUT_DIR) -> Path` returning the written PNG path; `render_readme_assets()` now also returns that path in its list.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_readme_assets.py`:

```python
"""Smoke test for the README gate asset renderer."""
import pytest


@pytest.mark.slow
def test_render_gate_assets_writes_png(tmp_path):
    from viz.render_readme_assets import render_gate_assets

    path = render_gate_assets(output_dir=tmp_path)

    assert path.name == "readme-gate-strip.png"
    assert path.exists()
    assert path.stat().st_size > 1000  # a real, non-empty PNG
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_readme_assets.py -v -m slow`
Expected: FAIL with `ImportError: cannot import name 'render_gate_assets'`.

- [ ] **Step 3: Add gate constants + helpers to `viz/render_readme_assets.py`**

Insert after the `GENERATORS = [...]` block (around line 33):

```python
GATE_GENERATORS = [
    ("bezier", "Bezier"),
    ("checkpoint", "Checkpoint"),
    ("hull", "Hull"),
    ("polar", "Polar"),
    ("voronoi", "Voronoi"),
]

# Illustrative gate geometry for the asset: a non-zero opening so the gate bars are
# visible, and a radius large enough that raw anchors overlap before the collision solve
# (so the phase-2 separation is unmistakable in the before/after).
GATE_ASSET_GATE_WIDTH = 0.16
GATE_ASSET_GATE_RADIUS = 0.06
GATE_ASSET_SOLVE_ITERS = 16


def _gate_batch(name: str, *, seed: int, solve_iters: int):
    """Run one CPU gate batch; return (position, tangent, left, right, valid, count) numpy."""
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG

    batch = 24
    cfg = GateGenConfig(
        generator=name,
        num_envs=batch,
        device="cpu",
        gate_radius=GATE_ASSET_GATE_RADIUS,
        gate_width=GATE_ASSET_GATE_WIDTH,
        gate_solve_iters=solve_iters,
    )
    rng = PerEnvSeededRNG(seeds=seed, num_envs=batch, device="cpu")
    gates = GateGenerator(cfg, rng).generate()
    g = gates.position.shape[0] // batch
    position = wp.to_torch(gates.position).cpu().numpy().reshape(batch, g, 2)
    tangent = wp.to_torch(gates.tangent).cpu().numpy().reshape(batch, g, 2)
    left = wp.to_torch(gates.left).cpu().numpy().reshape(batch, g, 2)
    right = wp.to_torch(gates.right).cpu().numpy().reshape(batch, g, 2)
    valid = wp.to_torch(gates.valid).cpu().numpy().astype(bool)
    count = wp.to_torch(gates.count).cpu().numpy().astype(int)
    return position, tangent, left, right, valid, count


def _choose_gate_env(valid: np.ndarray, count: np.ndarray, position: np.ndarray) -> int:
    """Pick a representative env: valid, finite, and with at least 4 gates; else first finite."""
    for e in range(position.shape[0]):
        c = int(count[e])
        if valid[e] and c >= 4 and np.isfinite(position[e, :c]).all():
            return e
    for e in range(position.shape[0]):
        c = int(count[e])
        if c >= 2 and np.isfinite(position[e, :c]).all():
            return e
    return 0


def _set_gate_limits(ax, pts: np.ndarray) -> None:
    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) < 2:
        return
    xmin, ymin = finite.min(axis=0)
    xmax, ymax = finite.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1.0e-3)
    pad = 0.18 * span + 1.6 * GATE_ASSET_GATE_RADIUS
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    ax.set_xlim(cx - 0.5 * span - pad, cx + 0.5 * span + pad)
    ax.set_ylim(cy - 0.5 * span - pad, cy + 0.5 * span + pad)


def _draw_gate_asset(ax, position, tangent, left, right, count, e: int, *, draw_frames: bool) -> None:
    c = int(count[e])
    pos = position[e, :c]
    finite = np.isfinite(pos).all(axis=1)
    pos = pos[finite]
    if len(pos) >= 2:
        closed = np.vstack([pos, pos[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="0.35", lw=0.7, ls="--", alpha=0.6, zorder=1)
    if len(pos) > 0:
        ax.scatter(pos[:, 0], pos[:, 1], s=16, color="#111827", zorder=4)
    for pnt in pos:
        ax.add_patch(plt.Circle((pnt[0], pnt[1]), GATE_ASSET_GATE_RADIUS, fill=False,
                                color="#64748b", lw=0.8, alpha=0.85, zorder=2))
    if draw_frames:
        tan = tangent[e, :c][finite]
        lft = left[e, :c][finite]
        rgt = right[e, :c][finite]
        if len(tan) == len(pos) and len(pos) > 0:
            ax.quiver(pos[:, 0], pos[:, 1], tan[:, 0], tan[:, 1], angles="xy",
                      scale_units="xy", scale=12, width=0.005, color="#f97316",
                      alpha=0.85, zorder=3)
        if len(lft) == len(rgt) == len(pos):
            for li, ri in zip(lft, rgt):
                ax.plot([li[0], ri[0]], [li[1], ri[1]], color="#2563eb", lw=1.3, zorder=3)
    _set_gate_limits(ax, pos)
    _style_axis(ax)
```

- [ ] **Step 4: Add `render_gate_assets()` to `viz/render_readme_assets.py`**

Insert immediately before the `def main()` definition (near the bottom of the file):

```python
def render_gate_assets(output_dir: Path = OUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    wp.init()
    n = len(GATE_GENERATORS)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 4.8), dpi=170, facecolor="white")
    for col, (name, label) in enumerate(GATE_GENERATORS):
        seed = 100 + 23 * col
        solved = _gate_batch(name, seed=seed, solve_iters=GATE_ASSET_SOLVE_ITERS)
        s_pos, s_tan, s_left, s_right, s_valid, s_count = solved
        env = _choose_gate_env(s_valid, s_count, s_pos)
        raw = _gate_batch(name, seed=seed, solve_iters=0)
        r_pos, r_tan, r_left, r_right, r_valid, r_count = raw
        _draw_gate_asset(axes[0, col], r_pos, r_tan, r_left, r_right, r_count, env, draw_frames=False)
        _draw_gate_asset(axes[1, col], s_pos, s_tan, s_left, s_right, s_count, env, draw_frames=True)
        axes[0, col].set_title(label, fontsize=12, fontweight="bold", color="#111827", pad=8)
    axes[0, 0].set_ylabel("raw anchors\n(gate_solve_iters=0)", rotation=0, ha="right",
                          va="center", labelpad=18, fontsize=9.5, fontweight="bold", color="#111827")
    axes[1, 0].set_ylabel("collision-solved", rotation=0, ha="right", va="center",
                          labelpad=18, fontsize=9.5, fontweight="bold", color="#111827")
    fig.suptitle("Phase-2 gate collision solve: raw anchors vs separated gates",
                 fontsize=15, fontweight="bold", y=1.0, color="#111827")
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.96), h_pad=0.6, w_pad=0.3)
    path = output_dir / "readme-gate-strip.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
```

- [ ] **Step 5: Wire the gate renderer into `render_readme_assets()` and `main()`**

In `render_readme_assets()`, change the final `return` (currently `return [grid_path, pipeline_path, strip_path]`) to:

```python
    gate_strip_path = render_gate_assets(output_dir)

    return [grid_path, pipeline_path, strip_path, gate_strip_path]
```

(`main()` already iterates the returned list and prints each path — no change needed there.)

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_readme_assets.py -v -m slow`
Expected: PASS.

- [ ] **Step 7: Generate the committed asset**

Run: `.venv/bin/python -m viz.render_readme_assets`
Expected: prints four paths including `docs/assets/readme-gate-strip.png`.

- [ ] **Step 8: Eyeball the asset**

Open `docs/assets/readme-gate-strip.png`. Confirm: top row shows **overlapping** gate-radius circles (raw anchors); bottom row shows **separated/tangent** circles plus orange tangent arrows and blue gate bars. If the top row does not overlap for some generator, raise `GATE_ASSET_GATE_RADIUS` (e.g. to `0.08`) and re-run Step 7.

- [ ] **Step 9: Commit**

```bash
git add viz/render_readme_assets.py tests/test_readme_assets.py docs/assets/readme-gate-strip.png
git commit -m "feat: render before/after gate collision asset for README"
```

---

### Task 2: README gate section — result table, aliasing, `gate_width`/`min_gates` prose, embedded image

**Files:**
- Modify: `README.md` (gate section, currently lines 97–148)

**Interfaces:**
- Consumes: the committed `docs/assets/readme-gate-strip.png` from Task 1.
- Produces: nothing code-facing (docs).

- [ ] **Step 1: Insert the result table + semantics after the gate example**

In `README.md`, find the line:

```
Registered first-stage gate generators are selected with `GateGenConfig(generator=...)`:
```

Insert the following block immediately BEFORE that line (after the example's closing ```` ``` ````):

```markdown
The same `GateSequence` instance and its Warp buffers are reused on every `generate()` call;
use `gates.clone()` when you need an independent snapshot.

| field | shape | meaning |
|---|---|---|
| `position` | `[E, G, 2]` | gate centers (`G = max_gates`) |
| `tangent`, `normal` | `[E, G, 2]` | unit tangent and left-normal at each gate |
| `left`, `right` | `[E, G, 2]` | gate endpoints (`center ± 0.5 * gate_width * normal`) |
| `valid` | `[E]` bool | per-sequence validity |
| `count` | `[E]` int | real gates per env; slots `i >= count[e]` are NaN padding |

`gate_width` is the full gate opening: the `left`/`right` endpoints sit at `±0.5 * gate_width`
along the gate normal, so the default `gate_width=0.0` collapses them onto the center (point
gates), and a positive `gate_width` additionally invalidates any sequence whose gate bars
cross. `max_gates` sizes the fixed output buffers and must be at least the chosen generator's
reachable gate count; `min_gates` rejects a configuration the generator cannot satisfy. Both
bounds are checked when the `GateGenerator` is constructed.

```

- [ ] **Step 2: Insert the collision-solve image after the gate-radius paragraph**

In `README.md`, find the line that ends the gate-radius paragraph:

```
ordered and bbox-normalized, not raw sampler-space coordinates).
```

Insert the following block immediately AFTER that line (before the existing
`![TrackGen standard generator grid]` image):

```markdown

![Phase-2 gate collision solve](docs/assets/readme-gate-strip.png)

*Phase-2 gate collision solve separates overlapping gate spheres to the `2 * gate_radius`
center-spacing target. Top: raw anchors (`gate_solve_iters=0`); bottom: after the solve, with
gate tangents and `gate_width` bars. Rendered by `.venv/bin/python -m viz.render_readme_assets`.*
```

- [ ] **Step 3: Verify the edits**

Run: `grep -n "readme-gate-strip\|GateSequence\` instance\|gate_width\` is the full\|center ± 0.5" README.md`
Expected: matches for the image embed, the aliasing sentence, the `gate_width` prose, and the table endpoint row.

Run: `.venv/bin/python -c "import pathlib; assert pathlib.Path('docs/assets/readme-gate-strip.png').exists()"`
Expected: no error (the embedded image exists).

- [ ] **Step 4: Proofread against ground truth**

Re-read the inserted block. Confirm shapes (`[E, G, 2]`, `[E]`), `gate_width` semantics (full opening, `±0.5*gate_width`), and the `min_gates`/`max_gates` construction-time check all match the spec's ground-truth facts.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add GateSequence result table, gate_width/min_gates notes, collision image"
```

---

### Task 3: README Parameter-explorer section — Gates tab block

**Files:**
- Modify: `README.md` (Parameter explorer section, currently lines 319–344)

**Interfaces:**
- Consumes: nothing. Independent of Tasks 1–2.
- Produces: nothing code-facing (docs).

- [ ] **Step 1: Append the Gates-tab block**

In `README.md`, find the final bullet of the explorer "Using it:" list:

```
  high `relax_iters`) untick it and use **Generate**. **Reroll** draws fresh seeds.
```

Insert the following block immediately AFTER that line:

```markdown

**Gates tab:** a second tab generates gate sequences instead of tracks (see *Gate sequence
generation* above). Controls are grouped —
- **Gate generator** — method selector and `ordering` (choices follow the selected generator).
- **Gate layout** — `gate_width` (full gate opening) and `scale` (pre-collision layout size).
- **Gate collisions** — `gate_radius`, `gate_solve_iters`, and **show raw anchors** (forces
  `gate_solve_iters=0` to inspect anchors before the collision solve). Center spacing target is
  `2 * gate_radius`.
- **Generator-specific sampling** — point-family (Bezier/Hull), Polar, Voronoi, or Checkpoint
  controls, shown for the selected generator.
- **Batch** — grid (n×n), seed, batch size; the stat line reports valid-yield and gate counts
  over the whole batch, and **◀ prev / next ▶** page through the batch without regenerating.
```

- [ ] **Step 2: Verify the edit**

Run: `grep -n "Gates tab:\|show raw anchors\|Pre-collision\|pre-collision layout" README.md`
Expected: matches for the Gates-tab heading and the show-raw-anchors / scale bullets.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document the explorer Gates tab"
```

---

## Self-Review

**Spec coverage:**
- Spec item 1 (gate renderer / collision imagery) → Task 1. ✔
- Spec item 2 (`GateSequence` table + aliasing) → Task 2 Step 1. ✔
- Spec item 3 (`gate_width` + `min_gates`/`max_gates` prose) → Task 2 Step 1. ✔
- Spec item 4 (Gates explorer tab) → Task 3. ✔
- Spec item 5 (embed image + caption) → Task 2 Step 2. ✔
- Spec "Testing/verification" (run renderer, check overlap vs separated, proofread) → Task 1 Steps 6–8, Task 2 Steps 3–4. ✔

**Placeholder scan:** No TBD/TODO; all code blocks and markdown blocks are complete and literal. ✔

**Type/name consistency:** `render_gate_assets`, `_gate_batch`, `_choose_gate_env`, `_set_gate_limits`, `_draw_gate_asset`, `GATE_ASSET_GATE_RADIUS`, and `docs/assets/readme-gate-strip.png` are used identically across Task 1 (definition), Task 1 test (`render_gate_assets`), and Task 2 (image path). `render_readme_assets()` return list extended consistently. ✔
