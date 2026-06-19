# track_gen

GPU-batched generation of closed-loop race tracks. Given a batch of per-environment
seeds, `track_gen` produces, in parallel, `E` smooth closed centerlines plus their
constant-width outer/inner borders and per-point frames — ready to drop into a batched
RL simulator.

The whole pipeline — **generation → resample → relaxation → inflation** — is expressed
as [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels. It runs on the Warp **`cpu`**
device (GPU-free, for tests/CI) and on **`cuda`** (production), with PyTorch acting only
as the array container at the boundary. The entire pipeline can be captured as a single
replayable **CUDA graph**.

```
seeds[E] ─► generate (single pass) ─► constant-spacing resample ─► XPBD relax ─► inflate ─► Track
            corners→prune-sort→bezier→de-cross                       thickness≥w              outer/center/inner
```

## Install

Python ≥ 3.10. The pipeline requires `warp-lang`; the torch geometry/inflation/relaxation
modules are warp-free (they serve as the test oracle and import without Warp).

### From scratch with [uv](https://docs.astral.sh/uv/) (recommended)

```bash
# 1. install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create the project venv — uv fetches Python 3.12 if it isn't present
uv venv --python 3.12

# 3. install track_gen (editable) with the warp + dev extras
uv pip install -e ".[warp,dev]"

# 4. verify
.venv/bin/python -m pytest -q
```

### With venv + pip

```bash
python -m venv .venv
.venv/bin/pip install -e ".[warp,dev]"
```

Both create a `.venv/`; run anything in it with `.venv/bin/python …` (or `source .venv/bin/activate`,
or `uv run …`). Core deps: `torch`, `scipy`, `numpy`. Extras: `warp` → `warp-lang`; `dev` → `pytest`, `matplotlib`.

## Quickstart

```python
import torch
import warp as wp; wp.init()   # PerEnvSeededRNG is Warp-backed; initialize Warp once up front
from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

E, device = 64, "cuda"  # or "cpu"
config = TrackGenConfig(num_envs=E, half_width=0.03, device=device)
# output_mode is "constant_spacing" (the only mode). spacing auto-couples to 0.6*half_width,
# so each track gets its own arc-uniform point count (≤ N_max, default 256), NaN-padded past it.

# The rng's per-env seed values seed the pipeline's built-in Warp RNG (one base seed/env).
seeds = torch.arange(E, dtype=torch.int32, device=device)
rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device=device)

track = TrackGenerator(config, rng).generate(E)

track.center   # [E, N_max, 2] centerline, arc-uniform then NaN-padded past track.count[e]
track.outer    # [E, N_max, 2] outer border   (constant half_width offset)
track.inner    # [E, N_max, 2] inner border
track.valid    # [E] bool — True where the track relaxed to a valid constant-width band
track.count    # [E] int  — real points per track (the rest of each row is NaN padding)
```

Only the **Bézier** generator is on the Warp path (`config.generator="bezier"`, the default).
The Fourier generator lives in `track_gen._experimental` and is **unsupported** — it is not
on the Warp pipeline and receives no compatibility guarantees.

### Output (constant spacing)

There is one output mode, `constant_spacing` — the only value `config.output_mode` accepts
(`__post_init__` raises otherwise). Each track is emitted at a constant arc *spacing* rather
than a constant point *count*: a per-track `count[e] = floor(perimeter/spacing)+1`, capped at
`N_max` and NaN-padded past it. `spacing` defaults to `None` → auto `0.6*half_width` (the
relax-friendly value; set it explicitly to override). The legacy `fixed` mode (constant
`num_points`) was **dropped**: a fixed count over-resolves short tracks, so the Jacobi XPBD
solve under-converges and the road self-overlaps; relaxing at a width-appropriate spacing
converges to smooth, valid tracks on fewer nodes. `num_points` now only sets the intermediate
dense-resample resolution. Size `N_max ≥ max(perimeter)/spacing + 1` so no track is truncated.

### The `Track` result

All boundary arrays are index-aligned (`outer[i]`, `center[i]`, `inner[i]` share one
cross-section normal). Half-width is recovered as `‖outer − center‖`.

| field | shape | meaning |
|---|---|---|
| `outer`, `center`, `inner` | `[E, N, 2]` | border / centerline / border points |
| `tangent`, `normal` | `[E, N, 2]` | unit tangent and left-normal along the centerline |
| `arclen` | `[E, N]` | cumulative arc length (0 at index 0) |
| `length` | `[E]` | closed-loop perimeter |
| `valid` | `[E]` bool | per-track validity |
| `count` | `[E]` int | real points per track (`floor(perimeter/spacing)+1`, capped at `N_max`); the rest of each `[N, 2]` row is NaN padding |

## Direct pure-Warp entry points

The facade above wraps the Warp pipeline. You can also call it directly via the public top-level imports:

```python
from track_gen import generate_tracks_warp, generate_tracks_warp_graph

# Eager: one Track from per-env seeds.
track = generate_tracks_warp(config, seeds)

# Captured: the WHOLE pipeline as one CUDA graph, replayed with new seeds (CUDA only).
captured = generate_tracks_warp_graph(config, seeds_template)
track = captured.replay(new_seeds)   # re-runs every stage on the GPU off the seed buffer
```

The CUDA graph is the deployable, GPU-resident path. At large batches the pipeline is
compute-bound (the relaxation dominates), so graph replay is ~the same wall-clock as the
eager call — capture's value is a single replayable graph, not a speedup.

## Architecture

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the full walkthrough: each stage
and its Warp kernels, the kernel conventions, the torch-as-test-oracle approach, and how
the end-to-end CUDA graph is captured.

In short: every stage is a Warp kernel over flat `[E*N]` arrays (one thread per element,
env index `= tid // N`). Generation uses Warp's built-in RNG in a single static pass (one
corner draw per env, no regen loop and no gate; any self-crossing track falls back to its
corner polygon, which XPBD re-rounds) so the whole thing is graph-capturable.
The torch reference implementation (`geometry`/`inflation`/`generators`/`relaxation`) lives
under `tests/_oracle/` and is **not** part of the shipped package — it serves purely as the
**verification oracle**: every Warp kernel has a test asserting it matches its torch
counterpart on both `cpu` and `cuda`.

## Project layout

```
track_gen/
  __init__.py        # curated public API (TrackGenerator, generate_tracks_warp[_graph],
                     #   TrackGenConfig, Track, PerEnvSeededRNG, __version__)
  _version.py
  _src/              # the Warp pipeline (private core)
    warp_pipeline.py warp_relax.py track_generator.py types.py rng_utils.py rng_kernels.py
  _experimental/     # Fourier generator (unsupported, not on the Warp path)
tests/
  _oracle/           # torch reference impl used to validate the Warp kernels
  test_*.py
benchmarks/  viz/  docs/
```

## Development

```bash
# Full test suite (most tests run on the Warp cpu device, so no GPU is required;
# cuda-only assertions are guarded by torch.cuda.is_available()).
.venv/bin/python -m pytest -q

# End-to-end benchmark (auto device, E=8192). --graph also captures + times the CUDA graph.
.venv/bin/python -m benchmarks.benchmark_pipeline --graph
.venv/bin/python -m benchmarks.benchmark_pipeline --E 2048 --cpu
```

**Conventions** (see `docs/ARCHITECTURE.md`): one thread per output element; flat `[E*N]`
`wp.vec2f` arrays; env index `e = tid // N`; launch with `device=str(tensor.device)`.
Post-generation stages are count-aware: they operate over flat `[E, N_max, 2]` buffers with
a per-track `count[e]` (the fixed-`N` parity path the oracle tests use is `count == N_max`).
Every new kernel
ships with a test asserting equivalence to its torch oracle on `cpu` and `cuda`.

## Parameter explorer (UI)

An interactive Gradio app to see how each parameter affects generation — sliders for the
regime / shape / resolution / relaxation knobs, a live track grid, and the valid-yield stat.

```bash
.venv/bin/pip install -e ".[ui]"     # adds gradio
.venv/bin/python -m viz.param_explorer   # opens a local URL (default http://127.0.0.1:7860)
```

**Using it:**
- Controls are grouped — **Regime** (width / box), **Shape** (corner count / `rad` / `edgy` /
  `handle_clamp_frac`), **Resolution** (`spacing` / `N_max`), **Relaxation**, **Batch**.
- Output is always **`constant_spacing`** (the only mode): **`spacing`** sets the arc step
  (≈ `0.6*half_width`) and **`N_max`** the per-track point cap. **`handle_clamp_frac`** trades
  Bézier-handle overshoot (the main self-crossing source) against corner roundness.
- **Batch size** generates that many tracks (256–8192); the **valid-yield % + mean length /
  thickness / count** shown above the grid are computed over the *whole batch* for honest stats.
- The grid shows one **page** of `grid_n × grid_n` tracks — **◀ prev / next ▶** page through the
  batch *without* regenerating. Invalid tracks get a red title.
- **Auto-update** (on) re-generates as you change a control; for heavy settings (large batch ×
  high `relax_iters`) untick it and use **Generate**. **Reroll** draws fresh seeds.

## License

BSD-3-Clause. Copyright (c) 2022-2025, The Isaac Lab Project Developers.
