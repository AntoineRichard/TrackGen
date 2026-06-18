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
seeds[E] ─► generate (static regen) ─► arc-length resample ─► XPBD relax ─► inflate ─► Track
            corners→sort→bezier→gates                          thickness≥w            outer/center/inner
```

## Install

Python ≥ 3.10. The pipeline requires `warp-lang`; the torch geometry/inflation/relaxation
modules are warp-free (they serve as the test oracle and import without Warp).

```bash
python -m venv .venv
.venv/bin/pip install -e ".[warp,dev]"
```

Core deps: `torch`, `scipy`, `numpy`. Extras: `warp` → `warp-lang`; `dev` → `pytest`, `matplotlib`.

## Quickstart

```python
import torch
import warp as wp; wp.init()   # PerEnvSeededRNG is Warp-backed; initialize Warp once up front
from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

E, device = 64, "cuda"  # or "cpu"
config = TrackGenConfig(num_envs=E, num_points=256, half_width=0.03, device=device)

# The rng's per-env seed values seed the pipeline's built-in Warp RNG (one base seed/env).
seeds = torch.arange(E, dtype=torch.int32, device=device)
rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device=device)

track = TrackGenerator(config, rng).generate(E)

track.center   # [E, 256, 2] arc-length-uniform centerline
track.outer    # [E, 256, 2] outer border   (constant half_width offset)
track.inner    # [E, 256, 2] inner border
track.valid    # [E] bool — True where the track relaxed to a valid constant-width band
```

Only the **Bézier** generator is supported (`config.generator="bezier"`, the default);
the legacy Fourier generator is not part of the Warp pipeline.

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
| `count` | `[E]` int | real point count (`== N` in the default fixed mode) |

## Direct pure-Warp entry points

The facade above wraps `track_gen.warp_pipeline`. You can call it directly:

```python
from track_gen import warp_pipeline as wpl

# Eager: one Track from per-env seeds.
track = wpl.generate_tracks_warp(config, seeds)

# Captured: the WHOLE pipeline as one CUDA graph, replayed with new seeds (CUDA only).
captured = wpl.generate_tracks_warp_graph(config, seeds_template)
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
env index `= tid // N`). Generation uses Warp's built-in RNG with a fixed-iteration,
masked "accept-first-valid" regen loop so the whole thing is static and graph-capturable.
The existing torch implementation (`geometry`/`inflation`/`generators`/`relaxation`) is
retained as the **verification oracle**: every Warp kernel has a test asserting it matches
its torch counterpart on both `cpu` and `cuda`.

## Project layout

```
track_gen/
  warp_pipeline.py    # the pure-Warp pipeline: all kernels + generate_tracks_warp(_graph)
  warp_relax.py       # fused-Warp XPBD relaxation solve (cpu+cuda)
  track_generator.py  # TrackGenerator facade -> generate_tracks_warp
  types.py            # TrackGenConfig, Track (dependency-free leaf dataclasses)
  geometry.py         # torch geometry primitives  ┐
  inflation.py        # torch inflate stages         ├ test oracles (warp-free); not on the
  generators.py       # torch Bézier/Fourier gen     │ runtime path
  relaxation.py       # torch relax backends        ┘
  rng_utils.py        # PerEnvSeededRNG (Warp RNG state); seeds the pipeline
tests/                # per-kernel oracle tests (cpu+cuda) + end-to-end + graph tests
benchmarks/           # benchmark_pipeline.py (end-to-end), benchmark_relaxation.py (backends)
viz/                  # plotting helpers
docs/                 # ARCHITECTURE.md + superpowers/ design/plan/handoff docs
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
Every new kernel ships with a test asserting equivalence to its torch oracle on `cpu` and
`cuda`.

## License

BSD-3-Clause. Copyright (c) 2022-2025, The Isaac Lab Project Developers.
