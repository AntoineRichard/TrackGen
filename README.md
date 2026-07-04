# TrackGen

**GPU-batched race tracks, gate courses, and the runtime signals to race
them — pure [NVIDIA Warp](https://github.com/NVIDIA/warp), CUDA-graph
native.**

[![Documentation](https://img.shields.io/badge/docs-antoinerichard.github.io%2FTrackGen-blue)](https://antoinerichard.github.io/TrackGen/)
[![docs build](https://github.com/AntoineRichard/TrackGen/actions/workflows/docs.yml/badge.svg)](https://github.com/AntoineRichard/TrackGen/actions/workflows/docs.yml)
![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue)
![Warp](https://img.shields.io/badge/warp--lang-%E2%89%A5%201.14-76b900)

Given a batch of per-environment seeds, TrackGen generates `E` closed-loop
tracks (or gate sequences) in parallel — and then keeps working at sim
time: out-of-bounds collision, checkpoint progress and rewards, prop
instancing, all as Warp kernels over the same batched buffers.

```
seeds[E] ─► generator ─► resample ─► XPBD relax ─► inflate ─► Track ─► collision · progress · props
            bezier/checkpoint/hull/polar/voronoi                       (runtime utilities)
```

![Representative phase-1 outputs by generator](docs/assets/readme-generator-strip.png)

## What's in the box

| | | docs |
|---|---|---|
| **Five track generators** | Bezier, checkpoint-steering, hull, polar, Voronoi — one config, per-env styles | [Generators](https://antoinerichard.github.io/TrackGen/generators/overview.html) |
| **Relaxation & inflation** | XPBD-relaxed centerlines inflated into constant-width road bands — index-aligned outer/center/inner borders with tangent/normal frames | [How it works](https://antoinerichard.github.io/TrackGen/how-it-works/inflation.html) |
| **Gate sequences** | Batched gate courses with tangent frames and collision-solved spacing | [Tutorial](https://antoinerichard.github.io/TrackGen/tutorials/gate-sequences.html) |
| **Out-of-bounds collision** | Oriented boxes vs the drivable band — exact scan or baked SDF — plus disc obstacles (gate posts, cones) | [Collision](https://antoinerichard.github.io/TrackGen/utilities/collision.html) |
| **Boundary props** | Cone lines and wall pieces along any boundary, seam-free, for instanced rendering | [Props](https://antoinerichard.github.io/TrackGen/utilities/props.html) |
| **Checkpoints & progress** | Ordered goals from gates *or* subsampled centerlines; pass/lap events and `dist_to_next` rewards | [Progress](https://antoinerichard.github.io/TrackGen/utilities/progress.html) |
| **The Course facade** | One object: `generate()` → `bind()` → `step()`/`reset(mask)`, CUDA-graph replays included | [Course](https://antoinerichard.github.io/TrackGen/utilities/course.html) |

![Runtime utilities on one generated track](docs/assets/utilities-overview.png)

## Install

Python ≥ 3.10; `numpy` and `warp-lang` are the only runtime deps. Runs on
the Warp `cpu` device (tests/CI) and `cuda` (production).

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"     # or:
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Extras: `dev` (tests, torch oracles, matplotlib) · `ui` (Gradio parameter
explorer) · `docs` (Sphinx). Details:
[installation](https://antoinerichard.github.io/TrackGen/getting-started/installation.html).

## Quickstart

Generate a batch of tracks:

```python
import warp as wp
wp.init()
from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

E, device = 64, "cuda"                       # or "cpu"
config = TrackGenConfig(num_envs=E, half_width=0.03, device=device)
rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
track = TrackGenerator(config, rng).generate()
# track.outer / center / inner: [E*N_max] vec2f, NaN-padded, count[e] real points
```

…or let one object run the whole loop — generation, out-of-bounds checks,
checkpoint progress:

```python
from track_gen.course import Course, CourseConfig

course = Course(CourseConfig(mode="track", gen=config, seeds=42,
                             collision="segments", checkpoint_spacing=0.6))
course.bind(position=robot_pos, yaw=robot_yaw, half_extents=robot_he)
course.generate()                            # whole batch + coherent refresh
res = course.step()                          # events + contacts, every sim step
course.reset(done_mask)                      # respawn finished envs
```

More: [batch generation](https://antoinerichard.github.io/TrackGen/tutorials/batch-of-tracks.html) ·
[runtime utilities](https://antoinerichard.github.io/TrackGen/utilities/overview.html) ·
[CUDA graphs in a sim](https://antoinerichard.github.io/TrackGen/tutorials/cuda-graph-in-a-sim.html)

## Documentation

| | |
|---|---|
| [Getting started](https://antoinerichard.github.io/TrackGen/getting-started/quickstart.html) | install, first batch, parameter explorer |
| [Tutorials](https://antoinerichard.github.io/TrackGen/tutorials/batch-of-tracks.html) | batches, gates, generator choice, CUDA graphs |
| [Generators](https://antoinerichard.github.io/TrackGen/generators/overview.html) | the five families, quality benchmarks |
| [Runtime utilities](https://antoinerichard.github.io/TrackGen/utilities/overview.html) | collision, props, checkpoints & progress, Course |
| [How it works](https://antoinerichard.github.io/TrackGen/how-it-works/resample.html) | resample → XPBD → inflation internals |
| [Configuration](https://antoinerichard.github.io/TrackGen/configuration/reference.html) | every knob, tuning guidance |
| [API reference](https://antoinerichard.github.io/TrackGen/reference/api.html) | the complete public surface |

## Development

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -m "not slow and not benchmark and not cuda"  # fast lane
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q                                              # full suite
python -m sphinx -W -b html docs docs/_build/html                                                 # docs
```

Contributing guides (dev setup, writing a generator, rendering the
committed figures) live in the
[docs](https://antoinerichard.github.io/TrackGen/contributing/dev-setup.html).
