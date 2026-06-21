# Pluggable First-Stage Generator Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make first-stage centerline generation a pluggable, user-selectable catalog (registry + static dispatch) and ship an offline harness that characterizes each generator's quality/diversity/speed/racing tradeoffs — without adding any new generator method.

**Architecture:** A `GeneratorSpec` (name, `alloc_scratch`, `generate`) registered in a `GENERATORS` dict; `TrackGenerator` resolves `config.generator` once at construction and the orchestrator calls the resolved spec. Only the chosen generator's kernels enter the captured CUDA graph. The generation OUTPUT buffers (`gen_centerline`, `gen_valid`) move to orchestrator ownership so each generator owns only private working scratch. The comparison harness is a dev-only tool in `benchmarks/` that reads the pre-relax `cs_center` and post-relax `Track.center` from one pipeline run and computes metrics host-side in numpy.

**Tech Stack:** Python 3.10+, NVIDIA Warp (`warp-lang`), numpy (runtime); torch + matplotlib (dev-only, harness). `.venv/bin/python` is the interpreter.

## Global Constraints

- The runtime package `track_gen/_src/**` stays **Warp-native and torch-free**. No `import torch`, no `torch.*`. (The single gated `count.numpy()` truncation warning in `resample_constant_spacing` is the only host readback and keeps its `not _CAPTURING` gate.) Verify with `grep -rnE "\btorch\b" track_gen/_src/*.py` → only doc-comment mentions of `wp.to_torch`.
- **Zero per-call allocation** on the `generate()` path. All buffers are pre-allocated in `TrackGenerator.__init__` (via `_inflate_warp_alloc`). The CUDA-graph capture region must stay allocation-free — the cuda graph parity test (`tests/test_warp_graph.py`) FAILS if an allocation slips into capture; it is the tripwire.
- **`generate()` / `_run()` / `_run_pipeline` stay graph-capturable**: no host syncs, no Python branching on device data inside them. Generator registry resolution is a plain Python dict lookup done in `_run_pipeline`/`__init__` (outside kernels) — that is fine.
- **The harness is dev-only**: `benchmarks/compare_generators.py` and `benchmarks/track_metrics.py` are NEVER imported by `track_gen/`. They may use torch/numpy/matplotlib.
- **Public API frozen**: `track_gen.__all__` stays exactly `["TrackGenerator", "TrackGenConfig", "Track", "PerEnvSeededRNG", "__version__"]`. `config.generator` (a `TrackGenConfig` str field) is the user selector.
- **Full suite stays green**: `.venv/bin/python -m pytest -q` must pass on this machine (has `cuda:0`) so the cuda graph test runs. Baseline before this plan: **236 passed**. Tasks add tests; note the new count each task.
- GPG signing fails (no TTY): every commit uses `--no-gpg-sign`.
- Run everything from repo root `/home/antoiner/Documents/TrackGen`.

---

## File Structure

- **Create** `track_gen/_src/generator_registry.py` — `GeneratorSpec` dataclass + `GENERATORS` dict + `register`/`get`/`available`/`_ensure_loaded`. Imports nothing from the package at module top (leaf); lazily imports generator modules inside `_ensure_loaded`.
- **Modify** `track_gen/_src/warp_generate.py` — add `bezier_alloc_scratch(config)` (the bezier private-scratch allocator, extracted from `_inflate_warp_alloc`); register the `"bezier"` spec at module load.
- **Modify** `track_gen/_src/warp_pipeline.py` — `GenScratch` loses `gen_centerline`/`gen_valid` (now orchestrator-owned); `_Scratch` gains them as bridge buffers; `_inflate_warp_alloc` delegates gen-scratch to the spec and owns the two output buffers; `_run_pipeline` resolves the generator via the registry.
- **Modify** `track_gen/_src/track_generator.py` — replace the `generator == "bezier"` assert with registry membership validation.
- **Create** `docs/generator-contract.md` — the durable generator contract (the brief for worktree agents).
- **Create** `benchmarks/track_metrics.py` — pure-numpy metric functions (unit-testable).
- **Create** `benchmarks/compare_generators.py` — the comparison harness (dev tool).
- **Create** `tests/test_generator_registry.py`, `tests/test_track_metrics.py`, `tests/test_compare_generators.py`.
- **Create** `docs/generator-baseline.md` — committed bezier baseline metrics table.
- **Modify** `README.md` and `ARCHITECTURE.md` (if present) — document `config.generator` + available generators.

---

## Task 1: Generator registry + dispatch seam (no behavior change)

Introduce the registry and route the existing bezier generator through it, moving the two generation OUTPUT buffers to orchestrator ownership. Bezier behavior must be byte-for-byte unchanged — the full suite is the proof.

**Files:**
- Create: `track_gen/_src/generator_registry.py`
- Create: `tests/test_generator_registry.py`
- Modify: `track_gen/_src/warp_generate.py` (add `bezier_alloc_scratch`, register spec)
- Modify: `track_gen/_src/warp_pipeline.py` (`GenScratch`, `_Scratch`, `_inflate_warp_alloc`, `_run_pipeline`)
- Modify: `track_gen/_src/track_generator.py` (`__init__` validation)

**Interfaces:**
- Produces:
  - `generator_registry.GeneratorSpec(name: str, alloc_scratch: Callable, generate: Callable)`
  - `generator_registry.register(spec) -> None`, `generator_registry.get(name) -> GeneratorSpec` (raises `ValueError` listing `available()` on unknown name), `generator_registry.available() -> list[str]`
  - `warp_generate.bezier_alloc_scratch(config) -> GenScratch` (bezier private scratch)
  - generator `generate(seeds_wp, config, out_centerline, out_valid_wp, scratch) -> None` (already bezier's signature)
- Consumes: existing `_inflate_warp_alloc(config) -> (Track, _Scratch)`, `_run_pipeline(config, seed_buf_wp, out, scratch)`.

- [ ] **Step 1: Write the failing registry test**

Create `tests/test_generator_registry.py`:

```python
from track_gen._src import generator_registry as reg


def test_bezier_is_registered():
    assert "bezier" in reg.available()
    spec = reg.get("bezier")
    assert spec.name == "bezier"
    assert callable(spec.alloc_scratch) and callable(spec.generate)


def test_unknown_generator_raises_with_available_list():
    import pytest
    with pytest.raises(ValueError) as e:
        reg.get("does-not-exist")
    assert "bezier" in str(e.value)  # error lists what IS available
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generator_registry.py -q`
Expected: FAIL (`ModuleNotFoundError: ...generator_registry`).

- [ ] **Step 3: Create the registry module**

Create `track_gen/_src/generator_registry.py`:

```python
"""Registry of first-stage centerline generators.

A generator is a (name, alloc_scratch, generate) triple. ``config.generator`` selects
one by name. ``TrackGenerator`` resolves it once at construction; the orchestrator calls
the resolved ``generate``. See docs/generator-contract.md for the contract every generator
implements.

This module imports nothing from the package at load time (leaf). Generator modules are
imported lazily in ``_ensure_loaded`` so each self-registers exactly once, with no import
cycle (warp_generate imports warp_pipeline, not this module's body).
"""
from __future__ import annotations

import dataclasses
from typing import Callable


@dataclasses.dataclass(frozen=True)
class GeneratorSpec:
    """One registered generator.

    name:          the ``config.generator`` string that selects it.
    alloc_scratch: ``(config) -> scratch`` — allocate this generator's PRIVATE working
                   buffers ONCE (fixed shapes from config, on config.device). The
                   generation OUTPUT buffers (centerline, valid) are owned by the
                   orchestrator and passed to ``generate``; they are NOT part of this.
    generate:      ``(seeds_wp, config, out_centerline, out_valid_wp, scratch) -> None`` —
                   write the closed centerline ([E*num_points] vec2f) into out_centerline
                   and per-env validity ([E] int32) into out_valid_wp, using scratch.
                   Pure Warp, in-place, graph-capturable, zero-alloc, no host sync.
    """
    name: str
    alloc_scratch: Callable
    generate: Callable


GENERATORS: dict[str, GeneratorSpec] = {}
_LOADED = False


def register(spec: GeneratorSpec) -> None:
    GENERATORS[spec.name] = spec


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    # Importing each generator module runs its module-level register(...) call.
    # Add one import line per new generator (the only shared touch-point).
    from . import warp_generate  # noqa: F401  (registers "bezier")
    _LOADED = True


def get(name: str) -> GeneratorSpec:
    _ensure_loaded()
    if name not in GENERATORS:
        raise ValueError(
            f"unknown generator {name!r}; available: {sorted(GENERATORS)}"
        )
    return GENERATORS[name]


def available() -> list[str]:
    _ensure_loaded()
    return sorted(GENERATORS)
```

- [ ] **Step 4: Add `bezier_alloc_scratch` + spec registration to `warp_generate.py`**

In `track_gen/_src/warp_generate.py`, add (after `generate_centerline_warp` is defined, near the end of the module) a private-scratch allocator and the registration. This is the `GenScratch(...)` block currently inline in `_inflate_warp_alloc` (warp_pipeline.py:1227-1246) MINUS `gen_centerline` and `gen_valid` (those move to orchestrator ownership in Step 5):

```python
def bezier_alloc_scratch(config):
    """Allocate the bezier generator's PRIVATE working scratch (one alloc per generator).

    The generation OUTPUT buffers (gen_centerline, gen_valid) are owned by the orchestrator
    and passed to generate_centerline_warp; they are not allocated here.
    """
    _pipe._init()
    E = int(config.num_envs)
    P = int(config.max_num_points)
    npseg = int(config.num_points_per_segment)
    M_dense = P * npseg
    N_gen = int(config.num_points)
    dev = str(config.device)
    return _pipe.GenScratch(
        gen_count=wp.empty(E, dtype=wp.int32, device=dev),
        gen_corners=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        gen_ordered=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        gen_used=wp.empty(E * P, dtype=wp.int32, device=dev),
        gen_keys=wp.empty(E * P, dtype=wp.float32, device=dev),
        gen_tan=wp.empty(E * P, dtype=wp.vec2f, device=dev),
        gen_scale=wp.empty(E * P, dtype=wp.float32, device=dev),
        gen_dense=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        gen_poly=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        gen_rs=wp.empty(E * N_gen, dtype=wp.vec2f, device=dev),
        gen_crossers=wp.empty(E, dtype=wp.int32, device=dev),
        gen_arc_real=wp.empty(E * M_dense, dtype=wp.vec2f, device=dev),
        gen_arc_seg=wp.empty(E * M_dense, dtype=wp.float32, device=dev),
        gen_arc_s=wp.empty(E * (M_dense + 1), dtype=wp.float32, device=dev),
        gen_arc_cr=wp.empty(E, dtype=wp.int32, device=dev),
        gen_arc_co=wp.empty(E, dtype=wp.int32, device=dev),
    )


from . import generator_registry as _registry  # noqa: E402
_registry.register(_registry.GeneratorSpec(
    name="bezier",
    alloc_scratch=bezier_alloc_scratch,
    generate=generate_centerline_warp,
))
```

- [ ] **Step 5: Move the two output buffers + delegate gen-scratch in `warp_pipeline.py`**

Make four edits in `track_gen/_src/warp_pipeline.py`:

(a) In `GenScratch` (the `__slots__` at ~line 1031-1036 and `__init__` params + body at ~1038-1076): **remove `gen_centerline` and `gen_valid`** (both the `__slots__` entries and the `__init__` parameter + `self.x = x` lines). Update the class docstring to note these two are now orchestrator-owned. GenScratch is now bezier's private scratch.

(b) In `_Scratch` (~line 1155): add `gen_centerline` and `gen_valid` to `__slots__` and to `__init__` (params `gen_centerline=None, gen_valid=None` and `self.gen_centerline = gen_centerline`, `self.gen_valid = gen_valid`). Update the class docstring's bridge-buffer list to include them. (Now `scratch.gen_centerline`/`scratch.gen_valid` resolve as direct slots, no `__getattr__` fallthrough.)

(c) In `_inflate_warp_alloc` (~line 1227-1266): replace the inline `gen = GenScratch(...)` block with a registry-delegated allocation, and allocate the two output buffers in the bridge:

```python
    from . import generator_registry
    gen = generator_registry.get(config.generator).alloc_scratch(config)
    N_gen = int(config.num_points)
    gen_centerline = wp.empty(E * N_gen, dtype=wp.vec2f, device=dev)
    gen_valid = wp.empty(E, dtype=wp.int32, device=dev)
```

and pass `gen_centerline=gen_centerline, gen_valid=gen_valid` into the `_Scratch(...)` constructor (alongside the existing `cs_center`/`cs_seg`/`cs_s`/`count`). Keep `gen=gen` as before. (`N_gen`/`P`/`npseg`/`M_dense` local vars that were only used by the removed `GenScratch(...)` block can go; `N_gen` is still needed for `gen_centerline`.)

(d) In `_run_pipeline` (~line 714-762): resolve the generator via the registry and use the orchestrator-owned output buffers:

```python
    from . import generator_registry

    n_max = int(config.N_max)
    gen = scratch.gen        # generator-private scratch
    relax = scratch.relax

    # 1. Generate centerline in-place into the orchestrator-owned output buffers.
    generate = generator_registry.get(config.generator).generate
    generate(seed_buf_wp, config,
             out_centerline=scratch.gen_centerline,
             out_valid_wp=scratch.gen_valid,
             scratch=gen)

    # 2. Constant-spacing resample (gen centerline -> bridge buffers).
    resample_constant_spacing(
        scratch.gen_centerline, float(config.spacing), n_max,
        out_wp=scratch.cs_center, count_wp=scratch.count,
        seg_wp=scratch.cs_seg, s_wp=scratch.cs_s,
    )
```

and change the inflate call's `valid=gen.gen_valid` (~line 760) to `valid=scratch.gen_valid`. Delete the now-unused `from . import warp_generate` line in `_run_pipeline`.

- [ ] **Step 6: Route `track_generator.__init__` through the registry**

In `track_gen/_src/track_generator.py`, replace the generator check (lines 64-68):

```python
        if config.generator != "bezier":
            raise ValueError(
                f"The pure-Warp pipeline supports generator='bezier' only; "
                f"got {config.generator!r}."
            )
```

with:

```python
        from . import generator_registry
        generator_registry.get(config.generator)  # raises ValueError listing available names
```

Update the `__init__`/class docstrings that say "must be `'bezier'`" / "Only the `bezier` generator is supported" to "must be a registered generator (see `generator_registry.available()`)".

- [ ] **Step 7: Run the registry test + verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generator_registry.py -q`
Expected: PASS (2 passed).

- [ ] **Step 8: Run the FULL suite — bezier behavior must be unchanged**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: **236 passed** (no change — the seam is behavior-preserving). The cuda graph test (`tests/test_warp_graph.py`) MUST pass, proving no allocation entered the capture region. Also run the torch-free guard:
`.venv/bin/python -c "import sys, track_gen; assert 'torch' not in sys.modules"` → exit 0.

If the count drops or the graph test fails, a buffer move or the registry resolution broke zero-alloc/capture — revert and fix before continuing.

- [ ] **Step 9: Commit**

```bash
git add track_gen/_src/generator_registry.py track_gen/_src/warp_generate.py \
        track_gen/_src/warp_pipeline.py track_gen/_src/track_generator.py \
        tests/test_generator_registry.py
git commit --no-gpg-sign -m "feat(gen): generator registry + static dispatch seam (bezier baseline)"
```

---

## Task 2: Generator contract doc

The durable interface, written so a fresh agent (e.g. one per worktree) can implement a generator from it alone.

**Files:**
- Create: `docs/generator-contract.md`

- [ ] **Step 1: Write the contract doc**

Create `docs/generator-contract.md` with the following content (this is the brief handed to each future generator author):

```markdown
# First-Stage Generator Contract

A generator produces the initial closed centerline that the pipeline then resamples,
relaxes (XPBD), and inflates. To add one, implement two callables and register a
`GeneratorSpec` (see `track_gen/_src/generator_registry.py`).

## What you implement

- `alloc_scratch(config) -> scratch` — allocate your generator's PRIVATE working buffers
  ONCE. Fixed shapes derived from `config` (e.g. `num_envs`, `max_num_points`,
  `num_points_per_segment`, `num_points`), all on `str(config.device)`. Return any object
  exposing the buffers your `generate` uses. Do NOT allocate the output centerline/valid
  here — the orchestrator owns those.
- `generate(seeds_wp, config, out_centerline, out_valid_wp, scratch) -> None`:
  - `seeds_wp`: `[E]` int32 wp.array, one base seed per env.
  - `out_centerline`: `[E*num_points]` `wp.vec2f` — write a CLOSED centerline of
    `config.num_points` points per env, in place.
  - `out_valid_wp`: `[E]` int32 — write 1 for envs that produced a usable centerline.
  - `scratch`: the object your `alloc_scratch` returned.

## Hard rules

- Pure Warp kernels (`wp.launch`), one env per row. NO torch in `track_gen/_src`.
- Zero dynamic allocation inside `generate` (all buffers come from `alloc_scratch`).
- CUDA-graph capturable: no host sync, no host-side retry loop conditioned on generated
  data, no per-env Python branching inside `generate`.
- Fixed bounds for every loop/buffer (graph capture needs static shapes).
- Deterministic in `(per-env seed, config)`; use the Warp RNG (`track_gen._src.rng_*`).
  cpu vs cuda RNG may differ (as elsewhere).

## What you do NOT have to guarantee

- A simple (non-self-intersecting) loop is preferred but not required — XPBD repairs
  thickness and the polygon fallback handles self-crossings. Output must be finite (no NaN)
  for valid envs.

## How a generator is judged

Run `benchmarks/compare_generators.py` (see docs/generator-baseline.md). Generators are
characterized, never gated: a method that scores worse on yield but better on speed or
style stays selectable.
```

- [ ] **Step 2: Commit**

```bash
git add docs/generator-contract.md
git commit --no-gpg-sign -m "docs: first-stage generator contract"
```

---

## Task 3: Metrics module (pure numpy, TDD)

Unit-testable numpy metric functions operating on per-env real points.

**Files:**
- Create: `benchmarks/track_metrics.py`
- Create: `tests/test_track_metrics.py`

**Interfaces:**
- Produces (all take `pts: np.ndarray [n, 2]` of a single env's REAL points, closed loop):
  - `perimeter(pts) -> float`
  - `polygon_area(pts) -> float`
  - `compactness(pts) -> float`  (`4*pi*area / perimeter**2`, 1.0 for a circle)
  - `turn_angles(pts) -> np.ndarray [n]`  (exterior angle at each vertex, radians)
  - `self_intersects(pts) -> bool`
  - `curvature(pts) -> np.ndarray [n]`  (turn-angle / mean adjacent segment length)
  - `racing_line_proxy(pts, a_lat_max=1.0) -> dict`  (`peak_curvature`, `integral_kappa2`, `lap_time`)

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_track_metrics.py`:

```python
import numpy as np
import pytest
from benchmarks import track_metrics as m


def _circle(n=256, r=2.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _square(s=3.0, n_per_side=50):
    side = np.linspace(0, s, n_per_side, endpoint=False)
    top = np.stack([side, np.full_like(side, s)], 1)
    right = np.stack([np.full_like(side, s), s - side], 1)
    bottom = np.stack([s - side, np.zeros_like(side)], 1)
    left = np.stack([np.zeros_like(side), side], 1)
    return np.concatenate([np.stack([side, np.zeros_like(side)], 1), right, top[::-1], left[::-1]])


def _figure_eight(n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.sin(t), np.sin(t) * np.cos(t)], axis=1)


def test_perimeter_and_area_of_circle():
    c = _circle(n=512, r=2.0)
    assert m.perimeter(c) == pytest.approx(2 * np.pi * 2.0, rel=1e-3)
    assert m.polygon_area(c) == pytest.approx(np.pi * 2.0 ** 2, rel=1e-2)


def test_compactness_circle_near_one_square_less():
    assert m.compactness(_circle(512)) == pytest.approx(1.0, abs=1e-2)
    assert m.compactness(_square()) < 0.9  # square is less compact than a circle


def test_curvature_of_circle_is_constant_inverse_radius():
    r = 2.0
    k = m.curvature(_circle(n=512, r=r))
    assert np.allclose(k, 1.0 / r, rtol=5e-2)


def test_self_intersection_detection():
    assert not m.self_intersects(_circle(64))
    assert m.self_intersects(_figure_eight(200))


def test_racing_line_proxy_keys_and_circle_values():
    out = m.racing_line_proxy(_circle(n=512, r=2.0), a_lat_max=1.0)
    assert set(out) == {"peak_curvature", "integral_kappa2", "lap_time"}
    # circle: constant curvature 1/r -> constant speed v=sqrt(a_lat/k); lap_time = perim/v
    assert out["peak_curvature"] == pytest.approx(0.5, rel=5e-2)
    assert out["lap_time"] > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_track_metrics.py -q`
Expected: FAIL (`ModuleNotFoundError: benchmarks.track_metrics`).

- [ ] **Step 3: Implement the metrics**

Create `benchmarks/track_metrics.py`:

```python
"""Pure-numpy geometric/racing metrics for comparing generators (dev tool — not runtime).

Each function takes one env's REAL points as ``pts`` ([n, 2], a closed loop; the segment
n-1 -> 0 closes it). Batched aggregation lives in compare_generators.py.
"""
from __future__ import annotations

import numpy as np


def _seg_vectors(pts: np.ndarray) -> np.ndarray:
    return np.roll(pts, -1, axis=0) - pts  # pts[i+1] - pts[i], wrapping


def perimeter(pts: np.ndarray) -> float:
    return float(np.linalg.norm(_seg_vectors(pts), axis=1).sum())


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) * 0.5)


def compactness(pts: np.ndarray) -> float:
    p = perimeter(pts)
    if p <= 0:
        return 0.0
    return float(4.0 * np.pi * polygon_area(pts) / (p * p))


def turn_angles(pts: np.ndarray) -> np.ndarray:
    v = _seg_vectors(pts)                       # outgoing edge at each vertex
    v_prev = np.roll(v, 1, axis=0)              # incoming edge
    a_prev = np.arctan2(v_prev[:, 1], v_prev[:, 0])
    a_cur = np.arctan2(v[:, 1], v[:, 0])
    d = a_cur - a_prev
    return (d + np.pi) % (2 * np.pi) - np.pi    # wrap to (-pi, pi]


def curvature(pts: np.ndarray) -> np.ndarray:
    seg_len = np.linalg.norm(_seg_vectors(pts), axis=1)
    mean_adj = 0.5 * (seg_len + np.roll(seg_len, 1))
    mean_adj = np.where(mean_adj > 1e-9, mean_adj, 1e-9)
    return np.abs(turn_angles(pts)) / mean_adj


def self_intersects(pts: np.ndarray) -> bool:
    n = len(pts)
    a = pts
    b = np.roll(pts, -1, axis=0)

    def _ccw(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    for i in range(n):
        for j in range(i + 1, n):
            if j == i or (i + 1) % n == j or (j + 1) % n == i:
                continue  # skip shared-endpoint / adjacent edges
            d1 = _ccw(a[i], b[i], a[j])
            d2 = _ccw(a[i], b[i], b[j])
            d3 = _ccw(a[j], b[j], a[i])
            d4 = _ccw(a[j], b[j], b[i])
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                return True
    return False


def racing_line_proxy(pts: np.ndarray, a_lat_max: float = 1.0) -> dict:
    k = curvature(pts)
    seg_len = np.linalg.norm(_seg_vectors(pts), axis=1)
    k_safe = np.where(k > 1e-6, k, 1e-6)
    v = np.sqrt(a_lat_max / k_safe)             # friction-circle cornering speed
    lap_time = float(np.sum(seg_len / np.where(v > 1e-9, v, 1e-9)))
    return {
        "peak_curvature": float(k.max()),
        "integral_kappa2": float(np.sum(k * k * seg_len)),
        "lap_time": lap_time,
    }
```

- [ ] **Step 4: Run the metric tests + verify pass**

Run: `.venv/bin/python -m pytest tests/test_track_metrics.py -q`
Expected: PASS (5 passed). (If `self_intersects` is too slow for large n in later use, it is only ever called per-env in the harness; fine here.)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/track_metrics.py tests/test_track_metrics.py
git commit --no-gpg-sign -m "feat(bench): pure-numpy track metrics for generator comparison"
```

---

## Task 4: Comparison harness

The dev tool that runs a generator over a fixed seed suite and prints a tradeoff table.

**Files:**
- Create: `benchmarks/compare_generators.py`
- Create: `tests/test_compare_generators.py`

**Interfaces:**
- Consumes: `generator_registry.available()`, `track_metrics.*`, `TrackGenerator`, `TrackGenConfig`, `PerEnvSeededRNG`.
- Produces:
  - `run_generator(name, seed_base, E, base_config) -> dict[str, float]` (one metrics row).
  - `compare(names, seed_base=0, E=4096, base_config=None) -> list[dict]` (+ a markdown printer `format_table(rows) -> str`).

- [ ] **Step 1: Write the failing harness smoke test**

Create `tests/test_compare_generators.py`:

```python
import dataclasses
from benchmarks import compare_generators as cg
from track_gen._src.types import TrackGenConfig

_EXPECTED_KEYS = {
    "generator", "yield", "pre_relax_self_intersection_rate", "xpbd_displacement",
    "mean_length", "mean_compactness", "peak_curvature", "lap_time",
    "gen_ms_per_call",
}


def test_run_generator_bezier_smoke():
    cfg = TrackGenConfig(device="cpu", num_envs=16, half_width=0.1)
    row = cg.run_generator("bezier", seed_base=0, E=16, base_config=cfg)
    assert _EXPECTED_KEYS.issubset(row.keys())
    assert 0.0 <= row["yield"] <= 1.0
    assert row["mean_length"] > 0.0


def test_compare_and_format_table():
    cfg = TrackGenConfig(device="cpu", num_envs=16, half_width=0.1)
    rows = cg.compare(["bezier"], seed_base=0, E=16, base_config=cfg)
    assert len(rows) == 1 and rows[0]["generator"] == "bezier"
    table = cg.format_table(rows)
    assert "bezier" in table and "yield" in table
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_compare_generators.py -q`
Expected: FAIL (`ModuleNotFoundError: benchmarks.compare_generators`).

- [ ] **Step 3: Implement the harness**

Create `benchmarks/compare_generators.py`:

```python
"""Compare first-stage generators on quality / diversity / speed (dev tool — NOT runtime).

Runs each registered generator over a fixed seed suite through the full pipeline, reads the
pre-relax centerline (scratch.cs_center) and post-relax Track in ONE pass, and computes
metrics host-side in numpy. Characterizes; never gates.

    .venv/bin/python -m benchmarks.compare_generators            # all generators, cpu, E=4096
    .venv/bin/python -m benchmarks.compare_generators --cuda --E 8192
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src import generator_registry
from benchmarks import track_metrics as tm


def _real_points(flat_xy: np.ndarray, e: int, n_max: int, count: int) -> np.ndarray:
    """Slice env e's first `count` real points from a flat [E*n_max, 2] numpy array."""
    base = e * n_max
    return flat_xy[base:base + count]


def run_generator(name, seed_base, E, base_config) -> dict:
    cfg = dataclasses.replace(base_config, generator=name, num_envs=E)
    n_max = int(cfg.N_max)
    rng = PerEnvSeededRNG(seeds=int(seed_base), num_envs=E, device=str(cfg.device))
    gen = TrackGenerator(cfg, rng)

    track = gen.generate(E)
    if "cuda" in str(cfg.device):
        wp.synchronize()

    # Read back once. cs_center is the pre-relax constant-spacing centerline (XPBD writes a
    # separate `relaxed` buffer, so cs_center survives the run). Track.center is post-relax.
    pre = wp.to_torch(gen._scratch.cs_center).reshape(-1, 2).cpu().numpy()
    post = wp.to_torch(track.center).reshape(-1, 2).cpu().numpy()
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    count = wp.to_torch(track.count).cpu().numpy().astype(int)

    lengths, compactness, peak_k, lap_times = [], [], [], []
    pre_self_int = 0
    disp_sum, disp_pts = 0.0, 0
    for e in range(E):
        c = int(count[e])
        if c < 4:
            continue
        post_e = _real_points(post, e, n_max, c)
        pre_e = _real_points(pre, e, n_max, c)
        if not np.isfinite(post_e).all():
            continue
        if np.isfinite(pre_e).all() and tm.self_intersects(pre_e):
            pre_self_int += 1
        if valid[e]:
            lengths.append(tm.perimeter(post_e))
            compactness.append(tm.compactness(post_e))
            rl = tm.racing_line_proxy(post_e)
            peak_k.append(rl["peak_curvature"])
            lap_times.append(rl["lap_time"])
        if np.isfinite(pre_e).all():
            disp_sum += float(np.linalg.norm(post_e - pre_e, axis=1).sum())
            disp_pts += c

    # Warm timing of generate() alone.
    for _ in range(2):
        gen.generate(E)
    if "cuda" in str(cfg.device):
        wp.synchronize()
    reps = 5
    t0 = time.time()
    for _ in range(reps):
        gen.generate(E)
    if "cuda" in str(cfg.device):
        wp.synchronize()
    gen_ms = (time.time() - t0) / reps * 1e3

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "generator": name,
        "yield": float(valid.mean()),
        "pre_relax_self_intersection_rate": pre_self_int / E,
        "xpbd_displacement": (disp_sum / disp_pts) if disp_pts else float("nan"),
        "mean_length": _mean(lengths),
        "mean_compactness": _mean(compactness),
        "peak_curvature": _mean(peak_k),
        "lap_time": _mean(lap_times),
        "gen_ms_per_call": gen_ms,
    }


def compare(names=None, seed_base=0, E=4096, base_config=None) -> list:
    if base_config is None:
        base_config = TrackGenConfig(num_envs=E)
    if names is None:
        names = generator_registry.available()
    return [run_generator(n, seed_base, E, base_config) for n in names]


def format_table(rows) -> str:
    cols = ["generator", "yield", "pre_relax_self_intersection_rate", "xpbd_displacement",
            "mean_length", "mean_compactness", "peak_curvature", "lap_time", "gen_ms_per_call"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for r in rows:
        cells = [str(r["generator"])] + [f"{r[c]:.4g}" for c in cols[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--generators", nargs="*", default=None)
    a = ap.parse_args()
    cfg = TrackGenConfig(device="cuda" if a.cuda else "cpu", num_envs=a.E)
    rows = compare(a.generators, seed_base=a.seed, E=a.E, base_config=cfg)
    print(format_table(rows))
```

- [ ] **Step 4: Run the harness smoke test + verify pass**

Run: `.venv/bin/python -m pytest tests/test_compare_generators.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/compare_generators.py tests/test_compare_generators.py
git commit --no-gpg-sign -m "feat(bench): generator comparison harness (quality/diversity/speed)"
```

---

## Task 5: Bezier baseline + user-facing docs

Produce the committed baseline and document `config.generator` for users.

**Files:**
- Create: `docs/generator-baseline.md`
- Modify: `README.md` (and `ARCHITECTURE.md` if it exists)

- [ ] **Step 1: Generate the baseline table**

Run: `.venv/bin/python -m benchmarks.compare_generators --E 4096 --seed 0`
(If `cuda:0` is available, also run `--cuda --E 8192` and include both.)
Copy the printed markdown table.

- [ ] **Step 2: Write the baseline doc**

Create `docs/generator-baseline.md`:

```markdown
# Generator Baseline Metrics

Reference metrics for the registered first-stage generators, produced by
`benchmarks/compare_generators.py`. New methods are reported against this table; it
characterizes tradeoffs (quality / diversity / speed) and never gates which generators
ship — every registered generator stays selectable via `config.generator`.

Suite: seed base 0, E=4096, default `TrackGenConfig` (cpu). Regenerate with
`.venv/bin/python -m benchmarks.compare_generators --E 4096 --seed 0`.

<PASTE THE TABLE FROM STEP 1 HERE>
```

Replace the `<PASTE ...>` line with the actual table from Step 1 (do not leave the
placeholder).

- [ ] **Step 3: Document the user-facing selector**

In `README.md` (and `ARCHITECTURE.md` if present), add a short subsection under the usage/config area:

```markdown
### Choosing a generator

The first-stage centerline generator is selected by `TrackGenConfig(generator=...)`.
Available generators: see `track_gen._src.generator_registry.available()` (currently
`"bezier"`). Adding a method is additive — see `docs/generator-contract.md` and the
tradeoff table in `docs/generator-baseline.md`.
```

- [ ] **Step 4: Verify the full suite is still green**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: green (236 + the new tests from Tasks 1/3/4: registry 2, metrics 5, harness 2 = **245 passed**; confirm the exact number and that the cuda graph test ran).

- [ ] **Step 5: Commit**

```bash
git add docs/generator-baseline.md README.md ARCHITECTURE.md 2>/dev/null
git commit --no-gpg-sign -m "docs: bezier generator baseline + config.generator usage"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** dispatch seam (Task 1) ✓; generator contract (Task 2) ✓; harness with pre-relax `cs_center` capture + speed axis (Task 4) ✓; metrics incl. racing-line proxy (Task 3) ✓; bezier baseline + `config.generator` docs (Task 5) ✓; invariants (Global Constraints + Task 1 Step 8 gate) ✓; "characterizes, never gates" (Tasks 2/5 doc text) ✓; parallel-worktree-friendliness (registry = one import + one line; Task 1 Step 3 comment) ✓. No new generator method (scope boundary) ✓.

**Placeholder scan:** the only intentional fill-in is the baseline TABLE in Task 5 Step 2, which Step 1 produces and Step 2 explicitly says to paste (not a code placeholder). No TBD/TODO in code steps.

**Type consistency:** `GeneratorSpec(name, alloc_scratch, generate)` consistent across registry/warp_generate/warp_pipeline; `generate(seeds_wp, config, out_centerline, out_valid_wp, scratch)` matches bezier's existing signature; harness metric keys in `run_generator` match `_EXPECTED_KEYS` in the test and `format_table` columns; `cs_center`/`Track.center`/`Track.valid`/`Track.count` match the runtime.
