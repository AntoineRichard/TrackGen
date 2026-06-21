# Segment-Grammar Generator (#6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth first-stage generator `"grammar"` that builds a closed centerline from an explicit racing-segment vocabulary (straights, sweepers, hairpins, chicanes, S-bends, kinks, clothoid/spiral transitions) via curvature integration, adding counted straights/hairpins/chicanes the catalog can't otherwise express.

**Architecture:** Per env (fixed-bound, graph-capturable): sample `S` segments → budget + antisymmetry bias → rasterize a per-sample curvature profile `κ[i]` → linear heading closure (set `mean(κ)` so net turn = 2π) → integrate (`θ=cumsum(κ)`, edges = unit tangents) → gap-distribution displacement closure (subtract the mean edge so edges sum to 0) → center + isotropic bbox-rescale. Prototype-first in torch/numpy to tune the budget/antisymmetry defaults against the shape-variety gate + renders, then port to Warp mirroring `warp_generate_polar.py`.

**Tech Stack:** Python 3.10+, NVIDIA Warp (runtime), numpy; torch/matplotlib (dev-only, prototype + harness). `.venv/bin/python`.

## Global Constraints

- The runtime package `track_gen/_src/**` stays **Warp-native and torch-free**. The prototype's torch/numpy lives ONLY in `track_gen/_experimental/` (never imported by the runtime) and `benchmarks/` (dev). Verify: `grep -rn "import torch" track_gen/_src/` stays empty.
- **Zero per-call allocation** on the `generate()` path: `GrammarScratch` is pre-allocated in `grammar_alloc_scratch`. The CUDA-graph capture region must stay allocation-free — `tests/test_warp_graph.py` is the tripwire (it passes today; it must keep passing).
- **CUDA-graph-capturable** generator: pure Warp kernels, one env per row, fixed-bound loops over `S` and `N`, NO host sync, NO host-side closure solve, NO per-env Python branching.
- **Deterministic** in `(per-env seed, config)`; use the Warp RNG with a distinct salt (`_GRAMMAR_SALT`), decorrelated from bezier's count/corner streams and polar's `_CONTROL_SALT = 7919`.
- Registering `"grammar"` is **additive**: one new module + one `GeneratorSpec` + one import line in `generator_registry._ensure_loaded` + the `grammar_*` config fields. `track_gen.__all__` is unchanged.
- **Acceptance is the shape-variety gate, NOT yield:** `tests/test_shape_variety.py` must pass for `"grammar"` (median post-relax compactness < 0.85), AND `benchmarks/compare_generators.py` must show `mean_chicanes` and `straight_frac` for `"grammar"` clearly above bezier/hull/polar, AND a rendered seed grid must show real straights/hairpins/chicanes. A perfect yield with no feature presence is a REJECT (the polar lesson).
- Full suite green on this machine (`cuda:0`). Commits use `--no-gpg-sign`. Run from `/home/antoiner/Documents/TrackGen`.

---

## File Structure

- **Create** `track_gen/_experimental/grammar_proto.py` — torch/numpy host prototype (vocabulary + grammar + integrate + closure + normalize + a tuning/render driver). Dev-only; never imported by the runtime. Carries the *validated, tuned reference* the Warp port mirrors.
- **Modify** `track_gen/_src/types.py` — add the `grammar_*` config fields (defaults from Task 1's tuning) + `__post_init__` validation.
- **Create** `track_gen/_src/warp_generate_grammar.py` — the Warp generator: constants, kernels, `GrammarScratch`, `grammar_alloc_scratch`, `generate_grammar_warp`, `GeneratorSpec` registration. Mirrors `warp_generate_polar.py`.
- **Modify** `track_gen/_src/generator_registry.py` — one import line in `_ensure_loaded`.
- **Create** `tests/test_generate_grammar.py` — e2e + determinism + feature-presence assertions.
- **Modify** `viz/param_explorer.py` — `grammar_*` sliders + `build_config`/`default_params` wiring (the dropdown already lists `"grammar"` via `registry.available()`).

---

## Task 1: torch/numpy prototype + tuning (the validated reference)

Build and tune the algorithm on host before any Warp work. Deliverable: a prototype whose tuned defaults make the shape-variety gate + feature metrics pass and whose rendered grid shows real racing features.

**Files:**
- Create: `track_gen/_experimental/grammar_proto.py`
- Test: `tests/test_grammar_proto.py`

**Interfaces:**
- Produces (consumed conceptually by Task 3's Warp port, which mirrors these):
  - `sample_segments(rng, S, cfg) -> np.ndarray [S, 3]` rows `(kappa_start, kappa_end, length_frac)`.
  - `rasterize_kappa(segments, N) -> np.ndarray [N]`.
  - `close_and_integrate(kappa) -> np.ndarray [N, 2]` (heading close + integrate + gap-distribution displacement close).
  - `normalize(pts, target_extent) -> np.ndarray [N, 2]`.
  - `generate_centerline(seed, cfg) -> np.ndarray [N, 2]` composing the above.
  - module constants `DEFAULTS = dict(grammar_segments=..., grammar_straight_frac=..., grammar_curvature_budget=..., grammar_chicane_bias=..., grammar_hairpin_max_frac=...)` — the tuned values Task 2 copies.

- [ ] **Step 1: Write the failing closure test**

Create `tests/test_grammar_proto.py`:

```python
import numpy as np
from track_gen._experimental import grammar_proto as gp


def test_close_and_integrate_returns_closed_loop():
    # a hand-built curvature profile must integrate to a CLOSED loop (edges sum ~0).
    N = 256
    kappa = np.zeros(N)
    kappa[:64] = 0.05          # one arc span; the rest straight-ish
    pts = gp.close_and_integrate(kappa)
    assert pts.shape == (N, 2)
    edges = np.roll(pts, -1, axis=0) - pts
    assert np.linalg.norm(edges.sum(axis=0)) < 1e-6   # closed: edge vectors sum to zero


def test_generate_centerline_is_deterministic_and_finite():
    a = gp.generate_centerline(7, gp.DEFAULTS)
    b = gp.generate_centerline(7, gp.DEFAULTS)
    assert np.array_equal(a, b)                        # deterministic in seed
    assert np.isfinite(a).all()
    assert a.shape[0] == gp.DEFAULTS["num_points"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_grammar_proto.py -q`
Expected: FAIL (`ModuleNotFoundError: ...grammar_proto`).

- [ ] **Step 3: Implement the prototype**

Create `track_gen/_experimental/grammar_proto.py`. This is the algorithmic reference; tune `DEFAULTS` in Step 5.

```python
"""Host (numpy) prototype of the segment-grammar generator (#6) — DEV ONLY.

Not imported by the runtime. Used to tune the budget/antisymmetry defaults and validate
the closure + character before the Warp port (track_gen/_src/warp_generate_grammar.py).
Two kappa-primitives (constant + linear-ramp) + named patterns; budgeted/antisymmetric
sampling; DC-shift heading closure; gap-distribution displacement closure.
"""
from __future__ import annotations
import numpy as np

# Tuned in Step 5; Task 2 copies these into TrackGenConfig defaults.
DEFAULTS = dict(
    num_points=256,
    grammar_segments=10,            # S (fixed bound)
    grammar_straight_frac=0.35,     # min arc-length fraction forced to straights
    grammar_curvature_budget=2.2,   # cap on summed |signed-turn| beyond the 2*pi winding
    grammar_chicane_bias=1.0,       # strength of opposite-sign pairing (0=off, 1=full pairs)
    grammar_hairpin_max_frac=0.12,  # cap on any single high-kappa feature's length frac
    scale=1.0,
)
_BEZIER_EXTENT = 1.44               # match warp_generate_polar._BEZIER_EXTENT


def sample_segments(rng: np.random.Generator, S: int, cfg: dict) -> np.ndarray:
    """Return [S, 3] rows (kappa_start, kappa_end, length_frac), pre-closure.

    Budget + antisymmetry: features are drawn in opposite-sign PAIRS (chicane bias) so net
    turning is near the 2*pi winding with small residual; a straight quota forces low-kappa
    spans; per-feature magnitude/length are clamped by the curvature budget + hairpin cap.
    """
    segs = []
    # Reserve a straight quota.
    straight_len = cfg["grammar_straight_frac"]
    n_feat = S - max(1, int(round(straight_len * S)))
    # Draw feature magnitudes in +/- pairs (antisymmetry), scaled into the curvature budget.
    pair_count = max(1, n_feat // 2)
    mags = rng.uniform(0.3, 1.0, size=pair_count)
    mags *= cfg["grammar_curvature_budget"] / (mags.sum() + 1e-9)  # budget clamp
    for m in mags:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        # paired opposite-sign features (chicane bias): + then -, mixing const arcs and ramps
        ln = min(cfg["grammar_hairpin_max_frac"], rng.uniform(0.04, cfg["grammar_hairpin_max_frac"]))
        if rng.random() < 0.5:        # constant-kappa arc (sweeper/hairpin/kink by mag,len)
            segs.append((sign * m, sign * m, ln))
            segs.append((-sign * m * cfg["grammar_chicane_bias"], -sign * m * cfg["grammar_chicane_bias"], ln))
        else:                          # linear-ramp (clothoid/spiral)
            segs.append((0.0, sign * m, ln))
            segs.append((-sign * m * cfg["grammar_chicane_bias"], 0.0, ln))
    # Fill the rest with straights (kappa ~ 0).
    while len(segs) < S:
        segs.append((0.0, 0.0, rng.uniform(0.04, 0.12)))
    segs = np.array(segs[:S], dtype=np.float64)
    segs[:, 2] /= segs[:, 2].sum()     # normalise length fractions to 1
    return segs


def rasterize_kappa(segments: np.ndarray, N: int) -> np.ndarray:
    """Assign per-sample curvature from the segment sequence; linear-interp kappa within
    each segment span (constant = equal endpoints, ramp = differing)."""
    bounds = np.concatenate([[0.0], np.cumsum(segments[:, 2])])  # [S+1] in [0,1]
    s = (np.arange(N) + 0.5) / N
    seg_idx = np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, len(segments) - 1)
    kappa = np.empty(N)
    for i in range(N):
        j = seg_idx[i]
        lo, hi = bounds[j], bounds[j + 1]
        u = 0.0 if hi <= lo else (s[i] - lo) / (hi - lo)
        kappa[i] = (1 - u) * segments[j, 0] + u * segments[j, 1]
    return kappa


def close_and_integrate(kappa: np.ndarray) -> np.ndarray:
    """Heading closure (set mean so net turn = 2*pi) + integrate + gap-distribution
    displacement closure (subtract the mean edge so edge vectors sum to zero)."""
    N = kappa.shape[0]
    ds = 1.0 / N
    # Heading closure: theta winds exactly once over s in [0,1).
    kappa = kappa - kappa.mean() + 2.0 * np.pi  # so sum(kappa)*ds == 2*pi
    theta = np.cumsum(kappa) * ds
    theta = theta - theta[0]
    edges = ds * np.stack([np.cos(theta), np.sin(theta)], axis=1)  # [N,2] unit tangents*ds
    edges = edges - edges.mean(axis=0)          # gap-distribution: edges now sum to 0 (closed)
    return np.cumsum(edges, axis=0)


def normalize(pts: np.ndarray, target_extent: float) -> np.ndarray:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = float(np.max(hi - lo))
    s = target_extent / max(extent, 1e-8)
    return (pts - center) * s


def generate_centerline(seed: int, cfg: dict) -> np.ndarray:
    rng = np.random.default_rng(seed)
    segs = sample_segments(rng, int(cfg["grammar_segments"]), cfg)
    kappa = rasterize_kappa(segs, int(cfg["num_points"]))
    pts = close_and_integrate(kappa)
    return normalize(pts, float(cfg["scale"]) * _BEZIER_EXTENT)
```

- [ ] **Step 4: Run the prototype tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_grammar_proto.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Tune `DEFAULTS` against the shape-variety metrics + renders**

Write a short tuning script inline (not committed, or as a `if __name__ == "__main__"` block in `grammar_proto.py`): generate ~500 seeds, compute over them with `benchmarks.track_metrics` — `compactness` percentiles (target median well < 0.85), `chicane_count` (target mean notably > bezier/hull/polar, e.g. ≥ 4), `straight_fraction` (target mean clearly > 0, e.g. ≥ 0.15) — and the closure residual *before* the gap correction (target median residual / extent < ~0.3 so the linear close stays mild). Render a 5×5 grid with matplotlib and eyeball: straights, hairpins, and chicanes must be visibly present; loops must not be collapsed or kinked. Adjust `grammar_curvature_budget`, `grammar_chicane_bias`, `grammar_straight_frac`, `grammar_hairpin_max_frac`, `grammar_segments` until all targets pass, and update the `DEFAULTS` dict to the chosen values.

- [ ] **Step 6: Commit**

```bash
git add track_gen/_experimental/grammar_proto.py tests/test_grammar_proto.py
git commit --no-gpg-sign -m "feat(grammar): host prototype + tuned defaults for the segment-grammar generator (#6)"
```

Record the tuned `DEFAULTS` in the commit message body for Task 2.

---

## Task 2: `grammar_*` config fields

**Files:**
- Modify: `track_gen/_src/types.py` (add fields after `polar_angular_jitter` at line ~85; add validation in `__post_init__` at line ~172)
- Test: `tests/test_types.py` (extend the defaults test)

**Interfaces:**
- Consumes: Task 1's tuned `DEFAULTS`.
- Produces: `TrackGenConfig.grammar_segments: int`, `grammar_straight_frac: float`, `grammar_curvature_budget: float`, `grammar_chicane_bias: float`, `grammar_hairpin_max_frac: float`.

- [ ] **Step 1: Add a failing assertion to `tests/test_types.py`**

In `test_config_defaults_instantiate`, after the polar asserts, add (use Task 1's tuned values):

```python
    # Segment-grammar (#6) params
    assert cfg.grammar_segments == 10
    assert cfg.grammar_straight_frac == 0.35
    assert cfg.grammar_curvature_budget == 2.2
    assert cfg.grammar_chicane_bias == 1.0
    assert cfg.grammar_hairpin_max_frac == 0.12
```

(Match the exact tuned `DEFAULTS` from Task 1; update these numbers if tuning changed them.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_types.py::test_config_defaults_instantiate -q`
Expected: FAIL (`AttributeError: ...grammar_segments`).

- [ ] **Step 3: Add the fields to `TrackGenConfig`**

In `track_gen/_src/types.py` after line 85 (`polar_angular_jitter`):

```python
    # --- Segment-grammar (#6) params ---
    grammar_segments: int = 10           # S: fixed segment count (graph-capture bound)
    grammar_straight_frac: float = 0.35  # min arc-length fraction forced to straights
    grammar_curvature_budget: float = 2.2   # cap on summed |signed-turn| beyond the 2*pi winding
    grammar_chicane_bias: float = 1.0    # opposite-sign pairing strength (antisymmetry)
    grammar_hairpin_max_frac: float = 0.12   # cap on any single high-kappa feature's length frac
```

In `__post_init__` (line ~172) add:

```python
        if int(self.grammar_segments) < 2:
            raise ValueError(f"grammar_segments must be >= 2, got {self.grammar_segments!r}")
        if not (0.0 <= float(self.grammar_straight_frac) < 1.0):
            raise ValueError(f"grammar_straight_frac must be in [0,1), got {self.grammar_straight_frac!r}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_types.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/types.py tests/test_types.py
git commit --no-gpg-sign -m "feat(grammar): grammar_* config fields with tuned defaults"
```

---

## Task 3: Warp port — `warp_generate_grammar.py` + registration

Port the *validated, tuned* Task-1 prototype to Warp, mirroring `track_gen/_src/warp_generate_polar.py` (read it first — same module shape: constants, kernels, a `*Scratch` class, `*_alloc_scratch(config)`, `generate_*_warp(seeds_wp, config, out_centerline, out_valid_wp, scratch)`, and a `_registry.register(_registry.GeneratorSpec(...))` at module load).

**Files:**
- Create: `track_gen/_src/warp_generate_grammar.py`
- Modify: `track_gen/_src/generator_registry.py` (one import line in `_ensure_loaded`)
- Test: covered by Task 4.

**Interfaces:**
- Consumes: the registry (`GeneratorSpec`, `register`), `warp_generate_polar._normalize_centerline_k` pattern (or its own copy), the `grammar_*` config fields.
- Produces: `generate_grammar_warp(seeds_wp, config, out_centerline, out_valid_wp, scratch)`, `grammar_alloc_scratch(config)`, registered name `"grammar"`.

- [ ] **Step 1: Implement `warp_generate_grammar.py`**

Structure (mirror `warp_generate_polar.py`; the per-sample math is the Task-1 prototype, kernelized — one env per row, fixed loops over `S` and `N`):

- Module constants: `_GRAMMAR_SALT = 6271` (distinct large odd, ≠ 7919/6151/9781), `_BEZIER_EXTENT = 1.44`.
- `GrammarScratch` (slots): `segments` `[E*S*3]` float32 (kappa_start, kappa_end, length_frac per segment), `kappa` `[E*N]` float32, `raw` `[E*N]` vec2f. (Outputs `out_centerline`/`out_valid_wp` are orchestrator-owned.)
- `grammar_alloc_scratch(config)`: `_pipe._init()`, read `E=num_envs`, `S=grammar_segments`, `N=num_points`, `dev=str(device)`; `wp.empty` the three buffers; return `GrammarScratch(...)`.
- Kernel `_grammar_sample_k(seeds, S, budget, straight_frac, chicane_bias, hairpin_max_frac, segments)`: one thread per env; `wp.rand_init(seeds[e]*_GRAMMAR_SALT)`; draw the budgeted/antisymmetric segment rows into `segments[e*S : ...]`, replicating `sample_segments` (paired opposite-sign features, straight quota, budget clamp, length-frac normalize). Fixed bounded loops over `S`.
- Kernel `_grammar_build_k(segments, S, N, target_extent, kappa, raw, out_centerline, out_valid)`: one thread per env; rasterize `kappa` from `segments` (prefix-sum bounds + interp, like `rasterize_kappa`); heading closure (subtract mean, add 2π winding); integrate `theta`/edges; gap-distribution displacement closure (subtract mean edge); accumulate bbox; second pass center + isotropic rescale to `target_extent` into `out_centerline`; `out_valid[e]=1`. (You may instead reuse polar's `_normalize_centerline_k` for the center+rescale pass — import it from `warp_generate_polar` or copy the pattern.)
- `generate_grammar_warp(seeds_wp, config, out_centerline, out_valid_wp, scratch)`: `_pipe._init()`; `E=out_valid_wp.shape[0]`; `S/N` from config; `target_extent = float(config.scale) * _BEZIER_EXTENT`; `wp.launch(_grammar_sample_k, dim=E, ...)`; `wp.launch(_grammar_build_k, dim=E, ...)`; `_pipe._sync(dev)`.
- Registration at module bottom:

```python
from . import generator_registry as _registry  # noqa: E402
_registry.register(_registry.GeneratorSpec(
    name="grammar",
    alloc_scratch=grammar_alloc_scratch,
    generate=generate_grammar_warp,
))
```

Constraints: pure Warp, no host sync inside the kernels, no per-env Python branching, fixed loop bounds (`S`, `N`), zero allocation in `generate_grammar_warp`. The numerics should match `grammar_proto` within FP tolerance for a given seed-derived segment set (the RNG streams differ between numpy and Warp, so exact parity is not required — geometric character + closure are what matter).

- [ ] **Step 2: Register the generator**

In `track_gen/_src/generator_registry._ensure_loaded`, after the `warp_generate_hull` line:

```python
    from . import warp_generate_grammar  # noqa: F401  (registers "grammar")
```

- [ ] **Step 3: Verify it loads + registers**

Run: `.venv/bin/python -c "from track_gen._src import generator_registry as r; print(r.available())"`
Expected: `['bezier', 'grammar', 'hull', 'polar']`.

- [ ] **Step 4: Smoke-generate (cpu + cuda) — no allocation in capture**

Run: `.venv/bin/python -c "import warp as wp; from track_gen._src.types import TrackGenConfig; from track_gen._src.track_generator import TrackGenerator; from track_gen._src.rng_utils import PerEnvSeededRNG; wp.init(); cfg=TrackGenConfig(generator='grammar', device='cuda', num_envs=64, half_width=0.1); t=TrackGenerator(cfg, PerEnvSeededRNG(seeds=0,num_envs=64,device='cuda')).generate(64); wp.synchronize(); import torch; print('valid', wp.to_torch(t.valid).float().mean().item())"`
Expected: runs without error; prints a valid fraction. (Confirms capture/replay works — the graph would error on in-capture allocation.)

- [ ] **Step 5: Full suite (tripwire) + torch-free check**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider` → green (incl. `tests/test_warp_graph.py` and `tests/test_shape_variety.py` which now also exercises `"grammar"`). `grep -rn "import torch" track_gen/_src/` → empty.

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/warp_generate_grammar.py track_gen/_src/generator_registry.py
git commit --no-gpg-sign -m "feat(grammar): warp-native segment-grammar generator, registered 'grammar'"
```

---

## Task 4: Acceptance — e2e test + feature-presence

**Files:**
- Create: `tests/test_generate_grammar.py`

**Interfaces:**
- Consumes: `TrackGenerator`, `generator_registry`, `benchmarks.compare_generators`.

- [ ] **Step 1: Write the acceptance test**

Create `tests/test_generate_grammar.py`:

```python
import numpy as np
import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from benchmarks import compare_generators as cg


def _gen(seed, E=64):
    cfg = TrackGenConfig(generator="grammar", device="cpu", num_envs=E, half_width=0.1, relax_iters=40)
    return TrackGenerator(cfg, PerEnvSeededRNG(seeds=seed, num_envs=E, device="cpu")).generate(E)


def test_grammar_e2e_finite_n_points():
    E = 64
    t = _gen(0, E)
    center = wp.to_torch(t.center).cpu().numpy().reshape(E, -1, 2)
    count = wp.to_torch(t.count).cpu().numpy().astype(int)
    valid = wp.to_torch(t.valid).cpu().numpy().astype(bool)
    assert valid.mean() > 0.5
    for e in range(E):
        c = int(count[e])
        assert 1 <= c <= center.shape[1]
        assert np.isfinite(center[e, :c]).all()


def test_grammar_is_deterministic_within_device():
    a = wp.to_torch(_gen(7).center).cpu().numpy()
    b = wp.to_torch(_gen(7).center).cpu().numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_grammar_adds_net_new_features_vs_other_generators():
    # The whole point: grammar makes MORE chicanes + straights than the star-shaped
    # generators (which produce them only as noise). Compare on a fixed suite.
    cfg = TrackGenConfig(device="cpu", num_envs=128, half_width=0.1, relax_iters=40)
    rows = {r["generator"]: r for r in cg.compare(["bezier", "polar", "hull", "grammar"],
                                                  seed_base=0, E=128, base_config=cfg)}
    g = rows["grammar"]
    others_chicanes = max(rows[k]["mean_chicanes"] for k in ("bezier", "polar", "hull"))
    others_straight = max(rows[k]["straight_frac"] for k in ("bezier", "polar", "hull"))
    assert g["mean_chicanes"] > others_chicanes, (g["mean_chicanes"], others_chicanes)
    assert g["straight_frac"] > others_straight, (g["straight_frac"], others_straight)
    assert g["shape_variety_pass"]  # not degenerate (median compactness < 0.85)
```

- [ ] **Step 2: Run to verify it fails, then passes after Task 3 is in**

Run: `.venv/bin/python -m pytest tests/test_generate_grammar.py -q`
Expected: PASS once Tasks 1–3 are complete. If `test_grammar_adds_net_new_features_vs_other_generators` fails, the grammar is not adding real features → return to Task 1 tuning (raise `grammar_chicane_bias` / `grammar_straight_frac`), do NOT weaken the assertion.

- [ ] **Step 3: Render a confirming grid (manual gate, not committed)**

Run `.venv/bin/python -m benchmarks.compare_generators --generators bezier polar hull grammar --E 1024` and a quick `viz` render of a grammar batch; eyeball that straights/hairpins/chicanes are visibly present. This is the human arbiter the spec requires.

- [ ] **Step 4: Commit**

```bash
git add tests/test_generate_grammar.py
git commit --no-gpg-sign -m "test(grammar): e2e + determinism + feature-presence acceptance"
```

---

## Task 5: Surface the grammar in the gradio explorer

**Files:**
- Modify: `viz/param_explorer.py` (`default_params`, `build_config`, the `controls`/`_collect` ordering if present, and the UI sliders)

**Interfaces:**
- Consumes: the `grammar_*` config fields; `generator_registry.available()` (the dropdown already lists `"grammar"`).

- [ ] **Step 1: Wire `grammar_*` into `build_config`**

In `viz/param_explorer.py` `build_config`, alongside the existing `polar_*` optional-kwargs block, add:

```python
    if p.get("grammar_segments") is not None:
        kw["grammar_segments"] = int(p["grammar_segments"])
    if p.get("grammar_straight_frac") is not None:
        kw["grammar_straight_frac"] = float(p["grammar_straight_frac"])
    if p.get("grammar_curvature_budget") is not None:
        kw["grammar_curvature_budget"] = float(p["grammar_curvature_budget"])
    if p.get("grammar_chicane_bias") is not None:
        kw["grammar_chicane_bias"] = float(p["grammar_chicane_bias"])
    if p.get("grammar_hairpin_max_frac") is not None:
        kw["grammar_hairpin_max_frac"] = float(p["grammar_hairpin_max_frac"])
```

- [ ] **Step 2: Add the `grammar_*` defaults to `default_params`**

In `default_params()`, add (reading the config defaults so it stays in sync):

```python
        "grammar_segments": cfg.grammar_segments,
        "grammar_straight_frac": cfg.grammar_straight_frac,
        "grammar_curvature_budget": cfg.grammar_curvature_budget,
        "grammar_chicane_bias": cfg.grammar_chicane_bias,
        "grammar_hairpin_max_frac": cfg.grammar_hairpin_max_frac,
```

- [ ] **Step 3: Add the UI sliders + controls/_collect wiring**

In `build_app`, add a `### Segment grammar` section with sliders for the five fields (e.g. `grammar_segments` `gr.Slider(4, 16, value=10, step=1)`, the fracs/biases `gr.Slider(0.0, 1.0, ...)` / `gr.Slider(0.5, 4.0, ...)`). If the file uses `controls`/`_collect` positional lists (check the current code), add the five components to BOTH the `controls` list and `_collect`'s key list at matching positions. (The current `build_config` reads via `p.get`, so the test helper `_params` without these keys still works — config defaults apply.)

- [ ] **Step 4: Verify the explorer test + a grammar build**

Run: `.venv/bin/python -m pytest tests/test_param_explorer.py -q` → green.
Run: `.venv/bin/python -c "from viz import param_explorer as px; c=px.build_config({**px.default_params(),'generator':'grammar'}); print(c.generator, c.grammar_segments, c.grammar_chicane_bias)"` → `grammar 10 1.0` (or the tuned values).

- [ ] **Step 5: Commit**

```bash
git add viz/param_explorer.py
git commit --no-gpg-sign -m "feat(viz): grammar generator knobs in the explorer"
```

---

## Self-Review (plan author)

**Spec coverage:** vocabulary + 2 κ-primitives + named patterns (Task 1 `sample_segments`/`rasterize_kappa`) ✓; budget + antisymmetry residual-taming (Task 1, tuned) ✓; heading DC-shift + gap-distribution closure, no host solve (Task 1 `close_and_integrate`, ported Task 3) ✓; normalize reuse (Task 3) ✓; config surface (Task 2) ✓; prototype-first in `_experimental` then warp port (Tasks 1→3) ✓; register `"grammar"` additively (Task 3) ✓; acceptance = shape-variety gate + feature presence + renders, not yield (Task 4 + Global Constraints) ✓; explorer knobs (Task 5) ✓; invariants (Global Constraints + Task 3 Steps 4–5) ✓.

**Placeholder scan:** the only deferred values are the tuned `DEFAULTS` (Task 1 Step 5 produces them; Task 2 copies the exact numbers) — explicitly flagged, not a code placeholder. No TBD/TODO in code steps.

**Type consistency:** `grammar_segments`/`grammar_straight_frac`/`grammar_curvature_budget`/`grammar_chicane_bias`/`grammar_hairpin_max_frac` used identically across the prototype `DEFAULTS`, the config fields, `build_config`, and `default_params`; `generate_grammar_warp(seeds_wp, config, out_centerline, out_valid_wp, scratch)` matches the contract used by `_run_pipeline`; `GrammarScratch` buffers (`segments`/`kappa`/`raw`) consistent between `grammar_alloc_scratch` and the kernels.
