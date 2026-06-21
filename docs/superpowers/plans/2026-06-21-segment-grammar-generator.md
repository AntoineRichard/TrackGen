# Segment-Grammar Generator (#6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth first-stage generator `"grammar"` that builds a closed centerline from an explicit racing-segment vocabulary (straights, sweepers, hairpins, chicanes, S-bends, kinks, clothoid/spiral transitions) via curvature integration, adding counted straights/hairpins/chicanes the catalog can't otherwise express.

**Architecture:** Per env (fixed-bound, graph-capturable): sample `S` segments (net-positive winding + budget) → rasterize a per-sample curvature profile `κ[i]` → scaling heading closure (`κ *= 2π/Σκ`, capped — preserves κ=0 straights) → integrate (`θ=cumsum(κ)`, edges = unit tangents) → gap-distribution displacement closure (subtract the mean edge so edges sum to 0) → center + isotropic bbox-rescale. Prototype-first in numpy to tune the grammar/budget defaults against the shape-variety gate + renders, then port to Warp mirroring `warp_generate_polar.py`.

**Tech Stack:** Python 3.10+, NVIDIA Warp (runtime), numpy; torch/matplotlib (dev-only, prototype + harness). `.venv/bin/python`.

## Global Constraints

- The runtime package `track_gen/_src/**` stays **Warp-native and torch-free**. The prototype's torch/numpy lives ONLY in `track_gen/_experimental/` (never imported by the runtime) and `benchmarks/` (dev). Verify: `grep -rn "import torch" track_gen/_src/` stays empty.
- **Zero per-call allocation** on the `generate()` path: `GrammarScratch` is pre-allocated in `grammar_alloc_scratch`. The CUDA-graph capture region must stay allocation-free — `tests/test_warp_graph.py` is the tripwire (it passes today; it must keep passing).
- **CUDA-graph-capturable** generator: pure Warp kernels, one env per row, fixed-bound loops over `S` and `N`, NO host sync, NO host-side closure solve, NO per-env Python branching.
- **Deterministic** in `(per-env seed, config)`; use the Warp RNG with a distinct salt (`_GRAMMAR_SALT`), decorrelated from bezier's count/corner streams and polar's `_CONTROL_SALT = 7919`.
- Registering `"grammar"` is **additive**: one new module + one `GeneratorSpec` + one import line in `generator_registry._ensure_loaded` + the `grammar_*` config fields. `track_gen.__all__` is unchanged.
- **Acceptance is the shape-variety gate, NOT yield:** `tests/test_shape_variety.py` must pass for `"grammar"` (median post-relax compactness < 0.65), AND `benchmarks/compare_generators.py` must show `straight_frac` for `"grammar"` clearly above bezier/hull/polar (its sustained straights — the star-shaped trio cannot hold a κ=0 span), AND a rendered seed grid must show real straights/corners. A perfect yield with no feature presence is a REJECT (the polar lesson). Do NOT gate on `mean_chicanes`: post-relax that metric measures wiggliness (turn-angle sign reversals on the dense relaxed curve), which is anti-correlated with grammar's net-winding design — grammar legitimately scores it LOWER than the star-shaped generators (see Task 4).
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

Not imported by the runtime. Tunes the grammar/budget defaults and validates closure +
character before the Warp port (track_gen/_src/warp_generate_grammar.py). Two kappa-primitives
(constant + linear-ramp) + named patterns. Heading is closed by SCALING the curvature
(kappa *= 2*pi/net_turn, capped), which PRESERVES kappa==0 straights — an earlier additive
DC-shift (kappa += 2*pi) lifted straights onto a constant arc and produced round blobs.
Displacement is closed by the gap-distribution pass. The grammar samples a NET-POSITIVE
winding (corners biased one way) with occasional gentle reversals (chicanes) + explicit
straight spans; scaling then normalizes the net turn to exactly 2*pi.
"""
from __future__ import annotations
import numpy as np

# Tuned in Step 5; Task 2 copies these into TrackGenConfig defaults.
DEFAULTS = dict(
    num_points=256,
    grammar_segments=18,             # S: alternating straight+corner segments (S//2 corners)
    grammar_straight_frac=0.45,      # target fraction of arc-length that is straight (kappa=0)
    grammar_curvature_budget=1.3,    # max per-corner turn angle (rad); sets hairpin tightness
    grammar_chicane_bias=0.22,       # fraction of corners that REVERSE sign (chicanes/S-bends)
    grammar_hairpin_max_frac=0.10,   # max arc-length fraction of any single corner span
    scale=1.0,
)
_BEZIER_EXTENT = 1.44               # match warp_generate_polar._BEZIER_EXTENT
# Cap the heading-closure scale factor (see close_and_integrate): when reverses nearly cancel
# the net winding, the raw 2*pi/net factor explodes and amplifies the curve into a tight
# self-crossing knot; clamping bounds that (XPBD repairs the small residual seam mismatch).
_HEADING_SCALE_CAP = 2.0


def sample_segments(rng: np.random.Generator, S: int, cfg: dict) -> np.ndarray:
    """Return [S, 3] rows (kappa_start, kappa_end, length_frac), pre-closure.

    Net-winding grammar: alternate a straight (kappa=0) with a corner. Corners are biased one
    direction (net winding) with varied turn angle (gentle sweeper .. tight hairpin); EXACTLY
    round(n_corner * chicane_bias) of them reverse sign (chicanes). Fixing the reverse COUNT
    (vs a per-corner coin flip) keeps net winding away from zero so the heading-closure scale
    factor stays bounded (rarely hits the cap -> far fewer self-crossing knots). Straight
    spans are longer on average so real straights dominate.
    """
    straight_frac = float(cfg["grammar_straight_frac"])
    sharp = float(cfg["grammar_curvature_budget"])       # max per-corner turn angle (rad)
    hairpin_max = float(cfg["grammar_hairpin_max_frac"])
    reverse_frac = float(cfg["grammar_chicane_bias"])
    n_corner = max(2, S // 2)
    n_neg = int(round(n_corner * reverse_frac))            # exact reverse count -> bounded winding
    neg_idx = set(rng.choice(n_corner, size=n_neg, replace=False).tolist()) if n_neg else set()
    segs = []
    for ci in range(n_corner):
        segs.append((0.0, 0.0, rng.uniform(0.06, 0.22)))   # straight (kappa=0), long on avg
        ang = rng.uniform(0.25, sharp)                      # corner turn angle (rad)
        ln = rng.uniform(0.02, hairpin_max)
        sgn = -1.0 if ci in neg_idx else 1.0                # chosen indices reverse = chicane
        k = sgn * ang / max(ln, 1e-6)                       # kappa = turn / span
        if rng.random() < 0.4:                              # 40% linear-ramp (clothoid)
            segs.append((0.0, k, ln))
        else:                                               # constant-kappa arc
            segs.append((k, k, ln))
    segs = np.array(segs[:S], dtype=np.float64)
    is_straight = (segs[:, 0] == 0.0) & (segs[:, 1] == 0.0)
    if is_straight.any() and (~is_straight).any():          # bias the straight/corner split
        segs[is_straight, 2] *= straight_frac / segs[is_straight, 2].sum()
        segs[~is_straight, 2] *= (1.0 - straight_frac) / segs[~is_straight, 2].sum()
    segs[:, 2] /= segs[:, 2].sum()                          # normalise length fractions to 1
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
    """Heading closure by SCALING (preserves kappa=0 straights) + integrate + gap-distribution
    displacement closure (subtract the mean edge so edge vectors sum to zero)."""
    N = kappa.shape[0]
    ds = 1.0 / N
    net = float((kappa * ds).sum())
    if abs(net) > 1e-6:
        sc = 2.0 * np.pi / net                  # net turn -> ~2*pi; zeros stay zero
        sc = max(-_HEADING_SCALE_CAP, min(_HEADING_SCALE_CAP, sc))
        kappa = kappa * sc
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

Use the `if __name__ == "__main__"` block in `grammar_proto.py`: generate ~500 seeds, compute over them with `benchmarks.track_metrics` — `compactness` percentiles (target median well < 0.65), `chicane_count` (target mean notably > bezier/hull/polar, e.g. ≥ 2), `straight_fraction` (target mean clearly > 0, e.g. ≥ 0.3) — and the pre-relax self-intersection rate (target in the budgeted ~35-45% band so the polygon fallback + XPBD recover ≥ the catalog's yields). Render a 5×5 grid with matplotlib and eyeball: straights, hairpins, and chicanes must be visibly present; most loops must read as real tracks (a tangled minority is expected and rides the fallback). Adjust `grammar_curvature_budget`, `grammar_chicane_bias`, `grammar_straight_frac`, `grammar_hairpin_max_frac`, `grammar_segments` until all targets pass, and update the `DEFAULTS` dict to the chosen values. (Tuned result, bounded-reverse grammar: median compactness ~0.56, chicanes ~2.9, straight_fraction ~0.53, self-intersection ~0.33.)

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
    assert cfg.grammar_segments == 18
    assert cfg.grammar_straight_frac == 0.45
    assert cfg.grammar_curvature_budget == 1.3
    assert cfg.grammar_chicane_bias == 0.22
    assert cfg.grammar_hairpin_max_frac == 0.10
```

(Match the exact tuned `DEFAULTS` from Task 1; update these numbers if tuning changed them.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_types.py::test_config_defaults_instantiate -q`
Expected: FAIL (`AttributeError: ...grammar_segments`).

- [ ] **Step 3: Add the fields to `TrackGenConfig`**

In `track_gen/_src/types.py` after line 85 (`polar_angular_jitter`):

```python
    # --- Segment-grammar (#6) params ---
    grammar_segments: int = 18           # S: fixed segment count (graph-capture bound); S//2 corners
    grammar_straight_frac: float = 0.45  # target arc-length fraction forced to straights (kappa=0)
    grammar_curvature_budget: float = 1.3   # max per-corner turn angle (rad); sets hairpin tightness
    grammar_chicane_bias: float = 0.22   # fraction of corners that reverse sign (chicane density)
    grammar_hairpin_max_frac: float = 0.10   # cap on any single corner's arc-length span
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
- Kernel `_grammar_sample_k(seeds, S, sharp, straight_frac, chicane_bias, hairpin_max_frac, segments)`: one thread per env; `wp.rand_init(seeds[e]*_GRAMMAR_SALT)`; draw the net-winding segment rows into `segments[e*S : ...]`, replicating `sample_segments` (alternate straight + corner over `S//2` corners; corner turn angle `uniform(0.25, sharp)`; flip EXACTLY `round(n_corner * chicane_bias)` corners negative = chicanes — a fixed reverse COUNT, not a per-corner coin flip, so net winding stays bounded away from zero and the scale cap rarely fires; pick the reverse indices with fixed-bound rejection sampling for uniqueness; `kappa = sign*ang/span`; 40% linear-ramp vs constant arc; then bias the straight/corner length split toward `straight_frac` and normalize length-fracs). Fixed bounded loops over `S`. `sharp` = `grammar_curvature_budget`.
- Kernel `_grammar_build_k(segments, S, N, target_extent, kappa, raw, out_centerline, out_valid)`: one thread per env; rasterize `kappa` from `segments` (prefix-sum bounds + interp, like `rasterize_kappa`); **scaling** heading closure (`net = Σκ·ds`; if `|net|>1e-6`, `κ *= clamp(2π/net, ±_HEADING_SCALE_CAP)` — preserves κ=0 straights, the additive DC-shift does NOT, and the cap stops a near-zero net winding from amplifying into a knot); integrate `theta`/edges; gap-distribution displacement closure (subtract mean edge); accumulate bbox; second pass center + isotropic rescale to `target_extent` into `out_centerline`; `out_valid[e]=1`. (You may instead reuse polar's `_normalize_centerline_k` for the center+rescale pass — import it from `warp_generate_polar` or copy the pattern.) `_HEADING_SCALE_CAP = 2.0`, a module constant.
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
    # The whole point: grammar makes sustained STRAIGHTS the star-shaped generators
    # structurally cannot (they have no kappa=0 spans). straight_fraction is the feature
    # metric that captures this and survives relaxation; assert grammar clearly leads it.
    #
    # NOTE: do NOT assert mean_chicanes > others. Post-relax, chicane_count counts turn-angle
    # SIGN REVERSALS on the dense relaxed centerline (i.e. wiggliness). The star-shaped
    # generators wander, so they score HIGHER on it; grammar is net-winding (mostly one
    # direction with a few deliberate chicanes), so it scores LOWER by design. The metric is
    # anti-correlated with grammar's character, so it is the wrong gate (measured: grammar
    # chicane_count ~9.8 vs bezier ~13.2 / polar ~12.7 at E=512, hw=0.1).
    cfg = TrackGenConfig(device="cpu", num_envs=128, half_width=0.1, relax_iters=40)
    rows = {r["generator"]: r for r in cg.compare(["bezier", "polar", "hull", "grammar"],
                                                  seed_base=0, E=128, base_config=cfg)}
    g = rows["grammar"]
    others_straight = max(rows[k]["straight_frac"] for k in ("bezier", "polar", "hull"))
    assert g["straight_frac"] > others_straight, (g["straight_frac"], others_straight)
    assert g["shape_variety_pass"]  # not degenerate (median compactness < 0.65)
```

- [ ] **Step 2: Run to verify it fails, then passes after Task 3 is in**

Run: `.venv/bin/python -m pytest tests/test_generate_grammar.py -q`
Expected: PASS once Tasks 1–3 are complete. If `test_grammar_adds_net_new_features_vs_other_generators` fails, the grammar is not adding real straights → return to Task 1 tuning (raise `grammar_straight_frac` and/or lengthen straight spans), do NOT weaken the assertion.

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

**Spec coverage:** vocabulary + 2 κ-primitives + named patterns (Task 1 `sample_segments`/`rasterize_kappa`) ✓; net-winding grammar + budget residual-taming (Task 1, tuned) ✓; scaling heading closure (capped) + gap-distribution closure, no host solve (Task 1 `close_and_integrate`, ported Task 3) ✓; normalize reuse (Task 3) ✓; config surface (Task 2) ✓; prototype-first in `_experimental` then warp port (Tasks 1→3) ✓; register `"grammar"` additively (Task 3) ✓; acceptance = shape-variety gate + feature presence + renders, not yield (Task 4 + Global Constraints) ✓; explorer knobs (Task 5) ✓; invariants (Global Constraints + Task 3 Steps 4–5) ✓.

**Placeholder scan:** the only deferred values are the tuned `DEFAULTS` (Task 1 Step 5 produces them; Task 2 copies the exact numbers) — explicitly flagged, not a code placeholder. No TBD/TODO in code steps.

**Type consistency:** `grammar_segments`/`grammar_straight_frac`/`grammar_curvature_budget`/`grammar_chicane_bias`/`grammar_hairpin_max_frac` used identically across the prototype `DEFAULTS`, the config fields, `build_config`, and `default_params`; `generate_grammar_warp(seeds_wp, config, out_centerline, out_valid_wp, scratch)` matches the contract used by `_run_pipeline`; `GrammarScratch` buffers (`segments`/`kappa`/`raw`) consistent between `grammar_alloc_scratch` and the kernels.
