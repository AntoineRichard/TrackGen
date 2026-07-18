# Knot-Based Track Elevation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide track altitude at ~10 arc-spaced control knots and interpolate with a periodic monotone cubic, so elevation smoothness stops depending on resample density.

**Architecture:** Knots sit at uniform arc fractions, so their cumulative table is analytic (`k·perim/K`) and knot sampling is the EXISTING `apply_z_profile` called with `stride=K`. One new kernel evaluates a Fritsch–Carlson-limited Hermite at each resampled point's arc fraction. `uniform`/`random_walk` route through knots; `noise` stays analytic per-point; `flat` is untouched. Spec: `docs/superpowers/specs/2026-07-18-knot-elevation-design.md`.

**Tech Stack:** Python, NVIDIA Warp (>= 1.14), pytest, Sphinx. No new dependencies.

**Execution model policy (user-mandated):** implementers/fixers on Opus or Sonnet ONLY; Fable reviews only. Suggested: Task 1 Opus (kernel math), Task 2 Opus (pipeline wiring), Task 3 Sonnet (docs/figures).

## Global Constraints

- Branch `feat/knot-elevation` off `main` (`d1d929e` or later).
- Golden gate: `pytest tests/test_golden_migration.py -v` exact after every task; `tests/goldens/pre_vec3f.npz` frozen. The `flat` path must not change at all.
- Gates must not change: `GateGenConfig` gains nothing; `pytest tests/test_gate_3d.py tests/test_gate_generator.py -q` is the no-change gate.
- `noise` output values must not change — it keeps the analytic per-point path.
- Everything added to `inflate_warp` stays inside the captured pipeline: allocation-free, sync-free, config-static Python branches only. New buffers are allocated in `_inflate_warp_alloc`.
- **Invariant:** XPBD solves in 2D; elevation is applied strictly after relaxation. Task 2 adds the regression test that pins it.
- GPU box: run `pytest tests/ -q` per task (cpu+cuda, currently 626) and `python -m sphinx -W -b html docs /tmp/sphinx_check` for docs tasks.
- Commits: conventional style, `git commit --no-gpg-sign`, append to every message:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019ABYQDYMWzWSk9H6aJ1x8p
```

---

### Task 1: Knot sampling + monotone cubic in `warp_zprofile`

Self-contained kernel work, unit-testable without touching the pipeline.

**Files:**
- Modify: `track_gen/_src/warp_zprofile.py` (append kernels + two functions after `apply_z_profile`, ~line 210)
- Modify: `track_gen/_src/types.py` (TrackGenConfig field ~line 658 block; validation after the `_validate_z_fields(self)` call at ~line 848)
- Test: `tests/test_zknots.py` (new)

**Interfaces:**
- Consumes: existing `apply_z_profile(config, seeds_wp, count, stride, cum, perim, z)` (unchanged).
- Produces, for Task 2:
  - `TrackGenConfig.z_control_points: int = 10` — validated `>= 3`; governs `uniform` and `random_walk` only.
  - `warp_zprofile.alloc_knot_scratch(num_envs, control_points, device) -> (knot_cum, knot_count, knot_z)` — `knot_cum`/`knot_z` are `[E*K] float32`, `knot_count` is `[E] int32`, all zero-filled.
  - `warp_zprofile.apply_z_profile_knots(config, seeds_wp, count, stride, arclen, perim, knot_cum, knot_count, knot_z, z) -> None` — launches only, capture-safe; fills `z` `[E*stride]` per point. `arclen` is the caller's `[E*stride]` plan-view cumulative table, `perim` its `[E]` closed-loop perimeter.
  - Kernel `_pchip_eval_k(arclen, perim, count, n_max, knot_z, K, z)` — exposed for direct unit testing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_zknots.py`:

```python
"""Knot sampling + periodic monotone-cubic interpolation for track elevation."""
import numpy as np
import pytest
import warp as wp

from track_gen._src import warp_zprofile
from track_gen._src.types import TrackGenConfig

E, K, N = 2, 4, 240


def _eval(knots, perim=1.0, n=N):
    """Evaluate _pchip_eval_k on one env at n uniformly-spaced arc positions."""
    kz = np.tile(np.asarray(knots, np.float32), (E, 1)).reshape(-1)
    arc = np.tile((np.arange(n, dtype=np.float32) / n) * perim, (E, 1)).reshape(-1)
    dev = "cpu"
    out = wp.zeros(E * n, dtype=wp.float32, device=dev)
    wp.launch(
        warp_zprofile._pchip_eval_k, dim=E * n,
        inputs=[wp.array(arc, dtype=wp.float32, device=dev),
                wp.array(np.full(E, perim, np.float32), dtype=wp.float32, device=dev),
                wp.array(np.full(E, n, np.int32), dtype=wp.int32, device=dev),
                n,
                wp.array(kz, dtype=wp.float32, device=dev),
                len(knots),
                out],
        device=dev)
    return out.numpy().reshape(E, n)[0]


def test_interpolates_knots_exactly():
    # knot j sits at arc fraction j/K -> sample index j*n/K
    knots = [0.0, 1.0, 0.5, 2.0]
    z = _eval(knots)
    for j, kv in enumerate(knots):
        assert abs(z[j * N // K] - kv) < 1e-5, f"knot {j}"


def test_no_overshoot_on_alternating_knots():
    # The classic overshoot trap: Catmull-Rom would exceed [0, 1] here.
    knots = [0.0, 1.0, 0.0, 1.0]
    z = _eval(knots)
    assert z.max() <= 1.0 + 1e-5
    assert z.min() >= 0.0 - 1e-5


def test_no_overshoot_on_monotone_run():
    knots = [0.0, 1.0, 2.0, 3.0]
    z = _eval(knots)
    assert z.max() <= 3.0 + 1e-5 and z.min() >= 0.0 - 1e-5


def test_flat_knots_give_flat_curve():
    z = _eval([1.25, 1.25, 1.25, 1.25])
    np.testing.assert_allclose(z, 1.25, atol=1e-6)


def test_periodic_closure_is_smooth():
    # Wrapping past the last knot must return toward knot 0 continuously:
    # the value just before the seam is close to the value just after it.
    knots = [0.0, 1.0, 0.5, 2.0]
    z = _eval(knots, n=1000)
    assert abs(z[-1] - z[0]) < 3.0 * abs(z[1] - z[0]) + 1e-4


def test_padding_slots_are_zero():
    dev = "cpu"
    n = 8
    out = wp.zeros(n, dtype=wp.float32, device=dev)
    wp.launch(
        warp_zprofile._pchip_eval_k, dim=n,
        inputs=[wp.array(np.linspace(0, 1, n, dtype=np.float32), dtype=wp.float32, device=dev),
                wp.array(np.array([1.0], np.float32), dtype=wp.float32, device=dev),
                wp.array(np.array([5], np.int32), dtype=wp.int32, device=dev),  # count=5 of 8
                n,
                wp.array(np.array([0.0, 1.0, 0.0, 1.0], np.float32), dtype=wp.float32, device=dev),
                4,
                out],
        device=dev)
    assert (out.numpy()[5:] == 0.0).all()


def test_knot_pipeline_bounds_and_smoothness():
    """End-to-end through apply_z_profile_knots: uniform draws at K knots,
    interpolated over many points, stays in bounds and stays smooth."""
    Kc, S = 8, 200
    cfg = TrackGenConfig(device="cpu", num_envs=E, z_profile="uniform",
                         z_min=0.5, z_max=1.5, z_control_points=Kc)
    dev = "cpu"
    perim_np = np.full(E, 4.0, np.float32)
    arc_np = np.tile(np.linspace(0.0, 4.0, S, endpoint=False, dtype=np.float32), (E, 1)).reshape(-1)
    knot_cum, knot_count, knot_z = warp_zprofile.alloc_knot_scratch(E, Kc, dev)
    z = wp.zeros(E * S, dtype=wp.float32, device=dev)
    warp_zprofile.apply_z_profile_knots(
        cfg, wp.array(np.array([3, 11], np.int32), dtype=wp.int32, device=dev),
        wp.array(np.full(E, S, np.int32), dtype=wp.int32, device=dev), S,
        wp.array(arc_np, dtype=wp.float32, device=dev),
        wp.array(perim_np, dtype=wp.float32, device=dev),
        knot_cum, knot_count, knot_z, z)
    zz = z.numpy().reshape(E, S)
    kz = knot_z.numpy().reshape(E, Kc)
    for e in range(E):
        assert (zz[e] >= 0.5 - 1e-5).all() and (zz[e] <= 1.5 + 1e-5).all()
        # no overshoot beyond the sampled knots
        assert zz[e].max() <= kz[e].max() + 1e-5
        assert zz[e].min() >= kz[e].min() - 1e-5
        # smooth: far fewer direction changes than points
        turns = int(np.sum(np.diff(np.sign(np.diff(zz[e]))) != 0))
        assert turns <= Kc, f"{turns} direction changes for K={Kc}"


def test_config_validation():
    with pytest.raises(ValueError, match="z_control_points"):
        TrackGenConfig(device="cpu", num_envs=1, z_control_points=2)
    assert TrackGenConfig(device="cpu", num_envs=1).z_control_points == 10
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zknots.py -x -q`
Expected: FAIL — `module 'track_gen._src.warp_zprofile' has no attribute '_pchip_eval_k'` (and unknown config field).

- [ ] **Step 3: Add the config knob**

In `track_gen/_src/types.py`, in `TrackGenConfig`'s Z-profile field block (~line 658, next to `z_valid_grade`):

```python
    z_control_points: int = 10
```

with a numpydoc entry in the class docstring's attribute list, matching the neighbouring `z_*` entries in style:

```
    z_control_points : int
        Number of arc-length-spaced altitude control knots. Altitude is decided
        at these knots and interpolated between them with a periodic monotone
        cubic, so smoothness is independent of the resampled point count. Must
        be >= 3. Applies to ``z_profile="uniform"`` and ``"random_walk"``;
        inert for ``"flat"`` (constant) and ``"noise"`` (analytically smooth —
        use ``z_noise_harmonics`` for its frequency). Default 10.
```

Validation goes in `TrackGenConfig`'s own `__post_init__`, immediately after the existing `_validate_z_fields(self)` call (~line 848) — NOT in the shared helper, since `GateGenConfig` has no such field:

```python
        if int(self.z_control_points) < 3:
            raise ValueError(
                "z_control_points must be >= 3, got "
                f"{self.z_control_points!r}")
```

- [ ] **Step 4: Add the kernels and functions**

Append to `track_gen/_src/warp_zprofile.py`, after `apply_z_profile`:

```python
@wp.func
def _pchip_tangent(d_prev: float, d_next: float) -> float:
    """Monotonicity-preserving knot tangent (uniform knot spacing).

    Zero at a local extremum (``d_prev * d_next <= 0``) — this is what kills
    overshoot — otherwise the secant average, magnitude-capped at
    ``3 * min(|d_prev|, |d_next|)``, the standard Fritsch-Carlson sufficient
    condition for a monotone cubic segment.
    """
    if d_prev * d_next <= 0.0:
        return 0.0
    avg = 0.5 * (d_prev + d_next)
    lim = 3.0 * wp.min(wp.abs(d_prev), wp.abs(d_next))
    if avg > lim:
        return lim
    if avg < -lim:
        return -lim
    return avg


@wp.kernel
def _knot_tables_k(
    perim: wp.array(dtype=wp.float32),
    K: int,
    knot_cum: wp.array(dtype=wp.float32),
    knot_count: wp.array(dtype=wp.int32),
):
    # Knots sit at uniform arc fractions, so their cumulative table is
    # analytic: knot k is at k * perim / K. All K slots are real.
    t = wp.tid()               # dim = E * K
    e = t // K
    k = t - e * K
    knot_cum[t] = float(k) * perim[e] / float(K)
    if k == 0:
        knot_count[e] = K


@wp.kernel
def _pchip_eval_k(
    arclen: wp.array(dtype=wp.float32),
    perim: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    knot_z: wp.array(dtype=wp.float32),
    K: int,
    z: wp.array(dtype=wp.float32),
):
    """Periodic monotone cubic through the K knots, sampled at each point's
    plan-view arc fraction. Padding slots (``i >= count[e]``) get 0.0, matching
    the per-point profile kernels."""
    t = wp.tid()               # dim = E * n_max
    e = t // n_max
    i = t - e * n_max
    m = count[e]
    if m > n_max:
        m = n_max
    if i >= m:
        z[t] = 0.0
        return

    kbase = e * K
    P = perim[e]
    if P <= 1.0e-9:
        z[t] = knot_z[kbase]
        return

    h = P / float(K)
    x = (arclen[t] / P) * float(K)
    k = int(x)
    if k < 0:
        k = 0
    if k > K - 1:
        k = K - 1
    u = wp.clamp(x - float(k), 0.0, 1.0)

    kp = (k + K - 1) % K
    k1 = (k + 1) % K
    k2 = (k + 2) % K
    z0 = knot_z[kbase + k]
    z1 = knot_z[kbase + k1]
    d_prev = (z0 - knot_z[kbase + kp]) / h
    d_here = (z1 - z0) / h
    d_next = (knot_z[kbase + k2] - z1) / h
    m0 = _pchip_tangent(d_prev, d_here)
    m1 = _pchip_tangent(d_here, d_next)

    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    z[t] = h00 * z0 + h10 * h * m0 + h01 * z1 + h11 * h * m1


def alloc_knot_scratch(num_envs: int, control_points: int, device):
    """``(knot_cum, knot_count, knot_z)`` knot-stage scratch, zero-initialized.

    ``knot_cum``/``knot_z`` are ``[E * K]`` float32; ``knot_count`` is ``[E]``
    int32. Allocated once by the pipeline; never allocated on the hot path.
    """
    E, K = int(num_envs), int(control_points)
    dev = str(device)
    return (wp.zeros(E * K, dtype=wp.float32, device=dev),
            wp.zeros(E, dtype=wp.int32, device=dev),
            wp.zeros(E * K, dtype=wp.float32, device=dev))


def apply_z_profile_knots(config, seeds_wp: wp.array, count: wp.array,
                          stride: int, arclen: wp.array, perim: wp.array,
                          knot_cum: wp.array, knot_count: wp.array,
                          knot_z: wp.array, z: wp.array) -> None:
    """Knot-based altitude: sample K control knots, then interpolate per point.

    Three launches, capture-safe (no alloc, no sync): build the analytic knot
    arc table, run the configured profile over the K knots via
    :func:`apply_z_profile` (``stride = K``), then evaluate the periodic
    monotone cubic at every point's arc fraction. Because the knots carry the
    profile's own clamp to ``[z_min, z_max]`` and the interpolant never leaves
    the interval between adjacent knots, the per-point result needs no
    additional clamping.

    Intended for ``z_profile`` in ``{"uniform", "random_walk"}``; ``"noise"``
    is analytically smooth and should call :func:`apply_z_profile` directly.
    """
    E = int(config.num_envs)
    K = int(config.z_control_points)
    dev = str(config.device)
    wp.launch(_knot_tables_k, dim=E * K,
              inputs=[perim, K, knot_cum, knot_count], device=dev)
    apply_z_profile(config, seeds_wp, knot_count, K, knot_cum, perim, knot_z)
    wp.launch(_pchip_eval_k, dim=E * int(stride),
              inputs=[arclen, perim, count, int(stride), knot_z, K, z],
              device=dev)
```

Update the module docstring's opening paragraph to mention both parameterizations: per-anchor (gates, `noise`) and knot-based (tracks, `uniform`/`random_walk`).

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_zknots.py -v` then `pytest tests/test_zprofile.py tests/test_types.py tests/test_gate_3d.py tests/test_golden_migration.py -q` then `pytest tests/ -q`
Expected: all PASS (nothing is wired into the pipeline yet, so track/gate behavior is unchanged).

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/warp_zprofile.py track_gen/_src/types.py tests/test_zknots.py
git commit --no-gpg-sign -m "feat: knot sampling + periodic monotone cubic for track elevation"
```

---

### Task 2: Wire the knot path into the pipeline

**Files:**
- Modify: `track_gen/_src/warp_pipeline.py` — scratch alloc (~1567 owned, ~1779 standalone, ~1798 lazy), the stage-4b dispatch (~1859-1866)
- Test: `tests/test_track_3d.py` (additions)

**Interfaces:**
- Consumes: Task 1's `alloc_knot_scratch` / `apply_z_profile_knots` / `TrackGenConfig.z_control_points`.
- Produces: track elevation whose smoothness is set by `z_control_points`, not point count. `flat`/`noise` paths and all gate paths unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_track_3d.py`:

```python
def _turns(z):
    """Number of direction changes in an elevation series."""
    return int(np.sum(np.diff(np.sign(np.diff(z))) != 0))


def test_smoothness_is_independent_of_resample_density():
    """THE regression gate. Elevation direction changes must track
    z_control_points, not the resampled point count: doubling the density
    must not roughly double the bumps (which is what per-point profiling did).
    """
    counts, turns = [], []
    for spacing in (0.06, 0.03):
        cfg = TrackGenConfig(device="cpu", num_envs=E, spacing=spacing,
                             z_profile="random_walk", z_base=1.0, z_min=0.2,
                             z_max=2.0, z_max_step=0.3, z_control_points=8)
        rng = PerEnvSeededRNG(seeds=1234, num_envs=E, device="cpu")
        track = TrackGenerator(cfg, rng).generate()
        e = int(np.flatnonzero(track.valid.numpy())[0])
        n_max = track.center.shape[0] // E
        m = int(track.count.numpy()[e])
        z = track.center.numpy().reshape(E, n_max, 3)[e, :m, 2]
        counts.append(m)
        turns.append(_turns(z))
    assert counts[1] > 1.5 * counts[0], f"densities not distinct: {counts}"
    assert turns[0] <= 8 and turns[1] <= 8, f"too many turns: {turns}"
    assert abs(turns[1] - turns[0]) <= 2, f"turns track density: {turns}"


def test_uniform_profile_is_smooth_not_jitter():
    cfg = TrackGenConfig(device="cpu", num_envs=E, z_profile="uniform",
                         z_min=0.5, z_max=1.5, z_control_points=10)
    rng = PerEnvSeededRNG(seeds=1234, num_envs=E, device="cpu")
    track = TrackGenerator(cfg, rng).generate()
    n_max = track.center.shape[0] // E
    for e in np.flatnonzero(track.valid.numpy()):
        m = int(track.count.numpy()[e])
        z = track.center.numpy().reshape(E, n_max, 3)[e, :m, 2]
        assert (z >= 0.5 - 1e-5).all() and (z <= 1.5 + 1e-5).all()
        assert _turns(z) <= 10, f"env {e}: {_turns(z)} turns"
        assert z.std() > 0.0


def test_xpbd_solves_in_2d_elevation_applies_after():
    """INVARIANT: relaxation runs before elevation exists, so the plan-view
    geometry must be bit-identical between a flat and a hilly config."""
    def gen(**kw):
        cfg = TrackGenConfig(device="cpu", num_envs=E, **kw)
        rng = PerEnvSeededRNG(seeds=99, num_envs=E, device="cpu")
        return TrackGenerator(cfg, rng).generate()

    flat = gen()
    hilly = gen(z_profile="random_walk", z_base=1.0, z_min=0.2, z_max=2.0,
                z_max_step=0.3)
    np.testing.assert_array_equal(flat.count.numpy(), hilly.count.numpy())
    np.testing.assert_array_equal(flat.valid.numpy(), hilly.valid.numpy())
    for name in ("center", "outer", "inner"):
        a = getattr(flat, name).numpy()[:, :2]
        b = getattr(hilly, name).numpy()[:, :2]
        np.testing.assert_array_equal(a, b, err_msg=f"{name} xy moved")
    assert getattr(hilly, "center").numpy()[:, 2].std() > 0.0  # z really varies


def test_no_overshoot_beyond_extremes():
    cfg = TrackGenConfig(device="cpu", num_envs=E, z_profile="random_walk",
                         z_base=1.0, z_min=0.6, z_max=1.4, z_max_step=0.4,
                         z_control_points=6)
    rng = PerEnvSeededRNG(seeds=7, num_envs=E, device="cpu")
    track = TrackGenerator(cfg, rng).generate()
    n_max = track.center.shape[0] // E
    for e in np.flatnonzero(track.valid.numpy()):
        m = int(track.count.numpy()[e])
        z = track.center.numpy().reshape(E, n_max, 3)[e, :m, 2]
        assert z.max() <= 1.4 + 1e-5 and z.min() >= 0.6 - 1e-5


def test_noise_profile_unchanged_by_control_points():
    """noise stays analytic per-point: z_control_points must not affect it."""
    def gen(K):
        cfg = TrackGenConfig(device="cpu", num_envs=E, z_profile="noise",
                             z_base=1.0, z_noise_amplitude=0.4, z_min=0.0,
                             z_max=2.0, z_control_points=K)
        rng = PerEnvSeededRNG(seeds=21, num_envs=E, device="cpu")
        return TrackGenerator(cfg, rng).generate().center.numpy()[:, 2]
    np.testing.assert_array_equal(gen(4), gen(20))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_track_3d.py -x -q -k "density or jitter or overshoot"`
Expected: FAIL — with per-point profiling, `test_smoothness_is_independent_of_resample_density` reports far more than 8 turns (measured ~52 at the default density).

- [ ] **Step 3: Allocate the knot scratch**

In `warp_pipeline.py`, wherever `stag.z` is created, add the knot buffers alongside. The owned allocation (~line 1567) and the standalone allocation (~line 1779) both currently build `z=wp.zeros(flat, ...)`; extend the staging structure with three more fields:

```python
        knot_cum=None, knot_count=None, knot_z=None,
```

and fill them from the config when the profile needs them (config-static, construction time only, not the hot path):

```python
    # Knot-stage scratch: only the knot-based profiles need it.
    if getattr(config, "z_profile", "flat") in ("uniform", "random_walk"):
        stag.knot_cum, stag.knot_count, stag.knot_z = \
            warp_zprofile.alloc_knot_scratch(
                E, int(config.z_control_points), dev)
```

Mirror this at the lazy `if stag.z is None:` site (~1798) so the standalone path also has them. Add the three fields to the staging dataclass definition and to its docstring's field list, following the existing `z:` entry's wording.

- [ ] **Step 4: Dispatch on the profile**

Replace the stage-4b block (~1859-1866) with:

```python
    # 4b. Elevation (2.5D): fill the per-point altitude from the configured profile.
    # CRITICAL ordering: out.arclen/out.length still hold the 2D (cum, perim) tables
    # from stage 3 here — the profiler consumes them as its plan-view arc
    # parameterization, BEFORE _track_frames3_k (stage 6, non-flat) overwrites them
    # with 3D values. Skipped on the flat path (scratch.z stays zero-filled), which
    # keeps the arclen/length/tangent tables byte-identical to the legacy 2D path.
    #
    # uniform/random_walk decide altitude at z_control_points arc-spaced KNOTS and
    # interpolate with a periodic monotone cubic, so smoothness is set by the knot
    # count rather than the resampled point count. noise is analytically smooth
    # already (its harmonics band-limit it), so it stays per-point.
    _is_flat = _z_profile == "flat"
    if _z_profile in ("uniform", "random_walk"):
        warp_zprofile.apply_z_profile_knots(
            config, seeds, cnt_wp, n_max, out.arclen, out.length,
            stag.knot_cum, stag.knot_count, stag.knot_z, stag.z)
    elif not _is_flat:
        warp_zprofile.apply_z_profile(
            config, seeds, cnt_wp, n_max, out.arclen, out.length, stag.z)
```

Nothing else in the stage order changes: validity still reads `stag.z`, and the lift/frames3 branch is untouched.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_track_3d.py tests/test_zknots.py tests/test_golden_migration.py -q`, then `pytest tests/ -q`, then `pytest tests/ -q -m cuda`
Expected: all PASS. Goldens exact (flat untouched); gates untouched; `noise` values unchanged.

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/warp_pipeline.py tests/test_track_3d.py
git commit --no-gpg-sign -m "feat: route uniform/random_walk track elevation through control knots"
```

---

### Task 3: Docs and figures

**Files:**
- Create: `viz/plot_z_profiles.py`
- Modify: `docs/tracks-25d.rst`, `docs/_static/tracks-25d.png` (re-shot), add `docs/_static/z-profiles.png`
- Modify: `track_gen/_src/warp_zprofile.py` (module docstring only, if Task 1 left it thin)

**Interfaces:** none new.

- [ ] **Step 1: Profile-comparison figure**

Create `viz/plot_z_profiles.py` following `viz/plot_tracks_3d.py`'s structure (Agg backend before pyplot, argparse `--seed`/`--out`/`--envs`, save-not-show). One ROW per profile (`flat`, `uniform`, `random_walk`, `noise`), same seed and same track layout across rows, two columns: plan view coloured by z, and elevation vs arclength. Use `z_base=1.0`, `z_min=0.5`, `z_max=1.5`, `z_max_step=0.3`, `z_noise_amplitude=0.4`, `z_control_points=10`. Title each row with the profile name and its governing knob. Render, then Read the PNG and confirm the four rows are visually distinct and all but `flat` show smooth rolling elevation (not jitter). Commit as `docs/_static/z-profiles.png`.

- [ ] **Step 2: Re-shoot the hero figure**

Run: `python viz/plot_tracks_3d.py --out docs/_static/tracks-25d.png --envs 3 --seed 5`
Read the PNG: the elevation panels must now be smooth rolling curves rather than the previous washboard. If relief looks too subtle to read, raise the figure's `z_max_step`/`z_min`/`z_max` in the script's defaults so the elevation is visible against the ~1.5 m track extent, and note the change in the report.

- [ ] **Step 3: Docs**

In `docs/tracks-25d.rst`:
- Add a `z_control_points` row to the knobs table: "Number of arc-spaced altitude control knots (default 10, must be ≥ 3). Applies to `uniform` and `random_walk`; `flat` is constant and `noise` uses `z_noise_harmonics` for its frequency."
- Replace the neutral per-profile descriptions with guidance on what each is *for*: `flat` = planar (the default), `uniform` = independent hill heights, `random_walk` = correlated rolling terrain with a grade-capped walk, `noise` = smooth periodic terrain with harmonic control.
- Add a short "Smoothness" paragraph: altitude is decided at the knots and interpolated with a periodic monotone cubic, so the interpolant never leaves the interval between adjacent knot values (`[z_min, z_max]` holds exactly, with no post-clamp), and raising the resample resolution samples the SAME road more finely rather than making it bumpier. State the grade caveat plainly: a monotone-cubic segment can reach up to 3× its knot-to-knot secant, so `z_max_step` shapes the walk at the knots while `z_valid_grade` is what gates realized per-point steepness.
- Reference the new `z-profiles.png` with a figure directive and a reproduction command line, matching the existing figure block's style.

- [ ] **Step 4: Verify**

Run: `python -m sphinx -W -b html docs /tmp/sphinx_check` (zero warnings), `pytest tests/ -q` (626+), `pytest tests/test_golden_migration.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add viz/plot_z_profiles.py docs/
git commit --no-gpg-sign -m "docs+viz: profile-comparison figure, knot smoothness guidance, re-shot hero"
```

---

## Plan Self-Review Notes (resolved)

- **Spec coverage:** §1 knot stage → Task 1 (config knob + validation, knot tables reusing `apply_z_profile` at `stride=K`, monotone limiter, bounds guarantee tested directly); §2 wiring + invariant → Task 2 (dispatch, scratch, XPBD bit-identical-XY test); §3 testing/docs/figures → Tasks 1–3 (density independence, no overshoot, closure, determinism via the `noise`-unchanged and seeded tests, padding, validation; both figures and the guidance prose). Spec non-goals respected: gates, `flat`, and `noise` values all untouched, and the heightfield medial-axis finding is explicitly NOT addressed here.
- **Type consistency:** `alloc_knot_scratch(num_envs, control_points, device) -> (knot_cum, knot_count, knot_z)` and `apply_z_profile_knots(config, seeds_wp, count, stride, arclen, perim, knot_cum, knot_count, knot_z, z)` are used with identical argument order in Task 1's test, Task 1's definition, and Task 2's dispatch. `_pchip_eval_k`'s parameter order matches between its definition, the orchestrator's launch, and the unit tests' direct launches.
- **Known softness, deliberate:** Task 2 Step 3 gives the staging-field edit as a pattern rather than exact line surgery, because the staging dataclass appears at three sites (owned alloc, standalone alloc, lazy fill) whose exact shape the implementer must read first — the required end state (three fields, allocated only for knot profiles, never on the hot path) is stated precisely. Task 3 Step 2 permits adjusting the figure's elevation defaults for legibility, with disclosure required.
