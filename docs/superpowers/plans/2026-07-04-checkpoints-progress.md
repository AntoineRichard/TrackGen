# Checkpoints, Progress & Disc Collision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the modular course-progress family: `track_gen.checkpoints` (shared `CheckpointSet` from gates or subsampled track centerlines), `track_gen.progress` (stateful `ProgressTracker` with pass/wrong-way/wrong-checkpoint/laps/`dist_to_next`), `track_gen.collision.DiscChecker` (box-vs-disc obstacles, e.g. gate posts), the stable input-binding contract across checkers, and the docs deliverable (tutorial page + three deterministic figures).

**Architecture:** `CheckpointSet` is the consumer contract (position/left/right/tangent/count, flat `[E*M]`); `from_gates` aliases `GateSequence` buffers zero-copy, `CheckpointSampler` resamples the centerline (props scan kernel reused) emitting index-aligned inner/outer as crossing segments. `ProgressTracker` owns `[E]` device state and advances it in one fused kernel per `update()`; `reset(mask)` uses a NaN-prev-pos sentinel. `DiscChecker` mirrors `CollisionChecker`'s shape with a disc loop. Per-step inputs can be bound once to user-owned stable arrays (graph-capture-native); all tool-owned buffers are preallocated with stable pointers.

**Tech Stack:** Python ≥ 3.10, warp-lang ≥ 1.14, numpy. Tests: pytest (+ torch in cuda tests). Docs: sphinx + the `viz/` deterministic asset-renderer pattern.

**Spec:** `docs/superpowers/specs/2026-07-04-checkpoints-progress-design.md`

## Global Constraints

- Runtime deps numpy + warp-lang only; numpy at construction time only (never in `update()`/`query()`/`sample()`).
- Everything via `wp.launch(..., inputs=[...])`; per-step methods allocation-free; module `_INITED`/`_CAPTURING`/`_init`/`_sync` pattern copied from `track_gen/_src/collision.py`.
- Flat NaN-padded layouts; in-place result reuse + `clone()`; results undefined for `valid[e] == 0`; NaN-proof numeric validation (`not (x > 0)` form).
- **Stable-buffer contract:** all tool-owned state/scratch/results allocated in `__init__` only, written in place. Bound-input mode: bind user arrays once (validated at bind time), per-step methods take no args and read them in place; per-call mode stays supported; mixing modes raises `ValueError`.
- Progress semantics (spec-exact): pass = swept segment `prev→pos` properly crosses target's `left↔right` AND `dot(pos−prev, tangent) > 0`; at most one advance per update; wrong-checkpoint scan runs against the ORIGINAL target index (a double-jump advances one and flags the second the same update); backward target crossing → `wrong_way=1`; `dist_to_next` computed AFTER any advance; NaN prev-pos = inert first step; `count[e] < 1` → inert events with NaN dist.
- Checkpoint snap rule (spec-exact, same as props): `n = clamp(round(perimeter/spacing), 3, max_checkpoints)`, `step = perimeter/n`, on the CENTERLINE closed polyline; degenerate (`track.count[e] < 3`) → checkpoint count 0, all NaN.
- Disc semantics: hit iff `distance(disc center, solid OBB) <= radius`; deepest disc reported; NaN discs skipped; NaN boxes inert (`hit=0, disc=-1, depth=0, nearest=NaN`).
- warp-1.14 codegen cautions (from prior builds): wrap module-constant seeds mutated in dynamic loops as `float(_BIG)`; literals are safe; compare snapped counts in float before int casts.
- Every pytest run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest …`
- Work on branch `feature/checkpoints-progress` off main (Task 1 Step 0).

---

### Task 1: Shared geometry — segment-segment crossing in `collision_geom`

**Files:**
- Modify: `track_gen/_src/collision_geom.py` (append two `@wp.func`s)
- Modify: `tests/test_collision_geom.py` (append kernel-wrapper tests)

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Tasks 4, 9): `_cross2(a: wp.vec2f, b: wp.vec2f) -> float` (2D cross product) and `_segs_cross(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, d: wp.vec2f) -> int` (1 iff segments ab and cd PROPERLY intersect — strict crossing, collinear/touching returns 0; degenerate zero-length cd returns 0).

- [ ] **Step 0: Create the feature branch**

```bash
git checkout -b feature/checkpoints-progress
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_collision_geom.py`:

```python
@wp.kernel
def _k_segs_cross(a: wp.array(dtype=wp.vec2f), b: wp.array(dtype=wp.vec2f),
                  c: wp.vec2f, d: wp.vec2f, out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = cg._segs_cross(a[i], b[i], c, d)


def test_segs_cross_proper_and_degenerate():
    # Fixed segment cd: vertical from (0,-1) to (0,1).
    a = np.array([[-1.0, 0.0], [-1.0, 2.0], [-1.0, 0.0], [0.0, 0.0]], np.float32)
    b = np.array([[1.0, 0.0], [1.0, 2.0], [-0.2, 0.0], [1.0, 0.0]], np.float32)
    aw = wp.array(a, dtype=wp.vec2f, device="cpu")
    bw = wp.array(b, dtype=wp.vec2f, device="cpu")
    out = wp.zeros(4, dtype=wp.int32, device="cpu")
    _run(_k_segs_cross, 4, [aw, bw, wp.vec2f(0.0, -1.0), wp.vec2f(0.0, 1.0), out])
    # crossing; passing above; stopping short; starting ON the segment (touch -> 0)
    assert list(out.numpy()) == [1, 0, 0, 0]


def test_segs_cross_degenerate_cd_never_crosses():
    # Zero-length cd (a gate with width 0) can never be properly crossed.
    a = np.array([[-1.0, 0.0]], np.float32)
    b = np.array([[1.0, 0.0]], np.float32)
    aw = wp.array(a, dtype=wp.vec2f, device="cpu")
    bw = wp.array(b, dtype=wp.vec2f, device="cpu")
    out = wp.zeros(1, dtype=wp.int32, device="cpu")
    _run(_k_segs_cross, 1, [aw, bw, wp.vec2f(0.0, 0.0), wp.vec2f(0.0, 0.0), out])
    assert list(out.numpy()) == [0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_geom.py -v -k segs_cross`
Expected: FAIL — `AttributeError: ... has no attribute '_segs_cross'`

- [ ] **Step 3: Append the implementation to `track_gen/_src/collision_geom.py`**

```python
@wp.func
def _cross2(a: wp.vec2f, b: wp.vec2f) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def _segs_cross(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f, d: wp.vec2f) -> int:
    """1 iff segments ab and cd properly intersect (strict crossing).

    Collinear overlap, endpoint touching, and degenerate (zero-length)
    segments all return 0 — a width-0 gate can never be crossed.
    """
    ab = b - a
    cd = d - c
    o1 = _cross2(ab, c - a)
    o2 = _cross2(ab, d - a)
    o3 = _cross2(cd, a - c)
    o4 = _cross2(cd, b - c)
    hit_ab = (o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)
    hit_cd = (o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0)
    if hit_ab and hit_cd:
        return int(1)
    return int(0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_geom.py -v`
Expected: all PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/collision_geom.py tests/test_collision_geom.py
git commit -m "feat: shared segment-segment proper-crossing helper"
```

---

### Task 2: `track_gen.checkpoints` — `CheckpointSet`, `from_gates`, `CheckpointSampler`

**Files:**
- Create: `track_gen/_src/checkpoints.py`
- Create: `track_gen/checkpoints.py`
- Modify: `track_gen/__init__.py` (import + `__all__`), `tests/test_public_api.py` (add `"checkpoints"`)
- Test: `tests/test_checkpoints.py`

**Interfaces:**
- Consumes: `_scan_boundary_k` from `track_gen._src.props` (generic closed-polyline scan: `[points, count, n_max, spacing, max_props, cum, out_count, out_step, out_truncated]`); `_safe_normalize2` from `collision_geom`; `Track`, `GateSequence` from `types`; fixtures `make_annulus_track`, `annulus_polylines`.
- Produces (Tasks 3–10 rely on):
  - `CheckpointSet` dataclass: `position, left, right, tangent` (`[E*M]` vec2f), `count` (`[E]` int32); `clone() -> CheckpointSet`; `classmethod from_gates(seq: GateSequence) -> CheckpointSet` (pure aliasing, no copy).
  - `CheckpointSampler(track, spacing, max_checkpoints=None)`: `.sample() -> CheckpointSet` (same instance each call, refreshed in place); attrs `_M`, `_set`, `truncated` (`[E]` int32 buffer), `step` (`[E]` float32 buffer); module flag `_CAPTURING`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_checkpoints.py`:

```python
"""Analytic annulus + aliasing tests for track_gen.checkpoints."""
from __future__ import annotations

import numpy as np
import pytest

from tests._collision_fixtures import annulus_polylines, make_annulus_track

N = 512
N_MAX = N + 8
RC, RI, RO = 1.0, 0.7, 1.3  # annulus fixture center/inner/outer radii


def _center_perimeter(track, e=0):
    m = int(track.count.numpy()[e])
    center = track.center.numpy().reshape(-1, N_MAX, 2)[e, :m]
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    return float(seg.sum())


def test_import_surface():
    import track_gen
    from track_gen.checkpoints import CheckpointSampler, CheckpointSet  # noqa: F401
    assert "checkpoints" in track_gen.__all__


def test_sampler_snap_count_and_positions_on_centerline():
    from track_gen.checkpoints import CheckpointSampler
    track = make_annulus_track(E=1, n=N)
    spacing = 0.8  # coarse: gate-like checkpoint counts
    sampler = CheckpointSampler(track, spacing=spacing)
    cps = sampler.sample()
    perim = _center_perimeter(track)
    n = int(cps.count.numpy()[0])
    assert n == int(round(perim / spacing))
    np.testing.assert_allclose(sampler.step.numpy()[0], perim / n, rtol=1e-5)
    pos = cps.position.numpy().reshape(-1, 2)[:n]
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), RC, atol=2e-3)


def test_crossing_segments_are_road_cross_sections():
    from track_gen.checkpoints import CheckpointSampler
    track = make_annulus_track(E=1, n=N)
    cps = CheckpointSampler(track, spacing=0.8).sample()
    n = int(cps.count.numpy()[0])
    left = cps.left.numpy().reshape(-1, 2)[:n]
    right = cps.right.numpy().reshape(-1, 2)[:n]
    pos = cps.position.numpy().reshape(-1, 2)[:n]
    tang = cps.tangent.numpy().reshape(-1, 2)[:n]
    # left on the inner circle, right on the outer circle, radially aligned
    # with the checkpoint position (annulus: cross-sections are radial).
    np.testing.assert_allclose(np.linalg.norm(left, axis=1), RI, atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(right, axis=1), RO, atol=2e-3)
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    np.testing.assert_allclose(left, RI * radial, atol=5e-3)
    np.testing.assert_allclose(right, RO * radial, atol=5e-3)
    # tangent unit, perpendicular to radial (CCW travel direction).
    np.testing.assert_allclose(np.linalg.norm(tang, axis=1), 1.0, atol=1e-5)
    assert np.abs((tang * radial).sum(axis=1)).max() < 0.02


def test_nan_padding_and_degenerate_env():
    from track_gen.checkpoints import CheckpointSampler
    track = make_annulus_track(E=2, n=N, counts=[N, 2])
    sampler = CheckpointSampler(track, spacing=0.8, max_checkpoints=32)
    cps = sampler.sample()
    counts = cps.count.numpy()
    assert counts[0] > 0 and counts[1] == 0
    M = sampler._M
    pos = cps.position.numpy().reshape(-1, 2)
    assert np.all(np.isnan(pos[counts[0]:M]))       # env 0 tail
    assert np.all(np.isnan(pos[M:2 * M]))           # env 1: all NaN


def test_from_gates_is_zero_copy_alias():
    import warp as wp
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.checkpoints import CheckpointSet
    E = 2
    cfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.05)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=5, num_envs=E, device="cpu"))
    seq = gen.generate()
    cps = CheckpointSet.from_gates(seq)
    # Zero-copy: same underlying buffers, not copies.
    assert cps.position.ptr == seq.position.ptr
    assert cps.left.ptr == seq.left.ptr
    assert cps.right.ptr == seq.right.ptr
    assert cps.tangent.ptr == seq.tangent.ptr
    assert cps.count.ptr == seq.count.ptr
    # Mutating the gate buffer is visible through the set (aliasing contract).
    wp.copy(seq.position, wp.zeros_like(seq.position))
    assert float(np.nanmax(np.abs(cps.position.numpy()))) == 0.0


def test_sampler_validation_and_derivation():
    import warp as wp
    from track_gen.checkpoints import CheckpointSampler
    track = make_annulus_track(E=1, n=N)
    with pytest.raises(ValueError, match="spacing"):
        CheckpointSampler(track, spacing=0.0)
    with pytest.raises(ValueError, match="spacing"):
        CheckpointSampler(track, spacing=float("nan"))
    with pytest.raises(ValueError, match="max_checkpoints"):
        CheckpointSampler(track, spacing=0.8, max_checkpoints=2)
    perim = _center_perimeter(track)
    sampler = CheckpointSampler(track, spacing=0.8)
    assert sampler._M == max(3, int(np.ceil(1.5 * perim / 0.8)))
    wp.copy(track.valid, wp.zeros(1, dtype=wp.int32, device="cpu"))
    with pytest.raises(ValueError, match="max_checkpoints"):
        CheckpointSampler(track, spacing=0.8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_checkpoints.py -v`
Expected: FAIL — `ModuleNotFoundError: track_gen.checkpoints`

- [ ] **Step 3: Implement `track_gen/_src/checkpoints.py`**

```python
"""Ordered course checkpoints from gates (zero-copy) or track centerlines.

``CheckpointSet`` is the shared consumer contract for course-following
utilities (``track_gen.progress``): per env an ordered list of checkpoints,
each with a center ``position``, a physical crossing segment ``left <->
right``, and a unit forward ``tangent``.

Two producers:

- ``CheckpointSet.from_gates(seq)`` aliases a ``GateSequence``'s buffers
  (gates already carry exactly these fields). Zero-copy: regenerated gates
  are seen automatically. With ``gate_width=0`` the crossing segment is
  degenerate (``left == right``) and pass-through can never trigger.
- ``CheckpointSampler`` resamples the track CENTERLINE at a coarse user
  spacing (snap rule as in ``track_gen.props``); because ``Track`` polylines
  are index-aligned, each checkpoint's crossing segment is the road
  cross-section: ``left``/``right`` are ``inner``/``outer`` interpolated at
  the same centerline segment and parameter.

Results are undefined for envs with ``valid[e] == 0``; callers gate on
``Track.valid`` / ``GateSequence.valid`` as everywhere in the library.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .collision_geom import _safe_normalize2
from .props import _scan_boundary_k
from .types import GateSequence, Track

_INITED = False
_CAPTURING = False


def _init() -> None:
    """Initialize Warp once (idempotent). Must run before any wp.launch."""
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _sync(device) -> None:
    if _CAPTURING:
        return
    if "cuda" in str(device):
        wp.synchronize()


@dataclass
class CheckpointSet:
    """Ordered per-env checkpoints, flat ``[E * M]`` per field.

    The consumer contract for :class:`track_gen.progress.ProgressTracker`.
    Sets produced by :meth:`from_gates` ALIAS the gate buffers (mutations to
    the gates are visible here); sets produced by ``CheckpointSampler`` are
    owned by the sampler and refreshed in place by ``sample()``.

    Attributes
    ----------
    position : wp.array
        ``vec2f`` checkpoint centers. NaN past ``count[e]``.
    left : wp.array
        ``vec2f`` crossing-segment endpoints (gate left / track inner).
    right : wp.array
        ``vec2f`` crossing-segment endpoints (gate right / track outer).
    tangent : wp.array
        ``vec2f`` unit forward (travel) directions.
    count : wp.array
        ``[E]`` ``int32`` real checkpoint counts. Meaningful only for envs
        with ``valid[e] == 1`` on the source batch.
    """

    position: wp.array
    left: wp.array
    right: wp.array
    tangent: wp.array
    count: wp.array

    @classmethod
    def from_gates(cls, seq: GateSequence) -> "CheckpointSet":
        """Zero-copy view of a :class:`GateSequence` as checkpoints.

        Aliases (does not copy) ``position``, ``left``, ``right``,
        ``tangent``, and ``count``. Progress state bound to this set must be
        reset after the gates are regenerated.
        """
        return cls(position=seq.position, left=seq.left, right=seq.right,
                   tangent=seq.tangent, count=seq.count)

    def clone(self) -> "CheckpointSet":
        """Return a deep copy whose Warp buffers do not alias this set."""
        return CheckpointSet(
            position=wp.clone(self.position),
            left=wp.clone(self.left),
            right=wp.clone(self.right),
            tangent=wp.clone(self.tangent),
            count=wp.clone(self.count),
        )


@wp.func
def _seg_at_arc(cum: wp.array(dtype=wp.float32), base: int, m: int,
                perim: float, s: float) -> wp.vec2f:
    """(segment index as float, lerp parameter t) at arc position s.

    Binary search for the largest i in [0, m-1] with cum[base+i] <= s; the
    closing segment (i == m-1) ends at arc length ``perim``.
    """
    lo = int(0)
    hi = m - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cum[base + mid] <= s:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    seg_end = perim
    if i + 1 < m:
        seg_end = cum[base + i + 1]
    seg_start = cum[base + i]
    denom = seg_end - seg_start
    t = 0.0
    if denom > 1.0e-12:
        t = wp.clamp((s - seg_start) / denom, 0.0, 1.0)
    return wp.vec2f(float(i), t)


@wp.kernel
def _place_checkpoints_k(
    center: wp.array(dtype=wp.vec2f),
    inner: wp.array(dtype=wp.vec2f),
    outer: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    cum: wp.array(dtype=wp.float32),
    cp_count: wp.array(dtype=wp.int32),
    cp_step: wp.array(dtype=wp.float32),
    max_cp: int,
    out_position: wp.array(dtype=wp.vec2f),
    out_left: wp.array(dtype=wp.vec2f),
    out_right: wp.array(dtype=wp.vec2f),
    out_tangent: wp.array(dtype=wp.vec2f),
):
    tid = wp.tid()
    e = tid // max_cp
    k = tid - e * max_cp

    n = cp_count[e]
    if k >= n:
        nan2 = wp.vec2f(wp.nan, wp.nan)
        out_position[tid] = nan2
        out_left[tid] = nan2
        out_right[tid] = nan2
        out_tangent[tid] = nan2
        return

    m = count[e]
    if m > n_max:
        m = n_max
    base = e * n_max
    step = cp_step[e]
    perim = step * float(n)

    it = _seg_at_arc(cum, base, m, perim, float(k) * step)
    i = int(it[0])
    t = it[1]
    j = i + 1
    if j == m:
        j = 0
    # Index-aligned lerp: the same segment/parameter on all three polylines
    # gives the road cross-section at this checkpoint.
    out_position[tid] = center[base + i] + (center[base + j] - center[base + i]) * t
    out_left[tid] = inner[base + i] + (inner[base + j] - inner[base + i]) * t
    out_right[tid] = outer[base + i] + (outer[base + j] - outer[base + i]) * t
    out_tangent[tid] = _safe_normalize2(center[base + j] - center[base + i])


class CheckpointSampler:
    """Resample a track's centerline into coarse course checkpoints.

    One sampler binds one :class:`Track` and one spacing; ``sample()``
    refreshes the owned :class:`CheckpointSet` in place from the CURRENT
    track batch (no rebind after ``generate()``; two kernel launches,
    allocation-free, CUDA-graph capturable under the module ``_CAPTURING``
    flag). Spacing is expected to be coarse — checkpoint counts similar to
    gate counts; nothing enforces this. Progress state bound to the sampled
    set must be reset after resampling a regenerated track.

    Producer diagnostics live on the sampler (not the set): ``truncated``
    (``[E]`` int32, 1 if ``max_checkpoints`` clipped the ring) and ``step``
    (``[E]`` float32 effective arc spacing).
    """

    def __init__(self, track: Track, spacing: float,
                 max_checkpoints: "int | None" = None) -> None:
        _init()
        if not (float(spacing) > 0.0):
            raise ValueError(f"spacing must be > 0, got {spacing!r}")
        if max_checkpoints is not None and int(max_checkpoints) < 3:
            raise ValueError(
                f"max_checkpoints must be >= 3 (or None for auto), got "
                f"{max_checkpoints!r}")
        E = int(track.count.shape[0])
        stride = int(track.outer.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"track batch layout invalid: outer has {stride} slots for {E} envs")
        self._track = track
        self._spacing = float(spacing)
        self._E = E
        self._n_max = stride // E
        self._device = str(track.outer.device)

        if max_checkpoints is None:
            max_checkpoints = self._derive_max_checkpoints()
        self._M = int(max_checkpoints)

        dev = self._device
        n = E * self._M
        self._cum = wp.zeros(E * self._n_max, dtype=wp.float32, device=dev)
        self.truncated = wp.zeros(E, dtype=wp.int32, device=dev)
        self.step = wp.zeros(E, dtype=wp.float32, device=dev)
        self._set = CheckpointSet(
            position=wp.zeros(n, dtype=wp.vec2f, device=dev),
            left=wp.zeros(n, dtype=wp.vec2f, device=dev),
            right=wp.zeros(n, dtype=wp.vec2f, device=dev),
            tangent=wp.zeros(n, dtype=wp.vec2f, device=dev),
            count=wp.zeros(E, dtype=wp.int32, device=dev),
        )

    def _derive_max_checkpoints(self) -> int:
        """ceil(1.5 * max valid-env CENTERLINE perimeter / spacing), floor 3.

        Host-side readback, construction time only.
        """
        E, n_max = self._E, self._n_max
        pts = self._track.center.numpy().reshape(E, n_max, 2)
        counts = self._track.count.numpy()
        valid = self._track.valid.numpy()
        best = 0.0
        for e in range(E):
            m = int(min(counts[e], n_max))
            if valid[e] == 0 or m < 3:
                continue
            poly = pts[e, :m]
            seg = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
            best = max(best, float(seg.sum()))
        if best <= 0.0:
            raise ValueError(
                "max_checkpoints=None needs at least one valid env in the "
                "bound track batch to derive a buffer size; pass "
                "max_checkpoints explicitly")
        return max(3, int(np.ceil(1.5 * best / self._spacing)))

    def sample(self) -> CheckpointSet:
        """Refresh the owned CheckpointSet from the bound Track; returns it."""
        t = self._track
        s = self._set
        wp.launch(
            _scan_boundary_k, dim=self._E,
            inputs=[t.center, t.count, self._n_max, self._spacing, self._M,
                    self._cum, s.count, self.step, self.truncated],
            device=self._device,
        )
        wp.launch(
            _place_checkpoints_k, dim=self._E * self._M,
            inputs=[t.center, t.inner, t.outer, t.count, self._n_max,
                    self._cum, s.count, self.step, self._M,
                    s.position, s.left, s.right, s.tangent],
            device=self._device,
        )
        _sync(self._device)
        return s
```

- [ ] **Step 4: Public shim + wiring**

Create `track_gen/checkpoints.py`:

```python
"""Public checkpoint API: ordered course goals from gates or tracks.

``CheckpointSet`` is the shared contract consumed by
``track_gen.progress.ProgressTracker``. Build one zero-copy from a
``GateSequence`` (``CheckpointSet.from_gates``) or by subsampling a track's
centerline at a coarse spacing (``CheckpointSampler`` — each checkpoint's
crossing segment is the road cross-section, a "virtual gate").
"""
from ._src.checkpoints import CheckpointSampler, CheckpointSet

__all__ = ["CheckpointSampler", "CheckpointSet"]
```

Modify `track_gen/__init__.py`: add `from . import checkpoints` after `from . import props`; add `"checkpoints"` to `__all__` (after `"props"`). Modify `tests/test_public_api.py`: add `"checkpoints"` to the curated set.

- [ ] **Step 5: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_checkpoints.py tests/test_public_api.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/checkpoints.py track_gen/checkpoints.py track_gen/__init__.py tests/test_public_api.py tests/test_checkpoints.py
git commit -m "feat: track_gen.checkpoints — CheckpointSet (from_gates) + centerline CheckpointSampler"
```

---

### Task 3: Checkpoint oracle + contract tests on generated tracks

**Files:**
- Create: `tests/_checkpoints_oracle.py`
- Test: `tests/test_checkpoints_contract.py`

**Interfaces:**
- Consumes: Task 2 API; `tests/_props_oracle.py` conventions.
- Produces: `_checkpoints_oracle.sample_checkpoints(center, inner, outer, spacing, max_cp) -> dict` with keys `position, left, right, tangent` (`[n,2]` arrays), `n`, `step` — independent numpy reference (same-segment interpolation).

- [ ] **Step 1: Write the oracle**

Create `tests/_checkpoints_oracle.py`:

```python
"""Independent numpy reference for centerline checkpoint sampling."""
from __future__ import annotations

import numpy as np


def sample_checkpoints(center, inner, outer, spacing, max_cp):
    """Mirror of CheckpointSampler semantics on one env's real polylines."""
    center = np.asarray(center, np.float64)
    inner = np.asarray(inner, np.float64)
    outer = np.asarray(outer, np.float64)
    m = len(center)
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    perim = float(cum[-1])
    n = int(np.clip(round(perim / spacing), 3, max_cp))
    step = perim / n
    s = np.arange(n) * step
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, m - 1)
    t = ((s - cum[idx]) / np.maximum(seg[idx], 1e-12))[:, None]
    j = (idx + 1) % m
    d = center[j] - center[idx]
    return {
        "position": center[idx] + d * t,
        "left": inner[idx] + (inner[j] - inner[idx]) * t,
        "right": outer[idx] + (outer[j] - outer[idx]) * t,
        "tangent": d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None],
        "n": n, "step": step,
    }
```

- [ ] **Step 2: Write the contract tests**

Create `tests/test_checkpoints_contract.py`:

```python
"""CheckpointSampler contracts: reuse, aliasing, truncation, oracle."""
from __future__ import annotations

import numpy as np
import warp as wp

from tests._checkpoints_oracle import sample_checkpoints
from tests._collision_fixtures import make_annulus_track
from track_gen.checkpoints import CheckpointSampler

N = 512


def test_sample_returns_same_set_and_clone_detaches():
    track = make_annulus_track(E=1, n=N)
    sampler = CheckpointSampler(track, spacing=0.8)
    s1 = sampler.sample()
    snap = s1.clone()
    pos_before = snap.position.numpy().copy()
    bigger = make_annulus_track(E=1, n=N, r_center=2.0)
    wp.copy(track.center, bigger.center)
    wp.copy(track.inner, bigger.inner)
    wp.copy(track.outer, bigger.outer)
    s2 = sampler.sample()
    assert s2 is s1
    assert int(s1.count.numpy()[0]) > int(snap.count.numpy()[0])
    np.testing.assert_allclose(snap.position.numpy(), pos_before)


def test_truncation_flag():
    track = make_annulus_track(E=1, n=N)
    sampler = CheckpointSampler(track, spacing=0.2, max_checkpoints=8)
    cps = sampler.sample()
    assert int(cps.count.numpy()[0]) == 8
    assert int(sampler.truncated.numpy()[0]) == 1


def test_matches_oracle_on_generated_tracks():
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=11, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    counts = track.count.numpy()
    n_max = track.outer.shape[0] // E
    spacing = 0.5
    sampler = CheckpointSampler(track, spacing=spacing, max_checkpoints=64)
    cps = sampler.sample()
    M = sampler._M
    center = track.center.numpy().reshape(E, n_max, 2)
    inner = track.inner.numpy().reshape(E, n_max, 2)
    outer = track.outer.numpy().reshape(E, n_max, 2)
    checked = 0
    for e in range(E):
        if not valid[e]:
            continue
        m = int(counts[e])
        ref = sample_checkpoints(center[e, :m], inner[e, :m], outer[e, :m],
                                 spacing, 64)
        assert int(cps.count.numpy()[e]) == ref["n"]
        sl = slice(e * M, e * M + ref["n"])
        np.testing.assert_allclose(cps.position.numpy().reshape(-1, 2)[sl],
                                   ref["position"], atol=1e-4)
        np.testing.assert_allclose(cps.left.numpy().reshape(-1, 2)[sl],
                                   ref["left"], atol=1e-4)
        np.testing.assert_allclose(cps.right.numpy().reshape(-1, 2)[sl],
                                   ref["right"], atol=1e-4)
        np.testing.assert_allclose(cps.tangent.numpy().reshape(-1, 2)[sl],
                                   ref["tangent"], atol=1e-3)
        checked += 1
    assert checked > 0
```

- [ ] **Step 3: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_checkpoints_contract.py -v`
Expected: 3 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/_checkpoints_oracle.py tests/test_checkpoints_contract.py
git commit -m "test: CheckpointSampler contracts and numpy oracle on generated tracks"
```

---

### Task 4: `track_gen.progress` — `ProgressTracker` with binding, square-course analytic tests

**Files:**
- Create: `track_gen/_src/progress.py`
- Create: `track_gen/progress.py`
- Modify: `track_gen/__init__.py`, `tests/test_public_api.py` (add `"progress"`)
- Test: `tests/test_progress.py`

**Interfaces:**
- Consumes: `CheckpointSet` (Task 2), `_segs_cross`/`_is_nan2` (Task 1 / existing).
- Produces (Tasks 5, 8–10 rely on):
  - `ProgressEvents` dataclass, all `[E]`: `passed, checkpoint_passed, next_checkpoint, laps, progress, wrong_way, wrong_checkpoint` (int32), `dist_to_next` (float32); `clone()`.
  - `ProgressTracker(checkpoints: CheckpointSet, position: wp.array | None = None)`; `update(position=None) -> ProgressEvents`; `reset(mask: wp.array) -> None`; internals `_E`, `_M`, `_events`, `_prev_pos`, `_next`, `_laps`, `_progress`, `_bound_pos`; module `_CAPTURING`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_progress.py` (the square course is 4 radial gates on a ring; the agent walks the unit circle in 45° steps):

```python
"""Analytic course tests for track_gen.progress (hand-built 4-gate ring)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

wp.init()

E = 1
M = 4


def _ring_checkpoints(device="cpu"):
    """4 checkpoints at angles 0/90/180/270 deg; crossing segments radial
    [0.3, 1.3]; tangents CCW. Agent paths on the unit circle cross them."""
    from track_gen.checkpoints import CheckpointSet
    ang = np.deg2rad([0.0, 90.0, 180.0, 270.0])
    radial = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
    tang = np.stack([-np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)

    def v2(a):
        return wp.array(a, dtype=wp.vec2f, device=device)

    return CheckpointSet(
        position=v2(radial * 1.0),
        left=v2(radial * 0.3),
        right=v2(radial * 1.3),
        tangent=v2(tang),
        count=wp.array(np.array([M], np.int32), dtype=wp.int32, device=device),
    )


def _pos(deg):
    a = np.deg2rad(deg)
    return wp.array(np.array([[np.cos(a), np.sin(a)]], np.float32),
                    dtype=wp.vec2f, device="cpu")


def test_ccw_lap_event_trace():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    trace = []
    for k in range(10):  # angles -22.5 + 45k: crossings at k = 1,3,5,7,9
        ev = tracker.update(_pos(-22.5 + 45.0 * k))
        trace.append((int(ev.passed.numpy()[0]), int(ev.next_checkpoint.numpy()[0]),
                      int(ev.laps.numpy()[0]), int(ev.progress.numpy()[0])))
    assert trace == [
        (0, 0, 0, 0),  # first update: init only
        (1, 1, 0, 1),  # crossed gate 0
        (0, 1, 0, 1),
        (1, 2, 0, 2),  # gate 1
        (0, 2, 0, 2),
        (1, 3, 0, 3),  # gate 2
        (0, 3, 0, 3),
        (1, 0, 1, 4),  # gate 3 -> lap complete
        (0, 0, 1, 4),
        (1, 1, 1, 5),  # gate 0 again on lap 2
    ]


def test_dist_to_next_matches_geometry():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    tracker.update(_pos(-22.5))
    ev = tracker.update(_pos(22.5))   # passed gate 0, next = gate 1 at (0,1)
    p = np.array([np.cos(np.deg2rad(22.5)), np.sin(np.deg2rad(22.5))])
    expected = np.linalg.norm(p - np.array([0.0, 1.0]))
    np.testing.assert_allclose(float(ev.dist_to_next.numpy()[0]), expected,
                               rtol=1e-5)


def test_wrong_way_and_wrong_checkpoint():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    tracker.update(_pos(22.5))
    ev = tracker.update(_pos(-22.5))  # backward through gate 0
    assert int(ev.wrong_way.numpy()[0]) == 1
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.next_checkpoint.numpy()[0]) == 0  # no advance

    tracker2 = ProgressTracker(_ring_checkpoints())
    tracker2.update(_pos(100.0))
    ev = tracker2.update(_pos(170.0))  # crosses gate 1 (90 deg), target is 0
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == 1


def test_double_jump_advances_one_and_flags_second():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    tracker.update(_pos(-10.0))
    ev = tracker.update(_pos(100.0))  # one step across gates 0 AND 1
    assert int(ev.passed.numpy()[0]) == 1
    assert int(ev.checkpoint_passed.numpy()[0]) == 0
    assert int(ev.next_checkpoint.numpy()[0]) == 1
    assert int(ev.wrong_checkpoint.numpy()[0]) == 1  # the skipped gate 1


def test_reset_mask_no_spurious_crossing():
    from track_gen.progress import ProgressTracker
    tracker = ProgressTracker(_ring_checkpoints())
    tracker.update(_pos(-22.5))
    tracker.update(_pos(22.5))
    assert int(tracker._progress.numpy()[0]) == 1
    mask = wp.array(np.array([1], np.int32), dtype=wp.int32, device="cpu")
    tracker.reset(mask)
    # Teleport across the whole course: first post-reset update is inert.
    ev = tracker.update(_pos(200.0))
    assert int(ev.passed.numpy()[0]) == 0
    assert int(ev.wrong_checkpoint.numpy()[0]) == -1
    assert int(ev.next_checkpoint.numpy()[0]) == 0
    assert int(ev.laps.numpy()[0]) == 0
    assert int(ev.progress.numpy()[0]) == 0


def test_bound_mode_equivalence_and_errors():
    from track_gen.progress import ProgressTracker
    buf = wp.zeros(E, dtype=wp.vec2f, device="cpu")
    bound = ProgressTracker(_ring_checkpoints(), position=buf)
    free = ProgressTracker(_ring_checkpoints())
    with pytest.raises(ValueError, match="bound"):
        bound.update(_pos(0.0))       # arg while bound
    with pytest.raises(ValueError, match="position"):
        free.update()                 # no-arg while unbound
    for k in range(6):
        p = _pos(-22.5 + 45.0 * k)
        wp.copy(buf, p)
        ev_b = bound.update()
        ev_f = free.update(p)
        assert int(ev_b.passed.numpy()[0]) == int(ev_f.passed.numpy()[0])
        assert int(ev_b.next_checkpoint.numpy()[0]) == int(ev_f.next_checkpoint.numpy()[0])
        np.testing.assert_allclose(ev_b.dist_to_next.numpy(),
                                   ev_f.dist_to_next.numpy(), rtol=1e-6)


def test_import_surface():
    import track_gen
    from track_gen.progress import ProgressEvents, ProgressTracker  # noqa: F401
    assert "progress" in track_gen.__all__
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_progress.py -v`
Expected: FAIL — `ModuleNotFoundError: track_gen.progress`

- [ ] **Step 3: Implement `track_gen/_src/progress.py`**

```python
"""Stateful course-progress tracking over a CheckpointSet (gates or track).

``ProgressTracker`` owns per-env device state (previous position, next
checkpoint, laps, total progress) and advances it in ONE fused kernel per
``update()``: swept-segment pass-through detection against the target's
crossing segment, wrong-way and wrong-checkpoint events, and the distance to
the next checkpoint center (``dist_to_next``) for delta-distance rewards
(``r_t = dist[t-1] - dist[t]``, differenced by the caller).

The tracker can LATCH onto a stable, user-owned position buffer at
construction (``position=...``): ``update()`` then takes no arguments and
reads the buffer in place — the natural CUDA-graph pattern (sim writes
poses, replays the captured update). All tracker-owned buffers are
preallocated with stable pointers.

Reset contract: ``reset(mask)`` clears state where ``mask[e] == 1`` and arms
the NaN previous-position sentinel, so the first update after a reset (or
after construction) can never emit a spurious crossing. Callers MUST reset
after regenerating the bound course. Results are undefined for envs with
``valid[e] == 0`` on the source batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .checkpoints import CheckpointSet
from .collision_geom import _is_nan2, _segs_cross

_INITED = False
_CAPTURING = False


def _init() -> None:
    """Initialize Warp once (idempotent). Must run before any wp.launch."""
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _sync(device) -> None:
    if _CAPTURING:
        return
    if "cuda" in str(device):
        wp.synchronize()


@dataclass
class ProgressEvents:
    """Per-env progress events, all ``[E]``; overwritten in place per update.

    .. warning::

        ``ProgressTracker.update()`` returns the SAME instance every call.
        ``clone()`` for snapshots.

    Attributes
    ----------
    passed : wp.array
        ``int32`` — 1 iff the target checkpoint was crossed forward this step.
    checkpoint_passed : wp.array
        ``int32`` — index of the checkpoint passed this step, -1 otherwise.
    next_checkpoint : wp.array
        ``int32`` — current target AFTER any advance this step.
    laps : wp.array
        ``int32`` — completed laps.
    progress : wp.array
        ``int32`` — total checkpoints passed since construction/reset.
    wrong_way : wp.array
        ``int32`` — 1 iff the target was crossed BACKWARD this step.
    wrong_checkpoint : wp.array
        ``int32`` — index of a non-target checkpoint crossed this step
        (either direction; first in index order), -1 otherwise.
    dist_to_next : wp.array
        ``float32`` — |position - next checkpoint center| after any advance.
        NaN for envs with no checkpoints.
    """

    passed: wp.array
    checkpoint_passed: wp.array
    next_checkpoint: wp.array
    laps: wp.array
    progress: wp.array
    wrong_way: wp.array
    wrong_checkpoint: wp.array
    dist_to_next: wp.array

    def clone(self) -> "ProgressEvents":
        """Return a deep copy whose Warp buffers do not alias this result."""
        return ProgressEvents(
            passed=wp.clone(self.passed),
            checkpoint_passed=wp.clone(self.checkpoint_passed),
            next_checkpoint=wp.clone(self.next_checkpoint),
            laps=wp.clone(self.laps),
            progress=wp.clone(self.progress),
            wrong_way=wp.clone(self.wrong_way),
            wrong_checkpoint=wp.clone(self.wrong_checkpoint),
            dist_to_next=wp.clone(self.dist_to_next),
        )


@wp.kernel
def _progress_update_k(
    cp_position: wp.array(dtype=wp.vec2f),
    cp_left: wp.array(dtype=wp.vec2f),
    cp_right: wp.array(dtype=wp.vec2f),
    cp_tangent: wp.array(dtype=wp.vec2f),
    cp_count: wp.array(dtype=wp.int32),
    max_cp: int,
    position: wp.array(dtype=wp.vec2f),
    prev_pos: wp.array(dtype=wp.vec2f),
    next_cp: wp.array(dtype=wp.int32),
    laps: wp.array(dtype=wp.int32),
    progress: wp.array(dtype=wp.int32),
    out_passed: wp.array(dtype=wp.int32),
    out_cp_passed: wp.array(dtype=wp.int32),
    out_next: wp.array(dtype=wp.int32),
    out_laps: wp.array(dtype=wp.int32),
    out_progress: wp.array(dtype=wp.int32),
    out_wrong_way: wp.array(dtype=wp.int32),
    out_wrong_cp: wp.array(dtype=wp.int32),
    out_dist: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    pos = position[e]
    n = cp_count[e]
    base = e * max_cp

    passed = int(0)
    cp_passed = int(-1)
    wway = int(0)
    wcp = int(-1)

    if n < 1:
        prev_pos[e] = pos
        out_passed[e] = 0
        out_cp_passed[e] = -1
        out_next[e] = next_cp[e]
        out_laps[e] = laps[e]
        out_progress[e] = progress[e]
        out_wrong_way[e] = 0
        out_wrong_cp[e] = -1
        out_dist[e] = wp.nan
        return

    g = next_cp[e]
    if g >= n or g < 0:
        # Defensive clamp: course regenerated without reset (documented as
        # caller error, but never index out of the real range).
        g = 0

    prev = prev_pos[e]
    if _is_nan2(prev) == 0:
        move = pos - prev
        if _segs_cross(prev, pos, cp_left[base + g], cp_right[base + g]) == 1:
            if wp.dot(move, cp_tangent[base + g]) > 0.0:
                passed = int(1)
                cp_passed = g
            else:
                wway = int(1)
        # Wrong-checkpoint scan vs the ORIGINAL target g: a double-jump
        # advances g and flags the second crossing in this same update.
        for i in range(n):
            if i != g and wcp == -1:
                if _segs_cross(prev, pos, cp_left[base + i], cp_right[base + i]) == 1:
                    wcp = i

    ng = g
    lp = laps[e]
    pr = progress[e]
    if passed == 1:
        ng = g + 1
        pr = pr + 1
        if ng == n:
            ng = 0
            lp = lp + 1

    prev_pos[e] = pos
    next_cp[e] = ng
    laps[e] = lp
    progress[e] = pr

    out_passed[e] = passed
    out_cp_passed[e] = cp_passed
    out_next[e] = ng
    out_laps[e] = lp
    out_progress[e] = pr
    out_wrong_way[e] = wway
    out_wrong_cp[e] = wcp
    out_dist[e] = wp.length(cp_position[base + ng] - pos)


@wp.kernel
def _progress_reset_k(
    mask: wp.array(dtype=wp.int32),
    prev_pos: wp.array(dtype=wp.vec2f),
    next_cp: wp.array(dtype=wp.int32),
    laps: wp.array(dtype=wp.int32),
    progress: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    if mask[e] != 0:
        prev_pos[e] = wp.vec2f(wp.nan, wp.nan)
        next_cp[e] = 0
        laps[e] = 0
        progress[e] = 0


class ProgressTracker:
    """Track ordered progress of one agent per env through a CheckpointSet.

    See the module docstring for semantics. Construct with ``position=`` (a
    stable ``[E]`` vec2f wp.array owned by your sim) for bound mode —
    ``update()`` then takes no arguments and reads the buffer in place; or
    leave unbound and pass positions per call. Mixing modes raises
    ``ValueError``.
    """

    def __init__(self, checkpoints: CheckpointSet,
                 position: "wp.array | None" = None) -> None:
        _init()
        E = int(checkpoints.count.shape[0])
        stride = int(checkpoints.position.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"checkpoint layout invalid: {stride} slots for {E} envs")
        self._cps = checkpoints
        self._E = E
        self._M = stride // E
        self._device = str(checkpoints.position.device)

        self._bound_pos: "wp.array | None" = None
        if position is not None:
            self._validate_position(position)
            self._bound_pos = position

        dev = self._device
        self._prev_pos = wp.array(np.full((E, 2), np.nan, np.float32),
                                  dtype=wp.vec2f, device=dev)
        self._next = wp.zeros(E, dtype=wp.int32, device=dev)
        self._laps = wp.zeros(E, dtype=wp.int32, device=dev)
        self._progress = wp.zeros(E, dtype=wp.int32, device=dev)
        self._events = ProgressEvents(
            passed=wp.zeros(E, dtype=wp.int32, device=dev),
            checkpoint_passed=wp.zeros(E, dtype=wp.int32, device=dev),
            next_checkpoint=wp.zeros(E, dtype=wp.int32, device=dev),
            laps=wp.zeros(E, dtype=wp.int32, device=dev),
            progress=wp.zeros(E, dtype=wp.int32, device=dev),
            wrong_way=wp.zeros(E, dtype=wp.int32, device=dev),
            wrong_checkpoint=wp.zeros(E, dtype=wp.int32, device=dev),
            dist_to_next=wp.zeros(E, dtype=wp.float32, device=dev),
        )

    def _validate_position(self, position) -> None:
        if not isinstance(position, wp.array):
            raise ValueError(f"position must be a wp.array, got {type(position)!r}")
        if position.shape != (self._E,):
            raise ValueError(
                f"position must have shape ({self._E},), got {position.shape}")
        if position.dtype is not wp.vec2f:
            raise ValueError(
                f"position must have dtype vec2f, got {position.dtype.__name__}")
        if str(position.device) != self._device:
            raise ValueError(
                f"position is on {position.device}, tracker is on {self._device}")

    def update(self, position: "wp.array | None" = None) -> ProgressEvents:
        """Advance one step; returns the tracker's preallocated events.

        Bound mode (constructed with ``position=``): call with no arguments.
        Per-call mode: pass the ``[E]`` vec2f position array — the SAME
        array (identical ``.ptr``) must be used across a CUDA-graph capture
        and its replays.
        """
        if self._bound_pos is not None:
            if position is not None:
                raise ValueError(
                    "tracker is bound to a position buffer; call update() "
                    "with no arguments")
            pos = self._bound_pos
        else:
            if position is None:
                raise ValueError(
                    "tracker is not bound; pass position to update() or "
                    "construct with position=")
            self._validate_position(position)
            pos = position
        c = self._cps
        ev = self._events
        wp.launch(
            _progress_update_k, dim=self._E,
            inputs=[c.position, c.left, c.right, c.tangent, c.count, self._M,
                    pos, self._prev_pos, self._next, self._laps, self._progress,
                    ev.passed, ev.checkpoint_passed, ev.next_checkpoint,
                    ev.laps, ev.progress, ev.wrong_way, ev.wrong_checkpoint,
                    ev.dist_to_next],
            device=self._device,
        )
        _sync(self._device)
        return ev

    def reset(self, mask: wp.array) -> None:
        """Clear state where ``mask[e] == 1`` (``[E]`` int32); arms the NaN
        previous-position sentinel so the next update cannot emit a spurious
        crossing. Required after regenerating the bound course."""
        if not isinstance(mask, wp.array) or mask.shape != (self._E,) \
                or mask.dtype is not wp.int32:
            raise ValueError(
                f"mask must be a [{self._E}] int32 wp.array")
        wp.launch(
            _progress_reset_k, dim=self._E,
            inputs=[mask, self._prev_pos, self._next, self._laps, self._progress],
            device=self._device,
        )
        _sync(self._device)
```

- [ ] **Step 4: Public shim + wiring**

Create `track_gen/progress.py`:

```python
"""Public progress-tracking API: ordered course progress + reward signals.

``ProgressTracker`` consumes any ``CheckpointSet`` (gate sequences via
``CheckpointSet.from_gates``, or subsampled track centerlines via
``CheckpointSampler``) and emits per-step ``ProgressEvents``: pass events,
laps, wrong-way / wrong-checkpoint flags, and ``dist_to_next`` — difference
it across steps for the classic negative-delta-distance reward.
"""
from ._src.progress import ProgressEvents, ProgressTracker

__all__ = ["ProgressEvents", "ProgressTracker"]
```

Modify `track_gen/__init__.py`: `from . import progress` (after `checkpoints`), add `"progress"` to `__all__`. Modify `tests/test_public_api.py`: add `"progress"`.

- [ ] **Step 5: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_progress.py tests/test_public_api.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/progress.py track_gen/progress.py track_gen/__init__.py tests/test_public_api.py tests/test_progress.py
git commit -m "feat: track_gen.progress — stateful ProgressTracker with stable input binding"
```

---

### Task 5: Progress oracle — random walks on generated gates AND track checkpoints

**Files:**
- Create: `tests/_progress_oracle.py`
- Test: `tests/test_progress_contract.py`

**Interfaces:**
- Consumes: Tasks 2–4 APIs.
- Produces: `_progress_oracle.ProgressOracle(positions, lefts, rights, tangents)` — stateful numpy mirror with `.update(pos) -> dict` (keys matching ProgressEvents field names) and `.reset()`.

- [ ] **Step 1: Write the oracle**

Create `tests/_progress_oracle.py`:

```python
"""Independent numpy mirror of ProgressTracker semantics (tests only)."""
from __future__ import annotations

import numpy as np


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def segs_cross(a, b, c, d):
    """Strict proper intersection of segments ab and cd."""
    ab, cd = b - a, d - c
    o1, o2 = _cross(ab, c - a), _cross(ab, d - a)
    o3, o4 = _cross(cd, a - c), _cross(cd, b - c)
    return (((o1 > 0) and (o2 < 0)) or ((o1 < 0) and (o2 > 0))) and \
           (((o3 > 0) and (o4 < 0)) or ((o3 < 0) and (o4 > 0)))


class ProgressOracle:
    """One env's worth of checkpoints; mirrors the kernel's update order."""

    def __init__(self, positions, lefts, rights, tangents):
        self.p = np.asarray(positions, float)
        self.l = np.asarray(lefts, float)
        self.r = np.asarray(rights, float)
        self.t = np.asarray(tangents, float)
        self.n = len(self.p)
        self.reset()

    def reset(self):
        self.prev = None
        self.next = 0
        self.laps = 0
        self.progress = 0

    def update(self, pos):
        pos = np.asarray(pos, float)
        ev = {"passed": 0, "checkpoint_passed": -1, "wrong_way": 0,
              "wrong_checkpoint": -1}
        g = self.next
        if self.prev is not None and self.n >= 1:
            if segs_cross(self.prev, pos, self.l[g], self.r[g]):
                if np.dot(pos - self.prev, self.t[g]) > 0:
                    ev["passed"] = 1
                    ev["checkpoint_passed"] = g
                else:
                    ev["wrong_way"] = 1
            for i in range(self.n):
                if i != g and segs_cross(self.prev, pos, self.l[i], self.r[i]):
                    ev["wrong_checkpoint"] = i
                    break
        if ev["passed"]:
            self.next = (g + 1) % self.n
            self.progress += 1
            if self.next == 0:
                self.laps += 1
        self.prev = pos
        ev["next_checkpoint"] = self.next
        ev["laps"] = self.laps
        ev["progress"] = self.progress
        ev["dist_to_next"] = float(np.linalg.norm(self.p[self.next] - pos))
        return ev
```

- [ ] **Step 2: Write the property tests**

Create `tests/test_progress_contract.py`:

```python
"""ProgressTracker vs numpy oracle on generated gates AND track checkpoints."""
from __future__ import annotations

import numpy as np
import warp as wp

from tests._progress_oracle import ProgressOracle
from track_gen.progress import ProgressTracker

STEPS = 60
FIELDS = ("passed", "checkpoint_passed", "next_checkpoint", "laps",
          "progress", "wrong_way", "wrong_checkpoint")


def _run_and_compare(cps, E, M, centers, rng):
    """Random-walk positions around each env's course; compare every field."""
    tracker = ProgressTracker(cps)
    counts = cps.count.numpy()
    pos_np = cps.position.numpy().reshape(E, M, 2)
    left_np = cps.left.numpy().reshape(E, M, 2)
    right_np = cps.right.numpy().reshape(E, M, 2)
    tang_np = cps.tangent.numpy().reshape(E, M, 2)
    oracles = {}
    for e in range(E):
        n = int(counts[e])
        if n >= 1:
            oracles[e] = ProgressOracle(pos_np[e, :n], left_np[e, :n],
                                        right_np[e, :n], tang_np[e, :n])
    walk = centers + rng.normal(0.0, 0.35, (STEPS, E, 2))
    for s in range(STEPS):
        p = wp.array(walk[s].astype(np.float32), dtype=wp.vec2f, device="cpu")
        ev = tracker.update(p)
        got = {f: getattr(ev, f).numpy() for f in FIELDS}
        dist = ev.dist_to_next.numpy()
        for e, oracle in oracles.items():
            ref = oracle.update(walk[s, e])
            for f in FIELDS:
                assert got[f][e] == ref[f], f"step {s} env {e} field {f}"
            np.testing.assert_allclose(dist[e], ref["dist_to_next"], atol=1e-4)
    assert oracles, "no env with checkpoints"


def test_oracle_on_generated_gates():
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.checkpoints import CheckpointSet
    E = 4
    cfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.15)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=21, num_envs=E, device="cpu"))
    seq = gen.generate()
    cps = CheckpointSet.from_gates(seq)
    M = cps.position.shape[0] // E
    valid = seq.valid.numpy().astype(bool)
    counts = cps.count.numpy()
    pos = np.nan_to_num(cps.position.numpy().reshape(E, M, 2), nan=0.0)
    centers = np.zeros((E, 2))
    for e in range(E):
        if valid[e] and counts[e] > 0:
            centers[e] = pos[e, :counts[e]].mean(axis=0)
    # Only valid envs are compared (undefined otherwise): zero out others.
    cps2 = cps.clone()
    cnp = cps2.count.numpy()
    cnp[~valid] = 0
    wp.copy(cps2.count, wp.array(cnp, dtype=wp.int32, device="cpu"))
    _run_and_compare(cps2, E, M, centers, np.random.default_rng(1))


def test_oracle_on_track_checkpoints():
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    from track_gen.checkpoints import CheckpointSampler
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=31, num_envs=E, device="cpu"))
    track = gen.generate()
    sampler = CheckpointSampler(track, spacing=0.6, max_checkpoints=48)
    cps = sampler.sample()
    M = sampler._M
    valid = track.valid.numpy().astype(bool)
    counts = cps.count.numpy()
    pos = np.nan_to_num(cps.position.numpy().reshape(E, M, 2), nan=0.0)
    centers = np.zeros((E, 2))
    for e in range(E):
        if valid[e] and counts[e] > 0:
            centers[e] = pos[e, :counts[e]].mean(axis=0)
    cps2 = cps.clone()
    cnp = cps2.count.numpy()
    cnp[~valid] = 0
    wp.copy(cps2.count, wp.array(cnp, dtype=wp.int32, device="cpu"))
    _run_and_compare(cps2, E, M, centers, np.random.default_rng(2))
```

- [ ] **Step 3: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_progress_contract.py -v`
Expected: 2 PASS (60 steps × 4 envs × 2 sources, every event field compared)

- [ ] **Step 4: Commit**

```bash
git add tests/_progress_oracle.py tests/test_progress_contract.py
git commit -m "test: ProgressTracker oracle property tests on gates and track checkpoints"
```

---

### Task 6: `DiscChecker` — box-vs-disc obstacles in the collision family

**Files:**
- Create: `track_gen/_src/collision_discs.py`
- Modify: `track_gen/collision.py` (re-export `DiscChecker`, `DiscContact`)
- Test: `tests/test_collision_discs.py`

**Interfaces:**
- Consumes: `_is_nan2`, `_rot2`, `_point_to_local_box_dist`, `_safe_normalize2` from `collision_geom`; `_init`/`_sync`/`_CAPTURING` pattern.
- Produces:
  - `DiscContact` dataclass `[E*max_boxes]`: `hit` (int32 0/1), `disc` (int32 idx/-1), `depth` (float32 ≥ 0), `nearest` (vec2f point on the disc boundary); `clone()`.
  - `DiscChecker(discs, radius, max_boxes, num_envs=None, count=None, position=None, yaw=None, half_extents=None)`; `query(position=None, yaw=None, half_extents=None) -> DiscContact`; module `_CAPTURING` in `track_gen._src.collision_discs`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collision_discs.py`:

```python
"""Analytic + recipe tests for DiscChecker (box vs disc obstacles)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

wp.init()


def _boxes(E, B, slots, device="cpu"):
    pos = np.full((E * B, 2), np.nan, np.float32)
    yaw = np.zeros(E * B, np.float32)
    he = np.zeros((E * B, 2), np.float32)
    for (e, b), (px, py, yw, hx, hy) in slots.items():
        i = e * B + b
        pos[i] = (px, py)
        yaw[i] = yw
        he[i] = (hx, hy)
    return (wp.array(pos, dtype=wp.vec2f, device=device),
            wp.array(yaw, dtype=wp.float32, device=device),
            wp.array(he, dtype=wp.vec2f, device=device))


def _discs(rows, device="cpu"):
    return wp.array(np.array(rows, np.float32), dtype=wp.vec2f, device=device)


def test_face_corner_graze_and_miss():
    from track_gen.collision import DiscChecker
    # One env, 4 discs; one axis-aligned box he=(0.1, 0.05) at origin.
    discs = _discs([[0.12, 0.0],            # face hit: pen 0.03-0.02=0.01
                    [0.12, 0.07],           # corner hit: dist=sqrt(0.02^2+0.02^2)
                    [0.13, 0.0],            # graze: dist 0.03 == radius -> hit
                    [0.20, 0.0]])           # miss
    checker = DiscChecker(discs, radius=0.03, max_boxes=4, num_envs=1)
    pos, yaw, he = _boxes(1, 4, {(0, b): (0.0, 0.0, 0.0, 0.1, 0.05)
                                 for b in range(4)})
    # All four boxes identical; each box sees ALL discs, so instead probe
    # per-disc behavior with per-box positions FAR from other discs:
    pos, yaw, he = _boxes(1, 4, {
        (0, 0): (0.0, 0.0, 0.0, 0.1, 0.05),
    })
    res = checker.query(pos, yaw, he)
    hit = res.hit.numpy()
    assert hit[0] == 1
    # Deepest disc is the face one (pen 0.01 > corner pen ~0.0017).
    assert int(res.disc.numpy()[0]) == 0
    np.testing.assert_allclose(float(res.depth.numpy()[0]), 0.01, atol=1e-6)
    # Nearest point on disc 0's boundary toward the box face: (0.09, 0).
    np.testing.assert_allclose(res.nearest.numpy().reshape(-1, 2)[0],
                               [0.09, 0.0], atol=1e-6)
    # Inactive slots inert.
    assert list(hit[1:]) == [0, 0, 0]
    assert list(res.disc.numpy()[1:]) == [-1, -1, -1]


def test_graze_counts_as_hit_and_miss_does_not():
    from track_gen.collision import DiscChecker
    discs = _discs([[0.13, 0.0], [0.14, 0.0]])
    checker = DiscChecker(discs, radius=0.03, max_boxes=2, num_envs=1)
    pos, yaw, he = _boxes(1, 2, {(0, 0): (0.0, 0.0, 0.0, 0.1, 0.05)})
    res = checker.query(pos, yaw, he)
    # Box 0 sees disc 0 at exactly radius (hit, depth 0) and disc 1 beyond.
    assert int(res.hit.numpy()[0]) == 1
    assert int(res.disc.numpy()[0]) == 0
    np.testing.assert_allclose(float(res.depth.numpy()[0]), 0.0, atol=1e-6)


def test_rotated_box_and_nan_discs_skipped():
    from track_gen.collision import DiscChecker
    # Disc straight above; box rotated 90 deg so its LONG side faces up.
    discs = _discs([[np.nan, np.nan], [0.0, 0.12]])
    checker = DiscChecker(discs, radius=0.03, max_boxes=1, num_envs=1)
    pos, yaw, he = _boxes(1, 1, {(0, 0): (0.0, 0.0, np.pi / 2, 0.1, 0.05)})
    res = checker.query(pos, yaw, he)
    # Rotated: half-extent along +y is now 0.1 -> dist 0.02 -> pen 0.01.
    assert int(res.hit.numpy()[0]) == 1
    assert int(res.disc.numpy()[0]) == 1
    np.testing.assert_allclose(float(res.depth.numpy()[0]), 0.01, atol=1e-6)


def test_explicit_count_limits_scan():
    from track_gen.collision import DiscChecker
    discs = _discs([[0.12, 0.0], [0.0, 0.0]])  # second disc INSIDE the box
    count = wp.array(np.array([1], np.int32), dtype=wp.int32, device="cpu")
    checker = DiscChecker(discs, radius=0.03, max_boxes=1, count=count)
    pos, yaw, he = _boxes(1, 1, {(0, 0): (0.0, 0.0, 0.0, 0.1, 0.05)})
    res = checker.query(pos, yaw, he)
    assert int(res.disc.numpy()[0]) == 0  # disc 1 never scanned


def test_constructor_validation():
    from track_gen.collision import DiscChecker
    discs = _discs([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="num_envs"):
        DiscChecker(discs, radius=0.1, max_boxes=1)  # neither count nor num_envs
    with pytest.raises(ValueError, match="radius"):
        DiscChecker(discs, radius=0.0, max_boxes=1, num_envs=1)
    with pytest.raises(ValueError, match="radius"):
        DiscChecker(discs, radius=float("nan"), max_boxes=1, num_envs=1)
    with pytest.raises(ValueError, match="divisible"):
        DiscChecker(_discs([[0.0, 0.0]] * 3), radius=0.1, max_boxes=1, num_envs=2)


def test_gate_post_recipe():
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.collision import DiscChecker
    E = 2
    cfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.06)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=9, num_envs=E, device="cpu"))
    seq = gen.generate()
    G = seq.position.shape[0] // E
    left = seq.left.numpy().reshape(E, G, 2)
    right = seq.right.numpy().reshape(E, G, 2)
    posts = np.empty((E, 2 * G, 2), np.float32)
    posts[:, 0::2] = left
    posts[:, 1::2] = right
    posts_wp = wp.array(posts.reshape(-1, 2), dtype=wp.vec2f, device="cpu")
    checker = DiscChecker(posts_wp, radius=0.02, max_boxes=1, num_envs=E)
    # Park a box exactly on env 0's gate 0 LEFT post.
    valid = seq.valid.numpy()
    e = int(np.argmax(valid))
    slots = {(e, 0): (float(left[e, 0, 0]), float(left[e, 0, 1]), 0.0, 0.03, 0.03)}
    pos, yaw, he = _boxes(E, 1, slots)
    res = checker.query(pos, yaw, he)
    assert int(res.hit.numpy()[e]) == 1
    disc = int(res.disc.numpy()[e])
    assert disc % 2 == 0 and disc // 2 == 0  # left post of gate 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_discs.py -v`
Expected: FAIL — `ImportError: cannot import name 'DiscChecker'`

- [ ] **Step 3: Implement `track_gen/_src/collision_discs.py`**

```python
"""Box-vs-disc obstacle collision (gate posts, physical cones, point props).

``DiscChecker`` binds a flat ``[E * D]`` vec2f array of disc centers (ALIASED
— regenerated buffers are seen automatically) and a scalar radius, and
queries batches of oriented boxes exactly like ``CollisionChecker``: hit iff
the distance from the disc center to the solid OBB is <= radius. The deepest
penetrating disc is reported per box.

Disc validity: pass ``count`` (``[E]`` int32 real discs per env) OR rely on
NaN-marked padding (slots with NaN centers are skipped) — GateSequence
``left``/``right`` arrays interleaved as posts work out of the box.

Per-step inputs can be bound at construction (``position=``, ``yaw=``,
``half_extents=`` — all three or none): ``query()`` then takes no arguments
and reads the stable buffers in place (the CUDA-graph pattern).
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .collision_geom import (
    _is_nan2,
    _point_to_local_box_dist,
    _rot2,
    _safe_normalize2,
)

_INITED = False
_CAPTURING = False


def _init() -> None:
    """Initialize Warp once (idempotent). Must run before any wp.launch."""
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _sync(device) -> None:
    if _CAPTURING:
        return
    if "cuda" in str(device):
        wp.synchronize()


@dataclass
class DiscContact:
    """Batched box-vs-disc result, flat ``[E * max_boxes]`` per field.

    .. warning::

        ``DiscChecker.query()`` returns the SAME instance every call and
        overwrites its buffers in place; ``clone()`` for snapshots.

    Attributes
    ----------
    hit : wp.array
        ``int32`` — 1 iff any disc touches the box (distance <= radius).
    disc : wp.array
        ``int32`` — index of the deepest-penetrating disc, -1 when none.
    depth : wp.array
        ``float32`` — penetration depth of that disc (>= 0; 0 on graze or
        no hit).
    nearest : wp.array
        ``vec2f`` — point on that disc's boundary nearest the box
        (approximate when the disc center lies deep inside the box). NaN
        when no hit or the box slot is inactive.
    """

    hit: wp.array
    disc: wp.array
    depth: wp.array
    nearest: wp.array

    def clone(self) -> "DiscContact":
        """Return a deep copy whose Warp buffers do not alias this result."""
        return DiscContact(
            hit=wp.clone(self.hit),
            disc=wp.clone(self.disc),
            depth=wp.clone(self.depth),
            nearest=wp.clone(self.nearest),
        )


@wp.kernel
def _box_query_discs_k(
    discs: wp.array(dtype=wp.vec2f),
    disc_count: wp.array(dtype=wp.int32),
    d_max: int,
    radius: float,
    max_boxes: int,
    position: wp.array(dtype=wp.vec2f),
    yaw: wp.array(dtype=wp.float32),
    half_extents: wp.array(dtype=wp.vec2f),
    out_hit: wp.array(dtype=wp.int32),
    out_disc: wp.array(dtype=wp.int32),
    out_depth: wp.array(dtype=wp.float32),
    out_nearest: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    e = t // max_boxes
    nan2 = wp.vec2f(wp.nan, wp.nan)

    pos = position[t]
    if _is_nan2(pos) == 1:
        out_hit[t] = 0
        out_disc[t] = -1
        out_depth[t] = 0.0
        out_nearest[t] = nan2
        return

    yw = yaw[t]
    he = half_extents[t]
    nd = disc_count[e]
    if nd > d_max:
        nd = d_max
    base = e * d_max

    best = int(-1)
    best_pen = float(0.0)
    for i in range(nd):
        c = discs[base + i]
        if _is_nan2(c) == 0:
            q = _rot2(-yw, c - pos)
            pen = radius - _point_to_local_box_dist(q, he)
            if pen >= 0.0 and (best == -1 or pen > best_pen):
                best = i
                best_pen = pen

    if best == -1:
        out_hit[t] = 0
        out_disc[t] = -1
        out_depth[t] = 0.0
        out_nearest[t] = nan2
        return

    c = discs[base + best]
    # Closest point of the box to the disc center, then step back onto the
    # disc boundary toward it (approximate when c is deep inside the box).
    q = _rot2(-yw, c - pos)
    qcl = wp.vec2f(wp.clamp(q[0], -he[0], he[0]), wp.clamp(q[1], -he[1], he[1]))
    wcl = pos + _rot2(yw, qcl)
    out_hit[t] = 1
    out_disc[t] = best
    out_depth[t] = best_pen
    out_nearest[t] = c + _safe_normalize2(wcl - c) * radius


class DiscChecker:
    """Batched oriented-box vs disc-obstacle checker (collision family).

    See the module docstring. ``num_envs`` is required when ``count`` is not
    given (a flat disc array alone cannot determine the env split).
    """

    def __init__(self, discs: wp.array, radius: float, max_boxes: int,
                 num_envs: "int | None" = None,
                 count: "wp.array | None" = None,
                 position: "wp.array | None" = None,
                 yaw: "wp.array | None" = None,
                 half_extents: "wp.array | None" = None) -> None:
        _init()
        if not (float(radius) > 0.0):
            raise ValueError(f"radius must be > 0, got {radius!r}")
        if int(max_boxes) < 1:
            raise ValueError(f"max_boxes must be >= 1, got {max_boxes!r}")
        if count is not None:
            E = int(count.shape[0])
        elif num_envs is not None:
            E = int(num_envs)
        else:
            raise ValueError("pass num_envs (or a count array): a flat disc "
                             "array alone cannot determine the env split")
        total = int(discs.shape[0])
        if E < 1 or total % E != 0:
            raise ValueError(
                f"discs length {total} not divisible by {E} envs")
        self._discs = discs
        self._radius = float(radius)
        self._E = E
        self._D = total // E
        self._B = int(max_boxes)
        self._device = str(discs.device)
        if count is not None:
            self._count = count            # aliased: stable user buffer
        else:
            self._count = wp.full(E, self._D, dtype=wp.int32,
                                  device=self._device)

        self._bound = None
        if position is not None or yaw is not None or half_extents is not None:
            if position is None or yaw is None or half_extents is None:
                raise ValueError(
                    "bind all of position/yaw/half_extents or none")
            self._validate_inputs(position, yaw, half_extents)
            self._bound = (position, yaw, half_extents)

        n = E * self._B
        dev = self._device
        self._contact = DiscContact(
            hit=wp.zeros(n, dtype=wp.int32, device=dev),
            disc=wp.zeros(n, dtype=wp.int32, device=dev),
            depth=wp.zeros(n, dtype=wp.float32, device=dev),
            nearest=wp.zeros(n, dtype=wp.vec2f, device=dev),
        )

    def _validate_inputs(self, position, yaw, half_extents) -> None:
        n = self._E * self._B
        for name, arr, dtype in (("position", position, wp.vec2f),
                                 ("yaw", yaw, wp.float32),
                                 ("half_extents", half_extents, wp.vec2f)):
            if not isinstance(arr, wp.array):
                raise ValueError(f"{name} must be a wp.array, got {type(arr)!r}")
            if arr.shape != (n,):
                raise ValueError(
                    f"{name} must have shape ({n},) = (E*max_boxes,), got {arr.shape}")
            if arr.dtype is not dtype:
                raise ValueError(
                    f"{name} must have dtype {dtype.__name__}, got {arr.dtype.__name__}")
            if str(arr.device) != self._device:
                raise ValueError(
                    f"{name} is on device {arr.device}, checker is on {self._device}")

    def query(self, position: "wp.array | None" = None,
              yaw: "wp.array | None" = None,
              half_extents: "wp.array | None" = None) -> DiscContact:
        """Box-vs-disc contact for ``E * max_boxes`` boxes.

        Bound mode (inputs bound at construction): call with no arguments.
        Per-call mode: pass all three arrays; under CUDA-graph capture the
        SAME arrays must be used at capture and every replay.
        """
        if self._bound is not None:
            if position is not None or yaw is not None or half_extents is not None:
                raise ValueError(
                    "checker inputs are bound; call query() with no arguments")
            position, yaw, half_extents = self._bound
        else:
            if position is None or yaw is None or half_extents is None:
                raise ValueError(
                    "checker is not bound; pass position, yaw and "
                    "half_extents to query()")
            self._validate_inputs(position, yaw, half_extents)
        c = self._contact
        wp.launch(
            _box_query_discs_k, dim=self._E * self._B,
            inputs=[self._discs, self._count, self._D, self._radius, self._B,
                    position, yaw, half_extents,
                    c.hit, c.disc, c.depth, c.nearest],
            device=self._device,
        )
        _sync(self._device)
        return c
```

- [ ] **Step 4: Re-export from the public collision shim**

Modify `track_gen/collision.py` — extend the import and `__all__`:

```python
from ._src.collision import BoxContact, CollisionChecker
from ._src.collision_discs import DiscChecker, DiscContact

__all__ = ["BoxContact", "CollisionChecker", "DiscChecker", "DiscContact"]
```

(Also extend the module docstring's backend list with one line: disc
obstacles via ``DiscChecker`` — posts, cones, point props.)

- [ ] **Step 5: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_discs.py -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/collision_discs.py track_gen/collision.py tests/test_collision_discs.py
git commit -m "feat: collision.DiscChecker — box vs disc obstacles (gate posts, cones)"
```

---

### Task 7: `CollisionChecker.bind_inputs` retrofit

**Files:**
- Modify: `track_gen/_src/collision.py` (refactor validation, add `bind_inputs`, no-arg `query`)
- Test: `tests/test_collision_bind.py`

**Interfaces:**
- Consumes: existing `CollisionChecker`.
- Produces: `CollisionChecker.bind_inputs(position, yaw, half_extents) -> None`; `query()` accepts no arguments when bound; per-call mode unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collision_bind.py`:

```python
"""Stable input binding for CollisionChecker (retrofit)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests._collision_fixtures import make_annulus_track, make_boxes
from track_gen.collision import CollisionChecker


def test_bound_mode_equivalence_and_live_buffer():
    track = make_annulus_track(E=1, n=256)
    B = 2
    checker_free = CollisionChecker(track, max_boxes=B)
    checker_bound = CollisionChecker(track, max_boxes=B)
    pos, yaw, he = make_boxes(1, B, {(0, 0): (1.1, 0.0, 0.0, 0.05, 0.05),
                                     (0, 1): (0.0, 0.0, 0.0, 0.05, 0.05)})
    checker_bound.bind_inputs(pos, yaw, he)
    r_free = checker_free.query(pos, yaw, he).clone()
    r_bound = checker_bound.query()
    np.testing.assert_array_equal(r_bound.oob.numpy(), r_free.oob.numpy())
    np.testing.assert_allclose(r_bound.distance.numpy(), r_free.distance.numpy(),
                               equal_nan=True)
    # Writing new poses into the bound buffer is seen without re-binding.
    pos2, _, _ = make_boxes(1, B, {(0, 0): (3.0, 0.0, 0.0, 0.05, 0.05),
                                   (0, 1): (0.0, 0.0, 0.0, 0.05, 0.05)})
    wp.copy(pos, pos2)
    r2 = checker_bound.query()
    assert int(r2.oob.numpy()[0]) == 1  # box 0 teleported out of bounds


def test_mode_misuse_errors():
    track = make_annulus_track(E=1, n=256)
    checker = CollisionChecker(track, max_boxes=1)
    pos, yaw, he = make_boxes(1, 1, {(0, 0): (1.0, 0.0, 0.0, 0.05, 0.05)})
    with pytest.raises(ValueError, match="not bound"):
        checker.query()
    checker.bind_inputs(pos, yaw, he)
    with pytest.raises(ValueError, match="bound"):
        checker.query(pos, yaw, he)


def test_bind_validation():
    track = make_annulus_track(E=2, n=256)
    checker = CollisionChecker(track, max_boxes=2)
    bad_pos, yaw, he = make_boxes(2, 1, {})  # wrong stride
    _, good_yaw, good_he = make_boxes(2, 2, {})
    with pytest.raises(ValueError, match="position"):
        checker.bind_inputs(bad_pos, good_yaw, good_he)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_bind.py -v`
Expected: FAIL — `AttributeError: ... 'bind_inputs'` / TypeError on no-arg query

- [ ] **Step 3: Retrofit `track_gen/_src/collision.py`**

In `CollisionChecker.__init__`, after `self._contact = BoxContact(...)` add:

```python
        self._bound: "tuple | None" = None
```

Extract the existing validation loop from `query()` into a method (verbatim
move of the loop body):

```python
    def _validate_inputs(self, position, yaw, half_extents) -> None:
        n = self._E * self._B
        for name, arr, dtype in (("position", position, wp.vec2f),
                                 ("yaw", yaw, wp.float32),
                                 ("half_extents", half_extents, wp.vec2f)):
            if not isinstance(arr, wp.array):
                raise ValueError(f"{name} must be a wp.array, got {type(arr)!r}")
            if arr.shape != (n,):
                raise ValueError(
                    f"{name} must have shape ({n},) = (E*max_boxes,), got {arr.shape}")
            if arr.dtype is not dtype:
                raise ValueError(
                    f"{name} must have dtype {dtype.__name__}, got {arr.dtype.__name__}")
            if str(arr.device) != self._device:
                raise ValueError(
                    f"{name} is on device {arr.device}, checker is on {self._device}")
```

Add after `bake()`:

```python
    def bind_inputs(self, position: wp.array, yaw: wp.array,
                    half_extents: wp.array) -> None:
        """Bind stable per-step input buffers (validated once, here).

        After binding, ``query()`` takes no arguments and reads these arrays
        in place each call — the natural CUDA-graph pattern: the sim writes
        poses into its stable buffers and replays the captured query. The
        arrays must keep the same ``.ptr`` for the binding's lifetime.
        """
        self._validate_inputs(position, yaw, half_extents)
        self._bound = (position, yaw, half_extents)
```

Change `query()`'s signature and head (the launch/dispatch below stays
untouched):

```python
    def query(self, position: "wp.array | None" = None,
              yaw: "wp.array | None" = None,
              half_extents: "wp.array | None" = None) -> BoxContact:
```

and replace the old validation loop at the top of `query()` with:

```python
        if self._bound is not None:
            if position is not None or yaw is not None or half_extents is not None:
                raise ValueError(
                    "checker inputs are bound; call query() with no arguments")
            position, yaw, half_extents = self._bound
        else:
            if position is None or yaw is None or half_extents is None:
                raise ValueError(
                    "checker is not bound; pass position, yaw and "
                    "half_extents to query() or call bind_inputs() first")
            self._validate_inputs(position, yaw, half_extents)
```

Update the `query()` docstring Args section accordingly (bound mode /
per-call mode, same-ptr rule under capture).

- [ ] **Step 4: Run the new tests plus the existing collision suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_bind.py tests/test_collision_segments.py tests/test_collision_contract.py tests/test_collision_sdf.py -q`
Expected: all PASS (no regression in per-call mode)

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/collision.py tests/test_collision_bind.py
git commit -m "feat: CollisionChecker.bind_inputs — stable input binding retrofit"
```

---

### Task 8: CUDA graph tests — bound mode, poisoned replay

**Files:**
- Test: `tests/test_course_cuda_graph.py`

**Interfaces:**
- Consumes: everything from Tasks 2–7; `_CAPTURING` flags in `track_gen._src.progress`, `track_gen._src.checkpoints`, `track_gen._src.collision_discs`.

- [ ] **Step 1: Write the tests**

Create `tests/test_course_cuda_graph.py`:

```python
"""CUDA-only: checkpoints/progress/discs under wp.ScopedCapture (bound mode).

Poisoned-buffer replay proves each captured graph recomputes results. The
progress test compares a captured-replay trace against an eager twin tracker
stepped over the same positions.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda"),
]

import warp as wp  # noqa: E402
from tests._collision_fixtures import make_annulus_track, make_boxes  # noqa: E402
from track_gen._src import checkpoints as cps_mod  # noqa: E402
from track_gen._src import collision_discs as discs_mod  # noqa: E402
from track_gen._src import progress as prog_mod  # noqa: E402
from track_gen.checkpoints import CheckpointSampler  # noqa: E402
from track_gen.collision import DiscChecker  # noqa: E402
from track_gen.progress import ProgressTracker  # noqa: E402

DEV = "cuda:0"


def test_checkpoint_sample_graph_replay():
    track = make_annulus_track(E=4, n=256, device=DEV)
    sampler = CheckpointSampler(track, spacing=0.8)
    eager = sampler.sample().clone()
    prev = cps_mod._CAPTURING
    cps_mod._CAPTURING = True
    try:
        sampler.sample()
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            sampler.sample()
    finally:
        cps_mod._CAPTURING = prev
    sampler._set.position.fill_(12345.0)
    sampler._set.count.fill_(-7)
    wp.capture_launch(cap.graph)
    wp.synchronize()
    np.testing.assert_array_equal(sampler._set.count.numpy(), eager.count.numpy())
    np.testing.assert_allclose(sampler._set.position.numpy(),
                               eager.position.numpy(), rtol=1e-5, equal_nan=True)


def test_progress_update_graph_replay_matches_eager_twin():
    E = 4
    track = make_annulus_track(E=E, n=256, device=DEV)
    cps = CheckpointSampler(track, spacing=0.8).sample()
    pos_buf = wp.zeros(E, dtype=wp.vec2f, device=DEV)
    bound = ProgressTracker(cps, position=pos_buf)
    eager = ProgressTracker(cps)

    # Capture one bound update (warmup on a twin state, then reset).
    prev = prog_mod._CAPTURING
    prog_mod._CAPTURING = True
    try:
        bound.update()
        wp.synchronize()
        mask = wp.full(E, 1, dtype=wp.int32, device=DEV)
        bound.reset(mask)
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            bound.update()
    finally:
        prog_mod._CAPTURING = prev

    # Walk the annulus centerline CCW; both trackers see identical positions.
    steps = [np.stack([np.cos(a) * 1.0 * np.ones(E), np.sin(a) * np.ones(E)],
                      axis=1).astype(np.float32)
             for a in np.deg2rad(np.arange(-20.0, 340.0, 40.0))]
    for s in steps:
        arr = wp.array(s, dtype=wp.vec2f, device=DEV)
        wp.copy(pos_buf, arr)
        bound._events.passed.fill_(-7)         # poison: replay must recompute
        bound._events.dist_to_next.fill_(12345.0)
        wp.capture_launch(cap.graph)
        wp.synchronize()
        ev_e = eager.update(arr)
        np.testing.assert_array_equal(bound._events.passed.numpy(),
                                      ev_e.passed.numpy())
        np.testing.assert_array_equal(bound._events.next_checkpoint.numpy(),
                                      ev_e.next_checkpoint.numpy())
        np.testing.assert_allclose(bound._events.dist_to_next.numpy(),
                                   ev_e.dist_to_next.numpy(), rtol=1e-5,
                                   equal_nan=True)


def test_disc_query_graph_replay_bound():
    E, B = 2, 4
    discs = wp.array(np.array([[0.12, 0.0], [0.5, 0.5]] * E, np.float32),
                     dtype=wp.vec2f, device=DEV)
    pos, yaw, he = make_boxes(E, B, {(e, 0): (0.0, 0.0, 0.0, 0.1, 0.05)
                                     for e in range(E)}, device=DEV)
    checker = DiscChecker(discs, radius=0.03, max_boxes=B, num_envs=E,
                          position=pos, yaw=yaw, half_extents=he)
    eager = checker.query().clone()
    prev = discs_mod._CAPTURING
    discs_mod._CAPTURING = True
    try:
        checker.query()
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            checker.query()
    finally:
        discs_mod._CAPTURING = prev
    checker._contact.hit.fill_(-7)
    checker._contact.depth.fill_(12345.0)
    wp.capture_launch(cap.graph)
    wp.synchronize()
    np.testing.assert_array_equal(checker._contact.hit.numpy(), eager.hit.numpy())
    np.testing.assert_allclose(checker._contact.depth.numpy(),
                               eager.depth.numpy(), rtol=1e-6)
```

- [ ] **Step 2: Run (GPU present on this machine — must RUN and PASS)**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_course_cuda_graph.py -v`
Expected: 3 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_course_cuda_graph.py
git commit -m "test: CUDA graph capture for checkpoints/progress/discs in bound mode"
```

---

### Task 9: Figures — three new deterministic renderers

**Files:**
- Modify: `viz/render_utility_assets.py` (append three functions + extend `main`)
- Modify: `tests/test_readme_assets.py` (append smoke test)
- Create (generated): `docs/assets/checkpoints-overview.png`, `docs/assets/progress-tracking.png`, `docs/assets/disc-collision.png`

**Interfaces:**
- Consumes: Tasks 2, 4, 6 public APIs; existing helpers `_close`, `_draw_track`, `GEN_SEED` in the module.
- Produces: `render_checkpoints_overview(output_dir=Path("docs/assets")) -> Path`, `render_progress_tracking(output_dir=...) -> Path`, `render_disc_collision(output_dir=...) -> Path`; `main()` renders all four assets.

- [ ] **Step 1: Append the renderers to `viz/render_utility_assets.py`**

```python
def render_checkpoints_overview(output_dir: Path = Path("docs/assets")) -> Path:
    """Track-sourced checkpoints (virtual gates) beside gate-sourced ones."""
    import warp as wp  # noqa: F401

    from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG,
                           TrackGenConfig, TrackGenerator)
    from track_gen.checkpoints import CheckpointSampler, CheckpointSet

    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    track = gen.generate()
    e = int(np.argmax(track.valid.numpy()))
    n_max = track.outer.shape[0] // E
    m = int(track.count.numpy()[e])
    inner = track.inner.numpy().reshape(E, n_max, 2)[e, :m]
    outer = track.outer.numpy().reshape(E, n_max, 2)[e, :m]

    sampler = CheckpointSampler(track, spacing=0.6)
    cps = sampler.sample()
    M = sampler._M
    n = int(cps.count.numpy()[e])
    sl = slice(e * M, e * M + n)
    pos = cps.position.numpy().reshape(-1, 2)[sl]
    left = cps.left.numpy().reshape(-1, 2)[sl]
    right = cps.right.numpy().reshape(-1, 2)[sl]
    tang = cps.tangent.numpy().reshape(-1, 2)[sl]

    gcfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.08)
    ggen = GateGenerator(gcfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    seq = ggen.generate()
    gset = CheckpointSet.from_gates(seq)
    ge = int(np.argmax(seq.valid.numpy()))
    GM = gset.position.shape[0] // E
    gn = int(gset.count.numpy()[ge])
    gsl = slice(ge * GM, ge * GM + gn)
    gpos = gset.position.numpy().reshape(-1, 2)[gsl]
    gleft = gset.left.numpy().reshape(-1, 2)[gsl]
    gright = gset.right.numpy().reshape(-1, 2)[gsl]
    gtang = gset.tangent.numpy().reshape(-1, 2)[gsl]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.suptitle("CheckpointSet: one contract, two sources", fontsize=14)

    ax = axes[0]
    _draw_track(ax, inner, outer)
    for lf, rt in zip(left, right):
        ax.plot([lf[0], rt[0]], [lf[1], rt[1]], "-", color="#7570b3", lw=2.0,
                zorder=3)
    ax.scatter(pos[:, 0], pos[:, 1], s=22, color="#d95f02", zorder=4)
    ax.quiver(pos[:, 0], pos[:, 1], tang[:, 0], tang[:, 1], color="#d95f02",
              width=0.004, scale=18, zorder=4)
    ax.set_title(f"CheckpointSampler(track, spacing=0.6): {n} virtual gates\n"
                 "(crossing segments = inner-outer road cross-sections)")

    ax = axes[1]
    for lf, rt in zip(gleft, gright):
        ax.plot([lf[0], rt[0]], [lf[1], rt[1]], "-", color="#1b9e77", lw=2.6,
                zorder=3)
    ax.scatter(gpos[:, 0], gpos[:, 1], s=22, color="#d95f02", zorder=4)
    ax.quiver(gpos[:, 0], gpos[:, 1], gtang[:, 0], gtang[:, 1],
              color="#d95f02", width=0.004, scale=18, zorder=4)
    ax.plot(*np.vstack([gpos, gpos[:1]]).T, ":", color="0.6", lw=0.8, zorder=2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"CheckpointSet.from_gates(seq): {gn} gates (zero-copy)")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "checkpoints-overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_progress_tracking(output_dir: Path = Path("docs/assets")) -> Path:
    """Scripted agent threading track checkpoints; dist_to_next sawtooth inset."""
    import warp as wp

    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    from track_gen.checkpoints import CheckpointSampler
    from track_gen.progress import ProgressTracker

    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    track = gen.generate()
    e = int(np.argmax(track.valid.numpy()))
    n_max = track.outer.shape[0] // E
    m = int(track.count.numpy()[e])
    inner = track.inner.numpy().reshape(E, n_max, 2)[e, :m]
    outer = track.outer.numpy().reshape(E, n_max, 2)[e, :m]
    center = track.center.numpy().reshape(E, n_max, 2)[e, :m]

    sampler = CheckpointSampler(track, spacing=0.9)
    cps = sampler.sample()
    M = sampler._M
    n = int(cps.count.numpy()[e])
    cpos = cps.position.numpy().reshape(-1, 2)[e * M:e * M + n]
    cleft = cps.left.numpy().reshape(-1, 2)[e * M:e * M + n]
    cright = cps.right.numpy().reshape(-1, 2)[e * M:e * M + n]

    tracker = ProgressTracker(cps)
    rng = np.random.default_rng(GEN_SEED)
    path_idx = np.arange(0, m, 3)
    path = center[path_idx] + rng.normal(0.0, 0.01, (len(path_idx), 2))
    prog_trace, dist_trace, passed_at = [], [], []
    for s, p in enumerate(path):
        full = np.zeros((E, 2), np.float32)
        full[e] = p
        ev = tracker.update(wp.array(full, dtype=wp.vec2f, device="cpu"))
        prog_trace.append(int(ev.progress.numpy()[e]))
        dist_trace.append(float(ev.dist_to_next.numpy()[e]))
        if int(ev.passed.numpy()[e]):
            passed_at.append(int(ev.checkpoint_passed.numpy()[e]))
    target = int(tracker._next.numpy()[e])

    fig, ax = plt.subplots(figsize=(10.5, 9))
    _draw_track(ax, inner, outer)
    for k, (lf, rt) in enumerate(zip(cleft, cright)):
        col = "#1a9641" if k in passed_at else ("#d95f02" if k == target else "0.6")
        lw = 3.0 if k == target else 2.0
        ax.plot([lf[0], rt[0]], [lf[1], rt[1]], "-", color=col, lw=lw, zorder=3)
    sc = ax.scatter(path[:, 0], path[:, 1], c=prog_trace, cmap="viridis", s=12,
                    zorder=4)
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="progress (checkpoints passed)")
    ax.plot([], [], "-", color="#1a9641", lw=2, label="passed")
    ax.plot([], [], "-", color="#d95f02", lw=3, label="current target")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("ProgressTracker on track checkpoints: path colored by progress")

    ins = ax.inset_axes([0.03, 0.03, 0.42, 0.22])
    ins.plot(dist_trace, lw=1.2, color="#7570b3")
    ins.set_title("dist_to_next per step (reward = -delta)", fontsize=8)
    ins.tick_params(labelsize=7)

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "progress-tracking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_disc_collision(output_dir: Path = Path("docs/assets")) -> Path:
    """Gate posts as discs; agent boxes colored by DiscChecker verdicts."""
    import warp as wp
    from matplotlib.patches import Circle

    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.collision import DiscChecker

    E = 4
    RADIUS = 0.03
    cfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.08)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    seq = gen.generate()
    e = int(np.argmax(seq.valid.numpy()))
    G = seq.position.shape[0] // E
    ln = int(seq.count.numpy()[e])
    left = seq.left.numpy().reshape(E, G, 2)
    right = seq.right.numpy().reshape(E, G, 2)

    posts = np.empty((E, 2 * G, 2), np.float32)
    posts[:, 0::2] = left
    posts[:, 1::2] = right
    posts_wp = wp.array(posts.reshape(-1, 2), dtype=wp.vec2f, device="cpu")

    B = 8
    rng = np.random.default_rng(BOX_SEED)
    pos_np = np.full((E * B, 2), np.nan, np.float32)
    yaw_np = np.zeros(E * B, np.float32)
    he_np = np.zeros((E * B, 2), np.float32)
    for b in range(B):
        g = int(rng.integers(0, ln))
        anchor = posts[e, 2 * g + (b % 2)]
        pos_np[e * B + b] = anchor + rng.normal(0.0, 0.05, 2)
        yaw_np[e * B + b] = rng.uniform(0, 2 * np.pi)
        he_np[e * B + b] = rng.uniform(0.02, 0.05, 2)
    checker = DiscChecker(posts_wp, radius=RADIUS, max_boxes=B, num_envs=E)
    res = checker.query(wp.array(pos_np.reshape(-1, 2), dtype=wp.vec2f, device="cpu"),
                        wp.array(yaw_np, dtype=wp.float32, device="cpu"),
                        wp.array(he_np, dtype=wp.vec2f, device="cpu"))
    hit = res.hit.numpy()[e * B:(e + 1) * B]

    fig, ax = plt.subplots(figsize=(10.5, 9))
    for g in range(ln):
        lf, rt = left[e, g], right[e, g]
        ax.plot([lf[0], rt[0]], [lf[1], rt[1]], "-", color="0.75", lw=1.2, zorder=1)
        for p in (lf, rt):
            ax.add_patch(Circle(p, RADIUS, facecolor="#7570b3", alpha=0.5,
                                edgecolor="#7570b3", zorder=2))
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    for b in range(B):
        c, yw, he_ = pos_np[e * B + b], yaw_np[e * B + b], he_np[e * B + b]
        rot = np.array([[np.cos(yw), -np.sin(yw)], [np.sin(yw), np.cos(yw)]])
        corners = c + (signs * he_) @ rot.T
        col = "#d7191c" if hit[b] else "#1a9641"
        ax.add_patch(MplPolygon(corners, closed=True, facecolor="none",
                                edgecolor=col, lw=2.0, zorder=3))
    ax.plot([], [], "-", color="#d7191c", lw=2, label="post hit")
    ax.plot([], [], "-", color="#1a9641", lw=2, label="clear")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("DiscChecker: gate posts as disc obstacles (radius %.2f)" % RADIUS)

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "disc-collision.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
```

And change `main()` to:

```python
def main() -> None:
    print(render_utilities_overview().resolve())
    print(render_checkpoints_overview().resolve())
    print(render_progress_tracking().resolve())
    print(render_disc_collision().resolve())
```

- [ ] **Step 2: Render and inspect**

Run: `python3 -m viz.render_utility_assets`
Expected: prints four paths; visually inspect the three new PNGs (Read them)
— aligned panels, legible legends, sawtooth visible in the progress inset.

- [ ] **Step 3: Append the smoke test to `tests/test_readme_assets.py`**

```python
@pytest.mark.slow
def test_render_course_assets_write_pngs(tmp_path):
    from viz.render_utility_assets import (render_checkpoints_overview,
                                           render_disc_collision,
                                           render_progress_tracking)

    names = {render_checkpoints_overview(output_dir=tmp_path).name,
             render_progress_tracking(output_dir=tmp_path).name,
             render_disc_collision(output_dir=tmp_path).name}
    assert names == {"checkpoints-overview.png", "progress-tracking.png",
                     "disc-collision.png"}
    for n in names:
        assert (tmp_path / n).stat().st_size > 1000
```

- [ ] **Step 4: Run the smoke tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_readme_assets.py -q`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add viz/render_utility_assets.py tests/test_readme_assets.py docs/assets/checkpoints-overview.png docs/assets/progress-tracking.png docs/assets/disc-collision.png
git commit -m "docs: checkpoints/progress/disc-collision figures (deterministic renderers)"
```

---

### Task 10: Tutorial page, API reference, final verification

**Files:**
- Create: `docs/tutorials/runtime-utilities.rst`
- Modify: `docs/index.rst` (add `tutorials/runtime-utilities` to the tutorials toctree, after the existing tutorial entries)
- Modify: `docs/reference/api.rst` (Checkpoints + Progress sections; DiscChecker/DiscContact under Collision queries)
- Modify: `docs/contributing/rendering-assets.rst` (extend the utilities-figure section to name the three new files)

- [ ] **Step 1: Write the tutorial page**

Create `docs/tutorials/runtime-utilities.rst`:

```rst
Runtime utilities: collision, props, checkpoints & progress
============================================================

Beyond generating tracks and gates, ``track_gen`` ships a family of
GPU-batched runtime utilities for the sim loop. They share the library's
conventions: flat NaN-padded batches, preallocated results overwritten in
place (``clone()`` for snapshots), CUDA-graph-capturable hot paths, and
results that are undefined for invalid envs (always gate on ``valid``).

.. figure:: ../assets/utilities-overview.png
   :alt: Overview of collision and props utilities.

   Out-of-bounds collision (segments + SDF backends) and boundary prop
   instancing on one generated track.

Out-of-bounds collision
-----------------------

``track_gen.collision.CollisionChecker`` answers, per oriented box, whether
it left the drivable band — with signed clearance, nearest boundary point,
and inward normal. Two backends: the exact ``segments`` scan (default; no
precompute, reads regenerated tracks automatically) and baked ``sdf`` grids
(O(1) queries after a per-regeneration ``bake()``; distances accurate to
about one grid cell). See the API reference for measured numbers; the rule
of thumb: ``segments`` unless a track batch serves hundreds of queries
between regenerations.

Boundary props (rendering-only instancing)
------------------------------------------

``track_gen.props.PropSampler`` resamples a boundary at a set spacing into
instancing poses — cones (``mode="points"``) or wall pieces
(``mode="segments"``, chord midpoint + yaw + length). Spacing snaps per env
so every ring closes without a seam. Props are not colliders; to make point
props physical, feed their positions to ``DiscChecker`` (below).

Checkpoints: one contract, two sources
--------------------------------------

``track_gen.checkpoints.CheckpointSet`` is an ordered list of course goals
per env — center ``position``, a physical crossing segment ``left <->
right``, and a forward ``tangent``:

.. code-block:: python

   from track_gen.checkpoints import CheckpointSampler, CheckpointSet

   cps = CheckpointSet.from_gates(gate_seq)          # zero-copy gate view
   cps = CheckpointSampler(track, spacing=0.6).sample()   # virtual gates

``from_gates`` aliases the ``GateSequence`` buffers (regenerated gates are
seen automatically). ``CheckpointSampler`` subsamples the CENTERLINE at a
coarse spacing; because track polylines are index-aligned, each checkpoint's
crossing segment is the road cross-section between ``inner`` and ``outer``.

.. figure:: ../assets/checkpoints-overview.png
   :alt: Track-sourced virtual gates beside gate-sourced checkpoints.

   The same ``CheckpointSet`` contract from a subsampled track (left) and a
   gate sequence (right).

Progress tracking & rewards
---------------------------

``track_gen.progress.ProgressTracker`` consumes any ``CheckpointSet`` and
maintains per-env device state: previous position, next target, laps, total
progress. Each ``update()`` detects forward pass-through of the target's
crossing segment (swept-segment test), wrong-way and wrong-checkpoint
crossings, and reports ``dist_to_next`` — the distance to the next goal:

.. code-block:: python

   from track_gen.progress import ProgressTracker

   tracker = ProgressTracker(cps, position=robot_pos)  # latch onto sim buffer
   prev_d = None
   for _ in range(steps):
       sim.step()                       # writes robot_pos in place
       ev = tracker.update()            # no args: bound mode
       d = wp.to_torch(ev.dist_to_next)
       reward = (prev_d - d) if prev_d is not None else 0.0   # -delta distance
       reward = reward + 10.0 * wp.to_torch(ev.passed)        # pass bonus
       prev_d = d.clone()
   tracker.reset(done_mask)             # episodic resets, per env

``reset(mask)`` arms a NaN previous-position sentinel, so the first step
after a reset (or a teleport respawn) can never emit a spurious crossing.
After regenerating the course (gates or track), call ``reset`` for all envs.

.. figure:: ../assets/progress-tracking.png
   :alt: Agent path colored by progress with a dist_to_next sawtooth inset.

   A scripted agent threading track checkpoints. The inset shows the
   ``dist_to_next`` sawtooth your negative-delta reward differentiates.

Gate posts & point obstacles
----------------------------

``track_gen.collision.DiscChecker`` checks oriented boxes against disc
obstacles. Gate posts are two lines of code:

.. code-block:: python

   import numpy as np, warp as wp
   from track_gen.collision import DiscChecker

   posts = np.empty((E, 2 * G, 2), np.float32)
   posts[:, 0::2] = wp.to_torch(seq.left).view(E, G, 2).cpu().numpy()
   posts[:, 1::2] = wp.to_torch(seq.right).view(E, G, 2).cpu().numpy()
   checker = DiscChecker(wp.array(posts.reshape(-1, 2), dtype=wp.vec2f,
                                  device=dev),
                         radius=0.03, max_boxes=1, num_envs=E)

NaN padding in the gate arrays carries over and NaN discs are skipped, so no
per-env count bookkeeping is needed. A hit reports the deepest disc; for
interleaved posts, ``gate = disc // 2``.

.. figure:: ../assets/disc-collision.png
   :alt: Gate posts as discs with boxes colored by hit.

   Gate posts as disc obstacles; the same checker makes cones physical.

Stable buffers and CUDA graphs
------------------------------

All utilities preallocate their state and results once (stable pointers) and
never allocate in the hot path. Per-step inputs can be BOUND once instead of
passed per call — ``ProgressTracker(cps, position=buf)``,
``DiscChecker(..., position=..., yaw=..., half_extents=...)``, and
``CollisionChecker.bind_inputs(...)`` — after which ``update()``/``query()``
take no arguments and read the buffers in place. Under graph capture this is
the intended pattern: the sim writes its stable pose buffers, then replays
the captured update. (Per-call mode also works under capture, but the SAME
arrays must be passed at capture and every replay.)
```

- [ ] **Step 2: Wire the toctree and API reference**

In `docs/index.rst`, add `tutorials/runtime-utilities` on a new line
directly after the last existing `tutorials/...` entry in the toctree.

In `docs/reference/api.rst`, append after the "Boundary props" section:

```rst
Checkpoints
-----------

Ordered course goals from gates (zero-copy) or subsampled track centerlines.

.. automodule:: track_gen.checkpoints
   :no-members:

.. autoclass:: track_gen.checkpoints.CheckpointSampler
   :members:

.. autoclass:: track_gen.checkpoints.CheckpointSet
   :no-members:

   .. automethod:: from_gates

   .. automethod:: clone

Progress tracking
-----------------

Stateful per-env course progress over any ``CheckpointSet``.

.. autoclass:: track_gen.progress.ProgressTracker
   :members:

.. autoclass:: track_gen.progress.ProgressEvents
   :no-members:

   .. automethod:: clone
```

and under the existing "Collision queries" section (after the `BoxContact`
entry, before "Performance"):

```rst
.. autoclass:: track_gen.collision.DiscChecker
   :members:

.. autoclass:: track_gen.collision.DiscContact
   :no-members:

   .. automethod:: clone
```

In `docs/contributing/rendering-assets.rst`, in the "Utilities overview
figure" section, change the intro sentence to say the module now renders
FOUR figures and list the three new filenames alongside
``utilities-overview.png``.

- [ ] **Step 3: Build docs and run the complete suite**

Run: `python3 -m sphinx -b html docs /tmp/claude-1000/-home-antoine-Documents-track-gen/a3819d36-c82d-4063-bcba-b7abbecf061d/scratchpad/docs-build -q 2>&1 | tail -3`
Expected: no warnings mentioning the new pages/figures.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all PASS (GPU present: cuda tests run)

- [ ] **Step 4: Commit**

```bash
git add docs/tutorials/runtime-utilities.rst docs/index.rst docs/reference/api.rst docs/contributing/rendering-assets.rst
git commit -m "docs: runtime-utilities tutorial + API reference for checkpoints/progress/discs"
```

---

## Self-Review Notes (completed during planning)

- **Spec coverage:** CheckpointSet + from_gates aliasing + CheckpointSampler snap/derivation/degenerate (Tasks 2–3), ProgressTracker semantics incl. double-jump/wrong-way/reset sentinel/dist_to_next + binding (Tasks 4–5), DiscChecker + gate-post recipe + NaN-skip/count modes (Task 6), CollisionChecker.bind_inputs retrofit (Task 7), stable-buffer/graph contract tests in bound mode (Task 8), tutorial + three figures + api.rst + toctree (Tasks 9–10). Out-of-scope list respected.
- **Type consistency:** `CheckpointSet` field names used identically in Tasks 2–5, 9–10; `ProgressEvents` fields match kernel outputs and oracle keys; `DiscChecker(discs, radius, max_boxes, num_envs, count, position, yaw, half_extents)` consistent between Tasks 6 and 8–10; `_segs_cross` signature matches Task 1.
- **Known risks called out to implementers:** warp codegen (float-wrapped constants) noted in Global Constraints; the progress kernel's dynamic `for i in range(n)` loop with early `wcp` guard is standard; `wp.full` availability for int fill (used in DiscChecker/reset test) — if absent on this warp version, replace with `wp.zeros` + `.fill_(1)`.
