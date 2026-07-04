# Boundary Prop Sampling Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `track_gen.props` — a Warp-native utility that resamples track boundaries at a user-set spacing into instancing poses for rendering-only props: `points` mode (cones/poles) and `segments` mode (wall pieces), snapped so each closed ring has no seam.

**Architecture:** A `PropSampler` facade binds one `Track` + one boundary + one mode + one spacing, preallocates a `PropSet` (flat `[E*max_props]` wp.arrays, in-place reuse contract) and a cumulative-arc-length scratch. `sample()` is two kernel launches: a thread-per-env scan (cumulative arc table, perimeter, snapped count/step/truncated) and a thread-per-slot placement (binary search + lerp → pose). Same module conventions as `track_gen/_src/collision.py` (`_init`/`_sync`/`_CAPTURING`).

**Tech Stack:** Python ≥ 3.10, NVIDIA Warp ≥ 1.14 (`warp-lang>=1.14` already in pyproject), numpy. Tests: pytest (+ torch only in the CUDA-marked test).

**Spec:** `docs/superpowers/specs/2026-07-04-props-utility-design.md`

## Global Constraints

- Runtime deps are **numpy + warp-lang only**; numpy allowed at construction time (host-side `max_props` derivation), never inside `sample()`.
- Everything passed via `wp.launch(..., inputs=[...])` — never `outputs=`.
- `sample()` is **allocation-free and, under graph capture (module `_CAPTURING` set), host-sync-free**; all buffers allocated in `__init__`; `_sync(device)` early-returns when `_CAPTURING` (copy the `collision.py` idiom).
- Flat `[E * max_props]` outputs, NaN-padded past `count[e]`. Degenerate env (`track.count[e] < 3`): `count=0`, `truncated=0`, `step=NaN`, all per-prop fields NaN.
- Snap rule: `n[e] = clamp(round(perimeter_e / spacing), 3, max_props)`, effective `step[e] = perimeter_e / n[e]`; `truncated[e]=1` iff the `max_props` clamp bound.
- points mode: on-curve position at arc `k*step`, tangent = containing polyline segment direction, `length = step`. segments mode: chord from sample `k` to sample `(k+1) mod n` — `position` = chord midpoint, tangent/yaw = chord direction, `length` = chord length; all `n` spans emitted.
- `yaw = atan2(tangent.y, tangent.x)` in both modes.
- In-place output contract: `sample()` returns the SAME `PropSet` every call; `clone()` for snapshots.
- warp-1.14 codegen caution (hit in the collision build): a kernel local seeded from a **bare module-level float constant** and then mutated inside a dynamic (`range(m)`) loop raises `WarpCodegenError` — wrap the seed as `float(CONST)`. Plain literals (`s = float(0.0)`) are fine.
- Every pytest run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest …`
- Work on a feature branch `feature/props-utility` off main (create in Task 1 Step 0).

---

### Task 1: Core module — kernels, `PropSampler`, `PropSet`, public wiring, points-mode analytic tests

**Files:**
- Create: `track_gen/_src/props.py`
- Create: `track_gen/props.py`
- Modify: `track_gen/__init__.py` (add `from . import props`, extend `__all__`)
- Modify: `tests/test_public_api.py` (add `"props"` to the curated expected `__all__` set — one line, same as `"collision"` was added)
- Test: `tests/test_props.py` (started here, extended in Task 2)

**Interfaces:**
- Consumes: `Track` from `track_gen._src.types`; `_safe_normalize2` from `track_gen._src.collision_geom`; test fixtures `make_annulus_track`, `annulus_polylines` from `tests/_collision_fixtures.py`.
- Produces (later tasks rely on these exact names):
  - `track_gen.props.PropSet` — dataclass, fields `position` (vec2f), `tangent` (vec2f), `yaw` (float32), `length` (float32) all `[E*max_props]`; `count` (int32), `truncated` (int32), `step` (float32) all `[E]`; method `clone() -> PropSet`.
  - `track_gen.props.PropSampler(track, spacing, boundary="outer", mode="points", max_props=None)`; method `sample() -> PropSet`; internals used by tests: `_M` (resolved max_props), `_props` (the PropSet), and module flag `track_gen._src.props._CAPTURING`.

- [ ] **Step 0: Create the feature branch**

```bash
git checkout -b feature/props-utility
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_props.py`:

```python
"""Analytic annulus tests for track_gen.props (boundary prop sampling)."""
from __future__ import annotations

import numpy as np

from tests._collision_fixtures import annulus_polylines, make_annulus_track

N = 512
N_MAX = N + 8
RO = 1.3  # outer boundary radius of the default annulus fixture
RI = 0.7


def _outer_perimeter(track, e=0):
    _, outer = annulus_polylines(track, e, N_MAX)
    seg = np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1)
    return float(seg.sum())


def test_import_surface():
    import track_gen
    from track_gen.props import PropSampler, PropSet  # noqa: F401
    assert "props" in track_gen.__all__
    assert track_gen.props.PropSampler is PropSampler


def test_points_mode_snapped_count_and_step():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    spacing = 0.1
    sampler = PropSampler(track, spacing=spacing, boundary="outer", mode="points")
    props = sampler.sample()
    perim = _outer_perimeter(track)
    n_expected = int(round(perim / spacing))
    assert int(props.count.numpy()[0]) == n_expected
    np.testing.assert_allclose(props.step.numpy()[0], perim / n_expected, rtol=1e-5)
    assert int(props.truncated.numpy()[0]) == 0


def test_points_mode_positions_on_circle_uniform_gaps():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points")
    props = sampler.sample()
    n = int(props.count.numpy()[0])
    pos = props.position.numpy().reshape(-1, 2)[:n]
    # On the outer circle (polyline tolerance).
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), RO, atol=2e-3)
    # Uniform angular gaps that close the ring: n gaps of 2*pi/n each.
    ang = np.arctan2(pos[:, 1], pos[:, 0])
    gaps = np.diff(np.concatenate([ang, ang[:1]]))
    gaps = np.mod(gaps, 2 * np.pi)
    np.testing.assert_allclose(gaps, 2 * np.pi / n, atol=2e-3)


def test_points_mode_tangent_yaw_length():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points")
    props = sampler.sample()
    n = int(props.count.numpy()[0])
    pos = props.position.numpy().reshape(-1, 2)[:n]
    tang = props.tangent.numpy().reshape(-1, 2)[:n]
    yaw = props.yaw.numpy()[:n]
    length = props.length.numpy()[:n]
    # Unit tangents, perpendicular to the radial direction (circle tangent).
    np.testing.assert_allclose(np.linalg.norm(tang, axis=1), 1.0, atol=1e-5)
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    assert np.abs((tang * radial).sum(axis=1)).max() < 0.02
    np.testing.assert_allclose(yaw, np.arctan2(tang[:, 1], tang[:, 0]), atol=1e-6)
    np.testing.assert_allclose(length, props.step.numpy()[0], rtol=1e-5)


def test_nan_padding_past_count():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points")
    props = sampler.sample()
    n = int(props.count.numpy()[0])
    assert sampler._M > n
    pos = props.position.numpy().reshape(-1, 2)
    assert np.all(np.isnan(pos[n:]))
    assert np.all(np.isnan(props.yaw.numpy()[n:]))
    assert np.all(np.isnan(props.length.numpy()[n:]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_props.py -v`
Expected: FAIL at import — `ModuleNotFoundError: track_gen.props`

- [ ] **Step 3: Implement `track_gen/_src/props.py`**

```python
"""Batched boundary prop sampling for rendering-only instancing.

``PropSampler`` resamples one track boundary (inner or outer) at a user-set
arc-length spacing and emits per-prop instancing poses into a preallocated
:class:`PropSet`. Two modes:

- ``"points"``: one pose per sample on the curve (cones, poles, markers).
- ``"segments"``: one pose per chord between consecutive samples (wall
  pieces): position = chord midpoint, yaw = chord direction, length = chord
  length, so a unit-length wall scaled by ``length`` tiles the ring.

Spacing is snapped per env — ``n = clamp(round(perimeter/spacing), 3,
max_props)`` at effective step ``perimeter/n`` — so the closed ring has no
seam gap or doubled prop. Props are NOT colliders (see ``track_gen.collision``
for out-of-bounds queries); this is the rendering/instancing complement.

Layout follows the package conventions: flat ``[E * max_props]`` wp.arrays,
NaN-padded past ``count[e]``, in-place reuse of the output ``PropSet`` across
``sample()`` calls (use ``clone()`` for snapshots), and no host syncs while
``_CAPTURING`` is set so ``sample()`` is CUDA-graph capturable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .collision_geom import _safe_normalize2
from .types import Track

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
class PropSet:
    """Batched prop poses, flat ``[E * max_props]`` per pose field.

    .. warning::

        ``PropSampler.sample()`` returns the SAME ``PropSet`` instance on
        every call and overwrites its buffers in place. Call ``clone()`` for
        a fully-owned snapshot.

    Attributes
    ----------
    position : wp.array
        ``vec2f`` pose positions: on-curve sample (points mode) or chord
        midpoint (segments mode). NaN past ``count[e]``.
    tangent : wp.array
        ``vec2f`` unit directions: curve tangent at the sample (points) or
        chord direction (segments). NaN past ``count[e]``.
    yaw : wp.array
        ``float32`` heading, ``atan2(tangent.y, tangent.x)``.
    length : wp.array
        ``float32`` per-prop extent: effective arc step (points mode) or
        chord length (segments mode).
    count : wp.array
        ``[E]`` ``int32`` real prop counts (0 for degenerate envs).
    truncated : wp.array
        ``[E]`` ``int32`` — 1 if ``max_props`` clipped this env's ring (the
        ring still closes, at a coarser effective spacing).
    step : wp.array
        ``[E]`` ``float32`` effective arc spacing ``perimeter / count``.
    """

    position: wp.array
    tangent: wp.array
    yaw: wp.array
    length: wp.array
    count: wp.array
    truncated: wp.array
    step: wp.array

    def clone(self) -> "PropSet":
        """Return a deep copy whose Warp buffers do not alias this set."""
        return PropSet(
            position=wp.clone(self.position),
            tangent=wp.clone(self.tangent),
            yaw=wp.clone(self.yaw),
            length=wp.clone(self.length),
            count=wp.clone(self.count),
            truncated=wp.clone(self.truncated),
            step=wp.clone(self.step),
        )


@wp.kernel
def _scan_boundary_k(
    points: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    spacing: float,
    max_props: int,
    cum: wp.array(dtype=wp.float32),
    out_count: wp.array(dtype=wp.int32),
    out_step: wp.array(dtype=wp.float32),
    out_truncated: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        out_count[e] = 0
        out_step[e] = wp.nan
        out_truncated[e] = 0
        return
    base = e * n_max
    s = float(0.0)
    cum[base] = 0.0
    prev = points[base]
    for i in range(1, m):
        p = points[base + i]
        s = s + wp.length(p - prev)
        cum[base + i] = s
        prev = p
    perim = s + wp.length(points[base] - prev)  # closing edge back to point 0

    n = int(wp.round(perim / spacing))
    if n < 3:
        n = 3
    trunc = int(0)
    if n > max_props:
        n = max_props
        trunc = int(1)
    out_count[e] = n
    out_step[e] = perim / float(n)
    out_truncated[e] = trunc


@wp.func
def _sample_at_arc(points: wp.array(dtype=wp.vec2f), cum: wp.array(dtype=wp.float32),
                   base: int, m: int, perim: float, s: float) -> wp.vec4f:
    """Point and segment direction at arc position s. Returns (px, py, dx, dy).

    Binary search for the largest i in [0, m-1] with cum[base+i] <= s, then
    lerp within segment i -> (i+1) mod m (the last segment closes the loop
    and ends at arc length ``perim``).
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
    j = i + 1
    seg_end = perim
    if j < m:
        seg_end = cum[base + j]
    else:
        j = 0
    seg_start = cum[base + i]
    denom = seg_end - seg_start
    t = 0.0
    if denom > 1.0e-12:
        t = wp.clamp((s - seg_start) / denom, 0.0, 1.0)
    a = points[base + i]
    b = points[base + j]
    p = a + (b - a) * t
    d = _safe_normalize2(b - a)
    return wp.vec4f(p[0], p[1], d[0], d[1])


@wp.kernel
def _place_props_k(
    points: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    cum: wp.array(dtype=wp.float32),
    prop_count: wp.array(dtype=wp.int32),
    prop_step: wp.array(dtype=wp.float32),
    max_props: int,
    mode: int,  # 0 = points, 1 = segments
    out_position: wp.array(dtype=wp.vec2f),
    out_tangent: wp.array(dtype=wp.vec2f),
    out_yaw: wp.array(dtype=wp.float32),
    out_length: wp.array(dtype=wp.float32),
):
    t = wp.tid()
    e = t // max_props
    k = t - e * max_props

    n = prop_count[e]
    if k >= n:
        nan2 = wp.vec2f(wp.nan, wp.nan)
        out_position[t] = nan2
        out_tangent[t] = nan2
        out_yaw[t] = wp.nan
        out_length[t] = wp.nan
        return

    m = count[e]
    if m > n_max:
        m = n_max
    base = e * n_max
    step = prop_step[e]
    perim = step * float(n)

    if mode == 0:
        s = float(k) * step
        smp = _sample_at_arc(points, cum, base, m, perim, s)
        tang = wp.vec2f(smp[2], smp[3])
        out_position[t] = wp.vec2f(smp[0], smp[1])
        out_tangent[t] = tang
        out_yaw[t] = wp.atan2(tang[1], tang[0])
        out_length[t] = step
    else:
        s0 = float(k) * step
        s1 = 0.0  # slot n-1 chords back to sample 0: the ring closes
        if k + 1 < n:
            s1 = float(k + 1) * step
        a4 = _sample_at_arc(points, cum, base, m, perim, s0)
        b4 = _sample_at_arc(points, cum, base, m, perim, s1)
        p0 = wp.vec2f(a4[0], a4[1])
        p1 = wp.vec2f(b4[0], b4[1])
        chord = p1 - p0
        tang = _safe_normalize2(chord)
        out_position[t] = (p0 + p1) * 0.5
        out_tangent[t] = tang
        out_yaw[t] = wp.atan2(tang[1], tang[0])
        out_length[t] = wp.length(chord)


class PropSampler:
    """Resample one track boundary into instancing poses at a set spacing.

    One sampler binds one :class:`Track`, one boundary curve, one mode, and
    one spacing, so all output shapes are fixed and ``sample()`` is
    CUDA-graph capturable (allocation-free; host-sync-free while the module
    ``_CAPTURING`` flag is set). Because ``TrackGenerator.generate()``
    overwrites its ``Track`` buffers in place, every ``sample()`` reads the
    CURRENT batch — no rebind after regeneration. Props are rendering-only;
    they are not colliders.
    """

    def __init__(self, track: Track, spacing: float, boundary: str = "outer",
                 mode: str = "points", max_props: "int | None" = None) -> None:
        _init()
        if float(spacing) <= 0.0:
            raise ValueError(f"spacing must be > 0, got {spacing!r}")
        if boundary not in ("inner", "outer"):
            raise ValueError(
                f"boundary must be one of {{'inner', 'outer'}}, got {boundary!r}")
        if mode not in ("points", "segments"):
            raise ValueError(
                f"mode must be one of {{'points', 'segments'}}, got {mode!r}")
        if max_props is not None and int(max_props) < 3:
            raise ValueError(
                f"max_props must be >= 3 (or None for auto), got {max_props!r}")
        E = int(track.count.shape[0])
        stride = int(track.outer.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"track batch layout invalid: outer has {stride} slots for {E} envs")
        self._track = track
        self._spacing = float(spacing)
        self._boundary = boundary
        self._mode = mode
        self._mode_int = 0 if mode == "points" else 1
        self._E = E
        self._n_max = stride // E
        self._points = track.inner if boundary == "inner" else track.outer
        self._device = str(track.outer.device)

        if max_props is None:
            max_props = self._derive_max_props()
        self._M = int(max_props)

        dev = self._device
        n = E * self._M
        self._cum = wp.zeros(E * self._n_max, dtype=wp.float32, device=dev)
        self._props = PropSet(
            position=wp.zeros(n, dtype=wp.vec2f, device=dev),
            tangent=wp.zeros(n, dtype=wp.vec2f, device=dev),
            yaw=wp.zeros(n, dtype=wp.float32, device=dev),
            length=wp.zeros(n, dtype=wp.float32, device=dev),
            count=wp.zeros(E, dtype=wp.int32, device=dev),
            truncated=wp.zeros(E, dtype=wp.int32, device=dev),
            step=wp.zeros(E, dtype=wp.float32, device=dev),
        )

    def _derive_max_props(self) -> int:
        """Slots to cover the batch bound NOW: ceil(1.5 * max perimeter / spacing).

        Host-side readback, construction time only. The 1.5x headroom absorbs
        longer boundaries from future regenerations; rings that still exceed
        it are truncated and flagged (``PropSet.truncated``).
        """
        E, n_max = self._E, self._n_max
        pts = self._points.numpy().reshape(E, n_max, 2)
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
                "max_props=None needs at least one valid env in the bound track "
                "batch to derive a buffer size; pass max_props explicitly")
        return max(3, int(np.ceil(1.5 * best / self._spacing)))

    def sample(self) -> PropSet:
        """Resample the bound boundary; returns the preallocated :class:`PropSet`.

        Two kernel launches (scan + place); reads the Track buffers as they
        are NOW, so it reflects the latest ``generate()`` with no rebind.
        """
        t = self._track
        p = self._props
        wp.launch(
            _scan_boundary_k, dim=self._E,
            inputs=[self._points, t.count, self._n_max, self._spacing, self._M,
                    self._cum, p.count, p.step, p.truncated],
            device=self._device,
        )
        wp.launch(
            _place_props_k, dim=self._E * self._M,
            inputs=[self._points, t.count, self._n_max, self._cum,
                    p.count, p.step, self._M, self._mode_int,
                    p.position, p.tangent, p.yaw, p.length],
            device=self._device,
        )
        _sync(self._device)
        return p
```

- [ ] **Step 4: Create the public shim and wire the package**

Create `track_gen/props.py`:

```python
"""Public boundary prop-sampling API: instancing poses along track boundaries.

``PropSampler`` resamples the inner or outer boundary at a set spacing into a
:class:`PropSet` of per-prop poses (position, tangent, yaw, length) for
rendering-only instancing — cone lines (``mode="points"``) or wall pieces
(``mode="segments"``). The complement of ``track_gen.collision``: these props
never collide; use ``CollisionChecker`` for out-of-bounds queries.
"""
from ._src.props import PropSampler, PropSet

__all__ = ["PropSampler", "PropSet"]
```

Modify `track_gen/__init__.py` — extend the imports and `__all__`:

```python
from ._version import __version__
from ._src.types import GateGenConfig, GateSequence, Track, TrackGenConfig
from ._src.track_generator import TrackGenerator
from ._src.gate_generator import GateGenerator
from ._src.rng_utils import PerEnvSeededRNG
from . import collision
from . import props

__all__ = [
    "TrackGenerator",
    "TrackGenConfig",
    "Track",
    "GateGenerator",
    "GateGenConfig",
    "GateSequence",
    "PerEnvSeededRNG",
    "collision",
    "props",
    "__version__",
]
```

Modify `tests/test_public_api.py`: add `"props"` to the hardcoded expected `__all__` set (exactly as `"collision"` appears there).

- [ ] **Step 5: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_props.py tests/test_public_api.py -v`
Expected: all PASS (first run compiles the kernels)

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/props.py track_gen/props.py track_gen/__init__.py tests/test_public_api.py tests/test_props.py
git commit -m "feat: track_gen.props — boundary prop sampling (points mode + core)"
```

---

### Task 2: Segments mode, inner boundary, truncation, degenerate envs, max_props derivation

**Files:**
- Modify: `tests/test_props.py` (append tests)
- Modify (only if a test exposes a kernel bug): `track_gen/_src/props.py`

**Interfaces:**
- Consumes: Task 1's `PropSampler`/`PropSet` exactly as produced.
- Produces: verified segments-mode + edge-case semantics that Tasks 3–4 rely on.

- [ ] **Step 1: Append the tests**

Append to `tests/test_props.py`:

```python
def test_segments_mode_chords_and_closure():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="segments")
    props = sampler.sample()
    n = int(props.count.numpy()[0])
    pos = props.position.numpy().reshape(-1, 2)[:n]
    tang = props.tangent.numpy().reshape(-1, 2)[:n]
    length = props.length.numpy()[:n]
    # Chord across one arc step of the circle: 2*R*sin(pi/n).
    np.testing.assert_allclose(length, 2 * RO * np.sin(np.pi / n), atol=2e-3)
    # Chord midpoints sit slightly inside the circle: radius R*cos(pi/n).
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1),
                               RO * np.cos(np.pi / n), atol=2e-3)
    # Ring closure: each chord's end == next chord's start (wraps at n-1 -> 0).
    starts = pos - tang * (length[:, None] / 2.0)
    ends = pos + tang * (length[:, None] / 2.0)
    np.testing.assert_allclose(ends, np.roll(starts, -1, axis=0), atol=1e-4)


def test_inner_boundary_sampling():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    props = PropSampler(track, spacing=0.1, boundary="inner", mode="points").sample()
    n = int(props.count.numpy()[0])
    pos = props.position.numpy().reshape(-1, 2)[:n]
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), RI, atol=2e-3)


def test_truncation_flag_and_closed_ring():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points",
                          max_props=10)
    props = sampler.sample()
    assert int(props.count.numpy()[0]) == 10
    assert int(props.truncated.numpy()[0]) == 1
    perim = _outer_perimeter(track)
    np.testing.assert_allclose(props.step.numpy()[0], perim / 10, rtol=1e-5)
    # Still a closed uniform ring at the coarser effective spacing.
    pos = props.position.numpy().reshape(-1, 2)[:10]
    ang = np.arctan2(pos[:, 1], pos[:, 0])
    gaps = np.mod(np.diff(np.concatenate([ang, ang[:1]])), 2 * np.pi)
    np.testing.assert_allclose(gaps, 2 * np.pi / 10, atol=5e-3)


def test_degenerate_env_zero_count_nan():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=2, n=N, counts=[N, 2])  # env 1 degenerate
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode="points",
                          max_props=128)
    props = sampler.sample()
    counts = props.count.numpy()
    assert counts[0] > 0 and counts[1] == 0
    assert np.isnan(props.step.numpy()[1])
    M = sampler._M
    pos = props.position.numpy().reshape(-1, 2)
    assert np.all(np.isnan(pos[M:2 * M]))  # env 1 slots all NaN


def test_max_props_auto_derivation():
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    spacing = 0.1
    sampler = PropSampler(track, spacing=spacing, boundary="outer", mode="points")
    perim = _outer_perimeter(track)
    assert sampler._M == max(3, int(np.ceil(1.5 * perim / spacing)))
    assert int(sampler.sample().truncated.numpy()[0]) == 0


def test_max_props_derivation_requires_valid_env():
    import warp as wp
    import pytest
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=N)
    wp.copy(track.valid, wp.zeros(1, dtype=wp.int32, device="cpu"))
    with pytest.raises(ValueError, match="max_props"):
        PropSampler(track, spacing=0.1)
    # Explicit max_props still works with no valid env.
    PropSampler(track, spacing=0.1, max_props=64)


def test_constructor_validation():
    import pytest
    from track_gen.props import PropSampler
    track = make_annulus_track(E=1, n=64)
    with pytest.raises(ValueError, match="spacing"):
        PropSampler(track, spacing=0.0)
    with pytest.raises(ValueError, match="boundary"):
        PropSampler(track, spacing=0.1, boundary="center")
    with pytest.raises(ValueError, match="mode"):
        PropSampler(track, spacing=0.1, mode="walls")
    with pytest.raises(ValueError, match="max_props"):
        PropSampler(track, spacing=0.1, max_props=2)
```

- [ ] **Step 2: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_props.py -v`
Expected: all PASS. A failure means a real kernel/constructor bug (likely
spots: the closing-edge handling in `_sample_at_arc` when `j == m`, the
`s1 = 0.0` wraparound for slot `n-1`, or the derivation loop's valid/count
guards). Fix `track_gen/_src/props.py` minimally; do not loosen tolerances.

- [ ] **Step 3: Commit**

```bash
git add tests/test_props.py track_gen/_src/props.py
git commit -m "test: segments mode, truncation, degenerate and derivation coverage for props"
```

---

### Task 3: Contract tests + numpy oracle on generated tracks

**Files:**
- Create: `tests/_props_oracle.py`
- Test: `tests/test_props_contract.py`

**Interfaces:**
- Consumes: Task 1–2 `PropSampler`/`PropSet`; `TrackGenerator`/`PerEnvSeededRNG`.
- Produces: `_props_oracle.sample_boundary(poly, spacing, max_props) -> (pos, tang, n, step, truncated)` — independent numpy reference used by the property test.

- [ ] **Step 1: Write the oracle**

Create `tests/_props_oracle.py`:

```python
"""Independent numpy reference for boundary prop sampling (tests only)."""
from __future__ import annotations

import numpy as np


def sample_boundary(poly, spacing, max_props):
    """Points-mode reference: closed-polyline arc resample with snap rule.

    Returns (positions [n,2], tangents [n,2], n, step, truncated).
    """
    poly = np.asarray(poly, dtype=np.float64)
    seg = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])  # cum[m] == perimeter
    perim = float(cum[-1])
    n = int(np.clip(round(perim / spacing), 3, max_props))
    truncated = int(round(perim / spacing) > max_props)
    step = perim / n
    s = np.arange(n) * step
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(poly) - 1)
    t = (s - cum[idx]) / np.maximum(seg[idx], 1e-12)
    p0 = poly[idx]
    p1 = poly[(idx + 1) % len(poly)]
    pos = p0 + (p1 - p0) * t[:, None]
    d = p1 - p0
    tang = d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]
    return pos, tang, n, step, truncated
```

- [ ] **Step 2: Write the contract tests**

Create `tests/test_props_contract.py`:

```python
"""PropSampler contracts: in-place reuse, clone, aliasing, oracle property test."""
from __future__ import annotations

import numpy as np
import warp as wp

from tests._collision_fixtures import make_annulus_track
from tests._props_oracle import sample_boundary
from track_gen.props import PropSampler, PropSet

N = 512


def test_sample_returns_same_instance_and_clone_detaches():
    track = make_annulus_track(E=1, n=N)
    sampler = PropSampler(track, spacing=0.1)
    p1 = sampler.sample()
    snap = p1.clone()
    assert isinstance(snap, PropSet)
    n_before = int(p1.count.numpy()[0])
    pos_before = snap.position.numpy().copy()
    # Mutate the bound track (same buffers, as generate() would) and resample.
    bigger = make_annulus_track(E=1, n=N, r_center=2.0)
    wp.copy(track.inner, bigger.inner)
    wp.copy(track.outer, bigger.outer)
    p2 = sampler.sample()
    assert p2 is p1
    assert int(p1.count.numpy()[0]) > n_before  # longer boundary -> more props
    np.testing.assert_allclose(snap.position.numpy(), pos_before)  # snapshot intact


def test_matches_oracle_on_generated_tracks():
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=123, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    counts = track.count.numpy()
    n_max = track.outer.shape[0] // E
    spacing = 0.07
    sampler = PropSampler(track, spacing=spacing, boundary="outer", mode="points",
                          max_props=512)
    props = sampler.sample()
    outer = track.outer.numpy().reshape(E, n_max, 2)
    M = sampler._M
    checked = 0
    for e in range(E):
        if not valid[e]:
            continue
        poly = outer[e, :int(counts[e])]
        ref_pos, ref_tang, ref_n, ref_step, ref_trunc = sample_boundary(
            poly, spacing, 512)
        assert int(props.count.numpy()[e]) == ref_n
        np.testing.assert_allclose(props.step.numpy()[e], ref_step, rtol=1e-4)
        got = props.position.numpy().reshape(-1, 2)[e * M:e * M + ref_n]
        np.testing.assert_allclose(got, ref_pos, atol=1e-4,
                                   err_msg=f"env {e} positions")
        got_t = props.tangent.numpy().reshape(-1, 2)[e * M:e * M + ref_n]
        np.testing.assert_allclose(got_t, ref_tang, atol=1e-3,
                                   err_msg=f"env {e} tangents")
        checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"


def test_segments_are_chords_of_points():
    # segments mode must equal chords between consecutive points-mode samples.
    track = make_annulus_track(E=1, n=N)
    pts = PropSampler(track, spacing=0.15, mode="points").sample().clone()
    segs = PropSampler(track, spacing=0.15, mode="segments").sample()
    n = int(pts.count.numpy()[0])
    assert int(segs.count.numpy()[0]) == n
    p = pts.position.numpy().reshape(-1, 2)[:n]
    p_next = np.roll(p, -1, axis=0)
    np.testing.assert_allclose(segs.position.numpy().reshape(-1, 2)[:n],
                               (p + p_next) / 2, atol=1e-4)
    np.testing.assert_allclose(segs.length.numpy()[:n],
                               np.linalg.norm(p_next - p, axis=1), atol=1e-4)
```

- [ ] **Step 3: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_props_contract.py -v`
Expected: 3 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/_props_oracle.py tests/test_props_contract.py
git commit -m "test: PropSampler contracts and generated-track oracle comparison"
```

---

### Task 4: CUDA graph capture test, docs, final verification

**Files:**
- Test: `tests/test_props_cuda_graph.py`
- Modify: `docs/reference/api.rst`

**Interfaces:**
- Consumes: `track_gen._src.props._CAPTURING`, `PropSampler`, fixtures.

- [ ] **Step 1: Write the CUDA graph test (poisoned-buffer replay, from the start)**

Create `tests/test_props_cuda_graph.py`:

```python
"""CUDA-only: PropSampler.sample() inside wp.ScopedCapture.

Buffers are poisoned between capture and replay so the comparison proves the
captured graph recomputes results (not stale pre-capture buffer contents).
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
from tests._collision_fixtures import make_annulus_track  # noqa: E402
from track_gen._src import props as props_mod  # noqa: E402
from track_gen.props import PropSampler  # noqa: E402

DEV = "cuda:0"


@pytest.mark.parametrize("mode", ["points", "segments"])
def test_sample_graph_replay_matches_eager(mode):
    track = make_annulus_track(E=4, n=256, device=DEV)
    sampler = PropSampler(track, spacing=0.1, boundary="outer", mode=mode)
    eager = sampler.sample().clone()

    prev = props_mod._CAPTURING
    props_mod._CAPTURING = True
    try:
        sampler.sample()  # warmup: modules loaded before capture
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            sampler.sample()
    finally:
        props_mod._CAPTURING = prev

    # Poison outputs so the comparison proves the REPLAY recomputed them.
    replay = sampler._props
    replay.position.fill_(12345.0)
    replay.count.fill_(-7)
    replay.step.fill_(12345.0)

    wp.capture_launch(cap.graph)
    wp.synchronize()

    np.testing.assert_array_equal(replay.count.numpy(), eager.count.numpy())
    np.testing.assert_allclose(replay.step.numpy(), eager.step.numpy(),
                               rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(replay.position.numpy(), eager.position.numpy(),
                               rtol=1e-5, atol=1e-6, equal_nan=True)
```

- [ ] **Step 2: Run it (GPU present on this machine)**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_props_cuda_graph.py -v`
Expected: 2 PASS (on a no-GPU machine: 2 SKIPPED)

- [ ] **Step 3: Add the docs section**

In `docs/reference/api.rst`, append after the collision section's
``Reproduce`` block (end of file):

```rst
Boundary props
--------------

Rendering-only instancing poses along track boundaries — cone lines
(``mode="points"``) and wall pieces (``mode="segments"``). Complementary to
the collision utility: props never collide.

.. automodule:: track_gen.props
   :no-members:

.. autoclass:: track_gen.props.PropSampler
   :members:

.. autoclass:: track_gen.props.PropSet
   :no-members:

   .. automethod:: clone
```

- [ ] **Step 4: Build docs and run the full suite**

Run: `python3 -m sphinx -b html docs /tmp/claude-1000/-home-antoine-Documents-track-gen/a3819d36-c82d-4063-bcba-b7abbecf061d/scratchpad/docs-build -q 2>&1 | tail -3`
Expected: no new warnings mentioning `props`.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all PASS (GPU present: cuda tests run too)

- [ ] **Step 5: Commit**

```bash
git add tests/test_props_cuda_graph.py docs/reference/api.rst
git commit -m "test+docs: props CUDA graph capture and API reference section"
```

---

## Self-Review Notes (completed during planning)

- **Spec coverage:** API + PropSet fields + validation (Task 1), snap rule / modes / degenerate / truncation / derivation semantics (Tasks 1–2), scan+place kernels with binary search (Task 1), file layout + `__init__` + public-api test (Task 1), analytic annulus + oracle + contract tests (Tasks 1–3), CUDA graph with poisoned replay (Task 4), docs (Task 4). Out-of-scope list respected (no offsets, no 3D, no jitter).
- **Type consistency:** `PropSet` field names (`position, tangent, yaw, length, count, truncated, step`), `PropSampler` internals (`_M`, `_props`, `_cum`, `_mode_int`) used identically across tasks; kernel input order matches both `wp.launch` calls.
- **Known conventions carried over from the collision build:** `float(0.0)` literal seeds in dynamic loops (safe), `fill_` for poisoning, `_CAPTURING` module flag, `wp.launch(..., inputs=)` only.
