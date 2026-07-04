# Collision / Out-of-Bounds Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `track_gen.collision` — a Warp-native, GPU-batched utility that reports, per oriented box, whether it left the drivable band of a track, with full contact info (OOB flag, signed clearance, nearest boundary point, normal, boundary id), via two backends: exact `segments` scan and baked `sdf` grid.

**Architecture:** A `CollisionChecker` facade binds to a `Track` batch and preallocates a `BoxContact` output struct (flat `[E*max_boxes]` wp.arrays, overwritten in place each `query()`, same contract as `Track`). The `segments` backend is one kernel, one thread per box, looping over ≤ 2·`count[e]` boundary segments. The `sdf` backend bakes per-env signed-distance + boundary-id grids (`bake()`), then queries with 5 bilinear samples per box. Shared `@wp.func` geometry lives in a leaf module `collision_geom.py`.

**Tech Stack:** Python ≥ 3.10, NVIDIA Warp 1.0.1 (`warp-lang`), numpy. Tests: pytest (+ torch only for the CUDA-marked test).

**Spec:** `docs/superpowers/specs/2026-07-04-collision-utility-design.md`

## Global Constraints

- Runtime deps are **numpy + warp-lang only** (pyproject). No scipy/shapely/torch in `track_gen/` code.
- Warp **1.0.1** compatibility: no `wp.isnan` (use `x != x`), no `wp.launch(..., outputs=)` (pass everything via `inputs=`).
- All batched arrays are **flat** `[E * stride]` wp.arrays (vec2f / float32 / int32), NaN-padded past `count[e]` — match `Track` conventions.
- `query()` and `bake()` must be CUDA-graph capturable: **no allocation, no host sync** inside them (module `_CAPTURING` flag suppresses `wp.synchronize`, same pattern as `warp_gate.py`).
- In-place output contract: `query()` returns the SAME `BoxContact` every call; `clone()` for snapshots.
- OOB semantics (spec): band = inside outer ∧ outside inner; box OOB iff any corner outside band ∨ any boundary segment intersects the box; `distance` = +min box↔boundary distance when inside, −(deepest corner penetration) when OOB; `normal` points into the band; `boundary` 0=inner, 1=outer, −1 for inactive/degenerate; NaN position ⇒ NaN outputs and `oob=0`.
- Every pytest run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest …` (repo convention — ROS plugin leak).
- Kernel style: module-level `_INITED`/`_init()`/`_CAPTURING`/`_sync(device)` helpers, `@wp.func` for shared math, flat tid decomposition `e = t // stride`.

**Warp 1.0.1 contingency:** the kernels below use `wp.vec4f` locals with component assignment inside `for k in range(4):` loops (constant range ⇒ unrolled ⇒ constant indices). If module load raises a Warp codegen error on those assignments, mechanically replace each vec4 accumulator with four named float locals (`cn_in0..cn_in3`, etc.) and expand the `for k in range(4)` bodies once per corner — no other change.

---

### Task 1: Shared Warp geometry helpers (`collision_geom.py`)

**Files:**
- Create: `track_gen/_src/collision_geom.py`
- Test: `tests/test_collision_geom.py`

**Interfaces:**
- Consumes: nothing (leaf module; imports only `warp`).
- Produces `@wp.func`s used by Tasks 3 and 6 (exact signatures):
  - `_is_nan2(v: wp.vec2f) -> int` — 1 if either component is NaN
  - `_safe_normalize2(v: wp.vec2f) -> wp.vec2f`
  - `_rot2(yaw: float, v: wp.vec2f) -> wp.vec2f` — rotate v by yaw
  - `_box_corner(center: wp.vec2f, ux: wp.vec2f, uy: wp.vec2f, he: wp.vec2f, k: int) -> wp.vec2f` — corner k∈{0..3}, CCW order (+,+), (−,+), (−,−), (+,−)
  - `_pick4(c0, c1, c2, c3: wp.vec2f, k: int) -> wp.vec2f`
  - `_closest_on_seg(p: wp.vec2f, a: wp.vec2f, b: wp.vec2f) -> wp.vec2f`
  - `_crossing(p: wp.vec2f, a: wp.vec2f, b: wp.vec2f) -> int` — +x-ray crossing parity contribution (half-open rule)
  - `_point_to_local_box_dist(q: wp.vec2f, he: wp.vec2f) -> float` — distance from box-local point to solid AABB `[-he, he]`, 0 inside
  - `_seg_hits_aabb(a: wp.vec2f, b: wp.vec2f, he: wp.vec2f) -> int` — slab test, segment in box-local coords vs solid AABB

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collision_geom.py`:

```python
"""Kernel-wrapper tests for the shared collision geometry @wp.funcs."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from track_gen._src import collision_geom as cg

wp.init()


@wp.kernel
def _k_closest(p: wp.array(dtype=wp.vec2f), a: wp.vec2f, b: wp.vec2f,
               out: wp.array(dtype=wp.vec2f)):
    i = wp.tid()
    out[i] = cg._closest_on_seg(p[i], a, b)


@wp.kernel
def _k_crossing(p: wp.array(dtype=wp.vec2f), a: wp.vec2f, b: wp.vec2f,
                out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = cg._crossing(p[i], a, b)


@wp.kernel
def _k_box_dist(q: wp.array(dtype=wp.vec2f), he: wp.vec2f,
                out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = cg._point_to_local_box_dist(q[i], he)


@wp.kernel
def _k_seg_hit(a: wp.array(dtype=wp.vec2f), b: wp.array(dtype=wp.vec2f),
               he: wp.vec2f, out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = cg._seg_hits_aabb(a[i], b[i], he)


@wp.kernel
def _k_corners(center: wp.vec2f, yaw: float, he: wp.vec2f,
               out: wp.array(dtype=wp.vec2f)):
    i = wp.tid()
    ux = cg._rot2(yaw, wp.vec2f(1.0, 0.0))
    uy = cg._rot2(yaw, wp.vec2f(0.0, 1.0))
    out[i] = cg._box_corner(center, ux, uy, he, i)


@wp.kernel
def _k_nan2(p: wp.array(dtype=wp.vec2f), out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = cg._is_nan2(p[i])


def _run(kernel, n, inputs):
    wp.launch(kernel, dim=n, inputs=inputs, device="cpu")


def test_closest_on_seg_projects_and_clamps():
    pts = wp.array(np.array([[0.5, 1.0], [-2.0, 1.0], [5.0, -3.0]], np.float32),
                   dtype=wp.vec2f, device="cpu")
    out = wp.zeros(3, dtype=wp.vec2f, device="cpu")
    _run(_k_closest, 3, [pts, wp.vec2f(0.0, 0.0), wp.vec2f(1.0, 0.0), out])
    got = out.numpy()
    np.testing.assert_allclose(got[0], [0.5, 0.0], atol=1e-6)  # interior projection
    np.testing.assert_allclose(got[1], [0.0, 0.0], atol=1e-6)  # clamped to a
    np.testing.assert_allclose(got[2], [1.0, 0.0], atol=1e-6)  # clamped to b


def test_crossing_parity_square():
    # Unit square CCW; point inside crosses exactly one of the 4 edges.
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    p_in = np.array([[0.5, 0.5]], np.float32)
    p_out = np.array([[1.5, 0.5]], np.float32)
    for p, expected in ((p_in, 1), (p_out, 0)):
        total = 0
        for i in range(4):
            a, b = square[i], square[(i + 1) % 4]
            pts = wp.array(p, dtype=wp.vec2f, device="cpu")
            out = wp.zeros(1, dtype=wp.int32, device="cpu")
            _run(_k_crossing, 1, [pts, wp.vec2f(*a), wp.vec2f(*b), out])
            total += int(out.numpy()[0])
        assert total % 2 == expected


def test_point_to_local_box_dist():
    q = wp.array(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]], np.float32),
                 dtype=wp.vec2f, device="cpu")
    out = wp.zeros(3, dtype=wp.float32, device="cpu")
    _run(_k_box_dist, 3, [q, wp.vec2f(1.0, 1.0), out])
    got = out.numpy()
    assert got[0] == 0.0                       # inside
    np.testing.assert_allclose(got[1], 2.0, atol=1e-6)   # face
    np.testing.assert_allclose(got[2], np.hypot(2.0, 3.0), atol=1e-6)  # corner


def test_seg_hits_aabb():
    a = np.array([[-2.0, 0.0], [-2.0, 2.0], [0.2, 0.2], [-2.0, 1.5]], np.float32)
    b = np.array([[2.0, 0.0], [2.0, 2.0], [0.3, 0.1], [1.5, -2.0]], np.float32)
    aw = wp.array(a, dtype=wp.vec2f, device="cpu")
    bw = wp.array(b, dtype=wp.vec2f, device="cpu")
    out = wp.zeros(4, dtype=wp.int32, device="cpu")
    _run(_k_seg_hit, 4, [aw, bw, wp.vec2f(1.0, 1.0), out])
    # through the box; passing above; fully inside; clipping a corner
    assert list(out.numpy()) == [1, 0, 1, 1]


def test_box_corners_rotated():
    out = wp.zeros(4, dtype=wp.vec2f, device="cpu")
    _run(_k_corners, 4, [wp.vec2f(1.0, 2.0), float(np.pi / 2.0), wp.vec2f(0.3, 0.1), out])
    got = out.numpy()
    # yaw=90deg: box-frame +x becomes world +y.  Corner 0 = c + 0.3*uy_world... :
    # ux=(0,1), uy=(-1,0) => corner0 = (1,2) + 0.3*(0,1) + 0.1*(-1,0) = (0.9, 2.3)
    np.testing.assert_allclose(got[0], [0.9, 2.3], atol=1e-6)
    np.testing.assert_allclose(got[2], [1.1, 1.7], atol=1e-6)  # opposite corner


def test_is_nan2():
    p = wp.array(np.array([[np.nan, 0.0], [0.0, np.nan], [1.0, 1.0]], np.float32),
                 dtype=wp.vec2f, device="cpu")
    out = wp.zeros(3, dtype=wp.int32, device="cpu")
    _run(_k_nan2, 3, [p, out])
    assert list(out.numpy()) == [1, 1, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_geom.py -v`
Expected: FAIL at import — `ModuleNotFoundError: track_gen._src.collision_geom`

- [ ] **Step 3: Write the implementation**

Create `track_gen/_src/collision_geom.py`:

```python
"""Shared pure-Warp geometry helpers for collision queries.

Leaf module (imports only warp): used by the segments backend in
``collision.py`` and the SDF backend in ``collision_sdf.py``. All helpers are
``@wp.func`` device functions; nothing here launches kernels.
"""
from __future__ import annotations

import warp as wp


@wp.func
def _is_nan2(v: wp.vec2f) -> int:
    # NaN != NaN; wp.isnan does not exist in warp 1.0.
    if v[0] != v[0] or v[1] != v[1]:
        return int(1)
    return int(0)


@wp.func
def _safe_normalize2(v: wp.vec2f) -> wp.vec2f:
    return v / wp.max(wp.length(v), 1.0e-8)


@wp.func
def _rot2(yaw: float, v: wp.vec2f) -> wp.vec2f:
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    return wp.vec2f(c * v[0] - s * v[1], s * v[0] + c * v[1])


@wp.func
def _box_corner(center: wp.vec2f, ux: wp.vec2f, uy: wp.vec2f,
                he: wp.vec2f, k: int) -> wp.vec2f:
    # CCW corner order in the box frame: (+,+), (-,+), (-,-), (+,-).
    sx = 1.0
    sy = 1.0
    if k == 1 or k == 2:
        sx = -1.0
    if k == 2 or k == 3:
        sy = -1.0
    return center + ux * (sx * he[0]) + uy * (sy * he[1])


@wp.func
def _pick4(c0: wp.vec2f, c1: wp.vec2f, c2: wp.vec2f, c3: wp.vec2f,
           k: int) -> wp.vec2f:
    if k == 0:
        return c0
    if k == 1:
        return c1
    if k == 2:
        return c2
    return c3


@wp.func
def _closest_on_seg(p: wp.vec2f, a: wp.vec2f, b: wp.vec2f) -> wp.vec2f:
    ab = b - a
    denom = wp.dot(ab, ab)
    t = 0.0
    if denom > 1.0e-12:
        t = wp.clamp(wp.dot(p - a, ab) / denom, 0.0, 1.0)
    return a + ab * t


@wp.func
def _crossing(p: wp.vec2f, a: wp.vec2f, b: wp.vec2f) -> int:
    """1 if the +x ray from p crosses segment ab (half-open rule), else 0."""
    if (a[1] > p[1]) != (b[1] > p[1]):
        x_hit = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        if p[0] < x_hit:
            return int(1)
    return int(0)


@wp.func
def _point_to_local_box_dist(q: wp.vec2f, he: wp.vec2f) -> float:
    """Distance from a box-local point to the solid AABB [-he, he]; 0 inside."""
    dx = wp.max(wp.abs(q[0]) - he[0], 0.0)
    dy = wp.max(wp.abs(q[1]) - he[1], 0.0)
    return wp.sqrt(dx * dx + dy * dy)


@wp.func
def _seg_hits_aabb(a: wp.vec2f, b: wp.vec2f, he: wp.vec2f) -> int:
    """1 if segment ab (box-local coords) intersects the solid AABB [-he, he].

    Liang-Barsky slab clip of the parametric segment; covers endpoint-inside,
    pass-through, and corner-clip cases.
    """
    d = b - a
    tmin = 0.0
    tmax = 1.0
    for axis in range(2):
        av = a[axis]
        dv = d[axis]
        hv = he[axis]
        if wp.abs(dv) < 1.0e-12:
            if av < -hv or av > hv:
                return int(0)
        else:
            t1 = (-hv - av) / dv
            t2 = (hv - av) / dv
            tmin = wp.max(tmin, wp.min(t1, t2))
            tmax = wp.min(tmax, wp.max(t1, t2))
            if tmin > tmax:
                return int(0)
    return int(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_geom.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/collision_geom.py tests/test_collision_geom.py
git commit -m "feat: shared Warp geometry helpers for collision queries"
```

---

### Task 2: Test fixtures — annulus Track + numpy oracle

**Files:**
- Create: `tests/_collision_fixtures.py`
- Create: `tests/_collision_oracle.py`
- Test: `tests/test_collision_fixtures.py`

**Interfaces:**
- Consumes: `track_gen._src.types.Track`.
- Produces (used by Tasks 3–8):
  - `_collision_fixtures.make_annulus_track(E=1, n=512, N_max=None, r_center=1.0, half_width=0.3, counts=None, device="cpu") -> Track` — concentric-circle Track batch; `counts[e]` real points per env (default `n`); `N_max` defaults to `n + 8` so a NaN tail exists; CCW winding; `normal` = radial outward.
  - `_collision_fixtures.make_boxes(E, B, slots, device="cpu") -> (position, yaw, half_extents)` — wp.arrays `[E*B]` (vec2f, float32, vec2f); `slots` is `{(e, b): (px, py, yaw, hx, hy)}`; unset slots get NaN position (inactive).
  - `_collision_oracle.box_contact(inner, outer, pos, yaw, he) -> dict` with keys `oob` (int), `distance` (float), `nearest` (np.ndarray shape (2,)), `boundary` (int) — same semantics as the spec; `inner`/`outer` are `[m, 2]` numpy polylines (real points only).
  - `_collision_oracle.point_in_poly(p, poly) -> bool`, `_collision_oracle.point_polyline_dist(p, poly) -> (float, np.ndarray)`.

- [ ] **Step 1: Write the fixture and oracle modules**

Create `tests/_collision_fixtures.py`:

```python
"""Synthetic annulus Track + box-input builders for collision tests."""
from __future__ import annotations

import numpy as np
import warp as wp

from track_gen._src.types import Track


def make_annulus_track(E=1, n=512, N_max=None, r_center=1.0, half_width=0.3,
                       counts=None, device="cpu"):
    """Concentric-circle Track batch: inner radius r-hw, outer r+hw, CCW."""
    wp.init()
    if N_max is None:
        N_max = n + 8  # NaN tail exercises count-aware kernels
    counts = [n] * E if counts is None else list(counts)
    assert len(counts) == E and max(counts) <= N_max
    ri, ro = r_center - half_width, r_center + half_width
    names = ("outer", "center", "inner", "tangent", "normal")
    fields = {k: np.full((E, N_max, 2), np.nan, np.float32) for k in names}
    arclen = np.full((E, N_max), np.nan, np.float32)
    length = np.zeros(E, np.float32)
    for e, m in enumerate(counts):
        th = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
        radial = np.stack([np.cos(th), np.sin(th)], axis=1)
        fields["center"][e, :m] = r_center * radial
        fields["outer"][e, :m] = ro * radial
        fields["inner"][e, :m] = ri * radial
        fields["tangent"][e, :m] = np.stack([-np.sin(th), np.cos(th)], axis=1)
        fields["normal"][e, :m] = radial
        step = 2.0 * r_center * np.sin(np.pi / m)  # chord length
        arclen[e, :m] = step * np.arange(m)
        length[e] = step * m

    def v2(a):
        return wp.array(a.reshape(-1, 2), dtype=wp.vec2f, device=device)

    return Track(
        outer=v2(fields["outer"]), center=v2(fields["center"]),
        inner=v2(fields["inner"]), tangent=v2(fields["tangent"]),
        normal=v2(fields["normal"]),
        arclen=wp.array(arclen.reshape(-1), dtype=wp.float32, device=device),
        length=wp.array(length, dtype=wp.float32, device=device),
        valid=wp.array(np.ones(E, np.int32), dtype=wp.int32, device=device),
        count=wp.array(np.array(counts, np.int32), dtype=wp.int32, device=device),
    )


def annulus_polylines(track, e, N_max):
    """Real (non-NaN) inner/outer polylines of env e as numpy [m, 2] arrays."""
    m = int(track.count.numpy()[e])
    inner = track.inner.numpy().reshape(-1, 2)[e * N_max:e * N_max + m]
    outer = track.outer.numpy().reshape(-1, 2)[e * N_max:e * N_max + m]
    return inner.astype(np.float64), outer.astype(np.float64)


def make_boxes(E, B, slots, device="cpu"):
    """Box input arrays [E*B]; slots {(e,b): (px, py, yaw, hx, hy)}; rest inactive."""
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
```

Create `tests/_collision_oracle.py`:

```python
"""Independent numpy reference implementation of the collision semantics.

Small, readable loops (slow but only used on tiny test batches). Mirrors the
spec: band = inside outer AND outside inner; OOB iff any corner outside band
or any boundary segment intersects the box; distance = +min box-boundary
distance inside, -(deepest corner penetration) when OOB.
"""
from __future__ import annotations

import numpy as np


def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def box_corners(pos, yaw, he):
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
    return np.asarray(pos)[None, :] + (signs * np.asarray(he)[None, :]) @ rot(yaw).T


def point_in_poly(p, poly):
    x, y = p
    xs, ys = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    cond = (ys > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xhit = xs + (y - ys) * (x2 - xs) / (y2 - ys)
    return int(np.count_nonzero(cond & (x < xhit))) % 2 == 1


def point_seg_dist(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    cp = a + t * ab
    return float(np.linalg.norm(p - cp)), cp


def point_polyline_dist(p, poly):
    best_d, best_cp = np.inf, None
    m = len(poly)
    for i in range(m):
        d, cp = point_seg_dist(p, poly[i], poly[(i + 1) % m])
        if d < best_d:
            best_d, best_cp = d, cp
    return best_d, best_cp


def point_box_dist(p, pos, yaw, he):
    q = rot(yaw).T @ (np.asarray(p) - np.asarray(pos))
    dx = max(abs(q[0]) - he[0], 0.0)
    dy = max(abs(q[1]) - he[1], 0.0)
    return float(np.hypot(dx, dy))


def seg_hits_box(a, b, pos, yaw, he):
    R = rot(yaw)
    al = R.T @ (a - np.asarray(pos))
    bl = R.T @ (b - np.asarray(pos))
    d = bl - al
    tmin, tmax = 0.0, 1.0
    for ax in range(2):
        if abs(d[ax]) < 1e-12:
            if al[ax] < -he[ax] or al[ax] > he[ax]:
                return False
        else:
            t1 = (-he[ax] - al[ax]) / d[ax]
            t2 = (he[ax] - al[ax]) / d[ax]
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
            if tmin > tmax:
                return False
    return True


def box_contact(inner, outer, pos, yaw, he):
    """Reference contact result for one box vs one env's polylines."""
    pos = np.asarray(pos, float)
    he = np.asarray(he, float)
    corners = box_corners(pos, yaw, he)
    crossed = False
    best = (np.inf, None, -1)  # (dist, boundary point, boundary id)
    for bnd, poly in ((0, inner), (1, outer)):
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            if seg_hits_box(a, b, pos, yaw, he):
                crossed = True
                cand = (0.0, point_seg_dist(pos, a, b)[1], bnd)
            else:
                cand = (point_box_dist(a, pos, yaw, he), a, bnd)
                for c in corners:
                    d, cp = point_seg_dist(c, a, b)
                    if d < cand[0]:
                        cand = (d, cp, bnd)
            if cand[0] < best[0]:
                best = cand
    inside = True
    worst_pen = 0.0
    for c in corners:
        pen = 0.0
        if point_in_poly(c, inner):        # in the hole
            inside = False
            pen = point_polyline_dist(c, inner)[0]
        if not point_in_poly(c, outer):    # outside the outer loop
            inside = False
            pen = max(pen, point_polyline_dist(c, outer)[0])
        worst_pen = max(worst_pen, pen)
    oob = (not inside) or crossed
    return {"oob": int(oob),
            "distance": -worst_pen if oob else best[0],
            "nearest": best[1],
            "boundary": best[2]}
```

- [ ] **Step 2: Write self-tests validating fixture + oracle against circle analytics**

Create `tests/test_collision_fixtures.py`:

```python
"""Sanity tests for the annulus fixture and the numpy oracle (tests of tests)."""
from __future__ import annotations

import numpy as np

from tests._collision_fixtures import annulus_polylines, make_annulus_track, make_boxes
from tests import _collision_oracle as oracle


def test_annulus_track_layout():
    E, n = 3, 128
    track = make_annulus_track(E=E, n=n, counts=[128, 100, 64])
    N_max = n + 8
    assert track.outer.shape == (E * N_max,)
    counts = track.count.numpy()
    assert list(counts) == [128, 100, 64]
    outer = track.outer.numpy().reshape(E, N_max, 2)
    # Real points on radius 1.3, NaN tail past count[e].
    r = np.linalg.norm(outer[1, :100], axis=1)
    np.testing.assert_allclose(r, 1.3, atol=1e-5)
    assert np.all(np.isnan(outer[1, 100:]))


def test_oracle_inside_box_clearance():
    track = make_annulus_track(E=1, n=512)
    inner, outer = annulus_polylines(track, 0, 512 + 8)
    # Axis-aligned box at (1.1, 0): outer gap = 1.3 - |far corner|.
    res = oracle.box_contact(inner, outer, (1.1, 0.0), 0.0, (0.05, 0.05))
    far = np.hypot(1.15, 0.05)
    assert res["oob"] == 0
    np.testing.assert_allclose(res["distance"], 1.3 - far, atol=2e-3)
    assert res["boundary"] == 1
    np.testing.assert_allclose(np.linalg.norm(res["nearest"]), 1.3, atol=2e-3)


def test_oracle_in_hole_and_crossing():
    track = make_annulus_track(E=1, n=512)
    inner, outer = annulus_polylines(track, 0, 512 + 8)
    hole = oracle.box_contact(inner, outer, (0.0, 0.0), 0.3, (0.1, 0.05))
    assert hole["oob"] == 1 and hole["distance"] < 0
    cross = oracle.box_contact(inner, outer, (1.3, 0.0), 0.0, (0.1, 0.1))
    assert cross["oob"] == 1
    # Deepest corner (1.4, +-0.1): penetration = |corner| - 1.3.
    pen = np.hypot(1.4, 0.1) - 1.3
    np.testing.assert_allclose(cross["distance"], -pen, atol=2e-3)


def test_make_boxes_nan_padding():
    pos, yaw, he = make_boxes(2, 4, {(0, 0): (1.0, 0.0, 0.1, 0.05, 0.02)})
    p = pos.numpy().reshape(-1, 2)
    assert not np.any(np.isnan(p[0]))
    assert np.all(np.isnan(p[1:]))
    assert yaw.numpy()[0] == np.float32(0.1)
```

- [ ] **Step 3: Run the self-tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_fixtures.py -v`
Expected: 4 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/_collision_fixtures.py tests/_collision_oracle.py tests/test_collision_fixtures.py
git commit -m "test: annulus Track fixture and numpy collision oracle"
```

---

### Task 3: Segments backend — `BoxContact`, `CollisionChecker`, kernel, public shim

**Files:**
- Create: `track_gen/_src/collision.py`
- Create: `track_gen/collision.py`
- Modify: `track_gen/__init__.py`
- Test: `tests/test_collision_segments.py` (started here, extended in Task 4)

**Interfaces:**
- Consumes: Task 1 helpers (`collision_geom`), `Track` from `types.py`.
- Produces (relied on by Tasks 4–9):
  - `track_gen.collision.BoxContact` — dataclass, fields `oob` (int32), `distance` (float32), `nearest` (vec2f), `normal` (vec2f), `boundary` (int32), all `[E*max_boxes]`; method `clone() -> BoxContact`.
  - `track_gen.collision.CollisionChecker(track, max_boxes, method="segments", sdf_resolution=128, sdf_padding=None)`; attributes used internally: `_E`, `_B`, `_n_max`, `_device`, `_track`, `_contact`, `_method`.
  - `CollisionChecker.query(position, yaw, half_extents) -> BoxContact` — wp.arrays `[E*max_boxes]` of vec2f/float32/vec2f.
  - Module-level `_CAPTURING` flag in `track_gen._src.collision` (Task 8 sets it during graph capture).

- [ ] **Step 1: Write the failing tests (OOB flag on the annulus)**

Create `tests/test_collision_segments.py`:

```python
"""Segments-backend collision tests against the analytic annulus."""
from __future__ import annotations

import numpy as np
import pytest

from tests._collision_fixtures import annulus_polylines, make_annulus_track, make_boxes

N = 512
N_MAX = N + 8


def _checker(track, B):
    from track_gen.collision import CollisionChecker
    return CollisionChecker(track, max_boxes=B, method="segments")


def test_oob_flags_annulus():
    track = make_annulus_track(E=1, n=N)
    B = 8
    boxes = {
        (0, 0): (1.1, 0.0, 0.0, 0.05, 0.05),    # fully inside the band
        (0, 1): (0.0, 0.0, 0.3, 0.10, 0.05),    # inside the hole
        (0, 2): (1.3, 0.0, 0.0, 0.10, 0.10),    # straddling the outer boundary
        (0, 3): (2.0, 0.1, 0.0, 0.05, 0.05),    # fully outside the outer loop
        (0, 4): (0.7, 0.0, np.pi / 4, 0.10, 0.02),  # straddling the inner boundary
        (0, 5): (1.0, 0.0, 0.7, 0.04, 0.02),    # inside, rotated
        # slots 6, 7 left inactive (NaN position)
    }
    pos, yaw, he = make_boxes(1, B, boxes)
    contact = _checker(track, B).query(pos, yaw, he)
    oob = contact.oob.numpy()
    assert list(oob[:6]) == [0, 1, 1, 1, 1, 0]
    assert list(oob[6:]) == [0, 0]              # inactive slots
    dist = contact.distance.numpy()
    assert np.all(np.isnan(dist[6:]))           # NaN outputs for inactive slots
    assert list(contact.boundary.numpy()[6:]) == [-1, -1]


def test_import_surface():
    import track_gen
    from track_gen.collision import BoxContact, CollisionChecker  # noqa: F401
    assert "collision" in track_gen.__all__
    assert track_gen.collision.CollisionChecker is CollisionChecker
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_segments.py -v`
Expected: FAIL — `ModuleNotFoundError: track_gen.collision` (no module yet)

- [ ] **Step 3: Implement `track_gen/_src/collision.py`**

```python
"""Batched box-vs-track out-of-bounds queries (segments backend + facade).

``CollisionChecker`` binds to a :class:`Track` batch and answers, per oriented
box, whether the box has left the drivable band (inside the outer loop AND
outside the inner loop), with full contact info. The default ``segments``
backend is exact: one Warp thread per box scans the env's boundary segments.
The ``sdf`` backend (``collision_sdf.py``) trades exactness near boundaries
for O(1) queries against baked per-env signed-distance grids.

Layout follows the package conventions: flat ``[E * max_boxes]`` wp.arrays,
NaN for inactive slots, in-place reuse of the output ``BoxContact`` across
``query()`` calls (use ``clone()`` for snapshots), and no host syncs while
``_CAPTURING`` is set so ``query()``/``bake()`` are CUDA-graph capturable.
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .collision_geom import (
    _box_corner,
    _closest_on_seg,
    _crossing,
    _is_nan2,
    _pick4,
    _point_to_local_box_dist,
    _rot2,
    _safe_normalize2,
    _seg_hits_aabb,
)
from .types import Track

_INITED = False
_CAPTURING = False
_BIG = 1.0e30


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
class BoxContact:
    """Batched box-vs-track contact result, flat ``[E * max_boxes]`` per field.

    .. warning::

        ``CollisionChecker.query()`` returns the SAME ``BoxContact`` instance on
        every call and overwrites its buffers in place. Call ``clone()`` for a
        fully-owned snapshot.

    Attributes
    ----------
    oob : wp.array
        ``int32`` — 1 if the box crosses a boundary or lies outside the band.
    distance : wp.array
        ``float32`` signed clearance: positive = margin from the box to the
        nearest boundary; negative = deepest corner penetration when OOB
        (0.0 when the box only edge-crosses a boundary with all corners
        inside). NaN for inactive slots.
    nearest : wp.array
        ``vec2f`` nearest point on the boundary polylines to the box.
    normal : wp.array
        ``vec2f`` boundary normal at ``nearest``, pointing INTO the band.
    boundary : wp.array
        ``int32`` — 0 = inner boundary, 1 = outer boundary, -1 = inactive slot
        or degenerate track (count < 3).
    """

    oob: wp.array
    distance: wp.array
    nearest: wp.array
    normal: wp.array
    boundary: wp.array

    def clone(self) -> "BoxContact":
        """Return a deep copy whose Warp buffers do not alias this result."""
        return BoxContact(
            oob=wp.clone(self.oob),
            distance=wp.clone(self.distance),
            nearest=wp.clone(self.nearest),
            normal=wp.clone(self.normal),
            boundary=wp.clone(self.boundary),
        )


@wp.kernel
def _box_query_segments_k(
    inner: wp.array(dtype=wp.vec2f),
    outer: wp.array(dtype=wp.vec2f),
    center: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    max_boxes: int,
    position: wp.array(dtype=wp.vec2f),
    yaw: wp.array(dtype=wp.float32),
    half_extents: wp.array(dtype=wp.vec2f),
    out_oob: wp.array(dtype=wp.int32),
    out_distance: wp.array(dtype=wp.float32),
    out_nearest: wp.array(dtype=wp.vec2f),
    out_normal: wp.array(dtype=wp.vec2f),
    out_boundary: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    e = t // max_boxes
    nan2 = wp.vec2f(wp.nan, wp.nan)

    pos = position[t]
    if _is_nan2(pos) == 1:
        out_oob[t] = 0
        out_distance[t] = wp.nan
        out_nearest[t] = nan2
        out_normal[t] = nan2
        out_boundary[t] = -1
        return

    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        # Degenerate/invalid track: conservative OOB, NaN geometry.
        out_oob[t] = 1
        out_distance[t] = wp.nan
        out_nearest[t] = nan2
        out_normal[t] = nan2
        out_boundary[t] = -1
        return

    yw = yaw[t]
    he = half_extents[t]
    ux = _rot2(yw, wp.vec2f(1.0, 0.0))
    uy = _rot2(yw, wp.vec2f(0.0, 1.0))
    c0 = _box_corner(pos, ux, uy, he, 0)
    c1 = _box_corner(pos, ux, uy, he, 1)
    c2 = _box_corner(pos, ux, uy, he, 2)
    c3 = _box_corner(pos, ux, uy, he, 3)

    base = e * n_max

    # Per-corner crossing counts and min distances to each boundary polyline.
    cn_in = wp.vec4f(0.0, 0.0, 0.0, 0.0)
    cn_out = wp.vec4f(0.0, 0.0, 0.0, 0.0)
    dc_in = wp.vec4f(_BIG, _BIG, _BIG, _BIG)
    dc_out = wp.vec4f(_BIG, _BIG, _BIG, _BIG)

    crossed = int(0)
    best_d = _BIG
    best_pt = wp.vec2f(0.0, 0.0)
    best_bnd = int(0)
    best_i = int(0)

    for j in range(2 * m):
        bnd = int(0)
        i = j
        if j >= m:
            bnd = int(1)
            i = j - m
        i2 = i + 1
        if i2 == m:
            i2 = 0
        a = wp.vec2f(0.0, 0.0)
        b = wp.vec2f(0.0, 0.0)
        if bnd == 0:
            a = inner[base + i]
            b = inner[base + i2]
        else:
            a = outer[base + i]
            b = outer[base + i2]

        # Box<->segment distance candidates: box corners vs the segment ...
        cand_d = _BIG
        cand_pt = a
        for k in range(4):
            ck = _pick4(c0, c1, c2, c3, k)
            cr = float(_crossing(ck, a, b))
            cp = _closest_on_seg(ck, a, b)
            dk = wp.length(ck - cp)
            if bnd == 0:
                cn_in[k] = cn_in[k] + cr
                dc_in[k] = wp.min(dc_in[k], dk)
            else:
                cn_out[k] = cn_out[k] + cr
                dc_out[k] = wp.min(dc_out[k], dk)
            if dk < cand_d:
                cand_d = dk
                cand_pt = cp

        # ... plus the segment start vertex vs the solid box (the end vertex is
        # the next segment's start, so the closed loop covers every vertex).
        al = _rot2(-yw, a - pos)
        bl = _rot2(-yw, b - pos)
        d_end = _point_to_local_box_dist(al, he)
        if d_end < cand_d:
            cand_d = d_end
            cand_pt = a
        if _seg_hits_aabb(al, bl, he) == 1:
            crossed = int(1)
            cand_d = 0.0
            cand_pt = _closest_on_seg(pos, a, b)

        if cand_d < best_d:
            best_d = cand_d
            best_pt = cand_pt
            best_bnd = bnd
            best_i = i

    inside = int(1)
    worst_pen = 0.0
    for k in range(4):
        pen_k = 0.0
        if int(cn_in[k]) % 2 == 1:   # corner inside the inner hole
            inside = int(0)
            pen_k = dc_in[k]
        if int(cn_out[k]) % 2 == 0:  # corner outside the outer loop
            inside = int(0)
            pen_k = wp.max(pen_k, dc_out[k])
        worst_pen = wp.max(worst_pen, pen_k)

    oob = int(0)
    if inside == 0 or crossed == 1:
        oob = int(1)

    dist = best_d
    if oob == 1:
        dist = -worst_pen

    # Normal of the argmin segment, oriented into the band via the
    # index-aligned centerline point.
    i2 = best_i + 1
    if i2 == m:
        i2 = 0
    sa = wp.vec2f(0.0, 0.0)
    sb = wp.vec2f(0.0, 0.0)
    if best_bnd == 0:
        sa = inner[base + best_i]
        sb = inner[base + i2]
    else:
        sa = outer[base + best_i]
        sb = outer[base + i2]
    seg = sb - sa
    nrm = wp.vec2f(-seg[1], seg[0])
    cpt = center[base + best_i]
    if wp.length(nrm) < 1.0e-8:
        nrm = cpt - best_pt
    nrm = _safe_normalize2(nrm)
    if wp.dot(cpt - best_pt, nrm) < 0.0:
        nrm = -nrm

    out_oob[t] = oob
    out_distance[t] = dist
    out_nearest[t] = best_pt
    out_normal[t] = nrm
    out_boundary[t] = best_bnd


class CollisionChecker:
    """Batched box-vs-track out-of-bounds checker bound to a :class:`Track`.

    Because ``TrackGenerator.generate()`` overwrites its ``Track`` buffers in
    place, the ``segments`` backend always reads the CURRENT track batch with
    no rebind step. The ``sdf`` backend requires ``bake()`` after each
    ``generate()`` call.

    ``query()`` is allocation-free and host-sync-free (CUDA-graph capturable)
    and returns the same preallocated :class:`BoxContact` on every call.
    """

    def __init__(self, track: Track, max_boxes: int, method: str = "segments",
                 sdf_resolution: int = 128, sdf_padding: "float | None" = None) -> None:
        _init()
        if int(max_boxes) < 1:
            raise ValueError(f"max_boxes must be >= 1, got {max_boxes!r}")
        if method not in ("segments", "sdf"):
            raise ValueError(
                f"method must be one of {{'segments', 'sdf'}}, got {method!r}")
        if int(sdf_resolution) < 8:
            raise ValueError(
                f"sdf_resolution must be >= 8, got {sdf_resolution!r}")
        if sdf_padding is not None and float(sdf_padding) <= 0.0:
            raise ValueError(
                f"sdf_padding must be > 0 (or None for auto), got {sdf_padding!r}")
        E = int(track.count.shape[0])
        stride = int(track.outer.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"track batch layout invalid: outer has {stride} slots for {E} envs")
        self._track = track
        self._method = method
        self._E = E
        self._n_max = stride // E
        self._B = int(max_boxes)
        self._device = str(track.outer.device)

        n = E * self._B
        dev = self._device
        self._contact = BoxContact(
            oob=wp.zeros(n, dtype=wp.int32, device=dev),
            distance=wp.zeros(n, dtype=wp.float32, device=dev),
            nearest=wp.zeros(n, dtype=wp.vec2f, device=dev),
            normal=wp.zeros(n, dtype=wp.vec2f, device=dev),
            boundary=wp.zeros(n, dtype=wp.int32, device=dev),
        )
        if method == "sdf":
            raise NotImplementedError(
                "the sdf backend is wired up in collision_sdf (next task)")

    def query(self, position: wp.array, yaw: wp.array,
              half_extents: wp.array) -> BoxContact:
        """Compute contact info for ``E * max_boxes`` oriented boxes.

        Args:
            position: ``[E*max_boxes]`` ``vec2f`` box centers. NaN marks an
                inactive slot (NaN outputs, ``oob=0``, ``boundary=-1``).
            yaw: ``[E*max_boxes]`` ``float32`` box orientations (radians).
            half_extents: ``[E*max_boxes]`` ``vec2f`` per-box half sizes.

        Returns:
            The checker's preallocated :class:`BoxContact` (same instance every
            call; buffers overwritten in place).

        Raises:
            ValueError: on shape/dtype/device mismatch.
        """
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
        t = self._track
        c = self._contact
        wp.launch(
            _box_query_segments_k, dim=n,
            inputs=[t.inner, t.outer, t.center, t.count, self._n_max, self._B,
                    position, yaw, half_extents,
                    c.oob, c.distance, c.nearest, c.normal, c.boundary],
            device=self._device,
        )
        _sync(self._device)
        return c
```

- [ ] **Step 4: Create the public shim and wire the top-level package**

Create `track_gen/collision.py`:

```python
"""Public collision-query API: box-vs-track out-of-bounds checks.

``CollisionChecker`` answers, for batches of oriented boxes against a batch of
generated tracks, whether each box has left the drivable band — with full
contact info (:class:`BoxContact`): OOB flag, signed clearance, nearest
boundary point, inward normal, and boundary id. Two Warp backends:

- ``method="segments"`` (default): exact, zero precompute; reads the bound
  ``Track`` buffers directly (fresh after every ``generate()``).
- ``method="sdf"``: bakes per-env signed-distance grids for O(1) queries;
  approximate within one grid cell near boundaries and requires ``bake()``
  after each ``generate()``. Memory ~ ``E * sdf_resolution**2 * 5`` bytes.

This module is the template for future query utilities: each gets its own
public sibling module (flat namespace, no grab-bag ``utils``).
"""
from ._src.collision import BoxContact, CollisionChecker

__all__ = ["BoxContact", "CollisionChecker"]
```

Modify `track_gen/__init__.py` — add the import after the existing `_src` imports and extend `__all__`:

```python
from ._version import __version__
from ._src.types import GateGenConfig, GateSequence, Track, TrackGenConfig
from ._src.track_generator import TrackGenerator
from ._src.gate_generator import GateGenerator
from ._src.rng_utils import PerEnvSeededRNG
from . import collision

__all__ = [
    "TrackGenerator",
    "TrackGenConfig",
    "Track",
    "GateGenerator",
    "GateGenConfig",
    "GateSequence",
    "PerEnvSeededRNG",
    "collision",
    "__version__",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_segments.py -v`
Expected: 2 PASS (first run compiles the kernel; slow once, then cached)

- [ ] **Step 6: Run the whole suite to catch import regressions**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q -x --ignore=tests/test_benchmark_smoke.py -m "not slow and not cuda"`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add track_gen/_src/collision.py track_gen/collision.py track_gen/__init__.py tests/test_collision_segments.py
git commit -m "feat: track_gen.collision — exact segments backend for box OOB queries"
```

---

### Task 4: Segments backend — contact-info correctness (distance / nearest / normal / boundary)

**Files:**
- Modify: `tests/test_collision_segments.py` (append tests)
- Modify (only if a test exposes a kernel bug): `track_gen/_src/collision.py`

**Interfaces:**
- Consumes: Task 3 `CollisionChecker.query()`; Task 2 fixtures/oracle.
- Produces: verified contact-info semantics that Tasks 6–7 (sdf agreement) rely on.

- [ ] **Step 1: Append the analytic contact-info tests**

Append to `tests/test_collision_segments.py`:

```python
def test_clearance_inside_matches_analytic():
    track = make_annulus_track(E=1, n=N)
    B = 2
    pos, yaw, he = make_boxes(1, B, {(0, 0): (1.1, 0.0, 0.0, 0.05, 0.05),
                                     (0, 1): (0.75, 0.0, 0.0, 0.02, 0.02)})
    contact = _checker(track, B).query(pos, yaw, he)
    d = contact.distance.numpy()
    bnd = contact.boundary.numpy()
    # Box 0: outer is nearest. Clearance = ro - |farthest corner|.
    np.testing.assert_allclose(d[0], 1.3 - np.hypot(1.15, 0.05), atol=2e-3)
    assert bnd[0] == 1
    # Box 1: inner is nearest. Clearance = |closest box point| - ri = 0.73 - 0.7.
    np.testing.assert_allclose(d[1], 0.03, atol=2e-3)
    assert bnd[1] == 0
    # nearest lies ON the corresponding boundary circle.
    near = contact.nearest.numpy().reshape(-1, 2)
    np.testing.assert_allclose(np.linalg.norm(near[0]), 1.3, atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(near[1]), 0.7, atol=2e-3)


def test_penetration_depth_when_oob():
    track = make_annulus_track(E=1, n=N)
    B = 2
    pos, yaw, he = make_boxes(1, B, {(0, 0): (1.3, 0.0, 0.0, 0.1, 0.1),
                                     (0, 1): (0.0, 0.0, 0.0, 0.1, 0.05)})
    contact = _checker(track, B).query(pos, yaw, he)
    d = contact.distance.numpy()
    # Box 0 straddles the outer: deepest corners (1.4, +-0.1).
    np.testing.assert_allclose(d[0], -(np.hypot(1.4, 0.1) - 1.3), atol=2e-3)
    # Box 1 fully in the hole: deepest corner is the one closest to the origin
    # (max penetration = ri - min corner radius).
    np.testing.assert_allclose(d[1], -(0.7 - np.hypot(0.1, 0.05)), atol=2e-3)


def test_normal_points_into_band():
    from tests import _collision_oracle as oracle
    track = make_annulus_track(E=1, n=N)
    inner, outer = annulus_polylines(track, 0, N_MAX)
    B = 4
    pos, yaw, he = make_boxes(1, B, {
        (0, 0): (1.1, 0.2, 0.0, 0.05, 0.05),   # near outer
        (0, 1): (0.8, -0.3, 0.4, 0.03, 0.03),  # near inner
        (0, 2): (1.35, 0.0, 0.0, 0.02, 0.02),  # outside the outer loop
        (0, 3): (0.3, 0.3, 0.0, 0.02, 0.02),   # in the hole
    })
    contact = _checker(track, B).query(pos, yaw, he)
    near = contact.nearest.numpy().reshape(-1, 2)
    nrm = contact.normal.numpy().reshape(-1, 2)
    eps = 1e-3
    for i in range(B):
        np.testing.assert_allclose(np.linalg.norm(nrm[i]), 1.0, atol=1e-5)
        probe_in = near[i] + eps * nrm[i]
        probe_out = near[i] - eps * nrm[i]
        in_band = (oracle.point_in_poly(probe_in, outer)
                   and not oracle.point_in_poly(probe_in, inner))
        out_band = (oracle.point_in_poly(probe_out, outer)
                    and not oracle.point_in_poly(probe_out, inner))
        assert in_band and not out_band, f"box {i}: normal not oriented into band"


def test_matches_oracle_on_random_boxes():
    from tests import _collision_oracle as oracle
    rng = np.random.default_rng(7)
    track = make_annulus_track(E=1, n=256, N_max=N_MAX)
    inner, outer = annulus_polylines(track, 0, N_MAX)
    B = 32
    slots = {}
    for b in range(B):
        r = rng.uniform(0.4, 1.6)
        th = rng.uniform(0.0, 2 * np.pi)
        slots[(0, b)] = (r * np.cos(th), r * np.sin(th),
                         rng.uniform(0, 2 * np.pi),
                         rng.uniform(0.01, 0.15), rng.uniform(0.01, 0.15))
    pos, yaw, he = make_boxes(1, B, slots)
    contact = _checker(track, B).query(pos, yaw, he)
    oob = contact.oob.numpy()
    dist = contact.distance.numpy()
    for b in range(B):
        px, py, yw, hx, hy = slots[(0, b)]
        ref = oracle.box_contact(inner, outer, (px, py), yw, (hx, hy))
        assert oob[b] == ref["oob"], f"box {b} oob mismatch"
        np.testing.assert_allclose(dist[b], ref["distance"], atol=1e-4,
                                   err_msg=f"box {b} distance mismatch")


def test_per_env_count_variation():
    E, B = 3, 1
    track = make_annulus_track(E=E, n=N, counts=[N, 300, 64])
    pos, yaw, he = make_boxes(E, B, {(e, 0): (1.1, 0.0, 0.0, 0.05, 0.05)
                                     for e in range(E)})
    contact = _checker(track, B).query(pos, yaw, he)
    d = contact.distance.numpy()
    expected = 1.3 - np.hypot(1.15, 0.05)
    np.testing.assert_allclose(d[0], expected, atol=2e-3)
    np.testing.assert_allclose(d[1], expected, atol=2e-3)
    np.testing.assert_allclose(d[2], expected, atol=6e-3)  # coarse 64-gon
    assert list(contact.oob.numpy()) == [0, 0, 0]
```

- [ ] **Step 2: Run the new tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_segments.py -v`
Expected: all PASS. If a contact-info test fails, debug the kernel in
`track_gen/_src/collision.py` (most likely spots: argmin bookkeeping in the
`cand_d < best_d` block, penetration accumulation, or normal orientation) —
the oracle test pinpoints which box case disagrees.

- [ ] **Step 3: Commit**

```bash
git add tests/test_collision_segments.py track_gen/_src/collision.py
git commit -m "test: analytic + oracle coverage for segments-backend contact info"
```

---

### Task 5: Contract & validation — errors, in-place reuse, clone, generated-track property test

**Files:**
- Test: `tests/test_collision_contract.py`

**Interfaces:**
- Consumes: Tasks 2–4; `TrackGenerator`/`PerEnvSeededRNG` for real generated tracks.
- Produces: locked-in error and buffer-reuse contracts (Tasks 6–8 reuse the same expectations).

- [ ] **Step 1: Write the tests**

Create `tests/test_collision_contract.py`:

```python
"""CollisionChecker construction/query contracts: validation, reuse, clone."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests._collision_fixtures import annulus_polylines, make_annulus_track, make_boxes
from track_gen.collision import BoxContact, CollisionChecker


def test_constructor_validation():
    track = make_annulus_track(E=1, n=64)
    with pytest.raises(ValueError, match="max_boxes"):
        CollisionChecker(track, max_boxes=0)
    with pytest.raises(ValueError, match="method"):
        CollisionChecker(track, max_boxes=1, method="bvh")
    with pytest.raises(ValueError, match="sdf_resolution"):
        CollisionChecker(track, max_boxes=1, sdf_resolution=4)
    with pytest.raises(ValueError, match="sdf_padding"):
        CollisionChecker(track, max_boxes=1, sdf_padding=-0.5)


def test_query_validation():
    track = make_annulus_track(E=2, n=64)
    checker = CollisionChecker(track, max_boxes=4)
    pos, yaw, he = make_boxes(2, 4, {})
    bad_pos, bad_yaw, _ = make_boxes(2, 3, {})
    with pytest.raises(ValueError, match="position"):
        checker.query(bad_pos, yaw, he)
    with pytest.raises(ValueError, match="yaw"):
        checker.query(pos, bad_yaw, he)
    with pytest.raises(ValueError, match="yaw"):
        checker.query(pos, pos, he)  # wrong dtype (vec2f where float32 expected)
    with pytest.raises(ValueError, match="position"):
        checker.query(np.zeros((8, 2), np.float32), yaw, he)  # not a wp.array


def test_query_returns_same_instance_and_clone_detaches():
    track = make_annulus_track(E=1, n=64)
    checker = CollisionChecker(track, max_boxes=1)
    pos, yaw, he = make_boxes(1, 1, {(0, 0): (1.0, 0.0, 0.0, 0.05, 0.05)})
    c1 = checker.query(pos, yaw, he)
    snap = c1.clone()
    assert isinstance(snap, BoxContact)
    d_before = float(c1.distance.numpy()[0])
    # Move the box out of bounds and re-query: c1 mutates, snap must not.
    pos2, yaw2, he2 = make_boxes(1, 1, {(0, 0): (3.0, 0.0, 0.0, 0.05, 0.05)})
    c2 = checker.query(pos2, yaw2, he2)
    assert c2 is c1
    assert float(c1.distance.numpy()[0]) < 0.0
    np.testing.assert_allclose(float(snap.distance.numpy()[0]), d_before)


def test_segments_sees_track_buffer_updates_without_rebind():
    # The checker aliases the Track buffers; writing new geometry into the SAME
    # buffers (as TrackGenerator.generate() does) must be reflected in queries.
    track = make_annulus_track(E=1, n=64)
    checker = CollisionChecker(track, max_boxes=1)
    pos, yaw, he = make_boxes(1, 1, {(0, 0): (1.0, 0.0, 0.0, 0.02, 0.02)})
    assert int(checker.query(pos, yaw, he).oob.numpy()[0]) == 0
    bigger = make_annulus_track(E=1, n=64, r_center=3.0)  # same shapes
    wp.copy(track.inner, bigger.inner)
    wp.copy(track.outer, bigger.outer)
    wp.copy(track.center, bigger.center)
    # Box at r=1 is now inside the hole of the r in [2.7, 3.3] annulus.
    assert int(checker.query(pos, yaw, he).oob.numpy()[0]) == 1


def test_generated_tracks_property_oracle():
    """Random boxes vs the numpy oracle on REAL generated tracks (cpu)."""
    from tests import _collision_oracle as oracle
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=123, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    counts = track.count.numpy()
    n_max = track.outer.shape[0] // E
    checker = CollisionChecker(track, max_boxes=4)
    rng = np.random.default_rng(0)
    center = track.center.numpy().reshape(E, n_max, 2)
    slots = {}
    for e in range(E):
        if not valid[e]:
            continue
        for b in range(4):
            i = int(rng.integers(0, counts[e]))
            jitter = rng.normal(0.0, 0.15, 2)
            px, py = center[e, i] + jitter
            slots[(e, b)] = (float(px), float(py), float(rng.uniform(0, 6.28)),
                             float(rng.uniform(0.005, 0.08)),
                             float(rng.uniform(0.005, 0.08)))
    pos, yaw, he = make_boxes(E, 4, slots)
    contact = checker.query(pos, yaw, he)
    oob = contact.oob.numpy()
    dist = contact.distance.numpy()
    inner_np = track.inner.numpy().reshape(E, n_max, 2)
    outer_np = track.outer.numpy().reshape(E, n_max, 2)
    checked = 0
    for (e, b), (px, py, yw, hx, hy) in slots.items():
        m = int(counts[e])
        ref = oracle.box_contact(inner_np[e, :m].astype(np.float64),
                                 outer_np[e, :m].astype(np.float64),
                                 (px, py), yw, (hx, hy))
        i = e * 4 + b
        assert oob[i] == ref["oob"], f"env {e} box {b}: oob mismatch"
        np.testing.assert_allclose(dist[i], ref["distance"], atol=1e-4,
                                   err_msg=f"env {e} box {b}")
        checked += 1
    assert checked > 0, "no valid envs generated — loosen the config/seed"
```

- [ ] **Step 2: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_contract.py -v`
Expected: 5 PASS (the property test generates real tracks on cpu — takes a few seconds)

- [ ] **Step 3: Commit**

```bash
git add tests/test_collision_contract.py
git commit -m "test: CollisionChecker validation, buffer-reuse and generated-track oracle contracts"
```

---

### Task 6: SDF backend — AABB + bake kernels, `bake()`

**Files:**
- Create: `track_gen/_src/collision_sdf.py`
- Modify: `track_gen/_src/collision.py` (constructor sdf branch + `bake()`)
- Test: `tests/test_collision_sdf.py` (started here, extended in Task 7)

**Interfaces:**
- Consumes: Task 1 helpers; Task 3 `CollisionChecker` internals (`_E`, `_B`, `_n_max`, `_device`, `_track`, `_contact`).
- Produces:
  - `collision_sdf._track_aabb_k` — kernel, dim `E`, inputs `[outer, count, n_max, padding, pad_frac, lo, hi]`; `padding <= 0` means auto (`pad_frac` × larger extent).
  - `collision_sdf._sdf_bake_k` — kernel, dim `E*R*R`, inputs `[inner, outer, count, n_max, res, lo, hi, phi, bid]`; `phi` float32 signed distance (+ inside band), `bid` int8 nearest-boundary id, NaN/−1 for degenerate envs.
  - `CollisionChecker.bake()` — re-bakes grids from the bound Track; `ValueError` if `method != "sdf"`.
  - Checker sdf state: `_sdf_resolution`, `_sdf_padding` (−1.0 = auto), `_sdf_lo`, `_sdf_hi` (`[E]` vec2f), `_sdf_phi` (`[E*R*R]` float32), `_sdf_bid` (`[E*R*R]` int8) — Task 7's query kernel reads these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collision_sdf.py`:

```python
"""SDF backend: bake correctness on the analytic annulus, then queries."""
from __future__ import annotations

import numpy as np
import pytest

from tests._collision_fixtures import make_annulus_track, make_boxes

N = 512
N_MAX = N + 8
R = 64


def _sdf_checker(track, B, res=R, padding=None):
    from track_gen.collision import CollisionChecker
    return CollisionChecker(track, max_boxes=B, method="sdf",
                            sdf_resolution=res, sdf_padding=padding)


def test_bake_grid_bounds_auto_padding():
    track = make_annulus_track(E=1, n=N)
    checker = _sdf_checker(track, 1)
    lo = checker._sdf_lo.numpy().reshape(-1, 2)[0]
    hi = checker._sdf_hi.numpy().reshape(-1, 2)[0]
    # AABB of the outer 1.3-circle is [-1.3, 1.3]^2; auto pad = 0.1 * 2.6 = 0.26.
    np.testing.assert_allclose(lo, [-1.56, -1.56], atol=2e-2)
    np.testing.assert_allclose(hi, [1.56, 1.56], atol=2e-2)


def test_bake_phi_matches_analytic_annulus():
    track = make_annulus_track(E=1, n=N)
    checker = _sdf_checker(track, 1)
    lo = checker._sdf_lo.numpy().reshape(-1, 2)[0]
    hi = checker._sdf_hi.numpy().reshape(-1, 2)[0]
    phi = checker._sdf_phi.numpy().reshape(R, R)
    bid = checker._sdf_bid.numpy().reshape(R, R)
    xs = lo[0] + (np.arange(R) + 0.5) / R * (hi[0] - lo[0])
    ys = lo[1] + (np.arange(R) + 0.5) / R * (hi[1] - lo[1])
    X, Y = np.meshgrid(xs, ys)          # row gy, col gx — matches bake layout
    r = np.hypot(X, Y)
    phi_true = np.minimum(r - 0.7, 1.3 - r)   # signed: + in band, - outside
    np.testing.assert_allclose(phi, phi_true, atol=5e-3)
    # Boundary id: 0 where inner circle is closer, 1 where outer is (skip ties).
    d_in, d_out = np.abs(r - 0.7), np.abs(r - 1.3)
    clear = np.abs(d_in - d_out) > 0.02
    np.testing.assert_array_equal(bid[clear] == 1, (d_out < d_in)[clear])


def test_bake_refreshes_after_track_buffer_update():
    import warp as wp
    track = make_annulus_track(E=1, n=N)
    checker = _sdf_checker(track, 1)
    phi_before = checker._sdf_phi.numpy().copy()
    bigger = make_annulus_track(E=1, n=N, r_center=2.0)
    wp.copy(track.inner, bigger.inner)
    wp.copy(track.outer, bigger.outer)
    wp.copy(track.center, bigger.center)
    checker.bake()
    assert not np.allclose(checker._sdf_phi.numpy(), phi_before)


def test_bake_rejected_for_segments_method():
    track = make_annulus_track(E=1, n=64)
    from track_gen.collision import CollisionChecker
    checker = CollisionChecker(track, max_boxes=1, method="segments")
    with pytest.raises(ValueError, match="bake"):
        checker.bake()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_sdf.py -v`
Expected: FAIL — `NotImplementedError` from the constructor sdf branch (and `AttributeError: bake`)

- [ ] **Step 3: Implement `track_gen/_src/collision_sdf.py`**

```python
"""SDF backend kernels: per-env signed-distance bake + boundary-id grids.

Bake is a brute-force O(E * R^2 * N) scan (GPU-oriented; CPU bakes are only
for small tests). phi stores signed distance to the band boundary, positive
inside the drivable band; bid stores which boundary (0 inner / 1 outer) is
nearest at each texel. Grids cover the per-env track AABB expanded by the
configured padding; queries outside the grid clamp to edge texels, which stay
negative there, so far-out boxes still read as OOB.
"""
from __future__ import annotations

import warp as wp

from .collision_geom import _closest_on_seg, _crossing

_BIG = 1.0e30


@wp.kernel
def _track_aabb_k(
    outer: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    padding: float,   # explicit per-side padding; <= 0 selects auto mode
    pad_frac: float,  # auto padding as a fraction of the larger AABB extent
    lo: wp.array(dtype=wp.vec2f),
    hi: wp.array(dtype=wp.vec2f),
):
    e = wp.tid()
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        lo[e] = wp.vec2f(-1.0, -1.0)
        hi[e] = wp.vec2f(1.0, 1.0)
        return
    base = e * n_max
    mnx = _BIG
    mny = _BIG
    mxx = -_BIG
    mxy = -_BIG
    for i in range(m):
        p = outer[base + i]
        mnx = wp.min(mnx, p[0])
        mny = wp.min(mny, p[1])
        mxx = wp.max(mxx, p[0])
        mxy = wp.max(mxy, p[1])
    pad = padding
    if pad <= 0.0:
        pad = pad_frac * wp.max(mxx - mnx, mxy - mny)
    lo[e] = wp.vec2f(mnx - pad, mny - pad)
    hi[e] = wp.vec2f(mxx + pad, mxy + pad)


@wp.kernel
def _sdf_bake_k(
    inner: wp.array(dtype=wp.vec2f),
    outer: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    res: int,
    lo: wp.array(dtype=wp.vec2f),
    hi: wp.array(dtype=wp.vec2f),
    phi: wp.array(dtype=wp.float32),
    bid: wp.array(dtype=wp.int8),
):
    t = wp.tid()
    cells = res * res
    e = t // cells
    rem = t - e * cells
    gy = rem // res
    gx = rem - gy * res

    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        phi[t] = wp.nan
        bid[t] = wp.int8(-1)
        return

    l = lo[e]
    h = hi[e]
    p = wp.vec2f(
        l[0] + (float(gx) + 0.5) / float(res) * (h[0] - l[0]),
        l[1] + (float(gy) + 0.5) / float(res) * (h[1] - l[1]),
    )

    base = e * n_max
    d_in = _BIG
    d_out = _BIG
    cn_in = int(0)
    cn_out = int(0)
    for i in range(m):
        i2 = i + 1
        if i2 == m:
            i2 = 0
        a = inner[base + i]
        b = inner[base + i2]
        cp = _closest_on_seg(p, a, b)
        d_in = wp.min(d_in, wp.length(p - cp))
        cn_in = cn_in + _crossing(p, a, b)
        a = outer[base + i]
        b = outer[base + i2]
        cp = _closest_on_seg(p, a, b)
        d_out = wp.min(d_out, wp.length(p - cp))
        cn_out = cn_out + _crossing(p, a, b)

    d = wp.min(d_in, d_out)
    inside = int(0)
    if cn_out % 2 == 1 and cn_in % 2 == 0:
        inside = int(1)
    if inside == 1:
        phi[t] = d
    else:
        phi[t] = -d
    if d_in <= d_out:
        bid[t] = wp.int8(0)
    else:
        bid[t] = wp.int8(1)
```

- [ ] **Step 4: Wire the checker — replace the constructor's `NotImplementedError` branch and add `bake()`**

In `track_gen/_src/collision.py`, replace:

```python
        if method == "sdf":
            raise NotImplementedError(
                "the sdf backend is wired up in collision_sdf (next task)")
```

with:

```python
        if method == "sdf":
            R = int(sdf_resolution)
            self._sdf_resolution = R
            self._sdf_padding = -1.0 if sdf_padding is None else float(sdf_padding)
            self._sdf_lo = wp.zeros(E, dtype=wp.vec2f, device=dev)
            self._sdf_hi = wp.zeros(E, dtype=wp.vec2f, device=dev)
            self._sdf_phi = wp.zeros(E * R * R, dtype=wp.float32, device=dev)
            self._sdf_bid = wp.zeros(E * R * R, dtype=wp.int8, device=dev)
            self.bake()
```

and add the method after `__init__`:

```python
    def bake(self) -> None:
        """(Re)bake the per-env SDF grids from the bound Track.

        Required after every ``TrackGenerator.generate()`` call when
        ``method="sdf"`` (the segments backend needs no rebake). Pure kernel
        launches — CUDA-graph capturable.

        Raises:
            ValueError: if this checker was constructed with ``method="segments"``.
        """
        if self._method != "sdf":
            raise ValueError("bake() is only valid for method='sdf' checkers")
        from . import collision_sdf
        t = self._track
        E = self._E
        R = self._sdf_resolution
        wp.launch(collision_sdf._track_aabb_k, dim=E,
                  inputs=[t.outer, t.count, self._n_max,
                          self._sdf_padding, 0.1, self._sdf_lo, self._sdf_hi],
                  device=self._device)
        wp.launch(collision_sdf._sdf_bake_k, dim=E * R * R,
                  inputs=[t.inner, t.outer, t.count, self._n_max, R,
                          self._sdf_lo, self._sdf_hi, self._sdf_phi, self._sdf_bid],
                  device=self._device)
        _sync(self._device)
```

- [ ] **Step 5: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_sdf.py -v`
Expected: 4 PASS (bake at R=64, E=1, N=512 on cpu takes a few seconds)

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/collision_sdf.py track_gen/_src/collision.py tests/test_collision_sdf.py
git commit -m "feat: sdf backend bake — per-env signed-distance + boundary-id grids"
```

---

### Task 7: SDF backend — query kernel, dispatch, backend agreement

**Files:**
- Modify: `track_gen/_src/collision_sdf.py` (append query kernel + sampling funcs)
- Modify: `track_gen/_src/collision.py` (`query()` dispatch)
- Modify: `tests/test_collision_sdf.py` (append tests)

**Interfaces:**
- Consumes: Task 6 grids (`_sdf_phi`, `_sdf_bid`, `_sdf_lo`, `_sdf_hi`); Task 3 query validation.
- Produces: `collision_sdf._box_query_sdf_k` — kernel, dim `E*B`, inputs `[lo, hi, phi, bid, res, max_boxes, position, yaw, half_extents, out_oob, out_distance, out_nearest, out_normal, out_boundary]`. `query()` now dispatches on `self._method`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_collision_sdf.py`:

```python
def test_sdf_query_annulus_cases():
    track = make_annulus_track(E=1, n=N)
    B = 4
    checker = _sdf_checker(track, B, res=128)
    cell = (2 * 1.56) / 128
    pos, yaw, he = make_boxes(1, B, {
        (0, 0): (1.1, 0.0, 0.0, 0.05, 0.05),   # inside
        (0, 1): (0.0, 0.0, 0.0, 0.10, 0.05),   # in the hole
        (0, 2): (2.0, 0.1, 0.0, 0.05, 0.05),   # far outside
        # slot 3 inactive
    })
    contact = checker.query(pos, yaw, he)
    oob = contact.oob.numpy()
    d = contact.distance.numpy()
    assert list(oob[:3]) == [0, 1, 1]
    assert oob[3] == 0 and np.isnan(d[3])
    np.testing.assert_allclose(d[0], 1.3 - np.hypot(1.15, 0.05), atol=2 * cell)
    # nearest lies near the corresponding circle; normal is unit and inward.
    near = contact.nearest.numpy().reshape(-1, 2)
    nrm = contact.normal.numpy().reshape(-1, 2)
    np.testing.assert_allclose(np.linalg.norm(near[0]), 1.3, atol=2 * cell)
    np.testing.assert_allclose(np.linalg.norm(nrm[0]), 1.0, atol=1e-4)
    assert np.dot(nrm[0], near[0]) < 0  # inward = toward the origin, near outer
    assert contact.boundary.numpy()[0] == 1
    assert contact.boundary.numpy()[1] == 0


def test_sdf_agrees_with_segments_backend():
    from track_gen.collision import CollisionChecker
    rng = np.random.default_rng(3)
    track = make_annulus_track(E=2, n=N, counts=[N, 300])
    B = 16
    res = 128
    cell = (2 * 1.56) / res
    slots = {}
    for e in range(2):
        for b in range(B):
            r = rng.uniform(0.3, 1.7)
            th = rng.uniform(0.0, 2 * np.pi)
            slots[(e, b)] = (r * np.cos(th), r * np.sin(th),
                             rng.uniform(0, 2 * np.pi),
                             rng.uniform(0.02, 0.12), rng.uniform(0.02, 0.12))
    pos, yaw, he = make_boxes(2, B, slots)
    exact = CollisionChecker(track, max_boxes=B, method="segments").query(
        pos, yaw, he).clone()
    approx = _sdf_checker(track, B, res=res).query(pos, yaw, he)
    d_ex = exact.distance.numpy()
    d_ap = approx.distance.numpy()
    oob_ex = exact.oob.numpy()
    oob_ap = approx.oob.numpy()
    np.testing.assert_allclose(d_ap, d_ex, atol=2 * cell)
    # OOB flags may only disagree in the +-2 cell band around zero clearance.
    disagree = oob_ex != oob_ap
    assert np.all(np.abs(d_ex[disagree]) < 2 * cell)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_sdf.py -v -k "query or agrees"`
Expected: FAIL — sdf checker `query()` currently runs the segments kernel path only (or errors)

- [ ] **Step 3: Append sampling funcs + query kernel to `collision_sdf.py`**

```python
from .collision_geom import _box_corner, _is_nan2, _pick4, _rot2, _safe_normalize2


@wp.func
def _grid_coord(pv: float, lov: float, hiv: float, res: int) -> float:
    f = (pv - lov) / (hiv - lov) * float(res) - 0.5
    return wp.clamp(f, 0.0, float(res) - 1.0)


@wp.func
def _sample_phi(phi: wp.array(dtype=wp.float32), base: int, res: int,
                lo: wp.vec2f, hi: wp.vec2f, p: wp.vec2f) -> float:
    fx = _grid_coord(p[0], lo[0], hi[0], res)
    fy = _grid_coord(p[1], lo[1], hi[1], res)
    x0 = int(fx)
    y0 = int(fy)
    x1 = wp.min(x0 + 1, res - 1)
    y1 = wp.min(y0 + 1, res - 1)
    tx = fx - float(x0)
    ty = fy - float(y0)
    v00 = phi[base + y0 * res + x0]
    v10 = phi[base + y0 * res + x1]
    v01 = phi[base + y1 * res + x0]
    v11 = phi[base + y1 * res + x1]
    return wp.lerp(wp.lerp(v00, v10, tx), wp.lerp(v01, v11, tx), ty)


@wp.kernel
def _box_query_sdf_k(
    lo: wp.array(dtype=wp.vec2f),
    hi: wp.array(dtype=wp.vec2f),
    phi: wp.array(dtype=wp.float32),
    bid: wp.array(dtype=wp.int8),
    res: int,
    max_boxes: int,
    position: wp.array(dtype=wp.vec2f),
    yaw: wp.array(dtype=wp.float32),
    half_extents: wp.array(dtype=wp.vec2f),
    out_oob: wp.array(dtype=wp.int32),
    out_distance: wp.array(dtype=wp.float32),
    out_nearest: wp.array(dtype=wp.vec2f),
    out_normal: wp.array(dtype=wp.vec2f),
    out_boundary: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    e = t // max_boxes
    nan2 = wp.vec2f(wp.nan, wp.nan)

    pos = position[t]
    if _is_nan2(pos) == 1:
        out_oob[t] = 0
        out_distance[t] = wp.nan
        out_nearest[t] = nan2
        out_normal[t] = nan2
        out_boundary[t] = -1
        return

    l = lo[e]
    h = hi[e]
    base = e * res * res
    yw = yaw[t]
    he = half_extents[t]
    ux = _rot2(yw, wp.vec2f(1.0, 0.0))
    uy = _rot2(yw, wp.vec2f(0.0, 1.0))
    c0 = _box_corner(pos, ux, uy, he, 0)
    c1 = _box_corner(pos, ux, uy, he, 1)
    c2 = _box_corner(pos, ux, uy, he, 2)
    c3 = _box_corner(pos, ux, uy, he, 3)

    phimin = _sample_phi(phi, base, res, l, h, pos)
    pmin = pos
    for k in range(4):
        ck = _pick4(c0, c1, c2, c3, k)
        v = _sample_phi(phi, base, res, l, h, ck)
        if v < phimin:
            phimin = v
            pmin = ck

    oob = int(0)
    if phimin < 0.0:
        oob = int(1)

    # Central-difference gradient of phi at the argmin sample; phi increases
    # into the band, so normalize(grad) already points inward.
    hx = (h[0] - l[0]) / float(res)
    hy = (h[1] - l[1]) / float(res)
    gxv = (_sample_phi(phi, base, res, l, h, pmin + wp.vec2f(hx, 0.0))
           - _sample_phi(phi, base, res, l, h, pmin - wp.vec2f(hx, 0.0))) / (2.0 * hx)
    gyv = (_sample_phi(phi, base, res, l, h, pmin + wp.vec2f(0.0, hy))
           - _sample_phi(phi, base, res, l, h, pmin - wp.vec2f(0.0, hy))) / (2.0 * hy)
    n = _safe_normalize2(wp.vec2f(gxv, gyv))
    nearest = pmin - phimin * n

    fx = _grid_coord(nearest[0], l[0], h[0], res)
    fy = _grid_coord(nearest[1], l[1], h[1], res)
    xi = wp.min(int(fx + 0.5), res - 1)
    yi = wp.min(int(fy + 0.5), res - 1)

    out_oob[t] = oob
    out_distance[t] = phimin
    out_nearest[t] = nearest
    out_normal[t] = n
    out_boundary[t] = int(bid[base + yi * res + xi])
```

- [ ] **Step 4: Dispatch in `CollisionChecker.query()`**

In `track_gen/_src/collision.py`, replace the single `wp.launch(_box_query_segments_k, ...)` call (keep the validation loop above it) with:

```python
        t = self._track
        c = self._contact
        if self._method == "segments":
            wp.launch(
                _box_query_segments_k, dim=n,
                inputs=[t.inner, t.outer, t.center, t.count, self._n_max, self._B,
                        position, yaw, half_extents,
                        c.oob, c.distance, c.nearest, c.normal, c.boundary],
                device=self._device,
            )
        else:
            from . import collision_sdf
            wp.launch(
                collision_sdf._box_query_sdf_k, dim=n,
                inputs=[self._sdf_lo, self._sdf_hi, self._sdf_phi, self._sdf_bid,
                        self._sdf_resolution, self._B,
                        position, yaw, half_extents,
                        c.oob, c.distance, c.nearest, c.normal, c.boundary],
                device=self._device,
            )
        _sync(self._device)
        return c
```

- [ ] **Step 5: Run the full sdf test file**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_sdf.py -v`
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/collision_sdf.py track_gen/_src/collision.py tests/test_collision_sdf.py
git commit -m "feat: sdf backend queries + segments/sdf agreement tests"
```

---

### Task 8: CUDA graph-capture smoke test

**Files:**
- Test: `tests/test_collision_cuda_graph.py`

**Interfaces:**
- Consumes: `track_gen._src.collision._CAPTURING`, `CollisionChecker`, fixtures.
- Produces: proof that `query()` (both backends) and `bake()` are capturable/replayable.

- [ ] **Step 1: Write the test (cuda-marked, skipped without a GPU)**

Create `tests/test_collision_cuda_graph.py`:

```python
"""CUDA-only: CollisionChecker.query()/bake() inside wp.ScopedCapture.

Follows the test_warp_graph.py pattern: whole module skipped without CUDA.
The _CAPTURING flag suppresses the checker's post-launch wp.synchronize so the
capture region stays sync-free.
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
from track_gen._src import collision as collision_mod  # noqa: E402
from track_gen.collision import CollisionChecker  # noqa: E402

DEV = "cuda:0"


@pytest.mark.parametrize("method", ["segments", "sdf"])
def test_query_graph_replay_matches_eager(method):
    track = make_annulus_track(E=4, n=256, device=DEV)
    B = 8
    checker = CollisionChecker(track, max_boxes=B, method=method,
                               sdf_resolution=64)
    slots = {(e, b): (1.1, 0.0, 0.3 * b, 0.05, 0.03)
             for e in range(4) for b in range(4)}
    pos, yaw, he = make_boxes(4, B, slots, device=DEV)

    eager = checker.query(pos, yaw, he).clone()

    prev = collision_mod._CAPTURING
    collision_mod._CAPTURING = True
    try:
        checker.query(pos, yaw, he)  # warmup: modules loaded before capture
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            checker.query(pos, yaw, he)
    finally:
        collision_mod._CAPTURING = prev

    wp.capture_launch(cap.graph)
    wp.synchronize()
    replay = checker._contact

    np.testing.assert_array_equal(replay.oob.numpy(), eager.oob.numpy())
    np.testing.assert_allclose(replay.distance.numpy(), eager.distance.numpy(),
                               rtol=1e-5, atol=1e-6, equal_nan=True)


def test_bake_graph_capturable():
    track = make_annulus_track(E=2, n=256, device=DEV)
    checker = CollisionChecker(track, max_boxes=1, method="sdf",
                               sdf_resolution=64)
    phi_eager = checker._sdf_phi.numpy().copy()
    prev = collision_mod._CAPTURING
    collision_mod._CAPTURING = True
    try:
        wp.synchronize()
        with wp.ScopedCapture(device=DEV) as cap:
            checker.bake()
    finally:
        collision_mod._CAPTURING = prev
    wp.capture_launch(cap.graph)
    wp.synchronize()
    np.testing.assert_allclose(checker._sdf_phi.numpy(), phi_eager,
                               rtol=1e-6, atol=1e-7, equal_nan=True)
```

- [ ] **Step 2: Run (skips cleanly without a GPU; passes on one)**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_collision_cuda_graph.py -v`
Expected without GPU: 3 SKIPPED. On a CUDA machine: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_collision_cuda_graph.py
git commit -m "test: CUDA graph capture smoke tests for collision query/bake"
```

---

### Task 9: Docs + final verification

**Files:**
- Modify: `docs/reference/api.rst`
- Test: full suite run

- [ ] **Step 1: Add the collision section to the API reference**

In `docs/reference/api.rst`, append after the "Result types" section:

```rst
Collision queries
-----------------

Box-vs-track out-of-bounds checks with full contact info. See
``track_gen.collision`` for backend trade-offs (exact ``segments`` scan vs
baked ``sdf`` grids).

.. automodule:: track_gen.collision
   :no-members:

.. autoclass:: track_gen.collision.CollisionChecker
   :members:

.. autoclass:: track_gen.collision.BoxContact
   :no-members:

   .. automethod:: clone
```

- [ ] **Step 2: Build the docs to verify the new section renders**

Run: `python3 -m sphinx -b html docs /tmp/claude-docs-build -q 2>&1 | tail -5`
Expected: no new warnings referencing `collision` (pre-existing unrelated warnings OK). Skip this step if sphinx is not installed in the environment (`pip show sphinx` fails) — note it in the commit message instead.

- [ ] **Step 3: Run the complete non-cuda test suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q -m "not cuda"`
Expected: all PASS (plus skips for cuda-marked modules)

- [ ] **Step 4: Commit**

```bash
git add docs/reference/api.rst
git commit -m "docs: API reference for track_gen.collision"
```

---

## Self-Review Notes (completed during planning)

- **Spec coverage:** public API + BoxContact (Task 3), semantics incl. NaN slots/degenerate tracks (Tasks 3–4), segments backend (Tasks 3–4), sdf bake/query incl. auto-padding and clamped sampling (Tasks 6–7), file layout + `__init__` wiring (Task 3), error handling (Task 5), analytic/oracle/agreement/contract/cuda tests (Tasks 2, 4, 5, 7, 8), docs (Task 9). The spec's `collision_geom.py` split is a planned refinement of the spec's two-file layout (shared funcs needed by both backends without an import cycle).
- **Type consistency check:** `BoxContact` field names (`oob`, `distance`, `nearest`, `normal`, `boundary`) and checker attrs (`_E`, `_B`, `_n_max`, `_sdf_*`) are used identically across Tasks 3–8; kernel input orders match between definitions and `wp.launch` calls.
- **Known approximations (by design, documented):** OOB `distance` is 0.0 for edge-crossing-only contact; sdf backend can miss sub-cell features; sdf `oob` may flip within ±2 cells of zero clearance (agreement test masks accordingly).
