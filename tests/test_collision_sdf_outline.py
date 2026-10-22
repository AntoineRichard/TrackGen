"""SDF box query: exact minimum of the bilinear phi over the box outline.

Side crossings (an edge bulging into a box side between four on-road
corners) on the analytic annulus and on a real racing track, the
``oob == (distance < 0)`` contract, agreement with the exact segments
backend on the racing near-edge set (slow), and the edge cases: NaN
half-extents, boxes partly outside the grid, boxes spanning many cells.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._collision_fixtures import make_annulus_track, make_boxes
from tests._collision_oracle import box_corners, rot
from tests.conftest import DEVS

HX, HY = 0.23, 0.17  # a 0.46 x 0.34 m car box


def _sdf(track, B, res, padding=None):
    from track_gen.collision import CollisionChecker
    return CollisionChecker(track, max_boxes=B, method="sdf",
                            sdf_resolution=res, sdf_padding=padding)


def _grid(checker, e=0):
    """(phi [R, R] float64 indexed [gy, gx], lo, hi, R) of env e."""
    R = checker._sdf_resolution
    phi = checker._sdf_phi.numpy().reshape(-1, R, R)[e].astype(np.float64)
    lo = checker._sdf_lo.numpy().reshape(-1, 2)[e].astype(np.float64)
    hi = checker._sdf_hi.numpy().reshape(-1, 2)[e].astype(np.float64)
    return phi, lo, hi, R


def _bilinear(grid, pts):
    """float64 replica of collision_sdf._sample_phi (edge-clamped bilinear)."""
    phi, lo, hi, R = grid
    f = np.clip((np.atleast_2d(pts) - lo) / (hi - lo) * R - 0.5, 0.0, R - 1.0)
    i0 = np.floor(f).astype(np.int64)
    i1 = np.minimum(i0 + 1, R - 1)
    tx, ty = (f - i0).T
    a = phi[i0[:, 1], i0[:, 0]] * (1 - tx) + phi[i0[:, 1], i1[:, 0]] * tx
    b = phi[i1[:, 1], i0[:, 0]] * (1 - tx) + phi[i1[:, 1], i1[:, 0]] * tx
    return a * (1 - ty) + b * ty


def _outline(pos, yaw, he, n_side):
    """n_side points per side, corners included, kernel corner order."""
    c = box_corners(pos, yaw, he)
    u = np.linspace(0.0, 1.0, n_side, endpoint=False)[:, None]
    return np.concatenate([c[k] + u * (c[(k + 1) % 4] - c[k]) for k in range(4)])


def _argmin_point(contact):
    """The query's argmin point, recovered as nearest + distance * normal."""
    near = contact.nearest.numpy().reshape(-1, 2).astype(np.float64)
    nrm = contact.normal.numpy().reshape(-1, 2).astype(np.float64)
    d = contact.distance.numpy().astype(np.float64)
    return near + d[:, None] * nrm


def _tangent_box(d_side, hx=HX, hy=HY):
    # Box on the +x axis, long axis along the tangent (+y), inner long side
    # at distance d_side from the origin: corners at hypot(d_side, hx).
    return (d_side + hy, 0.0, np.pi / 2, hx, hy)


def _interp_tol(cell, radius):
    """Bilinear interpolation error bound of a distance field whose level
    sets bend with radius >= ``radius``: (h^2 / 8) * (|phi_xx| + |phi_yy|)
    with |phi_xx|, |phi_yy| <= 1 / radius; plus float32 slack."""
    return cell * cell / (4.0 * radius) + 1.0e-4


# ---------------------------------------------------------------------------
# Criterion 1: side crossings with all four corners on the road
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("res", [41, 256])  # 41: 117 mm cells, racing R=256 size
def test_side_crossing_annulus(res, dev):
    # 1 m road, inner radius 1.0: a tight inner bend at racing scale.
    track = make_annulus_track(E=1, n=512, r_center=1.5, half_width=0.5, device=dev)
    checker = _sdf(track, 2, res)
    pos, yaw, he = make_boxes(1, 2, {(0, 0): _tangent_box(0.99),    # 10 mm past
                                     (0, 1): _tangent_box(1.005)},  # 5 mm short
                              device=dev)
    assert np.hypot(0.99, HX) > 1.0 + 0.016   # every corner 16 mm on the road
    c = checker.query(pos, yaw, he)
    oob, d = c.oob.numpy(), c.distance.numpy()
    _, lo, hi, _ = _grid(checker)
    tol = _interp_tol(float(np.max(hi - lo)) / res, 1.0)
    cell = float(np.max(hi - lo)) / res
    assert list(oob) == [1, 0]
    np.testing.assert_allclose(d, [-0.010, 0.005], atol=tol)
    assert np.array_equal(oob == 1, d < 0.0)
    # nearest / normal come from the argmin point, near the side midpoint.
    # Along an axis-aligned side the interpolant is linear inside each cell,
    # so the argmin is on the grid line nearest the midpoint (<= 1 cell).
    np.testing.assert_allclose(_argmin_point(c)[0], [0.99, 0.0], atol=cell)
    np.testing.assert_allclose(c.nearest.numpy()[0], [1.0, 0.0], atol=cell)
    np.testing.assert_allclose(c.normal.numpy()[0], [1.0, 0.0], atol=0.02)
    assert c.boundary.numpy()[0] == 0


def test_side_crossing_spanning_many_cells():
    # Default annulus (inner r 0.7, outer 1.3), R=64: 49 mm cells, the long
    # side spans 20 cells. Its midpoint is 50 mm inside the hole; corners and
    # centre are on the road, so the 5-sample query read +29 mm.
    track = make_annulus_track(E=1, n=512)
    checker = _sdf(track, 1, 64)
    pos, yaw, he = make_boxes(1, 1, {(0, 0): _tangent_box(0.65, 0.5, 0.08)})
    c = checker.query(pos, yaw, he)
    _, lo, hi, R = _grid(checker)
    cell = float(np.max(hi - lo)) / R
    assert c.oob.numpy()[0] == 1
    np.testing.assert_allclose(c.distance.numpy()[0], -0.05,
                               atol=_interp_tol(cell, 0.65))
    np.testing.assert_allclose(_argmin_point(c)[0], [0.65, 0.0], atol=cell)


def _poly_dist_inside(poly, pts):
    """Distance from pts [n, 2] to the closed polyline, and the even-odd
    inside flag (the +x ray rule of collision_geom._crossing)."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ap = pts[:, None, :] - a[None]
    t = np.clip(np.sum(ap * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-24), 0, 1)
    d = np.linalg.norm(ap - t[..., None] * ab, axis=-1).min(axis=1)
    x, y = pts[:, :1], pts[:, 1:]
    cond = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xhit = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    inside = np.count_nonzero(cond & (x < xhit), axis=1) % 2 == 1
    return d, inside


def _exact_phi(inner, outer, pts):
    d_in, in_hole = _poly_dist_inside(inner, pts)
    d_out, in_outer = _poly_dist_inside(outer, pts)
    d = np.minimum(d_in, d_out)
    return np.where(in_outer & ~in_hole, d, -d)


def _racing_tracks(E):
    from benchmarks.benchmark_collision import racing_tracks
    return racing_tracks(E, "cpu")


def _xy(arr, E):
    n_max = arr.shape[0] // E
    return arr.numpy().reshape(E, n_max, 3)[..., :2].astype(np.float64)


def test_side_crossing_real_track_inner_bend():
    from track_gen.collision import CollisionChecker
    track = _racing_tracks(1)
    m = int(track.count.numpy()[0])
    center = _xy(track.center, 1)[0, :m]
    inner = _xy(track.inner, 1)[0, :m]
    outer = _xy(track.outer, 1)[0, :m]
    # Tightest bend: signed turning per arc length over a +-3 vertex window.
    seg = np.roll(center, -1, axis=0) - center
    ln = np.linalg.norm(seg, axis=1)
    hd = np.arctan2(seg[:, 1], seg[:, 0])
    w = 3
    turn = np.angle(np.exp(1j * (np.roll(hd, -w) - np.roll(hd, w))))
    kappa = turn / (2 * w * ln.mean())
    i = int(np.argmax(np.abs(kappa)))
    tang = seg[i] / ln[i] + seg[i - 1] / ln[i - 1]
    heading = float(np.arctan2(tang[1], tang[0]))
    side = np.sign(kappa[i])                         # +1: left turn, inner = left
    left = np.array([-np.sin(heading), np.cos(heading)])
    # Push the long side on the inside of the bend across the edge until it
    # is >= 8 mm deep while every corner stays on the road. (The inside of a
    # bend is the inner polyline only when the bend turns with the track's
    # winding; the expected boundary id is read from the geometry.)
    for delta in np.arange(0.0, 0.04, 0.001):
        xy = center[i] + side * (0.5 - HY + delta) * left
        corners = box_corners(xy, heading, (HX, HY))
        pts = _outline(xy, heading, (HX, HY), 800)
        phi = _exact_phi(inner, outer, pts)
        ref = phi.min()
        if np.all(_exact_phi(inner, outer, corners) > 0) and ref < -0.008:
            break
    else:
        pytest.fail("no side-crossing placement found at the tightest bend")
    pos, yaw, he = make_boxes(1, 1, {(0, 0): (xy[0], xy[1], heading, HX, HY)})
    seg_c = CollisionChecker(track, max_boxes=1, method="segments").query(pos, yaw, he)
    assert seg_c.oob.numpy()[0] == 1                 # exact backend: OOB
    checker = _sdf(track, 1, 256)
    c = checker.query(pos, yaw, he)
    _, lo, hi, R = _grid(checker)
    cell = float(np.max(hi - lo)) / R
    r_edge = 1.0 / abs(kappa[i]) - 0.5               # inner-edge radius at the bend
    assert c.oob.numpy()[0] == 1
    assert c.distance.numpy()[0] < 0.0
    np.testing.assert_allclose(c.distance.numpy()[0], ref, atol=_interp_tol(cell, r_edge))
    p_ref = pts[np.argmin(phi)][None]
    expected_bnd = int(_poly_dist_inside(outer, p_ref)[0][0]
                       < _poly_dist_inside(inner, p_ref)[0][0])
    assert c.boundary.numpy()[0] == expected_bnd
    assert np.dot(c.normal.numpy()[0], -side * left) > 0.9   # toward the road


# ---------------------------------------------------------------------------
# The query is the exact minimum of the interpolant (edge cases included)
# ---------------------------------------------------------------------------

def _random_boxes(rng, n, span, he_max):
    return {(0, b): (rng.uniform(-span, span), rng.uniform(-span, span),
                     rng.uniform(0.0, 2.0 * np.pi),
                     rng.uniform(0.02, he_max), rng.uniform(0.02, he_max))
            for b in range(n)}


@pytest.mark.parametrize("dev", DEVS)
def test_outline_min_matches_dense_bilinear_reference(dev):
    # Coarse grid (R=24, 130 mm cells) and boxes up to 1.2 m wide: sides span
    # up to ~9 cells, and positions up to 1.9 m reach past the grid edge
    # (1.56 m), so pieces in the edge-clamped region are exercised too.
    track = make_annulus_track(E=1, n=256, device=dev)
    B = 96
    slots = _random_boxes(np.random.default_rng(7), B, 1.9, 0.6)
    pos, yaw, he = make_boxes(1, B, slots, device=dev)
    checker = _sdf(track, B, 24)
    c = checker.query(pos, yaw, he)
    grid = _grid(checker)
    phi, lo, hi, R = grid
    cell = (hi - lo) / R
    lip = np.sqrt(2.0) * max(np.abs(np.diff(phi, axis=1)).max() / cell[0],
                             np.abs(np.diff(phi, axis=0)).max() / cell[1])
    d = c.distance.numpy().astype(np.float64)
    oob = c.oob.numpy()
    pmin = _argmin_point(c)
    nrm = c.normal.numpy().reshape(-1, 2)
    n_side = 4000
    partly_out = 0
    for (_, b), (px, py, yw, hx, hy) in slots.items():
        p = np.array([px, py])
        pts = np.vstack([p, _outline(p, yw, (hx, hy), n_side)])
        dense = _bilinear(grid, pts).min()
        five = _bilinear(grid, np.vstack([p, box_corners(p, yw, (hx, hy))])).min()
        step = 2.0 * max(hx, hy) / n_side
        # exact minimum: at most the dense one, at least it minus the
        # interpolant's change over half a sampling step.
        assert d[b] <= dense + 1e-5, b
        assert d[b] >= dense - lip * step / 2 - 1e-5, b
        assert d[b] <= five + 1e-5, b                # never above the 5 samples
        assert oob[b] == int(d[b] < 0.0)
        # argmin point: the centre or on the outline, and phi there == distance
        q = rot(yw).T @ (pmin[b] - p)
        on_outline = abs(max(abs(q[0]) - hx, abs(q[1]) - hy)) < 1e-4
        assert on_outline or np.allclose(pmin[b], p, atol=1e-4), b
        np.testing.assert_allclose(_bilinear(grid, pmin[b])[0], d[b], atol=1e-5)
        # normal: normalized central-difference gradient at the argmin point
        g = np.array([
            _bilinear(grid, pmin[b] + [cell[0], 0])[0] - _bilinear(grid, pmin[b] - [cell[0], 0])[0],
            _bilinear(grid, pmin[b] + [0, cell[1]])[0] - _bilinear(grid, pmin[b] - [0, cell[1]])[0],
        ]) / (2 * cell)
        if np.linalg.norm(g) > 0.1:
            np.testing.assert_allclose(nrm[b], g / np.linalg.norm(g), atol=1e-3)
        corners = box_corners(p, yw, (hx, hy))
        partly_out += bool(np.any((corners < lo) | (corners > hi)))
    assert partly_out >= 10


@pytest.mark.parametrize("dev", DEVS)
def test_box_partly_or_fully_outside_grid(dev):
    # Small explicit padding: the grid ends 50 mm outside the outer edge.
    track = make_annulus_track(E=1, n=256, device=dev)
    checker = _sdf(track, 5, 64, padding=0.05)
    _, lo, hi, _ = _grid(checker)
    assert hi[0] < 1.36
    pos, yaw, he = make_boxes(1, 5, {
        (0, 0): (1.33, 0.0, 0.0, 0.10, 0.05),     # straddles the grid edge
        (0, 1): (1.0e30, 0.0, 0.3, 0.10, 0.05),   # far away (no int overflow)
        (0, 2): (1.0, 0.0, 0.0, 1.0e3, 1.0e3),    # swallows the whole track
        (0, 3): (1.0, 0.0, 0.0, 0.05, 0.05),      # on the road
        (0, 4): (0.0, 0.0, 0.0, 1.0e37, 1.0e37),  # side span overflows float
    }, device=dev)
    c = checker.query(pos, yaw, he)
    oob, d = c.oob.numpy(), c.distance.numpy()
    assert list(oob) == [1, 1, 1, 0, 1]
    assert np.all(np.isfinite(d)) and np.array_equal(oob == 1, d < 0.0)
    assert np.all(np.isfinite(c.normal.numpy()))
    # Past the grid edge phi clamps to the edge texel: the flag is exact, the
    # distance is the edge texel's (-29 mm here, the corner is 131 mm out).
    cell = float(hi[0] - lo[0]) / 64
    np.testing.assert_allclose(d[0], 1.3 - (hi[0] - 0.5 * cell), atol=2e-3)


@pytest.mark.parametrize("dev", DEVS)
def test_nan_half_extents_and_yaw_are_conservative_oob(dev):
    # Non-finite half-extents or yaw: oob=1 with NaN geometry (like a
    # degenerate env), the grid is not sampled; neighbours are unaffected.
    track = make_annulus_track(E=1, n=256, device=dev)
    checker = _sdf(track, 5, 64)
    pos, yaw, he = make_boxes(1, 5, {
        (0, 0): (1.0, 0.0, 0.0, 0.05, 0.05),
        (0, 1): (1.0, 0.0, 0.0, np.nan, 0.05),
        (0, 2): (1.0, 0.0, 0.0, np.inf, 0.05),
        (0, 3): (1.0, 0.0, np.nan, 0.05, 0.05),
        # slot 4 inactive (NaN position)
    }, device=dev)
    c = checker.query(pos, yaw, he)
    oob, d = c.oob.numpy(), c.distance.numpy()
    near, nrm = c.nearest.numpy(), c.normal.numpy()
    assert list(oob) == [0, 1, 1, 1, 0]
    assert list(c.boundary.numpy()) == [1, -1, -1, -1, -1]
    assert np.isfinite(d[0]) and np.all(np.isnan(d[1:]))
    assert np.all(np.isnan(near[1:])) and np.all(np.isnan(nrm[1:]))


def test_degenerate_env_still_conservative():
    track = make_annulus_track(E=2, n=256, counts=[256, 2])
    checker = _sdf(track, 1, 32)
    pos, yaw, he = make_boxes(2, 1, {(0, 0): (1.0, 0.0, 0.0, 0.05, 0.05),
                                     (1, 0): (1.0, 0.0, 0.0, 0.05, 0.05)})
    c = checker.query(pos, yaw, he)
    assert list(c.oob.numpy()) == [0, 1]
    assert np.isnan(c.distance.numpy()[1]) and c.boundary.numpy()[1] == -1


def test_query_deterministic():
    track = make_annulus_track(E=1, n=256)
    B = 64
    pos, yaw, he = make_boxes(1, B, _random_boxes(np.random.default_rng(3), B, 1.6, 0.3))
    checker = _sdf(track, B, 48)
    a = checker.query(pos, yaw, he).clone()
    b = checker.query(pos, yaw, he)
    for name in ("oob", "distance", "nearest", "normal", "boundary"):
        assert getattr(a, name).numpy().tobytes() == getattr(b, name).numpy().tobytes()


# ---------------------------------------------------------------------------
# Criteria 4 and 2: contract and agreement on the racing near-edge set
# ---------------------------------------------------------------------------

def test_oob_iff_negative_distance_racing_near_edge():
    from benchmarks.benchmark_collision import near_edge_boxes
    E, B = 8, 64
    track = _racing_tracks(E)
    pos, yaw, he = near_edge_boxes(track, E, B, np.random.default_rng(0))
    c = _sdf(track, B, 256).query(pos, yaw, he)
    oob, d = c.oob.numpy(), c.distance.numpy()
    assert np.array_equal(oob == 1, d < 0.0)


@pytest.mark.slow
def test_sdf_near_edge_agreement_racing_config():
    # 100 held-out racing worlds (seeds 1000000..1000099) x 16 near-edge car
    # boxes, R=256. The GPU-scale run (x 1024 boxes) is the benchmark's
    # --racing mode.
    from benchmarks.benchmark_collision import run_racing_benchmark
    m = run_racing_benchmark(E=100, B=16, device="cpu", sdf_resolution=256,
                             timing=False)
    print(f"agreement {m['agreement'] * 100:.3f}% FN {m['fn']} FP {m['fp']} "
          f"(beyond 2 mm {m['fp_beyond']})")
    assert m["agreement"] >= 0.9995
    assert m["fp_beyond"] == 0          # no false positive beyond interpolation error
    assert m["contract_violations"] == 0


@pytest.mark.slow
def test_sdf_near_edge_full_scale_disagreements_are_field_limited():
    # 100 worlds x 1024 boxes. Agreement sits at the 99.95% line here
    # (99.949% on CPU with this box seed), so rather than a threshold this
    # checks why boxes still disagree: each one's distance is the exact
    # outline minimum of the baked field (the miss is the grid's, e.g. an
    # off-road strip narrower than a cell), no false positive is new against
    # the previous 5-sample query, and most of its misses are gone.
    from benchmarks.benchmark_collision import near_edge_boxes
    from track_gen.collision import CollisionChecker
    E, B = 100, 1024
    track = _racing_tracks(E)
    pos, quat, he = near_edge_boxes(track, E, B, np.random.default_rng(0))
    exact = CollisionChecker(track, max_boxes=B, method="segments").query(
        pos, quat, he).clone()
    checker = _sdf(track, B, 256)
    c = checker.query(pos, quat, he)
    valid = np.repeat(track.valid.numpy().astype(bool), B)
    o_ex = exact.oob.numpy().astype(bool) & valid
    o_ap = c.oob.numpy().astype(bool) & valid
    d = c.distance.numpy()
    P = pos.numpy()[:, :2].astype(np.float64)
    q = quat.numpy().astype(np.float64)
    yaw = 2.0 * np.arctan2(q[:, 2], q[:, 3])
    H = he.numpy().astype(np.float64)
    # The previous query's flags: min of phi at the centre and 4 corners.
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], np.float64)
    loc = signs[None] * H[:, None]                                  # [n, 4, 2]
    cs, sn = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    corners = P[:, None] + np.stack([cs * loc[..., 0] - sn * loc[..., 1],
                                     sn * loc[..., 0] + cs * loc[..., 1]], -1)
    five = np.empty(E * B)
    for e in range(E):
        s = slice(e * B, (e + 1) * B)
        pts = np.concatenate([P[s, None], corners[s]], axis=1).reshape(-1, 2)
        five[s] = _bilinear(_grid(checker, e), pts).reshape(B, 5).min(axis=1)
    o_five = (five < 0.0) & valid
    agree = float(np.mean(o_ex[valid] == o_ap[valid]))
    fn, fp, fn5 = o_ex & ~o_ap, ~o_ex & o_ap, o_ex & ~o_five
    print(f"agreement {agree * 100:.3f}% FN {fn.sum()} FP {fp.sum()} | "
          f"5-sample {np.mean(o_ex[valid] == o_five[valid]) * 100:.3f}% FN {fn5.sum()}")
    assert np.array_equal(c.oob.numpy() == 1, d < 0.0)
    assert not np.any(fp & ~o_five)            # no new false positive
    assert not np.any(fn & ~fn5) and fn.sum() * 4 < fn5.sum()
    assert agree >= 0.999
    for t in np.nonzero(fn | fp)[0]:
        dense = _bilinear(_grid(checker, t // B),
                          np.vstack([P[t], _outline(P[t], yaw[t], H[t], 4000)])).min()
        assert abs(d[t] - dense) < 1e-5, t
