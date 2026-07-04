"""Kernel-wrapper tests for the shared collision geometry @wp.funcs."""
import numpy as np
import pytest
import warp as wp

wp.init()

from track_gen._src import collision_geom as cg


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
