"""Analytic predicates for Repulsive's finished finite-width road."""
from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest
import warp as wp

from track_gen.exec import LaunchContext
from tests.conftest import DEVS


@pytest.fixture(autouse=True)
def _isolate_plugin_registry():
    # These private geometry tests need no plugin discovery.
    yield


def _clearance():
    name = "track_gen._src.repulsive_clearance"
    assert importlib.util.find_spec(name) is not None, "Repulsive needs analytic clearance predicates"
    return importlib.import_module(name)


def _predicates_kernel():
    geom = _clearance()

    @wp.kernel
    def check(out: wp.array(dtype=wp.int32)):
        c = wp.vec2f(0.0, 0.0)
        out[0] = int(geom.segment_disc_clear(wp.vec2f(-2.0, 0.0), wp.vec2f(2.0, 0.0), c, 0.5))
        out[1] = int(geom.segment_disc_clear(wp.vec2f(-2.0, 1.0), wp.vec2f(2.0, 1.0), c, 0.5))
        out[2] = int(geom.road_cell_disc_clear(wp.vec2f(-2.0, -1.0), wp.vec2f(2.0, -1.0), wp.vec2f(2.0, 1.0), wp.vec2f(-2.0, 1.0), c, 0.25))
        out[3] = int(geom.road_cell_disc_clear(wp.vec2f(-2.0, 0.6), wp.vec2f(2.0, 0.6), wp.vec2f(2.0, 1.0), wp.vec2f(-2.0, 1.0), c, 0.5))
        out[4] = int(geom.swept_segment_disc_clear(wp.vec2f(-2.0, -1.0), wp.vec2f(2.0, -1.0), wp.vec2f(-2.0, 1.0), wp.vec2f(2.0, 1.0), c, 0.5))
        out[5] = int(geom.segment_disc_clear(wp.vec2f(-2.0, 0.5), wp.vec2f(2.0, 0.5), c, 0.5))
    return check


@pytest.mark.parametrize("dev", DEVS)
def test_exact_predicates_catch_crossing_enclosure_sweep_and_tangency(dev):
    out = wp.empty(6, dtype=wp.int32, device=dev)
    LaunchContext.current(dev).launch(_predicates_kernel(), 1, [out])
    np.testing.assert_array_equal(out.numpy(), [0, 1, 0, 1, 0, 0])


def _band_check(outer, inner, *, disc=(0.0, 0.0, 0.25), wall=5.0, size=1.0, dev="cpu"):
    geom = _clearance()
    n = len(outer)
    outer_wp = wp.array(outer, dtype=wp.vec2f, device=dev)
    inner_wp = wp.array(inner, dtype=wp.vec2f, device=dev)
    centers = wp.array([disc[:2]], dtype=wp.vec2f, device=dev)
    radii = wp.array([disc[2]], dtype=wp.float32, device=dev)
    scale = wp.array([size], dtype=wp.float32, device=dev)
    metadata = wp.empty(1, dtype=geom.RepulsiveBarriers, device=dev)
    count = wp.array([n], dtype=wp.int32, device=dev)
    valid = wp.array([1], dtype=wp.int32, device=dev)
    fail = wp.array([0], dtype=wp.int32, device=dev)
    ctx = LaunchContext.current(dev)
    ctx.launch(geom.publish_barriers_k, 1, [centers, radii, 1, wall, scale, metadata])
    geom.finished_band_validity_inplace(ctx, outer_wp, inner_wp, count, metadata, n, valid, fail)
    return int(valid.numpy()[0]), int(fail.numpy()[0])


@pytest.mark.parametrize("dev", DEVS)
def test_disc_inside_loop_hole_does_not_intersect_road(dev):
    outer = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
    inner = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    assert _band_check(outer, inner, dev=dev) == (1, 0)


@pytest.mark.parametrize("dev", DEVS)
def test_finished_band_disc_overlap_sets_actionable_fail_bit(dev):
    outer = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
    inner = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    assert _band_check(outer, inner, disc=(0, 1.5, 0.1), dev=dev) == (0, 1 << 8)


@pytest.mark.parametrize("dev", DEVS)
def test_wall_contains_entire_road_instead_of_only_centerline(dev):
    outer = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
    inner = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    assert _band_check(outer, inner, wall=2.5, dev=dev) == (0, 1 << 9)


@pytest.mark.parametrize("dev", DEVS)
def test_output_size_scales_analytic_discs_and_wall(dev):
    outer = [[-4, -4], [4, -4], [4, 4], [-4, 4]]
    inner = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
    assert _band_check(outer, inner, wall=3, size=2, dev=dev) == (1, 0)
    assert _band_check(outer, inner, disc=(0, 1.5, 0.1), wall=3, size=2, dev=dev) == (0, 1 << 8)


def _folded_cells_kernel():
    geom = _clearance()

    @wp.kernel
    def check(out: wp.array(dtype=wp.int32)):
        # Actual built-in offset cell: the generic2% thickness tolerance permits
        # its folded outer lobe. Its ruled strip contains the disc center.
        a = wp.vec2f(-6.5786151886, 7.9747247696)
        b = wp.vec2f(-6.7078304291, 8.2097263336)
        c = wp.vec2f(-4.8660326004, 1.5018973351)
        d = wp.vec2f(-0.5882952213, 4.4387283325)
        center = wp.vec2f(-6.5819940567, 8.0151453018)
        out[0] = int(geom.road_cell_disc_clear(a, b, c, d, center, 0.00412013894))
        # Conservatively cover the whole hull of a concave cell, including a
        # polygon-exterior pocket; a disc outside that hull remains allowed.
        a2 = wp.vec2f(-2.0, -1.0)
        b2 = wp.vec2f(2.0, -1.0)
        c2 = wp.vec2f(-0.5, 0.0)
        d2 = wp.vec2f(-2.0, 1.0)
        out[1] = int(geom.road_cell_disc_clear(a2, b2, c2, d2, wp.vec2f(0.0, -0.1), 0.05))
        out[2] = int(geom.road_cell_disc_clear(a2, b2, c2, d2, wp.vec2f(0.0, 0.3), 0.1))
    return check


@pytest.mark.parametrize('dev', DEVS)
def test_folded_ruled_strip_and_concave_hull_are_checked_conservatively(dev):
    a = np.array([-6.5786151886, 7.9747247696])
    b = np.array([-6.7078304291, 8.2097263336])
    c = np.array([-4.8660326004, 1.5018973351])
    d = np.array([-0.5882952213, 4.4387283325])
    center = np.array([-6.5819940567, 8.0151453018])
    t, s = .4125, .0116666666666667
    ruled_point = (1. - t) * ((1. - s) * a + s * d) + t * ((1. - s) * b + s * c)
    assert np.linalg.norm(ruled_point - center) < 1e-6
    out = wp.empty(3, dtype=wp.int32, device=dev)
    LaunchContext.current(dev).launch(_folded_cells_kernel(), 1, [out])
    np.testing.assert_array_equal(out.numpy(), [0, 0, 1])


def _precision_predicates_kernel():
    geom = _clearance()

    @wp.kernel
    def check(centers: wp.array(dtype=wp.vec2f), out: wp.array(dtype=wp.int32)):
        i = wp.tid()
        a = wp.vec2f(423.90789794921875, 856.0101928710938)
        b = wp.vec2f(435.1707763671875, 856.9830322265625)
        # A finite 0.1-wide cell lies on the side opposite the disc center.
        newa = wp.vec2f(423.91650390625, 855.9105834960938)
        newb = wp.vec2f(435.17938232421875, 856.8834228515625)
        radius = float(5.065876757726073e-5)
        out[3 * i] = int(geom.segment_disc_clear(a, b, centers[i], radius))
        out[3 * i + 1] = int(geom.swept_segment_disc_clear(a, b, newa, newb, centers[i], radius))
        out[3 * i + 2] = int(geom.road_cell_disc_clear(a, b, newb, newa, centers[i], radius))
    return check


@pytest.mark.parametrize('dev', DEVS)
def test_large_coordinate_small_disc_predicates_preserve_sub_ulp_clearance(dev):
    # Exact stored float32 coordinates: adding the projected offset to the large
    # endpoint used to round away the 2e-5 penetration and falsely certify clear.
    a = np.array([423.90789794921875, 856.0101928710938], np.float32).astype(np.float64)
    b = np.array([435.1707763671875, 856.9830322265625], np.float32).astype(np.float64)
    center = np.array([428.0730895996094, 856.3699951171875], np.float32)
    relative = a - center.astype(np.float64)
    edge = b - a
    t = np.clip(-np.dot(relative, edge) / np.dot(edge, edge), 0., 1.)
    distance = np.linalg.norm(relative + t * edge)
    radius = float(np.float32(5.065876757726073e-5))
    assert distance == pytest.approx(3.065876708829704e-5, abs=1e-16)
    assert radius - distance > 1.99e-5
    centers = wp.array([center, center + np.array([0., .02], np.float32)],
                       dtype=wp.vec2f, device=dev)
    out = wp.empty(6, dtype=wp.int32, device=dev)
    LaunchContext.current(dev).launch(_precision_predicates_kernel(), 2, [centers, out])
    np.testing.assert_array_equal(out.numpy(), [0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize('dev', DEVS)
def test_wall_certificate_keeps_its_margin_at_large_coordinates(dev):
    point = np.array([-397.0356750488281, -917.7922973632812], np.float32)
    # This point is inside the wall but just outside its conservative certificate.
    # Float32 dot products used to round the squared distance down to its limit.
    assert np.linalg.norm(point.astype(np.float64)) > 1000. - 1e-5 * 1000.
    outer = np.array([point, [-1., 0.], [0., -1.]], np.float32)
    inner = outer * np.float32(.99)
    assert _band_check(outer, inner, disc=(0., 0., 0.), wall=1000., dev=dev) == (0, 1 << 9)
    outer[0] *= np.float32(.999)
    assert _band_check(outer, inner, disc=(0., 0., 0.), wall=1000., dev=dev) == (1, 0)
