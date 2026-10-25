"""Check the independent full-band reference against analytic fixtures."""
import numpy as np
import pytest

from tests._repulsive_geometry_reference import minimum_band_gaps


@pytest.mark.parametrize("center,radius,expected", [
    ([0., 0.], .25, .75),       # inside the empty loop hole
    ([0., 1.5], .1, -.1),      # wholly enclosed in a road cell
    ([0., 2.5], .2, .3),       # beyond the outer boundary
    ([0., 2.5], .5, 0.),       # exact tangency
])
def test_square_band_reference(center, radius, expected):
    outer = [[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]]
    inner = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
    disc, wall = minimum_band_gaps(outer, inner, [center], [radius], 5.)
    assert disc == pytest.approx(expected)
    assert wall == pytest.approx(5. - np.sqrt(8.))


def test_folded_cell_reference_includes_its_crossed_lobe():
    outer = [[-6.5786151886, 7.9747247696], [-6.7078304291, 8.2097263336]]
    inner = [[-.5882952213, 4.4387283325], [-4.8660326004, 1.5018973351]]
    disc, _ = minimum_band_gaps(outer, inner, [[-6.5819940567, 8.0151453018]],
                               [.00412013894], 20.)
    assert disc < 0.


def test_degenerate_cell_and_wall_reference():
    disc, wall = minimum_band_gaps([[-1., 0.], [1., 0.]],
                                  [[-1., 0.], [1., 0.]], [[0., 0.]], [.1], .5)
    assert disc == pytest.approx(-.1)
    assert wall == pytest.approx(-.5)


@pytest.mark.parametrize("center,radius", [([0., 0.], np.nan),
                                           ([np.nan, 0.], .1), ([0., 0.], -.1)])
def test_invalid_obstacle_reference_fails_closed(center, radius):
    gaps = minimum_band_gaps([[-1., 0.], [1., 0.]],
                             [[-1., 0.], [1., 0.]], [center], [radius], 5.)
    assert gaps == (-np.inf, -np.inf)
