"""Construction motion must explain real geometric operations, not reveal whole panels."""
from __future__ import annotations

import numpy as np
import pytest

from viz import render_generator_guide as guide


def test_bezier_motion_uses_both_cubic_handles_from_the_generation_buffers():
    sample = guide.sample_batch("bezier", seeds=[7])[0]
    controls = guide.bezier_controls(sample)
    dense = sample["construction"]["dense"]
    nseg = sample["config"]["num_points_per_segment"]
    u = np.linspace(0, 1, nseg)[:, None]
    for i, (p0, p1, p2, p3) in enumerate(controls):
        evaluated = (1-u)**3*p0 + 3*u*(1-u)**2*p1 + 3*u*u*(1-u)*p2 + u**3*p3
        np.testing.assert_allclose(evaluated, dense[i*nseg:(i+1)*nseg], atol=3e-7)
    assert np.linalg.norm(controls[:, 1] - controls[:, 0], axis=1).min() > 0
    assert np.linalg.norm(controls[:, 2] - controls[:, 3], axis=1).min() > 0


@pytest.mark.parametrize("name", ("bezier", "hull", "polar", "voronoi", "checkpoint"))
def test_construction_drawing_exposes_recorded_prefixes_with_a_fixed_view(name):
    import matplotlib.pyplot as plt

    sample = guide.sample_batch(name, seeds=[7])[0]
    stages = guide.construction_stages(sample)
    assert len(stages) >= 4
    fig, ax = plt.subplots()
    stage = next(i for i, s in enumerate(stages) if s["progressive_curve"])
    first = guide.draw_construction_step(ax, sample, stage, .2)
    limits = ax.get_xlim(), ax.get_ylim()
    early = first["active_curve"].copy()
    ax.clear()
    last = guide.draw_construction_step(ax, sample, stage, 1)
    np.testing.assert_array_equal(early, last["active_curve"][:len(early)])
    assert 1 < len(early) < len(last["active_curve"])
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert all(np.isfinite(x).all() for x in (early, last["active_curve"]))
    if name == "checkpoint":
        chosen = int(np.argmin(sample["construction"]["candidate_crossings"]))
        np.testing.assert_array_equal(last["active_curve"], sample["construction"]["walks"][chosen])
        assert not np.array_equal(last["active_curve"][0], last["active_curve"][-1])
    plt.close(fig)


def test_voronoi_corner_cut_and_hull_displacements_show_the_actual_intermediate_geometry():
    voronoi = guide.sample_batch("voronoi", seeds=[7])[0]["construction"]
    anchors, cut = voronoi["anchors"], voronoi["augmented"]
    np.testing.assert_allclose(cut[::2], .75*anchors+.25*np.roll(anchors, -1, axis=0), atol=3e-7)
    np.testing.assert_allclose(cut[1::2], .25*anchors+.75*np.roll(anchors, -1, axis=0), atol=3e-7)
    assert all(np.any(np.all(voronoi["sites"] == p, axis=1)) for p in anchors)
    hull = guide.sample_batch("hull", seeds=[7])[0]["construction"]
    np.testing.assert_array_equal(hull["augmented"][::2], hull["anchors"])
    midpoints = (hull["anchors"] + np.roll(hull["anchors"], -1, axis=0))/2
    assert np.linalg.norm(hull["augmented"][1::2]-midpoints, axis=1).max() > .01


def test_closed_checkpoint_candidates_use_legible_shared_scale_insets():
    import matplotlib.pyplot as plt

    sample = guide.sample_batch("checkpoint", seeds=[7])[0]
    fig, ax = plt.subplots()
    guide.draw_construction_step(ax, sample, 3, 1)
    assert len(ax.child_axes) == sample["config"]["checkpoint_best_of_k"]
    limits = [(a.get_xlim(), a.get_ylim()) for a in ax.child_axes]
    assert all(x == limits[0] for x in limits)
    for inset in ax.child_axes:
        xy = inset.lines[0].get_xydata()
        assert np.ptp(xy, axis=0).max() / np.ptp(inset.get_xlim()) > .4
        assert "crossings" in inset.get_title()
    assert "selected" in ax.child_axes[0].get_title()
    plt.close(fig)


def test_bezier_resampling_step_draws_the_selected_actual_raw_samples():
    import matplotlib.pyplot as plt

    sample = guide.sample_batch("bezier", seeds=[7])[0]
    fig, ax = plt.subplots()
    frame = guide.draw_construction_step(ax, sample, 3, 1)
    np.testing.assert_array_equal(frame["active_curve"], sample["raw"])
    plt.close(fig)
