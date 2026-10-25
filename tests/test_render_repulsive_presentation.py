"""Growth overlays describe effective forces without changing the observed geometry."""
from __future__ import annotations

import numpy as np
import pytest

from viz import render_generator_guide as guide


@pytest.mark.parametrize("deactivate", [True, False])
def test_capture_records_effective_inner_forces_without_changing_raw_states(deactivate):
    sample = guide.sample_batch("repulsive", seeds=[7],
                                overrides={"repulsive_deactivate_obstacles": deactivate})[0]
    frames, matches = guide.capture_repulsive(sample)
    assert matches
    assert any(frame["reached"] for frame in frames)
    assert any(not frame["reached"] for frame in frames)
    assert all(frame["inner_forces_active"] == (not (deactivate and frame["reached"]))
               for frame in frames)
    assert all(frame["wall_forces_active"] for frame in frames)
    np.testing.assert_array_equal(frames[-1]["xy"], sample["raw"])
    repeated, _ = guide.capture_repulsive(sample)
    assert len(frames) == len(repeated)
    for first, second in zip(frames, repeated):
        np.testing.assert_array_equal(first["xy"], second["xy"])


def test_obstacle_overlay_uses_source_ring_layout_and_nonzero_finite_samples():
    sample, gen = guide.sample_batch("repulsive", seeds=[7], return_generator=True)
    overlay = guide.repulsive_force_samples(gen, env=0)
    from track_gen._src.warp_generate_repulsive import _N_WALL, _N_DISC
    arrays = guide._generator_scratch(gen)
    xy, weights = arrays["obs_pts"].numpy(), arrays["obs_mw"].numpy()
    mask = np.isfinite(xy).all(axis=1) & (weights != 0)
    assert overlay["n_wall"] == _N_WALL
    assert overlay["n_disc"] == _N_DISC
    np.testing.assert_array_equal(overlay["wall"], xy[:_N_WALL][mask[:_N_WALL]])
    inner = np.concatenate(overlay["inner_rings"])
    np.testing.assert_array_equal(inner[np.isfinite(inner).all(axis=1)], xy[_N_WALL:][mask[_N_WALL:]])
    assert len(overlay["wall"]) == _N_WALL
    assert all(len(ring) == _N_DISC for ring in overlay["inner_rings"])


def test_active_and_inactive_overlays_preserve_the_identical_curve_and_camera():
    import matplotlib.pyplot as plt

    sample, gen = guide.sample_batch("repulsive", seeds=[7], return_generator=True)
    samples = guide.repulsive_force_samples(gen, env=0)
    frames, _ = guide.capture_repulsive(sample[0])
    final = frames[-1]
    fig, ax = plt.subplots()
    guide._limits(ax, [sample[0]["construction"]["obstacles"]] + [f["xy"] for f in frames])
    limits = ax.get_xlim(), ax.get_ylim()
    active = guide.draw_repulsive_forces(ax, samples, inner_active=True)
    inactive = guide.draw_repulsive_forces(ax, samples, inner_active=False)
    assert active["wall"].get_color() == inactive["wall"].get_color()
    assert active["inner"][0].get_color() != inactive["inner"][0].get_color()
    assert inactive["inner"][0].get_linestyle() == "--"
    for a, b in zip(active["inner"], inactive["inner"]):
        np.testing.assert_array_equal(a.get_xydata(), b.get_xydata())
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    np.testing.assert_array_equal(final["xy"], sample[0]["raw"])
    plt.close(fig)


def test_deactivation_modes_share_the_same_geometry_before_target_reaching():
    captures = {}
    for deactivate in (True, False):
        sample = guide.sample_batch("repulsive", seeds=[7],
                                    overrides={"repulsive_deactivate_obstacles": deactivate})[0]
        frames, _ = guide.capture_repulsive(sample)
        captures[deactivate] = {f["iteration"]: f for f in frames if not f.get("final")}
    common = sorted(set(captures[True]) & set(captures[False]))
    early = [i for i in common if not captures[True][i]["reached"] and not captures[False][i]["reached"]]
    later = [i for i in common if captures[True][i]["reached"] and captures[False][i]["reached"]]
    assert len(early) > 10 and later
    for iteration in early:
        np.testing.assert_array_equal(captures[True][iteration]["xy"], captures[False][iteration]["xy"])
        assert captures[True][iteration]["inner_forces_active"]
        assert captures[False][iteration]["inner_forces_active"]
    for iteration in later:
        assert not captures[True][iteration]["inner_forces_active"]
        assert captures[False][iteration]["inner_forces_active"]
