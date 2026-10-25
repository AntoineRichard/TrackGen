"""Repulsive media shows actual hard geometry and always-active shaping forces."""
from __future__ import annotations

import numpy as np
import pytest

from viz import render_generator_guide as guide


@pytest.fixture(scope="module")
def observed():
    samples, gen = guide.sample_batch("repulsive", seeds=[7], return_generator=True)
    frames, matches = guide.capture_repulsive(samples[0])
    return samples[0], gen, frames, matches


def test_capture_records_always_active_forces_and_barriers_without_changing_raw_states(observed):
    sample, _, frames, matches = observed
    assert matches
    assert any(frame["reached"] for frame in frames)
    assert any(not frame["reached"] for frame in frames)
    assert all(frame["inner_forces_active"] and frame["wall_forces_active"]
               and frame["hard_barriers_active"] for frame in frames)
    np.testing.assert_array_equal(frames[-1]["xy"], sample["raw"])
    cfg = sample["config"]
    ratchet = int(np.ceil(np.log(cfg["repulsive_grow_mult_max"]) / np.log1p(cfg["repulsive_ratchet_rate"])))
    budget = int(np.ceil(1.6*ratchet)) + cfg["repulsive_settle_iters"]
    assert frames[-1]["iteration"] <= budget
    if frames[-1]["iteration"] < budget:
        assert frames[-1]["frozen"]


def test_overlay_reads_exact_analytic_discs_and_source_weighted_ring_layout(observed):
    _, gen, _, _ = observed
    overlay = guide.repulsive_force_samples(gen, env=0)
    from track_gen._src.warp_generate_repulsive import _N_WALL, _N_DISC
    arrays = guide._generator_scratch(gen)
    xy, weights = arrays["obs_pts"].numpy(), arrays["obs_mw"].numpy()
    mask = np.isfinite(xy).all(axis=1) & (weights != 0)
    assert overlay["n_wall"] == _N_WALL and overlay["n_disc"] == _N_DISC
    np.testing.assert_array_equal(overlay["wall"], xy[:_N_WALL][mask[:_N_WALL]])
    inner = np.concatenate(overlay["inner_rings"])
    np.testing.assert_array_equal(inner, xy[_N_WALL:][mask[_N_WALL:]])
    radii = arrays["disc_radii"].numpy()
    np.testing.assert_array_equal(overlay["disc_radii"], radii[radii > 0])
    np.testing.assert_array_equal(overlay["disc_centers"], arrays["disc_centers"].numpy()[radii > 0])
    assert overlay["unscaled_margin"] == pytest.approx(.1)
    assert overlay["domain_radius"] == pytest.approx(.35*4.9029)


def test_overlay_fills_hard_discs_and_reserves_the_entire_absolute_half_width(observed):
    import matplotlib.pyplot as plt

    sample, _, frames, _ = observed
    samples = sample["construction"]["force_samples"]
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    artists = guide.draw_repulsive_forces(ax, samples)
    assert artists["wall"].get_radius() == samples["domain_radius"]
    assert all(disc.get_fill() for disc in artists["inner"])
    for disc, reservation in zip(artists["inner"], artists["reservation"]):
        assert reservation.get_radius() - disc.get_radius() == pytest.approx(samples["half_width"])
    assert artists["reservation"][-1].get_radius() == pytest.approx(samples["domain_radius"]-samples["half_width"])
    assert (ax.get_xlim(), ax.get_ylim()) == ((-2., 2.), (-2., 2.))
    np.testing.assert_array_equal(frames[-1]["xy"], sample["raw"])
    plt.close(fig)


def test_every_captured_raw_state_keeps_its_width_reservation_clear(observed):
    sample, _, frames, _ = observed
    barriers = sample["construction"]["force_samples"]
    margin = barriers["half_width"]
    for frame in frames:
        points = frame["xy"].astype(np.float64)
        assert np.linalg.norm(points, axis=1).max() <= barriers["domain_radius"]-margin+2e-6
        edge = np.roll(points, -1, axis=0)-points
        for center, radius in zip(barriers["disc_centers"], barriers["disc_radii"]):
            t = np.clip(np.sum((center-points)*edge, axis=1)/np.sum(edge*edge, axis=1), 0, 1)
            distance = np.linalg.norm(points+t[:, None]*edge-center, axis=1)
            assert distance.min() >= radius+margin-2e-6


def test_removed_controls_are_absent_and_numerical_recipes_are_supported():
    from track_gen import TrackGenConfig

    fields = guide.SETTING_COVERAGE["repulsive"]
    assert "repulsive_deactivate_obstacles" not in fields
    assert "repulsive_w_len" not in fields
    assert fields["size"] == "generator-repulsive-controls-obstacles.png"
    for field, value in (("repulsive_alpha", 3.5), ("repulsive_beta", 6.5),
                         ("repulsive_tau", .2), ("repulsive_resample_every", 10)):
        config = TrackGenConfig(generator="repulsive", **{field: value})
        assert config.repulsive_alpha+2 <= config.repulsive_beta < 2*config.repulsive_alpha+1
