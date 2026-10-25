"""The generator guide uses real geometry and records what its selections mean."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from viz import render_generator_guide as guide


@pytest.mark.parametrize("name", guide.FAMILIES)
def test_samples_use_real_counts_and_record_validity(name):
    batch = guide.sample_batch(name, seeds=[7, 23])
    assert len(batch) == 2
    for sample, seed in zip(batch, [7, 23]):
        assert sample["seed"] == seed and sample["epoch"] == 0
        assert sample["config"]["generator"] == name
        assert sample["config"]["on_invalid"] == "keep"
        assert sample["raw"].shape == (256, 2)
        assert np.isfinite(sample["raw"]).all()
        assert 3 <= sample["count"] <= 384
        for field in ("center", "inner", "outer"):
            assert sample[field].shape == (sample["count"], 2)
            assert np.isfinite(sample[field]).all()
        assert sample["valid"] == (sample["fail"] == 0)


def test_same_seed_is_reproducible_and_control_changes_raw_curve():
    base = guide.sample_batch("polar", seeds=[23])[0]
    repeated = guide.sample_batch("polar", seeds=[23])[0]
    changed = guide.sample_batch("polar", seeds=[23],
                                 overrides={"polar_radial_jitter": 0.1})[0]
    np.testing.assert_array_equal(base["raw"], repeated["raw"])
    assert guide.curve_delta(base["raw"], changed["raw"]) > 0.01


def test_every_generator_setting_has_visual_or_precise_explanation():
    from track_gen._src import builtin_params

    for name in guide.FAMILIES:
        cls = getattr(builtin_params, name.title() + "Params")
        expected = {f.name for f in dataclasses.fields(cls)}
        assert set(guide.SETTING_COVERAGE[name]) == expected
        assert all(guide.SETTING_COVERAGE[name].values())


@pytest.mark.parametrize("name", ("bezier", "hull", "voronoi"))
def test_comparison_captions_expose_actual_fallback_selection(name, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    visible_captions = []

    def inspect_figure(fig, path):
        visible_captions.extend("\n".join(text.get_text() for text in ax.texts)
                                for ax in fig.axes if ax.get_visible())
        plt.close(fig)

    monkeypatch.setattr(guide, "_save", inspect_figure)
    records = guide.render_comparisons(tmp_path, name)
    samples = [sample for row in records for sample in row["samples"]]
    assert len(visible_captions) == len(samples)
    assert any(sample["polygon_fallback_used"] for sample in samples)
    for sample, caption in zip(samples, visible_captions):
        expected = "polygon fallback" if sample["polygon_fallback_used"] else "smooth curve selected"
        assert expected in caption, (name, sample["seed"], caption)
        if sample["polygon_fallback_used"]:
            assert f"{sample['smooth_raw_crossings']} crossings" in caption
        assert sample["caption"] == caption


@pytest.mark.parametrize("overrides", [None, {"repulsive_tau": 0.2}])
def test_repulsive_capture_preserves_the_real_generator_output(overrides):
    sample = guide.sample_batch("repulsive", seeds=[7], overrides=overrides)[0]
    frames, matches = guide.capture_repulsive(sample)
    assert matches
    assert len(frames) >= 10
    assert all(np.isfinite(frame["xy"]).all() for frame in frames)
    assert frames[0]["iteration"] == 0 and frames[0]["n"] == 64
    assert all(a["iteration"] < b["iteration"] for a, b in zip(frames[:-2], frames[1:-1]))
    assert {frame["n"] for frame in frames} == {64, 128, 256}
    initial_radii = np.linalg.norm(frames[0]["xy"], axis=1)
    assert np.ptp(initial_radii) < 1e-5
    assert np.ptp(np.linalg.norm(frames[-1]["xy"], axis=1)) > 0.2
    np.testing.assert_array_equal(frames[-1]["xy"], sample["raw"])


def test_committed_manifest_has_nonvacuous_comparisons_and_provenance():
    path = Path(__file__).resolve().parents[1] / "docs/assets/racing/generators-manifest.json"
    manifest = json.loads(path.read_text())
    assert set(manifest["families"]) == set(guide.FAMILIES)
    assert manifest["source"]["git_commit"]
    assert manifest["source"]["warp_version"]
    assert manifest["source"]["device"] == "cpu"
    assert set(manifest["source"]["generator_source_sha256"]) == set(guide.FAMILIES)
    assert manifest["selection"]["claims_yield"] is False
    for family, record in manifest["families"].items():
        assert len(record["examples"]) >= 3
        assert all(s["valid"] and s["count"] >= 3 for s in record["examples"])
        assert record["comparisons"]
        for comparison in record["comparisons"]:
            assert comparison["max_raw_delta_from_first"] > 1e-6, (family, comparison)
            assert len({s["seed"] for s in comparison["samples"]}) == 1
            assert all(s["epoch"] == 0 for s in comparison["samples"])
            assert all(s["raw_finite"] and s["real_geometry_finite"] for s in comparison["samples"])
            assert all(s["caption"] == guide._comparison_caption(s) for s in comparison["samples"])
            assert comparison["includes_default"]
            configs = [s["config"] for s in comparison["samples"]]
            changed = {key for cfg in configs[1:] for key in cfg if cfg[key] != configs[0][key]}
            field = comparison["field"]
            grouped = {
                "grow range": {"repulsive_grow_mult_min", "repulsive_grow_mult_max"},
                "obstacle count range": {"repulsive_obstacle_count_min", "repulsive_obstacle_count_max"},
                "obstacle radius range": {"repulsive_obstacle_radius_min_frac", "repulsive_obstacle_radius_max_frac"},
            }
            assert changed == grouped.get(field, {field})
    for file in manifest["assets"]:
        assert (path.parent / file).is_file()
    for family, fields in guide.SETTING_COVERAGE.items():
        for target in fields.values():
            if target.endswith(".png"):
                assert target in manifest["assets"], (family, target)
    for experiment in manifest["specials"]["numerical_controls"]:
        assert experiment["max_raw_delta"] > 1e-6
    growth = manifest["animations"]["repulsive"]
    assert growth["kind"] == "captured solver states"
    assert growth["iterations"] == sorted(set(growth["iterations"]))
    assert len(growth["iterations"]) >= 10
    assert growth["final_matches_generator"]
    assert growth["hard_collision_obstacles"]
    assert growth["geometry_interpolated"] is False
    assert all(state["inner_forces_active"] and state["wall_forces_active"]
               and state["hard_barriers_active"] for state in growth["force_states"])
    repulsive = manifest["families"]["repulsive"]
    # A selective movie capture must retain the older gallery's true provenance.
    assert repulsive["source"]["generator_source_sha256"] == growth["source"]["generator_source_sha256"]
    assert repulsive["source"]["repulsive_dependency_sha256"] == growth["source"]["repulsive_dependency_sha256"]
    assert growth["source"]["source_tree_dirty"] is False
    for suffix in (".png", ".mp4"):
        asset_source = manifest["asset_provenance"]["generator-repulsive-growth"+suffix]["source"]
        assert asset_source == growth["source"]
    assert repulsive["source"]["source_tree_dirty"] is False
    assert {"repulsive_clearance.py", "repulsive_barrier_energy.py", "repulsive_tangent_energy.py",
            "repulsive_parallel_energy.py", "repulsive_parallel_proposals.py"} <= set(
                repulsive["source"]["repulsive_dependency_sha256"])
    for sample in repulsive["examples"]:
        assert sample["generator_valid"] and sample["ratchet_complete"]
        assert "repulsive_deactivate_obstacles" not in sample["config"]
        assert "repulsive_w_len" not in sample["config"]
        assert sample["barriers"]["half_width"] == .1
    success, rejected = manifest["specials"]["finished_band_limit"]
    assert success["valid"]
    assert rejected["generator_valid"] and not rejected["valid"] and rejected["fail"] & 256
    assert [sample["config"]["size"] for sample in manifest["specials"]["width_reservation"]] == [.5, 1., 2.]
