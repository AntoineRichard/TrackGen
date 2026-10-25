"""Guard against vacuous or confounded solver-control figures."""
import numpy as np


def test_solver_comparisons_isolate_controls_on_real_finite_geometry():
    from viz.render_relaxation_guide import collect_comparisons

    comparisons = collect_comparisons()
    assert {row["field"] for row in comparisons} == {
        "relax_iters", "relax_margin", "relax_smooth_passes",
    }
    for row in comparisons:
        samples = row["samples"]
        assert len(samples) >= 3
        assert len({sample["seed"] for sample in samples}) == 1
        assert all(sample["config"]["on_invalid"] == "keep" for sample in samples)
        first = samples[0]
        for sample in samples:
            changed = {key for key, value in sample["config"].items()
                       if value != first["config"][key]}
            assert changed <= {row["field"]}
            np.testing.assert_array_equal(sample["raw"], first["raw"])
            assert 3 <= sample["count"] <= sample["config"]["N_max"]
            assert sample["valid"] == (sample["fail_mask"] == 0)
            for field in ("center", "inner", "outer"):
                assert sample[field].shape == (sample["count"], 2)
                assert np.isfinite(sample[field]).all()
        # Repeated/default curves or a renderer that only relabels the same data
        # must fail. All rows share a fixed pre-solve count, so indices align.
        assert max(np.linalg.norm(sample["center"] - first["center"], axis=1).max()
                   for sample in samples[1:]) > 0.01
        if row["field"] == "relax_smooth_passes":
            assert all(sample["config"]["relax_smooth_spacing_iters"] == 0
                       for sample in samples)


def test_solver_control_manifest_records_measured_comparisons(tmp_path):
    import json
    from viz.render_relaxation_guide import render_guide

    manifest = render_guide(tmp_path)
    saved = json.loads((tmp_path / "relaxation-manifest.json").read_text())
    assert saved == manifest
    assert manifest["source"]["warp_version"]
    assert manifest["source"]["solver_sha256"]
    assert manifest["selection"]["claims_yield"] is False
    for row in manifest["comparisons"]:
        assert (tmp_path / row["asset"]).is_file()
        assert row["max_center_displacement"] > 0.01
        assert all(sample["count"] > 3 for sample in row["samples"])
        assert all(sample["finite"] for sample in row["samples"])
        assert len({sample["seed"] for sample in row["samples"]}) == 1
