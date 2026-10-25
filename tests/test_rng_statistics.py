"""The RNG distribution gate measures shape and rejects material drift."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from tests.tools.rng_statistics import (compare, compare_summaries, metrics,
                                        sample_cases, summarize)


def test_shape_metrics_use_only_valid_geometry():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    points = np.concatenate([points, np.full_like(points, np.nan)])
    result = SimpleNamespace(center=wp.array(points, dtype=wp.vec3f, device="cpu"),
                             count=wp.array([4, 0], dtype=wp.int32, device="cpu"),
                             valid=wp.array([1, 0], dtype=wp.int32, device="cpu"))
    got = metrics(result, gates=False)
    assert got["valid"].tolist() == [1, 0]
    assert got["count"].tolist() == [4]
    assert got["length"].tolist() == [4]
    np.testing.assert_allclose(got["compactness"], [np.pi / 4])
    np.testing.assert_allclose(got["curvature"], [np.pi / 2])


def test_distribution_gate_accepts_identical_samples(tmp_path, capsys):
    path = tmp_path / "sample.npz"
    np.savez(path, length=np.linspace(0, 1, 4096), valid=np.ones(4096))
    assert all(row["passed"] for row in compare(str(path), str(path)).values())


@pytest.mark.parametrize("metric", ["length", "valid"])
def test_distribution_gate_rejects_material_drift(tmp_path, capsys, metric):
    old, new = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez(old, **{metric: np.zeros(4096)})
    np.savez(new, **{metric: np.ones(4096)})
    with pytest.raises(AssertionError, match="distribution drift"):
        compare(str(old), str(new))


def test_summary_records_yield_and_shape_quantiles():
    sample = {
        "track/example/valid": np.array([1, 1, 1, 1, 1, 0]),
        "track/example/count": np.array([10, 20, 30, 40, 50]),
        "track/example/length": np.array([1, 2, 3, 4, 5]),
        "track/example/compactness": np.array([.1, .2, .3, .4, .5]),
        "track/example/curvature": np.array([2, 4, 6, 8, 10]),
        "track/example/z_mean": np.zeros(5),
        "track/example/z_std": np.zeros(5),
    }
    got = summarize(sample)["track/example"]
    assert got["samples"] == 6
    assert got["valid"] == 5
    assert got["yield"] == pytest.approx(5 / 6)
    np.testing.assert_allclose(got["quantiles"]["count"], [14, 30, 46])
    np.testing.assert_allclose(got["quantiles"]["length"], [1.4, 3, 4.6])
    np.testing.assert_allclose(got["quantiles"]["compactness"], [.14, .3, .46])
    np.testing.assert_allclose(got["quantiles"]["curvature"], [2.8, 6, 9.2])


def _summary_fixture():
    return {"track/example": {
        "samples": 1024, "valid": 768, "yield": .75,
        "quantiles": {"length": [1., 2., 3.], "compactness": [.2, .4, .6],
                      "curvature": [2., 4., 6.], "count": [64., 128., 192.],
                      "z_mean": [0., 0., 0.], "z_std": [.1, .2, .3]},
    }}


def test_summary_gate_allows_sampling_noise_and_last_ulp_differences():
    expected, observed = _summary_fixture(), _summary_fixture()
    observed["track/example"]["yield"] += .01
    for values in observed["track/example"]["quantiles"].values():
        values[:] = [value * 1.04 + 1e-7 for value in values]
    compare_summaries(expected, observed)


@pytest.mark.parametrize("metric", ["yield", "length", "compactness", "curvature", "count", "z_mean", "z_std"])
def test_summary_gate_rejects_skewed_statistics(metric):
    expected, observed = _summary_fixture(), _summary_fixture()
    if metric == "yield":
        observed["track/example"][metric] /= 2
    elif metric == "z_mean":
        observed["track/example"]["quantiles"][metric] = [.01, .01, .01]
    else:
        observed["track/example"]["quantiles"][metric] = [
            value * 1.06 for value in observed["track/example"]["quantiles"][metric]]
    with pytest.raises(AssertionError, match=metric):
        compare_summaries(expected, observed)


def test_real_generator_distributions_match_rc1_reference():
    reference = json.loads((Path(__file__).parent / "goldens/rng_statistics.json").read_text())
    observed = summarize(sample_cases(reference["cases"]))
    compare_summaries(reference["summary"], observed, **reference["tolerances"])
