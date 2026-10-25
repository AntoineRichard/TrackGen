"""The RNG distribution gate measures shape and rejects material drift."""
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from tests.tools.rng_statistics import compare, metrics


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
