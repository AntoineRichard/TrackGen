"""Golden regression: the generation pipelines reproduce the frozen pre-vec3f batch.

The goldens were captured from the vec2f pipelines (Task 1) and are frozen.
Post-migration, the vec3f geometry fields must match the goldens EXACTLY on
xy, with z either 0 (real slots, and track padding — the track lift writes
z = 0 everywhere) or NaN (gate padding), and z NaN only where the golden xy
row is NaN. Everything else stays bit-exact.
"""
import numpy as np

from tests.tools.capture_goldens import GOLDEN, VEC3_GATES, VEC3_TRACK, capture


def test_pipelines_match_pre_vec3f_goldens():
    golden = np.load(GOLDEN)
    fresh = capture()
    for key in golden.files:
        kind, _, field = key.split("/")
        g = golden[key]
        f = fresh[key]
        is_vec3 = (kind == "track" and field in VEC3_TRACK) or \
                  (kind == "gates" and field in VEC3_GATES)
        if is_vec3:
            assert f.shape == (g.shape[0], 3), key
            np.testing.assert_allclose(
                f[:, :2], g, rtol=0.0, atol=0.0, equal_nan=True, err_msg=key)
            z = f[:, 2]
            assert ((z == 0.0) | np.isnan(z)).all(), f"{key}: z not 0-or-NaN"
            nan_rows = np.isnan(g).any(axis=1)
            assert nan_rows[np.isnan(z)].all(), \
                f"{key}: NaN z on a slot whose golden xy is finite"
        else:
            np.testing.assert_allclose(
                f, g, rtol=0.0, atol=0.0, equal_nan=True, err_msg=key)
