"""Golden regression: the generation pipelines reproduce the frozen pre-vec3f batch."""
import numpy as np
import pytest

from tests.tools.capture_goldens import GOLDEN, capture


def test_pipelines_match_pre_vec3f_goldens():
    golden = np.load(GOLDEN)
    fresh = capture()
    for key in golden.files:
        np.testing.assert_allclose(
            fresh[key], golden[key], rtol=0.0, atol=0.0, equal_nan=True,
            err_msg=key)
