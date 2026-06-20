"""The validity border self_intersections check is OPTIONAL and default-OFF.

It is redundant with the thickness/separation gate (a self-crossing or fat-band overlap
drives separation_min -> 0 -> thickness < half_width -> invalid), so disabling it does not
change the valid mask. The flag exists only to re-enable the extra O(N^2) check if wanted.
"""
import dataclasses

import pytest
import torch

pytest.importorskip("warp")
import warp as wp  # noqa: E402
wp.init()

from tests._warp_compare import to_t  # noqa: E402
from track_gen._src.types import TrackGenConfig  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen._src.rng_utils import PerEnvSeededRNG  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_validity_border_check_defaults_off():
    assert TrackGenConfig().validity_border_check is False


@pytest.mark.parametrize("dev", DEVS)
def test_border_check_is_redundant_so_default_off_loses_nothing(dev):
    E = 512
    cfg = TrackGenConfig(num_envs=E, num_points=256, half_width=0.5, scale=10.0,
                         output_mode="constant_spacing", spacing=0.30, N_max=384, device=dev)
    cfg_on = dataclasses.replace(cfg, validity_border_check=True)

    rng_off = PerEnvSeededRNG(seeds=42, num_envs=E, device=dev)
    rng_on = PerEnvSeededRNG(seeds=42, num_envs=E, device=dev)

    # Track.valid is wp.array; convert to tensor for comparison.
    v_off = to_t(TrackGenerator(cfg, rng_off).generate(E).valid)
    v_on = to_t(TrackGenerator(cfg_on, rng_on).generate(E).valid)

    # identical -> the border self_intersections adds no rejections beyond thickness/separation
    assert torch.equal(v_off, v_on)
