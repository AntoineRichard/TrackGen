import math

import torch

from tests._oracle.geometry import arc_length_resample


def _circle(n: int, radius: float) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)
    ang = 2.0 * math.pi * k / n
    pts = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    return pts.unsqueeze(0)


def test_circle_resample_is_arc_uniform_and_on_the_circle():
    r = 2.0
    pts = _circle(37, r)  # uneven count to force genuine interpolation
    out, count = arc_length_resample(pts, num=120)
    assert out.shape == (1, 120, 2)
    assert count.shape == (1,)
    assert int(count[0]) == 120
    radii = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(radii, torch.full_like(radii, r), atol=1e-2)
    step = torch.linalg.norm(out[:, 1:] - out[:, :-1], dim=-1)
    assert (step.std() / step.mean()) < 1e-2


def test_nan_padded_input_is_handled():
    real = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    nan_pad = torch.full((3, 2), float("nan"), dtype=torch.float64)
    pts = torch.cat([real, nan_pad], dim=0).unsqueeze(0)  # [1, 7, 2]
    out, count = arc_length_resample(pts, num=40)
    assert out.shape == (1, 40, 2)
    assert torch.isfinite(out).all()
    assert int(count[0]) == 40
    assert (out >= -1e-6).all() and (out <= 1.0 + 1e-6).all()


def test_constant_spacing_pads_to_n_max_with_nan():
    # Two circles of different perimeter -> different real counts -> padding to n_max.
    small = _circle(60, 1.0)
    big = _circle(60, 3.0)
    pts = torch.cat([small, big], dim=0)  # [2, 60, 2]
    out, count = arc_length_resample(pts, spacing=0.25, n_max=128)
    assert out.shape == (2, 128, 2)  # padded to n_max, not batch-max
    assert int(count[0]) < int(count[1])
    assert int(count[0]) <= 128 and int(count[1]) <= 128
    c0 = int(count[0])
    if c0 < 128:
        assert torch.isnan(out[0, c0:]).all()
    real0 = out[0, :c0]
    radii0 = torch.linalg.norm(real0, dim=-1)
    assert torch.allclose(radii0, torch.ones_like(radii0), atol=2e-2)


def test_fewer_than_two_real_points_yields_nan_row_count_zero():
    # One valid env, one all-NaN env: the all-NaN env must not crash, returns NaN row + count 0.
    good = _circle(40, 1.0)[0]  # [40, 2]
    bad = torch.full((40, 2), float("nan"), dtype=torch.float64)
    pts = torch.stack([good, bad], dim=0)  # [2, 40, 2]
    out, count = arc_length_resample(pts, num=32)
    assert out.shape == (2, 32, 2)
    assert int(count[0]) == 32
    assert int(count[1]) == 0  # all-NaN env: zero real points
    assert torch.isfinite(out[0]).all()
    assert torch.isnan(out[1]).all()
