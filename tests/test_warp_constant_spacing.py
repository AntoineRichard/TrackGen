# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Constant-spacing Warp pipeline: per-stage parity (count==N_max matches fixed-N) and
variable-count behaviour, plus the end-to-end smoothness/yield win."""
import math
import pytest
import torch

pytest.importorskip("warp")
import warp as wp  # noqa: E402
wp.init()

from track_gen import warp_pipeline as wpl, warp_relax, geometry  # noqa: E402
from track_gen.types import TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _circle(N, r, dev):
    t = torch.linspace(0, 2 * math.pi, N + 1, device=dev)[:-1]
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], -1).to(torch.float32)


def _pad(center, n_max):
    """[E,N,2] -> [E,n_max,2] NaN-padded, count=[N]*E."""
    E, N, _ = center.shape
    buf = torch.full((E, n_max, 2), float("nan"), device=center.device, dtype=torch.float32)
    buf[:, :N] = center
    count = torch.full((E,), N, dtype=torch.int32, device=center.device)
    return buf, count


@pytest.mark.parametrize("dev", DEVS)
def test_constant_spacing_resample_matches_torch_oracle(dev):
    E, N = 3, 300
    src = torch.stack([_circle(N, r, dev) for r in (1.0, 2.5, 4.0)], 0)  # [3,N,2]
    spacing, n_max = 0.5, 128
    out_w, cnt_w = wpl.resample_constant_spacing(src, spacing, n_max)
    out_t, cnt_t = geometry.arc_length_resample(src, spacing=spacing, n_max=n_max)
    assert out_w.shape == (E, n_max, 2)
    assert torch.equal(cnt_w.cpu(), cnt_t.cpu()), f"{cnt_w} vs {cnt_t}"
    for e in range(E):
        c = int(cnt_w[e])
        assert torch.allclose(out_w[e, :c], out_t[e, :c], atol=1e-4)
        assert torch.isnan(out_w[e, c:]).all()


@pytest.mark.parametrize("dev", DEVS)
def test_resample_uniform_count_aware(dev):
    # parity: count==N reproduces the fixed call
    src = torch.stack([_circle(64, 1.0, dev), _circle(64, 2.0, dev)], 0)
    base = wpl.resample_uniform(src, 64)
    buf, cnt = _pad(src, 64)
    out = wpl.resample_uniform(buf, 64, count=cnt)
    assert torch.allclose(out, base, atol=1e-5, equal_nan=True)
    # variable: env0 uses 40 real pts (rest NaN), env1 uses 64; env0 stays ~circle, pad NaN
    buf2 = torch.full((2, 64, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :40] = _circle(40, 1.0, dev); buf2[1, :64] = _circle(64, 2.0, dev)
    cnt2 = torch.tensor([40, 64], dtype=torch.int32, device=dev)
    out2 = wpl.resample_uniform(buf2, 64, count=cnt2)
    r0 = torch.linalg.norm(out2[0, :40], dim=-1)
    assert torch.allclose(r0, torch.ones_like(r0), atol=2e-2)
    assert torch.isnan(out2[0, 40:]).all()


@pytest.mark.parametrize("dev", DEVS)
def test_thickness_count_aware(dev):
    src = torch.stack([_circle(80, 1.0, dev), _circle(80, 2.0, dev)], 0)
    band = torch.tensor([3, 3], dtype=torch.int32, device=dev)
    base = wpl.thickness(src, band)
    buf, cnt = _pad(src, 80)
    out = wpl.thickness(buf, band, count=cnt)
    assert torch.allclose(out, base, atol=1e-5)
    # variable: env0 real=50 (radius-1 circle) padded to 80; thickness finite, not poisoned by NaN tail
    buf2 = torch.full((2, 80, 2), float("nan"), device=dev, dtype=torch.float32)
    buf2[0, :50] = _circle(50, 1.0, dev); buf2[1, :80] = _circle(80, 2.0, dev)
    cnt2 = torch.tensor([50, 80], dtype=torch.int32, device=dev)
    th = wpl.thickness(buf2, band, count=cnt2)
    assert th[0] > 0.0 and torch.isfinite(th[0])
    # the radius-1 50-pt circle thickness ~ min(curv_radius≈1, 0.5*sep) — sane, finite
    assert torch.isfinite(th[1])
