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


def test_constant_spacing_resample_matches_torch_oracle():
    dev = "cpu"
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
