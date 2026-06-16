import math

import pytest
import torch

from track_gen.generators import Centerline
from track_gen.types import Track, TrackGenConfig
from track_gen import inflation


def make_circle_centerline(radius=2.0, m=200, e=1, center=(0.0, 0.0), device="cpu"):
    """Build a Centerline whose points lie exactly on a circle (closed, no NaN padding)."""
    theta = torch.linspace(0, 2 * math.pi, m + 1, device=device)[:-1]  # drop duplicate endpoint
    x = center[0] + radius * torch.cos(theta)
    y = center[1] + radius * torch.sin(theta)
    pts = torch.stack([x, y], dim=-1)  # [m, 2]
    pts = pts.unsqueeze(0).expand(e, m, 2).contiguous()  # [e, m, 2]
    valid = torch.ones(e, dtype=torch.bool, device=device)
    return Centerline(points=pts, valid=valid)


def fixed_config(num_points=128, device="cpu", **overrides):
    """A TrackGenConfig in fixed output mode with self-distance clamp OFF by default."""
    kwargs = dict(
        device=device,
        num_envs=1,
        output_mode="fixed",
        num_points=num_points,
        clamp_self_distance=False,
    )
    kwargs.update(overrides)
    return TrackGenConfig(**kwargs)


def test_resample_stage_circle_is_arc_uniform_and_on_circle():
    radius = 2.0
    cl = make_circle_centerline(radius=radius, m=200, e=3)
    cfg = fixed_config(num_points=128, num_envs=3)

    res = inflation._resample_stage(cl, cfg)

    assert res.center.shape == (3, 128, 2)
    assert torch.equal(res.count, torch.full((3,), 128, dtype=res.count.dtype))
    r = torch.linalg.norm(res.center, dim=-1)  # [E, N]
    assert torch.allclose(r, torch.full_like(r, radius), atol=1e-3)
    seg = torch.linalg.norm(torch.diff(res.center, dim=1, append=res.center[:, :1]), dim=-1)
    assert seg.std(dim=1).max().item() < 1e-3


def test_frame_curvature_orthonormal_and_circle_kappa():
    radius = 2.0
    cl = make_circle_centerline(radius=radius, m=500, e=2)
    cfg = fixed_config(num_points=256, num_envs=2)

    res = inflation._resample_stage(cl, cfg)
    T, Nrm, kappa = inflation._frame_curvature_stage(res.center)

    t_norm = torch.linalg.norm(T, dim=-1)  # [E, N]
    assert torch.allclose(t_norm, torch.ones_like(t_norm), atol=1e-4)
    n_norm = torch.linalg.norm(Nrm, dim=-1)
    assert torch.allclose(n_norm, torch.ones_like(n_norm), atol=1e-4)
    dot = (T * Nrm).sum(dim=-1)  # [E, N]
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-4)
    assert torch.allclose(kappa, torch.full_like(kappa, 1.0 / radius), atol=1e-2)
