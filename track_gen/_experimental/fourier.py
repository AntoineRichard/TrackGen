"""Experimental truncated-Fourier centerline generator.

NOT part of the supported Warp pipeline: it was never ported to Warp, and the
TrackGenerator facade rejects ``generator != "bezier"``. Kept — private and
self-contained (torch only) — for experimentation. Vendors the two tiny torch
geometry helpers it needs so it has no dependency on the test oracle.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass

import torch
import warp as wp


@dataclass
class Centerline:
    """A closed, ordered, dense centerline batch (points [E, M, 2], valid [E])."""

    points: torch.Tensor
    valid: torch.Tensor


class CenterlineGenerator(abc.ABC):
    """Interface a centerline generator implements: one Centerline per env id."""

    @abc.abstractmethod
    def generate(self, ids: torch.Tensor) -> Centerline:
        raise NotImplementedError


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / norm.clamp_min(eps)


def _turning_number(points: torch.Tensor) -> torch.Tensor:
    """Signed total turning of a closed polygon (±2π for a simple loop)."""
    nxt = torch.roll(points, shifts=-1, dims=1)
    dirs = _safe_normalize(nxt - points)
    theta = torch.atan2(dirs[..., 1], dirs[..., 0])
    dtheta = theta - torch.roll(theta, shifts=1, dims=1)
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))  # wrap into (-pi, pi]
    return dtheta.sum(dim=1)


class FourierCenterlineGenerator(CenterlineGenerator):
    """Truncated-Fourier centerline generator: smooth-by-construction closed curves."""

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.device = config.device
        self.K = config.num_harmonics
        self.M = config.num_centerline_samples

        # Dense parameter grid over [0, 2*pi) (endpoint excluded so the loop closes cleanly).
        t = torch.linspace(0.0, 2.0 * math.pi, self.M + 1, device=self.device)[:-1]  # [M]
        k = torch.arange(1, self.K + 1, device=self.device, dtype=torch.float32)  # [K]
        self.cos_kt = torch.cos(k.unsqueeze(1) * t.unsqueeze(0))  # [K, M]
        self.sin_kt = torch.sin(k.unsqueeze(1) * t.unsqueeze(0))  # [K, M]
        # Per-harmonic std: amplitude / k**decay_p.
        self.std_k = config.amplitude / (k**config.decay_p)  # [K]

    def generate(self, ids: torch.Tensor) -> Centerline:
        # Sample standard normals (float args), then scale by the per-harmonic decay in torch.
        # NOTE: do NOT pass a tensor std into sample_normal_warp (warp dispatch rejects
        # float-mean / tensor-std, and only honors a per-env scalar std).
        wp_ids = wp.from_torch(ids.to(torch.int32), dtype=wp.int32)
        a = wp.to_torch(self.rng.sample_normal_warp(0.0, 1.0, (self.K, 2), ids=wp_ids))  # [E, K, 2]
        b = wp.to_torch(self.rng.sample_normal_warp(0.0, 1.0, (self.K, 2), ids=wp_ids))  # [E, K, 2]
        a = a * self.std_k.view(1, self.K, 1)
        b = b * self.std_k.view(1, self.K, 1)

        # c(t) = sum_k a_k cos(k t) + b_k sin(k t); c0 omitted (cancels under mean-centering).
        curve = torch.einsum("ekd,km->emd", a, self.cos_kt) + torch.einsum("ekd,km->emd", b, self.sin_kt)

        curve = curve - curve.mean(dim=1, keepdim=True)
        bbox = curve.amax(dim=1) - curve.amin(dim=1)  # [E, 2]
        longest = bbox.amax(dim=1, keepdim=True).clamp_min(1e-8)  # [E, 1]
        curve = curve * (self.config.scale / longest).unsqueeze(1)

        # valid via the turning-number safety net for rare low-K crossings.
        turn = _turning_number(curve)  # [E]
        valid = (turn.abs() - 2.0 * math.pi).abs() <= self.config.turning_tol

        return Centerline(points=curve, valid=valid)
