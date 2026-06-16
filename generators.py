# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import binom


@dataclass
class Centerline:
    """A closed, ordered, dense centerline batch.

    Attributes:
        points: [E, M_max, 2] closed dense samples; shorter tracks NaN-padded to M_max.
        valid: [E] bool generation-time validity (False if a generator gave up for an env).
    """

    points: torch.Tensor
    valid: torch.Tensor


class CenterlineGenerator(abc.ABC):
    """Interface every centerline generator implements.

    inflation.inflate consumes only a Centerline and never knows which generator ran.
    """

    @abc.abstractmethod
    def generate(self, ids: torch.Tensor) -> Centerline:
        """Generate one Centerline per env id in `ids`.

        Args:
            ids: [E] int tensor of environment ids to generate for.

        Returns:
            Centerline with points [E, M_max, 2] and valid [E].
        """
        raise NotImplementedError


def bernstein(n: int, k: int, t: np.ndarray) -> np.ndarray:
    """The k-th Bernstein basis polynomial of degree n at t (ported from track_generator.py)."""
    return binom(n, k) * t**k * (1.0 - t) ** (n - k)


class BezierCenterlineGenerator(CenterlineGenerator):
    """Closed-Bezier centerline generator (the repaired ccw_sort / get_bezier_curve pipeline)."""

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.device = config.device

        # p maps edginess into [0, 1]; it weights the outgoing vs incoming edge direction.
        # This is the vertex_tangents blend weight, NOT the Fourier decay exponent.
        self.p = math.atan(config.edgy) / math.pi + 0.5
        # Number of grid cells per axis; smaller min_point_distance => finer grid.
        self.num_cells = int(1.0 / (config.min_point_distance * 2))

        # Precompute the four cubic (degree-3) Bernstein basis vectors over a uniform t grid.
        t = np.linspace(0.0, 1.0, num=config.num_points_per_segment)
        self.bernstein_0 = torch.tensor(bernstein(3, 0, t), device=self.device, dtype=torch.float32)
        self.bernstein_1 = torch.tensor(bernstein(3, 1, t), device=self.device, dtype=torch.float32)
        self.bernstein_2 = torch.tensor(bernstein(3, 2, t), device=self.device, dtype=torch.float32)
        self.bernstein_3 = torch.tensor(bernstein(3, 3, t), device=self.device, dtype=torch.float32)

    def _sample_cell_indices(self, ids: torch.Tensor) -> torch.Tensor:
        """Per-env uniform subset (without replacement) of grid cell indices.

        Draws num_cells**2 i.i.d. uniforms per env; the indices of the max_num_points
        largest are a uniform k-subset without replacement (top-k trick). Device-resident,
        per-env seeded -- replaces the old numpy rng.choice host-sync path.

        Returns:
            [E, max_num_points] long tensor of cell indices in [0, num_cells**2).
        """
        n = self.num_cells * self.num_cells
        u = self.rng.sample_uniform_torch(0.0, 1.0, (n,), ids=ids)  # [E, n]
        cell_idxs = u.topk(self.config.max_num_points, dim=1).indices  # [E, max_num_points]
        return cell_idxs.long()

    def _sample_corner_points(self, ids: torch.Tensor) -> torch.Tensor:
        """Sample max_num_points corner points in scaled grid coordinates.

        Returns:
            [E, max_num_points, 2] float tensor.
        """
        cell_idxs = self._sample_cell_indices(ids)  # [E, max_num_points]
        x = (cell_idxs % self.num_cells).float()
        y = (cell_idxs // self.num_cells).float()
        # Per-corner uniform noise in [-0.5, 0.5) makes the discrete grid continuous.
        noise = self.rng.sample_uniform_torch(-0.5, 0.5, (self.config.max_num_points, 2), ids=ids)
        xy = torch.stack([x, y], dim=2) * (self.config.min_point_distance * 2.0) + noise
        return xy * self.config.scale

    def generate(self, ids: torch.Tensor) -> Centerline:
        raise NotImplementedError  # filled in by later tasks
