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

    def generate(self, ids: torch.Tensor) -> Centerline:
        raise NotImplementedError  # filled in by later tasks
