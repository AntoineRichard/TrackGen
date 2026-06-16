# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch


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
