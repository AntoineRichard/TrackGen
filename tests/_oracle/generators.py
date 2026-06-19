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

from .geometry import arc_length_resample, ccw_sort_count, safe_normalize, self_intersections, turning_number


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

    def _prune_corners(self, points: torch.Tensor, ids: torch.Tensor):
        """Sample a per-env corner count, then prune-then-sort: angle-sort the first
        ``count`` corners about THEIR OWN centroid (rows >= count are NaN).

        Sorting the kept subset about its own centre yields an angularly-monotone (winding
        +-1) polygon, avoiding the figure-eight that sort-then-prune produced (the kept
        corners were a mis-centered angular wedge with a long Bezier closing chord).

        Args:
            points: [E, max_num_points, 2] raw sampled corners.
            ids: [E] env ids (for per-env reproducible count sampling).

        Returns:
            (pruned [E, max_num_points, 2], count [E] long) where rows >= count are NaN.
        """
        E, P, _ = points.shape

        # Per-env corner count in [min_num_points, max_num_points] (inclusive).
        # sample_integer_torch samples in [low, high); high = max+1 for an inclusive upper bound.
        count = self.rng.sample_integer_torch(
            self.config.min_num_points,
            self.config.max_num_points + 1,
            (1,),
            ids=ids,
        ).view(E).long()
        count = count.clamp(max=P)

        pruned = ccw_sort_count(points, count)  # sort first `count` about own centroid; NaN tail
        return pruned, count

    def _assemble_centerline(self, corners: torch.Tensor) -> torch.Tensor:
        """Build the closed dense centerline from ccw-ordered (possibly NaN-padded) corners.

        Closes the loop with a real cubic Bezier over ALL ``count`` real corners: vertex
        tangents and segments wrap mod ``count`` (the number of finite corner rows), so the
        closing segment (corner ``count-1`` -> corner ``0``) is a genuine Bezier rather than
        the old straight chord that dropped the first & last corner. Rows ``i >= count`` are
        NaN (the pruned tail); their segments stay NaN and are dropped by the downstream
        arc-length resample.

        Args:
            corners: [E, P, 2]; NaN rows are pruned corners.

        Returns:
            [E, P * num_points_per_segment, 2] closed dense polyline (NaN where pruned).
        """
        E, P, _ = corners.shape
        device = corners.device
        # count = number of real (finite) corner rows per env; clamp guards the wrap of a
        # degenerate all-NaN env.
        count = torch.isfinite(corners).all(dim=-1).sum(dim=1)             # [E]
        cnt = count.clamp(min=1)

        row = torch.arange(P, device=device).unsqueeze(0).expand(E, -1)    # [E, P]
        nxt = torch.where(row + 1 < cnt.unsqueeze(1), row + 1, torch.zeros_like(row))
        prv = torch.where(row - 1 >= 0, row - 1, cnt.unsqueeze(1) - 1)

        def _gather(t, idx):
            return torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.size(-1)))

        c_i = corners
        c_next = _gather(corners, nxt)
        c_prev = _gather(corners, prv)
        # count-aware vertex tangents (geometry.vertex_tangents formula, wrapped mod count):
        # tangent_i = normalize(p * dir(i->next) + (1-p) * dir(prev->i)), with the derived
        # edgy-based blend weight self.p. NaN at pruned corners propagates via safe_normalize.
        u_out = safe_normalize(c_next - c_i)
        u_in = safe_normalize(c_i - c_prev)
        tangents = safe_normalize(self.p * u_out + (1.0 - self.p) * u_in)   # [E, P, 2]
        tan_next = _gather(tangents, nxt)

        # Cubic Bezier per segment i: corner i (tangent i) -> next corner (tangent next),
        # inner handles at rad*chord along the tangents. Segments with c_i NaN (i >= count)
        # stay NaN. Layout [E, P, npseg, 2] -> [E, P*npseg, 2] matches the Warp _assemble_k.
        chord = torch.linalg.norm(c_next - c_i, dim=-1, keepdim=True)       # [E, P, 1]
        # F2 adaptive clamp: cap each corner's handle at handle_clamp_frac * (its shorter
        # incident edge), so a long handle can't overshoot past a nearby corner and self-cross.
        edge_in = torch.linalg.norm(c_i - c_prev, dim=-1, keepdim=True)
        scale = torch.minimum(chord, edge_in)                              # [E, P, 1] shorter edge
        scale_next = _gather(scale, nxt)
        # getattr default disables the clamp for lightweight test configs lacking the field;
        # mirrors the Warp assemble() wrapper so oracle<->warp parity holds on any config.
        frac = getattr(self.config, "handle_clamp_frac", 1.0e9)
        h0 = torch.minimum(self.config.rad * chord, frac * scale)
        h1 = torch.minimum(self.config.rad * chord, frac * scale_next)
        p1 = c_i + tangents * h0
        p2 = c_next - tan_next * h1
        curve = (
            torch.einsum("s,epd->epsd", self.bernstein_0, c_i)
            + torch.einsum("s,epd->epsd", self.bernstein_1, p1)
            + torch.einsum("s,epd->epsd", self.bernstein_2, p2)
            + torch.einsum("s,epd->epsd", self.bernstein_3, c_next)
        )
        return curve.reshape(E, P * self.config.num_points_per_segment, 2)

    def _corner_angles(self, corners: torch.Tensor) -> torch.Tensor:
        """Interior angle at each corner via clamped arccos (NaN-safe).

        Args:
            corners: [E, P, 2] (may contain NaN pruned rows).

        Returns:
            [E, P] angles in radians; degenerate/NaN corners -> 0.0 (always fail).
            Boundary corners whose rolled neighbour is NaN also yield 0.0.
        """
        eps = 1e-7
        prev = torch.roll(corners, 1, dims=1)
        nxt = torch.roll(corners, -1, dims=1)
        u_in = safe_normalize(corners - prev)
        u_out = safe_normalize(nxt - corners)
        cos_turn = (u_in * u_out).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        angle = math.pi - torch.arccos(cos_turn)  # interior angle
        return torch.nan_to_num(angle, nan=0.0)

    def _real_turning_and_finite(self, dense: torch.Tensor):
        """Per-env turning number + finiteness computed over REAL (non-NaN) points only.

        The NaN-padded dense buffer would otherwise poison turning_number for any
        pruned (variable-count) env. We compact each env to a fixed-N real
        centerline via arc_length_resample (which drops NaN points), then gate on
        that. An env with < 2 real points yields turn = nan (fails) and finite = False.

        Args:
            dense: [n, M_max, 2] candidate centerlines (may contain NaN).

        Returns:
            (turn [n], finite_ok [n] bool).
        """
        # Resample onto a fixed-N real loop (NaN dropped); count[e] == 0 for all-NaN env.
        resampled, count = arc_length_resample(dense, num=self.config.num_points_per_segment)
        turn = turning_number(resampled)  # [n]; nan where the loop is degenerate/NaN
        finite_ok = (count >= 2) & torch.isfinite(turn)
        return turn, finite_ok

    def _generate_batch(self, ids: torch.Tensor):
        """One full draw for the given ids: corners -> prune -> dense centerline + control corners."""
        raw = self._sample_corner_points(ids)
        pruned, _count = self._prune_corners(raw, ids)
        dense = self._assemble_centerline(pruned)
        return dense, pruned

    def generate(self, ids: torch.Tensor) -> Centerline:
        E = len(ids)
        M_max = self.config.max_num_points * self.config.num_points_per_segment

        points = torch.full((E, M_max, 2), float("nan"), device=self.device)
        valid = torch.zeros((E,), dtype=torch.bool, device=self.device)
        pending = torch.arange(E, device=self.device)  # local rows still needing a good draw

        for _ in range(self.config.max_regen_iters):
            if pending.numel() == 0:
                break
            sub_ids = ids[pending]
            dense, corners = self._generate_batch(sub_ids)

            # Gate 1: every REAL interior corner (with real neighbours) exceeds min_angle.
            # NaN corners (pruned) yield angle 0.0 via nan_to_num but are excluded.
            # Boundary corners whose rolled neighbour is NaN (roll wraps into padding)
            # are also excluded — they belong to the circular closure of real corners.
            angles = self._corner_angles(corners)  # [n, P]
            real_corner = torch.isfinite(corners).all(dim=-1)  # [n, P] bool
            prev_real = torch.roll(real_corner, 1, dims=1)
            next_real = torch.roll(real_corner, -1, dims=1)
            # Corner is "constrained" only when it and both its neighbours are real.
            constrained = real_corner & prev_real & next_real
            # A constrained corner must pass; unconstrained corners are irrelevant.
            angle_ok = ((angles > self.config.min_angle) | ~constrained).all(dim=1)
            # Gates 2 & 3: turning number ~ 2*pi AND finite, evaluated on REAL points only.
            turn, finite_ok = self._real_turning_and_finite(dense)
            turn_ok = (turn.abs() - 2.0 * math.pi).abs() <= self.config.turning_tol
            # Gate 4: the dense centerline must be a SIMPLE (non-self-intersecting) loop
            # AT THE RESOLUTION THE PIPELINE USES. Relaxation by repulsion cannot untangle
            # a global self-crossing, so reject it here. We test on an arc-length resample
            # at the pipeline's output resolution (drops NaN, reconnects pruned gaps): this
            # catches genuine global crossings while ignoring sub-resolution corner cusps
            # that (a) the pipeline never sees and (b) the relaxation's bending rounds out
            # anyway. Falls back to 256 so lightweight unit-test configs (no num_points
            # field) still work.
            simple_n = int(getattr(self.config, "num_points", 256) or 256)
            simple_resampled, _ = arc_length_resample(dense, num=simple_n)
            simple_ok = self_intersections(simple_resampled) == 0
            ok = angle_ok & turn_ok & finite_ok & simple_ok

            good = pending[ok]
            points[good] = dense[ok]
            valid[good] = True
            pending = pending[~ok]

        return Centerline(points=points, valid=valid)


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
        self.t = t
        k = torch.arange(1, self.K + 1, device=self.device, dtype=torch.float32)  # [K]
        self.cos_kt = torch.cos(k.unsqueeze(1) * t.unsqueeze(0))  # [K, M]
        self.sin_kt = torch.sin(k.unsqueeze(1) * t.unsqueeze(0))  # [K, M]
        # Per-harmonic std: amplitude / k**decay_p.
        self.std_k = config.amplitude / (k**config.decay_p)  # [K]

    def generate(self, ids: torch.Tensor) -> Centerline:
        # Sample standard normals (float args), then scale by the per-harmonic decay in torch.
        # NOTE: do NOT pass a tensor std into sample_normal_torch (warp dispatch rejects
        # float-mean / tensor-std, and only honors a per-env scalar std).
        a = self.rng.sample_normal_torch(0.0, 1.0, (self.K, 2), ids=ids)  # [E, K, 2]
        b = self.rng.sample_normal_torch(0.0, 1.0, (self.K, 2), ids=ids)  # [E, K, 2]
        a = a * self.std_k.view(1, self.K, 1)
        b = b * self.std_k.view(1, self.K, 1)

        # c(t) = sum_k a_k cos(k t) + b_k sin(k t); c0 omitted (cancels under mean-centering).
        curve = torch.einsum("ekd,km->emd", a, self.cos_kt) + torch.einsum("ekd,km->emd", b, self.sin_kt)

        curve = curve - curve.mean(dim=1, keepdim=True)
        bbox = curve.amax(dim=1) - curve.amin(dim=1)  # [E, 2]
        longest = bbox.amax(dim=1, keepdim=True).clamp_min(1e-8)  # [E, 1]
        curve = curve * (self.config.scale / longest).unsqueeze(1)

        # valid via the turning-number safety net for rare low-K crossings.
        turn = turning_number(curve)  # [E]
        valid = (turn.abs() - 2.0 * math.pi).abs() <= self.config.turning_tol

        return Centerline(points=curve, valid=valid)
