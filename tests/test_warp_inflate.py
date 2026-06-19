# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""inflate_warp == torch inflation stages on a constant_spacing centerline.

Builds two closed loops (a fat circle -> valid, a thin ellipse -> invalid),
constant-spacing-resamples them (the production path: NaN-padded center + per-env
real-point count), and checks the pure-Warp inflate composition field-by-field
against the torch oracle on both the Warp cpu and cuda devices.

Constant_spacing is count-aware: each env keeps only ``count[e]`` real points, the
rest of the [E, N_max, *] arrays are NaN padding. Comparisons mask to the real
``[:count[e]]`` slice per env. Two oracle caveats drive the comparison shape:

* The torch oracle's frame/offset stages (``geometry.tangents_normals`` etc.) are
  NOT count-aware: they central-difference over the full padded tensor, so the two
  seam points (index 0 and count-1) get poisoned by the adjacent NaN padding. The
  Warp ``_frame_k`` IS count-aware (wraps within ``count[e]``), so it is correct at
  the seam. We therefore compare tangent/normal/outer/inner against the oracle only
  on the INTERIOR real points and check the Warp seam points independently (finite,
  unit-norm, outer/inner == center +/- hw*normal).
* The torch oracle's ``_validity_stage`` is likewise NOT count-aware (its docstring
  says so): in constant_spacing the NaN padding poisons turning/thickness and flags
  every padded track invalid. The Warp ``_validity_k`` IS count-masked and is the
  authority, so validity is asserted directly from geometry (fat circle valid, thin
  ellipse invalid) rather than against the poisoned oracle.

center / arclen / length use the oracle's count-aware stages and match outright.
"""

import math

import pytest
import torch

pytest.importorskip("warp")

from tests._oracle import inflation  # noqa: E402
from track_gen._src import warp_pipeline as wpl  # noqa: E402
from track_gen._src.types import Track, TrackGenConfig  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _build_center(dev: str) -> torch.Tensor:
    """[2, 256, 2]: a fat circle (r=2) and a thin ellipse (x=2cos, y=0.04sin)."""
    n = 256
    t = torch.linspace(0.0, 2.0 * math.pi, n + 1, device=dev)[:-1]  # closed, no dup
    circle = torch.stack([2.0 * torch.cos(t), 2.0 * torch.sin(t)], dim=-1)
    ellipse = torch.stack([2.0 * torch.cos(t), 0.04 * torch.sin(t)], dim=-1)
    return torch.stack([circle, ellipse], dim=0).to(torch.float32)


def _build_cs(dev: str, config: TrackGenConfig):
    """Constant-spacing-resample the dense loops -> (NaN-padded center [2, N_max, 2], count [2]).

    This is exactly the production path (warp_pipeline.generate_tracks_warp's
    resample_constant_spacing), so the center handed to inflate_warp is the
    count-aware NaN-padded tensor the constant_spacing pipeline actually feeds it.
    """
    center = _build_center(dev)
    cs_center, count = wpl.resample_constant_spacing(center, float(config.spacing), config.N_max)
    return cs_center, count


@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_matches_oracle(dev):
    config = TrackGenConfig(half_width=0.1, num_points=256, N_max=256)
    cs_center, count = _build_cs(dev, config)
    valid = torch.ones(2, dtype=torch.bool, device=dev)

    got = wpl.inflate_warp(cs_center, config, valid, count=count)
    assert isinstance(got, Track)

    # count is the per-env real-point count; Track.count must echo it exactly.
    assert torch.equal(got.count.cpu(), count.cpu())
    # Both loops resample to >=2 real points (otherwise the comparisons below are vacuous).
    assert (count >= 3).all()

    # The Warp inflate re-resamples its real points (resample_uniform) BEFORE building the
    # frame/offset, so the oracle reference is its stages applied to got.center (the SAME
    # resampled centerline). center is fully count-aware on both sides; arclen/length too.
    rs = got.center
    T, Nrm, kappa = inflation._frame_curvature_stage(rs)
    w = inflation._width_stage(rs, kappa, config)
    outer, inner = inflation._offset_stage(rs, Nrm, w)
    arclen, length = inflation._arclength(rs, count)

    for e in range(2):
        c = int(count[e].item())

        # center: count-aware on both sides -> identical real points, NaN padding beyond.
        assert torch.allclose(got.center[e, :c], rs[e, :c], atol=1e-4), f"center env{e} on {dev}"
        assert torch.isnan(got.center[e, c:]).all(), f"center pad env{e} on {dev}"

        # tangent/normal/outer/inner: oracle is count-aware everywhere EXCEPT the two seam
        # points (0 and c-1), which its non-count-aware central difference poisons with NaN.
        # Compare the interior real points against the oracle; check the seam independently.
        interior = slice(1, c - 1)
        for field, ref in (("tangent", T), ("normal", Nrm), ("outer", outer), ("inner", inner)):
            g = getattr(got, field)
            assert torch.allclose(g[e, interior], ref[e, interior], atol=1e-4), \
                f"{field} interior env{e} on {dev}"
            # Warp is count-aware: every real point (incl. the seam) must be finite.
            assert torch.isfinite(g[e, :c]).all(), f"{field} real-finite env{e} on {dev}"
            # Padding is NaN beyond the real count.
            assert torch.isnan(g[e, c:]).all(), f"{field} pad env{e} on {dev}"

        # Frame relations over ALL real points (seam included): unit tangent/normal,
        # normal is the left-rotation of tangent, and outer/inner == center +/- hw*normal.
        tg = got.tangent[e, :c]
        ng = got.normal[e, :c]
        assert torch.allclose(torch.linalg.norm(tg, dim=-1),
                              torch.ones(c, device=cs_center.device), atol=1e-4), \
            f"|tangent| env{e} on {dev}"
        assert torch.allclose(torch.linalg.norm(ng, dim=-1),
                              torch.ones(c, device=cs_center.device), atol=1e-4), \
            f"|normal| env{e} on {dev}"
        left = torch.stack([-tg[:, 1], tg[:, 0]], dim=-1)
        assert torch.allclose(ng, left, atol=1e-4), f"normal=left(tangent) env{e} on {dev}"
        hw = float(config.half_width)
        assert torch.allclose(got.outer[e, :c], rs[e, :c] + hw * ng, atol=1e-4) or \
               torch.allclose(got.outer[e, :c], rs[e, :c] - hw * ng, atol=1e-4), \
            f"outer = center +/- hw*normal env{e} on {dev}"
        # outer and inner straddle the centerline: their midpoint is the centerline.
        assert torch.allclose(0.5 * (got.outer[e, :c] + got.inner[e, :c]), rs[e, :c], atol=1e-4), \
            f"center == midpoint(outer, inner) env{e} on {dev}"

        # arclen: count-aware on both sides; real points match, padding is NaN.
        assert torch.allclose(got.arclen[e, :c], arclen[e, :c], atol=1e-3), f"arclen env{e} on {dev}"
        assert torch.isnan(got.arclen[e, c:]).all(), f"arclen pad env{e} on {dev}"

    # length: count-aware total perimeter, matches the oracle's count-aware _arclength.
    assert torch.allclose(got.length, length, atol=1e-3), f"length on {dev}"

    # Validity is decided by the Warp count-masked _validity_k (the oracle's _validity_stage
    # is NOT count-aware -> NaN-poisoned in constant_spacing, so it is NOT a usable reference).
    # The fat circle is thick enough to be valid; the thin ellipse is not.
    assert got.valid.cpu().tolist() == [True, False], f"valid on {dev}"


@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_valid_defaults_to_all_true(dev):
    # Omitting `valid` must behave like passing an all-True generation flag: the Warp
    # validity (the authoritative count-masked gate) is identical either way.
    config = TrackGenConfig(half_width=0.1, num_points=256, N_max=256)
    cs_center, count = _build_cs(dev, config)

    ref = wpl.inflate_warp(
        cs_center, config, torch.ones(2, dtype=torch.bool, device=dev), count=count
    )
    got = wpl.inflate_warp(cs_center, config, count=count)  # valid omitted -> all-True default
    assert torch.equal(got.valid.cpu(), ref.valid.cpu())
    assert got.valid.cpu().tolist() == [True, False]


def test_inflate_warp_count_none_rejects_wrong_N():
    # REPURPOSED from the dropped-"fixed"-mode test (output_mode="fixed" now raises at
    # construction). The surviving contract is inflate_warp's count=None convenience path:
    # a generic fixed-N entry (NOT tied to any output_mode) that requires the centerline to
    # be exactly config.num_points wide and rejects a mismatched N. This keeps the "clean
    # fixed-N contract is enforced, not silently mismatched" intent for the path that remains.
    center = torch.zeros(2, 200, 2, dtype=torch.float32)  # N=200 != num_points
    config = TrackGenConfig(half_width=0.1, num_points=256)
    with pytest.raises(AssertionError):
        wpl.inflate_warp(center, config)  # count=None -> fixed-N convenience path


def test_constant_spacing_is_the_only_output_mode():
    # The legacy "fixed" (constant point COUNT) output mode was dropped; constructing it
    # must raise. This pins the contract the rest of this module relies on.
    with pytest.raises(ValueError):
        TrackGenConfig(half_width=0.1, num_points=256, output_mode="fixed")
