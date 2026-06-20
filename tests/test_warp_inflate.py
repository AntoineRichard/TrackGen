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

Track fields are wp.array; test reads are wrapped with to_t() from _warp_compare
to convert to torch tensors at the oracle boundary.
"""

import math

import pytest
import torch

pytest.importorskip("warp")

from tests._oracle import inflation  # noqa: E402
from tests._warp_compare import to_t  # noqa: E402
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

    got = wpl.inflate_warp(cs_center, config, valid=valid, count=count)
    assert isinstance(got, Track)

    # count is the per-env real-point count; Track.count must echo it exactly.
    assert torch.equal(to_t(got.count).cpu(), count.cpu())
    # Both loops resample to >=2 real points (otherwise the comparisons below are vacuous).
    assert (count >= 3).all()

    # The Warp inflate re-resamples its real points (resample_uniform) BEFORE building the
    # frame/offset. The oracle reference for center is therefore the independent re-sample
    # of cs_center: rs_ref = resample_uniform(cs_center, ..., count=count).
    # We compute rs_ref here (from the INPUT, not from got.center) and use it as the
    # oracle for all field comparisons below — this is the real cross-check.
    import warp as wp
    E = 2
    n_max = config.N_max
    count_i32 = count.to(torch.int32)
    rs_ref = wpl.resample_uniform(cs_center, n_max, count=count_i32)  # [E, N, 2] torch
    rs_wp = wp.to_torch(got.center).view(E, n_max, 2)   # [E, N, 2] torch (from output)
    T, Nrm, kappa = inflation._frame_curvature_stage(rs_ref)
    w = inflation._width_stage(rs_ref, kappa, config)
    outer, inner = inflation._offset_stage(rs_ref, Nrm, w)
    arclen, length = inflation._arclength(rs_ref, count)

    for e in range(2):
        c = int(count[e].item())

        # center: inflate_warp stores resample_uniform(cs_center) in Track.center.
        # Cross-check the output against the independently-computed rs_ref (derived from
        # the INPUT, not from got.center itself — avoids the tautological self-comparison).
        assert torch.allclose(rs_wp[e, :c], rs_ref[e, :c], atol=1e-4), f"center env{e} on {dev}"
        assert torch.isnan(rs_wp[e, c:]).all(), f"center pad env{e} on {dev}"

        # tangent/normal/outer/inner: oracle is count-aware everywhere EXCEPT the two seam
        # points (0 and c-1), which its non-count-aware central difference poisons with NaN.
        # Compare the interior real points against the oracle; check the seam independently.
        interior = slice(1, c - 1)
        field_map = {
            "tangent": (wp.to_torch(got.tangent).view(E, n_max, 2), T),
            "normal": (wp.to_torch(got.normal).view(E, n_max, 2), Nrm),
            "outer": (wp.to_torch(got.outer).view(E, n_max, 2), outer),
            "inner": (wp.to_torch(got.inner).view(E, n_max, 2), inner),
        }
        for field, (g_full, ref) in field_map.items():
            g = g_full
            assert torch.allclose(g[e, interior], ref[e, interior], atol=1e-4), \
                f"{field} interior env{e} on {dev}"
            # Warp is count-aware: every real point (incl. the seam) must be finite.
            assert torch.isfinite(g[e, :c]).all(), f"{field} real-finite env{e} on {dev}"
            # Padding is NaN beyond the real count.
            assert torch.isnan(g[e, c:]).all(), f"{field} pad env{e} on {dev}"

        # Frame relations over ALL real points (seam included): unit tangent/normal,
        # normal is the left-rotation of tangent, and outer/inner == center +/- hw*normal.
        tg_full = wp.to_torch(got.tangent).view(E, n_max, 2)
        ng_full = wp.to_torch(got.normal).view(E, n_max, 2)
        tg = tg_full[e, :c]
        ng = ng_full[e, :c]
        assert torch.allclose(torch.linalg.norm(tg, dim=-1),
                              torch.ones(c, device=cs_center.device), atol=1e-4), \
            f"|tangent| env{e} on {dev}"
        assert torch.allclose(torch.linalg.norm(ng, dim=-1),
                              torch.ones(c, device=cs_center.device), atol=1e-4), \
            f"|normal| env{e} on {dev}"
        left = torch.stack([-tg[:, 1], tg[:, 0]], dim=-1)
        assert torch.allclose(ng, left, atol=1e-4), f"normal=left(tangent) env{e} on {dev}"
        hw = float(config.half_width)
        outer_e = wp.to_torch(got.outer).view(E, n_max, 2)[e, :c]
        inner_e = wp.to_torch(got.inner).view(E, n_max, 2)[e, :c]
        rs_e = rs_wp[e, :c]
        assert torch.allclose(outer_e, rs_e + hw * ng, atol=1e-4) or \
               torch.allclose(outer_e, rs_e - hw * ng, atol=1e-4), \
            f"outer = center +/- hw*normal env{e} on {dev}"
        # outer and inner straddle the centerline: their midpoint is the centerline.
        assert torch.allclose(0.5 * (outer_e + inner_e), rs_e, atol=1e-4), \
            f"center == midpoint(outer, inner) env{e} on {dev}"

        # arclen: count-aware on both sides; real points match, padding is NaN.
        arclen_e = wp.to_torch(got.arclen).view(E, n_max)[e]
        assert torch.allclose(arclen_e[:c], arclen[e, :c], atol=1e-3), f"arclen env{e} on {dev}"
        assert torch.isnan(arclen_e[c:]).all(), f"arclen pad env{e} on {dev}"

    # length: count-aware total perimeter, matches the oracle's count-aware _arclength.
    length_t = wp.to_torch(got.length)
    assert torch.allclose(length_t, length, atol=1e-3), f"length on {dev}"

    # Validity is decided by the Warp count-masked _validity_k (the oracle's _validity_stage
    # is NOT count-aware -> NaN-poisoned in constant_spacing, so it is NOT a usable reference).
    # The fat circle is thick enough to be valid; the thin ellipse is not.
    valid_t = wp.to_torch(got.valid).bool()
    assert valid_t.cpu().tolist() == [True, False], f"valid on {dev}"


@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_valid_defaults_to_all_true(dev):
    # Omitting `valid` must behave like passing an all-True generation flag: the Warp
    # validity (the authoritative count-masked gate) is identical either way.
    config = TrackGenConfig(half_width=0.1, num_points=256, N_max=256)
    cs_center, count = _build_cs(dev, config)

    ref = wpl.inflate_warp(
        cs_center, config, valid=torch.ones(2, dtype=torch.bool, device=dev), count=count
    )
    got = wpl.inflate_warp(cs_center, config, count=count)  # valid omitted -> all-True default
    import warp as wp
    assert torch.equal(wp.to_torch(got.valid).bool().cpu(), wp.to_torch(ref.valid).bool().cpu())
    assert wp.to_torch(got.valid).bool().cpu().tolist() == [True, False]


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
