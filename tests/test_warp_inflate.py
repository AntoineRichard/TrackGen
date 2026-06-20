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
to convert to tensors at the oracle boundary.
"""

import math

import pytest
import torch

pytest.importorskip("warp")

import warp as wp  # noqa: E402
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
    """Constant-spacing-resample the dense loops via the pure-Warp in-place API.

    Returns (cs_center_wp, count_wp) — both wp.arrays — matching the production
    path that inflate_warp receives on the owned pipeline.
    """
    center = _build_center(dev)
    E, N, _ = center.shape
    cf = wp.from_torch(center.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    n_max = config.N_max
    out_wp = wp.empty(E * n_max, dtype=wp.vec2f, device=dev)
    count_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wpl.resample_constant_spacing(cf, float(config.spacing), n_max,
                                  out_wp=out_wp, count_wp=count_wp)
    return out_wp, count_wp


@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_matches_oracle(dev):
    config = TrackGenConfig(half_width=0.1, num_points=256, N_max=256)
    cs_center_wp, count_wp = _build_cs(dev, config)
    E = 2
    n_max = config.N_max

    valid_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wp.launch(wpl._fill_i32_k, dim=E, inputs=[valid_wp, 1], device=dev)

    got = wpl.inflate_warp(cs_center_wp, config, valid=valid_wp, count=count_wp)
    assert isinstance(got, Track)

    count_t = wp.to_torch(count_wp).long()
    # count is the per-env real-point count; Track.count must echo it exactly.
    assert torch.equal(to_t(got.count).cpu(), count_t.cpu())
    # Both loops resample to >=2 real points (otherwise the comparisons below are vacuous).
    assert (count_t >= 3).all()

    # The Warp inflate re-resamples its real points (resample_uniform) BEFORE building the
    # frame/offset. The oracle reference for center is therefore the independent re-sample
    # of cs_center: compute rs_ref from the INPUT (not from got.center).
    _flat = E * n_max
    _dev = str(cs_center_wp.device)
    _out_wp = wp.empty(_flat, dtype=wp.vec2f, device=_dev)
    wpl.resample_uniform(cs_center_wp, _out_wp, n_max, count_wp, device=_dev)
    if "cuda" in _dev:
        wp.synchronize()
    rs_ref = wp.to_torch(_out_wp).view(E, n_max, 2)  # [E, N, 2] torch
    rs_wp = wp.to_torch(got.center).view(E, n_max, 2)

    # Oracle expects torch tensor [E, N, 2]; give it rs_ref.
    cs_center_t = wp.to_torch(cs_center_wp).view(E, n_max, 2)
    T, Nrm, kappa = inflation._frame_curvature_stage(rs_ref)
    w = inflation._width_stage(rs_ref, kappa, config)
    outer, inner = inflation._offset_stage(rs_ref, Nrm, w)
    arclen, length = inflation._arclength(rs_ref, count_t)

    for e in range(2):
        c = int(count_t[e].item())

        # center: inflate_warp stores resample_uniform(cs_center) in Track.center.
        assert torch.allclose(rs_wp[e, :c], rs_ref[e, :c], atol=1e-4), f"center env{e} on {dev}"
        assert torch.isnan(rs_wp[e, c:]).all(), f"center pad env{e} on {dev}"

        # tangent/normal/outer/inner: compare interior real points against the oracle.
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
            assert torch.isfinite(g[e, :c]).all(), f"{field} real-finite env{e} on {dev}"
            assert torch.isnan(g[e, c:]).all(), f"{field} pad env{e} on {dev}"

        # Frame relations over ALL real points (seam included).
        tg_full = wp.to_torch(got.tangent).view(E, n_max, 2)
        ng_full = wp.to_torch(got.normal).view(E, n_max, 2)
        tg = tg_full[e, :c]
        ng = ng_full[e, :c]
        assert torch.allclose(torch.linalg.norm(tg, dim=-1),
                              torch.ones(c, device=rs_wp.device), atol=1e-4), \
            f"|tangent| env{e} on {dev}"
        assert torch.allclose(torch.linalg.norm(ng, dim=-1),
                              torch.ones(c, device=rs_wp.device), atol=1e-4), \
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
        assert torch.allclose(0.5 * (outer_e + inner_e), rs_e, atol=1e-4), \
            f"center == midpoint(outer, inner) env{e} on {dev}"

        # arclen: count-aware on both sides.
        arclen_e = wp.to_torch(got.arclen).view(E, n_max)[e]
        assert torch.allclose(arclen_e[:c], arclen[e, :c], atol=1e-3), f"arclen env{e} on {dev}"
        assert torch.isnan(arclen_e[c:]).all(), f"arclen pad env{e} on {dev}"

    # length: count-aware total perimeter.
    length_t = wp.to_torch(got.length)
    assert torch.allclose(length_t, length, atol=1e-3), f"length on {dev}"

    # Validity: the fat circle is valid; the thin ellipse is not.
    valid_t = wp.to_torch(got.valid).bool()
    assert valid_t.cpu().tolist() == [True, False], f"valid on {dev}"


@pytest.mark.parametrize("dev", DEVS)
def test_inflate_warp_valid_defaults_to_all_true(dev):
    # Omitting `valid` must behave like passing an all-True generation flag.
    config = TrackGenConfig(half_width=0.1, num_points=256, N_max=256)
    cs_center_wp, count_wp = _build_cs(dev, config)
    E = 2

    valid_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wp.launch(wpl._fill_i32_k, dim=E, inputs=[valid_wp, 1], device=dev)

    ref = wpl.inflate_warp(cs_center_wp, config, valid=valid_wp, count=count_wp)
    got = wpl.inflate_warp(cs_center_wp, config, count=count_wp)  # valid omitted -> all-True

    assert torch.equal(wp.to_torch(got.valid).bool().cpu(), wp.to_torch(ref.valid).bool().cpu())
    assert wp.to_torch(got.valid).bool().cpu().tolist() == [True, False]


def test_inflate_warp_count_none_rejects_wrong_N():
    # The count=None convenience path requires the centerline to be exactly
    # config.num_points wide and rejects a mismatched N.
    n_wrong = 200
    config = TrackGenConfig(half_width=0.1, num_points=256)
    E = 2
    dev = "cpu"
    # Build a wp.array with wrong N
    center_wp = wp.zeros(E * n_wrong, dtype=wp.vec2f, device=dev)
    with pytest.raises(AssertionError):
        wpl.inflate_warp(center_wp, config)  # count=None -> fixed-N convenience path


def test_constant_spacing_is_the_only_output_mode():
    # The legacy "fixed" (constant point COUNT) output mode was dropped.
    with pytest.raises(ValueError):
        TrackGenConfig(half_width=0.1, num_points=256, output_mode="fixed")
