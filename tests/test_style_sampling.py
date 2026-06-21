"""Tests for per-env style sampling (method #1: "Per-env style randomization").

Two guarantees:
  (a) style_sampling=False reproduces the current bezier centerline BIT-FOR-BIT (the
      default path uses the original scalar kernels, untouched). Additionally, the style
      kernels with *_range=None (each knob collapsed to its config scalar -> uniform per-env
      arrays equal to the scalars) reproduce the default path bit-for-bit too, proving the
      style kernels are bit-identical to the scalar kernels given uniform inputs.
  (b) style_sampling=True with non-trivial ranges yields MORE per-env diversity than the
      un-sampled batch (higher variance of centerline length AND compactness across envs),
      measured with benchmarks.track_metrics.

Reproducibility is asserted WITHIN a device only (Warp RNG may differ cpu vs cuda).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

pytest.importorskip("warp")

import warp as wp  # noqa: E402
wp.init()

from track_gen._src import warp_pipeline as wpl  # noqa: E402
from track_gen._src.types import TrackGenConfig  # noqa: E402
from benchmarks import track_metrics as tm  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

E = 256


def _generate(config, dev):
    """Run generate_centerline_warp into a fresh scratch and return the [E,N,2] centerline."""
    N = int(config.num_points)
    _, scratch = wpl._inflate_warp_alloc(config)
    seeds_t = torch.arange(E, dtype=torch.int32, device=dev)
    seeds_wp = wp.from_torch(seeds_t, dtype=wp.int32)
    wpl.generate_centerline_warp(
        seeds_wp, config,
        out_centerline=scratch.gen_centerline,
        out_valid_wp=scratch.gen_valid,
        scratch=scratch,
    )
    if "cuda" in dev:
        wp.synchronize()
    return wp.to_torch(scratch.gen_centerline).view(E, N, 2).clone()


def _real_points(cl_e: torch.Tensor) -> np.ndarray:
    """Drop NaN-padded rows from one env's [N,2] centerline -> numpy [n,2]."""
    arr = cl_e.cpu().numpy()
    return arr[np.isfinite(arr).all(axis=1)]


def _per_env_metric_variance(cl: torch.Tensor):
    """Return (length_variance, compactness_variance) across the E envs of a [E,N,2] batch."""
    lengths, compact = [], []
    for e in range(cl.shape[0]):
        pts = _real_points(cl[e])
        if len(pts) < 4 or not np.isfinite(pts).all():
            continue
        lengths.append(tm.perimeter(pts))
        compact.append(tm.compactness(pts))
    return float(np.var(lengths)), float(np.var(compact)), len(lengths)


# ---------------------------------------------------------------------------
# (a) style_sampling=False is byte-for-byte the current bezier output.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dev", DEVS)
def test_style_off_is_bitexact_default(dev):
    config = TrackGenConfig(num_envs=E, device=dev)  # style_sampling defaults to False
    a = _generate(config, dev)
    b = _generate(config, dev)
    # Deterministic + identical to itself (sanity).
    assert torch.equal(a, b)
    # The flag is OFF by default; this IS the current bezier path. Re-run with the flag
    # explicitly False to confirm no hidden dependence on the new fields.
    config_explicit = dataclasses.replace(
        config, style_sampling=False,
        rad_range=None, scale_range=None, handle_clamp_frac_range=None)
    c = _generate(config_explicit, dev)
    assert torch.equal(a, c)


@pytest.mark.parametrize("dev", DEVS)
def test_style_kernels_match_scalar_when_ranges_none(dev):
    """style_sampling=True but every *_range=None -> per-env arrays collapse to the config
    scalars; the bit-identical style kernels must reproduce the default scalar path exactly."""
    base = TrackGenConfig(num_envs=E, device=dev)
    default_cl = _generate(base, dev)

    style_collapsed = dataclasses.replace(
        base, style_sampling=True,
        rad_range=None, scale_range=None, handle_clamp_frac_range=None)
    style_cl = _generate(style_collapsed, dev)

    assert torch.equal(default_cl, style_cl), (
        "style kernels with collapsed (scalar) ranges must match the scalar path bit-for-bit")


# ---------------------------------------------------------------------------
# (b) style_sampling=True yields MORE per-env diversity.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dev", DEVS)
def test_style_on_increases_diversity(dev):
    base = TrackGenConfig(num_envs=E, device=dev)
    base_cl = _generate(base, dev)

    styled = dataclasses.replace(
        base,
        style_sampling=True,
        rad_range=(0.15, 0.65),
        scale_range=(0.6, 1.4),
        handle_clamp_frac_range=(0.15, 0.65),
    )
    styled_cl = _generate(styled, dev)

    base_var_len, base_var_comp, n_base = _per_env_metric_variance(base_cl)
    sty_var_len, sty_var_comp, n_sty = _per_env_metric_variance(styled_cl)

    # Enough valid envs to make the variance comparison meaningful.
    assert n_base > E // 2 and n_sty > E // 2

    assert sty_var_len > base_var_len, (
        f"style ON length variance {sty_var_len:.4g} !> base {base_var_len:.4g}")
    assert sty_var_comp > base_var_comp, (
        f"style ON compactness variance {sty_var_comp:.4g} !> base {base_var_comp:.4g}")


@pytest.mark.parametrize("dev", DEVS)
def test_style_on_is_reproducible(dev):
    styled = TrackGenConfig(
        num_envs=E, device=dev, style_sampling=True,
        rad_range=(0.15, 0.65), scale_range=(0.6, 1.4),
        handle_clamp_frac_range=(0.15, 0.65))
    a = _generate(styled, dev)
    b = _generate(styled, dev)
    assert torch.equal(a, b)
