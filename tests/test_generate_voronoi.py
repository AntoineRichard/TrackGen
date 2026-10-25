"""End-to-end tests for the standard Voronoi first-stage generator.

Voronoi's own properties: a reasonable valid yield, crossing-free generated loops, a
site count that changes the tracks, and shapes that are not round blobs. The generator
contract (finite points before count and NaN padding after it, count range, determinism,
the hot path) is checked for every generator by the conformance kit, in
tests/test_conformance.py.
"""
from __future__ import annotations
from tests._reference_generators import generate_raw

import numpy as np
import pytest
import torch

pytest.importorskip("warp")

import warp as wp  # noqa: E402

from track_gen import PerEnvSeededRNG  # noqa: E402
from track_gen._src import warp_generate_voronoi  # noqa: E402
from track_gen._src.track_generator import Track, TrackGenConfig, TrackGenerator  # noqa: E402
from tests._warp_compare import self_intersections, to_t  # noqa: E402

from tests.conftest import DEVS  # noqa: E402


def _make_rng(num_envs: int, seed: int = 0, device: str = "cpu"):
    return PerEnvSeededRNG(seeds=seed, num_envs=num_envs, device=device)


def _compactness(pts: np.ndarray) -> np.ndarray:
    nxt = np.roll(pts, -1, axis=1)
    perimeter = np.linalg.norm(nxt - pts, axis=2).sum(axis=1)
    area = 0.5 * np.abs(
        (pts[:, :, 0] * nxt[:, :, 1] - nxt[:, :, 0] * pts[:, :, 1]).sum(axis=1)
    )
    return 4.0 * np.pi * area / np.maximum(perimeter * perimeter, 1.0e-12)


def test_voronoi_config_defaults_are_available():
    cfg = TrackGenConfig(generator="voronoi")

    assert cfg.voronoi_num_sites >= cfg.voronoi_control_points
    assert cfg.voronoi_site_layout in {"ring", "void_ring", "clustered", "mixed"}
    assert cfg.voronoi_control_points >= 6
    assert cfg.voronoi_radial_variation > 0.0


@pytest.mark.parametrize("dev", DEVS)
def test_voronoi_end_to_end(dev):
    E, n_max = 64, 256
    cfg = TrackGenConfig(
        generator="voronoi",
        num_envs=E,
        num_points=128,
        N_max=n_max,
        half_width=0.1,
        device=dev,
    )
    gen = TrackGenerator(cfg, _make_rng(E, device=dev))

    track = gen.generate(E)

    assert isinstance(track, Track)
    assert float(to_t(track.valid).bool().float().mean()) > 0.50


@pytest.mark.parametrize("dev", DEVS)
def test_voronoi_generation_outputs_simple_finite_centerlines(dev):
    E = 128
    cfg = TrackGenConfig(
        generator="voronoi",
        num_envs=E,
        num_points=128,
        device=dev,
    )
    # The frozen legacy entry points (the registered "voronoi" is a Stage since PR13c).
    scratch = warp_generate_voronoi._alloc_voronoi_scratch(cfg)
    seeds_t = torch.arange(E, dtype=torch.int32, device=dev)
    seeds_wp = wp.from_torch(seeds_t, dtype=wp.int32)
    out = wp.empty(E * int(cfg.num_points), dtype=wp.vec2f, device=dev)
    valid_wp = wp.empty(E, dtype=wp.int32, device=dev)

    generate_raw("voronoi", seeds_wp, cfg, out, valid_wp, scratch)
    if "cuda" in dev:
        wp.synchronize()

    centerline = to_t(out).view(E, int(cfg.num_points), 2)
    valid = to_t(valid_wp).bool()

    assert valid.all()
    assert torch.isfinite(centerline).all()
    crossing_free = (self_intersections(centerline) == 0).float().mean().item()
    assert crossing_free >= 0.95, f"voronoi crossing-free {crossing_free} < 0.95 on {dev}"


def test_voronoi_cell_count_changes_tracks_cpu():
    # (Determinism at a given site count is the conformance kit's determinism check,
    # run on this 64-site config too in tests/test_conformance.py.)
    E = 32
    base = dict(generator="voronoi", num_envs=E, num_points=96, N_max=192, device="cpu")
    cfg_a = TrackGenConfig(**base, voronoi_num_sites=64)
    cfg_b = TrackGenConfig(**base, voronoi_num_sites=256)

    gen_a = TrackGenerator(cfg_a, _make_rng(E, seed=7, device="cpu"))
    gen_b = TrackGenerator(cfg_b, _make_rng(E, seed=7, device="cpu"))

    a1 = to_t(gen_a.generate(E).center).clone()
    b = to_t(gen_b.generate(E).center).clone()

    fin1 = torch.isfinite(a1).all(dim=-1)
    both = fin1 & torch.isfinite(b).all(dim=-1)
    assert both.any()
    assert not torch.allclose(a1[both], b[both], atol=1e-3)


def test_voronoi_default_centerlines_are_not_round_blobs_cpu():
    E = 64
    cfg = TrackGenConfig(
        generator="voronoi",
        num_envs=E,
        num_points=128,
        device="cpu",
    )
    gen = TrackGenerator(cfg, _make_rng(E, device="cpu"))
    gen.generate(E)
    centerline = wp.to_torch(gen._scratch.gen_centerline).cpu().numpy().reshape(E, cfg.num_points, 2)

    compactness = _compactness(centerline)

    assert float(compactness.mean()) < 0.88
    assert float((compactness > 0.92).mean()) < 0.25
