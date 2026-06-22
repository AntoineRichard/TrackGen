"""End-to-end tests for the standard Voronoi first-stage generator."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("warp")

import warp as wp  # noqa: E402

from track_gen import PerEnvSeededRNG  # noqa: E402
from track_gen._src import generator_registry as reg  # noqa: E402
from track_gen._src.track_generator import Track, TrackGenConfig, TrackGenerator  # noqa: E402
from tests._warp_compare import self_intersections, to_t  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_rng(num_envs: int, seed: int = 0, device: str = "cpu"):
    return PerEnvSeededRNG(seeds=seed, num_envs=num_envs, device=device)


def _compactness(pts: np.ndarray) -> np.ndarray:
    nxt = np.roll(pts, -1, axis=1)
    perimeter = np.linalg.norm(nxt - pts, axis=2).sum(axis=1)
    area = 0.5 * np.abs(
        (pts[:, :, 0] * nxt[:, :, 1] - nxt[:, :, 0] * pts[:, :, 1]).sum(axis=1)
    )
    return 4.0 * np.pi * area / np.maximum(perimeter * perimeter, 1.0e-12)


def test_voronoi_is_registered():
    assert "voronoi" in reg.available()
    spec = reg.get("voronoi")
    assert spec.name == "voronoi"
    assert callable(spec.alloc_scratch) and callable(spec.generate)


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
    center = to_t(track.center).view(E, n_max, 2)
    valid = to_t(track.valid).bool()
    count = to_t(track.count)

    assert valid.shape == (E,)
    assert count.shape == (E,)
    assert torch.all(count >= 1) and torch.all(count <= n_max)
    assert float(valid.float().mean()) > 0.50

    for e in range(E):
        c = int(count[e])
        finite = torch.isfinite(center[e]).all(dim=-1)
        assert torch.all(finite[:c])
        assert not torch.any(finite[c:])


@pytest.mark.parametrize("dev", DEVS)
def test_voronoi_generation_outputs_simple_finite_centerlines(dev):
    E = 128
    cfg = TrackGenConfig(
        generator="voronoi",
        num_envs=E,
        num_points=128,
        device=dev,
    )
    spec = reg.get("voronoi")
    scratch = spec.alloc_scratch(cfg)
    seeds_t = torch.arange(E, dtype=torch.int32, device=dev)
    seeds_wp = wp.from_torch(seeds_t, dtype=wp.int32)
    out = wp.empty(E * int(cfg.num_points), dtype=wp.vec2f, device=dev)
    valid_wp = wp.empty(E, dtype=wp.int32, device=dev)

    spec.generate(seeds_wp, cfg, out, valid_wp, scratch)
    if "cuda" in dev:
        wp.synchronize()

    centerline = to_t(out).view(E, int(cfg.num_points), 2)
    valid = to_t(valid_wp).bool()

    assert valid.all()
    assert torch.isfinite(centerline).all()
    crossing_free = (self_intersections(centerline) == 0).float().mean().item()
    assert crossing_free >= 0.95, f"voronoi crossing-free {crossing_free} < 0.95 on {dev}"


def test_voronoi_deterministic_and_cell_count_changes_tracks_cpu():
    E = 32
    base = dict(generator="voronoi", num_envs=E, num_points=96, N_max=192, device="cpu")
    cfg_a = TrackGenConfig(**base, voronoi_num_sites=64)
    cfg_b = TrackGenConfig(**base, voronoi_num_sites=256)

    gen_a1 = TrackGenerator(cfg_a, _make_rng(E, seed=7, device="cpu"))
    gen_a2 = TrackGenerator(cfg_a, _make_rng(E, seed=7, device="cpu"))
    gen_b = TrackGenerator(cfg_b, _make_rng(E, seed=7, device="cpu"))

    a1 = to_t(gen_a1.generate(E).center).clone()
    a2 = to_t(gen_a2.generate(E).center).clone()
    b = to_t(gen_b.generate(E).center).clone()

    fin1 = torch.isfinite(a1).all(dim=-1)
    fin2 = torch.isfinite(a2).all(dim=-1)
    assert torch.equal(fin1, fin2)
    assert torch.allclose(a1[fin1], a2[fin2], atol=1e-5)

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
