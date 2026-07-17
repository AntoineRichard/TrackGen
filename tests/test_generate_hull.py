"""End-to-end + determinism tests for the ``"hull"`` first-stage generator.

The hull generator (convex-hull stand-in via angle-sort + one midpoint-displacement layer
+ closed Catmull-Rom smoothing) is selected by ``config.generator="hull"`` and runs through
the same pure-Warp pipeline as bezier. These tests assert the contract the pipeline relies
on: finite real points, N-point closed loops, a reasonable valid yield, and that the same
seeds reproduce the same tracks within a device.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("warp")

import warp as wp  # noqa: E402
wp.init()

from track_gen import PerEnvSeededRNG  # noqa: E402
from track_gen._src.track_generator import Track, TrackGenConfig, TrackGenerator  # noqa: E402
from track_gen._src import generator_registry as reg  # noqa: E402
from tests._warp_compare import self_intersections, to_t  # noqa: E402

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_rng(num_envs, seed=0, device="cpu"):
    return PerEnvSeededRNG(seeds=seed, num_envs=num_envs, device=device)


def test_hull_is_registered():
    assert "hull" in reg.available()
    spec = reg.get("hull")
    assert spec.name == "hull"
    assert callable(spec.alloc_scratch) and callable(spec.generate)


@pytest.mark.parametrize("dev", DEVS)
def test_hull_end_to_end(dev):
    """generate(generator='hull') -> finite real points, N_max-padded closed loops, and a
    high valid yield under the default hull displacement setting."""
    E, N_max = 64, 256
    cfg = TrackGenConfig(
        generator="hull", num_envs=E, num_points=128, N_max=N_max,
        half_width=0.1, device=dev,
    )
    rng = _make_rng(E, device=dev)
    gen = TrackGenerator(cfg, rng)

    track = gen.generate(E)
    assert isinstance(track, Track)

    center = to_t(track.center).view(E, N_max, 3)[..., :2]
    outer = to_t(track.outer).view(E, N_max, 3)[..., :2]
    inner = to_t(track.inner).view(E, N_max, 3)[..., :2]
    valid = to_t(track.valid).bool()
    count = to_t(track.count)

    assert valid.shape == (E,)
    assert count.shape == (E,)
    assert torch.all(count >= 1) and torch.all(count <= N_max)

    yield_frac = float(valid.float().mean())
    assert yield_frac >= 0.98, f"hull valid yield too low: {yield_frac}"

    # Per env: real points are exactly the first count[e] rows (finite), the rest NaN-padded,
    # and the three boundary arrays share the same finite mask (index-aligned closed loop).
    for e in range(E):
        c = int(count[e])
        finite_center = torch.isfinite(center[e]).all(dim=-1)
        finite_outer = torch.isfinite(outer[e]).all(dim=-1)
        finite_inner = torch.isfinite(inner[e]).all(dim=-1)
        assert torch.equal(finite_center, finite_outer)
        assert torch.equal(finite_center, finite_inner)
        assert torch.all(finite_center[:c])
        assert not torch.any(finite_center[c:])


@pytest.mark.parametrize("dev", DEVS)
def test_hull_generation_polygon_fallback_removes_self_crossers(dev):
    """The hull first-stage generator should mirror bezier's fallback contract: every env is
    gen-valid, finite, and nearly always simple before relaxation/inflation."""
    E = 256
    cfg = TrackGenConfig(generator="hull", num_envs=E, num_points=128, device=dev)
    spec = reg.get("hull")
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
    assert crossing_free >= 0.99, f"hull crossing-free {crossing_free} < 0.99 on {dev}"


@pytest.mark.parametrize("dev", DEVS)
def test_hull_deterministic(dev):
    """Same seeds + config -> identical tracks within a device (deterministic Warp RNG)."""
    E = 32
    cfg = TrackGenConfig(generator="hull", num_envs=E, num_points=96, N_max=192, device=dev)

    g1 = TrackGenerator(cfg, _make_rng(E, seed=7, device=dev))
    c1 = to_t(g1.generate(E).center).clone()
    v1 = to_t(g1.generate(E).valid).clone()  # second call is a no-op re-run; same buffers

    g2 = TrackGenerator(cfg, _make_rng(E, seed=7, device=dev))
    t2 = g2.generate(E)
    c2 = to_t(t2.center)
    v2 = to_t(t2.valid)

    # NaN-padded loops: compare via finite mask equality + allclose on the finite region.
    fin1 = torch.isfinite(c1).all(dim=-1)
    fin2 = torch.isfinite(c2).all(dim=-1)
    assert torch.equal(fin1, fin2)
    assert torch.allclose(c1[fin1], c2[fin2], atol=1e-5)
    assert torch.equal(v1, v2)


def test_hull_differs_from_bezier_cpu():
    """The midpoint-displacement layer makes hull geometrically distinct from bezier (it did
    not collapse toward the plain angle-sort / bezier shape)."""
    E = 32
    base = dict(num_envs=E, num_points=128, N_max=256, device="cpu")
    gb = TrackGenerator(TrackGenConfig(generator="bezier", **base), _make_rng(E, device="cpu"))
    gh = TrackGenerator(TrackGenConfig(generator="hull", **base), _make_rng(E, device="cpu"))
    cb = to_t(gb.generate(E).center).view(E, 256, 3)[..., :2]
    ch = to_t(gh.generate(E).center).view(E, 256, 3)[..., :2]
    # Same seeds, same coordinate scale, but the shapes must differ substantially.
    fin = torch.isfinite(cb).all(dim=-1) & torch.isfinite(ch).all(dim=-1)
    assert fin.any()
    assert not torch.allclose(cb[fin], ch[fin], atol=1e-2)
