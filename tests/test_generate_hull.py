"""End-to-end tests for the ``"hull"`` first-stage generator.

The hull generator (convex-hull stand-in via angle-sort + one midpoint-displacement layer
+ closed Catmull-Rom smoothing) is selected by ``config.generator="hull"`` and runs through
the same pure-Warp pipeline as bezier. These tests assert hull's own properties: a high
valid yield, a crossing-free generated loop, and shapes distinct from bezier's. The
generator contract (finite real points, NaN padding, count range, closed loops,
determinism within a device, the hot path) is checked for every generator by the
conformance kit, in tests/test_conformance.py.
"""
from __future__ import annotations
from tests._reference_generators import generate_raw

import pytest
import torch

pytest.importorskip("warp")

import warp as wp  # noqa: E402
wp.init()

from track_gen import PerEnvSeededRNG  # noqa: E402
from track_gen._src.track_generator import Track, TrackGenConfig, TrackGenerator  # noqa: E402
from track_gen._src import warp_generate_hull  # noqa: E402
from tests._warp_compare import self_intersections, to_t  # noqa: E402

from tests.conftest import DEVS  # noqa: E402


def _make_rng(num_envs, seed=0, device="cpu"):
    return PerEnvSeededRNG(seeds=seed, num_envs=num_envs, device=device)


@pytest.mark.parametrize("dev", DEVS)
def test_hull_end_to_end(dev):
    """generate(generator='hull') -> a high valid yield under the default hull
    displacement setting."""
    E, N_max = 64, 256
    cfg = TrackGenConfig(
        generator="hull", num_envs=E, num_points=128, N_max=N_max,
        half_width=0.1, device=dev,
    )
    rng = _make_rng(E, device=dev)
    gen = TrackGenerator(cfg, rng)

    track = gen.generate(E)
    assert isinstance(track, Track)
    yield_frac = float(to_t(track.valid).bool().float().mean())
    assert yield_frac >= 0.98, f"hull valid yield too low: {yield_frac}"


@pytest.mark.parametrize("dev", DEVS)
def test_hull_generation_polygon_fallback_removes_self_crossers(dev):
    """The hull first-stage generator should mirror bezier's fallback contract: every env is
    gen-valid, finite, and nearly always simple before relaxation/inflation."""
    E = 256
    cfg = TrackGenConfig(generator="hull", num_envs=E, num_points=128, device=dev)
    # The frozen legacy entry points (the registered "hull" is a Stage since PR13c).
    scratch = warp_generate_hull._alloc_hull_scratch(cfg)

    seeds_t = torch.arange(E, dtype=torch.int32, device=dev)
    seeds_wp = wp.from_torch(seeds_t, dtype=wp.int32)
    out = wp.empty(E * int(cfg.num_points), dtype=wp.vec2f, device=dev)
    valid_wp = wp.empty(E, dtype=wp.int32, device=dev)

    generate_raw("hull", seeds_wp, cfg, out, valid_wp, scratch)
    if "cuda" in dev:
        wp.synchronize()

    centerline = to_t(out).view(E, int(cfg.num_points), 2)
    valid = to_t(valid_wp).bool()

    assert valid.all()
    assert torch.isfinite(centerline).all()
    crossing_free = (self_intersections(centerline) == 0).float().mean().item()
    assert crossing_free >= 0.99, f"hull crossing-free {crossing_free} < 0.99 on {dev}"


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
