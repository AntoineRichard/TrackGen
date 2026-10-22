"""End-to-end tests for the periodic polar-spline generator (``config.generator="polar"``).

Drives the full pure-Warp pipeline through the public ``TrackGenerator`` facade and
asserts the polar-specific properties: the loops are centred and sized like bezier's,
are not round blobs, and yield a reasonable valid fraction. The generator contract
(raw shape and finiteness, closed loop, NaN padding, determinism, diversity, batch
independence, the hot path) is checked by the conformance kit for every generator,
in tests/test_conformance.py.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("warp")
import warp as wp  # noqa: E402

from track_gen._src.types import TrackGenConfig  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen._src.rng_utils import PerEnvSeededRNG  # noqa: E402


def _compactness(pts: np.ndarray) -> np.ndarray:
    nxt = np.roll(pts, -1, axis=1)
    perimeter = np.linalg.norm(nxt - pts, axis=2).sum(axis=1)
    area = 0.5 * np.abs(
        (pts[:, :, 0] * nxt[:, :, 1] - nxt[:, :, 0] * pts[:, :, 1]).sum(axis=1)
    )
    return 4.0 * np.pi * area / np.maximum(perimeter * perimeter, 1.0e-12)


def _run(seed=0, E=64, **overrides):
    cfg = TrackGenConfig(generator="polar", device="cpu", num_envs=E, **overrides)
    rng = PerEnvSeededRNG(seeds=int(seed), num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, rng)
    track = gen.generate(E)
    return cfg, gen, track


def test_polar_e2e_centered_and_sized():
    # The loops are centred near the origin and rescaled to the bezier extent so
    # downstream half_width / spacing / relax see the same coordinate range.
    cfg, gen, track = _run()
    N = int(cfg.num_points)
    gc = wp.to_torch(gen._scratch.gen_centerline).cpu().numpy().reshape(cfg.num_envs, N, 2)
    bbox = gc.max(axis=1) - gc.min(axis=1)
    longest = bbox.max(axis=1)
    # Each env's longest bbox dim normalised to scale * ~1.44.
    assert np.allclose(longest, longest[0], atol=1e-3)
    assert 1.0 < float(longest.mean()) < 2.0
    center = gc.mean(axis=1)
    assert np.abs(center).max() < 0.3  # near origin (mean over a near-symmetric loop)


def test_polar_default_shapes_are_not_mostly_round_blobs():
    # The SOTA/pre-relax docs call out the failure mode for polar functions: high-yield
    # nearly-circular blobs. The default polar generator should exercise the intended
    # random-radial-knot design space instead of clustering near compactness == 1.
    cfg, gen, track = _run(seed=0, E=64)
    N = int(cfg.num_points)
    gc = wp.to_torch(gen._scratch.gen_centerline).cpu().numpy().reshape(cfg.num_envs, N, 2)
    c = _compactness(gc)
    assert float(c.mean()) < 0.90
    assert float((c > 0.90).mean()) < 0.25


def test_polar_e2e_reasonable_yield():
    cfg, gen, track = _run()
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    yield_frac = float(valid.mean())
    # The polar loop is simple by construction, so yield should be high; we do NOT
    # hard-fail on a tight threshold, only on a clearly-broken (near-zero) yield.
    assert yield_frac > 0.5, f"polar yield unexpectedly low: {yield_frac}"
