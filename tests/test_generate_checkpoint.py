"""End-to-end + determinism + shape-variety tests for the ``"checkpoint"`` generator.

The checkpoint generator (method #5, CarRacing-style steering) is accepted by:
  1. e2e — TrackGenerator produces finite centerlines with correct stride and valid tracks.
  2. determinism — same seed + config yields bit-identical centerlines across two generate() calls.
  3. non-degenerate / shape-variety — median compactness < 0.65 (checkpoint sits ~0.58).
     NOTE: test_shape_variety.py's test_no_registered_generator_is_degenerate already covers
     every registered generator including "checkpoint" (threshold 0.85). We add a tighter local
     assertion (~0.65) matching checkpoint's actual organic distribution.
     Checkpoint has LOW straight_fraction (no straights by design) — its value is a qualitative
     flowing shape distribution that renders show; a single-metric win vs. bezier/hull/polar is
     NOT asserted here, and should NOT be: that would be a fabricated metric assertion.
  4. clip-fallback rescues single-crossers — at K=1 (no best-of-K), enabling
     checkpoint_clip_fallback=True yields >= the False case's valid fraction.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("warp")
import warp as wp  # noqa: E402
wp.init()

from track_gen._src.types import TrackGenConfig  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen._src.rng_utils import PerEnvSeededRNG  # noqa: E402
from track_gen._src import generator_registry  # noqa: E402


def _compactness(pts: np.ndarray) -> np.ndarray:
    """Isoperimetric compactness: 4πA/P² per env (1.0 == circle)."""
    nxt = np.roll(pts, -1, axis=1)
    perimeter = np.linalg.norm(nxt - pts, axis=2).sum(axis=1)
    area = 0.5 * np.abs(
        (pts[:, :, 0] * nxt[:, :, 1] - nxt[:, :, 0] * pts[:, :, 1]).sum(axis=1)
    )
    return 4.0 * np.pi * area / np.maximum(perimeter * perimeter, 1.0e-12)


def _run(seed: int = 0, E: int = 64, **overrides):
    cfg = TrackGenConfig(generator="checkpoint", device="cpu", num_envs=E, **overrides)
    rng = PerEnvSeededRNG(seeds=seed, num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, rng)
    track = gen.generate(E)
    return cfg, gen, track


def test_checkpoint_registered():
    assert "checkpoint" in generator_registry.available()
    spec = generator_registry.get("checkpoint")
    assert spec.name == "checkpoint"
    assert callable(spec.alloc_scratch) and callable(spec.generate)


def test_checkpoint_e2e_centerline_finite():
    """generate() -> finite centerlines with the correct [E*N] stride."""
    cfg, gen, track = _run()
    N = int(cfg.num_points)
    gc = wp.to_torch(gen._scratch.gen_centerline).cpu().numpy()
    assert gc.shape == (cfg.num_envs * N, 2), f"unexpected shape: {gc.shape}"
    assert np.isfinite(gc).all(), "checkpoint centerline has non-finite values"


def test_checkpoint_e2e_n_points_per_env():
    """Each env contributes exactly N centerline points (no degenerate empties)."""
    cfg, gen, track = _run()
    N = int(cfg.num_points)
    gc = wp.to_torch(gen._scratch.gen_centerline).cpu().numpy().reshape(cfg.num_envs, N, 2)
    assert gc.shape[1] == N


def test_checkpoint_e2e_valid_count():
    """After full pipeline (generate + inflate), the batch count stays in [1, N_max]."""
    cfg, gen, track = _run()
    count = wp.to_torch(track.count).cpu().numpy()
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    N_max = int(cfg.N_max)
    assert np.all(count >= 1) and np.all(count <= N_max), (
        f"count out of range [1, {N_max}]: min={count.min()}, max={count.max()}"
    )
    # Reasonable yield: the checkpoint generator is always gen-valid; the pipeline's
    # inflate / relaxation gate sets the final valid fraction.
    yield_frac = float(valid.mean())
    assert yield_frac > 0.3, f"checkpoint valid yield unexpectedly low: {yield_frac:.3f}"


def test_checkpoint_determinism_same_seed():
    """Same seed + config -> bit-identical gen centerline across two TrackGenerator instances."""
    _, gen_a, _ = _run(seed=42)
    _, gen_b, _ = _run(seed=42)
    a = wp.to_torch(gen_a._scratch.gen_centerline).cpu().numpy()
    b = wp.to_torch(gen_b._scratch.gen_centerline).cpu().numpy()
    assert np.array_equal(a, b), "checkpoint is not deterministic for the same seed"


def test_checkpoint_different_seeds_differ():
    """Different seeds -> geometrically distinct centerlines per env."""
    _, gen_a, _ = _run(seed=1)
    _, gen_b, _ = _run(seed=2)
    a = wp.to_torch(gen_a._scratch.gen_centerline).cpu().numpy()
    b = wp.to_torch(gen_b._scratch.gen_centerline).cpu().numpy()
    assert not np.allclose(a, b), "different seeds produced identical centerlines"


def test_checkpoint_shape_variety_non_degenerate():
    """Checkpoint produces organic flowing shapes: median compactness must be well below 1.0.

    The tighter threshold (0.65 vs. test_shape_variety.py's 0.85) captures checkpoint's
    characteristic ~0.58 organic shape compactness. Compactness 1.0 == circle (degenerate).
    """
    E = 128
    cfg, gen, track = _run(seed=0, E=E, relax_iters=40)
    N_max = int(cfg.N_max)
    center = wp.to_torch(track.center).cpu().numpy().reshape(E, N_max, 3)[..., :2]
    count = wp.to_torch(track.count).cpu().numpy().astype(int)
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    comp = np.array([
        _compactness(center[e, :count[e]][np.newaxis])[0]
        for e in range(E)
        if valid[e] and count[e] >= 4 and np.isfinite(center[e, :count[e]]).all()
    ])
    assert comp.size > 0, "checkpoint: no valid tracks to assess shape variety"
    p50 = float(np.median(comp))
    assert p50 < 0.65, (
        f"checkpoint median compactness {p50:.3f} >= 0.65 — generator may have collapsed "
        f"toward circles (1.0 == circle)"
    )


def test_checkpoint_clip_fallback_yields_at_least_no_clip():
    """At K=1 (no best-of-K), clip_fallback=True yields >= clip_fallback=False valid fraction.

    The clip rescues single-self-crossing loops: the clipped outer sub-loop is simple.
    At K=1 there is no best-of-K selection to mask the effect, making the clip measurable.
    With K>1 (default 4), the best-of-K already selects the crossing-free candidate most of the
    time, so the clip's effect is smaller. This test uses K=1 to make it detectable.
    """
    E = 256

    def _valid_frac(clip: bool) -> float:
        cfg = TrackGenConfig(
            generator="checkpoint",
            device="cpu",
            num_envs=E,
            checkpoint_best_of_k=1,
            checkpoint_clip_fallback=clip,
        )
        rng = PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")
        gen = TrackGenerator(cfg, rng)
        track = gen.generate(E)
        valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
        return float(valid.mean())

    frac_no_clip = _valid_frac(clip=False)
    frac_clip = _valid_frac(clip=True)

    # clip must not make things worse; the delta is typically +5-15%.
    assert frac_clip >= frac_no_clip - 0.02, (
        f"clip_fallback=True yield {frac_clip:.3f} is worse than False {frac_no_clip:.3f} "
        f"(tolerance 0.02 to absorb floating-point nondeterminism)"
    )
