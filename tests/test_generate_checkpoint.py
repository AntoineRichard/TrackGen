"""End-to-end yield + shape-variety tests for the ``"checkpoint"`` generator.

The checkpoint generator (method #5, CarRacing-style steering) is accepted by:
  1. e2e — the pipeline keeps a reasonable valid yield.
  2. non-degenerate / shape-variety — median compactness < 0.65 (checkpoint sits ~0.58).
     NOTE: test_shape_variety.py's test_no_registered_generator_is_degenerate already covers
     every registered generator including "checkpoint" (threshold 0.85). We add a tighter local
     assertion (~0.65) matching checkpoint's actual organic distribution.
     Checkpoint has LOW straight_fraction (no straights by design) — its value is a qualitative
     flowing shape distribution that renders show; a single-metric win vs. bezier/hull/polar is
     NOT asserted here, and should NOT be: that would be a fabricated metric assertion.
  3. clip-fallback rescues single-crossers — at K=1 (no best-of-K), enabling
     checkpoint_clip_fallback=True yields >= the False case's valid fraction.

The generator contract (finite centerlines with the right stride, count range,
determinism, diversity, batch independence, the hot path) is checked for every generator
by the conformance kit, in tests/test_conformance.py.
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


def test_checkpoint_e2e_reasonable_yield():
    """After the full pipeline (generate + inflate), a reasonable fraction is valid (the
    count range is the conformance kit's output.count check)."""
    cfg, gen, track = _run()
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    # Reasonable yield: the checkpoint generator is always gen-valid; the pipeline's
    # inflate / relaxation gate sets the final valid fraction.
    yield_frac = float(valid.mean())
    assert yield_frac > 0.3, f"checkpoint valid yield unexpectedly low: {yield_frac:.3f}"


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
