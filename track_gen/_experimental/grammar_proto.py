"""Host (numpy) prototype of the segment-grammar generator (#6) — DEV ONLY.

Not imported by the runtime. Tunes the grammar defaults and validates closure + character
before the Warp port. Two kappa-primitives (constant + linear-ramp) + named patterns.

CLOSURE (corrected after the first prototype blobbed): heading is closed by SCALING the
curvature (kappa *= 2*pi / net_turn), which preserves kappa==0 spans so STRAIGHTS stay
straight — the earlier additive DC-shift (kappa += 2*pi) lifted straights onto a constant
arc and produced round blobs. Displacement is closed by the gap-distribution pass (subtract
the mean edge so edge vectors sum to zero). The grammar samples a NET-POSITIVE winding
(corners biased one way so net turn ~ one loop) with occasional reversals (chicanes/S-bends)
and explicit straight spans; scaling then normalizes the net turn to exactly 2*pi.
"""
from __future__ import annotations
import numpy as np

# Tuned in the __main__ driver; Task 2 copies these into TrackGenConfig defaults.
DEFAULTS = dict(
    num_points=256,
    grammar_segments=18,             # S: alternating straight+corner segments (S//2 corners)
    grammar_straight_frac=0.45,      # target fraction of arc-length that is straight (kappa=0)
    grammar_curvature_budget=1.3,    # max per-corner turn angle (rad); sets hairpin tightness
    grammar_chicane_bias=0.22,       # fraction of corners that REVERSE sign (chicanes/S-bends)
    grammar_hairpin_max_frac=0.10,   # max arc-length fraction of any single corner span
    scale=1.0,
)
_BEZIER_EXTENT = 1.44               # match warp_generate_polar._BEZIER_EXTENT
# Cap the heading-closure scale factor. When reverses nearly cancel the net winding, the raw
# 2*pi/net factor explodes and amplifies the curve into a tight self-crossing knot; clamping it
# bounds that amplification (the small residual heading mismatch at the seam is XPBD's to repair).
_HEADING_SCALE_CAP = 2.0


def sample_segments(rng: np.random.Generator, S: int, cfg: dict) -> np.ndarray:
    """Return [S, 3] rows (kappa_start, kappa_end, length_frac), pre-closure.

    Net-winding grammar: alternate a straight (kappa=0) with a corner. Corners are biased
    one direction (net winding) with varied turn angle (gentle sweeper .. tight hairpin);
    EXACTLY round(n_corner * chicane_bias) of them reverse sign (chicanes). Fixing the reverse
    COUNT (vs an independent per-corner coin flip) keeps the net winding reliably away from
    zero, so the heading-closure scale factor stays bounded and rarely hits _HEADING_SCALE_CAP
    — the coin-flip's heavy tail produced net~0 seeds that the cap then folded into
    self-crossing knots. Straight spans are longer on average so real straights dominate.
    """
    straight_frac = float(cfg["grammar_straight_frac"])
    sharp = float(cfg["grammar_curvature_budget"])       # max per-corner turn angle (rad)
    hairpin_max = float(cfg["grammar_hairpin_max_frac"])
    reverse_frac = float(cfg["grammar_chicane_bias"])
    n_corner = max(2, S // 2)
    n_neg = int(round(n_corner * reverse_frac))          # exact reverse count -> bounded net winding
    neg_idx = set(rng.choice(n_corner, size=n_neg, replace=False).tolist()) if n_neg else set()
    segs = []
    for ci in range(n_corner):
        # straight (kappa = 0) — long on average so straights are visible
        segs.append((0.0, 0.0, rng.uniform(0.06, 0.22)))
        # corner — varied turn angle, biased + (winding); chosen indices reverse = chicane
        ang = rng.uniform(0.25, sharp)
        ln = rng.uniform(0.02, hairpin_max)
        sgn = -1.0 if ci in neg_idx else 1.0
        k = sgn * ang / max(ln, 1e-6)                    # kappa = turn / span -> visible feature
        if rng.random() < 0.4:                           # 40% linear-ramp (clothoid/spiral)
            segs.append((0.0, k, ln))
        else:                                            # constant-kappa arc
            segs.append((k, k, ln))
    segs = np.array(segs[:S] if len(segs) >= S else segs, dtype=np.float64)
    # Bias the straight/corner length split toward the target straight fraction.
    is_straight = (segs[:, 0] == 0.0) & (segs[:, 1] == 0.0)
    if is_straight.any() and (~is_straight).any():
        segs[is_straight, 2] *= straight_frac / segs[is_straight, 2].sum()
        segs[~is_straight, 2] *= (1.0 - straight_frac) / segs[~is_straight, 2].sum()
    segs[:, 2] /= segs[:, 2].sum()
    return segs


def rasterize_kappa(segments: np.ndarray, N: int) -> np.ndarray:
    """Per-sample curvature from the segment sequence; linear-interp within each span."""
    bounds = np.concatenate([[0.0], np.cumsum(segments[:, 2])])
    s = (np.arange(N) + 0.5) / N
    seg_idx = np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, len(segments) - 1)
    kappa = np.empty(N)
    for i in range(N):
        j = seg_idx[i]
        lo, hi = bounds[j], bounds[j + 1]
        u = 0.0 if hi <= lo else (s[i] - lo) / (hi - lo)
        kappa[i] = (1 - u) * segments[j, 0] + u * segments[j, 1]
    return kappa


def close_and_integrate(kappa: np.ndarray) -> np.ndarray:
    """Heading closure by SCALING (preserves kappa=0 straights) + integrate + gap-distribution
    displacement closure (subtract the mean edge so edge vectors sum to zero)."""
    N = kappa.shape[0]
    ds = 1.0 / N
    net = float((kappa * ds).sum())
    if abs(net) > 1e-6:
        sc = 2.0 * np.pi / net                       # net turn -> ~2*pi; zeros stay zero
        sc = max(-_HEADING_SCALE_CAP, min(_HEADING_SCALE_CAP, sc))
        kappa = kappa * sc
    theta = np.cumsum(kappa) * ds
    theta = theta - theta[0]
    edges = ds * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    edges = edges - edges.mean(axis=0)               # gap-distribution: edges sum to 0 (closed)
    return np.cumsum(edges, axis=0)


def normalize(pts: np.ndarray, target_extent: float) -> np.ndarray:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = float(np.max(hi - lo))
    s = target_extent / max(extent, 1e-8)
    return (pts - center) * s


def generate_centerline(seed: int, cfg: dict) -> np.ndarray:
    rng = np.random.default_rng(seed)
    segs = sample_segments(rng, int(cfg["grammar_segments"]), cfg)
    kappa = rasterize_kappa(segs, int(cfg["num_points"]))
    pts = close_and_integrate(kappa)
    return normalize(pts, float(cfg["scale"]) * _BEZIER_EXTENT)


if __name__ == "__main__":
    import os, sys
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmarks.track_metrics import compactness, chicane_count, straight_fraction, self_intersects

    cfg = DEFAULTS.copy()
    comp, chic, strt, sint = [], [], [], []
    for seed in range(500):
        p = generate_centerline(seed, cfg)
        comp.append(compactness(p)); chic.append(chicane_count(p)); strt.append(straight_fraction(p))
        if seed < 150:
            sint.append(self_intersects(p))
    comp = np.array(comp)
    print("=== FIXED grammar proto (scaling closure + net winding), 500 seeds ===")
    print(f"compactness  p10/p50/p90 = {np.percentile(comp,10):.3f}/{np.percentile(comp,50):.3f}/{np.percentile(comp,90):.3f}")
    print(f"chicane_count mean = {np.mean(chic):.2f}   straight_fraction mean = {np.mean(strt):.3f}")
    print(f"pre-relax self-intersection rate (150 seeds) = {np.mean(sint):.3f}")

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        p = generate_centerline(i, cfg); cp = np.vstack([p, p[0]])
        ax.plot(cp[:, 0], cp[:, 1], "b-", lw=0.9); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"s={i} c={compactness(p):.2f} st={straight_fraction(p):.2f}", fontsize=7)
    plt.suptitle("Grammar Proto FIXED — scaling closure + net winding (S=16)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "viz", "out", "grammar_proto_grid.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"saved {out}")
