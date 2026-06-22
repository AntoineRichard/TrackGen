"""Host (numpy) reference for the checkpoint-steering generator (#5) — DEV ONLY.

Not imported by the runtime. A PROTOTYPE of the Gymnasium ``CarRacing`` track family,
ADAPTED so it is portable to a fixed-shape NVIDIA-Warp kernel later (no host retry loop,
no variable-length loop-trim). Mirrors ``grammar_proto.py`` in structure and validation
methodology so it can be compared head-to-head with the segment-grammar generator (#6).

WHAT CARRACING DOES (and why we can't copy it verbatim)
-------------------------------------------------------
Gymnasium's ``CarRacing._create_track`` samples ~12 checkpoints on radii ``U(R/3, R)`` at
angles ``2*pi*c/C + noise``, then steers a bounded-turn-rate path that chases each checkpoint
in turn. It over-generates ~4 laps' worth of path, scans for where the path re-crosses the
start, TRIMS to that sub-loop, and (in a ``while True``) REGENERATES from scratch if the loop
didn't close or wasn't "well glued". Both data-dependent steps — the unbounded retry and the
variable-length trim — are NOT CUDA-graph-capturable, so we replace them exactly like the
grammar port did:

  * FIXED-N steering, ONE lap.  Sample C checkpoints. Walk a fixed N-step path with a bounded
    per-step turn rate; at each step steer the heading toward the CURRENT target checkpoint
    (proportional steering, turn clamped to +/- turn_rate), and advance the angular target as
    we pass each checkpoint. The angular budget is sized to complete ~one lap over N steps.
    NO over-generate-and-trim.

  * EXPLICIT CLOSURE by construction (no trim, no reject).  THREE variants implemented; the
    task asked to try the two book variants and keep what renders best — a third (the winner)
    emerged between them:
      (a) "pos"     — close in POSITION space: subtract the linear endpoint-residual ramp (the
                      open-curve gap spread uniformly over the path) via the gap-distribution
                      edge step. Preserves the literal steered shape, BUT the steered head-arc
                      and return tail-arc physically overlap near the start -> ~100% self-cross
                      (best-of-K can't rescue an all-bad pool). Kept for comparison only.
      (b) "kappa"   — extract the steered heading, close it like grammar's T1: MULTIPLICATIVELY
                      rescale every turn so the net turn is exactly 2*pi (turning number 1),
                      then gap-distribution the displacement. Guarantees no inner loops, but the
                      uniform rescale washes the shape toward a smooth convex blob (compactness
                      ~0.88, indistinguishable from polar/hull) -> loses the CarRacing flow.
      (c) "heading_ramp" (DEFAULT, the winner) — close the heading to turning-number-1 ADDITIVELY:
                      add a CONSTANT drift (2*pi - net_turn)/N to every step so the net turn is
                      exactly 2*pi WITHOUT rescaling the local curvature variation, then
                      gap-distribution the displacement. Like (b) it guarantees turning number 1
                      (no inner loops -> best-of-K starts from a low crossing rate it can drive to
                      0), but like (a) it preserves the steered sweep/undulation (compactness
                      ~0.71, clearly NOT a blob). This is grammar's T1 displacement closure with
                      an additive (not multiplicative) heading correction.

  * BEST-OF-K selection (replaces the ``while True`` reject-retry).  Generate K decorrelated
    candidates per seed and KEEP the one with the fewest self-intersections (deterministic
    argmin, ties -> lowest index). Bounded and capturable — the same trick that took grammar's
    self-intersection rate from ~33% to ~5%.

Everything is fixed-shape and free of data-dependent host control flow (fixed C, N, K; bounded
steering; the only "branches" are capture-time constants), so this maps cleanly to a future
``warp_generate_checkpoint.py``.
"""
from __future__ import annotations
import numpy as np

# Tuned via the __main__ driver (renders are the arbiter). A future TrackGenConfig would mirror
# these as checkpoint_* defaults.
DEFAULTS = dict(
    num_points=256,            # N: fixed path length (one lap)
    checkpoint_count=12,       # C: radial checkpoints (CarRacing's canonical 12). More -> wavier
                               # (chicane_count scales ~ C); fewer -> calmer/rounder.
    radius_min_frac=0.33,      # inner radius fraction: checkpoint radius ~ U(rmin*R, R). This is
                               # CarRacing's R/3 exactly — the radial drama that gives the inlets.
    angle_jitter=0.55,         # +/- fraction of the per-checkpoint angular slot for angle noise
                               # (slot = 2*pi/C). <1 keeps checkpoints monotone in angle.
    turn_rate=0.42,            # max heading change per step (rad). Bounds curvature -> flow.
    steer_gain=0.65,           # proportional steering gain toward the target bearing (0..1].
    lookahead_frac=0.16,       # advance to the next checkpoint when within this*R of the target
    closure="heading_ramp",    # "heading_ramp" (winner) | "kappa" (T1 rescale) | "pos" (position)
    candidates=8,              # K for the driver's best-of-K block (driver overrides per table)
    scale=1.0,
)
_BEZIER_EXTENT = 1.44          # match warp_generate_polar._BEZIER_EXTENT / grammar_proto

# ---------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------

def sample_checkpoints(rng: np.random.Generator, C: int, cfg: dict) -> np.ndarray:
    """Return [C, 2] checkpoint positions on randomized radii, monotone in angle (one lap).

    Angle of checkpoint c: ``2*pi*c/C + jitter`` with ``jitter`` in +/-``angle_jitter`` of the
    angular slot (kept < one slot so the sequence stays angle-monotone -> the steered path winds
    once). Radius ~ ``U(radius_min_frac*R, R)`` with R=1 (normalised later). This is CarRacing's
    checkpoint distribution, just with the jitter bounded so a single fixed-N lap can track it.
    """
    R = 1.0
    rmin = float(cfg["radius_min_frac"]) * R
    slot = 2.0 * np.pi / C
    jitter = (rng.random(C) * 2.0 - 1.0) * float(cfg["angle_jitter"]) * slot
    ang = np.arange(C) * slot + jitter
    rad = rng.uniform(rmin, R, size=C)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


# ---------------------------------------------------------------------------------------------
# Fixed-N bounded-turn steering through the checkpoints
# ---------------------------------------------------------------------------------------------

def steer_path(checkpoints: np.ndarray, N: int, cfg: dict) -> np.ndarray:
    """Walk a fixed N-step bounded-turn path that chases the checkpoints once around.

    State: position p, heading theta. Each step:
      target_bearing = atan2(target - p);  err = wrap(target_bearing - theta)
      dtheta = clamp(steer_gain * err, +/- turn_rate);  theta += dtheta;  step forward by dl.
    The target advances to the next checkpoint when within ``lookahead_frac*R`` of it.

    The one thing that makes this complete ~exactly one lap is the step length:

      * dl = (checkpoint-ring perimeter) / N.  The ring perimeter is a fixed reduction over the
        checkpoint positions (capturable in Warp), so N steps of length dl cover ~one lap of the
        actual ring rather than a fixed 2*pi guess that over-shoots into a second coil. (With a
        fixed dl the path wound 1.1-1.5 laps and self-crossed; ring-dl pins it to ~1 lap.)

    NOTE: this returns an OPEN path that ends NEAR but not AT the start, and whose first and last
    headings differ. We deliberately do NOT add a "return-to-start" stretch — pinning the tail
    back to checkpoint 0 makes the return arc cross the departure arc near the start (a guaranteed
    cusp). Closure is the closer's job: ``close_heading_ramp`` re-closes the heading to turning
    number 1 and gap-distributes the displacement, which removes the residual cleanly.

    Returns [N, 2] open path positions (NOT yet closed). The target-advance is a per-step compare
    that maps to a Warp int-counter increment (no data-dependent host branch).
    """
    C = len(checkpoints)
    turn_rate = float(cfg["turn_rate"])
    gain = float(cfg["steer_gain"])
    reach = float(cfg["lookahead_frac"])
    # Step length from the actual checkpoint-ring perimeter -> N steps ~= one lap.
    ring = np.vstack([checkpoints, checkpoints[0]])
    perimeter = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
    dl = perimeter / N

    # Start at the first checkpoint, heading toward the second (deterministic, capturable).
    p = checkpoints[0].copy()
    theta = float(np.arctan2(*(checkpoints[1] - checkpoints[0])[::-1]))
    tgt = 1                                   # index of current target checkpoint

    out = np.empty((N, 2), dtype=np.float64)
    for i in range(N):
        out[i] = p
        # Advance target if we are close enough (chase the NEXT checkpoint). Wrap the ring.
        target = checkpoints[tgt % C]
        if np.hypot(*(target - p)) < reach:
            tgt += 1
            target = checkpoints[tgt % C]
        # Proportional bounded steering toward the target bearing.
        bearing = float(np.arctan2(target[1] - p[1], target[0] - p[0]))
        err = (bearing - theta + np.pi) % (2.0 * np.pi) - np.pi
        dtheta = np.clip(gain * err, -turn_rate, turn_rate)
        theta += dtheta
        p = p + dl * np.array([np.cos(theta), np.sin(theta)])
    return out


# ---------------------------------------------------------------------------------------------
# Closure (three variants; "heading_ramp" is the tuned winner)
# ---------------------------------------------------------------------------------------------

def close_positions(path: np.ndarray) -> np.ndarray:
    """Variant (a): close an open path in POSITION space.

    Treat the path as N edges; the open-loop residual is ``path[-1]->path[0]`` would-be edge
    not summing to zero. Subtract the mean edge (gap-distribution) so the edge vectors sum to
    zero, then re-integrate. Equivalent to spreading the linear endpoint residual uniformly —
    the grammar's ``close_and_integrate`` displacement step applied to a position path.
    """
    edges = np.roll(path, -1, axis=0) - path        # edge i = p[i+1]-p[i], wrapping
    edges = edges - edges.mean(axis=0)              # edges now sum to zero -> closed loop
    closed = np.cumsum(edges, axis=0)
    return closed - closed.mean(axis=0)


def close_kappa(path: np.ndarray) -> np.ndarray:
    """Variant (b): extract the steered heading, T1-close it like grammar, re-integrate.

    Heading per edge theta_i = atan2(edge_i). Net turn = sum of wrapped heading increments;
    scale the increments so the net turn is exactly 2*pi (turning number 1), rebuild headings,
    then gap-distribution the unit-step displacement. Reuses grammar's T1 closure directly and
    guarantees no inner loops.
    """
    N = len(path)
    edges = np.roll(path, -1, axis=0) - path
    theta = np.arctan2(edges[:, 1], edges[:, 0])
    dtheta = (np.diff(theta, prepend=theta[0]) + np.pi) % (2.0 * np.pi) - np.pi  # wrapped increments
    net = dtheta.sum()
    if abs(net) > 1e-6:
        dtheta = dtheta * (2.0 * np.pi / net)       # T1: scale net turn to 2*pi
    theta_closed = np.cumsum(dtheta)
    theta_closed = theta_closed - theta_closed[0]
    ds = 1.0 / N
    e = ds * np.stack([np.cos(theta_closed), np.sin(theta_closed)], axis=1)
    e = e - e.mean(axis=0)                          # gap-distribution displacement closure
    closed = np.cumsum(e, axis=0)
    return closed - closed.mean(axis=0)


def close_heading_ramp(path: np.ndarray) -> np.ndarray:
    """Variant (c) — THE DEFAULT. Close the heading to turning-number-1 ADDITIVELY.

    Same skeleton as ``close_kappa`` (grammar's T1 displacement closure), but the heading
    correction is ADDITIVE not MULTIPLICATIVE: instead of scaling every turn by 2*pi/net (which
    flattens local curvature toward a circle), we add a CONSTANT drift ``(2*pi - net)/N`` to each
    step. Net turn becomes exactly 2*pi (turning number 1 -> no inner loops) while the steered
    local curvature variation — the inlets, sweeps and bulges — is preserved. Displacement is
    then gap-distribution closed exactly as in grammar's close_and_integrate.
    """
    N = len(path)
    edges = np.roll(path, -1, axis=0) - path
    theta = np.arctan2(edges[:, 1], edges[:, 0])
    dtheta = (np.diff(theta, prepend=theta[0]) + np.pi) % (2.0 * np.pi) - np.pi  # wrapped increments
    dtheta = dtheta + (2.0 * np.pi - dtheta.sum()) / N   # additive drift -> net turn == 2*pi
    theta_closed = np.cumsum(dtheta)
    theta_closed = theta_closed - theta_closed[0]
    ds = 1.0 / N
    e = ds * np.stack([np.cos(theta_closed), np.sin(theta_closed)], axis=1)
    e = e - e.mean(axis=0)                          # gap-distribution displacement closure
    closed = np.cumsum(e, axis=0)
    return closed - closed.mean(axis=0)


_CLOSERS = {
    "pos": close_positions,
    "kappa": close_kappa,
    "heading_ramp": close_heading_ramp,
}


def normalize(pts: np.ndarray, target_extent: float) -> np.ndarray:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = float(np.max(hi - lo))
    s = target_extent / max(extent, 1e-8)
    return (pts - center) * s


# ---------------------------------------------------------------------------------------------
# Single-candidate generation + best-of-K
# ---------------------------------------------------------------------------------------------

def _resample_closed(pts: np.ndarray, N: int) -> np.ndarray:
    """Arc-length resample a closed loop to N points (keeps spacing even after position-close)."""
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 1e-9:
        return pts
    targets = (np.arange(N) / N) * total
    x = np.interp(targets, s, closed[:, 0])
    y = np.interp(targets, s, closed[:, 1])
    return np.stack([x, y], axis=1)


def generate_candidate(seed: int, cfg: dict) -> np.ndarray:
    """One checkpoint-steered, closed, normalised centerline (single candidate / K=1)."""
    rng = np.random.default_rng(seed)
    C = int(cfg["checkpoint_count"])
    N = int(cfg["num_points"])
    cps = sample_checkpoints(rng, C, cfg)
    path = steer_path(cps, N, cfg)
    closer = _CLOSERS[cfg["closure"]]
    closed = closer(path)
    if cfg["closure"] == "pos":
        closed = _resample_closed(closed, N)        # even spacing for the position variant only;
                                                    # the heading-space closers emit even ds by construction
    return normalize(closed, float(cfg["scale"]) * _BEZIER_EXTENT)


def generate_centerline(seed: int, cfg: dict, K: int | None = None,
                        score=None) -> np.ndarray:
    """Best-of-K: generate K decorrelated candidates, keep the fewest-self-intersection one.

    Decorrelation: candidate k uses seed ``seed*K + k`` so the K draws are independent (same
    trick as grammar's best-of-K). ``score`` returns a sortable badness; default counts
    self-intersections (deterministic argmin, ties -> lowest k). K=1 reduces to a single
    candidate at the base seed (k=0).
    """
    if K is None:
        K = int(cfg["candidates"])
    best, best_score = None, None
    for k in range(K):
        cand = generate_candidate(seed * K + k, cfg)
        sc = score(cand) if score is not None else (1 if _self_intersects(cand) else 0)
        if best_score is None or sc < best_score:
            best, best_score = cand, sc
            if best_score == 0:                     # can't do better than zero crossings
                break
    return best


# ---------------------------------------------------------------------------------------------
# Local self-intersection COUNT (for best-of-K scoring; track_metrics' is a boolean).
# ---------------------------------------------------------------------------------------------

def _self_intersections_count(pts: np.ndarray) -> int:
    """Number of non-adjacent edge-pair crossings on the closed loop (segment-intersection).

    Vectorised over the upper-triangular edge-pair grid (the same proper-crossing test as
    track_metrics.self_intersects). This is a host SCORING helper for best-of-K only — the Warp
    port would use a tile/grid kernel, not this; the per-step steering loop is what ports.
    """
    n = len(pts)
    a = pts                              # [n,2] segment starts
    b = np.roll(pts, -1, axis=0)         # [n,2] segment ends

    def ccw(o, p, q):                    # o,p,q: [...,2] -> [...] signed area sign source
        return (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - \
               (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])

    ai = a[:, None, :]; bi = b[:, None, :]   # [n,1,2]
    aj = a[None, :, :]; bj = b[None, :, :]   # [1,n,2]
    d1 = ccw(ai, bi, aj); d2 = ccw(ai, bi, bj)
    d3 = ccw(aj, bj, ai); d4 = ccw(aj, bj, bi)
    crosses = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
    i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    adjacent = ((i + 1) % n == j) | ((j + 1) % n == i)
    mask = (j > i) & ~adjacent           # upper triangle, skip shared-endpoint/adjacent
    return int(np.count_nonzero(crosses & mask))


def _self_intersects(pts: np.ndarray) -> bool:
    return _self_intersections_count(pts) > 0


# ---------------------------------------------------------------------------------------------
# Driver: metrics tables (K=1, 4, 8) + single-candidate and best-of-K render grids.
# ---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _ROOT)
    from benchmarks.track_metrics import compactness, self_intersects
    # chicane_count / straight_fraction may not exist on every branch's track_metrics copy;
    # fall back to local equivalents built on the same primitives if so.
    try:
        from benchmarks.track_metrics import chicane_count, straight_fraction
    except ImportError:
        from benchmarks.track_metrics import turn_angles, curvature

        def chicane_count(pts, min_turn=0.05):
            ta = turn_angles(pts)
            sig = np.sign(ta)[np.abs(ta) >= min_turn]
            if sig.size < 2:
                return 0
            return int(np.count_nonzero(sig[1:] != sig[:-1]))

        def straight_fraction(pts, rel=0.5):
            k = curvature(pts); m = float(np.mean(k))
            return 0.0 if m <= 1e-9 else float(np.mean(k < rel * m))

    def score_intersections(pts):
        return _self_intersections_count(pts)

    cfg = DEFAULTS.copy()
    N_SEEDS = 500
    N_SINT = 150                                     # self-intersection sampled on first 150 (O(n^2))

    def run_block(K: int):
        comp, chic, strt, sint = [], [], [], []
        for seed in range(N_SEEDS):
            p = generate_centerline(seed, cfg, K=K, score=score_intersections)
            comp.append(compactness(p)); chic.append(chicane_count(p)); strt.append(straight_fraction(p))
            if seed < N_SINT:
                sint.append(self_intersects(p))
        comp = np.array(comp)
        return dict(
            c10=np.percentile(comp, 10), c50=np.percentile(comp, 50), c90=np.percentile(comp, 90),
            chic=float(np.mean(chic)), strt=float(np.mean(strt)), sint=float(np.mean(sint)),
        )

    print(f"=== checkpoint proto #5 (closure={cfg['closure']}, C={cfg['checkpoint_count']}, "
          f"turn_rate={cfg['turn_rate']}, N={cfg['num_points']}), {N_SEEDS} seeds ===")
    print(f"{'K':>3} | {'comp p10/p50/p90':>22} | {'chicane':>7} | {'straight':>8} | "
          f"{'self-int rate':>13}")
    print("-" * 70)
    results = {}
    for K in (1, 4, 8):
        r = run_block(K)
        results[K] = r
        print(f"{K:>3} | {r['c10']:>6.3f}/{r['c50']:.3f}/{r['c90']:.3f}    | "
              f"{r['chic']:>7.2f} | {r['strt']:>8.3f} | {r['sint']:>13.3f}")

    os.makedirs(os.path.join(_ROOT, "viz", "out"), exist_ok=True)

    def render_grid(K: int, fname: str, title: str):
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        for i, ax in enumerate(axes.flat):
            p = generate_centerline(i, cfg, K=K, score=score_intersections)
            cp = np.vstack([p, p[0]])
            crossing = self_intersects(p)
            ax.plot(cp[:, 0], cp[:, 1], "r-" if crossing else "b-", lw=0.9)
            ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(f"s={i} c={compactness(p):.2f} st={straight_fraction(p):.2f}", fontsize=7)
        plt.suptitle(title, fontsize=12); plt.tight_layout()
        out = os.path.join(_ROOT, "viz", "out", fname)
        plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
        print(f"saved {out}")

    render_grid(1, "checkpoint_proto_grid_k1.png",
                f"Checkpoint Proto #5 — single candidate (K=1, closure={cfg['closure']})")
    render_grid(int(cfg["candidates"]), "checkpoint_proto_grid_k8.png",
                f"Checkpoint Proto #5 — best-of-{cfg['candidates']} (closure={cfg['closure']})")
