#!/usr/bin/env python3
"""Render the relaxation iteration-evolution video (``docs/_static/relaxation-iterations.mp4``).

This is the visual answer to "why does the solve need iterations". Four tracks whose raw
constant-spacing centerlines are deeply self-overlapping are shown side by side, evolving
from the raw centerline through the pure iterative XPBD solve and then through the
post-solve smoothing tail.

Why regenerating per frame is a true snapshot
---------------------------------------------
The solve is a *fixed, deterministic* launch sequence: running ``relax_iters=k`` executes
exactly the first ``k`` sweeps that a ``relax_iters=50`` run would, because the Chebyshev
omega schedule and the ``relax_sep_every`` cache-refresh cadence are indexed by the sweep
number, not by the total sweep count (verified: ``_cheby_schedule(k)`` equals the length-k
prefix of ``_cheby_schedule(50)``). Each env is generated independently from its per-env
seed, so regenerating the whole batch with a different ``relax_iters`` and slicing out the
chosen envs yields a genuine snapshot of the same trajectory. The script asserts the raw
(k=0) centerlines are bit-identical across two generations and aborts if not.

Frame plan
----------
* Main phase: k = 0 .. 50. k=0 is the raw frame (``relax_enable=False``); k=1..50 use
  ``relax_iters=k`` with the smoothing tail OFF so the sequence is the pure iterative solve.
* Tail phase: ``relax_iters=50`` with ``relax_smooth_passes=p`` and
  ``relax_smooth_spacing_iters=10`` for p = 0 .. 5.
* The final frame is held ~2 s (duplicated) so the converged track lingers.

Gate self-collision relaxation video (``--gates``)
--------------------------------------------------
``main_gates()`` renders the same deliverable for the GATE self-collision relaxation
(``docs/_static/relaxation-gates-iterations.mp4``): five gate sequences whose raw anchors
(``gate_solve_iters=0``) are heavily overlapping, evolving round by round through the
per-env Gauss-Seidel sphere separation. The snapshot argument is the same: the solve is a
fixed deterministic round loop — the coincident-pair tie-break angle hash in
``_relax_gate_spheres_k`` is indexed by the ROUND index ``it``, not by the total budget —
so a run with ``gate_solve_iters=k`` executes exactly the first ``k`` rounds of a longer
run, and regenerating the batch per ``k`` yields true snapshots of one trajectory. The
script asserts the raw anchors are bit-identical across two generations, and only shows
envs whose state at the round cap equals a much longer run (i.e. the early exit fired).

This module is standalone and is NOT part of ``render_readme_assets()`` (it needs ffmpeg).
Run from the repository root:

    .venv/bin/python -m viz.render_relaxation_video            # track video
    .venv/bin/python -m viz.render_relaxation_video --gates    # gate video

Outputs (all under ``docs/_static`` so Sphinx copies them verbatim into the build):
    docs/_static/relaxation-iterations.mp4
    docs/_static/relaxation-iterations-poster.png
    docs/_static/relaxation-gates-iterations.mp4          (--gates)
    docs/_static/relaxation-gates-iterations-poster.png   (--gates)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

from track_gen import TrackGenConfig, TrackGenerator  # noqa: E402
from viz.plot_tracks import make_rng  # noqa: E402

OUT_DIR = Path("docs/_static")
MP4_PATH = OUT_DIR / "relaxation-iterations.mp4"
POSTER_PATH = OUT_DIR / "relaxation-iterations-poster.png"
GATES_MP4_PATH = OUT_DIR / "relaxation-gates-iterations.mp4"
GATES_POSTER_PATH = OUT_DIR / "relaxation-gates-iterations-poster.png"

# Metric benchmark regime — identical to the relaxation still assets in render_readme_assets.
REGIME = dict(half_width=0.5, scale=10.0, spacing=0.30, N_max=384)
TAIL_OFF = dict(relax_smooth_passes=0, relax_smooth_spacing_iters=0)

BATCH = 64
SEED = 0
N_TRACKS = 4
MAIN_SWEEPS = 50            # k = 0 .. 50
TAIL_PASSES = 5            # p = 0 .. 5
SMOOTH_SPACING_ITERS = 10
FRAMERATE = 8
HOLD_FRAMES = 16          # ~2 s hold on the converged frame at 8 fps
SEP_BAND_EXCLUDE = 4      # circular-neighbour window excluded from the self-overlap metric

DEVICE = "cpu"           # overridden to "cuda" in main() when a GPU is present


def _gen(seed: int, batch: int, overrides: dict):
    """One deterministic CUDA/CPU batch in the metric regime; return numpy border views."""
    cfg = TrackGenConfig(num_envs=batch, device=DEVICE, **REGIME, **overrides)
    rng = make_rng(batch, seed=seed, device=DEVICE)
    track = TrackGenerator(cfg, rng).generate()
    n_max = track.center.shape[0] // batch
    center = wp.to_torch(track.center).cpu().numpy().reshape(batch, n_max, 3)[..., :2]
    outer = wp.to_torch(track.outer).cpu().numpy().reshape(batch, n_max, 3)[..., :2]
    inner = wp.to_torch(track.inner).cpu().numpy().reshape(batch, n_max, 3)[..., :2]
    valid = wp.to_torch(track.valid).cpu().numpy().astype(bool)
    count = wp.to_torch(track.count).cpu().numpy().astype(int)
    return dict(center=center, outer=outer, inner=inner, valid=valid, count=count)


def _min_nonband_dist(center: np.ndarray, n: int) -> float:
    """Smallest distance between non-neighbour vertices — a self-overlap proxy for the raw
    centerline. Values well below ``2 * half_width`` mean the raw band self-intersects."""
    p = center[:n]
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            circ = min(abs(i - j), n - abs(i - j))
            if circ > SEP_BAND_EXCLUDE:
                d = float(np.linalg.norm(p[i] - p[j]))
                if d < best:
                    best = d
    return best


def _choose_envs(raw: dict, final: dict) -> list[int]:
    """Pick the N_TRACKS envs with the worst raw self-overlap whose final track is valid,
    so the raw frame is dramatically kinked and the converged frame is clean."""
    scores: list[tuple[float, int]] = []
    for e in range(BATCH):
        n = int(raw["count"][e])
        if n < 8 or not np.isfinite(raw["center"][e, :n]).all() or not final["valid"][e]:
            continue
        scores.append((_min_nonband_dist(raw["center"][e], n), e))
    scores.sort()
    return [e for _, e in scores[:N_TRACKS]]


def _fixed_limits(raw: dict, final: dict, envs: list[int]) -> list[tuple]:
    """Per-track fixed (xlim, ylim) from the union of the raw and final borders, +10% pad,
    forced square (equal aspect) so nothing jitters across frames."""
    limits = []
    for e in envs:
        pts = []
        for src in (raw, final):
            n = int(src["count"][e])
            for key in ("outer", "inner"):
                a = src[key][e, :n]
                pts.append(a[np.isfinite(a).all(axis=1)])
        pts = np.vstack(pts)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = 0.5 * max(xmax - xmin, ymax - ymin) * 1.10
        limits.append(((cx - half, cx + half), (cy - half, cy + half)))
    return limits


def _draw_panel(ax, data: dict, e: int, xlim, ylim) -> None:
    n = int(data["count"][e])

    def closed(a):
        a = a[np.isfinite(a).all(axis=1)]
        return np.vstack([a, a[0]]) if len(a) else a

    o = closed(data["outer"][e, :n])
    inn = closed(data["inner"][e, :n])
    c = closed(data["center"][e, :n])
    if len(o) and len(inn) and len(o) == len(inn):
        ax.fill(np.concatenate([o[:, 0], inn[::-1, 0]]),
                np.concatenate([o[:, 1], inn[::-1, 1]]),
                color="0.82", zorder=1, linewidth=0)
    if len(o):
        ax.plot(o[:, 0], o[:, 1], color="#1f77b4", lw=1.1, zorder=3)
    if len(inn):
        ax.plot(inn[:, 0], inn[:, 1], color="#d62728", lw=1.1, zorder=3)
    if len(c):
        ax.plot(c[:, 0], c[:, 1], color="0.25", lw=0.7, ls="--", zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.8")


def _render_frame(path: Path, data: dict, envs: list[int], limits: list[tuple],
                  banner: str, accent: str) -> None:
    # Fixed figsize + dpi and NO tight bbox -> every frame is exactly the same pixel size
    # (a hard requirement for the encoder). 1440 x 468 px, both even.
    fig, axes = plt.subplots(1, N_TRACKS, figsize=(14.4, 4.68), dpi=100, facecolor="white")
    for ax, e, (xlim, ylim) in zip(axes, envs, limits):
        _draw_panel(ax, data, e, xlim, ylim)
    fig.suptitle(banner, fontsize=17, fontweight="bold", color=accent, y=0.965)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.06)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _encode(frame_dir: Path, crf: int, mp4_path: Path = MP4_PATH,
            framerate: int = FRAMERATE) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(framerate),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags", "+faststart",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------------------
# Gate self-collision relaxation video (--gates)
# --------------------------------------------------------------------------------------
# Illustrative gate geometry: same generator family styling as the README gate strip
# (viz/render_readme_assets.py) but with a larger gate_radius so the raw anchors overlap
# densely and the separation is unmistakable frame to frame. gate_width only draws the
# gate bar; it does not enter the separation target (2 * gate_radius = 0.26).
GATES_GENERATOR = "bezier"
GATES_RADIUS = 0.13
GATES_WIDTH = 0.16
GATES_MAX_GATES = 32
GATES_BATCH = 64
GATES_SEED = 0
GATES_N_ENVS = 5
GATES_ROUND_CAP = 24       # upper bound on the displayed round budget K
GATES_FRAMERATE = 3        # low fps: the visible motion dies in ~8 rounds, let it read
GATES_HOLD_FRAMES = 6      # ~2 s hold on the converged frame at 3 fps
GATES_START_HOLD = 2       # extra copies of the raw frame (~1 s total) so it registers
# Visual convergence tolerance for the per-env stop round and the round budget K. The
# solve keeps making bit-level float32 micro-moves (1e-4 .. 1e-6) for another ~10 rounds
# after all visible motion has stopped; at the panel scale (~2.2 world units across ~290
# px) one pixel is ~8e-3, so 1e-3 is comfortably sub-pixel. Frames past this point are
# pixel-identical dead time, hence the tolerance.
GATES_CONV_TOL = 1.0e-3


def _gen_gates(solve_iters: int):
    """One deterministic gate batch at round budget ``solve_iters``; numpy views."""
    from track_gen import GateGenConfig, GateGenerator

    cfg = GateGenConfig(
        generator=GATES_GENERATOR,
        num_envs=GATES_BATCH,
        device=DEVICE,
        gate_radius=GATES_RADIUS,
        gate_width=GATES_WIDTH,
        max_gates=GATES_MAX_GATES,
        gate_solve_iters=solve_iters,
    )
    rng = make_rng(GATES_BATCH, seed=GATES_SEED, device=DEVICE)
    gates = GateGenerator(cfg, rng).generate()
    g = gates.position.shape[0] // GATES_BATCH
    return dict(
        position=wp.to_torch(gates.position).cpu().numpy().reshape(GATES_BATCH, g, 3)[..., :2],
        tangent=wp.to_torch(gates.tangent).cpu().numpy().reshape(GATES_BATCH, g, 3)[..., :2],
        left=wp.to_torch(gates.left).cpu().numpy().reshape(GATES_BATCH, g, 3)[..., :2],
        right=wp.to_torch(gates.right).cpu().numpy().reshape(GATES_BATCH, g, 3)[..., :2],
        valid=wp.to_torch(gates.valid).cpu().numpy().astype(bool),
        count=wp.to_torch(gates.count).cpu().numpy().astype(int),
    )


def _gates_state(data: dict, e: int) -> np.ndarray:
    return data["position"][e, : int(data["count"][e])]


def _raw_overlap_pairs(data: dict, e: int, target: float) -> int:
    """Number of gate pairs closer than the separation target — the drama ranking."""
    p = _gates_state(data, e)
    n = len(p)
    return sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if float(np.linalg.norm(p[i] - p[j])) < target
    )


def _choose_gate_envs(seq: list[dict], long: dict, target: float) -> tuple[list[int], list[int]]:
    """Pick the GATES_N_ENVS envs with the densest raw overlap among envs that are valid
    at the cap, finite, and CONVERGED (state at the cap equals a much longer run — the
    early exit fired). Returns (envs, per-env visual stop round: the first round whose
    state is within GATES_CONV_TOL of the converged state)."""
    raw, fin = seq[0], seq[-1]
    rows: list[tuple[int, int, int]] = []
    for e in range(GATES_BATCH):
        c = int(raw["count"][e])
        if c < 4 or not fin["valid"][e] or not np.isfinite(_gates_state(raw, e)).all():
            continue
        if not np.array_equal(_gates_state(fin, e), _gates_state(long, e)):
            continue  # still moving at the round cap — cannot display a converged panel
        stop = next(
            k for k in range(len(seq))
            if float(np.linalg.norm(_gates_state(seq[k], e) - _gates_state(fin, e),
                                    axis=1).max()) <= GATES_CONV_TOL
        )
        rows.append((_raw_overlap_pairs(raw, e, target), stop, e))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    chosen = rows[:GATES_N_ENVS]
    return [e for _, _, e in chosen], [s for _, s, _ in chosen]


def _gate_limits(raw: dict, fin: dict, envs: list[int]) -> list[tuple]:
    """Per-env fixed square (xlim, ylim) from the union of raw and final gate extents
    (centres +/- gate_radius and the gate-bar endpoints), +10% pad."""
    limits = []
    for e in envs:
        pts = []
        for src in (raw, fin):
            c = int(src["count"][e])
            p = src["position"][e, :c]
            p = p[np.isfinite(p).all(axis=1)]
            pts += [p + GATES_RADIUS, p - GATES_RADIUS]
            for key in ("left", "right"):
                a = src[key][e, :c]
                pts.append(a[np.isfinite(a).all(axis=1)])
        pts = np.vstack(pts)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = 0.5 * max(xmax - xmin, ymax - ymin) * 1.10
        limits.append(((cx - half, cx + half), (cy - half, cy + half)))
    return limits


def _draw_gate_panel(ax, data: dict, e: int, xlim, ylim, *, converged: bool) -> None:
    """One env's gate sequence, styled like the README gate strip (dashed sequence line,
    centre dots, gate_radius circles, tangent ticks, gate-width bars) on FIXED limits."""
    c = int(data["count"][e])
    pos = data["position"][e, :c]
    finite = np.isfinite(pos).all(axis=1)
    pos = pos[finite]
    if len(pos) >= 2:
        closed = np.vstack([pos, pos[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="0.35", lw=0.7, ls="--", alpha=0.6, zorder=1)
    for pnt in pos:
        ax.add_patch(plt.Circle((pnt[0], pnt[1]), GATES_RADIUS, fill=False,
                                color="#64748b", lw=0.8, alpha=0.85, zorder=2))
    tan = data["tangent"][e, :c][finite]
    lft = data["left"][e, :c][finite]
    rgt = data["right"][e, :c][finite]
    if len(tan) == len(pos) and len(pos) > 0:
        ax.quiver(pos[:, 0], pos[:, 1], tan[:, 0], tan[:, 1], angles="xy",
                  scale_units="xy", scale=12, width=0.005, color="#f97316",
                  alpha=0.85, zorder=3)
    if len(lft) == len(rgt) == len(pos):
        for li, ri in zip(lft, rgt):
            ax.plot([li[0], ri[0]], [li[1], ri[1]], color="#2563eb", lw=1.3, zorder=3)
    if len(pos) > 0:
        ax.scatter(pos[:, 0], pos[:, 1], s=16, color="#111827", zorder=4)
    if converged:
        ax.text(0.97, 0.03, "converged", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, fontweight="bold", color="#15803d", zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.8")


def _render_gate_frame(path: Path, data: dict, envs: list[int], limits: list[tuple],
                       stops: list[int], k: int, banner: str, accent: str) -> None:
    # Fixed figsize + dpi and NO tight bbox -> constant even pixel size (1500 x 360 px).
    fig, axes = plt.subplots(1, GATES_N_ENVS, figsize=(15.0, 3.6), dpi=100,
                             facecolor="white")
    for ax, e, (xlim, ylim), stop in zip(axes, envs, limits, stops):
        _draw_gate_panel(ax, data, e, xlim, ylim, converged=k >= stop)
    fig.suptitle(banner, fontsize=15, fontweight="bold", color=accent, y=0.955)
    fig.subplots_adjust(left=0.008, right=0.992, top=0.84, bottom=0.03, wspace=0.06)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main_gates() -> None:
    global DEVICE
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"
    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: ffmpeg not found on PATH — required to encode the video.")

    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = 2.0 * GATES_RADIUS
    print(f"device={DEVICE}  batch={GATES_BATCH}  seed={GATES_SEED}  "
          f"generator={GATES_GENERATOR}  gate_radius={GATES_RADIUS} (target {target})")

    raw = _gen_gates(0)
    raw_check = _gen_gates(0)
    max_diff = float(np.nanmax(np.abs(raw["position"] - raw_check["position"])))
    if max_diff != 0.0:
        sys.exit(f"ERROR: raw gate anchors are not deterministic across two runs "
                 f"(max diff {max_diff}); the per-frame snapshots would not be a true "
                 f"trajectory. Aborting.")
    print(f"determinism check OK (raw max diff {max_diff})")

    # Snapshot every round budget once; reuse the batches for env choice AND frames.
    seq = [raw] + [_gen_gates(k) for k in range(1, GATES_ROUND_CAP + 1)]
    long = _gen_gates(GATES_ROUND_CAP + 16)
    envs, stops = _choose_gate_envs(seq, long, target)
    if len(envs) < GATES_N_ENVS:
        sys.exit(f"ERROR: only {len(envs)} valid converged envs found "
                 f"(need {GATES_N_ENVS}); raise GATES_ROUND_CAP or change the config.")
    K = min(max(stops) + 1, GATES_ROUND_CAP)
    print(f"chosen envs (densest raw overlap, valid + converged): {envs}")
    for e, s in zip(envs, stops):
        print(f"  env {e}: {_raw_overlap_pairs(raw, e, target)} raw overlapping pairs, "
              f"visually converged (within {GATES_CONV_TOL}) after round {s}")
    print(f"round budget K = {K}")

    # Monotone stabilization check: once an env is within tolerance of its converged
    # state it must never leave again — otherwise the "converged" tag would lie.
    for e, s in zip(envs, stops):
        for k in range(s, GATES_ROUND_CAP + 1):
            d = float(np.linalg.norm(_gates_state(seq[k], e) - _gates_state(seq[-1], e),
                                     axis=1).max())
            if d > GATES_CONV_TOL:
                sys.exit(f"ERROR: env {e} left the converged state at round {k} "
                         f"(dist {d}); stabilization is not monotone. Aborting.")
    print("monotone stabilization check OK (no env leaves its converged state)")

    limits = _gate_limits(raw, seq[K], envs)

    accent = "#1d4ed8"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        idx = 0
        for k in range(0, K + 1):
            if k == 0:
                banner = (f"raw overlapping gate anchors (gate_solve_iters=0)    |    "
                          f"round 0 / {K}")
            else:
                banner = f"gate self-collision relaxation    |    round {k} / {K}"
            _render_gate_frame(tmp_dir / f"frame_{idx:04d}.png", seq[k], envs, limits,
                               stops, k, banner, accent)
            idx += 1
            if k == 0:
                # Hold the raw frame ~1 s so the overlap registers before round 1 moves.
                for _ in range(GATES_START_HOLD):
                    shutil.copyfile(tmp_dir / "frame_0000.png",
                                    tmp_dir / f"frame_{idx:04d}.png")
                    idx += 1
        last_frame = tmp_dir / f"frame_{idx - 1:04d}.png"
        print(f"rendered {K + 1} round frames (plus {GATES_START_HOLD}-frame raw hold)")

        for _ in range(GATES_HOLD_FRAMES):
            shutil.copyfile(last_frame, tmp_dir / f"frame_{idx:04d}.png")
            idx += 1
        total = idx
        print(f"total frames (incl. {GATES_HOLD_FRAMES}-frame hold): {total}")

        # Poster = the final separated frame.
        shutil.copyfile(last_frame, GATES_POSTER_PATH)

        crf = 23
        _encode(tmp_dir, crf, GATES_MP4_PATH, GATES_FRAMERATE)
        while GATES_MP4_PATH.stat().st_size > 4_000_000 and crf < 28:
            crf += 2
            print(f"mp4 over 4 MB, re-encoding at crf={crf}")
            _encode(tmp_dir, crf, GATES_MP4_PATH, GATES_FRAMERATE)

    size = GATES_MP4_PATH.stat().st_size
    print(f"wrote {GATES_MP4_PATH}  ({size / 1e6:.2f} MB, {total} frames, "
          f"{total / GATES_FRAMERATE:.1f} s at {GATES_FRAMERATE} fps, crf={crf})")
    print(f"wrote {GATES_POSTER_PATH}  ({GATES_POSTER_PATH.stat().st_size} bytes)")


def main() -> None:
    global DEVICE
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"
    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: ffmpeg not found on PATH — required to encode the video.")

    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}  batch={BATCH}  seed={SEED}")

    raw = _gen(SEED, BATCH, {"relax_enable": False})
    raw_check = _gen(SEED, BATCH, {"relax_enable": False})
    max_diff = float(np.nanmax(np.abs(raw["center"] - raw_check["center"])))
    if max_diff != 0.0:
        sys.exit(f"ERROR: raw centerlines are not deterministic across two runs "
                 f"(max diff {max_diff}); the per-frame snapshots would not be a true "
                 f"trajectory. Aborting.")
    print(f"determinism check OK (raw max diff {max_diff})")

    final = _gen(SEED, BATCH, dict(relax_iters=MAIN_SWEEPS,
                                   relax_smooth_passes=TAIL_PASSES,
                                   relax_smooth_spacing_iters=SMOOTH_SPACING_ITERS))
    envs = _choose_envs(raw, final)
    print(f"chosen envs (worst raw self-overlap, valid final): {envs}")
    for e in envs:
        n = int(raw["count"][e])
        print(f"  env {e}: raw min non-band dist "
              f"{_min_nonband_dist(raw['center'][e], n):.3f} (2*hw = {2 * REGIME['half_width']})")
    limits = _fixed_limits(raw, final, envs)

    accent_main = "#1d4ed8"
    accent_tail = "#7c3aed"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        idx = 0

        # --- main phase: pure iterative solve, tail off ---
        for k in range(0, MAIN_SWEEPS + 1):
            if k == 0:
                data = raw
                banner = "raw self-overlapping centerline    |    sweep 0 / 50"
            else:
                data = _gen(SEED, BATCH, dict(relax_iters=k, **TAIL_OFF))
                banner = f"XPBD iterative solve    |    sweep {k} / 50"
            _render_frame(tmp_dir / f"frame_{idx:04d}.png", data, envs, limits,
                          banner, accent_main)
            idx += 1
        print(f"rendered {idx} main-phase frames")

        # --- tail phase: smoothing tail, increasing Taubin passes ---
        last_frame = None
        for p in range(0, TAIL_PASSES + 1):
            data = _gen(SEED, BATCH, dict(relax_iters=MAIN_SWEEPS,
                                          relax_smooth_passes=p,
                                          relax_smooth_spacing_iters=SMOOTH_SPACING_ITERS))
            banner = f"smoothing tail — pass {p} / 5    |    (50 sweeps + Taubin + polish)"
            frame_path = tmp_dir / f"frame_{idx:04d}.png"
            _render_frame(frame_path, data, envs, limits, banner, accent_tail)
            last_frame = frame_path
            idx += 1
        print(f"rendered {TAIL_PASSES + 1} tail-phase frames")

        # --- hold the converged frame ~2 s ---
        for _ in range(HOLD_FRAMES):
            shutil.copyfile(last_frame, tmp_dir / f"frame_{idx:04d}.png")
            idx += 1
        total = idx
        print(f"total frames (incl. {HOLD_FRAMES}-frame hold): {total}")

        # Poster = the final converged frame.
        shutil.copyfile(last_frame, POSTER_PATH)

        crf = 23
        _encode(tmp_dir, crf)
        while MP4_PATH.stat().st_size > 4_000_000 and crf < 28:
            crf += 2
            print(f"mp4 over 4 MB, re-encoding at crf={crf}")
            _encode(tmp_dir, crf)

    size = MP4_PATH.stat().st_size
    print(f"wrote {MP4_PATH}  ({size / 1e6:.2f} MB, {total} frames, "
          f"{total / FRAMERATE:.1f} s at {FRAMERATE} fps, crf={crf})")
    print(f"wrote {POSTER_PATH}  ({POSTER_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    if "--gates" in sys.argv[1:]:
        main_gates()
    else:
        main()
