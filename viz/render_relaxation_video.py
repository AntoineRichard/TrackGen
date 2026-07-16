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

This module is standalone and is NOT part of ``render_readme_assets()`` (it needs ffmpeg).
Run from the repository root:

    .venv/bin/python -m viz.render_relaxation_video

Outputs (both under ``docs/_static`` so Sphinx copies them verbatim into the build):
    docs/_static/relaxation-iterations.mp4
    docs/_static/relaxation-iterations-poster.png
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
    center = wp.to_torch(track.center).cpu().numpy().reshape(batch, n_max, 2)
    outer = wp.to_torch(track.outer).cpu().numpy().reshape(batch, n_max, 2)
    inner = wp.to_torch(track.inner).cpu().numpy().reshape(batch, n_max, 2)
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


def _encode(frame_dir: Path, crf: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FRAMERATE),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags", "+faststart",
        str(MP4_PATH),
    ]
    subprocess.run(cmd, check=True)


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
    main()
