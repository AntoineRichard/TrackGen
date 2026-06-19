#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a multi-page PDF report of the E=8192 yield study with illustrative plots.

Pages: (1) overview + sweep table, (2) quantitative lever impact (yield vs chain-links /
relax-iters / PBD-step, plus speed), (3) fixed-seed x relax-iters, (4) fixed-seed x
PBD-step, (5) fixed-seed x chain-links, (6) findings & caveats. The fixed-seed pages
re-generate the SAME tracks (identical per-env seeds) under varying settings so each row
is one track and you can see the lever's impact directly. Everything is in metres
(half_width=0.5 m, scale=10 -> ~20 m box; the 1 m-track regime).

Quantitative numbers are the measured E=8192 results from benchmark_yield_sweep.py.
Headless (Agg); writes viz/out/track_gen_report.pdf.

    .venv/bin/python -m viz.make_report
"""
from __future__ import annotations

import os
import sys
import textwrap

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

import warp as wp

from track_gen.types import TrackGenConfig
from track_gen import warp_pipeline as wpl

OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "track_gen_report.pdf")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF_WIDTH = 0.5     # 1 m track width
SCALE = 10.0         # ~20 m box
SCALEBAR_M = 5.0

# ------------------------------------------------------------------ measured E=8192 data
# (from benchmark_yield_sweep.py, 1m/20m regime, RTX 5000 Ada; yield is at the working
#  resolution -- see the caveats page). Filled from the sweep run.
LINKS = {"x": [128, 256, 384, 512], "yield": [0.992, 0.684, 0.203, 0.031],
         "sec": [0.213, 0.782, 1.754, 3.081]}
ITERS = {"x": [50, 150, 300, 600], "yield": [0.522, 0.684, 0.766, 0.825],
         "sec": [0.569, 0.782, 1.105, 1.779]}
REGEN = {"x": [10, 20, 40], "yield": [0.684, 0.684, 0.684],
         "sec": [0.782, 1.162, 1.910]}
# XPBD over-relaxation: LARGER steps HURT (Jacobi solve is unstable for factor > 1).
PBD = {"x": [1.0, 1.5, 2.0, 2.5], "yield": [0.684, 0.341, 0.022, 0.000],
       "sec": [0.782, 0.785, 0.787, 0.795]}
PBD_EXTRA = {  # margin / combined probes (256 links, iters=150 unless noted)
    "margin 0.15->0.30": {"yield": 0.453, "sec": 0.796},
    "step x2.0 + margin0.30": {"yield": 0.000, "sec": 0.789},
    "step x2.0, iters=50": {"yield": 0.023, "sec": 0.565},
}

BASELINE = {"links": 256, "iters": 150, "regen": 10, "yield": 0.684, "sec": 0.782}

# Constant-spacing result (E=8192, spacing=0.30 m ≈ 0.6×half_width, N_max=384, iters=150)
CS = {"spacing": 0.30, "n_max": 384, "yield": 0.999, "sec": 0.557, "peak_mb": 353}


# ------------------------------------------------------------------ track drawing
def _np_loop(arr2d: torch.Tensor):
    pts = arr2d.detach().cpu().numpy()
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return np.array([]), np.array([])
    pts = np.vstack([pts, pts[0]])
    return pts[:, 0], pts[:, 1]


def _draw(ax, track, e, title_prefix):
    cx, cy = _np_loop(track.center[e])
    ox, oy = _np_loop(track.outer[e])
    ix, iy = _np_loop(track.inner[e])
    if ox.size and ix.size and ox.size == ix.size:
        ax.fill(np.concatenate([ox, ix[::-1]]), np.concatenate([oy, iy[::-1]]),
                color="0.80", zorder=1, linewidth=0)
    if ox.size:
        ax.plot(ox, oy, color="#1f77b4", lw=0.8, zorder=3)
    if ix.size:
        ax.plot(ix, iy, color="#d62728", lw=0.8, zorder=3)
    if cx.size:
        ax.plot(cx, cy, color="0.25", lw=0.5, ls="--", zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if cx.size:
        ax.relim(); ax.autoscale_view()
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.autoscale(False)
        sx, sy = x0 + 0.06 * (x1 - x0), y0 + 0.06 * (y1 - y0)
        ax.plot([sx, sx + SCALEBAR_M], [sy, sy], color="k", lw=1.2, zorder=6)
        ax.text(sx, sy, f" {SCALEBAR_M:g}m", fontsize=4, va="bottom", zorder=6)
    else:
        # No finite points -> the relaxation diverged (over-relaxation blew up).
        ax.text(0.5, 0.5, "diverged\n(non-finite)", transform=ax.transAxes,
                ha="center", va="center", fontsize=6, color="0.55")
    invalid = not bool(track.valid[e].item())
    ax.set_title(f"{title_prefix}{'  X' if invalid else ''}", fontsize=6,
                 color=("red" if invalid else "black"), pad=1.5)


def _gen(links, iters, n, regen=10, sr=1.0, pr=1.0, margin=0.15, seed0=0,
         output_mode="constant_spacing", spacing=0.10, n_max=256):
    cfg = TrackGenConfig(num_envs=n, num_points=links, half_width=HALF_WIDTH, scale=SCALE,
                         relax_iters=iters, max_regen_iters=regen, relax_sep_relax=sr,
                         relax_spc_relax=pr, relax_margin=margin, device=DEVICE,
                         output_mode=output_mode, spacing=spacing, N_max=n_max)
    seeds = (torch.arange(n, dtype=torch.int32) + seed0).to(DEVICE)
    return wpl.generate_tracks_warp(cfg, seeds)


# ------------------------------------------------------------------ pages
def _text_page(pdf, title, lines, fontsize=10):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.5, 0.95, title, ha="center", fontsize=15, weight="bold")
    fig.text(0.07, 0.88, "\n".join(lines), ha="left", va="top", fontsize=fontsize,
             family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def page_overview(pdf):
    lines = [
        "Pure-Warp track-generation pipeline -- E=8192 yield study",
        f"device: {DEVICE}    regime: half_width=0.5 m, scale=10  (1 m track / ~20 m box)",
        "pipeline: generate -> resample -> XPBD relax -> inflate  (all NVIDIA Warp)",
        "",
        "Sweep results (end-to-end generate_tracks_warp, warmed + synced, E=8192):",
        "",
        f"{'lever':<26}{'yield':>8}{'s/8192':>9}",
        "-" * 43,
        f"{'baseline 256 links/150 it':<26}{BASELINE['yield']:>8.3f}{BASELINE['sec']:>9.3f}",
        "",
        "chain links (iters=150):",
    ]
    for x, y, s in zip(LINKS["x"], LINKS["yield"], LINKS["sec"]):
        lines.append(f"  {x:>4} links{'':<14}{y:>8.3f}{s:>9.3f}")
    lines.append("relax iters (256 links):")
    for x, y, s in zip(ITERS["x"], ITERS["yield"], ITERS["sec"]):
        lines.append(f"  {x:>4} iters{'':<14}{y:>8.3f}{s:>9.3f}")
    lines.append("regen attempts (256 links):")
    for x, y, s in zip(REGEN["x"], REGEN["yield"], REGEN["sec"]):
        lines.append(f"  {x:>4} regen{'':<14}{y:>8.3f}{s:>9.3f}")
    lines.append("XPBD step scale sr=pr (256 links, 150 it):")
    for x, y, s in zip(PBD["x"], PBD["yield"], PBD["sec"]):
        if y is not None:
            lines.append(f"  x{x:<4} steps{'':<13}{y:>8.3f}{s:>9.3f}")
    for k, v in PBD_EXTRA.items():
        lines.append(f"  {k:<22}{v['yield']:>8.3f}{v['sec']:>9.3f}")
    lines += [
        "",
        "constant_spacing (the convergence fix):",
        f"  spacing=0.30 m, N_max=384 {'':<2}{CS['yield']:>8.3f}{CS['sec']:>9.3f}"
        f"  <- yield {BASELINE['yield']:.3f} -> {CS['yield']:.3f}",
    ]
    _text_page(pdf, "Track-gen yield study", lines, fontsize=9)


def page_quant(pdf):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    (a, b), (c, d) = axes

    a.plot(LINKS["x"], LINKS["yield"], "o-", color="#1f77b4")
    a.set_title("Yield vs chain links (iters=150)\n(resolution-relative -- see caveats)")
    a.set_xlabel("num_points (chain links)"); a.set_ylabel("valid yield"); a.set_ylim(0, 1.02)
    a.grid(alpha=0.3)

    b.plot(ITERS["x"], ITERS["yield"], "o-", color="#2ca02c", label="yield")
    b.set_title("Yield vs relax iters (256 links, fair comparison)")
    b.set_xlabel("relax_iters"); b.set_ylabel("valid yield"); b.set_ylim(0, 1.02)
    b.grid(alpha=0.3)

    pbx = [x for x, y in zip(PBD["x"], PBD["yield"]) if y is not None]
    pby = [y for y in PBD["yield"] if y is not None]
    c.plot(pbx, pby, "o-", color="#d62728")
    c.axhline(BASELINE["yield"], ls=":", color="0.5", label="baseline (step x1)")
    c.set_title("Yield vs XPBD step scale (256 links, iters=150)")
    c.set_xlabel("sep/spc relaxation factor (over-relaxation)")
    c.set_ylabel("valid yield"); c.set_ylim(0, 1.02); c.legend(fontsize=8); c.grid(alpha=0.3)

    d.plot(LINKS["x"], LINKS["sec"], "o-", color="#1f77b4", label="vs links")
    d.plot([], [])
    d2 = d.twiny()
    d2.plot(ITERS["x"], ITERS["sec"], "s-", color="#2ca02c", label="vs iters")
    d.set_title("End-to-end speed @ E=8192")
    d.set_xlabel("num_points", color="#1f77b4"); d2.set_xlabel("relax_iters", color="#2ca02c")
    d.set_ylabel("s / call (8192 tracks)"); d.grid(alpha=0.3)
    d.plot(REGEN["x"], REGEN["sec"], "^:", color="0.5")
    fig.suptitle("Lever impact on yield and speed (measured E=8192)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def _fixed_seed_page(pdf, title, subtitle, col_settings, gen_fn, n_seeds=5):
    """Rows = fixed seeds, cols = lever settings. gen_fn(setting) -> Track for the n_seeds."""
    ncol = len(col_settings)
    fig, axes = plt.subplots(n_seeds, ncol, figsize=(2.0 * ncol, 2.0 * n_seeds))
    axes = np.atleast_2d(axes)
    for j, (label, setting) in enumerate(col_settings):
        tr = gen_fn(setting, n_seeds)
        for i in range(n_seeds):
            ax = axes[i, j]
            _draw(ax, tr, i, label if i == 0 else "")
            if j == 0:
                ax.set_ylabel(f"seed {i}", fontsize=7)
    sub = "\n".join(textwrap.wrap(subtitle, width=max(55, 20 * ncol)))  # fit the figure width
    fig.suptitle(f"{title}\n{sub}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def page_constant_spacing(pdf):
    """Fixed-seed comparison: over-resolved spacing (jagged) vs ~0.6*half_width (smooth).

    (The legacy fixed-count mode was dropped; the same convergence lesson is now shown
    within constant_spacing by varying the arc-length step.)"""
    caption = (
        f"Constant spacing is the convergence fix: a too-fine arc-length step over-resolves "
        f"the chain (jagged XPBD, pinching at tight bends), while ~0.6×half_width gives the "
        f"relaxation a uniform, well-conditioned chain. Right column (spacing=0.30 m): "
        f"yield={CS['yield']:.3f} (E=8192, {CS['sec']:.3f} s/call, {CS['peak_mb']:.0f} MB). "
        f"Relaxation is lossless (valid ≈ generation-valid), so generation is the ceiling. "
        f"X = invalid track."
    )
    _fixed_seed_page(
        pdf,
        "Constant spacing — the convergence fix",
        caption,
        [
            ("constant_spacing\nspacing=0.05 m (over-resolved)", 0.05),
            ("constant_spacing\nspacing=0.30 m (relax-friendly)", CS["spacing"]),
        ],
        lambda spacing, n: _gen(
            256, 150, n,
            output_mode="constant_spacing",
            spacing=spacing,
            n_max=CS["n_max"],
        ),
        n_seeds=5,
    )


def main():
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        page_overview(pdf)
        page_quant(pdf)

        # p3: fixed-seed x relax iters (same generated track, relaxed progressively).
        _fixed_seed_page(
            pdf, "Fixed-seed impact of RELAX ITERS (256 links)",
            "Same generated track per row; columns relax it more. Watch pinched tracks "
            "converge (X = invalid). Yield is convergence-limited.",
            [(f"{it} iters", it) for it in (50, 150, 300, 600)],
            lambda it, n: _gen(256, it, n),
        )
        # p4: fixed-seed x PBD step scale (same gen track, bigger XPBD steps).
        _fixed_seed_page(
            pdf, "Fixed-seed impact of XPBD STEP SIZE (256 links, iters=150)",
            "Same generated track per row; columns use larger over-relaxation steps "
            "(sep/spc factor). The Jacobi solve is unstable for factor > 1: bigger steps "
            "OSCILLATE/diverge -> tracks degrade into invalid (X). Larger != faster here.",
            [(f"step x{s:g}", s) for s in (1.0, 1.5, 2.0, 2.5)],
            lambda s, n: _gen(256, 150, n, sr=s, pr=s),
        )
        # p5: fixed-seed x chain links (gen may differ slightly -- gate resolution).
        _fixed_seed_page(
            pdf, "Fixed-seed impact of CHAIN LINKS (iters=150)",
            "Same seeds; more links = finer relax/validation resolution (note: the "
            "accepted track can differ since the gate resolution changes).",
            [(f"{n} links", n) for n in (128, 256, 512)],
            lambda lk, n: _gen(lk, 150, n),
        )
        # p6: constant-spacing vs fixed-256 comparison.
        page_constant_spacing(pdf)

        findings = [
            "FINDINGS (1 m / 20 m regime, E=8192):",
            "",
            "* SPEED: baseline 256 links/150 it = 0.79 s/8192 (235 MB). Scales ~O(links^2)",
            "  and ~linear in iters; memory is tiny (<0.5 GB of 16).",
            "",
            "* RELAX ITERS -> genuinely raise yield (fair, fixed-256-res): 0.52/0.68/0.77/",
            "  0.83 at 50/150/300/600 it. Yield is RELAXATION-CONVERGENCE-limited.",
            "",
            "* XPBD STEP SIZE (over-relaxation) -> larger steps HURT, decisively:",
            "  yield 0.68/0.34/0.02/0.00 at sep/spc factor x1/1.5/2/2.5 (speed unchanged).",
            "  The solve is JACOBI (constraints applied simultaneously), so factor > 1",
            "  oscillates/diverges; x1 is already the stability ceiling. Aiming further",
            "  (margin 0.15->0.30) also drops yield to 0.45. Faster convergence would need",
            "  a better solver (Gauss-Seidel / Chebyshev / chunking), not bigger steps.",
            "",
            "* CHAIN LINKS -> more links LOWERS native yield + costs O(N^2). BUT yield is",
            "  RESOLUTION-RELATIVE (the thickness curvature term is sampled at num_points),",
            "  so link counts are NOT directly comparable -- fewer links = looser gate, not",
            "  better tracks. Don't tune links to chase yield.",
            "",
            "* REGEN ATTEMPTS -> NO yield change (10/20/40 all 0.684), only slower. The",
            "  regen loop gates on GENERATION simplicity (saturated ~100%); final validity",
            "  is the POST-RELAX check applied after the loop, which regen can't fix.",
            "",
            "CAVEATS (adversarially verified):",
            "* Yield is 'valid at the working resolution', not resolution-absolute.",
            "* Re-validating at finer resolution collapses the thickness CURVATURE term",
            "  (artifact of resampling a polyline); SEPARATION stays honest (~0.47 m vs the",
            "  0.49 m target) -- the 1 m-band tracks genuinely do NOT self-overlap.",
            "",
            "RECOMMENDATION (1 m/20 m): keep num_points=256; raise relax_iters to ~300-600",
            "for higher honest yield (0.77-0.83). Do NOT enlarge XPBD steps or margin (they",
            "destabilize), and leave max_regen_iters=10 (more is wasted). The ceiling is",
            "relaxation convergence -- the real win is a faster-converging solver.",
        ]
        _text_page(pdf, "Findings & recommendation", findings, fontsize=9)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    wp.init()
    main()
