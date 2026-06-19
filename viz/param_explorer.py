#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Interactive Gradio explorer for track-generation parameters.

A thin Gradio shell over a pure core (`build_config` + `render_grid`) that drives the real
pure-Warp pipeline (`track_gen.warp_pipeline.generate_tracks_warp`) and renders a grid of
tracks + yield/quality stats. Launch:  `.venv/bin/python -m viz.param_explorer`
(needs the `ui` extra: `pip install -e ".[ui]"`).
"""
from __future__ import annotations

import math
import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot

import matplotlib.pyplot as plt
import torch

import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src import warp_pipeline as wpl
from viz.plot_tracks import draw_track

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_config(p: dict) -> TrackGenConfig:
    """Map a params dict to a TrackGenConfig, clamping degenerate inputs.

    Output is always constant_spacing (the only supported mode): ``spacing`` is the
    arc-length step and ``N_max`` the per-track point cap. ``num_points`` is the
    intermediate dense-resample resolution before constant-spacing (optional; the
    config default is used when absent).
    """
    lo = min(int(p["min_num_points"]), int(p["max_num_points"]))
    hi = max(int(p["min_num_points"]), int(p["max_num_points"]))
    grid_n = int(p["grid_n"])
    num_envs = int(p.get("batch_size", grid_n * grid_n))
    kw = {}
    if p.get("num_points") is not None:
        kw["num_points"] = int(p["num_points"])
    if p.get("spacing") is not None:
        kw["spacing"] = float(p["spacing"])
    return TrackGenConfig(
        num_envs=num_envs,
        half_width=float(p["half_width"]),
        scale=float(p["scale"]),
        min_num_points=lo,
        max_num_points=hi,
        rad=float(p["rad"]),
        edgy=float(p["edgy"]),
        handle_clamp_frac=float(p.get("handle_clamp_frac", 0.4)),
        output_mode="constant_spacing",
        N_max=int(p["n_max"]),
        relax_iters=int(p["relax_iters"]),
        relax_sep_relax=float(p["relax_sep_relax"]),
        relax_spc_relax=float(p["relax_spc_relax"]),
        relax_bend_relax=float(p["relax_bend_relax"]),
        relax_margin=float(p["relax_margin"]),
        device=DEVICE,
        **kw,
    )


def n_pages(E: int, grid_n: int) -> int:
    """Return the number of pages needed to display E tracks in a grid_n×grid_n grid."""
    return max(1, math.ceil(E / (grid_n * grid_n)))


def generate_batch(p: dict):
    """Generate a full batch of tracks (batch_size envs). Returns the Track object."""
    wp.init()
    cfg = build_config(p)
    E = cfg.num_envs
    seeds = (torch.arange(E, dtype=torch.int32) + int(p["seed"])).to(DEVICE)
    return wpl.generate_tracks_warp(cfg, seeds)


def render_page(track, page: int, grid_n: int):
    """Draw a grid_n×grid_n window of the cached Track starting at page*grid_n**2.

    Returns a matplotlib Figure with grid_n**2 axes. Cells beyond the batch are left blank.
    """
    E = track.center.shape[0]
    np_ = n_pages(E, grid_n)
    start = page * grid_n * grid_n
    fig, axes = plt.subplots(grid_n, grid_n, figsize=(2.1 * grid_n, 2.1 * grid_n))
    axes = axes.flatten() if grid_n > 1 else [axes]
    for k, ax in enumerate(axes):
        idx = start + k
        if idx < E:
            draw_track(ax, track, idx)
        else:
            ax.axis("off")
    fig.suptitle(f"{grid_n}x{grid_n}  ·  page {page + 1}/{np_}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _stats(track) -> dict:
    """Aggregate readout over the batch (means taken over valid tracks).

    Output is always constant_spacing: ``count`` is the per-track real-point count and
    VARIES per env, so the reported ``count`` is the mean over valid tracks.
    """
    valid = track.valid
    n = int(valid.sum())
    if n == 0:
        return {"yield": 0.0, "n_valid": 0, "mean_len": float("nan"),
                "mean_thickness": float("nan"), "count": 0.0}
    # half-width from the first REAL point of each env (index 0 is always real / non-NaN).
    hw = float(torch.linalg.norm(track.outer[:, 0] - track.center[:, 0], dim=-1).median())
    cnt = track.count.clamp_min(1)
    band = (2.0 * hw / (track.length / cnt.float()).clamp_min(1e-9)).round().to(torch.int32).clamp_min(1)
    th = wpl.thickness(track.center, band, count=track.count.to(torch.int32))
    return {
        "yield": float(valid.float().mean()),
        "n_valid": n,
        "mean_len": float(track.length[valid].mean()),
        "mean_thickness": float(th[valid].mean()),
        "count": float(track.count[valid].float().mean()),
    }


def render_grid(p: dict):
    """Generate a batch, draw page 0, compute stats over the full batch. Returns (Figure, stats dict).
    Any pipeline error is caught and returned as a small error figure + an 'error' stat."""
    try:
        track = generate_batch(p)
        fig = render_page(track, 0, int(p["grid_n"]))
        st = _stats(track)
        return fig, st
    except Exception as exc:  # never crash the UI
        fig = plt.figure(figsize=(5, 3))
        fig.text(0.5, 0.5, f"error: {exc}", ha="center", va="center", fontsize=9, color="red", wrap=True)
        return fig, {"error": str(exc), "yield": 0.0, "n_valid": 0, "mean_len": float("nan"),
                     "mean_thickness": float("nan"), "count": 0}


def _collect(*vals) -> dict:
    keys = ["half_width", "scale", "min_num_points", "max_num_points", "rad", "edgy",
            "handle_clamp_frac", "spacing", "n_max", "relax_iters",
            "relax_sep_relax", "relax_spc_relax", "relax_bend_relax", "relax_margin",
            "grid_n", "seed", "batch_size"]
    return dict(zip(keys, vals))


def _stats_md(st: dict) -> str:
    if "error" in st:
        return f"**error:** {st['error']}"
    return (f"**valid yield: {st['yield'] * 100:.0f}%**  ·  {st['n_valid']} valid  ·  "
            f"mean length {st['mean_len']:.1f} m  ·  mean thickness {st['mean_thickness']:.3f} m  ·  "
            f"mean count {st['count']:.0f}")


def build_app():
    """Build the Gradio Blocks UI (does not launch). Requires the `ui` extra."""
    import gradio as gr

    with gr.Blocks(title="Track-gen parameter explorer") as app:
        gr.Markdown("## Track-gen parameter explorer")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Regime")
                half_width = gr.Slider(0.05, 1.0, value=0.5, step=0.01, label="half_width (m)")
                scale = gr.Slider(1.0, 20.0, value=10.0, step=0.5, label="scale (box)")
                gr.Markdown("### Shape")
                min_np = gr.Slider(5, 20, value=9, step=1, label="min corners")
                max_np = gr.Slider(5, 20, value=13, step=1, label="max corners")
                rad = gr.Slider(0.0, 0.6, value=0.4, step=0.01, label="rad (roundness)")
                edgy = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="edgy")
                handle_clamp = gr.Slider(0.0, 1.0, value=0.4, step=0.01,
                                         label="handle_clamp_frac (overshoot↔roundness)")
                gr.Markdown("### Resolution (constant-spacing)")
                spacing = gr.Slider(0.1, 1.0, value=0.30, step=0.02, label="spacing (m)")
                n_max = gr.Slider(128, 512, value=384, step=8, label="N_max")
                gr.Markdown("### Relaxation")
                relax_iters = gr.Slider(0, 600, value=150, step=10, label="relax_iters")
                sep = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="sep factor")
                spc = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="spc factor")
                bend = gr.Slider(0.0, 2.0, value=1.5, step=0.1, label="bend factor")
                margin = gr.Slider(0.0, 0.5, value=0.15, step=0.01, label="relax_margin")
                gr.Markdown("### Batch")
                grid_n = gr.Dropdown([3, 4, 5, 6], value=4, label="grid (n x n)")
                seed = gr.Number(value=0, precision=0, label="seed")
                batch_size = gr.Dropdown([256, 1024, 2048, 4096, 8192], value=2048, label="batch size")
                with gr.Row():
                    reroll = gr.Button("reroll seed")
                    generate = gr.Button("Generate", variant="primary")
                auto = gr.Checkbox(value=True, label="auto-update")
            with gr.Column(scale=2):
                stats = gr.Markdown("")
                with gr.Row():
                    prev_btn = gr.Button("◀ prev")
                    page_lbl = gr.Markdown("page 1/1")
                    next_btn = gr.Button("next ▶")
                plot = gr.Plot()

        # State: cached Track object and current page index
        track_state = gr.State(None)
        page_state = gr.State(0)

        controls = [half_width, scale, min_np, max_np, rad, edgy, handle_clamp,
                    spacing, n_max, relax_iters, sep, spc, bend, margin, grid_n, seed,
                    batch_size]

        def _generate(*vals):
            p = _collect(*vals)
            gn = int(p["grid_n"])
            try:
                track = generate_batch(p)
                fig = render_page(track, 0, gn)
                st = _stats(track)
                lbl = f"page 1/{n_pages(track.center.shape[0], gn)}"
                return fig, _stats_md(st), track, 0, lbl
            except Exception as exc:
                err_fig = plt.figure(figsize=(5, 3))
                err_fig.text(0.5, 0.5, f"error: {exc}", ha="center", va="center",
                             fontsize=9, color="red", wrap=True)
                err_st = {"error": str(exc), "yield": 0.0, "n_valid": 0,
                          "mean_len": float("nan"), "mean_thickness": float("nan"), "count": 0}
                return err_fig, _stats_md(err_st), None, 0, "page 1/1"

        def _go(track, page, gn, delta):
            if track is None:
                return gr.update(), page, gr.update()
            np_ = n_pages(track.center.shape[0], int(gn))
            new = max(0, min(int(page) + delta, np_ - 1))
            fig = render_page(track, new, int(gn))
            return fig, new, f"page {new + 1}/{np_}"

        generate.click(_generate, controls, [plot, stats, track_state, page_state, page_lbl])
        reroll.click(lambda s: int(s) + 1, seed, seed).then(
            _generate, controls, [plot, stats, track_state, page_state, page_lbl])

        prev_btn.click(lambda t, pg, g: _go(t, pg, g, -1),
                       [track_state, page_state, grid_n], [plot, page_state, page_lbl])
        next_btn.click(lambda t, pg, g: _go(t, pg, g, +1),
                       [track_state, page_state, grid_n], [plot, page_state, page_lbl])

        def _maybe(*vals):
            *rest, auto_on = vals
            if not auto_on:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            return _generate(*rest)
        for c in controls:
            ev = c.release if hasattr(c, "release") else c.change
            ev(_maybe, controls + [auto], [plot, stats, track_state, page_state, page_lbl])

        app.load(_generate, controls, [plot, stats, track_state, page_state, page_lbl])
    return app


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
