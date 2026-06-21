#!/usr/bin/env python3
"""Interactive Gradio explorer for track-generation parameters.

A thin Gradio shell over a pure core (`build_config` + `render_grid`) that drives the real
pure-Warp pipeline (`TrackGenerator.generate`) and renders a grid of
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
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src import generator_registry
from viz.plot_tracks import draw_track

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def default_params() -> dict:
    """Return the Gradio explorer defaults as a params dict accepted by build_config."""
    cfg = TrackGenConfig()
    return {
        "generator": "polar",
        "half_width": 0.5,
        "scale": 10.0,
        "min_num_points": cfg.min_num_points,
        "max_num_points": cfg.max_num_points,
        "rad": cfg.rad,
        "edgy": cfg.edgy,
        "handle_clamp_frac": cfg.handle_clamp_frac,
        "polar_num_knots": cfg.polar_num_knots,
        "polar_radial_jitter": cfg.polar_radial_jitter,
        "polar_angular_jitter": cfg.polar_angular_jitter,
        "num_points": cfg.num_points,
        "spacing": 0.30,
        "n_max": cfg.N_max,
        "relax_iters": cfg.relax_iters,
        "relax_sep_relax": cfg.relax_sep_relax,
        "relax_spc_relax": cfg.relax_spc_relax,
        "relax_bend_relax": cfg.relax_bend_relax,
        "relax_margin": cfg.relax_margin,
        "relax_sep_every": cfg.relax_sep_every,
        "relax_sep_cache_slots": cfg.relax_sep_cache_slots,
        "relax_sep_cache_skin": cfg.relax_sep_cache_skin,
        "grid_n": 4,
        "seed": 0,
        "batch_size": 2048,
    }


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
    # Phase-1 generator selector (registered name); absent -> config default ("bezier").
    if p.get("generator") is not None:
        kw["generator"] = str(p["generator"])
    if p.get("polar_num_knots") is not None:
        kw["polar_num_knots"] = int(p["polar_num_knots"])
    if p.get("polar_radial_jitter") is not None:
        kw["polar_radial_jitter"] = float(p["polar_radial_jitter"])
    if p.get("polar_angular_jitter") is not None:
        kw["polar_angular_jitter"] = float(p["polar_angular_jitter"])
    # PBD separation broadphase/narrowphase knobs; absent -> config defaults.
    if p.get("relax_sep_every") is not None:
        kw["relax_sep_every"] = int(p["relax_sep_every"])
    if p.get("relax_sep_cache_slots") is not None:
        kw["relax_sep_cache_slots"] = int(p["relax_sep_cache_slots"])
    if p.get("relax_sep_cache_skin") is not None:
        kw["relax_sep_cache_skin"] = float(p["relax_sep_cache_skin"])
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
    rng = PerEnvSeededRNG(seeds=int(p["seed"]), num_envs=E, device=DEVICE)
    gen = TrackGenerator(cfg, rng)
    return gen.generate(E)


def _track_num_envs(track) -> int:
    """Return E (number of envs) from a Track regardless of whether fields are wp.array or Tensor."""
    v = track.valid
    if isinstance(v, torch.Tensor):
        return v.shape[0]
    # wp.array: valid is [E] int32/bool
    return wp.to_torch(v).shape[0]


def render_page(track, page: int, grid_n: int):
    """Draw a grid_n×grid_n window of the cached Track starting at page*grid_n**2.

    Returns a matplotlib Figure with grid_n**2 axes. Cells beyond the batch are left blank.
    """
    E = _track_num_envs(track)
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


def _to_torch_track(track):
    """Convert a wp.array Track to plain torch tensors for stats computation.

    Returns a namespace with:
      valid:  [E] bool
      count:  [E] int32
      length: [E] float32
      outer:  [E, N_max, 2] float32
      center: [E, N_max, 2] float32
    Handles both wp.array (new) and torch.Tensor (oracle/legacy) tracks.
    """
    import types as _types
    ns = _types.SimpleNamespace()
    if isinstance(track.valid, torch.Tensor):
        ns.valid = track.valid.bool()
        ns.count = track.count
        ns.length = track.length
        ns.outer = track.outer
        ns.center = track.center
        return ns
    # wp.array: valid/count/length are [E]; center/outer are flat [E*N_max] vec2f.
    ns.valid = wp.to_torch(track.valid).bool()
    ns.count = wp.to_torch(track.count)
    ns.length = wp.to_torch(track.length)
    E = ns.valid.shape[0]
    N_max = track.center.shape[0] // E
    ns.outer = wp.to_torch(track.outer).view(E, N_max, 2)
    ns.center = wp.to_torch(track.center).view(E, N_max, 2)
    return ns


def _stats(track) -> dict:
    """Aggregate readout over the batch (means taken over valid tracks).

    Output is always constant_spacing: ``count`` is the per-track real-point count and
    VARIES per env, so the reported ``count`` is the mean over valid tracks.
    """
    t = _to_torch_track(track)
    valid = t.valid
    n = int(valid.sum())
    if n == 0:
        return {"yield": 0.0, "n_valid": 0, "mean_len": float("nan"),
                "mean_thickness": float("nan"), "count": 0.0}
    # half-width from the first REAL point of each env (index 0 is always real / non-NaN).
    hw = float(torch.linalg.norm(t.outer[:, 0] - t.center[:, 0], dim=-1).median())
    cnt = t.count.clamp_min(1)
    band = (2.0 * hw / (t.length / cnt.float()).clamp_min(1e-9)).round().to(torch.int32).clamp_min(1)
    # thickness: call the kernel in-place via wp.array (_thickness_k from warp_pipeline)
    from track_gen._src import warp_pipeline as _wpl
    E, n_max, _ = t.center.shape
    dev = str(t.center.device)
    pf = wp.from_torch(t.center.reshape(E * n_max, 2).contiguous(), dtype=wp.vec2f)
    band_wp = wp.from_torch(band.contiguous(), dtype=wp.int32)
    cnt_wp = wp.from_torch(t.count.to(torch.int32).contiguous(), dtype=wp.int32)
    out_wp = wp.zeros(E, dtype=wp.float32, device=dev)
    wp.launch(_wpl._thickness_k, dim=E, inputs=[pf, band_wp, n_max, cnt_wp, out_wp], device=dev)
    if "cuda" in dev:
        wp.synchronize()
    th = wp.to_torch(out_wp)
    return {
        "yield": float(valid.float().mean()),
        "n_valid": n,
        "mean_len": float(t.length[valid].mean()),
        "mean_thickness": float(th[valid].mean()),
        "count": float(t.count[valid].float().mean()),
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
    keys = ["generator", "half_width", "scale", "min_num_points", "max_num_points", "rad", "edgy",
            "handle_clamp_frac", "polar_num_knots", "polar_radial_jitter",
            "polar_angular_jitter", "spacing", "n_max", "relax_iters",
            "relax_sep_relax", "relax_spc_relax", "relax_bend_relax", "relax_margin",
            "relax_sep_every", "relax_sep_cache_slots", "relax_sep_cache_skin",
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

    defaults = default_params()

    with gr.Blocks(title="Track-gen parameter explorer") as app:
        gr.Markdown("## Track-gen parameter explorer")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Phase-1 generator")
                available_generators = generator_registry.available()
                generator_default = defaults["generator"] if defaults["generator"] in available_generators else "bezier"
                generator = gr.Dropdown(available_generators, value=generator_default,
                                        label="generator method")
                gr.Markdown("### Regime")
                half_width = gr.Slider(0.05, 1.0, value=defaults["half_width"], step=0.01, label="half_width (m)")
                scale = gr.Slider(1.0, 20.0, value=defaults["scale"], step=0.5, label="scale (box)")
                gr.Markdown("### Bezier / hull controls")
                min_np = gr.Slider(5, 20, value=defaults["min_num_points"], step=1, label="min corners")
                max_np = gr.Slider(5, 20, value=defaults["max_num_points"], step=1, label="max corners")
                rad = gr.Slider(0.0, 0.6, value=defaults["rad"], step=0.01, label="rad (roundness)")
                edgy = gr.Slider(0.0, 1.0, value=defaults["edgy"], step=0.05, label="edgy")
                handle_clamp = gr.Slider(0.0, 1.0, value=defaults["handle_clamp_frac"], step=0.01,
                                         label="handle_clamp_frac (overshoot<->roundness)")
                gr.Markdown("### Polar knot spline")
                polar_knots = gr.Slider(4, 24, value=defaults["polar_num_knots"], step=1, label="polar knots")
                polar_radial = gr.Slider(0.0, 0.85, value=defaults["polar_radial_jitter"], step=0.01,
                                         label="polar radial jitter")
                polar_angular = gr.Slider(0.0, 0.45, value=defaults["polar_angular_jitter"], step=0.01,
                                          label="polar angular jitter")
                gr.Markdown("### Resolution (constant-spacing)")
                spacing = gr.Slider(0.1, 1.0, value=defaults["spacing"], step=0.02, label="spacing (m)")
                n_max = gr.Slider(128, 512, value=defaults["n_max"], step=8, label="N_max")
                gr.Markdown("### Relaxation")
                relax_iters = gr.Slider(0, 600, value=defaults["relax_iters"], step=10, label="relax_iters")
                sep = gr.Slider(0.0, 2.0, value=defaults["relax_sep_relax"], step=0.1, label="sep factor")
                spc = gr.Slider(0.0, 2.0, value=defaults["relax_spc_relax"], step=0.1, label="spc factor")
                bend = gr.Slider(0.0, 2.0, value=defaults["relax_bend_relax"], step=0.1, label="bend factor")
                margin = gr.Slider(0.0, 0.5, value=defaults["relax_margin"], step=0.01, label="relax_margin")
                gr.Markdown("### PBD separation (broadphase / narrowphase)")
                sep_every = gr.Slider(1, 150, value=defaults["relax_sep_every"], step=1,
                                      label="K — broadphase refresh interval (sweeps)")
                sep_slots = gr.Slider(0, 64, value=defaults["relax_sep_cache_slots"], step=1,
                                      label="cache slots — broadphase candidates/bead (0 = exact dense)")
                sep_skin = gr.Slider(0.0, 2.0, value=defaults["relax_sep_cache_skin"], step=0.1,
                                     label="cache skin — broadphase margin (× target)")
                gr.Markdown("### Batch")
                grid_n = gr.Dropdown([3, 4, 5, 6], value=defaults["grid_n"], label="grid (n x n)")
                seed = gr.Number(value=defaults["seed"], precision=0, label="seed")
                batch_size = gr.Dropdown([256, 1024, 2048, 4096, 8192], value=defaults["batch_size"], label="batch size")
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

        controls = [generator, half_width, scale, min_np, max_np, rad, edgy, handle_clamp,
                    polar_knots, polar_radial, polar_angular, spacing, n_max, relax_iters, sep, spc, bend, margin,
                    sep_every, sep_slots, sep_skin, grid_n, seed, batch_size]

        def _generate(*vals):
            p = _collect(*vals)
            gn = int(p["grid_n"])
            try:
                track = generate_batch(p)
                fig = render_page(track, 0, gn)
                st = _stats(track)
                lbl = f"page 1/{n_pages(_track_num_envs(track), gn)}"
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
            np_ = n_pages(_track_num_envs(track), int(gn))
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
