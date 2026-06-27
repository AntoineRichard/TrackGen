#!/usr/bin/env python3
"""Interactive Gradio explorer for track-generation parameters.

A thin Gradio shell over a pure core (`build_config` + `render_grid`) that drives the real
pure-Warp pipeline (`TrackGenerator.generate`) and renders a grid of
tracks + yield/quality stats. Launch:  `.venv/bin/python -m viz.param_explorer`
(needs the `ui` extra: `pip install -e ".[ui]"`).
"""
from __future__ import annotations

import logging
import math
import os
import sys
import traceback

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot

import matplotlib.pyplot as plt
import torch

import warp as wp

from track_gen import GateGenConfig, GateGenerator
from track_gen._src.types import TrackGenConfig
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src import gate_generator_registry, generator_registry, warp_gate
from viz.plot_tracks import draw_track

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger("param_explorer")


def _ui_error_message(exc: Exception, context: str) -> str:
    """Log a render failure and return the message to show in the UI.

    The render handlers must keep the UI alive, but swallowing the bare ``str(exc)``
    loses the traceback and makes a genuine bug (kernel launch failure, shape
    mismatch after a refactor, renamed field) indistinguishable from a user typo.
    A ``ValueError`` is a rejected config the user can act on, so its message is
    enough; anything else is unexpected and gets its full stack logged to the
    console so it stays debuggable.
    """
    if isinstance(exc, ValueError):
        logger.warning("%s rejected input: %s", context, exc)
        return str(exc)
    logger.error("%s crashed:\n%s", context, traceback.format_exc())
    return f"internal error: {exc} (see console log)"

GATE_CONTROL_KEYS = [
    "gate_generator", "gate_ordering", "gate_width", "gate_radius", "gate_solve_iters",
    "gate_show_raw", "gate_scale", "gate_min_num_points", "gate_max_num_points",
    "gate_min_point_distance",
    "gate_polar_num_knots", "gate_polar_radial_jitter", "gate_polar_angular_jitter",
    "gate_voronoi_num_sites", "gate_voronoi_site_layout", "gate_voronoi_control_points",
    "gate_voronoi_radial_variation", "gate_voronoi_angular_jitter",
    "gate_checkpoint_count", "gate_checkpoint_radius_min_frac",
    "gate_checkpoint_angle_jitter", "gate_grid_n", "gate_seed", "gate_batch_size",
]


def default_params() -> dict:
    """Return the Gradio explorer defaults as a params dict accepted by build_config."""
    cfg = TrackGenConfig()
    return {
        "generator": "polar",
        "half_width": 0.5,
        "scale": 10.0,
        "min_num_points": cfg.min_num_points,
        "max_num_points": cfg.max_num_points,
        "min_point_distance": cfg.min_point_distance,
        "num_points_per_segment": cfg.num_points_per_segment,
        "hull_displacement": cfg.hull_displacement,
        "rad": cfg.rad,
        "edgy": cfg.edgy,
        "handle_clamp_frac": cfg.handle_clamp_frac,
        "polar_num_knots": cfg.polar_num_knots,
        "polar_radial_jitter": cfg.polar_radial_jitter,
        "polar_angular_jitter": cfg.polar_angular_jitter,
        "voronoi_num_sites": cfg.voronoi_num_sites,
        "voronoi_site_layout": cfg.voronoi_site_layout,
        "voronoi_control_points": cfg.voronoi_control_points,
        "voronoi_radial_variation": cfg.voronoi_radial_variation,
        "voronoi_angular_jitter": cfg.voronoi_angular_jitter,
        "checkpoint_count": cfg.checkpoint_count,
        "checkpoint_radius_min_frac": cfg.checkpoint_radius_min_frac,
        "checkpoint_angle_jitter": cfg.checkpoint_angle_jitter,
        "checkpoint_turn_rate": cfg.checkpoint_turn_rate,
        "checkpoint_steer_gain": cfg.checkpoint_steer_gain,
        "checkpoint_lookahead_frac": cfg.checkpoint_lookahead_frac,
        "checkpoint_best_of_k": cfg.checkpoint_best_of_k,
        "checkpoint_clip_fallback": cfg.checkpoint_clip_fallback,
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
    if p.get("num_points_per_segment") is not None:
        kw["num_points_per_segment"] = int(p["num_points_per_segment"])
    if p.get("spacing") is not None:
        kw["spacing"] = float(p["spacing"])
    if p.get("min_point_distance") is not None:
        kw["min_point_distance"] = float(p["min_point_distance"])
    if p.get("hull_displacement") is not None:
        kw["hull_displacement"] = float(p["hull_displacement"])
    # Phase-1 generator selector (registered name); absent -> config default ("bezier").
    if p.get("generator") is not None:
        kw["generator"] = str(p["generator"])
    if p.get("polar_num_knots") is not None:
        kw["polar_num_knots"] = int(p["polar_num_knots"])
    if p.get("polar_radial_jitter") is not None:
        kw["polar_radial_jitter"] = float(p["polar_radial_jitter"])
    if p.get("polar_angular_jitter") is not None:
        kw["polar_angular_jitter"] = float(p["polar_angular_jitter"])
    if p.get("voronoi_num_sites") is not None:
        kw["voronoi_num_sites"] = int(p["voronoi_num_sites"])
    if p.get("voronoi_site_layout") is not None:
        kw["voronoi_site_layout"] = str(p["voronoi_site_layout"])
    if p.get("voronoi_control_points") is not None:
        kw["voronoi_control_points"] = int(p["voronoi_control_points"])
    if p.get("voronoi_radial_variation") is not None:
        kw["voronoi_radial_variation"] = float(p["voronoi_radial_variation"])
    if p.get("voronoi_angular_jitter") is not None:
        kw["voronoi_angular_jitter"] = float(p["voronoi_angular_jitter"])
    # Checkpoint steering knobs; absent -> config defaults.
    if p.get("checkpoint_count") is not None:
        kw["checkpoint_count"] = int(p["checkpoint_count"])
    if p.get("checkpoint_radius_min_frac") is not None:
        kw["checkpoint_radius_min_frac"] = float(p["checkpoint_radius_min_frac"])
    if p.get("checkpoint_angle_jitter") is not None:
        kw["checkpoint_angle_jitter"] = float(p["checkpoint_angle_jitter"])
    if p.get("checkpoint_turn_rate") is not None:
        kw["checkpoint_turn_rate"] = float(p["checkpoint_turn_rate"])
    if p.get("checkpoint_steer_gain") is not None:
        kw["checkpoint_steer_gain"] = float(p["checkpoint_steer_gain"])
    if p.get("checkpoint_lookahead_frac") is not None:
        kw["checkpoint_lookahead_frac"] = float(p["checkpoint_lookahead_frac"])
    if p.get("checkpoint_best_of_k") is not None:
        kw["checkpoint_best_of_k"] = int(p["checkpoint_best_of_k"])
    if p.get("checkpoint_clip_fallback") is not None:
        kw["checkpoint_clip_fallback"] = bool(p["checkpoint_clip_fallback"])
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


def _estimate_half_width(outer, center, valid) -> float:
    """Median half-width over VALID envs only. Index 0 is real (count>=1) for valid envs;
    including invalid (count==0, fully NaN-padded) envs would NaN-poison the median."""
    d = torch.linalg.norm(outer[valid, 0] - center[valid, 0], dim=-1)
    return float(d.median())


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
    # half-width from the first REAL point of each VALID env (invalid envs have count==0
    # and are fully NaN-padded, which would NaN-poison the median). n > 0 here, so non-empty.
    hw = _estimate_half_width(t.outer, t.center, valid)
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
    except Exception as exc:  # keep the UI alive; full traceback is logged
        msg = _ui_error_message(exc, "track render")
        fig = plt.figure(figsize=(5, 3))
        fig.text(0.5, 0.5, f"error: {msg}", ha="center", va="center", fontsize=9, color="red", wrap=True)
        return fig, {"error": msg, "yield": 0.0, "n_valid": 0, "mean_len": float("nan"),
                     "mean_thickness": float("nan"), "count": 0}


def default_gate_params() -> dict:
    cfg = GateGenConfig()
    return {
        "gate_generator": cfg.generator,
        "gate_ordering": cfg.gate_ordering,
        "gate_width": 0.05,
        "gate_radius": cfg.gate_radius,
        "gate_solve_iters": cfg.gate_solve_iters,
        "gate_show_raw": False,
        "gate_scale": cfg.scale,
        "gate_min_num_points": cfg.min_num_points,
        "gate_max_num_points": cfg.max_num_points,
        "gate_min_point_distance": cfg.min_point_distance,
        "gate_polar_num_knots": cfg.polar_num_knots,
        "gate_polar_radial_jitter": cfg.polar_radial_jitter,
        "gate_polar_angular_jitter": cfg.polar_angular_jitter,
        "gate_voronoi_num_sites": cfg.voronoi_num_sites,
        "gate_voronoi_site_layout": cfg.voronoi_site_layout,
        "gate_voronoi_control_points": cfg.voronoi_control_points,
        "gate_voronoi_radial_variation": cfg.voronoi_radial_variation,
        "gate_voronoi_angular_jitter": cfg.voronoi_angular_jitter,
        "gate_checkpoint_count": cfg.checkpoint_count,
        "gate_checkpoint_radius_min_frac": cfg.checkpoint_radius_min_frac,
        "gate_checkpoint_angle_jitter": cfg.checkpoint_angle_jitter,
        "gate_grid_n": 4,
        "gate_seed": 0,
        "gate_batch_size": 256,
    }


def gate_supported_orderings(generator: str) -> list[str]:
    try:
        supported = gate_generator_registry.get(str(generator)).supported_orderings
    except ValueError as exc:
        logger.warning("gate ordering lookup failed for %r: %s", generator, exc)
        supported = frozenset({"ccw"})
    ordered = [name for name in ("ccw", "raw", "random_pairs") if name in supported]
    return ordered or ["ccw"]


def gate_visible_sections(generator: str) -> dict[str, bool]:
    name = str(generator)
    return {
        "point": name in {"bezier", "hull"},
        "polar": name == "polar",
        "voronoi": name == "voronoi",
        "checkpoint": name == "checkpoint",
    }


def track_visible_sections(generator: str) -> dict[str, bool]:
    name = str(generator)
    point = name in {"bezier", "hull"}
    return {
        "sampling": point,
        "smoothing": name in {"bezier", "hull", "polar", "voronoi"},
        "bezier": name == "bezier",
        "hull": name == "hull",
        "polar": name == "polar",
        "voronoi": name == "voronoi",
        "checkpoint": name == "checkpoint",
    }


# Ordered (section, component_count) segments for the Tracks tab's generator-specific
# controls. Single source of truth shared by the build-time visibility flags, the
# generator.change handler, and the build-time length guard. Each section contributes
# its header markdown plus its controls, in the SAME order as ``track_mode_outputs``.
TRACK_MODE_SECTION_SIZES = (
    ("sampling", 4), ("smoothing", 2), ("bezier", 4), ("hull", 2),
    ("polar", 4), ("voronoi", 6), ("checkpoint", 9),
)


def track_mode_visibility(generator: str) -> list[bool]:
    """Per-output visibility flags for the track generator-specific components,
    ordered to match ``track_mode_outputs``."""
    sections = track_visible_sections(generator)
    flags: list[bool] = []
    for key, count in TRACK_MODE_SECTION_SIZES:
        flags.extend([sections[key]] * count)
    return flags


def build_gate_config(p: dict) -> GateGenConfig:
    generator = str(p["gate_generator"])
    min_points = min(int(p["gate_min_num_points"]), int(p["gate_max_num_points"]))
    max_points = max(int(p["gate_min_num_points"]), int(p["gate_max_num_points"]))
    polar_knots = max(4, int(p["gate_polar_num_knots"]))
    vor_control = max(3, int(p["gate_voronoi_control_points"]))
    checkpoint_count = max(3, int(p["gate_checkpoint_count"]))
    if generator in {"bezier", "hull"}:
        min_gates = max(2, min_points)
        max_gates = max(min_gates, max_points)
    elif generator == "polar":
        min_gates = polar_knots
        max_gates = polar_knots
    elif generator == "voronoi":
        min_gates = vor_control
        max_gates = vor_control
    elif generator == "checkpoint":
        min_gates = checkpoint_count
        max_gates = checkpoint_count
    else:
        min_gates = 2
        max_gates = max(2, max_points)
    radius = float(p["gate_radius"])
    supported_orderings = gate_supported_orderings(generator)
    ordering = str(p["gate_ordering"])
    if ordering not in supported_orderings:
        ordering = supported_orderings[0]
    return GateGenConfig(
        generator=generator,
        gate_ordering=ordering,
        num_envs=int(p.get("gate_batch_size", int(p["gate_grid_n"]) ** 2)),
        min_gates=min_gates,
        max_gates=max_gates,
        gate_width=float(p["gate_width"]),
        gate_radius=max(0.0, radius),
        gate_solve_iters=(0 if bool(p.get("gate_show_raw", False)) else int(p["gate_solve_iters"])),
        scale=float(p["gate_scale"]),
        min_num_points=min_points,
        max_num_points=max_points,
        min_point_distance=float(p["gate_min_point_distance"]),
        polar_num_knots=polar_knots,
        polar_radial_jitter=float(p["gate_polar_radial_jitter"]),
        polar_angular_jitter=float(p["gate_polar_angular_jitter"]),
        voronoi_num_sites=max(vor_control, int(p["gate_voronoi_num_sites"])),
        voronoi_site_layout=str(p["gate_voronoi_site_layout"]),
        voronoi_control_points=vor_control,
        voronoi_radial_variation=float(p["gate_voronoi_radial_variation"]),
        voronoi_angular_jitter=float(p["gate_voronoi_angular_jitter"]),
        checkpoint_count=checkpoint_count,
        checkpoint_radius_min_frac=float(p["gate_checkpoint_radius_min_frac"]),
        checkpoint_angle_jitter=float(p["gate_checkpoint_angle_jitter"]),
        device=DEVICE,
    )


def _gate_center_target(cfg: GateGenConfig) -> float:
    return warp_gate._gate_center_distance(cfg)


def generate_gate_batch(p: dict):
    wp.init()
    cfg = build_gate_config(p)
    rng = PerEnvSeededRNG(seeds=int(p["gate_seed"]), num_envs=cfg.num_envs, device=DEVICE)
    gen = GateGenerator(cfg, rng)
    return gen.generate(cfg.num_envs), cfg


def _gate_tensors(gates):
    import types as _types
    ns = _types.SimpleNamespace()
    ns.valid = wp.to_torch(gates.valid).bool()
    ns.count = wp.to_torch(gates.count)
    E = ns.valid.shape[0]
    G = gates.position.shape[0] // E
    ns.position = wp.to_torch(gates.position).view(E, G, 2)
    ns.tangent = wp.to_torch(gates.tangent).view(E, G, 2)
    ns.left = wp.to_torch(gates.left).view(E, G, 2)
    ns.right = wp.to_torch(gates.right).view(E, G, 2)
    return ns


def _nice_scale_length(span: float) -> float:
    if span <= 0.0 or not math.isfinite(span):
        return 1.0
    raw = span / 4.0
    exp = math.floor(math.log10(raw))
    base = raw / (10.0 ** exp)
    if base < 2.0:
        nice = 1.0
    elif base < 5.0:
        nice = 2.0
    else:
        nice = 5.0
    return nice * (10.0 ** exp)


def _draw_gate_sequence(ax, gt, cfg: GateGenConfig, e: int) -> None:
    c = int(gt.count[e].item())
    valid = bool(gt.valid[e].item())
    pos = gt.position[e, :c].detach().cpu()
    tangent = gt.tangent[e, :c].detach().cpu()
    left = gt.left[e, :c].detach().cpu()
    right = gt.right[e, :c].detach().cpu()
    finite = torch.isfinite(pos).all(dim=-1)
    pos = pos[finite]
    tangent = tangent[finite]
    left = left[finite]
    right = right[finite]

    if pos.shape[0] >= 2:
        path = torch.cat([pos, pos[:1]], dim=0) if pos.shape[0] > 2 else pos
        ax.plot(path[:, 0], path[:, 1], color="0.25", lw=0.7, ls="--", alpha=0.55, zorder=1)
    if pos.shape[0] > 0:
        ax.scatter(pos[:, 0], pos[:, 1], s=14, color="#111827", zorder=4)
        if tangent.shape[0] == pos.shape[0]:
            ax.quiver(pos[:, 0], pos[:, 1], tangent[:, 0], tangent[:, 1], angles="xy",
                      scale_units="xy", scale=14, width=0.004, color="#f97316", alpha=0.75,
                      zorder=3)
    if left.shape[0] == right.shape[0] == pos.shape[0]:
        for li, ri in zip(left, right):
            ax.plot([li[0], ri[0]], [li[1], ri[1]], color="#2563eb", lw=1.2, zorder=2)
    if float(cfg.gate_radius) > 0.0:
        radius = float(cfg.gate_radius)
        for pnt in pos:
            ax.add_patch(plt.Circle((float(pnt[0]), float(pnt[1])), radius, fill=False,
                                    color="#64748b", lw=0.7, alpha=0.75, zorder=0))

    if pos.shape[0] > 0:
        all_pts = pos
        if left.shape[0] == right.shape[0] == pos.shape[0]:
            all_pts = torch.cat([pos, left, right], dim=0)
        xmin, ymin = all_pts.min(dim=0).values.tolist()
        xmax, ymax = all_pts.max(dim=0).values.tolist()
        span = max(xmax - xmin, ymax - ymin, 1.0e-3)
        pad = 0.22 * span + max(float(cfg.gate_radius or 0.0), float(cfg.gate_width) * 0.5)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        ax.set_xlim(cx - 0.5 * span - pad, cx + 0.5 * span + pad)
        ax.set_ylim(cy - 0.5 * span - pad, cy + 0.5 * span + pad)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bar = _nice_scale_length(max(x1 - x0, y1 - y0))
        sx = x0 + 0.07 * (x1 - x0)
        sy = y0 + 0.07 * (y1 - y0)
        ax.plot([sx, sx + bar], [sy, sy], color="#111827", lw=1.1, zorder=6)
        ax.text(sx, sy, f" {bar:g} m", fontsize=4.8, va="bottom", ha="left",
                color="#111827", zorder=6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.6)
    ax.set_title(f"env {e} · n={c}{'' if valid else ' · INVALID'}",
                 fontsize=6.5, color=("#111827" if valid else "#dc2626"), pad=2)


def _gate_num_envs(gates) -> int:
    return wp.to_torch(gates.valid).shape[0]


def render_gate_page(gates, cfg: GateGenConfig, page: int, grid_n: int):
    E = _gate_num_envs(gates)
    np_ = n_pages(E, grid_n)
    start = page * grid_n * grid_n
    gt = _gate_tensors(gates)
    fig, axes = plt.subplots(grid_n, grid_n, figsize=(2.1 * grid_n, 2.1 * grid_n))
    axes = axes.flatten() if grid_n > 1 else [axes]
    for k, ax in enumerate(axes):
        idx = start + k
        if idx < E:
            _draw_gate_sequence(ax, gt, cfg, idx)
        else:
            ax.axis("off")
    fig.suptitle(f"gate sequences {grid_n}x{grid_n}  ·  page {page + 1}/{np_}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _gate_stats(gates, cfg: GateGenConfig) -> dict:
    gt = _gate_tensors(gates)
    valid = gt.valid
    n_valid = int(valid.sum().item())
    # Vectorized min pairwise center distance across the whole batch: one cdist over
    # the [E, G, G] gate grid with non-finite (NaN-padded past count) slots and the
    # self-pairs masked to +inf, then a single host sync. Avoids the per-env
    # .item()/pdist loop, which issued thousands of device syncs at large batch sizes.
    pos = gt.position
    finite = torch.isfinite(pos).all(dim=-1)
    safe = torch.where(finite.unsqueeze(-1), pos, torch.zeros_like(pos))
    dmat = torch.cdist(safe, safe)
    pair_ok = finite.unsqueeze(1) & finite.unsqueeze(2)
    pair_ok = pair_ok & ~torch.eye(pos.shape[1], dtype=torch.bool, device=pos.device)
    dmat = torch.where(pair_ok, dmat, torch.full_like(dmat, float("inf")))
    per_env_min = dmat.flatten(1).min(dim=1).values
    per_env_min = per_env_min[torch.isfinite(per_env_min)]
    min_dist = float(per_env_min.min().item()) if per_env_min.numel() else float("nan")
    counts = gt.count.float()
    return {
        "yield": float(valid.float().mean().item()),
        "n_valid": n_valid,
        "n_invalid": int((~valid).sum().item()),
        "mean_count": float(counts[valid].mean().item()) if n_valid else 0.0,
        "min_center_distance": min_dist,
        "target_center_distance": _gate_center_target(cfg),
    }


def render_gate_grid(p: dict):
    try:
        gates, cfg = generate_gate_batch(p)
        fig = render_gate_page(gates, cfg, 0, int(p["gate_grid_n"]))
        return fig, _gate_stats(gates, cfg)
    except Exception as exc:  # keep the UI alive; full traceback is logged
        msg = _ui_error_message(exc, "gate render")
        fig = plt.figure(figsize=(5, 3))
        fig.text(0.5, 0.5, f"error: {msg}", ha="center", va="center", fontsize=9,
                 color="red", wrap=True)
        return fig, {"error": msg, "yield": 0.0, "n_valid": 0, "n_invalid": 0,
                     "mean_count": 0.0, "min_center_distance": float("nan"),
                     "target_center_distance": float("nan")}


def _collect_gate(*vals) -> dict:
    if len(vals) != len(GATE_CONTROL_KEYS):
        raise ValueError(
            f"expected {len(GATE_CONTROL_KEYS)} gate controls, got {len(vals)}"
        )
    return dict(zip(GATE_CONTROL_KEYS, vals))


def _gate_stats_md(st: dict) -> str:
    if "error" in st:
        return f"**error:** {st['error']}"
    min_dist = st["min_center_distance"]
    min_txt = "nan" if math.isnan(min_dist) else f"{min_dist:.3f}"
    return (f"**valid yield: {st['yield'] * 100:.0f}%**  ·  {st['n_valid']} valid  ·  "
            f"{st['n_invalid']} invalid  ·  mean gates {st['mean_count']:.1f}  ·  "
            f"min center distance {min_txt} / target {st['target_center_distance']:.3f}")


def _collect(*vals) -> dict:
    keys = ["generator", "half_width", "scale", "min_num_points", "max_num_points",
            "min_point_distance", "num_points_per_segment", "hull_displacement",
            "rad", "edgy", "handle_clamp_frac", "polar_num_knots", "polar_radial_jitter",
            "polar_angular_jitter", "voronoi_num_sites", "voronoi_site_layout",
            "voronoi_control_points", "voronoi_radial_variation", "voronoi_angular_jitter",
            "checkpoint_count", "checkpoint_radius_min_frac", "checkpoint_angle_jitter",
            "checkpoint_turn_rate", "checkpoint_steer_gain", "checkpoint_lookahead_frac",
            "checkpoint_best_of_k", "checkpoint_clip_fallback",
            "spacing", "n_max", "relax_iters",
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
    gate_defaults = default_gate_params()

    with gr.Blocks(title="Track-gen parameter explorer") as app:
        gr.Markdown("## Track-gen parameter explorer")
        with gr.Tabs():
            with gr.Tab("Tracks"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Phase-1 generator")
                        available_generators = generator_registry.available()
                        generator_default = defaults["generator"] if defaults["generator"] in available_generators else "bezier"
                        generator = gr.Dropdown(available_generators, value=generator_default,
                                                label="generator method")
                        track_mode_visible = track_visible_sections(generator_default)
                        gr.Markdown("### Regime")
                        half_width = gr.Slider(0.05, 1.0, value=defaults["half_width"], step=0.01, label="half_width (m)")
                        scale = gr.Slider(1.0, 20.0, value=defaults["scale"], step=0.5, label="scale (box)")
                        sampling_md = gr.Markdown("### Sampling (point-family)",
                                                  visible=track_mode_visible["sampling"])
                        min_np = gr.Slider(5, 20, value=defaults["min_num_points"], step=1,
                                           label="min corners", visible=track_mode_visible["sampling"])
                        max_np = gr.Slider(5, 20, value=defaults["max_num_points"], step=1,
                                           label="max corners", visible=track_mode_visible["sampling"])
                        min_dist = gr.Slider(0.02, 0.20, value=defaults["min_point_distance"], step=0.005,
                                             label="min_point_distance (sampling spread)",
                                             visible=track_mode_visible["sampling"])
                        smoothing_md = gr.Markdown("### Curve smoothing",
                                                   visible=track_mode_visible["smoothing"])
                        samples_per_seg = gr.Slider(8, 60, value=defaults["num_points_per_segment"], step=1,
                                                    label="num_points_per_segment (generator smoothing samples)",
                                                    visible=track_mode_visible["smoothing"])
                        bezier_md = gr.Markdown("### Bezier controls", visible=track_mode_visible["bezier"])
                        rad = gr.Slider(0.0, 0.6, value=defaults["rad"], step=0.01, label="rad (roundness)",
                                        visible=track_mode_visible["bezier"])
                        edgy = gr.Slider(0.0, 1.0, value=defaults["edgy"], step=0.05, label="edgy",
                                         visible=track_mode_visible["bezier"])
                        handle_clamp = gr.Slider(0.0, 1.0, value=defaults["handle_clamp_frac"], step=0.01,
                                                 label="handle_clamp_frac (overshoot<->roundness)",
                                                 visible=track_mode_visible["bezier"])
                        hull_md = gr.Markdown("### Hull controls", visible=track_mode_visible["hull"])
                        hull_disp = gr.Slider(0.0, 0.8, value=defaults["hull_displacement"], step=0.01,
                                              label="hull_displacement (hull midpoint displacement)",
                                              visible=track_mode_visible["hull"])
                        polar_md = gr.Markdown("### Polar knot spline", visible=track_mode_visible["polar"])
                        polar_knots = gr.Slider(4, 24, value=defaults["polar_num_knots"], step=1,
                                                label="polar knots", visible=track_mode_visible["polar"])
                        polar_radial = gr.Slider(0.0, 0.85, value=defaults["polar_radial_jitter"], step=0.01,
                                                 label="polar radial jitter", visible=track_mode_visible["polar"])
                        polar_angular = gr.Slider(0.0, 0.45, value=defaults["polar_angular_jitter"], step=0.01,
                                                  label="polar angular jitter", visible=track_mode_visible["polar"])
                        vor_md = gr.Markdown("### Voronoi graph cycle", visible=track_mode_visible["voronoi"])
                        vor_sites = gr.Slider(32, 512, value=defaults["voronoi_num_sites"], step=16,
                                              label="voronoi sites", visible=track_mode_visible["voronoi"])
                        vor_layout = gr.Dropdown(["void_ring", "ring", "clustered", "mixed"],
                                                 value=defaults["voronoi_site_layout"],
                                                 label="voronoi site layout", visible=track_mode_visible["voronoi"])
                        vor_control = gr.Slider(6, 32, value=defaults["voronoi_control_points"], step=1,
                                                label="voronoi control points", visible=track_mode_visible["voronoi"])
                        vor_radial = gr.Slider(0.0, 0.85, value=defaults["voronoi_radial_variation"], step=0.01,
                                               label="voronoi radial variation", visible=track_mode_visible["voronoi"])
                        vor_angular = gr.Slider(0.0, 0.25, value=defaults["voronoi_angular_jitter"], step=0.01,
                                                label="voronoi angular jitter", visible=track_mode_visible["voronoi"])
                        checkpoint_md = gr.Markdown("### Checkpoint steering", visible=track_mode_visible["checkpoint"])
                        checkpoint_count = gr.Slider(4, 24, value=defaults["checkpoint_count"], step=1,
                                                     label="checkpoint_count (radial waypoints)",
                                                     visible=track_mode_visible["checkpoint"])
                        checkpoint_radius_min_frac = gr.Slider(0.1, 0.9, value=defaults["checkpoint_radius_min_frac"],
                                                               step=0.01, label="checkpoint_radius_min_frac",
                                                               visible=track_mode_visible["checkpoint"])
                        checkpoint_angle_jitter = gr.Slider(0.0, 0.9, value=defaults["checkpoint_angle_jitter"],
                                                            step=0.01, label="checkpoint_angle_jitter",
                                                            visible=track_mode_visible["checkpoint"])
                        checkpoint_turn_rate = gr.Slider(0.1, 1.0, value=defaults["checkpoint_turn_rate"],
                                                         step=0.01, label="checkpoint_turn_rate",
                                                         visible=track_mode_visible["checkpoint"])
                        checkpoint_steer_gain = gr.Slider(0.1, 1.0, value=defaults["checkpoint_steer_gain"],
                                                          step=0.01, label="checkpoint_steer_gain",
                                                          visible=track_mode_visible["checkpoint"])
                        checkpoint_lookahead_frac = gr.Slider(0.05, 0.4, value=defaults["checkpoint_lookahead_frac"],
                                                              step=0.01, label="checkpoint_lookahead_frac",
                                                              visible=track_mode_visible["checkpoint"])
                        checkpoint_best_of_k = gr.Slider(1, 8, value=defaults["checkpoint_best_of_k"], step=1,
                                                         label="checkpoint_best_of_k (candidates)",
                                                         visible=track_mode_visible["checkpoint"])
                        checkpoint_clip_fallback = gr.Checkbox(value=defaults["checkpoint_clip_fallback"],
                                                               label="checkpoint_clip_fallback (single-crossing rescue)",
                                                               visible=track_mode_visible["checkpoint"])
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

                controls = [generator, half_width, scale, min_np, max_np, min_dist, samples_per_seg,
                            hull_disp, rad, edgy, handle_clamp, polar_knots, polar_radial,
                            polar_angular, vor_sites, vor_layout, vor_control, vor_radial, vor_angular,
                            checkpoint_count, checkpoint_radius_min_frac, checkpoint_angle_jitter,
                            checkpoint_turn_rate, checkpoint_steer_gain, checkpoint_lookahead_frac,
                            checkpoint_best_of_k, checkpoint_clip_fallback,
                            spacing, n_max, relax_iters, sep, spc, bend, margin,
                            sep_every, sep_slots, sep_skin, grid_n, seed, batch_size]

                track_mode_outputs = [
                    sampling_md, min_np, max_np, min_dist,
                    smoothing_md, samples_per_seg,
                    bezier_md, rad, edgy, handle_clamp,
                    hull_md, hull_disp,
                    polar_md, polar_knots, polar_radial, polar_angular,
                    vor_md, vor_sites, vor_layout, vor_control, vor_radial, vor_angular,
                    checkpoint_md, checkpoint_count, checkpoint_radius_min_frac,
                    checkpoint_angle_jitter, checkpoint_turn_rate, checkpoint_steer_gain,
                    checkpoint_lookahead_frac, checkpoint_best_of_k, checkpoint_clip_fallback,
                ]

                if len(track_mode_outputs) != len(track_mode_visibility(generator_default)):
                    raise RuntimeError(
                        "track_mode_outputs is out of sync with TRACK_MODE_SECTION_SIZES"
                    )

                def _track_mode_update(generator_name):
                    return [gr.update(visible=flag)
                            for flag in track_mode_visibility(generator_name)]

                generator.change(_track_mode_update, [generator], track_mode_outputs)

                def _generate(*vals):
                    p = _collect(*vals)
                    gn = int(p["grid_n"])
                    try:
                        track = generate_batch(p)
                        fig = render_page(track, 0, gn)
                        st = _stats(track)
                        lbl = f"page 1/{n_pages(_track_num_envs(track), gn)}"
                        return fig, _stats_md(st), track, 0, lbl
                    except Exception as exc:  # keep the UI alive; full traceback is logged
                        msg = _ui_error_message(exc, "track render")
                        err_fig = plt.figure(figsize=(5, 3))
                        err_fig.text(0.5, 0.5, f"error: {msg}", ha="center", va="center",
                                     fontsize=9, color="red", wrap=True)
                        err_st = {"error": msg, "yield": 0.0, "n_valid": 0,
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
            with gr.Tab("Gates"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Gate generator")
                        gate_available_generators = gate_generator_registry.available()
                        gate_generator_default = gate_defaults["gate_generator"] if gate_defaults["gate_generator"] in gate_available_generators else "bezier"
                        gate_mode_visible = gate_visible_sections(gate_generator_default)
                        gate_ordering_choices = gate_supported_orderings(gate_generator_default)
                        gate_ordering_default = gate_defaults["gate_ordering"]
                        if gate_ordering_default not in gate_ordering_choices:
                            gate_ordering_default = gate_ordering_choices[0]
                        gate_generator = gr.Dropdown(gate_available_generators, value=gate_generator_default,
                                                     label="generator method")
                        gate_ordering = gr.Dropdown(gate_ordering_choices,
                                                    value=gate_ordering_default, label="ordering")
                        gr.Markdown("### Gate layout")
                        gate_width = gr.Slider(0.0, 1.0, value=gate_defaults["gate_width"], step=0.01,
                                               label="gate_width [world units]")
                        gate_scale = gr.Slider(0.25, 20.0, value=gate_defaults["gate_scale"], step=0.25,
                                               label="scale [x]")
                        gr.Markdown("### Gate Collisions")
                        gr.Markdown("Center spacing target = 2 * gate_radius.")
                        gate_radius = gr.Slider(0.0, 10.0, value=gate_defaults["gate_radius"], step=0.005,
                                                label="gate_radius [world units]")
                        gate_solve_iters = gr.Slider(0, 64, value=gate_defaults["gate_solve_iters"], step=1,
                                                     label="gate_solve_iters")
                        gate_show_raw = gr.Checkbox(value=gate_defaults["gate_show_raw"],
                                                    label="show raw anchors (skip collisions)")
                        gate_point_md = gr.Markdown("### Sampling (point-family)", visible=gate_mode_visible["point"])
                        gate_point_note = gr.Markdown("For Bezier/Hull gates, sampled anchors become gate centers.",
                                                      visible=gate_mode_visible["point"])
                        gate_min_np = gr.Slider(2, 32, value=gate_defaults["gate_min_num_points"], step=1,
                                                label="min sampled anchors", visible=gate_mode_visible["point"])
                        gate_max_np = gr.Slider(2, 32, value=gate_defaults["gate_max_num_points"], step=1,
                                                label="max sampled anchors", visible=gate_mode_visible["point"])
                        gate_min_point_distance = gr.Slider(0.01, 0.30, value=gate_defaults["gate_min_point_distance"],
                                                            step=0.005, label="min_point_distance [pre-scale world units]",
                                                            visible=gate_mode_visible["point"])
                        gate_polar_md = gr.Markdown("### Polar controls", visible=gate_mode_visible["polar"])
                        gate_polar_knots = gr.Slider(4, 32, value=gate_defaults["gate_polar_num_knots"], step=1,
                                                     label="polar knots", visible=gate_mode_visible["polar"])
                        gate_polar_radial = gr.Slider(0.0, 0.85, value=gate_defaults["gate_polar_radial_jitter"],
                                                      step=0.01, label="polar radial jitter",
                                                      visible=gate_mode_visible["polar"])
                        gate_polar_angular = gr.Slider(0.0, 0.45, value=gate_defaults["gate_polar_angular_jitter"],
                                                       step=0.01, label="polar angular jitter",
                                                       visible=gate_mode_visible["polar"])
                        gate_vor_md = gr.Markdown("### Voronoi controls", visible=gate_mode_visible["voronoi"])
                        gate_vor_sites = gr.Slider(32, 512, value=gate_defaults["gate_voronoi_num_sites"], step=16,
                                                   label="voronoi sites", visible=gate_mode_visible["voronoi"])
                        gate_vor_layout = gr.Dropdown(["void_ring", "ring", "clustered", "mixed"],
                                                      value=gate_defaults["gate_voronoi_site_layout"],
                                                      label="voronoi site layout",
                                                      visible=gate_mode_visible["voronoi"])
                        gate_vor_control = gr.Slider(3, 40, value=gate_defaults["gate_voronoi_control_points"],
                                                     step=1, label="voronoi control points",
                                                     visible=gate_mode_visible["voronoi"])
                        gate_vor_radial = gr.Slider(0.0, 0.85, value=gate_defaults["gate_voronoi_radial_variation"],
                                                    step=0.01, label="voronoi radial variation",
                                                    visible=gate_mode_visible["voronoi"])
                        gate_vor_angular = gr.Slider(0.0, 0.25, value=gate_defaults["gate_voronoi_angular_jitter"],
                                                     step=0.01, label="voronoi angular jitter",
                                                     visible=gate_mode_visible["voronoi"])
                        gate_checkpoint_md = gr.Markdown("### Checkpoint controls", visible=gate_mode_visible["checkpoint"])
                        gate_checkpoint_count = gr.Slider(3, 32, value=gate_defaults["gate_checkpoint_count"],
                                                          step=1, label="checkpoint_count",
                                                          visible=gate_mode_visible["checkpoint"])
                        gate_checkpoint_radius_min_frac = gr.Slider(0.1, 0.9,
                                                                    value=gate_defaults["gate_checkpoint_radius_min_frac"],
                                                                    step=0.01, label="checkpoint_radius_min_frac",
                                                                    visible=gate_mode_visible["checkpoint"])
                        gate_checkpoint_angle_jitter = gr.Slider(0.0, 0.9,
                                                                 value=gate_defaults["gate_checkpoint_angle_jitter"],
                                                                 step=0.01, label="checkpoint_angle_jitter",
                                                                 visible=gate_mode_visible["checkpoint"])
                        gr.Markdown("### Batch")
                        gate_grid_n = gr.Dropdown([3, 4, 5, 6], value=gate_defaults["gate_grid_n"],
                                                  label="grid (n x n)")
                        gate_seed = gr.Number(value=gate_defaults["gate_seed"], precision=0, label="seed")
                        gate_batch_size = gr.Dropdown([64, 256, 1024, 2048, 4096],
                                                      value=gate_defaults["gate_batch_size"], label="batch size")
                        with gr.Row():
                            gate_reroll = gr.Button("reroll seed")
                            gate_generate = gr.Button("Generate gates", variant="primary")
                        gate_auto = gr.Checkbox(value=True, label="auto-update")
                    with gr.Column(scale=2):
                        gate_stats = gr.Markdown("")
                        with gr.Row():
                            gate_prev_btn = gr.Button("◀ prev")
                            gate_page_lbl = gr.Markdown("page 1/1")
                            gate_next_btn = gr.Button("next ▶")
                        gate_plot = gr.Plot()

                gate_state = gr.State(None)
                gate_config_state = gr.State(None)
                gate_page_state = gr.State(0)
                gate_controls = [gate_generator, gate_ordering, gate_width, gate_radius, gate_solve_iters,
                                 gate_show_raw, gate_scale, gate_min_np, gate_max_np,
                                 gate_min_point_distance,
                                 gate_polar_knots, gate_polar_radial, gate_polar_angular,
                                 gate_vor_sites, gate_vor_layout, gate_vor_control,
                                 gate_vor_radial, gate_vor_angular,
                                 gate_checkpoint_count, gate_checkpoint_radius_min_frac,
                                 gate_checkpoint_angle_jitter, gate_grid_n, gate_seed, gate_batch_size]
                if len(gate_controls) != len(GATE_CONTROL_KEYS):
                    raise RuntimeError("gate control list and key list are out of sync")

                gate_mode_outputs = [
                    gate_ordering,
                    gate_point_md, gate_point_note, gate_min_np, gate_max_np, gate_min_point_distance,
                    gate_polar_md, gate_polar_knots, gate_polar_radial, gate_polar_angular,
                    gate_vor_md, gate_vor_sites, gate_vor_layout, gate_vor_control,
                    gate_vor_radial, gate_vor_angular,
                    gate_checkpoint_md, gate_checkpoint_count, gate_checkpoint_radius_min_frac,
                    gate_checkpoint_angle_jitter,
                ]

                def _gate_mode_update(generator_name, current_ordering):
                    supported = gate_supported_orderings(generator_name)
                    ordering_value = current_ordering if current_ordering in supported else supported[0]
                    visible = gate_visible_sections(generator_name)
                    return [
                        gr.update(choices=supported, value=ordering_value),
                        gr.update(visible=visible["point"]),
                        gr.update(visible=visible["point"]),
                        gr.update(visible=visible["point"]),
                        gr.update(visible=visible["point"]),
                        gr.update(visible=visible["point"]),
                        gr.update(visible=visible["polar"]),
                        gr.update(visible=visible["polar"]),
                        gr.update(visible=visible["polar"]),
                        gr.update(visible=visible["polar"]),
                        gr.update(visible=visible["voronoi"]),
                        gr.update(visible=visible["voronoi"]),
                        gr.update(visible=visible["voronoi"]),
                        gr.update(visible=visible["voronoi"]),
                        gr.update(visible=visible["voronoi"]),
                        gr.update(visible=visible["voronoi"]),
                        gr.update(visible=visible["checkpoint"]),
                        gr.update(visible=visible["checkpoint"]),
                        gr.update(visible=visible["checkpoint"]),
                        gr.update(visible=visible["checkpoint"]),
                    ]

                def _generate_gates(*vals):
                    p = _collect_gate(*vals)
                    gn = int(p["gate_grid_n"])
                    try:
                        gates, cfg = generate_gate_batch(p)
                        fig = render_gate_page(gates, cfg, 0, gn)
                        st = _gate_stats(gates, cfg)
                        lbl = f"page 1/{n_pages(_gate_num_envs(gates), gn)}"
                        return fig, _gate_stats_md(st), gates, cfg, 0, lbl
                    except Exception as exc:  # keep the UI alive; full traceback is logged
                        msg = _ui_error_message(exc, "gate render")
                        err_fig = plt.figure(figsize=(5, 3))
                        err_fig.text(0.5, 0.5, f"error: {msg}", ha="center", va="center",
                                     fontsize=9, color="red", wrap=True)
                        err_st = {"error": msg, "yield": 0.0, "n_valid": 0,
                                  "n_invalid": 0, "mean_count": 0.0,
                                  "min_center_distance": float("nan"),
                                  "target_center_distance": float("nan")}
                        return err_fig, _gate_stats_md(err_st), None, None, 0, "page 1/1"

                def _go_gates(gates, cfg, page, gn, delta):
                    if gates is None or cfg is None:
                        return gr.update(), page, gr.update()
                    np_ = n_pages(_gate_num_envs(gates), int(gn))
                    new = max(0, min(int(page) + delta, np_ - 1))
                    fig = render_gate_page(gates, cfg, new, int(gn))
                    return fig, new, f"page {new + 1}/{np_}"

                gate_generator.change(_gate_mode_update, [gate_generator, gate_ordering], gate_mode_outputs)
                gate_generate.click(_generate_gates, gate_controls,
                                    [gate_plot, gate_stats, gate_state, gate_config_state,
                                     gate_page_state, gate_page_lbl])
                gate_reroll.click(lambda s: int(s) + 1, gate_seed, gate_seed).then(
                    _generate_gates, gate_controls,
                    [gate_plot, gate_stats, gate_state, gate_config_state, gate_page_state, gate_page_lbl])
                gate_prev_btn.click(lambda g, cfg, pg, gn: _go_gates(g, cfg, pg, gn, -1),
                                    [gate_state, gate_config_state, gate_page_state, gate_grid_n],
                                    [gate_plot, gate_page_state, gate_page_lbl])
                gate_next_btn.click(lambda g, cfg, pg, gn: _go_gates(g, cfg, pg, gn, +1),
                                    [gate_state, gate_config_state, gate_page_state, gate_grid_n],
                                    [gate_plot, gate_page_state, gate_page_lbl])

                def _maybe_gates(*vals):
                    *rest, auto_on = vals
                    if not auto_on:
                        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    return _generate_gates(*rest)
                for c in gate_controls:
                    ev = c.release if hasattr(c, "release") else c.change
                    ev(_maybe_gates, gate_controls + [gate_auto],
                       [gate_plot, gate_stats, gate_state, gate_config_state, gate_page_state, gate_page_lbl])

                app.load(_generate_gates, gate_controls,
                         [gate_plot, gate_stats, gate_state, gate_config_state, gate_page_state, gate_page_lbl])

    return app


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
