"""Render the committed utilities figures for the docs.

Produces the ``docs/assets`` figures for the utilities pages:
``utilities-overview.png`` (four-panel overview), ``boundary-props.png``
(cones/walls modes for the props page), ``oob-collision.png`` (out-of-bounds
contact report for the collision page), plus the checkpoints, progress, and
disc-collision figures.

Deterministic like ``viz.render_readme_assets``: fixed seeds, cpu device.

    .venv/bin/python -m viz.render_utility_assets
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

GEN_SEED = 7
BOX_SEED = 3
SDF_RES = 192


def _lift3(p2):
    """[n, 2] xy -> [n, 3] with z = 0 (vec3f query/position buffers)."""
    return np.concatenate([p2, np.zeros((p2.shape[0], 1), np.float32)], axis=1)


def _yaw_quats(yaw):
    """Yaw angles -> quatf components (x, y, z, w) about +z, float32 [n, 4]."""
    h = 0.5 * yaw.astype(np.float64)
    q = np.zeros((yaw.shape[0], 4), np.float32)
    q[:, 2] = np.sin(h)
    q[:, 3] = np.cos(h)
    return q


def _close(p):
    return np.vstack([p, p[:1]])


def _draw_track(ax, inner, outer):
    band = np.vstack([_close(outer), _close(inner)[::-1]])
    ax.add_patch(MplPolygon(band, closed=True, facecolor="0.85",
                            edgecolor="none", zorder=0))
    ax.plot(*_close(outer).T, "-", color="0.45", lw=1.0, zorder=1)
    ax.plot(*_close(inner).T, "-", color="0.45", lw=1.0, zorder=1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def render_utilities_overview(output_dir: Path = Path("docs/assets")) -> Path:
    """Render the four-panel utilities figure; returns the written path."""
    import warp as wp

    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    from track_gen.collision import CollisionChecker
    from track_gen.props import PropSampler

    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    track = gen.generate()
    valid = track.valid.numpy()
    e = int(np.argmax(valid))
    assert valid[e], "no valid env at the fixed seed"
    n_max = track.outer.shape[0] // E
    m = int(track.count.numpy()[e])
    inner = track.inner.numpy().reshape(E, n_max, 3)[e, :m, :2]
    outer = track.outer.numpy().reshape(E, n_max, 3)[e, :m, :2]
    center = track.center.numpy().reshape(E, n_max, 3)[e, :m, :2]

    def props_of(sampler):
        p = sampler.sample()
        n = int(p.count.numpy()[e])
        sl = slice(e * sampler._M, e * sampler._M + n)
        return (p.position.numpy().reshape(-1, 2)[sl],
                p.tangent.numpy().reshape(-1, 2)[sl],
                p.length.numpy()[sl], n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12.5))
    fig.suptitle("track_gen query/instancing utilities on one generated track",
                 fontsize=14)

    # One shared frame for all four panels so they align despite the SDF
    # panel's padded grid extent: track AABB plus a fixed margin.
    lo_lim = outer.min(axis=0)
    hi_lim = outer.max(axis=0)
    margin = 0.14 * float((hi_lim - lo_lim).max())
    xlim = (lo_lim[0] - margin, hi_lim[0] + margin)
    ylim = (lo_lim[1] - margin, hi_lim[1] + margin)

    # (a) points mode: cones on both boundaries.
    ax = axes[0, 0]
    _draw_track(ax, inner, outer)
    for boundary, color in (("outer", "#d95f02"), ("inner", "#1b9e77")):
        pos, _, _, n = props_of(
            PropSampler(track, spacing=0.12, boundary=boundary, mode="points"))
        ax.scatter(pos[:, 0], pos[:, 1], s=28, marker="^", color=color,
                   zorder=3, label=f"{boundary}: {n} cones")
    ax.set_title('props: cones (mode="points", spacing 0.12)')
    ax.legend(loc="upper right", fontsize=9)

    # (b) segments mode: wall pieces as chords.
    ax = axes[0, 1]
    _draw_track(ax, inner, outer)
    for boundary, color in (("outer", "#7570b3"), ("inner", "#e7298a")):
        pos, tang, length, n = props_of(
            PropSampler(track, spacing=0.18, boundary=boundary, mode="segments"))
        starts = pos - tang * (length[:, None] / 2)
        ends = pos + tang * (length[:, None] / 2)
        for s_, e_ in zip(starts, ends):
            ax.plot([s_[0], e_[0]], [s_[1], e_[1]], "-", color=color, lw=3.2,
                    solid_capstyle="butt", zorder=3)
        ax.scatter(pos[:, 0], pos[:, 1], s=8, color="k", zorder=4)
        ax.plot([], [], "-", color=color, lw=3.2,
                label=f"{boundary}: {n} wall pieces")
    ax.set_title('props: walls (mode="segments", spacing 0.18)')
    ax.legend(loc="upper right", fontsize=9)

    # (c) spacing comparison (points mode, outer), offset outward for visibility.
    ax = axes[1, 0]
    _draw_track(ax, inner, outer)
    ctr = center.mean(axis=0)
    for spacing, off, color in ((0.06, 0.0, "#66c2a5"), (0.12, 0.035, "#fc8d62"),
                                (0.24, 0.07, "#8da0cb")):
        pos, tang, length, n = props_of(
            PropSampler(track, spacing=spacing, boundary="outer", mode="points"))
        nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        sign = np.sign(((pos - ctr) * nrm).sum(axis=1, keepdims=True))
        shifted = pos + sign * nrm * off
        ax.scatter(shifted[:, 0], shifted[:, 1], s=14, color=color, zorder=3,
                   label=f"spacing {spacing}: {n} props (step {length[0]:.3f})")
    ax.set_title("props: effect of spacing (outer, offset for visibility)")
    ax.legend(loc="upper right", fontsize=9)

    # (d) collision: baked SDF field + boxes classified by the exact backend.
    ax = axes[1, 1]
    B = 14
    sdf = CollisionChecker(track, max_boxes=1, method="sdf", sdf_resolution=SDF_RES)
    lo = sdf._sdf_lo.numpy().reshape(-1, 2)[e]
    hi = sdf._sdf_hi.numpy().reshape(-1, 2)[e]
    phi = sdf._sdf_phi.numpy().reshape(E, SDF_RES, SDF_RES)[e]
    vmax = float(np.nanmax(np.abs(phi)))
    im = ax.imshow(phi, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]],
                   cmap="RdBu", vmin=-vmax, vmax=vmax, zorder=0)
    ax.contour(np.linspace(lo[0], hi[0], SDF_RES),
               np.linspace(lo[1], hi[1], SDF_RES), phi,
               levels=[0.0], colors="k", linewidths=0.8, zorder=1)

    rng = np.random.default_rng(BOX_SEED)
    idx = rng.integers(0, m, B)
    pos_np = np.full((E * B, 2), np.nan, np.float32)
    yaw_np = np.zeros(E * B, np.float32)
    he_np = np.zeros((E * B, 2), np.float32)
    pos_np[e * B:(e + 1) * B] = center[idx] + rng.normal(0, 0.09, (B, 2))
    yaw_np[e * B:(e + 1) * B] = rng.uniform(0, 2 * np.pi, B)
    he_np[e * B:(e + 1) * B] = rng.uniform(0.02, 0.06, (B, 2))
    contact = CollisionChecker(track, max_boxes=B, method="segments").query(
        wp.array(_lift3(pos_np), dtype=wp.vec3f, device="cpu"),
        wp.array(_yaw_quats(yaw_np), dtype=wp.quatf, device="cpu"),
        wp.array(he_np, dtype=wp.vec2f, device="cpu"))
    oob = contact.oob.numpy()[e * B:(e + 1) * B]
    near = contact.nearest.numpy().reshape(-1, 2)[e * B:(e + 1) * B]
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    for b in range(B):
        c, yw, he_ = pos_np[e * B + b], yaw_np[e * B + b], he_np[e * B + b]
        rot = np.array([[np.cos(yw), -np.sin(yw)], [np.sin(yw), np.cos(yw)]])
        corners = c + (signs * he_) @ rot.T
        col = "#d7191c" if oob[b] else "#1a9641"
        ax.add_patch(MplPolygon(corners, closed=True, facecolor="none",
                                edgecolor=col, lw=2.0, zorder=3))
        ax.plot([c[0], near[b, 0]], [c[1], near[b, 1]], ":", color=col,
                lw=0.9, zorder=2)
    ax.plot([], [], "-", color="#1a9641", lw=2, label="inside band")
    ax.plot([], [], "-", color="#d7191c", lw=2, label="out of bounds")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("collision: SDF field (blue = inside) + boxes vs exact backend")
    ax.legend(loc="upper right", fontsize=9)
    # Inset colorbar: keeps this panel's axes the same size as the others.
    cax = ax.inset_axes([1.02, 0.08, 0.03, 0.84])
    fig.colorbar(im, cax=cax, label="signed distance")

    for ax_ in axes.flat:
        ax_.set_xlim(*xlim)
        ax_.set_ylim(*ylim)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "utilities-overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_boundary_props(output_dir: Path = Path("docs/assets")) -> Path:
    """Dedicated props figure: cones (points mode) beside walls (segments mode)."""
    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    from track_gen.props import PropSampler

    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    track = gen.generate()
    e = int(np.argmax(track.valid.numpy()))
    n_max = track.outer.shape[0] // E
    m = int(track.count.numpy()[e])
    inner = track.inner.numpy().reshape(E, n_max, 3)[e, :m, :2]
    outer = track.outer.numpy().reshape(E, n_max, 3)[e, :m, :2]

    def props_of(sampler):
        p = sampler.sample()
        n = int(p.count.numpy()[e])
        sl = slice(e * sampler._M, e * sampler._M + n)
        return (p.position.numpy().reshape(-1, 2)[sl],
                p.tangent.numpy().reshape(-1, 2)[sl],
                p.length.numpy()[sl], n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.suptitle("PropSampler: one boundary, two prop modes", fontsize=14)

    ax = axes[0]
    _draw_track(ax, inner, outer)
    for boundary, color in (("outer", "#d95f02"), ("inner", "#1b9e77")):
        pos, _, _, n = props_of(
            PropSampler(track, spacing=0.12, boundary=boundary, mode="points"))
        ax.scatter(pos[:, 0], pos[:, 1], s=32, marker="^", color=color,
                   zorder=3, label=f"{boundary}: {n} cones")
    ax.set_title('mode="points": cone poses every ~0.12 along each boundary')
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    _draw_track(ax, inner, outer)
    for boundary, color in (("outer", "#7570b3"), ("inner", "#e7298a")):
        pos, tang, length, n = props_of(
            PropSampler(track, spacing=0.18, boundary=boundary, mode="segments"))
        starts = pos - tang * (length[:, None] / 2)
        ends = pos + tang * (length[:, None] / 2)
        for s_, e_ in zip(starts, ends):
            ax.plot([s_[0], e_[0]], [s_[1], e_[1]], "-", color=color, lw=3.2,
                    solid_capstyle="butt", zorder=3)
        ax.scatter(pos[:, 0], pos[:, 1], s=8, color="k", zorder=4)
        ax.plot([], [], "-", color=color, lw=3.2,
                label=f"{boundary}: {n} wall pieces")
    ax.set_title('mode="segments": wall chords (midpoint + yaw + length)')
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "boundary-props.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_oob_collision(output_dir: Path = Path("docs/assets")) -> Path:
    """Out-of-bounds figure: boxes classified against the drivable band, with
    the full contact report drawn — nearest boundary point, inward normal,
    signed clearance."""
    import warp as wp

    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    from track_gen.collision import CollisionChecker

    E = 4
    B = 10
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    track = gen.generate()
    e = int(np.argmax(track.valid.numpy()))
    n_max = track.outer.shape[0] // E
    m = int(track.count.numpy()[e])
    inner = track.inner.numpy().reshape(E, n_max, 3)[e, :m, :2]
    outer = track.outer.numpy().reshape(E, n_max, 3)[e, :m, :2]
    center = track.center.numpy().reshape(E, n_max, 3)[e, :m, :2]

    rng = np.random.default_rng(BOX_SEED)
    idx = rng.integers(0, m, B)
    pos_np = np.full((E * B, 2), np.nan, np.float32)
    yaw_np = np.zeros(E * B, np.float32)
    he_np = np.zeros((E * B, 2), np.float32)
    pos_np[e * B:(e + 1) * B] = center[idx] + rng.normal(0, 0.09, (B, 2))
    yaw_np[e * B:(e + 1) * B] = rng.uniform(0, 2 * np.pi, B)
    he_np[e * B:(e + 1) * B] = rng.uniform(0.02, 0.06, (B, 2))
    contact = CollisionChecker(track, max_boxes=B, method="segments").query(
        wp.array(_lift3(pos_np), dtype=wp.vec3f, device="cpu"),
        wp.array(_yaw_quats(yaw_np), dtype=wp.quatf, device="cpu"),
        wp.array(he_np, dtype=wp.vec2f, device="cpu"))
    sl = slice(e * B, (e + 1) * B)
    oob = contact.oob.numpy()[sl]
    clearance = contact.distance.numpy()[sl]
    near = contact.nearest.numpy().reshape(-1, 2)[sl]
    normal = contact.normal.numpy().reshape(-1, 2)[sl]

    fig, ax = plt.subplots(figsize=(10.5, 9))
    _draw_track(ax, inner, outer)
    nrm_len = 0.05 * float((outer.max(0) - outer.min(0)).max())
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    for b in range(B):
        c, yw, he_ = pos_np[e * B + b], yaw_np[e * B + b], he_np[e * B + b]
        rot = np.array([[np.cos(yw), -np.sin(yw)], [np.sin(yw), np.cos(yw)]])
        corners = c + (signs * he_) @ rot.T
        col = "#d7191c" if oob[b] else "#1a9641"
        ax.add_patch(MplPolygon(corners, closed=True, facecolor="none",
                                edgecolor=col, lw=2.0, zorder=3))
        ax.plot([c[0], near[b, 0]], [c[1], near[b, 1]], ":", color=col,
                lw=1.0, zorder=2)
        ax.scatter(*near[b], s=16, color=col, zorder=4)
        ax.annotate("", xy=near[b] + normal[b] * nrm_len, xytext=near[b],
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4),
                    zorder=4)
        ax.annotate(f"{clearance[b]:+.3f}", c, textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color=col, zorder=5)
    ax.plot([], [], "-", color="#1a9641", lw=2, label="inside band")
    ax.plot([], [], "-", color="#d7191c", lw=2, label="out of bounds")
    ax.plot([], [], ":", color="0.4", lw=1, label="nearest boundary point")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("CollisionChecker (segments): OOB flag, signed clearance,\n"
                 "nearest boundary point and inward normal per box")

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "oob-collision.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_checkpoints_overview(output_dir: Path = Path("docs/assets")) -> Path:
    """Track-sourced checkpoints (virtual gates) beside gate-sourced ones."""
    from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG,
                           TrackGenConfig, TrackGenerator)
    from track_gen.checkpoints import CheckpointSampler, CheckpointSet

    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    track = gen.generate()
    e = int(np.argmax(track.valid.numpy()))
    n_max = track.outer.shape[0] // E
    m = int(track.count.numpy()[e])
    inner = track.inner.numpy().reshape(E, n_max, 3)[e, :m, :2]
    outer = track.outer.numpy().reshape(E, n_max, 3)[e, :m, :2]

    sampler = CheckpointSampler(track, spacing=0.6)
    cps = sampler.sample()
    M = sampler._M
    n = int(cps.count.numpy()[e])
    sl = slice(e * M, e * M + n)
    pos = cps.position.numpy().reshape(-1, 3)[sl, :2]
    left = cps.left.numpy().reshape(-1, 3)[sl, :2]
    right = cps.right.numpy().reshape(-1, 3)[sl, :2]
    tang = cps.tangent.numpy().reshape(-1, 3)[sl, :2]

    gcfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.08)
    ggen = GateGenerator(gcfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    seq = ggen.generate()
    gset = CheckpointSet.from_gates(seq)
    ge = int(np.argmax(seq.valid.numpy()))
    GM = gset.position.shape[0] // E
    gn = int(gset.count.numpy()[ge])
    gsl = slice(ge * GM, ge * GM + gn)
    gpos = gset.position.numpy().reshape(-1, 3)[gsl, :2]
    gleft = gset.left.numpy().reshape(-1, 3)[gsl, :2]
    gright = gset.right.numpy().reshape(-1, 3)[gsl, :2]
    gtang = gset.tangent.numpy().reshape(-1, 3)[gsl, :2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.suptitle("CheckpointSet: one contract, two sources", fontsize=14)

    ax = axes[0]
    _draw_track(ax, inner, outer)
    for lf, rt in zip(left, right):
        ax.plot([lf[0], rt[0]], [lf[1], rt[1]], "-", color="#7570b3", lw=2.0,
                zorder=3)
    ax.scatter(pos[:, 0], pos[:, 1], s=22, color="#d95f02", zorder=4)
    ax.quiver(pos[:, 0], pos[:, 1], tang[:, 0], tang[:, 1], color="#d95f02",
              width=0.004, scale=18, zorder=4)
    ax.set_title(f"CheckpointSampler(track, spacing=0.6): {n} virtual gates\n"
                 "(crossing segments = inner-outer road cross-sections)")

    ax = axes[1]
    for lf, rt in zip(gleft, gright):
        ax.plot([lf[0], rt[0]], [lf[1], rt[1]], "-", color="#1b9e77", lw=2.6,
                zorder=3)
    ax.scatter(gpos[:, 0], gpos[:, 1], s=22, color="#d95f02", zorder=4)
    ax.quiver(gpos[:, 0], gpos[:, 1], gtang[:, 0], gtang[:, 1],
              color="#d95f02", width=0.004, scale=18, zorder=4)
    ax.plot(*np.vstack([gpos, gpos[:1]]).T, ":", color="0.6", lw=0.8, zorder=2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"CheckpointSet.from_gates(seq): {gn} gates (zero-copy)")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "checkpoints-overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_progress_tracking(output_dir: Path = Path("docs/assets")) -> Path:
    """Scripted agent threading track checkpoints; dist_to_next sawtooth lower panel."""
    import warp as wp

    from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
    from track_gen.checkpoints import CheckpointSampler
    from track_gen.progress import ProgressTracker

    E = 4
    cfg = TrackGenConfig(num_envs=E, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    track = gen.generate()
    e = int(np.argmax(track.valid.numpy()))
    n_max = track.outer.shape[0] // E
    m = int(track.count.numpy()[e])
    inner = track.inner.numpy().reshape(E, n_max, 3)[e, :m, :2]
    outer = track.outer.numpy().reshape(E, n_max, 3)[e, :m, :2]
    center = track.center.numpy().reshape(E, n_max, 3)[e, :m, :2]

    sampler = CheckpointSampler(track, spacing=0.9)
    cps = sampler.sample()
    M = sampler._M
    n = int(cps.count.numpy()[e])
    cpos = cps.position.numpy().reshape(-1, 3)[e * M:e * M + n, :2]
    cleft = cps.left.numpy().reshape(-1, 3)[e * M:e * M + n, :2]
    cright = cps.right.numpy().reshape(-1, 3)[e * M:e * M + n, :2]

    tracker = ProgressTracker(cps)
    rng = np.random.default_rng(GEN_SEED)
    path_idx = np.arange(0, m, 3)
    path = center[path_idx] + rng.normal(0.0, 0.01, (len(path_idx), 2))
    prog_trace, dist_trace, passed_at = [], [], []
    for s, p in enumerate(path):
        full = np.zeros((E, 3), np.float32)
        full[e, :2] = p
        ev = tracker.update(wp.array(full, dtype=wp.vec3f, device="cpu"))
        prog_trace.append(int(ev.progress.numpy()[e]))
        dist_trace.append(float(ev.dist_to_next.numpy()[e]))
        if int(ev.passed.numpy()[e]):
            passed_at.append(int(ev.checkpoint_passed.numpy()[e]))
    target = int(tracker._next.numpy()[e])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 11),
                                  gridspec_kw={"height_ratios": [4, 1]})
    _draw_track(ax, inner, outer)
    for k, (lf, rt) in enumerate(zip(cleft, cright)):
        col = "#d95f02" if k == target else ("#1a9641" if k in passed_at else "0.6")
        lw = 3.0 if k == target else 2.0
        ax.plot([lf[0], rt[0]], [lf[1], rt[1]], "-", color=col, lw=lw, zorder=3)
    sc = ax.scatter(path[:, 0], path[:, 1], c=prog_trace, cmap="viridis", s=12,
                    zorder=4)
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="progress (checkpoints passed)")
    ax.plot([], [], "-", color="#1a9641", lw=2, label="passed")
    ax.plot([], [], "-", color="#d95f02", lw=3, label="current target")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("ProgressTracker on track checkpoints: path colored by progress")

    ax2.plot(dist_trace, lw=1.2, color="#7570b3")
    ax2.set_title("dist_to_next per step (reward = -delta)")
    ax2.set_xlabel("step")

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "progress-tracking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_disc_collision(output_dir: Path = Path("docs/assets")) -> Path:
    """Gate posts as discs; agent boxes colored by DiscChecker verdicts."""
    import warp as wp
    from matplotlib.patches import Circle

    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.collision import DiscChecker

    E = 4
    RADIUS = 0.03
    cfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.08)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=GEN_SEED, num_envs=E, device="cpu"))
    seq = gen.generate()
    e = int(np.argmax(seq.valid.numpy()))
    G = seq.position.shape[0] // E
    ln = int(seq.count.numpy()[e])
    left = seq.left.numpy().reshape(E, G, 3)[..., :2]
    right = seq.right.numpy().reshape(E, G, 3)[..., :2]

    posts = np.empty((E, 2 * G, 2), np.float32)
    posts[:, 0::2] = left
    posts[:, 1::2] = right
    posts_wp = wp.array(_lift3(posts.reshape(-1, 2)), dtype=wp.vec3f, device="cpu")

    B = 8
    rng = np.random.default_rng(BOX_SEED)
    pos_np = np.full((E * B, 2), np.nan, np.float32)
    yaw_np = np.zeros(E * B, np.float32)
    he_np = np.zeros((E * B, 2), np.float32)
    for b in range(B):
        g = int(rng.integers(0, ln))
        anchor = posts[e, 2 * g + (b % 2)]
        pos_np[e * B + b] = anchor + rng.normal(0.0, 0.05, 2)
        yaw_np[e * B + b] = rng.uniform(0, 2 * np.pi)
        he_np[e * B + b] = rng.uniform(0.02, 0.05, 2)
    checker = DiscChecker(posts_wp, radius=RADIUS, max_boxes=B, num_envs=E)
    res = checker.query(wp.array(_lift3(pos_np), dtype=wp.vec3f, device="cpu"),
                        wp.array(_yaw_quats(yaw_np), dtype=wp.quatf, device="cpu"),
                        wp.array(he_np, dtype=wp.vec2f, device="cpu"))
    hit = res.hit.numpy()[e * B:(e + 1) * B]

    fig, ax = plt.subplots(figsize=(10.5, 9))
    for g in range(ln):
        lf, rt = left[e, g], right[e, g]
        ax.plot([lf[0], rt[0]], [lf[1], rt[1]], "-", color="0.75", lw=1.2, zorder=1)
        for p in (lf, rt):
            ax.add_patch(Circle(p, RADIUS, facecolor="#7570b3", alpha=0.5,
                                edgecolor="#7570b3", zorder=2))
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    for b in range(B):
        c, yw, he_ = pos_np[e * B + b], yaw_np[e * B + b], he_np[e * B + b]
        rot = np.array([[np.cos(yw), -np.sin(yw)], [np.sin(yw), np.cos(yw)]])
        corners = c + (signs * he_) @ rot.T
        col = "#d7191c" if hit[b] else "#1a9641"
        ax.add_patch(MplPolygon(corners, closed=True, facecolor="none",
                                edgecolor=col, lw=2.0, zorder=3))
    ax.plot([], [], "-", color="#d7191c", lw=2, label="post hit")
    ax.plot([], [], "-", color="#1a9641", lw=2, label="clear")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("DiscChecker: gate posts as disc obstacles (radius %.2f)" % RADIUS)

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "disc-collision.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    print(render_utilities_overview().resolve())
    print(render_boundary_props().resolve())
    print(render_oob_collision().resolve())
    print(render_checkpoints_overview().resolve())
    print(render_progress_tracking().resolve())
    print(render_disc_collision().resolve())


if __name__ == "__main__":
    main()
