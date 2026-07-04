"""Render the committed utilities-overview figure for the docs.

Produces ``docs/assets/utilities-overview.png``: a four-panel overview of the
query/instancing utilities on one generated track — ``track_gen.props`` cones
(points mode), walls (segments mode), the effect of spacing, and a
``track_gen.collision`` panel (baked SDF field + boxes classified by the exact
segments backend).

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
    inner = track.inner.numpy().reshape(E, n_max, 2)[e, :m]
    outer = track.outer.numpy().reshape(E, n_max, 2)[e, :m]
    center = track.center.numpy().reshape(E, n_max, 2)[e, :m]

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
        wp.array(pos_np.reshape(-1, 2), dtype=wp.vec2f, device="cpu"),
        wp.array(yaw_np, dtype=wp.float32, device="cpu"),
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


def main() -> None:
    print(render_utilities_overview().resolve())


if __name__ == "__main__":
    main()
