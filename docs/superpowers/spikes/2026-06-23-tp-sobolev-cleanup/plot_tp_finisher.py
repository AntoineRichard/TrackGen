"""Generate pre/post plots for the TP-Sobolev smoothing finisher.

This is a spike utility, not runtime code. It compares the finisher input
(``xpbd`` output) against the finisher output (``xpbd`` + short TP-Sobolev flow)
on the same seeded batch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from benchmarks.benchmark_relaxation import _gen_simple_tracks
from track_gen._src.types import TrackGenConfig
from tests._oracle import geometry, relaxation


def _inflate(center: torch.Tensor, half_width: float) -> tuple[torch.Tensor, torch.Tensor]:
    _, normals = geometry.tangents_normals(center)
    return center + half_width * normals, center - half_width * normals


def _half_clearance(center: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    kappa = geometry.menger_curvature(center)
    crad = 1.0 / kappa.clamp_min(1e-12)
    n = center.shape[1]
    dmat = torch.cdist(center, center)
    circ = geometry.circ_index_dist(n, center.device)
    dmat = dmat.masked_fill(circ[None] <= band.view(-1, 1, 1), float("inf"))
    nn = dmat.amin(dim=-1)
    return torch.minimum(crad, 0.5 * nn)


def _metrics(center: torch.Tensor, band: torch.Tensor) -> dict[str, torch.Tensor]:
    hc = _half_clearance(center, band)
    kappa = geometry.menger_curvature(center)
    return {
        "clearance_cv": hc.std(dim=1) / hc.mean(dim=1).clamp_min(1e-12),
        "kmax": kappa.amax(dim=1),
        "thickness": geometry.thickness(center, band),
    }


def _plot_curve_grid(
    xpbd: torch.Tensor,
    fin: torch.Tensor,
    half_width: float,
    idxs: torch.Tensor,
    out_path: Path,
) -> None:
    xpbd = xpbd.detach().cpu()
    fin = fin.detach().cpu()
    n_cols = len(idxs)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.0 * n_cols, 6.0), squeeze=False)
    for col, e_t in enumerate(idxs):
        e = int(e_t)
        for row, (curves, label) in enumerate(((xpbd, "XPBD"), (fin, "XPBD + TP finisher"))):
            ax = axes[row, col]
            center = curves[e].unsqueeze(0)
            outer, inner = _inflate(center, half_width)
            for poly, color, lw, alpha in (
                (outer[0], "#2f80ed", 0.8, 0.9),
                (inner[0], "#d64545", 0.8, 0.9),
                (center[0], "#111111", 1.0, 0.95),
            ):
                closed = torch.cat((poly, poly[:1]), dim=0)
                ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw, alpha=alpha)
            ax.set_aspect("equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"env {e}", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(f"TP-Sobolev finisher pre/post, half_width={half_width}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_overlay_grid(
    xpbd: torch.Tensor,
    fin: torch.Tensor,
    idxs: torch.Tensor,
    out_path: Path,
) -> None:
    xpbd = xpbd.detach().cpu()
    fin = fin.detach().cpu()
    n_cols = len(idxs)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.1), squeeze=False)
    for col, e_t in enumerate(idxs):
        e = int(e_t)
        ax = axes[0, col]
        for curve, color, label, lw in (
            (xpbd[e], "#9aa0a6", "XPBD", 1.0),
            (fin[e], "#111111", "TP finisher", 1.2),
        ):
            closed = torch.cat((curve, curve[:1]), dim=0)
            ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw, label=label)
        ax.set_title(f"env {e}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Centerline overlay: pre vs post finisher", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_metric_pairs(metrics_x: dict[str, torch.Tensor], metrics_f: dict[str, torch.Tensor], out_path: Path) -> None:
    labels = [
        ("clearance_cv", "Clearance CV"),
        ("kmax", "Peak Menger curvature"),
        ("thickness", "Thickness"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (key, title) in zip(axes, labels, strict=True):
        pre = metrics_x[key].detach().cpu()
        post = metrics_f[key].detach().cpu()
        ax.scatter(pre, post, s=16, alpha=0.7, color="#111111")
        lo = float(torch.minimum(pre.min(), post.min()))
        hi = float(torch.maximum(pre.max(), post.max()))
        pad = max((hi - lo) * 0.05, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#d64545", lw=1.0, alpha=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("XPBD")
        ax.set_ylabel("XPBD + TP")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--E", type=int, default=64)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--half-width", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_suffix(""))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    center0 = _gen_simple_tracks(args.E, args.N, 1.0, args.device, args.seed)
    base = dict(
        device=args.device,
        num_envs=args.E,
        num_points=args.N,
        half_width=args.half_width,
        relax_solver="xpbd",
        relax_iters=150,
        relax_bend_relax=1.5,
        relax_margin=0.15,
    )
    xpbd = relaxation.relax(center0, TrackGenConfig(**base, smooth_finish=False))
    fin = relaxation.relax(
        center0,
        TrackGenConfig(**base, smooth_finish=True, smooth_finish_iters=8, smooth_finish_tau=0.2),
    )

    band = relaxation._band(center0, TrackGenConfig(**base))
    metrics_x = _metrics(xpbd, band)
    metrics_f = _metrics(fin, band)
    improvement = metrics_x["clearance_cv"] - metrics_f["clearance_cv"]
    idxs = torch.argsort(improvement, descending=True)[:6].detach().cpu()

    _plot_curve_grid(xpbd, fin, args.half_width, idxs, args.out_dir / "tp-finisher-pre-post-grid.png")
    _plot_overlay_grid(xpbd, fin, idxs, args.out_dir / "tp-finisher-centerline-overlay.png")
    _plot_metric_pairs(metrics_x, metrics_f, args.out_dir / "tp-finisher-metric-pairs.png")

    def med(key: str, values: dict[str, torch.Tensor]) -> float:
        return float(values[key].median().detach().cpu())

    print(f"output_dir={args.out_dir}")
    print(f"selected_envs={','.join(str(int(i)) for i in idxs)}")
    print(f"clearance_cv_median: xpbd={med('clearance_cv', metrics_x):.6f} tp={med('clearance_cv', metrics_f):.6f}")
    print(f"kmax_median: xpbd={med('kmax', metrics_x):.6f} tp={med('kmax', metrics_f):.6f}")
    print(f"thickness_median: xpbd={med('thickness', metrics_x):.6f} tp={med('thickness', metrics_f):.6f}")


if __name__ == "__main__":
    main()
