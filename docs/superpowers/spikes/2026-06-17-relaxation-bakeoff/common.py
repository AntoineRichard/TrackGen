"""Shared, FAIR bake-off harness for the track-relaxation spike.

Both solver spikes (XPBD projection vs differentiable energy) import THIS module so
they are scored identically. A spike only implements:

    relax(center0, half_width, band, **hparams) -> (center_relaxed [E,N,2], info: dict)

and then calls `evaluate(...)` + `plot_before_after(...)` + `print_scorecard(...)`.

Inputs are FIXED: load_tracks() returns the same simple (non-self-intersecting),
arc-length-uniform Bezier centerlines every run (cached to broken_tracks.pt).

Geometry conventions match the repo (closed loop, left-normal = (-Ty, Tx)).
Pure torch, CPU. No warp needed once the cache exists.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/tmp/tg_run")  # so `import track_gen` resolves via the symlink
import torch

CACHE = "/tmp/tg_run/bakeoff/broken_tracks.pt"

# ---------------------------------------------------------------------------
# Geometry helpers (self-contained; mirror track_gen.geometry conventions)
# ---------------------------------------------------------------------------

def _roll(x, k):  # roll along the point axis (dim=1)
    return torch.roll(x, shifts=k, dims=1)


def safe_normalize(v, eps=1e-9):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def perimeter(center):  # [E,N,2] -> [E]
    seg = _roll(center, -1) - center
    return torch.linalg.norm(seg, dim=-1).sum(dim=1)


def mean_seg_len(center):  # [E] mean spacing L
    return perimeter(center) / center.shape[1]


def tangents_normals(center):
    T = safe_normalize(_roll(center, -1) - _roll(center, 1))
    Nrm = torch.stack([-T[..., 1], T[..., 0]], dim=-1)
    return T, Nrm


def menger_curvature(center, eps=1e-12):  # [E,N] >= 0
    pp, pc, pn = _roll(center, 1), center, _roll(center, -1)
    a, b, c = pc - pp, pn - pc, pn - pp
    la, lb, lc = (torch.linalg.norm(x, dim=-1) for x in (a, b, c))
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    area = 0.5 * cross.abs()
    return 4.0 * area / (la * lb * lc).clamp_min(eps)


def polygon_area(center):  # signed shoelace [E]
    x, y = center[..., 0], center[..., 1]
    xn, yn = _roll(x.unsqueeze(-1), -1).squeeze(-1), _roll(y.unsqueeze(-1), -1).squeeze(-1)
    return 0.5 * (x * yn - xn * y).sum(dim=1)


def circ_index_dist(N, device):  # [N,N] circular |i-j|
    idx = torch.arange(N, device=device)
    d = (idx[None] - idx[:, None]).abs()
    return torch.minimum(d, N - d)


def band_per_track(center, half_width):  # [E] int excluded-neighbor half-window ~ D/L
    L = mean_seg_len(center)
    D = 2.0 * half_width
    return (D / L).round().long().clamp_min(1)


def separation_min(center, band):  # min non-adjacent Euclidean distance [E]
    E, N, _ = center.shape
    dmat = torch.cdist(center, center)  # [E,N,N]
    circ = circ_index_dist(N, center.device)  # [N,N]
    mask = circ[None] <= band.view(E, 1, 1)
    dmat = dmat.masked_fill(mask, float("inf"))
    return dmat.amin(dim=(-1, -2))  # [E]


def curvature_radius_min(center):  # 1 / max curvature [E]
    kappa = menger_curvature(center)
    return 1.0 / kappa.amax(dim=1).clamp_min(1e-12)


def thickness(center, band):  # [E] discrete thickness ~ min(radius, sep/2)
    return torch.minimum(curvature_radius_min(center), 0.5 * separation_min(center, band))


def self_intersections(poly):  # [E] count of proper crossings of a closed polyline
    E, N, _ = poly.shape
    A = poly
    B = _roll(poly, -1)

    def ccw(o, p, q):  # broadcast cross [o:E,?,?] etc -> sign
        return (q[..., 1] - o[..., 1]) * (p[..., 0] - o[..., 0]) - \
               (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])

    Ai = A[:, :, None, :]; Bi = B[:, :, None, :]   # segment i along dim1
    Aj = A[:, None, :, :]; Bj = B[:, None, :, :]   # segment j along dim2
    d1 = ccw(Aj, Bj, Ai); d2 = ccw(Aj, Bj, Bi)
    d3 = ccw(Ai, Bi, Aj); d4 = ccw(Ai, Bi, Bj)
    cross = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))  # [E,N,N]
    circ = circ_index_dist(N, poly.device)
    adj = circ[None] <= 1  # exclude self + adjacent (shared endpoint)
    cross = cross & ~adj
    return (cross.sum(dim=(-1, -2)) // 2).long()  # each pair counted twice


def inflate_constant(center, half_width):  # -> outer, inner [E,N,2]
    _, Nrm = tangents_normals(center)
    wn = half_width * Nrm
    a, b = center + wn, center - wn
    area_a = polygon_area(a).abs()
    area_b = polygon_area(b).abs()
    a_out = (area_a >= area_b).view(-1, 1, 1)
    return torch.where(a_out, a, b), torch.where(a_out, b, a)


# ---------------------------------------------------------------------------
# Track generation + cache
# ---------------------------------------------------------------------------

def _resample_uniform(points, N):  # [E,M,2] (NaN-padded) -> [E,N,2] arc-uniform
    out = torch.full((points.shape[0], N, 2), float("nan"))
    for e in range(points.shape[0]):
        pe = points[e]
        pe = pe[torch.isfinite(pe).all(dim=-1)]
        if pe.shape[0] < 3:
            continue
        closed = torch.cat([pe, pe[:1]], dim=0)
        seg = torch.linalg.norm(closed[1:] - closed[:-1], dim=-1)
        s = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
        total = s[-1]
        targets = torch.arange(N, dtype=torch.float32) * (total / N)
        idx = torch.searchsorted(s[1:], targets, right=False).clamp(max=seg.shape[0] - 1)
        frac = ((targets - s[idx]) / seg[idx].clamp_min(1e-12)).clamp(0, 1).unsqueeze(-1)
        out[e] = closed[idx] + frac * (closed[idx + 1] - closed[idx])
    return out


def build_cache(E_keep=64, N=256, scale=1.0, seed0=20, device="cpu"):
    """Generate Bezier centerlines, resample to N uniform pts, keep simple ones."""
    import warp as wp; wp.init()
    from track_gen.types import TrackGenConfig
    from track_gen.generators import BezierCenterlineGenerator
    from track_gen.rng_utils import PerEnvSeededRNG

    keep = []
    seed = seed0
    while len(keep) < E_keep:
        E = 128
        seeds = torch.arange(E, dtype=torch.int32) + seed
        rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device=device)
        rng.set_seeds(seeds, ids=torch.arange(E, dtype=torch.int32))
        cfg = TrackGenConfig(generator="bezier", device=device, num_envs=E, scale=scale,
                             max_regen_iters=20, turning_tol=0.35)
        cl = BezierCenterlineGenerator(cfg, rng).generate(torch.arange(E))
        uni = _resample_uniform(cl.points, N)  # [E,N,2]
        ok = torch.isfinite(uni).all(dim=(1, 2))
        si = torch.zeros(E, dtype=torch.long)
        si[ok] = self_intersections(uni[ok])
        simple = ok & (si == 0)
        for e in torch.where(simple)[0].tolist():
            keep.append(uni[e])
            if len(keep) >= E_keep:
                break
        seed += E
    center0 = torch.stack(keep[:E_keep], dim=0)
    torch.save({"center0": center0, "N": N, "scale": scale}, CACHE)
    return center0


def load_tracks():
    import os
    if not os.path.exists(CACHE):
        build_cache()
    d = torch.load(CACHE)
    return d["center0"]  # [E,N,2]


# ---------------------------------------------------------------------------
# Scoring + plotting
# ---------------------------------------------------------------------------

def evaluate(name, center0, center_relaxed, half_width, seconds, iters, tol=0.02):
    """Return a scorecard dict comparing relaxed tracks to target half_width."""
    band = band_per_track(center0, half_width)  # fixed band from the (uniform) init geometry
    th0 = thickness(center0, band)
    th1 = thickness(center_relaxed, band)
    target = half_width * (1.0 - tol)

    o1, i1 = inflate_constant(center_relaxed, half_width)
    border_x = self_intersections(o1) + self_intersections(i1)
    center_x = self_intersections(center_relaxed)
    valid = (th1 >= target) & (border_x == 0) & (center_x == 0)

    disp = torch.linalg.norm(center_relaxed - center0, dim=-1).mean(dim=1)  # [E]
    area_ratio = polygon_area(center_relaxed).abs() / polygon_area(center0).abs().clamp_min(1e-9)
    E = center0.shape[0]
    return {
        "name": name, "E": E, "half_width": half_width, "seconds": seconds, "iters": iters,
        "valid_frac": valid.float().mean().item(),
        "n_valid": int(valid.sum()),
        "thickness_init_med": th0.median().item(),
        "thickness_relaxed_min": th1.min().item(),
        "thickness_relaxed_med": th1.median().item(),
        "thickness_target": target,
        "frac_meeting_thickness": (th1 >= target).float().mean().item(),
        "border_xings_total": int(border_x.sum()),
        "tracks_with_border_xings": int((border_x > 0).sum()),
        "mean_displacement_med": disp.median().item(),
        "mean_displacement_max": disp.max().item(),
        "area_ratio_med": area_ratio.median().item(),
        "valid_mask": valid,
    }


def print_scorecard(sc):
    print(f"\n================  {sc['name']}  ================")
    print(f"envs={sc['E']}  half_width={sc['half_width']:.3f}  iters={sc['iters']}  time={sc['seconds']:.3f}s")
    print(f"VALID (thickness>=target AND zero border/center crossings): {sc['n_valid']}/{sc['E']}  ({100*sc['valid_frac']:.0f}%)")
    print(f"thickness  init(med)={sc['thickness_init_med']:.4f} -> relaxed med={sc['thickness_relaxed_med']:.4f} min={sc['thickness_relaxed_min']:.4f}  (target {sc['thickness_target']:.4f})")
    print(f"frac meeting thickness target: {100*sc['frac_meeting_thickness']:.0f}%")
    print(f"border self-crossings: total={sc['border_xings_total']}  tracks_affected={sc['tracks_with_border_xings']}")
    print(f"shape change: mean-displacement med={sc['mean_displacement_med']:.4f} max={sc['mean_displacement_max']:.4f}  area_ratio med={sc['area_ratio_med']:.3f}")


def plot_before_after(center0, center_relaxed, half_width, path, idxs=range(6)):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idxs = list(idxs)
    fig, axes = plt.subplots(2, len(idxs), figsize=(3 * len(idxs), 6))
    for col, e in enumerate(idxs):
        for row, (C, title) in enumerate([(center0, "init"), (center_relaxed, "relaxed")]):
            ax = axes[row, col]
            c = C[e]
            o, i = inflate_constant(c.unsqueeze(0), half_width)
            o, i = o[0], i[0]
            for poly, st in [(c, "k--"), (o, "b-"), (i, "r-")]:
                pp = torch.cat([poly, poly[:1]])
                ax.plot(pp[:, 0], pp[:, 1], st, lw=0.8)
            ax.set_aspect("equal"); ax.axis("off")
            if row == 0:
                ax.set_title(f"env {e}", fontsize=8)
            ax.set_ylabel(title)
    fig.suptitle(f"before(top)/after(bottom)  half_width={half_width}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=90); print(f"saved {path}")
