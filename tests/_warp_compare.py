"""Test-side helpers for warp_pipeline in-place API.

``to_t``                 -- wp.array -> torch.Tensor (zero-copy).
Torch-in/torch-out wrappers allocate wp.array buffers, call the in-place
warp function, and return torch tensors.  All torch usage stays test-side.
"""
import warp as wp


def to_t(a):
    """wp.array -> torch.Tensor (zero-copy, same device) for oracle comparisons."""
    import torch  # tests are dev-side; torch is available
    return a if a.__class__.__module__.startswith("torch") else wp.to_torch(a)


# ---------------------------------------------------------------------------
# Torch-in / torch-out convenience wrappers (test-side only)
# ---------------------------------------------------------------------------

def corner_count_sample(seeds, attempt, config):
    """Torch-in/out: per-env corner count [E] int32."""
    import torch
    from track_gen._src import warp_pipeline as wpl
    E = seeds.shape[0]
    dev = str(seeds.device)
    seeds_wp = wp.from_torch(seeds.to(torch.int32).contiguous(), dtype=wp.int32)
    out_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wpl.corner_count_sample_inplace(seeds_wp, attempt, config, out_wp)
    return wp.to_torch(out_wp)


def corner_sample(seeds, attempt, config):
    """Torch-in/out: corners [E, P, 2] float32."""
    return _corner_sample_raw(seeds, attempt, config)[0]


def _corner_sample_raw(seeds, attempt, config):
    """Torch-in/out: (corners [E, P, 2] float32, cells [E, P] int32)."""
    import torch
    from track_gen._src import warp_pipeline as wpl
    E = seeds.shape[0]
    P = int(config.max_num_points)
    dev = str(seeds.device)
    seeds_wp = wp.from_torch(seeds.to(torch.int32).contiguous(), dtype=wp.int32)
    out_wp = wp.empty(E * P, dtype=wp.vec2f, device=dev)
    used_wp = wp.empty(E * P, dtype=wp.int32, device=dev)
    wpl.corner_sample_inplace(seeds_wp, attempt, config, out_wp, used_wp)
    corners = wp.to_torch(out_wp).view(E, P, 2)
    cells = wp.to_torch(used_wp).view(E, P)
    return corners, cells


def ccw_sort(points, count=None):
    """Torch-in/out: sorted corners [E, P, 2] float32."""
    import torch
    from track_gen._src import warp_pipeline as wpl
    E, P, _ = points.shape
    dev = str(points.device)
    flat = E * P
    pf = wp.from_torch(points.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    if count is None:
        count_t = torch.full((E,), P, device=points.device, dtype=torch.int32)
    else:
        count_t = count.to(dtype=torch.int32, device=points.device).contiguous()
    cnt_wp = wp.from_torch(count_t, dtype=wp.int32)
    keys_wp = wp.empty(flat, dtype=wp.float32, device=dev)
    out_wp = wp.empty(flat, dtype=wp.vec2f, device=dev)
    wpl.ccw_sort_inplace(pf, cnt_wp, keys_wp, out_wp, P)
    return wp.to_torch(out_wp).view(E, P, 2)


def assemble(corners, count, config):
    """Torch-in/out: dense centerline [E, P*npseg, 2] float32."""
    import torch
    from track_gen._src import warp_pipeline as wpl
    E, P, _ = corners.shape
    npseg = int(config.num_points_per_segment)
    dev = str(corners.device)
    flat = E * P
    cf = wp.from_torch(corners.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    cnt_t = count.to(dtype=torch.int32, device=corners.device).contiguous()
    cnt_wp = wp.from_torch(cnt_t, dtype=wp.int32)
    tan_wp = wp.empty(flat, dtype=wp.vec2f, device=dev)
    scale_wp = wp.empty(flat, dtype=wp.float32, device=dev)
    out_wp = wp.empty(E * P * npseg, dtype=wp.vec2f, device=dev)
    wpl.assemble_inplace(cf, cnt_wp, config, tan_wp, scale_wp, out_wp)
    return wp.to_torch(out_wp).view(E, P * npseg, 2)


def arc_length_resample_warp(points, num):
    """Torch-in/out: (resampled [E, num, 2] float32, count [E] long)."""
    import torch
    from track_gen._src import warp_pipeline as wpl
    E, M, _ = points.shape
    dev = str(points.device)
    pts_wp = wp.from_torch(points.reshape(E * M, 2).contiguous(), dtype=wp.vec2f)
    real_wp = wp.empty(E * M, dtype=wp.vec2f, device=dev)
    seg_wp = wp.empty(E * M, dtype=wp.float32, device=dev)
    s_wp = wp.empty(E * (M + 1), dtype=wp.float32, device=dev)
    count_r_wp = wp.empty(E, dtype=wp.int32, device=dev)
    count_out_wp = wp.empty(E, dtype=wp.int32, device=dev)
    out_wp = wp.empty(E * num, dtype=wp.vec2f, device=dev)
    wpl.arc_length_resample_inplace(pts_wp, M, num, real_wp, seg_wp, s_wp,
                                    count_r_wp, count_out_wp, out_wp, dev)
    resampled = wp.to_torch(out_wp).view(E, num, 2)
    count = wp.to_torch(count_out_wp).long()
    return resampled, count


def turning_number(center, count=None):
    """Torch-in/out: signed total turning [E] float32."""
    import torch
    from track_gen._src import warp_pipeline as wpl
    E, N, _ = center.shape
    dev = str(center.device)
    cf = wp.from_torch(center.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    if count is None:
        count_t = torch.full((E,), N, device=center.device, dtype=torch.int32)
    else:
        count_t = count.to(dtype=torch.int32, device=center.device).contiguous()
    cnt_wp = wp.from_torch(count_t, dtype=wp.int32)
    out_wp = wp.empty(E, dtype=wp.float32, device=dev)
    wpl.turning_number_inplace(cf, N, cnt_wp, out_wp)
    return wp.to_torch(out_wp)


def self_intersections(poly, count=None):
    """Torch-in/out: crossing count [E] int32."""
    import torch
    from track_gen._src import warp_pipeline as wpl
    E, N, _ = poly.shape
    dev = str(poly.device)
    pf = wp.from_torch(poly.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    if count is None:
        count_t = torch.full((E,), N, device=poly.device, dtype=torch.int32)
    else:
        count_t = count.to(dtype=torch.int32, device=poly.device).contiguous()
    cnt_wp = wp.from_torch(count_t, dtype=wp.int32)
    out_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wpl.self_intersections_inplace(pf, cnt_wp, out_wp, N)
    return wp.to_torch(out_wp)


def thickness(points, band, count=None):
    """Torch-in/out: min thickness per env [E] float32."""
    import torch
    from track_gen._src import warp_pipeline as wpl
    E, N, _ = points.shape
    dev = str(points.device)
    pf = wp.from_torch(points.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    band_t = band.to(dtype=torch.int32, device=points.device).contiguous()
    band_wp = wp.from_torch(band_t, dtype=wp.int32)
    if count is None:
        count_t = torch.full((E,), N, device=points.device, dtype=torch.int32)
    else:
        count_t = count.to(dtype=torch.int32, device=points.device).contiguous()
    cnt_wp = wp.from_torch(count_t, dtype=wp.int32)
    out_wp = wp.empty(E, dtype=wp.float32, device=dev)
    wp.launch(wpl._thickness_k, dim=E,
              inputs=[pf, band_wp, N, cnt_wp, out_wp],
              device=dev)
    if "cuda" in dev:
        wp.synchronize()
    return wp.to_torch(out_wp)


def validity(center, w, count, gen_valid, config, outer=None, inner=None):
    """Torch-in/out: validity mask [E] bool."""
    import torch
    from track_gen._src import warp_pipeline as wpl
    E, N = w.shape
    dev = str(center.device)
    flat = E * N
    cf = wp.from_torch(center.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    wf = wp.from_torch(w.reshape(flat).contiguous().to(torch.float32), dtype=wp.float32)
    cnt_wp = wp.from_torch(count.to(torch.int32).contiguous(), dtype=wp.int32)
    gv_wp = wp.from_torch(gen_valid.to(torch.int32).contiguous(), dtype=wp.int32)
    if outer is None or inner is None:
        has_border = 0
        ob = cf
        ib = cf
    else:
        has_border = 1
        ob = wp.from_torch(outer.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
        ib = wp.from_torch(inner.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    out_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wpl.validity_inplace(cf, wf, cnt_wp, gv_wp, ob, ib, has_border, N, out_wp, config)
    return wp.to_torch(out_wp).bool()


def gates(corners, dense, count, config):
    """Torch-in/out: accept mask [E] bool. Composed test-side from in-place primitives."""
    import math
    import torch
    from track_gen._src import warp_pipeline as wpl

    E, P, _ = corners.shape
    dev = str(corners.device)

    # --- ANGLE gate ---
    cf = wp.from_torch(corners.reshape(E * P, 2).contiguous(), dtype=wp.vec2f)
    cnt_t = count.to(device=corners.device, dtype=torch.int32).contiguous()
    cnt_wp = wp.from_torch(cnt_t, dtype=wp.int32)
    angle_ok_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wp.launch(wpl._corner_angles_gate_k, dim=E,
              inputs=[cf, cnt_wp, P, float(config.min_angle), angle_ok_wp],
              device=dev)

    # --- TURN + FINITE ---
    rs_turn, cnt_turn = arc_length_resample_warp(dense, int(config.num_points_per_segment))
    turn = turning_number(rs_turn)

    # --- SIMPLE ---
    rs_simple, _ = arc_length_resample_warp(dense, int(config.num_points))
    cross = self_intersections(rs_simple)

    # --- COMBINE ---
    out_wp = wp.empty(E, dtype=wp.int32, device=dev)
    wp.launch(
        wpl._gates_combine_k, dim=E,
        inputs=[angle_ok_wp,
                wp.from_torch(turn.contiguous(), dtype=wp.float32),
                wp.from_torch(cnt_turn.to(torch.int32).contiguous(), dtype=wp.int32),
                wp.from_torch(cross.to(torch.int32).contiguous(), dtype=wp.int32),
                float(config.turning_tol),
                out_wp],
        device=dev,
    )
    if "cuda" in dev:
        wp.synchronize()
    return wp.to_torch(out_wp).bool()


def xpbd_solve(center0, band, L0, config, count=None):
    """Torch-in/out: full XPBD solve [E, N, 2] float32. Delegates to xpbd_solve_inplace."""
    import torch
    from track_gen._src import warp_relax
    from track_gen._src import warp_pipeline as wpl
    E, N, _ = center0.shape
    n_max = N
    dev = str(center0.device)
    flat = E * n_max

    cw_in = wp.from_torch(center0.reshape(flat, 2).contiguous(), dtype=wp.vec2f)
    bw = wp.from_torch(band.to(torch.int32).contiguous(), dtype=wp.int32)
    lw = wp.from_torch(L0.to(torch.float32).contiguous(), dtype=wp.float32)
    if count is None:
        count_t = torch.full((E,), N, device=center0.device, dtype=torch.int32)
    else:
        count_t = count.to(torch.int32).contiguous()
    count_wp = wp.from_torch(count_t, dtype=wp.int32)
    out_wp = wp.empty(flat, dtype=wp.vec2f, device=dev)
    db_wp = wp.empty(flat, dtype=wp.vec2f, device=dev)
    warp_relax.xpbd_solve_inplace(cw_in, out_wp, db_wp, bw, lw, count_wp, n_max, config)
    return wp.to_torch(out_wp).view(E, n_max, 2)


def separation_disp(center, band, target):
    """Torch-in/out: fused Warp separation disp [E, N, 2] float32 (CUDA only)."""
    import torch
    E, N, _ = center.shape
    cf = wp.from_torch(center.reshape(E * N, 2).contiguous(), dtype=wp.vec2f)
    bw = wp.from_torch(band.to(torch.int32).contiguous(), dtype=wp.int32)
    out_t = torch.empty(E * N, 2, device=center.device, dtype=torch.float32)
    ow = wp.from_torch(out_t, dtype=wp.vec2f)
    from track_gen._src.warp_relax import _sep_kernel, _INITED
    if not _INITED:
        wp.init()
    wp.launch(_sep_kernel, dim=E * N, inputs=[cf, bw, N, float(target), ow],
              device=str(center.device))
    import torch as _torch
    _torch.cuda.synchronize()
    return out_t.view(E, N, 2)
