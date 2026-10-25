"""Private numeric reference wrappers for pre-ops test inputs."""
from __future__ import annotations
from track_gen._src import warp_pipeline as _implementation
from track_gen._src.runtime import _init, _sync
from track_gen.ops import resample as _ops_resample
from track_gen.ops.resample import *
from track_gen.ops.resample import ArcScratch as _ArcScratch
from track_gen.ops.geometry import _self_intersections_launch, _turning_launch
from track_gen.ops.fill import _fill_f32_k, _fill_i32_k

globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})

def self_intersections_inplace(poly_wp, count_wp, out_wp, n_max,
                               ctx: "LaunchContext | None" = None):
    """In-place: writes [E] int32 crossing counts into out_wp. Zero alloc.
    poly_wp: [E*n_max] vec2f; count_wp: [E] int32; out_wp: [E] int32.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    dev = str(out_wp.device)
    ctx = _resolve_ctx(ctx, dev)
    _self_intersections_launch(ctx, poly_wp, count_wp, n_max, out_wp)
    ctx.sync()

def _arc_resample_inplace(points_wp, M, num, real_wp, seg_wp, s_wp, count_r_wp, count_out_wp, out_wp, dev,
                          ctx: "LaunchContext | None" = None):
    """In-place NaN-aware arc-length resample. All args are wp.arrays. Zero alloc.

    points_wp: [E*M] vec2f input; real_wp: [E*M] vec2f scratch; seg_wp: [E*M] float32 scratch;
    s_wp: [E*(M+1)] float32 scratch; count_r_wp: [E] int32 scratch; count_out_wp: [E] int32 output;
    out_wp: [E*num] vec2f output; dev: warp device string.
    ctx: optional track_gen.exec.LaunchContext; default resolves the
        device's stream (see track_gen.exec).
    """
    ctx = _resolve_ctx(ctx, dev)
    _ops_resample._arc_resample_launch(
        ctx, points_wp, M, num, _ArcScratch(real_wp, seg_wp, s_wp, count_r_wp),
        out_wp, count_out_wp, E=points_wp.shape[0] // M)

def arc_length_resample_inplace(
    points_wp: "wp.array",
    M: int,
    num: int,
    real_wp: "wp.array",
    seg_wp: "wp.array",
    s_wp: "wp.array",
    count_r_wp: "wp.array",
    count_out_wp: "wp.array",
    out_wp: "wp.array",
    dev: str,
    ctx: "LaunchContext | None" = None,
) -> None:
    """Public in-place NaN-aware arc-length resample. All args are wp.arrays. Zero alloc.

    Identical to _arc_resample_inplace (which remains for internal use); this public name
    allows tests + the standalone generate path to call it without using the private alias.

    points_wp:   [E*M] vec2f input (may contain NaN).
    real_wp:     [E*M] vec2f scratch (compacted real points).
    seg_wp:      [E*M] float32 scratch (per-segment lengths).
    s_wp:        [E*(M+1)] float32 scratch (cumulative arc length).
    count_r_wp:  [E] int32 scratch (real-point count per env).
    count_out_wp:[E] int32 output (R>=2 -> num, else 0).
    out_wp:      [E*num] vec2f output.
    dev:         Warp device string.
    ctx: optional track_gen.exec.LaunchContext; default resolves the
        device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, dev)
    _arc_resample_inplace(points_wp, M, num, real_wp, seg_wp, s_wp, count_r_wp, count_out_wp, out_wp, dev,
                          ctx=ctx)
    ctx.sync()

def _arc_resample_selected_inplace(
    points_wp, active_wp, M, num, real_wp, seg_wp, s_wp, count_r_wp, count_out_wp, out_wp, dev,
    ctx: "LaunchContext | None" = None,
):
    """Selected NaN-aware arc-length resample (launches only). Zero alloc.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    ctx = _resolve_ctx(ctx, dev)
    _ops_resample._arc_resample_launch(
        ctx, points_wp, M, num, _ArcScratch(real_wp, seg_wp, s_wp, count_r_wp),
        out_wp, count_out_wp, E=points_wp.shape[0] // M, active=active_wp)

def arc_length_resample_selected_inplace(
    points_wp: "wp.array",
    active_wp: "wp.array",
    M: int,
    num: int,
    real_wp: "wp.array",
    seg_wp: "wp.array",
    s_wp: "wp.array",
    count_r_wp: "wp.array",
    count_out_wp: "wp.array",
    out_wp: "wp.array",
    dev: str,
    ctx: "LaunchContext | None" = None,
) -> None:
    """Selected NaN-aware arc-length resample. Inactive envs leave output untouched.

    Args:
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).
    """
    _init()
    ctx = _resolve_ctx(ctx, dev)
    _arc_resample_selected_inplace(
        points_wp, active_wp, M, num, real_wp, seg_wp, s_wp,
        count_r_wp, count_out_wp, out_wp, dev, ctx=ctx,
    )
    ctx.sync()

def turning_number_inplace(
    center_wp: "wp.array",
    n_max: int,
    count_wp: "wp.array",
    out_wp: "wp.array",
    ctx: "LaunchContext | None" = None,
) -> None:
    """In-place signed total turning of each closed polygon. Zero alloc.

    center_wp: [E*n_max] vec2f flat input centerline (NaN-padded beyond count[e]).
    n_max:     int stride per env in the flat array.
    count_wp:  [E] int32 real point count per env.
    out_wp:    [E] float32 output (signed turning in radians per env).
    ctx: optional track_gen.exec.LaunchContext; default resolves the
        device's stream (see track_gen.exec).

    Pure Warp (cpu+cuda); matches geometry.turning_number to allclose(atol=1e-4).
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_wp.device))
    _turning_launch(ctx, center_wp, count_wp, n_max, out_wp)
    ctx.sync()

def resample_uniform(
    center_wp: "wp.array",
    out_wp: "wp.array",
    n: int,
    count_wp: "wp.array",
    seg_wp: "wp.array | None" = None,
    s_wp: "wp.array | None" = None,
    device: str = "cpu",
    ctx: "LaunchContext | None" = None,
) -> None:
    """Arc-length-uniform resample of each closed loop to n points — in-place.

    Matches track_gen.relaxation._resample_uniform within FP tolerance (~1e-4).
    Two Warp kernels: scan (one thread per env, builds seg+cumulative s from points)
    then lookup (one thread per output point, linear-scan searchsorted + lerp).

    The owned pipeline path (``seg_wp`` and ``s_wp`` pre-allocated) is zero-alloc and
    safe inside a CUDA graph capture region. When ``seg_wp`` or ``s_wp`` is ``None``
    those arrays are allocated internally (not zero-alloc).

    Args:
        center_wp: [E*n_max] wp.vec2f flat input centerline.
        out_wp:    [E*n_max] wp.vec2f flat output buffer (written in-place).
        n:         output point count per env (== n_max, the buffer stride).
        count_wp:  [E] wp.int32 real point count per env.
        seg_wp:    [E*n_max] wp.float32 scan scratch (allocated internally if None).
        s_wp:      [E*(n_max+1)] wp.float32 scan scratch (allocated internally if None).
        device:    Warp device string (e.g. "cpu", "cuda:0").
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).

    PARITY INVARIANT: count=full((E,), N) reproduces the former fixed-N behaviour
    bit-exactly.  Pure Warp (cpu+cuda), zero compute outside of Warp kernels.
    """
    _init()
    ctx = _resolve_ctx(ctx, device)
    E = count_wp.shape[0]
    n_max = n
    flat = E * n_max

    if seg_wp is None:
        seg_wp = wp.empty(flat, dtype=wp.float32, device=device)
    if s_wp is None:
        s_wp = wp.empty(E * (n_max + 1), dtype=wp.float32, device=device)

    _ops_resample._resample_uniform_launch(ctx, center_wp, count_wp, n_max, seg_wp, s_wp,
                                           out_wp)

def resample_constant_spacing(
    center: "wp.array",
    spacing: float,
    n_max: int,
    out_wp: "wp.array | None" = None,
    count_wp: "wp.array | None" = None,
    seg_wp: "wp.array | None" = None,
    s_wp: "wp.array | None" = None,
    truncated_wp: "wp.array | None" = None,
    ctx: "LaunchContext | None" = None,
) -> "tuple[wp.array, wp.array] | None":
    """Arc-length resample each fully-real closed loop to constant ``spacing``, padded to
    ``n_max`` with NaN.  Matches ``geometry.arc_length_resample(points, spacing=spacing,
    n_max=n_max)``. Pure Warp (cpu+cuda).

    ``center`` must be a ``wp.array [E*N] vec2f`` flat.

    In-place mode (``out_wp`` and ``count_wp`` provided):
        ``out_wp``       — [E*n_max] wp.vec2f written in-place (center output).
        ``count_wp``     — [E] wp.int32 written in-place (real point count per env).
        ``seg_wp``       — [E*N] wp.float32 scan scratch.
        ``s_wp``         — [E*(N+1)] wp.float32 scan scratch.
        ``truncated_wp`` — [E] wp.int32 written in-place (1 where the env was truncated).
        Returns ``None`` (caller reads out_wp / count_wp / truncated_wp directly). With
        all five buffers provided this path is zero-alloc, does no host readback, and is
        safe inside a CUDA graph capture region; scratch it is not given is allocated.

    Allocating-mode (``out_wp`` or ``count_wp`` omitted -> allocated internally):
        Returns ``(out_wp, count_wp)`` — both freshly-allocated wp.arrays. This
        branch allocates and reads ``truncated`` back to the host; do not use inside a
        CUDA graph capture region.

    ``count[e]`` is capped at ``n_max`` (required for fixed-buffer sizing). An env whose
    true point count exceeds ``n_max`` is flagged ``truncated[e] = 1`` on device; the
    pipeline's validity gate then fails it. Allocating mode additionally emits a
    ``RuntimeWarning`` (a host readback); in-place mode never reads back. Choose
    ``N_max >= max(perimeter) / spacing + 1`` across all envs to avoid truncation.

    ``ctx``: optional track_gen.exec.LaunchContext; default resolves the device's stream
    (see track_gen.exec)."""
    _init()
    if count_wp is not None:
        E = count_wp.shape[0]
    elif out_wp is not None:
        E = out_wp.shape[0] // n_max
    else:
        raise ValueError(
            "resample_constant_spacing: need count_wp or out_wp to infer E from wp.array input")
    N = center.shape[0] // E
    dev = str(center.device)
    ctx = _resolve_ctx(ctx, dev)

    _allocating = out_wp is None or count_wp is None
    if out_wp is None:
        out_wp = wp.empty(E * n_max, dtype=wp.vec2f, device=dev)
    if count_wp is None:
        count_wp = wp.empty(E, dtype=wp.int32, device=dev)
    if seg_wp is None:
        seg_wp = wp.empty(E * N, dtype=wp.float32, device=dev)
    if s_wp is None:
        s_wp = wp.empty(E * (N + 1), dtype=wp.float32, device=dev)
    if truncated_wp is None:
        truncated_wp = wp.empty(E, dtype=wp.int32, device=dev)

    _ops_resample._constant_spacing_launch(ctx, center, N, spacing, n_max, seg_wp, s_wp,
                                           out_wp, count_wp, truncated_wp)
    ctx.sync()

    if _allocating:
        # Convenience path only (it allocates, so it is never on a capture path): warn
        # on host. The in-place path relies on the device-side truncated flag instead.
        import warnings
        if bool(truncated_wp.numpy().any()):
            warnings.warn(
                f"constant_spacing: a track needed more than N_max={n_max} points "
                f"(spacing={spacing}); it was truncated — increase N_max to avoid "
                f"truncation.",
                RuntimeWarning,
                stacklevel=2,
            )
        return out_wp, count_wp

from tests._reference_generators.warp_generate import (
    _generate as _generate_bezier, _alloc_scratch as _alloc_bezier,
    assemble_inplace, corner_sample_inplace, corner_count_sample_inplace, ccw_sort_inplace,
    _assemble_k, _vertex_tangents_k,
)

def _run_pipeline(
    config,
    seed_buf_wp: wp.array,
    out: "Track",
    scratch: "_Scratch",
    generator_spec=None,
    ctx: "LaunchContext | None" = None,
) -> "Track":
    """Execute the owned pure-Warp pipeline into pre-allocated buffers. Zero alloc.

    Frozen PR9 composition: generate the centerline, resample, relax, and inflate
    into preallocated buffers. Every generator receives ``ctx=ctx`` and every stage
    runs on that context's stream. The live facades run a pipeline ``Program``;
    this test utility retains the independent composition as its byte reference.

    Args:
        config:       TrackGenConfig.
        seed_buf_wp:  [E] int32 wp.array — per-env base seeds.
        out:          Pre-allocated Track (wp.array fields) — written in-place.
        scratch:      Pre-allocated _Scratch — all intermediates written in-place.
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).

    Returns:
        out (the same Track instance).
    """
    ctx = _resolve_ctx(ctx, str(config.device))
    if generator_spec is None:
        generator_spec = getattr(scratch, "generator_spec", None)
    if generator_spec is None:
        generator_spec = _registered_generator_spec(config.generator)
    _reject_stage_class(generator_spec)

    n_max = int(config.N_max)
    gen = scratch.gen        # generator-private scratch
    relax = scratch.relax    # RelaxScratch — band/L0 + XPBD buffers
    if out.truncated is None:
        # Without the flag buffer a truncated track would pass validity (fail open).
        raise ValueError("_run_pipeline requires out.truncated; allocate out via "
                         "_inflate_warp_alloc")

    # 1. Generate centerline in-place into the orchestrator-owned output buffers.
    generate = generator_spec.generate
    generate(seed_buf_wp, config, out_centerline=scratch.gen_centerline,
             out_valid_wp=scratch.gen_valid, scratch=gen, ctx=ctx)

    # 2. Constant-spacing resample (gen centerline -> bridge buffers).
    resample_constant_spacing(
        scratch.gen_centerline, float(config.spacing), n_max,
        out_wp=scratch.cs_center, count_wp=scratch.count,
        seg_wp=scratch.cs_seg, s_wp=scratch.cs_s, truncated_wp=out.truncated,
        ctx=ctx,
    )

    # 3-4. Relaxation (band/L0 setup + XPBD solve) in-place, then choose its input to
    # inflate. relax_enable=False is an identity pass-through (matches the oracle's
    # `if not relax_enable: return center`): skip the relax band/L0 + solve and inflate
    # the constant-spacing centerline directly. The branch is resolved at capture time,
    # so the captured graph is fixed and allocation-free either way.
    if config.relax_enable:
        # 3. Relaxation setup: band / L0 (the relax module owns its own setup).
        warp_relax.band_l0_inplace(
            scratch.cs_center, n_max, relax.band, relax.L0,
            scratch.count, config, ctx=ctx,
        )

        # 4. XPBD relaxation in-place.
        warp_relax.xpbd_solve_inplace(
            scratch.cs_center, relax.relaxed, relax.xpbd_db,
            relax.band, relax.L0, scratch.count, n_max, config,
            sep_cache_idx_wp=relax.sep_cache_idx,
            sep_cache_count_wp=relax.sep_cache_count,
            sep_cache_overflow_wp=relax.sep_cache_overflow,
            cheb_prev_wp=relax.cheb_prev,
            ctx=ctx,
        )
        relax_out = relax.relaxed
    else:
        relax_out = scratch.cs_center

    # 5. Inflate (resample_uniform + frame + offset + validity) in-place. _inflate_warp
    # reads scratch.inflate (kappa/w/area_a/area_b/z + staging) and the bridge buffers
    # (cs_seg/cs_s/count) off the composite scratch.
    return _inflate_warp(
        relax_out, config, out=out, valid=scratch.gen_valid,
        count=scratch.count, scratch=scratch, seeds=seed_buf_wp,
        truncated=out.truncated, ctx=ctx,
    )

def _inflate_warp(center, config, out=None,
                 valid: "wp.array | None" = None,
                 count: "wp.array | None" = None,
                 scratch: "_Scratch | None" = None,
                 seeds: "wp.array | None" = None,
                 truncated: "wp.array | None" = None,
                 ctx: "LaunchContext | None" = None):
    """Pure-Warp inflation: resample -> frame -> offset -> z-profile -> validity -> lift.

    The tail of the PR9 composition (see :func:`_run_pipeline`); the
    ``track_gen.pipeline`` built-in stages wrap the same calls one stage each.

    Writes results into out's wp.array Track buffers (or allocates a fresh Track).
    All inputs are wp.arrays; center is a flat [E*n_max] vec2f wp.array.

    The 2D stages operate on the vec2f staging buffers (scratch.inflate.ctr2/
    tan2/nrm2/out2/inn2); a final lift kernel copies them into the public vec3f
    Track buffers. On the flat default (``config.z_profile == "flat"``) the lift
    writes a constant z (``config.z_base``, 0.0 by default) and the arclen/length/
    tangent tables stay the byte-identical 2D values from stages 1-5. On a non-flat
    profile the per-point altitude ``scratch.inflate.z`` is filled by
    ``warp_zprofile.apply_z_profile`` (from the 2D ``(cum, perim)`` tables still in
    out.arclen/out.length), the three polylines are lifted level to that z, and
    ``_track_frames3_k`` recomputes the tangent + true 3D arclen/length (overwriting
    the 2D tables). A non-flat profile REQUIRES ``seeds`` (the standalone path is
    flat-only otherwise).
    Scratch (area_a/area_b/kappa/w/z + staging) is passed in via scratch (zero
    allocation on owned path) or allocated here (the out=None / standalone path).

    Constant-spacing path (count provided — the pipeline always uses this):
        center is [E*n_max] vec2f flat, NaN-padded (real points in [0, count[e])).
        count is threaded into every sub-stage; Track.count = count.

    count=None convenience path (generic fixed-N, fully-finite centerline):
        Requires center to be [E*N] vec2f with no NaN and center.shape[0]//E == N ==
        config.num_points. All sub-stages run with full count (count[e] == N).

    Args:
        center:   [E*n_max] vec2f wp.array (flat).
        config:   TrackGenConfig.
        out:      Optional pre-allocated Track. Allocated fresh when None.
        valid:    [E] int32 wp.array generation flag. Defaults to all-1 when None.
        count:    [E] int32 wp.array real point count. None -> fixed-N path.
        scratch:  Optional _Scratch. Required when out is provided (owned path).
        seeds:    [E] int32 wp.array per-env base seeds for the z profiler. Required
                  when config.z_profile != "flat"; ignored on the flat path.
        truncated: [E] int32 wp.array N_max truncation flags from
                  resample_constant_spacing; flagged envs fail validity. None -> no env
                  is treated as truncated (a zero buffer is allocated).
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).

    Returns:
        track_gen.types.Track with wp.array fields.

    Raises:
        ValueError: if config.z_profile != "flat" and seeds is None (the standalone
            inflate path is flat-only unless seeds are provided).
    """
    from track_gen._src.types import Layout, Track  # private test reference

    _init()

    # A non-flat elevation profile needs per-env seeds to drive the profiler; the
    # standalone inflate path (seeds omitted) supports only the flat profile.
    _z_profile = str(getattr(config, "z_profile", "flat"))
    if _z_profile != "flat" and seeds is None:
        raise ValueError(
            "z_profile != 'flat' requires seeds= (the standalone inflate path is "
            "flat-only otherwise)"
        )

    # --- Resolve E, n_max, dev from wp.array center ---
    if count is not None:
        E = count.shape[0]
    else:
        # count=None convenience: infer E from out (if provided) or assume square
        # (caller must ensure center.shape[0] is divisible by E == num_envs).
        if out is not None:
            E = out.valid.shape[0]
        else:
            # Fall back: n_max == num_points, E = total / n_max.
            n_pts = int(config.num_points)
            E = center.shape[0] // n_pts

    n_max = center.shape[0] // E
    dev = str(center.device)
    ctx = _resolve_ctx(ctx, dev)
    hw = float(config.half_width)
    flat = E * n_max

    # --- Allocate Track + scratch when not pre-provided ---
    if out is None:
        out = Track(
            outer=wp.empty(flat, dtype=wp.vec3f, device=dev),
            center=wp.empty(flat, dtype=wp.vec3f, device=dev),
            inner=wp.empty(flat, dtype=wp.vec3f, device=dev),
            tangent=wp.empty(flat, dtype=wp.vec3f, device=dev),
            normal=wp.empty(flat, dtype=wp.vec3f, device=dev),
            arclen=wp.empty(flat, dtype=wp.float32, device=dev),
            length=wp.empty(E, dtype=wp.float32, device=dev),
            valid=wp.empty(E, dtype=wp.int32, device=dev),
            count=wp.empty(E, dtype=wp.int32, device=dev),
            winding=wp.empty(E, dtype=wp.float32, device=dev),
            truncated=(truncated if truncated is not None else
                       wp.zeros(E, dtype=wp.int32, device=dev)),
            generation=wp.zeros(E, dtype=wp.int32, device=dev),
            fail_mask=wp.zeros(E, dtype=wp.int32, device=dev),
            half_width=wp.empty(flat, dtype=wp.float32, device=dev),
            layout=Layout(E, n_max, dev),
        )
        if scratch is None:
            # Standalone path: only the inflate group is needed (no gen/relax buffers);
            # cs_seg/cs_s left None so _inflate_warp allocates its own resample scan scratch.
            scratch = _Scratch(
                inflate=InflateScratch(
                    area_a=wp.zeros(E, dtype=wp.float32, device=dev),
                    area_b=wp.zeros(E, dtype=wp.float32, device=dev),
                    kappa=wp.empty(flat, dtype=wp.float32, device=dev),
                    w=wp.empty(flat, dtype=wp.float32, device=dev),
                    out2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    ctr2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    inn2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    tan2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    nrm2=wp.empty(flat, dtype=wp.vec2f, device=dev),
                    z=wp.empty(flat, dtype=wp.float32, device=dev),
                ),
            )
            # Zero on ctx's stream (wp.zeros memsets Warp's current stream): the flat
            # profile never writes z and the grade check reads it.
            ctx.fill(scratch.inflate.z, 0.0)
    else:
        # Owned path: the caller MUST pass pre-allocated scratch.
        assert scratch is not None, (
            "_inflate_warp(out=...) requires a pre-allocated scratch=_Scratch(...) "
            "(zero-allocation contract); pass scratch alongside out."
        )

    # The 2D pipeline kernels write the vec2f staging buffers; the lift at the
    # end copies them into the public vec3f Track arrays with z = 0.
    stag = scratch.inflate
    stag.fail = out.fail_mask
    stag.hw_out = out.half_width
    if truncated is None:
        ctx.fill(out.truncated, 0)
    elif truncated is not out.truncated:
        ctx.copy(out.truncated, truncated)
    if stag.ctr2 is None:
        stag.out2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
        stag.ctr2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
        stag.inn2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
        stag.tan2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
        stag.nrm2 = wp.empty(flat, dtype=wp.vec2f, device=dev)
    if stag.z is None:
        stag.z = wp.empty(flat, dtype=wp.float32, device=dev)
        ctx.fill(stag.z, 0.0)
    if stag.knot_cum is None:
        _fill_knot_scratch(stag, config, E, dev)

    # --- Allocate resample scan scratch (seg/s) from _Scratch or standalone ---
    if scratch.cs_seg is not None and scratch.cs_s is not None:
        rs_seg_wp = scratch.cs_seg
        rs_s_wp = scratch.cs_s
    else:
        rs_seg_wp = None
        rs_s_wp = None

    if count is None:
        # --- count=None convenience: fixed-N, fully-finite centerline ---
        assert n_max == int(config.num_points), "center N must equal config.num_points"
        N = n_max

        # Build a full count wp.array (== N for all envs).
        cnt_wp = wp.empty(E, dtype=wp.int32, device=dev)
        ctx.launch(_fill_i32_k, E, [cnt_wp, N])

        # 1. resample_uniform: writes into the vec2f center staging buffer.
        resample_uniform(center, stag.ctr2, N, cnt_wp,
                         seg_wp=rs_seg_wp, s_wp=rs_s_wp, device=dev, ctx=ctx)
        ctx.sync()

        # Per-point half-width kernel-filled into the inflate group's w.
        ctx.launch(_fill_f32_k, flat, [stag.w, hw])

        # Track.count == N for all envs.
        ctx.launch(_fill_i32_k, E, [out.count, N])

    else:
        # --- Constant-spacing path: variable count per env, NaN-padded output ---
        cnt_wp = count  # wp.int32 array

        # 1. resample_uniform: writes into the vec2f center staging buffer.
        resample_uniform(center, stag.ctr2, n_max, cnt_wp,
                         seg_wp=rs_seg_wp, s_wp=rs_s_wp, device=dev, ctx=ctx)
        ctx.sync()

        # Per-point half-width kernel-filled into the inflate group's w.
        ctx.launch(_fill_f32_k, flat, [stag.w, hw])

        # Track.count == per-env real point count.
        ctx.copy(out.count, cnt_wp)

    # 2. frame + curvature — into the vec2f tangent/normal staging buffers.
    frame_curvature(stag.ctr2, stag.tan2, stag.nrm2, stag.kappa, cnt_wp, ctx=ctx)

    # 3. cumulative arc length + total length — in-place into out.arclen/out.length.
    _arclength(stag.ctr2, out.arclen, out.length, cnt_wp, ctx=ctx)

    # 3b. signed loop winding (+1 CCW / -1 CW / 0 degenerate) — in-place into out.winding.
    _winding(stag.ctr2, out.winding, cnt_wp, ctx=ctx)

    # 4. offset to outer/inner borders — into the vec2f outer/inner staging buffers.
    offset(stag.ctr2, stag.nrm2, hw, stag.out2, stag.inn2,
           stag.area_a, stag.area_b, cnt_wp, ctx=ctx,
           hw_out=getattr(stag, "hw_out", None))

    # 4b. Elevation (2.5D): fill the per-point altitude from the configured profile.
    # CRITICAL ordering: out.arclen/out.length still hold the 2D (cum, perim) tables
    # from stage 3 here — the profiler consumes them as its plan-view arc
    # parameterization, BEFORE _track_frames3_k (stage 6, non-flat) overwrites them
    # with 3D values. Skipped on the flat path (stag.z stays zero-filled), which
    # keeps the arclen/length/tangent tables byte-identical to the legacy 2D path.
    #
    # uniform/random_walk decide altitude at z_control_points arc-spaced KNOTS and
    # interpolate with a periodic monotone cubic, so smoothness is set by the knot
    # count rather than the resampled point count. noise is analytically smooth
    # already (its harmonics band-limit it), so it stays per-point.
    _is_flat = _z_profile == "flat"
    if _z_profile in ("uniform", "random_walk"):
        warp_zprofile._apply_knots(
            config, seeds, cnt_wp, n_max, out.arclen, out.length,
            stag.knot_cum, stag.knot_count, stag.knot_z, stag.z, ctx=ctx)
    elif not _is_flat:
        warp_zprofile._apply(
            config, seeds, cnt_wp, n_max, out.arclen, out.length, stag.z, ctx=ctx)

    # 5. per-track validity gate (in-place: writes directly into out.valid).
    if valid is not None:
        gv_wp = valid  # wp.int32 array
    else:
        # Standalone / test path: allocate a small all-ones temp.
        gv_wp = wp.empty(E, dtype=wp.int32, device=dev)
        ctx.launch(_fill_i32_k, E, [gv_wp, 1])

    # Border self_intersections is optional (config.validity_border_check, default off).
    _bc = getattr(config, "validity_border_check", False)
    has_border = 1 if _bc else 0

    # validity_inplace writes into out.valid directly. stag.z drives the grade
    # gate; on the flat path z is zero-filled and z_valid_grade defaults to 0 -> the
    # grade branch is a no-op and validity is byte-identical to the legacy path.
    validity_inplace(stag.ctr2, stag.w, cnt_wp, gv_wp,
                     stag.out2, stag.inn2, has_border, n_max, stag.z,
                     out.valid, config, truncated_wp=out.truncated, ctx=ctx,
                     fail_wp=stag.fail)

    # 6. Lift the vec2f staging buffers into the public vec3f Track arrays.
    if _is_flat:
        # Flat: constant z = z_base (0.0 by default). arclen/length/tangent keep the
        # 2D values from stages 2-3 (a constant z-offset changes neither). NaN xy
        # padding lifts to NaN-xy.
        ctx.launch(_lift_track_k, flat,
                   [stag.out2, stag.ctr2, stag.inn2, stag.tan2, stag.nrm2,
                    float(config.z_base),
                    out.outer, out.center, out.inner, out.tangent,
                    out.normal])
    else:
        # Non-flat: level cross-sections share the per-point centerline z, then
        # recompute tangent + true 3D arclen/length (overwriting the 2D tables).
        ctx.launch(_lift_track_zvar_k, flat,
                   [stag.out2, stag.ctr2, stag.inn2, stag.nrm2, stag.z,
                    out.outer, out.center, out.inner, out.normal])
        ctx.launch(_track_frames3_k, E,
                   [out.center, cnt_wp, n_max,
                    out.tangent, out.arclen, out.length])
    ctx.sync()

    return out

from tests._reference_generators import spec as _scalar_spec
from tests._reference_generators import warp_zprofile
from track_gen.ops.geometry import _thickness_k
from track_gen._src.warp_generate import _corner_angles_gate_k, _gates_combine_k


def _inflate_warp_alloc(config, generator_spec=None, *, seed=None):
    if generator_spec is None:
        generator_spec = _scalar_spec(config.generator)
    return _implementation._inflate_warp_alloc(config, generator_spec, seed=seed)
