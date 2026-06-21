"""Pure-Warp segment-grammar centerline generator (registered as ``"grammar"``).

Method (#6): sample a net-winding kappa grammar (alternating straight + corner segments),
rasterize it to a dense kappa array, close the heading by SCALING the curvature (preserves
kappa=0 straights — the additive DC-shift does NOT, and produced round blobs), integrate
theta/edges, close displacement by subtracting the mean edge (gap-distribution close), and
normalize the resulting centerline to the same coordinate range as the other generators.

CLOSURE NOTE (match grammar_proto.py exactly): the heading is closed by SCALING kappa
(kappa *= 2*pi / net_turn), NOT adding a DC offset. Scaling leaves kappa==0 spans at zero
so STRAIGHTS stay straight; a DC shift would arc them. A cap (_HEADING_SCALE_CAP) prevents
near-zero net windings from amplifying the curve into a self-crossing knot.

Hard invariants (shared with all of ``track_gen/_src``): Warp-native + torch-free,
ZERO per-call allocation (all buffers come from ``grammar_alloc_scratch``), CUDA-graph-
capturable (fixed bounds over ``S`` and ``N``, no host sync, no per-env Python branching),
deterministic in (per-env seed, config).

RNG salt: ``_GRAMMAR_SALT = 6271`` — distinct from polar (7919), bezier (9781/6151),
hull (7919/5119/3083).
"""
from __future__ import annotations

import warp as wp

from . import warp_pipeline as _pipe

# Match warp_generate_polar._BEZIER_EXTENT so the coordinate range is comparable.
_BEZIER_EXTENT = 1.44

# Distinct RNG salt (large odd, unused by polar/bezier/hull).
_GRAMMAR_SALT = 6271

# Cap on the heading-closure scale factor 2*pi/net. A near-zero net winding would explode
# this factor and produce a knot; clamping it bounds the amplification while leaving the
# small residual heading gap for XPBD to repair.
_HEADING_SCALE_CAP = 2.0


# Stride for subsampling the grammar centerline before angle-sort. Taking every
# _POLY_STRIDE-th point from the N dense centerline gives Np = N // _POLY_STRIDE
# sparse control points. These are then angle-sorted into a simple polygon and
# arc-resampled back to N, matching hull's pattern (P sparse -> angle-sort -> resample N).
# With N=128 and stride=8, Np=16 — similar to hull's typical P=10-20 control points.
_POLY_STRIDE = 8


@wp.kernel
def _grammar_angle_sort_k(
    points: wp.array(dtype=wp.vec2f),
    N: int,
    stride: int,
    Np: int,
    keys: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Stride-subsample the N grammar centerline by taking every
    # `stride`-th point (indices 0, stride, 2*stride, ...) giving Np = N//stride points.
    # Then angle-sort those Np points ascending by atan2(dx, dy) (X first, matching the
    # bezier/hull ccw-sort convention) about their centroid. Insertion sort — O(Np^2)
    # but Np<=32 so this is fast. The Np-point result in `out` is then arc-resampled back
    # to N, producing a smooth simple polygon like hull's sparse-point fallback.
    e = wp.tid()
    n_base = e * N
    p_base = e * Np

    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    for c in range(Np):
        p = points[n_base + c * stride]
        sx = sx + wp.float64(p[0])
        sy = sy + wp.float64(p[1])
    cx = wp.float32(sx / wp.float64(Np))
    cy = wp.float32(sy / wp.float64(Np))

    for c in range(Np):
        p = points[n_base + c * stride]
        key = wp.atan2(p[0] - cx, p[1] - cy)   # X first!
        j = c - 1
        while j >= 0 and keys[p_base + j] > key:
            keys[p_base + j + 1] = keys[p_base + j]
            out[p_base + j + 1] = out[p_base + j]
            j = j - 1
        keys[p_base + j + 1] = key
        out[p_base + j + 1] = p


@wp.kernel
def _grammar_normalize_k(
    points: wp.array(dtype=wp.vec2f),
    N: int,
    target_extent: float,
):
    # One thread per env e. Center by bbox and scale so each env's longest bbox dimension
    # exactly matches target_extent. Mirrors _normalize_centerline_k from warp_generate_polar.
    e = wp.tid()
    b = e * N
    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)

    for i in range(N):
        p = points[b + i]
        min_x = wp.min(min_x, p[0])
        max_x = wp.max(max_x, p[0])
        min_y = wp.min(min_y, p[1])
        max_y = wp.max(max_y, p[1])

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)

    for i in range(N):
        p = points[b + i]
        points[b + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)


@wp.kernel
def _grammar_sample_k(
    seeds: wp.array(dtype=wp.int32),
    S: int,
    sharp: float,
    straight_frac: float,
    chicane_bias: float,
    hairpin_max_frac: float,
    segments: wp.array(dtype=wp.float32),
):
    """One thread per env ``e``.

    Writes ``S`` rows into ``segments[e*S*3 : (e+1)*S*3]`` as flat float32 triples
    ``(kappa_start, kappa_end, length_frac)``. Alternates a straight (kappa=0) and a corner
    for ``n_corner = max(2, S//2)`` corners. All corners are generated positive first, then
    exactly ``n_neg = round(n_corner * chicane_bias)`` unique corners are flipped negative
    via rejection sampling (Pass 5). This guarantees a net turn > pi so the heading-closure
    scale cap rarely fires and self-intersection rate stays low. Finally, the straight/corner
    length split is biased toward ``straight_frac`` and length-fracs are normalised to sum 1.
    """
    e = wp.tid()
    state = wp.rand_init(seeds[e] * _GRAMMAR_SALT)
    base = e * S * 3
    n_corner = wp.max(S // 2, 2)

    # Pass 1: fill the S segment rows with ALL-POSITIVE corners.
    # Exactly n_neg = round(n_corner * chicane_bias) corners will be flipped negative
    # in Pass 5 (rejection-sampling flip), ensuring a consistent net turn > pi so that
    # the heading-closure scale cap (_HEADING_SCALE_CAP=2.0) rarely fires and SI rate
    # stays low. Generating all-positive first guarantees k > 0 is the "not yet flipped"
    # sentinel used by the flip pass.
    for c in range(n_corner):
        si = c * 2        # straight index
        ci = c * 2 + 1    # corner index
        if si < S:
            # Straight: kappa_start = kappa_end = 0, random length in [0.06, 0.22]
            ln_s = wp.randf(state, 0.06, 0.22)
            segments[base + si * 3 + 0] = float(0.0)
            segments[base + si * 3 + 1] = float(0.0)
            segments[base + si * 3 + 2] = ln_s
        if ci < S:
            # Corner: turn angle in [0.25, sharp], length in [0.02, hairpin_max_frac].
            # Always positive — chicane negation happens in Pass 5.
            ang = wp.randf(state, 0.25, sharp)
            ln_c = wp.randf(state, 0.02, hairpin_max_frac)
            safe_ln = wp.max(ln_c, float(1.0e-6))
            k = ang / safe_ln  # ALWAYS POSITIVE (no sgn)
            # 40% probability: linear-ramp clothoid (kappa_start=0 -> kappa_end=k)
            if wp.randf(state) < float(0.4):
                segments[base + ci * 3 + 0] = float(0.0)
                segments[base + ci * 3 + 1] = k
            else:
                # constant-kappa arc
                segments[base + ci * 3 + 0] = k
                segments[base + ci * 3 + 1] = k
            segments[base + ci * 3 + 2] = ln_c

    # Pass 2: sum straight lengths and corner lengths separately.
    sum_straight = float(0.0)
    sum_corner = float(0.0)
    for i in range(S):
        ks = segments[base + i * 3 + 0]
        ke = segments[base + i * 3 + 1]
        ln = segments[base + i * 3 + 2]
        # A segment is straight iff kappa_start == kappa_end == 0.0
        is_straight = int(0)
        if ks == float(0.0) and ke == float(0.0):
            is_straight = int(1)
        if is_straight == int(1):
            sum_straight = sum_straight + ln
        else:
            sum_corner = sum_corner + ln

    # Pass 3: rescale each group toward straight_frac / (1-straight_frac) split,
    # then normalise all length_fracs to sum 1.
    corner_frac = float(1.0) - straight_frac
    if sum_straight > float(1.0e-9) and sum_corner > float(1.0e-9):
        for i in range(S):
            ks = segments[base + i * 3 + 0]
            ke = segments[base + i * 3 + 1]
            ln = segments[base + i * 3 + 2]
            is_straight = int(0)
            if ks == float(0.0) and ke == float(0.0):
                is_straight = int(1)
            if is_straight == int(1):
                segments[base + i * 3 + 2] = ln * straight_frac / sum_straight
            else:
                segments[base + i * 3 + 2] = ln * corner_frac / sum_corner

    # Pass 4: normalise length_fracs so they sum to 1.
    total_len = float(0.0)
    for i in range(S):
        total_len = total_len + segments[base + i * 3 + 2]
    inv_total = float(1.0) / wp.max(total_len, float(1.0e-9))
    for i in range(S):
        segments[base + i * 3 + 2] = segments[base + i * 3 + 2] * inv_total

    # Pass 5: flip exactly n_neg unique corners negative (chicane / S-bend).
    # Uses rejection sampling: draw a random corner index; if its kappa_end is still
    # positive (not yet flipped), flip both kappa_start and kappa_end and increment
    # already_flipped. Retry up to n_corner*3 times so the loop has a fixed bound
    # compatible with Warp / CUDA-graph capture. n_neg = round(n_corner * chicane_bias).
    n_neg = wp.max(int(1), int(float(n_corner) * chicane_bias + float(0.5)))
    already_flipped = int(0)
    max_tries = n_corner * int(3)
    for _try in range(max_tries):
        if already_flipped >= n_neg:
            break
        flip_c = int(wp.randf(state) * float(n_corner)) % n_corner
        ci = flip_c * int(2) + int(1)
        if ci < S:
            ke = segments[base + ci * 3 + int(1)]
            if ke > float(0.0):
                segments[base + ci * 3 + int(0)] = -segments[base + ci * 3 + int(0)]
                segments[base + ci * 3 + int(1)] = -segments[base + ci * 3 + int(1)]
                already_flipped = already_flipped + int(1)


@wp.kernel
def _grammar_build_k(
    segments: wp.array(dtype=wp.float32),
    S: int,
    N: int,
    rast_kappa: wp.array(dtype=wp.float32),
    raw: wp.array(dtype=wp.vec2f),
    out_centerline: wp.array(dtype=wp.vec2f),
    out_valid: wp.array(dtype=wp.int32),
):
    """One thread per env ``e``.

    1. Rasterise ``kappa[e*N : (e+1)*N]`` from the segment table (prefix-sum bounds +
       linear interp within each span — matches ``rasterize_kappa`` in grammar_proto).
    2. SCALING heading closure: ``net = sum(kappa*ds)``; if ``|net| > 1e-6``, multiply
       all kappa values by ``clamp(2*pi/net, -_HEADING_SCALE_CAP, +_HEADING_SCALE_CAP)``.
       This preserves kappa=0 straights exactly (scaling zeros stays zero).
    3. Integrate theta = cumsum(kappa)*ds, set theta[0] = 0, build edge vectors
       ``(cos(theta), sin(theta)) * ds``.
    4. Gap-distribution displacement closure: subtract mean edge from every edge so
       their vector sum is exactly zero (closed loop).
    5. Cumulative sum of closed-loop edges -> positions in ``raw[e*N : (e+1)*N]``.
    6. Copy raw positions to ``out_centerline``; set ``out_valid[e] = 1``.
       (``_normalize_centerline_k`` is launched separately to center + scale.)
    """
    e = wp.tid()
    seg_base = e * S * 3
    k_base = e * N
    r_base = e * N
    ds = float(1.0) / float(N)
    two_pi = float(2.0) * wp.pi

    # Build cumulative arc-length bounds for the S segments (S+1 values: bounds[0]=0,
    # bounds[i+1] = bounds[i] + length_frac[i]).  We compute on the fly with a running sum.

    # Pass 1: rasterize kappa from segment table.
    # For each sample point s = (i + 0.5) / N, find which segment it falls in by
    # comparing against prefix-sum boundaries, then linearly interpolate kappa_start/end.
    # We use a running-prefix approach: walk through segments once per env.
    bound = float(0.0)
    seg_idx = int(0)
    for i in range(N):
        s = (float(i) + float(0.5)) * ds
        # Advance seg_idx until the current segment covers s.
        # bound is the START of segment seg_idx.
        next_bound = bound + segments[seg_base + seg_idx * 3 + 2]
        for _adv in range(S):
            if s < next_bound or seg_idx >= S - 1:
                break
            bound = next_bound
            seg_idx = seg_idx + 1
            next_bound = bound + segments[seg_base + seg_idx * 3 + 2]
        # Interpolate within segment seg_idx.
        ks = segments[seg_base + seg_idx * 3 + 0]
        ke = segments[seg_base + seg_idx * 3 + 1]
        seg_len = segments[seg_base + seg_idx * 3 + 2]
        if seg_len > float(1.0e-9):
            u = wp.clamp((s - bound) / seg_len, float(0.0), float(1.0))
        else:
            u = float(0.0)
        rast_kappa[k_base + i] = (float(1.0) - u) * ks + u * ke

    # Pass 2: compute net = sum(rast_kappa * ds).
    net = float(0.0)
    for i in range(N):
        net = net + rast_kappa[k_base + i] * ds

    # Pass 3: scaling heading closure (preserves kappa=0 straights).
    if wp.abs(net) > float(1.0e-6):
        sc = two_pi / net
        sc = wp.clamp(sc, float(-_HEADING_SCALE_CAP), float(_HEADING_SCALE_CAP))
        for i in range(N):
            rast_kappa[k_base + i] = rast_kappa[k_base + i] * sc

    # Pass 4: integrate theta = cumulative (rast_kappa * ds), zero-referenced at index 0.
    # theta[i] = sum_{j=0}^{i-1} rast_kappa[j]*ds  ->  theta[0] = 0.
    # We compute this as a running sum and directly build edge vectors.
    theta = float(0.0)
    mean_ex = float(0.0)
    mean_ey = float(0.0)
    for i in range(N):
        ex = wp.cos(theta) * ds
        ey = wp.sin(theta) * ds
        raw[r_base + i] = wp.vec2f(ex, ey)
        mean_ex = mean_ex + ex
        mean_ey = mean_ey + ey
        theta = theta + rast_kappa[k_base + i] * ds

    mean_ex = mean_ex * ds   # divide by N (ds = 1/N, so sum * ds = sum/N)
    mean_ey = mean_ey * ds

    # Pass 5: gap-distribution closure — subtract mean edge so edges sum to zero.
    for i in range(N):
        e_vec = raw[r_base + i]
        raw[r_base + i] = wp.vec2f(e_vec[0] - mean_ex, e_vec[1] - mean_ey)

    # Pass 6: cumulative sum of edges -> positions.
    px = float(0.0)
    py = float(0.0)
    for i in range(N):
        e_vec = raw[r_base + i]
        px = px + e_vec[0]
        py = py + e_vec[1]
        out_centerline[k_base + i] = wp.vec2f(px, py)

    out_valid[e] = int(1)


class GrammarScratch:
    """Pre-allocated private working scratch for the ``"grammar"`` generator.

    segments:   [E*S*3] float32 — flat (kappa_start, kappa_end, length_frac) per segment.
    rast_kappa: [E*N]   float32 — rasterized curvature (reused for closure scaling in place).
                Named ``rast_kappa`` (not ``kappa``) to avoid shadowing ``InflateScratch.kappa``
                via ``_Scratch.__getattr__``; both slots would match "kappa" and GrammarScratch
                would win the name lookup, returning the wrong (smaller) buffer to inflate_warp.
    raw:        [E*N]   vec2f   — scratch for closed-loop edge vectors before cumsum.

    Polygon-fallback buffers (mirrors hull's crosser + arc-resample slots).
    The fallback stride-subsamples the N grammar points to Np = N//_POLY_STRIDE sparse
    control points before angle-sorting, matching hull's sparse-control-point pattern:
    crossers:   [E]        int32   — self-intersection counts after grammar centerline is built.
    poly:       [E*Np]     vec2f   — angle-sorted sparse polygon (Np = N//_POLY_STRIDE).
    sort_keys:  [E*Np]     float32 — angle-sort key scratch.
    arc_real:   [E*Np]     vec2f   — arc-resample compacted-real scratch.
    arc_seg:    [E*Np]     float32 — arc-resample per-segment length scratch.
    arc_s:      [E*(Np+1)] float32 — arc-resample cumulative arc-length scratch.
    arc_cr:     [E]        int32   — arc-resample real-point-count scratch.
    arc_co:     [E]        int32   — arc-resample output-count scratch.
    """

    __slots__ = (
        "segments", "rast_kappa", "raw",
        "crossers", "poly", "sort_keys",
        "arc_real", "arc_seg", "arc_s", "arc_cr", "arc_co",
    )

    def __init__(self, segments, rast_kappa, raw,
                 crossers, poly, sort_keys,
                 arc_real, arc_seg, arc_s, arc_cr, arc_co) -> None:
        self.segments = segments
        self.rast_kappa = rast_kappa
        self.raw = raw
        self.crossers = crossers
        self.poly = poly
        self.sort_keys = sort_keys
        self.arc_real = arc_real
        self.arc_seg = arc_seg
        self.arc_s = arc_s
        self.arc_cr = arc_cr
        self.arc_co = arc_co


def grammar_alloc_scratch(config):
    """Allocate the grammar generator's PRIVATE working scratch (one alloc per generator)."""
    _pipe._init()
    E = int(config.num_envs)
    S = int(config.grammar_segments)
    N = int(config.num_points)
    Np = max(N // _POLY_STRIDE, 3)  # sparse control-point count for angle-sort polygon
    dev = str(config.device)
    return GrammarScratch(
        segments=wp.empty(E * S * 3, dtype=wp.float32, device=dev),
        rast_kappa=wp.empty(E * N, dtype=wp.float32, device=dev),
        raw=wp.empty(E * N, dtype=wp.vec2f, device=dev),
        crossers=wp.empty(E, dtype=wp.int32, device=dev),
        poly=wp.empty(E * Np, dtype=wp.vec2f, device=dev),
        sort_keys=wp.empty(E * Np, dtype=wp.float32, device=dev),
        arc_real=wp.empty(E * Np, dtype=wp.vec2f, device=dev),
        arc_seg=wp.empty(E * Np, dtype=wp.float32, device=dev),
        arc_s=wp.empty(E * (Np + 1), dtype=wp.float32, device=dev),
        arc_cr=wp.empty(E, dtype=wp.int32, device=dev),
        arc_co=wp.empty(E, dtype=wp.int32, device=dev),
    )


def generate_grammar_warp(
    seeds_wp: wp.array,
    config,
    out_centerline: wp.array,
    out_valid_wp: wp.array,
    scratch,
) -> None:
    """Segment-grammar centerline generation — in-place, pure Warp, zero-alloc.

    Draws per-env segment grammars (sample_k), rasterizes + closes + integrates (build_k),
    normalizes the bbox (normalize_k), then rescues self-crossing envs with a simple
    angle-sorted polygon fallback (mirrors hull's Step 6-7). Non-crossing envs keep their
    grammar curve untouched.

    Polygon fallback (CUDA-graph-capturable, zero host sync):
    1. Detect self-crossers via self_intersections_inplace -> scratch.crossers.
    2. For crossing envs only: angle-sort the N grammar centerline points about their
       centroid into scratch.poly (produces a star-shaped simple loop).
    3. Arc-resample the selected polygon back into out_centerline for crossing envs only
       (non-crossing rows are left untouched by the selected resampler).

    Args:
        seeds_wp:       [E] int32 wp.array — per-env base seeds.
        config:         TrackGenConfig (uses grammar_* fields + num_points).
        out_centerline: [E*N] vec2f wp.array — written in-place.
        out_valid_wp:   [E] int32 wp.array — filled with 1.
        scratch:        GrammarScratch from grammar_alloc_scratch.
    """
    _pipe._init()
    assert scratch is not None, "generate_grammar_warp requires scratch"

    E = int(out_valid_wp.shape[0])
    S = int(config.grammar_segments)
    N = int(config.num_points)
    target_extent = float(config.scale) * _BEZIER_EXTENT
    dev = str(out_centerline.device)

    sharp = float(config.grammar_curvature_budget)
    straight_frac = float(config.grammar_straight_frac)
    chicane_bias = float(config.grammar_chicane_bias)
    hairpin_max_frac = float(config.grammar_hairpin_max_frac)

    # Step 1-3: sample grammar, rasterize + close + integrate, normalize bbox.
    wp.launch(
        _grammar_sample_k,
        dim=E,
        inputs=[seeds_wp, S, sharp, straight_frac, chicane_bias, hairpin_max_frac,
                scratch.segments],
        device=dev,
    )
    wp.launch(
        _grammar_build_k,
        dim=E,
        inputs=[scratch.segments, S, N, scratch.rast_kappa, scratch.raw,
                out_centerline, out_valid_wp],
        device=dev,
    )
    wp.launch(
        _grammar_normalize_k,
        dim=E,
        inputs=[out_centerline, N, target_extent],
        device=dev,
    )

    # Step 4: detect self-crossers in the grammar centerline.
    # scratch.arc_co is used as a fixed-N count array (all envs have N real points);
    # self_intersections_inplace reads it but does NOT write it, so it stays valid as
    # input to _arc_resample_selected_inplace below (where it becomes the output-count scratch).
    wp.launch(_pipe._fill_i32_k, dim=E, inputs=[scratch.arc_co, N], device=dev)
    _pipe.self_intersections_inplace(out_centerline, scratch.arc_co, scratch.crossers, N)

    # Step 5: stride-subsample + angle-sort the grammar centerline -> scratch.poly.
    # Np = N // _POLY_STRIDE sparse control points are taken from out_centerline (every
    # stride-th point), then angle-sorted about their centroid into scratch.poly.  This
    # gives a sparse simple polygon (like hull's P control points) that arc-resamples
    # cleanly to N, rather than sorting all N dense points which produces irregular geometry.
    # We launch for ALL envs; the selected resampler below skips non-crossers.
    Np = int(scratch.poly.shape[0]) // E
    wp.launch(
        _grammar_angle_sort_k,
        dim=E,
        inputs=[out_centerline, N, _POLY_STRIDE, Np, scratch.sort_keys, scratch.poly],
        device=dev,
    )

    # Step 6: arc-resample the angle-sorted sparse polygon (Np pts) into out_centerline
    # (N pts) for crossing envs only. Non-crossing envs' rows are left untouched.
    _pipe._arc_resample_selected_inplace(
        scratch.poly, scratch.crossers, Np, N,
        scratch.arc_real, scratch.arc_seg, scratch.arc_s,
        scratch.arc_cr, scratch.arc_co, out_centerline, dev,
    )

    _pipe._sync(dev)


from . import generator_registry as _registry  # noqa: E402
_registry.register(_registry.GeneratorSpec(
    name="grammar",
    alloc_scratch=grammar_alloc_scratch,
    generate=generate_grammar_warp,
))
