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
    """

    __slots__ = ("segments", "rast_kappa", "raw")

    def __init__(self, segments, rast_kappa, raw) -> None:
        self.segments = segments
        self.rast_kappa = rast_kappa
        self.raw = raw


def grammar_alloc_scratch(config):
    """Allocate the grammar generator's PRIVATE working scratch (one alloc per generator)."""
    _pipe._init()
    E = int(config.num_envs)
    S = int(config.grammar_segments)
    N = int(config.num_points)
    dev = str(config.device)
    return GrammarScratch(
        segments=wp.empty(E * S * 3, dtype=wp.float32, device=dev),
        rast_kappa=wp.empty(E * N, dtype=wp.float32, device=dev),
        raw=wp.empty(E * N, dtype=wp.vec2f, device=dev),
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
    then normalizes the bbox (normalize_k). Marks all envs valid at generation stage.
    CUDA-graph-capturable: fixed-bound loops over S and N, no host sync.

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
    _pipe._sync(dev)


from . import generator_registry as _registry  # noqa: E402
_registry.register(_registry.GeneratorSpec(
    name="grammar",
    alloc_scratch=grammar_alloc_scratch,
    generate=generate_grammar_warp,
))
