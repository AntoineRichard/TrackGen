"""Pure-Warp checkpoint-steering centerline generation (method #5, ``"checkpoint"``).

A fixed-shape NVIDIA-Warp port of Gymnasium ``CarRacing``'s track family, validated host-side
in ``track_gen/_experimental/checkpoint_proto.py`` (read it for the algorithm rationale). The
prototype's data-dependent steps — CarRacing's unbounded reject-retry and variable-length
loop-trim — are replaced by fixed-shape, CUDA-graph-capturable equivalents exactly as the
grammar/hull ports did:

1. ``sample_checkpoints``: C checkpoints at angle ``2*pi*c/C + jitter`` (jitter +/-
   ``angle_jitter`` of the slot ``2*pi/C``) on radius ``U(radius_min_frac*R, R)``, R=1.
2. ``steer`` (fixed-N bounded-turn): step length ``dl = ring_perimeter / N`` pins the walk to
   ~one lap. Start at checkpoint 0 heading toward 1; each step steer the heading toward the
   current target bearing (proportional gain, turn clamped to +/- turn_rate), advance the
   target index when within ``lookahead_frac*R`` (a per-step int-counter increment — capturable,
   no host branch).
3. ``close_heading_ramp`` (the default closure): per-edge heading, wrapped increments, ADD a
   constant drift ``(2*pi - net_turn)/N`` so the net turn is exactly 2*pi (turning number 1,
   no inner loops) WITHOUT rescaling local curvature; rebuild headings, gap-distribute the
   displacement (subtract the mean edge), cumsum to positions, recenter.
4. best-of-K: generate K decorrelated candidates per env, keep the fewest-self-intersection one
   (deterministic argmin, ties -> lowest k). Replaces the reject-retry.
5. OPT-IN single-crossing clip (``config.checkpoint_clip_fallback``, default off, like the
   bezier/hull polygon fallback): after selection, clip the selected loop's FIRST self-crossing,
   keep the longer sub-loop arc, arc-resample back to N. A capture-time Python branch.

Hard invariants (shared with the rest of ``track_gen/_src``): Warp-native + torch-free, ZERO
per-call allocation (all buffers come from ``checkpoint_alloc_scratch``), CUDA-graph-capturable
(fixed bounds over C / N / K, no host sync, no per-env Python branching — K>1/K==1 and the clip
are capture-time Python branches), deterministic in (seed, config) including best-of-K selection.

Convention (shared with ``warp_pipeline``): one thread per output element; flat arrays
``[E*N]`` of ``wp.vec2f`` and ``[E]`` per-env scalars; env index ``e = tid // N``.

The best-of-K crossing count, the clip's arc resample, the extent normalisation and the
fills are the public ``track_gen.ops`` primitives; the checkpoint draw is exposed as
:func:`sample_checkpoint_rings` (shared with the checkpoint gate generator).

Roadmap PR13c: the registered ``"checkpoint"`` generator is the :class:`Checkpoint`
Stage, whose radius / angle jitter / turn rate are dynamic (per-env arrays,
:func:`sample_checkpoint_rings_per_env` and :func:`_steer_dyn_k`, clamped to the
declared bounds at use; the steering gain and lookahead stay static);
``generate_checkpoint_warp`` / ``checkpoint_alloc_scratch`` /
``sample_checkpoint_rings`` keep their signatures and the scalar kernels on the same
private core, :func:`_enqueue_checkpoint`.
"""
from __future__ import annotations

import warp as wp

from .. import ops
from ..exec.context import LaunchContext, resolve as _resolve_ctx
from ..ops import funcs
from ..ops.constants import TARGET_EXTENT
from ..plugin.stage import CaptureClass, Stage
from .builtin_params import CheckpointParams, _declared_bounds
from .builtin_stage import DeprecatedParamsAlias, check_dyn, default_alloc, run_view
from .runtime import _init

# Legacy private name (value only): the loop is normalised to config.scale * TARGET_EXTENT
# (the same extent as polar / bezier, so half_width / spacing / relax see comparable
# coordinates across generators).
_BEZIER_EXTENT = TARGET_EXTENT

# Independent RNG salt for checkpoint sampling. Distinct from polar (7919), bezier (9781, 6151),
# and hull (7919, 5119, 3083) so this generator draws a different checkpoint set.
_CHECKPOINT_SALT = 4099
# Per-candidate offset salt: candidate k of env e uses rand_init(seeds[e]*_CHECKPOINT_SALT +
# k*_CAND_SALT + 1). A large odd multiplier decorrelates the K candidates (the best-of-K trick).
_CAND_SALT = 2741

#: Base ring radius. The loop is normalized after closure, so this only sets a stable
#: pre-scale.
BASE_RADIUS = 1.0
_BASE_RADIUS = BASE_RADIUS  # legacy private name (value only)


@wp.kernel
def _sample_checkpoints_k(
    seeds: wp.array(dtype=wp.int32),
    K: int,
    C: int,
    radius_min_frac: float,
    angle_jitter: float,
    base_radius: float,
    checkpoints: wp.array(dtype=wp.vec2f),
):
    # One thread per (env e, candidate k) slot. Builds C checkpoints monotone in angle:
    # angle c = 2*pi*c/C + jitter (jitter in +/- angle_jitter * slot, slot = 2*pi/C),
    # radius ~ U(radius_min_frac*R, R), R = base_radius. Stride per slot is C.
    slot_id = wp.tid()
    e = slot_id // K
    k = slot_id % K
    state = wp.rand_init(seeds[e] * _CHECKPOINT_SALT + k * _CAND_SALT + 1)
    b = slot_id * C
    two_pi = 2.0 * wp.pi
    slot = two_pi / float(C)
    rmin = radius_min_frac * base_radius

    for c in range(C):
        jitter = (2.0 * wp.randf(state) - 1.0) * angle_jitter * slot
        ang = float(c) * slot + jitter
        r = rmin + wp.randf(state) * (base_radius - rmin)
        checkpoints[b + c] = wp.vec2f(r * wp.cos(ang), r * wp.sin(ang))


@wp.kernel
def _steer_k(
    checkpoints: wp.array(dtype=wp.vec2f),
    K: int,
    C: int,
    N: int,
    turn_rate: float,
    steer_gain: float,
    lookahead: float,
    path: wp.array(dtype=wp.vec2f),
):
    # One thread per (env, candidate) slot. Walks a fixed N-step bounded-turn path that chases
    # the C checkpoints once around, writing the OPEN path into path[slot*N .. slot*N+N).
    #
    # Step length dl = ring_perimeter / N (a reduction over the C checkpoints) pins the walk to
    # ~one lap. Start at checkpoint 0 heading toward 1; each step: bearing to the current target,
    # err = wrap(bearing - theta), dtheta = clamp(steer_gain*err, +/- turn_rate), theta += dtheta,
    # advance position by dl*(cos,sin). The target index advances when within lookahead*R of it
    # (a per-step int-counter increment — capturable, no data-dependent host branch).
    slot_id = wp.tid()
    cb = slot_id * C
    pb = slot_id * N
    two_pi = 2.0 * wp.pi

    # Ring perimeter (closed C-gon over the checkpoints) -> dl. Fixed reduction over C.
    perim = float(0.0)
    for c in range(C):
        d = checkpoints[cb + (c + 1) % C] - checkpoints[cb + c]
        perim += wp.length(d)
    dl = perim / float(N)

    # Start at checkpoint 0, heading toward checkpoint 1 (deterministic, capturable).
    p = checkpoints[cb + 0]
    d01 = checkpoints[cb + 1] - checkpoints[cb + 0]
    theta = wp.atan2(d01[1], d01[0])
    tgt = int(1)  # index of current target checkpoint

    for i in range(N):
        path[pb + i] = p
        # Advance target if close enough (chase the NEXT checkpoint). Wrap the ring.
        target = checkpoints[cb + (tgt % C)]
        if wp.length(target - p) < lookahead:
            tgt = tgt + 1
            target = checkpoints[cb + (tgt % C)]
        # Proportional bounded steering toward the target bearing.
        bearing = wp.atan2(target[1] - p[1], target[0] - p[0])
        err = wp.atan2(wp.sin(bearing - theta), wp.cos(bearing - theta))  # wrap to (-pi, pi]
        dtheta = wp.clamp(steer_gain * err, -turn_rate, turn_rate)
        theta = theta + dtheta
        p = p + dl * wp.vec2f(wp.cos(theta), wp.sin(theta))


# The declared bounds of CheckpointParams' dynamic knobs (the twins' clamps, float32;
# CheckpointGateParams declares the same radius / angle jitter bounds).
_RMF_LO = wp.constant(wp.float32(_declared_bounds(CheckpointParams,
                                                  "checkpoint_radius_min_frac")[0]))
_RMF_HI = wp.constant(wp.float32(_declared_bounds(CheckpointParams,
                                                  "checkpoint_radius_min_frac")[1]))
_AJ_LO = wp.constant(wp.float32(_declared_bounds(CheckpointParams,
                                                 "checkpoint_angle_jitter")[0]))
_AJ_HI = wp.constant(wp.float32(_declared_bounds(CheckpointParams,
                                                 "checkpoint_angle_jitter")[1]))
_TURN_LO = wp.constant(wp.float32(_declared_bounds(CheckpointParams,
                                                   "checkpoint_turn_rate")[0]))
_TURN_HI = wp.constant(wp.float32(_declared_bounds(CheckpointParams,
                                                   "checkpoint_turn_rate")[1]))


@wp.kernel
def _sample_checkpoints_dyn_k(
    seeds: wp.array(dtype=wp.int32),
    K: int,
    C: int,
    rmf_e: wp.array(dtype=wp.float32),
    aj_e: wp.array(dtype=wp.float32),
    base_radius: float,
    checkpoints: wp.array(dtype=wp.vec2f),
):
    # _sample_checkpoints_k with the radius fraction / angle jitter read per env (roadmap
    # PR13c): each loaded once and clamped to its declared [0, 1] by comparisons (NaN and
    # every in-range value pass bit for bit), then the scalar kernel's body verbatim.
    slot_id = wp.tid()
    e = slot_id // K
    k = slot_id % K
    radius_min_frac = rmf_e[e]
    radius_min_frac = wp.where(radius_min_frac < _RMF_LO, _RMF_LO, radius_min_frac)
    radius_min_frac = wp.where(radius_min_frac > _RMF_HI, _RMF_HI, radius_min_frac)
    angle_jitter = aj_e[e]
    angle_jitter = wp.where(angle_jitter < _AJ_LO, _AJ_LO, angle_jitter)
    angle_jitter = wp.where(angle_jitter > _AJ_HI, _AJ_HI, angle_jitter)
    state = wp.rand_init(seeds[e] * _CHECKPOINT_SALT + k * _CAND_SALT + 1)
    b = slot_id * C
    two_pi = 2.0 * wp.pi
    slot = two_pi / float(C)
    rmin = radius_min_frac * base_radius

    for c in range(C):
        jitter = (2.0 * wp.randf(state) - 1.0) * angle_jitter * slot
        ang = float(c) * slot + jitter
        r = rmin + wp.randf(state) * (base_radius - rmin)
        checkpoints[b + c] = wp.vec2f(r * wp.cos(ang), r * wp.sin(ang))


@wp.kernel
def _steer_dyn_k(
    checkpoints: wp.array(dtype=wp.vec2f),
    K: int,
    C: int,
    N: int,
    turn_rate_e: wp.array(dtype=wp.float32),
    steer_gain: float,
    lookahead: float,
    path: wp.array(dtype=wp.vec2f),
):
    # _steer_k with the turn rate read per env (roadmap PR13c; the env of slot e*K + k is
    # slot // K): loaded once and clamped to its declared [0, pi] by comparisons, then the
    # scalar kernel's body verbatim (the steering gain and lookahead stay host scalars).
    slot_id = wp.tid()
    e = slot_id // K
    turn_rate = turn_rate_e[e]
    turn_rate = wp.where(turn_rate < _TURN_LO, _TURN_LO, turn_rate)
    turn_rate = wp.where(turn_rate > _TURN_HI, _TURN_HI, turn_rate)
    cb = slot_id * C
    pb = slot_id * N
    two_pi = 2.0 * wp.pi

    perim = float(0.0)
    for c in range(C):
        d = checkpoints[cb + (c + 1) % C] - checkpoints[cb + c]
        perim += wp.length(d)
    dl = perim / float(N)

    p = checkpoints[cb + 0]
    d01 = checkpoints[cb + 1] - checkpoints[cb + 0]
    theta = wp.atan2(d01[1], d01[0])
    tgt = int(1)

    for i in range(N):
        path[pb + i] = p
        target = checkpoints[cb + (tgt % C)]
        if wp.length(target - p) < lookahead:
            tgt = tgt + 1
            target = checkpoints[cb + (tgt % C)]
        bearing = wp.atan2(target[1] - p[1], target[0] - p[0])
        err = wp.atan2(wp.sin(bearing - theta), wp.cos(bearing - theta))
        dtheta = wp.clamp(steer_gain * err, -turn_rate, turn_rate)
        theta = theta + dtheta
        p = p + dl * wp.vec2f(wp.cos(theta), wp.sin(theta))


@wp.kernel
def _close_heading_ramp_k(
    path: wp.array(dtype=wp.vec2f),
    N: int,
    cand: wp.array(dtype=wp.vec2f),
):
    # One thread per (env, candidate) slot. Closes the open steered path to turning-number-1
    # ADDITIVELY and writes the closed (recentered) candidate into cand[slot*N .. slot*N+N).
    #
    # Per edge heading theta_i = atan2(edge_i); wrapped increments dtheta_i; ADD a constant drift
    # (2*pi - sum dtheta)/N to every step so the net turn is exactly 2*pi (no inner loops) WITHOUT
    # rescaling local curvature; rebuild headings (cumsum, zeroed at start); e = ds*(cos,sin) with
    # ds = 1/N; gap-distribute (subtract mean edge); cumsum to positions; subtract centroid.
    slot_id = wp.tid()
    b = slot_id * N
    two_pi = 2.0 * wp.pi
    ds = 1.0 / float(N)

    # Heading of edge 0 (used as the walking reference for wrapped-increment accumulation only;
    # NOT the cumsum zeroing reference — see drift below).
    e0 = path[b + 1] - path[b + 0]
    theta_prev = wp.atan2(e0[1], e0[0])
    theta0 = theta_prev

    # Sum of wrapped heading increments dtheta_i over the closed edge ring. dtheta_0 = 0
    # (np.diff(prepend=theta[0]) gives a leading zero), so start the accumulation at i = 1.
    sum_d = float(0.0)
    for i in range(1, N):
        ei = path[b + (i + 1) % N] - path[b + i]
        theta_i = wp.atan2(ei[1], ei[0])
        dth = theta_i - theta_prev
        dth = wp.atan2(wp.sin(dth), wp.cos(dth))  # wrap to (-pi, pi]
        sum_d += dth
        theta_prev = theta_i

    drift = (two_pi - sum_d) / float(N)  # additive correction per step -> net turn == 2*pi

    # Rebuild headings with the drift, integrate the unit-step displacement, accumulate the mean
    # edge (for gap-distribution) and the raw cumulative position sum (for the centroid). Each
    # quantity is a sequential reduction in this single thread — no scratch, no host sync.
    #
    # After adding drift: dtheta[0] = 0 + drift = drift (since dtheta_0 = 0 by construction),
    # so theta_closed[0] = cumsum[0] = drift. Subtracting drift zeros the first element:
    # theta_closed[i] = (cumsum of (dtheta + drift))[i] - drift, with theta_closed[0] := 0.
    # (Matches the proto's `theta_closed -= theta_closed[0]` which subtracts drift for the same
    # reason. Using theta0 instead would apply a constant ~3 rad offset — a rigid rotation.)
    theta_acc = float(0.0)        # running cumsum of corrected increments (pre drift shift)
    theta_prev2 = theta0
    sum_ex = float(0.0)           # sum of edge x (for mean-edge subtraction)
    sum_ey = float(0.0)
    # First pass: compute mean edge by integrating headings. We need the mean before cumsum to
    # positions, so do it in two passes over N (both bounded, single-thread sequential).
    for i in range(N):
        if i == 0:
            dth = float(0.0)
        else:
            ei = path[b + (i + 1) % N] - path[b + i]
            theta_i = wp.atan2(ei[1], ei[0])
            dthr = theta_i - theta_prev2
            dth = wp.atan2(wp.sin(dthr), wp.cos(dthr))
            theta_prev2 = theta_i
        theta_acc += dth + drift
        th_closed = theta_acc - drift   # subtract drift so theta_closed[0] := 0
        sum_ex += ds * wp.cos(th_closed)
        sum_ey += ds * wp.sin(th_closed)
    mean_ex = sum_ex / float(N)
    mean_ey = sum_ey / float(N)

    # Second pass: rebuild headings identically, subtract the mean edge, cumsum to positions,
    # and accumulate the centroid. Store the un-centered positions in cand, then recenter.
    theta_acc = float(0.0)
    theta_prev3 = theta0
    px = float(0.0)
    py = float(0.0)
    sum_px = float(0.0)
    sum_py = float(0.0)
    for i in range(N):
        if i == 0:
            dth = float(0.0)
        else:
            ei = path[b + (i + 1) % N] - path[b + i]
            theta_i = wp.atan2(ei[1], ei[0])
            dthr = theta_i - theta_prev3
            dth = wp.atan2(wp.sin(dthr), wp.cos(dthr))
            theta_prev3 = theta_i
        theta_acc += dth + drift
        th_closed = theta_acc - drift   # subtract drift so theta_closed[0] := 0
        ex = ds * wp.cos(th_closed) - mean_ex
        ey = ds * wp.sin(th_closed) - mean_ey
        px += ex
        py += ey
        cand[b + i] = wp.vec2f(px, py)
        sum_px += px
        sum_py += py

    cx = sum_px / float(N)
    cy = sum_py / float(N)
    for i in range(N):
        q = cand[b + i]
        cand[b + i] = wp.vec2f(q[0] - cx, q[1] - cy)


@wp.kernel
def _select_best_k(
    cand: wp.array(dtype=wp.vec2f),
    cross: wp.array(dtype=wp.int32),
    K: int,
    N: int,
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Deterministic argmin over this env's K candidates by self-intersection
    # count (ties -> lowest k), copying the chosen candidate into out[e*N .. e*N+N). cross is the
    # [E*K] per-candidate crossing count computed by ops.self_intersections over E*K slots.
    e = wp.tid()
    best_k = int(0)
    best_c = cross[e * K + 0]
    for k in range(1, K):
        ck = cross[e * K + k]
        if ck < best_c:
            best_c = ck
            best_k = k
    src = (e * K + best_k) * N
    dst = e * N
    for i in range(N):
        out[dst + i] = cand[src + i]


@wp.kernel
def _copy_single_k(
    cand: wp.array(dtype=wp.vec2f),
    N: int,
    out: wp.array(dtype=wp.vec2f),
):
    # K == 1 fast path: one thread per output point t (dim = E*N); copy the sole candidate
    # (slot == env) straight through. Avoids the per-candidate crossing pass entirely.
    t = wp.tid()
    out[t] = cand[t]


# ---------------------------------------------------------------------------------------------
# OPT-IN single-crossing clip (config.checkpoint_clip_fallback). Mirrors the proto's
# clip_single_crossing: find the FIRST crossing's segment pair (i, j) + intersection point P,
# split into inner arc pts[i+1..j] and outer arc pts[j+1..]+pts[..i] (each closed via P), keep
# the LONGER by arc length, write into a NaN-padded dense buffer, arc-resample back to N.
# ---------------------------------------------------------------------------------------------

@wp.kernel
def _clip_assemble_k(
    pts: wp.array(dtype=wp.vec2f),
    N: int,
    M: int,
    dense: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Finds the deterministic FIRST self-crossing (scan i ascending, then
    # j ascending) of the closed N-loop pts[e*N..], computes the intersection point P, builds the
    # kept (longer) sub-loop into dense[e*M .. e*M+M) with NaN padding beyond its length. If there
    # is NO crossing, copies the loop through unchanged (then NaN-pads). dense feeds the existing
    # NaN-aware arc-resampler, so the output stays fixed-N either way. M >= N + 2 (one extra slot
    # for the intersection vertex P plus headroom).
    e = wp.tid()
    base = e * N
    db = e * M

    # --- find first crossing (strict >0 sign-flip predicate, matching the proto's
    # _find_crossings; intentionally different from ops.self_intersections
    # which uses a scale-relative tolerance for best-of-K scoring — not a bug) ---
    found = int(0)
    fi = int(0)
    fj = int(0)
    Px = float(0.0)
    Py = float(0.0)
    for i in range(N):
        if found == 0:
            Ai = pts[base + i]
            Bi = pts[base + (i + 1) % N]
            for j in range(i + 1, N):
                if found == 0:
                    if ((i + 1) % N == j) or ((j + 1) % N == i):
                        pass  # adjacent / shared-endpoint -> skip
                    else:
                        Aj = pts[base + j]
                        Bj = pts[base + (j + 1) % N]
                        d1 = funcs.ccw(Ai[0], Ai[1], Bi[0], Bi[1], Aj[0], Aj[1])
                        d2 = funcs.ccw(Ai[0], Ai[1], Bi[0], Bi[1], Bj[0], Bj[1])
                        d3 = funcs.ccw(Aj[0], Aj[1], Bj[0], Bj[1], Ai[0], Ai[1])
                        d4 = funcs.ccw(Aj[0], Aj[1], Bj[0], Bj[1], Bi[0], Bi[1])
                        if ((d1 > 0.0) != (d2 > 0.0)) and ((d3 > 0.0) != (d4 > 0.0)):
                            # Intersection point of segments Ai->Bi and Aj->Bj.
                            rx = Bi[0] - Ai[0]
                            ry = Bi[1] - Ai[1]
                            sx = Bj[0] - Aj[0]
                            sy = Bj[1] - Aj[1]
                            denom = rx * sy - ry * sx
                            if wp.abs(denom) < 1.0e-12:
                                Px = 0.5 * (Ai[0] + Aj[0])
                                Py = 0.5 * (Ai[1] + Aj[1])
                            else:
                                tt = ((Aj[0] - Ai[0]) * sy - (Aj[1] - Ai[1]) * sx) / denom
                                Px = Ai[0] + tt * rx
                                Py = Ai[1] + tt * ry
                            fi = i
                            fj = j
                            found = 1

    if found == 0:
        # No crossing: copy the loop unchanged, then NaN-pad to M.
        for i in range(N):
            dense[db + i] = pts[base + i]
        for i in range(N, M):
            dense[db + i] = wp.vec2f(wp.nan, wp.nan)
        return

    P = wp.vec2f(Px, Py)

    # Inner sub-loop: P, pts[fi+1 .. fj], P. Vertices = (fj - fi) interior + 2 endpoints.
    # Outer sub-loop: P, pts[fj+1 .. N-1], pts[0 .. fi], P. Vertices = (N - (fj - fi)) interior + 2.
    # Arc lengths (closed polylines including the closing P->start edge are implicit; compare the
    # open polyline lengths the proto compares: norm of consecutive diffs over the stacked array).
    inner_len = float(0.0)
    prev = P
    for idx in range(fi + 1, fj + 1):
        cur = pts[base + idx]
        inner_len += wp.length(cur - prev)
        prev = cur
    inner_len += wp.length(P - prev)  # close back to P

    outer_len = float(0.0)
    prev = P
    for idx in range(fj + 1, N):
        cur = pts[base + idx]
        outer_len += wp.length(cur - prev)
        prev = cur
    for idx in range(0, fi + 1):
        cur = pts[base + idx]
        outer_len += wp.length(cur - prev)
        prev = cur
    outer_len += wp.length(P - prev)

    w = int(0)  # write cursor into dense
    if inner_len >= outer_len:
        # Keep inner: P, pts[fi+1 .. fj]. (Drop the trailing duplicate P; resampler closes it.)
        dense[db + w] = P
        w += 1
        for idx in range(fi + 1, fj + 1):
            dense[db + w] = pts[base + idx]
            w += 1
    else:
        # Keep outer: P, pts[fj+1 .. N-1], pts[0 .. fi].
        dense[db + w] = P
        w += 1
        for idx in range(fj + 1, N):
            dense[db + w] = pts[base + idx]
            w += 1
        for idx in range(0, fi + 1):
            dense[db + w] = pts[base + idx]
            w += 1
    # NaN-pad the rest so the arc-resampler drops the unused slots.
    for idx in range(w, M):
        dense[db + idx] = wp.vec2f(wp.nan, wp.nan)


class CheckpointScratch:
    """Pre-allocated PRIVATE working scratch for the ``"checkpoint"`` generator (one alloc).

    Sized for E envs * K candidates (EK slots). The generation OUTPUT buffers (out_centerline,
    out_valid) are orchestrator-owned and passed to generate; they are NOT part of this class.

    checkpoints: [EK*C] vec2f — per-(env,candidate) checkpoint ring.
    path:        [EK*N] vec2f — per-candidate open steered path.
    cand:        [EK*N] vec2f — per-candidate closed centerline (best-of-K input).
    cross:       [EK]   int32 — per-candidate self-intersection count (best-of-K score).
    cand_cnt:    [EK]   int32 — per-candidate point count (== N) for ops.self_intersections.
    clip_dense:  [E*M]  vec2f — selected-loop clip assembly (NaN-padded); None if no clip.
    arc_real:    [E*M]  vec2f — clip arc-resample compacted-real scratch; None if no clip.
    arc_seg:     [E*M]  float32 — clip arc-resample per-segment length scratch; None if no clip.
    arc_s:       [E*(M+1)] float32 — clip arc-resample cumulative arc-length scratch; None ...
    arc_cr:      [E]    int32 — clip arc-resample real-point-count scratch; None if no clip.
    arc_co:      [E]    int32 — clip arc-resample output-count scratch; None if no clip.
    """

    __slots__ = (
        "checkpoints", "path", "cand", "cross", "cand_cnt",
        "clip_dense", "arc_real", "arc_seg", "arc_s", "arc_cr", "arc_co",
    )

    def __init__(self, checkpoints, path, cand, cross, cand_cnt,
                 clip_dense, arc_real, arc_seg, arc_s, arc_cr, arc_co):
        self.checkpoints = checkpoints
        self.path = path
        self.cand = cand
        self.cross = cross
        self.cand_cnt = cand_cnt
        self.clip_dense = clip_dense
        self.arc_real = arc_real
        self.arc_seg = arc_seg
        self.arc_s = arc_s
        self.arc_cr = arc_cr
        self.arc_co = arc_co


def sample_checkpoint_rings(ctx: "LaunchContext", seeds: wp.array, K: int, C: int,
                            radius_min_frac: float, angle_jitter: float, out: wp.array, *,
                            base_radius: float = BASE_RADIUS) -> None:
    """Draw ``K`` candidate rings of ``C`` checkpoints per env into ``out``.

    Slot ``e*K + k`` (env ``e``, candidate ``k``) gets ``C`` checkpoints monotone in
    angle: checkpoint ``c`` at angle ``2*pi*c/C`` plus a jitter of up to
    ``angle_jitter`` of the slot ``2*pi/C``, radius uniform in
    ``[radius_min_frac*base_radius, base_radius)``, from
    ``wp.rand_init(seeds[e] * 4099 + k * 2741 + 1)``. ``out`` is ``[E*K*C]`` ``vec2f``.
    One launch (``seeds.shape[0] * K`` threads); no allocation, no host sync, no
    validation.
    """
    K = int(K)
    ctx.launch(_sample_checkpoints_k, seeds.shape[0] * K,
               [seeds, K, int(C), float(radius_min_frac), float(angle_jitter),
                float(base_radius), out])


def sample_checkpoint_rings_per_env(ctx: "LaunchContext", seeds: wp.array, K: int, C: int,
                                    radius_min_frac: wp.array, angle_jitter: wp.array,
                                    out: wp.array, *,
                                    base_radius: float = BASE_RADIUS) -> None:
    """:func:`sample_checkpoint_rings` with per-env knobs (roadmap PR13c): ``[E]``
    ``float32`` arrays, every candidate ring of env ``e`` drawn with
    ``radius_min_frac[e]`` / ``angle_jitter[e]``, each clamped at use to the checkpoint
    generator's declared bounds (``[0, 1]``, compared in float32; a NaN passes). With
    every value inside its bounds, env ``e`` equals :func:`sample_checkpoint_rings` at
    the scalars. One launch (``seeds.shape[0] * K`` threads); no allocation, no host
    sync.
    """
    K = int(K)
    ctx.launch(_sample_checkpoints_dyn_k, seeds.shape[0] * K,
               [seeds, K, int(C), radius_min_frac, angle_jitter, float(base_radius), out])


#: The dynamic fields of the Checkpoint Stage (roadmap PR13c), in Params order.
_DYNAMIC = ("checkpoint_radius_min_frac", "checkpoint_angle_jitter", "checkpoint_turn_rate")


def checkpoint_alloc_scratch(config):
    """Allocate the checkpoint generator's PRIVATE working scratch (one alloc per generator).

    Sized by K = config.checkpoint_best_of_k. The clip buffers are allocated only when
    config.checkpoint_clip_fallback is True (the clip is a capture-time Python branch)."""
    return _alloc_checkpoint_scratch(config)


def _alloc_checkpoint_scratch(config, alloc=None):
    """:func:`checkpoint_alloc_scratch` with every buffer from ``alloc(name, dtype, n)``
    (named after its :class:`CheckpointScratch` slot; default ``wp.empty`` on
    ``config.device``). The Stage passes its ``StageBuilder.scratch``."""
    _init()
    alloc = default_alloc(config) if alloc is None else alloc
    E = int(config.num_envs)
    N = int(config.num_points)
    C = int(config.checkpoint_count)
    K = max(int(config.checkpoint_best_of_k), 1)
    EK = E * K

    clip = bool(getattr(config, "checkpoint_clip_fallback", False))
    if clip:
        # The kept sub-loop is at most the N original vertices + the intersection point P, so
        # M = N + 2 gives one spare slot beyond the worst case.
        M = N + 2
        clip_dense = alloc("clip_dense", wp.vec2f, E * M)
        arc_real = alloc("arc_real", wp.vec2f, E * M)
        arc_seg = alloc("arc_seg", wp.float32, E * M)
        arc_s = alloc("arc_s", wp.float32, E * (M + 1))
        arc_cr = alloc("arc_cr", wp.int32, E)
        arc_co = alloc("arc_co", wp.int32, E)
    else:
        clip_dense = arc_real = arc_seg = arc_s = arc_cr = arc_co = None

    return CheckpointScratch(
        checkpoints=alloc("checkpoints", wp.vec2f, EK * C),
        path=alloc("path", wp.vec2f, EK * N),
        cand=alloc("cand", wp.vec2f, EK * N),
        cross=alloc("cross", wp.int32, EK),
        cand_cnt=alloc("cand_cnt", wp.int32, EK),
        clip_dense=clip_dense,
        arc_real=arc_real,
        arc_seg=arc_seg,
        arc_s=arc_s,
        arc_cr=arc_cr,
        arc_co=arc_co,
    )


def generate_checkpoint_warp(seeds_wp: wp.array, config,
                             out_centerline: wp.array, out_valid_wp: wp.array,
                             scratch, ctx: "LaunchContext | None" = None) -> None:
    """Checkpoint-steering centerline generation — in-place owned path.

    Pure-Warp: sample checkpoints -> steer (fixed-N) -> close_heading_ramp -> best-of-K select
    -> (opt-in) single-crossing clip -> normalize -> write the chosen centerline into
    out_centerline. Marks all envs valid (the generation gate is always True; inflate does the
    real validity gate, as for bezier/polar/hull).

    The legacy entry point (frozen signature; the knobs of ``config`` as host scalars):
    the registered ``"checkpoint"`` generator is the :class:`Checkpoint` Stage on the same
    private core (roadmap PR13c).

    Args:
        seeds_wp:       [E] int32 wp.array per-env base seeds.
        config:         TrackGenConfig.
        out_centerline: [E*num_points] vec2f wp.array — written in-place with the centerline.
        out_valid_wp:   [E] int32 wp.array — filled with 1 (all valid at generation stage).
        scratch:        a CheckpointScratch from checkpoint_alloc_scratch.
        ctx: optional track_gen.exec.LaunchContext; default resolves the
            device's stream (see track_gen.exec).

    K > 1 / K == 1 and clip_fallback are capture-time Python branches (no per-env host control
    flow), so the captured graph stays fixed and allocation-free.
    """
    _init()
    ctx = _resolve_ctx(ctx, str(out_centerline.device))
    assert scratch is not None, "generate_checkpoint_warp requires scratch"
    _enqueue_checkpoint(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx, None)


def _enqueue_checkpoint(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx,
                        dyn) -> None:
    """The checkpoint generator's launches on ``ctx``: ``dyn=None`` (the legacy path:
    ``config``'s knobs, :func:`sample_checkpoint_rings` and :func:`_steer_k`) or the
    radius / angle jitter / turn rate ``[E]`` arrays (the Stage:
    :func:`sample_checkpoint_rings_per_env` and :func:`_steer_dyn_k`)."""
    check_dyn(dyn, _DYNAMIC, "checkpoint")
    E = int(config.num_envs)
    N = int(config.num_points)
    C = int(config.checkpoint_count)
    K = max(int(config.checkpoint_best_of_k), 1)
    EK = E * K

    steer_gain = float(config.checkpoint_steer_gain)
    lookahead = float(config.checkpoint_lookahead_frac) * BASE_RADIUS
    target_extent = float(config.scale) * TARGET_EXTENT

    if dyn is None:
        radius_min_frac = float(config.checkpoint_radius_min_frac)
        angle_jitter = float(config.checkpoint_angle_jitter)
        turn_rate = float(config.checkpoint_turn_rate)

        # 1. Sample C checkpoints per (env, candidate) slot.
        sample_checkpoint_rings(ctx, seeds_wp, K, C, radius_min_frac, angle_jitter,
                                scratch.checkpoints)

        # 2. Steer the fixed-N bounded-turn open path per slot.
        ctx.launch(_steer_k, EK,
                   [scratch.checkpoints, K, C, N, turn_rate, steer_gain, lookahead,
                    scratch.path])
    else:
        sample_checkpoint_rings_per_env(ctx, seeds_wp, K, C,
                                        dyn["checkpoint_radius_min_frac"],
                                        dyn["checkpoint_angle_jitter"], scratch.checkpoints)
        ctx.launch(_steer_dyn_k, EK,
                   [scratch.checkpoints, K, C, N, dyn["checkpoint_turn_rate"], steer_gain,
                    lookahead, scratch.path])

    # 3. Close each candidate to turning-number-1 (additive heading ramp + displacement close).
    ctx.launch(_close_heading_ramp_k, EK, [scratch.path, N, scratch.cand])

    # 4. best-of-K selection -> out_centerline. K == 1 is a straight copy (capture-time branch).
    if K == 1:
        ctx.launch(_copy_single_k, E * N, [scratch.cand, N, out_centerline])
    else:
        # Per-candidate self-intersection count over EK slots (count[slot] == N).
        ops.fill(ctx, scratch.cand_cnt, N)
        ops.self_intersections(ctx, scratch.cand, N, scratch.cross, count=scratch.cand_cnt)
        ctx.launch(_select_best_k, E, [scratch.cand, scratch.cross, K, N, out_centerline])

    # 5. OPT-IN single-crossing clip of the selected centerline (capture-time Python branch).
    if bool(getattr(config, "checkpoint_clip_fallback", False)):
        # The clip buffers are allocated only when the flag was set at alloc time; a flag
        # flipped on between alloc and generate would launch with None buffers. Fail loudly.
        assert scratch.clip_dense is not None, (
            "checkpoint_clip_fallback=True requires scratch allocated with the same flag; "
            "reallocate the generator after changing checkpoint_clip_fallback."
        )
        M = N + 2
        ctx.launch(_clip_assemble_k, E, [out_centerline, N, M, scratch.clip_dense])
        ops.arc_resample(
            ctx, scratch.clip_dense, M, N,
            ops.ArcScratch(scratch.arc_real, scratch.arc_seg, scratch.arc_s, scratch.arc_cr),
            out_centerline, scratch.arc_co,
        )

    # 6. Normalize the selected centerline's bbox extent to match the bezier baseline.
    ops.normalize_extent(ctx, out_centerline, N, target_extent)

    # 7. Mark all envs valid (gen gate is always True; inflate does the real gate).
    ops.fill(ctx, out_valid_wp, 1)

    ctx.sync()


class Checkpoint(Stage):
    """``"checkpoint"``: the checkpoint-steering track generator
    (:func:`generate_checkpoint_warp`) as a ``kind="generator"`` Stage (roadmap PR13c).
    ``checkpoint_radius_min_frac`` / ``checkpoint_angle_jitter`` / ``checkpoint_turn_rate``
    are dynamic: per-env arrays read by :func:`sample_checkpoint_rings_per_env` and
    :func:`_steer_dyn_k`, clamped to ``[0, 1]`` / ``[0, 1]`` / ``[0, pi]`` at use."""

    kind = "generator"
    name = "checkpoint"
    output = "centerline"
    capture = CaptureClass.STATIC
    Params = CheckpointParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}
    #: Deprecated (0.2.0.dev4's ``GeneratorSpec.params``): use ``Params``.
    params = DeprecatedParamsAlias()

    def build(self, b, p):
        cfg = run_view(p, b)
        scratch = _alloc_checkpoint_scratch(cfg, b.scratch)
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        dyn = {k: b.param(k) for k in _DYNAMIC}
        seed = b.seed

        def enqueue(ctx):
            _enqueue_checkpoint(seed, cfg, xy, valid, scratch, ctx, dyn)
        return enqueue


def register(reg) -> None:
    """Register the built-in ``"checkpoint"`` track generator (the :class:`Checkpoint`
    Stage) into ``reg`` (called by ``track_gen._src.builtin_plugins``; importing this
    module registers nothing)."""
    reg.add(Checkpoint)
