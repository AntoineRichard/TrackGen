"""Track-frame localization: centerline projection, curvature, speed profiles.

``TrackLocalizer`` binds to a :class:`Track` batch and answers, for one point
per env, WHERE it is in the track's own frame: arc length ``s`` along the
centerline, signed lateral offset ``n``, and the centerline segment index.
This is the standard (s, n) Frenet-style parameterization racing controllers
and reward shapers consume every sim step.

Two search modes, one kernel:

- Cold scan (default): one Warp thread per env walks all ``count[e]``
  centerline segments — exact, zero state.
- Warm start (``warm_window=W``): the scan covers only the ``2*W + 1``
  segments centered on the previous query's result. Exact whenever the true
  nearest segment lies inside that window — the usual case for the intended
  use (queries every sim step, motion per step well under ``W * spacing``),
  but NOT implied by small motion alone: near track pinch points the nearest
  segment can jump half a lap discontinuously, and the warm result then
  stays on the traveled branch instead. ``reset(mask)`` drops the memory
  (next query falls back to the full scan), REQUIRED after regenerating the
  bound track and after teleports.

``curvature()`` and ``speed_profile()`` are per-generation helpers in the
same module: per-point signed centerline curvature (turn angle over arc
length, wrap-aware moving-average smoothing) and the classic
curvature-limited speed profile (steady-state ``sqrt(a_lat_max / |kappa|)``
capped at ``v_cap``, then a forward acceleration pass and a backward braking
pass over the closed loop). Unlike the per-step ``query()``, they ALLOCATE
their results — call them after ``generate()``, outside capture regions.

Layout follows the package conventions: flat ``[E * N_max]`` wp.arrays, NaN
past ``count[e]``, in-place reuse of the ``TrackFrame`` result across
``query()`` calls (use ``clone()`` for snapshots), and no host syncs while
capturing is enabled (``track_gen.set_capturing``) so ``query()``/``reset()``
are CUDA-graph capturable (fixed iteration bounds, no data-dependent host
branches). Results are undefined for envs with ``valid[e] == 0``.
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .collision_geom import _is_nan3, _safe_normalize3
from .runtime import _BIG, _check_arr, _init, _sync
from .types import Track


@dataclass
class TrackFrame:
    """Per-env track-frame coordinates, all ``[E]``; overwritten per query.

    .. warning::

        ``TrackLocalizer.query()`` returns the SAME instance every call.
        ``clone()`` for snapshots.

    Attributes
    ----------
    s : wp.array
        ``float32`` — arc length along the centerline at the projected point,
        in ``[0, Track.length[e])``, same units as the track coordinates.
        NaN for NaN positions and degenerate tracks (count < 3).
    n : wp.array
        ``float32`` — signed lateral offset from the centerline: positive to
        the RIGHT of the centerline's direction of travel (the direction of
        increasing ``s``), negative to the left. Whether the right side is
        the outer or the inner boundary depends on the loop's winding, which
        is generator-dependent (CCW loops have ``outer`` on the right, CW
        loops on the left); ``|n| <= half_width`` means on the road either
        way. NaN for NaN positions and degenerate tracks.
    n_up : wp.array
        ``float32`` — signed vertical offset in the roll-free frame at the
        foot point; equals ``position.z`` for planar tracks. NaN for NaN
        positions and degenerate tracks.
    segment : wp.array
        ``int32`` — index ``i`` of the nearest centerline segment (from point
        ``i`` to ``i + 1``, the closing segment being ``count[e] - 1 -> 0``).
        -1 for NaN positions and degenerate tracks.
    """

    s: wp.array
    n: wp.array
    n_up: wp.array
    segment: wp.array

    def clone(self) -> "TrackFrame":
        """Return a deep copy whose Warp buffers do not alias this result."""
        return TrackFrame(
            s=wp.clone(self.s),
            n=wp.clone(self.n),
            n_up=wp.clone(self.n_up),
            segment=wp.clone(self.segment),
        )


@wp.kernel
def _localize_k(
    center: wp.array(dtype=wp.vec3f),
    arclen: wp.array(dtype=wp.float32),
    length: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    warm_window: int,
    position: wp.array(dtype=wp.vec3f),
    last_seg: wp.array(dtype=wp.int32),
    out_s: wp.array(dtype=wp.float32),
    out_n: wp.array(dtype=wp.float32),
    out_n_up: wp.array(dtype=wp.float32),
    out_seg: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    p = position[e]
    m = count[e]
    if m > n_max:
        m = n_max
    if _is_nan3(p) == 1 or m < 3:
        # NaN position pauses the env; degenerate track has no frame. Both
        # drop the warm-start memory so the next real query does a full scan.
        out_s[e] = wp.nan
        out_n[e] = wp.nan
        out_n_up[e] = wp.nan
        out_seg[e] = -1
        last_seg[e] = -1
        return

    base = e * n_max

    # Window selection: full ring by default; the 2W+1 segments around the
    # previous result when warm-started. Exact iff the nearest segment lies
    # inside the window (see the class docstring for the motion contract).
    start = int(0)
    span = m
    last = last_seg[e]
    if warm_window > 0 and last >= 0 and last < m:
        span = 2 * warm_window + 1
        if span > m:
            span = m
        start = last - warm_window

    best_d2 = float(_BIG)
    best_i = int(0)
    best_u = float(0.0)
    for k in range(span):
        i = start + k
        i = ((i % m) + m) % m
        j = i + 1
        if j == m:
            j = 0
        a = center[base + i]
        ab = center[base + j] - a
        denom = wp.max(wp.dot(ab, ab), 1.0e-12)
        u = wp.clamp(wp.dot(p - a, ab) / denom, 0.0, 1.0)
        q = a + ab * u
        d2 = wp.dot(p - q, p - q)
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
            best_u = u

    j = best_i + 1
    if j == m:
        j = 0
    a = center[base + best_i]
    ab = center[base + j] - a
    q = a + ab * best_u

    # Arc length from the track's own cumulative table (the closing segment
    # ends at the perimeter), so s is consistent with Track.arclen/length.
    seg_start = arclen[base + best_i]
    seg_end = length[e]
    if best_i + 1 < m:
        seg_end = arclen[base + best_i + 1]

    # Signed offsets in the roll-free frame at the foot point: right_hat =
    # t x up_world points to the RIGHT of the direction of travel (increasing
    # s) — (t.y, -t.x, 0) for planar tangents. Which boundary that is depends
    # on the loop's winding — see the TrackFrame docstring.
    t = _safe_normalize3(ab)
    right_hat = wp.cross(t, wp.vec3f(0.0, 0.0, 1.0))   # (t.y, -t.x, 0)
    rl = wp.length(right_hat)
    if rl < 1.0e-6:
        right_hat = wp.vec3f(1.0, 0.0, 0.0)            # vertical segment guard
    else:
        right_hat = right_hat / rl
    up_hat = wp.cross(right_hat, t)

    s = seg_start + best_u * (seg_end - seg_start)
    if s >= length[e]:
        s = s - length[e]  # u == 1 on the closing segment: wrap the seam
    out_s[e] = s
    out_n[e] = wp.dot(p - q, right_hat)
    out_n_up[e] = wp.dot(p - q, up_hat)
    out_seg[e] = best_i
    last_seg[e] = best_i


@wp.kernel
def _localize_reset_k(
    mask: wp.array(dtype=wp.int32),
    last_seg: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    if mask[e] != 0:
        last_seg[e] = -1


class TrackLocalizer:
    """Project one point per env onto the bound Track's centerline.

    Because ``TrackGenerator.generate()`` overwrites its ``Track`` buffers in
    place, ``query()`` always reads the CURRENT track batch — but a
    warm-started localizer remembers segment indices of the OLD geometry, so
    callers MUST ``reset()`` all envs after regenerating (cold-scan
    localizers, ``warm_window=None``, need nothing).

    ``query()`` is allocation-free and, under graph capture (enabled via
    ``track_gen.set_capturing``), host-sync-free (CUDA-graph capturable). In
    normal (non-capturing) use a ``wp.synchronize()`` follows the launch, per
    the codebase idiom. ``query()`` returns the same preallocated
    :class:`TrackFrame` on every call.

    Warm vs cold results are IDENTICAL whenever the nearest centerline
    segment stays within ``warm_window`` segments of the previous query's
    result — the usual regime (per-step motion small against
    ``warm_window * spacing``). Small motion alone does NOT guarantee it:
    where the loop pinches close to itself, the nearest segment can jump
    half a lap for arbitrarily small motion, and the warm result then stays
    on the traveled branch (often preferable for racing consumers — s stays
    continuous — but not the cold answer). Outside the contract (teleports,
    huge steps), a warm query may likewise lock onto a local minimum; call
    ``reset()`` on the affected envs to force a full rescan.
    """

    def __init__(self, track: Track, warm_window: "int | None" = None,
                 position: "wp.array | None" = None) -> None:
        """Bind to a :class:`Track` batch and allocate the result buffers.

        Args:
            track: the bound track batch; ``query()`` reads its centerline
                buffers directly on every call.
            warm_window: ``None`` (default) scans all segments every query;
                an int >= 1 scans only the ``2 * warm_window + 1`` segments
                centered on the previous result (per env; the first query
                after construction or ``reset()`` is always a full scan).
            position: optional stable ``[E]`` vec3f buffer to bind at
                construction (bound mode); equivalent to calling
                :meth:`bind` right after construction.
        """
        _init()
        if warm_window is not None and int(warm_window) < 1:
            raise ValueError(
                f"warm_window must be >= 1 (or None for full scans), got "
                f"{warm_window!r}")
        E = int(track.count.shape[0])
        stride = int(track.center.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"track batch layout invalid: center has {stride} slots for "
                f"{E} envs")
        self._track = track
        self._E = E
        self._n_max = stride // E
        self._W = 0 if warm_window is None else int(warm_window)
        self._device = str(track.center.device)

        self._bound_pos: "wp.array | None" = None
        if position is not None:
            self.bind(position)

        dev = self._device
        self._last = wp.full(E, -1, dtype=wp.int32, device=dev)
        self._frame = TrackFrame(
            s=wp.zeros(E, dtype=wp.float32, device=dev),
            n=wp.zeros(E, dtype=wp.float32, device=dev),
            n_up=wp.zeros(E, dtype=wp.float32, device=dev),
            segment=wp.zeros(E, dtype=wp.int32, device=dev),
        )

    def _validate_position(self, position) -> None:
        _check_arr("position", position, (self._E,), wp.vec3f, self._device)

    def bind(self, position: wp.array) -> None:
        """Bind (or rebind) a stable ``[E]`` vec3f position buffer.

        After binding, ``query()`` takes no arguments and reads the buffer
        in place. Validation happens here, once; the array must keep the
        same ``.ptr`` for the binding's lifetime (CUDA-graph contract).
        """
        self._validate_position(position)
        self._bound_pos = position

    def query(self, position: "wp.array | None" = None) -> TrackFrame:
        """Localize the batch; returns the localizer's preallocated frame.

        Bound mode (constructed with ``position=`` or after :meth:`bind`):
        call with no arguments. Per-call mode: pass the ``[E]`` vec3f
        position array — the SAME array (identical ``.ptr``) must be used
        across a CUDA-graph capture and its replays.

        NaN ``position[e]`` yields NaN ``s``/``n`` and ``segment = -1`` for
        that env, and drops its warm-start memory (the next finite query
        does a full scan) — safe for envs mid-teleport or awaiting respawn.

        Returns:
            The localizer's preallocated :class:`TrackFrame` (same instance
            every call; buffers overwritten in place).

        Raises:
            ValueError: on shape/dtype/device mismatch, or on mode misuse
                (passing a position while bound, or omitting it while not
                bound).
        """
        if self._bound_pos is not None:
            if position is not None:
                raise ValueError(
                    "localizer is bound to a position buffer; call query() "
                    "with no arguments")
            pos = self._bound_pos
        else:
            if position is None:
                raise ValueError(
                    "localizer is not bound; pass position to query() or "
                    "construct with position=")
            self._validate_position(position)
            pos = position
        t = self._track
        f = self._frame
        wp.launch(
            _localize_k, dim=self._E,
            inputs=[t.center, t.arclen, t.length, t.count, self._n_max,
                    self._W, pos, self._last, f.s, f.n, f.n_up, f.segment],
            device=self._device,
        )
        _sync(self._device)
        return f

    def reset(self, mask: wp.array) -> None:
        """Drop warm-start memory where ``mask[e]`` is nonzero (``[E]`` int32).

        The next query for those envs performs a full centerline scan.
        Required for ALL envs after regenerating the bound track when
        ``warm_window`` is set, and for any env that teleports further than
        the warm window covers. A no-op for cold-scan localizers (but valid
        to call, so regeneration code need not special-case).
        """
        _check_arr("mask", mask, (self._E,), wp.int32, self._device)
        wp.launch(
            _localize_reset_k, dim=self._E,
            inputs=[mask, self._last],
            device=self._device,
        )
        _sync(self._device)


@wp.kernel
def _curvature_raw_k(
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    out_kappa: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    e = tid // n_max
    i = tid - e * n_max
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3 or i >= m:
        out_kappa[tid] = wp.nan
        return
    base = e * n_max
    ip = i - 1
    if ip < 0:
        ip = m - 1
    j = i + 1
    if j == m:
        j = 0
    v1 = center[base + i] - center[base + ip]
    v2 = center[base + j] - center[base + i]
    # Discrete curvature: turn angle at point i over the mean incident arc.
    # The +z cross component keeps the documented CCW-positive sign and is
    # bit-identical to the legacy 2D scalar cross for planar (z = 0) tracks.
    cr = wp.cross(v1, v2)
    ang = wp.atan2(cr[2], wp.dot(v1, v2))
    ds = 0.5 * (wp.length(v1) + wp.length(v2))
    out_kappa[tid] = ang / wp.max(ds, 1.0e-9)


@wp.kernel
def _smooth_wrap_k(
    raw: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    window: int,
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    e = tid // n_max
    i = tid - e * n_max
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3 or i >= m:
        out[tid] = wp.nan
        return
    base = e * n_max
    span = 2 * window + 1
    if span > m:
        span = m  # tiny loops: average each point once, not double-counted
    acc = float(0.0)
    for k in range(span):
        idx = i - window + k
        idx = ((idx % m) + m) % m
        acc = acc + raw[base + idx]
    out[tid] = acc / float(span)


def curvature(track: Track, window: int = 2) -> wp.array:
    """Per-point signed centerline curvature, flat ``[E * N_max]`` float32.

    Discrete curvature at point ``i``: the turn angle between the incoming
    and outgoing centerline edges over the mean incident edge length,
    ``atan2(cross(v1, v2), dot(v1, v2)) / max(0.5 * (|v1| + |v2|), eps)``,
    then a wrap-aware moving average over ``2 * window + 1`` points. Units
    are 1 / track units. The sign follows the loop orientation: positive
    where the centerline turns counter-clockwise — a CCW circle of radius
    ``R`` gives ``kappa ~ +1/R`` everywhere. NaN past ``count[e]`` and for
    degenerate envs (``count[e] < 3``); results are undefined for envs with
    ``valid[e] == 0``.

    On a flat track (the default ``z_profile``) ``v1``/``v2`` are planar edge
    vectors and this is exactly the familiar 2D signed curvature. On a lifted
    track (a non-flat ``z_profile``), ``center`` carries real per-point
    altitude, so ``v1``/``v2`` are full 3D edge vectors, and the formula picks
    up grade in two ways: the numerator (``cross(v1, v2)[2]``) is still exactly
    the PLAN-VIEW cross product (unaffected by z on its own), but the
    denominator — both ``dot(v1, v2)`` inside the ``atan2`` and the ``ds``
    normalizer — gains a z contribution. Net effect: a genuine plan-view turn
    taken on a graded section reports a slightly SMALLER magnitude than the
    same turn taken flat (the added 3D edge length dilutes it); and a
    perfectly straight-in-plan-view section with a steep enough grade reversal
    (a crest or dip sharp enough to flip the sign of ``dot(v1, v2)``) can
    register as if it were a sharp turn even though the centerline never bends
    in plan view. Mild grade changes (no sign flip in the dot product) still
    read as zero curvature, same as flat.

    Unlike the per-step utilities, this ALLOCATES its result (and a scratch
    when ``window > 0``) — a per-generation helper: call it after
    ``generate()``, outside capture regions.

    Args:
        track: the track batch to read the centerline from.
        window: moving-average half-width in points (>= 0); 0 disables
            smoothing. Default 2 (a 5-tap average).

    Returns:
        A NEW flat ``[E * N_max]`` float32 wp.array of signed curvatures on
        the track's device (owned by the caller; not refreshed by later
        ``generate()`` calls).

    Raises:
        ValueError: on ``window < 0`` or an invalid track batch layout.
    """
    _init()
    if int(window) < 0:
        raise ValueError(f"window must be >= 0, got {window!r}")
    E = int(track.count.shape[0])
    stride = int(track.center.shape[0])
    if E < 1 or stride % E != 0:
        raise ValueError(
            f"track batch layout invalid: center has {stride} slots for "
            f"{E} envs")
    n_max = stride // E
    device = str(track.center.device)
    raw = wp.zeros(stride, dtype=wp.float32, device=device)
    wp.launch(
        _curvature_raw_k, dim=stride,
        inputs=[track.center, track.count, n_max, raw],
        device=device,
    )
    if int(window) == 0:
        _sync(device)
        return raw
    out = wp.zeros(stride, dtype=wp.float32, device=device)
    wp.launch(
        _smooth_wrap_k, dim=stride,
        inputs=[raw, track.count, n_max, int(window), out],
        device=device,
    )
    _sync(device)
    return out


@wp.kernel
def _speed_profile_k(
    kappa: wp.array(dtype=wp.float32),
    arclen: wp.array(dtype=wp.float32),
    length: wp.array(dtype=wp.float32),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    a_lat_max: float,
    a_accel: float,
    a_brake: float,
    v_cap: float,
    out_v: wp.array(dtype=wp.float32),
):
    # One thread per env; the accel/brake passes are inherently sequential.
    e = wp.tid()
    m = count[e]
    if m > n_max:
        m = n_max
    base = e * n_max
    if m < 3:
        for i in range(n_max):
            out_v[base + i] = wp.nan
        return

    # Steady state: lateral-acceleration limit, capped at v_cap. NaN kappa
    # on a real point (defensive) is treated as straight.
    for i in range(m):
        k = wp.abs(kappa[base + i])
        v = v_cap
        if k == k and k > 1.0e-9:
            v = wp.min(wp.sqrt(a_lat_max / k), v_cap)
        out_v[base + i] = v
    for i in range(m, n_max):
        out_v[base + i] = wp.nan

    # Two wrap laps so the closed loop converges: a forward acceleration
    # pass, then a backward braking pass, both over segment i -> i+1 whose
    # length comes from the track's cumulative arc table.
    for lap in range(2):
        for i in range(m):
            j = i + 1
            if j == m:
                j = 0
            seg_end = length[e]
            if i + 1 < m:
                seg_end = arclen[base + i + 1]
            ds = seg_end - arclen[base + i]
            vi = out_v[base + i]
            cap = wp.sqrt(vi * vi + 2.0 * a_accel * ds)
            if cap < out_v[base + j]:
                out_v[base + j] = cap
        for r in range(m):
            i = m - 1 - r
            j = i + 1
            if j == m:
                j = 0
            seg_end = length[e]
            if i + 1 < m:
                seg_end = arclen[base + i + 1]
            ds = seg_end - arclen[base + i]
            vj = out_v[base + j]
            cap = wp.sqrt(vj * vj + 2.0 * a_brake * ds)
            if cap < out_v[base + i]:
                out_v[base + i] = cap


def speed_profile(track: Track, a_lat_max: float, a_accel: float,
                  a_brake: float, v_cap: float,
                  kappa: "wp.array | None" = None,
                  window: int = 2) -> wp.array:
    """Curvature-limited target speed per centerline point, ``[E * N_max]``.

    Three stages, batched per env (one Warp thread per env; the passes are
    sequential by nature): the steady-state limit
    ``min(sqrt(a_lat_max / |kappa|), v_cap)``, a forward pass capping
    acceleration out of corners (``v[i+1]^2 <= v[i]^2 + 2 * a_accel * ds``),
    and a backward pass capping braking into them
    (``v[i]^2 <= v[i+1]^2 + 2 * a_brake * ds``) — each run twice around the
    closed loop so the wrap converges. Speeds are in track units per second
    when the accelerations are in track units per second squared.

    ``ds`` (from ``Track.arclen``/``Track.length``) is the true 3D segment
    length on a lifted (non-flat ``z_profile``) track, so the accel/brake
    ramps size their distance budget off the actual climbing/descending path
    length, not its plan-view projection. The corner-speed limit still comes
    from ``kappa`` (:func:`curvature`'s default, if not passed in), which on a
    lifted track can be perturbed by grade as documented there — this function
    adds no further elevation-specific handling of its own.

    Like :func:`curvature`, this ALLOCATES its result — a per-generation
    helper: call it after ``generate()``, outside capture regions.

    Args:
        track: the track batch; segment lengths come from ``Track.arclen``
            and ``Track.length``.
        a_lat_max: maximum lateral acceleration (> 0).
        a_accel: maximum forward acceleration (>= 0; 0 pins the profile to
            each point's corner speed going forward).
        a_brake: maximum braking deceleration (>= 0).
        v_cap: straight-line speed cap (> 0).
        kappa: optional precomputed flat ``[E * N_max]`` float32 curvature
            (e.g. from :func:`curvature`, possibly reused across several
            profiles). ``None`` (default) computes it internally.
        window: smoothing half-width forwarded to :func:`curvature` when
            ``kappa`` is None; ignored otherwise.

    Returns:
        A NEW flat ``[E * N_max]`` float32 wp.array of target speeds on the
        track's device, NaN past ``count[e]`` and for degenerate envs
        (``count[e] < 3``). Undefined for envs with ``valid[e] == 0``.

    Raises:
        ValueError: on a non-positive ``a_lat_max`` / ``v_cap``, a negative
            ``a_accel`` / ``a_brake``, a ``kappa`` shape/dtype/device
            mismatch, or an invalid track batch layout.
    """
    _init()
    if not (float(a_lat_max) > 0.0):
        raise ValueError(f"a_lat_max must be > 0, got {a_lat_max!r}")
    if float(a_accel) < 0.0:
        raise ValueError(f"a_accel must be >= 0, got {a_accel!r}")
    if float(a_brake) < 0.0:
        raise ValueError(f"a_brake must be >= 0, got {a_brake!r}")
    if not (float(v_cap) > 0.0):
        raise ValueError(f"v_cap must be > 0, got {v_cap!r}")
    E = int(track.count.shape[0])
    stride = int(track.center.shape[0])
    if E < 1 or stride % E != 0:
        raise ValueError(
            f"track batch layout invalid: center has {stride} slots for "
            f"{E} envs")
    n_max = stride // E
    device = str(track.center.device)
    if kappa is None:
        kappa = curvature(track, window=window)
    else:
        _check_arr("kappa", kappa, (stride,), wp.float32, device)
    out = wp.zeros(stride, dtype=wp.float32, device=device)
    wp.launch(
        _speed_profile_k, dim=E,
        inputs=[kappa, track.arclen, track.length, track.count, n_max,
                float(a_lat_max), float(a_accel), float(a_brake),
                float(v_cap), out],
        device=device,
    )
    _sync(device)
    return out
