"""Batched boundary prop sampling for rendering-only instancing.

``PropSampler`` resamples one track boundary (inner or outer) at a user-set
arc-length spacing and emits per-prop instancing poses into a preallocated
:class:`PropSet`. Two modes:

- ``"points"``: one pose per sample on the curve (cones, poles, markers).
- ``"segments"``: one pose per chord between consecutive samples (wall
  pieces): position = chord midpoint, yaw = chord direction, length = chord
  length, so a unit-length wall scaled by ``length`` tiles the ring.

Spacing is snapped per env — ``n = clamp(round(perimeter/spacing), 3,
max_props)`` at effective step ``perimeter/n`` — so the closed ring has no
seam gap or doubled prop. Props are NOT colliders (see ``track_gen.collision``
for out-of-bounds queries); this is the rendering/instancing complement.

Layout follows the package conventions: flat ``[E * max_props]`` wp.arrays,
NaN-padded past ``count[e]``, in-place reuse of the output ``PropSet`` across
``sample()`` calls (use ``clone()`` for snapshots), and no host syncs while
``_CAPTURING`` is set so ``sample()`` is CUDA-graph capturable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .collision_geom import _safe_normalize2
from .types import Track

_INITED = False
_CAPTURING = False


def _init() -> None:
    """Initialize Warp once (idempotent). Must run before any wp.launch."""
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _sync(device) -> None:
    if _CAPTURING:
        return
    if "cuda" in str(device):
        wp.synchronize()


@dataclass
class PropSet:
    """Batched prop poses, flat ``[E * max_props]`` per pose field.

    .. warning::

        ``PropSampler.sample()`` returns the SAME ``PropSet`` instance on
        every call and overwrites its buffers in place. Call ``clone()`` for
        a fully-owned snapshot.

    Attributes
    ----------
    position : wp.array
        ``vec2f`` pose positions: on-curve sample (points mode) or chord
        midpoint (segments mode). NaN past ``count[e]``.
    tangent : wp.array
        ``vec2f`` unit directions: curve tangent at the sample (points) or
        chord direction (segments). NaN past ``count[e]``.
    yaw : wp.array
        ``float32`` heading, ``atan2(tangent.y, tangent.x)``.
    length : wp.array
        ``float32`` per-prop extent: effective arc step (points mode) or
        chord length (segments mode).
    count : wp.array
        ``[E]`` ``int32`` real prop counts (0 for degenerate envs). Meaningful
        only for envs with ``valid[e] == 1``.
    truncated : wp.array
        ``[E]`` ``int32`` — 1 if ``max_props`` clipped this env's ring (the
        ring still closes, at a coarser effective spacing).
    step : wp.array
        ``[E]`` ``float32`` effective arc spacing ``perimeter / count``.
    """

    position: wp.array
    tangent: wp.array
    yaw: wp.array
    length: wp.array
    count: wp.array
    truncated: wp.array
    step: wp.array

    def clone(self) -> "PropSet":
        """Return a deep copy whose Warp buffers do not alias this set."""
        return PropSet(
            position=wp.clone(self.position),
            tangent=wp.clone(self.tangent),
            yaw=wp.clone(self.yaw),
            length=wp.clone(self.length),
            count=wp.clone(self.count),
            truncated=wp.clone(self.truncated),
            step=wp.clone(self.step),
        )


@wp.kernel
def _scan_boundary_k(
    points: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    spacing: float,
    max_props: int,
    cum: wp.array(dtype=wp.float32),
    out_count: wp.array(dtype=wp.int32),
    out_step: wp.array(dtype=wp.float32),
    out_truncated: wp.array(dtype=wp.int32),
):
    """Also launched by ``_src/checkpoints.py`` (CheckpointSampler) — keep the
    signature and snap semantics in sync."""
    e = wp.tid()
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        out_count[e] = 0
        out_step[e] = wp.nan
        out_truncated[e] = 0
        return
    base = e * n_max
    s = float(0.0)
    cum[base] = 0.0
    prev = points[base]
    for i in range(1, m):
        p = points[base + i]
        s = s + wp.length(p - prev)
        cum[base + i] = s
        prev = p
    perim = s + wp.length(points[base] - prev)  # closing edge back to point 0

    # Snap in float first: float->int32 conversion of out-of-range values
    # (absurdly small spacing) differs between CPU (INT_MIN) and CUDA
    # (saturating), so clamp against max_props before the cast.
    nf = wp.round(perim / spacing)
    trunc = int(0)
    n = int(0)
    if nf > float(max_props):
        n = max_props
        trunc = int(1)
    else:
        n = int(nf)
        if n < 3:
            n = 3
    out_count[e] = n
    out_step[e] = perim / float(n)
    out_truncated[e] = trunc


@wp.func
def _sample_at_arc(points: wp.array(dtype=wp.vec2f), cum: wp.array(dtype=wp.float32),
                   base: int, m: int, perim: float, s: float) -> wp.vec4f:
    """Point and segment direction at arc position s. Returns (px, py, dx, dy).

    Binary search for the largest i in [0, m-1] with cum[base+i] <= s, then
    lerp within segment i -> (i+1) mod m (the last segment closes the loop
    and ends at arc length ``perim``).
    """
    lo = int(0)
    hi = m - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cum[base + mid] <= s:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    j = i + 1
    seg_end = perim
    if j < m:
        seg_end = cum[base + j]
    else:
        j = 0
    seg_start = cum[base + i]
    denom = seg_end - seg_start
    t = 0.0
    if denom > 1.0e-12:
        t = wp.clamp((s - seg_start) / denom, 0.0, 1.0)
    a = points[base + i]
    b = points[base + j]
    p = a + (b - a) * t
    d = _safe_normalize2(b - a)
    return wp.vec4f(p[0], p[1], d[0], d[1])


@wp.kernel
def _place_props_k(
    points: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    cum: wp.array(dtype=wp.float32),
    prop_count: wp.array(dtype=wp.int32),
    prop_step: wp.array(dtype=wp.float32),
    max_props: int,
    mode: int,  # 0 = points, 1 = segments
    out_position: wp.array(dtype=wp.vec2f),
    out_tangent: wp.array(dtype=wp.vec2f),
    out_yaw: wp.array(dtype=wp.float32),
    out_length: wp.array(dtype=wp.float32),
):
    t = wp.tid()
    e = t // max_props
    k = t - e * max_props

    n = prop_count[e]
    if k >= n:
        nan2 = wp.vec2f(wp.nan, wp.nan)
        out_position[t] = nan2
        out_tangent[t] = nan2
        out_yaw[t] = wp.nan
        out_length[t] = wp.nan
        return

    m = count[e]
    if m > n_max:
        m = n_max
    base = e * n_max
    step = prop_step[e]
    perim = step * float(n)

    if mode == 0:
        s = float(k) * step
        smp = _sample_at_arc(points, cum, base, m, perim, s)
        tang = wp.vec2f(smp[2], smp[3])
        out_position[t] = wp.vec2f(smp[0], smp[1])
        out_tangent[t] = tang
        out_yaw[t] = wp.atan2(tang[1], tang[0])
        out_length[t] = step
    else:
        s0 = float(k) * step
        s1 = 0.0  # slot n-1 chords back to sample 0: the ring closes
        if k + 1 < n:
            s1 = float(k + 1) * step
        a4 = _sample_at_arc(points, cum, base, m, perim, s0)
        b4 = _sample_at_arc(points, cum, base, m, perim, s1)
        p0 = wp.vec2f(a4[0], a4[1])
        p1 = wp.vec2f(b4[0], b4[1])
        chord = p1 - p0
        tang = _safe_normalize2(chord)
        out_position[t] = (p0 + p1) * 0.5
        out_tangent[t] = tang
        out_yaw[t] = wp.atan2(tang[1], tang[0])
        out_length[t] = wp.length(chord)


class PropSampler:
    """Resample one track boundary into instancing poses at a set spacing.

    One sampler binds one :class:`Track`, one boundary curve, one mode, and
    one spacing, so all output shapes are fixed and ``sample()`` is
    CUDA-graph capturable (allocation-free; host-sync-free while the module
    ``_CAPTURING`` flag is set). Because ``TrackGenerator.generate()``
    overwrites its ``Track`` buffers in place, every ``sample()`` reads the
    CURRENT batch — no rebind after regeneration. Props are rendering-only;
    they are not colliders. Results for envs with ``valid[e] == 0`` are
    undefined — an invalid track can still have >= 3 boundary points and
    will yield well-formed-looking props with no NaN signal; callers must
    gate on ``Track.valid``, as everywhere in the library.
    """

    def __init__(self, track: Track, spacing: float, boundary: str = "outer",
                 mode: str = "points", max_props: "int | None" = None) -> None:
        _init()
        if not (float(spacing) > 0.0):
            raise ValueError(f"spacing must be > 0, got {spacing!r}")
        if boundary not in ("inner", "outer"):
            raise ValueError(
                f"boundary must be one of {{'inner', 'outer'}}, got {boundary!r}")
        if mode not in ("points", "segments"):
            raise ValueError(
                f"mode must be one of {{'points', 'segments'}}, got {mode!r}")
        if max_props is not None and int(max_props) < 3:
            raise ValueError(
                f"max_props must be >= 3 (or None for auto), got {max_props!r}")
        E = int(track.count.shape[0])
        stride = int(track.outer.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"track batch layout invalid: outer has {stride} slots for {E} envs")
        self._track = track
        self._spacing = float(spacing)
        self._boundary = boundary
        self._mode = mode
        self._mode_int = 0 if mode == "points" else 1
        self._E = E
        self._n_max = stride // E
        self._points = track.inner if boundary == "inner" else track.outer
        self._device = str(track.outer.device)

        if max_props is None:
            max_props = self._derive_max_props()
        self._M = int(max_props)

        dev = self._device
        n = E * self._M
        self._cum = wp.zeros(E * self._n_max, dtype=wp.float32, device=dev)
        self._props = PropSet(
            position=wp.zeros(n, dtype=wp.vec2f, device=dev),
            tangent=wp.zeros(n, dtype=wp.vec2f, device=dev),
            yaw=wp.zeros(n, dtype=wp.float32, device=dev),
            length=wp.zeros(n, dtype=wp.float32, device=dev),
            count=wp.zeros(E, dtype=wp.int32, device=dev),
            truncated=wp.zeros(E, dtype=wp.int32, device=dev),
            step=wp.zeros(E, dtype=wp.float32, device=dev),
        )

    def _derive_max_props(self) -> int:
        """Slots to cover the batch bound NOW: ceil(1.5 * max perimeter / spacing).

        Host-side readback, construction time only. The 1.5x headroom absorbs
        longer boundaries from future regenerations; rings that still exceed
        it are truncated and flagged (``PropSet.truncated``).
        """
        E, n_max = self._E, self._n_max
        pts = self._points.numpy().reshape(E, n_max, 2)
        counts = self._track.count.numpy()
        valid = self._track.valid.numpy()
        best = 0.0
        for e in range(E):
            m = int(min(counts[e], n_max))
            if valid[e] == 0 or m < 3:
                continue
            poly = pts[e, :m]
            seg = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
            best = max(best, float(seg.sum()))
        if best <= 0.0:
            raise ValueError(
                "max_props=None needs at least one valid env in the bound track "
                "batch to derive a buffer size; pass max_props explicitly")
        return max(3, int(np.ceil(1.5 * best / self._spacing)))

    def sample(self) -> PropSet:
        """Resample the bound boundary; returns the preallocated :class:`PropSet`.

        Two kernel launches (scan + place); reads the Track buffers as they
        are NOW, so it reflects the latest ``generate()`` with no rebind.
        """
        t = self._track
        p = self._props
        wp.launch(
            _scan_boundary_k, dim=self._E,
            inputs=[self._points, t.count, self._n_max, self._spacing, self._M,
                    self._cum, p.count, p.step, p.truncated],
            device=self._device,
        )
        wp.launch(
            _place_props_k, dim=self._E * self._M,
            inputs=[self._points, t.count, self._n_max, self._cum,
                    p.count, p.step, self._M, self._mode_int,
                    p.position, p.tangent, p.yaw, p.length],
            device=self._device,
        )
        _sync(self._device)
        return p
