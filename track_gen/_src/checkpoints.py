"""Ordered course checkpoints from gates (zero-copy) or track centerlines.

``CheckpointSet`` is the shared consumer contract for course-following
utilities (``track_gen.progress``): per env an ordered list of checkpoints,
each with a center ``position``, a physical crossing segment ``left <->
right``, and a unit forward ``tangent``.

Two producers:

- ``CheckpointSet.from_gates(seq)`` aliases a ``GateSequence``'s buffers
  (gates already carry exactly these fields). Zero-copy: regenerated gates
  are seen automatically. With ``gate_width=0`` the crossing segment is
  degenerate (``left == right``) and pass-through can never trigger.
- ``CheckpointSampler`` resamples the track CENTERLINE at a coarse user
  spacing (snap rule as in ``track_gen.props``); because ``Track`` polylines
  are index-aligned, each checkpoint's crossing segment is the road
  cross-section: ``left``/``right`` are ``inner``/``outer`` interpolated at
  the same centerline segment and parameter.

Results are undefined for envs with ``valid[e] == 0``; callers gate on
``Track.valid`` / ``GateSequence.valid`` as everywhere in the library.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .collision_geom import _safe_normalize2
from .props import _scan_boundary_k
from .runtime import _init, _sync
from .types import GateSequence, Track


@dataclass
class CheckpointSet:
    """Ordered per-env checkpoints, flat ``[E * M]`` per field.

    The consumer contract for :class:`track_gen.progress.ProgressTracker`.
    Sets produced by :meth:`from_gates` ALIAS the gate buffers (mutations to
    the gates are visible here); sets produced by ``CheckpointSampler`` are
    owned by the sampler and refreshed in place by ``sample()``.

    Attributes
    ----------
    position : wp.array
        ``vec2f`` checkpoint centers. NaN past ``count[e]``.
    left : wp.array
        ``vec2f`` crossing-segment endpoints (gate left / track inner).
    right : wp.array
        ``vec2f`` crossing-segment endpoints (gate right / track outer).
    tangent : wp.array
        ``vec2f`` unit forward (travel) directions.
    count : wp.array
        ``[E]`` ``int32`` real checkpoint counts. Meaningful only for envs
        with ``valid[e] == 1`` on the source batch.
    """

    position: wp.array
    left: wp.array
    right: wp.array
    tangent: wp.array
    count: wp.array

    @classmethod
    def from_gates(cls, seq: GateSequence) -> "CheckpointSet":
        """Zero-copy view of a :class:`GateSequence` as checkpoints.

        Aliases (does not copy) ``position``, ``left``, ``right``,
        ``tangent``, and ``count``. Progress state bound to this set must be
        reset after the gates are regenerated.
        """
        return cls(position=seq.position, left=seq.left, right=seq.right,
                   tangent=seq.tangent, count=seq.count)

    def clone(self) -> "CheckpointSet":
        """Return a deep copy whose Warp buffers do not alias this set."""
        return CheckpointSet(
            position=wp.clone(self.position),
            left=wp.clone(self.left),
            right=wp.clone(self.right),
            tangent=wp.clone(self.tangent),
            count=wp.clone(self.count),
        )


@wp.func
def _seg_at_arc(cum: wp.array(dtype=wp.float32), base: int, m: int,
                perim: float, s: float) -> wp.vec2f:
    """(segment index as float, lerp parameter t) at arc position s.

    Binary search for the largest i in [0, m-1] with cum[base+i] <= s; the
    closing segment (i == m-1) ends at arc length ``perim``.
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
    seg_end = perim
    if i + 1 < m:
        seg_end = cum[base + i + 1]
    seg_start = cum[base + i]
    denom = seg_end - seg_start
    t = 0.0
    if denom > 1.0e-12:
        t = wp.clamp((s - seg_start) / denom, 0.0, 1.0)
    return wp.vec2f(float(i), t)


@wp.kernel
def _place_checkpoints_k(
    center: wp.array(dtype=wp.vec2f),
    inner: wp.array(dtype=wp.vec2f),
    outer: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    cum: wp.array(dtype=wp.float32),
    cp_count: wp.array(dtype=wp.int32),
    cp_step: wp.array(dtype=wp.float32),
    max_cp: int,
    out_position: wp.array(dtype=wp.vec2f),
    out_left: wp.array(dtype=wp.vec2f),
    out_right: wp.array(dtype=wp.vec2f),
    out_tangent: wp.array(dtype=wp.vec2f),
):
    tid = wp.tid()
    e = tid // max_cp
    k = tid - e * max_cp

    n = cp_count[e]
    if k >= n:
        nan2 = wp.vec2f(wp.nan, wp.nan)
        out_position[tid] = nan2
        out_left[tid] = nan2
        out_right[tid] = nan2
        out_tangent[tid] = nan2
        return

    m = count[e]
    if m > n_max:
        m = n_max
    base = e * n_max
    step = cp_step[e]
    # Reconstructs the scan's perimeter (step = perim/n); max queried arc is
    # (n-1)*step < perim, so the closing segment is never overrun.
    perim = step * float(n)

    it = _seg_at_arc(cum, base, m, perim, float(k) * step)
    i = int(it[0])
    t = it[1]
    j = i + 1
    if j == m:
        j = 0
    # Index-aligned lerp: the same segment/parameter on all three polylines
    # gives the road cross-section at this checkpoint.
    out_position[tid] = center[base + i] + (center[base + j] - center[base + i]) * t
    out_left[tid] = inner[base + i] + (inner[base + j] - inner[base + i]) * t
    out_right[tid] = outer[base + i] + (outer[base + j] - outer[base + i]) * t
    out_tangent[tid] = _safe_normalize2(center[base + j] - center[base + i])


class CheckpointSampler:
    """Resample a track's centerline into coarse course checkpoints.

    One sampler binds one :class:`Track` and one spacing; ``sample()``
    refreshes the owned :class:`CheckpointSet` in place from the CURRENT
    track batch (no rebind after ``generate()``; two kernel launches,
    allocation-free, CUDA-graph capturable via the shared capture flag —
    ``track_gen.set_capturing``). Spacing is expected to be coarse —
    checkpoint counts similar to gate counts; nothing enforces this.
    Progress state bound to the sampled set must be reset after resampling a
    regenerated track.

    Producer diagnostics live on the sampler (not the set): ``truncated``
    (``[E]`` int32, 1 if ``max_checkpoints`` clipped the ring) and ``step``
    (``[E]`` float32 effective arc spacing). They live here rather than on
    ``CheckpointSet`` because the set has a SECOND, sampling-free producer
    (``CheckpointSet.from_gates``) that has no truncation or step concept —
    keeping the diagnostics on the sampler avoids meaningless fields on
    gate-derived sets (contrast ``track_gen.props.PropSet``, whose only
    producer is ``PropSampler``, so its diagnostics live on the result).
    """

    def __init__(self, track: Track, spacing: float,
                 max_checkpoints: "int | None" = None) -> None:
        """Bind to a :class:`Track` centerline and allocate the result buffers.

        Args:
            track: the bound track batch; ``sample()`` reads its centerline
                buffer directly on every call (no rebind needed after
                ``generate()``).
            spacing: target arc-length spacing between checkpoints (> 0);
                snapped per env so each ring closes without a seam (same
                snap rule as ``track_gen.props``).
            max_checkpoints: buffer capacity per env; ``None`` (default)
                derives it from the CURRENT batch (see
                :meth:`_derive_max_checkpoints`). Must be >= 3 when given
                explicitly.
        """
        _init()
        if not (float(spacing) > 0.0):
            raise ValueError(f"spacing must be > 0, got {spacing!r}")
        if max_checkpoints is not None and int(max_checkpoints) < 3:
            raise ValueError(
                f"max_checkpoints must be >= 3 (or None for auto), got "
                f"{max_checkpoints!r}")
        E = int(track.count.shape[0])
        stride = int(track.outer.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"track batch layout invalid: outer has {stride} slots for {E} envs")
        self._track = track
        self._spacing = float(spacing)
        self._E = E
        self._n_max = stride // E
        self._device = str(track.outer.device)

        if max_checkpoints is None:
            max_checkpoints = self._derive_max_checkpoints()
        self._M = int(max_checkpoints)

        dev = self._device
        n = E * self._M
        self._cum = wp.zeros(E * self._n_max, dtype=wp.float32, device=dev)
        self.truncated = wp.zeros(E, dtype=wp.int32, device=dev)
        self.step = wp.zeros(E, dtype=wp.float32, device=dev)
        self._set = CheckpointSet(
            position=wp.zeros(n, dtype=wp.vec2f, device=dev),
            left=wp.zeros(n, dtype=wp.vec2f, device=dev),
            right=wp.zeros(n, dtype=wp.vec2f, device=dev),
            tangent=wp.zeros(n, dtype=wp.vec2f, device=dev),
            count=wp.zeros(E, dtype=wp.int32, device=dev),
        )

    def _derive_max_checkpoints(self) -> int:
        """ceil(1.5 * max valid-env CENTERLINE perimeter / spacing), floor 3.

        Host-side readback, construction time only.
        """
        E, n_max = self._E, self._n_max
        pts = self._track.center.numpy().reshape(E, n_max, 2)
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
                "max_checkpoints=None needs at least one valid env in the "
                "bound track batch to derive a buffer size; pass "
                "max_checkpoints explicitly")
        return max(3, int(np.ceil(1.5 * best / self._spacing)))

    def sample(self) -> CheckpointSet:
        """Refresh the owned CheckpointSet from the bound Track; returns it."""
        t = self._track
        s = self._set
        wp.launch(
            _scan_boundary_k, dim=self._E,
            inputs=[t.center, t.count, self._n_max, self._spacing, self._M,
                    self._cum, s.count, self.step, self.truncated],
            device=self._device,
        )
        wp.launch(
            _place_checkpoints_k, dim=self._E * self._M,
            inputs=[t.center, t.inner, t.outer, t.count, self._n_max,
                    self._cum, s.count, self.step, self._M,
                    s.position, s.left, s.right, s.tangent],
            device=self._device,
        )
        _sync(self._device)
        return s
