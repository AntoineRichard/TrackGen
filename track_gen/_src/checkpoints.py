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
  spacing (snap rule as in ``track_gen.props``; spacing measures true 3D arc
  length on lifted tracks — the scan kernel ``_scan_boundary_k`` is shared
  with ``track_gen.props`` and identical to the planar length when z = 0);
  because ``Track`` polylines are index-aligned, each checkpoint's crossing
  segment is the road cross-section: ``left``/``right`` are ``inner``/
  ``outer`` interpolated at the same centerline segment and parameter.

Invalid envs (roadmap PR14): the sampler's set reads like a poisoned env's for
``valid[e] == 0`` (``count = 0``, NaN slots), and carries ``valid`` / ``generation``
(owned, written by ``sample()``) so its consumer (``ProgressTracker``) freezes invalid
envs and self-resets on a new generation. A ``from_gates`` set aliases the gates (their
rows and count, and their ``valid`` / ``generation``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .collision_geom import _safe_normalize3
from .props import _scan_boundary_k
from ..exec.context import LaunchContext, _call_context  # noqa: F401  (LaunchContext: re-export)
from .runtime import _BIG, _init, _sync  # noqa: F401  (_sync: legacy re-export)
from ..results import HasBand, HasGateFrames, HasGeneration, _check_and_bind


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
        ``vec3f`` checkpoint centers. NaN past ``count[e]``.
    left : wp.array
        ``vec3f`` crossing-segment endpoints (gate left / track inner).
    right : wp.array
        ``vec3f`` crossing-segment endpoints (gate right / track outer).
    tangent : wp.array
        ``vec3f`` unit forward (travel) directions. For gate-derived sets this
        is the gate's ``forward`` (the physical gate-plane normal), not the 3D
        spline tangent.
    up_half : wp.array
        ``[E * M]`` ``float32`` vertical half-openings for the crossing test:
        ``_BIG`` (unbounded) for track cross-sections, and the gate's
        ``half_size`` for gate-derived sets (aliases
        ``GateSequence.half_size``). NaN past ``count[e]``.
    count : wp.array
        ``[E]`` ``int32`` real checkpoint counts (a sampler's set: 0 for an invalid
        env).
    valid : wp.array or None
        ``[E]`` ``int32`` validity of the course the rows were built from: owned by
        ``CheckpointSampler`` (written by ``sample()``, so ``count``, the rows,
        ``valid`` and ``generation`` always describe the same ring), the gates' own
        ``valid`` for :meth:`from_gates`. ``None`` on a hand-built set (then every env
        is treated as valid). ``ProgressTracker`` freezes an env whose ``valid`` is 0.
    generation : wp.array or None
        ``[E]`` ``int32`` generation (epoch) of the course the rows were built from:
        owned by ``CheckpointSampler`` (written by ``sample()``, 0 when
        its track has none), the gates' own ``generation`` for :meth:`from_gates`.
        ``ProgressTracker`` resets an env's state when it changes (self-reset).
        ``None``: never.
    """

    position: wp.array
    left: wp.array
    right: wp.array
    tangent: wp.array
    up_half: wp.array
    count: wp.array
    valid: "wp.array | None" = None
    generation: "wp.array | None" = None

    @classmethod
    def from_gates(cls, seq: HasGateFrames) -> "CheckpointSet":
        """Zero-copy view of a :class:`~track_gen.results.GateSequence` as checkpoints.

        Aliases (does not copy) ``position``, ``left``, ``right``,
        ``forward`` (as ``tangent``: the physical gate-plane normal),
        ``half_size`` (as ``up_half``), ``count`` and, when the source has them,
        ``valid`` and ``generation``. ``seq`` is any
        :class:`~track_gen.results.HasGateFrames` with those members, checked
        here (``track_gen.results.check_source``; ``valid`` / ``generation`` are
        checked when present). A ``ProgressTracker`` on the set resets an env's state
        by itself when its ``generation`` changes (a regeneration with ``advance``);
        reset it explicitly after a reseed (see ``ProgressTracker``).
        """
        _, seq = _check_and_bind(seq, (HasGateFrames, HasGeneration),
                                 consumer="CheckpointSet.from_gates",
                                 members=("position", "left", "right", "forward",
                                          "half_size", "count"),
                                 optional=("valid", "generation"))
        return cls(position=seq.position, left=seq.left, right=seq.right,
                   tangent=seq.forward, up_half=seq.half_size, count=seq.count,
                   valid=seq.valid, generation=seq.generation)

    def clone(self) -> "CheckpointSet":
        """Return a deep copy whose Warp buffers do not alias this set (``valid`` /
        ``generation`` too, when set)."""
        return CheckpointSet(
            position=wp.clone(self.position),
            left=wp.clone(self.left),
            right=wp.clone(self.right),
            tangent=wp.clone(self.tangent),
            up_half=wp.clone(self.up_half),
            count=wp.clone(self.count),
            valid=None if self.valid is None else wp.clone(self.valid),
            generation=None if self.generation is None else wp.clone(self.generation),
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
    center: wp.array(dtype=wp.vec3f),
    inner: wp.array(dtype=wp.vec3f),
    outer: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    cum: wp.array(dtype=wp.float32),
    cp_count: wp.array(dtype=wp.int32),
    cp_step: wp.array(dtype=wp.float32),
    max_cp: int,
    out_position: wp.array(dtype=wp.vec3f),
    out_left: wp.array(dtype=wp.vec3f),
    out_right: wp.array(dtype=wp.vec3f),
    out_tangent: wp.array(dtype=wp.vec3f),
    out_up_half: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
    generation: wp.array(dtype=wp.int32),
    has_gen: int,
    out_valid: wp.array(dtype=wp.int32),
    out_generation: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    e = tid // max_cp
    k = tid - e * max_cp

    if k == 0:
        # The set's own valid / generation (roadmap PR14): the ring's, written with it.
        out_valid[e] = valid[e]
        if has_gen != 0:
            out_generation[e] = generation[e]
        else:
            out_generation[e] = 0

    n = cp_count[e]
    if k >= n:
        nan3 = wp.vec3f(wp.nan, wp.nan, wp.nan)
        out_position[tid] = nan3
        out_left[tid] = nan3
        out_right[tid] = nan3
        out_tangent[tid] = nan3
        out_up_half[tid] = wp.nan
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
    out_tangent[tid] = _safe_normalize3(center[base + j] - center[base + i])
    # Track cross-sections are vertically unbounded for the crossing test.
    out_up_half[tid] = _BIG


def _readback_max_checkpoints(track, E: int, n_max: int, spacing: float) -> int:
    """``ceil(1.5 * max valid-env centerline perimeter / spacing)``, floor 3: the
    capacity read back from the CURRENT batch (host readback, construction only; the
    legacy ``CheckpointSampler(max_checkpoints=None)`` default, and ``Course``'s when
    its plan declares no bound)."""
    pts = track.center.numpy().reshape(E, n_max, 3)
    counts = track.count.numpy()
    valid = track.valid.numpy()
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
    return max(3, int(np.ceil(1.5 * best / spacing)))


class CheckpointSampler:
    """Resample a track's centerline into coarse course checkpoints.

    One sampler binds one :class:`~track_gen.results.Track` and one spacing; ``sample()``
    refreshes the owned :class:`CheckpointSet` in place from the CURRENT
    track batch (no rebind after ``generate()``; two kernel launches,
    allocation-free and host-sync-free, so CUDA-graph capturable as-is).
    Spacing is expected to be coarse —
    checkpoint counts similar to gate counts; nothing enforces this.
    Progress state bound to the sampled set must be reset after resampling a
    regenerated track.

    Producer diagnostics live on the sampler (not the set): ``truncated``
    (``[E]`` int32, 1 if ``max_checkpoints`` clipped the ring) and ``step``
    (``[E]`` float32 effective arc spacing). An env with ``valid[e] == 0`` reads like a
    poisoned one (``count = 0``, ``step`` NaN, ``truncated = 0``, NaN slots, the set's
    ``valid = 0``). They live here rather than on
    ``CheckpointSet`` because the set has a SECOND, sampling-free producer
    (``CheckpointSet.from_gates``) that has no truncation or step concept —
    keeping the diagnostics on the sampler avoids meaningless fields on
    gate-derived sets (contrast ``track_gen.props.PropSet``, whose only
    producer is ``PropSampler``, so its diagnostics live on the result).
    """

    #: The topologies this component supports (closed loops only in 0.x).
    topology = "closed"

    def __init__(self, track: HasBand, spacing: float,
                 max_checkpoints: "int | None" = None) -> None:
        """Bind to a :class:`~track_gen.results.Track` centerline and allocate the
        result buffers.

        Args:
            track: the bound track batch; ``sample()`` reads its centerline
                buffer directly on every call (no rebind needed after
                ``generate()``).
            spacing: target arc-length spacing between checkpoints (> 0);
                snapped per env so each ring closes without a seam (same
                snap rule as ``track_gen.props``).
            max_checkpoints: buffer capacity per env; ``None`` (default)
                derives it from the CURRENT batch (see
                ``_derive_max_checkpoints``: a construction-time readback, kept
                as a supported convenience). Must be >= 3 when given explicitly.
                The size-independent bound ``max(3, ceil(max_loop_length /
                spacing))`` from the producer's plan is the default of
                the component route (``track_gen.tools.Checkpoints(spacing)``
                attached through ``ComponentContext.for_producer(generator)``) and of
                ``Course``.

        The track's ``generation`` is read when present (checked like the required
        members): ``sample()`` copies it (and ``valid``) into the set.
        """
        _init()
        if not (float(spacing) > 0.0):
            raise ValueError(f"spacing must be > 0, got {spacing!r}")
        if max_checkpoints is not None and int(max_checkpoints) < 3:
            raise ValueError(
                f"max_checkpoints must be >= 3 (or None for auto), got "
                f"{max_checkpoints!r}")
        # The members checked here are the ones bound (read once: stable pointers).
        lay, self._track = _check_and_bind(track, (HasBand, HasGeneration),
                                           consumer="CheckpointSampler",
                                           members=("center", "inner", "outer", "count",
                                                    "valid"),
                                           optional=("generation",),
                                           topology=self.topology)
        E = int(lay.num_envs)
        self._spacing = float(spacing)
        self._E = E
        self._n_max = int(lay.stride)
        self._device = lay.device

        if max_checkpoints is None:
            max_checkpoints = self._derive_max_checkpoints()
        self._M = int(max_checkpoints)

        dev = self._device
        n = E * self._M
        self._cum = wp.zeros(E * self._n_max, dtype=wp.float32, device=dev)
        self.truncated = wp.zeros(E, dtype=wp.int32, device=dev)
        self.step = wp.zeros(E, dtype=wp.float32, device=dev)
        self._set = CheckpointSet(
            position=wp.zeros(n, dtype=wp.vec3f, device=dev),
            left=wp.zeros(n, dtype=wp.vec3f, device=dev),
            right=wp.zeros(n, dtype=wp.vec3f, device=dev),
            tangent=wp.zeros(n, dtype=wp.vec3f, device=dev),
            up_half=wp.zeros(n, dtype=wp.float32, device=dev),
            count=wp.zeros(E, dtype=wp.int32, device=dev),
            valid=wp.zeros(E, dtype=wp.int32, device=dev),
            generation=wp.zeros(E, dtype=wp.int32, device=dev),
        )

    @property
    def max_checkpoints(self) -> int:
        """The per-env capacity (slots per env) of the :class:`CheckpointSet`
        (read-only)."""
        return self._M

    @property
    def result(self) -> CheckpointSet:
        """The preallocated :class:`CheckpointSet` :meth:`sample` refreshes and returns
        (the same instance for the sampler's lifetime)."""
        return self._set

    def _derive_max_checkpoints(self) -> int:
        """ceil(1.5 * max valid-env CENTERLINE perimeter / spacing), floor 3.

        Host-side readback, construction time only.
        """
        return _readback_max_checkpoints(self._track, self._E, self._n_max,
                                         self._spacing)

    def sample(self, stream=None, *, ctx=None) -> CheckpointSet:
        """Refresh the owned CheckpointSet from the bound Track; returns it.

        Args:
            stream: optional wp.Stream (or 'torch') to run on; by default the
                stream is resolved per track_gen.exec (an enclosing
                torch.cuda.graph capture is adopted, else Warp's current
                stream).
            ctx: optional :class:`~track_gen.exec.LaunchContext` on the sampler's
                device to emit on (exclusive with ``stream=``: ``TypeError``).
        """
        ctx = _call_context(self._device, stream, ctx)
        t = self._track
        s = self._set
        gen = t.generation
        ctx.launch(
            _scan_boundary_k, self._E,
            [t.center, t.count, t.valid, self._n_max, self._spacing, self._M,
             self._cum, s.count, self.step, self.truncated],
        )
        ctx.launch(
            _place_checkpoints_k, self._E * self._M,
            [t.center, t.inner, t.outer, t.count, self._n_max,
             self._cum, s.count, self.step, self._M,
             s.position, s.left, s.right, s.tangent, s.up_half,
             t.valid, t.count if gen is None else gen, 0 if gen is None else 1,
             s.valid, s.generation],
        )
        ctx.sync()
        return s
