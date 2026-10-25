"""Stateful course-progress tracking over a CheckpointSet (gates or track).

``ProgressTracker`` owns per-env device state (previous position, next
checkpoint, laps, total progress) and advances it in ONE fused kernel per
``update()``: swept-segment plane-crossing detection against the target's
gate plane within its opening (the left-right extent, and vertically within
``CheckpointSet.up_half`` — ``_BIG`` keeps track cross-sections unbounded, so
planar behavior is unchanged for planar motion), wrong-way and
wrong-checkpoint events, and the distance to
the next checkpoint center (``dist_to_next``) for delta-distance rewards
(``r_t = dist[t-1] - dist[t]``, differenced by the caller).

The tracker can LATCH onto a stable, user-owned position buffer at
construction (``position=...``): ``update()`` then takes no arguments and
reads the buffer in place — the natural CUDA-graph pattern (sim writes
poses, replays the captured update). All tracker-owned buffers are
preallocated with stable pointers.

Reset contract: ``reset(mask)`` clears state where ``mask[e] == 1`` and arms
the NaN previous-position sentinel, so the first update after a reset (or
after construction) can never emit a spurious crossing. Since  the
tracker also resets an env BY ITSELF (the same writes, on device, before reading the
state) when the set's ``generation`` changes: a ``CheckpointSampler`` set changes it at
``sample()`` (so an ``update()`` between a regeneration and the resample does not reset
against the old ring), a ``from_gates`` set with the gates. ``generation`` is the epoch:
a reseed may or may not change it (``Course.generate(seeds=)`` zeroes non-zero epochs, a
bare ``rng.set_seeds_warp`` changes nothing), and ``reset_epoch()`` followed by the same
number of advances leaves it equal (ABA): callers that reseed or roll epochs back must
still ``reset(mask)`` (``Course`` does). A set without ``generation`` never self-resets.

Invalid envs: an env whose set has ``valid[e] == 0`` is FROZEN: its
state is kept (``next_checkpoint`` / ``laps`` / ``progress`` report it), ``passed = 0``,
``checkpoint_passed = -1``, ``wrong_way = 0``, ``wrong_checkpoint = -1``,
``dist_to_next = 0``, and the previous position is re-armed (NaN), so the env cannot
emit a spurious crossing when it becomes valid again. ``dist_to_next`` is NaN only for a
valid env with no checkpoints (a hand-built set): ``valid`` is the invalid-env signal.

Like the sibling utilities, ``update()``/``reset()`` perform no host sync,
so they are CUDA-graph capturable as-is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .checkpoints import CheckpointSet
from .collision_geom import _is_nan3, _plane_pass
from ..exec.context import (LaunchContext, _call_context, _note_rebind,  # noqa: F401
                            _probe_recorded)
from .runtime import _check_arr, _init, _sync  # noqa: F401  (_sync: legacy re-export)
from ..results import HasCheckpoints, HasGeneration, HasValidity, _check_and_bind


@dataclass
class ProgressEvents:
    """Per-env progress events, all ``[E]``; overwritten in place per update.

    .. warning::

        ``ProgressTracker.update()`` returns the SAME instance every call.
        ``clone()`` for snapshots.

    Attributes
    ----------
    passed : wp.array
        ``int32`` — 1 iff the target checkpoint was crossed forward this step.
    checkpoint_passed : wp.array
        ``int32`` — index of the checkpoint passed this step, -1 otherwise.
    next_checkpoint : wp.array
        ``int32`` — current target AFTER any advance this step.
    laps : wp.array
        ``int32`` — completed laps.
    progress : wp.array
        ``int32`` — total checkpoints passed since construction/reset.
    wrong_way : wp.array
        ``int32`` — 1 iff the target was crossed BACKWARD this step.
    wrong_checkpoint : wp.array
        ``int32`` — index of a non-target checkpoint crossed this step
        (either direction; first in index order), -1 otherwise.
    dist_to_next : wp.array
        ``float32`` — ``|position - next checkpoint center|`` after any advance.
        NaN for a (valid) env with no checkpoints; 0 for an invalid env (frozen).
    """

    passed: wp.array
    checkpoint_passed: wp.array
    next_checkpoint: wp.array
    laps: wp.array
    progress: wp.array
    wrong_way: wp.array
    wrong_checkpoint: wp.array
    dist_to_next: wp.array

    def clone(self) -> "ProgressEvents":
        """Return a deep copy whose Warp buffers do not alias this result."""
        return ProgressEvents(
            passed=wp.clone(self.passed),
            checkpoint_passed=wp.clone(self.checkpoint_passed),
            next_checkpoint=wp.clone(self.next_checkpoint),
            laps=wp.clone(self.laps),
            progress=wp.clone(self.progress),
            wrong_way=wp.clone(self.wrong_way),
            wrong_checkpoint=wp.clone(self.wrong_checkpoint),
            dist_to_next=wp.clone(self.dist_to_next),
        )


@wp.kernel
def _progress_update_k(
    cp_position: wp.array(dtype=wp.vec3f),
    cp_left: wp.array(dtype=wp.vec3f),
    cp_right: wp.array(dtype=wp.vec3f),
    cp_tangent: wp.array(dtype=wp.vec3f),
    cp_up_half: wp.array(dtype=wp.float32),
    cp_count: wp.array(dtype=wp.int32),
    max_cp: int,
    valid: wp.array(dtype=wp.int32),
    has_valid: int,
    generation: wp.array(dtype=wp.int32),
    has_gen: int,
    seen: wp.array(dtype=wp.int32),
    position: wp.array(dtype=wp.vec3f),
    prev_pos: wp.array(dtype=wp.vec3f),
    next_cp: wp.array(dtype=wp.int32),
    laps: wp.array(dtype=wp.int32),
    progress: wp.array(dtype=wp.int32),
    out_passed: wp.array(dtype=wp.int32),
    out_cp_passed: wp.array(dtype=wp.int32),
    out_next: wp.array(dtype=wp.int32),
    out_laps: wp.array(dtype=wp.int32),
    out_progress: wp.array(dtype=wp.int32),
    out_wrong_way: wp.array(dtype=wp.int32),
    out_wrong_cp: wp.array(dtype=wp.int32),
    out_dist: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    if has_gen != 0:
        gen = generation[e]
        if seen[e] != gen:
            # The env got a new course since this state was built (roadmap PR14):
            # exactly _progress_reset_k's writes, before the state is read.
            seen[e] = gen
            prev_pos[e] = wp.vec3f(wp.nan, wp.nan, wp.nan)
            next_cp[e] = 0
            laps[e] = 0
            progress[e] = 0
    if has_valid != 0:
        if valid[e] == 0:
            # Invalid env: frozen (roadmap PR14), prev_pos re-armed (no spurious event
            # when it becomes valid again).
            prev_pos[e] = wp.vec3f(wp.nan, wp.nan, wp.nan)
            out_passed[e] = 0
            out_cp_passed[e] = -1
            out_next[e] = next_cp[e]
            out_laps[e] = laps[e]
            out_progress[e] = progress[e]
            out_wrong_way[e] = 0
            out_wrong_cp[e] = -1
            out_dist[e] = 0.0
            return
    pos = position[e]
    n = cp_count[e]
    base = e * max_cp

    passed = int(0)
    cp_passed = int(-1)
    wway = int(0)
    wcp = int(-1)

    if n < 1:
        prev_pos[e] = pos
        out_passed[e] = 0
        out_cp_passed[e] = -1
        out_next[e] = next_cp[e]
        out_laps[e] = laps[e]
        out_progress[e] = progress[e]
        out_wrong_way[e] = 0
        out_wrong_cp[e] = -1
        out_dist[e] = wp.nan
        return

    g = next_cp[e]
    if g >= n or g < 0:
        # Defensive clamp: course regenerated without reset (documented as
        # caller error, but never index out of the real range).
        g = 0

    prev = prev_pos[e]
    if _is_nan3(prev) == 0:
        c = _plane_pass(prev, pos, cp_tangent[base + g],
                        cp_left[base + g], cp_right[base + g],
                        cp_up_half[base + g])
        if c == 1:
            passed = int(1)
            cp_passed = g
        elif c == -1:
            wway = int(1)
        # Wrong-checkpoint scan vs the ORIGINAL target g: a double-jump
        # advances g and flags the second crossing in this same update.
        for i in range(n):
            if i != g and wcp == -1:
                if _plane_pass(prev, pos, cp_tangent[base + i],
                               cp_left[base + i], cp_right[base + i],
                               cp_up_half[base + i]) != 0:
                    wcp = i

    ng = g
    lp = laps[e]
    pr = progress[e]
    if passed == 1:
        ng = g + 1
        pr = pr + 1
        if ng == n:
            ng = 0
            lp = lp + 1

    prev_pos[e] = pos
    next_cp[e] = ng
    laps[e] = lp
    progress[e] = pr

    out_passed[e] = passed
    out_cp_passed[e] = cp_passed
    out_next[e] = ng
    out_laps[e] = lp
    out_progress[e] = pr
    out_wrong_way[e] = wway
    out_wrong_cp[e] = wcp
    out_dist[e] = wp.length(cp_position[base + ng] - pos)


@wp.kernel
def _progress_reset_k(
    mask: wp.array(dtype=wp.int32),
    prev_pos: wp.array(dtype=wp.vec3f),
    next_cp: wp.array(dtype=wp.int32),
    laps: wp.array(dtype=wp.int32),
    progress: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    if mask[e] != 0:
        prev_pos[e] = wp.vec3f(wp.nan, wp.nan, wp.nan)
        next_cp[e] = 0
        laps[e] = 0
        progress[e] = 0


class ProgressTracker:
    """Track ordered progress of one agent per env through a CheckpointSet.

    See the module docstring for semantics. Construct with ``position=`` (a
    stable ``[E]`` vec3f wp.array owned by your sim) for bound mode —
    ``update()`` then takes no arguments and reads the buffer in place; or
    leave unbound and pass positions per call. Mixing modes raises
    ``ValueError``.
    """

    def __init__(self, checkpoints: CheckpointSet,
                 position: "wp.array | None" = None) -> None:
        """Bind to a :class:`~track_gen.checkpoints.CheckpointSet` and allocate the
        tracker's state.

        Args:
            checkpoints: the ordered per-env checkpoint set to track against
                (from ``CheckpointSet.from_gates`` or
                ``CheckpointSampler.sample()``; any
                :class:`~track_gen.results.HasCheckpoints`). Aliased, not copied —
                later mutation (e.g. a resampled or regenerated set) is seen on
                the next ``update()``. Its ``valid`` / ``generation`` are read when
                present (``[E]`` int32 on the set's device, checked here): invalid
                envs are frozen, and an env resets by itself when its
                ``generation`` changes. The state of that check starts at the set's
                ``generation`` now (a device copy on the call's stream: the ambient
                context or Warp's current stream), so the first ``update()`` does not
                reset.
            position: optional stable ``[E]`` vec3f buffer to bind at
                construction (bound mode); equivalent to calling
                :meth:`bind` right after construction.
        """
        _init()
        E = int(checkpoints.count.shape[0])
        stride = int(checkpoints.position.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"checkpoint layout invalid: {stride} slots for {E} envs")
        for name in ("position", "left", "right", "tangent"):
            arr = getattr(checkpoints, name)
            if not isinstance(arr, wp.array) or arr.shape != (stride,) \
                    or arr.dtype is not wp.vec3f:
                raise ValueError(
                    f"checkpoints.{name} must be a [{stride}] vec3f wp.array")
            if str(arr.device) != str(checkpoints.position.device):
                raise ValueError(
                    f"checkpoints.{name} is on {arr.device}, expected "
                    f"{checkpoints.position.device}")
        _check_arr("checkpoints.up_half", checkpoints.up_half, (stride,),
                   wp.float32, str(checkpoints.position.device))
        if checkpoints.count.dtype is not wp.int32 \
                or str(checkpoints.count.device) != str(checkpoints.position.device):
            raise ValueError(
                f"checkpoints.count must be a [{E}] int32 wp.array on "
                f"{checkpoints.position.device}")
        # The optional members (roadmap PR14), checked against the set's count / device
        # with the required-member messages; read once.
        _, opt = _check_and_bind(checkpoints, (HasCheckpoints, HasValidity, HasGeneration),
                                 consumer="ProgressTracker", topology="any",
                                 members=("count",), optional=("valid", "generation"))
        self._cps = checkpoints
        self._valid = opt.valid
        self._generation = opt.generation
        self._E = E
        self._M = stride // E
        self._device = str(checkpoints.position.device)
        self._recorded = False

        self._bound_pos: "wp.array | None" = None
        if position is not None:
            self.bind(position)

        dev = self._device
        # Generation-aware self-reset state: the generation the state was built for.
        self._seen = wp.empty(E, dtype=wp.int32, device=dev)
        ctx = LaunchContext.for_call(dev)
        if self._generation is not None:
            ctx.copy(self._seen, self._generation)
        else:
            ctx.fill(self._seen, 0)
        self._prev_pos = wp.array(np.full((E, 3), np.nan, np.float32),
                                  dtype=wp.vec3f, device=dev)
        self._next = wp.zeros(E, dtype=wp.int32, device=dev)
        self._laps = wp.zeros(E, dtype=wp.int32, device=dev)
        self._progress = wp.zeros(E, dtype=wp.int32, device=dev)
        self._events = ProgressEvents(
            passed=wp.zeros(E, dtype=wp.int32, device=dev),
            checkpoint_passed=wp.zeros(E, dtype=wp.int32, device=dev),
            next_checkpoint=wp.zeros(E, dtype=wp.int32, device=dev),
            laps=wp.zeros(E, dtype=wp.int32, device=dev),
            progress=wp.zeros(E, dtype=wp.int32, device=dev),
            wrong_way=wp.zeros(E, dtype=wp.int32, device=dev),
            wrong_checkpoint=wp.zeros(E, dtype=wp.int32, device=dev),
            dist_to_next=wp.zeros(E, dtype=wp.float32, device=dev),
        )

    @property
    def result(self) -> ProgressEvents:
        """The preallocated :class:`ProgressEvents` :meth:`update` writes and returns
        (the same instance for the tracker's lifetime)."""
        return self._events

    @property
    def next_checkpoint_state(self) -> wp.array:
        """Aliased read-only view of the ``[E]`` int32 current-target state.

        Exposes the tracker's live ``_next`` buffer (updated in place by every
        ``update()``) for consumers that window their queries around the
        current target checkpoint (e.g. ``FrameChecker.bind_window``). Do NOT
        mutate — the tracker owns this state.
        """
        return self._next

    def _validate_position(self, position) -> None:
        _check_arr("position", position, (self._E,), wp.vec3f, self._device)

    def bind(self, position: wp.array) -> None:
        """Bind (or rebind) a stable ``[E]`` vec3f position buffer.

        After binding, ``update()`` takes no arguments and reads the buffer
        in place. Validation happens here, once; the array must keep the
        same ``.ptr`` for the binding's lifetime (CUDA-graph contract): replacing
        it after ``update()`` was recorded into a capture warns
        (:class:`~track_gen.exec.RebindAfterCaptureWarning`).
        """
        self._validate_position(position)
        old = None if self._bound_pos is None else (self._bound_pos,)
        _note_rebind(self, old, (position,), "update", stacklevel=2)
        self._bound_pos = position

    def update(self, position: "wp.array | None" = None,
               stream=None, *, ctx=None) -> ProgressEvents:
        """Advance one step; returns the tracker's preallocated events.

        Bound mode (constructed with ``position=``): call with no arguments.
        Per-call mode: pass the ``[E]`` vec3f position array — the SAME
        array (identical ``.ptr``) must be used across a CUDA-graph capture
        and its replays.

        NaN-position semantics: a NaN ``position[e]`` PAUSES env ``e`` for
        this step rather than erroring — no crossing/wrong-way/wrong-checkpoint
        event can fire (the crossing test always misses against a NaN
        endpoint), ``dist_to_next[e]`` is NaN, and the NaN gets written into
        ``prev_pos[e]``, which re-arms the same sentinel ``reset()`` uses: the
        FIRST update after a finite position resumes also reports no event
        (it only re-primes ``prev_pos``), and normal crossing detection
        resumes the update after that. Useful for envs mid-teleport or
        awaiting a fresh spawn without a full ``reset()``.

        Args:
            position: ``[E]`` vec3f positions (per-call mode only).
            stream: optional wp.Stream (or 'torch') to run on; by default the
                stream is resolved per track_gen.exec (an enclosing
                torch.cuda.graph capture is adopted, else Warp's current
                stream).
            ctx: optional :class:`~track_gen.exec.LaunchContext` on the tracker's
                device to emit on (exclusive with ``stream=``: ``TypeError``).
        """
        if self._bound_pos is not None:
            if position is not None:
                raise ValueError(
                    "tracker is bound to a position buffer; call update() "
                    "with no arguments")
            pos = self._bound_pos
        else:
            if position is None:
                raise ValueError(
                    "tracker is not bound; pass position to update() or "
                    "construct with position=")
            self._validate_position(position)
            pos = position
        ctx = _call_context(self._device, stream, ctx)
        _probe_recorded(self, ctx)
        c = self._cps
        ev = self._events
        valid, gen = self._valid, self._generation
        ctx.launch(
            _progress_update_k, self._E,
            [c.position, c.left, c.right, c.tangent, c.up_half,
             c.count, self._M,
             c.count if valid is None else valid, 0 if valid is None else 1,
             c.count if gen is None else gen, 0 if gen is None else 1, self._seen,
             pos, self._prev_pos, self._next, self._laps, self._progress,
             ev.passed, ev.checkpoint_passed, ev.next_checkpoint,
             ev.laps, ev.progress, ev.wrong_way, ev.wrong_checkpoint,
             ev.dist_to_next],
        )
        ctx.sync()
        return ev

    def reset(self, mask: wp.array, stream=None, *, ctx=None) -> None:
        """Clear state where ``mask[e] == 1`` (``[E]`` int32); arms the NaN
        previous-position sentinel so the next update cannot emit a spurious
        crossing. Required after a reseed or an epoch roll-back of the bound
        course, and on respawns; after a regeneration that changes an env's
        ``generation`` the tracker resets it by itself (byte-identical to this call).

        Args:
            mask: ``[E]`` int32 reset mask.
            stream: optional wp.Stream (or 'torch') to run on; by default the
                stream is resolved per track_gen.exec (an enclosing
                torch.cuda.graph capture is adopted, else Warp's current
                stream).
            ctx: optional :class:`~track_gen.exec.LaunchContext` on the tracker's
                device to emit on (exclusive with ``stream=``: ``TypeError``).
        """
        _check_arr("mask", mask, (self._E,), wp.int32, self._device)
        ctx = _call_context(self._device, stream, ctx)
        ctx.launch(
            _progress_reset_k, self._E,
            [mask, self._prev_pos, self._next, self._laps, self._progress],
        )
        ctx.sync()
