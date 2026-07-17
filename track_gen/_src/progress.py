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
after construction) can never emit a spurious crossing. Callers MUST reset
after regenerating the bound course. Results are undefined for envs with
``valid[e] == 0`` on the source batch.

Like the sibling utilities, ``update()``/``reset()`` perform no host sync
while capturing is enabled (``track_gen.set_capturing``), so they are
CUDA-graph capturable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .checkpoints import CheckpointSet
from .collision_geom import _is_nan3, _plane_pass
from .runtime import _check_arr, _init, _sync


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
        NaN for envs with no checkpoints.
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
        """Bind to a :class:`CheckpointSet` and allocate the tracker's state.

        Args:
            checkpoints: the ordered per-env checkpoint set to track against
                (from ``CheckpointSet.from_gates`` or
                ``CheckpointSampler.sample()``). Aliased, not copied — later
                mutation (e.g. a resampled or regenerated set) is seen on
                the next ``update()``.
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
        self._cps = checkpoints
        self._E = E
        self._M = stride // E
        self._device = str(checkpoints.position.device)

        self._bound_pos: "wp.array | None" = None
        if position is not None:
            self.bind(position)

        dev = self._device
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

    def _validate_position(self, position) -> None:
        _check_arr("position", position, (self._E,), wp.vec3f, self._device)

    def bind(self, position: wp.array) -> None:
        """Bind (or rebind) a stable ``[E]`` vec3f position buffer.

        After binding, ``update()`` takes no arguments and reads the buffer
        in place. Validation happens here, once; the array must keep the
        same ``.ptr`` for the binding's lifetime (CUDA-graph contract).
        """
        self._validate_position(position)
        self._bound_pos = position

    def update(self, position: "wp.array | None" = None) -> ProgressEvents:
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
        c = self._cps
        ev = self._events
        wp.launch(
            _progress_update_k, dim=self._E,
            inputs=[c.position, c.left, c.right, c.tangent, c.up_half,
                    c.count, self._M,
                    pos, self._prev_pos, self._next, self._laps, self._progress,
                    ev.passed, ev.checkpoint_passed, ev.next_checkpoint,
                    ev.laps, ev.progress, ev.wrong_way, ev.wrong_checkpoint,
                    ev.dist_to_next],
            device=self._device,
        )
        _sync(self._device)
        return ev

    def reset(self, mask: wp.array) -> None:
        """Clear state where ``mask[e] == 1`` (``[E]`` int32); arms the NaN
        previous-position sentinel so the next update cannot emit a spurious
        crossing. Required after regenerating the bound course."""
        _check_arr("mask", mask, (self._E,), wp.int32, self._device)
        wp.launch(
            _progress_reset_k, dim=self._E,
            inputs=[mask, self._prev_pos, self._next, self._laps, self._progress],
            device=self._device,
        )
        _sync(self._device)
