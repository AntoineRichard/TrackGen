"""Box-vs-disc obstacle collision (gate posts, physical cones, point props).

``DiscChecker`` binds a flat ``[E * D]`` vec3f array of disc centers (ALIASED
— regenerated buffers are seen automatically) and a scalar radius, and
queries batches of oriented boxes exactly like ``CollisionChecker``: hit iff
the xy distance from the disc center to the solid planar OBB is <= radius
(collision semantics stay planar; z is ignored). The deepest
penetrating disc is reported per box.

Disc validity: pass ``count`` (``[E]`` int32 real discs per env) OR rely on
NaN-marked padding (slots with NaN centers are skipped) — GateSequence
``left``/``right`` arrays interleaved as posts work out of the box.

Per-step inputs can be bound at construction (``position=``, ``orientation=``,
``half_extents=`` — all three or none): ``query()`` then takes no arguments
and reads the stable buffers in place (the CUDA-graph pattern).

Like the sibling ``CollisionChecker``, ``query()`` performs no host sync
while capturing is enabled (``track_gen.set_capturing``), so it is
CUDA-graph capturable.
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .collision_geom import (
    _is_nan3,
    _point_to_local_box_dist,
    _quat_yaw,
    _rot2,
    _safe_normalize2,
)
from .runtime import _check_arr, _init, _sync


@dataclass
class DiscContact:
    """Batched box-vs-disc result, flat ``[E * max_boxes]`` per field.

    .. warning::

        ``DiscChecker.query()`` returns the SAME instance every call and
        overwrites its buffers in place; ``clone()`` for snapshots.

    Attributes
    ----------
    hit : wp.array
        ``int32`` — 1 iff any disc touches the box (distance <= radius).
    disc : wp.array
        ``int32`` — index of the deepest-penetrating disc, -1 when none.
    depth : wp.array
        ``float32`` — penetration depth of that disc (>= 0; 0 on graze or
        no hit). Saturates at ``radius`` once the disc center lies inside
        the box (ties broken by lowest disc index). Cf.
        :class:`BoxContact.distance <track_gen.collision.BoxContact>`, which
        reports a SIGNED clearance (positive margin / negative penetration)
        rather than an unsigned depth.
    nearest : wp.array
        ``vec2f`` — point on that disc's boundary nearest the box
        (approximate when the disc center lies deep inside the box). NaN
        when no hit or the box slot is inactive.
    """

    hit: wp.array
    disc: wp.array
    depth: wp.array
    nearest: wp.array

    def clone(self) -> "DiscContact":
        """Return a deep copy whose Warp buffers do not alias this result."""
        return DiscContact(
            hit=wp.clone(self.hit),
            disc=wp.clone(self.disc),
            depth=wp.clone(self.depth),
            nearest=wp.clone(self.nearest),
        )


@wp.kernel
def _box_query_discs_k(
    discs: wp.array(dtype=wp.vec3f),
    disc_count: wp.array(dtype=wp.int32),
    d_max: int,
    radius: float,
    max_boxes: int,
    position: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    half_extents: wp.array(dtype=wp.vec2f),
    out_hit: wp.array(dtype=wp.int32),
    out_disc: wp.array(dtype=wp.int32),
    out_depth: wp.array(dtype=wp.float32),
    out_nearest: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    e = t // max_boxes
    nan2 = wp.vec2f(wp.nan, wp.nan)

    pos3 = position[t]
    if _is_nan3(pos3) == 1:
        out_hit[t] = 0
        out_disc[t] = -1
        out_depth[t] = 0.0
        out_nearest[t] = nan2
        return
    # Planar collision semantics: project the box pose to xy.
    pos = wp.vec2f(pos3[0], pos3[1])

    yw = _quat_yaw(orientation[t])
    he = half_extents[t]
    nd = disc_count[e]
    if nd > d_max:
        nd = d_max
    base = e * d_max

    best = int(-1)
    best_pen = float(0.0)
    for i in range(nd):
        c3 = discs[base + i]
        if _is_nan3(c3) == 0:
            c = wp.vec2f(c3[0], c3[1])
            q = _rot2(-yw, c - pos)
            pen = radius - _point_to_local_box_dist(q, he)
            if pen >= 0.0 and (best == -1 or pen > best_pen):
                best = i
                best_pen = pen

    if best == -1:
        out_hit[t] = 0
        out_disc[t] = -1
        out_depth[t] = 0.0
        out_nearest[t] = nan2
        return

    c3 = discs[base + best]
    c = wp.vec2f(c3[0], c3[1])
    # Closest point of the box to the disc center, then step back onto the
    # disc boundary toward it (approximate when c is deep inside the box).
    q = _rot2(-yw, c - pos)
    qcl = wp.vec2f(wp.clamp(q[0], -he[0], he[0]), wp.clamp(q[1], -he[1], he[1]))
    wcl = pos + _rot2(yw, qcl)
    out_hit[t] = 1
    out_disc[t] = best
    out_depth[t] = best_pen
    out_nearest[t] = c + _safe_normalize2(wcl - c) * radius


class DiscChecker:
    """Batched oriented-box vs disc-obstacle checker (collision family).

    See the module docstring. ``num_envs`` is required when ``count`` is not
    given (a flat disc array alone cannot determine the env split).
    """

    def __init__(self, discs: wp.array, radius: float, max_boxes: int,
                 num_envs: "int | None" = None,
                 count: "wp.array | None" = None,
                 position: "wp.array | None" = None,
                 orientation: "wp.array | None" = None,
                 half_extents: "wp.array | None" = None) -> None:
        """Bind a flat disc array and allocate the result buffers.

        Args:
            discs: ``[E * D]`` vec3f disc centers (ALIASED — regenerated
                buffers are seen automatically). NaN-marked slots are
                skipped.
            radius: shared disc radius (> 0); every disc uses the same
                radius.
            max_boxes: query stride (boxes per env); must be >= 1.
            num_envs: required when ``count`` is not given (a flat disc
                array alone cannot determine the env split); must match
                ``count.shape[0]`` when both are given.
            count: optional ``[E]`` int32 real-disc-count array (aliased,
                stable user buffer); omit to rely on NaN padding alone.
            position, orientation, half_extents: optional constructor binding
                (all-or-none); equivalent to calling :meth:`bind_inputs`
                right after construction.
        """
        _init()
        if not (float(radius) > 0.0):
            raise ValueError(f"radius must be > 0, got {radius!r}")
        if int(max_boxes) < 1:
            raise ValueError(f"max_boxes must be >= 1, got {max_boxes!r}")
        if count is not None:
            E = int(count.shape[0])
            if num_envs is not None and int(num_envs) != E:
                raise ValueError(
                    f"count.shape[0] ({E}) must equal num_envs ({int(num_envs)})")
        elif num_envs is not None:
            E = int(num_envs)
        else:
            raise ValueError("pass num_envs (or a count array): a flat disc "
                             "array alone cannot determine the env split")
        total = int(discs.shape[0])
        if E < 1 or total % E != 0:
            raise ValueError(
                f"discs length {total} not divisible by {E} envs")
        self._discs = discs
        self._radius = float(radius)
        self._E = E
        self._D = total // E
        self._B = int(max_boxes)
        self._device = str(discs.device)
        if count is not None:
            self._count = count            # aliased: stable user buffer
        else:
            self._count = wp.full(E, self._D, dtype=wp.int32,
                                  device=self._device)

        self._bound = None
        if position is not None or orientation is not None \
                or half_extents is not None:
            if position is None or orientation is None or half_extents is None:
                raise ValueError(
                    "bind all of position/orientation/half_extents or none")
            self.bind_inputs(position, orientation, half_extents)

        n = E * self._B
        dev = self._device
        self._contact = DiscContact(
            hit=wp.zeros(n, dtype=wp.int32, device=dev),
            disc=wp.zeros(n, dtype=wp.int32, device=dev),
            depth=wp.zeros(n, dtype=wp.float32, device=dev),
            nearest=wp.zeros(n, dtype=wp.vec2f, device=dev),
        )

    def _validate_inputs(self, position, orientation, half_extents) -> None:
        n = (self._E * self._B,)
        _check_arr("position", position, n, wp.vec3f, self._device)
        _check_arr("orientation", orientation, n, wp.quatf, self._device)
        _check_arr("half_extents", half_extents, n, wp.vec2f, self._device)

    def bind_inputs(self, position: wp.array, orientation: wp.array,
                    half_extents: wp.array) -> None:
        """Bind (or rebind) stable per-step input buffers (validated once).

        After binding, ``query()`` takes no arguments and reads these arrays
        in place; same-``.ptr`` rule applies under CUDA-graph capture.
        """
        self._validate_inputs(position, orientation, half_extents)
        self._bound = (position, orientation, half_extents)

    def query(self, position: "wp.array | None" = None,
              orientation: "wp.array | None" = None,
              half_extents: "wp.array | None" = None) -> DiscContact:
        """Box-vs-disc contact for ``E * max_boxes`` boxes.

        Bound mode (inputs bound at construction): call with no arguments.
        Per-call mode: pass all three arrays; under CUDA-graph capture the
        SAME arrays must be used at capture and every replay.
        """
        if self._bound is not None:
            if position is not None or orientation is not None \
                    or half_extents is not None:
                raise ValueError(
                    "checker inputs are bound; call query() with no arguments")
            position, orientation, half_extents = self._bound
        else:
            if position is None or orientation is None or half_extents is None:
                raise ValueError(
                    "checker is not bound; pass position, orientation and "
                    "half_extents to query()")
            self._validate_inputs(position, orientation, half_extents)
        c = self._contact
        wp.launch(
            _box_query_discs_k, dim=self._E * self._B,
            inputs=[self._discs, self._count, self._D, self._radius, self._B,
                    position, orientation, half_extents,
                    c.hit, c.disc, c.depth, c.nearest],
            device=self._device,
        )
        _sync(self._device)
        return c
