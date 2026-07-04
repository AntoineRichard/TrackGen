"""Box-vs-disc obstacle collision (gate posts, physical cones, point props).

``DiscChecker`` binds a flat ``[E * D]`` vec2f array of disc centers (ALIASED
— regenerated buffers are seen automatically) and a scalar radius, and
queries batches of oriented boxes exactly like ``CollisionChecker``: hit iff
the distance from the disc center to the solid OBB is <= radius. The deepest
penetrating disc is reported per box.

Disc validity: pass ``count`` (``[E]`` int32 real discs per env) OR rely on
NaN-marked padding (slots with NaN centers are skipped) — GateSequence
``left``/``right`` arrays interleaved as posts work out of the box.

Per-step inputs can be bound at construction (``position=``, ``yaw=``,
``half_extents=`` — all three or none): ``query()`` then takes no arguments
and reads the stable buffers in place (the CUDA-graph pattern).
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .collision_geom import (
    _is_nan2,
    _point_to_local_box_dist,
    _rot2,
    _safe_normalize2,
)

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
        the box (ties broken by lowest disc index).
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
    discs: wp.array(dtype=wp.vec2f),
    disc_count: wp.array(dtype=wp.int32),
    d_max: int,
    radius: float,
    max_boxes: int,
    position: wp.array(dtype=wp.vec2f),
    yaw: wp.array(dtype=wp.float32),
    half_extents: wp.array(dtype=wp.vec2f),
    out_hit: wp.array(dtype=wp.int32),
    out_disc: wp.array(dtype=wp.int32),
    out_depth: wp.array(dtype=wp.float32),
    out_nearest: wp.array(dtype=wp.vec2f),
):
    t = wp.tid()
    e = t // max_boxes
    nan2 = wp.vec2f(wp.nan, wp.nan)

    pos = position[t]
    if _is_nan2(pos) == 1:
        out_hit[t] = 0
        out_disc[t] = -1
        out_depth[t] = 0.0
        out_nearest[t] = nan2
        return

    yw = yaw[t]
    he = half_extents[t]
    nd = disc_count[e]
    if nd > d_max:
        nd = d_max
    base = e * d_max

    best = int(-1)
    best_pen = float(0.0)
    for i in range(nd):
        c = discs[base + i]
        if _is_nan2(c) == 0:
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

    c = discs[base + best]
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
                 yaw: "wp.array | None" = None,
                 half_extents: "wp.array | None" = None) -> None:
        _init()
        if not (float(radius) > 0.0):
            raise ValueError(f"radius must be > 0, got {radius!r}")
        if int(max_boxes) < 1:
            raise ValueError(f"max_boxes must be >= 1, got {max_boxes!r}")
        if count is not None:
            E = int(count.shape[0])
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
        if position is not None or yaw is not None or half_extents is not None:
            if position is None or yaw is None or half_extents is None:
                raise ValueError(
                    "bind all of position/yaw/half_extents or none")
            self.bind_inputs(position, yaw, half_extents)

        n = E * self._B
        dev = self._device
        self._contact = DiscContact(
            hit=wp.zeros(n, dtype=wp.int32, device=dev),
            disc=wp.zeros(n, dtype=wp.int32, device=dev),
            depth=wp.zeros(n, dtype=wp.float32, device=dev),
            nearest=wp.zeros(n, dtype=wp.vec2f, device=dev),
        )

    def _validate_inputs(self, position, yaw, half_extents) -> None:
        n = self._E * self._B
        for name, arr, dtype in (("position", position, wp.vec2f),
                                 ("yaw", yaw, wp.float32),
                                 ("half_extents", half_extents, wp.vec2f)):
            if not isinstance(arr, wp.array):
                raise ValueError(f"{name} must be a wp.array, got {type(arr)!r}")
            if arr.shape != (n,):
                raise ValueError(
                    f"{name} must have shape ({n},) = (E*max_boxes,), got {arr.shape}")
            if arr.dtype is not dtype:
                raise ValueError(
                    f"{name} must have dtype {dtype.__name__}, got {arr.dtype.__name__}")
            if str(arr.device) != self._device:
                raise ValueError(
                    f"{name} is on device {arr.device}, checker is on {self._device}")

    def bind_inputs(self, position: wp.array, yaw: wp.array,
                    half_extents: wp.array) -> None:
        """Bind (or rebind) stable per-step input buffers (validated once).

        After binding, ``query()`` takes no arguments and reads these arrays
        in place; same-``.ptr`` rule applies under CUDA-graph capture.
        """
        self._validate_inputs(position, yaw, half_extents)
        self._bound = (position, yaw, half_extents)

    def query(self, position: "wp.array | None" = None,
              yaw: "wp.array | None" = None,
              half_extents: "wp.array | None" = None) -> DiscContact:
        """Box-vs-disc contact for ``E * max_boxes`` boxes.

        Bound mode (inputs bound at construction): call with no arguments.
        Per-call mode: pass all three arrays; under CUDA-graph capture the
        SAME arrays must be used at capture and every replay.
        """
        if self._bound is not None:
            if position is not None or yaw is not None or half_extents is not None:
                raise ValueError(
                    "checker inputs are bound; call query() with no arguments")
            position, yaw, half_extents = self._bound
        else:
            if position is None or yaw is None or half_extents is None:
                raise ValueError(
                    "checker is not bound; pass position, yaw and "
                    "half_extents to query()")
            self._validate_inputs(position, yaw, half_extents)
        c = self._contact
        wp.launch(
            _box_query_discs_k, dim=self._E * self._B,
            inputs=[self._discs, self._count, self._D, self._radius, self._B,
                    position, yaw, half_extents,
                    c.hit, c.disc, c.depth, c.nearest],
            device=self._device,
        )
        _sync(self._device)
        return c
