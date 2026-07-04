"""Batched box-vs-track out-of-bounds queries (segments backend + facade).

``CollisionChecker`` binds to a :class:`Track` batch and answers, per oriented
box, whether the box has left the drivable band (inside the outer loop AND
outside the inner loop), with full contact info. The default ``segments``
backend is exact: one Warp thread per box scans the env's boundary segments.
The ``sdf`` backend (``collision_sdf.py``) trades exactness near boundaries
for O(1) queries against baked per-env signed-distance grids.

Layout follows the package conventions: flat ``[E * max_boxes]`` wp.arrays,
NaN for inactive slots, in-place reuse of the output ``BoxContact`` across
``query()`` calls (use ``clone()`` for snapshots), and no host syncs while
capturing is enabled (``track_gen.set_capturing``) so ``query()``/``bake()``
are CUDA-graph capturable.
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .collision_geom import (
    _box_corner,
    _closest_on_seg,
    _crossing,
    _is_nan2,
    _pick4,
    _point_to_local_box_dist,
    _rot2,
    _safe_normalize2,
    _seg_hits_aabb,
)
from .runtime import _BIG, _check_arr, _init, _sync
from .types import Track


@dataclass
class BoxContact:
    """Batched box-vs-track contact result, flat ``[E * max_boxes]`` per field.

    .. warning::

        ``CollisionChecker.query()`` returns the SAME ``BoxContact`` instance on
        every call and overwrites its buffers in place. Call ``clone()`` for a
        fully-owned snapshot.

    Attributes
    ----------
    oob : wp.array
        ``int32`` — 1 if the box crosses a boundary or lies outside the band.
    distance : wp.array
        ``float32`` signed clearance: positive = margin from the box to the
        nearest boundary; negative = deepest corner penetration when OOB
        (0.0 when the box only edge-crosses a boundary with all corners
        inside). NaN for inactive slots. Cf. :class:`DiscContact.depth
        <track_gen.collision.DiscContact>`, which reports an UNSIGNED
        penetration depth (0.0 = no hit) rather than a signed clearance.
    nearest : wp.array
        ``vec2f`` nearest point on the boundary polylines to the box.
    normal : wp.array
        ``vec2f`` boundary normal at ``nearest``, pointing INTO the band.
    boundary : wp.array
        ``int32`` — 0 = inner boundary, 1 = outer boundary, -1 = inactive slot
        or degenerate track (count < 3).
    """

    oob: wp.array
    distance: wp.array
    nearest: wp.array
    normal: wp.array
    boundary: wp.array

    def clone(self) -> "BoxContact":
        """Return a deep copy whose Warp buffers do not alias this result."""
        return BoxContact(
            oob=wp.clone(self.oob),
            distance=wp.clone(self.distance),
            nearest=wp.clone(self.nearest),
            normal=wp.clone(self.normal),
            boundary=wp.clone(self.boundary),
        )


@wp.kernel
def _box_query_segments_k(
    inner: wp.array(dtype=wp.vec2f),
    outer: wp.array(dtype=wp.vec2f),
    center: wp.array(dtype=wp.vec2f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    max_boxes: int,
    position: wp.array(dtype=wp.vec2f),
    yaw: wp.array(dtype=wp.float32),
    half_extents: wp.array(dtype=wp.vec2f),
    out_oob: wp.array(dtype=wp.int32),
    out_distance: wp.array(dtype=wp.float32),
    out_nearest: wp.array(dtype=wp.vec2f),
    out_normal: wp.array(dtype=wp.vec2f),
    out_boundary: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    e = t // max_boxes
    nan2 = wp.vec2f(wp.nan, wp.nan)

    pos = position[t]
    if _is_nan2(pos) == 1:
        out_oob[t] = 0
        out_distance[t] = wp.nan
        out_nearest[t] = nan2
        out_normal[t] = nan2
        out_boundary[t] = -1
        return

    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        # Degenerate/invalid track: conservative OOB, NaN geometry.
        out_oob[t] = 1
        out_distance[t] = wp.nan
        out_nearest[t] = nan2
        out_normal[t] = nan2
        out_boundary[t] = -1
        return

    yw = yaw[t]
    he = half_extents[t]
    ux = _rot2(yw, wp.vec2f(1.0, 0.0))
    uy = _rot2(yw, wp.vec2f(0.0, 1.0))
    c0 = _box_corner(pos, ux, uy, he, 0)
    c1 = _box_corner(pos, ux, uy, he, 1)
    c2 = _box_corner(pos, ux, uy, he, 2)
    c3 = _box_corner(pos, ux, uy, he, 3)

    base = e * n_max

    # Per-corner crossing counts and min distances to each boundary polyline.
    cn_in = wp.vec4f(0.0, 0.0, 0.0, 0.0)
    cn_out = wp.vec4f(0.0, 0.0, 0.0, 0.0)
    dc_in = wp.vec4f(_BIG, _BIG, _BIG, _BIG)
    dc_out = wp.vec4f(_BIG, _BIG, _BIG, _BIG)

    crossed = int(0)
    best_d = _BIG
    best_pt = wp.vec2f(0.0, 0.0)
    best_bnd = int(0)
    best_i = int(0)

    for j in range(2 * m):
        bnd = int(0)
        i = j
        if j >= m:
            bnd = int(1)
            i = j - m
        i2 = i + 1
        if i2 == m:
            i2 = 0
        a = wp.vec2f(0.0, 0.0)
        b = wp.vec2f(0.0, 0.0)
        if bnd == 0:
            a = inner[base + i]
            b = inner[base + i2]
        else:
            a = outer[base + i]
            b = outer[base + i2]

        # Box<->segment distance candidates: box corners vs the segment ...
        cand_d = _BIG
        cand_pt = a
        for k in range(4):
            ck = _pick4(c0, c1, c2, c3, k)
            cr = float(_crossing(ck, a, b))
            cp = _closest_on_seg(ck, a, b)
            dk = wp.length(ck - cp)
            if bnd == 0:
                cn_in[k] = cn_in[k] + cr
                dc_in[k] = wp.min(dc_in[k], dk)
            else:
                cn_out[k] = cn_out[k] + cr
                dc_out[k] = wp.min(dc_out[k], dk)
            if dk < cand_d:
                cand_d = dk
                cand_pt = cp

        # ... plus the segment start vertex vs the solid box (the end vertex is
        # the next segment's start, so the closed loop covers every vertex).
        al = _rot2(-yw, a - pos)
        bl = _rot2(-yw, b - pos)
        d_end = _point_to_local_box_dist(al, he)
        if d_end < cand_d:
            cand_d = d_end
            cand_pt = a
        if _seg_hits_aabb(al, bl, he) == 1:
            crossed = int(1)
            cand_d = 0.0
            cand_pt = _closest_on_seg(pos, a, b)

        if cand_d < best_d:
            best_d = cand_d
            best_pt = cand_pt
            best_bnd = bnd
            best_i = i

    inside = int(1)
    worst_pen = 0.0
    for k in range(4):
        pen_k = 0.0
        if int(cn_in[k]) % 2 == 1:   # corner inside the inner hole
            inside = int(0)
            pen_k = dc_in[k]
        if int(cn_out[k]) % 2 == 0:  # corner outside the outer loop
            inside = int(0)
            pen_k = wp.max(pen_k, dc_out[k])
        worst_pen = wp.max(worst_pen, pen_k)

    oob = int(0)
    if inside == 0 or crossed == 1:
        oob = int(1)

    dist = best_d
    if oob == 1:
        dist = -worst_pen

    # Normal of the argmin segment, oriented into the band via the
    # index-aligned centerline point.
    i2 = best_i + 1
    if i2 == m:
        i2 = 0
    sa = wp.vec2f(0.0, 0.0)
    sb = wp.vec2f(0.0, 0.0)
    if best_bnd == 0:
        sa = inner[base + best_i]
        sb = inner[base + i2]
    else:
        sa = outer[base + best_i]
        sb = outer[base + i2]
    seg = sb - sa
    nrm = wp.vec2f(-seg[1], seg[0])
    cpt = center[base + best_i]
    if wp.length(nrm) < 1.0e-8:
        nrm = cpt - best_pt
    nrm = _safe_normalize2(nrm)
    if wp.dot(cpt - best_pt, nrm) < 0.0:
        nrm = -nrm

    out_oob[t] = oob
    out_distance[t] = dist
    out_nearest[t] = best_pt
    out_normal[t] = nrm
    out_boundary[t] = best_bnd


class CollisionChecker:
    """Batched box-vs-track out-of-bounds checker bound to a :class:`Track`.

    Because ``TrackGenerator.generate()`` overwrites its ``Track`` buffers in
    place, the ``segments`` backend always reads the CURRENT track batch with
    no rebind step. The ``sdf`` backend requires ``bake()`` after each
    ``generate()`` call.

    ``query()`` is allocation-free and, under graph capture (enabled via
    ``track_gen.set_capturing``), host-sync-free (CUDA-graph capturable). In
    normal (non-capturing) use a ``wp.synchronize()`` follows the launch,
    per the codebase idiom. ``query()`` returns the same preallocated
    :class:`BoxContact` on every call.
    """

    def __init__(self, track: Track, max_boxes: int, method: str = "segments",
                 sdf_resolution: int = 128, sdf_padding: "float | None" = None,
                 position: "wp.array | None" = None,
                 yaw: "wp.array | None" = None,
                 half_extents: "wp.array | None" = None) -> None:
        """Bind to a :class:`Track` batch and allocate the result buffers.

        Args:
            track: the bound track batch; the ``segments`` backend reads its
                buffers directly on every ``query()`` (no rebind needed after
                ``generate()``).
            max_boxes: query stride (boxes per env); must be >= 1.
            method: ``"segments"`` (exact, default) or ``"sdf"`` (baked,
                approximate near boundaries; see the module docstring).
            sdf_resolution: ``method="sdf"`` only — grid side length in
                texels per env (>= 8; default 128). Ignored for ``"segments"``.
            sdf_padding: ``method="sdf"`` only — per-side AABB padding in
                world units the grid is expanded by. ``None`` (default)
                selects AUTO padding: 10% of the larger AABB extent
                (``max(width, height)``) of each env's track, computed at
                bake time. Pass a positive float to override with a fixed
                padding for every env.
            position, yaw, half_extents: optional constructor binding
                (all-or-none); equivalent to calling :meth:`bind_inputs`
                right after construction.
        """
        _init()
        if int(max_boxes) < 1:
            raise ValueError(f"max_boxes must be >= 1, got {max_boxes!r}")
        if method not in ("segments", "sdf"):
            raise ValueError(
                f"method must be one of {{'segments', 'sdf'}}, got {method!r}")
        if int(sdf_resolution) < 8:
            raise ValueError(
                f"sdf_resolution must be >= 8, got {sdf_resolution!r}")
        if sdf_padding is not None and float(sdf_padding) <= 0.0:
            raise ValueError(
                f"sdf_padding must be > 0 (or None for auto), got {sdf_padding!r}")
        E = int(track.count.shape[0])
        stride = int(track.outer.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"track batch layout invalid: outer has {stride} slots for {E} envs")
        self._track = track
        self._method = method
        self._E = E
        self._n_max = stride // E
        self._B = int(max_boxes)
        self._device = str(track.outer.device)

        n = E * self._B
        dev = self._device
        self._contact = BoxContact(
            oob=wp.zeros(n, dtype=wp.int32, device=dev),
            distance=wp.zeros(n, dtype=wp.float32, device=dev),
            nearest=wp.zeros(n, dtype=wp.vec2f, device=dev),
            normal=wp.zeros(n, dtype=wp.vec2f, device=dev),
            boundary=wp.zeros(n, dtype=wp.int32, device=dev),
        )
        self._bound: "tuple | None" = None
        if position is not None or yaw is not None or half_extents is not None:
            if position is None or yaw is None or half_extents is None:
                raise ValueError(
                    "bind all of position/yaw/half_extents or none")
            self.bind_inputs(position, yaw, half_extents)
        if method == "sdf":
            R = int(sdf_resolution)
            self._sdf_resolution = R
            self._sdf_padding = -1.0 if sdf_padding is None else float(sdf_padding)
            self._sdf_lo = wp.zeros(E, dtype=wp.vec2f, device=dev)
            self._sdf_hi = wp.zeros(E, dtype=wp.vec2f, device=dev)
            self._sdf_phi = wp.zeros(E * R * R, dtype=wp.float32, device=dev)
            self._sdf_bid = wp.zeros(E * R * R, dtype=wp.int8, device=dev)
            self.bake()

    def bake(self) -> None:
        """(Re)bake the per-env SDF grids from the bound Track.

        Required after every ``TrackGenerator.generate()`` call when
        ``method="sdf"`` (the segments backend needs no rebake). Pure kernel
        launches — CUDA-graph capturable.

        Raises:
            ValueError: if this checker was constructed with ``method="segments"``.
        """
        if self._method != "sdf":
            raise ValueError("bake() is only valid for method='sdf' checkers")
        from . import collision_sdf
        t = self._track
        E = self._E
        R = self._sdf_resolution
        wp.launch(collision_sdf._track_aabb_k, dim=E,
                  inputs=[t.outer, t.count, self._n_max,
                          self._sdf_padding, 0.1, self._sdf_lo, self._sdf_hi],
                  device=self._device)
        wp.launch(collision_sdf._sdf_bake_k, dim=E * R * R,
                  inputs=[t.inner, t.outer, t.count, self._n_max, R,
                          self._sdf_lo, self._sdf_hi, self._sdf_phi, self._sdf_bid],
                  device=self._device)
        _sync(self._device)

    def _validate_inputs(self, position, yaw, half_extents) -> None:
        n = (self._E * self._B,)
        _check_arr("position", position, n, wp.vec2f, self._device)
        _check_arr("yaw", yaw, n, wp.float32, self._device)
        _check_arr("half_extents", half_extents, n, wp.vec2f, self._device)

    def bind_inputs(self, position: wp.array, yaw: wp.array,
                    half_extents: wp.array) -> None:
        """Bind stable per-step input buffers (validated once, here).

        After binding, ``query()`` takes no arguments and reads these arrays
        in place each call — the natural CUDA-graph pattern: the sim writes
        poses into its stable buffers and replays the captured query. The
        arrays must keep the same ``.ptr`` for the binding's lifetime.
        """
        self._validate_inputs(position, yaw, half_extents)
        self._bound = (position, yaw, half_extents)

    def query(self, position: "wp.array | None" = None,
              yaw: "wp.array | None" = None,
              half_extents: "wp.array | None" = None) -> BoxContact:
        """Compute contact info for ``E * max_boxes`` oriented boxes.

        Two modes:

        - Bound mode: after ``bind_inputs()``, call ``query()`` with no
          arguments; it reads the bound arrays in place each call (the
          arrays must keep the same ``.ptr`` for the binding's lifetime —
          write new poses into them rather than rebinding, e.g. under
          CUDA-graph capture).
        - Per-call mode: if not bound, pass ``position``, ``yaw`` and
          ``half_extents`` explicitly on every call.

        Args:
            position: ``[E*max_boxes]`` ``vec2f`` box centers. NaN marks an
                inactive slot (NaN outputs, ``oob=0``, ``boundary=-1``).
                Required in per-call mode; omit in bound mode.
            yaw: ``[E*max_boxes]`` ``float32`` box orientations (radians).
                Required in per-call mode; omit in bound mode.
            half_extents: ``[E*max_boxes]`` ``vec2f`` per-box half sizes.
                Required in per-call mode; omit in bound mode.

        Returns:
            The checker's preallocated :class:`BoxContact` (same instance every
            call; buffers overwritten in place).

        Raises:
            ValueError: on shape/dtype/device mismatch, or on mode misuse
                (passing arguments while bound, or omitting them while not
                bound).
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
                    "half_extents to query() or call bind_inputs() first")
            self._validate_inputs(position, yaw, half_extents)
        n = self._E * self._B
        t = self._track
        c = self._contact
        if self._method == "segments":
            wp.launch(
                _box_query_segments_k, dim=n,
                inputs=[t.inner, t.outer, t.center, t.count, self._n_max, self._B,
                        position, yaw, half_extents,
                        c.oob, c.distance, c.nearest, c.normal, c.boundary],
                device=self._device,
            )
        else:
            from . import collision_sdf
            wp.launch(
                collision_sdf._box_query_sdf_k, dim=n,
                inputs=[self._sdf_lo, self._sdf_hi, self._sdf_phi, self._sdf_bid,
                        self._sdf_resolution, self._B,
                        position, yaw, half_extents,
                        c.oob, c.distance, c.nearest, c.normal, c.boundary],
                device=self._device,
            )
        _sync(self._device)
        return c
