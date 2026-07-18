"""Per-env road heightfield bake for external physics solvers.

Nearest-edge continuation: every texel takes the road z of the NEAREST
centerline cross-section by plan-view distance — on-road texels get the
road surface, off-road texels continue the nearest edge's height outward.
The surface is continuous (no cliffs at the road edge); flat tracks bake a
constant z_base sheet; invalid/degenerate envs bake NaN.

Grid layout mirrors collision_sdf: square [E * res * res] float32, row-major
(y outer, x inner), texel centers, per-env AABB from the plan-view band +
padding (AUTO = 10% of the larger extent). Bake is a brute-force
O(E * res^2 * N) scan like the SDF bake; ``bake()`` is allocation-free and
capture-safe and refreshes the SAME HeightField in place.
"""
from dataclasses import dataclass

import warp as wp

from .collision_sdf import _track_aabb_k
from .runtime import _BIG, _init, _sync
from .types import Track


@dataclass
class HeightField:
    """Baked per-env height grid; overwritten in place by ``bake()``.

    height : wp.array
        ``[E * res * res]`` float32 road-surface heights (NaN for invalid
        envs). Reshape via ``.numpy().reshape(E, res, res)`` (row-major:
        ``[e, y, x]``).
    lo, hi : wp.array
        ``[E]`` vec2f world AABB corners; texel (x, y) center is at
        ``lo + ((x, y) + 0.5) / res * (hi - lo)``.
    res : int
        Grid resolution per side.
    """

    height: wp.array
    lo: wp.array
    hi: wp.array
    res: int

    def clone(self) -> "HeightField":
        return HeightField(height=wp.clone(self.height), lo=wp.clone(self.lo),
                           hi=wp.clone(self.hi), res=self.res)


@wp.kernel
def _height_bake_k(
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    n_max: int,
    res: int,
    lo: wp.array(dtype=wp.vec2f),
    hi: wp.array(dtype=wp.vec2f),
    height: wp.array(dtype=wp.float32),
):
    t = wp.tid()
    cells = res * res
    e = t // cells
    rem = t - e * cells
    gy = rem // res
    gx = rem - gy * res

    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3 or valid[e] == 0:
        height[t] = wp.nan
        return

    l = lo[e]
    h = hi[e]
    p = wp.vec2f(
        l[0] + (float(gx) + 0.5) / float(res) * (h[0] - l[0]),
        l[1] + (float(gy) + 0.5) / float(res) * (h[1] - l[1]),
    )

    base = e * n_max
    best_d2 = float(_BIG)
    best_z = float(0.0)
    for i in range(m):
        j = i + 1
        if j == m:
            j = 0
        a3 = center[base + i]
        b3 = center[base + j]
        a = wp.vec2f(a3[0], a3[1])
        ab = wp.vec2f(b3[0] - a3[0], b3[1] - a3[1])
        denom = wp.max(wp.dot(ab, ab), 1.0e-12)
        u = wp.clamp(wp.dot(p - a, ab) / denom, 0.0, 1.0)
        q = a + ab * u
        d2 = wp.dot(p - q, p - q)
        if d2 < best_d2:
            best_d2 = d2
            best_z = a3[2] + (b3[2] - a3[2]) * u
    height[t] = best_z


class HeightFieldBaker:
    """Bake the bound track batch into a per-env heightfield.

    Lifecycle mirrors the SDF baker: construct once (allocates), ``bake()``
    after every regeneration (kernel launches only; the facade does this in
    its captured refresh when ``CourseConfig.heightfield_resolution`` is
    set). Results are undefined for envs with ``valid[e] == 0`` (baked NaN).
    """

    def __init__(self, track: Track, resolution: int,
                 padding: "float | None" = None) -> None:
        _init()
        if int(resolution) < 8:
            raise ValueError(f"resolution must be >= 8, got {resolution!r}")
        if padding is not None and not (float(padding) > 0.0):
            raise ValueError(f"padding must be > 0 (or None for auto), got "
                             f"{padding!r}")
        E = int(track.count.shape[0])
        stride = int(track.center.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"track batch layout invalid: {stride} slots for {E} envs")
        self._track = track
        self._E = E
        self._n_max = stride // E
        self._res = int(resolution)
        self._pad = 0.0 if padding is None else float(padding)
        self._pad_frac = 0.1
        self._device = str(track.center.device)
        dev = self._device
        self._hf = HeightField(
            height=wp.zeros(E * self._res * self._res, dtype=wp.float32,
                            device=dev),
            lo=wp.zeros(E, dtype=wp.vec2f, device=dev),
            hi=wp.zeros(E, dtype=wp.vec2f, device=dev),
            res=self._res,
        )

    def bake(self) -> HeightField:
        """Re-bake from the CURRENT track batch; returns the same instance."""
        t = self._track
        f = self._hf
        wp.launch(_track_aabb_k, dim=self._E,
                  inputs=[t.outer, t.count, self._n_max, self._pad,
                          self._pad_frac, f.lo, f.hi],
                  device=self._device)
        wp.launch(_height_bake_k, dim=self._E * self._res * self._res,
                  inputs=[t.center, t.count, t.valid, self._n_max, self._res,
                          f.lo, f.hi, f.height],
                  device=self._device)
        _sync(self._device)
        return f
