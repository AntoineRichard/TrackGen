"""Resampled 3D centerline through a gate sequence (closed Catmull-Rom).

``CourseLine`` gives gates mode a Track-shaped centerline so the standard
Track consumers (``TrackLocalizer``; later the corridor tube) work unchanged:
``center``/``tangent``/``arclen``/``length``/``count``/``valid`` are real,
``outer``/``inner``/``normal`` are NaN (no road band), ``winding`` is 0.

count[e] = samples_per_gate * gate_count[e]; N_max = samples_per_gate *
max_gates. ``refresh()`` is two kernel launches, allocation-free and
capture-safe; call it after every gate regeneration (the facade does).
"""
import warp as wp

from .runtime import _init, _sync
from .types import GateSequence, Track


@wp.func
def _cr_point(p0: wp.vec3f, p1: wp.vec3f, p2: wp.vec3f, p3: wp.vec3f,
              t: float) -> wp.vec3f:
    # uniform Catmull-Rom (tangent at knot i = 0.5*(p_{i+1} - p_{i-1}),
    # matching the gate pipeline's central-difference gate tangents)
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2.0 * p1) + (p2 - p0) * t +
                  (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                  (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)


@wp.kernel
def _sample_line_k(
    gate_pos: wp.array(dtype=wp.vec3f),
    gate_count: wp.array(dtype=wp.int32),
    gate_valid: wp.array(dtype=wp.int32),
    max_gates: int,
    spg: int,
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    n_max = max_gates * spg
    e = tid // n_max
    j = tid - e * n_max
    n = gate_count[e]
    if n > max_gates:
        n = max_gates
    m = n * spg
    degenerate = n < 3 or gate_valid[e] == 0
    if j == 0:
        if degenerate:
            count[e] = 0
            valid[e] = 0
        else:
            count[e] = m
            valid[e] = gate_valid[e]
    if j >= m or degenerate:
        center[tid] = wp.vec3f(wp.nan, wp.nan, wp.nan)
        return
    gbase = e * max_gates
    i = j // spg
    t = float(j - i * spg) / float(spg)
    i0 = ((i - 1) % n + n) % n
    i2 = (i + 1) % n
    i3 = (i + 2) % n
    center[tid] = _cr_point(gate_pos[gbase + i0], gate_pos[gbase + i],
                            gate_pos[gbase + i2], gate_pos[gbase + i3], t)


@wp.kernel
def _line_frames_k(
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    tangent: wp.array(dtype=wp.vec3f),
    arclen: wp.array(dtype=wp.float32),
    length: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    base = e * n_max
    m = count[e]
    if m > n_max:
        m = n_max
    if m < 3:
        length[e] = 0.0
        for i in range(n_max):
            tangent[base + i] = wp.vec3f(wp.nan, wp.nan, wp.nan)
            arclen[base + i] = wp.nan
        return
    acc = float(0.0)
    for i in range(n_max):
        if i < m:
            if i > 0:
                acc = acc + wp.length(center[base + i] - center[base + i - 1])
            arclen[base + i] = acc
            prev = ((i - 1) % m + m) % m
            nxt = (i + 1) % m
            d = center[base + nxt] - center[base + prev]
            l = wp.length(d)
            if l > 1.0e-12:
                tangent[base + i] = d / l
            else:
                tangent[base + i] = wp.vec3f(0.0, 0.0, 0.0)
        else:
            tangent[base + i] = wp.vec3f(wp.nan, wp.nan, wp.nan)
            arclen[base + i] = wp.nan
    length[e] = acc + wp.length(center[base] - center[base + m - 1])


class CourseLine:
    """See module docstring."""

    def __init__(self, seq: GateSequence, samples_per_gate: int = 8) -> None:
        _init()
        if int(samples_per_gate) < 2:
            raise ValueError(
                f"samples_per_gate must be >= 2, got {samples_per_gate!r}")
        E = int(seq.count.shape[0])
        stride = int(seq.position.shape[0])
        if E < 1 or stride % E != 0:
            raise ValueError(
                f"gate batch layout invalid: {stride} slots for {E} envs")
        self._seq = seq
        self._E = E
        self._G = stride // E
        self._spg = int(samples_per_gate)
        self._n_max = self._G * self._spg
        self._device = str(seq.position.device)
        dev = self._device
        n = E * self._n_max
        nan3 = float("nan")
        self.track = Track(
            outer=wp.full(n, wp.vec3f(nan3, nan3, nan3), device=dev),
            center=wp.zeros(n, dtype=wp.vec3f, device=dev),
            inner=wp.full(n, wp.vec3f(nan3, nan3, nan3), device=dev),
            tangent=wp.zeros(n, dtype=wp.vec3f, device=dev),
            normal=wp.full(n, wp.vec3f(nan3, nan3, nan3), device=dev),
            arclen=wp.zeros(n, dtype=wp.float32, device=dev),
            length=wp.zeros(E, dtype=wp.float32, device=dev),
            valid=wp.zeros(E, dtype=wp.int32, device=dev),
            count=wp.zeros(E, dtype=wp.int32, device=dev),
            winding=wp.zeros(E, dtype=wp.float32, device=dev),
        )

    def refresh(self) -> None:
        """Resample from the CURRENT gate batch (capture-safe)."""
        s, t = self._seq, self.track
        wp.launch(_sample_line_k, dim=self._E * self._n_max,
                  inputs=[s.position, s.count, s.valid, self._G, self._spg,
                          t.center, t.count, t.valid],
                  device=self._device)
        wp.launch(_line_frames_k, dim=self._E,
                  inputs=[t.center, t.count, self._n_max,
                          t.tangent, t.arclen, t.length],
                  device=self._device)
        _sync(self._device)
