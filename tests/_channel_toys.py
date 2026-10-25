"""Toy plugins for per-point channels, the width registry and 3D station spacing (roadmap
PR13a), written against the public plugin surface only (``track_gen.plugin``,
``track_gen.ops``), and the fake ``acme-channels`` distribution that registers them.

Generators:

- ``acme.ribbon``: a smooth closed loop (an ellipse of semi-axes ``scale`` and ``0.8 *
  scale``, a salted per-env phase) that writes the carried families
  ``raw.acme.theta`` (f32, ``angle``: the parametric angle wrapped into ``[-pi, pi)``),
  ``raw.acme.surface_id`` (int32, ``nearest``: the quadrant of ``theta``),
  ``raw.acme.lane`` (int32, ``previous``: ``4 * i // N``), ``raw.acme.grip`` (f32,
  ``linear``: ``cos(theta)``) and ``raw.acme.xy_copy`` (vec2f, ``linear``: ``raw.xy``
  itself), all exported, plus ``raw.acme.hw`` (f32, ``linear``, not exported:
  ``hw_mean[e] + hw_amp[e] * sin(waves * theta)``, both dynamic). The dynamic flag
  ``invalid`` (per env, 0 / 1) writes ``gen.valid = 0``.
- ``acme.ribbon_plain``: the same ``raw.xy`` (and ``gen.valid``), no family.
- ``acme.helix_loop`` (``provides={"z"}``): a circle of radius ``scale`` in plan view,
  ``raw.z = height * sin(waves * theta)``.
- ``acme.pinch`` (``provides={"half_width"}``): a circle, ``raw.half_width = hw_mean[e]
  + hw_amp[e] * sin(waves * theta)``.
- ``acme.ribbon_metres``: ribbon's loop with one ``linear`` exported attribute in metres
  (``raw.acme.station_m = 100 + 50 * cos(theta)``, |v| ~ 100).
- ``acme.ribbon_partial``: a broken ribbon that leaves the second half of
  ``raw.acme.grip`` unwritten (``attributes.written`` must fail).

Width stages:

- ``acme.wave``: ``cl.half_width[i] = hw_mean[e] + hw_amp[e] * sin(2 pi waves i /
  count)`` (a writer; dynamic ``hw_mean`` / ``hw_amp``, static ``waves``);
- ``acme.wave_after``: the same values as a REWRITE (reads ``cl.half_width``);
- ``acme.rescale``: a rewrite, ``factor[e] * w`` (the constant declaration is lost: the
  consumers run the per-point kernels);
- ``acme.negative``: writes ``-1`` everywhere (``width.values`` must fail);
- ``acme.curv_width``: writes ``cl.half_width`` (constant 0.1 by kernel) and
  ``cl.acme.curv`` (per point) for the stale-read rule.

Gate side:

- ``acme.tagged`` (gate generator, ``orderings={"raw"}``): ``A`` anchors on a ring and
  ``anchors.acme.tag`` (int32, exported: the anchor index modulo 3);
- ``acme.gate_clearance`` (validity on gates): ``gates.acme.clearance`` (f32 per gate,
  exported: the distance to the next gate) and a fail bit where some real gate's
  clearance is below ``min_clearance[e]``;
- ``acme.st`` (stations plugin, no ``spacing_metric``): ``gates`` anchors at equal index
  steps of the raw loop.

Track checks:

- ``acme.curv_check`` (validity): reads ``cl.acme.curv``, ORs its bit where it exceeds
  ``limit``.
- ``acme.slope`` (elevation): writes ``cl.z`` (flat 0) and ``cl.acme.slope`` (per point,
  f32, exported), the plugin channel the gate pipeline rewires to ``gates.acme.slope``.

``acme_channels_entry_point()`` is the fake entry point that installs them.
"""
from __future__ import annotations

import dataclasses

import warp as wp

from tests._plugin_toys import FakeDist, FakeEntryPoint
from track_gen import plugin
from track_gen.ops import hash32

#: The plugin API version this module targets (read by the registry's loader).
track_gen_api = 1

ACME = FakeDist("acme-channels", "0.1.0")
RAW = 192                       # raw points per env of every toy generator
_TWO_PI = wp.constant(6.2831853)
_PI = wp.constant(3.14159265)


@wp.func
def _phase(seed: wp.int32, salt: wp.uint32) -> float:
    st = wp.rand_init(wp.int32(hash32(wp.uint32(seed) ^ salt) & wp.uint32(0x7FFFFFFF)))
    return _TWO_PI * wp.randf(st)


@wp.func
def _wrap(x: float) -> float:
    # Into [-pi, pi).
    return x - _TWO_PI * wp.floor((x + _PI) / _TWO_PI)


# ---------------------------------------------------------------------------------------
# acme.ribbon / acme.ribbon_plain
# ---------------------------------------------------------------------------------------

@wp.kernel
def _ribbon_xy_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, n: int, a: float,
                 b: float, invalid: wp.array(dtype=wp.float32),
                 xy: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    th = _TWO_PI * float(i) / float(n) + _phase(seed[e], salt)
    xy[t] = wp.vec2f(a * wp.cos(th), b * wp.sin(th))
    if i == 0:
        if invalid[e] != 0.0:
            valid[e] = 0
        else:
            valid[e] = 1


@wp.kernel
def _ribbon_attrs_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, n: int,
                    hw_mean: wp.array(dtype=wp.float32), hw_amp: wp.array(dtype=wp.float32),
                    waves: float, xy: wp.array(dtype=wp.vec2f),
                    theta: wp.array(dtype=wp.float32), surface: wp.array(dtype=wp.int32),
                    lane: wp.array(dtype=wp.int32), grip: wp.array(dtype=wp.float32),
                    hw: wp.array(dtype=wp.float32), xy_copy: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    th = _wrap(_TWO_PI * float(i) / float(n) + _phase(seed[e], salt))
    theta[t] = th
    q = int(wp.floor((th + _PI) / (0.5 * _PI)))
    surface[t] = wp.min(wp.max(q, 0), 3)
    lane[t] = (4 * i) // n
    grip[t] = wp.cos(th)
    hw[t] = hw_mean[e] + hw_amp[e] * wp.sin(waves * th)
    xy_copy[t] = xy[t]


@dataclasses.dataclass(frozen=True, kw_only=True)
class RibbonParams(plugin.Params):
    hw_mean: float = plugin.dynamic(0.1, lo=0.0, hi=1.0, doc="mean of raw.acme.hw")
    hw_amp: float = plugin.dynamic(0.03, lo=0.0, hi=2.0, doc="amplitude of raw.acme.hw")
    waves: int = plugin.static(3, lo=1, doc="raw.acme.hw waves per loop")
    invalid: float = plugin.dynamic(0.0, lo=0.0, hi=1.0, doc="1: gen.valid = 0")


class RibbonPlain(plugin.Stage):
    kind = "generator"
    name = "acme.ribbon_plain"
    output = "centerline"
    Params = RibbonParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}

    def capacity(self, p, layout):
        return {"raw": RAW}

    def _xy(self, b):
        xy, ok = b.write("xy", wp.vec2f, per="raw"), b.write("valid", wp.int32)
        salt = b.salt("phase")
        a, bb = 1.0 * b.scale, 0.8 * b.scale
        invalid = b.param("invalid")
        seed, n = b.seed, RAW

        def launch(ctx):
            ctx.launch(_ribbon_xy_k, b.E * n, [seed, salt, n, a, bb, invalid, xy, ok])
        return xy, salt, launch

    def build(self, b, p):
        _, _, launch = self._xy(b)
        return launch


class Ribbon(RibbonPlain):
    name = "acme.ribbon"
    writes = {"xy": "raw.xy", "valid": "gen.valid", "theta": "raw.acme.theta",
              "surface": "raw.acme.surface_id", "lane": "raw.acme.lane",
              "grip": "raw.acme.grip", "hw": "raw.acme.hw", "xy_copy": "raw.acme.xy_copy"}

    def build(self, b, p):
        xy, salt, launch_xy = self._xy(b)
        theta = b.write("theta", wp.float32, interp="angle", export=True)
        surface = b.write("surface", wp.int32, interp="nearest", export=True)
        lane = b.write("lane", wp.int32, interp="previous", export="acme.lane")
        grip = b.write("grip", wp.float32, interp="linear", export=True)
        hw = b.write("hw", wp.float32, interp="linear")
        xy_copy = b.write("xy_copy", wp.vec2f, interp="linear", export=True)
        hw_mean, hw_amp = b.param("hw_mean"), b.param("hw_amp")
        waves, seed, n = float(p.waves), b.seed, RAW

        def enqueue(ctx):
            launch_xy(ctx)
            ctx.launch(_ribbon_attrs_k, b.E * n,
                       [seed, salt, n, hw_mean, hw_amp, waves, xy, theta, surface, lane,
                        grip, hw, xy_copy])
        return enqueue


@wp.kernel
def _metres_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, n: int,
              out: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    th = _TWO_PI * float(i) / float(n) + _phase(seed[e], salt)
    out[t] = 100.0 + 50.0 * wp.cos(th)


class RibbonMetres(RibbonPlain):
    name = "acme.ribbon_metres"
    writes = {"xy": "raw.xy", "valid": "gen.valid", "m": "raw.acme.station_m"}

    def build(self, b, p):
        _, salt, launch_xy = self._xy(b)
        m = b.write("m", wp.float32, interp="linear", export=True)
        seed, n = b.seed, RAW

        def enqueue(ctx):
            launch_xy(ctx)
            ctx.launch(_metres_k, b.E * n, [seed, salt, n, m])
        return enqueue


@wp.kernel
def _half_fill_k(n: int, v: float, out: wp.array(dtype=wp.float32)):
    t = wp.tid()
    i = t - (t // n) * n
    if i < n // 2:
        out[t] = v


class RibbonPartial(RibbonPlain):
    name = "acme.ribbon_partial"
    writes = {"xy": "raw.xy", "valid": "gen.valid", "grip": "raw.acme.grip"}

    def build(self, b, p):
        _, _, launch_xy = self._xy(b)
        grip = b.write("grip", wp.float32, interp="linear", export=True)
        n = RAW

        def enqueue(ctx):
            launch_xy(ctx)
            ctx.launch(_half_fill_k, b.E * n, [n, 0.5, grip])   # the second half: never
        return enqueue


# ---------------------------------------------------------------------------------------
# acme.helix_loop (provides z) / acme.pinch (provides half_width)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _circle_attr_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, n: int, r: float,
                   mean: wp.array(dtype=wp.float32), amp: wp.array(dtype=wp.float32),
                   waves: float, xy: wp.array(dtype=wp.vec2f),
                   valid: wp.array(dtype=wp.int32), attr: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    th = _TWO_PI * float(i) / float(n)
    ph = _phase(seed[e], salt)
    xy[t] = wp.vec2f(r * wp.cos(th + ph), r * wp.sin(th + ph))
    attr[t] = mean[e] + amp[e] * wp.sin(waves * th)
    if i == 0:
        valid[e] = 1


@dataclasses.dataclass(frozen=True, kw_only=True)
class HelixParams(plugin.Params):
    height: float = plugin.dynamic(0.3, lo=0.0, hi=2.0, doc="altitude amplitude")
    base: float = plugin.dynamic(0.0, lo=-5.0, hi=5.0, doc="mean altitude")
    waves: int = plugin.static(3, lo=1, doc="climbs per loop")


class HelixLoop(plugin.Stage):
    kind = "generator"
    name = "acme.helix_loop"
    output = "centerline"
    Params = HelixParams
    provides = frozenset({"z"})
    writes = {"xy": "raw.xy", "valid": "gen.valid", "z": "raw.z"}

    def capacity(self, p, layout):
        return {"raw": RAW}

    def build(self, b, p):
        xy, ok = b.write("xy", wp.vec2f, per="raw"), b.write("valid", wp.int32)
        z = b.write("z")                      # the catalogue's float32, linear
        salt, seed, n = b.salt("phase"), b.seed, RAW
        height, base, waves = b.param("height"), b.param("base"), float(p.waves)
        r = 1.0 * b.scale

        def enqueue(ctx):
            ctx.launch(_circle_attr_k, b.E * n,
                       [seed, salt, n, r, base, height, waves, xy, ok, z])
        return enqueue


@dataclasses.dataclass(frozen=True, kw_only=True)
class PinchParams(plugin.Params):
    hw_mean: float = plugin.dynamic(0.1, lo=0.0, hi=1.0, doc="mean half-width")
    hw_amp: float = plugin.dynamic(0.04, lo=0.0, hi=1.0, doc="half-width amplitude")
    waves: int = plugin.static(4, lo=1, doc="narrowings per loop")


class Pinch(plugin.Stage):
    kind = "generator"
    name = "acme.pinch"
    output = "centerline"
    Params = PinchParams
    provides = frozenset({"half_width"})
    writes = {"xy": "raw.xy", "valid": "gen.valid", "hw": "raw.half_width"}

    def capacity(self, p, layout):
        return {"raw": RAW}

    def build(self, b, p):
        xy, ok = b.write("xy", wp.vec2f, per="raw"), b.write("valid", wp.int32)
        hw = b.write("hw", wp.float32, interp="linear")
        salt, seed, n = b.salt("phase"), b.seed, RAW
        mean, amp, waves = b.param("hw_mean"), b.param("hw_amp"), float(p.waves)
        r = 1.0 * b.scale

        def enqueue(ctx):
            ctx.launch(_circle_attr_k, b.E * n,
                       [seed, salt, n, r, mean, amp, waves, xy, ok, hw])
        return enqueue


# ---------------------------------------------------------------------------------------
# Width stages
# ---------------------------------------------------------------------------------------

@wp.kernel
def _wave_k(count: wp.array(dtype=wp.int32), mean: wp.array(dtype=wp.float32),
            amp: wp.array(dtype=wp.float32), waves: float, S: int,
            out: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // S
    i = t - e * S
    c = count[e]
    if i >= c:
        out[t] = wp.nan
        return
    out[t] = mean[e] + amp[e] * wp.sin(_TWO_PI * waves * float(i) / float(c))


@dataclasses.dataclass(frozen=True, kw_only=True)
class WaveParams(plugin.Params):
    hw_mean: float = plugin.dynamic(0.1, lo=0.0, hi=1.0, doc="mean half-width")
    hw_amp: float = plugin.dynamic(0.03, lo=0.0, hi=1.0, doc="half-width amplitude")
    waves: int = plugin.static(3, lo=1, doc="narrowings per loop")


class Wave(plugin.Stage):
    kind = "width"
    name = "acme.wave"
    Params = WaveParams
    reads = {"count": "cl.count"}
    writes = {"half_width": "cl.half_width"}

    def build(self, b, p):
        count = b.read("count")
        out = b.write("half_width", wp.float32, per="cl")
        mean, amp, waves = b.param("hw_mean"), b.param("hw_amp"), float(p.waves)
        S = int(b.layout.stride)

        def enqueue(ctx):
            ctx.launch(_wave_k, b.E * S, [count, mean, amp, waves, S, out])
        return enqueue


class WaveAfter(Wave):
    name = "acme.wave_after"
    reads = {"count": "cl.count", "half_width": "cl.half_width"}

    def build(self, b, p):
        b.read("half_width")
        return super().build(b, p)


@wp.kernel
def _rescale_k(src: wp.array(dtype=wp.float32), factor: wp.array(dtype=wp.float32),
               count: wp.array(dtype=wp.int32), S: int, out: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // S
    i = t - e * S
    if i >= count[e]:
        out[t] = wp.nan
        return
    out[t] = factor[e] * src[t]


@dataclasses.dataclass(frozen=True, kw_only=True)
class RescaleParams(plugin.Params):
    factor: float = plugin.dynamic(1.2, lo=0.0, hi=10.0, doc="half-width factor")


class Rescale(plugin.Stage):
    kind = "width"
    name = "acme.rescale"
    Params = RescaleParams
    reads = {"half_width": "cl.half_width", "count": "cl.count"}
    writes = {"half_width": "cl.half_width"}

    def build(self, b, p):
        src, count = b.read("half_width"), b.read("count")
        out = b.write("half_width", wp.float32, per="cl")
        factor, S = b.param("factor"), int(b.layout.stride)

        def enqueue(ctx):
            ctx.launch(_rescale_k, b.E * S, [src, factor, count, S, out])
        return enqueue


@wp.kernel
def _const_k(v: float, out: wp.array(dtype=wp.float32)):
    out[wp.tid()] = v


class Negative(plugin.Stage):
    kind = "width"
    name = "acme.negative"
    writes = {"half_width": "cl.half_width"}

    def build(self, b, p):
        out = b.write("half_width", wp.float32, per="cl")
        n = int(out.shape[0])

        def enqueue(ctx):
            ctx.launch(_const_k, n, [-1.0, out])
        return enqueue


@wp.kernel
def _curv_width_k(xy: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32), S: int,
                  w: wp.array(dtype=wp.float32), curv: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // S
    i = t - e * S
    c = count[e]
    if i >= c:
        w[t] = wp.nan
        curv[t] = wp.nan
        return
    w[t] = 0.1
    p0 = xy[e * S + (i + c - 1) % c]
    p2 = xy[e * S + (i + 1) % c]
    curv[t] = wp.length(p2 - 2.0 * xy[t] + p0)


class CurvWidth(plugin.Stage):
    kind = "width"
    name = "acme.curv_width"
    reads = {"xy": "cl.xy", "count": "cl.count"}
    writes = {"half_width": "cl.half_width", "curv": "cl.acme.curv"}

    def build(self, b, p):
        xy, count = b.read("xy"), b.read("count")
        w = b.write("half_width", wp.float32, per="cl")
        curv = b.write("curv", wp.float32, per="cl")
        S = int(b.layout.stride)

        def enqueue(ctx):
            ctx.launch(_curv_width_k, b.E * S, [xy, count, S, w, curv])
        return enqueue


@wp.kernel
def _curv_check_k(curv: wp.array(dtype=wp.float32), count: wp.array(dtype=wp.int32),
                  S: int, limit: float, bit: int, fail_in: wp.array(dtype=wp.int32),
                  fail_out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    f = fail_in[e]
    for i in range(count[e]):
        if curv[e * S + i] > limit:
            f = f | (1 << bit)
    fail_out[e] = f


class CurvCheck(plugin.Stage):
    kind = "validity"
    name = "acme.curv_check"
    reads = {"curv": "cl.acme.curv", "count": "cl.count", "fail": "valid.fail"}
    writes = {"fail": "valid.fail"}

    def build(self, b, p):
        curv, count = b.read("curv"), b.read("count")
        fail_in, fail_out = b.read("fail"), b.write("fail", wp.int32)
        bit, S = b.fail_bit(), int(b.layout.stride)

        def enqueue(ctx):
            ctx.launch(_curv_check_k, b.E, [curv, count, S, 1.0, bit, fail_in, fail_out])
        return enqueue


# ---------------------------------------------------------------------------------------
# Elevation writing a plugin channel (rewired to gates.acme.slope on gates)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _slope_k(count: wp.array(dtype=wp.int32), S: int, z: wp.array(dtype=wp.float32),
             slope: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // S
    i = t - e * S
    if i >= count[e]:
        z[t] = 0.0
        slope[t] = wp.nan
        return
    z[t] = 0.0
    slope[t] = float(i)


class Slope(plugin.Stage):
    kind = "elevation"
    name = "acme.slope"
    reads = {"count": "cl.count"}
    writes = {"z": "cl.z", "slope": "cl.acme.slope"}

    def build(self, b, p):
        count = b.read("count")
        z = b.write("z", wp.float32, per="cl")
        slope_per = b.per_of("z")                  # "cl" on a track, "gates" on gates
        slope = b.write("slope", wp.float32, per=slope_per, export=True)
        S = b.stride_of("z")

        def enqueue(ctx):
            ctx.launch(_slope_k, b.E * S, [count, S, z, slope])
        return enqueue


# ---------------------------------------------------------------------------------------
# Gate side
# ---------------------------------------------------------------------------------------

TAG_A = 8


@wp.kernel
def _tagged_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, A: int, r: float,
              xy: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32),
              tag: wp.array(dtype=wp.int32)):
    t = wp.tid()
    e = t // A
    i = t - e * A
    th = _TWO_PI * float(i) / float(A) + _phase(seed[e], salt)
    rr = r * (1.0 + 0.1 * wp.sin(3.0 * th))
    xy[t] = wp.vec2f(rr * wp.cos(th), rr * wp.sin(th))
    tag[t] = i % 3
    if i == 0:
        count[e] = A


class Tagged(plugin.Stage):
    kind = "gate_generator"
    name = "acme.tagged"
    output = "anchors"
    orderings = frozenset({"raw"})
    preferred = {"order": "raw"}
    writes = {"xy": "anchors.xy", "count": "anchors.count", "tag": "anchors.acme.tag"}

    def capacity(self, p, layout):
        return {"gates": TAG_A, "min_gates": TAG_A}

    def build(self, b, p):
        xy = b.write("xy", wp.vec2f, per="anchors")
        count = b.write("count", wp.int32)
        tag = b.write("tag", wp.int32, export=True)
        salt, seed = b.salt("phase"), b.seed
        r = 1.0 * b.scale

        def enqueue(ctx):
            ctx.launch(_tagged_k, b.E * TAG_A, [seed, salt, TAG_A, r, xy, count, tag])
        return enqueue


@wp.kernel
def _clearance_k(xy: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32), G: int,
                 min_c: wp.array(dtype=wp.float32), bit: int,
                 fail_in: wp.array(dtype=wp.int32), clear: wp.array(dtype=wp.float32),
                 fail_out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    c = wp.min(count[e], G)
    f = fail_in[e]
    for i in range(G):
        t = e * G + i
        if i < c:
            d = wp.length(xy[e * G + (i + 1) % c] - xy[t])
            clear[t] = d
            if d < min_c[e]:
                f = f | (1 << bit)
        else:
            clear[t] = wp.nan
    fail_out[e] = f


@dataclasses.dataclass(frozen=True, kw_only=True)
class ClearanceParams(plugin.Params):
    min_clearance: float = plugin.dynamic(0.0, lo=0.0, doc="smallest gate-to-gate step")


class GateClearance(plugin.Stage):
    kind = "validity"
    name = "acme.gate_clearance"
    Params = ClearanceParams
    reads = {"xy": "gates.xy", "count": "gates.count", "fail": "valid.fail"}
    writes = {"fail": "valid.fail", "clearance": "gates.acme.clearance"}

    def build(self, b, p):
        xy, count, fail_in = b.read("xy"), b.read("count"), b.read("fail")
        fail_out = b.write("fail", wp.int32)
        clear = b.write("clearance", wp.float32, per="gates", export=True)
        bit, G = b.fail_bit(), int(b.layout.stride)
        min_c = b.param("min_clearance")

        def enqueue(ctx):
            ctx.launch(_clearance_k, b.E,
                       [xy, count, G, min_c, bit, fail_in, clear, fail_out])
        return enqueue


@wp.kernel
def _st_k(xy: wp.array(dtype=wp.vec2f), n: int, A: int, out: wp.array(dtype=wp.vec2f),
          count: wp.array(dtype=wp.int32)):
    t = wp.tid()
    e = t // A
    k = t - e * A
    out[t] = xy[e * n + (k * n) // A]
    if k == 0:
        count[e] = A


@dataclasses.dataclass(frozen=True, kw_only=True)
class StParams(plugin.Params):
    gates: int = plugin.static(12, lo=3, doc="stations per env")


class St(plugin.Stage):
    kind = "stations"
    name = "acme.st"
    Params = StParams
    reads = {"xy": "raw.xy"}
    writes = {"anchors": "anchors.xy", "anchor_count": "anchors.count"}

    def capacity(self, p, layout):
        return {"gates": int(p.gates), "min_gates": 0}

    def build(self, b, p):
        xy = b.read("xy")
        out = b.write("anchors", wp.vec2f, per="anchors")
        count = b.write("anchor_count", wp.int32)
        n, A = b.stride_of("xy"), int(p.gates)

        def enqueue(ctx):
            ctx.launch(_st_k, b.E * A, [xy, n, A, out, count])
        return enqueue


GENERATORS = (Ribbon, RibbonPlain, RibbonMetres, RibbonPartial, HelixLoop, Pinch)
STAGES = (Wave, WaveAfter, Rescale, Negative, CurvWidth, CurvCheck, Slope, GateClearance,
          St)
GATE_GENERATORS = (Tagged,)


def register(reg) -> None:
    """The ``acme-channels`` plugin's ``register(reg)``."""
    for cls in (*GENERATORS, *STAGES, *GATE_GENERATORS):
        reg.add(cls)


def acme_channels_entry_point() -> FakeEntryPoint:
    """The fake ``track_gen.plugins`` entry point of the ``acme-channels``
    distribution."""
    return FakeEntryPoint("acme_channels", register, ACME)
