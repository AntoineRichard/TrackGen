"""Toy gate-pipeline plugins (roadmap PR11), written against the public plugin surface
only (``track_gen.plugin``, ``track_gen.ops``), and the fake ``acme-gates`` distribution
that registers them.

- gate generator ``acme.helix``: ``gates`` anchors on a circle (a salted per-env phase),
  in loop order, that PROVIDE their own altitude and tangent (``anchors.z``: a 2-wave
  helix, ``anchors.tangent``: its analytic derivative). ``orderings={"raw"}`` (provided
  attributes are carried by index), ``preferred={"order": "raw"}``.
- gate generator ``acme.helix_posed``: the helix that also provides ``half_size``
  (``0.02 + 0.01 * (i % 3)``) and ``roll`` (``0.3 * sin(i)``).
- gate generator ``acme.scatter``: a plain anchors cloud (``gates`` uniform points in
  ``[-1, 1]^2 * scale``, count ``gates``), ``orderings=None`` with no exclusion (every
  registered order), ``preferred={"order": "ccw"}``.
- order ``acme.reverse``: the first ``m`` anchors in reverse order (NaN padding), the
  count copied; ping-pongs through ``b.temp_like("xy")`` (a ``gates.*`` port).
- generator ``acme.superformula``: a third-party TRACK generator (target-architecture
  §11(a)): ``out[t] = r(th + phase) * (cos th, sin th) * 1.44 * scale`` with the
  superformula radius; ``m`` and ``n1`` are dynamic. Gates through ``stations``.
- validity ``acme.max_height``: ORs its own fail bit where the highest real gate is
  above ``z_cap[e]`` (dynamic).

``acme_gates_entry_point()`` is the fake entry point that installs them.
"""
from __future__ import annotations

import dataclasses

import warp as wp

from tests._plugin_toys import FakeDist, FakeEntryPoint
from track_gen import plugin
from track_gen.ops import hash32

#: The plugin API version this module targets (read by the registry's loader).
track_gen_api = 1

ACME_GATES = FakeDist("acme-gates", "0.1.0")
HELIX = "acme.helix"
HELIX_POSED = "acme.helix_posed"
SCATTER = "acme.scatter"
REVERSE = "acme.reverse"
SUPERFORMULA = "acme.superformula"
MAX_HEIGHT = "acme.max_height"
_TWO_PI = wp.constant(6.2831853)


@wp.func
def _salted(seed: wp.int32, salt: wp.uint32):
    return wp.rand_init(wp.int32(hash32(wp.uint32(seed) ^ salt) & wp.uint32(0x7FFFFFFF)))


# ---------------------------------------------------------------------------------------
# gate generators: acme.helix / acme.helix_posed / acme.scatter
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, kw_only=True)
class HelixParams(plugin.Params):
    gates: int = plugin.static(16, lo=3, hi=64, doc="anchors per env")
    radius: float = plugin.static(0.6, lo=0.1, hi=2.0, doc="circle radius (x scale)")
    rise: float = plugin.static(0.3, lo=0.0, hi=2.0, doc="helix amplitude")


@wp.kernel
def _helix_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, A: int, radius: float,
             rise: float, posed: int, xy: wp.array(dtype=wp.vec2f),
             count: wp.array(dtype=wp.int32), z: wp.array(dtype=wp.float32),
             tangent: wp.array(dtype=wp.vec3f), half_size: wp.array(dtype=wp.float32),
             roll: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // A
    i = t - e * A
    st = _salted(seed[e], salt)
    phase = _TWO_PI * wp.randf(st)
    th = _TWO_PI * float(i) / float(A) + phase
    xy[t] = wp.vec2f(radius * wp.cos(th), radius * wp.sin(th))
    z[t] = 0.5 + rise * wp.sin(2.0 * th)
    tangent[t] = wp.vec3f(-radius * wp.sin(th), radius * wp.cos(th),
                          2.0 * rise * wp.cos(2.0 * th))
    if posed != 0:
        half_size[t] = 0.02 + 0.01 * float(i % 3)
        roll[t] = 0.3 * wp.sin(float(i))
    if i == 0:
        count[e] = A


class Helix(plugin.Stage):
    kind = "gate_generator"
    name = HELIX
    output = "anchors"
    Params = HelixParams
    writes = {"xy": "anchors.xy", "count": "anchors.count", "z": "anchors.z",
              "tangent": "anchors.tangent"}
    provides = frozenset({"z", "tangent"})
    orderings = frozenset({"raw"})
    preferred = {"order": "raw"}

    def capacity(self, params, layout):
        return {"gates": int(params.gates), "min_gates": int(params.gates)}

    def _posed(self, b):
        return None, None

    def build(self, b, p):
        xy = b.write("xy", wp.vec2f, per="anchors")
        count = b.write("count", wp.int32)
        z = b.write("z", wp.float32, per="anchors")
        tangent = b.write("tangent", wp.vec3f, per="anchors")
        half_size, roll = self._posed(b)
        posed = 1 if half_size is not None else 0
        if half_size is None:                      # unused by the kernel when posed == 0
            half_size, roll = z, z
        A = b.stride_of("xy")
        salt, seed = b.salt("phase"), b.seed
        radius, rise = float(p.radius) * b.scale, float(p.rise)

        def enqueue(ctx):
            ctx.launch(_helix_k, b.E * A, [seed, salt, A, radius, rise, posed, xy, count,
                                           z, tangent, half_size, roll])
        return enqueue


class HelixPosed(Helix):
    name = HELIX_POSED
    writes = {**Helix.writes, "half_size": "anchors.half_size", "roll": "anchors.roll"}
    provides = frozenset({"z", "tangent", "half_size", "roll"})

    def _posed(self, b):
        return (b.write("half_size", wp.float32, per="anchors"),
                b.write("roll", wp.float32, per="anchors"))


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScatterParams(plugin.Params):
    gates: int = plugin.static(10, lo=3, hi=64, doc="anchors per env")


@wp.kernel
def _scatter_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, A: int, scale: float,
               xy: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32)):
    e = wp.tid()
    st = _salted(seed[e], salt)
    for i in range(A):
        x = 2.0 * wp.randf(st) - 1.0
        y = 2.0 * wp.randf(st) - 1.0
        xy[e * A + i] = wp.vec2f(x * scale, y * scale)
    count[e] = A


class Scatter(plugin.Stage):
    kind = "gate_generator"
    name = SCATTER
    output = "anchors"
    Params = ScatterParams
    writes = {"xy": "anchors.xy", "count": "anchors.count"}
    preferred = {"order": "ccw"}

    def capacity(self, params, layout):
        return {"gates": int(params.gates), "min_gates": int(params.gates)}

    def build(self, b, p):
        xy = b.write("xy", wp.vec2f, per="anchors")
        count = b.write("count", wp.int32)
        A = b.stride_of("xy")
        salt, seed, scale = b.salt("points"), b.seed, b.scale

        def enqueue(ctx):
            ctx.launch(_scatter_k, b.E, [seed, salt, A, scale, xy, count])
        return enqueue


# ---------------------------------------------------------------------------------------
# order: acme.reverse
# ---------------------------------------------------------------------------------------

@wp.kernel
def _gather_k(src: wp.array(dtype=wp.vec2f), A: int, count: wp.array(dtype=wp.int32),
              G: int, dst: wp.array(dtype=wp.vec2f)):
    # One thread per gate slot: the first m anchors in their order, NaN padding.
    t = wp.tid()
    e = t // G
    i = t - e * G
    m = wp.max(0, wp.min(count[e], wp.min(A, G)))
    if i < m:
        dst[t] = src[e * A + i]
    else:
        dst[t] = wp.vec2f(wp.nan, wp.nan)


@wp.kernel
def _reverse_k(src: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32), A: int,
               G: int, dst: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // G
    i = t - e * G
    m = wp.max(0, wp.min(count[e], wp.min(A, G)))
    if i < m:
        dst[t] = src[e * G + (m - 1 - i)]
    else:
        dst[t] = wp.vec2f(wp.nan, wp.nan)


class Reverse(plugin.Stage):
    kind = "order"
    name = REVERSE
    reads = {"anchors": "anchors.xy", "anchor_count": "anchors.count"}
    writes = {"xy": "gates.xy", "count": "gates.count"}

    def build(self, b, p):
        src, acount = b.read("anchors"), b.read("anchor_count")
        dst = b.write("xy", wp.vec2f, per="gates")
        count = b.write("count", wp.int32)
        tmp = b.temp_like("xy")                    # [E * max_gates], a gates.* port
        A, G = b.stride_of("anchors"), b.stride_of("xy")
        n = b.E * G

        def enqueue(ctx):
            ctx.launch(_gather_k, n, [src, A, acount, G, tmp])
            ctx.launch(_reverse_k, n, [tmp, acount, A, G, dst])
            ctx.copy(count, acount)
        return enqueue


# ---------------------------------------------------------------------------------------
# generator: acme.superformula (a track generator, gates through stations)
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, kw_only=True)
class SuperformulaParams(plugin.Params):
    points: int = plugin.static(256, lo=16, doc="raw points per env")
    m: float = plugin.dynamic(5.0, lo=2.0, hi=12.0, doc="rotational symmetry")
    n1: float = plugin.dynamic(0.3, lo=0.1, hi=2.0, doc="overall exponent")
    n2: float = plugin.static(1.7, doc="term exponent")


@wp.kernel
def _superformula_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, n: int,
                    m: wp.array(dtype=wp.float32), n1: wp.array(dtype=wp.float32),
                    n2: float, scale: float, out: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    st = _salted(seed[e], salt)
    phase = _TWO_PI * wp.randf(st)
    th = _TWO_PI * float(i) / float(n)
    phi = th + phase
    q = m[e] * phi / 4.0
    s = wp.pow(wp.abs(wp.cos(q)), n2) + wp.pow(wp.abs(wp.sin(q)), n2)
    r = wp.pow(s, -1.0 / n1[e])
    out[t] = wp.vec2f(r * wp.cos(th), r * wp.sin(th)) * (1.44 * scale)


class Superformula(plugin.Stage):
    kind = "generator"
    name = SUPERFORMULA
    output = "centerline"
    Params = SuperformulaParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}

    def capacity(self, params, layout):
        return {"raw": int(params.points)}

    def build(self, b, p):
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        m, n1 = b.param("m"), b.param("n1")
        salt, seed = b.salt("phase"), b.seed
        n, n2, scale = int(b.raw), float(p.n2), float(b.scale)

        def enqueue(ctx):
            ctx.launch(_superformula_k, b.E * n, [seed, salt, n, m, n1, n2, scale, xy])
            ctx.fill(valid, 1)
        return enqueue


# ---------------------------------------------------------------------------------------
# validity: acme.max_height
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, kw_only=True)
class MaxHeightParams(plugin.Params):
    z_cap: float = plugin.dynamic(1.0, lo=-100.0, hi=100.0, doc="highest gate altitude")


@wp.kernel
def _max_height_k(position: wp.array(dtype=wp.vec3f), count: wp.array(dtype=wp.int32),
                  z_cap: wp.array(dtype=wp.float32), G: int, bit: int,
                  fail_in: wp.array(dtype=wp.int32), fail_out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    top = float(-3.0e38)
    n = wp.max(0, wp.min(count[e], G))
    for i in range(n):
        z = position[e * G + i][2]
        if z > top:                                # NaN never wins
            top = z
    f = fail_in[e]
    if top > z_cap[e]:
        f = f | (1 << bit)
    fail_out[e] = f


class MaxHeight(plugin.Stage):
    kind = "validity"
    name = MAX_HEIGHT
    Params = MaxHeightParams
    reads = {"position": "gates.position", "count": "gates.count", "fail": "valid.fail"}
    writes = {"fail": "valid.fail"}

    def build(self, b, p):
        position, count = b.read("position"), b.read("count")
        fail_in, fail_out = b.read("fail"), b.write("fail", wp.int32)
        z_cap, G = b.param("z_cap"), int(b.layout.stride)
        bit = b.fail_bit()

        def enqueue(ctx):
            ctx.launch(_max_height_k, b.E, [position, count, z_cap, G, bit, fail_in,
                                            fail_out])
        return enqueue


STAGES = (Helix, HelixPosed, Scatter, Reverse, Superformula, MaxHeight)


def register(reg) -> None:
    """The ``acme-gates`` plugin's ``register(reg)``."""
    for cls in STAGES:
        reg.add(cls)


def acme_gates_entry_point() -> FakeEntryPoint:
    """The fake ``track_gen.plugins`` entry point of the ``acme-gates`` distribution."""
    return FakeEntryPoint("acme_gates", register, ACME_GATES)
