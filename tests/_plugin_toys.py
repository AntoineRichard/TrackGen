"""Toy out-of-tree plugins and a fake ``importlib.metadata`` distribution for tests.

The toys use only the public plugin surface (``track_gen.plugin``, ``track_gen.exec``):

- ``toy.circle`` (kind ``"generator"``): a pure-Warp kernel writing a circle of
  ``config.num_points`` points per env (radius jittered per seed), using the supplied context.
- ``toy.ring`` (kind ``"gate_generator"``): eight gate anchors on a jittered ring,
  a gate Stage with ``orderings={"raw"}``.

Every launch goes through the ``LaunchContext`` (stream-explicit), so the toys also run
under the no-host-sync and stream-explicit probes.
"""
from __future__ import annotations

import warp as wp

from track_gen.exec import LaunchContext, resolve
from track_gen.plugin import ENTRY_POINT_GROUP, GeneratorSpec, Stage

TOY_GATES = 8
_CIRCLE_RADIUS = 0.6   # in units of config.scale (bezier-like extent of ~1.2)


@wp.kernel
def _toy_circle_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
                  out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    state = wp.rand_init(seeds[e])
    r = radius * (0.9 + 0.2 * wp.randf(state))
    t = 2.0 * wp.pi * float(i) / float(n)
    out[e * n + i] = wp.vec2f(r * wp.cos(t), r * wp.sin(t))
    if i == 0:
        valid[e] = 1


def toy_circle_alloc_scratch(config):
    return None


def toy_circle_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                        ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N = int(config.num_envs), int(config.num_points)
    ctx.launch(_toy_circle_k, (E, N),
               [seeds_wp, N, _CIRCLE_RADIUS * float(config.scale), out_centerline,
                out_valid_wp])


@wp.kernel
def _toy_ring_k(seeds: wp.array(dtype=wp.int32), k: int, g: int, radius: float,
                pos: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    if i < k:
        state = wp.rand_init(seeds[e], i)
        t = 2.0 * wp.pi * (float(i) + 0.3 * (wp.randf(state) - 0.5)) / float(k)
        pos[e * g + i] = wp.vec2f(radius * wp.cos(t), radius * wp.sin(t))
    else:
        pos[e * g + i] = wp.vec2f(wp.nan, wp.nan)
    if i == 0:
        count[e] = k


def toy_ring_alloc_scratch(config):
    return None


def toy_ring_generate(seeds_wp, config, out, scratch,
                      ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, G = int(config.num_envs), int(config.max_gates)
    ctx.launch(_toy_ring_k, (E, G),
               [seeds_wp, TOY_GATES, G, _CIRCLE_RADIUS * float(config.scale),
                out.position, out.count])


def toy_ring_max_gates(config) -> int:
    return TOY_GATES


TOY_CIRCLE = GeneratorSpec(
    name="toy.circle",
    alloc_scratch=toy_circle_alloc_scratch,
    generate=toy_circle_generate,
)

class ToyRing(Stage):
    kind = "gate_generator"
    name = "toy.ring"
    output = "anchors"
    writes = {"xy": "anchors.xy", "count": "anchors.count"}
    orderings = frozenset({"raw"})
    preferred = {"order": "raw"}

    def capacity(self, p, layout):
        return {"gates": TOY_GATES}

    def build(self, b, p):
        xy = b.write("xy", wp.vec2f, per="anchors")
        count = b.write("count", wp.int32)
        radius = _CIRCLE_RADIUS * float(b.scale)

        def enqueue(ctx):
            ctx.launch(_toy_ring_k, (b.E, TOY_GATES),
                       [b.seed, TOY_GATES, TOY_GATES, radius, xy, count])
        return enqueue


TOY_RING = ToyRing


def register_toys(reg) -> None:
    """The plugin's ``register(reg)`` entry point: both toys."""
    reg.add(TOY_CIRCLE)
    reg.add(TOY_RING)


# ---------------------------------------------------------------------------------------
# A fake distribution: EntryPoint-like objects for a monkeypatched entry_points()
# ---------------------------------------------------------------------------------------

class FakeDist:
    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self.metadata = {"Name": name, "Version": version}


class FakeEntryPoint:
    """``load()`` returns ``target`` (or raises it, when it is an exception)."""

    def __init__(self, name: str, target, dist: "FakeDist | None" = None,
                 value: "str | None" = None, group: str = ENTRY_POINT_GROUP) -> None:
        self.name = name
        self.group = group
        self.dist = dist
        self._target = target
        if value is None:
            module = getattr(target, "__module__", None) or name
            value = f"{module}:{getattr(target, '__name__', 'register')}"
        self.value = value
        self.module = value.partition(":")[0]
        self.loads = 0

    def load(self):
        self.loads += 1
        if isinstance(self._target, BaseException):
            raise self._target
        return self._target

    def __repr__(self) -> str:
        return f"FakeEntryPoint({self.name!r}, {self.value!r})"


def install_entry_points(monkeypatch, *eps: FakeEntryPoint) -> list:
    """Make the registry's ``entry_points()`` return ``eps``; returns the list of
    ``group=`` values it was called with."""
    from track_gen.plugin import _registry

    calls: list = []

    def entry_points(**params):
        calls.append(params.get("group"))
        return [ep for ep in eps if ep.group == params.get("group")]

    monkeypatch.setattr(_registry, "entry_points", entry_points)
    return calls


# ---------------------------------------------------------------------------------------
# Params / Stage toys (roadmap PR8): public surface only (track_gen.plugin, track_gen.ops)
# ---------------------------------------------------------------------------------------
#
# - ``toy.wobble``: a ``plugin.Stage`` generator with one STATIC field (``lobes``) and one
#   DYNAMIC field (``amp``, an [E] float32 array read by pointer), a raw capacity of 128
#   points (!= num_points) and a salted per-env RNG stream.
# - ``toy.wobble_host``: the same with ``capture = HOST``.
# - ``toy.wobble_input``: also declares a per-env input ``difficulty`` (default 0.5) that
#   scales ``amp``.
# - ``toy.lobed``: a simple ``GeneratorSpec`` with a static ``params=`` class; its generate
#   reads ``config.generator_params.lobes``.

import dataclasses as _dc  # noqa: E402

from track_gen import plugin as _plugin  # noqa: E402
from track_gen.ops import hash32 as _hash32  # noqa: E402

WOBBLE_RAW = 128
_TWO_PI = wp.constant(6.2831853)


@wp.func
def _wobble_state(seed: wp.int32, salt: wp.uint32):
    return wp.rand_init(wp.int32(_hash32(wp.uint32(seed) ^ salt) & wp.uint32(0x7FFFFFFF)))


@wp.kernel
def _wobble_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32,
              amp: wp.array(dtype=wp.float32), lobes: int, n: int, radius: float,
              xy: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    st = _wobble_state(seed[e], salt)
    th = _TWO_PI * float(i) / float(n)
    r = radius * (1.0 + amp[e] * wp.sin(float(lobes) * th + _TWO_PI * wp.randf(st)))
    xy[t] = wp.vec2f(r * wp.cos(th), r * wp.sin(th))


@wp.kernel
def _wobble_input_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32,
                    amp: wp.array(dtype=wp.float32), difficulty: wp.array(dtype=wp.float32),
                    lobes: int, n: int, radius: float, xy: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    st = _wobble_state(seed[e], salt)
    th = _TWO_PI * float(i) / float(n)
    a = amp[e] * difficulty[e]
    r = radius * (1.0 + a * wp.sin(float(lobes) * th + _TWO_PI * wp.randf(st)))
    xy[t] = wp.vec2f(r * wp.cos(th), r * wp.sin(th))


@_dc.dataclass(frozen=True, kw_only=True)
class ToyWobbleParams(_plugin.Params):
    lobes: int = _plugin.static(5, lo=2, hi=12, doc="lobes per loop")
    amp: float = _plugin.dynamic(0.15, lo=0.0, hi=0.4, doc="radial wobble amplitude",
                                 ui="slider")


class ToyWobble(_plugin.Stage):
    kind = "generator"
    name = "toy.wobble"
    output = "centerline"
    Params = ToyWobbleParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}
    RAW = WOBBLE_RAW                     # != num_points (256): exercises capacity()

    def capacity(self, p, layout):
        return {"raw": self.RAW}

    def build(self, b, p):
        xy, ok = b.write("xy", wp.vec2f, per="raw"), b.write("valid", wp.int32)
        amp, salt = b.param("amp"), b.salt("shape")
        n, radius, lobes = self.RAW, 0.6 * b.scale, int(p.lobes)

        def enqueue(ctx):
            ctx.launch(_wobble_k, b.E * n, [b.seed, salt, amp, lobes, n, radius, xy])
            ctx.fill(ok, 1)

        return enqueue


class ToyWobbleHost(ToyWobble):
    name = "toy.wobble_host"
    capture = _plugin.CaptureClass.HOST


class ToyWobbleInput(ToyWobble):
    name = "toy.wobble_input"

    def build(self, b, p):
        xy, ok = b.write("xy", wp.vec2f, per="raw"), b.write("valid", wp.int32)
        amp, salt = b.param("amp"), b.salt("shape")
        difficulty = b.input("difficulty", wp.float32, 0.5)
        n, radius, lobes = self.RAW, 0.6 * b.scale, int(p.lobes)

        def enqueue(ctx):
            ctx.launch(_wobble_input_k, b.E * n,
                       [b.seed, salt, amp, difficulty, lobes, n, radius, xy])
            ctx.fill(ok, 1)

        return enqueue


@_dc.dataclass(frozen=True, kw_only=True)
class ToyLobedParams(_plugin.Params):
    lobes: int = _plugin.static(4, lo=2, hi=12, doc="lobes per loop")
    amp: float = _plugin.static(0.1, lo=0.0, hi=0.3, doc="radial amplitude")


@wp.kernel
def _lobed_k(seeds: wp.array(dtype=wp.int32), lobes: int, amp: float, n: int,
             radius: float, out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    state = wp.rand_init(seeds[e])
    phase = _TWO_PI * wp.randf(state)
    t = _TWO_PI * float(i) / float(n)
    r = radius * (1.0 + amp * wp.sin(float(lobes) * t + phase))
    out[e * n + i] = wp.vec2f(r * wp.cos(t), r * wp.sin(t))
    if i == 0:
        valid[e] = 1


def toy_lobed_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                       ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    p = config.generator_params                 # the resolved ToyLobedParams
    E, N = int(config.num_envs), int(config.num_points)
    ctx.launch(_lobed_k, (E, N), [seeds_wp, int(p.lobes), float(p.amp), N,
                                  _CIRCLE_RADIUS * float(config.scale), out_centerline,
                                  out_valid_wp])


TOY_LOBED = GeneratorSpec(
    name="toy.lobed",
    alloc_scratch=toy_circle_alloc_scratch,
    generate=toy_lobed_generate,
    params=ToyLobedParams,
)


def register_stage_toys(reg) -> None:
    """A second plugin ``register(reg)``: the Stage toys and the legacy ``params=`` toy."""
    reg.add(ToyWobble)
    reg.add(ToyWobbleHost)
    reg.add(ToyWobbleInput)
    reg.add(TOY_LOBED)


# ---------------------------------------------------------------------------------------
# Roadmap PR13b: a centerline generator whose declared writes depend on its params
# (Stage.declare): raw.z (provides z) and raw.acme.tag only when ``tagged`` is set.
# ---------------------------------------------------------------------------------------

DECLARED = "acme.declared"


@wp.kernel
def _declared_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, lobes: int, n: int,
                radius: float, xy: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    st = _wobble_state(seed[e], salt)
    th = _TWO_PI * float(i) / float(n)
    r = radius * (1.0 + 0.15 * wp.sin(float(lobes) * th + _TWO_PI * wp.randf(st)))
    xy[t] = wp.vec2f(r * wp.cos(th), r * wp.sin(th))


@wp.kernel
def _declared_attrs_k(n: int, z: wp.array(dtype=wp.float32),
                      tag: wp.array(dtype=wp.int32)):
    t = wp.tid()
    i = t - (t // n) * n
    z[t] = 0.2 + 0.1 * wp.sin(_TWO_PI * float(i) / float(n))
    tag[t] = (4 * i) // n                    # labels 0..3 along the loop


@_dc.dataclass(frozen=True, kw_only=True)
class ToyDeclaredParams(_plugin.Params):
    lobes: int = _plugin.static(5, lo=2, hi=12, doc="lobes per loop")
    tagged: bool = _plugin.static(False, doc="also write raw.z and raw.acme.tag")


class ToyDeclared(_plugin.Stage):
    kind = "generator"
    name = DECLARED
    output = "centerline"
    Params = ToyDeclaredParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}
    seeded = True

    @classmethod
    def declare(cls, params):
        writes, provides = dict(cls.writes), frozenset()
        if params.tagged:
            writes.update(z="raw.z", tag="raw.acme.tag")
            provides = frozenset({"z"})
        return writes, provides

    def build(self, b, p):
        xy, ok = b.write("xy", wp.vec2f, per="raw"), b.write("valid", wp.int32)
        salt, n, radius, lobes = b.salt("shape"), b.raw, 0.6 * b.scale, int(p.lobes)
        z = tag = None
        if p.tagged:
            z = b.write("z", wp.float32, per="raw")
            tag = b.write("tag", wp.int32, per="raw", interp="nearest", export=True)

        def enqueue(ctx):
            ctx.launch(_declared_k, b.E * n, [b.seed, salt, lobes, n, radius, xy])
            if z is not None:
                ctx.launch(_declared_attrs_k, b.E * n, [n, z, tag])
            ctx.fill(ok, 1)

        return enqueue


def register_declared(reg) -> None:
    """Register ``acme.declared`` (roadmap PR13b: ``Stage.declare``)."""
    reg.add(ToyDeclared)
