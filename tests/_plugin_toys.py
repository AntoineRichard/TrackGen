"""Toy out-of-tree plugins and a fake ``importlib.metadata`` distribution for tests.

The toys use only the public plugin surface (``track_gen.plugin``, ``track_gen.exec``):

- ``toy.circle`` (kind ``"generator"``): a pure-Warp kernel writing a circle of
  ``config.num_points`` points per env (radius jittered per seed), ``accepts_ctx=True``.
- ``toy.ring`` (kind ``"gate_generator"``): eight gate anchors on a jittered ring,
  ``supported_orderings={"raw"}``, ``accepts_ctx=True``.

Every launch goes through the ``LaunchContext`` (stream-explicit), so the toys also run
under the no-host-sync and stream-explicit probes.
"""
from __future__ import annotations

import warp as wp

from track_gen.exec import LaunchContext, resolve
from track_gen.plugin import ENTRY_POINT_GROUP, GateGeneratorSpec, GeneratorSpec

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
    accepts_ctx=True,
)

TOY_RING = GateGeneratorSpec(
    name="toy.ring",
    alloc_scratch=toy_ring_alloc_scratch,
    generate=toy_ring_generate,
    max_gates=toy_ring_max_gates,
    supported_orderings=frozenset({"raw"}),
    accepts_ctx=True,
)


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
