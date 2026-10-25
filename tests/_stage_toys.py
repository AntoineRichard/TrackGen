"""Toy pipeline-stage plugins (roadmap PR10b): one relax, one elevation and one validity
stage, written against the public plugin surface only (``track_gen.plugin``,
``track_gen.ops``), and the fake ``acme-stages`` distribution that registers them.

- relax ``acme.taubin``: Taubin smoothing of ``cl.xy`` as a REWRITE (reads the previous
  version, ping-pongs through ``b.temp_like``, ends in the ``b.write`` buffer: an even
  number of passes needs no copy). ``passes`` is static, ``lam`` / ``mu`` are dynamic.
- elevation ``acme.sinusoid_hills``: ``cl.z = base + amplitude * sin(2 pi waves s / L +
  phase)`` over the plan-view arc length, with a salted per-env phase; ``amplitude``
  is dynamic. ``from_legacy`` maps the flat ``z_base``.
- validity ``acme.max_curvature``: ORs its own fail bit (``b.fail_bit()``) into
  ``valid.fail`` where ``max(cl.kappa) > kappa_max[e]``; ``kappa_max`` is dynamic.

``acme_entry_point()`` is the fake entry point that installs them (``register``).
"""
from __future__ import annotations

import dataclasses

import warp as wp

from tests._plugin_toys import FakeDist, FakeEntryPoint
from track_gen import plugin
from track_gen.ops import hash32

#: The plugin API version this module targets (read by the registry's loader).
track_gen_api = 1

ACME = FakeDist("acme-stages", "0.1.0")
TAUBIN = "acme.taubin"
HILLS = "acme.sinusoid_hills"
KMAX = "acme.max_curvature"
_TWO_PI = wp.constant(6.2831853)


# ---------------------------------------------------------------------------------------
# relax: acme.taubin
# ---------------------------------------------------------------------------------------

@wp.kernel
def _laplace_k(src: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32),
               f: wp.array(dtype=wp.float32), S: int, dst: wp.array(dtype=wp.vec2f)):
    # One thread per (env, point): one umbrella step x + f * (mean(neighbours) - x) over
    # the closed loop of count[e] real points; padding is carried (NaN stays NaN).
    t = wp.tid()
    e = t // S
    i = t - e * S
    n = count[e]
    if i >= n or n < 3:
        dst[t] = src[t]
        return
    base = e * S
    prev = src[base + (i + n - 1) % n]
    nxt = src[base + (i + 1) % n]
    x = src[t]
    dst[t] = x + f[e] * (0.5 * (prev + nxt) - x)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TaubinParams(plugin.Params):
    passes: int = plugin.static(10, lo=1, doc="(lam, mu) pass pairs")
    lam: float = plugin.dynamic(0.5, lo=0.0, hi=1.0, doc="shrink step")
    mu: float = plugin.dynamic(-0.53, lo=-1.0, hi=0.0, doc="inflate step")


class Taubin(plugin.Stage):
    kind = "relax"
    name = TAUBIN
    Params = TaubinParams
    reads = {"xy": "cl.xy", "count": "cl.count"}
    writes = {"xy": "cl.xy"}                      # a rewrite, through one port

    def build(self, b, p):
        x, count = b.read("xy"), b.read("count")
        out, tmp = b.write("xy", wp.vec2f, per="cl"), b.temp_like("xy")
        lam, mu, S = b.param("lam"), b.param("mu"), int(b.layout.stride)
        steps = [f for _ in range(int(p.passes)) for f in (lam, mu)]   # static count
        n = b.E * S

        def enqueue(ctx):
            src = x
            for k, f in enumerate(steps):          # x -> tmp -> out -> tmp ... -> out
                dst = tmp if k % 2 == 0 else out
                ctx.launch(_laplace_k, n, [src, count, f, S, dst])
                src = dst
        return enqueue


# ---------------------------------------------------------------------------------------
# elevation: acme.sinusoid_hills
# ---------------------------------------------------------------------------------------

@wp.kernel
def _hills_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32,
             cum: wp.array(dtype=wp.float32), perim: wp.array(dtype=wp.float32),
             count: wp.array(dtype=wp.int32), amp: wp.array(dtype=wp.float32),
             waves: float, z_base: float, S: int, z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // S
    i = t - e * S
    if i >= count[e]:
        z[t] = 0.0                                 # the built-ins' padding
        return
    st = wp.rand_init(wp.int32(hash32(wp.uint32(seed[e]) ^ salt) & wp.uint32(0x7FFFFFFF)))
    phase = _TWO_PI * wp.randf(st)
    s = cum[t] / wp.max(perim[e], 1.0e-9)
    z[t] = z_base + amp[e] * wp.sin(_TWO_PI * waves * s + phase)


@dataclasses.dataclass(frozen=True, kw_only=True)
class HillsParams(plugin.Params):
    amplitude: float = plugin.dynamic(0.2, lo=0.0, hi=5.0, doc="hill height")
    waves: int = plugin.static(3, lo=1, doc="hills per loop")
    base: float = plugin.static(0.0, doc="baseline altitude")

    @classmethod
    def from_legacy(cls, cfg):
        return cls(base=float(cfg.z_base))


class SinusoidHills(plugin.Stage):
    kind = "elevation"
    name = HILLS
    Params = HillsParams
    reads = {"cum": "cl.arclen2", "perim": "cl.length2", "count": "cl.count"}
    writes = {"z": "cl.z"}

    def build(self, b, p):
        cum, perim, count = b.read("cum"), b.read("perim"), b.read("count")
        z = b.write("z", wp.float32, per="cl")
        amp, salt = b.param("amplitude"), b.salt("phase")
        S = int(b.layout.stride)
        waves, base, seed = float(p.waves), float(p.base), b.seed

        def enqueue(ctx):
            ctx.launch(_hills_k, b.E * S,
                       [seed, salt, cum, perim, count, amp, waves, base, S, z])
        return enqueue


# ---------------------------------------------------------------------------------------
# validity: acme.max_curvature
# ---------------------------------------------------------------------------------------

@wp.kernel
def _kmax_k(kappa: wp.array(dtype=wp.float32), count: wp.array(dtype=wp.int32),
            kappa_max: wp.array(dtype=wp.float32), S: int, bit: int,
            fail_in: wp.array(dtype=wp.int32), fail_out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    base = e * S
    m = float(-3.0e38)
    for i in range(count[e]):
        k = kappa[base + i]
        if k > m:                                  # NaN never wins
            m = k
    f = fail_in[e]
    if m > kappa_max[e]:
        f = f | (1 << bit)
    fail_out[e] = f


@dataclasses.dataclass(frozen=True, kw_only=True)
class KappaParams(plugin.Params):
    kappa_max: float = plugin.dynamic(50.0, lo=0.0, doc="largest curvature allowed")


class MaxCurvature(plugin.Stage):
    kind = "validity"
    name = KMAX
    Params = KappaParams
    reads = {"kappa": "cl.kappa", "count": "cl.count", "fail": "valid.fail"}
    writes = {"fail": "valid.fail"}

    def build(self, b, p):
        kappa, count = b.read("kappa"), b.read("count")
        fail_in, fail_out = b.read("fail"), b.write("fail", wp.int32)
        bit = b.fail_bit()
        kmax, S = b.param("kappa_max"), int(b.layout.stride)

        def enqueue(ctx):
            ctx.launch(_kmax_k, b.E, [kappa, count, kmax, S, bit, fail_in, fail_out])
        return enqueue


STAGES = (Taubin, SinusoidHills, MaxCurvature)


def register(reg) -> None:
    """The ``acme-stages`` plugin's ``register(reg)``."""
    for cls in STAGES:
        reg.add(cls)


def acme_entry_point() -> FakeEntryPoint:
    """The fake ``track_gen.plugins`` entry point of the ``acme-stages`` distribution."""
    return FakeEntryPoint("acme", register, ACME)
