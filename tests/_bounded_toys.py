"""Toy BOUNDED and HOST stages (roadmap PR12), public plugin surface only (``track_gen.plugin``,
``track_gen.ops``, ``ctx.bounded_loop`` / ``ctx.host()``).

``register_bounded_toys(reg)`` registers the four toys the CPU and CUDA tests share:

- relax ``toy.converging_smooth`` (``capture=BOUNDED``): ``iters`` Laplacian smoothing
  steps of ``cl.xy`` in a ``ctx.bounded_loop``; every ``check_every`` iterations an env
  whose largest step fell below its own tolerance (``tol`` scaled by a per-env factor in
  ``[1, 1 + tol_spread)`` drawn from its seed) stops (``running[e] = 0``, never back),
  and ``active = any(running)`` (``ops.any_set``) is refreshed for the next window. A
  stopped env's rows are never written again: an ABSORBING freeze, so every lowering
  of the loop gives the same bytes and env ``e`` depends on its own seed only.
- relax ``toy.leaky_smooth``: the same schedule and ``active`` refresh, but the smoothing
  ignores ``running`` (a converged env keeps moving): NOT absorbing. Under the CPU host
  poll the loop stops once the whole batch converged, so an env's result depends on the
  other envs (it fails ``bounded.absorbing`` and ``batch_independence``).
- generator ``toy.host_circle`` (``capture=HOST``): reads the effective seeds back through
  ``ctx.host()``, draws each env's closed loop with numpy's host RNG (a circle with two
  low harmonics) and uploads it (``b.host_buffer`` buffers); ``gen.valid = 1``.
- gate generator ``toy.host_ring`` (``capture=HOST``): ``gates`` anchors on a jittered ring,
  drawn on the host and uploaded the same way.

``register_broken_bounded_toys(reg)`` adds the variants the CPU tests break on purpose:
``toy.undeclared_loop`` (the converging smoother declared STATIC), ``toy.syncing_smooth``
(its body reads ``active`` back), ``toy.poll_caller`` (its enqueue calls the library's
host poll itself), ``toy.late_alloc_smooth`` / ``toy.late_sync_smooth`` (allocate / read
back only in the last iteration, which the CPU host poll never reaches for the kit's
seeds), the HOST relax ``toy.host_identity`` (reads ``cl.xy`` back and uploads it
unchanged) and the HOST relax ``toy.sneaky_host`` (the same through a direct readback, a
device-wide sync and a per-call allocation: the kit fails it).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import warp as wp

from track_gen import ops, plugin
from track_gen.ops import hash32

CONVERGING = "toy.converging_smooth"
LEAKY = "toy.leaky_smooth"
HOST_CIRCLE = "toy.host_circle"
HOST_RING = "toy.host_ring"
UNDECLARED = "toy.undeclared_loop"
SYNCING = "toy.syncing_smooth"
POLL_CALLER = "toy.poll_caller"
LATE_ALLOC = "toy.late_alloc_smooth"
LATE_SYNC = "toy.late_sync_smooth"
HOST_IDENTITY = "toy.host_identity"
SNEAKY_HOST = "toy.sneaky_host"

_RADIUS = 0.6          # in units of config.scale (the plugin toys' circle)


# ---------------------------------------------------------------------------------------
# relax: the bounded smoothers
# ---------------------------------------------------------------------------------------

@wp.kernel
def _smooth_k(x: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32),
              running: wp.array(dtype=wp.int32), gated: int, lam: float, S: int,
              tmp: wp.array(dtype=wp.vec2f), step: wp.array(dtype=wp.float32)):
    # One thread per (env, point): tmp = x + lam * Laplacian(x) over the closed loop of
    # count[e] points; the largest step of the env into step[e] (max: order-free).
    e, i = wp.tid()
    n = count[e]
    if i >= n or n < 3:
        return
    if gated == 1 and running[e] == 0:
        return
    b = e * S
    d = lam * (x[b + (i + n - 1) % n] + x[b + (i + 1) % n] - 2.0 * x[b + i])
    tmp[b + i] = x[b + i] + d
    wp.atomic_max(step, e, wp.length(d))


@wp.kernel
def _commit_k(tmp: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32),
              running: wp.array(dtype=wp.int32), gated: int, S: int,
              x: wp.array(dtype=wp.vec2f)):
    e, i = wp.tid()
    n = count[e]
    if i >= n or n < 3:
        return
    if gated == 1 and running[e] == 0:
        return
    x[e * S + i] = tmp[e * S + i]


@wp.kernel
def _converge_k(step: wp.array(dtype=wp.float32), seed: wp.array(dtype=wp.int32),
                salt: wp.uint32, tol: float, spread: float,
                running: wp.array(dtype=wp.int32)):
    # An env stops once its largest step falls below its own tolerance (never restarts).
    e = wp.tid()
    u = float(hash32(wp.uint32(seed[e]) ^ salt) & wp.uint32(0xFFFF)) / 65536.0
    if running[e] == 1 and step[e] < tol * (1.0 + spread * u):
        running[e] = 0


@dataclasses.dataclass(frozen=True, kw_only=True)
class SmoothParams(plugin.Params):
    iters: int = plugin.static(64, lo=1, doc="iteration bound")
    first_check: int = plugin.static(8, lo=0, doc="the first iteration that may be skipped")
    check_every: int = plugin.static(8, lo=1, doc="iterations per window")
    lam: float = plugin.static(0.3, lo=0.0, hi=0.5, doc="Laplacian step")
    tol: float = plugin.static(2.5e-3, lo=0.0, doc="convergence tolerance (largest step)")
    tol_spread: float = plugin.static(1.0, lo=0.0, doc="per-env tolerance spread")


class ConvergingSmooth(plugin.Stage):
    """``toy.converging_smooth``: bounded Laplacian smoothing, absorbing per-env freeze."""

    kind = "relax"
    name = CONVERGING
    capture = plugin.CaptureClass.BOUNDED
    Params = SmoothParams
    reads = {"xy": "cl.xy", "count": "cl.count"}
    writes = {"xy": "cl.xy"}                     # a rewrite, through one port

    gated = 1                                    # the smoothing honours `running`

    def _late(self, ctx, it, iters, active, running) -> None:
        """Hook of the variants (extra work after each iteration's launches)."""

    def build(self, b, p):
        x_in, count = b.read("xy"), b.read("count")
        out = b.write("xy", wp.vec2f, per="cl")
        tmp = b.temp_like("xy")
        E, S = b.E, b.stride_of("xy")
        running = b.scratch("running", wp.int32, E)
        step = b.scratch("step", wp.float32, E)
        active = b.scratch("active", wp.int32, 1)
        iters, first, every = int(p.iters), int(p.first_check), int(p.check_every)
        lam, tol, spread = float(p.lam), float(p.tol), float(p.tol_spread)
        salt, seed, gated = b.salt("tol"), b.seed, int(self.gated)

        def body(ctx, it):
            ctx.fill(step, 0.0)
            ctx.launch(_smooth_k, (E, S), [out, count, running, gated, lam, S, tmp, step])
            ctx.launch(_commit_k, (E, S), [tmp, count, running, gated, S, out])
            if (it + 1) % every == 0:
                ctx.launch(_converge_k, E, [step, seed, salt, tol, spread, running])
                ops.any_set(ctx, running, active)
            self._late(ctx, it, iters, active, running)

        def enqueue(ctx):
            ctx.copy(out, x_in)                  # padding carried (NaN stays NaN)
            ctx.fill(running, 1)
            ctx.bounded_loop(iters, body, active=active, first_check=first,
                             check_every=every)
        return enqueue


class LeakySmooth(ConvergingSmooth):
    """``toy.leaky_smooth``: a converged env keeps smoothing (NOT absorbing)."""

    name = LEAKY
    gated = 0


# ---------------------------------------------------------------------------------------
# HOST: generators drawn on the host
# ---------------------------------------------------------------------------------------

def _host_loop(seed: int, points: int, scale: float) -> np.ndarray:
    """``[points, 2]`` float32: one env's closed loop from numpy's host RNG."""
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    radius = _RADIUS * scale * rng.uniform(0.9, 1.1)
    a2, a3 = rng.uniform(0.0, 0.08, 2)
    p2, p3 = rng.uniform(0.0, 2.0 * np.pi, 2)
    t = 2.0 * np.pi * np.arange(points) / points
    r = radius * (1.0 + a2 * np.cos(2.0 * t + p2) + a3 * np.cos(3.0 * t + p3))
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=-1).astype(np.float32)


@dataclasses.dataclass(frozen=True, kw_only=True)
class HostCircleParams(plugin.Params):
    points: int = plugin.static(64, lo=3, doc="raw points per env")


class HostCircle(plugin.Stage):
    """``toy.host_circle``: a HOST track generator (host RNG, uploaded through
    ``ctx.host()``)."""

    kind = "generator"
    name = HOST_CIRCLE
    output = "centerline"
    capture = plugin.CaptureClass.HOST
    Params = HostCircleParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}

    def capacity(self, params, layout):
        return {"raw": int(params.points)}

    def build(self, b, p):
        P, E, scale = int(p.points), b.E, float(b.scale)
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        seed_h = b.host_buffer(wp.int32, E)
        xy_h = b.host_buffer(wp.vec2f, E * P)
        seed = b.seed

        def enqueue(ctx):
            h = ctx.host()
            seeds = h.readback(seed, via=seed_h)
            pts = np.concatenate([_host_loop(int(s), P, scale) for s in seeds])
            h.upload(xy, pts, via=xy_h)
            ctx.fill(valid, 1)
        return enqueue


@dataclasses.dataclass(frozen=True, kw_only=True)
class HostRingParams(plugin.Params):
    gates: int = plugin.static(8, lo=2, doc="anchors per env")


class HostRing(plugin.Stage):
    """``toy.host_ring``: a HOST gate generator (anchors drawn on the host)."""

    kind = "gate_generator"
    name = HOST_RING
    output = "anchors"
    capture = plugin.CaptureClass.HOST
    Params = HostRingParams
    writes = {"xy": "anchors.xy", "count": "anchors.count"}

    def capacity(self, params, layout):
        return {"gates": int(params.gates), "min_gates": int(params.gates)}

    def build(self, b, p):
        G, E, scale = int(p.gates), b.E, float(b.scale)
        xy = b.write("xy", wp.vec2f, per="anchors")
        count = b.write("count", wp.int32)
        seed_h = b.host_buffer(wp.int32, E)
        xy_h = b.host_buffer(wp.vec2f, E * G)
        count_h = b.host_buffer(wp.int32, E)
        seed = b.seed

        def enqueue(ctx):
            h = ctx.host()
            seeds = h.readback(seed, via=seed_h)
            rows = []
            for s in seeds:
                rng = np.random.default_rng(int(s) & 0x7FFFFFFF)
                t = 2.0 * np.pi * (np.arange(G) + rng.uniform(-0.15, 0.15, G)) / G
                r = _RADIUS * scale * rng.uniform(0.9, 1.1, G)
                rows.append(np.stack([r * np.cos(t), r * np.sin(t)], axis=-1))
            h.upload(xy, np.concatenate(rows).astype(np.float32), via=xy_h)
            h.upload(count, np.full(E, G, np.int32), via=count_h)
        return enqueue


def register_bounded_toys(reg) -> None:
    """Register the four shared toys into ``reg`` (a registry, or the test's scope)."""
    for cls in (ConvergingSmooth, LeakySmooth, HostCircle, HostRing):
        reg.add(cls)


# ---------------------------------------------------------------------------------------
# Broken / extra variants (CPU tests)
# ---------------------------------------------------------------------------------------

class UndeclaredLoop(ConvergingSmooth):
    """``toy.undeclared_loop``: the converging smoother declared STATIC (refused at its
    first run: a loop with ``active=`` needs ``capture = CaptureClass.BOUNDED``)."""

    name = UNDECLARED
    capture = plugin.CaptureClass.STATIC


class SyncingSmooth(ConvergingSmooth):
    """``toy.syncing_smooth``: its body reads ``active`` back (a plugin's own host sync,
    caught although the library's poll is sanctioned)."""

    name = SYNCING

    def _late(self, ctx, it, iters, active, running) -> None:
        if (it + 1) % 8 == 0:
            int(active.numpy()[0])


class LateAllocSmooth(ConvergingSmooth):
    """``toy.late_alloc_smooth``: allocates in its last iteration only."""

    name = LATE_ALLOC

    def _late(self, ctx, it, iters, active, running) -> None:
        if it == iters - 1:
            wp.empty(1, dtype=wp.int32, device=ctx.device)


class LateSyncSmooth(ConvergingSmooth):
    """``toy.late_sync_smooth``: reads ``active`` back in its last iteration only."""

    name = LATE_SYNC

    def _late(self, ctx, it, iters, active, running) -> None:
        if it == iters - 1:
            int(active.numpy()[0])


class PollCaller(plugin.Stage):
    """``toy.poll_caller``: a STATIC relax (a copy) whose enqueue calls the library's
    private host poll itself, outside any loop's window driver: not sanctioned."""

    kind = "relax"
    name = POLL_CALLER
    reads = {"xy": "cl.xy"}
    writes = {"xy": "cl.xy"}

    def build(self, b, p):
        from track_gen.exec.host import _poll_i32

        x_in = b.read("xy")
        out = b.write("xy", wp.vec2f, per="cl")
        flag = b.scratch("flag", wp.int32, 1)

        def enqueue(ctx):
            ctx.copy(out, x_in)
            ctx.fill(flag, 1)
            _poll_i32(ctx, flag)
        return enqueue


class HostIdentity(plugin.Stage):
    """``toy.host_identity``: a HOST relax that reads ``cl.xy`` back and uploads it
    unchanged (``ctx.host()``)."""

    kind = "relax"
    name = HOST_IDENTITY
    capture = plugin.CaptureClass.HOST
    reads = {"xy": "cl.xy"}
    writes = {"xy": "cl.xy"}

    def build(self, b, p):
        x_in = b.read("xy")
        out = b.write("xy", wp.vec2f, per="cl")
        buf = b.host_buffer(wp.vec2f, int(x_in.shape[0]))

        def enqueue(ctx):
            h = ctx.host()
            pts = np.array(h.readback(x_in, via=buf), copy=True)
            h.upload(out, pts, via=buf)
        return enqueue


class SneakyHost(plugin.Stage):
    """``toy.sneaky_host``: a HOST relax that reaches the host OUTSIDE ``ctx.host()``: a
    direct ``.numpy()`` readback, a device-wide ``wp.synchronize_device`` and a per-call
    ``wp.array`` upload (the kit fails it; HOST waives none of that)."""

    kind = "relax"
    name = SNEAKY_HOST
    capture = plugin.CaptureClass.HOST
    reads = {"xy": "cl.xy"}
    writes = {"xy": "cl.xy"}

    def build(self, b, p):
        x_in = b.read("xy")
        out = b.write("xy", wp.vec2f, per="cl")

        def enqueue(ctx):
            pts = x_in.numpy()
            wp.synchronize_device(ctx.device)
            tmp = wp.array(pts, dtype=wp.vec2f, device=ctx.device)
            ctx.copy(out, tmp)
        return enqueue


def register_broken_bounded_toys(reg) -> None:
    for cls in (UndeclaredLoop, SyncingSmooth, LateAllocSmooth, LateSyncSmooth,
                PollCaller, HostIdentity, SneakyHost):
        reg.add(cls)
