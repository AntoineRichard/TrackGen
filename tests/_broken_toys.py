"""Deliberately broken generator plugins: each breaks ONE rule of the generator contract.

``tests/test_conformance.py`` registers them (``register_broken``) and asserts that
``track_gen.testing.check_generator`` rejects each with exactly the check it breaks:

=========================  ====================================  ========================
name                       defect                                failing check
=========================  ====================================  ========================
``broken.host_sync``       reads the seeds back (``.numpy()``)   ``hot_path.host_sync``
``broken.alloc``           allocates a staging buffer per call   ``hot_path.allocation``
``broken.nan``             writes NaN into valid envs            ``output.finite``
``broken.topology``        launch dims follow a call counter     ``hot_path.topology``
``broken.batch``           env e reads env e+1's seed            ``batch_independence``
``broken.legacy_stream``   ``wp.launch`` without ``stream=``     ``hot_path.stream``
``broken.stateful``        a device counter changes every call   ``statelessness``
``broken.constant``        ignores the seeds                     ``diversity``
``broken.open``            writes half a circle                  ``output.closed``
``broken.global_buffer``   stages through a module-level buffer  ``hot_path.buffers``
``broken.shared_scratch``  its scratch holds a buffer shared by  ``hot_path.buffers``
                           every instance
``broken.unwritten``       some envs never write their last      ``output.finite``
                           point
``broken.no_valid``        never writes ``out_valid``            ``output.shape``
``broken.tiled``           ``wp.launch_tiled`` without           ``hot_path.stream``
                           ``stream=``
``broken.assign``          ``out_centerline.assign(staging)``    ``hot_path.stream``
=========================  ====================================  ========================

The first five are the ones the conformance spec names (:data:`FIVE`). Apart from its
defect each one is :mod:`tests._plugin_toys`' ``toy.circle`` (a jittered circle of
``num_points`` points per env, keyed by the env's seed), so every other check passes.

:data:`EDGE` holds toys that look unusual but honour the contract, so every check must
pass: ``edge.stadium`` (a loop closed by one of its two long straights),
``edge.rejecting`` (rejects the odd-seeded envs by writing NaN and ``out_valid = 0``)
and ``edge.struct_scratch`` (its scratch is a per-instance ``wp.struct``, launched as a
struct argument).
"""
from __future__ import annotations

import itertools

import warp as wp

from track_gen.exec import LaunchContext, resolve
from track_gen.plugin import GeneratorSpec

_RADIUS = 0.6      # in units of config.scale, as toy.circle


@wp.func
def _circle_point(seed: wp.int32, i: int, n: int, radius: float):
    state = wp.rand_init(seed)
    r = radius * (0.9 + 0.2 * wp.randf(state))
    t = 2.0 * wp.pi * float(i) / float(n)
    return wp.vec2f(r * wp.cos(t), r * wp.sin(t))


@wp.kernel
def _circle_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
              out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    out[e * n + i] = _circle_point(seeds[e], i, n, radius)
    if i == 0:
        valid[e] = 1


@wp.kernel
def _circle_flat_k(seeds: wp.array(dtype=wp.int32), n: int, total: int, radius: float,
                   out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    t = wp.tid()
    if t >= total:          # tolerates a launch larger than the batch
        return
    e = t // n
    i = t - e * n
    out[t] = _circle_point(seeds[e], i, n, radius)
    if i == 0:
        valid[e] = 1


@wp.kernel
def _circle_nan_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
                  out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    p = _circle_point(seeds[e], i, n, radius)
    if seeds[e] % 4 == 3 and i == n // 2:
        p = wp.vec2f(wp.nan, wp.nan)          # the defect: NaN in an env marked valid
    out[e * n + i] = p
    if i == 0:
        valid[e] = 1


@wp.kernel
def _circle_neighbour_k(seeds: wp.array(dtype=wp.int32), n: int, num_envs: int,
                        radius: float, out: wp.array(dtype=wp.vec2f),
                        valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    # the defect: the radius jitter is drawn from the NEXT env's seed
    out[e * n + i] = _circle_point(seeds[(e + 1) % num_envs], i, n, radius)
    if i == 0:
        valid[e] = 1


@wp.kernel
def _circle_offset_k(seeds: wp.array(dtype=wp.int32), offset: wp.array(dtype=wp.int32),
                     n: int, radius: float, out: wp.array(dtype=wp.vec2f),
                     valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    # the defect: the draw also depends on a counter the generator keeps on device
    out[e * n + i] = _circle_point(seeds[e] + offset[0], i, n, radius)
    if i == 0:
        valid[e] = 1


@wp.kernel
def _bump_k(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def _circle_constant_k(n: int, radius: float, out: wp.array(dtype=wp.vec2f),
                       valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    t = 2.0 * wp.pi * float(i) / float(n)        # the defect: no seed at all
    out[e * n + i] = wp.vec2f(radius * wp.cos(t), radius * wp.sin(t))
    if i == 0:
        valid[e] = 1


@wp.kernel
def _half_circle_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
                   out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    state = wp.rand_init(seeds[e])
    r = radius * (0.9 + 0.2 * wp.randf(state))
    t = wp.pi * float(i) / float(n - 1)          # the defect: only half a turn
    out[e * n + i] = wp.vec2f(r * wp.cos(t), r * wp.sin(t))
    if i == 0:
        valid[e] = 1


@wp.kernel
def _circle_short_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
                    out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    # the defect: an env with an odd seed never writes its last point
    if not (seeds[e] % 2 == 1 and i == n - 1):
        out[e * n + i] = _circle_point(seeds[e], i, n, radius)
    if i == 0:
        valid[e] = 1


@wp.kernel
def _circle_points_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
                     out: wp.array(dtype=wp.vec2f)):
    e, i = wp.tid()
    out[e * n + i] = _circle_point(seeds[e], i, n, radius)   # the defect: no out_valid


@wp.kernel
def _circle_tiled_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
                    out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    t, _lane = wp.tid()
    e = t // n
    i = t - e * n
    out[t] = _circle_point(seeds[e], i, n, radius)
    if i == 0:
        valid[e] = 1


@wp.kernel
def _stadium_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
               out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    # Two semicircles of n/2 points each, joined by two straights: edge (h-1 -> h) and
    # the closing edge (n-1 -> 0), both far longer than the arc steps.
    e, i = wp.tid()
    state = wp.rand_init(seeds[e])
    r = radius * (0.45 + 0.1 * wp.randf(state))
    half = radius * 0.8
    h = n / 2
    if i < h:
        t = -0.5 * wp.pi + wp.pi * float(i) / float(h - 1)
        out[e * n + i] = wp.vec2f(half + r * wp.cos(t), r * wp.sin(t))
    else:
        t = 0.5 * wp.pi + wp.pi * float(i - h) / float(h - 1)
        out[e * n + i] = wp.vec2f(-half + r * wp.cos(t), r * wp.sin(t))
    if i == 0:
        valid[e] = 1


@wp.kernel
def _circle_rejecting_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
                        out: wp.array(dtype=wp.vec2f), valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    rejected = seeds[e] % 2 == 1
    if rejected:
        out[e * n + i] = wp.vec2f(wp.nan, wp.nan)
    else:
        out[e * n + i] = _circle_point(seeds[e], i, n, radius)
    if i == 0:
        if rejected:
            valid[e] = 0
        else:
            valid[e] = 1


@wp.struct
class _Staging:
    xy: wp.array(dtype=wp.vec2f)


@wp.kernel
def _circle_struct_k(seeds: wp.array(dtype=wp.int32), n: int, radius: float,
                     staging: _Staging, valid: wp.array(dtype=wp.int32)):
    e, i = wp.tid()
    staging.xy[e * n + i] = _circle_point(seeds[e], i, n, radius)
    if i == 0:
        valid[e] = 1


def _alloc_none(config):
    return None


def _shape(config) -> "tuple[int, int, float]":
    return int(config.num_envs), int(config.num_points), _RADIUS * float(config.scale)


def host_sync_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                       ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    if int(seeds_wp.numpy().min()) < -2**31:      # the defect: a host readback
        raise AssertionError("unreachable")
    ctx.launch(_circle_k, (E, N), [seeds_wp, N, radius, out_centerline, out_valid_wp])


def alloc_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                   ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    staging = wp.empty(E * N, dtype=wp.vec2f, device=str(config.device))  # the defect
    ctx.launch(_circle_k, (E, N), [seeds_wp, N, radius, staging, out_valid_wp])
    ctx.copy(out_centerline, staging)


def nan_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                 ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_nan_k, (E, N), [seeds_wp, N, radius, out_centerline, out_valid_wp])


class _Counter:
    """Per-generator call counter (held in the scratch the spec allocates)."""

    def __init__(self) -> None:
        self.calls = itertools.count()


def topology_alloc(config):
    return _Counter()


def topology_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                      ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    extra = next(scratch.calls) % 2               # the defect: dims follow a counter
    ctx.launch(_circle_flat_k, E * N + extra,
               [seeds_wp, N, E * N, radius, out_centerline, out_valid_wp])


def batch_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                   ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_neighbour_k, (E, N),
               [seeds_wp, N, E, radius, out_centerline, out_valid_wp])


def legacy_stream_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch) -> None:
    E, N, radius = _shape(config)
    # the defect: no ctx, no stream= (a legacy accepts_ctx=False plugin)
    wp.launch(_circle_k, dim=(E, N),
              inputs=[seeds_wp, N, radius, out_centerline, out_valid_wp],
              device=str(config.device))


def stateful_alloc(config):
    return wp.zeros(1, dtype=wp.int32, device=str(config.device))


def stateful_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                      ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_offset_k, (E, N),
               [seeds_wp, scratch, N, radius, out_centerline, out_valid_wp])
    ctx.launch(_bump_k, 1, [scratch])            # hidden state, carried to the next call


def constant_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                      ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_constant_k, (E, N), [N, radius, out_centerline, out_valid_wp])


def open_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                  ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_half_circle_k, (E, N), [seeds_wp, N, radius, out_centerline, out_valid_wp])


# the defect: staging buffers shared by every generator instance (allocated at
# construction, so no allocation happens on the hot path)
_SHARED: dict = {}


def global_buffer_alloc(config):
    key = (str(config.device), int(config.num_envs) * int(config.num_points))
    if key not in _SHARED:
        _SHARED[key] = wp.empty(key[1], dtype=wp.vec2f, device=key[0])
    return None


def global_buffer_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                           ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    staging = _SHARED[(str(config.device), E * N)]
    ctx.launch(_circle_k, (E, N), [seeds_wp, N, radius, staging, out_valid_wp])
    ctx.copy(out_centerline, staging)


class _SharedStaging:
    """A per-generator scratch object that holds a buffer every instance shares."""

    def __init__(self, buf) -> None:
        self.buf = buf


def shared_scratch_alloc(config):
    key = ("scratch", str(config.device), int(config.num_envs) * int(config.num_points))
    if key not in _SHARED:
        _SHARED[key] = wp.empty(key[2], dtype=wp.vec2f, device=key[1])
    return _SharedStaging(_SHARED[key])          # the defect: the buffer is shared


def shared_scratch_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                            ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_k, (E, N), [seeds_wp, N, radius, scratch.buf, out_valid_wp])
    ctx.copy(out_centerline, scratch.buf)


def unwritten_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                       ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_short_k, (E, N), [seeds_wp, N, radius, out_centerline, out_valid_wp])


def no_valid_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                      ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_points_k, (E, N), [seeds_wp, N, radius, out_centerline])


def tiled_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                   ctx: "LaunchContext | None" = None) -> None:
    E, N, radius = _shape(config)
    # the defect: a tiled launch without stream= (it lands on Warp's current stream)
    wp.launch_tiled(_circle_tiled_k, dim=[E * N], block_dim=1,
                    inputs=[seeds_wp, N, radius, out_centerline, out_valid_wp],
                    device=str(config.device))


def staging_alloc(config):
    return wp.empty(int(config.num_envs) * int(config.num_points), dtype=wp.vec2f,
                    device=str(config.device))


def assign_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                    ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_k, (E, N), [seeds_wp, N, radius, scratch, out_valid_wp])
    out_centerline.assign(scratch)        # the defect: a copy on Warp's current stream


def stadium_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                     ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_stadium_k, (E, N), [seeds_wp, N, radius, out_centerline, out_valid_wp])


def rejecting_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                       ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_rejecting_k, (E, N),
               [seeds_wp, N, radius, out_centerline, out_valid_wp])


def struct_scratch_alloc(config):
    staging = _Staging()
    staging.xy = staging_alloc(config)
    return staging


def struct_scratch_generate(seeds_wp, config, out_centerline, out_valid_wp, scratch,
                            ctx: "LaunchContext | None" = None) -> None:
    ctx = resolve(ctx, str(config.device))
    E, N, radius = _shape(config)
    ctx.launch(_circle_struct_k, (E, N), [seeds_wp, N, radius, scratch, out_valid_wp])
    ctx.copy(out_centerline, scratch.xy)


BROKEN = {
    "broken.host_sync": (GeneratorSpec(name="broken.host_sync", alloc_scratch=_alloc_none,
                                       generate=host_sync_generate, accepts_ctx=True),
                         "hot_path.host_sync"),
    "broken.alloc": (GeneratorSpec(name="broken.alloc", alloc_scratch=_alloc_none,
                                   generate=alloc_generate, accepts_ctx=True),
                     "hot_path.allocation"),
    "broken.nan": (GeneratorSpec(name="broken.nan", alloc_scratch=_alloc_none,
                                 generate=nan_generate, accepts_ctx=True),
                   "output.finite"),
    "broken.topology": (GeneratorSpec(name="broken.topology", alloc_scratch=topology_alloc,
                                      generate=topology_generate, accepts_ctx=True),
                        "hot_path.topology"),
    "broken.batch": (GeneratorSpec(name="broken.batch", alloc_scratch=_alloc_none,
                                   generate=batch_generate, accepts_ctx=True),
                     "batch_independence"),
    "broken.legacy_stream": (GeneratorSpec(name="broken.legacy_stream",
                                           alloc_scratch=_alloc_none,
                                           generate=legacy_stream_generate),
                             "hot_path.stream"),
    "broken.stateful": (GeneratorSpec(name="broken.stateful", alloc_scratch=stateful_alloc,
                                      generate=stateful_generate, accepts_ctx=True),
                        "statelessness"),
    "broken.constant": (GeneratorSpec(name="broken.constant", alloc_scratch=_alloc_none,
                                      generate=constant_generate, accepts_ctx=True),
                        "diversity"),
    "broken.open": (GeneratorSpec(name="broken.open", alloc_scratch=_alloc_none,
                                  generate=open_generate, accepts_ctx=True),
                    "output.closed"),
    "broken.global_buffer": (GeneratorSpec(name="broken.global_buffer",
                                           alloc_scratch=global_buffer_alloc,
                                           generate=global_buffer_generate,
                                           accepts_ctx=True),
                             "hot_path.buffers"),
    "broken.shared_scratch": (GeneratorSpec(name="broken.shared_scratch",
                                            alloc_scratch=shared_scratch_alloc,
                                            generate=shared_scratch_generate,
                                            accepts_ctx=True),
                              "hot_path.buffers"),
    "broken.unwritten": (GeneratorSpec(name="broken.unwritten", alloc_scratch=_alloc_none,
                                       generate=unwritten_generate, accepts_ctx=True),
                         "output.finite"),
    "broken.no_valid": (GeneratorSpec(name="broken.no_valid", alloc_scratch=_alloc_none,
                                      generate=no_valid_generate, accepts_ctx=True),
                        "output.shape"),
    "broken.tiled": (GeneratorSpec(name="broken.tiled", alloc_scratch=_alloc_none,
                                   generate=tiled_generate, accepts_ctx=True),
                     "hot_path.stream"),
    "broken.assign": (GeneratorSpec(name="broken.assign", alloc_scratch=staging_alloc,
                                    generate=assign_generate, accepts_ctx=True),
                      "hot_path.stream"),
}

#: Toys that look unusual but honour the contract: every check passes.
EDGE = {
    "edge.stadium": GeneratorSpec(name="edge.stadium", alloc_scratch=_alloc_none,
                                  generate=stadium_generate, accepts_ctx=True),
    "edge.rejecting": GeneratorSpec(name="edge.rejecting", alloc_scratch=_alloc_none,
                                    generate=rejecting_generate, accepts_ctx=True),
    "edge.struct_scratch": GeneratorSpec(name="edge.struct_scratch",
                                         alloc_scratch=struct_scratch_alloc,
                                         generate=struct_scratch_generate,
                                         accepts_ctx=True),
}

#: The five toys the conformance spec names (the others are extras).
FIVE = ("broken.host_sync", "broken.alloc", "broken.nan", "broken.topology",
        "broken.batch")


def register_broken(reg) -> None:
    """Register every broken toy and every edge toy (a plugin ``register(reg)``)."""
    for spec, _ in BROKEN.values():
        reg.add(spec)
    for spec in EDGE.values():
        reg.add(spec)
