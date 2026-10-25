"""No host sync and no host readback on any hot path (CPU probe).

Every hot-path entry point must be recordable into a caller's CUDA graph, so none of
them may host-synchronize or read device data back to the host. That covers the
pipeline wrappers, every registered generator's ``_run``, the facades'
``generate()``/``enqueue()`` (plain and ``advance=True, mask=...``), every utility's
query/update/reset/sample/bake/refresh, ``Course.step``/``reset``/``regenerate`` (plain,
``advance``, masked)/``generate``/``enqueue_generate``, ``SeedState``'s device ops and the
rng's ``set_seeds_warp`` and ``sample_*_warp(out=...)``. The CPU probe uses :func:`_probe` (the library guard
:func:`track_gen.testing.contract.no_host_sync`) replaces ``wp.synchronize*`` and
``wp.array.numpy`` / ``list`` with tripwires that record the call and raise. Each driver
then runs its entry point under the probe, and the test asserts zero calls. Construction
(allocation, readback-based sizing) runs before the probe and is out of scope.

``test_every_wrapper_is_driven`` traces the drivers and fails when a module-level
function of the pipeline modules is reached by none of them. A new wrapper therefore
needs a driver here before it can land.

The facades' ``generate(sync=True)`` blocks on its own stream outside a capture through
``track_gen.exec._compat.stream_synchronize`` (a no-op on CPU and while capturing), never
through ``wp.synchronize*``; on CPU every generation path is probed.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import warnings

import numpy as np
import pytest
import warp as wp

from track_gen._src import (
    localize,
    rng_kernels,
    runtime,
    warp_gate,
    warp_generate,
    warp_generate_checkpoint,
    warp_generate_checkpoint_gates,
    warp_generate_gates,
    warp_generate_hull,
    warp_generate_polar,
    warp_generate_polar_gates,
    warp_generate_repulsive,
    warp_generate_voronoi,
    warp_generate_voronoi_gates,
    warp_pipeline,
    warp_relax,
    warp_width,
    warp_zprofile,
)
from track_gen._src.checkpoints import CheckpointSampler, CheckpointSet
from track_gen._src.collision import CollisionChecker
from track_gen._src.collision_discs import DiscChecker
from track_gen._src.collision_frames import FrameChecker
from track_gen._src.course import Course, CourseConfig
from track_gen._src.course_line import CourseLine
from track_gen._src.gate_generator import GateGenerator
from track_gen._src.heightfield import HeightFieldBaker
from track_gen._src.localize import TrackLocalizer
from track_gen._src.progress import ProgressTracker
from track_gen._src.props import PropSampler
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.track_generator import TrackGenerator
from track_gen._src.types import GateGenConfig, TrackGenConfig
from track_gen import ops
from track_gen import pipeline as pipeline_mod
from track_gen import seeds as seeds_mod
from track_gen import stages as stages_mod
from track_gen.exec import LaunchContext
from track_gen.ops import reduce as reduce_mod
from track_gen.plugin import registry
from track_gen.seeds import SeedState
from track_gen.testing.contract import HotPathGuard, no_host_sync

DEV = "cpu"
E = 2

# Known hot-path readbacks, as (api, module, function) -> the PR that removes them.
# The probe lets these through instead of raising. The test also fails if one is no
# longer hit, so the entry is removed together with the readback it allows.
# Host readbacks that hot paths are still allowed to perform. Empty: repulsive's early-exit
# readback (`if int(s.frozen.numpy().sum()) >= E: break`) was removed by the quick-wins
# absorbing freeze, so every generator hot path is now readback-free.
_KNOWN_READBACKS: set = set()


def _probe() -> HotPathGuard:
    """Tripwire every host sync and host readback for the duration of the block.

    The library guard (:func:`track_gen.testing.contract.no_host_sync`): any call is
    recorded and raises, a call swallowed by a broad ``except`` still fails the block on
    exit, and the debug sync (``TRACKGEN_DEBUG_SYNC``) is off. The known readbacks in
    :data:`_KNOWN_READBACKS` are let through and listed in the guard's ``allowed``.
    """
    return no_host_sync(allow=_KNOWN_READBACKS)


# ---------------------------------------------------------------------------------------
# Drivers: each setup runs OUTSIDE the probe and returns the hot-path callable to probe.
# ---------------------------------------------------------------------------------------

def _rng(seed=3):
    return PerEnvSeededRNG(seeds=seed, num_envs=E, device=DEV)


def _track_gen(generator="bezier", **kw):
    gen = TrackGenerator(TrackGenConfig(generator=generator, device=DEV, num_envs=E, **kw),
                         _rng())
    gen.generate()
    return gen


def _gate_gen(generator="bezier", **kw):
    kw.setdefault("gate_width", 0.2)
    gen = GateGenerator(GateGenConfig(generator=generator, device=DEV, num_envs=E, **kw),
                        _rng())
    gen.generate()
    return gen


def _vec3(n, value=(0.0, 0.0, 0.0)):
    return wp.array(np.tile(np.asarray(value, np.float32), (n, 1)), dtype=wp.vec3f,
                    device=DEV)


def _boxes(n):
    ori = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
                   dtype=wp.quatf, device=DEV)
    he = wp.array(np.full((n, 2), 0.02, np.float32), dtype=wp.vec2f, device=DEV)
    return _vec3(n, (0.1, 0.0, 0.0)), ori, he


def _all_mask():
    return wp.full(E, 1, dtype=wp.int32, device=DEV)


def _run_track(generator, **kw):
    def setup():
        return _track_gen(generator, **kw)._run
    return setup


def _run_gates(generator, **kw):
    def setup():
        return _gate_gen(generator, **kw)._run
    return setup


def _run_toy_plugin(kind):
    """An out-of-tree plugin (tests._plugin_toys, with explicit ``ctx``) registered
    through the public registry: its hot path is held to the same rules. The registry is
    restored once the generator has resolved its spec."""
    def setup():
        from tests._plugin_toys import register_toys

        with registry.scoped():
            register_toys(registry)
            if kind == "track":
                return _track_gen("toy.circle")._run
            return _gate_gen("toy.ring", gate_ordering="raw")._run
    return setup


def _run_bounded_toy():
    """A BOUNDED relax plugin (tests._bounded_toys, ``ctx.bounded_loop``) in the default
    track pipeline: on CPU its loop's host poll is the one sanctioned readback (the guard
    lists it in ``sanctioned``, not ``allowed``); the plugin's own launches stay clean."""
    def setup():
        from tests._bounded_toys import CONVERGING, register_bounded_toys

        with registry.scoped():
            register_bounded_toys(registry)
            gen = TrackGenerator(pipeline_mod.presets.track(num_envs=E, device=DEV,
                                                            relax=CONVERGING), _rng())
        gen.generate()
        return gen._run
    return setup


def _collision(method):
    def setup():
        track = _track_gen().generate()
        checker = CollisionChecker(track, max_boxes=1, method=method, sdf_resolution=16)
        checker.bind_inputs(*_boxes(E))

        def hot():
            if method == "sdf":
                checker.bake()
            checker.query()
        return hot
    return setup


def _props():
    sampler = PropSampler(_track_gen().generate(), spacing=0.2)
    return sampler.sample


def _checkpoints():
    sampler = CheckpointSampler(_track_gen().generate(), spacing=0.3)
    return sampler.sample


def _progress():
    tracker = ProgressTracker(CheckpointSampler(_track_gen().generate(), 0.3).sample(),
                              position=_vec3(E))
    mask = _all_mask()

    def hot():
        tracker.update()
        tracker.reset(mask)
    return hot


def _localizer():
    loc = TrackLocalizer(_track_gen().generate(), warm_window=4, position=_vec3(E))
    mask = _all_mask()

    def hot():
        loc.query()
        loc.reset(mask)
    return hot


def _localize_profiles():
    # Module-level helpers. Without out= they allocate their outputs but must not sync
    # or read back; the out= forms (roadmap PR14) are also allocation-free (see
    # test_localize_profiles_out_forms_pass_the_full_guard).
    track = _track_gen().generate()
    n = int(track.center.shape[0])
    k, sc, v = (wp.empty(n, dtype=wp.float32, device=DEV) for _ in range(3))

    def hot():
        kappa = localize.curvature(track)
        localize.speed_profile(track, 4.0, 2.0, 4.0, 3.0, kappa=kappa)
        localize.curvature(track, out=k, scratch=sc)
        localize.speed_profile(track, 4.0, 2.0, 4.0, 3.0, kappa=k, out=v)
    return hot


def _minimal_centerline(src):
    """Only the ``HasCenterline`` members of ``src`` (no ``valid``, no ``layout``)."""
    import types

    return types.SimpleNamespace(center=src.center, arclen=src.arclen, length=src.length,
                                 count=src.count)


@pytest.mark.parametrize("source", ["track", "centerline", "minimal"])
def test_localize_profiles_out_forms_pass_the_full_guard(source):
    """Roadmap PR14: ``curvature(out=, scratch=)`` / ``speed_profile(kappa=, out=)``
    allocate nothing, sync nothing and launch on the call's stream (the full hot-path
    guard), on a ``Track``, a ``Centerline`` and a minimal centerline without ``valid``
    (its absent ``valid`` costs no allocation: the kernels get ``count`` as a dummy)."""
    from track_gen.testing import hot_path_guard

    if source == "centerline":
        line = CourseLine(_gate_gen().generate(), 4)
        line.refresh()
        src = line.centerline
    else:
        src = _track_gen().generate()
        if source == "minimal":
            src = _minimal_centerline(src)
    n = int(src.center.shape[0])
    k, sc, v = (wp.empty(n, dtype=wp.float32, device=DEV) for _ in range(3))
    ctx = LaunchContext.for_call(DEV)
    localize.curvature(src, 2, ctx, out=k, scratch=sc)            # warm (module loads)
    localize.speed_profile(src, 4.0, 2.0, 4.0, 3.0, kappa=k, ctx=ctx, out=v)
    with hot_path_guard() as guard:
        for window in (0, 2):
            assert localize.curvature(src, window, ctx, out=k, scratch=sc) is k
        assert localize.speed_profile(src, 4.0, 2.0, 4.0, 3.0, kappa=k, ctx=ctx,
                                      out=v) is v
    assert not guard.violations, guard.violations


def _heightfield():
    return HeightFieldBaker(_track_gen(z_profile="uniform").generate(), 16).bake


def _commit_envs_driver():
    # Roadmap PR14: the private helper the component kit's invalid.sentinel check uses
    # (fail envs of a committed result after the fact) launches only.
    from track_gen.stages.commit import _commit_envs

    track = _track_gen().generate().clone()
    seq = _gate_gen().generate().clone()
    fail = wp.array(np.arange(E, dtype=np.int32) % 2, dtype=wp.int32, device=DEV)
    ctx = LaunchContext.for_call(DEV)

    def hot():
        _commit_envs(track, fail, poison=False, ctx=ctx)
        _commit_envs(seq, fail, poison=True, ctx=ctx)
    return hot


def _course_line():
    return CourseLine(_gate_gen().generate(), 4).refresh


def _discs():
    seq = _gate_gen().generate()
    posts = wp.zeros(2 * int(seq.position.shape[0]), dtype=wp.vec3f, device=DEV)
    pos, ori, he = _boxes(E)
    checker = DiscChecker(posts, radius=0.05, max_boxes=1, num_envs=E,
                          position=pos, orientation=ori, half_extents=he)
    return checker.query


def _frames():
    seq = _gate_gen().generate()
    checker = FrameChecker(seq, num_envs=E, radius=0.05, frame_thickness=0.02,
                           frame_depth=0.02)
    tracker = ProgressTracker(CheckpointSet.from_gates(seq), position=_vec3(E))
    checker.bind_inputs(_vec3(E))
    checker.bind_window(tracker.next_checkpoint_state)
    return checker.query


def _facade_generate(cls, **kw):
    def setup():
        gen = _track_gen(**kw) if cls is TrackGenerator else _gate_gen(**kw)

        def hot():
            gen.generate()
            gen.generate(sync=False)
            gen.enqueue()
        return hot
    return setup


def _course_regenerate(mode, **kw):
    def setup():
        gen = (TrackGenConfig(device=DEV, num_envs=E) if mode == "track"
               else GateGenConfig(device=DEV, num_envs=E, gate_width=0.2))
        if mode == "track":
            kw.setdefault("checkpoint_spacing", 0.3)
        course = Course(CourseConfig(mode=mode, gen=gen, seeds=3, **kw))
        seeds = wp.array(np.arange(E, dtype=np.int32) + 17, dtype=wp.int32, device=DEV)

        def hot():
            course.regenerate()
            course.generate()
            course.generate(seeds=5)
            course.generate(seeds=seeds)
            course.enqueue_generate()
        return hot
    return setup


def _i32(values):
    return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=DEV)


def _facade_advance(cls, **kw):
    # generate()/enqueue() with an on-device epoch advance (all envs, masked, the
    # persistent inputs["regen_mask"]), plus the SeedState ops the facade exposes.
    def setup():
        gen = _track_gen(**kw) if cls is TrackGenerator else _gate_gen(**kw)
        m = _i32([1, 0])

        def hot():
            gen.generate(advance=True)
            gen.generate(advance=True, mask=m)
            gen.generate(advance=True, mask=gen.inputs["regen_mask"], sync=False)
            gen.enqueue(advance=True, mask=m)
            gen.enqueue(advance=True)
            gen.seeds.advance(m)
            gen.seeds.reset_epoch(m)
            gen.seeds.reset_epoch()
        return hot
    return setup


def _course_masked(mode, **kw):
    # Masked / advancing regeneration (any_set + skip_unless + masked resets), the
    # persistent regen_mask input, and a reseed (which zeroes the epochs).
    def setup():
        gen = (TrackGenConfig(device=DEV, num_envs=E) if mode == "track"
               else GateGenConfig(device=DEV, num_envs=E, gate_width=0.2))
        if mode == "track":
            kw.setdefault("checkpoint_spacing", 0.3)
        course = Course(CourseConfig(mode=mode, gen=gen, seeds=3, **kw))
        m = _i32([0, 1])
        zero = _i32([0, 0])

        def hot():
            course.regenerate(mask=m, advance=True)
            course.regenerate(mask=zero, advance=True)
            course.regenerate(mask=course.inputs["regen_mask"], advance=True, sync=False)
            course.regenerate(mask=m)
            course.regenerate(advance=True)
            course.enqueue_generate(mask=m, advance=True)
            course.generate(seeds=5)
        return hot
    return setup


def _seed_state():
    owned = SeedState(E, DEV, 7)
    bound = SeedState(E, DEV, _i32([11, 12]))
    copied = SeedState(E, DEV, _rng(), copy=True)
    new, sub, ids, m = _i32([3, 4]), _i32([9]), _i32([1]), _i32([1, 0])
    out = wp.empty(E, dtype=wp.int32, device=DEV)

    def hot():
        owned.set(new)
        owned.set(sub, ids)
        copied.set(new)
        for st in (owned, bound, copied):
            st.advance()
            st.advance(m)
            st.reset_epoch(m)
            st.effective(out)
            st.reset_epoch()
    return hot


def _rng_sample_out():
    # Every sampler (scalar and tensorized bounds, 1-D/2-D/3-D outputs, all and
    # partial ids) into preallocated out= buffers, plus both reseed forms.
    rng = _rng()
    ids = _i32([1])
    f32 = lambda n: wp.array(np.full(n, 0.5, np.float32), dtype=wp.float32, device=DEV)  # noqa: E731
    i32 = lambda n, v: wp.array(np.full(n, v, np.int32), dtype=wp.int32, device=DEV)  # noqa: E731
    calls = []
    for shape in ((1,), (3,), (2, 3)):
        for sel in (None, ids):
            n = E if sel is None else 1
            kshape = (n,) if shape[0] == 1 else (n,) + shape

            def buf(dtype, _k=kshape):
                return wp.empty(_k, dtype=dtype, device=DEV)
            calls += [
                (rng.sample_uniform_warp, (0.0, 1.0, shape), sel, buf(wp.float32)),
                (rng.sample_uniform_warp, (f32(n), f32(n), shape), sel, buf(wp.float32)),
                (rng.sample_sign_warp, ("int", shape), sel, buf(wp.int32)),
                (rng.sample_sign_warp, ("float", shape), sel, buf(wp.float32)),
                (rng.sample_integer_warp, (0, 5, shape), sel, buf(wp.int32)),
                (rng.sample_integer_warp, (i32(n, 0), i32(n, 5), shape), sel, buf(wp.int32)),
                (rng.sample_normal_warp, (0.0, 1.0, shape), sel, buf(wp.float32)),
                (rng.sample_normal_warp, (f32(n), f32(n), shape), sel, buf(wp.float32)),
                (rng.sample_poisson_warp, (2.0, shape), sel, buf(wp.int32)),
                (rng.sample_poisson_warp, (f32(n), shape), sel, buf(wp.int32)),
                (rng.sample_quaternion_warp, (shape,), sel, buf(wp.quatf)),
            ]
    seeds, sub = _i32([5, 6]), _i32([8])

    def hot():
        rng.set_seeds_warp(seeds, None)
        rng.set_seeds_warp(sub, ids)
        for fn, args, sel, out in calls:
            assert fn(*args, ids=sel, out=out) is out
    return hot


def _course(mode, **kw):
    def setup():
        gen = (TrackGenConfig(device=DEV, num_envs=E) if mode == "track"
               else GateGenConfig(device=DEV, num_envs=E, gate_width=0.2))
        if mode == "track":
            kw.setdefault("checkpoint_spacing", 0.3)
        course = Course(CourseConfig(mode=mode, gen=gen, seeds=3, **kw))
        pos, ori, he = _boxes(E)
        if mode == "gates" and not kw.get("post_radius"):
            course.bind(position=pos)
        else:
            course.bind(position=pos, orientation=ori, half_extents=he)
        course.generate()
        mask = _all_mask()

        def hot():
            course.step()
            course.reset(mask)
        return hot
    return setup


# Pipeline wrappers that no generator _run reaches, driven directly on a real batch.

def _wrapper_inputs():
    gen = _track_gen()
    track, s = gen._track, gen._scratch
    n_max = int(gen._config.N_max)
    return gen, track, s, n_max




def _geometry_wrappers():
    gen, track, s, n_max = _wrapper_inputs()
    flat = E * n_max
    center2 = s.cs_center
    T = wp.empty(flat, dtype=wp.vec2f, device=DEV)
    Nrm = wp.empty(flat, dtype=wp.vec2f, device=DEV)
    kappa = wp.empty(flat, dtype=wp.float32, device=DEV)
    outer = wp.empty(flat, dtype=wp.vec2f, device=DEV)
    inner = wp.empty(flat, dtype=wp.vec2f, device=DEV)
    area_a = wp.empty(E, dtype=wp.float32, device=DEV)
    area_b = wp.empty(E, dtype=wp.float32, device=DEV)
    ctx = LaunchContext.for_call(DEV)

    def hot():
        warp_pipeline.frame_curvature(center2, T, Nrm, kappa, s.count, ctx=ctx)
        warp_pipeline.offset(center2, Nrm, 0.03, outer, inner, area_a, area_b, s.count, ctx=ctx)
    return hot


def _ops_all():
    # Every public track_gen.ops op and variant on a small synthetic batch (inputs and
    # ArcScratch allocated here, outside the probe).
    E, M, num, n, P, K, ps = 3, 12, 8, 10, 6, 5, 3
    rs = np.random.default_rng(0)
    t = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    dense = np.stack([np.cos(t), np.sin(t)], -1)[None] * rs.uniform(0.5, 2.0, (E, 1, 1))
    dense = dense.astype(np.float32)
    dense[1, 3:5] = np.nan
    v2 = lambda a: wp.array(np.asarray(a, np.float32).reshape(-1, 2), dtype=wp.vec2f,  # noqa: E731
                            device=DEV)
    src, loops = v2(dense), v2(dense[:, :n])
    ctx = LaunchContext.for_call(DEV)
    sc_m = ops.ArcScratch.alloc(E, M, DEV)
    sc_n = ops.ArcScratch.alloc(E, n, DEV, compact=False)
    act, cnt_n = _i32([1, 0, 1]), _i32([n, 4, 0])
    cnt_p, cnt_k, band = _i32([P, 2, P + 1]), _i32([K, 2, 1]), _i32([1, 1, 1])
    seeds = _i32([1, 2147483647, -7])
    pts, ctrl = v2(rs.normal(size=(E * P, 2))), v2(rs.normal(size=(E * K, 2)))
    scale = wp.array(np.asarray([1.0, 0.5, 2.0], np.float32), dtype=wp.float32, device=DEV)
    # roadmap PR13d: the per-env size of the grid / normalize ops (the sized twins)
    size = wp.array(np.asarray([1.0, 0.5, 20.0], np.float32), dtype=wp.float32, device=DEV)
    f32 = lambda k: wp.empty(k, dtype=wp.float32, device=DEV)  # noqa: E731
    i32 = lambda k: wp.empty(k, dtype=wp.int32, device=DEV)  # noqa: E731
    vv = lambda k: wp.empty(k, dtype=wp.vec2f, device=DEV)  # noqa: E731
    o_num, o_n, o_cs, o_p, o_p2, o_cr = vv(E * num), vv(E * n), vv(E * 16), vv(E * P), \
        vv(E * (P + 2)), vv(E * K * ps)
    co, cs_c, cs_t, xi, o_c, used, flag = i32(E), i32(E), i32(E), i32(E), i32(E), \
        i32(E * P), i32(1)
    tf, keys = f32(E), f32(E * (P + 2))
    # roadmap PR13a: the table-writing resamples and the channel carries
    tab_u = ops.ResampleTables.alloc(E * n, E * n, DEV)
    tab_c = ops.ResampleTables.alloc(E * 16, E * n, DEV)
    ch_f = wp.array(rs.normal(size=E * n).astype(np.float32), dtype=wp.float32, device=DEV)
    ch_i = wp.array(np.arange(E * n, dtype=np.int32), dtype=wp.int32, device=DEV)
    o_cf, o_ci, o_cv = f32(E * 16), i32(E * 16), vv(E * 16)
    o_ua, o_un = f32(E * n), f32(E * n)

    def hot():
        ops.arc_resample(ctx, src, M, num, sc_m, o_num, co)
        ops.arc_resample(ctx, src, M, num, sc_m, o_num, co, active=act)
        ops.resample_uniform(ctx, loops, n, sc_n, o_n, count=cnt_n)
        ops.resample_constant_spacing(ctx, loops, n, 0.3, 16, sc_n, o_cs, cs_c, cs_t)
        ops.self_intersections(ctx, loops, n, xi, count=cnt_n)
        ops.turning_number(ctx, loops, n, tf, count=cnt_n)
        ops.thickness(ctx, loops, n, band, tf, count=cnt_n)
        ops.normalize_extent(ctx, o_n, n, 1.44)
        ops.normalize_extent(ctx, o_n, n, 1.44, count=cnt_n)
        ops.normalize_extent(ctx, o_n, n, 1.44, size=size)
        ops.angle_sort(ctx, pts, P, keys, o_p, count=cnt_p)
        ops.angle_sort(ctx, pts, P, keys, o_p2, count=cnt_p, out_stride=P + 2)
        ops.sample_counts(ctx, seeds, 6151, 0, 2, P, o_c)
        nc, cell = ops.grid_cells(0.05)
        ops.sample_grid_points(ctx, seeds, 9781, 0, P, nc, cell, 1.0, used, o_p)
        ops.sample_grid_points(ctx, seeds, 7919, 1, P, nc, cell, scale, used, o_p)
        ops.sample_grid_points(ctx, seeds, 9781, 0, P, nc, cell, 1.0, used, o_p, size=size)
        ops.sample_grid_points(ctx, seeds, 7919, 1, P, nc, cell, scale, used, o_p,
                               size=size)
        ops.closed_catmull_rom(ctx, ctrl, K, ps, o_cr)
        ops.closed_catmull_rom(ctx, ctrl, K, ps, o_cr, count=cnt_k, endpoint=True)
        ops.closed_polyline(ctx, ctrl, K, ps, o_cr)
        ops.closed_polyline(ctx, ctrl, K, ps, o_cr, active=act)
        ops.closed_polyline(ctx, ctrl, K, ps, o_cr, count=cnt_k, endpoint=True, active=act)
        ops.fill(ctx, o_cr, (0.0, 1.0))
        ops.any_set(ctx, act, flag)
        ops.all_set(ctx, act, flag)
        ops.check_array("act", act, (E,), wp.int32, DEV)
        ops.resample_uniform(ctx, loops, n, sc_n, o_n, count=cnt_n, tables=tab_u)
        ops.resample_constant_spacing(ctx, loops, n, 0.3, 16, sc_n, o_cs, cs_c, cs_t,
                                      tables=tab_c)
        ops.resample_channels(ctx, tab_c, [(ch_f, o_cf), (ch_i, o_ci), (loops, o_cv)],
                              ["linear", "nearest", "previous"])
        ops.resample_channels(ctx, tab_u, [(ch_f, o_ua), (ch_f, o_un)], ["angle", "linear"])
    return hot









def _program_standalone():
    # A Program built straight from a PipelineSpec (track_gen.pipeline, PR10a): run()
    # (plain, advancing, masked) and enqueue() on an explicit context.
    spec = pipeline_mod.legacy.to_pipeline_spec(
        TrackGenConfig(device=DEV, num_envs=E, z_profile="noise", z_max=0.2,
                       z_noise_amplitude=0.1))
    program = pipeline_mod.build(spec, seeds=3)
    m = _i32([0, 1])
    ctx = LaunchContext.for_call(DEV)

    def hot():
        program.run()
        program.run(advance=True, mask=m, sync=True)
        program.run(policy="eager")
        program.enqueue(ctx)
    return hot


def _program_toys_poison():
    # The three toy pipeline-stage plugins (tests/_stage_toys.py: a relax rewrite, a
    # salted elevation, a fail-bit check) in one preset with on_invalid="poison", run
    # plainly (the commit derives valid and poisons) and masked (hold + commit).
    from tests._stage_toys import HILLS, KMAX, TAUBIN, register

    with registry.scoped():
        register(registry)
        spec = pipeline_mod.presets.track(
            num_envs=E, device=DEV, relax=TAUBIN, elevation=HILLS,
            checks=("default", (KMAX, {"kappa_max": 20.0})), on_invalid="poison")
        gen = TrackGenerator(spec, _rng())
    m = _i32([1, 0])

    def hot():
        gen.generate()
        gen.generate(advance=True, mask=m)
        gen.enqueue(advance=True, mask=gen.inputs["regen_mask"])
        gen.program.run(policy="eager")
    return hot


def _gate_program_toys_poison():
    # The gate toys (tests/_gate_toys.py: a provider, a third-party order on a plain
    # cloud, a gate validity check with a dynamic field) and a track elevation plugin
    # rewired onto gates, with on_invalid="poison", run plainly (the commit derives valid
    # and poisons) and masked (hold + commit) (roadmap PR11).
    from tests import _gate_toys as gt
    from tests._stage_toys import HILLS, register

    with registry.scoped():
        gt.register(registry)
        register(registry)
        helix = GateGenerator(pipeline_mod.presets.gates(
            num_envs=E, device=DEV, generator=gt.HELIX_POSED, on_invalid="poison"), _rng())
        scatter = GateGenerator(pipeline_mod.presets.gates(
            num_envs=E, device=DEV, generator=gt.SCATTER, order=gt.REVERSE,
            elevation=HILLS, checks=("gate_default", gt.MAX_HEIGHT),
            on_invalid="poison"), _rng())
    m = _i32([1, 0])

    def hot():
        for gen in (helix, scatter):
            gen.generate()
            gen.generate(advance=True, mask=m)
            gen.enqueue(advance=True, mask=gen.inputs["regen_mask"])
            gen.program.run(policy="eager")
    return hot


def _program_channels():
    # Roadmap PR13a: a generator's carried attributes (tests/_channel_toys.py ribbon: five
    # exported families + raw.acme.hw), width.from_channel on the carried cl.acme.hw (the
    # per-point XPBD / offset / validity kernels), on_invalid="poison": plain (the commit
    # derives valid and poisons, attributes included) and masked (hold + commit per
    # attribute); a generator that provides its half-widths and one that provides z.
    from tests._channel_toys import register

    with registry.scoped():
        register(registry)
        spec = pipeline_mod.presets.track(
            num_envs=E, device=DEV, generator="acme.ribbon",
            width=pipeline_mod.StageRef("width", "from_channel",
                                        ports={"src": "cl.acme.hw"}),
            on_invalid="poison")
        gens = [TrackGenerator(spec, _rng())] + [
            TrackGenerator(pipeline_mod.presets.track(num_envs=E, device=DEV,
                                                      generator=name), _rng())
            for name in ("acme.pinch", "acme.helix_loop")]
    m = _i32([1, 0])

    def hot():
        for gen in gens:
            gen.generate()
            gen.generate(advance=True, mask=m)
            gen.enqueue(advance=True, mask=gen.inputs["regen_mask"])
    return hot


def _gate_program_channels():
    # Roadmap PR13a: stations carrying a generator's attributes to the gates
    # (carry.attributes), xyz stations on a z-providing loop and on a relaxed centerline
    # with its elevation, an anchors generator's own attribute; masked and poison.
    from tests._channel_toys import register

    noise = ("noise", {"z_base": 0.5, "z_max": 1.5, "z_noise_amplitude": 0.5,
                       "z_noise_harmonics": 4})
    with registry.scoped():
        register(registry)
        specs = [
            pipeline_mod.presets.gates(num_envs=E, device=DEV, generator="acme.ribbon",
                                       gates=8, on_invalid="poison"),
            pipeline_mod.presets.gates(num_envs=E, device=DEV,
                                       generator="acme.helix_loop", gates=8,
                                       spacing_metric="xyz"),
            pipeline_mod.presets.gates(num_envs=E, device=DEV, generator="bezier",
                                       source="relaxed_stations", gates=8,
                                       elevation=noise, spacing_metric="xyz"),
            pipeline_mod.presets.gates(num_envs=E, device=DEV, generator="acme.tagged",
                                       on_invalid="poison"),
        ]
        gens = [GateGenerator(spec, _rng()) for spec in specs]
    m = _i32([1, 0])

    def hot():
        for gen in gens:
            gen.generate()
            gen.generate(advance=True, mask=m)
    return hot


def _legacy_offset_without_hw_out():
    # The legacy offset / _inflate_warp call forms of 0.2.0.dev3 (no hw_out: a null array
    # is passed, nothing stored, nothing allocated), on buffers allocated at setup.
    gen = _track_gen()
    s = gen._scratch
    stag = s.inflate
    dev3 = warp_pipeline.InflateScratch(
        area_a=stag.area_a, area_b=stag.area_b, kappa=stag.kappa, w=stag.w,
        out2=stag.out2, ctr2=stag.ctr2, inn2=stag.inn2, tan2=stag.tan2, nrm2=stag.nrm2,
        z=stag.z, fail=stag.fail)
    scratch = warp_pipeline._Scratch(inflate=dev3, cs_seg=s.cs_seg, cs_s=s.cs_s)
    out, _ = warp_pipeline._inflate_warp_alloc(gen._config)
    center = gen.program.tap("cl.xy", after="relax")
    count, valid = gen.program.tap("cl.count"), gen.program.tap("gen.valid")
    ctx = LaunchContext.for_call(DEV)

    def hot():
        warp_pipeline.offset(stag.ctr2, stag.nrm2, 0.1, stag.out2, stag.inn2,
                             stag.area_a, stag.area_b, count, ctx=ctx)
        warp_pipeline._inflate_warp(center, gen._config, out=out, valid=valid,
                                   count=count, scratch=scratch, seeds=gen._seed_buf,
                                   truncated=out.truncated, ctx=ctx)
    return hot


def _gate_run_stations():
    # Gates through the stations of a third-party TRACK generator (raw and relaxed).
    from tests import _gate_toys as gt

    with registry.scoped():
        gt.register(registry)
        gens = [GateGenerator(pipeline_mod.presets.gates(
            num_envs=E, device=DEV, generator=gt.SUPERFORMULA, source=source), _rng())
            for source in ("stations", "relaxed_stations")]
    m = _i32([1, 0])

    def hot():
        for gen in gens:
            gen.generate()
            gen.generate(advance=True, mask=m)
    return hot


def _gate_wrappers():
    # The standalone warp_gate wrappers (a hand-built GateSequence), on buffers
    # allocated at setup.
    G = 8
    cfg = GateGenConfig(device=DEV, num_envs=E, max_gates=G, gate_width=0.1)
    gates = warp_gate.alloc_gate_sequence(cfg)
    pos2 = wp.array(np.random.default_rng(0).random((E * G, 2)).astype(np.float32),
                    dtype=wp.vec2f, device=DEV)
    count = wp.full(E, 6, dtype=wp.int32, device=DEV)
    fallbacks = wp.zeros(E, dtype=wp.int32, device=DEV)
    fail = wp.zeros(E, dtype=wp.int32, device=DEV)
    ctx = LaunchContext.for_call(DEV)

    def hot():
        warp_gate.normalize_positions(pos2, G, count, 1.0, ctx=ctx)
        warp_gate.relax_gate_spheres(pos2, G, count, 0.05, 4, ctx=ctx)
        ctx.launch(warp_gate._lift_positions_k, E * G, [pos2, gates.half_size,
                                                        gates.position])
        warp_gate.tangents_from_positions(gates.position, gates.tangent, G, count, ctx=ctx)
        warp_gate.finalize_gate_sequence(gates, cfg, fallbacks=fallbacks, ctx=ctx,
                                         fail=fail)
        # The pre-PR11 call form: the fail bits go to gates.fail_mask.
        warp_gate.finalize_gate_sequence(gates, cfg, fallbacks=fallbacks, ctx=ctx)
    return hot




def _run_track_plugin_profile():
    # TrackGenConfig.z_profile naming an elevation plugin (the flat shortcut).
    from tests._stage_toys import HILLS, register

    with registry.scoped():
        register(registry)
        return _track_gen(z_profile=HILLS)._run


def _legacy_pipeline_wrappers():
    # The PR9 composition (warp_pipeline._run_pipeline -> _inflate_warp), kept as the
    # byte-identity reference, run on a TrackGenerator's _scratch view (relax on).
    gen = _track_gen()
    ctx = LaunchContext.for_call(DEV)

    from tests._reference_generators.warp_pipeline import _run_pipeline

    def hot():
        _run_pipeline(gen._config, gen._seed_buf, out=gen._track,
                                    scratch=gen._scratch,
                                    generator_spec=gen._generator_spec, ctx=ctx)
    return hot


def _run_randomized():
    """Roadmap PR13b: a ``randomize.uniform`` stage drawing a plugin generator's dynamic
    ``amp`` (toy.wobble) every generation, under a masked advance."""
    def setup():
        from tests._plugin_toys import register_stage_toys

        with registry.scoped():
            register_stage_toys(registry)
            gen = TrackGenerator(pipeline_mod.presets.track(
                num_envs=E, device=DEV, generator="toy.wobble",
                randomize={"generator.amp": (0.05, 0.35)}), _rng())
        mask = wp.array(np.arange(E) % 2, dtype=wp.int32, device=DEV)
        ctx = LaunchContext.for_call(DEV)

        def hot():
            gen._enqueue(ctx, advance=True, mask=mask)
        return hot
    return setup


def _result_copy(kind):
    """Roadmap PR13b: ``copy_from`` (the hot-path snapshot) of a whole result and of a
    view of it, into preallocated results."""
    def setup():
        gen = _track_gen() if kind == "track" else _gate_gen("bezier")
        result = gen.program.result
        snap = result.clone()
        half = type(result).alloc(result.view(0, E // 2).layout)

        def hot():
            snap.copy_from(result)
            half.copy_from(result.view(0, E // 2))
        return hot
    return setup


def _inflate_fixed_n():
    # _inflate_warp's count=None convenience path (fixed N == num_points) on a scratch
    # allocated at construction.
    cfg = TrackGenConfig(device=DEV, num_envs=E)
    gen = TrackGenerator(cfg, _rng())
    gen.generate()
    N = int(cfg.num_points)
    out, scratch = warp_pipeline._inflate_warp_alloc(
        TrackGenConfig(device=DEV, num_envs=E, N_max=N))
    center = gen._scratch.gen_centerline

    ctx = LaunchContext.for_call(DEV)

    def hot():
        warp_pipeline._inflate_warp(center, cfg, out=out, scratch=scratch, ctx=ctx)
    return hot


DRIVERS = {
    **{f"track_run[{g}]": _run_track(g) for g in registry.available("generator")},
    "track_run[toy.circle plugin]": _run_toy_plugin("track"),
    "track_run[randomize]": _run_randomized(),
    "copy_from[track]": _result_copy("track"),
    "copy_from[gates]": _result_copy("gates"),
    "track_run[relax=toy.converging_smooth]": _run_bounded_toy(),
    "track_run[bezier,z=uniform]": _run_track("bezier", z_profile="uniform"),
    "track_run[bezier,z=noise]": _run_track("bezier", z_profile="noise"),
    "track_run[bezier,no-relax]": _run_track("bezier", relax_enable=False),
    **{f"gate_run[{g}]": _run_gates(g) for g in registry.available("gate_generator")},
    "gate_run[toy.ring plugin]": _run_toy_plugin("gate"),
    "gate_run[bezier,z=uniform]": _run_gates("bezier", z_profile="uniform"),
    "collision[segments]": _collision("segments"),
    "collision[sdf]": _collision("sdf"),
    "props": _props,
    "checkpoints": _checkpoints,
    "progress": _progress,
    "localizer": _localizer,
    "localize_profiles": _localize_profiles,
    "heightfield": _heightfield,
    "commit[_commit_envs]": _commit_envs_driver,
    "course_line": _course_line,
    "discs": _discs,
    "frames": _frames,
    "course[track,segments]": _course("track", collision="segments",
                                      heightfield_resolution=16),
    "course[track,sdf]": _course("track", collision="sdf", sdf_resolution=16),
    "course[gates,posts]": _course("gates", post_radius=0.02),
    "course[gates,frames]": _course("gates", frame_collision=True, agent_radius=0.05,
                                    frame_thickness=0.02, frame_depth=0.02),
    "generate[track]": _facade_generate(TrackGenerator),
    "generate[gates]": _facade_generate(GateGenerator),
    "regenerate[track,segments]": _course_regenerate("track", collision="segments",
                                                     heightfield_resolution=16),
    "regenerate[track,sdf]": _course_regenerate("track", collision="sdf",
                                                sdf_resolution=16),
    "regenerate[gates,posts]": _course_regenerate("gates", post_radius=0.02),
    "regenerate[gates,frames]": _course_regenerate("gates", frame_collision=True,
                                                   agent_radius=0.05,
                                                   frame_thickness=0.02,
                                                   frame_depth=0.02),
    "advance[track]": _facade_advance(TrackGenerator),
    "advance[gates]": _facade_advance(GateGenerator),
    "regenerate_masked[track,segments]": _course_masked("track", collision="segments",
                                                        heightfield_resolution=16),
    "regenerate_masked[track,sdf]": _course_masked("track", collision="sdf",
                                                   sdf_resolution=16),
    "regenerate_masked[gates,posts]": _course_masked("gates", post_radius=0.02,
                                                     localize_window=2),
    "regenerate_masked[gates,frames]": _course_masked("gates", frame_collision=True,
                                                      agent_radius=0.05,
                                                      frame_thickness=0.02,
                                                      frame_depth=0.02),
    "seed_state": _seed_state,
    "rng[sample_out]": _rng_sample_out,
    "wrappers[geometry]": _geometry_wrappers,
    "wrappers[inflate_fixed_n]": _inflate_fixed_n,
    "wrappers[legacy_pipeline]": _legacy_pipeline_wrappers,
    "program[standalone]": _program_standalone,
    "program[toys,poison]": _program_toys_poison,
    "gate_program[toys,poison]": _gate_program_toys_poison,
    "gate_run[stations superformula]": _gate_run_stations,
    "wrappers[gate]": _gate_wrappers,
    "track_run[z=acme.sinusoid_hills]": _run_track_plugin_profile,
    "ops[all]": _ops_all,
    "program[attrs,from_channel,poison,masked]": _program_channels,
    "gate_program[stations xyz,attrs]": _gate_program_channels,
    "wrappers[legacy offset/inflate, no hw_out]": _legacy_offset_without_hw_out,
}


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_hot_path_has_no_host_sync_or_readback(name):
    hot = DRIVERS[name]()
    with _probe() as guard:
        hot()
    expected = {k for k in _KNOWN_READBACKS if name == "track_run[repulsive]"}
    assert set(guard.allowed) == expected, \
        f"known-readback allowlist out of date: {guard.allowed}"
    # The bounded loops' CPU host poll (repulsive, the bounded toy) is sanctioned, not
    # allowed: the guard lists it apart.
    if name in ("track_run[repulsive]", "track_run[relax=toy.converging_smooth]"):
        assert set(guard.sanctioned) == {
            ("wp.array.numpy", "track_gen.exec.host", "_poll_i32")}, guard.sanctioned


# ---------------------------------------------------------------------------------------
# Coverage: every module-level function of the pipeline modules is driven above.
# ---------------------------------------------------------------------------------------

_PIPELINE_MODULES = (
    warp_pipeline, warp_gate, warp_relax, warp_width, warp_zprofile, warp_generate,
    warp_generate_checkpoint, warp_generate_hull, warp_generate_polar,
    warp_generate_repulsive, warp_generate_voronoi, warp_generate_gates,
    warp_generate_checkpoint_gates, warp_generate_polar_gates,
    warp_generate_voronoi_gates,
    # not pipeline stages, but hot paths with module-level wrappers too (PR4)
    rng_kernels, seeds_mod, reduce_mod,
    # the public track_gen.ops primitives (PR7); ops.fill is shadowed by the function
    *(importlib.import_module(f"track_gen.ops.{m}")
      for m in ("check", "resample", "geometry", "sampling", "curves", "fill")),
    # the pipeline planner and the built-in track stages (PR10a): their enqueues are
    # closures and methods; every module-level function is construction-time
    pipeline_mod, stages_mod,
    *(importlib.import_module(f"track_gen.pipeline.{m}")
      for m in ("spec", "planner", "program", "legacy", "presets", "ingest")),
    *(importlib.import_module(f"track_gen.stages.{m}")
      for m in ("channels", "params", "track", "commit", "gates", "randomize",
                "external")),
)

# Construction-time functions (allocation, registration, host precompute for sizing).
# They run before any probe and are allowed to allocate or read back.
_CONSTRUCTION_ONLY = {
    'track_gen._src.warp_pipeline': {
        '_fill_knot_scratch',
        '_inflate_warp_alloc',
        '_registered_generator_spec',
    },
    'track_gen._src.warp_gate': {
        '_gate_center_distance',
        'alloc_gate_sequence',
    },
    'track_gen._src.warp_zprofile': {
        'alloc_knot_scratch',
        'alloc_z_scratch',
    },
    'track_gen._src.warp_generate': {
        '_alloc_bezier_scratch',
        'register',
    },
    'track_gen._src.warp_generate_checkpoint': {
        '_alloc_checkpoint_scratch',
        'register',
    },
    'track_gen._src.warp_generate_hull': {
        '_alloc_hull_scratch',
        'register',
    },
    'track_gen._src.warp_generate_polar': {
        '_alloc_polar_scratch',
        'polar_num_knots',
        'register',
    },
    'track_gen._src.warp_generate_repulsive': {
        '_alloc_repulsive_scratch',
        '_sobolev_circulant_row',
        '_stage_schedule',
        'register',
    },
    'track_gen._src.warp_generate_voronoi': {
        '_alloc_voronoi_scratch',
        'register',
        'voronoi_layout_mode',
    },
    'track_gen._src.warp_generate_gates': {
        'register',
    },
    'track_gen._src.warp_generate_checkpoint_gates': {
        'register',
    },
    'track_gen._src.warp_generate_polar_gates': {
        'register',
    },
    'track_gen._src.warp_generate_voronoi_gates': {
        'register',
    },
    'track_gen.seeds': {
        '_is_rng',
        '_wrap_int32',
        'effective_seed',
    },
    'track_gen.pipeline.spec': {
        '_canonical_param',
        '_capacity_value',
        '_draw_target',
        '_frozen_map',
        '_normalize_size_hint',
        '_port_value',
        '_randomizers',
        '_required',
        '_thaw',
        '_with_randomizers',
    },
    'track_gen.ops.resample': {
        '_carry_kernel',
        '_check_tables',
    },
    'track_gen.pipeline.planner': {
        '_bind',
        '_build',
        '_capture_classes',
        '_carry_set',
        '_check_agreement',
        '_check_caller_capturable',
        '_check_capacity',
        '_check_draws',
        '_check_exports',
        '_check_fail_start',
        '_check_flat_lift',
        '_check_gate_exports',
        '_check_gate_fail_start',
        '_check_gate_out',
        '_check_gate_plan',
        '_check_out',
        '_check_overlap',
        '_check_param_writers',
        '_check_provided',
        '_check_renormalized_size',
        '_check_repulsive_clearance',  # planner port/ordering checks, before hot calls
        '_check_result_out',
        '_check_seeds',
        '_check_size_truncation',
        '_check_spec_shape',
        '_check_topology',
        '_check_width_placement',
        '_decl',
        '_domain_layout',
        '_draw_bounds',
        '_effective_ports',
        '_env_span',
        '_export_sources',
        '_gen_slot',
        '_host_label',
        '_is_int',
        '_is_partial',
        '_is_raw_family',
        '_is_seed_form',
        '_label',
        '_live_overlaps',
        '_missing_hint',
        '_offset_seeds',
        '_pad_value',
        '_plan_channels',
        '_plugin_params',
        '_ref_param',
        '_register_result',
        '_resolve',
        '_result_attributes',
        '_result_kind',
        '_root_of',
        '_seed_state',
        '_segments',
        '_slot_info',
        '_stage_bounds',
        '_stage_class',
        '_stage_hints',
        'build',
    },
    'track_gen.pipeline.legacy': {
        '_check_flat_size',
        '_legacy_ordering',
        '_legacy_params',
        '_params_of',
        '_scratch_view',
        '_to_gate_pipeline_spec',
        '_to_pipeline_spec',
        'gate_generator_config',
        'generator_config',
        'to_gate_pipeline_spec',
        'to_pipeline_spec',
    },
    'track_gen.pipeline.presets': {
        '_check_centerline_args',
        '_check_half_width',
        '_declared',
        '_gate_stages',
        '_has_half_width_field',
        '_is_flat',
        '_is_one',
        '_is_pair',
        '_lookup',
        '_param',
        '_preferred_relax',
        '_provides',
        '_relax_on',
        '_sel_name',
        '_select',
        '_size_fields',
        '_size_remedy',
        '_with_gate_validity',
        '_with_half_width',
        '_with_params',
        '_with_size',
        'gates',
        'track',
    },
    'track_gen.pipeline.ingest': {
        '_check_values',
        '_envs',
        '_perimeter',
        '_program_of',
        '_rows',
        '_sizing',
        'load_centerlines',
    },
    'track_gen.stages.external': {
        'register',
    },
    'track_gen.stages': {
        '_check_builtin_stage',
        'register',
        'register_builtins',
    },
    'track_gen.stages.channels': {
        '_c',
        'channel_domain',
        'family_key',
        'is_gate_channel',
        'is_plugin_channel',
        'required_port_domain',
    },
    'track_gen.stages.params': {
        '_check_control_points',
        '_check_half_width',
        '_check_z_draws',
        '_check_z_range',
        '_f32_distance',
        '_warn_inert',
    },
    'track_gen.stages.track': {
        '_alloc_track',
        '_carry_ops',
        '_tables',
        '_with_half_width',
        '_zcfg',
        '_zdyn',
        'raw_attributes',
    },
    'track_gen.stages.gates': {
        '_alloc_gate_sequence',
        '_default_order',
        '_orderings_of',
        '_preferred',
        '_supported_orderings',
    },
}


def _module_functions(mod):
    return {name for name, obj in vars(mod).items()
            if inspect.isfunction(obj) and obj.__module__ == mod.__name__}


def test_every_wrapper_is_driven():
    files = {inspect.getfile(m): m.__name__ for m in _PIPELINE_MODULES}
    driven: set[tuple[str, str]] = set()

    def tracer(frame, event, arg):
        if event == "call":
            mod = files.get(frame.f_code.co_filename)
            if mod is not None:
                driven.add((mod, frame.f_code.co_name))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for setup in DRIVERS.values():
            hot = setup()
            sys.setprofile(tracer)
            try:
                hot()
            finally:
                sys.setprofile(None)

    undriven, stale = [], []
    for mod in _PIPELINE_MODULES:
        funcs = _module_functions(mod)
        exempt = _CONSTRUCTION_ONLY.get(mod.__name__, set())
        stale += [f"{mod.__name__}.{f}" for f in sorted(exempt - funcs)]
        undriven += [f"{mod.__name__}.{f}" for f in sorted(funcs - exempt)
                     if (mod.__name__, f) not in driven]
    assert not stale, f"_CONSTRUCTION_ONLY names that no longer exist: {stale}"
    assert not undriven, f"wrappers no driver reaches (add one to DRIVERS): {undriven}"


# ---------------------------------------------------------------------------------------
# The debug sync itself.
# ---------------------------------------------------------------------------------------

class _FakeStream:
    def __init__(self, capturing):
        self.is_capturing = capturing


class _FakeDevice:
    is_cuda = True

    def __init__(self, stream_capturing=False, device_capturing=False):
        self.stream = _FakeStream(stream_capturing)
        self.is_capturing = device_capturing or stream_capturing


def _fake_cuda(monkeypatch, **kw):
    dev = _FakeDevice(**kw)
    synced = []
    monkeypatch.setattr(runtime.wp, "get_device", lambda d=None: dev)
    monkeypatch.setattr(runtime.wp, "synchronize_stream", synced.append)
    monkeypatch.setattr(runtime.wp, "synchronize",
                        lambda: pytest.fail("_sync must never sync every device"))
    return dev, synced


def test_sync_is_a_noop_without_debug_env(monkeypatch):
    # Returns before resolving the device: no CUDA call is possible.
    monkeypatch.setattr(runtime.wp, "get_device",
                        lambda d=None: pytest.fail("device resolved without debug env"))
    with _probe():
        runtime._sync("cuda:0")


def test_debug_sync_blocks_on_the_current_stream(monkeypatch):
    monkeypatch.setenv("TRACKGEN_DEBUG_SYNC", "1")
    dev, synced = _fake_cuda(monkeypatch)
    runtime._sync("cuda:0")
    assert synced == [dev.stream]
    explicit = _FakeStream(False)
    runtime._sync("cuda:0", stream=explicit)
    assert synced == [dev.stream, explicit]


@pytest.mark.parametrize("kw", [dict(stream_capturing=True), dict(device_capturing=True)])
def test_debug_sync_skips_while_capturing(monkeypatch, kw):
    monkeypatch.setenv("TRACKGEN_DEBUG_SYNC", "1")
    _, synced = _fake_cuda(monkeypatch, **kw)
    runtime._sync("cuda:0")
    assert synced == []


def test_debug_sync_is_a_noop_on_cpu(monkeypatch):
    monkeypatch.setenv("TRACKGEN_DEBUG_SYNC", "1")
    monkeypatch.setattr(runtime.wp, "synchronize_stream",
                        lambda s: pytest.fail("synced on cpu"))
    runtime._sync("cpu")
