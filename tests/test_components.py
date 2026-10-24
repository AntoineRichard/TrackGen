"""Roadmap PR14: the component protocol and adapters (CPU).

1. ``Component`` / ``ComponentContext`` / ``InputSpec``; ``for_producer``'s capacities.
2. ``check_component`` passes for every built-in adapter.
3. The kit's negative toys each fail exactly their check; the writing-a-component toy
   (``CurvatureLookahead``, public imports only) passes every check.
4. Byte identity for valid envs: the base goldens (``tests/goldens/pr14_utilities.json``)
   and the frozen base kernels (``tests/_pr14_base_kernels.py``) launched next to the
   edited ones.
5. ``Course``: identity with a hand-wired composition of the legacy utilities,
   ``StepResult.valid``, the posts, the course line's generation, the default checkpoint
   capacity and its warning.
6. Generation-aware self-reset.
7. Invalid-env sentinels (the table values, K == P per utility).
8. Capacities and the grid-resolution warning.
9. ``curvature`` / ``speed_profile`` ``out=``.
10. ``ctx=`` on the legacy hot methods, ``.result``.
11. ``RebindAfterCaptureWarning``.
12. ``CheckpointSet.valid`` / ``.generation``.
13. The RL loop with a forced-invalid env.
14. ``check_source(optional=)``.

The CUDA proofs (capture, the trajectory twin, the base kernels on ``cuda:0``) are
``tests/test_components_cuda.py``.
"""
from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
import types
import warnings

import numpy as np
import pytest
import warp as wp

from tests import _component_toys as toys
from tests._pr14_base_kernels import BASE_KERNELS
from tests._results_helpers import MinimalCenterline
from tests.conftest import DEVS
from tests.tools import make_pr14_goldens as goldens
from track_gen import GateGenConfig, TrackGenConfig, tools
from track_gen._src import (checkpoints as ckpt_mod, collision as coll_mod,
                            collision_discs as discs_mod, collision_sdf as sdf_mod,
                            course_line as line_mod, localize as loc_mod,
                            progress as prog_mod, props as props_mod)
from track_gen._src.collision_frames import FrameChecker
from track_gen._src.course_line import CourseLine
from track_gen._src.gate_generator import GateGenerator
from track_gen._src.track_generator import TrackGenerator
from track_gen.checkpoints import CheckpointSampler, CheckpointSet
from track_gen.collision import CollisionChecker, DiscChecker
from track_gen.course import Course, CourseConfig, StepResult
from track_gen.exec import LaunchContext, RebindAfterCaptureWarning
from track_gen.heightfield import HeightFieldBaker
from track_gen.localize import TrackLocalizer, curvature, speed_profile
from track_gen.progress import ProgressTracker
from track_gen.props import PropSampler
from track_gen.results import (HasBand, HasCenterline, HasCheckpoints, HasGeneration,
                               HasValidity, Layout, check_source)
from track_gen.stages.commit import _commit_envs
from track_gen.testing import CHECKS, check_component, hot_path_guard, trace_launches
from track_gen.testing.components import _builtin_components

pytest_plugins = ["track_gen.testing.pytest_plugin"]
pytestmark = pytest.mark.filterwarnings(
    "ignore:GateGeneratorSpec .* is deprecated:DeprecationWarning")

DEV = "cpu"
E = 6
NAN3 = np.full(3, np.nan, np.float32)


def _i32(values, device=DEV) -> wp.array:
    return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=device)


def _vec3(host, device=DEV) -> wp.array:
    return wp.array(np.asarray(host, np.float32).reshape(-1, 3), dtype=wp.vec3f,
                    device=device)


def _track_gen(seeds=11, E=E, device=DEV, **kw) -> TrackGenerator:
    return TrackGenerator(TrackGenConfig(num_envs=E, device=device, **kw), seeds=seeds)


def _gate_gen(seeds=5, E=E, device=DEV, **kw) -> GateGenerator:
    kw.setdefault("gate_width", 0.1)
    return GateGenerator(GateGenConfig(num_envs=E, device=device, **kw), seeds=seeds)


def _host(a) -> np.ndarray:
    return np.ascontiguousarray(a.numpy()).copy()


def _bytes(a) -> bytes:
    return _host(a).tobytes()


def _rows(a, n_envs=E) -> np.ndarray:
    h = _host(a)
    return h.reshape(n_envs, -1) if h.ndim == 1 else h.reshape(n_envs, -1, *h.shape[1:])


def _same_rows(a, b, envs, n_envs=E) -> bool:
    ra, rb = _rows(a, n_envs), _rows(b, n_envs)
    return all(ra[e].tobytes() == rb[e].tobytes() for e in envs)


def _points(src, step: int, *, gates: bool = False) -> np.ndarray:
    """One point per env walking around the source (slightly off it), starting a few
    points before point 0 (so the first checkpoint / gate is crossed early)."""
    pts = (src.position if gates else src.center).numpy()
    n = pts.shape[0] // E
    p = pts.reshape(E, n, 3)
    cnt = src.count.numpy()
    out = np.zeros((E, 3), np.float32)
    for e in range(E):
        m = int(min(cnt[e], n))
        if m >= 2:
            k = (2 * step + e - 4) % m if not gates else (step // 2 + m - 1) % m
            a, b = p[e, k], p[e, (k + 1) % m]
            out[e] = a + (b - a) * np.float32(0.5) + np.float32([0.01 * ((step + e) % 3 - 1),
                                                                   0.0, 0.0])
    return out


def _boxes(n: int, yaw: float = 0.2):
    q = np.tile(np.float32([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]), (n, 1))
    he = np.tile(np.float32([0.03, 0.015]), (n, 1))
    return (wp.array(q, dtype=wp.quatf, device=DEV), wp.array(he, dtype=wp.vec2f,
                                                               device=DEV))


def _keep_and_poison(src, fail, device=DEV):
    """``(K, P)``: clones of ``src`` with the envs of ``fail`` failed by the commit's own
    kernel at ``poison=0`` (geometry kept) and ``poison=1``."""
    K, P = src.clone(), src.clone()
    f = _i32(fail, device)
    ctx = LaunchContext.for_call(device)
    _commit_envs(K, f, poison=False, ctx=ctx)
    _commit_envs(P, f, poison=True, ctx=ctx)
    return K, P


_FAIL = [0, 1, 0, 0, 1, 0]
_BAD = [e for e in range(E) if _FAIL[e]]
_GOOD = [e for e in range(E) if not _FAIL[e]]


class _Capturing(LaunchContext):
    """A context that reports ``capturing`` (arms the rebind probe)."""

    @property
    def capturing(self) -> bool:
        return True


# ---------------------------------------------------------------------------------------
# 1. Component, ComponentContext, InputSpec
# ---------------------------------------------------------------------------------------

def test_component_defaults_are_immutable_and_not_shared():
    c = tools.Component()
    assert isinstance(tools.Component.inputs, types.MappingProxyType)
    with pytest.raises(TypeError):
        tools.Component.inputs["x"] = tools.InputSpec(wp.vec3f)  # type: ignore[index]
    assert tools.Component.requires == tools.Component.optional == tools.Component.after \
        == ()
    assert (c.name, c.source, c.topology, c.stateful, c.layout, c.result) == \
        ("component", None, "closed", False, None, None)
    # ``utility`` belongs to the adapters (the legacy utility they wrap), not the contract
    assert not hasattr(tools.Component, "utility")
    assert tools.Progress().utility is None and tools.GateCheckpoints().utility is None
    with pytest.raises(NotImplementedError):
        c.attach(None, None)
    ctx = LaunchContext.for_call(DEV)
    assert c.step(ctx) is None and c.on_generate(ctx, None) is None
    assert c.reset(ctx, _i32(np.zeros(E))) is None
    # the adapters set their per-instance declarations on the instance only
    a, b = tools.BandCollision(max_boxes=2), tools.BandCollision()
    assert a.inputs["box_position"].slots == 2 and b.inputs["box_position"].slots == 1
    assert tools.BandCollision.inputs is tools.Component.inputs
    assert not isinstance(tools.Component, type(HasBand))   # a plain base class


def test_input_spec_validation_and_equality():
    with pytest.raises(ValueError, match="slots must be an int >= 1"):
        tools.InputSpec(wp.vec3f, 0)
    with pytest.raises(TypeError, match="required must be a bool"):
        tools.InputSpec(wp.vec3f, required=1)
    with pytest.raises(TypeError, match="default must be an input name"):
        tools.InputSpec(wp.vec3f, default=3)
    assert tools.InputSpec(wp.vec3f, doc="a") == tools.InputSpec(wp.vec3f, doc="b")
    assert tools.InputSpec(wp.vec3f, 2) != tools.InputSpec(wp.vec3f, 1)
    lay = Layout(E, 1, DEV, True)
    spec = tools.InputSpec(wp.vec3f, 2)
    spec.check("box", wp.zeros(2 * E, dtype=wp.vec3f, device=DEV), lay, consumer="c")
    with pytest.raises(ValueError, match=r"^c: input box must have shape \(12,\)"):
        spec.check("box", wp.zeros(E, dtype=wp.vec3f, device=DEV), lay, consumer="c")
    with pytest.raises(ValueError, match=r"^c: input box must have dtype vec3f"):
        spec.check("box", wp.zeros(2 * E, dtype=wp.vec2f, device=DEV), lay, consumer="c")
    strided = wp.zeros(4 * E, dtype=wp.vec3f, device=DEV)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        spec.check("box", strided, lay, consumer="c")


def test_bind_routes_defaults_and_refuses_bad_inputs():
    gen = _track_gen()
    cc = tools.ComponentContext.for_producer(gen)
    band = cc.add(tools.BandCollision(), gen.program.result)
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    q, he = _boxes(E)
    band.bind(position=pos, orientation=q, half_extents=he)     # box_position <- position
    assert band.utility._bound[0] is pos
    band.bind(box_position=pos, orientation=q, half_extents=he)
    with pytest.raises(TypeError, match=r"collision: missing inputs \['box_position'\]"):
        band.bind(orientation=q, half_extents=he)
    with pytest.raises(TypeError, match=r"collision: unknown inputs \['speed'\]"):
        band.bind(position=pos, orientation=q, half_extents=he, speed=pos)
    with pytest.raises(ValueError, match=r"^collision: input half_extents must have dtype"):
        band.bind(position=pos, orientation=q, half_extents=pos)
    two = tools.BandCollision(max_boxes=2, name="two")
    tools.ComponentContext.for_producer(gen).add(two, gen.program.result)
    with pytest.raises(TypeError, match=r"two: unknown inputs \['position'\]"):
        two.bind(position=pos, orientation=q, half_extents=he)  # no default with 2 boxes
    with pytest.raises(RuntimeError, match="bind\\(\\) before attach"):
        tools.Progress().bind(position=pos)
    # A routed input's error names the argument it was routed from.
    nine = wp.zeros(E + 3, dtype=wp.vec3f, device=DEV)
    with pytest.raises(ValueError, match=r"^collision: input box_position \(routed from "
                                         r"position\) must have shape"):
        band.bind(position=nine, orientation=q, half_extents=he)


class _OptionalInput(tools.Component):
    """Takes a required ``position`` and an optional ``speed`` input."""

    name = "opt_in"
    inputs = types.MappingProxyType({
        "position": tools.InputSpec(wp.vec3f),
        "speed": tools.InputSpec(wp.float32, required=False)})

    def attach(self, source, cc):
        self.result = types.SimpleNamespace()

    def on_bind(self, position, speed):          # no default: an absent one is None
        self.bound = (position, speed)


def test_an_absent_optional_input_reaches_on_bind_as_none():
    gen = _track_gen()
    comp = tools.ComponentContext.for_producer(gen).add(_OptionalInput(),
                                                        gen.program.result)
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    speed = wp.zeros(E, dtype=wp.float32, device=DEV)
    comp.bind(position=pos)
    assert comp.bound == (pos, None)
    comp.bind(position=pos, speed=speed)
    assert comp.bound == (pos, speed)
    with pytest.raises(ValueError, match=r"^opt_in: input speed must have shape"):
        comp.bind(position=pos, speed=wp.zeros(E + 1, dtype=wp.float32, device=DEV))
    with pytest.raises(TypeError, match=r"opt_in: missing inputs \['position'\]"):
        comp.bind(speed=speed)


class _LayoutProbe(tools.Component):
    """Records the layout it sees in attach."""

    name = "probe"
    requires = (HasCenterline,)
    inputs = types.MappingProxyType({"position": tools.InputSpec(wp.vec2f)})

    def attach(self, source, cc):
        self.seen = self.layout
        self.result = types.SimpleNamespace()


def test_context_add_checks_dependencies_names_inputs_and_sources():
    gen = _track_gen()
    t = gen.program.result
    cc = tools.ComponentContext.for_producer(gen)
    with pytest.raises(ValueError, match=r"progress: \['checkpoints'\] must be attached "
                                         r"before it"):
        cc.add(tools.Progress(), t)
    cps = cc.add(tools.Checkpoints(0.5), t)
    prog = cc.add(tools.Progress(), t)
    assert prog.utility._cps is cps.result                  # source resolution
    assert prog.layout == Layout(E, cps.utility.max_checkpoints, DEV, True)
    with pytest.raises(ValueError, match="'progress' is already attached"):
        cc.add(tools.Progress(), t)
    with pytest.raises(ValueError, match=r"input 'position': probe declares \(vec2f, 1\) "
                                         r"but progress declares \(vec3f, 1\)"):
        cc.add(_LayoutProbe(), t)
    probe = _LayoutProbe()
    probe.inputs = types.MappingProxyType({})
    cc.add(probe, t)
    assert probe.seen == t.layout                            # set BEFORE attach
    with pytest.raises(TypeError, match=r"^collision needs a HasBand source"):
        cc.add(tools.BandCollision(), MinimalCenterline(t))
    with pytest.raises(KeyError, match=r"no component 'nope' is attached \(attached: "):
        cc.component("nope")
    assert cc.names == ("checkpoints", "progress", "probe")
    bad = tools.Checkpoints(0.5, name="c2")
    bad.requires = (int,)
    with pytest.raises(TypeError, match="requires must be a tuple of track_gen.results"):
        cc.add(bad, t)


class _OptionalOnly(tools.Component):
    """Declares ``optional`` protocols only (``valid`` / ``generation`` when present)."""

    name = "opt_only"
    optional = (HasValidity, HasGeneration)

    def attach(self, source, cc):
        self.seen = (self.layout, getattr(source, "valid", None))
        self.result = types.SimpleNamespace()


def test_context_add_checks_optional_only_declarations_and_the_batch():
    gen = _track_gen()
    t = gen.program.result
    cc = tools.ComponentContext.for_producer(gen)
    # A malformed optional member raises at add (the required members' messages), with
    # or without a layout on the target.
    with pytest.raises(ValueError, match=r"^opt_only: SimpleNamespace\.valid: expected a "
                                         r"contiguous \[6\] int32"):
        cc.add(_OptionalOnly(), types.SimpleNamespace(
            valid=wp.zeros(E, dtype=wp.float32, device=DEV)))
    with pytest.raises(ValueError, match=r"^opt_only: .*generation"):
        cc.add(_OptionalOnly(), types.SimpleNamespace(
            generation=wp.zeros(99, dtype=wp.int32, device=DEV)))
    with pytest.raises(ValueError, match=r"^opt_only: .*valid"):
        cc.add(_OptionalOnly(), types.SimpleNamespace(
            layout=t.layout, valid=wp.zeros(3, dtype=wp.int32, device=DEV)))
    # A target with none of them: the context's layout.
    ok = cc.add(_OptionalOnly(), types.SimpleNamespace())
    assert ok.seen == (cc.layout, None)
    ok2 = cc.add(_named(_OptionalOnly(), "o2"), t)
    assert ok2.seen == (t.layout, t.valid)
    # A producer-result target of another batch is refused.
    other = _track_gen(E=4).program.result
    with pytest.raises(ValueError, match=r"^checkpoints: its read target holds .*num_envs=4"
                                         r".*this context's batch"):
        cc.add(tools.Checkpoints(0.5), other)


class _OptionalCheckpoints(tools.Component):
    """Reads the checkpoint set, declaring ``HasCheckpoints`` / ``HasValidity`` as
    ``optional`` only: its stride is the set's M slots, not the producer's."""

    name = "opt_ck"
    source = "checkpoints"
    optional = (HasCheckpoints, HasValidity)

    def attach(self, source, cc):
        self.result = types.SimpleNamespace()


def test_optional_only_declarations_take_the_stride_of_a_per_point_member():
    gen = _track_gen()
    t = gen.program.result
    cc = tools.ComponentContext.for_producer(gen)
    cps = cc.add(tools.Checkpoints(0.5), t).result
    M = int(cps.position.shape[0]) // E
    assert not hasattr(cps, "layout") and M != t.layout.stride
    # A CheckpointSet has no layout: count gives num_envs, position the stride (M).
    comp = cc.add(_OptionalCheckpoints(), t)
    assert (comp.layout.num_envs, comp.layout.stride) == (E, M)
    assert comp.layout == cc.add(_named(_OptionalCheckpoints(), "opt_ck2"), t).layout
    # The same declaration under requires infers the same layout.
    req = _named(_OptionalCheckpoints(), "req_ck")
    req.requires, req.optional = (HasCheckpoints,), (HasValidity,)
    assert cc.add(req, t).layout == comp.layout
    # A per-point member alone (no count / per-env member): num_envs from the context,
    # the stride from the member.
    only = _named(_OptionalCheckpoints(), "opt_pos")
    only.source = None
    pos = types.SimpleNamespace(position=cps.position)
    assert cc.add(only, pos).layout == Layout(E, M, DEV, cc.layout.closed)
    # A malformed per-point member is still refused.
    bad = _named(_OptionalCheckpoints(), "opt_bad")
    bad.source = None
    with pytest.raises(ValueError, match=r"^opt_bad: SimpleNamespace\.position"):
        cc.add(bad, types.SimpleNamespace(
            count=cps.count, position=wp.zeros(E * M + 1, dtype=wp.vec3f, device=DEV)))


def _named(comp, name):
    comp.name = name
    return comp


def test_for_producer_needs_a_plan():
    gen = _track_gen(E=2)
    track = gen.generate()
    with pytest.raises(TypeError, match=r"for_producer takes a TrackGenerator / "
                                        r"GateGenerator / Program, got Track: pass the "
                                        r"generator, or use ComponentContext\(track\.layout, "
                                        r"capacity\)"):
        tools.ComponentContext.for_producer(track)
    assert dict(tools.ComponentContext.for_producer(gen.program).capacity) == \
        dict(tools.ComponentContext.for_producer(gen).capacity)


def test_context_validates_its_arguments():
    lay = Layout(E, 8, DEV, True)
    with pytest.raises(TypeError, match="layout must be a track_gen.Layout"):
        tools.ComponentContext((E, 8))
    with pytest.raises(ValueError, match="capacity 'max_loop_length' must be a finite"):
        tools.ComponentContext(lay, {"max_loop_length": float("nan")})
    with pytest.raises(TypeError, match="ctx must be a LaunchContext"):
        tools.ComponentContext(lay, ctx="cpu")
    cc = tools.ComponentContext(lay, {"max_loop_length": 3.0})
    with pytest.raises(TypeError):
        cc.capacity["max_loop_length"] = 4.0                 # read-only
    assert cc.ctx.device.alias == DEV


def _cap(producer) -> dict:
    return dict(tools.ComponentContext.for_producer(producer).capacity)


def test_for_producer_capacity_keys():
    from track_gen.pipeline import StageRef, build, presets

    base = 1.5 * 384 * 0.06
    assert _cap(_track_gen(E=2)) == {"max_loop_length": base}
    assert base == pytest.approx(34.56)
    assert _cap(build(presets.track(num_envs=2, device=DEV, n_max=512, spacing=0.05))) \
        == {"max_loop_length": 1.5 * 512 * 0.05}
    # Elevation: dZ from DECLARED bounds only.
    up = _track_gen(E=2, z_profile="uniform", z_min=0.0, z_max=2.0)
    assert _cap(up) == {}                                   # z_min / z_max unbounded
    graded = _track_gen(E=2, z_profile="uniform", z_min=0.0, z_max=2.0, z_valid_grade=3.0)
    assert _cap(graded) == {"max_loop_length": base + 3.0 * base}
    tab = up.program.params["elevation"]
    tab._narrow("z_min", -0.5, None, "test")
    tab._narrow("z_max", None, 1.5, "test")
    assert _cap(up) == {"max_loop_length": base + 10 * (1.5 - -0.5)}
    noise = TrackGenerator(presets.track(num_envs=2, device=DEV, elevation=(
        "noise", {"z_min": -1.0, "z_max": 1.0, "z_noise_amplitude": 0.3,
                  "z_noise_harmonics": 3})), seeds=1)
    assert _cap(noise) == {}
    ntab = noise.program.params["elevation"]
    ntab._narrow("z_noise_amplitude", None, 0.4, "test")
    hh = 1.0 + 1.0 / 2 + 1.0 / 3
    assert _cap(noise) == {"max_loop_length": base + 4.0 * 0.4 * 3 / hh}
    ntab._narrow("z_min", -0.1, None, "test")
    ntab._narrow("z_max", None, 0.1, "test")
    assert _cap(noise) == {"max_loop_length": base + min(2 * 3 * 0.2, 4.0 * 0.4 * 3 / hh)}
    # A randomizer's range does not narrow the drawn field's declared bounds (its own
    # lo / hi are dynamic): no key.
    rnd = TrackGenerator(presets.track(num_envs=2, device=DEV, elevation=(
        "uniform", {"z_max": 1.0}), randomize={"elevation.z_min": (-0.2, 0.0),
                                               "elevation.z_max": (0.5, 1.0)}), seeds=1)
    assert _cap(rnd) == {}
    # Variable width, a plugin geometry stage, no constant-spacing resample: no key.
    prog = _track_gen(E=2).program
    fake = lambda spec: types.SimpleNamespace(spec=spec, params=prog.params,  # noqa: E731
                                              layout=prog.layout)
    wide = prog.spec.replace("width", StageRef("width", "from_channel",
                                               ports={"src": "cl.acme.hw"}))
    assert _cap(fake(wide)) == {}
    plug = prog.spec.replace("relax", StageRef("relax", "acme.smooth"))
    assert _cap(fake(plug)) == {}
    assert _cap(_gate_gen(E=2)) == {}
    assert _cap(fake(prog.spec)) == {"max_loop_length": base}
    # Only the built-in validity.default's z_valid_grade counts.
    gspec = graded.program.spec
    vref = next(r for r in gspec.stages if r.kind == "validity")
    renamed = gspec.replace("validity", dataclasses.replace(vref, name="acme.grade"))
    gp = graded.program
    assert _cap(types.SimpleNamespace(spec=renamed, params=gp.params, layout=gp.layout)) \
        == {}
    assert _cap(types.SimpleNamespace(spec=gspec, params=gp.params, layout=gp.layout)) \
        == {"max_loop_length": base + 3.0 * base}


def test_for_producer_capacity_keys_of_attribute_providers():
    """A generator that provides ``z`` (no elevation stage, ``lift.elevated``) or
    ``half_width`` (no width stage) declares no bound (its loops can be far longer than
    ``1.5 * N_max * spacing``), unless a grade bounds a provided ``z``."""
    from tests import _channel_toys as chan
    from track_gen.pipeline import StageRef, presets
    from track_gen.plugin import registry

    base = 1.5 * 384 * 0.06
    with registry.scoped():
        chan.register(registry)

        def gen(name, params, **kw):
            return TrackGenerator(presets.track(
                num_envs=2, device=DEV, generator=StageRef("generator", name, params), **kw),
                seeds=3)

        helix = gen("acme.helix_loop", {"waves": 12, "height": 2.0})
        assert ("elevation" not in {r.kind for r in helix.program.spec.stages})
        assert _cap(helix) == {}
        graded = gen("acme.helix_loop", {"waves": 12, "height": 2.0},
                     checks=(("default", {"z_valid_grade": 2.0}),))
        assert _cap(graded) == {"max_loop_length": base + 2.0 * base}
        pinch = gen("acme.pinch", {"hw_mean": 0.6, "hw_amp": 0.3})
        assert "width" not in {r.kind for r in pinch.program.spec.stages}
        assert _cap(pinch) == {}


def test_import_track_gen_imports_tools_but_no_pipeline_or_plugin():
    code = "\n".join([
        "import sys, track_gen",
        "assert 'track_gen.tools' in sys.modules",
        "bad = sorted(m for m in sys.modules if m.startswith(('track_gen.pipeline', "
        "'track_gen.plugin', 'track_gen.stages', 'track_gen.testing')))",
        "assert not bad, bad",
        "assert track_gen.tools.Progress and track_gen.HasValidity and "
        "track_gen.HasCheckpoints",
    ])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------------------
# 2-3. check_component
# ---------------------------------------------------------------------------------------

def test_the_component_check_names():
    assert CHECKS["component"][0] == "construct"
    assert len(set(CHECKS["component"])) == len(CHECKS["component"]) == 20
    for name in ("requires.complete", "invalid.sentinel", "regenerate.self_reset",
                 "capacity.clamp", "source.view", "capture.caller", "capture.self_reset"):
        assert name in CHECKS["component"]


@pytest.mark.parametrize("cid", sorted(_builtin_components()))
def test_every_builtin_component_passes_check_component(cid, track_gen_check_component):
    factory, upstream, source = _builtin_components()[cid]
    report = track_gen_check_component(factory, source=source, upstream=upstream,
                                       devices=[DEV], num_envs=E, steps=24)
    assert [r.name for r in report] == list(CHECKS["component"])
    ran = {r.name for r in report if not r.skipped}
    assert {"construct", "result.stable", "determinism", "hot_path.allocation",
            "hot_path.topology", "hot_path.buffers"} <= ran
    assert "regenerate.masked" in ran
    if cid != "disc_collision[raw]":
        assert {"requires.complete", "topology", "source.view"} <= ran
    else:   # declares no read: attached to an empty read target, then skipped
        assert report.result("requires.complete").message.startswith("reads no source")
    if cid != "gate_checkpoints":
        assert "invalid.sentinel" in ran
    comp = factory(tools.ComponentContext(Layout(E, 8, DEV, True), {}))
    if comp.stateful:
        assert {"reset.exact", "regenerate.self_reset"} <= ran
        assert report.result("reset.exact").passed
    else:
        assert "statelessness" in ran
    if cid in ("checkpoints", "props[points]", "props[segments,inner]"):
        assert report.result("capacity.clamp").details["truncated"] > 0
    if cid not in ("gate_checkpoints", "disc_collision[raw]"):
        assert report.result("regenerate.masked").details["masked_rows_changed"] > 0


@pytest.mark.parametrize("toy", list(toys.EXPECTED), ids=lambda c: c.__name__)
def test_each_broken_toy_fails_exactly_its_check(toy):
    report = check_component(lambda cc: toy(), devices=[DEV], num_envs=E, steps=16)
    assert report.failed_names == [toys.EXPECTED[toy]], str(report)


@pytest.mark.parametrize("toy", list(toys.GENERATION_WRITTEN), ids=lambda c: c.__name__)
def test_components_with_an_array_only_on_generate_writes_pass_every_check(toy):
    """The toys ``tests/test_components_cuda.py`` runs through ``capture.caller`` (an
    array that only ``on_generate`` writes; one duck-typed) are correct components."""
    report = check_component(lambda cc, t=toy: t(), devices=[DEV], num_envs=E, steps=16)
    report.raise_for_failures()
    assert report.result("regenerate.masked").details["masked_rows_changed"] > 0


def test_the_writing_a_component_toy_passes_every_check():
    report = check_component(lambda cc: toys.CurvatureLookahead(ahead=5),
                             upstream=[lambda cc: tools.Localizer(warm_window=3)],
                             devices=[DEV], num_envs=E, steps=24)
    report.raise_for_failures()
    assert not report.result("invalid.sentinel").skipped
    assert report.result("regenerate.masked").details["masked_rows_changed"] > 0
    src = open(toys.__file__).read()
    lookahead = src[src.index("# The writing-a-component toy"):]
    assert "track_gen._src" not in src and "._" not in lookahead.replace("self._", "")


def test_check_component_validates_its_arguments():
    with pytest.raises(TypeError, match="factory must be a callable"):
        check_component(tools.Progress())
    with pytest.raises(ValueError, match="num_envs must be >= 2"):
        check_component(lambda cc: tools.Checkpoints(0.5), num_envs=1)
    with pytest.raises(ValueError, match="unknown capture mode"):
        check_component(lambda cc: tools.Checkpoints(0.5), modes=("jax",))
    report = check_component(lambda cc: tools.Progress(), devices=[DEV], num_envs=E)
    assert report.failed_names == ["construct"]               # Progress needs checkpoints
    assert all(r.skipped for r in report if r.name != "construct")
    unknown = check_component(lambda cc: toys.CurvatureLookahead(),
                              upstream=[lambda cc: _SpeedInput()], devices=[DEV],
                              num_envs=E)
    assert unknown.failed_names == ["construct"]
    assert "need drive=" in unknown.result("construct").message


class _SpeedInput(tools.Localizer):
    def __init__(self):
        super().__init__()
        self.inputs = types.MappingProxyType({"position": tools.InputSpec(wp.vec3f),
                                              "speed": tools.InputSpec(wp.float32)})

    def on_bind(self, position, speed=None):
        self.utility.bind(position)


# ---------------------------------------------------------------------------------------
# 4. Byte identity for valid envs
# ---------------------------------------------------------------------------------------

def test_utility_outputs_for_valid_envs_match_the_base_goldens():
    gold = goldens.load().get(goldens.platform_key())
    if gold is None:
        pytest.skip(f"no PR14 utility digests for {goldens.platform_key()}")
    got = goldens.compute()
    assert sorted(got) == sorted(gold)
    for case, fields in gold.items():
        assert sorted(got[case]) == sorted(fields), case
        bad = [f for f in fields if got[case][f] != fields[f]]
        assert not bad, f"{case}: {bad}"


class _KernelRig:
    """Inputs of the twelve edited kernels on ``device``: a keep-invalid track batch
    (envs of :data:`_FAIL` with ``valid = 0``, geometry kept) and a gate batch."""

    def __init__(self, device: str) -> None:
        self.dev = device
        gen = _track_gen(seeds=13, device=device, z_profile="uniform", z_min=0.0,
                         z_max=0.6)
        self.t, _ = _keep_and_poison(gen.program.result, _FAIL, device)
        self.n = int(self.t.layout.stride)
        self.valid = self.t.valid.numpy() != 0
        assert not self.valid[_BAD].any() and self.valid.sum() >= 3
        self.seq, _ = _keep_and_poison(_gate_gen(device=device).program.result, _FAIL,
                                       device)
        self.gvalid = self.seq.valid.numpy() != 0

    def zeros(self, n, dtype):
        return wp.zeros(n, dtype=dtype, device=self.dev)

    def launch(self, kernel, dim, args):
        wp.launch(kernel, dim=dim, inputs=args, device=self.dev)

    def assert_valid_rows(self, olds, news, valid, label):
        for i, (a, b) in enumerate(zip(olds, news)):
            assert _same_rows(a, b, [e for e in range(E) if valid[e]]), f"{label}: output {i}"


def _base_vs_new(rig: _KernelRig, name: str) -> None:
    t, n, Z = rig.t, rig.n, rig.zeros
    old_k, new_k = BASE_KERNELS[name], None
    if name == "_scan_boundary_k":
        outs = [[Z(E * n, wp.float32), Z(E, wp.int32), Z(E, wp.float32), Z(E, wp.int32)]
                for _ in range(2)]
        rig.launch(old_k, E, [t.outer, t.count, n, 0.2, 40, *outs[0]])
        rig.launch(props_mod._scan_boundary_k, E, [t.outer, t.count, t.valid, n, 0.2, 40,
                                                   *outs[1]])
        rig.assert_valid_rows(*outs, rig.valid, name)
        return
    if name == "_place_checkpoints_k":
        M = 32
        cum, cnt, stp, tr = Z(E * n, wp.float32), Z(E, wp.int32), Z(E, wp.float32), \
            Z(E, wp.int32)
        rig.launch(props_mod._scan_boundary_k, E, [t.center, t.count, t.valid, n, 0.5, M,
                                                   cum, cnt, stp, tr])
        outs = [[Z(E * M, wp.vec3f), Z(E * M, wp.vec3f), Z(E * M, wp.vec3f),
                 Z(E * M, wp.vec3f), Z(E * M, wp.float32)] for _ in range(2)]
        common = [t.center, t.inner, t.outer, t.count, n, cum, cnt, stp, M]
        rig.launch(old_k, E * M, common + outs[0])
        ov, og = Z(E, wp.int32), Z(E, wp.int32)
        rig.launch(ckpt_mod._place_checkpoints_k, E * M,
                   common + outs[1] + [t.valid, t.generation, 1, ov, og])
        rig.assert_valid_rows(*outs, rig.valid, name)
        assert _bytes(ov) == _bytes(t.valid) and _bytes(og) == _bytes(t.generation)
        return
    if name in ("_box_query_segments_k", "_box_query_discs_k"):
        B = 2
        gates = name == "_box_query_discs_k"
        valid = rig.gvalid if gates else rig.valid
        q, he = (wp.array(_host(a), dtype=a.dtype, device=rig.dev) for a in _boxes(E * B))
        if gates:
            G = int(rig.seq.layout.stride)
            posts_h = np.empty((2 * E * G, 3), np.float32)
            posts_h[0::2], posts_h[1::2] = rig.seq.left.numpy(), rig.seq.right.numpy()
            posts = wp.array(posts_h, dtype=wp.vec3f, device=rig.dev)
            dcount = _i32(2 * np.minimum(rig.seq.count.numpy(), G), rig.dev)
        for s in range(6):
            p = np.repeat(_points(rig.seq if gates else t, s, gates=gates), B, axis=0)
            p[1::2] += np.float32([0.03 * s, -0.04, 0.0])
            if s == 3:
                p[0::4] = np.nan
            pos = _vec3(p, rig.dev)
            outs = [[Z(E * B, wp.int32), Z(E * B, wp.int32 if gates else wp.float32),
                     Z(E * B, wp.float32 if gates else wp.vec2f), Z(E * B, wp.vec2f)]
                    + ([] if gates else [Z(E * B, wp.int32)]) for _ in range(2)]
            if gates:
                rig.launch(old_k, E * B, [posts, dcount, 2 * G, 0.04, B, pos, q, he,
                                          *outs[0]])
                rig.launch(discs_mod._box_query_discs_k, E * B,
                           [posts, dcount, rig.seq.valid, 1, 2 * G, 0.04, B, pos, q, he,
                            *outs[1]])
            else:
                rig.launch(old_k, E * B, [t.inner, t.outer, t.center, t.count, n, B, pos,
                                          q, he, *outs[0]])
                rig.launch(coll_mod._box_query_segments_k, E * B,
                           [t.inner, t.outer, t.center, t.count, t.valid, 1, n, B, pos,
                            q, he, *outs[1]])
            rig.assert_valid_rows(*outs, valid, f"{name} step {s}")
        return
    if name in ("_track_aabb_k", "_sdf_bake_k"):
        R = 12
        lo, hi = Z(E, wp.vec2f), Z(E, wp.vec2f)
        boxes = [[Z(E, wp.vec2f), Z(E, wp.vec2f)] for _ in range(2)]
        rig.launch(BASE_KERNELS["_track_aabb_k"], E, [t.outer, t.count, n, -1.0, 0.1,
                                                      *boxes[0]])
        rig.launch(sdf_mod._track_aabb_k, E, [t.outer, t.count, t.valid, 1, n, -1.0, 0.1,
                                              *boxes[1]])
        rig.assert_valid_rows(*boxes, rig.valid, "_track_aabb_k")
        lo, hi = boxes[0]
        grids = [[Z(E * R * R, wp.float32), Z(E * R * R, wp.int8)] for _ in range(2)]
        rig.launch(BASE_KERNELS["_sdf_bake_k"], E * R * R, [t.inner, t.outer, t.count, n,
                                                            R, lo, hi, *grids[0]])
        rig.launch(sdf_mod._sdf_bake_k, E * R * R, [t.inner, t.outer, t.count, t.valid, 1,
                                                    n, R, lo, hi, *grids[1]])
        rig.assert_valid_rows(*grids, rig.valid, "_sdf_bake_k")
        return
    if name in ("_progress_update_k", "_localize_k"):
        sampler = CheckpointSampler(t, 0.5, max_checkpoints=32)
        c = sampler.sample()
        mask = _i32(np.arange(E) % 2 == 0, rig.dev)
        if name == "_progress_update_k":
            states = [[_vec3(np.full((E, 3), np.nan), rig.dev), Z(E, wp.int32),
                       Z(E, wp.int32), Z(E, wp.int32)] for _ in range(2)]
            seen = wp.clone(t.generation)
        else:
            states = [[wp.full(E, -1, dtype=wp.int32, device=rig.dev)] for _ in range(2)]
            seen = wp.clone(t.generation)
        for s in range(24):
            pos = _vec3(_points(t, s), rig.dev)
            if name == "_progress_update_k":
                outs = [[Z(E, wp.int32) for _ in range(7)] + [Z(E, wp.float32)]
                        for _ in range(2)]
                cp = [c.position, c.left, c.right, c.tangent, c.up_half, c.count, 32]
                rig.launch(old_k, E, cp + [pos, *states[0], *outs[0]])
                rig.launch(prog_mod._progress_update_k, E,
                           cp + [t.valid, 1, t.generation, 1, seen, pos, *states[1],
                                 *outs[1]])
                if s == 12:
                    for st in states:
                        rig.launch(prog_mod._progress_reset_k, E, [mask, *st])
            else:
                outs = [[Z(E, wp.float32), Z(E, wp.float32), Z(E, wp.float32),
                         Z(E, wp.int32)] for _ in range(2)]
                geo = [t.center, t.arclen, t.length, t.count, n, 3]
                rig.launch(old_k, E, geo + [pos, *states[0], *outs[0]])
                rig.launch(loc_mod._localize_k, E,
                           geo + [t.valid, 1, t.generation, 1, seen, pos, *states[1],
                                  *outs[1]])
            rig.assert_valid_rows(outs[0] + states[0], outs[1] + states[1], rig.valid,
                                  f"{name} step {s}")
        return
    if name in ("_curvature_raw_k", "_smooth_wrap_k", "_speed_profile_k"):
        N = E * n
        raw = [Z(N, wp.float32), Z(N, wp.float32)]
        rig.launch(BASE_KERNELS["_curvature_raw_k"], N, [t.center, t.count, n, raw[0]])
        rig.launch(loc_mod._curvature_raw_k, N, [t.center, t.count, t.valid, 1, n, raw[1]])
        rig.assert_valid_rows([raw[0]], [raw[1]], rig.valid, "_curvature_raw_k")
        sm = [Z(N, wp.float32), Z(N, wp.float32)]
        rig.launch(BASE_KERNELS["_smooth_wrap_k"], N, [raw[0], t.count, n, 2, sm[0]])
        rig.launch(loc_mod._smooth_wrap_k, N, [raw[0], t.count, t.valid, 1, n, 2, sm[1]])
        rig.assert_valid_rows([sm[0]], [sm[1]], rig.valid, "_smooth_wrap_k")
        v = [Z(N, wp.float32), Z(N, wp.float32)]
        rig.launch(BASE_KERNELS["_speed_profile_k"], E, [sm[0], t.arclen, t.length,
                                                         t.count, n, 4.0, 2.0, 3.0, 5.0,
                                                         v[0]])
        rig.launch(loc_mod._speed_profile_k, E, [sm[0], t.arclen, t.length, t.count,
                                                 t.valid, 1, n, 4.0, 2.0, 3.0, 5.0, v[1]])
        rig.assert_valid_rows([v[0]], [v[1]], rig.valid, "_speed_profile_k")
        return
    if name == "_sample_line_k":
        seq, spg = rig.seq, 8
        G = int(seq.layout.stride)
        nl = G * spg
        outs = [[Z(E * nl, wp.vec3f), Z(E, wp.int32), Z(E, wp.int32)] for _ in range(2)]
        rig.launch(old_k, E * nl, [seq.position, seq.count, seq.valid, G, spg, *outs[0]])
        gen_out = Z(E, wp.int32)
        rig.launch(line_mod._sample_line_k, E * nl, [seq.position, seq.count, seq.valid, G,
                                                     spg, *outs[1], seq.generation, 1,
                                                     gen_out])
        rig.assert_valid_rows(*outs, rig.gvalid, name)
        assert _bytes(gen_out) == _bytes(seq.generation)
        return
    raise AssertionError(f"no comparison for {name}")


_RIGS: dict = {}


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("kernel", sorted(BASE_KERNELS))
def test_edited_kernels_equal_the_base_kernels_on_valid_envs(kernel, dev):
    device = str(wp.get_device(dev))
    rig = _RIGS.get(device)
    if rig is None:
        rig = _RIGS[device] = _KernelRig(device)
    _base_vs_new(rig, kernel)
    wp.synchronize_device(device)


def test_the_frozen_base_kernels_are_the_twelve_edited_ones():
    assert sorted(BASE_KERNELS) == sorted([
        "_box_query_discs_k", "_box_query_segments_k", "_curvature_raw_k", "_localize_k",
        "_place_checkpoints_k", "_progress_update_k", "_sample_line_k", "_scan_boundary_k",
        "_sdf_bake_k", "_smooth_wrap_k", "_speed_profile_k", "_track_aabb_k"])
    src = open(sys.modules[BASE_KERNELS["_localize_k"].func.__module__].__file__).read()
    assert "9267388" in src and "VERBATIM" in src


# ---------------------------------------------------------------------------------------
# 5. Course
# ---------------------------------------------------------------------------------------

def _gen_cfg(mode, **kw):
    if mode == "track":
        return TrackGenConfig(num_envs=E, device=DEV, **kw)
    kw.setdefault("gate_width", 0.1)
    return GateGenConfig(num_envs=E, device=DEV, **kw)


class _HandWired:
    """The 0.2.0.dev5 Course's composition of the legacy utilities, hand-wired (gates
    posts given their count and valid, the frames window on the tracker's state), for
    the identity test with the adapter-driven Course."""

    def __init__(self, cfg: CourseConfig) -> None:
        from track_gen import PerEnvSeededRNG

        self.cfg = cfg
        self.rng = PerEnvSeededRNG(seeds=cfg.seeds, num_envs=E, device=DEV)
        if cfg.mode == "track":
            self.gen = TrackGenerator(cfg.gen, self.rng)
            self.result = self.gen.program.result
            self.sampler = CheckpointSampler(self.result, cfg.checkpoint_spacing,
                                             max_checkpoints=cfg.max_checkpoints)
            self.checkpoints = self.sampler.result
            self.collision = None if cfg.collision is None else CollisionChecker(
                self.result, cfg.max_boxes, cfg.collision,
                sdf_resolution=cfg.sdf_resolution or 128)
            self.heightfield = None if cfg.heightfield_resolution is None else \
                HeightFieldBaker(self.result, cfg.heightfield_resolution)
            self.line = self.localizer = self.posts = None
        else:
            self.gen = GateGenerator(cfg.gen, self.rng)
            self.result = self.gen.program.result
            self.sampler = self.heightfield = None
            self.checkpoints = CheckpointSet.from_gates(self.result)
            self.line = CourseLine(self.result, cfg.samples_per_gate)
            self.localizer = TrackLocalizer(self.line.centerline,
                                            warm_window=cfg.localize_window)
            self.posts = self.post_count = None
            self.collision = None
            if cfg.post_radius > 0.0:
                G = int(self.result.layout.stride)
                self.posts = wp.zeros(2 * E * G, dtype=wp.vec3f, device=DEV)
                self.post_count = wp.zeros(E, dtype=wp.int32, device=DEV)
                self.collision = DiscChecker(self.posts, cfg.post_radius, cfg.max_boxes,
                                             count=self.post_count, valid=self.result.valid)
            elif cfg.frame_collision:
                self.collision = FrameChecker(self.result, E, cfg.agent_radius,
                                              cfg.frame_thickness, cfg.frame_depth)
        self.progress = ProgressTracker(self.checkpoints)
        self._refresh(full=True)

    def _refresh(self, mask=None, full=False) -> None:
        if self.sampler is not None:
            self.sampler.sample()
        if self.line is not None:
            self.line.refresh()
        if isinstance(self.collision, CollisionChecker) and self.collision.method == "sdf":
            self.collision.bake()
        if self.heightfield is not None:
            self.heightfield.bake()
        if self.posts is not None:
            G = int(self.result.layout.stride)
            h = np.empty((2 * E * G, 3), np.float32)
            h[0::2], h[1::2] = self.result.left.numpy(), self.result.right.numpy()
            self.posts.assign(h)
            self.post_count.assign(2 * np.minimum(self.result.count.numpy(), G)
                                   .astype(np.int32))
        m = _i32(np.ones(E)) if full else mask
        if m is not None:
            self.reset(m)

    def bind(self, position, orientation=None, half_extents=None, box_position=None):
        self.progress.bind(position)
        if self.localizer is not None:
            self.localizer.bind(position)
        if isinstance(self.collision, FrameChecker):
            self.collision.bind_inputs(position)
            self.collision.bind_window(self.progress.next_checkpoint_state)
        elif self.collision is not None:
            self.collision.bind_inputs(box_position if box_position is not None
                                       else position, orientation, half_extents)

    def reset(self, mask) -> None:
        if self.localizer is not None:
            self.localizer.reset(mask)
        self.progress.reset(mask)

    def regenerate(self, mask=None, advance=False) -> None:
        self.gen.generate(advance=advance, mask=mask)
        self._refresh(mask=mask, full=mask is None)

    def generate(self, seeds: int) -> None:
        self.rng.set_seeds_warp(_i32(seeds + np.arange(E)), None)
        self.gen.seeds.reset_epoch()
        self.regenerate()

    def step(self):
        ev = self.progress.update()
        contacts = None if self.collision is None else self.collision.query()
        frame = None if self.localizer is None else self.localizer.query()
        return ev, contacts, frame


_COURSE_CASES = [
    ("track", dict(collision="segments", checkpoint_spacing=0.5, max_checkpoints=48)),
    ("track", dict(collision="sdf", sdf_resolution=16, heightfield_resolution=16,
                   checkpoint_spacing=0.5, max_checkpoints=48)),
    ("track", dict(collision="segments", max_boxes=2, checkpoint_spacing=0.4,
                   max_checkpoints=64)),
    ("track", dict(collision=None, checkpoint_spacing=0.5, max_checkpoints=48)),
    ("gates", dict(post_radius=0.03, localize_window=3)),
    ("gates", dict(post_radius=0.03, max_boxes=2)),
    ("gates", dict(frame_collision=True, agent_radius=0.05, frame_thickness=0.02,
                   frame_depth=0.02)),
    ("gates", dict()),
]


def _snap_course(c, sr) -> dict:
    from track_gen.testing.capture import result_arrays

    out = {f"result.{k}": _bytes(v) for k, v in result_arrays(c.result).items()}
    out.update({f"cp.{k}": _bytes(v) for k, v in result_arrays(c.checkpoints).items()})
    for label, obj in zip(("events", "contacts", "frame"), sr):
        if obj is not None:
            out.update({f"{label}.{k}": _bytes(v) for k, v in result_arrays(obj).items()})
    if c.heightfield is not None:
        out.update({f"hf.{k}": _bytes(v) for k, v in result_arrays(
            c.heightfield.result).items()})
    line = getattr(c, "course_line", None) or getattr(c, "line", None)
    if line is not None:
        out.update({f"line.{k}": _bytes(v) for k, v in result_arrays(
            line.centerline).items()})
    p = c.progress
    out["progress.state"] = b"".join(_bytes(a) for a in (p._prev_pos, p._next, p._laps,
                                                          p._progress, p._seen))
    if c.localizer is not None:
        out["localizer.state"] = _bytes(c.localizer._last) + _bytes(c.localizer._seen)
    return out


@pytest.mark.parametrize("mode,kw", _COURSE_CASES,
                         ids=[f"{m}-{'-'.join(sorted(k)) or 'plain'}" for m, k in
                              _COURSE_CASES])
def test_course_equals_the_hand_wired_legacy_composition(mode, kw):
    cfg = CourseConfig(mode=mode, gen=_gen_cfg(mode), seeds=11, **kw)
    a, b = Course(cfg), _HandWired(cfg)
    B = int(kw.get("max_boxes", 1))
    bufs = []
    for c in (a, b):
        pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
        bp = wp.zeros(E * B, dtype=wp.vec3f, device=DEV) if B > 1 else None
        q, he = _boxes(E * B)
        if a._needs_boxes():
            c.bind(position=pos, orientation=q, half_extents=he, box_position=bp)
        else:
            c.bind(position=pos)
        bufs.append((pos, bp))
    gates = mode == "gates"
    for s in range(40):
        for c, (pos, bp) in zip((a, b), bufs):
            src = c.course_line.centerline if gates and c is a else \
                (c.line.centerline if gates else c.result)
            x = _points(src, s)
            pos.assign(x)
            if bp is not None:
                bp.assign(np.repeat(x, B, axis=0) + np.float32([0.02, 0.0, 0.0]))
        sa = a.step()
        sb = b.step()
        got, ref = _snap_course(a, (sa.events, sa.contacts, sa.frame)), _snap_course(b, sb)
        bad = [k for k in ref if got.get(k) != ref[k]]
        assert not bad, f"step {s}: {bad}"
        for c in (a, b):
            if s == 9:
                c.regenerate(mask=_i32([1, 0, 1, 0, 0, 1]), advance=True)
            if s == 17:
                c.reset(_i32([0, 1, 1, 0, 1, 0]))
            if s == 23:
                c.regenerate()
            if s == 29:
                c.generate(seeds=77)
            if s == 33:
                c.regenerate(mask=_i32([1, 1, 0, 0, 0, 1]), advance=True)


def test_step_result_carries_valid():
    c = Course(CourseConfig(mode="track", gen=_gen_cfg("track"), checkpoint_spacing=0.5))
    c.bind(position=wp.zeros(E, dtype=wp.vec3f, device=DEV))
    sr = c.step()
    assert sr.valid is c.result.valid and c.step() is sr
    snap = sr.clone()
    assert snap.valid is not sr.valid and _bytes(snap.valid) == _bytes(sr.valid)
    assert StepResult(sr.events, None, None).valid is None          # positional: unchanged
    assert StepResult(sr.events, None, None).clone().valid is None


def test_gates_course_posts_get_the_gate_count_and_validity():
    c = Course(CourseConfig(mode="gates", gen=_gen_cfg("gates"), seeds=7, post_radius=0.03))
    G = int(c.result.layout.stride)
    assert c.collision._count is c._components[3].post_count
    assert c.collision._valid is c.result.valid
    np.testing.assert_array_equal(c.collision._count.numpy(),
                                  2 * np.minimum(c.result.count.numpy(), G))
    assert _bytes(c._posts) == _bytes(c._by_name["collision"].posts)
    h = c._posts.numpy()
    np.testing.assert_array_equal(h[0::2], c.result.left.numpy())
    np.testing.assert_array_equal(h[1::2], c.result.right.numpy())


def test_course_line_generation_follows_the_gates():
    c = Course(CourseConfig(mode="gates", gen=_gen_cfg("gates"), seeds=7))
    line = c.course_line.centerline
    assert line.generation is not None and line.generation is not c.result.generation
    m = _i32([0, 1, 0, 1, 1, 0])
    c.regenerate(mask=m, advance=True)
    np.testing.assert_array_equal(line.generation.numpy(), [0, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(line.generation.numpy(), c.result.generation.numpy())
    assert c.localizer._track.generation is line.generation


@pytest.mark.parametrize("cs,want", [(0.5, 70), (0.6, 58), (0.3, 116), (1.0, 35)])
def test_default_checkpoint_capacity_is_the_plan_bound(cs, want):
    c = Course(CourseConfig(mode="track", gen=_gen_cfg("track"), checkpoint_spacing=cs))
    assert c.checkpoint_sampler.max_checkpoints == want
    assert c.checkpoints.position.shape == (E * want,)
    assert want == max(3, math.ceil(1.5 * 384 * 0.06 / cs))


def test_default_capacity_keeps_every_real_row_of_the_readback_capacity():
    cfg = dict(mode="track", gen=_gen_cfg("track"), seeds=3, checkpoint_spacing=0.5)
    new = Course(CourseConfig(**cfg))
    old_m = ckpt_mod._readback_max_checkpoints(new.result, E, 384, 0.5)
    old = Course(CourseConfig(**cfg, max_checkpoints=old_m))
    assert old_m < 70 and not old.checkpoint_sampler.truncated.numpy().any()
    cnt = new.checkpoints.count.numpy()
    np.testing.assert_array_equal(cnt, old.checkpoints.count.numpy())
    for f in ("position", "left", "right", "tangent", "up_half"):
        a = _host(getattr(new.checkpoints, f)).reshape(E, 70, -1)
        b = _host(getattr(old.checkpoints, f)).reshape(E, old_m, -1)
        for e in range(E):
            assert a[e, :cnt[e]].tobytes() == b[e, :cnt[e]].tobytes(), (f, e)


def test_elevated_default_keeps_the_readback():
    gen = _gen_cfg("track", z_profile="uniform", z_min=0.0, z_max=2.0)
    c = Course(CourseConfig(mode="track", gen=gen, seeds=0, checkpoint_spacing=0.5))
    want = ckpt_mod._readback_max_checkpoints(c.result, E, 384, 0.5)
    assert c.checkpoint_sampler.max_checkpoints == want < 70
    graded = _gen_cfg("track", z_profile="uniform", z_min=0.0, z_max=2.0, z_valid_grade=3.0)
    g = Course(CourseConfig(mode="track", gen=graded, seeds=0, checkpoint_spacing=0.5))
    assert g.checkpoint_sampler.max_checkpoints == math.ceil(4 * 1.5 * 384 * 0.06 / 0.5)


def test_a_clipping_default_capacity_warns_once_at_construction(monkeypatch):
    from track_gen.tools import _component

    monkeypatch.setattr(_component, "_plan_capacity", lambda prog: {"max_loop_length": 2.0})
    cfg = dict(mode="track", gen=_gen_cfg("track"), seeds=3, checkpoint_spacing=0.5)
    with pytest.warns(UserWarning, match=r"^Course: the default max_checkpoints=4 \(the "
                                         r"plan bound .*\) clips the checkpoint ring of "
                                         r"\d+ valid env\(s\)") as rec:
        Course(CourseConfig(**cfg))
    hits = [w for w in rec if "clips the checkpoint ring" in str(w.message)]
    assert len(hits) == 1 and hits[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Course(CourseConfig(**cfg, max_checkpoints=4))         # explicit: silent


def test_course_components_are_driven_in_the_legacy_order():
    c = Course(CourseConfig(mode="gates", gen=_gen_cfg("gates"), post_radius=0.03))
    assert [type(x).__name__ for x in c._components] == [
        "GateCheckpoints", "CenterlineFromGates", "Progress", "DiscCollision", "Localizer"]
    c.bind(position=wp.zeros(E, dtype=wp.vec3f, device=DEV), orientation=_boxes(E)[0],
           half_extents=_boxes(E)[1])
    with trace_launches() as tr:
        c.step()
    assert [k.split(".")[-1] for k in tr.kernels()] == [
        "_progress_update_k", "_box_query_discs_k", "_localize_k"]
    with trace_launches() as tr:
        c.reset(_i32(np.ones(E)))
    assert [k.split(".")[-1] for k in tr.kernels()] == ["_localize_reset_k",
                                                         "_progress_reset_k"]


# ---------------------------------------------------------------------------------------
# 6. Generation-aware self-reset
# ---------------------------------------------------------------------------------------

def _plain_set(cps):
    """``cps`` without ``valid`` / ``generation`` (never self-resets, never frozen)."""
    return CheckpointSet(cps.position, cps.left, cps.right, cps.tangent, cps.up_half,
                         cps.count)


def test_self_reset_equals_an_explicit_reset():
    gen = _track_gen(seeds=3)
    t = gen.program.result
    sampler = CheckpointSampler(t, 0.5, max_checkpoints=48)
    cps = sampler.sample()
    no_gen_src = types.SimpleNamespace(center=t.center, arclen=t.arclen, length=t.length,
                                       count=t.count, valid=t.valid)
    A = (ProgressTracker(_plain_set(cps)), TrackLocalizer(no_gen_src, warm_window=4))
    B = (ProgressTracker(cps), TrackLocalizer(t, warm_window=4))
    C = (ProgressTracker(cps), TrackLocalizer(t, warm_window=4))
    D = (ProgressTracker(_plain_set(cps)), TrackLocalizer(no_gen_src, warm_window=4))
    regen = {10: [1, 0, 1, 0, 0, 1], 25: [0, 1, 1, 0, 1, 0], 31: [1] * E}
    differs = 0
    for s in range(45):
        if s in regen:
            m = _i32(regen[s])
            gen.generate(advance=True, mask=m)
            sampler.sample()
            for tr, loc in (A, C):          # the 0.x contract: explicit reset
                loc.reset(m)
                tr.reset(m)
        pos = _vec3(_points(t, s))
        outs = []
        for tr, loc in (A, B, C, D):        # D: no generation, no reset
            ev, fr = tr.update(pos), loc.query(pos)
            outs.append([_bytes(getattr(o, f.name)) for o in (ev, fr)
                         for f in dataclasses.fields(o)])
        assert outs[0] == outs[1] == outs[2], f"step {s}"
        differs += sum(a != d for a, d in zip(outs[0], outs[3]))
    assert differs > 0, "vacuous: without any reset the trajectories never differ"
    np.testing.assert_array_equal(B[0]._seen.numpy(), t.generation.numpy())


def test_self_reset_follows_the_sampled_ring_not_the_regeneration():
    gen = _track_gen(seeds=3)
    t = gen.program.result
    sampler = CheckpointSampler(t, 0.5, max_checkpoints=48)
    tr = ProgressTracker(sampler.sample())
    np.testing.assert_array_equal(tr._seen.numpy(), sampler.result.generation.numpy())
    for s in range(30):                     # accumulate progress
        tr.update(_vec3(_points(t, s)))
    before = _host(tr._progress)
    assert before.any()
    gen.generate(advance=True, mask=_i32([1, 0, 0, 0, 0, 0]))
    tr.update(_vec3(_points(t, 30)))        # set not resampled: no reset yet
    assert (_host(tr._progress)[0] >= before[0])
    sampler.sample()
    tr.update(_vec3(_points(t, 31)))        # the first update after sample(): env 0 resets
    assert _host(tr._progress)[0] == 0 and _host(tr._seen)[0] == 1
    assert (_host(tr._progress)[1:] >= before[1:]).all()


def test_reseeds_and_aba_follow_the_epoch():
    gen = _track_gen(seeds=3)
    t = gen.program.result
    sampler = CheckpointSampler(t, 0.5, max_checkpoints=48)
    tr = ProgressTracker(sampler.sample())

    def run(start, n):
        for s in range(start, start + n):
            tr.update(_vec3(_points(t, s)))
        return _host(tr._progress)

    p0 = run(0, 25)
    assert p0.any()
    # A bare reseed (epochs stay 0): the generation does not change, no self-reset (the
    # documented caveat: reset(mask) after a reseed).
    gen.seeds.set(_i32(100 + np.arange(E)))
    gen.generate()
    sampler.sample()
    assert (run(25, 1) >= p0).all()
    # An advance changes every generation: every env resets by itself.
    gen.generate(advance=True)
    sampler.sample()
    assert (run(26, 1) == 0).all()
    p1 = run(27, 25)
    assert p1.any()
    # ABA: reset_epoch() + the same number of advances leaves generation equal.
    gen.seeds.reset_epoch()
    gen.generate(advance=True)
    sampler.sample()
    np.testing.assert_array_equal(t.generation.numpy(), np.ones(E))
    assert (run(52, 1) >= p1).all()
    # Course.generate(seeds=) after advances zeroes the epochs: a generation change.
    c = Course(CourseConfig(mode="track", gen=_gen_cfg("track"), checkpoint_spacing=0.5))
    c.regenerate(advance=True)
    assert (_host(c.checkpoints.generation) == 1).all()
    c.generate(seeds=5)
    assert (_host(c.checkpoints.generation) == 0).all()


def test_a_set_without_generation_never_self_resets():
    gen = _track_gen(seeds=3)
    t = gen.program.result
    sampler = CheckpointSampler(t, 0.5, max_checkpoints=48)
    tr = ProgressTracker(_plain_set(sampler.sample()))
    for s in range(25):
        tr.update(_vec3(_points(t, s)))
    p = _host(tr._progress)
    gen.generate(advance=True)
    sampler.sample()
    tr.update(_vec3(_points(t, 25)))
    assert (_host(tr._progress) >= p).all()   # the 0.x behaviour: callers must reset


# ---------------------------------------------------------------------------------------
# 7. Invalid-env sentinels
# ---------------------------------------------------------------------------------------

def _kp_track():
    gen = _track_gen(seeds=13, z_profile="uniform", z_min=0.0, z_max=0.5)
    return _keep_and_poison(gen.program.result, _FAIL)


def _kp_gates():
    return _keep_and_poison(_gate_gen().program.result, _FAIL)


def _same(objs_k, objs_p, label):
    for a, b in zip(objs_k, objs_p):
        for f in dataclasses.fields(a):
            x, y = getattr(a, f.name), getattr(b, f.name)
            if isinstance(x, wp.array):
                assert _bytes(x) == _bytes(y), f"{label}: {f.name}"


def test_track_utility_sentinels_and_k_equals_p():
    K, P = _kp_track()
    n = int(K.layout.stride)
    out = {}
    for label, src in (("K", K), ("P", P)):
        res = {}
        res["props"] = PropSampler(src, 0.2, max_props=64).sample()
        cs = CheckpointSampler(src, 0.5, max_checkpoints=48)
        res["cps"] = cs.sample()
        res["cps_diag"] = (_bytes(cs.truncated), _bytes(cs.step))
        res["hf"] = HeightFieldBaker(src, 16).bake()
        seg = CollisionChecker(src, 2)
        sdf = CollisionChecker(src, 2, "sdf", sdf_resolution=16)
        res["sdf_grid"] = (_bytes(sdf._sdf_phi), _bytes(sdf._sdf_bid), _bytes(sdf._sdf_lo),
                           _bytes(sdf._sdf_hi))
        p = np.repeat(_points(K, 3), 2, axis=0)
        p[3::4] = np.nan                                     # inactive slots
        q, he = _boxes(2 * E)
        res["seg"] = seg.query(_vec3(p), q, he).clone()
        res["sdf"] = sdf.query(_vec3(p), q, he).clone()
        res["loc"] = TrackLocalizer(src).query(_vec3(_points(K, 2))).clone()
        res["kappa"] = curvature(src)
        res["speed"] = speed_profile(src, 4.0, 2.0, 3.0, 5.0)
        tr = ProgressTracker(res["cps"])
        for s in range(6):
            ev = tr.update(_vec3(_points(K, s)))
        res["ev"] = ev
        out[label] = res
    k, pp = out["K"], out["P"]
    _same([k["props"], k["cps"], k["hf"], k["seg"], k["sdf"], k["loc"], k["ev"]],
          [pp["props"], pp["cps"], pp["hf"], pp["seg"], pp["sdf"], pp["loc"], pp["ev"]],
          "K == P")
    assert k["cps_diag"] == pp["cps_diag"] and k["sdf_grid"] == pp["sdf_grid"]
    assert _bytes(k["kappa"]) == _bytes(pp["kappa"])
    assert _bytes(k["speed"]) == _bytes(pp["speed"])
    # The table values (roadmap PR14 sentinel table).
    bad = _BAD
    assert (k["props"].count.numpy()[bad] == 0).all()
    assert np.isnan(k["props"].step.numpy()[bad]).all()
    assert (k["props"].truncated.numpy()[bad] == 0).all()
    assert np.isnan(_rows(k["props"].position)[bad]).all()
    assert (k["cps"].count.numpy()[bad] == 0).all() and (k["cps"].valid.numpy()[bad] == 0).all()
    assert np.isnan(_rows(k["hf"].height)[bad]).all()
    np.testing.assert_array_equal(k["hf"].lo.numpy()[bad], -np.ones((len(bad), 2)))
    np.testing.assert_array_equal(k["hf"].hi.numpy()[bad], np.ones((len(bad), 2)))
    for c in (k["seg"], k["sdf"]):
        oob, bnd = _rows(c.oob), _rows(c.boundary)
        dist = _rows(c.distance)
        for e in bad:
            for j in (0, 1):
                if (2 * e + j) % 4 == 3:            # inactive slot: its own rule
                    assert oob[e, j] == 0 and bnd[e, j] == -1
                else:
                    assert oob[e, j] == 1 and bnd[e, j] == -1 and np.isnan(dist[e, j])
        assert np.isnan(_rows(c.nearest)[bad]).all() and np.isnan(_rows(c.normal)[bad]).all()
    loc = k["loc"]
    assert np.isnan(loc.s.numpy()[bad]).all() and np.isnan(loc.n.numpy()[bad]).all()
    assert np.isnan(loc.n_up.numpy()[bad]).all() and (loc.segment.numpy()[bad] == -1).all()
    assert np.isnan(_rows(k["kappa"])[bad]).all() and np.isnan(_rows(k["speed"])[bad]).all()
    ev = k["ev"]
    assert (ev.passed.numpy()[bad] == 0).all() and (ev.checkpoint_passed.numpy()[bad] == -1).all()
    assert (ev.wrong_way.numpy()[bad] == 0).all() and (ev.wrong_checkpoint.numpy()[bad] == -1).all()
    assert (ev.dist_to_next.numpy()[bad] == 0.0).all()
    good = _GOOD
    assert np.isfinite(ev.dist_to_next.numpy()[good]).all()
    assert (k["cps"].count.numpy()[good] > 0).all()
    del n


def test_gate_utility_sentinels_and_k_equals_p():
    K, P = _kp_gates()
    out = {}
    for label, src in (("K", K), ("P", P)):
        res = {}
        line = CourseLine(src, 8)
        line.refresh()
        res["line"] = line.centerline
        G = int(src.layout.stride)
        h = np.empty((2 * E * G, 3), np.float32)
        h[0::2], h[1::2] = K.left.numpy(), K.right.numpy()      # K's posts in both
        posts = _vec3(h)
        cnt = _i32(2 * np.minimum(K.count.numpy(), G))
        q, he = _boxes(E)
        at_gate = _vec3(K.position.numpy().reshape(E, G, 3)[:, 1])   # on gate 1's center
        res["discs"] = DiscChecker(posts, 0.2, 1, count=cnt, valid=src.valid,
                                   position=at_gate, orientation=q,
                                   half_extents=he).query()
        pos = _vec3(_points(K, 1, gates=True))
        tr = ProgressTracker(CheckpointSet.from_gates(src))
        fc = FrameChecker(src, E, 0.05, 0.02, 0.02)
        fc.bind_inputs(pos)
        fc.bind_window(tr.result.next_checkpoint)
        for s in range(8):
            pos.assign(_points(K, s, gates=True))
            ev = tr.update(pos)
            fr = fc.query()
        res["ev"], res["frames"] = ev, fr
        res["loc"] = TrackLocalizer(line.centerline).query(pos)
        out[label] = res
    k, pp = out["K"], out["P"]
    for name in ("discs", "ev", "frames", "loc"):
        _same([k[name]], [pp[name]], f"K == P: {name}")
    for f in ("center", "tangent", "arclen", "length", "count", "valid"):
        assert _bytes(getattr(k["line"], f)) == _bytes(getattr(pp["line"], f)), f
    bad = _BAD
    d = k["discs"]
    assert (d.hit.numpy()[bad] == 0).all() and (d.disc.numpy()[bad] == -1).all()
    assert (d.depth.numpy()[bad] == 0).all() and np.isnan(d.nearest.numpy()[bad]).all()
    assert d.hit.numpy()[_GOOD].any(), "vacuous: no valid env hits a post"
    assert (k["frames"].hit.numpy()[bad] == 0).all()
    assert (k["ev"].dist_to_next.numpy()[bad] == 0).all()
    assert (k["line"].count.numpy()[bad] == 0).all() and (k["line"].valid.numpy()[bad] == 0).all()


def test_poison_mode_dist_to_next_is_zero_and_a_hand_built_empty_set_stays_nan():
    _, P = _kp_track()
    cs = CheckpointSampler(P, 0.5, max_checkpoints=48)
    ev = ProgressTracker(cs.sample()).update(_vec3(np.zeros((E, 3))))
    assert (ev.dist_to_next.numpy()[_BAD] == 0.0).all()
    plain = _plain_set(cs.result)                       # no valid: "no checkpoints" NaN
    ev = ProgressTracker(plain).update(_vec3(np.zeros((E, 3))))
    assert np.isnan(ev.dist_to_next.numpy()[_BAD]).all()


def test_a_revalidated_env_emits_no_event_on_its_first_update():
    gen = _track_gen(seeds=3)
    t = gen.program.result
    cps = CheckpointSampler(t, 0.5, max_checkpoints=48).sample()
    valid = wp.clone(cps.valid)
    s = CheckpointSet(cps.position, cps.left, cps.right, cps.tangent, cps.up_half,
                      cps.count, valid=valid)
    tr = ProgressTracker(s)
    # Find a crossing step for env 0, then invalidate it just before.
    tr_ref = ProgressTracker(s.clone())
    cross = None
    for st in range(60):
        if tr_ref.update(_vec3(_points(t, st))).passed.numpy()[0]:
            cross = st
            break
    assert cross is not None and cross >= 1
    for st in range(cross - 1):
        tr.update(_vec3(_points(t, st)))
    v = valid.numpy()
    v[0] = 0
    valid.assign(v)
    tr.update(_vec3(_points(t, cross - 1)))          # frozen, prev_pos re-armed
    v[0] = 1
    valid.assign(v)
    ev = tr.update(_vec3(_points(t, cross)))          # would cross: no event
    assert ev.passed.numpy()[0] == 0 and ev.wrong_way.numpy()[0] == 0
    assert np.isfinite(tr._prev_pos.numpy()[0]).all()  # re-primed by this update


# ---------------------------------------------------------------------------------------
# 8. Capacities and the resolution warning
# ---------------------------------------------------------------------------------------

def test_components_need_a_capacity_or_an_explicit_one():
    gen = _gate_gen()
    t = _track_gen().program.result
    cc = tools.ComponentContext(t.layout)                     # no capacity
    for comp, what in ((tools.Props(0.2), "max_props"),
                       (tools.Checkpoints(0.5), "max_checkpoints")):
        with pytest.raises(ValueError, match=rf"{what}=None needs the producer's capacity "
                                             rf"'max_loop_length'"):
            cc.add(comp, t)
    p = cc.add(tools.Props(0.2, max_props=17), t)
    c = cc.add(tools.Checkpoints(0.5, max_checkpoints=9), t)
    assert (p.utility.max_props, c.utility.max_checkpoints) == (17, 9)
    cc2 = tools.ComponentContext.for_producer(_track_gen())
    p2 = cc2.add(tools.Props(0.2), t)
    assert p2.utility.max_props == math.ceil(1.5 * 384 * 0.06 / 0.2)
    with pytest.raises(ValueError, match="max_checkpoints must be >= 3"):
        tools.Checkpoints(0.5, max_checkpoints=2)
    del gen


def test_legacy_samplers_expose_their_capacity_and_keep_the_readback():
    t = _track_gen().program.result
    ps = PropSampler(t, 0.2)
    cs = CheckpointSampler(t, 0.5)
    assert ps.max_props == ps._M and cs.max_checkpoints == cs._M
    assert cs.max_checkpoints == ckpt_mod._readback_max_checkpoints(t, E, 384, 0.5)
    with pytest.raises(AttributeError):
        cs.max_checkpoints = 3


@pytest.mark.parametrize("make,bpt", [
    (lambda **kw: tools.BandCollision("sdf", sdf_resolution=96, **kw), 5),
    (lambda **kw: tools.HeightFieldBake(96, **kw), 4)], ids=["sdf", "heightfield"])
def test_the_grid_resolution_warning(make, bpt):
    gen = _track_gen()
    t = gen.program.result
    cc = tools.ComponentContext.for_producer(gen)
    need = math.ceil(128 * 1.5)
    mib = E * need * need * bpt / 2 ** 20
    with pytest.warns(UserWarning, match=rf"resolution 96 does not keep the accuracy of "
                                         rf"128 at size 1 up to size_hi=1.5: .*resolution >= "
                                         rf"{need} \(~{mib:.0f} MiB for {E} envs\)") as rec:
        cc.add(make(size_hi=1.5, name="a"), t)
    assert rec[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cc.add(make(size_hi=1.0, name="b"), t)                       # size <= 1: never
        cc.add(make(size_hi=1.5, reference_resolution=64, name="c"), t)  # accepted
        cc.add(make(name="d"), t)                                     # no size declared
        cc.add(tools.BandCollision("segments", size_hi=4.0, name="e"), t)
    with pytest.raises(ValueError, match="size_hi must be > 0"):
        make(size_hi=0.0)


# ---------------------------------------------------------------------------------------
# 9. out= for curvature / speed_profile
# ---------------------------------------------------------------------------------------

def _sources():
    t = _track_gen(z_profile="uniform", z_min=0.0, z_max=0.5).program.result
    K, _ = _keep_and_poison(t, _FAIL)
    line = CourseLine(_gate_gen().program.result, 4)
    line.refresh()
    return {"track": t, "keep-invalid": K, "centerline": line.centerline,
            "minimal": MinimalCenterline(t)}


@pytest.mark.parametrize("name", ["track", "keep-invalid", "centerline", "minimal"])
def test_out_forms_equal_the_allocating_forms_and_allocate_nothing(name):
    src = _sources()[name]
    n = int(src.center.shape[0])
    k, sc, v = (wp.empty(n, dtype=wp.float32, device=DEV) for _ in range(3))
    ctx = LaunchContext.for_call(DEV)
    for window in (0, 2):
        ref = _bytes(curvature(src, window))
        curvature(src, window, ctx, out=k, scratch=sc)              # warm
        with hot_path_guard():
            assert curvature(src, window, ctx, out=k, scratch=sc) is k
        assert _bytes(k) == ref
    ref_v = _bytes(speed_profile(src, 4.0, 2.0, 3.0, 5.0, kappa=k))
    with hot_path_guard():
        assert speed_profile(src, 4.0, 2.0, 3.0, 5.0, kappa=k, ctx=ctx, out=v) is v
    assert _bytes(v) == ref_v
    ref_own = _bytes(speed_profile(src, 4.0, 2.0, 3.0, 5.0, window=2))
    assert _bytes(speed_profile(src, 4.0, 2.0, 3.0, 5.0, kappa=curvature(src, 2))) == ref_own
    with trace_launches() as t1:
        curvature(src, 2, ctx, out=k, scratch=sc)
    with trace_launches() as t2:
        curvature(src, 2, ctx, out=k, scratch=sc)
    assert t1.same_topology(t2)


def test_out_form_errors():
    t = _track_gen().program.result
    n = int(t.center.shape[0])
    k, sc = wp.empty(n, dtype=wp.float32, device=DEV), wp.empty(n, dtype=wp.float32,
                                                                device=DEV)
    with pytest.raises(ValueError, match=r"curvature\(out=..., window > 0\) needs scratch="):
        curvature(t, 2, out=k)
    curvature(t, 0, out=k)                                   # window 0: no scratch needed
    with pytest.raises(ValueError, match="scratch must not overlap out"):
        curvature(t, 2, out=k, scratch=k)
    big = wp.empty(2 * n, dtype=wp.float32, device=DEV)
    with pytest.raises(ValueError, match="scratch must not overlap out"):
        curvature(t, 2, out=big[:n], scratch=big[n // 2:n // 2 + n])
    curvature(t, 2, out=big[:n], scratch=big[n:])             # adjacent: fine
    with pytest.raises(ValueError, match=r"out must have shape"):
        curvature(t, 0, out=wp.empty(n - 1, dtype=wp.float32, device=DEV))
    with pytest.raises(ValueError, match=r"out must have dtype float32"):
        curvature(t, 0, out=wp.empty(n, dtype=wp.int32, device=DEV))
    with pytest.raises(ValueError, match=r"scratch must have shape"):
        curvature(t, 2, out=k, scratch=wp.empty(3, dtype=wp.float32, device=DEV))
    with pytest.raises(ValueError, match=r"speed_profile\(out=...\) needs kappa=: compute "
                                         r"it with curvature\(track, out=k, scratch=s\)"):
        speed_profile(t, 4.0, 2.0, 3.0, 5.0, out=sc)
    with pytest.raises(ValueError, match="out must not overlap kappa"):
        speed_profile(t, 4.0, 2.0, 3.0, 5.0, kappa=k, out=k)
    with pytest.raises(ValueError, match="out must be contiguous"):
        curvature(t, 0, out=wp.empty(2 * n, dtype=wp.float32, device=DEV)[::2])


# ---------------------------------------------------------------------------------------
# 10. ctx= and .result
# ---------------------------------------------------------------------------------------

class _CountingCtx(LaunchContext):
    calls: list

    def launch(self, kernel, dim, inputs=(), outputs=(), **kw):
        self.calls.append(kernel.key)
        super().launch(kernel, dim, inputs, outputs, **kw)


def _legacy_hot_calls():
    """``(label, utility, hot(ctx) -> returned)`` for every legacy hot method."""
    gen = _track_gen()
    t = gen.program.result
    seq = _gate_gen().program.result
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    q, he = _boxes(E)
    m = _i32(np.ones(E))
    seg = CollisionChecker(t, 1, position=pos, orientation=q, half_extents=he)
    sdf = CollisionChecker(t, 1, "sdf", sdf_resolution=16, position=pos, orientation=q,
                           half_extents=he)
    cs = CheckpointSampler(t, 0.5, max_checkpoints=48)
    ps = PropSampler(t, 0.2, max_props=64)
    hf = HeightFieldBaker(t, 16)
    tr = ProgressTracker(cs.result, position=pos)
    loc = TrackLocalizer(t, warm_window=2, position=pos)
    line = CourseLine(seq, 4)
    G = int(seq.layout.stride)
    discs = DiscChecker(wp.zeros(2 * E * G, dtype=wp.vec3f, device=DEV), 0.05, 1,
                        num_envs=E, position=pos, orientation=q, half_extents=he)
    fc = FrameChecker(seq, E, 0.05, 0.02, 0.02)
    fc.bind_inputs(pos)
    fc.bind_window(tr.result.next_checkpoint)
    return [
        ("CollisionChecker.query", seg, lambda c: seg.query(ctx=c)),
        ("CollisionChecker.bake", sdf, lambda c: sdf.bake(ctx=c)),
        ("CollisionChecker[sdf].query", sdf, lambda c: sdf.query(ctx=c)),
        ("CheckpointSampler.sample", cs, lambda c: cs.sample(ctx=c)),
        ("PropSampler.sample", ps, lambda c: ps.sample(ctx=c)),
        ("HeightFieldBaker.bake", hf, lambda c: hf.bake(ctx=c)),
        ("ProgressTracker.update", tr, lambda c: tr.update(ctx=c)),
        ("ProgressTracker.reset", tr, lambda c: tr.reset(m, ctx=c)),
        ("TrackLocalizer.query", loc, lambda c: loc.query(ctx=c)),
        ("TrackLocalizer.reset", loc, lambda c: loc.reset(m, ctx=c)),
        ("CourseLine.refresh", line, lambda c: line.refresh(ctx=c)),
        ("DiscChecker.query", discs, lambda c: discs.query(ctx=c)),
        ("FrameChecker.query", fc, lambda c: fc.query(ctx=c)),
    ]


def test_every_legacy_hot_method_takes_ctx_and_launches_only_through_it():
    for label, util, hot in _legacy_hot_calls():
        ctx = _CountingCtx(DEV, None)
        ctx.calls = []
        with trace_launches() as tr:
            got = hot(ctx)
        assert len(tr) == len(ctx.calls) > 0, label
        assert all(r.stream is None for r in tr), label
        if got is not None:
            assert got is util.result, f"{label}: .result is the returned instance"
        with pytest.raises(TypeError, match="pass stream= or ctx=, not both"):
            _with_stream_and_ctx(util, label)
        other = LaunchContext.for_call(DEV)
        other.device = types.SimpleNamespace(alias="cuda:7")
        with pytest.raises(ValueError, match=r"ctx is on 'cuda:7'"):
            hot(other)
        with pytest.raises(TypeError, match="ctx must be a track_gen.exec.LaunchContext"):
            hot("cpu")


def _with_stream_and_ctx(util, label):
    ctx = LaunchContext.for_call(DEV)
    name = label.split(".")[-1]
    fn = getattr(util, name)
    if name == "reset":
        fn(_i32(np.ones(E)), stream=object(), ctx=ctx)
    else:
        fn(stream=object(), ctx=ctx)


def test_result_properties_and_method():
    t = _track_gen().program.result
    c = CollisionChecker(t, 1, "sdf", sdf_resolution=8)
    assert c.method == "sdf" and c.result is c._contact
    with pytest.raises(AttributeError):
        c.method = "segments"
    assert CollisionChecker(t, 1).method == "segments"
    line = CourseLine(_gate_gen().program.result)
    assert line.result is line.centerline


# ---------------------------------------------------------------------------------------
# 11. RebindAfterCaptureWarning
# ---------------------------------------------------------------------------------------

def _rebind_hits(rec):
    return [w for w in rec if issubclass(w.category, RebindAfterCaptureWarning)]


def test_rebind_after_a_recorded_update_warns_once_at_the_callers_line():
    t = _track_gen().program.result
    pos, pos2 = (wp.zeros(E, dtype=wp.vec3f, device=DEV) for _ in range(2))
    tr = ProgressTracker(CheckpointSampler(t, 0.5, max_checkpoints=48).sample(),
                         position=pos)
    tr.update()
    assert tr._recorded is False                      # eager: never armed
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tr.bind(pos2)
        tr.bind(pos)
    assert not _rebind_hits(rec)
    tr.update(ctx=_Capturing(DEV, None))
    assert tr._recorded is True
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tr.bind(pos)                                  # the same array: silent
        line = sys._getframe().f_lineno + 1
        tr.bind(pos2)
        tr.bind(pos)                                  # cleared: silent
    hits = _rebind_hits(rec)
    assert len(hits) == 1 and (hits[0].filename, hits[0].lineno) == (__file__, line)
    msg = str(hits[0].message)
    assert msg.startswith("ProgressTracker: rebinding after update() was recorded into a "
                          "capture: that graph keeps reading the arrays bound when it was "
                          "recorded; record it again after rebinding")
    assert issubclass(RebindAfterCaptureWarning, UserWarning)


def test_course_and_component_rebind_warn_once_and_a_cpu_capture_arms_them():
    c = Course(CourseConfig(mode="track", gen=_gen_cfg("track"), collision="segments",
                            checkpoint_spacing=0.5))
    pos, pos2 = (wp.zeros(E, dtype=wp.vec3f, device=DEV) for _ in range(2))
    q, he = _boxes(E)
    c.bind(pos, q, he)
    c.step()
    with wp.ScopedCapture(device=DEV) as cap:
        c.step()
    wp.capture_launch(cap.graph)
    assert c.progress._recorded and c.collision._recorded
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.bind(pos, q, he)                            # same arrays: silent
        line = sys._getframe().f_lineno + 1
        c.bind(pos2, q, he)                           # two utilities replaced: ONE warning
        c.bind(pos, q, he)
    hits = _rebind_hits(rec)
    assert len(hits) == 1 and (hits[0].filename, hits[0].lineno) == (__file__, line)
    assert not c.progress._recorded and not c.collision._recorded
    # A component: Component.bind warns at its caller's line, once.
    gen = _track_gen()
    cc = tools.ComponentContext.for_producer(gen)
    cc.add(tools.Checkpoints(0.5), gen.program.result)
    prog = cc.add(tools.Progress(), gen.program.result)
    prog.bind(position=pos)
    prog.step(_Capturing(DEV, None))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1
        prog.bind(position=pos2)
    hits = _rebind_hits(rec)
    assert len(hits) == 1 and (hits[0].filename, hits[0].lineno) == (__file__, line)


def test_the_rebind_probe_changes_no_emission():
    t = _track_gen().program.result
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    loc = TrackLocalizer(t, warm_window=2, position=pos)
    plain, armed = LaunchContext.for_call(DEV), _Capturing(DEV, None)
    with trace_launches() as a:
        loc.query(ctx=plain)
    with trace_launches() as b:
        loc.query(ctx=armed)
    with trace_launches() as c:
        loc.query(ctx=armed)                          # armed: no further probe
    assert a.same_topology(b) and b.same_topology(c) and loc._recorded


# ---------------------------------------------------------------------------------------
# 12. CheckpointSet.valid / .generation
# ---------------------------------------------------------------------------------------

def test_checkpoint_set_fields():
    gen = _track_gen()
    t = gen.program.result
    cs = CheckpointSampler(t, 0.5, max_checkpoints=48)
    s = cs.sample()
    assert s.valid is not t.valid and s.generation is not t.generation   # owned
    assert _bytes(s.valid) == _bytes(t.valid)
    gen.generate(advance=True, mask=_i32([1, 0, 0, 0, 0, 0]))
    assert s.generation.numpy()[0] == 0                   # follows sample(), not the track
    cs.sample()
    assert s.generation.numpy()[0] == 1
    seq = _gate_gen().program.result
    g = CheckpointSet.from_gates(seq)
    assert g.valid is seq.valid and g.generation is seq.generation           # aliases
    cl = s.clone()
    assert cl.valid is not s.valid and _bytes(cl.generation) == _bytes(s.generation)
    plain = CheckpointSet(s.position, s.left, s.right, s.tangent, s.up_half, s.count)
    assert plain.valid is None and plain.generation is None and plain.clone().valid is None
    no_gen = CheckpointSet.from_gates(types.SimpleNamespace(
        **{k: getattr(seq, k) for k in ("position", "left", "right", "forward",
                                        "orientation", "half_size", "count", "valid")}))
    assert no_gen.generation is None and no_gen.valid is seq.valid
    bad = dataclasses.replace(s, generation=wp.zeros(E * 48, dtype=wp.int32, device=DEV))
    with pytest.raises(ValueError, match=r"^ProgressTracker: CheckpointSet\.generation: "
                                         r"expected a contiguous \[6\] int32"):
        ProgressTracker(bad)
    bad = dataclasses.replace(s, valid=wp.zeros(E, dtype=wp.float32, device=DEV))
    with pytest.raises(ValueError, match=r"^ProgressTracker: CheckpointSet\.valid"):
        ProgressTracker(bad)


# ---------------------------------------------------------------------------------------
# 13. The RL loop of the docs with a forced-invalid env
# ---------------------------------------------------------------------------------------

def test_the_rl_loop_gates_on_valid_and_regenerates_invalid_envs():
    c = Course(CourseConfig(mode="track", gen=_gen_cfg("track"), seeds=21,
                            collision="segments", checkpoint_spacing=0.5))
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    q, he = _boxes(E)
    c.bind(pos, q, he)
    # Force env 2 invalid (geometry kept: what on_invalid="keep" leaves) and refresh.
    v = c.result.valid.numpy()
    v[2] = 0
    c.result.valid.assign(v)
    c._refresh_geometry(LaunchContext.for_call(DEV))
    done = c.inputs["regen_mask"]
    rewards = []
    for s in range(12):
        pos.assign(_points(c.result, s))
        res = c.step()
        valid = res.valid.numpy() != 0
        oob = res.contacts.oob.numpy()
        d = res.events.dist_to_next.numpy()
        # The documented recipe: select, never multiply (NaN * 0 is NaN).
        r = np.where(valid, res.events.passed.numpy() - 0.1 * oob, 0.0)
        obs = np.where(valid[:, None], np.stack([d, res.contacts.distance.numpy()], -1),
                       0.0)
        assert np.isfinite(r).all() and np.isfinite(obs).all()
        assert not np.isfinite(res.contacts.distance.numpy()[2]) and oob[2] == 1
        rewards.append(float(r.mean()))
        dn = (1 - valid).astype(np.int32)              # done |= 1 - valid
        done.assign(dn)
        if dn.any():
            c.regenerate(mask=done, advance=True)
            done.assign(np.zeros(E, np.int32))
            break
    assert np.isfinite(rewards).all()
    assert c.result.valid.numpy()[2] == 1, "the advanced regeneration gave env 2 a new course"
    assert c.result.generation.numpy()[2] == 1
    pos.assign(_points(c.result, 0))
    res = c.step()
    assert np.isfinite(res.events.dist_to_next.numpy()).all()


# ---------------------------------------------------------------------------------------
# 14. check_source(optional=)
# ---------------------------------------------------------------------------------------

def test_check_source_optional_members():
    from track_gen.results import _check_and_bind

    t = _track_gen().program.result
    lay = check_source(t, (HasCenterline, HasValidity, HasGeneration), consumer="x",
                       optional=("valid", "generation"))
    assert lay == t.layout
    mini = MinimalCenterline(t)
    _, bound = _check_and_bind(mini, (HasCenterline, HasValidity, HasGeneration),
                               consumer="x", optional=("valid", "generation"))
    assert bound.valid is None and bound.generation is None and bound.center is t.center
    bad = types.SimpleNamespace(**vars(mini), valid=wp.zeros(E + 1, dtype=wp.int32,
                                                             device=DEV))
    with pytest.raises(ValueError, match=r"^x: SimpleNamespace\.valid: expected a "
                                         r"contiguous \[6\] int32"):
        check_source(bad, (HasCenterline, HasValidity), consumer="x", optional=("valid",))
    with pytest.raises(ValueError, match=r"optional members \['outer'\] are not members"):
        check_source(t, HasCenterline, consumer="x", optional=("outer",))
    with pytest.raises(ValueError, match="none of the optional members"):
        check_source(types.SimpleNamespace(), HasValidity, consumer="x",
                     optional=("valid",))
    # HasCheckpoints reads a CheckpointSet (up_half is a schema member since PR14)
    s = CheckpointSampler(t, 0.5, max_checkpoints=48).sample()
    assert check_source(s, HasCheckpoints, consumer="x") == Layout(E, 48, DEV, True)
    assert isinstance(s, HasCheckpoints) and isinstance(t, HasValidity)
