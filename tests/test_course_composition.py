"""Course as the composition root (roadmap PR15), on CPU.

What this module proves (the CUDA proofs are ``tests/test_course_composition_cuda.py``):

0. The 0.x path: ``Course(CourseConfig)`` reproduces the d58ac55 goldens
   (``tests/goldens/pr15_course.json``, :mod:`tests.tools.make_pr15_course_goldens`):
   launch traces by buffer role (on every platform) and every state buffer, every step
   (where the platform has an entry), ten variants.
1. The three front doors, their ``TypeError`` s, subclasses.
2. The presets: names, sources, capacities (one path: ``Checkpoints("auto")``), the clip
   warning, ``gate_width``, ``size_hi`` from the plan, the pre-build checks.
3. Ordering (a stable topological sort), its errors, the step / reset launch order.
4. ``bind`` routing by declared inputs, atomicity, the 0.x refusals.
5. One course per producer (supersession).
6. ``StepResult``: the 0.x dataclass plus read-only mapping access.
7. The regeneration body (one body, ``any_set`` + ``skip_unless`` first).
8. The all-zero-mask identity (the unconditional lowering is exact).
9. The RL loop recorded into ONE CPU graph equals an eager twin.
10. The CurvatureLookahead toy (public imports only).
11. The contract rows: no bound input in the regeneration body (kit and course level),
    and the kit's ``capture.caller`` chain.
12. A minimal Producer; the ``Producer`` surface members.
13. ``generate(seeds=)`` by seed owner.
14. ``prepare=False`` / ``prepare()``.
15. The hot path; 16. CPU graph lifetime (and what a recorded call anchors); 17. the
default-stream guard.
"""
from __future__ import annotations

import ast
import collections.abc
import copy
import dataclasses
import gc
import inspect
import pprint
import types
import warnings
import weakref

import numpy as np
import pytest
import warp as wp

from tests import _component_toys as toys
from tests.tools import make_pr15_course_goldens as goldens
from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenConfig,
                       TrackGenerator, tools)
from track_gen._src import course as course_mod
from track_gen.course import Course, CourseConfig, StepResult
from track_gen.exec import (CaptureError, LaunchContext, NotCapturableError,
                            RebindAfterCaptureWarning)
from track_gen.exec import _compat
from track_gen.pipeline import Producer, build, presets
from track_gen.plugin import CaptureClass
from track_gen.testing import check_component, hot_path_guard, trace_launches
from track_gen.testing.capture import result_arrays
from track_gen.testing.components import _caller_chain, _done_schedule, _result_arrays
from track_gen.testing.trace import BufferSet

E = 6
DEV = "cpu"


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def _i32(values) -> "wp.array":
    return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=DEV)


def _b(arr) -> bytes:
    return np.ascontiguousarray(arr.numpy()).tobytes()


def _tree(obj, prefix: str) -> dict:
    return {f"{prefix}.{k}": _b(v) for k, v in result_arrays(obj).items()}


def _deep(prefix: str, obj) -> dict:
    """Every array reachable from ``obj`` (the kit's walker), as bytes."""
    return {f"{prefix}.{k}": _b(v) for k, v in _result_arrays(obj).items()}


def _cfg(**kw) -> TrackGenConfig:
    return TrackGenConfig(num_envs=E, device=DEV, **kw)


def _gcfg(**kw) -> GateGenConfig:
    kw.setdefault("gate_width", 0.3)
    return GateGenConfig(num_envs=E, device=DEV, **kw)


def _vec3(n: int) -> "wp.array":
    return wp.zeros(n, dtype=wp.vec3f, device=DEV)


def _boxes(n: int) -> "tuple[wp.array, wp.array]":
    q = wp.array(np.tile(np.float32([0, 0, 0, 1]), (n, 1)), dtype=wp.quatf, device=DEV)
    he = wp.array(np.full((n, 2), 0.03, np.float32), dtype=wp.vec2f, device=DEV)
    return q, he


def _pos_on(line, s: int) -> np.ndarray:
    pts = line.center.numpy().reshape(E, -1, 3)
    cnt = line.count.numpy()
    out = np.zeros((E, 3), np.float32)
    for e in range(E):
        m = int(min(cnt[e], pts.shape[1]))
        if m >= 3:
            out[e] = pts[e, (s * 5 + 3 * e) % m]
    return out


def _line(c):
    return c.course_line.centerline if c.course_line is not None else c.result


def _track_toy(**kw) -> Course:
    """The track preset with a Localizer and the CurvatureLookahead toy (listed out of
    order: the toy reads the localizer)."""
    kw.setdefault("seeds", 3)
    return Course.track(_cfg(), checkpoint_spacing=0.5, collision="segments",
                        extra=[toys.CurvatureLookahead(6), tools.Localizer()], **kw)


def _bind_all(c, pos=None) -> "wp.array":
    pos = _vec3(E) if pos is None else pos
    if "orientation" in {k for comp in c.components.values() for k in comp.inputs}:
        q, he = _boxes(E)
        c.bind(position=pos, orientation=q, half_extents=he)
    else:
        c.bind(position=pos)
    return pos


def _state(c) -> dict:
    """Every observable buffer of a course: the result, every component (its result,
    utility state, scratch), seeds and epochs."""
    out = _deep("result", c.result)
    for name, comp in c.components.items():
        out.update(_deep(name, comp))
    if c.seeds is not None:
        out["seeds"] = _b(c.seeds.seeds)
        out["epoch"] = _b(c.seeds.epoch)
    return out


def _raises(exc, fn, match: str = ""):
    with pytest.raises(exc, match=match):
        fn()


class _Counting:
    """Counts constructions of a generator class patched into the course module."""

    def __init__(self, monkeypatch, name: str) -> None:
        real = getattr(course_mod, name)
        self.n = 0

        def make(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(course_mod, name, make)


# ---------------------------------------------------------------------------------------
# 0. The 0.x path against the d58ac55 goldens
# ---------------------------------------------------------------------------------------

def _diff_launches(exp: dict, got: dict) -> list:
    bad = []
    if got["max_checkpoints"] != exp["max_checkpoints"]:
        bad.append(("max_checkpoints", exp["max_checkpoints"], got["max_checkpoints"]))
    if got["construct"] != exp["construct"]:
        bad.append(("construct", exp["construct"], got["construct"]))
    bad += [(k, exp["calls"][k], got["calls"].get(k)) for k in exp["calls"]
            if got["calls"].get(k) != exp["calls"][k]]
    if set(got["calls"]) != set(exp["calls"]):
        bad.append(("calls", sorted(set(got["calls"]) ^ set(exp["calls"]))))
    return bad


def _diff_states(exp: dict, got: dict) -> list:
    bad = []
    steps = [i for i, (a, b) in enumerate(zip(exp["state"], got["state"])) if a != b]
    if steps or len(got["state"]) != len(exp["state"]):
        bad.append(("state at step", steps[:3]))
    fields = [k for k in set(exp["final"]) | set(got["final"])
              if exp["final"].get(k) != got["final"].get(k)]
    if fields:
        bad.append(("final", sorted(fields)[:5]))
    return bad


def _launch_goldens() -> dict:
    """This platform's entry when the file has one, else the platform-independent
    launch digests (kernels, dims, host scalars and buffer roles: no float result)."""
    data = goldens.load()
    return data.get(goldens.platform_key()) or data[goldens.TOPOLOGY]


@pytest.mark.parametrize("name", list(goldens.CASES))
def test_config_course_launches_match_the_pr14_goldens(name):
    """S2, every platform: every call's launch trace (kernel, dims, scalars, buffers by
    role) and the checkpoint capacity equal the d58ac55 Course's."""
    got, _ = goldens.run_case(name)
    bad = _diff_launches(_launch_goldens()[name], got)
    assert not bad, bad


@pytest.mark.parametrize("name", list(goldens.CASES))
def test_config_course_states_match_the_pr14_goldens(name):
    """S2, where this platform has an entry: every state buffer after every step equals
    the d58ac55 Course's (float results depend on the platform and the Warp version: add
    an entry with tests/tools/make_pr15_course_goldens.py from a d58ac55 tree)."""
    gold = goldens.load().get(goldens.platform_key())
    if gold is None:
        pytest.skip(f"no PR15 course state goldens for {goldens.platform_key()}")
    got, _ = goldens.run_case(name)
    bad = _diff_states(gold[name], got)
    assert not bad, bad


def test_the_platform_independent_launch_goldens_are_the_platform_ones():
    """The ``topology`` entry holds the launch digests of every platform entry that the
    tool did not report as different (today: the one it was taken from)."""
    data = goldens.load()
    topo = data[goldens.TOPOLOGY]
    assert set(topo) == set(goldens.CASES)
    platforms = [k for k in data if k != goldens.TOPOLOGY]
    assert platforms and any(goldens.topology_of(data[k]) == topo for k in platforms)


def test_the_goldens_catch_a_forward_reset_order(monkeypatch):
    """Non-vacuity: the 0.x resets run localizer then progress; a forward order is a
    different launch sequence."""
    def forward(self, ctx, mask):
        for comp in self._components:
            comp.reset(ctx, mask)

    monkeypatch.setattr(Course, "_reset_runtime", forward)
    got, _ = goldens.run_case("gates[posts]")
    bad = _diff_launches(_launch_goldens()["gates[posts]"], got)
    assert any(k[0] == "construct" or "regenerate" in k[0] or "reset" in k[0]
               for k in bad), bad


# ---------------------------------------------------------------------------------------
# 1. Constructors
# ---------------------------------------------------------------------------------------

def test_the_three_config_front_doors_build_the_same_course():
    cfg = CourseConfig(mode="track", gen=_cfg(), seeds=5, collision="segments",
                       checkpoint_spacing=0.5)
    a, b, c = Course(cfg), Course(config=cfg), Course.from_config(cfg)
    for x in (a, b, c):
        assert type(x) is Course and x._cfg is cfg
        assert list(x.components) == ["checkpoints", "progress", "collision"]
    assert _tree(a.result, "r") == _tree(b.result, "r") == _tree(c.result, "r")
    assert _tree(a.checkpoints, "c") == _tree(c.checkpoints, "c")
    assert list(inspect.signature(Course).parameters) == [
        "generator", "components", "stream", "prepare", "config"]


class _Duck:
    name, inputs, after, source, requires, optional = "duck", {}, (), None, (), ()
    topology, stateful, layout = "any", False, None

    def attach(self, s, cc):
        self.result = None

    def on_generate(self, ctx, mask):
        pass


def test_constructor_type_errors(monkeypatch):
    cfg = CourseConfig(mode="track", gen=_cfg(), checkpoint_spacing=0.5)
    gen = TrackGenerator(_cfg(), seeds=0)
    _raises(TypeError, lambda: Course(cfg, [tools.Progress()]), "takes no components")
    _raises(TypeError, lambda: Course(cfg, stream=object()), "takes no components")
    _raises(TypeError, lambda: Course(cfg, prepare=False), "takes no components")
    _raises(TypeError, lambda: Course(gen, config=cfg), "not both")
    _raises(TypeError, lambda: Course(_cfg()), "Course.track")
    _raises(TypeError, lambda: Course(presets.track(num_envs=E, device=DEV)),
            "Course.track")
    _raises(TypeError, lambda: Course(), "needs a generator")
    _raises(TypeError, lambda: Course(types.SimpleNamespace(
        result=gen.program.result, layout=gen.program.layout)), r"\['enqueue'\]")
    _raises(TypeError, lambda: Course(gen, [_Duck()]), r"\['bind', 'step', 'reset'\]")
    _raises(TypeError, lambda: Course(gen, tools.Checkpoints(0.5)), "sequence")
    _raises(TypeError, lambda: Course.from_config(_cfg()), "takes a CourseConfig")


class MyCourse(Course):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reward_scale = 2.0


def test_a_subclass_init_runs_for_every_front_door():
    cfg = CourseConfig(mode="track", gen=_cfg(), checkpoint_spacing=0.5)
    made = [MyCourse(cfg), MyCourse.from_config(cfg),
            MyCourse.track(_cfg(), checkpoint_spacing=0.5),
            MyCourse.gates(_gcfg(), collision=0.03),
            MyCourse(TrackGenerator(_cfg(), seeds=1), [tools.Checkpoints(0.5)])]
    for x in made:
        assert type(x) is MyCourse and x.reward_scale == 2.0


# ---------------------------------------------------------------------------------------
# 2. Presets
# ---------------------------------------------------------------------------------------

def test_preset_component_names_and_order():
    assert list(Course.track(_cfg(), checkpoint_spacing=0.5).components) == [
        "checkpoints", "progress"]                       # collision defaults to None
    assert list(Course.track(_cfg(), checkpoint_spacing=0.5,
                             collision="sdf").components) == [
        "checkpoints", "progress", "collision"]
    assert list(Course.gates(_gcfg()).components) == [
        "checkpoints", "centerline", "progress", "localizer"]
    g = Course.gates(_gcfg(), collision=0.03, localize_window=3)
    assert list(g.components) == ["checkpoints", "centerline", "progress", "collision",
                                  "localizer"]
    assert type(g.components["collision"]).__name__ == "DiscCollision"
    assert g.localizer is g.components["localizer"].utility
    assert g.course_line is g.components["centerline"].utility


def test_track_preset_equals_a_hand_composition():
    """``Course(generator, components)`` (out of order) == ``Course.track`` with the same
    components, bytes over 12 steps and two masked regenerations."""
    a = Course(TrackGenerator(_cfg(), seeds=3),
               [tools.Checkpoints(0.5), tools.Progress(), tools.BandCollision(),
                toys.CurvatureLookahead(6), tools.Localizer()])
    b = _track_toy()
    assert list(a.components) == list(b.components)
    pa, pb = _bind_all(a), _bind_all(b)
    for s in range(12):
        x = _pos_on(a.result, s)
        pa.assign(x)
        pb.assign(x)
        for c in (a, b):
            c.step()
            if s == 4:
                c.regenerate(mask=c.inputs["regen_mask"], advance=True)
            if s == 6:
                c.regenerate(mask=_i32([0, 1, 1, 0, 0, 0]), advance=True)
        assert _state(a) == _state(b), s


def test_gates_preset_with_an_rng_equals_the_config_course():
    f1 = Course(CourseConfig(mode="gates", gen=_gcfg(), seeds=5, post_radius=0.03,
                             localize_window=3))
    f2 = Course.gates(_gcfg(), collision=0.03, localize_window=3,
                      seeds=PerEnvSeededRNG(seeds=5, num_envs=E, device=DEV))
    assert f2.rng is f2.generator.rng is not None
    p1, p2 = _bind_all(f1), _bind_all(f2)
    for s in range(10):
        x = _pos_on(f1.course_line.centerline, s)
        p1.assign(x)
        p2.assign(x)
        for c in (f1, f2):
            c.step()
            if s == 3:
                c.regenerate(mask=_i32([1, 0, 0, 1, 0, 0]), advance=True)
            if s == 6:
                c.generate(seeds=9)
        for part in ("events", "contacts", "frame"):
            assert _tree(getattr(f1._step_result, part), part) == \
                _tree(getattr(f2._step_result, part), part), (s, part)
        assert _tree(f1.result, "r") == _tree(f2.result, "r"), s


def test_presets_take_built_generators_and_programs():
    g = TrackGenerator(_cfg(), seeds=3)
    cb = Course.track(g, checkpoint_spacing=0.5)
    ref = Course.track(_cfg(), checkpoint_spacing=0.5, seeds=3)
    assert cb.generator is g and _tree(cb.checkpoints, "c") == _tree(ref.checkpoints, "c")
    prog = build(presets.track(num_envs=E, device=DEV), seeds=3)
    cp = Course.track(prog, checkpoint_spacing=0.5)
    assert cp.generator is prog and cp.rng is None
    assert cp.checkpoint_sampler.max_checkpoints == 70
    gg = GateGenerator(_gcfg(), seeds=2)
    assert Course.gates(gg, collision=0.03).generator is gg
    _raises(TypeError, lambda: Course.track(TrackGenerator(_cfg(), seeds=0),
                                            checkpoint_spacing=0.5, seeds=4),
            "built generator")
    _raises(TypeError, lambda: Course.gates(gg, stream=object()), "built generator")
    _raises(TypeError, lambda: Course.gates(TrackGenerator(_cfg(), seeds=0)),
            "Course.track")
    _raises(TypeError, lambda: Course.track(gg, checkpoint_spacing=0.5), "Course.gates")
    _raises(TypeError, lambda: Course.track(_gcfg(), checkpoint_spacing=0.5),
            "Course.gates")
    _raises(TypeError, lambda: Course.track(presets.gates(num_envs=E, device=DEV,
                                                          gate_width=0.3),
                                            checkpoint_spacing=0.5), "Course.gates")
    _raises(TypeError, lambda: Course.gates(CourseConfig(mode="gates", gen=_gcfg())),
            "from_config")


def test_gates_over_a_stations_spec_with_frame_collision():
    g = Course.gates(presets.gates(num_envs=E, device=DEV, source="stations",
                                   gate_width=0.3),
                     collision=tools.FrameCollision(0.05, 0.02, 0.02), seeds=3)
    g.bind(position=_vec3(E))
    sr = g.step()
    assert list(sr.keys()) == ["progress", "collision", "localizer"]
    assert type(sr.contacts).__name__ == "FrameContact"


def test_the_collision_slot_needs_the_name_collision():
    _raises(ValueError, lambda: Course.track(
        _cfg(), checkpoint_spacing=0.5,
        collision=tools.BandCollision("segments", name="acme.walls")), "named 'collision'")
    _raises(ValueError, lambda: Course.gates(
        _gcfg(), collision=tools.DiscCollision(0.03, name="posts")), "named 'collision'")
    c = Course.track(_cfg(), checkpoint_spacing=0.5,
                     collision=tools.BandCollision("segments", max_boxes=2))
    assert c.collision is c.components["collision"].utility
    walls = Course.track(_cfg(), checkpoint_spacing=0.5,
                         extra=[tools.BandCollision("segments", name="acme.walls")])
    assert walls.collision is None and "acme.walls" in walls.components


def test_one_capacity_path_for_every_front_door():
    """Flat: the plan bound (70 at spacing 0.5); elevated: the readback, equal across the
    generic root (``Checkpoints("auto")``), the preset and the config course;
    ``max_checkpoints=None`` stays strict (PR14)."""
    c = Course.track(_cfg(), checkpoint_spacing=0.5)
    assert c.checkpoint_sampler.max_checkpoints == 70
    assert c.components["checkpoints"].defaulted.startswith("the plan bound")
    assert c.components["checkpoints"].max_checkpoints == "auto"
    ecfg = _cfg(z_profile="uniform", z_min=0.0, z_max=2.0)
    ga = Course(TrackGenerator(ecfg, seeds=0),
                [tools.Checkpoints(0.5, max_checkpoints="auto"), tools.Progress()])
    gb = Course.track(ecfg, checkpoint_spacing=0.5, seeds=0)
    gc_ = Course(CourseConfig(mode="track", gen=ecfg, seeds=0, checkpoint_spacing=0.5))
    ms = {x.checkpoint_sampler.max_checkpoints for x in (ga, gb, gc_)}
    assert len(ms) == 1 and ms.pop() < 70
    assert ga.components["checkpoints"].defaulted == "read back from the construction batch"
    _raises(ValueError, lambda: Course(TrackGenerator(ecfg, seeds=0),
                                       [tools.Checkpoints(0.5), tools.Progress()]),
            "max_loop_length")
    fixed = Course.track(_cfg(), checkpoint_spacing=0.5, max_checkpoints=40)
    assert fixed.checkpoint_sampler.max_checkpoints == 40
    assert fixed.components["checkpoints"].defaulted is None
    _raises(ValueError, lambda: tools.Checkpoints(0.5, max_checkpoints="all"), "'auto'")


def test_the_clip_warning_of_a_preset_and_of_a_config_course(monkeypatch):
    from track_gen.tools import _component

    monkeypatch.setattr(_component, "_plan_capacity", lambda prog: {"max_loop_length": 2.0})
    with pytest.warns(UserWarning, match=r"^Course: the default max_checkpoints=4 .* "
                                         r"pass a larger max_checkpoints$") as rec:
        Course.track(_cfg(), checkpoint_spacing=0.5, seeds=3)
    hits = [w for w in rec if "clips the checkpoint ring" in str(w.message)]
    assert len(hits) == 1 and hits[0].filename == __file__
    with pytest.warns(UserWarning, match=r"pass a larger CourseConfig\.max_checkpoints$"):
        Course(CourseConfig(mode="track", gen=_cfg(), seeds=3, checkpoint_spacing=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Course.track(_cfg(), checkpoint_spacing=0.5, seeds=3, max_checkpoints=4)


def test_gate_width_is_checked_before_anything_is_built(monkeypatch):
    count = _Counting(monkeypatch, "GateGenerator")
    _raises(ValueError, lambda: Course.gates(GateGenConfig(num_envs=E, device=DEV)),
            "gate_width > 0")
    _raises(ValueError, lambda: Course.gates(presets.gates(num_envs=E, device=DEV)),
            "gate_width > 0")
    assert count.n == 0
    c = Course.gates(_gcfg(), collision=0.03)
    assert count.n == 1
    with pytest.raises(ValueError):
        c.generator.params["gate_frames"].set("gate_width", 0.0)
    gg = GateGenerator(_gcfg(), seeds=1)
    Course.gates(gg)
    with pytest.raises(ValueError):                       # a given generator is narrowed
        gg.params["gate_frames"].set("gate_width", -1.0)


def test_a_randomized_gate_width_is_checked_on_the_randomizer_lo(monkeypatch):
    """A width the randomizer draws every generation: its construction ``lo`` must be
    ``> 0`` and the construction width (``presets.gates`` defaults to 0) is not used."""
    count = _Counting(monkeypatch, "GateGenerator")
    spec = presets.gates(num_envs=E, device=DEV, gate_width=0.0)
    lo0 = presets.gates(num_envs=E, device=DEV, gate_width=0.2).with_randomize(
        {"gate_frames.gate_width": (0.0, 0.3)})
    _raises(ValueError, lambda: Course.gates(lo0), r"randomizer's lo > 0, got lo=0\.0")
    assert count.n == 0
    _raises(ValueError, lambda: Course.gates(GateGenerator(lo0, seeds=1)),
            "randomizer's lo > 0")                      # a built generator too
    c = Course.gates(spec.with_randomize({"gate_frames.gate_width": (0.1, 0.3)}),
                     collision=0.03, seeds=1)
    w = c.generator.params["gate_frames"]["gate_width"].numpy()
    assert ((w >= np.float32(0.1)) & (w <= np.float32(0.3))).all(), w
    with pytest.raises(ValueError):                     # the scalar set stays narrowed
        c.generator.params["gate_frames"].set("gate_width", 0.0)


def test_preset_arguments_get_clear_errors_before_anything_is_built(monkeypatch):
    tg = _Counting(monkeypatch, "TrackGenerator")
    gg = _Counting(monkeypatch, "GateGenerator")
    t, g = _cfg(), _gcfg()
    cases = [
        (ValueError, "post radius > 0, None or a component",
         lambda: Course.gates(g, collision="posts")),
        (ValueError, "post radius > 0, None or a component",
         lambda: Course.gates(g, collision=True)),
        (ValueError, "'segments', 'sdf', None or a component",
         lambda: Course.track(t, checkpoint_spacing=0.5, collision=True)),
        (ValueError, "'segments', 'sdf', None or a component",
         lambda: Course.track(t, checkpoint_spacing=0.5, collision=0.03)),
        (ValueError, "checkpoint_spacing must be a number > 0, got None",
         lambda: Course.track(t, checkpoint_spacing=None)),
        (ValueError, "checkpoint_spacing must be a number > 0",
         lambda: Course.track(t, checkpoint_spacing="0.5")),
        (ValueError, "an int >= 3, 'auto' or None, got 'all'",
         lambda: Course.track(t, checkpoint_spacing=0.5, max_checkpoints="all")),
        (ValueError, "an int >= 3",
         lambda: Course.track(t, checkpoint_spacing=0.5, max_checkpoints=True)),
        (TypeError, r"extra must be a sequence of components, got Localizer \(pass "
                    r"\[component\]\)",
         lambda: Course.track(t, checkpoint_spacing=0.5, extra=tools.Localizer())),
        (TypeError, "extra must be a sequence of components",
         lambda: Course.gates(g, extra=tools.Progress())),
    ]
    for exc, match, fn in cases:
        _raises(exc, fn, match)
    assert tg.n == 0 and gg.n == 0
    auto = Course.track(t, checkpoint_spacing=0.5, max_checkpoints="auto", seeds=1)
    assert auto.checkpoint_sampler.max_checkpoints == 70
    assert auto.components["checkpoints"].defaulted.startswith("the plan bound")


def _size_warnings(fn) -> list:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in rec if "size_hi" in str(w.message)]


def test_size_hi_comes_from_the_plan_for_every_front_door():
    rspec = presets.track(num_envs=E, device=DEV, randomize={"generator.size": (1.0, 2.5)})
    gen = TrackGenerator(rspec, seeds=0)
    assert dict(tools.ComponentContext.for_producer(gen).bounds) == {"size_hi": 2.5}
    assert dict(tools.ComponentContext.for_producer(
        TrackGenerator(_cfg(size=1.5), seeds=0)).bounds) == {"size_hi": 1.5}
    assert dict(tools.ComponentContext.for_producer(TrackGenerator(
        presets.track(num_envs=E, device=DEV, generator="external", num_points=64),
        seeds=0)).bounds) == {}
    w = _size_warnings(lambda: Course.track(rspec, checkpoint_spacing=0.5, collision="sdf"))
    assert w and "up to size_hi=2.5" in w[0] and "BandCollision" in w[0]
    assert _size_warnings(lambda: Course.track(
        rspec, checkpoint_spacing=0.5, collision=tools.BandCollision("sdf", max_boxes=2)))
    assert _size_warnings(lambda: Course.track(rspec, checkpoint_spacing=0.5,
                                               extra=[tools.HeightFieldBake(64)]))
    assert not _size_warnings(lambda: Course.track(_cfg(), checkpoint_spacing=0.5,
                                                   collision="sdf"))
    assert not _size_warnings(lambda: Course(CourseConfig(
        mode="track", gen=_cfg(size=2.0), collision="sdf", heightfield_resolution=16,
        checkpoint_spacing=0.5)))
    assert not _size_warnings(lambda: Course.track(
        rspec, checkpoint_spacing=0.5, collision=tools.BandCollision("sdf", size_hi=None)))


def test_preset_checks_raise_before_any_generator_is_built(monkeypatch):
    tg = _Counting(monkeypatch, "TrackGenerator")
    gg = _Counting(monkeypatch, "GateGenerator")
    attached = tools.Localizer()
    Course.track(_cfg(), checkpoint_spacing=0.5, extra=[attached])
    tg.n = 0
    cases = [
        (ValueError, "reads 'localizer'",
         lambda: Course.track(_cfg(), checkpoint_spacing=0.5,
                              extra=[toys.CurvatureLookahead(4)])),
        (ValueError, "two components are named 'progress'",
         lambda: Course.track(_cfg(), checkpoint_spacing=0.5, extra=[tools.Progress()])),
        (ValueError, "two components are named 'collision'",
         lambda: Course.gates(_gcfg(), collision=0.03,
                              extra=[tools.FrameCollision(0.05, 0.02, 0.02)])),
        (ValueError, "already attached",
         lambda: Course.track(_cfg(), checkpoint_spacing=0.5, extra=[attached])),
        (TypeError, "lacks", lambda: Course.track(_cfg(), checkpoint_spacing=0.5,
                                                  extra=[_Duck()])),
        (ValueError, "checkpoint_spacing", lambda: Course.track(_cfg(),
                                                                checkpoint_spacing=0.0)),
        (ValueError, "collision must be", lambda: Course.track(
            _cfg(), checkpoint_spacing=0.5, collision="boxes")),
        (ValueError, "max_checkpoints", lambda: Course.track(
            _cfg(), checkpoint_spacing=0.5, max_checkpoints=2)),
        (ValueError, "post radius", lambda: Course.gates(_gcfg(), collision=0.0)),
        (ValueError, "samples_per_gate", lambda: Course.gates(_gcfg(), samples_per_gate=1)),
    ]
    for exc, match, fn in cases:
        _raises(exc, fn, match)
    assert tg.n == 0 and gg.n == 0
    ok = Course.track(_cfg(), checkpoint_spacing=0.5,
                      extra=[toys.CurvatureLookahead(4), tools.Localizer()])
    assert tg.n == 1
    assert list(ok.components) == ["checkpoints", "progress", "localizer",
                                   "acme.lookahead"]


# ---------------------------------------------------------------------------------------
# 3. Ordering
# ---------------------------------------------------------------------------------------

class _Named(tools.Component):
    def __init__(self, name, after=()):
        self.name, self.after = name, tuple(after)

    def attach(self, s, cc):
        self.result = None


def test_the_order_is_a_stable_topological_sort():
    from track_gen.tools._component import _order

    for cfg in (CourseConfig(mode="track", gen=_cfg(), checkpoint_spacing=0.5,
                             collision="sdf", heightfield_resolution=16),
                CourseConfig(mode="gates", gen=_gcfg(), frame_collision=True,
                             agent_radius=0.05, frame_thickness=0.02, frame_depth=0.02)):
        comps = course_mod._components_for(cfg)
        assert _order(comps) == comps                       # identity on 0.x lists
    a, b, c, d = _Named("a", ["c"]), _Named("b"), _Named("c"), _Named("d", ["a", "b"])
    assert [x.name for x in _order([a, b, c, d])] == ["b", "c", "a", "d"]
    assert [x.name for x in _order([d, c, b, a])] == ["c", "b", "a", "d"]
    gen = TrackGenerator(_cfg(), seeds=1)
    _raises(ValueError, lambda: Course(gen, [tools.Progress()]),
            r"component 'progress' reads 'checkpoints' \(after / source\), which is not "
            r"one of its components")
    _raises(ValueError, lambda: Course(gen, [_Named("a", ["b"]), _Named("b", ["a"])]),
            r"components \['a', 'b'\] depend on each other")
    _raises(ValueError, lambda: Course(gen, [tools.Checkpoints(0.5),
                                             tools.Checkpoints(0.4)]),
            "two components are named 'checkpoints'")
    _raises(TypeError, lambda: Course(gen, [_Named("")]), "non-empty str")
    first = tools.Checkpoints(0.5)
    Course(TrackGenerator(_cfg(), seeds=1), [first])
    _raises(ValueError, lambda: Course(gen, [first]), "already attached")


def test_step_runs_forward_and_reset_runs_backward():
    c = _track_toy()
    names = list(c.components)
    assert names == ["checkpoints", "progress", "collision", "localizer", "acme.lookahead"]
    _bind_all(c)
    with trace_launches() as tr:
        c.step()
    assert [k.split(".")[-1] for k in tr.kernels()] == [
        "_progress_update_k", "_box_query_segments_k", "_localize_k", "_lookahead_k"]
    with trace_launches() as tr:
        c.reset()
    assert [k.split(".")[-1] for k in tr.kernels()] == ["_localize_reset_k",
                                                        "_progress_reset_k"]


# ---------------------------------------------------------------------------------------
# 4. bind
# ---------------------------------------------------------------------------------------

class _Picky(tools.Component):
    name = "picky"
    inputs = {"position": tools.InputSpec(wp.vec3f)}

    def attach(self, s, cc):
        self.result = None

    def on_bind(self, position):
        if getattr(position, "_reject", False):
            raise ValueError("picky refuses this array")
        self.position = position


def test_bind_routes_one_array_to_every_declarer():
    c = _track_toy()
    pos = _vec3(E)
    q, he = _boxes(E)
    c.bind(position=pos, orientation=q, half_extents=he)
    look = c.components["acme.lookahead"]
    assert look._position is pos and c.localizer._bound_pos is pos
    assert c.progress._bound_pos is pos
    assert c.collision._bound[0] is pos                     # box_position -> position
    bp = _vec3(E)
    c.bind(position=pos, orientation=q, half_extents=he, box_position=bp)
    assert c.collision._bound[0] is bp                      # explicit beats the default
    t = Course.track(_cfg(), checkpoint_spacing=0.5)       # no collision component
    t.bind(pos, q, he)                                      # the 0.x names are ignored
    assert t._bind_args is not None and t.progress._bound_pos is pos


def test_bind_refusals():
    c = _track_toy()
    pos = _vec3(E)
    q, he = _boxes(E)
    _raises(TypeError, lambda: c.bind(position=pos, orientation=q, half_extents=he,
                                      speed=pos), r"unknown inputs \['speed'\]")
    _raises(TypeError, lambda: c.bind(orientation=q, half_extents=he),
            r"missing inputs \['position'\]")
    _raises(ValueError, lambda: c.bind(position=_vec3(E + 1), orientation=q,
                                       half_extents=he), "position")
    _raises(RuntimeError, lambda: c.step(), r"call bind\(\) before step\(\)")
    spec = presets.track(num_envs=E, device=DEV, generator="external", num_points=64)
    xc = Course(TrackGenerator(spec, seeds=0),
                [tools.Checkpoints(0.5, max_checkpoints=64), tools.Progress()])
    xy = wp.zeros(E * 64, dtype=wp.vec2f, device=DEV)
    _raises(TypeError, lambda: xc.bind(position=pos, **{"centerline.xy": xy}),
            r"course\.generator\.inputs\['centerline\.xy'\] in place")
    _raises(TypeError, lambda: xc.bind(position=pos, regen_mask=_i32(np.zeros(E))),
            "unknown inputs")
    cfgc = Course(CourseConfig(mode="track", gen=_cfg(), checkpoint_spacing=0.5))
    _raises(TypeError, lambda: cfgc.bind(), "missing 1 required argument: 'position'")
    _raises(ValueError, lambda: cfgc.bind(None), "position")
    box = Course(CourseConfig(mode="track", gen=_cfg(), checkpoint_spacing=0.5,
                              collision="segments"))
    _raises(RuntimeError, lambda: box.bind(pos), "bind orientation and half_extents")
    stateless = Course(TrackGenerator(_cfg(), seeds=0), [tools.Checkpoints(0.5)])
    assert stateless.step() is stateless._step_result     # nothing to bind


def test_a_refused_bind_keeps_every_previous_binding():
    c = Course.track(_cfg(), checkpoint_spacing=0.5, extra=[_Picky()])
    p1 = _vec3(E)
    c.bind(position=p1)
    with wp.ScopedCapture(device=DEV):
        c.step()                                    # arms the rebind probe
    p2 = _vec3(E)
    p2._reject = True
    with warnings.catch_warnings():
        warnings.simplefilter("error", RebindAfterCaptureWarning)
        _raises(ValueError, lambda: c.bind(position=p2), "picky refuses")
        _raises(ValueError, lambda: c.bind(position=_vec3(E + 2)), "position")
    assert c.progress._bound_pos is p1 and c._bind_args["position"] is p1
    assert c.components["picky"].position is p1
    first = Course.track(_cfg(), checkpoint_spacing=0.5, extra=[_Picky()])
    _raises(ValueError, lambda: first.bind(position=p2), "picky refuses")
    _raises(RuntimeError, lambda: first.step(), "call bind")    # a failed first bind


class _PinsThenRefuses(tools.Component):
    """``on_bind`` pins ``position`` first, then refuses a rejected ``speed``."""

    name = "acme.half"
    inputs = {"position": tools.InputSpec(wp.vec3f),
              "speed": tools.InputSpec(wp.float32, required=False)}

    def attach(self, s, cc):
        self.result = None
        self.position = None

    def on_bind(self, position, speed=None):
        self.position = position
        if speed is not None and getattr(speed, "_reject", False):
            raise ValueError("acme.half refuses this speed")


def test_a_refusing_on_bind_that_pinned_first_is_restored_and_rebinds_still_warn():
    """The component that refused is bound again too (its ``on_bind`` pinned the new
    array before raising), and the rebind-after-capture probes the refused call cleared
    are restored: a real rebind after the capture still warns."""
    c = Course.track(_cfg(), checkpoint_spacing=0.5, extra=[_PinsThenRefuses()], seeds=1)
    half = c.components["acme.half"]
    p1, p2, p3 = _vec3(E), _vec3(E), _vec3(E)
    ok = wp.zeros(E, dtype=wp.float32, device=DEV)
    bad = wp.zeros(E, dtype=wp.float32, device=DEV)
    bad._reject = True
    c.bind(position=p1, speed=ok)
    with wp.ScopedCapture(device=DEV):
        c.step()                                    # arms the rebind probe
    assert c.progress._recorded
    with warnings.catch_warnings():
        warnings.simplefilter("error", RebindAfterCaptureWarning)
        _raises(ValueError, lambda: c.bind(position=p2, speed=bad), "acme.half refuses")
    assert half.position is p1 and c.progress._bound_pos is p1
    assert c._bind_args["position"] is p1 and c._bind_args["speed"] is ok
    assert c.progress._recorded                     # still armed
    with pytest.warns(RebindAfterCaptureWarning, match="ProgressTracker"):
        c.bind(position=p3, speed=ok)
    assert half.position is p3 and c.progress._bound_pos is p3


class _Speed(tools.Component):
    name = "speed"
    inputs = {"position": tools.InputSpec(wp.vec2f)}

    def attach(self, s, cc):
        self.result = None


class _GenInput(tools.Component):
    name = "acme.xy"
    inputs = {"centerline.xy": tools.InputSpec(wp.vec2f, 64)}

    def attach(self, s, cc):
        self.result = None


def test_input_declarations_are_checked_at_construction():
    _raises(ValueError, lambda: Course(TrackGenerator(_cfg(), seeds=3),
                                       [tools.Checkpoints(0.5), tools.Progress(), _Speed()]),
            "input 'position'")
    spec = presets.track(num_envs=E, device=DEV, generator="external", num_points=64)
    _raises(ValueError, lambda: Course(TrackGenerator(spec, seeds=0), [_GenInput()]),
            "declared by the generator and by a component")


# ---------------------------------------------------------------------------------------
# 5. One course per producer
# ---------------------------------------------------------------------------------------

def test_a_newer_course_over_the_same_generator_supersedes_the_older():
    gen = TrackGenerator(_cfg(), seeds=2)
    a = Course(gen, [tools.Checkpoints(0.5), tools.Progress()])
    pos = _bind_all(a)
    a.step()
    b = Course(gen, [tools.Checkpoints(0.5), tools.Progress()])
    b.bind(position=pos)
    b.step()
    m = _i32(np.zeros(E))
    for call in (lambda: a.step(), lambda: a.reset(m), lambda: a.regenerate(),
                 lambda: a.enqueue_generate(), lambda: a.generate(seeds=4)):
        _raises(RuntimeError, call, "driven by another Course")
    b.regenerate(mask=m, advance=True)
    b.generate(seeds=4)
    x = Course(TrackGenerator(_cfg(), seeds=2), [tools.Checkpoints(0.5)])
    x = Course(x.generator, [tools.Checkpoints(0.5)])     # the rebuild pattern
    x.regenerate()
    old = Course(CourseConfig(mode="track", gen=_cfg(), checkpoint_spacing=0.5))
    Course(old.generator, [tools.Checkpoints(0.5)])
    _raises(RuntimeError, lambda: old.regenerate(), "driven by another Course")


def test_a_construction_that_raises_supersedes_nothing(monkeypatch):
    """Under warnings-as-errors the clip warning makes a second construction raise: the
    first course keeps the generator (supersession is the last step of construction)."""
    from track_gen.tools import _component

    gen = TrackGenerator(_cfg(), seeds=1)
    a = Course(gen, [tools.Checkpoints(0.5, max_checkpoints=40), tools.Progress()])
    a.bind(position=_vec3(E))
    monkeypatch.setattr(_component, "_plan_capacity", lambda prog: {"max_loop_length": 2.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _raises(UserWarning, lambda: Course(gen, [tools.Checkpoints(0.5),
                                                  tools.Progress()]), "clips")
    a.step()
    a.regenerate()
    Course(gen, [tools.Checkpoints(0.5, max_checkpoints=40)])
    _raises(RuntimeError, lambda: a.step(), "driven by another Course")


@dataclasses.dataclass(frozen=True)
class _ValueProducer:
    """A producer hashed and compared by value (a frozen dataclass)."""

    tag: str
    prog: object = dataclasses.field(compare=False, repr=False)

    @property
    def result(self):
        return self.prog.result

    @property
    def layout(self):
        return self.prog.layout

    def enqueue(self, ctx, *, advance=False, mask=None):
        return self.prog.enqueue(ctx, advance=advance, mask=mask)


def test_supersession_follows_producer_identity():
    """Two equal (value-hashed) producers are two producers; an unhashable one is
    tracked too."""
    p1 = _ValueProducer("x", TrackGenerator(_cfg(), seeds=1).program)
    p2 = _ValueProducer("x", TrackGenerator(_cfg(), seeds=2).program)
    assert p1 == p2 and hash(p1) == hash(p2)
    a = Course(p1, [tools.Checkpoints(0.5, max_checkpoints=40)])
    Course(p2, [tools.Checkpoints(0.5, max_checkpoints=40)])
    a.regenerate()                                    # not superseded by an equal one
    Course(p1, [tools.Checkpoints(0.5, max_checkpoints=40)])
    _raises(RuntimeError, lambda: a.regenerate(), "driven by another Course")

    class Unhashable(_Minimal):
        __hash__ = None

    u = Unhashable(TrackGenerator(_cfg(), seeds=3))
    b = Course(u, [tools.Checkpoints(0.5, max_checkpoints=40)])
    Course(u, [tools.Checkpoints(0.5, max_checkpoints=40)])
    _raises(RuntimeError, lambda: b.regenerate(), "driven by another Course")
    alive = weakref.ref(u)
    del u, b
    gc.collect()
    assert alive() is None                            # entries go with their producer
    assert all(p() is not None for p, _ in course_mod._DRIVERS.values())


# ---------------------------------------------------------------------------------------
# 6. StepResult
# ---------------------------------------------------------------------------------------

def test_step_result_mapping_access():
    c = _track_toy()
    _bind_all(c)
    sr = c._step_result
    assert c.step() is sr and c.step() is sr                   # built once, at construction
    assert list(sr.keys()) == ["progress", "collision", "localizer", "acme.lookahead"]
    assert "checkpoints" not in sr and "progress" in sr and "nope" not in sr
    look = c.components["acme.lookahead"]
    assert sr["acme.lookahead"] is look.result and sr.get("nope") is None
    assert sr.events is sr["progress"] and sr.contacts is sr["collision"]
    assert sr.frame is sr["localizer"] and sr.valid is c.result.valid
    assert dict(sr) == dict(sr.items()) and list(sr.values())[0] is sr.events
    with pytest.raises(KeyError, match=r"no component 'checkpoints' in this StepResult "
                                       r"\(per-step components: \['progress'"):
        sr["checkpoints"]
    assert not isinstance(sr, collections.abc.Mapping) and bool(sr)
    with pytest.raises(TypeError, match="not iterable"):
        iter(sr)
    assert [f.name for f in dataclasses.fields(sr)] == ["events", "contacts", "frame",
                                                        "valid"]
    assert repr(sr).startswith("StepResult(events=ProgressEvents(") and \
        "entries=" not in repr(sr)                               # the dataclass repr
    assert pprint.pformat(sr, width=60).startswith("StepResult(events=ProgressEvents(")
    assert "\n" in pprint.pformat(sr, width=60)                  # pprint's dataclass layout
    r = dataclasses.replace(sr, contacts=None)
    assert list(r.keys()) == ["progress", "localizer"]          # the legacy fields only
    kap = look.result.kappa.numpy()
    assert np.isfinite(kap[c.result.valid.numpy() != 0]).all()


def test_step_result_of_a_config_course_and_by_hand():
    c = Course(CourseConfig(mode="gates", gen=_gcfg(), post_radius=0.03))
    _bind_all(c)
    sr = c.step()
    assert list(sr.keys()) == ["progress", "collision", "localizer"]
    assert repr(sr).startswith("StepResult(events=ProgressEvents(") and \
        "entries=" not in repr(sr)
    t = Course(CourseConfig(mode="track", gen=_cfg(), checkpoint_spacing=0.5))
    _bind_all(t)
    assert list(t.step().keys()) == ["progress"] and t.step().frame is None
    legacy_sr = StepResult(sr.events, None, None)
    assert list(legacy_sr.keys()) == ["progress"] and legacy_sr.valid is None
    assert legacy_sr.clone().valid is None and legacy_sr == StepResult(sr.events, None,
                                                                      None)
    empty = Course(TrackGenerator(_cfg(), seeds=0),
                   [tools.Checkpoints(0.5), tools.HeightFieldBake(16)]).step()
    assert bool(empty) and list(empty.keys()) == []


def test_step_result_fields_and_entries_stay_in_sync():
    """Assigning a 0.x field updates its entry, on that instance only: ``copy.copy`` then
    a reassignment, and the dataclass branch of Lightning's ``apply_to_collection`` /
    ``move_data_to_device`` (a deepcopy with the field values in the memo, then a
    ``setattr`` per field)."""
    c = Course(CourseConfig(mode="gates", gen=_gcfg(), post_radius=0.03))
    _bind_all(c)
    sr = c.step()
    r2 = copy.copy(sr)
    r2.contacts = None
    assert list(r2.keys()) == ["progress", "localizer"] and "collision" in sr
    assert sr["collision"] is sr.contacts
    r2.contacts = sr.contacts
    assert r2["collision"] is sr.contacts
    vals = {f.name: getattr(sr, f.name) for f in dataclasses.fields(sr)}
    moved = copy.deepcopy(sr, memo={id(v): v for v in vals.values()})
    for name, v in vals.items():
        setattr(moved, name, None if v is None else ("moved", name))
    assert moved["progress"] == ("moved", "events") and moved.events == ("moved", "events")
    assert moved["collision"] == ("moved", "contacts")
    assert moved["localizer"] == ("moved", "frame")
    assert sr["progress"] is sr.events and c.step() is sr
    assert list(sr.keys()) == ["progress", "collision", "localizer"]


def test_step_result_clone():
    c = _track_toy()
    _bind_all(c)
    sr = c.step()
    snap = sr.clone()
    assert list(snap.keys()) == list(sr.keys())
    look = sr["acme.lookahead"]
    assert snap["acme.lookahead"].kappa is not look.kappa
    assert _b(snap["acme.lookahead"].kappa) == _b(look.kappa)
    assert snap.valid is not sr.valid and _b(snap.valid) == _b(sr.valid)
    assert _tree(snap.events, "e") == _tree(sr.events, "e")
    arr = _i32(np.arange(E))
    by_hand = StepResult._of({"a": arr, "progress": sr.events}, None)
    cl = by_hand.clone()
    assert cl["a"] is not arr and _b(cl["a"]) == _b(arr) and cl.events is cl["progress"]
    with pytest.raises(TypeError, match="component 'x'"):
        StepResult._of({"x": object()}, None).clone()


# ---------------------------------------------------------------------------------------
# 7. The regeneration body
# ---------------------------------------------------------------------------------------

def test_the_regeneration_body_structure():
    c = _track_toy()
    m = _i32([1, 0, 1, 0, 0, 1])
    runs = {}
    for label, fn in (("plain", lambda: c.regenerate()),
                      ("advance", lambda: c.regenerate(advance=True)),
                      ("masked", lambda: c.regenerate(mask=m, advance=True))):
        with trace_launches() as tr:
            fn()
        runs[label] = [k.split(".")[-1] for k in tr.kernels()]
    plain, adv, masked = runs["plain"], runs["advance"], runs["masked"]
    assert plain[0] == "_effective_seed_k" and adv == ["_advance_all_k"] + plain
    assert masked[:3] == ["_fill_i32_k", "_any_set_k", "_advance_masked_k"]
    assert masked.index("_effective_seed_masked_k") > masked.index("_advance_masked_k")
    for ks in runs.values():
        # producer -> on_generate in order (checkpoints before the toy) -> resets reversed
        assert ks[-2:] == ["_localize_reset_k", "_progress_reset_k"]
        assert ks.index("_place_checkpoints_k") < ks.index("_curvature_raw_k")
    assert "_any_set_k" not in plain


def test_regenerate_body_runs_under_skip_unless(monkeypatch):
    c = _track_toy()
    seen = []
    real = LaunchContext.skip_unless

    def spy(self, flag, body):
        seen.append(flag)
        return real(self, flag, body)

    monkeypatch.setattr(LaunchContext, "skip_unless", spy)
    c.regenerate(mask=c.inputs["regen_mask"], advance=True)
    assert seen == [c._regen_flag]
    c.regenerate(advance=True)
    assert seen == [c._regen_flag]                          # unmasked: no skip


# ---------------------------------------------------------------------------------------
# 8. The all-zero mask is an exact identity
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: _track_toy(),
    lambda: Course.gates(_gcfg(), collision=0.03, localize_window=3, seeds=4),
    lambda: Course(CourseConfig(mode="track", gen=_cfg(), seeds=4, collision="sdf",
                                sdf_resolution=16, heightfield_resolution=16,
                                checkpoint_spacing=0.5))], ids=["track", "gates", "config"])
def test_a_zero_mask_regeneration_changes_no_byte(make):
    c = make()
    pos = _bind_all(c)
    line = _line(c)
    for s in range(5):
        pos.assign(_pos_on(line, s))
        c.step()
    s0 = _state(c)
    c.regenerate(mask=_i32([0, 1, 0, 0, 1, 0]), advance=True)
    assert _state(c) != s0                                  # the snapshot sees a change
    pos.assign(_pos_on(line, 6))
    c.step()
    before = _state(c)
    c.regenerate(mask=_i32(np.zeros(E)), advance=True)     # eager: unconditional
    assert _state(c) == before


# ---------------------------------------------------------------------------------------
# 9. The RL loop recorded into ONE CPU graph == an eager twin
# ---------------------------------------------------------------------------------------

@wp.kernel
def _move_k(center: wp.array(dtype=wp.vec3f), count: wp.array(dtype=wp.int32), S: int,
            speed: wp.array(dtype=wp.int32), t: wp.array(dtype=wp.int32),
            steps: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3f)):
    e = wp.tid()
    m = wp.min(count[e], S)
    t[e] = t[e] + speed[e]
    steps[e] = steps[e] + 1
    if m < 3:
        pos[e] = wp.vec3f(0.0, 0.0, 0.0)
        return
    i = t[e] % m
    if i < 0:
        i = i + m
    pos[e] = center[e * S + i] + wp.vec3f(0.002 * float(e % 3), 0.0, 0.0)


@wp.kernel
def _done_k(laps: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
            period: wp.array(dtype=wp.int32), t: wp.array(dtype=wp.int32),
            steps: wp.array(dtype=wp.int32), done: wp.array(dtype=wp.int32)):
    e = wp.tid()
    d = 0
    if laps[e] >= 1 or steps[e] >= period[e] or valid[e] == 0:
        d = 1
    done[e] = d
    if d == 1:
        t[e] = 0
        steps[e] = 0


class _Agents:
    def __init__(self, line, n: int) -> None:
        self.n, self.line = n, line
        self.S = int(line.center.shape[0]) // n
        self.speed = wp.array((2 + np.arange(n) % 4).astype(np.int32), dtype=wp.int32,
                              device=DEV)
        self.period = wp.array((15 + 7 * np.arange(n)).astype(np.int32), dtype=wp.int32,
                               device=DEV)
        self.t = wp.zeros(n, dtype=wp.int32, device=DEV)
        self.steps = wp.zeros(n, dtype=wp.int32, device=DEV)
        self.pos = wp.zeros(n, dtype=wp.vec3f, device=DEV)

    def move(self) -> None:
        wp.launch(_move_k, self.n, [self.line.center, self.line.count, self.S, self.speed,
                                    self.t, self.steps, self.pos], device=DEV)

    def decide(self, sr, done) -> None:
        wp.launch(_done_k, self.n, [sr.events.laps, sr.valid, self.period, self.t,
                                    self.steps, done], device=DEV)


@pytest.mark.parametrize("kind", ["track", "gates"])
def test_the_rl_loop_in_one_cpu_graph_equals_an_eager_twin(kind):
    """ONE caller graph per RL step ``[physics stub, course.step(), done kernel,
    course.regenerate(mask=inputs["regen_mask"], advance=True)]`` recorded into
    ``wp.ScopedCapture("cpu")`` (the masked regeneration recorded unconditionally, so
    every replay runs it, also with an all-zero mask), replayed 64 times == an eager twin
    that regenerates only when an env is done, bitwise, every buffer, every step."""
    n = 8

    def make():
        if kind == "track":
            return Course.track(TrackGenConfig(num_envs=n, device=DEV),
                                checkpoint_spacing=0.5, collision="segments",
                                extra=[toys.CurvatureLookahead(6), tools.Localizer()],
                                seeds=4)
        return Course.gates(GateGenConfig(num_envs=n, device=DEV, gate_width=0.2,
                                          scale=3.0, generator="checkpoint"),
                            collision=0.02, localize_window=4, seeds=4)

    a, b = make(), make()
    ag_a, ag_b = _Agents(_line(a), n), _Agents(_line(b), n)
    for c, ag in ((a, ag_a), (b, ag_b)):
        q = wp.array(np.tile(np.float32([0, 0, 0, 1]), (n, 1)), dtype=wp.quatf, device=DEV)
        he = wp.array(np.full((n, 2), 0.02, np.float32), dtype=wp.vec2f, device=DEV)
        c.bind(position=ag.pos, orientation=q, half_extents=he)
    done_a = a.inputs["regen_mask"]
    done_b = wp.zeros(n, dtype=wp.int32, device=DEV)
    sr_a, sr_b = a.step(), b.step()
    with wp.ScopedCapture(device=DEV) as cap:
        ag_a.move()
        sr = a.step()
        ag_a.decide(sr, done_a)
        a.regenerate(mask=done_a, advance=True)
    assert sr is sr_a
    regen = idle = 0
    for k in range(64):
        wp.capture_launch(cap.graph)
        ag_b.move()
        b.step()
        ag_b.decide(sr_b, done_b)
        if done_b.numpy().any():
            b.enqueue_generate(mask=done_b, advance=True)
            regen += 1
        else:
            idle += 1
        sa = {**_state(a), "pos": _b(ag_a.pos), "t": _b(ag_a.t), "done": _b(done_a)}
        sb = {**_state(b), "pos": _b(ag_b.pos), "t": _b(ag_b.t), "done": _b(done_b)}
        bad = [x for x in sa if sa[x] != sb[x]]
        assert not bad, (k, bad[:5])
    assert regen > 0 and idle > 0 and int(b.seeds.epoch.numpy().max()) >= 2


# ---------------------------------------------------------------------------------------
# 10. The CurvatureLookahead toy
# ---------------------------------------------------------------------------------------

_PUBLIC = ("track_gen.plugin", "track_gen.ops", "track_gen.exec", "track_gen.results",
           "track_gen.tools", "track_gen.testing", "track_gen.localize")


def test_the_curvature_lookahead_toy_plugs_in_with_public_imports_only():
    c = Course.track(_cfg(), checkpoint_spacing=0.5,
                     extra=[toys.CurvatureLookahead(6), tools.Localizer()], seeds=1)
    pos = _bind_all(c)
    pos.assign(_pos_on(c.result, 2))
    sr = c.step()
    look = sr["acme.lookahead"]
    valid = c.result.valid.numpy() != 0
    assert valid.any() and np.isfinite(look.kappa.numpy()[valid]).all()
    # A forced-invalid env (valid = 0, geometry kept) answers NaN (the sentinels).
    v = c.result.valid.numpy()
    v[1] = 0
    c.result.valid.assign(v)
    c._refresh_geometry(LaunchContext.for_call(DEV))
    k = c.step()["acme.lookahead"].kappa.numpy()
    assert np.isnan(k[1]) and np.isfinite(k[v != 0]).all()
    # The toy's module imports only the public surface (and warp / the stdlib).
    tree = ast.parse(open(toys.__file__).read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods |= {node.module} | {f"{node.module}.{a.name}" for a in node.names
                                     if node.module == "track_gen"}
    tg = sorted(m for m in mods if m.split(".")[0] == "track_gen" and m != "track_gen")
    assert tg and all(m.startswith(_PUBLIC) for m in tg), tg
    other = {m.split(".")[0] for m in mods} - {"track_gen", "warp", "__future__",
                                               "dataclasses", "math", "typing"}
    assert not other, other


# ---------------------------------------------------------------------------------------
# 11. The contract rows
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("label,make", [
    ("track sdf + heightfield + props + localizer + toy",
     lambda: Course.track(_cfg(), checkpoint_spacing=0.5, collision="sdf",
                          extra=[tools.HeightFieldBake(16), tools.Props(0.3),
                                 tools.Localizer(), toys.CurvatureLookahead(6)], seeds=1)),
    ("gates posts", lambda: Course.gates(_gcfg(), collision=0.03, seeds=1)),
    ("gates frames", lambda: Course.gates(
        _gcfg(), collision=tools.FrameCollision(0.05, 0.02, 0.02), seeds=1))])
def test_no_bound_array_in_any_regeneration_body(label, make):
    c = make()
    pos = _vec3(E)
    q, he = _boxes(E)
    c.bind(pos, q, he)
    bound = BufferSet()
    for i, a in enumerate((pos, q, he)):
        bound.add(a, f"in{i}")
    ctx = c._runner.context()
    m = _i32([1, 0, 1, 0, 0, 0])
    hits = {}
    for what, fn in [("plain", lambda: c._enqueue_regenerate(ctx)),
                     ("advance", lambda: c._enqueue_regenerate(ctx, advance=True)),
                     ("masked", lambda: c._enqueue_regenerate(ctx, advance=True, mask=m)),
                     ("reset", lambda: c.reset(m)), ("step", lambda: c.step())]:
        with trace_launches() as t:
            fn()
        hits[what] = sum(1 for r in t for _, p in r.arrays
                         if p and bound.locate(p) is not None)
    assert hits["step"] > 0 and all(hits[k] == 0 for k in ("plain", "advance", "masked",
                                                           "reset")), hits


def test_the_kit_records_the_course_chain():
    """``capture.caller`` / ``capture.self_reset`` record what a Course records:
    ``any_set`` first, the masked generation + on_generate + reset under
    ``skip_unless(flag)``, then step; every fourth done mask is all zero."""
    from track_gen.testing.components import _DeviceRun, _Kit

    kit = _Kit(lambda cc: tools.Progress(), (lambda cc: tools.Checkpoints(0.5),), "track",
               E, 8, np.arange(E, dtype=np.int32), None, ())
    dr = _DeviceRun(kit, DEV)
    prod = dr.producer()
    run = dr.run_on(prod)
    kit.drive(run, 0)
    done = _i32([0, 1, 0, 0, 1, 0])
    flag = wp.zeros(1, dtype=wp.int32, device=DEV)
    seen = []
    real = LaunchContext.skip_unless

    def spy(self, f, body):
        seen.append(f)
        return real(self, f, body)

    ctx = LaunchContext.for_call(DEV)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LaunchContext, "skip_unless", spy)
        with trace_launches() as tr:
            _caller_chain(ctx, run, prod.program, done, flag, reset=True)
    ks = [k.split(".")[-1] for k in tr.kernels()]
    assert seen == [flag] and ks[:3] == ["_fill_i32_k", "_any_set_k", "_advance_masked_k"]
    assert ks[-2:] == ["_progress_reset_k", "_progress_update_k"]
    assert "_place_checkpoints_k" in ks
    with trace_launches() as tr:
        assert _caller_chain(ctx, run, prod.program, done, flag, reset=False) is False
    assert "_progress_reset_k" not in [k.split(".")[-1] for k in tr.kernels()]
    # skip=True (the eager twin of a replay whose conditional node skips the body):
    # any_set, then step only.
    seen.clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LaunchContext, "skip_unless", spy)
        with trace_launches() as tr:
            assert _caller_chain(ctx, run, prod.program, done, flag, reset=True,
                                 skip=True) is False
    assert seen == [] and [k.split(".")[-1] for k in tr.kernels()] == \
        ["_fill_i32_k", "_any_set_k", "_progress_update_k"]
    sched = [_done_schedule(i, 8) for i in range(8)]
    assert not any(sched[3]) and not any(sched[7]) and all(any(s) for s in sched[:3])


def test_check_component_flags_an_input_read_in_the_body():
    for toy in (toys.BrokenReadsInputInGenerate, toys.BrokenReadsInputInReset):
        report = check_component(lambda cc, t=toy: t(), devices=[DEV], num_envs=E, steps=8)
        assert report.failed_names == ["hot_path.buffers"], str(report)
        assert "bound input 'position'" in report.result("hot_path.buffers").message


# ---------------------------------------------------------------------------------------
# 12. A minimal Producer; the Producer surface
# ---------------------------------------------------------------------------------------

class _Minimal:
    """Only the required Producer members."""

    def __init__(self, gen) -> None:
        self._g = gen.program
        self.result, self.layout = self._g.result, self._g.layout

    def enqueue(self, ctx, *, advance=False, mask=None):
        return self._g.enqueue(ctx, advance=advance, mask=mask)


class _Host(_Minimal):
    capture_class = CaptureClass.HOST


class _Bounded(_Minimal):
    capacity = {"max_loop_length": 12.0}
    bounds = {"size_hi": 3.0}


def _rows(t) -> dict:
    return {k: v.numpy().reshape(E, -1).copy() for k, v in result_arrays(t).items()
            if v.shape[0] % E == 0 and v.shape[0] > E}


def test_a_minimal_producer_drives_a_course():
    c = Course(_Minimal(TrackGenerator(_cfg(), seeds=2)),
               [tools.Checkpoints(0.5, max_checkpoints=70), tools.Progress()])
    assert c.seeds is None and c.rng is None and c._runner.name == "course[_Minimal]"
    before, cp0 = _rows(c.result), c.checkpoints.position.numpy().reshape(E, -1).copy()
    c.regenerate(mask=_i32([0, 1, 0, 0, 0, 0]), advance=True)
    after, cp1 = _rows(c.result), c.checkpoints.position.numpy().reshape(E, -1)
    keep = [e for e in range(E) if e != 1]
    for k in before:
        assert before[k][keep].tobytes() == after[k][keep].tobytes(), k
    assert before["center"][1].tobytes() != after["center"][1].tobytes()
    assert cp0[1].tobytes() != cp1[1].tobytes()
    with trace_launches() as tr:
        _raises(TypeError, lambda: c.generate(seeds=3), "no SeedState")
    assert len(tr) == 0
    h = Course(_Host(TrackGenerator(_cfg(), seeds=2)),
               [tools.Checkpoints(0.5, max_checkpoints=70)])
    h.regenerate()
    with pytest.raises(NotCapturableError):
        with wp.ScopedCapture(device=DEV):
            h.regenerate(mask=h.inputs["regen_mask"], advance=True)
    bc = Course(_Bounded(TrackGenerator(_cfg(), seeds=2)), [tools.Checkpoints(0.5)])
    assert bc.checkpoint_sampler.max_checkpoints == 24          # ceil(12 / 0.5)
    cc = tools.ComponentContext.for_producer(_Bounded(TrackGenerator(_cfg(), seeds=2)))
    assert dict(cc.capacity) == {"max_loop_length": 12.0} and \
        dict(cc.bounds) == {"size_hi": 3.0}
    with pytest.raises(ValueError, match="bound 'size_hi'"):
        tools.ComponentContext(cc.layout, bounds={"size_hi": 0.0})


class _BoundedWrapper(_Bounded):
    """A bounded producer that keeps the generator it wraps as ``self.generator``."""

    def __init__(self, gen) -> None:
        super().__init__(gen)
        self.generator = gen


class _BoundedUnrelated(_Bounded):
    """A bounded producer with an unrelated attribute named ``program``."""

    program = "v2"


@pytest.mark.parametrize("cls", [_BoundedWrapper, _BoundedUnrelated])
def test_for_producer_reads_a_third_party_producers_own_declarations(cls):
    """Only a real Course is unwrapped to its generator and only a real Program is a
    plan: a producer's own ``capacity`` / ``bounds`` win over an attribute that merely
    shares a name."""
    gen = TrackGenerator(_cfg(), seeds=2)
    cc = tools.ComponentContext.for_producer(cls(gen))
    assert dict(cc.capacity) == {"max_loop_length": 12.0}
    assert dict(cc.bounds) == {"size_hi": 3.0}
    c = Course(cls(TrackGenerator(_cfg(), seeds=2)), [tools.Checkpoints(0.5)])
    assert c.checkpoint_sampler.max_checkpoints == 24            # ceil(12 / 0.5)


class _Unrelated(_Minimal):
    """Attributes named like Producer extras that are not: a host rng, a program
    string."""

    program = "v2"

    def __init__(self, gen) -> None:
        super().__init__(gen)
        self.rng = np.random.default_rng(0)
        self.seeds = gen.program.seeds


class _RngNoSeeds(_Minimal):
    def __init__(self, gen) -> None:
        super().__init__(gen)
        self.rng = PerEnvSeededRNG(seeds=1, num_envs=E, device=DEV)


class _LazyUnchecked(_Minimal):
    """A third-party producer whose ``prepare()`` does not check for a capture."""

    prepared = False

    def prepare(self) -> None:
        self.calls = getattr(self, "calls", 0) + 1
        self.prepared = True


def test_unrelated_program_and_rng_attributes_are_not_producer_members():
    gen = TrackGenerator(_cfg(), seeds=2)
    c = Course(_Unrelated(gen), [tools.Checkpoints(0.5, max_checkpoints=70)])
    assert c.rng is None and c._producer is c.generator
    c.generate(seeds=5)                                  # SeedState.set, no rng
    ref = Course.track(_cfg(), checkpoint_spacing=0.5, max_checkpoints=70, seeds=5)
    assert _tree(c.result, "r") == _tree(ref.result, "r")
    r = Course(_RngNoSeeds(TrackGenerator(_cfg(), seeds=2)),
               [tools.Checkpoints(0.5, max_checkpoints=70)])
    assert isinstance(r.rng, PerEnvSeededRNG) and r.seeds is None
    with trace_launches() as tr:
        _raises(TypeError, lambda: r.generate(seeds=3), "no SeedState")
    assert len(tr) == 0                                  # refused before any launch


def test_a_capture_is_refused_before_the_producer_prepares():
    lazy = _LazyUnchecked(TrackGenerator(_cfg(), seeds=2))
    with wp.ScopedCapture(device=DEV):
        _raises(CaptureError, lambda: Course(lazy, [tools.Checkpoints(0.5,
                                                                      max_checkpoints=70)]),
                "outside any")
    assert not hasattr(lazy, "calls") and lazy.prepared is False
    Course(lazy, [tools.Checkpoints(0.5, max_checkpoints=70)])
    assert lazy.calls == 1


def test_the_producer_surface_members():
    import track_gen.pipeline as pipeline

    assert "Producer" in pipeline.__all__ and Producer.__doc__
    prog = build(presets.track(num_envs=E, device=DEV), seeds=7, prepare=False)
    assert prog.prepared is False and prog.default_stream is None
    c = Course(prog, [tools.Checkpoints(0.5), tools.Progress()])  # prepared first
    assert prog.prepared is True and c.checkpoint_sampler.max_checkpoints == 70
    rng = PerEnvSeededRNG(seeds=1, num_envs=E, device=DEV)
    assert TrackGenerator(_cfg(), rng).rng is rng
    assert TrackGenerator(_cfg(), seeds=1).rng is None
    assert GateGenerator(_gcfg(), seeds=rng).rng is rng
    c.regenerate(mask=c.inputs["regen_mask"], advance=True)
    c.generate(seeds=11)
    ref = Course(build(presets.track(num_envs=E, device=DEV), seeds=11),
                 [tools.Checkpoints(0.5), tools.Progress()])
    assert _tree(c.result, "r") == _tree(ref.result, "r")


# ---------------------------------------------------------------------------------------
# 13. generate(seeds=) by seed owner
# ---------------------------------------------------------------------------------------

def test_generate_seeds_follows_the_seed_owner():
    o1 = Course.track(_cfg(), checkpoint_spacing=0.5, collision="segments", seeds=3)
    o1.regenerate(advance=True)
    o1.generate(seeds=40)
    o2 = Course.track(_cfg(), checkpoint_spacing=0.5, collision="segments", seeds=40)
    assert _tree(o1.result, "r") == _tree(o2.result, "r")
    assert _b(o1.seeds.epoch) == _b(o2.seeds.epoch) and not o1.seeds.epoch.numpy().any()
    arr = _i32(np.arange(E) + 40)
    o1.generate(seeds=arr)
    assert _tree(o1.result, "r") == _tree(o2.result, "r")
    bound = Course.track(_cfg(), checkpoint_spacing=0.5, seeds=_i32(np.arange(E)))
    with trace_launches() as tr:
        _raises(RuntimeError, lambda: bound.generate(seeds=4), "PerEnvSeededRNG")
    assert len(tr) == 0
    r = Course.track(_cfg(), checkpoint_spacing=0.5,
                     seeds=PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    with trace_launches() as tr:
        r.generate(seeds=40)
    ks = [k.split(".")[-1] for k in tr.kernels()]
    assert ks[0] == "_seed_range_k" and \
        ks.index("_zero_all_k") < ks.index("_effective_seed_k")    # reseed, then the body
    assert _tree(r.result, "r") == _tree(o2.result, "r")
    assert r.rng.seeds_warp.numpy().tolist() == list(range(40, 40 + E))


# ---------------------------------------------------------------------------------------
# 14. prepare=False and prepare()
# ---------------------------------------------------------------------------------------

def _drive(c, pos, steps: int = 12):
    ctr = c.checkpoints.position.numpy().reshape(E, -1, 3)
    cnt = c.checkpoints.count.numpy()
    r = None
    for k in range(steps):
        pos.assign(np.stack([ctr[e, k % max(int(cnt[e]), 1)] for e in range(E)]))
        r = c.step()
    return r


def test_prepare_false_defers_the_warm_up_and_keeps_unmasked_state():
    with trace_launches() as tr:
        lazy = Course.track(_cfg(), checkpoint_spacing=0.5, seeds=3, prepare=False)
    eager = Course.track(_cfg(), checkpoint_spacing=0.5, seeds=3)
    assert not lazy._runner.prepared and eager._runner.prepared
    assert eager._zero_mask is None and lazy._zero_mask is not None
    with trace_launches() as tr2:
        Course.track(_cfg(), checkpoint_spacing=0.5, seeds=3)
    assert len(tr2) > len(tr)                                  # the warm-up is deferred
    with pytest.raises(CaptureError, match="prepare=False"):
        with wp.ScopedCapture(device=DEV):
            lazy.regenerate(mask=lazy.inputs["regen_mask"], advance=True)
    for c in (lazy, eager):
        pos = _vec3(E)
        c.bind(position=pos)
        before = _drive(c, pos).events.progress.numpy().copy()
        c.regenerate(mask=_i32([1, 0, 0, 0, 0, 0]), advance=True)
        after = c.step().events.progress.numpy()
        assert before.any() and before[1:].tolist() == after[1:].tolist()
    assert lazy._runner.prepared


def test_explicit_prepare_changes_no_byte_and_is_idempotent():
    c = Course.track(_cfg(), checkpoint_spacing=0.5, seeds=3, extra=[tools.Localizer()],
                     prepare=False)
    pos = _vec3(E)
    c.bind(position=pos)
    _drive(c, pos, 7)
    s0 = _state(c)
    c.prepare()
    assert c._runner.prepared and _state(c) == s0
    with trace_launches() as tr:
        c.prepare()
    assert len(tr) == 0
    p = Course.track(_cfg(), checkpoint_spacing=0.5, seeds=3)
    pp = _vec3(E)
    p.bind(position=pp)
    st = _drive(p, pp, 6).events.progress.numpy().copy()
    with trace_launches() as tr:
        p.prepare()
    assert len(tr) == 0 and p.step().events.progress.numpy().tolist() == st.tolist()


# ---------------------------------------------------------------------------------------
# 15. The hot path
# ---------------------------------------------------------------------------------------

def test_the_hot_path_of_a_composed_course():
    c = _track_toy()
    _bind_all(c)
    m = _i32([1, 0, 1, 0, 0, 1])
    ctx = LaunchContext.for_call(DEV)
    c.regenerate(mask=m, advance=True)            # module loads outside the guard
    c.step()
    c.reset(m)
    with hot_path_guard(strict=False) as g:
        c._enqueue_regenerate(ctx, advance=True, mask=m)
        c._enqueue_regenerate(ctx, advance=True)
        c.step()
        c.reset(m)
        c.reset()
        c.step()
    assert not g.violations, g.violations[:3]
    for fn in (lambda: c.step(), lambda: c.reset(m),
               lambda: c._enqueue_regenerate(ctx, advance=True, mask=m)):
        t1, t2 = trace_launches(), trace_launches()
        with t1:
            fn()
        with t2:
            fn()
        assert not t1.diff(t2)


# ---------------------------------------------------------------------------------------
# 16. CPU graph lifetime
# ---------------------------------------------------------------------------------------

def test_a_cpu_graph_keeps_the_arrays_it_recorded_after_the_course_is_dropped():
    c = Course.track(_cfg(), checkpoint_spacing=0.5, extra=[tools.Localizer()], seeds=3)
    c.bind(position=_vec3(E))
    wc = weakref.ref(c)
    arrays = [weakref.ref(a) for a in (c.result.center, c.progress._laps,
                                       c.localizer._last, c.checkpoints.position)]
    with wp.ScopedCapture(device=DEV) as cap:
        c.step()
        c.regenerate(mask=c.inputs["regen_mask"], advance=True)
    del c
    gc.collect()
    assert wc() is None and all(w() is not None for w in arrays)
    for _ in range(3):
        wp.capture_launch(cap.graph)
    del cap
    gc.collect()
    assert not any(w() is not None for w in arrays)


def test_recorded_calls_anchor_the_arrays_their_graph_reads(monkeypatch):
    """What each call hands ``_compat.anchor`` (a no-op unless Warp registered the
    capture; the CUDA lane checks the graph): ``step()`` the course and the binding it
    read (a rebind makes a new one), ``reset(m)`` / ``regenerate(mask=m)`` /
    ``enqueue_generate(mask=m)`` the mask, ``generate(seeds=arr)`` the seeds."""
    seen = []
    monkeypatch.setattr(_compat, "anchor", lambda stream, *owners: seen.append(owners))
    c = Course.track(_cfg(), checkpoint_spacing=0.5, seeds=1)
    c.bind(position=_vec3(E))
    first = c._bind_args
    c.step()
    assert seen[-1][0] is c and seen[-1][1] is first
    c.bind(position=_vec3(E))
    c.step()
    assert c._bind_args is not first and seen[-1][1] is c._bind_args
    m = _i32([1, 0, 0, 1, 0, 0])
    c.reset(m)
    assert seen[-1][0] is c and seen[-1][1] is m
    for call in (lambda: c.regenerate(mask=m, advance=True),
                 lambda: c.enqueue_generate(mask=m)):
        seen.clear()
        call()
        assert any(o is m for owners in seen for o in owners)
    arr = _i32(np.arange(E) + 9)
    seen.clear()
    c.generate(seeds=arr)
    assert any(o is arr for owners in seen for o in owners)


# ---------------------------------------------------------------------------------------
# 17. The default-stream guard (the logic; the CUDA lane runs it for real)
# ---------------------------------------------------------------------------------------

class _FakeStream:
    def __init__(self, handle: int, capturing: bool = False) -> None:
        self.cuda_stream = handle
        self.is_capturing = capturing


class _FakeCtx:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.device = types.SimpleNamespace(is_cuda=True, alias="cuda:0")
        self.capturing = bool(stream.is_capturing)


def test_the_default_stream_guard(monkeypatch):
    c = Course.track(_cfg(), checkpoint_spacing=0.5, seeds=1)
    ds = _FakeStream(0x10)
    cur = _FakeStream(0x20, capturing=True)
    monkeypatch.setattr(c._runner, "_default_stream", ds)
    monkeypatch.setattr(c._runner, "context", lambda stream=None: _FakeCtx(
        ds if stream is None else stream))
    monkeypatch.setattr(wp, "get_stream", lambda device=None: cur)
    monkeypatch.setattr(course_mod, "_torch_capturing", lambda dev: None)
    with pytest.raises(CaptureError, match=r"built with stream= .* Pass stream="):
        c._ctx(None)
    explicit = _FakeStream(0x30)
    assert c._ctx(explicit).stream is explicit                  # stream= is the escape
    monkeypatch.setattr(_compat, "library_capture", lambda s: s is cur)
    assert c._ctx(None).stream is ds                            # the library's capture
    monkeypatch.setattr(_compat, "library_capture", lambda s: False)
    cur.is_capturing = False
    assert c._ctx(None).stream is ds                            # nothing captures
    monkeypatch.setattr(course_mod, "_torch_capturing", lambda dev: object())
    with pytest.raises(CaptureError):
        c._ctx(None)                                            # torch captures here
    monkeypatch.setattr(c._runner, "_default_stream", None)
    assert c._ctx(None).stream is ds                            # no default: no guard
