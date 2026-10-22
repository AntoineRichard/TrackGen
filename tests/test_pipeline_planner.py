"""The pipeline planner, PipelineSpec and Program (roadmap PR10a, CPU).

``track_gen.pipeline`` describes the track pipeline as data (``StageRef`` /
``BatchSpec`` / ``PipelineSpec``), plans and allocates it once (``build``) and runs it
as a ``Program``. These tests cover the default stage list, the pure-data spec
operations and their serialisation, every planner rule (all raised before anything is
allocated), channel versioning and exports, the ``TrackGenerator._scratch`` view, and
the Program's own calls. Byte identity with the PR9 composition is
``test_pipeline_parity``.
"""
from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest
import warp as wp

from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen import stages
from track_gen._src import warp_pipeline
from track_gen.exec import CaptureError, LaunchContext
from track_gen.pipeline import (BatchSpec, PipelineSpec, PlanError, Program, StageRef,
                                build, legacy)
from track_gen.plugin import Layout, NoParams, Params, Stage, dynamic, registry, static
from track_gen.plugin.stage import CaptureClass, StageBuilder
from track_gen.stages import params as sp
from track_gen.testing.contract import no_allocation

DEV = "cpu"
E = 2
TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
                "valid", "count", "winding", "truncated", "generation")
DEFAULT_IDS = ("generator", "resample", "relax", "uniform", "width", "frames", "offset",
               "elevation", "validity", "lift")


def _cfg(**kw):
    return TrackGenConfig(device=DEV, num_envs=E, **kw)


def _spec(**kw) -> PipelineSpec:
    return legacy.to_pipeline_spec(_cfg(**kw))


def _snap(track) -> dict:
    return {f: getattr(track, f).numpy().tobytes() for f in TRACK_FIELDS
            if getattr(track, f) is not None}


def _names(spec) -> list:
    return [(r.slot_id, f"{r.kind}.{r.name}") for r in spec.stages]


# ---------------------------------------------------------------------------------------
# The default stage list
# ---------------------------------------------------------------------------------------

def test_default_stage_list_flat():
    spec = _spec()
    assert spec.ids == DEFAULT_IDS
    assert _names(spec) == [
        ("generator", "generator.bezier"), ("resample", "resample.constant_spacing"),
        ("relax", "relax.xpbd"), ("uniform", "resample.uniform"),
        ("width", "width.constant"), ("frames", "frames.default"),
        ("offset", "offset.default"), ("elevation", "elevation.flat"),
        ("validity", "validity.default"), ("lift", "lift.flat")]
    assert spec.batch == BatchSpec(num_envs=E, device="cpu", n_max=384, closed=True,
                                   scale=1.0)
    gen, resample, relax = spec.stages[:3]
    assert dict(gen.capacity) == {"raw": 256}
    assert gen.params == registry.get("generator", "bezier").params.from_legacy(_cfg())
    assert resample.params == sp.ConstantSpacingParams(spacing=0.6 * 0.1)
    assert relax.params == sp.XpbdParams.from_legacy(_cfg())
    assert spec.stages[7].params is None
    assert spec.stages[9].params == sp.LiftFlatParams(z_base=0.0)


@pytest.mark.parametrize("profile,params", [
    ("uniform", sp.ElevationUniformParams), ("random_walk", sp.ElevationRandomWalkParams),
    ("noise", sp.ElevationNoiseParams)])
def test_default_stage_list_elevated(profile, params):
    cfg = _cfg(z_profile=profile, z_max=0.2)
    spec = legacy.to_pipeline_spec(cfg)
    assert spec.ids == DEFAULT_IDS
    elevation, lift = spec.stages[7], spec.stages[9]
    assert (elevation.kind, elevation.name) == ("elevation", profile)
    assert elevation.params == params.from_legacy(cfg)
    assert (lift.kind, lift.name, lift.params) == ("lift", "elevated", None)


def test_default_stage_list_without_relax():
    spec = _spec(relax_enable=False)
    assert spec.stages[2] == StageRef("relax", "none")
    assert spec.ids == DEFAULT_IDS


def test_to_pipeline_spec_raises_the_track_generator_errors():
    with pytest.raises(ValueError, match=r"TrackGenConfig.relax_solver='energy' is an "
                                         r"oracle-only backend \(tests/_oracle\), not a "
                                         r"relax stage; TrackGenerator runs a registered "
                                         r"relax stage: \['none', 'xpbd'\]"):
        legacy.to_pipeline_spec(_cfg(relax_solver="energy"))
    with pytest.raises(ValueError, match=r"TrackGenConfig.relax_solver: unknown relax "
                                         r"'bogus'; available: \['none', 'xpbd'\]"):
        legacy.to_pipeline_spec(_cfg(relax_solver="bogus"))
    with pytest.raises(ValueError, match=r"TrackGenConfig.z_profile: unknown elevation "
                                         r"'bogus'; available: \['flat', 'noise', "
                                         r"'random_walk', 'uniform'\]"):
        legacy.to_pipeline_spec(_cfg(z_profile="bogus"))
    with pytest.warns(DeprecationWarning):
        cfg = _cfg(smooth_finish=True)
    with pytest.raises(ValueError, match="smooth_finish=True"):
        legacy.to_pipeline_spec(cfg)
    with pytest.raises(ValueError, match="unknown generator"):
        legacy.to_pipeline_spec(_cfg(generator="nope"))


def test_the_flat_params_validate_with_the_config_messages():
    layout = Layout(E, 384, "cpu")
    for p, match in [
        (sp.XpbdParams(relax_cheby_rho=1.0), r"relax_cheby_rho must be in \[0, 1\)"),
        (sp.XpbdParams(relax_sep_every=0), "relax_sep_every must be >= 1"),
        (sp.XpbdParams(half_width=0.0), "half_width must be > 0"),
        (sp.ConstantSpacingParams(spacing=0.0), "spacing must be > 0"),
        (sp.ElevationUniformParams(z_min=1.0), "z_min must be <= z_max"),
        (sp.ElevationRandomWalkParams(z_control_points=2), "z_control_points must be >= 3"),
        (sp.ElevationNoiseParams(z_noise_harmonics=0), "z_noise_harmonics must be >= 1"),
        (sp.ValidityParams(z_valid_grade=-1.0), "z_valid_grade must be >= 0"),
    ]:
        with pytest.raises(ValueError, match=match):
            p.validate(layout)


# ---------------------------------------------------------------------------------------
# PipelineSpec: pure data
# ---------------------------------------------------------------------------------------

def test_stage_ref_is_frozen_data():
    ref = StageRef("relax", "xpbd", params={"relax_iters": 3}, ports={"xy": "cl.xy"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.kind = "x"
    with pytest.raises(TypeError):
        ref.params["relax_iters"] = 4                        # a frozen mapping
    assert ref.slot_id == "relax" and StageRef("a", "b", id="c").slot_id == "c"
    with pytest.raises(ValueError, match="without '/'"):
        StageRef("relax", "xpbd", id="a/b")
    with pytest.raises(TypeError, match="non-empty string"):
        StageRef("", "xpbd")
    with pytest.raises(TypeError, match="capacity"):
        StageRef("generator", "bezier", capacity={"raw": 1.5})


def test_edit_operations():
    spec = _spec()
    none = spec.replace("relax", StageRef("relax", "none"))
    assert none.stages[2] == StageRef("relax", "none") and spec.stages[2].name == "xpbd"
    kept = spec.replace("uniform", StageRef("resample", "uniform"))
    assert kept.stages[3].id == "uniform" and kept.ids == DEFAULT_IDS   # slot id kept
    extra = StageRef("width", "constant", id="w2")
    assert spec.insert_after("width", extra).ids[4:6] == ("width", "w2")
    assert spec.insert_before("width", extra).ids[4:6] == ("w2", "width")
    assert spec.remove("relax").ids == DEFAULT_IDS[:2] + DEFAULT_IDS[3:]
    for op in (lambda: spec.remove("nope"), lambda: spec.replace("nope", extra),
               lambda: spec.insert_after("nope", extra),
               lambda: spec.insert_before("nope", extra)):
        with pytest.raises(ValueError, match=r"no stage 'nope'.*stages: \['generator'"):
            op()
    assert spec.ids == DEFAULT_IDS                       # never mutated


@pytest.mark.parametrize("kw", [{}, dict(z_profile="noise", z_max=0.3,
                                         z_noise_amplitude=0.1),
                                dict(style_sampling=True, rad_range=(0.1, 0.3)),
                                dict(relax_enable=False, z_profile="uniform", z_max=0.1)])
def test_to_dict_round_trip_is_json_and_byte_identical(kw):
    spec = _spec(**kw)
    d = spec.to_dict()
    assert d["format"] == "track_gen.pipeline/1"
    text = json.dumps(d)
    back = PipelineSpec.from_dict(json.loads(text))
    assert back.to_dict() == d and back.ids == spec.ids
    a = build(spec, seeds=11).run(sync=True)
    b = build(back, seeds=11).run(sync=True)
    assert _snap(a) == _snap(b)


def test_from_dict_rejects_another_format():
    with pytest.raises(ValueError, match="unsupported pipeline format"):
        PipelineSpec.from_dict({"format": "x", "batch": {}, "stages": []})


def test_from_dict_names_the_missing_key():
    d = _spec().to_dict()
    for edit, match in [
        (lambda d: d.pop("batch"), r"pipeline data is missing 'batch'"),
        (lambda d: d["batch"].pop("n_max"), r"batch is missing 'n_max'"),
        (lambda d: d["stages"][3].pop("kind"), r"stages\[3\] is missing 'kind'"),
        (lambda d: d["stages"].__setitem__(1, "relax"), r"stages\[1\] must be a mapping"),
        (lambda d: d.__setitem__("stages", "relax"), "stages must be a list"),
    ]:
        bad = json.loads(json.dumps(d))
        edit(bad)
        with pytest.raises(ValueError, match=match):
            PipelineSpec.from_dict(bad)


# ---------------------------------------------------------------------------------------
# build(): standalone, out=, seeds
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{}, dict(relax_enable=False),
                                dict(z_profile="random_walk", z_max=0.3, z_max_step=0.5)])
def test_standalone_build_equals_track_generator(kw):
    gen = TrackGenerator(_cfg(**kw), PerEnvSeededRNG(seeds=5, num_envs=E, device=DEV))
    program = build(_spec(**kw), seeds=5)
    assert isinstance(program, Program)
    assert _snap(program.result) == _snap(gen._track)      # prepare() already generated
    assert _snap(program.run(sync=True)) == _snap(gen.generate())
    assert program.run(advance=True) is program.result
    assert _snap(program.result) == _snap(gen.generate(advance=True))


def test_build_into_out_binds_the_track_buffers():
    out = warp_pipeline._inflate_warp_alloc(_cfg())[0]
    program = build(_spec(), seeds=5, out=out)
    assert program.result is out
    for field, ch in (("arclen", "cl.arclen2"), ("length", "cl.length2"),
                      ("valid", "track.valid"), ("count", "track.count"),
                      ("truncated", "cl.truncated"), ("center", "track.center")):
        assert program._channel(ch) is getattr(out, field)
    gen = TrackGenerator(_cfg(), PerEnvSeededRNG(seeds=5, num_envs=E, device=DEV))
    assert _snap(out) == _snap(gen._track)


def test_build_rejects_a_mismatched_out():
    out = warp_pipeline._inflate_warp_alloc(_cfg(N_max=200))[0]
    with pytest.raises(ValueError, match=r"out\.outer: expected a contiguous \[768\]"):
        build(_spec(), out=out)


def test_build_seeds_forms():
    a = build(_spec(), seeds=5).result.center.numpy()
    rng = PerEnvSeededRNG(seeds=5, num_envs=E, device=DEV)
    np.testing.assert_array_equal(build(_spec(), seeds=rng).result.center.numpy(), a)
    arr = wp.array([5, 6], dtype=wp.int32, device=DEV)
    np.testing.assert_array_equal(build(_spec(), seeds=arr).result.center.numpy(), a)
    p = build(_spec(), prepare=False)                     # seeds=None: 0 + arange(E)
    assert p.runner.prepared is False
    p.prepare()
    assert p.runner.prepared is True


@pytest.mark.parametrize("seeds", [np.array([5, 6], np.int32), [5, 6], 1.5, "5"])
def test_build_rejects_other_seed_types_before_allocating(seeds, no_alloc):
    with pytest.raises(TypeError, match=r"seeds must be None, an int, an \[E\] int32 "
                                        r"wp.array, a PerEnvSeededRNG or a SeedState"):
        build(_spec(), seeds=seeds)


def test_build_checks_seeds_before_the_track_and_channels(monkeypatch):
    from track_gen.stages import track as stages_track

    other = build(dataclasses.replace(
        _spec(), batch=dataclasses.replace(_spec().batch, num_envs=3)), prepare=False)
    monkeypatch.setattr(stages_track, "_alloc_track",
                        lambda layout: pytest.fail("allocated the Track"))
    with pytest.raises(ValueError, match="seeds"):           # SeedState's own check
        build(_spec(), seeds=wp.array([1, 2, 3], dtype=wp.int32, device=DEV))
    with pytest.raises(ValueError, match="a SeedState for 3 envs"):
        build(_spec(), seeds=other.seeds)


def test_run_prepares_a_program_built_without_prepare():
    want = _snap(build(_spec(), seeds=5).result)
    p = build(_spec(), seeds=5, prepare=False)
    assert p.run(policy="eager") is p.result
    assert p.runner.prepared is True and _snap(p.result) == want


def test_an_unprepared_program_refuses_to_be_recorded(monkeypatch):
    p = build(_spec(), seeds=5, prepare=False)
    capturing = types.SimpleNamespace(capturing=True, stream=None)   # a caller's capture
    monkeypatch.setattr(p.runner, "context", lambda stream=None: capturing)
    with pytest.raises(CaptureError, match=r"call Program.prepare\(\) outside"):
        p.run()
    with pytest.raises(CaptureError, match=r"call Program.prepare\(\) outside"):
        p.enqueue(capturing)
    assert p.runner.prepared is False
    assert int(p.result.generation.numpy().sum()) == 0      # nothing launched


# ---------------------------------------------------------------------------------------
# Planner rules (all raised before anything is allocated)
# ---------------------------------------------------------------------------------------

@pytest.fixture
def no_alloc(monkeypatch):
    """Fail on any allocation: the rule under test must fire before allocating."""
    for name in ("empty", "zeros", "full"):
        monkeypatch.setattr(wp, name, lambda *a, _n=name, **k: pytest.fail(f"wp.{_n}"))


@pytest.mark.parametrize("edit,match", [
    (lambda s: s.replace("relax", StageRef("relax", "nope")),
     r"stage 'relax' \(relax.nope\): unknown relax 'nope'; available: \['none', "
     r"'xpbd'\]"),
    (lambda s: s.replace("frames", StageRef("frames", "nope")),
     r"unknown frames stage 'nope'; built-in frames stages: \['default'\] \(kind "
     r"'frames' is not open to plugins\)"),
    (lambda s: s.insert_after("lift", StageRef("paint", "x")),
     r"unknown stage kind 'paint'; kinds: \['generator', 'elevation'"),
    (lambda s: s.insert_after("width", StageRef("width", "constant")),
     r"duplicate stage ids \['width'\]"),
    (lambda s: s.remove("generator"), "the first stage must be the generator"),
    (lambda s: s.insert_after("lift", StageRef("generator", "bezier", id="g2")),
     "exactly one generator stage, got 2"),
    (lambda s: s.remove("frames"),
     r"stage 'offset' \(offset.default\) reads 'cl.normal2' \(port 'normal'\), but no "
     r"earlier stage writes it"),
    (lambda s: s.replace("width", StageRef("width", "constant",
                                           params={"half_width": 0.2})),
     "the stages disagree on half_width"),
    (lambda s: s.replace("offset", dataclasses.replace(s.stages[6], ports={"nope": "cl.xy"})),
     r"ports \['nope'\] are not ports of the stage"),
    (lambda s: s.replace("offset", dataclasses.replace(s.stages[6],
                                                       ports={"normal": "cl.bogus"})),
     "rewired to unknown channel 'cl.bogus'"),
    (lambda s: s.replace("offset", dataclasses.replace(s.stages[6],
                                                       ports={"normal": "cl.kappa"})),
     "cannot be rewired to 'cl.kappa'"),
    (lambda s: s.remove("validity"),
     r"no stage writes Track.valid \(track.valid or valid.fail\), Track.fail_mask "
     r"\(valid.fail\): every Track field needs a writer"),
    (lambda s: s.remove("lift"),
     r"no stage writes Track.outer \(track.outer\), Track.center \(track.center\), "
     r"Track.inner \(track.inner\), Track.tangent \(track.tangent\), Track.normal "
     r"\(track.normal\)"),
    (lambda s: s.replace("elevation", StageRef("elevation", "uniform",
                                               params={"z_max": 0.5})),
     r"stage 'lift' \(lift.flat\) ignores cl.z written by stage 'elevation' "
     r"\(elevation.uniform\); use lift.elevated"),
    (lambda s: s.replace("relax", StageRef("relax", "xpbd", capacity={"raw": 8})),
     r"stage 'relax' \(relax.xpbd\): unknown capacity keys \['raw'\]; only the "
     r"generator takes capacity"),
    (lambda s: s.replace("generator", dataclasses.replace(s.stages[0],
                                                          capacity={"raw": 8, "n": 3})),
     r"stage 'generator' \(generator.bezier\): unknown capacity keys \['n'\]"),
])
def test_plan_errors(edit, match, no_alloc):
    spec = edit(_spec())
    with pytest.raises(PlanError, match=match):
        build(spec)


def test_batch_errors(no_alloc):
    spec = _spec()
    for batch, match in [
        (dataclasses.replace(spec.batch, num_envs=0), "num_envs must be an int >= 1"),
        (dataclasses.replace(spec.batch, n_max=2), "n_max must be an int >= 3"),
        (dataclasses.replace(spec.batch, closed=False), "open tracks"),
    ]:
        with pytest.raises(PlanError, match=match):
            build(dataclasses.replace(spec, batch=batch))
    with pytest.raises(PlanError, match="no stages"):
        build(PipelineSpec(spec.batch, ()))
    with pytest.raises(TypeError, match="expects a PipelineSpec"):
        build(spec.to_dict())


def test_params_errors(no_alloc):
    spec = _spec()
    with pytest.raises(ValueError, match=r"unknown params for stage 'relax' \(relax.xpbd\): "
                                         r"\['bogus'\]; valid fields: \['half_width'"):
        build(spec.replace("relax", StageRef("relax", "xpbd", params={"bogus": 1})))
    with pytest.raises(TypeError, match="params for stage 'relax' .* must be a XpbdParams"):
        build(spec.replace("relax", StageRef("relax", "xpbd", params=3)))
    with pytest.raises(ValueError, match=r"stage 'uniform' \(resample.uniform\) takes no "
                                         r"params"):
        build(spec.replace("uniform", StageRef("resample", "uniform", params={"x": 1})))
    with pytest.raises(ValueError, match="relax_cheby_gamma must be in"):
        build(spec.replace("relax", StageRef("relax", "xpbd",
                                             params={"relax_cheby_gamma": 0.0})))
    with pytest.raises(ValueError, match=r"relax_accel must be 'none' or 'chebyshev'"):
        build(spec.replace("relax", StageRef("relax", "xpbd",
                                             params={"relax_accel": "fast"})))
    with pytest.raises(ValueError, match="unknown generator_params"):
        build(spec.replace("generator", StageRef("generator", "bezier",
                                                 params={"bogus": 1})))


def test_build_inside_a_capture_raises_before_allocating(monkeypatch):
    from tests._fake_cuda import FakeCuda

    fake = FakeCuda(monkeypatch)
    fake.current.is_capturing = True
    for name in ("empty", "zeros", "full"):
        monkeypatch.setattr(wp, name, lambda *a, _n=name, **k: pytest.fail(f"wp.{_n}"))
    spec = PipelineSpec(BatchSpec(E, "cuda", 384), _spec().stages)
    with pytest.raises(CaptureError):
        build(spec)


# ---------------------------------------------------------------------------------------
# Toy stages: through the plugin registry (relax / width / elevation / validity) or the
# private catalogue (the other kinds)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _ramp_k(count: wp.array(dtype=wp.int32), n_max: int, slope: float,
            z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // n_max
    i = t - e * n_max
    if i < count[e]:
        z[t] = slope * float(i) / float(count[e])
    else:
        z[t] = 0.0


@wp.kernel
def _gain_fill_k(w: wp.array(dtype=wp.float32), hw: float, n_max: int,
                 gain: wp.array(dtype=wp.float32)):
    t = wp.tid()
    w[t] = hw * gain[t // n_max]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RampParams(Params):
    slope: float = static(0.2, lo=0.0)


class _Ramp(Stage):
    kind, name, Params = "elevation", "test.ramp", _RampParams
    reads = {"count": "cl.count"}
    writes = {"z": "cl.z"}

    def build(self, b, p):
        count, z = b.read("count"), b.write("z", wp.float32, per="cl")
        S, n, slope = int(b.layout.stride), b.E * int(b.layout.stride), float(p.slope)

        def enqueue(ctx):
            ctx.launch(_ramp_k, n, [count, S, slope, z])
        return enqueue


@dataclasses.dataclass(frozen=True, kw_only=True)
class _GainParams(Params):
    half_width: float = static(0.1)
    gain: float = dynamic(1.0, lo=0.0)


class _GainWidth(Stage):
    kind, name, Params = "width", "test.gain", _GainParams
    writes = {"half_width": "cl.half_width"}

    def build(self, b, p):
        w = b.write("half_width", wp.float32, per="cl")
        gain, hw, S = b.param("gain"), float(p.half_width), int(b.layout.stride)
        n = b.E * S

        def enqueue(ctx):
            ctx.launch(_gain_fill_k, n, [w, hw, S, gain])
        return enqueue


def _toy(**attrs):
    """A toy width stage (registered with ``registry.add``); ``attrs`` override class
    attributes / build."""
    body = attrs.pop("build", None)
    ns = dict(kind="width", name="test.toy", reads={}, writes={"half_width": "cl.half_width"})
    ns.update(attrs)

    def build_(self, b, p):
        if body is not None:
            return body(self, b, p)
        w = b.write("half_width", wp.float32, per="cl")
        return lambda ctx: ctx.fill(w, 0.1)
    ns["build"] = build_
    return type("Toy", (Stage,), ns)


@pytest.fixture
def catalogue():
    with stages._catalogue.scoped() as cat:
        yield cat


def test_a_toy_elevation_stage_runs_in_the_pipeline():
    registry.add(_Ramp)
    spec = _spec(z_profile="noise", z_max=1.0).replace(
        "elevation", StageRef("elevation", "test.ramp", params={"slope": 0.5}))
    p = build(spec, seeds=3)
    track = p.run(sync=True)
    count = track.count.numpy()
    z = track.center.numpy().reshape(E, -1, 3)[..., 2]
    for e in range(E):
        np.testing.assert_allclose(z[e, :count[e]], 0.5 * np.arange(count[e]) / count[e],
                                   rtol=1e-6)
    assert p.spec.stages[7].params == _RampParams(slope=0.5)
    assert list(p.params) == list(DEFAULT_IDS)


def test_a_dynamic_param_of_a_stage_is_live_and_latched():
    registry.add(_GainWidth)
    spec = _spec().replace("width", StageRef("width", "test.gain"))
    p = build(spec, seeds=3)
    live = p.params["width"]["gain"]
    assert live.dtype is wp.float32 and live.shape == (E,)
    np.testing.assert_allclose(p._channel("cl.half_width").numpy(), 0.1)
    p.params["width"].set("gain", 2.0)
    p.run(sync=True)
    np.testing.assert_allclose(p._channel("cl.half_width").numpy(), 0.2, rtol=1e-6)
    with pytest.raises(KeyError, match="static"):
        p.params["width"].set("half_width", 0.3)


def test_a_second_writer_must_read_the_channel(catalogue):
    # A width toy writing cl.xy is refused at registration (the kind write rule); a
    # catalogue-kind stage reaches the planner's rule.
    with pytest.raises(ValueError, match=r"Stage Toy \(kind 'width'\) writes 'cl.xy' "
                                         r"\(port 'half_width'\): a width stage may write "
                                         r"\['cl.half_width'\] and plugin channels"):
        registry.add(_toy(writes={"half_width": "cl.xy"}))
    catalogue.add(_toy(kind="offset", writes={"outer": "band.outer2", "xy": "cl.xy"},
                       build=lambda self, b, p: (lambda ctx: None)))
    with pytest.raises(PlanError, match=r"writes 'cl.xy', already written by stage "
                                        r"'uniform': a second writer must also read the "
                                        r"channel"):
        build(_spec().replace("offset", StageRef("offset", "test.toy")))


@pytest.mark.parametrize("attrs,exc,match", [
    (dict(build=lambda self, b, p: (lambda ctx: None)), PlanError,
     r"declares write ports \['half_width'\] but build\(\) did not call b.write"),
    (dict(build=lambda self, b, p: 3), TypeError, "must return an enqueue"),
    (dict(build=lambda self, b, p: b.write("half_width", wp.vec2f, per="cl")), TypeError,
     "channel 'cl.half_width' has dtype float32, got vec2f"),
    (dict(build=lambda self, b, p: b.write("half_width", wp.float32, per="env")), TypeError,
     "channel 'cl.half_width' is per='cl', got per='env'"),
    (dict(build=lambda self, b, p: b.write("nope")), ValueError,
     r"declares no write port 'nope'; declared ports: \['half_width'\]"),
    (dict(build=lambda self, b, p: b.read("nope")), ValueError,
     r"declares no read port 'nope'; declared read ports: \[\]"),
    (dict(build=lambda self, b, p: b.temp_like("nope")), ValueError,
     "declares no port 'nope'"),
])
def test_stage_rules(attrs, exc, match):
    registry.add(_toy(**attrs))
    with pytest.raises(exc, match=match):
        build(_spec().replace("width", StageRef("width", "test.toy")))


def test_catalogue_rules(catalogue):
    for attrs, match in [
        (dict(kind="generator"), "non-generator kind"),
        (dict(kind="width"), r"kind 'width' is open to plugins: register it with "
                             r"track_gen.plugin.registry.add"),
        (dict(writes={"z": "cl.bogus"}), r"unknown channels \['cl.bogus'\]"),
        (dict(writes={"s": "seed"}), "only the Program's prologue"),
        (dict(reads={"x": "cl.xy"}, writes={"x": "cl.z"}), "both read and write ports"),
        (dict(capture=CaptureClass.HOST), "STATIC"),
    ]:
        with pytest.raises(ValueError, match=match):
            catalogue.add(_toy(**{"kind": "frames", **attrs}))
    catalogue.add(_toy(kind="frames"))
    with pytest.raises(ValueError, match="already in the catalogue"):
        catalogue.add(_toy(kind="frames"))
    assert "test.toy" in stages.available("frames")
    assert stages.available() == ["frames", "lift", "offset", "resample"]
    assert stages.OPEN_KINDS == ("relax", "width", "elevation", "validity")


def test_input_and_salt_collisions_across_stages(monkeypatch):
    def with_input(self, b, p):
        b.input("difficulty", wp.float32, 0.5)
        b.salt("noise")
        w = b.write("half_width", wp.float32, per="cl")
        return lambda ctx: ctx.fill(w, 0.1)

    def input_only(self, b, p):
        b.input("difficulty", wp.float32, 0.5)
        b.salt("noise")
        return lambda ctx: None

    registry.add(_toy(build=with_input))
    registry.add(_toy(name="test.toy2", writes={}, build=input_only))
    spec = _spec().replace("width", StageRef("width", "test.toy"))
    p = build(spec)
    assert set(p.inputs) == {"regen_mask", "difficulty"}
    twice = spec.insert_after("width", StageRef("width", "test.toy2", id="w2"))
    with pytest.raises(PlanError, match="input 'difficulty' is declared twice"):
        build(twice)
    from track_gen.plugin import stage as stage_mod

    monkeypatch.setattr(stage_mod, "_salt_of", lambda stage_id, stream: 0x1234)
    registry.add(_toy(name="test.toy3", writes={},
                      build=lambda self, b, p: (b.salt("noise"), lambda ctx: None)[1]))
    with pytest.raises(PlanError, match=r"salt stream 'noise' collides with stage "
                                        r"'width' stream 'noise' \(0x00001234\)"):
        build(spec.insert_after("width", StageRef("width", "test.toy3", id="w3")))


def test_stage_builder_salt_is_keyed_by_the_stage_id():
    seen = {}

    def record(self, b, p):
        seen[b.stage_id] = b.salt("noise")
        w = b.write("half_width", wp.float32, per="cl")
        return lambda ctx: ctx.fill(w, 0.1)

    registry.add(_toy(build=record))
    build(_spec().replace("width", StageRef("width", "test.toy")))
    from track_gen.plugin.stage import _salt_of
    assert seen == {"width": _salt_of("width", "noise")}


# ---------------------------------------------------------------------------------------
# Versioning, exports and the TrackGenerator view
# ---------------------------------------------------------------------------------------

def test_an_elevated_lift_takes_any_elevation():
    spec = _spec().replace("elevation", StageRef("elevation", "uniform",
                                                 params={"z_max": 0.5})) \
        .replace("lift", StageRef("lift", "elevated"))
    p = build(spec, seeds=3)
    z = p.result.center.numpy().reshape(E, -1, 3)[..., 2]
    zc = p._channel("cl.z").numpy().reshape(E, -1)
    count = p.result.count.numpy()
    for e in range(E):
        np.testing.assert_array_equal(z[e, :count[e]], zc[e, :count[e]])
    assert np.nanmax(np.abs(z)) > 0.0


def test_channel_per_is_checked():
    from track_gen.stages.channels import Channel

    with pytest.raises(ValueError, match=r"Channel.per must be one of \('env', 'raw', "
                                         r"'cl'\), got 'gates'"):
        Channel(per="gates", dtype=wp.float32, interp=None, doc="")


def test_channel_versions_with_relax():
    p = build(_spec(), seeds=3)
    assert p._writers("cl.xy") == ("resample", "relax", "uniform")
    v = [p._channel("cl.xy", after=sid) for sid in ("resample", "relax", "uniform")]
    assert len({id(a) for a in v}) == 3                    # three buffers (PR9 topology)
    assert p._channel("cl.xy", after="width") is v[2] and p._channel("cl.xy") is v[2]
    for ch, field in (("track.outer", "outer"), ("cl.arclen2", "arclen"),
                      ("cl.length2", "length"), ("cl.winding", "winding"),
                      ("track.valid", "valid"), ("track.count", "count"),
                      ("cl.truncated", "truncated")):
        assert p._channel(ch) is getattr(p.result, field), ch
    with pytest.raises(KeyError):
        p._channel("cl.xy", after="generator")


def test_channel_versions_without_relax():
    p = build(_spec(relax_enable=False), seeds=3)
    assert p._writers("cl.xy") == ("resample", "uniform")
    assert p._channel("cl.xy", after="relax") is p._channel("cl.xy", after="resample")
    assert "relax" not in p._scratch or p._scratch["relax"] == {}


def test_the_elevated_lift_writes_its_own_arc_tables():
    p = build(_spec(z_profile="noise", z_max=0.4, z_noise_amplitude=0.3), seeds=3)
    assert p._writers("cl.arclen2") == ("frames",) and p._writers("track.arclen") == ("lift",)
    assert p._writers("cl.length2") == ("frames",) and p._writers("track.length") == ("lift",)
    plan_view = p._channel("cl.arclen2")
    assert plan_view is not p.result.arclen and p._channel("track.arclen") is p.result.arclen
    assert p._channel("cl.length2") is not p.result.length
    assert p._channel("track.length") is p.result.length
    count = p.result.count.numpy()
    a2, a3 = plan_view.numpy().reshape(E, -1), p.result.arclen.numpy().reshape(E, -1)
    for e in range(E):                                     # 3D >= plan-view
        assert (a3[e, 1:count[e]] >= a2[e, 1:count[e]] - 1e-5).all()


def test_track_generator_scratch_is_a_view_of_the_program():
    gen = TrackGenerator(_cfg(), PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    p, s = gen._program, gen._scratch
    assert isinstance(s, warp_pipeline._Scratch)
    assert s.generator_spec is gen._generator_spec
    assert s.gen_centerline is p._channel("raw.xy") and s.gen_valid is p._channel("gen.valid")
    assert s.cs_center is p._channel("cl.xy", after="resample")
    assert s.count is p._channel("cl.count")
    assert s.cs_seg is p._scratch["resample"]["seg"] and s.cs_s is p._scratch["resample"]["s"]
    assert s.cs_seg.shape == (E * 384,) and s.cs_s.shape == (E * 385,)
    r = s.relax
    assert r.relaxed is p._channel("cl.xy", after="relax")
    assert r.band is p._scratch["relax"]["band"] and r.L0 is p._scratch["relax"]["L0"]
    assert r.xpbd_db is p._scratch["relax"]["xy.tmp0"]
    assert r.cheb_prev is p._scratch["relax"]["xy.tmp1"]
    assert r.sep_cache_idx.shape == (E * 384 * 16,)
    i = s.inflate
    assert i.ctr2 is p._channel("cl.xy", after="uniform")
    assert i.w is p._channel("cl.half_width") and i.z is p._channel("cl.z")
    assert i.out2 is p._channel("band.outer2") and i.nrm2 is p._channel("cl.normal2")
    assert i.area_a is p._scratch["offset"]["area_a"] and i.knot_cum is None
    assert gen._track is p.result and gen._seed_buf is p.seed and gen._runner is p.runner
    assert gen._inputs is p.inputs and gen._regen_mask is p.inputs["regen_mask"]
    # The PR9 composition still runs on the view and writes the same Track.
    want = _snap(gen._track)
    warp_pipeline._run_pipeline(gen._config, gen._seed_buf, out=gen._track, scratch=s,
                                generator_spec=gen._generator_spec,
                                ctx=LaunchContext.for_call(DEV))
    assert _snap(gen._track) == want


def test_scratch_view_without_relax_and_with_knots():
    gen = TrackGenerator(_cfg(relax_enable=False, z_profile="uniform", z_max=0.2),
                         PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    s, p = gen._scratch, gen._program
    assert s.relax is None and s.inflate.ctr2 is p._channel("cl.xy", after="uniform")
    assert s.inflate.knot_cum.shape == (E * 10,)
    assert s.inflate.knot_count is p._scratch["elevation"]["knot_count"]


def test_track_generator_params_stay_the_generator_only():
    gen = TrackGenerator(_cfg(), PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    assert list(gen.params) == ["generator"]
    assert list(gen._program.params) == list(DEFAULT_IDS)
    assert gen._program.params["relax"].params == sp.XpbdParams.from_legacy(gen._config)
    assert gen._program.params["generator"].params == gen.params["generator"].params


# ---------------------------------------------------------------------------------------
# The Program
# ---------------------------------------------------------------------------------------

def test_program_enqueue_allocates_nothing_and_returns_the_result():
    p = build(_spec(z_profile="random_walk", z_max=0.2, z_max_step=0.3), seeds=3)
    ctx = LaunchContext.for_call(DEV)
    m = wp.array([1, 0], dtype=wp.int32, device=DEV)
    with no_allocation():
        assert p.enqueue(ctx) is p.result
        assert p.enqueue(ctx, advance=True, mask=m) is p.result
        p.run()
    assert p.result.generation.numpy().tolist() == [1, 0]


def test_program_state_and_call_errors():
    p = build(_spec(), seeds=3)
    assert p.layout == Layout(E, 384, "cpu") and p.device.alias == "cpu"
    assert p.capturable and p.capture_class == CaptureClass.STATIC and not p.legacy_stream
    assert p.graph is None and p.seed.shape == (E,) and p.seeds.num_envs == E
    assert repr(p).startswith("Program(stages=['generator', 'resample'")
    with pytest.raises(ValueError, match="requires advance=True"):
        p.run(mask=p.inputs["regen_mask"])
    with pytest.raises(TypeError, match=r"created by track_gen.pipeline.build\(\)"):
        Program(spec=None, layout=None, binding=None, result=None, seeds=None, seed=None,
                inputs={}, params=None, pairs=(), enqueues=(), versions={}, scratch={},
                stage_ids=(), kinds={}, gen_scratch=None)


def test_the_resolved_spec_holds_params_instances():
    spec = _spec().replace("relax", StageRef("relax", "xpbd", params={"relax_iters": 7}))
    p = build(spec, seeds=3)
    assert p.spec.stages[2].params == sp.XpbdParams(relax_iters=7)
    assert p.spec.stages[3].params == NoParams()
    assert p.spec.ids == DEFAULT_IDS


def test_a_stage_generator_runs_through_the_program():
    from tests._plugin_toys import ToyWobble, register_stage_toys

    register_stage_toys(registry)
    spec = PipelineSpec(BatchSpec(E, DEV, 384), (
        StageRef("generator", "toy.wobble", params={"lobes": 6}),) + _spec().stages[1:])
    p = build(spec, seeds=3)
    assert p._channel("raw.xy").shape == (E * ToyWobble.RAW,)
    assert list(p.params["generator"]) == ["amp"]
    gen = TrackGenerator(_cfg(generator="toy.wobble", generator_params={"lobes": 6}),
                         PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    assert _snap(p.result) == _snap(gen._track)


def test_stage_builder_temp_like_in_generator_mode():
    seen = {}

    class Temp(Stage):
        kind, name, output = "generator", "acme.temp", "centerline"
        writes = {"xy": "raw.xy"}

        def build(self, b, p):
            xy = b.write("xy", wp.vec2f, per="raw")
            seen["tmp"] = b.temp_like("xy")
            with pytest.raises(ValueError, match=r"declared read ports: \[\]"):
                b.read("xy")
            src = wp.array(np.tile(np.stack([0.6 * np.cos(np.linspace(0, 6.28, b.raw)),
                                             0.6 * np.sin(np.linspace(0, 6.28, b.raw))],
                                            1).astype(np.float32), (b.E, 1)),
                           dtype=wp.vec2f, device=b.device)
            return lambda ctx: ctx.copy(xy, src)

    registry.add(Temp)
    gen = TrackGenerator(_cfg(generator="acme.temp"),
                         PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    assert seen["tmp"].shape == (E * 256,) and seen["tmp"].dtype is wp.vec2f
    assert seen["tmp"] is gen._scratch.gen.scratch["scratch:generator/xy.tmp0"]
    assert isinstance(gen._scratch.gen.builder, StageBuilder)


def test_import_track_gen_does_not_import_the_pipeline():
    import subprocess
    import sys

    code = ("import sys, track_gen; "
            "assert 'track_gen.pipeline' not in sys.modules; "
            "assert 'track_gen.stages' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------------------------
# Plugin stages: rewiring, plugin channels, capture classes (roadmap PR10b)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _fill_gain_k(w: wp.array(dtype=wp.float32)):
    w[wp.tid()] = 1.0


def _stage(kind, name, reads, writes, build_fn=None, **attrs):
    def build_(self, b, p):
        if build_fn is not None:
            return build_fn(self, b, p)
        for port in writes:
            ch = writes[port]
            if ch.startswith("cl.acme."):
                b.write(port, wp.float32, per="cl")
            elif ch == "valid.fail":
                b.fail_bit()
                b.write(port, wp.int32)
            else:
                dtype, per = b._plan.channel(ch)
                b.write(port, dtype, per=per)
        return lambda ctx: None

    return type("Toy", (Stage,), dict(kind=kind, name=name, reads=reads, writes=writes,
                                      build=build_, **attrs))


def test_the_rewrite_rule_holds_after_rewiring():
    registry.add(_stage("relax", "test.r1", {"a": "cl.xy"}, {"b": "cl.acme.out"}))
    with pytest.raises(PlanError, match=r"stage 'relax' \(relax.test.r1\): channel 'cl.xy' "
                                        r"is read through port 'a' and written through port "
                                        r"'b'; an in-place rewrite reads and writes it "
                                        r"through one port"):
        build(_spec().replace("relax", StageRef("relax", "test.r1", ports={"b": "cl.xy"})))


def test_the_kind_write_rule_holds_after_rewiring():
    registry.add(_stage("relax", "test.r2", {"xy": "cl.xy"}, {"xy": "cl.xy"}))
    with pytest.raises(PlanError, match=r"stage 'relax' \(relax.test.r2\) \(kind 'relax'\) "
                                        r"writes 'band.outer2' \(port 'xy'\): a relax stage "
                                        r"may write \['cl.xy'\]"):
        build(_spec().replace("relax", StageRef("relax", "test.r2",
                                                ports={"xy": "band.outer2"})))
    # rewiring to a plugin channel is allowed (its writer fixes the dtype)
    registry.add(_stage("width", "test.w", {}, {"half_width": "cl.half_width",
                                                "aux": "cl.acme.aux"}))
    p = build(_spec().replace("width", StageRef("width", "test.w",
                                                ports={"aux": "cl.acme.gain"})))
    assert p.channels["cl.acme.gain"] == ("width",)
    assert p.tap("cl.acme.gain").dtype is wp.float32


def test_plugin_channels():
    def writer(self, b, p):
        b.write("half_width", wp.float32, per="cl")
        g = b.write("gain", wp.float32, per="cl")
        return lambda ctx: ctx.launch(_fill_gain_k, g.shape[0], [g])

    registry.add(_stage("width", "test.gain_w", {}, {"half_width": "cl.half_width",
                                                     "gain": "cl.acme.gain"}, writer))
    registry.add(_stage("validity", "test.reader", {"gain": "cl.acme.gain",
                                                    "fail": "valid.fail"},
                        {"fail": "valid.fail"}))
    spec = _spec().replace("width", StageRef("width", "test.gain_w")).insert_after(
        "validity", StageRef("validity", "test.reader", id="reader"))
    p = build(spec, seeds=3)
    assert p.channels["cl.acme.gain"] == ("width",)
    gain = p.tap("cl.acme.gain")
    assert gain.shape == (E * 384,) and (gain.numpy() == 1.0).all()
    # read without an earlier writer
    with pytest.raises(PlanError, match=r"reads 'cl.acme.gain' \(port 'gain'\), but no "
                                        r"earlier stage writes it"):
        build(_spec().insert_after("validity", StageRef("validity", "test.reader",
                                                        id="reader")))

    # a later writer with another dtype
    def int_writer(self, b, p):
        b.write("gain", wp.int32, per="cl")
        return lambda ctx: None

    registry.add(_stage("width", "test.int_w", {"gain": "cl.acme.gain"},
                        {"gain": "cl.acme.gain"}, int_writer))
    with pytest.raises(TypeError, match=r"channel 'cl.acme.gain' has dtype float32, got "
                                        r"int32"):
        build(_spec().replace("width", StageRef("width", "test.gain_w")).insert_after(
            "width", StageRef("width", "test.int_w", id="w2")))

    def raw_writer(self, b, p):
        b.write("gain", wp.float32, per="raw")

    registry.add(_stage("width", "test.raw_w", {}, {"gain": "cl.acme.raw"}, raw_writer))
    with pytest.raises(ValueError, match=r"plugin channel 'cl.acme.raw': per must be one of "
                                         r"\['env', 'cl'\], got 'raw'"):
        build(_spec().insert_after("width", StageRef("width", "test.raw_w", id="w2")))

    def early_temp(self, b, p):
        b.temp_like("gain")

    registry.add(_stage("width", "test.temp_w", {}, {"gain": "cl.acme.tmp"}, early_temp))
    with pytest.raises(ValueError, match=r"temp_like\('gain'\): plugin channel "
                                         r"'cl.acme.tmp' has no dtype yet"):
        build(_spec().insert_after("width", StageRef("width", "test.temp_w", id="w2")))

    # dtype / per omitted: taken from the channel once it has one
    def untyped(self, b, p):
        b.write("gain")
        return lambda ctx: None

    registry.add(_stage("width", "test.untyped_w", {"gain": "cl.acme.gain"},
                        {"gain": "cl.acme.gain"}, untyped))
    p = build(_spec().replace("width", StageRef("width", "test.gain_w")).insert_after(
        "width", StageRef("width", "test.untyped_w", id="w2")))
    assert p.tap("cl.acme.gain").dtype is wp.float32
    assert p.tap("cl.acme.gain").shape == (E * 384,)
    registry.add(_stage("width", "test.first_untyped", {}, {"gain": "cl.acme.new"}, untyped))
    with pytest.raises(TypeError, match=r"plugin channel 'cl.acme.new': its first writer "
                                        r"passes dtype="):
        build(_spec().insert_after("width", StageRef("width", "test.first_untyped",
                                                     id="w2")))


def test_a_host_stage_makes_the_program_host():
    registry.add(_stage("relax", "test.host", {"xy": "cl.xy"}, {"xy": "cl.xy"},
                        capture=CaptureClass.HOST))
    spec = _spec().replace("relax", StageRef("relax", "test.host"))
    p = build(spec, seeds=3)
    assert not p.capturable and p.capture_class is CaptureClass.HOST
    gen = TrackGenerator(spec, PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    assert gen._capturable is False and gen.program.capturable is False
    assert registry.describe("relax", "test.host").capturable is False


def test_a_bounded_capture_class_is_refused_at_plan_time(no_alloc):
    Bounded = _stage("relax", "test.bounded", {"xy": "cl.xy"}, {"xy": "cl.xy"})
    Bounded.capture_class = classmethod(lambda cls, params: CaptureClass.BOUNDED)
    registry.add(Bounded)
    with pytest.raises(PlanError, match=r"stage 'relax' \(relax.test.bounded\): "
                                        r"Toy.capture_class\(\) returned BOUNDED: arrives "
                                        r"with roadmap PR12"):
        build(_spec().replace("relax", StageRef("relax", "test.bounded")))


def test_tap_and_channels():
    p = build(_spec(), seeds=3)
    assert p.tap("cl.xy") is p._channel("cl.xy")
    assert p.tap("cl.xy", after="resample") is not p.tap("cl.xy", after="relax")
    assert p.tap("seed") is p.seed
    assert p.channels["seed"] == ("<prologue>",)
    assert p.channels["cl.xy"] == ("resample", "relax", "uniform")
    assert p.channels["valid.fail"] == ("validity",)
    with pytest.raises(KeyError, match=r"no stage writes channel 'cl.bogus'; channels: "
                                       r"\[.*'cl.xy'"):
        p.tap("cl.bogus")
    with pytest.raises(KeyError, match=r"no stage 'nope'; stages: \['generator'"):
        p.tap("cl.xy", after="nope")
    with pytest.raises(KeyError, match=r"channel 'cl.xy' is not written by 'generator' or "
                                       r"earlier"):
        p.tap("cl.xy", after="generator")
    with pytest.raises(TypeError):
        p.channels["x"] = ("y",)                                # read-only


def test_open_kinds_resolve_through_the_given_registry():
    from track_gen.plugin import Registry

    other = Registry()
    other.add(_stage("relax", "test.elsewhere", {"xy": "cl.xy"}, {"xy": "cl.xy"}))
    spec = _spec().replace("relax", StageRef("relax", "test.elsewhere"))
    p = build(spec, registry=other, seeds=3)
    assert p._kinds["relax"] == ("relax", "test.elsewhere")
    with pytest.raises(PlanError, match="unknown relax 'test.elsewhere'"):
        build(spec)
