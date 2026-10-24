"""``track_gen.plugin.Stage`` generators: registration rules, describe(), the
StageBuilder, dynamic params and inputs through ``gen.params`` / ``gen.inputs``, the
masked latch, salts and capture classes (roadmap PR8). CPU only; the CUDA twin (caller
graphs, writes between replays) lives in test_params_cuda.py.
"""
from __future__ import annotations

import dataclasses
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import warp as wp

from tests._plugin_toys import (
    WOBBLE_RAW,
    FakeDist,
    FakeEntryPoint,
    ToyWobble,
    ToyWobbleParams,
    install_entry_points,
    register_stage_toys,
)
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen.plugin import (
    POLICY_ENV,
    CaptureClass,
    Layout,
    NoParams,
    ParamTable,
    Params,
    Stage,
    StageBuilder,
    StageParams,
    registry,
    static,
)

E = 4
DEV = "cpu"


@pytest.fixture
def toys():
    register_stage_toys(registry)
    yield


def _rng(seed=11, n=E):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _gen(name="toy.wobble", seed=11, n=E, **kw):
    return TrackGenerator(TrackGenConfig(generator=name, device=DEV, num_envs=n, **kw),
                          _rng(seed, n))


def _snap(track) -> dict:
    return {f: np.array(getattr(track, f).numpy(), copy=True)
            for f in ("center", "outer", "inner", "tangent", "arclen", "length", "valid",
                      "count")}


def _same(a: dict, b: dict, envs=None) -> bool:
    for k in a:
        x, y = a[k], b[k]
        if envs is not None:
            x = x.reshape(E, -1)[envs]
            y = y.reshape(E, -1)[envs]
        if x.tobytes() != y.tobytes():
            return False
    return True


def _env_rows(snap: dict, e: int) -> dict:
    return {k: v.reshape(E, -1)[e] for k, v in snap.items()}


# ---------------------------------------------------------------------------------------
# Registration rules
# ---------------------------------------------------------------------------------------

def _stage(**attrs):
    def build(self, b, p):
        xy = b.write("xy", wp.vec2f, per="raw")
        return lambda ctx: None

    ns = dict(kind="generator", name="acme.s", output="centerline",
              writes={"xy": "raw.xy"}, build=build)
    ns.update(attrs)
    return type("S", (Stage,), ns)


def test_register_a_stage_class():
    S = _stage()
    registry.add(S)
    assert registry.get("generator", "acme.s") is S
    assert "acme.s" in registry.available("generator")


def test_registration_rejects_an_instance_and_an_abstract_class():
    with pytest.raises(TypeError, match="register the Stage class, not an instance"):
        registry.add(_stage()())

    class Abstract(Stage):
        kind = "generator"
        name = "acme.abstract"
        output = "centerline"

    with pytest.raises(TypeError, match="abstract"):
        registry.add(Abstract)


def test_registration_checks_a_gate_generator_stage():
    # PR11: gate generators are Stages (kind="gate_generator", output="anchors").
    with pytest.raises(ValueError, match="a gate generator Stage declares "
                                         "output='anchors', got 'centerline'"):
        registry.add(_stage(kind="gate_generator"))
    with pytest.raises(ValueError, match="declares kind 'generator', not 'relax'"):
        registry.add(_stage(), kind="relax")


def test_registration_rejects_a_wrong_output():
    with pytest.raises(ValueError, match="output='anchors' is a gate generator: declare "
                                         "kind='gate_generator'"):
        registry.add(_stage(output="anchors"))
    with pytest.raises(ValueError, match="must declare output='centerline'"):
        registry.add(_stage(output=None))


@pytest.mark.parametrize("attrs,match", [
    (dict(writes={"xy": "raw.z"}), "must map a port to 'raw.xy'"),
    (dict(writes={"xy": "raw.xy", "k": "cl.kappa"}), "are not generator channels"),
    (dict(writes={"ok": "gen.valid"}), "must map a port to 'raw.xy'"),
    (dict(writes={"a": "raw.xy", "b": "raw.xy"}), "two ports to one channel"),
    (dict(reads={"x": "raw.xy"}), "reads must be empty"),
    # roadmap PR13a: a centerline generator provides exactly the catalogue attributes
    # it writes (raw.z -> 'z', raw.half_width -> 'half_width').
    (dict(provides=frozenset({"z"})), r"provides \['z'\] must name exactly the catalogue "
                                      r"attributes it writes \(\[\]"),
    (dict(writes={"xy": "raw.xy", "z": "raw.z"}), r"provides \[\] must name exactly"),
    (dict(topology="open"), "topology must be 'closed' or 'any'"),
    (dict(api_version=2), "api_version=2"),
    (dict(capture=1), "must be a CaptureClass"),
])
def test_registration_rejects_a_bad_declaration(attrs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        registry.add(_stage(**attrs))
    assert "acme.s" not in registry.available("generator")


def test_registration_checks_the_params_class():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(Params):
        x: int

    with pytest.raises(TypeError, match="needs a default"):
        registry.add(_stage(Params=P))


def test_the_track_pipeline_kinds_are_open_and_the_others_are_not():
    S = _stage(kind="relax", output=None, writes={})            # roadmap PR10b
    registry.add(S)
    assert registry.get("relax", "acme.s") is S
    R = _stage(kind="resample", output=None, writes={})
    # PR11 opened gate_generator, order and stations.
    with pytest.raises(ValueError, match=r"Stage S: Stage kind 'resample' is not open to "
                                         r"plugins; open kinds: \['elevation', "
                                         r"'gate_generator', 'generator', 'order', "
                                         r"'relax', 'stations', 'validity', 'width'\]"):
        registry.add(R)
    assert "acme.s" not in registry.available("resample")


def _pipeline_stage(**attrs):
    def build(self, b, p):
        return lambda ctx: None

    ns = dict(kind="relax", name="acme.r", reads={"xy": "cl.xy"}, writes={"xy": "cl.xy"},
              build=build)
    ns.update(attrs)
    return type("R", (Stage,), ns)


@pytest.mark.parametrize("attrs,exc,match", [
    (dict(output="centerline"), ValueError,
     r"Stage R: output is for generators \(kind 'relax' declares none\)"),
    (dict(writes={"xy": "cl.bogus"}), ValueError,
     r"Stage R: unknown channels \['cl.bogus'\]; use catalogue channels or namespaced "
     r"plugin channels 'cl.<package>.<name>'"),
    (dict(writes={"xy": "cl.xy", "k": "cl.kappa"}), ValueError,
     r"Stage R \(kind 'relax'\) writes 'cl.kappa' \(port 'k'\): a relax stage may write "
     r"\['cl.xy'\] and plugin channels 'cl.<package>.<name>'"),
    (dict(kind="width", reads={}, writes={"w": "cl.z"}), ValueError,
     r"\(kind 'width'\) writes 'cl.z' \(port 'w'\): a width stage may write "
     r"\['cl.half_width'\]"),
    (dict(kind="elevation", reads={}, writes={"s": "seed"}), ValueError,
     r"\(kind 'elevation'\) writes 'seed'"),
    (dict(reads={"a": "cl.xy"}, writes={"b": "cl.xy"}), ValueError,
     r"Stage R: channel 'cl.xy' is read through port 'a' and written through port 'b'; "
     r"an in-place rewrite reads and writes it through one port"),
    (dict(reads={"xy": "cl.count"}, writes={"xy": "cl.xy"}), ValueError,
     r"port 'xy' reads 'cl.count' and writes 'cl.xy'"),
    (dict(writes={"a": "cl.xy", "b": "cl.xy"}, reads={}), ValueError,
     r"writes 'cl.xy' through two ports"),
    (dict(kind="validity", reads={}, writes={"x": "cl.acme.score"}), ValueError,
     r"Stage R \(kind 'validity'\) must write 'valid.fail'"),
    (dict(writes=[("xy", "cl.xy")]), TypeError, "writes must be a mapping"),
])
def test_pipeline_stage_registration_rules(attrs, exc, match):
    with pytest.raises(exc, match=match):
        registry.add(_pipeline_stage(**attrs))
    assert "acme.r" not in registry.available(attrs.get("kind", "relax"))


def test_namespaced_plugin_channels_are_accepted():
    S = _pipeline_stage(writes={"xy": "cl.xy", "aux": "cl.acme.aux"})
    registry.add(S)
    V = _pipeline_stage(kind="validity", name="acme.v", reads={"aux": "cl.acme.aux"},
                        writes={"fail": "valid.fail"})
    registry.add(V)
    assert dict(registry.describe("validity", "acme.v").reads) == {"aux": "cl.acme.aux"}
    for bad in ("cl.acme", "acme.aux", "cl.1x.y", "cl.acme.", "band.acme.aux"):
        with pytest.raises(ValueError, match="unknown channels"):
            registry.add(_pipeline_stage(name="acme.bad", writes={"xy": "cl.xy",
                                                                   "aux": bad}))
    with pytest.raises(TypeError, match=r"a width spec must be a Stage subclass "
                                        r"\(registry.add\(MyStage\)\)"):
        registry.add(SimpleNamespace(name="acme.w"), kind="width")


def test_a_bounded_stage_registers():
    # Roadmap PR12: BOUNDED (a ctx.bounded_loop with active=) is accepted for every kind.
    S = _stage(capture=CaptureClass.BOUNDED)
    registry.add(S)
    info = registry.describe("generator", "acme.s")
    assert info.capture is CaptureClass.BOUNDED and info.capturable
    R = _pipeline_stage(capture=CaptureClass.BOUNDED)
    registry.add(R)
    assert registry.describe("relax", "acme.r").capture is CaptureClass.BOUNDED


# ---------------------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------------------

def test_describe_a_stage_generator(toys):
    info = registry.describe("generator", "toy.wobble")
    assert info.capturable is True and info.accepts_ctx is True
    assert info.origin == "tests._plugin_toys" and not info.builtin
    assert info.params_schema["title"] == "ToyWobbleParams"
    assert info.params_schema["properties"]["amp"]["x-dynamic"] is True
    assert registry.describe("generator", "toy.wobble_host").capturable is False
    assert info.capture is CaptureClass.STATIC and info.output == "centerline"
    assert dict(info.writes) == {"xy": "raw.xy", "valid": "gen.valid"}
    assert info.provides == frozenset()
    assert registry.describe("generator", "toy.wobble_host").capture is CaptureClass.HOST
    S = _stage()
    registry.add(S)
    assert registry.describe("generator", "acme.s").params_schema == {}   # NoParams
    # A legacy GeneratorSpec (the toy with params=; "bezier" is a Stage since roadmap
    # PR13c, described like any other).
    legacy = registry.describe("generator", "toy.lobed")
    assert legacy.capture is None and legacy.output is None
    assert dict(legacy.writes) == {} and legacy.provides == frozenset()
    bezier = registry.describe("generator", "bezier")
    assert bezier.capture is CaptureClass.STATIC and bezier.output == "centerline"
    assert dict(bezier.writes) == {"xy": "raw.xy", "valid": "gen.valid"}
    assert bezier.builtin and bezier.params_schema["properties"]["rad"]["x-dynamic"]


# ---------------------------------------------------------------------------------------
# End to end on CPU
# ---------------------------------------------------------------------------------------

def test_stage_generator_runs_end_to_end(toys):
    gen = _gen()
    track = gen.generate()
    assert track.valid.numpy().tolist() == [1] * E
    assert gen._scratch.gen_centerline.shape == (E * WOBBLE_RAW,)
    count = track.count.numpy()
    center = track.center.numpy().reshape(E, -1, 3)
    for e in range(E):
        radius = np.linalg.norm(center[e, :count[e], :2], axis=1)
        assert 0.4 < radius.min() and radius.max() < 0.8
    assert len(set(np.round(track.length.numpy(), 5))) > 1          # salted per-env RNG
    sp = gen.params["generator"]
    assert isinstance(gen.params, ParamTable) and isinstance(sp, StageParams)
    assert list(gen.params) == ["generator"] and list(sp) == ["amp"] and len(sp) == 1
    assert sp.stage_id == "generator" and sp.params == ToyWobbleParams()
    assert sp["amp"].shape == (E,) and sp["amp"].dtype is wp.float32
    np.testing.assert_allclose(sp["amp"].numpy(), 0.15)
    assert set(gen.inputs) == {"regen_mask"}
    assert gen._latch is not None


def test_dict_and_instance_params_are_equivalent(toys):
    a = _snap(_gen(generator_params={"lobes": 7, "amp": 0.1}).generate())
    gen = _gen(generator_params=ToyWobbleParams(lobes=7, amp=0.1))
    b = _snap(gen.generate())
    assert _same(a, b)
    assert not _same(a, _snap(_gen().generate()))
    assert gen.config.generator_params == ToyWobbleParams(lobes=7, amp=0.1)
    again = TrackGenerator(gen.config, _rng())
    assert _same(b, _snap(again.generate()))


def test_stage_params_are_validated_at_construction(toys):
    with pytest.raises(ValueError, match=r"ToyWobbleParams.amp must be in \[0.0, 0.4\]"):
        _gen(generator_params={"amp": 0.5})
    with pytest.raises(ValueError, match="unknown generator_params for generator "
                                         "'toy.wobble': \\['lobe'\\]"):
        _gen(generator_params={"lobe": 3})
    with pytest.raises(TypeError, match="must be a ToyWobbleParams"):
        _gen(generator_params=NoParams())


# ---------------------------------------------------------------------------------------
# Dynamic params
# ---------------------------------------------------------------------------------------

def test_set_scalar_takes_effect_at_the_next_generation(toys):
    gen = _gen()
    before = _snap(gen.generate())
    gen.params["generator"].set("amp", 0.3)
    np.testing.assert_allclose(gen.params["generator"]["amp"].numpy(), 0.3)
    after = _snap(gen.generate())
    assert not _same(before, after)
    fresh = _snap(_gen(generator_params={"amp": 0.3}).generate())
    assert _same(after, fresh)
    # ParamTable.set is the same call.
    gen.params.set("generator", "amp", 0.15)
    assert _same(before, _snap(gen.generate()))


def test_set_per_env_array(toys):
    amps = np.array([0.0, 0.1, 0.2, 0.3], np.float32)
    gen = _gen()
    gen.params["generator"].set("amp", wp.array(amps, dtype=wp.float32, device=DEV))
    got = _snap(gen.generate())
    for e, a in enumerate(amps):
        ref = _snap(_gen(generator_params={"amp": float(a)}).generate())
        assert _same(_env_rows(got, e), _env_rows(ref, e)), e


def test_writing_the_live_array_directly_takes_effect(toys):
    gen = _gen()
    gen.params["generator"]["amp"].assign(np.full(E, 0.3, np.float32))
    assert _same(_snap(gen.generate()),
                 _snap(_gen(generator_params={"amp": 0.3}).generate()))


def test_set_rejections(toys):
    sp = _gen().params["generator"]
    with pytest.raises(KeyError, match="static"):
        sp.set("lobes", 3)
    with pytest.raises(KeyError, match="static"):
        sp["lobes"]
    with pytest.raises(KeyError, match="dynamic fields: \\['amp'\\]"):
        sp.set("nope", 1.0)
    with pytest.raises(ValueError, match=r"must be in \[0.0, 0.4\], got 0.5"):
        sp.set("amp", 0.5)
    with pytest.raises(ValueError, match="expected shape"):
        sp.set("amp", wp.zeros(E + 1, dtype=wp.float32, device=DEV))
    with pytest.raises(ValueError, match="expected a float32 array"):
        sp.set("amp", wp.zeros(E, dtype=wp.int32, device=DEV))
    with pytest.raises(ValueError, match="expected a scalar or an \\[E\\] wp.array"):
        sp.set("amp", np.zeros(E, np.float32))
    with pytest.raises(ValueError, match="expected a scalar"):
        sp.set("amp", "0.1")
    with pytest.raises(ValueError, match="must be a real number"):
        sp.set("amp", True)
    np.testing.assert_allclose(sp["amp"].numpy(), 0.15)        # nothing was written


def test_set_rejects_an_array_on_another_device(toys):
    sp = _gen().params["generator"]
    other = wp.zeros(E, dtype=wp.float32, device=DEV)

    class OtherDevice:
        def __str__(self):
            return "cuda:7"

    other.device = OtherDevice()                   # an instance attribute of wp.array
    with pytest.raises(ValueError, match="is on cuda:7"):
        sp.set("amp", other)
    np.testing.assert_allclose(sp["amp"].numpy(), 0.15)


def test_masked_latch_keeps_unmasked_envs_bitwise(toys):
    gen = _gen()
    base = _snap(gen.generate())
    gen.params["generator"].set("amp", 0.3)
    zero = wp.zeros(E, dtype=wp.int32, device=DEV)
    assert _same(base, _snap(gen.generate(advance=True, mask=zero)))      # exact no-op
    mask = wp.array(np.array([1, 0, 0, 0], np.int32), dtype=wp.int32, device=DEV)
    got = _snap(gen.generate(advance=True, mask=mask))
    assert _same(base, got, envs=[1, 2, 3])
    assert not _same(base, got, envs=[0])
    # env 0 = epoch 1 with amp 0.3; the latch only reached it.
    np.testing.assert_allclose(gen.params["generator"]["amp"].numpy(), 0.3)
    # A full generation then latches every env.
    gen.seeds.reset_epoch()
    full = _snap(gen.generate())
    assert _same(full, _snap(_gen(generator_params={"amp": 0.3}).generate()))


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------

def test_stage_inputs_appear_in_gen_inputs_and_take_effect(toys):
    gen = _gen("toy.wobble_input")
    assert set(gen.inputs) == {"regen_mask", "difficulty"}
    difficulty = gen.inputs["difficulty"]
    assert difficulty.dtype is wp.float32 and difficulty.shape == (E,)
    np.testing.assert_allclose(difficulty.numpy(), 0.5)
    half = _snap(gen.generate())
    assert _same(half, _snap(_gen(generator_params={"amp": 0.075}).generate()))
    difficulty.assign(np.ones(E, np.float32))
    assert _same(_snap(gen.generate()), _snap(_gen().generate()))
    assert gen.inputs["difficulty"].ptr == difficulty.ptr


# ---------------------------------------------------------------------------------------
# StageBuilder
# ---------------------------------------------------------------------------------------

def _fnv1a(data: bytes) -> int:
    h = 0x811C9DC5
    for byte in data:
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h


def _builder_stage(name, body, *, writes=None, params=NoParams):
    """A generator Stage whose build runs ``body(b, p)`` and then writes raw.xy."""
    def build(self, b, p):
        out = body(b, p)
        if "xy" in type(self).writes and "xy" not in b._written:
            xy = b.write("xy", wp.vec2f, per="raw")

            def enqueue(ctx):
                ctx.fill(xy, wp.vec2f(0.0, 0.0))
            return out if callable(out) else enqueue
        return out if callable(out) else (lambda ctx: None)

    S = type("BuilderStage", (Stage,), dict(
        kind="generator", name=name, output="centerline", Params=params,
        writes=writes if writes is not None else {"xy": "raw.xy", "ok": "gen.valid"},
        build=build))
    registry.add(S)
    return S


def test_salts_follow_the_frozen_formula():
    from track_gen.ops import hash32_host

    seen = {}

    def body(b, p):
        seen["a"] = b.salt("shape")
        seen["again"] = b.salt("shape")
        seen["b"] = b.salt("phase")
    _builder_stage("acme.salts", body)
    _gen("acme.salts", n=E)
    assert seen["a"] == seen["again"] == hash32_host(_fnv1a(b"generator/shape"))
    assert seen["b"] == hash32_host(_fnv1a(b"generator/phase")) != seen["a"]


def test_a_salt_collision_raises(monkeypatch):
    from track_gen.plugin import stage as stage_mod

    monkeypatch.setattr(stage_mod, "_fnv1a_32", lambda data: 42)

    def body(b, p):
        b.salt("one")
        b.salt("two")
    _builder_stage("acme.collide", body)
    with pytest.raises(ValueError, match="salt streams 'one' and 'two' collide"):
        _gen("acme.collide")


@pytest.mark.parametrize("misuse,exc,match", [
    (lambda b: b.write("nope", wp.vec2f, per="raw"), ValueError,
     r"declares no write port 'nope'; declared ports: \['ok', 'xy'\]"),
    (lambda b: b.write("xy", wp.vec3f, per="raw"), TypeError, "has dtype vec2f"),
    (lambda b: b.write("xy", wp.vec2f, per="env"), TypeError,
     "is per='raw', got per='env'"),
    (lambda b: (b.write("ok", wp.int32), b.write("ok", wp.int32)), ValueError,
     "already written"),
    (lambda b: b.write("xy", wp.vec2f, per="raw", interp="linear"), ValueError,
     r"interp= is for a generator's raw attributes .*raw.xy is resampled by the "
     r"pipeline itself"),
    (lambda b: b.param("amp"), KeyError, "no dynamic parameter 'amp'"),
    (lambda b: b.input("regen_mask", wp.int32), ValueError, "already exists"),
    (lambda b: b.input("x", wp.vec2f), TypeError, "wp.float32 or wp.int32"),
    (lambda b: (b.scratch("t", wp.float32, 4), b.scratch("t", wp.float32, 4)), ValueError,
     "already allocated"),
])
def test_builder_misuse_raises_at_construction(misuse, exc, match):
    _builder_stage("acme.misuse", lambda b, p: misuse(b))
    with pytest.raises(exc, match=match):
        _gen("acme.misuse")


def test_builder_write_takes_dtype_and_per_from_the_channel():
    seen = {}

    def body(b, p):
        seen["xy"], seen["ok"] = b.write("xy"), b.write("ok")

    _builder_stage("acme.inferred", body)
    _gen("acme.inferred")
    assert seen["xy"].dtype is wp.vec2f and seen["ok"].dtype is wp.int32
    assert seen["ok"].shape == (E,)


def test_builder_static_param_and_input_clash():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(Params):
        k: int = static(3)

    _builder_stage("acme.static", lambda b, p: b.param("k"), params=P)
    with pytest.raises(KeyError, match="static parameter: read it from params"):
        _gen("acme.static")

    def two_inputs(b, p):
        b.input("x", wp.float32)
        b.input("x", wp.float32)
    _builder_stage("acme.inputs", two_inputs)
    with pytest.raises(ValueError, match="input 'x' already exists"):
        _gen("acme.inputs")


def test_a_missing_raw_write_and_a_non_callable_build_are_rejected():
    class NoRaw(Stage):
        kind, name, output = "generator", "acme.noraw2", "centerline"
        writes = {"xy": "raw.xy"}

        def build(self, b, p):
            return lambda ctx: None

    registry.add(NoRaw)
    with pytest.raises(ValueError, match="must write 'raw.xy'"):
        _gen("acme.noraw2")

    class NotCallable(NoRaw):
        name = "acme.notcallable"

        def build(self, b, p):
            b.write("xy", wp.vec2f, per="raw")
            return 3

    registry.add(NotCallable)
    with pytest.raises(TypeError, match="must return an enqueue"):
        _gen("acme.notcallable")


def test_gen_valid_defaults_to_one_and_scratch_is_zeroed():
    seen = {}

    class Circle(Stage):
        kind, name, output = "generator", "acme.circle", "centerline"
        writes = {"xy": "raw.xy"}                       # no gen.valid

        def capacity(self, p, layout):
            seen["layout"] = layout
            return {}

        def build(self, b, p):
            xy = b.write("xy", wp.vec2f, per="raw")
            t = b.scratch("t", wp.float32, (b.E,))
            seen.update(scratch=t.numpy().copy(), raw=b.raw, scale=b.scale, E=b.E,
                        seed=b.seed, b=b)
            pts = np.stack([0.6 * np.cos(np.linspace(0, 2 * np.pi, b.raw, endpoint=False)),
                            0.6 * np.sin(np.linspace(0, 2 * np.pi, b.raw, endpoint=False))],
                           axis=1).astype(np.float32)
            src = wp.array(np.tile(pts, (b.E, 1)), dtype=wp.vec2f, device=b.device)

            def enqueue(ctx):
                ctx.copy(xy, src)
            return enqueue

    registry.add(Circle)
    gen = _gen("acme.circle", scale=1.0)
    assert gen.generate().valid.numpy().tolist() == [1] * E
    assert gen._scratch.gen_valid.numpy().tolist() == [1] * E
    assert (seen["scratch"] == 0).all() and seen["raw"] == 256 and seen["E"] == E
    assert seen["seed"] is gen._seed_buf
    assert seen["layout"] == Layout(E, 384, "cpu")
    assert gen._latch is None                           # no dynamic params, no inputs
    assert list(gen.params["generator"]) == []
    with pytest.raises(RuntimeError, match="after build"):
        seen["b"].scratch("late", wp.float32, 1)


def test_capacity_must_be_at_least_three():
    class Tiny(ToyWobble):
        name = "acme.tiny"

        def capacity(self, p, layout):
            return {"raw": 2}

    registry.add(Tiny)
    with pytest.raises(ValueError, match="capacity\\(\\)\\['raw'\\] must be an int >= 3"):
        _gen("acme.tiny")


def test_capacity_rejects_unknown_keys_and_takes_any_integral(toys):
    class Typo(ToyWobble):
        name = "acme.typo"

        def capacity(self, p, layout):
            return {"points": 128}

    registry.add(Typo)
    with pytest.raises(ValueError, match=r"unknown keys \['points'\]; a generator Stage "
                                         r"accepts \['raw'\]"):
        _gen("acme.typo")

    class NumpyRaw(ToyWobble):
        name = "acme.numpy_raw"

        def capacity(self, p, layout):
            return {"raw": np.int64(self.RAW)}

    registry.add(NumpyRaw)
    gen = _gen("acme.numpy_raw")
    assert gen._generator_spec.raw_points == ToyWobble.RAW
    assert _same(_snap(gen.generate()), _snap(_gen().generate()))


def test_build_capacity_and_capture_class_see_static_fields_only(toys):
    seen = {}

    class Reader(ToyWobble):
        name = "acme.reader"

        @classmethod
        def capture_class(cls, params):
            seen["capture"] = params
            return CaptureClass.STATIC

        def capacity(self, p, layout):
            seen["capacity"] = p
            return {"raw": self.RAW}

        def build(self, b, p):
            seen["build"] = p
            return super().build(b, p)

    registry.add(Reader)
    gen = _gen("acme.reader", generator_params={"lobes": 7, "amp": 0.3})
    for where in ("capture", "capacity", "build"):
        p = seen[where]
        assert isinstance(p, ToyWobbleParams) and p.lobes == 7, where
        with pytest.raises(AttributeError, match=r"ToyWobbleParams.amp is dynamic: .*"
                                                 r"b.param\('amp'\)"):
            p.amp                                          # noqa: B018
        assert repr(p) == "ToyWobbleParams(lobes=7, amp=<dynamic>)"
    assert gen.params["generator"].params == ToyWobbleParams(lobes=7, amp=0.3)

    class Baker(ToyWobble):
        name = "acme.baker"

        def build(self, b, p):
            float(p.amp)                                   # a host scalar: refused
            return super().build(b, p)

    registry.add(Baker)
    with pytest.raises(AttributeError, match="amp is dynamic"):
        _gen("acme.baker")


def test_stage_builder_is_created_by_the_orchestrator_only():
    with pytest.raises(TypeError, match="created by the orchestrator"):
        StageBuilder(stage=None, params=None, layout=Layout(1, 1, "cpu"), scale=1.0,
                     seed=None, raw=3, outputs={}, dynamic={})


def test_private_pipeline_helpers_refuse_a_stage_class(toys):
    from track_gen._src import warp_pipeline

    cfg = TrackGenConfig(generator="toy.wobble", device=DEV, num_envs=E)
    with pytest.raises(TypeError, match="Stage generators run through TrackGenerator"):
        warp_pipeline._inflate_warp_alloc(cfg)
    gen = _gen()
    with pytest.raises(TypeError, match="Stage generators run through TrackGenerator"):
        warp_pipeline._run_pipeline(cfg, gen._seed_buf, out=gen._track, scratch=gen._scratch,
                                    generator_spec=ToyWobble)
    with pytest.raises(RuntimeError, match="already built"):
        warp_pipeline._inflate_warp_alloc(gen._config, generator_spec=gen._generator_spec)


# ---------------------------------------------------------------------------------------
# Capture classes
# ---------------------------------------------------------------------------------------

def test_host_stage_runs_eagerly(toys):
    gen = _gen("toy.wobble_host")
    assert gen._capturable is False and gen._runner.capturable is False
    assert registry.describe("generator", "toy.wobble_host").capturable is False
    assert _same(_snap(gen.generate()), _snap(_gen().generate()))
    assert gen.graph is None


def test_capture_class_may_depend_on_params():
    class Dependent(ToyWobble):
        name = "acme.dependent"

        @classmethod
        def capture_class(cls, params):
            return CaptureClass.HOST if params.lobes > 6 else CaptureClass.STATIC

    registry.add(Dependent)
    assert registry.describe("generator", "acme.dependent").capturable is True
    assert _gen("acme.dependent")._capturable is True
    assert _gen("acme.dependent", generator_params={"lobes": 8})._capturable is False

    class Bounded(ToyWobble):
        name = "acme.bounded"

        @classmethod
        def capture_class(cls, params):
            return CaptureClass.BOUNDED

    registry.add(Bounded)
    gen = _gen("acme.bounded")             # BOUNDED: capturable (roadmap PR12)
    assert gen._capturable is True
    assert gen.program.capture_class is CaptureClass.BOUNDED
    assert len(gen.program.segments) == 1 and gen.program.host_stage_ids == ()


# ---------------------------------------------------------------------------------------
# Entry points, selection-only validation, Course
# ---------------------------------------------------------------------------------------

def test_stage_plugin_from_a_fake_distribution_runs(monkeypatch):
    monkeypatch.delenv(POLICY_ENV, raising=False)
    with registry.scoped(fresh=True):
        install_entry_points(monkeypatch, FakeEntryPoint(
            "toystage", register_stage_toys, FakeDist("toy-stage-plugins", "0.2.0")))
        gen = _gen()
        assert gen.generate().valid.numpy().tolist() == [1] * E
        info = registry.describe("generator", "toy.wobble")
        assert info.distribution == ("toy-stage-plugins", "0.2.0")
        assert info.entry_point == "toystage"
        assert not registry.quarantined


def test_a_broken_stage_plugin_is_quarantined(monkeypatch):
    monkeypatch.delenv(POLICY_ENV, raising=False)

    class Broken(ToyWobble):
        name = "broken.stage"
        output = "anchors"

    with registry.scoped(fresh=True):
        install_entry_points(monkeypatch, FakeEntryPoint(
            "broken", lambda reg: reg.add(Broken), FakeDist("broken-stage")))
        with pytest.warns(RuntimeWarning, match="quarantined"):
            registry.available("generator")
        assert "output='anchors' is a gate generator" in registry.quarantined["broken"]


def test_params_are_validated_only_for_the_selected_generator():
    calls = []

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Recording(Params):
        x: int = static(1)

        def validate(self, layout):
            calls.append(layout)

    class S(ToyWobble):
        name = "acme.recording"
        Params = Recording

        def build(self, b, p):
            xy = b.write("xy", wp.vec2f, per="raw")
            return lambda ctx: ctx.fill(xy, wp.vec2f(0.0, 0.0))

    registry.add(S)
    TrackGenerator(TrackGenConfig(generator="bezier", device=DEV, num_envs=1), _rng(n=1))
    assert calls == []
    _gen("acme.recording")
    assert calls == [Layout(E, 384, "cpu")]


def test_course_passes_generator_params_through(toys):
    from track_gen.course import Course, CourseConfig

    course = Course(CourseConfig(
        mode="track", gen=TrackGenConfig(generator="toy.wobble", device=DEV, num_envs=2,
                                         generator_params={"lobes": 3}),
        seeds=3, checkpoint_spacing=0.3))
    assert course.generator.params["generator"].params == ToyWobbleParams(lobes=3)
    assert course.result.valid.numpy().tolist() == [1, 1]


def test_param_table_and_stage_params_are_read_only_mappings(toys):
    gen = _gen()
    assert isinstance(gen.inputs, MappingProxyType)
    with pytest.raises(TypeError):
        gen.params["generator"]["amp"] = None                      # type: ignore[index]
    with pytest.raises(KeyError, match="no stage 'relax'"):
        gen.params["relax"]
