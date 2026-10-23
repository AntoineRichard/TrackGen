"""``randomize.uniform`` (roadmap PR13b): per-env uniform draws of another stage's
DYNAMIC float32 param, in the graph: the presets, the planner rules, the semantics
(range, determinism, redraws, masks, same-generation reach), the drawn-array contract of
``params`` and the launch cost. CPU; the caller-graph proof is
tests/test_results_cuda.py."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests import _channel_toys as ctoys
from tests import _gate_toys as gtoys
from tests._plugin_toys import register_stage_toys
from track_gen import GateGenerator, TrackGenerator
from track_gen.pipeline import PipelineSpec, PlanError, StageRef, build, presets
from track_gen.plugin import registry
from track_gen.stages.randomize import RandomizeUniformParams
from track_gen.testing.trace import trace_launches

DEV = "cpu"
E = 6
RID = "randomize.generator.amp"
LO, HI = 0.05, 0.35


@pytest.fixture(autouse=True)
def toys():
    register_stage_toys(registry)
    ctoys.register(registry)
    gtoys.register(registry)


def _spec(randomize=None, generator="toy.wobble", n=E, **kw):
    return presets.track(num_envs=n, device=DEV, generator=generator, randomize=randomize,
                         **kw)


def _wobble(n=E, lo=LO, hi=HI, **kw):
    return _spec({"generator.amp": (lo, hi)}, n=n, **kw)


def _drawn(gen) -> np.ndarray:
    return gen.params["generator"]["amp"].numpy().copy()


def _seeds(values):
    return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=DEV)


# ---------------------------------------------------------------------------------------
# Presets and the spec
# ---------------------------------------------------------------------------------------

def test_presets_insert_each_randomizer_right_before_its_target():
    spec = _wobble()
    ref = spec.stages[0]
    assert ref == StageRef("randomize", "uniform", params={
        "stage": "generator", "field": "amp", "lo": LO, "hi": HI}, id=RID)
    assert spec.ids[:2] == (RID, "generator")
    w = presets.track(num_envs=E, device=DEV, generator="acme.ribbon",
                      width=StageRef("width", "from_channel", ports={"src": "cl.acme.hw"}),
                      randomize={"width.scale": (0.8, 1.2)})
    i = w.ids.index("width")
    assert w.ids[i - 1] == "randomize.width.scale"
    back = PipelineSpec.from_dict(spec.to_dict())                # format /2, additive
    assert back.ids == spec.ids and dict(back.stages[0].params) == dict(ref.params)
    assert spec.to_dict()["format"] == "track_gen.pipeline/2"
    assert build(back, seeds=1).spec.ids[0] == RID
    seq = [StageRef("randomize", "uniform", params={"stage": "generator", "field": "amp",
                                                    "lo": 0.1, "hi": 0.2}, id="r")]
    assert presets.track(num_envs=E, device=DEV, generator="toy.wobble",
                         randomize=seq).ids[0] == "r"
    with pytest.raises(ValueError, match=r"randomize: key 'amp' must be '<stage id>\."
                                         r"<field>'"):
        _spec({"amp": (0.1, 0.2)})
    with pytest.raises(TypeError, match=r"randomize\['generator\.amp'\] must be a \(lo, "
                                        r"hi\) pair"):
        _spec({"generator.amp": 0.1})
    with pytest.raises(TypeError, match=r"randomize: expected StageRef\('randomize'"):
        _spec([StageRef("relax", "none")])
    with pytest.raises(ValueError, match=r"^randomize: key 'nope\.amp' names stage "
                                         r"'nope', which this pipeline does not have; "
                                         r"stages: \['generator', "):
        _spec({"nope.amp": (0.1, 0.2)})
    g = presets.gates(num_envs=E, device=DEV, generator=gtoys.SUPERFORMULA,
                      source="stations", randomize={"generator.m": (3.0, 9.0)})
    assert g.ids[:2] == ("randomize.generator.m", "generator")


def test_a_preset_spec_with_a_randomizer_after_insert_after_builds():
    # insert_after("generator", x) puts x right after the generator; the randomizer
    # stays listed before it, and the spec still builds.
    spec = _wobble().insert_after("generator", StageRef("relax", "none", id="pre"))
    spec = spec.remove("relax")
    assert spec.ids[:3] == (RID, "generator", "pre")
    assert build(spec, seeds=1).spec.ids[:3] == (RID, "generator", "pre")


# ---------------------------------------------------------------------------------------
# Planner rules
# ---------------------------------------------------------------------------------------

def _with(spec, ref, before):
    return spec.insert_before(before, ref)


def _r(stage="generator", field="amp", lo=LO, hi=HI, id=None):
    return StageRef("randomize", "uniform", params={"stage": stage, "field": field,
                                                    "lo": lo, "hi": hi},
                    id=id or f"randomize.{stage}.{field}")


def test_the_planner_rules():
    base = _spec()
    with pytest.raises(PlanError, match=r"^stage 'randomize\.generator\.amp' "
                                        r"\(randomize\.uniform\) draws generator\.amp, so "
                                        r"it must be listed before stage 'generator': "
                                        r"spec\.insert_before\('generator', \.\.\.\)$"):
        build(base.insert_after("generator", _r()))
    with pytest.raises(PlanError, match=r"stage 'randomize\.width\.scale' "
                                        r"\(randomize\.uniform\) is listed before the "
                                        r"generator: only the generator's randomizers "
                                        r"\(stage='generator'\) may precede it; list it "
                                        r"right before stage 'width'"):
        build(_with(base, _r("width", "scale"), "generator"))
    with pytest.raises(PlanError, match=r"the first stage must be the generator"):
        build(_with(base, StageRef("relax", "none", id="r0"), "generator"))
    with pytest.raises(PlanError, match=r"draws nope\.amp, but the pipeline has no stage "
                                        r"'nope'"):
        build(_with(base, _r("nope"), "resample"))
    with pytest.raises(PlanError, match=r"stage 'generator' \(generator\.toy\.wobble\) has "
                                        r"no param 'nope'; its dynamic fields: \['amp'\]"):
        build(_with(base, _r(field="nope"), "generator"))
    # A plugin's own static field: the plugin wording (not "built-in knobs").
    with pytest.raises(PlanError, match=r"^generator\.lobes is static \(declared with "
                                        r"plugin\.static\): only plugin\.dynamic float32 "
                                        r"fields can be drawn; drawable fields in this "
                                        r"pipeline: \['generator\.amp'\]$"):
        build(_with(base, _r(field="lobes"), "generator"))
    bez = presets.track(num_envs=E, device=DEV, randomize={"generator.rad": (0.2, 0.3)})
    with pytest.raises(PlanError, match=r"generator\.rad is static: built-in knobs are "
                                        r"static in 0\.x; dynamic float32 fields in this "
                                        r"pipeline: \[\]"):
        build(bez)
    with pytest.raises(PlanError, match=r"^relax\.relax_iters is static: built-in knobs"):
        build(_with(base, _r("relax", "relax_iters"), "relax"))
    rib = presets.track(num_envs=E, device=DEV, generator="acme.ribbon",
                        width=StageRef("width", "from_channel",
                                       ports={"src": "cl.acme.hw"}),
                        randomize={"width.scale": (0.8, 1.2)})
    with pytest.raises(PlanError, match=r"draws randomize\.width\.scale\.lo: stage "
                                        r"'randomize\.width\.scale' is a randomizer"):
        build(_with(rib, _r("randomize.width.scale", "lo", id="rr"),
                    "randomize.width.scale"))
    with pytest.raises(PlanError, match=r"stages 'randomize\.generator\.amp' and 'r2' both "
                                        r"draw generator\.amp"):
        build(_with(_wobble(), _r(id="r2"), "generator"))
    with pytest.raises(PlanError, match=r"the draw range \[0\.05, 0\.5\] lies outside "
                                        r"generator\.amp's declared range \[0\.0, 0\.4\]"):
        build(_wobble(hi=0.5))
    with pytest.raises(ValueError, match=r"randomize\.uniform: lo must be <= hi"):
        build(_wobble(lo=0.3, hi=0.1))
    with pytest.raises(ValueError, match=r"randomize\.uniform: stage must name"):
        build(_with(base, _r(stage=""), "resample"))


def test_an_int32_dynamic_field_is_refused():
    import dataclasses

    from track_gen import plugin

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(plugin.Params):
        k: int = plugin.dynamic(3, lo=0, hi=10)

    class W(plugin.Stage):
        kind = "relax"
        name = "acme.int_relax"
        Params = P
        reads = {"xy": "cl.xy"}
        writes = {"xy": "cl.xy"}

        def build(self, b, p):
            x, out = b.read("xy"), b.write("xy", wp.vec2f, per="cl")
            return lambda ctx: ctx.copy(out, x)

    registry.add(W)
    spec = presets.track(num_envs=E, device=DEV, relax="acme.int_relax",
                         randomize={"relax.k": (1, 2)})
    with pytest.raises(PlanError, match=r"^relax\.k is int32: randomize\.uniform draws "
                                        r"float32 fields in 0\.x$"):
        build(spec)


# ---------------------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------------------

def test_draws_are_in_range_distinct_and_deterministic():
    a = TrackGenerator(_wobble(), seeds=3)
    b = TrackGenerator(_wobble(), seeds=3)
    da, db = _drawn(a), _drawn(b)
    assert ((da >= LO) & (da < HI)).all() and len(set(da.tolist())) == E
    assert da.tobytes() == db.tobytes()                       # per (seed, epoch)
    c = TrackGenerator(_wobble(), seeds=50)
    assert not np.array_equal(_drawn(c), da)
    a.generate()
    assert _drawn(a).tobytes() == da.tobytes()                # no advance: same draws
    a.generate(advance=True)
    redrawn = _drawn(a)
    assert (redrawn != da).all()                              # advance redraws every env
    b.generate(advance=True)
    assert _drawn(b).tobytes() == redrawn.tobytes()


def test_the_draw_reaches_the_generator_in_the_same_generation():
    gen = TrackGenerator(_wobble(), seeds=3)
    for step in range(3):
        if step:
            gen.generate(advance=True)
        drawn = _drawn(gen)
        twin = TrackGenerator(_spec(), seeds=3)
        for _ in range(step):
            twin.seeds.advance()
        twin.params["generator"].set("amp", wp.array(drawn, dtype=wp.float32,
                                                     device=DEV))
        twin.generate()
        for f in ("center", "outer", "valid", "count", "length"):
            assert getattr(gen.program.result, f).numpy().tobytes() == \
                getattr(twin.program.result, f).numpy().tobytes(), (step, f)


def test_masked_advance_redraws_the_masked_envs_only():
    gen = TrackGenerator(_wobble(), seeds=3)
    before = _drawn(gen)
    center = gen.program.result.center.numpy().reshape(E, -1, 3).copy()
    m = np.array([1, 0, 1, 0, 0, 1], np.int32)
    gen.generate(advance=True, mask=wp.array(m, dtype=wp.int32, device=DEV))
    after = _drawn(gen)
    assert (after[m == 1] != before[m == 1]).all()
    assert after[m == 0].tobytes() == before[m == 0].tobytes()
    now = gen.program.result.center.numpy().reshape(E, -1, 3)
    assert now[m == 0].tobytes() == center[m == 0].tobytes()


def test_the_drawn_array_contract_and_the_curriculum():
    gen = TrackGenerator(_wobble(), seeds=3)
    sp = gen.params["generator"]
    assert dict(sp.drawn) == {"amp": RID}
    assert f"amp=<drawn by '{RID}'>" in repr(sp)
    with pytest.raises(ValueError, match=r"^generator\.amp is drawn every generation by "
                                         r"stage 'randomize\.generator\.amp'; set its "
                                         r"bounds: params\['randomize\.generator\.amp'\]\."
                                         r"set\('lo' \| 'hi', \.\.\.\)$"):
        sp.set("amp", 0.1)
    assert gen._latch is None                    # the only dynamic field is drawn
    rp = gen.params[RID]
    assert set(rp) == {"lo", "hi"}
    rp.set("lo", 0.2)
    rp.set("hi", 0.21)
    gen.generate(advance=True)
    d = _drawn(gen)
    assert ((d >= 0.2) & (d < 0.21)).all()
    # bounds outside the field's declared range at run time are clamped to it
    rp.set("hi", 0.39)
    rp.set("lo", 0.39)
    gen.generate(advance=True)
    assert np.allclose(_drawn(gen), 0.39)
    assert gen.program.result.valid.numpy().any()
    assert isinstance(gen.program.spec.stages[0].params, RandomizeUniformParams)
    # A scalar bound past the drawn field's declared range is refused, like any bounded
    # dynamic field (the kernel would clamp: a point mass on the bound).
    with pytest.raises(ValueError, match=r"^RandomizeUniformParams\.hi must be in "
                                         r"\[0\.0, 0\.4\], the declared range of "
                                         r"generator\.amp \(the field this stage draws: "
                                         r"a draw past it would pile onto the bound\), "
                                         r"got 0\.9$"):
        rp.set("hi", 0.9)
    with pytest.raises(ValueError, match=r"RandomizeUniformParams\.lo must be in "
                                         r"\[0\.0, 0\.4\]"):
        rp.set("lo", -0.1)
    # An array write is not range-checked (no host sync): the kernel clamps it.
    rp.set("hi", wp.full(E, 0.9, dtype=wp.float32, device=DEV))
    rp.set("lo", wp.full(E, 0.5, dtype=wp.float32, device=DEV))
    gen.generate(advance=True)
    assert np.allclose(_drawn(gen), 0.4)


def test_the_stages_package_does_not_load_the_pipeline_package():
    """``track_gen.stages`` (L3) re-exports PARAM_WRITER_KINDS from ``track_gen.plugin``,
    not from ``track_gen.pipeline`` (L4), and both constants agree."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    import track_gen.pipeline.spec as spec_mod  # noqa: PLC0415
    import track_gen.stages as stages_mod  # noqa: PLC0415

    assert stages_mod.PARAM_WRITER_KINDS == spec_mod.PARAM_WRITER_KINDS == ("randomize",)
    code = ("import sys, track_gen.stages, track_gen.stages.randomize, "
            "track_gen.stages.external; "
            "assert 'track_gen.pipeline' not in sys.modules, "
            "sorted(m for m in sys.modules if m.startswith('track_gen.pipeline'))")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_launches_and_segments():
    plain = TrackGenerator(_spec(), seeds=3)
    gen = TrackGenerator(_wobble(), seeds=3)
    with trace_launches() as tp:
        plain.generate()
    with trace_launches() as tr:
        gen.generate()
    names = [r.kernel for r in tr]
    assert len(tr) == len(tp) + 2                         # draw + lo/hi latch - amp latch
    i = names.index("track_gen.stages.randomize._randomize_uniform_k")
    assert names[:i] == ["track_gen.seeds._effective_seed_k",
                         "track_gen._src.generator_binding._latch_all_f32_k",
                         "track_gen._src.generator_binding._latch_all_f32_k"]
    assert names[i + 1] == "tests._plugin_toys._wobble_k"
    assert gen.program.spec.ids[0] == RID
    host = TrackGenerator(presets.track(num_envs=E, device=DEV, generator="toy.wobble_host",
                                        randomize={"generator.amp": (LO, HI)}), seeds=3)
    segs = host.program.segments
    assert segs[0].stage_ids == (RID,) and not segs[0].host
    assert segs[1].stage_ids == ("generator",) and segs[1].host


def test_gate_and_width_randomizers():
    g = GateGenerator(presets.gates(num_envs=E, device=DEV, generator=gtoys.SUPERFORMULA,
                                    source="stations",
                                    randomize={"generator.m": (3.0, 9.0)}), seeds=2)
    m = g.params["generator"]["m"].numpy()
    assert ((m >= 3.0) & (m < 9.0)).all() and len(set(m.tolist())) == E
    assert g.program.result.valid.numpy().any()
    spec = presets.track(num_envs=E, device=DEV, generator="acme.ribbon",
                         width=StageRef("width", "from_channel",
                                        ports={"src": "cl.acme.hw"}),
                         randomize={"width.scale": (0.8, 1.2)})
    t = TrackGenerator(spec, seeds=2)
    s = t.params["width"]["scale"].numpy()
    assert ((s >= 0.8) & (s < 1.2)).all()
    hw = t.program.tap("cl.half_width").numpy().reshape(E, -1)
    src = t.program.tap("cl.acme.hw").numpy().reshape(E, -1)
    cnt = t.program.result.count.numpy()
    for e in range(E):
        np.testing.assert_allclose(hw[e, :cnt[e]], s[e] * src[e, :cnt[e]], rtol=1e-6)
