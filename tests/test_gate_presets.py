"""``presets.gates``, ``legacy.to_gate_pipeline_spec``, the gate planner rules and the
``GateGenerator(spec)`` dispatch (roadmap PR11, CPU).

The preset is the default gate pipeline as data; ``presets.gates(num_envs=E, device=d)``
resolves to the same stages, with equal params, as ``GateGenerator(GateGenConfig(
num_envs=E, device=d))`` and builds byte-identical GateSequences. The planner plans gate
Programs (``BatchSpec.max_gates`` set) with the track rules plus the gate ones (per-stage
point domains, the elevation rewiring, capacity, order compatibility, provided
attributes, parameter agreement).
"""
from __future__ import annotations

import dataclasses
import json
import warnings

import numpy as np
import pytest
import warp as wp

from tests import _gate_toys as gt
from tests._stage_toys import HILLS
from tests._stage_toys import register as register_stage_toys
from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenerator
from track_gen.pipeline import (BatchSpec, PipelineSpec, PlanError, StageRef, build,
                                legacy, presets)
from track_gen.ops import hash32
from track_gen.plugin import Stage, registry
from track_gen.stages import params as sp

DEV = "cpu"
E = 3
FIELDS = ("position", "tangent", "forward", "orientation", "half_size", "left", "right",
          "valid", "count", "generation", "fail_mask")


@pytest.fixture(autouse=True)
def toys():
    gt.register(registry)
    register_stage_toys(registry)


def _rng(seed=7, n=E):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _gates(**kw):
    return presets.gates(num_envs=E, device=DEV, **kw)


def _bytes(seq) -> dict:
    return {f: getattr(seq, f).numpy().tobytes() for f in FIELDS}


def _resolved(spec) -> list:
    return [(r.kind, r.name, r.slot_id, r.params) for r in build(spec, seeds=1).spec.stages]


# ---------------------------------------------------------------------------------------
# The preset is the default gate pipeline
# ---------------------------------------------------------------------------------------

_NOISE = dict(z_base=0.1, z_min=0.0, z_max=1.5, z_noise_amplitude=0.5,
              z_noise_harmonics=4)


@pytest.mark.parametrize("kw,cfg", [
    ({}, {}),
    (dict(generator="hull"), dict(generator="hull")),
    (dict(generator="polar"), dict(generator="polar")),
    (dict(generator="checkpoint", order="raw"), dict(generator="checkpoint",
                                                     gate_ordering="raw")),
    (dict(generator="voronoi"), dict(generator="voronoi")),
    (dict(order="random_pairs"), dict(gate_ordering="random_pairs")),
    (dict(elevation=("noise", _NOISE)), dict(z_profile="noise", **_NOISE)),
    (dict(gate_solve_iters=0), dict(gate_solve_iters=0)),
    (dict(gate_align="full_tangent", gate_width=0.05),
     dict(gate_align="full_tangent", gate_width=0.05)),
    (dict(min_gates=6, gate_radius=0.03, z_valid_grade=0.5, scale=1.5),
     dict(min_gates=6, gate_radius=0.03, z_valid_grade=0.5, scale=1.5)),
    (dict(z_base=0.5), dict(z_base=0.5)),
])
def test_the_preset_builds_byte_identical_to_the_config(kw, cfg):
    spec = _gates(**kw)
    ref = legacy.to_gate_pipeline_spec(GateGenConfig(device=DEV, num_envs=E, **cfg))
    assert spec.ids == ref.ids
    assert _resolved(spec) == _resolved(ref)
    a = GateGenerator(spec, _rng())
    b = GateGenerator(GateGenConfig(device=DEV, num_envs=E, **cfg), _rng())
    c = build(spec, seeds=_rng())
    assert _bytes(a.generate()) == _bytes(b.generate()) == _bytes(c.run(sync=True))


def test_the_default_stage_lists():
    assert _gates().ids == ("generator", "order", "gate_relax", "elevation", "lift",
                            "tangents", "gate_frames", "validity")
    assert _gates(generator="polar").ids == (
        "generator", "order", "normalize", "gate_relax", "elevation", "lift", "tangents",
        "gate_frames", "validity")
    assert _gates(elevation="uniform", gate_solve_iters=0).ids == (
        "generator", "order", "gate_relax", "gate_arclen", "elevation", "lift",
        "tangents", "gate_frames", "validity")
    spec = _gates()
    assert (spec.stages[0].kind, spec.stages[0].name) == ("gate_generator", "bezier")
    assert spec.batch == BatchSpec(num_envs=E, device=DEV, n_max=384, max_gates=32)
    # gate_relax.none when the float32 2r is 0 (the dd181f0 kernel returned at once)
    for kw in (dict(gate_solve_iters=0), dict(gate_radius=0.0), dict(gate_radius=1e-46)):
        assert [r.name for r in _gates(**kw).stages if r.kind == "gate_relax"] == ["none"]


def test_source_auto_and_explicit():
    assert _gates(generator="bezier").stages[0].kind == "gate_generator"
    sf = _gates(generator=gt.SUPERFORMULA)
    assert [r.kind for r in sf.stages[:3]] == ["generator", "stations", "order"]
    assert sf.stages[2].name == "raw" and sf.batch.max_gates == 12
    st = _gates(generator="bezier", source="stations", gates=8)
    assert (st.stages[0].kind, st.stages[1].name) == ("generator", "raw")
    assert st.stages[1].params == {"gates": 8} and st.batch.max_gates == 8
    rel = _gates(generator=gt.SUPERFORMULA, source="relaxed_stations")
    # Since 0.2.0.dev4 the relaxed_stations sub-pipeline has the width stage (before
    # relax): width.constant launches nothing, the PR11 trace is unchanged.
    assert rel.ids[:6] == ("generator", "resample", "width", "relax", "stations", "order")
    assert _ref(rel, "stations").name == "relaxed"
    ref = StageRef("gate_generator", "hull")
    assert _gates(generator=ref).stages[0].slot_id == "generator"
    with pytest.raises(ValueError, match=r"presets.gates: unknown generator 'nope'; gate "
                                         r"generators: \[.*'bezier'.*\]; track generators "
                                         r"\(gates through stations\): \[.*'repulsive'"):
        _gates(generator="nope")
    with pytest.raises(ValueError, match="gates= sets the station count of a centerline "
                                         "source; an anchors generator decides its own"):
        _gates(gates=8)
    with pytest.raises(ValueError, match="source must be one of"):
        _gates(source="bogus")
    with pytest.raises(ValueError, match="stations= selects the stations of a centerline"):
        _gates(stations="raw")


def test_selection_errors():
    with pytest.raises(ValueError, match="gate_align must be one of"):
        _gates(gate_align="tilted")
    with pytest.raises(ValueError, match="on_invalid must be 'keep' or 'poison'"):
        _gates(on_invalid="drop")
    with pytest.raises(ValueError, match="checks must start with 'gate_default'"):
        _gates(checks=(gt.MAX_HEIGHT,))
    with pytest.raises(TypeError, match="checks must be a sequence of selections"):
        _gates(checks="gate_default")
    with pytest.raises(ValueError, match="expected a StageRef of kind 'order'"):
        _gates(order=StageRef("elevation", "flat"))
    with pytest.raises(ValueError, match="order=: gate generator 'toy.ring' is a legacy "
                                         "GateGeneratorSpec that orders its own gates"):
        from tests._plugin_toys import TOY_RING

        registry.add(TOY_RING)
        _gates(generator="toy.ring", order="raw")


def _ref(spec, sid):
    return next(r for r in spec.stages if r.slot_id == sid)


# ---------------------------------------------------------------------------------------
# Provided attributes (acme.helix)
# ---------------------------------------------------------------------------------------

def test_a_provider_drops_the_stages_it_provides():
    spec = _gates(generator=gt.HELIX)
    for dropped in ("gate_arclen", "elevation", "tangents"):
        assert dropped not in spec.ids
    assert spec.ids == ("generator", "order", "carry_z", "carry_tangent", "gate_relax",
                        "lift", "gate_frames", "validity")
    assert _ref(spec, "order").name == "raw"
    posed = _gates(generator=gt.HELIX_POSED)
    assert _ref(posed, "gate_frames").name == "posed"
    assert _ref(posed, "validity").params["check_crossing"] == "always"
    with pytest.raises(ValueError, match=r"presets.gates\(elevation='noise'\): gate "
                                         r"generator 'acme.helix' provides its own "
                                         r"altitudes \(z\); leave elevation at its default"):
        _gates(generator=gt.HELIX, elevation="noise")
    _gates(generator=gt.HELIX, elevation=None)                 # flat: admitted
    with pytest.raises(ValueError, match="gate_width: gate generator 'acme.helix_posed' "
                                         "provides per-gate half_size"):
        _gates(generator=gt.HELIX_POSED, gate_width=0.05)


def test_a_provider_through_the_config():
    # gate_ordering=None (the default) is the generator's default order, the rule of
    # presets.gates(order=None): "raw" for the helix.
    cfg = GateGenConfig(generator=gt.HELIX, num_envs=E)
    gen = GateGenerator(cfg, _rng())
    assert cfg.gate_ordering is None and gen.config.gate_ordering is None
    assert gen._config.gate_ordering == "raw"
    assert _ref(legacy.to_gate_pipeline_spec(cfg), "order").name == "raw"
    want = GateGenerator(_gates(generator=gt.HELIX), _rng())
    assert _bytes(gen.generate()) == _bytes(want.generate())
    # An explicit name keeps its check and message.
    with pytest.raises(ValueError, match=r"gate generator 'acme.helix' does not support "
                                         r"gate_ordering 'ccw'; supported: \['raw'\] \(its "
                                         r"preferred order is 'raw'\)"):
        GateGenerator(GateGenConfig(generator=gt.HELIX, gate_ordering="ccw"))
    GateGenerator(GateGenConfig(generator=gt.HELIX, gate_ordering="raw"))
    for kw, match in ((dict(z_profile="noise"), r"GateGenConfig.z_profile='noise': gate "
                                                r"generator 'acme.helix' provides its own "
                                                r"altitudes"),
                      (dict(z_base=0.5), r"GateGenConfig.z_base=0.5: gate generator "
                                         r"'acme.helix' provides its own altitudes \(z\); "
                                         r"leave z_profile='flat' and z_base=0.0")):
        with pytest.raises(ValueError, match=match):
            GateGenerator(GateGenConfig(generator=gt.HELIX, **kw))
    with pytest.raises(ValueError, match="GateGenConfig.gate_width: gate generator "
                                         "'acme.helix_posed' provides per-gate half_size"):
        GateGenerator(GateGenConfig(generator=gt.HELIX_POSED, gate_ordering="raw",
                                    gate_width=0.1))


def test_a_provided_attribute_nobody_reads_is_refused():
    spec = _gates(generator="checkpoint", order="raw").replace(
        "generator", StageRef("gate_generator", gt.HELIX))
    with pytest.raises(PlanError, match=r"stage 'generator' \(gate_generator.acme.helix\) "
                                        r"provides 'tangent' \(anchors.tangent\), but no "
                                        r"stage reads it; stage 'tangents' "
                                        r"\(tangents.central\) writes gates.tangent "
                                        r"instead: carry it with carry.tangent"):
        build(spec, seeds=1)
    no_tangents = spec.remove("tangents").insert_after(
        "order", StageRef("carry", "tangent", id="carry_tangent"))
    with pytest.raises(PlanError, match=r"provides 'z' \(anchors.z\), but no stage reads "
                                        r"it; stage 'elevation' \(elevation.flat\) writes "
                                        r"gates.z instead: carry it with carry.z"):
        build(no_tangents, seeds=1)


def test_carry_needs_order_raw():
    spec = _gates().insert_after("order", StageRef("carry", "z", id="carry_z"))
    with pytest.raises(PlanError, match=r"stage 'carry_z' \(carry.z\) carries anchors.z by "
                                        r"anchor index, which needs order.raw; stage 'order' "
                                        r"is order.ccw \(it permutes the anchors\)"):
        build(spec, seeds=1)
    spec = _gates(generator=gt.HELIX).replace("order", StageRef("order", "ccw"))
    with pytest.raises(PlanError, match=r"gate generator 'acme.helix' does not support "
                                        r"gate_ordering 'ccw'"):
        build(spec, seeds=1)
    # a whitelist-free provider would still be refused by the carry rule
    from tests._gate_toys import Helix

    class Wide(Helix):
        name = "acme.wide"
        orderings = frozenset({"raw", "ccw"})

    with pytest.raises(ValueError, match=r"so orderings must be frozenset\(\{'raw'\}\)"):
        registry.add(Wide)


# ---------------------------------------------------------------------------------------
# Order compatibility, unknown names, capacity
# ---------------------------------------------------------------------------------------

def test_order_compatibility():
    for gen in ("polar", "checkpoint", "voronoi"):
        with pytest.raises(ValueError, match=rf"gate generator '{gen}' does not support "
                                             rf"gate_ordering 'random_pairs'; supported: "
                                             rf"\['acme.reverse', 'ccw', 'raw'\]"):
            GateGenerator(GateGenConfig(generator=gen, gate_ordering="random_pairs"))
        with pytest.raises(PlanError, match=rf"gate generator '{gen}' does not support "
                                            rf"gate_ordering 'random_pairs'"):
            build(_gates(generator=gen, order="random_pairs"), seeds=1)
    with pytest.raises(PlanError, match=r"supported: \['raw'\] \(its preferred order is "
                                        r"'raw'\)"):
        build(_gates(generator=gt.HELIX, order=gt.REVERSE), seeds=1)
    # A third-party order runs with every built-in that does not exclude it.
    for gen in ("bezier", "polar"):
        GateGenerator(GateGenConfig(generator=gen, gate_ordering=gt.REVERSE))


@pytest.mark.parametrize("gen", ["bezier", "hull", "polar", "checkpoint", "voronoi"])
def test_the_default_ordering_of_a_builtin_is_ccw(gen):
    """``GateGenConfig.gate_ordering=None`` resolves to 0.2.0.dev3's ``"ccw"`` for every
    built-in: the same spec (hence bytes) as the explicit name."""
    from track_gen.stages.gates import _default_order

    assert _default_order(registry.get("gate_generator", gen), registry) == "ccw"
    cfg = GateGenConfig(generator=gen, num_envs=E, device=DEV)
    assert cfg.gate_ordering is None
    explicit = dataclasses.replace(cfg, gate_ordering="ccw")
    assert legacy.to_gate_pipeline_spec(cfg) == legacy.to_gate_pipeline_spec(explicit)
    assert GateGenerator(cfg, _rng())._config.gate_ordering == "ccw"


def test_supported_lists_are_heads_with_only_the_builtin_orders():
    from track_gen.stages.gates import _supported_orderings

    with registry.scoped():
        registry.unregister("order", gt.REVERSE)
        assert _supported_orderings(registry, "bezier") == ["ccw", "random_pairs"]
        assert _supported_orderings(registry, "polar") == ["ccw", "raw"]
        with pytest.raises(ValueError, match=r"supported: \['ccw', 'raw'\] \(its preferred "
                                             r"order is 'ccw'\)$"):
            GateGenerator(GateGenConfig(generator="polar", gate_ordering="random_pairs"))
    assert _supported_orderings(registry, "bezier") == ["acme.reverse", "ccw",
                                                       "random_pairs"]


def test_unknown_order_and_elevation_names():
    with pytest.raises(ValueError, match=r"GateGenConfig.gate_ordering: unknown order "
                                         r"'spiral'; available: \['acme.reverse', 'ccw', "
                                         r"'random_pairs', 'raw'\]"):
        GateGenerator(GateGenConfig(gate_ordering="spiral"))
    with pytest.raises(ValueError, match=r"GateGenConfig.z_profile: unknown elevation "
                                         r"'bogus'; available: \['acme.sinusoid_hills', "
                                         r"'flat', 'noise', 'random_walk', 'uniform'\]"):
        GateGenerator(GateGenConfig(z_profile="bogus"))
    with pytest.raises(PlanError, match=r"stage 'order' \(order.spiral\): unknown order "
                                        r"'spiral'"):
        build(_gates(order="spiral"), seeds=1)


def test_capacity_errors():
    with pytest.raises(ValueError, match=r"GateGenConfig.min_gates=20 is too large for gate "
                                         r"generator 'polar'; generator can produce at "
                                         r"most 12 gates."):
        GateGenerator(GateGenConfig(generator="polar", min_gates=20))
    with pytest.raises(ValueError, match=r"GateGenConfig.max_gates=8 is too small for gate "
                                         r"generator 'bezier'; required max_gates 13."):
        GateGenerator(GateGenConfig(max_gates=8, min_gates=4))
    with pytest.raises(PlanError, match=r"stage 'generator' \(gate_generator.bezier\): "
                                        r"BatchSpec.max_gates=8 is too small for gate "
                                        r"generator 'bezier'; required max_gates 13."):
        build(_gates(max_gates=8), seeds=1)
    with pytest.raises(PlanError, match=r"stage 'validity' \(validity.gate_default\): "
                                        r"min_gates=20 is too large for gate generator "
                                        r"'polar'; generator can produce at most 12 gates."):
        build(_gates(generator="polar", min_gates=20), seeds=1)


def test_gate_relax_params():
    for p in ({"gate_solve_iters": 0}, {"gate_radius": 1e-46}):
        spec = _gates().replace("gate_relax", StageRef("gate_relax", "spheres", params=p))
        with pytest.raises(ValueError, match=r"gate_relax.spheres needs gate_solve_iters >= "
                                             r"1 and a gate_radius whose float32 2\*r is > "
                                             r"0; use gate_relax.none"):
            build(spec, seeds=1)


def test_parameter_agreement():
    spec = _gates().replace("gate_frames", StageRef("gate_frames", "default",
                                                    params={"gate_width": 0.05}))
    with pytest.raises(PlanError, match=r"the stages disagree on gate_width: \{'gate_frames':"
                                        r" 0.05, 'validity': 0.0\} \(one value per pipeline"):
        build(spec, seeds=1)
    spec = _gates().replace("validity", StageRef("validity", "gate_default",
                                                 params={"gate_radius": 0.03}))
    with pytest.raises(PlanError, match=r"the stages disagree on gate_radius: "
                                        r"\{'gate_relax': 0.025, 'validity': 0.03\}"):
        build(spec, seeds=1)


def test_posed_frames_need_check_crossing_always():
    spec = _gates(generator=gt.HELIX_POSED).replace(
        "validity", StageRef("validity", "gate_default"))
    with pytest.raises(PlanError, match=r"stage 'validity' \(validity.gate_default\): posed "
                                        r"gates \(gate_frames.posed\) take their widths from "
                                        r"the generator; set check_crossing='always'"):
        build(spec, seeds=1)


# ---------------------------------------------------------------------------------------
# Domains, rewiring, plugin channels
# ---------------------------------------------------------------------------------------

def test_elevation_rewiring_on_gates():
    # presets.gates(elevation=HILLS), the config shortcut and a spec edit (no ports in
    # the ref: the planner rewires cl.* -> gates.*) are the same Program.
    by_preset = GateGenerator(_gates(elevation=HILLS), _rng())
    by_config = GateGenerator(GateGenConfig(num_envs=E, z_profile=HILLS), _rng())
    by_edit = GateGenerator(_gates(elevation="uniform").replace(
        "elevation", StageRef("elevation", HILLS)), _rng())
    want = _bytes(by_preset.generate())
    assert _bytes(by_config.generate()) == want == _bytes(by_edit.generate())
    reads, writes = by_preset.program._ports["elevation"]
    assert reads == {"cum": "gates.cum", "perim": "gates.perim", "count": "gates.count"}
    assert writes == {"z": "gates.z"}
    # z == the hills formula on the gates' own arc tables, with the stage's own salt
    # (keyed by stage id "elevation", as on tracks): numpy within float32 rounding.
    from track_gen.plugin.stage import _salt_of

    p = by_preset.program
    cum = p.tap("gates.cum").numpy().reshape(E, -1)
    perim = p.tap("gates.perim").numpy()
    z = p.tap("gates.z").numpy().reshape(E, -1)
    count = by_preset._gate_sequence.count.numpy()
    phase = wp.zeros(E, dtype=wp.float32, device=DEV)
    wp.launch(_hills_phase_k, E, [p.seed, wp.uint32(_salt_of("elevation", "phase")), phase],
              device=DEV)
    phase = phase.numpy()
    for e in range(E):
        m = int(count[e])
        s = cum[e, :m] / max(float(perim[e]), 1e-9)
        want = 0.0 + 0.2 * np.sin(2.0 * np.pi * 3.0 * s + phase[e])
        np.testing.assert_allclose(z[e, :m], want, atol=2e-5)
        assert np.all(z[e, m:] == 0.0) and m >= 4
    # replacing a FLAT elevation needs the arc tables: the error says so
    with pytest.raises(PlanError, match=r"reads 'gates.cum' \(port 'cum'\), but no earlier "
                                        r"stage writes it; insert StageRef\('gate_arclen', "
                                        r"'default'\) before it"):
        build(_gates().replace("elevation", StageRef("elevation", HILLS)), seeds=1)


@wp.kernel
def _hills_phase_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32,
                   out: wp.array(dtype=wp.float32)):
    e = wp.tid()
    st = wp.rand_init(wp.int32(hash32(wp.uint32(seed[e]) ^ salt) & wp.uint32(0x7FFFFFFF)))
    out[e] = 6.2831853 * wp.randf(st)


class _Probe(Stage):
    """Records the layout / strides it sees and a temp_like length."""

    kind = "validity"
    name = "test.probe"
    reads = {"xy": "gates.xy", "anchors": "anchors.xy", "fail": "valid.fail"}
    writes = {"fail": "valid.fail"}
    seen: dict = {}

    def build(self, b, p):
        _Probe.seen = dict(stride=int(b.layout.stride), xy=b.stride_of("xy"),
                           anchors=b.stride_of("anchors"),
                           tmp_gates=int(b.temp_like("xy").shape[0]),
                           tmp_anchors=int(b.temp_like("anchors").shape[0]),
                           per=b.per_of("anchors"))
        fail_in, fail_out = b.read("fail"), b.write("fail", wp.int32)
        b.fail_bit()

        def enqueue(ctx):
            ctx.copy(fail_out, fail_in)
        return enqueue


def test_per_stage_layouts():
    registry.add(_Probe)
    gen = GateGenerator(_gates(checks=("gate_default", "test.probe")), _rng())
    assert _Probe.seen == dict(stride=32, xy=32, anchors=13, tmp_gates=E * 32,
                               tmp_anchors=E * 13, per="anchors")
    assert gen.max_gates == 32 and gen.num_envs == E and gen.device == "cpu"
    # the relaxed-stations sub-pipeline runs on N_max, the gate stages on max_gates
    rel = GateGenerator(_gates(generator=gt.SUPERFORMULA, source="relaxed_stations",
                               n_max=200), _rng())
    assert rel.program._ports["stations"][0]["xy"] == "cl.xy"
    assert rel.program.tap("cl.xy").shape == (E * 200,)
    assert rel.program.tap("anchors.xy").shape == (E * 12,)
    assert rel.program.tap("gates.xy").shape == (E * 12,)


def test_domain_and_plugin_channel_rules():
    class Mixed(Stage):
        kind = "validity"
        name = "test.mixed"
        reads = {"xy": "cl.xy", "g": "gates.xy", "fail": "valid.fail"}
        writes = {"fail": "valid.fail"}

        def build(self, b, p):
            return lambda ctx: None

    class Plug(Mixed):
        name = "test.plug"
        reads = {"own": "cl.acme.thing", "g": "gates.xy", "fail": "valid.fail"}

    registry.add(Mixed)
    registry.add(Plug)
    with pytest.raises(PlanError, match=r"stage 'test.mixed' \(validity.test.mixed\) has "
                                        r"centerline \(cl.\*\) and gate \(gates.\*\) ports "
                                        r"after rewiring; a stage runs on one point domain"):
        build(_gates(checks=("gate_default", "test.mixed")), seeds=1)
    with pytest.raises(PlanError, match=r"plugin channel 'cl.acme.thing' is a centerline "
                                        r"channel; a gate-domain stage reads and writes "
                                        r"gates.<pkg>.<name>"):
        build(_gates(checks=("gate_default", "test.plug")), seeds=1)
    # an elevation port with no gate counterpart
    spec = _gates(elevation="noise").replace("elevation", StageRef(
        "elevation", "noise", ports={"count": "cl.count"}))
    with pytest.raises(PlanError, match=r"port 'count' reads 'cl.count', which has no gate "
                                        r"counterpart"):
        build(spec, seeds=1)
    # gate channels in a track pipeline
    with pytest.raises(PlanError, match="belongs to the gate pipeline"):
        build(presets.track(num_envs=E, device=DEV, checks=("default", gt.MAX_HEIGHT)),
              seeds=1)


# ---------------------------------------------------------------------------------------
# BatchSpec.max_gates, dispatch, the facade
# ---------------------------------------------------------------------------------------

def test_batch_max_gates_validation_and_round_trip():
    for bad in (1, 0, 2.5, True, "8"):
        spec = dataclasses.replace(_gates(), batch=dataclasses.replace(
            _gates().batch, max_gates=bad))
        with pytest.raises(PlanError, match="BatchSpec.max_gates must be None"):
            build(spec, seeds=1)
    for spec in (_gates(), _gates(generator=gt.SUPERFORMULA, source="relaxed_stations"),
                 _gates(generator=gt.HELIX_POSED, on_invalid="poison")):
        d = spec.to_dict()
        assert d["batch"]["max_gates"] == spec.batch.max_gates
        back = PipelineSpec.from_dict(json.loads(json.dumps(d)))
        assert back.to_dict() == d
        assert _bytes(GateGenerator(back, _rng()).generate()) == \
            _bytes(GateGenerator(spec, _rng()).generate())
    track = presets.track(num_envs=E, device=DEV)
    assert track.to_dict()["batch"]["max_gates"] is None


def test_dispatch_both_ways():
    for spec in (_gates(), _gates(generator=gt.SUPERFORMULA)):
        program = build(spec, seeds=_rng())
        assert type(program.result).__name__ == "GateSequence"
        assert program.layout.stride == spec.batch.max_gates
    for spec in (_gates(), _gates(generator=gt.HELIX)):
        with pytest.raises(ValueError, match=r"TrackGenerator runs a track pipeline; this "
                                             r"spec is a gate pipeline \(BatchSpec.max_gates "
                                             r"set\): use GateGenerator"):
            TrackGenerator(spec)
    with pytest.raises(ValueError, match=r"GateGenerator runs a gate pipeline "
                                         r"\(BatchSpec.max_gates set, e.g. presets.gates\); "
                                         r"this spec is a track pipeline: use TrackGenerator"):
        GateGenerator(presets.track(num_envs=E, device=DEV))
    with pytest.raises(TypeError, match="GateGenerator expects a GateGenConfig or a "
                                        "track_gen.pipeline.PipelineSpec, got dict"):
        GateGenerator({})
    with pytest.raises(PlanError, match=r"a gate generator needs a gate pipeline"):
        build(dataclasses.replace(_gates(), batch=BatchSpec(E, DEV, 384)), seeds=1)
    with pytest.raises(PlanError, match=r"the first stage of a gate pipeline must be the "
                                        r"generator"):
        build(_gates().remove("generator"), seeds=1)


def test_facade_spec_config_and_build_agree():
    cfg = GateGenConfig(num_envs=E, gate_width=0.05)
    a = GateGenerator(cfg, _rng())
    b = GateGenerator(legacy.to_gate_pipeline_spec(cfg), _rng())
    c = build(legacy.to_gate_pipeline_spec(cfg), seeds=_rng())
    assert _bytes(a.generate()) == _bytes(b.generate()) == _bytes(c.run(sync=True))
    rebuilt = GateGenerator(a.config, _rng())
    assert _bytes(rebuilt.generate()) == _bytes(a.generate())
    assert b.config is b._user_config and isinstance(a.config, GateGenConfig)
    with pytest.raises(ValueError, match="pass rng= or seeds=, not both"):
        GateGenerator(cfg, _rng(), seeds=3)
    owned = GateGenerator(cfg)
    assert owned.seeds.seeds.numpy().tolist() == list(range(E))
    assert _bytes(owned.generate()) == _bytes(GateGenerator(cfg, seeds=0).generate())
    with pytest.raises(ValueError, match=r"rng has 2 seeds but spec.batch.num_envs=3"):
        GateGenerator(_gates(), _rng(n=2))


def test_the_facade_config_is_always_a_gate_config():
    from tests._plugin_toys import TOY_RING

    registry.add(TOY_RING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_gen = GateGenerator(_legacy_spec(), _rng())
    for gen in (GateGenerator(_gates(min_gates=5), _rng()),
                GateGenerator(_gates(generator=gt.SUPERFORMULA), _rng()), legacy_gen):
        assert isinstance(gen._config, GateGenConfig)
        assert gen._config.max_gates == gen.max_gates
    assert GateGenerator(_gates(min_gates=5), _rng())._config.min_gates == 5
    sf = GateGenerator(_gates(generator=gt.SUPERFORMULA), _rng())
    assert sf._config.generator == gt.SUPERFORMULA and sf._config.gate_ordering == "raw"
    assert type(sf._generator_config).__name__ == "TrackGenConfig"
    assert legacy_gen._config.gate_ordering == "raw"
    assert "order" not in legacy_gen.program.spec.ids


def test_a_legacy_generator_in_a_spec_keeps_its_capacity_checks():
    from tests._plugin_toys import TOY_RING

    registry.add(TOY_RING)
    spec = _legacy_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        small = dataclasses.replace(spec, batch=dataclasses.replace(spec.batch,
                                                                    max_gates=6))
        with pytest.raises(PlanError, match=r"BatchSpec.max_gates=6 is too small for gate "
                                            r"generator 'toy.ring'; required max_gates 8."):
            build(small, seeds=1)
        many = spec.replace("validity", StageRef("validity", "gate_default",
                                                 params={"min_gates": 9}))
        many = dataclasses.replace(many, batch=dataclasses.replace(many.batch,
                                                                   max_gates=12))
        with pytest.raises(PlanError, match=r"min_gates=9 is too large for gate generator "
                                            r"'toy.ring'; generator can produce at most 8"):
            build(many, seeds=1)


def _legacy_spec():
    return PipelineSpec(BatchSpec(E, DEV, 384, max_gates=8), (
        StageRef("gate_generator", "toy.ring", id="generator"),
        StageRef("gate_relax", "spheres"), StageRef("elevation", "flat"),
        StageRef("lift", "gates"), StageRef("tangents", "central"),
        StageRef("gate_frames", "default"), StageRef("validity", "gate_default")))


# ---------------------------------------------------------------------------------------
# One default-order rule; the preset's own arguments reach every stage
# ---------------------------------------------------------------------------------------

def test_a_provider_without_a_preferred_order_gets_its_only_order():
    class NoPref(gt.Helix):
        name = "acme.helix_nopref"
        preferred = {}

    registry.add(NoPref)
    from track_gen.testing.conformance import _default_ordering

    spec = _gates(generator="acme.helix_nopref")
    assert _ref(spec, "order").name == "raw" == _default_ordering("acme.helix_nopref")
    assert GateGenerator(spec, _rng()).generate().valid.numpy().any()
    assert _ref(_gates(generator=gt.SCATTER), "order").name == "ccw"      # its preferred


def test_validity_params_in_checks_keep_the_preset_arguments():
    spec = _gates(min_gates=6, gate_radius=0.03, gate_width=0.05,
                  checks=(("gate_default", {"z_valid_grade": 0.5}),))
    assert dict(_ref(spec, "validity").params) == {
        "min_gates": 6, "gate_radius": 0.03, "gate_width": 0.05, "z_valid_grade": 0.5}
    assert _resolved(spec) == _resolved(_gates(min_gates=6, gate_radius=0.03,
                                               gate_width=0.05, z_valid_grade=0.5))
    # the check's own min_gates wins; a posed generator gets check_crossing="always"
    assert _ref(_gates(checks=(("gate_default", {"min_gates": 8}),)),
                "validity").params["min_gates"] == 8
    posed = _gates(generator=gt.HELIX_POSED, checks=(("gate_default", None),))
    assert _ref(posed, "validity").params["check_crossing"] == "always"
    for field, v in (("gate_radius", 0.05), ("gate_width", 0.1)):
        with pytest.raises(ValueError, match=rf"checks: validity.gate_default sets "
                                             rf"{field}={v!r} but {field}="):
            _gates(checks=(("gate_default", {field: v}),))
    with pytest.raises(ValueError, match=r"validity.gate_default sets gate_radius=0.05"):
        _gates(checks=(StageRef("validity", "gate_default",
                                params=sp.GateValidityParams(gate_radius=0.05)),))
    # A half_size provider: a gate_width in the check is the provider error (passing it
    # to presets.gates would be refused too), not "pass gate_width=... to presets.gates".
    for params in ({"gate_width": 0.1}, sp.GateValidityParams(gate_width=0.1)):
        with pytest.raises(ValueError, match=r"checks: gate generator 'acme.helix_posed' "
                                             r"provides per-gate half_size; "
                                             r"validity.gate_default's gate_width must "
                                             r"stay 0, got 0.1$"):
            _gates(generator=gt.HELIX_POSED,
                   checks=(StageRef("validity", "gate_default", params=params),))
    with pytest.raises(ValueError, match=r"validity.gate_default sets gate_radius=0.05 .*"
                                         r"pass gate_radius=0.05 to presets.gates"):
        _gates(generator=gt.HELIX_POSED, checks=(("gate_default", {"gate_radius": 0.05}),))


def test_z_base():
    spec = _gates(z_base=1.5)
    assert dict(_ref(spec, "elevation").params) == {"z_base": 1.5}
    z = GateGenerator(spec, _rng()).generate().position.numpy()[:, 2]
    assert set(z[np.isfinite(z)].tolist()) == {1.5}
    assert _ref(_gates(z_base=1.5, elevation=("flat", {"z_base": 1.5})),
                "elevation").params["z_base"] == 1.5
    with pytest.raises(ValueError, match=r"z_base=0.5 is the flat elevation's altitude "
                                         r"\(elevation='flat'\); elevation.noise takes its "
                                         r"own"):
        _gates(z_base=0.5, elevation="noise")
    with pytest.raises(ValueError, match=r"presets.gates\(z_base=0.5\): gate generator "
                                         r"'acme.helix' provides its own altitudes"):
        _gates(generator=gt.HELIX, z_base=0.5)
    with pytest.raises(ValueError, match=r"elevation.flat sets z_base=2.0 but z_base=0.5"):
        _gates(z_base=0.5, elevation=("flat", {"z_base": 2.0}))
    # on a track, elevation.flat's z_base is refused by the preset and, before
    # allocating, by the planner
    with pytest.raises(ValueError, match=r"elevation.flat sets z_base=1.5, which sets gate "
                                         r"altitudes"):
        presets.track(num_envs=E, device=DEV, elevation=("flat", {"z_base": 1.5}))
    track = presets.track(num_envs=E, device=DEV).replace(
        "elevation", StageRef("elevation", "flat", params={"z_base": 1.5}))
    with pytest.raises(PlanError, match=r"stage 'elevation' \(elevation.flat\): "
                                        r"z_base=1.5 sets gate altitudes"):
        build(track, seeds=1)


def test_centerline_arguments_need_a_source_that_reads_them():
    for kw in (dict(relax="acme.nope"), dict(n_max=3), dict(half_width=99.0),
               dict(num_points=64), dict(spacing=0.1), dict(relax=None)):
        with pytest.raises(ValueError, match=r"configures the centerline sub-pipeline of "
                                             r"a stations source; source='anchors' \(gate "
                                             r"generator 'polar'\) does not run one"):
            _gates(generator="polar", **kw)
    for kw in (dict(relax="none"), dict(half_width=0.2), dict(spacing=0.1)):
        with pytest.raises(ValueError, match=r"configures the relaxed_stations "
                                             r"sub-pipeline .*source='stations' places the "
                                             r"stations on the raw loop"):
            _gates(generator=gt.SUPERFORMULA, **kw)
    _gates(generator=gt.SUPERFORMULA, num_points=128, n_max=300)          # read
    rel = _gates(generator=gt.SUPERFORMULA, source="relaxed_stations", relax=None,
                 half_width=0.2, spacing=0.1)
    assert _ref(rel, "relax").name == "none"
    _gates(generator="polar", n_max=384, relax="xpbd", half_width=0.1)    # the defaults


def test_station_count_is_set_once():
    with pytest.raises(ValueError, match=r"stations: stations.raw sets gates=20 but "
                                         r"gates=10; set the station count once"):
        _gates(generator=gt.SUPERFORMULA, gates=10, stations=("raw", {"gates": 20}))
    spec = _gates(generator=gt.SUPERFORMULA, gates=10, stations="raw")
    assert dict(_ref(spec, "stations").params) == {"gates": 10}
    assert spec.batch.max_gates == 10
    st = _gates(generator=gt.SUPERFORMULA, stations=("raw", {"gates": 20}))
    assert st.batch.max_gates == 20
    with pytest.raises(PlanError, match=r"min_gates=4 is too large for stations stage "
                                        r"'raw'; it writes at most 3 anchors per env"):
        build(_gates(generator=gt.SUPERFORMULA, gates=3), seeds=1)


@pytest.mark.parametrize("source", ["stations", "relaxed_stations"])
def test_a_legacy_generator_spec_through_stations(source):
    from tests._plugin_toys import register_toys

    register_toys(registry)
    seq = GateGenerator(_gates(generator="toy.circle", gates=8, source=source),
                        _rng()).generate()
    assert (seq.count.numpy() == 8).all() and (seq.valid.numpy() == 1).all()


def test_planner_hints_for_gate_pipeline_mistakes():
    with pytest.raises(PlanError, match=r"validity.default is a track check; the gate "
                                        r"pipeline's built-in check is "
                                        r"validity.gate_default"):
        build(_gates(checks=("gate_default", "default")), seeds=1)
    st = _gates(generator="bezier", source="stations")
    no_st = dataclasses.replace(st, stages=tuple(r for r in st.stages
                                                 if r.kind != "stations"))
    with pytest.raises(PlanError, match=r"reads 'anchors.xy' .*generator 'bezier' is a "
                                        r"centerline \(track\) generator: insert "
                                        r"StageRef\('stations', 'raw'"):
        build(no_st, seeds=1)
    track = presets.track(num_envs=E, device=DEV)
    with pytest.raises(PlanError, match=r"or set BatchSpec.max_gates=None for a track "
                                        r"pipeline"):
        build(dataclasses.replace(track, batch=dataclasses.replace(track.batch,
                                                                   max_gates=16)), seeds=1)


def test_spec_info_reports_the_supported_orderings():
    assert registry.describe("gate_generator", "bezier").supported_orderings == \
        ("acme.reverse", "ccw", "random_pairs")
    assert registry.describe("gate_generator", "polar").supported_orderings == \
        ("acme.reverse", "ccw", "raw")
    assert registry.describe("gate_generator", gt.HELIX).supported_orderings == ("raw",)
    assert registry.describe("generator", "bezier").supported_orderings == ()
    from tests._plugin_toys import TOY_RING

    registry.add(TOY_RING)
    assert registry.describe("gate_generator", "toy.ring").supported_orderings == ("raw",)


def test_gate_and_track_specs_pickle():
    import pickle

    for spec in (_gates(), _gates(generator=gt.SUPERFORMULA, source="relaxed_stations"),
                 _gates(checks=(("gate_default", {"min_gates": 5}),)),
                 presets.track(num_envs=E, device=DEV)):
        assert pickle.loads(pickle.dumps(spec)) == spec
    resolved = build(_gates(), seeds=1).spec
    assert pickle.loads(pickle.dumps(resolved)) == resolved
    # A nested frozen mapping (another ref's params) pickles too, as a dict.
    inner = StageRef("relax", "xpbd", params={"a": 1})
    ref = StageRef("relax", "xpbd", params={"sub": inner.params,
                                            "seq": [inner.params, (inner.params,)]})
    back = pickle.loads(pickle.dumps(ref))
    assert back == ref and type(back.params["sub"]) is dict
    assert type(back.params["seq"][1][0]) is dict
