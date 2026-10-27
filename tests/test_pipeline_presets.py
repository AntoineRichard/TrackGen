"""``track_gen.pipeline.presets.track`` and ``TrackGenerator(PipelineSpec)`` (roadmap PR10b,
CPU).

The preset is the default track pipeline as data, one argument per replaceable stage;
``presets.track(num_envs=E, device=d)`` builds byte-identically to
``TrackGenerator(TrackGenConfig(num_envs=E, device=d))``. ``TrackGenerator`` runs any
``PipelineSpec`` and takes its seeds as ``rng=`` or ``seeds=``.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import warp as wp

from tests._stage_toys import HILLS, KMAX, TAUBIN, register
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen.pipeline import (BatchSpec, PipelineSpec, PlanError, StageRef, build,
                                legacy, presets)
from track_gen.plugin import registry
from track_gen.seeds import SeedState

DEV = "cpu"
E = 3
FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length", "valid",
          "count", "winding", "truncated", "generation", "fail_mask")
# Since 0.2.0.dev4 (roadmap PR13a) the width runs before relax.
IDS = ("generator", "resample", "width", "relax", "uniform", "frames", "offset",
       "elevation", "validity", "lift")


@pytest.fixture(autouse=True)
def toys():
    register(registry)


def _rng(seed=7, n=E):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _spec(**kw):
    return presets.track(num_envs=E, device=DEV, **kw)


def _bytes(track) -> dict:
    return {f: getattr(track, f).numpy().tobytes() for f in FIELDS}


def _resolved(spec) -> list:
    return [(r.kind, r.name, r.slot_id, r.params) for r in build(spec, seeds=1).spec.stages]


# ---------------------------------------------------------------------------------------
# The preset is the default pipeline
# ---------------------------------------------------------------------------------------

_NOISE = dict(z_base=0.1, z_min=-0.3, z_max=0.4, z_noise_amplitude=0.2,
              z_noise_harmonics=4)


@pytest.mark.parametrize("kw,cfg", [
    ({}, {}),
    (dict(relax=None), dict(relax_enable=False)),
    (dict(elevation=("noise", _NOISE)), dict(z_profile="noise", **_NOISE)),
    (dict(elevation=("uniform", {"z_max": 0.2})), dict(z_profile="uniform", z_max=0.2)),
    (dict(checks=(("default", {"validity_border_check": True}),)),
     dict(validity_border_check=True)),
    (dict(generator="polar", half_width=0.08), dict(generator="polar", half_width=0.08)),
    (dict(n_max=64, spacing=0.06), dict(N_max=64, spacing=0.06)),
    (dict(z_base=0.5), dict(z_base=0.5)),
])
def test_the_preset_builds_byte_identical_to_the_config(kw, cfg):
    spec = _spec(**kw)
    ref = legacy.to_pipeline_spec(TrackGenConfig(device=DEV, num_envs=E, **cfg))
    assert spec.ids == ref.ids == IDS
    assert _resolved(spec) == _resolved(ref)
    a = TrackGenerator(spec, _rng())
    b = TrackGenerator(TrackGenConfig(device=DEV, num_envs=E, **cfg), _rng())
    assert _bytes(a._track) == _bytes(b._track)


def test_the_default_stage_list():
    spec = _spec()
    assert [(r.kind, r.name) for r in spec.stages] == [
        ("generator", "bezier"), ("resample", "constant_spacing"), ("width", "constant"),
        ("relax", "xpbd"), ("resample", "uniform"), ("frames", "default"),
        ("offset", "default"), ("elevation", "flat"), ("validity", "default"),
        ("lift", "flat")]
    assert spec.batch == BatchSpec(num_envs=E, device=DEV, n_max=384, closed=True,
                                   scale=1.0, on_invalid="poison")
    assert dict(spec.stages[0].capacity) == {"raw": 256}
    assert dict(spec.stages[1].params) == {"spacing": pytest.approx(0.06)}
    assert spec.stages[9].params == {"z_base": 0.0}
    with pytest.raises(ValueError, match=r"z_base=0.5 is the flat lift's altitude"):
        _spec(elevation="noise", z_base=0.5)
    elevated = _spec(elevation="noise")
    assert (elevated.stages[9].kind, elevated.stages[9].name) == ("lift", "elevated")


def test_half_width_reaches_the_width_stage_only():
    # Since 0.2.0.dev4 cl.half_width is the one per-point half-width: half_width= fills
    # the width stage (and the default spacing); relax / offset / validity refuse it.
    spec = _spec(half_width=0.2)
    hw = {r.slot_id: r.params.get("half_width") for r in spec.stages
          if isinstance(r.params, dict) or hasattr(r.params, "get")}
    assert {k: v for k, v in hw.items() if v is not None} == {"width": 0.2}
    assert spec.stages[2].slot_id == "width"
    assert dict(spec.stages[1].params) == {"spacing": pytest.approx(0.12)}
    assert dict(_spec(half_width=0.2, spacing=0.05).stages[1].params) == {"spacing": 0.05}
    # a relax mapping is taken as is (no half_width field any more)
    assert _spec(half_width=0.2, relax=("xpbd", {"relax_iters": 3})).stages[3].params == \
        {"relax_iters": 3}
    for slot, ref in (("relax", StageRef("relax", "xpbd", params={"half_width": 0.2})),
                      ("offset", StageRef("offset", "default", params={"half_width": 0.2})),
                      ("validity", StageRef("validity", "default",
                                            params={"half_width": 0.2}))):
        with pytest.raises(PlanError, match=rf"stage '{slot}' \({ref.kind}.{ref.name}\): "
                                            rf"half_width is no longer a "
                                            rf"{ref.kind}.{ref.name} param \(0.2.0.dev4\)"):
            build(_spec().replace(slot, ref), seeds=1)
    # params setting another half_width: a preset-level hint, not the planner's error
    with pytest.raises(ValueError, match=r"width: width.constant sets half_width=0.2 but "
                                         r"half_width=0.1; pass half_width=0.2"):
        _spec(width=("constant", {"half_width": 0.2}))
    assert _spec(half_width=0.2, width=("constant", {"half_width": 0.2})).stages[2].params \
        == {"half_width": 0.2}


# ---------------------------------------------------------------------------------------
# Selection forms
# ---------------------------------------------------------------------------------------

def test_selection_forms():
    by_name = _spec(relax=TAUBIN)
    by_pair = _spec(relax=(TAUBIN, {"passes": 2}))
    by_instance = _spec(relax=(TAUBIN, registry.get("relax", TAUBIN).Params(passes=2)))
    by_ref = _spec(relax=StageRef("relax", TAUBIN, params={"passes": 2}))
    assert by_name.stages[3] == StageRef("relax", TAUBIN)
    assert by_pair.stages[3] == by_ref.stages[3] == StageRef("relax", TAUBIN,
                                                              params={"passes": 2})
    assert _resolved(by_pair)[3] == _resolved(by_instance)[3] == _resolved(by_ref)[3]
    assert _spec(relax=None).stages[3] == StageRef("relax", "none")
    assert _spec(elevation=None).stages[7] == StageRef("elevation", "flat")
    assert _spec(relax=StageRef("relax", TAUBIN, id="smooth")).stages[3].slot_id == "smooth"


def test_selection_errors():
    with pytest.raises(ValueError, match=r"relax=StageRef\('width', 'constant'\): "
                                         r"expected a StageRef of kind 'relax'"):
        _spec(relax=StageRef("width", "constant"))
    with pytest.raises(ValueError, match="width=None: a width stage must write "
                                         "cl.half_width"):
        _spec(width=None)
    with pytest.raises(TypeError, match="relax must be a stage name, a \\(name, params\\) "
                                        "pair, a StageRef or None; got int"):
        _spec(relax=3)
    with pytest.raises(ValueError, match="pass generator_params inside the StageRef"):
        _spec(generator=StageRef("generator", "polar"), generator_params={})
    with pytest.raises(ValueError, match="expected a StageRef of kind 'generator'"):
        _spec(generator=StageRef("relax", "xpbd"))
    with pytest.raises(ValueError, match=r"checks: duplicate stage id "
                                         r"'acme.max_curvature'; give one an explicit "
                                         r"StageRef\(id=...\)"):
        _spec(checks=("default", KMAX, KMAX))


def test_check_ids_and_generator_forms():
    spec = _spec(checks=("default", KMAX, StageRef("validity", KMAX, id="k2")))
    assert spec.ids[8:11] == ("validity", KMAX, "k2")
    for one in ("default", StageRef("validity", "default"), ("default", None)):
        with pytest.raises(TypeError, match="checks must be a sequence of selections"):
            _spec(checks=one)
    with pytest.raises(ValueError, match=r"checks must start with 'default' .*: "
                                         r"checks=\('default', 'acme.max_curvature'\)"):
        _spec(checks=(KMAX, "default"))
    with pytest.raises(ValueError, match=r"checks must start with 'default'"):
        _spec(checks=())
    g = _spec(generator=StageRef("generator", "polar", params={"polar_num_knots": 9}),
              num_points=128)
    assert g.stages[0] == StageRef("generator", "polar", params={"polar_num_knots": 9},
                                   capacity={"raw": 128})
    assert _spec(generator="hull", generator_params={"hull_displacement": 0.3}).stages[0] \
        == StageRef("generator", "hull", params={"hull_displacement": 0.3},
                    capacity={"raw": 256})


def test_edit_ops_on_a_preset():
    spec = _spec().insert_after("validity", StageRef("validity", KMAX, id="kmax"))
    assert spec.ids[-3:] == ("validity", "kmax", "lift")
    p = build(spec, seeds=1)
    assert p.fail_bits["kmax"] == 16
    swapped = _spec().replace("relax", StageRef("relax", TAUBIN))
    assert swapped.stages[3].slot_id == "relax" and swapped.stages[3].name == TAUBIN


def test_json_round_trip_with_plugin_stages_builds_byte_identical():
    spec = _spec(relax=(TAUBIN, {"passes": 3, "lam": 0.6}),
                 elevation=(HILLS, {"amplitude": 0.4, "waves": 2}),
                 checks=("default", (KMAX, {"kappa_max": 20.0})), on_invalid="poison")
    d = spec.to_dict()
    assert d["batch"]["on_invalid"] == "poison"
    back = PipelineSpec.from_dict(json.loads(json.dumps(d)))
    assert back.to_dict() == d and back.batch.on_invalid == "poison"
    a = TrackGenerator(spec, _rng())
    b = TrackGenerator(back, _rng())
    assert _bytes(a._track) == _bytes(b._track)
    old = dict(d, batch={k: v for k, v in d["batch"].items() if k != "on_invalid"})
    assert PipelineSpec.from_dict(old).batch.on_invalid == "poison"


# ---------------------------------------------------------------------------------------
# TrackGenerator(PipelineSpec)
# ---------------------------------------------------------------------------------------

def test_track_generator_runs_a_spec_like_build_and_like_the_config():
    spec = _spec()
    gen = TrackGenerator(spec, _rng())
    program = build(spec, seeds=_rng())
    cfg = TrackGenerator(TrackGenConfig(device=DEV, num_envs=E), _rng())
    assert _bytes(gen._track) == _bytes(program.result) == _bytes(cfg._track)
    assert gen.config is spec and gen.program is gen._program
    assert list(gen.params) == ["generator"]
    rebuilt = TrackGenerator(gen.config, _rng())
    assert _bytes(rebuilt._track) == _bytes(gen._track)
    assert gen._config.num_envs == E and gen._config.generator == "bezier"


def test_track_generator_seeds_forms():
    want = TrackGenerator(_spec(), _rng(5))._track.center.numpy()
    arr = wp.array(np.arange(5, 5 + E, dtype=np.int32), dtype=wp.int32, device=DEV)
    for seeds in (5, arr, _rng(5), SeedState(E, DEV, 5)):
        gen = TrackGenerator(_spec(), seeds=seeds)
        np.testing.assert_array_equal(gen._track.center.numpy(), want)
    owned = TrackGenerator(_spec())                                # seeds 0 .. E-1
    np.testing.assert_array_equal(owned.seeds.seeds.numpy(), np.arange(E))
    assert owned._rng is None and TrackGenerator(_spec(), _rng())._rng is not None
    np.testing.assert_array_equal(
        TrackGenerator(TrackGenConfig(device=DEV, num_envs=E), seeds=5)._track.center
        .numpy(), want)


def test_track_generator_argument_errors():
    with pytest.raises(ValueError, match="pass rng= or seeds=, not both"):
        TrackGenerator(_spec(), _rng(), seeds=3)
    with pytest.raises(TypeError, match="TrackGenerator expects a TrackGenConfig or a "
                                        "track_gen.pipeline.PipelineSpec, got dict"):
        TrackGenerator(_spec().to_dict())
    with pytest.raises(ValueError, match="rng has 2 seeds but spec.batch.num_envs=3"):
        TrackGenerator(_spec(), _rng(n=2))
    with pytest.raises(ValueError, match="rng has 2 seeds but config.num_envs=3"):
        TrackGenerator(TrackGenConfig(device=DEV, num_envs=3), _rng(n=2))
    with pytest.raises(PlanError, match="the first stage must be the generator"):
        TrackGenerator(_spec().remove("generator"))


def test_track_generator_params_hold_the_stages_with_dynamic_fields():
    gen = TrackGenerator(_spec(relax=TAUBIN, elevation=HILLS, checks=("default", KMAX)),
                         _rng())
    assert list(gen.params) == ["generator", "relax", "elevation", KMAX]
    assert gen.params["relax"] is gen.program.params["relax"]
    assert set(gen.params["relax"]) == {"lam", "mu"}
    assert "frames" in gen.program.params and "frames" not in gen.params


def test_a_config_with_a_plugin_profile_uses_the_elevated_lift():
    gen = TrackGenerator(TrackGenConfig(device=DEV, num_envs=E, z_profile=HILLS), _rng())
    stages = gen.program.spec.stages
    assert (stages[7].kind, stages[7].name) == ("elevation", HILLS)
    assert (stages[9].kind, stages[9].name) == ("lift", "elevated")
    assert dataclasses.asdict(stages[7].params) == {"amplitude": 0.2, "waves": 3,
                                                    "base": 0.0}
