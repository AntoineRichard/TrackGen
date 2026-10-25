"""Third-party pipeline stages end to end on CPU (roadmap PR10b).

The toy plugins of ``tests/_stage_toys.py`` (relax ``acme.taubin``, elevation
``acme.sinusoid_hills``, validity ``acme.max_curvature``) are installed through a fake
``track_gen.plugins`` entry point (the registry's real discovery path) and selected
through ``track_gen.pipeline.presets.track`` and, where one exists, the flat
``TrackGenConfig`` shortcut. Covered: registration and ``describe``, byte equality of
the two selections, the rewrite model (the version a relax stage read stays intact),
dynamic params, the hills formula and the curvature bit against independent host
evaluations, ``check_stage`` on every toy, the broken toys failing exactly their check,
and quarantine. The CUDA twin (caller capture) is ``tests/test_stages_cuda.py``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

from tests._broken_stage_toys import BROKEN, register_broken_stages
from tests._plugin_toys import FakeDist, FakeEntryPoint, install_entry_points
from tests._stage_toys import (ACME, HILLS, KMAX, TAUBIN, HillsParams, KappaParams,
                               TaubinParams, acme_entry_point)
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator, plugin
from track_gen.ops import hash32
from track_gen.pipeline import PlanError, StageRef, build, presets
from track_gen.plugin import POLICY_ENV, PluginLoadError, registry
from track_gen.plugin.stage import CaptureClass
from track_gen.testing import CHECKS, check_stage

DEV = "cpu"
E = 4
FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length", "valid",
          "count", "winding", "truncated", "generation", "fail_mask")


@pytest.fixture(autouse=True)
def acme(monkeypatch):
    """The ``acme-stages`` distribution through a fake entry point, in a fresh registry
    scope (discovery re-runs; restored afterwards)."""
    monkeypatch.delenv(POLICY_ENV, raising=False)
    install_entry_points(monkeypatch, acme_entry_point())
    with registry.scoped(fresh=True):
        yield registry


def _rng(seed=11, n=E):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _gen(spec_or_cfg, seed=11):
    n = spec_or_cfg.batch.num_envs if hasattr(spec_or_cfg, "batch") else \
        spec_or_cfg.num_envs
    return TrackGenerator(spec_or_cfg, _rng(seed, n))


def _spec(**kw):
    return presets.track(num_envs=E, device=DEV, **kw)


def _bytes(track) -> dict:
    return {f: getattr(track, f).numpy().tobytes() for f in FIELDS}


# ---------------------------------------------------------------------------------------
# Registration, describe
# ---------------------------------------------------------------------------------------

def test_the_toys_are_discovered_and_described():
    assert TAUBIN in registry.available("relax")
    assert HILLS in registry.available("elevation")
    assert KMAX in registry.available("validity")
    info = registry.describe("relax", TAUBIN)
    assert info.distribution == (ACME.name, ACME.version) and info.entry_point == "acme"
    assert not info.builtin and info.origin == "tests._stage_toys"
    assert info.capture is CaptureClass.STATIC and info.capturable and info.output is None
    assert dict(info.reads) == {"xy": "cl.xy", "count": "cl.count"}
    assert dict(info.writes) == {"xy": "cl.xy"}
    props = info.params_schema["properties"]
    assert props["passes"]["x-static"] is True and props["lam"]["x-dynamic"] is True
    k = registry.describe("validity", KMAX)
    assert dict(k.writes) == {"fail": "valid.fail"} and "fail" in k.reads
    assert registry.describe("validity", "default").builtin
    assert dict(registry.describe("validity", "default").writes) == {
        "valid": "track.valid", "fail": "valid.fail"}


def test_scoped_restores_the_stage_kinds():
    before = {k: registry.available(k) for k in ("relax", "width", "elevation", "validity")}
    with registry.scoped():
        registry.unregister("relax", TAUBIN)
        registry.unregister("validity", "default")
        assert TAUBIN not in registry.available("relax")
    assert {k: registry.available(k) for k in before} == before


# ---------------------------------------------------------------------------------------
# relax: acme.taubin
# ---------------------------------------------------------------------------------------

def test_taubin_through_presets_and_the_flat_shortcut_is_byte_equal():
    a = _gen(_spec(relax=TAUBIN))
    b = _gen(TrackGenConfig(device=DEV, num_envs=E, relax_solver=TAUBIN))
    assert _bytes(a._track) == _bytes(b._track)
    assert b.program.spec.stages[3].params == TaubinParams()
    assert (b.program.spec.stages[3].kind, b.program.spec.stages[3].name) == \
        ("relax", TAUBIN)


def test_the_flat_shortcut_gives_a_plugin_half_width_field_the_config_value():
    """A plugin relax / elevation whose Params has a ``half_width`` field and keeps the
    default ``from_legacy``: ``TrackGenConfig(half_width=...)`` sets it (the planner
    requires every stage with the field to agree; the user has no other knob)."""
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class SepParams(plugin.Params):
        half_width: float = plugin.static(0.1, doc="separation target")

    class Sep(plugin.Stage):
        kind, name, Params = "relax", "acme.sep", SepParams
        reads = {"xy": "cl.xy"}
        writes = {"xy": "cl.xy"}

        def build(self, b, p):
            x, out = b.read("xy"), b.write("xy", wp.vec2f, per="cl")
            return lambda ctx: ctx.copy(out, x)

    registry.add(Sep)
    g = _gen(TrackGenConfig(device=DEV, num_envs=E, relax_solver="acme.sep", half_width=0.2))
    assert g.program.spec.stages[3].params == SepParams(half_width=0.2)


def test_taubin_is_a_rewrite_that_leaves_the_version_it_read_intact():
    g = _gen(_spec(relax=TAUBIN))
    none = _gen(_spec(relax=None))
    tap = g.program.tap
    assert g.program.channels["cl.xy"] == ("resample", "relax", "uniform")
    before, after = tap("cl.xy", after="resample"), tap("cl.xy", after="relax")
    assert before is not after
    assert before.numpy().tobytes() == \
        none.program.tap("cl.xy", after="resample").numpy().tobytes()
    assert not np.array_equal(before.numpy(), after.numpy(), equal_nan=True)
    # NaN padding carried, real points smoothed
    count = tap("cl.count").numpy()
    xy = after.numpy().reshape(E, -1, 2)
    for e in range(E):
        assert np.isfinite(xy[e, :count[e]]).all() and np.isnan(xy[e, count[e]:]).all()
    # the ping-pong partner is the stage's own scratch
    assert set(g.program._scratch["relax"]) == {"xy.tmp0"}


def test_taubin_dynamic_lam_is_live():
    g = _gen(_spec(relax=TAUBIN))
    assert list(g.params) == ["generator", "relax"]
    live = g.params["relax"]["lam"]
    ptr = live.ptr
    base = g.program.tap("cl.xy", after="relax").numpy().copy()
    g.params["relax"].set("lam", 0.9)
    g.generate()
    moved = g.program.tap("cl.xy", after="relax").numpy()
    assert not np.array_equal(base, moved, equal_nan=True)
    twin = _gen(_spec(relax=(TAUBIN, {"lam": 0.9})))
    assert _bytes(g._track) == _bytes(twin._track)
    assert g.params["relax"]["lam"].ptr == ptr
    with pytest.raises(KeyError, match="static"):
        g.params["relax"].set("passes", 3)


# ---------------------------------------------------------------------------------------
# elevation: acme.sinusoid_hills
# ---------------------------------------------------------------------------------------

@wp.kernel
def _phase_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32,
             out: wp.array(dtype=wp.float32)):
    e = wp.tid()
    st = wp.rand_init(wp.int32(hash32(wp.uint32(seed[e]) ^ salt) & wp.uint32(0x7FFFFFFF)))
    out[e] = wp.randf(st)


def test_hills_through_presets_and_the_flat_shortcut_is_byte_equal():
    a = _gen(_spec(elevation=HILLS))
    b = _gen(TrackGenConfig(device=DEV, num_envs=E, z_profile=HILLS))
    assert _bytes(a._track) == _bytes(b._track)
    ids = b.program.spec.ids
    assert b.program._kinds["lift"] == ("lift", "elevated") and ids[-1] == "lift"
    c = _gen(TrackGenConfig(device=DEV, num_envs=E, z_profile=HILLS, z_base=0.3))
    assert c.program.spec.stages[7].params == HillsParams(base=0.3)   # from_legacy


def test_hills_altitude_matches_a_host_evaluation():
    g = _gen(_spec(elevation=(HILLS, {"waves": 2})))
    g.params["elevation"].set("amplitude", 0.7)
    track = g.generate()
    tap = g.program.tap
    cum = tap("cl.arclen2").numpy().reshape(E, -1).astype(np.float64)
    perim = tap("cl.length2").numpy().astype(np.float64)
    count = tap("cl.count").numpy()
    salt = plugin.stage._salt_of("elevation", "phase")
    u = wp.empty(E, dtype=wp.float32, device=DEV)
    wp.launch(_phase_k, E, [g.program.seed, wp.uint32(salt), u], device=DEV)
    phase = 2.0 * np.pi * u.numpy().astype(np.float64)
    z = track.center.numpy().reshape(E, -1, 3)[..., 2]
    zc = tap("cl.z").numpy().reshape(E, -1)
    for e in range(E):
        n = count[e]
        want = 0.7 * np.sin(2.0 * np.pi * 2.0 * cum[e, :n] / max(perim[e], 1e-9) + phase[e])
        np.testing.assert_allclose(z[e, :n], want, atol=2e-5)
        assert (zc[e, n:] == 0.0).all()                      # the padding
    assert np.ptp(z[0, :count[0]]) > 0.5


# ---------------------------------------------------------------------------------------
# validity: acme.max_curvature
# ---------------------------------------------------------------------------------------

def _kmax(g) -> np.ndarray:
    tap = g.program.tap
    kappa = tap("cl.kappa").numpy().reshape(E, -1)
    count = tap("cl.count").numpy()
    return np.array([kappa[e, :count[e]].max() if count[e] else -np.inf
                     for e in range(E)], np.float32)


def test_max_curvature_sets_its_bit_exactly_where_kappa_exceeds_the_threshold():
    probe = _gen(_spec(checks=("default", KMAX)))
    km = np.sort(_kmax(probe))
    thr = float(0.5 * (km[1] + km[2]))
    g = _gen(_spec(checks=("default", (KMAX, {"kappa_max": thr}))))
    bits = g.program.fail_bits
    assert bits[KMAX] == 16 and dict(bits) == {
        "generator": 0, "turning": 1, "finite": 2, "thickness": 3, "w_floor": 4,
        "border": 5, "grade": 6, "truncation": 7, KMAX: 16}
    assert g._track.fail_bits == bits
    fm = g._track.fail_mask.numpy()
    own = (fm >> 16) & 1
    np.testing.assert_array_equal(own, (_kmax(g) > np.float32(thr)).astype(own.dtype))
    assert own.any() and not own.all()
    np.testing.assert_array_equal(g._track.valid.numpy(), (fm == 0).astype(np.int32))
    # the built-in bits are untouched (equal to the pipeline without the check)
    plain = _gen(_spec())
    np.testing.assert_array_equal(fm & 0xFFFF, plain._track.fail_mask.numpy())
    # validity.default's own flag is still the 0/1 of its checks (a tap), the Track's is
    # derived by the commit
    assert g.program.tap("track.valid") is not g._track.valid
    np.testing.assert_array_equal(g.program.tap("track.valid").numpy(),
                                  plain._track.valid.numpy())


def test_max_curvature_dynamic_threshold_moves_the_bit():
    g = _gen(_spec(checks=("default", KMAX)))
    assert list(g.params) == ["generator", KMAX]
    assert not ((g._track.fail_mask.numpy() >> 16) & 1).any()      # default 50: none
    g.params[KMAX].set("kappa_max", 0.0)
    g.generate()
    fm = g._track.fail_mask.numpy()
    count = g._track.count.numpy()
    np.testing.assert_array_equal((fm >> 16) & 1, (count > 0).astype(np.int32))
    assert (g._track.valid.numpy() == 0).all()


def test_two_checks_get_bits_in_pipeline_order():
    spec = _spec(checks=("default", StageRef("validity", KMAX, id="k1"),
                         StageRef("validity", KMAX, id="k2")))
    p = build(spec, seeds=3)
    assert [(n, b) for n, b in p.fail_bits.items() if b >= 16] == [("k1", 16), ("k2", 17)]
    assert p._fail_owner["k1"] == "k1" and p._fail_owner["truncation"] == "validity"


# ---------------------------------------------------------------------------------------
# All three together, poison
# ---------------------------------------------------------------------------------------

def test_all_three_toys_in_one_program():
    spec = _spec(relax=TAUBIN, elevation=HILLS, checks=("default", KMAX))
    g = _gen(spec)
    assert list(g.params) == ["generator", "relax", "elevation", KMAX]
    assert g.program.spec.ids == ("generator", "resample", "width", "relax", "uniform",
                                  "frames", "offset", "elevation", "validity", KMAX,
                                  "lift")
    assert g.program.capturable and g.program.capture_class is CaptureClass.STATIC
    np.testing.assert_array_equal(g._track.valid.numpy(),
                                  (g._track.fail_mask.numpy() == 0).astype(np.int32))
    again = _gen(spec)
    assert _bytes(g._track) == _bytes(again._track)


# ---------------------------------------------------------------------------------------
# check_stage
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind,name", [("relax", TAUBIN), ("elevation", HILLS),
                                       ("validity", KMAX)])
def test_each_toy_passes_check_stage(kind, name):
    report = check_stage(kind, name, devices=DEV, num_envs=4)
    report.raise_for_failures()
    names = [r.name for r in report]
    assert names == [n for n in CHECKS[kind]
                     if not n.startswith("capture.") and n != "bounded.lowering"]
    # The BOUNDED / HOST tier checks are "not applicable" to these STATIC stages.
    tiers = ("bounded.absorbing", "host.require_caller_capturable")
    assert all(not r.skipped for r in report
               if r.name != "output.accepted" and r.name not in tiers)
    assert all(report.result(n).skipped for n in tiers)


def test_check_stage_takes_params_and_upstream():
    report = check_stage("validity", KMAX, {"kappa_max": 0.0}, devices=DEV, num_envs=4,
                         upstream={"generator": "polar", "elevation": HILLS})
    report.raise_for_failures()
    assert report.result("validity.bits").details["own_fraction"] == 1.0
    with pytest.raises(ValueError, match=r"upstream= must not set \['relax'\]"):
        check_stage("relax", TAUBIN, upstream={"relax": "xpbd"})
    with pytest.raises(ValueError, match="kind must be one of"):
        check_stage("generator", "bezier")
    with pytest.raises(ValueError, match="unknown relax 'acme.nope'"):
        check_stage("relax", "acme.nope")


@pytest.mark.parametrize("key,kind,failing", [
    ("relax", "relax", {"stage.isolation"}),
    ("stateful", "elevation", {"stage.isolation", "statelessness"}),
    ("validity", "validity", {"validity.bits"}),
    ("drops_bits", "validity", {"validity.bits"}),
    # output.view (roadmap PR13b): the unwritten envs keep what the CALLER's buffer
    # held (a poisoned out= view), not the standalone batch's.
    ("failures_only", "validity", {"validity.bits", "stage.isolation", "statelessness",
                                   "output.view"}),
    ("padding", "elevation", {"stage.isolation", "statelessness"}),
    ("alloc", "elevation", {"hot_path.allocation"}),
])
def test_a_broken_stage_fails_the_named_check(key, kind, failing):
    register_broken_stages(registry)
    report = check_stage(kind, BROKEN[key], devices=DEV,
                         num_envs=8 if key == "failures_only" else 4)
    assert set(report.failed_names) == failing, str(report)
    if key == "relax":
        assert "wrote its read port 'xy'" in report.result("stage.isolation").message
    if key == "stateful":
        assert "its own previous output" in report.result("stage.isolation").message
    if key == "validity":
        assert "bits it does not own" in report.result("validity.bits").message
    if key == "drops_bits":
        assert "dropped or changed the bits of earlier checks" in \
            report.result("validity.bits").message
    if key == "failures_only":
        assert "left valid.fail unwritten" in report.result("validity.bits").message
    if key == "padding":
        assert "slot(s) of 'cl.z' (port 'z') unwritten" in \
            report.result("stage.isolation").message


# ---------------------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------------------

class _BadRelax(plugin.Stage):
    kind = "relax"
    name = "bad.relax"
    reads = {"xy": "cl.xy"}
    writes = {"k": "cl.kappa"}               # a relax stage may not write cl.kappa

    def build(self, b, p):
        return lambda ctx: None


def _register_bad(reg):
    reg.add(_BadRelax)


def test_a_plugin_registering_an_invalid_stage_is_quarantined(monkeypatch):
    install_entry_points(monkeypatch, acme_entry_point(),
                         FakeEntryPoint("bad", _register_bad, FakeDist("bad-stages")))
    with registry.scoped(fresh=True), pytest.warns(RuntimeWarning, match="quarantined"):
        assert "bad" in registry.quarantined
        assert "writes 'cl.kappa'" in registry.quarantined["bad"]
        assert TAUBIN in registry.available("relax")            # acme still loads
        with pytest.raises(PluginLoadError, match="quarantined"):
            build(_spec(relax="bad.relax"))
        with pytest.raises(PluginLoadError, match="quarantined"):
            TrackGenerator(TrackGenConfig(device=DEV, num_envs=E, relax_solver="bad.relax"),
                           _rng())
        with pytest.raises(PlanError, match=r"unknown relax 'nope'.*quarantined plugins"):
            build(_spec(relax="nope"))


def test_unknown_names_in_presets_fail_at_build():
    spec = _spec(elevation="acme.nope")                      # presets never resolve
    with pytest.raises(PlanError, match=r"stage 'elevation' \(elevation.acme.nope\): "
                                        r"unknown elevation 'acme.nope'"):
        build(spec)
    for kp in (KappaParams, TaubinParams, HillsParams):
        kp()                                                   # every field has a default
