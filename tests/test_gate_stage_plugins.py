"""Gate-pipeline plugins (roadmap PR11, CPU): the toys of ``tests/_gate_toys.py`` through
the public surface only.

- an anchors generator that PROVIDES z / tangent (``acme.helix``) and half_size / roll
  (``acme.helix_posed``): the provided values reach the GateSequence (z and half_size
  bit for bit, the tangent normalized exactly as the frames stage does, the roll applied
  to the frame);
- a third-party order (``acme.reverse``) on a stations source, on a plain cloud
  (``acme.scatter``) and on a built-in generator, through the CONFIG path;
- a third-party TRACK generator (``acme.superformula``) through ``stations`` and
  ``relaxed_stations``;
- a gate validity check with a dynamic field and its own fail bit (``acme.max_height``);
- the conformance kit on the toys, the registration rules of gate generator Stages,
  quarantine, and gate Stage registration.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
import warp as wp

from tests import _gate_toys as gt
from tests._plugin_toys import (TOY_RING, FakeDist, FakeEntryPoint, install_entry_points,
                                register_toys)
from tests._stage_toys import HILLS, HillsParams, SinusoidHills
from tests._stage_toys import register as register_stage_toys
from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
from track_gen._src.collision_geom import _safe_normalize3
from track_gen.pipeline import GATE_FAIL_BITS, presets
from track_gen.plugin import POLICY_ENV, Params, Stage, registry
from track_gen.testing import check_gate_generator, check_stage

DEV = "cpu"
E = 4


@pytest.fixture(autouse=True)
def toys():
    gt.register(registry)
    register_stage_toys(registry)


def _rng(seed=11, n=E):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _gen(**kw):
    gen = GateGenerator(presets.gates(num_envs=E, device=DEV, **kw), _rng())
    gen.generate()
    return gen


# ---------------------------------------------------------------------------------------
# Provided attributes
# ---------------------------------------------------------------------------------------

@wp.kernel
def _normalize3_k(src: wp.array(dtype=wp.vec3f), dst: wp.array(dtype=wp.vec3f)):
    t = wp.tid()
    dst[t] = _safe_normalize3(src[t])


def _provided(gen):
    tap, seq = gen.program.tap, gen._gate_sequence
    A = int(tap("anchors.xy").shape[0]) // E
    G = gen.max_gates
    count = seq.count.numpy()
    at = tap("anchors.tangent")
    want_t = wp.empty_like(at)
    wp.launch(_normalize3_k, int(at.shape[0]), [at, want_t], device=DEV)
    return (A, G, count, tap("anchors.z").numpy().reshape(E, A),
            want_t.numpy().reshape(E, A, 3))


def test_helix_z_and_tangent_reach_the_sequence():
    gen = _gen(generator=gt.HELIX)
    seq = gen._gate_sequence
    A, G, count, z, want_t = _provided(gen)
    pos = seq.position.numpy().reshape(E, G, 3)
    tan = seq.tangent.numpy().reshape(E, G, 3)
    assert (count == A).all() and seq.valid.numpy().all()
    for e in range(E):
        m = int(count[e])
        assert pos[e, :m, 2].tobytes() == z[e, :m].tobytes()          # bit for bit
        assert tan[e, :m].tobytes() == want_t[e, :m].tobytes()        # bit for bit
        assert np.isnan(pos[e, m:]).all()
    assert np.ptp(pos[..., 2][np.isfinite(pos[..., 2])]) > 0.1        # not flat


def test_helix_posed_half_size_and_roll():
    gen = _gen(generator=gt.HELIX_POSED)
    seq = gen._gate_sequence
    A, G, count, z, want_t = _provided(gen)
    hs_a = gen.program.tap("anchors.half_size").numpy().reshape(E, A)
    roll = gen.program.tap("anchors.roll").numpy().reshape(E, A)
    hs = seq.half_size.numpy().reshape(E, G)
    pos = seq.position.numpy().reshape(E, G, 3)
    left, right = (seq.left.numpy().reshape(E, G, 3), seq.right.numpy().reshape(E, G, 3))
    fwd = seq.forward.numpy().reshape(E, G, 3)
    for e in range(E):
        m = int(count[e])
        assert hs[e, :m].tobytes() == hs_a[e, :m].tobytes()
        assert pos[e, :m, 2].tobytes() == z[e, :m].tobytes()
        for i in range(m):
            # the yaw-only frame's left axis, rolled about the (horizontal) forward axis
            f = np.array([fwd[e, i, 0], fwd[e, i, 1], 0.0], np.float64)
            f /= np.linalg.norm(f)
            la0 = np.array([-f[1], f[0], 0.0])
            r = float(roll[e, i])
            la = np.cos(r) * la0 + np.sin(r) * np.array([0.0, 0.0, 1.0])
            np.testing.assert_allclose(left[e, i], pos[e, i] + hs[e, i] * la, atol=1e-6)
            np.testing.assert_allclose(right[e, i], pos[e, i] - hs[e, i] * la, atol=1e-6)
    assert seq.valid.numpy().all()


@wp.kernel
def _crossing_k(xy: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32),
                z: wp.array(dtype=wp.float32), tangent: wp.array(dtype=wp.vec3f),
                half_size: wp.array(dtype=wp.float32), roll: wp.array(dtype=wp.float32)):
    # Three gates: 0 and 1 are 0.15 apart with perpendicular forwards and 0.2 half-sizes,
    # so their bars cross in plan view; gate 2 is far away.
    t = wp.tid()
    i = t % 3
    if i == 0:
        xy[t] = wp.vec2f(0.0, 0.0)
        tangent[t] = wp.vec3f(1.0, 0.0, 0.0)
    elif i == 1:
        xy[t] = wp.vec2f(0.15, 0.0)
        tangent[t] = wp.vec3f(0.0, 1.0, 0.0)
    else:
        xy[t] = wp.vec2f(0.5, 0.5)
        tangent[t] = wp.vec3f(1.0, 0.0, 0.0)
    z[t] = 0.0
    half_size[t] = 0.2
    roll[t] = 0.0
    if i == 0:
        count[t // 3] = 3


class _Crossing(Stage):
    """A posed provider whose first two bars cross in plan view."""

    kind = "gate_generator"
    name = "acme.crossing"
    output = "anchors"
    writes = {"xy": "anchors.xy", "count": "anchors.count", "z": "anchors.z",
              "tangent": "anchors.tangent", "half_size": "anchors.half_size",
              "roll": "anchors.roll"}
    provides = frozenset({"z", "tangent", "half_size", "roll"})
    orderings = frozenset({"raw"})
    preferred = {"order": "raw"}

    def capacity(self, params, layout):
        return {"gates": 3, "min_gates": 3}

    def build(self, b, p):
        outs = [b.write(port) for port in ("xy", "count", "z", "tangent", "half_size",
                                           "roll")]
        n = b.E * 3

        def enqueue(ctx):
            ctx.launch(_crossing_k, n, outs)
        return enqueue


def test_posed_bars_that_cross_set_the_crossing_bit_with_gate_width_zero():
    registry.add(_Crossing)
    gen = _gen(generator="acme.crossing", gate_solve_iters=0, gate_radius=0.01,
               min_gates=3)
    fm = gen._gate_sequence.fail_mask.numpy()
    assert (fm == 1 << GATE_FAIL_BITS["crossing"]).all(), fm
    assert not gen._gate_sequence.valid.numpy().any()


# ---------------------------------------------------------------------------------------
# A third-party order
# ---------------------------------------------------------------------------------------

def _reversed_anchors_match(gen):
    tap = gen.program.tap
    anchors = tap("anchors.xy").numpy()
    A = anchors.shape[0] // E
    ordered = tap("gates.xy", after="order").numpy().reshape(E, gen.max_gates, 2)
    count = tap("gates.count").numpy()
    assert (count == tap("anchors.count").numpy()).all()
    for e in range(E):
        m = int(count[e])
        assert m > 0
        assert ordered[e, :m].tobytes() == anchors.reshape(E, A, 2)[e, :m][::-1].tobytes()
        assert np.isnan(ordered[e, m:]).all()


def test_reverse_on_stations_scatter_and_bezier():
    raw = _gen(generator=gt.SUPERFORMULA, gates=12)
    rev = _gen(generator=gt.SUPERFORMULA, gates=12, order=gt.REVERSE)
    _reversed_anchors_match(rev)
    a = raw.program.tap("gates.xy", after="order").numpy().reshape(E, 12, 2)
    b = rev.program.tap("gates.xy", after="order").numpy().reshape(E, 12, 2)
    assert a[:, ::-1].tobytes() == b.tobytes()
    for cfg in (GateGenConfig(num_envs=E, generator=gt.SCATTER, gate_ordering=gt.REVERSE),
                GateGenConfig(num_envs=E, generator="bezier", gate_ordering=gt.REVERSE)):
        gen = GateGenerator(cfg, _rng())
        gen.generate()
        assert (gen.program.spec.stages[1].kind, gen.program.spec.stages[1].name) == \
            ("order", gt.REVERSE)
        _reversed_anchors_match(gen)
        assert gen._gate_sequence.valid.numpy().any()


# ---------------------------------------------------------------------------------------
# Stations of a third-party track generator
# ---------------------------------------------------------------------------------------

def _stations_ref(xy: np.ndarray, A: int) -> np.ndarray:
    """``A`` points at equal arc length on the closed polyline ``xy`` (float64)."""
    pts = xy.astype(np.float64)
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    out = []
    for k in range(A):
        s = L * k / A
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(pts) - 1)
        u = 0.0 if seg[i] == 0 else (s - cum[i]) / seg[i]
        out.append(pts[i] + u * (pts[(i + 1) % len(pts)] - pts[i]))
    return np.array(out)


@pytest.mark.parametrize("source", ["stations", "relaxed_stations"])
def test_superformula_stations_are_arc_spaced(source):
    gen = _gen(generator=gt.SUPERFORMULA, gates=12, source=source)
    seq, tap = gen._gate_sequence, gen.program.tap
    assert (seq.count.numpy() == 12).all() and seq.valid.numpy().mean() > 0
    anchors = tap("anchors.xy").numpy().reshape(E, 12, 2)
    if source == "stations":
        loops = tap("raw.xy").numpy().reshape(E, -1, 2)
        counts = [loops.shape[1]] * E
    else:
        loops = tap("cl.xy").numpy().reshape(E, -1, 2)
        counts = tap("cl.count").numpy()
    for e in range(E):
        want = _stations_ref(loops[e, :int(counts[e])], 12)
        np.testing.assert_allclose(anchors[e], want, atol=2e-5)


# ---------------------------------------------------------------------------------------
# A gate validity check (fail bit 16, dynamic z_cap)
# ---------------------------------------------------------------------------------------

def test_max_height_sets_its_bit_exactly_where_the_highest_gate_is_above_the_cap():
    gen = _gen(elevation=("uniform", {"z_min": 0.0, "z_max": 1.0}),
               checks=("gate_default", gt.MAX_HEIGHT))
    bit = gen.program.fail_bits[gt.MAX_HEIGHT]
    assert bit == 16 and gen._gate_sequence.fail_bits[gt.MAX_HEIGHT] == 16
    seq = gen._gate_sequence
    cap = np.array([0.2, 0.5, 0.8, 2.0], np.float32)
    gen.params[gt.MAX_HEIGHT].set("z_cap", wp.array(cap, dtype=wp.float32, device=DEV))
    gen.generate()
    pos = seq.position.numpy().reshape(E, gen.max_gates, 3)
    top = np.array([np.nanmax(pos[e, :int(seq.count.numpy()[e]), 2]) for e in range(E)])
    own = (seq.fail_mask.numpy() >> bit) & 1
    np.testing.assert_array_equal(own, (top > cap).astype(own.dtype))
    assert own.any() and not own.all()
    np.testing.assert_array_equal(seq.valid.numpy(), (seq.fail_mask.numpy() == 0))


# ---------------------------------------------------------------------------------------
# Conformance on the toys (they build GateGenerator(presets.gates(...)) internally)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [gt.HELIX, gt.SCATTER])
def test_toy_gate_generators_conform(name):
    check_gate_generator(name, devices=["cpu"]).raise_for_failures()


@pytest.mark.parametrize("kind,name", [("order", gt.REVERSE), ("stations", "raw"),
                                       ("stations", "relaxed"),
                                       ("validity", gt.MAX_HEIGHT)])
def test_gate_stages_conform(kind, name):
    check_stage(kind, name, devices=["cpu"]).raise_for_failures()


# ---------------------------------------------------------------------------------------
# Registration rules, quarantine
# ---------------------------------------------------------------------------------------

def _gate_stage(**attrs):
    def build(self, b, p):
        return lambda ctx: None

    ns = dict(kind="gate_generator", name="acme.g", output="anchors",
              writes={"xy": "anchors.xy", "count": "anchors.count"}, build=build)
    ns.update(attrs)
    return type("G", (Stage,), ns)


@pytest.mark.parametrize("attrs,exc,match", [
    (dict(output="centerline"), ValueError,
     "Stage G: a gate generator Stage declares output='anchors', got 'centerline'"),
    (dict(writes={"xy": "anchors.xy", "count": "anchors.count", "w": "cl.z"}), ValueError,
     r"Stage G.writes: channels \['cl.z'\] are not anchors channels; a gate generator "
     r"writes \['anchors.count', 'anchors.half_size'"),
    (dict(writes={"xy": "anchors.xy"}), ValueError,
     "Stage G.writes must map ports to 'anchors.xy' and 'anchors.count'"),
    (dict(reads={"s": "seed"}), ValueError,
     r"Stage G.reads must be empty \(a generator reads no channel\)"),
    (dict(provides=frozenset({"z"})), ValueError,
     r"Stage G: provides \['z'\] must name exactly the anchors attributes it writes "
     r"\(\[\]\)"),
    (dict(writes={"xy": "anchors.xy", "count": "anchors.count", "z": "anchors.z"},
          provides=frozenset({"z"})), ValueError,
     r"Stage G provides per-anchor attributes \['z'\]: they are carried to the gate slots "
     r"by index, so orderings must be frozenset\(\{'raw'\}\)"),
    (dict(writes={"xy": "anchors.xy", "count": "anchors.count",
                  "h": "anchors.half_size"},
          provides=frozenset({"half_size"}), orderings=frozenset({"raw"})), ValueError,
     "Stage G provides 'half_size' without 'roll'"),
    (dict(orderings={"raw"}), TypeError, "Stage G.orderings must be None or a non-empty "
                                         "frozenset"),
    (dict(excluded_orderings=("raw",)), TypeError, "Stage G.excluded_orderings must be a "
                                                   "frozenset"),
    (dict(orderings=frozenset({"raw"}), excluded_orderings=frozenset({"ccw"})), ValueError,
     r"declare orderings \(a whitelist\) or excluded_orderings \(with orderings=None\), "
     r"not both"),
    (dict(preferred={"sort": "ccw"}), ValueError, r"Stage G.preferred: unknown keys "
                                                   r"\['sort'\]"),
    (dict(orderings=frozenset({"raw"}), preferred={"order": "ccw"}), ValueError,
     r"Stage G.preferred\['order'\]='ccw' is not one of its orderings \['raw'\]"),
    (dict(excluded_orderings=frozenset({"ccw"}), preferred={"order": "ccw"}), ValueError,
     "is one of its excluded_orderings"),
])
def test_gate_generator_registration_rules(attrs, exc, match):
    with pytest.raises(exc, match=match):
        registry.add(_gate_stage(**attrs))
    assert "acme.g" not in registry.available("gate_generator")


def test_other_rules_of_the_new_kinds():
    G = _gate_stage(kind="generator")
    with pytest.raises(ValueError, match="output='anchors' is a gate generator: declare "
                                         "kind='gate_generator'"):
        registry.add(G)

    def build(self, b, p):
        return lambda ctx: None

    W = type("W", (Stage,), dict(kind="width", name="acme.w", preferred={"order": "raw"},
                                 writes={"w": "cl.half_width"}, build=build))
    with pytest.raises(ValueError, match=r"Stage W.preferred is read for gate generators "
                                         r"only \(roadmap PR11\)"):
        registry.add(W)
    O = type("O", (Stage,), dict(kind="order", name="acme.o", build=build,
                                 reads={"a": "anchors.xy"}, writes={"xy": "gates.xy"}))
    with pytest.raises(ValueError, match=r"Stage O \(kind 'order'\) must write 'gates.xy' "
                                         r"and 'gates.count'"):
        registry.add(O)
    registry.add(_gate_stage())
    info = registry.describe("gate_generator", "acme.g")
    assert info.output == "anchors" and info.orderings is None
    assert info.excluded_orderings == frozenset() and dict(info.preferred) == {}
    assert registry.describe("gate_generator", gt.HELIX).orderings == frozenset({"raw"})


def test_a_broken_anchors_plugin_is_quarantined(monkeypatch):
    monkeypatch.delenv(POLICY_ENV, raising=False)
    Broken = _gate_stage(name="broken.g", provides=frozenset({"z"}))
    with registry.scoped(fresh=True):
        install_entry_points(monkeypatch, FakeEntryPoint(
            "broken", lambda reg: reg.add(Broken), FakeDist("broken-gates")))
        with pytest.warns(RuntimeWarning, match="quarantined"):
            registry.available("gate_generator")
        assert "must name exactly the anchors attributes" in registry.quarantined["broken"]
        with pytest.raises(Exception, match="quarantined"):
            GateGenerator(GateGenConfig(generator="broken.g", num_envs=2))


# ---------------------------------------------------------------------------------------
# Gate Stage registration and default-order resolution
# ---------------------------------------------------------------------------------------

def test_registering_a_gate_stage_is_silent(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        registry.add(TOY_RING)
    monkeypatch.delenv(POLICY_ENV, raising=False)
    with registry.scoped(fresh=True):
        install_entry_points(monkeypatch, FakeEntryPoint("toys", register_toys,
                                                         FakeDist("toy-plugins")))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert "toy.ring" in registry.available("gate_generator")




@pytest.mark.parametrize("supported,want", [({"raw"}, "raw"),
                                            ({"ccw", "raw", "random_pairs"}, "ccw")])
def test_gate_stage_params_read_the_resolved_default_ordering(supported, want):
    seen = []

    @dataclasses.dataclass(frozen=True)
    class RingParams(Params):
        @classmethod
        def from_legacy(cls, config):
            seen.append(config.gate_ordering)
            return cls()

    class Ring(TOY_RING):
        name = "toy.ring_seen"
        Params = RingParams
        orderings = frozenset(supported)
        preferred = {}

    registry.add(Ring)
    cfg = GateGenConfig(generator="toy.ring_seen", num_envs=2)
    gen = GateGenerator(cfg)
    gen.generate()
    assert cfg.gate_ordering is None and gen._config.gate_ordering == want
    assert seen and set(seen) == {want}


def test_course_with_a_provider_and_the_default_ordering():
    """``CourseConfig`` with a generator that accepts only ``"raw"`` and no explicit
    ``gate_ordering``: the config default resolves to the generator's order."""
    from track_gen.course import Course, CourseConfig

    course = Course(CourseConfig(mode="gates", gen=GateGenConfig(
        generator=gt.HELIX, gate_width=0.05, num_envs=2, device=DEV), seeds=1,
        post_radius=0.02))
    assert course.generator._config.gate_ordering == "raw"
    assert course.generator._gate_sequence.valid.numpy().any()




# ---------------------------------------------------------------------------------------
# Elevation plugins on gates: the conformance kit and from_legacy(GateGenConfig)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", [(HILLS, None), ("uniform", {"z_max": 1.0})])
def test_elevation_stages_conform_on_gates(name, params):
    report = check_stage("elevation", name, params, pipeline="gates", devices=["cpu"])
    report.raise_for_failures()
    assert report.result("construct", "cpu").passed


def test_check_stage_pipeline_argument_is_checked():
    with pytest.raises(ValueError, match="pipeline must be None, 'track' or 'gates'"):
        check_stage("elevation", "flat", pipeline="gate")
    with pytest.raises(ValueError, match=r"pipeline='gates': a relax stage runs in the "
                                         r"track pipeline only"):
        check_stage("relax", "xpbd", pipeline="gates")
    with pytest.raises(ValueError, match=r"pipeline='track': a order stage runs in the "
                                         r"gate pipeline only"):
        check_stage("order", "raw", pipeline="track")


def test_a_track_from_legacy_on_gates_names_the_missing_field():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class TrackOnlyParams(HillsParams):
        @classmethod
        def from_legacy(cls, cfg):
            return cls(waves=int(cfg.num_points) // 64)     # a TrackGenConfig-only field

    class TrackOnlyHills(SinusoidHills):
        name = "acme.hills_track_only"
        Params = TrackOnlyParams

    registry.add(TrackOnlyHills)
    with pytest.raises(ValueError, match=r"GateGenConfig.z_profile='acme.hills_track_only':"
                                         r" TrackOnlyHills.Params.from_legacy\(\) received "
                                         r"a GateGenConfig and read 'num_points'"):
        GateGenerator(GateGenConfig(z_profile="acme.hills_track_only", num_envs=2))
    # presets.gates(elevation=...) passes the params explicitly: no from_legacy.
    _gen(elevation="acme.hills_track_only")


def test_course_with_a_half_size_provider_is_deferred():
    """Deferred (CHANGELOG): ``Course`` requires ``gen.gate_width > 0`` while a
    ``half_size`` provider refuses any ``gate_width``."""
    from track_gen.course import Course, CourseConfig

    gen = dict(generator=gt.HELIX_POSED, num_envs=2, device=DEV)
    with pytest.raises(ValueError, match="gates mode requires gen.gate_width > 0"):
        CourseConfig(mode="gates", gen=GateGenConfig(**gen), seeds=1, post_radius=0.02)
    with pytest.raises(ValueError, match="'acme.helix_posed' provides per-gate half_size"):
        Course(CourseConfig(mode="gates", gen=GateGenConfig(gate_width=0.05, **gen),
                            seeds=1, post_radius=0.02))
