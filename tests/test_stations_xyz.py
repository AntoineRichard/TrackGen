"""3D station spacing, provided altitude and gate plugin channels (roadmap PR13a, CPU).

``stations.raw`` / ``stations.relaxed`` take ``spacing_metric="xy"`` (plan view, PR11's
code, byte-identical) or ``"xyz"`` (3D arc length, which needs the loop's altitude: a
generator that provides ``z``, or an elevation run on the relaxed centerline before the
stations: ``presets.gates(elevation_on="centerline")``, the ``xyz`` default). The
built-in stations carry the families a later stage reads to ``anchors.<attr>`` through
their tables (``carry.z``, ``carry.attributes``), writing the table sentinels on every
exit path. Gate-domain stages write ``gates.<pkg>.<name>`` channels and export them as
``GateSequence.attributes``. The toys are ``tests/_channel_toys.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests import _channel_toys as toys
from tests.tools import capture_pr13a_relaxed_stations as pr11
from track_gen import GateGenerator, PerEnvSeededRNG
from track_gen.pipeline import PlanError, StageRef, build, presets
from track_gen.plugin import registry

DEV = "cpu"
E = 6
G = 12
_NOISE = ("noise", {"z_base": 0.5, "z_max": 1.5, "z_noise_amplitude": 0.5,
                    "z_noise_harmonics": 4})
RIBBON_KEYS = {"acme.theta", "acme.surface_id", "acme.lane", "acme.grip", "acme.xy_copy"}


@pytest.fixture(autouse=True)
def acme():
    with registry.scoped():
        toys.register(registry)
        yield registry


def _rng(n=E, seed=11):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _gates(**kw):
    return presets.gates(num_envs=kw.pop("num_envs", E), device=DEV, **kw)


def _gen(spec, seed=11):
    return GateGenerator(spec, _rng(spec.batch.num_envs, seed))


def _bytes(a) -> bytes:
    return np.ascontiguousarray(a.numpy()).tobytes()


def _arc3(xy, z):
    """Cumulative 3D arc length of a closed polyline (float64), and its segments."""
    x = xy.astype(np.float64)
    zz = z.astype(np.float64)
    nxt = np.roll(np.arange(len(x)), -1)
    seg = np.sqrt(((x[nxt] - x) ** 2).sum(-1) + (zz[nxt] - zz) ** 2)
    return np.concatenate([[0.0], np.cumsum(seg)]), seg


# ---------------------------------------------------------------------------------------
# xyz spacing on a z-providing loop
# ---------------------------------------------------------------------------------------

def _station_positions(p, e):
    """The 3D arc position of every station of env ``e`` along the raw polyline, from
    the stations' own tables (float64)."""
    raw = p.tap("raw.xy").numpy().reshape(E, -1, 2)[e]
    z = p.tap("raw.z").numpy().reshape(E, -1)[e]
    cum, seg = _arc3(raw, z)
    sc = p._scratch["stations"]
    n = raw.shape[0]
    i0 = sc["tables.i0"].numpy().reshape(E, -1)[e] - e * n
    fr = sc["tables.frac"].numpy().reshape(E, -1)[e].astype(np.float64)
    return cum[i0] + fr * seg[i0], cum[-1]


# Five climbs per loop: 12 plan-view stations are not 3D-equal by symmetry (3 would be).
_HELIX = {"waves": 5}


def test_xyz_stations_are_equally_spaced_in_3d():
    spec = _gates(generator="acme.helix_loop", generator_params=_HELIX, gates=G,
                  spacing_metric="xyz")
    assert spec.ids == ("generator", "stations", "order", "carry_z", "gate_relax", "lift",
                        "tangents", "gate_frames", "validity")
    assert dict(spec.stages[1].params) == {"gates": G, "spacing_metric": "xyz"}
    gen = _gen(spec)
    p = gen.program
    seq = gen._gate_sequence
    for e in range(E):
        pos, L = _station_positions(p, e)
        gaps = np.diff(np.concatenate([pos, [pos[0] + L]]))
        np.testing.assert_allclose(gaps, L / G, rtol=1e-6)
    # the xy metric does not space them equally in 3D
    xy = _gen(_gates(generator="acme.helix_loop", generator_params=_HELIX, gates=G))
    worst = 0.0
    for e in range(E):
        pos, L = _station_positions(xy.program, e)
        gaps = np.diff(np.concatenate([pos, [pos[0] + L]]))
        worst = max(worst, float(np.max(np.abs(gaps - L / G) / (L / G))))
    assert worst > 1e-3
    # z reaches the gates by index (carry.z), linear through the stations' tables
    az = p.tap("anchors.z").numpy()
    count = seq.count.numpy()
    pos3 = seq.position.numpy().reshape(E, -1, 3)
    for e in range(E):
        np.testing.assert_array_equal(pos3[e, :count[e], 2], az.reshape(E, -1)[e, :count[e]])
    assert np.ptp(az) > 0.3


def test_relaxed_stations_xyz_runs_the_elevation_on_the_centerline():
    spec = _gates(generator="bezier", source="relaxed_stations", gates=G,
                  elevation=_NOISE, spacing_metric="xyz")
    assert spec.ids == ("generator", "resample", "width", "relax", "frames", "elevation",
                        "stations", "order", "carry_z", "gate_relax", "lift", "tangents",
                        "gate_frames", "validity")
    gen = _gen(spec)
    p = gen.program
    assert p.channels["cl.z"] == ("elevation",)
    assert p.channels["anchors.z"] == ("stations",)
    assert p.channels["gates.z"] == ("carry_z",)
    assert "gate_arclen" not in spec.ids
    assert gen._gate_sequence.valid.numpy().any()
    # elevation_on="centerline" with the xy metric: the same placement, plan-view spacing
    xy = _gates(generator="bezier", source="relaxed_stations", gates=G, elevation=_NOISE,
                elevation_on="centerline")
    assert xy.ids == spec.ids
    assert "spacing_metric" not in dict(xy.stages[6].params)


def test_relaxed_stations_xy_is_byte_identical_to_pr11():
    """``presets.gates(source="relaxed_stations")`` with the default ``xy`` /
    ``elevation_on="gates"`` keeps the PR11 stage list (plus ``width.constant``, which
    launches nothing) and its bytes (digests captured at ed075f6, per platform)."""
    gold = pr11.load().get(pr11.platform_key())
    if gold is None:
        pytest.skip(f"no PR11 relaxed_stations digests for {pr11.platform_key()}")
    for name, kw in pr11.CASES.items():
        got = pr11.capture(kw)
        bad = sorted(k for k in gold[name] if gold[name][k] != got.get(k))
        assert not bad, f"{name}: differs from PR11 at {bad}"
    spec = _gates(**pr11.CASES["bezier/noise"])
    assert spec.ids == ("generator", "resample", "width", "relax", "stations", "order",
                        "gate_relax", "gate_arclen", "elevation", "lift", "tangents",
                        "gate_frames", "validity")


@pytest.mark.parametrize("kw,match", [
    (dict(generator="bezier", source="stations", elevation=_NOISE, spacing_metric="xyz"),
     r"spacing_metric='xyz' with an elevation: the elevation must run on the centerline "
     r"before the stations, which needs source='relaxed_stations'"),
    (dict(generator="bezier", source="relaxed_stations", elevation=_NOISE,
          spacing_metric="xyz", elevation_on="gates"),
     r"spacing_metric='xyz' needs the loop's altitude before the stations are placed"),
    (dict(generator="bezier", source="stations", elevation_on="centerline"),
     r"elevation_on='centerline' runs the elevation on the relaxed centerline"),
    (dict(generator="bezier", spacing_metric="xyz"),
     r"spacing_metric= places stations; gate generator 'bezier' places its own gates"),
    (dict(generator="acme.helix_loop", gates=G, order="ccw"),
     r"generator 'acme.helix_loop' provides z, carried to the gates by index: "
     r"order='raw'"),
    (dict(generator="acme.helix_loop", gates=G, elevation="noise"),
     r"generator 'acme.helix_loop' provides its own altitudes \(z\)"),
    (dict(generator="acme.helix_loop", gates=G, stations="acme.st"),
     r"stations 'acme.st' carries no centerline channels \(plugin carriers are "
     r"deferred\); generator 'acme.helix_loop' provides z"),
    (dict(generator="bezier", source="stations", stations="acme.st",
          spacing_metric="xyz"),
     r"spacing_metric='xyz': stations 'acme.st' has no spacing_metric parameter"),
    (dict(generator="bezier", source="stations", spacing_metric="xz"),
     r"spacing_metric must be 'xy' or 'xyz'"),
    (dict(generator="bezier", source="stations", gates=G, spacing_metric="xyz"),
     r"spacing_metric='xyz' measures 3D arc length along the loop, which carries no "
     r"altitude here \(a flat elevation and generator 'bezier' provides no z\): use a "
     r"generator that provides z \(raw.z\), or source='relaxed_stations' with a "
     r"non-flat elevation"),
    (dict(generator="bezier", source="relaxed_stations", gates=G, spacing_metric="xyz"),
     r"which carries no altitude here .* or a non-flat elevation \(it runs on the "
     r"relaxed centerline\)"),
    (dict(generator="acme.helix_loop", gates=G, spacing_metric="xyz",
          elevation_on="gates"),
     r"elevation_on='gates' places the elevation stage; generator 'acme.helix_loop' "
     r"provides its own altitudes \(z\), so no elevation runs"),
    (dict(generator="acme.helix_loop", gates=G, source="relaxed_stations",
          elevation_on="centerline"),
     r"elevation_on='centerline' places the elevation stage; generator "
     r"'acme.helix_loop' provides its own altitudes"),
])
def test_the_gate_preset_composition_errors(kw, match):
    with pytest.raises(ValueError, match=match):
        _gates(**kw)


def test_xyz_without_an_altitude_is_a_plan_error():
    spec = _gates(generator="bezier", source="stations", gates=G).replace(
        "stations", StageRef("stations", "raw", params={"gates": G,
                                                         "spacing_metric": "xyz"}))
    with pytest.raises(PlanError, match=r"stage 'stations' \(stations.raw\): "
                                        r"spacing_metric='xyz' measures 3D arc length along "
                                        r"the loop, which carries no altitude"):
        build(spec, seeds=1)


# ---------------------------------------------------------------------------------------
# Carried attributes on the gates
# ---------------------------------------------------------------------------------------

def test_stations_carry_ribbon_attributes_and_keep_the_pr11_geometry():
    a = _gen(_gates(generator="acme.ribbon", gates=G))
    b = _gen(_gates(generator="acme.ribbon_plain", gates=G))
    assert "carry_attributes" in a.program.spec.ids
    assert "carry_attributes" not in b.program.spec.ids
    for ch in ("anchors.xy", "anchors.count"):
        assert _bytes(a.program.tap(ch)) == _bytes(b.program.tap(ch)), ch
    for f in ("position", "tangent", "forward", "orientation", "half_size", "left",
              "right", "valid", "count", "fail_mask"):
        assert _bytes(getattr(a._gate_sequence, f)) == \
            _bytes(getattr(b._gate_sequence, f)), f
    # a linear carry of raw.xy through the anchors tables reproduces the anchors
    assert _bytes(a.program.tap("anchors.acme.xy_copy")) == _bytes(a.program.tap("anchors.xy"))
    attrs = a._gate_sequence.attributes
    assert set(attrs) == RIBBON_KEYS == set(a.program.attributes)
    count = a._gate_sequence.count.numpy()
    for k, arr in attrs.items():
        rows = arr.numpy().reshape(E, -1, *arr.numpy().shape[1:])
        an = a.program.tap("anchors." + k).numpy()
        an = an.reshape(E, -1, *an.shape[1:])
        for e in range(E):
            m = int(count[e])
            assert rows[e, :m].tobytes() == an[e, :m].tobytes(), (k, e)     # by index
            tail = rows[e, m:]
            assert np.all(tail == -1) if arr.dtype is wp.int32 else np.all(np.isnan(tail))
        assert a.program.attributes[k].domain == "gates"
    # relaxed stations carry them too
    rel = _gen(_gates(generator="acme.ribbon", gates=G, source="relaxed_stations"))
    assert set(rel._gate_sequence.attributes) == RIBBON_KEYS


def test_a_permuting_order_leaves_the_attributes_out():
    gen = _gen(_gates(generator="acme.ribbon", gates=G, order="ccw"))
    assert "carry_attributes" not in gen.program.spec.ids
    assert dict(gen._gate_sequence.attributes) == {} and dict(gen.program.attributes) == {}
    assert "anchors.acme.grip" not in gen.program.channels      # nothing reads them
    spec = _gates(generator="acme.ribbon", gates=G)
    bad = spec.replace("order", StageRef("order", "ccw"))
    with pytest.raises(PlanError, match=r"stage 'carry_attributes' \(carry.attributes\) "
                                        r"carries the anchors attributes by anchor index, "
                                        r"which needs order.raw"):
        build(bad, seeds=1)


def _flags(gen, values):
    gen.params["generator"].set(
        "invalid", wp.array(np.asarray(values, np.float32), dtype=wp.float32, device=DEV))


def test_skipped_envs_get_pure_padding_and_the_table_sentinels():
    """Envs the stations skip (the toy's gen.valid = 0 flag) get count 0, pure padding in
    every carried anchors image and gates attribute, and the sentinel in the anchors
    tables, on the first generation and after the flags flip (envs that HAD valid
    tables)."""
    gen = _gen(_gates(generator="acme.ribbon", gates=G))
    p = gen.program
    flip = np.array([1, 0] * (E // 2), np.float32)
    for k, flags in enumerate((flip, 1.0 - flip)):
        _flags(gen, flags)
        gen.generate()
        bad = flags != 0
        seq = gen._gate_sequence
        np.testing.assert_array_equal(seq.count.numpy()[bad], 0)
        assert np.all(seq.count.numpy()[~bad] == G)
        sc = p._scratch["stations"]
        i0 = sc["tables.i0"].numpy().reshape(E, -1)
        fr = sc["tables.frac"].numpy().reshape(E, -1)
        assert np.all(i0[bad] == -1) and np.all(np.isnan(fr[bad])), k
        assert np.all(i0[~bad] >= 0), k
        for key in RIBBON_KEYS:
            for arr in (p.tap("anchors." + key), seq.attributes[key]):
                rows = arr.numpy().reshape(E, -1)
                ok = np.all(rows[bad] == -1) if arr.dtype is wp.int32 \
                    else np.all(np.isnan(rows[bad]))
                assert ok, (k, key)


def test_an_anchors_generator_attribute_reaches_the_gates_by_index():
    spec = _gates(generator="acme.tagged")
    assert spec.ids[:3] == ("generator", "order", "carry_attributes")
    gen = _gen(spec)
    tag = gen._gate_sequence.attributes["acme.tag"].numpy().reshape(E, -1)
    count = gen._gate_sequence.count.numpy()
    for e in range(E):
        np.testing.assert_array_equal(tag[e, :count[e]], np.arange(count[e]) % 3)
        assert np.all(tag[e, count[e]:] == -1)
    assert gen.program.attributes["acme.tag"].interp is None
    # through the flat config as well (GateGenConfig)
    from track_gen import GateGenConfig

    cfg = GateGenerator(GateGenConfig(device=DEV, num_envs=E, generator="acme.tagged",
                                      max_gates=toys.TAG_A), _rng())
    assert "carry_attributes" in cfg.program.spec.ids
    got = cfg._gate_sequence.attributes["acme.tag"].numpy().reshape(E, -1)
    for e in range(E):
        np.testing.assert_array_equal(got[e, :count[e]], tag[e, :count[e]])


def test_an_unread_anchors_attribute_is_a_plan_error():
    spec = _gates(generator="acme.tagged")
    bare = spec.remove("carry_attributes")
    with pytest.raises(PlanError, match=r"stage 'generator' \(gate_generator.acme.tagged\) "
                                        r"writes anchors.acme.tag, but no stage reads it: "
                                        r"carry it to the gate slots with carry.attributes"):
        build(bare, seeds=1)


def test_a_gate_check_exports_its_own_channel_and_fail_bit():
    gen = _gen(_gates(generator="acme.tagged", checks=("gate_default",
                                                       "acme.gate_clearance")))
    seq = gen._gate_sequence
    assert set(seq.attributes) == {"acme.tag", "acme.clearance"}
    clear = seq.attributes["acme.clearance"].numpy().reshape(E, -1)
    pos = seq.position.numpy().reshape(E, -1, 3)[..., :2]
    count = seq.count.numpy()
    for e in range(E):
        m = int(count[e])
        want = np.linalg.norm(np.roll(pos[e, :m], -1, axis=0) - pos[e, :m], axis=-1)
        np.testing.assert_allclose(clear[e, :m], want, rtol=1e-5)
        assert np.all(np.isnan(clear[e, m:]))
    bit = gen.program.fail_bits["acme.gate_clearance"]
    assert not np.any((seq.fail_mask.numpy() >> bit) & 1)
    gen.params["acme.gate_clearance"].set("min_clearance", 10.0)
    gen.generate()
    assert np.all((gen._gate_sequence.fail_mask.numpy() >> bit) & 1)


def test_gate_attributes_masked_and_poisoned():
    spec = _gates(generator="acme.ribbon", gates=G, on_invalid="poison")
    gen = _gen(spec)
    flags = np.array([0, 1, 0, 0, 1, 0], np.float32)
    _flags(gen, flags)
    gen.generate()
    seq = gen._gate_sequence
    bad = seq.fail_mask.numpy() != 0
    assert bad.tolist() == (flags != 0).tolist()
    for k, arr in seq.attributes.items():
        rows = arr.numpy().reshape(E, -1)
        assert np.all(rows[bad] == -1) if arr.dtype is wp.int32 else np.all(np.isnan(rows[bad]))
    before = {k: a.numpy().copy() for k, a in seq.attributes.items()}
    mask = np.array([1, 0, 1, 0, 0, 1], np.int32)
    gen.generate(advance=True, mask=wp.array(mask, dtype=wp.int32, device=DEV))
    for k, a in gen._gate_sequence.attributes.items():
        now, was = a.numpy().reshape(E, -1), before[k].reshape(E, -1)
        for e in np.flatnonzero(mask == 0):
            assert now[e].tobytes() == was[e].tobytes(), (k, e)
        assert any(now[e].tobytes() != was[e].tobytes() for e in np.flatnonzero(mask))


def test_a_gate_elevation_plugin_channel_is_rewired_to_the_gates():
    gen = _gen(_gates(generator="bezier", elevation="acme.slope"))
    p = gen.program
    reads, writes = p._ports["elevation"]
    assert writes == {"z": "gates.z", "slope": "gates.acme.slope"}
    slope = gen._gate_sequence.attributes["acme.slope"].numpy().reshape(E, -1)
    count = gen._gate_sequence.count.numpy()
    for e in range(E):
        np.testing.assert_array_equal(slope[e, :count[e]], np.arange(count[e]))
    # on a track it stays a centerline channel
    from track_gen import TrackGenerator

    t = TrackGenerator(presets.track(num_envs=E, device=DEV, elevation="acme.slope"),
                       _rng())
    assert t.program._ports["elevation"][1]["slope"] == "cl.acme.slope"
    assert set(t._track.attributes) == {"acme.slope"}


def test_an_elevation_writing_per_cl_runs_on_gates_unchanged():
    """The natural track form, ``write("slope", ..., per="cl", export=True)``, runs in a
    gate Program as is: ``per=`` names the declared channel's point class (cl), the
    rewired ``gates.acme.slope`` is sized per gate."""
    from track_gen import TrackGenerator
    from track_gen.testing import check_stage

    class SlopeCl(toys.Slope):
        name = "acme.slope_cl"

        def build(self, b, p):
            count = b.read("count")
            z = b.write("z", wp.float32, per="cl")
            slope = b.write("slope", wp.float32, per="cl", export=True)
            S = b.stride_of("z")

            def enqueue(ctx):
                ctx.launch(toys._slope_k, b.E * S, [count, S, z, slope])
            return enqueue

    registry.add(SlopeCl)
    for kw in ({}, {"source": "relaxed_stations"}):
        gen = _gen(_gates(generator="bezier", elevation="acme.slope_cl", **kw))
        assert gen.program._ports["elevation"][1]["slope"] == "gates.acme.slope"
        slope = gen._gate_sequence.attributes["acme.slope"].numpy().reshape(E, -1)
        count = gen._gate_sequence.count.numpy()
        for e in range(E):
            np.testing.assert_array_equal(slope[e, :count[e]], np.arange(count[e]))
    t = TrackGenerator(presets.track(num_envs=E, device=DEV, elevation="acme.slope_cl"),
                       _rng())
    assert set(t._track.attributes) == {"acme.slope"}
    report = check_stage("elevation", "acme.slope_cl", pipeline="gates", num_envs=4,
                         capture_modes=())
    assert report.ok, report


def test_a_gate_plugin_channel_on_a_centerline_stage_is_refused():
    from track_gen.plugin import Stage

    class Mixed(Stage):
        kind = "validity"
        name = "acme.mixed"
        reads = {"xy": "cl.xy", "g": "gates.acme.clearance", "fail": "valid.fail"}
        writes = {"fail": "valid.fail"}

        def build(self, b, p):
            return lambda ctx: None

    registry.add(Mixed)
    spec = _gates(generator="acme.tagged", checks=("gate_default", "acme.gate_clearance",
                                                   "acme.mixed"))
    with pytest.raises(PlanError, match=r"plugin channel 'gates.acme.clearance' is a gate "
                                        r"channel; a centerline stage reads and writes "
                                        r"cl.<pkg>.<name>"):
        build(spec, seeds=1)


def test_a_plugin_stations_stage_carries_nothing_and_says_so():
    spec = _gates(generator="acme.ribbon", source="stations", stations="acme.st")
    assert "carry_attributes" not in spec.ids
    gen = _gen(spec)
    assert dict(gen._gate_sequence.attributes) == {}
    assert gen._gate_sequence.valid.numpy().any()
