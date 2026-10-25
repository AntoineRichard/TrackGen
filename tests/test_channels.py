"""Per-point channels carried through the track pipeline (roadmap PR13a, CPU).

A centerline generator writes ``raw.<attr>`` attributes with an ``interp`` policy; the
constant-spacing resample carries every one to ``cl.<attr>`` through its lookup's
tables, relax keeps them on their beads (bead attachment: no new version), the uniform
resample re-interpolates them onto the final points, and the exported ones are
``Track.attributes`` (pointer-stable, committed and poisoned with the Track). Plus the
public ``ops.resample_channels`` / ``ResampleTables``, the ``interp=`` / ``export=``
rules, and a generator that provides its altitude (``raw.z``). The toys are
``tests/_channel_toys.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests import _channel_toys as toys
from track_gen import PerEnvSeededRNG, TrackGenerator
from track_gen import ops
from track_gen.exec import LaunchContext
from track_gen.pipeline import AttributeInfo, PlanError, StageRef, build, presets
from track_gen.plugin import Stage, registry
from track_gen.stages.track import _alloc_track
from track_gen.testing.trace import trace_launches

DEV = "cpu"
E = 6
ATTRS = {"acme.theta": (wp.float32, "angle"), "acme.surface_id": (wp.int32, "nearest"),
         "acme.lane": (wp.int32, "previous"), "acme.grip": (wp.float32, "linear"),
         "acme.xy_copy": (wp.vec2f, "linear")}
GEOMETRY = ("outer", "center", "inner", "tangent", "normal", "arclen", "length", "valid",
            "count", "winding", "truncated", "fail_mask", "half_width")


@pytest.fixture(autouse=True)
def acme():
    with registry.scoped():
        toys.register(registry)
        yield registry


def _rng(n=E, seed=11):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _track(**kw):
    return presets.track(num_envs=E, device=DEV, **kw)


def _gen(spec, seed=11):
    return TrackGenerator(spec, _rng(spec.batch.num_envs, seed))


def _bytes(a) -> bytes:
    return np.ascontiguousarray(a.numpy()).tobytes()


def _rows(arr, n=E):
    a = arr.numpy()
    return a.reshape(n, -1, *a.shape[1:])


def _ctx():
    return LaunchContext.for_call(DEV)


# ---------------------------------------------------------------------------------------
# Carry exactness
# ---------------------------------------------------------------------------------------

def test_families_never_change_the_geometry():
    a = _gen(_track(generator="acme.ribbon", on_invalid="keep"))
    b = _gen(_track(generator="acme.ribbon_plain", on_invalid="keep"))
    for f in GEOMETRY:
        assert _bytes(getattr(a._track, f)) == _bytes(getattr(b._track, f)), f
    for sid in ("resample", "relax", "uniform"):
        assert _bytes(a.program.tap("cl.xy", after=sid)) == \
            _bytes(b.program.tap("cl.xy", after=sid)), sid
    assert set(a._track.attributes) == set(ATTRS) and not b._track.attributes
    # Both carriers ran their table-writing lookups (ribbon) / the plain ones (plain).
    with trace_launches() as ta:
        a.enqueue()
    with trace_launches() as tb:
        b.enqueue()
    ka = {k.rsplit(".", 1)[-1] for k in ta.kernels()}
    kb = {k.rsplit(".", 1)[-1] for k in tb.kernels()}
    assert {"_cs_lookup_tables_k", "_resample_lookup_tables_k"} <= ka
    assert {"_cs_lookup_k", "_resample_lookup_k"} <= kb
    assert not {"_cs_lookup_tables_k", "_resample_lookup_tables_k"} & kb
    # one carry launch per raw family per carrier (and ribbon's own attribute kernel),
    # nothing else
    assert len(ta) - len(tb) == 2 * 6 + 1


def test_a_linear_carry_of_the_raw_loop_reproduces_the_resampled_points():
    p = _gen(_track(generator="acme.ribbon")).program
    assert _bytes(p.tap("cl.acme.xy_copy", after="resample")) == \
        _bytes(p.tap("cl.xy", after="resample"))
    q = _gen(_track(generator="acme.ribbon", relax=None)).program
    assert _bytes(q.tap("cl.acme.xy_copy")) == _bytes(q.tap("cl.xy", after="uniform"))
    assert q.result.attributes["acme.xy_copy"] is q.tap("cl.acme.xy_copy")


def test_relax_keeps_attributes_on_their_beads():
    p = _gen(_track(generator="acme.ribbon")).program
    for key in ATTRS:
        ch = "cl." + key
        assert p.tap(ch, after="relax") is p.tap(ch, after="resample"), ch
        assert p.channels[ch] == ("resample", "uniform"), ch
    assert p.channels["raw.acme.grip"] == ("generator",)


# ---------------------------------------------------------------------------------------
# Values on the final points (analytic, against the carried theta)
# ---------------------------------------------------------------------------------------

def _quadrant(theta):
    return np.clip(np.floor((theta + np.pi) / (0.5 * np.pi)), 0, 3).astype(np.int32)


def test_attribute_values_follow_their_policy():
    gen = _gen(_track(generator="acme.ribbon"))
    t = gen._track
    count = t.count.numpy()
    theta = _rows(t.attributes["acme.theta"])
    surf = _rows(t.attributes["acme.surface_id"])
    lane = _rows(t.attributes["acme.lane"])
    grip = _rows(t.attributes["acme.grip"])
    p = gen.program
    raw_step = 2.0 * np.pi / toys.RAW
    for e in range(E):
        m = int(count[e])
        th = theta[e, :m].astype(np.float64)
        assert np.all(th >= -np.float32(np.pi)) and np.all(th <= np.float32(np.pi))
        # one bead step in theta: the loop's angular step between final points (<= 2x)
        bead = 2.0 * np.pi / m * 2.0
        # nearest: the quadrant of theta, except near a quadrant boundary
        q = _quadrant(th)
        dist = np.abs(((th + np.pi) % (0.5 * np.pi) + 0.25 * np.pi) % (0.5 * np.pi)
                      - 0.25 * np.pi)
        far = dist > raw_step + bead
        np.testing.assert_array_equal(surf[e, :m][far], q[far])
        assert set(np.unique(surf[e, :m])) <= {0, 1, 2, 3}
        # linear: cos(theta) to 2e-3
        np.testing.assert_allclose(grip[e, :m], np.cos(th), atol=2e-3)
        # previous: non-decreasing lanes 0..3 that never lead the carried 4 i // N
        # boundary and lag it by less than one raw + one bead step
        ln = lane[e, :m]
        assert set(np.unique(ln)) <= {0, 1, 2, 3}
        theta0 = th[0]
        frac = ((th - theta0) % (2.0 * np.pi)) / (2.0 * np.pi)       # raw index fraction
        want = np.floor(4.0 * frac + 1e-9).astype(np.int32)
        lag = (want - ln) % 4
        assert np.all((lag == 0) | ((lag == 1) & ((frac * 4.0) % 1.0 * 0.5 * np.pi
                                                    < raw_step + bead)))
        # padding
        assert np.all(np.isnan(theta[e, m:])) and np.all(np.isnan(grip[e, m:]))
        assert np.all(surf[e, m:] == -1) and np.all(lane[e, m:] == -1)
        assert np.all(np.isnan(_rows(t.attributes["acme.xy_copy"])[e, m:]))
    # the cl image of raw.acme.hw is not exported, so not in the Track
    assert "acme.hw" not in t.attributes and "cl.acme.hw" in p.channels


# ---------------------------------------------------------------------------------------
# ops.resample_channels on hand-built tables
# ---------------------------------------------------------------------------------------

def _tables(i0, i1, frac, n_src):
    t = ops.ResampleTables.alloc(len(i0), n_src, DEV)
    t.i0.assign(np.asarray(i0, np.int32))
    t.i1.assign(np.asarray(i1, np.int32))
    t.frac.assign(np.asarray(frac, np.float32))
    return t


def test_interp_policies_on_hand_built_tables():
    half = 0.5
    after = np.nextafter(np.float32(0.5), np.float32(1.0))
    t = _tables([0, 0, 0, 0, -1], [1, 1, 1, 1, -1], [0.0, half, after, 1.0, np.nan], 2)
    out = {}
    for dtype, src, pol in ((wp.float32, [1.0, 2.0], "nearest"),
                            (wp.int32, [3, 4], "nearest"),
                            (wp.int32, [3, 4], "previous"),
                            (wp.float32, [1.0, 2.0], "previous"),
                            (wp.float32, [1.0, 3.0], "linear"),
                            (wp.vec2f, [[0.0, 0.0], [2.0, 4.0]], "linear"),
                            (wp.vec3f, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], "nearest")):
        s = wp.array(np.asarray(src, np.float32 if dtype is not wp.int32 else np.int32),
                     dtype=dtype, device=DEV)
        d = wp.zeros(5, dtype=dtype, device=DEV)
        ops.resample_channels(_ctx(), t, [(s, d)], [pol])
        out[(dtype.__name__, pol)] = d.numpy()
    n = out[("float32", "nearest")]
    assert n[1] == 1.0 and n[2] == 2.0 and np.isnan(n[4])       # a tie goes to the start
    assert out[("int32", "nearest")].tolist() == [3, 3, 4, 4, -1]
    assert out[("int32", "previous")].tolist() == [3, 3, 3, 3, -1]
    assert out[("float32", "previous")][:4].tolist() == [1.0] * 4
    assert out[("float32", "linear")][:2].tolist() == [1.0, 2.0]
    np.testing.assert_array_equal(out[("vec2f", "linear")][1], [1.0, 2.0])
    assert np.all(np.isnan(out[("vec2f", "linear")][4]))
    assert np.all(np.isnan(out[("vec3f", "nearest")][4]))
    # angle: the short way round across the seam, output in [-pi, pi]
    rs = np.random.default_rng(0)
    m = 4096
    a = rs.uniform(-np.pi, np.pi, m).astype(np.float32)
    b = rs.uniform(-np.pi, np.pi, m).astype(np.float32)
    a[:8], b[:8] = np.float32(3.1), np.float32(-3.1)
    src = wp.array(np.concatenate([a, b]), dtype=wp.float32, device=DEV)
    f = rs.uniform(0.0, 1.0, m).astype(np.float32)
    ta = _tables(np.arange(m), np.arange(m) + m, f, 2 * m)
    d = wp.zeros(m, dtype=wp.float32, device=DEV)
    ops.resample_channels(_ctx(), ta, [(src, d)], ["angle"])
    got = d.numpy()
    lim = np.float32(np.pi)
    assert np.all(got >= -lim) and np.all(got <= lim)
    short = np.angle(np.exp(1j * (b.astype(np.float64) - a)))
    want = np.angle(np.exp(1j * (a + f * short)))
    np.testing.assert_allclose(np.angle(np.exp(1j * (got - want))), 0.0, atol=1e-5)
    assert np.all(np.abs(got[:8]) > 3.0)                  # across the seam, not via 0


def test_resample_channels_checks_its_arguments_on_the_host():
    t = _tables([0, 1], [1, 0], [0.5, 0.5], 2)
    f2 = wp.zeros(2, dtype=wp.float32, device=DEV)
    f3 = wp.zeros(3, dtype=wp.float32, device=DEV)
    i2 = wp.zeros(2, dtype=wp.int32, device=DEV)
    with pytest.raises(ValueError, match=r"pairs\[0\] src must have shape \(2,\)"):
        ops.resample_channels(_ctx(), t, [(f3, f2)], ["linear"])
    with pytest.raises(ValueError, match=r"pairs\[0\] dst must have shape \(2,\)"):
        ops.resample_channels(_ctx(), t, [(f2, f3)], ["linear"])
    with pytest.raises(TypeError, match=r"pairs\[0\]: dst has dtype int32, src float32"):
        ops.resample_channels(_ctx(), t, [(f2, i2)], ["nearest"])
    with pytest.raises(TypeError, match=r"interp='linear' carries .*got int32"):
        ops.resample_channels(_ctx(), t, [(i2, wp.zeros(2, dtype=wp.int32, device=DEV))],
                              ["linear"])
    with pytest.raises(ValueError, match=r"interp must be one of"):
        ops.resample_channels(_ctx(), t, [(f2, wp.zeros_like(f2))], ["cubic"])
    with pytest.raises(ValueError, match="1 pair\\(s\\) but 2 interp"):
        ops.resample_channels(_ctx(), t, [(f2, wp.zeros_like(f2))], ["linear", "linear"])
    with pytest.raises(ValueError, match="must not"):
        ops.resample_channels(_ctx(), t, [(f2, f2)], ["linear"])
    with pytest.raises(ValueError, match="tables.n_src is 3"):
        ops.resample_constant_spacing(
            _ctx(), wp.zeros(8, dtype=wp.vec2f, device=DEV), 4, 0.1, 2,
            ops.ArcScratch.alloc(2, 4, DEV, compact=False),
            wp.zeros(4, dtype=wp.vec2f, device=DEV), wp.zeros(2, dtype=wp.int32, device=DEV),
            wp.zeros(2, dtype=wp.int32, device=DEV),
            tables=ops.ResampleTables.alloc(4, 3, DEV))
    fresh = ops.ResampleTables.alloc(3, 5, DEV)
    assert fresh.i0.numpy().tolist() == [-1, -1, -1] and np.all(np.isnan(fresh.frac.numpy()))
    assert "n_src=5" in repr(fresh)


def test_the_table_writing_resamples_keep_the_plain_output():
    rs = np.random.default_rng(1)
    E2, n = 3, 24
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    loops = (np.stack([np.cos(th), np.sin(th)], -1)[None]
             * rs.uniform(0.5, 2.0, (E2, 1, 1))).astype(np.float32)
    src = wp.array(loops.reshape(-1, 2), dtype=wp.vec2f, device=DEV)
    sc = ops.ArcScratch.alloc(E2, n, DEV, compact=False)
    count = wp.array(np.array([n, 10, 0], np.int32), device=DEV)
    a, b = (wp.zeros(E2 * n, dtype=wp.vec2f, device=DEV) for _ in range(2))
    t = ops.ResampleTables.alloc(E2 * n, E2 * n, DEV)
    ops.resample_uniform(_ctx(), src, n, sc, a, count=count)
    ops.resample_uniform(_ctx(), src, n, sc, b, count=count, tables=t)
    assert _bytes(a) == _bytes(b)
    again = wp.zeros_like(a)
    ops.resample_channels(_ctx(), t, [(src, again)], ["linear"])
    assert _bytes(again) == _bytes(b)
    i0 = t.i0.numpy().reshape(E2, n)
    assert np.all(i0[2] == -1) and np.all(i0[1, 10:] == -1) and np.all(i0[0] >= 0)
    cs_out = wp.zeros(E2 * 40, dtype=wp.vec2f, device=DEV)
    cs_out2 = wp.zeros_like(cs_out)
    cnt, tr = (wp.zeros(E2, dtype=wp.int32, device=DEV) for _ in range(2))
    tc = ops.ResampleTables.alloc(E2 * 40, E2 * n, DEV)
    ops.resample_constant_spacing(_ctx(), src, n, 0.2, 40, sc, cs_out, cnt, tr)
    ops.resample_constant_spacing(_ctx(), src, n, 0.2, 40, sc, cs_out2, cnt, tr, tables=tc)
    assert _bytes(cs_out) == _bytes(cs_out2)
    carried = wp.zeros_like(cs_out)
    ops.resample_channels(_ctx(), tc, [(src, carried)], ["linear"])
    assert _bytes(carried) == _bytes(cs_out2)


# ---------------------------------------------------------------------------------------
# interp= / export= rules
# ---------------------------------------------------------------------------------------

def _gen_toy(writes, body, provides=frozenset(), name="acme.bad"):
    class Bad(Stage):
        kind = "generator"
        output = "centerline"

        def capacity(self, p, layout):
            return {"raw": 16}

        def build(self, b, p):
            b.write("xy", wp.vec2f, per="raw")
            body(b)
            return lambda ctx: None

    Bad.name = name
    Bad.writes = {"xy": "raw.xy", **writes}
    Bad.provides = provides
    registry.add(Bad)
    return name


@pytest.mark.parametrize("writes,body,exc,match", [
    ({"s": "raw.acme.s"}, lambda b: b.write("s", wp.int32, interp="linear"), TypeError,
     r"an int32 attribute is carried with interp='nearest' or 'previous'"),
    ({"s": "raw.acme.s"}, lambda b: b.write("s", wp.vec2f, interp="angle"), TypeError,
     r"interp='angle' carries float32 radians, got vec2f"),
    ({"s": "raw.acme.s"}, lambda b: b.write("s", wp.float32), TypeError,
     r"its writer passes interp="),
    ({"s": "raw.acme.s"}, lambda b: b.write("s", wp.uint8, interp="nearest"), TypeError,
     r"a carried attribute is float32, int32, vec2f or vec3f, got uint8"),
    ({"s": "raw.acme.s"}, lambda b: b.write("s", wp.float32, interp="cubic"), ValueError,
     r"interp must be one of"),
    ({"s": "raw.acme.s"}, lambda b: b.write("s", wp.float32, interp="linear",
                                            export="surface"), ValueError,
     r"export='surface': the key of raw.acme.s is 'acme.s' \(renaming is not supported\)"),
    ({"s": "raw.acme.s"}, lambda b: b.write("s", wp.float32, interp="linear", export=3),
     TypeError, r"export= must be True, False or the channel's key"),
    ({"z": "raw.z"}, lambda b: b.write("z", export=True), ValueError,
     r"raw.z reaches Track.center.z; export= is for plugin channels"),
    ({"z": "raw.z"}, lambda b: b.write("z", interp="nearest"), TypeError,
     r"raw channel 'raw.z' is carried with interp='linear'"),
    ({"s": "raw.acme.s"}, lambda b: b.write("s", wp.float32, per="cl", interp="linear"),
     TypeError, r"is per='raw' \(fixed by its raw. prefix\)"),
])
def test_raw_attribute_write_rules(writes, body, exc, match):
    provides = frozenset({"z"}) if "z" in writes else frozenset()
    name = _gen_toy(writes, body, provides)
    with pytest.raises(exc, match=match):
        TrackGenerator(_track(generator=name), _rng())


def test_stage_written_channel_rules():
    class W(Stage):
        kind = "elevation"
        name = "acme.w"
        reads = {"count": "cl.count"}
        writes = {"z": "cl.z", "x": "cl.acme.x"}
        mode = "interp"

        def build(self, b, p):
            b.write("z", wp.float32, per="cl")
            if self.mode == "interp":
                b.write("x", wp.float32, per="cl", interp="linear")
            elif self.mode == "env":
                b.write("x", wp.float32, export=True)
            else:
                b.write("x", wp.float32, per="cl", export=True)
            return lambda ctx: None

    class R(Stage):
        kind = "validity"
        name = "acme.r"
        reads = {"x": "cl.acme.x", "fail": "valid.fail"}
        writes = {"x": "cl.acme.x", "fail": "valid.fail"}

        def build(self, b, p):
            b.read("x")
            b.write("x", export=True)
            b.write("fail", wp.int32)
            b.fail_bit()
            return lambda ctx: None

    registry.add(W)
    with pytest.raises(ValueError, match=r"interp= is declared by the generator's raw "
                                         r"write \(raw.<pkg>.<name>\); a stage-written "
                                         r"channel \('cl.acme.x'\) is not resampled"):
        build(_track(elevation="acme.w"), seeds=1)
    W.mode = "env"
    with pytest.raises(ValueError, match=r"export= needs a per-point channel: per='cl' "
                                         r"\(got the default per='env'\)"):
        build(_track(elevation="acme.w"), seeds=1)
    W.mode = "export"
    registry.add(R)
    with pytest.raises(ValueError, match=r"export= belongs to the channel's first writer"):
        build(_track(elevation="acme.w", checks=("default", "acme.r")), seeds=1)
    # A stage-written exported channel on the final points is an attribute.
    p = build(_track(elevation="acme.w"), seeds=1)
    assert set(p.result.attributes) == {"acme.x"}
    assert p.attributes["acme.x"] == AttributeInfo(
        key="acme.x", channel="cl.acme.x", dtype=wp.float32, interp=None,
        padding=p.attributes["acme.x"].padding, domain="cl")


def test_only_a_generator_writes_raw_attributes_and_plugin_domains_are_checked():
    class Raw(Stage):
        kind = "width"
        name = "acme.raw"
        writes = {"half_width": "cl.half_width", "r": "raw.acme.r"}

        def build(self, b, p):
            return lambda ctx: None

    with pytest.raises(ValueError, match=r"raw.<package>.<name> attributes are written by "
                                         r"a centerline generator only"):
        registry.add(Raw)

    class Anch(Stage):
        kind = "relax"
        name = "acme.anch"
        reads = {"xy": "cl.xy"}
        writes = {"xy": "cl.xy", "a": "anchors.acme.a"}

        def build(self, b, p):
            return lambda ctx: None

    with pytest.raises(ValueError, match=r"a relax stage may write \['cl.xy'\] and plugin "
                                         r"channels 'cl.<package>.<name>' or "
                                         r"'gates.<package>.<name>'"):
        registry.add(Anch)


# ---------------------------------------------------------------------------------------
# Track.attributes: stable pointers, clone, masked commit, poison, out=, launch counts
# ---------------------------------------------------------------------------------------

def test_track_attributes_are_stable_and_described():
    gen = _gen(_track(generator="acme.ribbon"))
    t = gen._track
    ptrs = {k: a.ptr for k, a in t.attributes.items()}
    for _ in range(3):
        gen.generate(advance=True)
        assert {k: a.ptr for k, a in gen._track.attributes.items()} == ptrs
    info = gen.program.attributes
    assert set(info) == set(ATTRS)
    for key, (dtype, interp) in ATTRS.items():
        arr = t.attributes[key]
        assert arr.dtype is dtype and arr.shape == (E * gen.n_max,), key
        assert info[key].dtype is dtype and info[key].interp == interp
        assert info[key].channel == "cl." + key and info[key].domain == "cl"
        assert (info[key].padding == -1) if dtype is wp.int32 else np.isnan(
            info[key].padding)
        assert gen.program.tap("cl." + key) is arr
    with pytest.raises(TypeError):
        t.attributes["x"] = t.center                         # read-only mapping
    c = t.clone()
    assert set(c.attributes) == set(ATTRS)
    for k in ATTRS:
        assert c.attributes[k] is not t.attributes[k]
        assert _bytes(c.attributes[k]) == _bytes(t.attributes[k])


def _perturbed(gen):
    """Wrap the uniform resample (test-only): after it, add a per-generation offset to
    its cl.xy and to every exported attribute's final version (the Track's own array),
    so a recomputed unmasked env differs from its held bytes (the PR10b method)."""
    p = gen.program
    i = list(p._stage_ids).index("uniform")
    base = p._enqueues[i]
    counter = [0]
    targets = [p.tap("cl.xy", after="uniform"), *gen._track.attributes.values(),
               p.tap("cl.half_width")]

    def enqueue(ctx):
        base(ctx)
        counter[0] += 1
        for a in targets:
            h = a.numpy()
            if h.dtype.kind == "i":
                h = h + counter[0]
            else:
                h = h + np.float32(1e-3 * counter[0])
            a.assign(h)

    p._enqueues[i] = enqueue


@pytest.mark.parametrize("on_invalid", ["keep", "poison"])
def test_masked_regeneration_keeps_unmasked_attributes_byte_for_byte(on_invalid):
    gen = _gen(_track(generator="acme.ribbon", on_invalid=on_invalid,
                      width=StageRef("width", "from_channel", ports={"src": "cl.acme.hw"})))
    amp = np.full(E, 0.03, np.float32)
    amp[[1, 4]] = 0.2                                # pinches below 0: w_floor fails
    gen.params["generator"].set("hw_amp", wp.array(amp, dtype=wp.float32, device=DEV))
    gen.generate()
    _perturbed(gen)
    names = ["half_width", *[f"attributes/{k}" for k in ATTRS], "center", "outer"]

    def snap():
        t = gen._track
        out = {"half_width": t.half_width.numpy().copy(), "center": t.center.numpy().copy(),
               "outer": t.outer.numpy().copy()}
        out.update({f"attributes/{k}": a.numpy().copy() for k, a in t.attributes.items()})
        return out

    mask = np.array([1, 0, 1, 0, 1, 0], np.int32)
    before = snap()
    gen.generate(advance=True, mask=wp.array(mask, dtype=wp.int32, device=DEV))
    after = snap()
    for name in names:
        rows_b, rows_a = (x[name].reshape(E, -1) for x in (before, after))
        for e in np.flatnonzero(mask == 0):
            assert rows_a[e].tobytes() == rows_b[e].tobytes(), (name, e)
    # an all-zero mask is an exact no-op
    zero = wp.zeros(E, dtype=wp.int32, device=DEV)
    gen.generate(advance=True, mask=zero)
    for name, v in snap().items():
        assert v.tobytes() == after[name].tobytes(), name
    if on_invalid == "poison":
        fail = gen._track.fail_mask.numpy()
        assert fail[1] and fail[4]
        for e in (1, 4):
            assert np.all(gen._track.half_width.numpy().reshape(E, -1)[e] != 0) and \
                np.all(np.isnan(gen._track.half_width.numpy().reshape(E, -1)[e]))
            for k, a in gen._track.attributes.items():
                row = a.numpy().reshape(E, -1, *a.numpy().shape[1:])[e]
                assert np.all(row == -1) if a.dtype is wp.int32 else np.all(np.isnan(row))


def test_poison_pads_every_attribute_row_of_an_invalid_env():
    spec_keep = _track(generator="acme.ribbon", on_invalid="keep",
                       width=StageRef("width", "from_channel", ports={"src": "cl.acme.hw"}))
    keep, poison = _gen(spec_keep), _gen(_track(
        generator="acme.ribbon", on_invalid="poison",
        width=StageRef("width", "from_channel", ports={"src": "cl.acme.hw"})))
    amp = wp.array(np.array([0.03, 0.2, 0.03, 0.2, 0.03, 0.03], np.float32), device=DEV)
    for g in (keep, poison):
        g.params["generator"].set("hw_amp", amp)
        g.generate()
    bad = keep._track.fail_mask.numpy() != 0
    assert bad.tolist() == [False, True, False, True, False, False]
    for k in ATTRS:
        a, b = keep._track.attributes[k].numpy(), poison._track.attributes[k].numpy()
        ra, rb = a.reshape(E, -1, *a.shape[1:]), b.reshape(E, -1, *b.shape[1:])
        assert ra[~bad].tobytes() == rb[~bad].tobytes(), k
        assert np.all(rb[bad] == -1) if a.dtype == np.int32 else np.all(np.isnan(rb[bad]))
    hw = poison._track.half_width.numpy().reshape(E, -1)
    assert np.all(np.isnan(hw[bad]))
    assert keep._track.half_width.numpy().reshape(E, -1)[~bad].tobytes() == hw[~bad].tobytes()


def test_attribute_commit_launch_counts():
    def count(spec, mask=None):
        g = _gen(spec)
        with trace_launches() as t:
            if mask is None:
                g.enqueue()
            else:
                g.enqueue(advance=True, mask=wp.array(mask, dtype=wp.int32, device=DEV))
        return len(t)

    n = len(ATTRS)
    m = [1, 0, 1, 0, 1, 0]
    keep = _track(generator="acme.ribbon", on_invalid="keep")
    plain = _track(generator="acme.ribbon_plain", on_invalid="keep")
    poison = _track(generator="acme.ribbon", on_invalid="poison")
    poison_plain = _track(generator="acme.ribbon_plain", on_invalid="poison")
    # ribbon vs ribbon_plain: its attribute kernel + one carry per family per resampler
    carries = 1 + 2 * 6
    assert count(keep) - count(plain) == carries                            # keep: +0
    assert count(poison) - count(poison_plain) == carries + n               # poison: +n
    assert count(keep, m) - count(plain, m) == carries + 2 * n              # masked: +2n


# ---------------------------------------------------------------------------------------
# CPU graph capture: Programs that carry and commit attributes of several dtypes
# ---------------------------------------------------------------------------------------

_FROM_CHANNEL = StageRef("width", "from_channel", ports={"src": "cl.acme.hw"})
_CAPTURE_CASES = {
    "track ribbon from_channel poison": lambda: _track(
        generator="acme.ribbon", width=_FROM_CHANNEL, on_invalid="poison"),
    "track ribbon constant": lambda: _track(generator="acme.ribbon"),
    "gates ribbon stations poison": lambda: presets.gates(
        num_envs=E, device=DEV, generator="acme.ribbon", gates=8, on_invalid="poison"),
    "gates tagged poison": lambda: presets.gates(
        num_envs=E, device=DEV, generator="acme.tagged", on_invalid="poison"),
}


@pytest.mark.parametrize("case", list(_CAPTURE_CASES))
def test_attribute_programs_replay_from_a_cpu_graph_capture(case):
    """A Program launching carries and attribute commits of several dtypes back to back
    (f32, i32, vec2f), recorded into ``wp.ScopedCapture("cpu")``: the replay equals an
    eager twin, plain (outputs poisoned first) and masked (``inputs["regen_mask"]``
    written after recording), and writes nothing outside its own arrays (the
    build-time ``width.constant`` version keeps ``hw`` everywhere)."""
    from track_gen import GateGenerator  # noqa: PLC0415
    from track_gen.testing.capture import poison_all, result_arrays  # noqa: PLC0415

    spec = _CAPTURE_CASES[case]()
    cls = GateGenerator if case.startswith("gates") else TrackGenerator
    gen, twin = cls(spec, _rng()), cls(spec, _rng())
    if case.startswith("track ribbon from_channel"):
        amp = wp.array(np.array([0.03, 0.2, 0.03, 0.2, 0.03, 0.03], np.float32), device=DEV)
        for g in (gen, twin):
            g.params["generator"].set("hw_amp", amp)     # envs 1, 3 fail w_floor: poison
    prog, ref = gen.program, twin.program

    def host(result) -> dict:
        return {k: a.numpy().copy() for k, a in result_arrays(result).items()}

    def assert_equal(got: dict, want: dict, label: str) -> None:
        assert set(got) == set(want)
        assert any(k.startswith("attributes/") for k in got)
        for k in want:
            assert np.array_equal(got[k], want[k], equal_nan=True), (label, k)

    with wp.ScopedCapture(device=DEV) as plain:
        prog.enqueue(_ctx())
    poison_all(prog.result)
    wp.capture_launch(plain.graph)
    ref.enqueue(_ctx())
    assert_equal(host(prog.result), host(ref.result), "plain")

    mask = prog.inputs["regen_mask"]
    with wp.ScopedCapture(device=DEV) as masked:
        prog.enqueue(_ctx(), advance=True, mask=mask)
    m = np.array([1, 0, 0, 1, 0, 1], np.int32)
    wp.copy(mask, wp.array(m, dtype=wp.int32, device=DEV))
    wp.capture_launch(masked.graph)
    ref.enqueue(_ctx(), advance=True, mask=wp.array(m, dtype=wp.int32, device=DEV))
    assert_equal(host(prog.result), host(ref.result), "masked")
    if case == "track ribbon constant":
        hw = prog.tap("cl.half_width", after="width").numpy()
        assert np.all(hw == np.float32(0.1))


def test_out_with_and_without_attributes():
    spec = _track(generator="acme.ribbon")
    layout = build(spec, seeds=1).layout
    # Attributes may be absent, but the width array is required and bound in place.
    out = _alloc_track(layout)
    p = build(spec, seeds=1, out=out)
    assert p.result is out and dict(out.attributes) == {}
    assert set(p.attributes) == set(ATTRS)
    assert p.tap("track.half_width") is out.half_width
    assert np.isfinite(out.half_width.numpy()).any()
    # out= with an attribute: bound (the final version IS the caller's array).
    out2 = _alloc_track(layout)
    mine = wp.zeros(layout.num_envs * layout.stride, dtype=wp.float32, device=DEV)
    out2.attributes = {"acme.grip": mine}
    p2 = build(spec, seeds=1, out=out2)
    assert p2.tap("cl.acme.grip") is mine and out2.attributes == {"acme.grip": mine}
    assert _bytes(mine) == _bytes(p.tap("cl.acme.grip"))
    # A key this pipeline does not export: refused on a whole result (roadmap PR13b pads
    # it only on a view of a larger result, a mixed batch: tests/test_results.py).
    out3 = _alloc_track(layout)
    out3.attributes = {"acme.nope": mine}
    with pytest.raises(ValueError, match=r"out.attributes\['acme.nope'\]: this pipeline does "
                                         r"not export 'acme.nope'; exported: \["):
        build(spec, seeds=1, out=out3)
    out4 = _alloc_track(layout)
    out4.attributes = {"acme.grip": wp.zeros(3, dtype=wp.float32, device=DEV)}
    with pytest.raises(ValueError, match=r"out.attributes\['acme.grip'\]: expected a "
                                         r"contiguous \[\d+\] array"):
        build(spec, seeds=1, out=out4)
    out5 = _alloc_track(layout)
    out5.attributes = {"acme.surface_id": mine}               # float32 for an int32 family
    with pytest.raises(ValueError, match=r"out\.attributes\['acme\.surface_id'\] is "
                                         r"float32; this pipeline exports "
                                         r"'acme\.surface_id' as int32"):
        build(spec, seeds=1, out=out5)


# ---------------------------------------------------------------------------------------
# A generator that provides its altitude
# ---------------------------------------------------------------------------------------

def test_a_z_provider_reaches_the_track_through_the_carriers():
    spec = _track(generator="acme.helix_loop")
    assert "elevation" not in spec.ids
    assert (spec.stages[-1].kind, spec.stages[-1].name) == ("lift", "elevated")
    gen = _gen(spec)
    p = gen.program
    assert p.channels["cl.z"] == ("resample", "uniform")
    raw = p.tap("raw.z").numpy()
    tables = p._scratch["resample"]
    i0, i1, fr = (tables[k].numpy() for k in ("tables.i0", "tables.i1", "tables.frac"))
    real = i0 >= 0
    want = raw[i0[real]] + fr[real] * (raw[i1[real]] - raw[i0[real]])
    np.testing.assert_allclose(p.tap("cl.z", after="resample").numpy()[real], want,
                               rtol=1e-6, atol=1e-7)
    z = gen._track.center.numpy()[:, 2]
    zc = p.tap("cl.z").numpy()
    fin = np.isfinite(zc)
    np.testing.assert_array_equal(z[fin], zc[fin])
    assert np.ptp(z[fin]) > 0.3
    with pytest.raises(ValueError, match=r"presets.track\(elevation='noise'\): generator "
                                         r"'acme.helix_loop' provides its own altitudes "
                                         r"\(z\); leave elevation at its default"):
        _track(generator="acme.helix_loop", elevation="noise")
    with pytest.raises(ValueError, match=r"z_base=0.2.*provides its own altitudes"):
        _track(generator="acme.helix_loop", z_base=0.2)
    hand = spec.insert_after("offset", StageRef("elevation", "flat"))
    with pytest.raises(PlanError, match=r"writes 'cl.z', already written by stage "
                                        r"'uniform'.*generator 'acme.helix_loop' provides z "
                                        r"\(raw.z -> cl.z\): drop the elevation stage"):
        build(hand, seeds=1)
    with pytest.raises(PlanError, match=r"lift.flat\) ignores cl.z written by"):
        build(spec.replace("lift", StageRef("lift", "flat")), seeds=1)
