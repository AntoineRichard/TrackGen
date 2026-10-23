"""``generator/external`` (roadmap PR13b): ingestion of caller-provided centerlines through
the public plugin surface (``StageBuilder.input(per="raw")``, ``Stage.declare``,
``Stage.seeded``, ``Stage.preferred``), the declaration routing, the analytic and
padding-exactness proofs, the device validity, the host loader and the kit. CPU; the
caller-graph proofs are tests/test_results_cuda.py."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

from tests._plugin_toys import DECLARED, register_declared
from tests._results_helpers import circle_rows, ellipse_rows, ramanujan
from track_gen import GateGenerator, TrackGenConfig, TrackGenerator, plugin
from track_gen.exec import LaunchContext
from track_gen.pipeline import InputInfo, StageRef, build, load_centerlines, presets
from track_gen.plugin import registry
from track_gen.stages import external as ext
from track_gen.testing import check_generator
from track_gen.testing.capture import result_arrays
from track_gen.testing.trace import trace_launches

DEV = "cpu"
E = 4
N = 256
FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length", "valid",
          "count", "winding", "half_width", "fail_mask", "truncated")


def _spec(n=E, num_points=N, **kw):
    return presets.track(num_envs=n, device=DEV, generator="external",
                         num_points=num_points, **kw)


def _gen(n=E, num_points=N, **kw):
    return TrackGenerator(_spec(n, num_points, **kw), seeds=0)


def _bytes(a) -> bytes:
    return np.ascontiguousarray(a.numpy()).tobytes()


# ---------------------------------------------------------------------------------------
# Registration, description, Params
# ---------------------------------------------------------------------------------------

def test_registration_and_describe():
    info = registry.describe("generator", "external")
    assert info.builtin and info.seeded is False and info.capturable
    assert dict(info.preferred) == {"relax": "none"}
    assert dict(info.writes) == {"xy": "raw.xy", "valid": "gen.valid"}
    assert info.provides == frozenset()
    assert registry.describe("generator", "bezier").seeded is True
    props = info.params_schema["properties"]
    assert set(props) == {"with_z", "with_half_width", "attributes"}


@pytest.mark.parametrize("attrs,match", [
    ((("acme",  "float32", "linear"),), r"key 'acme' must be '<package>\.<name>'"),
    ((("acme.t", "float64", "linear"),), r"dtype must be one of"),
    ((("acme.t", "int32", "linear"),), r"an int32 attribute is carried with "
                                       r"interp='nearest' or 'previous'"),
    ((("acme.t", "vec2f", "angle"),), r"interp='angle' carries float32 radians"),
    ((("acme.t", "float32", "cubic"),), r"interp must be one of"),
    ((("acme.t", "float32", "linear"), ("acme.t", "int32", "nearest")),
     r"attribute 'acme\.t' is given twice"),
    ((("acme.t", "float32"),), r"each attribute is a \(key, dtype, interp\) triple"),
])
def test_params_validation(attrs, match):
    with pytest.raises(ValueError, match=match):
        build(_spec(generator_params={"attributes": attrs}))


def test_params_round_trip_through_json():
    spec = _spec(generator_params={"with_z": True,
                                   "attributes": [["acme.tag", "int32", "nearest"]]})
    import json  # noqa: PLC0415

    from track_gen.pipeline import PipelineSpec
    back = PipelineSpec.from_dict(spec.to_dict())
    # The spec itself round-trips (lists and tuples are both JSON arrays: a params value
    # is canonicalised to tuples), so specs compare and hash by value after JSON.
    assert back == spec
    assert PipelineSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    p = build(back, seeds=0)
    assert p.spec.stages[0].params.attributes == (("acme.tag", "int32", "nearest"),)
    assert hash(p.spec.stages[0].params)


# ---------------------------------------------------------------------------------------
# Stage.declare routing (acme.declared and external)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", [
    (DECLARED, {"tagged": True}),
    ("external", {"with_z": True, "attributes": (("acme.tag", "int32", "nearest"),)}),
])
def test_declared_channels_reach_every_result(name, params):
    register_declared(registry)
    key = "acme.tag"
    track = presets.track(num_envs=E, device=DEV, generator=name, generator_params=params)
    assert "elevation" not in track.ids and track.stages[-1].name == "elevated"
    t = TrackGenerator(track, seeds=1)
    if name == "external":
        rows = circle_rows(E, N, 1.0)
        z = [0.3 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
             for _ in range(E)]
        tag = [(4 * np.arange(N)) // N for _ in range(E)]
        load_centerlines(t, rows, z=z, attributes={key: tag})
        t.generate()
    res = t.program.result
    assert key in res.attributes and np.nanmax(res.center.numpy()[:, 2]) > 0.2
    vals = res.attributes[key].numpy().reshape(E, -1)
    cnt = res.count.numpy()
    assert set(np.unique(vals[0, :cnt[0]]).tolist()) <= {0, 1, 2, 3}
    assert (vals[0, cnt[0]:] == -1).all()
    for source in ("stations", "relaxed_stations"):
        gspec = presets.gates(num_envs=E, device=DEV, generator=name,
                              generator_params=params, source=source)
        assert "carry_attributes" in gspec.ids and "carry_z" in gspec.ids
        g = GateGenerator(gspec, seeds=1)
        if name == "external":
            load_centerlines(g, rows, z=z, attributes={key: tag})
            g.generate()
        gr = g.program.result
        assert key in gr.attributes, source
        assert np.nanmax(gr.position.numpy()[:, 2]) > 0.2, source
    cfg = TrackGenerator(TrackGenConfig(generator=name, num_envs=E,
                                        generator_params=params), seeds=1)
    assert key in cfg.program.result.attributes
    assert cfg.program.spec.stages[-1].name == "elevated"


def test_declare_is_refused_on_other_kinds_and_checked_at_bind():
    class R(plugin.Stage):
        kind = "relax"
        name = "acme.declares"
        reads = {"xy": "cl.xy"}
        writes = {"xy": "cl.xy"}

        @classmethod
        def declare(cls, params):
            return dict(cls.writes), frozenset()

        def build(self, b, p):
            return lambda ctx: None

    with pytest.raises(ValueError, match=r"Stage R\.declare\(\) is read for centerline "
                                         r"generators only in 0\.x"):
        registry.add(R)

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(plugin.Params):
        bad: bool = plugin.static(False)

    class G(plugin.Stage):
        kind = "generator"
        name = "acme.bad_declare"
        output = "centerline"
        Params = P
        writes = {"xy": "raw.xy"}

        @classmethod
        def declare(cls, params):
            if params.bad:
                return {"xy": "raw.xy", "z": "raw.z"}, frozenset()     # z not provided
            return dict(cls.writes), frozenset()

        def build(self, b, p):
            xy = b.write("xy", wp.vec2f, per="raw")
            return lambda ctx: ctx.launch(ext._external_circle_k, xy.shape[0],
                                          [b.raw, 1.0, xy])

    registry.add(G)
    build(presets.track(num_envs=E, device=DEV, generator="acme.bad_declare"))
    with pytest.raises(ValueError, match=r"Stage G\.declare\(\): provides \[\] must name "
                                         r"exactly the catalogue attributes it writes "
                                         r"\(\['z'\]"):
        build(presets.track(num_envs=E, device=DEV, generator="acme.bad_declare",
                            generator_params={"bad": True}))


def test_preferred_relax_and_seeded_registration_rules():
    def build_(self, b, p):
        return lambda ctx: None

    bad = type("B", (plugin.Stage,), dict(kind="generator", name="acme.b",
                                          output="centerline", writes={"xy": "raw.xy"},
                                          preferred={"relax": "auto"}, build=build_))
    with pytest.raises(ValueError, match=r"preferred\['relax'\] must be a relax stage name "
                                         r"or 'none', got 'auto'"):
        registry.add(bad)
    bad2 = type("B2", (plugin.Stage,), dict(kind="generator", name="acme.b2",
                                            output="centerline", writes={"xy": "raw.xy"},
                                            preferred={"order": "raw"}, build=build_))
    with pytest.raises(ValueError, match=r"unknown keys \['order'\]; a centerline "
                                         r"generator's key is 'relax'"):
        registry.add(bad2)
    bad3 = type("B3", (plugin.Stage,), dict(kind="generator", name="acme.b3",
                                            output="centerline", writes={"xy": "raw.xy"},
                                            seeded="no", build=build_))
    with pytest.raises(TypeError, match=r"Stage B3\.seeded must be a bool"):
        registry.add(bad3)
    auto = type("A", (plugin.Stage,), dict(kind="relax", name="auto",
                                           reads={"xy": "cl.xy"}, writes={"xy": "cl.xy"},
                                           build=build_))
    with pytest.raises(ValueError, match=r"the relax name 'auto' is reserved"):
        registry.add(auto)


def test_relax_auto_resolves_per_generator():
    def relax(spec):
        return spec.stages[spec.ids.index("relax")]

    assert relax(presets.track(num_envs=E, device=DEV)).name == "xpbd"     # built-ins
    assert relax(_spec()) == StageRef("relax", "none")                     # preferred
    assert relax(_spec(relax="xpbd")).name == "xpbd"                       # explicit
    assert relax(presets.gates(num_envs=E, device=DEV, generator="external",
                               source="relaxed_stations")).name == "none"


# ---------------------------------------------------------------------------------------
# Analytic and exactness proofs
# ---------------------------------------------------------------------------------------

def test_circles_match_the_analytic_expectation():
    R = 1.25
    gen = _gen()
    load_centerlines(gen, circle_rows(E, N, R))
    res = gen.generate()
    c = res.center.numpy().reshape(E, -1, 3)
    cnt = res.count.numpy()
    L = res.length.numpy()
    assert (res.valid.numpy() == 1).all()
    for e in range(E):
        r = np.linalg.norm(c[e, :cnt[e], :2], axis=1)
        # inside the chord sagitta of the coarsest polygon the points lie on (the
        # resampled loop's count[e] vertices; the N input points are finer)
        assert np.abs(r - R).max() <= R * (1 - math.cos(math.pi / cnt[e])) + 1e-5
        assert abs(L[e] - 2 * math.pi * R) / (2 * math.pi * R) < 1e-3
        assert cnt[e] == math.floor(float(L[e]) / 0.06) + 1
    assert np.isnan(c[0, cnt[0]:, :2]).all()           # padding (z: the flat z_base)
    # relax="xpbd" re-enables relax: the geometry may move, the result stays valid
    relaxed = _gen(relax="xpbd")
    load_centerlines(relaxed, circle_rows(E, N, R))
    assert relaxed.generate().valid.numpy().all()


def test_an_ellipse_length_matches_ramanujan():
    a, b = 1.5, 0.8
    gen = _gen()
    load_centerlines(gen, ellipse_rows(E, N, a, b))
    L = gen.generate().length.numpy()
    assert np.all(np.abs(L - ramanujan(a, b)) / ramanujan(a, b) < 1e-3)


def test_padding_is_exact():
    ms = [3, 97, 150, 200, N]
    rows = [ellipse_rows(1, m, 1.0 + 0.1 * k, 0.7, phase=0.3 * k)[0]
            for k, m in enumerate(ms)]
    gen = _gen(n=len(ms))
    load_centerlines(gen, rows)
    res = gen.generate()
    for e, m in enumerate(ms):
        one = _gen(n=1, num_points=m)
        load_centerlines(one, [rows[e]])
        ref = one.generate()
        for f in FIELDS:
            arr = getattr(res, f).numpy()
            per = arr.shape[0] // len(ms)
            assert arr[e * per:(e + 1) * per].tobytes() == _bytes(getattr(ref, f)), (m, f)
    g = GateGenerator(presets.gates(num_envs=len(ms) - 1, device=DEV, generator="external",
                                    source="stations", num_points=N), seeds=0)
    load_centerlines(g, rows[1:])
    gr = g.generate()
    for e, m in enumerate(ms[1:]):
        one = GateGenerator(presets.gates(num_envs=1, device=DEV, generator="external",
                                          source="stations", num_points=m), seeds=0)
        load_centerlines(one, [rows[e + 1]])
        ref = one.generate()
        for f, arr in result_arrays(gr).items():
            a = arr.numpy()
            per = a.shape[0] // (len(ms) - 1)
            assert a[e * per:(e + 1) * per].tobytes() == _bytes(result_arrays(ref)[f]), \
                (m, f)


@wp.kernel
def _nan_first_k(xy: wp.array(dtype=wp.vec2f)):
    xy[0] = wp.vec2f(wp.nan, wp.nan)


class _Interleaved(LaunchContext):
    """Writes NaN into ``centerline.xy[0]`` right before the validity kernel: an unordered
    writer between the snapshot (the pads) and the validity."""

    __slots__ = ()
    target = None

    def launch(self, kernel, dim, inputs=(), outputs=(), **kw):
        if kernel is ext._external_valid_k:
            super().launch(_nan_first_k, 1, [_Interleaved.target])
        super().launch(kernel, dim, inputs, outputs, **kw)


def test_the_inputs_are_copied_first_and_validity_reads_the_copies():
    gen = _gen()
    load_centerlines(gen, circle_rows(E, N, 1.0))
    with trace_launches() as tr:
        gen.generate()
    names = [r.kernel.rsplit(".", 1)[-1] for r in tr]
    i = names.index("_external_pad_vec2f_k")
    assert names[i + 1] == "_external_valid_k"
    _Interleaved.target = gen.program.inputs["centerline.xy"]
    gen.program.enqueue(_Interleaved(DEV, None))
    assert np.isnan(gen.program.inputs["centerline.xy"].numpy()[0]).all()
    assert gen.program.result.valid.numpy()[0] == 1          # the copy was finite
    gen.generate()                                           # the next generation sees it
    assert gen.program.result.valid.numpy()[0] == 0
    z = _gen(generator_params={"with_z": True, "with_half_width": True,
                               "attributes": (("acme.t", "float32", "linear"),)})
    with trace_launches() as tr:
        z.generate()
    names = [r.kernel.rsplit(".", 1)[-1] for r in tr]
    j = names.index("_external_pad_vec2f_k")
    assert names[j:j + 5] == ["_external_pad_vec2f_k", "_external_pad_f32_k",
                              "_external_pad_f32_k", "_external_pad_f32_k",
                              "_external_valid_k"]
    assert names[:2] == ["_effective_seed_k", "_latch_all_i32_k"]   # the count latch


def test_with_z_with_half_width_and_an_int_attribute():
    params = {"with_z": True, "with_half_width": True,
              "attributes": (("acme.lane", "int32", "nearest"),)}
    spec = _spec(generator_params=params)
    assert "elevation" not in spec.ids and "width" not in spec.ids
    gen = TrackGenerator(spec, seeds=0)
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    z = [0.5 * np.sin(2 * t) + 0.6] * E
    hw = [0.08 + 0.03 * np.cos(3 * t)] * E
    lane = [(np.arange(N) * 5) // N] * E
    load_centerlines(gen, circle_rows(E, N, 1.2), z=z, half_width=hw,
                     attributes={"acme.lane": lane})
    res = gen.generate()
    assert res.valid.numpy().all()
    c = res.center.numpy().reshape(E, -1, 3)
    cnt = res.count.numpy()
    assert c[0, :cnt[0], 2].min() > 0.09 and c[0, :cnt[0], 2].max() < 1.11
    w = res.half_width.numpy().reshape(E, -1)[0, :cnt[0]]
    assert 0.049 < w.min() < 0.06 and 0.10 < w.max() < 0.111
    lab = res.attributes["acme.lane"].numpy().reshape(E, -1)[0]
    assert set(np.unique(lab[:cnt[0]]).tolist()) == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------------------
# Device validity
# ---------------------------------------------------------------------------------------

def test_invalid_inputs_fail_bit_zero_and_are_poisoned():
    params = {"with_z": True, "with_half_width": True}
    gen = _gen(n=6, generator_params=params, on_invalid="poison")
    load_centerlines(gen, circle_rows(6, N, 1.0), z=[np.zeros(N)] * 6,
                     half_width=[np.full(N, 0.1)] * 6)
    inp = gen.program.inputs
    xy = inp["centerline.xy"].numpy()
    xy[2 * N + 17] = np.nan                          # env 2: a NaN point
    inp["centerline.xy"].assign(xy)
    z = inp["centerline.z"].numpy()
    z[3 * N + 5] = np.nan                            # env 3: a NaN altitude
    inp["centerline.z"].assign(z)
    hw = inp["centerline.half_width"].numpy()
    hw[4 * N + 9] = 0.0                              # env 4: a zero half-width
    inp["centerline.half_width"].assign(hw)
    inp["centerline.count"].assign(np.array([2, N + 1, N, N, N, N], np.int32))
    res = gen.generate()
    assert res.valid.numpy().tolist() == [0, 0, 0, 0, 0, 1]
    assert (res.fail_mask.numpy()[:5] & 1).all() and res.fail_mask.numpy()[5] == 0
    assert (res.count.numpy()[:5] == 0).all()
    assert np.isnan(res.center.numpy().reshape(6, -1, 3)[:5]).all()


# ---------------------------------------------------------------------------------------
# The host loader
# ---------------------------------------------------------------------------------------

def test_the_three_xy_forms_and_envs():
    rows = circle_rows(E, N, 1.0)
    a = _gen()
    load_centerlines(a, rows)                                     # a sequence
    b = _gen()
    load_centerlines(b, np.stack(rows))                           # [E, m, 2]
    c = _gen()
    load_centerlines(c, np.concatenate(rows))                     # [E*N, 2]
    for g in (a, b, c):
        g.generate()
    assert _bytes(a.program.result.center) == _bytes(b.program.result.center) == \
        _bytes(c.program.result.center)
    assert a.program.inputs["centerline.count"].numpy().tolist() == [N] * E
    short = [r[:100] for r in rows]
    load_centerlines(a, short[:2], envs=[3, 1])
    assert a.program.inputs["centerline.count"].numpy().tolist() == [N, 100, N, 100]
    load_centerlines(a, rows, count=[50, 60, 70, 80])
    assert a.program.inputs["centerline.count"].numpy().tolist() == [50, 60, 70, 80]
    pad = a.program.inputs["centerline.xy"].numpy().reshape(E, N, 2)
    assert (pad[0, 50:] == pad[0, 49]).all()                      # padded with the last


def test_loader_errors():
    gen = _gen()
    rows = circle_rows(E, N, 1.0)
    with pytest.raises(ValueError, match=r"this generator has no 'centerline\.xy' input: "
                                         r"its generator is 'bezier'"):
        load_centerlines(TrackGenerator(presets.track(num_envs=E, device=DEV)), rows)
    with pytest.raises(TypeError, match=r"target must be a TrackGenerator"):
        load_centerlines(object(), rows)
    with pytest.raises(ValueError, match=r"xy holds 3 rows; 4 envs are loaded"):
        load_centerlines(gen, rows[:3])
    with pytest.raises(ValueError, match=r"env 1: 2 points; a row holds 3 to 256 points"):
        load_centerlines(gen, [rows[0], rows[1][:2], rows[2], rows[3]])
    with pytest.raises(ValueError, match=r"env 0: 300 points"):
        load_centerlines(gen, [np.zeros((300, 2))] + rows[1:])
    bad = [r.copy() for r in rows]
    bad[2][7] = np.nan
    with pytest.raises(ValueError, match=r"env 2: point 7 is not finite"):
        load_centerlines(gen, bad)
    with pytest.raises(ValueError, match=r"z= given, but generator 'external' was built "
                                         r"without it \(generator_params=\{'with_z': True\}"):
        load_centerlines(gen, rows, z=[np.zeros(N)] * E)
    hw = _gen(generator_params={"with_half_width": True})
    with pytest.raises(ValueError, match=r"env 0: half_width must be > 0"):
        load_centerlines(hw, rows, half_width=[np.zeros(N)] * E)
    with pytest.raises(ValueError, match=r"has no attribute 'acme\.x'; its attributes: \[\]"):
        load_centerlines(gen, rows, attributes={"acme.x": [np.zeros(N)] * E})
    with pytest.raises(ValueError, match=r"envs holds 9; env indices are ints in \[0, 4\)"):
        load_centerlines(gen, rows[:1], envs=[9])
    with pytest.raises(ValueError, match=r"count must be \[4\] ints"):
        load_centerlines(gen, rows, count=[3.0, 3.0, 3.0, 3.0])
    with pytest.raises(ValueError, match=r"xy has shape \(4, 256, 3\)"):
        load_centerlines(gen, np.zeros((E, N, 3)))


def test_loader_requires_every_declared_input_and_checks_value_kinds():
    """Each per-point input the generator was built with is loaded with the rows (an
    omitted one would keep the previous rows under the new xy), and every value has the
    kind of its input (an int32 attribute takes integers: a float would be truncated)."""
    rows = circle_rows(E, N, 1.0)
    params = {"with_z": True, "with_half_width": True,
              "attributes": (("acme.tag", "int32", "nearest"),
                             ("acme.gain", "float32", "linear"))}
    gen = _gen(generator_params=params)
    z, hw = [np.zeros(N)] * E, [np.full(N, 0.1)] * E
    tag, gain = [np.arange(N) % 3] * E, [np.ones(N)] * E
    ok = dict(z=z, half_width=hw, attributes={"acme.tag": tag, "acme.gain": gain})
    load_centerlines(gen, rows, **ok)
    before = {k: a.numpy().copy() for k, a in gen.program.inputs.items()}
    with pytest.raises(ValueError, match=r"^load_centerlines: generator 'external' was "
                                         r"built with_z: pass z= for the loaded rows "
                                         r"\(otherwise the input keeps the previous "
                                         r"rows\)$"):
        load_centerlines(gen, rows, **{**ok, "z": None})
    with pytest.raises(ValueError, match=r"built with_half_width: pass half_width="):
        load_centerlines(gen, rows, **{**ok, "half_width": None})
    with pytest.raises(ValueError, match=r"built with attribute 'acme\.gain': pass "
                                         r"attributes=\{'acme\.gain': \.\.\.\}"):
        load_centerlines(gen, rows, **{**ok, "attributes": {"acme.tag": tag}})
    with pytest.raises(ValueError, match=r"^load_centerlines: env 0: attributes\['acme\.tag'"
                                         r"\] is an int32 attribute; got float64 values"):
        load_centerlines(gen, rows, **{**ok, "attributes": {
            "acme.tag": [np.full(N, 2.9)] * E, "acme.gain": gain}})
    with pytest.raises(ValueError, match=r"attributes\['acme\.tag'\] is an int32 "
                                         r"attribute; its values exceed the int32 range"):
        load_centerlines(gen, rows, **{**ok, "attributes": {
            "acme.tag": [np.full(N, 2 ** 40)] * E, "acme.gain": gain}})
    with pytest.raises(ValueError, match=r"env 0: attributes\['acme\.gain'\] holds <U1 "
                                         r"values; expected a float32 attribute"):
        load_centerlines(gen, rows, **{**ok, "attributes": {
            "acme.tag": tag, "acme.gain": [np.array(["a"] * N)] * E}})
    with pytest.raises(ValueError, match=r"^load_centerlines: env 0: xy holds object "
                                         r"values; expected real numbers$"):
        load_centerlines(gen, [np.array(r, dtype=object) for r in rows], **ok)
    with pytest.raises(ValueError, match=r"env 0: z holds <U1 values"):
        load_centerlines(gen, rows, **{**ok, "z": [np.array(["0"] * N)] * E})
    for k, a in gen.program.inputs.items():          # a refused load uploads nothing
        np.testing.assert_array_equal(a.numpy(), before[k], err_msg=k)


def test_the_sizing_check():
    R = 4000.0 / (2 * math.pi)                                     # a 4 km circuit
    rows = circle_rows(1, 1024, R)
    gen = _gen(n=1, num_points=1024)
    with pytest.raises(ValueError, match=r"^load_centerlines: env 0: perimeter 4000\.0 "
                                         r"needs 66667 points at spacing 0\.06 > n_max 384; "
                                         r"set spacing / n_max, or rescale the data "
                                         r"\(half_width and spacing are in the data's "
                                         r"units\)$"):
        load_centerlines(gen, rows)
    tiny = circle_rows(1, 16, 0.001)
    with pytest.raises(ValueError, match=r"too short for spacing"):
        load_centerlines(gen, tiny)
    ok = TrackGenerator(presets.track(num_envs=1, device=DEV, generator="external",
                                      num_points=1024, n_max=2048, spacing=2.0,
                                      half_width=5.0), seeds=0)
    load_centerlines(ok, rows)
    res = ok.generate()
    assert res.valid.numpy()[0] == 1 and res.truncated.numpy()[0] == 0
    assert abs(float(res.length.numpy()[0]) - 4000.0) < 1.0


def test_masked_regeneration_after_loading_the_masked_envs():
    gen = _gen()
    load_centerlines(gen, circle_rows(E, N, 1.0))
    gen.generate()
    before = {k: a.numpy().copy() for k, a in result_arrays(gen.program.result).items()}
    load_centerlines(gen, circle_rows(2, N, 1.3), envs=[1, 3])
    m = wp.array([0, 1, 0, 1], dtype=wp.int32, device=DEV)
    gen.generate(advance=True, mask=m)
    after = result_arrays(gen.program.result)
    for k, a in after.items():
        now = a.numpy()
        per = now.shape[0] // E
        for e in (0, 2):
            assert now[e * per:(e + 1) * per].tobytes() == \
                before[k][e * per:(e + 1) * per].tobytes(), (k, e)
    L = gen.program.result.length.numpy()
    assert abs(L[1] - 2 * math.pi * 1.3) < 0.01 and abs(L[0] - 2 * math.pi) < 0.01


def test_input_info():
    gen = _gen(generator_params={"with_z": True,
                                 "attributes": (("acme.v", "vec3f", "linear"),)})
    info = gen.program.input_info
    assert info["centerline.xy"] == InputInfo("centerline.xy", wp.vec2f, "raw", E * N,
                                              False, "generator")
    assert info["centerline.count"] == InputInfo("centerline.count", wp.int32, "env", E,
                                                 True, "generator")
    assert info["centerline.z"].latched is False and info["centerline.z"].per == "raw"
    assert info["centerline.acme.v"].dtype is wp.vec3f
    assert info["regen_mask"] == InputInfo("regen_mask", wp.int32, "env", E, False, None)
    assert set(info) == set(gen.program.inputs) == set(gen.inputs)
    xy = gen.program.inputs["centerline.xy"].numpy().reshape(E, N, 2)
    assert np.allclose(np.linalg.norm(xy, axis=-1), 0.6)          # the initial circle


def test_kit_passes_without_skips():
    report = check_generator("external", devices=["cpu"])
    report.raise_for_failures()
    div = report.result("diversity")
    assert div.skipped and "seeded=False" in div.message
    assert report.result("output.view").passed
