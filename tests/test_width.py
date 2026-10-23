"""The width registry and the per-point half-width (roadmap PR13a, CPU).

Since 0.2.0.dev4 the width runs BEFORE relax and ``cl.half_width`` is the one per-point
half-width ``relax.xpbd``, ``offset.default`` and ``validity.default`` read. A constant
version (``width.constant``: allocated filled at build, no launch) selects the original
scalar-argument kernels (byte-identical: ``test_pipeline_parity``); any other width runs
the per-point kernels of ``track_gen._src.warp_width``. This file covers the order, the
two paths and their equivalence at constant width, ``Track.half_width``, the real
``w_floor`` / thickness checks, ``width.from_channel``, a generator that provides its
width, the placement and stale-read rules, the preset rules and the
``track_gen.pipeline/1`` migration. The toys are ``tests/_channel_toys.py``.
"""
from __future__ import annotations

import json
import types
import warnings

import numpy as np
import pytest
import warp as wp

from tests import _channel_toys as toys
from track_gen import PerEnvSeededRNG, TrackGenerator
from track_gen._src import warp_pipeline, warp_relax, warp_width
from track_gen.exec import LaunchContext
from track_gen.pipeline import (BUILTIN_FAIL_BITS, PipelineSpec, PlanError, StageRef,
                                build, presets)
from track_gen.plugin import registry
from track_gen.testing.contract import no_allocation
from track_gen.testing.trace import trace_launches

DEV = "cpu"
E = 8
W_FLOOR = BUILTIN_FAIL_BITS["w_floor"]
THICK = BUILTIN_FAIL_BITS["thickness"]
IDS = ("generator", "resample", "width", "relax", "uniform", "frames", "offset",
       "elevation", "validity", "lift")


@pytest.fixture(autouse=True)
def acme():
    with registry.scoped():
        toys.register(registry)
        yield registry


def _rng(n=E, seed=11):
    return PerEnvSeededRNG(seeds=seed, num_envs=n, device=DEV)


def _track(**kw):
    return presets.track(num_envs=kw.pop("num_envs", E), device=DEV, **kw)


def _gen(spec, seed=11):
    return TrackGenerator(spec, _rng(spec.batch.num_envs, seed))


def _bytes(a) -> bytes:
    return np.ascontiguousarray(a.numpy()).tobytes()


def _kernels(gen) -> set:
    with trace_launches() as t:
        gen.enqueue()
    return {k.rsplit(".", 1)[-1] for k in t.kernels()}


def _from_channel(**params):
    return StageRef("width", "from_channel", params=params or None,
                    ports={"src": "cl.acme.hw"})


def _rows(arr, n=E):
    a = arr.numpy()
    return a.reshape(n, -1, *a.shape[1:])


# ---------------------------------------------------------------------------------------
# The order and the two paths
# ---------------------------------------------------------------------------------------

def test_width_runs_before_relax_and_constant_launches_nothing():
    gen = _gen(_track(half_width=0.12))
    p = gen.program
    assert p.spec.ids == IDS
    assert p.channels["cl.half_width"] == ("width", "uniform")
    # width.constant: no launch; its version is the value on EVERY slot (build time).
    i = list(p._stage_ids).index("width")
    with trace_launches() as t:
        p._enqueues[i](LaunchContext.for_call(DEV))
    assert len(t) == 0
    w = p.tap("cl.half_width", after="width").numpy()
    np.testing.assert_array_equal(w, np.full(w.shape, np.float32(0.12)))
    # The final version is the uniform resample's re-fill (dev3's width fill).
    final = p.tap("cl.half_width").numpy()
    np.testing.assert_array_equal(final, np.full(final.shape, np.float32(0.12)))
    assert p.tap("cl.half_width") is not p.tap("cl.half_width", after="width")


def test_constant_width_runs_the_scalar_kernels_and_a_variable_one_the_per_point_ones():
    const = _kernels(_gen(_track()))
    assert {"_step_cached_kernel", "_offset_build_k", "_validity_k", "_band_l0_k"} <= const
    assert not {k for k in const if k.endswith("_w_k") or k.endswith("_w_kernel")}
    assert "_xpbd_radius_k" not in const
    var = _kernels(_gen(_track(width="acme.wave")))
    assert {"_xpbd_radius_k", "_band_l0_w_k", "_step_cached_w_kernel",
            "_build_sep_cache_w_kernel", "_offset_build_w_k", "_offset_assign_w_k",
            "_validity_w_k"} <= var
    assert not {"_step_cached_kernel", "_offset_build_k", "_validity_k"} & var
    # A rewrite of a constant version (acme.rescale) loses the constant declaration.
    rescaled = _kernels(_gen(_track().insert_after(
        "width", StageRef("width", "acme.rescale", id="rescale"))))
    assert "_xpbd_radius_k" in rescaled and "_validity_w_k" in rescaled


@pytest.mark.parametrize("hw", [0.1, 0.25])
@pytest.mark.parametrize("relax", [None, "xpbd"], ids=["no-relax", "xpbd"])
def test_the_per_point_path_equals_the_scalar_one_at_constant_width(hw, relax):
    """``acme.wave`` with ``hw_amp = 0`` writes the constant through the per-point path
    (band from 2 * w_max, the per-bead radii, the per-point offset, the per-pair /
    per-point thickness): every Track array is byte-equal to ``width.constant`` at the
    default margin (E1 / E2 / E3a / E3b-c of the design)."""
    var = _gen(_track(half_width=hw, relax=relax,
                      width=("acme.wave", {"hw_mean": hw, "hw_amp": 0.0})))
    const = _gen(_track(half_width=hw, relax=relax))
    assert "_validity_w_k" in _kernels(var)
    for f in ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
              "valid", "count", "fail_mask", "half_width"):
        assert _bytes(getattr(var._track, f)) == _bytes(getattr(const._track, f)), f
    if relax:
        for name in ("band", "L0"):
            assert _bytes(var.program._scratch["relax"][name]) == \
                _bytes(const.program._scratch["relax"][name]), name
    else:                                       # the thickness bit is exercised
        assert ((const._track.fail_mask.numpy() >> THICK) & 1).any()


def _cfg_ns(p, **over):
    fields = {f: getattr(p, f) for f in p.__dataclass_fields__}
    fields.update(over)
    return types.SimpleNamespace(**fields)


def test_the_xpbd_variant_with_f32_consistent_scalars_is_exact_and_the_planner_picks_the_scalar_path():
    """At ``(hw, margin) = (0.07, 0.2)`` an in-kernel float32 target differs from the
    scalar path's host-double one (design E0 / E3d): the planner selects the scalar
    kernels for ``width.constant`` (bytes equal to a hand-run scalar solve), and the
    variant equals the scalar solve exactly when both get float32-consistent scalars
    (E3c)."""
    hw, margin = 0.07, 0.2
    spec = _track(half_width=hw, relax=("xpbd", {"relax_margin": margin}))
    gen = _gen(spec)
    p = gen.program
    xp = p.spec.stages[3].params
    x_in, count = p.tap("cl.xy", after="resample"), p.tap("cl.count")
    S = int(p.layout.stride)

    def buffers():
        return dict(relaxed=wp.zeros_like(x_in), db=wp.zeros_like(x_in),
                    prev=wp.zeros_like(x_in), band=wp.zeros(E, dtype=wp.int32, device=DEV),
                    l0=wp.zeros(E, dtype=wp.float32, device=DEV),
                    cidx=wp.zeros(E * S * xp.relax_sep_cache_slots, dtype=wp.int32,
                                  device=DEV),
                    ccnt=wp.zeros(E * S, dtype=wp.int32, device=DEV),
                    cov=wp.zeros(1, dtype=wp.int32, device=DEV))

    def scalar(cfg, band_cfg):
        b = buffers()
        warp_relax.band_l0_inplace(x_in, S, b["band"], b["l0"], count, band_cfg)
        warp_relax.xpbd_solve_inplace(x_in, b["relaxed"], b["db"], b["band"], b["l0"],
                                      count, S, cfg, sep_cache_idx_wp=b["cidx"],
                                      sep_cache_count_wp=b["ccnt"],
                                      sep_cache_overflow_wp=b["cov"], cheb_prev_wp=b["prev"])
        return b["relaxed"]

    # The planner's choice: the scalar path, byte for byte.
    ref = scalar(_cfg_ns(xp, half_width=hw), _cfg_ns(xp, half_width=hw))
    assert _bytes(p.tap("cl.xy", after="relax")) == _bytes(ref)
    # The variant (per-point radii) on a constant-valued width.
    b = buffers()
    w = wp.full(E * S, hw, dtype=wp.float32, device=DEV)
    r = wp.zeros(E * S, dtype=wp.float32, device=DEV)
    warp_width.band_l0_w_inplace(x_in, S, w, b["band"], b["l0"], count, xp)
    warp_width.xpbd_solve_w_inplace(x_in, b["relaxed"], b["db"], b["band"], b["l0"], count,
                                    S, xp, w, r, sep_cache_idx_wp=b["cidx"],
                                    sep_cache_count_wp=b["ccnt"],
                                    sep_cache_overflow_wp=b["cov"], cheb_prev_wp=b["prev"])
    assert _bytes(b["band"]) == _bytes(p._scratch["relax"]["band"])     # 2 w_max == 2 hw
    assert _bytes(b["relaxed"]) != _bytes(ref)       # the host-double targets differ
    k = np.float32(1.0 + margin)
    rr = np.float32(np.float32(hw) * k)
    skin = float(np.float32(1.0 + xp.relax_sep_cache_skin)) - 1.0
    consistent = scalar(_cfg_ns(xp, half_width=float(rr), relax_margin=0.0,
                                relax_sep_cache_skin=skin), _cfg_ns(xp, half_width=hw))
    assert _bytes(b["relaxed"]) == _bytes(consistent)


def test_the_xpbd_radii_are_one_product_per_bead_and_the_pair_targets_symmetric():
    gen = _gen(_track(width="acme.wave"))
    p = gen.program
    r = p._scratch["relax"]["xpbd.r"].numpy()
    w = p.tap("cl.half_width", after="width").numpy()
    k = np.float32(1.0 + p.spec.stages[3].params.relax_margin)
    np.testing.assert_array_equal(r, (w * k).astype(np.float32))
    rows = r.reshape(E, -1)
    count = p.tap("cl.count").numpy()
    for e in range(E):
        m = int(count[e])
        a = rows[e, :m][:, None] + rows[e, :m][None, :]
        assert a.tobytes() == a.T.tobytes()


# ---------------------------------------------------------------------------------------
# Track.half_width
# ---------------------------------------------------------------------------------------

def _assert_band(track, label):
    count = track.count.numpy()
    hw = _rows(track.half_width)
    c, o, i = (_rows(getattr(track, f))[..., :2] for f in ("center", "outer", "inner"))
    for e in range(E):
        m = int(count[e])
        assert np.all(np.isnan(hw[e, m:])), f"{label}: env {e}: padding"
        np.testing.assert_allclose(np.linalg.norm(o[e, :m] - c[e, :m], axis=-1), hw[e, :m],
                                   atol=1e-6, err_msg=f"{label}: env {e}: outer")
        np.testing.assert_allclose(np.linalg.norm(c[e, :m] - i[e, :m], axis=-1), hw[e, :m],
                                   atol=1e-6, err_msg=f"{label}: env {e}: inner")


def test_track_half_width_is_the_band_at_constant_and_variable_width():
    const = _gen(_track(half_width=0.09))._track
    count = const.count.numpy()
    hw = _rows(const.half_width)
    for e in range(E):
        np.testing.assert_array_equal(hw[e, :count[e]], np.float32(0.09))
    _assert_band(const, "constant")
    gen = _gen(_track(width="acme.wave"))
    var = gen._track
    _assert_band(var, "variable")
    w = _rows(gen.program.tap("cl.half_width"))
    vh = _rows(var.half_width)
    for e in range(E):
        m = int(var.count.numpy()[e])
        np.testing.assert_array_equal(vh[e, :m], w[e, :m])
        assert np.ptp(vh[e, :m]) > 1e-3
    assert gen.program.tap("track.half_width") is var.half_width
    clone = var.clone()
    assert clone.half_width is not var.half_width
    assert _bytes(clone.half_width) == _bytes(var.half_width)


# ---------------------------------------------------------------------------------------
# The real w_floor (bit 4) and thickness (bit 3) checks
# ---------------------------------------------------------------------------------------

def _per_env(values, dtype=wp.float32):
    return wp.array(np.asarray(values, np.float32), dtype=dtype, device=DEV)


@pytest.mark.parametrize("source", ["wave", "pinch"])
def test_w_floor_fails_exactly_where_the_width_pinches_below_it(source):
    """``hw_mean - hw_amp`` well below ``w_floor`` (by more than 0.3 hw_amp, so the
    final samples land below it too) on envs 1 and 3 only, set per env through the live
    params: bit 4 is set exactly there (the width stage before relax, and a generator
    that provides its width)."""
    if source == "wave":
        spec, sid = _track(width="acme.wave"), "width"
    else:
        spec, sid = _track(generator="acme.pinch"), "generator"
        assert "width" not in spec.ids
    gen = _gen(spec)
    mean = np.full(E, 0.1, np.float32)
    amp = np.full(E, 0.03, np.float32)
    mean[[1, 3]], amp[[1, 3]] = 0.03, 0.06           # min -0.03: 0.031 below w_floor
    assert np.all(1e-3 - (mean[[1, 3]] - amp[[1, 3]]) >= 0.3 * amp[[1, 3]])
    gen.params[sid].set("hw_mean", _per_env(mean))
    gen.params[sid].set("hw_amp", _per_env(amp))
    gen.generate()
    fail = gen._track.fail_mask.numpy()
    np.testing.assert_array_equal((fail >> W_FLOOR) & 1, np.isin(np.arange(E), [1, 3]))


def test_thickness_fails_where_the_band_is_wider_than_the_curvature_radius():
    """Without relax (deterministic), ribbon_plain's ellipse (smallest curvature radius
    0.64): a width never below 0.7 overlaps itself at the major-axis ends (bit 3); the
    same loop at 0.1 passes."""
    wide = _gen(_track(generator="acme.ribbon_plain", relax=None,
                       width=("acme.wave", {"hw_mean": 0.75, "hw_amp": 0.05})))
    thin = _gen(_track(generator="acme.ribbon_plain", relax=None,
                       width=("acme.wave", {"hw_mean": 0.1, "hw_amp": 0.05})))
    assert np.all((wide._track.fail_mask.numpy() >> THICK) & 1)
    assert not np.any((thin._track.fail_mask.numpy() >> THICK) & 1)
    assert np.all(thin._track.valid.numpy() == 1)


def test_xpbd_separates_by_the_per_point_width():
    """(A) ``acme.wave`` before relax (XPBD separates every bead by its own width) vs
    (B) ``width.constant(hw_mean - hw_amp)`` before relax and the same sinusoid as a
    rewrite after the uniform resample (XPBD blind to it): A accepts more envs, and B
    fails the thickness check."""
    n, mean, amp = 32, 0.12, 0.06
    a = _gen(_track(num_envs=n, width=("acme.wave", {"hw_mean": mean, "hw_amp": amp})))
    b = _gen(_track(num_envs=n, half_width=mean - amp).insert_after(
        "uniform", StageRef("width", "acme.wave_after", id="wave_after",
                            params={"hw_mean": mean, "hw_amp": amp})))
    fa, fb = a._track.fail_mask.numpy(), b._track.fail_mask.numpy()
    assert (fa == 0).mean() > (fb == 0).mean()
    assert ((fb >> THICK) & 1).any()


# ---------------------------------------------------------------------------------------
# width.from_channel
# ---------------------------------------------------------------------------------------

def test_from_channel_scales_and_clamps_the_carried_source():
    gen = _gen(_track(generator="acme.ribbon",
                      width=_from_channel(scale=1.5, lo=0.05, hi=0.14)))
    p = gen.program
    src = _rows(p.tap("cl.acme.hw", after="resample"))
    w = _rows(p.tap("cl.half_width", after="width"))
    count = p.tap("cl.count").numpy()
    for e in range(E):
        m = int(count[e])
        want = np.minimum(np.maximum(np.float32(1.5) * src[e, :m], np.float32(0.05)),
                          np.float32(0.14))
        np.testing.assert_array_equal(w[e, :m], want)
        assert np.all(np.isnan(w[e, m:]))
    assert (w[:, 0] == np.float32(0.14)).any() or (w == np.float32(0.14)).any()
    _assert_band(gen._track, "from_channel")


def test_from_channel_passes_nan_through_and_the_env_fails_w_floor():
    gen = _gen(_track(generator="acme.ribbon", width=_from_channel()))
    mean = np.full(E, 0.1, np.float32)
    mean[2] = np.nan                                  # an array set: no range check
    gen.params["generator"].set("hw_mean", _per_env(mean))
    gen.generate()
    w = _rows(gen.program.tap("cl.half_width", after="width"))
    count = gen.program.tap("cl.count").numpy()
    assert np.all(np.isnan(w[2, :count[2]]))
    fail = gen._track.fail_mask.numpy()
    np.testing.assert_array_equal((fail >> W_FLOOR) & 1, np.arange(E) == 2)
    # The kernel itself: min / max drop NaN, the explicit branch keeps it.
    src = wp.array(np.array([np.nan, 0.5, -1.0, np.inf], np.float32), device=DEV)
    out = wp.zeros(4, dtype=wp.float32, device=DEV)
    one = lambda v: wp.array(np.array([v], np.float32), device=DEV)  # noqa: E731
    wp.launch(warp_width._width_from_channel_k, 4,
              [src, wp.array(np.array([4], np.int32), device=DEV), one(2.0), one(0.0),
               one(1.0e9), 4, out], device=DEV)
    got = out.numpy()
    assert np.isnan(got[0]) and got[1] == 1.0 and got[2] == 0.0 and got[3] == 1.0e9


def test_from_channel_bounds_at_run_time():
    """``hi`` is range-checked on the host (``>= 0``); ``lo <= hi`` is a build check,
    and a run-time ``hi`` below ``lo`` clamps to ``lo`` (never a negative width)."""
    gen = _gen(_track(generator="acme.ribbon", width=_from_channel()))
    with pytest.raises(ValueError, match="hi"):
        gen.params["width"].set("hi", -0.5)
    gen.params["width"].set("lo", 0.3)
    gen.params["width"].set("hi", 0.2)
    gen.generate()
    w = _rows(gen.program.tap("cl.half_width", after="width"))
    count = gen.program.tap("cl.count").numpy()
    for e in range(E):
        assert np.all(w[e, :count[e]] == np.float32(0.3)), e


def test_from_channel_wiring_errors():
    with pytest.raises(PlanError, match=r"stage 'width' \(width.from_channel\): read port "
                                        r"'src' \(cl.\*\) must be wired: "
                                        r"StageRef\('width', 'from_channel', "
                                        r"ports=\{'src': 'cl.<pkg>.<name>'\}\); "
                                        r"plugin channels written before it: \[.*"
                                        r"'cl.acme.hw', .*\]"):
        build(_track(generator="acme.ribbon", width="from_channel"), seeds=1)
    with pytest.raises(PlanError, match=r"read port 'src' \(cl.\*\) is wired to "
                                        r"'raw.acme.hw', not a cl channel"):
        build(_track(generator="acme.ribbon", width=StageRef(
            "width", "from_channel", ports={"src": "raw.acme.hw"})), seeds=1)
    with pytest.raises(ValueError, match=r"src 'cl.acme.surface_id' has dtype int32; it "
                                         r"reads a float32 channel"):
        build(_track(generator="acme.ribbon", width=StageRef(
            "width", "from_channel", ports={"src": "cl.acme.surface_id"})), seeds=1)
    with pytest.raises(PlanError, match=r"stage 'width' \(width.from_channel\): read port "
                                        r"'src' \(cl.\*\) is wired to 'cl.half_width', "
                                        r"which the stage itself writes \(port "
                                        r"'half_width'\); .* rescaling the current width\) "
                                        r"is a rewrite stage"):
        build(_track(generator="acme.ribbon", width=StageRef(
            "width", "from_channel", ports={"src": "cl.half_width"})), seeds=1)
    with pytest.raises(ValueError, match="lo must be <= hi"):
        build(_track(generator="acme.ribbon", width=_from_channel(lo=0.5, hi=0.1)),
              seeds=1)
    assert registry.describe("width", "from_channel").reads["src"] == "cl.*"


# ---------------------------------------------------------------------------------------
# A generator that provides its width
# ---------------------------------------------------------------------------------------

def test_a_width_provider_is_carried_straight_into_cl_half_width():
    spec = _track(generator="acme.pinch")
    assert "width" not in spec.ids and spec.ids[2] == "relax"
    gen = _gen(spec)
    p = gen.program
    assert p.channels["cl.half_width"] == ("resample", "uniform")
    raw = p.tap("raw.half_width").numpy()
    tables = p._scratch["resample"]
    i0, i1, fr = (tables[k].numpy() for k in ("tables.i0", "tables.i1", "tables.frac"))
    got = p.tap("cl.half_width", after="resample").numpy()
    real = i0 >= 0
    want = raw[i0[real]] + fr[real] * (raw[i1[real]] - raw[i0[real]])
    np.testing.assert_allclose(got[real], want, rtol=1e-6, atol=1e-7)
    assert np.all(np.isnan(got[~real]))
    _assert_band(gen._track, "pinch")
    msg = (r"^width='acme.wave': generator 'acme.pinch' provides its own half-widths "
           r"\(raw.half_width -> cl.half_width\)")
    with pytest.raises(ValueError, match=msg):
        _track(generator="acme.pinch", width="acme.wave")
    # presets.gates builds a relaxed_stations centerline through presets.track: the
    # message names no preset (not presets.track).
    with pytest.raises(ValueError, match=msg):
        presets.gates(num_envs=E, device=DEV, generator="acme.pinch",
                      source="relaxed_stations", width="acme.wave")
    # A hand-built spec keeping the width stage: the second-writer rule, with a hint.
    hand = spec.insert_before("relax", StageRef("width", "constant",
                                                params={"half_width": 0.1}))
    with pytest.raises(PlanError, match=r"writes 'cl.half_width', already written by stage "
                                        r"'resample'.*generator 'acme.pinch' provides "
                                        r"half_width \(raw.half_width -> cl.half_width\): "
                                        r"drop the width stage"):
        build(hand, seeds=1)


# ---------------------------------------------------------------------------------------
# Placement and stale reads
# ---------------------------------------------------------------------------------------

def test_a_width_written_after_the_offset_is_refused():
    spec = _track().insert_after("offset", StageRef("width", "acme.wave_after", id="late"))
    with pytest.raises(PlanError, match=r"stage 'late' writes cl.half_width after "
                                        r"'offset': the band and Track.half_width come "
                                        r"from offset's read"):
        build(spec, seeds=1)
    # A rewrite after the uniform resample (before the offset) is legitimate.
    ok = _track().insert_after("uniform", StageRef("width", "acme.wave_after", id="wa"))
    gen = _gen(ok)
    assert gen.program.channels["cl.half_width"] == ("width", "uniform", "wa")
    _assert_band(gen._track, "rewrite after uniform")


def test_a_stage_written_channel_read_across_the_uniform_resample_is_refused():
    spec = _track(width="acme.curv_width", checks=("default", "acme.curv_check"))
    with pytest.raises(PlanError, match=r"stage 'acme.curv_check' reads cl.acme.curv, "
                                        r"written by 'width' before the uniform resample "
                                        r"\('uniform'\), which does not carry "
                                        r"stage-written channels"):
        build(spec, seeds=1)
    # Written after the uniform resample, it is read on the final points.
    late = _track(checks=("default", "acme.curv_check")).insert_after(
        "uniform", StageRef("width", "acme.curv_width", id="cw"))
    with pytest.raises(PlanError, match="already written"):
        build(late, seeds=1)               # cl.half_width: a second writer, not a rewrite


# ---------------------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------------------

def test_the_preset_fills_a_width_stage_half_width_field():
    import dataclasses

    from track_gen import plugin

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class RippleParams(plugin.Params):
        half_width: float = plugin.static(0.1, doc="nominal half-width")

    class Ripple(toys.Wave):
        name = "acme.ripple"
        Params = RippleParams

        def build(self, b, p):
            w = b.write("half_width", wp.float32, per="cl")
            hw = float(p.half_width)
            return lambda ctx: ctx.fill(w, hw)

    registry.add(Ripple)
    spec = _track(half_width=0.15, width="acme.ripple")
    assert dict(spec.stages[2].params) == {"half_width": 0.15}
    assert dict(spec.stages[1].params) == {"spacing": pytest.approx(0.09)}
    with pytest.raises(ValueError, match=r"width: width.acme.ripple sets half_width=0.2 but "
                                         r"half_width=0.1"):
        _track(width=("acme.ripple", {"half_width": 0.2}))
    with pytest.raises(ValueError, match=r"width: width.constant sets half_width=0.2 but "
                                         r"half_width=0.1; pass half_width=0.2"):
        _track(width=("constant", {"half_width": 0.2}))
    # A width without the field: half_width= only sets the default spacing ...
    wave = _track(half_width=0.2, width="acme.wave")
    assert "half_width" not in dict(wave.stages[2].params or {})
    assert dict(wave.stages[1].params) == {"spacing": pytest.approx(0.12)}
    # ... so with an explicit spacing it reaches nothing.
    with pytest.raises(ValueError, match=r"half_width= sets width.constant's value and the "
                                         r"default spacing; width.acme.wave takes its "
                                         r"width from elsewhere"):
        _track(half_width=0.2, spacing=0.05, width="acme.wave")
    with pytest.raises(ValueError, match=r"generator 'acme.pinch' provides its own "
                                         r"half-widths and spacing= is given"):
        _track(generator="acme.pinch", half_width=0.2, spacing=0.05)


def test_dev3_specs_get_targeted_messages():
    for slot, kind, name in (("relax", "relax", "xpbd"), ("offset", "offset", "default"),
                             ("validity", "validity", "default")):
        spec = _track().replace(slot, StageRef(kind, name, params={"half_width": 0.1}))
        with pytest.raises(PlanError, match=rf"half_width is no longer a {kind}.{name} "
                                            rf"param \(0.2.0.dev4\)"):
            build(spec, seeds=1)
    # The dev3 order (width after the uniform resample) with relax.xpbd: the read error
    # names the new order.
    spec = _track()
    stages = [r for r in spec.stages if r.slot_id != "width"]
    stages.insert(4, spec.stages[2])
    old = PipelineSpec(spec.batch, tuple(stages))
    assert old.ids[3:5] == ("uniform", "width")
    with pytest.raises(PlanError, match=r"reads 'cl.half_width' \(port 'half_width'\), but "
                                        r"no earlier stage writes it; the width stage runs "
                                        r"before relax since 0.2.0.dev4"):
        build(old, seeds=1)
    # A relax-less dev3 order still plans: the same bytes, one launch fewer.
    nr = _track(relax=None)
    stages = [r for r in nr.stages if r.slot_id != "width"]
    stages.insert(4, nr.stages[2])
    old_nr = PipelineSpec(nr.batch, tuple(stages))
    a, b = _gen(nr), _gen(old_nr)
    for f in ("outer", "center", "inner", "valid", "fail_mask", "half_width"):
        assert _bytes(getattr(a._track, f)) == _bytes(getattr(b._track, f)), f
    with trace_launches() as ta:
        a.enqueue()
    with trace_launches() as tb:
        b.enqueue()
    assert len(tb) == len(ta) - 1


# ---------------------------------------------------------------------------------------
# The track_gen.pipeline/1 migration
# ---------------------------------------------------------------------------------------

def _v1(spec, half_width) -> dict:
    """``spec`` as format /1 wrote it (0.2.0.dev3, and v2 before PR13a for the
    relaxed_stations gate specs of PR11): the width stage after the uniform
    resample (when there is one), half_width in relax.xpbd / offset / validity params,
    and no width stage in a relaxed_stations gate spec."""
    d = json.loads(json.dumps(spec.to_dict()))
    d["format"] = "track_gen.pipeline/1"
    stages = d["stages"]
    widths = [s for s in stages if s["kind"] == "width"]
    rest = [s for s in stages if s["kind"] != "width"]
    kinds = [s["kind"] for s in rest]
    if "frames" in kinds:                       # a track: width after the uniform
        rest.insert(kinds.index("frames"), widths[0])
    for s in rest:
        if (s["kind"], s["name"]) in (("relax", "xpbd"), ("offset", "default"),
                                      ("validity", "default")):
            s["params"] = {**(s["params"] or {}), "half_width": half_width}
    d["stages"] = rest
    return d


@pytest.mark.parametrize("make", [
    lambda: _track(half_width=0.09),
    lambda: _track(relax=None),
    lambda: presets.gates(num_envs=4, device=DEV, generator="bezier",
                          source="relaxed_stations", gates=10, half_width=0.09),
], ids=["default", "relax-less", "relaxed_stations"])
def test_a_dev3_spec_migrates_and_builds_the_same_outputs(make):
    spec = make()
    hw = 0.09 if any(r.kind == "width" and dict(r.params or {}).get("half_width") == 0.09
                     for r in spec.stages) else 0.1
    old = _v1(spec, hw)
    with pytest.warns(DeprecationWarning, match=r"track_gen.pipeline/1 \(0.2.0.dev3\) "
                                                r"migrated to track_gen.pipeline/2"):
        back = PipelineSpec.from_dict(old)
    assert back.ids == spec.ids
    assert back.to_dict()["format"] == "track_gen.pipeline/2"
    n = spec.batch.num_envs
    a = build(spec, seeds=PerEnvSeededRNG(seeds=5, num_envs=n, device=DEV)).run(sync=True)
    b = build(back, seeds=PerEnvSeededRNG(seeds=5, num_envs=n, device=DEV)).run(sync=True)
    fields = ("center", "outer", "valid", "count", "fail_mask") if spec.batch.max_gates \
        is None else ("position", "tangent", "valid", "count", "fail_mask")
    for f in fields:
        assert _bytes(getattr(a, f)) == _bytes(getattr(b, f)), f


def test_a_dev3_spec_with_disagreeing_half_widths_is_refused():
    old = _v1(_track(), 0.1)
    for s in old["stages"]:
        if s["kind"] == "validity":
            s["params"]["half_width"] = 0.2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match=r"stage 'validity' \(validity.default\) sets "
                                             r"half_width=0.2 but the width is 0.1"):
            PipelineSpec.from_dict(old)
    with pytest.raises(ValueError, match="unsupported pipeline format 'x'"):
        PipelineSpec.from_dict({**old, "format": "x"})


def test_a_dev3_spec_with_a_plugin_width_migrates_with_an_output_warning():
    """A dev3 dict whose width stage is a plugin migrates (the stage moves before relax)
    but its outputs change: the warning says so, naming the stage."""
    spec = _track(width=("acme.wave", {"hw_mean": 0.1, "hw_amp": 0.02}))
    old = _v1(spec, 0.1)
    with pytest.warns(DeprecationWarning, match=r"the width stage\(s\) \['width.acme.wave'\] "
                                                r"now run before relax .* the outputs differ "
                                                r"from 0.2.0.dev3"):
        back = PipelineSpec.from_dict(old)
    assert back.ids == spec.ids
    build(back, seeds=1)
    # width.constant: no output warning
    with pytest.warns(DeprecationWarning) as rec:
        PipelineSpec.from_dict(_v1(_track(), 0.1))
    assert all("outputs differ" not in str(w.message) for w in rec)


def test_a_dev3_width_after_frames_is_refused_by_the_migration():
    """dev3 documented a width stage reading cl.kappa after frames; with relax.xpbd and no
    width before it the migration cannot place a width.constant without a second writer
    of cl.half_width, so it names the stage."""
    old = _v1(_track(width=("acme.wave", {"hw_mean": 0.1, "hw_amp": 0.02})), 0.1)
    stages = old["stages"]
    w = next(s for s in stages if s["kind"] == "width")
    stages.remove(w)
    stages.insert([s["kind"] for s in stages].index("frames") + 1, w)
    with pytest.raises(ValueError, match=r"width stage 'width' \(width.acme.wave\) runs "
                                         r"after frames; .* make 'width' a rewrite of "
                                         r"cl.half_width"):
        PipelineSpec.from_dict(old)


# ---------------------------------------------------------------------------------------
# The legacy call forms allocate nothing
# ---------------------------------------------------------------------------------------

def test_legacy_offset_and_inflate_without_hw_out_allocate_nothing():
    gen = _gen(_track(num_envs=2))
    s = gen._scratch
    stag = s.inflate
    dev3 = warp_pipeline.InflateScratch(
        area_a=stag.area_a, area_b=stag.area_b, kappa=stag.kappa, w=stag.w,
        out2=stag.out2, ctr2=stag.ctr2, inn2=stag.inn2, tan2=stag.tan2, nrm2=stag.nrm2,
        z=stag.z, fail=stag.fail)
    assert dev3.hw_out is None
    scratch = warp_pipeline._Scratch(inflate=dev3, cs_seg=s.cs_seg, cs_s=s.cs_s)
    out, _ = warp_pipeline._inflate_warp_alloc(gen._config)
    p = gen.program
    center, count = p.tap("cl.xy", after="relax"), p.tap("cl.count")
    ctx = LaunchContext.for_call(DEV)
    warp_pipeline.inflate_warp(center, gen._config, out=out, valid=p.tap("gen.valid"),
                               count=count, scratch=scratch, seeds=gen._seed_buf,
                               truncated=out.truncated, ctx=ctx)
    with no_allocation():
        warp_pipeline.offset(stag.ctr2, stag.nrm2, 0.1, stag.out2, stag.inn2, stag.area_a,
                             stag.area_b, count, ctx=ctx)
        warp_pipeline.inflate_warp(center, gen._config, out=out, valid=p.tap("gen.valid"),
                                   count=count, scratch=scratch, seeds=gen._seed_buf,
                                   truncated=out.truncated, ctx=ctx)
    # The PR9 composition binds the offset's half-width as its Track's.
    ref_track, ref_scratch = warp_pipeline._inflate_warp_alloc(gen._config)
    assert ref_scratch.inflate.hw_out is ref_track.half_width
    # A given hw_out is checked on the host (the kernel writes every slot of out_outer).
    flat = int(stag.out2.shape[0])
    w = p.tap("cl.half_width", after="width")
    for bad, match in ((wp.zeros(flat - 1, dtype=wp.float32, device=DEV), "have shape"),
                       (wp.zeros(flat, dtype=wp.int32, device=DEV), "have dtype float32"),
                       (np.zeros(flat, np.float32), "be a wp.array")):
        with pytest.raises(ValueError, match=f"hw_out must {match}"):
            warp_pipeline.offset(stag.ctr2, stag.nrm2, 0.1, stag.out2, stag.inn2,
                                 stag.area_a, stag.area_b, count, ctx=ctx, hw_out=bad)
        with pytest.raises(ValueError, match=f"hw_out must {match}"):
            warp_width.offset_w(stag.ctr2, stag.nrm2, w, stag.out2, stag.inn2,
                                stag.area_a, stag.area_b, count, hw_out=bad, ctx=ctx)
