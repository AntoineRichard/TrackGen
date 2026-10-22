"""The commit: what each env's ``Track`` holds after a generation (roadmap PR10b, CPU).

The stages recompute every env; the final version of each exported channel is the
``Track``'s own array. The commit (``track_gen.stages.commit``) then settles the
``Track``: a masked regeneration restores every unmasked env's rows bit for bit (held
before the stages ran), ``valid`` is derived as ``fail_mask == 0`` when a plugin check
ran after ``validity.default``, and ``on_invalid="poison"`` leaves every invalid env with
``count = 0`` and NaN geometry. Covered: exactness without relying on the
recomputation (a stage's output is perturbed on every call), the all-zero mask, poison
(and the utilities' degenerate paths on it), ``out=``, the launch counts.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests._stage_toys import KMAX, register
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import warp_pipeline
from track_gen._src.checkpoints import CheckpointSampler
from track_gen._src.collision import CollisionChecker
from track_gen._src.localize import TrackLocalizer
from track_gen._src.props import PropSampler
from track_gen.exec import LaunchContext
from track_gen.pipeline import BatchSpec, PipelineSpec, PlanError, build, presets
from track_gen.plugin import registry
from track_gen.testing.capture import poison_all
from track_gen.testing.trace import trace_launches

DEV = "cpu"
E = 4
POINT = ("outer", "center", "inner", "tangent", "normal", "arclen")
FIELDS = POINT + ("length", "valid", "count", "winding", "truncated", "generation",
                  "fail_mask")
MASK = np.array([1, 0, 1, 0], np.int32)


@pytest.fixture(autouse=True)
def toys():
    register(registry)


def _rng(seed=11):
    return PerEnvSeededRNG(seeds=seed, num_envs=E, device=DEV)


def _spec(**kw):
    return presets.track(num_envs=E, device=DEV, **kw)


def _gen(spec, seed=11):
    return TrackGenerator(spec, _rng(seed))


def _i32(values):
    return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=DEV)


def _host(track) -> dict:
    return {f: getattr(track, f).numpy().copy() for f in FIELDS}


def _rows(host: dict, e: int) -> dict:
    return {f: v.reshape(E, -1)[e].tobytes() for f, v in host.items()}


def _kappa_split():
    """kappa_max per env: 0 (every non-empty loop fails) / 1e6 (none fails)."""
    return np.array([0.0, 1.0e6] * (E // 2), np.float32)


# ---------------------------------------------------------------------------------------
# Unmasked: the Track arrays are the final versions
# ---------------------------------------------------------------------------------------

def test_the_track_arrays_are_the_final_channel_versions():
    g = _gen(_spec())
    p, t = g.program, g._track
    for field, ch in (("outer", "track.outer"), ("center", "track.center"),
                      ("arclen", "cl.arclen2"), ("length", "cl.length2"),
                      ("winding", "cl.winding"), ("count", "track.count"),
                      ("valid", "track.valid"), ("truncated", "cl.truncated"),
                      ("fail_mask", "valid.fail")):
        assert p.tap(ch) is getattr(t, field), field
    assert not p._commit.always                       # nothing to do unmasked
    ptrs = {f: getattr(t, f).ptr for f in FIELDS}
    for _ in range(3):
        assert g.generate(advance=True) is t
        assert {f: getattr(t, f).ptr for f in FIELDS} == ptrs


def test_a_derived_valid_is_fail_mask_zero():
    spec = _spec(checks=("default", KMAX))
    g = _gen(spec)
    g.params[KMAX].set("kappa_max", wp.array(_kappa_split(), dtype=wp.float32,
                                               device=DEV))
    t = g.generate()
    p = g.program
    assert p._commit.derive_valid and p.tap("track.valid") is not t.valid
    fm = t.fail_mask.numpy()
    np.testing.assert_array_equal(t.valid.numpy(), (fm == 0).astype(np.int32))
    assert (t.valid.numpy()[0::2] == 0).all()
    np.testing.assert_array_equal(p.tap("track.valid").numpy(),
                                  ((fm & 0xFFFF) == 0).astype(np.int32))


# ---------------------------------------------------------------------------------------
# Masked: exact by construction
# ---------------------------------------------------------------------------------------

@wp.kernel
def _shift_k(xy: wp.array(dtype=wp.vec2f), dx: float):
    t = wp.tid()
    xy[t] = xy[t] + wp.vec2f(dx, 0.0)


def _perturb(program, stage_id: str, channel: str):
    """Wrap ``stage_id``'s enqueue (test-only): after it, shift its version of
    ``channel`` for EVERY env by an amount that changes on every call, so the pipeline's
    recomputation of an env never equals the previous one."""
    i = list(program._stage_ids).index(stage_id)
    base = program._enqueues[i]
    xy = program.tap(channel, after=stage_id)
    calls = [0]

    def enqueue(ctx):
        base(ctx)
        calls[0] += 1
        ctx.launch(_shift_k, int(xy.shape[0]), [xy, 1.0e-3 * calls[0]])

    program._enqueues[i] = enqueue
    return calls


@pytest.mark.parametrize("kw", [{}, dict(checks=("default", KMAX)),
                                dict(on_invalid="poison"),
                                dict(elevation=("noise", {"z_max": 0.3,
                                                          "z_noise_amplitude": 0.2}))])
def test_a_masked_regeneration_never_writes_an_unmasked_env(kw):
    g = _gen(_spec(**kw))
    calls = _perturb(g.program, "uniform", "cl.xy")
    tap_before = g.program.tap("cl.xy").numpy().reshape(E, -1).copy()
    before = _host(g._track)
    g.generate(advance=True, mask=_i32(MASK))
    after = _host(g._track)
    tap_after = g.program.tap("cl.xy").numpy().reshape(E, -1)
    assert calls[0] == 1
    for e in range(E):
        # the stages DID recompute every env, differently
        assert not np.array_equal(tap_before[e], tap_after[e], equal_nan=True), e
        if MASK[e]:
            assert _rows(after, e) != _rows(before, e), e
            assert after["generation"][e] == 1
        else:
            assert _rows(after, e) == _rows(before, e), e        # every field, bit for bit


def test_an_all_zero_mask_writes_nothing():
    for kw in ({}, dict(on_invalid="poison", checks=("default", KMAX))):
        g = _gen(_spec(**kw))
        poison_all(g._track)                       # sentinels the stages never produce
        before = _host(g._track)
        g.generate(advance=True, mask=_i32([0] * E))
        after = _host(g._track)
        for f in FIELDS:
            assert after[f].tobytes() == before[f].tobytes(), (kw, f)
        assert g.seeds.epoch.numpy().tolist() == [0] * E


def test_masked_envs_equal_a_fresh_generator_at_the_same_epoch():
    g = _gen(_spec(checks=("default", KMAX), on_invalid="poison"))
    g.generate(advance=True, mask=_i32(MASK))
    g.generate(advance=True, mask=_i32(MASK))
    twin = _gen(_spec(checks=("default", KMAX), on_invalid="poison"))
    twin.seeds.epoch.assign(np.array([2, 0, 2, 0], np.int32))
    twin.generate()
    a, b = _host(g._track), _host(twin._track)
    for e in range(E):
        rows_a = {f: v for f, v in _rows(a, e).items() if f != "generation"}
        rows_b = {f: v for f, v in _rows(b, e).items() if f != "generation"}
        assert rows_a == rows_b, e


# ---------------------------------------------------------------------------------------
# on_invalid="poison"
# ---------------------------------------------------------------------------------------

def _poisoned(keep: dict) -> dict:
    out = {k: v.copy() for k, v in keep.items()}
    bad = keep["fail_mask"] != 0
    for f in POINT:
        out[f].reshape(E, -1)[bad] = np.nan
    out["count"][bad] = 0
    out["length"][bad] = 0.0
    out["winding"][bad] = 0.0
    out["valid"][bad] = 0
    return out


def _split(g):
    g.params[KMAX].set("kappa_max", wp.array(_kappa_split(), dtype=wp.float32,
                                               device=DEV))
    return g.generate()


def test_poison_is_keep_with_the_invalid_envs_poisoned():
    keep = _gen(_spec(checks=("default", KMAX)))
    poison = _gen(_spec(checks=("default", KMAX), on_invalid="poison"))
    assert poison.program.on_invalid == "poison" and keep.program.on_invalid == "keep"
    a, b = _host(_split(keep)), _host(_split(poison))
    bad = a["fail_mask"] != 0
    assert bad.any() and not bad.all()
    want = _poisoned(a)
    for f in FIELDS:
        assert b[f].tobytes() == want[f].tobytes(), f
    assert (b["count"][bad] == 0).all() and np.isnan(b["center"].reshape(E, -1)[bad]).all()
    np.testing.assert_array_equal(b["fail_mask"], a["fail_mask"])   # diagnostics kept


def test_poison_of_truncated_envs():
    kw = dict(n_max=64, spacing=0.06)
    keep, poison = _gen(_spec(**kw)), _gen(_spec(on_invalid="poison", **kw))
    a, b = _host(keep._track), _host(poison._track)
    assert a["truncated"].any()
    np.testing.assert_array_equal((a["fail_mask"] >> 7) & 1, a["truncated"])
    for f in FIELDS:
        assert b[f].tobytes() == _poisoned(a)[f].tobytes(), f
    np.testing.assert_array_equal(b["truncated"], a["truncated"])


def test_masked_poison_touches_the_masked_envs_only():
    g = _gen(_spec(checks=("default", KMAX), on_invalid="poison"))
    before = _host(g._track)
    assert (before["valid"] == 1).any()
    g.params[KMAX].set("kappa_max", 0.0)               # every loop fails from now on
    g.generate(advance=True, mask=_i32(MASK))
    after = _host(g._track)
    for e in range(E):
        if MASK[e]:
            assert after["count"][e] == 0
            assert np.isnan(after["center"].reshape(E, -1)[e]).all()
            assert after["fail_mask"][e] & (1 << 16)
        else:
            assert _rows(after, e) == _rows(before, e)


def test_the_utilities_take_their_degenerate_paths_on_a_poisoned_env():
    g = _gen(_spec(checks=("default", KMAX), on_invalid="poison"))
    t = _split(g)
    bad = np.flatnonzero(t.fail_mask.numpy() != 0)
    good = np.flatnonzero(t.fail_mask.numpy() == 0)
    assert bad.size and good.size
    center = t.center.numpy().reshape(E, -1, 3)
    pos = np.zeros((E, 3), np.float32)
    for e in good:
        pos[e] = center[e, 0]
    position = wp.array(pos, dtype=wp.vec3f, device=DEV)

    frame = TrackLocalizer(t).query(position)
    s, seg = frame.s.numpy(), frame.segment.numpy()
    assert np.isnan(s[bad]).all() and (seg[bad] == -1).all()
    assert np.isfinite(s[good]).all() and (seg[good] >= 0).all()

    checker = CollisionChecker(t, max_boxes=1, method="segments")
    ori = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (E, 1)).astype(np.float32),
                   dtype=wp.quatf, device=DEV)
    he = wp.array(np.full((E, 2), 0.01, np.float32), dtype=wp.vec2f, device=DEV)
    contact = checker.query(position, ori, he)
    assert (contact.oob.numpy()[bad] == 1).all()
    assert (contact.boundary.numpy()[bad] == -1).all()
    assert np.isnan(contact.distance.numpy()[bad]).all()

    cps = CheckpointSampler(t, spacing=0.3).sample()
    assert (cps.count.numpy()[bad] == 0).all() and (cps.count.numpy()[good] > 0).all()
    props = PropSampler(t, spacing=0.2).sample()
    assert (props.count.numpy()[bad] == 0).all() and (props.count.numpy()[good] > 0).all()


# ---------------------------------------------------------------------------------------
# out=, on_invalid validation, launch counts
# ---------------------------------------------------------------------------------------

def test_build_into_an_out_track_with_and_without_fail_mask():
    spec = _spec()
    ref = build(spec, seeds=5).result
    out = warp_pipeline._inflate_warp_alloc(TrackGenConfig(device=DEV, num_envs=E))[0]
    assert out.fail_mask is None
    p = build(spec, seeds=5, out=out)
    assert p.result is out and out.fail_mask is None
    assert p.tap("valid.fail").shape == (E,)                 # a work buffer
    assert dict(out.fail_bits) == dict(p.fail_bits)
    for f in FIELDS[:-1]:
        assert getattr(out, f).numpy().tobytes() == getattr(ref, f).numpy().tobytes(), f
    out2 = warp_pipeline._inflate_warp_alloc(TrackGenConfig(device=DEV, num_envs=E))[0]
    out2.fail_mask = wp.zeros(E, dtype=wp.int32, device=DEV)
    p2 = build(spec, seeds=5, out=out2)
    assert p2.tap("valid.fail") is out2.fail_mask
    assert out2.fail_mask.numpy().tobytes() == ref.fail_mask.numpy().tobytes()
    out2.fail_mask = wp.zeros(E, dtype=wp.float32, device=DEV)
    with pytest.raises(ValueError, match=r"out\.fail_mask: expected a contiguous \[4\] "
                                         r"int32"):
        build(spec, out=out2)


def test_on_invalid_is_validated():
    spec = _spec()
    bad = PipelineSpec(BatchSpec(E, DEV, 384, on_invalid="drop"), spec.stages)
    with pytest.raises(PlanError, match="BatchSpec.on_invalid must be 'keep' or "
                                        "'poison', got 'drop'"):
        build(bad)
    with pytest.raises(ValueError, match="on_invalid must be 'keep' or 'poison'"):
        _spec(on_invalid="drop")
    assert BatchSpec(E, DEV, 384).on_invalid == "keep"


def _kernels(trace) -> list:
    return [r.kernel.rsplit(".", 1)[-1] for r in trace]


def test_launch_counts():
    ctx = LaunchContext.for_call(DEV)
    default = _gen(_spec())
    with trace_launches() as plain:
        default.program.enqueue(ctx)
    assert len(plain) == 152                                    # 0.2.0.dev2's count
    assert "_hold_k" not in _kernels(plain) and "_commit_k" not in _kernels(plain)
    with trace_launches() as masked:
        default.program.enqueue(ctx, advance=True, mask=_i32(MASK))
    k = _kernels(masked)
    assert len(masked) == 152 + 1 + 2                          # + the epoch advance
    assert k.count("_hold_k") == 1 and k.count("_commit_k") == 1
    assert k.index("_hold_k") == 2                             # advance, seed, hold
    assert k[-2] == "_commit_k" and k[-1] == "_copy_masked_k"   # then generation
    poison = _gen(_spec(on_invalid="poison"))
    with trace_launches() as t:
        poison.program.enqueue(ctx)
    assert len(t) == 153 and _kernels(t)[-2] == "_commit_k"
    checked = _gen(_spec(checks=("default", KMAX)))
    with trace_launches() as t:
        checked.program.enqueue(ctx)
    # + the kappa_max latch, the check, the commit (valid only: one thread per env)
    assert len(t) == 152 + 3 and _kernels(t)[-2] == "_derive_valid_k"
    assert t[-2].dim == (E,), t[-2].dim
    assert _kernels(t)[1] == "_latch_all_f32_k" and "_kmax_k" in _kernels(t)
