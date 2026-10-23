"""Gate validity as fail bits and the gate commit (roadmap PR11, CPU).

``validity.gate_default`` keeps the fused ``_finalize_validity_k`` (``valid`` byte for
byte) and writes one ``valid.fail`` bit per failed check (``GATE_FAIL_BITS``); plugin
gate checks take bits 16-30. The gate Program commits its ``GateSequence`` with the
PR10b hold / restore epilogue: a masked regeneration never writes an unmasked env (also
when the stages recompute it differently), an all-zero mask is an exact no-op, and
``on_invalid="poison"`` leaves an invalid env with ``count = 0``, NaN per-gate fields and
``valid = 0``.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests import _gate_toys as gt
from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
from track_gen._src import warp_gate
from track_gen.checkpoints import CheckpointSet
from track_gen.exec import LaunchContext
from track_gen.pipeline import GATE_FAIL_BITS, presets
from track_gen.plugin import registry
from track_gen.progress import ProgressTracker
from track_gen.testing import trace_launches
from track_gen.testing.capture import poison_all

DEV = "cpu"
E = 4
MASK = (1, 0, 1, 0)
FIELDS = ("position", "tangent", "forward", "orientation", "half_size", "left", "right",
          "valid", "count", "generation", "fail_mask")
POINT = ("position", "tangent", "forward", "orientation", "half_size", "left", "right")


@pytest.fixture(autouse=True)
def toys():
    gt.register(registry)


def _rng(seed=11):
    return PerEnvSeededRNG(seeds=seed, num_envs=E, device=DEV)


def _gen(seed=11, **kw):
    gen = GateGenerator(presets.gates(num_envs=E, device=DEV, **kw), _rng(seed))
    gen.generate()
    return gen


def _i32(values):
    return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=DEV)


def _host(seq) -> dict:
    return {f: getattr(seq, f).numpy().copy() for f in FIELDS}


def _rows(host: dict, e: int) -> dict:
    return {f: v.reshape(E, -1)[e].tobytes() for f, v in host.items()}


def _count_split() -> int:
    counts = _gen().program.result.count.numpy()
    uniq = np.unique(counts)
    assert uniq.size >= 2, counts
    return int(uniq[uniq.size // 2])


# ---------------------------------------------------------------------------------------
# The bits
# ---------------------------------------------------------------------------------------

def test_gate_fail_bits():
    assert dict(GATE_FAIL_BITS) == {"count": 0, "finite": 1, "tangent": 2, "spacing": 3,
                                    "crossing": 4, "grade": 5}
    g = _gen()
    assert dict(g.program.fail_bits) == dict(GATE_FAIL_BITS)
    assert dict(g._gate_sequence.fail_bits) == dict(GATE_FAIL_BITS)
    with pytest.raises(TypeError):
        GATE_FAIL_BITS["x"] = 6
    seq = g._gate_sequence
    np.testing.assert_array_equal(seq.valid.numpy(), (seq.fail_mask.numpy() == 0))
    clone = seq.clone()
    assert clone.fail_mask.ptr != seq.fail_mask.ptr and clone.fail_bits is seq.fail_bits


def _sequence(points, *, tangents=None, count=None, G=4):
    """A hand-built 1-env GateSequence (the frames / validity launches run on it)."""
    cfg = GateGenConfig(num_envs=1, max_gates=G, min_gates=2, gate_radius=0.025,
                        gate_width=0.0)
    seq = warp_gate.alloc_gate_sequence(cfg)
    pts = np.full((G, 3), np.nan, np.float32)
    pts[:len(points)] = points
    seq.position.assign(pts)
    tan = np.full((G, 3), np.nan, np.float32)
    if tangents is None:
        n = len(points)
        p = np.asarray(points, np.float32)
        tangents = [p[(i + 1) % n] - p[(i - 1) % n] for i in range(n)]
    tan[:len(tangents)] = tangents
    seq.tangent.assign(tan)
    seq.count.assign(np.array([len(points) if count is None else count], np.int32))
    return seq, cfg


_SQUARE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]


@pytest.mark.parametrize("case,bit", [
    ("ok", None), ("count", "count"), ("finite", "finite"), ("tangent", "tangent"),
    ("spacing", "spacing"), ("crossing", "crossing"), ("grade", "grade")])
def test_each_check_sets_exactly_its_bit(case, bit):
    points, kw = [list(p) for p in _SQUARE], {}
    tangents = None
    cfg_kw = {}
    if case == "count":
        kw["count"] = 1
        points = points[:1]
        tangents = [[1.0, 0.0, 0.0]]
    elif case == "finite":
        points[1] = [np.inf, 0.0, 0.0]
        tangents = [[1.0, 0.0, 0.0]] * 4
    elif case == "tangent":
        tangents = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    elif case == "spacing":
        points[1] = [0.01, 0.0, 0.0]
    elif case == "crossing":
        points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 0.0], [5.0, 6.0, 0.0]]
        tangents = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        cfg_kw["gate_width"] = 0.5
    elif case == "grade":
        points[2][2] = 3.0
        cfg_kw["z_valid_grade"] = 0.5
    seq, cfg = _sequence(points, tangents=tangents, **kw)
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    fail = wp.full(1, -1, dtype=wp.int32, device=DEV)
    warp_gate.finalize_gate_sequence(seq, cfg, fail=fail)
    want = 0 if bit is None else 1 << GATE_FAIL_BITS[bit]
    assert int(fail.numpy()[0]) == want, case
    assert int(seq.valid.numpy()[0]) == (1 if bit is None else 0)


def test_a_count_above_max_gates_stays_in_its_row():
    """An over-reported count (an anchors / order plugin writing ``count > max_gates``)
    fails the count bit only: the grade check's closing chord wraps at the slot
    capacity, never reading the next env's row (env 1 sits 10 above env 0)."""
    G = 4
    cfg = GateGenConfig(num_envs=2, max_gates=G, min_gates=2, gate_radius=0.025,
                        gate_width=0.0, z_valid_grade=0.5)
    seq = warp_gate.alloc_gate_sequence(cfg)
    sq = np.asarray(_SQUARE, np.float32)
    lifted = sq + np.float32([0.0, 0.0, 10.0])
    seq.position.assign(np.concatenate([sq, lifted]))
    tan = np.asarray([sq[(i + 1) % G] - sq[(i - 1) % G] for i in range(G)], np.float32)
    seq.tangent.assign(np.concatenate([tan, tan]))
    seq.count.assign(np.array([G + 2, G], np.int32))
    fail = wp.full(2, -1, dtype=wp.int32, device=DEV)
    warp_gate.finalize_gate_sequence(seq, cfg, fail=fail)
    assert fail.numpy().tolist() == [1 << GATE_FAIL_BITS["count"], 0]
    assert seq.valid.numpy().tolist() == [0, 1]


# ---------------------------------------------------------------------------------------
# Masked regeneration: exact by construction
# ---------------------------------------------------------------------------------------

@wp.kernel
def _shift_k(xy: wp.array(dtype=wp.vec2f), dx: float):
    t = wp.tid()
    xy[t] = xy[t] + wp.vec2f(dx, 0.0)


def _perturb(program, stage_id: str, channel: str):
    """Wrap ``stage_id``'s enqueue (test-only): after it, shift its version of
    ``channel`` for EVERY env by an amount that changes on every call."""
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


@pytest.mark.parametrize("kw", [{}, dict(checks=("gate_default", gt.MAX_HEIGHT)),
                                dict(on_invalid="poison"),
                                dict(elevation="noise", generator="polar")])
def test_a_masked_regeneration_never_writes_an_unmasked_env(kw):
    g = _gen(**kw)
    calls = _perturb(g.program, "gate_relax", "gates.xy")
    tap_before = g.program.tap("gates.xy").numpy().reshape(E, -1).copy()
    before = _host(g._gate_sequence)
    g.generate(advance=True, mask=_i32(MASK))
    after = _host(g._gate_sequence)
    tap_after = g.program.tap("gates.xy").numpy().reshape(E, -1)
    assert calls[0] == 1
    for e in range(E):
        assert not np.array_equal(tap_before[e], tap_after[e], equal_nan=True), e
        if MASK[e]:
            assert _rows(after, e) != _rows(before, e), e
            assert after["generation"][e] == 1
        else:
            assert _rows(after, e) == _rows(before, e), e


def test_an_all_zero_mask_writes_nothing():
    for kw in ({}, dict(on_invalid="poison", checks=("gate_default", gt.MAX_HEIGHT))):
        g = _gen(**kw)
        poison_all(g._gate_sequence)
        before = _host(g._gate_sequence)
        g.generate(advance=True, mask=_i32([0] * E))
        after = _host(g._gate_sequence)
        for f in FIELDS:
            assert after[f].tobytes() == before[f].tobytes(), (kw, f)
        assert g.seeds.epoch.numpy().tolist() == [0] * E


def test_masked_envs_equal_a_fresh_generator_at_the_same_epoch():
    g = _gen(on_invalid="poison", checks=("gate_default", gt.MAX_HEIGHT))
    g.generate(advance=True, mask=_i32(MASK))
    g.generate(advance=True, mask=_i32(MASK))
    twin = _gen(on_invalid="poison", checks=("gate_default", gt.MAX_HEIGHT))
    twin.seeds.advance(_i32(MASK))
    twin.seeds.advance(_i32(MASK))
    twin.generate()
    a, b = _host(g._gate_sequence), _host(twin._gate_sequence)
    for e in range(E):
        assert _rows(a, e) == _rows(b, e), e


# ---------------------------------------------------------------------------------------
# on_invalid="poison"
# ---------------------------------------------------------------------------------------

def _poisoned(keep: dict) -> dict:
    out = {k: v.copy() for k, v in keep.items()}
    bad = keep["fail_mask"] != 0
    for name in POINT:
        out[name].reshape(E, -1)[bad] = np.nan
    out["count"][bad] = 0
    out["valid"][bad] = 0
    return out


def test_poison_is_keep_with_the_invalid_envs_poisoned():
    split = _count_split()
    keep = _host(_gen(min_gates=split)._gate_sequence)
    poison = _host(_gen(min_gates=split, on_invalid="poison")._gate_sequence)
    bad = keep["fail_mask"] != 0
    assert bad.any() and not bad.all()
    assert ((keep["fail_mask"][bad] >> GATE_FAIL_BITS["count"]) & 1).all()
    want = _poisoned(keep)
    for f in FIELDS:
        assert poison[f].tobytes() == want[f].tobytes(), f
    # gate_radius=50: every env fails the spacing check
    every = _host(_gen(gate_radius=50.0, gate_solve_iters=0, on_invalid="poison")
                  ._gate_sequence)
    assert ((every["fail_mask"] >> GATE_FAIL_BITS["spacing"]) & 1).all()
    assert (every["count"] == 0).all() and (every["valid"] == 0).all()
    assert np.isnan(every["position"]).all() and np.isnan(every["orientation"]).all()


def test_the_utilities_take_their_degenerate_paths_on_a_poisoned_env():
    g = _gen(min_gates=_count_split(), on_invalid="poison")
    seq = g._gate_sequence
    bad = np.flatnonzero(seq.fail_mask.numpy() != 0)
    good = np.flatnonzero(seq.fail_mask.numpy() == 0)
    assert bad.size and good.size
    cps = CheckpointSet.from_gates(seq)
    pos = np.zeros((E, 3), np.float32)
    tracker = ProgressTracker(cps)
    ev = tracker.update(wp.array(pos, dtype=wp.vec3f, device=DEV))
    assert (ev.checkpoint_passed.numpy()[bad] == -1).all()
    assert (ev.passed.numpy()[bad] == 0).all()
    assert (cps.count.numpy()[bad] == 0).all() and (cps.count.numpy()[good] > 0).all()


# ---------------------------------------------------------------------------------------
# Launch counts
# ---------------------------------------------------------------------------------------

def _kernels(trace) -> list:
    return [r.kernel.rsplit(".", 1)[-1] for r in trace]


def test_launch_counts():
    ctx = LaunchContext.for_call(DEV)
    default = _gen()
    with trace_launches() as plain:
        default.program.enqueue(ctx)
    assert len(plain) == 14                                      # 0.2.0.dev3's count
    assert "_gate_hold_k" not in _kernels(plain) and "_gate_commit_k" not in _kernels(plain)
    polar = _gen(generator="polar")
    with trace_launches() as t:
        polar.program.enqueue(ctx)
    assert len(t) == 14             # the normalization runs out of place: no extra copy
    with trace_launches() as masked:
        default.program.enqueue(ctx, advance=True, mask=_i32(MASK))
    k = _kernels(masked)
    assert len(masked) == 14 + 1 + 2                            # + the epoch advance
    assert k.count("_gate_hold_k") == 1 and k.count("_gate_commit_k") == 1
    assert k.index("_gate_hold_k") == 2                         # advance, seed, hold
    assert k[-2] == "_gate_commit_k" and k[-1] == "_copy_masked_k"
    poison = _gen(on_invalid="poison")
    with trace_launches() as t:
        poison.program.enqueue(ctx)
    assert len(t) == 15 and _kernels(t)[-2] == "_gate_commit_k"
    checked = _gen(checks=("gate_default", gt.MAX_HEIGHT))
    with trace_launches() as t:
        checked.program.enqueue(ctx)
    # + the z_cap latch, the check, the commit (valid only: one thread per env)
    assert len(t) == 14 + 3 and _kernels(t)[-2] == "_derive_valid_k"
    assert t[-2].dim == (E,)
    assert _kernels(t)[1] == "_latch_all_f32_k" and "_max_height_k" in _kernels(t)


def test_frame_fallbacks_is_the_frames_scratch():
    g = _gen()
    assert g.frame_fallbacks is g.program._scratch["gate_frames"]["fallbacks"]
    assert g.frame_fallbacks.shape == (E,)
    assert isinstance(g._raw_anchors, wp.array) and g._raw_anchors is \
        g.program.tap("anchors.xy")
