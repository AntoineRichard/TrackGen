"""The result-object surface (roadmap PR13b): ``Layout``, ``validate``, ``view``,
``copy_from``, ``clone``, ``alloc``, ``out=`` (mixed batches, attributes, seeds, the
overlap registry), the capability protocols and ``check_source``, ``CourseLine ->
Centerline`` and the ``closed`` topology contract. CPU; the CUDA proofs are
tests/test_results_cuda.py."""
from __future__ import annotations

import dataclasses
import gc
import types

import numpy as np
import pytest
import warp as wp

import track_gen
from tests import _channel_toys as ctoys
from tests import _stage_toys as stoys
from tests._results_helpers import (LazyProxy, MinimalBand, MinimalCenterline,
                                    MinimalGates, bytes_of, outside_is_poison, poisoned,
                                    rows)
from track_gen import GateGenerator, TrackGenerator, plugin
from track_gen.checkpoints import CheckpointSampler, CheckpointSet
from track_gen.collision import CollisionChecker
from track_gen._src.course_line import CourseLine
from track_gen.exec import CaptureError, LaunchContext
from track_gen.heightfield import HeightFieldBaker
from track_gen.localize import TrackLocalizer, curvature, speed_profile
from track_gen.pipeline import PlanError, StageRef, build, presets
from track_gen.plugin import registry
from track_gen.props import PropSampler
from track_gen.results import (Centerline, GateSequence, HasBand, HasCenterline,
                               HasGateFrames, HasGates, HasGeneration, Layout, Track,
                               check_source)
from track_gen.testing.capture import result_arrays
from track_gen.testing.contract import hot_path_guard
from track_gen.testing.trace import trace_launches

from track_gen._src.collision_frames import FrameChecker  # noqa: E402

DEV = "cpu"
E = 4
S = 384


def _track_spec(n=E, **kw):
    return presets.track(num_envs=n, device=DEV, **kw)


def _gate_spec(n=E, **kw):
    return presets.gates(num_envs=n, device=DEV, **kw)


@pytest.fixture(scope="module")
def track():
    """A generated 4-env bezier Track (owned: a clone of a generator's result)."""
    return TrackGenerator(_track_spec(), seeds=3).program.result.clone()


@pytest.fixture(scope="module")
def gates():
    return GateGenerator(_gate_spec(), seeds=3).program.result.clone()


def _seeds(values):
    return wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=DEV)


# ---------------------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------------------

def test_layout_is_one_class_everywhere():
    assert track_gen.results.Layout is track_gen.Layout is plugin.Layout
    assert plugin.stage.Layout is Layout


@pytest.mark.parametrize("args,exc,match", [
    ((True, 3, "cpu"), TypeError, r"Layout.num_envs must be an int, got True"),
    ((2, False, "cpu"), TypeError, r"Layout.stride must be an int"),
    ((0, 3, "cpu"), ValueError, r"Layout.num_envs must be >= 1, got 0"),
    ((2, -1, "cpu"), ValueError, r"Layout.stride must be >= 1, got -1"),
    ((2.0, 3, "cpu"), TypeError, r"Layout.num_envs must be an int, got float"),
    ((2, 3, wp.get_device("cpu")), TypeError, r"Layout.device must be a device string"),
    ((2, 3, "cpu", 1), TypeError, r"Layout.closed must be a bool, got 1"),
    ((2, 3, "cpu", np.True_), TypeError, r"Layout.closed must be a bool"),
])
def test_layout_post_init(args, exc, match):
    with pytest.raises(exc, match=match):
        Layout(*args)


def test_layout_accepts_numpy_ints_and_stores_the_device_as_given():
    lay = Layout(np.int64(4), np.int32(384), "cuda")
    assert (lay.num_envs, lay.stride) == (4, 384) and type(lay.num_envs) is int
    assert lay.device == "cuda" and lay.closed is True        # never canonicalised here


def test_every_library_result_carries_its_programs_layout():
    gen = TrackGenerator(_track_spec(), seeds=1)
    assert gen.program.result.layout == gen.program.layout == Layout(E, S, "cpu", True)
    g = GateGenerator(_gate_spec(), seeds=1)
    assert g.program.result.layout == g.program.layout
    p = build(_track_spec(n=2), seeds=1)
    assert p.result.layout == p.layout == Layout(2, S, "cpu", True)


def test_a_hand_built_track_requires_and_keeps_its_layout(track):
    hand = Track(**{f.name: getattr(track, f.name) for f in dataclasses.fields(Track)
                    if f.name not in ("fail_bits", "attributes")})
    assert hand.layout == Layout(E, S, "cpu", True)
    assert hand.validate() == hand.layout


# ---------------------------------------------------------------------------------------
# Schemas and validate()
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [Track, GateSequence, Centerline])
def test_every_array_field_is_in_the_schema(cls):
    schema = {name for name, _, _ in cls._SCHEMA}
    fields = [f.name for f in dataclasses.fields(cls)]
    assert [n for n in fields if n not in ("fail_bits", "attributes", "layout")] == \
        [n for n, _, _ in cls._SCHEMA]
    assert schema <= set(fields)
    assert cls._OPTIONAL <= schema


def _replace(obj, **kw):
    out = dataclasses.replace(obj, **kw)
    return out


def _bad_dtype(arr):
    other = wp.float32 if arr.dtype is not wp.float32 else wp.int32
    return wp.zeros(arr.shape[0], dtype=other, device=DEV)


def _short(arr):
    return wp.zeros(arr.shape[0] - 1, dtype=arr.dtype, device=DEV)


def _strided(arr):
    return wp.zeros(2 * arr.shape[0], dtype=arr.dtype, device=DEV)[::2]


TRACK_FIELDS = [n for n, _, _ in Track._SCHEMA]
GATE_FIELDS = [n for n, _, _ in GateSequence._SCHEMA]


@pytest.mark.parametrize("field", TRACK_FIELDS)
@pytest.mark.parametrize("how", ["dtype", "short", "strided"])
def test_validate_rejects_every_mismatched_track_field(track, field, how):
    arr = getattr(track, field)
    bad = {"dtype": _bad_dtype, "short": _short, "strided": _strided}[how](arr)
    obj = _replace(track, **{field: bad})
    n = arr.shape[0]
    with pytest.raises(ValueError, match=rf"^Track\.{field}: expected a contiguous \[{n}\] "):
        obj.validate()



@pytest.mark.parametrize("field", GATE_FIELDS)
def test_validate_rejects_every_mismatched_gate_field(gates, field):
    arr = getattr(gates, field)
    obj = _replace(gates, **{field: _bad_dtype(arr)})
    with pytest.raises(ValueError, match=rf"^GateSequence\.{field}: expected a "
                                         rf"contiguous \[{arr.shape[0]}\] "):
        obj.validate()


def test_validate_messages_attributes_need_and_stored_layout(track):
    assert track.validate() == track.layout
    msg = (r"^Track\.inner: expected a contiguous \[1536\] vec3f \(\[E\*stride\] = "
           r"\[4\*384\]\) array on cpu, got shape \(1535,\) vec3f on cpu$")
    with pytest.raises(ValueError, match=msg):
        _replace(track, inner=_short(track.inner)).validate()
    with pytest.raises(ValueError, match=r"Track\.winding: expected a contiguous \[4\] "
                                         r"float32 \(\[E\] = \[4\]\) array on cpu, got "
                                         r"shape \(3,\) float32"):
        _replace(track, winding=_short(track.winding)).validate()
    bad = _replace(track, attributes={"acme.k": wp.zeros(5, dtype=wp.float32, device=DEV)})
    with pytest.raises(ValueError, match=r"Track\.attributes\['acme\.k'\]: expected a "
                                         r"contiguous \[1536\] array of float32 / int32 / "
                                         r"vec2f / vec3f on cpu, got shape \(5,\)"):
        bad.validate()
    bad = _replace(track, attributes={"acme.k": wp.zeros(1536, dtype=wp.quatf, device=DEV)})
    with pytest.raises(ValueError, match=r"attributes\['acme\.k'\]"):
        bad.validate()
    no_hw = _replace(track)
    no_hw.half_width = None
    with pytest.raises(ValueError, match=r"Track\.half_width is None"):
        no_hw.validate()
    with pytest.raises(ValueError, match=r"unknown fields \['nope'\]"):
        track.validate(need=("nope",))
    assert track.validate(need="half_width") == track.layout
    # a stored layout contradicting count
    with pytest.raises(ValueError, match=r"^Track\.outer: expected a contiguous \[3072\]"):
        _replace(track, layout=Layout(8, S, "cpu")).validate()
    with pytest.raises(TypeError, match=r"Track\.layout must be a track_gen\.Layout"):
        _replace(track, layout=(4, 384)).validate()
    with pytest.raises(ValueError, match=r"Track\.count is None"):
        _replace(track, count=None)



# ---------------------------------------------------------------------------------------
# The test gate: a hand-built Track with ONE mismatched field
# ---------------------------------------------------------------------------------------

def _hand(track, **kw):
    """A hand-built Track with its explicit layout and ``kw`` replaced."""
    return _replace(track, **kw)


# Roadmap PR14: the optional members a utility reads when present (``valid``: the
# invalid-env sentinels; ``generation``: the self-reset, or the copy into an owned
# result) are checked at construction like the required ones, so they are listed too.
_TRACK_CONSUMERS = {
    "CollisionChecker": (lambda t: CollisionChecker(t, 1),
                         ("center", "outer", "inner", "count", "valid")),
    "CollisionChecker[sdf]": (lambda t: CollisionChecker(t, 1, method="sdf",
                                                         sdf_resolution=8),
                              ("center", "outer", "inner", "count", "valid")),
    "PropSampler": (lambda t: PropSampler(t, 0.2), ("outer", "inner", "count", "valid")),
    "CheckpointSampler": (lambda t: CheckpointSampler(t, 0.2),
                          ("center", "inner", "outer", "count", "valid", "generation")),
    "HeightFieldBaker": (lambda t: HeightFieldBaker(t, 8),
                         ("outer", "center", "count", "valid")),
    "TrackLocalizer": (lambda t: TrackLocalizer(t), ("center", "arclen", "length",
                                                     "count", "valid", "generation")),
    "localize.curvature": (lambda t: curvature(t), ("center", "count", "valid")),
    "localize.speed_profile": (lambda t: speed_profile(t, 1.0, 1.0, 1.0, 5.0),
                               ("center", "arclen", "length", "count", "valid")),
}
_GATE_CONSUMERS = {
    "CourseLine": (lambda g: CourseLine(g), ("position", "count", "valid", "generation")),
    "FrameChecker": (lambda g: FrameChecker(g, E, 0.05, 0.01, 0.01),
                     ("position", "orientation", "half_size", "count", "valid")),
    "CheckpointSet.from_gates": (lambda g: CheckpointSet.from_gates(g),
                                 ("position", "left", "right", "forward", "half_size",
                                  "count", "valid", "generation")),
}
_TRACK_CASES = [(c, f) for c, (_, fs) in _TRACK_CONSUMERS.items() for f in fs]
_GATE_CASES = [(c, f) for c, (_, fs) in _GATE_CONSUMERS.items() for f in fs]


def _prefix(consumer: str) -> str:
    return consumer.split("[")[0]


@pytest.mark.parametrize("consumer,field", _TRACK_CASES)
def test_every_utility_rejects_a_mismatched_field_it_reads(track, consumer, field):
    make, _ = _TRACK_CONSUMERS[consumer]
    bad = _hand(track, **{field: _bad_dtype(getattr(track, field))})
    with pytest.raises(ValueError, match=rf"^{_prefix(consumer)}: Track\.{field}: expected "
                                         rf"a contiguous"):
        make(bad)
    with pytest.raises(ValueError, match=rf"^Track\.{field}: expected"):
        bad.validate()                                    # validate() rejects it too


@pytest.mark.parametrize("consumer,field", _GATE_CASES)
def test_every_gate_utility_rejects_a_mismatched_field_it_reads(gates, consumer, field):
    make, _ = _GATE_CONSUMERS[consumer]
    bad = _replace(gates, **{field: _bad_dtype(getattr(gates, field))})
    with pytest.raises(ValueError, match=rf"^{consumer}: GateSequence\.{field}: "
                                         rf"expected a contiguous"):
        make(bad)


@pytest.mark.parametrize("consumer", list(_TRACK_CONSUMERS))
def test_a_utility_accepts_a_mismatch_in_a_field_it_does_not_read(track, consumer):
    make, reads = _TRACK_CONSUMERS[consumer]
    unread = next(f for f in ("winding", "normal", "tangent", "truncated")
                  if f not in reads)
    bad = _hand(track, **{unread: _bad_dtype(getattr(track, unread))})
    with pytest.raises(ValueError):
        bad.validate()                                    # the result IS malformed
    make(bad)                                             # ... but not where it reads


def test_frame_checker_cross_checks_num_envs(gates):
    with pytest.raises(ValueError, match=r"FrameChecker: num_envs=3, but the gates hold 4 "
                                         r"envs"):
        FrameChecker(gates, 3, 0.05, 0.01, 0.01)


# ---------------------------------------------------------------------------------------
# Protocols and check_source
# ---------------------------------------------------------------------------------------

def test_protocol_membership(track, gates):
    line = CourseLine(gates).centerline
    assert isinstance(track, HasCenterline) and isinstance(track, HasBand)
    assert isinstance(track, HasGeneration)
    assert not isinstance(track, HasGates)
    assert isinstance(gates, HasGates) and isinstance(gates, HasGateFrames)
    assert isinstance(gates, HasGeneration) and not isinstance(gates, HasBand)
    assert isinstance(line, HasCenterline) and not isinstance(line, HasBand)
    assert isinstance(MinimalCenterline(track), HasCenterline)
    assert not isinstance(MinimalCenterline(track), HasBand)
    assert isinstance(MinimalBand(track), HasBand)
    assert isinstance(MinimalGates(gates, frames=True), HasGateFrames)
    assert not isinstance(MinimalGates(gates), HasGateFrames)
    from track_gen.results import _members_of
    assert _members_of(HasGateFrames) == ("position", "count", "valid", "forward", "left",
                                          "right", "orientation", "half_size")


@pytest.mark.parametrize("make", [lambda t: t, MinimalCenterline, LazyProxy],
                         ids=["track", "minimal", "proxy"])
def test_centerline_consumers_accept_any_source_with_the_members(track, make):
    src = make(track)
    lay = check_source(src, HasCenterline, consumer="test")
    assert lay == Layout(E, S, "cpu", True)
    pos = wp.array(np.zeros((E, 3), np.float32), dtype=wp.vec3f, device=DEV)
    ref = TrackLocalizer(track).query(pos)
    ref_s = ref.s.numpy().copy()
    got = TrackLocalizer(src).query(pos)
    assert got.s.numpy().tobytes() == ref_s.tobytes()
    assert curvature(src).numpy().tobytes() == curvature(track).numpy().tobytes()
    a = speed_profile(src, 1.0, 1.0, 1.0, 5.0).numpy()
    assert a.tobytes() == speed_profile(track, 1.0, 1.0, 1.0, 5.0).numpy().tobytes()


def test_band_consumers_accept_a_minimal_band_and_refuse_a_centerline(track, gates):
    band = MinimalBand(track)
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    rot = wp.array(np.tile([0, 0, 0, 1], (E, 1)).astype(np.float32), dtype=wp.quatf,
                   device=DEV)
    half = wp.array(np.full((E, 2), 0.05, np.float32), dtype=wp.vec2f, device=DEV)
    a = CollisionChecker(track, 1).query(pos, rot, half)
    da = a.distance.numpy().copy()
    b = CollisionChecker(band, 1).query(pos, rot, half)
    assert b.distance.numpy().tobytes() == da.tobytes()
    line = CourseLine(gates).centerline
    with pytest.raises(TypeError, match=r"^CollisionChecker needs a HasBand source "
                                        r"\(center, outer, inner, count; optional: "
                                        r"valid\): Centerline has no 'outer', 'inner'$"):
        CollisionChecker(line, 1)
    with pytest.raises(TypeError, match=r"^HeightFieldBaker needs a HasBand source .*: "
                                        r"Centerline has no 'outer'"):
        HeightFieldBaker(line, 8)


def test_check_source_reads_members_only_and_never_duck_calls_validate(track):
    calls = []

    class Mine:
        def __init__(self):
            self.center, self.arclen = track.center, track.arclen
            self.length, self.count = track.length, track.count

        def validate(self, *args, **kwargs):          # unrelated to track_gen's
            calls.append(args)
            raise RuntimeError("never called")

    assert check_source(Mine(), HasCenterline, consumer="x") == track.layout
    assert not calls
    with pytest.raises(TypeError, match=r"^x needs a HasCenterline source \(center, "
                                        r"arclen, length, count\): SimpleNamespace has no "
                                        r"'arclen'$"):
        check_source(types.SimpleNamespace(center=track.center, count=track.count),
                     HasCenterline, consumer="x", members=("center", "arclen"))
    # A user object's unrelated `layout` attribute is ignored (inferred instead); a
    # TrackGen result's must be a Layout or None.
    weird = types.SimpleNamespace(**vars(MinimalCenterline(track)), layout="4x384")
    assert check_source(weird, HasCenterline, consumer="x") == track.layout
    bad = dataclasses.replace(track, layout="4x384")
    with pytest.raises(TypeError, match=r"^x: Track\.layout must be a track_gen\.Layout"):
        check_source(bad, HasCenterline, consumer="x")


def test_check_source_checks_its_arguments_and_composes_protocols(track):
    """A non-protocol or members outside the protocol are the consumer's mistake
    (refused as such); a per-env-only protocol reads only its members (HasGeneration
    alone infers num_envs from generation, never count); a tuple of protocols checks
    their members against one layout."""
    with pytest.raises(TypeError, match=r"protocol must be a capability protocol"):
        check_source(track, int, consumer="x")
    with pytest.raises(TypeError, match=r"protocol must be a capability protocol"):
        check_source(track, (), consumer="x")
    with pytest.raises(ValueError, match=r"^check_source: members \['centre'\] are not "
                                         r"members of HasBand \(center, outer, inner, "
                                         r"count, valid\)$"):
        check_source(track, HasBand, consumer="x", members=("centre",))
    gen_only = types.SimpleNamespace(generation=track.generation)
    lay = check_source(gen_only, HasGeneration, consumer="Comp")
    assert (lay.num_envs, lay.stride) == (track.layout.num_envs, 1)
    both = types.SimpleNamespace(**vars(MinimalCenterline(track)),
                                 generation=track.generation)
    assert check_source(both, (HasCenterline, HasGeneration), consumer="Comp") == \
        track.layout
    with pytest.raises(TypeError, match=r"^Comp needs a HasCenterline \+ HasGeneration "
                                        r"source \(center, arclen, length, count, "
                                        r"generation\): SimpleNamespace has no "
                                        r"'generation'$"):
        check_source(MinimalCenterline(track), (HasCenterline, HasGeneration),
                     consumer="Comp")
    short = types.SimpleNamespace(**vars(MinimalCenterline(track)),
                                  generation=track.generation[:2])
    with pytest.raises(ValueError, match=r"^Comp: SimpleNamespace\.generation: expected "
                                         r"a contiguous \[4\] int32"):
        check_source(short, (HasCenterline, HasGeneration), consumer="Comp")


def test_a_two_dimensional_member_is_named_as_such(track):
    E, S = track.layout.num_envs, track.layout.stride
    flat = MinimalCenterline(track)
    two_d = types.SimpleNamespace(**{**vars(flat), "center": track.center.reshape((E, S))})
    msg = (r"SimpleNamespace\.center: expected a flat 1-D \[E\*stride\] array, got a "
           r"2-D array of shape \(4, 384\) \(flatten it")
    with pytest.raises(ValueError, match=r"^x: " + msg):
        check_source(two_d, HasCenterline, consumer="x")
    with pytest.raises(ValueError, match=r"^Track\.center: expected a flat 1-D"):
        dataclasses.replace(track, center=track.center.reshape((E, S))).validate()


def test_utilities_bind_the_members_they_checked(track):
    """A source's members are read once, at construction: a member reassigned later is
    neither re-read nor re-validated (stable pointers, no proxy work per call)."""
    src = MinimalBand(track)
    checker = CollisionChecker(src, max_boxes=1)
    held = checker._track.center
    assert held.ptr == track.center.ptr
    src.center = wp.zeros(3, dtype=wp.vec3f, device=track.center.device)   # malformed
    assert checker._track.center is held
    proxy = LazyProxy(MinimalCenterline(track))
    loc = TrackLocalizer(proxy)
    assert loc._track.center.ptr == track.center.ptr and \
        not isinstance(loc._track, LazyProxy)


def test_course_line_produces_a_centerline(gates):
    line = CourseLine(gates)
    c = line.centerline
    assert type(c) is Centerline and c.layout == Layout(E, 8 * 32, "cpu", True)
    line.refresh()
    assert c.validate() == c.layout
    assert not hasattr(line, "track")
    assert not isinstance(c, Track)
    assert not hasattr(c, "outer")


# ---------------------------------------------------------------------------------------
# view / copy_from / clone / alloc
# ---------------------------------------------------------------------------------------

def test_view_is_a_zero_copy_env_slice(track):
    t = _replace(track, attributes={"acme.k": wp.zeros(E * S, dtype=wp.vec2f, device=DEV)})
    v = t.view(1, 3)
    assert v.layout == Layout(2, S, "cpu", True) and v._env_offset == 1
    item = 12
    assert v.center.ptr - t.center.ptr == 1 * S * item
    assert v.count.ptr - t.count.ptr == 4 and v.count.shape == (2,)
    assert v.half_width.ptr == t.half_width.ptr + S * 4
    assert v.fail_bits is t.fail_bits
    assert v.attributes["acme.k"].ptr - t.attributes["acme.k"].ptr == S * 8
    assert v.center._ref is t.center
    vv = v.view(1, 2)
    assert vv._env_offset == 2 and vv.layout.num_envs == 1
    assert vv.center.ptr == t.center.ptr + 2 * S * item
    for e0, e1 in ((-1, 2), (0, 5), (5, 6)):
        with pytest.raises(IndexError, match=rf"Track\.view\({e0}, {e1}\): env range "
                                             rf"outside \[0, 4\)"):
            t.view(e0, e1)
    for e0, e1 in ((0, 0), (2, 2), (3, 2)):
        with pytest.raises(IndexError, match=rf"^Track\.view\({e0}, {e1}\): empty env "
                                             rf"range: need e0 < e1$"):
            t.view(e0, e1)
    with pytest.raises(TypeError, match="env indices must be ints"):
        t.view(True, 2)

    class Sub(Track):
        pass
    s = Sub(**{f.name: getattr(t, f.name) for f in dataclasses.fields(Track)})
    assert type(s.view(0, 1)) is Sub and type(s.clone()) is Sub


def test_utilities_read_a_view_as_an_owned_copy(track):
    v = track.view(1, 3)
    owned = v.clone()
    pos = wp.array(np.zeros((2, 3), np.float32), dtype=wp.vec3f, device=DEV)
    assert TrackLocalizer(v).query(pos).s.numpy().tobytes() == \
        TrackLocalizer(owned).query(pos).s.numpy().tobytes()
    assert curvature(v).numpy().tobytes() == curvature(owned).numpy().tobytes()
    a = HeightFieldBaker(v, 8).bake().height.numpy().copy()
    assert a.tobytes() == HeightFieldBaker(owned, 8).bake().height.numpy().tobytes()
    a = PropSampler(v, 0.2).sample().position.numpy().copy()
    assert a.tobytes() == PropSampler(owned, 0.2).sample().position.numpy().tobytes()


def test_copy_from_copies_without_allocating(track):
    snap = Track.alloc(track.layout)
    with hot_path_guard() as guard:
        snap.copy_from(track)
    assert guard.ok, guard
    assert bytes_of(snap) == bytes_of(track)
    with trace_launches() as tr:
        snap.copy_from(track)
    held = sum(getattr(track, n) is not None for n in TRACK_FIELDS)
    assert len(tr) == held                                   # one copy per held field
    snap.copy_from(snap)                                     # a no-op
    half = Track.alloc(track.view(0, 2).layout)
    half.copy_from(track.view(2, 4))
    assert bytes_of(half)["center"] == rows(track.center, 2, 4, S)


def test_copy_from_errors(track, gates):
    snap = Track.alloc(track.layout)
    with pytest.raises(TypeError, match=r"Track\.copy_from: src must be a Track, got "
                                        r"GateSequence"):
        snap.copy_from(gates)
    with pytest.raises(ValueError, match=r"Track\.copy_from: layouts differ: this Track is "
                                         r"Layout\(4, 384, 'cpu', True\), src is "
                                         r"Layout\(2, 384, 'cpu', True\)"):
        snap.copy_from(track.view(0, 2))
    corrupted = _replace(track)
    corrupted.half_width = None
    with pytest.raises(ValueError, match=r"Track\.half_width is None"):
        snap.copy_from(corrupted)
    mine = _replace(snap, attributes={"acme.k": wp.zeros(E * S, dtype=wp.float32, device=DEV)})
    with pytest.raises(ValueError, match=r"src has no attribute 'acme\.k'"):
        mine.copy_from(track)
    other = _replace(track, attributes={"acme.k": wp.zeros(E * S, dtype=wp.int32, device=DEV)})
    with pytest.raises(ValueError, match=r"attributes\['acme\.k'\] is float32 here, int32 "
                                         r"in src"):
        mine.copy_from(other)
    with pytest.raises(TypeError, match="pass either stream= or ctx=, not both"):
        snap.copy_from(track, stream="torch", ctx=LaunchContext.for_call(DEV))
    big = Track.alloc(Layout(3, S, DEV))
    with pytest.raises(ValueError, match=r"Track\.copy_from: outer partially overlaps "
                                         r"src\.outer"):
        big.view(0, 2).copy_from(big.view(1, 3))
    with pytest.raises(ValueError, match="layouts differ"):
        snap.copy_from(Track.alloc(Layout(E, 192, DEV)))


def test_clone_owns_its_memory_and_shares_fail_bits(track):
    gen = TrackGenerator(_track_spec(), seeds=3)
    res = gen.program.result
    c = res.clone()
    assert type(c) is Track and c.layout == res.layout
    for name, arr in result_arrays(res).items():
        mine = result_arrays(c)[name]
        assert mine.ptr != arr.ptr
        assert mine.numpy().tobytes() == wp.clone(arr).numpy().tobytes(), name
    assert c.fail_bits is res.fail_bits
    gen.generate(advance=True)
    assert c.center.numpy().tobytes() != res.center.numpy().tobytes()
    with pytest.raises(ValueError, match=r"^Track\.inner"):
        _replace(track, inner=_short(track.inner)).clone()


def test_alloc_fills_the_no_track_state():
    t = Track.alloc(Layout(2, 5, "cpu"), attributes={"acme.f": wp.float32,
                                                     "acme.i": wp.int32,
                                                     "acme.v": wp.vec2f})
    assert t.layout == Layout(2, 5, "cpu", True) and dict(t.fail_bits) == {}
    for name, per, dtype in Track._SCHEMA:
        a = getattr(t, name).numpy()
        if per == "env":
            assert (a == 0).all(), name
        else:
            assert np.isnan(a).all(), name
    assert np.isnan(t.attributes["acme.f"].numpy()).all()
    assert (t.attributes["acme.i"].numpy() == -1).all()
    assert np.isnan(t.attributes["acme.v"].numpy()).all()
    g = GateSequence.alloc(Layout(2, 4, "cpu"))
    assert np.isnan(g.orientation.numpy()).all() and (g.count.numpy() == 0).all()
    c = Centerline.alloc(Layout(2, 4, "cpu"))
    assert c.validate() == c.layout
    with pytest.raises(TypeError, match=r"Centerline\.alloc: a Centerline has no "
                                        r"attributes"):
        Centerline.alloc(Layout(2, 4, "cpu"), attributes={"acme.f": wp.float32})
    with pytest.raises(TypeError, match=r"attributes\['acme\.q'\] must be float32"):
        Track.alloc(Layout(2, 4, "cpu"), attributes={"acme.q": wp.quatf})
    # The dtype names of ExternalParams.attributes are accepted too; keys are plugin keys.
    named = Track.alloc(Layout(2, 4, "cpu"), attributes={"acme.x": "float32",
                                                         "acme.n": "int32"})
    assert named.attributes["acme.x"].dtype is wp.float32
    assert named.attributes["acme.n"].dtype is wp.int32
    with pytest.raises(TypeError, match=r"attributes\['acme\.q'\] must be float32 / int32 "
                                        r"/ vec2f / vec3f \(a wp dtype or its name\), got "
                                        r"the str 'quatf'"):
        Track.alloc(Layout(2, 4, "cpu"), attributes={"acme.q": "quatf"})
    with pytest.raises(ValueError, match=r"attribute key 'nodot' must be a plugin key "
                                         r"'<package>\.<name>'"):
        Track.alloc(Layout(2, 4, "cpu"), attributes={"nodot": wp.float32})
    with pytest.raises(TypeError, match=r"layout must be a track_gen\.Layout"):
        Track.alloc((2, 4))


def test_alloc_and_clone_refuse_a_capture(track):
    with wp.ScopedCapture(device=DEV):
        with pytest.raises(CaptureError, match=r"Track\.alloc\(\) allocates on Warp's "
                                               r"current stream"):
            Track.alloc(track.layout)
        with pytest.raises(CaptureError, match=r"Track\.clone\(\) allocates"):
            track.clone()
    snap = Track.alloc(track.layout)
    with wp.ScopedCapture(device=DEV) as cap:
        snap.copy_from(track)                       # recorded, no allocation
    assert cap.graph is not None


# ---------------------------------------------------------------------------------------
# out=: mixed batches
# ---------------------------------------------------------------------------------------

def _assert_half(big, e0, e1, ref, label):
    """Envs [e0, e1) of ``big`` equal ``ref`` (a standalone result), byte for byte."""
    S_ = big.layout.stride
    for name, arr in result_arrays(ref).items():
        mine = result_arrays(big)[name]
        per_point = arr.shape[0] != ref.layout.num_envs
        assert rows(mine, e0, e1, S_, per_point) == np.ascontiguousarray(
            arr.numpy()).tobytes(), f"{label}: {name}"


def test_two_generators_write_disjoint_halves_of_one_track():
    seeds = np.arange(100, 108, dtype=np.int32)
    big = poisoned(Track.alloc(Layout(8, S, DEV)))
    a = TrackGenerator(_track_spec(generator="bezier"), seeds=_seeds(seeds[:4]),
                       out=big.view(0, 4))
    b = TrackGenerator(_track_spec(generator="polar"), seeds=_seeds(seeds[4:]),
                       out=big.view(4, 8))
    ra = TrackGenerator(_track_spec(generator="bezier"), seeds=_seeds(seeds[:4]))
    rb = TrackGenerator(_track_spec(generator="polar"), seeds=_seeds(seeds[4:]))
    assert a.program.result.center.ptr == big.center.ptr
    assert b.program.result.center.ptr == big.center.ptr + 4 * S * 12
    for label in ("prepare", "plain", "advance"):
        if label == "plain":
            for g in (a, b, ra, rb):
                g.generate()
        if label == "advance":
            for g in (a, b, ra, rb):
                g.generate(advance=True)
        _assert_half(big, 0, 4, ra.program.result, f"{label} a")
        _assert_half(big, 4, 8, rb.program.result, f"{label} b")
    # masked regeneration of one half: the other half stays bitwise
    before = bytes_of(big)
    m = wp.array([1, 0, 1, 0], dtype=wp.int32, device=DEV)
    a.generate(advance=True, mask=m)
    ra.generate(advance=True, mask=wp.array([1, 0, 1, 0], dtype=wp.int32, device=DEV))
    _assert_half(big, 0, 4, ra.program.result, "masked a")
    after = bytes_of(big)
    for name in after:
        arr = result_arrays(big)[name]
        per_point = arr.shape[0] != 8
        n = S if per_point else 1
        item = len(after[name]) // arr.shape[0]
        lo = 4 * n * item
        assert after[name][lo:] == before[name][lo:], name
        # a's unmasked envs (1, 3) kept too
        for e in (1, 3):
            assert after[name][e * n * item:(e + 1) * n * item] == \
                before[name][e * n * item:(e + 1) * n * item], (name, e)
    assert dict(a.program.result.fail_bits) == dict(a.program.fail_bits)
    assert dict(big.fail_bits) == {}                 # the parent's stays empty


def test_poison_on_a_view_and_gate_mixed_batch():
    big = poisoned(Track.alloc(Layout(8, S, DEV)))
    spec = _track_spec(generator="bezier", on_invalid="poison", spacing=0.001)
    a = TrackGenerator(spec, seeds=5, out=big.view(2, 6))
    ref = TrackGenerator(spec, seeds=5 + 2)
    _assert_half(big, 2, 6, ref.program.result, "poison")
    assert (big.count.numpy()[2:6] == 0).all() and a.program.result.valid.numpy().sum() == 0
    for name, arr in result_arrays(big).items():
        assert outside_is_poison(arr, 2, 6, S, arr.shape[0] != 8), name
    gs = poisoned(GateSequence.alloc(Layout(8, 32, DEV)))
    seeds = np.arange(7, 15, dtype=np.int32)
    ga = GateGenerator(_gate_spec(generator="bezier"), seeds=_seeds(seeds[:4]),
                       out=gs.view(0, 4))
    gb = GateGenerator(_gate_spec(generator="polar"), seeds=_seeds(seeds[4:]),
                       out=gs.view(4, 8))
    rga = GateGenerator(_gate_spec(generator="bezier"), seeds=_seeds(seeds[:4]))
    rgb = GateGenerator(_gate_spec(generator="polar"), seeds=_seeds(seeds[4:]))
    for g in (ga, gb, rga, rgb):
        g.generate(advance=True)
    _assert_half(gs, 0, 4, rga.program.result, "gates a")
    _assert_half(gs, 4, 8, rgb.program.result, "gates b")


_E2_CASES = {
    "track/bezier": lambda n: _track_spec(n, generator="bezier"),
    "track/polar-noise": lambda n: _track_spec(n, generator="polar",
                                               elevation=("noise", {"z_max": 0.3})),
    "track/hull-no-relax": lambda n: _track_spec(n, generator="hull", relax=None),
    "track/ribbon": lambda n: _track_spec(n, generator="acme.ribbon"),
    "track/helix": lambda n: _track_spec(n, generator="acme.helix_loop"),
    "track/taubin+kmax": lambda n: _track_spec(n, relax=stoys.TAUBIN,
                                               checks=("default", stoys.KMAX)),
    "gates/bezier": lambda n: _gate_spec(n, generator="bezier"),
    "gates/stations-ribbon": lambda n: _gate_spec(n, generator="acme.ribbon",
                                                  source="stations"),
}


@pytest.mark.parametrize("case", sorted(_E2_CASES))
def test_build_into_a_poisoned_view_writes_exactly_its_rows(case):
    with registry.scoped():
        ctoys.register(registry)
        stoys.register(registry)
        make = _E2_CASES[case]
        spec = make(3)
        ref = build(make(3), seeds=1002)
        res = ref.result
        attrs = {k: info.dtype for k, info in ref.attributes.items()}
        Sx = res.layout.stride
        cls = type(res)
        big = poisoned(cls.alloc(Layout(7, Sx, DEV), attributes=attrs))
        prog = build(spec, out=big.view(2, 5), seeds=1000)        # offset: 1000 + 2
        m_ref = wp.array([1, 0, 1], dtype=wp.int32, device=DEV)
        m_view = wp.array([1, 0, 1], dtype=wp.int32, device=DEV)
        for step in ("prepare", "plain", "masked"):
            if step == "plain":
                prog.run()
                ref.run()
            if step == "masked":
                prog.run(advance=True, mask=m_view)
                ref.run(advance=True, mask=m_ref)
            _assert_half(big, 2, 5, res, f"{case}: {step}")
            for name, arr in result_arrays(big).items():
                assert outside_is_poison(arr, 2, 5, Sx, arr.shape[0] != 7), (case, name)


def test_out_errors_layout_and_type():
    spec = _track_spec()
    with pytest.raises(TypeError, match=r"out= must be a Track \(this pipeline's result\), "
                                        r"got GateSequence"):
        build(spec, out=GateSequence.alloc(Layout(E, 32, DEV)))
    with pytest.raises(ValueError, match=r"^out: a Track for Layout\(2, 384, 'cpu', True\); "
                                         r"this pipeline writes Layout\(4, 384, 'cpu', "
                                         r"True\)$"):
        build(spec, out=Track.alloc(Layout(2, S, DEV)))
    hand = Track.alloc(Layout(E, S, DEV))
    p = build(spec, out=hand, seeds=1)
    assert hand.layout == p.layout
    assert p.result is hand
    with pytest.raises(ValueError, match=r"^out\.outer: expected a contiguous \[1536\]"):
        build(spec, out=_replace(Track.alloc(Layout(E, S, DEV)),
                                 outer=wp.zeros(5, dtype=wp.vec3f, device=DEV)))
    corrupted = Track.alloc(Layout(E, S, DEV))
    corrupted.truncated = None
    with pytest.raises(ValueError, match=r"^out\.truncated is None"):
        build(spec, out=corrupted)


def test_a_failed_build_leaves_no_overlap_entry(monkeypatch):
    """The overlap registry records a result only once its Program (or facade) is
    usable: a warm-up that raises does not refuse a retry with the same out=, even while
    the exception (and its traceback holding the half-built Program) is kept."""
    from track_gen.pipeline.program import Program  # noqa: PLC0415

    big = Track.alloc(Layout(8, S, DEV))
    real = Program.prepare

    def boom(self, *a, **k):
        raise RuntimeError("warm-up failed")

    monkeypatch.setattr(Program, "prepare", boom)
    with pytest.raises(RuntimeError, match="warm-up failed") as build_exc:
        build(_track_spec(), out=big.view(0, 4), seeds=1)
    with pytest.raises(RuntimeError, match="warm-up failed") as facade_exc:
        TrackGenerator(_track_spec(), seeds=1, out=big.view(0, 4))
    monkeypatch.setattr(Program, "prepare", real)
    p = build(_track_spec(), out=big.view(0, 4), seeds=1)      # no spurious PlanError
    assert build_exc.value is not None and facade_exc.value is not None
    with pytest.raises(PlanError, match=r"already written by a live Program"):
        build(_track_spec(), out=big.view(0, 4), seeds=1)
    del p


def test_default_and_int_seeds_on_views_are_offset_like_one_batch():
    one = TrackGenerator(_track_spec(8), seeds=None).program.result
    for base in (None, 40):
        big = Track.alloc(Layout(8, S, DEV))
        a = TrackGenerator(_track_spec(), seeds=base, out=big.view(0, 4))
        b = TrackGenerator(_track_spec(), seeds=base, out=big.view(4, 8))
        ref = one if base is None else TrackGenerator(_track_spec(8), seeds=base) \
            .program.result
        assert bytes_of(big) == bytes_of(ref), base
        assert a.seeds.seeds.numpy().tolist() == [(base or 0) + e for e in range(4)]
        assert b.seeds.seeds.numpy().tolist() == [(base or 0) + 4 + e for e in range(4)]
    # build() too; and the offset of a view of a view
    big = Track.alloc(Layout(8, S, DEV))
    p = build(_track_spec(2), out=big.view(4, 8).view(1, 3), seeds=7)
    assert p.seeds.seeds.numpy().tolist() == [12, 13]
    # an explicit array is bound as given (no offset)
    arr = _seeds([3, 4])
    p2 = build(_track_spec(2), out=Track.alloc(Layout(8, S, DEV)).view(6, 8), seeds=arr)
    assert p2.seeds.seeds is arr


def test_overlapping_views_of_live_programs_are_refused():
    big = Track.alloc(Layout(8, S, DEV))
    a = build(_track_spec(), out=big.view(0, 4), seeds=1)
    with pytest.raises(PlanError, match=r"^out: envs \[2, 6\) of this storage are already "
                                        r"written by a live Program \(envs \[0, 4\) of the "
                                        r"same allocation\); views given to different "
                                        r"generators must be disjoint; to rebuild into "
                                        r"the same storage, release the earlier generator "
                                        r"/ Program first \(del gen\)$"):
        build(_track_spec(), out=big.view(2, 6), seeds=1)
    build(_track_spec(), out=big.view(4, 8), seeds=1)        # disjoint: fine
    del a
    gc.collect()
    build(_track_spec(), out=big.view(0, 4), seeds=1)        # freed: fine again
    gen = TrackGenerator(_track_spec(), seeds=1)
    with pytest.raises(PlanError, match=r"already written by a live Program"):
        build(_track_spec(), out=gen.program.result, seeds=1)
    with pytest.raises(PlanError, match=r"already written by a live Program"):
        TrackGenerator(_track_spec(2), out=gen.program.result.view(1, 3))


# ---------------------------------------------------------------------------------------
# out.attributes: bound iff exported, else padded once
# ---------------------------------------------------------------------------------------

def test_foreign_attribute_keys_are_padded_and_stay_padded():
    with registry.scoped():
        ctoys.register(registry)
        big = Track.alloc(Layout(8, S, DEV), attributes={"acme.grip": wp.float32,
                                                          "acme.surface_id": wp.int32})
        poisoned(big)
        rib = TrackGenerator(_track_spec(generator="acme.ribbon"), seeds=1,
                             out=big.view(0, 4))
        bez = TrackGenerator(_track_spec(generator="bezier", on_invalid="poison"),
                             seeds=1, out=big.view(4, 8))
        grip = big.attributes["acme.grip"]
        assert rib.program.tap("cl.acme.grip").ptr == grip.ptr
        assert "acme.grip" in rib.program.attributes
        assert dict(bez.program.attributes) == {}
        for step in ("build", "plain", "masked"):
            if step == "plain":
                rib.generate()
                bez.generate()
            if step == "masked":
                m = wp.array([1, 0, 1, 1], dtype=wp.int32, device=DEV)
                bez.generate(advance=True, mask=m)
                rib.generate(advance=True, mask=m)
            g = grip.numpy().reshape(8, S)
            assert np.isnan(g[4:]).all(), step                # bezier rows: padding
            assert np.isfinite(g[:4, 0]).all(), step          # ribbon rows: values
            sid = big.attributes["acme.surface_id"].numpy().reshape(8, S)
            assert (sid[4:] == -1).all(), step


def test_a_carried_but_not_exported_key_is_padded_not_bound():
    with registry.scoped():
        ctoys.register(registry)
        hw_all = wp.zeros(5 * S, dtype=wp.float32, device=DEV)
        hw = hw_all[S:]
        parent = Track.alloc(Layout(5, S, DEV))
        parent.attributes = {"acme.hw": hw_all}
        out = parent.view(1, 5)                    # a view: a mixed batch's share
        spec = _track_spec(generator="acme.ribbon", on_invalid="poison",
                           width=StageRef("width", "from_channel",
                                          ports={"src": "cl.acme.hw"}))
        p = build(spec, out=out, seeds=1)
        assert "acme.hw" not in p.attributes
        assert p.tap("cl.acme.hw") is not hw and p.tap("cl.acme.hw").ptr != hw.ptr
        assert np.isnan(hw.numpy()).all()
        p.run(advance=True, mask=wp.array([1, 0, 1, 0], dtype=wp.int32, device=DEV))
        p.run()
        assert np.isnan(hw.numpy()).all()
        del p
        gc.collect()
        bad = Track.alloc(Layout(5, S, DEV))
        bad.attributes = {"acme.hw": wp.zeros(5 * S, dtype=wp.int32, device=DEV)}
        with pytest.raises(ValueError, match=r"out\.attributes\['acme\.hw'\] is int32; "
                                             r"this pipeline's 'acme\.hw' attribute is "
                                             r"float32"):
            build(spec, out=bad.view(1, 5), seeds=1)
        # On the whole of its allocation (no other generator can write the key), an
        # unexported key is refused as in 0.2.0.dev4: a typo must not NaN-fill a buffer.
        whole = Track.alloc(Layout(4, S, DEV))
        whole.attributes = {"acme.hw": wp.zeros(4 * S, dtype=wp.float32, device=DEV)}
        with pytest.raises(ValueError, match=r"^out\.attributes\['acme\.hw'\]: this "
                                             r"pipeline does not export 'acme\.hw'; "
                                             r"exported: \['acme\.grip', "):
            build(spec, out=whole, seeds=1)
        assert not np.isnan(whole.attributes["acme.hw"].numpy()).any()   # not filled
        q = Track.alloc(Layout(4, S, DEV))
        q.attributes = {"acme.k": wp.zeros(4 * S, dtype=wp.quatf, device=DEV)}
        with pytest.raises(ValueError, match=r"out\.attributes\['acme\.k'\]: expected a "
                                             r"contiguous \[1536\] array of float32"):
            build(spec, out=q, seeds=1)


# ---------------------------------------------------------------------------------------
# closed: the 0.x topology contract
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, kw_only=True)
class _OpenAwareParams(plugin.Params):
    def validate(self, layout):
        if not layout.closed:
            raise AssertionError("Params.validate must not run for closed=False")


class _AnyRelax(plugin.Stage):
    kind = "relax"
    name = "acme.any_relax"
    topology = "any"
    Params = _OpenAwareParams
    reads = {"xy": "cl.xy"}
    writes = {"xy": "cl.xy"}

    def build(self, b, p):
        x, out = b.read("xy"), b.write("xy", wp.vec2f, per="cl")
        return lambda ctx: ctx.copy(out, x)


def _open(spec):
    return dataclasses.replace(spec, batch=dataclasses.replace(spec.batch, closed=False))


def test_closed_false_is_refused_naming_the_closed_only_stages():
    with registry.scoped():
        registry.add(_AnyRelax)
        spec = _open(_track_spec(relax="acme.any_relax"))
        msg = (r"^BatchSpec\.closed=False: open tracks are not supported in 0\.x: stages "
               r"'generator' \(bezier\), 'resample' \(resample\.constant_spacing\), "
               r"'width' \(width\.constant\), 'uniform' \(resample\.uniform\), 'frames' "
               r"\(frames\.default\), 'offset' \(offset\.default\), 'elevation' "
               r"\(elevation\.flat\), 'validity' \(validity\.default\), 'lift' "
               r"\(lift\.flat\) support closed loops only \(Stage\.topology='closed'\); "
               r"open topology arrives after 1\.0$")
        with pytest.raises(PlanError, match=msg):
            build(spec)
        with pytest.raises(PlanError, match=msg):
            TrackGenerator(spec)
        gspec = _open(_gate_spec())
        with pytest.raises(PlanError, match=r"stages 'generator' \(bezier\), 'order' "
                                            r"\(order\.ccw\)"):
            GateGenerator(gspec)
        with pytest.raises(PlanError, match=r"open tracks"):
            build(gspec)
        one = dataclasses.replace(spec, batch=dataclasses.replace(spec.batch, closed=1))
        with pytest.raises(PlanError, match=r"^BatchSpec\.closed must be a bool, got 1$"):
            build(one)


def _opened(obj):
    lay = obj.validate()
    return _replace(obj, layout=dataclasses.replace(lay, closed=False))


@pytest.mark.parametrize("consumer", sorted(_TRACK_CONSUMERS))
def test_every_track_utility_refuses_an_open_source(track, consumer):
    make, _ = _TRACK_CONSUMERS[consumer]
    with pytest.raises(ValueError, match=rf"^{_prefix(consumer)} supports closed loops only "
                                         rf"\(topology='closed'\); its source has "
                                         rf"layout\.closed=False$"):
        make(_opened(track))


@pytest.mark.parametrize("consumer", ["CourseLine", "FrameChecker"])
def test_gate_utilities_refuse_an_open_source(gates, consumer):
    make, _ = _GATE_CONSUMERS[consumer]
    with pytest.raises(ValueError, match=rf"^{consumer} supports closed loops only"):
        make(_opened(gates))


def test_components_declare_their_topology():
    for cls in (CollisionChecker, PropSampler, CheckpointSampler, HeightFieldBaker,
                TrackLocalizer, CourseLine, FrameChecker):
        assert cls.topology == "closed", cls


def test_copy_from_across_topologies_is_a_layout_mismatch(track):
    snap = Track.alloc(track.layout)
    with pytest.raises(ValueError, match=r"layouts differ: .*Layout\(4, 384, 'cpu', "
                                         r"False\)"):
        snap.copy_from(_opened(track))
