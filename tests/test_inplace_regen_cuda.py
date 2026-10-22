"""CUDA-only: device-side reseed, masked regeneration and in-graph auto-reset (roadmap PR4).

The goal: an RL loop keeps everything inside ONE caller graph,
``[physics] -> course.step() -> done mask (device) -> course.regenerate(mask=done, advance=True)``:

- envs with ``mask[e] != 0`` advance their epoch on device and get a NEW batch on every
  replay (their per-env runtime state is reset); envs with ``mask[e] == 0`` are left
  bitwise unchanged (the track generator's commit restores the ``Track`` rows it held
  before the stages, roadmap PR10b; the gate pipeline recomputes them to identical
  values until its commit, roadmap PR11);
- an all-zero mask skips the regeneration on device (a CUDA conditional node) inside a
  Warp-registered capture where conditional nodes are supported, and is an exact no-op
  everywhere else (the body then runs unconditionally).

Method (as in tests/test_caller_capture_cuda.py): record on a NON-current side stream,
poison the outputs with FINITE, dtype-aware sentinels (outputs are NaN-padded and
``assert_array_equal`` treats NaN == NaN, so a NaN poison would hide a missed write),
replay, and compare with an eager twin driven through the same sequence outside any
capture. The twins run the body EAGERLY (``enqueue()`` / ``enqueue_generate()``, which
never self-graph; ``skip_unless`` then runs the body directly), so a defect common to
every graph recording of the masked body (a value baked at record time, a node dropped
from the conditional's child graph) cannot hide in the reference. Every recording also
runs with

- the Warp mempool disabled, so any allocation inside the capture fails loudly;
- ``wp.capture_launch`` patched to raise: a graph launch cannot be recorded into another
  capture (CUDA error 900), so a producer must re-enqueue its work.

Each recorded path is warmed up outside the capture first (construction prepares the
default body; the masked bodies, ``step()`` and the test's own kernels are launched once
explicitly), so no module is loaded while capturing.
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import weakref

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

from track_gen import GateGenConfig, PerEnvSeededRNG, TrackGenConfig  # noqa: E402
from track_gen._src.gate_generator import GateGenerator  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen.course import Course, CourseConfig  # noqa: E402
from track_gen.exec import _compat as tg_compat  # noqa: E402

DEV = "cuda:0"
E = 8
_KINDS = ("track", "gate")
_GEOMETRY = {"track": "center", "gate": "position"}
_COURSE_VARIANTS = ("track-segments", "track-sdf-heightfield", "gates-posts", "gates-frames")

_ZERO = np.zeros(E, np.int32)
# Mixed mask: neighbours differ, so a write that leaks across an env boundary shows up.
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)

_NO_CONDITIONAL = (
    "CUDA conditional graph nodes are unsupported here (wp.is_conditional_graph_supported() "
    "is False: Warp must be built with CUDA Toolkit >= 12.4 and run on a >= 12.4 driver), so "
    "a zero mask cannot skip the body on device and the 'poison stays' assertion cannot hold; "
    "the unconditional lowering (body always runs, exact no-op on a zero mask) is covered by "
    "the 'unconditional' variants")

_NONDETERMINISTIC = (
    "the eager reference is not bitwise reproducible on this device (float atomics in "
    "offset()'s shoelace-area accumulation?), so per-env bitwise comparisons cannot hold")


# ---------------------------------------------------------------------------------------
# Helpers (capture discipline as in tests/test_caller_capture_cuda.py)
# ---------------------------------------------------------------------------------------

def _arrays(obj) -> dict:
    """name -> wp.array for every array field of a result dataclass (None fields skipped)."""
    if obj is None:
        return {}
    out = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, wp.array):
            out[f.name] = v
    return out


def _poison(arr) -> None:
    """Fill with a finite, dtype-aware sentinel (on Warp's current stream)."""
    dt = arr.dtype
    if dt in (wp.int8, wp.int16, wp.int32, wp.int64):
        arr.fill_(-7)
    elif dt in (wp.float16, wp.float32, wp.float64):
        arr.fill_(-12345.0)
    else:
        n = getattr(dt, "_length_", None)
        arr.fill_(dt(*([-12345.0] * n)) if n else 7)


# ``generation`` is written only for the envs a masked regeneration selects (unmasked envs
# keep the epoch their output was built from), so it is not poisoned; the tests assert it
# against the expected epochs directly. Every other field is poisoned for EVERY env: a
# masked env must be rewritten (the poison gone), and an unmasked env must end as
# :data:`_COMMITS` says (its poisoned bytes kept, or recomputed to the twin's values).
_PER_ENV_FIELDS = ("generation",)

# Whether the generator commits a masked generation (roadmap PR10b,
# ``track_gen.stages.commit``): the track pipeline holds every unmasked env's ``Track`` rows
# before its stages and restores them after, so an unmasked env ends with exactly the bytes
# it had (a poison included). The gate pipeline has no commit yet (roadmap PR11): it
# recomputes every env, unmasked ones to identical values.
_COMMITS = {"track": True, "gate": False}


def _poison_all(*objs) -> None:
    for obj in objs:
        for name, arr in _arrays(obj).items():
            if name not in _PER_ENV_FIELDS:
                _poison(arr)
    wp.synchronize()


def _host(x) -> np.ndarray:
    return x.numpy() if isinstance(x, wp.array) else x


def _host_all(d: dict) -> dict:
    return {k: np.array(_host(v), copy=True) for k, v in d.items()}


def _assert_state(got: dict, ref: dict, label: str) -> None:
    """Bitwise equality of two name -> (wp.array | ndarray) dicts."""
    assert got.keys() == ref.keys(), f"{label}: keys differ {sorted(set(got) ^ set(ref))}"
    for name in ref:
        np.testing.assert_array_equal(_host(got[name]), _host(ref[name]),
                                      err_msg=f"{label}: {name}")


def _assert_same(got, ref, label: str) -> None:
    ref_arrays = _arrays(ref)
    assert ref_arrays, label
    for name, arr in ref_arrays.items():
        np.testing.assert_array_equal(getattr(got, name).numpy(), arr.numpy(),
                                      err_msg=f"{label}: {name}")


def _differs(a, b) -> bool:
    return any(not np.array_equal(x.numpy(), getattr(b, k).numpy(), equal_nan=True)
               for k, x in _arrays(a).items())


def _rows(arrays: dict) -> dict:
    """name -> [E, -1] host copy: row ``e`` is env ``e``'s slice of an env-major array."""
    return {k: np.array(_host(a), copy=True).reshape(E, -1) for k, a in arrays.items()}


def _assert_masked_result(kind: str, out, ref, poisoned: dict, mk, label: str) -> None:
    """``out`` after a masked generation that started from ``poisoned`` (:func:`_rows`
    of ``out`` right after :func:`_poison_all`): a masked env equals the eager twin
    ``ref``; an unmasked env still holds its poisoned bytes on a committing generator
    (:data:`_COMMITS`), and equals the twin (recomputed) otherwise. ``generation`` is
    never poisoned, so an unmasked env's ``generation`` is its previous epoch either way."""
    cur, want = _rows(_arrays(out)), _rows(_arrays(ref))
    assert cur.keys() == want.keys() == poisoned.keys(), label
    for e in range(E):
        held = mk[e] == 0 and _COMMITS[kind]
        what = "unmasked env, held by the commit" if held else "vs eager twin"
        for name, r in cur.items():
            exp = poisoned[name][e] if held else want[name][e]
            np.testing.assert_array_equal(r[e], exp, err_msg=f"{label}: env {e} ({what}): {name}")


@contextlib.contextmanager
def _mempool_disabled(device=DEV):
    was = wp.is_mempool_enabled(device)
    wp.set_mempool_enabled(device, False)
    try:
        yield
    finally:
        wp.set_mempool_enabled(device, was)


@contextlib.contextmanager
def _capture_launch_forbidden():
    """Make track_gen's ``wp.capture_launch`` raise for the duration of the block."""
    from track_gen.exec import runner

    calls = []

    def _forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError(
            "wp.capture_launch called while recording into the caller's capture "
            "(a graph launch cannot be recorded: CUDA error 900)")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner.wp, "capture_launch", _forbidden)
        if runner.wp is not wp:
            mp.setattr(wp, "capture_launch", _forbidden)
        yield calls
    # Also catches an implementation that swallowed the AssertionError.
    assert not calls, "wp.capture_launch was called during the caller's recording"


@contextlib.contextmanager
def _recording(stream):
    """The caller's (Warp-registered) capture on ``stream``: mempool disabled,
    capture_launch forbidden. Context managers exit in reverse order: the patch is
    lifted before ``capture_end`` and the mempool is restored after it."""
    with _mempool_disabled(), wp.ScopedCapture(device=DEV, stream=stream) as cap, \
            _capture_launch_forbidden():
        yield cap


def _side_stream() -> wp.Stream:
    s = wp.Stream(DEV)
    assert s.cuda_stream != wp.get_stream(DEV).cuda_stream, "side stream must not be current"
    return s


def _replay(graph, stream) -> None:
    wp.capture_launch(graph, stream=stream)
    wp.synchronize()


def _mask_array(values) -> wp.array:
    a = wp.array(np.asarray(values, np.int32), dtype=wp.int32, device=DEV)
    wp.synchronize()   # uploaded on Warp's current stream; side-stream readers need it done
    return a


def _write(dst: wp.array, values) -> None:
    """Write a host mask into a device buffer, outside any capture (read at replay time)."""
    wp.copy(dst, _mask_array(values))
    wp.synchronize()


def _marker():
    """A tiny copy the caller records itself, so its graph is never empty."""
    src = wp.array(np.array([1, 2, 3], np.int32), dtype=wp.int32, device=DEV)
    dst = wp.zeros(3, dtype=wp.int32, device=DEV)
    wp.synchronize()
    return dst, src


def _inputs_mask(producer) -> wp.array:
    """``producer.inputs["regen_mask"]`` right after construction: a persistent, zeroed
    ``[E]`` int32 buffer on the producer's device with a stable pointer."""
    m = producer.inputs["regen_mask"]
    assert isinstance(m, wp.array), type(m)
    assert m.dtype is wp.int32 and tuple(m.shape) == (E,), (m.dtype, m.shape)
    assert str(m.device) == str(wp.get_device(DEV)), m.device
    assert producer.inputs["regen_mask"].ptr == m.ptr, "inputs['regen_mask'] must be persistent"
    np.testing.assert_array_equal(m.numpy(), _ZERO, err_msg="inputs['regen_mask'] starts zeroed")
    return m


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch built without CUDA")
    return torch


# ---------------------------------------------------------------------------------------
# Generator facades
# ---------------------------------------------------------------------------------------

def _make(kind: str, seeds=11):
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device=DEV)
    if kind == "track":
        return TrackGenerator(TrackGenConfig(device=DEV, num_envs=E), rng), rng
    return GateGenerator(GateGenConfig(device=DEV, num_envs=E, gate_width=0.05), rng), rng


def _result(gen):
    return gen._track if isinstance(gen, TrackGenerator) else gen._gate_sequence


def _facade_mask(gen, mask_src: str) -> wp.array:
    if mask_src == "inputs":
        return _inputs_mask(gen)
    a = wp.zeros(E, dtype=wp.int32, device=DEV)
    wp.synchronize()   # zeroed on Warp's current stream; side-stream readers need it done
    return a


def _eager_twin_after(kind: str, mask_src: str, masks):
    """A fresh twin advanced eagerly through ``masks`` (one masked advance each)."""
    twin, _ = _make(kind, 11)
    m = _facade_mask(twin, mask_src)
    out = _result(twin)
    for mk in masks:
        _write(m, mk)
        out = twin.enqueue(advance=True, mask=m)
    wp.synchronize()
    return out


def _check_or_xfail(check, ref, rebuild, label: str) -> None:
    """Run ``check``. If it fails, xfail ONLY when the eager reference itself is shown not
    to be bitwise reproducible (``rebuild()`` returns a fresh reference that differs from
    ``ref``); otherwise the failure stands. Never weakens the comparison silently."""
    try:
        check()
    except AssertionError:
        for _ in range(3):
            if _differs(rebuild(), ref):
                pytest.xfail(f"{label}: {_NONDETERMINISTIC}")
        raise


@pytest.mark.parametrize("mask_src", ["inputs", "user"])
@pytest.mark.parametrize("kind", _KINDS)
def test_masked_advance_in_caller_graph_gives_a_new_batch_per_replay(kind, mask_src):
    """``generate(advance=True, mask=m)`` recorded into the caller's graph: the advance is
    part of the recording (nothing runs at record time), so 5 replays give 5 distinct new
    batches for the masked envs (epochs 1..5) while the unmasked envs stay bitwise
    unchanged; the mask is read at replay time (an all-zero mask changes nothing, its
    complement advances the other envs). Every replay starts from a fully poisoned
    output: the masked envs equal an eager twin advanced identically, the unmasked envs
    keep the poison byte for byte (track: the commit) or equal the twin (gate:
    recomputed), and ``generation`` equals the epoch."""
    gen, _ = _make(kind, 11)
    twin, _ = _make(kind, 11)
    m = _facade_mask(gen, mask_src)
    m_t = _facade_mask(twin, mask_src)
    out = _result(gen)
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), _ZERO, err_msg="fresh epoch")
    np.testing.assert_array_equal(out.generation.numpy(), _ZERO,
                                  err_msg="construction generates epoch 0")
    _write(m, _M1)
    _write(m_t, _M1)
    before = _host_all(_arrays(out))

    s = _side_stream()
    with _recording(s) as cap:
        assert gen.generate(stream=s, advance=True, mask=m) is out
    assert gen.graph is None, "recording into the caller's graph must not self-capture"
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), _ZERO,
                                  err_msg="the recording executed the epoch advance")
    _assert_state(_arrays(out), before, f"{kind}: the recording executed the pipeline")

    schedule = [_M1] * 5 + [_ZERO, 1 - _M1]
    epoch = np.zeros(E, np.int32)
    prev = _rows(_arrays(out))
    seen = {e: [prev[_GEOMETRY[kind]][e]] for e in range(E)}   # env e's batch per epoch
    for k, mk in enumerate(schedule):
        label = f"{kind}/{mask_src}: replay {k} mask {mk.tolist()}"
        _write(m, mk)
        _write(m_t, mk)
        _poison_all(out)
        poisoned = _rows(_arrays(out))
        _replay(cap.graph, s)
        ref = twin.enqueue(advance=True, mask=m_t)
        wp.synchronize()
        epoch += mk
        cur = _rows(_arrays(out))

        def check(cur=cur, prev=prev, poisoned=poisoned, mk=mk, ref=ref, label=label):
            _assert_masked_result(kind, out, ref, poisoned, mk, label)
            if _COMMITS[kind]:
                return
            for e in np.flatnonzero(mk == 0):
                for name, r in cur.items():
                    np.testing.assert_array_equal(
                        r[e], prev[name][e], err_msg=f"{label}: unmasked env {e} changed {name}")

        _check_or_xfail(check, ref,
                        lambda k=k: _eager_twin_after(kind, mask_src, schedule[:k + 1]), label)
        np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch, err_msg=label)
        np.testing.assert_array_equal(twin.seeds.epoch.numpy(), epoch, err_msg=label)
        np.testing.assert_array_equal(out.generation.numpy(), epoch,
                                      err_msg=f"{label}: generation must equal the epoch")
        for e in np.flatnonzero(mk):
            row = cur[_GEOMETRY[kind]][e]
            for j, old in enumerate(seen[e]):
                assert not np.array_equal(row, old, equal_nan=True), \
                    f"{label}: env {e} at epoch {epoch[e]} repeats its epoch-{j} batch"
            seen[e].append(row)
        prev = cur
    for e in np.flatnonzero(_M1):
        assert len(seen[e]) == 6, f"env {e}: expected epoch 0 + 5 distinct new batches"


@pytest.mark.parametrize("kind", _KINDS)
def test_self_graphs_are_cached_per_advance_and_mask(kind):
    """Outside any capture ``generate()`` keeps one self-graph PER ``(advance, mask)``
    variant: :attr:`graph` is the plain (no-advance) one and never advances, the masked
    one advances the masked envs on every replay (on any stream), ``advance=True`` without
    a mask advances every env, and every call equals an eager twin driven the same way
    (a masked call: the masked envs do, the unmasked ones as :func:`_assert_masked_result`
    says, from a fully poisoned output)."""
    gen, _ = _make(kind, 11)
    twin, _ = _make(kind, 11)
    out = _result(gen)
    m, m_t = _mask_array(_M1), _mask_array(_M1)
    s = _side_stream()
    ones = np.ones(E, np.int32)
    calls = [
        ("plain", {}, {}, _ZERO),
        ("masked", {"advance": True, "mask": m}, {"advance": True, "mask": m_t}, _M1),
        ("masked again", {"advance": True, "mask": m}, {"advance": True, "mask": m_t}, _M1),
        ("plain after masked", {}, {}, _ZERO),
        ("all envs", {"advance": True}, {"advance": True}, ones),
        ("masked, side stream, sync=False",
         {"advance": True, "mask": m, "stream": s, "sync": False},
         {"advance": True, "mask": m_t}, _M1),
        ("plain, side stream", {"stream": s}, {}, _ZERO),
    ]
    epoch = np.zeros(E, np.int32)
    g0 = None
    for label, kw, twin_kw, inc in calls:
        label = f"{kind}: {label}"
        _poison_all(out)
        poisoned = _rows(_arrays(out))
        assert gen.generate(**kw) is out
        wp.synchronize()
        ref = twin.enqueue(**twin_kw)
        wp.synchronize()
        epoch += inc
        if "mask" in kw:
            _assert_masked_result(kind, out, ref, poisoned, inc, label)
        else:
            _assert_same(out, ref, label)
        np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch, err_msg=label)
        np.testing.assert_array_equal(out.generation.numpy(), epoch, err_msg=label)
        if g0 is None:
            g0 = gen.graph
            assert g0 is not None, "the first plain CUDA generate() captures the self-graph"
        assert gen.graph is g0, f"{label}: graph must stay the plain (default-key) self-graph"
        masked_key = (True, int(m.ptr))
        if label.endswith(": masked"):
            g_masked = gen._runner.graph_for(masked_key)
            assert g_masked is not None, f"{label}: the masked call captured no self-graph"
        elif "masked" in label:
            assert gen._runner.graph_for(masked_key) is g_masked, \
                f"{label}: the masked self-graph was captured again"
    assert set(gen._runner.graph_keys) == {None, (True, int(m.ptr)), (True, 0)}, \
        gen._runner.graph_keys
    assert twin.graph is None, "enqueue() must never self-graph"


@pytest.mark.parametrize("producer", ["track", "gate", "course"])
def test_invalid_mask_is_refused_before_any_launch_in_caller_capture(producer):
    """A mask that is not ``[E]`` int32 on the producer's device raises ValueError
    host-side, before any launch: the caller's capture stays valid and its replay runs
    only the caller's own work (outputs and epochs untouched)."""
    if producer == "course":
        obj, _ = _bound_course("track-segments")
        out, seeds, call = obj.result, obj.generator.seeds, obj.regenerate
    else:
        obj, _ = _make(producer, 11)
        out, seeds, call = _result(obj), obj.seeds, obj.generate
    bad = {
        "float32": wp.zeros(E, dtype=wp.float32, device=DEV),
        "wrong shape": wp.zeros(E + 1, dtype=wp.int32, device=DEV),
        "2-D": wp.zeros((E, 1), dtype=wp.int32, device=DEV),
        "host device": wp.zeros(E, dtype=wp.int32, device="cpu"),
    }
    s = _side_stream()
    marker, marker_src = _marker()
    with _recording(s) as cap:
        wp.copy(marker, marker_src, stream=s)
        for name, mask in bad.items():
            try:
                call(stream=s, advance=True, mask=mask)
            except ValueError as exc:
                assert "mask" in str(exc), f"{producer}/{name}: {exc}"
            else:
                pytest.fail(f"{producer}: a {name} mask was accepted")
    _poison_all(out)
    poisoned = _host_all(_arrays(out))
    _replay(cap.graph, s)
    np.testing.assert_array_equal(marker.numpy(), marker_src.numpy())
    _assert_state(_arrays(out), poisoned, f"{producer}: outputs written by the caller's graph")
    np.testing.assert_array_equal(seeds.epoch.numpy(), _ZERO, err_msg="epoch advanced")


# ---------------------------------------------------------------------------------------
# Course: masked regeneration, conditional skip
# ---------------------------------------------------------------------------------------

def _course_cfg(variant: str, seeds) -> CourseConfig:
    if variant == "track-segments":
        return CourseConfig(mode="track", gen=TrackGenConfig(num_envs=E, device=DEV),
                            seeds=seeds, collision="segments", checkpoint_spacing=0.6,
                            max_checkpoints=64)
    if variant == "track-sdf-heightfield":
        return CourseConfig(mode="track", gen=TrackGenConfig(num_envs=E, device=DEV),
                            seeds=seeds, collision="sdf", sdf_resolution=32,
                            heightfield_resolution=16, checkpoint_spacing=0.6,
                            max_checkpoints=64)
    if variant in ("gates-posts", "gates-posts-checkpoint"):
        # scale=3.0 keeps 0.2-wide gates valid, so the CourseLine is non-degenerate. The
        # "checkpoint" gate generator (RL loop) gives sequences an agent following the
        # CourseLine laps in a few dozen steps (bezier ones often fold back at a gate).
        kw = {"generator": "checkpoint"} if variant == "gates-posts-checkpoint" else {}
        return CourseConfig(mode="gates",
                            gen=GateGenConfig(num_envs=E, device=DEV, gate_width=0.2,
                                              scale=3.0, **kw),
                            seeds=seeds, post_radius=0.02)
    assert variant == "gates-frames", variant
    gcfg = GateGenConfig(num_envs=E, device=DEV, gate_width=0.2, scale=3.0,
                         z_profile="uniform", z_min=0.5, z_max=1.5)
    return CourseConfig(mode="gates", gen=gcfg, seeds=seeds, frame_collision=True,
                        agent_radius=0.05, frame_thickness=0.05, frame_depth=0.05)


def _bound_course(variant: str, seeds=5):
    course = Course(_course_cfg(variant, seeds))
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    if variant == "gates-frames":
        course.bind(pos)
    else:
        orient = wp.array(np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (E, 1)),
                          dtype=wp.quatf, device=DEV)
        he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f, device=DEV)
        course.bind(position=pos, orientation=orient, half_extents=he)
    wp.synchronize()
    return course, pos


# CourseLine fields its refresh() rewrites (outer/inner/normal/winding are constant).
_LINE_FIELDS = ("center", "tangent", "arclen", "length", "valid", "count")


def _regenerated(course) -> dict:
    """Every buffer a regeneration produces, for every env (the generator's result, the
    refreshed sub-tools); see :func:`_rewritten` for what a masked body rewrites."""
    out = {}
    objs = [("result", course.result), ("checkpoints", course.checkpoints)]
    if course.heightfield is not None:
        objs.append(("heightfield", course.heightfield._hf))
    for label, obj in objs:
        for k, a in _arrays(obj).items():
            if k not in _PER_ENV_FIELDS:
                out[f"{label}.{k}"] = a
    if course._posts is not None:
        out["posts"] = course._posts
    if course.course_line is not None:
        for k in _LINE_FIELDS:
            out[f"line.{k}"] = getattr(course.course_line.track, k)
    return out


def _rewritten(course) -> dict:
    """The buffers a zero-mask regeneration body rewrites for EVERY env (poisoned before
    a zero-mask replay): :func:`_regenerated`, minus the ``Track`` of a track course. Its
    generator commits (:data:`_COMMITS`): every unmasked env's ``Track`` rows end the
    body exactly as they began, so a poisoned ``Track`` would stay poisoned, feed the
    refresh, and stay so for every env a later masked replay does not select."""
    regen = _regenerated(course)
    if _COMMITS["track" if isinstance(course.generator, TrackGenerator) else "gate"]:
        regen = {k: a for k, a in regen.items() if not k.startswith("result.")}
    return regen


def _runtime(course) -> dict:
    """Per-env runtime state: a masked regeneration resets it for the masked envs only."""
    p = course.progress
    out = {f"progress.{k}": getattr(p, k) for k in ("_prev_pos", "_next", "_laps", "_progress")}
    if course.localizer is not None:
        out["localizer._last"] = course.localizer._last
    return out


def _epoch(course) -> wp.array:
    return course.generator.seeds.epoch


def _course_state(course, step_result=None) -> dict:
    out = {**_regenerated(course), **_runtime(course), "epoch": _epoch(course),
           # per-env: unchanged for unmasked envs, the new epoch for masked ones
           "result.generation": course.result.generation}
    if step_result is not None:
        for label, obj in (("events", step_result.events), ("contacts", step_result.contacts),
                           ("frame", step_result.frame)):
            for k, a in _arrays(obj).items():
                out[f"{label}.{k}"] = a
    return out


def _assert_only_masked_envs_changed(course, before: dict, mk: np.ndarray, label: str) -> None:
    """Per env: masked envs got a new batch and a reset runtime state; every other env's
    result and runtime state are bitwise what they were in ``before`` (a host snapshot of
    :func:`_course_state`)."""
    geom = "center" if hasattr(course.result, "center") else "position"
    res_now = _rows(_arrays(course.result))
    rt_now = _rows(_runtime(course))
    for e in range(E):
        if mk[e]:
            old = before[f"result.{geom}"].reshape(E, -1)[e]
            assert not np.array_equal(res_now[geom][e], old, equal_nan=True), \
                f"{label}: masked env {e} was not regenerated"
            assert np.isnan(rt_now["progress._prev_pos"][e]).all(), \
                f"{label}: masked env {e}: progress not reset (prev_pos sentinel)"
            for k in ("_next", "_laps", "_progress"):
                assert (rt_now[f"progress.{k}"][e] == 0).all(), f"{label}: env {e} {k}"
            continue
        for k, r in res_now.items():
            np.testing.assert_array_equal(r[e], before[f"result.{k}"].reshape(E, -1)[e],
                                          err_msg=f"{label}: unmasked env {e}: result.{k}")
        for k, r in rt_now.items():
            np.testing.assert_array_equal(r[e], before[k].reshape(E, -1)[e],
                                          err_msg=f"{label}: unmasked env {e}: {k}")


def _force_unconditional(monkeypatch) -> None:
    """Make LaunchContext.skip_unless take its unconditional lowering (as on a pre-12.4
    driver), whatever this machine supports."""
    monkeypatch.setattr(tg_compat, "conditional_ok", lambda device: False)


def _prime(course, twin, stream) -> None:
    """One step at the (zero) bound positions on both: prev_pos becomes finite, so a masked
    progress reset (NaN sentinel) is observable."""
    course.step(stream=stream)
    twin.step()
    wp.synchronize()


@pytest.mark.parametrize("lowering", ["conditional", "unconditional"])
@pytest.mark.parametrize("variant", _COURSE_VARIANTS)
def test_course_masked_regenerate_zero_mask_in_warp_capture(variant, lowering, monkeypatch):
    """``course.regenerate(mask=m, advance=True)`` recorded inside a Warp ScopedCapture on a
    side stream (a registered capture).

    Zero mask (the :func:`_rewritten` buffers poisoned): with conditional nodes the body
    is skipped on device, so poisoned outputs stay poisoned; with the unconditional
    lowering the body runs and is an exact no-op (poison overwritten by bitwise-identical
    values; a track course's ``Track`` is held and restored by the commit). Either way
    the runtime state and the epochs are untouched. Then ``m[0] = 1``: env 0 gets a new
    batch (epoch 1) and a reset runtime state, every other env is bitwise what it was
    before, and everything equals an eager twin regenerated with the same mask."""
    if lowering == "conditional" and not wp.is_conditional_graph_supported():
        pytest.skip(_NO_CONDITIONAL)
    if lowering == "unconditional":
        _force_unconditional(monkeypatch)
    course, _ = _bound_course(variant)
    twin, _ = _bound_course(variant)
    m = _inputs_mask(course)
    m_t = _inputs_mask(twin)
    s = _side_stream()
    _prime(course, twin, s)
    # Warm the masked body outside any capture (zero mask: must be an exact no-op).
    assert course.regenerate(mask=m, advance=True, stream=s) is course.result
    wp.synchronize()
    _assert_state(_course_state(course), _course_state(twin),
                  f"{variant}: zero-mask regenerate outside a capture is not a no-op")
    before = _host_all(_course_state(course))

    with _recording(s) as cap:
        assert course.regenerate(mask=m, advance=True, stream=s) is course.result
    _assert_state(_course_state(course), before, f"{variant}: the recording executed work")

    # -- zero mask ---------------------------------------------------------------------
    regen = _regenerated(course)
    for a in _rewritten(course).values():
        _poison(a)
    wp.synchronize()
    poisoned = _host_all(regen)
    _replay(cap.graph, s)
    if lowering == "conditional":
        _assert_state(regen, poisoned,
                      f"{variant}: zero mask must skip the body on device (poison stays)")
    else:
        _assert_state(regen, {k: before[k] for k in regen},
                      f"{variant}: zero mask, unconditional body must be an exact no-op")
    _assert_state(_runtime(course), {k: before[k] for k in _runtime(course)},
                  f"{variant}/{lowering}: zero mask touched the runtime state")
    np.testing.assert_array_equal(_epoch(course).numpy(), _ZERO,
                                  err_msg=f"{variant}/{lowering}: zero mask advanced an epoch")

    # -- mask[0] = 1 -------------------------------------------------------------------
    mk = _ZERO.copy()
    mk[0] = 1
    _write(m, mk)
    _write(m_t, mk)
    _replay(cap.graph, s)
    assert twin.enqueue_generate(mask=m_t, advance=True) is twin.result   # eager
    wp.synchronize()
    assert twin._runner.graph_keys == (), "the eager twin must never self-graph"
    label = f"{variant}/{lowering}: mask {mk.tolist()}"
    np.testing.assert_array_equal(_epoch(course).numpy(), mk, err_msg=label)
    np.testing.assert_array_equal(course.result.generation.numpy(), mk, err_msg=label)
    _assert_only_masked_envs_changed(course, before, mk, label)
    _assert_state(_course_state(course), _course_state(twin), f"{label}: vs eager twin")


@pytest.mark.parametrize("registered", [False, True], ids=["unregistered", "registered"])
@pytest.mark.parametrize("variant", ["track-segments", "gates-posts"])
def test_course_masked_regenerate_under_torch_cuda_graph(variant, registered):
    """Inside ``torch.cuda.graph(g)`` (no ``stream=``: torch's capture stream is adopted):

    - WITHOUT registration Warp does not know the capture, so the body is recorded
      unconditionally: a zero-mask replay overwrites the poisoned :func:`_rewritten`
      outputs with exactly the no-op values;
    - WITH ``track_gen.interop.torch.registered(conditional=True)`` the capture is
      registered, conditional nodes are allowed, and the zero-mask replay skips the body
      on device (poison stays). The regeneration is the last op of the torch graph, as
      ``registered`` documents for ``conditional=True``.

    Then ``m[0] = 1`` regenerates env 0 only, as an eager twin does; ``retain()`` keeps
    owners alive with the torch graph."""
    torch = _torch_cuda()
    if registered and not wp.is_conditional_graph_supported():
        pytest.skip(_NO_CONDITIONAL)
    from track_gen.interop import torch as tg_torch

    course, _ = _bound_course(variant)
    twin, _ = _bound_course(variant)
    m = wp.zeros(E, dtype=wp.int32, device=DEV)
    m_t = wp.zeros(E, dtype=wp.int32, device=DEV)
    _prime(course, twin, None)
    course.regenerate(mask=m, advance=True)   # warm-up outside the capture (no-op)
    wp.synchronize()
    _assert_state(_course_state(course), _course_state(twin), f"{variant}: warm-up no-op")
    before = _host_all(_course_state(course))
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with _mempool_disabled():
        with torch.cuda.graph(g):
            with _capture_launch_forbidden():
                scope = (tg_torch.registered(conditional=True) if registered
                         else contextlib.nullcontext())
                with scope as ws:
                    if registered:
                        assert isinstance(ws, wp.Stream) and ws.is_capturing
                    res = course.regenerate(mask=m, advance=True)   # adopts torch's stream
    torch.cuda.synchronize()
    assert res is course.result
    _assert_state(_course_state(course), before, f"{variant}: the recording executed work")

    class _Owner:
        pass

    sentinel = _Owner()
    alive = weakref.ref(sentinel)
    tg_torch.retain(g, course, sentinel)
    del sentinel
    gc.collect()
    assert alive() is not None, "retain() must keep owners alive with the torch graph"

    def replay():
        g.replay()
        torch.cuda.synchronize()

    regen = _regenerated(course)
    for a in _rewritten(course).values():
        _poison(a)
    wp.synchronize()
    poisoned = _host_all(regen)
    replay()
    if registered:
        _assert_state(regen, poisoned,
                      f"{variant}: registered torch capture must skip the body (poison stays)")
    else:
        _assert_state(regen, {k: before[k] for k in regen},
                      f"{variant}: unregistered torch capture: unconditional exact no-op")
    _assert_state(_runtime(course), {k: before[k] for k in _runtime(course)},
                  f"{variant}: zero mask touched the runtime state")
    np.testing.assert_array_equal(_epoch(course).numpy(), _ZERO)

    mk = _ZERO.copy()
    mk[0] = 1
    _write(m, mk)
    _write(m_t, mk)
    replay()
    twin.enqueue_generate(mask=m_t, advance=True)                          # eager
    wp.synchronize()
    label = f"{variant}/{'registered' if registered else 'unregistered'}: mask {mk.tolist()}"
    np.testing.assert_array_equal(_epoch(course).numpy(), mk, err_msg=label)
    _assert_only_masked_envs_changed(course, before, mk, label)
    _assert_state(_course_state(course), _course_state(twin), f"{label}: vs eager twin")


def test_torch_allocations_after_regenerate_under_default_registration():
    """``registered()`` without ``conditional=True`` only registers torch's capture (for
    anchoring): the masked regeneration is recorded unconditionally, so torch's capture
    keeps its capture id and torch ops recorded AFTER it, inside and after the block,
    still allocate from the torch graph's private pool. Checked by replaying after eager
    churn: a temporary wrongly served from the general pool (and freed at capture end)
    would be handed to the eager allocations, which a replay then overwrites."""
    torch = _torch_cuda()
    from track_gen.interop import torch as tg_torch

    course, _ = _bound_course("track-segments")
    m = wp.zeros(E, dtype=wp.int32, device=DEV)
    course.regenerate(mask=m, advance=True)   # warm-up outside the capture (no-op)
    wp.synchronize()
    x = torch.arange(4096, device=DEV, dtype=torch.float32)
    out = torch.zeros_like(x)
    out.copy_(x * 2.0 + 1.0)                  # warm the torch kernels up
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with _mempool_disabled():
        with torch.cuda.graph(g):
            with tg_torch.registered() as ws:
                assert not tg_compat.conditional_allowed(ws), "conditional must be opt-in"
                course.regenerate(mask=m, advance=True)
                out.copy_(x * 2.0 + 1.0)      # temporaries allocated after the regeneration
            y = x * 3.0                        # after the block, still in torch's capture
    tg_torch.retain(g, course)
    torch.cuda.synchronize()

    churn = [torch.full_like(x, -5.0) for _ in range(8)]
    x.add_(1.0)
    out.zero_()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * 2.0 + 1.0)
    torch.testing.assert_close(y, x * 3.0)
    for c in churn:
        assert bool((c == -5.0).all()), "a graph temporary was handed to eager code"
    np.testing.assert_array_equal(_epoch(course).numpy(), _ZERO)


# ---------------------------------------------------------------------------------------
# RL loop: auto-reset inside ONE caller graph
# ---------------------------------------------------------------------------------------

_RL_STEPS = 100


@wp.kernel
def _move_agents_k(
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    speed: wp.array(dtype=wp.int32),
    t: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3f),
):
    # Stub physics: each agent advances `speed[e]` samples along its env's CURRENT
    # centerline (a regenerated env is followed on its new centerline from the next
    # step). It sits at the midpoint of two consecutive samples, never exactly on a
    # gate (gate centers are centerline samples in gates mode).
    e = wp.tid()
    k = t[e] + speed[e]
    t[e] = k
    steps[e] = steps[e] + 1
    n = count[e]
    if n > 0:
        i = ((k % n) + n) % n
        base = e * n_max
        pos[e] = 0.5 * (center[base + i] + center[base + (i + 1) % n])
    else:
        pos[e] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def _done_k(
    laps: wp.array(dtype=wp.int32),
    period: wp.array(dtype=wp.int32),
    speed: wp.array(dtype=wp.int32),
    t: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
):
    # Device-side done rule: a completed lap, or a per-env step budget. A done env's
    # agent restarts two moves before the start of its (new) centerline: its first
    # position primes the reset progress state, its second one crosses checkpoint 0.
    e = wp.tid()
    d = int(0)
    if laps[e] >= 1 or steps[e] >= period[e]:
        d = 1
        t[e] = -2 * speed[e]
        steps[e] = 0
    done[e] = d


class _Agents:
    """The caller's per-env agent state (identical device logic for both twins)."""

    def __init__(self, course, mode: str, pos: wp.array) -> None:
        self.line = course.result if mode == "track" else course.course_line.track
        self.n_max = int(self.line.center.shape[0]) // E
        # Below one checkpoint spacing per step (track: ~9 samples between checkpoints at
        # spacing 0.6; gates: 8 line samples per gate), so no checkpoint is jumped over.
        speed = (2 + np.arange(E) % 5 if mode == "track" else 1 + np.arange(E) % 4)
        speed = speed.astype(np.int32)
        self.speed = wp.array(speed, dtype=wp.int32, device=DEV)
        # Step budgets on both sides of the lap times, so both done rules fire.
        self.period = wp.array((20 + 9 * np.arange(E)).astype(np.int32), dtype=wp.int32,
                               device=DEV)
        self.t = wp.array(-2 * speed, dtype=wp.int32, device=DEV)   # as after a done
        self.steps = wp.zeros(E, dtype=wp.int32, device=DEV)
        self.pos = pos
        wp.synchronize()   # uploads/zeroing ran on Warp's current stream; moves may use another

    def move(self, stream=None) -> None:
        wp.launch(_move_agents_k, dim=E,
                  inputs=[self.line.center, self.line.count, self.n_max, self.speed,
                          self.t, self.steps, self.pos],
                  device=DEV, stream=stream)

    def decide(self, laps: wp.array, done: wp.array, stream=None) -> None:
        wp.launch(_done_k, dim=E,
                  inputs=[laps, self.period, self.speed, self.t, self.steps, done],
                  device=DEV, stream=stream)

    def arrays(self) -> dict:
        return {"agent.t": self.t, "agent.steps": self.steps, "agent.pos": self.pos}


def _rl_state(course, agents: _Agents, done: wp.array, step_result=None) -> dict:
    return {**_course_state(course, step_result), **agents.arrays(), "done": done}


@pytest.mark.parametrize("lowering", ["auto", "unconditional"])
@pytest.mark.parametrize("mode", ["track", "gates"])
def test_rl_auto_reset_loop_in_one_caller_graph(mode, lowering, monkeypatch):
    """ONE caller graph per RL step: move agents (stub kernel) -> ``course.step()`` ->
    device done rule (lap completed or per-env step budget) writing
    ``course.inputs["regen_mask"]`` -> ``course.regenerate(mask=done, advance=True)``.
    100 replays with no host involvement equal an eager twin driven step by step whose
    host only calls ``enqueue_generate(mask=done, advance=True)`` (eager, never a graph)
    when some env is done (so the graph's zero-mask steps must be exact no-ops),
    comparing everything every step:
    agents, step events / contacts / frame, every regenerated buffer, progress and
    localizer state, epochs and generation. ``auto`` uses the conditional skip where the
    device supports it; ``unconditional`` forces the always-run lowering."""
    if lowering == "unconditional":
        _force_unconditional(monkeypatch)
    variant = "track-segments" if mode == "track" else "gates-posts-checkpoint"
    course, pos_c = _bound_course(variant)
    twin, pos_t = _bound_course(variant)
    agents_c = _Agents(course, mode, pos_c)
    agents_t = _Agents(twin, mode, pos_t)
    done_c = _inputs_mask(course)        # the course's own persistent mask buffer
    done_t = wp.zeros(E, dtype=wp.int32, device=DEV)
    s = _side_stream()

    # Warm-up outside the capture: the test kernels (on scratch buffers), step() and the
    # masked regeneration (zero mask: an exact no-op), then a full reset on both.
    scratch = _Agents(course, mode, wp.zeros(E, dtype=wp.vec3f, device=DEV))
    z0, z1 = (wp.zeros(E, dtype=wp.int32, device=DEV) for _ in range(2))
    # Allocations are zeroed on Warp's current stream; Warp streams do not order against
    # each other, so sync before the side stream reads them.
    wp.synchronize()
    scratch.move(stream=s)
    scratch.decide(z0, z1, stream=s)
    sr_c = course.step(stream=s)
    assert course.regenerate(mask=done_c, advance=True, stream=s) is course.result
    sr_t = twin.step()
    full = wp.full(E, 1, dtype=wp.int32, device=DEV)
    wp.synchronize()                     # full is filled on the current stream, read on s
    course.reset(full, stream=s)
    twin.reset(full)
    wp.synchronize()
    _assert_state(_rl_state(course, agents_c, done_c, sr_c),
                  _rl_state(twin, agents_t, done_t, sr_t), f"{mode}: after warm-up")

    with _recording(s) as cap:
        agents_c.move(stream=s)
        assert course.step(stream=s) is sr_c
        agents_c.decide(sr_c.events.laps, done_c, stream=s)
        assert course.regenerate(mask=done_c, advance=True, stream=s) is course.result
    _assert_state(_rl_state(course, agents_c, done_c, sr_c),
                  _rl_state(twin, agents_t, done_t, sr_t), f"{mode}: the recording executed")

    lap_done = budget_done = idle_steps = reset_steps = 0
    for k in range(_RL_STEPS):
        _replay(cap.graph, s)

        agents_t.move()
        assert twin.step() is sr_t
        agents_t.decide(sr_t.events.laps, done_t)
        wp.synchronize()
        d = done_t.numpy()
        laps = sr_t.events.laps.numpy()
        if d.any():
            twin.enqueue_generate(mask=done_t, advance=True)
            wp.synchronize()
            reset_steps += 1
            lap_done += int(((laps >= 1) & (d != 0)).sum())
            budget_done += int(((laps < 1) & (d != 0)).sum())
        else:
            idle_steps += 1
        _assert_state(_rl_state(course, agents_c, done_c, sr_c),
                      _rl_state(twin, agents_t, done_t, sr_t),
                      f"{mode}/{lowering}: step {k} (done {d.tolist()})")
    assert twin._runner.graph_keys == (), "the eager twin must never self-graph"
    epoch = _epoch(course).numpy()
    np.testing.assert_array_equal(course.result.generation.numpy(), epoch)
    # Non-vacuity: both done rules fired, some steps reset envs and some reset none.
    assert lap_done > 0, "no env ever completed a lap (the laps>=1 rule was never exercised)"
    assert budget_done > 0, "the per-env step budget never fired"
    assert reset_steps > 0 and idle_steps > 0, (reset_steps, idle_steps)
    assert epoch.max() >= 2, f"no env was regenerated twice: epochs {epoch.tolist()}"
