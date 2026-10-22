"""CUDA-only: dynamic Stage params and inputs in caller graphs and self-graphs (roadmap PR8).

A Stage generator's DYNAMIC Params field is a live ``[E]`` device array
(``gen.params["generator"]["amp"]``) with a stable pointer. Every generation latches it
into the stage's working copy (``StageBuilder.param``) for the envs it regenerates, so:

- a write to the live array between replays (``StageParams.set`` / ``ParamTable.set``,
  ``.assign``, the caller's own kernel) reaches the NEXT replay of an already-recorded
  graph, with no re-recording;
- ``set(..., stream=s)`` recorded inside the caller's graph is replayed with it: an
  array source is read at replay time, a scalar is baked into the graph;
- a masked generation latches the masked envs only, so the unmasked envs stay bitwise
  unchanged after a param write;
- a ``CaptureClass.HOST`` stage refuses a caller capture before any launch.

The toys are ``tests/_plugin_toys.py``'s ``toy.wobble`` family (public plugin surface
only). The tests wrap them in an instrumented subclass that counts Python body runs
and keeps the stage-side work array, without changing their output.

Method (as in tests/test_caller_capture_cuda.py): record on a NON-current side stream,
poison the outputs with FINITE, dtype-aware sentinels (outputs are NaN-padded and
``assert_array_equal`` treats NaN == NaN, so a NaN poison would hide a missed write),
replay, and compare with an eager twin. A twin is a fresh generator built with
``generator_params={"amp": v}`` and run with ``enqueue()``, which never self-graphs. A
per-env reference stitches env ``e``'s rows from the scalar twin with ``amp = v[e]``, so
the per-env latch is not also the reference. Every recording also runs with

- the Warp mempool disabled, so any allocation inside the capture fails loudly;
- ``wp.capture_launch`` patched to raise: a graph launch cannot be recorded into another
  capture (CUDA error 900), so a producer must re-enqueue its work.

Construction prepares the default body (one real eager pass), which loads the toy, latch
and fill kernels. The masked body and the test's own kernel are warmed up explicitly
before they are recorded, so no module loads while capturing.
"""
from __future__ import annotations

import contextlib
import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

import track_gen.exec as tg_exec  # noqa: E402
from tests._plugin_toys import (  # noqa: E402
    ToyWobble,
    ToyWobbleHost,
    ToyWobbleParams,
    register_stage_toys,
)
from track_gen import PerEnvSeededRNG, TrackGenConfig  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen.plugin import registry  # noqa: E402

DEV = "cuda:0"
E = 8
_DEFAULT = ToyWobbleParams().amp                       # 0.15
# Per-env values: neighbours differ, so a write that leaks across an env boundary shows.
_PER_ENV = np.array([0.05, 0.35] * (E // 2), np.float32)
_ZERO = np.zeros(E, np.int32)
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)


def _full(v) -> np.ndarray:
    return np.full(E, v, np.float32)


@pytest.fixture(autouse=True)
def _toys():
    """Register the Stage toys; undone by the per-test registry scope (tests/conftest.py)."""
    register_stage_toys(registry)


# ---------------------------------------------------------------------------------------
# Capture discipline (as in tests/test_caller_capture_cuda.py)
# ---------------------------------------------------------------------------------------

def _arrays(obj) -> dict:
    """name -> wp.array for every array field of a result dataclass (None fields skipped)."""
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


def _poison_all(obj, *, keep=()) -> None:
    for name, arr in _arrays(obj).items():
        if name not in keep:
            _poison(arr)
    wp.synchronize()


def _snapshot(obj) -> dict:
    return {k: np.array(a.numpy(), copy=True) for k, a in _arrays(obj).items()}


def _rows(host: dict, envs) -> dict:
    return {k: v.reshape(E, -1)[envs] for k, v in host.items()}


def _assert_host(got: dict, ref: dict, label: str) -> None:
    assert ref and got.keys() == ref.keys(), f"{label}: {sorted(set(got) ^ set(ref))}"
    for name in ref:
        np.testing.assert_array_equal(got[name], ref[name], err_msg=f"{label}: {name}")


def _assert_out(out, ref: dict, label: str) -> None:
    _assert_host(_snapshot(out), ref, label)


def _differs(a: dict, b: dict) -> bool:
    return any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a)


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
    """The caller's Warp capture on ``stream``: mempool disabled, capture_launch forbidden."""
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


def _dev(values: np.ndarray) -> wp.array:
    """A float32 device copy of ``values``, complete before any other stream reads it."""
    arr = wp.array(np.asarray(values, np.float32), dtype=wp.float32, device=DEV)
    wp.synchronize()
    return arr


def _write(arr: wp.array, values: np.ndarray) -> None:
    arr.assign(values)
    wp.synchronize()


def _marker():
    """A tiny copy the caller records itself, so its graph is never empty."""
    src = wp.array(np.array([1, 2, 3], np.int32), dtype=wp.int32, device=DEV)
    dst = wp.zeros(3, dtype=wp.int32, device=DEV)
    wp.synchronize()
    return dst, src


@wp.kernel
def _scaled_copy_k(src: wp.array(dtype=wp.float32), k: float,
                   dst: wp.array(dtype=wp.float32)):
    e = wp.tid()
    dst[e] = src[e] * k


def _user_write(live: wp.array, values: np.ndarray, *, stream=None, src=None) -> None:
    """The caller's own kernel writing the live array: ``live = src * 0.5`` with
    ``src = 2 * values`` (exact in float32)."""
    if src is None:
        src = _dev(2.0 * np.asarray(values, np.float32))
    wp.launch(_scaled_copy_k, dim=E, inputs=[src, 0.5], outputs=[live], device=DEV,
              stream=stream)


# ---------------------------------------------------------------------------------------
# Generators, instrumented toys and eager twins
# ---------------------------------------------------------------------------------------

def _make(generator: str = "toy.wobble", seeds: int = 11, *, amp=None):
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device=DEV)
    params = None if amp is None else {"amp": float(amp)}
    cfg = TrackGenConfig(generator=generator, device=DEV, num_envs=E, generator_params=params)
    return TrackGenerator(cfg, rng), rng


class _Probe:
    """What an instrumented toy saw: Python body runs, and the stage-side (work) copy of
    ``amp`` its kernel reads (``StageBuilder.param``)."""

    def __init__(self) -> None:
        self.calls = 0
        self.work: "wp.array | None" = None


def _instrument(base, name: str) -> _Probe:
    """Register ``name``: ``base`` with a counting enqueue; same output as ``base`` (the
    salt depends on the stage id, not on the name)."""
    probe = _Probe()

    def build(self, b, p):
        probe.work = b.param("amp")
        enqueue = base.build(self, b, p)

        def counted(ctx):
            probe.calls += 1
            enqueue(ctx)

        return counted

    registry.add(type(base)(f"Counting{base.__name__}", (base,),
                            {"name": name, "build": build, "__module__": __name__}))
    return probe


def _eager_twin(generator: str = "toy.wobble", *, amp=None):
    """A fresh twin run eagerly (``enqueue`` never self-graphs)."""
    twin, _ = _make(generator, amp=amp)
    out = twin.enqueue()
    wp.synchronize()
    assert twin.graph is None, "enqueue() must never self-graph"
    return twin, out


class _Refs:
    """Host batches of ``toy.wobble`` eager twins, one per distinct ``amp`` (cached)."""

    def __init__(self) -> None:
        self._by_amp: dict = {}

    def uniform(self, amp) -> dict:
        key = float(np.float32(amp))
        if key not in self._by_amp:
            twin, out = _eager_twin(amp=key)
            self._by_amp[key] = _snapshot(out)
        return self._by_amp[key]

    def rows(self, amps) -> dict:
        """The batch where env ``e`` generates with ``amp = amps[e]``: env ``e``'s rows of
        the scalar twin built with that value."""
        twins = [self.uniform(a) for a in np.asarray(amps, np.float32)]
        return {name: np.stack([t[name].reshape(E, -1)[e] for e, t in enumerate(twins)])
                .reshape(twins[0][name].shape) for name in twins[0]}


# ---------------------------------------------------------------------------------------
# Caller graph: recording, and writes between replays
# ---------------------------------------------------------------------------------------

def test_stage_generator_records_into_caller_capture():
    """``toy.wobble`` is capturable. ``generate(stream=s)`` is recorded into the caller's
    capture with no self-graph, no allocation and no capture_launch, running the Python
    body exactly once. Each replay into poisoned outputs equals the eager twin and runs
    no Python; the live and work arrays keep the default."""
    assert registry.describe("generator", "toy.wobble").capturable is True
    probe = _instrument(ToyWobble, "toy.count")
    gen, _ = _make("toy.count")
    assert probe.calls == 1, "construction runs exactly one real pass"
    live = gen.params["generator"]["amp"]
    assert live.dtype is wp.float32 and live.shape == (E,) and str(live.device) == DEV
    ref = _Refs().uniform(_DEFAULT)

    s = _side_stream()
    with _recording(s) as cap:
        out = gen.generate(stream=s)
    assert gen.graph is None, "recording into the caller's graph must not self-capture"
    assert probe.calls == 2, "the recording runs the Python body exactly once"
    for k in range(2):
        _poison_all(out)
        _replay(cap.graph, s)
        _assert_out(out, ref, f"replay {k}")
    assert probe.calls == 2, "caller replays must not run the Python body"
    np.testing.assert_array_equal(live.numpy(), _full(_DEFAULT))
    np.testing.assert_array_equal(probe.work.numpy(), _full(_DEFAULT))


_BETWEEN = ("set-scalar", "set-array", "param-table", "assign", "user-kernel")


def _write_between(gen, how: str, values: np.ndarray) -> None:
    """Write the live ``amp`` array in place, outside any capture."""
    sp = gen.params["generator"]
    if how == "set-scalar":
        assert np.all(values == values[0])
        sp.set("amp", float(values[0]))
    elif how == "set-array":
        sp.set("amp", _dev(values))
    elif how == "param-table":
        gen.params.set("generator", "amp", _dev(values))
    elif how == "assign":
        sp["amp"].assign(values)
    else:
        assert how == "user-kernel", how
        _user_write(sp["amp"], values)
    wp.synchronize()


@pytest.mark.parametrize("how", _BETWEEN)
def test_param_written_between_replays_reaches_the_next_replay(how):
    """ONE recording of ``generate(stream=s)``. Between replays the live ``amp`` array is
    written in place, by ``StageParams.set`` (scalar or ``[E]`` array),
    ``ParamTable.set``, ``.assign`` or the caller's own kernel. Each write lands in the
    live array (stable pointer) and is not latched before the next generation. The
    next replay latches it and equals the eager twin with the same values: a scalar
    twin, or a per-env reference stitched from scalar twins. Nothing is re-recorded:
    the Python body ran once for the recording, and the generator has no self-graph."""
    probe = _instrument(ToyWobble, "toy.count")
    gen, _ = _make("toy.count")
    live = gen.params["generator"]["amp"]
    ptr = live.ptr
    refs = _Refs()
    s = _side_stream()
    with _recording(s) as cap:
        out = gen.generate(stream=s)
    graph = cap.graph
    assert probe.calls == 2

    if how == "set-scalar":
        schedule = (_full(0.3), _full(0.05), _full(_DEFAULT))
    else:
        schedule = (_full(0.3), _PER_ENV, _full(_DEFAULT))
    assert _differs(refs.rows(schedule[0]), refs.uniform(_DEFAULT)), "amp must change the batch"
    assert _differs(refs.rows(schedule[1]), refs.uniform(_DEFAULT))
    latched = _full(_DEFAULT)
    for k, values in enumerate(schedule):
        label = f"{how}: replay after write {k} {values.tolist()}"
        _write_between(gen, how, values)
        np.testing.assert_array_equal(live.numpy(), values, err_msg=f"{label}: live")
        np.testing.assert_array_equal(probe.work.numpy(), latched,
                                      err_msg=f"{label}: latched before a generation")
        _poison_all(out)
        _replay(graph, s)
        _assert_out(out, refs.rows(values), label)
        np.testing.assert_array_equal(probe.work.numpy(), values, err_msg=f"{label}: work")
        latched = values
    assert probe.calls == 2, "writes and replays must not re-record the body"
    assert gen.graph is None
    assert gen.params["generator"]["amp"] is live and live.ptr == ptr, "stable live array"


# ---------------------------------------------------------------------------------------
# Caller graph: the write is part of the recording
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("how", ["stage-params", "param-table", "user-kernel"])
def test_param_write_recorded_in_caller_graph_reads_its_source_at_replay(how):
    """ONE caller graph holds ``[write amp from src] -> generate()``. The write is
    ``gen.params["generator"].set("amp", src, stream=s)``,
    ``gen.params.set("generator", "amp", src, stream=s)``, or the caller's own kernel
    on ``s`` (the in-graph curriculum). Recording executes nothing: the live array
    keeps the default. Before each replay ``src`` is rewritten outside the graph. The
    replay copies it into the live array, latches it, and equals the eager twin with
    those per-env values. The body runs once, for the recording."""
    probe = _instrument(ToyWobble, "toy.count")
    gen, _ = _make("toy.count")
    live = gen.params["generator"]["amp"]
    refs = _Refs()
    src = _dev(_full(_DEFAULT))
    if how == "user-kernel":
        _user_write(wp.zeros(E, dtype=wp.float32, device=DEV), _full(0.0), src=src)  # warm-up
        wp.synchronize()

    s = _side_stream()
    with _recording(s) as cap:
        if how == "stage-params":
            gen.params["generator"].set("amp", src, stream=s)
        elif how == "param-table":
            gen.params.set("generator", "amp", src, stream=s)
        else:
            _user_write(live, None, stream=s, src=src)
        out = gen.generate(stream=s)
    assert gen.graph is None
    assert probe.calls == 2
    np.testing.assert_array_equal(live.numpy(), _full(_DEFAULT),
                                  err_msg="the recording executed the write")

    for values in (_full(0.3), _PER_ENV, _full(0.05)):
        label = f"{how}: src {values.tolist()}"
        _write(src, 2.0 * values if how == "user-kernel" else values)
        _poison_all(out)
        _replay(cap.graph, s)
        np.testing.assert_array_equal(live.numpy(), values, err_msg=f"{label}: live")
        np.testing.assert_array_equal(probe.work.numpy(), values, err_msg=f"{label}: work")
        _assert_out(out, refs.rows(values), label)
    assert probe.calls == 2, "caller replays must not run the Python body"


@pytest.mark.parametrize("via", ["stage-params", "param-table"])
def test_scalar_set_recorded_in_caller_graph_is_baked(via):
    """``set("amp", 0.3, stream=s)`` recorded before ``generate(stream=s)`` bakes the
    scalar into the caller's graph. The recording leaves the live array at its default.
    Every replay refills 0.3 and equals the ``amp=0.3`` twin, even after an eager write
    of another value outside the graph."""
    gen, _ = _make()
    live = gen.params["generator"]["amp"]
    refs = _Refs()
    s = _side_stream()
    with _recording(s) as cap:
        if via == "stage-params":
            gen.params["generator"].set("amp", 0.3, stream=s)
        else:
            gen.params.set("generator", "amp", 0.3, stream=s)
        out = gen.generate(stream=s)
    np.testing.assert_array_equal(live.numpy(), _full(_DEFAULT),
                                  err_msg="the recording executed the fill")
    for k in range(2):
        gen.params["generator"].set("amp", 0.05)        # eager, outside the graph
        wp.synchronize()
        np.testing.assert_array_equal(live.numpy(), _full(0.05))
        _poison_all(out)
        _replay(cap.graph, s)
        np.testing.assert_array_equal(live.numpy(), _full(0.3), err_msg=f"replay {k}: live")
        _assert_out(out, refs.uniform(0.3), f"{via}: replay {k}")


# ---------------------------------------------------------------------------------------
# Self-graph
# ---------------------------------------------------------------------------------------

def test_self_graph_replays_pick_up_param_writes():
    """Outside any capture the first ``generate()`` captures the self-graph once. Later
    ``set()`` calls (default stream, or ``stream=s`` with a side-stream replay) are
    picked up by the next replay of that same graph, with no re-capture. Each result
    equals the eager twin with the written values."""
    probe = _instrument(ToyWobble, "toy.count")
    gen, _ = _make("toy.count")
    refs = _Refs()
    out = gen.generate()
    g = gen.graph
    assert g is not None
    assert probe.calls == 2, "construction + the one self-graph capture"
    _assert_out(out, refs.uniform(_DEFAULT), "first self-graph call")

    s = _side_stream()
    for i, values in enumerate((_full(0.3), _PER_ENV, _full(_DEFAULT))):
        _poison_all(out)
        if i == 0:
            gen.params["generator"].set("amp", 0.3)
            res = gen.generate()
        elif i == 1:
            gen.params["generator"].set("amp", _dev(values), stream=s)
            res = gen.generate(stream=s, sync=False)          # stream-ordered after set
            wp.synchronize()
        else:
            gen.params.set("generator", "amp", _DEFAULT, stream=s)
            res = gen.generate(stream=s)
        assert res is out
        assert gen.graph is g, "the self-graph is captured once, then replayed"
        _assert_out(out, refs.rows(values), f"self-graph replay {i}")
        np.testing.assert_array_equal(probe.work.numpy(), values)
    assert probe.calls == 2, "self-graph replays must not run the Python body"


# ---------------------------------------------------------------------------------------
# Masked regeneration under capture
# ---------------------------------------------------------------------------------------

def test_masked_generation_in_caller_graph_latches_only_masked_envs():
    """``amp`` is written (0.15 -> 0.3) before a recorded
    ``generate(advance=True, mask=m)``. Replays with mask schedule [M1, 0, ~M1]:

    - masked envs latch 0.3, advance their epoch and equal an ``amp=0.3`` twin advanced
      the same way;
    - unmasked envs keep their 0.15 work value and stay bitwise unchanged: every replay
      starts from a poisoned Track (``generation`` aside), and the commit restores the
      rows it held, so an unmasked env still holds the poison byte for byte, and its
      ``generation`` its previous epoch;
    - an all-zero mask is an exact no-op.

    After the complement every env has been regenerated at epoch 1, and the rows each env
    got when it was regenerated equal the twin's final batch."""
    probe = _instrument(ToyWobble, "toy.count")
    gen, _ = _make("toy.count")
    twin, _ = _make(amp=0.3)
    m, m_t = gen.inputs["regen_mask"], twin.inputs["regen_mask"]
    # Warm up the masked body outside the capture (an all-zero mask is an exact no-op).
    out = gen.enqueue(advance=True, mask=m)
    twin.enqueue(advance=True, mask=m_t)
    wp.synchronize()
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), _ZERO)
    base = _snapshot(out)
    _assert_host(base, _Refs().uniform(_DEFAULT), "construction batch")

    gen.params["generator"].set("amp", 0.3)
    wp.synchronize()
    s = _side_stream()
    with _recording(s) as cap:
        assert gen.generate(stream=s, advance=True, mask=m) is out
    assert gen.graph is None
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), _ZERO,
                                  err_msg="the recording executed the epoch advance")
    np.testing.assert_array_equal(probe.work.numpy(), _full(_DEFAULT),
                                  err_msg="the recording executed the latch")
    _assert_out(out, base, "the recording executed the pipeline")

    epoch = np.zeros(E, np.int32)
    work = _full(_DEFAULT)
    # Env e's rows as of the replay that last regenerated it (the construction batch
    # before): what the batch would hold had nothing been poisoned.
    regenerated = {k: v.reshape(E, -1).copy() for k, v in base.items()}
    for k, mk in enumerate((_M1, _ZERO, 1 - _M1)):
        label = f"replay {k} mask {mk.tolist()}"
        _write(m, mk)
        _write(m_t, mk)
        _poison_all(out, keep=("generation",))   # written for the masked envs only
        poisoned = _snapshot(out)
        _replay(cap.graph, s)
        ref_out = twin.enqueue(advance=True, mask=m_t)
        wp.synchronize()
        ref = _snapshot(ref_out)
        epoch += mk
        work = np.where(mk != 0, np.float32(0.3), work).astype(np.float32)
        cur = _snapshot(out)
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)
        _assert_host(_rows(cur, unmasked), _rows(poisoned, unmasked),
                     f"{label}: unmasked envs (held and restored by the commit)")
        _assert_host(_rows(cur, masked), _rows(ref, masked), f"{label}: masked envs vs twin")
        np.testing.assert_array_equal(probe.work.numpy(), work, err_msg=f"{label}: work")
        np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch, err_msg=label)
        np.testing.assert_array_equal(out.generation.numpy(), epoch, err_msg=label)
        for name, v in cur.items():
            regenerated[name][masked] = v.reshape(E, -1)[masked]
    _assert_host({k: v.reshape(-1, *ref[k].shape[1:]) for k, v in regenerated.items()}, ref,
                 "every env regenerated at epoch 1 with amp 0.3")
    assert probe.calls == 3, "warm-up + construction + the one recording"


# ---------------------------------------------------------------------------------------
# Stage inputs, HOST stages, torch adoption
# ---------------------------------------------------------------------------------------

def test_stage_input_written_between_replays_reaches_the_next_replay():
    """A ``StageBuilder.input`` (``toy.wobble_input``'s ``difficulty``) is latched like a
    dynamic param. Writing ``gen.inputs["difficulty"]`` between replays of ONE
    recording changes the next replay, which equals an eager twin whose input holds
    the same per-env values."""
    gen, _ = _make("toy.wobble_input")
    difficulty = gen.inputs["difficulty"]
    np.testing.assert_array_equal(difficulty.numpy(), _full(0.5))
    s = _side_stream()
    with _recording(s) as cap:
        out = gen.generate(stream=s)
    assert gen.graph is None

    seen = []
    for values in (_full(1.0), np.array([1.0, 0.25] * (E // 2), np.float32)):
        twin, _ = _make("toy.wobble_input")
        _write(twin.inputs["difficulty"], values)
        ref_out = twin.enqueue()
        wp.synchronize()
        ref = _snapshot(ref_out)
        _write(difficulty, values)
        _poison_all(out)
        _replay(cap.graph, s)
        _assert_out(out, ref, f"difficulty {values.tolist()}")
        assert gen.inputs["difficulty"] is difficulty
        seen.append(ref)
    assert _differs(seen[0], seen[1]), "difficulty must change the batch"


def test_host_stage_refuses_caller_capture_before_any_launch():
    """A ``CaptureClass.HOST`` stage reports ``capturable=False`` and runs eagerly
    outside a capture. Inside one, ``generate`` and ``enqueue`` (with or without a
    mask) raise NotCapturableError naming it before any launch: the body does not run,
    and the caller's replay leaves the outputs, the latch and the epochs untouched. A
    param write made meanwhile reaches the next eager call."""
    probe = _instrument(ToyWobbleHost, "toy.count_host")
    assert registry.describe("generator", "toy.count_host").capturable is False
    gen, _ = _make("toy.count_host")
    refs = _Refs()
    assert probe.calls == 1
    out = gen.generate()
    wp.synchronize()
    assert probe.calls == 2, "outside a capture the body runs eagerly once per call"
    assert gen.graph is None, "a non-capturable generator never self-graphs"
    _assert_out(out, refs.uniform(_DEFAULT), "eager HOST stage")

    gen.params["generator"].set("amp", 0.3)
    wp.synchronize()
    s = _side_stream()
    marker, marker_src = _marker()
    with _recording(s) as cap:
        wp.copy(marker, marker_src, stream=s)       # the caller's own recorded work
        with pytest.raises(tg_exec.NotCapturableError, match="toy.count_host"):
            gen.generate(stream=s)
        with pytest.raises(tg_exec.NotCapturableError, match="toy.count_host"):
            gen.enqueue(stream=s)
        with pytest.raises(tg_exec.NotCapturableError, match="toy.count_host"):
            gen.generate(stream=s, advance=True, mask=gen.inputs["regen_mask"])
    assert probe.calls == 2, "the HOST body ran under the caller's capture"
    assert gen.graph is None

    _poison_all(out)
    poisoned = _snapshot(out)
    _replay(cap.graph, s)                           # the capture stayed valid
    np.testing.assert_array_equal(marker.numpy(), marker_src.numpy())
    _assert_out(out, poisoned, "outputs written by the caller's graph")
    np.testing.assert_array_equal(probe.work.numpy(), _full(_DEFAULT),
                                  err_msg="a latch was recorded")
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), _ZERO,
                                  err_msg="an epoch advance was recorded")

    _assert_out(gen.generate(), refs.uniform(0.3), "eager HOST stage after the write")
    assert probe.calls == 3


def test_param_set_and_generate_adopted_by_torch_cuda_graph():
    """Inside ``torch.cuda.graph(g)``, ``set("amp", src)`` and ``generate()`` without
    ``stream=`` both adopt torch's capture stream and are recorded into ``g``. Each
    replay copies ``src`` (read at replay time) and equals the twin with its values."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch built without CUDA")
    probe = _instrument(ToyWobble, "toy.count")
    gen, _ = _make("toy.count")
    live = gen.params["generator"]["amp"]
    refs = _Refs()
    src = _dev(_full(_DEFAULT))
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with _mempool_disabled():
        with torch.cuda.graph(g):
            with _capture_launch_forbidden():
                gen.params["generator"].set("amp", src)   # no stream=: adopted
                out = gen.generate()
    torch.cuda.synchronize()
    assert gen.graph is None
    assert probe.calls == 2
    np.testing.assert_array_equal(live.numpy(), _full(_DEFAULT),
                                  err_msg="the recording executed the copy")

    for values in (_full(0.3), _PER_ENV):
        _write(src, values)
        _poison_all(out)
        g.replay()
        torch.cuda.synchronize()
        np.testing.assert_array_equal(live.numpy(), values)
        _assert_out(out, refs.rows(values), f"torch replay {values.tolist()}")
    assert probe.calls == 2
