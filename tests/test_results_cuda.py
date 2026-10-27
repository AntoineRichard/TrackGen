"""CUDA-only: the result-object surface under caller capture and on side streams
(roadmap PR13b, design section 9.4).

What this module proves on ``cuda:0``:

1. Mixed batches (``out=`` a view of ONE caller-allocated result): two generators
   (``bezier`` + ``polar``; track and gate pipelines) write the disjoint halves
   ``big.view(0, 4)`` / ``big.view(4, 8)`` of a shared ``Track`` / ``GateSequence``
   through the caller's own memory (stable pointers into ``big``), their warm-ups write
   their halves, and both ``generate(advance=True)`` calls recorded in ONE caller graph
   (a Warp ``ScopedCapture``, or ``torch.cuda.graph`` registered through
   ``interop.torch.registered``) replay into a fully poisoned ``big`` to exactly the
   rows of standalone eager twins advanced alike, before and after a reseed of the
   caller's seed array.
2. Masked regeneration of one half (``a.generate(advance=True, mask=done[0:4])``
   recorded with the mask produced in-graph): every byte of the other half and every
   byte of ``a``'s unmasked envs keep the poison written before the replay; masked envs
   equal the twin at the same (seed, epoch); also with ``on_invalid="poison"`` and
   envs that fail (truncation).
3. ``copy_from`` records into a caller graph (memcpy nodes, no allocation) and snapshots
   the replay's batch; ``clone()`` / ``alloc()`` / ``load_centerlines`` refuse a
   capture with ``CaptureError`` before any work (the caller's capture stays valid and
   nothing is uploaded).
4. Streams: after ``generate(stream=s, sync=False)`` (``s`` held busy, so the
   generation is still queued), ``clone(stream=s)``, ``clone()`` with the generator's
   constructor ``stream=s`` (the producer default), and ``copy_from`` with the
   producer default or ``stream=s`` all snapshot the NEW batch; a ``clone`` is readable
   on Warp's current stream right away, and a fresh ``alloc`` / ``clone`` result is
   complete on every stream (``copy_from`` into it on ``s`` right after, with the
   current stream held busy, is never overwritten). ``Track.alloc`` inside an ambient
   ``LaunchContext.scope(other)`` fills on Warp's current stream, so a generator built
   afterwards with ``out=`` a view and ``stream=s`` writes its warm-up after the fills
   (both other streams held busy), and the rows outside the view keep the padding.
5. ``validate`` on CUDA: a CPU array in a CUDA ``Track`` / ``GateSequence`` is named
   with both devices; a hand-set ``Layout(E, S, "cuda")`` is accepted against the
   planner's canonical ``"cuda:0"`` by ``out=`` and ``copy_from``.
6. ``randomize.uniform`` in a caller graph: every replay of ``generate(advance=True)``
   redraws every env, in bounds, equal to the eager twin (draws and Track) and to a
   non-randomized twin whose ``amp`` is set to the drawn values (the draw reaches the
   generator in the same generation); an in-graph curriculum (the randomizer's
   ``lo`` / ``hi`` written in the caller's graph) bounds the draws of that very replay;
   a masked replay redraws the masked envs only.
7. ``generator/external``: a caller graph that writes ``program.inputs["centerline.xy"]``
   and ``["centerline.count"]`` (a kernel drawing per-env ellipses from a device replay
   counter, NaN past the count) and then generates equals the eager twin fed the same
   rows by ``load_centerlines`` + ``enqueue``, and follows the analytic ellipse (relax
   off by default); the masked variant regenerates the masked envs only;
   ``load_centerlines`` runs on the generator's stream (constructor ``stream=``, or
   ``stream=``), so an in-flight ``generate(sync=False)`` still builds the OLD rows.
8. A utility (``CollisionChecker``, segments and sdf) bound to the OFFSET half of a
   mixed batch, recorded in the same caller graph, equals the same checker on an owned
   clone of those rows; the tutorial's step graph (both halves regenerated with OFFSET
   slices of one caller mask, collision on the whole shared Track) keeps every
   unmasked row and matches the per-half twins.
9. The conformance kit passes ``check_generator("external")`` on ``cuda:0``
   (``diversity`` skipped: ``seeded=False``; ``output.view`` ran).

Method (as in tests/test_caller_capture_cuda.py and tests/test_channels_cuda.py; the
helpers are :mod:`track_gen.testing.capture`'s): record on a NON-current side stream,
poison the outputs with FINITE, dtype-aware sentinels (outputs are NaN / -1 padded and
``assert_array_equal`` treats NaN == NaN, so a NaN poison would hide a missed write),
replay, and compare with an eager twin: a fresh generator run with ``enqueue()`` (never a
graph). Every recording runs with the Warp mempool disabled (an allocation inside the
capture fails loudly) and ``wp.capture_launch`` replaced by a tripwire (a graph launch
cannot be recorded into another capture). Construction prepares the default body; masked
bodies are warmed up with an all-zero mask (an exact no-op) and the test's own kernels
on scratch buffers, so no module loads while capturing. The stream tests hold a stream
busy with a long single-thread kernel, so an unordered copy would read the poison. The
CPU lane only collects this module (the CPU proofs are tests/test_results.py,
tests/test_randomize.py and tests/test_external.py).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

from tests._plugin_toys import register_stage_toys  # noqa: E402
from track_gen import GateGenerator, TrackGenerator  # noqa: E402
from track_gen.collision import CollisionChecker  # noqa: E402
from track_gen.exec import CaptureError, LaunchContext  # noqa: E402
from track_gen.pipeline import load_centerlines, presets  # noqa: E402
from track_gen.plugin import registry  # noqa: E402
from track_gen.results import GateSequence, Layout, Track  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402
from track_gen.testing import check_generator  # noqa: E402

DEV = "cuda:0"
E = 8
H = E // 2                      # envs per half of a mixed batch
_ZERO = np.zeros(E, np.int32)
_ONES = np.ones(E, np.int32)
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)
_SPIN = 1 << 22                 # iterations of _spin_k: holds a stream busy for a while
_MODES = ("warp", "torch")      # caller capture: Warp ScopedCapture / torch.cuda.graph

#: toy.wobble's dynamic ``amp`` (declared bounds [0, 0.4]) drawn per env.
WOBBLE = "toy.wobble"
RID = "randomize.generator.amp"
LO, HI = 0.05, 0.35
#: generator/external.
EXTERNAL = "external"
_STEP = wp.constant(16)          # in-graph rows: count = N - 16 * ((k + e) % 4)
_TWO_PI = wp.constant(6.283185307179586)
_ELLIPSE_TOL = 2.0e-3            # |q - 1| on the resampled centerline (chord sagitta)


# ---------------------------------------------------------------------------------------
# Test kernels (one Warp module: _load_test_kernels() loads it outside any capture)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _spin_k(n: int, out: wp.array(dtype=wp.float32)):
    # One thread, a dependent chain of n sines: holds a stream busy for a while.
    acc = float(0.0)
    for _i in range(n):
        acc = wp.sin(acc + 1.0)
    out[0] = acc


@wp.kernel
def _bump_k(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def _ellipse_rows_k(counter: wp.array(dtype=wp.int32), n: int,
                    xy: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32)):
    # The caller's in-graph centerline writer: env e of replay k (= counter[0]) is an
    # ellipse of m points (semi-axes from k and e); the slots past m hold NaN, which the
    # ingestion must never read (it pads with the last real point).
    t = wp.tid()
    e = t // n
    i = t - e * n
    k = counter[0]
    m = n - _STEP * ((k + e) % 4)
    ax = 1.0 + 0.05 * float(k % 5) + 0.02 * float(e)
    by = 0.7 + 0.03 * float((k + e) % 3)
    if i < m:
        th = _TWO_PI * float(i) / float(m)
        xy[t] = wp.vec2f(ax * wp.cos(th), by * wp.sin(th))
    else:
        xy[t] = wp.vec2f(wp.nan, wp.nan)
    if i == 0:
        count[e] = m


def _ellipse_params(k: int, e: int, n: int) -> "tuple[int, float, float]":
    """Host replica of :func:`_ellipse_rows_k`'s ``(m, ax, by)`` for replay ``k``."""
    return (n - 16 * ((k + e) % 4), 1.0 + 0.05 * (k % 5) + 0.02 * e,
            0.7 + 0.03 * ((k + e) % 3))


def _load_test_kernels() -> None:
    """Load this module's kernels outside any capture (scratch buffers only)."""
    c = wp.zeros(1, dtype=wp.int32, device=DEV)
    xy = wp.zeros(2 * 8, dtype=wp.vec2f, device=DEV)
    cnt = wp.zeros(2, dtype=wp.int32, device=DEV)
    busy = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.launch(_spin_k, 1, inputs=[16, busy], device=DEV)
    wp.launch(_bump_k, 1, inputs=[c], device=DEV)
    wp.launch(_ellipse_rows_k, 16, inputs=[c, 8, xy, cnt], device=DEV)
    wp.synchronize()


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _snapshot(obj) -> dict:
    return {k: np.array(a.numpy(), copy=True)
            for k, a in tg_capture.result_arrays(obj).items()}


def _rows(host: dict, envs, n_envs: int = E) -> dict:
    return {k: v.reshape(n_envs, -1)[envs] for k, v in host.items()}


def _assert_host(got: dict, ref: dict, label: str) -> None:
    assert ref and got.keys() == ref.keys(), f"{label}: {sorted(set(got) ^ set(ref))}"
    for name in ref:
        np.testing.assert_array_equal(got[name], ref[name], err_msg=f"{label}: {name}")


def _differs(a: dict, b: dict) -> bool:
    return any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def _side_stream() -> wp.Stream:
    return tg_capture.side_stream(DEV)


def _recording(stream):
    """The caller's Warp capture on ``stream``: mempool disabled, capture_launch
    forbidden."""
    return tg_capture.recording(DEV, stream)


def _replay(graph, stream) -> None:
    wp.capture_launch(graph, stream=stream)
    wp.synchronize()


def _dev(values, dtype=wp.float32) -> wp.array:
    """A device copy of ``values``, complete before any other stream reads it."""
    np_dtype = np.float32 if dtype is wp.float32 else np.int32
    arr = wp.array(np.asarray(values, np_dtype), dtype=dtype, device=DEV)
    wp.synchronize()
    return arr


def _write(dst: wp.array, values) -> None:
    """Write host values into a device buffer, outside any capture (read at replay)."""
    dst.assign(np.asarray(values, dst.numpy().dtype))
    wp.synchronize()


def _per_env(values) -> np.ndarray:
    v = np.asarray(values, np.float32)
    return np.resize(v, E).astype(np.float32) if v.ndim else np.full(E, v, np.float32)


def _marker():
    """A tiny copy the caller records itself, so its graph is never empty."""
    src = wp.array(np.array([1, 2, 3], np.int32), dtype=wp.int32, device=DEV)
    dst = wp.zeros(3, dtype=wp.int32, device=DEV)
    wp.synchronize()
    return dst, src


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch built without CUDA")
    return torch


def _caller_graph(mode: str, body):
    """Record ``body(stream)`` into ONE caller graph and return its ``replay()``.

    ``"warp"``: a Warp ``ScopedCapture`` on a non-current side stream. ``"torch"``: a
    ``torch.cuda.graph`` capture registered with Warp through
    ``track_gen.interop.torch.registered`` (``body`` gets the registered stream).
    Both with the mempool disabled and ``wp.capture_launch`` forbidden."""
    if mode == "warp":
        s = _side_stream()
        wp.synchronize()
        with _recording(s) as cap:
            body(s)

        def replay(_g=cap.graph, _s=s):
            _replay(_g, _s)
        return replay
    assert mode == "torch", mode
    torch = _torch_cuda()
    from track_gen.interop.torch import registered  # noqa: PLC0415

    ordinal = wp.get_device(DEV).ordinal
    with torch.cuda.device(ordinal):
        g = torch.cuda.CUDAGraph()
        wp.synchronize()
        torch.cuda.synchronize(ordinal)
        with tg_capture.mempool_disabled(DEV):
            with torch.cuda.graph(g):
                with registered(graph=g) as ws, tg_capture.capture_launch_forbidden():
                    body(ws)
        torch.cuda.synchronize(ordinal)

    def replay(_g=g):
        with torch.cuda.device(ordinal):
            _g.replay()
            torch.cuda.synchronize(ordinal)
        wp.synchronize()
    return replay


def _spec(kind: str, generator: str, num_envs: int = E, **kw):
    if kind == "track":
        return presets.track(num_envs=num_envs, device=DEV, generator=generator, **kw)
    return presets.gates(num_envs=num_envs, device=DEV, generator=generator, **kw)


def _stride(spec) -> int:
    return int(spec.batch.n_max) if spec.batch.max_gates is None \
        else int(spec.batch.max_gates)


def _gen_cls(spec):
    return TrackGenerator if spec.batch.max_gates is None else GateGenerator


def _result_cls(spec):
    return Track if spec.batch.max_gates is None else GateSequence


def _eager(spec, seeds=11) -> dict:
    """A fresh twin of ``spec`` run eagerly (``enqueue`` never self-graphs)."""
    twin = _gen_cls(spec)(spec, seeds=seeds)
    out = twin.enqueue()
    wp.synchronize()
    assert twin.graph is None, "enqueue() must never self-graph"
    return _snapshot(out)


def _prepare_masked(gen, mask: wp.array) -> None:
    """Warm up ``gen``'s masked body with an all-zero ``mask``: an exact no-op (nothing
    committed, no epoch advanced)."""
    out = gen.program.result
    base = _snapshot(out)
    epoch = gen.seeds.epoch.numpy().copy()
    np.testing.assert_array_equal(mask.numpy(), 0, err_msg="the warm-up mask is zero")
    assert gen.enqueue(advance=True, mask=mask) is out
    wp.synchronize()
    _assert_host(_snapshot(out), base, "an all-zero mask must change nothing")
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch)


def _assert_valid_is_fail_mask_zero(host: dict, label: str) -> None:
    np.testing.assert_array_equal(host["valid"], (host["fail_mask"] == 0).astype(np.int32),
                                  err_msg=f"{label}: valid != (fail_mask == 0)")


# ---------------------------------------------------------------------------------------
# Mixed batches: two generators over the halves of one caller-allocated result
# ---------------------------------------------------------------------------------------

class _Mixed:
    """``big = alloc(Layout(E, S, DEV))``; generator ``i`` writes ``big.view(i*H,
    (i+1)*H)`` with ``seeds=`` the matching slice of ONE caller seed array (bound by
    pointer); twin ``i`` is a standalone generator of the same spec whose own seed array
    holds the same values."""

    def __init__(self, kind: str, generators=("bezier", "polar"), *, base: int = 1000,
                 **kw) -> None:
        self.specs = tuple(_spec(kind, g, H, **kw) for g in generators)
        strides = {_stride(s) for s in self.specs}
        assert len(strides) == 1, f"one stride for every generator: {strides}"
        self.S = strides.pop()
        vals = (base + np.arange(E)).astype(np.int32)
        self.seeds = _dev(vals, wp.int32)
        self.big = _result_cls(self.specs[0]).alloc(Layout(E, self.S, DEV))
        self.views = (self.big.view(0, H), self.big.view(H, E))
        cls = _gen_cls(self.specs[0])
        self.gens = tuple(cls(spec, seeds=self.seeds[i * H:(i + 1) * H], out=self.views[i])
                          for i, spec in enumerate(self.specs))
        self.twin_seeds = tuple(_dev(vals[i * H:(i + 1) * H], wp.int32) for i in range(2))
        self.twins = tuple(cls(spec, seeds=self.twin_seeds[i])
                           for i, spec in enumerate(self.specs))
        wp.synchronize()

    def reseed(self, base: int) -> None:
        """Rewrite the caller's seed array (and the twins'), outside any capture."""
        vals = (base + np.arange(E)).astype(np.int32)
        _write(self.seeds, vals)
        for i in range(2):
            _write(self.twin_seeds[i], vals[i * H:(i + 1) * H])

    def twin_refs(self) -> list:
        return [_snapshot(t.program.result) for t in self.twins]


def _assert_halves(got: dict, refs, label: str) -> None:
    """Half ``i`` (envs ``[i*H, (i+1)*H)``) of the shared result equals ``refs[i]``, a
    standalone ``H``-env result, on every field the standalone result holds."""
    for i, ref in enumerate(refs):
        assert ref, label
        for name, want in ref.items():
            assert name in got, f"{label}: {name} missing from the shared result"
            np.testing.assert_array_equal(got[name].reshape(E, -1)[i * H:(i + 1) * H],
                                          want.reshape(H, -1),
                                          err_msg=f"{label}: half {i}: {name}")


def _assert_binds_the_view(gen, view, label: str) -> None:
    """The Program's result arrays ARE the caller's view (same device pointers)."""
    res = tg_capture.result_arrays(gen.program.result)
    want = tg_capture.result_arrays(view)
    assert want, label
    for name, arr in want.items():
        assert name in res, f"{label}: {name} not bound"
        assert res[name].ptr == arr.ptr, f"{label}: {name} is not the caller's view"


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("kind", ["track", "gate"])
def test_two_generators_write_one_shared_result_in_one_caller_graph(kind, mode):
    """The test gate: ``out=big.view(0, 4)`` (bezier) and ``out=big.view(4, 8)``
    (polar) of ONE caller-allocated result. The Programs write the caller's memory (the
    second half at ``big``'s pointer + 4 rows), the warm-ups fill each half with its
    twin's batch, and ONE caller graph holding both ``generate(advance=True)`` calls
    replays into a fully poisoned ``big`` to the standalone eager twins advanced alike,
    byte for byte per half; a reseed of the caller's seed array outside the graph
    reaches the next replay."""
    mx = _Mixed(kind)
    a, b = mx.gens
    for i, (g, v) in enumerate(zip(mx.gens, mx.views)):
        _assert_binds_the_view(g, v, f"{kind}: half {i}")
    S = mx.S
    item = 12                                       # vec3f bytes
    first = "center" if kind == "track" else "position"
    assert getattr(b.program.result, first).ptr == getattr(mx.big, first).ptr + H * S * item
    base = _snapshot(mx.big)
    _assert_halves(base, mx.twin_refs(), f"{kind}: the warm-ups wrote each half")

    def body(stream):
        assert a.generate(stream=stream, advance=True) is a.program.result
        assert b.generate(stream=stream, advance=True) is b.program.result

    replay = _caller_graph(mode, body)
    for g in mx.gens:
        assert g.graph is None, "recording into the caller's graph must not self-capture"
        np.testing.assert_array_equal(g.seeds.epoch.numpy(), 0,
                                      err_msg="the recording executed the epoch advance")
    _assert_host(_snapshot(mx.big), base, f"{kind}: the recording executed the pipelines")

    prev = base
    for k, reseed in enumerate((None, 2000, 1000)):
        label = f"{kind}/{mode}: replay {k}"
        if reseed is not None:
            mx.reseed(reseed)
        tg_capture.poison_all(mx.big)
        replay()
        for t in mx.twins:
            t.enqueue(advance=True)
        wp.synchronize()
        got = _snapshot(mx.big)
        _assert_halves(got, mx.twin_refs(), label)
        for g in mx.gens:
            np.testing.assert_array_equal(g.seeds.epoch.numpy(), k + 1, err_msg=label)
        assert _differs(got, prev), f"{label}: the replay did not change the batch"
        prev = got


@pytest.mark.parametrize("on_invalid", ["keep", "poison"])
def test_masked_regeneration_of_one_half_leaves_the_other_half_untouched(on_invalid):
    """ONE caller graph holds ``[copy done_src -> done] -> a.generate(advance=True,
    mask=done[0:4])`` (``a`` writes ``big.view(0, 4)``, ``b`` the other half). Before
    every replay the WHOLE shared result is poisoned; after it every byte of ``b``'s
    half (whose entries of ``done`` are set) and of ``a``'s unmasked envs still holds the
    poison, and ``a``'s masked envs equal the twin driven with the same masks (same seed
    and epoch). ``"poison"`` runs at ``n_max=80`` (truncation fails some envs): masked
    invalid envs are poisoned like the twin's."""
    kw = {} if on_invalid == "keep" else dict(n_max=80, spacing=0.06, on_invalid="poison")
    mx = _Mixed("track", **kw)
    a, b = mx.gens
    ta = mx.twins[0]
    done = wp.zeros(E, dtype=wp.int32, device=DEV)
    done_src = _dev(_ZERO, wp.int32)
    mask_a = done[0:H]
    _prepare_masked(a, mask_a)
    tmask = ta.inputs["regen_mask"]
    _prepare_masked(ta, tmask)
    base = _snapshot(mx.big)

    s = _side_stream()
    with _recording(s) as cap:
        wp.copy(done, done_src, stream=s)
        assert a.generate(stream=s, advance=True, mask=mask_a) is a.program.result
    assert a.graph is None
    _assert_host(_snapshot(mx.big), base, "the recording executed the pipeline")

    schedule = [_M1[:H], _M1[:H], _ZERO[:H], 1 - _M1[:H], _ONES[:H]]
    epoch = np.zeros(H, np.int32)
    poisoned_masked = 0
    for k, mk in enumerate(schedule):
        label = f"{on_invalid}: replay {k} mask {mk.tolist()}"
        _write(done_src, np.concatenate([mk, _ONES[:H]]))   # b's entries set: never read
        tg_capture.poison_all(mx.big)
        poison = _snapshot(mx.big)
        _replay(cap.graph, s)
        np.testing.assert_array_equal(done.numpy()[:H], mk, err_msg=f"{label}: mask copy")
        _write(tmask, mk)
        ta.enqueue(advance=True, mask=tmask)
        wp.synchronize()
        ref = _snapshot(ta.program.result)
        epoch += mk
        cur = _snapshot(mx.big)
        _assert_host(_rows(cur, slice(H, E)), _rows(poison, slice(H, E)),
                     f"{label}: the other half changed")
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)
        _assert_host(_rows(cur, unmasked), _rows(poison, unmasked),
                     f"{label}: an unmasked env of the regenerated half changed")
        _assert_host(_rows(cur, masked), _rows(ref, masked, H),
                     f"{label}: masked envs vs twin")
        np.testing.assert_array_equal(a.seeds.epoch.numpy(), epoch, err_msg=label)
        np.testing.assert_array_equal(b.seeds.epoch.numpy(), 0, err_msg=label)
        np.testing.assert_array_equal(cur["generation"][masked], epoch[masked],
                                      err_msg=label)
        _assert_valid_is_fail_mask_zero(_rows(cur, masked), label)
        if on_invalid == "poison":
            bad = masked[cur["fail_mask"][masked] != 0]
            np.testing.assert_array_equal(cur["count"][bad], 0, err_msg=label)
            assert np.all(np.isnan(cur["center"].reshape(E, -1)[bad])), label
            poisoned_masked += len(bad)
    if on_invalid == "poison":
        assert poisoned_masked, "no masked env failed: the poison variant is vacuous"


# ---------------------------------------------------------------------------------------
# copy_from under capture; cold calls refuse a capture
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["track", "gate"])
def test_copy_from_records_a_snapshot_into_the_caller_graph(kind):
    """``snap = out.clone()`` (cold, outside the capture), then ONE caller graph holds
    ``generate(advance=True) -> snap.copy_from(out, stream=s)``: no allocation (the
    mempool is disabled), and every replay into poisoned ``out`` AND ``snap`` leaves
    both equal to the eager twin advanced alike."""
    spec = _spec(kind, "bezier")
    gen = _gen_cls(spec)(spec, seeds=11)
    twin = _gen_cls(spec)(spec, seeds=11)
    out = gen.program.result
    snap = out.clone()
    wp.synchronize()
    assert snap.layout == out.layout == gen.program.layout
    _assert_host(_snapshot(snap), _snapshot(out), f"{kind}: clone")
    snap.copy_from(out)                              # eager: memcpy only
    wp.synchronize()

    s = _side_stream()
    with _recording(s) as cap:
        assert gen.generate(stream=s, advance=True) is out
        snap.copy_from(out, stream=s)
    for k in range(3):
        label = f"{kind}: replay {k}"
        tg_capture.poison_all(out, snap)
        _replay(cap.graph, s)
        twin.enqueue(advance=True)
        wp.synchronize()
        ref = _snapshot(twin.program.result)
        _assert_host(_snapshot(out), ref, f"{label}: result")
        _assert_host(_snapshot(snap), ref, f"{label}: snapshot")


def test_cold_calls_refuse_a_capture_before_any_work():
    """``clone()`` / ``alloc()`` allocate on Warp's current stream and
    ``load_centerlines`` allocates staging and blocks: each raises ``CaptureError``
    before any work, both for a capture on the stream they were told the producer uses
    (``stream=s``) and for a capture on Warp's current stream. The caller's capture
    stays valid (its own copy replays) and nothing was uploaded."""
    _load_test_kernels()
    gen = TrackGenerator(_spec("track", "bezier"), seeds=11)
    out = gen.program.result
    ext = TrackGenerator(_spec("track", EXTERNAL))
    inputs = {k: np.array(a.numpy(), copy=True) for k, a in ext.program.inputs.items()}
    rows = [_ellipse(200, 1.0, 0.8) for _ in range(E)]

    s = _side_stream()
    marker, marker_src = _marker()
    with _recording(s) as cap:
        wp.copy(marker, marker_src, stream=s)
        with pytest.raises(CaptureError):
            out.clone(stream=s)
        with pytest.raises(CaptureError):
            load_centerlines(ext, rows, stream=s)
    _replay(cap.graph, s)
    np.testing.assert_array_equal(marker.numpy(), marker_src.numpy())

    s2 = _side_stream()
    marker2, _ = _marker()
    with wp.ScopedStream(s2):                        # the capture stream is current
        with _recording(s2) as cap2:
            wp.copy(marker2, marker_src, stream=s2)
            with pytest.raises(CaptureError):
                out.clone()
            with pytest.raises(CaptureError):
                Track.alloc(out.layout)
            with pytest.raises(CaptureError):
                GateSequence.alloc(Layout(E, 32, DEV))
            with pytest.raises(CaptureError):
                load_centerlines(ext, rows)
    _replay(cap2.graph, s2)
    np.testing.assert_array_equal(marker2.numpy(), marker_src.numpy())
    for k, a in ext.program.inputs.items():
        np.testing.assert_array_equal(a.numpy(), inputs[k],
                                      err_msg=f"a refused load wrote inputs[{k!r}]")


# ---------------------------------------------------------------------------------------
# Streams: snapshots after an async generation; alloc fills before a side-stream warm-up
# ---------------------------------------------------------------------------------------

_SNAPSHOT_CASES = ("clone-stream", "clone-producer-default", "copy_from-producer-default",
                   "copy_from-stream")


@pytest.mark.parametrize("case", _SNAPSHOT_CASES)
def test_snapshot_after_an_async_generation_on_a_busy_stream(case):
    """``s`` is held busy, then ``generate(stream=s, sync=False, advance=True)``
    replays the (already captured) self-graph behind it, so the result still holds the
    poison when the snapshot is requested:

    - ``clone(stream=s)``, and ``clone()`` of a generator built with ``stream=s`` (the
      producer default): Warp's current stream waits on ``s``, so the clone read right
      away with ``.numpy()`` (on the current stream, no other sync) is the new batch;
    - ``copy_from(out)`` with the producer default, and ``copy_from(out, stream=s)``:
      the copies run on ``s`` after the generation (read after a device sync).

    Each snapshot equals the eager twin advanced alike (and ``out`` itself)."""
    _load_test_kernels()
    spec = _spec("track", "bezier")
    s = _side_stream()
    ctor = s if case.endswith("producer-default") else None
    gen = TrackGenerator(spec, seeds=11, stream=ctor)
    out = gen.program.result
    twin = TrackGenerator(spec, seeds=11)
    gen.generate(stream=s, advance=True)             # captures the advance self-graph
    snap = out.clone() if case.startswith("copy_from") else None
    for _ in range(2):
        twin.enqueue(advance=True)
    wp.synchronize()
    ref = _snapshot(twin.program.result)
    assert _differs(ref, _snapshot(out)), "the next generation must change the batch"

    tg_capture.poison_all(out)
    busy = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.synchronize()
    wp.launch(_spin_k, 1, inputs=[_SPIN, busy], device=DEV, stream=s)   # s busy
    assert gen.generate(stream=s, sync=False, advance=True) is out
    if case == "clone-stream":
        snap = out.clone(stream=s)
    elif case == "clone-producer-default":
        snap = out.clone()
    elif case == "copy_from-producer-default":
        snap.copy_from(out)
    else:
        snap.copy_from(out, stream=s)
    if case.startswith("copy_from"):
        wp.synchronize()                             # the copies ran on s
    got = _snapshot(snap)                            # a clone: read on the current stream
    wp.synchronize()
    _assert_host(got, ref, f"{case}: snapshot vs twin")
    _assert_host(_snapshot(out), ref, f"{case}: result vs twin")


@pytest.mark.parametrize("make", ["alloc", "clone"])
def test_a_fresh_snapshot_is_usable_on_the_producer_stream_right_away(make):
    """The documented hot-path snapshot pattern with NO intervening sync: Warp's current
    stream is held busy, ``snap = Track.alloc(layout)`` (its fills queued behind the
    spin) or ``snap = out.clone()``, then ``snap.copy_from(out)`` on the producer's
    default stream ``s`` (``TrackGenerator(..., stream=s)``). ``alloc`` / ``clone``
    return a result complete on every stream (a host wait on the current stream), so no
    late fill or clone copy lands on the snapshot after the copies on ``s``, and ``s``
    never writes a block the current stream still uses."""
    _load_test_kernels()
    spec = _spec("track", "bezier")
    s = _side_stream()
    gen = TrackGenerator(spec, seeds=11, stream=s)
    out = gen.program.result
    ref = _snapshot(out)
    busy = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.synchronize()
    wp.launch(_spin_k, 1, inputs=[_SPIN, busy], device=DEV)      # the current stream
    snap = Track.alloc(out.layout) if make == "alloc" else out.clone()
    snap.copy_from(out)                              # the producer default: stream s
    wp.synchronize()
    _assert_host(_snapshot(snap), ref, f"{make}: snapshot vs result")


def _assert_alloc_padding(host: dict, envs, label: str) -> None:
    """Rows ``envs`` of an ``alloc`` result in the "no track yet" state: per-point
    fields NaN (int32: -1), per-env fields 0."""
    for name, v in host.items():
        rows = v.reshape(E + 2, -1)[envs]
        if v.shape[0] == E + 2:
            assert np.all(rows == 0), f"{label}: {name}: per-env padding {np.unique(rows)}"
        elif np.issubdtype(v.dtype, np.integer):
            assert np.all(rows == -1), f"{label}: {name}: int padding"
        else:
            assert np.all(np.isnan(rows)), f"{label}: {name}: float padding"


def test_alloc_fills_on_the_current_stream_before_a_side_stream_warm_up():
    """``Track.alloc`` inside an ambient ``LaunchContext.scope(other)`` allocates AND
    fills on Warp's current stream (not the ambient one), with both ``other`` and the
    current stream held busy. A generator built afterwards (outside the scope) with
    ``out=big.view(1, E + 1)`` and ``stream=s`` waits on the current stream before its
    warm-up, so the view equals the standalone eager batch (no late fill overwrote it)
    and the rows outside the view keep the padding. Its int seed is offset by the view's
    first env: ``seeds=11`` on a view starting at env 1 seeds like a standalone
    ``seeds=12`` (a mixed batch with int seeds is seeded like one batch)."""
    _load_test_kernels()
    spec = _spec("track", "bezier")
    ref = _eager(spec, seeds=12)
    S = _stride(spec)
    other, s = _side_stream(), _side_stream()
    busy_o = wp.zeros(1, dtype=wp.float32, device=DEV)
    busy_c = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.synchronize()
    with LaunchContext.for_call(DEV, other).scope():
        wp.launch(_spin_k, 1, inputs=[_SPIN, busy_o], device=DEV, stream=other)
        wp.launch(_spin_k, 1, inputs=[_SPIN, busy_c], device=DEV)   # current stream
        big = Track.alloc(Layout(E + 2, S, DEV))
    assert big.layout == Layout(E + 2, S, DEV, True)
    view = big.view(1, E + 1)
    gen = TrackGenerator(spec, seeds=11, out=view, stream=s)
    wp.synchronize()
    _assert_binds_the_view(gen, view, "alloc view")
    got = _snapshot(big)
    _assert_host(_rows(got, slice(1, E + 1), E + 2), _rows(ref, slice(0, E)),
                 "the warm-up rows vs the eager batch")
    _assert_alloc_padding(got, [0, E + 1], "rows outside the view")


# ---------------------------------------------------------------------------------------
# validate / Layout on CUDA
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["track", "gate"])
def test_validate_names_a_field_on_the_wrong_device(kind):
    """A CPU array in a CUDA result: ``validate()`` names the field, the expected
    device (canonical ``cuda:0``) and the array's (``cpu``), with canonical and
    shorthand explicit layouts."""
    spec = _spec(kind, "bezier")
    gen = _gen_cls(spec)(spec, seeds=11)
    out = gen.program.result
    assert out.validate() == gen.program.layout == out.layout
    assert gen.program.layout.device == DEV
    field, dtype, owner = ("inner", wp.vec3f, "Track") if kind == "track" \
        else ("left", wp.vec3f, "GateSequence")
    n = int(getattr(out, field).shape[0])
    bad = dataclasses.replace(out, **{field: wp.zeros(n, dtype=dtype, device="cpu")})
    msg = rf"{owner}\.{field}: expected a contiguous \[{n}\] vec3f .* on {DEV}, " \
          rf"got shape \({n},\) vec3f on cpu"
    with pytest.raises(ValueError, match=msg):
        bad.validate()
    with pytest.raises(ValueError, match=msg):
        dataclasses.replace(bad, layout=Layout(E, gen.program.layout.stride, "cuda")).validate()


def test_hand_set_cuda_layout_matches_the_canonical_device():
    """``Layout(E, S, "cuda")`` keeps its device as given, and the library compares the
    canonical device: ``TrackGenerator(out=)`` of a result carrying it builds (the
    Program's layout is ``"cuda:0"``) and writes the eager batch, and ``copy_from``
    accepts it in both directions."""
    spec = _spec("track", "bezier")
    S = _stride(spec)
    hand_layout = Layout(E, S, "cuda")
    assert hand_layout.device == "cuda"
    hand = dataclasses.replace(Track.alloc(Layout(E, S, DEV)), layout=hand_layout)
    assert hand.validate() == hand_layout
    gen = TrackGenerator(spec, seeds=11, out=hand)
    assert gen.program.layout == Layout(E, S, DEV, True)
    wp.synchronize()
    ref = _eager(spec, seeds=11)
    _assert_host(_snapshot(hand), ref, "out= with a hand-set 'cuda' layout")

    snap = dataclasses.replace(Track.alloc(Layout(E, S, DEV)), layout=hand_layout)
    snap.copy_from(gen.program.result)
    wp.synchronize()
    _assert_host(_snapshot(snap), ref, "copy_from into a hand-set 'cuda' layout")
    owned = gen.program.result.clone()
    tg_capture.poison_all(owned)
    owned.copy_from(snap)
    wp.synchronize()
    _assert_host(_snapshot(owned), ref, "copy_from out of a hand-set 'cuda' layout")


# ---------------------------------------------------------------------------------------
# randomize.uniform in a caller graph
# ---------------------------------------------------------------------------------------

@pytest.fixture
def wobble():
    """``toy.wobble`` (dynamic ``amp`` in [0, 0.4]); undone by the per-test registry
    scope (tests/conftest.py)."""
    register_stage_toys(registry)


def _wobble_spec(randomize=True, **kw):
    r = {"generator.amp": (LO, HI)} if randomize else None
    return presets.track(num_envs=E, device=DEV, generator=WOBBLE, randomize=r, **kw)


def _drawn(gen) -> np.ndarray:
    return np.array(gen.params["generator"]["amp"].numpy(), copy=True)


def test_randomize_uniform_redraws_on_every_replay_of_a_caller_graph(wobble):
    """ONE caller graph holds ``generate(advance=True)`` of a pipeline whose
    ``generator.amp`` is drawn by ``randomize.uniform`` (listed before the generator).
    Every replay, into a poisoned Track AND a poisoned drawn array: the draws are in
    ``[lo, hi]``, distinct across envs, redrawn for every env (vs the previous replay),
    equal to the eager twin's (same seeds, advanced alike) and so is the Track; and the
    Track equals a non-randomized twin whose live ``amp`` is set to the drawn values
    before its generation (the draw reaches the generator in the SAME generation)."""
    spec = _wobble_spec()
    ids = list(spec.ids)
    assert RID in ids and ids.index(RID) < ids.index("generator"), ids
    gen = TrackGenerator(spec, seeds=11)
    twin = TrackGenerator(spec, seeds=11)
    plain = TrackGenerator(_wobble_spec(randomize=False), seeds=11)
    out = gen.program.result
    drawn_arr = gen.params["generator"]["amp"]
    prev = _drawn(gen)
    np.testing.assert_array_equal(prev, _drawn(twin), err_msg="construction draws")

    s = _side_stream()
    with _recording(s) as cap:
        assert gen.generate(stream=s, advance=True) is out
    assert gen.graph is None
    np.testing.assert_array_equal(_drawn(gen), prev, err_msg="the recording drew")

    for k in range(3):
        label = f"replay {k}"
        tg_capture.poison_all(out)
        tg_capture.poison(drawn_arr)
        wp.synchronize()
        _replay(cap.graph, s)
        twin.enqueue(advance=True)
        plain.params["generator"].set("amp", drawn_arr)
        plain.enqueue(advance=True)
        wp.synchronize()
        d = _drawn(gen)
        assert np.all((d >= np.float32(LO)) & (d <= np.float32(HI))), f"{label}: {d}"
        assert len(np.unique(d)) == E, f"{label}: draws not distinct across envs: {d}"
        assert np.all(d != prev), f"{label}: advance=True must redraw every env"
        np.testing.assert_array_equal(d, _drawn(twin), err_msg=f"{label}: draws vs twin")
        got = _snapshot(out)
        _assert_host(got, _snapshot(twin.program.result), f"{label}: Track vs twin")
        _assert_host(got, _snapshot(plain.program.result),
                     f"{label}: Track vs the twin given the drawn amp")
        prev = d


def test_randomize_bounds_written_inside_the_caller_graph(wobble):
    """An in-graph curriculum: ONE caller graph holds ``[params[randomizer].set("lo" /
    "hi", device sources)] -> generate(advance=True)``. Before each replay the sources
    are rewritten outside the graph; the replay copies them into the live bounds,
    latches them and draws within the per-env ``[lo, hi]`` of THAT replay, equal to the
    eager twin given the same bounds; a degenerate ``lo == hi == 0.3`` draws exactly
    0.3 and the Track equals a non-randomized twin with ``amp = 0.3``."""
    spec = _wobble_spec()
    gen = TrackGenerator(spec, seeds=11)
    twin = TrackGenerator(spec, seeds=11)
    fixed = TrackGenerator(_wobble_spec(randomize=False, generator_params={"amp": 0.3}),
                           seeds=11)
    out = gen.program.result
    lo_src, hi_src = _dev(_per_env(LO)), _dev(_per_env(HI))
    rp = gen.params[RID]
    rp.set("lo", lo_src)                             # warm-up: no-op writes
    rp.set("hi", hi_src)
    wp.synchronize()

    s = _side_stream()
    with _recording(s) as cap:
        rp.set("lo", lo_src, stream=s)
        rp.set("hi", hi_src, stream=s)
        assert gen.generate(stream=s, advance=True) is out
    assert gen.graph is None

    schedule = ((0.05, 0.10), ([0.0, 0.3], [0.1, 0.4]), (0.3, 0.3), (LO, HI))
    for k, (lo, hi) in enumerate(schedule):
        label = f"replay {k} lo={lo} hi={hi}"
        lo_v, hi_v = _per_env(lo), _per_env(hi)
        _write(lo_src, lo_v)
        _write(hi_src, hi_v)
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        np.testing.assert_array_equal(rp["lo"].numpy(), lo_v, err_msg=f"{label}: live lo")
        np.testing.assert_array_equal(rp["hi"].numpy(), hi_v, err_msg=f"{label}: live hi")
        lo_d, hi_d = _dev(lo_v), _dev(hi_v)        # alive until the sync below
        twin.params[RID].set("lo", lo_d)
        twin.params[RID].set("hi", hi_d)
        twin.enqueue(advance=True)
        fixed.enqueue(advance=True)
        wp.synchronize()
        d = _drawn(gen)
        assert np.all((d >= lo_v) & (d <= hi_v)), f"{label}: draws {d}"
        np.testing.assert_array_equal(d, _drawn(twin), err_msg=f"{label}: draws vs twin")
        got = _snapshot(out)
        _assert_host(got, _snapshot(twin.program.result), f"{label}: Track vs twin")
        if lo == hi == 0.3:
            np.testing.assert_array_equal(d, np.float32(0.3), err_msg=label)
            _assert_host(got, _snapshot(fixed.program.result),
                         f"{label}: Track vs the amp=0.3 twin")


def test_randomize_masked_replay_redraws_the_masked_envs_only(wobble):
    """ONE caller graph holds ``[copy mask_src -> regen_mask] -> generate(advance=True,
    mask=regen_mask)``. Each replay into a poisoned Track: masked envs get new draws and
    the twin's rows; unmasked envs redraw their previous values bit for bit (same
    effective seed, same latched bounds) and keep every Track byte (the poison)."""
    spec = _wobble_spec()
    gen = TrackGenerator(spec, seeds=11)
    twin = TrackGenerator(spec, seeds=11)
    out = gen.program.result
    m, tm = gen.inputs["regen_mask"], twin.inputs["regen_mask"]
    _prepare_masked(gen, m)
    _prepare_masked(twin, tm)
    mask_src = _dev(_ZERO, wp.int32)

    s = _side_stream()
    with _recording(s) as cap:
        wp.copy(m, mask_src, stream=s)
        assert gen.generate(stream=s, advance=True, mask=m) is out
    prev = _drawn(gen)
    for k, mk in enumerate((_M1, 1 - _M1, _ZERO, _ONES)):
        label = f"replay {k} mask {mk.tolist()}"
        _write(mask_src, mk)
        tg_capture.poison_all(out)
        poison = _snapshot(out)
        _replay(cap.graph, s)
        _write(tm, mk)
        twin.enqueue(advance=True, mask=tm)
        wp.synchronize()
        d = _drawn(gen)
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)
        np.testing.assert_array_equal(d[unmasked], prev[unmasked],
                                      err_msg=f"{label}: unmasked draws")
        assert np.all(d[masked] != prev[masked]), f"{label}: masked envs not redrawn"
        np.testing.assert_array_equal(d, _drawn(twin), err_msg=f"{label}: draws vs twin")
        cur, ref = _snapshot(out), _snapshot(twin.program.result)
        _assert_host(_rows(cur, unmasked), _rows(poison, unmasked),
                     f"{label}: an unmasked env's Track changed")
        _assert_host(_rows(cur, masked), _rows(ref, masked), f"{label}: masked vs twin")
        prev = d


# ---------------------------------------------------------------------------------------
# generator/external: in-graph ingestion and the loader's stream
# ---------------------------------------------------------------------------------------

def _external(**kw) -> TrackGenerator:
    return TrackGenerator(_spec("track", EXTERNAL, **kw))


def _ellipse(m: int, ax: float, by: float, phase: float = 0.0) -> np.ndarray:
    th = phase + 2.0 * np.pi * np.arange(m) / m
    return np.stack([ax * np.cos(th), by * np.sin(th)], -1).astype(np.float32)


def _input_rows(gen) -> list:
    """The rows the caller's graph wrote: env ``e``'s first ``count[e]`` points."""
    xy = gen.program.inputs["centerline.xy"].numpy()
    count = gen.program.inputs["centerline.count"].numpy()
    n = xy.shape[0] // E
    return [np.array(xy[e * n:e * n + int(count[e])], copy=True) for e in range(E)]


def _assert_on_ellipses(host: dict, k: int, n: int, label: str) -> None:
    """The ingested centerline (relax off by default) lies on replay ``k``'s ellipses
    (within the chord sagitta of the resampling), every env valid."""
    np.testing.assert_array_equal(host["valid"], 1, err_msg=f"{label}: valid")
    c = host["center"].reshape(E, -1, 3)
    for e in range(E):
        _, ax, by = _ellipse_params(k, e, n)
        m = int(host["count"][e])
        q = np.hypot(c[e, :m, 0] / ax, c[e, :m, 1] / by)
        assert np.max(np.abs(q - 1.0)) <= _ELLIPSE_TOL, \
            f"{label}: env {e}: off its ellipse by {np.max(np.abs(q - 1.0))}"
        assert np.all(c[e, :m, 2] == 0.0), f"{label}: env {e}: z"


def test_external_inputs_written_inside_the_caller_graph():
    """ONE caller graph holds ``[bump a device replay counter] -> [a kernel writes
    program.inputs["centerline.xy"] (per-env ellipses from the counter, NaN past the
    count) and ["centerline.count"]] -> generate()``. Every replay into a poisoned
    Track equals the eager twin fed the same rows by ``load_centerlines`` (the sequence
    form: only the first ``count[e]`` points) and ``enqueue()``, follows the analytic
    ellipse of that replay (relax off by default), and changes the batch."""
    _load_test_kernels()
    gen, twin = _external(), _external()
    inputs = gen.program.inputs
    assert gen.inputs["centerline.xy"] is inputs["centerline.xy"]
    xy, count = inputs["centerline.xy"], inputs["centerline.count"]
    assert xy.dtype is wp.vec2f and count.dtype is wp.int32 and count.shape == (E,)
    n = int(xy.shape[0]) // E
    out = gen.program.result
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)
    wp.synchronize()

    s = _side_stream()
    with _recording(s) as cap:
        wp.launch(_bump_k, 1, inputs=[counter], device=DEV, stream=s)
        wp.launch(_ellipse_rows_k, E * n, inputs=[counter, n, xy, count], device=DEV,
                  stream=s)
        assert gen.generate(stream=s) is out
    assert gen.graph is None
    assert int(counter.numpy()[0]) == 0, "the recording executed the caller's kernels"

    prev = _snapshot(out)
    for k in range(1, 4):
        label = f"replay {k}"
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        assert int(counter.numpy()[0]) == k
        np.testing.assert_array_equal(count.numpy(),
                                      [_ellipse_params(k, e, n)[0] for e in range(E)],
                                      err_msg=f"{label}: in-graph count")
        load_centerlines(twin, _input_rows(gen))
        twin.enqueue()
        wp.synchronize()
        got = _snapshot(out)
        _assert_host(got, _snapshot(twin.program.result), f"{label}: vs eager twin")
        _assert_on_ellipses(got, k, n, label)
        assert _differs(got, prev), f"{label}: the replay did not change the batch"
        prev = got


def test_external_masked_ingestion_inside_the_caller_graph():
    """ONE caller graph writes new rows for EVERY env (as above), copies the mask, then
    ``generate(advance=True, mask=regen_mask)``. Each replay into a poisoned Track:
    unmasked envs keep every byte (the commit restores them; their new rows are
    ignored), masked envs equal the twin loaded with the same rows and driven with the
    same mask."""
    _load_test_kernels()
    gen, twin = _external(), _external()
    xy = gen.program.inputs["centerline.xy"]
    count = gen.program.inputs["centerline.count"]
    n = int(xy.shape[0]) // E
    out = gen.program.result
    m, tm = gen.inputs["regen_mask"], twin.inputs["regen_mask"]
    _prepare_masked(gen, m)
    _prepare_masked(twin, tm)
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)
    mask_src = _dev(_ZERO, wp.int32)

    s = _side_stream()
    with _recording(s) as cap:
        wp.launch(_bump_k, 1, inputs=[counter], device=DEV, stream=s)
        wp.launch(_ellipse_rows_k, E * n, inputs=[counter, n, xy, count], device=DEV,
                  stream=s)
        wp.copy(m, mask_src, stream=s)
        assert gen.generate(stream=s, advance=True, mask=m) is out

    for k, mk in enumerate((_M1, 1 - _M1, _ZERO, _ONES), start=1):
        label = f"replay {k} mask {mk.tolist()}"
        _write(mask_src, mk)
        tg_capture.poison_all(out)
        poison = _snapshot(out)
        _replay(cap.graph, s)
        load_centerlines(twin, _input_rows(gen))
        _write(tm, mk)
        twin.enqueue(advance=True, mask=tm)
        wp.synchronize()
        cur, ref = _snapshot(out), _snapshot(twin.program.result)
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)
        _assert_host(_rows(cur, unmasked), _rows(poison, unmasked),
                     f"{label}: an unmasked env's Track changed")
        _assert_host(_rows(cur, masked), _rows(ref, masked), f"{label}: masked vs twin")
    _assert_host(cur, ref, "after an all-ones mask every env is committed")


@pytest.mark.parametrize("how", ["constructor-stream", "stream-argument"])
def test_load_centerlines_is_ordered_after_an_in_flight_generation(how):
    """``s`` is held busy, ``generate(stream=s, sync=False)`` is queued behind it (it
    reads rows A), then ``load_centerlines(gen, rows_B)``: on the generator's
    constructor stream by default (``"constructor-stream"``) or with ``stream=s``. The
    upload is ordered after the in-flight generation, so that generation builds rows
    A (the twin of A), and the next one rows B."""
    _load_test_kernels()
    s = _side_stream()
    gen = _external() if how == "stream-argument" else \
        TrackGenerator(_spec("track", EXTERNAL), stream=s)
    n = int(gen.program.inputs["centerline.xy"].shape[0]) // E
    rows_a = [_ellipse(n - 8 * e, 1.0 + 0.02 * e, 0.8, 0.1 * e) for e in range(E)]
    rows_b = [_ellipse(n, 1.3, 0.6) for _ in range(E)]
    refs = []
    for rows in (rows_a, rows_b):
        twin = _external()
        load_centerlines(twin, rows)
        twin.enqueue()
        wp.synchronize()
        refs.append(_snapshot(twin.program.result))
    assert _differs(refs[0], refs[1]), "rows A and B must build different batches"
    kw = {} if how == "constructor-stream" else {"stream": s}
    load_centerlines(gen, rows_a, **kw)
    out = gen.generate(stream=s)                     # captures the self-graph
    _assert_host(_snapshot(out), refs[0], "rows A")

    tg_capture.poison_all(out)
    busy = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.synchronize()
    wp.launch(_spin_k, 1, inputs=[_SPIN, busy], device=DEV, stream=s)   # s busy
    assert gen.generate(stream=s, sync=False) is out                     # reads A
    load_centerlines(gen, rows_b, **kw)
    wp.synchronize()
    _assert_host(_snapshot(out), refs[0], f"{how}: the in-flight generation read B")
    gen.generate(stream=s)
    _assert_host(_snapshot(out), refs[1], f"{how}: the load did not happen")


# ---------------------------------------------------------------------------------------
# A utility bound to a view, in the same caller graph
# ---------------------------------------------------------------------------------------

def _boxes(n_envs: int, B: int):
    pts = [(0.0, 0.0), (0.8, 0.0), (0.0, 0.75), (-0.65, -0.3)][:B]
    pos = np.array([[x, y, 0.0] for _ in range(n_envs) for x, y in pts], np.float32)
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (n_envs * B, 1))
    he = np.tile(np.array([0.03, 0.02], np.float32), (n_envs * B, 1))
    out = (wp.array(pos, dtype=wp.vec3f, device=DEV),
           wp.array(quat, dtype=wp.quatf, device=DEV),
           wp.array(he, dtype=wp.vec2f, device=DEV))
    wp.synchronize()
    return out


@pytest.mark.parametrize("method", ["segments", "sdf"])
def test_collision_on_a_view_inside_the_caller_graph(method):
    """A ``CollisionChecker`` bound to the OFFSET half ``big.view(4, 8)`` of a mixed
    batch; ONE caller graph holds both ``generate(advance=True)`` calls, the sdf
    ``bake`` and the bound ``query``. Every replay's contacts (poisoned before) equal
    those of the same checker built on an owned ``clone()`` of the view's rows, and
    change with the batch."""
    mx = _Mixed("track")
    a, b = mx.gens
    view = mx.views[1]
    B = 4
    pos, orient, he = _boxes(H, B)
    kw = dict(max_boxes=B, method=method, sdf_resolution=32, position=pos,
              orientation=orient, half_extents=he)
    chk = CollisionChecker(view, **kw)
    contact = chk.query()                            # warm-up (sdf: baked at construction)
    wp.synchronize()

    s = _side_stream()
    with _recording(s) as cap:
        a.generate(stream=s, advance=True)
        b.generate(stream=s, advance=True)
        if method == "sdf":
            chk.bake(stream=s)
        assert chk.query(stream=s) is contact
    prev = None
    for k in range(3):
        label = f"{method}: replay {k}"
        tg_capture.poison_all(contact)
        _replay(cap.graph, s)
        owned = view.clone()
        ref = CollisionChecker(owned, **kw).query()
        wp.synchronize()
        _assert_host(_snapshot(owned), _snapshot(view), f"{label}: clone of the view")
        got = _snapshot(contact)
        _assert_host(got, _snapshot(ref), f"{label}: contacts on the view vs a clone")
        if prev is not None:
            assert _differs(got, prev), f"{label}: contacts did not follow the batch"
        prev = got


def test_documented_mixed_batch_step_graph():
    """The tutorial's mixed-batch step graph (docs/tutorials/cuda-graph-in-a-sim.rst):
    ONE caller graph holds ``[copy done_src -> done] -> gens[k].generate(advance=True,
    mask=done[k*4:(k+1)*4])`` for both halves (the second mask is an OFFSET slice of the
    caller's mask) and a ``CollisionChecker`` query on the WHOLE shared Track. Each
    replay: per half, masked envs equal that half's twin driven with the same mask slice
    and unmasked envs keep their rows bit for bit; the contacts (poisoned before) equal
    those of the same checker on an owned clone of the whole result."""
    mx = _Mixed("track")
    B = 2
    pos, orient, he = _boxes(E, B)
    kw = dict(max_boxes=B, position=pos, orientation=orient, half_extents=he)
    chk = CollisionChecker(mx.big, **kw)
    contact = chk.query()
    done = wp.zeros(E, dtype=wp.int32, device=DEV)
    done_src = _dev(_ZERO, wp.int32)
    masks = tuple(done[i * H:(i + 1) * H] for i in range(2))
    for g, m in zip(mx.gens, masks):
        _prepare_masked(g, m)
    for t in mx.twins:
        _prepare_masked(t, t.inputs["regen_mask"])

    s = _side_stream()
    with _recording(s) as cap:
        wp.copy(done, done_src, stream=s)
        for g, m in zip(mx.gens, masks):
            assert g.generate(stream=s, advance=True, mask=m) is g.program.result
        assert chk.query(stream=s) is contact

    prev = _snapshot(mx.big)
    epoch = np.zeros(E, np.int32)
    for k, mk in enumerate((_M1, 1 - _M1, _ZERO, _ONES)):
        label = f"replay {k} mask {mk.tolist()}"
        _write(done_src, mk)
        tg_capture.poison_all(contact)
        _replay(cap.graph, s)
        for i, t in enumerate(mx.twins):
            _write(t.inputs["regen_mask"], mk[i * H:(i + 1) * H])
            t.enqueue(advance=True, mask=t.inputs["regen_mask"])
        owned = mx.big.clone()
        ref_contact = CollisionChecker(owned, **kw).query()
        wp.synchronize()
        epoch += mk
        cur = _snapshot(mx.big)
        for i, (g, t) in enumerate(zip(mx.gens, mx.twins)):
            half = mk[i * H:(i + 1) * H]
            masked, unmasked = np.flatnonzero(half), np.flatnonzero(half == 0)
            ref = _snapshot(t.program.result)
            _assert_host(_rows(cur, masked + i * H), _rows(ref, masked, H),
                         f"{label}: half {i}: masked envs vs twin")
            _assert_host(_rows(cur, unmasked + i * H), _rows(prev, unmasked + i * H),
                         f"{label}: half {i}: an unmasked env changed")
            np.testing.assert_array_equal(g.seeds.epoch.numpy(), epoch[i * H:(i + 1) * H],
                                          err_msg=f"{label}: half {i}: epochs")
        _assert_host(_snapshot(contact), _snapshot(ref_contact),
                     f"{label}: contacts on the shared Track vs a clone")
        prev = cur


# ---------------------------------------------------------------------------------------
# The conformance kit on generator/external
# ---------------------------------------------------------------------------------------

def test_external_passes_check_generator_on_cuda():
    """``check_generator("external", devices=[cuda:0])`` passes with no ``skip=``:
    ``diversity`` is reported skipped (``seeded=False``), ``output.view`` ran, and the
    caller-capture checks did not fail."""
    report = check_generator(EXTERNAL, devices=[DEV])
    report.raise_for_failures()
    assert report.result("diversity", DEV).skipped, str(report)
    assert not report.result("output.view", DEV).skipped, str(report)
    assert not report.result("capture.self_graph", DEV).skipped, str(report)
