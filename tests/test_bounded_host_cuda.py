"""CUDA-only: the BOUNDED tier (``ctx.bounded_loop``), repulsive as a BOUNDED Stage, and
segmented HOST execution (roadmap PR12).

BOUNDED: ``LaunchContext.bounded_loop(n, body, active=, first_check=, check_every=)``
has three lowerings, chosen per call from the capture state (never from data):

- (a) a host poll that stops the loop (CPU eager by default; on CUDA only through the
  private ``_force_lowering("poll")`` seam, with a pinned buffer the slot allocates on
  that first forced poll);
- (b) one CUDA conditional node per window (``wp.capture_if``) inside a Warp-registered
  capture with conditional-node support (a producer's own graph, a Warp
  ``ScopedCapture``, ``interop.torch.registered(conditional=True)``);
- (c) every window runs (or is recorded) unconditionally: CUDA eager calls (no host
  sync, the ``enqueue()`` contract), plain ``torch.cuda.graph``, ``conditional_ok``
  False (an older driver).

With an absorbing freeze the three are bitwise identical; a non-absorbing loop is not.

HOST: a pipeline with HOST stages is split into segments of equal capture class; the
runner self-graphs every non-HOST segment and runs the HOST ones eagerly between the
graph launches; under a caller's capture it refuses (``NotCapturableError`` naming the
HOST stage ids) before any launch.

Method (as in tests/test_caller_capture_cuda.py and tests/test_inplace_regen_cuda.py;
the helpers are :mod:`track_gen.testing.capture`'s): record on a NON-current side
stream, with the Warp mempool disabled (an allocation inside the capture fails loudly)
and ``wp.capture_launch`` replaced by a tripwire (a graph launch cannot be recorded into
another capture); poison the outputs with FINITE, dtype-aware sentinels before every
replay (outputs are NaN-padded and ``assert_array_equal`` treats NaN == NaN, so a NaN
poison would hide a missed write); compare with an EAGER twin (``enqueue()``, which never
self-graphs and, on CUDA, runs every window: lowering (c)); reseed between replays (a
replay must read the live seeds). Construction prepares every producer (forced unroll),
so every kernel a window can launch is loaded outside the captures. Lowerings are forced
through the private seams the conformance kit uses (``track_gen.exec.context.
_force_lowering`` / ``_probe_loops``) and ``_compat.conditional_ok`` (as
tests/test_inplace_regen_cuda.py does).

The toys come from tests/_bounded_toys.py (public plugin surface only): the BOUNDED
relax smoothers ``toy.converging_smooth`` (absorbing per-env convergence freeze) and
``toy.leaky_smooth`` (same schedule, NOT absorbing), the HOST generator
``toy.host_circle`` (numpy host RNG uploaded through ``ctx.host()``) and the HOST gate
generator ``toy.host_ring``. Module-level imports are warp / numpy / track_gen / the
toys only (torch is imported inside the tests that need it), so the module collects on
a CPU-only machine (the ``cuda`` marker deselects / skips it there).
"""
from __future__ import annotations

import contextlib
import ctypes
import gc
import sys
import threading
import weakref

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

import track_gen.exec as tg_exec  # noqa: E402
from tests import _bounded_toys  # noqa: E402
from track_gen import GateGenConfig, PerEnvSeededRNG, TrackGenConfig  # noqa: E402
from track_gen._src import runtime as tg_runtime  # noqa: E402
from track_gen._src import warp_pipeline  # noqa: E402
from track_gen._src.gate_generator import GateGenerator  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen.course import Course, CourseConfig  # noqa: E402
from track_gen.exec import _compat as tg_compat  # noqa: E402
from track_gen.exec import context as tg_context  # noqa: E402
from track_gen.exec import runner as runner_mod  # noqa: E402
from track_gen.pipeline import build as build_program  # noqa: E402
from track_gen.pipeline import presets  # noqa: E402
from track_gen.plugin import CaptureClass, GeneratorSpec, registry  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402
from track_gen.testing import check_gate_generator, check_generator, check_stage  # noqa: E402

DEV = "cuda:0"
E = 8

CONVERGING = _bounded_toys.CONVERGING      # "toy.converging_smooth"
LEAKY = _bounded_toys.LEAKY                # "toy.leaky_smooth"
HOST_CIRCLE = _bounded_toys.HOST_CIRCLE    # "toy.host_circle"
HOST_RING = _bounded_toys.HOST_RING        # "toy.host_ring"

# The small workload preserves the defaults' growth schedule at three lower
# resolutions: 189 iterations, first check 96, six windows of 16. Ordinary feasible
# outputs may use the full budget; early convergence is a separate promise.
_SMALL = dict(num_points=64, repulsive_stages=(16, 32, 64), N_max=384, half_width=0.1)
_REPULSIVE_SCHEDULE = {"n_iters": 189, "first_check": 96, "check_every": 16, "windows": 6}
# A supported near-circle problem for genuine absorbing-freeze checks (the same
# workload as test_conformance.py). Fixed seeds must skip windows; no seed search or
# optional skip may hide a failure to exercise the different lowerings.
_CONVERGING_REPULSIVE = dict(**_SMALL, repulsive_grow_mult_min=1.1,
                            repulsive_grow_mult_max=1.1,
                            repulsive_obstacle_count_min=1,
                            repulsive_obstacle_count_max=1,
                            repulsive_stall_area_tol=0.5)
_CONVERGING_SCHEDULE = {"n_iters": 53, "first_check": 16, "check_every": 16, "windows": 3}

_ZERO = np.zeros(E, np.int32)
# Mixed mask: neighbours differ, so a write that leaks across an env boundary shows up.
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)

_NO_CONDITIONAL = (
    "CUDA conditional graph nodes are unsupported here (wp.is_conditional_graph_supported() "
    "is False: Warp must be built with CUDA Toolkit >= 12.4 and run on a >= 12.4 driver), so "
    "lowering (b) (one capture_if per window) cannot be recorded; (c) is covered by the "
    "'warp-unrolled' / 'torch' variants")


# ---------------------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _toys(_isolate_plugin_registry):
    """The bounded / HOST toys, registered inside the per-test registry scope of
    tests/conftest.py (undone after the test)."""
    _bounded_toys.register_bounded_toys(registry)
    yield


def _force_lowering(mode):
    return tg_context._force_lowering(mode)


def _probe_loops():
    return tg_context._probe_loops()


def _conditional_supported() -> bool:
    return bool(wp.is_conditional_graph_supported())


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch built without CUDA")
    return torch


def _side_stream() -> wp.Stream:
    return tg_capture.side_stream(DEV)


def _recording(stream):
    """The caller's Warp capture on ``stream``: mempool disabled, capture_launch
    forbidden (tests/test_caller_capture_cuda.py)."""
    return tg_capture.recording(DEV, stream)


def _replay(graph, stream) -> None:
    wp.capture_launch(graph, stream=stream)
    wp.synchronize()


def _arrays(obj) -> dict:
    return tg_capture.result_arrays(obj)


def _snapshot(obj) -> dict:
    return {k: np.array(a.numpy(), copy=True) for k, a in _arrays(obj).items()}


def _poison_all(*objs) -> None:
    tg_capture.poison_all(*objs)


def _poison_all_but_generation(obj) -> None:
    """Poison every field but ``generation`` (a masked generation writes it for the
    masked envs only: the tests assert it against the epochs directly)."""
    for name, arr in _arrays(obj).items():
        if name != "generation":
            tg_capture.poison(arr)
    wp.synchronize()


def _assert_host(got: dict, ref: dict, label: str) -> None:
    assert ref and got.keys() == ref.keys(), f"{label}: {sorted(set(got) ^ set(ref))}"
    for name in ref:
        np.testing.assert_array_equal(got[name], ref[name], err_msg=f"{label}: {name}")


def _differs(a: dict, b: dict) -> bool:
    return any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def _rows(host: dict, n: int = E) -> dict:
    """name -> [n, -1]: row ``e`` is env ``e``'s slice of an env-major array."""
    return {k: v.reshape(n, -1) for k, v in host.items()}


def _seed_values(seeds, n: int = E) -> np.ndarray:
    """The ``[n]`` int32 seeds of ``PerEnvSeededRNG(seeds=...)``: ``base + e`` for an int."""
    if isinstance(seeds, (int, np.integer)):
        return (int(seeds) + np.arange(n)).astype(np.int32)
    return np.asarray(seeds, np.int32)


def _rng(seeds, n: int = E) -> PerEnvSeededRNG:
    arr = wp.array(_seed_values(seeds, n), dtype=wp.int32, device=DEV)
    wp.synchronize()
    return PerEnvSeededRNG(seeds=arr, num_envs=n, device=DEV)


def _reseed(rng, seeds, n: int = E) -> None:
    """Reseed outside any capture (the graphs read the rng's seeds by pointer)."""
    rng.set_seeds_warp(wp.array(_seed_values(seeds, n), dtype=wp.int32, device=DEV), None)
    wp.synchronize()


def _track(generator: str, seeds=11, *, n: int = E, **cfg):
    rng = _rng(seeds, n)
    gen = TrackGenerator(TrackGenConfig(generator=generator, num_envs=n, device=DEV, **cfg),
                         rng)
    return gen, rng


def _track_spec(spec, seeds=11):
    rng = _rng(seeds, int(spec.batch.num_envs))
    return TrackGenerator(spec, rng), rng


def _gates(generator: str, seeds=11):
    rng = _rng(seeds)
    cfg = GateGenConfig(generator=generator, num_envs=E, device=DEV, gate_width=0.05)
    return GateGenerator(cfg, rng), rng


def _eager(gen, **kw) -> dict:
    """``gen.enqueue(**kw)`` (eager: never a graph, lowering (c) on CUDA), snapshot."""
    out = gen.enqueue(**kw)
    wp.synchronize()
    assert gen.graph is None, "enqueue() must never self-graph"
    return _snapshot(out)


def _stage_id(spec, kind: str, name: str) -> str:
    ids = [r.slot_id for r in spec.stages if r.kind == kind and r.name == name]
    assert len(ids) == 1, f"{kind}.{name} in {spec.ids}"
    return ids[0]


def _record(records: list, stage_id: str, label: str):
    """The one bounded loop of ``stage_id`` the probe saw."""
    mine = [r for r in records if r.stage_id == stage_id]
    assert len(mine) == 1, (
        f"{label}: expected one bounded loop of {stage_id!r}; the probe saw "
        f"{[(r.stage_id, r.lowering) for r in records]}")
    return mine[0]


@contextlib.contextmanager
def _counting_syncs():
    """Count ``_compat.stream_synchronize`` calls (the only host wait of the library's
    hot paths: the forced CUDA poll, HostOps, ``sync=True``) for the block."""
    calls: list = []
    real = tg_compat.stream_synchronize

    def counting(stream):
        calls.append(stream)
        return real(stream)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tg_compat, "stream_synchronize", counting)
        yield calls


@contextlib.contextmanager
def _counting_capture_launch(probe=None):
    """Count graph launches (``wp.capture_launch``, also the runner's reference) for the
    block; ``probe()`` runs before each launch."""
    calls: list = []
    real = wp.capture_launch

    def counting(graph, *args, **kwargs):
        calls.append(graph)
        if probe is not None:
            probe()
        return real(graph, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wp, "capture_launch", counting)
        if runner_mod.wp is not wp:   # pragma: no cover - the runner imports warp as wp
            mp.setattr(runner_mod.wp, "capture_launch", counting)
        yield calls


@contextlib.contextmanager
def _unconditional():
    """``skip_unless`` (hence every bounded window) takes its unconditional lowering (c),
    as on a pre-12.4 driver, whatever this machine supports."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tg_compat, "conditional_ok", lambda device: False)
        yield


def _marker():
    """A tiny copy the caller records itself, so its graph is never empty."""
    src = wp.array(np.array([1, 2, 3], np.int32), dtype=wp.int32, device=DEV)
    dst = wp.zeros(3, dtype=wp.int32, device=DEV)
    wp.synchronize()
    return dst, src


# ---------------------------------------------------------------------------------------
# BOUNDED units: ordinary Repulsive workloads and explicit converging workloads
# ---------------------------------------------------------------------------------------

class _Unit:
    """A BOUNDED stage under test inside a TrackGenerator: ``make(seeds)`` builds a
    fresh (prepared) generator and its rng; ``stage_id`` is the slot running the loop."""

    def __init__(self, label: str, make, stage_id: str, schedule: dict,
                 *, must_skip: bool = False) -> None:
        self.label = label
        self.make = make
        self.stage_id = stage_id
        self.schedule = schedule
        self.must_skip = must_skip

    def check_schedule(self, rec, label: str) -> None:
        got = {"n_iters": rec.n_iters, "first_check": rec.first_check,
               "check_every": rec.check_every, "windows": rec.windows}
        assert got == self.schedule, f"{label}: loop schedule {got} != {self.schedule}"


def _converging_spec(name: str = CONVERGING, n: int = E):
    return presets.track(num_envs=n, device=DEV, relax=name)


def _toy_schedule(name: str) -> dict:
    p = registry.get("relax", name).Params()
    n, f, w = int(p.iters), int(p.first_check), int(p.check_every)
    return {"n_iters": n, "first_check": f, "check_every": w,
            "windows": len(range(f, n, w))}


def _unit(name: str) -> _Unit:
    if name == "repulsive-small":
        return _Unit(name, lambda seeds: _track("repulsive", seeds, **_SMALL), "generator",
                     _REPULSIVE_SCHEDULE)
    if name == "repulsive-default":
        return _Unit(name, lambda seeds: _track("repulsive", seeds), "generator",
                     _REPULSIVE_SCHEDULE)
    if name == "repulsive-converging":
        return _Unit(name, lambda seeds: _track("repulsive", seeds,
                                               **_CONVERGING_REPULSIVE),
                     "generator", _CONVERGING_SCHEDULE, must_skip=True)
    assert name in (CONVERGING, LEAKY), name
    spec = _converging_spec(name)
    return _Unit(name, lambda seeds: _track_spec(_converging_spec(name), seeds),
                 _stage_id(spec, "relax", name), _toy_schedule(name), must_skip=True)


_UNITS = ("repulsive-small", "repulsive-default", "repulsive-converging", CONVERGING)


def _poll(gen, unit: _Unit, label: str):
    """One eager generation under the forced host poll (lowering (a) on CUDA); returns
    ``(snapshot, record, stream_synchronize calls)``."""
    _poison_all(gen.program.result)
    with _force_lowering("poll"), _probe_loops() as recs, _counting_syncs() as syncs:
        out = gen.enqueue()
    wp.synchronize()
    rec = _record(recs, unit.stage_id, label)
    assert rec.lowering == "poll", f"{label}: forced poll ran as {rec.lowering!r}"
    return _snapshot(out), rec, len(syncs)


def _assert_early_exit(unit: _Unit, base: int) -> None:
    """Converging workloads must exercise a real skipped window on fixed seeds."""
    gen, _ = unit.make(base)
    _, rec, _ = _poll(gen, unit, f"{unit.label}: base {base}")
    unit.check_schedule(rec, unit.label)
    assert rec.ran < rec.windows, f"{unit.label}: (a) skipped no window (vacuous)"
    if unit.label == "repulsive-converging":
        assert np.all(_frozen(gen)), "the converging Repulsive batch did not freeze"


# ---------------------------------------------------------------------------------------
# 1. Lowerings (a) == (b) == (c), bitwise
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("unit_name", _UNITS)
def test_cuda_eager_is_the_unrolled_lowering_and_never_host_syncs(unit_name):
    """CUDA eager calls (``enqueue()``) keep the "never host-syncs" contract: every window
    runs (lowering (c), probe ``ran == windows``) and no stream is synchronized. The loop
    has its pinned schedule, including the real defaults and the separate converging
    Repulsive workload."""
    unit = _unit(unit_name)
    gen, _ = unit.make(11)
    _poison_all(gen.program.result)
    with _probe_loops() as recs, _counting_syncs() as syncs:
        out = gen.enqueue()
    assert syncs == [], f"{unit_name}: CUDA eager enqueue() host-synced {len(syncs)} time(s)"
    wp.synchronize()
    rec = _record(recs, unit.stage_id, unit_name)
    assert rec.lowering == "unroll", f"{unit_name}: CUDA eager lowered as {rec.lowering!r}"
    assert rec.ran == rec.windows
    unit.check_schedule(rec, unit_name)
    assert gen.graph is None
    _assert_host(_snapshot(out), _eager(unit.make(11)[0]), f"{unit_name}: eager twin")


_MODES = ("poll", "warp", "warp-unrolled", "torch", "torch-registered")


def _record_and_replay(unit: _Unit, base: int, mode: str, ref: dict, ref2: dict,
                       base2: int) -> None:
    """Record one generation of a fresh generator in ``mode``, replay it into poisoned
    outputs (== ``ref``), reseed to ``base2`` and replay again (== ``ref2``)."""
    label = f"{unit.label}/{mode}"
    gen, rng = unit.make(base)
    out = gen.program.result
    if mode in ("warp", "warp-unrolled"):
        s = _side_stream()
        wp.synchronize()
        scope = _unconditional() if mode == "warp-unrolled" else contextlib.nullcontext()
        with scope, _probe_loops() as recs, _recording(s) as cap:
            assert gen.generate(stream=s) is out

        def replay():
            _replay(cap.graph, s)
    else:
        torch = _torch_cuda()
        from track_gen.interop import torch as tg_torch

        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with tg_capture.mempool_disabled(DEV), _probe_loops() as recs:
            with torch.cuda.graph(g):
                with tg_capture.capture_launch_forbidden():
                    if mode == "torch":
                        assert gen.generate() is out     # adopts torch's capture stream
                    else:
                        # conditional=True: the generation must be the LAST op of the
                        # torch graph (capture_if renews the CUDA capture id).
                        with tg_torch.registered(conditional=True) as ws:
                            assert isinstance(ws, wp.Stream) and ws.is_capturing
                            assert gen.generate() is out
        torch.cuda.synchronize()
        tg_torch.retain(g, gen)

        def replay():
            g.replay()
            torch.cuda.synchronize()

    assert gen.graph is None, f"{label}: recording into the caller's graph self-captured"
    rec = _record(recs, unit.stage_id, label)
    unit.check_schedule(rec, label)
    if mode in ("warp", "torch-registered"):
        assert rec.lowering == "conditional" and rec.ran is None, \
            f"{label}: expected one conditional node per window, got {rec.lowering!r}"
    else:
        assert rec.lowering == "unroll" and rec.ran == rec.windows, \
            f"{label}: expected the unrolled lowering, got {rec.lowering!r}"

    _poison_all(out)
    replay()
    _assert_host(_snapshot(out), ref, f"{label}: replay vs eager (c)")
    _reseed(rng, base2)
    _poison_all(out)
    replay()
    _assert_host(_snapshot(out), ref2, f"{label}: replay after a reseed vs fresh eager twin")


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("unit_name", _UNITS)
def test_lowerings_are_bitwise_identical(unit_name, mode):
    """Ordinary Repulsive (small/default), the converging Repulsive workload and the
    converging toy, E=8: the eager reference (c) equals

    - ``poll``: an eager run under the forced host poll (a): at least one window
      skipped for converging workloads, at most one stream synchronize per window check;
    - ``warp``: a Warp ``ScopedCapture`` replay (b) (one conditional node per window);
    - ``warp-unrolled``: the same recording with ``conditional_ok`` False (c);
    - ``torch``: a plain ``torch.cuda.graph`` replay (c) (unregistered);
    - ``torch-registered``: ``interop.torch.registered(conditional=True)`` (b);

    and every recorded graph, reseeded, equals a fresh eager twin of the new seeds.
    For each converging case, a forced-poll preflight must actually skip windows,
    establishing that conditional skipping is nonvacuous. Unrolled captures and
    plain torch deliberately execute every window; ordinary workloads retain their
    full-budget parity coverage."""
    if mode in ("warp", "torch-registered") and not _conditional_supported():
        pytest.skip(_NO_CONDITIONAL)
    unit = _unit(unit_name)
    base = 11
    if unit.must_skip:
        _assert_early_exit(unit, base)
    ref = _eager(unit.make(base)[0])
    if mode == "poll":
        gen, _ = unit.make(base)
        got, rec, n_syncs = _poll(gen, unit, f"{unit_name}/poll")
        unit.check_schedule(rec, f"{unit_name}/poll")
        assert rec.ran <= rec.windows
        if unit.must_skip:
            assert rec.ran < rec.windows, f"{unit_name}: (a) skipped no window (vacuous)"
        assert 1 <= n_syncs <= rec.windows, \
            f"{unit_name}: {n_syncs} stream syncs for {rec.windows} window checks"
        _assert_host(got, ref, f"{unit_name}: (a) poll vs (c) eager")
        return
    base2 = base + 7919
    ref2 = _eager(unit.make(base2)[0])
    assert _differs(ref, ref2), "reseeding must change the batch (vacuous test otherwise)"
    _record_and_replay(unit, base, mode, ref, ref2, base2)


def test_non_absorbing_toy_lowerings_differ():
    """``toy.leaky_smooth`` (same schedule and ``active`` refresh as the converging toy,
    but a converged env keeps smoothing) is NOT absorbing: the forced poll (a) stops early
    and its batch differs from the full-bound eager run (c). So the equality of the
    previous test is a real check, not something any loop satisfies."""
    unit = _unit(LEAKY)
    gen, _ = unit.make(11)
    got, rec, _ = _poll(gen, unit, "leaky/poll")
    assert rec.ran < rec.windows, "leaky toy: (a) skipped no window (vacuous)"
    ref = _eager(unit.make(11)[0])
    assert _differs(got, ref), "a non-absorbing loop gave the same batch in (a) and (c)"


# ---------------------------------------------------------------------------------------
# 2. Repulsive under caller captures and as a self-graph
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("config", ["small", "default"])
def test_repulsive_records_into_caller_captures_and_reseeds(config):
    """``assert_caller_capturable``: repulsive recorded into a Warp capture (windows as
    conditional nodes), a registered torch capture and a plain torch capture (unrolled),
    each replay equal to the eager twin, before and after a reseed."""
    kw = _SMALL if config == "small" else {}
    gen, _ = _track("repulsive", 11, **kw)
    status = tg_capture.assert_caller_capturable(gen)
    assert status["warp"] == "ok", status
    for mode in ("external", "torch"):
        assert status[mode] == "ok" or status[mode].startswith("skipped"), status
    assert gen.graph is None


@pytest.mark.parametrize("config", ["small", "default"])
def test_repulsive_self_graph_matches_eager_twins(config):
    """The producer's own graph (Warp-registered private stream: (b) where conditional
    nodes are supported) is captured once and every replay, on any stream and after
    reseeds, equals a fresh eager twin (c)."""
    kw = _SMALL if config == "small" else {}
    gen, rng = _track("repulsive", 11, **kw)
    with _probe_loops() as recs:
        out = gen.generate()
    g = gen.graph
    assert g is not None
    rec = _record(recs, "generator", f"{config}: self-graph capture")
    want = "conditional" if _conditional_supported() else "unroll"
    assert rec.lowering == want, f"{config}: self-graph lowered as {rec.lowering!r}"
    _assert_host(_snapshot(out), _eager(_track("repulsive", 11, **kw)[0]), f"{config}: first")
    s = _side_stream()
    for i, base in enumerate((101, 202, 303)):
        ref = _eager(_track("repulsive", base, **kw)[0])
        _reseed(rng, base)
        _poison_all(out)
        if i == 0:
            gen.generate()
        elif i == 1:
            gen.generate(stream=s, sync=False)
            wp.synchronize()
        else:
            gen.generate(stream=s)
        assert gen.graph is g, "the self-graph is captured once, then replayed"
        _assert_host(_snapshot(out), ref, f"{config}: self-graph replay {i}")


# ---------------------------------------------------------------------------------------
# 3. Node counts of the default Program's self-graph
# ---------------------------------------------------------------------------------------

_CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
_LIBCUDA: list = []


def _libcuda():
    """The CUDA driver library (already loaded by Warp), or ``None``."""
    if not _LIBCUDA:
        names = ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
        lib = None
        for name in names:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        _LIBCUDA.append(lib)
    return _LIBCUDA[0]


def _graph_nodes(graph) -> "tuple[int, int] | None":
    """``(top-level nodes, conditional nodes)`` of a captured ``wp.Graph`` through the
    driver API (``cuGraphGetNodes`` / ``cuGraphNodeGetType`` on ``graph.graph``, the
    ``CUgraph`` Warp keeps after instantiation); ``None`` if unavailable. A conditional
    node counts as one node (its body graph is not walked)."""
    handle = getattr(graph, "graph", None)
    ptr = getattr(handle, "value", handle)
    lib = _libcuda()
    if not ptr or lib is None:
        return None
    get_nodes = getattr(lib, "cuGraphGetNodes", None)
    get_type = getattr(lib, "cuGraphNodeGetType", None)
    if get_nodes is None or get_type is None:
        return None
    get_nodes.restype = ctypes.c_int
    get_nodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    get_type.restype = ctypes.c_int
    get_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    n = ctypes.c_size_t(0)
    if get_nodes(ctypes.c_void_p(ptr), None, ctypes.byref(n)) != 0:
        return None
    nodes = (ctypes.c_void_p * max(int(n.value), 1))()
    if get_nodes(ctypes.c_void_p(ptr), nodes, ctypes.byref(n)) != 0:
        return None
    conditional = 0
    for i in range(int(n.value)):
        t = ctypes.c_int(-1)
        if get_type(nodes[i], ctypes.byref(t)) != 0:
            return None
        conditional += int(t.value == _CU_GRAPH_NODE_TYPE_CONDITIONAL)
    return int(n.value), conditional


def test_repulsive_self_graph_node_counts():
    """Default config, E=8: the unrolled self-graph (``conditional_ok`` False before the
    first ``generate()``) stays within its fixed-search node budget and has no
    conditional node; the windowed one
    (lowering (b)) has exactly one conditional node per window (6) and far fewer
    top-level nodes. Both replay the eager batch."""
    ref = _eager(_track("repulsive", 11)[0])
    with _unconditional():
        unrolled, _ = _track("repulsive", 11)
        out = unrolled.generate()
    _assert_host(_snapshot(out), ref, "unrolled self-graph")
    counts_u = _graph_nodes(unrolled.graph)
    if counts_u is None:
        pytest.skip("cannot count graph nodes: the CUDA driver library or "
                    "cuGraphGetNodes / cuGraphNodeGetType is unavailable through ctypes")
    total_u, cond_u = counts_u
    assert cond_u == 0, f"unrolled self-graph holds {cond_u} conditional node(s)"
    # The parallel safety search deliberately records every bounded trial:
    # shape 4 + 24*7 + 1, growth 3 + 16*3 + 1, and 16 other core launches give
    # about 241*189 = 45,549 nodes before periodic/tail work (45,830 measured).
    # Keep a tight ceiling above that fixed schedule; convergence cannot shrink
    # an unrolled graph. The conditional graph still has to reduce it below 60%.
    node_budget = 48_000
    assert total_u < node_budget, \
        f"unrolled self-graph has {total_u} nodes (budget {node_budget})"
    if not _conditional_supported():
        pytest.skip(_NO_CONDITIONAL)
    windowed, _ = _track("repulsive", 11)
    out = windowed.generate()
    _assert_host(_snapshot(out), ref, "windowed self-graph")
    total_w, cond_w = _graph_nodes(windowed.graph)
    assert cond_w == _REPULSIVE_SCHEDULE["windows"], \
        f"windowed self-graph: {cond_w} conditional nodes, expected one per window"
    assert total_w < 0.6 * total_u, \
        f"windowed self-graph has {total_w} top-level nodes vs {total_u} unrolled"


# ---------------------------------------------------------------------------------------
# 4. Batch independence on CUDA (env e identical at E=8 and inside E=64)
# ---------------------------------------------------------------------------------------

def _frozen(gen) -> np.ndarray:
    """Per env: whether repulsive froze it (its ``frozen`` scratch, after the run; the
    generator Stage's private scratch, as tests/test_generate_repulsive.py reads it)."""
    scratch = gen._scratch.gen.scratch
    names = [k for k in scratch if str(k).rsplit("/", 1)[-1] == "frozen"]
    assert len(names) == 1, sorted(scratch)
    return scratch[names[0]].numpy().astype(bool)


def _three_lowerings(make, label: str) -> dict:
    """``{lowering: (snapshot, probe record)}`` for eager (c), forced poll (a) and the
    self-graph (b where supported), each on a fresh generator of ``make()``."""
    def state(gen, result):
        return {**_snapshot(result), "solver.frozen": _frozen(gen)}

    out = {}
    gen, _ = make()
    _poison_all(gen.program.result)
    with _probe_loops() as recs:
        res = gen.enqueue()
    wp.synchronize()
    out["eager"] = (state(gen, res), _record(recs, "generator", f"{label}/eager"))
    gen, _ = make()
    _poison_all(gen.program.result)
    with _force_lowering("poll"), _probe_loops() as recs:
        res = gen.enqueue()
    wp.synchronize()
    out["poll"] = (state(gen, res), _record(recs, "generator", f"{label}/poll"))
    gen, _ = make()
    _poison_all(gen.program.result)
    with _probe_loops() as recs:
        res = gen.generate()
    assert gen.graph is not None
    out["self-graph"] = (state(gen, res), _record(recs, "generator", f"{label}/self-graph"))
    return out


@pytest.mark.parametrize("workload", ["default", "converging"])
def test_repulsive_env_is_independent_of_the_batch_in_every_lowering(workload):
    """Fixed scrambled rows from E=64, rerun at E=8, have identical per-env results
    under all three lowerings. Real defaults cover the full workload independently
    of convergence. In the converging workload an infeasible final E=64 row keeps
    that batch active after its selected E=8 subset has frozen, without affecting
    the selected rows' results."""
    kw = dict(_CONVERGING_REPULSIVE, half_width=0.2) if workload == "converging" else {}
    big_seeds = 11 + np.arange(64, dtype=np.int32)
    rows = np.asarray([7, 0, 5, 2, 6, 1, 4, 3])

    def make_big():
        gen, rng = _track("repulsive", big_seeds, n=64, **kw)
        if workload == "converging":
            # A NaN world scale can still leave the unscaled solver converged.
            # Instead use a finite impossible width reservation: .2/.1 = 2
            # exceeds the domain radius 1.716. Only this unselected row changes;
            # both batches otherwise have the same supported near-circle problem.
            sizes = np.ones(64, np.float32)
            sizes[-1] = 0.1
            gen.params["generator"].set("size", wp.array(sizes, dtype=wp.float32,
                                                         device=DEV))
        return gen, rng

    big = _three_lowerings(make_big, "E=64")
    small = _three_lowerings(lambda: _track("repulsive", big_seeds[rows], **kw), "E=8")
    ref = _rows(big["eager"][0], 64)
    for name, (snap, _) in big.items():
        _assert_host(snap, big["eager"][0], f"E=64 {name} vs eager")
    for name, (snap, _) in small.items():
        got = _rows(snap)
        for i, r in enumerate(rows):
            for k in ref:
                np.testing.assert_array_equal(
                    got[k][i], ref[k][r],
                    err_msg=f"E=8 {name}: env {i} (seed {big_seeds[r]}) != E=64 env {r}: {k}")
    if workload == "converging":
        for name, (snap, _) in big.items():
            assert snap["valid"][-1] == 0, f"{name}: the infeasible row was marked valid"
            assert snap["fail_mask"][-1] != 0, f"{name}: the invalid row has no failure bit"
            assert not snap["solver.frozen"][-1], f"{name}: the infeasible row froze"
        assert np.all(small["eager"][0]["valid"]), "the selected E=8 rows must be valid"
        assert np.all(small["eager"][0]["solver.frozen"]), \
            "the fixed E=8 convergence batch did not freeze"
        rec_big, rec_small = big["poll"][1], small["poll"][1]
        assert rec_small.ran < rec_small.windows, "the all-frozen E=8 batch skipped no window"
        assert rec_big.ran == rec_big.windows, "the active infeasible row skipped a window"
        assert rec_big.ran > rec_small.ran, "the two batches must stop at different windows"


# ---------------------------------------------------------------------------------------
# 5-8. Segmented HOST execution
# ---------------------------------------------------------------------------------------

def _counted(kind: str, base_name: str, calls: list) -> str:
    """Register a subclass of the registered Stage ``base_name`` whose enqueue appends
    ``ctx.capturing`` to ``calls`` (the Python body runs), then runs the toy's body."""
    base = registry.get(kind, base_name)
    counted_name = f"{base_name}_counted"

    class Counted(base):
        name = counted_name

        def build(self, b, p):
            body = super().build(b, p)

            def enqueue(ctx):
                calls.append(bool(ctx.capturing))
                body(ctx)
            return enqueue

    registry.add(Counted)
    return counted_name


def _assert_segmented(program, label: str) -> "wp.Graph":
    """A [HOST generator] + [captured tail] Program after a self-graph call: no plain
    graph, a graph at the tail segment only. Returns the tail graph."""
    assert program.capture_class is CaptureClass.HOST, label
    assert program.host_stage_ids == ("generator",), (label, program.host_stage_ids)
    assert not program.capturable, label
    hosts = [seg.host for seg in program.segments]
    assert hosts == [True, False], f"{label}: segments {program.segments}"
    assert program.graph is None, f"{label}: a segmented Program has no single graph"
    graphs = program.segment_graphs()
    assert len(graphs) == 2 and graphs[0] is None and isinstance(graphs[1], wp.Graph), \
        f"{label}: segment graphs {graphs}"
    return graphs[1]


def test_host_generator_runs_eagerly_and_its_tail_from_a_graph():
    """``toy.host_circle`` (HOST: host RNG uploaded through ``ctx.host()``) under
    ``generate()``: each call runs the HOST body once, eagerly (not capturing), and
    launches exactly ONE graph (the captured tail, captured on the first call and then
    reused); ``gen.graph`` stays None; every call (default stream, side stream, sync or
    not) equals the eager ``enqueue()`` twin, and after a reseed a fresh twin."""
    calls: list = []
    name = _counted("generator", HOST_CIRCLE, calls)
    gen, rng = _track(name, 11)
    assert calls == [False], f"construction runs one eager pass, got {calls}"
    ref = _eager(_track(HOST_CIRCLE, 11)[0])
    out = gen.program.result
    s = _side_stream()
    tail = None
    with _counting_capture_launch() as launches:
        for k in range(4):
            _poison_all(out)
            if k < 2:
                assert gen.generate() is out
            elif k == 2:
                assert gen.generate(stream=s, sync=False) is out
                wp.synchronize()
            else:
                assert gen.generate(stream=s) is out
            assert calls == [False] * (k + 2), f"call {k}: body runs {calls}"
            assert len(launches) == k + 1, f"call {k}: {len(launches)} graph launches"
            g = _assert_segmented(gen.program, f"call {k}")
            assert tail is None or g is tail, "the tail graph is captured once"
            tail = g
            assert launches[-1] is tail
            assert gen.graph is None
            _assert_host(_snapshot(out), ref, f"segmented call {k} vs eager twin")
    for base in (101, 202):
        want = _eager(_track(HOST_CIRCLE, base)[0])
        assert _differs(want, ref), "reseeding must change the batch"
        _reseed(rng, base)
        _poison_all(out)
        gen.generate()
        _assert_host(_snapshot(out), want, f"segmented call after reseed {base}")
    assert gen.program.segment_graphs()[1] is tail


def test_host_generator_masked_regeneration_is_exact():
    """``generate(advance=True, mask=inputs["regen_mask"])`` on a segmented HOST Program:
    the HOST head (prologue: advance, effective seeds, latch, hold; then the generator)
    runs eagerly, the tail (stages + commit) from ONE graph cached for the mask key and
    reused by later masked calls. Masked envs equal an eager twin advanced the same way;
    unmasked envs keep their (poisoned) rows bit for bit (the commit); epochs and
    ``generation`` follow the masks."""
    calls: list = []
    name = _counted("generator", HOST_CIRCLE, calls)
    gen, _ = _track(name, 11)
    twin, _ = _track(HOST_CIRCLE, 11)
    m, m_t = gen.inputs["regen_mask"], twin.inputs["regen_mask"]
    out = gen.program.result
    key = (True, int(m.ptr))
    epoch = np.zeros(E, np.int32)
    first = None
    with _counting_capture_launch() as launches:
        for k, mk in enumerate((_M1, _M1, 1 - _M1, _ZERO)):
            label = f"masked call {k} mask {mk.tolist()}"
            for dst in (m, m_t):
                dst.assign(mk)
            wp.synchronize()
            _poison_all_but_generation(out)
            poisoned = _rows(_snapshot(out))
            assert gen.generate(advance=True, mask=m) is out
            ref = _rows(_eager(twin, advance=True, mask=m_t))
            epoch += mk
            cur = _rows(_snapshot(out))
            for e in range(E):
                exp = ref if mk[e] else poisoned
                for f, rows in cur.items():
                    np.testing.assert_array_equal(
                        rows[e], exp[f][e],
                        err_msg=f"{label}: env {e} ({'masked' if mk[e] else 'held'}): {f}")
            np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch, err_msg=label)
            np.testing.assert_array_equal(out.generation.numpy(), epoch, err_msg=label)
            assert len(calls) == k + 2 and len(launches) == k + 1, (label, calls, launches)
            graphs = gen.program.segment_graphs(key)
            assert graphs[0] is None and isinstance(graphs[1], wp.Graph), (label, graphs)
            if first is None:
                first = graphs[1]
            assert graphs[1] is first, f"{label}: the masked tail graph was captured again"
    assert gen.graph is None


def _assert_refused(exc_info, label: str) -> None:
    exc = exc_info.value
    assert exc.stage_ids == ("generator",), f"{label}: stage_ids {exc.stage_ids}"
    assert "'generator'" in str(exc), f"{label}: {exc}"


@pytest.mark.parametrize("mode", ["warp", "torch"])
def test_host_program_refuses_caller_capture_before_any_launch(mode):
    """Under a caller's capture (Warp ``ScopedCapture`` on a side stream, or a plain
    ``torch.cuda.graph``), every entry point of a HOST Program raises
    ``NotCapturableError`` naming the HOST stage ids BEFORE any launch: the caller's
    capture stays valid and its replay runs only the caller's own work (marker written,
    poisoned outputs, seeds and epochs untouched); the HOST body never ran."""
    calls: list = []
    name = _counted("generator", HOST_CIRCLE, calls)
    gen, _ = _track(name, 11)
    gen.generate()                       # the tail graph exists: never launch it here
    prog = gen.program
    out = prog.result
    n = len(calls)
    _poison_all(out)
    poisoned = _snapshot(out)
    seeds0 = gen.seeds.seeds.numpy().copy()
    epoch0 = gen.seeds.epoch.numpy().copy()
    m = gen.inputs["regen_mask"]

    def refuse_all(stream, ctx):
        kw = {} if stream is None else {"stream": stream}
        for label, call in (
                ("generate", lambda: gen.generate(**kw)),
                ("generate(advance, mask)", lambda: gen.generate(advance=True, mask=m, **kw)),
                ("enqueue", lambda: gen.enqueue(**kw)),
                ("program.run(advance)", lambda: prog.run(advance=True, **kw)),
                ("program.enqueue", lambda: prog.enqueue(ctx)),
                ("program.enqueue_segment(0)", lambda: prog.enqueue_segment(0, ctx))):
            with pytest.raises(tg_exec.NotCapturableError, match=name) as ei:
                call()
            _assert_refused(ei, f"{mode}: {label}")

    if mode == "warp":
        s = _side_stream()
        marker, marker_src = _marker()
        with _recording(s) as cap:
            wp.copy(marker, marker_src, stream=s)
            refuse_all(s, tg_exec.LaunchContext(DEV, s))
        _replay(cap.graph, s)
        np.testing.assert_array_equal(marker.numpy(), marker_src.numpy())
    else:
        torch = _torch_cuda()
        t_src = torch.arange(3, device=DEV, dtype=torch.int32) + 1
        t_dst = torch.zeros(3, device=DEV, dtype=torch.int32)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with tg_capture.mempool_disabled(DEV):
            with torch.cuda.graph(g):
                t_dst.copy_(t_src)
                with tg_capture.capture_launch_forbidden():
                    ctx = tg_exec.LaunchContext.for_call(DEV)     # adopts torch's stream
                    assert ctx.capturing
                    refuse_all(None, ctx)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        assert t_dst.tolist() == [1, 2, 3]
    assert len(calls) == n, "the HOST body ran under the caller's capture"
    _assert_host(_snapshot(out), poisoned, f"{mode}: outputs written by the caller's graph")
    np.testing.assert_array_equal(gen.seeds.seeds.numpy(), seeds0)
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch0)
    assert gen.graph is None


def test_unprepared_host_program_refuses_with_not_capturable_error():
    """A HOST Program built with ``prepare=False``, called under a caller's capture: the
    HOST refusal comes FIRST (``NotCapturableError`` naming the stage, not the plain
    "not prepared" ``CaptureError``) for ``run`` / ``enqueue`` / ``enqueue_segment(0)``;
    the non-HOST tail segment alone gets the "not prepared" ``CaptureError``. Nothing is
    recorded or prepared; outside the capture ``run()`` prepares and equals the eager
    twin."""
    spec = presets.track(num_envs=E, device=DEV, generator=HOST_CIRCLE)
    prog = build_program(spec, seeds=_rng(11), prepare=False)
    wp.synchronize()
    assert not prog.runner.prepared
    assert [seg.host for seg in prog.segments] == [True, False]
    out = prog.result
    _poison_all(out)
    poisoned = _snapshot(out)
    s = _side_stream()
    ctx = tg_exec.LaunchContext(DEV, s)
    marker, marker_src = _marker()
    with _recording(s) as cap:
        wp.copy(marker, marker_src, stream=s)
        for label, call in (("run", lambda: prog.run(stream=s)),
                            ("run(advance)", lambda: prog.run(stream=s, advance=True)),
                            ("enqueue", lambda: prog.enqueue(ctx)),
                            ("enqueue_segment(0)", lambda: prog.enqueue_segment(0, ctx))):
            with pytest.raises(tg_exec.NotCapturableError) as ei:
                call()
            _assert_refused(ei, f"unprepared: {label}")
        with pytest.raises(tg_exec.CaptureError) as ei:
            prog.enqueue_segment(1, ctx)
        assert not isinstance(ei.value, tg_exec.NotCapturableError), ei.value
    assert not prog.runner.prepared
    _replay(cap.graph, s)
    np.testing.assert_array_equal(marker.numpy(), marker_src.numpy())
    _assert_host(_snapshot(out), poisoned, "unprepared: outputs written by the caller's graph")
    prog.run(sync=True)
    assert prog.runner.prepared
    _assert_host(_snapshot(out), _eager(_track(HOST_CIRCLE, 11)[0]), "unprepared: first run")


def test_caller_records_the_tail_segment_and_runs_the_host_head_eagerly():
    """The ``enqueue_segment`` contract: once prepared, the non-HOST tail of a HOST
    Program records into a caller's capture (the HOST body does not run, no prologue is
    recorded), and the caller drives a generation by running the HOST head eagerly
    (``enqueue_segment(0, ctx)``: prologue + host generator) then replaying the tail
    graph, on the same stream: each such pair equals a fresh eager twin of the current
    seeds (the tail reads what the head just wrote)."""
    calls: list = []
    name = _counted("generator", HOST_CIRCLE, calls)
    gen, rng = _track(name, 11)
    prog = gen.program
    out = prog.result
    s = _side_stream()
    ctx = tg_exec.LaunchContext(DEV, s)
    wp.synchronize()
    tail = [i for i, seg in enumerate(prog.segments) if not seg.host]
    assert tail == [1], prog.segments
    with _recording(s) as cap:
        for i in tail:
            prog.enqueue_segment(i, ctx)
    assert calls == [False], "recording the tail ran the HOST body"
    for k, base in enumerate((101, 202)):
        want = _eager(_track(HOST_CIRCLE, base)[0])
        _reseed(rng, base)
        _poison_all(out)
        prog.enqueue_segment(0, ctx)               # eager HOST head on s
        assert calls == [False] * (k + 2)
        _replay(cap.graph, s)                      # the caller's tail graph, also on s
        _assert_host(_snapshot(out), want, f"head eager + tail replay, seeds {base}")


@wp.kernel
def _copy_seeds_k(src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int32)):
    e = wp.tid()
    dst[e] = src[e]






def test_host_gate_generator_is_segmented_and_matches_its_eager_twin():
    """``toy.host_ring`` (a HOST gate generator writing anchors through ``ctx.host()``):
    the gate pipeline is [HOST generator] + [captured tail] too, its self-graph calls
    equal the eager ``enqueue()`` twin, and after a reseed a fresh twin."""
    gen, rng = _gates(HOST_RING, 11)
    ref = _eager(_gates(HOST_RING, 11)[0])
    out = gen.program.result
    tail = None
    for k in range(2):
        _poison_all(out)
        assert gen.generate() is out
        g = _assert_segmented(gen.program, f"gate call {k}")
        assert tail is None or g is tail
        tail = g
        _assert_host(_snapshot(out), ref, f"gate segmented call {k}")
    want = _eager(_gates(HOST_RING, 101)[0])
    assert _differs(want, ref)
    _reseed(rng, 101)
    _poison_all(out)
    gen.generate()
    _assert_host(_snapshot(out), want, "gate segmented call after reseed")


# ---------------------------------------------------------------------------------------
# 9. Nested conditionals: bounded windows inside a skip_unless body
# ---------------------------------------------------------------------------------------

@wp.kernel
def _decay_k(x: wp.array(dtype=wp.float32), left: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if left[e] > 0:
        x[e] = x[e] * 0.75 + 1.0
        left[e] = left[e] - 1


@wp.kernel
def _any_left_k(left: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if left[e] > 0:
        wp.atomic_max(active, 0, 1)


@wp.kernel
def _tick_k(ticks: wp.array(dtype=wp.int32)):
    ticks[0] = ticks[0] + 1


class _Countdown:
    """A direct ``ctx.bounded_loop`` (no Program slot): env ``e`` decays ``x[e]`` for
    ``left0[e]`` iterations, then freezes (absorbing for ``x`` / ``left``); ``active`` =
    any env left, refreshed on the iteration before each window. ``ticks`` counts the
    iterations that EXECUTED: a witness of on-device window skipping, deliberately
    outside the absorbing contract (compared separately)."""

    N, F, W = 40, 8, 8
    LEFT = np.array([3, 9, 17, 0, 12, 5, 20, 1], np.int32)

    def __init__(self) -> None:
        self.x0 = wp.array(np.linspace(-1.0, 1.0, E).astype(np.float32), dtype=wp.float32,
                           device=DEV)
        self.left0 = wp.array(self.LEFT, dtype=wp.int32, device=DEV)
        self.x = wp.zeros(E, dtype=wp.float32, device=DEV)
        self.left = wp.zeros(E, dtype=wp.int32, device=DEV)
        self.active = wp.zeros(1, dtype=wp.int32, device=DEV)
        self.ticks = wp.zeros(1, dtype=wp.int32, device=DEV)
        wp.synchronize()

    @classmethod
    def executed_when_skipping(cls) -> int:
        """Iterations run when windows are skipped on device: up to the first window
        start ``lo`` with no env left after ``lo`` iterations."""
        for lo in range(cls.F, cls.N, cls.W):
            if int(cls.LEFT.max()) <= lo:
                return lo
        return cls.N

    def body(self, c, it: int) -> None:
        c.launch(_decay_k, E, [self.x, self.left])
        c.launch(_tick_k, 1, [self.ticks])
        if (it + 1) % self.W == 0:
            c.fill(self.active, 0)
            c.launch(_any_left_k, E, [self.left, self.active])

    def enqueue(self, c) -> None:
        c.copy(self.x, self.x0)
        c.copy(self.left, self.left0)
        c.fill(self.ticks, 0)
        c.fill(self.active, 1)
        c.bounded_loop(self.N, self.body, active=self.active, first_check=self.F,
                       check_every=self.W)

    def state(self) -> dict:
        return {"x": self.x.numpy().copy(), "left": self.left.numpy().copy()}

    def poison(self) -> None:
        for a in (self.x, self.left, self.ticks):
            tg_capture.poison(a)
        wp.synchronize()


@pytest.mark.parametrize("nested", [False, True], ids=["plain", "nested"])
def test_bounded_windows_skip_on_device_also_inside_a_conditional_body(nested):
    """A bounded loop recorded in a Warp capture skips its converged windows ON DEVICE
    (the witness counter shows 24 of 40 iterations executed) and ends in exactly the
    eager state (every iteration executed); with ``conditional_ok`` False it records
    every window (40). Nested inside ``ctx.skip_unless(flag, ...)`` (conditional nodes
    inside a conditional body, CUDA >= 12.4): flag 1 replays the same, flag 0 skips the
    whole body (poison stays)."""
    if not _conditional_supported():
        pytest.skip(_NO_CONDITIONAL)
    cd = _Countdown()
    s = _side_stream()
    ctx = tg_exec.LaunchContext(DEV, s)
    flag = wp.ones(1, dtype=wp.int32, device=DEV)
    wp.synchronize()
    with _probe_loops() as recs:
        cd.enqueue(ctx)                                   # eager on CUDA: (c), loads kernels
    wp.synchronize()
    assert [r.lowering for r in recs] == ["unroll"]
    ref = cd.state()
    assert int(cd.ticks.numpy()[0]) == cd.N
    assert np.all(ref["left"] == 0)
    skipped_run = cd.executed_when_skipping()
    assert skipped_run < cd.N, "the countdown must converge before the bound (vacuous)"

    def record(conditional: bool):
        with _probe_loops() as recs, _recording(s) as cap:
            if nested:
                assert ctx.skip_unless(flag, cd.enqueue) is conditional
            else:
                cd.enqueue(ctx)
        return cap.graph, recs

    graph, recs = record(True)
    assert [(r.stage_id, r.lowering) for r in recs] == [(None, "conditional")], recs
    cd.poison()
    _replay(graph, s)
    _assert_host(cd.state(), ref, "windowed replay vs eager")
    assert int(cd.ticks.numpy()[0]) == skipped_run, "converged windows were not skipped"

    with _unconditional():
        graph_u, recs_u = record(False)
    assert [r.lowering for r in recs_u] == ["unroll"], recs_u
    cd.poison()
    _replay(graph_u, s)
    _assert_host(cd.state(), ref, "unrolled replay vs eager")
    if not nested:
        assert int(cd.ticks.numpy()[0]) == cd.N
        return
    # nested, conditional_ok False: the outer skip is unconditional too (body runs)
    assert int(cd.ticks.numpy()[0]) == cd.N
    flag.fill_(0)
    wp.synchronize()
    cd.poison()
    poisoned = {**cd.state(), "ticks": cd.ticks.numpy().copy()}
    _replay(graph, s)
    _assert_host({**cd.state(), "ticks": cd.ticks.numpy().copy()}, poisoned,
                 "flag 0: the outer conditional must skip the whole body")


def test_course_masked_regeneration_nests_repulsive_windows():
    """``Course.regenerate(mask=m, advance=True)`` over repulsive (small config) recorded
    in a Warp capture: the course's ``skip_unless`` wraps the generation, and the
    repulsive windows are recorded as conditional nodes INSIDE that conditional body. A
    zero-mask replay skips everything (the poisoned result stays); ``mask[0] = 1``
    regenerates env 0 exactly as an eager twin, and the unmasked envs keep their rows."""
    if not _conditional_supported():
        pytest.skip(_NO_CONDITIONAL)

    def cfg():
        return CourseConfig(mode="track",
                            gen=TrackGenConfig(generator="repulsive", num_envs=E, device=DEV,
                                               **_SMALL),
                            seeds=5, collision="segments", checkpoint_spacing=0.6,
                            max_checkpoints=64)

    course, twin = Course(cfg()), Course(cfg())
    m, m_t = course.inputs["regen_mask"], twin.inputs["regen_mask"]
    s = _side_stream()
    assert course.regenerate(mask=m, advance=True, stream=s) is course.result   # warm-up
    wp.synchronize()
    with _probe_loops() as recs, _recording(s) as cap:
        assert course.regenerate(mask=m, advance=True, stream=s) is course.result
    rec = _record(recs, "generator", "course recording")
    assert rec.lowering == "conditional", f"nested windows lowered as {rec.lowering!r}"
    epoch = course.generator.seeds.epoch

    _poison_all_but_generation(course.result)
    poisoned = _snapshot(course.result)
    _replay(cap.graph, s)
    _assert_host(_snapshot(course.result), poisoned, "zero mask: the body must be skipped")
    np.testing.assert_array_equal(epoch.numpy(), _ZERO)

    mk = _ZERO.copy()
    mk[0] = 1
    for dst in (m, m_t):
        dst.assign(mk)
    wp.synchronize()
    _replay(cap.graph, s)
    twin.enqueue_generate(mask=m_t, advance=True)
    wp.synchronize()
    cur, want, held = (_rows(_snapshot(course.result)), _rows(_snapshot(twin.result)),
                       _rows(poisoned))
    for e in range(E):
        exp = want if mk[e] else held
        for f, rows in cur.items():
            np.testing.assert_array_equal(rows[e], exp[f][e], err_msg=f"env {e}: {f}")
    np.testing.assert_array_equal(epoch.numpy(), mk)


# ---------------------------------------------------------------------------------------
# 10. The conformance kit on CUDA, with the new checks
# ---------------------------------------------------------------------------------------

def test_kit_repulsive_passes_the_bounded_checks_on_cuda():
    """The explicit converging workload passes every kit check on CUDA, with both
    absorbing freeze and lowering parity actually exercised."""
    report = check_generator("repulsive", devices=[DEV], config=_CONVERGING_REPULSIVE,
                             num_envs=E)
    report.raise_for_failures()
    for name in ("bounded.absorbing", "bounded.lowering"):
        assert not report.result(name, DEV).skipped, str(report)
    assert sum(report.result("bounded.absorbing", DEV).details["skipped"].values()) >= 1
    assert report.result("capture.caller", DEV).details.get("warp") == "ok", str(report)


def test_kit_bounded_toys_on_cuda():
    """``check_stage`` on ``cuda:0``: the converging toy passes with both bounded checks
    exercised; the leaky toy fails both of them (a non-absorbing loop is caught on the
    GPU too)."""
    report = check_stage("relax", CONVERGING, devices=[DEV])
    report.raise_for_failures()
    for name in ("bounded.absorbing", "bounded.lowering"):
        assert not report.result(name, DEV).skipped, str(report)
    leaky = check_stage("relax", LEAKY, devices=[DEV])
    assert {"bounded.absorbing", "bounded.lowering"} <= set(leaky.failed_names), str(leaky)


@pytest.mark.parametrize("kind,name", [("generator", HOST_CIRCLE),
                                       ("gate_generator", HOST_RING)])
def test_kit_host_toys_on_cuda(kind, name):
    """The HOST toys pass the kit on ``cuda:0``: ``capture.caller`` now expects the
    refusal before any launch and ``capture.self_graph`` the segmented graphs."""
    check = check_generator if kind == "generator" else check_gate_generator
    report = check(name, devices=[DEV])
    report.raise_for_failures()
    for check_name in ("capture.caller", "capture.self_graph"):
        assert not report.result(check_name, DEV).skipped, str(report)


# ---------------------------------------------------------------------------------------
# 11. HOST work while another capture is open (the recorded deviation of the design)
# ---------------------------------------------------------------------------------------

def _run_in_thread_while(capture_fn, main_fn) -> dict:
    """Run ``capture_fn(recorded, release)`` in a thread (it opens a capture, sets
    ``recorded``, waits for ``release``, ends it); ``main_fn()`` runs in this thread
    while that capture is open. Returns the thread's box (``"error"`` on failure)."""
    recorded, release = threading.Event(), threading.Event()
    box: dict = {}

    def target():
        try:
            capture_fn(recorded, release, box)
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            box["error"] = exc
        finally:
            recorded.set()

    # Free earlier garbage (eager twins) now, and keep the cyclic GC off while the other
    # capture is open: a collected Warp array frees device memory, which is not what this
    # test is about (the HOST call itself allocates nothing).
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    t = threading.Thread(target=target, daemon=True)
    t.start()
    try:
        assert recorded.wait(timeout=120), "the capturing thread never recorded"
        assert "error" not in box, box.get("error")
        main_fn()
    finally:
        release.set()
        t.join(timeout=120)
        if was_enabled:
            gc.enable()
    assert not t.is_alive()
    assert "error" not in box, f"the other capture failed: {box.get('error')!r}"
    return box


@pytest.mark.parametrize("other", ["warp-thread", "torch-global-thread",
                                   "torch-this-thread"])
def test_host_segment_runs_while_another_capture_is_open(other):
    """A HOST Program's eager HOST segment (``generate()``: HOST head + tail replay;
    ``enqueue()``: all eager) runs while (i) another thread holds a Warp capture of a
    capturable producer on its own stream, (ii) another thread holds a
    ``torch.cuda.graph`` capture in torch's default GLOBAL mode, or (iii) this thread is
    inside ``torch.cuda.graph`` and the call resolves an explicit non-capturing stream.
    HostOps only copies through build-time pinned buffers on the call's stream and waits
    in relaxed mode, so the call completes and equals its eager twin, and the other
    capture is not invalidated (its replay equals its own reference). No readback
    happens while the other capture is open."""
    calls: list = []
    name = _counted("generator", HOST_CIRCLE, calls)
    gen, _ = _track(name, 11)
    ref = _eager(_track(HOST_CIRCLE, 11)[0])
    gen.generate()                       # the tail graph is captured (and instantiated)
    wp.synchronize()
    out = gen.program.result
    _poison_all(out)
    n = len(calls)

    if other == "warp-thread":
        gen_b, _ = _track("polar", 11)
        ref_b = _eager(_track("polar", 11)[0])
        s_b = _side_stream()
        wp.synchronize()

        def capture_b(recorded, release, box):
            with wp.ScopedCapture(device=DEV, stream=s_b) as cap:
                box["out"] = gen_b.generate(stream=s_b)
                recorded.set()
                if not release.wait(timeout=120):
                    raise TimeoutError("the main thread never released the capture")
            box["graph"] = cap.graph

        def main():
            assert wp.get_device(DEV).is_capturing, "thread B's capture is registered"
            assert gen.generate() is out
            assert gen.enqueue() is out

        box = _run_in_thread_while(capture_b, main)
        wp.synchronize()
        _poison_all(box["out"])
        _replay(box["graph"], s_b)
        _assert_host(_snapshot(box["out"]), ref_b, "thread B's replay")
    else:
        torch = _torch_cuda()
        x = torch.arange(4096, device=DEV, dtype=torch.float32)
        y = torch.zeros_like(x)
        y.copy_(x * 2.0 + 1.0)            # warm the torch kernels up
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        if other == "torch-global-thread":
            def capture_b(recorded, release, box):
                with torch.cuda.graph(g):         # capture_error_mode="global"
                    y.copy_(x * 2.0 + 1.0)
                    recorded.set()
                    if not release.wait(timeout=120):
                        raise TimeoutError("the main thread never released the capture")

            def main():
                assert gen.generate() is out
                assert gen.enqueue() is out

            _run_in_thread_while(capture_b, main)
        else:
            s_x = _side_stream()
            wp.synchronize()
            gc.collect()
            gc.disable()                          # as in _run_in_thread_while
            try:
                with torch.cuda.graph(g):
                    y.copy_(x * 2.0 + 1.0)
                    assert gen.generate(stream=s_x) is out     # not capturing: runs now
                    assert gen.enqueue(stream=s_x) is out
            finally:
                gc.enable()
        torch.cuda.synchronize()
        y.zero_()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x * 2.0 + 1.0)
    wp.synchronize()
    assert calls == [False] * (n + 2), f"{other}: HOST body runs {calls[n:]}"
    _assert_host(_snapshot(out), ref, f"{other}: HOST call during the other capture")
