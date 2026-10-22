"""Record a producer into a caller's CUDA graph and check the replays (CUDA only).

:func:`assert_caller_capturable` is the GPU half of the hot-path contract: it records a
producer's generation (``producer.generate(stream=s)`` by default) into a capture the
CALLER owns, then replays that graph and compares every output with an eager twin. The
recording uses the discipline TrackGen's own capture tests use:

- a side stream that is not Warp's current stream (so work that ignores the call's
  stream escapes the capture and is caught);
- the Warp mempool disabled on the device (Warp's default CUDA allocator cannot
  allocate while a capture is active, so an allocation fails loudly);
- ``wp.capture_launch`` replaced by a tripwire (a graph launch cannot be recorded into
  another capture, CUDA error 900);
- outputs poisoned with FINITE, dtype-aware sentinels before each replay (outputs are
  NaN-padded and NaN compares equal in the checks, so a NaN poison would hide a missed
  write);
- a reseed between replays: the graph must read the live seeds, so the replay after a
  reseed must equal a fresh twin built with the new seeds.

Capture modes (``modes=``):

- ``"warp"``: a Warp ``wp.ScopedCapture`` on the side stream;
- ``"external"``: a ``torch.cuda.graph`` capture registered with Warp through
  :func:`track_gen.interop.torch.registered` (``ScopedCapture(external=True)``);
- ``"torch"``: a plain ``torch.cuda.graph`` capture; the producer is called without
  ``stream=`` and must adopt torch's capture stream.

The torch modes import torch inside the mode only, and are reported as skipped when
torch (with CUDA) is not available. The helpers (:func:`side_stream`,
:func:`mempool_disabled`, :func:`capture_launch_forbidden`, :func:`recording`,
:func:`poison`, :func:`poison_all`, :func:`result_arrays`) are public building blocks
for custom capture tests.
"""
from __future__ import annotations

import contextlib
import dataclasses
from typing import Callable, Iterable, Iterator

import numpy as np
import warp as wp

__all__ = [
    "CAPTURE_MODES",
    "assert_caller_capturable",
    "capture_launch_forbidden",
    "mempool_disabled",
    "poison",
    "poison_all",
    "recording",
    "result_arrays",
    "side_stream",
]

#: Every capture mode :func:`assert_caller_capturable` knows, in run order.
CAPTURE_MODES = ("warp", "external", "torch")

_INT_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
_FLOAT_DTYPES = ("float16", "float32", "float64")


def result_arrays(obj) -> dict:
    """``name -> wp.array`` for every array field of a result dataclass (``Track``,
    ``GateSequence``, ...); ``None`` fields are skipped. A plain mapping of arrays is
    returned filtered the same way."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if isinstance(v, wp.array)}
    if isinstance(obj, wp.array):
        return {"array": obj}
    out = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, wp.array):
            out[f.name] = v
    return out


def poison(arr: "wp.array") -> None:
    """Fill ``arr`` with a finite, dtype-aware sentinel (``-7`` for signed and ``7`` for
    unsigned integers, ``-12345.0`` for floats and every float component of a vector /
    quaternion), on Warp's current stream. Never NaN: outputs are NaN-padded."""
    dt = arr.dtype
    name = getattr(dt, "__name__", "")
    if name in _INT_DTYPES:
        arr.fill_(7 if name.startswith("uint") else -7)
    elif name in _FLOAT_DTYPES:
        arr.fill_(-12345.0)
    else:
        n = getattr(dt, "_length_", None)
        arr.fill_(dt(*([-12345.0] * n)) if n else 7)


def poison_all(*results) -> None:
    """:func:`poison` every array of every result, then synchronize the device(s)."""
    devices = set()
    for obj in results:
        for arr in result_arrays(obj).values():
            poison(arr)
            devices.add(str(arr.device))
    for dev in devices:
        wp.synchronize_device(dev)


def _snapshot(obj) -> dict:
    return {k: a.numpy().copy() for k, a in result_arrays(obj).items()}


def _assert_same(got: dict, ref: dict, label: str) -> None:
    assert ref, f"{label}: no output array to compare"
    for name, want in ref.items():
        if name not in got:
            raise AssertionError(f"{label}: output {name!r} is missing")
        np.testing.assert_array_equal(got[name], want, err_msg=f"{label}: {name}")


def _differs(a: dict, b: dict) -> bool:
    return any(not np.array_equal(x, b[k], equal_nan=True) for k, x in a.items())


def side_stream(device) -> "wp.Stream":
    """A new ``wp.Stream`` on ``device`` that is not Warp's current stream."""
    s = wp.Stream(device)
    if s.cuda_stream == wp.get_stream(device).cuda_stream:  # pragma: no cover - paranoia
        raise RuntimeError("the side stream must not be Warp's current stream")
    return s


@contextlib.contextmanager
def mempool_disabled(device) -> Iterator[None]:
    """Disable the Warp mempool on ``device`` for the block (an allocation inside a
    capture then fails instead of being served from the pool)."""
    was = wp.is_mempool_enabled(device)
    wp.set_mempool_enabled(device, False)
    try:
        yield
    finally:
        wp.set_mempool_enabled(device, was)


@contextlib.contextmanager
def capture_launch_forbidden() -> Iterator[list]:
    """Make ``wp.capture_launch`` raise for the block (also the reference
    ``track_gen.exec.runner`` holds). Yields the list of attempted calls; on a normal
    exit the block also fails if one was made (an implementation that swallowed the
    ``AssertionError`` is caught too)."""
    from ..exec import runner  # noqa: PLC0415 - avoid a package-level cycle

    calls: list = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError(
            "wp.capture_launch called while recording into the caller's capture "
            "(a graph launch cannot be recorded: CUDA error 900)")

    owners = [wp] + ([runner.wp] if runner.wp is not wp else [])
    saved = [(o, o.capture_launch) for o in owners]
    for o in owners:
        o.capture_launch = forbidden
    try:
        yield calls
    finally:
        for o, real in saved:
            o.capture_launch = real
    assert not calls, "wp.capture_launch was called during the caller's recording"


@contextlib.contextmanager
def recording(device, stream: "wp.Stream") -> Iterator[object]:
    """The caller's Warp capture on ``stream``: mempool disabled, ``capture_launch``
    forbidden. Yields the ``wp.ScopedCapture`` (its ``.graph`` after the block).

    Context managers exit in reverse order: the tripwire is lifted before
    ``capture_end``, and the mempool is restored after it."""
    with mempool_disabled(device), wp.ScopedCapture(device=device, stream=stream) as cap, \
            capture_launch_forbidden():
        yield cap


def _torch_cuda():
    try:
        import torch  # noqa: PLC0415 - only the torch capture modes need it
    except ImportError:
        return None
    try:
        return torch if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover - a broken torch install
        return None


def _write_seeds(producer, values: np.ndarray) -> None:
    """Reseed ``producer`` (outside any capture) and wait for it."""
    dev = str(producer.seeds.device)
    arr = wp.array(values.astype(np.int32), dtype=wp.int32, device=dev)
    rng = getattr(producer, "_rng", None)
    if producer.seeds.bound and rng is not None and \
            getattr(rng, "seeds_warp", None) is producer.seeds.seeds:
        rng.set_seeds_warp(arr, None)
    elif producer.seeds.bound:
        wp.copy(producer.seeds.seeds, arr)
    else:
        producer.seeds.set(arr)
    wp.synchronize_device(dev)


def _eager_twin(producer, seeds: np.ndarray):
    """A fresh producer of the same type and config, built with ``seeds`` and run once
    with ``enqueue()`` (eager: never self-graphs)."""
    from .._src.rng_utils import PerEnvSeededRNG  # noqa: PLC0415

    dev = str(producer.seeds.device)
    E = int(seeds.shape[0])
    rng = PerEnvSeededRNG(seeds=wp.array(seeds.astype(np.int32), dtype=wp.int32,
                                         device=dev), num_envs=E, device=dev)
    twin = type(producer)(producer.config, rng)
    out = twin.enqueue()
    wp.synchronize_device(dev)
    return _snapshot(out)


def assert_caller_capturable(producer, *, modes: Iterable[str] = CAPTURE_MODES,
                             record: "Callable | None" = None,
                             twin: "Callable | None" = None,
                             reseed: "np.ndarray | None" = None) -> dict:
    """Record ``producer`` into a caller's capture, replay it and compare with an eager
    twin; reseed between replays and compare again.

    Args:
        producer: a CUDA ``TrackGenerator`` / ``GateGenerator`` (anything with
            ``generate(stream=)``, ``seeds`` (a ``SeedState``) and ``config``), whose
            epochs are all 0 (never advanced).
        modes: the capture modes to run (see the module docstring).
        record: ``record(stream) -> result`` recording one generation (default
            ``producer.generate(stream=stream)``; ``stream`` is ``None`` in the
            ``"torch"`` mode, where the producer must adopt torch's capture stream).
            ``result`` is a dataclass of arrays (or a mapping name -> array).
        twin: ``twin(seeds) -> {name: numpy array}``, the eager reference for the
            ``[E]`` int32 ``seeds`` (default: a fresh producer of the same type built
            from ``producer.config`` with those seeds, run with ``enqueue()``).
        reseed: the ``[E]`` seeds to reseed to between replays (default: the current
            seeds + 7919).

    Returns:
        ``{mode: "ok" | "skipped: <reason>"}``.

    Raises:
        AssertionError: a recording failed, a replay differs from the twin, or the
            recording self-captured or launched a graph.
        ValueError: ``producer`` is not on a CUDA device, its epochs are not 0, an
            unknown mode, or a reseed that does not change the batch (the reseed
            comparison would be vacuous).
    """
    modes = tuple(modes)
    unknown = [m for m in modes if m not in CAPTURE_MODES]
    if unknown:
        raise ValueError(f"unknown capture mode(s) {unknown}; known: {CAPTURE_MODES}")
    dev = wp.get_device(producer.seeds.device)
    if not dev.is_cuda:
        raise ValueError(f"assert_caller_capturable needs a CUDA producer; this one is "
                         f"on {dev.alias!r}")
    if np.any(producer.seeds.epoch.numpy() != 0):
        raise ValueError("assert_caller_capturable needs a producer at epoch 0 (its eager "
                         "twin is built fresh); call producer.seeds.reset_epoch() first")
    record = record or (lambda stream: producer.generate(stream=stream))
    twin = twin or (lambda seeds: _eager_twin(producer, seeds))
    seeds0 = producer.seeds.seeds.numpy().astype(np.int64)
    seeds1 = np.asarray(reseed, dtype=np.int64) if reseed is not None else seeds0 + 7919
    seeds0 = ((seeds0 + 2**31) % 2**32 - 2**31).astype(np.int32)
    seeds1 = ((seeds1 + 2**31) % 2**32 - 2**31).astype(np.int32)
    ref0, ref1 = twin(seeds0), twin(seeds1)
    if not _differs(ref0, ref1):
        raise ValueError("reseeding does not change the eager batch: the replay-after-"
                         "reseed comparison would be vacuous")
    graph_before = getattr(producer, "graph", None)
    status: dict = {}
    for mode in modes:
        torch = None
        if mode in ("external", "torch"):
            torch = _torch_cuda()
            if torch is None:
                status[mode] = "skipped: torch with CUDA is not available"
                continue
        _write_seeds(producer, seeds0)
        label = f"{mode} capture"
        if mode == "warp":
            s = side_stream(dev)
            wp.synchronize_device(dev)
            with recording(dev, s) as cap:
                out = record(s)

            def replay(_g=cap.graph, _s=s):
                wp.capture_launch(_g, stream=_s)
                wp.synchronize_device(dev)
        else:
            # torch captures, replays and synchronizes on its CURRENT device: make it
            # the producer's for the whole mode.
            ordinal = dev.ordinal
            with torch.cuda.device(ordinal):
                g = torch.cuda.CUDAGraph()
                torch.cuda.synchronize(ordinal)
                with mempool_disabled(dev):
                    with torch.cuda.graph(g):
                        if mode == "external":
                            from ..interop.torch import registered  # noqa: PLC0415
                            with registered(graph=g) as ws, capture_launch_forbidden():
                                out = record(ws)
                        else:
                            with capture_launch_forbidden():
                                out = record(None)
                torch.cuda.synchronize(ordinal)

            def replay(_g=g, _torch=torch, _ordinal=ordinal):
                with _torch.cuda.device(_ordinal):
                    _g.replay()
                    _torch.cuda.synchronize(_ordinal)
        if getattr(producer, "graph", None) is not graph_before:
            raise AssertionError(f"{label}: recording into the caller's graph created or "
                                 f"replaced the producer's own graph (self-capture)")
        # A recording does not execute: the replays below are the first runs.
        poison_all(out)
        replay()
        _assert_same(_snapshot(out), ref0, f"{label}: replay")
        _write_seeds(producer, seeds1)
        poison_all(out)
        replay()
        _assert_same(_snapshot(out), ref1, f"{label}: replay after a reseed")
        _write_seeds(producer, seeds0)
        status[mode] = "ok"
    return status
