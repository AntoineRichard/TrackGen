"""Does a generator honour the contract? ``check_generator`` / ``check_gate_generator``.

One call runs a registered generator through every check the generator contract
implies (see ``docs/contributing/writing-a-generator.rst``) and returns a
:class:`Report`::

    from track_gen.testing import check_generator

    report = check_generator("acme.superformula", params={"m": 7.0})  # cpu (+ cuda:0)
    report.raise_for_failures()

Every check runs per device. A check is a named :class:`CheckResult`; its name is stable,
so a test can assert exactly which check a broken generator fails.

Checks (:data:`CHECKS`), for a track generator:

``construct``
    The generator resolves, validates its params, allocates, runs its warm-up pass and
    a first ``generate()`` (into a poisoned raw output: the batch the checks read).
``output.shape``
    The raw output (the ``[E*R]`` vec2f centerline and the ``[E]`` int32 validity flags)
    and every ``Track`` field have the documented shape and dtype; flags are 0 or 1.
``output.finite``
    The raw centerline is finite for every env the generator marked valid, and the final
    ``Track`` is finite before ``count[e]`` for every valid env. The raw output is
    poisoned before the generation the checks read, so a point the generator never
    writes is reported as such (it would otherwise hold stale, possibly finite memory).
``output.padding``
    Every per-point ``Track`` field is NaN past ``count[e]`` (every env).
``output.count``
    ``0 <= count[e] <= N_max`` for every env, ``3 <= count[e]`` for valid ones.
``output.closed``
    The raw loop closes (the last-to-first gap is at most 3x the median step, or 1.5x
    the longest step between consecutive points) and the pipeline accepts it (valid
    fraction > 0).
``determinism``
    Two fresh generators built with the same seeds produce bitwise identical batches
    (raw output and every result field, NaN padding included).
``diversity``
    Different seeds produce different tracks: no env repeats under a disjoint set of
    seeds, and no two envs of one batch are identical (envs valid in both batches only:
    two rejected envs may legitimately be alike).
``batch_independence``
    Env ``e``'s track depends on its own seed only: it is identical alone (``E = 1``),
    inside a larger batch, and when the other envs' seeds are permuted.
``statelessness``
    ``generate()`` twice with unchanged seeds reproduces the first batch, a reseed
    reproduces a fresh generator with the new seeds, and reseeding back reproduces the
    first batch (no hidden state carried between calls).
``hot_path.host_sync`` / ``hot_path.allocation`` / ``hot_path.stream`` / ``hot_path.capture`` (CPU)
    Two ``generate()`` calls under :func:`~track_gen.testing.contract.hot_path_guard`:
    no host sync or readback, no allocation, every launch / copy from track_gen or the
    plugin's package passes ``stream=`` (no ``assign()`` / ``zero_()`` / ``fill_()``),
    no nested capture or graph launch. On CPU the stream check sees that ``stream=`` is
    passed, not which stream (``stream=None`` is what a CPU ``LaunchContext`` passes):
    a launch on the wrong stream is caught by ``capture.caller`` on CUDA.
``hot_path.topology`` (CPU)
    The two calls emit the same launch sequence (:class:`~track_gen.testing.trace.LaunchTrace`):
    kernels, dimensions, buffers and scalar arguments.
``hot_path.buffers`` (CPU)
    Every pointer launched on belongs to the generator (its result, scratch, seed and
    parameter buffers; see :func:`~track_gen.testing.trace.buffers_of`), and a second
    generator built alike launches on none of the first one's buffers.
``capture.caller`` (CUDA)
    :func:`~track_gen.testing.capture.assert_caller_capturable`: recorded into a caller's
    capture, replays equal the eager batch, before and after a reseed.
``capture.self_graph`` (CUDA)
    ``generate()`` captures the generator's own graph; its replays (into poisoned
    outputs, before and after a reseed) equal the eager batch.

A gate generator runs the same checks on its ``GateSequence`` (raw output: the ``[E*G]``
vec2f anchors before ``count[e]``), with ``output.accepted`` (valid fraction > 0) in
place of ``output.closed`` and ``min_gates <= count[e] <= max_gates`` for valid envs.

A ``capturable=False`` generator (eager on CUDA every call) waives
``hot_path.host_sync``, ``hot_path.allocation``, ``hot_path.topology``,
``hot_path.buffers`` and ``capture.caller`` (reported as skipped); ``capture.self_graph``
then checks that it never self-graphs.

:func:`check_stage` runs a ``relax`` / ``width`` / ``elevation`` / ``validity`` stage
inside the default track pipeline (:func:`track_gen.pipeline.presets.track`, the stage
in its slot; a validity check after ``validity.default``) through the same machinery,
with these checks (no ``diversity`` / ``output.closed``: they are the generator's
business):

``output.accepted`` (relax / width / elevation)
    The pipeline accepts some env (valid fraction > 0). When it accepts none, the same
    pipeline with the slot's neutral built-in (``relax.none``, ``width.constant``,
    ``elevation.flat``) is run: if it accepts none either, the check is skipped (the
    stage did not reject them).
``stage.isolation``
    After a generation, re-running ONLY the stage's ``enqueue`` on the same inputs
    leaves the versions it read unchanged (it never writes a read port) and reproduces
    the versions it wrote (no stale scratch, no dependency on its own previous output).
``validity.bits`` (validity)
    The stage owns at least one fail bit (``StageBuilder.fail_bit``), sets no bit it
    does not own (the ``fail_mask`` of the pipeline without it, OR its own bits), and
    ``Track.valid == (fail_mask == 0)``.

``determinism``, ``batch_independence``, ``statelessness``, ``hot_path.*`` and
``capture.*`` are the generator kit's, on the whole Program (``capturable``: the built
Program's).
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import traceback
import types
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import warp as wp

from .capture import CAPTURE_MODES, _write_seeds, assert_caller_capturable, poison_all
from .capture import result_arrays
from .contract import _HINTS, HotPathGuard
from .trace import BufferSet, LaunchTrace, buffers_of

__all__ = [
    "CHECKS",
    "CheckResult",
    "Report",
    "check_gate_generator",
    "check_generator",
    "check_stage",
]

_COMMON = ("determinism", "diversity", "batch_independence", "statelessness",
           "hot_path.host_sync", "hot_path.allocation", "hot_path.stream",
           "hot_path.capture", "hot_path.topology", "hot_path.buffers",
           "capture.caller", "capture.self_graph")
_STAGE_COMMON = ("determinism", "batch_independence", "statelessness",
                 "hot_path.host_sync", "hot_path.allocation", "hot_path.stream",
                 "hot_path.capture", "hot_path.topology", "hot_path.buffers",
                 "capture.caller", "capture.self_graph")
_STAGE_OUTPUT = ("construct", "output.shape", "output.finite", "output.padding",
                 "output.count")
#: The track-pipeline stage kinds :func:`check_stage` runs.
STAGE_KINDS = ("relax", "width", "elevation", "validity")
# The neutral built-in of each slot (the output.accepted reference).
_NEUTRAL = {"relax": "none", "width": "constant", "elevation": "flat"}

#: The check names of each registry kind, in run order.
CHECKS: Mapping[str, tuple] = types.MappingProxyType({
    "generator": ("construct", "output.shape", "output.finite", "output.padding",
                  "output.count", "output.closed") + _COMMON,
    "gate_generator": ("construct", "output.shape", "output.finite", "output.padding",
                       "output.count", "output.accepted") + _COMMON,
    **{k: _STAGE_OUTPUT + ("output.accepted", "stage.isolation") + _STAGE_COMMON
       for k in ("relax", "width", "elevation")},
    "validity": _STAGE_OUTPUT + ("stage.isolation", "validity.bits") + _STAGE_COMMON,
})

_HOT = ("hot_path.host_sync", "hot_path.allocation", "hot_path.stream",
        "hot_path.capture", "hot_path.topology", "hot_path.buffers")
_CAPTURE = ("capture.caller", "capture.self_graph")
# Waived for a capturable=False generator (eager every call: may sync / allocate).
_WAIVED_EAGER = ("hot_path.host_sync", "hot_path.allocation", "hot_path.topology",
                 "hot_path.buffers", "capture.caller")
_GAP_RATIO = 3.0       # closed loop: last->first gap <= 3x the median step, or
_GAP_MAX_RATIO = 1.5   # 1.5x the longest step between consecutive points
# The raw output is poisoned before the generation the checks read: a slot the
# generator never writes keeps the poison. Points: a NaN with a payload no arithmetic
# produces (so "never written" is told apart from "written NaN"); flags: 7.
_POINT_POISON_BITS = 0x7FC0A5A5
_FLAG_POISON = 7


# ---------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CheckResult:
    """The outcome of one check on one device.

    Attributes:
        name: the check name (see :data:`CHECKS`).
        device: the Warp device alias (``"cpu"``, ``"cuda:0"``).
        passed: ``True`` when the check passed or was skipped.
        message: why it failed (or was skipped); a short note otherwise.
        skipped: the check did not run (not applicable, waived, or ``skip=``).
        details: measurements, e.g. ``{"valid_fraction": 0.875}`` for
            ``output.closed``.
    """

    name: str
    device: str
    passed: bool
    message: str = ""
    skipped: bool = False
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict, compare=False)

    @property
    def status(self) -> str:
        return "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")

    def __str__(self) -> str:
        text = f"[{self.device}] {self.name}: {self.status}"
        return f"{text} - {self.message}" if self.message else text


@dataclasses.dataclass
class Report:
    """Every :class:`CheckResult` of one :func:`check_generator` /
    :func:`check_gate_generator` run (iterable, ``len()``)."""

    kind: str
    name: str
    results: list

    @property
    def ok(self) -> bool:
        """Every check passed or was skipped."""
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list:
        """The failed :class:`CheckResult` s, in run order."""
        return [r for r in self.results if not r.passed]

    @property
    def failed_names(self) -> list:
        """The names of the failed checks (each once, sorted)."""
        return sorted({r.name for r in self.failures})

    def result(self, name: str, device: "str | None" = None) -> CheckResult:
        """The result of check ``name`` (on ``device``, default: the first device)."""
        for r in self.results:
            if r.name == name and (device is None or r.device == device):
                return r
        raise KeyError(f"no result for check {name!r}"
                       + (f" on {device!r}" if device else ""))

    def raise_for_failures(self) -> "Report":
        """Raise one ``AssertionError`` listing every failed check; return ``self``
        when all passed."""
        failures = self.failures
        if failures:
            lines = "\n".join(f"  {r}" for r in failures)
            raise AssertionError(
                f"{_label(self.kind)} {self.name!r} failed {len(failures)} conformance "
                f"check(s):\n{lines}")
        return self

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __str__(self) -> str:
        ran = [r for r in self.results if not r.skipped]
        passed = sum(r.passed for r in ran)
        head = (f"{_label(self.kind)} {self.name!r}: {passed}/{len(ran)} checks passed"
                f"{f', {len(self.results) - len(ran)} skipped' if len(ran) < len(self.results) else ''}")
        if not self.results:
            return head
        w_dev = max(len(r.device) for r in self.results)
        w_name = max(len(r.name) for r in self.results)
        rows = [head]
        for r in self.results:
            row = f"  {r.device:<{w_dev}}  {r.name:<{w_name}}  {r.status}"
            if r.message:
                row += f"  {r.message}"
            rows.append(row)
        return "\n".join(rows)


def _label(kind: str) -> str:
    if kind in STAGE_KINDS:
        return f"{kind} stage"
    return "gate generator" if kind == "gate_generator" else "generator"


# ---------------------------------------------------------------------------------------
# Snapshots: a host copy of one producer's raw output and result
# ---------------------------------------------------------------------------------------

_TRACK_POINT = {"outer": wp.vec3f, "center": wp.vec3f, "inner": wp.vec3f,
                "tangent": wp.vec3f, "normal": wp.vec3f, "arclen": wp.float32}
_TRACK_ENV = {"length": wp.float32, "valid": wp.int32, "count": wp.int32,
              "winding": wp.float32, "truncated": wp.int32, "generation": wp.int32,
              "fail_mask": wp.int32}
_GATE_POINT = {"position": wp.vec3f, "tangent": wp.vec3f, "forward": wp.vec3f,
               "orientation": wp.quatf, "half_size": wp.float32, "left": wp.vec3f,
               "right": wp.vec3f}
_GATE_ENV = {"valid": wp.int32, "count": wp.int32, "generation": wp.int32}
_OPTIONAL = {"truncated", "generation", "fail_mask"}


class _Snapshot:
    def __init__(self, kind: str, gen) -> None:
        cfg = gen._config
        self.kind = kind
        self.E = E = int(cfg.num_envs)
        if kind == "generator":
            result = gen._track
            raw = {"raw.centerline": gen._scratch.gen_centerline,
                   "raw.valid": gen._scratch.gen_valid}
            self.stride = int(cfg.N_max)
            self.raw_stride = int(raw["raw.centerline"].shape[0]) // E
            self.point, self.env_fields = _TRACK_POINT, _TRACK_ENV
        else:
            result = gen._gate_sequence
            raw = {"raw.position": gen._raw_anchors}
            self.stride = self.raw_stride = int(cfg.max_gates)
            self.point, self.env_fields = _GATE_POINT, _GATE_ENV
        arrays = {**raw, **result_arrays(result)}
        self.meta = {k: (tuple(int(s) for s in a.shape), a.dtype) for k, a in arrays.items()}
        self.data = {k: a.numpy().copy() for k, a in arrays.items()}
        self.count = self.data["count"].astype(np.int64)
        self.valid = self.data["valid"].astype(np.int64)
        self.min_count = 3 if kind == "generator" else int(getattr(cfg, "min_gates", 2))

    def clamped(self, e: int) -> int:
        return int(min(max(self.count[e], 0), self.stride))

    def unwritten(self, name: str, e: int, start: int = 0,
                  stop: "int | None" = None) -> np.ndarray:
        """Indices (from ``start``) of the raw points of env ``e`` that still hold the
        kit's poison."""
        rows = np.ascontiguousarray(self.rows(name, e, start, stop))
        bits = rows.view(np.uint32).reshape(rows.shape[0], -1)
        return np.flatnonzero((bits == _POINT_POISON_BITS).any(axis=1))

    def comparable(self) -> np.ndarray:
        """Per env: valid (a track generator: also marked valid by the generator)."""
        ok = self.valid == 1
        if self.kind == "generator":
            ok &= self.data["raw.valid"] == 1
        return ok

    def rows(self, name: str, e: int, start: int = 0, stop: "int | None" = None):
        a = self.data[name]
        n = self.raw_stride if name.startswith("raw.") else self.stride
        stop = n if stop is None else stop
        return a[e * n + start:e * n + stop]

    def env(self, e: int) -> dict:
        """field -> bytes of env ``e`` (gate raw anchors: before ``count[e]`` only)."""
        out = {}
        for name, a in self.data.items():
            if name == "raw.position":
                out[name] = self.rows(name, e, 0, self.clamped(e)).tobytes()
            elif name.startswith("raw.") and a.shape[0] == self.E:
                out[name] = a[e:e + 1].tobytes()
            elif name.startswith("raw."):
                out[name] = self.rows(name, e).tobytes()
            elif a.shape[0] == self.E * self.stride:
                out[name] = self.rows(name, e).tobytes()
            elif a.shape[0] == self.E:
                out[name] = a[e:e + 1].tobytes()
        return out

    def digest(self, e: int) -> str:
        h = hashlib.sha256()
        for k, v in sorted(self.env(e).items()):
            h.update(k.encode())
            h.update(v)
        return h.hexdigest()


def _raw_buffers(kind: str, gen) -> list:
    """``(array, is_flag)`` of a producer's raw output buffers."""
    if kind == "generator":
        return [(gen._scratch.gen_centerline, False), (gen._scratch.gen_valid, True)]
    return [(gen._raw_anchors, False)]


def _poison_raw(kind: str, gen) -> None:
    """Fill the raw output with the kit's poison (outside any guard) and wait."""
    devices = set()
    for arr, is_flag in _raw_buffers(kind, gen):
        n = int(arr.shape[0])
        if is_flag:
            host = np.full(n, _FLAG_POISON, dtype=np.int32)
        else:
            width = int(getattr(arr.dtype, "_length_", 1) or 1)
            host = np.full((n, width), _POINT_POISON_BITS, dtype=np.uint32).view(np.float32)
        wp.copy(arr, wp.array(host, dtype=arr.dtype, device=arr.device))
        devices.add(str(arr.device))
    for dev in devices:
        wp.synchronize_device(dev)


# stage.isolation fills a stage's write buffers with this byte before re-running it.
_BYTE_POISON = 0xA5


def _byte_poisoned(host: np.ndarray) -> np.ndarray:
    """A copy of ``host`` with every byte set to :data:`_BYTE_POISON`."""
    out = np.empty_like(host)
    out.view(np.uint8)[...] = _BYTE_POISON
    return out


def _is_byte_poison(host: np.ndarray) -> np.ndarray:
    """Per slot (first axis): every byte is :data:`_BYTE_POISON`."""
    n = int(host.shape[0])
    return (np.ascontiguousarray(host).view(np.uint8).reshape(n, -1) == _BYTE_POISON).all(axis=1)


def _raw_delta(a: _Snapshot, ea: int, b: _Snapshot, eb: int) -> float:
    """The largest coordinate difference between the raw outputs of env ``ea`` of ``a``
    and env ``eb`` of ``b`` (``inf`` when their gate counts differ)."""
    if a.kind == "generator":
        x, y = a.rows("raw.centerline", ea), b.rows("raw.centerline", eb)
    else:
        ca, cb = a.clamped(ea), b.clamped(eb)
        if ca != cb:
            return math.inf
        x, y = a.rows("raw.position", ea, 0, ca), b.rows("raw.position", eb, 0, cb)
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else 0.0


def _env_diff(a: _Snapshot, ea: int, b: _Snapshot, eb: int) -> list:
    """The fields in which env ``ea`` of ``a`` and env ``eb`` of ``b`` differ."""
    x, y = a.env(ea), b.env(eb)
    return sorted((k for k in set(x) | set(y) if x.get(k) != y.get(k)),
                  key=lambda k: (not k.startswith("raw."), k))


def _fields_text(fields: Sequence[str], n: int = 4) -> str:
    return ", ".join(fields[:n]) + (", ..." if len(fields) > n else "")


def _list_text(items: Sequence[str], n: int = 4) -> str:
    items = list(items)
    text = "; ".join(items[:n])
    return text + (f"; (+{len(items) - n} more)" if len(items) > n else "")


def _exc_text(exc: BaseException) -> str:
    tb = traceback.extract_tb(exc.__traceback__)
    where = ""
    if tb:
        fr = tb[-1]
        where = f" (at {os.path.basename(fr.filename)}:{fr.lineno} in {fr.name})"
    msg = str(exc).strip().splitlines()
    first = msg[0] if msg else ""
    return f"raised {type(exc).__name__}: {first}{where}"


def _as_rows(a: np.ndarray) -> np.ndarray:
    """``a`` as ``[rows, components]`` (also for zero rows)."""
    return a.reshape(a.shape[0], int(np.prod(a.shape[1:], dtype=np.int64)))


def _int32(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64)
    return ((v + 2**31) % 2**32 - 2**31).astype(np.int32)


# ---------------------------------------------------------------------------------------
# One device
# ---------------------------------------------------------------------------------------

class _DeviceRun:
    def __init__(self, kind: str, name: str, params, device: str, num_envs: int,
                 seeds_a: np.ndarray, seeds_b: np.ndarray, overrides: Mapping,
                 info, probe: tuple, modes: tuple, skip: frozenset,
                 stage: "_StageUnderTest | None" = None) -> None:
        self.kind, self.name, self.params = kind, name, params
        self.device = device
        self.E = num_envs
        self.seeds_a, self.seeds_b = seeds_a, seeds_b
        self.overrides = dict(overrides)
        self.info = info
        self.probe = probe
        self.modes = modes
        self.skip = skip
        self.stage = stage
        # What a snapshot reads: a stage runs inside a TrackGenerator.
        self.snap_kind = "generator" if stage is not None else kind
        self.is_cuda = wp.get_device(device).is_cuda
        self.results: list = []
        self.snap_b: "_Snapshot | None" = None
        self.capturable = bool(info.capturable)
        self.accepts_ctx = bool(info.accepts_ctx)

    # -- construction ------------------------------------------------------------
    def make(self, seeds: np.ndarray):
        from .._src.rng_utils import PerEnvSeededRNG  # noqa: PLC0415

        dev = self.device
        E = int(seeds.shape[0])
        rng = PerEnvSeededRNG(seeds=wp.array(_int32(seeds), dtype=wp.int32, device=dev),
                              num_envs=E, device=dev)
        if self.stage is not None:
            from .._src.track_generator import TrackGenerator  # noqa: PLC0415

            return TrackGenerator(self.stage.spec(E, dev), rng)
        kw = dict(self.overrides)
        if self.params is not None:
            kw["generator_params"] = self.params
        if self.kind == "generator":
            from .._src.track_generator import TrackGenerator  # noqa: PLC0415
            from .._src.types import TrackGenConfig  # noqa: PLC0415

            return TrackGenerator(TrackGenConfig(generator=self.name, device=dev,
                                                 num_envs=E, **kw), rng)
        from .._src.gate_generator import GateGenerator  # noqa: PLC0415
        from .._src.types import GateGenConfig  # noqa: PLC0415

        if "gate_ordering" not in kw:
            from ..plugin import registry  # noqa: PLC0415

            supported = getattr(registry.get("gate_generator", self.name),
                                "supported_orderings", None) or {"ccw"}
            kw["gate_ordering"] = "ccw" if "ccw" in supported else sorted(supported)[0]
        return GateGenerator(GateGenConfig(generator=self.name, device=dev, num_envs=E,
                                           **kw), rng)

    def snapshot(self, gen) -> _Snapshot:
        if self.is_cuda:
            wp.synchronize_device(self.device)
        return _Snapshot(self.snap_kind, gen)

    def eager(self, gen) -> None:
        """One EAGER generation (``enqueue()``: never self-captures), then a device sync.

        The data checks run the generator eagerly so they judge its output on its own;
        capture is exercised only by the ``capture.*`` checks, so a capture-hostile
        generator fails there with a precise message instead of failing ``construct``.
        """
        gen.enqueue()
        if self.is_cuda:
            wp.synchronize_device(self.device)

    def poisoned(self, gen) -> _Snapshot:
        """Poison ``gen``'s raw output, generate once (eagerly), snapshot."""
        _poison_raw(self.snap_kind, gen)
        self.eager(gen)
        return self.snapshot(gen)

    def fresh(self, seeds: np.ndarray) -> _Snapshot:
        """The poisoned snapshot of a new generator built with ``seeds``."""
        return self.poisoned(self.make(seeds))

    # -- bookkeeping ---------------------------------------------------------------
    def add(self, name: str, passed: bool, message: str = "", *, skipped: bool = False,
            details: "Mapping | None" = None) -> None:
        self.results.append(CheckResult(name, self.device, bool(passed), message,
                                        skipped, types.MappingProxyType(dict(details or {}))))

    def skipped(self, name: str) -> bool:
        return name in self.skip or name.split(".")[0] in self.skip

    def guarded(self, names: Sequence[str], fn) -> None:
        """Run ``fn() -> [(name, passed, message, details)]`` for the ``names`` not
        skipped by the caller; an exception fails every one of them."""
        todo = [n for n in names if not self.skipped(n)]
        for n in names:
            if self.skipped(n):
                self.add(n, True, "skipped by the caller (skip=)", skipped=True)
        if not todo:
            return
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 - every failure is a report entry
            for n in todo:
                self.add(n, False, _exc_text(exc))
            return
        for name, passed, message, details in out:
            if name in todo:
                if passed is None:
                    self.add(name, True, message, skipped=True, details=details)
                else:
                    self.add(name, passed, message, details=details)

    # -- run -------------------------------------------------------------------------
    def run(self) -> list:
        names = [n for n in CHECKS[self.kind]
                 if not (n in _HOT and self.is_cuda) and not (n in _CAPTURE and not self.is_cuda)]
        try:
            gen = self.make(self.seeds_a)
            snap = self.poisoned(gen)
        except Exception as exc:  # noqa: BLE001
            self.add("construct", False, _exc_text(exc))
            for n in names[1:]:
                self.add(n, True, "not run: construct failed", skipped=True)
            return self.results
        if self.stage is not None:
            self.capturable = bool(gen.program.capturable)
            self.accepts_ctx = True
        self.add("construct", True, details={"capturable": self.capturable,
                                             "accepts_ctx": self.accepts_ctx})
        out_names = [n for n in names if n.startswith("output.")]
        self.guarded(out_names, lambda: self.output_checks(snap))
        if self.stage is not None:
            self.guarded(["stage.isolation"], lambda: [self.stage_isolation(gen)])
            if self.kind == "validity":
                self.guarded(["validity.bits"], lambda: [self.validity_bits(gen, snap)])
        self.guarded(["determinism"], lambda: [self.determinism(snap)])
        if self.stage is None:          # a stage need not add diversity: the generator's
            self.guarded(["diversity"], lambda: [self.diversity(snap)])
        self.guarded(["batch_independence"], lambda: [self.batch_independence(snap)])
        self.guarded(["statelessness"], lambda: [self.statelessness(gen, snap)])
        if self.is_cuda:
            self.guarded(["capture.caller"], lambda: [self.capture_caller()])
            self.guarded(["capture.self_graph"], lambda: [self.capture_self_graph(snap)])
        else:
            self.guarded(list(_HOT), lambda: self.hot_path(gen))
        return self.results

    # -- 1. output -------------------------------------------------------------------
    def output_checks(self, s: _Snapshot) -> list:
        E, N = s.E, s.stride
        track = self.snap_kind == "generator"
        out = []

        # shape / dtype / flag values
        expect = {}
        if track:
            R = s.raw_stride
            expect["raw.centerline"] = ((E * R,), wp.vec2f)
            expect["raw.valid"] = ((E,), wp.int32)
        else:
            expect["raw.position"] = ((E * N,), wp.vec2f)
        expect.update({k: ((E * N,), dt) for k, dt in s.point.items()})
        expect.update({k: ((E,), dt) for k, dt in s.env_fields.items()})
        problems = []
        if track and s.raw_stride < 3:
            problems.append(f"the raw centerline holds {s.raw_stride} point(s) per env "
                            f"(at least 3 needed)")
        for k, (shape, dtype) in expect.items():
            if k not in s.meta:
                if k not in _OPTIONAL:
                    problems.append(f"{k} is missing")
                continue
            got_shape, got_dtype = s.meta[k]
            if got_shape != shape or got_dtype is not dtype:
                problems.append(f"{k} is {got_dtype.__name__}{list(got_shape)}, expected "
                                f"{dtype.__name__}{list(shape)}")
        flags = ["valid"] + (["raw.valid"] if track else [])
        for k in flags:
            vals = np.unique(s.data[k])
            if k == "raw.valid" and (s.data[k] == _FLAG_POISON).any():
                envs = np.flatnonzero(s.data[k] == _FLAG_POISON).tolist()
                problems.append(f"out_valid[e] was not written for env(s) {envs[:6]}"
                                f"{' ...' if len(envs) > 6 else ''} (it still holds the "
                                f"kit's poison {_FLAG_POISON}): write 0 or 1 for every env "
                                f"on every call")
                vals = vals[vals != _FLAG_POISON]
            if not set(vals.tolist()) <= {0, 1}:
                problems.append(f"{k} holds {vals.tolist()} (flags must be 0 or 1)")
        out.append(("output.shape", not problems, _list_text(problems), {}))

        # finite (raw for generator-valid envs; final before count for valid envs)
        bad = []
        if track:
            raw_valid = s.data["raw.valid"]
            raw_finite = [bool(np.isfinite(s.rows("raw.centerline", e)).all())
                          for e in range(E)]
            for e in range(E):
                if raw_valid[e] == 1 and not raw_finite[e]:
                    unwritten = s.unwritten("raw.centerline", e)
                    if unwritten.size:
                        bad.append(f"env {e} is marked valid (out_valid=1) but "
                                   f"{unwritten.size} raw point(s) were not written, first "
                                   f"at index {int(unwritten[0])} (it still holds the "
                                   f"kit's poison): write all {s.raw_stride} points of "
                                   f"every env on every call")
                        continue
                    rows = s.rows("raw.centerline", e)
                    idx = np.flatnonzero(~np.isfinite(rows).all(axis=-1))
                    bad.append(f"env {e} is marked valid (out_valid=1) but its raw "
                               f"centerline has {idx.size} non-finite point(s), first at "
                               f"index {int(idx[0])}")
            details = {"raw_valid_fraction": float(np.mean(raw_valid == 1)),
                       "raw_finite_all": all(raw_finite)}
        else:
            details = {}
            for e in range(E):
                if s.valid[e] == 1:
                    rows = s.rows("raw.position", e, 0, s.clamped(e))
                    unwritten = s.unwritten("raw.position", e, 0, s.clamped(e))
                    if unwritten.size:
                        bad.append(f"valid env {e}: raw gate anchor {int(unwritten[0])} "
                                   f"(before count={int(s.count[e])}) was not written (it "
                                   f"still holds the kit's poison)")
                    elif not np.isfinite(rows).all():
                        bad.append(f"valid env {e}: a raw gate anchor before count="
                                   f"{int(s.count[e])} is not finite")
        final_all = True
        for e in range(E):
            c = s.clamped(e)
            for k in s.point:
                if k not in s.data:
                    continue
                fin = np.isfinite(_as_rows(s.rows(k, e, 0, c))).all(axis=1)
                if not fin.all():
                    final_all = False
                    if s.valid[e] == 1:
                        bad.append(f"valid env {e}: {k}[{int(np.flatnonzero(~fin)[0])}] "
                                   f"(before count={int(s.count[e])}) is not finite")
        details["final_finite_all"] = final_all
        out.append(("output.finite", not bad,
                    _list_text(bad) + (". Output must be finite for every env the "
                                       "generator marks valid; write 0 to out_valid to "
                                       "reject an env." if bad else ""), details))

        # NaN padding past count
        bad = []
        for e in range(E):
            c = s.clamped(e)
            for k in s.point:
                if k not in s.data or c >= N:
                    continue
                padded = np.isnan(_as_rows(s.rows(k, e, c, N))).any(axis=1)
                if not padded.all():
                    i = c + int(np.flatnonzero(~padded)[0])
                    bad.append(f"env {e}: {k}[{i}] (past count={int(s.count[e])}) is not "
                               f"NaN")
                    break
        out.append(("output.padding", not bad, _list_text(bad), {}))

        # count range
        lo = s.min_count
        bad = []
        for e in range(E):
            c = int(s.count[e])
            if not 0 <= c <= N:
                bad.append(f"env {e}: count={c} is outside [0, {N}]")
            elif s.valid[e] == 1 and c < lo:
                bad.append(f"valid env {e}: count={c} < {lo}")
        out.append(("output.count", not bad, _list_text(bad),
                    {"min": int(s.count.min()), "max": int(s.count.max())}))

        # closed loop / accepted
        frac = float(np.mean(s.valid == 1))
        if self.stage is not None:
            if "output.accepted" in CHECKS[self.kind]:
                out.append(self.stage_accepted(frac))
        elif track:
            bad, worst = [], 0.0
            R = s.raw_stride
            for e in range(E):
                if s.data["raw.valid"][e] != 1:
                    continue
                pts = s.rows("raw.centerline", e).astype(np.float64)
                if not np.isfinite(pts).all():
                    continue
                steps = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
                med = float(np.median(steps)) if steps.size else 0.0
                gap = float(np.linalg.norm(pts[0] - pts[-1]))
                if med <= 0.0:
                    bad.append(f"env {e}: the raw points are not distinct (median step 0)")
                    continue
                ratio = gap / med
                worst = max(worst, ratio)
                # The closing edge is held to the interior edges: a straight that closes
                # the loop is fine when the loop has straights of that length elsewhere.
                if gap > max(_GAP_RATIO * med, _GAP_MAX_RATIO * float(steps.max())):
                    bad.append(f"env {e}: {ratio:.1f}x")
            msg = []
            if bad:
                msg.append(f"the gap from the last raw point back to the first exceeds "
                           f"{_GAP_RATIO:g}x the median step and {_GAP_MAX_RATIO:g}x the "
                           f"longest step: {_list_text(bad)}. Write a closed loop of {R} "
                           f"points, the last one next to the first")
            if frac == 0.0:
                msg.append(f"the pipeline rejected every env (valid 0/{E})")
            out.append(("output.closed", not msg, "; ".join(msg),
                        {"valid_fraction": frac, "max_gap_ratio": worst}))
        else:
            out.append(("output.accepted", frac > 0.0,
                        "" if frac > 0.0 else f"the gate pipeline rejected every env "
                                              f"(valid 0/{E})",
                        {"valid_fraction": frac}))
        return out

    # -- stage checks ---------------------------------------------------------------------
    def stage_accepted(self, frac: float):
        details = {"valid_fraction": frac}
        if frac > 0.0:
            return ("output.accepted", True, "", details)
        neutral = _NEUTRAL[self.kind]
        ref = self.stage.reference(self.E, self.device, neutral, self.seeds_a)
        rfrac = float(np.mean(_Snapshot("generator", ref).valid == 1))
        details["reference_valid_fraction"] = rfrac
        if rfrac == 0.0:
            return ("output.accepted", None,
                    f"skipped: the pipeline rejects every env (valid 0/{self.E}) with the "
                    f"neutral {self.kind}.{neutral} too; the stage did not reject them",
                    details)
        return ("output.accepted", False,
                f"the pipeline rejected every env (valid 0/{self.E}) while it accepts "
                f"{rfrac:.0%} of them with {self.kind}.{neutral} in the slot", details)

    def stage_isolation(self, gen):
        program = gen.program
        sid = self.stage.stage_id
        ids = list(program._stage_ids)
        i = ids.index(sid)
        reads, writes = program._ports[sid]
        before = ids[i - 1]
        self.eager(gen)
        read_arrays = {p: program.tap(ch, after=before) for p, ch in reads.items()}
        write_arrays = {p: program.tap(ch, after=sid) for p, ch in writes.items()}
        snap_r = {p: a.numpy().tobytes() for p, a in read_arrays.items()}
        snap_w = {p: a.numpy().copy() for p, a in write_arrays.items()}
        from ..exec.context import LaunchContext  # noqa: PLC0415
        from .trace import trace_launches  # noqa: PLC0415

        # Every write slot gets the poison byte first: a slot the stage leaves
        # unwritten (padding included) keeps it instead of passing as unchanged.
        for p, a in write_arrays.items():
            a.assign(_byte_poisoned(snap_w[p]))
        with trace_launches() as trace:
            program._enqueues[i](LaunchContext.for_call(self.device))
        wp.synchronize_device(self.device)
        bad = []
        for p, a in read_arrays.items():
            if a.numpy().tobytes() != snap_r[p]:
                bad.append(f"the stage wrote its read port {p!r} ({reads[p]}): write into "
                           f"b.write({p!r})")
        # A stage that launches nothing (elevation.flat) writes nothing: its outputs are
        # their build-time zero initialisation, constant by construction.
        for p, a in (write_arrays.items() if len(trace) else ()):
            now, was = a.numpy(), snap_w[p]
            if now.tobytes() == was.tobytes():
                continue
            left = np.flatnonzero(_is_byte_poison(now) & ~_is_byte_poison(was))
            if left.size:
                bad.append(f"the stage left {left.size} slot(s) of {writes[p]!r} (port "
                           f"{p!r}) unwritten, first {left[:4].tolist()}: write every "
                           f"slot, padding included")
            else:
                bad.append(f"re-running the stage on the same inputs changed "
                           f"{writes[p]!r} (port {p!r}): it reads stale scratch or its "
                           f"own previous output")
        for p, a in write_arrays.items():        # leave the generation as it was
            a.assign(snap_w[p])
        wp.synchronize_device(self.device)
        return ("stage.isolation", not bad, _list_text(bad),
                {"reads": sorted(reads.values()), "writes": sorted(writes.values())})

    def validity_bits(self, gen, snap: _Snapshot):
        program = gen.program
        sid = self.stage.stage_id
        own = [n for n, owner in program._fail_owner.items() if owner == sid]
        if not own:
            return ("validity.bits", False, "the stage owns no fail bit (call "
                    "b.fail_bit() in build)", {})
        own_mask = 0
        for n in own:
            own_mask |= 1 << int(program.fail_bits[n])
        fm = snap.data["fail_mask"].astype(np.int64)
        valid = snap.data["valid"].astype(np.int64)
        bad = []
        if not np.array_equal(valid, (fm == 0).astype(np.int64)):
            envs = np.flatnonzero(valid != (fm == 0)).tolist()
            bad.append(f"Track.valid != (fail_mask == 0) for envs {envs[:6]}")
        details = {"bits": {n: int(program.fail_bits[n]) for n in own},
                   "own_fraction": float(np.mean((fm & own_mask) != 0))}
        message = ""
        if self.stage.initializer:
            # validity.default: the planner refuses any other stage that starts valid.fail.
            if np.any(fm & ~own_mask):
                bad.append("fail_mask holds bits the stage does not own")
            message = ("the stage initialises valid.fail: no pipeline without it to "
                       "compare with")
        else:
            other = _Snapshot("generator", self.stage.without(self.E, self.device,
                                                              self.seeds_a))
            fm0 = other.data["fail_mask"].astype(np.int64)
            diff = np.flatnonzero((fm & ~own_mask) != fm0).tolist()
            if diff:
                bad.append(f"the check changed bits it does not own (envs {diff[:6]}): "
                           f"OR only (1 << b.fail_bit()) into valid.fail")
            bad += self._foreign_bits(program, sid, own_mask)
        return ("validity.bits", not bad, _list_text(bad) if bad else message, details)

    def _foreign_bits(self, program, sid: str, own_mask: int) -> list:
        """Re-run only the check on a synthetic ``valid.fail`` input that holds bits it
        does not own (alternating patterns over bits 0-30) into a write buffer filled
        with ``-1``: every env must come out with exactly those bits besides its own
        (``fail_out = fail_in | own``). Restores both buffers afterwards."""
        from ..exec.context import LaunchContext  # noqa: PLC0415
        from ..stages.channels import FAIL  # noqa: PLC0415

        ids = list(program._stage_ids)
        i = ids.index(sid)
        fail_in = program.tap(FAIL, after=ids[i - 1])
        fail_out = program.tap(FAIL, after=sid)
        saved_in, saved_out = fail_in.numpy().copy(), fail_out.numpy().copy()
        pattern = np.where(np.arange(self.E) % 2 == 0, 0x55555555, 0x2AAAAAAA)
        pattern = pattern.astype(np.int64) & ~own_mask
        try:
            fail_in.assign(pattern.astype(np.int32))
            fail_out.fill_(-1)
            program._enqueues[i](LaunchContext.for_call(self.device))
            wp.synchronize_device(self.device)
            out = fail_out.numpy().astype(np.int64)
        finally:
            fail_in.assign(saved_in)
            fail_out.assign(saved_out)
            wp.synchronize_device(self.device)
        bad = []
        unwritten = np.flatnonzero(out == -1).tolist()
        if unwritten:
            bad.append(f"the check left valid.fail unwritten for envs {unwritten[:6]}: "
                       f"write every env (an env that passes gets fail_in[e])")
        dropped = np.flatnonzero((out != -1) & ((out & ~own_mask) != pattern)).tolist()
        if dropped:
            bad.append(f"the check dropped or changed the bits of earlier checks (envs "
                       f"{dropped[:6]}): write fail_in[e] | own, not own alone")
        return bad

    # -- 2-4. determinism, diversity, batch independence --------------------------------
    def determinism(self, snap: _Snapshot):
        twin = self.fresh(self.seeds_a)
        bad = [f"env {e} ({_fields_text(d)})" for e in range(self.E)
               if (d := _env_diff(snap, e, twin, e))]
        msg = ("two generators built with the same seeds differ: " + _list_text(bad) +
               ". Draw every random number from the env's seed.") if bad else ""
        return ("determinism", not bad, msg, {})

    def diversity(self, snap: _Snapshot):
        other = self.snap_b = self.fresh(self.seeds_b)
        # Only envs valid in both batches: a generator may reject an env by writing
        # NaN (or nothing) and 0 to out_valid, and two rejected envs can be alike.
        ok_a = snap.comparable()
        both = np.flatnonzero(ok_a & other.comparable()).tolist()
        mine = np.flatnonzero(ok_a).tolist()
        if not both and len(mine) < 2:
            return ("diversity", None, "skipped: no env is valid in both batches (see "
                    "output.closed / output.accepted)", {"compared": 0})
        bad, deltas = [], []
        for e in both:
            deltas.append(_raw_delta(snap, e, other, e))
            if not _env_diff(snap, e, other, e):
                bad.append(f"env {e} is identical for seeds {int(self.seeds_a[e])} and "
                           f"{int(self.seeds_b[e])}")
        groups: dict = {}
        for e in mine:
            groups.setdefault(snap.digest(e), []).append(e)
        for envs in groups.values():
            if len(envs) > 1:
                bad.append(f"envs {envs} (seeds {[int(self.seeds_a[e]) for e in envs]}) "
                           f"produced identical tracks")
        pairs = [_raw_delta(snap, i, snap, j) for k, i in enumerate(mine)
                 for j in mine[k + 1:]]
        details = {"compared": len(both),
                   "min_raw_delta": min(deltas) if deltas else math.inf,
                   "min_pair_delta": min(pairs) if pairs else math.inf}
        msg = ("different seeds must produce different tracks: " + _list_text(bad)) \
            if bad else ""
        return ("diversity", not bad, msg, details)

    def batch_independence(self, snap: _Snapshot):
        E, A, B = self.E, self.seeds_a, self.seeds_b
        bad = []
        # permuted: the other envs' seeds reversed around each env
        perm = self.fresh(A[::-1].copy())
        for e in range(E):
            d = _env_diff(snap, e, perm, E - 1 - e)
            if d:
                bad.append(f"env {e} (seed {int(A[e])}) changes when the other envs' seeds "
                           f"are permuted ({_fields_text(d)})")
        # inside a larger batch, shifted by one
        extra = B[:3] if E >= 3 else B
        big_seeds = np.concatenate([extra[:1], A, extra[1:]])
        big = self.fresh(big_seeds)
        for e in range(E):
            d = _env_diff(snap, e, big, e + 1)
            if d:
                bad.append(f"env {e} (seed {int(A[e])}) changes inside a batch of "
                           f"{big_seeds.size} ({_fields_text(d)})")
        # alone
        for e in sorted({0, E - 1}):
            alone = self.fresh(A[e:e + 1].copy())
            d = _env_diff(snap, e, alone, 0)
            if d:
                bad.append(f"env {e} (seed {int(A[e])}) changes when generated alone "
                           f"({_fields_text(d)})")
        msg = ("env e's track must depend on its own seed only: " + _list_text(bad)) \
            if bad else ""
        return ("batch_independence", not bad, msg, {})

    # -- 6. statelessness ------------------------------------------------------------------
    def statelessness(self, gen, snap: _Snapshot):
        bad = []
        for i in (1, 2):
            self.eager(gen)
            d = [f"env {e} ({_fields_text(x)})" for e in range(self.E)
                 if (x := _env_diff(snap, e, self.snapshot(gen), e))]
            if d:
                bad.append(f"generate() #{i} with unchanged seeds differs from the first "
                           f"batch: {_list_text(d, 2)}")
                break
        other = self.snap_b or self.fresh(self.seeds_b)
        _write_seeds(gen, self.seeds_b)
        self.eager(gen)
        now = self.snapshot(gen)
        d = [e for e in range(self.E) if _env_diff(other, e, now, e)]
        if d:
            bad.append(f"after a reseed, generate() differs from a fresh generator built "
                       f"with the new seeds (envs {d[:6]})")
        _write_seeds(gen, self.seeds_a)
        self.eager(gen)
        now = self.snapshot(gen)
        d = [e for e in range(self.E) if _env_diff(snap, e, now, e)]
        if d:
            bad.append(f"after reseeding away and back, generate() differs from the first "
                       f"batch (envs {d[:6]})")
        msg = ("hidden state carried between calls: " + _list_text(bad)) if bad else ""
        return ("statelessness", not bad, msg, {})

    # -- 5. hot path (CPU) --------------------------------------------------------------
    def hot_path(self, gen) -> list:
        eager = not self.capturable
        # A second instance built alike (constructed here: construction may allocate).
        twin = None if eager or self.skipped("hot_path.buffers") else \
            self.make(self.seeds_a)
        guard = HotPathGuard(modules=self.probe, strict=False)
        with guard:
            with LaunchTrace() as first:
                gen.generate()
            with LaunchTrace() as second:
                gen.generate()
        waived = ("waived: capturable=False (eager every call, may sync and allocate)")
        out = []

        def category(check: str, cat: str, what: str) -> None:
            if eager and check in _WAIVED_EAGER:
                out.append((check, None, waived, {}))
                return
            found = guard.of(cat)
            uniq = list(dict.fromkeys(str(v) for v in found))
            msg = (f"{len(found)} {what} in two generate() calls: {_list_text(uniq)}. "
                   f"{_HINTS[cat]}") if found else ""
            out.append((check, not found, msg, {"count": len(found)}))

        category("hot_path.host_sync", "host_sync", "host sync/readback(s)")
        category("hot_path.allocation", "allocation", "allocation(s)")
        category("hot_path.stream", "stream", "call(s) off the call's stream")
        name, passed, msg, det = out[-1]
        if not passed and not self.accepts_ctx:
            out[-1] = (name, passed, msg + " A legacy spec (accepts_ctx=False) runs its own "
                       "launches on Warp's current stream: set accepts_ctx=True and emit "
                       "through ctx.", det)
        category("hot_path.capture", "capture", "nested capture / graph launch call(s)")

        allocated = guard.allocated()
        owned = buffers_of(gen, root="gen")
        if eager:
            out.append(("hot_path.topology", None, waived, {}))
            out.append(("hot_path.buffers", None, waived, {}))
            return out
        shared = []
        if twin is not None:
            # Buffers shared by every instance (a module global, a closure of the spec,
            # a class attribute, ...) show up in both instances' launches, however
            # they are reached.
            other = HotPathGuard(host_sync=False, allocation=True, stream=False,
                                 capture=False, strict=False)
            with other, LaunchTrace() as third:
                twin.generate()
            allocated += other.allocated()
            temporary = BufferSet()
            for arr in allocated:
                temporary.add(arr, "<allocated>")
            mine = {p for t in (first, second) for r in t for _, p in r.arrays if p}
            shared = [(r, i, p) for r in third for i, p in r.arrays
                      if p in mine and temporary.locate(p) is None]
        diff = first.diff(second, buffers=owned, ignore=allocated)
        out.append(("hot_path.topology", not diff,
                    ("the second generate() emitted a different launch sequence than the "
                     f"first ({len(first)} vs {len(second)} launches/copies): "
                     + _list_text(diff, 3)
                     + ". A CUDA graph replays what it recorded: kernels, launch dims, "
                       "buffers and scalar arguments must not depend on a call counter, "
                       "host data or an allocation.") if diff else "",
                    {"launches": len(first)}))
        foreign = first.foreign_pointers(owned, ignore=allocated) + \
            second.foreign_pointers(owned, ignore=allocated)
        problems = []
        uniq = list(dict.fromkeys(f"{r.kernel} argument {i} (from {r.site})"
                                  for r, i, _ in foreign))
        if uniq:
            problems.append(f"{len(uniq)} launch argument(s) point outside the "
                            f"generator's own buffers (result, scratch, seeds, params): "
                            f"{_list_text(uniq)}")
        uniq = list(dict.fromkeys(f"{r.kernel} argument {i} (from {r.site})"
                                  for r, i, _ in shared))
        if uniq:
            problems.append(f"{len(uniq)} launch argument(s) use the same buffer in two "
                            f"generator instances: {_list_text(uniq)}")
        out.append(("hot_path.buffers", not problems,
                    ("; ".join(problems) + ". Allocate every buffer once per generator, "
                     "in alloc_scratch or build: a buffer held by a module global, a "
                     "closure or a class is shared by every generator instance.")
                    if problems else "",
                    {"buffers": len(owned), "shared": len(shared)}))
        return out

    # -- 7. CUDA ------------------------------------------------------------------------
    def capture_caller(self):
        if not self.capturable:
            return ("capture.caller", None,
                    "waived: registered capturable=False (refuses a caller's capture)", {})
        gen = self.make(self.seeds_a)
        status = assert_caller_capturable(gen, modes=self.modes)
        return ("capture.caller", True, ", ".join(f"{m}: {s}" for m, s in status.items()),
                dict(status))

    def capture_self_graph(self, snap: _Snapshot):
        gen = self.make(self.seeds_a)
        result = gen._track if self.snap_kind == "generator" else gen._gate_sequence
        bad = []
        _poison_raw(self.snap_kind, gen)          # as the reference snapshots
        gen.generate()
        if self.capturable and gen.graph is None:
            bad.append("generate() on CUDA did not capture the generator's own graph")
        if not self.capturable and gen.graph is not None:
            bad.append("a capturable=False generator captured a graph")

        def compare(ref: _Snapshot, label: str) -> None:
            now = self.snapshot(gen)
            d = [f"env {e} ({_fields_text(x)})" for e in range(self.E)
                 if (x := _env_diff(ref, e, now, e))]
            if d:
                bad.append(f"{label} differs from the eager batch: {_list_text(d, 2)}")

        compare(snap, "the first generate()")
        poison_all(result)
        gen.generate()
        compare(snap, "a replay into poisoned outputs")
        other = self.snap_b or self.fresh(self.seeds_b)
        _write_seeds(gen, self.seeds_b)
        poison_all(result)
        gen.generate()
        compare(other, "a replay after a reseed")
        return ("capture.self_graph", not bad, _list_text(bad), {})


# ---------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------

def _devices(devices) -> list:
    wp.init()
    if devices is None:
        out = ["cpu"]
        if wp.is_cuda_available():
            out.append("cuda:0")
        return out
    if isinstance(devices, str):
        devices = [devices]
    return [str(wp.get_device(d).alias) for d in devices]


def _top_package(module: str) -> str:
    return module.split(".")[0] if module else ""


def _check(kind: str, name: str, params, *, devices, num_envs: int, seeds,
           config: "Mapping | None", capture_modes, skip, modules) -> Report:
    from ..plugin import registry  # noqa: PLC0415

    info = registry.describe(kind, name)
    E = int(num_envs)
    if E < 2:
        raise ValueError(f"num_envs must be >= 2 (diversity and batch independence "
                         f"compare envs), got {num_envs!r}")
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) != 2:
        raise ValueError(f"seeds must be two base seeds (batch A, batch B), got {seeds!r}")
    seeds_a = _int32(seeds[0] + np.arange(E))
    seeds_b = _int32(seeds[1] + np.arange(E))
    if set(seeds_a.tolist()) & set(seeds_b.tolist()):
        raise ValueError(f"seeds={seeds!r}: the per-env seeds {seeds[0]}..{seeds[0] + E - 1}"
                         f" and {seeds[1]}..{seeds[1] + E - 1} overlap; pick base seeds at "
                         f"least num_envs apart")
    skip = frozenset(skip)
    known = (set(CHECKS[kind]) | {n.split(".")[0] for n in CHECKS[kind]}) - {"construct"}
    unknown = sorted(skip - known)
    if unknown:
        raise ValueError(f"unknown check name(s) in skip: {unknown}; known: "
                         f"{list(CHECKS[kind])}")
    modes = tuple(capture_modes)
    bad_modes = [m for m in modes if m not in CAPTURE_MODES]
    if bad_modes:
        raise ValueError(f"unknown capture mode(s) {bad_modes}; known: {CAPTURE_MODES}")
    overrides = dict(config or {})
    reserved = sorted(set(overrides) & {"generator", "device", "num_envs"})
    if reserved:
        raise ValueError(f"config= must not set {reserved} (the kit sets them)")
    if "generator_params" in overrides and params is not None:
        raise ValueError("pass the generator's params either as params= or as "
                         "config['generator_params'], not both")
    if isinstance(modules, str):
        modules = (modules,)
    probe = tuple(dict.fromkeys(("track_gen", _top_package(info.origin),
                                 *(str(m) for m in modules or ()))))
    results: list = []
    for dev in _devices(devices):
        results += _DeviceRun(kind, name, params, dev, E, seeds_a, seeds_b, overrides,
                              info, probe, modes, skip).run()
    return Report(kind, name, results)


def check_generator(name: str, params=None, *, devices=None, num_envs: int = 8,
                    seeds: Sequence[int] = (11, 1009), config: "Mapping | None" = None,
                    capture_modes: Iterable[str] = CAPTURE_MODES,
                    skip: Iterable[str] = (), modules: Iterable[str] = ()) -> Report:
    """Run every conformance check (:data:`CHECKS` ``["generator"]``) on the registered
    track generator ``name``.

    Args:
        name: the registered generator name (``registry.available("generator")``).
        params: its ``generator_params`` (a Params instance or a mapping of its fields);
            ``None``: the defaults.
        devices: Warp devices to run on (default: ``"cpu"``, plus ``"cuda:0"`` when Warp
            sees a CUDA device). The ``hot_path.*`` checks run on CPU devices, the
            ``capture.*`` checks on CUDA devices.
        num_envs: the batch size ``E`` (>= 2).
        seeds: two base seeds; batch A seeds env ``e`` with ``seeds[0] + e``, batch B
            (the "different seeds" batch) with ``seeds[1] + e``. They must not overlap.
        config: extra ``TrackGenConfig`` fields (e.g. ``{"num_points": 64}`` to keep a
            slow generator fast); ``generator``, ``device`` and ``num_envs`` are set by
            the kit.
        capture_modes: the modes of ``capture.caller`` (see
            :func:`~track_gen.testing.capture.assert_caller_capturable`).
        skip: check names (or a prefix such as ``"hot_path"``) to report as skipped.
        modules: more dotted module prefixes for ``hot_path.stream`` to probe, besides
            ``track_gen`` and the top-level package of the module defining the
            generator (``registry.describe(kind, name).origin``), e.g. a helper package
            of yours with another top-level name.

    Returns:
        A :class:`Report`; call ``.raise_for_failures()`` to fail on any failed check.

    Raises:
        ValueError: invalid arguments (unknown check names, overlapping seeds, ...).
        Exceptions of ``registry.describe`` for an unknown or quarantined name
        (``PluginLoadError`` carries the plugin's load error).
    """
    return _check("generator", name, params, devices=devices, num_envs=num_envs,
                  seeds=seeds, config=config, capture_modes=capture_modes, skip=skip,
                  modules=modules)


def check_gate_generator(name: str, params=None, *, devices=None, num_envs: int = 8,
                         seeds: Sequence[int] = (11, 1009),
                         config: "Mapping | None" = None,
                         capture_modes: Iterable[str] = CAPTURE_MODES,
                         skip: Iterable[str] = (), modules: Iterable[str] = ()) -> Report:
    """As :func:`check_generator`, for the registered gate generator ``name``
    (:data:`CHECKS` ``["gate_generator"]``). ``config`` holds extra ``GateGenConfig``
    fields; without a ``gate_ordering`` the kit uses ``"ccw"`` when the generator
    supports it, else its first supported ordering."""
    return _check("gate_generator", name, params, devices=devices, num_envs=num_envs,
                  seeds=seeds, config=config, capture_modes=capture_modes, skip=skip,
                  modules=modules)


# ---------------------------------------------------------------------------------------
# Track-pipeline stages
# ---------------------------------------------------------------------------------------

class _StageUnderTest:
    """The pipelines :func:`check_stage` builds around one stage."""

    def __init__(self, kind: str, name: str, params, upstream: Mapping, cls) -> None:
        from ..pipeline.spec import StageRef  # noqa: PLC0415
        from ..stages.channels import FAIL  # noqa: PLC0415

        self.kind, self.name, self.params = kind, name, params
        self.upstream = dict(upstream)
        # A validity stage that starts valid.fail (writes it without reading it): the
        # planner admits only validity.default there (a plugin one fails construct).
        self.initializer = kind == "validity" and FAIL in dict(cls.writes).values() \
            and FAIL not in dict(cls.reads).values()
        if kind != "validity":
            self.stage_id = kind
            self.ref = StageRef(kind, name, params=params)
        elif self.initializer:
            self.stage_id = "validity"
            self.ref = StageRef(kind, name, params=params)
        else:
            self.stage_id = name
            self.ref = StageRef(kind, name, params=params, id=name)

    def _preset(self, E: int, device: str, slot):
        from ..pipeline import presets  # noqa: PLC0415

        kw = dict(self.upstream)
        if self.kind != "validity":
            kw[self.kind] = slot
        elif self.initializer:
            kw["checks"] = () if slot is None else (slot,)
        else:
            checks = tuple(kw.get("checks", ("default",)))
            kw["checks"] = checks if slot is None else (*checks, slot)
        return presets.track(num_envs=E, device=device, **kw)

    def spec(self, E: int, device: str):
        """The pipeline holding the stage under test."""
        return self._preset(E, device, self.ref)

    def _generator(self, spec, seeds: np.ndarray, device: str):
        from .._src.rng_utils import PerEnvSeededRNG  # noqa: PLC0415
        from .._src.track_generator import TrackGenerator  # noqa: PLC0415

        E = int(seeds.shape[0])
        rng = PerEnvSeededRNG(seeds=wp.array(_int32(seeds), dtype=wp.int32,
                                             device=device), num_envs=E, device=device)
        gen = TrackGenerator(spec, rng)
        gen.enqueue()
        wp.synchronize_device(device)
        return gen

    def reference(self, E: int, device: str, neutral: str, seeds=None):
        """A generator of the same pipeline with the slot's neutral built-in."""
        seeds = np.arange(E) if seeds is None else seeds
        return self._generator(self._preset(E, device, neutral), np.asarray(seeds),
                               device)

    def without(self, E: int, device: str, seeds: np.ndarray):
        """A generator of the same pipeline without the validity check under test."""
        return self._generator(self._preset(E, device, None), seeds, device)


def check_stage(kind: str, name: str, params=None, *, upstream: "Mapping | None" = None,
                devices=None, num_envs: int = 8, seeds: Sequence[int] = (11, 1009),
                capture_modes: Iterable[str] = CAPTURE_MODES, skip: Iterable[str] = (),
                modules: Iterable[str] = ()) -> Report:
    """Run every conformance check (:data:`CHECKS` ``[kind]``) on the registered
    ``relax`` / ``width`` / ``elevation`` / ``validity`` stage ``name``, inside the
    default track pipeline (:func:`track_gen.pipeline.presets.track`).

    The stage takes its slot (``relax=``, ``width=``, ``elevation=``); a validity check
    runs after the default checks (``upstream["checks"]``, default ``("default",)``)
    with id ``name``, and a validity stage that initialises ``valid.fail`` (writes it
    without reading it) replaces them.

    Args:
        kind: ``"relax"``, ``"width"``, ``"elevation"`` or ``"validity"``.
        name: the registered stage name (``registry.available(kind)``).
        params: its params (a Params instance or a mapping of its fields); ``None``:
            the defaults.
        upstream: more :func:`~track_gen.pipeline.presets.track` keyword arguments
            (``generator``, ``generator_params``, ``num_points``, ``n_max``,
            ``half_width``, the other kinds' slots, ``checks`` (the checks BEFORE a
            validity check under test), ``on_invalid``). ``num_envs``, ``device`` and
            the slot under test are set by the kit.
        devices, num_envs, seeds, capture_modes, skip, modules: as
            :func:`check_generator`.

    Returns:
        A :class:`Report`; call ``.raise_for_failures()`` to fail on any failed check.

    Raises:
        ValueError: invalid arguments (an unknown kind, a reserved ``upstream`` key,
            unknown check names, overlapping seeds, ...).
        Exceptions of ``registry.describe`` for an unknown or quarantined name
        (``PluginLoadError`` carries the plugin's load error).
    """
    from ..plugin import registry  # noqa: PLC0415

    if kind not in STAGE_KINDS:
        raise ValueError(f"kind must be one of {list(STAGE_KINDS)}, got {kind!r}")
    info = registry.describe(kind, name)
    upstream = dict(upstream or {})
    reserved = {"num_envs", "device"} | ({kind} if kind != "validity" else set())
    bad = sorted(set(upstream) & reserved)
    if bad:
        raise ValueError(f"upstream= must not set {bad} (the kit sets them)")
    E = int(num_envs)
    if E < 2:
        raise ValueError(f"num_envs must be >= 2 (batch independence compares envs), got "
                         f"{num_envs!r}")
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) != 2:
        raise ValueError(f"seeds must be two base seeds (batch A, batch B), got {seeds!r}")
    seeds_a = _int32(seeds[0] + np.arange(E))
    seeds_b = _int32(seeds[1] + np.arange(E))
    if set(seeds_a.tolist()) & set(seeds_b.tolist()):
        raise ValueError(f"seeds={seeds!r}: the per-env seeds {seeds[0]}..{seeds[0] + E - 1}"
                         f" and {seeds[1]}..{seeds[1] + E - 1} overlap; pick base seeds at "
                         f"least num_envs apart")
    skip = frozenset(skip)
    known = (set(CHECKS[kind]) | {n.split(".")[0] for n in CHECKS[kind]}) - {"construct"}
    unknown = sorted(skip - known)
    if unknown:
        raise ValueError(f"unknown check name(s) in skip: {unknown}; known: "
                         f"{list(CHECKS[kind])}")
    modes = tuple(capture_modes)
    bad_modes = [m for m in modes if m not in CAPTURE_MODES]
    if bad_modes:
        raise ValueError(f"unknown capture mode(s) {bad_modes}; known: {CAPTURE_MODES}")
    if isinstance(modules, str):
        modules = (modules,)
    probe = tuple(dict.fromkeys(("track_gen", _top_package(info.origin),
                                 *(str(m) for m in modules or ()))))
    stage = _StageUnderTest(kind, name, params, upstream, registry.get(kind, name))
    results: list = []
    for dev in _devices(devices):
        results += _DeviceRun(kind, name, params, dev, E, seeds_a, seeds_b, {}, info,
                              probe, modes, skip, stage=stage).run()
    return Report(kind, name, results)
