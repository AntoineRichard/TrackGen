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
    and every ``Track`` field have the documented shape and dtype; flags are 0 or 1; the
    result's ``layout`` is the Program's (canonical device) and ``result.validate()``
    passes (roadmap PR13b).
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
``output.view`` (roadmap PR13b; every kind)
    The unit's pipeline built with ``out=`` the middle view ``big.view(1, E + 1)`` of a
    byte-poisoned result of ``E + 2`` envs (with the kit's explicit seeds) writes exactly
    the bytes of the standalone batch into the view, after construction and after a
    masked advance, and leaves every byte outside it untouched: a kernel that indexes
    past ``[0, E)`` of a result array (or reads a result buffer it did not write) fails
    here. ``check_stage`` builds the stage's composed pipeline the same way.
``attributes.written`` / ``attributes.carried`` (roadmap PR13a)
    For a generator that writes per-point attributes (``raw.z``, ``raw.half_width``,
    ``raw.<pkg>.<name>``; a gate generator ``anchors.<pkg>.<name>``; skipped "no
    attributes" otherwise): every raw slot of every attribute is written for every env
    the generator marks valid (a gate generator: every slot before ``anchors.count``;
    the raw buffers are poisoned like the centerline), and every EXPORTED result
    attribute (``Track.attributes`` / ``GateSequence.attributes``) holds its padding
    (NaN / -1) beyond ``count[e]`` and, on the real points of every env the generator
    marked valid, finite values; a ``linear`` attribute lies within the env's raw
    ``[min, max]`` per component (slack ``4 * spacing(float32(max|v|))``), a
    ``nearest`` / ``previous`` / int32 one (or one carried by index) in the env's raw
    value set, an ``angle`` one in ``[-pi, pi]``.
``determinism``
    Two fresh generators built with the same seeds produce bitwise identical batches
    (raw output and every result field, NaN padding included).
``diversity``
    Different seeds produce different tracks: no env repeats under a disjoint set of
    seeds, and no two envs of one batch are identical (envs valid in both batches only:
    two rejected envs may legitimately be alike). Reported skipped for a generator that
    declares ``seeded = False`` (an ingestion generator: its output does not depend on
    the seed).
``batch_independence``
    Env ``e``'s track depends on its own seed only: it is identical alone (``E = 1``),
    inside a larger batch, and when the other envs' seeds are permuted.
``statelessness``
    ``generate()`` twice with unchanged seeds reproduces the first batch, a reseed
    reproduces a fresh generator with the new seeds, and reseeding back reproduces the
    first batch (no hidden state carried between calls).
``params.per_env`` (roadmap PR13c; every kind)
    For the unit's stage (the generator, or the stage under test) with dynamic fields
    that no randomizer draws (skipped "no dynamic parameters" otherwise): every such
    field is set to a per-env ``[E]`` array (float32: ``d + s * 0.25 * span * (e + 1) /
    E``, clamped to the declared bounds, toward the farther bound, ``span = hi - lo`` or,
    with a bound missing, ``max(|d|, 0.1)``: away from a field's one bound, alternating
    directions for fields without bounds; int32: ``d + (e % 3) - 1``, bool:
    ``(d + e) % 2``), one generation (eager, same seeds);
    then, for each env ``e``, every field is set to env ``e``'s value as a scalar and
    env ``e`` of that generation must equal env ``e`` of the per-env one, bitwise (raw
    output and every result field). The live arrays (and the stage's latched copies)
    must read back what was set: a stage reads ``param[e]``, never ``param[0]`` or a
    host copy, and never writes its param arrays. The fields are restored afterwards.
``bounded.absorbing``
    For a unit whose ``enqueue`` runs a ``ctx.bounded_loop`` with ``active=`` (seen
    while the unit runs, whatever it declares; skipped "not applicable" otherwise):
    with every ``active`` flag poisoned to 0 first, an eager generation with the host
    poll lowering (a), which stops at the first window whose check reads 0, writes the
    same bytes (raw output, result, the stage's write channels) as one with every
    window unrolled (c), for batch A and batch B. Fails for a non-absorbing loop (an
    iteration after ``active[0]`` reads 0 changes an output) and for a body that does
    not write ``active`` before the first check. Skipped "not exercised" when no
    window was skipped for the kit's seeds.
``bounded.lowering`` (CUDA)
    For the same units: the eager default (c), the forced host poll (a), a Warp capture
    with conditional windows (b; skipped without conditional-node support) and the same
    capture unrolled (c), replayed into poisoned outputs, before and after a reseed,
    are bitwise equal.
``host.require_caller_capturable`` (CPU)
    For a pipeline with HOST stages (skipped "not applicable" otherwise): its spec with
    ``require_caller_capturable=True`` is refused at build with a ``PlanError`` naming
    every HOST stage.
``hot_path.host_sync`` / ``hot_path.allocation`` / ``hot_path.stream`` / ``hot_path.capture`` (CPU)
    Two ``generate()`` calls under :func:`~track_gen.testing.contract.hot_path_guard`,
    every ``ctx.bounded_loop`` unrolled (the superset of what any lowering emits):
    no host sync or readback, no allocation, every launch / copy from track_gen or the
    plugin's package passes ``stream=`` (no ``assign()`` / ``zero_()`` / ``fill_()``),
    no nested capture or graph launch. On CPU the stream check sees that ``stream=`` is
    passed, not which stream (``stream=None`` is what a CPU ``LaunchContext`` passes):
    a launch on the wrong stream is caught by ``capture.caller`` on CUDA. A unit with a
    bounded loop also runs once with the default lowering: its host polls are the
    sanctioned readback (``details["sanctioned"]``), any other one fails
    ``hot_path.host_sync``. A HOST stage's ``ctx.host()`` operations are sanctioned too
    (``details["host_ops"]``); its own syncs, readbacks and allocations fail.
``hot_path.topology`` (CPU)
    The two calls emit the same launch sequence (:class:`~track_gen.testing.trace.LaunchTrace`):
    kernels, dimensions, buffers and scalar arguments.
``hot_path.buffers`` (CPU)
    Every pointer launched on belongs to the generator (its result, scratch, seed and
    parameter buffers; see :func:`~track_gen.testing.trace.buffers_of`), and a second
    generator built alike launches on none of the first one's buffers.
``capture.caller`` (CUDA)
    :func:`~track_gen.testing.capture.assert_caller_capturable`: recorded into a caller's
    capture, replays equal the eager batch, before and after a reseed. A pipeline with
    HOST stages must instead refuse: ``generate(stream=s, advance=True)`` inside a
    caller's Warp capture raises ``NotCapturableError`` naming every HOST stage before
    any launch, and the caller's graph replays its own work only (outputs, seeds and
    epochs untouched).
``capture.self_graph`` (CUDA)
    ``generate()`` captures the generator's own graph (a pipeline with HOST stages: one
    graph per non-HOST segment, none at the HOST ones, ``graph`` stays ``None``); its
    replays (into poisoned outputs, before and after a reseed) equal the eager batch.

A gate generator runs the same checks on its ``GateSequence`` (raw output: the vec2f
anchors before ``count[e]``: ``anchors.xy``, ``[E*A]``, for a Stage generator; the
``[E*G]`` gates a legacy spec writes), with ``output.accepted`` (valid fraction > 0) in
place of ``output.closed`` and ``min_gates <= count[e] <= max_gates`` for valid envs; a
Stage generator's ``anchors.count`` must also lie within its ``capacity()``
(``min_gates <= anchors.count[e] <= gates``) for every env.

A pipeline with HOST stages (run eagerly, between the graph launches of the rest of the
pipeline on CUDA) waives ``hot_path.topology`` and ``hot_path.buffers`` (reported as
skipped: its launches may depend on host data); ``hot_path.host_sync`` and
``hot_path.allocation`` still run, with the HOST stages' ``ctx.host()`` operations
sanctioned. A legacy ``capturable=False`` generator spec (which cannot use ``ctx.host()``)
also waives ``hot_path.host_sync`` and ``hot_path.allocation``. ``capture.caller`` then
checks that the pipeline refuses a caller's capture and ``capture.self_graph`` its
segmented self-graphs.

:func:`check_stage` runs a ``relax`` / ``width`` / ``elevation`` / ``validity`` stage
inside the default track pipeline (:func:`track_gen.pipeline.presets.track`, the stage
in its slot; a validity check after ``validity.default``), and an ``order`` /
``stations`` stage (or a validity check that reads gate channels) inside the default
gate pipeline (:func:`track_gen.pipeline.presets.gates`: an order on the stations of the
bezier track generator, stations on its raw loop, ``relaxed_stations`` for a stage that
reads ``cl.*``; a gate check after ``validity.gate_default``; with ``pipeline="gates"``,
an ``elevation`` stage on the bezier gate generator's gates, its ports rewired to
``gates.*``), through the same machinery, with these checks (no ``diversity`` /
``output.closed``: they are the generator's business):

``output.accepted`` (relax / width / elevation / order / stations)
    The pipeline accepts some env (valid fraction > 0). When it accepts none, the same
    pipeline with the slot's neutral built-in (``relax.none``, ``width.constant``,
    ``elevation.flat``, ``order.raw``, ``stations.raw``) is run: if it accepts none
    either, the check is skipped (the stage did not reject them).
``stage.isolation``
    After a generation, re-running ONLY the stage's ``enqueue`` on the same inputs
    leaves the versions it read unchanged (it never writes a read port) and reproduces
    the versions it wrote (no stale scratch, no dependency on its own previous output).
``validity.bits`` (validity)
    The stage owns at least one fail bit (``StageBuilder.fail_bit``), sets no bit it
    does not own (the ``fail_mask`` of the pipeline without it, OR its own bits), and
    ``Track.valid == (fail_mask == 0)``.
``width.values`` (width)
    The stage's version of ``cl.half_width`` (``tap("cl.half_width", after=<id>)``) is
    finite and ``>= 0`` on every real point of every env the generator marks valid
    (floor violations are validity's job, bit 4). A width stage that reads
    ``cl.half_width`` (a rewrite) runs right after ``width.constant``; a required read
    port (``"cl.*"``) is wired with ``ports=``.

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

from ..exec.context import _force_lowering, _probe_loops
from .capture import CAPTURE_MODES, _write_seeds, assert_caller_capturable, poison_all
from .capture import recording, result_arrays, side_stream
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

# The BOUNDED / HOST checks (roadmap PR12): not applicable to most units (skipped).
_TIERS = ("bounded.absorbing", "bounded.lowering", "host.require_caller_capturable")
_COMMON = ("determinism", "diversity", "batch_independence", "statelessness",
           "params.per_env") + _TIERS + (
           "hot_path.host_sync", "hot_path.allocation", "hot_path.stream",
           "hot_path.capture", "hot_path.topology", "hot_path.buffers",
           "capture.caller", "capture.self_graph")
_STAGE_COMMON = ("determinism", "batch_independence", "statelessness",
                 "params.per_env") + _TIERS + (
                 "hot_path.host_sync", "hot_path.allocation", "hot_path.stream",
                 "hot_path.capture", "hot_path.topology", "hot_path.buffers",
                 "capture.caller", "capture.self_graph")
_STAGE_OUTPUT = ("construct", "output.shape", "output.finite", "output.padding",
                 "output.count")
#: The pipeline stage kinds :func:`check_stage` runs.
STAGE_KINDS = ("relax", "width", "elevation", "validity", "order", "stations")
# The neutral built-in of each slot (the output.accepted reference).
_NEUTRAL = {"relax": "none", "width": "constant", "elevation": "flat", "order": "raw",
            "stations": "raw"}

#: The check names of each registry kind, in run order.
_ATTRIBUTES = ("attributes.written", "attributes.carried")
CHECKS: Mapping[str, tuple] = types.MappingProxyType({
    "generator": ("construct", "output.shape", "output.finite", "output.padding",
                  "output.count", "output.closed", "output.view") + _ATTRIBUTES + _COMMON,
    "gate_generator": ("construct", "output.shape", "output.finite", "output.padding",
                       "output.count", "output.accepted", "output.view") + _ATTRIBUTES
    + _COMMON,
    **{k: _STAGE_OUTPUT + ("output.accepted", "output.view", "stage.isolation")
       + _STAGE_COMMON for k in ("relax", "elevation", "order", "stations")},
    "width": _STAGE_OUTPUT + ("output.accepted", "output.view", "stage.isolation",
                              "width.values") + _STAGE_COMMON,
    "validity": _STAGE_OUTPUT + ("output.view", "stage.isolation", "validity.bits")
    + _STAGE_COMMON,
})

_HOT = ("hot_path.host_sync", "hot_path.allocation", "hot_path.stream",
        "hot_path.capture", "hot_path.topology", "hot_path.buffers")
_CAPTURE = ("capture.caller", "capture.self_graph")
# Run on one device kind only.
_CUDA_ONLY = ("bounded.lowering",)
_CPU_ONLY = ("host.require_caller_capturable",)
# Waived for a pipeline with HOST stages (eager every call: its launches may depend on
# host data), and also the sync / allocation checks for a legacy capturable=False
# generator spec (it cannot reach the host through ctx.host()).
_WAIVED_HOST = ("hot_path.topology", "hot_path.buffers")
_WAIVED_LEGACY = ("hot_path.host_sync", "hot_path.allocation") + _WAIVED_HOST
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
                "tangent": wp.vec3f, "normal": wp.vec3f, "arclen": wp.float32,
                "half_width": wp.float32}
_TRACK_ENV = {"length": wp.float32, "valid": wp.int32, "count": wp.int32,
              "winding": wp.float32, "truncated": wp.int32, "generation": wp.int32,
              "fail_mask": wp.int32}
_GATE_POINT = {"position": wp.vec3f, "tangent": wp.vec3f, "forward": wp.vec3f,
               "orientation": wp.quatf, "half_size": wp.float32, "left": wp.vec3f,
               "right": wp.vec3f}
_GATE_ENV = {"valid": wp.int32, "count": wp.int32, "generation": wp.int32,
             "fail_mask": wp.int32}
_OPTIONAL = {"truncated", "generation", "fail_mask", "half_width"}
# The prefix of a raw attribute buffer in a snapshot (``"raw.attr:raw.acme.grip"``).
_RAW_ATTR = "raw.attr:"


def _anchor_capacity(gen) -> "dict | None":
    """A gate generator Stage's ``capacity()`` (``None`` for a legacy spec or a stations
    source)."""
    from ..plugin.params import _static_view  # noqa: PLC0415
    from ..plugin.stage import _is_stage  # noqa: PLC0415

    binding = gen.program._binding
    if binding.kind != "gate_generator" or not _is_stage(binding.spec):
        return None
    cap = dict(binding.spec().capacity(_static_view(binding.params), binding.layout))
    return {"gates": int(cap["gates"]), "min_gates": int(cap.get("min_gates", 0))}


class _Snapshot:
    def __init__(self, kind: str, gen) -> None:
        cfg = gen._config
        self.kind = kind
        self.E = E = int(gen.num_envs)
        self.capacity = None
        self.anchor_count = None
        if kind == "generator":
            result = gen._track
            raw = {"raw.centerline": gen._scratch.gen_centerline,
                   "raw.valid": gen._scratch.gen_valid}
            self.stride = int(cfg.N_max)
            self.raw_stride = int(raw["raw.centerline"].shape[0]) // E
            self.point, self.env_fields = _TRACK_POINT, _TRACK_ENV
            self.min_count = 3
        else:
            result = gen._gate_sequence
            raw = {"raw.position": gen._raw_anchors}
            self.stride = int(gen.max_gates)
            self.raw_stride = int(gen._raw_anchors.shape[0]) // E
            self.point, self.env_fields = _GATE_POINT, _GATE_ENV
            # Always a GateGenConfig (validity.gate_default's min_gates for a spec).
            self.min_count = int(cfg.min_gates)
            self.capacity = _anchor_capacity(gen)
            if self.capacity is not None:
                self.anchor_count = gen.program.tap("anchors.count").numpy().astype(
                    np.int64)
        raw.update({_RAW_ATTR + ch: arr for ch, arr in _attribute_buffers(kind, gen)})
        arrays = {**raw, **result_arrays(result)}
        self.meta = {k: (tuple(int(s) for s in a.shape), a.dtype) for k, a in arrays.items()}
        self.data = {k: a.numpy().copy() for k, a in arrays.items()}
        self.count = self.data["count"].astype(np.int64)
        self.valid = self.data["valid"].astype(np.int64)

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

    def raw_count(self, e: int) -> int:
        """The real raw gate anchors of env ``e`` (``count[e]`` clamped to the raw
        stride)."""
        return int(min(max(self.count[e], 0), self.raw_stride))

    def env(self, e: int) -> dict:
        """field -> bytes of env ``e`` (gate raw anchors: before ``count[e]`` only)."""
        out = {}
        for name, a in self.data.items():
            if name == "raw.position":
                out[name] = self.rows(name, e, 0, self.raw_count(e)).tobytes()
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


def _attribute_buffers(kind: str, gen) -> list:
    """``(channel, array)`` of a generator's per-point attribute buffers (roadmap
    PR13a): a centerline generator's ``raw.z`` / ``raw.half_width`` /
    ``raw.<pkg>.<name>``, a gate generator's ``anchors.<pkg>.<name>`` (its writes, in
    declaration order); ``[]`` for a stage's pipeline or a legacy spec."""
    from ..plugin.stage import _is_raw_attribute, _is_stage  # noqa: PLC0415
    from ..stages.channels import is_plugin_channel  # noqa: PLC0415

    program = getattr(gen, "program", None)
    if program is None:
        return []
    spec = program._binding.spec
    if not _is_stage(spec):
        return []
    out = []
    for ch in dict(spec.writes).values():
        ok = _is_raw_attribute(ch) if kind == "generator" else (
            is_plugin_channel(ch) and ch.startswith("anchors."))
        if ok:
            out.append((ch, program.tap(ch, after="generator")))
    return out


def _seeded(spec) -> bool:
    """Whether a generator's output depends on its seed (``Stage.seeded``; a legacy spec:
    ``True``)."""
    return bool(getattr(spec, "seeded", True)) if spec is not None else True


def _layout_problems(gen) -> "list[str]":
    """``output.shape``'s layout rules (roadmap PR13b): the result's ``layout`` is the
    Program's (canonical device) and the result validates."""
    from .._src.types import _fmt_layout, _layout_key  # noqa: PLC0415

    program = getattr(gen, "program", None)
    result = getattr(program, "result", None)
    if result is None or not hasattr(result, "validate"):
        return []
    lay = getattr(result, "layout", None)
    problems = []
    if lay is None:
        problems.append("the result has no layout (the planner sets it)")
    elif _layout_key(lay) != _layout_key(program.layout):
        problems.append(f"the result's layout {_fmt_layout(lay)} is not the Program's "
                        f"{_fmt_layout(program.layout)}")
    try:
        result.validate()
    except (TypeError, ValueError) as exc:
        problems.append(f"the result does not validate: {exc}")
    return problems


def _writes_raw_valid(gen) -> bool:
    """Whether a centerline generator writes its valid flag on every call: a legacy
    ``GeneratorSpec`` always does (``out_valid``); a Stage generator only when its build
    wrote ``gen.valid`` (an optional write: otherwise the binding fills it with 1 once,
    at build, and the kit must not poison it)."""
    program = getattr(gen, "program", None)
    built = getattr(getattr(program, "_binding", None), "pipeline_spec", None)
    built = getattr(built, "_built", None)
    if built is None:
        return True
    return "gen.valid" in built.builder._written_channels


def _raw_buffers(kind: str, gen) -> list:
    """``(array, is_flag)`` of a producer's raw output buffers (the attribute buffers
    included: poisoned like the centerline; ``gen.valid`` only when the generator
    writes it)."""
    attrs = [(arr, False) for _, arr in _attribute_buffers(kind, gen)]
    if kind == "generator":
        valid = [(gen._scratch.gen_valid, True)] if _writes_raw_valid(gen) else []
        return [(gen._scratch.gen_centerline, False), *valid, *attrs]
    return [(gen._raw_anchors, False), *attrs]


def _poison_raw(kind: str, gen) -> None:
    """Fill the raw output with the kit's poison (outside any guard) and wait."""
    devices = set()
    for arr, is_flag in _raw_buffers(kind, gen):
        n = int(arr.shape[0])
        if is_flag:
            host = np.full(n, _FLAG_POISON, dtype=np.int32)
        elif arr.dtype is wp.int32:
            host = np.full(n, _POINT_POISON_BITS, dtype=np.uint32).view(np.int32)
        else:
            width = int(getattr(arr.dtype, "_length_", 1) or 1)
            host = np.full((n, width), _POINT_POISON_BITS, dtype=np.uint32).view(np.float32)
            if width == 1:
                host = host.reshape(n)
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
        ca, cb = a.raw_count(ea), b.raw_count(eb)
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


def _undrawn_dynamic(sp) -> list:
    """The dynamic fields of a ``StageParams`` that nothing draws (no randomizer, not
    the stage itself), in declaration order; ``[]`` without a Params instance."""
    from ..plugin.params import _fields  # noqa: PLC0415

    if sp is None or sp.params is None:
        return []
    drawn = set(sp.drawn)
    return [f for f in _fields(type(sp.params))
            if f.kind == "dynamic" and f.name not in drawn and f.name in sp]


def _per_env_values(fields, params, E: int) -> dict:
    """``field -> [E]`` values of the ``params.per_env`` check (see :data:`CHECKS`):
    every env differs, each value inside the field's declared bounds."""
    out, unbounded = {}, 0
    for f in fields:
        d = getattr(params, f.name)
        if f.dtype is wp.int32:
            d = int(d)
            if f.py_type is bool:
                v = np.array([(d + e) % 2 for e in range(E)], np.int64)
            else:
                v = np.array([d + (e % 3) - 1 for e in range(E)], np.int64)
            lo = -2**31 if f.lo is None else int(math.ceil(float(f.lo)))
            hi = 2**31 - 1 if f.hi is None else int(math.floor(float(f.hi)))
            out[f.name] = np.clip(v, lo, hi).astype(np.int32)
            continue
        d = float(d)
        lo = None if f.lo is None else float(f.lo)
        hi = None if f.hi is None else float(f.hi)
        if lo is not None and hi is not None:
            span = hi - lo
            sign = 1.0 if hi - d >= d - lo else -1.0
        else:
            span = max(abs(d), 0.1)
            if lo is not None:
                sign = 1.0
            elif hi is not None:
                sign = -1.0
            else:
                sign = -1.0 if unbounded % 2 == 0 else 1.0
                unbounded += 1
        v = d + sign * 0.25 * span * (np.arange(E, dtype=np.float64) + 1.0) / E
        if lo is not None:
            v = np.maximum(v, lo)
        if hi is not None:
            v = np.minimum(v, hi)
        out[f.name] = v.astype(np.float32)
    return out


def _work_arrays(program, live: Mapping) -> dict:
    """``field -> the stage's latched working copy`` of each live param array (the
    Program's latch pairs)."""
    pairs = getattr(program, "_pairs", ())
    out = {}
    for n, arr in live.items():
        for lv, wk in pairs:
            if lv is arr:
                out[n] = wk
                break
    return out


def _written_params(sp, work: Mapping, values: Mapping) -> list:
    """The fields whose live array or latched copy no longer holds ``values``."""
    bad = []
    for n, v in values.items():
        want = np.ascontiguousarray(v).tobytes()
        arrays = [sp[n]] + ([work[n]] if n in work else [])
        if any(a.numpy().tobytes() != want for a in arrays):
            bad.append(n)
    return bad


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
        # What a snapshot reads: a stage runs inside a TrackGenerator (a gate stage inside
        # a GateGenerator).
        self.snap_kind = kind if stage is None else (
            "gate_generator" if stage.gate else "generator")
        self.is_cuda = wp.get_device(device).is_cuda
        self.results: list = []
        self.snap_b: "_Snapshot | None" = None
        self.capturable = bool(info.capturable)
        self.accepts_ctx = bool(info.accepts_ctx)
        # The unit's bounded loops with active= (seen during its first generation).
        self.loops: list = []
        self.gen_spec = None            # the bound generator spec (seeded=, ...)

    # -- construction ------------------------------------------------------------
    def make(self, seeds: np.ndarray, out=None):
        """A generator of the unit for ``seeds`` (``out=``: writing into a caller-owned
        result, the ``output.view`` check)."""
        from .._src.rng_utils import PerEnvSeededRNG  # noqa: PLC0415

        dev = self.device
        E = int(seeds.shape[0])
        rng = PerEnvSeededRNG(seeds=wp.array(_int32(seeds), dtype=wp.int32, device=dev),
                              num_envs=E, device=dev)
        if self.stage is not None:
            return self.stage.facade()(self.stage.spec(E, dev), rng, out=out)
        kw = dict(self.overrides)
        if self.params is not None:
            kw["generator_params"] = self.params
        if self.kind == "generator":
            from .._src.track_generator import TrackGenerator  # noqa: PLC0415
            from .._src.types import TrackGenConfig  # noqa: PLC0415

            return TrackGenerator(TrackGenConfig(generator=self.name, device=dev,
                                                 num_envs=E, **kw), rng, out=out)
        from .._src.gate_generator import GateGenerator  # noqa: PLC0415
        from .._src.types import GateGenConfig  # noqa: PLC0415

        if "gate_ordering" not in kw:
            kw["gate_ordering"] = _default_ordering(self.name)
        return GateGenerator(GateGenConfig(generator=self.name, device=dev, num_envs=E,
                                           **kw), rng, out=out)

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
    def unit_id(self) -> str:
        """The Program slot of the unit under test (``"generator"`` for a generator)."""
        return "generator" if self.stage is None else self.stage.stage_id

    def run(self) -> list:
        names = [n for n in CHECKS[self.kind]
                 if not (n in _HOT and self.is_cuda) and not (n in _CAPTURE and not self.is_cuda)
                 and not (n in _CUDA_ONLY and not self.is_cuda)
                 and not (n in _CPU_ONLY and self.is_cuda)]
        try:
            gen = self.make(self.seeds_a)
            with _probe_loops() as loops:
                snap = self.poisoned(gen)
        except Exception as exc:  # noqa: BLE001
            self.add("construct", False, _exc_text(exc))
            for n in names[1:]:
                self.add(n, True, "not run: construct failed", skipped=True)
            return self.results
        self.loops = [r for r in loops if r.stage_id == self.unit_id()]
        self.gen_spec = gen.program._binding.spec
        # The Program decides: a pipeline with a HOST stage is not caller-capturable.
        self.capturable = bool(gen.program.capturable)
        if self.stage is not None:
            self.accepts_ctx = True
        self.add("construct", True, details={"capturable": self.capturable,
                                             "accepts_ctx": self.accepts_ctx})
        out_names = [n for n in names if n.startswith("output.") and n != "output.view"]
        self.guarded(out_names, lambda: self.output_checks(snap, gen))
        self.guarded(["output.view"], lambda: [self.output_view(gen)])
        if self.stage is None:
            self.guarded(list(_ATTRIBUTES), lambda: self.attribute_checks(gen, snap))
        if self.stage is not None:
            self.guarded(["stage.isolation"], lambda: [self.stage_isolation(gen)])
            if self.kind == "validity":
                self.guarded(["validity.bits"], lambda: [self.validity_bits(gen, snap)])
            if self.kind == "width":
                self.guarded(["width.values"], lambda: [self.width_values(gen)])
        self.guarded(["determinism"], lambda: [self.determinism(snap)])
        if self.stage is None:          # a stage need not add diversity: the generator's
            self.guarded(["diversity"], lambda: [self.diversity(snap)])
        self.guarded(["batch_independence"], lambda: [self.batch_independence(snap)])
        self.guarded(["statelessness"], lambda: [self.statelessness(gen, snap)])
        self.guarded(["params.per_env"], lambda: [self.params_per_env(gen)])
        self.guarded(["bounded.absorbing"], lambda: [self.bounded_absorbing()])
        if self.is_cuda:
            self.guarded(["bounded.lowering"], lambda: [self.bounded_lowering()])
        else:
            self.guarded(["host.require_caller_capturable"],
                         lambda: [self.host_require_caller_capturable(gen)])
        if self.is_cuda:
            self.guarded(["capture.caller"], lambda: [self.capture_caller()])
            self.guarded(["capture.self_graph"], lambda: [self.capture_self_graph(snap)])
        else:
            self.guarded(list(_HOT), lambda: self.hot_path(gen))
        return self.results

    # -- 1. output -------------------------------------------------------------------
    def output_checks(self, s: _Snapshot, gen=None) -> list:
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
            expect["raw.position"] = ((E * s.raw_stride,), wp.vec2f)
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
        if gen is not None:
            problems += _layout_problems(gen)
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
                    rows = s.rows("raw.position", e, 0, s.raw_count(e))
                    unwritten = s.unwritten("raw.position", e, 0, s.raw_count(e))
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
        if s.capacity is not None:
            k, A = s.capacity["min_gates"], s.capacity["gates"]
            for e in range(E):
                a = int(s.anchor_count[e])
                if not k <= a <= A:
                    bad.append(f"env {e}: anchors.count={a} is outside the generator's "
                               f"capacity [min_gates={k}, gates={A}]")
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

    # -- attributes (roadmap PR13a) -------------------------------------------------------
    def output_view(self, gen):
        """``output.view``: the unit's pipeline built with ``out=`` the middle view
        ``big.view(1, E + 1)`` of a byte-poisoned result of ``E + 2`` envs (explicit
        seeds: no env offset applies) writes exactly the standalone batch's bytes into
        the view, after construction and after a masked advance, and never a byte
        outside it (a kernel indexing past ``[0, E)`` of a result array fails here)."""
        from .._src.types import Layout  # noqa: PLC0415

        program = gen.program
        cls = type(program.result)._RESULT_BASE
        E, S = self.E, int(program.layout.stride)
        attrs = {k: info.dtype for k, info in program.attributes.items()}
        big = cls.alloc(Layout(E + 2, S, self.device), attributes=attrs)
        if self.is_cuda:
            wp.synchronize_device(self.device)
        arrays = result_arrays(big)
        for arr in arrays.values():
            arr.assign(_byte_poisoned(arr.numpy()))
        view = big.view(1, E + 1)
        twin = self.make(self.seeds_a)
        sub = self.make(self.seeds_a, out=view)
        problems: list = []

        def compare(label: str) -> None:
            if self.is_cuda:
                wp.synchronize_device(self.device)
            want = result_arrays(twin.program.result)
            got = result_arrays(view)
            for name, ref in want.items():
                mine = got.get(name)
                if mine is None:
                    continue
                if mine.numpy().tobytes() != ref.numpy().tobytes():
                    problems.append(f"{label}: {name} of the view differs from the "
                                    f"standalone batch")
            for name, arr in arrays.items():
                host = arr.numpy()
                per = S if int(host.shape[0]) == (E + 2) * S else 1
                outside = np.concatenate([host[:per], host[(E + 1) * per:]])
                if not _is_byte_poison(outside).all():
                    problems.append(f"{label}: {name} was written outside the view (env 0 "
                                    f"or {E + 1} of a {E + 2}-env result): a kernel "
                                    f"indexes past [0, E) of a result array")
        compare("construct")
        mask = np.zeros(E, np.int32)
        mask[::2] = 1
        for g in (twin, sub):
            g.enqueue(advance=True, mask=wp.array(mask, dtype=wp.int32,
                                                  device=self.device))
        compare("masked advance")
        msg = _list_text(problems)
        if problems:
            msg += (". Index every result array from element 0 of the array you are "
                    "given (e * stride + i): out= may be a view of a larger result")
        return ("output.view", not problems, msg, {"envs": E})

    def attribute_checks(self, gen, s: _Snapshot) -> list:
        attrs = [ch for ch, _ in _attribute_buffers(self.snap_kind, gen)]
        if not attrs:
            return [(n, None, "skipped: no attributes", {}) for n in _ATTRIBUTES]
        out = []
        E = s.E
        track = self.snap_kind == "generator"
        # written: every raw slot of a generator-valid env (gate generators: before the
        # env's anchors.count) holds a written value, not the kit's poison.
        bad = []
        for ch in attrs:
            name = _RAW_ATTR + ch
            for e in range(E):
                if track:
                    if s.data["raw.valid"][e] != 1:
                        continue
                    left = s.unwritten(name, e)
                else:
                    left = s.unwritten(name, e, 0, s.raw_count(e) if s.anchor_count is None
                                       else int(min(max(s.anchor_count[e], 0),
                                                    s.raw_stride)))
                if left.size:
                    bad.append(f"{ch}: env {e}: {left.size} slot(s) not written, first at "
                               f"index {int(left[0])} (it still holds the kit's poison)")
                    break
        out.append(("attributes.written", not bad,
                    (_list_text(bad) + ". Write every slot of every attribute on every "
                     "call (the raw buffers are not cleared between generations).")
                    if bad else "", {"attributes": attrs}))
        # carried: the exported result attributes.
        infos = dict(gen.program.attributes)
        if not infos:
            out.append(("attributes.carried", None, "skipped: no exported attribute",
                        {"attributes": attrs}))
            return out
        prefix = "raw." if track else "anchors."
        bad = []
        for key, info in sorted(infos.items()):
            got = s.data.get(f"attributes/{key}")
            if got is None:
                bad.append(f"{key}: missing from the result's attributes")
                continue
            src_name = _RAW_ATTR + prefix + key
            src = s.data.get(src_name)
            problem = self._carried_problem(s, key, info, got, src, src_name, track)
            if problem:
                bad.append(problem)
        out.append(("attributes.carried", not bad, _list_text(bad),
                    {"exported": sorted(infos)}))
        return out

    def _carried_problem(self, s: _Snapshot, key: str, info, got, src, src_name,
                         track: bool) -> str:
        """The first problem of exported attribute ``key`` (``got``: its result array),
        or ``""``."""
        E, S = s.E, s.stride
        rows = _as_rows(got.reshape(E * S, -1) if got.ndim > 1 else got.reshape(E * S, 1))
        rows = rows.reshape(E, S, -1)
        is_int = np.issubdtype(got.dtype, np.integer)
        for e in range(E):
            c = s.clamped(e)
            tail = rows[e, c:]
            pad_ok = np.all(tail == -1) if is_int else np.all(np.isnan(tail))
            if not pad_ok:
                return (f"{key}: env {e}: a slot past count={int(s.count[e])} is not the "
                        f"padding ({'-1' if is_int else 'NaN'})")
        if src is None:
            return ""
        for e in range(E):
            if track and s.data["raw.valid"][e] != 1:
                continue
            c = s.clamped(e)
            if c == 0:
                continue
            real = rows[e, :c]
            if not is_int and not np.isfinite(real).all():
                i = int(np.flatnonzero(~np.isfinite(real).all(axis=1))[0])
                return f"{key}: env {e}: point {i} (before count={c}) is not finite"
            n_src = s.raw_stride if track else (
                s.raw_count(e) if s.anchor_count is None
                else int(min(max(s.anchor_count[e], 0), s.raw_stride)))
            raw = _as_rows(np.asarray(s.rows(src_name, e, 0, n_src)).reshape(n_src, -1))
            if raw.size == 0:
                continue
            interp = info.interp
            if interp == "angle":
                lim = float(np.float32(np.pi))
                if np.any(real < -lim) or np.any(real > lim):
                    return f"{key}: env {e}: an angle outside [-pi, pi]"
            elif interp == "linear":
                lo, hi = raw.min(axis=0), raw.max(axis=0)
                mag = np.maximum(np.abs(lo), np.abs(hi)).astype(np.float32)
                slack = 4.0 * np.spacing(mag).astype(np.float64)
                if np.any(real < lo - slack) or np.any(real > hi + slack):
                    return (f"{key}: env {e}: a linear value outside the env's raw "
                            f"[min, max] (a linear carry interpolates the raw values)")
            else:
                values = {tuple(r) for r in raw.tolist()}
                outside = [tuple(r) for r in real.tolist() if tuple(r) not in values]
                if outside:
                    return (f"{key}: env {e}: {outside[0]} is not one of the env's raw "
                            f"values (interp={interp!r} never blends)")
        return ""

    # -- stage checks ---------------------------------------------------------------------
    def width_values(self, gen):
        program = gen.program
        sid = self.stage.stage_id
        w = program.tap("cl.half_width", after=sid).numpy()
        count = program.tap("cl.count", after=sid).numpy()
        valid = program.tap("gen.valid").numpy()
        S = int(w.shape[0]) // self.E
        rows = w.reshape(self.E, S)
        bad, lo, hi = [], math.inf, -math.inf
        for e in range(self.E):
            if valid[e] != 1:
                continue
            c = int(min(max(count[e], 0), S))
            real = rows[e, :c]
            if real.size:
                fin = real[np.isfinite(real)]
                if fin.size:
                    lo, hi = min(lo, float(fin.min())), max(hi, float(fin.max()))
            idx = np.flatnonzero(~(np.isfinite(real) & (real >= 0.0)))
            if idx.size:
                bad.append(f"env {e}: cl.half_width[{int(idx[0])}] = {real[idx[0]]!r} "
                           f"(before count={c}) is not a finite value >= 0")
        return ("width.values", not bad,
                _list_text(bad) + (". Write a finite half-width >= 0 on every real point "
                                   "(the floor is validity's check, bit 4)."
                                   if bad else ""),
                {"min": lo, "max": hi})

    def stage_accepted(self, frac: float):
        details = {"valid_fraction": frac}
        if frac > 0.0:
            return ("output.accepted", True, "", details)
        neutral = _NEUTRAL[self.kind]
        ref = self.stage.reference(self.E, self.device, neutral, self.seeds_a)
        rfrac = float(np.mean(_Snapshot(self.snap_kind, ref).valid == 1))
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
            what = "GateSequence" if self.snap_kind == "gate_generator" else "Track"
            bad.append(f"{what}.valid != (fail_mask == 0) for envs {envs[:6]}")
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
            other = _Snapshot(self.snap_kind, self.stage.without(self.E, self.device,
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
        if not _seeded(self.gen_spec):
            return ("diversity", None, "skipped: generator declares seeded=False (its "
                    "output does not depend on the seed)", {"compared": 0})
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

    # -- 6a. per-env dynamic params (roadmap PR13c) ----------------------------------------
    def params_per_env(self, gen):
        name = "params.per_env"
        sid = self.unit_id()
        table = gen.params
        sp = table[sid] if sid in table else None
        fields = _undrawn_dynamic(sp)
        if not fields:
            return (name, None, "skipped: no dynamic parameters", {})
        values = _per_env_values(fields, sp.params, self.E)
        work = _work_arrays(gen.program, {n: sp[n] for n in values})
        initial = {f.name: getattr(sp.params, f.name) for f in fields}
        bad, written = [], []
        try:
            arrays = {n: wp.array(v, dtype=sp[n].dtype, device=self.device)
                      for n, v in values.items()}
            for n, arr in arrays.items():
                sp.set(n, arr)
            self.eager(gen)
            ref = self.snapshot(gen)
            written = _written_params(sp, work, values)
            del arrays
            for e in range(self.E):
                for n, v in values.items():
                    sp.set(n, v[e].item())
                self.eager(gen)
                now = self.snapshot(gen)
                d = _env_diff(ref, e, now, e)
                if d:
                    bad.append(f"env {e}: {_fields_text(d)}")
            last = {n: np.full(self.E, v[self.E - 1], dtype=v.dtype)
                    for n, v in values.items()}
            written += [n for n in _written_params(sp, work, last) if n not in written]
        finally:
            for n, v in initial.items():
                sp.set(n, v)
            if self.is_cuda:
                wp.synchronize_device(self.device)
        problems = []
        if bad:
            problems.append(f"set per env, env e differs from a run with e's values as "
                            f"scalars ({_list_text(bad, 3)}): a per-env dynamic field must "
                            f"be read as param[e], never param[0] or on the host")
        if written:
            problems.append(f"the stage wrote its param array(s) {written} (READ-ONLY for "
                            f"the stage: the generation's values come from the live arrays)")
        details = {"fields": sorted(values), "stage": sid}
        return (name, not problems, "; ".join(problems), details)

    # -- 6b. bounded loops and HOST stages -------------------------------------------------
    def _unit_channels(self, gen) -> dict:
        """``channel@stage -> host copy`` of the stage-under-test's write channels (none
        for a generator: its raw output is in the snapshot)."""
        if self.stage is None:
            return {}
        program = gen.program
        sid = self.unit_id()
        _, writes = program._ports[sid]
        return {f"{ch}@{sid}": program.tap(ch, after=sid).numpy().copy()
                for ch in writes.values()}

    def _outputs(self, gen) -> "tuple[_Snapshot, dict]":
        snap = self.snapshot(gen)
        return snap, self._unit_channels(gen)

    def _output_diff(self, a, b) -> list:
        """``env e (fields)`` of every env whose outputs differ between ``a`` and ``b``
        (each a :meth:`_outputs`)."""
        snap_a, ch_a = a
        snap_b, ch_b = b
        out = []
        for e in range(self.E):
            fields = []
            for name, x in ch_a.items():
                y = ch_b[name]
                n = int(x.shape[0])
                stride = n // self.E if n % self.E == 0 else n
                rows = slice(e * stride, (e + 1) * stride) if stride != n else slice(None)
                if np.ascontiguousarray(x[rows]).tobytes() != \
                        np.ascontiguousarray(y[rows]).tobytes():
                    fields.append(name)
            fields += _env_diff(snap_a, e, snap_b, e)
            if fields:
                out.append(f"env {e} ({_fields_text(fields)})")
        return out

    def _loops_of(self, records) -> list:
        sid = self.unit_id()
        return [r for r in records if r.stage_id == sid]

    @staticmethod
    def _zero_flags(flags) -> None:
        devices = set()
        for arr in flags:
            arr.zero_()
            devices.add(str(arr.device))
        for dev in devices:
            wp.synchronize_device(dev)

    def bounded_absorbing(self):
        name = "bounded.absorbing"
        if not self.loops:
            return (name, None, "skipped: not applicable (the unit runs no "
                                "ctx.bounded_loop with active=)", {})
        details = {"loops": len(self.loops),
                   "windows": [int(r.windows) for r in self.loops]}
        problems, skipped = [], {}
        for label, seeds in (("A", self.seeds_a), ("B", self.seeds_b)):
            gen = self.make(seeds)
            with _probe_loops() as records:
                self.eager(gen)             # this generator's own active flags
            flags = [r.active for r in self._loops_of(records)]
            outs, runs = {}, {}
            for mode in ("poll", "unroll"):
                # A stale flag: the body must write active before the first check.
                self._zero_flags(flags)
                _poison_raw(self.snap_kind, gen)
                with _force_lowering(mode), _probe_loops() as records:
                    self.eager(gen)
                outs[mode] = self._outputs(gen)
                runs[mode] = self._loops_of(records)
            skipped[label] = sum(int(r.windows) - int(r.ran) for r in runs["poll"])
            bad = self._output_diff(outs["poll"], outs["unroll"])
            if bad:
                stop = next((r for r in runs["poll"] if r.ran < r.windows),
                            runs["poll"][0] if runs["poll"] else None)
                where = (f" the loop stopped at window {int(stop.ran) + 1}/"
                         f"{int(stop.windows)} (active[0] == 0) and" if stop is not None
                         and stop.ran < stop.windows else "")
                problems.append(f"with lowering (a) (batch {label}){where} the outputs "
                                f"differ from the unrolled lowering (c): "
                                f"{_list_text(bad, 3)}")
        details["skipped"] = skipped
        if problems:
            return (name, False,
                    "; ".join(problems) + ". Every iteration after active[0] reads 0 must "
                    "leave the stage's outputs bitwise unchanged (an absorbing freeze), "
                    "and the body must write active before the first check.", details)
        if not any(skipped.values()):
            return (name, None, "skipped: not exercised (no window was skipped for the "
                                "kit's seeds)", details)
        return (name, True, "", details)

    def bounded_lowering(self):
        name = "bounded.lowering"
        if not self.loops:
            return (name, None, "skipped: not applicable (the unit runs no "
                                "ctx.bounded_loop with active=)", {})
        dev = wp.get_device(self.device)
        ref_a = self._outputs_of_fresh(self.seeds_a)
        ref_b = self._outputs_of_fresh(self.seeds_b)
        bad, status = [], {}

        def compare(ref, got, label: str) -> None:
            d = self._output_diff(ref, got)
            if d:
                bad.append(f"{label} differs from the eager lowering (c): "
                           f"{_list_text(d, 2)}")

        # (a): the forced host poll, eager.
        gen = self.make(self.seeds_a)
        _poison_raw(self.snap_kind, gen)
        with _force_lowering("poll"), _probe_loops() as records:
            self.eager(gen)
        compare(ref_a, self._outputs(gen), "(a) the host poll")
        polled = self._loops_of(records)
        status["a"] = "ok" if any(r.ran < r.windows for r in polled) else \
            "ok (not exercised: no window skipped)"
        # (b) conditional windows / (c) unrolled, recorded into a caller's Warp capture.
        for label, forced in (("b", None), ("c", "unroll")):
            if not self.capturable:
                status[label] = ("skipped: the pipeline has HOST stages (it refuses a "
                                 "caller's capture)")
                continue
            if label == "b" and not wp.is_conditional_graph_supported():
                status[label] = "skipped: no conditional graph support (CUDA >= 12.4)"
                continue
            gen = self.make(self.seeds_a)
            result = gen._track if self.snap_kind == "generator" else gen._gate_sequence
            s = side_stream(dev)
            wp.synchronize_device(dev)
            with _force_lowering(forced), _probe_loops() as records:
                with recording(dev, s) as cap:
                    gen.generate(stream=s)
            if label == "b" and not any(r.lowering == "conditional"
                                        for r in self._loops_of(records)):
                status[label] = "skipped: no window was recorded under a conditional node"
                continue

            def replay(_g=cap.graph, _s=s):
                _poison_raw(self.snap_kind, gen)
                poison_all(result)
                wp.capture_launch(_g, stream=_s)
                wp.synchronize_device(dev)

            replay()
            compare(ref_a, self._outputs(gen), f"({label}) a recorded replay")
            _write_seeds(gen, self.seeds_b)
            replay()
            compare(ref_b, self._outputs(gen), f"({label}) a replay after a reseed")
            _write_seeds(gen, self.seeds_a)
            status[label] = "ok"
        return (name, not bad, _list_text(bad), status)

    def _outputs_of_fresh(self, seeds):
        gen = self.make(seeds)
        _poison_raw(self.snap_kind, gen)
        self.eager(gen)
        return self._outputs(gen)

    def host_require_caller_capturable(self, gen):
        name = "host.require_caller_capturable"
        program = gen.program
        host = tuple(program.host_stage_ids)
        if not host:
            return (name, None, "skipped: not applicable (no HOST stage)", {})
        from ..pipeline import planner  # noqa: PLC0415
        from ..plugin import registry  # noqa: PLC0415

        spec = dataclasses.replace(program.spec, require_caller_capturable=True)
        try:
            # The host-side phase refuses it before anything is allocated.
            planner._build(spec, binding=program._binding, prepare=False,
                           registry=registry)
        except planner.PlanError as exc:
            msg = str(exc)
            missing = [sid for sid in host if repr(sid) not in msg]
            if missing:
                return (name, False, f"the PlanError does not name HOST stage(s) "
                                     f"{missing}: {msg}", {"host_stage_ids": list(host)})
            return (name, True, "", {"host_stage_ids": list(host)})
        return (name, False, f"PipelineSpec(require_caller_capturable=True) was not "
                             f"refused although stage(s) {list(host)} are HOST",
                {"host_stage_ids": list(host)})

    # -- 5. hot path (CPU) --------------------------------------------------------------
    def hot_path(self, gen) -> list:
        from ..plugin.stage import _is_stage  # noqa: PLC0415

        eager = not self.capturable
        binding = gen.program._binding
        # A legacy capturable=False spec may sync and allocate (it has no ctx.host()).
        legacy_host = eager and not binding.capturable and not _is_stage(binding.spec)
        # A second instance built alike (constructed here: construction may allocate).
        twin = None if eager or self.skipped("hot_path.buffers") else \
            self.make(self.seeds_a)
        guard = HotPathGuard(modules=self.probe, strict=False)
        # Every bounded loop unrolled: the superset of what any lowering emits (a sync or
        # an allocation in a late window is seen).
        with _force_lowering("unroll"), guard:
            with LaunchTrace() as first:
                gen.generate()
            with LaunchTrace() as second:
                gen.generate()
        sanctioned, polled = None, []
        if self.loops:
            # One call with the default lowering ((a) on CPU): its host polls are the
            # sanctioned readback; any other sync there is a violation.
            extra = HotPathGuard(host_sync=True, allocation=False, stream=False,
                                 capture=False, strict=False)
            with extra:
                gen.generate()
            sanctioned = len(extra.sanctioned)
            polled = extra.of("host_sync")
        if legacy_host:
            waived = ("waived: capturable=False (a legacy generator spec, eager every "
                      "call, may sync and allocate)")
            waived_checks = _WAIVED_LEGACY
        else:
            waived = (f"waived: HOST stage(s) {list(gen.program.host_stage_ids)} run "
                      f"eagerly every call (their launches may depend on host data)")
            waived_checks = _WAIVED_HOST if eager else ()
        out = []

        def category(check: str, cat: str, what: str) -> None:
            if check in waived_checks:
                out.append((check, None, waived, {}))
                return
            found = guard.of(cat)
            uniq = list(dict.fromkeys(str(v) for v in found))
            msg = (f"{len(found)} {what} in two generate() calls: {_list_text(uniq)}. "
                   f"{_HINTS[cat]}") if found else ""
            if found and eager and cat in ("host_sync", "allocation"):
                msg += (" A HOST stage reaches the host only through ctx.host() (HostOps), "
                        "with buffers from StageBuilder.host_buffer() allocated at build.")
            details = {"count": len(found)}
            if cat == "host_sync":
                details["sanctioned"] = sanctioned
                if eager:
                    details["host_ops"] = len(guard.sanctioned)
                if polled:
                    extra_uniq = list(dict.fromkeys(str(v) for v in polled))
                    msg = (msg + " " if msg else "") + (
                        f"{len(polled)} host sync/readback(s) besides the sanctioned "
                        f"bounded_loop poll in a generate() call with the default "
                        f"lowering: {_list_text(extra_uniq)}. {_HINTS[cat]}")
                    found = found + polled
            out.append((check, not found, msg, details))

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
            with _force_lowering("unroll"), other, LaunchTrace() as third:
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
            return self.capture_caller_refused()
        gen = self.make(self.seeds_a)
        status = assert_caller_capturable(gen, modes=self.modes,
                                          seeded=_seeded(self.gen_spec))
        return ("capture.caller", True, ", ".join(f"{m}: {s}" for m, s in status.items()),
                dict(status))

    def capture_caller_refused(self):
        """A pipeline with HOST stages under a caller's Warp capture: ``NotCapturableError``
        naming them before any launch; the caller's graph replays its own work only."""
        from ..exec.errors import NotCapturableError  # noqa: PLC0415

        gen = self.make(self.seeds_a)
        host = tuple(gen.program.host_stage_ids)
        result = gen._track if self.snap_kind == "generator" else gen._gate_sequence
        dev = wp.get_device(self.device)
        _poison_raw(self.snap_kind, gen)
        poison_all(result)
        before = self.snapshot(gen)
        seeds0 = gen.seeds.seeds.numpy().copy()
        epoch0 = gen.seeds.epoch.numpy().copy()
        marker = wp.zeros(1, dtype=wp.int32, device=dev)
        source = wp.full(1, 7, dtype=wp.int32, device=dev)
        s = side_stream(dev)
        wp.synchronize_device(dev)
        raised = None
        with recording(dev, s) as cap:
            wp.copy(marker, source, stream=s)        # the caller's own recorded work
            try:
                gen.generate(stream=s, advance=True)
            except NotCapturableError as exc:
                raised = exc
        wp.capture_launch(cap.graph, stream=s)
        wp.synchronize_device(dev)
        bad = []
        if raised is None:
            bad.append("generate(stream=s, advance=True) inside a caller's capture did not "
                       "raise NotCapturableError")
        else:
            missing = [sid for sid in host if sid not in raised.stage_ids]
            if missing:
                bad.append(f"NotCapturableError does not name HOST stage(s) {missing}")
        if int(marker.numpy()[0]) != 7:
            bad.append("the caller's capture did not replay its own work (invalidated)")
        after = self.snapshot(gen)
        changed = sorted(k for k in before.data
                         if before.data[k].tobytes() != after.data[k].tobytes())
        if changed:
            bad.append(f"replaying the caller's graph changed the outputs "
                       f"({_fields_text(changed)}): work was recorded before the refusal")
        if not np.array_equal(gen.seeds.seeds.numpy(), seeds0) or \
                not np.array_equal(gen.seeds.epoch.numpy(), epoch0):
            bad.append("the seeds or epochs changed")
        return ("capture.caller", not bad, _list_text(bad),
                {"host_stage_ids": list(host)})

    def capture_self_graph(self, snap: _Snapshot):
        gen = self.make(self.seeds_a)
        result = gen._track if self.snap_kind == "generator" else gen._gate_sequence
        bad = []
        _poison_raw(self.snap_kind, gen)          # as the reference snapshots
        gen.generate()
        if self.capturable and gen.graph is None:
            bad.append("generate() on CUDA did not capture the generator's own graph")
        if not self.capturable:
            program = gen.program
            segments = program.segments
            graphs = program.segment_graphs()
            if gen.graph is not None:
                bad.append("a pipeline with HOST stages captured a whole-pipeline graph")
            if len(segments) > 1:
                want = [not seg.host for seg in segments]
                got = None if graphs is None else [g is not None for g in graphs]
                if got != want:
                    bad.append(f"the per-segment self-graphs are {got}, expected a graph "
                               f"exactly at the non-HOST segments {want}")
            elif graphs is not None:
                bad.append("a pipeline that is one HOST segment captured a graph")

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


def _default_ordering(name: str) -> str:
    """The kit's ordering of gate generator ``name``: its ``preferred["order"]`` when it
    supports it, else ``"ccw"`` when supported, else its first supported ordering (the
    rule ``presets.gates`` applies)."""
    from ..plugin import registry  # noqa: PLC0415
    from ..stages.gates import _default_order  # noqa: PLC0415

    return _default_order(registry.get("gate_generator", name), registry)


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
    """As :func:`check_generator`, for the registered gate generator ``name`` (a
    ``gate_generator`` Stage or a legacy ``GateGeneratorSpec``; :data:`CHECKS`
    ``["gate_generator"]``). ``config`` holds extra ``GateGenConfig`` fields; without a
    ``gate_ordering`` the kit uses the generator's ``preferred["order"]`` when it
    supports it, else ``"ccw"`` when supported, else its first supported ordering."""
    return _check("gate_generator", name, params, devices=devices, num_envs=num_envs,
                  seeds=seeds, config=config, capture_modes=capture_modes, skip=skip,
                  modules=modules)


# ---------------------------------------------------------------------------------------
# Track-pipeline stages
# ---------------------------------------------------------------------------------------

def _is_gate_stage(kind: str, cls) -> bool:
    """Whether :func:`check_stage` runs ``cls`` in the gate pipeline: an ``order`` or
    ``stations`` stage, or a validity stage that reads a gate channel."""
    from ..stages.channels import is_gate_channel  # noqa: PLC0415

    if kind in ("order", "stations"):
        return True
    chans = {*dict(cls.reads).values(), *dict(cls.writes).values()}
    return kind == "validity" and any(is_gate_channel(ch) for ch in chans)


class _StageUnderTest:
    """The pipelines :func:`check_stage` builds around one stage."""

    def __init__(self, kind: str, name: str, params, upstream: Mapping, cls,
                 pipeline: "str | None" = None, ports: "Mapping | None" = None) -> None:
        from ..pipeline.spec import StageRef  # noqa: PLC0415
        from ..stages.channels import FAIL  # noqa: PLC0415

        self.kind, self.name, self.params = kind, name, params
        self.upstream = dict(upstream)
        ports = dict(ports or {})
        # A width stage that reads cl.half_width (a rewrite) runs right after the
        # neutral width.constant.
        self.rewrite = kind == "width" and "cl.half_width" in dict(cls.reads).values()
        self.gate = _is_gate_stage(kind, cls) if pipeline is None else pipeline == "gates"
        # A stations stage that reads the relaxed centerline runs on relaxed_stations.
        self.relaxed = kind == "stations" and any(
            str(ch).startswith("cl.") for ch in dict(cls.reads).values())
        # A validity stage that starts valid.fail (writes it without reading it): the
        # planner admits only validity.default / validity.gate_default there (a plugin
        # one fails construct).
        self.initializer = kind == "validity" and FAIL in dict(cls.writes).values() \
            and FAIL not in dict(cls.reads).values()
        if kind != "validity" and not self.rewrite:
            self.stage_id = kind
            self.ref = StageRef(kind, name, params=params, ports=ports)
        elif self.initializer:
            self.stage_id = "validity"
            self.ref = StageRef(kind, name, params=params, ports=ports)
        else:
            self.stage_id = name
            self.ref = StageRef(kind, name, params=params, id=name, ports=ports)

    def facade(self):
        """The facade class that runs the pipeline (``TrackGenerator`` /
        ``GateGenerator``)."""
        if self.gate:
            from .._src.gate_generator import GateGenerator  # noqa: PLC0415

            return GateGenerator
        from .._src.track_generator import TrackGenerator  # noqa: PLC0415

        return TrackGenerator

    def _preset(self, E: int, device: str, slot):
        from ..pipeline import presets  # noqa: PLC0415

        kw = dict(self.upstream)
        first = "gate_default" if self.gate else "default"
        if self.rewrite:
            spec = presets.track(num_envs=E, device=device, **kw)
            if slot is None or (isinstance(slot, str) and slot == _NEUTRAL["width"]):
                return spec
            return spec.insert_after("width", slot)
        if self.kind != "validity":
            kw[self.kind] = slot
        elif self.initializer:
            kw["checks"] = () if slot is None else (slot,)
        else:
            checks = tuple(kw.get("checks", (first,)))
            kw["checks"] = checks if slot is None else (*checks, slot)
        if not self.gate:
            return presets.track(num_envs=E, device=device, **kw)
        if self.kind in ("order", "stations"):
            kw.setdefault("source", "relaxed_stations" if self.relaxed else "stations")
            kw.setdefault("generator", "bezier")
        return presets.gates(num_envs=E, device=device, **kw)

    def spec(self, E: int, device: str):
        """The pipeline holding the stage under test."""
        return self._preset(E, device, self.ref)

    def _generator(self, spec, seeds: np.ndarray, device: str):
        from .._src.rng_utils import PerEnvSeededRNG  # noqa: PLC0415

        E = int(seeds.shape[0])
        rng = PerEnvSeededRNG(seeds=wp.array(_int32(seeds), dtype=wp.int32,
                                             device=device), num_envs=E, device=device)
        gen = self.facade()(spec, rng)
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
                ports: "Mapping[str, str] | None" = None,
                pipeline: "str | None" = None, devices=None, num_envs: int = 8,
                seeds: Sequence[int] = (11, 1009),
                capture_modes: Iterable[str] = CAPTURE_MODES, skip: Iterable[str] = (),
                modules: Iterable[str] = ()) -> Report:
    """Run every conformance check (:data:`CHECKS` ``[kind]``) on the registered
    ``relax`` / ``width`` / ``elevation`` / ``validity`` / ``order`` / ``stations`` stage
    ``name``, inside the default track pipeline (:func:`track_gen.pipeline.presets.track`)
    or, for an ``order`` / ``stations`` stage and a validity stage that reads gate
    channels, the default gate pipeline (:func:`track_gen.pipeline.presets.gates`).
    ``pipeline="gates"`` runs an ``elevation`` stage on gates (``presets.gates(
    elevation=...)``, its ports rewired to ``gates.*``; the default anchors generator
    ``bezier`` unless ``upstream`` names another): call it once per pipeline to check
    both.

    The stage takes its slot (``relax=``, ``width=``, ``elevation=``, ``order=``,
    ``stations=``); a validity check runs after the default checks
    (``upstream["checks"]``, default ``("default",)`` / ``("gate_default",)``) with id
    ``name``, and a validity stage that initialises ``valid.fail`` (writes it without
    reading it) replaces them. An order runs on the stations of the bezier TRACK
    generator (``source="stations"``: no ordering whitelist), a stations stage on its
    raw loop (``source="relaxed_stations"`` when it reads ``cl.*``).

    Args:
        kind: ``"relax"``, ``"width"``, ``"elevation"``, ``"validity"``, ``"order"`` or
            ``"stations"``.
        name: the registered stage name (``registry.available(kind)``).
        params: its params (a Params instance or a mapping of its fields); ``None``:
            the defaults.
        ports: the stage's ``StageRef.ports`` (wires a required read port, e.g.
            ``{"src": "cl.acme.hw"}`` for ``width.from_channel``, whose generator then
            goes in ``upstream``).
        upstream: more :func:`~track_gen.pipeline.presets.track` /
            :func:`~track_gen.pipeline.presets.gates` keyword arguments (``generator``,
            ``generator_params``, ``num_points``, ``n_max``, ``half_width``, the other
            kinds' slots, ``checks`` (the checks BEFORE a validity check under test),
            ``on_invalid``, ``source``). ``num_envs``, ``device`` and the slot under
            test are set by the kit.
        pipeline: ``None`` (default: the gate pipeline for an ``order`` / ``stations``
            stage and a validity stage that reads gate channels, else the track
            pipeline), ``"track"`` or ``"gates"``. ``"gates"`` admits ``elevation``,
            ``validity``, ``order`` and ``stations`` stages, ``"track"`` ``relax``,
            ``width``, ``elevation`` and ``validity`` stages.
        devices, num_envs, seeds, capture_modes, skip, modules: as
            :func:`check_generator`.

    Returns:
        A :class:`Report`; call ``.raise_for_failures()`` to fail on any failed check.

    Raises:
        ValueError: invalid arguments (an unknown kind, a ``pipeline`` the kind does not
            run in, a reserved ``upstream`` key, unknown check names, overlapping
            seeds, ...).
        Exceptions of ``registry.describe`` for an unknown or quarantined name
        (``PluginLoadError`` carries the plugin's load error).
    """
    from ..plugin import registry  # noqa: PLC0415

    if kind not in STAGE_KINDS:
        raise ValueError(f"kind must be one of {list(STAGE_KINDS)}, got {kind!r}")
    if pipeline not in (None, "track", "gates"):
        raise ValueError(f"pipeline must be None, 'track' or 'gates', got {pipeline!r}")
    if (pipeline == "gates" and kind in ("relax", "width")) or \
            (pipeline == "track" and kind in ("order", "stations")):
        raise ValueError(f"pipeline={pipeline!r}: a {kind} stage runs in the "
                         f"{'track' if pipeline == 'gates' else 'gate'} pipeline only")
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
    stage = _StageUnderTest(kind, name, params, upstream, registry.get(kind, name),
                            pipeline=pipeline, ports=ports)
    results: list = []
    for dev in _devices(devices):
        results += _DeviceRun(kind, name, params, dev, E, seeds_a, seeds_b, {}, info,
                              probe, modes, skip, stage=stage).run()
    return Report(kind, name, results)
