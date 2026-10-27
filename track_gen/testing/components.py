"""Does a component honour the contract? ``check_component``.

One call runs a :class:`~track_gen.tools.Component` (a built-in adapter or your own)
through every check the component contract implies (see :ref:`writing-a-component`) and
returns a :class:`~track_gen.testing.Report`::

    from track_gen import tools
    from track_gen.testing import check_component

    report = check_component(lambda cc: tools.Progress(),
                             upstream=[lambda cc: tools.Checkpoints(0.5)])
    report.raise_for_failures()

The kit builds the producer (``source``: ``"track"`` = ``presets.track``, ``"gates"`` =
``presets.gates(gate_width=0.1)`` (passable gates), or a ``PipelineSpec`` /
``TrackGenConfig`` / ``GateGenConfig``) with
explicit seeds and ``on_invalid="keep"``, attaches the ``upstream`` components (factories,
in order) and then the component under test (``factory(cc)``: the kit passes the
:class:`~track_gen.tools.ComponentContext`, so a component that owns device arrays builds
them for the kit's device and ``E``) through
:meth:`ComponentContext.for_producer <track_gen.tools.ComponentContext.for_producer>`.
Every check and every capture drives the whole chain in order, upstream first
(``on_generate`` of each, then ``reset`` of each, then ``step`` of each), and compares only
the component under test.

**Owned vs aliased.** The kit finds every ``wp.array`` reachable from the component's
``result`` (a bare array; dataclass fields, mapping values, named-tuple, tuple and list
items, ``__slots__`` and instance attributes, nested ones named with dots, e.g.
``contact.oob``); a result that exposes none fails ``construct``. Each one is classified
by pointer: aliased when it is an array of the read target, of an upstream result or of
the producer's result (e.g. ``GateCheckpoints``: every array), owned otherwise. Only owned
arrays are compared by the byte checks, and poisoned by the capture checks.

**Inputs.** Every input name the chain declares gets one ``[E * slots]`` array per run.
``drive(step, inputs, source)`` writes them before each step; the default drive knows
``position`` (``[E]`` vec3f: walking along the source's centerline, or between
consecutive gates, with lateral offsets that leave the road, or the gate opening, now
and then),
``box_position`` (that point plus per-slot offsets, every fourth slot inactive (NaN) on
some steps), ``orientation`` (the yaw of the motion) and ``half_extents`` (``(0.03,
0.015)``). Any other input needs ``drive=`` (``construct`` fails naming it).

Checks (:data:`~track_gen.testing.CHECKS` ``["component"]``), per device:

``construct``
    The producer and the chain build (``cc.add`` checks the declaration: ``name``,
    ``source``, ``requires`` / ``optional`` protocols, ``topology``, ``inputs``,
    ``after``); ``layout`` and ``result`` are set (with at least one ``wp.array``); the
    source has a valid env (else every check would be vacuous; ``details["valid_envs"]``);
    the inputs are allocated and bound; a warm-up ``on_generate(ctx, None)`` and
    ``reset(ctx, all)``; then the attach contract: a chain attached now (no warm-up)
    steps to the same owned bytes as the warmed one (``attach`` ends with
    ``on_generate(cc.ctx, None)`` and initialises the per-env state as ``reset`` does).
``requires.complete``
    On the read target: a proxy exposing only the members of ``requires`` + ``optional``
    (plus ``layout``) attaches and gives the same owned bytes over a trajectory; for
    each required member, a proxy without it makes ``cc.add`` raise ``TypeError``; for
    each optional member, a proxy without it attaches. A component that declares no
    protocol and reads the producer (no ``requires``, ``optional`` or ``source``) is
    attached to an empty read target (only a ``layout``): it must attach and give the
    same owned bytes, then the check reports "reads no source" (skipped); reading an
    undeclared member fails. One that reads another component's result (``source``)
    without declaring a protocol of it is skipped (nothing to check against).
``topology``
    A proxy whose ``layout`` is a ``closed=False`` copy: ``"closed"`` -> ``cc.add`` raises
    ``ValueError``, ``"any"`` -> it attaches. Skipped without ``requires`` / ``optional``
    (the check is the protocols').
``result.stable``
    ``step()`` returns ``component.result`` (identity); the result's arrays keep their
    pointers after ``on_generate``, ``reset`` and ``step``; each aliased array IS an
    array of the read target / upstream / producer.
``determinism``
    Two producers and chains with the same seeds and drive: identical owned bytes every
    step.
``statelessness`` (``stateful = False``)
    After a history of steps, ``step`` equals a FRESH component's (attached now to the
    same upstream instances: its ``attach`` ends with ``on_generate(ctx, None)``) on the
    same inputs. Skipped for a stateful component.
``reset.exact`` (``stateful = True``)
    ``reset(ctx, m)`` mid-trajectory: from then on the masked envs equal a fresh
    component's, the unmasked ones a never-reset twin's, every step.
``regenerate.masked``
    The producer advanced with mask ``m`` (``generate(advance=True, mask=m)``), the
    chain's ``on_generate(ctx, m)`` and ``reset(ctx, m)``: from then on the unmasked envs
    equal a never-advanced twin, the masked envs a fresh chain on a twin producer
    advanced with ``m`` before attaching; ``details["masked_rows_changed"]`` reports
    non-vacuity.
``regenerate.self_reset`` (stateful, reads a source that carries ``generation``)
    The same run WITHOUT ``reset`` gives the same bytes as with it (the
    generation-aware self-reset). Not applicable otherwise (skipped).
``invalid.sentinel``
    K: the producer's result with half the envs failed (``fail_mask |= 1``) by the
    commit's own kernel at ``poison=0`` (geometry kept, ``valid = 0``); P: the same at
    ``poison=1`` (``count = 0``, NaN geometry). The same chain on each, driven from K's
    geometry: byte-identical owned outputs over the trajectory (an invalid env reads
    exactly like a poisoned one). Skipped when the result has aliased arrays and no owned
    one (its consumers are checked with it upstream).
``capacity.clamp`` (a component that reads a ``cc.capacity`` key)
    Attached with the key scaled down (by 4, or more until some env truncates when the
    result or the utility exposes ``truncated``): each env's owned rows equal those of
    the same chain attached to that env's one-env view (no write past its own rows), and
    some env reports truncation. Skipped when no key is read.
``source.view``
    The chain attached to ``big.view(1, E + 1)`` of a byte-poisoned ``E + 2`` result
    holding the batch: equal owned bytes to the standalone run over a trajectory, and
    every byte of ``big`` unchanged (a component never writes its source, nor outside
    its env range). Skipped without ``requires`` / ``optional``.
``hot_path.host_sync`` / ``.allocation`` / ``.stream`` / ``.capture`` (CPU)
    ``on_generate(ctx, None)``, ``on_generate(ctx, m)``, ``reset(ctx, m)``, ``step(ctx)``
    twice under :func:`~track_gen.testing.hot_path_guard` (``modules=("track_gen",
    <the component's package>)``), on a chain whose hot methods never ran before (a
    throwaway chain ran them first, so module loads stay outside the guard): work done
    only on a first hot call (a lazy allocation, a one-time readback) is caught.
``hot_path.topology`` (CPU)
    Two calls of each hot method emit the same :class:`~track_gen.testing.LaunchTrace`;
    also ``step`` with the rebind flag of the wrapped utility armed (a ``LaunchContext``
    reporting ``capturing``) vs unarmed: arming it changes no emission.
``hot_path.buffers`` (CPU)
    Every pointer the hot calls launch on belongs to the component, its upstream, the
    source, the inputs or the masks (:func:`~track_gen.testing.buffers_of`); and
    ``on_generate(ctx, None)``, ``on_generate(ctx, m)`` and ``reset(ctx, m)``
    launch on no pointer of a bound input (a ``Course`` runs them in its regeneration
    body, whose own CUDA graph is keyed on ``(advance, mask)`` only and would bake the
    input: read inputs in ``step``; a respawn snapshot of an input belongs in ``step``,
    with ``reset`` arming a per-env flag that ``step`` consumes). The upstream
    components are bound to the same input arrays, so an input reached through an
    upstream result that aliases it is caught too.
``capture.caller`` (CUDA)
    Per mode of ``modes`` (``"warp"``, ``"external"``, ``"torch"``): on a chain whose hot
    methods never ran eagerly, the chain a ``Course`` records:
    ``[any_set(done) -> flag; ctx.skip_unless(flag, [producer.enqueue(advance=True,
    mask=done) -> chain on_generate(ctx, done) -> chain reset(ctx, done)]) -> chain
    step(ctx)]``, recorded on a side stream (mempool disabled, ``wp.capture_launch``
    forbidden) and replayed ``steps`` times with host-written inputs and a deterministic
    ``done`` schedule whose every fourth mask is all zero (the skipped branch replays);
    the masked part is a CUDA conditional-node body where the capture allows one
    (``"warp"`` with CUDA >= 12.4; unconditional otherwise), so ``on_generate`` /
    ``reset`` must emit only kernels, fills and device-to-device copies on ``ctx``'s
    stream; an eager twin runs the same chain, skipping the masked part on an all-zero
    ``done`` exactly when the capture recorded it under a conditional node (the replay
    skips it there), so both runs skip or run the same ``on_generate`` / ``reset``;
    before each replay / eager iteration the owned arrays of the component under test
    are poisoned in BOTH runs (a skipped ``on_generate`` leaves the poison in both);
    equal every step. ``details[mode]``: ``"ok"`` or ``"skipped: <reason>"`` (torch
    modes without torch).
``capture.self_reset`` (CUDA, as ``regenerate.self_reset``)
    The same graph without the chain's ``reset``: equal to the eager twin WITH it.

A component that declares no read (no ``requires``, ``optional`` or ``source``: a raw-mode
``DiscCollision``) and attaches to an empty read target reports ``requires.complete``,
``topology`` and ``source.view`` as skipped ("reads no source"); ``regenerate.masked`` and
``invalid.sentinel`` run for every component (they pass trivially for one that reads
nothing), and ``regenerate.self_reset`` for every stateful one that reads a source whose
read target carries ``generation`` (``optional=(HasGeneration,)`` alone qualifies).
"""
from __future__ import annotations

import dataclasses
import numbers
import types
from typing import Callable, Mapping, Sequence

import numpy as np
import warp as wp

from ..exec.context import LaunchContext
from ..ops.reduce import any_set
from ..results import Layout, _members_of
from .capture import (CAPTURE_MODES, _torch_cuda, capture_launch_forbidden,
                      mempool_disabled, poison, recording, result_arrays, side_stream)
from .conformance import (_COMPONENT_CHECKS, CheckResult, Report, _devices, _exc_text,
                          _top_package)
from .contract import HotPathGuard
from .trace import BufferSet, LaunchTrace, buffers_of

__all__ = ["check_component"]

_HOT_GUARD = ("hot_path.host_sync", "hot_path.allocation", "hot_path.stream",
              "hot_path.capture")
_CPU_ONLY = _HOT_GUARD + ("hot_path.topology", "hot_path.buffers")
_CUDA_ONLY = ("capture.caller", "capture.self_reset")
_NO_SOURCE = ("topology", "source.view")
_KNOWN_INPUTS = ("position", "box_position", "orientation", "half_extents")
_BYTE_POISON = 0xA5


class _Skip(Exception):
    """A check that does not apply (reported as skipped)."""


_LEAF_MODULES = ("warp", "numpy", "torch", "jax")


def _members(v) -> "list[tuple[str, object]]":
    """The named members of a container ``v`` (see :func:`_result_arrays`)."""
    if isinstance(v, Mapping):
        return [(str(k), x) for k, x in v.items()]
    if dataclasses.is_dataclass(v):
        return [(f.name, getattr(v, f.name, None)) for f in dataclasses.fields(v)]
    if isinstance(v, tuple) and hasattr(v, "_asdict"):       # a NamedTuple
        return list(v._asdict().items())
    if isinstance(v, (tuple, list)):
        return [(str(i), x) for i, x in enumerate(v)]
    out = dict(getattr(v, "__dict__", None) or {})
    for cls in type(v).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for name in ((slots,) if isinstance(slots, str) else slots):
            if name not in out and name not in ("__dict__", "__weakref__"):
                out[name] = getattr(v, name, None)
    return list(out.items())


def _result_arrays(obj) -> dict:
    """``name -> wp.array`` of every array reachable from a component's result: a bare
    array (``"result"``), or through dataclass fields, mapping values, named-tuple /
    tuple / list items, ``__slots__`` and instance attributes, recursively (nested names
    dotted: ``contact.oob``, ``pair.0``)."""
    out: dict = {}
    seen: set = set()

    def walk(v, name: str, depth: int) -> None:
        if v is None:
            return
        if isinstance(v, wp.array):
            out.setdefault(name or "result", v)
            return
        if depth > 6 or id(v) in seen or isinstance(
                v, (str, bytes, numbers.Number, type, types.ModuleType, Layout)) or \
                callable(v) or type(v).__module__.split(".")[0] in _LEAF_MODULES:
            return
        seen.add(id(v))
        for k, x in _members(v):
            walk(x, f"{name}.{k}" if name else k, depth + 1)

    walk(obj, "", 0)
    return out


def _host(arr) -> np.ndarray:
    return np.ascontiguousarray(arr.numpy()).copy()


def _snap(arrays: Mapping) -> dict:
    return {k: _host(a) for k, a in arrays.items()}


def _bytes(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).reshape(-1).view(np.uint8)


def _rows(a: np.ndarray, E: int) -> "np.ndarray | None":
    """``a``'s bytes as ``[E, row bytes]`` (``None`` when its length is not a multiple
    of ``E``)."""
    b = _bytes(a)
    if a.shape[0] % E or b.size % E:
        return None
    return b.reshape(E, -1)


def _diff(a: dict, b: dict) -> list:
    """Names whose bytes differ (or that are missing from ``b``)."""
    return [k for k, v in a.items() if k not in b or not np.array_equal(_bytes(v),
                                                                       _bytes(b[k]))]


def _row_diff(a: dict, b: dict, E: int, envs) -> list:
    """``name[e]`` for the rows of ``envs`` whose bytes differ."""
    out = []
    for k, v in a.items():
        ra, rb = _rows(v, E), _rows(b[k], E)
        if ra is None or rb is None:
            if not np.array_equal(_bytes(v), _bytes(b[k])):
                out.append(k)
            continue
        out += [f"{k}[{e}]" for e in envs if not np.array_equal(ra[e], rb[e])]
    return out


def _text(names: Sequence[str], n: int = 4) -> str:
    names = list(names)
    return ", ".join(names[:n]) + (f", ... ({len(names)} in all)" if len(names) > n else "")


def _i32(values, device) -> "wp.array":
    return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)


def _target_layout(target) -> "Layout | None":
    lay = getattr(target, "layout", None)
    return lay if isinstance(lay, Layout) else None


class _CapturingContext(LaunchContext):
    """A ``LaunchContext`` that reports ``capturing`` (arms the rebind probe; no other
    hot path branches on it)."""

    @property
    def capturing(self) -> bool:
        return True


# ---------------------------------------------------------------------------------------
# Producers and runs
# ---------------------------------------------------------------------------------------

def _producer(source, E: int, device: str, seeds: np.ndarray):
    """A fresh producer of ``source`` (``on_invalid="keep"``) seeded with ``seeds``."""
    from .._src.gate_generator import GateGenerator  # noqa: PLC0415
    from .._src.track_generator import TrackGenerator  # noqa: PLC0415
    from .._src.types import GateGenConfig, TrackGenConfig  # noqa: PLC0415
    from ..pipeline import presets  # noqa: PLC0415
    from ..pipeline.spec import PipelineSpec  # noqa: PLC0415

    arr = _i32(seeds, device)
    if source == "track":
        return TrackGenerator(presets.track(num_envs=E, device=device, on_invalid="keep"), seeds=arr)
    if source == "gates":
        return GateGenerator(presets.gates(num_envs=E, device=device, gate_width=0.1,
                                           on_invalid="keep"),
                             seeds=arr)
    if isinstance(source, TrackGenConfig):
        return TrackGenerator(dataclasses.replace(source, num_envs=E, device=device,
                                                  on_invalid="keep"),
                              seeds=arr)
    if isinstance(source, GateGenConfig):
        return GateGenerator(dataclasses.replace(source, num_envs=E, device=device,
                                                  on_invalid="keep"),
                             seeds=arr)
    if isinstance(source, PipelineSpec):
        batch = dataclasses.replace(source.batch, num_envs=E, device=device,
                                    on_invalid="keep")
        spec = dataclasses.replace(source, batch=batch)
        cls = GateGenerator if batch.max_gates is not None else TrackGenerator
        return cls(spec, seeds=arr)
    raise TypeError(f"check_component: source must be 'track', 'gates', a PipelineSpec, "
                    f"a TrackGenConfig or a GateGenConfig, got {source!r}")


def _result_of(producer):
    return producer.program.result


def _geometry(src) -> "tuple[np.ndarray, np.ndarray, bool]":
    """Host ``[E, S, 3]`` points and ``[E]`` counts of a source (centerline, or gate
    positions), and whether they are gates."""
    gates = getattr(src, "center", None) is None and getattr(src, "position", None) is not None
    pts = (src.position if gates else src.center).numpy()
    cnt = src.count.numpy()
    E = int(cnt.shape[0])
    return pts.reshape(E, -1, 3), cnt, gates


def _done_schedule(i: int, E: int) -> list:
    """The capture checks' ``done`` mask of step ``i``: env ``e`` where ``(i + e) % 5 ==
    0``, all zero every fourth step (the conditional body is skipped on those
    replays)."""
    if i % 4 == 3:
        return [0] * E
    return [1 if (i + e) % 5 == 0 else 0 for e in range(E)]


def _caller_chain(ctx, run, program, done, flag, *, reset: bool,
                  skip: bool = False) -> bool:
    """The chain the capture checks record, as a ``Course`` records its regeneration and
    step: ``any_set(done)`` into ``flag``, then, under
    ``ctx.skip_unless(flag, ...)`` (a CUDA conditional node where the capture allows
    one), the producer's masked advancing generation, the chain's ``on_generate(done)``
    and (``reset``) ``reset(done)``; then the chain's ``step``.

    ``skip``: leave the masked part out (the eager twin of a replay whose conditional
    node skips it: an all-zero ``done``). Returns what ``skip_unless`` returned: whether
    the masked part was recorded under a conditional node (``False`` when skipped)."""
    any_set(ctx, done, flag)

    def regenerate(c) -> None:
        program.enqueue(c, advance=True, mask=done)
        run.on_generate(c, done)
        if reset:
            run.reset(c, done)

    conditional = False if skip else bool(ctx.skip_unless(flag, regenerate))
    run.step(ctx)
    return conditional


class _Kit:
    """The configuration of one ``check_component`` call (shared by every device)."""

    def __init__(self, factory, upstream, source, E, steps, seeds, drive, modes):
        self.factory = factory
        self.upstream = tuple(upstream)
        self.source = source
        self.E = int(E)
        self.steps = int(steps)
        self.seeds = seeds
        self.drive_fn = drive
        self.modes = tuple(modes)

    # -- inputs -----------------------------------------------------------------------
    def specs(self, chain) -> dict:
        out: dict = {}
        for c in chain:
            for k, s in dict(getattr(c, "inputs", {}) or {}).items():
                out.setdefault(k, s)
        return out

    def drive(self, run, step: int, geom=None) -> None:
        """Write ``run``'s inputs for ``step`` from ``geom`` (default: the run's source)."""
        src = run.source if geom is None else geom
        if self.drive_fn is not None:
            self.drive_fn(step, run.inputs, src)
            return
        if not run.inputs:
            return
        pts, cnt, gates = _geometry(src)
        E = self.E
        pos = np.full((E, 3), np.nan, np.float32)
        yaw = np.zeros(E, np.float32)
        for e in range(E):
            n = int(min(max(int(cnt[e]), 0), pts.shape[1]))
            if n < 2:
                continue
            # Start just before the first target (checkpoint / gate 0), so progress
            # events fire; gates: four steps per gap, one of them close to a gate plane.
            if gates:
                k = (step // 4 - 1) % n
                t = np.float32(((step % 4) + 0.1) / 4.0)
            else:
                k = (2 * step - 3 + e % 3) % n
                t = np.float32(0.4)
            a, b = pts[e, k], pts[e, (k + 1) % n]
            d = b - a
            yaw[e] = np.arctan2(d[1], d[0])
            nrm = np.float32([-np.sin(yaw[e]), np.cos(yaw[e]), 0.0])
            off = np.float32((0.025 if gates else 0.04) * ((step + e) % 7 - 3))
            pos[e] = a + t * d + off * nrm
        for name, arr in run.inputs.items():
            slots = int(arr.shape[0]) // E
            if name == "position":
                host = pos
            elif name == "box_position":
                host = np.repeat(pos, slots, axis=0)
                for j in range(1, slots):
                    host[j::slots] += np.float32([0.05 * j, -0.03 * j, 0.0])
                if step % 5 == 2:
                    host[3::4] = np.nan
            elif name == "orientation":
                yy = np.repeat(yaw, slots) + np.float32(0.3) * np.tile(np.arange(slots),
                                                                         E)
                host = np.zeros((E * slots, 4), np.float32)
                host[:, 2] = np.sin(yy / 2)
                host[:, 3] = np.cos(yy / 2)
            elif name == "half_extents":
                host = np.tile(np.float32([0.03, 0.015]), (E * slots, 1))
            else:  # pragma: no cover - construct refuses it first
                raise ValueError(f"input {name!r} needs drive=")
            arr.assign(host.astype(np.float32))


class _Run:
    """One source (a producer's result, or a static result) with the chain attached:
    upstream components, then the component under test, each bound to this run's own
    input arrays."""

    def __init__(self, kit: _Kit, source, cc, *, producer=None) -> None:
        from ..tools import ComponentContext  # noqa: PLC0415

        assert isinstance(cc, ComponentContext)
        self.kit = kit
        self.source = source
        self.producer = producer
        self.cc = cc
        self.upstream = []
        for f in kit.upstream:
            c = f(cc)
            cc.add(c, source)
            self.upstream.append(c)
        self.comp = kit.factory(cc)
        cc.add(self.comp, source)
        self.chain = self.upstream + [self.comp]
        specs = kit.specs(self.chain)
        dev = cc.layout.device
        self.inputs = {k: wp.zeros(kit.E * s.slots, dtype=s.dtype, device=dev)
                       for k, s in specs.items()}
        for c in self.chain:
            self.bind(c)
        self.target = source if self.comp.source is None else \
            cc.component(self.comp.source).result
        alias = buffers_of(self.target, *[u.result for u in self.upstream], source)
        self.alias_ptrs = {int(s) for s, _, _ in alias._entries}
        arrays = _result_arrays(self.comp.result)
        self.owned = {k: a for k, a in arrays.items()
                      if int(a.ptr) and alias.locate(int(a.ptr)) is None}
        self.aliased = sorted(k for k, a in arrays.items()
                              if int(a.ptr) and alias.locate(int(a.ptr)) is not None)

    def bind(self, c) -> None:
        names = dict(getattr(c, "inputs", {}) or {})
        if names:
            c.bind(**{k: self.inputs[k] for k in names})

    # -- the component under test, attached again to the same upstream instances --------
    def twin(self, *, target=None):
        """A fresh component from the factory, attached to this run's upstream instances
        (its read target replaced by ``target`` when given) and bound to this run's
        inputs."""
        from ..tools import ComponentContext  # noqa: PLC0415

        cc2 = ComponentContext(self.cc.layout, self.cc.capacity, ctx=self.cc.ctx)
        cc2._components.update({u.name: u for u in self.upstream})
        comp = self.kit.factory(cc2)
        if target is None:
            cc2.add(comp, self.source)
        elif comp.source is None:
            cc2.add(comp, target)
        else:
            up = cc2.component(comp.source)
            real = up.result
            up.result = target
            try:
                cc2.add(comp, self.source)
            finally:
                up.result = real
        self.bind(comp)
        return comp

    def arrays_of(self, comp) -> dict:
        """``comp``'s result arrays named as this run's owned arrays."""
        got = _result_arrays(comp.result)
        return {k: got[k] for k in self.owned}

    # -- driving ------------------------------------------------------------------------
    def on_generate(self, ctx, mask, extra=()) -> None:
        for c in (*self.chain, *extra):
            c.on_generate(ctx, mask)

    def reset(self, ctx, mask, extra=()) -> None:
        for c in (*self.chain, *extra):
            c.reset(ctx, mask)

    def step(self, ctx, extra=()) -> None:
        for c in (*self.chain, *extra):
            c.step(ctx)

    def copy_inputs_to(self, other: "_Run") -> None:
        for k, a in self.inputs.items():
            wp.copy(other.inputs[k], a)


# ---------------------------------------------------------------------------------------
# One device
# ---------------------------------------------------------------------------------------

class _DeviceRun:
    def __init__(self, kit: _Kit, device: str) -> None:
        self.kit = kit
        self.device = device
        self.dev = wp.get_device(device)
        self.E = kit.E
        self.ctx = LaunchContext.for_call(device)
        self.name = None
        self._hot = None
        self._regen_snaps = None
        self._p3 = None

    # -- helpers ------------------------------------------------------------------------
    def producer(self):
        return _producer(self.kit.source, self.E, self.device, self.kit.seeds)

    def run_on(self, producer) -> _Run:
        from ..tools import ComponentContext  # noqa: PLC0415

        cc = ComponentContext.for_producer(producer, ctx=self.ctx)
        return _Run(self.kit, _result_of(producer), cc, producer=producer)

    def run_static(self, result, capacity) -> _Run:
        from ..tools import ComponentContext  # noqa: PLC0415

        cc = ComponentContext(result.validate(), capacity, ctx=self.ctx)
        return _Run(self.kit, result, cc)

    def mask(self) -> "wp.array":
        return _i32([1 if e % 3 == 1 else 0 for e in range(self.E)], self.device)

    def ones(self) -> "wp.array":
        return _i32(np.ones(self.E), self.device)

    def sync(self) -> None:
        wp.synchronize_device(self.device)

    def declares_protocols(self) -> bool:
        """Whether the component declares protocols of its read target (``requires`` or
        ``optional``): what ``requires.complete``, ``topology`` and ``source.view``
        check."""
        c = self.main.comp
        return bool(tuple(getattr(c, "requires", ()) or ()) or
                    tuple(getattr(c, "optional", ()) or ()))

    def no_protocol_reason(self) -> str:
        src = getattr(self.main.comp, "source", None)
        return "reads no source" if src is None else \
            f"declares no protocol of its read target (the result of {src!r})"

    # -- the run ------------------------------------------------------------------------
    def results(self) -> list:
        out = []
        try:
            details = self._construct()
            out.append(CheckResult("construct", self.device, True,
                                   f"{type(self.main.comp).__name__} {self.name!r}",
                                   details=details))
        except Exception as exc:  # noqa: BLE001 - reported
            out.append(CheckResult("construct", self.device, False, _exc_text(exc)))
            for name in _COMPONENT_CHECKS[1:]:
                out.append(CheckResult(name, self.device, True, "construct failed",
                                       skipped=True))
            return out
        for name in _COMPONENT_CHECKS[1:]:
            out.append(self._one(name))
        return out

    def _one(self, name: str) -> CheckResult:
        if name in _CPU_ONLY and self.dev.is_cuda:
            return CheckResult(name, self.device, True, "CPU only (on CUDA: capture.caller)",
                               skipped=True)
        if name in _CUDA_ONLY and not self.dev.is_cuda:
            return CheckResult(name, self.device, True, "CUDA only", skipped=True)
        if name in _NO_SOURCE and not self.declares_protocols():
            return CheckResult(name, self.device, True, self.no_protocol_reason(),
                               skipped=True)
        fn = getattr(self, "_" + name.replace(".", "_"))
        try:
            res = fn()
        except _Skip as exc:
            return CheckResult(name, self.device, True, str(exc), skipped=True)
        except AssertionError as exc:
            return CheckResult(name, self.device, False, str(exc).strip().splitlines()[0]
                               if str(exc).strip() else "assertion failed")
        except Exception as exc:  # noqa: BLE001 - reported
            return CheckResult(name, self.device, False, _exc_text(exc))
        msg, details = res if isinstance(res, tuple) else (res or "", {})
        return CheckResult(name, self.device, True, msg, details=details)

    # -- construct ----------------------------------------------------------------------
    def _construct(self) -> dict:
        self.main_producer = self.producer()
        self.main = self.run_on(self.main_producer)
        comp = self.main.comp
        self.name = comp.name
        if comp.layout is None or not isinstance(comp.layout, Layout):
            raise AssertionError("cc.add left component.layout unset")
        if comp.result is None:
            raise AssertionError("attach left component.result unset")
        if not _result_arrays(comp.result):
            raise AssertionError(f"component.result ({type(comp.result).__name__}) exposes "
                                 f"no wp.array: the kit cannot check it")
        if not isinstance(comp.stateful, bool):
            raise TypeError(f"stateful must be a bool, got {comp.stateful!r}")
        valid = getattr(_result_of(self.main_producer), "valid", None)
        n_valid = None if valid is None else int((valid.numpy() != 0).sum())
        if n_valid == 0:
            raise ValueError(f"the source has 0 valid envs of {self.E}: every check would "
                             f"be vacuous (an invalid env reads like a poisoned one in "
                             f"every run); pass another source or seeds")
        if self.kit.drive_fn is None:
            unknown = sorted(k for k in self.main.inputs if k not in _KNOWN_INPUTS)
            if unknown:
                raise ValueError(f"inputs {unknown} need drive= (the default drive "
                                 f"writes {list(_KNOWN_INPUTS)})")
        self.kit.drive(self.main, 0)
        self.main.on_generate(self.ctx, None)
        self.main.reset(self.ctx, self.ones())
        self.sync()
        # The attach contract: a chain attached now (no warm-up) steps like the warmed one.
        fresh = self.run_on(self.main_producer)
        self.main.copy_inputs_to(fresh)
        self.main.step(self.ctx)
        fresh.step(self.ctx)
        bad = _diff(_snap(self.main.owned), _snap(fresh.owned))
        if bad:
            raise AssertionError(
                f"a freshly attached chain's first step differs from one after "
                f"on_generate(ctx, None) and reset(ctx, all) in {_text(bad)}: attach must "
                f"end with on_generate(cc.ctx, None) and initialise the per-env state as "
                f"reset does")
        return {"valid_envs": n_valid, "owned": sorted(self.main.owned),
                "aliased": self.main.aliased}

    # -- requires.complete / topology --------------------------------------------------
    def _proxy(self, target, *, drop=(), layout=None):
        comp = self.main.comp
        req = [m for p in comp.requires for m in _members_of(p)]
        opt = [m for p in (comp.optional or ()) for m in _members_of(p) if m not in req]
        kw = {m: getattr(target, m, None) for m in dict.fromkeys(req + opt)}
        kw = {m: v for m, v in kw.items() if v is not None and m not in drop}
        lay = layout if layout is not None else _target_layout(target)
        if lay is not None:
            kw["layout"] = lay
        return types.SimpleNamespace(**kw)

    def _twin_on(self, run, proxy, what: str):
        try:
            return run.twin(target=proxy)
        except AttributeError as exc:
            raise AssertionError(f"attached to {what}, attach raised AttributeError "
                                 f"({exc}): it reads a member of its read target it does "
                                 f"not declare") from None

    def _same_as_twin(self, run, twin, what: str) -> None:
        for s in range(min(self.kit.steps, 12)):
            self.kit.drive(run, s)
            run.step(self.ctx, extra=(twin,))
            bad = _diff(_snap(run.owned), _snap(run.arrays_of(twin)))
            assert not bad, (f"step {s}: attached to {what}, owned outputs differ: "
                             f"{_text(bad)} (it reads a member it does not declare)")

    def _requires_complete(self):
        run = self.run_on(self.main_producer)
        comp, target = run.comp, run.target
        if not self.declares_protocols():
            if comp.source is not None:
                raise _Skip(self.no_protocol_reason())
            what = "an empty read target (only a layout) while declaring no read"
            lay = _target_layout(target) or run.cc.layout
            self._same_as_twin(run, self._twin_on(run, types.SimpleNamespace(layout=lay),
                                                  what), what)
            raise _Skip("reads no source (attached to an empty read target: the same "
                        "owned bytes)")
        req = list(dict.fromkeys(m for p in comp.requires for m in _members_of(p)))
        opt = [m for p in (comp.optional or ()) for m in _members_of(p)
               if m not in req and getattr(target, m, None) is not None]
        what = "a proxy with only the members of requires + optional"
        self._same_as_twin(run, self._twin_on(run, self._proxy(target), what), what)
        for m in req:
            try:
                run.twin(target=self._proxy(target, drop=(m,)))
            except TypeError:
                continue
            raise AssertionError(f"a read target without the required member {m!r} "
                                 f"attached (cc.add must raise TypeError)")
        for m in opt:
            run.twin(target=self._proxy(target, drop=(m,)))
        return (f"{len(req)} required, {len(opt)} optional members",
                {"required": req, "optional": opt})

    def _topology(self):
        run = self.main
        comp = run.comp
        lay = comp.layout
        opened = Layout(lay.num_envs, lay.stride, lay.device, False)
        proxy = self._proxy(run.target, layout=opened)
        if comp.topology == "closed":
            try:
                run.twin(target=proxy)
            except ValueError:
                return "closed: refuses a closed=False read target"
            raise AssertionError("topology='closed' but a closed=False read target "
                                 "attached (cc.add must raise ValueError)")
        run.twin(target=proxy)
        return "any: attaches to a closed=False read target"

    # -- result.stable ----------------------------------------------------------------
    def _result_stable(self):
        run = self.run_on(self.main_producer)
        comp = run.comp
        r0 = comp.result
        arrays0 = _result_arrays(r0)
        ptrs0 = {k: int(a.ptr) for k, a in arrays0.items()}
        aliased = [k for k in arrays0 if k not in run.owned]
        bad = [k for k in aliased if int(arrays0[k].ptr) not in run.alias_ptrs]
        assert not bad, (f"aliased result arrays {bad} point inside, not at, an array of "
                         f"the read target / upstream / producer")
        m = self.mask()
        self.kit.drive(run, 1)
        run.on_generate(self.ctx, None)
        run.on_generate(self.ctx, m)
        run.reset(self.ctx, m)
        for _ in range(2):
            for c in run.upstream:
                c.step(self.ctx)
            got = comp.step(self.ctx)
            assert got is r0, "step() must return component.result (the same instance)"
        assert comp.result is r0, "component.result was replaced"
        now = {k: int(a.ptr) for k, a in _result_arrays(comp.result).items()}
        assert now == ptrs0, f"result arrays moved: {_text(sorted(set(now.items()) ^ set(ptrs0.items())))}"
        return (f"{len(run.owned)} owned, {len(aliased)} aliased arrays",
                {"owned": sorted(run.owned), "aliased": sorted(aliased)})

    # -- determinism -------------------------------------------------------------------
    def _determinism(self):
        a = self.run_on(self.main_producer)
        b = self.run_on(self.twin_producer())
        m = self.mask()
        for s in range(self.kit.steps):
            self.kit.drive(a, s)
            a.copy_inputs_to(b)
            if s == self.kit.steps // 2:
                for r in (a, b):
                    r.on_generate(self.ctx, None)
                    r.reset(self.ctx, m)
            a.step(self.ctx)
            b.step(self.ctx)
            bad = _diff(_snap(a.owned), _snap(b.owned))
            assert not bad, f"step {s}: two equal runs differ in {_text(bad)}"
        return f"{self.kit.steps} steps"

    def twin_producer(self):
        p = getattr(self, "_twin", None)
        if p is None:
            p = self._twin = self.producer()
        else:
            p.seeds.reset_epoch()
            p.generate()
        return p

    # -- statelessness / reset.exact ----------------------------------------------------
    def _statelessness(self):
        if self.main.comp.stateful:
            raise _Skip("stateful (see reset.exact)")
        run = self.run_on(self.main_producer)
        m = self.mask()
        half = max(2, self.kit.steps // 2)
        for s in range(half):
            self.kit.drive(run, s)
            if s == half // 2:
                run.on_generate(self.ctx, m)
                run.reset(self.ctx, m)
            run.step(self.ctx)
        fresh = run.twin()
        self.kit.drive(run, half)
        run.step(self.ctx, extra=(fresh,))
        bad = _diff(_snap(run.owned), _snap(run.arrays_of(fresh)))
        assert not bad, (f"after {half} steps, step differs from a fresh component's in "
                         f"{_text(bad)}: it keeps state (declare stateful=True), or its "
                         f"attach does not end with on_generate(cc.ctx, None)")
        return f"history of {half} steps"

    def _reset_exact(self):
        if not self.main.comp.stateful:
            raise _Skip("stateless (see statelessness)")
        run = self.run_on(self.main_producer)
        never = run.twin()
        m = self.mask()
        mk = m.numpy() != 0
        masked = [e for e in range(self.E) if mk[e]]
        unmasked = [e for e in range(self.E) if not mk[e]]
        k = max(2, self.kit.steps // 2)
        fresh = None
        changed = 0
        for s in range(k + self.kit.steps):
            self.kit.drive(run, s)
            if s == k:
                run.comp.reset(self.ctx, m)
                fresh = run.twin()
            run.step(self.ctx, extra=(never,) if fresh is None else (never, fresh))
            got = _snap(run.owned)
            ref = _snap(run.arrays_of(never))
            if fresh is None:
                bad = _diff(got, ref)
                assert not bad, f"step {s}: two equal components differ in {_text(bad)}"
                continue
            bad = _row_diff(got, ref, self.E, unmasked)
            assert not bad, f"step {s}: reset changed unmasked rows {_text(bad)}"
            bad = _row_diff(got, _snap(run.arrays_of(fresh)), self.E, masked)
            assert not bad, (f"step {s}: masked rows after reset differ from a fresh "
                             f"component's: {_text(bad)}")
            changed += len(_row_diff(got, ref, self.E, masked))
        return (f"reset at step {k}; masked rows that differ from the never-reset twin: "
                f"{changed}", {"masked_rows_changed": changed})

    # -- regenerate.* -------------------------------------------------------------------
    def _p3_producer(self, m):
        if self._p3 is None:
            p = self.producer()
            p.generate(advance=True, mask=m)
            self._p3 = p
        return self._p3

    def _regen_run(self, with_reset: bool):
        """The masked-regeneration trajectory; returns ``(per-step owned snapshots,
        masked rows changed, compared rows)``."""
        m = self.mask()
        mk = m.numpy() != 0
        masked = [e for e in range(self.E) if mk[e]]
        unmasked = [e for e in range(self.E) if not mk[e]]
        p1 = self.twin_producer()
        r1 = self.run_on(p1)
        r2 = self.run_on(self.main_producer)
        r3 = None
        k = max(2, self.kit.steps // 3)
        snaps, changed, compared = [], 0, 0
        for s in range(self.kit.steps):
            if s == k:
                p1.generate(advance=True, mask=m)
                r1.on_generate(self.ctx, m)
                if with_reset:
                    r1.reset(self.ctx, m)
                r3 = self.run_on(self._p3_producer(m))
            self.kit.drive(r1, s)
            r1.copy_inputs_to(r2)
            if r3 is not None:
                r1.copy_inputs_to(r3)
            r1.step(self.ctx)
            r2.step(self.ctx)
            if r3 is not None:
                r3.step(self.ctx)
            got, ref = _snap(r1.owned), _snap(r2.owned)
            snaps.append(got)
            if r3 is None:
                bad = _diff(got, ref)
                assert not bad, f"step {s}: two equal runs differ in {_text(bad)}"
                continue
            bad = _row_diff(got, ref, self.E, unmasked)
            assert not bad, (f"step {s}: after a masked regeneration, unmasked rows differ "
                             f"from the never-advanced twin: {_text(bad)}")
            bad = _row_diff(got, _snap(r3.owned), self.E, masked)
            assert not bad, (f"step {s}: masked rows differ from a fresh chain on a twin "
                             f"advanced before attaching: {_text(bad)}")
            changed += len(_row_diff(got, ref, self.E, masked))
            compared += len(masked) * len(got)
        return snaps, changed, compared

    def _regenerate_masked(self):
        snaps, changed, compared = self._regen_run(True)
        self._regen_snaps = snaps
        return (f"masked rows changed: {changed} / {compared}",
                {"masked_rows_changed": changed, "masked_rows": compared})

    def _self_reset_applies(self) -> None:
        if not self.main.comp.stateful:
            raise _Skip("not applicable: stateless")
        if not self.declares_protocols() and self.main.comp.source is None:
            raise _Skip("not applicable: reads no source")
        if getattr(self.main.target, "generation", None) is None:
            raise _Skip("not applicable: the read target carries no generation")

    def _regenerate_self_reset(self):
        self._self_reset_applies()
        if self._regen_snaps is None:
            raise _Skip("regenerate.masked did not complete")
        snaps, _, _ = self._regen_run(False)
        for s, (a, b) in enumerate(zip(snaps, self._regen_snaps)):
            bad = _diff(a, b)
            assert not bad, (f"step {s}: without reset(mask) after the masked regeneration, "
                             f"owned outputs differ from the run with it: {_text(bad)} (no "
                             f"generation-aware self-reset)")
        return f"{len(snaps)} steps equal to the run with reset"

    # -- invalid.sentinel ------------------------------------------------------------
    def _invalid_sentinel(self):
        from ..stages.commit import _commit_envs  # noqa: PLC0415

        if not self.main.owned and self.main.aliased:
            raise _Skip("every result array is aliased (its consumers are checked with "
                        "it upstream)")
        src = _result_of(self.main_producer)
        K, P = src.clone(), src.clone()
        fail = _i32([e % 2 for e in range(self.E)], self.device)
        _commit_envs(K, fail, poison=False, ctx=self.ctx)
        _commit_envs(P, fail, poison=True, ctx=self.ctx)
        self.sync()
        cap = self.main.cc.capacity
        rk, rp = self.run_static(K, cap), self.run_static(P, cap)
        m = self.mask()
        for s in range(self.kit.steps):
            self.kit.drive(rk, s, geom=K)
            rk.copy_inputs_to(rp)
            if s == self.kit.steps // 2:
                for r in (rk, rp):
                    r.on_generate(self.ctx, None)
                    r.reset(self.ctx, m)
            rk.step(self.ctx)
            rp.step(self.ctx)
            bad = _diff(_snap(rk.owned), _snap(rp.owned))
            assert not bad, (f"step {s}: an invalid env (valid = 0, geometry kept) does not "
                             f"read like a poisoned one in {_text(bad)}")
        return f"{int(sum(e % 2 for e in range(self.E)))} of {self.E} envs failed"

    # -- capacity.clamp -----------------------------------------------------------------
    def _capacity_clamp(self):
        from ..tools import ComponentContext  # noqa: PLC0415

        read: list = []

        class _Logged(dict):
            def get(self, key, default=None):
                read.append(key)
                return super().get(key, default)

            def __getitem__(self, key):
                read.append(key)
                return super().__getitem__(key)

        class _LoggingContext(ComponentContext):
            @property
            def capacity(self):
                return _Logged(self._capacity)

        probe_cc = _LoggingContext.for_producer(self.main_producer, ctx=self.ctx)
        src = _result_of(self.main_producer)
        for f in self.kit.upstream:
            probe_cc.add(f(probe_cc), src)
        read.clear()
        probe_cc.add(self.kit.factory(probe_cc), src)
        keys = sorted(set(read))
        if not keys:
            raise _Skip("reads no capacity key")
        base = dict(self.main.cc.capacity)
        missing = [k for k in keys if k not in base]
        if missing:
            raise _Skip(f"the producer declares no {missing}")
        last = None
        for scale in (4, 8, 16, 32):
            cap = {k: (v / scale if k in keys else v) for k, v in base.items()}
            batch = _Run(self.kit, src, ComponentContext(self.main.cc.layout, cap,
                                                         ctx=self.ctx))
            tr = getattr(batch.comp.result, "truncated", None)
            if tr is None:
                tr = getattr(getattr(batch.comp, "utility", None), "truncated", None)
            last = (scale, batch, tr)
            if tr is None or bool((tr.numpy() != 0).any()):
                break
        scale, batch, tr = last
        self.kit.drive(batch, 0)
        batch.step(self.ctx)
        got = _snap(batch.owned)
        for e in range(self.E):
            view = src.view(e, e + 1)
            one = _Run(self.kit, view, ComponentContext(view.validate(), cap, ctx=self.ctx))
            for k, a in batch.inputs.items():
                n = int(a.shape[0]) // self.E
                wp.copy(one.inputs[k], a, src_offset=e * n, count=n)
            one.step(self.ctx)
            ref = _snap(one.owned)
            for k, v in got.items():
                rows = _rows(v, self.E)
                assert rows is not None, f"owned array {k!r} is not per env"
                assert np.array_equal(rows[e], _bytes(ref[k])), (
                    f"capacity/{scale}: env {e}'s rows of {k!r} differ from the same chain "
                    f"on that env's one-env view (a write past its own rows)")
        n_tr = None if tr is None else int((tr.numpy() != 0).sum())
        if tr is not None:
            assert n_tr > 0, (f"no env reports truncation with {keys} divided by {scale} "
                              f"(the clamp check is vacuous)")
        return (f"{keys} / {scale}: rows equal per env" +
                ("" if n_tr is None else f"; {n_tr} env(s) truncated"),
                {"keys": keys, "scale": scale, "truncated": n_tr})

    # -- source.view ----------------------------------------------------------------
    def _source_view(self):
        src = _result_of(self.main_producer)
        lay = src.validate()
        E, S = int(lay.num_envs), int(lay.stride)
        attrs = {k: a.dtype for k, a in dict(getattr(src, "attributes", None) or {}).items()}
        kw = {"attributes": attrs} if attrs else {}
        big = type(src).alloc(Layout(E + 2, S, lay.device, True), **kw)
        for a in result_arrays(big).values():
            host = a.numpy()
            host.view(np.uint8)[...] = _BYTE_POISON
            a.assign(host)
        view = big.view(1, E + 1)
        view.copy_from(src, ctx=self.ctx)
        self.sync()
        before = _snap(result_arrays(big))
        rv = self.run_static(view, self.main.cc.capacity)
        rs = self.run_on(self.main_producer)
        m = self.mask()
        for s in range(self.kit.steps):
            self.kit.drive(rs, s)
            rs.copy_inputs_to(rv)
            if s == self.kit.steps // 2:
                for r in (rs, rv):
                    r.on_generate(self.ctx, m)
                    r.reset(self.ctx, m)
            rs.step(self.ctx)
            rv.step(self.ctx)
            bad = _diff(_snap(rs.owned), _snap(rv.owned))
            assert not bad, (f"step {s}: attached to big.view(1, E + 1) of a poisoned "
                             f"batch, owned outputs differ from the standalone run: "
                             f"{_text(bad)} (a read outside its env range)")
        bad = _diff(before, _snap(result_arrays(big)))
        assert not bad, (f"the chain wrote into its source (the viewed batch, or the "
                         f"poisoned envs around it): {_text(bad)}")
        return f"{self.kit.steps} steps; the source is unchanged"

    # -- hot_path.* ---------------------------------------------------------------------
    def _hot_calls(self, run, ctx=None):
        ctx = ctx or self.ctx
        c = run.comp
        m = self._hot_mask
        return [("on_generate(ctx, None)", lambda: c.on_generate(ctx, None)),
                ("on_generate(ctx, m)", lambda: c.on_generate(ctx, m)),
                ("reset(ctx, m)", lambda: c.reset(ctx, m)),
                ("step(ctx)", lambda: c.step(ctx)),
                ("step(ctx) again", lambda: c.step(ctx))]

    def _hot_guard(self):
        if self._hot is None:
            self._hot_mask = self.mask()
            warm = self.run_on(self.main_producer)
            self.kit.drive(warm, 0)
            for _, f in self._hot_calls(warm):  # module loads, process-wide, unguarded
                f()
            # A FRESH chain, guarded from its first hot call: work done only once (a
            # lazy allocation, a one-time readback or launch) is caught.
            run = self.run_on(self.main_producer)
            self.kit.drive(run, 0)
            self.sync()
            calls = self._hot_calls(run)
            pkg = _top_package(type(run.comp).__module__)
            modules = tuple(dict.fromkeys(("track_gen", pkg)))
            guard = HotPathGuard(modules=modules, strict=False)
            with guard:
                for _, f in calls:
                    f()
            self._hot = guard
        return self._hot

    def _hot_category(self, category: str):
        guard = self._hot_guard()
        bad = [v for v in guard.violations if v.category == category]
        assert not bad, f"{bad[0]}" + (f" (+{len(bad) - 1} more)" if len(bad) > 1 else "")
        return "on_generate x2, reset, step x2"

    def _hot_path_host_sync(self):
        return self._hot_category("host_sync")

    def _hot_path_allocation(self):
        return self._hot_category("allocation")

    def _hot_path_stream(self):
        return self._hot_category("stream")

    def _hot_path_capture(self):
        return self._hot_category("capture")

    def _hot_path_topology(self):
        run = self.run_on(self.main_producer)
        self._hot_mask = self.mask()
        self.kit.drive(run, 0)
        calls = self._hot_calls(run)
        for label, f in calls[:4]:
            t1, t2 = LaunchTrace(), LaunchTrace()
            with t1:
                f()
            with t2:
                f()
            diff = t1.diff(t2)
            assert not diff, f"{label}: two calls emit different launches: {diff[0]}"
        util = getattr(run.comp, "utility", None)
        flag = getattr(util, "_recorded", None)
        if flag is None:
            return "same trace per hot method"
        cap_ctx = _CapturingContext(self.ctx.device, self.ctx.stream)
        try:
            util._recorded = False
            t_plain = LaunchTrace()
            with t_plain:
                run.comp.step(self.ctx)
            assert util._recorded is False, "an eager step armed the rebind flag"
            t_armed = LaunchTrace()
            with t_armed:
                run.comp.step(cap_ctx)
            assert util._recorded is True, ("a step under a capturing context did not arm "
                                            "the rebind flag")
            t_after = LaunchTrace()
            with t_after:
                run.comp.step(self.ctx)
        finally:
            util._recorded = flag
        for label, t in (("armed", t_armed), ("after arming", t_after)):
            diff = t_plain.diff(t)
            assert not diff, f"step with the rebind flag {label} emits differently: {diff[0]}"
        return "same trace per hot method; the rebind probe changes no emission"

    def _hot_path_buffers(self):
        run = self.run_on(self.main_producer)
        self._hot_mask = self.mask()
        self.kit.drive(run, 0)
        calls = self._hot_calls(run)
        allowed = buffers_of(run.comp, *run.upstream, run.source, run.target, run.inputs,
                             self._hot_mask)
        inputs = BufferSet()
        for k, a in run.inputs.items():
            inputs.add(a, k)
        traces = []
        for label, f in calls:
            t = LaunchTrace()
            with t:
                f()
            traces.append((label, t))
        foreign = [x for _, t in traces for x in t.foreign_pointers(allowed)]
        assert not foreign, (f"{len(foreign)} pointer(s) outside the component / upstream / "
                             f"source / inputs, e.g. {foreign[0][0].kernel} argument "
                             f"{foreign[0][1]}")
        # Roadmap PR15: the regeneration-body methods read no bound input.
        for label, t in traces:
            if label.startswith("step"):
                continue
            for r in t:
                for i, ptr in r.arrays:
                    loc = inputs.locate(ptr) if ptr else None
                    if loc is not None:
                        method = label.split("(")[0]
                        raise AssertionError(
                            f"{method} launches with bound input {loc[0]!r} ({label}: "
                            f"{r.kernel} argument {i}): the course self-graph would bake "
                            f"it (the regeneration body is keyed on (advance, mask) "
                            f"only); read inputs in step() (a respawn snapshot: reset "
                            f"arms a per-env flag that step consumes)")
        return f"{sum(len(t) for _, t in traces)} launches/copies"

    # -- capture.* (CUDA) ---------------------------------------------------------------
    def _capture(self, *, with_reset: bool) -> dict:
        status: dict = {}
        E = self.E
        for mode in self.kit.modes:
            torch = None
            if mode in ("external", "torch"):
                torch = _torch_cuda()
                if torch is None:
                    status[mode] = "skipped: torch with CUDA is not available"
                    continue
            pc, pt = self.producer(), self.producer()
            rc, rt = self.run_on(pc), self.run_on(pt)
            done_c = wp.zeros(E, dtype=wp.int32, device=self.device)
            done_t = wp.zeros(E, dtype=wp.int32, device=self.device)
            flag_c = wp.zeros(1, dtype=wp.int32, device=self.device)
            flag_t = wp.zeros(1, dtype=wp.int32, device=self.device)
            # Load the any_set modules outside the capture (a zero mask: writes the flag).
            any_set(LaunchContext.for_call(self.device), done_t, flag_t)
            self.sync()

            recorded: dict = {}

            def body(ctx, run=rc, prod=pc, done=done_c, flag=flag_c, reset=with_reset):
                recorded["conditional"] = _caller_chain(ctx, run, prod.program, done,
                                                        flag, reset=reset)

            if mode == "warp":
                s = side_stream(self.dev)
                self.sync()
                with recording(self.dev, s) as cap:
                    body(LaunchContext.for_call(self.device, s))

                def replay(_g=cap.graph, _s=s):
                    wp.capture_launch(_g, stream=_s)
                    wp.synchronize_device(self.device)
            else:
                ordinal = self.dev.ordinal
                with torch.cuda.device(ordinal):
                    g = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize(ordinal)
                    with mempool_disabled(self.dev):
                        with torch.cuda.graph(g):
                            if mode == "external":
                                from ..interop.torch import registered  # noqa: PLC0415
                                with registered(graph=g) as ws, capture_launch_forbidden():
                                    body(LaunchContext.for_call(self.device, ws))
                            else:
                                with capture_launch_forbidden():
                                    body(LaunchContext.for_call(self.device))
                    torch.cuda.synchronize(ordinal)

                def replay(_g=g, _torch=torch, _ordinal=ordinal):
                    with _torch.cuda.device(_ordinal):
                        _g.replay()
                        _torch.cuda.synchronize(_ordinal)
            ectx = LaunchContext.for_call(self.device)
            # Under a conditional node the replay skips the masked part on an all-zero
            # done: the eager twin skips it then too, so a whole-batch on_generate (or one
            # writing an array that step does not rewrite) leaves the poison in both runs.
            conditional = bool(recorded.get("conditional"))
            for i in range(self.kit.steps):
                sched = _done_schedule(i, E)
                done = _i32(sched, self.device)
                wp.copy(done_c, done)
                wp.copy(done_t, done)
                self.kit.drive(rt, i)
                rt.copy_inputs_to(rc)
                for a in (*rc.owned.values(), *rt.owned.values()):
                    poison(a)
                self.sync()
                replay()
                _caller_chain(ectx, rt, pt.program, done_t, flag_t, reset=True,
                              skip=conditional and not any(sched))
                self.sync()
                bad = _diff(_snap(rc.owned), _snap(rt.owned))
                assert not bad, (f"{mode} capture: replay {i} differs from the eager twin "
                                 f"in {_text(bad)}")
            status[mode] = "ok"
        return status

    def _capture_caller(self):
        status = self._capture(with_reset=True)
        return (", ".join(f"{k}: {v}" for k, v in status.items()), status)

    def _capture_self_reset(self):
        self._self_reset_applies()
        status = self._capture(with_reset=False)
        return (", ".join(f"{k}: {v}" for k, v in status.items()), status)


# ---------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------

def _seed_values(seeds, E: int) -> np.ndarray:
    if seeds is None:
        return (np.arange(E, dtype=np.int64) * 7 + 3).astype(np.int32)
    if isinstance(seeds, (int, np.integer)) and not isinstance(seeds, bool):
        return ((int(seeds) + np.arange(E, dtype=np.int64) + 2**31) % 2**32
                - 2**31).astype(np.int32)
    arr = np.asarray(seeds, dtype=np.int64).reshape(-1)
    if arr.shape[0] != E:
        raise ValueError(f"seeds must hold num_envs={E} values, got {arr.shape[0]}")
    return ((arr + 2**31) % 2**32 - 2**31).astype(np.int32)


def check_component(factory: "Callable", *, source="track",
                    upstream: "Sequence[Callable]" = (), devices=None, num_envs: int = 8,
                    steps: int = 32, seeds=None, drive: "Callable | None" = None,
                    modes=CAPTURE_MODES) -> Report:
    """Run a component through the component contract's checks (see the module
    docstring) and return the :class:`~track_gen.testing.Report`.

    Args:
        factory: ``factory(cc) -> Component``: a fresh, unattached component (``cc`` is
            the :class:`~track_gen.tools.ComponentContext` it will be attached through:
            its ``layout`` gives the device and ``E``).
        source: ``"track"`` (``presets.track``), ``"gates"`` (``presets.gates`` with
            ``gate_width=0.1``), or a ``PipelineSpec`` / ``TrackGenConfig`` /
            ``GateGenConfig`` (its ``num_envs``
            / ``device`` replaced). Every source uses ``on_invalid="keep"`` so the
            checks can inspect raw invalid geometry.
        upstream: Factories of the components it reads (``after``, ``source``), attached
            first, in order.
        devices: Devices to run on (default: ``"cpu"``, plus ``"cuda:0"`` when
            available).
        num_envs: ``E`` (``>= 2``).
        steps: Trajectory length of the byte checks (``>= 6``).
        seeds: ``None`` (``7 * e + 3``), an int base seed (``base + e``) or ``E``
            values.
        drive: ``drive(step, inputs, source)`` writing the bound ``inputs`` (``name ->
            wp.array``) before each step; default: see the module docstring.
        modes: The capture modes of ``capture.*`` (CUDA).

    Returns:
        The :class:`~track_gen.testing.Report` (``kind="component"``; the checks of
        ``CHECKS["component"]`` per device).
    """
    if not callable(factory):
        raise TypeError(f"check_component: factory must be a callable factory(cc) -> "
                        f"Component, got {type(factory).__name__}")
    upstream = tuple(upstream)
    for f in upstream:
        if not callable(f):
            raise TypeError(f"check_component: upstream entries must be factories, got "
                            f"{type(f).__name__}")
    E = int(num_envs)
    if E < 2:
        raise ValueError(f"num_envs must be >= 2, got {num_envs!r}")
    if int(steps) < 6:
        raise ValueError(f"steps must be >= 6, got {steps!r}")
    modes = tuple(modes)
    unknown = [m for m in modes if m not in CAPTURE_MODES]
    if unknown:
        raise ValueError(f"unknown capture mode(s) {unknown}; known: {CAPTURE_MODES}")
    kit = _Kit(factory, upstream, source, E, int(steps), _seed_values(seeds, E), drive,
               modes)
    results, name = [], None
    for dev in _devices(devices):
        run = _DeviceRun(kit, dev)
        results.extend(run.results())
        name = name or run.name
    return Report("component", name or getattr(factory, "__name__", "component"), results)


# ---------------------------------------------------------------------------------------
# The built-in components (private: the library's own tests)
# ---------------------------------------------------------------------------------------

def _raw_discs(cc):
    """Raw-mode ``DiscCollision`` on the kit's device: two rings of discs around the
    origin (``D = 16`` per env), ``count`` ``16 - e % 3``, env 1 invalid."""
    from .. import tools  # noqa: PLC0415

    E, dev = int(cc.layout.num_envs), cc.layout.device
    ang = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.concatenate([np.stack([r * np.cos(ang + 0.2 * r), r * np.sin(ang + 0.2 * r),
                                     np.zeros(8)], -1) for r in (0.5, 1.0)])
    discs = np.tile(ring.astype(np.float32), (E, 1))
    count = np.array([16 - e % 3 for e in range(E)], np.int32)
    valid = np.ones(E, np.int32)
    valid[1 % E] = 0
    return tools.DiscCollision(
        0.08, discs=wp.array(discs, dtype=wp.vec3f, device=dev),
        count=wp.array(count, dtype=wp.int32, device=dev),
        valid=wp.array(valid, dtype=wp.int32, device=dev))


def _builtin_components() -> dict:
    """``id -> (factory, upstream, source)`` of the built-in adapters the library's own
    tests check (private: not part of the public API)."""
    from .. import tools  # noqa: PLC0415

    return {
        "band_collision[segments]": (lambda cc: tools.BandCollision("segments"), (),
                                     "track"),
        "band_collision[sdf]": (lambda cc: tools.BandCollision("sdf", sdf_resolution=16),
                                (), "track"),
        "disc_collision[posts]": (lambda cc: tools.DiscCollision(0.03), (), "gates"),
        "disc_collision[raw]": (_raw_discs, (), "track"),
        "frame_collision": (lambda cc: tools.FrameCollision(0.05, 0.02, 0.02),
                            (lambda cc: tools.GateCheckpoints(),
                             lambda cc: tools.Progress()), "gates"),
        "props[points]": (lambda cc: tools.Props(0.2), (), "track"),
        "props[segments,inner]": (lambda cc: tools.Props(0.2, boundary="inner",
                                                         mode="segments"), (), "track"),
        "checkpoints": (lambda cc: tools.Checkpoints(0.5), (), "track"),
        "gate_checkpoints": (lambda cc: tools.GateCheckpoints(), (), "gates"),
        "progress[track]": (lambda cc: tools.Progress(),
                            (lambda cc: tools.Checkpoints(0.5),), "track"),
        "progress[gates]": (lambda cc: tools.Progress(),
                            (lambda cc: tools.GateCheckpoints(),), "gates"),
        "localizer[track,cold]": (lambda cc: tools.Localizer(), (), "track"),
        "localizer[track,warm]": (lambda cc: tools.Localizer(warm_window=3), (), "track"),
        "localizer[line,cold]": (lambda cc: tools.Localizer(source="centerline"),
                                 (lambda cc: tools.CenterlineFromGates(8),), "gates"),
        "localizer[line,warm]": (lambda cc: tools.Localizer(warm_window=2,
                                                            source="centerline"),
                                 (lambda cc: tools.CenterlineFromGates(8),), "gates"),
        "heightfield": (lambda cc: tools.HeightFieldBake(16), (), "track"),
        "centerline_from_gates": (lambda cc: tools.CenterlineFromGates(8), (), "gates"),
    }
