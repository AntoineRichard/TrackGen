"""``Component``, ``ComponentContext`` and ``InputSpec`` (roadmap PR14).

A component is a runtime consumer of a producer's result (a ``Track``, a
``GateSequence``, or another component's result), with a two-phase lifecycle: a cold
:meth:`Component.attach` that checks what it reads, allocates every buffer and does one
first ``on_generate``, then hot methods (``on_generate`` after every generation,
``step`` per sim step, ``reset`` on respawns) that launch through the
:class:`~track_gen.exec.LaunchContext` they are given, allocate nothing and never
synchronize. :class:`ComponentContext` attaches components in dependency order and
carries the producer's static capacities (:meth:`ComponentContext.for_producer`).
"""
from __future__ import annotations

import dataclasses
import math
import numbers
import sys
import types
from typing import Mapping

import warp as wp

from .._src.types import _MEMBER_SCHEMA
from ..exec.context import LaunchContext, _rebind_close, _rebind_open
from ..ops.check import check_array as _check_arr
from ..results import Layout, _check_and_bind, _members_of, _protocols

__all__ = ["Component", "ComponentContext", "InputSpec"]

#: The headroom of the ``max_loop_length`` capacity key over ``N_max * spacing``.
LOOP_HEADROOM = 1.5


def _user_stacklevel() -> int:
    """The ``stacklevel`` (for a ``warnings.warn`` in the CALLER of this function) of the
    first frame outside the ``track_gen`` package: the user's call."""
    frame, level = sys._getframe(1), 1
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if name != "track_gen" and not name.startswith("track_gen."):
            break
        frame, level = frame.f_back, level + 1
    return level


@dataclasses.dataclass(frozen=True)
class InputSpec:
    """One bound input of a component: a ``[E * slots]`` array of ``dtype``.

    Attributes:
        dtype: The Warp dtype (``wp.vec3f``, ``wp.quatf``, ``wp.vec2f``,
            ``wp.float32``, ...).
        slots: Elements per env: the array is ``[E * slots]``.
        required: Whether :meth:`Component.bind` must be given it (after ``default``
            routing). An absent optional input reaches :meth:`Component.on_bind` as
            ``None`` (``def on_bind(self, position, speed=None)`` style hooks work, and so
            does ``def on_bind(self, position, speed)``).
        default: Another input name whose array is used when this one is not given
            (e.g. ``box_position`` defaults to ``position`` with one box per env).
        doc: Free text (not part of the equality).
    """

    dtype: type
    slots: int = 1
    required: bool = True
    default: "str | None" = None
    doc: str = dataclasses.field(default="", compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.slots, bool) or not isinstance(self.slots, numbers.Integral) \
                or int(self.slots) < 1:
            raise ValueError(f"InputSpec.slots must be an int >= 1, got {self.slots!r}")
        object.__setattr__(self, "slots", int(self.slots))
        if not isinstance(self.required, bool):
            raise TypeError(f"InputSpec.required must be a bool, got {self.required!r}")
        if self.default is not None and not isinstance(self.default, str):
            raise TypeError(f"InputSpec.default must be an input name or None, got "
                            f"{self.default!r}")

    def check(self, name: str, array, layout: Layout, *, consumer: str) -> None:
        """Check ``array`` against this spec on ``layout``: a contiguous 1-D
        ``[num_envs * slots]`` array of :attr:`dtype` on the layout's device.

        Raises:
            ValueError: naming ``consumer`` and ``name``.
        """
        n = int(layout.num_envs) * self.slots
        try:
            _check_arr(name, array, (n,), self.dtype, layout.device)
        except ValueError as exc:
            raise ValueError(f"{consumer}: input {exc} ([E * slots] = [{layout.num_envs} "
                             f"* {self.slots}])") from None
        if not array.is_contiguous:
            raise ValueError(f"{consumer}: input {name} must be contiguous")


class Component:
    """A runtime consumer of a producer's result, driven by a ``Course`` (or by hand).

    A plain base class with immutable defaults (subclass it; the library duck-types and
    never ``isinstance``-checks a component). Declare, as class attributes or in
    ``__init__``:

    - ``name``: unique among the components attached to one :class:`ComponentContext`;
    - ``source``: the read target: ``None`` for the producer's result given to
      :meth:`ComponentContext.add`, else the ``name`` of the component whose ``result``
      it reads;
    - ``requires``: capability protocols (:mod:`track_gen.results`) of the read target
      whose members are all read (strict: a missing one raises ``TypeError``);
    - ``optional``: protocols whose members are read when present (``valid``,
      ``generation``: checked like the required ones when present);
    - ``topology``: ``"closed"`` (refuses a read target with ``layout.closed=False``) or
      ``"any"``;
    - ``stateful``: whether it keeps per-env runtime state that :meth:`reset` clears;
    - ``inputs``: ``name -> InputSpec`` of the arrays :meth:`bind` takes;
    - ``after``: names of other components it reads (attached first).

    :meth:`ComponentContext.add` sets :attr:`layout` (the read target's layout) and calls
    :meth:`attach`, which sets :attr:`result` (allocated once, never replaced). Override
    :meth:`attach`, :meth:`on_bind` and the hot methods; never :meth:`bind`. (The
    built-in adapters also expose the legacy utility they wrap as ``utility``; it is not
    part of the component contract.)

    Contracts (checked by :func:`track_gen.testing.check_component`): ``attach`` is cold
    (once, outside any capture), allocates every buffer, launches through ``cc.ctx``,
    ends with one ``on_generate(cc.ctx, None)`` and loads the module of every kernel its
    hot methods launch; the hot methods launch through the ``ctx`` they are given only,
    never allocate, synchronize, read back or change Python state that alters a later
    emission; ``on_generate(ctx, mask)`` leaves the rows and state of envs with
    ``mask[e] == 0`` bitwise unchanged (a whole-batch recompute qualifies); ``step``
    returns :attr:`result`; ``reset(ctx, mask)`` clears the per-env runtime state where
    ``mask[e] != 0`` and nothing else (a stateless component emits nothing).

    In a ``Course`` (roadmap PR15): ``on_generate`` and ``reset`` run in the masked
    regeneration body, under a CUDA conditional node where the capture allows one, so
    they read and write no bound input (``step`` reads them; a respawn snapshot of an
    input belongs in ``step``: ``reset`` arms a per-env flag that the next ``step``
    consumes) and emit only kernels, ``ctx.fill`` and device-to-device ``ctx.copy`` on
    ``ctx``'s stream (no copy to or from host or pinned memory, no event, no other
    stream); ``reset`` reads only the component's own state and the mask (resets run in
    reverse order, before the components they read); a component that overrides
    ``step`` is an entry of the course's ``StepResult`` (its :attr:`result`), one that
    keeps the base ``step`` (per-generation work) is read through
    ``course.components[name].result``.
    """

    # -- declaration ---------------------------------------------------------------
    name: str = "component"
    source: "str | None" = None
    requires: tuple = ()
    optional: tuple = ()
    topology: str = "closed"
    stateful: bool = False
    inputs: Mapping = types.MappingProxyType({})
    after: tuple = ()
    # -- set by ComponentContext.add / attach -----------------------------------------
    layout: "Layout | None" = None
    result: object = None

    # -- lifecycle -----------------------------------------------------------------------
    def attach(self, source, cc: "ComponentContext") -> None:
        """Cold, once: allocate, set :attr:`result`, end with ``on_generate(cc.ctx,
        None)`` and load the hot kernels' modules. ``source`` is the read target,
        already checked against ``requires`` / ``optional``."""
        raise NotImplementedError(f"{type(self).__name__} must implement attach()")

    def bind(self, **arrays: wp.array) -> None:
        """Bind the stable input arrays (cold, before recording :meth:`step`).

        Routes ``InputSpec.default`` (an input not given takes the array of its
        ``default`` name), refuses unknown names and missing required ones
        (``TypeError``), checks each array against its spec on :attr:`layout`
        (``ValueError``; a routed input's message names the argument it was routed
        from), then calls :meth:`on_bind` with every declared input (``None`` for an
        absent optional one). A name that is only the ``default`` of a declared input is
        accepted (and routed). The built-in adapters warn once
        (:class:`~track_gen.exec.RebindAfterCaptureWarning`) when an array is replaced
        after a hot method that reads it was recorded into a capture (their wrapped
        utilities arm that probe); a component of your own warns only if its hot methods
        do the same: record the graph again after rebinding.
        """
        token, hits = _rebind_open()
        try:
            self._bind_inputs(dict(arrays))
        finally:
            _rebind_close(token, hits, stacklevel=2)

    def _bind_inputs(self, arrays: dict) -> None:
        if self.layout is None:
            raise RuntimeError(f"{self.name}: bind() before attach (attach it with "
                               f"ComponentContext.add first)")
        self.on_bind(**_route(self, arrays))

    def on_bind(self, **arrays: wp.array) -> None:
        """Hook: the checked input arrays (pin them; an absent optional input is
        ``None``). Default: nothing."""

    def on_generate(self, ctx: LaunchContext, mask: "wp.array | None") -> None:
        """Hot: after every generation of the source (``mask``: its ``[E]`` int32 mask,
        or ``None`` for the whole batch). Default: nothing."""

    def step(self, ctx: LaunchContext):
        """Hot: per sim step; returns :attr:`result`. Default: returns it."""
        return self.result

    def reset(self, ctx: LaunchContext, mask: wp.array) -> None:
        """Hot: clear per-env runtime state where ``mask[e] != 0``. Default: nothing."""


def _declared_protocols(value, what: str, owner: str) -> tuple:
    group = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if not group:
        return ()
    try:
        return _protocols(group)
    except TypeError:
        raise TypeError(f"{owner}.{what} must be a tuple of track_gen.results "
                        f"capability protocols, got {value!r}") from None


def _check_declaration(comp) -> "tuple[tuple, tuple]":
    """Type-check a component's declaration; returns ``(requires, optional)``."""
    owner = type(comp).__name__
    if not isinstance(getattr(comp, "name", None), str) or not comp.name:
        raise TypeError(f"{owner}.name must be a non-empty str, got "
                        f"{getattr(comp, 'name', None)!r}")
    src = getattr(comp, "source", None)
    if src is not None and not isinstance(src, str):
        raise TypeError(f"{owner}.source must be None or a component name, got {src!r}")
    if getattr(comp, "topology", "closed") not in ("closed", "any"):
        raise ValueError(f"{owner}.topology must be 'closed' or 'any', got "
                         f"{comp.topology!r}")
    after = getattr(comp, "after", ())
    if isinstance(after, str) or not all(isinstance(a, str) for a in after):
        raise TypeError(f"{owner}.after must be a tuple of component names, got {after!r}")
    inputs = getattr(comp, "inputs", {})
    if not isinstance(inputs, Mapping) or \
            not all(isinstance(k, str) and isinstance(v, InputSpec)
                    for k, v in inputs.items()):
        raise TypeError(f"{owner}.inputs must map input names to InputSpec, got "
                        f"{inputs!r}")
    for k, spec in inputs.items():
        if spec.default is not None and spec.default == k:
            raise ValueError(f"{owner}.inputs[{k!r}].default names itself")
    return (_declared_protocols(getattr(comp, "requires", ()), "requires", owner),
            _declared_protocols(getattr(comp, "optional", ()), "optional", owner))


def _route(comp, arrays: Mapping) -> dict:
    """The validation half of :meth:`Component.bind`: route ``InputSpec.default`` (an
    input not given takes the array of its ``default`` name), refuse unknown names and
    missing required ones (``TypeError``), check each array against its spec on
    ``comp.layout`` (``ValueError``; a routed input's message names the argument it was
    routed from). Returns ``{input name: array or None}`` (what ``on_bind`` receives);
    binds nothing."""
    inputs = dict(comp.inputs)
    arrays = dict(arrays)
    sources = {s.default for s in inputs.values() if s.default is not None}
    unknown = sorted(k for k in arrays if k not in inputs and k not in sources)
    if unknown:
        raise TypeError(f"{comp.name}: unknown inputs {unknown} (declared: "
                        f"{sorted(inputs)})")
    routed = {}
    for k, spec in inputs.items():
        if k not in arrays and spec.default is not None and spec.default in arrays:
            arrays[k] = arrays[spec.default]
            routed[k] = spec.default
    missing = sorted(k for k, s in inputs.items() if s.required and arrays.get(k) is None)
    if missing:
        raise TypeError(f"{comp.name}: missing inputs {missing}")
    declared = {k: arrays.get(k) for k in inputs}
    for k, arr in declared.items():
        if arr is not None:
            label = k if k not in routed else f"{k} (routed from {routed[k]})"
            inputs[k].check(label, arr, comp.layout, consumer=comp.name)
    return declared


def _deps(comp) -> tuple:
    """The names a component reads: ``after`` + its ``source`` (when not ``None``)."""
    after = tuple(getattr(comp, "after", ()) or ())
    src = getattr(comp, "source", None)
    return after + ((src,) if src is not None else ())


def _order(components) -> list:
    """The components in execution order (roadmap PR15): a stable topological sort of
    ``after`` + ``source`` over the given order (Kahn's algorithm, picking at each step
    the first remaining component whose dependencies are placed: the identity on an
    already-ordered list).

    Raises:
        TypeError: a ``name`` that is not a non-empty ``str``, or a malformed ``after``.
        ValueError: two components share a name, a component reads a name that is not
            one of the components, or the dependencies form a cycle.
    """
    comps = list(components)
    names = []
    for c in comps:
        n = getattr(c, "name", None)
        if not isinstance(n, str) or not n:
            raise TypeError(f"{type(c).__name__}.name must be a non-empty str, got {n!r}")
        after = getattr(c, "after", ()) or ()
        if isinstance(after, str) or not all(isinstance(a, str) for a in after):
            raise TypeError(f"{type(c).__name__}.after must be a tuple of component "
                            f"names, got {after!r}")
        names.append(n)
    seen, dup = set(), None
    for n in names:
        if n in seen:
            dup = n
            break
        seen.add(n)
    if dup is not None:
        raise ValueError(f"Course: two components are named {dup!r}; component names "
                         f"must be unique (components: {names})")
    for c in comps:
        for d in _deps(c):
            if d not in seen:
                raise ValueError(f"Course: component {c.name!r} reads {d!r} (after / "
                                 f"source), which is not one of its components {names}")
    placed, out, rest = set(), [], list(comps)
    while rest:
        for i, c in enumerate(rest):
            if all(d in placed for d in _deps(c)):
                out.append(c)
                placed.add(c.name)
                del rest[i]
                break
        else:
            raise ValueError(f"Course: components {[c.name for c in rest]} depend on each "
                             f"other (after / source form a cycle)")
    return out


def _per_step(comp) -> bool:
    """Whether ``comp`` has per-step output (a ``StepResult`` entry): it overrides
    :meth:`Component.step` (a duck-typed component defines its own ``step`` and
    counts)."""
    return getattr(type(comp), "step", None) is not Component.step


def _check_target(target, requires: tuple, optional: tuple, *, consumer: str,
                  topology: str, layout: Layout) -> "Layout | None":
    """Check ``target`` against the declared protocols: every member of ``requires``
    (strict), the members of ``optional`` the target has (checked when present; with
    ``optional`` alone, against the target's layout, else the layout its present
    members give: ``num_envs`` from ``count`` or a per-env member, the stride from a
    per-point member; the context's ``layout`` fills what none of them gives).
    Returns its layout (``None`` when nothing is declared)."""
    if not requires and not optional:
        return None
    req = tuple(dict.fromkeys(m for p in requires for m in _members_of(p)))
    opt = tuple(dict.fromkeys(m for p in optional for m in _members_of(p)
                              if m not in req))
    if not requires:
        if not isinstance(getattr(target, "layout", None), Layout):
            present = {m: v for m in opt
                       if (v := getattr(target, m, None)) is not None}
            pers = {_MEMBER_SCHEMA[m][0] for m in present if m in _MEMBER_SCHEMA}
            if "point" not in pers or "env" not in pers:
                # The present members do not give both num_envs and the stride: check
                # them against the context's layout (its stride replaced by the one a
                # per-point member gives, when one is present and divides by num_envs),
                # through a proxy named after the target, for the messages.
                lay0 = layout
                if "point" in pers:
                    first = next(v for m, v in present.items()
                                 if _MEMBER_SCHEMA.get(m, ("",))[0] == "point")
                    E = int(layout.num_envs)
                    n = int(first.shape[0]) if isinstance(first, wp.array) and \
                        first.ndim == 1 else 0
                    if n >= E and n % E == 0:
                        lay0 = Layout(E, n // E, layout.device, layout.closed)
                proxy = type(type(target).__name__, (types.SimpleNamespace,), {})
                target = proxy(layout=lay0, **present)
            # else: count / a per-env member gives num_envs and a per-point member the
            # stride (a CheckpointSet: M slots per env), inferred from the target.
        lay, _ = _check_and_bind(target, optional, consumer=consumer, topology=topology,
                                 members=(), optional=opt)
        return lay
    lay, _ = _check_and_bind(target, requires + optional, consumer=consumer,
                             topology=topology, members=req, optional=opt)
    return lay


class ComponentContext:
    """What :meth:`Component.attach` is given: the producer's layout, its static
    capacities, a cold launch context and the components attached before.

    Args:
        layout: The producer's :class:`~track_gen.results.Layout`.
        capacity: Static capacities of the producer's plan (``name -> number``; see
            :meth:`for_producer`). Hand-built sources pass their own.
        ctx: The launch context of the cold work of ``attach`` (default:
            ``LaunchContext.for_call(layout.device)``: an enclosing ``ctx.scope()``,
            else Warp's current stream). Attach on the stream where allocations happen
            (Warp's current stream: ``attach`` allocates and zero-fills there, then
            launches its first ``on_generate`` on ``ctx``; a context on another stream
            races its own allocations), after the producer's work completed on it (or
            make that stream wait for the producer's).
        bounds: Declared bounds of the producer's plan (``name -> number``, roadmap
            PR15; see :meth:`for_producer`): ``size_hi``, the largest course size the
            plan can draw. Read by ``BandCollision`` / ``HeightFieldBake`` with
            ``size_hi="auto"``. Hand-built sources pass their own (default: none).
    """

    def __init__(self, layout: Layout, capacity: "Mapping[str, float] | None" = None, *,
                 ctx: "LaunchContext | None" = None,
                 bounds: "Mapping[str, float] | None" = None) -> None:
        if not isinstance(layout, Layout):
            raise TypeError(f"ComponentContext: layout must be a track_gen.Layout, got "
                            f"{type(layout).__name__}")
        cap = dict(capacity or {})
        for k, v in cap.items():
            if not isinstance(k, str) or isinstance(v, bool) or \
                    not isinstance(v, numbers.Real) or not math.isfinite(float(v)) or \
                    float(v) <= 0.0:
                raise ValueError(f"ComponentContext: capacity {k!r} must be a finite "
                                 f"number > 0, got {v!r}")
        bnd = dict(bounds or {})
        for k, v in bnd.items():
            if not isinstance(k, str) or isinstance(v, bool) or \
                    not isinstance(v, numbers.Real) or not math.isfinite(float(v)) or \
                    float(v) <= 0.0:
                raise ValueError(f"ComponentContext: bound {k!r} must be a finite "
                                 f"number > 0, got {v!r}")
        if ctx is None:
            ctx = LaunchContext.for_call(layout.device)
        elif not isinstance(ctx, LaunchContext):
            raise TypeError(f"ComponentContext: ctx must be a LaunchContext, got "
                            f"{type(ctx).__name__}")
        elif ctx.device.alias != wp.get_device(layout.device).alias:
            raise ValueError(f"ComponentContext: ctx is on {ctx.device.alias!r}, the "
                             f"layout on {layout.device!r}")
        self._layout = layout
        self._capacity = types.MappingProxyType(cap)
        self._bounds = types.MappingProxyType(bnd)
        self._ctx = ctx
        self._components: dict = {}

    def __repr__(self) -> str:
        return (f"ComponentContext({self._layout!r}, capacity={dict(self._capacity)!r}, "
                f"components={list(self._components)!r})")

    @property
    def layout(self) -> Layout:
        """The producer's layout."""
        return self._layout

    @property
    def capacity(self) -> Mapping:
        """The producer's static capacities (read-only mapping). ``max_loop_length``:
        an upper bound, with measured headroom, on the 3D length of the centerline and
        of either boundary loop of every valid env (see :meth:`for_producer`); a
        consumer sizes from it, clamps to its own capacity and reports truncation."""
        return self._capacity

    @property
    def bounds(self) -> Mapping:
        """The producer's declared bounds (read-only mapping, roadmap PR15; see
        :meth:`for_producer`). ``size_hi``: the largest course size the plan can draw (a
        size randomizer's construction ``hi``, else the generator's construction
        ``size``); absent when the plan declares no size."""
        return self._bounds

    @property
    def ctx(self) -> LaunchContext:
        """The launch context of the cold work of ``attach``."""
        return self._ctx

    @property
    def names(self) -> "tuple[str, ...]":
        """The names of the attached components, in attach order."""
        return tuple(self._components)

    def component(self, name: str) -> Component:
        """The attached component named ``name``.

        Raises:
            KeyError: none is attached under that name (the message lists them).
        """
        try:
            return self._components[name]
        except KeyError:
            raise KeyError(f"no component {name!r} is attached (attached: "
                           f"{list(self._components)})") from None

    def add(self, component, source) -> Component:
        """Check ``component``'s declaration and read target, attach it and register it.

        The read target is ``source`` (the producer's result) when ``component.source``
        is ``None``, else the ``result`` of the attached component it names. The target
        is checked against ``component.requires`` (every member) and
        ``component.optional`` (the members it has; with ``optional`` alone and a target
        without a layout, against the layout its present members give, this context's
        layout filling what they do not), with ``component.topology``;
        ``component.layout`` is set (to this context's layout when nothing is declared),
        then ``component.attach(target, self)`` runs. A producer-result target
        (``component.source is None``) must hold this context's batch: the same
        ``num_envs`` and device.

        Returns:
            ``component``.

        Raises:
            ValueError: a component named in ``after`` / ``source`` is not attached, the
                name is taken, an input name is declared by an attached component with a
                different ``(dtype, slots)``, the target does not match (the
                ``check_source`` messages), or a producer-result target holds another
                batch (``num_envs`` or device) than this context's layout.
            TypeError: a malformed declaration, or a required member missing on the
                target.
        """
        requires, optional = _check_declaration(component)
        deps = tuple(component.after) + \
            (() if component.source is None else (component.source,))
        missing = [d for d in deps if d not in self._components]
        if missing:
            raise ValueError(f"{component.name}: {missing} must be attached before it "
                             f"(attached: {list(self._components)})")
        if component.name in self._components:
            raise ValueError(f"a component named {component.name!r} is already attached")
        for k, spec in component.inputs.items():
            for other in self._components.values():
                o = other.inputs.get(k)
                if o is not None and (o.dtype, o.slots) != (spec.dtype, spec.slots):
                    raise ValueError(
                        f"input {k!r}: {component.name} declares "
                        f"({getattr(spec.dtype, '__name__', spec.dtype)}, {spec.slots}) "
                        f"but {other.name} declares "
                        f"({getattr(o.dtype, '__name__', o.dtype)}, {o.slots})")
        target = source if component.source is None else \
            self.component(component.source).result
        lay = _check_target(target, requires, optional, consumer=component.name,
                            topology=component.topology, layout=self._layout)
        if lay is not None and component.source is None and (
                int(lay.num_envs) != int(self._layout.num_envs) or
                wp.get_device(lay.device).alias != wp.get_device(self._layout.device).alias):
            raise ValueError(f"{component.name}: its read target holds {lay!r}, this "
                             f"context's batch is {self._layout!r} (attach it through a "
                             f"ComponentContext of the producer it reads)")
        component.layout = lay if lay is not None else self._layout
        component.attach(target, self)
        self._components[component.name] = component
        return component

    @classmethod
    def for_producer(cls, producer, *, ctx: "LaunchContext | None" = None
                     ) -> "ComponentContext":
        """The context of a producer's components: its layout and the static capacities
        of its plan (no device readback).

        ``producer`` is a ``TrackGenerator`` / ``GateGenerator`` (its ``program``), a
        built ``Program``, a ``Course`` (its ``generator``), or any object with the
        ``track_gen.pipeline.Producer`` surface that declares ``capacity`` and / or
        ``bounds`` itself (read before the plan, roadmap PR15); its ``spec`` /
        ``params`` / ``layout`` are read. A result (``gen.generate()``'s ``Track``) has
        no plan: build a ``ComponentContext(track.layout, capacity)`` for a hand-built
        source.

        Bound ``size_hi`` (roadmap PR15, :attr:`bounds`): the construction ``hi`` of a
        ``randomize`` stage drawing ``size`` (``generator.size`` / ``normalize.size``;
        the largest of them), else the construction ``size`` of the generator (or of the
        ``normalize`` stage of a ring gate pipeline); omitted when the plan has no size
        (``external``, a legacy spec). It is the declared bound, never a current value:
        a later ``set("hi", ...)`` or ``set("size", ...)`` curriculum is not seen.

        Capacity key ``max_loop_length`` (0.x): ``1.5 * N_max * spacing + dZ`` from the
        ``("resample", "constant_spacing")`` slot's ``spacing`` and the batch's
        ``n_max``. A non-truncated env's plan-view centerline is shorter than ``N_max *
        spacing`` before relax (else the resample truncates it and the env fails
        validity); the 1.5 covers relax growth and a constant-width boundary's extra
        ``2 pi * half_width`` (measured, not proven: at most 0.904 of ``N_max *
        spacing``). ``dZ`` bounds the elevation's total variation from the knobs'
        DECLARED bounds (``bounds()``, never their current values): 0 for
        ``elevation.flat``; ``K * (z_max_hi - z_min_lo)`` for ``uniform`` /
        ``random_walk`` (``K = z_control_points``); ``min(2 * H * (z_max_hi -
        z_min_lo), 4 * A_hi * H / H_H)`` for ``noise`` (``H = z_noise_harmonics``,
        ``H_H = sum_{k<=H} 1/k``, ``A_hi`` the amplitude's upper bound); ``g * 1.5 *
        N_max * spacing`` with ``validity.z_valid_grade = g > 0``; the smallest finite
        term (only the grade of the built-in ``validity.default`` counts: a plugin check
        promises nothing about the final altitudes). The key is OMITTED (pass explicit
        capacities) when no term is finite (an elevation whose ``z_min`` / ``z_max``
        declare no range: every elevated ``TrackGenConfig`` today; a randomizer's range
        does not count: its ``lo`` / ``hi`` are dynamic), when the altitudes come from
        the generator (it ``provides`` ``z``: ``lift.elevated`` with no elevation stage
        after the resample) and no grade is declared, when no ``width.constant`` stage
        follows the resample (another width stage, or a generator that ``provides``
        ``half_width``: an unbounded per-point width), with a plugin stage of kind
        ``relax`` / ``width`` / ``offset`` / ``elevation`` / ``lift`` / ``resample`` after
        the constant-spacing resample, and without that resample (gate pipelines).

        Raises:
            TypeError: ``producer`` has no plan and declares neither ``capacity`` nor
                ``bounds`` (a ``Track`` or another result, a minimal producer), or it
                has no ``Layout``.
        """
        from .._src.course import Course  # noqa: PLC0415 - import cycle
        from ..pipeline.program import Program  # noqa: PLC0415 - keeps import cheap

        # Strict unwrapping: only a real Course gives its generator, and only a real
        # Program is a plan (an unrelated `generator` / `program` / `spec` attribute of a
        # third-party producer is its own business: its capacity / bounds are read).
        obj = producer.generator if isinstance(producer, Course) else producer
        inner = getattr(obj, "program", None)
        prog = inner if isinstance(inner, Program) else obj
        what = type(producer).__name__
        own_cap = getattr(prog, "capacity", None)
        own_bounds = getattr(prog, "bounds", None)
        for label, own in (("capacity", own_cap), ("bounds", own_bounds)):
            if own is not None and not isinstance(own, Mapping):
                raise TypeError(f"ComponentContext.for_producer: {what}.{label} must be "
                                f"a mapping of name -> number, got {type(own).__name__}")
        # A plan: a Program (or an object with its spec / params surface). A producer's
        # own capacity / bounds win over it anyway.
        planned = isinstance(prog, Program) or getattr(prog, "spec", None) is not None
        if not planned and own_cap is None and own_bounds is None:
            noun = "track" if what == "Track" else "result"
            raise TypeError(f"ComponentContext.for_producer takes a TrackGenerator / "
                            f"GateGenerator / Program, got {what}: pass the generator, "
                            f"or use ComponentContext({noun}.layout, capacity) for a "
                            f"hand-built source (a track_gen.pipeline.Producer without "
                            f"a plan qualifies when it declares capacity and / or "
                            f"bounds)")
        layout = getattr(prog, "layout", None)
        if not isinstance(layout, Layout):
            raise TypeError(f"ComponentContext.for_producer: {what} has no Layout (pass "
                            f"a generator or a built Program)")
        capacity = dict(own_cap) if own_cap is not None else \
            (_plan_capacity(prog) if planned else {})
        bounds = dict(own_bounds) if own_bounds is not None else \
            (_plan_bounds(prog) if planned else {})
        return cls(layout, capacity, ctx=ctx, bounds=bounds)


# ---------------------------------------------------------------------------------------
# The plan's static capacities
# ---------------------------------------------------------------------------------------

#: The built-in stages that may follow the constant-spacing resample, by kind (a stage of
#: one of these kinds with another name, or a plugin's, drops the capacity key).
_BOUNDED_STAGES = {
    "relax": ("none", "xpbd"),
    "width": ("constant",),
    "offset": ("default",),
    "elevation": ("flat", "uniform", "random_walk", "noise"),
    "lift": ("flat", "elevated"),
    "resample": ("constant_spacing", "uniform"),
}


#: The built-in validity stage whose ``z_valid_grade`` bounds the final altitudes.
_GRADED_STAGES = {"validity": ("default",)}


def _is_builtin(ref, table: Mapping = _BOUNDED_STAGES) -> bool:
    names = table.get(ref.kind)
    if names is None or ref.name not in names:
        return False
    mod = type(ref.params).__module__ if ref.params is not None else "track_gen."
    return mod == "track_gen" or mod.startswith("track_gen.")


def _field_bounds(params, ref, field: str) -> tuple:
    """``(lo, hi)`` of a field of stage ``ref``: a dynamic field's declared bounds
    (``StageParams.bounds``: narrowed per instance, never by a randomizer's dynamic
    range), a static field's value on both sides."""
    table = None
    try:
        table = params[ref.slot_id] if params is not None else None
    except (KeyError, TypeError):
        table = None
    if table is not None:
        try:
            return tuple(table.bounds(field))
        except KeyError:
            pass
    value = getattr(ref.params, field, None)
    return (None, None) if value is None else (float(value), float(value))


def _finite(*values) -> bool:
    return all(v is not None and math.isfinite(float(v)) for v in values)


def _elevation_variation(ref, params, base: float) -> "float | None":
    """``dZ`` of the elevation stage ``ref`` (``None``: unbounded)."""
    if ref.name == "flat":
        return 0.0
    p = ref.params
    cands = []
    zlo = _field_bounds(params, ref, "z_min")[0]
    zhi = _field_bounds(params, ref, "z_max")[1]
    if ref.name in ("uniform", "random_walk"):
        if _finite(zlo, zhi):
            cands.append(int(p.z_control_points) * (float(zhi) - float(zlo)))
    elif ref.name == "noise":
        H = int(p.z_noise_harmonics)
        if _finite(zlo, zhi):
            cands.append(2 * H * (float(zhi) - float(zlo)))
        a_hi = _field_bounds(params, ref, "z_noise_amplitude")[1]
        if _finite(a_hi):
            cands.append(4.0 * float(a_hi) * H / sum(1.0 / k for k in range(1, H + 1)))
    return min(cands) if cands else None


#: The stage slots whose ``size`` field is the course size (roadmap PR13d): the track
#: and bezier / hull gate generators' ``generator.size``, the ring gate pipelines'
#: ``normalize.size``.
_SIZE_SLOTS = ("generator", "normalize")


def _plan_bounds(prog) -> dict:
    """``{"size_hi": ...}`` of a built Program's plan, or ``{}`` (see
    :meth:`ComponentContext.for_producer`)."""
    spec = getattr(prog, "spec", None)
    stages = list(getattr(spec, "stages", None) or ())
    drawn = []
    for ref in stages:
        p = ref.params
        if ref.kind == "randomize" and getattr(p, "field", None) == "size" and \
                getattr(p, "stage", None) in _SIZE_SLOTS:
            hi = getattr(p, "hi", None)
            if isinstance(hi, numbers.Real) and math.isfinite(float(hi)) and \
                    float(hi) > 0.0:
                drawn.append(float(hi))
    if drawn:
        return {"size_hi": max(drawn)}
    params = getattr(prog, "params", None)
    for slot in _SIZE_SLOTS:
        size = None
        try:
            table = params[slot] if params is not None and slot in params else None
            size = getattr(getattr(table, "params", None), "size", None)
        except (KeyError, TypeError):
            size = None
        if size is None:
            ref = next((r for r in stages if r.slot_id == slot), None)
            size = getattr(getattr(ref, "params", None), "size", None)
        if isinstance(size, numbers.Real) and not isinstance(size, bool) and \
                math.isfinite(float(size)) and float(size) > 0.0:
            return {"size_hi": float(size)}
    return {}


def _plan_capacity(prog) -> dict:
    """``{"max_loop_length": ...}`` of a built Program's plan, or ``{}`` (see
    :meth:`ComponentContext.for_producer`)."""
    spec = getattr(prog, "spec", None)
    stages = list(getattr(spec, "stages", None) or ())
    batch = getattr(spec, "batch", None)
    idx = next((i for i, r in enumerate(stages)
                if (r.kind, r.name) == ("resample", "constant_spacing")), None)
    if idx is None or batch is None or getattr(batch, "n_max", None) is None:
        return {}
    spacing = getattr(stages[idx].params, "spacing", None)
    if spacing is None or not float(spacing) > 0.0:
        return {}
    base = LOOP_HEADROOM * int(batch.n_max) * float(spacing)
    params = getattr(prog, "params", None)
    dz = 0.0
    grade = 0.0
    elevated = width = lift = None
    for ref in stages[idx + 1:]:
        if ref.kind in _BOUNDED_STAGES:
            if not _is_builtin(ref):
                return {}
            if ref.kind == "elevation":
                elevated = ref
            elif ref.kind == "width":
                width = ref
            elif ref.kind == "lift":
                lift = ref
        elif _is_builtin(ref, _GRADED_STAGES):
            g = getattr(ref.params, "z_valid_grade", None)
            if g is not None and float(g) > 0.0:
                grade = max(grade, float(g))
    if width is None or lift is None:
        # No width stage after the resample: the half-widths come from elsewhere (a
        # generator that provides half_width): unbounded. No lift: no track altitudes.
        return {}
    if lift.name == "elevated" and elevated is None:
        # The altitudes come from the generator (it provides z: raw.z -> cl.z): only a
        # declared grade bounds them.
        if grade <= 0.0:
            return {}
        dz = grade * base
    elif elevated is not None and elevated.name != "flat":
        cands = []
        term = _elevation_variation(elevated, params, base)
        if term is not None:
            cands.append(term)
        if grade > 0.0:
            cands.append(grade * base)
        if not cands:
            return {}
        dz = min(cands)
    return {"max_loop_length": base + dz}
