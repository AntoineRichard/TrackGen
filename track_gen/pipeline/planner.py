"""The planner: :func:`build` turns a :class:`PipelineSpec` into a :class:`Program`.

The spec rules are checked host-side before anything is allocated (:class:`PlanError`,
a ``ValueError``; the stages' params keep their own ``ValueError`` / ``TypeError``, and
an unknown generator raises the registry's own ``ValueError``); then the planner
allocates every channel version once, namespaces each stage's scratch, runs every
``Stage.build`` once and binds the result. The build-time rules (each stage's write
ports, its fail bits, a plugin channel's dtype) are checked right after the stage's
``build`` (a ``PlanError`` raised after allocation).

Stages of the kinds open to plugins (``relax``, ``width``, ``elevation``, ``validity``,
``order``, ``stations``) are resolved through the plugin registry; the others
(``resample``, ``frames``, ``offset``, ``lift``, and the gate pipeline's ``normalize``,
``carry``, ``gate_relax``, ``gate_arclen``, ``tangents``, ``gate_frames``) from
TrackGen's built-in catalogue.

Channel carry (roadmap PR13a): the resamples and the built-in stations carry the
per-point channels of their source loop to their output points (``raw.<attr>`` ->
``cl.<attr>`` -> uniform ``cl.<attr>`` -> ``anchors.<attr>`` -> ``gates.<attr>``). Which
families each carrier carries is static, computed here from the declared writes and
reads before any allocation, and appears as the carrier's implicit ports (``~<channel>``)
in :attr:`Program.channels <track_gen.pipeline.Program.channels>`, the read-before-write
and second-writer rules and the exports.

Rule 6 (roadmap PR12): the stages are grouped into segments of equal host-ness
(``CaptureClass.HOST`` vs ``STATIC`` / ``BOUNDED``, :attr:`Program.segments
<track_gen.pipeline.Program.segments>`); ``PipelineSpec.require_caller_capturable``
refuses any HOST stage, host-side. Every stage's ``enqueue`` runs inside its slot context
(its id and declared capture class: ``ctx.host()`` and the ``ctx.bounded_loop``
declaration check read it).

One planner, two result kinds (roadmap PR11): ``BatchSpec.max_gates=None`` plans a
track Program (its result a ``Track``), an int a gate Program (a ``GateSequence``). In a
gate Program each stage runs on one point domain: the centerline generator and any
stage with a ``cl.*`` port on ``N_max`` points, every other stage on ``max_gates`` gate
slots (``Layout.stride``); the anchors are ``[E * A]`` (``A``: the anchors producer's
``capacity()["gates"]``).
"""
from __future__ import annotations

import dataclasses
import numbers
import types
from collections.abc import Mapping as _MappingABC

import warp as wp

from .spec import BatchSpec, PipelineSpec, StageRef

__all__ = ["PlanError", "build"]

_GENERATOR = "generator"
_GATE_GENERATOR = "gate_generator"
_GENERATOR_KINDS = (_GENERATOR, _GATE_GENERATOR)
_PROLOGUE = "<prologue>"
_TRACK_DTYPES = {"vec3f": wp.vec3f, "float32": wp.float32, "int32": wp.int32,
                 "quatf": wp.quatf}
_ON_INVALID = ("keep", "poison")
# Params fields baked as host scalars by several stages: one value per pipeline.
_AGREE = {
    "half_width": "the stages with a half_width field share one value: the nominal width "
                  "(the per-point width is cl.half_width)",
    "gate_width": "one value per pipeline: gate_frames bakes the bar width, "
                  "validity.gate_default the crossing toggle",
    "gate_radius": "one value per pipeline: gate_relax.spheres separates the gates to "
                   "the distance validity.gate_default checks",
}


class PlanError(ValueError):
    """A pipeline spec the planner refuses; raised before anything is allocated, except
    for the build-time rules checked right after a stage's ``build`` (its declared write
    ports, its fail bits, a plugin channel's dtype)."""


def build(spec: PipelineSpec, *, registry=None, seeds=None, out=None, stream=None,
          prepare: bool = True):
    """Plan, allocate and bind ``spec``; return its :class:`~track_gen.pipeline.Program`.

    The generator (the first stage) is resolved through ``registry`` exactly as the
    facades resolve ``config.generator``: a centerline generator (``kind="generator"``)
    with the config :func:`track_gen.pipeline.legacy.generator_config` derives from the
    spec, a gate generator (``kind="gate_generator"``) with
    :func:`track_gen.pipeline.legacy.gate_generator_config`. The open-kind stages
    (``relax`` / ``width`` / ``elevation`` / ``validity`` / ``order`` / ``stations``) are
    resolved through ``registry`` too (the class is captured at build: later registry
    edits never change a built Program); the others come from TrackGen's built-in
    catalogue. ``spec.batch.max_gates`` selects the result: a ``Track`` (``None``) or a
    ``GateSequence``.

    Args:
        spec: The :class:`PipelineSpec`.
        registry: The plugin registry (default ``track_gen.plugin.registry``).
        seeds: A ``SeedState``, or anything ``SeedState`` accepts (``None``, an int, an
            ``[E]`` int32 ``wp.array``, a ``PerEnvSeededRNG``).
        out: Optional pre-allocated ``Track`` (``GateSequence``) to write into (every
            field checked; ``generation`` and ``fail_mask`` may be ``None``).
        stream: Optional default ``wp.Stream`` of the Program.
        prepare: Run the warm-up generation (:meth:`Program.prepare`) before returning
            (otherwise the first :meth:`Program.run` does).

    Raises:
        PlanError: an invalid spec (see :doc:`/how-it-works/pipeline-preview`), incl.
            an unknown relax / width / elevation / validity / order / stations name
            (the registry's message) and a HOST stage under
            ``PipelineSpec.require_caller_capturable``.
        ValueError: invalid params (the stage's own message), an unknown generator (the
            registry's message), an ``out=`` or ``seeds`` mismatch.
        TypeError: params or ``seeds`` of the wrong type.
        track_gen.plugin.PluginLoadError: a stage named explicitly is provided by a
            quarantined plugin.
        track_gen.exec.CaptureError: called inside a capture (before allocating).
    """
    from ..plugin import registry as _default_registry  # noqa: PLC0415

    _check_spec_shape(spec)
    _check_seeds(seeds)
    reg = _default_registry if registry is None else registry
    binding = _bind(spec, reg)
    return _build(spec, binding=binding, seeds=seeds, out=out, stream=stream,
                  prepare=prepare, registry=reg)


def _bind(spec: PipelineSpec, registry):
    """The generator binding of ``spec`` (its first stage): a gate generator through
    ``legacy.gate_generator_config(spec)``, a centerline generator (a track pipeline, or
    the stations source of a gate pipeline) through ``legacy.generator_config(spec)``."""
    from .._src.generator_binding import bind_generator  # noqa: PLC0415
    from . import legacy  # noqa: PLC0415

    if spec.stages[0].kind == _GATE_GENERATOR:
        return bind_generator(_GATE_GENERATOR, legacy.gate_generator_config(spec),
                              registry=registry)
    return bind_generator(_GENERATOR, legacy.generator_config(spec), registry=registry)


# ---------------------------------------------------------------------------------------
# Host-side checks
# ---------------------------------------------------------------------------------------

def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _check_spec_shape(spec) -> None:
    """The batch, the ids and the generator slot (no registry, no allocation)."""
    if not isinstance(spec, PipelineSpec):
        raise TypeError(f"build() expects a PipelineSpec, got {type(spec).__name__}")
    batch: BatchSpec = spec.batch
    if not _is_int(batch.num_envs) or batch.num_envs < 1:
        raise PlanError(f"BatchSpec.num_envs must be an int >= 1, got {batch.num_envs!r}")
    if not _is_int(batch.n_max) or batch.n_max < 3:
        raise PlanError(f"BatchSpec.n_max must be an int >= 3, got {batch.n_max!r}")
    gate = batch.max_gates is not None
    if gate and (not _is_int(batch.max_gates) or batch.max_gates < 2):
        raise PlanError(f"BatchSpec.max_gates must be None (a track pipeline) or an int "
                        f">= 2, got {batch.max_gates!r}")
    if batch.closed is not True:
        raise PlanError("BatchSpec.closed=False: open tracks arrive after 1.0; closed "
                        "loops only")
    if batch.on_invalid not in _ON_INVALID:
        raise PlanError(f"BatchSpec.on_invalid must be 'keep' or 'poison', got "
                        f"{batch.on_invalid!r}")
    if not spec.stages:
        raise PlanError("the pipeline has no stages; the first must be the generator")
    ids = spec.ids
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise PlanError(f"duplicate stage ids {dup}; give one of them an explicit "
                        f"StageRef(id=...)")
    first = spec.stages[0]
    if gate:
        if first.kind not in _GENERATOR_KINDS:
            raise PlanError(f"the first stage of a gate pipeline must be the generator "
                            f"(kind='gate_generator', or kind='generator' with a stations "
                            f"stage), got {first.kind!r}")
    else:
        for ref in spec.stages:
            if ref.kind == _GATE_GENERATOR:
                raise PlanError(f"{_label(ref)}: a gate generator needs a gate pipeline "
                                f"(BatchSpec.max_gates)")
        if first.kind != _GENERATOR:
            raise PlanError(f"the first stage must be the generator (kind='generator'), "
                            f"got {first.kind!r}")
    gens = [ref.slot_id for ref in spec.stages if ref.kind in _GENERATOR_KINDS]
    if len(gens) != 1:
        raise PlanError(f"exactly one generator stage, got {len(gens)}: {gens}")
    if first.slot_id != _GENERATOR:
        raise PlanError(f"the generator stage's id must be 'generator', got "
                        f"{first.slot_id!r}")
    # capacity: the centerline generator's raw default only (Stage.capacity() overrides
    # it).
    for ref in spec.stages:
        allowed = ("raw",) if ref.kind == _GENERATOR else ()
        bad = sorted(k for k in ref.capacity if k not in allowed)
        if bad:
            raise PlanError(
                f"{_label(ref)}: unknown capacity keys {bad}; only the generator takes "
                f"capacity, and only 'raw' (the default raw point count)")


def _is_seed_form(seeds) -> bool:
    """Whether ``seeds`` is a form ``SeedState`` builds from (``None``, an int, a
    ``wp.array``, a ``PerEnvSeededRNG`` or anything with ``seeds_warp``)."""
    from .._src.rng_utils import PerEnvSeededRNG  # noqa: PLC0415

    return seeds is None or isinstance(seeds, (numbers.Integral, wp.array, PerEnvSeededRNG)) \
        or hasattr(seeds, "seeds_warp")


def _check_seeds(seeds) -> None:
    """``build(seeds=...)``: a ``SeedState`` or a form it builds from (host-only)."""
    from ..seeds import SeedState  # noqa: PLC0415

    if not (_is_seed_form(seeds) or isinstance(seeds, SeedState)):
        raise TypeError(f"seeds must be None, an int, an [E] int32 wp.array, a "
                        f"PerEnvSeededRNG or a SeedState; got {type(seeds).__name__}")


def _label(ref: StageRef) -> str:
    return f"stage {ref.slot_id!r} ({ref.kind}.{ref.name})"


@dataclasses.dataclass
class _Slot:
    """One resolved stage of the plan."""

    index: int
    id: str
    ref: StageRef
    cls: object                 # the Stage class (or the generator slot's class)
    instance: object
    params: object              # the resolved Params (generator: binding.params)
    reads: dict                 # port -> channel (after the ports)
    writes: dict
    builtin: bool = True        # shipped with track_gen (the registry's own record)
    ports: dict = dataclasses.field(default_factory=dict)   # effective rewiring
    layout: object = None       # the stage's Layout (its point domain)
    generator: "str | None" = None  # "centerline" | "anchors" | "legacy" | None
    capacity: "dict | None" = None  # an anchors producer's {"gates": A, "min_gates": k}
    carries: tuple = ()             # the (src, dst) channels a carrier carries
    centerline: bool = False        # a gate Program's stage before the anchors producer


def _decl(s) -> "tuple[dict, dict]":
    """The declared ``(reads, writes)`` of a slot: its instance's (the generator slot's
    writes include the generator's raw attributes), else its class's."""
    src = s.instance if s.instance is not None else s.cls
    return dict(getattr(src, "reads", {})), dict(getattr(src, "writes", {}))


def _effective_ports(cls, ref: StageRef, gate: bool) -> dict:
    """``StageRef.ports``, plus, for an ``elevation`` stage of a gate Program that runs
    on the gates (after the anchors producer), the default gate rewiring of its ``cl.*``
    ports (explicit ports override it): the catalogue pairs of
    ``ELEVATION_GATE_CHANNELS``, and plugin channels by prefix (``cl.<pkg>.<name>`` ->
    ``gates.<pkg>.<name>``)."""
    from ..stages.channels import ELEVATION_GATE_CHANNELS, is_plugin_channel  # noqa

    ports = {}
    if gate and ref.kind == "elevation":
        for p, ch in {**dict(cls.reads), **dict(cls.writes)}.items():
            if ch in ELEVATION_GATE_CHANNELS:
                ports[p] = ELEVATION_GATE_CHANNELS[ch]
            elif is_plugin_channel(ch) and ch.startswith("cl."):
                ports[p] = "gates." + ch[3:]
    ports.update(ref.ports)
    return ports


def _domain_layout(slot_cls, ref: StageRef, ports: dict, *, gate: bool, batch,
                   generator: "str | None", centerline: bool = False):
    """The stage's :class:`Layout` from its (rewired) ports, before any validation:
    ``N_max`` on a track and for the centerline stages of a gate Program, ``max_gates``
    for every gate stage. ``centerline``: a gate Program's stage before the anchors
    producer (a ``relaxed_stations`` sub-pipeline stage, e.g. an elevation on the
    relaxed centerline: its ``cl.*`` ports stay)."""
    from ..plugin.stage import Layout  # noqa: PLC0415
    from ..stages.channels import (CHANNELS, channel_domain, is_gate_channel,  # noqa
                                   is_plugin_channel)

    E, dev = int(batch.num_envs), str(wp.get_device(batch.device))
    n_max = int(batch.n_max)
    if generator in ("centerline",):
        return Layout(E, n_max, dev, True)
    if generator in ("anchors", "legacy"):
        return Layout(E, int(batch.max_gates), dev, True)
    declared = {**dict(slot_cls.reads), **dict(slot_cls.writes)}
    chans = {p: ports.get(p, ch) for p, ch in declared.items()}
    label = _label(ref)
    gate_chans = [ch for ch in chans.values() if is_gate_channel(ch) or (
        is_plugin_channel(ch) and channel_domain(ch) in ("anchors", "gates"))]
    if not gate:
        if gate_chans:
            raise PlanError(f"{label}: channel {gate_chans[0]!r} belongs to the gate "
                            f"pipeline (BatchSpec.max_gates); a track pipeline has none")
        return Layout(E, n_max, dev, True)
    cl_plugin = [ch for ch in chans.values()
                 if is_plugin_channel(ch) and channel_domain(ch) == "cl"]
    cl_cat = [ch for ch in chans.values() if ch in CHANNELS and ch.startswith("cl.")]
    on_gates = [ch for ch in gate_chans if ch.startswith("gates.")]
    gates_plugin = [ch for ch in on_gates if is_plugin_channel(ch)]
    if cl_plugin and on_gates:
        raise PlanError(f"{label}: plugin channel {cl_plugin[0]!r} is a centerline "
                        f"channel; a gate-domain stage reads and writes "
                        f"gates.<pkg>.<name>")
    if gates_plugin and cl_cat:
        raise PlanError(f"{label}: plugin channel {gates_plugin[0]!r} is a gate channel; "
                        f"a centerline stage reads and writes cl.<pkg>.<name>")
    if ref.kind == "elevation" and not centerline:
        for p, ch in chans.items():
            if ch in CHANNELS and ch.startswith("cl."):
                raise PlanError(f"{label}: port {p!r} reads {ch!r}, which has no gate "
                                f"counterpart (the default gate rewiring covers cl.xy, "
                                f"cl.count, cl.arclen2, cl.length2 and cl.z); wire it "
                                f"with StageRef(ports=...)")
    if cl_cat and on_gates:
        raise PlanError(f"{label} has centerline (cl.*) and gate (gates.*) ports after "
                        f"rewiring; a stage runs on one point domain")
    stride = n_max if (cl_cat or cl_plugin) else int(batch.max_gates)
    return Layout(E, stride, dev, True)


def _check_capacity(cls, cap) -> dict:
    """An anchors producer's ``capacity()``: ``{"gates": A >= 1, "min_gates": 0..A}``."""
    name = cls.__name__
    if not isinstance(cap, dict):
        try:
            cap = dict(cap or {})
        except TypeError:
            raise ValueError(f"{name}.capacity() must return a mapping, got {cap!r}") \
                from None
    unknown = sorted(str(k) for k in cap if k not in ("gates", "min_gates"))
    if unknown:
        raise ValueError(f"{name}.capacity() returned unknown keys {unknown}; an anchors "
                         f"stage accepts ['gates', 'min_gates']")
    A = cap.get("gates")
    if not _is_int(A) or A < 1:
        raise ValueError(f"{name}.capacity()['gates'] must be an int >= 1 (the anchor "
                         f"rows per env), got {A!r}")
    k = cap.get("min_gates", 0)
    if not _is_int(k) or not 0 <= k <= A:
        raise ValueError(f"{name}.capacity()['min_gates'] must be an int in [0, "
                         f"{int(A)}], got {k!r}")
    return {"gates": int(A), "min_gates": int(k)}


def _resolve(spec: PipelineSpec, binding, layout, registry=None) -> "list[_Slot]":
    from .. import stages as _stages  # noqa: PLC0415
    from ..plugin import registry as _default_registry  # noqa: PLC0415
    from ..plugin.params import _check_values, _coerce, _static_view  # noqa: PLC0415
    from ..plugin.stage import _is_stage  # noqa: PLC0415
    from ..stages.gates import _LegacyGateSlot  # noqa: PLC0415
    from ..stages.track import (ElevationFlat, OffsetDefault, ValidityDefault,  # noqa
                                Xpbd, _GeneratorSlot)

    reg = _default_registry if registry is None else registry
    batch = spec.batch
    gate = batch.max_gates is not None
    slots = []
    # A gate Program: whether the anchors producer (a gate generator or a stations stage)
    # ran: an elevation after it runs on the gates (rewired), one before it on the
    # relaxed centerline.
    anchored = False
    for i, ref in enumerate(spec.stages):
        sid = ref.slot_id
        if ref.kind in _GENERATOR_KINDS:
            if binding.name != ref.name:
                raise PlanError(f"{_label(ref)}: the bound generator is "
                                f"{binding.name!r}")
            if binding.kind != ref.kind:
                raise PlanError(f"{_label(ref)}: the bound generator is a "
                                f"{binding.kind.replace('_', ' ')}")
            if ref.kind == _GENERATOR:
                kind, cls, inst = "centerline", _GeneratorSlot, _GeneratorSlot(binding)
            elif _is_stage(binding.spec):
                kind, cls, inst = "anchors", binding.spec, None
            else:
                kind, cls, inst = "legacy", _LegacyGateSlot, _LegacyGateSlot(binding)
            slot_layout = _domain_layout(cls, ref, {}, gate=gate, batch=batch,
                                         generator=kind)
            bl = binding.layout
            if (bl.num_envs, bl.stride, bl.device) != \
                    (slot_layout.num_envs, slot_layout.stride, slot_layout.device):
                raise PlanError(f"{_label(ref)}: the generator was bound for {bl}, the "
                                f"batch is {slot_layout}")
            slot = _Slot(i, sid, ref, cls, inst, binding.params, {}, {}, layout=slot_layout,
                         generator=kind)
            if kind == "anchors":
                slot.builtin = bool(reg.describe(_GATE_GENERATOR, ref.name).builtin)
                slot.capacity = _check_capacity(cls, cls().capacity(
                    _static_view(binding.params), slot_layout))
            anchored = kind in ("anchors", "legacy")
            slots.append(slot)
            continue
        builtin = True
        if ref.kind in _stages.OPEN_KINDS:
            try:
                cls = reg.get(ref.kind, ref.name)
            except ValueError as exc:            # unknown name (PluginLoadError passes)
                raise PlanError(f"{_label(ref)}: {exc}") from None
            builtin = bool(reg.describe(ref.kind, ref.name).builtin)
        else:
            cls = _stages.get(ref.kind, ref.name)
        if cls is None:
            track_kinds = sorted(_stages.TRACK_KINDS)
            gate_kinds = sorted({_GATE_GENERATOR, *(
                k for k in (*_stages.available(), *_stages.OPEN_KINDS)
                if k not in _stages.TRACK_KINDS and k != _GENERATOR)})
            if ref.kind not in {*_stages.available(), *_stages.OPEN_KINDS}:
                raise PlanError(f"{_label(ref)}: unknown stage kind {ref.kind!r}; kinds: "
                                f"{['generator', *track_kinds]}; gate pipeline kinds: "
                                f"{gate_kinds}")
            raise PlanError(
                f"{_label(ref)}: unknown {ref.kind} stage {ref.name!r}; built-in "
                f"{ref.kind} stages: {_stages.available(ref.kind)} (kind {ref.kind!r} is "
                f"not open to plugins)")
        # A gate Program's stage before a later anchors producer (a stations stage)
        # runs on the centerline; without any anchors producer every stage is a gate
        # one (the PR11 rewiring, and its hints).
        centerline = gate and not anchored and any(
            r.kind == "stations" for r in spec.stages[i + 1:])
        ports = _effective_ports(cls, ref, gate and not centerline)
        slot_layout = _domain_layout(cls, ref, ports, gate=gate, batch=batch,
                                     generator=None, centerline=centerline)
        if cls in (Xpbd, OffsetDefault, ValidityDefault) and \
                isinstance(ref.params, _MappingABC) and "half_width" in ref.params:
            raise PlanError(
                f"{_label(ref)}: half_width is no longer a {ref.kind}.{ref.name} param "
                f"(0.2.0.dev4): the half-width is cl.half_width, written by the width "
                f"stage (presets.track(half_width=...) / width.constant)")
        params = _coerce(cls.Params, ref.params, what=_label(ref), label="params")
        params.validate(slot_layout)
        _check_values(params)
        if cls is ElevationFlat and float(params.z_base) != 0.0 \
                and not str(ports.get("z", "cl.z")).startswith("gates."):
            raise PlanError(f"{_label(ref)}: z_base={params.z_base!r} sets gate altitudes "
                            f"(gate pipelines); a track's flat altitude is lift.flat's "
                            f"z_base (presets.track(z_base=...))")
        slot = _Slot(i, sid, ref, cls, None, params, {}, {}, builtin, ports=ports,
                     layout=slot_layout, centerline=centerline)
        if "anchors.xy" in {ports.get(p, ch) for p, ch in dict(cls.writes).items()}:
            slot.capacity = _check_capacity(cls, cls().capacity(_static_view(params),
                                                                slot_layout))
            anchored = True
        slots.append(slot)
    _check_agreement(slots)
    return slots


def _check_agreement(slots) -> None:
    """Every stage that bakes ``half_width`` / ``gate_width`` / ``gate_radius`` as a
    host scalar agrees on its value."""
    for field, why in _AGREE.items():
        vals = {s.id: getattr(s.params, field) for s in slots
                if s.generator is None and s.params is not None
                and field in getattr(type(s.params), "__dataclass_fields__", {})}
        if len({float(v) for v in vals.values()}) > 1:
            raise PlanError(f"the stages disagree on {field}: {vals} ({why})")


def _check_gate_plan(spec, slots, registry) -> int:
    """The gate Program's anchors / order / carry / validity rules (before allocating);
    returns ``A`` (the anchors' rows per env, 0 without an anchors producer)."""
    from ..plugin import registry as _default_registry  # noqa: PLC0415
    from ..stages.gates import GateFramesPosed, GateValidityDefault, OrderRaw  # noqa
    from ..stages.gates import _orderings_of, _preferred  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    G = int(spec.batch.max_gates)
    gen = slots[0]
    producers = [s for s in slots if s.capacity is not None]
    A = producers[0].capacity["gates"] if producers else 0
    name = gen.ref.name
    for s in producers:
        cap = s.capacity["gates"]
        what = (f"gate generator {s.ref.name!r}" if s.generator == "anchors" else
                f"{s.ref.kind} stage {s.ref.name!r}")
        if cap > G:
            raise PlanError(f"{_label(s.ref)}: BatchSpec.max_gates={G} is too small for "
                            f"{what}; required max_gates {cap}.")
    if gen.generator == "legacy":
        # A legacy spec writes its gates itself: its max_gates(config) callable is its
        # capacity (0.2.0.dev3's checks).
        binding = gen.instance.binding
        A = int(binding.pipeline_spec.max_gates(binding.run_config))
        if A > G:
            raise PlanError(f"{_label(gen.ref)}: BatchSpec.max_gates={G} is too small for "
                            f"gate generator {name!r}; required max_gates {A}.")
        producers = [gen]
    if producers:
        for s in slots:
            m = getattr(s.params, "min_gates", None) if s.generator is None else None
            if m is not None and int(m) > A:
                p = producers[0]
                if p.generator is not None:
                    what = (f"gate generator {p.ref.name!r}; generator can produce at "
                            f"most {A} gates.")
                else:
                    what = (f"{p.ref.kind} stage {p.ref.name!r}; it writes at most {A} "
                            f"anchors per env (its capacity()['gates']).")
                raise PlanError(f"{_label(s.ref)}: min_gates={m} is too large for {what}")
    orders = [s for s in slots if s.ref.kind == "order"]
    if gen.generator == "legacy":
        A = 0                     # no anchors channel: the legacy spec writes gates.*
    if gen.generator == "anchors":
        supported = _orderings_of(gen.cls, reg)
        preferred = _preferred(gen.cls).get("order")
        for s in orders:
            if s.ref.name not in supported:
                hint = f" (its preferred order is {preferred!r})" if preferred else ""
                raise PlanError(f"{_label(s.ref)}: gate generator {name!r} does not "
                                f"support gate_ordering {s.ref.name!r}; supported: "
                                f"{supported}{hint}")
    for s in slots:
        if s.ref.kind == "carry":
            bad = [o for o in orders if o.cls is not OrderRaw]
            if bad:
                o = bad[0]
                what = "the anchors attributes" if s.ref.name == "attributes" else \
                    f"anchors.{s.ref.name}"
                raise PlanError(f"{_label(s.ref)} carries {what} by anchor "
                                f"index, which needs order.raw; stage {o.id!r} is "
                                f"{o.ref.kind}.{o.ref.name} (it permutes the anchors)")
    posed = any(s.cls is GateFramesPosed for s in slots)
    if posed:
        for s in slots:
            if s.cls is GateValidityDefault and s.params.check_crossing == "width":
                raise PlanError(f"{_label(s.ref)}: posed gates (gate_frames.posed) take "
                                f"their widths from the generator; set "
                                f"check_crossing='always'")
    return A


def _check_provided(slots) -> None:
    """A gate generator's provided attribute must be read by a later stage (a hand-built
    or edited spec cannot silently discard what the generator provides); so must every
    per-anchor attribute a gate generator or a plugin stations stage writes
    (``anchors.z``, ``anchors.<pkg>.<name>``: ``carry.z`` / ``carry.attributes``)."""
    from ..stages.channels import is_plugin_channel  # noqa: PLC0415

    for w in slots:
        if not (w.generator == "anchors" or (w.ref.kind == "stations" and not w.builtin)):
            continue
        later = [s for s in slots if s.index > w.index]
        for ch in sorted(w.writes.values()):
            plugin = is_plugin_channel(ch) and ch.startswith("anchors.")
            stations_z = w.generator is None and ch == "anchors.z"
            if not (plugin or stations_z):
                continue
            if any(ch in s.reads.values() for s in later):
                continue
            stage = "carry.attributes" if plugin else "carry.z"
            raise PlanError(f"{_label(w.ref)} writes {ch}, but no stage reads it: carry it "
                            f"to the gate slots with {stage} after order.raw "
                            f"(presets.gates adds it)")
    gen = slots[0]
    if gen.generator != "anchors":
        return
    for attr in sorted(frozenset(gen.cls.provides)):
        ch = f"anchors.{attr}"
        if any(ch in s.reads.values() for s in slots[1:]):
            continue
        target = f"gates.{attr}"
        other = next((s for s in slots[1:] if target in s.writes.values()), None)
        text = (f"{_label(gen.ref)} provides {attr!r} ({ch}), but no stage reads it")
        if other is not None:
            text += (f"; stage {other.id!r} ({other.ref.kind}.{other.ref.name}) writes "
                     f"{target} instead: carry it with carry.{attr} (presets.gates drops "
                     f"the stages a generator provides)")
        else:
            text += f": carry it with carry.{attr}"
        raise PlanError(text)


def _capture_classes(slots, binding) -> list:
    """Each slot's ``CaptureClass``: the generator's from its binding (a Stage's
    ``capture_class`` of its static params; a legacy spec: STATIC if ``capturable`` else
    HOST), every other stage's ``capture_class`` of its (static) params."""
    from ..plugin.params import _static_view  # noqa: PLC0415
    from ..plugin.stage import CaptureClass  # noqa: PLC0415

    caps = []
    for s in slots:
        if s.generator is not None:
            caps.append(CaptureClass(binding.capture))
            continue
        got = s.cls.capture_class(_static_view(s.params))
        if not isinstance(got, CaptureClass):
            raise TypeError(f"{s.cls.__name__}.capture_class() must return a "
                            f"CaptureClass, got {got!r}")
        caps.append(got)
    return caps


def _host_label(s, binding) -> str:
    """How the ``require_caller_capturable`` refusal names a HOST slot."""
    from ..plugin.stage import _is_stage  # noqa: PLC0415

    if s.generator is not None and not _is_stage(binding.spec):
        return (f"{s.id!r} ({s.ref.kind} {s.ref.name!r}, {type(binding.spec).__name__} "
                f"capturable=False)")
    return f"{s.id!r} ({s.ref.kind}.{s.ref.name})"


def _check_caller_capturable(slots, caps, binding) -> None:
    """``PipelineSpec.require_caller_capturable``: no HOST slot (host-side, before
    anything is allocated)."""
    from ..plugin.stage import CaptureClass  # noqa: PLC0415

    host = [_host_label(s, binding) for s, cap in zip(slots, caps)
            if cap is CaptureClass.HOST]
    if host:
        raise PlanError(
            f"PipelineSpec.require_caller_capturable=True, but stage(s) "
            f"{', '.join(host)} {'is' if len(host) == 1 else 'are'} HOST (they read "
            f"device data back or run host control flow, so they run eagerly and refuse "
            f"a caller's capture): use STATIC / BOUNDED stages or drop "
            f"require_caller_capturable")


def _segments(slots, caps) -> tuple:
    """Planner rule 6: the maximal runs of consecutive slots of equal host-ness (HOST
    vs STATIC / BOUNDED), as :class:`~track_gen.pipeline.Segment` s."""
    from ..plugin.stage import CaptureClass  # noqa: PLC0415
    from .program import Segment  # noqa: PLC0415

    runs: list = []
    for s, cap in zip(slots, caps):
        host = cap is CaptureClass.HOST
        if runs and runs[-1][0] == host:
            runs[-1][1].append((s.id, cap))
        else:
            runs.append((host, [(s.id, cap)]))
    return tuple(Segment(i, max(c for _, c in members),
                         tuple(sid for sid, _ in members))
                 for i, (_, members) in enumerate(runs))


def _slot_info(s, cap, binding, dev):
    """The per-slot context of ``s`` (``track_gen.exec.host._SlotInfo``): its id, label,
    declared capture class (``None``: a legacy generator spec, unchecked) and, for a
    HOST Stage slot, its build-time ``[1]`` int32 host buffer (``HostOps.read_i32``)."""
    from ..exec.host import _SlotInfo  # noqa: PLC0415
    from ..plugin.stage import CaptureClass, _is_stage  # noqa: PLC0415

    legacy = s.generator is not None and not _is_stage(binding.spec)
    if legacy:
        label = f"{s.ref.kind} {s.ref.name!r} ({type(binding.spec).__name__})"
        declared = None
    else:
        cls = binding.spec if s.generator is not None else s.cls
        label = f"{s.ref.kind} {s.ref.name!r} ({cls.__module__}.{cls.__qualname__})"
        declared = cap
    i32 = None
    if cap is CaptureClass.HOST and not legacy:
        i32 = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=dev.is_cuda)
    return _SlotInfo(s.id, label, declared, str(dev.alias), i32)


def _missing_hint(ch: str, s: _Slot, slots, gate: bool) -> str:
    """A hint for the common mistakes behind a read nobody wrote."""
    if ch == "cl.half_width":
        return ("; the width stage runs before relax since 0.2.0.dev4 (generator, "
                "resample, width, relax, uniform, ...): move it, or insert "
                "StageRef('width', 'constant') before 'relax'")
    if ch in ("gates.cum", "gates.perim"):
        return ("; insert StageRef('gate_arclen', 'default') before it (presets.gates "
                "adds it for every elevation but flat)")
    if not gate:
        return ""
    centerline = slots[0].generator == "centerline"
    has_stations = any(o.ref.kind == "stations" for o in slots)
    if centerline and not has_stations and (ch.startswith("anchors.")
                                            or ch.startswith("gates.")):
        return (f"; generator {slots[0].ref.name!r} is a centerline (track) generator: "
                f"insert StageRef('stations', 'raw', params={{'gates': G}}) after it "
                f"(presets.gates(source='stations') does), or set BatchSpec.max_gates="
                f"None for a track pipeline")
    if ch.split(".")[0] in ("cl", "band", "track"):
        if s.ref.kind == "validity":
            return (f"; validity.{s.ref.name} is a track check; the gate pipeline's "
                    f"built-in check is validity.gate_default (checks=('gate_default', "
                    f"...))")
        return (f"; {s.ref.kind}.{s.ref.name} runs on a centerline: in a gate pipeline it "
                f"needs the centerline sub-pipeline of presets.gates("
                f"source='relaxed_stations')")
    return ""


# The catalogue attributes a centerline generator provides, and the stage kind a
# preset drops for each (the second-writer hint).
_PROVIDED_SLOT = {"cl.z": ("z", "elevation"), "cl.half_width": ("half_width", "width")}


def _is_raw_family(ch: str) -> bool:
    return ch.startswith("raw.") and ch != "raw.xy"


def _carry_set(s: _Slot, carries, versions, images, later: set,
               carry_attr_after: bool) -> "tuple[list, list]":
    """The ``(src, dst)`` channels carrier ``s`` carries (static: from the channels
    written before it and the reads after it), and its extra read-only sources (the
    xyz stations' altitude)."""
    from ..stages.channels import family_key, is_plugin_channel  # noqa: PLC0415

    src_dom, dst_dom, mode = carries
    if mode == "index":                     # carry.attributes: anchors -> gates
        return [(ch, "gates." + family_key(ch)) for ch in versions
                if ch.startswith("anchors.") and is_plugin_channel(ch)], []
    if src_dom == "raw":
        candidates = [ch for ch in versions if _is_raw_family(ch)]
    else:                                   # cl: the catalogue families and the images
        candidates = [ch for ch in versions if ch in ("cl.half_width", "cl.z")
                      or (ch.startswith("cl.") and ch in images)]
    if mode == "all":                       # the resamples: every family
        if dst_dom == src_dom:
            return [(ch, ch) for ch in candidates], []
        return [(ch, f"{dst_dom}." + family_key(ch)) for ch in candidates], []
    # stations: a family is carried to anchors only when a later stage reads it.
    pairs = []
    for ch in candidates:
        if ch == "cl.half_width":
            continue
        dst = "anchors." + family_key(ch)
        if dst in later or (is_plugin_channel(ch) and carry_attr_after):
            pairs.append((ch, dst))
    extra = []
    if str(getattr(s.params, "spacing_metric", "xy")) == "xyz":
        z = f"{src_dom}.z"
        if z not in versions:
            raise PlanError(
                f"{_label(s.ref)}: spacing_metric='xyz' measures 3D arc length along the "
                f"loop, which carries no altitude: use a generator that provides z "
                f"(raw.z), or source='relaxed_stations' with a non-flat elevation")
        if z not in {src for src, _ in pairs}:
            extra.append(z)
    return pairs, extra


def _plan_channels(slots, gate: bool = False) -> "dict[str, list[str]]":
    """Resolve the ports (and each carrier's implicit carry ports) and walk the channel
    versions; returns ``channel -> [writer ids]`` (the seed's writer is the prologue)."""
    from ..plugin.stage import _rewrite_problems, _write_rule_problem  # noqa: PLC0415
    from ..stages import CHANNELS, OPEN_KINDS, SEED  # noqa: PLC0415
    from ..stages.channels import (PER_CLASS, is_plugin_channel,  # noqa: PLC0415
                                   required_port_domain)

    def known(ch) -> bool:
        return ch in CHANNELS or is_plugin_channel(ch)

    versions: dict = {SEED: [_PROLOGUE]}
    images: dict = {}                    # carried image channel -> carrier id
    # What every later slot reads (after rewiring), and whether a carry.attributes
    # follows: the stations carry only the families read downstream.
    decl_read_sets = []
    for s in slots:
        r, _ = _decl(s)
        decl_read_sets.append({s.ports.get(p, ch) for p, ch in r.items()})
    later_reads, carry_attr_after = [], []
    acc, flag = set(), False
    for i in range(len(slots) - 1, -1, -1):
        later_reads.append(set(acc))
        carry_attr_after.append(flag)
        acc |= decl_read_sets[i]
        if getattr(slots[i].cls, "_carries", (None, None, None))[2] == "index":
            flag = True
    later_reads.reverse()
    carry_attr_after.reverse()
    for i, s in enumerate(slots):
        ref = s.ref
        ports = s.ports
        decl_reads, decl_writes = _decl(s)
        unknown = sorted(p for p in ref.ports if p not in decl_reads and p not in decl_writes)
        if unknown:
            raise PlanError(f"{_label(ref)}: ports {unknown} are not ports of the stage; "
                            f"read ports: {sorted(decl_reads)}, write ports: "
                            f"{sorted(decl_writes)}")
        for p, ch in decl_reads.items():
            dom = required_port_domain(ch)
            if dom is None:
                continue
            wired = ref.ports.get(p)
            if wired is None:
                example = f"{dom}.<pkg>.<name>"
                avail = sorted(c for c in versions
                               if c.startswith(dom + ".") and is_plugin_channel(c))
                hint = (f"; plugin channels written before it: {avail}" if avail else
                        f"; the {dom} image of a generator's raw.<pkg>.<name> is "
                        f"{dom}.<pkg>.<name>")
                raise PlanError(
                    f"{_label(ref)}: read port {p!r} ({ch}) must be wired: "
                    f"StageRef({ref.kind!r}, {ref.name!r}, ports={{{p!r}: {example!r}}})"
                    f"{hint}")
            if not wired.startswith(dom + ".") or required_port_domain(wired):
                raise PlanError(f"{_label(ref)}: read port {p!r} ({ch}) is wired to "
                                f"{wired!r}, not a {dom} channel")
            own = next((wp_ for wp_, wch in decl_writes.items()
                        if ports.get(wp_, wch) == wired), None)
            if own is not None:
                raise PlanError(
                    f"{_label(ref)}: read port {p!r} ({ch}) is wired to {wired!r}, which "
                    f"the stage itself writes (port {own!r}); a required read port names "
                    f"another stage's channel. Transforming {wired} in place (e.g. "
                    f"rescaling the current width) is a rewrite stage: one port that "
                    f"reads and writes it")
        for p, ch in ports.items():
            if not known(ch):
                raise PlanError(f"{_label(ref)}: port {p!r} is rewired to unknown channel "
                                f"{ch!r}; channels: {sorted(CHANNELS)} and plugin "
                                f"channels '<domain>.<package>.<name>'")
            declared = decl_reads.get(p, decl_writes.get(p))
            if declared not in CHANNELS or ch not in CHANNELS:
                continue                  # a plugin channel: its writer fixes the dtype
            a, b = CHANNELS[declared], CHANNELS[ch]
            if a.dtype is not b.dtype or PER_CLASS[a.per] != PER_CLASS[b.per]:
                raise PlanError(f"{_label(ref)}: port {p!r} ({declared!r}, "
                                f"{a.dtype.__name__} per={a.per!r}) cannot be rewired to "
                                f"{ch!r} ({b.dtype.__name__} per={b.per!r})")
        reads = {p: ports.get(p, ch) for p, ch in decl_reads.items()}
        writes = {p: ports.get(p, ch) for p, ch in decl_writes.items()}
        if ports:
            problems = _rewrite_problems(_label(ref), reads, writes)
            if problems:
                raise PlanError(problems[0])
            if ref.kind in OPEN_KINDS and not s.builtin:
                problem = _write_rule_problem(_label(ref), ref.kind, writes)
                if problem is not None:
                    raise PlanError(problem)
        carries = getattr(s.cls, "_carries", None) if s.generator is None else None
        if carries is not None:
            pairs, extra = _carry_set(s, carries, versions, images, later_reads[i],
                                      carry_attr_after[i])
            for src, dst in pairs:
                reads["~" + src] = src
                writes["~" + dst] = dst
            for src in extra:
                reads["~" + src] = src
            s.carries = tuple(pairs)
        for p, ch in reads.items():
            if not known(ch):
                raise PlanError(f"{_label(ref)} reads unknown channel {ch!r} (port {p!r})")
            if ch not in versions:
                raise PlanError(f"{_label(ref)} reads {ch!r} (port {p!r}), but no earlier "
                                f"stage writes it{_missing_hint(ch, s, slots, gate)}")
        seen: dict = {}
        for p, ch in writes.items():
            if not known(ch):
                raise PlanError(f"{_label(ref)} writes unknown channel {ch!r} (port {p!r})")
            if ch == SEED:
                raise PlanError(f"{_label(ref)} writes {SEED!r}: only the Program's "
                                f"prologue writes it")
            if ch in seen:
                raise PlanError(f"{_label(ref)} writes {ch!r} through two ports "
                                f"({seen[ch]!r}, {p!r})")
            seen[ch] = p
            if ch in versions and ch not in reads.values():
                hint = ""
                if ch in images and ch in _PROVIDED_SLOT and \
                        f"raw.{_PROVIDED_SLOT[ch][0]}" in versions:
                    attr, kind = _PROVIDED_SLOT[ch]
                    hint = (f"; generator {slots[0].ref.name!r} provides {attr} "
                            f"(raw.{attr} -> {ch}): drop the {kind} stage")
                raise PlanError(
                    f"{_label(ref)} writes {ch!r}, already written by stage "
                    f"{versions[ch][-1]!r}: a second writer must also read the channel "
                    f"(an in-place rewrite){hint}")
        for p, ch in writes.items():
            versions.setdefault(ch, []).append(s.id)
            if p.startswith("~"):
                images[ch] = s.id
        s.reads, s.writes = reads, writes
    _check_width_placement(slots)
    return versions


def _check_width_placement(slots) -> None:
    """A ``cl.half_width`` version written after the stage that writes
    ``track.half_width`` (the offset) would let validity check another width than the
    band the Track has."""
    offset = next((s for s in slots if "track.half_width" in s.writes.values()), None)
    if offset is None:
        return
    for s in slots:
        if s.index > offset.index and "cl.half_width" in s.writes.values():
            raise PlanError(f"stage {s.id!r} writes cl.half_width after {offset.id!r}: "
                            f"the band and Track.half_width come from {offset.id}'s read, "
                            f"so validity would check another width than the band; move "
                            f"it before {offset.id!r}")


# ---------------------------------------------------------------------------------------
# The result kinds (Track / GateSequence)
# ---------------------------------------------------------------------------------------

# Result fields a pipeline may leave without a writer (then None / the commit's own).
_OPTIONAL_EXPORTS = ("half_width",)


def _check_exports(writers) -> None:
    """Every ``Track`` field has a writer: the ``Track`` reads them all (``half_width``
    excepted: a pipeline without ``offset.default`` leaves it ``None``)."""
    from ..stages.track import TRACK_EXPORTS  # noqa: PLC0415

    missing = [f"Track.{field} ({' or '.join(chs)})"
               for field, chs in TRACK_EXPORTS.items()
               if field not in _OPTIONAL_EXPORTS and not any(ch in writers for ch in chs)]
    if missing:
        raise PlanError(f"no stage writes {', '.join(missing)}: every Track field needs "
                        f"a writer")


def _check_gate_exports(writers) -> None:
    """Every ``GateSequence`` field has a writer."""
    from ..stages.gates import GATE_EXPORTS  # noqa: PLC0415

    missing = [f"GateSequence.{field} ({' or '.join(chs)})"
               for field, chs in GATE_EXPORTS.items()
               if not any(ch in writers for ch in chs)]
    if missing:
        raise PlanError(f"no stage writes {', '.join(missing)}: every GateSequence field "
                        f"needs a writer")


def _export_sources(writers, exports=None, own_valid: str = "track.valid") -> \
        "dict[str, str | None]":
    """``result field -> the channel whose FINAL version is the field's array`` (the
    first written of its candidates), or ``None`` for a field the commit writes:
    ``valid`` when the built-in check (``validity.default`` / ``validity.gate_default``,
    whose own flag is ``own_valid``) is not also the last writer of ``valid.fail`` (a
    plugin check ran after it, or none ran). A field without any writer is absent (only
    reachable when the exports check is lifted)."""
    from ..stages.channels import FAIL  # noqa: PLC0415
    from ..stages.track import TRACK_EXPORTS  # noqa: PLC0415

    exports = TRACK_EXPORTS if exports is None else exports
    out: dict = {}
    for field, chs in exports.items():
        if field == "valid":
            if own_valid in writers and FAIL in writers and \
                    writers[own_valid][-1] == writers[FAIL][-1]:
                out[field] = own_valid
            elif FAIL in writers:
                out[field] = None                     # derived: fail_mask == 0
            elif own_valid in writers:
                out[field] = own_valid
            continue
        ch = next((c for c in chs if c in writers), None)
        if ch is not None:
            out[field] = ch
    return out


def _check_fail_start(slots, writers) -> None:
    """``valid.fail`` starts at ``validity.default``: until the split built-in checks
    exist, a stage that starts the bitmask in its place (writes ``valid.fail`` without
    reading it) drops the generator, turning, thickness and truncation checks, so a
    truncated loop would pass (truncation must fail closed)."""
    from ..stages.channels import FAIL  # noqa: PLC0415
    from ..stages.track import ValidityDefault  # noqa: PLC0415

    first = writers.get(FAIL, (None,))[0]
    slot = next((s for s in slots if s.id == first), None)
    if slot is not None and slot.cls is not ValidityDefault:
        raise PlanError(
            f"{_label(slot.ref)} starts valid.fail (writes it without reading it) in "
            f"place of validity.default, which would drop the built-in generator, "
            f"turning, thickness and truncation checks; keep 'default' first "
            f"(checks=('default', {slot.ref.name!r})) and read valid.fail in the check "
            f"(fail_out = fail_in | own). Replacing validity.default arrives with the "
            f"split built-in checks")


def _check_gate_fail_start(slots, writers) -> None:
    """``valid.fail`` starts at ``validity.gate_default`` in a gate Program."""
    from ..stages.channels import FAIL  # noqa: PLC0415
    from ..stages.gates import GateValidityDefault  # noqa: PLC0415

    first = writers.get(FAIL, (None,))[0]
    slot = next((s for s in slots if s.id == first), None)
    if slot is not None and slot.cls is not GateValidityDefault:
        raise PlanError(
            f"{_label(slot.ref)} starts valid.fail (writes it without reading it) in "
            f"place of validity.gate_default, which would drop the built-in count, "
            f"finite, tangent, spacing, crossing and grade checks; keep 'gate_default' "
            f"first (checks=('gate_default', {slot.ref.name!r})) and read valid.fail in "
            f"the check (fail_out = fail_in | own)")


def _check_flat_lift(slots) -> None:
    """``lift.flat`` lifts at a constant ``z_base`` and ignores ``cl.z``: refuse it after
    a ``cl.z`` writer other than ``elevation.flat`` (the geometry would be flat while the
    validity grade check uses the elevation)."""
    from ..stages.track import ElevationFlat, LiftFlat  # noqa: PLC0415

    z_writers = []
    for s in slots:
        if s.cls is LiftFlat:
            other = [w for w in z_writers if w.cls is not ElevationFlat]
            if other:
                raise PlanError(f"{_label(s.ref)} ignores cl.z written by "
                                f"{_label(other[-1].ref)}; use lift.elevated")
        if "cl.z" in s.writes.values():
            z_writers.append(s)


def _check_arrays(out, want: dict, optional, dev: str) -> None:
    for field, (dtype, n) in want.items():
        arr = getattr(out, field, None)
        if arr is None and field in optional:
            continue
        if not isinstance(arr, wp.array):
            raise ValueError(f"out.{field}: expected a wp.array, got {type(arr).__name__}")
        got = (getattr(arr.dtype, "__name__", str(arr.dtype)), tuple(arr.shape),
               str(arr.device))
        if arr.dtype is not _TRACK_DTYPES[dtype] or tuple(arr.shape) != (n,) \
                or str(arr.device) != dev or not arr.is_contiguous:
            raise ValueError(f"out.{field}: expected a contiguous [{n}] {dtype} array on "
                             f"{dev}, got {got}")


def _check_out_attributes(out, n: int, dev: str) -> None:
    """``out.attributes``: a mapping of ``[n]`` contiguous arrays on ``dev`` (keys and
    dtypes are checked against the exports once the stages are built)."""
    attrs = getattr(out, "attributes", None)
    if attrs is None:
        return
    if not isinstance(attrs, _MappingABC):
        raise ValueError(f"out.attributes: expected a mapping of wp.array, got "
                         f"{type(attrs).__name__}")
    for key, arr in attrs.items():
        if not isinstance(arr, wp.array) or tuple(arr.shape) != (n,) or \
                str(arr.device) != dev or not arr.is_contiguous:
            got = (getattr(getattr(arr, "dtype", None), "__name__", None),
                   tuple(getattr(arr, "shape", ())), str(getattr(arr, "device", None)))
            raise ValueError(f"out.attributes[{key!r}]: expected a contiguous [{n}] array "
                             f"on {dev}, got {got}")


def _check_out(out, layout) -> None:
    """``out=``: every Track field an ``[n]`` contiguous array of the right dtype on the
    batch's device (``generation``, ``fail_mask`` and ``half_width`` may be ``None``)."""
    E, S = int(layout.num_envs), int(layout.stride)
    want = {f: ("vec3f", E * S) for f in ("outer", "center", "inner", "tangent", "normal")}
    want.update(arclen=("float32", E * S), length=("float32", E), winding=("float32", E),
                valid=("int32", E), count=("int32", E), truncated=("int32", E),
                generation=("int32", E), fail_mask=("int32", E),
                half_width=("float32", E * S))
    dev = str(wp.get_device(layout.device))
    _check_arrays(out, want, ("generation", "fail_mask", "half_width"), dev)
    _check_out_attributes(out, E * S, dev)


def _check_gate_out(out, layout) -> None:
    """``out=`` of a gate Program: every ``GateSequence`` field (``generation`` and
    ``fail_mask`` may be ``None``)."""
    E, G = int(layout.num_envs), int(layout.stride)
    want = {f: ("vec3f", E * G) for f in ("position", "tangent", "forward", "left",
                                          "right")}
    want.update(orientation=("quatf", E * G), half_size=("float32", E * G),
                valid=("int32", E), count=("int32", E), generation=("int32", E),
                fail_mask=("int32", E))
    dev = str(wp.get_device(layout.device))
    _check_arrays(out, want, ("generation", "fail_mask"), dev)
    _check_out_attributes(out, E * G, dev)


@dataclasses.dataclass(frozen=True)
class _ResultKind:
    """What differs between a track and a gate Program."""

    exports: object
    own_valid: str
    check_exports: object
    check_fail_start: object
    check_out: object
    alloc: object
    commit_cls: object
    fields: tuple
    domain: str = "cl"          # the result's point domain (its attributes' prefix)


def _result_kind(gate: bool) -> _ResultKind:
    from ..stages import commit as _commit_mod  # noqa: PLC0415
    from ..stages import gates as _gates_mod  # noqa: PLC0415
    from ..stages import track as _track_mod  # noqa: PLC0415

    if gate:
        return _ResultKind(
            _gates_mod.GATE_EXPORTS, "gates.valid", _check_gate_exports,
            _check_gate_fail_start, _check_gate_out, _gates_mod._alloc_gate_sequence,
            _commit_mod._GateCommit,
            (*_commit_mod.GATE_POINT_FIELDS, *_commit_mod.GATE_ENV_FIELDS), "gates")
    return _ResultKind(
        _track_mod.TRACK_EXPORTS, "track.valid", _check_exports, _check_fail_start,
        _check_out, _track_mod._alloc_track, _commit_mod._Commit,
        (*_commit_mod.POINT_FIELDS, *_commit_mod.ENV_FIELDS))


# ---------------------------------------------------------------------------------------
# The plan (allocation)
# ---------------------------------------------------------------------------------------

class _Plan:
    """Channel versions, exports, plugin channels and fail bits, shared by every
    stage's builder."""

    def __init__(self, layout, raw: int, device, writers, bound, *, n_max=None,
                 max_gates=None, anchors: int = 0) -> None:
        self.layout = layout
        self.raw = int(raw)
        self.device = device
        self.n_max = int(layout.stride if n_max is None else n_max)
        self.max_gates = None if max_gates is None else int(max_gates)
        self.anchors = int(anchors)
        self.final = {ch: ids[-1] for ch, ids in writers.items()}
        self.bound = dict(bound)              # exported channel -> result array
        self.current: dict = {}               # channel -> latest version
        self.versions: dict = {}              # channel -> [(index, stage_id, array)]
        self.plugin_channels: dict = {}       # plugin channel -> (dtype, per)
        self.fail_bits: dict = {}             # bit name -> bit index (assignment order)
        self.fail_owner: dict = {}            # bit name -> stage id
        self.bits_of: dict = {}               # stage id -> [bit names]
        # Roadmap PR13a: the carried families (key -> _Family, fixed by the first
        # writer), the constant versions (id(array) -> value) and the cl resamplers'
        # slot indices (the stale-read rule).
        self.families: dict = {}
        self.constants: dict = {}
        self.images: set = set()
        self.resamplers: tuple = ()
        self.slot_ids: list = []              # slot index -> stage id
        self.index_of: dict = {}              # id(array) -> the writer's slot index

    def length(self, per: str) -> int:
        E = int(self.layout.num_envs)
        if per == "env":
            return E
        if per == "raw":
            return E * self.raw
        if per == "cl":
            return E * self.n_max
        if per == "anchors" and self.anchors > 0:
            return E * self.anchors
        if per == "gates" and self.max_gates is not None:
            return E * self.max_gates
        if per in ("anchors", "gates"):
            raise ValueError(f"per={per!r}: this pipeline has no {per} domain")
        raise ValueError(f"unknown per={per!r}; expected 'env', 'raw', 'cl', 'anchors' "
                         f"or 'gates'")

    def add(self, channel: str, index: int, stage_id: str, arr) -> None:
        self.current[channel] = arr
        self.versions.setdefault(channel, []).append((index, stage_id, arr))
        self.index_of[id(arr)] = index

    def declare_constant(self, arr, value: float) -> None:
        self.constants[id(arr)] = float(value)

    def constant_of(self, arr) -> "float | None":
        return self.constants.get(id(arr))

    def record_family(self, channel: str, dtype, interp, export, writer: str) -> None:
        """The first writer of a family channel fixes its dtype / interp / export."""
        from ..stages.channels import family_key  # noqa: PLC0415

        key = family_key(channel)
        if key not in self.families:
            self.families[key] = _Family(key, channel, dtype, interp, export, writer)

    def for_stage(self, slot: _Slot) -> "_StagePlan":
        return _StagePlan(self, slot)

    # -- fail bits ------------------------------------------------------------------
    def _claim(self, slot: _Slot, name: str, bit: int) -> None:
        if name in self.fail_owner:
            raise PlanError(f"{_label(slot.ref)}: fail bit name {name!r} is already taken "
                            f"by stage {self.fail_owner[name]!r}")
        self.fail_bits[name] = bit
        self.fail_owner[name] = slot.id
        self.bits_of.setdefault(slot.id, []).append(name)

    def fail_bit(self, slot: _Slot, name: str) -> int:
        from ..stages.channels import PLUGIN_FAIL_BITS  # noqa: PLC0415

        taken = {b for b in self.fail_bits.values() if b in PLUGIN_FAIL_BITS}
        free = [b for b in PLUGIN_FAIL_BITS if b not in taken]
        if not free:
            names = [n for n, b in self.fail_bits.items() if b in PLUGIN_FAIL_BITS]
            raise PlanError(f"{_label(slot.ref)}: no plugin fail bit left: bits 16-30 are "
                            f"taken by {names}; a pipeline holds at most "
                            f"{len(PLUGIN_FAIL_BITS)} plugin fail bits")
        self._claim(slot, name, free[0])
        return free[0]

    def reserve_fail_bits(self, slot: _Slot, cls, table) -> None:
        if not slot.builtin:
            raise PlanError(f"{_label(slot.ref)}: bits 0-15 are reserved for TrackGen's "
                            f"built-in checks; a plugin check takes its bit from "
                            f"b.fail_bit()")
        for name, bit in table.items():
            if not 0 <= int(bit) < 16:
                raise PlanError(f"{_label(slot.ref)}: built-in fail bit {name!r}={bit} is "
                                f"outside 0-15")
            self._claim(slot, str(name), int(bit))


@dataclasses.dataclass(frozen=True)
class _Family:
    """A carried (or exported) attribute: its dtype, interp and export key, fixed by its
    first writer and shared by every image."""

    key: str
    origin: str                 # the first writer's channel (raw.acme.grip, cl.z, ...)
    dtype: object
    interp: "str | None"
    export: "str | None"        # the result key, or None
    writer: str                 # the first writer's stage id


@dataclasses.dataclass(frozen=True)
class _Carry:
    """One family a carrier carries (``StageBuilder._carried()``)."""

    src_port: str
    dst_port: str
    src: str
    dst: str
    key: str
    dtype: object
    interp: "str | None"
    constant: "float | None"    # the source version's constant value


class _StagePlan:
    """What a stage's :class:`~track_gen.plugin.StageBuilder` sees of the plan."""

    def __init__(self, plan: _Plan, slot: _Slot) -> None:
        self._plan = plan
        self._slot = slot
        self.reads = types.MappingProxyType(dict(slot.reads))
        self.writes = types.MappingProxyType(dict(slot.writes))
        decl_reads, decl_writes = _decl(slot)
        self._declared = {**decl_reads, **decl_writes}
        # The versions written by EARLIER stages (a channel this stage rewrites keeps
        # reading the version before its own).
        self._visible = {ch: plan.current[ch] for ch in slot.reads.values()}

    def channel(self, name: str):
        """``(dtype, per)`` of a catalogue channel or of a plugin channel already
        written; ``None`` for a plugin channel nobody wrote yet."""
        from ..stages import CHANNELS  # noqa: PLC0415

        c = CHANNELS.get(name)
        if c is not None:
            return c.dtype, c.per
        return self._plan.plugin_channels.get(name)

    def declared_channel(self, port: str) -> "str | None":
        """The channel the stage class declares for ``port`` (before any rewiring)."""
        return self._declared.get(port)

    def length(self, per: str) -> int:
        return self._plan.length(per)

    def define(self, name: str, dtype, per: str) -> None:
        """Fix a plugin channel's dtype and size (its first writer)."""
        self._plan.plugin_channels[name] = (dtype, per)

    def is_first_write(self, channel: str) -> bool:
        """Whether no earlier stage wrote ``channel`` (its first writer fixes its dtype,
        interp and export)."""
        return channel not in self._plan.current

    def read(self, port: str):
        ch = self.reads[port]
        arr = self._visible[ch]
        self._check_stale(ch, arr)
        return arr

    def _check_stale(self, ch: str, arr) -> None:
        """The stale-read rule: a ``per="cl"`` plugin channel a STAGE wrote (not a
        carried image) before a cl resampler that lies between its writer and this
        reader sits on the pre-resample points."""
        plan, slot = self._plan, self._slot
        from ..stages.channels import is_plugin_channel  # noqa: PLC0415

        if not (is_plugin_channel(ch) and ch.startswith("cl.")) or ch in plan.images:
            return
        info = plan.plugin_channels.get(ch)
        if info is None or info[1] != "cl":
            return
        w = plan.index_of.get(id(arr))
        if w is None:
            return
        between = [u for u in plan.resamplers if w < u < slot.index]
        if between:
            writer = next(sid for idx, sid, a in plan.versions[ch] if a is arr)
            res = plan.slot_ids[between[0]]
            raise PlanError(
                f"stage {slot.id!r} reads {ch}, written by {writer!r} before the uniform "
                f"resample ({res!r}), which does not carry stage-written channels (the "
                f"values sit on the pre-uniform points): write it after {res!r}, or "
                f"carry it from the generator (raw.{ch[3:]})")

    def write(self, port: str, *, interp=None, export=None, fill=None):
        plan, slot = self._plan, self._slot
        ch = self.writes[port]
        dtype, per = self.channel(ch)
        bound = plan.bound.get(ch)
        if bound is not None and plan.final.get(ch) == slot.id:
            if bound.dtype is not dtype:
                raise ValueError(
                    f"out=: the array bound to {ch} has dtype "
                    f"{getattr(bound.dtype, '__name__', bound.dtype)}; the channel is "
                    f"{getattr(dtype, '__name__', dtype)}")
            arr = bound
        elif fill is not None:
            arr = wp.full(plan.length(per), fill, dtype=dtype, device=plan.device)
        else:
            arr = wp.zeros(plan.length(per), dtype=dtype, device=plan.device)
        first = ch not in plan.current
        plan.add(ch, slot.index, slot.id, arr)
        from ..stages.channels import CHANNELS, channel_domain  # noqa: PLC0415

        if port.startswith("~"):
            # A carried image: the family of its source; a constant source stays one.
            plan.images.add(ch)
            src = next(src for src, dst in slot.carries if dst == ch)
            value = plan.constant_of(self._visible.get(src))
            if value is not None:
                plan.declare_constant(arr, value)
        elif first and channel_domain(ch) is not None and per != "env" and (
                ch not in CHANNELS or ch in ("raw.z", "raw.half_width")):
            interp = interp if interp is not None else (
                CHANNELS[ch].interp if ch in CHANNELS else None)
            plan.record_family(ch, dtype, interp, export, slot.id)
        return arr

    def declare_constant(self, arr, value: float) -> None:
        self._plan.declare_constant(arr, value)

    def constant_of(self, arr) -> "float | None":
        return self._plan.constant_of(arr)

    def carried(self) -> list:
        """The families this carrier carries, in order (``_Carry``)."""
        from ..stages.channels import CHANNELS, family_key  # noqa: PLC0415

        plan = self._plan
        out = []
        for src, dst in self._slot.carries:
            key = family_key(src)
            fam = plan.families.get(key)
            if src in CHANNELS:
                dtype, interp = CHANNELS[src].dtype, CHANNELS[src].interp
            elif fam is not None:
                dtype, interp = fam.dtype, fam.interp
            else:
                dtype, interp = plan.plugin_channels[src][0], None
            out.append(_Carry("~" + src, "~" + dst, src, dst, key, dtype, interp,
                              plan.constant_of(self._visible[src])))
        return out

    def fail_bit(self, name: str) -> int:
        return self._plan.fail_bit(self._slot, name)

    def reserve_fail_bits(self, cls, table) -> None:
        self._plan.reserve_fail_bits(self._slot, cls, table)


def _result_attributes(plan: _Plan, kind) -> "tuple[dict, dict]":
    """``(key -> final array, key -> AttributeInfo)`` of every exported family whose
    result-domain image exists (``cl.<key>`` for a Track, ``gates.<key>`` for a
    GateSequence: best effort, an attribute with no path to the gates is left out). An
    exported channel a STAGE wrote before a cl resampler is refused (the commit's read
    follows the stale-read rule)."""
    from .program import AttributeInfo  # noqa: PLC0415

    attrs, infos = {}, {}
    for key, fam in plan.families.items():
        if fam.export is None:
            continue
        ch = f"{kind.domain}.{key}"
        arr = plan.current.get(ch)
        if arr is None:
            continue
        if kind.domain == "cl" and ch not in plan.images:
            w = plan.index_of.get(id(arr), -1)
            later = [u for u in plan.resamplers if u > w]
            if later:
                writer = plan.versions[ch][-1][1]
                raise PlanError(
                    f"{ch} is exported, but stage {writer!r} writes it before the uniform "
                    f"resample ({plan.slot_ids[later[0]]!r}), which does not carry "
                    f"stage-written channels: write it after "
                    f"{plan.slot_ids[later[0]]!r}, or carry it from the generator "
                    f"(raw.{key})")
        pad = -1 if fam.dtype is wp.int32 else float("nan")
        attrs[key] = arr
        infos[key] = AttributeInfo(key=key, channel=ch, dtype=fam.dtype,
                                   interp=fam.interp, padding=pad, domain=kind.domain)
    return attrs, infos


def _seed_state(seeds, E: int, dev, stream):
    """The Program's ``SeedState``: built from a seed form, or ``seeds`` itself (checked
    against the batch). ``build`` admits only those (:func:`_check_seeds`);
    ``TrackGenerator`` passes the SeedState it built (a test stand-in passes through)."""
    from ..seeds import SeedState  # noqa: PLC0415

    if _is_seed_form(seeds):
        return SeedState(E, dev, seeds, default_stream=stream)
    if isinstance(seeds, SeedState):
        if seeds.num_envs != E or str(wp.get_device(seeds.device)) != str(dev):
            raise ValueError(f"seeds: a SeedState for {seeds.num_envs} envs on "
                             f"{seeds.device!r}; the batch is {E} envs on {dev}")
    return seeds


def _build(spec: PipelineSpec, *, binding, seeds=None, out=None, stream=None,
           prepare: bool = True, owner=None, registry=None):
    """:func:`build` with the generator already bound (the facades pass their own
    binding and run config)."""
    from .._src.generator_binding import _StageGenerator, param_table  # noqa: PLC0415
    from ..exec.host import _SlotEnqueue  # noqa: PLC0415
    from ..exec.runner import _check_default_stream, ensure_not_capturing  # noqa: PLC0415
    from ..plugin.params import _fields, _new_param_table, _new_stage_params, \
        _static_view  # noqa: PLC0415
    from ..plugin.stage import Layout, _close_builder, _new_builder  # noqa: PLC0415
    from ..stages.channels import FAIL  # noqa: PLC0415
    from .program import _PROGRAM_TOKEN, Program  # noqa: PLC0415

    # -- 1-5: host-side checks, nothing allocated --------------------------------------
    _check_spec_shape(spec)
    batch = spec.batch
    gate = batch.max_gates is not None
    kind = _result_kind(gate)
    E = int(batch.num_envs)
    S = int(batch.max_gates) if gate else int(batch.n_max)
    dev = wp.get_device(batch.device)
    layout = Layout(E, S, str(dev), True)
    slots = _resolve(spec, binding, layout, registry)
    A = _check_gate_plan(spec, slots, registry) if gate else 0
    caps = _capture_classes(slots, binding)
    if spec.require_caller_capturable:
        _check_caller_capturable(slots, caps, binding)
    writers = _plan_channels(slots, gate)
    if gate:
        _check_provided(slots)
    kind.check_exports(writers)
    kind.check_fail_start(slots, writers)
    _check_flat_lift(slots)
    if out is not None:
        kind.check_out(out, layout)
    adapter = binding.pipeline_spec if isinstance(binding.pipeline_spec, _StageGenerator) \
        else None
    if adapter is not None:
        raw = int(adapter.raw_points)
    else:
        raw = int(getattr(binding.run_config, "num_points", 0) or 0) \
            if binding.kind == _GENERATOR else 0
    _check_default_stream(dev, stream)

    # -- 6: shared allocations (refuse a capture first; the seeds' own checks next) ---
    ensure_not_capturing(dev, default_stream=stream, label=binding.name)
    seed_state = _seed_state(seeds, E, dev, stream)
    result = out if out is not None else kind.alloc(layout)
    # The final version of each exported channel IS the result's array (no copy).
    sources = _export_sources(writers, kind.exports, kind.own_valid)
    if out is None and "half_width" in kind.exports and "half_width" not in sources:
        result.half_width = None          # no offset.default: nothing writes it
    bound = {}
    for field, ch in sources.items():
        arr = getattr(result, field, None)
        if ch is not None and arr is not None:
            bound[ch] = arr
    # out.attributes: a key whose result-domain channel this pipeline writes is bound
    # (its dtype checked at the write); the rest are refused once the stages are built.
    out_attrs = dict(getattr(out, "attributes", None) or {}) if out is not None else {}
    for key, arr in out_attrs.items():
        ch = f"{kind.domain}.{key}"
        if ch in writers:
            bound[ch] = arr
    seed = wp.empty(E, dtype=wp.int32, device=dev)
    regen_mask = wp.zeros(E, dtype=wp.int32, device=dev)
    plan = _Plan(layout, raw, dev, writers, bound, n_max=int(batch.n_max),
                 max_gates=batch.max_gates, anchors=A)
    plan.slot_ids = [s.id for s in slots]
    plan.resamplers = tuple(s.index for s in slots if s.generator is None and getattr(
        s.cls, "_carries", (None, None, None))[:2] == ("cl", "cl"))
    plan.add("seed", -1, _PROLOGUE, seed)

    # -- 7: build every stage, in order ------------------------------------------------
    enqueues, pairs, scratch, salts = [], [], {}, {}
    inputs = {"regen_mask": regen_mask}
    stage_params = {}
    gen_scratch = None
    for s, cap in zip(slots, caps):
        bound_gen = s.generator in ("centerline", "legacy")
        params = None if bound_gen else s.params
        dynamic = {}
        if params is not None:
            for f in _fields(type(params)):
                if f.kind != "dynamic":
                    continue
                v = getattr(params, f.name)
                v = float(v) if f.dtype is wp.float32 else int(v)
                dynamic[f.name] = (wp.full(E, v, dtype=f.dtype, device=dev),
                                   wp.full(E, v, dtype=f.dtype, device=dev))
        view = None if params is None else _static_view(params)
        instance = s.instance if bound_gen else s.cls()
        b = _new_builder(stage=instance, params=view, layout=s.layout, scale=batch.scale,
                         seed=seed, raw=raw, outputs={}, dynamic=dynamic, stage_id=s.id,
                         plan=plan.for_stage(s), capture=cap)
        enqueue = instance.build(b, view)
        _close_builder(b)
        if not callable(enqueue):
            raise TypeError(f"{type(instance).__name__}.build() must return an "
                            f"enqueue(ctx) callable, got {enqueue!r}")
        missing = sorted(p for p in s.writes if p not in b._written)
        if missing:
            raise PlanError(f"{_label(s.ref)} declares write ports {missing} but build() "
                            f"did not call b.write() for them")
        if s.ref.kind == "validity" and not plan.bits_of.get(s.id):
            raise PlanError(f"{_label(s.ref)} never called b.fail_bit(): a validity stage "
                            f"owns at least one fail bit")
        # Every stage runs inside its slot context (ctx.host(), the bounded_loop
        # declaration check and host poll); the wrapper itself emits nothing.
        enqueues.append(_SlotEnqueue(_slot_info(s, cap, binding, dev), enqueue))
        if bound_gen:
            gen_scratch = instance.scratch
            if adapter is not None:
                pairs.extend(adapter._pairs)
                stage_inputs = dict(adapter.inputs)
                stage_salts = dict(adapter._built.builder._salts)
            else:
                stage_inputs, stage_salts = {}, {}
        else:
            pairs.extend(list(dynamic.values()) + list(b._inputs.values()))
            stage_inputs = {name: pair[0] for name, pair in b._inputs.items()}
            stage_salts = dict(b._salts)
            stage_params[s.id] = _new_stage_params(
                stage_id=s.id, params=s.params,
                arrays={name: pair[0] for name, pair in dynamic.items()},
                num_envs=E, device=str(dev), default_stream=stream)
            if s.generator == "anchors":
                gen_scratch = types.MappingProxyType(
                    {k.split("/", 1)[1]: v for k, v in b._scratch.items()})
        for name, arr in stage_inputs.items():
            if name in inputs:
                raise PlanError(f"{_label(s.ref)}: input {name!r} is declared twice")
            inputs[name] = arr
        for stream_name, value in stage_salts.items():
            for (other, other_stream), v in salts.items():
                if v == value:
                    raise PlanError(f"{_label(s.ref)}: salt stream {stream_name!r} "
                                    f"collides with stage {other!r} stream "
                                    f"{other_stream!r} (0x{value:08x}); rename one")
            salts[(s.id, stream_name)] = value
        prefix = f"scratch:{s.id}/"
        scratch[s.id] = {k[len(prefix):] if k.startswith(prefix) else k: v
                         for k, v in b._scratch.items()}

    # -- 7b: the exported attributes (roadmap PR13a) ------------------------------------
    attributes, attribute_info = _result_attributes(plan, kind)
    extra = sorted(k for k in out_attrs if k not in attributes)
    if extra:
        raise ValueError(f"out.attributes[{extra[0]!r}]: this pipeline does not export "
                         f"{extra[0]!r}; exported: {sorted(attributes)} (out= with keys "
                         f"from another pipeline is reserved for PR13b's mixed batches)")
    if out is None and hasattr(result, "attributes"):
        result.attributes = types.MappingProxyType(dict(attributes))

    # -- 8: the commit (every result field has a source; else the result is not ours) --
    commit = None
    fields = kind.fields
    if all(f in sources or f in _OPTIONAL_EXPORTS for f in fields):
        # The result's arrays; fail_mask: the final valid.fail version (the result's own
        # unless out= brought none); half_width: the final track.half_width version
        # (out= may bring none), or a private buffer when nothing writes it.
        targets = {f: getattr(result, f) for f in fields
                   if f not in ("fail_mask", "half_width")}
        targets["fail_mask"] = plan.current[FAIL]
        if "half_width" in fields:
            hw = plan.current.get("track.half_width")
            targets["half_width"] = hw if hw is not None else wp.full(
                E * S, float("nan"), dtype=wp.float32, device=dev)
        commit = kind.commit_cls(
            targets=targets, E=E, S=S, device=dev, dummy_mask=regen_mask,
            poison=batch.on_invalid == "poison", derive_valid=sources["valid"] is None,
            attributes=attributes)
    if hasattr(result, "fail_bits"):
        result.fail_bits = types.MappingProxyType(dict(plan.fail_bits))

    # -- 9: the Program ----------------------------------------------------------------
    if _GENERATOR in stage_params:
        gen_params = stage_params.pop(_GENERATOR)
    else:
        gen_params = param_table(binding, default_stream=stream)[_GENERATOR]
    table = _new_param_table({_GENERATOR: gen_params, **stage_params})
    resolved = dataclasses.replace(spec, stages=tuple(
        dataclasses.replace(s.ref, params=s.params) for s in slots))
    program = Program(
        _token=_PROGRAM_TOKEN, spec=resolved, layout=layout, binding=binding,
        result=result, seeds=seed_state, seed=seed, inputs=inputs, params=table,
        pairs=pairs, enqueues=enqueues, versions=plan.versions, scratch=scratch,
        stage_ids=[s.id for s in slots],
        kinds={s.id: (s.ref.kind, s.ref.name) for s in slots}, gen_scratch=gen_scratch,
        stream=stream, owner=owner, commit=commit, fail_bits=plan.fail_bits,
        fail_owner=plan.fail_owner, capture_classes=caps,
        segments=_segments(slots, caps), on_invalid=batch.on_invalid, writers=writers,
        ports={s.id: (dict(s.reads), dict(s.writes)) for s in slots},
        attributes=attribute_info)
    if prepare:
        program.prepare()
    return program
