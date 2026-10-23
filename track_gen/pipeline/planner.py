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
    "half_width": "one value until roadmap PR13",
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
            (the registry's message).
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


def _effective_ports(cls, ref: StageRef, gate: bool) -> dict:
    """``StageRef.ports``, plus, for an ``elevation`` stage of a gate Program, the
    default gate rewiring of its ``cl.*`` ports (explicit ports override it)."""
    from ..stages.channels import ELEVATION_GATE_CHANNELS  # noqa: PLC0415

    ports = {}
    if gate and ref.kind == "elevation":
        for p, ch in {**dict(cls.reads), **dict(cls.writes)}.items():
            if ch in ELEVATION_GATE_CHANNELS:
                ports[p] = ELEVATION_GATE_CHANNELS[ch]
    ports.update(ref.ports)
    return ports


def _domain_layout(slot_cls, ref: StageRef, ports: dict, *, gate: bool, batch,
                   generator: "str | None"):
    """The stage's :class:`Layout` from its (rewired) ports, before any validation:
    ``N_max`` on a track and for the centerline stages of a gate Program, ``max_gates``
    for every gate stage."""
    from ..plugin.stage import Layout  # noqa: PLC0415
    from ..stages.channels import CHANNELS, is_gate_channel, is_plugin_channel  # noqa

    E, dev = int(batch.num_envs), str(wp.get_device(batch.device))
    n_max = int(batch.n_max)
    if generator in ("centerline",):
        return Layout(E, n_max, dev, True)
    if generator in ("anchors", "legacy"):
        return Layout(E, int(batch.max_gates), dev, True)
    declared = {**dict(slot_cls.reads), **dict(slot_cls.writes)}
    chans = {p: ports.get(p, ch) for p, ch in declared.items()}
    label = _label(ref)
    gate_chans = [ch for ch in chans.values() if is_gate_channel(ch)]
    if not gate:
        if gate_chans:
            raise PlanError(f"{label}: channel {gate_chans[0]!r} belongs to the gate "
                            f"pipeline (BatchSpec.max_gates); a track pipeline has none")
        return Layout(E, n_max, dev, True)
    plugin = [ch for ch in chans.values() if is_plugin_channel(ch)]
    if gate_chans and plugin:
        raise PlanError(f"{label}: plugin channel {plugin[0]!r} is a centerline channel; "
                        f"gate plugin channels (gates.<pkg>.<name>) arrive with roadmap "
                        f"PR13")
    cl = [ch for ch in chans.values() if ch in CHANNELS and ch.startswith("cl.")]
    if ref.kind == "elevation":
        for p, ch in chans.items():
            if ch in CHANNELS and ch.startswith("cl."):
                raise PlanError(f"{label}: port {p!r} reads {ch!r}, which has no gate "
                                f"counterpart (the default gate rewiring covers cl.xy, "
                                f"cl.count, cl.arclen2, cl.length2 and cl.z); wire it "
                                f"with StageRef(ports=...)")
    if cl and any(ch.startswith("gates.") for ch in gate_chans):
        raise PlanError(f"{label} has centerline (cl.*) and gate (gates.*) ports after "
                        f"rewiring; a stage runs on one point domain")
    stride = n_max if cl else int(batch.max_gates)
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
    from ..stages.track import ElevationFlat, _GeneratorSlot  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    batch = spec.batch
    gate = batch.max_gates is not None
    slots = []
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
        ports = _effective_ports(cls, ref, gate)
        slot_layout = _domain_layout(cls, ref, ports, gate=gate, batch=batch,
                                     generator=None)
        params = _coerce(cls.Params, ref.params, what=_label(ref), label="params")
        params.validate(slot_layout)
        _check_values(params)
        if cls is ElevationFlat and float(params.z_base) != 0.0 \
                and not str(ports.get("z", "cl.z")).startswith("gates."):
            raise PlanError(f"{_label(ref)}: z_base={params.z_base!r} sets gate altitudes "
                            f"(gate pipelines); a track's flat altitude is lift.flat's "
                            f"z_base (presets.track(z_base=...))")
        slot = _Slot(i, sid, ref, cls, None, params, {}, {}, builtin, ports=ports,
                     layout=slot_layout)
        if "anchors.xy" in {ports.get(p, ch) for p, ch in dict(cls.writes).items()}:
            slot.capacity = _check_capacity(cls, cls().capacity(_static_view(params),
                                                                slot_layout))
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
                raise PlanError(f"{_label(s.ref)} carries anchors.{s.ref.name} by anchor "
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
    or edited spec cannot silently discard what the generator provides)."""
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


def _capture_class(slots, binding):
    """The Program's capture class: the largest of the generator's and every stage's
    ``capture_class`` of its (static) params. BOUNDED is refused (roadmap PR12)."""
    from ..plugin.params import _static_view  # noqa: PLC0415
    from ..plugin.stage import CaptureClass  # noqa: PLC0415

    cap = CaptureClass.STATIC if binding.capturable else CaptureClass.HOST
    for s in slots:
        if s.generator is not None:
            continue
        got = s.cls.capture_class(_static_view(s.params))
        if not isinstance(got, CaptureClass):
            raise TypeError(f"{s.cls.__name__}.capture_class() must return a "
                            f"CaptureClass, got {got!r}")
        if got == CaptureClass.BOUNDED:
            raise PlanError(f"{_label(s.ref)}: {s.cls.__name__}.capture_class() returned "
                            f"BOUNDED: arrives with roadmap PR12")
        cap = max(cap, got)
    return cap


def _missing_hint(ch: str, s: _Slot, slots, gate: bool) -> str:
    """A hint for the common gate-pipeline mistakes behind a read nobody wrote."""
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


def _plan_channels(slots, gate: bool = False) -> "dict[str, list[str]]":
    """Resolve the ports and walk the channel versions; returns ``channel -> [writer
    ids]`` (the seed's writer is the prologue)."""
    from ..plugin.stage import _rewrite_problems, _write_rule_problem  # noqa: PLC0415
    from ..stages import CHANNELS, OPEN_KINDS, SEED  # noqa: PLC0415
    from ..stages.channels import PER_CLASS, is_plugin_channel  # noqa: PLC0415

    def known(ch) -> bool:
        return ch in CHANNELS or is_plugin_channel(ch)

    versions: dict = {SEED: [_PROLOGUE]}
    for s in slots:
        ref, cls = s.ref, s.cls
        ports = s.ports
        decl_reads, decl_writes = dict(cls.reads), dict(cls.writes)
        unknown = sorted(p for p in ref.ports if p not in decl_reads and p not in decl_writes)
        if unknown:
            raise PlanError(f"{_label(ref)}: ports {unknown} are not ports of the stage; "
                            f"read ports: {sorted(decl_reads)}, write ports: "
                            f"{sorted(decl_writes)}")
        for p, ch in ports.items():
            if not known(ch):
                raise PlanError(f"{_label(ref)}: port {p!r} is rewired to unknown channel "
                                f"{ch!r}; channels: {sorted(CHANNELS)} and plugin "
                                f"channels 'cl.<package>.<name>'")
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
                raise PlanError(
                    f"{_label(ref)} writes {ch!r}, already written by stage "
                    f"{versions[ch][-1]!r}: a second writer must also read the channel "
                    f"(an in-place rewrite)")
        for ch in writes.values():
            versions.setdefault(ch, []).append(s.id)
        s.reads, s.writes = reads, writes
    return versions


# ---------------------------------------------------------------------------------------
# The result kinds (Track / GateSequence)
# ---------------------------------------------------------------------------------------

def _check_exports(writers) -> None:
    """Every ``Track`` field has a writer: the ``Track`` reads them all."""
    from ..stages.track import TRACK_EXPORTS  # noqa: PLC0415

    missing = [f"Track.{field} ({' or '.join(chs)})"
               for field, chs in TRACK_EXPORTS.items()
               if not any(ch in writers for ch in chs)]
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


def _check_out(out, layout) -> None:
    """``out=``: every Track field an ``[n]`` contiguous array of the right dtype on the
    batch's device (``generation`` and ``fail_mask`` may be ``None``)."""
    E, S = int(layout.num_envs), int(layout.stride)
    want = {f: ("vec3f", E * S) for f in ("outer", "center", "inner", "tangent", "normal")}
    want.update(arclen=("float32", E * S), length=("float32", E), winding=("float32", E),
                valid=("int32", E), count=("int32", E), truncated=("int32", E),
                generation=("int32", E), fail_mask=("int32", E))
    _check_arrays(out, want, ("generation", "fail_mask"),
                  str(wp.get_device(layout.device)))


def _check_gate_out(out, layout) -> None:
    """``out=`` of a gate Program: every ``GateSequence`` field (``generation`` and
    ``fail_mask`` may be ``None``)."""
    E, G = int(layout.num_envs), int(layout.stride)
    want = {f: ("vec3f", E * G) for f in ("position", "tangent", "forward", "left",
                                          "right")}
    want.update(orientation=("quatf", E * G), half_size=("float32", E * G),
                valid=("int32", E), count=("int32", E), generation=("int32", E),
                fail_mask=("int32", E))
    _check_arrays(out, want, ("generation", "fail_mask"),
                  str(wp.get_device(layout.device)))


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


def _result_kind(gate: bool) -> _ResultKind:
    from ..stages import commit as _commit_mod  # noqa: PLC0415
    from ..stages import gates as _gates_mod  # noqa: PLC0415
    from ..stages import track as _track_mod  # noqa: PLC0415

    if gate:
        return _ResultKind(
            _gates_mod.GATE_EXPORTS, "gates.valid", _check_gate_exports,
            _check_gate_fail_start, _check_gate_out, _gates_mod._alloc_gate_sequence,
            _commit_mod._GateCommit,
            (*_commit_mod.GATE_POINT_FIELDS, *_commit_mod.GATE_ENV_FIELDS))
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


class _StagePlan:
    """What a stage's :class:`~track_gen.plugin.StageBuilder` sees of the plan."""

    def __init__(self, plan: _Plan, slot: _Slot) -> None:
        self._plan = plan
        self._slot = slot
        self.reads = types.MappingProxyType(dict(slot.reads))
        self.writes = types.MappingProxyType(dict(slot.writes))
        cls = slot.cls
        self._declared = {**dict(getattr(cls, "reads", {})),
                          **dict(getattr(cls, "writes", {}))}
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

    def read(self, port: str):
        return self._visible[self.reads[port]]

    def write(self, port: str):
        plan, slot = self._plan, self._slot
        ch = self.writes[port]
        bound = plan.bound.get(ch)
        if bound is not None and plan.final.get(ch) == slot.id:
            arr = bound
        else:
            dtype, per = self.channel(ch)
            arr = wp.zeros(plan.length(per), dtype=dtype, device=plan.device)
        plan.add(ch, slot.index, slot.id, arr)
        return arr

    def fail_bit(self, name: str) -> int:
        return self._plan.fail_bit(self._slot, name)

    def reserve_fail_bits(self, cls, table) -> None:
        self._plan.reserve_fail_bits(self._slot, cls, table)


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
    capture_class = _capture_class(slots, binding)
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
    bound = {}
    for field, ch in sources.items():
        arr = getattr(result, field, None)
        if ch is not None and arr is not None:
            bound[ch] = arr
    seed = wp.empty(E, dtype=wp.int32, device=dev)
    regen_mask = wp.zeros(E, dtype=wp.int32, device=dev)
    plan = _Plan(layout, raw, dev, writers, bound, n_max=int(batch.n_max),
                 max_gates=batch.max_gates, anchors=A)
    plan.add("seed", -1, _PROLOGUE, seed)

    # -- 7: build every stage, in order ------------------------------------------------
    enqueues, pairs, scratch, salts = [], [], {}, {}
    inputs = {"regen_mask": regen_mask}
    stage_params = {}
    gen_scratch = None
    for s in slots:
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
                         plan=plan.for_stage(s))
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
        enqueues.append(enqueue)
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

    # -- 8: the commit (every result field has a source; else the result is not ours) --
    commit = None
    fields = kind.fields
    if all(f in sources for f in fields):
        # The result's arrays; fail_mask: the final valid.fail version (the result's own
        # unless out= brought none).
        targets = {f: getattr(result, f) for f in fields if f != "fail_mask"}
        targets["fail_mask"] = plan.current[FAIL]
        commit = kind.commit_cls(
            targets=targets, E=E, S=S, device=dev, dummy_mask=regen_mask,
            poison=batch.on_invalid == "poison", derive_valid=sources["valid"] is None)
    if hasattr(result, "fail_bits"):
        result.fail_bits = types.MappingProxyType(dict(plan.fail_bits))

    # -- 9: the Program ----------------------------------------------------------------
    if _GENERATOR in stage_params:
        gen_params = stage_params.pop(_GENERATOR)
    else:
        gen_params = param_table(binding, default_stream=stream)[_GENERATOR]
    table = _new_param_table({_GENERATOR: gen_params, **stage_params})
    resolved = PipelineSpec(batch=batch, stages=tuple(
        dataclasses.replace(s.ref, params=s.params) for s in slots))
    program = Program(
        _token=_PROGRAM_TOKEN, spec=resolved, layout=layout, binding=binding,
        result=result, seeds=seed_state, seed=seed, inputs=inputs, params=table,
        pairs=pairs, enqueues=enqueues, versions=plan.versions, scratch=scratch,
        stage_ids=[s.id for s in slots],
        kinds={s.id: (s.ref.kind, s.ref.name) for s in slots}, gen_scratch=gen_scratch,
        stream=stream, owner=owner, commit=commit, fail_bits=plan.fail_bits,
        fail_owner=plan.fail_owner, capture_class=capture_class,
        on_invalid=batch.on_invalid, writers=writers,
        ports={s.id: (dict(s.reads), dict(s.writes)) for s in slots})
    if prepare:
        program.prepare()
    return program
