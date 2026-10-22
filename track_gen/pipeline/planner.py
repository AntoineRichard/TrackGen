"""The planner: :func:`build` turns a :class:`PipelineSpec` into a :class:`Program`.

The spec rules are checked host-side before anything is allocated (:class:`PlanError`,
a ``ValueError``; the stages' params keep their own ``ValueError`` / ``TypeError``, and
an unknown generator raises the registry's own ``ValueError``); then the planner
allocates every channel version once, namespaces each stage's scratch, runs every
``Stage.build`` once and binds the result. The build-time rules (each stage's write
ports, its fail bits, a plugin channel's dtype) are checked right after the stage's
``build`` (a ``PlanError`` raised after allocation).

Stages of the kinds open to plugins (``relax``, ``width``, ``elevation``, ``validity``)
are resolved through the plugin registry; the others (``resample``, ``frames``,
``offset``, ``lift``) from TrackGen's built-in catalogue.
"""
from __future__ import annotations

import dataclasses
import numbers
import types
import warp as wp

from .spec import BatchSpec, PipelineSpec, StageRef

__all__ = ["PlanError", "build"]

_GENERATOR = "generator"
_PROLOGUE = "<prologue>"
_TRACK_DTYPES = {"vec3f": wp.vec3f, "float32": wp.float32, "int32": wp.int32}
_ON_INVALID = ("keep", "poison")


class PlanError(ValueError):
    """A pipeline spec the planner refuses; raised before anything is allocated, except
    for the build-time rules checked right after a stage's ``build`` (its declared write
    ports, its fail bits, a plugin channel's dtype)."""


def build(spec: PipelineSpec, *, registry=None, seeds=None, out=None, stream=None,
          prepare: bool = True):
    """Plan, allocate and bind ``spec``; return its :class:`~track_gen.pipeline.Program`.

    The generator (the first stage, ``kind="generator"``) is resolved through
    ``registry`` exactly as ``TrackGenerator`` resolves ``config.generator``, with the
    config :func:`track_gen.pipeline.legacy.generator_config` derives from the spec.
    The ``relax`` / ``width`` / ``elevation`` / ``validity`` stages are resolved through
    ``registry`` too (the class is captured at build: later registry edits never change
    a built Program); the others come from TrackGen's built-in catalogue.

    Args:
        spec: The :class:`PipelineSpec`.
        registry: The plugin registry (default ``track_gen.plugin.registry``).
        seeds: A ``SeedState``, or anything ``SeedState`` accepts (``None``, an int, an
            ``[E]`` int32 ``wp.array``, a ``PerEnvSeededRNG``).
        out: Optional pre-allocated ``Track`` to write into (every field checked;
            ``generation`` and ``fail_mask`` may be ``None``).
        stream: Optional default ``wp.Stream`` of the Program.
        prepare: Run the warm-up generation (:meth:`Program.prepare`) before returning
            (otherwise the first :meth:`Program.run` does).

    Raises:
        PlanError: an invalid spec (see :doc:`/how-it-works/pipeline-preview`), incl.
            an unknown relax / width / elevation / validity name (the registry's
            message).
        ValueError: invalid params (the stage's own message), an unknown generator (the
            registry's message), an ``out=`` or ``seeds`` mismatch.
        TypeError: params or ``seeds`` of the wrong type.
        track_gen.plugin.PluginLoadError: a stage named explicitly is provided by a
            quarantined plugin.
        track_gen.exec.CaptureError: called inside a capture (before allocating).
    """
    from .._src.generator_binding import bind_generator  # noqa: PLC0415
    from ..plugin import registry as _default_registry  # noqa: PLC0415
    from . import legacy  # noqa: PLC0415

    _check_spec_shape(spec)
    _check_seeds(seeds)
    reg = _default_registry if registry is None else registry
    binding = bind_generator(_GENERATOR, legacy.generator_config(spec), registry=reg)
    return _build(spec, binding=binding, seeds=seeds, out=out, stream=stream,
                  prepare=prepare, registry=reg)


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
    gens = [ref.slot_id for ref in spec.stages if ref.kind == _GENERATOR]
    if spec.stages[0].kind != _GENERATOR:
        raise PlanError(f"the first stage must be the generator (kind='generator'), "
                        f"got {spec.stages[0].kind!r}")
    if len(gens) != 1:
        raise PlanError(f"exactly one generator stage, got {len(gens)}: {gens}")
    if spec.stages[0].slot_id != _GENERATOR:
        raise PlanError(f"the generator stage's id must be 'generator', got "
                        f"{spec.stages[0].slot_id!r}")
    # capacity: the generator's raw default only (Stage.capacity() overrides it).
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
    reads: dict                 # port -> channel (after StageRef.ports)
    writes: dict
    builtin: bool = True        # shipped with track_gen (the registry's own record)


def _resolve(spec: PipelineSpec, binding, layout, registry=None) -> "list[_Slot]":
    from .. import stages as _stages  # noqa: PLC0415
    from ..plugin import registry as _default_registry  # noqa: PLC0415
    from ..plugin.params import _check_values, _coerce  # noqa: PLC0415
    from ..stages.track import _GeneratorSlot  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    slots = []
    for i, ref in enumerate(spec.stages):
        sid = ref.slot_id
        if ref.kind == _GENERATOR:
            if binding.name != ref.name:
                raise PlanError(f"{_label(ref)}: the bound generator is "
                                f"{binding.name!r}")
            bl = binding.layout
            if (bl.num_envs, bl.stride, bl.device) != \
                    (layout.num_envs, layout.stride, layout.device):
                raise PlanError(f"{_label(ref)}: the generator was bound for {bl}, the "
                                f"batch is {layout}")
            slots.append(_Slot(i, sid, ref, _GeneratorSlot, _GeneratorSlot(binding),
                               binding.params, {}, {}))
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
            kinds = sorted({*(k for k in _stages.available() if k != _GENERATOR),
                            *_stages.OPEN_KINDS})
            if ref.kind not in kinds:
                raise PlanError(f"{_label(ref)}: unknown stage kind {ref.kind!r}; kinds: "
                                f"{['generator', *kinds]}")
            raise PlanError(
                f"{_label(ref)}: unknown {ref.kind} stage {ref.name!r}; built-in "
                f"{ref.kind} stages: {_stages.available(ref.kind)} (kind {ref.kind!r} is "
                f"not open to plugins)")
        params = _coerce(cls.Params, ref.params, what=_label(ref), label="params")
        params.validate(layout)
        _check_values(params)
        slots.append(_Slot(i, sid, ref, cls, None, params, {}, {}, builtin))
    # Interim rule (until the kernels read cl.half_width, roadmap PR13): every stage
    # that bakes half_width as a host scalar agrees on it.
    hw = {s.id: getattr(s.params, "half_width") for s in slots
          if s.ref.kind != _GENERATOR and s.params is not None
          and "half_width" in getattr(type(s.params), "__dataclass_fields__", {})}
    if len({float(v) for v in hw.values()}) > 1:
        raise PlanError(f"the stages disagree on half_width: {hw} (one value until "
                        f"roadmap PR13)")
    return slots


def _capture_class(slots, binding):
    """The Program's capture class: the largest of the generator's and every stage's
    ``capture_class`` of its (static) params. BOUNDED is refused (roadmap PR12)."""
    from ..plugin.params import _static_view  # noqa: PLC0415
    from ..plugin.stage import CaptureClass  # noqa: PLC0415

    cap = CaptureClass.STATIC if binding.capturable else CaptureClass.HOST
    for s in slots:
        if s.ref.kind == _GENERATOR:
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


def _plan_channels(slots) -> "dict[str, list[str]]":
    """Resolve the ports and walk the channel versions; returns ``channel -> [writer
    ids]`` (the seed's writer is the prologue)."""
    from ..plugin.stage import _rewrite_problems, _write_rule_problem  # noqa: PLC0415
    from ..stages import CHANNELS, OPEN_KINDS, SEED  # noqa: PLC0415
    from ..stages.channels import is_plugin_channel  # noqa: PLC0415

    def known(ch) -> bool:
        return ch in CHANNELS or is_plugin_channel(ch)

    versions: dict = {SEED: [_PROLOGUE]}
    for s in slots:
        ref, cls = s.ref, s.cls
        decl_reads, decl_writes = dict(cls.reads), dict(cls.writes)
        unknown = sorted(p for p in ref.ports if p not in decl_reads and p not in decl_writes)
        if unknown:
            raise PlanError(f"{_label(ref)}: ports {unknown} are not ports of the stage; "
                            f"read ports: {sorted(decl_reads)}, write ports: "
                            f"{sorted(decl_writes)}")
        for p, ch in ref.ports.items():
            if not known(ch):
                raise PlanError(f"{_label(ref)}: port {p!r} is rewired to unknown channel "
                                f"{ch!r}; channels: {sorted(CHANNELS)} and plugin "
                                f"channels 'cl.<package>.<name>'")
            declared = decl_reads.get(p, decl_writes.get(p))
            if declared not in CHANNELS or ch not in CHANNELS:
                continue                  # a plugin channel: its writer fixes the dtype
            a, b = CHANNELS[declared], CHANNELS[ch]
            if (a.dtype, a.per) != (b.dtype, b.per):
                raise PlanError(f"{_label(ref)}: port {p!r} ({declared!r}, "
                                f"{a.dtype.__name__} per={a.per!r}) cannot be rewired to "
                                f"{ch!r} ({b.dtype.__name__} per={b.per!r})")
        reads = {p: ref.ports.get(p, ch) for p, ch in decl_reads.items()}
        writes = {p: ref.ports.get(p, ch) for p, ch in decl_writes.items()}
        if ref.ports:
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
                                f"stage writes it")
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


def _check_exports(writers) -> None:
    """Every ``Track`` field has a writer: the ``Track`` reads them all."""
    from ..stages.track import TRACK_EXPORTS  # noqa: PLC0415

    missing = [f"Track.{field} ({' or '.join(chs)})"
               for field, chs in TRACK_EXPORTS.items()
               if not any(ch in writers for ch in chs)]
    if missing:
        raise PlanError(f"no stage writes {', '.join(missing)}: every Track field needs "
                        f"a writer")


def _export_sources(writers) -> "dict[str, str | None]":
    """``Track field -> the channel whose FINAL version is the field's array`` (the
    first written of its candidates), or ``None`` for a field the commit writes:
    ``valid`` when ``validity.default`` (``track.valid``) is not also the last writer of
    ``valid.fail`` (a plugin check ran after it, or none ran). A field without any
    writer is absent (only reachable when ``_check_exports`` is lifted)."""
    from ..stages.channels import FAIL  # noqa: PLC0415
    from ..stages.track import TRACK_EXPORTS  # noqa: PLC0415

    out: dict = {}
    for field, chs in TRACK_EXPORTS.items():
        if field == "valid":
            if "track.valid" in writers and FAIL in writers and \
                    writers["track.valid"][-1] == writers[FAIL][-1]:
                out[field] = "track.valid"
            elif FAIL in writers:
                out[field] = None                     # derived: fail_mask == 0
            elif "track.valid" in writers:
                out[field] = "track.valid"
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


def _check_out(out, layout) -> None:
    """``out=``: every Track field an ``[n]`` contiguous array of the right dtype on the
    batch's device (``generation`` and ``fail_mask`` may be ``None``)."""
    E, S = int(layout.num_envs), int(layout.stride)
    want = {f: ("vec3f", E * S) for f in ("outer", "center", "inner", "tangent", "normal")}
    want.update(arclen=("float32", E * S), length=("float32", E), winding=("float32", E),
                valid=("int32", E), count=("int32", E), truncated=("int32", E),
                generation=("int32", E), fail_mask=("int32", E))
    dev = str(wp.get_device(layout.device))
    for field, (dtype, n) in want.items():
        arr = getattr(out, field, None)
        if arr is None and field in ("generation", "fail_mask"):
            continue
        if not isinstance(arr, wp.array):
            raise ValueError(f"out.{field}: expected a wp.array, got {type(arr).__name__}")
        got = (getattr(arr.dtype, "__name__", str(arr.dtype)), tuple(arr.shape),
               str(arr.device))
        if arr.dtype is not _TRACK_DTYPES[dtype] or tuple(arr.shape) != (n,) \
                or str(arr.device) != dev or not arr.is_contiguous:
            raise ValueError(f"out.{field}: expected a contiguous [{n}] {dtype} array on "
                             f"{dev}, got {got}")


# ---------------------------------------------------------------------------------------
# The plan (allocation)
# ---------------------------------------------------------------------------------------

class _Plan:
    """Channel versions, exports, plugin channels and fail bits, shared by every
    stage's builder."""

    def __init__(self, layout, raw: int, device, writers, bound) -> None:
        self.layout = layout
        self.raw = int(raw)
        self.device = device
        self.final = {ch: ids[-1] for ch, ids in writers.items()}
        self.bound = dict(bound)              # exported channel -> Track array
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
            return E * int(self.layout.stride)
        raise ValueError(f"unknown per={per!r}; expected 'env', 'raw' or 'cl'")

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
    """:func:`build` with the generator already bound (``TrackGenerator`` passes its
    own binding and run config)."""
    from .._src.generator_binding import _StageGenerator, param_table  # noqa: PLC0415
    from ..exec.runner import _check_default_stream, ensure_not_capturing  # noqa: PLC0415
    from ..plugin.params import _fields, _new_param_table, _new_stage_params, \
        _static_view  # noqa: PLC0415
    from ..plugin.stage import Layout, _close_builder, _new_builder  # noqa: PLC0415
    from ..stages import commit as _commit_mod  # noqa: PLC0415
    from ..stages import track as _track_mod  # noqa: PLC0415
    from ..stages.channels import FAIL  # noqa: PLC0415
    from .program import _PROGRAM_TOKEN, Program  # noqa: PLC0415

    # -- 1-5: host-side checks, nothing allocated --------------------------------------
    _check_spec_shape(spec)
    batch = spec.batch
    E, S = int(batch.num_envs), int(batch.n_max)
    dev = wp.get_device(batch.device)
    layout = Layout(E, S, str(dev), True)
    slots = _resolve(spec, binding, layout, registry)
    capture_class = _capture_class(slots, binding)
    writers = _plan_channels(slots)
    _check_exports(writers)
    _check_fail_start(slots, writers)
    _check_flat_lift(slots)
    if out is not None:
        _check_out(out, layout)
    adapter = binding.pipeline_spec if isinstance(binding.pipeline_spec, _StageGenerator) \
        else None
    raw = int(adapter.raw_points) if adapter is not None else int(binding.run_config.num_points)
    _check_default_stream(dev, stream)

    # -- 6: shared allocations (refuse a capture first; the seeds' own checks next) ---
    ensure_not_capturing(dev, default_stream=stream, label=binding.name)
    seed_state = _seed_state(seeds, E, dev, stream)
    track = out if out is not None else _track_mod._alloc_track(layout)
    # The final version of each exported channel IS the Track's array (no copy).
    sources = _export_sources(writers)
    bound = {}
    for field, ch in sources.items():
        arr = getattr(track, field, None)
        if ch is not None and arr is not None:
            bound[ch] = arr
    seed = wp.empty(E, dtype=wp.int32, device=dev)
    regen_mask = wp.zeros(E, dtype=wp.int32, device=dev)
    plan = _Plan(layout, raw, dev, writers, bound)
    plan.add("seed", -1, _PROLOGUE, seed)

    # -- 7: build every stage, in order ------------------------------------------------
    enqueues, pairs, scratch, salts = [], [], {}, {}
    inputs = {"regen_mask": regen_mask}
    stage_params = {}
    gen_scratch = None
    for s in slots:
        is_gen = s.ref.kind == _GENERATOR
        params = None if is_gen else s.params
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
        instance = s.instance if is_gen else s.cls()
        b = _new_builder(stage=instance, params=view, layout=layout, scale=batch.scale,
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
        if is_gen:
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

    # -- 8: the commit (every Track field has a source; else the Track is not ours) ----
    commit = None
    fields = (*_commit_mod.POINT_FIELDS, *_commit_mod.ENV_FIELDS)
    if all(f in sources for f in fields):
        # The Track's arrays; fail_mask: the final valid.fail version (the Track's own
        # unless out= brought none).
        targets = {f: getattr(track, f) for f in fields if f != "fail_mask"}
        targets["fail_mask"] = plan.current[FAIL]
        commit = _commit_mod._Commit(
            targets=targets, E=E, S=S, device=dev, dummy_mask=regen_mask,
            poison=batch.on_invalid == "poison", derive_valid=sources["valid"] is None)
    if hasattr(track, "fail_bits"):
        track.fail_bits = types.MappingProxyType(dict(plan.fail_bits))

    # -- 9: the Program ----------------------------------------------------------------
    gen_params = param_table(binding, default_stream=stream)[_GENERATOR]
    table = _new_param_table({_GENERATOR: gen_params, **stage_params})
    resolved = PipelineSpec(batch=batch, stages=tuple(
        dataclasses.replace(s.ref, params=s.params) for s in slots))
    program = Program(
        _token=_PROGRAM_TOKEN, spec=resolved, layout=layout, binding=binding,
        result=track, seeds=seed_state, seed=seed, inputs=inputs, params=table,
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
