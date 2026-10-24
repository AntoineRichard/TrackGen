"""The pipeline as data: :class:`StageRef`, :class:`BatchSpec`, :class:`PipelineSpec`.

Pure data, no registry and no Warp: every operation here is a host-side value
transformation, and everything beyond the shape of the data is validated by
:func:`track_gen.pipeline.build`.
"""
from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Mapping as _MappingABC
from types import MappingProxyType
from typing import Any, Mapping

__all__ = ["BatchSpec", "PipelineSpec", "StageRef"]

FORMAT = "track_gen.pipeline/2"
#: The stage kinds that write another stage's dynamic params (roadmap PR13b): listed
#: before their target; only the generator's may precede the generator. Kept here (this
#: module imports nothing from track_gen) and equal to
#: ``track_gen.stages.PARAM_WRITER_KINDS`` (a test pins it).
PARAM_WRITER_KINDS = ("randomize",)
#: The 0.2.0.dev3 format (``from_dict`` migrates it: the width stage moved before relax
#: and the half_width params of relax.xpbd / offset.default / validity.default became
#: the width stage's).
_FORMAT_V1 = "track_gen.pipeline/1"
# The built-ins that lost their half_width field in /2, and the dev3 default value.
_V1_HALF_WIDTH = (("relax", "xpbd"), ("offset", "default"), ("validity", "default"))
_V1_DEFAULT_HALF_WIDTH = 0.1


def _frozen_map(value, what: str, *, values=None) -> "MappingProxyType":
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, _MappingABC):
        raise TypeError(f"StageRef.{what} must be a mapping, got {type(value).__name__}")
    out = {}
    for k, v in value.items():
        if not isinstance(k, str) or not k:
            raise TypeError(f"StageRef.{what} keys must be non-empty strings, got {k!r}")
        if values is not None:
            v = values(k, v)
        out[k] = v
    return MappingProxyType(out)


def _canonical_param(k, v):
    """A params value with every list (at any depth inside lists / tuples) as a tuple:
    both are JSON arrays and Params coercion converts by annotation, so a spec read back
    from JSON (``from_dict``) equals the spec written (``to_dict``)."""
    if isinstance(v, (list, tuple)) and not hasattr(v, "_fields"):
        return tuple(_canonical_param(k, x) for x in v)
    return v


def _thaw(value):
    """``value`` with every ``MappingProxyType`` (at any depth inside dict / list /
    tuple values) turned into a dict: what ``StageRef.__reduce__`` pickles."""
    if isinstance(value, (MappingProxyType, dict)):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_thaw(v) for v in value]
    if type(value) is tuple:
        return tuple(_thaw(v) for v in value)
    return value


def _port_value(k, v):
    if not isinstance(v, str) or not v:
        raise TypeError(f"StageRef.ports[{k!r}] must be a channel name, got {v!r}")
    return v


def _capacity_value(k, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"StageRef.capacity[{k!r}] must be an int, got {v!r}")
    return int(v)


@dataclasses.dataclass(frozen=True)
class StageRef:
    """One stage slot of a :class:`PipelineSpec`.

    Attributes:
        kind: The stage kind (``"generator"``, ``"gate_generator"``, ``"resample"``,
            ``"relax"``, ``"order"``, ...).
        name: The stage name within its kind (``"bezier"``, ``"xpbd"``, ...).
        params: The stage's Params instance, a mapping of its fields (frozen here,
            coerced and validated by ``build``), or ``None`` for its defaults (the
            generator: ``Params.from_legacy`` of a default config).
        id: The slot id (default: ``kind``). Two stages of one kind need explicit ids.
            Must not contain ``"/"`` (it namespaces the stage's scratch and salts).
        ports: ``local port -> channel`` rewiring, applied to the stage's read and
            write ports.
        capacity: The (centerline) generator's ``{"raw": num_points}``: the DEFAULT raw
            point count (``num_points``, 256 when absent), which a Stage generator's
            own ``Stage.capacity()`` overrides. ``build`` rejects any other key, and any
            capacity on another stage (a gate generator's capacity is its
            ``Stage.capacity()``).
    """

    kind: str
    name: str
    params: Any = None
    id: "str | None" = None
    ports: Mapping[str, str] = dataclasses.field(default_factory=dict)
    capacity: Mapping[str, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for attr in ("kind", "name"):
            v = getattr(self, attr)
            if not isinstance(v, str) or not v:
                raise TypeError(f"StageRef.{attr} must be a non-empty string, got {v!r}")
        if self.id is not None and (not isinstance(self.id, str) or not self.id
                                    or "/" in self.id):
            raise ValueError(f"StageRef.id must be None or a non-empty string without "
                             f"'/', got {self.id!r}")
        params = self.params
        if isinstance(params, _MappingABC):
            object.__setattr__(self, "params", _frozen_map(params, "params",
                                                           values=_canonical_param))
        object.__setattr__(self, "ports", _frozen_map(self.ports, "ports",
                                                      values=_port_value))
        object.__setattr__(self, "capacity", _frozen_map(self.capacity, "capacity",
                                                         values=_capacity_value))

    def __reduce__(self):
        # The frozen mappings (MappingProxyType) do not pickle: rebuild the ref through
        # the constructor from plain dicts (it freezes the top level again; a nested
        # mappingproxy, e.g. another ref's params, comes back as a dict).
        params = _thaw(self.params) if isinstance(self.params, _MappingABC) \
            else self.params
        return (type(self), (self.kind, self.name, params, self.id, dict(self.ports),
                             dict(self.capacity)))

    @property
    def slot_id(self) -> str:
        """``id`` or, when ``None``, ``kind``."""
        return self.id or self.kind

    def __repr__(self) -> str:
        parts = [repr(self.kind), repr(self.name)]
        if self.params is not None:
            p = dict(self.params) if isinstance(self.params, _MappingABC) else self.params
            parts.append(f"params={p!r}")
        if self.id is not None:
            parts.append(f"id={self.id!r}")
        if self.ports:
            parts.append(f"ports={dict(self.ports)!r}")
        if self.capacity:
            parts.append(f"capacity={dict(self.capacity)!r}")
        return f"StageRef({', '.join(parts)})"

    def _to_dict(self) -> dict:
        from ..plugin.params import _jsonable  # noqa: PLC0415

        params = self.params
        if params is None:
            data = None
        elif isinstance(params, _MappingABC):
            data = {str(k): _jsonable(v) for k, v in params.items()}
        elif dataclasses.is_dataclass(params) and not isinstance(params, type):
            data = {f.name: _jsonable(getattr(params, f.name))
                    for f in dataclasses.fields(params)}
        else:
            raise TypeError(f"stage {self.slot_id!r}: params of type "
                            f"{type(params).__name__} cannot be serialised")
        return {"kind": self.kind, "name": self.name, "id": self.id, "params": data,
                "ports": dict(self.ports), "capacity": dict(self.capacity)}


@dataclasses.dataclass(frozen=True)
class BatchSpec:
    """The batch every stage shares.

    Attributes:
        num_envs: ``E``.
        device: The Warp device string.
        n_max: Per-env point capacity of the centerline channels (``N_max``); in a gate
            pipeline, read only by a centerline sub-pipeline (``stations``).
        closed: Closed loops: a bool, ``True`` (``closed=False`` is refused by ``build``
            in 0.x, naming the stages that support closed loops only; every Layout the
            planner builds carries it).
        scale: ``TrackGenConfig.scale``, handed to the stages as ``StageBuilder.scale``.
        on_invalid: What an invalid env (``fail_mask != 0``) leaves in the ``Track``:
            ``"keep"`` (default: its computed geometry, ``valid = 0``) or
            ``"poison"`` (``count = 0``, NaN geometry, ``length = winding = 0``, so
            every utility's degenerate ``count < 3`` path fires; ``fail_mask`` and
            ``truncated`` are kept). A gate pipeline: ``count = 0``, every per-gate
            field NaN, ``valid = 0`` (``fail_mask`` kept).
        max_gates: ``None`` for a track pipeline (the result is a ``Track``); an int
            ``>= 2`` makes it a gate pipeline (e.g.
            :func:`~track_gen.pipeline.presets.gates`), whose result is a ``GateSequence``
            of ``max_gates`` slots per env.
    """

    num_envs: int
    device: str
    n_max: int
    closed: bool = True
    scale: float = 1.0
    on_invalid: str = "keep"
    max_gates: "int | None" = None

    def _to_dict(self) -> dict:
        return {"num_envs": int(self.num_envs), "device": str(self.device),
                "n_max": int(self.n_max), "closed": bool(self.closed),
                "scale": float(self.scale), "on_invalid": str(self.on_invalid),
                "max_gates": None if self.max_gates is None else int(self.max_gates)}


@dataclasses.dataclass(frozen=True)
class PipelineSpec:
    """A pipeline: the :class:`BatchSpec` and the ordered stage slots.

    The editing operations return a NEW spec; they only check that an anchor id
    exists (``ValueError`` listing the ids). Everything else is validated by
    :func:`track_gen.pipeline.build`.

    Attributes:
        batch: The :class:`BatchSpec`.
        stages: The ordered :class:`StageRef` slots: the generator first (preceded only
            by its own randomizers, ``randomize.uniform`` stages drawing its dynamic
            params).
        require_caller_capturable: Refuse, at build (``PlanError`` naming them), any
            ``CaptureClass.HOST`` stage: the Program is then recordable into a caller's
            capture as a whole. Default ``False`` (HOST stages run eagerly between the
            graph launches of the rest; a caller capture of such a Program raises
            ``NotCapturableError``).
    """

    batch: BatchSpec
    stages: "tuple[StageRef, ...]"
    require_caller_capturable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.batch, BatchSpec):
            raise TypeError(f"PipelineSpec.batch must be a BatchSpec, got "
                            f"{type(self.batch).__name__}")
        if not isinstance(self.require_caller_capturable, bool):
            raise TypeError(f"PipelineSpec.require_caller_capturable must be a bool, got "
                            f"{type(self.require_caller_capturable).__name__}")
        stages = tuple(self.stages)
        for ref in stages:
            if not isinstance(ref, StageRef):
                raise TypeError(f"PipelineSpec.stages must hold StageRef, got "
                                f"{type(ref).__name__}")
        object.__setattr__(self, "stages", stages)

    @property
    def ids(self) -> "tuple[str, ...]":
        """The slot id of every stage, in order."""
        return tuple(ref.slot_id for ref in self.stages)

    def _generator_index(self) -> int:
        """The index of the generator slot: the first stage whose kind is not a
        param writer's (``randomize``; only the generator's randomizers may precede it);
        ``len(stages)`` when every stage is one."""
        for i, ref in enumerate(self.stages):
            if ref.kind not in PARAM_WRITER_KINDS:
                return i
        return len(self.stages)

    def _index(self, stage_id: str) -> int:
        ids = self.ids
        if stage_id not in ids:
            raise ValueError(f"no stage {stage_id!r} in the pipeline; stages: {list(ids)}")
        return ids.index(stage_id)

    def _with(self, stages) -> "PipelineSpec":
        return dataclasses.replace(self, stages=tuple(stages))

    def replace(self, stage_id: str, ref: StageRef) -> "PipelineSpec":
        """Put ``ref`` in slot ``stage_id``; ``ref`` keeps the slot's id when its own
        ``id`` is ``None``."""
        i = self._index(stage_id)
        if ref.id is None and ref.kind != stage_id:
            ref = dataclasses.replace(ref, id=stage_id)
        stages = list(self.stages)
        stages[i] = ref
        return self._with(stages)

    def insert_after(self, stage_id: str, ref: StageRef) -> "PipelineSpec":
        """Insert ``ref`` right after slot ``stage_id``."""
        i = self._index(stage_id)
        stages = list(self.stages)
        stages.insert(i + 1, ref)
        return self._with(stages)

    def insert_before(self, stage_id: str, ref: StageRef) -> "PipelineSpec":
        """Insert ``ref`` right before slot ``stage_id``."""
        i = self._index(stage_id)
        stages = list(self.stages)
        stages.insert(i, ref)
        return self._with(stages)

    def remove(self, stage_id: str) -> "PipelineSpec":
        """Drop slot ``stage_id``."""
        i = self._index(stage_id)
        stages = list(self.stages)
        del stages[i]
        return self._with(stages)

    def with_randomize(self, randomize) -> "PipelineSpec":
        """This spec with per-env uniform draws of DYNAMIC float32 params: each a
        ``randomize.uniform`` stage (id ``"randomize.<stage>.<field>"``) inserted right
        before the stage it draws.

        The helper behind ``presets.track(randomize=)`` / ``presets.gates(randomize=)``,
        for a spec built any other way, e.g. from a config::

            from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG,
                                   TrackGenConfig, TrackGenerator)
            from track_gen.pipeline import legacy

            E, d = 64, "cuda:0"
            rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=d)
            cfg = TrackGenConfig(num_envs=E, device=d, z_profile="uniform", z_max=0.6)
            spec = legacy.to_pipeline_spec(cfg).with_randomize(
                {"generator.rad": (0.1, 0.35), "elevation.z_max": (0.2, 0.6)})
            gen = TrackGenerator(spec, rng)
            # a GateGenConfig: its gate spec, then GateGenerator
            gcfg = GateGenConfig(num_envs=E, device=d, generator="polar")
            spec = legacy.to_gate_pipeline_spec(gcfg).with_randomize(
                {"generator.polar_radial_jitter": (0.2, 0.8)})
            gen = GateGenerator(spec, rng)

        A field the spec already randomizes is refused, not replaced: to change a draw
        range (e.g. at a curriculum phase), replace that randomizer,
        ``spec.replace("randomize.generator.rad", StageRef("randomize", "uniform",
        {"stage": "generator", "field": "rad", "lo": 0.1, "hi": 0.4}))``, or write its
        live ``lo`` / ``hi`` (``gen.params["randomize.generator.rad"].set("hi", 0.4)``).

        Args:
            randomize: ``{"<stage id>.<field>": (lo, hi)}`` (the key split at its LAST
                dot) or a sequence of ``StageRef("randomize", "uniform", {...})``;
                ``None`` returns this spec. The course size is ``"generator.size"`` (the
                track generators but ``external``, the bezier / hull gate generators) or
                ``"normalize.size"`` (the polar / voronoi / checkpoint gate pipelines),
                the bar width ``"gate_frames.gate_width"``.

        Raises:
            ValueError: a malformed key, a target stage the spec does not have (lists
                the stage ids), or a field (or a stage id) the spec already has a
                randomizer for.
            TypeError: a range that is not a ``(lo, hi)`` pair, or a sequence item
                that is not a ``StageRef("randomize", ...)``.

        The field's rules (dynamic, float32, the range inside its declared bounds, the
        cross-field rules) are checked by :func:`track_gen.pipeline.build`.
        """
        return _with_randomizers(self, randomize)

    def to_dict(self) -> dict:
        """The spec as JSON data (``"format": "track_gen.pipeline/2"``): Params become
        ``{field: value}`` (tuples as lists, enums as their values);
        ``"require_caller_capturable": true`` only when it is set."""
        out = {"format": FORMAT, "batch": self.batch._to_dict(),
               "stages": [ref._to_dict() for ref in self.stages]}
        if self.require_caller_capturable:
            out["require_caller_capturable"] = True
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> "PipelineSpec":
        """Read :meth:`to_dict` data back (params stay mappings; ``build`` coerces
        them). No registry is needed.

        ``track_gen.pipeline/1`` data (written by earlier releases) is migrated,
        with one ``DeprecationWarning``: the ``half_width`` params of ``relax.xpbd`` /
        ``offset.default`` / ``validity.default`` are dropped (each must equal the width
        stage's ``half_width``, else ``ValueError``), every width stage between the
        first relax and the first frames stage moves to just before that relax, and a
        ``relax.xpbd`` without a width stage before it (a ``relaxed_stations`` gate spec
        in format /1) gets ``width.constant`` with that half-width.
        With ``width.constant`` (or no width stage) the migrated spec builds the outputs
        of the /1 spec byte for byte; another width stage now runs before relax and is
        read per point by XPBD and the offset, so its outputs change (the warning says
        so). A width stage after the first frames stage of a spec whose ``relax.xpbd``
        has no width before it (an earlier width stage reading ``cl.kappa``) is refused with
        ``ValueError``: it cannot be both the width XPBD reads and a stage after
        frames."""
        if not isinstance(data, _MappingABC):
            raise TypeError(f"PipelineSpec.from_dict expects a mapping, got "
                            f"{type(data).__name__}")
        fmt = data.get("format")
        if fmt == _FORMAT_V1:
            data = _migrate_v1(data)
        elif fmt != FORMAT:
            raise ValueError(f"unsupported pipeline format {fmt!r}; expected {FORMAT!r}")
        b = _required(data, "pipeline data", ("batch", "stages"))["batch"]
        _required(b, "batch", ("num_envs", "device", "n_max"))
        batch = BatchSpec(num_envs=b["num_envs"], device=b["device"], n_max=b["n_max"],
                          closed=b.get("closed", True), scale=b.get("scale", 1.0),
                          on_invalid=b.get("on_invalid", "keep"),
                          max_gates=b.get("max_gates"))
        raw_stages = data["stages"]
        if isinstance(raw_stages, (str, bytes)) or not isinstance(raw_stages, (list, tuple)):
            raise ValueError(f"stages must be a list, got {type(raw_stages).__name__}")
        stages = []
        for i, s in enumerate(raw_stages):
            _required(s, f"stages[{i}]", ("kind", "name"))
            stages.append(StageRef(kind=s["kind"], name=s["name"], params=s.get("params"),
                                   id=s.get("id"), ports=s.get("ports") or {},
                                   capacity=s.get("capacity") or {}))
        return cls(batch=batch, stages=tuple(stages),
                   require_caller_capturable=data.get("require_caller_capturable", False))


def _randomizers(randomize) -> "list[StageRef]":
    """``randomize=`` as ``randomize.uniform`` StageRefs: a mapping ``{"<stage>.<field>":
    (lo, hi)}`` (the key split at its LAST dot: stage ids may contain dots, field names
    may not), or a sequence of ``StageRef("randomize", ...)``."""
    if randomize is None:
        return []
    if isinstance(randomize, _MappingABC):
        refs = []
        for key, rng in randomize.items():
            if not isinstance(key, str) or "." not in key.strip(".") or key.endswith("."):
                raise ValueError(f"randomize: key {key!r} must be '<stage id>.<field>' "
                                 f"(e.g. 'generator.amp')")
            stage, field = key.rsplit(".", 1)
            if isinstance(rng, (str, bytes)) or not isinstance(rng, (tuple, list)) or \
                    len(rng) != 2:
                raise TypeError(f"randomize[{key!r}] must be a (lo, hi) pair, got {rng!r}")
            lo, hi = rng
            refs.append(StageRef("randomize", "uniform",
                                 params={"stage": stage, "field": field, "lo": float(lo),
                                         "hi": float(hi)}, id=f"randomize.{key}"))
        return refs
    if isinstance(randomize, (str, bytes, StageRef)):
        raise TypeError(f"randomize must be a mapping {{'<stage>.<field>': (lo, hi)}} or a "
                        f"sequence of StageRef('randomize', ...), got {randomize!r}")
    refs = list(randomize)
    for ref in refs:
        if not isinstance(ref, StageRef) or ref.kind != "randomize":
            raise TypeError(f"randomize: expected StageRef('randomize', ...), got {ref!r}")
    return refs


def _draw_target(ref: StageRef) -> "tuple":
    """``(stage, field)`` a randomizer StageRef names (``None`` where it names none)."""
    p = ref.params
    if isinstance(p, _MappingABC):
        return p.get("stage"), p.get("field")
    return getattr(p, "stage", None), getattr(p, "field", None)


def _normalize_size_hint(spec: PipelineSpec) -> str:
    """Roadmap PR13d: the hint of a ``normalize.size`` randomizer on a pipeline without a
    normalize stage: a gate generator that is not re-normalized holds its size; a
    centerline generator's per-env size is its own ``generator.size`` (``external`` has
    none: no hint)."""
    gen = next((r for sid, r in zip(spec.ids, spec.stages) if sid == "generator"), None)
    if gen is None or (gen.kind, gen.name) == ("generator", "external"):
        return ""
    if gen.kind == "gate_generator":
        return ("; this gate generator is not re-normalized and holds its size: use "
                "generator.size, where it declares one (a legacy GateGeneratorSpec has "
                "none)")
    return "; a centerline generator's per-env size is generator.size, where it declares one"


def _with_randomizers(spec: PipelineSpec, randomize) -> PipelineSpec:
    """``spec`` with each randomizer inserted right before the stage it draws
    (:meth:`PipelineSpec.with_randomize`); a field (or a stage id) the spec already has
    a randomizer for is refused."""
    for ref in _randomizers(randomize):
        target, field = _draw_target(ref)
        if not isinstance(target, str) or not target:
            raise ValueError(f"randomize: {ref!r} names no target stage (params 'stage')")
        if target not in spec.ids:
            hint = _normalize_size_hint(spec) if (target, field) == \
                ("normalize", "size") else ""
            raise ValueError(f"randomize: key '{target}.{field}' names stage {target!r}, "
                             f"which this pipeline does not have; stages: "
                             f"{list(spec.ids)}{hint}")
        for sid, old in zip(spec.ids, spec.stages):
            if old.kind not in PARAM_WRITER_KINDS:
                continue
            if _draw_target(old) == (target, field):
                what = f"'{target}.{field}' is already drawn by stage {sid!r}"
            elif sid == ref.slot_id:
                what = f"the spec already has a randomizer with id {sid!r}"
            else:
                continue
            raise ValueError(f"randomize: {what}; replace it, spec.replace({sid!r}, "
                             f"StageRef('randomize', 'uniform', {{...}})), or remove it "
                             f"first (spec.remove({sid!r}))")
        spec = spec.insert_before(target, ref)
    return spec


def _migrate_v1(data: Mapping) -> dict:
    """``track_gen.pipeline/1`` data as ``/2`` data (see :meth:`PipelineSpec.from_dict`);
    registry free, on the data."""
    import warnings  # noqa: PLC0415

    raw_stages = data.get("stages")
    if isinstance(raw_stages, (str, bytes)) or not isinstance(raw_stages, (list, tuple)):
        return dict(data)                  # from_dict reports the shape error
    stages = []
    for s in raw_stages:
        if isinstance(s, _MappingABC):
            s = dict(s)
            if isinstance(s.get("params"), _MappingABC):
                s["params"] = dict(s["params"])
        stages.append(s)
    kinds = [(s.get("kind"), s.get("name")) if isinstance(s, dict) else (None, None)
             for s in stages]
    # (1) the dropped half_width params, which must agree with the width stage's.
    popped = []
    for s, kn in zip(stages, kinds):
        if kn in _V1_HALF_WIDTH and isinstance(s.get("params"), dict) and \
                "half_width" in s["params"]:
            popped.append((s.get("id") or kn[0], kn, s["params"].pop("half_width")))
    width_hw = None
    for s, kn in zip(stages, kinds):
        if kn == ("width", "constant"):
            p = s.get("params")
            width_hw = (p or {}).get("half_width", _V1_DEFAULT_HALF_WIDTH) \
                if isinstance(p, (dict, type(None))) else _V1_DEFAULT_HALF_WIDTH
            break
    hw = width_hw
    for sid, kn, v in popped:
        if hw is None:
            hw = v
        if float(v) != float(hw):
            raise ValueError(
                f"pipeline format {_FORMAT_V1}: stage {sid!r} ({kn[0]}.{kn[1]}) sets "
                f"half_width={v!r} but the width is {hw!r}; half_width is no longer a "
                f"{kn[0]}.{kn[1]} param (0.2.0.dev4): the half-width is cl.half_width, "
                f"written by the width stage (presets.track(half_width=...) / "
                f"width.constant)")
    if hw is None:
        hw = _V1_DEFAULT_HALF_WIDTH
    # (2) the width stages in the dev3 slot (after the first relax, before the first
    # frames) move to just before that relax.
    kinds_of = [s.get("kind") if isinstance(s, dict) else None for s in stages]
    if "relax" in kinds_of:
        r = kinds_of.index("relax")
        f = kinds_of.index("frames") if "frames" in kinds_of else len(stages)
        moved = [s for i, s in enumerate(stages) if r < i < f and kinds_of[i] == "width"]
        if moved:
            rest = [s for i, s in enumerate(stages)
                    if not (r < i < f and kinds_of[i] == "width")]
            stages = rest[:r] + moved + rest[r:]
            kinds_of = [s.get("kind") if isinstance(s, dict) else None for s in stages]
    # (3) a relax.xpbd with no width stage before it gets width.constant; a width stage
    # left after frames (dev3: a width reading cl.kappa) would then be a second writer.
    for i, s in enumerate(stages):
        if isinstance(s, dict) and (s.get("kind"), s.get("name")) == ("relax", "xpbd"):
            if "width" not in kinds_of[:i]:
                late = [st for j, st in enumerate(stages)
                        if j > i and kinds_of[j] == "width"]
                if late:
                    sid = late[0].get("id") or "width"
                    raise ValueError(
                        f"pipeline format {_FORMAT_V1}: width stage {sid!r} "
                        f"({late[0].get('kind')}.{late[0].get('name')}) runs after "
                        f"frames; since {FORMAT} the width relax.xpbd reads comes before "
                        f"relax (width.constant), so make {sid!r} a rewrite of "
                        f"cl.half_width (read it) after a width.constant before relax, or "
                        f"move it before relax")
                stages.insert(i, {"kind": "width", "name": "constant", "id": None,
                                  "params": {"half_width": hw}, "ports": {},
                                  "capacity": {}})
            break
    plugin = sorted({f"{s.get('kind')}.{s.get('name')}" for s in stages
                     if isinstance(s, dict) and s.get("kind") == "width"
                     and s.get("name") != "constant"})
    changed = (f"; the width stage(s) {plugin} now run before relax and XPBD, the offset "
               f"and validity read their per-point width, so the outputs differ from "
               f"0.2.0.dev3 (CHANGELOG 0.2.0.dev4, 'Changed (outputs)')") if plugin else ""
    warnings.warn(f"pipeline format {_FORMAT_V1} (0.2.0.dev3) migrated to {FORMAT}: the "
                  f"width stage runs before relax and the half_width params of "
                  f"relax.xpbd / offset.default / validity.default moved to it{changed}; "
                  f"save the spec again (to_dict) to store the new format",
                  DeprecationWarning, stacklevel=3)
    out = dict(data)
    out["format"] = FORMAT
    out["stages"] = stages
    return out


def _required(data, what: str, keys) -> Mapping:
    """``data`` as a mapping holding every key of ``keys`` (``ValueError`` otherwise)."""
    if not isinstance(data, _MappingABC):
        raise ValueError(f"{what} must be a mapping, got {type(data).__name__}")
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"{what} is missing {', '.join(repr(k) for k in missing)}")
    return data
