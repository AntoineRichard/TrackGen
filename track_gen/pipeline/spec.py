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

FORMAT = "track_gen.pipeline/1"


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
            object.__setattr__(self, "params", _frozen_map(params, "params"))
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
        closed: Closed loops (``True``; open tracks are refused by ``build``).
        scale: ``TrackGenConfig.scale``, handed to the stages as ``StageBuilder.scale``.
        on_invalid: What an invalid env (``fail_mask != 0``) leaves in the ``Track``:
            ``"keep"`` (default: its computed geometry, ``valid = 0``) or
            ``"poison"`` (``count = 0``, NaN geometry, ``length = winding = 0``, so
            every utility's degenerate ``count < 3`` path fires; ``fail_mask`` and
            ``truncated`` are kept). A gate pipeline: ``count = 0``, every per-gate
            field NaN, ``valid = 0`` (``fail_mask`` kept).
        max_gates: ``None`` for a track pipeline (the result is a ``Track``); an int
            ``>= 2`` makes it a gate pipeline (e.g. :func:`~track_gen.pipeline.presets.
            gates`), whose result is a ``GateSequence`` of ``max_gates`` slots per env.
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
    """

    batch: BatchSpec
    stages: "tuple[StageRef, ...]"

    def __post_init__(self) -> None:
        if not isinstance(self.batch, BatchSpec):
            raise TypeError(f"PipelineSpec.batch must be a BatchSpec, got "
                            f"{type(self.batch).__name__}")
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

    def to_dict(self) -> dict:
        """The spec as JSON data (``"format": "track_gen.pipeline/1"``): Params become
        ``{field: value}`` (tuples as lists, enums as their values)."""
        return {"format": FORMAT, "batch": self.batch._to_dict(),
                "stages": [ref._to_dict() for ref in self.stages]}

    @classmethod
    def from_dict(cls, data: Mapping) -> "PipelineSpec":
        """Read :meth:`to_dict` data back (params stay mappings; ``build`` coerces
        them). No registry is needed."""
        if not isinstance(data, _MappingABC):
            raise TypeError(f"PipelineSpec.from_dict expects a mapping, got "
                            f"{type(data).__name__}")
        fmt = data.get("format")
        if fmt != FORMAT:
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
        return cls(batch=batch, stages=tuple(stages))


def _required(data, what: str, keys) -> Mapping:
    """``data`` as a mapping holding every key of ``keys`` (``ValueError`` otherwise)."""
    if not isinstance(data, _MappingABC):
        raise ValueError(f"{what} must be a mapping, got {type(data).__name__}")
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"{what} is missing {', '.join(repr(k) for k in missing)}")
    return data
