"""Per-stage parameters: :class:`Params`, the :func:`static` / :func:`dynamic` field
markers, and the runtime tables a generator exposes as ``gen.params``.

A stage (a generator) declares its parameters as a frozen, keyword-only dataclass
deriving from :class:`Params`; every field needs a default::

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class WobbleParams(plugin.Params):
        lobes: int = plugin.static(5, lo=2, hi=12, doc="lobes per loop")
        amp: float = plugin.dynamic(0.15, lo=0.0, hi=0.4, doc="radial wobble amplitude")

- A **static** field is a host value baked in at construction (shapes, iteration
  counts, enums, Python branches). Changing it means building a new generator.
- A **dynamic** field is a per-env ``[E]`` device array (``float`` -> ``wp.float32``,
  ``int`` / ``bool`` -> ``wp.int32``) the stage reads by pointer
  (:meth:`track_gen.plugin.StageBuilder.param`). The caller writes it through
  ``gen.params[stage_id][field]`` / :meth:`StageParams.set`, between or inside graph
  replays, with no re-capture.

A field without :func:`static` / :func:`dynamic` metadata counts as static.

Users select the values with ``TrackGenConfig(generator=..., generator_params=...)``:
``None`` (the class's :meth:`Params.from_legacy` of the flat config, ``cls()`` by
default), an instance of the class, or a mapping of its fields (unknown keys raise and
list the valid fields).
"""
from __future__ import annotations

import dataclasses
import enum
import numbers
import sys
import types
import typing
from collections.abc import Mapping as _MappingABC
from collections.abc import Sequence as _SequenceABC
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import warp as wp

__all__ = ["NoParams", "ParamTable", "Params", "StageParams", "dynamic", "static"]

# Field names a Params class may not use: its methods, and the names a later release may
# give Params helpers (PipelineSpec.to_dict / from_dict serialise Params). Names starting
# with "_" are reserved too.
_RESERVED = frozenset({"validate", "from_legacy", "to_dict", "from_dict", "replace",
                       "schema", "fields", "describe"})
_META = "tg"                      # metadata key marking a static()/dynamic() field
_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_TABLE_TOKEN = object()           # guards the StageParams / ParamTable constructors


def static(default, *, lo=None, hi=None, choices=None, doc: str = "", ui=None):
    """A field baked in at construction: shapes, iteration counts, enums, Python
    branches.

    Args:
        default: The field's default value (required: every Params field has one).
        lo: Inclusive lower bound (JSON schema ``minimum``), or ``None``.
        hi: Inclusive upper bound (JSON schema ``maximum``), or ``None``.
        choices: The allowed values (JSON schema ``enum``), or ``None``.
        doc: A one-line description (JSON schema ``description``).
        ui: Free-form UI hint: a widget name such as ``"slider"``, or a mapping with
            ``widget`` / ``label`` / ``step`` / ``group`` keys (JSON schema ``x-ui``).
    """
    return dataclasses.field(default=default, metadata=MappingProxyType({
        _META: "static", "lo": lo, "hi": hi,
        "choices": None if choices is None else tuple(choices), "doc": str(doc), "ui": ui}))


def dynamic(default, *, lo=None, hi=None, doc: str = "", ui=None):
    """A per-env field: an ``[E]`` device array the stage reads by pointer
    (:meth:`~track_gen.plugin.StageBuilder.param`).

    The annotation selects the dtype: ``float`` -> ``wp.float32``, ``int`` and ``bool``
    -> ``wp.int32``. The array is filled with the field's value at construction and is
    writable between or inside graph replays with no re-capture
    (``gen.params[stage_id][field]``, :meth:`StageParams.set`). Arguments as in
    :func:`static` (no ``choices``).
    """
    return dataclasses.field(default=default, metadata=MappingProxyType({
        _META: "dynamic", "lo": lo, "hi": hi, "choices": None, "doc": str(doc), "ui": ui}))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Params:
    """Base of every stage's parameters: a frozen, keyword-only dataclass whose fields
    all have defaults.

    Subclasses must be decorated with ``@dataclasses.dataclass(frozen=True,
    kw_only=True)``; fields use :func:`static` or :func:`dynamic` (a plain field counts
    as static). Reserved field names: ``validate``, ``from_legacy``, ``to_dict``,
    ``from_dict``, ``replace``, ``schema``, ``fields``, ``describe`` and any name
    starting with ``_``.

    A static field annotated ``int``, ``float`` or ``bool`` must hold a value of that
    type (``int``: an integral, not a bool; ``float``: any real, not a bool; ``bool``:
    a bool, or 0 / 1), checked with its ``lo`` / ``hi`` / ``choices``.
    """

    # The built-in flat Params keep the legacy configs' lenient value types.
    _lenient_static_types = False

    def validate(self, layout) -> None:
        """Validate ONLY this object's own fields; raise ``ValueError``.

        Called at generator construction, and only for the selected generator, before
        the generic ``lo`` / ``hi`` / ``choices`` checks. The default does nothing.

        Args:
            layout: The run's :class:`~track_gen.plugin.Layout` (``num_envs``,
                ``stride``, ``device``).
        """

    @classmethod
    def from_legacy(cls, cfg) -> "Params":
        """Map a flat ``TrackGenConfig`` / ``GateGenConfig`` to this class.

        Used when ``cfg.generator_params`` is ``None``. The default returns ``cls()``.
        A plugin migrating off a ``TrackGenConfig`` subclass overrides it to read its
        old flat fields (``getattr(cfg, "acme_x", default)``).
        """
        return cls()


@dataclasses.dataclass(frozen=True, kw_only=True)
class NoParams(Params):
    """The Params of a stage without parameters."""


# ---------------------------------------------------------------------------------------
# Field introspection (private: used by the registry, the orchestrator and describe())
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Field:
    """One Params field, normalised."""

    name: str
    kind: str                      # "static" | "dynamic"
    default: Any
    py_type: Any                   # resolved annotation (or type(default))
    lo: Any
    hi: Any
    choices: "tuple | None"
    doc: str
    ui: Any
    dtype: Any                     # wp.float32 / wp.int32 for dynamic fields, else None


def _type_hints(cls) -> dict:
    """The resolved field annotations; per field when one of them cannot be resolved
    (e.g. a name local to a function), so the others still resolve."""
    try:
        return typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001 - unresolvable forward refs: fall back per field
        pass
    module = sys.modules.get(cls.__module__)
    namespace = dict(vars(module)) if module is not None else {}
    out = {}
    for f in dataclasses.fields(cls):
        tp = f.type
        try:
            for _ in range(3):                # a quoted annotation under __future__
                if not isinstance(tp, str):
                    break
                tp = eval(tp, namespace, dict(vars(cls)))  # noqa: S307 - an annotation
        except Exception:  # noqa: BLE001 - fall back to type(default)
            continue
        if not isinstance(tp, str):
            out[f.name] = tp
    return out


def _field_default(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:          # type: ignore[misc]
        return f.default_factory()                             # type: ignore[misc]
    return dataclasses.MISSING


def _dynamic_dtype(py_type, default):
    """``wp.float32`` / ``wp.int32`` for a dynamic field's type, or ``None``."""
    for candidate in (py_type, None if default is dataclasses.MISSING else type(default)):
        if candidate is bool or candidate is int:
            return wp.int32
        if candidate is float:
            return wp.float32
    return None


def _fields(cls) -> "tuple[_Field, ...]":
    """The normalised fields of a Params class (dataclass field order)."""
    hints = _type_hints(cls)
    out = []
    for f in dataclasses.fields(cls):
        meta = f.metadata or {}
        kind = meta.get(_META, "static")
        default = _field_default(f)
        py_type = hints.get(f.name)
        if py_type is None:
            py_type = f.type if not isinstance(f.type, str) else (
                None if default is dataclasses.MISSING else type(default))
        out.append(_Field(
            name=f.name, kind=kind, default=default, py_type=py_type,
            lo=meta.get("lo"), hi=meta.get("hi"), choices=meta.get("choices"),
            doc=meta.get("doc", "") or "", ui=meta.get("ui"),
            dtype=_dynamic_dtype(py_type, default) if kind == "dynamic" else None))
    return tuple(out)


def _check_params_class(cls, *, allow_dynamic: bool = True, owner: str = "") -> None:
    """The registration rules of a Params class (raises TypeError / ValueError)."""
    where = f" (declared by {owner})" if owner else ""
    if not isinstance(cls, type) or not issubclass(cls, Params):
        raise TypeError(f"Params class{where} must be a subclass of track_gen.plugin.Params, "
                        f"got {cls!r}")
    name = cls.__name__
    if "__dataclass_fields__" not in vars(cls):
        raise TypeError(
            f"{name}{where} must be decorated with "
            f"@dataclasses.dataclass(frozen=True, kw_only=True)")
    if not cls.__dataclass_params__.frozen:                    # type: ignore[attr-defined]
        raise TypeError(
            f"{name}{where} must be a frozen dataclass: decorate it with "
            f"@dataclasses.dataclass(frozen=True, kw_only=True)")
    for f in dataclasses.fields(cls):
        if f.name in _RESERVED or f.name.startswith("_"):
            raise ValueError(f"{name}.{f.name}{where}: the field name {f.name!r} is reserved "
                             f"(reserved: {sorted(_RESERVED)} and names starting with '_')")
        if _field_default(f) is dataclasses.MISSING:
            raise TypeError(f"{name}.{f.name}{where} needs a default: every Params field "
                            f"has one, so {name}() and from_legacy() always work")
    for fld in _fields(cls):
        if fld.kind != "dynamic":
            continue
        if not allow_dynamic:
            raise TypeError(
                f"{name}.{fld.name}{where} is dynamic: dynamic fields need a Stage "
                f"(StageBuilder.param); a GeneratorSpec/GateGeneratorSpec params= class "
                f"takes static fields only")
        if fld.dtype is None:
            raise TypeError(
                f"{name}.{fld.name}{where}: a dynamic field must be annotated float, int "
                f"or bool, got {fld.py_type!r}")
    try:
        default = cls()
    except Exception as exc:  # noqa: BLE001 - surface any construction error as TypeError
        raise TypeError(f"{name}(){where} failed: {type(exc).__name__}: {exc}") from exc
    _check_values(default)


def _is_union(origin) -> bool:
    return origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType)


def _from_data(tp, v):
    """``v`` (a JSON / YAML / OmegaConf value) in the form the annotation ``tp`` expects:
    a sequence becomes a tuple for a ``tuple`` annotation (recursively, also inside
    ``T | None``), and a value becomes the member ``tp(v)`` of an ``Enum`` annotation.
    Anything else is returned unchanged."""
    if v is None or tp is None:
        return v
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return v if isinstance(v, tp) else tp(v)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if _is_union(origin):
        rest = [a for a in args if a is not type(None)]
        return _from_data(rest[0], v) if len(rest) == 1 else v
    if (origin is tuple or tp is tuple) and isinstance(v, _SequenceABC) \
            and not isinstance(v, (str, bytes)):
        items = list(v)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from_data(args[0], x) for x in items)
        if args and len(args) == len(items):
            return tuple(_from_data(a, x) for a, x in zip(args, items))
        return tuple(items)
    return v


def _coerce(cls, value, *, what: str, cfg=None, label: str = "generator_params") -> Params:
    """Resolve a ``generator_params`` value to an instance of ``cls``.

    ``None`` -> ``cls.from_legacy(cfg)`` (``cls()`` without a config); an instance of
    ``cls`` -> itself; a mapping -> ``cls(**mapping)`` (unknown keys raise), its values
    converted by annotation (:func:`_from_data`: arrays -> tuples, enum values ->
    members) so a schema-shaped mapping (JSON, YAML, Hydra) gives the same instance.
    ``what`` names the selection in messages (e.g. ``"generator 'polar'"``), ``label``
    the argument (``"generator_params"``; a pipeline stage's ``"params"``).
    """
    if value is None:
        params = cls.from_legacy(cfg) if cfg is not None else cls()
        if not isinstance(params, cls):
            raise TypeError(f"{cls.__name__}.from_legacy() must return a {cls.__name__}, "
                            f"got {type(params).__name__}")
        return params
    if isinstance(value, cls):
        return value
    if isinstance(value, _MappingABC):
        valid = [f.name for f in dataclasses.fields(cls)]
        keys = list(value)
        if cls is NoParams and keys:
            raise ValueError(f"{what} takes no {label}; got {sorted(map(str, keys))}")
        unknown = sorted(str(k) for k in keys if k not in valid)
        if unknown:
            raise ValueError(f"unknown {label} for {what}: {unknown}; "
                             f"valid fields: {valid}")
        types_ = {f.name: f.py_type for f in _fields(cls)}
        kwargs = {}
        for k, v in value.items():
            try:
                kwargs[k] = _from_data(types_[k], v)
            except ValueError as exc:          # an Enum value that is not a member
                raise ValueError(f"{cls.__name__}.{k}: {exc}") from None
        return cls(**kwargs)
    if cls is NoParams:
        raise TypeError(f"{what} takes no {label}; got a "
                        f"{type(value).__name__}")
    raise TypeError(f"{label} for {what} must be a {cls.__name__} or a mapping of "
                    f"its fields, got {type(value).__name__}")


def _bounds_text(lo, hi) -> str:
    return f"[{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]"


def _is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _type_check(tp, v) -> "tuple[bool, str]":
    """``(ok, expected)`` of value ``v`` for the scalar type ``tp`` (bool / int / float):
    a bool (or a 0/1 integral) for bool, an integral (not a bool) for int, a real (not
    a bool) for float."""
    if tp is bool:
        return (isinstance(v, bool) or (isinstance(v, numbers.Integral) and int(v) in (0, 1)),
                "a bool")
    if tp is int:
        return isinstance(v, numbers.Integral) and not isinstance(v, bool), "an int"
    return _is_real(v), "a real number"


def _check_dynamic_value(owner: str, fld: _Field, v) -> None:
    """A dynamic value is a real number (bool, or a 0/1 int, for a bool field; an
    integer for an int field) inside [lo, hi]."""
    tp = bool if fld.py_type is bool else (int if fld.dtype is wp.int32 else float)
    ok, expected = _type_check(tp, v)
    if not ok:
        raise ValueError(f"{owner}.{fld.name} is dynamic and must be {expected}, "
                         f"got {v!r}")
    _check_bounds(owner, fld, v)


def _check_bounds(owner: str, fld: _Field, v) -> None:
    # `v < lo` / `v > hi`: a NaN passes, exactly like the legacy comparisons.
    if not _is_real(v):
        return
    if (fld.lo is not None and v < fld.lo) or (fld.hi is not None and v > fld.hi):
        raise ValueError(f"{owner}.{fld.name} must be in {_bounds_text(fld.lo, fld.hi)}, "
                         f"got {v!r}")


def _check_values(params) -> None:
    """lo/hi on every real-valued field, choices on static fields, and the value types
    of dynamic fields and of static fields annotated int / float / bool (except on the
    built-in flat Params, which keep the legacy configs' leniency). Runs after
    ``validate()``."""
    owner = type(params).__name__
    strict = not type(params)._lenient_static_types
    for fld in _fields(type(params)):
        v = getattr(params, fld.name)
        if fld.kind == "dynamic":
            _check_dynamic_value(owner, fld, v)
            continue
        if strict and fld.py_type in (bool, int, float):
            ok, expected = _type_check(fld.py_type, v)
            if not ok:
                raise ValueError(f"{owner}.{fld.name} must be {expected}, got {v!r}")
        _check_bounds(owner, fld, v)
        if fld.choices is not None and v not in fld.choices:
            raise ValueError(f"{owner}.{fld.name} must be one of {list(fld.choices)}, "
                             f"got {v!r}")


def _static_view(params):
    """``params`` as :meth:`Stage.build`, :meth:`Stage.capacity` and
    :meth:`Stage.capture_class` see it: an instance of a generated subclass of its class
    (``isinstance`` holds) whose DYNAMIC fields raise ``AttributeError``.

    A dynamic field's value is per env and changes at run time, so a stage reads it on
    the device through ``StageBuilder.param``; a host scalar derived from it at build
    would be baked into the launches and ignore every later ``set``. ``params`` itself
    when the class has no dynamic field.
    """
    cls = type(params)
    dyn = frozenset(f.name for f in _fields(cls) if f.kind == "dynamic")
    if not dyn:
        return params

    def _dynamic_attr(name):
        def get(self):
            raise AttributeError(
                f"{cls.__name__}.{name} is dynamic: build(), capacity() and "
                f"capture_class() see the static fields only; read it on the device "
                f"through b.param({name!r})")
        return property(get)

    def __repr__(self):
        parts = ", ".join(f"{f.name}=<dynamic>" if f.name in dyn
                          else f"{f.name}={getattr(self, f.name)!r}"
                          for f in dataclasses.fields(cls))
        return f"{cls.__qualname__}({parts})"

    ns = {name: _dynamic_attr(name) for name in dyn}
    ns.update(__repr__=__repr__, __module__=cls.__module__, __qualname__=cls.__qualname__)
    view = object.__new__(type(cls.__name__, (cls,), ns))
    for f in dataclasses.fields(cls):
        if f.name not in dyn:
            object.__setattr__(view, f.name, getattr(params, f.name))
    return view


# ---------------------------------------------------------------------------------------
# JSON schema (SpecInfo.params_schema)
# ---------------------------------------------------------------------------------------

def _jsonable(v):
    """``v`` as plain JSON data: tuples -> lists, enums -> values, others -> repr."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, enum.Enum):
        return _jsonable(v.value)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        return float(v)
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, _MappingABC):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return repr(v)


def _json_type(tp) -> dict:
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return {"enum": [_jsonable(m.value) for m in tp]}
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if _is_union(origin):
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1 and len(rest) < len(args):
            inner = _json_type(rest[0])
            if isinstance(inner.get("type"), str):
                inner["type"] = [inner["type"], "null"]
            return inner
        return {}
    if origin in (tuple, list):
        out: dict = {"type": "array"}
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            items = _json_type(args[0])
        elif origin is tuple and args:
            items = _json_type(args[0]) if all(a == args[0] for a in args) else {}
            if not items:
                out["prefixItems"] = [_json_type(a) for a in args]
            out["minItems"] = out["maxItems"] = len(args)
        else:
            items = _json_type(args[0]) if args else {}
        if items:
            out["items"] = items
        return out
    return {}


def _schema(cls) -> dict:
    """The JSON schema of a Params class (always ``json.dumps``-able)."""
    props = {}
    for fld in _fields(cls):
        prop = _json_type(fld.py_type)
        prop["default"] = _jsonable(fld.default)
        if fld.lo is not None:
            prop["minimum"] = _jsonable(fld.lo)
        if fld.hi is not None:
            prop["maximum"] = _jsonable(fld.hi)
        if fld.choices is not None:
            prop["enum"] = _jsonable(list(fld.choices))
        if fld.doc:
            prop["description"] = fld.doc
        if fld.kind == "dynamic":
            prop["x-dynamic"] = True
            prop["x-dtype"] = "float32" if fld.dtype is wp.float32 else "int32"
        else:
            prop["x-static"] = True
        if fld.ui is not None:
            prop["x-ui"] = _jsonable(fld.ui)
        props[fld.name] = prop
    return {"$schema": _SCHEMA_URI, "title": cls.__name__, "type": "object",
            "additionalProperties": False, "properties": props}


# ---------------------------------------------------------------------------------------
# Runtime tables (gen.params)
# ---------------------------------------------------------------------------------------

class StageParams(_MappingABC):
    """One stage's dynamic parameters: ``field -> LIVE [E] array``.

    The arrays have a stable pointer for the generator's lifetime; iteration yields
    the dynamic field names only. :attr:`params` is the resolved Params instance (its
    static values and the INITIAL dynamic values), or ``None`` for a legacy generator
    without a Params class.

    Each generation latches the live arrays into the stage's working copy for the envs
    it regenerates (every env, or ``mask[e] != 0``), so a write takes effect at the next
    generation and envs left out of a masked regeneration keep the value their current
    output was built with.

    Unlatched arrays (roadmap PR13b): a field drawn by a ``randomize.*`` stage
    (:attr:`drawn`) is a single array OWNED by its randomizer and written by it every
    generation: ``self[field]`` returns it (observable), :meth:`set` raises (set the
    randomizer's ``lo`` / ``hi`` instead), and a direct write is overwritten by the next
    draw.

    Created by the generator facades, never by user code: the constructor is private
    (it raises ``TypeError``) and may change; the read API is ``[field]``, :meth:`set`,
    :attr:`params`, :attr:`stage_id` and :attr:`drawn`.
    """

    def __init__(self, *, _token=None, stage_id: str, params: "Params | None",
                 arrays: "Mapping | None" = None, num_envs: int = 0, device=None,
                 default_stream=None, drawn: "Mapping | None" = None,
                 bounds: "Mapping | None" = None) -> None:
        if _token is not _TABLE_TOKEN:
            raise TypeError("StageParams is created by the generator facades; read it "
                            "through gen.params[stage_id]")
        self._stage_id = str(stage_id)
        self._params = params
        self._arrays = dict(arrays or {})
        self._fields = {} if params is None else {f.name: f for f in _fields(type(params))}
        self._E = int(num_envs)
        self._device = device
        self._default_stream = default_stream
        self._drawn = dict(drawn or {})               # field -> randomizer stage id
        # field -> (lo, hi, what): bounds narrower than the field's own (a randomizer's
        # lo / hi take the declared range of the field it draws).
        self._bounds = dict(bounds or {})

    @property
    def drawn(self) -> Mapping[str, str]:
        """``field -> id of the randomize stage that draws it`` (empty unless a
        ``randomize.*`` stage targets this stage)."""
        return MappingProxyType(dict(self._drawn))

    @property
    def stage_id(self) -> str:
        """The stage id (``"generator"`` for the generator stage)."""
        return self._stage_id

    @property
    def params(self) -> "Params | None":
        """The resolved Params instance as of construction (``None``: a legacy generator
        without one): its static values and the INITIAL values of its dynamic fields.
        A later :meth:`set` (or a write to the live array) does not change it; the
        current values are the live arrays, ``self[field]``."""
        return self._params

    def __repr__(self) -> str:
        if self._drawn:
            dyn = ", ".join(f"{k}=<drawn by {self._drawn[k]!r}>" if k in self._drawn else k
                            for k in sorted(self._arrays))
            return f"StageParams({self._stage_id!r}, dynamic=[{dyn}], " \
                   f"params={self._params!r})"
        return (f"StageParams({self._stage_id!r}, dynamic={sorted(self._arrays)}, "
                f"params={self._params!r})")

    def __getitem__(self, field: str) -> "wp.array":
        arr = self._arrays.get(field)
        if arr is not None:
            return arr
        if field in self._fields:
            raise KeyError(f"{field!r} is a static parameter of stage {self._stage_id!r}: "
                           f"static: build a new generator to change it (read it from "
                           f".params)")
        raise KeyError(f"stage {self._stage_id!r} has no dynamic parameter {field!r}; "
                       f"dynamic fields: {sorted(self._arrays)}")

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def set(self, field: str, value, *, stream=None) -> None:
        """Write a dynamic field's LIVE ``[E]`` array, in place.

        An ENQUEUE on the resolved stream (``stream=`` > ambient scope > the
        generator's constructor stream > torch capture > Warp's current stream): one
        fill launch for a scalar (checked on the host against lo/hi first), one copy for
        an array (no range check, no host sync). Inside a capture it is recorded: a
        scalar is baked into the caller's graph, an array is read at replay. Takes
        effect at the next generation. Never allocates.

        Args:
            field: A dynamic field of this stage.
            value: A host scalar (broadcast to every env) or an ``[E]`` contiguous
                ``wp.array`` of the field's dtype on the generator's device.
            stream: Optional ``wp.Stream`` (or ``"torch"``) to enqueue on.

        Raises:
            KeyError: ``field`` is static or unknown.
            ValueError: wrong shape, dtype or device; a value that is neither a
                ``wp.array`` nor a scalar of the field's type; a scalar outside lo/hi
                (for a randomizer's ``lo`` / ``hi``: outside the declared range of the
                field it draws).
        """
        arr = self[field]
        writer = self._drawn.get(field)
        if writer is not None:
            raise ValueError(f"{self._stage_id}.{field} is drawn every generation by stage "
                             f"{writer!r}; set its bounds: params[{writer!r}].set('lo' | "
                             f"'hi', ...)")
        fld = self._fields[field]
        owner = type(self._params).__name__
        if isinstance(value, wp.array):
            if value.dtype is not arr.dtype:
                raise ValueError(
                    f"{owner}.{field}: expected a {arr.dtype.__name__} array, got "
                    f"{getattr(value.dtype, '__name__', value.dtype)}")
            if tuple(value.shape) != (self._E,):
                raise ValueError(f"{owner}.{field}: expected shape ({self._E},), got "
                                 f"{tuple(value.shape)}")
            if str(value.device) != str(arr.device):
                raise ValueError(f"{owner}.{field}: the array is on {value.device}, the "
                                 f"generator on {arr.device}")
            if not value.is_contiguous:
                raise ValueError(f"{owner}.{field}: the array must be contiguous")
            ctx = self._ctx(stream)
            ctx.copy(arr, value)
            return
        if isinstance(value, (str, bytes)) or not isinstance(value, numbers.Number):
            raise ValueError(f"{owner}.{field}: expected a scalar or an [E] wp.array, got "
                             f"{type(value).__name__}")
        _check_dynamic_value(owner, fld, value)
        extra = self._bounds.get(field)
        if extra is not None and _is_real(value):
            lo, hi, what = extra
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                raise ValueError(f"{owner}.{field} must be in {_bounds_text(lo, hi)}, the "
                                 f"declared range of {what} (the field this stage draws: "
                                 f"a draw past it would pile onto the bound), got "
                                 f"{value!r}")
        v = float(value) if arr.dtype is wp.float32 else int(value)
        self._ctx(stream).fill(arr, v)

    def _ctx(self, stream):
        from ..exec.context import LaunchContext  # noqa: PLC0415 - keep plugin import light

        return LaunchContext.for_call(self._device, stream, default=self._default_stream)


class ParamTable(_MappingABC):
    """``stage_id -> StageParams`` (``gen.params``). The PR8 facades hold exactly one
    stage, ``"generator"``. Created by the generator facades, never by user code (the
    constructor is private and may change)."""

    def __init__(self, *, _token=None, stages: "Mapping[str, StageParams]") -> None:
        if _token is not _TABLE_TOKEN:
            raise TypeError("ParamTable is created by the generator facades; read it "
                            "through gen.params")
        self._stages = dict(stages)

    def __repr__(self) -> str:
        return f"ParamTable({self._stages!r})"

    def __getitem__(self, stage_id: str) -> StageParams:
        try:
            return self._stages[stage_id]
        except KeyError:
            raise KeyError(f"no stage {stage_id!r}; stages: {sorted(self._stages)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._stages)

    def __len__(self) -> int:
        return len(self._stages)

    def set(self, stage_id: str, field: str, value, *, stream=None) -> None:
        """``self[stage_id].set(field, value, stream=stream)``."""
        self[stage_id].set(field, value, stream=stream)


def _new_stage_params(**kwargs) -> StageParams:
    """The facades' StageParams constructor."""
    return StageParams(_token=_TABLE_TOKEN, **kwargs)


def _new_param_table(stages: "Mapping[str, StageParams]") -> ParamTable:
    """The facades' ParamTable constructor."""
    return ParamTable(_token=_TABLE_TOKEN, stages=stages)
