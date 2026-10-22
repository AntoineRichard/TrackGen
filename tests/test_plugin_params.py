"""``track_gen.plugin`` Params: static/dynamic fields, registration rules, coercion of
``generator_params``, bound checks and the JSON schema (roadmap PR8). CPU only."""
from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
import warp as wp

from tests._plugin_toys import ToyLobedParams, ToyWobbleParams
from track_gen import plugin
from track_gen.plugin import GeneratorSpec, NoParams, Params, dynamic, registry, static
from track_gen.plugin.params import (
    _check_params_class,
    _check_values,
    _coerce,
    _fields,
    _schema,
)


def _gen_spec(name, params):
    return GeneratorSpec(name=name, alloc_scratch=lambda config: None,
                         generate=lambda *a, **k: None, params=params)


# ---------------------------------------------------------------------------------------
# Field markers
# ---------------------------------------------------------------------------------------

def test_static_and_dynamic_metadata():
    fields = {f.name: f for f in _fields(ToyWobbleParams)}
    lobes, amp = fields["lobes"], fields["amp"]
    assert (lobes.kind, lobes.default, lobes.lo, lobes.hi) == ("static", 5, 2, 12)
    assert lobes.doc == "lobes per loop" and lobes.ui is None and lobes.dtype is None
    assert (amp.kind, amp.default, amp.lo, amp.hi) == ("dynamic", 0.15, 0.0, 0.4)
    assert amp.ui == "slider" and amp.dtype is wp.float32
    raw = {f.name: f for f in dataclasses.fields(ToyWobbleParams)}
    assert raw["lobes"].metadata["tg"] == "static"
    assert raw["amp"].metadata["tg"] == "dynamic"
    assert raw["lobes"].metadata["choices"] is None


def test_static_choices_are_a_tuple_and_a_plain_field_is_static():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(Params):
        mode: str = static("a", choices=["a", "b"])
        plain: int = 3

    fields = {f.name: f for f in _fields(P)}
    assert fields["mode"].choices == ("a", "b")
    assert fields["plain"].kind == "static" and fields["plain"].default == 3


def test_no_params():
    assert dataclasses.fields(NoParams) == ()
    assert NoParams() == NoParams()
    assert NoParams.from_legacy(object()) == NoParams()
    assert NoParams().validate(None) is None
    _check_params_class(NoParams)


def test_params_base_defaults():
    p = ToyWobbleParams()
    assert p.validate(None) is None
    assert ToyWobbleParams.from_legacy(object()) == p
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.lobes = 3                                                  # type: ignore[misc]
    with pytest.raises(TypeError):
        ToyWobbleParams(3)                                           # keyword-only


@pytest.mark.parametrize("annotation,default,dtype", [
    (float, 0.5, wp.float32), (int, 3, wp.int32), (bool, True, wp.int32)])
def test_dynamic_dtype_mapping(annotation, default, dtype):
    P = dataclasses.make_dataclass(
        "P", [("x", annotation, dynamic(default))], bases=(Params,), frozen=True,
        kw_only=True)
    _check_params_class(P)
    (f,) = _fields(P)
    assert f.dtype is dtype


# ---------------------------------------------------------------------------------------
# Registration rules
# ---------------------------------------------------------------------------------------

def test_registration_rejects_a_non_frozen_params():
    # Python itself refuses a non-frozen dataclass deriving from the frozen Params ...
    with pytest.raises(TypeError, match="frozen"):
        @dataclasses.dataclass(kw_only=True)
        class Mutable(Params):
            x: int = 1

    # ... and the registry checks the flag (e.g. a class whose dataclass params lie).
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Lying(Params):
        x: int = 1

    Lying.__dataclass_params__ = dataclasses._DataclassParams(  # type: ignore[attr-defined]
        *[getattr(Lying.__dataclass_params__, a) if a != "frozen" else False
          for a in Lying.__dataclass_params__.__slots__])
    with pytest.raises(TypeError, match="frozen"):
        registry.add(_gen_spec("acme.mutable", Lying))
    assert "acme.mutable" not in registry.available("generator")


def test_registration_rejects_an_undecorated_subclass():
    class Undecorated(ToyLobedParams):
        extra: int = 3

    with pytest.raises(TypeError, match=r"@dataclasses.dataclass\(frozen=True, kw_only=True\)"):
        registry.add(_gen_spec("acme.undecorated", Undecorated))


def test_registration_rejects_a_field_without_default():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class NoDefault(Params):
        x: int

    with pytest.raises(TypeError, match="needs a default"):
        registry.add(_gen_spec("acme.nodefault", NoDefault))


@pytest.mark.parametrize("reserved", ["validate", "from_legacy", "to_dict", "from_dict",
                                      "replace", "schema", "fields", "describe", "_x"])
def test_registration_rejects_a_reserved_field_name(reserved):
    P = dataclasses.make_dataclass("P", [(reserved, int, static(1))], bases=(Params,),
                                   frozen=True, kw_only=True)
    with pytest.raises(ValueError, match="reserved"):
        registry.add(_gen_spec("acme.reserved", P))


def test_registration_rejects_a_non_numeric_dynamic_field():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(Params):
        mode: str = dynamic("a")

    with pytest.raises(TypeError, match="float, int or bool"):
        _check_params_class(P)


def test_registration_rejects_a_dynamic_field_on_a_legacy_spec():
    with pytest.raises(TypeError, match="dynamic fields need a Stage"):
        registry.add(_gen_spec("acme.dyn", ToyWobbleParams))
    assert "acme.dyn" not in registry.available("generator")


def test_registration_rejects_a_default_outside_its_bounds_and_a_non_params_class():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(Params):
        x: int = static(1, lo=2)

    with pytest.raises(ValueError, match=r"P.x must be in \[2, inf\], got 1"):
        registry.add(_gen_spec("acme.bounds", P))

    @dataclasses.dataclass(frozen=True)
    class NotParams:
        x: int = 1

    with pytest.raises(TypeError, match="subclass of track_gen.plugin.Params"):
        registry.add(_gen_spec("acme.notparams", NotParams))


def test_a_legacy_spec_with_static_params_registers_with_a_schema():
    registry.add(_gen_spec("acme.lobed", ToyLobedParams))
    assert registry.get("generator", "acme.lobed").params is ToyLobedParams
    schema = registry.describe("generator", "acme.lobed").params_schema
    assert set(schema["properties"]) == {"lobes", "amp"}


# ---------------------------------------------------------------------------------------
# _coerce: None / dict / instance / wrong class / unknown keys
# ---------------------------------------------------------------------------------------

def test_coerce_none_uses_from_legacy():
    class Cfg:
        pass

    assert _coerce(ToyWobbleParams, None, what="generator 'x'") == ToyWobbleParams()

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Legacy(Params):
        k: int = static(1)

        @classmethod
        def from_legacy(cls, cfg):
            return cls(k=getattr(cfg, "acme_k", 1))

    cfg = Cfg()
    cfg.acme_k = 7
    assert _coerce(Legacy, None, what="generator 'x'", cfg=cfg) == Legacy(k=7)


def test_coerce_dict_and_instance():
    p = _coerce(ToyWobbleParams, {"lobes": 7}, what="generator 'toy.wobble'")
    assert p == ToyWobbleParams(lobes=7, amp=0.15)
    inst = ToyWobbleParams(amp=0.3)
    assert _coerce(ToyWobbleParams, inst, what="generator 'toy.wobble'") is inst
    assert _coerce(ToyWobbleParams, {}, what="g") == ToyWobbleParams()


def test_coerce_converts_schema_shaped_values_by_annotation():
    import enum

    class Mode(enum.Enum):
        A = "a"
        B = "b"

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class EP(Params):
        mode: Mode = static(Mode.A)
        pair: tuple[float, float] = static((0.1, 0.2))
        span: "tuple[float, float] | None" = static((0.3, 0.4))
        stages: tuple[int, ...] = static((4, 8))
        nested: tuple[tuple[int, int], ...] = static(((1, 2),))
        n: int = static(3)

    defaults = {k: v["default"] for k, v in _schema(EP)["properties"].items()}
    data = json.loads(json.dumps(defaults))                 # lists and enum values
    p = _coerce(EP, data, what="generator 'x'")
    assert p == EP() and hash(p) == hash(EP())
    assert p.mode is Mode.A and p.pair == (0.1, 0.2) and p.nested == ((1, 2),)
    assert _coerce(EP, {"span": None}, what="g").span is None
    assert _coerce(EP, {"mode": Mode.B}, what="g").mode is Mode.B
    with pytest.raises(ValueError, match="EP.mode: 'c' is not a valid"):
        _coerce(EP, {"mode": "c"}, what="g")


def test_coerce_wrong_class_is_a_type_error():
    with pytest.raises(TypeError, match="must be a ToyWobbleParams"):
        _coerce(ToyWobbleParams, ToyLobedParams(), what="generator 'toy.wobble'")
    with pytest.raises(TypeError, match="must be a ToyWobbleParams"):
        _coerce(ToyWobbleParams, 3, what="generator 'toy.wobble'")


def test_coerce_unknown_keys_list_the_valid_fields():
    with pytest.raises(ValueError) as info:
        _coerce(ToyWobbleParams, {"lobes": 3, "x": 1, "amplitude": 2},
                what="generator 'toy.wobble'")
    assert str(info.value) == ("unknown generator_params for generator 'toy.wobble': "
                               "['amplitude', 'x']; valid fields: ['lobes', 'amp']")


def test_coerce_no_params():
    assert _coerce(NoParams, {}, what="generator 'toy.circle'") == NoParams()
    with pytest.raises(ValueError, match="generator 'toy.circle' takes no generator_params"):
        _coerce(NoParams, {"x": 1}, what="generator 'toy.circle'")
    with pytest.raises(TypeError, match="takes no generator_params"):
        _coerce(NoParams, ToyWobbleParams(), what="generator 'toy.circle'")


# ---------------------------------------------------------------------------------------
# Bounds and dynamic value types
# ---------------------------------------------------------------------------------------

def test_lo_hi_enforced_inclusively_and_nan_passes():
    _check_values(ToyWobbleParams(lobes=2, amp=0.0))
    _check_values(ToyWobbleParams(lobes=12, amp=0.4))
    _check_values(ToyWobbleParams(amp=math.nan))                 # like the legacy `<`
    with pytest.raises(ValueError, match=r"ToyWobbleParams.amp must be in \[0.0, 0.4\], "
                                         r"got 0.5"):
        _check_values(ToyWobbleParams(amp=0.5))
    with pytest.raises(ValueError, match=r"ToyWobbleParams.lobes must be in \[2, 12\]"):
        _check_values(ToyWobbleParams(lobes=1))


def test_dynamic_values_must_be_numbers_of_the_field_type():
    with pytest.raises(ValueError, match="must be a real number"):
        _check_values(ToyWobbleParams(amp="0.1"))                    # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a real number"):
        _check_values(ToyWobbleParams(amp=True))

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(Params):
        n: int = dynamic(2)
        flag: bool = dynamic(False)

    _check_values(P(n=3, flag=True))
    _check_values(P(flag=1))
    with pytest.raises(ValueError, match="P.n is dynamic and must be an int"):
        _check_values(P(n=2.5))
    with pytest.raises(ValueError, match="P.flag is dynamic and must be a bool"):
        _check_values(P(flag=2))


def test_static_values_must_match_an_int_float_or_bool_annotation():
    with pytest.raises(ValueError, match="ToyWobbleParams.lobes must be an int, got '70'"):
        _check_values(ToyWobbleParams(lobes="70"))                # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ToyWobbleParams.lobes must be an int, got 7.9"):
        _check_values(ToyWobbleParams(lobes=7.9))                 # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ToyWobbleParams.lobes must be an int"):
        _check_values(ToyWobbleParams(lobes=True))
    _check_values(ToyWobbleParams(lobes=np.int64(7)))

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(Params):
        ratio: float = static(0.5)
        flag: bool = static(False)
        label: str = static("x")

    _check_values(P(ratio=1, flag=True, label="y"))                 # an int is a real
    with pytest.raises(ValueError, match="P.ratio must be a real number, got '0.5'"):
        _check_values(P(ratio="0.5"))                               # type: ignore[arg-type]
    with pytest.raises(ValueError, match="P.flag must be a bool, got 'yes'"):
        _check_values(P(flag="yes"))                                # type: ignore[arg-type]


def test_builtin_flat_params_keep_the_legacy_value_types():
    params_cls = registry.get("generator", "bezier").params
    _check_values(params_cls(min_num_points=9.0))                  # accepted, as before


def test_static_choices_are_checked():
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(Params):
        mode: str = static("a", choices=("a", "b"))

    _check_values(P(mode="b"))
    with pytest.raises(ValueError, match=r"P.mode must be one of \['a', 'b'\], got 'c'"):
        _check_values(P(mode="c"))


# ---------------------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------------------

def test_schema_of_the_toy_params_matches_exactly():
    schema = _schema(ToyWobbleParams)
    assert schema == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ToyWobbleParams", "type": "object", "additionalProperties": False,
        "properties": {
            "lobes": {"type": "integer", "default": 5, "minimum": 2, "maximum": 12,
                      "description": "lobes per loop", "x-static": True},
            "amp": {"type": "number", "default": 0.15, "minimum": 0.0, "maximum": 0.4,
                    "description": "radial wobble amplitude", "x-dynamic": True,
                    "x-dtype": "float32", "x-ui": "slider"}}}
    assert json.loads(json.dumps(schema)) == schema


def test_schema_type_mapping():
    import enum

    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class P(Params):
        flag: bool = static(True)
        count: int = static(3)
        ratio: float = static(0.5)
        name: str = static("x", choices=("x", "y"))
        stages: tuple[int, ...] = static((4, 8))
        pair: "tuple[float, float] | None" = static(None)
        color: Color = static(Color.RED, ui={"widget": "dropdown", "label": "Color"})
        items: list[int] = static(None)
        other: object = static(None)

    props = _schema(P)["properties"]
    assert props["flag"] == {"type": "boolean", "default": True, "x-static": True}
    assert props["count"]["type"] == "integer" and props["ratio"]["type"] == "number"
    assert props["name"] == {"type": "string", "default": "x", "enum": ["x", "y"],
                             "x-static": True}
    assert props["stages"] == {"type": "array", "items": {"type": "integer"},
                               "default": [4, 8], "x-static": True}
    assert props["pair"] == {"type": ["array", "null"], "items": {"type": "number"},
                             "minItems": 2, "maxItems": 2, "default": None,
                             "x-static": True}
    assert props["color"] == {"enum": ["red", "blue"], "default": "red", "x-static": True,
                              "x-ui": {"widget": "dropdown", "label": "Color"}}
    assert props["items"]["type"] == "array" and props["items"]["items"] == {
        "type": "integer"}
    assert "type" not in props["other"]
    json.dumps(_schema(P))


def test_plugin_exports_the_params_surface():
    for name in ("Params", "NoParams", "static", "dynamic", "ParamTable", "StageParams"):
        assert name in plugin.__all__ and getattr(plugin, name) is not None


def test_a_spec_of_another_kind_with_a_params_attribute_is_not_checked():
    import types as _types

    spec = _types.SimpleNamespace(name="acme.relaxer", params={"iters": 3},
                                  build=lambda *a: None)
    registry.add(spec, kind="relax")
    info = registry.describe("relax", "acme.relaxer")
    assert info.params_schema == {}


def test_stage_params_and_param_table_are_built_by_the_facades_only():
    with pytest.raises(TypeError, match="created by the generator facades"):
        plugin.StageParams(stage_id="generator", params=None)
    with pytest.raises(TypeError, match="created by the generator facades"):
        plugin.ParamTable(stages={})
