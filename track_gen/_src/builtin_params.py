"""Params classes of the built-in track and gate generators.

Each class keeps the FLAT ``TrackGenConfig`` / ``GateGenConfig`` field names (so the
legacy error messages and ``generator_params={"polar_num_knots": 9}`` read the same).
The orchestrator resolves the Params (``from_legacy`` when ``generator_params`` is
``None``) and writes them back onto its private run snapshot (``_apply_legacy``).

Roadmap PR13d: every built-in track generator and the ``bezier`` / ``hull`` gate
generators declare a dynamic ``size`` (float32 in :data:`track_gen.ops.constants.
SIZE_BOUNDS`, default 1.0, the LAST field: the flat ``TrackGenConfig.size`` /
``GateGenConfig.size``), applied where the generator applies ``scale``; bezier / hull's
``min_num_points`` is a dynamic ``int32`` (clamped to ``[2, max_num_points]`` at use,
``set`` checks it through the per-instance bound of ``_instance_bounds()``). A dynamic
``int32`` field's non-bool value is coerced with ``int()`` at construction (the legacy
``int(config.x)`` read: a YAML ``9.0``, a ``numpy.int64``, ``"9"``).

Roadmap PR13c: the knobs that shape the output and are read only inside kernels are
DYNAMIC (per-env ``[E]`` float32 arrays with declared ``lo`` / ``hi``): bezier ``rad``
/ ``handle_clamp_frac``, ``hull_displacement``, the polar jitters, the voronoi radial
variation / angular jitter and the checkpoint radius / angle jitter / turn rate (the
gate generators' copies of the polar / voronoi / checkpoint ones too). The generator
Stages read them on the device (``StageBuilder.param``) through array-reading twins of
their kernels, clamped to the declared bounds at use; the initial value is the field's
value (a flat config's knob), so default outputs are byte-identical. Every other field
(counts, sizes, enums, branches, host-transformed or solver knobs) is static. The
static view a built-in Stage builds from binds a poison sentinel to each dynamic name
(``_poison_dynamic``): any host read raises. A dynamic field's value that is not a
bool and that ``float()`` accepts (a YAML ``"1e9"``, a 0-d numpy array, a torch scalar)
is coerced to a ``float`` at construction (``__post_init__``), as the legacy
``float(config.x)`` reads accepted it; a bool is rejected.

``validate()`` holds the generator-specific rules that ``TrackGenConfig.__post_init__``
/ ``GateGenConfig.__post_init__`` used to run for every config: same conditions, same
messages, same order. They now run at generator construction, and only for the
selected generator. ``lo``/``hi`` are set only where a legacy rule rejects the same
inclusive bound, so the generic bound check never fires before a legacy message.

Not part of the public API: reach a class through ``registry.get(kind, name).Params``
(built-in track and gate generators are Stages) or ``registry.describe(kind, name).params_schema``, or pass a
dict.
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Mapping

from ..ops.constants import _MIN_COUNT_LO, SIZE_BOUNDS
from ..plugin.params import Params, _f32, _fields, _is_real, _out_of_range, dynamic, static

__all__ = [
    "BezierParams",
    "CheckpointGateParams",
    "CheckpointParams",
    "HullParams",
    "PointGateParams",
    "PolarGateParams",
    "PolarParams",
    "RepulsiveParams",
    "VoronoiGateParams",
    "VoronoiParams",
]

_LAYOUTS = ("ring", "void_ring", "clustered", "mixed")


def _norm(v):
    """Comparable form of a flat value: lists -> tuples (``repulsive_stages`` may be a
    list), so ``(64, 128, 256) == [64, 128, 256]``."""
    if isinstance(v, list):
        return tuple(_norm(x) for x in v)
    if isinstance(v, tuple):
        return tuple(_norm(x) for x in v)
    return v


_DYNAMIC_F32: dict = {}
_DYNAMIC_I32: dict = {}


def _dynamic_f32(cls) -> "tuple[str, ...]":
    """The dynamic float32 field names of a Params class (cached per class)."""
    names = _DYNAMIC_F32.get(cls)
    if names is None:
        import warp as wp  # noqa: PLC0415

        names = tuple(f.name for f in _fields(cls)
                      if f.kind == "dynamic" and f.dtype is wp.float32)
        _DYNAMIC_F32[cls] = names
    return names


def _dynamic_i32(cls) -> "tuple[str, ...]":
    """The dynamic int32 (non-bool) field names of a Params class (cached per class)."""
    names = _DYNAMIC_I32.get(cls)
    if names is None:
        import warp as wp  # noqa: PLC0415

        names = tuple(f.name for f in _fields(cls)
                      if f.kind == "dynamic" and f.dtype is wp.int32
                      and f.py_type is not bool)
        _DYNAMIC_I32[cls] = names
    return names


#: Roadmap PR13d hints (appended to the errors of a per-env use, or an unknown
#: ``generator_params`` key, of a field the Params class has not as a dynamic one).
_SCALE_HINT = ("the core scale is static (TrackGenConfig.scale / BatchSpec.scale); the "
               "per-env multiplier is size")
_RENORMALIZED_SIZE_HINT = ("this gate generator is re-normalized by normalize.extent: use "
                           "normalize.size, whose initial value is GateGenConfig.size / "
                           "presets.gates(size=)")


def _corner_factor(p) -> float:
    """How the longest bezier / hull loop grows with the static corner capacity:
    ``(max_num_points / 13) ** 0.6`` above the default 13 (measured on CPU, E=512: the
    longest loop x1.11 / 1.17 / 1.29 / 1.54 / 1.93 at 16 / 20 / 24 / 30 / 40 corners;
    the factor is at or above each), 1 below it."""
    return max(1.0, float(p.max_num_points) / 13.0) ** 0.6


#: The longest raw loop at ``size = 1``, ``scale = 1`` and the default shape knobs, per
#: built-in (roadmap PR13d; the tuning guide's table, its "truncates from" column is
#: ``22.98 / L``): the maximum over 8192 envs on CPU (seeds ``0..4095`` and ``7919 * e +
#: 13`` for ``e < 4096``), rounded up. A larger batch can have a longer one.
_LONGEST_LOOP = {"bezier": 8.68, "hull": 8.30, "polar": 6.24, "voronoi": 5.01,
                 "checkpoint": 7.26}


def _shape_at_default(p, names, drawn) -> bool:
    """The loop-shape knobs ``names`` at their class defaults and not drawn per env:
    where the nominal longest loop was measured (a knob that lengthens the loop past it
    turns the truncation warning off rather than letting it under-estimate)."""
    fields = type(p).__dataclass_fields__
    return all(n not in drawn and getattr(p, n) == fields[n].default for n in names)


def _size_field():
    """The per-env ``size`` multiplier: dynamic float32 in
    :data:`~track_gen.ops.constants.SIZE_BOUNDS`, default 1.0, declared LAST."""
    return dynamic(1.0, lo=SIZE_BOUNDS[0], hi=SIZE_BOUNDS[1],
                   doc="Per-env size multiplier of the plan view, applied after scale "
                       "(per env; 1: unchanged; z and widths are absolute).")


def _is_bool_like(v) -> bool:
    """A Python / numpy bool, or a numpy array / torch tensor of a bool dtype (checked by
    name: torch is not imported)."""
    if isinstance(v, bool) or type(v).__name__ == "bool_":
        return True
    dtype = getattr(v, "dtype", None)
    return dtype is not None and (getattr(dtype, "kind", None) == "b"
                                  or str(dtype) == "torch.bool")


def _as_int(v):
    """A dynamic int32 field's value as the legacy ``int(config.x)`` read it: a value
    ``int()`` accepts (a YAML ``9.0``, a ``numpy.int64``, ``"9"``) becomes that ``int``;
    a bool (Python, numpy or a bool array / tensor), an ``int`` and anything ``int()``
    refuses are returned unchanged (the generic value check then rejects a bool or a
    non-integer)."""
    if type(v) is int or _is_bool_like(v):
        return v
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return v


def _as_float(v):
    """A dynamic float32 field's value as the legacy ``float(config.x)`` read it: a
    value ``float()`` accepts (a numeric string such as PyYAML's ``"1e9"``, a 0-d numpy
    array, a torch scalar tensor, an int) becomes that ``float``; a bool (Python, numpy
    or a bool array / tensor), a ``float`` and anything ``float()`` refuses are returned
    unchanged (the generic value check then rejects a bool or a non-number)."""
    if type(v) is float or _is_bool_like(v):
        return v
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return v


def _declared_bounds(cls, name: str) -> "tuple[float, float]":
    """``(lo, hi)`` of a dynamic field as float32-clamp bounds (``-inf`` / ``inf`` for an
    undeclared side): the one source of the twins' clamp constants."""
    meta = cls.__dataclass_fields__[name].metadata
    lo, hi = meta.get("lo"), meta.get("hi")
    return (-math.inf if lo is None else float(lo), math.inf if hi is None else float(hi))


@dataclasses.dataclass(frozen=True, kw_only=True)
class _FlatParams(Params):
    """Built-in Params whose field names ARE the flat config field names."""

    _lenient_static_types = True        # the legacy flat values are not type-checked
    # The static view binds a raising sentinel to each dynamic name (plugin.params).
    _poison_dynamic = True

    def __post_init__(self) -> None:
        for name in _dynamic_f32(type(self)):
            v = getattr(self, name)
            c = _as_float(v)
            if c is not v:
                object.__setattr__(self, name, c)
        for name in _dynamic_i32(type(self)):          # roadmap PR13d: legacy int() reads
            v = getattr(self, name)
            c = _as_int(v)
            if c is not v:
                object.__setattr__(self, name, c)

    @classmethod
    def from_legacy(cls, cfg):
        """An instance holding the values of the config fields of the same names (every
        field of this class is a flat ``TrackGenConfig`` / ``GateGenConfig`` field),
        uncast."""
        # Raw values, no casts: the run snapshot then carries exactly the flat values.
        return cls(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cls)})

    def _names(self) -> "list[str]":
        return [f.name for f in dataclasses.fields(self)]

    def _apply_legacy(self, cfg) -> None:
        """Write every field onto ``cfg`` (the orchestrator's private run snapshot)."""
        for name in self._names():
            object.__setattr__(cfg, name, getattr(self, name))

    def _check_no_flat_conflict(self, cfg) -> None:
        """With ``generator_params`` given, a flat field this class owns must stay at
        its default (this class's, which equals the base config's) or equal the given
        value. A config subclass that changes the default of such a field therefore
        conflicts with a ``generator_params`` that leaves it at the class default."""
        defaults = {f.name: f.default for f in dataclasses.fields(type(self))}
        dyn = set(_dynamic_f32(type(self)))
        dyn_i = set(_dynamic_i32(type(self)))

        def differs(n, ref) -> bool:
            # A flat dynamic value compares after the construction coercion, in float32
            # (the value its array holds), so np.float32(0.4) / torch.tensor(0.4) equal
            # 0.4 as they did under NumPy 2's comparison rules; a dynamic int32 one after
            # int() (roadmap PR13d: YAML 9.0 / numpy.int64(9) equal 9).
            v = getattr(cfg, n)
            if n in dyn:
                v = _as_float(v)
                if _is_real(v) and _is_real(ref):
                    return _f32(v) != _f32(ref)
            if n in dyn_i:
                v, ref = _as_int(v), _as_int(ref)
            return _norm(v) != _norm(ref)

        bad = sorted(
            n for n in self._names()
            if hasattr(cfg, n)
            and differs(n, defaults[n])
            and differs(n, getattr(self, n)))
        if bad:
            raise ValueError(
                f"generator_params and the flat {type(cfg).__name__} fields {bad} both "
                f"configure generator {cfg.generator!r} with different values; set them "
                f"in one place (generator_params, or the flat fields alone)")

    def _check_legacy_config(self, cfg) -> None:
        """Cross-field rules that need the (core) config; none by default."""

    def _instance_bounds(self) -> dict:
        """``field -> (lo, hi, reason)``: per-instance bounds narrower than a dynamic
        field's declared ones, checked by ``StageParams.set`` on a scalar and reported by
        ``StageParams.bounds``; none by default."""
        return {}

    def _nominal_perimeter(self, scale: float, drawn: "Mapping[str, float]") -> \
            "float | None":
        """A nominal maximum raw-loop perimeter at ``size == 1`` for the planner's
        truncation warning, or ``None`` (not checked). ``drawn``: the
        largest initial value (a randomizer's ``hi``) of each field drawn per env."""
        return None


# ---------------------------------------------------------------------------------------
# Track generators
# ---------------------------------------------------------------------------------------

def _check_num_points_per_segment(p) -> None:
    if int(p.num_points_per_segment) < 2:
        raise ValueError(
            f"num_points_per_segment must be >= 2, got {p.num_points_per_segment!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PointFamilyParams(_FlatParams):
    """The corner sampler shared by ``"bezier"`` and ``"hull"``."""

    min_num_points: int = dynamic(9, lo=_MIN_COUNT_LO,
                                  doc="Minimum corner count (per env; each env's count "
                                      "is uniform on [min, max_num_points]; clamped to "
                                      "max_num_points at use).")
    max_num_points: int = static(13, doc="Maximum corner-anchor count per env "
                                         "(>= min_num_points; sizes the corner buffers).")
    num_points_per_segment: int = static(30, lo=2, doc="Dense samples per "
                                                       "corner-to-corner segment.")
    min_point_distance: float = static(0.05, hi=0.5, doc="Minimum grid-cell spacing for "
                                                         "corner sampling, in (0, 0.5].")

    def validate(self, layout) -> None:
        # Point-family sampler inputs: min_point_distance is a divisor
        # (num_cells = int(1/(min_point_distance*2))); min/max_num_points feed wp.randi.
        if int(self.min_num_points) < 2:
            raise ValueError(f"min_num_points must be >= 2, got {self.min_num_points!r}")
        if int(self.max_num_points) < int(self.min_num_points):
            raise ValueError(
                "max_num_points must be >= min_num_points, got "
                f"{self.max_num_points!r} < {self.min_num_points!r}")
        if not (0.0 < float(self.min_point_distance) <= 0.5):
            raise ValueError(
                f"min_point_distance must be in (0, 0.5], got {self.min_point_distance!r}")
        _check_num_points_per_segment(self)

    _field_hints = {"scale": _SCALE_HINT}

    def _instance_bounds(self) -> dict:
        # The per-env corner count's upper end is the static corner capacity.
        return {"min_num_points": (_MIN_COUNT_LO, int(self.max_num_points),
                                   "at most max_num_points (the static corner "
                                   "capacity)")}


@dataclasses.dataclass(frozen=True, kw_only=True)
class BezierParams(_PointFamilyParams):
    """Parameters of the ``"bezier"`` track generator."""

    rad: float = dynamic(0.4, lo=0.0, hi=1.0,
                         doc="Cubic-Bezier handle length as a fraction of the segment "
                             "chord (per env; inert at or above handle_clamp_frac).")
    edgy: float = static(0.0, doc="Corner-tangent blend weight.")
    handle_clamp_frac: float = dynamic(0.4, lo=0.0,
                                       doc="Adaptive Bezier-handle clamp (fraction of the "
                                           "shorter incident edge; per env; very large "
                                           "disables it).")
    size: float = _size_field()

    def _nominal_perimeter(self, scale: float, drawn: "Mapping[str, float]") -> float:
        return _LONGEST_LOOP["bezier"] * float(scale) * _corner_factor(self)

    @classmethod
    def _check_draws(cls, params, ranges, randomized) -> None:
        """Plan-time rule on this stage's drawn fields (``ranges``: ``field -> (lo,
        hi)`` of its randomizers and self-drawn fields, possibly empty; ``randomized``:
        the fields a randomizer draws): a ``rad`` randomizer reaching above every value
        ``handle_clamp_frac`` can take (its drawn ``hi``, or its value when not drawn)
        warns, since ``rad`` has no effect there (the handle is ``min(rad * chord,
        clamp * shorter edge)`` and the shorter edge is at most the chord). Roadmap
"""
        rad = ranges.get("rad")
        if rad is None or "rad" not in randomized:
            return
        drawn = ranges.get("handle_clamp_frac")
        clamp = float(params.handle_clamp_frac) if drawn is None else float(drawn[1])
        if float(rad[1]) > clamp:
            from .generator_binding import _outside_track_gen_stacklevel  # noqa: PLC0415

            what = f"handle_clamp_frac={clamp!r}" if drawn is None else \
                f"the handle_clamp_frac draws' hi {clamp!r}"
            warnings.warn(
                f"generator.rad is drawn up to {rad[1]!r}, above {what}: rad has no "
                f"effect at or above handle_clamp_frac (the handle is min(rad * chord, "
                f"handle_clamp_frac * shorter edge)); keep hi(rad) <= lo(handle_clamp_frac)"
                f", or accept that rad is inert in the envs where rad >= "
                f"handle_clamp_frac", UserWarning,
                stacklevel=_outside_track_gen_stacklevel())


@dataclasses.dataclass(frozen=True, kw_only=True)
class HullParams(_PointFamilyParams):
    """Parameters of the ``"hull"`` track generator."""

    hull_displacement: float = dynamic(0.15, lo=0.0, hi=1.0,
                                       doc="Maximum per-edge radial midpoint displacement "
                                           "fraction (per env).")
    size: float = _size_field()

    def _nominal_perimeter(self, scale: float, drawn: "Mapping[str, float]") -> float:
        # Measured longest loop at the defaults (_LONGEST_LOOP), times the
        # corner-capacity factor and a linear bound on hull_displacement above its
        # default 0.15 (its initial value or a randomizer's hi; measured x1.07 / x1.36
        # longest, x1.10 / x1.52 p99 at 0.5 / 1.0).
        d = float(drawn.get("hull_displacement", self.hull_displacement))
        return _LONGEST_LOOP["hull"] * float(scale) * _corner_factor(self) * \
            (1.0 + 0.6 * max(0.0, d - 0.15))


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolarParams(_FlatParams):
    """Parameters of the ``"polar"`` track generator (the jitters' bounds are the
    generator's former clamps: ``0.85`` keeps the radius positive, ``0.45`` the knot
    order)."""

    num_points_per_segment: int = static(30, lo=2, doc="Dense samples per knot segment.")
    polar_num_knots: int = static(12, doc="Periodic cubic-spline control knots "
                                          "(clamped to >= 4).")
    polar_radial_jitter: float = dynamic(0.60, lo=0.0, hi=0.85,
                                         doc="Radial knot jitter as a fraction of the base "
                                             "radius (per env).")
    polar_angular_jitter: float = dynamic(0.30, lo=0.0, hi=0.45,
                                          doc="Angular knot jitter as a fraction of a knot "
                                              "spacing (per env).")
    size: float = _size_field()

    _field_hints = {"scale": _SCALE_HINT}

    def validate(self, layout) -> None:
        _check_num_points_per_segment(self)
        _check_polar_jitters(self)

    def _nominal_perimeter(self, scale: float, drawn: "Mapping[str, float]") -> \
            "float | None":
        # Measured longest loop at the defaults (_LONGEST_LOOP); more knots (x1.50 at
        # 24) or a larger radial jitter (x1.10 at 0.85) lengthen it: not checked then.
        if not _shape_at_default(self, ("polar_num_knots", "polar_radial_jitter"), drawn):
            return None
        return _LONGEST_LOOP["polar"] * float(scale)


def _check_polar_jitters(p) -> None:
    """The polar jitters' bounds, with the 0.2.0.dev4 migration in the message (it
    clamped a value outside to the bound rejects it). Compared in
    float32, like the generic bound check that follows; a bool or a non-number is left
    to it."""
    for name, hi in (("polar_radial_jitter", 0.85), ("polar_angular_jitter", 0.45)):
        v = getattr(p, name)
        if _is_real(v) and _out_of_range(v, 0.0, hi, True):
            near = 0.0 if _f32(v) < 0.0 else hi
            raise ValueError(
                f"{name} must be in [0, {hi}], got {v!r} (0.2.0.dev4 clamped it to that "
                f"range: set {name}={near} for the same output)")


def _check_voronoi_layout(p) -> None:
    if p.voronoi_site_layout not in {"ring", "void_ring", "clustered", "mixed"}:
        raise ValueError(
            "voronoi_site_layout must be one of "
            "{'ring', 'void_ring', 'clustered', 'mixed'}, got "
            f"{p.voronoi_site_layout!r}")


def _check_voronoi_shape(p, floor: int) -> None:
    if int(p.voronoi_control_points) < floor:
        raise ValueError(
            f"voronoi_control_points must be >= {floor}, got {p.voronoi_control_points!r}")
    if int(p.voronoi_num_sites) < int(p.voronoi_control_points):
        raise ValueError(
            "voronoi_num_sites must be >= voronoi_control_points, got "
            f"{p.voronoi_num_sites!r} < {p.voronoi_control_points!r}")


def _check_voronoi_variation(p) -> None:
    if float(p.voronoi_radial_variation) < 0.0:
        raise ValueError(
            f"voronoi_radial_variation must be >= 0, got {p.voronoi_radial_variation!r}")
    if float(p.voronoi_angular_jitter) < 0.0:
        raise ValueError(
            f"voronoi_angular_jitter must be >= 0, got {p.voronoi_angular_jitter!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoronoiParams(_FlatParams):
    """Parameters of the ``"voronoi"`` track generator."""

    num_points_per_segment: int = static(30, lo=2, doc="Dense samples per anchor segment.")
    voronoi_num_sites: int = static(256, doc="Total cell sites (>= voronoi_control_points).")
    voronoi_site_layout: str = static("void_ring", choices=_LAYOUTS,
                                      doc="Site distribution layout.")
    voronoi_control_points: int = static(18, lo=6, doc="Angular anchor sites selected "
                                                       "from the site field.")
    voronoi_radial_variation: float = dynamic(0.62, lo=0.0, hi=0.85,
                                              doc="Radial modulation of anchor targets "
                                                  "(per env; saturates by 0.85).")
    voronoi_angular_jitter: float = dynamic(0.08, lo=0.0, hi=math.pi,
                                            doc="Angular jitter of anchor sector targets "
                                                "(rad, per env; useful up to ~0.5).")
    size: float = _size_field()

    _field_hints = {"scale": _SCALE_HINT}

    def validate(self, layout) -> None:
        _check_num_points_per_segment(self)
        _check_voronoi_shape(self, 6)
        _check_voronoi_layout(self)
        _check_voronoi_variation(self)

    def _nominal_perimeter(self, scale: float, drawn: "Mapping[str, float]") -> \
            "float | None":
        # Measured longest loop at the defaults (_LONGEST_LOOP); the angular jitter
        # (x2.5 at 1.5), the anchor and site counts lengthen it: not checked then.
        if not _shape_at_default(self, ("voronoi_angular_jitter", "voronoi_control_points",
                                        "voronoi_num_sites"), drawn):
            return None
        return _LONGEST_LOOP["voronoi"] * float(scale)


def _check_checkpoint_count(p) -> None:
    if int(p.checkpoint_count) < 3:
        raise ValueError(f"checkpoint_count must be >= 3, got {p.checkpoint_count!r}")


def _check_checkpoint_radius(p) -> None:
    if not (0.0 <= float(p.checkpoint_radius_min_frac) < 1.0):
        raise ValueError(
            "checkpoint_radius_min_frac must be in [0, 1), got "
            f"{p.checkpoint_radius_min_frac!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CheckpointParams(_FlatParams):
    """Parameters of the ``"checkpoint"`` track generator."""

    checkpoint_count: int = static(12, lo=3, doc="Number of radial checkpoints.")
    checkpoint_radius_min_frac: float = dynamic(0.33, lo=0.0, hi=1.0,
                                                doc="Inner-radius fraction of checkpoint "
                                                    "placement (per env; the configured "
                                                    "value in [0, 1), set / randomize "
                                                    "up to the bound 1).")
    checkpoint_angle_jitter: float = dynamic(0.55, lo=0.0, hi=1.0,
                                             doc="Angular jitter as a fraction of the "
                                                 "per-checkpoint slot (per env).")
    checkpoint_turn_rate: float = dynamic(0.42, lo=0.0, hi=math.pi,
                                          doc="Maximum heading change (rad) per steering "
                                              "step (per env).")
    checkpoint_steer_gain: float = static(0.65, doc="Proportional steering gain, in (0, 1].")
    checkpoint_lookahead_frac: float = static(0.16, doc="Advance-to-next threshold as a "
                                                        "fraction of the outer radius.")
    checkpoint_best_of_k: int = static(4, lo=1, doc="Best-of-K candidate count.")
    checkpoint_clip_fallback: bool = static(False, doc="Clip the first self-crossing of "
                                                       "the selected centerline.")
    size: float = _size_field()

    _field_hints = {"scale": _SCALE_HINT}

    def validate(self, layout) -> None:
        _check_checkpoint_count(self)
        if int(self.checkpoint_best_of_k) < 1:
            raise ValueError(
                f"checkpoint_best_of_k must be >= 1, got {self.checkpoint_best_of_k!r}")
        _check_checkpoint_radius(self)

    def _nominal_perimeter(self, scale: float, drawn: "Mapping[str, float]") -> \
            "float | None":
        # Measured longest loop at the defaults (_LONGEST_LOOP); each placement /
        # steering knob moves it by up to ~12% (radius_min_frac 0, best_of_k 1): not
        # checked then.
        if not _shape_at_default(self, ("checkpoint_count", "checkpoint_radius_min_frac",
                                        "checkpoint_angle_jitter",
                                        "checkpoint_steer_gain",
                                        "checkpoint_lookahead_frac",
                                        "checkpoint_best_of_k"), drawn):
            return None
        return _LONGEST_LOOP["checkpoint"] * float(scale)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RepulsiveParams(_FlatParams):
    """Parameters of the ``"repulsive"`` track generator."""

    repulsive_grow_mult_min: float = static(4.5, lo=1.0, doc="Lower bound of the per-env "
                                                             "target-perimeter multiplier.")
    repulsive_grow_mult_max: float = static(5.5, doc="Upper bound of the target-perimeter "
                                                     "multiplier (> 1, >= min).")
    repulsive_domain_frac: float = static(0.35, doc="Confinement-disc radius fraction "
                                                    "(> 0).")
    repulsive_domain_init_ratio: float = static(4.0, doc="Domain radius over initial "
                                                         "seed-circle radius (> 1).")
    repulsive_obstacle_count_min: int = static(8, lo=1, doc="Lower bound of the inner-disc "
                                                            "obstacle count.")
    repulsive_obstacle_count_max: int = static(12, doc="Upper bound of the inner-disc "
                                                       "obstacle count (>= min).")
    repulsive_obstacle_radius_min_frac: float = static(0.02, doc="Lower bound of the "
                                                                 "inner-disc radius "
                                                                 "fraction (> 0).")
    repulsive_obstacle_radius_max_frac: float = static(0.045, doc="Upper bound of the "
                                                                  "inner-disc radius "
                                                                  "fraction (>= min).")
    repulsive_ratchet_rate: float = static(0.012, doc="Per-iteration length-growth factor "
                                                      "(> 0).")
    repulsive_alpha: float = static(3.0, doc="Tangent-point energy exponent alpha (> 0).")
    repulsive_beta: float = static(6.0, doc="Tangent-point energy exponent beta (> 0).")
    repulsive_tau: float = static(0.4, doc="Normalized flow step size.")
    repulsive_w_len: float = static(30.0, doc="Weight of the inert length regularizer.")
    repulsive_stages: "tuple[int, ...]" = static(
        (64, 128, 256), doc="Coarse-to-fine resolution schedule: strictly increasing "
                            "positive multiples of 4, the last equal to num_points.")
    repulsive_settle_iters: int = static(40, lo=1, doc="Settle-phase iteration budget.")
    repulsive_resample_every: int = static(25, lo=1, doc="Arc-length reparameterization "
                                                         "interval.")
    repulsive_stall_window: int = static(16, lo=1, doc="Iterations between per-env stall "
                                                       "checks.")
    repulsive_stall_area_tol: float = static(0.05, doc="Relative area-change tolerance of "
                                                       "the stall check.")
    repulsive_deactivate_obstacles: bool = static(True, doc="Zero the obstacle weights "
                                                            "once an env reaches its "
                                                            "target length.")
    size: float = _size_field()

    _field_hints = {"scale": _SCALE_HINT}

    def _nominal_perimeter(self, scale: float, drawn: "Mapping[str, float]") -> float:
        # Exact from the static params: the largest target perimeter the growth draws,
        # grow_mult_max * 2 pi r_init (r_init = domain_frac * P_ref / init_ratio).
        from .warp_generate_repulsive import _DOMAIN_SCALE_REF  # noqa: PLC0415

        r_init = float(self.repulsive_domain_frac) * _DOMAIN_SCALE_REF * float(scale) / \
            float(self.repulsive_domain_init_ratio)
        return float(self.repulsive_grow_mult_max) * 2.0 * math.pi * r_init

    def validate(self, layout) -> None:
        # grow_mult_min >= 1 and grow_mult_max > 1: a multiplier <= 1 never grows, which
        # drives the ratchet budget to <= 0 (an empty growth loop, garbage tracks).
        if float(self.repulsive_grow_mult_min) < 1.0:
            raise ValueError(
                f"repulsive_grow_mult_min must be >= 1, got {self.repulsive_grow_mult_min!r}")
        if float(self.repulsive_grow_mult_max) <= 1.0:
            raise ValueError(
                f"repulsive_grow_mult_max must be > 1, got {self.repulsive_grow_mult_max!r}")
        if float(self.repulsive_grow_mult_max) < float(self.repulsive_grow_mult_min):
            raise ValueError(
                "repulsive_grow_mult_max must be >= repulsive_grow_mult_min, got "
                f"{self.repulsive_grow_mult_max!r} < {self.repulsive_grow_mult_min!r}")
        if float(self.repulsive_domain_frac) <= 0.0:
            raise ValueError(
                f"repulsive_domain_frac must be > 0, got {self.repulsive_domain_frac!r}")
        if float(self.repulsive_domain_init_ratio) <= 1.0:
            raise ValueError(
                "repulsive_domain_init_ratio must be > 1, got "
                f"{self.repulsive_domain_init_ratio!r}")
        if int(self.repulsive_obstacle_count_min) < 1:
            raise ValueError(
                "repulsive_obstacle_count_min must be >= 1, got "
                f"{self.repulsive_obstacle_count_min!r}")
        if int(self.repulsive_obstacle_count_max) < int(self.repulsive_obstacle_count_min):
            raise ValueError(
                "repulsive_obstacle_count_max must be >= repulsive_obstacle_count_min, got "
                f"{self.repulsive_obstacle_count_max!r} < {self.repulsive_obstacle_count_min!r}")
        if float(self.repulsive_obstacle_radius_min_frac) <= 0.0:
            raise ValueError(
                "repulsive_obstacle_radius_min_frac must be > 0, got "
                f"{self.repulsive_obstacle_radius_min_frac!r}")
        if (float(self.repulsive_obstacle_radius_max_frac)
                < float(self.repulsive_obstacle_radius_min_frac)):
            raise ValueError(
                "repulsive_obstacle_radius_max_frac must be >= "
                "repulsive_obstacle_radius_min_frac, got "
                f"{self.repulsive_obstacle_radius_max_frac!r} < "
                f"{self.repulsive_obstacle_radius_min_frac!r}")
        if float(self.repulsive_ratchet_rate) <= 0.0:
            raise ValueError(
                f"repulsive_ratchet_rate must be > 0, got {self.repulsive_ratchet_rate!r}")
        if float(self.repulsive_alpha) <= 0.0:
            raise ValueError(f"repulsive_alpha must be > 0, got {self.repulsive_alpha!r}")
        if float(self.repulsive_beta) <= 0.0:
            raise ValueError(f"repulsive_beta must be > 0, got {self.repulsive_beta!r}")
        stages = tuple(self.repulsive_stages)
        if len(stages) == 0:
            raise ValueError("repulsive_stages must be non-empty")
        if any(int(s) <= 0 or int(s) % 4 != 0 for s in stages):
            raise ValueError(
                f"repulsive_stages entries must be positive multiples of 4, got {stages!r}")
        if any(int(stages[i]) <= int(stages[i - 1]) for i in range(1, len(stages))):
            raise ValueError(
                f"repulsive_stages must be strictly increasing, got {stages!r}")
        # settle_iters >= 1 keeps the growth loop non-empty; resample_every /
        # stall_window feed `(it+1) % k` guards (k=0 would divide by zero mid-generate).
        if int(self.repulsive_settle_iters) < 1:
            raise ValueError(
                f"repulsive_settle_iters must be >= 1, got {self.repulsive_settle_iters!r}")
        if int(self.repulsive_resample_every) < 1:
            raise ValueError(
                f"repulsive_resample_every must be >= 1, got {self.repulsive_resample_every!r}")
        if int(self.repulsive_stall_window) < 1:
            raise ValueError(
                f"repulsive_stall_window must be >= 1, got {self.repulsive_stall_window!r}")

    def _check_legacy_config(self, cfg) -> None:
        # The last stage is the tail input resolution (a cross-stage capacity rule).
        stages = tuple(self.repulsive_stages)
        if int(stages[-1]) != int(cfg.num_points):
            raise ValueError(
                "repulsive_stages[-1] must equal num_points for generator='repulsive', got "
                f"{stages[-1]!r} != {cfg.num_points!r}")


# ---------------------------------------------------------------------------------------
# Gate generators
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, kw_only=True)
class PointGateParams(_FlatParams):
    """Parameters of the ``"bezier"`` and ``"hull"`` gate generators (the corner
    sampler; the curve knobs are inert for gates)."""

    min_num_points: int = static(9, lo=2, doc="Minimum corner-anchor count per env (the "
                                              "gate count capacity, min_gates).")
    max_num_points: int = static(13, doc="Maximum corner-anchor count per env "
                                         "(>= min_num_points).")
    min_point_distance: float = static(0.05, doc="Minimum grid-cell spacing for anchor "
                                                 "sampling (> 0).")
    size: float = _size_field()

    #: Appended to the KeyError / PlanError of a per-env read of a static field
    #: (roadmap PR13d).
    _field_hints = {"min_num_points": "on gate generators it is the gate count capacity "
                                      "(min_gates): not per env in 0.x",
                    "scale": _SCALE_HINT}

    def validate(self, layout) -> None:
        if float(self.min_point_distance) <= 0.0:
            raise ValueError(
                f"min_point_distance must be > 0, got {self.min_point_distance!r}")
        if int(self.min_num_points) < 2:
            raise ValueError(
                f"min_num_points must be >= 2, got {self.min_num_points!r}")
        if int(self.max_num_points) < int(self.min_num_points):
            raise ValueError(
                "max_num_points must be >= min_num_points, got "
                f"{self.max_num_points!r} < {self.min_num_points!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolarGateParams(_FlatParams):
    """Parameters of the ``"polar"`` gate generator."""

    #: No size: the normalize.extent stage re-normalizing its gates holds it.
    _field_hints = {"size": _RENORMALIZED_SIZE_HINT}

    polar_num_knots: int = static(12, doc="Periodic cubic-spline control knots (= gates).")
    polar_radial_jitter: float = dynamic(0.60, lo=0.0, hi=0.85,
                                         doc="Radial knot jitter as a fraction of the base "
                                             "radius (per env).")
    polar_angular_jitter: float = dynamic(0.30, lo=0.0, hi=0.45,
                                          doc="Angular knot jitter as a fraction of a knot "
                                              "spacing (per env).")

    def validate(self, layout) -> None:
        _check_polar_jitters(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoronoiGateParams(_FlatParams):
    """Parameters of the ``"voronoi"`` gate generator (anchors are emitted directly as
    gates, so three control points suffice)."""

    #: No size: the normalize.extent stage re-normalizing its gates holds it.
    _field_hints = {"size": _RENORMALIZED_SIZE_HINT}

    voronoi_num_sites: int = static(256, doc="Total cell sites (>= voronoi_control_points).")
    voronoi_site_layout: str = static("void_ring", choices=_LAYOUTS,
                                      doc="Site distribution layout.")
    voronoi_control_points: int = static(18, lo=3, doc="Anchor sites selected from the "
                                                       "site field (= gates).")
    voronoi_radial_variation: float = dynamic(0.62, lo=0.0, hi=0.85,
                                              doc="Radial modulation of anchor targets "
                                                  "(per env).")
    voronoi_angular_jitter: float = dynamic(0.08, lo=0.0, hi=math.pi,
                                            doc="Angular jitter of anchor sector targets "
                                                "(rad, per env).")

    def validate(self, layout) -> None:
        _check_voronoi_shape(self, 3)
        if float(self.voronoi_radial_variation) < 0.0:
            raise ValueError(
                "voronoi_radial_variation must be >= 0, got "
                f"{self.voronoi_radial_variation!r}")
        if float(self.voronoi_angular_jitter) < 0.0:
            raise ValueError(
                "voronoi_angular_jitter must be >= 0, got "
                f"{self.voronoi_angular_jitter!r}")
        _check_voronoi_layout(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CheckpointGateParams(_FlatParams):
    """Parameters of the ``"checkpoint"`` gate generator."""

    #: No size: the normalize.extent stage re-normalizing its gates holds it.
    _field_hints = {"size": _RENORMALIZED_SIZE_HINT}

    checkpoint_count: int = static(12, lo=3, doc="Number of radial checkpoints (= gates).")
    checkpoint_radius_min_frac: float = dynamic(0.33, lo=0.0, hi=1.0,
                                                doc="Inner-radius fraction of checkpoint "
                                                    "placement (per env; the configured "
                                                    "value in [0, 1), set / randomize "
                                                    "up to the bound 1).")
    checkpoint_angle_jitter: float = dynamic(0.55, lo=0.0, hi=1.0,
                                             doc="Angular jitter as a fraction of the "
                                                 "per-checkpoint slot (per env).")

    def validate(self, layout) -> None:
        _check_checkpoint_count(self)
        _check_checkpoint_radius(self)
