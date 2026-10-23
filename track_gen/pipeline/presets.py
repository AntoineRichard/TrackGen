"""Preset pipelines: :func:`track` builds the default track :class:`PipelineSpec` and
:func:`gates` the default gate one, with one argument per replaceable stage (preview,
roadmap PR10b / PR11).

:func:`track` reads the registry only for what shapes the stage list (roadmap PR13a:
the generator's ``provides``, and whether a width stage's Params has a ``half_width``
field); an unknown name is not an error there: it fails at
:func:`track_gen.pipeline.build` (or ``TrackGenerator`` construction) with the
registry's message. :func:`gates` reads the registry for the generator (its kind,
``provides``, orderings, ``preferred`` and attributes) and the stations stage's Params.
Edit the result with the :class:`PipelineSpec` operations::

    from track_gen.pipeline import StageRef, presets

    spec = presets.track(num_envs=1024, device="cuda:0", relax="acme.taubin",
                         elevation=("noise", {"z_max": 0.3, "z_noise_amplitude": 0.2}),
                         checks=("default", ("acme.max_curvature", {"kappa_max": 40.0})))
    spec = spec.insert_after("validity", StageRef("validity", "acme.other", id="other"))
    gen = TrackGenerator(spec)

A stage is selected as a name (``"xpbd"``), a ``(name, params)`` pair (params: a
Params instance, a mapping of its fields, or ``None`` for its defaults) or a
:class:`StageRef` of the slot's kind.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping as _MappingABC
from typing import Sequence, Union

from .spec import BatchSpec, PipelineSpec, StageRef

__all__ = ["StageSel", "gates", "track"]

#: A stage selection: a name, a ``(name, params)`` pair or a :class:`StageRef`.
StageSel = Union[str, tuple, StageRef]

# The width stages whose Params class has a half_width field (the preset fills it with
# the pipeline's half_width unless their params set one): width.constant, and any
# registered width stage with the field (looked up in the registry).
_HALF_WIDTH_STAGES = {("width", "constant")}
# The track preset's default half_width (an explicit other value must reach a stage or
# the default spacing).
_DEFAULT_HALF_WIDTH = 0.1


def _with_half_width(kind: str, name: str, params, half_width: float,
                     fill: "bool | None" = None):
    """``params`` with ``half_width`` filled in for a width stage whose Params has the
    field (``fill``; default: the built-in ``width.constant``)."""
    if fill is None:
        fill = (kind, name) in _HALF_WIDTH_STAGES
    if not fill:
        return params
    if params is None:
        return {"half_width": half_width}
    if isinstance(params, _MappingABC) and "half_width" not in params:
        return {**dict(params), "half_width": half_width}
    return params                                  # a Params instance: taken as is


def _check_half_width(slot: str, ref: StageRef, half_width: float) -> StageRef:
    """A selection whose params set ``half_width`` must agree with the preset's (the
    nominal width every stage with the field shares): say so here rather than as the
    planner's agreement error."""
    p = ref.params
    if isinstance(p, _MappingABC):
        v = p.get("half_width")
    elif p is not None and "half_width" in getattr(type(p), "__dataclass_fields__", {}):
        v = p.half_width
    else:
        return ref
    if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) != half_width:
        raise ValueError(f"{slot}: {ref.kind}.{ref.name} sets half_width={v!r} but "
                         f"half_width={half_width!r}; pass half_width={v!r} to "
                         f"presets.track (the stages with a half_width field share one "
                         f"nominal value; the per-point width is cl.half_width)")
    return ref


def _is_pair(sel) -> bool:
    """``(name, params)``: a 2-tuple of a name and a Params instance / mapping / None."""
    return isinstance(sel, tuple) and len(sel) == 2 and isinstance(sel[0], str) \
        and not isinstance(sel[1], (str, tuple, StageRef))


def _select(slot: str, kind: str, sel, half_width: float,
            fill: "bool | None" = None) -> StageRef:
    """``sel`` (a selection form) as a StageRef of ``kind`` (a StageRef keeps its id);
    ``fill``: fill ``half_width`` into its params (see :func:`_with_half_width`)."""
    if isinstance(sel, StageRef):
        if sel.kind != kind:
            raise ValueError(f"{slot}={sel!r}: expected a StageRef of kind {kind!r}")
        return _check_half_width(slot, dataclasses.replace(
            sel, params=_with_half_width(kind, sel.name, sel.params, half_width, fill)),
            half_width)
    if isinstance(sel, str):
        name, params = sel, None
    elif _is_pair(sel):
        name, params = sel
    else:
        raise TypeError(f"{slot} must be a stage name, a (name, params) pair, a StageRef "
                        f"or None; got {type(sel).__name__}")
    if not name:
        raise ValueError(f"{slot}: the stage name must be a non-empty string")
    return _check_half_width(slot, StageRef(
        kind, name, params=_with_half_width(kind, name, params, half_width, fill)),
        half_width)


def _sel_name(sel) -> "str | None":
    """The stage name of a selection form."""
    if isinstance(sel, StageRef):
        return sel.name
    if isinstance(sel, str):
        return sel
    if _is_pair(sel):
        return sel[0]
    return None


def _provides(reg, kind: str, name: str) -> frozenset:
    """The ``provides`` of a registered generator Stage (empty for an unknown name or a
    legacy spec: ``build`` reports an unknown name)."""
    from ..plugin.stage import _is_stage  # noqa: PLC0415

    try:
        spec = _lookup(reg, kind, name)
    except Exception:  # noqa: BLE001 - a quarantined plugin: build reports it
        return frozenset()
    return frozenset(spec.provides) if spec is not None and _is_stage(spec) else frozenset()


def _has_half_width_field(reg, name: str) -> bool:
    """Whether width stage ``name``'s Params class has a ``half_width`` field
    (``width.constant``, a plugin width like ``acme.ripple``)."""
    if ("width", name) in _HALF_WIDTH_STAGES:
        return True
    try:
        cls = _lookup(reg, "width", name)
    except Exception:  # noqa: BLE001
        return False
    fields = getattr(getattr(cls, "Params", None), "__dataclass_fields__", {})
    return "half_width" in fields


def track(*, num_envs: int, device: str, n_max: int = 384, scale: float = 1.0,
          generator: "str | StageRef" = "bezier", generator_params=None,
          num_points: int = 256, half_width: float = 0.1,
          spacing: "float | None" = None, relax: "StageSel | None" = "xpbd",
          width: StageSel = "constant", elevation: "StageSel | None" = "flat",
          checks: Sequence[StageSel] = ("default",),
          on_invalid: str = "keep", z_base: float = 0.0, registry=None) -> PipelineSpec:
    """The track pipeline: generator -> resample -> width -> relax -> uniform resample
    -> frames -> offset -> elevation -> checks -> lift (since 0.2.0.dev4 the width runs
    before relax: ``cl.half_width`` is the per-point half-width XPBD, the offset and
    validity read).

    ``presets.track(num_envs=E, device=d)`` equals the spec of
    ``TrackGenerator(TrackGenConfig(num_envs=E, device=d))`` (same stages, equal
    params; byte-identical Tracks).

    A generator that ``provides`` attributes (roadmap PR13a) has the stage computing
    them dropped: ``"z"`` (its ``raw.z`` is carried to ``cl.z``): no elevation stage,
    ``lift.elevated``; ``"half_width"`` (``raw.half_width`` -> ``cl.half_width``): no
    width stage.

    Args:
        num_envs: ``E``.
        device: The Warp device.
        n_max: Per-env point capacity (``N_max``).
        scale: ``TrackGenConfig.scale``.
        generator: A registered generator name, or a ``StageRef("generator", ...)``
            (then pass its params inside it).
        generator_params: The generator's params (a Params instance or a mapping;
            ``None``: its defaults).
        num_points: The generator's default raw point count (``capacity["raw"]``).
        half_width: The track half-width: the value of a width stage whose Params
            has a ``half_width`` field (``width.constant``, or a plugin's), unless its
            params set it, and the default spacing. A width stage without the field
            (``width.from_channel``) or a generator that provides its width takes its
            width elsewhere: then an explicit ``half_width`` other than 0.1 with an
            explicit ``spacing`` reaches nothing and raises ``ValueError``.
        spacing: The constant-spacing step (default ``0.6 * half_width``).
        relax: The ``relax`` stage (``None``: ``relax.none``).
        width: The ``width`` stage (runs before relax).
        elevation: The ``elevation`` stage (``None``: ``elevation.flat``). The lift is
            ``lift.flat`` for ``elevation.flat``, else ``lift.elevated``.
        checks: The ``validity`` stages, in order: a sequence of selections whose
            first is ``"default"`` (``validity.default``; plugin checks follow it).
            The first one's id is ``"validity"``; each further one's is its
            ``StageRef.id`` or its name.
        on_invalid: ``"keep"`` (default) or ``"poison"`` (``BatchSpec.on_invalid``).
        z_base: The altitude of the flat lift (``elevation.flat``; the flat
            ``TrackGenConfig.z_base``). Another elevation takes its altitude from its
            own params, so a nonzero ``z_base`` with it raises ``ValueError``.
        registry: The plugin registry the generator's ``provides`` and a width
            stage's Params are looked up in (default ``track_gen.plugin.registry``;
            an unknown name is reported by ``build``).

    Raises:
        ValueError: an invalid selection (wrong StageRef kind, ``width=None``,
            ``checks`` not starting with ``"default"``, a duplicate check id, an unknown
            ``on_invalid``, ``generator_params`` with a generator StageRef, params
            setting a ``half_width`` other than ``half_width``, ``elevation.flat``
            params setting a non-zero ``z_base``, a width other than the default, a
            non-flat elevation or a non-zero ``z_base`` for a generator that provides
            them, a ``half_width`` that
            reaches no stage).
        TypeError: a selection of another type, or ``checks`` given as one
            selection instead of a sequence.
    """
    if on_invalid not in ("keep", "poison"):
        raise ValueError(f"on_invalid must be 'keep' or 'poison', got {on_invalid!r}")
    if isinstance(generator, StageRef):
        if generator.kind != "generator":
            raise ValueError(f"generator={generator!r}: expected a StageRef of kind "
                             f"'generator'")
        if generator_params is not None:
            raise ValueError("pass generator_params inside the StageRef")
        gen = generator
        if not gen.capacity:
            gen = dataclasses.replace(gen, capacity={"raw": int(num_points)})
    elif isinstance(generator, str) and generator:
        gen = StageRef("generator", generator, params=generator_params,
                       capacity={"raw": int(num_points)})
    else:
        raise TypeError(f"generator must be a generator name or a StageRef, got "
                        f"{type(generator).__name__}")
    from ..plugin import registry as _default_registry  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    provides = _provides(reg, "generator", gen.name)
    hw = float(half_width)
    step = 0.6 * hw if spacing is None else spacing
    relax_ref = StageRef("relax", "none") if relax is None else \
        _select("relax", "relax", relax, hw)
    if width is None:
        raise ValueError("width=None: a width stage must write cl.half_width")
    width_ref = None
    if "half_width" in provides:
        if width != "constant":
            # No preset name: presets.gates(source="relaxed_stations") builds its
            # centerline through this function too.
            raise ValueError(f"width={width!r}: generator {gen.name!r} "
                             f"provides its own half-widths (raw.half_width -> "
                             f"cl.half_width); leave width at its default, or add a "
                             f"rewrite of cl.half_width to a hand-built spec")
        reads_hw = False
    else:
        name = _sel_name(width)
        reads_hw = name is not None and _has_half_width_field(reg, name)
        width_ref = _select("width", "width", width, hw, fill=reads_hw)
    if not reads_hw and spacing is not None and hw != _DEFAULT_HALF_WIDTH:
        what = (f"generator {gen.name!r} provides its own half-widths"
                if width_ref is None else
                f"width.{width_ref.name} takes its width from elsewhere (no half_width "
                f"field)")
        raise ValueError(f"half_width= sets width.constant's value and the default "
                         f"spacing; {what} and spacing= is given")
    if "z" in provides:
        if elevation is not None and not (isinstance(elevation, str)
                                          and elevation == "flat"):
            raise ValueError(f"presets.track(elevation={elevation!r}): generator "
                             f"{gen.name!r} provides its own altitudes (z); leave "
                             f"elevation at its default ('flat')")
        if float(z_base) != 0.0:
            raise ValueError(f"presets.track(z_base={z_base!r}): generator {gen.name!r} "
                             f"provides its own altitudes (z); leave z_base=0.0")
    elev_ref = StageRef("elevation", "flat") if elevation is None else \
        _select("elevation", "elevation", elevation, hw)
    if isinstance(checks, (str, StageRef)) or _is_pair(checks):
        raise TypeError(f"checks must be a sequence of selections, got the single "
                        f"selection {checks!r}; e.g. checks=('default',) or "
                        f"checks=('default', 'acme.max_curvature')")
    check_refs = []
    for k, sel in enumerate(checks):
        ref = _select("checks", "validity", sel, hw)
        if ref.id is None and k > 0:        # the first check is slot "validity"
            ref = dataclasses.replace(ref, id=ref.name)
        check_refs.append(ref)
    if not check_refs or check_refs[0].name != "default":
        first = check_refs[0].name if check_refs else None
        raise ValueError(f"checks must start with 'default' (validity.default starts "
                         f"valid.fail; plugin checks run after it): "
                         f"checks=('default'{'' if first is None else f', {first!r}'})"
                         f", got {tuple(checks)!r}")
    ids = [r.slot_id for r in check_refs]
    dup = next((i for i in ids if ids.count(i) > 1), None)
    if dup is not None:
        raise ValueError(f"checks: duplicate stage id {dup!r}; give one an explicit "
                         f"StageRef(id=...)")
    flat = (elev_ref.kind, elev_ref.name) == ("elevation", "flat")
    if not flat and z_base != 0.0:
        raise ValueError(f"z_base={z_base!r} is the flat lift's altitude "
                         f"(elevation='flat'); elevation.{elev_ref.name} takes its own: "
                         f"pass it in the elevation's params")
    if flat and not _is_flat(elev_ref):
        raise ValueError(f"elevation: elevation.flat sets z_base="
                         f"{_param(elev_ref, 'z_base')!r}, which sets gate altitudes (gate "
                         f"pipelines); a track's flat altitude is lift.flat's: pass "
                         f"z_base= to presets.track")
    provided_z = "z" in provides
    stages = (
        gen,
        StageRef("resample", "constant_spacing", params={"spacing": step}),
        *(() if width_ref is None else (width_ref,)),
        relax_ref,
        StageRef("resample", "uniform", id="uniform"),
        StageRef("frames", "default"),
        StageRef("offset", "default"),
        *(() if provided_z else (elev_ref,)),
        *check_refs,
        StageRef("lift", "flat", params={"z_base": float(z_base)})
        if flat and not provided_z else StageRef("lift", "elevated"),
    )
    return PipelineSpec(batch=BatchSpec(num_envs=num_envs, device=device, n_max=n_max,
                                        scale=scale, on_invalid=on_invalid),
                        stages=stages)


# ---------------------------------------------------------------------------------------
# The gate pipeline (roadmap PR11)
# ---------------------------------------------------------------------------------------

_GATE_ALIGNS = ("yaw_only", "full_tangent")
_PROVIDABLE = ("z", "tangent", "half_size", "roll")
_SOURCES = ("auto", "anchors", "stations", "relaxed_stations")


def _relax_on(gate_radius, gate_solve_iters) -> bool:
    """Whether the sphere solve runs: iterations and a float32 ``2 * r`` > 0 (the
    0.2.0.dev3 kernel returned at once otherwise)."""
    import numpy as np  # noqa: PLC0415

    return int(gate_solve_iters) > 0 and float(np.float32(2.0 * float(gate_radius))) > 0.0


def _param(ref: StageRef, field: str):
    """``field`` of a selection's params (a mapping or a Params instance), or ``None``
    when they do not set it."""
    p = ref.params
    if isinstance(p, _MappingABC):
        return p.get(field)
    if p is not None and field in getattr(type(p), "__dataclass_fields__", {}):
        return getattr(p, field)
    return None


def _with_params(ref: StageRef, fill: dict) -> StageRef:
    """``ref`` with ``fill``'s fields added to mapping (or ``None``) params that do not
    set them; a Params instance is taken as is."""
    p = ref.params
    if p is None:
        return dataclasses.replace(ref, params=dict(fill))
    if isinstance(p, _MappingABC):
        return dataclasses.replace(ref, params={**fill, **dict(p)})
    return ref


def _is_flat(ref: StageRef) -> bool:
    """``elevation.flat`` with default params (a z provider's only admitted elevation)."""
    if (ref.kind, ref.name) != ("elevation", "flat"):
        return False
    z = _param(ref, "z_base")
    return z is None or float(z) == 0.0


def _with_gate_validity(ref: StageRef, fill: dict,
                        posed_by: "str | None" = None) -> StageRef:
    """The first check, ``validity.gate_default``, with the preset's validity arguments
    (``fill``) where its params do not set them (a Params instance is taken as is).
    ``gate_radius`` / ``gate_width`` set to another value than the preset's raise here
    rather than as the planner's agreement error. ``posed_by``: the name of a generator
    that provides ``half_size`` (its ``gate_width`` must stay 0, whatever the preset is
    passed)."""
    ref = _with_params(ref, fill)
    for field in ("gate_radius", "gate_width"):
        v = _param(ref, field)
        if isinstance(v, (int, float)) and not isinstance(v, bool) \
                and float(v) != float(fill[field]):
            if field == "gate_width" and posed_by is not None:
                raise ValueError(f"checks: gate generator {posed_by!r} provides per-gate "
                                 f"half_size; validity.gate_default's gate_width must "
                                 f"stay 0, got {v!r}")
            raise ValueError(f"checks: validity.gate_default sets {field}={v!r} but "
                             f"{field}={fill[field]!r}; pass {field}={v!r} to "
                             f"presets.gates (gate_relax / gate_frames and "
                             f"validity.gate_default share one value)")
    return ref


# The centerline sub-pipeline's arguments and their defaults, and the sources that
# read them (an anchors source runs no centerline; stations sample the raw loop).
_CENTERLINE_ARGS = (("n_max", 384), ("num_points", 256), ("half_width", 0.1),
                    ("spacing", None), ("relax", "xpbd"), ("width", "constant"))
_CENTERLINE_READS = {"anchors": (), "stations": ("n_max", "num_points"),
                     "relaxed_stations": tuple(k for k, _ in _CENTERLINE_ARGS)}


def _check_centerline_args(src: str, name: str, given: dict) -> None:
    """A centerline argument set (not its default) for a source that does not read it
    raises."""
    for k, default in _CENTERLINE_ARGS:
        if k in _CENTERLINE_READS[src] or given[k] == default:
            continue
        if src == "anchors":
            raise ValueError(f"{k}= configures the centerline sub-pipeline of a stations "
                             f"source; source='anchors' (gate generator {name!r}) does "
                             f"not run one")
        raise ValueError(f"{k}= configures the relaxed_stations sub-pipeline (resample, "
                         f"relax); source='stations' places the stations on the raw loop "
                         f"(pass source='relaxed_stations')")


def _gate_stages(gen_ref: StageRef, *, provides=frozenset(), upstream=(),
                 stations_ref: "StageRef | None" = None,
                 order_ref: "StageRef | None" = None, normalize: bool = False,
                 gate_radius: float = 0.025, gate_solve_iters: int = 8,
                 gate_width: float = 0.0, gate_align: str = "yaw_only",
                 elevation_ref: "StageRef | None" = None, min_gates: int = 4,
                 z_valid_grade: float = 0.0, validity_ref: "StageRef | None" = None,
                 checks=(), carry_attributes: bool = False) -> tuple:
    """The gate stage list (shared by :func:`gates` and
    ``legacy.to_gate_pipeline_spec``): generator [-> upstream] [-> stations] [-> order
    [-> normalize]] [-> carry_*] [-> carry_attributes] -> gate_relax [-> gate_arclen ->
    elevation] -> lift [-> tangents] -> gate_frames -> validity [-> checks].
    ``order_ref=None``: a legacy generator that orders its own gates. The stages a
    generator ``provides`` are dropped (z: gate_arclen + elevation; tangent: tangents)
    and carried instead; ``half_size`` + ``roll`` select ``gate_frames.posed``;
    ``carry_attributes`` carries the per-anchor plugin attributes to the gate slots."""
    provides = frozenset(provides)
    stages = [gen_ref, *upstream]
    if stations_ref is not None:
        stages.append(stations_ref)
    if order_ref is not None:
        stages.append(order_ref)
        if normalize:
            stages.append(StageRef("normalize", "extent"))
    for attr in _PROVIDABLE:
        if attr in provides:
            stages.append(StageRef("carry", attr, id=f"carry_{attr}"))
    if carry_attributes:
        stages.append(StageRef("carry", "attributes", id="carry_attributes"))
    if _relax_on(gate_radius, gate_solve_iters):
        stages.append(StageRef("gate_relax", "spheres", params={
            "gate_radius": gate_radius, "gate_solve_iters": gate_solve_iters}))
    else:
        stages.append(StageRef("gate_relax", "none"))
    if "z" not in provides:
        elev = elevation_ref if elevation_ref is not None else StageRef("elevation",
                                                                        "flat")
        if (elev.kind, elev.name) != ("elevation", "flat"):
            stages.append(StageRef("gate_arclen", "default"))
        stages.append(elev)
    stages.append(StageRef("lift", "gates"))
    if "tangent" not in provides:
        stages.append(StageRef("tangents", "central"))
    posed = "half_size" in provides
    if posed:
        stages.append(StageRef("gate_frames", "posed", params={"gate_align": gate_align}))
    else:
        stages.append(StageRef("gate_frames", "default", params={
            "gate_width": gate_width, "gate_align": gate_align}))
    if validity_ref is None:
        vparams = {"min_gates": min_gates, "gate_radius": gate_radius,
                   "gate_width": gate_width, "z_valid_grade": z_valid_grade}
        if posed:
            vparams["check_crossing"] = "always"
        validity_ref = StageRef("validity", "gate_default", params=vparams)
    stages.append(validity_ref)
    stages.extend(checks)
    return tuple(stages)


def _lookup(reg, kind: str, name: str):
    """``reg.get(kind, name)``, or ``None`` for an unknown name (a quarantined plugin's
    ``PluginLoadError`` passes)."""
    try:
        return reg.get(kind, name)
    except ValueError as exc:
        from ..plugin import PluginLoadError  # noqa: PLC0415

        if isinstance(exc, PluginLoadError):
            raise
        return None


def gates(*, num_envs: int, device: str, generator: "str | StageRef" = "bezier",
          generator_params=None, source: str = "auto", gates: "int | None" = None,
          max_gates: "int | None" = None, min_gates: int = 4,
          order: "StageSel | None" = None, stations: "StageSel | None" = None,
          gate_radius: float = 0.025, gate_solve_iters: int = 8, gate_width: float = 0.0,
          gate_align: str = "yaw_only", elevation: "StageSel | None" = "flat",
          z_base: float = 0.0, z_valid_grade: float = 0.0,
          checks: Sequence[StageSel] = ("gate_default",), scale: float = 1.0,
          on_invalid: str = "keep", n_max: int = 384, num_points: int = 256,
          half_width: float = 0.1, spacing: "float | None" = None,
          relax: "StageSel | None" = "xpbd", width: StageSel = "constant",
          spacing_metric: str = "xy", elevation_on: "str | None" = None,
          registry=None) -> PipelineSpec:
    """The gate pipeline: generator -> order -> [normalize] -> gate_relax ->
    [gate_arclen -> elevation] -> lift -> tangents -> gate_frames -> checks.

    ``presets.gates(num_envs=E, device=d)`` equals the spec of
    ``GateGenerator(GateGenConfig(num_envs=E, device=d))`` (same stages, equal params;
    byte-identical GateSequences).

    The generator is either a gate generator (``source="anchors"``: a registered
    ``gate_generator``, whose anchors the ``order`` stage orders), or any track
    generator (``source="stations"``: ``gates`` stations at equal arc length on its raw
    loop; ``"relaxed_stations"``: on the relaxed centerline, after the track pipeline's
    resample and relax stages). ``source="auto"`` picks ``"anchors"`` for a name
    registered as a gate generator (the built-in names are both: anchors win), else
    ``"stations"``. A generator that ``provides`` per-anchor attributes has the stages
    computing them dropped (``z``: gate_arclen and elevation; ``tangent``: tangents;
    ``half_size`` + ``roll``: ``gate_frames.posed`` and
    ``validity.gate_default(check_crossing="always")``).

    Args:
        num_envs: ``E``.
        device: The Warp device.
        generator: A gate generator or track generator name, or a ``StageRef`` of kind
            ``"gate_generator"`` / ``"generator"``.
        generator_params: The generator's params (``None``: its defaults).
        source: ``"auto"``, ``"anchors"``, ``"stations"`` or ``"relaxed_stations"``.
        gates: The station count of a centerline source (default 12; it fills the
            ``gates`` field of the stations stage's params when they do not set it).
        max_gates: ``BatchSpec.max_gates`` (default: 32 for anchors, the station count
            for stations).
        min_gates: ``validity.gate_default``'s fewest real gates.
        order: The ``order`` stage (default: the generator's ``preferred["order"]`` when
            it accepts it, else ``"ccw"`` when it accepts it, else the first order it
            accepts; ``"raw"`` for stations).
        stations: The ``stations`` stage (default ``stations.raw`` /
            ``stations.relaxed`` with ``{"gates": gates}``).
        gate_radius, gate_solve_iters: The sphere solve (``gate_relax.spheres``, or
            ``gate_relax.none`` without iterations or radius) and the spacing check.
        gate_width: The FULL bar width (``half_size = 0.5 * gate_width``).
        gate_align: ``"yaw_only"`` or ``"full_tangent"``.
        elevation: The ``elevation`` stage (``None``: flat), its ports rewired to the
            gates by the planner.
        z_base: The altitude of every real gate with ``elevation.flat`` (its
            ``z_base``; ``GateGenConfig.z_base``). Another elevation takes its altitude
            from its own params, so a nonzero ``z_base`` with it (or with a generator
            that provides ``z``) raises ``ValueError``.
        z_valid_grade: The validity grade cap (0: off).
        checks: The ``validity`` stages: ``"gate_default"`` first, plugin checks after.
            ``validity.gate_default`` gets ``min_gates``, ``gate_radius``,
            ``gate_width`` and ``z_valid_grade`` (and ``check_crossing="always"`` for
            posed gates) unless its params set them (a Params instance is taken as is).
        scale: ``BatchSpec.scale``.
        on_invalid: ``"keep"`` or ``"poison"``.
        n_max, num_points, half_width, spacing, relax, width: The centerline
            sub-pipeline of a stations source (as :func:`track`): a ``stations``
            source reads ``n_max`` and ``num_points``, a ``relaxed_stations`` source
            all six; setting one that the source does not read raises ``ValueError``.
        spacing_metric: ``"xy"`` (default: the stations are equally spaced in plan
            view) or ``"xyz"`` (in 3D arc length: the loop needs its altitude, from a
            generator that provides ``z`` or an elevation run on the relaxed
            centerline; ``ValueError`` when the loop has none). Filled into the
            stations stage's params (a stations source only).
        elevation_on: Where the elevation runs for ``source="relaxed_stations"``:
            ``"gates"`` (the gates' own profile, after the order) or ``"centerline"``
            (on the relaxed centerline, before the stations, whose altitude the gates
            then carry by index). Default: ``"centerline"`` for ``"xyz"``, ``"gates"``
            for ``"xy"``. An explicit value with a generator that provides ``z`` (no
            elevation runs) raises ``ValueError``.
        registry: The plugin registry the generator is looked up in (default
            ``track_gen.plugin.registry``).

    Raises:
        ValueError: an unknown generator name, an invalid selection or argument
            (``gates=`` with an anchors source, ``gates=`` and a stations selection
            setting another count, ``order=`` with a legacy gate generator, an explicit
            elevation, ``z_base`` or ``gate_width`` for a generator that provides them,
            a nonzero ``z_base`` with another elevation than flat, a centerline
            argument the source does not read, ``validity.gate_default`` params setting
            another ``gate_radius`` / ``gate_width``, ``gate_align``,
            ``on_invalid``).
        TypeError: a selection of another type.
        track_gen.plugin.PluginLoadError: the generator belongs to a quarantined
            plugin.
    """
    from ..plugin import registry as _default_registry  # noqa: PLC0415
    from ..plugin.stage import _is_stage  # noqa: PLC0415
    from ..stages.gates import _default_order, _preferred  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    if on_invalid not in ("keep", "poison"):
        raise ValueError(f"on_invalid must be 'keep' or 'poison', got {on_invalid!r}")
    if gate_align not in _GATE_ALIGNS:
        raise ValueError("gate_align must be one of {'yaw_only', 'full_tangent'}, got "
                         f"{gate_align!r}")
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {list(_SOURCES)}, got {source!r}")
    if spacing_metric not in ("xy", "xyz"):
        raise ValueError(f"spacing_metric must be 'xy' or 'xyz', got {spacing_metric!r}")
    if elevation_on not in (None, "centerline", "gates"):
        raise ValueError(f"elevation_on must be None, 'centerline' or 'gates', got "
                         f"{elevation_on!r}")
    hw = float(half_width)
    # -- the generator and its source --------------------------------------------------
    if isinstance(generator, StageRef):
        if generator.kind not in ("gate_generator", "generator"):
            raise ValueError(f"generator={generator!r}: expected a StageRef of kind "
                             f"'gate_generator' or 'generator'")
        if generator_params is not None:
            raise ValueError("pass generator_params inside the StageRef")
        name = generator.name
        if generator.kind == "gate_generator":
            if source not in ("auto", "anchors"):
                raise ValueError(f"source={source!r}: a gate generator StageRef is an "
                                 f"anchors source")
            src = "anchors"
        else:
            src = "relaxed_stations" if source == "relaxed_stations" else "stations"
            if source == "anchors":
                raise ValueError("source='anchors': a StageRef of kind 'generator' is a "
                                 "centerline (stations) source")
        gen_in = generator
    elif isinstance(generator, str) and generator:
        name = generator
        gen_in = None
        if source == "auto":
            if _lookup(reg, "gate_generator", name) is not None:
                src = "anchors"
            elif _lookup(reg, "generator", name) is not None:
                src = "stations"
            else:
                raise ValueError(
                    f"presets.gates: unknown generator {name!r}; gate generators: "
                    f"{reg.available('gate_generator')}; track generators (gates through "
                    f"stations): {reg.available('generator')}")
        else:
            src = source
    else:
        raise TypeError(f"generator must be a generator name or a StageRef, got "
                        f"{type(generator).__name__}")
    _check_centerline_args(src, name, dict(n_max=n_max, num_points=num_points,
                                           half_width=half_width, spacing=spacing,
                                           relax=relax, width=width))
    anchors = src == "anchors"
    if anchors and spacing_metric != "xy":
        raise ValueError(f"spacing_metric= places stations; gate generator {name!r} "
                         f"places its own gates")
    if anchors and elevation_on is not None:
        raise ValueError(f"elevation_on= places the elevation of a relaxed_stations "
                         f"source; gate generator {name!r} has no centerline")
    relaxed = src == "relaxed_stations"
    explicit_on = elevation_on
    if elevation_on is None:
        elevation_on = "centerline" if spacing_metric == "xyz" and relaxed else "gates"
    if anchors:
        if gates is not None:
            raise ValueError("gates= sets the station count of a centerline source; an "
                             "anchors generator decides its own (capacity)")
        spec_cls = reg.get("gate_generator", name)
        gen_ref = gen_in if gen_in is not None else StageRef(
            "gate_generator", name, params=generator_params, id="generator")
        if gen_ref.id is None:
            gen_ref = dataclasses.replace(gen_ref, id="generator")
        stage = _is_stage(spec_cls)
        provides = frozenset(spec_cls.provides) if stage else frozenset()
        G = 32 if max_gates is None else max_gates
    else:
        n_st = 12 if gates is None else gates
        provides = frozenset()
        G = n_st if max_gates is None else max_gates
        stage = True
        spec_cls = None
        # A centerline generator's provided attributes (roadmap PR13a) and raw plugin
        # attributes: z on the loop is carried by the stations to the gates (carry.z),
        # the attributes by carry.attributes.
        gen_name = gen_in.name if gen_in is not None else name
        gen_provides = _provides(reg, "generator", gen_name)
        gen_cls = _lookup(reg, "generator", gen_name)
        gen_attrs = []
        if gen_cls is not None and _is_stage(gen_cls):
            from ..stages.channels import is_plugin_channel  # noqa: PLC0415

            gen_attrs = [ch for ch in dict(gen_cls.writes).values()
                         if is_plugin_channel(ch) and ch.startswith("raw.")]
        if explicit_on is not None and "z" in gen_provides:
            raise ValueError(f"elevation_on={explicit_on!r} places the elevation stage; "
                             f"generator {gen_name!r} provides its own altitudes (z), so "
                             f"no elevation runs: leave elevation_on at its default "
                             f"(None)")
        if explicit_on == "centerline" and not relaxed:
            raise ValueError("elevation_on='centerline' runs the elevation on the relaxed "
                             "centerline: source='relaxed_stations' (a stations source "
                             "has no centerline sub-pipeline)")
        if spacing_metric == "xyz" and relaxed and elevation_on == "gates" and \
                "z" not in gen_provides:
            raise ValueError("spacing_metric='xyz' needs the loop's altitude before the "
                             "stations are placed: elevation_on='centerline' (the "
                             "default for 'xyz')")
    # -- order / normalize ---------------------------------------------------------------
    upstream, stations_ref = (), None
    normalize = False
    if anchors and not stage:
        if order is not None:
            raise ValueError(f"order=: gate generator {name!r} is a legacy "
                             f"GateGeneratorSpec that orders its own gates; choose its "
                             f"ordering with GateGenConfig(gate_ordering=...)")
        order_ref = None
    else:
        explicit_order = order is not None
        if order is None:
            order = _default_order(spec_cls, reg) if anchors else "raw"
        order_ref = _select("order", "order", order, hw)
        normalize = anchors and _preferred(spec_cls).get("normalize") == "extent"
    elev_sel_flat = elevation is None or _is_flat(
        _select("elevation", "elevation", elevation, hw))
    loop_z = False                  # the stations carry the loop's altitude (carry.z)
    if not anchors:
        if spacing_metric == "xyz" and not relaxed and not elev_sel_flat and \
                "z" not in gen_provides:
            raise ValueError("spacing_metric='xyz' with an elevation: the elevation must "
                             "run on the centerline before the stations, which needs "
                             "source='relaxed_stations' (a stations source places them "
                             "on the raw loop)")
        loop_z = "z" in gen_provides or (relaxed and elevation_on == "centerline"
                                         and not elev_sel_flat)
        if loop_z and explicit_order and order_ref.name != "raw":
            what = (f"generator {gen_name!r} provides z" if "z" in gen_provides else
                    "the centerline elevation's z")
            raise ValueError(f"{what}, carried to the gates by index: order='raw' (the "
                             f"stations default)")
        if relaxed:
            track_spec = track(num_envs=num_envs, device=device, n_max=n_max, scale=scale,
                               generator=gen_in if gen_in is not None else name,
                               generator_params=generator_params, num_points=num_points,
                               half_width=half_width, spacing=spacing, relax=relax,
                               width=width, registry=reg)
            ids = [r.slot_id for r in track_spec.stages]
            gen_ref, *upstream = track_spec.stages[:ids.index("relax") + 1]
            if loop_z and "z" not in gen_provides:
                # The elevation on the relaxed centerline (its cl.* ports kept), after
                # the plan-view frames it reads.
                upstream += [StageRef("frames", "default"),
                             _select("elevation", "elevation", elevation, hw)]
            upstream = tuple(upstream)
        elif gen_in is not None:
            gen_ref = gen_in if gen_in.capacity else dataclasses.replace(
                gen_in, capacity={"raw": int(num_points)})
        else:
            gen_ref = StageRef("generator", name, params=generator_params,
                               capacity={"raw": int(num_points)})
        default_st = "relaxed" if relaxed else "raw"
        stations_ref = _select("stations", "stations",
                               stations if stations is not None else default_st, hw)
        # The station count: gates= fills a stations selection that does not set it;
        # one that does sets the count (and so max_gates), and must agree with gates=.
        n_sel = _param(stations_ref, "gates")
        st_cls = _lookup(reg, "stations", stations_ref.name)
        st_fields = getattr(getattr(st_cls, "Params", None), "__dataclass_fields__", {})
        if n_sel is None:
            if "gates" in st_fields:
                stations_ref = _with_params(stations_ref, {"gates": n_st})
        elif gates is not None and n_sel != gates:
            raise ValueError(f"stations: stations.{stations_ref.name} sets gates="
                             f"{n_sel!r} but gates={gates!r}; set the station count once")
        elif max_gates is None and isinstance(n_sel, int) and not isinstance(n_sel, bool):
            G = n_sel
        # spacing_metric: filled into a stations Params with the field (as gates=).
        m_sel = _param(stations_ref, "spacing_metric")
        if "spacing_metric" in st_fields:
            if m_sel is None:
                if spacing_metric != "xy":
                    stations_ref = _with_params(stations_ref,
                                                {"spacing_metric": spacing_metric})
            elif m_sel != spacing_metric:
                raise ValueError(f"stations: stations.{stations_ref.name} sets "
                                 f"spacing_metric={m_sel!r} but spacing_metric="
                                 f"{spacing_metric!r}; set it once")
        elif spacing_metric != "xy":
            raise ValueError(f"spacing_metric={spacing_metric!r}: stations "
                             f"{stations_ref.name!r} has no spacing_metric parameter")
        st_builtin = st_cls is not None and stations_ref.name in ("raw", "relaxed")
        st_writes = dict(getattr(st_cls, "writes", {}) or {}) if st_cls is not None \
            else {}
        if not st_builtin and st_cls is not None:
            if "z" in gen_provides:
                raise ValueError(f"stations {stations_ref.name!r} carries no centerline "
                                 f"channels (plugin carriers are deferred); generator "
                                 f"{gen_name!r} provides z")
            loop_z = "anchors.z" in st_writes.values()
            gen_attrs = [ch for ch in st_writes.values() if ch.startswith("anchors.")
                         and ch not in ("anchors.xy", "anchors.count", "anchors.z")]
        if spacing_metric == "xyz" and st_builtin and not loop_z:
            fix = ("a non-flat elevation (it runs on the relaxed centerline)" if relaxed
                   else "source='relaxed_stations' with a non-flat elevation")
            raise ValueError(f"spacing_metric='xyz' measures 3D arc length along the "
                             f"loop, which carries no altitude here (a flat elevation and "
                             f"generator {gen_name!r} provides no z): use a generator "
                             f"that provides z (raw.z), or {fix}")
        if loop_z:
            provides = frozenset({"z"})
        carry_attrs = bool(gen_attrs) and order_ref.name == "raw"
    elif stations is not None:
        raise ValueError("stations= selects the stations of a centerline source; an "
                         "anchors generator writes its own anchors")
    else:
        carry_attrs = False
    if anchors and stage:
        from ..stages.channels import is_plugin_channel  # noqa: PLC0415

        carry_attrs = order_ref is not None and order_ref.name == "raw" and any(
            is_plugin_channel(ch) and ch.startswith("anchors.")
            for ch in dict(spec_cls.writes).values())
    # -- elevation / provided attributes -------------------------------------------------
    elev_ref = StageRef("elevation", "flat") if elevation is None else \
        _select("elevation", "elevation", elevation, hw)
    flat = (elev_ref.kind, elev_ref.name) == ("elevation", "flat")
    if not anchors and "z" in gen_provides and not _is_flat(elev_ref):
        raise ValueError(f"presets.gates(elevation={elevation!r}): generator {gen_name!r} "
                         f"provides its own altitudes (z); leave elevation at its "
                         f"default ('flat')")
    if float(z_base) != 0.0:
        if "z" in provides:
            who = f"gate generator {name!r}" if anchors else (
                f"generator {gen_name!r}" if "z" in gen_provides else
                "the centerline elevation")
            raise ValueError(f"presets.gates(z_base={z_base!r}): {who} provides its own "
                             f"altitudes (z); leave z_base=0.0")
        if not flat:
            raise ValueError(f"z_base={z_base!r} is the flat elevation's altitude "
                             f"(elevation='flat'); elevation.{elev_ref.name} takes its "
                             f"own: pass it in the elevation's params")
        v = _param(elev_ref, "z_base")
        if v is not None and float(v) != float(z_base):
            raise ValueError(f"elevation: elevation.flat sets z_base={v!r} but z_base="
                             f"{z_base!r}; set the flat altitude once")
        elev_ref = _with_params(elev_ref, {"z_base": float(z_base)})
    if anchors and "z" in provides and not _is_flat(elev_ref):
        raise ValueError(f"presets.gates(elevation={elevation!r}): gate generator "
                         f"{name!r} provides its own altitudes (z); leave elevation at its "
                         f"default ('flat')")
    posed = "half_size" in provides
    if posed and float(gate_width) != 0.0:
        raise ValueError(f"gate_width: gate generator {name!r} provides per-gate "
                         f"half_size")
    # -- checks --------------------------------------------------------------------------
    if isinstance(checks, (str, StageRef)) or _is_pair(checks):
        raise TypeError(f"checks must be a sequence of selections, got the single "
                        f"selection {checks!r}; e.g. checks=('gate_default',) or "
                        f"checks=('gate_default', 'acme.max_height')")
    check_refs = []
    for k, sel in enumerate(checks):
        ref = _select("checks", "validity", sel, hw)
        if ref.id is None and k > 0:        # the first check is slot "validity"
            ref = dataclasses.replace(ref, id=ref.name)
        check_refs.append(ref)
    if not check_refs or check_refs[0].name != "gate_default":
        first = check_refs[0].name if check_refs else None
        raise ValueError(f"checks must start with 'gate_default' (validity.gate_default "
                         f"starts valid.fail; plugin checks run after it): "
                         f"checks=('gate_default'{'' if first is None else f', {first!r}'})"
                         f", got {tuple(checks)!r}")
    ids = [r.slot_id for r in check_refs]
    dup = next((i for i in ids if ids.count(i) > 1), None)
    if dup is not None:
        raise ValueError(f"checks: duplicate stage id {dup!r}; give one an explicit "
                         f"StageRef(id=...)")
    bar_width = 0.0 if posed else gate_width
    vfill = {"min_gates": min_gates, "gate_radius": gate_radius, "gate_width": bar_width,
             "z_valid_grade": z_valid_grade}
    if posed:
        vfill["check_crossing"] = "always"
    validity_ref = _with_gate_validity(check_refs[0], vfill,
                                       posed_by=name if posed else None)
    stages = _gate_stages(
        gen_ref, provides=provides, upstream=upstream, stations_ref=stations_ref,
        order_ref=order_ref, normalize=normalize, gate_radius=gate_radius,
        gate_solve_iters=gate_solve_iters, gate_width=bar_width, gate_align=gate_align,
        elevation_ref=elev_ref, min_gates=min_gates, z_valid_grade=z_valid_grade,
        validity_ref=validity_ref, checks=check_refs[1:], carry_attributes=carry_attrs)
    return PipelineSpec(batch=BatchSpec(num_envs=num_envs, device=device, n_max=n_max,
                                        scale=scale, on_invalid=on_invalid, max_gates=G),
                        stages=stages)
