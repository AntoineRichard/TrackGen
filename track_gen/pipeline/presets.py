"""Preset pipelines: :func:`track` builds the default track :class:`PipelineSpec` with
one argument per replaceable stage (preview, roadmap PR10b).

A preset is pure data: it never touches the registry, so an unknown stage name fails
at :func:`track_gen.pipeline.build` (or ``TrackGenerator`` construction) with the
registry's message. Edit the result with the :class:`PipelineSpec` operations::

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

__all__ = ["StageSel", "track"]

#: A stage selection: a name, a ``(name, params)`` pair or a :class:`StageRef`.
StageSel = Union[str, tuple, StageRef]

# The built-ins that bake half_width as a host scalar (until roadmap PR13): the preset
# gives them the pipeline's half_width unless their params set one.
_HALF_WIDTH_STAGES = {("relax", "xpbd"), ("width", "constant"), ("offset", "default"),
                      ("validity", "default")}


def _with_half_width(kind: str, name: str, params, half_width: float):
    if (kind, name) not in _HALF_WIDTH_STAGES:
        return params
    if params is None:
        return {"half_width": half_width}
    if isinstance(params, _MappingABC) and "half_width" not in params:
        return {**dict(params), "half_width": half_width}
    return params                                  # a Params instance: taken as is


def _check_half_width(slot: str, ref: StageRef, half_width: float) -> StageRef:
    """A selection whose params set ``half_width`` must agree with the preset's (one
    value until roadmap PR13): say so here rather than as the planner's agreement
    error."""
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
                         f"presets.track (the stages that bake half_width share one "
                         f"value until roadmap PR13)")
    return ref


def _is_pair(sel) -> bool:
    """``(name, params)``: a 2-tuple of a name and a Params instance / mapping / None."""
    return isinstance(sel, tuple) and len(sel) == 2 and isinstance(sel[0], str) \
        and not isinstance(sel[1], (str, tuple, StageRef))


def _select(slot: str, kind: str, sel, half_width: float) -> StageRef:
    """``sel`` (a selection form) as a StageRef of ``kind`` (a StageRef keeps its id)."""
    if isinstance(sel, StageRef):
        if sel.kind != kind:
            raise ValueError(f"{slot}={sel!r}: expected a StageRef of kind {kind!r}")
        return _check_half_width(slot, dataclasses.replace(
            sel, params=_with_half_width(kind, sel.name, sel.params, half_width)),
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
        kind, name, params=_with_half_width(kind, name, params, half_width)), half_width)


def track(*, num_envs: int, device: str, n_max: int = 384, scale: float = 1.0,
          generator: "str | StageRef" = "bezier", generator_params=None,
          num_points: int = 256, half_width: float = 0.1,
          spacing: "float | None" = None, relax: "StageSel | None" = "xpbd",
          width: StageSel = "constant", elevation: "StageSel | None" = "flat",
          checks: Sequence[StageSel] = ("default",),
          on_invalid: str = "keep", z_base: float = 0.0) -> PipelineSpec:
    """The track pipeline: generator -> resample -> relax -> uniform resample -> width
    -> frames -> offset -> elevation -> checks -> lift.

    ``presets.track(num_envs=E, device=d)`` equals the spec of
    ``TrackGenerator(TrackGenConfig(num_envs=E, device=d))`` (same stages, equal
    params; byte-identical Tracks).

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
        half_width: The track half-width, given to the built-ins that bake it
            (``relax.xpbd``, ``width.constant``, ``offset.default``,
            ``validity.default``) unless their params set it (until roadmap PR13).
        spacing: The constant-spacing step (default ``0.6 * half_width``).
        relax: The ``relax`` stage (``None``: ``relax.none``).
        width: The ``width`` stage.
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

    Raises:
        ValueError: an invalid selection (wrong StageRef kind, ``width=None``,
            ``checks`` not starting with ``"default"``, a duplicate check id, an unknown ``on_invalid``, ``generator_params`` with a
            generator StageRef, params setting a ``half_width`` other than
            ``half_width``).
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
    hw = float(half_width)
    step = 0.6 * hw if spacing is None else spacing
    relax_ref = StageRef("relax", "none") if relax is None else \
        _select("relax", "relax", relax, hw)
    if width is None:
        raise ValueError("width=None: a width stage must write cl.half_width")
    width_ref = _select("width", "width", width, hw)
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
    stages = (
        gen,
        StageRef("resample", "constant_spacing", params={"spacing": step}),
        relax_ref,
        StageRef("resample", "uniform", id="uniform"),
        width_ref,
        StageRef("frames", "default"),
        StageRef("offset", "default", params={"half_width": hw}),
        elev_ref,
        *check_refs,
        StageRef("lift", "flat", params={"z_base": float(z_base)}) if flat
        else StageRef("lift", "elevated"),
    )
    return PipelineSpec(batch=BatchSpec(num_envs=num_envs, device=device, n_max=n_max,
                                        scale=scale, on_invalid=on_invalid),
                        stages=stages)
