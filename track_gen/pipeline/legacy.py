"""``TrackGenConfig`` / ``GateGenConfig`` <-> ``PipelineSpec``: the default stage lists.

:func:`to_pipeline_spec` is what ``TrackGenerator`` builds. The stage list reproduces
the 0.2.0.dev2 composition (``warp_pipeline._run_pipeline`` + ``inflate_warp``) launch
for launch. :func:`to_gate_pipeline_spec` is what ``GateGenerator`` builds (roadmap
PR11): the gate stage list reproduces the 0.2.0.dev3 ``warp_gate._run_gate_pipeline``
(the same kernels, dims and scalars).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping as _MappingABC

from .spec import BatchSpec, PipelineSpec, StageRef

__all__ = ["gate_generator_config", "generator_config", "to_gate_pipeline_spec",
           "to_pipeline_spec"]


def to_pipeline_spec(config, *, registry=None) -> PipelineSpec:
    """The default :class:`~track_gen.pipeline.PipelineSpec` of a ``TrackGenConfig``.

    Ids, in order: ``generator``, ``resample`` (``resample.constant_spacing``),
    ``relax`` (``relax.<relax_solver>``, or ``relax.none`` when ``relax_enable=False``),
    ``uniform`` (``resample.uniform``), ``width`` (``width.constant``), ``frames``,
    ``offset``, ``elevation`` (``elevation.<z_profile>``), ``validity`` and ``lift``
    (``lift.flat`` on the flat profile, else ``lift.elevated``). Each stage's params
    are its Params class's ``from_legacy(config)`` (the flat field names); the
    generator's are its resolved ``generator_params``. ``relax_solver`` and
    ``z_profile`` name any stage registered under ``relax`` / ``elevation``; a
    ``half_width`` field of their Params is ``config.half_width`` (every stage that
    bakes it shares one value until roadmap PR13).

    Args:
        config: A ``TrackGenConfig``.
        registry: The plugin registry the generator, ``relax_solver`` and
            ``z_profile`` are resolved through (default ``track_gen.plugin.registry``).

    Raises:
        ValueError: an unknown or oracle-only ``relax_solver``, an unknown
            ``z_profile``, ``smooth_finish=True`` (as ``TrackGenerator``), or the
            generator's params are invalid.
        track_gen.plugin.PluginLoadError: ``relax_solver`` / ``z_profile`` names a
            stage of a quarantined plugin.
    """
    from .._src.generator_binding import bind_generator  # noqa: PLC0415
    from ..plugin import registry as _default_registry  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    binding = bind_generator("generator", config, registry=reg)
    return _to_pipeline_spec(binding.run_config, binding, registry=reg)


# The deprecated relaxation backends that only exist as test oracles.
_ORACLE_SOLVERS = ("energy", "tp_sobolev")


def _legacy_params(cls, cfg, what: str):
    """``cls.Params.from_legacy(cfg)``, with a ``half_width`` field set to
    ``cfg.half_width``: the planner requires every stage that has one to agree, and a
    ``from_legacy`` that does not map it would keep the class default."""
    from ..plugin.params import _coerce  # noqa: PLC0415

    params = _coerce(cls.Params, None, what=what, cfg=cfg)
    if "half_width" in getattr(type(params), "__dataclass_fields__", {}) \
            and hasattr(cfg, "half_width") and params.half_width != cfg.half_width:
        params = dataclasses.replace(params, half_width=cfg.half_width)
    return params


def _to_pipeline_spec(cfg, binding, registry=None) -> PipelineSpec:
    """The default spec of the bound run config ``cfg`` (``binding.run_config``);
    ``relax_solver`` / ``z_profile`` are resolved through ``registry`` (default
    ``track_gen.plugin.registry``)."""
    from ..plugin import registry as _default_registry  # noqa: PLC0415
    from ..stages import params as sp  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    solver = cfg.relax_solver
    if solver in _ORACLE_SOLVERS:
        raise ValueError(
            f"TrackGenConfig.relax_solver={solver!r} is an oracle-only backend "
            f"(tests/_oracle), not a relax stage; TrackGenerator runs a registered relax "
            f"stage: {reg.available('relax')}")
    try:
        relax_cls = reg.get("relax", solver)
    except ValueError as exc:                 # PluginLoadError (quarantined) passes
        raise ValueError(f"TrackGenConfig.relax_solver: {exc}") from None
    if cfg.smooth_finish:
        raise ValueError(
            "TrackGenerator does not support smooth_finish=True; "
            "set smooth_finish=False."
        )
    profile = cfg.z_profile
    try:
        elevation_cls = reg.get("elevation", profile)
    except ValueError as exc:
        raise ValueError(f"TrackGenConfig.z_profile: {exc}") from None
    batch = BatchSpec(num_envs=int(cfg.num_envs), device=str(cfg.device),
                      n_max=int(cfg.N_max), closed=True, scale=cfg.scale)
    if not cfg.relax_enable:
        relax = StageRef("relax", "none")
    else:
        relax = StageRef("relax", solver, params=_legacy_params(
            relax_cls, cfg, f"relax stage {solver!r}"))
    if profile == "flat":
        elevation = StageRef("elevation", "flat")
    else:
        elevation = StageRef("elevation", profile, params=_legacy_params(
            elevation_cls, cfg, f"elevation stage {profile!r}"))
    stages = (
        StageRef("generator", binding.name, params=binding.params,
                 capacity={"raw": int(cfg.num_points)}),
        StageRef("resample", "constant_spacing",
                 params=sp.ConstantSpacingParams.from_legacy(cfg)),
        relax,
        StageRef("resample", "uniform", id="uniform"),
        StageRef("width", "constant", params=sp.WidthConstantParams.from_legacy(cfg)),
        StageRef("frames", "default"),
        StageRef("offset", "default", params=sp.OffsetParams.from_legacy(cfg)),
        elevation,
        StageRef("validity", "default", params=sp.ValidityParams.from_legacy(cfg)),
        StageRef("lift", "flat", params=sp.LiftFlatParams.from_legacy(cfg))
        if profile == "flat" else StageRef("lift", "elevated"),
    )
    return PipelineSpec(batch=batch, stages=stages)


def generator_config(spec: PipelineSpec):
    """The ``TrackGenConfig`` ``build`` binds the generator with: the batch
    (``num_envs``, ``device``, ``N_max``, ``scale``), ``num_points =
    capacity.get("raw", 256)`` and ``generator_params`` from the generator's
    ``StageRef``. Enough for the built-ins and Stage generators; a ``GeneratorSpec``
    that reads other config fields needs ``TrackGenerator``."""
    from .._src.types import TrackGenConfig  # noqa: PLC0415

    ref, batch = spec.stages[0], spec.batch
    params = ref.params
    if isinstance(params, _MappingABC):
        params = dict(params)               # the config is deep-copied: no mappingproxy
    return TrackGenConfig(generator=ref.name, device=batch.device,
                          num_envs=batch.num_envs, N_max=batch.n_max, scale=batch.scale,
                          num_points=int(ref.capacity.get("raw", 256)),
                          generator_params=params)


# ---------------------------------------------------------------------------------------
# The gate pipeline (roadmap PR11)
# ---------------------------------------------------------------------------------------

def to_gate_pipeline_spec(config, *, registry=None) -> PipelineSpec:
    """The default gate :class:`~track_gen.pipeline.PipelineSpec` of a ``GateGenConfig``
    (what ``GateGenerator(config)`` builds).

    Ids, in order: ``generator`` (``gate_generator.<generator>``), ``order``
    (``order.<gate_ordering>``; ``gate_ordering=None`` is the generator's default
    order, ``stages.gates._default_order``: its ``preferred["order"]`` when it accepts
    it, else ``"ccw"`` when it accepts it, else the first order it accepts; ``"ccw"`` for
    the five built-ins), ``normalize`` (``normalize.extent``, for a generator
    whose ``preferred["normalize"]`` is ``"extent"``: polar, checkpoint, voronoi), the
    ``carry_<attr>`` of each attribute the generator provides, ``gate_relax``
    (``gate_relax.spheres``, or ``gate_relax.none`` when ``gate_solve_iters == 0`` or the
    float32 ``2 * gate_radius`` is 0), ``gate_arclen`` (unless the profile is flat),
    ``elevation`` (``elevation.<z_profile>``; both dropped when the generator provides
    ``z``), ``lift`` (``lift.gates``), ``tangents`` (``tangents.central``, dropped when it
    provides ``tangent``), ``gate_frames`` and ``validity`` (``validity.gate_default``).
    A legacy ``GateGeneratorSpec`` orders its own gates: no order / normalize stage; it
    reads the resolved name from its run config (never ``None``).
    ``BatchSpec(n_max=384, max_gates=config.max_gates, scale=config.scale)``.

    Raises:
        ValueError: an unknown generator, ``gate_ordering`` (not a registered order, or
            not one the generator supports) or ``z_profile``; ``min_gates`` /
            ``max_gates`` incompatible with the generator's capacity; an explicit
            ``z_profile`` / ``z_base`` / ``gate_width`` for a generator that provides
            them; invalid generator params; an elevation plugin whose
            ``Params.from_legacy`` reads a field ``GateGenConfig`` lacks.
        track_gen.plugin.PluginLoadError: a name of a quarantined plugin.
    """
    from .._src.generator_binding import bind_generator  # noqa: PLC0415
    from ..plugin import registry as _default_registry  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    binding = bind_generator("gate_generator", config, registry=reg)
    return _to_gate_pipeline_spec(binding.run_config, binding, registry=reg)


def _to_gate_pipeline_spec(cfg, binding, registry=None) -> PipelineSpec:
    """The default gate spec of the bound run config ``cfg`` (``binding.run_config``),
    with 0.2.0.dev3's checks and messages in its order (ordering, capacity), then the
    z-profile and provided-attribute rules. ``cfg.gate_ordering=None`` is resolved here
    to the generator's default order and written onto ``cfg`` (the binding's own
    snapshot), so every later reader of the run config (a legacy spec's ``max_gates`` /
    ``alloc_scratch`` / ``generate``, an elevation's ``from_legacy``,
    ``GateGenerator._config``) sees the name."""
    from ..plugin import registry as _default_registry  # noqa: PLC0415
    from ..plugin.params import _static_view  # noqa: PLC0415
    from ..plugin.stage import _is_stage  # noqa: PLC0415
    from ..stages.gates import _default_order, _orderings_of, _preferred  # noqa: PLC0415
    from .planner import _check_capacity  # noqa: PLC0415
    from .presets import _gate_stages  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    spec = binding.spec
    stage = _is_stage(spec)
    if cfg.gate_ordering is None:
        cfg.gate_ordering = _default_order(spec, reg)
    name, ordering = cfg.generator, cfg.gate_ordering
    # 1. the ordering
    if stage:
        try:
            reg.get("order", ordering)
        except ValueError as exc:            # PluginLoadError (quarantined) passes
            from ..plugin import PluginLoadError  # noqa: PLC0415

            if isinstance(exc, PluginLoadError):
                raise
            raise ValueError(f"GateGenConfig.gate_ordering: {exc}") from None
    supported = _orderings_of(spec, reg)
    if ordering not in supported:
        preferred = _preferred(spec).get("order")
        hint = f" (its preferred order is {preferred!r})" if preferred else ""
        raise ValueError(f"gate generator {name!r} does not support gate_ordering "
                         f"{ordering!r}; supported: {supported}{hint}")
    # 2. the capacity
    if stage:
        A = _check_capacity(spec, spec().capacity(_static_view(binding.params),
                                                  binding.layout))["gates"]
    else:
        A = int(spec.max_gates(cfg))
    if A < int(cfg.min_gates):
        raise ValueError(f"GateGenConfig.min_gates={cfg.min_gates!r} is too large for "
                         f"gate generator {name!r}; generator can produce at most {A} "
                         f"gates.")
    if A > int(cfg.max_gates):
        raise ValueError(f"GateGenConfig.max_gates={cfg.max_gates!r} is too small for "
                         f"gate generator {name!r}; required max_gates {A}.")
    # 3. the elevation and the provided attributes
    profile = cfg.z_profile
    try:
        elevation_cls = reg.get("elevation", profile)
    except ValueError as exc:
        from ..plugin import PluginLoadError  # noqa: PLC0415

        if isinstance(exc, PluginLoadError):
            raise
        raise ValueError(f"GateGenConfig.z_profile: {exc}") from None
    provides = frozenset(spec.provides) if stage else frozenset()
    if "z" in provides and (profile != "flat" or float(cfg.z_base) != 0.0):
        field = f"z_profile={profile!r}" if profile != "flat" else f"z_base={cfg.z_base!r}"
        raise ValueError(f"GateGenConfig.{field}: gate generator {name!r} provides its own "
                         f"altitudes (z); leave z_profile='flat' and z_base=0.0")
    if "half_size" in provides and float(cfg.gate_width) != 0.0:
        raise ValueError(f"GateGenConfig.gate_width: gate generator {name!r} provides "
                         f"per-gate half_size")
    try:
        elevation_params = _legacy_params(elevation_cls, cfg,
                                          f"elevation stage {profile!r}")
    except AttributeError as exc:
        # A from_legacy written for tracks reading a TrackGenConfig-only field.
        missing = getattr(exc, "name", None) or str(exc)
        raise ValueError(
            f"GateGenConfig.z_profile={profile!r}: {elevation_cls.__name__}.Params."
            f"from_legacy() received a GateGenConfig and read {missing!r}; on gates only "
            f"the z_* fields are shared with TrackGenConfig (read others with getattr("
            f"config, name, default), or select the stage with presets.gates("
            f"elevation=...))") from exc
    elevation = StageRef("elevation", profile, params=elevation_params)
    # 4. the stage list
    stages = _gate_stages(
        StageRef("gate_generator", binding.name, params=binding.params, id="generator"),
        provides=provides,
        order_ref=StageRef("order", ordering) if stage else None,
        normalize=stage and _preferred(spec).get("normalize") == "extent",
        gate_radius=cfg.gate_radius, gate_solve_iters=cfg.gate_solve_iters,
        gate_width=cfg.gate_width, gate_align=cfg.gate_align, elevation_ref=elevation,
        min_gates=cfg.min_gates, z_valid_grade=cfg.z_valid_grade)
    batch = BatchSpec(num_envs=int(cfg.num_envs), device=str(cfg.device), n_max=384,
                      closed=True, scale=cfg.scale, max_gates=int(cfg.max_gates))
    return PipelineSpec(batch=batch, stages=stages)


def _params_of(ref):
    """A StageRef's params as a ``{field: value}`` mapping (``{}`` for ``None``)."""
    p = ref.params
    if p is None:
        return {}
    if isinstance(p, _MappingABC):
        return dict(p)
    return {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}


def gate_generator_config(spec: PipelineSpec, *, registry=None):
    """The ``GateGenConfig`` of a gate pipeline: what ``build`` binds a gate generator
    with, and ``GateGenerator._config`` for any gate spec. The batch (``num_envs``,
    ``device``, ``max_gates``, ``scale``), the generator name and params (a centerline
    source: ``generator_params=None``, its params live in its own binding),
    ``min_gates`` (``validity.gate_default``'s, clamped to ``max_gates``),
    ``gate_ordering`` (the order stage's name; a legacy generator: ``"ccw"`` if it
    supports it, else its first ordering), ``gate_width`` / ``gate_radius`` /
    ``gate_solve_iters`` / ``gate_align`` / ``z_valid_grade`` / ``z_profile`` from their
    stages; other fields default. A legacy ``GateGeneratorSpec`` in a PipelineSpec sees
    only these fields."""
    from .._src.types import GateGenConfig  # noqa: PLC0415

    batch = spec.batch
    G = int(batch.max_gates)
    first = spec.stages[0]
    by_kind: dict = {}
    for ref in spec.stages[1:]:
        by_kind.setdefault((ref.kind, ref.name), ref)
        by_kind.setdefault(ref.kind, ref)
    kw: dict = dict(generator=first.name, device=batch.device, num_envs=batch.num_envs,
                    max_gates=G, scale=batch.scale)
    if first.kind == "gate_generator":
        params = first.params
        if isinstance(params, _MappingABC):
            params = dict(params)
        kw["generator_params"] = params
    validity = by_kind.get(("validity", "gate_default"))
    vals = _params_of(validity) if validity is not None else {}
    kw["min_gates"] = max(2, min(int(vals.get("min_gates", 4)), G))
    for f in ("gate_radius", "gate_width", "z_valid_grade"):
        if f in vals:
            kw[f] = vals[f]
    relax = by_kind.get("gate_relax")
    if relax is not None:
        rp = _params_of(relax)
        kw["gate_solve_iters"] = int(rp.get("gate_solve_iters", 8)) \
            if relax.name == "spheres" else 0
        if "gate_radius" in rp and "gate_radius" not in kw:
            kw["gate_radius"] = rp["gate_radius"]
    frames = by_kind.get("gate_frames")
    if frames is not None:
        fp = _params_of(frames)
        if "gate_align" in fp:
            kw["gate_align"] = fp["gate_align"]
        if "gate_width" in fp and "gate_width" not in kw:
            kw["gate_width"] = fp["gate_width"]
    elevation = by_kind.get("elevation")
    if elevation is not None:
        kw["z_profile"] = elevation.name
    order = by_kind.get("order")
    if order is not None:
        kw["gate_ordering"] = order.name
    elif first.kind == "gate_generator":
        kw["gate_ordering"] = _legacy_ordering(first.name, registry)
    return GateGenConfig(**kw)


def _legacy_ordering(name: str, registry=None) -> str:
    """A gate generator without an order stage (a legacy spec): its default order
    (``stages.gates._default_order``: ``"ccw"`` if it supports it, else its first
    supported ordering; ``"ccw"`` when it cannot be looked up)."""
    from ..plugin import registry as _default_registry  # noqa: PLC0415
    from ..stages.gates import _default_order  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    try:
        spec = reg.get("gate_generator", name)
    except ValueError:
        return "ccw"
    return _default_order(spec, reg)


def _scratch_view(program):
    """A PR9 ``warp_pipeline._Scratch`` whose attributes ALIAS the Program's channel
    versions and stage scratch (``TrackGenerator._scratch``): the same field names and
    meanings, so ``_run_pipeline(..., scratch=gen._scratch)`` still runs on it. A slot
    whose stage is absent is ``None``."""
    from .._src.warp_pipeline import InflateScratch, RelaxScratch, _Scratch  # noqa: PLC0415

    ch = program._channel_or_none
    sc = program._scratch
    kinds = program._kinds

    def scratch(stage_id, name):
        return sc.get(stage_id, {}).get(name)

    relax = None
    if kinds.get("relax") == ("relax", "xpbd"):
        relax = RelaxScratch(
            relaxed=ch("cl.xy", "relax"), band=scratch("relax", "band"),
            L0=scratch("relax", "L0"), xpbd_db=scratch("relax", "xy.tmp0"),
            cheb_prev=scratch("relax", "xy.tmp1"),
            sep_cache_idx=scratch("relax", "sep_cache_idx"),
            sep_cache_count=scratch("relax", "sep_cache_count"),
            sep_cache_overflow=scratch("relax", "sep_cache_overflow"))
    has_uniform = kinds.get("uniform") == ("resample", "uniform")
    inflate = InflateScratch(
        area_a=scratch("offset", "area_a"), area_b=scratch("offset", "area_b"),
        kappa=ch("cl.kappa"), w=ch("cl.half_width"),
        out2=ch("band.outer2"), ctr2=ch("cl.xy", "uniform") if has_uniform else None,
        inn2=ch("band.inner2"), tan2=ch("cl.tangent2"), nrm2=ch("cl.normal2"),
        z=ch("cl.z"), knot_cum=scratch("elevation", "knot_cum"),
        knot_count=scratch("elevation", "knot_count"),
        knot_z=scratch("elevation", "knot_z"),
        fail=ch("valid.fail", "validity") if "validity" in program._stage_ids else None)
    has_resample = kinds.get("resample") == ("resample", "constant_spacing")
    return _Scratch(
        inflate=inflate, generator_spec=program._binding.pipeline_spec,
        gen=program._gen_scratch, relax=relax,
        gen_centerline=ch("raw.xy"), gen_valid=ch("gen.valid"),
        cs_center=ch("cl.xy", "resample") if has_resample else None,
        cs_seg=scratch("resample", "seg"), cs_s=scratch("resample", "s"),
        count=ch("cl.count"))
