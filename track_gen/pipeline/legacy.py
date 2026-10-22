"""``TrackGenConfig`` <-> ``PipelineSpec``: the default track stage list.

:func:`to_pipeline_spec` is what ``TrackGenerator`` builds. The stage list reproduces
the 0.2.0.dev2 composition (``warp_pipeline._run_pipeline`` + ``inflate_warp``) launch
for launch.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping as _MappingABC

from .spec import BatchSpec, PipelineSpec, StageRef

__all__ = ["generator_config", "to_pipeline_spec"]


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
            and params.half_width != cfg.half_width:
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
