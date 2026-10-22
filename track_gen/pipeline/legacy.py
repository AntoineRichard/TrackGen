"""``TrackGenConfig`` <-> ``PipelineSpec``: the default track stage list.

:func:`to_pipeline_spec` is what ``TrackGenerator`` builds. The stage list reproduces
the 0.2.0.dev2 composition (``warp_pipeline._run_pipeline`` + ``inflate_warp``) launch
for launch.
"""
from __future__ import annotations

from collections.abc import Mapping as _MappingABC

from .spec import BatchSpec, PipelineSpec, StageRef

__all__ = ["generator_config", "to_pipeline_spec"]


def to_pipeline_spec(config, *, registry=None) -> PipelineSpec:
    """The default :class:`~track_gen.pipeline.PipelineSpec` of a ``TrackGenConfig``.

    Ids, in order: ``generator``, ``resample`` (``resample.constant_spacing``),
    ``relax`` (``relax.xpbd``, or ``relax.none`` when ``relax_enable=False``),
    ``uniform`` (``resample.uniform``), ``width`` (``width.constant``), ``frames``,
    ``offset``, ``elevation`` (``elevation.<z_profile>``), ``validity`` and ``lift``
    (``lift.flat`` on the flat profile, else ``lift.elevated``). Each stage's params
    are its Params class's ``from_legacy(config)`` (the flat field names); the
    generator's are its resolved ``generator_params``.

    Args:
        config: A ``TrackGenConfig``.
        registry: The plugin registry the generator is resolved through (default
            ``track_gen.plugin.registry``).

    Raises:
        ValueError: ``relax_solver != "xpbd"`` or ``smooth_finish=True`` (as
            ``TrackGenerator``), or the generator's params are invalid.
    """
    from .._src.generator_binding import bind_generator  # noqa: PLC0415
    from ..plugin import registry as _default_registry  # noqa: PLC0415

    reg = _default_registry if registry is None else registry
    binding = bind_generator("generator", config, registry=reg)
    return _to_pipeline_spec(binding.run_config, binding)


def _to_pipeline_spec(cfg, binding) -> PipelineSpec:
    """The default spec of the bound run config ``cfg`` (``binding.run_config``)."""
    from ..stages import params as sp  # noqa: PLC0415

    if cfg.relax_solver != "xpbd":
        raise ValueError(
            f"TrackGenerator only supports relax_solver='xpbd'; "
            f"got {cfg.relax_solver!r}."
        )
    if cfg.smooth_finish:
        raise ValueError(
            "TrackGenerator does not support smooth_finish=True; "
            "set smooth_finish=False."
        )
    batch = BatchSpec(num_envs=int(cfg.num_envs), device=str(cfg.device),
                      n_max=int(cfg.N_max), closed=True, scale=cfg.scale)
    profile = str(cfg.z_profile)
    profiles = {
        "flat": None,
        "uniform": sp.ElevationUniformParams,
        "random_walk": sp.ElevationRandomWalkParams,
        "noise": sp.ElevationNoiseParams,
    }
    if profile not in profiles:
        raise ValueError(f"z_profile must be one of {{'flat', 'uniform', 'random_walk', "
                         f"'noise'}}, got {cfg.z_profile!r}")
    elevation = profiles[profile]
    stages = (
        StageRef("generator", binding.name, params=binding.params,
                 capacity={"raw": int(cfg.num_points)}),
        StageRef("resample", "constant_spacing",
                 params=sp.ConstantSpacingParams.from_legacy(cfg)),
        StageRef("relax", "xpbd", params=sp.XpbdParams.from_legacy(cfg))
        if cfg.relax_enable else StageRef("relax", "none"),
        StageRef("resample", "uniform", id="uniform"),
        StageRef("width", "constant", params=sp.WidthConstantParams.from_legacy(cfg)),
        StageRef("frames", "default"),
        StageRef("offset", "default", params=sp.OffsetParams.from_legacy(cfg)),
        StageRef("elevation", profile,
                 params=None if elevation is None else elevation.from_legacy(cfg)),
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
        knot_z=scratch("elevation", "knot_z"))
    has_resample = kinds.get("resample") == ("resample", "constant_spacing")
    return _Scratch(
        inflate=inflate, generator_spec=program._binding.pipeline_spec,
        gen=program._gen_scratch, relax=relax,
        gen_centerline=ch("raw.xy"), gen_valid=ch("gen.valid"),
        cs_center=ch("cl.xy", "resample") if has_resample else None,
        cs_seg=scratch("resample", "seg"), cs_s=scratch("resample", "s"),
        count=ch("cl.count"))
