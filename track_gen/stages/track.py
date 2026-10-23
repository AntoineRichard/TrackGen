"""The built-in stages of the track pipeline (roadmap PR10a).

Each stage wraps the EXISTING kernels and wrappers of ``warp_pipeline`` /
``warp_relax`` / ``warp_zprofile`` verbatim: the same launches, in the same order, with
the same scalars as the 0.2.0.dev2 composition (``warp_pipeline._run_pipeline`` +
``inflate_warp``), so a :class:`~track_gen.pipeline.Program` of the default stage list
is byte-identical to it. The wrappers are looked up on their module at every call (not
bound at import), so a test that wraps one sees every call.

The relax / width / elevation / validity stages (:data:`REGISTRY_STAGES`) are registered
in :data:`track_gen.plugin.registry` (:func:`track_gen.stages.register_builtins`); the
resample / frames / offset / lift stages (:data:`CATALOGUE_STAGES`) stay in the private
catalogue (:func:`track_gen.stages.register`).
"""
from __future__ import annotations

import types
from typing import Callable

import warp as wp

from .._src import warp_pipeline as _wpl
from .._src import warp_relax as _relax
from .._src import warp_zprofile as _zprof
from ..plugin.params import NoParams
from ..plugin.stage import CaptureClass, Stage
from . import params as _p

__all__ = ["BUILTIN_STAGES", "CATALOGUE_STAGES", "REGISTRY_STAGES", "TRACK_EXPORTS"]

# Track field -> the channels it may come from, in order of preference: the field's
# buffer is the FINAL version of the first one a stage writes. ``arclen`` / ``length``
# are the 3D tables of ``lift.elevated`` when it runs, else the plan-view ones of
# ``frames`` (a flat lift leaves them unchanged). ``valid`` is ``validity.default``'s
# own flag when that stage is also the last writer of ``valid.fail``, else it is derived
# by the Program's commit as ``fail_mask == 0`` (``valid.fail``); ``fail_mask`` is the
# final ``valid.fail``.
TRACK_EXPORTS = types.MappingProxyType({
    "outer": ("track.outer",),
    "center": ("track.center",),
    "inner": ("track.inner",),
    "tangent": ("track.tangent",),
    "normal": ("track.normal",),
    "arclen": ("track.arclen", "cl.arclen2"),
    "length": ("track.length", "cl.length2"),
    "winding": ("cl.winding",),
    "valid": ("track.valid", "valid.fail"),
    "count": ("track.count",),
    "truncated": ("cl.truncated",),
    "fail_mask": ("valid.fail",),
})


def _alloc_track(layout):
    """The pipeline's result :class:`~track_gen.Track`, allocated exactly as
    ``warp_pipeline._inflate_warp_alloc`` allocates it (``wp.empty`` except the
    zero-initialised ``truncated`` and ``generation``), plus the zeroed ``fail_mask``."""
    from .._src.types import Track  # noqa: PLC0415

    E, dev = int(layout.num_envs), str(layout.device)
    flat = E * int(layout.stride)
    return Track(
        outer=wp.empty(flat, dtype=wp.vec3f, device=dev),
        center=wp.empty(flat, dtype=wp.vec3f, device=dev),
        inner=wp.empty(flat, dtype=wp.vec3f, device=dev),
        tangent=wp.empty(flat, dtype=wp.vec3f, device=dev),
        normal=wp.empty(flat, dtype=wp.vec3f, device=dev),
        arclen=wp.empty(flat, dtype=wp.float32, device=dev),
        length=wp.empty(E, dtype=wp.float32, device=dev),
        valid=wp.empty(E, dtype=wp.int32, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
        winding=wp.empty(E, dtype=wp.float32, device=dev),
        truncated=wp.zeros(E, dtype=wp.int32, device=dev),
        generation=wp.zeros(E, dtype=wp.int32, device=dev),
        fail_mask=wp.zeros(E, dtype=wp.int32, device=dev),
    )


class _Builtin(Stage):
    """Base of the built-in non-generator stages (STATIC, no ``output``)."""

    capture = CaptureClass.STATIC
    output = None


# ---------------------------------------------------------------------------------------
# The generator slot (private: wraps the PR8 binding, never in the catalogue)
# ---------------------------------------------------------------------------------------

class _GeneratorSlot:
    """Stage 0: runs the bound generator exactly as ``_run_pipeline`` step 1 does.

    A Stage plugin (its ``_StageGenerator`` adapter) is built on the PR8 builder code
    path (``adapter.build``) bound to this slot's ``raw.xy`` / ``gen.valid`` versions and
    the Program's seed; a ``GeneratorSpec`` gets ``alloc_scratch(run_config)`` and its
    ``generate`` (with ``ctx=``, or inside ``ctx.scope()`` for a legacy spec).
    """

    kind = "generator"
    Params = NoParams
    reads = types.MappingProxyType({"seed": "seed"})
    writes = types.MappingProxyType({"xy": "raw.xy", "valid": "gen.valid"})

    def __init__(self, binding) -> None:
        self.binding = binding
        self.name = binding.name
        self.scratch = None            # the generator's private scratch (_scratch.gen)

    def build(self, b, params) -> Callable:
        from .._src.generator_binding import _StageGenerator  # noqa: PLC0415

        binding = self.binding
        spec, run_cfg = binding.pipeline_spec, binding.run_config
        seed = b.seed
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        if isinstance(spec, _StageGenerator):
            gen = spec.build(seed=seed, out_xy=xy, out_valid=valid)
        else:
            gen = spec.alloc_scratch(run_cfg)
        self.scratch = gen
        generate = spec.generate
        if getattr(spec, "accepts_ctx", False):
            def enqueue(ctx):
                generate(seed, run_cfg, out_centerline=xy, out_valid_wp=valid,
                         scratch=gen, ctx=ctx)
        else:
            def enqueue(ctx):
                # Legacy plugin: no ctx= keyword. Make ctx ambient so the _pipe helpers
                # it calls with ctx=None resolve to this call's stream.
                with ctx.scope():
                    generate(seed, run_cfg, out_centerline=xy, out_valid_wp=valid,
                             scratch=gen)
        return enqueue


# ---------------------------------------------------------------------------------------
# Resample / relax
# ---------------------------------------------------------------------------------------

class ConstantSpacing(_Builtin):
    """``resample.constant_spacing``: ``raw.xy`` -> ``cl.xy`` (constant arc step,
    NaN-padded to ``N_max``), ``cl.count``, ``cl.truncated``."""

    kind = "resample"
    name = "constant_spacing"
    Params = _p.ConstantSpacingParams
    reads = {"raw": "raw.xy"}
    writes = {"xy": "cl.xy", "count": "cl.count", "truncated": "cl.truncated"}

    def build(self, b, p):
        S = int(b.layout.stride)
        raw_xy = b.read("raw")
        xy = b.write("xy", wp.vec2f, per="cl")
        count = b.write("count", wp.int32)
        truncated = b.write("truncated", wp.int32)
        # Sized by the INPUT count (raw points) as well as N_max (PR9's cs_seg/cs_s), so
        # a config with num_points > N_max cannot overrun them.
        m = max(S, int(b.raw))
        seg = b.scratch("seg", wp.float32, b.E * m)
        s = b.scratch("s", wp.float32, b.E * (m + 1))
        spacing = float(p.spacing)

        def enqueue(ctx):
            _wpl.resample_constant_spacing(
                raw_xy, spacing, S, out_wp=xy, count_wp=count, seg_wp=seg, s_wp=s,
                truncated_wp=truncated, ctx=ctx)
        return enqueue


class Xpbd(_Builtin):
    """``relax.xpbd``: the XPBD relaxation, an in-place rewrite of ``cl.xy``."""

    kind = "relax"
    name = "xpbd"
    Params = _p.XpbdParams
    reads = {"xy": "cl.xy", "count": "cl.count"}
    writes = {"xy": "cl.xy"}

    def build(self, b, p):
        S = int(b.layout.stride)
        x_in, count = b.read("xy"), b.read("count")
        x_out = b.write("xy", wp.vec2f, per="cl")
        band = b.scratch("band", wp.int32, b.E)
        l0 = b.scratch("L0", wp.float32, b.E)
        db = b.temp_like("xy")
        # Chebyshev previous-iterate buffer: only when acceleration is on (PR9's rule).
        cheb_prev = b.temp_like("xy") if str(getattr(p, "relax_accel", "none")) == \
            "chebyshev" else None
        slots = max(0, int(getattr(p, "relax_sep_cache_slots", 0)))
        if slots > 0:
            cache_idx = b.scratch("sep_cache_idx", wp.int32, b.E * S * slots)
            cache_count = b.scratch("sep_cache_count", wp.int32, b.E * S)
            cache_overflow = b.scratch("sep_cache_overflow", wp.int32, 1)
        else:
            cache_idx = cache_count = cache_overflow = None

        def enqueue(ctx):
            _relax.band_l0_inplace(x_in, S, band, l0, count, p, ctx=ctx)
            _relax.xpbd_solve_inplace(
                x_in, x_out, db, band, l0, count, S, p,
                sep_cache_idx_wp=cache_idx, sep_cache_count_wp=cache_count,
                sep_cache_overflow_wp=cache_overflow, cheb_prev_wp=cheb_prev, ctx=ctx)
        return enqueue


class RelaxNone(_Builtin):
    """``relax.none``: no relaxation (``relax_enable=False``); the uniform resample then
    reads the constant-spacing ``cl.xy``. Keeps the ``relax`` slot replaceable."""

    kind = "relax"
    name = "none"

    def build(self, b, p):
        def enqueue(ctx):
            pass
        return enqueue


class Uniform(_Builtin):
    """``resample.uniform``: arc-length-uniform resample of ``cl.xy`` (in place)."""

    kind = "resample"
    name = "uniform"
    reads = {"xy": "cl.xy", "count": "cl.count"}
    writes = {"xy": "cl.xy"}

    def build(self, b, p):
        S = int(b.layout.stride)
        x_in, count = b.read("xy"), b.read("count")
        x_out = b.write("xy", wp.vec2f, per="cl")
        # Own scan scratch (PR9 shared the constant-spacing one): the scan writes
        # seg[b..b+cn) / s[es..es+cn] before the lookup reads exactly those slots.
        seg = b.scratch("seg", wp.float32, b.E * S)
        s = b.scratch("s", wp.float32, b.E * (S + 1))
        dev = str(x_in.device)

        def enqueue(ctx):
            _wpl.resample_uniform(x_in, x_out, S, count, seg_wp=seg, s_wp=s, device=dev,
                                  ctx=ctx)
            ctx.sync()              # debug-only (TRACKGEN_DEBUG_SYNC=1), as inflate_warp
        return enqueue


# ---------------------------------------------------------------------------------------
# Width / frames / offset
# ---------------------------------------------------------------------------------------

class WidthConstant(_Builtin):
    """``width.constant``: ``cl.half_width`` filled with ``half_width``."""

    kind = "width"
    name = "constant"
    Params = _p.WidthConstantParams
    writes = {"half_width": "cl.half_width"}

    def build(self, b, p):
        w = b.write("half_width", wp.float32, per="cl")
        n = b.E * int(b.layout.stride)
        hw = float(p.half_width)

        def enqueue(ctx):
            ctx.launch(_wpl._fill_f32_k, n, [w, hw])
        return enqueue


class FramesDefault(_Builtin):
    """``frames.default``: ``Track.count``, the plan-view frames, curvature, arc length
    (``cl.arclen2`` / ``cl.length2``) and winding of ``cl.xy``."""

    kind = "frames"
    name = "default"
    reads = {"xy": "cl.xy", "count": "cl.count"}
    writes = {"count_out": "track.count", "tangent": "cl.tangent2",
              "normal": "cl.normal2", "kappa": "cl.kappa", "arclen": "cl.arclen2",
              "length": "cl.length2", "winding": "cl.winding"}

    def build(self, b, p):
        xy, count = b.read("xy"), b.read("count")
        count_out = b.write("count_out", wp.int32)
        tan2 = b.write("tangent", wp.vec2f, per="cl")
        nrm2 = b.write("normal", wp.vec2f, per="cl")
        kappa = b.write("kappa", wp.float32, per="cl")
        arclen = b.write("arclen", wp.float32, per="cl")
        length = b.write("length", wp.float32)
        winding = b.write("winding", wp.float32)

        def enqueue(ctx):
            ctx.copy(count_out, count)
            _wpl.frame_curvature(xy, tan2, nrm2, kappa, count, ctx=ctx)
            _wpl._arclength(xy, arclen, length, count, ctx=ctx)
            _wpl._winding(xy, winding, count, ctx=ctx)
        return enqueue


class OffsetDefault(_Builtin):
    """``offset.default``: the plan-view borders at ``half_width``."""

    kind = "offset"
    name = "default"
    Params = _p.OffsetParams
    reads = {"xy": "cl.xy", "normal": "cl.normal2", "count": "cl.count"}
    writes = {"outer": "band.outer2", "inner": "band.inner2"}

    def build(self, b, p):
        xy, nrm2, count = b.read("xy"), b.read("normal"), b.read("count")
        outer2 = b.write("outer", wp.vec2f, per="cl")
        inner2 = b.write("inner", wp.vec2f, per="cl")
        area_a = b.scratch("area_a", wp.float32, b.E)
        area_b = b.scratch("area_b", wp.float32, b.E)
        hw = float(p.half_width)

        def enqueue(ctx):
            _wpl.offset(xy, nrm2, hw, outer2, inner2, area_a, area_b, count, ctx=ctx)
        return enqueue


# ---------------------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------------------

def _zcfg(b, profile: str, p):
    """The namespace the z-profile wrappers read (``num_envs``, ``device``,
    ``z_profile`` and the profile's ``z_*`` fields), built once."""
    fields = {f: getattr(p, f) for f in p.__dataclass_fields__}
    return types.SimpleNamespace(num_envs=b.E, device=b.layout.device, z_profile=profile,
                                 **fields)


class ElevationFlat(_Builtin):
    """``elevation.flat``: on a track, ``cl.z`` stays at its zero initialisation (no
    launch; the flat lift adds ``z_base``, and the planner refuses a non-zero
    ``z_base`` here before allocating). On a gate pipeline (``z`` rewired to
    ``gates.z``) every real gate is at ``z_base`` (``_z_flat_k``)."""

    kind = "elevation"
    name = "flat"
    Params = _p.ElevationFlatParams
    reads = {"count": "cl.count"}
    writes = {"z": "cl.z"}

    def build(self, b, p):
        z = b.write("z", wp.float32, per="cl")
        if b.per_of("z") != "gates":
            def enqueue(ctx):
                pass
            return enqueue
        count = b.read("count")
        G = int(b.layout.stride)
        zcfg = _zcfg(b, self.name, p)

        def enqueue(ctx):
            _zprof.apply_z_profile(zcfg, None, count, G, None, None, z, ctx=ctx)
        return enqueue


class _ElevationKnots(_Builtin):
    """Knot-based profiles: K arc-spaced knots, then a periodic monotone cubic. On a
    gate pipeline (ports rewired to ``gates.*``) the profile is drawn per gate instead
    (the gates are already sparse, knot-like anchors), as 0.2.0.dev3 did."""

    kind = "elevation"
    reads = {"seed": "seed", "count": "cl.count", "arclen": "cl.arclen2",
             "length": "cl.length2"}
    writes = {"z": "cl.z"}

    def build(self, b, p):
        S = int(b.layout.stride)
        seed = b.read("seed")
        count, arclen, length = b.read("count"), b.read("arclen"), b.read("length")
        z = b.write("z", wp.float32, per="cl")
        if b.per_of("z") == "gates":
            zcfg = _zcfg(b, self.name, p)

            def enqueue(ctx):
                _zprof.apply_z_profile(zcfg, seed, count, S, arclen, length, z, ctx=ctx)
            return enqueue
        K = int(p.z_control_points)
        knot_cum = b.scratch("knot_cum", wp.float32, b.E * K)
        knot_count = b.scratch("knot_count", wp.int32, b.E)
        knot_z = b.scratch("knot_z", wp.float32, b.E * K)
        zcfg = _zcfg(b, self.name, p)

        def enqueue(ctx):
            _zprof.apply_z_profile_knots(zcfg, seed, count, S, arclen, length, knot_cum,
                                         knot_count, knot_z, z, ctx=ctx)
        return enqueue


class ElevationUniform(_ElevationKnots):
    """``elevation.uniform``: i.i.d. knot altitudes in ``[z_min, z_max]``."""

    name = "uniform"
    Params = _p.ElevationUniformParams


class ElevationRandomWalk(_ElevationKnots):
    """``elevation.random_walk``: a grade-capped random walk over the knots."""

    name = "random_walk"
    Params = _p.ElevationRandomWalkParams


class ElevationNoise(_Builtin):
    """``elevation.noise``: periodic harmonic noise, per point."""

    kind = "elevation"
    name = "noise"
    Params = _p.ElevationNoiseParams
    reads = {"seed": "seed", "count": "cl.count", "arclen": "cl.arclen2",
             "length": "cl.length2"}
    writes = {"z": "cl.z"}

    def build(self, b, p):
        S = int(b.layout.stride)
        seed = b.read("seed")
        count, arclen, length = b.read("count"), b.read("arclen"), b.read("length")
        z = b.write("z", wp.float32, per="cl")
        zcfg = _zcfg(b, self.name, p)

        def enqueue(ctx):
            _zprof.apply_z_profile(zcfg, seed, count, S, arclen, length, z, ctx=ctx)
        return enqueue


# ---------------------------------------------------------------------------------------
# Validity / lift
# ---------------------------------------------------------------------------------------

class ValidityDefault(_Builtin):
    """``validity.default``: the fused per-env gate (``_validity_k``, one launch): its
    0/1 flag into ``track.valid`` and one ``valid.fail`` bit per failed check (bits 0-7,
    ``BUILTIN_FAIL_BITS``). It initialises ``valid.fail`` (writes it without reading it),
    so it comes before any plugin check."""

    kind = "validity"
    name = "default"
    Params = _p.ValidityParams
    reads = {"xy": "cl.xy", "half_width": "cl.half_width", "count": "cl.count",
             "gen_valid": "gen.valid", "outer": "band.outer2", "inner": "band.inner2",
             "z": "cl.z", "truncated": "cl.truncated"}
    writes = {"valid": "track.valid", "fail": "valid.fail"}

    def build(self, b, p):
        from .channels import BUILTIN_FAIL_BITS  # noqa: PLC0415

        S = int(b.layout.stride)
        xy, w, count = b.read("xy"), b.read("half_width"), b.read("count")
        gen_valid, outer2, inner2 = b.read("gen_valid"), b.read("outer"), b.read("inner")
        z, truncated = b.read("z"), b.read("truncated")
        valid = b.write("valid", wp.int32)
        fail = b.write("fail", wp.int32)
        b._reserve_fail_bits(BUILTIN_FAIL_BITS)
        has_border = 1 if getattr(p, "validity_border_check", False) else 0

        def enqueue(ctx):
            _wpl.validity_inplace(xy, w, count, gen_valid, outer2, inner2, has_border, S,
                                  z, valid, p, truncated_wp=truncated, ctx=ctx,
                                  fail_wp=fail)
        return enqueue


_LIFT_WRITES = {"outer3": "track.outer", "center3": "track.center",
                "inner3": "track.inner", "tangent3": "track.tangent",
                "normal3": "track.normal"}


class LiftFlat(_Builtin):
    """``lift.flat``: the plan-view geometry at constant ``z = z_base``."""

    kind = "lift"
    name = "flat"
    Params = _p.LiftFlatParams
    reads = {"outer": "band.outer2", "xy": "cl.xy", "inner": "band.inner2",
             "tangent": "cl.tangent2", "normal": "cl.normal2"}
    writes = dict(_LIFT_WRITES)

    def build(self, b, p):
        out2, ctr2, inn2 = b.read("outer"), b.read("xy"), b.read("inner")
        tan2, nrm2 = b.read("tangent"), b.read("normal")
        outer = b.write("outer3", wp.vec3f, per="cl")
        center = b.write("center3", wp.vec3f, per="cl")
        inner = b.write("inner3", wp.vec3f, per="cl")
        tangent = b.write("tangent3", wp.vec3f, per="cl")
        normal = b.write("normal3", wp.vec3f, per="cl")
        n = b.E * int(b.layout.stride)
        z_base = float(p.z_base)

        def enqueue(ctx):
            ctx.launch(_wpl._lift_track_k, n,
                       [out2, ctr2, inn2, tan2, nrm2, z_base,
                        outer, center, inner, tangent, normal])
            ctx.sync()              # debug-only (TRACKGEN_DEBUG_SYNC=1), as inflate_warp
        return enqueue


class LiftElevated(_Builtin):
    """``lift.elevated``: level cross-sections at ``cl.z``, then the 3D tangent and arc
    length (``track.arclen`` / ``track.length``; the plan-view ``cl.arclen2`` /
    ``cl.length2`` stay intact)."""

    kind = "lift"
    name = "elevated"
    reads = {"outer": "band.outer2", "xy": "cl.xy", "inner": "band.inner2",
             "normal": "cl.normal2", "z": "cl.z", "count": "cl.count"}
    writes = {**_LIFT_WRITES, "arclen": "track.arclen", "length": "track.length"}

    def build(self, b, p):
        S = int(b.layout.stride)
        out2, ctr2, inn2 = b.read("outer"), b.read("xy"), b.read("inner")
        nrm2, z, count = b.read("normal"), b.read("z"), b.read("count")
        outer = b.write("outer3", wp.vec3f, per="cl")
        center = b.write("center3", wp.vec3f, per="cl")
        inner = b.write("inner3", wp.vec3f, per="cl")
        tangent = b.write("tangent3", wp.vec3f, per="cl")
        normal = b.write("normal3", wp.vec3f, per="cl")
        # Every slot of arclen / tangent and every length[e] is written by
        # _track_frames3_k.
        arclen = b.write("arclen", wp.float32, per="cl")
        length = b.write("length", wp.float32)
        E = b.E
        n = E * S

        def enqueue(ctx):
            ctx.launch(_wpl._lift_track_zvar_k, n,
                       [out2, ctr2, inn2, nrm2, z, outer, center, inner, normal])
            ctx.launch(_wpl._track_frames3_k, E,
                       [center, count, S, tangent, arclen, length])
            ctx.sync()              # debug-only (TRACKGEN_DEBUG_SYNC=1), as inflate_warp
        return enqueue


# The kinds open to plugins: registered in track_gen.plugin.registry.
REGISTRY_STAGES = (
    Xpbd, RelaxNone, WidthConstant, ElevationFlat, ElevationUniform, ElevationRandomWalk,
    ElevationNoise, ValidityDefault,
)
# The built-in-only kinds: the private catalogue.
CATALOGUE_STAGES = (
    ConstantSpacing, Uniform, FramesDefault, OffsetDefault, LiftFlat, LiftElevated,
)
# Every built-in non-generator stage.
BUILTIN_STAGES = CATALOGUE_STAGES + REGISTRY_STAGES
