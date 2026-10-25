"""``generator/external``: ingest caller-provided centerlines.

A centerline generator Stage (``kind="generator"``, ``name="external"``) that reads the
raw closed centerline of every env from device inputs the caller fills
(``Program.inputs``, stable pointers; :func:`track_gen.pipeline.load_centerlines` loads
numpy data outside the hot path, or the caller's own graph writes them on the
generation's stream), so a centerline from elsewhere (a dataset, a learned model, a
hand-drawn track) runs through the standard pipeline (resample, width, frames, offset,
validity, gates through stations, ...). Built only from the public plugin surface:
``StageBuilder.input(per="raw")``,
:meth:`Stage.declare <track_gen.plugin.Stage.declare>`, ``Stage.seeded = False`` and
``Stage.preferred = {"relax": "none"}`` (``presets.track(relax="auto")`` runs no relax:
the ingested geometry is not moved; pass ``relax="xpbd"`` to repair it).

Inputs (``N`` = the raw capacity, ``presets.track(num_points=N)``):

- ``centerline.xy``: ``[E * N]`` vec2f, UNLATCHED (read during the generation); initial
  value: a circle of radius ``0.6 * scale`` per env (a fresh generator yields valid
  tracks at the default ``scale`` / ``half_width`` / ``spacing``; with settings in the
  data's units the placeholder is too small for the spacing: load centerlines before
  the first generation);
- ``centerline.count``: ``[E]`` int32, LATCHED like every per-env input; the real point
  count of each row (initial ``N``);
- ``centerline.z`` (``with_z``): ``[E * N]`` float32, the altitude (provides ``z``);
- ``centerline.half_width`` (``with_half_width``): ``[E * N]`` float32, the half-width
  (provides ``half_width``; initial ``0.1 * scale``, a placeholder);
- ``centerline.<pkg>.<name>`` per ingested attribute: ``[E * N]`` of its dtype, carried
  with its ``interp`` and exported as ``Track.attributes["<pkg>.<name>"]``.

Per generation: each input is copied first (the snapshot the pipeline consumes), padded
past ``count[e]`` with the row's last real point (exact: the padding slots are
zero-length segments, so the resample sees the ``count[e]``-point loop), then the
validity is computed on the COPIES: ``gen.valid[e] = 1`` iff ``3 <= count[e] <= N``,
the first ``count[e]`` points are finite and, when present, ``z`` is finite and
``half_width`` finite and ``> 0`` (else fail bit 0, ``"generator"``; poisoned with
``on_invalid="poison"``).

No per-env ``size``: the ingested geometry is taken as given (a size
would scale it about the data's own origin while its altitude, half-width and attributes
stay, and :func:`~track_gen.pipeline.load_centerlines`' host sizing check could not see
it). Scale ``centerline.xy`` in your own graph; ``generator.size`` (a randomizer,
``set("size")``, a flat ``TrackGenConfig.size`` other than 1) raises with that hint.
"""
from __future__ import annotations

import dataclasses

import warp as wp

from ..exec.context import LaunchContext
from ..plugin.params import Params, static
from ..plugin.stage import CaptureClass, Stage

__all__ = ["External", "ExternalParams"]

_DTYPES = {"float32": wp.float32, "int32": wp.int32, "vec2f": wp.vec2f,
           "vec3f": wp.vec3f}
_INTERPS = ("linear", "nearest", "previous", "angle")
#: The ingestion input namespace (``Program.inputs``).
PREFIX = "centerline."


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExternalParams(Params):
    """``generator/external``: which per-point channels the caller provides."""

    #: Appended to the KeyError / PlanError / ValueError of a ``size`` on external
    #: (it has none).
    _field_hints = {"size": "external ingests your geometry as given: scale "
                            "centerline.xy in your graph"}

    with_z: bool = static(False, doc="inputs['centerline.z'] gives the altitude "
                                     "(provides z)")
    with_half_width: bool = static(False, doc="inputs['centerline.half_width'] gives the "
                                              "half-width (provides half_width)")
    attributes: "tuple[tuple[str, str, str], ...]" = static(
        (), doc="(key, dtype, interp) per ingested attribute, e.g. ('acme.tag', 'int32', "
                "'nearest'); dtype float32 / int32 / vec2f / vec3f")

    def validate(self, layout) -> None:
        from ..stages.channels import is_plugin_channel  # noqa: PLC0415

        if not isinstance(self.attributes, tuple):
            raise ValueError(f"external: attributes must be a tuple of (key, dtype, "
                             f"interp), got {type(self.attributes).__name__}")
        seen = set()
        for item in self.attributes:
            if not isinstance(item, tuple) or len(item) != 3 or \
                    not all(isinstance(v, str) for v in item):
                raise ValueError(f"external: each attribute is a (key, dtype, interp) "
                                 f"triple of strings, got {item!r}")
            key, dtype, interp = item
            if not is_plugin_channel(f"raw.{key}"):
                raise ValueError(f"external: attribute key {key!r} must be "
                                 f"'<package>.<name>' (e.g. 'acme.tag')")
            if key in seen:
                raise ValueError(f"external: attribute {key!r} is given twice")
            seen.add(key)
            if dtype not in _DTYPES:
                raise ValueError(f"external: attribute {key!r}: dtype must be one of "
                                 f"{sorted(_DTYPES)}, got {dtype!r}")
            if interp not in _INTERPS:
                raise ValueError(f"external: attribute {key!r}: interp must be one of "
                                 f"{list(_INTERPS)}, got {interp!r}")
            if dtype == "int32" and interp not in ("nearest", "previous"):
                raise ValueError(f"external: attribute {key!r}: an int32 attribute is "
                                 f"carried with interp='nearest' or 'previous', got "
                                 f"{interp!r}")
            if interp == "angle" and dtype != "float32":
                raise ValueError(f"external: attribute {key!r}: interp='angle' carries "
                                 f"float32 radians, got {dtype!r}")


# ---------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------

@wp.kernel
def _external_circle_k(n: int, radius: float, out: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    i = t - (t // n) * n
    a = 6.283185307179586 * float(i) / float(n)
    out[t] = wp.vec2f(radius * wp.cos(a), radius * wp.sin(a))


@wp.func
def _row_count(count: wp.array(dtype=wp.int32), e: int, n: int) -> int:
    return wp.max(1, wp.min(count[e], n))


@wp.kernel
def _external_pad_vec2f_k(src: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32),
                          n: int, dst: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    dst[t] = src[e * n + wp.min(i, _row_count(count, e, n) - 1)]


@wp.kernel
def _external_pad_vec3f_k(src: wp.array(dtype=wp.vec3f), count: wp.array(dtype=wp.int32),
                          n: int, dst: wp.array(dtype=wp.vec3f)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    dst[t] = src[e * n + wp.min(i, _row_count(count, e, n) - 1)]


@wp.kernel
def _external_pad_f32_k(src: wp.array(dtype=wp.float32), count: wp.array(dtype=wp.int32),
                        n: int, dst: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    dst[t] = src[e * n + wp.min(i, _row_count(count, e, n) - 1)]


@wp.kernel
def _external_pad_i32_k(src: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                        n: int, dst: wp.array(dtype=wp.int32)):
    t = wp.tid()
    e = t // n
    i = t - e * n
    dst[t] = src[e * n + wp.min(i, _row_count(count, e, n) - 1)]


_PAD = {wp.vec2f: _external_pad_vec2f_k, wp.vec3f: _external_pad_vec3f_k,
        wp.float32: _external_pad_f32_k, wp.int32: _external_pad_i32_k}


@wp.kernel
def _external_valid_k(xy: wp.array(dtype=wp.vec2f), count: wp.array(dtype=wp.int32),
                      n: int, z: wp.array(dtype=wp.float32), has_z: int,
                      hw: wp.array(dtype=wp.float32), has_hw: int,
                      valid: wp.array(dtype=wp.int32)):
    e = wp.tid()
    m = count[e]
    ok = int(1)
    if m < 3 or m > n:
        ok = 0
        m = wp.max(0, wp.min(m, n))
    base = e * n
    for i in range(m):
        p = xy[base + i]
        if not (wp.isfinite(p[0]) and wp.isfinite(p[1])):
            ok = 0
        if has_z != 0:
            if not wp.isfinite(z[base + i]):
                ok = 0
        if has_hw != 0:
            w = hw[base + i]
            if not (wp.isfinite(w) and w > 0.0):
                ok = 0
    valid[e] = ok


# ---------------------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------------------

class External(Stage):
    """``generator/external`` (see the module docstring)."""

    kind = "generator"
    name = "external"
    output = "centerline"
    Params = ExternalParams
    capture = CaptureClass.STATIC
    writes = {"xy": "raw.xy", "valid": "gen.valid"}
    preferred = {"relax": "none"}
    seeded = False

    @classmethod
    def declare(cls, params):
        """``(writes, provides)`` for these params: ``raw.xy`` and ``gen.valid`` always;
        ``raw.z`` and ``provides`` ``z`` with ``with_z``; ``raw.half_width`` and
        ``provides`` ``half_width`` with ``with_half_width``; one ``raw.<key>`` per
        declared attribute (see :meth:`track_gen.plugin.Stage.declare`)."""
        writes = dict(cls.writes)
        provides = set()
        if params.with_z:
            writes["z"] = "raw.z"
            provides.add("z")
        if params.with_half_width:
            writes["half_width"] = "raw.half_width"
            provides.add("half_width")
        for key, _, _ in params.attributes:
            writes[key] = f"raw.{key}"
        return writes, frozenset(provides)

    def build(self, b, p):
        E, N = b.E, b.raw
        dev = b.device
        xy_in = b.input(PREFIX + "xy", wp.vec2f, 0.0, per="raw")
        count = b.input(PREFIX + "count", wp.int32, N)
        LaunchContext.current(dev).launch(_external_circle_k, E * N,
                                          [N, 0.6 * float(b.scale), xy_in])
        xy = b.write("xy", wp.vec2f, per="raw")
        valid = b.write("valid", wp.int32)
        pads = [(xy_in, xy)]
        z = hw = None
        if p.with_z:
            z_in = b.input(PREFIX + "z", wp.float32, 0.0, per="raw")
            z = b.write("z", wp.float32, per="raw")
            pads.append((z_in, z))
        if p.with_half_width:
            hw_in = b.input(PREFIX + "half_width", wp.float32, 0.1 * float(b.scale),
                            per="raw")
            hw = b.write("half_width", wp.float32, per="raw")
            pads.append((hw_in, hw))
        for key, dtype, interp in p.attributes:
            dt = _DTYPES[dtype]
            src = b.input(PREFIX + key, dt, 0, per="raw")
            dst = b.write(key, dt, per="raw", interp=interp, export=True)
            pads.append((src, dst))
        dummy = b.scratch("absent", wp.float32, 1)
        z_arr, has_z = (z, 1) if z is not None else (dummy, 0)
        hw_arr, has_hw = (hw, 1) if hw is not None else (dummy, 0)
        launches = [(_PAD[dst.dtype], E * N, [src, count, N, dst]) for src, dst in pads]
        launches.append((_external_valid_k, E,
                         [xy, count, N, z_arr, has_z, hw_arr, has_hw, valid]))

        def enqueue(ctx):
            for kernel, dim, args in launches:
                ctx.launch(kernel, dim, args)
        return enqueue


def register(reg) -> None:
    """Register ``generator/external`` into ``reg`` (the built-in plugin load)."""
    reg.add(External)
