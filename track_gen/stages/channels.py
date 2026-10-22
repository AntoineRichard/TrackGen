"""The channel catalogue of the track pipeline (channels v1, roadmap PR10a/PR10b).

A channel is a named per-env buffer the stages of a :mod:`track_gen.pipeline` program
read and write. ``per`` sizes it: ``"env"`` is ``[E]``, ``"raw"`` is ``[E * raw]`` (the
generator's capacity), ``"cl"`` is ``[E * N_max]``. ``interp`` records how a per-point
channel is carried through a resample (roadmap PR13; nothing is carried in this
release). Besides these channels, a plugin stage may read and write its own namespaced
channels ``cl.<package>.<name>`` (:data:`PLUGIN_CHANNEL`): their dtype and size are
fixed by the first stage that writes one.

:data:`BUILTIN_FAIL_BITS` names the bits of the ``valid.fail`` mask that TrackGen's
built-in checks own (0-15; bits 8-15 are reserved for future built-in checks). Plugin
validity stages get bits 16-30 from ``StageBuilder.fail_bit``; bit 31 is never used.
"""
from __future__ import annotations

import dataclasses
import re
from types import MappingProxyType
from typing import Mapping

import warp as wp

__all__ = ["BUILTIN_FAIL_BITS", "CHANNELS", "Channel", "FAIL", "PERS", "PLUGIN_CHANNEL",
           "SEED", "is_plugin_channel"]

SEED = "seed"          # written by the Program's prologue; no stage writes it
FAIL = "valid.fail"    # the per-env fail bitmask (Track.fail_mask)
PERS = ("env", "raw", "cl")   # the sizes a channel may have

#: A plugin's own channel: ``cl.<package>.<name>`` (at least two dotted components after
#: ``cl.``, each a Python identifier), never a catalogue name.
PLUGIN_CHANNEL = re.compile(r"^cl\.[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")

#: The fail bits of TrackGen's built-in checks (``validity.default``): ``valid.fail`` bit
#: ``i`` set means check ``i`` failed. The border (5) and grade (6) bits are named even
#: when their check is off (then never set).
BUILTIN_FAIL_BITS: Mapping[str, int] = MappingProxyType({
    "generator": 0, "turning": 1, "finite": 2, "thickness": 3, "w_floor": 4,
    "border": 5, "grade": 6, "truncation": 7})

#: The bits plugin validity stages are given, in pipeline order (``StageBuilder.fail_bit``).
PLUGIN_FAIL_BITS = range(16, 31)


@dataclasses.dataclass(frozen=True)
class Channel:
    """One catalogue entry."""

    per: str                    # "env" | "raw" | "cl"
    dtype: object               # a Warp dtype
    interp: "str | None"
    doc: str

    def __post_init__(self) -> None:
        if self.per not in PERS:
            raise ValueError(f"Channel.per must be one of {PERS}, got {self.per!r}")


def _c(per, dtype, doc, interp=None) -> Channel:
    return Channel(per=per, dtype=dtype, interp=interp, doc=doc)


CHANNELS: Mapping[str, Channel] = MappingProxyType({
    SEED: _c("env", wp.int32, "The effective per-env seed (seed and epoch folded)."),
    "raw.xy": _c("raw", wp.vec2f, "The generator's raw closed centerline."),
    "gen.valid": _c("env", wp.int32, "The generator's per-env flag (0/1)."),
    "cl.xy": _c("cl", wp.vec2f, "The centerline (constant-spacing, relaxed, uniform)."),
    "cl.count": _c("env", wp.int32, "Real points per env; NaN padding after."),
    "cl.truncated": _c("env", wp.int32, "1 where N_max cut the loop."),
    "cl.half_width": _c("cl", wp.float32, "Per-point half-width.", "linear"),
    "cl.tangent2": _c("cl", wp.vec2f, "Plan-view unit tangent."),
    "cl.normal2": _c("cl", wp.vec2f, "Plan-view unit left normal."),
    "cl.kappa": _c("cl", wp.float32, "Menger curvature."),
    "cl.arclen2": _c("cl", wp.float32, "Plan-view cumulative arc length."),
    "cl.length2": _c("env", wp.float32, "Plan-view loop length."),
    "cl.winding": _c("env", wp.float32, "Signed loop winding (+1 CCW, -1 CW, 0)."),
    "band.outer2": _c("cl", wp.vec2f, "Plan-view outer border."),
    "band.inner2": _c("cl", wp.vec2f, "Plan-view inner border."),
    "cl.z": _c("cl", wp.float32, "Per-point elevation.", "linear"),
    "track.count": _c("env", wp.int32, "Track.count."),
    "track.arclen": _c("cl", wp.float32, "Track.arclen: 3D cumulative arc length (lifted)."),
    "track.length": _c("env", wp.float32, "Track.length: 3D loop length (lifted)."),
    "track.valid": _c("env", wp.int32, "validity.default's own 0/1 flag; Track.valid is "
                                       "fail_mask == 0 (equal when no other check runs)."),
    FAIL: _c("env", wp.int32, "Per-env fail bitmask: bit i set = check i failed "
                              "(Track.fail_mask)."),
    "track.outer": _c("cl", wp.vec3f, "Track.outer (lifted)."),
    "track.center": _c("cl", wp.vec3f, "Track.center (lifted)."),
    "track.inner": _c("cl", wp.vec3f, "Track.inner (lifted)."),
    "track.tangent": _c("cl", wp.vec3f, "Track.tangent (lifted)."),
    "track.normal": _c("cl", wp.vec3f, "Track.normal (lifted)."),
})


def is_plugin_channel(name) -> bool:
    """Whether ``name`` is a namespaced plugin channel (``cl.<package>.<name>``, not a
    catalogue name)."""
    return isinstance(name, str) and name not in CHANNELS \
        and PLUGIN_CHANNEL.match(name) is not None
