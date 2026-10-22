"""The channel catalogue of the track pipeline (channels v1, roadmap PR10a).

A channel is a named per-env buffer the stages of a :mod:`track_gen.pipeline` program
read and write. ``per`` sizes it: ``"env"`` is ``[E]``, ``"raw"`` is ``[E * raw]`` (the
generator's capacity), ``"cl"`` is ``[E * N_max]``. ``interp`` records how a per-point
channel is carried through a resample (roadmap PR13; nothing is carried in this
release). Only these channels may be read or written in this release.
"""
from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Mapping

import warp as wp

__all__ = ["CHANNELS", "Channel", "PERS", "SEED"]

SEED = "seed"          # written by the Program's prologue; no stage writes it
PERS = ("env", "raw", "cl")   # the sizes a channel may have


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
    "track.valid": _c("env", wp.int32, "Track.valid."),
    "track.outer": _c("cl", wp.vec3f, "Track.outer (lifted)."),
    "track.center": _c("cl", wp.vec3f, "Track.center (lifted)."),
    "track.inner": _c("cl", wp.vec3f, "Track.inner (lifted)."),
    "track.tangent": _c("cl", wp.vec3f, "Track.tangent (lifted)."),
    "track.normal": _c("cl", wp.vec3f, "Track.normal (lifted)."),
})
