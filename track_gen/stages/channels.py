"""The channel catalogue of the track and gate pipelines (channels v1, roadmap
PR10a/PR10b/PR11/PR13a).

A channel is a named per-env buffer the stages of a :mod:`track_gen.pipeline` program
read and write. ``per`` sizes it: ``"env"`` is ``[E]``, ``"raw"`` is ``[E * raw]`` (the
generator's capacity), ``"cl"`` is ``[E * N_max]``, ``"anchors"`` is ``[E * A]`` (``A``:
the ``capacity()["gates"]`` of the stage that writes the anchors: a gate generator or a
``stations`` stage) and ``"gates"`` is ``[E * max_gates]`` (``BatchSpec.max_gates``).
``interp`` records how a per-point
channel is carried through a resample (roadmap PR13a: ``raw.z`` / ``raw.half_width``
and the generators' ``raw.<package>.<name>`` attributes are carried to the centerline,
see :data:`CARRIED_CATALOGUE`). Besides these channels, a plugin may read and write its
own namespaced channels ``<domain>.<package>.<name>`` (:data:`PLUGIN_CHANNEL`; the
domain prefix is ``raw``, ``cl``, ``anchors`` or ``gates``): their dtype and size are
fixed by the first stage that writes one.

:data:`BUILTIN_FAIL_BITS` names the bits of the ``valid.fail`` mask that TrackGen's
built-in checks own (0-15; bits 8-15 are reserved for future built-in checks), and
:data:`GATE_FAIL_BITS` the bits of ``validity.gate_default`` in a gate pipeline (0-5;
6-15 reserved). Plugin validity stages get bits 16-30 from ``StageBuilder.fail_bit``; bit
31 is never used.
"""
from __future__ import annotations

import dataclasses
import re
from types import MappingProxyType
from typing import Mapping

import warp as wp

__all__ = ["BUILTIN_FAIL_BITS", "CARRIED_CATALOGUE", "CHANNELS", "Channel", "DOMAINS",
           "ELEVATION_GATE_CHANNELS", "FAIL", "GATE_FAIL_BITS", "INTERPS", "PERS",
           "PLUGIN_CHANNEL", "SEED", "channel_domain", "family_key", "is_plugin_channel",
           "required_port_domain"]

SEED = "seed"          # written by the Program's prologue; no stage writes it
FAIL = "valid.fail"    # the per-env fail bitmask (Track.fail_mask)
PERS = ("env", "raw", "cl", "anchors", "gates")   # the sizes a channel may have
#: The point-domain class of each ``per``: a port may be rewired to a channel of the
#: same dtype and the same class (``cl`` <-> ``gates``: one elevation stage serves the
#: track's ``[E*N_max]`` points and the gate pipeline's ``[E*max_gates]`` gates).
PER_CLASS = {"env": "env", "raw": "raw", "cl": "points", "gates": "points",
             "anchors": "anchors"}

#: The point domains a plugin channel may live in (its name's prefix): the generator's
#: raw loop, the centerline, the anchors and the gate slots.
DOMAINS = ("raw", "cl", "anchors", "gates")
#: A plugin's own channel: ``<domain>.<package>.<name>`` with ``<domain>`` one of
#: :data:`DOMAINS` (at least two dotted components after it, each a Python identifier),
#: never a catalogue name. The prefix fixes the domain: ``raw.acme.grip`` is written by
#: a centerline generator and carried to ``cl.acme.grip``; ``gates.acme.x`` lives on the
#: gate slots.
PLUGIN_CHANNEL = re.compile(r"^(raw|cl|anchors|gates)\.[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")
#: A required read port (``Stage.reads`` value ``"<domain>.*"``): the stage reads a
#: channel of that domain the user wires with ``StageRef.ports``.
REQUIRED_PORT = re.compile(r"^(raw|cl|anchors|gates)\.\*$")
#: The carry policies a generator's raw attribute declares (``StageBuilder.write(...,
#: interp=)``): see :doc:`/how-it-works/pipeline-stages`.
INTERPS = ("linear", "nearest", "previous", "angle")
#: The catalogue channels carried through the resamples (with their ``interp``):
#: a generator's provided ``raw.z`` / ``raw.half_width`` and the centerline's ``cl.z`` /
#: ``cl.half_width``, whoever wrote them.
CARRIED_CATALOGUE = frozenset({"raw.z", "raw.half_width", "cl.z", "cl.half_width"})

#: The fail bits of TrackGen's built-in checks (``validity.default``): ``valid.fail`` bit
#: ``i`` set means check ``i`` failed. The border (5) and grade (6) bits are named even
#: when their check is off (then never set).
BUILTIN_FAIL_BITS: Mapping[str, int] = MappingProxyType({
    "generator": 0, "turning": 1, "finite": 2, "thickness": 3, "w_floor": 4,
    "border": 5, "grade": 6, "truncation": 7})

#: The fail bits of ``validity.gate_default`` (the fused gate validity check): bit ``i``
#: set means check ``i`` failed. The crossing (4) and grade (5) bits are named even when
#: their check is off (then never set).
GATE_FAIL_BITS: Mapping[str, int] = MappingProxyType({
    "count": 0, "finite": 1, "tangent": 2, "spacing": 3, "crossing": 4, "grade": 5})

#: The bits plugin validity stages are given, in pipeline order (``StageBuilder.fail_bit``).
PLUGIN_FAIL_BITS = range(16, 31)


@dataclasses.dataclass(frozen=True)
class Channel:
    """One catalogue entry."""

    per: str                    # "env" | "raw" | "cl" | "anchors" | "gates"
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
    "raw.z": _c("raw", wp.float32, "Per-raw-point altitude a centerline generator "
                                   "provides (carried to cl.z).", "linear"),
    "raw.half_width": _c("raw", wp.float32, "Per-raw-point half-width a centerline "
                                            "generator provides (carried to "
                                            "cl.half_width).", "linear"),
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
    "track.half_width": _c("cl", wp.float32, "Track.half_width: the band's half-width on "
                                             "real points, NaN padding (offset)."),
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
    # -- the gate pipeline (roadmap PR11) ------------------------------------------------
    "anchors.xy": _c("anchors", wp.vec2f, "Gate anchors (a gate generator's or the "
                                          "stations'), unordered or generator-ordered."),
    "anchors.count": _c("env", wp.int32, "Real anchors per env."),
    "anchors.z": _c("anchors", wp.float32, "Per-anchor altitude a generator provides."),
    "anchors.tangent": _c("anchors", wp.vec3f, "Per-anchor tangent a generator provides."),
    "anchors.half_size": _c("anchors", wp.float32, "Per-anchor half-size a generator "
                                                   "provides."),
    "anchors.roll": _c("anchors", wp.float32, "Per-anchor roll (rad) a generator "
                                              "provides."),
    "gates.xy": _c("gates", wp.vec2f, "Ordered gate centres (plan view); NaN padding."),
    "gates.count": _c("env", wp.int32, "GateSequence.count."),
    "gates.cum": _c("gates", wp.float32, "Plan-view cumulative chord length per gate."),
    "gates.perim": _c("env", wp.float32, "Plan-view closed-loop perimeter of the gates."),
    "gates.z": _c("gates", wp.float32, "Per-gate altitude."),
    "gates.position": _c("gates", wp.vec3f, "GateSequence.position."),
    "gates.tangent": _c("gates", wp.vec3f, "GateSequence.tangent."),
    "gates.forward": _c("gates", wp.vec3f, "GateSequence.forward."),
    "gates.left": _c("gates", wp.vec3f, "GateSequence.left."),
    "gates.right": _c("gates", wp.vec3f, "GateSequence.right."),
    "gates.orientation": _c("gates", wp.quatf, "GateSequence.orientation."),
    "gates.half_size": _c("gates", wp.float32, "GateSequence.half_size."),
    "gates.roll": _c("gates", wp.float32, "Per-gate roll (rad), carried from the anchors."),
    "gates.valid": _c("env", wp.int32, "validity.gate_default's own 0/1 flag; "
                                       "GateSequence.valid is fail_mask == 0."),
})

#: The default port rewiring of an ``elevation`` stage in a gate pipeline: the planner
#: maps each of the stage's ports on one of these centerline channels to its gate
#: counterpart (an explicit ``StageRef.ports`` entry overrides it).
ELEVATION_GATE_CHANNELS: Mapping[str, str] = MappingProxyType({
    "cl.xy": "gates.xy", "cl.count": "gates.count", "cl.arclen2": "gates.cum",
    "cl.length2": "gates.perim", "cl.z": "gates.z"})


def is_gate_channel(name) -> bool:
    """Whether ``name`` is a catalogue channel of the gate point domains (``anchors.*``
    / ``gates.*``)."""
    return isinstance(name, str) and name.startswith(("anchors.", "gates.")) \
        and name in CHANNELS


def is_plugin_channel(name) -> bool:
    """Whether ``name`` is a namespaced plugin channel (``<domain>.<package>.<name>``, not
    a catalogue name)."""
    return isinstance(name, str) and name not in CHANNELS \
        and PLUGIN_CHANNEL.match(name) is not None


def channel_domain(name) -> "str | None":
    """The point domain of a plugin channel or of a catalogue channel named after one
    (``"raw"``, ``"cl"``, ``"anchors"`` or ``"gates"``: the name's prefix), else
    ``None``."""
    if not isinstance(name, str):
        return None
    head = name.split(".", 1)[0]
    if head in DOMAINS and (name in CHANNELS or is_plugin_channel(name)):
        return head
    return None


def family_key(name: str) -> str:
    """A carried channel's family key: its name without the domain prefix
    (``raw.acme.grip`` -> ``"acme.grip"``, ``cl.z`` -> ``"z"``)."""
    return name.split(".", 1)[1]


def required_port_domain(name) -> "str | None":
    """The domain of a required read port value (``"cl.*"`` -> ``"cl"``), else
    ``None``."""
    if isinstance(name, str):
        m = REQUIRED_PORT.match(name)
        if m is not None:
            return m.group(1)
    return None
