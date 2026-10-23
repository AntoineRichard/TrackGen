"""Params classes of the built-in track and gate stages (roadmap PR10a / PR11 / PR13a).

Like the built-in generators' Params (:mod:`track_gen._src.builtin_params`), every class
keeps the FLAT ``TrackGenConfig`` field names, every field is static, and
``from_legacy(config)`` copies the flat values (no casts), except
``WidthFromChannelParams`` (dynamic ``scale`` / ``lo`` / ``hi``) and ``StationsParams``
(``gates``, ``spacing_metric``), which have no ``TrackGenConfig`` counterpart. The stage
hands its Params instance (or, for elevation, a namespace built from it) to the legacy
wrappers as their ``config`` argument, so the same values reach the same ``float()`` /
``int()`` conversions and the launches are unchanged.

``validate()`` re-runs the ``TrackGenConfig.__post_init__`` rules of the class's own
fields with the same messages: a spec made from a ``TrackGenConfig`` always passes; the
rules exist for hand-built specs. The stage Params map the fields a config HAS
(``from_legacy`` of a ``GateGenConfig`` leaves ``z_control_points`` and
``check_crossing`` at their class defaults).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .._src.builtin_params import _FlatParams
from ..plugin.params import Params, dynamic, static

__all__ = [
    "ConstantSpacingParams",
    "ElevationFlatParams",
    "ElevationNoiseParams",
    "ElevationRandomWalkParams",
    "ElevationUniformParams",
    "GateFramesParams",
    "GateFramesPosedParams",
    "GateRelaxParams",
    "GateValidityParams",
    "LiftFlatParams",
    "OffsetParams",
    "StationsParams",
    "ValidityParams",
    "WidthConstantParams",
    "WidthFromChannelParams",
    "XpbdParams",
]

_ALIGNS = ("yaw_only", "full_tangent")


@dataclasses.dataclass(frozen=True, kw_only=True)
class _StageFlatParams(_FlatParams):
    """A stage's flat Params: ``from_legacy`` copies the fields the config has (a
    ``TrackGenConfig`` or a ``GateGenConfig``) and leaves the others at their class
    defaults."""

    @classmethod
    def from_legacy(cls, cfg):
        return cls(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cls)
                      if hasattr(cfg, f.name)})

_HW_DOC = ("Track half-width: every point's (cl.half_width; at constant width the "
           "consumers take it as a host scalar).")


def _check_half_width(p) -> None:
    if float(p.half_width) <= 0.0:
        raise ValueError(f"half_width must be > 0, got {p.half_width!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstantSpacingParams(_FlatParams):
    """``resample.constant_spacing``: arc-length resample of the raw loop to a constant
    step, padded to ``N_max``."""

    spacing: float = static(0.6 * 0.1, doc="Constant arc-length step between points "
                                           "(> 0).")

    def validate(self, layout) -> None:
        if float(self.spacing) <= 0.0:
            raise ValueError(f"spacing must be > 0, got {self.spacing!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class XpbdParams(_FlatParams):
    """``relax.xpbd``: the XPBD relaxation (band / rest length setup + solve). The
    half-width is ``cl.half_width`` (the width stage runs before relax)."""

    relax_band: "int | None" = static(None, doc="Separation band override (None: auto).")
    relax_iters: int = static(50, doc="XPBD Jacobi sweeps.")
    relax_margin: float = static(0.15, doc="Separation target margin.")
    relax_sep_relax: float = static(1.0, doc="Separation correction scale.")
    relax_spc_relax: float = static(1.0, doc="Edge-spacing correction scale.")
    relax_bend_relax: float = static(1.5, doc="Bending correction scale.")
    relax_sep_every: int = static(20, lo=1, doc="Broadphase refresh interval (>= 1).")
    relax_sep_cache_slots: int = static(16, lo=0, doc="Separation cache slots per bead "
                                                      "(0: no cache).")
    relax_sep_cache_skin: float = static(0.5, lo=0.0, doc="Broadphase skin (>= 0).")
    relax_accel: str = static("chebyshev", choices=("none", "chebyshev"),
                              doc="Sweep acceleration.")
    relax_cheby_rho: float = static(0.98, doc="Chebyshev spectral radius, in [0, 1).")
    relax_cheby_gamma: float = static(0.9, doc="Chebyshev under-relaxation, in (0, 1].")
    relax_cheby_start: int = static(8, lo=1, doc="First accelerated sweep (>= 1).")
    relax_smooth_passes: int = static(5, lo=0, doc="Taubin smoothing passes (>= 0).")
    relax_smooth_spacing_iters: int = static(10, lo=0, doc="Spacing-only polish sweeps "
                                                           "(>= 0).")

    def validate(self, layout) -> None:
        if int(self.relax_sep_every) < 1:
            raise ValueError(f"relax_sep_every must be >= 1, got {self.relax_sep_every!r}")
        if int(self.relax_sep_cache_slots) < 0:
            raise ValueError(
                f"relax_sep_cache_slots must be >= 0, got {self.relax_sep_cache_slots!r}")
        if float(self.relax_sep_cache_skin) < 0.0:
            raise ValueError(
                f"relax_sep_cache_skin must be >= 0, got {self.relax_sep_cache_skin!r}")
        if str(self.relax_accel) not in ("none", "chebyshev"):
            raise ValueError(
                f"relax_accel must be 'none' or 'chebyshev', got {self.relax_accel!r}")
        if not (0.0 <= float(self.relax_cheby_rho) < 1.0):
            raise ValueError(
                f"relax_cheby_rho must be in [0, 1), got {self.relax_cheby_rho!r}")
        if not (0.0 < float(self.relax_cheby_gamma) <= 1.0):
            raise ValueError(
                f"relax_cheby_gamma must be in (0, 1], got {self.relax_cheby_gamma!r}")
        if int(self.relax_cheby_start) < 1:
            raise ValueError(
                f"relax_cheby_start must be >= 1, got {self.relax_cheby_start!r}")
        if int(self.relax_smooth_passes) < 0:
            raise ValueError(
                f"relax_smooth_passes must be >= 0, got {self.relax_smooth_passes!r}")
        if int(self.relax_smooth_spacing_iters) < 0:
            raise ValueError(
                "relax_smooth_spacing_iters must be >= 0, got "
                f"{self.relax_smooth_spacing_iters!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class WidthConstantParams(_FlatParams):
    """``width.constant``: one half-width for every point."""

    half_width: float = static(0.1, doc=_HW_DOC)

    def validate(self, layout) -> None:
        _check_half_width(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class WidthFromChannelParams(Params):
    """``width.from_channel``: ``cl.half_width = clamp(scale * src, lo, hi)`` from a
    float32 per-point channel the user wires (``StageRef.ports={"src": "cl.<pkg>.<name>"}``,
    e.g. the carried image of a generator's ``raw.<pkg>.<name>``). A NaN source point
    stays NaN (and fails the width floor). ``lo`` defaults to 0: a source that pinches
    below ``w_floor`` fails validity bit 4 instead of being clamped away. ``lo <= hi`` is
    checked at build; a run-time ``hi`` below ``lo`` clamps to ``lo`` (the kernel uses
    ``max(hi, lo)`` as the upper bound)."""

    scale: float = dynamic(1.0, lo=0.0, doc="Factor applied to the source (>= 0).")
    lo: float = dynamic(0.0, lo=0.0, doc="Lower clamp (>= 0).")
    hi: float = dynamic(1.0e9, lo=0.0, doc="Upper clamp (>= 0 and >= lo).")

    def validate(self, layout) -> None:
        if float(self.lo) > float(self.hi):
            raise ValueError(f"lo must be <= hi, got {self.lo!r} > {self.hi!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class OffsetParams(_FlatParams):
    """``offset.default``: the outer / inner borders at ``cl.half_width`` (since
    0.2.0.dev4 the width is the channel, written by the width stage)."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ValidityParams(_FlatParams):
    """``validity.default``: the fused per-env validity gate (the thickness and width
    floor checks read ``cl.half_width``)."""

    turning_tol: float = static(0.1, doc="Turning-number tolerance (rad).")
    w_floor: float = static(1e-3, doc="Width floor: every real point's cl.half_width "
                                      "must exceed it (bit 4).")
    relax_tol: float = static(0.02, doc="Thickness tolerance: valid when every point's "
                                        "clearance >= (1 - relax_tol) * its half-width.")
    z_valid_grade: float = static(0.0, lo=0.0, doc="Maximum |dz|/ds grade (0: off).")
    validity_border_check: bool = static(False, doc="Add the border self-intersection "
                                                    "check.")

    def validate(self, layout) -> None:
        if float(self.z_valid_grade) < 0.0:
            raise ValueError(f"z_valid_grade must be >= 0, got {self.z_valid_grade!r}")


def _check_z_range(p) -> None:
    if float(p.z_min) > float(p.z_max):
        raise ValueError(f"z_min must be <= z_max, got {p.z_min!r} > {p.z_max!r}")


def _check_control_points(p) -> None:
    if int(p.z_control_points) < 3:
        raise ValueError(f"z_control_points must be >= 3, got {p.z_control_points!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElevationFlatParams(_StageFlatParams):
    """``elevation.flat``: every real point at ``z_base``. On a gate pipeline it fills
    ``gates.z`` with ``z_base`` (``GateGenConfig.z_base``); on a track it must stay 0
    (the flat lift adds ``presets.track(z_base=...)``)."""

    z_base: float = static(0.0, doc="Altitude of every real gate (gate pipelines; a "
                                    "track's flat altitude is lift.flat's z_base).")


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElevationUniformParams(_StageFlatParams):
    """``elevation.uniform``: i.i.d. knot altitudes in ``[z_min, z_max]``."""

    z_min: float = static(0.0, doc="Lower altitude clamp (<= z_max).")
    z_max: float = static(0.0, doc="Upper altitude clamp.")
    z_control_points: int = static(10, lo=3, doc="Arc-spaced altitude knots (>= 3).")

    def validate(self, layout) -> None:
        _check_z_range(self)
        _check_control_points(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElevationRandomWalkParams(_StageFlatParams):
    """``elevation.random_walk``: a grade-capped random walk over the knots."""

    z_base: float = static(0.0, doc="Baseline altitude.")
    z_min: float = static(0.0, doc="Lower altitude clamp (<= z_max).")
    z_max: float = static(0.0, doc="Upper altitude clamp.")
    z_max_step: float = static(0.0, lo=0.0, doc="Grade cap |dz| per unit arc (>= 0).")
    z_control_points: int = static(10, lo=3, doc="Arc-spaced altitude knots (>= 3).")

    def validate(self, layout) -> None:
        _check_z_range(self)
        if float(self.z_max_step) < 0.0:
            raise ValueError(f"z_max_step must be >= 0, got {self.z_max_step!r}")
        _check_control_points(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElevationNoiseParams(_StageFlatParams):
    """``elevation.noise``: periodic harmonic noise around ``z_base``."""

    z_base: float = static(0.0, doc="Baseline altitude.")
    z_min: float = static(0.0, doc="Lower altitude clamp (<= z_max).")
    z_max: float = static(0.0, doc="Upper altitude clamp.")
    z_noise_amplitude: float = static(0.0, lo=0.0, doc="Oscillation amplitude (>= 0).")
    z_noise_harmonics: int = static(3, lo=1, doc="Summed harmonics (>= 1).")

    def validate(self, layout) -> None:
        _check_z_range(self)
        if float(self.z_noise_amplitude) < 0.0:
            raise ValueError(
                f"z_noise_amplitude must be >= 0, got {self.z_noise_amplitude!r}")
        if int(self.z_noise_harmonics) < 1:
            raise ValueError(
                f"z_noise_harmonics must be >= 1, got {self.z_noise_harmonics!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class LiftFlatParams(_FlatParams):
    """``lift.flat``: the constant altitude of every point."""

    z_base: float = static(0.0, doc="Altitude of every point.")


# ---------------------------------------------------------------------------------------
# The gate pipeline (roadmap PR11)
# ---------------------------------------------------------------------------------------

def _f32_distance(gate_radius) -> float:
    """``float32(2 * gate_radius)``: the target distance the relax kernel receives."""
    return float(np.float32(2.0 * float(gate_radius)))


@dataclasses.dataclass(frozen=True, kw_only=True)
class GateRelaxParams(_StageFlatParams):
    """``gate_relax.spheres``: separate overlapping gate spheres (plan view) to a centre
    distance of ``2 * gate_radius``, for at most ``gate_solve_iters`` sweeps."""

    gate_radius: float = static(0.025, lo=0.0, doc="Gate sphere radius (the centre "
                                                   "distance is 2 * gate_radius).")
    gate_solve_iters: int = static(8, lo=0, doc="Maximum separation sweeps (>= 1).")

    def validate(self, layout) -> None:
        if int(self.gate_solve_iters) < 1 or not _f32_distance(self.gate_radius) > 0.0:
            raise ValueError("gate_relax.spheres needs gate_solve_iters >= 1 and a "
                             "gate_radius whose float32 2*r is > 0; use gate_relax.none "
                             "to skip the solve")


@dataclasses.dataclass(frozen=True, kw_only=True)
class GateFramesParams(_StageFlatParams):
    """``gate_frames.default``: the gate frames, ``half_size = 0.5 * gate_width``."""

    gate_width: float = static(0.0, lo=0.0, doc="FULL gate bar width (half_size = 0.5 "
                                                "* gate_width; 0: point gates).")
    gate_align: str = static("yaw_only", choices=_ALIGNS,
                             doc="yaw_only (level gates) or full_tangent (tilted).")

    def validate(self, layout) -> None:
        if float(self.gate_width) < 0.0:
            raise ValueError(f"gate_width must be >= 0, got {self.gate_width!r}")
        if self.gate_align not in _ALIGNS:
            raise ValueError("gate_align must be one of {'yaw_only', 'full_tangent'}, got "
                             f"{self.gate_align!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class GateFramesPosedParams(_StageFlatParams):
    """``gate_frames.posed``: the gate frames of a generator that provides per-gate
    ``half_size`` and ``roll``."""

    gate_align: str = static("yaw_only", choices=_ALIGNS,
                             doc="yaw_only (level gates) or full_tangent (tilted).")

    def validate(self, layout) -> None:
        if self.gate_align not in _ALIGNS:
            raise ValueError("gate_align must be one of {'yaw_only', 'full_tangent'}, got "
                             f"{self.gate_align!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class GateValidityParams(_StageFlatParams):
    """``validity.gate_default``: the fused per-env gate validity check."""

    min_gates: int = static(4, lo=2, doc="Fewest real gates of a valid env (>= 2).")
    gate_radius: float = static(0.025, lo=0.0, doc="Gate sphere radius: real gates "
                                                   "closer than 2 * gate_radius fail.")
    gate_width: float = static(0.0, lo=0.0, doc="FULL gate bar width; > 0 turns the "
                                                "bar-crossing check on.")
    z_valid_grade: float = static(0.0, lo=0.0, doc="Maximum |dz|/ds chord grade (0: "
                                                   "off).")
    check_crossing: str = static("width", choices=("width", "always"),
                                 doc="'width': the crossing check runs when gate_width > "
                                     "0; 'always': it always runs (posed gates).")

    def validate(self, layout) -> None:
        if int(self.min_gates) < 2:
            raise ValueError(f"min_gates must be >= 2, got {self.min_gates!r}")
        if float(self.gate_radius) < 0.0:
            raise ValueError(f"gate_radius must be >= 0, got {self.gate_radius!r}")
        if float(self.gate_width) < 0.0:
            raise ValueError(f"gate_width must be >= 0, got {self.gate_width!r}")
        if float(self.z_valid_grade) < 0.0:
            raise ValueError(f"z_valid_grade must be >= 0, got {self.z_valid_grade!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class StationsParams(_StageFlatParams):
    """``stations.raw`` / ``stations.relaxed``: ``gates`` anchors at equal arc length
    along a centerline loop: plan-view (``spacing_metric="xy"``) or 3D (``"xyz"``, which
    needs the loop's altitude: a generator that provides ``z``, or an elevation run on
    the relaxed centerline)."""

    gates: int = static(12, lo=3, doc="Stations (anchors) per env (>= 3).")
    spacing_metric: str = static("xy", choices=("xy", "xyz"),
                                 doc="Arc length the stations are equally spaced in: "
                                     "plan view (xy) or 3D (xyz).")

    def validate(self, layout) -> None:
        if self.spacing_metric not in ("xy", "xyz"):
            raise ValueError(f"spacing_metric must be 'xy' or 'xyz', got "
                             f"{self.spacing_metric!r}")
