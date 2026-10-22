"""Params classes of the built-in track stages (roadmap PR10a).

Like the built-in generators' Params (:mod:`track_gen._src.builtin_params`), every class
keeps the FLAT ``TrackGenConfig`` field names, every field is static, and
``from_legacy(config)`` copies the flat values (no casts). The stage hands its Params
instance (or, for elevation, a namespace built from it) to the legacy wrappers as their
``config`` argument, so the same values reach the same ``float()`` / ``int()``
conversions and the launches are unchanged.

``validate()`` re-runs the ``TrackGenConfig.__post_init__`` rules of the class's own
fields with the same messages: a spec made from a ``TrackGenConfig`` always passes; the
rules exist for hand-built specs.
"""
from __future__ import annotations

import dataclasses

from .._src.builtin_params import _FlatParams
from ..plugin.params import static

__all__ = [
    "ConstantSpacingParams",
    "ElevationNoiseParams",
    "ElevationRandomWalkParams",
    "ElevationUniformParams",
    "LiftFlatParams",
    "OffsetParams",
    "ValidityParams",
    "WidthConstantParams",
    "XpbdParams",
]

_HW_DOC = "Track half-width (the kernels take it as a host scalar until roadmap PR13)."


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
    """``relax.xpbd``: the XPBD relaxation (band / rest length setup + solve)."""

    half_width: float = static(0.1, doc=_HW_DOC)
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
        _check_half_width(self)
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
class OffsetParams(_FlatParams):
    """``offset.default``: the outer / inner borders at ``half_width``."""

    half_width: float = static(0.1, doc=_HW_DOC)

    def validate(self, layout) -> None:
        _check_half_width(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ValidityParams(_FlatParams):
    """``validity.default``: the fused per-env validity gate."""

    half_width: float = static(0.1, doc=_HW_DOC)
    turning_tol: float = static(0.1, doc="Turning-number tolerance (rad).")
    w_floor: float = static(1e-3, doc="Width floor of every real point.")
    relax_tol: float = static(0.02, doc="Thickness tolerance: valid when thickness >= "
                                        "(1 - relax_tol) * half_width.")
    z_valid_grade: float = static(0.0, lo=0.0, doc="Maximum |dz|/ds grade (0: off).")
    validity_border_check: bool = static(False, doc="Add the border self-intersection "
                                                    "check.")

    def validate(self, layout) -> None:
        _check_half_width(self)
        if float(self.z_valid_grade) < 0.0:
            raise ValueError(f"z_valid_grade must be >= 0, got {self.z_valid_grade!r}")


def _check_z_range(p) -> None:
    if float(p.z_min) > float(p.z_max):
        raise ValueError(f"z_min must be <= z_max, got {p.z_min!r} > {p.z_max!r}")


def _check_control_points(p) -> None:
    if int(p.z_control_points) < 3:
        raise ValueError(f"z_control_points must be >= 3, got {p.z_control_points!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElevationUniformParams(_FlatParams):
    """``elevation.uniform``: i.i.d. knot altitudes in ``[z_min, z_max]``."""

    z_min: float = static(0.0, doc="Lower altitude clamp (<= z_max).")
    z_max: float = static(0.0, doc="Upper altitude clamp.")
    z_control_points: int = static(10, lo=3, doc="Arc-spaced altitude knots (>= 3).")

    def validate(self, layout) -> None:
        _check_z_range(self)
        _check_control_points(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElevationRandomWalkParams(_FlatParams):
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
class ElevationNoiseParams(_FlatParams):
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
