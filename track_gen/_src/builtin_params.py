"""Params classes of the built-in track and gate generators.

Each class keeps the FLAT ``TrackGenConfig`` / ``GateGenConfig`` field names (so the
legacy error messages and ``generator_params={"polar_num_knots": 9}`` read the same),
and every field is static (the built-in kernels take host scalars). The generator
functions keep reading the flat config fields: the orchestrator resolves the Params
(``from_legacy`` when ``generator_params`` is ``None``) and writes them back onto its
private run snapshot (``_apply_legacy``), so the values the kernels see are exactly
the flat ones and outputs stay byte-identical.

``validate()`` holds the generator-specific rules that ``TrackGenConfig.__post_init__``
/ ``GateGenConfig.__post_init__`` used to run for every config: same conditions, same
messages, same order. They now run at generator construction, and only for the
selected generator. ``lo``/``hi`` are set only where a legacy rule rejects the same
inclusive bound, so the generic bound check never fires before a legacy message.

Not part of the public API: reach a class through ``registry.get(kind, name).params``
(a track generator's ``GeneratorSpec``), ``registry.get("gate_generator", name).Params``
(the gate generators are ``gate_generator`` Stages since roadmap PR11) or
``registry.describe(kind, name).params_schema``, or pass a dict.
"""
from __future__ import annotations

import dataclasses

from ..plugin.params import Params, static

__all__ = [
    "BezierParams",
    "CheckpointGateParams",
    "CheckpointParams",
    "HullParams",
    "PointGateParams",
    "PolarGateParams",
    "PolarParams",
    "RepulsiveParams",
    "VoronoiGateParams",
    "VoronoiParams",
]

_LAYOUTS = ("ring", "void_ring", "clustered", "mixed")


def _norm(v):
    """Comparable form of a flat value: lists -> tuples (``repulsive_stages`` may be a
    list), so ``(64, 128, 256) == [64, 128, 256]``."""
    if isinstance(v, list):
        return tuple(_norm(x) for x in v)
    if isinstance(v, tuple):
        return tuple(_norm(x) for x in v)
    return v


@dataclasses.dataclass(frozen=True, kw_only=True)
class _FlatParams(Params):
    """Built-in Params whose field names ARE the flat config field names."""

    _lenient_static_types = True        # the legacy flat values are not type-checked

    @classmethod
    def from_legacy(cls, cfg):
        # Raw values, no casts: the run snapshot then carries exactly the flat values.
        return cls(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cls)})

    def _names(self) -> "list[str]":
        return [f.name for f in dataclasses.fields(self)]

    def _apply_legacy(self, cfg) -> None:
        """Write every field onto ``cfg`` (the orchestrator's private run snapshot)."""
        for name in self._names():
            setattr(cfg, name, getattr(self, name))

    def _check_no_flat_conflict(self, cfg) -> None:
        """With ``generator_params`` given, a flat field this class owns must stay at
        its default (this class's, which equals the base config's) or equal the given
        value. A config subclass that changes the default of such a field therefore
        conflicts with a ``generator_params`` that leaves it at the class default."""
        defaults = {f.name: f.default for f in dataclasses.fields(type(self))}
        bad = sorted(
            n for n in self._names()
            if hasattr(cfg, n)
            and _norm(getattr(cfg, n)) != _norm(defaults[n])
            and _norm(getattr(cfg, n)) != _norm(getattr(self, n)))
        if bad:
            raise ValueError(
                f"generator_params and the flat {type(cfg).__name__} fields {bad} both "
                f"configure generator {cfg.generator!r} with different values; set them "
                f"in one place (generator_params, or the flat fields alone)")

    def _check_legacy_config(self, cfg) -> None:
        """Cross-field rules that need the (core) config; none by default."""


# ---------------------------------------------------------------------------------------
# Track generators
# ---------------------------------------------------------------------------------------

def _check_num_points_per_segment(p) -> None:
    if int(p.num_points_per_segment) < 2:
        raise ValueError(
            f"num_points_per_segment must be >= 2, got {p.num_points_per_segment!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PointFamilyParams(_FlatParams):
    """The corner sampler shared by ``"bezier"`` and ``"hull"``."""

    min_num_points: int = static(9, lo=2, doc="Minimum corner-anchor count per env.")
    max_num_points: int = static(13, doc="Maximum corner-anchor count per env "
                                         "(>= min_num_points).")
    num_points_per_segment: int = static(30, lo=2, doc="Dense samples per "
                                                       "corner-to-corner segment.")
    min_point_distance: float = static(0.05, hi=0.5, doc="Minimum grid-cell spacing for "
                                                         "corner sampling, in (0, 0.5].")

    def validate(self, layout) -> None:
        # Point-family sampler inputs: min_point_distance is a divisor
        # (num_cells = int(1/(min_point_distance*2))); min/max_num_points feed wp.randi.
        if int(self.min_num_points) < 2:
            raise ValueError(f"min_num_points must be >= 2, got {self.min_num_points!r}")
        if int(self.max_num_points) < int(self.min_num_points):
            raise ValueError(
                "max_num_points must be >= min_num_points, got "
                f"{self.max_num_points!r} < {self.min_num_points!r}")
        if not (0.0 < float(self.min_point_distance) <= 0.5):
            raise ValueError(
                f"min_point_distance must be in (0, 0.5], got {self.min_point_distance!r}")
        _check_num_points_per_segment(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class BezierParams(_PointFamilyParams):
    """Parameters of the ``"bezier"`` track generator."""

    rad: float = static(0.4, doc="Cubic-Bezier handle length as a fraction of the "
                                 "segment chord.")
    edgy: float = static(0.0, doc="Corner-tangent blend weight.")
    handle_clamp_frac: float = static(0.4, doc="Adaptive Bezier-handle clamp (fraction "
                                               "of the shorter incident edge).")
    style_sampling: bool = static(False, doc="Per-env style randomisation from the "
                                             "*_range fields.")
    rad_range: "tuple[float, float] | None" = static(
        None, doc="(lo, hi) per-env rad range when style_sampling.")
    scale_range: "tuple[float, float] | None" = static(
        None, doc="(lo, hi) per-env scale range when style_sampling.")
    handle_clamp_frac_range: "tuple[float, float] | None" = static(
        None, doc="(lo, hi) per-env handle_clamp_frac range when style_sampling.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class HullParams(_PointFamilyParams):
    """Parameters of the ``"hull"`` track generator."""

    hull_displacement: float = static(0.15, doc="Maximum per-edge radial midpoint "
                                                "displacement fraction.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolarParams(_FlatParams):
    """Parameters of the ``"polar"`` track generator (the polar_* knobs are clamped by
    the generator, so they carry no bounds)."""

    num_points_per_segment: int = static(30, lo=2, doc="Dense samples per knot segment.")
    polar_num_knots: int = static(12, doc="Periodic cubic-spline control knots "
                                          "(clamped to >= 4).")
    polar_radial_jitter: float = static(0.60, doc="Radial knot jitter as a fraction of "
                                                  "the base radius (clamped to [0, 0.85]).")
    polar_angular_jitter: float = static(0.30, doc="Angular knot jitter as a fraction of "
                                                   "a half cell (clamped to [0, 0.45]).")

    def validate(self, layout) -> None:
        _check_num_points_per_segment(self)


def _check_voronoi_layout(p) -> None:
    if p.voronoi_site_layout not in {"ring", "void_ring", "clustered", "mixed"}:
        raise ValueError(
            "voronoi_site_layout must be one of "
            "{'ring', 'void_ring', 'clustered', 'mixed'}, got "
            f"{p.voronoi_site_layout!r}")


def _check_voronoi_shape(p, floor: int) -> None:
    if int(p.voronoi_control_points) < floor:
        raise ValueError(
            f"voronoi_control_points must be >= {floor}, got {p.voronoi_control_points!r}")
    if int(p.voronoi_num_sites) < int(p.voronoi_control_points):
        raise ValueError(
            "voronoi_num_sites must be >= voronoi_control_points, got "
            f"{p.voronoi_num_sites!r} < {p.voronoi_control_points!r}")


def _check_voronoi_variation(p) -> None:
    if float(p.voronoi_radial_variation) < 0.0:
        raise ValueError(
            f"voronoi_radial_variation must be >= 0, got {p.voronoi_radial_variation!r}")
    if float(p.voronoi_angular_jitter) < 0.0:
        raise ValueError(
            f"voronoi_angular_jitter must be >= 0, got {p.voronoi_angular_jitter!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoronoiParams(_FlatParams):
    """Parameters of the ``"voronoi"`` track generator."""

    num_points_per_segment: int = static(30, lo=2, doc="Dense samples per anchor segment.")
    voronoi_num_sites: int = static(256, doc="Total cell sites (>= voronoi_control_points).")
    voronoi_site_layout: str = static("void_ring", choices=_LAYOUTS,
                                      doc="Site distribution layout.")
    voronoi_control_points: int = static(18, lo=6, doc="Angular anchor sites selected "
                                                       "from the site field.")
    voronoi_radial_variation: float = static(0.62, lo=0.0, doc="Radial modulation of "
                                                               "anchor targets.")
    voronoi_angular_jitter: float = static(0.08, lo=0.0, doc="Angular jitter of anchor "
                                                             "sector targets.")

    def validate(self, layout) -> None:
        _check_num_points_per_segment(self)
        _check_voronoi_shape(self, 6)
        _check_voronoi_layout(self)
        _check_voronoi_variation(self)


def _check_checkpoint_count(p) -> None:
    if int(p.checkpoint_count) < 3:
        raise ValueError(f"checkpoint_count must be >= 3, got {p.checkpoint_count!r}")


def _check_checkpoint_radius(p) -> None:
    if not (0.0 <= float(p.checkpoint_radius_min_frac) < 1.0):
        raise ValueError(
            "checkpoint_radius_min_frac must be in [0, 1), got "
            f"{p.checkpoint_radius_min_frac!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CheckpointParams(_FlatParams):
    """Parameters of the ``"checkpoint"`` track generator."""

    checkpoint_count: int = static(12, lo=3, doc="Number of radial checkpoints.")
    checkpoint_radius_min_frac: float = static(0.33, lo=0.0, doc="Inner-radius fraction "
                                                                 "of checkpoint placement, "
                                                                 "in [0, 1).")
    checkpoint_angle_jitter: float = static(0.55, doc="Angular jitter as a fraction of "
                                                      "the per-checkpoint slot.")
    checkpoint_turn_rate: float = static(0.42, doc="Maximum heading change (rad) per "
                                                   "steering step.")
    checkpoint_steer_gain: float = static(0.65, doc="Proportional steering gain, in (0, 1].")
    checkpoint_lookahead_frac: float = static(0.16, doc="Advance-to-next threshold as a "
                                                        "fraction of the outer radius.")
    checkpoint_best_of_k: int = static(4, lo=1, doc="Best-of-K candidate count.")
    checkpoint_clip_fallback: bool = static(False, doc="Clip the first self-crossing of "
                                                       "the selected centerline.")

    def validate(self, layout) -> None:
        _check_checkpoint_count(self)
        if int(self.checkpoint_best_of_k) < 1:
            raise ValueError(
                f"checkpoint_best_of_k must be >= 1, got {self.checkpoint_best_of_k!r}")
        _check_checkpoint_radius(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RepulsiveParams(_FlatParams):
    """Parameters of the ``"repulsive"`` track generator."""

    repulsive_grow_mult_min: float = static(4.5, lo=1.0, doc="Lower bound of the per-env "
                                                             "target-perimeter multiplier.")
    repulsive_grow_mult_max: float = static(5.5, doc="Upper bound of the target-perimeter "
                                                     "multiplier (> 1, >= min).")
    repulsive_domain_frac: float = static(0.35, doc="Confinement-disc radius fraction "
                                                    "(> 0).")
    repulsive_domain_init_ratio: float = static(4.0, doc="Domain radius over initial "
                                                         "seed-circle radius (> 1).")
    repulsive_obstacle_count_min: int = static(8, lo=1, doc="Lower bound of the inner-disc "
                                                            "obstacle count.")
    repulsive_obstacle_count_max: int = static(12, doc="Upper bound of the inner-disc "
                                                       "obstacle count (>= min).")
    repulsive_obstacle_radius_min_frac: float = static(0.02, doc="Lower bound of the "
                                                                 "inner-disc radius "
                                                                 "fraction (> 0).")
    repulsive_obstacle_radius_max_frac: float = static(0.045, doc="Upper bound of the "
                                                                  "inner-disc radius "
                                                                  "fraction (>= min).")
    repulsive_ratchet_rate: float = static(0.012, doc="Per-iteration length-growth factor "
                                                      "(> 0).")
    repulsive_alpha: float = static(3.0, doc="Tangent-point energy exponent alpha (> 0).")
    repulsive_beta: float = static(6.0, doc="Tangent-point energy exponent beta (> 0).")
    repulsive_tau: float = static(0.4, doc="Normalized flow step size.")
    repulsive_w_len: float = static(30.0, doc="Weight of the inert length regularizer.")
    repulsive_stages: "tuple[int, ...]" = static(
        (64, 128, 256), doc="Coarse-to-fine resolution schedule: strictly increasing "
                            "positive multiples of 4, the last equal to num_points.")
    repulsive_settle_iters: int = static(40, lo=1, doc="Settle-phase iteration budget.")
    repulsive_resample_every: int = static(25, lo=1, doc="Arc-length reparameterization "
                                                         "interval.")
    repulsive_stall_window: int = static(16, lo=1, doc="Iterations between per-env stall "
                                                       "checks.")
    repulsive_stall_area_tol: float = static(0.05, doc="Relative area-change tolerance of "
                                                       "the stall check.")
    repulsive_deactivate_obstacles: bool = static(True, doc="Zero the obstacle weights "
                                                            "once an env reaches its "
                                                            "target length.")

    def validate(self, layout) -> None:
        # grow_mult_min >= 1 and grow_mult_max > 1: a multiplier <= 1 never grows, which
        # drives the ratchet budget to <= 0 (an empty growth loop, garbage tracks).
        if float(self.repulsive_grow_mult_min) < 1.0:
            raise ValueError(
                f"repulsive_grow_mult_min must be >= 1, got {self.repulsive_grow_mult_min!r}")
        if float(self.repulsive_grow_mult_max) <= 1.0:
            raise ValueError(
                f"repulsive_grow_mult_max must be > 1, got {self.repulsive_grow_mult_max!r}")
        if float(self.repulsive_grow_mult_max) < float(self.repulsive_grow_mult_min):
            raise ValueError(
                "repulsive_grow_mult_max must be >= repulsive_grow_mult_min, got "
                f"{self.repulsive_grow_mult_max!r} < {self.repulsive_grow_mult_min!r}")
        if float(self.repulsive_domain_frac) <= 0.0:
            raise ValueError(
                f"repulsive_domain_frac must be > 0, got {self.repulsive_domain_frac!r}")
        if float(self.repulsive_domain_init_ratio) <= 1.0:
            raise ValueError(
                "repulsive_domain_init_ratio must be > 1, got "
                f"{self.repulsive_domain_init_ratio!r}")
        if int(self.repulsive_obstacle_count_min) < 1:
            raise ValueError(
                "repulsive_obstacle_count_min must be >= 1, got "
                f"{self.repulsive_obstacle_count_min!r}")
        if int(self.repulsive_obstacle_count_max) < int(self.repulsive_obstacle_count_min):
            raise ValueError(
                "repulsive_obstacle_count_max must be >= repulsive_obstacle_count_min, got "
                f"{self.repulsive_obstacle_count_max!r} < {self.repulsive_obstacle_count_min!r}")
        if float(self.repulsive_obstacle_radius_min_frac) <= 0.0:
            raise ValueError(
                "repulsive_obstacle_radius_min_frac must be > 0, got "
                f"{self.repulsive_obstacle_radius_min_frac!r}")
        if (float(self.repulsive_obstacle_radius_max_frac)
                < float(self.repulsive_obstacle_radius_min_frac)):
            raise ValueError(
                "repulsive_obstacle_radius_max_frac must be >= "
                "repulsive_obstacle_radius_min_frac, got "
                f"{self.repulsive_obstacle_radius_max_frac!r} < "
                f"{self.repulsive_obstacle_radius_min_frac!r}")
        if float(self.repulsive_ratchet_rate) <= 0.0:
            raise ValueError(
                f"repulsive_ratchet_rate must be > 0, got {self.repulsive_ratchet_rate!r}")
        if float(self.repulsive_alpha) <= 0.0:
            raise ValueError(f"repulsive_alpha must be > 0, got {self.repulsive_alpha!r}")
        if float(self.repulsive_beta) <= 0.0:
            raise ValueError(f"repulsive_beta must be > 0, got {self.repulsive_beta!r}")
        stages = tuple(self.repulsive_stages)
        if len(stages) == 0:
            raise ValueError("repulsive_stages must be non-empty")
        if any(int(s) <= 0 or int(s) % 4 != 0 for s in stages):
            raise ValueError(
                f"repulsive_stages entries must be positive multiples of 4, got {stages!r}")
        if any(int(stages[i]) <= int(stages[i - 1]) for i in range(1, len(stages))):
            raise ValueError(
                f"repulsive_stages must be strictly increasing, got {stages!r}")
        # settle_iters >= 1 keeps the growth loop non-empty; resample_every /
        # stall_window feed `(it+1) % k` guards (k=0 would divide by zero mid-generate).
        if int(self.repulsive_settle_iters) < 1:
            raise ValueError(
                f"repulsive_settle_iters must be >= 1, got {self.repulsive_settle_iters!r}")
        if int(self.repulsive_resample_every) < 1:
            raise ValueError(
                f"repulsive_resample_every must be >= 1, got {self.repulsive_resample_every!r}")
        if int(self.repulsive_stall_window) < 1:
            raise ValueError(
                f"repulsive_stall_window must be >= 1, got {self.repulsive_stall_window!r}")

    def _check_legacy_config(self, cfg) -> None:
        # The last stage is the tail input resolution (a cross-stage capacity rule).
        stages = tuple(self.repulsive_stages)
        if int(stages[-1]) != int(cfg.num_points):
            raise ValueError(
                "repulsive_stages[-1] must equal num_points for generator='repulsive', got "
                f"{stages[-1]!r} != {cfg.num_points!r}")


# ---------------------------------------------------------------------------------------
# Gate generators
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, kw_only=True)
class PointGateParams(_FlatParams):
    """Parameters of the ``"bezier"`` and ``"hull"`` gate generators (the corner
    sampler; the curve knobs are inert for gates)."""

    min_num_points: int = static(9, lo=2, doc="Minimum corner-anchor count per env.")
    max_num_points: int = static(13, doc="Maximum corner-anchor count per env "
                                         "(>= min_num_points).")
    min_point_distance: float = static(0.05, doc="Minimum grid-cell spacing for anchor "
                                                 "sampling (> 0).")

    def validate(self, layout) -> None:
        if float(self.min_point_distance) <= 0.0:
            raise ValueError(
                f"min_point_distance must be > 0, got {self.min_point_distance!r}")
        if int(self.min_num_points) < 2:
            raise ValueError(
                f"min_num_points must be >= 2, got {self.min_num_points!r}")
        if int(self.max_num_points) < int(self.min_num_points):
            raise ValueError(
                "max_num_points must be >= min_num_points, got "
                f"{self.max_num_points!r} < {self.min_num_points!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolarGateParams(_FlatParams):
    """Parameters of the ``"polar"`` gate generator."""

    polar_num_knots: int = static(12, doc="Periodic cubic-spline control knots (= gates).")
    polar_radial_jitter: float = static(0.60, doc="Radial knot jitter as a fraction of "
                                                  "the base radius.")
    polar_angular_jitter: float = static(0.30, doc="Angular knot jitter as a fraction of "
                                                   "a half cell.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoronoiGateParams(_FlatParams):
    """Parameters of the ``"voronoi"`` gate generator (anchors are emitted directly as
    gates, so three control points suffice)."""

    voronoi_num_sites: int = static(256, doc="Total cell sites (>= voronoi_control_points).")
    voronoi_site_layout: str = static("void_ring", choices=_LAYOUTS,
                                      doc="Site distribution layout.")
    voronoi_control_points: int = static(18, lo=3, doc="Anchor sites selected from the "
                                                       "site field (= gates).")
    voronoi_radial_variation: float = static(0.62, lo=0.0, doc="Radial modulation of "
                                                               "anchor targets.")
    voronoi_angular_jitter: float = static(0.08, lo=0.0, doc="Angular jitter of anchor "
                                                             "sector targets.")

    def validate(self, layout) -> None:
        _check_voronoi_shape(self, 3)
        if float(self.voronoi_radial_variation) < 0.0:
            raise ValueError(
                "voronoi_radial_variation must be >= 0, got "
                f"{self.voronoi_radial_variation!r}")
        if float(self.voronoi_angular_jitter) < 0.0:
            raise ValueError(
                "voronoi_angular_jitter must be >= 0, got "
                f"{self.voronoi_angular_jitter!r}")
        _check_voronoi_layout(self)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CheckpointGateParams(_FlatParams):
    """Parameters of the ``"checkpoint"`` gate generator."""

    checkpoint_count: int = static(12, lo=3, doc="Number of radial checkpoints (= gates).")
    checkpoint_radius_min_frac: float = static(0.33, lo=0.0, doc="Inner-radius fraction "
                                                                 "of checkpoint placement, "
                                                                 "in [0, 1).")
    checkpoint_angle_jitter: float = static(0.55, doc="Angular jitter as a fraction of "
                                                      "the per-checkpoint slot.")

    def validate(self, layout) -> None:
        _check_checkpoint_count(self)
        _check_checkpoint_radius(self)
