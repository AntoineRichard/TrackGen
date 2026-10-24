"""``randomize.uniform``: per-env uniform draws of another stage's DYNAMIC param, in the
graph (roadmap PR13b: curriculum and domain randomization).

A randomizer is an ordinary STATIC stage of kind ``randomize`` (built-in only in 0.x),
listed right BEFORE the stage whose field it draws (only the generator's randomizers may
precede the generator): every generation it writes ``u * (hi - lo) + lo`` per env into
the target field's array, keyed by the env's effective seed and its own salt, so the
draws are deterministic per ``(seed, epoch)``, differ across envs, are redrawn by
``advance=True`` (every replay of a caller graph) and reach the target in the SAME
generation. Its ``lo`` / ``hi`` are dynamic (a curriculum writes them, in its own graph
too). A drawn field is a single UNLATCHED array owned by the randomizer
(``params[stage][field]`` observes it; ``set`` raises). The drawable fields are the
dynamic float32 fields of the pipeline: the built-in generators' shape knobs and the
elevation profiles' window / grade / amplitude (roadmap PR13c, e.g.
``"generator.rad"``, ``"generator.polar_radial_jitter"``, ``"elevation.z_max"``; roadmap
PR13d: the course size, ``"generator.size"`` on every built-in track generator but
``external`` and on the bezier / hull gate generators, ``"normalize.size"`` on the
polar / voronoi / checkpoint gate pipelines, and ``"gate_frames.gate_width"``),
plugin Params' and ``width.from_channel``'s ``scale`` / ``lo`` / ``hi``. An ``int32``
dynamic field (bezier / hull's ``min_num_points``) is not drawable in 0.x. The initial
range must lie inside the field's declared bounds (compared in float32) and the draws
are clamped to them. A knob bezier's ``style_sampling`` draws (its ``*_range`` set) is
not drawable (set the range to ``None``): ``style_sampling``'s ``rad_range`` /
``handle_clamp_frac_range`` are superseded by ``randomize={"generator.rad": ...,
"generator.handle_clamp_frac": ...}`` (different random streams, same distributions).
"""
from __future__ import annotations

import dataclasses

import warp as wp

from ..ops.hash import hash32
from ..plugin.params import Params, dynamic, static
from ..plugin.stage import CaptureClass, Stage

__all__ = ["RandomizeUniform", "RandomizeUniformParams"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class RandomizeUniformParams(Params):
    """``randomize.uniform``: ``params[stage][field] = lo + u * (hi - lo)`` per env, ``u``
    uniform in ``[0, 1)`` (clamped to the field's declared ``lo`` / ``hi``)."""

    stage: str = static("", doc="Id of the stage whose field is drawn (listed after this "
                                "one).")
    field: str = static("", doc="A DYNAMIC float32 field of that stage's Params.")
    lo: float = dynamic(0.0, doc="Lower end of the draw (per env; write it for a "
                                 "curriculum).")
    hi: float = dynamic(1.0, doc="Upper end of the draw (per env).")

    def validate(self, layout) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("randomize.uniform: stage must name the stage whose field is "
                             f"drawn, got {self.stage!r}")
        if not isinstance(self.field, str) or not self.field or "." in self.field:
            raise ValueError("randomize.uniform: field must name a param of that stage "
                             f"(no dots), got {self.field!r}")
        if float(self.lo) > float(self.hi):
            raise ValueError(f"randomize.uniform: lo must be <= hi, got {self.lo!r} > "
                             f"{self.hi!r}")


@wp.kernel
def _randomize_uniform_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32,
                         lo: wp.array(dtype=wp.float32), hi: wp.array(dtype=wp.float32),
                         f_lo: float, f_hi: float, out: wp.array(dtype=wp.float32)):
    e = wp.tid()
    st = wp.rand_init(wp.int32(hash32(wp.uint32(seed[e]) ^ salt) & wp.uint32(0x7FFFFFFF)))
    v = lo[e] + wp.randf(st) * (hi[e] - lo[e])
    if not wp.isnan(v):
        v = wp.min(wp.max(v, f_lo), f_hi)          # the target field's declared bounds
    out[e] = v


class RandomizeUniform(Stage):
    """``randomize.uniform`` (see the module docstring)."""

    kind = "randomize"
    name = "uniform"
    Params = RandomizeUniformParams
    capture = CaptureClass.STATIC
    reads = {"seed": "seed"}
    writes = {}
    output = None
    #: Writes another stage's dynamic param (the planner's param-writer rules key on it).
    _param_writer = True

    def build(self, b, p):
        out, f_lo, f_hi = b._drawn_target()
        lo, hi = b.param("lo"), b.param("hi")
        salt = wp.uint32(b.salt("draw"))
        seed, E = b.seed, b.E

        def enqueue(ctx):
            ctx.launch(_randomize_uniform_k, E, [seed, salt, lo, hi, f_lo, f_hi, out])
        return enqueue


CATALOGUE_STAGES = (RandomizeUniform,)
