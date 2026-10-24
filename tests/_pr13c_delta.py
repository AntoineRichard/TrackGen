"""The PR13c launch-sequence delta (roadmap PR13c, design section 4), as code.

The built-in knobs became dynamic params: a generation's trace is the 0.2.0.dev5 PR13b
trace (HEAD: the scalar kernels, which the frozen legacy entry points still launch) with

1. one ``_latch_all_f32_k`` record (masked: ``_latch_masked_f32_k``) of dim ``(E,)`` per
   LATCHED dynamic field, inserted right after the effective-seed record (after the
   advance when masked), in slot order then Params field order (a field a randomizer or
   ``style_sampling`` draws is not latched);
2. each knob kernel's record renamed to its array-reading twin and its knob scalars
   removed (the twin takes an ``[E]`` array at the same argument index); bezier's
   ``_assemble_k`` -> ``_assemble_dyn_k`` also gains its four bound scalars after them;
3. everything else unchanged (dims, arrays, the other scalars).

Shared by ``tests/test_pipeline_parity.py``, ``tests/test_gate_pipeline_parity.py``
(goldens not re-captured: the rule is applied to the recorded dd181f0 sequence) and
``tests/test_dynamic_knobs.py``.
"""
from __future__ import annotations

import math

LATCH_ALL = "track_gen._src.generator_binding._latch_all_f32_k"
LATCH_MASKED = "track_gen._src.generator_binding._latch_masked_f32_k"
_EFFECTIVE = ("_effective_seed_k", "_effective_seed_masked_k")

#: scalar kernel -> (its twin, the scalar argument indices that became [E] arrays).
TWINS = {
    "_assemble_k": ("_assemble_dyn_k", (6, 7)),
    "_midpoint_displace_k": ("_midpoint_displace_dyn_k", (4,)),
    "_polar_controls_k": ("_polar_controls_dyn_k", (2, 3)),
    "_sample_checkpoints_k": ("_sample_checkpoints_dyn_k", (3, 4)),
    "_steer_k": ("_steer_dyn_k", (4,)),
    "_select_anchor_sites_k": ("_select_anchor_sites_dyn_k", (6, 7)),
    "_z_uniform_k": ("_z_uniform_dyn_k", (3, 4)),
    "_z_walk_k": ("_z_walk_dyn_k", (6, 7, 8)),
    "_z_noise_k": ("_z_noise_dyn_k", (6, 7, 8)),
}
#: _assemble_dyn_k's bound scalars of a latched (rad, handle_clamp_frac): the declared
#: (lo, hi) of each (BezierParams), at argument indices 8..11.
ASSEMBLE_BOUNDS = ((8, 0.0), (9, 1.0), (10, 0.0), (11, math.inf))

#: Latched dynamic fields per built-in (design section 2.2).
GENERATOR_LATCHES = {"bezier": 2, "hull": 1, "polar": 2, "voronoi": 2, "checkpoint": 3,
                     "repulsive": 0}
GATE_GENERATOR_LATCHES = {"bezier": 0, "hull": 0, "polar": 2, "voronoi": 2,
                          "checkpoint": 2}
ELEVATION_LATCHES = {"flat": 0, "uniform": 2, "random_walk": 3, "noise": 3}


def _short(kernel) -> str:
    return str(kernel).rsplit(".", 1)[-1]


def twin_record(op, kernel, dim, scalars, *, assemble_bounds=ASSEMBLE_BOUNDS):
    """``(op, kernel, dim, scalars)`` of a scalar-kernel record as its twin's (the
    record unchanged when ``kernel`` has no twin). ``scalars``: ``(index, value)``
    pairs (tuples, or lists as in a JSON golden: the result keeps the input's type)."""
    name = _short(kernel)
    if name not in TWINS:
        return op, kernel, dim, scalars
    twin, knobs = TWINS[name]
    new_kernel = str(kernel)[:-len(name)] + twin
    kept = [s for s in scalars if s[0] not in knobs]
    if name == "_assemble_k":
        as_list = bool(scalars) and isinstance(scalars[0], list)
        kept += [list(b) if as_list else tuple(b) for b in assemble_bounds]
    return op, new_kernel, dim, type(scalars)(kept) if isinstance(scalars, tuple) \
        else kept


def latch_record(E: int, masked: bool, *, as_list: bool = False):
    kernel = LATCH_MASKED if masked else LATCH_ALL
    if as_list:
        return ["launch", kernel, [E], []]
    return ("launch", kernel, (E,), ())


def pr13c_sequence(rows, n_latch: int, E: int, *, masked: bool = False,
                   insert_after: "int | None" = None, twins: bool = True,
                   assemble_bounds=ASSEMBLE_BOUNDS) -> list:
    """HEAD's ``rows`` (``(op, kernel, dim, scalars)`` records, or JSON lists) under the
    PR13c rule: ``n_latch`` latch records inserted after the effective-seed record (or
    after record ``insert_after`` when given), and every knob kernel record renamed to
    its twin (``twins=False``: renaming skipped)."""
    as_list = bool(rows) and isinstance(rows[0], list)
    out, done = [], False
    for i, row in enumerate(rows):
        op, kernel, dim, scalars = row
        if twins:
            op, kernel, dim, scalars = twin_record(op, kernel, dim, scalars,
                                                   assemble_bounds=assemble_bounds)
        out.append([op, kernel, dim, scalars] if as_list else (op, kernel, dim, scalars))
        hit = (i == insert_after) if insert_after is not None else \
            (_short(kernel) in _EFFECTIVE)
        if hit and not done:
            out += [latch_record(E, masked, as_list=as_list) for _ in range(n_latch)]
            done = True
    assert done or n_latch == 0, "no effective-seed record to insert the latches after"
    return out
