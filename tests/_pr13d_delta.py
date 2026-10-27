"""The PR13d launch-sequence delta (roadmap PR13d, design section 6), as code.

Layered on ``tests/_pr13c_delta.py``. From a PR13c trace (or HEAD's legacy-composition
trace through :func:`tests._pr13c_delta.pr13c_sequence`, which this module applies first
when asked), a generation's PR13d trace is:

1. **The latch block** right after the effective-seed record (``_effective_seed_k``;
   masked: ``_advance_masked_k``, ``_effective_seed_masked_k``, then the block, then the
   hold): the Program's LATCHED dynamic fields in slot order then Params field order,
   each stage's latched per-env inputs after its fields, one ``_latch_all_{i32,f32}_k``
   (masked ``_latch_masked_*``) of dim ``(E,)`` per entry by dtype; a drawn field is
   not latched. :func:`latch_dtypes` computes it from the declarations; PR13d adds
   bezier / hull ``min_num_points`` (i32, FIRST: a base-class field) and ``size`` (f32,
   LAST of the generator), repulsive's ``size``, the point gate generators' ``size``,
   ``normalize.size`` and ``gate_frames.gate_width``.
2. **Twins** (:data:`TWINS`): ``_sample_counts_h_k`` -> ``_sample_counts_dyn_h_k`` (scalar
   3, ``lo``, becomes an array; the bezier / hull TRACK generators only: the gate ones
   keep a static count); ``_finalize_frame_k`` -> ``_finalize_frame_dyn_k`` (scalar 9
   becomes an array); ``_finalize_validity_k`` -> ``_finalize_validity_bars_k`` (scalar
   11, the crossing toggle, REMOVED: the later scalar indices move down by one; every
   gate pipeline). Arrays APPENDED (scalars unchanged): the sized grid / normalize
   twins.
3. **Repulsive**: the generator's final ``copy`` record (the last copy before the
   constant-spacing resample, or of the sequence without one: the frozen entry point's
   own trace) becomes ``("launch", "..._copy_sized_k", (E*N,), ((1, N),))``.
4. Everything else unchanged (dims, arrays, other scalars, order). External: none.

Shared by ``tests/test_dynamic_knobs.py``, ``tests/test_gate_pipeline_parity.py`` (the
dd181f0 golden is not re-captured: the rules are applied to it),
``tests/test_commit.py`` / ``tests/test_gate_commit.py`` (the block checks) and
``tests/test_difficulty_axes.py``.
"""
from __future__ import annotations

import warp as wp

from tests._pr13c_delta import _short, pr13c_sequence

_BINDING = "track_gen._src.generator_binding"
LATCH_ALL = {"f32": f"{_BINDING}._latch_all_f32_k", "i32": f"{_BINDING}._latch_all_i32_k"}
LATCH_MASKED = {"f32": f"{_BINDING}._latch_masked_f32_k",
                "i32": f"{_BINDING}._latch_masked_i32_k"}
_LATCH_NAMES = {_short(k) for k in (*LATCH_ALL.values(), *LATCH_MASKED.values())}
_DTYPE = {wp.float32: "f32", wp.int32: "i32"}

#: scalar kernel -> (its PR13d twin, the scalar argument indices it takes as [E] arrays
#: or drops, whether the later scalar indices move down: the argument is REMOVED).
TWINS = {
    "_sample_counts_h_k": ("_sample_counts_dyn_h_k", (3,), False),
    "_sample_grid_points_h_k": ("_sample_grid_points_sized_h_k", (), False),
    "_sample_grid_points_scaled_h_k": ("_sample_grid_points_scaled_sized_h_k", (), False),
    "_normalize_extent_k": ("_normalize_extent_sized_k", (), False),
    "_normalize_extent_to_k": ("_normalize_extent_to_sized_k", (), False),
    "_finalize_frame_k": ("_finalize_frame_dyn_k", (9,), False),
    "_finalize_validity_k": ("_finalize_validity_bars_k", (11,), True),
}
REPULSIVE_COPY = "track_gen._src.warp_generate_repulsive._copy_sized_k"

#: The latched dynamic fields per built-in unit under PR13d (design section 6.2), by
#: dtype, in Params field order (a field ``style_sampling`` or a randomizer draws drops
#: out). Cross-checks of :func:`latch_dtypes`.
GENERATOR_LATCHES = {"bezier": ("i32", "f32", "f32", "f32"), "hull": ("i32", "f32", "f32"),
                     "polar": ("f32",) * 3, "voronoi": ("f32",) * 3,
                     "checkpoint": ("f32",) * 4, "repulsive": ("f32",), "external": ()}
GATE_GENERATOR_LATCHES = {"bezier": ("f32",), "hull": ("f32",), "polar": ("f32",) * 2,
                          "voronoi": ("f32",) * 2, "checkpoint": ("f32",) * 2}
#: normalize.size (the polar / voronoi / checkpoint gate pipelines).
NORMALIZE_LATCHES = ("f32",)
#: gate_frames.default's gate_width (every built-in gate pipeline).
GATE_FRAMES_LATCHES = ("f32",)
ELEVATION_LATCHES = {"flat": (), "uniform": ("f32",) * 2, "random_walk": ("f32",) * 3,
                     "noise": ("f32",) * 3}
NORMALIZING = ("polar", "voronoi", "checkpoint")


def gate_latches(name: str, z: str = "flat") -> "list[str]":
    """The latch block of a built-in anchors gate pipeline (slot order: generator,
    normalize, elevation, gate_frames)."""
    return [*GATE_GENERATOR_LATCHES[name],
            *(NORMALIZE_LATCHES if name in NORMALIZING else ()),
            *ELEVATION_LATCHES[z], *GATE_FRAMES_LATCHES]


def latch_dtypes(program) -> "list[str]":
    """The latch block of ``program`` computed from its declarations (rule 1): per
    stage in slot order, its latched dynamic fields in Params field order (a drawn field
    is not latched), then its latched per-env inputs; ``"f32"`` / ``"i32"`` each."""
    out = []
    infos = program.input_info
    for sid in program.spec.ids:
        sp = program.params[sid] if sid in program.params else None
        if sp is not None and sp.params is not None:
            from track_gen.plugin.params import _fields  # noqa: PLC0415

            for f in _fields(type(sp.params)):
                if f.kind == "dynamic" and f.name not in sp.drawn:
                    out.append(_DTYPE[f.dtype])
        for name, info in infos.items():
            if info.stage_id == sid and info.per == "env" and info.latched:
                out.append(_DTYPE[info.dtype])
    return out


def _as_rows(rows) -> "tuple[list, bool]":
    as_list = bool(rows) and isinstance(rows[0], list)
    return [tuple(r) for r in rows], as_list


def twin_record(op, kernel, dim, scalars, *, count_twin: bool = True):
    """``(op, kernel, dim, scalars)`` of a PR13c record as its PR13d twin's (unchanged
    without a twin; ``count_twin=False`` keeps ``_sample_counts_h_k``: a gate point
    generator's static count). ``scalars``: ``(index, value)`` pairs, tuples or lists."""
    name = _short(kernel)
    if name not in TWINS or (name == "_sample_counts_h_k" and not count_twin):
        return op, kernel, dim, scalars
    twin, idx, shift = TWINS[name]
    as_list = isinstance(scalars, list)
    kept = []
    for s in scalars:
        i, v = s[0], s[1]
        if i in idx:
            continue
        if shift and isinstance(i, int):
            i = i - sum(1 for d in idx if d < i)
        kept.append([i, v] if as_list else (i, v))
    new_kernel = str(kernel)[:-len(name)] + twin
    return op, new_kernel, dim, kept if as_list else tuple(kept)


def pr13d_sequence(rows, latches, E: int, *, masked: bool = False,
                   count_twin: bool = True, legacy: bool = False,
                   repulsive: bool = False) -> list:
    """``rows`` (``(op, kernel, dim, scalars)`` records, or JSON lists) under the PR13d
    rule: every latch record dropped, the twins renamed (:func:`twin_record`), the latch
    block of ``latches`` (``"f32"`` / ``"i32"`` each; masked: ``_latch_masked_*``)
    inserted right after the effective-seed record, and with ``repulsive`` the
    generator's final copy replaced by :data:`REPULSIVE_COPY`. ``legacy=True``: ``rows``
    are a HEAD legacy-composition trace (the scalar kernels): the PR13c twin renames
    (``pr13c_sequence`` with no latch) are applied first."""
    rows, as_list = _as_rows(rows)
    if legacy:
        rows = [tuple(r) for r in pr13c_sequence([list(r) if as_list else r for r in rows],
                                                 0, E)]
    rows = [r for r in rows if _short(r[1]) not in _LATCH_NAMES]
    if repulsive:
        cs = next((i for i, r in enumerate(rows) if _short(r[1]) == "_cs_scan_k"),
                  len(rows))
        k = max(i for i in range(cs) if rows[i][0] == "copy")
        n = int(rows[k][2][0])
        N = n // E
        rows[k] = ("launch", REPULSIVE_COPY, [n] if as_list else (n,),
                   [[1, N]] if as_list else ((1, N),))
    out, done = [], False
    kinds = LATCH_MASKED if masked else LATCH_ALL
    for op, kernel, dim, scalars in rows:
        op, kernel, dim, scalars = twin_record(op, kernel, dim, scalars,
                                               count_twin=count_twin)
        out.append([op, kernel, dim, scalars] if as_list else (op, kernel, dim, scalars))
        if not done and _short(kernel) in ("_effective_seed_k", "_effective_seed_masked_k"):
            for d in latches:
                out.append(["launch", kinds[d], [E], []] if as_list else
                           ("launch", kinds[d], (E,), ()))
            done = True
    assert done or not latches, "no effective-seed record to insert the latches after"
    return out


def assert_latch_block(program, plain, masked=None) -> "list[str]":
    """Rule 1 on traces of ``program`` (``LaunchRecord`` sequences): ``plain`` (one
    generation) has the latch block of :func:`latch_dtypes` right after
    ``_effective_seed_k`` and no other latch record; ``masked`` (a masked advance), when
    given: the advance, ``_effective_seed_masked_k``, the ``_latch_masked_*`` block, then
    the hold (``_hold_k`` / ``_gate_hold_k``), and no other latch. Returns the block."""
    want = latch_dtypes(program)
    names = [_short(r.kernel) for r in plain]
    i = names.index("_effective_seed_k")
    block = names[i + 1:i + 1 + len(want)]
    assert block == [_short(LATCH_ALL[d]) for d in want], (block, want)
    assert i + 1 + len(want) == len(names) or names[i + 1 + len(want)] not in _LATCH_NAMES
    assert sum(n in _LATCH_NAMES for n in names) == len(want), names
    if masked is not None:
        m = [_short(r.kernel) for r in masked]
        j = m.index("_effective_seed_masked_k")
        assert m[:j] == ["_advance_masked_k"], m[:j]
        mblock = m[j + 1:j + 1 + len(want)]
        assert mblock == [_short(LATCH_MASKED[d]) for d in want], (mblock, want)
        assert m[j + 1 + len(want)] in ("_hold_k", "_gate_hold_k"), m[j + 1 + len(want)]
        assert sum(n in _LATCH_NAMES for n in m) == len(want), m
    return want
