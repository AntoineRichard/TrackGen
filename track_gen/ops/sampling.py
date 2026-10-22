"""Per-env random sampling with explicit RNG keying (flat per-env layout).

RNG keying (kept byte-identical to the built-in generators): each env's stream is
``wp.rand_init(seeds[e] * salt + offset)`` evaluated in int32 wrap-around arithmetic.
Pick a ``(salt, offset)`` pair no other stream of your generator uses. The built-in
pairs are listed in the writing-a-generator guide; in particular:

========================  ===========  ==========================
stream                    salt         offset
========================  ===========  ==========================
bezier corner count       6151         ``attempt`` (0 built in)
bezier corner positions   9781         ``attempt`` (0 built in)
hull point count          5119         1
hull point positions      7919         1
========================  ===========  ==========================

Every op takes a :class:`~track_gen.exec.LaunchContext` first, launches only through
it, never host-synchronizes and never allocates.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .check import _check_1d, _check_float, _check_int, _check_min, check_array
from .fill import _fill_i32_k

if TYPE_CHECKING:  # pragma: no cover
    from ..exec.context import LaunchContext

__all__ = ["grid_cells", "sample_counts", "sample_grid_points"]


# ---------------------------------------------------------------------------------------
# Kernels (track_gen/_src/warp_generate.py _corner_count_sample_k / _corner_sample_k /
# _corner_sample_style_k at 1077b18, with the literal keying `seeds[e] * 9781 + attempt`
# parametrised as `seeds[e] * salt + offset`; same int32 ops, same draw order).
# ---------------------------------------------------------------------------------------

@wp.kernel
def _sample_counts_k(
    seeds: wp.array(dtype=wp.int32),
    salt: int,
    offset: int,
    lo: int,
    hi: int,
    out: wp.array(dtype=wp.int32),
):
    # One thread per env e. A single uniform randf in [0, 1) maps to an inclusive
    # integer count in [lo, hi] via floor(randf * span), span = hi - lo + 1, then a
    # clamp to hi folds the measure-zero randf == 1.0 edge back in range.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * salt + offset)
    span = hi - lo + 1
    count = lo + int(wp.randf(state) * float(span))
    out[e] = wp.min(count, hi)


@wp.kernel
def _sample_grid_points_k(
    seeds: wp.array(dtype=wp.int32),
    salt: int,
    offset: int,
    num_cells: int,
    nc2: int,
    cell_size: float,
    scale: float,
    P: int,
    used: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # One thread per env e. Draw ORDER per point c (fixed, so a point's noise is
    # deterministic given its retries): one cell-selection randf per duplicate-rejection
    # retry, then the two noise randfs (nx, ny). A cell is redrawn (up to 8 tries) while
    # it collides with the cell of an EARLIER point of this env; the history lives in
    # used[e*P + 0 .. e*P + c] (pre-filled with -1).
    e = wp.tid()
    state = wp.rand_init(seeds[e] * salt + offset)
    base = e * P
    for c in range(P):
        cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
        # Bounded duplicate rejection against earlier points of this env.
        # dup is an int flag (Warp can't mutate a Python bool in a dynamic loop).
        for _retry in range(8):
            dup = int(0)
            for k in range(c):
                if used[base + k] == cell:
                    dup = int(1)
            if dup == 0:
                break
            cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
        used[base + c] = cell

        x = float(cell % num_cells)
        y = float(cell // num_cells)
        nx = wp.randf(state) - 0.5            # [0,1) -> [-0.5, 0.5)
        ny = wp.randf(state) - 0.5
        out[base + c] = wp.vec2f((x * cell_size + nx) * scale,
                                 (y * cell_size + ny) * scale)


@wp.kernel
def _sample_grid_points_scaled_k(
    seeds: wp.array(dtype=wp.int32),
    salt: int,
    offset: int,
    num_cells: int,
    nc2: int,
    cell_size: float,
    scale: wp.array(dtype=wp.float32),
    P: int,
    used: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec2f),
):
    # Per-env-scale variant of _sample_grid_points_k: identical except `scale` is read
    # per env (scale[e]); for a uniform array equal to the scalar it is bit-identical.
    e = wp.tid()
    state = wp.rand_init(seeds[e] * salt + offset)
    base = e * P
    sc = scale[e]
    for c in range(P):
        cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
        for _retry in range(8):
            dup = int(0)
            for k in range(c):
                if used[base + k] == cell:
                    dup = int(1)
            if dup == 0:
                break
            cell = wp.min(int(wp.randf(state) * float(nc2)), nc2 - 1)
        used[base + c] = cell

        x = float(cell % num_cells)
        y = float(cell // num_cells)
        nx = wp.randf(state) - 0.5
        ny = wp.randf(state) - 0.5
        out[base + c] = wp.vec2f((x * cell_size + nx) * sc,
                                 (y * cell_size + ny) * sc)


# ---------------------------------------------------------------------------------------
# Host helper + private launchers (no validation; the legacy _src wrappers call these).
# ---------------------------------------------------------------------------------------

def grid_cells(min_point_distance: float) -> "tuple[int, float]":
    """Grid for :func:`sample_grid_points` from the built-in ``min_point_distance``
    knob (host only).

    Returns ``(num_cells, cell_size)`` = ``(int(1 / (2*d)), 2*d)``: the built-in
    generators' grid of ``num_cells x num_cells`` cells of side ``2*d``. The name
    follows the config knob; it is not a distance guarantee (see
    :func:`sample_grid_points`).

    Raises:
        ValueError: ``min_point_distance`` is not a finite float in ``(0, 0.5]`` (the
            range ``TrackGenConfig`` accepts; above 0.5 the grid would be empty).
    """
    d = _check_float("min_point_distance", min_point_distance, positive=True)
    if d > 0.5:
        raise ValueError(f"min_point_distance must be <= 0.5, got {d}")
    return int(1.0 / (min_point_distance * 2)), min_point_distance * 2.0


def _sample_counts_launch(ctx, seeds, salt, offset, lo, hi, out) -> None:
    ctx.launch(_sample_counts_k, out.shape[0],
               [seeds, int(salt), int(offset), int(lo), int(hi), out])


def _sample_grid_points_launch(ctx, seeds, salt, offset, P, num_cells, cell_size, scale,
                               used, out, per_env=None) -> None:
    # per_env selects the per-env-scale kernel; by default it follows the type of
    # `scale`. The legacy wrapper passes it explicitly because it historically accepted
    # any array-like [E] scale (e.g. a numpy float32 array), which Warp's launch takes.
    E = seeds.shape[0]
    num_cells = int(num_cells)
    nc2 = num_cells * num_cells
    if per_env is None:
        per_env = isinstance(scale, wp.array)
    ctx.launch(_fill_i32_k, E * P, [used, -1])
    if per_env:
        ctx.launch(_sample_grid_points_scaled_k, E,
                   [seeds, int(salt), int(offset), num_cells, nc2, float(cell_size),
                    scale, P, used, out])
    else:
        ctx.launch(_sample_grid_points_k, E,
                   [seeds, int(salt), int(offset), num_cells, nc2, float(cell_size),
                    float(scale), P, used, out])


# ---------------------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------------------

def sample_counts(ctx: "LaunchContext", seeds: "wp.array", salt: int, offset: int,
                  lo: int, hi: int, out: "wp.array") -> None:
    """One uniformly drawn integer in ``[lo, hi]`` per env.

    ``out[e] = min(lo + int(randf * (hi - lo + 1)), hi)`` with one ``randf`` from
    ``wp.rand_init(seeds[e] * salt + offset)`` (int32 wrap-around). Deterministic
    per ``(seed, salt, offset)``. One launch (``E`` threads); no allocation, no host
    sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        seeds: ``[E]`` ``int32`` per-env seeds; ``E`` is ``seeds.shape[0]``.
        salt: int32 stream multiplier.
        offset: int32 stream offset.
        lo: smallest value (int32).
        hi: largest value (int32, ``>= lo``).
        out: ``[E]`` ``int32`` output.

    Raises:
        ValueError: a wrong type, shape, dtype or device, ``lo > hi`` or a value out
            of int32 range (host-side, before any launch).
    """
    dev = ctx.device
    salt = _check_int("salt", salt)
    offset = _check_int("offset", offset)
    lo = _check_int("lo", lo)
    hi = _check_int("hi", hi)
    if lo > hi:
        raise ValueError(f"lo must be <= hi, got lo={lo}, hi={hi}")
    if hi - lo + 1 > 2 ** 31 - 1:
        raise ValueError(f"hi - lo + 1 must fit in int32, got {hi - lo + 1}")
    E = _check_1d("seeds", seeds, wp.int32, dev)
    check_array("out", out, (E,), wp.int32, dev)
    _sample_counts_launch(ctx, seeds, salt, offset, lo, hi, out)


def sample_grid_points(ctx: "LaunchContext", seeds: "wp.array", salt: int, offset: int,
                       num: int, num_cells: int, cell_size: float, scale,
                       used: "wp.array", out: "wp.array") -> None:
    """``num`` jittered grid points per env, on distinct cells where possible.

    Per env, from ``wp.rand_init(seeds[e] * salt + offset)`` (int32 wrap-around):
    for each point ``c < num`` draw a cell ``min(int(randf * num_cells**2),
    num_cells**2 - 1)``, redrawing (up to 8 times) while it equals the cell of an
    earlier point of the env; then draw ``nx, ny = randf - 0.5`` and write
    ``((cell % num_cells) * cell_size + nx, (cell // num_cells) * cell_size + ny) * scale``.
    ``used`` is filled with -1 first and receives the chosen cells.

    Guarantees: distinct cells are best effort (a point keeps a duplicate cell after 8
    redraws), and the jitter is an absolute ``[-0.5, 0.5)`` (not scaled by
    ``cell_size``), so there is no minimum distance between points. Each coordinate
    lies in ``[-0.5, (num_cells - 1) * cell_size + 0.5) * scale``. Use
    :func:`grid_cells` for the built-in grid. Two launches (fill, ``E`` threads); no
    allocation, no host sync.

    Args:
        ctx: the :class:`~track_gen.exec.LaunchContext` to launch on.
        seeds: ``[E]`` ``int32`` per-env seeds; ``E`` is ``seeds.shape[0]``.
        salt: int32 stream multiplier.
        offset: int32 stream offset.
        num: points per env (``>= 1``).
        num_cells: grid cells per side (``>= 1``).
        cell_size: cell side length.
        scale: ``float`` applied to every env, or ``[E]`` ``float32`` per-env scale
            (bit-identical to the scalar form for a uniform array).
        used: ``[>= E*num]`` ``int32`` scratch (chosen cell per point).
        out: ``[E*num]`` ``vec2f`` output.

    Raises:
        ValueError: a wrong type, shape, dtype or device, or a bad scalar (host-side,
            before any launch).
    """
    dev = ctx.device
    salt = _check_int("salt", salt)
    offset = _check_int("offset", offset)
    num = _check_int("num", num, 1)
    num_cells = _check_int("num_cells", num_cells, 1, 46340)   # num_cells**2 fits int32
    cell_size = _check_float("cell_size", cell_size)
    E = _check_1d("seeds", seeds, wp.int32, dev)
    if isinstance(scale, wp.array):
        check_array("scale", scale, (E,), wp.float32, dev)
    else:
        scale = _check_float("scale", scale)
    check_array("out", out, (E * num,), wp.vec2f, dev)
    _check_min("used", used, E * num, wp.int32, dev)
    _sample_grid_points_launch(ctx, seeds, salt, offset, num, num_cells, cell_size, scale,
                               used, out)
