"""Frozen rc1 linear sampler bodies for independent numeric regression checks only.

Production samplers hash all stream keys. Tests precompute the linear initializer's
input to isolate that keying change from count/grid arithmetic and draw order.
These kernels are not imported by the shipped library.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from track_gen.ops.fill import _fill_i32_k
from track_gen.ops.hash import hash32_host


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
# Array-reading twins (roadmap PR13d): the scalar kernels above with ONE per-env value
# read from an [E] array and clamped by comparisons (a NaN passes), then the scalar body
# verbatim. _sample_counts_dyn_k reads ``lo`` per env (the bezier / hull ``min_num_points``,
# clamped to [2, hi]); the sized grid twins multiply each point LAST by the env's ``size``
# (clamped to SIZE_BOUNDS): ``((x * cell_size + nx) * scale) * size``. The product feeds
# no add / sub, so nvcc's FMA contraction is unchanged and ``size == 1`` is bit-exact
# (``x * 1.0f`` is exact without flush-to-zero).
# ---------------------------------------------------------------------------------------

_C_SIZE_LO = wp.constant(wp.float32(0.1))
_C_SIZE_HI = wp.constant(wp.float32(10.0))
_C_COUNT_LO = wp.constant(2)


@wp.kernel
def _sample_counts_dyn_k(
    seeds: wp.array(dtype=wp.int32),
    salt: int,
    offset: int,
    lo_e: wp.array(dtype=wp.int32),
    hi: int,
    out: wp.array(dtype=wp.int32),
):
    # _sample_counts_k with lo read per env (min_num_points), clamped to [2, hi] by
    # comparisons, then the scalar body verbatim.
    e = wp.tid()
    lo = lo_e[e]
    lo = wp.where(lo < _C_COUNT_LO, _C_COUNT_LO, lo)
    lo = wp.where(lo > hi, hi, lo)
    state = wp.rand_init(seeds[e] * salt + offset)
    span = hi - lo + 1
    count = lo + int(wp.randf(state) * float(span))
    out[e] = wp.min(count, hi)


@wp.kernel
def _sample_grid_points_sized_k(
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
    size_e: wp.array(dtype=wp.float32),
):
    # _sample_grid_points_k with a per-env size applied LAST: ((x*cs + nx) * scale) * size.
    e = wp.tid()
    size = size_e[e]
    size = wp.where(size < _C_SIZE_LO, _C_SIZE_LO, size)
    size = wp.where(size > _C_SIZE_HI, _C_SIZE_HI, size)
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
        out[base + c] = wp.vec2f(((x * cell_size + nx) * scale) * size,
                                 ((y * cell_size + ny) * scale) * size)


@wp.kernel
def _sample_grid_points_scaled_sized_k(
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
    size_e: wp.array(dtype=wp.float32),
):
    # _sample_grid_points_scaled_k with a per-env size applied LAST (bezier's
    # style_sampling scale draw times the dynamic size).
    e = wp.tid()
    size = size_e[e]
    size = wp.where(size < _C_SIZE_LO, _C_SIZE_LO, size)
    size = wp.where(size > _C_SIZE_HI, _C_SIZE_HI, size)
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
        out[base + c] = wp.vec2f(((x * cell_size + nx) * sc) * size,
                                 ((y * cell_size + ny) * sc) * size)


def _sample_counts_launch(ctx, seeds, salt, offset, lo, hi, out):
    kernel = _sample_counts_dyn_k if isinstance(lo, wp.array) else _sample_counts_k
    ctx.launch(kernel, out.shape[0], [seeds, int(salt), int(offset), lo, int(hi), out])


def _sample_grid_points_launch(ctx, seeds, salt, offset, P, num_cells, cell_size,
                               scale, used, out, *, size=None):
    E = seeds.shape[0]
    per_env = isinstance(scale, wp.array)
    ctx.launch(_fill_i32_k, E * P, [used, -1])
    if size is None:
        kernel = _sample_grid_points_scaled_k if per_env else _sample_grid_points_k
    else:
        kernel = (_sample_grid_points_scaled_sized_k if per_env
                  else _sample_grid_points_sized_k)
    args = [seeds, int(salt), int(offset), int(num_cells), int(num_cells) ** 2,
            float(cell_size), scale, P, used, out]
    if size is not None:
        args.append(size)
    ctx.launch(kernel, E, args)


def linearized_seeds(seeds, salt, index, *, linear_salt=1, linear_offset=0, device="cpu"):
    """Seed the frozen linear body with the independently expanded hashed input.

    Warp rand_init(key, i) = rand_pcg(key + rand_pcg(i)). An odd linear
    multiplier is invertible modulo 2**32, including the PR7 literal salts.
    """
    mask = 0xFFFFFFFF
    b = ((int(index) & mask) * 747796405 + 2891336453) & mask
    c = (((b >> ((b >> 28) + 4)) ^ b) * 277803737) & mask
    offset_state = (c >> 22) ^ c
    inv = pow(int(linear_salt), -1, 2 ** 32)
    words = [(((hash32_host((int(seed) & mask) ^ int(salt)) & 0x7FFFFFFF)
               + offset_state - int(linear_offset)) * inv) & mask for seed in seeds]
    return wp.array(np.asarray(words, np.uint32).view(np.int32),
                    dtype=wp.int32, device=device)
