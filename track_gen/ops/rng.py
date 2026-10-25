"""Hashed stream initialization for user and built-in Warp kernels.

The formulas are fixed: hash the seed XOR salt, clear the sign bit, then call
Warp's one- or two-argument ``rand_init``. In particular, ``init_at(..., 0)``
uses Warp's two-argument form and differs from ``init(...)``. The host twins
reproduce the same uint32 states without launching a kernel.
"""
from __future__ import annotations

import warp as wp

from .hash import hash32, hash32_host

__all__ = ["init", "init_at", "key_host", "init_host", "init_at_host"]

_MASK32 = 0xFFFFFFFF


@wp.func
def init(seed: wp.int32, salt: wp.uint32) -> wp.uint32:
    """Initialize a stream from an int32 seed and a uint32 namespace salt."""
    key = wp.int32(hash32(wp.uint32(seed) ^ salt) & wp.uint32(0x7FFFFFFF))
    return wp.rand_init(key)


@wp.func
def init_at(seed: wp.int32, salt: wp.uint32, index: wp.int32) -> wp.uint32:
    """Initialize an indexed stream using Warp's two-argument ``rand_init``."""
    key = wp.int32(hash32(wp.uint32(seed) ^ salt) & wp.uint32(0x7FFFFFFF))
    return wp.rand_init(key, index)


def key_host(seed: int, salt: int) -> int:
    """Host stream key: ``hash32(uint32(seed) ^ uint32(salt)) & 0x7fffffff``."""
    return hash32_host(int(seed) ^ int(salt)) & 0x7FFFFFFF


def _rand_pcg_host(state: int) -> int:
    # Warp native/rand.h's rand_pcg, with uint32 wrap-around at each product.
    b = ((int(state) & _MASK32) * 747796405 + 2891336453) & _MASK32
    c = (((b >> ((b >> 28) + 4)) ^ b) * 277803737) & _MASK32
    return (c >> 22) ^ c


def init_host(seed: int, salt: int) -> int:
    """Bit-exact host twin of :func:`init`, returning a uint32 state as an int."""
    return _rand_pcg_host(key_host(seed, salt))


def init_at_host(seed: int, salt: int, index: int) -> int:
    """Bit-exact host twin of :func:`init_at`, including signed int32 indices."""
    return _rand_pcg_host(key_host(seed, salt) + _rand_pcg_host(index))
