"""Integer hashing for device code.

:func:`hash32` is the 32-bit finaliser of MurmurHash3 (``fmix32``): a bijection on
``uint32`` with full avalanche, so nearby inputs (consecutive seeds, epochs, salts)
map to unrelated outputs. It is a ``@wp.func`` for use inside kernels;
:func:`hash32_host` is the bit-exact host twin (tests, host-side reproduction).
"""
from __future__ import annotations

import warp as wp

__all__ = ["hash32", "hash32_host"]

_M1 = 0x85EBCA6B
_M2 = 0xC2B2AE35
_MASK32 = 0xFFFFFFFF


@wp.func
def hash32(x: wp.uint32) -> wp.uint32:
    """MurmurHash3 ``fmix32`` finaliser (a ``uint32`` bijection; wrap-around
    arithmetic)."""
    h = x ^ (x >> wp.uint32(16))
    h = h * wp.uint32(0x85EBCA6B)
    h = h ^ (h >> wp.uint32(13))
    h = h * wp.uint32(0xC2B2AE35)
    h = h ^ (h >> wp.uint32(16))
    return h


def hash32_host(x: int) -> int:
    """Host twin of :func:`hash32` (``x`` is reduced to ``uint32`` first)."""
    h = int(x) & _MASK32
    h ^= h >> 16
    h = (h * _M1) & _MASK32
    h ^= h >> 13
    h = (h * _M2) & _MASK32
    h ^= h >> 16
    return h
