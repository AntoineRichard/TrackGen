"""Helpers of the result-object tests (roadmap PR13b): byte poisoning, row slicing,
minimal protocol objects (no ``layout``), a ``__getattr__`` proxy and analytic rows."""
from __future__ import annotations

import types

import numpy as np
import warp as wp

from track_gen.testing.capture import result_arrays

POISON = 0xA5


def poisoned(result):
    """``result`` with every byte of every array (fields and attributes) set to
    :data:`POISON` (host writes; returns ``result``)."""
    for arr in result_arrays(result).values():
        host = arr.numpy()
        host.view(np.uint8)[...] = POISON
        arr.assign(host)
    return result


def rows(arr, e0: int, e1: int, S: int, per_point: bool = True) -> bytes:
    """The bytes of envs ``[e0, e1)`` of a flat result array."""
    h = np.ascontiguousarray(arr.numpy())
    n = S if per_point else 1
    return h[e0 * n:e1 * n].tobytes()


def outside_is_poison(arr, e0: int, e1: int, S: int, per_point: bool = True) -> bool:
    """Whether every byte of ``arr`` outside envs ``[e0, e1)`` still holds the poison."""
    h = np.ascontiguousarray(arr.numpy())
    n = S if per_point else 1
    out = np.concatenate([h[:e0 * n].reshape(-1), h[e1 * n:].reshape(-1)])
    return bool((out.view(np.uint8) == POISON).all()) if out.size else True


def bytes_of(obj) -> dict:
    """``name -> bytes`` of every array of a result (fields and attributes)."""
    return {k: np.ascontiguousarray(a.numpy()).tobytes()
            for k, a in result_arrays(obj).items()}


def MinimalCenterline(src):
    """Only the ``HasCenterline`` members of ``src`` (no ``layout``)."""
    return types.SimpleNamespace(center=src.center, arclen=src.arclen, length=src.length,
                                 count=src.count)


def MinimalBand(src):
    """Only the ``HasBand`` members of ``src`` (no ``layout``)."""
    return types.SimpleNamespace(center=src.center, outer=src.outer, inner=src.inner,
                                 count=src.count, valid=src.valid)


def MinimalGates(src, frames: bool = False):
    """Only the ``HasGates`` (``HasGateFrames`` with ``frames``) members of ``src``."""
    kw = dict(position=src.position, count=src.count, valid=src.valid)
    if frames:
        kw.update(forward=src.forward, left=src.left, right=src.right,
                  orientation=src.orientation, half_size=src.half_size)
    return types.SimpleNamespace(**kw)


class LazyProxy:
    """A ``__getattr__``-based proxy of ``src`` (no attributes of its own)."""

    def __init__(self, src) -> None:
        object.__setattr__(self, "_src", src)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_src"), name)


def circle_rows(E: int, m: int, R: float, phase: float = 0.1) -> "list[np.ndarray]":
    """``E`` circles of radius ``R`` sampled at ``m`` points (env ``e`` rotated by
    ``phase * e``)."""
    t = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    return [np.stack([R * np.cos(t + phase * e), R * np.sin(t + phase * e)], -1)
            .astype(np.float32) for e in range(E)]


def ellipse_rows(E: int, m: int, a: float, b: float,
                 phase: float = 0.0) -> "list[np.ndarray]":
    """``E`` ellipses of semi-axes ``a``, ``b`` at ``m`` points."""
    t = phase + np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    row = np.stack([a * np.cos(t), b * np.sin(t)], -1).astype(np.float32)
    return [row.copy() for _ in range(E)]


def ramanujan(a: float, b: float) -> float:
    """Ramanujan's second approximation of an ellipse's perimeter."""
    h = ((a - b) / (a + b)) ** 2
    return float(np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h))))


def mask(E: int, envs) -> wp.array:
    m = np.zeros(E, np.int32)
    m[list(envs)] = 1
    return wp.array(m, dtype=wp.int32, device="cpu")
