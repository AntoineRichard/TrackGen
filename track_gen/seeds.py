"""Per-env seeds and epochs: the device-side state that keys every generation.

A :class:`SeedState` holds two ``[E]`` ``int32`` arrays:

- ``seeds``: the per-env base seeds. Either an array the state OWNS (built from
  ``None`` / an int, or copied with ``copy=True``), or a caller array BOUND by
  pointer (a ``wp.array``, or a ``PerEnvSeededRNG``'s ``seeds_warp``).
  ``SeedState`` never writes bound seeds, and neither do the generators: write
  new seeds into that array yourself (they are read at every generation, also by
  every graph replay), or use an owned state and :meth:`SeedState.set`. The rng
  itself does write its ``seeds_warp``: ``PerEnvSeededRNG.set_seeds_warp`` and
  ``Course.generate(seeds=...)`` overwrite it in place, and it IS the caller's
  array when the rng (or ``CourseConfig.seeds``) was given a ``wp.array``.
- ``epoch``: owned, zero-initialised, and only ever changed on device
  (:meth:`SeedState.advance`, :meth:`SeedState.reset_epoch`,
  :meth:`SeedState.set`), so it can be driven from inside a CUDA graph.

Every generation first folds the two into an EFFECTIVE seed per env
(``_effective_seed_k``, the first launch of every generator body)::

    effective = seed                                       if epoch == 0
              = hash32(u32(seed) ^ hash32(u32(epoch) * 0x9E3779B9)) & 0x7FFFFFFF
                                                           otherwise

Epoch 0 leaves the seed unchanged, so a never-advanced env generates exactly what
it did before epochs existed (goldens are byte-identical). A MASKED generation
(``effective(out, mask=m)``) re-derives the effective seed of the envs with
``m[e] != 0`` only: every other env keeps the effective seed its current output
was built from, so it is recomputed to identical values whatever happened to its
``(seed, epoch)`` since. Determinism: the pair ``(seed, epoch)`` fully reproduces
an env. A generator whose seed for that env is
``effective_seed(seed, epoch)`` at epoch 0 produces the identical track or gate
sequence (:func:`effective_seed` is the bit-exact host twin of the kernel).
"""
from __future__ import annotations

import numbers

import numpy as np
import warp as wp

from ._src.runtime import _check_arr
from .exec.context import LaunchContext
from .ops.hash import hash32, hash32_host

__all__ = ["SeedState", "effective_seed"]

_GOLDEN = 0x9E3779B9
_MASK32 = 0xFFFFFFFF


# All SeedState kernels live in this one module, so loading it once (the facades'
# warm-up pass launches _effective_seed_k) loads every one of them before any capture.

@wp.func
def _effective_seed(seed: wp.int32, ep: wp.int32) -> wp.int32:
    s = seed
    if ep != 0:
        # epoch 0 -> seed unchanged (byte-identical to the pre-epoch keying)
        h = hash32(wp.uint32(seed) ^ hash32(wp.uint32(ep) * wp.uint32(0x9E3779B9)))
        s = wp.int32(h & wp.uint32(0x7FFFFFFF))
    return s


@wp.kernel
def _effective_seed_k(
    seeds: wp.array(dtype=wp.int32),
    epoch: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    out[e] = _effective_seed(seeds[e], epoch[e])


@wp.kernel
def _effective_seed_masked_k(
    seeds: wp.array(dtype=wp.int32),
    epoch: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    # Unmasked envs keep the effective seed their current output was built from.
    e = wp.tid()
    if mask[e] != 0:
        out[e] = _effective_seed(seeds[e], epoch[e])


@wp.kernel
def _copy_masked_k(
    src: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.int32),
):
    e = wp.tid()
    if mask[e] != 0:
        dst[e] = src[e]


@wp.kernel
def _advance_all_k(epoch: wp.array(dtype=wp.int32)):
    e = wp.tid()
    epoch[e] = wp.int32(wp.uint32(epoch[e]) + wp.uint32(1))   # wraps, never UB


@wp.kernel
def _advance_masked_k(epoch: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] != 0:
        epoch[e] = wp.int32(wp.uint32(epoch[e]) + wp.uint32(1))


@wp.kernel
def _zero_all_k(arr: wp.array(dtype=wp.int32)):
    arr[wp.tid()] = 0


@wp.kernel
def _zero_masked_k(arr: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] != 0:
        arr[e] = 0


@wp.kernel
def _scatter_seeds_k(
    src: wp.array(dtype=wp.int32),
    ids: wp.array(dtype=wp.int32),
    seeds: wp.array(dtype=wp.int32),
    epoch: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    e = ids[i]
    seeds[e] = src[i]
    epoch[e] = 0


def _wrap_int32(value: int) -> int:
    """``value`` reduced to the int32 range with two's-complement wrap-around."""
    return ((int(value) + 2**31) % 2**32) - 2**31


def effective_seed(seed: int, epoch: int) -> int:
    """Host twin of the effective-seed kernel: the seed env generation actually
    uses for base seed ``seed`` at epoch ``epoch`` (both taken as int32).

    ``effective_seed(s, 0) == s``; for ``epoch != 0`` the result is a
    non-negative int32 hash of ``(s, epoch)``. A generator whose rng seed for an
    env is ``effective_seed(s, k)`` (at epoch 0) reproduces that env's batch at
    ``(s, k)`` exactly.
    """
    s = _wrap_int32(seed)
    ep = _wrap_int32(epoch)
    if ep == 0:
        return s
    h = hash32_host((s & _MASK32) ^ hash32_host(((ep & _MASK32) * _GOLDEN) & _MASK32))
    return int(h & 0x7FFFFFFF)


def _is_rng(obj) -> bool:
    return not isinstance(obj, wp.array) and hasattr(obj, "seeds_warp")


class SeedState:
    """Per-env base seeds plus a device-side epoch counter (see the module docstring).

    Args:
        num_envs: number of environments ``E`` (>= 1).
        device: Warp device of the state.
        seeds: ``None`` (owned ``0 + arange(E)``), an int ``k`` (owned
            ``k + arange(E)`` with int32 wrap-around, the same expansion as
            ``PerEnvSeededRNG(seeds=k)``), an ``[E]`` int32 ``wp.array`` on
            ``device`` (BOUND by pointer, never written by ``SeedState``), or a
            ``PerEnvSeededRNG`` (binds its ``seeds_warp`` by pointer, so
            ``rng.set_seeds_warp(...)`` keeps reseeding generation). The rng
            must therefore live on ``device``.
        copy: with an array or rng ``seeds``, take an OWNED copy instead of
            binding (then :meth:`set` is allowed).
        default_stream: optional ``wp.Stream`` the methods use when a call passes
            neither ``stream=`` nor ``ctx=`` (and no ambient context applies), as
            :func:`track_gen.exec.resolve_stream`'s ``default``. The facades pass
            their constructor ``stream=``, so ``gen.seeds.advance()`` is ordered
            with ``gen.generate()``.

    Construction allocates (and, for an owned int expansion, uploads) on Warp's
    current stream: build it outside any CUDA graph capture. Every other method
    is one or two launches on the call's stream: allocation-free, sync-free and
    capturable.

    Raises:
        TypeError: ``seeds`` is not one of the accepted kinds.
        ValueError: ``num_envs < 1``, or a seed array that is not ``[E]`` int32
            on ``device``.
    """

    def __init__(self, num_envs: int, device, seeds=None, *, copy: bool = False,
                 default_stream: "wp.Stream | None" = None) -> None:
        E = int(num_envs)
        if E < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs!r}")
        if default_stream is not None and not isinstance(default_stream, wp.Stream):
            raise TypeError("default_stream must be a wp.Stream or None; got "
                            f"{type(default_stream).__name__}")
        dev = wp.get_device(device)
        self._E = E
        self._device = str(dev)
        self._default_stream = default_stream if dev.is_cuda else None
        src = seeds.seeds_warp if _is_rng(seeds) else seeds
        if src is None or (isinstance(src, numbers.Integral) and not isinstance(src, bool)):
            base = 0 if src is None else int(src)
            vals = ((base + np.arange(E, dtype=np.int64) + 2**31) % 2**32 - 2**31)
            self._seeds = wp.array(vals.astype(np.int32), dtype=wp.int32, device=dev)
            self._bound = False
        elif isinstance(src, wp.array):
            try:
                _check_arr("seeds", src, (E,), wp.int32, self._device)
            except ValueError as exc:
                if not _is_rng(seeds):
                    raise
                raise ValueError(
                    f"rng.seeds_warp cannot be bound: {exc}. Construct the "
                    f"PerEnvSeededRNG with num_envs={E} on device {self._device!r} (the "
                    "generator reads its seeds by pointer on every generation).") from None
            if copy:
                self._seeds = wp.empty(E, dtype=wp.int32, device=dev)
                LaunchContext.current(dev).copy(self._seeds, src)
                self._bound = False
            else:
                self._seeds = src
                self._bound = True
        else:
            raise TypeError(
                "seeds must be None, an int, an [E] int32 wp.array or a PerEnvSeededRNG; "
                f"got {type(seeds).__name__}")
        self._epoch = wp.zeros(E, dtype=wp.int32, device=dev)

    # -- state --------------------------------------------------------------------
    @property
    def seeds(self) -> wp.array:
        """``[E]`` int32 base seeds (a caller array when :attr:`bound`)."""
        return self._seeds

    @property
    def epoch(self) -> wp.array:
        """``[E]`` int32 per-env epoch (owned; changed only on device)."""
        return self._epoch

    @property
    def bound(self) -> bool:
        """Whether :attr:`seeds` is a caller array bound by pointer (never written
        by ``SeedState``; :meth:`set` raises)."""
        return self._bound

    @property
    def num_envs(self) -> int:
        return self._E

    @property
    def device(self) -> str:
        return self._device

    def __repr__(self) -> str:
        kind = "bound" if self._bound else "owned"
        return f"SeedState(num_envs={self._E}, device={self._device!r}, seeds={kind})"

    # -- helpers ------------------------------------------------------------------
    def _ctx(self, stream, ctx) -> LaunchContext:
        if ctx is not None:
            if stream is not None:
                raise TypeError("pass either stream= or ctx=, not both")
            return ctx
        return LaunchContext.for_call(self._device, stream, default=self._default_stream)

    def check_mask(self, mask, name: str = "mask") -> None:
        """Validate a contiguous ``[E]`` int32 mask on this state's device
        (host-side). Contiguous, because the producers cache a self-graph per mask
        POINTER: two views with one pointer and different strides would share it."""
        _check_arr(name, mask, (self._E,), wp.int32, self._device)
        if not mask.is_contiguous:
            raise ValueError(f"{name} must be a contiguous array (got strides "
                             f"{tuple(mask.strides)}); pass a contiguous copy")

    # -- device ops ---------------------------------------------------------------
    def advance(self, mask: "wp.array | None" = None, *, stream=None, ctx=None) -> None:
        """``epoch[e] += 1`` for every env (``mask=None``) or where ``mask[e] != 0``.

        One launch on the call's stream; capturable (inside a caller's graph every
        replay advances again).

        Args:
            mask: optional ``[E]`` int32 mask on this device, read on device.
            stream: optional ``wp.Stream`` (or ``"torch"``); resolved per
                :mod:`track_gen.exec`.
            ctx: a :class:`~track_gen.exec.LaunchContext` to use instead of
                ``stream``.
        """
        if mask is not None:
            self.check_mask(mask)
        ctx = self._ctx(stream, ctx)
        if mask is None:
            ctx.launch(_advance_all_k, self._E, [self._epoch])
        else:
            ctx.launch(_advance_masked_k, self._E, [self._epoch, mask])

    def reset_epoch(self, mask: "wp.array | None" = None, *, stream=None, ctx=None) -> None:
        """``epoch[e] = 0`` for every env (``mask=None``) or where ``mask[e] != 0``
        (one launch; capturable)."""
        if mask is not None:
            self.check_mask(mask)
        ctx = self._ctx(stream, ctx)
        if mask is None:
            ctx.launch(_zero_all_k, self._E, [self._epoch])
        else:
            ctx.launch(_zero_masked_k, self._E, [self._epoch, mask])

    def set(self, seeds: wp.array, ids: "wp.array | None" = None, *, stream=None,
            ctx=None) -> None:
        """Write new base seeds into an OWNED state and zero those envs' epochs.

        ``ids=None``: ``seeds`` is ``[E]`` int32 and replaces every env's seed.
        Otherwise ``ids`` is ``[n]`` int32 env indices and ``seeds`` the ``[n]``
        int32 new seeds (``seeds[i]`` goes to env ``ids[i]``; duplicate ids race).
        One launch (or a copy plus a launch); capturable.

        Raises:
            RuntimeError: the seeds are :attr:`bound` to a caller array
                (``SeedState`` never writes bound seeds): write that array
                yourself, or build the state with ``copy=True``.
            ValueError: wrong shape, dtype or device.
        """
        if self._bound:
            raise RuntimeError(
                "SeedState.set(): these seeds are bound to a caller array, which "
                "SeedState never writes. Write the new seeds into that array (or reseed "
                "the bound rng with set_seeds_warp), or construct SeedState(..., "
                "copy=True) to own the seeds.")
        if ids is None:
            _check_arr("seeds", seeds, (self._E,), wp.int32, self._device)
            ctx = self._ctx(stream, ctx)
            ctx.copy(self._seeds, seeds)
            ctx.launch(_zero_all_k, self._E, [self._epoch])
            return
        if not isinstance(ids, wp.array) or ids.ndim != 1:
            raise ValueError("ids must be a 1-D int32 wp.array")
        n = int(ids.shape[0])
        _check_arr("ids", ids, (n,), wp.int32, self._device)
        _check_arr("seeds", seeds, (n,), wp.int32, self._device)
        ctx = self._ctx(stream, ctx)
        if n > 0:
            ctx.launch(_scatter_seeds_k, n, [seeds, ids, self._seeds, self._epoch])

    def effective(self, out: wp.array, mask: "wp.array | None" = None, *, stream=None,
                  ctx=None) -> None:
        """Write the effective per-env seeds (see the module docstring) into the
        ``[E]`` int32 ``out``: for every env, or (``mask``) only where
        ``mask[e] != 0``, leaving the other entries of ``out`` untouched. One
        launch; capturable. This is the first launch of every generator body (after
        the epoch advance, if any)."""
        _check_arr("out", out, (self._E,), wp.int32, self._device)
        if mask is not None:
            self.check_mask(mask)
        ctx = self._ctx(stream, ctx)
        if mask is None:
            ctx.launch(_effective_seed_k, self._E, [self._seeds, self._epoch, out])
        else:
            ctx.launch(_effective_seed_masked_k, self._E,
                       [self._seeds, self._epoch, mask, out])

    def _write_generation(self, out: wp.array, mask: "wp.array | None",
                          ctx: LaunchContext) -> None:
        """``out[e] = epoch[e]`` for every env, or only where ``mask[e] != 0`` (the
        generators' ``generation`` field; masks are validated by the caller)."""
        if mask is None:
            ctx.copy(out, self._epoch)
        else:
            ctx.launch(_copy_masked_k, self._E, [self._epoch, mask, out])
