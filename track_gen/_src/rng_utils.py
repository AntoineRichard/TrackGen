import numpy as np

import warp as wp

from ..exec.context import LaunchContext
from .rng_kernels import integer, normal, poisson, quaternion, rand_sign_fn, set_states, uniform


class PerEnvSeededRNG:
    """Per-environment seeded random number generator backed by NVIDIA Warp kernels.

    Each of the ``num_envs`` environments owns an independent RNG state so that
    parallel sampling yields reproducible, environment-diverse results.  A scalar
    ``seeds`` int is expanded to ``seed + arange(num_envs)`` (diversity expansion):
    because Warp seeds its state purely from the seed value, a flat broadcast would
    make every environment identical; the offset guarantees each env starts from a
    distinct stream.  Pass a ``wp.array`` of ``int32`` to control every env's seed
    explicitly.

    After construction the states are immediately initialised from the seeds, so
    no separate ``set_seeds_warp`` call is required before the first sample.
    """

    def __init__(self, seeds: int | wp.array, num_envs: int, device: str):
        """Initialize the random number generator.


        Args:
            seeds: Per-env seeds. An int scalar seeds the whole BATCH: it expands to
                distinct per-env seeds ``seed + arange(num_envs)`` so the batch is
                reproducible AND diverse. (Each env's RNG state is ``wp.rand_init(seed)``
                with no env-index folding, so identical per-env seeds would yield
                identical tracks — hence the expansion.) Pass a wp.array of int32 to
                control every env's seed explicitly.
            num_envs: The number of environments.
            device: The device to use.
        """

        self._device = device
        self._num_envs = num_envs

        # Instantiate buffers
        if isinstance(seeds, int):
            # Distinct per-env seeds -> diverse batch (a flat broadcast made every env
            # identical, since the state is seeded purely from the seed value).
            self._seeds = wp.array(seeds + np.arange(num_envs), dtype=wp.int32, device=device)
        else:
            self._seeds = seeds

        self._states = wp.zeros(self._seeds.shape, dtype=wp.uint32, device=device)
        self._new_states = wp.zeros(self._seeds.shape, dtype=wp.uint32, device=device)
        self._ALL_INDICES = wp.array(np.arange(num_envs), dtype=wp.int32, device=device)

        # Auto-initialize states from seeds so that construction alone is sufficient
        # to produce reproducible, seed-determined outputs without a separate set_seeds call.
        # Construction runs on Warp's CURRENT stream, the same stream the wp.zeros /
        # wp.array allocations above were zeroed/uploaded on (no ambient or torch adoption).
        # set_seeds_warp also mirrors the states into the _new_states double-buffer, so
        # a first PARTIAL-ids sample (which only writes new_states for the selected envs,
        # then copies the WHOLE array back) cannot zero the untouched envs' states.
        ctx = LaunchContext.current(device)
        self.set_seeds_warp(self._seeds, None, ctx=ctx)

    @property
    def seeds_warp(self) -> wp.array:
        """The ``[num_envs]`` ``int32`` Warp array of per-environment base seeds."""
        return self._seeds

    @property
    def states_warp(self) -> wp.array:
        """The ``[num_envs]`` ``uint32`` Warp array of live per-environment RNG states."""
        return self._states

    @staticmethod
    def to_tuple(shape: int | tuple[int]) -> tuple:
        """Casts to a tuple."""
        if isinstance(shape, int):
            return (shape,)
        else:
            return shape

    def _ctx(self, stream, ctx) -> LaunchContext:
        """The launch context of one public call: ``ctx`` when given, else the
        context resolved for this RNG's device and ``stream`` (see track_gen.exec)."""
        if ctx is not None:
            if stream is not None:
                raise TypeError("pass either stream= or ctx=, not both")
            return ctx
        return LaunchContext.for_call(self._device, stream)

    def set_seeds_warp(self, seeds: wp.array, ids: wp.array | None, stream=None, ctx=None) -> None:
        """Set the seeds for each environment (and re-initialise their RNG states).

        Two operations on the call's stream: the reseed launch, then a copy that
        mirrors the states into the sampling double-buffer (so a later partial-ids
        sample cannot revert the reseed). Capturable and allocation-free.

        The new seeds are written into :attr:`seeds_warp` in place. That array is
        the caller's own when the rng was constructed from a ``wp.array`` (it is
        bound without a copy), and it is what a generator built on this rng binds
        and reads at every generation.

        Args:
            seeds: The seeds for each environment.
            ids: The ids of the environments.
            stream: optional wp.Stream (or 'torch') to run on; by default the stream is resolved per
                track_gen.exec (an enclosing torch.cuda.graph capture is adopted, else Warp's current stream).
            ctx: optional track_gen.exec.LaunchContext to run on instead of ``stream`` (for internal
                callers that already hold one); default resolves the device's stream (see track_gen.exec).
        """
        ctx = self._ctx(stream, ctx)

        if ids is None:
            ids = self._ALL_INDICES

        num_instances = len(seeds)
        ctx.launch(
            set_states,
            num_instances,
            [seeds, self._seeds, self._states, ids],
        )
        # Mirror the reseeded states into the sampling double-buffer: a sampler writes
        # new_states only for the envs it samples, then copies the WHOLE buffer back
        # into states, so a stale new_states would revert this reseed for every env a
        # later partial-ids sample does not touch.
        ctx.copy(self._new_states, self._states)

    def sample_uniform_warp(
        self,
        low: float | wp.array,
        high: float | wp.array,
        shape: tuple | int,
        ids: wp.array | None = None,
        stream=None,
        ctx=None,
        out: "wp.array | None" = None,
    ) -> wp.array:
        """Sample from a uniform distribution. Warp implementation.

        If low and high are arrays, their shapes need to match that of the ids.

        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
            stream: optional wp.Stream (or 'torch') to run on; by default the stream is resolved per
                track_gen.exec (an enclosing torch.cuda.graph capture is adopted, else Warp's current stream).
            ctx: optional track_gen.exec.LaunchContext to run on instead of ``stream`` (for internal
                callers that already hold one); default resolves the device's stream (see track_gen.exec).
            out: optional preallocated output ``wp.array`` of the exact output shape, dtype and
                device; written in place and returned, so the call allocates nothing (capturable
                sampling). Validated before any launch (``ValueError`` on a mismatch).
        Returns:
            The sampled values."""
        ctx = self._ctx(stream, ctx)
        if ids is None:
            ids = self._ALL_INDICES
        return uniform(low, high, self._states, self._new_states, ids, self.to_tuple(shape), self._device,
                       ctx=ctx, out=out)

    def sample_sign_warp(
        self, dtype: str, shape: tuple | int, ids: wp.array | None = None, stream=None, ctx=None,
        out: "wp.array | None" = None,
    ) -> wp.array:
        """Sample a sign. Warp implementation.

        Args:
            dtype: The data type of the output tensor.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
            stream: optional wp.Stream (or 'torch') to run on; by default the stream is resolved per
                track_gen.exec (an enclosing torch.cuda.graph capture is adopted, else Warp's current stream).
            ctx: optional track_gen.exec.LaunchContext to run on instead of ``stream`` (for internal
                callers that already hold one); default resolves the device's stream (see track_gen.exec).
            out: optional preallocated output ``wp.array`` of the exact output shape, dtype and
                device; written in place and returned, so the call allocates nothing (capturable
                sampling). Validated before any launch (``ValueError`` on a mismatch).
        Returns:
            The sampled values."""
        ctx = self._ctx(stream, ctx)
        if ids is None:
            ids = self._ALL_INDICES
        return rand_sign_fn(
            self._states, self._new_states, ids, self.to_tuple(shape), dtype, self._device, ctx=ctx, out=out
        )

    def sample_integer_warp(
        self,
        low: int | wp.array,
        high: int | wp.array,
        shape: tuple | int,
        ids: wp.array | None = None,
        stream=None,
        ctx=None,
        out: "wp.array | None" = None,
    ) -> wp.array:
        """Sample for a random integer. Warp implementation.

        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
            stream: optional wp.Stream (or 'torch') to run on; by default the stream is resolved per
                track_gen.exec (an enclosing torch.cuda.graph capture is adopted, else Warp's current stream).
            ctx: optional track_gen.exec.LaunchContext to run on instead of ``stream`` (for internal
                callers that already hold one); default resolves the device's stream (see track_gen.exec).
            out: optional preallocated output ``wp.array`` of the exact output shape, dtype and
                device; written in place and returned, so the call allocates nothing (capturable
                sampling). Validated before any launch (``ValueError`` on a mismatch).
        Returns:
            wp.array: The sampled values."""
        ctx = self._ctx(stream, ctx)
        if ids is None:
            ids = self._ALL_INDICES
        return integer(low, high, self._states, self._new_states, ids, self.to_tuple(shape), self._device,
                       ctx=ctx, out=out)

    def sample_normal_warp(
        self,
        mean: float | wp.array,
        std: float | wp.array,
        shape: tuple | int,
        ids: wp.array | None = None,
        stream=None,
        ctx=None,
        out: "wp.array | None" = None,
    ) -> wp.array:
        """Sample from a normal distribution. Warp implementation.

        Args:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
            stream: optional wp.Stream (or 'torch') to run on; by default the stream is resolved per
                track_gen.exec (an enclosing torch.cuda.graph capture is adopted, else Warp's current stream).
            ctx: optional track_gen.exec.LaunchContext to run on instead of ``stream`` (for internal
                callers that already hold one); default resolves the device's stream (see track_gen.exec).
            out: optional preallocated output ``wp.array`` of the exact output shape, dtype and
                device; written in place and returned, so the call allocates nothing (capturable
                sampling). Validated before any launch (``ValueError`` on a mismatch).
        Returns:
            wp.array: The sampled values."""
        ctx = self._ctx(stream, ctx)
        if ids is None:
            ids = self._ALL_INDICES
        return normal(mean, std, self._states, self._new_states, ids, self.to_tuple(shape), self._device,
                       ctx=ctx, out=out)

    def sample_poisson_warp(
        self, lam: float | wp.array, shape: tuple | int, ids: wp.array | None = None, stream=None, ctx=None,
        out: "wp.array | None" = None,
    ) -> wp.array:
        """Sample from a poisson distribution. Warp implementation.

        Args:
            lam: The rate of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
            stream: optional wp.Stream (or 'torch') to run on; by default the stream is resolved per
                track_gen.exec (an enclosing torch.cuda.graph capture is adopted, else Warp's current stream).
            ctx: optional track_gen.exec.LaunchContext to run on instead of ``stream`` (for internal
                callers that already hold one); default resolves the device's stream (see track_gen.exec).
            out: optional preallocated output ``wp.array`` of the exact output shape, dtype and
                device; written in place and returned, so the call allocates nothing (capturable
                sampling). Validated before any launch (``ValueError`` on a mismatch).
        Returns:
            The sampled values."""
        ctx = self._ctx(stream, ctx)
        if ids is None:
            ids = self._ALL_INDICES
        return poisson(lam, self._states, self._new_states, ids, self.to_tuple(shape), self._device, ctx=ctx, out=out)

    def sample_quaternion_warp(
        self, shape: tuple | int, ids: wp.array | None = None, stream=None, ctx=None,
        out: "wp.array | None" = None,
    ) -> wp.array:
        """Sample a quaternion. Warp implementation.

        Args:
            shape: The shape of the output tensor.
            ids: The ids of the environments.
            stream: optional wp.Stream (or 'torch') to run on; by default the stream is resolved per
                track_gen.exec (an enclosing torch.cuda.graph capture is adopted, else Warp's current stream).
            ctx: optional track_gen.exec.LaunchContext to run on instead of ``stream`` (for internal
                callers that already hold one); default resolves the device's stream (see track_gen.exec).
            out: optional preallocated output ``wp.array`` of the exact output shape, dtype and
                device; written in place and returned, so the call allocates nothing (capturable
                sampling). Validated before any launch (``ValueError`` on a mismatch).
        Returns:
            wp.array: The sampled values."""
        ctx = self._ctx(stream, ctx)
        if ids is None:
            ids = self._ALL_INDICES
        return quaternion(self._states, self._new_states, ids, self.to_tuple(shape), self._device, ctx=ctx, out=out)
