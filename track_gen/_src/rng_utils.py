import numpy as np

import warp as wp

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
            device: The device to use."""

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
        self.set_seeds_warp(self._seeds, None)
        # Mirror states into the double-buffer so a first PARTIAL-ids sample (which only
        # writes new_states for selected envs, then wp.copy's the WHOLE array back) cannot
        # zero the untouched envs' seed-derived states.
        wp.copy(self._new_states, self._states)

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

    def set_seeds_warp(self, seeds: wp.array, ids: wp.array | None) -> None:
        """Set the seeds for each environment.
        Args:
            seeds: The seeds for each environment.
            ids: The ids of the environments."""

        if ids is None:
            ids = self._ALL_INDICES

        num_instances = len(seeds)
        wp.launch(
            kernel=set_states,
            dim=num_instances,
            inputs=[seeds, self._seeds, self._states, ids],
            device=self._device,
        )

    def sample_uniform_warp(
        self, low: float | wp.array, high: float | wp.array, shape: tuple | int, ids: wp.array | None = None
    ) -> wp.array:
        """Sample from a uniform distribution. Warp implementation.

        If low and high are arrays, their shapes need to match that of the ids.

        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return uniform(low, high, self._states, self._new_states, ids, self.to_tuple(shape), self._device)

    def sample_sign_warp(self, dtype: str, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
        """Sample a sign. Warp implementation.
        Args:
            dtype: The data type of the output tensor.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return rand_sign_fn(self._states, self._new_states, ids, self.to_tuple(shape), dtype, self._device)

    def sample_integer_warp(
        self, low: int | wp.array, high: int | wp.array, shape: tuple | int, ids: wp.array | None = None
    ) -> wp.array:
        """Sample for a random integer. Warp implementation.
        Args:
            low: The lower bound of the distribution.
            high: The upper bound of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            wp.array: The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return integer(low, high, self._states, self._new_states, ids, self.to_tuple(shape), self._device)

    def sample_normal_warp(
        self, mean: float | wp.array, std: float | wp.array, shape: tuple | int, ids: wp.array | None = None
    ) -> wp.array:
        """Sample from a normal distribution. Warp implementation.
        Args:
            mean: The mean of the distribution.
            std: The standard deviation of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            wp.array: The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return normal(mean, std, self._states, self._new_states, ids, self.to_tuple(shape), self._device)

    def sample_poisson_warp(self, lam: float | wp.array, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
        """Sample from a poisson distribution. Warp implementation.
        Args:
            lam: The rate of the distribution.
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return poisson(lam, self._states, self._new_states, ids, self.to_tuple(shape), self._device)

    def sample_quaternion_warp(self, shape: tuple | int, ids: wp.array | None = None) -> wp.array:
        """Sample a quaternion. Warp implementation.
        Args:
            shape: The shape of the output tensor.
            ids: The ids of the environments.
        Returns:
            wp.array: The sampled values."""
        if ids is None:
            ids = self._ALL_INDICES
        return quaternion(self._states, self._new_states, ids, self.to_tuple(shape), self._device)
