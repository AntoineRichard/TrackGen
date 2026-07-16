"""Top-level facade for native batched gate sequence generation.

``GateGenerator`` mirrors the fixed-batch contract of ``TrackGenerator``: construction
owns persistent output and scratch buffers, while ``generate()`` refreshes the per-env
seed buffer and writes into the same ``GateSequence`` instance every call. Native gate
generators provide centerline-generator anchors through the registry; ``warp_gate`` owns the
common ordering, collision, frame, and validity pipeline.
"""

import warp as wp

from .types import GateGenConfig, GateSequence

__all__ = ["GateGenConfig", "GateGenerator", "GateSequence"]


class GateGenerator:
    """Facade for fixed-batch native gate sequence generation.

    Mirrors the fixed-batch contract of :class:`~track_gen.TrackGenerator`: construction
    allocates persistent output and scratch buffers once (via ``warp_gate._gate_warp_alloc``),
    and ``generate()`` refreshes the per-env seed buffer then writes results into the same
    :class:`~track_gen.GateSequence` instance on every call (stable ``.ptr`` pointers).
    Use ``GateSequence.clone()`` when an independent snapshot is needed.

    On a CUDA device the gate pipeline is auto-captured into a ``wp.Graph`` on the first
    ``generate()`` call and replayed on subsequent calls. On the Warp ``cpu`` device the
    pipeline runs eagerly.
    """

    def __init__(self, config: GateGenConfig, rng) -> None:
        """
        Args:
            config: The gate generation configuration. ``config.generator`` must be a
                registered gate generator; ``config.gate_ordering`` must be supported
                by that generator; and ``config.max_gates`` must satisfy the
                generator's capacity requirements.
            rng: A ``PerEnvSeededRNG`` instance with one seed per configured env.

        Raises:
            ValueError: if ``rng`` is ``None``.
            ValueError: if the number of seeds in ``rng`` does not match
                ``config.num_envs``.
            ValueError: if ``config.gate_ordering`` is not supported by the selected
                gate generator.
            ValueError: if ``config.min_gates`` exceeds the maximum gate count the
                generator can produce, or if ``config.max_gates`` is too small for
                the generator's required capacity.
        """
        if rng is None:
            raise ValueError("A random number generator must be provided.")
        # Catch an env-count mismatch at construction (clear message) rather than as an
        # opaque wp.copy size error on the first generate() call.
        if int(rng.seeds_warp.shape[0]) != int(config.num_envs):
            raise ValueError(
                f"rng has {int(rng.seeds_warp.shape[0])} seeds but config.num_envs="
                f"{config.num_envs!r}; construct the rng with num_envs={config.num_envs!r}."
            )

        from . import gate_generator_registry

        generator_spec = gate_generator_registry.get(config.generator)
        if config.gate_ordering not in generator_spec.supported_orderings:
            raise ValueError(
                f"gate generator {config.generator!r} does not support "
                f"gate_ordering {config.gate_ordering!r}; supported: "
                f"{sorted(generator_spec.supported_orderings)}"
            )

        required_max_gates = int(generator_spec.max_gates(config))
        if required_max_gates < int(config.min_gates):
            raise ValueError(
                f"GateGenConfig.min_gates={config.min_gates!r} is too large for "
                f"gate generator {config.generator!r}; generator can produce at most "
                f"{required_max_gates} gates."
            )
        if required_max_gates > int(config.max_gates):
            raise ValueError(
                f"GateGenConfig.max_gates={config.max_gates!r} is too small for "
                f"gate generator {config.generator!r}; required max_gates "
                f"{required_max_gates}."
            )

        self._config = config
        self._rng = rng
        self._generator_spec = generator_spec

        from . import warp_gate

        self._gate_sequence: GateSequence
        self._gate_sequence, self._scratch = warp_gate._gate_warp_alloc(
            config, generator_spec=generator_spec
        )

        self._seed_buf: wp.array = wp.empty(
            int(config.num_envs), dtype=wp.int32, device=str(config.device)
        )
        self._graph: "wp.Graph | None" = None

    def _run(self) -> None:
        from . import warp_gate

        warp_gate._run_gate_pipeline(
            self._config,
            self._seed_buf,
            out=self._gate_sequence,
            scratch=self._scratch,
            generator_spec=self._generator_spec,
        )

    def generate(self, num_or_ids=None) -> GateSequence:
        """Generate gates for the fixed configured batch.

        ``GateGenerator`` always operates on exactly ``config.num_envs`` environments.
        Passing an integer is accepted only when it matches that batch size. Explicit
        environment-id sequences are rejected because selected-env execution is outside
        the fixed-batch graph-capturable contract.

        Writes results into ``self._gate_sequence`` in place and returns the SAME
        instance every call (stable ``.ptr`` pointers). Use ``GateSequence.clone()``
        to obtain an independent snapshot.

        Args:
            num_or_ids: Optional. Either ``None`` or the integer batch size. When an
                integer is provided, it must equal ``config.num_envs``. Explicit
                environment-id sequences are not supported.

        Returns:
            ``self._gate_sequence`` — the same :class:`~track_gen.GateSequence` instance
            every call (stable pointers).

        Raises:
            TypeError: if ``num_or_ids`` is not an ``int`` (i.e. an explicit
                environment-id sequence was passed).
            ValueError: if an integer ``num_or_ids`` differs from ``config.num_envs``.
        """
        if num_or_ids is not None:
            if not isinstance(num_or_ids, int):
                raise TypeError(
                    "GateGenerator.generate() does not accept explicit environment ids; "
                    "construct a generator with the desired fixed num_envs instead."
                )
            if num_or_ids != self._config.num_envs:
                raise ValueError(
                    f"GateGenerator is fixed-batch for {self._config.num_envs} envs; "
                    f"got num_or_ids={num_or_ids}. Construct a new GateGenerator "
                    f"with num_envs={num_or_ids} instead."
                )

        wp.copy(self._seed_buf, self._rng.seeds_warp)

        dev = str(self._config.device)
        if "cuda" in dev:
            if self._graph is None:
                from . import warp_gate, warp_pipeline

                prev_gate_capturing = warp_gate._CAPTURING
                prev_pipeline_capturing = warp_pipeline._CAPTURING
                warp_gate._CAPTURING = True
                warp_pipeline._CAPTURING = True
                try:
                    for _ in range(3):
                        self._run()
                    wp.synchronize()

                    with wp.ScopedCapture(device=dev) as cap:
                        self._run()
                    self._graph = cap.graph
                finally:
                    warp_gate._CAPTURING = prev_gate_capturing
                    warp_pipeline._CAPTURING = prev_pipeline_capturing

            wp.capture_launch(self._graph)
            wp.synchronize()
        else:
            self._run()

        return self._gate_sequence
