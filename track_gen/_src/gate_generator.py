"""Top-level facade skeleton for native batched gate sequence generation.

``GateGenerator`` mirrors the fixed-batch contract of ``TrackGenerator``: construction
owns persistent output and scratch buffers, while ``generate()`` refreshes the per-env
seed buffer and writes into the same ``GateSequence`` instance every call. The actual
Warp allocation and run functions live in the future ``warp_gate`` module, so today this
facade mainly provides the registry and validation boundary.
"""

import warp as wp

from .types import GateGenConfig, GateSequence

__all__ = ["GateGenConfig", "GateGenerator", "GateSequence"]


class GateGenerator:
    """Facade for fixed-batch native gate sequence generation."""

    def __init__(self, config: GateGenConfig, rng) -> None:
        """Args:
        config: The gate generation configuration.
        rng: A ``PerEnvSeededRNG`` instance with one seed per configured env.
        """
        if rng is None:
            raise ValueError("A random number generator must be provided.")

        from . import gate_generator_registry

        generator_spec = gate_generator_registry.get(config.generator)
        if config.gate_ordering not in generator_spec.supported_orderings:
            raise ValueError(
                f"gate generator {config.generator!r} does not support "
                f"gate_ordering {config.gate_ordering!r}; supported: "
                f"{sorted(generator_spec.supported_orderings)}"
            )

        required_max_gates = int(generator_spec.max_gates(config))
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
                from . import warp_gate

                warp_gate._CAPTURING = True
                try:
                    for _ in range(3):
                        self._run()
                    wp.synchronize()

                    with wp.ScopedCapture(device=dev) as cap:
                        self._run()
                    self._graph = cap.graph
                finally:
                    warp_gate._CAPTURING = False

            wp.capture_launch(self._graph)
            wp.synchronize()
        else:
            self._run()

        return self._gate_sequence
