"""Top-level facade for native batched gate sequence generation.

``GateGenerator`` mirrors the fixed-batch contract of ``TrackGenerator``: construction
owns persistent output and scratch buffers (and runs one real eager generation), while
each generation refreshes the per-env seed buffer from the rng and writes into the same
``GateSequence`` instance every call. Native gate generators provide centerline-generator
anchors through the registry; ``warp_gate`` owns the common ordering, collision, frame,
and validity pipeline.

Execution follows :class:`track_gen.exec.GraphRunner`, exactly like ``TrackGenerator``:
record into a caller's capture, self-graph replay on CUDA, eager on CPU.
"""

import warp as wp

from ..exec.context import LaunchContext
from ..exec.runner import GraphRunner
from .runtime import _snapshot_config
from .types import GateGenConfig, GateSequence

__all__ = ["GateGenConfig", "GateGenerator", "GateSequence"]


class GateGenerator:
    """Facade for fixed-batch native gate sequence generation.

    Mirrors the fixed-batch contract of :class:`~track_gen.TrackGenerator`: construction
    allocates persistent output and scratch buffers once (via ``warp_gate._gate_warp_alloc``),
    and every generation refreshes the per-env seed buffer then writes results into the
    same :class:`~track_gen.GateSequence` instance (stable ``.ptr`` pointers).
    Use ``GateSequence.clone()`` when an independent snapshot is needed.

    Construction runs one real eager generation (warm-up), so it must happen outside any
    CUDA graph capture. Afterwards ``generate()`` / :meth:`enqueue` can be recorded into a
    caller's CUDA graph; outside a capture ``generate()`` replays the generator's own graph
    (:attr:`graph`) on CUDA, unless the generator registers ``capturable=False``, in which
    case it runs eagerly on CUDA every call (and refuses to be recorded). On the Warp
    ``cpu`` device the pipeline runs eagerly.
    """

    def __init__(self, config: GateGenConfig, rng, *, stream: "wp.Stream | None" = None) -> None:
        """
        Args:
            config: The gate generation configuration. ``config.generator`` must be a
                registered gate generator; ``config.gate_ordering`` must be supported
                by that generator; and ``config.max_gates`` must satisfy the
                generator's capacity requirements. The generator keeps a private copy
                (a ``wp.Device`` value is shared, not copied): mutating ``config``
                afterwards has no effect.
            rng: A ``PerEnvSeededRNG`` instance with one seed per configured env. Its
                CURRENT seeds are read on every generation (also by every replay).
            stream: optional default ``wp.Stream`` for calls that pass no ``stream=``
                and run outside an ambient ``LaunchContext`` scope (see
                :mod:`track_gen.exec`).

        Raises:
            ValueError: if ``rng`` is ``None``.
            ValueError: if the number of seeds in ``rng`` does not match
                ``config.num_envs``.
            ValueError: if ``config.gate_ordering`` is not supported by the selected
                gate generator.
            ValueError: if ``config.min_gates`` exceeds the maximum gate count the
                generator can produce, or if ``config.max_gates`` is too small for
                the generator's required capacity.
            track_gen.exec.CaptureError: if the resolved stream or Warp's current
                stream is capturing (construction allocates and runs a real warm-up
                pass); raised before anything is allocated.
        """
        if rng is None:
            raise ValueError("A random number generator must be provided.")
        # Catch an env-count mismatch at construction (clear message) rather than as an
        # opaque wp.copy size error on the first generation.
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

        # Snapshot (see TrackGenerator): host scalars from the config are baked into
        # the launches and any recorded graph.
        self._config = _snapshot_config(config)
        config = self._config
        self._rng = rng
        self._generator_spec = generator_spec
        # Non-capturable generators (host-driven loops / per-call allocation) run eagerly
        # on CUDA every call and refuse to be recorded into a caller's graph.
        self._capturable = generator_spec.capturable

        self._runner = GraphRunner(
            str(config.device),
            capturable=self._capturable,
            legacy_stream=not getattr(generator_spec, "accepts_ctx", False),
            default_stream=stream,
            owner=self,
            name=config.generator,
        )
        # Refuse a capture BEFORE allocating (see TrackGenerator).
        self._runner.ensure_not_capturing()

        from . import warp_gate

        self._gate_sequence: GateSequence
        self._gate_sequence, self._scratch = warp_gate._gate_warp_alloc(
            config, generator_spec=generator_spec
        )
        # Debug-only: per-env [E] int32 counter of frame fallbacks (near-vertical
        # or degenerate tangents) accumulated during the last generation. Reset
        # and rewritten each call; not part of the public GateSequence contract.
        # scratch is (gen_scratch, pos2, cum, perim, z, fallbacks) — fallbacks
        # is at index 5 (moved from 4 when the z-profile perim slot was added).
        self.frame_fallbacks: wp.array = self._scratch[5]

        self._seed_buf: wp.array = wp.empty(
            int(config.num_envs), dtype=wp.int32, device=str(config.device)
        )

        # One real eager generation outside any capture (loads every module the hot
        # path uses; leaves the batch for the rng's current seeds in the result), then
        # a host wait on its stream: the result is complete when construction returns.
        self._runner.prepare(self._enqueue)

    # -- properties ---------------------------------------------------------------
    @property
    def config(self) -> GateGenConfig:
        """A copy of the configuration this generator was built with (a
        ``wp.Device`` value is shared, not copied).

        Read-only in effect: mutating the returned object (or the object passed to
        the constructor) does not change this generator.
        """
        return _snapshot_config(self._config)

    @property
    def graph(self) -> "wp.Graph | None":
        """The generator's own CUDA graph: captured on the first CUDA ``generate()``
        made outside any capture, replayed by every later one. ``None`` on CPU, for a
        non-capturable generator, and before that first call."""
        return self._runner.graph

    @property
    def _graph(self) -> "wp.Graph | None":
        """Read-only alias of :attr:`graph` (backward compatibility)."""
        return self._runner.graph

    # -- enqueue body -------------------------------------------------------------
    def _enqueue(self, ctx: LaunchContext) -> None:
        """Emit one whole gate generation on ``ctx``: seed refresh then pipeline.

        The ONLY generation body. Zero allocation and host-sync free; the seed copy
        is part of it, so a recorded graph reads the rng's live seeds on every replay.
        """
        from . import warp_gate

        ctx.copy(self._seed_buf, self._rng.seeds_warp)
        warp_gate._run_gate_pipeline(
            self._config,
            self._seed_buf,
            out=self._gate_sequence,
            scratch=self._scratch,
            generator_spec=self._generator_spec,
            ctx=ctx,
        )

    def _run(self, ctx: "LaunchContext | None" = None) -> None:
        """Backward-compatible alias: run :meth:`_enqueue` on ``ctx`` (default:
        ``LaunchContext.current(device)``, Warp's current stream)."""
        self._enqueue(ctx or LaunchContext.current(str(self._config.device)))

    # -- public calls -------------------------------------------------------------
    def generate(self, num_or_ids=None, *, stream=None, sync: bool = True) -> GateSequence:
        """Generate gates for the fixed configured batch.

        ``GateGenerator`` always operates on exactly ``config.num_envs`` environments.
        Passing an integer is accepted only when it matches that batch size. Explicit
        environment-id sequences are rejected because selected-env execution is outside
        the fixed-batch graph-capturable contract.

        Execution (stream resolved per :mod:`track_gen.exec`):

        - inside a capture on the resolved stream: RECORDED into the caller's graph
          (seed refresh included; no sync, no self-capture, no ``wp.capture_launch``);
        - outside a capture on CUDA: replay of :attr:`graph` (captured on the first
          such call) on the resolved stream;
        - on the Warp ``cpu`` device, or for a ``capturable=False`` generator: eager.

        Writes results into ``self._gate_sequence`` in place and returns the SAME
        instance every call (stable ``.ptr`` pointers). Use ``GateSequence.clone()``
        to obtain an independent snapshot.

        Args:
            num_or_ids: Optional. Either ``None`` or the integer batch size. When an
                integer is provided, it must equal ``config.num_envs``. Explicit
                environment-id sequences are not supported.
            stream: optional ``wp.Stream`` (or ``"torch"``) to run on.
            sync: block the host until the generation finished on the resolved stream
                (default ``True``). Never blocks while that stream is capturing, nor
                while torch is capturing in the calling thread. ``sync=False`` returns
                immediately; results are stream-ordered.

        Returns:
            ``self._gate_sequence`` — the same :class:`~track_gen.GateSequence` instance
            every call (stable pointers).

        Raises:
            TypeError: if ``num_or_ids`` is not an ``int`` (i.e. an explicit
                environment-id sequence was passed).
            ValueError: if an integer ``num_or_ids`` differs from ``config.num_envs``.
            track_gen.exec.NotCapturableError: recording a ``capturable=False``
                generator into a capture (raised before any launch).
            track_gen.exec.LegacyPluginStreamError: recording a legacy plugin
                generator on a stream that is not Warp's current stream.
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
        self._runner.run(self._enqueue, stream=stream, sync=sync)
        return self._gate_sequence

    def enqueue(self, *, stream=None) -> GateSequence:
        """Enqueue one generation without self-graphing or blocking.

        Inside a capture on the resolved stream the generation is recorded into the
        caller's graph; otherwise it runs eagerly on the resolved stream. Never
        captures :attr:`graph`, never host-syncs.

        Returns:
            ``self._gate_sequence`` (results are stream-ordered).

        Raises:
            track_gen.exec.NotCapturableError: recording a ``capturable=False``
                generator into a capture (raised before any launch).
        """
        self._runner.run(self._enqueue, stream=stream, policy="eager")
        return self._gate_sequence
