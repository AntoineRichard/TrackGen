"""The :class:`Program`: a built pipeline, run under the ``GraphRunner`` policy."""
from __future__ import annotations

import functools
import types
from typing import TYPE_CHECKING, Mapping

from .._src.generator_binding import _LATCH_ALL, _LATCH_MASKED
from ..exec.runner import GraphRunner

if TYPE_CHECKING:  # pragma: no cover
    import warp as wp

    from .._src.types import Track
    from ..exec.context import LaunchContext
    from ..plugin.params import ParamTable
    from ..seeds import SeedState
    from .spec import PipelineSpec

__all__ = ["Program"]

_PROGRAM_TOKEN = object()


class Program:
    """A built pipeline: every buffer allocated once, every stage's ``enqueue`` bound.

    Created by :func:`track_gen.pipeline.build` (never directly). A ``GraphRunner``
    launchable: :meth:`run` records into a caller's capture, replays the Program's own
    graph on CUDA, or runs eagerly, exactly as ``TrackGenerator.generate`` does.

    One generation (:meth:`enqueue`) is: ``[seeds.advance(mask)]`` -> the effective
    seeds (``seed``) -> the dynamic-parameter / input latch -> every stage in order ->
    ``result.generation <- epoch``. It allocates nothing and never host-syncs.
    """

    def __init__(self, *, _token=None, spec, layout, binding, result, seeds, seed,
                 inputs, params, pairs, enqueues, versions, scratch, stage_ids, kinds,
                 gen_scratch, stream=None, owner=None) -> None:
        if _token is not _PROGRAM_TOKEN:
            raise TypeError("Program is created by track_gen.pipeline.build()")
        import warp as wp  # noqa: PLC0415

        from ..plugin.stage import CaptureClass  # noqa: PLC0415

        self._spec = spec
        self._layout = layout
        self._binding = binding
        self._device = wp.get_device(layout.device)
        self._result = result
        self._seeds = seeds
        self._seed = seed
        self._inputs = types.MappingProxyType(dict(inputs))
        self._params = params
        self._pairs = list(pairs)                 # (live, work) latch pairs, in order
        self._enqueues = list(enqueues)           # one per stage, in order
        self._versions = versions                 # channel -> [(index, stage_id, array)]
        self._scratch = scratch                   # stage_id -> {name: array}
        self._stage_ids = tuple(stage_ids)
        self._kinds = dict(kinds)                 # stage_id -> (kind, name)
        self._gen_scratch = gen_scratch           # the generator's private scratch
        self._capturable = bool(binding.capturable)
        self._legacy_stream = not bool(binding.accepts_ctx)
        self._capture_class = CaptureClass.STATIC if self._capturable else \
            CaptureClass.HOST
        self._E = int(layout.num_envs)
        self._runner = GraphRunner(
            self._device, capturable=self._capturable, legacy_stream=self._legacy_stream,
            default_stream=stream, owner=self if owner is None else owner,
            name=str(binding.name))

    def __repr__(self) -> str:
        return (f"Program(stages={list(self._stage_ids)}, E={self._E}, "
                f"device={self._layout.device!r})")

    # -- state ------------------------------------------------------------------
    @property
    def spec(self) -> "PipelineSpec":
        """The resolved spec: every ``StageRef.params`` is the stage's Params instance
        (the generator's: its resolved Params, ``None`` for a legacy spec without
        one)."""
        return self._spec

    @property
    def layout(self):
        """The :class:`~track_gen.plugin.Layout` every stage was built with."""
        return self._layout

    @property
    def device(self):
        """The ``wp.Device``."""
        return self._device

    @property
    def result(self) -> "Track":
        """The :class:`~track_gen.Track` every generation writes: the same instance,
        with the same array addresses, for the Program's lifetime."""
        return self._result

    @property
    def seeds(self) -> "SeedState":
        """The per-env :class:`~track_gen.seeds.SeedState` (seeds and epoch)."""
        return self._seeds

    @property
    def seed(self) -> "wp.array":
        """The ``[E]`` int32 effective seed (channel ``"seed"``), written by the
        prologue of every generation; stages read it as ``StageBuilder.seed``."""
        return self._seed

    @property
    def inputs(self) -> Mapping:
        """Persistent device inputs: ``"regen_mask"`` (``[E]`` int32, zeroed) and every
        input a stage declared (``StageBuilder.input``). Stable pointers."""
        return self._inputs

    @property
    def params(self) -> "ParamTable":
        """Runtime parameters keyed by stage id (live ``[E]`` arrays of the dynamic
        fields; the built-in stages' fields are all static)."""
        return self._params

    @property
    def capture_class(self):
        """``CaptureClass.STATIC`` (capturable) or ``HOST`` (eager every call), from
        the generator; the built-in stages are STATIC."""
        return self._capture_class

    @property
    def capturable(self) -> bool:
        """Whether the Program may be recorded into a caller's capture."""
        return self._capturable

    @property
    def legacy_stream(self) -> bool:
        """Whether the generator is a legacy-ABI spec (launches on Warp's current
        stream; see :class:`~track_gen.exec.GraphRunner`)."""
        return self._legacy_stream

    @property
    def runner(self) -> GraphRunner:
        """The :class:`~track_gen.exec.GraphRunner` that runs the Program."""
        return self._runner

    @property
    def graph(self) -> "wp.Graph | None":
        """The Program's own CUDA graph for a plain :meth:`run` (``None`` on CPU, for
        a non-capturable Program, and before the first such CUDA call)."""
        return self._runner.graph

    # -- the body ---------------------------------------------------------------
    def enqueue(self, ctx: "LaunchContext", *, advance: bool = False,
                mask: "wp.array | None" = None) -> "Track":
        """Launch one generation on ``ctx`` and return :attr:`result`.

        No allocation, no host sync, no branch on device data. With ``mask`` the
        prologue and the ``generation`` write cover the masked envs only (validate
        ``mask`` first, as :meth:`run` does).

        Args:
            ctx: The :class:`~track_gen.exec.LaunchContext` to emit on.
            advance: Bump ``seeds.epoch`` first (every env, or where ``mask[e] != 0``).
            mask: Optional ``[E]`` int32 mask.

        Raises:
            track_gen.exec.CaptureError: ``ctx`` is capturing and :meth:`prepare` has
                not run (the first launches would load modules inside the capture).
        """
        if not self._runner.prepared and ctx.capturing:
            raise self._not_prepared()
        seeds = self._seeds
        if advance:
            seeds.advance(mask, ctx=ctx)
        seeds.effective(self._seed, mask, ctx=ctx)
        E = self._E
        for live, work in self._pairs:     # dynamic params / inputs, live -> work
            if mask is None:
                ctx.launch(_LATCH_ALL[work.dtype], E, [live, work])
            else:
                ctx.launch(_LATCH_MASKED[work.dtype], E, [live, mask, work])
        for enqueue in self._enqueues:
            enqueue(ctx)
        result = self._result
        if result.generation is not None:
            seeds._write_generation(result.generation, mask, ctx)
        return result

    def _body(self, advance: bool, mask) -> "tuple[object, object]":
        """``(body, self-graph key)`` of one call; validates ``advance`` / ``mask``
        host-side, before any launch."""
        if mask is not None:
            if not advance:
                raise ValueError(
                    "mask= selects the envs whose epoch advances; it requires "
                    "advance=True (without it every env is regenerated identically)")
            self._seeds.check_mask(mask)
        if not advance:
            return self.enqueue, None
        key = (True, 0 if mask is None else int(mask.ptr))
        return functools.partial(self.enqueue, advance=True, mask=mask), key

    def _not_prepared(self):
        from ..exec.errors import CaptureError  # noqa: PLC0415

        return CaptureError(
            f"Program {self._binding.name!r} was built with prepare=False and is being "
            f"recorded into a capture: call Program.prepare() outside the capture first "
            f"(its warm-up loads every kernel module)")

    # -- calls --------------------------------------------------------------------
    def prepare(self) -> None:
        """Run one real generation outside any capture and wait for it (the warm-up:
        loads every kernel the hot path uses). ``build(..., prepare=True)`` calls it.

        Raises:
            track_gen.exec.CaptureError: a capture is open (see ``GraphRunner``).
        """
        self._runner.prepare(self.enqueue)

    def run(self, *, stream=None, sync: bool = False, advance: bool = False,
            mask: "wp.array | None" = None, policy: str = "auto") -> "Track":
        """Run one generation under the ``GraphRunner`` policy (record into a caller's
        capture, replay the Program's own graph on CUDA, or eager) and return
        :attr:`result`. A Program built with ``prepare=False`` runs :meth:`prepare`
        first (outside a capture; inside one it raises ``CaptureError``).

        Args:
            stream: Optional ``wp.Stream`` (or ``"torch"``).
            sync: Block the host until the generation finished (never while
                recording).
            advance: Bump the epoch first (see ``TrackGenerator.generate``).
            mask: Optional ``[E]`` int32 mask of the envs to advance; requires
                ``advance=True``.
            policy: ``"auto"`` or ``"eager"`` (never self-graph).

        Raises:
            ValueError: an invalid ``mask``.
            track_gen.exec.NotCapturableError: recording a non-capturable Program.
            track_gen.exec.CaptureError: recording a Program that was never prepared.
        """
        body, key = self._body(advance, mask)
        if not self._runner.prepared:
            if self._runner.context(stream).capturing:
                raise self._not_prepared()
            self.prepare()
        self._runner.run(body, stream=stream, sync=sync, key=key, policy=policy)
        return self._result

    # -- introspection (tests; a public tap() arrives with roadmap PR10b) ---------
    def _channel(self, name: str, *, after: "str | None" = None) -> "wp.array":
        """The version of channel ``name`` visible after stage ``after`` (default: the
        final version). ``KeyError`` when no such version exists."""
        versions = self._versions.get(name)
        if not versions:
            raise KeyError(f"no stage writes channel {name!r}")
        if after is None:
            return versions[-1][2]
        if after not in self._stage_ids:
            raise KeyError(f"no stage {after!r}; stages: {list(self._stage_ids)}")
        limit = self._stage_ids.index(after)
        visible = [arr for index, _, arr in versions if index <= limit]
        if not visible:
            raise KeyError(f"channel {name!r} is not written by {after!r} or earlier")
        return visible[-1]

    def _channel_or_none(self, name: str, after: "str | None" = None):
        try:
            return self._channel(name, after=after)
        except KeyError:
            return None

    def _writers(self, name: str) -> "tuple[str, ...]":
        """The ids of the stages that wrote ``name``, in order."""
        return tuple(sid for _, sid, _ in self._versions.get(name, ()))
