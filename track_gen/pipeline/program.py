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
    seeds (``seed``) -> the dynamic-parameter / input latch -> [masked: hold the
    unmasked envs' result rows] -> every stage in order (EVERY env) -> the commit
    (masked: restore the unmasked envs; derive ``valid``; ``on_invalid="poison"``) ->
    ``result.generation <- epoch`` (masked envs only, with a mask). It allocates nothing
    and never host-syncs. See :mod:`track_gen.stages.commit` for what the commit costs
    (nothing for an unmasked generation of the default pipeline). The result is a
    ``Track``, or a ``GateSequence`` for a gate pipeline (``BatchSpec.max_gates`` set,
    roadmap PR11).
    """

    def __init__(self, *, _token=None, spec, layout, binding, result, seeds, seed,
                 inputs, params, pairs, enqueues, versions, scratch, stage_ids, kinds,
                 gen_scratch, stream=None, owner=None, commit=None, fail_bits=None,
                 fail_owner=None, capture_class=None, on_invalid: str = "keep",
                 writers=None, ports=None) -> None:
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
        self._commit = commit                     # stages.commit._Commit, or None
        self._fail_bits = types.MappingProxyType(dict(fail_bits or {}))
        self._fail_owner = dict(fail_owner or {})  # bit name -> stage id
        self._on_invalid = str(on_invalid)
        self._channel_writers = {ch: tuple(ids) for ch, ids in (writers or {}).items()}
        self._ports = dict(ports or {})            # stage_id -> (reads, writes)
        if capture_class is None:
            capture_class = CaptureClass.STATIC if binding.capturable else \
                CaptureClass.HOST
        self._capture_class = CaptureClass(capture_class)
        self._capturable = self._capture_class != CaptureClass.HOST
        self._legacy_stream = not bool(binding.accepts_ctx)
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
        """The result's :class:`~track_gen.plugin.Layout`: ``stride`` is ``N_max`` for a
        track, ``max_gates`` for a gate pipeline (each stage was built with the layout
        of its own point domain)."""
        return self._layout

    @property
    def device(self):
        """The ``wp.Device``."""
        return self._device

    @property
    def result(self) -> "Track":
        """The :class:`~track_gen.Track` (a gate pipeline: the
        :class:`~track_gen.GateSequence`) every generation writes: the same instance,
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
        """``CaptureClass.STATIC`` (capturable) or ``HOST`` (eager every call): the
        largest of the generator's and every stage's (``Stage.capture_class`` of its
        params); the built-in stages are STATIC."""
        return self._capture_class

    @property
    def fail_bits(self) -> Mapping[str, int]:
        """``name -> bit`` of every ``valid.fail`` bit, in assignment order: the
        built-in checks' (``BUILTIN_FAIL_BITS``, bits 0-7; a gate pipeline:
        ``GATE_FAIL_BITS``, bits 0-5) then each plugin validity stage's (16, 17, ... in
        pipeline order). ``Track.fail_bits`` / ``GateSequence.fail_bits`` is the same
        mapping."""
        return self._fail_bits

    @property
    def on_invalid(self) -> str:
        """``"keep"`` or ``"poison"`` (``BatchSpec.on_invalid``)."""
        return self._on_invalid

    @property
    def channels(self) -> Mapping[str, "tuple[str, ...]"]:
        """``channel -> ids of the stages that write it``, in order (``"seed"``:
        ``("<prologue>",)``)."""
        return types.MappingProxyType(dict(self._channel_writers))

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
        prologue, the commit and the ``generation`` write cover the masked envs only:
        every other env's ``Track`` rows end the generation bit for bit as they were
        (validate ``mask`` first, as :meth:`run` does). They are held, recomputed by the
        stages and restored within the generation, all on ``ctx``'s stream: work
        ordered after the generation on that stream (or after an event recorded on it)
        never sees the recomputed rows, a reader on another stream without such a
        dependency can (see :mod:`track_gen.stages.commit`).

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
        commit = self._commit
        if commit is not None and mask is not None:
            commit.enqueue_hold(ctx, mask)
        for enqueue in self._enqueues:
            enqueue(ctx)
        if commit is not None:
            commit.enqueue_commit(ctx, mask)
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
        if self._commit is not None:
            # The masked commit's kernels run on masked generations only: load them
            # now, outside any capture, so a masked generation recorded into a caller's
            # graph never loads a module while capturing.
            import warp as wp  # noqa: PLC0415

            from ..stages import commit as _commit_mod  # noqa: PLC0415

            wp.load_module(_commit_mod, device=self._device)

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

    # -- introspection --------------------------------------------------------------
    def tap(self, channel: str, *, after: "str | None" = None) -> "wp.array":
        """The working buffer of ``channel``: its final version, or the version visible
        right after stage ``after`` (what the stages after it read).

        Read-only by contract: never write it. Stable for the Program's lifetime; it
        holds the last generation's values once that generation completed on its
        stream. Every env is recomputed each generation; the final version of a
        channel the ``Track`` exports (``track.*``, ``cl.winding``, ``cl.truncated``,
        ``valid.fail``, and ``cl.arclen2`` / ``cl.length2`` under a flat lift) IS the
        ``Track``'s array, so it holds the committed values (after a masked
        regeneration: an unmasked env's previous track; with ``on_invalid="poison"``:
        the poisoned rows). One exception: ``track.valid`` is ``Track.valid`` only when
        ``validity.default`` is the last writer of ``valid.fail``. After a plugin check
        the commit derives ``Track.valid`` from ``valid.fail``, and ``tap("track.valid")``
        is ``validity.default``'s own 0/1 flag, which ignores the later checks (and
        poison): read ``Track.valid``, or ``tap("valid.fail") == 0``.
        ``tap("seed")`` is the effective seed.

        Raises:
            KeyError: no stage writes ``channel``, ``after`` is not a stage id, or
                ``channel`` is not written by ``after`` or an earlier stage.
        """
        versions = self._versions.get(channel)
        if not versions:
            raise KeyError(f"no stage writes channel {channel!r}; channels: "
                           f"{sorted(self._versions)}")
        if after is None:
            return versions[-1][2]
        if after not in self._stage_ids:
            raise KeyError(f"no stage {after!r}; stages: {list(self._stage_ids)}")
        limit = self._stage_ids.index(after)
        visible = [arr for index, _, arr in versions if index <= limit]
        if not visible:
            raise KeyError(f"channel {channel!r} is not written by {after!r} or earlier")
        return visible[-1]

    def _channel(self, name: str, *, after: "str | None" = None) -> "wp.array":
        """Private alias of :meth:`tap` (tests, the ``_scratch`` view)."""
        return self.tap(name, after=after)

    def _channel_or_none(self, name: str, after: "str | None" = None):
        try:
            return self._channel(name, after=after)
        except KeyError:
            return None

    def _writers(self, name: str) -> "tuple[str, ...]":
        """The ids of the stages that wrote ``name``, in order."""
        return tuple(sid for _, sid, _ in self._versions.get(name, ()))
