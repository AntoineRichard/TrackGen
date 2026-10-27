"""The :class:`Program`: a built pipeline, run under the ``GraphRunner`` policy."""
from __future__ import annotations

import dataclasses
import functools
import types
from typing import TYPE_CHECKING, Mapping

from .._src.generator_binding import _LATCH_ALL, _LATCH_MASKED
from ..exec.context import _host_blocked
from ..exec.errors import NotCapturableError
from ..exec.runner import GraphRunner, _BodyPart, _SegmentedBody
from ..plugin.stage import CaptureClass

if TYPE_CHECKING:  # pragma: no cover
    import warp as wp

    from .._src.types import Track
    from ..exec.context import LaunchContext
    from ..plugin.params import ParamTable
    from ..seeds import SeedState
    from .spec import PipelineSpec

__all__ = ["AttributeInfo", "InputInfo", "Program", "Segment"]

_PROGRAM_TOKEN = object()


@dataclasses.dataclass(frozen=True)
class AttributeInfo:
    """One exported per-point attribute of a :class:`Program` (``Program.attributes``):
    what ``Track.attributes[key]`` / ``GateSequence.attributes[key]``
    holds.

    Attributes:
        key: The result key: the channel name without its domain prefix
            (``"acme.surface_id"``).
        channel: The result-domain channel whose final version is the array
            (``"cl.acme.surface_id"``, ``"gates.acme.tag"``).
        dtype: The Warp dtype (``wp.float32``, ``wp.int32``, ``wp.vec2f`` or
            ``wp.vec3f``).
        interp: The carry policy its generator declared (``"linear"``, ``"nearest"``,
            ``"previous"``, ``"angle"``), ``None`` for an attribute carried by index
            only or written by a stage on the result's points.
        padding: The value past ``count[e]``: NaN for floats (every vector component),
            ``-1`` for int32.
        domain: ``"cl"`` (a Track's points) or ``"gates"`` (a GateSequence's slots).
    """

    key: str
    channel: str
    dtype: object
    interp: "str | None"
    padding: object
    domain: str


@dataclasses.dataclass(frozen=True)
class InputInfo:
    """One persistent device input of a :class:`Program` (``Program.input_info``): what
    ``Program.inputs[name]`` is and how a generation reads it.

    Attributes:
        name: The input's name (``"regen_mask"``, a stage's ``StageBuilder.input`` name;
            ``"centerline.xy"``, ... for an ingestion generator).
        dtype: The Warp dtype.
        per: ``"env"`` (``[E]``) or ``"raw"`` (``[E * raw]``, a centerline generator's
            per-point input).
        length: The array's length.
        latched: ``True``: snapshotted (latched) at the start of every generation for
            the envs it regenerates, like a dynamic param (a write takes effect at the
            next generation); ``False``: read DURING the generation (a ``per="raw"``
            input, ``"regen_mask"``), so a writer must be stream-ordered with the
            generation (the same stream, or events).
        stage_id: The id of the stage that declared it (``None`` for
            ``"regen_mask"``).
    """

    name: str
    dtype: object
    per: str
    length: int
    latched: bool
    stage_id: "str | None"


@dataclasses.dataclass(frozen=True)
class Segment:
    """One segment of a :class:`Program` (planner rule 6): a maximal run of consecutive
    stages of one kind, HOST (run eagerly every call) or capturable (STATIC / BOUNDED:
    replayed from the Program's own graph outside a capture).

    Attributes:
        index: The segment's position (0 opens with the seed prologue, the last one
            closes with the commit).
        capture_class: ``CaptureClass.HOST``, or the largest of its stages' classes
            (``STATIC`` / ``BOUNDED``).
        stage_ids: The ids of its stages, in order.
    """

    index: int
    capture_class: CaptureClass
    stage_ids: "tuple[str, ...]"

    @property
    def host(self) -> bool:
        """Whether the segment holds HOST stages (it runs eagerly every call)."""
        return self.capture_class is CaptureClass.HOST


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
    and never host-syncs, except inside its HOST stages (their ``ctx.host()``
    readbacks, uploads and waits on the call's stream). See
    ``track_gen.stages.commit`` for what the commit costs (nothing for an unmasked
    generation of the default pipeline). The result is a
    ``Track``, or a ``GateSequence`` for a gate pipeline (``BatchSpec.max_gates`` set).

    A Program whose stages include ``CaptureClass.HOST`` ones is split into
    :attr:`segments` (consecutive stages of equal host-ness): outside a capture its
    HOST segments run eagerly and every other segment replays its own graph on CUDA
    (:meth:`enqueue_segment`); a caller's capture refuses it before any launch
    (``NotCapturableError`` naming :attr:`host_stage_ids`).
    """

    def __init__(self, *, _token=None, spec, layout, binding, result, seeds, seed,
                 inputs, params, pairs, enqueues, versions, scratch, stage_ids, kinds,
                 gen_scratch, stream=None, owner=None, commit=None, fail_bits=None,
                 fail_owner=None, capture_classes=None, segments=None,
                 on_invalid: str = "poison", writers=None, ports=None,
                 attributes=None, input_info=None) -> None:
        if _token is not _PROGRAM_TOKEN:
            raise TypeError("Program is created by track_gen.pipeline.build()")
        import warp as wp  # noqa: PLC0415

        self._spec = spec
        self._layout = layout
        self._binding = binding
        self._device = wp.get_device(layout.device)
        self._result = result
        self._seeds = seeds
        self._seed = seed
        self._inputs = types.MappingProxyType(dict(inputs))
        self._input_info = types.MappingProxyType(dict(input_info or {}))
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
        self._attributes = types.MappingProxyType(dict(attributes or {}))
        if capture_classes is None:
            capture_classes = [CaptureClass(binding.capture)] + \
                [CaptureClass.STATIC] * (len(self._stage_ids) - 1)
        caps = tuple(CaptureClass(c) for c in capture_classes)
        self._caps = caps                         # stage index -> its CaptureClass
        if segments is None:
            segments = (Segment(0, max(caps), self._stage_ids),)
        self._segments = tuple(segments)
        # segment index -> the stage indices it runs
        index = {sid: i for i, sid in enumerate(self._stage_ids)}
        self._seg_slots = tuple(tuple(index[sid] for sid in seg.stage_ids)
                                for seg in self._segments)
        self._host_stage_ids = tuple(sid for sid, c in zip(self._stage_ids, caps)
                                     if c is CaptureClass.HOST)
        self._capture_class = max(caps)
        self._capturable = not self._host_stage_ids
        self._E = int(layout.num_envs)
        self._runner = GraphRunner(
            self._device, capturable=self._capturable,
            default_stream=stream, owner=self if owner is None else owner,
            name=str(binding.name), _host_ids=self._host_stage_ids)

    def __repr__(self) -> str:
        return (f"Program(stages={list(self._stage_ids)}, E={self._E}, "
                f"device={self._layout.device!r})")

    # -- state ------------------------------------------------------------------
    @property
    def spec(self) -> "PipelineSpec":
        """The resolved spec, in which every ``StageRef.params`` is the stage's Params
        instance (the generator's resolved Params; ``None`` for a legacy spec without
        one)."""
        return self._spec

    @property
    def layout(self):
        """The result's :class:`~track_gen.results.Layout`, whose ``stride`` is ``N_max``
        for a track and ``max_gates`` for a gate pipeline (each stage was built with the
        layout of its own point domain)."""
        return self._layout

    @property
    def device(self):
        """The ``wp.Device``."""
        return self._device

    @property
    def result(self) -> "Track":
        """The :class:`~track_gen.results.Track` (a gate pipeline: the
        :class:`~track_gen.results.GateSequence`) every generation writes: the same
        instance, with the same array addresses, for the Program's lifetime."""
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
        """The persistent device inputs, ``"regen_mask"`` (``[E]`` int32, zeroed) and every
        input a stage declared (``StageBuilder.input``). Stable pointers.

        Latched inputs (``per="env"``) are snapshotted at the start of a generation, so
        a write takes effect for the envs the next generation regenerates. Unlatched
        inputs (``per="raw"``, e.g. an ingestion generator's ``centerline.*``) are read
        DURING the generation: write them on the generation's stream (or order the
        streams with events). :attr:`input_info` tells them apart."""
        return self._inputs

    @property
    def input_info(self) -> Mapping[str, "InputInfo"]:
        """``name -> InputInfo`` of every entry of :attr:`inputs` (its dtype, ``per``,
        length, declaring stage, and whether it is LATCHED at the start of a generation
        or read during it)."""
        return self._input_info

    @property
    def params(self) -> "ParamTable":
        """Runtime parameters keyed by stage id (live ``[E]`` arrays of the dynamic
        fields). The dynamic fields of the built-ins are the generators' shape knobs
        (``rad``, ``handle_clamp_frac``, ``hull_displacement``, the polar, voronoi and
        checkpoint knobs), the elevation stages' ``z_min`` / ``z_max`` / ``z_max_step``
        / ``z_noise_amplitude``, ``width.from_channel``'s ``scale`` / ``lo`` / ``hi``
        and a ``randomize.uniform`` stage's ``lo`` / ``hi`` (see
        :ref:`tuning-per-env`); every other field is static (``set`` raises
        ``KeyError``; build a new generator to change it). A field drawn by a
        ``randomize.*`` stage maps to its drawn array (written every generation by the
        randomizer; ``set`` raises)."""
        return self._params

    @property
    def capture_class(self) -> CaptureClass:
        """The largest of the generator's and every stage's ``CaptureClass``
        (``Stage.capture_class`` of its params; a legacy spec: ``STATIC``, or ``HOST``
        with ``capturable=False``): ``STATIC``, ``BOUNDED`` (an iterative stage; still
        recordable into a caller's capture) or ``HOST`` (see :attr:`segments`). The
        built-in stages are STATIC."""
        return self._capture_class

    @property
    def segments(self) -> "tuple[Segment, ...]":
        """The Program's :class:`Segment` s: maximal runs of consecutive stages of equal
        host-ness (one segment unless some stage is HOST). Outside a capture each HOST
        segment runs eagerly and every other segment replays its own graph on CUDA."""
        return self._segments

    @property
    def host_stage_ids(self) -> "tuple[str, ...]":
        """The ids of the ``CaptureClass.HOST`` stages (``()``: the Program is
        recordable into a caller's capture)."""
        return self._host_stage_ids

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
        ``("<prologue>",)``), the carriers' implicit writes included (a resample
        writing the ``cl`` image of a generator's raw attribute)."""
        return types.MappingProxyType(dict(self._channel_writers))

    @property
    def attributes(self) -> Mapping[str, "AttributeInfo"]:
        """``key -> AttributeInfo`` of every per-point attribute the result exports
        (``Track.attributes`` / ``GateSequence.attributes``): the source of truth for
        what reached the result (gate attributes are best effort). Empty when no stage
        exports one."""
        return self._attributes

    @property
    def capturable(self) -> bool:
        """Whether the Program may be recorded into a caller's capture (it has no HOST
        stage)."""
        return self._capturable


    @property
    def runner(self) -> GraphRunner:
        """The :class:`~track_gen.exec.GraphRunner` that runs the Program."""
        return self._runner

    @property
    def prepared(self) -> bool:
        """Whether :meth:`prepare` ran (one real generation outside any capture;
        ``build(..., prepare=True)`` and the facades run it). Part of the
        ``track_gen.pipeline.Producer`` surface."""
        return self._runner.prepared

    @property
    def default_stream(self) -> "wp.Stream | None":
        """The stream calls run on when they pass no ``stream=`` and no ambient scope
        applies (``build(..., stream=)``, the facades' ``stream=``; ``None``: resolved
        per :mod:`track_gen.exec`). Part of the ``track_gen.pipeline.Producer``
        surface."""
        return self._runner.default_stream

    @property
    def graph(self) -> "wp.Graph | None":
        """The Program's own CUDA graph for a plain :meth:`run` (``None`` on CPU, for
        a Program with HOST stages, whose capturable segments get one graph each, and
        before the first such CUDA call)."""
        return self._runner.graph

    def segment_graphs(self, key=None) -> "tuple | None":
        """The per-segment CUDA graphs of a Program with HOST stages (its own graph, cut
        at each HOST segment), one per segment in order (``None`` at a HOST segment),
        for the graph key ``key`` (``None``:
        a plain :meth:`run`; ``(True, mask.ptr or 0)`` for ``advance=True``); ``None``
        before the first such CUDA call outside a capture, on CPU, and for a Program
        without HOST stages (see :attr:`graph`)."""
        return self._runner.segment_graphs_for(key)

    # -- the body ---------------------------------------------------------------
    def enqueue(self, ctx: "LaunchContext", *, advance: bool = False,
                mask: "wp.array | None" = None) -> "Track":
        """Launch one generation on ``ctx`` and return :attr:`result`.

        No allocation, no branch on device data, and no host sync except inside the
        HOST stages (their ``ctx.host()`` operations wait on ``ctx``'s stream). With
        ``mask`` the
        prologue, the commit and the ``generation`` write cover the masked envs only:
        every other env's ``Track`` rows end the generation bit for bit as they were
        (validate ``mask`` first, as :meth:`run` does). They are held, recomputed by the
        stages and restored within the generation, all on ``ctx``'s stream: work
        ordered after the generation on that stream (or after an event recorded on it)
        never sees the recomputed rows, a reader on another stream without such a
        dependency can (see ``track_gen.stages.commit``).

        Args:
            ctx: The :class:`~track_gen.exec.LaunchContext` to emit on.
            advance: Bump ``seeds.epoch`` first (every env, or where ``mask[e] != 0``).
            mask: Optional ``[E]`` int32 mask.

        An iterative (``BOUNDED``) stage runs its full iteration bound when called
        eagerly on CUDA (its own CUDA graph skips converged windows on device).

        Raises:
            track_gen.exec.NotCapturableError: the Program has HOST stages and ``ctx`` is
                capturing (or a CPU graph capture is open on the device); raised before
                any launch.
            track_gen.exec.CaptureError: ``ctx`` is capturing and :meth:`prepare` has
                not run (the first launches would load modules inside the capture).
        """
        if self._host_stage_ids and _host_blocked(ctx):
            raise self._not_capturable(self._host_stage_ids)
        if not self._runner.prepared and ctx.capturing:
            raise self._not_prepared()

        def body(c) -> None:
            for seg in self._segments:
                self._enqueue_segment(seg.index, c, advance, mask)
        body(ctx)
        return self._result

    def enqueue_segment(self, i: int, ctx: "LaunchContext", *, advance: bool = False,
                        mask: "wp.array | None" = None) -> "Track":
        """Launch segment ``i`` of one generation on ``ctx`` (see :attr:`segments`) and
        return :attr:`result`. Segment 0 opens with the prologue (``advance``, the
        effective seeds, the latch, the masked hold), the last one closes with the
        commit and the ``generation`` write; :meth:`enqueue` runs every segment in
        order. A segment that is not HOST may be recorded into a caller's capture once
        the Program is prepared (e.g. record the capturable tail, run the HOST head
        eagerly).

        One generation is every segment, in order, on one stream, with the SAME
        ``advance`` and ``mask`` (a recorded segment keeps the arguments it was recorded
        with: run the eager segments with those). This is not checked: a replayed
        segment runs no Python, so the Program cannot see which arguments it had.
        Mixing arguments, running segments out of order or skipping one leaves the
        result undefined.

        Raises:
            IndexError: no segment ``i``.
            track_gen.exec.NotCapturableError: segment ``i`` is HOST and ``ctx`` is
                capturing (or a CPU graph capture is open on the device).
            track_gen.exec.CaptureError: ``ctx`` is capturing and :meth:`prepare` has
                not run.
        """
        if isinstance(i, bool) or not isinstance(i, int) or \
                not 0 <= i < len(self._segments):
            raise IndexError(f"no segment {i!r}; the Program has {len(self._segments)}")
        seg = self._segments[i]
        if seg.host and _host_blocked(ctx):
            raise self._not_capturable(tuple(seg.stage_ids))
        if not self._runner.prepared and ctx.capturing:
            raise self._not_prepared()
        self._enqueue_segment(i, ctx, advance, mask)
        return self._result


    def _enqueue_segment(self, i: int, ctx, advance: bool, mask) -> None:
        """Segment ``i``'s launches (no checks): [the prologue] -> its stages -> [the
        commit and the generation write]."""
        seeds = self._seeds
        commit = self._commit
        if i == 0:
            if advance:
                seeds.advance(mask, ctx=ctx)
            seeds.effective(self._seed, mask, ctx=ctx)
            E = self._E
            for live, work in self._pairs:     # dynamic params / inputs, live -> work
                if mask is None:
                    ctx.launch(_LATCH_ALL[work.dtype], E, [live, work])
                else:
                    ctx.launch(_LATCH_MASKED[work.dtype], E, [live, mask, work])
            if commit is not None and mask is not None:
                commit.enqueue_hold(ctx, mask)
        enqueues = self._enqueues
        for j in self._seg_slots[i]:
            enqueues[j](ctx)
        if i == len(self._segments) - 1:
            if commit is not None:
                commit.enqueue_commit(ctx, mask)
            result = self._result
            if result.generation is not None:
                seeds._write_generation(result.generation, mask, ctx)

    def _body(self, advance: bool, mask) -> "tuple[object, object]":
        """``(body, self-graph key)`` of one call; validates ``advance`` / ``mask``
        host-side, before any launch. A segmented Program's body is one part per
        segment (the runner self-graphs the parts that are not HOST)."""
        if mask is not None:
            if not advance:
                raise ValueError(
                    "mask= selects the envs whose epoch advances; it requires "
                    "advance=True (without it every env is regenerated identically)")
            self._seeds.check_mask(mask)
        key = (True, 0 if mask is None else int(mask.ptr)) if advance else None
        if len(self._segments) > 1:
            kw = {"advance": True, "mask": mask} if advance else {}
            return _SegmentedBody(
                _BodyPart(functools.partial(self.enqueue_segment, seg.index, **kw),
                          host=seg.host)
                for seg in self._segments), key
        if not advance:
            return self.enqueue, None
        return functools.partial(self.enqueue, advance=True, mask=mask), key

    def _not_capturable(self, stage_ids) -> NotCapturableError:
        return NotCapturableError(str(self._binding.name), stage_ids=stage_ids)

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
            policy: ``"auto"`` or ``"eager"`` (never builds the Program's own CUDA
                graph).

        Raises:
            ValueError: an invalid ``mask``.
            track_gen.exec.NotCapturableError: recording a Program with HOST stages
                (also one built with ``prepare=False``), before any launch.
            track_gen.exec.CaptureError: recording a Program that was never prepared.
        """
        body, key = self._body(advance, mask)
        ctx = None
        if self._host_stage_ids:
            ctx = self._runner.context(stream)
            if _host_blocked(ctx):
                raise self._not_capturable(self._host_stage_ids)
        if not self._runner.prepared:
            ctx = ctx if ctx is not None else self._runner.context(stream)
            if ctx.capturing:
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
