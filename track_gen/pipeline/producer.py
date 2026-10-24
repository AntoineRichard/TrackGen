"""The :class:`Producer` protocol: what a ``Course`` drives."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    import warp as wp

    from ..exec.context import LaunchContext
    from ..results import Layout

__all__ = ["Producer"]


class Producer(Protocol):
    """The generator surface ``track_gen.course.Course`` drives (documentation only:
    the library duck-types it and never ``isinstance``-checks it).

    ``TrackGenerator`` / ``GateGenerator`` (through their ``program``) and a built
    :class:`~track_gen.pipeline.Program` implement it; a third-party producer (e.g. a
    mixed batch of several generators writing views of one result) implements the
    required members, and the optional ones it needs.

    Required:

    - ``result``: the result every generation writes (a ``Track``, a ``GateSequence`` or
      another result the components read), the same instance with the same array
      addresses for the producer's lifetime; its ``valid`` (when present) is
      ``StepResult.valid``.
    - ``layout``: the result's :class:`~track_gen.results.Layout` (``num_envs``, the
      stride, the device).
    - ``enqueue(ctx, *, advance=False, mask=None)``: one generation on ``ctx`` (a
      :class:`~track_gen.exec.LaunchContext`), allocation-free and host-sync free. With
      ``advance``: bump each env's epoch first (every env, or the masked ones). With
      ``mask`` (``[E]`` int32, read on device): the unmasked envs' result rows end the
      generation bit for bit as they were (a masked commit), and an all-zero mask is an
      exact no-op. It may run inside a CUDA conditional-node body (the course's masked
      regeneration): kernels, fills and device-to-device copies on ``ctx``'s stream only.

    Optional (read when present):

    - ``seeds``: its :class:`~track_gen.seeds.SeedState` (``course.seeds``;
      ``Course.generate(seeds=)`` sets owned seeds through it).
    - ``capture_class``: a :class:`~track_gen.plugin.CaptureClass` (default
      ``STATIC``); ``HOST`` runs eagerly outside a capture and is refused inside one
      (``NotCapturableError``).
    - ``prepared`` and ``prepare()``: an unprepared producer is prepared first (its own
      warm-up, loading every module ``enqueue`` uses). Without them, the producer must
      have loaded every module its ``enqueue`` uses before it reaches a Course (e.g. one
      real ``enqueue`` at its construction).
    - ``default_stream``: the course's default stream when ``Course(..., stream=)`` is
      not given.
    - ``capacity`` and ``bounds`` (``name -> number`` mappings): the static capacities
      (``max_loop_length``) and declared bounds (``size_hi``) a plan would give, read by
      :meth:`ComponentContext.for_producer <track_gen.tools.ComponentContext.for_producer>`
      before the plan.
    - ``inputs`` and ``input_info``: its persistent device inputs (listed, never bound by
      the course: write them in place).
    """

    @property
    def result(self) -> object:
        """The result that every generation writes (the same instance, with the same
        array addresses, for the producer's lifetime)."""
        ...

    @property
    def layout(self) -> "Layout":
        """The result's :class:`~track_gen.results.Layout`."""
        ...

    def enqueue(self, ctx: "LaunchContext", *, advance: bool = False,
                mask: "wp.array | None" = None) -> object:
        """Enqueue one generation on ``ctx``: allocation-free and host-sync free; with
        ``advance``, bump the epochs first; with ``mask``, leave the unmasked envs' rows
        bit for bit as they were (see the class docstring)."""
        ...

