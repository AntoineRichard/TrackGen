"""Errors raised by the execution layer (:mod:`track_gen.exec`).

All of them derive from :class:`CaptureError`, itself a ``RuntimeError`` (Warp
reports capture failures as ``RuntimeError`` too), so callers can catch the
whole family at once. :class:`RebindAfterCaptureWarning` is a warning.
"""
from __future__ import annotations

__all__ = ["CaptureError", "NotCapturableError",
           "RebindAfterCaptureWarning"]


class RebindAfterCaptureWarning(UserWarning):
    """A bound input array was replaced after a hot method that reads it was recorded
    into a capture.

    The recorded graph keeps reading the arrays that were bound when it was recorded, so
    after a rebind its replays silently read the old buffers. Emitted (once per bind
    call) by ``Course.bind``, ``Component.bind`` and the utilities' ``bind`` /
    ``bind_inputs`` / ``bind_window`` when a pointer changes after their ``query()`` /
    ``update()`` ran while its stream (or a CPU graph capture) was capturing; rebinding
    the same arrays is silent. The documented recipe "rebind, then record again" also
    warns once: filter this category when you re-record on purpose::

        import warnings

        import track_gen.exec

        warnings.simplefilter("ignore", track_gen.exec.RebindAfterCaptureWarning)
    """


class CaptureError(RuntimeError):
    """A call was made in a capture state it does not support.

    For example, constructing a generator (which runs one real warm-up pass)
    while the resolved stream is being captured into a CUDA graph.
    """


class NotCapturableError(CaptureError):
    """A non-capturable producer was asked to record into a caller's capture (a CUDA
    graph capture, or a CPU graph capture: ``wp.ScopedCapture("cpu")``).

    Raised BEFORE any device work is issued, so the caller's capture is left
    untouched. Outside a capture the same producer runs: its HOST stages eagerly, the
    rest of a pipeline from its own graphs (segmented execution, see
    :class:`~track_gen.exec.GraphRunner`).

    Attributes:
        name: The producer's name (e.g. the generator name).
        stage_ids: The ids of the HOST stages that make it non-capturable (``()``
            when unknown, e.g. a bare ``GraphRunner(capturable=False)``).
    """

    def __init__(self, name: str = "", detail: str = "", *,
                 stage_ids=()) -> None:
        self.name = name
        self.stage_ids = tuple(str(s) for s in stage_ids)
        what = repr(name) if name else "this producer"
        if self.stage_ids:
            msg = (
                f"{what} cannot be recorded into a graph capture: HOST stage(s) "
                f"{list(self.stage_ids)} read device data back or run host control "
                "flow, so they run eagerly (outside a capture the rest still replays "
                "from graphs). Call it outside the capture, or use STATIC / BOUNDED "
                "stages (PipelineSpec(require_caller_capturable=True) rejects HOST "
                "stages at build)."
            )
        else:
            msg = (
                f"{what} cannot be recorded into a graph capture: its generator is "
                "registered with capturable=False (it may host-synchronize, read back "
                "or allocate per call). Call it outside the capture (it then runs "
                "eagerly) or use a capturable generator."
            )
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)
