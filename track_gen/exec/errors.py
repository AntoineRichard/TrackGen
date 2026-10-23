"""Errors raised by the execution layer (:mod:`track_gen.exec`).

All of them derive from :class:`CaptureError`, itself a ``RuntimeError`` (Warp
reports capture failures as ``RuntimeError`` too), so callers can catch the
whole family at once.
"""
from __future__ import annotations

__all__ = ["CaptureError", "NotCapturableError", "LegacyPluginStreamError"]


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


class LegacyPluginStreamError(CaptureError):
    """A legacy-ABI plugin would launch on the wrong stream under capture.

    Legacy generator plugins (registered without ``accepts_ctx=True``) issue
    their own ``wp.launch`` calls on Warp's CURRENT stream. Recording one into
    a caller's capture is therefore only correct when the capture stream IS
    Warp's current stream.
    """

    def __init__(self, name: str = "", detail: str = "") -> None:
        self.name = name
        what = repr(name) if name else "this producer"
        msg = (
            f"{what} uses a legacy plugin (no accepts_ctx=True) that launches on "
            "Warp's current stream, but the call resolved a different, capturing "
            "stream. Make the capture stream Warp's current stream for the "
            "duration of the capture: capture Warp's current stream (a plain "
            "`wp.ScopedCapture(device)`), or, for torch.cuda.graph, pass it an "
            "explicit capture stream s and make that stream current first: "
            "`with wp.ScopedStream(wp.stream_from_torch(s)): with "
            "torch.cuda.graph(g, stream=s): ...` (a plain `torch.cuda.graph(g)` "
            "captures on torch's internal stream, which cannot be made current "
            "beforehand). Porting the plugin to accept ctx= removes the "
            "restriction."
        )
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)
