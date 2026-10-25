"""PyTorch interop for recording track_gen work into ``torch.cuda.graph`` captures.

A plain ``torch.cuda.graph(g)`` capture is invisible to Warp: library calls made
inside it are still recorded into ``g`` (they adopt torch's capture stream, see
:mod:`track_gen.exec`), but two Warp features need the capture to be REGISTERED
with Warp:

- lifetime anchoring: a producer recorded into a registered capture is attached to
  the Warp graph object of that capture;
- device-side conditional skips (:meth:`track_gen.exec.LaunchContext.skip_unless`,
  used by ``Course.regenerate(mask=...)`` and by every window of an iterative
  ``BOUNDED`` stage such as the ``repulsive`` generator), opt-in with
  ``conditional=True``: otherwise the guarded body is recorded unconditionally
  (always correct, never skipped). A hand-made Warp registration of torch's capture
  (``wp.capture_begin(stream=..., external=True)`` on torch's capture stream) gets no
  conditional node either: opt in through :func:`registered`. (Your own
  ``wp.ScopedCapture`` of torch's current stream, when torch is not capturing, opts in
  with :func:`track_gen.exec.conditional`.)

:func:`registered` registers torch's capture with Warp for the duration of a
``with`` block; :func:`retain` ties Python objects to the torch graph's lifetime (here
``course`` is a bound ``Course`` and ``done`` your ``[E]`` int32 episode-end mask):

.. snippet: cuda

.. snippet: setup

    import warp as wp
    from track_gen.course import Course
    from track_gen.pipeline import presets

    E = 64
    course = Course.track(presets.track(num_envs=E, device="cuda:0"), checkpoint_spacing=0.6)
    course.bind(position=wp.zeros(E, dtype=wp.vec3f, device="cuda:0"))
    done = wp.zeros(E, dtype=wp.int32, device="cuda:0")

.. code-block:: python

    import torch
    import track_gen.interop.torch

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with track_gen.interop.torch.registered(graph=g):
            course.step()
            course.regenerate(mask=done, advance=True)
    track_gen.interop.torch.retain(g, course)   # keep the course's buffers alive with g

.. snippet: after

    g.replay()
    torch.cuda.synchronize()

This module imports torch lazily (only when a function is called).
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import warp as wp

from ..exec import _compat
from ..exec.context import _torch_stream_as_warp
from ..exec.errors import CaptureError

__all__ = ["registered", "retain"]

_OWNERS_ATTR = "_track_gen_owners"

_MODES = {"global": "GLOBAL", "thread_local": "THREAD_LOCAL", "relaxed": "RELAXED"}


def _capture_mode(mode) -> "wp.CaptureMode":
    """``None`` -> GLOBAL (``torch.cuda.graph``'s default ``capture_error_mode``);
    a torch mode string or a ``wp.CaptureMode`` otherwise."""
    if mode is None:
        return wp.CaptureMode.GLOBAL
    if isinstance(mode, str):
        try:
            return getattr(wp.CaptureMode, _MODES[mode])
        except KeyError:
            raise ValueError(
                f"capture_mode must be one of {sorted(_MODES)} or a wp.CaptureMode, "
                f"got {mode!r}") from None
    return wp.CaptureMode(mode)


def _warp_stream(stream) -> "wp.Stream":
    """The long-lived ``wp.Stream`` for ``stream`` (``None``: torch's current stream
    on the current CUDA device; a ``torch.cuda.Stream``; or a ``wp.Stream``)."""
    if isinstance(stream, wp.Stream):
        return stream
    import torch  # noqa: PLC0415 - lazy: never at track_gen import time

    ts = torch.cuda.current_stream() if stream is None else stream
    index = getattr(ts, "device_index", None)
    if index is None:
        index = ts.device.index
    dev = wp.get_device(f"cuda:{int(index)}")
    return _torch_stream_as_warp(dev, ts)


@contextlib.contextmanager
def registered(stream=None, *, graph=None, capture_mode=None,
               conditional: bool = False) -> Iterator["wp.Stream"]:
    """Register an open ``torch.cuda.graph`` capture with Warp for this block.

    Use INSIDE ``torch.cuda.graph(g)``. Resolves the capture stream to the
    library's cached Warp wrapper of that handle (never a throwaway one: dropping a
    ``wp.stream_from_torch`` wrapper unregisters the handle process-wide), then:

    - if Warp already has a capture registered on that handle (e.g. an enclosing
      ``registered()`` or Warp ``ScopedCapture``), yields that registered stream and
      does nothing else (``graph=`` and ``conditional=`` are then ignored: the
      enclosing registration owns the Warp-side graph object);
    - if the stream only JOINED a capture Warp registered on another stream (a
      side stream forked from it), yields the stream and registers nothing (a
      second registration of the same capture would break the first one's
      teardown); the conditional skip then does not apply on this stream;
    - otherwise opens ``wp.ScopedCapture(stream=s, external=True)`` on it (which does
      not begin a new CUDA capture, only registers torch's) and closes it with the
      SAME stream object at the end of the block (Warp looks an open capture up by
      stream object identity).

    Inside the block, library calls adopt the stream as usual and additionally get
    lifetime anchoring and, with ``conditional=True``, conditional skips
    (``LaunchContext.skip_unless``, CUDA 12.4+). The registration ends with the
    block; the torch capture continues.

    ``conditional=True`` caveat: a conditional node (``wp.capture_if``) ends and
    restarts the CUDA capture of the stream, which gives it a new capture id, and
    torch's caching allocator recognises its graph's capture by the id taken at
    ``capture_begin``. After the first conditional node, torch therefore no longer
    routes the allocations of later torch ops in the same ``torch.cuda.graph`` to the
    graph's private pool (they either fail to capture or take blocks from the general
    pool that eager code may reuse while the graph still writes them), and other torch
    features that look the capture up by its id may fail. The first conditional node
    comes with a masked ``Course.regenerate(mask=...)`` or with ANY generation that
    runs a ``BOUNDED`` stage (the ``repulsive`` generator, a plugin's
    ``ctx.bounded_loop`` with ``active=``: one conditional node per window), not only
    with masked regeneration. So with ``conditional=True``, record those calls AFTER
    every torch op of the graph that allocates (or preallocate their outputs and use
    ``out=`` / in-place ops), up to the end of the ``torch.cuda.graph`` block, not
    just of this one. Ending the capture also fails while a side stream that joined it
    has not been joined back. The graph also cannot be used as a CUDA child graph.
    Without ``conditional=True`` the masked regeneration and every ``BOUNDED`` window
    are recorded unconditionally (exact no-ops when skipped would be, costing their
    full work per replay) and torch's capture is never touched.

    Args:
        stream: ``None`` (torch's current stream on the current CUDA device, which
            is the capture stream inside ``torch.cuda.graph``), a
            ``torch.cuda.Stream``, or a ``wp.Stream``.
        graph: optional ``torch.cuda.CUDAGraph`` being captured. When given, the
            Warp-side graph object of the registration (it holds the producers
            anchored during the block and the Warp modules the recorded kernels
            use) is :func:`retain`-ed on it, so they live as long as ``graph``.
            Without it they are released when the block exits: keep the producers
            alive yourself, e.g. ``retain(g, course)``.
        capture_mode: the mode torch began the capture with
            (``torch.cuda.graph(..., capture_error_mode=...)``): ``"global"``
            (default, torch's default), ``"thread_local"``, ``"relaxed"``, or a
            ``wp.CaptureMode``.
        conditional: allow ``LaunchContext.skip_unless`` to record CUDA conditional
            nodes into this registration (a masked ``Course.regenerate``, every
            window of a ``BOUNDED`` stage; default ``False``; see the caveat
            above).

    Yields:
        The registered ``wp.Stream`` (pass it as ``stream=`` if you like; library
        calls inside the block resolve to it anyway).

    Raises:
        track_gen.exec.CaptureError: the stream is not capturing (not inside
            ``torch.cuda.graph``).
    """
    ws = _warp_stream(stream)
    if not ws.is_capturing:
        raise CaptureError(
            "interop.torch.registered() must be used inside torch.cuda.graph(...): the "
            f"stream {hex(int(ws.cuda_stream or 0))} on {ws.device} is not capturing.")
    key = _compat.registered_stream(ws)
    if key is not None:
        yield key
        return
    if _compat.registered_graph(ws) is not None:
        # ws joined a capture Warp registered on another stream (same capture id):
        # registering it again would overwrite, then delete, that registration.
        yield ws
        return
    mode = _capture_mode(capture_mode)
    with wp.ScopedCapture(device=ws.device, stream=ws, external=True,
                          capture_mode=mode) as cap:
        _compat.set_conditional_allowed(ws, bool(conditional))
        yield ws
    if graph is not None and cap.graph is not None:
        retain(graph, cap.graph)


def retain(torch_graph, *owners) -> None:
    """Keep ``owners`` alive for as long as ``torch_graph`` is.

    A CUDA graph stores raw device pointers; it does not keep the buffers it reads
    and writes alive. Replaying ``torch_graph`` after a producer (``Course``,
    ``TrackGenerator``, ...) recorded into it was garbage-collected reads freed
    memory. ``retain(g, course)`` stores strong references in an attribute of the
    torch graph object (``g._track_gen_owners``), so they are released together
    with ``g``. Idempotent per object; ``None`` owners are ignored.

    Raises:
        TypeError: ``torch_graph`` cannot carry attributes (no ``__dict__``).
    """
    try:
        held = vars(torch_graph).setdefault(_OWNERS_ATTR, [])
    except TypeError:
        raise TypeError(
            f"retain(): cannot attach owners to a {type(torch_graph).__name__} "
            "(no __dict__); keep the producers alive yourself") from None
    for owner in owners:
        if owner is not None and not any(o is owner for o in held):
            held.append(owner)
