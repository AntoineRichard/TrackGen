"""Every library launch, copy and fill is stream-explicit (CPU probe; roadmap PR2).

A library call can only be recorded into a caller's CUDA graph if all of its device work
lands on the stream being captured. Library code therefore emits work through a
``track_gen.exec.LaunchContext``, which passes ``stream=`` to every ``wp.launch`` /
``wp.copy`` and never uses ``wp.array.zero_()`` / ``fill_()`` (their memset targets Warp's
current stream). CPU has no streams, so the probe checks the *call shape*: any
``wp.launch`` / ``wp.copy`` issued from a ``track_gen`` module (or from the toy
out-of-tree plugin, ``tests._plugin_toys``) without a ``stream=`` keyword, and any
``zero_()`` / ``fill_()`` issued from one, is a failure. The drivers are shared with
``test_no_host_sync`` and the probe covers their construction too.
"""
from __future__ import annotations

import contextlib
import sys

import pytest
import warp as wp

from tests.test_no_host_sync import DRIVERS
from track_gen.exec import LaunchContext, resolve, resolve_stream
from track_gen.ops import fill


def _caller():
    f = sys._getframe(2)
    return f.f_globals.get("__name__", ""), f.f_code.co_name, f.f_lineno


# Library modules plus the out-of-tree toy plugin (tests._plugin_toys, accepts_ctx=True),
# which is held to the same stream-explicit contract.
_PROBED = ("track_gen", "tests._plugin_toys")


@contextlib.contextmanager
def _stream_probe():
    bad: list[str] = []
    real_launch, real_copy = wp.launch, wp.copy
    real_zero, real_fill = wp.array.zero_, wp.array.fill_

    def launch(*args, **kwargs):
        mod, fn, line = _caller()
        if mod.startswith(_PROBED) and "stream" not in kwargs:
            bad.append(f"wp.launch without stream= from {mod}.{fn}:{line}")
        return real_launch(*args, **kwargs)

    def copy(*args, **kwargs):
        mod, fn, line = _caller()
        if mod.startswith(_PROBED) and "stream" not in kwargs:
            bad.append(f"wp.copy without stream= from {mod}.{fn}:{line}")
        return real_copy(*args, **kwargs)

    def zero_(self):
        mod, fn, line = _caller()
        if mod.startswith(_PROBED):
            bad.append(f"array.zero_() from {mod}.{fn}:{line}")
        return real_zero(self)

    def fill_(self, value):
        mod, fn, line = _caller()
        if mod.startswith(_PROBED):
            bad.append(f"array.fill_() from {mod}.{fn}:{line}")
        return real_fill(self, value)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wp, "launch", launch)
        mp.setattr(wp, "copy", copy)
        mp.setattr(wp.array, "zero_", zero_)
        mp.setattr(wp.array, "fill_", fill_)
        yield bad


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_every_library_launch_is_stream_explicit(name):
    with _stream_probe() as bad:
        hot = DRIVERS[name]()      # construction + first generate are probed too
        hot()
    assert bad == [], "\n".join(sorted(set(bad)))


# ---------------------------------------------------------------------------------------
# LaunchContext / resolve_stream / ops.fill unit behaviour (CPU).
# ---------------------------------------------------------------------------------------

def test_cpu_context_has_no_stream():
    ctx = LaunchContext.for_call("cpu")
    assert ctx.device.alias == "cpu" and ctx.stream is None and not ctx.capturing
    assert resolve_stream("cpu") is None


def test_ambient_scope_is_reused_and_restored():
    outer = LaunchContext.for_call("cpu")
    assert LaunchContext.for_call("cpu") is not outer
    with outer.scope():
        assert LaunchContext.for_call("cpu") is outer
        assert resolve(None, "cpu") is outer
        inner = LaunchContext("cpu", None)
        with inner.scope():
            assert LaunchContext.for_call("cpu") is inner
        assert LaunchContext.for_call("cpu") is outer
    assert LaunchContext.for_call("cpu") is not outer


def test_explicit_ctx_wins_over_ambient():
    amb, mine = LaunchContext("cpu", None), LaunchContext("cpu", None)
    with amb.scope():
        assert resolve(mine, "cpu") is mine


def test_resolve_stream_rejects_bad_values():
    with pytest.raises(TypeError):
        resolve_stream("cpu", stream=3)
    with pytest.raises(ValueError):
        resolve_stream("cpu", stream="cuda")


@pytest.mark.parametrize("dtype,value,expect", [
    (wp.int32, 7, 7), (wp.uint32, 9, 9), (wp.float32, 1.5, 1.5),
    (wp.vec2f, (1.0, 2.0), (1.0, 2.0)), (wp.vec3f, 0.0, (0.0, 0.0, 0.0)),
])
def test_ops_fill(dtype, value, expect):
    import numpy as np
    arr = wp.empty(4, dtype=dtype, device="cpu")
    fill(LaunchContext.for_call("cpu"), arr, value)
    np.testing.assert_allclose(arr.numpy().reshape(4, -1)[0], np.atleast_1d(expect))


def test_ops_fill_rejects_unsupported_dtype():
    with pytest.raises(TypeError):
        fill(LaunchContext.for_call("cpu"), wp.empty(2, dtype=wp.float64, device="cpu"), 0)
