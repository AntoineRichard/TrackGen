"""CUDA-only: the component protocol and the PR14 utility changes on ``cuda:0`` (roadmap
PR14; design section 9.4 and the S3 gate of section 15).

What this module proves on ``cuda:0`` (the CPU proofs are ``tests/test_components.py``
and the shipped CPU suites; the CPU lane only collects this module):

1. **Old-vs-new kernel identity for VALID envs, per utility.** The frozen base copies of
   the twelve kernels PR14 edits (``tests/_pr14_base_kernels.py``, verbatim from
   9267388) launched on the same inputs as each utility's public call give byte-identical
   rows for every valid env: segments and SDF collision (bake included), discs on gate
   posts (the base scanned ``2 * max_gates`` posts, the new ``DiscChecker`` is given the
   real post count and ``valid``), props, checkpoints, progress and the cold / warm
   localizer along a trajectory with masked advancing regenerations (base: explicit
   resets; new: explicit resets plus the self-reset), the heightfield, ``curvature`` /
   ``speed_profile`` (allocating and ``out=`` forms) and the course line. This is the
   CUDA codegen of the edited kernels: the CPU goldens say nothing about it.
2. **The eager-vs-captured trajectory twin of an RL loop**, track and gates modes, in a
   Warp ``ScopedCapture`` on a side stream and under ``torch.cuda.graph`` with
   ``track_gen.interop.torch.registered()``: ONE caller graph ``[physics stub ->
   course.step() -> done mask (a lap, a per-env step budget, or ``1 -
   StepResult.valid``) -> course.regenerate(mask=done, advance=True)]`` replayed 200
   times with no host involvement equals an eager twin (``step()`` and, when an env is
   done, ``enqueue_generate(mask=done, advance=True)``) EVERY step, bitwise: the result
   and its ``generation``, epochs, the checkpoint set (with its owned ``valid`` /
   ``generation``), sampler diagnostics, heightfield, posts, course line, the progress /
   localizer runtime state (``seen`` generations included), the step outputs and
   ``StepResult.valid``. Only the StepResult outputs ``events`` / ``contacts`` /
   ``frame`` are poisoned before each replay (every env is written by ``step()``);
   producer state (``course.result``, ``StepResult.valid`` which aliases it, the gates-mode
   checkpoint views) and the sub-tools' derived buffers are compared, never poisoned
   (the commit restores unmasked rows and writes ``generation`` for masked envs only, so a
   poison would persist and trigger self-resets). Variants: track (segments collision +
   heightfield, the DEFAULT ``max_checkpoints``: the plan bound, 58 at spacing 0.6),
   track with two envs driven invalid (size 8: truncation; the sentinel table checked
   after every replay on every env invalid at that step, which the done rule gives a
   new course), gates with posts + a warm localizer, gates with frames.
3. **Invalid-env sentinels under capture** (an invalid env reads exactly like a poisoned
   one): every utility built on K (half the envs failed by the commit's own kernel at
   ``poison=0``: geometry kept, ``valid = 0``) and on P (the same envs committed at
   ``poison=1``), recorded in ONE caller graph and replayed along a trajectory, writes
   byte-identical outputs on K and P, equal to an eager run on K, and the table values
   on the failed envs.
4. **Generation-aware self-reset in a caller graph**: ``[move -> tracker.update() ->
   localizer.query() -> device done schedule -> generate(advance=True, mask=done) ->
   sample() / refresh()]`` recorded WITHOUT any reset equals an eager twin that resets
   explicitly (today's contract), every step, in track mode (sampler-owned set
   ``generation``) and gates mode (``from_gates`` aliases, the course line's owned
   ``generation``).
5. ``curvature(out=, scratch=)`` / ``speed_profile(kappa=, out=)`` recorded with an
   advancing generation: poisoned outputs recomputed equal to the eager allocating forms,
   on a ``Track``, a ``Centerline`` and a ``MinimalCenterline`` (no ``valid``).
6. ``check_component`` passes for every built-in adapter on ``cuda:0`` (``capture.caller``
   in the Warp mode, and in the torch modes when torch with CUDA is installed;
   ``capture.self_reset`` for every stateful adapter), and (roadmap PR15) for the correct
   toys whose result holds an array that only ``on_generate`` writes (a duck-typed
   per-generation component, a per-step one exposing a per-generation table): on an
   all-zero ``done`` the replay's conditional node skips ``on_generate`` and the kit's
   eager twin skips it too.
7. ``RebindAfterCaptureWarning`` with real captures (Warp and torch): rebinding other
   arrays after a hot method was recorded warns once at the caller's line (``Course.bind``,
   a utility's ``bind``, a component's ``bind``), the same arrays and eager-only use are
   silent; and the rebind probe's host cost on ``Course.step`` (printed; must stay below
   2 % of the step's host time, else A.7 falls back to docs only).

Method (as in tests/test_inplace_regen_cuda.py and tests/test_results_cuda.py; the helpers
are :mod:`track_gen.testing.capture`'s): record on a NON-current side stream, with the
Warp mempool disabled (an allocation inside a capture fails loudly) and
``wp.capture_launch`` replaced by a tripwire; poison with FINITE, dtype-aware sentinels
(outputs are NaN-padded, so a NaN poison would hide a missed write); warm every recorded
path up outside the capture (no module loads while capturing); every array is allocated
with an explicit device. Comparisons are BYTE comparisons (NaN payloads and signed zeros
included), with a value diff in the message.
"""
from __future__ import annotations

import contextlib
import gc
import math
import time
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

from tests import _component_toys as _toys  # noqa: E402
from tests._pr14_base_kernels import BASE_KERNELS as _BASE  # noqa: E402
from tests._results_helpers import MinimalCenterline  # noqa: E402
from track_gen import GateGenConfig, PerEnvSeededRNG, TrackGenConfig, tools  # noqa: E402
from track_gen._src.collision_frames import FrameChecker  # noqa: E402
from track_gen._src.collision_sdf import _box_query_sdf_k  # noqa: E402
from track_gen._src.course_line import CourseLine, _line_frames_k  # noqa: E402
from track_gen._src.gate_generator import GateGenerator  # noqa: E402
from track_gen._src.heightfield import _height_bake_k  # noqa: E402
from track_gen._src.localize import _localize_reset_k  # noqa: E402
from track_gen._src.progress import _progress_reset_k  # noqa: E402
from track_gen._src.props import _place_props_k  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen.checkpoints import CheckpointSampler, CheckpointSet  # noqa: E402
from track_gen.collision import CollisionChecker, DiscChecker  # noqa: E402
from track_gen.course import Course, CourseConfig  # noqa: E402
from track_gen.exec import LaunchContext, RebindAfterCaptureWarning  # noqa: E402
from track_gen.exec import context as exec_context  # noqa: E402
from track_gen.heightfield import HeightFieldBaker  # noqa: E402
from track_gen.localize import TrackLocalizer, curvature, speed_profile  # noqa: E402
from track_gen.progress import ProgressTracker  # noqa: E402
from track_gen.props import PropSampler  # noqa: E402
from track_gen.stages.commit import _commit_envs  # noqa: E402
from track_gen.testing import check_component  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402
from track_gen.testing.components import _builtin_components  # noqa: E402
from track_gen.tools import ComponentContext  # noqa: E402

DEV = "cuda:0"
E = 8
_ONES = np.ones(E, np.int32)
#: Masks of the masked advancing regenerations (neighbours differ: a write that leaks
#: across an env boundary shows up).
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)
_M2 = np.array([0, 1, 1, 0, 1, 0, 0, 1], np.int32)
#: The envs K / P fail (half the batch, neighbours differ).
_FAIL = np.array([0, 1, 0, 1, 1, 0, 1, 0], np.int32)
#: Elevation variants of the track batches the identity tests read.
_Z = {"flat": {}, "uniform z": dict(z_profile="uniform", z_min=0.0, z_max=0.6)}


# ---------------------------------------------------------------------------------------
# Test kernels (one Warp module: _load_test_kernels() loads it outside any capture)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _move_agents_k(
    center: wp.array(dtype=wp.vec3f),
    count: wp.array(dtype=wp.int32),
    n_max: int,
    speed: wp.array(dtype=wp.int32),
    offset: wp.array(dtype=wp.float32),
    t: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3f),
    orient: wp.array(dtype=wp.quatf),
):
    # Stub physics: each agent advances `speed[e]` samples along its env's CURRENT
    # line (a regenerated env is followed on its new line from the next step), at the
    # midpoint of two consecutive samples, `offset[e]` to the side (plan view), its box
    # yawed along the motion.
    e = wp.tid()
    k = t[e] + speed[e]
    t[e] = k
    steps[e] = steps[e] + 1
    n = count[e]
    if n > n_max:
        n = n_max
    if n >= 2:
        i = ((k % n) + n) % n
        j = i + 1
        if j == n:
            j = 0
        base_i = e * n_max
        a = center[base_i + i]
        b = center[base_i + j]
        d = b - a
        L = wp.sqrt(d[0] * d[0] + d[1] * d[1])
        side = wp.vec3f(0.0, 0.0, 0.0)
        if L > 1.0e-9:
            side = wp.vec3f(-d[1] / L, d[0] / L, 0.0)
        pos[e] = 0.5 * (a + b) + offset[e] * side
        orient[e] = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), wp.atan2(d[1], d[0]))
    else:
        pos[e] = wp.vec3f(0.0, 0.0, 0.0)
        orient[e] = wp.quatf(0.0, 0.0, 0.0, 1.0)


@wp.kernel
def _done_k(
    laps: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    period: wp.array(dtype=wp.int32),
    speed: wp.array(dtype=wp.int32),
    t: wp.array(dtype=wp.int32),
    steps: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
):
    # The documented RL recipe on device: done on a completed lap, on a per-env step
    # budget, and for an invalid env (done |= 1 - valid: it gets a NEW course). A done
    # env's agent restarts two moves before the start of its (new) line.
    e = wp.tid()
    d = int(0)
    if laps[e] >= 1 or steps[e] >= period[e] or valid[e] == 0:
        d = 1
        t[e] = -2 * speed[e]
        steps[e] = 0
    done[e] = d


@wp.kernel
def _schedule_k(counter: wp.array(dtype=wp.int32), period: int,
                mask: wp.array(dtype=wp.int32)):
    # A deterministic device-side done schedule from a replay counter.
    e = wp.tid()
    k = counter[0]
    m = int(0)
    if (k + 3 * e) % period == period - 1:
        m = 1
    mask[e] = m


@wp.kernel
def _bump_k(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


def _load_test_kernels() -> None:
    """Load this module's kernels on ``DEV`` outside any capture (scratch buffers)."""
    i = wp.zeros(E, dtype=wp.int32, device=DEV)
    f = wp.zeros(E, dtype=wp.float32, device=DEV)
    v = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    q = wp.zeros(E, dtype=wp.quatf, device=DEV)
    wp.launch(_move_agents_k, E, inputs=[v, i, 1, i, f, i, i, v, q], device=DEV)
    wp.launch(_done_k, E, inputs=[i, i, i, i, i, i, i], device=DEV)
    wp.launch(_schedule_k, E, inputs=[i, 3, i], device=DEV)
    wp.launch(_bump_k, 1, inputs=[i], device=DEV)
    wp.synchronize()


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _dev(values, dtype) -> "wp.array":
    """An array on ``DEV`` from host values (uploaded on Warp's current stream, then
    synchronized: side-stream readers need it done)."""
    a = wp.array(np.asarray(values), dtype=dtype, device=DEV)
    wp.synchronize()
    return a


def _i32(values) -> "wp.array":
    return _dev(np.asarray(values, np.int32), wp.int32)


def _write(dst: "wp.array", values) -> None:
    """Write host values into a device buffer, outside any capture (read at replay)."""
    wp.copy(dst, _dev(np.asarray(values).astype(_np_dtype(dst)), dst.dtype))
    wp.synchronize()


def _np_dtype(arr) -> type:
    return np.int32 if arr.dtype is wp.int32 else np.float32


def _host(x) -> np.ndarray:
    """A contiguous host COPY (on a CPU device ``numpy()`` is a view of the live array)."""
    return np.array(x.numpy() if isinstance(x, wp.array) else x, copy=True, order="C")


def _arrays(obj, prefix: str) -> dict:
    """``"<prefix>.<field>" -> wp.array`` for every array of a result dataclass."""
    return {f"{prefix}.{k}": a for k, a in tg_capture.result_arrays(obj).items()}


def _snapshot(d: dict) -> dict:
    return {k: np.array(_host(v), copy=True) for k, v in d.items()}


def _assert_bytes(got, ref, label: str) -> None:
    """Byte equality of two arrays (NaN payloads, signed zeros), a value diff on failure."""
    a, b = _host(got), _host(ref)
    assert a.shape == b.shape and a.dtype == b.dtype, \
        (label, a.shape, b.shape, a.dtype, b.dtype)
    if a.tobytes() != b.tobytes():
        np.testing.assert_array_equal(a, b, err_msg=label)
        raise AssertionError(f"{label}: equal values, different bytes (NaN payload or "
                             f"signed zero)")


def _assert_state(got: dict, ref: dict, label: str) -> None:
    assert got.keys() == ref.keys(), f"{label}: keys differ {sorted(set(got) ^ set(ref))}"
    for name in ref:
        _assert_bytes(got[name], ref[name], f"{label}: {name}")


def _rows(arr) -> np.ndarray:
    """``[E, k, ...]`` host copy of an env-major flat array."""
    h = _host(arr)
    return h.reshape((E, h.shape[0] // E) + h.shape[1:])


def _assert_valid_rows(new, old, valid: np.ndarray, label: str) -> None:
    """Byte equality of the rows of the valid envs (env-major flat arrays)."""
    n, o = _rows(new), _rows(old)
    assert n.shape == o.shape and n.dtype == o.dtype, (label, n.shape, o.shape)
    for e in np.flatnonzero(valid):
        _assert_bytes(n[e], o[e], f"{label}: env {e}")


def _side_stream() -> "wp.Stream":
    return tg_capture.side_stream(DEV)


def _recording(stream):
    """The caller's Warp capture on ``stream``: mempool disabled, capture_launch
    forbidden (:func:`track_gen.testing.capture.recording`)."""
    return tg_capture.recording(DEV, stream)


def _replay(graph, stream) -> None:
    wp.capture_launch(graph, stream=stream)
    wp.synchronize()


def _poison(arrays) -> None:
    for a in arrays:
        tg_capture.poison(a)
    wp.synchronize()


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch built without CUDA")
    return torch


def _caller_graph(mode: str, record, owners=()):
    """Record ``record(kernel_stream, library_stream)`` into ONE caller graph and return
    ``(replay, result)``.

    - ``"warp"``: a Warp ``ScopedCapture`` on a side stream (both streams are it);
    - ``"torch"``: ``torch.cuda.graph`` with ``interop.torch.registered()``: the test's
      kernels go to the registered stream, library calls get ``stream=None`` and adopt
      torch's capture stream. ``owners`` are retained with the torch graph.

    Both run with the Warp mempool disabled and ``wp.capture_launch`` forbidden."""
    if mode == "warp":
        s = _side_stream()
        wp.synchronize()
        with _recording(s) as cap:
            out = record(s, s)
        return (lambda: _replay(cap.graph, s)), out
    assert mode == "torch", mode
    torch = _torch_cuda()
    from track_gen.interop import torch as tg_torch

    g = torch.cuda.CUDAGraph()
    wp.synchronize()
    torch.cuda.synchronize()
    with tg_capture.mempool_disabled(DEV):
        with torch.cuda.graph(g):
            with tg_capture.capture_launch_forbidden():
                with tg_torch.registered() as ws:
                    assert isinstance(ws, wp.Stream) and ws.is_capturing
                    out = record(ws, None)
    torch.cuda.synchronize()
    if owners:
        tg_torch.retain(g, *owners)

    def replay():
        g.replay()
        torch.cuda.synchronize()
        wp.synchronize()

    return replay, out


def _rng(seeds) -> PerEnvSeededRNG:
    return PerEnvSeededRNG(seeds=seeds, num_envs=E, device=DEV)


def _track_gen(seeds=11, **kw) -> TrackGenerator:
    return TrackGenerator(TrackGenConfig(num_envs=E, device=DEV, **kw), _rng(seeds))


def _gate_gen(seeds=5, **kw) -> GateGenerator:
    # scale=3.0 keeps 0.2-wide gates valid (at scale 1 most sequences fail).
    kw.setdefault("gate_width", 0.2)
    kw.setdefault("scale", 3.0)
    return GateGenerator(GateGenConfig(num_envs=E, device=DEV, **kw), _rng(seeds))


def _points(src, s: int, *, gates: bool = False) -> np.ndarray:
    """One deterministic point per env on the geometry of ``src`` (host copy): a point
    40 % along a chord, walking around the line, slightly off it."""
    pts = _host(src.position if gates else src.center)
    n = pts.shape[0] // E
    p = pts.reshape(E, n, 3)
    cnt = _host(src.count)
    out = np.zeros((E, 3), np.float32)
    for e in range(E):
        m = int(min(cnt[e], n))
        if m >= 1:
            a = p[e, (s * 3 + 2 * e) % m]
            b = p[e, (s * 3 + 2 * e + 1) % m]
            out[e] = a + (b - a) * np.float32(0.4) + np.float32(
                [0.01 * ((s + e) % 3 - 1), 0.005 * (e % 2), 0.0])
    return out


def _boxes(p: np.ndarray, s: int, B: int):
    """``B`` boxes per env around the points ``p`` (``[E, 3]``): slot 0 on the point,
    slot ``b`` offset sideways by ``0.07 * b`` (out of bounds on narrow bands); slot 1 is
    inactive (NaN) on every third step. Returns host ``(position, orientation,
    half_extents)``, each ``[E * B, ...]``."""
    pos = np.repeat(p, B, axis=0).astype(np.float32)
    for b in range(1, B):
        pos[b::B] += np.float32([0.07 * b, -0.05 * b, 0.0])
    if B > 1 and s % 3 == 2:
        pos[1::B] = np.nan
    yaw = np.float32(0.3 * s) + np.arange(E * B, dtype=np.float32) * np.float32(0.1)
    q = np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2),
                  np.cos(yaw / 2)], axis=1).astype(np.float32)
    he = np.tile(np.float32([0.03, 0.015]), (E * B, 1))
    return pos, q, he


class _Inputs:
    """Stable bound input buffers (``[E]`` agent position, ``[E * B]`` boxes) written
    from the host outside any capture."""

    def __init__(self, B: int = 1) -> None:
        self.B = B
        self.pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
        self.boxes = wp.zeros(E * B, dtype=wp.vec3f, device=DEV)
        self.orient = wp.zeros(E * B, dtype=wp.quatf, device=DEV)
        self.he = wp.zeros(E * B, dtype=wp.vec2f, device=DEV)
        wp.synchronize()

    def write(self, src, s: int, *, gates: bool = False) -> None:
        p = _points(src, s, gates=gates)
        bp, q, he = _boxes(p, s, self.B)
        for dst, v in ((self.pos, p), (self.boxes, bp), (self.orient, q), (self.he, he)):
            wp.copy(dst, wp.array(v, dtype=dst.dtype, device=DEV))
        wp.synchronize()


# ---------------------------------------------------------------------------------------
# 1. Old-vs-new kernel identity for valid envs (frozen base kernels on cuda:0)
# ---------------------------------------------------------------------------------------

def _valid_track(z: str, seeds: int = 11):
    gen = _track_gen(seeds, **_Z[z])
    t = gen.program.result
    valid = _host(t.valid)
    assert valid.sum() >= E - 2, f"{z}: too few valid envs to compare: {valid.tolist()}"
    return gen, t, valid


def _zeros(n: int, dtype) -> "wp.array":
    return wp.zeros(n, dtype=dtype, device=DEV)


def _contact_arrays(n: int) -> list:
    """oob, distance, nearest, normal, boundary (``BoxContact`` order)."""
    return [_zeros(n, wp.int32), _zeros(n, wp.float32), _zeros(n, wp.vec2f),
            _zeros(n, wp.vec2f), _zeros(n, wp.int32)]


_BOX_FIELDS = ("oob", "distance", "nearest", "normal", "boundary")
_DISC_FIELDS = ("hit", "disc", "depth", "nearest")
_EVENT_FIELDS = ("passed", "checkpoint_passed", "next_checkpoint", "laps", "progress",
                 "wrong_way", "wrong_checkpoint", "dist_to_next")
_FRAME_FIELDS = ("s", "n", "n_up", "segment")
_PROP_FIELDS = ("position", "tangent", "yaw", "length")
_CP_FIELDS = ("position", "left", "right", "tangent", "up_half")


@pytest.mark.parametrize("z", sorted(_Z))
def test_collision_matches_the_base_kernels_for_valid_envs(z):
    """Segments and SDF ``CollisionChecker`` (2 boxes per env, inactive NaN slots on
    some steps) vs the base ``_box_query_segments_k`` / ``_track_aabb_k`` +
    ``_sdf_bake_k`` (+ the unchanged ``_box_query_sdf_k`` on the base bake): every
    output row of every valid env is byte-identical, bake arrays included."""
    _, t, valid = _valid_track(z)
    n_max, B, R = int(t.layout.stride), 2, 32
    inp = _Inputs(B)
    seg = CollisionChecker(t, max_boxes=B, method="segments")
    sdf = CollisionChecker(t, max_boxes=B, method="sdf", sdf_resolution=R)
    lo, hi = _zeros(E, wp.vec2f), _zeros(E, wp.vec2f)
    phi, bid = _zeros(E * R * R, wp.float32), _zeros(E * R * R, wp.int8)
    wp.launch(_BASE["_track_aabb_k"], E, [t.outer, t.count, n_max, -1.0, 0.1, lo, hi],
              device=DEV)
    wp.launch(_BASE["_sdf_bake_k"], E * R * R,
              [t.inner, t.outer, t.count, n_max, R, lo, hi, phi, bid], device=DEV)
    wp.synchronize()
    for name, new, old in (("lo", sdf._sdf_lo, lo), ("hi", sdf._sdf_hi, hi),
                           ("phi", sdf._sdf_phi, phi), ("bid", sdf._sdf_bid, bid)):
        _assert_valid_rows(new, old, valid, f"{z}: sdf bake {name}")
    oob_seen = set()
    for s in range(8):
        inp.write(t, s)
        new_seg = seg.query(inp.boxes, inp.orient, inp.he)
        new_sdf = sdf.query(inp.boxes, inp.orient, inp.he)
        old_seg, old_sdf = _contact_arrays(E * B), _contact_arrays(E * B)
        wp.launch(_BASE["_box_query_segments_k"], E * B,
                  [t.inner, t.outer, t.center, t.count, n_max, B, inp.boxes, inp.orient,
                   inp.he, *old_seg], device=DEV)
        wp.launch(_box_query_sdf_k, E * B,
                  [lo, hi, phi, bid, R, B, inp.boxes, inp.orient, inp.he, *old_sdf],
                  device=DEV)
        wp.synchronize()
        for f, a in zip(_BOX_FIELDS, old_seg):
            _assert_valid_rows(getattr(new_seg, f), a, valid,
                               f"{z}: step {s}: segments {f}")
        for f, a in zip(_BOX_FIELDS, old_sdf):
            _assert_valid_rows(getattr(new_sdf, f), a, valid, f"{z}: step {s}: sdf {f}")
        oob_seen |= set(_rows(new_seg.oob)[valid != 0].ravel().tolist())
    assert oob_seen == {0, 1}, f"{z}: boxes never (or always) out of bounds: {oob_seen}"


@pytest.mark.parametrize("z", sorted(_Z))
def test_samplers_and_bakes_match_the_base_kernels_for_valid_envs(z):
    """``PropSampler`` (both boundaries, both modes), ``CheckpointSampler`` (set +
    ``truncated`` / ``step``) and ``HeightFieldBaker`` vs the base ``_scan_boundary_k``
    / ``_place_checkpoints_k`` / ``_track_aabb_k`` with the unchanged place / bake
    kernels, at explicit capacities: byte-identical rows for every valid env."""
    _, t, valid = _valid_track(z)
    n_max = int(t.layout.stride)
    # Props.
    MP, sp = 96, 0.15
    for boundary in ("outer", "inner"):
        for mode in ("points", "segments"):
            new = PropSampler(t, sp, boundary=boundary, mode=mode, max_props=MP).sample()
            pts = t.outer if boundary == "outer" else t.inner
            cum = _zeros(E * n_max, wp.float32)
            cnt, stp, tr = _zeros(E, wp.int32), _zeros(E, wp.float32), _zeros(E, wp.int32)
            out = [_zeros(E * MP, wp.vec3f), _zeros(E * MP, wp.vec2f),
                   _zeros(E * MP, wp.float32), _zeros(E * MP, wp.float32)]
            wp.launch(_BASE["_scan_boundary_k"], E,
                      [pts, t.count, n_max, sp, MP, cum, cnt, stp, tr], device=DEV)
            wp.launch(_place_props_k, E * MP,
                      [pts, t.count, n_max, cum, cnt, stp, MP, 0 if mode == "points" else 1,
                       *out], device=DEV)
            wp.synchronize()
            label = f"{z}: props[{boundary},{mode}]"
            for f, a in zip(_PROP_FIELDS, out):
                _assert_valid_rows(getattr(new, f), a, valid, f"{label} {f}")
            for f, a in (("count", cnt), ("step", stp), ("truncated", tr)):
                _assert_valid_rows(getattr(new, f), a, valid, f"{label} {f}")
    # Checkpoints.
    M, cs = 48, 0.5
    sampler = CheckpointSampler(t, cs, max_checkpoints=M)
    new = sampler.sample()
    cum = _zeros(E * n_max, wp.float32)
    cnt, stp, tr = _zeros(E, wp.int32), _zeros(E, wp.float32), _zeros(E, wp.int32)
    out = [_zeros(E * M, wp.vec3f) for _ in range(4)] + [_zeros(E * M, wp.float32)]
    wp.launch(_BASE["_scan_boundary_k"], E,
              [t.center, t.count, n_max, cs, M, cum, cnt, stp, tr], device=DEV)
    wp.launch(_BASE["_place_checkpoints_k"], E * M,
              [t.center, t.inner, t.outer, t.count, n_max, cum, cnt, stp, M, *out],
              device=DEV)
    wp.synchronize()
    for f, a in zip(_CP_FIELDS, out):
        _assert_valid_rows(getattr(new, f), a, valid, f"{z}: checkpoints {f}")
    _assert_valid_rows(new.count, cnt, valid, f"{z}: checkpoints count")
    _assert_valid_rows(sampler.step, stp, valid, f"{z}: checkpoints step")
    _assert_valid_rows(sampler.truncated, tr, valid, f"{z}: checkpoints truncated")
    # The set owns valid / generation: equal to the source's.
    _assert_bytes(new.valid, t.valid, f"{z}: checkpoints valid (owned copy)")
    _assert_bytes(new.generation, t.generation, f"{z}: checkpoints generation (owned copy)")
    assert new.valid.ptr != t.valid.ptr and new.generation.ptr != t.generation.ptr
    # Heightfield.
    R = 16
    hf = HeightFieldBaker(t, R).bake()
    lo, hi, h = _zeros(E, wp.vec2f), _zeros(E, wp.vec2f), _zeros(E * R * R, wp.float32)
    wp.launch(_BASE["_track_aabb_k"], E, [t.outer, t.count, n_max, 0.0, 0.1, lo, hi],
              device=DEV)
    wp.launch(_height_bake_k, E * R * R, [t.center, t.count, t.valid, n_max, R, lo, hi, h],
              device=DEV)
    wp.synchronize()
    for f, a in (("height", h), ("lo", lo), ("hi", hi)):
        _assert_valid_rows(getattr(hf, f), a, valid, f"{z}: heightfield {f}")


@pytest.mark.parametrize("z", sorted(_Z))
def test_progress_and_localizer_match_the_base_kernels_along_a_trajectory(z):
    """``ProgressTracker`` and a cold / warm ``TrackLocalizer`` vs the base
    ``_progress_update_k`` / ``_localize_k`` over 45 steps with three masked advancing
    regenerations (the producer's own masked commit), each followed by ``sample()`` and
    explicit resets on both sides (the base reset kernels are unchanged): every event,
    frame and state row of every env valid at that step is byte-identical. The new
    kernels also self-reset on the generation change; idempotent with the explicit
    reset, so nothing differs (non-vacuity: progress was made and reset)."""
    gen, t, _ = _valid_track(z, seeds=3)
    n_max, M, W = int(t.layout.stride), 64, 3
    sampler = CheckpointSampler(t, 0.5, max_checkpoints=M)
    cps = sampler.sample()
    pos = _zeros(E, wp.vec3f)
    wp.synchronize()
    tracker = ProgressTracker(cps, position=pos)
    cold = TrackLocalizer(t, position=pos)
    warm = TrackLocalizer(t, warm_window=W, position=pos)
    prev = _dev(np.full((E, 3), np.nan, np.float32), wp.vec3f)
    nxt, laps, prog = _zeros(E, wp.int32), _zeros(E, wp.int32), _zeros(E, wp.int32)
    ev = [_zeros(E, wp.int32) for _ in range(7)] + [_zeros(E, wp.float32)]
    last = {"cold": _dev(np.full(E, -1, np.int32), wp.int32),
            "warm": _dev(np.full(E, -1, np.int32), wp.int32)}
    fr = {k: [_zeros(E, wp.float32) for _ in range(3)] + [_zeros(E, wp.int32)]
          for k in last}
    schedule = {12: _M1, 27: _M2, 33: _ONES}
    progressed = 0
    for s in range(45):
        if s in schedule:
            m = _i32(schedule[s])
            progressed += int((_host(prog) * schedule[s]).sum())
            gen.enqueue(advance=True, mask=m)    # eager: never a self-graph
            sampler.sample()
            tracker.reset(m)
            cold.reset(m)
            warm.reset(m)
            wp.launch(_progress_reset_k, E, [m, prev, nxt, laps, prog], device=DEV)
            for k in last:
                wp.launch(_localize_reset_k, E, [m, last[k]], device=DEV)
            wp.synchronize()
        _write(pos, _points(t, s))
        new_ev, new_c, new_w = tracker.update(), cold.query(), warm.query()
        wp.launch(_BASE["_progress_update_k"], E,
                  [cps.position, cps.left, cps.right, cps.tangent, cps.up_half,
                   cps.count, M, pos, prev, nxt, laps, prog, *ev], device=DEV)
        for k, Wk in (("cold", 0), ("warm", W)):
            wp.launch(_BASE["_localize_k"], E,
                      [t.center, t.arclen, t.length, t.count, n_max, Wk, pos, last[k],
                       *fr[k]], device=DEV)
        wp.synchronize()
        valid = _host(t.valid)
        label = f"{z}: step {s}"
        for f, a in zip(_EVENT_FIELDS, ev):
            _assert_valid_rows(getattr(new_ev, f), a, valid, f"{label}: events {f}")
        for f, a in (("_prev_pos", prev), ("_next", nxt), ("_laps", laps),
                     ("_progress", prog)):
            _assert_valid_rows(getattr(tracker, f), a, valid, f"{label}: progress {f}")
        for k, new in (("cold", new_c), ("warm", new_w)):
            for f, a in zip(_FRAME_FIELDS, fr[k]):
                _assert_valid_rows(getattr(new, f), a, valid, f"{label}: {k} frame {f}")
        _assert_valid_rows(warm._last, last["warm"], valid, f"{label}: warm last segment")
    assert progressed > 0, "no regenerated env had made progress: the resets were vacuous"
    assert _host(gen.seeds.epoch).max() >= 2


def test_gate_utilities_match_the_base_kernels_for_valid_envs():
    """Gates: ``CourseLine.refresh()`` vs the base ``_sample_line_k`` (+ the unchanged
    ``_line_frames_k``); the post ``DiscChecker`` given the real post count and
    ``valid`` vs the base ``_box_query_discs_k`` scanning ``2 * max_gates`` posts (the
    pre-PR14 Course); ``curvature`` / ``speed_profile`` on the course line vs the base
    kernels: byte-identical rows for every valid env, and the line's owned
    ``generation`` equals the gates'."""
    gen = _gate_gen(5)
    seq = gen.program.result
    valid = _host(seq.valid)
    assert valid.sum() >= E - 1, valid.tolist()
    G, spg = int(seq.layout.stride), 8
    n = G * spg
    line = CourseLine(seq, spg)
    line.refresh()
    c = line.centerline
    center, tangent = _zeros(E * n, wp.vec3f), _zeros(E * n, wp.vec3f)
    arclen, length = _zeros(E * n, wp.float32), _zeros(E, wp.float32)
    count, lvalid = _zeros(E, wp.int32), _zeros(E, wp.int32)
    wp.launch(_BASE["_sample_line_k"], E * n,
              [seq.position, seq.count, seq.valid, G, spg, center, count, lvalid],
              device=DEV)
    wp.launch(_line_frames_k, E, [center, count, n, tangent, arclen, length], device=DEV)
    wp.synchronize()
    for f, a in (("center", center), ("tangent", tangent), ("arclen", arclen),
                 ("length", length), ("count", count), ("valid", lvalid)):
        _assert_valid_rows(getattr(c, f), a, valid, f"course line {f}")
    _assert_bytes(c.generation, seq.generation, "course line generation (owned copy)")
    assert c.generation.ptr != seq.generation.ptr
    # Discs on the posts.
    left, right = _host(seq.left), _host(seq.right)
    posts_h = np.empty((2 * E * G, 3), np.float32)
    posts_h[0::2], posts_h[1::2] = left, right
    posts = _dev(posts_h, wp.vec3f)
    post_count = _i32(2 * np.minimum(_host(seq.count), G))
    full = _i32(np.full(E, 2 * G))
    inp = _Inputs(1)
    discs = DiscChecker(posts, radius=0.04, max_boxes=1, count=post_count, valid=seq.valid)
    hits = 0
    cnt = np.maximum(np.minimum(_host(seq.count), G), 1)
    for s in range(12):
        # A box on (or next to) a real left post of every env.
        p = _host(seq.left).reshape(E, G, 3)[np.arange(E), (s + np.arange(E)) % cnt]
        p = p + np.float32([0.01 * (s % 3), -0.01, 0.0])
        bp, q, he = _boxes(p.astype(np.float32), s, 1)
        for dst, v in ((inp.boxes, bp), (inp.orient, q), (inp.he, he)):
            wp.copy(dst, wp.array(v, dtype=dst.dtype, device=DEV))
        wp.synchronize()
        new = discs.query(inp.boxes, inp.orient, inp.he)
        old = [_zeros(E, wp.int32), _zeros(E, wp.int32), _zeros(E, wp.float32),
               _zeros(E, wp.vec2f)]
        wp.launch(_BASE["_box_query_discs_k"], E,
                  [posts, full, 2 * G, 0.04, 1, inp.boxes, inp.orient, inp.he, *old],
                  device=DEV)
        wp.synchronize()
        for f, a in zip(_DISC_FIELDS, old):
            _assert_valid_rows(getattr(new, f), a, valid, f"step {s}: discs {f}")
        hits += int(_host(new.hit)[valid != 0].sum())
    assert hits > 0, "no box ever hit a post: the disc comparison is vacuous"
    _assert_curvature_identity(c, _host(c.valid), "course line")


def _assert_curvature_identity(src, valid: np.ndarray, label: str) -> None:
    n_max = int(src.center.shape[0]) // E
    N = E * n_max
    for window in (0, 2):
        raw, sm = _zeros(N, wp.float32), _zeros(N, wp.float32)
        wp.launch(_BASE["_curvature_raw_k"], N, [src.center, src.count, n_max, raw],
                  device=DEV)
        ref_k = raw
        if window:
            wp.launch(_BASE["_smooth_wrap_k"], N, [raw, src.count, n_max, window, sm],
                      device=DEV)
            ref_k = sm
        v = _zeros(N, wp.float32)
        wp.launch(_BASE["_speed_profile_k"], E,
                  [ref_k, src.arclen, src.length, src.count, n_max, 4.0, 2.0, 3.0, 5.0, v],
                  device=DEV)
        wp.synchronize()
        lab = f"{label}: window {window}"
        _assert_valid_rows(curvature(src, window), ref_k, valid,
                           f"{lab}: curvature (allocating)")
        k_out, scratch = _zeros(N, wp.float32), _zeros(N, wp.float32)
        got = curvature(src, window, out=k_out, scratch=scratch)
        assert got is k_out, f"{lab}: curvature(out=) must return out"
        _assert_valid_rows(k_out, ref_k, valid, f"{lab}: curvature(out=)")
        v_out = _zeros(N, wp.float32)
        got = speed_profile(src, 4.0, 2.0, 3.0, 5.0, kappa=k_out, out=v_out)
        assert got is v_out, f"{lab}: speed_profile(out=) must return out"
        _assert_valid_rows(v_out, v, valid, f"{lab}: speed_profile(out=)")
        _assert_valid_rows(speed_profile(src, 4.0, 2.0, 3.0, 5.0, kappa=ref_k), v, valid,
                           f"{lab}: speed_profile (allocating)")


@pytest.mark.parametrize("z", sorted(_Z))
def test_curvature_and_speed_profile_match_the_base_kernels_for_valid_envs(z):
    """``curvature`` (window 0 and 2; allocating and ``out=`` / ``scratch=``) and
    ``speed_profile`` (allocating and ``out=``) on a ``Track`` vs the base
    ``_curvature_raw_k`` / ``_smooth_wrap_k`` / ``_speed_profile_k``: byte-identical rows
    for every valid env."""
    _, t, valid = _valid_track(z)
    _assert_curvature_identity(t, valid, z)


# ---------------------------------------------------------------------------------------
# 2. The eager-vs-captured trajectory twin of an RL loop (Course, track and gates modes)
# ---------------------------------------------------------------------------------------

_RL_STEPS = 200
_TWIN_VARIANTS = ("track", "track-invalid", "gates-posts", "gates-frames")
#: The envs the track-invalid variant drives at size 8 (truncated: invalid, but for a
#: rare regeneration that fits).
_INVALID_ENVS = (2, 5)
_BIG_SIZE = 8.0
#: Sideways offsets of the agents (plan view): track half width 0.1, so envs 3 and 5
#: drive out of bounds; gate half width 0.1, posts radius 0.02: envs 4 and 5 graze posts.
_OFFSETS = {"track": np.float32([0.0, 0.03, 0.06, -0.13, 0.0, 0.13, -0.06, 0.0]),
            "gates": np.float32([0.0, 0.02, -0.03, 0.0, 0.075, -0.075, 0.0, 0.01])}
_SPACING = 0.6


def _twin_cfg(variant: str, seeds: int = 5) -> CourseConfig:
    if variant.startswith("track"):
        # Default max_checkpoints: the plan bound (PR14), not a readback.
        return CourseConfig(mode="track", gen=TrackGenConfig(num_envs=E, device=DEV),
                            seeds=seeds, collision="segments", heightfield_resolution=16,
                            checkpoint_spacing=_SPACING)
    # scale=3.0 keeps 0.2-wide gates valid; the "checkpoint" gate generator gives
    # sequences an agent following the course line laps in a few dozen steps.
    gcfg = dict(num_envs=E, device=DEV, gate_width=0.2, scale=3.0, generator="checkpoint")
    if variant == "gates-posts":
        return CourseConfig(mode="gates", gen=GateGenConfig(**gcfg), seeds=seeds,
                            post_radius=0.02, localize_window=4)
    assert variant == "gates-frames", variant
    gcfg.update(z_profile="uniform", z_min=0.5, z_max=1.5)
    return CourseConfig(mode="gates", gen=GateGenConfig(**gcfg), seeds=seeds,
                        frame_collision=True, agent_radius=0.05, frame_thickness=0.05,
                        frame_depth=0.05, localize_window=4)


def _twin_course(variant: str) -> Course:
    course = Course(_twin_cfg(variant))
    if variant == "track-invalid":
        sizes = np.ones(E, np.float32)
        sizes[list(_INVALID_ENVS)] = _BIG_SIZE
        course.generator.params["generator"].set("size", _dev(sizes, wp.float32))
        course.enqueue_generate()                      # eager: never a self-graph
        wp.synchronize()
        v = _host(course.result.valid)
        assert (v[list(_INVALID_ENVS)] == 0).all(), f"size {_BIG_SIZE} must truncate: {v}"
    return course


class _Agents:
    """The caller's per-env agent state and bound buffers (identical device logic for
    both twins)."""

    def __init__(self, course: Course, variant: str) -> None:
        mode = "track" if variant.startswith("track") else "gates"
        self.frames = variant == "gates-frames"
        self.line = course.result if mode == "track" else course.course_line.centerline
        self.n_max = int(self.line.center.shape[0]) // E
        # Below one checkpoint spacing per step (track: ~10 samples between checkpoints
        # at spacing 0.6; gates: 8 line samples per gate), so no checkpoint is jumped.
        speed = (2 + np.arange(E) % 5 if mode == "track" else 1 + np.arange(E) % 4)
        speed = speed.astype(np.int32)
        self.speed = wp.array(speed, dtype=wp.int32, device=DEV)
        self.offset = wp.array(_OFFSETS[mode], dtype=wp.float32, device=DEV)
        # Step budgets on both sides of the lap times, so both done rules fire.
        self.period = wp.array((20 + 9 * np.arange(E)).astype(np.int32), dtype=wp.int32,
                               device=DEV)
        self.t = wp.array(-2 * speed, dtype=wp.int32, device=DEV)   # as after a done
        self.steps = wp.zeros(E, dtype=wp.int32, device=DEV)
        self.pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
        self.orient = wp.array(np.tile(np.float32([0, 0, 0, 1]), (E, 1)), dtype=wp.quatf,
                               device=DEV)
        self.he = wp.array(np.full((E, 2), 0.02, np.float32), dtype=wp.vec2f, device=DEV)
        wp.synchronize()   # uploads ran on Warp's current stream; moves may use another

    def bind(self, course: Course) -> None:
        if self.frames:
            course.bind(self.pos)
        else:
            course.bind(position=self.pos, orientation=self.orient, half_extents=self.he)

    def move(self, stream=None) -> None:
        wp.launch(_move_agents_k, dim=E,
                  inputs=[self.line.center, self.line.count, self.n_max, self.speed,
                          self.offset, self.t, self.steps, self.pos, self.orient],
                  device=DEV, stream=stream)

    def decide(self, laps, valid, done, stream=None) -> None:
        wp.launch(_done_k, dim=E,
                  inputs=[laps, valid, self.period, self.speed, self.t, self.steps, done],
                  device=DEV, stream=stream)

    def arrays(self) -> dict:
        return {"agent.t": self.t, "agent.steps": self.steps, "agent.pos": self.pos,
                "agent.orient": self.orient}


def _seen(utility) -> "wp.array":
    """A stateful utility's per-env ``seen`` generation (design section 4)."""
    return utility._seen


def _step_outputs(sr) -> dict:
    """The StepResult outputs ``step()`` writes for every env (poisoned before a replay);
    ``StepResult.valid`` aliases the producer's ``valid`` and is never poisoned."""
    out = {}
    for label, obj in (("events", sr.events), ("contacts", sr.contacts),
                       ("frame", sr.frame)):
        out.update(_arrays(obj, label))
    return out


def _course_state(course: Course, sr=None) -> dict:
    """Every buffer a Course step / regeneration produces, and the runtime state."""
    out = {**_arrays(course.result, "result"), "epoch": course.seeds.epoch,
           **_arrays(course.checkpoints, "checkpoints")}
    if course.checkpoint_sampler is not None:
        out["sampler.truncated"] = course.checkpoint_sampler.truncated
        out["sampler.step"] = course.checkpoint_sampler.step
    if course.heightfield is not None:
        out.update(_arrays(course.heightfield.result, "heightfield"))
    if course._posts is not None:
        out["posts"] = course._posts
    if isinstance(course.collision, DiscChecker):
        out["post_count"] = course.collision._count
    if course.course_line is not None:
        out.update(_arrays(course.course_line.centerline, "line"))
    p = course.progress
    out.update({f"progress.{k}": getattr(p, k)
                for k in ("_prev_pos", "_next", "_laps", "_progress")})
    out["progress.seen"] = _seen(p)
    if course.localizer is not None:
        out["localizer._last"] = course.localizer._last
        out["localizer.seen"] = _seen(course.localizer)
    if sr is not None:
        out.update(_step_outputs(sr))
        out["step.valid"] = sr.valid
    return out


def _rl_state(course, agents: _Agents, done, sr) -> dict:
    return {**_course_state(course, sr), **agents.arrays(), "done": done}


def _record_rl_step(course: Course, agents: _Agents, done, kstream, lstream):
    """One RL step: physics stub -> step() -> device done rule -> masked advancing
    regeneration (the test's kernels on ``kstream``, library calls on ``lstream``)."""
    agents.move(stream=kstream)
    sr = course.step(stream=lstream)
    agents.decide(sr.events.laps, sr.valid, done, stream=kstream)
    assert course.regenerate(mask=done, advance=True, stream=lstream) is course.result
    return sr


def _assert_invalid_env_sentinels(course, sr, valid_at_step: np.ndarray, done: np.ndarray,
                                  label: str) -> int:
    """The sentinel table after a replay of the RL step (max_boxes 1): the step outputs
    of every env invalid when ``step()`` ran (``valid_at_step == 0``) and the derived
    buffers of every env invalid after the regeneration (``sr.valid == 0``). Returns the
    number of envs invalid at the step."""
    J = np.flatnonzero(valid_at_step == 0)
    assert (done[J] != 0).all(), f"{label}: an invalid env was not done (1 - valid)"
    ev = {f: _host(getattr(sr.events, f)) for f in _EVENT_FIELDS}
    for f, want in (("passed", 0), ("checkpoint_passed", -1), ("wrong_way", 0),
                    ("wrong_checkpoint", -1), ("dist_to_next", 0.0),
                    # frozen at the state of its last reset (its last regeneration)
                    ("next_checkpoint", 0), ("laps", 0), ("progress", 0)):
        assert (ev[f][J] == want).all(), f"{label}: events.{f} {ev[f].tolist()} (J {J})"
    assert np.isnan(_host(course.progress._prev_pos)[J]).all(), \
        f"{label}: prev_pos re-armed"
    c = sr.contacts
    assert (_host(c.oob)[J] == 1).all() and (_host(c.boundary)[J] == -1).all(), \
        f"{label}: oob {_host(c.oob).tolist()} (J {J})"
    for f in ("distance", "nearest", "normal"):
        assert np.isnan(_rows(getattr(c, f))[J]).all(), f"{label}: contacts.{f}"
    Jn = np.flatnonzero(_host(sr.valid) == 0)
    assert (_host(course.checkpoints.count)[Jn] == 0).all(), f"{label}: checkpoint count"
    assert (_host(course.checkpoints.valid)[Jn] == 0).all(), f"{label}: checkpoint valid"
    assert np.isnan(_rows(course.checkpoints.position)[Jn]).all(), \
        f"{label}: checkpoint rows"
    hf = course.heightfield.result
    assert (_host(hf.lo)[Jn] == -1.0).all() and (_host(hf.hi)[Jn] == 1.0).all(), label
    assert np.isnan(_rows(hf.height)[Jn]).all(), f"{label}: heights"
    return len(J)


@pytest.mark.parametrize("variant", _TWIN_VARIANTS)
@pytest.mark.parametrize("mode", ["warp", "torch"])
def test_rl_loop_in_one_caller_graph_equals_the_eager_twin(mode, variant):
    """ONE caller graph per RL step (physics stub -> ``course.step()`` -> device done
    rule reading ``events.laps`` and ``StepResult.valid`` -> ``course.regenerate(
    mask=course.inputs["regen_mask"], advance=True)``), recorded in ``mode`` and replayed
    200 times with no host involvement, equals an eager twin driven step by step (its
    host calls ``enqueue_generate(mask=done, advance=True)`` only when an env is done,
    so the graph's zero-mask steps must be exact no-ops) EVERY step, bitwise, on every
    buffer of :func:`_rl_state`. Before each replay the StepResult outputs are poisoned
    (a write missing from the graph shows); nothing that aliases producer state is.
    Non-vacuity: both done rules fired, steps with and without a regeneration, some env
    regenerated twice; track: the default checkpoint capacity is the plan bound and both
    oob values occur; track-invalid: after every replay the sentinel table holds on the
    step outputs of every env invalid when ``step()`` ran, and on the checkpoints and
    heightfield of every env invalid after the regeneration (about two envs per step);
    gates-posts: some box hit a post."""
    _load_test_kernels()
    course, twin = _twin_course(variant), _twin_course(variant)
    if variant.startswith("track"):
        M = math.ceil(1.5 * 384 * 0.06 / _SPACING)
        assert course.checkpoints.position.shape[0] == E * M == E * 58, \
            "the default max_checkpoints must be the plan bound"
    agents_c, agents_t = _Agents(course, variant), _Agents(twin, variant)
    agents_c.bind(course)
    agents_t.bind(twin)
    done_c = course.inputs["regen_mask"]       # the course's own persistent mask buffer
    done_t = wp.zeros(E, dtype=wp.int32, device=DEV)
    s = _side_stream()

    # Warm-up outside any capture: the test kernels (scratch buffers), step() and the
    # masked regeneration (zero mask: an exact no-op), then a full reset on both.
    scratch = _Agents(course, variant)
    z0 = wp.zeros(E, dtype=wp.int32, device=DEV)
    z1 = wp.zeros(E, dtype=wp.int32, device=DEV)
    ones = wp.full(E, 1, dtype=wp.int32, device=DEV)
    wp.synchronize()
    scratch.move(stream=s)
    scratch.decide(z0, ones, z1, stream=s)
    sr_c = course.step(stream=s)
    assert course.regenerate(mask=done_c, advance=True, stream=s) is course.result
    sr_t = twin.step()
    wp.synchronize()
    course.reset(ones, stream=s)
    twin.reset(ones)
    wp.synchronize()
    assert sr_c.valid.ptr == course.result.valid.ptr, \
        "StepResult.valid aliases result.valid"
    _assert_state(_rl_state(course, agents_c, done_c, sr_c),
                  _rl_state(twin, agents_t, done_t, sr_t), f"{variant}: after warm-up")
    before = _snapshot(_rl_state(course, agents_c, done_c, sr_c))

    replay, sr = _caller_graph(
        mode, lambda ks, ls: _record_rl_step(course, agents_c, done_c, ks, ls),
        owners=(course,))
    assert sr is sr_c, "step() must return the same StepResult"
    _assert_state(_rl_state(course, agents_c, done_c, sr_c), before,
                  f"{mode}/{variant}: the recording executed work")

    lap_done = budget_done = invalid_done = idle_steps = reset_steps = 0
    invalid_at_step, oob_seen, hits = 0, set(), 0
    small = np.ones(E, bool)                   # the envs at size 1
    if variant == "track-invalid":
        small[list(_INVALID_ENVS)] = False
    for k in range(_RL_STEPS):
        _poison(_step_outputs(sr_c).values())
        replay()

        agents_t.move()
        assert twin.step() is sr_t
        agents_t.decide(sr_t.events.laps, sr_t.valid, done_t)
        wp.synchronize()
        d = _host(done_t).copy()
        laps = _host(sr_t.events.laps)
        vt = _host(sr_t.valid)
        if d.any():
            twin.enqueue_generate(mask=done_t, advance=True)
            wp.synchronize()
            reset_steps += 1
            lap_done += int(((laps >= 1) & (d != 0) & (vt != 0)).sum())
            budget_done += int(((laps < 1) & (d != 0) & (vt != 0)).sum())
            invalid_done += int(((d != 0) & (vt == 0)).sum())
        else:
            idle_steps += 1
        label = f"{mode}/{variant}: step {k} (done {d.tolist()})"
        _assert_state(_rl_state(course, agents_c, done_c, sr_c),
                      _rl_state(twin, agents_t, done_t, sr_t), label)
        if variant == "track-invalid":       # vt: the validity step() read
            invalid_at_step += _assert_invalid_env_sentinels(course, sr_c, vt, d, label)
        if variant.startswith("track"):
            oob_seen |= set(_host(sr_c.contacts.oob)[vt != 0].tolist())
        if variant == "gates-posts":
            hits += int(_host(sr_c.contacts.hit).sum())
    assert twin._runner.graph_keys == (), "the eager twin must never self-graph"
    epoch = _host(course.seeds.epoch)
    _assert_bytes(course.result.generation, epoch, f"{variant}: generation == epoch")
    # Non-vacuity.
    assert lap_done > 0, "no valid env ever completed a lap (laps >= 1 never exercised)"
    assert budget_done > 0, "the per-env step budget never fired"
    assert reset_steps > 0, reset_steps
    if variant == "track-invalid":
        # About two of the eight envs are invalid at every step (size 8 truncates;
        # a regeneration may fit now and then).
        assert invalid_at_step >= _RL_STEPS and invalid_done == invalid_at_step, \
            (invalid_at_step, invalid_done)
    else:
        assert idle_steps > 0, \
            "every step regenerated some env: zero-mask steps unexercised"
    assert epoch[small].max() >= 2, f"no size-1 env was regenerated twice: {epoch.tolist()}"
    if variant.startswith("track"):
        assert oob_seen == {0, 1}, f"valid envs' oob never varied: {oob_seen}"
    if variant == "gates-posts":
        assert hits > 0, "no box ever hit a gate post"


# ---------------------------------------------------------------------------------------
# 3. Invalid-env sentinels under capture: K (valid = 0, geometry kept) == P (poisoned)
# ---------------------------------------------------------------------------------------

def _keep_and_poison(src):
    """``(K, P)``: two clones of ``src`` with the envs of :data:`_FAIL` failed by the
    commit's own kernel, at ``poison=0`` (geometry kept, ``valid = 0``: what
    ``on_invalid="keep"`` leaves) and ``poison=1`` (``count = 0``, NaN geometry)."""
    K, P = src.clone(), src.clone()
    fail = _i32(_FAIL)
    ctx = LaunchContext.for_call(DEV)
    _commit_envs(K, fail, poison=False, ctx=ctx)
    _commit_envs(P, fail, poison=True, ctx=ctx)
    wp.synchronize()
    for name, r in (("K", K), ("P", P)):
        np.testing.assert_array_equal(_host(r.valid), (1 - _FAIL) * _host(src.valid),
                                      err_msg=f"{name}: valid")
    assert (_host(P.count)[_FAIL != 0] == 0).all() and \
        (_host(K.count)[_FAIL != 0] > 0).all(), "K keeps the geometry, P poisons it"
    return K, P


class _TrackUtilities:
    """Every track utility on one source, bound to shared inputs; ``run(ctx)`` makes
    their hot calls. ``outputs()``: the owned arrays compared (results and runtime
    state); ``written()``: the result arrays every ``run`` rewrites (poisoned before
    each run)."""

    def __init__(self, src, inp: _Inputs) -> None:
        B = inp.B
        self.seg = CollisionChecker(src, max_boxes=B, method="segments", position=inp.boxes,
                                    orientation=inp.orient, half_extents=inp.he)
        self.sdf = CollisionChecker(src, max_boxes=B, method="sdf", sdf_resolution=16,
                                    position=inp.boxes, orientation=inp.orient,
                                    half_extents=inp.he)
        self.props = PropSampler(src, 0.15, max_props=96)
        self.sampler = CheckpointSampler(src, 0.5, max_checkpoints=48)
        self.tracker = ProgressTracker(self.sampler.result, position=inp.pos)
        self.loc = TrackLocalizer(src, warm_window=3, position=inp.pos)
        self.hf = HeightFieldBaker(src, 16)
        N = int(src.center.shape[0])
        self.kappa = wp.zeros(N, dtype=wp.float32, device=DEV)
        self.scratch = wp.zeros(N, dtype=wp.float32, device=DEV)
        self.speed = wp.zeros(N, dtype=wp.float32, device=DEV)
        self.src = src

    def run(self, ctx) -> None:
        self.seg.query(ctx=ctx)
        self.sdf.bake(ctx=ctx)
        self.sdf.query(ctx=ctx)
        self.props.sample(ctx=ctx)
        self.sampler.sample(ctx=ctx)
        self.tracker.update(ctx=ctx)
        self.loc.query(ctx=ctx)
        self.hf.bake(ctx=ctx)
        curvature(self.src, 2, ctx, out=self.kappa, scratch=self.scratch)
        speed_profile(self.src, 4.0, 2.0, 3.0, 5.0, kappa=self.kappa, ctx=ctx,
                      out=self.speed)

    def written(self) -> dict:
        s = self.sdf
        return {**_arrays(self.seg.result, "segments"), **_arrays(self.sdf.result, "sdf"),
                "sdf.lo": s._sdf_lo, "sdf.hi": s._sdf_hi, "sdf.phi": s._sdf_phi,
                "sdf.bid": s._sdf_bid, **_arrays(self.props.result, "props"),
                **_arrays(self.sampler.result, "checkpoints"),
                "sampler.truncated": self.sampler.truncated,
                "sampler.step": self.sampler.step,
                **_arrays(self.tracker.result, "events"),
                **_arrays(self.loc.result, "frame"),
                **_arrays(self.hf.result, "heightfield"), "curvature": self.kappa,
                "curvature.scratch": self.scratch, "speed": self.speed}

    def outputs(self) -> dict:
        t, loc = self.tracker, self.loc
        return {**self.written(), "progress._prev_pos": t._prev_pos,
                "progress._next": t._next, "progress._laps": t._laps,
                "progress._progress": t._progress, "progress.seen": _seen(t),
                "localizer._last": loc._last, "localizer.seen": _seen(loc)}


class _GateUtilities:
    """Every gate utility on one source: the course line (+ a warm localizer on it),
    posts discs given ``valid``, progress on ``from_gates`` and frame collision windowed
    on it."""

    def __init__(self, src, inp: _Inputs) -> None:
        G = int(src.layout.stride)
        self.line = CourseLine(src, 8)
        posts_h = np.empty((2 * E * G, 3), np.float32)
        posts_h[0::2], posts_h[1::2] = _host(src.left), _host(src.right)
        self.posts = _dev(posts_h, wp.vec3f)
        self.post_count = _i32(2 * np.minimum(_host(src.count), G))
        self.discs = DiscChecker(self.posts, radius=0.04, max_boxes=1,
                                 count=self.post_count, valid=src.valid, position=inp.boxes,
                                 orientation=inp.orient, half_extents=inp.he)
        self.cps = CheckpointSet.from_gates(src)
        self.tracker = ProgressTracker(self.cps, position=inp.pos)
        self.frames = FrameChecker(src, E, radius=0.05, frame_thickness=0.02,
                                   frame_depth=0.02)
        self.frames.bind_inputs(inp.pos)
        self.frames.bind_window(self.tracker.result.next_checkpoint)
        self.line.refresh()
        wp.synchronize()
        self.loc = TrackLocalizer(self.line.centerline, warm_window=2, position=inp.pos)

    def run(self, ctx) -> None:
        self.line.refresh(ctx=ctx)
        self.discs.query(ctx=ctx)
        self.tracker.update(ctx=ctx)
        self.frames.query(ctx=ctx)
        self.loc.query(ctx=ctx)

    def written(self) -> dict:
        return {**_arrays(self.line.centerline, "line"),
                **_arrays(self.discs.result, "discs"),
                **_arrays(self.tracker.result, "events"),
                **_arrays(self.frames.result, "frames"),
                **_arrays(self.loc.result, "frame")}

    def outputs(self) -> dict:
        t, loc = self.tracker, self.loc
        return {**self.written(), "progress._prev_pos": t._prev_pos,
                "progress._next": t._next, "progress._laps": t._laps,
                "progress._progress": t._progress, "progress.seen": _seen(t),
                "localizer._last": loc._last, "localizer.seen": _seen(loc)}


def _assert_track_sentinels(u: _TrackUtilities, label: str) -> None:
    J = np.flatnonzero(_FAIL)
    for name, c in (("segments", u.seg.result), ("sdf", u.sdf.result)):
        oob, bnd = _rows(c.oob), _rows(c.boundary)
        # Slot 0 is always active: oob = 1, boundary = -1, NaN geometry.
        assert (oob[J, 0] == 1).all() and (bnd[J, 0] == -1).all(), f"{label}: {name} oob"
        for f in ("distance", "nearest", "normal"):
            assert np.isnan(_rows(getattr(c, f))[J, 0]).all(), f"{label}: {name}.{f}"
    assert (_host(u.sdf._sdf_lo)[J] == -1.0).all(), f"{label}: sdf lo"
    assert (_host(u.sdf._sdf_hi)[J] == 1.0).all(), f"{label}: sdf hi"
    assert np.isnan(_rows(u.sdf._sdf_phi)[J]).all(), f"{label}: sdf phi"
    assert (_rows(u.sdf._sdf_bid)[J] == -1).all(), f"{label}: sdf bid"
    p = u.props.result
    assert (_host(p.count)[J] == 0).all() and np.isnan(_host(p.step)[J]).all(), label
    assert (_host(p.truncated)[J] == 0).all(), f"{label}: props truncated"
    assert np.isnan(_rows(p.position)[J]).all(), f"{label}: props rows"
    cps = u.sampler.result
    assert (_host(cps.count)[J] == 0).all() and (_host(cps.valid)[J] == 0).all(), label
    assert np.isnan(_rows(cps.position)[J]).all(), f"{label}: checkpoint rows"
    assert np.isnan(_host(u.sampler.step)[J]).all(), f"{label}: checkpoint step"
    _assert_frozen_events(u.tracker, J, label)
    _assert_nan_frame(u.loc, J, label)
    hf = u.hf.result
    assert np.isnan(_rows(hf.height)[J]).all(), f"{label}: heights"
    assert (_host(hf.lo)[J] == -1.0).all() and (_host(hf.hi)[J] == 1.0).all(), label
    for name, a in (("curvature", u.kappa), ("speed", u.speed)):
        assert np.isnan(_rows(a)[J]).all(), f"{label}: {name} rows"


def _assert_frozen_events(tracker, J, label: str) -> None:
    ev = tracker.result
    for f, want in (("passed", 0), ("checkpoint_passed", -1), ("wrong_way", 0),
                    ("wrong_checkpoint", -1), ("dist_to_next", 0.0)):
        assert (_host(getattr(ev, f))[J] == want).all(), f"{label}: events.{f}"
    for f, state in (("next_checkpoint", tracker._next), ("laps", tracker._laps),
                     ("progress", tracker._progress)):
        _assert_bytes(_host(getattr(ev, f))[J], _host(state)[J], f"{label}: frozen {f}")
    assert np.isnan(_host(tracker._prev_pos)[J]).all(), f"{label}: prev_pos re-armed"


def _assert_nan_frame(loc, J, label: str) -> None:
    f = loc.result
    for name in ("s", "n", "n_up"):
        assert np.isnan(_host(getattr(f, name))[J]).all(), f"{label}: frame.{name}"
    assert (_host(f.segment)[J] == -1).all() and (_host(loc._last)[J] == -1).all(), label


def _assert_gate_sentinels(u: _GateUtilities, label: str) -> None:
    J = np.flatnonzero(_FAIL)
    c = u.line.centerline
    assert (_host(c.count)[J] == 0).all() and (_host(c.valid)[J] == 0).all(), label
    assert np.isnan(_rows(c.center)[J]).all(), f"{label}: line rows"
    d = u.discs.result
    assert (_host(d.hit)[J] == 0).all(), f"{label}: discs hit"
    assert (_host(d.disc)[J] == -1).all(), f"{label}: discs disc"
    assert (_host(d.depth)[J] == 0.0).all() and np.isnan(_rows(d.nearest)[J]).all(), label
    fr = u.frames.result
    assert (_host(fr.hit)[J] == 0).all() and (_host(fr.depth)[J] == 0.0).all(), label
    _assert_frozen_events(u.tracker, J, label)
    _assert_nan_frame(u.loc, J, label)


@pytest.mark.parametrize("kind", ["track", "gates"])
def test_invalid_envs_read_like_poisoned_ones_under_capture(kind):
    """Every utility built on K and on P (see :func:`_keep_and_poison`), all recorded in
    ONE caller graph (with the ``ctx=`` forms) and replayed along a 24-step trajectory
    (inputs from K's geometry, written identically for both): every owned output
    (results and runtime state) is byte-identical on K and P every step, equal to an
    eager run on K, and the failed envs hold the sentinel table's values. The result
    arrays are poisoned before every replay / eager run, in all three runs (a write
    missing from the graph shows; a legitimately unwritten slot keeps the same poison
    everywhere). Non-vacuity: the valid envs' outputs are not sentinels."""
    _load_test_kernels()
    if kind == "track":
        src = _track_gen(11, z_profile="uniform", z_min=0.0, z_max=0.6).program.result
        make, check, inp = _TrackUtilities, _assert_track_sentinels, _Inputs(2)
    else:
        src = _gate_gen(5).program.result
        make, check, inp = _GateUtilities, _assert_gate_sentinels, _Inputs(1)
    K, P = _keep_and_poison(src)
    gates = kind == "gates"
    inp.write(K, 0, gates=gates)
    uk, up, ue = make(K, inp), make(P, inp), make(K, inp)
    s = _side_stream()
    wp.synchronize()
    for u in (uk, up, ue):                       # warm-up: loads every module
        u.run(LaunchContext.for_call(DEV, s))
    wp.synchronize()
    _assert_state(uk.outputs(), up.outputs(), f"{kind}: K vs P after the warm-up")
    with _recording(s) as cap:
        ctx = LaunchContext.for_call(DEV, s)
        uk.run(ctx)
        up.run(ctx)
    eager = LaunchContext.for_call(DEV)
    finite = 0
    for step in range(1, 25):
        inp.write(K, step, gates=gates)
        _poison([*uk.written().values(), *up.written().values(), *ue.written().values()])
        _replay(cap.graph, s)
        ue.run(eager)
        wp.synchronize()
        label = f"{kind}: step {step}"
        _assert_state(uk.outputs(), up.outputs(), f"{label}: K vs P (captured)")
        _assert_state(uk.outputs(), ue.outputs(), f"{label}: captured vs eager")
        check(uk, label)
        finite += int(np.isfinite(_host(uk.loc.result.s))[_FAIL == 0].sum())
    assert finite > 0, f"{kind}: the valid envs never localized: vacuous comparison"


# ---------------------------------------------------------------------------------------
# 4. Generation-aware self-reset in a caller graph
# ---------------------------------------------------------------------------------------

class _SelfResetRig:
    """A producer with its stateful consumers, driven by device kernels only:
    track: ``CheckpointSampler`` (owned set ``generation``) + ``ProgressTracker`` + warm
    ``TrackLocalizer`` on the ``Track``; gates: ``from_gates`` + ``ProgressTracker`` +
    ``CourseLine`` (owned ``generation``) + warm ``TrackLocalizer`` on its line."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        if kind == "track":
            self.gen = _track_gen(3)
            self.src = self.gen.program.result
            self.sampler = CheckpointSampler(self.src, 0.5, max_checkpoints=64)
            self.cps = self.sampler.sample()
            self.line = None
            line = self.src
        else:
            self.gen = _gate_gen(4, generator="checkpoint")
            self.src = self.gen.program.result
            self.sampler = None
            self.cps = CheckpointSet.from_gates(self.src)
            self.line = CourseLine(self.src, 8)
            self.line.refresh()
            line = self.line.centerline
        wp.synchronize()
        self.mask = wp.zeros(E, dtype=wp.int32, device=DEV)
        self.counter = wp.zeros(1, dtype=wp.int32, device=DEV)
        self.pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
        self.orient = wp.zeros(E, dtype=wp.quatf, device=DEV)
        speed = (1 + np.arange(E) % 4).astype(np.int32)
        self.speed = wp.array(speed, dtype=wp.int32, device=DEV)
        self.offset = wp.zeros(E, dtype=wp.float32, device=DEV)
        self.t = wp.zeros(E, dtype=wp.int32, device=DEV)
        self.steps = wp.zeros(E, dtype=wp.int32, device=DEV)
        self.line_arrays = line
        self.n_max = int(line.center.shape[0]) // E
        wp.synchronize()
        self.tracker = ProgressTracker(self.cps, position=self.pos)
        self.loc = TrackLocalizer(line, warm_window=3, position=self.pos)

    def step(self, ctx, *, explicit_reset: bool, eager: bool) -> None:
        """move -> update / query -> done schedule -> masked advancing generation ->
        refresh (-> explicit resets)."""
        s = ctx.stream
        wp.launch(_move_agents_k, E, [self.line_arrays.center, self.line_arrays.count,
                                      self.n_max, self.speed, self.offset, self.t,
                                      self.steps, self.pos, self.orient],
                  device=DEV, stream=s)
        self.tracker.update(ctx=ctx)
        self.loc.query(ctx=ctx)
        wp.launch(_schedule_k, E, [self.counter, 9, self.mask], device=DEV, stream=s)
        wp.launch(_bump_k, 1, [self.counter], device=DEV, stream=s)
        if eager:
            self.gen.enqueue(advance=True, mask=self.mask, stream=s)
        else:
            self.gen.generate(advance=True, mask=self.mask, stream=s)
        if self.sampler is not None:
            self.sampler.sample(ctx=ctx)
        else:
            self.line.refresh(ctx=ctx)
        if explicit_reset:
            self.tracker.reset(self.mask, ctx=ctx)
            self.loc.reset(self.mask, ctx=ctx)

    def state(self) -> dict:
        """Everything but the per-env runtime state: geometry, set, line, the step
        outputs, the seen generations, the schedule and the agents."""
        t, loc = self.tracker, self.loc
        out = {**_arrays(self.src, "result"), **_arrays(self.cps, "checkpoints"),
               **_arrays(t.result, "events"), **_arrays(loc.result, "frame"),
               "progress.seen": _seen(t), "localizer.seen": _seen(loc), "mask": self.mask,
               "counter": self.counter, "pos": self.pos, "agent.t": self.t}
        if self.line is not None:
            out.update(_arrays(self.line.centerline, "line"))
        return out

    def runtime(self) -> dict:
        """The per-env runtime state a reset clears."""
        t = self.tracker
        return {"progress._prev_pos": t._prev_pos, "progress._next": t._next,
                "progress._laps": t._laps, "progress._progress": t._progress,
                "localizer._last": self.loc._last}


@pytest.mark.parametrize("kind", ["track", "gates"])
def test_self_reset_in_a_caller_graph_equals_explicit_resets(kind):
    """``[move -> tracker.update() -> localizer.query() -> device done schedule ->
    generate(advance=True, mask=done) -> sample() / refresh()]`` recorded in ONE caller
    graph WITHOUT any reset, replayed 60 times, equals an eager twin that resets the
    masked envs explicitly after each regeneration (today's contract): the progress and
    warm-localizer state of a regenerated env resets on device at its NEXT update / query
    because its ``generation`` (the sampler's owned copy in track mode, the gates' own
    array through ``from_gates`` and the course line's owned copy in gates mode) changed.
    Every step: the events / frame (poisoned before every replay), geometry, sets, line,
    ``seen`` generations and agents are byte-identical, and so is the runtime state of
    every env not regenerated in that step (a regenerated env's state is the twin's
    one step later, after the self-reset; no env is regenerated two steps in a row).
    Non-vacuity: envs that had made progress / held a warm segment were regenerated."""
    _load_test_kernels()
    rig, twin = _SelfResetRig(kind), _SelfResetRig(kind)
    s = _side_stream()
    wp.synchronize()
    # Warm-up outside the capture: one step on each (the schedule masks no env at
    # k = 0, so the masked generation is an exact no-op); both counters are then at 1.
    rig.step(LaunchContext.for_call(DEV, s), explicit_reset=False, eager=True)
    twin.step(LaunchContext.for_call(DEV), explicit_reset=True, eager=True)
    wp.synchronize()
    assert not _host(rig.mask).any(), "the warm-up step must mask no env"
    _assert_state({**rig.state(), **rig.runtime()}, {**twin.state(), **twin.runtime()},
                  f"{kind}: after the warm-up")
    with _recording(s) as cap:
        rig.step(LaunchContext.for_call(DEV, s), explicit_reset=False, eager=False)
    eager = LaunchContext.for_call(DEV)
    with_progress = with_segment = 0
    prev_mask = np.zeros(E, bool)
    for k in range(60):
        # Progress only grows between resets: > 0 here means > 0 at this step's
        # regeneration (after its update).
        prog_before = _host(twin.tracker._progress).copy()
        last_before = _host(twin.loc._last).copy()
        _poison([*_arrays(rig.tracker.result, "e").values(),
                 *_arrays(rig.loc.result, "f").values()])
        _replay(cap.graph, s)
        twin.step(eager, explicit_reset=True, eager=True)
        wp.synchronize()
        m = _host(twin.mask) != 0
        with_progress += int((m & (prog_before > 0)).sum())
        with_segment += int((m & (last_before >= 0)).sum())
        label = f"{kind}: replay {k} (mask {m.astype(int)})"
        assert not (m & prev_mask).any(), f"{label}: an env regenerated two steps in a row"
        _assert_state(rig.state(), twin.state(), label)
        got, ref = rig.runtime(), twin.runtime()
        for name in ref:
            _assert_valid_rows(got[name], ref[name], ~m, f"{label}: {name}")
        prev_mask = m
    assert with_progress > 0, "no regenerated env had made progress: self-reset unexercised"
    assert with_segment > 0, "no regenerated env held a warm segment"
    assert _host(rig.gen.seeds.epoch).max() >= 2


# ---------------------------------------------------------------------------------------
# 5. curvature(out=) / speed_profile(out=) in a caller graph
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["track", "centerline", "minimal"])
def test_curvature_and_speed_profile_out_forms_record_into_a_caller_graph(source):
    """``[generate(advance=True) (-> course line refresh) -> curvature(src, 2, ctx,
    out=k, scratch=sc) -> speed_profile(src, ..., kappa=k, ctx=ctx, out=v)]`` recorded
    in ONE caller graph (mempool disabled: any allocation fails), replayed 4 times into
    poisoned outputs: each replay equals the eager ALLOCATING forms on an eager twin
    advanced alike, and changes the outputs (a new batch per replay). Sources: a
    ``Track``, a ``Centerline`` (``CourseLine``) and a ``MinimalCenterline`` (no
    ``valid``, no ``layout``)."""

    def build():
        if source == "centerline":
            gen = _gate_gen(5)
            line = CourseLine(gen.program.result, 8)
            line.refresh()
            wp.synchronize()
            return gen, line, line.centerline
        gen = _track_gen(11)
        t = gen.program.result
        return gen, None, (MinimalCenterline(t) if source == "minimal" else t)

    gen, line, src = build()
    twin, tline, tsrc = build()
    N = int(src.center.shape[0])
    k, sc, v = (wp.zeros(N, dtype=wp.float32, device=DEV) for _ in range(3))
    s = _side_stream()
    wp.synchronize()
    ctx = LaunchContext.for_call(DEV, s)
    curvature(src, 2, ctx, out=k, scratch=sc)                          # warm-up
    speed_profile(src, 4.0, 2.0, 3.0, 5.0, kappa=k, ctx=ctx, out=v)
    wp.synchronize()
    with _recording(s) as cap:
        gen.generate(advance=True, stream=s)
        if line is not None:
            line.refresh(ctx=ctx)
        assert curvature(src, 2, ctx, out=k, scratch=sc) is k
        assert speed_profile(src, 4.0, 2.0, 3.0, 5.0, kappa=k, ctx=ctx, out=v) is v
    prev = None
    for r in range(4):
        _poison([k, sc, v])
        _replay(cap.graph, s)
        twin.enqueue(advance=True)
        if tline is not None:
            tline.refresh()
        kt = curvature(tsrc, 2)
        vt = speed_profile(tsrc, 4.0, 2.0, 3.0, 5.0, kappa=kt)
        wp.synchronize()
        _assert_bytes(k, kt, f"{source}: replay {r}: curvature(out=)")
        _assert_bytes(v, vt, f"{source}: replay {r}: speed_profile(out=)")
        if prev is not None:
            assert _host(k).tobytes() != prev, f"{source}: replay {r} did not regenerate"
        prev = _host(k).tobytes()


# ---------------------------------------------------------------------------------------
# 6. check_component for every built-in adapter on cuda:0
# ---------------------------------------------------------------------------------------

def _builtin(cid: str):
    """``(factory, upstream, source)`` of a built-in component entry."""
    factory, upstream, source = _builtin_components()[cid]
    return factory, tuple(upstream), source


@pytest.mark.parametrize("cid", sorted(_builtin_components()))
def test_builtin_component_passes_check_component_on_cuda(cid):
    """``check_component(..., devices=["cuda:0"])`` passes for every built-in adapter:
    ``capture.caller`` ran (Warp mode ok; the torch modes ok, or skipped without torch
    with CUDA), and every stateful adapter (``reset.exact`` applies) ran
    ``capture.self_reset`` (every built-in stateful adapter reads a ``generation``)."""
    factory, upstream, source = _builtin(cid)
    report = check_component(factory, source=source, upstream=upstream, devices=[DEV])
    report.raise_for_failures()
    cap = report.result("capture.caller", DEV)
    assert not cap.skipped, str(report)
    assert cap.details.get("warp") == "ok", str(report)
    for mode in ("external", "torch"):
        got = str(cap.details.get(mode))
        assert got == "ok" or got.startswith("skipped"), f"{mode}: {got}\n{report}"
    if not report.result("reset.exact", DEV).skipped:
        assert not report.result("capture.self_reset", DEV).skipped, str(report)


@pytest.mark.parametrize("toy", list(_toys.GENERATION_WRITTEN), ids=lambda c: c.__name__)
def test_capture_caller_passes_a_component_with_an_array_only_on_generate_writes(toy):
    """A correct component whose result holds an array that only ``on_generate`` writes
    (for the whole batch): ``DuckPerGeneration`` (duck-typed, so counted as per-step,
    its ``step`` returning the result unchanged) and ``PerStepWithTable`` (a per-step
    ``value`` plus a per-generation ``table``). ``capture.caller`` in the ``warp`` mode
    passes: where conditional nodes are supported, the replay skips ``on_generate`` on
    every all-zero ``done`` (the poison stays in that array) and the kit's eager twin
    skips it too; elsewhere both run it."""
    report = check_component(lambda cc, t=toy: t(), source="track", devices=[DEV],
                             modes=("warp",))
    report.raise_for_failures()
    cap = report.result("capture.caller", DEV)
    assert not cap.skipped and cap.details.get("warp") == "ok", str(report)


# ---------------------------------------------------------------------------------------
# 7. RebindAfterCaptureWarning with real captures; the probe's host cost
# ---------------------------------------------------------------------------------------

def _bound_track_course():
    course = Course(_twin_cfg("track"))
    agents = _Agents(course, "track")
    agents.bind(course)
    return course, agents


def _other_buffers(agents: _Agents):
    return (wp.zeros(E, dtype=wp.vec3f, device=DEV), wp.clone(agents.orient),
            wp.clone(agents.he))


@contextlib.contextmanager
def _no_rebind_warning():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        yield
    bad = [w for w in rec if issubclass(w.category, RebindAfterCaptureWarning)]
    assert not bad, [str(w.message) for w in bad]


def _one_rebind_warning(rec) -> None:
    hits = [w for w in rec if issubclass(w.category, RebindAfterCaptureWarning)]
    assert len(hits) == 1, [str(w.message) for w in hits]
    assert hits[0].filename == __file__, f"warned at {hits[0].filename}:{hits[0].lineno}"
    assert issubclass(RebindAfterCaptureWarning, UserWarning)


@pytest.mark.parametrize("mode", ["warp", "torch"])
def test_course_rebind_after_a_real_capture_warns_once(mode):
    """``course.step()`` recorded into a real caller capture (Warp ``ScopedCapture`` on a
    side stream, or ``torch.cuda.graph`` adopting torch's stream): rebinding the SAME
    arrays is silent; rebinding OTHER arrays warns ``RebindAfterCaptureWarning`` exactly
    once, at the caller's line; the flag is then clear (rebinding again is silent until a
    new recording). Eager steps never arm it."""
    course, agents = _bound_track_course()
    course.step()
    wp.synchronize()
    pos2, orient2, he2 = _other_buffers(agents)
    with _no_rebind_warning():                    # eager only: never armed
        course.bind(position=pos2, orientation=orient2, half_extents=he2)
        agents.bind(course)
    _caller_graph(mode, lambda ks, ls: course.step(stream=ls), owners=(course,))
    with _no_rebind_warning():
        agents.bind(course)                       # the same arrays: silent
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        course.bind(position=pos2, orientation=orient2, half_extents=he2)
    _one_rebind_warning(rec)
    with _no_rebind_warning():                    # cleared by the warning
        agents.bind(course)


def test_utility_and_component_rebind_after_a_real_capture_warn_once():
    """The same contract on a legacy utility (``ProgressTracker.bind``) and on a
    component (``tools.Progress`` attached through ``ComponentContext``: ``Component.bind``
    warns at its caller's line, once, not once per nested utility)."""
    gen = _track_gen(3)
    t = gen.program.result
    pos = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    pos2 = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    wp.synchronize()
    tracker = ProgressTracker(CheckpointSampler(t, 0.5, max_checkpoints=48).sample(),
                              position=pos)
    tracker.update()
    cc = ComponentContext.for_producer(gen)
    cc.add(tools.Checkpoints(0.5, max_checkpoints=48), t)
    prog = cc.add(tools.Progress(), t)
    prog.bind(position=pos)
    s = _side_stream()
    wp.synchronize()
    with _recording(s):
        ctx = LaunchContext.for_call(DEV, s)
        tracker.update(ctx=ctx)
        assert prog.step(ctx) is prog.result
    with _no_rebind_warning():
        tracker.bind(pos)
        prog.bind(position=pos)
    for rebind in (lambda: tracker.bind(pos2), lambda: prog.bind(position=pos2)):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            rebind()
        _one_rebind_warning(rec)


def _time_steps(course, n: int) -> float:
    """Host nanoseconds per ``course.step()`` over ``n`` eager calls (queued, not
    synchronized inside the timed loop)."""
    t0 = time.perf_counter_ns()
    for _ in range(n):
        course.step()
    dt = time.perf_counter_ns() - t0
    wp.synchronize()
    return dt / n


@pytest.mark.parametrize("variant", ["track", "gates-posts"])
def test_rebind_probe_host_cost_on_course_step(variant):
    """S3: the rebind probe (``_host_blocked(ctx)`` in the hot methods of the bound
    utilities while their ``_recorded`` flag is unarmed) costs less than 2 % of
    ``Course.step``'s host time. Measured A/B on the SAME course: flags unarmed (the
    probe runs in every hot method, the eager steady state) vs armed (no probe: the
    documented "once armed, no further probe"), 15 interleaved trials of 400 steps, the
    minimum per-call time of each side (the cost floor; robust to scheduler noise); the
    median of three such measurements is asserted. Prints the ratio, every attempt and an
    isolated estimate (probe calls per step x one probe's cost / one step's cost)."""
    course = Course(_twin_cfg(variant))
    agents = _Agents(course, variant)
    agents.bind(course)
    utils = [u for u in (course.progress, course.collision, course.localizer)
             if u is not None]
    assert utils and all(hasattr(u, "_recorded") for u in utils), utils
    flags = [u._recorded for u in utils]
    assert not any(flags), "an eager-only course must not be armed"
    N, TRIALS = 400, 15

    def measure() -> "tuple[float, float]":
        best = {False: math.inf, True: math.inf}
        for trial in range(TRIALS):
            for armed in ((False, True) if trial % 2 == 0 else (True, False)):
                for u in utils:
                    u._recorded = armed
                best[armed] = min(best[armed], _time_steps(course, N))
        return best[False], best[True]

    gc_was = gc.isenabled()
    gc.disable()
    try:
        _time_steps(course, 100)                   # warm-up
        ratios = []
        for _ in range(3):
            unarmed, armed = measure()
            ratios.append((unarmed - armed) / armed)
        ctx = LaunchContext.for_call(DEV)
        t0 = time.perf_counter_ns()
        for _ in range(20000):
            exec_context._host_blocked(ctx)
        probe = (time.perf_counter_ns() - t0) / 20000
    finally:
        for u, f in zip(utils, flags):
            u._recorded = f
        if gc_was:
            gc.enable()
    isolated = len(utils) * probe / armed
    ratio = float(np.median(ratios))
    print(f"\n[rebind probe, {variant}] Course.step host time: {armed / 1e3:.2f} us armed, "
          f"{unarmed / 1e3:.2f} us unarmed; A/B probe cost ratio (median of 3) "
          f"{ratio * 100:.3f} % (attempts {[f'{r * 100:.3f} %' for r in ratios]}); "
          f"isolated estimate {len(utils)} x {probe:.0f} ns = {isolated * 100:.3f} % of a "
          f"step")
    assert ratio < 0.02, (
        f"the rebind probe costs {ratio * 100:.2f} % of Course.step's host time (> 2 %): "
        f"fall back to A.7's alternative (docs only)")
