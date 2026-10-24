"""CUDA-only: Course as the composition root, regeneration inside the caller's graph
(roadmap PR15; design section 7.4 and the S1 / S3 gates of its section 15).

What this module proves on ``cuda:0`` (the CPU proofs are
``tests/test_course_composition.py``; the CPU lane only collects this module):

1. **The roadmap gate.** ONE caller graph per RL step, ``[physics stub ->
   res = course.step() -> device done rule (a lap, a per-env step budget, or
   ``res.valid == 0``) into course.inputs["regen_mask"] -> course.regenerate(
   mask=course.inputs["regen_mask"], advance=True)]``, replayed 200 times with no host
   involvement, equals an eager twin EVERY step, bitwise, for a composed track course
   (``Course.track`` + segments collision + a warm ``tools.Localizer`` + the third-party
   ``CurvatureLookahead`` toy, public imports only, sorted after the localizer it reads)
   and a gates course (``Course.gates`` with posts and a warm localizer on the course
   line), in four capture modes: a Warp ``ScopedCapture`` on a side stream (``warp``), a
   plain ``torch.cuda.graph`` (``torch``), and ``torch.cuda.graph`` registered with Warp
   through ``track_gen.interop.torch.registered(graph=g)`` (``external``) or
   ``registered(graph=g, conditional=True)`` (``external-conditional``). The masked
   regeneration is ONE conditional node keyed on the course's flag where conditional
   nodes apply (``warp``, ``external-conditional``) and inline elsewhere; the course is
   anchored on the Warp graph (``warp``) or the registration's graph retained on ``g``
   (``external*``).
2. **The keyed self-graph with a third-party component.** Outside any capture,
   ``regenerate(mask=inputs["regen_mask"], advance=True)`` captures ONE graph (one
   conditional node on the course's flag where supported, the toy's ``on_generate``
   inside its body) and replays it on 40 calls with host-written masks (zero masks
   included): equal to the eager twin every call; ``regenerate()`` adds the plain graph.
3. **Both lowerings of the zero-mask body** (design section 2.7): recorded with and
   without ``track_gen.exec.unconditional()``, replayed 5 times: every buffer unchanged;
   a poisoned derived buffer tells the lowerings apart (skipped on device: the poison
   stays; inline: recomputed to its exact bytes).
4. **Lifetime (G6).** A Warp graph that recorded ``[step, regenerate(mask)]`` keeps the
   course alive after every other reference is dropped, its replays keep regenerating
   (== an eager twin), and the course is released with the graph. A graph that recorded
   ``[step, reset(m), regenerate(mask=m')]`` also anchors the binding ``step`` read and
   the caller's masks: after a rebind and dropping every caller reference they stay
   alive while the graph replays, and are freed with it.
5. **The conditional-body contract (C.2).** A component whose ``on_generate`` emits an
   operation a CUDA conditional-node body cannot hold fails ``check_component``'s
   ``capture.caller`` (``warp`` mode; it passes every other check), and the course's
   keyed self-graph and a caller's ``ScopedCapture`` refuse it with a ``CaptureError``
   naming the component; under ``exec.unconditional()`` the same course regenerates
   (== an eager twin). Two offences: a copy from a PINNED host array (the design's toy)
   and an external event record.
6. **The default-stream guard (B.11).** A course with a default stream
   (``Course.track(cfg, stream=s)``, or ``Course(TrackGenerator(cfg, stream=s), ...)``)
   runs ``step`` / ``reset`` / ``enqueue_generate`` / the self-graph replays on ``s``;
   called without ``stream=`` while Warp's current stream or torch is capturing, every
   public call raises ``CaptureError`` before any launch and leaves the capture usable;
   with ``stream=`` it records, and the replays equal an eager twin.
7. **The cold attach phase on Warp's current stream** (design section 1.1 step 5): a
   course over a generator with a default stream ``s``, built while Warp's current
   stream is busy (a long kernel), with a component whose ``attach`` fills a table with
   a kernel (state written ONLY at attach), equals the same course built on the current
   stream, bitwise (every component result and runtime state, the step outputs). A
   control attached through a context on ``s`` races (the late memset zeroes the table).
   A course built with ``stream=s1`` over a generator whose default stream ``s2`` still
   runs an advancing generation makes ``s1`` wait for ``s2`` first, and equals a twin
   built after that generation finished.
8. **S3**: the host cost of ``Course.step()``'s anchor probe and default-stream guard,
   A/B against no-ops, printed as ratios (no assertion).

Method (as in tests/test_components_cuda.py and tests/test_inplace_regen_cuda.py; the
helpers are :mod:`track_gen.testing.capture`'s): record on a NON-current side stream,
with the Warp mempool disabled (an allocation inside a capture fails loudly) and
``wp.capture_launch`` replaced by a tripwire; poison with FINITE, dtype-aware sentinels,
only the step outputs every ``step()`` rewrites for every env (never ``StepResult.valid``
or any other producer state: the masked commit restores unmasked rows, so a poison there
would persist); the test's kernels launch with an explicit ``stream=`` (a bare
``wp.launch`` would go to Warp's current stream and escape the graph); warm every
recorded path up outside the capture; every array is allocated with an explicit device
(Warp's default device is ``cuda:0`` on a GPU machine). Comparisons are BYTE comparisons
(NaN payloads and signed zeros included), with a value diff in the message.
"""
from __future__ import annotations

import contextlib
import gc
import math
import re
import sys
import time
import warnings
import weakref

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

import tests._component_toys as _toys  # noqa: E402
from tests._component_toys import CurvatureLookahead  # noqa: E402
from track_gen import GateGenConfig, TrackGenConfig, TrackGenerator, tools  # noqa: E402
from track_gen.course import Course  # noqa: E402
from track_gen.exec import (CaptureError, LaunchContext,  # noqa: E402
                            RebindAfterCaptureWarning, resolve_stream, unconditional)
from track_gen.exec import _compat as tg_compat  # noqa: E402
from track_gen.testing import check_component, trace_launches  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402

DEV = "cuda:0"
E = 8
_SPACING = 0.6
_TOY = "acme.lookahead"
_RL_STEPS = 200
_VARIANTS = ("track", "gates")
#: check_component's capture-mode names, plus the conditional opt-in of ``registered``.
_MODES = ("warp", "torch", "external", "external-conditional")
#: Execution order of the components (the toy is listed BEFORE the localizer it reads).
_ORDER = {"track": ["checkpoints", "progress", "collision", "localizer", _TOY],
          "gates": ["checkpoints", "centerline", "progress", "collision", "localizer"]}
#: StepResult entries: the components that override ``step``, in execution order.
_ENTRIES = {"track": ["progress", "collision", "localizer", _TOY],
            "gates": ["progress", "collision", "localizer"]}
#: Sideways offsets of the agents (plan view): track half width 0.1, so envs 3 and 5
#: drive out of bounds; gate half width 0.1, posts radius 0.02: envs 4 and 5 graze posts.
_OFFSETS = {"track": np.float32([0.0, 0.03, 0.06, -0.13, 0.0, 0.13, -0.06, 0.0]),
            "gates": np.float32([0.0, 0.02, -0.03, 0.0, 0.075, -0.075, 0.0, 0.01])}
_ZERO = np.zeros(E, np.int32)
#: Masks of host-written regenerations (neighbours differ: a write that leaks across an
#: env boundary shows up).
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)
_M2 = np.array([0, 1, 1, 0, 1, 0, 0, 1], np.int32)
#: Runtime state of the wrapped utilities (whichever exist on a component's utility).
_UTILITY_STATE = ("_prev_pos", "_next", "_laps", "_progress", "_seen", "_last",
                  "truncated", "step", "_count")
#: Derived buffers and scratch of the components (the toy's curvature, the posts).
_COMPONENT_STATE = ("_kappa", "_scratch", "posts")

_NO_CONDITIONAL = (
    "CUDA conditional graph nodes are unsupported here "
    "(wp.is_conditional_graph_supported() is False: Warp must be built with CUDA Toolkit "
    ">= 12.4 and run on a >= 12.4 driver)")


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
    # Stub physics: each agent advances `speed[e]` samples along its env's CURRENT line
    # (a regenerated env is followed on its new line from the next step), at the midpoint
    # of two consecutive samples, `offset[e]` to the side (plan view), its box yawed
    # along the motion.
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
def _fill_table_k(table: wp.array(dtype=wp.float32)):
    # Never 0: a late memset of the table shows.
    i = wp.tid()
    table[i] = 1.0 + 0.5 * float(i)


@wp.kernel
def _spin_k(n: int, out: wp.array(dtype=wp.float32)):
    # Keeps one stream busy for a host-calibrated time.
    i = wp.tid()
    x = float(i) * 1.0e-3
    for k in range(n):
        x = wp.sin(x) + 1.0e-3
    out[i] = x


def _load_test_kernels() -> None:
    """Load this module's kernels on ``DEV`` outside any capture (scratch buffers)."""
    i = wp.zeros(E, dtype=wp.int32, device=DEV)
    f = wp.zeros(E, dtype=wp.float32, device=DEV)
    v = wp.zeros(E, dtype=wp.vec3f, device=DEV)
    q = wp.zeros(E, dtype=wp.quatf, device=DEV)
    wp.launch(_move_agents_k, E, inputs=[v, i, 1, i, f, i, i, v, q], device=DEV)
    wp.launch(_done_k, E, inputs=[i, i, i, i, i, i, i], device=DEV)
    wp.launch(_fill_table_k, E, inputs=[f], device=DEV)
    wp.launch(_spin_k, 1, inputs=[1, f], device=DEV)
    wp.synchronize()


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _host(x) -> np.ndarray:
    """A contiguous host COPY."""
    return np.array(x.numpy() if isinstance(x, wp.array) else x, copy=True, order="C")


def _dev(values, dtype) -> "wp.array":
    """An array on ``DEV`` from host values (uploaded on Warp's current stream, then
    synchronized: side-stream readers need it done)."""
    a = wp.array(np.asarray(values), dtype=dtype, device=DEV)
    wp.synchronize()
    return a


def _write(dst: "wp.array", values) -> None:
    """Write host int32 values into a device buffer, outside any capture."""
    wp.copy(dst, _dev(np.asarray(values, np.int32), wp.int32))
    wp.synchronize()


def _zeros_i32() -> "wp.array":
    a = wp.zeros(E, dtype=wp.int32, device=DEV)
    wp.synchronize()
    return a


def _arrays(obj, prefix: str) -> dict:
    """``"<prefix>.<field>" -> wp.array`` for every array of a result (a dataclass, a
    mapping or a bare array); ``{}`` for ``None``."""
    return {f"{prefix}.{k}": a for k, a in tg_capture.result_arrays(obj).items()}


def _snapshot(d: dict) -> dict:
    return {k: _host(v) for k, v in d.items()}


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


def _side_stream() -> "wp.Stream":
    return tg_capture.side_stream(DEV)


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


def _course(variant: str, **kw) -> Course:
    """The composed courses of the roadmap gate (public API and imports only)."""
    if variant == "track":
        return Course.track(TrackGenConfig(num_envs=E, device=DEV),
                            checkpoint_spacing=_SPACING, collision="segments",
                            extra=[CurvatureLookahead(6, name=_TOY),
                                   tools.Localizer(warm_window=4)],
                            seeds=5, **kw)
    assert variant == "gates", variant
    # scale=3.0 keeps 0.2-wide gates valid; the "checkpoint" gate generator gives
    # sequences an agent following the course line laps in a few dozen steps.
    return Course.gates(GateGenConfig(num_envs=E, device=DEV, gate_width=0.2, scale=3.0,
                                      generator="checkpoint"),
                        collision=0.02, localize_window=4, seeds=5, **kw)


def _line(course: Course, variant: str):
    """The centerline the agents follow: the track, or the gates' course line."""
    return course.result if variant == "track" else course.components["centerline"].result


class _Agents:
    """The caller's per-env agent state and bound buffers (identical device logic for
    both twins)."""

    def __init__(self, course: Course, variant: str) -> None:
        self.line = _line(course, variant)
        self.n_max = int(self.line.center.shape[0]) // E
        # Below one checkpoint spacing per step (track: ~10 samples between checkpoints
        # at spacing 0.6; gates: 8 line samples per gate), so no checkpoint is jumped.
        speed = (2 + np.arange(E) % 5 if variant == "track" else 1 + np.arange(E) % 4)
        speed = speed.astype(np.int32)
        self.speed = wp.array(speed, dtype=wp.int32, device=DEV)
        self.offset = wp.array(_OFFSETS[variant], dtype=wp.float32, device=DEV)
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
        # ONE bind routes position to Progress, the Localizer and the toy (and, at one
        # box per env, to the collision's box_position).
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


def _step_outputs(sr) -> dict:
    """Every StepResult entry's arrays: what ``step()`` writes for every env (poisoned
    before a replay). ``StepResult.valid`` aliases the producer's ``valid``: never."""
    out = {}
    for name, res in sr.items():
        out.update(_arrays(res, f"step[{name}]"))
    return out


def _course_state(course: Course, sr=None) -> dict:
    """Every buffer a course step / regeneration produces: the result and its epochs,
    every component's result, derived buffers and scratch, every wrapped utility's
    runtime state (``seen`` generations included) and result; with ``sr``, the step
    outputs and ``StepResult.valid``."""
    out = {**_arrays(course.result, "result")}
    if course.seeds is not None:
        out["epoch"] = course.seeds.epoch
    for name, comp in course.components.items():
        out.update(_arrays(getattr(comp, "result", None), f"{name}.result"))
        for attr in _COMPONENT_STATE:
            a = getattr(comp, attr, None)
            if isinstance(a, wp.array):
                out[f"{name}.{attr}"] = a
        u = getattr(comp, "utility", None)
        if u is None:
            continue
        for attr in _UTILITY_STATE:
            a = getattr(u, attr, None)
            if isinstance(a, wp.array):
                out[f"{name}.utility.{attr}"] = a
    if sr is not None:
        out.update(_step_outputs(sr))
        out["step.valid"] = sr.valid
    return out


def _rl_state(course, agents: _Agents, done, sr) -> dict:
    return {**_course_state(course, sr), **agents.arrays(), "done": done}


@contextlib.contextmanager
def _regen_conditionals(course: Course):
    """Yield the list of ``wp.capture_if`` calls made in the block whose condition is
    the course's regeneration flag (a BOUNDED stage's windows use other flags)."""
    calls: list = []
    real = wp.capture_if
    flag = course._regen_flag

    def spy(condition, *args, **kwargs):
        if condition is flag:
            calls.append(kwargs.get("stream"))
        return real(condition, *args, **kwargs)

    wp.capture_if = spy
    try:
        yield calls
    finally:
        wp.capture_if = real


def _owners(obj) -> list:
    """The owners anchored / retained on a ``wp.Graph`` or a torch graph."""
    return list(getattr(obj, "_track_gen_owners", ()) or ())


def _caller_graph(mode: str, course: Course, record):
    """Record ``record(kernel_stream, library_stream)`` into ONE caller graph and return
    ``(replay, result, n_regen_conditionals, graph)`` (``graph``: the ``wp.Graph`` in
    ``warp`` mode, the ``torch.cuda.CUDAGraph`` otherwise).

    - ``"warp"``: ``wp.ScopedCapture`` on a side stream (both streams are it);
    - ``"torch"``: a plain ``torch.cuda.graph``: library calls get ``stream=None`` and
      adopt torch's capture stream; the test's kernels go to the stream
      ``track_gen.exec.resolve_stream`` adopts; the course is ``retain``-ed on ``g``;
    - ``"external"`` / ``"external-conditional"``: ``torch.cuda.graph`` with
      ``interop.torch.registered(graph=g[, conditional=True])``: the test's kernels go
      to the stream it yields, library calls get ``stream=None``; nothing is retained
      by hand (the registration's Warp graph is, with the course anchored on it).

    Every mode runs with the Warp mempool disabled and ``wp.capture_launch`` forbidden."""
    with _regen_conditionals(course) as calls:
        if mode == "warp":
            s = _side_stream()
            wp.synchronize()
            with tg_capture.recording(DEV, s) as cap:
                out = record(s, s)
            graph = cap.graph

            def replay(_g=graph, _s=s):
                _replay(_g, _s)

            return replay, out, len(calls), graph
        assert mode in ("torch", "external", "external-conditional"), mode
        torch = _torch_cuda()
        from track_gen.interop import torch as tg_torch

        g = torch.cuda.CUDAGraph()
        wp.synchronize()
        torch.cuda.synchronize()
        with tg_capture.mempool_disabled(DEV):
            with torch.cuda.graph(g):
                with tg_capture.capture_launch_forbidden():
                    if mode == "torch":
                        ks = resolve_stream(DEV)       # torch's capture stream, adopted
                        assert ks.is_capturing
                        out = record(ks, None)
                    else:
                        with tg_torch.registered(
                                graph=g, conditional=mode == "external-conditional") as ws:
                            assert isinstance(ws, wp.Stream) and ws.is_capturing
                            out = record(ws, None)
        torch.cuda.synchronize()
        if mode == "torch":
            tg_torch.retain(g, course)

    def replay_torch(_g=g, _torch=torch):
        _g.replay()
        _torch.cuda.synchronize()
        wp.synchronize()

    return replay_torch, out, len(calls), g


def _record_rl_step(course: Course, agents: _Agents, kstream, lstream):
    """One RL step: physics stub -> step() -> device done rule into the course's
    regen_mask -> masked advancing regeneration (the test's kernels on ``kstream``,
    library calls on ``lstream``)."""
    done = course.inputs["regen_mask"]
    agents.move(stream=kstream)
    sr = course.step(stream=lstream)
    agents.decide(sr["progress"].laps, sr.valid, done, stream=kstream)
    assert course.regenerate(mask=done, advance=True, stream=lstream) is course.result
    return sr


# ---------------------------------------------------------------------------------------
# 1. The roadmap gate: the RL graph replayed 200 steps == the eager twin
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_rl_graph_of_a_composed_course_equals_the_eager_twin(variant, mode):
    """ONE caller graph per RL step (physics stub -> ``res = course.step()`` -> device
    done rule reading ``res["progress"].laps`` and ``res.valid`` into
    ``course.inputs["regen_mask"]`` -> ``course.regenerate(mask=..., advance=True)``),
    recorded in ``mode`` and replayed 200 times with no host involvement, equals an
    eager twin driven step by step (its host calls ``enqueue_generate(mask=done,
    advance=True)`` only when an env is done, so the graph's zero-mask steps must be
    exact no-ops) EVERY step, bitwise, on every buffer of :func:`_rl_state`: the result
    and its epochs, every component's result (checkpoints, course line, posts, the toy's
    lookahead), the toy's curvature scratch, the progress / localizer runtime state
    (``seen`` generations included), the step outputs, ``StepResult.valid``, the agents
    and the done mask. Before each replay every StepResult entry is poisoned (a write
    missing from the graph shows); nothing that aliases producer state is.

    Also: the components run in sorted order (the toy after the localizer it reads), the
    StepResult entries are the per-step components with the 0.x fields among them,
    ``step()`` returns one StepResult, the track preset's checkpoint capacity is the
    plan bound; the recording executes nothing; the masked regeneration is ONE
    conditional node on the course's flag in ``warp`` / ``external-conditional`` (where
    conditional nodes are supported) and none in ``torch`` / ``external``; the course is
    anchored on the Warp graph (``warp``) or the registration graph retained on ``g``
    (``external*``). Non-vacuity: both done rules fired, steps with and without a
    regeneration, some env regenerated twice, the lookahead was finite; track: both oob
    values occurred; gates: some box hit a post."""
    _load_test_kernels()
    course, twin = _course(variant), _course(variant)
    assert list(course.components) == _ORDER[variant], list(course.components)
    if variant == "track":
        M = math.ceil(1.5 * 384 * 0.06 / _SPACING)
        assert course.checkpoints.position.shape[0] == E * M == E * 58, \
            "Checkpoints('auto') must take the plan bound"
    agents_c, agents_t = _Agents(course, variant), _Agents(twin, variant)
    agents_c.bind(course)
    agents_t.bind(twin)
    done_c = course.inputs["regen_mask"]       # the course's own stable mask buffer
    done_t = _zeros_i32()
    s = _side_stream()

    # Warm-up outside any capture: step() and the masked regeneration (zero mask: an
    # exact no-op; captures the course's keyed self-graph), then a full reset on both.
    wp.synchronize()
    sr_c = course.step(stream=s)
    assert course.regenerate(mask=done_c, advance=True, stream=s) is course.result
    sr_t = twin.step()
    wp.synchronize()
    course.reset(stream=s)
    twin.reset()
    wp.synchronize()
    assert list(sr_c.keys()) == _ENTRIES[variant], list(sr_c.keys())
    assert sr_c.events is sr_c["progress"] and sr_c.contacts is sr_c["collision"] \
        and sr_c.frame is sr_c["localizer"], "the 0.x fields are the named entries"
    assert "checkpoints" not in sr_c, "a per-generation component is not an entry"
    assert sr_c.valid.ptr == course.result.valid.ptr, "StepResult.valid aliases valid"
    _assert_state(_rl_state(course, agents_c, done_c, sr_c),
                  _rl_state(twin, agents_t, done_t, sr_t), f"{variant}: after warm-up")
    before = _snapshot(_rl_state(course, agents_c, done_c, sr_c))

    replay, sr, n_cond, graph = _caller_graph(
        mode, course, lambda ks, ls: _record_rl_step(course, agents_c, ks, ls))
    assert sr is sr_c, "step() must return the same StepResult"
    _assert_state(_rl_state(course, agents_c, done_c, sr_c), before,
                  f"{mode}/{variant}: the recording executed work")
    conditional = mode in ("warp", "external-conditional") and \
        wp.is_conditional_graph_supported()
    assert n_cond == (1 if conditional else 0), \
        f"{mode}: {n_cond} conditional node(s) on the course's regeneration flag"
    if mode == "warp":
        assert any(o is course for o in _owners(graph)), \
            "the course must be anchored on the caller's wp.Graph"
    elif mode.startswith("external"):
        regs = [o for o in _owners(graph) if isinstance(o, wp.Graph)]
        assert regs, "registered(graph=g) must retain its Warp graph on g"
        assert any(any(o is course for o in _owners(r)) for r in regs), \
            "the course must be anchored on the registration's Warp graph"

    lap_done = budget_done = idle_steps = reset_steps = finite = 0
    oob_seen, hits = set(), 0
    for k in range(_RL_STEPS):
        _poison(_step_outputs(sr_c).values())
        replay()

        agents_t.move()
        assert twin.step() is sr_t
        agents_t.decide(sr_t["progress"].laps, sr_t.valid, done_t)
        wp.synchronize()
        d = _host(done_t)
        laps = _host(sr_t["progress"].laps)
        vt = _host(sr_t.valid)
        if d.any():
            twin.enqueue_generate(mask=done_t, advance=True)
            wp.synchronize()
            reset_steps += 1
            lap_done += int(((laps >= 1) & (d != 0)).sum())
            budget_done += int(((laps < 1) & (d != 0) & (vt != 0)).sum())
        else:
            idle_steps += 1
        label = f"{mode}/{variant}: step {k} (done {d.tolist()})"
        _assert_state(_rl_state(course, agents_c, done_c, sr_c),
                      _rl_state(twin, agents_t, done_t, sr_t), label)
        if variant == "track":
            finite += int(np.isfinite(_host(sr_c[_TOY].kappa)).sum())
            oob_seen |= set(_host(sr_c["collision"].oob)[vt != 0].tolist())
        else:
            hits += int(_host(sr_c["collision"].hit).sum())
    assert twin._runner.graph_keys == (), "the eager twin must never self-graph"
    epoch = _host(course.seeds.epoch)
    _assert_bytes(course.result.generation, epoch, f"{variant}: generation == epoch")
    # Non-vacuity.
    assert lap_done > 0, "no env ever completed a lap (laps >= 1 never exercised)"
    assert budget_done > 0, "the per-env step budget never fired"
    assert reset_steps > 0 and idle_steps > 0, \
        f"regenerating steps {reset_steps}, zero-mask steps {idle_steps}: both must occur"
    assert epoch.max() >= 2, f"no env was regenerated twice: {epoch.tolist()}"
    if variant == "track":
        assert finite > 0, "the lookahead was never finite: the toy's step is vacuous"
        assert oob_seen == {0, 1}, f"valid envs' oob never varied: {oob_seen}"
    else:
        assert hits > 0, "no box ever hit a gate post"


# ---------------------------------------------------------------------------------------
# 2. The keyed self-graph: ONE graph per key, a third-party component in its body
# ---------------------------------------------------------------------------------------

def _mask_schedule(k: int) -> np.ndarray:
    """Host-written masks of the self-graph test: a zero mask every third call."""
    if k % 3 == 2:
        return _ZERO.copy()
    return np.array([1 if (k + 3 * e) % 5 == 0 else 0 for e in range(E)], np.int32)


def test_keyed_self_graph_holds_a_third_party_component_in_one_graph():
    """Outside any capture, ``regenerate(mask=course.inputs["regen_mask"],
    advance=True)`` on the composed track course (the ``CurvatureLookahead`` toy's
    ``on_generate`` in the masked body) captures ONE self-graph on the first call (one
    ``wp.capture_if`` on the course's flag where conditional nodes are supported: the
    conditional lowering) and replays it on the 39 later calls; with host-written masks
    (zero masks included) and eager steps between the calls, every call equals an eager
    twin (``enqueue_generate`` only when a mask is non-zero), bitwise.
    ``_runner.graph_keys`` is exactly ``((True, regen_mask.ptr),)``; a plain
    ``regenerate()`` then adds the default-key graph (``course.graph``), equal to the
    twin's eager ``enqueue_generate()``."""
    _load_test_kernels()
    course, twin = _course("track"), _course("track")
    agents_c, agents_t = _Agents(course, "track"), _Agents(twin, "track")
    agents_c.bind(course)
    agents_t.bind(twin)
    m = course.inputs["regen_mask"]
    m_t = _zeros_i32()
    assert course._runner.graph_keys == (), "construction prepares eagerly: no graph yet"
    zero_calls = masked_calls = progressed = 0
    with _regen_conditionals(course) as calls:
        for k in range(40):
            mk = _mask_schedule(k)
            agents_c.move()
            agents_t.move()
            course.step()
            twin.step()
            wp.synchronize()
            progressed += int((_host(course.progress._progress) * mk).sum())
            _write(m, mk)
            _write(m_t, mk)
            assert course.regenerate(mask=m, advance=True) is course.result
            if mk.any():
                twin.enqueue_generate(mask=m_t, advance=True)
                masked_calls += 1
            else:
                zero_calls += 1
            wp.synchronize()
            _assert_state(_course_state(course), _course_state(twin),
                          f"call {k} (mask {mk.tolist()})")
    supported = wp.is_conditional_graph_supported()
    assert len(calls) == (1 if supported else 0), \
        f"{len(calls)} capture_if on the regeneration flag: the keyed graph must be " \
        f"captured once{' with one conditional node' if supported else ''}"
    assert course._runner.graph_keys == ((True, int(m.ptr)),), course._runner.graph_keys
    assert twin._runner.graph_keys == ()
    assert course.graph is None
    assert course.regenerate() is course.result
    twin.enqueue_generate()
    wp.synchronize()
    assert course.graph is not None, "a plain regenerate() captures the default graph"
    assert set(course._runner.graph_keys) == {None, (True, int(m.ptr))}
    _assert_state(_course_state(course), _course_state(twin), "plain regenerate()")
    # Non-vacuity.
    assert zero_calls > 0 and masked_calls > 0, (zero_calls, masked_calls)
    assert progressed > 0, "no regenerated env had made progress: the resets are vacuous"
    assert _host(twin.seeds.epoch).max() >= 2


# ---------------------------------------------------------------------------------------
# 3. Both lowerings of the zero-mask body
# ---------------------------------------------------------------------------------------

def _probe(course: Course, variant: str) -> "wp.array":
    """A derived buffer the masked body rewrites for every env from the unchanged
    result: the toy's curvature (track), the course line's samples (gates)."""
    if variant == "track":
        return course.components[_TOY]._kappa
    return course.components["centerline"].result.center


@pytest.mark.parametrize("lowering", ["conditional", "unconditional"])
@pytest.mark.parametrize("variant", _VARIANTS)
def test_zero_mask_body_is_an_identity_under_both_lowerings(variant, lowering):
    """``[course.regenerate(mask=zeros, advance=True)]`` recorded into a Warp
    ``ScopedCapture`` with (``unconditional``: inside ``track_gen.exec.unconditional()``)
    and without (``conditional``) the opt-out, on a course stepped to non-zero progress
    and warm localizer state: one conditional node on the course's flag, or none;
    replayed 5 times, every buffer (result, epochs, every component's result and state,
    the step outputs, the agents) is unchanged, bitwise. Then a poisoned derived buffer
    (:func:`_probe`) tells the lowerings apart on one more replay: skipped on device, the
    poison stays; recorded inline, the body recomputes it to its exact bytes (the
    all-zero-mask identity of design section 2.7, which the unconditional recording and
    the lazy ``prepare()`` rely on)."""
    if lowering == "conditional" and not wp.is_conditional_graph_supported():
        pytest.skip(_NO_CONDITIONAL)
    _load_test_kernels()
    course = _course(variant)
    agents = _Agents(course, variant)
    agents.bind(course)
    zeros = course.inputs["regen_mask"]
    s = _side_stream()
    wp.synchronize()
    sr = None
    for _ in range(7):
        agents.move(stream=s)
        sr = course.step(stream=s)
    # Warm the masked path outside any capture (an exact no-op on the zero mask).
    assert course.regenerate(mask=zeros, advance=True, stream=s) is course.result
    wp.synchronize()
    assert _host(course.progress._progress).any(), "the course made no progress"
    state = {**_course_state(course, sr), **agents.arrays()}
    before = _snapshot(state)
    scope = unconditional() if lowering == "unconditional" else contextlib.nullcontext()
    with _regen_conditionals(course) as calls:
        with scope, tg_capture.recording(DEV, s) as cap:
            assert course.regenerate(mask=zeros, advance=True, stream=s) is course.result
    assert len(calls) == (1 if lowering == "conditional" else 0), (lowering, len(calls))
    for r in range(5):
        _replay(cap.graph, s)
        _assert_state(state, before, f"{variant}/{lowering}: zero-mask replay {r}")
    probe = _probe(course, variant)
    _poison([probe])
    poisoned = _host(probe)
    _replay(cap.graph, s)
    label = f"{variant}/{lowering}: probe"
    if lowering == "conditional":
        _assert_bytes(probe, poisoned, f"{label}: the skipped body must not run (poison "
                                       f"stays)")
    else:
        key = next(k for k, a in state.items() if a.ptr == probe.ptr)
        _assert_bytes(probe, before[key], f"{label}: the inline body must recompute it "
                                          f"exactly")
    rest = {k: a for k, a in state.items() if a.ptr != probe.ptr}
    _assert_state(rest, {k: before[k] for k in rest}, f"{label}: the rest unchanged")


# ---------------------------------------------------------------------------------------
# 4. Lifetime (G6): the caller's graph keeps the course alive, and only it
# ---------------------------------------------------------------------------------------

def test_a_recorded_course_lives_exactly_as_long_as_the_caller_graph():
    """``[move -> course.step() -> course.regenerate(mask=inputs["regen_mask"],
    advance=True)]`` recorded into a Warp ``ScopedCapture``; every reference to the
    course dropped (``del`` + ``gc.collect()``): the course is still alive (anchored on
    the graph by ``step()`` and ``regenerate``), and 6 replays with host-written masks
    (zero masks included) write the kept result and step-output arrays, equal to an
    eager twin. Dropping the graph then releases the course."""
    _load_test_kernels()
    course, twin = _course("track"), _course("track")
    agents_c, agents_t = _Agents(course, "track"), _Agents(twin, "track")
    agents_c.bind(course)
    agents_t.bind(twin)
    m = course.inputs["regen_mask"]
    m_t = _zeros_i32()
    s = _side_stream()
    wp.synchronize()
    sr = course.step(stream=s)
    assert course.regenerate(mask=m, advance=True, stream=s) is course.result
    sr_t = twin.step()
    wp.synchronize()
    state = _course_state(course, sr)          # arrays only: no reference to the course
    _assert_state(state, _course_state(twin, sr_t), "after the warm-up")
    with tg_capture.recording(DEV, s) as cap:
        agents_c.move(stream=s)
        assert course.step(stream=s) is sr
        course.regenerate(mask=m, advance=True, stream=s)
    alive = weakref.ref(course)
    del course
    gc.collect()
    assert alive() is not None, "the caller's wp.Graph must keep the recorded course alive"
    for r, mk in enumerate((_M1, _ZERO, _M2, _ZERO, np.ones(E, np.int32), _M1)):
        _write(m, mk)
        _write(m_t, mk)
        _poison(_step_outputs(sr).values())
        _replay(cap.graph, s)
        agents_t.move()
        twin.step()
        if mk.any():
            twin.enqueue_generate(mask=m_t, advance=True)
        wp.synchronize()
        _assert_state(state, _course_state(twin, sr_t),
                      f"replay {r} after the course was dropped (mask {mk.tolist()})")
    assert _host(twin.seeds.epoch).max() >= 2
    del cap
    gc.collect()
    assert alive() is None, "dropping the caller's graph must release the course"


def test_a_recorded_call_anchors_the_binding_and_the_masks_it_reads():
    """``[course.step() -> course.reset(m_reset) -> course.regenerate(mask=m_regen,
    advance=True)]`` recorded into a Warp ``ScopedCapture`` with caller-owned masks: the
    graph anchors the course, the binding ``step()`` read (``course._bind_args``) and
    both masks. After a rebind to new arrays (the course drops the old ones) and every
    caller reference dropped (``del`` + ``gc.collect()``), the bound arrays and the masks
    are alive and the replays run; dropping the graph frees them."""
    _load_test_kernels()
    course = _course("track")
    agents = _Agents(course, "track")
    agents.bind(course)
    m_regen, m_reset = _dev(_M1, wp.int32), _dev(_M2, wp.int32)
    s = _side_stream()
    course.step(stream=s)                       # warm-up of the recorded paths
    course.reset(m_reset, stream=s)
    course.enqueue_generate(mask=m_regen, advance=True, stream=s)
    wp.synchronize()
    with tg_capture.recording(DEV, s) as cap:
        course.step(stream=s)
        course.reset(m_reset, stream=s)
        course.regenerate(mask=m_regen, advance=True, stream=s)
    owners = _owners(cap.graph)
    for what, obj in (("the course", course), ("the binding", course._bind_args),
                      ("the regeneration mask", m_regen), ("the reset mask", m_reset)):
        assert any(o is obj for o in owners), \
            f"{what} must be anchored on the caller's wp.Graph"
    refs = {k: weakref.ref(a) for k, a in (("position", agents.pos),
                                           ("orientation", agents.orient),
                                           ("half_extents", agents.he),
                                           ("regen mask", m_regen),
                                           ("reset mask", m_reset))}
    other = _Agents(course, "track")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RebindAfterCaptureWarning)
        other.bind(course)                      # the course drops the recorded arrays
    del agents, m_regen, m_reset, owners
    gc.collect()
    dead = [k for k, r in refs.items() if r() is None]
    assert not dead, f"freed while the graph that reads them lives: {dead}"
    for _ in range(3):
        _replay(cap.graph, s)
    del cap
    gc.collect()
    alive = [k for k, r in refs.items() if r() is not None]
    assert not alive, f"still alive after the graph was dropped: {alive}"


# ---------------------------------------------------------------------------------------
# 5. The conditional-body contract (C.2): a refused component
# ---------------------------------------------------------------------------------------

class _CopiesFromPinnedHost(_toys._MeanX):
    """A correct per-step component (the kit's ``_MeanX`` baseline: the mean x of the
    centerline, NaN for an invalid env) whose ``on_generate`` also copies a constant
    table from a PINNED host array into its own device buffer."""

    name = "acme.pinned_copy"

    def attach(self, source, cc) -> None:
        E_, dev = int(self.layout.num_envs), self.layout.device
        self._table_host = wp.array(np.arange(E_, dtype=np.float32), dtype=wp.float32,
                                    device="cpu", pinned=True)
        self._table = wp.zeros(E_, dtype=wp.float32, device=dev)
        super().attach(source, cc)           # runs the first on_generate

    def on_generate(self, ctx, mask) -> None:
        ctx.copy(self._table, self._table_host)


class _RecordsAnEvent(_toys._MeanX):
    """The same baseline whose ``on_generate`` records an (external) event on ``ctx``'s
    stream: an event-record node inside a capture."""

    name = "acme.event_record"

    def attach(self, source, cc) -> None:
        self._event = wp.Event(self.layout.device)
        super().attach(source, cc)

    def on_generate(self, ctx, mask) -> None:
        ctx.stream.record_event(self._event, external=True)


_OFFENCES = {"pinned-copy": _CopiesFromPinnedHost, "event-record": _RecordsAnEvent}


@pytest.mark.parametrize("offence", sorted(_OFFENCES))
def test_a_component_a_conditional_body_cannot_hold_is_refused(offence):
    """A component whose ``on_generate`` emits an operation a CUDA conditional-node body
    cannot hold (C.2: a copy from pinned host memory; an external event record), correct
    otherwise:

    - ``check_component(..., devices=["cuda:0"], modes=("warp",))`` fails exactly
      ``capture.caller`` (its ``warp`` mode records the course's ``any_set`` +
      ``skip_unless`` chain), with a message naming the conditional body;
    - in ``Course.track(..., extra=[toy])``, ``regenerate(mask=inputs["regen_mask"],
      advance=True)`` outside any capture (the keyed self-graph, conditional) and inside
      a caller's ``ScopedCapture`` raise ``CaptureError`` naming the component, change
      nothing and cache no graph;
    - the same call inside ``track_gen.exec.unconditional()`` runs (captured inline),
      and its graph then replays outside the block: equal to an eager twin, masked and
      zero-mask calls.

    If the ``pinned-copy`` variant is accepted, CUDA allowed the copy inside the body
    (its programming guide permits pinned device-mapped host memory in conditional
    bodies, and Warp's body check accepts memcpy nodes): then the C.2 row "no copy to or
    from host or pinned memory" is stricter than the hardware, and the event-record
    variant is the negative toy."""
    if not wp.is_conditional_graph_supported():
        pytest.skip(_NO_CONDITIONAL)
    toy_cls = _OFFENCES[offence]
    hint = (" (CUDA accepted a pinned-host copy inside a conditional body: relax C.2's "
            "'no copy to or from host or pinned memory' and keep the event-record toy)"
            if offence == "pinned-copy" else "")
    report = check_component(lambda cc: toy_cls(), source="track", devices=[DEV],
                             modes=("warp",))
    cap = report.result("capture.caller", DEV)
    assert not cap.passed and not cap.skipped, f"capture.caller passed{hint}\n{report}"
    assert "conditional" in cap.message.lower(), \
        f"capture.caller must name the conditional body: {cap.message!r}"
    assert report.failed_names == ["capture.caller"], str(report)

    course = Course.track(TrackGenConfig(num_envs=E, device=DEV),
                          checkpoint_spacing=_SPACING, extra=[toy_cls()], seeds=5)
    twin = Course.track(TrackGenConfig(num_envs=E, device=DEV),
                        checkpoint_spacing=_SPACING, extra=[toy_cls()], seeds=5)
    m = course.inputs["regen_mask"]
    m_t = _zeros_i32()
    before = _snapshot(_course_state(course))
    name = re.escape(toy_cls.name)
    _write(m, _M1)
    with pytest.raises(CaptureError, match=name) as exc:
        course.regenerate(mask=m, advance=True)
    assert "conditional" in str(exc.value).lower(), str(exc.value)
    assert course._runner.graph_keys == (), "a refused capture must not be cached"
    s = _side_stream()
    wp.synchronize()
    with pytest.raises(CaptureError, match=name):
        with tg_capture.recording(DEV, s):
            course.regenerate(mask=m, advance=True, stream=s)
    wp.synchronize()
    _assert_state(_course_state(course), before, f"{offence}: a refused call changed state")

    with unconditional():
        assert course.regenerate(mask=m, advance=True) is course.result
    _write(m_t, _M1)
    twin.enqueue_generate(mask=m_t, advance=True)
    wp.synchronize()
    _assert_state(_course_state(course), _course_state(twin), f"{offence}: unconditional")
    for label, mk in (("masked replay", _M2), ("zero-mask replay", _ZERO)):
        _write(m, mk)
        _write(m_t, mk)
        assert course.regenerate(mask=m, advance=True) is course.result
        if mk.any():
            twin.enqueue_generate(mask=m_t, advance=True)
        wp.synchronize()
        _assert_state(_course_state(course), _course_state(twin), f"{offence}: {label}")
    assert course._runner.graph_keys == ((True, int(m.ptr)),), course._runner.graph_keys
    np.testing.assert_array_equal(_host(course.seeds.epoch), _M1 + _M2)


# ---------------------------------------------------------------------------------------
# 6. A course with a default stream: its stream, and the guard under another capture
# ---------------------------------------------------------------------------------------

def _stream_course(front: str, stream) -> Course:
    """A track course whose default stream is ``stream`` (``None``: no default stream),
    through the preset or the generic constructor over a generator built with it."""
    cfg = TrackGenConfig(num_envs=E, device=DEV)
    if front == "preset":
        return Course.track(cfg, checkpoint_spacing=_SPACING, collision="segments",
                            extra=[tools.Localizer(warm_window=4)], seeds=5, stream=stream)
    assert front == "generic", front
    gen = TrackGenerator(cfg, seeds=5, stream=stream)
    return Course(gen, [tools.Checkpoints(_SPACING), tools.Progress(),
                        tools.BandCollision("segments"), tools.Localizer(warm_window=4)])


@pytest.mark.parametrize("front", ["preset", "generic"])
def test_a_default_stream_course_runs_every_call_on_its_stream(front):
    """``Course.track(cfg, stream=s)`` and ``Course(TrackGenerator(cfg, stream=s),
    comps)``: ``step()``, ``reset()``, ``reset(inputs["reset_mask"])`` and
    ``enqueue_generate(mask=..., advance=True)`` without ``stream=`` launch every kernel
    and copy on ``s`` (``trace_launches``), and the self-graph replays of
    ``regenerate(mask=..., advance=True)`` and ``regenerate()`` are launched on ``s``."""
    _load_test_kernels()
    s = _side_stream()
    course = _stream_course(front, s)
    agents = _Agents(course, "track")
    agents.bind(course)
    m = course.inputs["regen_mask"]
    wp.synchronize()
    S = int(s.cuda_stream)
    with trace_launches() as tr:
        course.step()
        course.reset()
        course.reset(course.inputs["reset_mask"])
        course.enqueue_generate(mask=m, advance=True)
    wp.synchronize()
    assert len(tr) > 0
    off = sorted({str(r) for r in tr if r.stream != S})
    assert not off, f"{front}: launches off the default stream {hex(S)}: {off[:6]}"
    replays: list = []
    real = wp.capture_launch

    def spy(graph, *args, **kwargs):
        stream = kwargs.get("stream", args[0] if args else None)
        replays.append(None if stream is None else int(stream.cuda_stream))
        return real(graph, *args, **kwargs)

    wp.capture_launch = spy
    try:
        assert course.regenerate(mask=m, advance=True) is course.result
        assert course.regenerate() is course.result
    finally:
        wp.capture_launch = real
    wp.synchronize()
    assert replays == [S, S], f"{front}: self-graph replays on {replays}, expected {hex(S)}"


@pytest.mark.parametrize("capture", ["warp", "torch"])
@pytest.mark.parametrize("front", ["preset", "generic"])
def test_a_default_stream_course_refuses_another_streams_capture(front, capture):
    """A course with a default stream ``s``, called without ``stream=`` while another
    stream records (``warp``: a ``wp.ScopedCapture`` of Warp's CURRENT stream; ``torch``:
    a plain ``torch.cuda.graph``): ``step``, ``reset``, ``regenerate(mask=...)``,
    ``enqueue_generate`` and ``generate`` raise ``CaptureError`` (naming ``stream=``)
    before any launch, instead of running eagerly on ``s`` outside the graph. In the
    same capture the RL step recorded with ``stream=`` (the capture stream, or
    ``"torch"``) then replays 40 times equal to an eager twin without a default stream:
    the refusals left the capture usable."""
    _load_test_kernels()
    s = _side_stream()
    course, twin = _stream_course(front, s), _stream_course(front, None)
    agents_c, agents_t = _Agents(course, "track"), _Agents(twin, "track")
    agents_c.bind(course)
    agents_t.bind(twin)
    done_c = course.inputs["regen_mask"]
    done_t = _zeros_i32()
    wp.synchronize()
    sr_c = course.step()
    assert course.regenerate(mask=done_c, advance=True) is course.result   # zero mask
    sr_t = twin.step()
    wp.synchronize()
    course.reset()
    twin.reset()
    wp.synchronize()
    _assert_state(_rl_state(course, agents_c, done_c, sr_c),
                  _rl_state(twin, agents_t, done_t, sr_t), f"{front}: after warm-up")
    before = _snapshot(_rl_state(course, agents_c, done_c, sr_c))

    def refused() -> None:
        calls = {"step": lambda: course.step(),
                 "reset": lambda: course.reset(),
                 "regenerate": lambda: course.regenerate(mask=done_c, advance=True),
                 "enqueue_generate": lambda: course.enqueue_generate(mask=done_c,
                                                                     advance=True),
                 "generate": lambda: course.generate(),
                 "generate(seeds=)": lambda: course.generate(seeds=7)}
        with trace_launches() as tr:
            for what, call in calls.items():
                with pytest.raises(CaptureError, match="stream="):
                    call()
        assert len(tr) == 0, f"a refused call launched: {[str(r) for r in tr][:4]}"

    def record(ks, lib):
        refused()
        agents_c.move(stream=ks)
        sr = course.step(stream=lib)
        agents_c.decide(sr["progress"].laps, sr.valid, done_c, stream=ks)
        assert course.regenerate(mask=done_c, advance=True, stream=lib) is course.result
        return sr

    if capture == "warp":
        cur = wp.get_stream(DEV)
        assert int(cur.cuda_stream) != int(s.cuda_stream)
        with tg_capture.recording(DEV, cur) as cap:
            sr = record(cur, cur)

        def replay():
            _replay(cap.graph, cur)
    else:
        torch = _torch_cuda()
        from track_gen.interop import torch as tg_torch

        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with tg_capture.mempool_disabled(DEV):
            with torch.cuda.graph(g):
                with tg_capture.capture_launch_forbidden():
                    sr = record(resolve_stream(DEV), "torch")
        torch.cuda.synchronize()
        tg_torch.retain(g, course)

        def replay():
            g.replay()
            torch.cuda.synchronize()
            wp.synchronize()

    assert sr is sr_c
    _assert_state(_rl_state(course, agents_c, done_c, sr_c), before,
                  f"{front}/{capture}: the recording executed work")
    reset_steps = idle_steps = 0
    for k in range(40):
        _poison(_step_outputs(sr_c).values())
        replay()
        agents_t.move()
        twin.step()
        agents_t.decide(sr_t["progress"].laps, sr_t.valid, done_t)
        wp.synchronize()
        d = _host(done_t)
        if d.any():
            twin.enqueue_generate(mask=done_t, advance=True)
            wp.synchronize()
            reset_steps += 1
        else:
            idle_steps += 1
        _assert_state(_rl_state(course, agents_c, done_c, sr_c),
                      _rl_state(twin, agents_t, done_t, sr_t),
                      f"{front}/{capture}: replay {k} (done {d.tolist()})")
    assert reset_steps > 0 and idle_steps > 0, (reset_steps, idle_steps)


# ---------------------------------------------------------------------------------------
# 7. The cold attach phase runs on Warp's current stream
# ---------------------------------------------------------------------------------------

class _TableAtAttach(tools.Component):
    """Writes a per-env table ONCE, at ``attach``, with a kernel on ``cc.ctx`` (nothing
    rewrites it later): its buffer is allocated and zero-filled on Warp's current
    stream, so a fill on another, unordered stream can be zeroed by the late memset."""

    stateful = False
    name = "acme.table"
    K = 64

    def attach(self, source, cc) -> None:
        lay = self.layout
        n = int(lay.num_envs) * self.K
        self.result = _toys.Value(wp.zeros(n, dtype=wp.float32, device=lay.device))
        cc.ctx.launch(_fill_table_k, n, [self.result.value])
        self.attached_at = time.perf_counter()


def _expected_table() -> np.ndarray:
    return (1.0 + 0.5 * np.arange(E * _TableAtAttach.K)).astype(np.float32)


def _spin_iterations(target_s: float) -> int:
    """The ``_spin_k`` iteration count that keeps one stream busy ~``target_s``."""
    out = wp.zeros(1, dtype=wp.float32, device=DEV)
    n, dt = 1 << 14, 0.0
    while n < (1 << 30):
        wp.synchronize()
        t0 = time.perf_counter()
        wp.launch(_spin_k, 1, inputs=[n, out], device=DEV)
        wp.synchronize()
        dt = time.perf_counter() - t0
        if dt >= target_s / 8:
            break
        n *= 4
    return int(min(2**31 - 1, max(n, n * target_s / max(dt, 1e-6))))


def _busy_current_stream(n: int) -> float:
    """Enqueue a long kernel on Warp's CURRENT stream; return the host time."""
    out = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.synchronize()
    t0 = time.perf_counter()
    wp.launch(_spin_k, 1, inputs=[n, out], device=DEV)
    return t0


def _attach_state(course: Course) -> dict:
    """Everything ``attach`` wrote: every per-generation component's result (the base
    ``step``), every wrapped utility's runtime state and result."""
    out = {}
    for name, comp in course.components.items():
        if type(comp).step is tools.Component.step:
            out.update(_arrays(comp.result, f"{name}.result"))
        u = getattr(comp, "utility", None)
        if u is None:
            continue
        for attr in _UTILITY_STATE:
            a = getattr(u, attr, None)
            if isinstance(a, wp.array):
                out[f"{name}.utility.{attr}"] = a
    return out


def test_the_cold_attach_phase_runs_on_warps_current_stream():
    """``Course(gen, [table, Checkpoints, Progress, BandCollision, Localizer],
    prepare=False)`` over a generator built with ``stream=s`` (the course's default
    stream), with ``_TableAtAttach`` (a table written only at ``attach``, by a kernel;
    listed first, so it attaches first), built while Warp's current stream is busy with
    a long kernel (~0.3 s; enqueued AFTER the generator, whose construction
    synchronizes; a preset with a config would build the generator, and so synchronize,
    inside the call): every component result and runtime state written at attach, the
    table, and the step outputs of one step at the same agent positions equal the same
    course built on the current stream, bitwise (the attach phase ran on the stream where
    the allocations happen, after ``s``'s work, and ``s`` waited for it). Non-vacuity:
    the table's attach ran while the long kernel was still running, and a control
    attached through a context on ``s`` races (its table is zeroed by the late memset)
    where the mempool is enabled."""
    _load_test_kernels()
    target = 0.3
    n = _spin_iterations(target)
    cfg = TrackGenConfig(num_envs=E, device=DEV)
    s = _side_stream()

    def build(stream):
        gen = TrackGenerator(cfg, seeds=5, stream=stream)
        wp.synchronize()
        toy = _TableAtAttach()
        t0 = _busy_current_stream(n)
        course = Course(gen, [toy, tools.Checkpoints(_SPACING), tools.Progress(),
                              tools.BandCollision("segments"),
                              tools.Localizer(warm_window=4)], prepare=False)
        window = toy.attached_at - t0
        wp.synchronize()
        return course, toy, window

    course_s, toy_s, window = build(s)
    course_c, toy_c, _ = build(None)
    assert window < 0.5 * target, \
        f"attach ran {window:.3f} s after the long kernel was enqueued: no race window"
    expected = _expected_table()
    _assert_bytes(toy_c.result.value, expected, "the table, built on the current stream")
    _assert_bytes(toy_s.result.value, expected, "the table, built with stream=s")
    _assert_state(_attach_state(course_s), _attach_state(course_c), "attach state")
    agents_s, agents_c = _Agents(course_s, "track"), _Agents(course_c, "track")
    agents_s.bind(course_s)
    agents_c.bind(course_c)
    agents_s.move(stream=s)
    agents_c.move()
    wp.synchronize()
    sr_s, sr_c = course_s.step(), course_c.step()
    wp.synchronize()
    _assert_state({**_attach_state(course_s), **_step_outputs(sr_s)},
                  {**_attach_state(course_c), **_step_outputs(sr_c)}, "after one step")

    # Control: the same component attached through a context on s while the current
    # stream is busy: the fill on s runs at once, the zero-fill after the long kernel.
    gen = TrackGenerator(cfg, seeds=5, stream=s)
    wp.synchronize()
    cc = tools.ComponentContext.for_producer(gen, ctx=LaunchContext.for_call(DEV, s))
    toy = _TableAtAttach()
    _busy_current_stream(n)
    cc.add(toy, gen.program.result)
    wp.synchronize()
    raced = _host(toy.result.value).tobytes() != expected.tobytes()
    print(f"\n[attach race control] mempool {wp.is_mempool_enabled(DEV)}: an attach on "
          f"another stream {'raced' if raced else 'did not race'}")
    if wp.is_mempool_enabled(DEV):
        assert raced, "the control did not race: the comparison above is vacuous"


def test_a_course_on_another_stream_waits_for_the_producers_stream(monkeypatch):
    """``Course(gen, comps, stream=s1)`` over ``TrackGenerator(cfg, stream=s2)`` while
    ``s2`` still runs a long kernel and then ``gen.generate(advance=True, sync=False)``:
    the course stream ``s1`` waits for ``s2`` before the attach phase and the warm-up
    (a ``wait_stream`` spy), and the course equals a twin built after an advancing
    generation that finished (every result and attach state, bitwise)."""
    _load_test_kernels()
    n = _spin_iterations(0.2)
    cfg = TrackGenConfig(num_envs=E, device=DEV)
    comps = lambda: [tools.Checkpoints(_SPACING), tools.Progress(),  # noqa: E731
                     tools.BandCollision("segments"), tools.Localizer(warm_window=4)]
    s1, s2 = _side_stream(), _side_stream()
    gen = TrackGenerator(cfg, seeds=5, stream=s2)
    twin_gen = TrackGenerator(cfg, seeds=5)
    twin_gen.generate(advance=True)
    twin = Course(twin_gen, comps())
    out = wp.zeros(1, dtype=wp.float32, device=DEV)
    wp.synchronize()
    waits = []
    real = wp.Stream.wait_stream

    def spy(self, other, *args, **kwargs):
        waits.append((int(self.cuda_stream or 0), int(other.cuda_stream or 0)))
        return real(self, other, *args, **kwargs)

    monkeypatch.setattr(wp.Stream, "wait_stream", spy)
    wp.launch(_spin_k, 1, inputs=[n, out], device=DEV, stream=s2)
    gen.generate(advance=True, sync=False)                 # pending on s2
    course = Course(gen, comps(), stream=s1)
    monkeypatch.setattr(wp.Stream, "wait_stream", real)
    wp.synchronize()
    assert (int(s1.cuda_stream), int(s2.cuda_stream)) in waits, \
        f"the course stream did not wait for the producer's stream: {waits}"
    _assert_state({**_arrays(course.result, "result"), **_attach_state(course)},
                  {**_arrays(twin.result, "result"), **_attach_state(twin)},
                  "a course built while its producer's stream was busy")


# ---------------------------------------------------------------------------------------
# 8. S3: host cost of the anchor probe and the default-stream guard in step()
# ---------------------------------------------------------------------------------------

def _time_steps(course: Course, n: int) -> float:
    """Host nanoseconds per ``course.step()`` over ``n`` eager calls (queued, not
    synchronized inside the timed loop)."""
    t0 = time.perf_counter_ns()
    for _ in range(n):
        course.step()
    dt = time.perf_counter_ns() - t0
    wp.synchronize()
    return dt / n


def _per_call_ns(fn, n: int = 20000) -> float:
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    return (time.perf_counter_ns() - t0) / n


def test_s3_host_cost_of_the_step_anchor_and_the_default_stream_guard():
    """S3: the host cost ``Course.step()`` gains in PR15: the anchor probe
    (``_compat.anchor(ctx.stream, course, course._bind_args)``: a dict truthiness check
    when no Warp capture is open) on every course, and the default-stream guard (``Course._ctx``) on a
    ``stream=`` course. A/B on the SAME course: real vs both replaced by no-ops, 15
    interleaved trials of 400 steps, the minimum per-call time of each side (the cost
    floor), the median of three ratios; plus isolated per-call costs. Printed (expected
    ~0); no assertion (design section 7.4 test 8)."""
    import track_gen._src.course as course_mod

    _load_test_kernels()
    s = _side_stream()
    courses = {"no default stream": _course("track"),
               "stream= course": _course("track", stream=s)}
    for c in courses.values():
        _Agents(c, "track").bind(c)
    assert "_ctx" in vars(Course), "the default-stream guard is expected as Course._ctx"
    real_anchor, real_ctx = tg_compat.anchor, vars(Course)["_ctx"]
    aliases = [k for k in ("anchor", "_anchor")
               if getattr(course_mod, k, None) is real_anchor]

    def noop_anchor(stream, *owners) -> None:
        return None

    def plain_ctx(self, stream=None):
        return self._runner.context(stream)

    def noops(on: bool) -> None:
        tg_compat.anchor = noop_anchor if on else real_anchor
        for k in aliases:
            setattr(course_mod, k, noop_anchor if on else real_anchor)
        Course._ctx = plain_ctx if on else real_ctx

    N, TRIALS = 400, 15
    lines = []
    gc_was = gc.isenabled()
    gc.disable()
    try:
        for label, course in courses.items():
            _time_steps(course, 100)                        # warm-up
            ratios, real_ns, noop_ns = [], math.inf, math.inf
            for _ in range(3):
                best = {False: math.inf, True: math.inf}
                for trial in range(TRIALS):
                    for on in ((False, True) if trial % 2 == 0 else (True, False)):
                        noops(on)
                        best[on] = min(best[on], _time_steps(course, N))
                noops(False)
                ratios.append((best[False] - best[True]) / best[True])
                real_ns, noop_ns = min(real_ns, best[False]), min(noop_ns, best[True])
            ctx = course._runner.context()
            anchor_ns = _per_call_ns(lambda: real_anchor(ctx.stream, course,
                                                         course._bind_args))
            guard_ns = _per_call_ns(lambda: real_ctx(course, None)) - \
                _per_call_ns(lambda: course._runner.context(None))
            ratio = float(np.median(ratios))
            assert math.isfinite(ratio)
            lines.append(
                f"[S3, {label}] Course.step host time: {noop_ns / 1e3:.2f} us with no-ops, "
                f"{real_ns / 1e3:.2f} us with the anchor + guard; cost ratio (median of 3) "
                f"{ratio * 100:.3f} % (attempts {[f'{r * 100:.3f} %' for r in ratios]}); "
                f"isolated: anchor {anchor_ns:.0f} ns, guard {guard_ns:.0f} ns "
                f"({guard_ns / noop_ns * 100:.3f} % of a step)")
    finally:
        noops(False)
        if gc_was:
            gc.enable()
    print("\n" + "\n".join(lines) + f"\n[S3] torch imported: {'torch' in sys.modules}")
