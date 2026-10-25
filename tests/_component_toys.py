"""Component toys of ``tests/test_components.py`` (roadmap PR14).

Each ``Broken*`` toy breaks exactly ONE clause of the component contract and must fail
exactly the matching check of ``track_gen.testing.check_component``
(:data:`EXPECTED`; roadmap PR15 adds the two that read a bound input in the regeneration
body). :data:`GENERATION_WRITTEN` (roadmap PR15) holds correct components whose result
has an array that only ``on_generate`` writes (a duck-typed per-generation one, a
per-step one exposing a per-generation table): they pass every check, ``capture.caller``
on CUDA included. :class:`CurvatureLookahead` is the writing-a-component toy: public
imports only (``track_gen.tools``, ``track_gen.results``, ``track_gen.localize``), it
overrides ``attach`` / ``on_bind`` / the hot methods and passes every check.
"""
from __future__ import annotations

import dataclasses
import math
import typing

import warp as wp

from track_gen import tools
from track_gen.localize import curvature
from track_gen.results import (HasCenterline, HasGeneration, HasValidity, check_source)


@dataclasses.dataclass
class Value:
    """A toy result: one ``[E]`` (or ``[E * M]``) float32 array."""

    value: wp.array


@dataclasses.dataclass
class Counters:
    steps: wp.array
    acc: wp.array


@wp.kernel
def _mean_x_k(center: wp.array(dtype=wp.vec3f), count: wp.array(dtype=wp.int32),
              valid: wp.array(dtype=wp.int32), has_valid: int, S: int,
              out: wp.array(dtype=wp.float32)):
    e = wp.tid()
    m = count[e]
    if m > S:
        m = S
    if has_valid != 0:
        if valid[e] == 0:
            m = 0
    if m < 1:
        out[e] = wp.nan
        return
    acc = float(0.0)
    for i in range(m):
        acc = acc + center[e * S + i][0]
    out[e] = acc / float(m)


@wp.kernel
def _fill_nan_k(out: wp.array(dtype=wp.float32)):
    out[wp.tid()] = wp.nan


@wp.kernel
def _reverse_samples_k(center: wp.array(dtype=wp.vec3f), count: wp.array(dtype=wp.int32),
                       valid: wp.array(dtype=wp.int32), S: int, M: int,
                       out: wp.array(dtype=wp.float32)):
    # Writes a quarter of the points BACKWARD from the end of the env's row, without
    # clamping to its capacity: past M points it writes into the previous env's rows.
    e = wp.tid()
    m = count[e]
    if m > S:
        m = S
    if valid[e] == 0:
        m = 0
    n = m // 4
    for k in range(n):
        idx = e * M + (M - 1 - k)
        if idx < 0:
            idx = 0
        out[idx] = center[e * S + k][0]


@wp.kernel
def _accumulate_k(position: wp.array(dtype=wp.vec3f), valid: wp.array(dtype=wp.int32),
                  has_valid: int, generation: wp.array(dtype=wp.int32), has_gen: int,
                  seen: wp.array(dtype=wp.int32), steps: wp.array(dtype=wp.int32),
                  acc: wp.array(dtype=wp.float32), out_steps: wp.array(dtype=wp.int32),
                  out_acc: wp.array(dtype=wp.float32)):
    e = wp.tid()
    if has_gen != 0:
        g = generation[e]
        if seen[e] != g:
            seen[e] = g
            steps[e] = 0
            acc[e] = 0.0
    frozen = int(0)
    if has_valid != 0:
        if valid[e] == 0:
            frozen = int(1)
    p = position[e]
    if frozen == 0 and p[0] == p[0]:
        steps[e] = steps[e] + 1
        acc[e] = acc[e] + p[0]
    out_steps[e] = steps[e]
    out_acc[e] = acc[e]


@wp.kernel
def _partial_reset_k(mask: wp.array(dtype=wp.int32), steps: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] != 0:
        steps[e] = 0          # forgets acc


@wp.kernel
def _count_k(counter: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)):
    e = wp.tid()
    counter[e] = counter[e] + 1
    out[e] = float(counter[e])


@wp.kernel
def _zero_masked_k(mask: wp.array(dtype=wp.int32), use_mask: int,
                   counter: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if use_mask == 0 or mask[e] != 0:
        counter[e] = 0


def _seed_seen(ctx, seen, generation) -> None:
    if generation is not None:
        ctx.copy(seen, generation)
    else:
        ctx.fill(seen, 0)


class _MeanX(tools.Component):
    """Correct baseline: per env, the mean x of the centerline's real points (NaN for an
    invalid env), recomputed every step."""

    name = "mean_x"
    requires = (HasCenterline, HasValidity)
    _has_valid = 1

    def attach(self, source, cc) -> None:
        lay = self.layout
        self._E, self._S = int(lay.num_envs), int(lay.stride)
        self._src = (source.center, source.count, source.valid)
        self.result = Value(wp.zeros(self._E, dtype=wp.float32, device=lay.device))
        self.on_generate(cc.ctx, None)
        wp.load_module(__name__, device=lay.device)

    def _launch(self, ctx) -> None:
        center, count, valid = self._src
        ctx.launch(_mean_x_k, self._E, [center, count, valid, self._has_valid, self._S,
                                        self.result.value])

    def step(self, ctx):
        self._launch(ctx)
        return self.result


class BrokenAllocates(_MeanX):
    """Allocates in ``step`` (``hot_path.allocation``)."""

    def step(self, ctx):
        wp.empty(1, dtype=wp.float32, device=self.layout.device)
        return super().step(ctx)


class BrokenReadsBack(_MeanX):
    """Reads its result back on the host in ``step`` (``hot_path.host_sync``)."""

    def step(self, ctx):
        out = super().step(ctx)
        out.value.numpy()
        return out


class BrokenLaunchesWithoutCtx(_MeanX):
    """Launches on Warp's current stream, not ``ctx``'s (``hot_path.stream``)."""

    def step(self, ctx):
        center, count, valid = self._src
        wp.launch(_mean_x_k, dim=self._E, inputs=[center, count, valid, 1, self._S,
                                                  self.result.value],
                  device=self.layout.device)
        return self.result


class BrokenReplacesResult(_MeanX):
    """Swaps ``self.result`` for a spare (written identically) on a masked
    ``on_generate`` (``result.stable``)."""

    def attach(self, source, cc) -> None:
        super().attach(source, cc)
        self._pair = (self.result, Value(wp.zeros(self._E, dtype=wp.float32,
                                                  device=self.layout.device)))

    def step(self, ctx):
        center, count, valid = self._src
        for r in self._pair:
            ctx.launch(_mean_x_k, self._E, [center, count, valid, 1, self._S, r.value])
        return self.result

    def on_generate(self, ctx, mask) -> None:
        if mask is not None:
            a, b = self._pair
            self.result = b if self.result is a else a


class BrokenWritesOutsideRows(tools.Component):
    """Sized from ``max_loop_length``; writes past its own rows when the capacity is too
    small (``capacity.clamp``)."""

    name = "reverse_samples"
    requires = (HasCenterline, HasValidity)

    def attach(self, source, cc) -> None:
        lay = self.layout
        self._E, self._S = int(lay.num_envs), int(lay.stride)
        self._M = max(3, math.ceil(cc.capacity["max_loop_length"] / 0.5))
        self._src = (source.center, source.count, source.valid)
        self.result = Value(wp.zeros(self._E * self._M, dtype=wp.float32,
                                     device=lay.device))
        self.on_generate(cc.ctx, None)

    def on_generate(self, ctx, mask) -> None:
        center, count, valid = self._src
        ctx.launch(_fill_nan_k, self._E * self._M, [self.result.value])
        ctx.launch(_reverse_samples_k, self._E, [center, count, valid, self._S, self._M,
                                                 self.result.value])


class BrokenIgnoresValid(_MeanX):
    """Ignores ``valid`` (``invalid.sentinel``): an invalid env keeps its geometric
    answer."""

    requires = (HasCenterline,)
    _has_valid = 0

    def attach(self, source, cc) -> None:
        lay = self.layout
        self._E, self._S = int(lay.num_envs), int(lay.stride)
        self._src = (source.center, source.count, source.count)
        self.result = Value(wp.zeros(self._E, dtype=wp.float32, device=lay.device))
        wp.load_module(__name__, device=lay.device)


class BrokenForgetsReset(tools.Component):
    """Stateful: ``reset`` clears one of its two state fields (``reset.exact``); its
    generation-aware self-reset clears both (so masked regenerations stay exact)."""

    name = "accumulator"
    stateful = True
    requires = (HasCenterline,)
    optional = (HasValidity, HasGeneration)
    inputs = {"position": tools.InputSpec(wp.vec3f)}

    def attach(self, source, cc) -> None:
        lay = self.layout
        E, dev = int(lay.num_envs), lay.device
        self._E = E
        self._valid = getattr(source, "valid", None)
        self._gen = getattr(source, "generation", None)
        self._count = source.count
        self._steps = wp.zeros(E, dtype=wp.int32, device=dev)
        self._acc = wp.zeros(E, dtype=wp.float32, device=dev)
        self._seen = wp.zeros(E, dtype=wp.int32, device=dev)
        self.result = Counters(wp.zeros(E, dtype=wp.int32, device=dev),
                               wp.zeros(E, dtype=wp.float32, device=dev))
        _seed_seen(cc.ctx, self._seen, self._gen)
        wp.load_module(__name__, device=dev)

    def on_bind(self, position) -> None:
        self._pos = position

    def step(self, ctx):
        v, g = self._valid, self._gen
        ctx.launch(_accumulate_k, self._E,
                   [self._pos, self._count if v is None else v, 0 if v is None else 1,
                    self._count if g is None else g, 0 if g is None else 1, self._seen,
                    self._steps, self._acc, self.result.steps, self.result.acc])
        return self.result

    def reset(self, ctx, mask) -> None:
        ctx.launch(_partial_reset_k, self._E, [mask, self._steps])


class BrokenStaleRecompute(_MeanX):
    """Recomputes from a copy of the geometry taken at attach (``regenerate.masked``:
    after a masked regeneration the masked envs keep the old answer)."""

    def attach(self, source, cc) -> None:
        lay = self.layout
        self._E, self._S = int(lay.num_envs), int(lay.stride)
        self._stale = wp.clone(source.center)
        self._src = (self._stale, source.count, source.valid)
        self.result = Value(wp.zeros(self._E, dtype=wp.float32, device=lay.device))
        self.on_generate(cc.ctx, None)
        wp.load_module(__name__, device=lay.device)

    def on_generate(self, ctx, mask) -> None:
        self._launch(ctx)

    def step(self, ctx):
        return self.result


class BrokenHiddenState(tools.Component):
    """Declares ``stateful=False`` but counts steps (``statelessness``)."""

    name = "hidden"
    requires = (HasCenterline,)

    def attach(self, source, cc) -> None:
        lay = self.layout
        E, dev = int(lay.num_envs), lay.device
        self._E = E
        self._counter = wp.zeros(E, dtype=wp.int32, device=dev)
        self._ones = wp.zeros(E, dtype=wp.int32, device=dev)
        self.result = Value(wp.zeros(E, dtype=wp.float32, device=dev))
        self.on_generate(cc.ctx, None)
        wp.load_module(__name__, device=dev)

    def on_generate(self, ctx, mask) -> None:
        ctx.launch(_zero_masked_k, self._E,
                   [self._ones if mask is None else mask, 0 if mask is None else 1,
                    self._counter])

    def step(self, ctx):
        ctx.launch(_count_k, self._E, [self._counter, self.result.value])
        return self.result


class BrokenReadsUndeclared(_MeanX):
    """Reads ``outer``, which ``requires`` / ``optional`` do not declare
    (``requires.complete``)."""

    def attach(self, source, cc) -> None:
        self._outer_len = int(source.outer.shape[0])
        super().attach(source, cc)


class BrokenClaimsAnyTopology(_MeanX):
    """Declares ``topology="any"`` but refuses an open loop itself (``topology``)."""

    topology = "any"

    def attach(self, source, cc) -> None:
        check_source(source, HasCenterline, consumer="BrokenClaimsAnyTopology")
        super().attach(source, cc)


class BrokenAllocatesOnce(_MeanX):
    """Allocates a scratch buffer on its FIRST ``step`` only (``hot_path.allocation``)."""

    _scratch = None

    def step(self, ctx):
        if self._scratch is None:
            self._scratch = wp.zeros(self._E, dtype=wp.float32, device=self.layout.device)
        return super().step(ctx)


class BrokenReadsBackOnce(_MeanX):
    """Reads the source's count back on its FIRST ``step`` only (``hot_path.host_sync``)."""

    _n = None

    def step(self, ctx):
        if self._n is None:
            self._n = int(self._src[1].numpy().max())
        return super().step(ctx)


class BrokenLaunchesWithoutCtxOnce(_MeanX):
    """Launches on Warp's current stream on its FIRST ``step`` only, through ``ctx``
    afterwards (``hot_path.stream``)."""

    _first = True

    def step(self, ctx):
        if self._first:
            self._first = False
            center, count, valid = self._src
            wp.launch(_mean_x_k, dim=self._E, inputs=[center, count, valid, 1, self._S,
                                                      self.result.value],
                      device=self.layout.device)
            return self.result
        return super().step(ctx)


class BrokenDeclaresNothing(_MeanX):
    """Reads ``center`` / ``count`` / ``valid`` of its read target but declares no
    ``requires`` (``requires.complete``: an empty read target makes attach fail)."""

    requires = ()


class BrokenNoSelfReset(tools.Component):
    """Stateful, reads only ``optional`` members (``valid``, ``generation``) and keeps
    ``generation`` unread: no self-reset after a regeneration (``regenerate.self_reset``);
    its ``reset`` is exact."""

    name = "steps"
    stateful = True
    optional = (HasValidity, HasGeneration)
    inputs = {"position": tools.InputSpec(wp.vec3f)}

    def attach(self, source, cc) -> None:
        lay = self.layout
        E, dev = int(lay.num_envs), lay.device
        self._E = E
        self._valid = getattr(source, "valid", None)
        self._steps = wp.zeros(E, dtype=wp.int32, device=dev)
        self._acc = wp.zeros(E, dtype=wp.float32, device=dev)
        self._seen = wp.zeros(E, dtype=wp.int32, device=dev)
        self.result = Counters(wp.zeros(E, dtype=wp.int32, device=dev),
                               wp.zeros(E, dtype=wp.float32, device=dev))
        wp.load_module(__name__, device=dev)

    def on_bind(self, position) -> None:
        self._pos = position

    def step(self, ctx):
        v = self._valid
        ctx.launch(_accumulate_k, self._E,
                   [self._pos, self._steps if v is None else v, 0 if v is None else 1,
                    self._steps, 0, self._seen, self._steps, self._acc,
                    self.result.steps, self.result.acc])
        return self.result

    def reset(self, ctx, mask) -> None:
        ctx.launch(_zero_masked_k, self._E, [mask, 1, self._steps])
        ctx.launch(_zero_masked_f_k, self._E, [mask, self._acc])


@wp.kernel
def _zero_masked_f_k(mask: wp.array(dtype=wp.int32), acc: wp.array(dtype=wp.float32)):
    e = wp.tid()
    if mask[e] != 0:
        acc[e] = 0.0


class _Pair(typing.NamedTuple):
    value: wp.array


class BrokenHiddenStateTuple(BrokenHiddenState):
    """``BrokenHiddenState`` with a ``NamedTuple`` result: its array is found all the
    same (``statelessness``)."""

    def attach(self, source, cc) -> None:
        super().attach(source, cc)
        self.result = _Pair(self.result.value)


class BrokenSkipsAttachGenerate(_MeanX):
    """Computes in ``on_generate`` only, and its ``attach`` does not end with
    ``on_generate(cc.ctx, None)`` (``construct``: the attach contract)."""

    def attach(self, source, cc) -> None:
        lay = self.layout
        self._E, self._S = int(lay.num_envs), int(lay.stride)
        self._src = (source.center, source.count, source.valid)
        self.result = Value(wp.zeros(self._E, dtype=wp.float32, device=lay.device))
        wp.load_module(__name__, device=lay.device)

    def on_generate(self, ctx, mask) -> None:
        self._launch(ctx)

    def step(self, ctx):
        return self.result


@wp.kernel
def _peek_k(position: wp.array(dtype=wp.vec3f), scratch: wp.array(dtype=wp.float32)):
    e = wp.tid()
    scratch[e] = position[e][0]


class BrokenReadsInputInGenerate(_MeanX):
    """Launches on its bound ``position`` in ``on_generate`` (into a private scratch, so
    its outputs are unaffected): a Course's regeneration self-graph would bake the input
    (``hot_path.buffers``, roadmap PR15)."""

    inputs = {"position": tools.InputSpec(wp.vec3f)}

    def attach(self, source, cc) -> None:
        E, dev = int(self.layout.num_envs), self.layout.device
        self._pos = wp.zeros(E, dtype=wp.vec3f, device=dev)     # until bound
        self._scratch = wp.zeros(E, dtype=wp.float32, device=dev)
        super().attach(source, cc)

    def on_bind(self, position) -> None:
        self._pos = position

    def on_generate(self, ctx, mask) -> None:
        ctx.launch(_peek_k, self._E, [self._pos, self._scratch])


class BrokenReadsInputInReset(BrokenReadsInputInGenerate):
    """The same launch in ``reset`` instead (``hot_path.buffers``)."""

    def on_generate(self, ctx, mask) -> None:
        pass

    def reset(self, ctx, mask) -> None:
        ctx.launch(_peek_k, self._E, [self._pos, self._scratch])


#: toy -> the one check it must fail.
EXPECTED = {
    BrokenAllocates: "hot_path.allocation",
    BrokenReadsBack: "hot_path.host_sync",
    BrokenLaunchesWithoutCtx: "hot_path.stream",
    BrokenReplacesResult: "result.stable",
    BrokenWritesOutsideRows: "capacity.clamp",
    BrokenIgnoresValid: "invalid.sentinel",
    BrokenForgetsReset: "reset.exact",
    BrokenStaleRecompute: "regenerate.masked",
    BrokenHiddenState: "statelessness",
    BrokenReadsUndeclared: "requires.complete",
    BrokenClaimsAnyTopology: "topology",
    BrokenAllocatesOnce: "hot_path.allocation",
    BrokenReadsBackOnce: "hot_path.host_sync",
    BrokenLaunchesWithoutCtxOnce: "hot_path.stream",
    BrokenDeclaresNothing: "requires.complete",
    BrokenNoSelfReset: "regenerate.self_reset",
    BrokenHiddenStateTuple: "statelessness",
    BrokenSkipsAttachGenerate: "construct",
    BrokenReadsInputInGenerate: "hot_path.buffers",
    BrokenReadsInputInReset: "hot_path.buffers",
}


# ---------------------------------------------------------------------------------------
# Correct components with an array that only on_generate writes (capture.caller on CUDA:
# a replay whose conditional node skips on_generate keeps its previous bytes there)
# ---------------------------------------------------------------------------------------

class DuckPerGeneration:
    """Duck-typed (no ``tools.Component`` base), per generation: per env, the mean x of
    the centerline's real points (NaN for an invalid env), recomputed for the whole batch
    in ``on_generate`` (unmasked envs get their own bytes again); ``step`` returns the
    result unchanged. A duck-typed component defines its own ``step``, so a course counts
    it as per-step (a ``StepResult`` entry) although ``step`` writes nothing."""

    name = "acme.duck_mean_x"
    source = None
    requires = (HasCenterline, HasValidity)
    optional = ()
    topology = "closed"
    stateful = False
    inputs: dict = {}
    after = ()
    layout = None
    result = None

    def attach(self, source, cc) -> None:
        lay = self.layout
        self._E, self._S = int(lay.num_envs), int(lay.stride)
        self._src = (source.center, source.count, source.valid)
        self.result = Value(wp.zeros(self._E, dtype=wp.float32, device=lay.device))
        self.on_generate(cc.ctx, None)
        wp.load_module(__name__, device=lay.device)

    def bind(self, **arrays) -> None:
        if arrays:
            raise TypeError(f"{self.name}: unknown inputs {sorted(arrays)} (declared: [])")

    def on_generate(self, ctx, mask) -> None:
        center, count, valid = self._src
        ctx.launch(_mean_x_k, self._E, [center, count, valid, 1, self._S,
                                        self.result.value])

    def step(self, ctx):
        return self.result

    def reset(self, ctx, mask) -> None:
        pass


@dataclasses.dataclass
class ValueAndTable:
    """``value``: per step; ``table``: per generation."""

    value: wp.array
    table: wp.array


class PerStepWithTable(_MeanX):
    """Per step (``_MeanX``'s ``step`` rewrites ``value`` for every env), and its result
    also exposes ``table``, the same means recomputed for the whole batch in
    ``on_generate`` only (as a lookahead exposing its curvature table would)."""

    name = "acme.mean_x_table"

    def attach(self, source, cc) -> None:
        lay = self.layout
        self._E, self._S = int(lay.num_envs), int(lay.stride)
        self._src = (source.center, source.count, source.valid)
        self.result = ValueAndTable(wp.zeros(self._E, dtype=wp.float32, device=lay.device),
                                    wp.zeros(self._E, dtype=wp.float32, device=lay.device))
        self.on_generate(cc.ctx, None)
        wp.load_module(__name__, device=lay.device)

    def on_generate(self, ctx, mask) -> None:
        center, count, valid = self._src
        ctx.launch(_mean_x_k, self._E, [center, count, valid, 1, self._S,
                                        self.result.table])


#: Correct components whose result holds an array that only ``on_generate`` writes.
GENERATION_WRITTEN = (DuckPerGeneration, PerStepWithTable)


# ---------------------------------------------------------------------------------------
# The writing-a-component toy (public imports only)
# ---------------------------------------------------------------------------------------

@dataclasses.dataclass
class Lookahead:
    """Per env: the centerline curvature ``ahead`` points past the agent's segment, and
    the distance from the agent to that point (NaN without a frame)."""

    kappa: wp.array
    distance: wp.array


@wp.kernel
def _lookahead_k(kappa: wp.array(dtype=wp.float32), center: wp.array(dtype=wp.vec3f),
                 count: wp.array(dtype=wp.int32), S: int, ahead: int,
                 segment: wp.array(dtype=wp.int32), position: wp.array(dtype=wp.vec3f),
                 out_kappa: wp.array(dtype=wp.float32),
                 out_distance: wp.array(dtype=wp.float32)):
    e = wp.tid()
    m = count[e]
    if m > S:
        m = S
    seg = segment[e]
    if seg < 0 or m < 3:
        out_kappa[e] = wp.nan
        out_distance[e] = wp.nan
        return
    i = (seg + ahead) % m
    out_kappa[e] = kappa[e * S + i]
    out_distance[e] = wp.length(center[e * S + i] - position[e])


class CurvatureLookahead(tools.Component):
    """The curvature ``ahead`` centerline points past the agent: reads the track
    (``HasCenterline``; ``valid`` when present, through ``curvature``) and the
    ``Localizer`` named ``localizer`` (``after``); per generation ``curvature(...,
    out=, scratch=)`` into its own arrays, per step one kernel. Named
    ``"acme.lookahead"`` (a third-party name: a package prefix), it is an entry of a
    course's ``StepResult`` (it overrides ``step``)."""

    stateful = False
    requires = (HasCenterline,)
    optional = (HasValidity,)
    inputs = {"position": tools.InputSpec(wp.vec3f, doc="the agent")}

    def __init__(self, ahead: int = 6, *, window: int = 2, localizer: str = "localizer",
                 name: str = "acme.lookahead") -> None:
        if int(ahead) < 0:
            raise ValueError(f"ahead must be >= 0, got {ahead!r}")
        self.ahead, self.window, self.name = int(ahead), int(window), name
        self.localizer = localizer
        self.after = (localizer,)

    def attach(self, source, cc) -> None:
        lay = self.layout
        E, S, dev = int(lay.num_envs), int(lay.stride), lay.device
        self._E, self._S = E, S
        self._source = source
        self._center, self._count = source.center, source.count
        self._frame = cc.component(self.localizer).result
        self._kappa = wp.zeros(E * S, dtype=wp.float32, device=dev)
        self._scratch = wp.zeros(E * S, dtype=wp.float32, device=dev)
        self.result = Lookahead(wp.zeros(E, dtype=wp.float32, device=dev),
                                wp.zeros(E, dtype=wp.float32, device=dev))
        self._position = None
        self.on_generate(cc.ctx, None)
        wp.load_module(__name__, device=dev)

    def on_bind(self, position) -> None:
        self._position = position

    def on_generate(self, ctx, mask) -> None:
        curvature(self._source, self.window, ctx, out=self._kappa, scratch=self._scratch)

    def step(self, ctx):
        r = self.result
        ctx.launch(_lookahead_k, self._E,
                   [self._kappa, self._center, self._count, self._S, self.ahead,
                    self._frame.segment, self._position, r.kappa, r.distance])
        return r
