"""Components: runtime consumers of a generator's result.

A :class:`Component` declares what it reads (``requires`` / ``optional`` capability
protocols of its read target, ``source``, ``topology``), what it is bound to (``inputs``:
:class:`InputSpec`), and whether it keeps per-env state (``stateful``); a
:class:`ComponentContext` checks the declaration and attaches it, with the producer's
static capacities (:meth:`ComponentContext.for_producer`). After ``attach`` its hot
methods (``on_generate``, ``step``, ``reset``) launch through the
:class:`~track_gen.exec.LaunchContext` they are given, allocate nothing and never
synchronize, so a ``Course`` (or your sim loop) can record them into a CUDA graph::

    import warp as wp

    from track_gen import TrackGenerator, tools
    from track_gen.exec import LaunchContext
    from track_gen.pipeline import presets

    E, device = 64, "cuda:0"
    gen = TrackGenerator(presets.track(num_envs=E, device=device))
    pos = wp.zeros(E, dtype=wp.vec3f, device=device)       # your agents' positions

    track = gen.generate()                                 # the producer's result
    cc = tools.ComponentContext.for_producer(gen)          # layout + plan capacities
    cps = cc.add(tools.Checkpoints(0.5), track)            # plan-bound capacity
    prog = cc.add(tools.Progress(), track)                 # reads cps.result
    prog.bind(position=pos)
    ctx = LaunchContext.for_call(track.layout.device)
    events = prog.step(ctx)

The built-in adapters each wrap one 0.x utility (``adapter.utility``; the legacy classes
keep every name): :class:`BandCollision` (``CollisionChecker``), :class:`DiscCollision`
(``DiscChecker``), :class:`FrameCollision` (``FrameChecker``), :class:`Props`
(``PropSampler``), :class:`Checkpoints` (``CheckpointSampler``),
:class:`GateCheckpoints` (``CheckpointSet.from_gates``), :class:`Progress`
(``ProgressTracker``), :class:`Localizer` (``TrackLocalizer``),
:class:`HeightFieldBake` (``HeightFieldBaker``) and :class:`CenterlineFromGates`
(``CourseLine``: its result is the ``Centerline``). ``track_gen.testing.check_component``
checks a component (yours or a built-in) against the contract.
"""
from ._adapters import (BandCollision, CenterlineFromGates, Checkpoints, DiscCollision,
                        FrameCollision, GateCheckpoints, HeightFieldBake, Localizer,
                        Progress, Props)
from ._component import Component, ComponentContext, InputSpec

__all__ = [
    "BandCollision",
    "CenterlineFromGates",
    "Checkpoints",
    "Component",
    "ComponentContext",
    "DiscCollision",
    "FrameCollision",
    "GateCheckpoints",
    "HeightFieldBake",
    "InputSpec",
    "Localizer",
    "Progress",
    "Props",
]
