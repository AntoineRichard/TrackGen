"""The track and gate pipelines as data (preview, roadmap PR10 / PR11).

A :class:`PipelineSpec` lists the stages of the track pipeline in order
(:class:`StageRef`: kind, name, params) over one :class:`BatchSpec`. :func:`build` plans
it (every channel a stage reads has an earlier writer, every channel is allocated once,
each stage's scratch is namespaced, validity stages get their fail bits), runs each
stage's ``build`` once and returns a :class:`Program`: a ``GraphRunner`` launchable
whose ``enqueue`` runs the stages in order, recordable into a caller's CUDA graph like
``TrackGenerator.generate``; a pipeline with HOST stages runs in :class:`Segment` s
(its HOST stages eagerly, the rest from graphs). :func:`presets.track <track_gen.pipeline.presets.track>`
builds the default track spec with one argument per replaceable stage, and
``TrackGenerator(spec)`` runs any spec.

The ``relax``, ``width``, ``elevation`` and ``validity`` stages come from
:data:`track_gen.plugin.registry` (TrackGen's built-ins and third-party plugins); the
others (``resample``, ``frames``, ``offset``, ``lift``) are TrackGen's built-ins.
:data:`BUILTIN_FAIL_BITS` names the ``valid.fail`` bits of the built-in checks.

A gate pipeline (``BatchSpec.max_gates`` set; :func:`presets.gates
<track_gen.pipeline.presets.gates>`, ``GateGenerator(spec)``) builds a Program whose
result is a ``GateSequence``: a gate generator's anchors (or a track generator's
stations), an ``order`` stage, the sphere solve, an ``elevation`` stage (the track ones,
rewired to the gates), the lift, tangents, frames and ``validity.gate_default``
(:data:`GATE_FAIL_BITS`).

Roadmap PR13b: ``build(spec, out=...)`` (and the facades' ``out=``) writes into a
caller-owned result, a :meth:`view <track_gen.Track.view>` of a larger one included
(mixed batches); ``randomize.uniform`` stages (``presets.track(randomize=...)``) draw
dynamic params per env in the graph; ``generator="external"`` ingests caller-provided
centerlines through ``Program.inputs`` (:class:`InputInfo`), loaded with
:func:`load_centerlines`.

Preview: names and signatures may change before 1.0. ``import track_gen`` does not
import this package.
"""
from __future__ import annotations

from ..stages.channels import BUILTIN_FAIL_BITS, GATE_FAIL_BITS
from . import legacy, presets
from .ingest import load_centerlines
from .planner import PlanError, build
from .program import AttributeInfo, InputInfo, Program, Segment
from .spec import BatchSpec, PipelineSpec, StageRef

__all__ = [
    "AttributeInfo",
    "BUILTIN_FAIL_BITS",
    "BatchSpec",
    "GATE_FAIL_BITS",
    "InputInfo",
    "PipelineSpec",
    "PlanError",
    "Program",
    "Segment",
    "StageRef",
    "build",
    "legacy",
    "load_centerlines",
    "presets",
]
