"""The track pipeline as data (preview, roadmap PR10a).

A :class:`PipelineSpec` lists the stages of the track pipeline in order
(:class:`StageRef`: kind, name, params) over one :class:`BatchSpec`. :func:`build` plans
it (every channel a stage reads has an earlier writer, every channel is allocated once,
each stage's scratch is namespaced), runs each stage's ``build`` once and returns a
:class:`Program`: a ``GraphRunner`` launchable whose ``enqueue`` runs the stages in
order, recordable into a caller's CUDA graph like ``TrackGenerator.generate``.
``TrackGenerator`` builds its pipeline with :func:`legacy.to_pipeline_spec
<track_gen.pipeline.legacy.to_pipeline_spec>` and :func:`build`.

Preview: names and signatures may change before roadmap PR10b. Only the generator
stage may come from a plugin in this release; the other stages are TrackGen's
built-ins. ``import track_gen`` does not import this package.
"""
from __future__ import annotations

from . import legacy
from .planner import PlanError, build
from .program import Program
from .spec import BatchSpec, PipelineSpec, StageRef

__all__ = [
    "BatchSpec",
    "PipelineSpec",
    "PlanError",
    "Program",
    "StageRef",
    "build",
    "legacy",
]
