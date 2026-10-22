"""The track pipeline as data (preview, roadmap PR10).

A :class:`PipelineSpec` lists the stages of the track pipeline in order
(:class:`StageRef`: kind, name, params) over one :class:`BatchSpec`. :func:`build` plans
it (every channel a stage reads has an earlier writer, every channel is allocated once,
each stage's scratch is namespaced, validity stages get their fail bits), runs each
stage's ``build`` once and returns a :class:`Program`: a ``GraphRunner`` launchable
whose ``enqueue`` runs the stages in order, recordable into a caller's CUDA graph like
``TrackGenerator.generate``. :func:`presets.track <track_gen.pipeline.presets.track>`
builds the default track spec with one argument per replaceable stage, and
``TrackGenerator(spec)`` runs any spec.

The ``relax``, ``width``, ``elevation`` and ``validity`` stages come from
:data:`track_gen.plugin.registry` (TrackGen's built-ins and third-party plugins); the
others (``resample``, ``frames``, ``offset``, ``lift``) are TrackGen's built-ins.
:data:`BUILTIN_FAIL_BITS` names the ``valid.fail`` bits of the built-in checks.

Preview: names and signatures may change before 1.0. ``import track_gen`` does not
import this package.
"""
from __future__ import annotations

from ..stages.channels import BUILTIN_FAIL_BITS
from . import legacy, presets
from .planner import PlanError, build
from .program import Program
from .spec import BatchSpec, PipelineSpec, StageRef

__all__ = [
    "BUILTIN_FAIL_BITS",
    "BatchSpec",
    "PipelineSpec",
    "PlanError",
    "Program",
    "StageRef",
    "build",
    "legacy",
    "presets",
]
