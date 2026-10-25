"""Public course facade: a generator and its runtime components in one object.

:class:`Course` is the composition root: ``Course(generator,
components)`` over any producer and any :mod:`track_gen.tools` components, the presets
``Course.track`` / ``Course.gates``, and ``Course.from_config(CourseConfig)`` (the 0.x
``Course(CourseConfig)``, unchanged). See :class:`Course` for the lifecycle (construct ->
bind -> step/reset, with ``regenerate()`` / ``generate(seeds=...)``). ``step()``,
``reset()``, ``regenerate()`` and ``generate()`` can be recorded into a caller's CUDA
graph as-is; :class:`StepResult` gives the 0.x fields plus read-only access to every
per-step component's result by name.
"""
from ._src.course import Course, CourseConfig, StepResult

__all__ = ["Course", "CourseConfig", "StepResult"]
