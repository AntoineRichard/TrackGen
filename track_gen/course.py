"""Public course facade: generation + collision + progress in one object.

See :class:`Course` for the lifecycle (construct -> bind -> step/reset, with
whole-batch ``regenerate()`` / ``generate(seeds=...)``). ``step()``,
``reset()``, ``regenerate()`` and ``generate()`` can be recorded into a
caller's CUDA graph as-is; ``set_capturing`` is a deprecated no-op kept for
compatibility.
"""
from ._src.course import Course, CourseConfig, StepResult, set_capturing

__all__ = ["Course", "CourseConfig", "StepResult", "set_capturing"]
