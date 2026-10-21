"""Public course facade: generation + collision + progress in one object.

See :class:`Course` for the lifecycle (bind -> generate -> step/reset).
``step()`` / ``reset()`` can be recorded into a caller's CUDA graph as-is;
``set_capturing`` is a deprecated no-op kept for compatibility.
"""
from ._src.course import Course, CourseConfig, StepResult, set_capturing

__all__ = ["Course", "CourseConfig", "StepResult", "set_capturing"]
