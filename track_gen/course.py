"""Public course facade: generation + collision + progress in one object.

See :class:`Course` for the lifecycle (bind -> generate -> step/reset) and
``set_capturing`` for one-switch CUDA-graph capture of the step path.
"""
from ._src.course import CourseConfig, StepResult, set_capturing

__all__ = ["CourseConfig", "StepResult", "set_capturing"]
