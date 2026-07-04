"""Public progress-tracking API: ordered course progress + reward signals.

``ProgressTracker`` consumes any ``CheckpointSet`` (gate sequences via
``CheckpointSet.from_gates``, or subsampled track centerlines via
``CheckpointSampler``) and emits per-step ``ProgressEvents``: pass events,
laps, wrong-way / wrong-checkpoint flags, and ``dist_to_next`` — difference
it across steps for the classic negative-delta-distance reward.
"""
from ._src.progress import ProgressEvents, ProgressTracker

__all__ = ["ProgressEvents", "ProgressTracker"]
