"""Public checkpoint API: ordered course goals from gates or tracks.

``CheckpointSet`` is the shared contract consumed by
``track_gen.progress.ProgressTracker``. Build one zero-copy from a
``GateSequence`` (``CheckpointSet.from_gates``) or by subsampling a track's
centerline at a coarse spacing (``CheckpointSampler`` — each checkpoint's
crossing segment is the road cross-section, a "virtual gate").
"""
from ._src.checkpoints import CheckpointSampler, CheckpointSet

__all__ = ["CheckpointSampler", "CheckpointSet"]
