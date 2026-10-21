"""Sync-free, allocation-free device primitives. Every op takes a ``LaunchContext`` first."""
from .fill import fill

__all__ = ["fill"]
