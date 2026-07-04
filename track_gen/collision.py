"""Public collision-query API: box-vs-track out-of-bounds checks.

``CollisionChecker`` answers, for batches of oriented boxes against a batch of
generated tracks, whether each box has left the drivable band — with full
contact info (:class:`BoxContact`): OOB flag, signed clearance, nearest
boundary point, inward normal, and boundary id. It has two Warp backends:

- ``method="segments"`` (default): exact, zero precompute; reads the bound
  ``Track`` buffers directly (fresh after every ``generate()``).
- ``method="sdf"``: bakes per-env signed-distance grids for O(1) queries;
  approximate within one grid cell near boundaries and requires ``bake()``
  after each ``generate()``. Memory ~ ``E * sdf_resolution**2 * 5`` bytes.

``DiscChecker`` is a sibling checker in this module, not a third backend of
``CollisionChecker``: it answers the same oriented-box query against disc
obstacles instead of a track (gate posts, physical cones, point props).

This module is the template for future query utilities: each gets its own
public sibling module (flat namespace, no grab-bag ``utils``).
"""
from ._src.collision import BoxContact, CollisionChecker
from ._src.collision_discs import DiscChecker, DiscContact

__all__ = ["BoxContact", "CollisionChecker", "DiscChecker", "DiscContact"]
