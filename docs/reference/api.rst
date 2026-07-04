API reference
=============

Runtime facades
---------------

.. autoclass:: track_gen.TrackGenerator
   :members:

.. autoclass:: track_gen.GateGenerator
   :members:

Seeded RNG
----------

.. autoclass:: track_gen.PerEnvSeededRNG
   :members:

Result types
------------

.. autoclass:: track_gen.Track
   :no-members:

   .. automethod:: clone

.. autoclass:: track_gen.GateSequence
   :no-members:

   .. automethod:: clone

Collision queries
-----------------

Box-vs-track out-of-bounds checks with full contact info. See
``track_gen.collision`` for backend trade-offs (exact ``segments`` scan vs
baked ``sdf`` grids).

.. automodule:: track_gen.collision
   :no-members:

.. autoclass:: track_gen.collision.CollisionChecker
   :members:

.. autoclass:: track_gen.collision.BoxContact
   :no-members:

   .. automethod:: clone
