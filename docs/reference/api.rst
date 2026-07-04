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

.. figure:: ../assets/utilities-overview.png
   :alt: Cones and walls placed by track_gen.props, spacing comparison, and
         the collision SDF field with boxes classified by the exact backend.

   The query/instancing utilities on one generated track: ``track_gen.props``
   cones (points mode) and wall pieces (segments mode) on both boundaries,
   the effect of the spacing knob, and ``track_gen.collision``'s baked SDF
   field with boxes classified by the exact segments backend (green inside,
   red out of bounds, dotted lines to the nearest boundary point).
   Regenerate with ``python -m viz.render_utility_assets``.

.. automodule:: track_gen.collision
   :no-members:

.. autoclass:: track_gen.collision.CollisionChecker
   :members:

.. autoclass:: track_gen.collision.BoxContact
   :no-members:

   .. automethod:: clone

.. autoclass:: track_gen.collision.DiscChecker
   :members:

.. autoclass:: track_gen.collision.DiscContact
   :no-members:

   .. automethod:: clone

Performance
~~~~~~~~~~~

Measured numbers, backend trade-offs, and the reproduce command live in
:doc:`the collision utility page </utilities/collision>`.

Boundary props
--------------

Rendering-only instancing poses along track boundaries — cone lines
(``mode="points"``) and wall pieces (``mode="segments"``). Complementary to
the collision utility: props never collide. Results for invalid envs
(``Track.valid == 0``) are undefined — always gate on ``valid``.

.. automodule:: track_gen.props
   :no-members:

.. autoclass:: track_gen.props.PropSampler
   :members:

.. autoclass:: track_gen.props.PropSet
   :no-members:

   .. automethod:: clone

Checkpoints
-----------

Ordered course goals from gates (zero-copy) or subsampled track centerlines.

.. automodule:: track_gen.checkpoints
   :no-members:

.. autoclass:: track_gen.checkpoints.CheckpointSampler
   :members:

.. autoclass:: track_gen.checkpoints.CheckpointSet
   :no-members:

   .. automethod:: from_gates

   .. automethod:: clone

Progress tracking
-----------------

Stateful per-env course progress over any ``CheckpointSet``.

.. automodule:: track_gen.progress
   :no-members:

.. autoclass:: track_gen.progress.ProgressTracker
   :members:

.. autoclass:: track_gen.progress.ProgressEvents
   :no-members:

   .. automethod:: clone

Course facade
-------------

One object bundling generation, collision, and progress per mode; see the
:doc:`Course facade page </utilities/course>` for the lifecycle.

.. automodule:: track_gen.course
   :no-members:

.. autoclass:: track_gen.course.CourseConfig
   :no-members:

.. autoclass:: track_gen.course.Course
   :members:

.. autoclass:: track_gen.course.StepResult
   :no-members:

   .. automethod:: clone

.. autofunction:: track_gen.course.set_capturing
