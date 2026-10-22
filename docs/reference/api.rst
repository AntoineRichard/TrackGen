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

Seed state
----------

Per-env seeds and device-side epochs, the pair that keys every generation; see
:doc:`/how-it-works/cuda-graph`.

.. automodule:: track_gen.seeds
   :no-members:

.. autoclass:: track_gen.seeds.SeedState
   :members:

.. autofunction:: track_gen.seeds.effective_seed

Execution layer
---------------

Stream resolution, the record / eager / self-graph policy shared by the producers, the
device-side conditional skip, and the capture errors; see :doc:`/how-it-works/cuda-graph`.

.. automodule:: track_gen.exec
   :no-members:

.. autoclass:: track_gen.exec.LaunchContext
   :members:

.. autoclass:: track_gen.exec.GraphRunner
   :members:

.. autofunction:: track_gen.exec.resolve_stream

.. autoexception:: track_gen.exec.CaptureError

.. autoexception:: track_gen.exec.NotCapturableError

.. autoexception:: track_gen.exec.LegacyPluginStreamError

Device ops
----------

.. automodule:: track_gen.ops
   :no-members:

.. autofunction:: track_gen.ops.any_set

.. autofunction:: track_gen.ops.fill

Torch interop
-------------

Registering a ``torch.cuda.graph`` capture with Warp, and tying producers' lifetime to a
torch graph; see :doc:`/tutorials/cuda-graph-in-a-sim`. Importing ``track_gen`` does not
import torch.

.. automodule:: track_gen.interop.torch
   :no-members:

.. autofunction:: track_gen.interop.torch.registered

.. autofunction:: track_gen.interop.torch.retain

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

Track-frame localization
------------------------

Per-env (s, n) centerline projection with an optional warm start, plus the
per-generation curvature and speed-profile helpers.

.. automodule:: track_gen.localize
   :no-members:

.. autoclass:: track_gen.localize.TrackLocalizer
   :members:

.. autoclass:: track_gen.localize.TrackFrame
   :no-members:

   .. automethod:: clone

.. autofunction:: track_gen.localize.curvature

.. autofunction:: track_gen.localize.speed_profile

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
