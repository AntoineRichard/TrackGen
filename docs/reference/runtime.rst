.. _reference-runtime:

Courses and runtime utilities
=============================

The objects that run alongside your simulator: the ``Course`` (one generator plus its
:term:`components <component>`, see :ref:`courses`), the component protocol and its
adapters (:ref:`utilities-components`), and the runtime utilities the adapters wrap
(collision, props, checkpoints, progress, localization, heightfields). Each one reads a
generator's result and answers for every :term:`env` at once; an :term:`invalid env`
reads as an env with no track (:ref:`utilities-invalid-envs`). ``import track_gen``
imports all of these modules.

Courses
-------

The composition root: a generator (any :class:`~track_gen.pipeline.Producer`) plus an
ordered list of components, with the :term:`presets <preset>` ``Course.track`` /
``Course.gates`` and the 0.x ``Course(CourseConfig)`` (``Course.from_config``;
``config=`` is the 0.x keyword of the constructor). See :ref:`courses` for the lifecycle
and :ref:`composing-a-course` for composing your own.

.. automodule:: track_gen.course
   :no-members:

.. autoclass:: track_gen.course.Course
   :members:

.. autoclass:: track_gen.course.CourseConfig
   :no-members:

.. autoclass:: track_gen.course.StepResult
   :no-members:

   .. automethod:: __getitem__

   .. automethod:: get

   .. automethod:: keys

   .. automethod:: values

   .. automethod:: items

   .. automethod:: clone

Components
----------

The component protocol and the adapters of the runtime utilities; see
:ref:`writing-a-component` to write your own.

.. automodule:: track_gen.tools
   :no-members:

.. autoclass:: track_gen.tools.Component
   :members:

.. autoclass:: track_gen.tools.ComponentContext
   :members:

.. autoclass:: track_gen.tools.InputSpec
   :no-members:

   .. automethod:: check

Adapters
~~~~~~~~

Each adapter wraps one runtime utility (``.utility``) and exposes its result
(``.result``).

.. autoclass:: track_gen.tools.BandCollision
   :no-members:

.. autoclass:: track_gen.tools.DiscCollision
   :no-members:

.. autoclass:: track_gen.tools.FrameCollision
   :no-members:

.. autoclass:: track_gen.tools.Props
   :no-members:

.. autoclass:: track_gen.tools.Checkpoints
   :no-members:

.. autoclass:: track_gen.tools.GateCheckpoints
   :no-members:

.. autoclass:: track_gen.tools.Progress
   :no-members:

.. autoclass:: track_gen.tools.Localizer
   :no-members:

.. autoclass:: track_gen.tools.HeightFieldBake
   :no-members:

.. autoclass:: track_gen.tools.CenterlineFromGates
   :no-members:

Collision queries
-----------------

Box-vs-track out-of-bounds checks with full contact info, and the same box query against
disc obstacles (gate posts, physical cones, point props). :ref:`collision-backends`
compares the backends (the exact ``segments`` scan and the baked ``sdf`` grid), their
accuracy and their cost.

.. figure:: ../assets/utilities-overview.png
   :alt: Cones and walls placed by track_gen.props, spacing comparison, and
         the collision SDF field with boxes classified by the exact backend.

   The query and instancing utilities on one generated track: ``track_gen.props``
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

Boundary props
--------------

Rendering-only instancing poses along track boundaries: cone lines
(``mode="points"``) and wall pieces (``mode="segments"``). Props never collide. An
invalid env (``Track.valid == 0``) gets no props (``count = 0``), like an env with no
track.

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

Per-env ``(s, n)`` centerline projection with an optional warm start, plus the
per-:term:`generation` curvature and speed-profile helpers.

.. automodule:: track_gen.localize
   :no-members:

.. autoclass:: track_gen.localize.TrackLocalizer
   :members:

.. autoclass:: track_gen.localize.TrackFrame
   :no-members:

   .. automethod:: clone

.. autofunction:: track_gen.localize.curvature

.. autofunction:: track_gen.localize.speed_profile

Heightfields
------------

A per-env road-surface height grid, baked after every generation.

.. automodule:: track_gen.heightfield
   :no-members:

.. autoclass:: track_gen.heightfield.HeightField
   :members:

.. autoclass:: track_gen.heightfield.HeightFieldBaker
   :members:
