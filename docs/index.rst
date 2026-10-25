TrackGen
========

**Build race tracks and drone gate courses for batched simulation.** Choose a shape,
control its width and difficulty, and use the generated geometry with collision,
checkpoints and progress on CPU or CUDA.

.. grid:: 2
   :gutter: 2

   .. grid-item::

      .. button-ref:: quickstart
         :ref-type: ref
         :color: primary
         :expand:

         Create a race track

   .. grid-item::

      .. button-ref:: quickstart-gates
         :ref-type: ref
         :color: primary
         :outline:
         :expand:

         Create a gate course

.. figure:: assets/racing/generators-gallery.png
   :alt: Finished road bands from the six procedural generator families, with centerlines and labeled examples.

   Six different ways to build a racing circuit. Start with the result you want in
   :ref:`generators-overview`; each generator guide shows its centerlines,
   construction steps and settings.

Find your next step
-------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Choose a generator
      :link: generators-overview
      :link-type: ref

      Compare real tracks, understand each algorithm and see what its controls change.

   .. grid-item-card:: Shape a course
      :link: course-controls
      :link-type: ref

      Visual guides to size, road width, sampling, capacity, relaxation and elevation.

   .. grid-item-card:: Connect your simulator
      :link: courses
      :link-type: ref

      Bind agent buffers once; step progress and collision, then reset or regenerate
      selected environments.

   .. grid-item-card:: Understand collisions and failures
      :link: collision-backends
      :link-type: ref

      See vehicle footprints, boundaries and distance grids. Learn which checks reject
      generated geometry and which queries detect an agent leaving it.

   .. grid-item-card:: Use your own geometry
      :link: importing-a-track
      :link-type: ref

      Load a circuit from your data, make a road or place gates along it.

   .. grid-item-card:: Extend TrackGen
      :link: extending-trackgen
      :link-type: ref

      Add a generator, a pipeline stage or a runtime component using public APIs.

What your simulator receives
-----------------------------

A ``Track`` contains a centerline, road borders and local frames. A ``GateSequence``
contains gate positions and orientations. Both use fixed-capacity device arrays, one
row per environment, with explicit real counts and validity. :ref:`results` shows the
format visually and gives the reading pattern.

A :ref:`Course <courses>` adds the runtime operations: checkpoint progress,
localization, collisions, resets and regeneration. Construction allocates storage;
repeated calls reuse stable buffers and can be recorded in a CUDA graph.

**Always read validity.** A generator can produce an invalid environment; it does not
retry automatically. Its geometry is poisoned by default, while failure bits retain the
cause. :ref:`invalid-envs` explains how to diagnose and regenerate it.

.. toctree::
   :maxdepth: 1
   :caption: Start racing
   :hidden:

   getting-started/installation
   getting-started/quickstart
   getting-started/gates
   getting-started/concepts
   user-guide/results

.. toctree::
   :maxdepth: 1
   :caption: Choose a generator
   :hidden:

   generators/overview
   tutorials/choosing-a-generator
   getting-started/parameter-explorer

.. toctree::
   :maxdepth: 1
   :caption: Shape your course
   :hidden:

   configuration/tuning
   tracks-25d
   gates-3d
   user-guide/generating
   user-guide/curricula
   tutorials/batch-of-tracks
   tutorials/gate-sequences

.. toctree::
   :maxdepth: 1
   :caption: Use in a simulator
   :hidden:

   utilities/course
   utilities/composing-a-course
   utilities/overview
   tutorials/cuda-graph-in-a-sim
   user-guide/rl-auto-reset

.. toctree::
   :maxdepth: 1
   :caption: Build and extend
   :hidden:

   contributing/overview
   contributing/importing-a-track
   contributing/writing-a-generator
   contributing/writing-a-gate-generator
   contributing/writing-a-stage
   contributing/writing-a-component
   contributing/testing-a-plugin
   contributing/packaging-a-plugin
   contributing/plugin-abis

.. toctree::
   :maxdepth: 1
   :caption: API and contracts
   :hidden:

   reference/api
   configuration/reference
   reference/result-contracts

.. toctree::
   :maxdepth: 1
   :caption: How it works
   :hidden:

   how-it-works/pipeline
   how-it-works/pipeline-stages
   how-it-works/gate-pipeline
   how-it-works/cuda-graph
   how-it-works/resample
   how-it-works/inflation
   how-it-works/conventions
   relaxation/overview
   relaxation/gates

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   contributing/dev-setup
   contributing/rendering-assets
