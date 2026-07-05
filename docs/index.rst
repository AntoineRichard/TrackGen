TrackGen
========

**GPU-batched generation of closed-loop race tracks** — thousands of smooth,
validated tracks per ``generate()`` call, expressed as NVIDIA Warp kernels and
ready to drop into a batched RL simulator.

.. grid:: 3
   :gutter: 2

   .. grid-item::
      :columns: auto

      .. button-ref:: getting-started/installation
         :ref-type: doc
         :color: primary

         Get started

   .. grid-item::
      :columns: auto

      .. button-ref:: generators/overview
         :ref-type: doc
         :color: primary
         :outline:

         Browse generators

   .. grid-item::
      :columns: auto

      .. button-link:: https://github.com/AntoineRichard/TrackGen
         :color: secondary
         :outline:

         GitHub

.. figure:: assets/readme-pipeline-stages.png
   :alt: TrackGen pipeline stages
   :align: center

   The runtime pipeline turns a raw Phase-1 centerline into a constant-spacing path,
   relaxes it with XPBD, then inflates it into a constant-width road band.

Features
--------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: GPU-batched

      Generate ``E`` tracks in parallel per ``generate()`` call — the Warp ``cpu``
      device for tests/CI, ``cuda`` for production.

   .. grid-item-card:: Pure Warp pipeline

      Generation → constant-spacing resample → XPBD relaxation → inflation, every
      stage a Warp kernel over flat ``[E*N]`` arrays.

   .. grid-item-card:: Six generators

      Bezier, Hull, Polar, Voronoi, Checkpoint, and Repulsive — pluggable first-stage
      generators, each with a distinct shape family.

   .. grid-item-card:: CUDA-graph capture

      The whole pipeline is captured once into a replayable CUDA graph and replayed
      on every later call for high throughput (all generators except ``repulsive``,
      which runs eagerly every call — see :doc:`generators/repulsive`).

   .. grid-item-card:: Gate sequences

      Drone-style gate courses — gate centres and orientations — straight from the
      first-stage anchors via ``GateGenerator``.

   .. grid-item-card:: RL-ready runtime

      Out-of-bounds collision, checkpoint progress and rewards, prop
      instancing — and one Course object that bundles them.

Gallery
-------

.. figure:: assets/readme-generator-grid.png
   :alt: Six generators, one batch
   :align: center

   Six generators, one batch — representative raw Phase-1 centerlines from each
   standard generator.

.. figure:: assets/progress-tracking.png
   :alt: Progress tracking on a generated track.
   :align: center

   Runtime utilities in action: checkpoint progress on a generated track
   with the reward-ready ``dist_to_next`` signal.

.. figure:: assets/readme-gate-strip.png
   :alt: Gate sequences with collision relaxation
   :align: center

   Gate sequences with the phase-2 collision solve — raw anchors (top) versus
   separated gates (bottom).

Explore the docs
----------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting-started/installation
      :link-type: doc

      Install the library and generate your first batch.

   .. grid-item-card:: Tutorials
      :link: tutorials/batch-of-tracks
      :link-type: doc

      End-to-end recipes for tracks, gates, and CUDA-graph sim loops.

   .. grid-item-card:: Generators
      :link: generators/overview
      :link-type: doc

      How each first-stage generator works and when to use it.

   .. grid-item-card:: Runtime utilities
      :link: utilities/overview
      :link-type: doc

      Collision, props, checkpoints & progress, and the Course facade.

   .. grid-item-card:: How it works
      :link: how-it-works/pipeline
      :link-type: doc

      The pipeline, XPBD relaxation, inflation, and CUDA-graph capture.

   .. grid-item-card:: Configuration & tuning
      :link: configuration/tuning
      :link-type: doc

      Every knob, plus a guide to trading yield, diversity, and throughput.

   .. grid-item-card:: API reference
      :link: reference/api
      :link-type: doc

      ``TrackGenerator``, ``GateGenerator``, configs, and result types.

.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   getting-started/installation
   getting-started/quickstart
   getting-started/parameter-explorer

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/batch-of-tracks
   tutorials/gate-sequences
   tutorials/choosing-a-generator
   tutorials/cuda-graph-in-a-sim

.. toctree::
   :maxdepth: 1
   :caption: Generators
   :hidden:

   generators/overview
   generators/bezier
   generators/hull
   generators/polar
   generators/voronoi
   generators/checkpoint
   generators/repulsive
   generators/benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Runtime utilities
   :hidden:

   utilities/overview
   utilities/collision
   utilities/props
   utilities/progress
   utilities/localize
   utilities/course

.. toctree::
   :maxdepth: 1
   :caption: How it works
   :hidden:

   how-it-works/pipeline
   how-it-works/resample
   how-it-works/relaxation
   how-it-works/inflation
   how-it-works/cuda-graph
   how-it-works/conventions

.. toctree::
   :maxdepth: 1
   :caption: Configuration & tuning
   :hidden:

   configuration/reference
   configuration/tuning

.. toctree::
   :maxdepth: 1
   :caption: API reference
   :hidden:

   reference/api

.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :hidden:

   contributing/writing-a-generator
   contributing/dev-setup
   contributing/rendering-assets

.. toctree::
   :maxdepth: 1
   :caption: Related work
   :hidden:

   related-work/prior-art
   related-work/state-of-the-art

.. toctree::
   :maxdepth: 1
   :caption: Appendix
   :hidden:

   appendix/future-generators
