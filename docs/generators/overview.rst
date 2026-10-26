.. _generators-overview:

Choose a generator
==================

TrackGen ships six procedural road families. Start with a shape below, then open its
guide to see raw centerlines, finished roads, same-seed controls and construction.
All six feed the shared resample → relax → width/offset → validity pipeline.

.. figure:: ../assets/racing/generators-gallery.png
   :alt: Finished roads from Bezier, Hull, Polar, Voronoi, Checkpoint and Repulsive.
   :align: center

   Current CPU output at default geometry and equal coordinate scale. Each is a
   selected valid example, not a family success-rate or quality ranking. The blue
   line is the final road centerline.

.. list-table:: Find a useful starting family
   :header-rows: 1
   :widths: 18 42 40

   * - Guide
     - What constructs the shape
     - Start here for
   * - :doc:`bezier`
     - Sampled corners joined by cubic handles
     - Corner-count and tangent/handle control
   * - :doc:`hull`
     - Angle-sorted anchors plus displaced midpoints
     - Inward/outward lobes and pinches
   * - :doc:`polar`
     - Jittered radial knots and a periodic spline
     - Direct control of radial and angular disorder
   * - :doc:`voronoi`
     - Sites selected against an angular/radial target cycle
     - Different site layouts and irregular anchor cycles
   * - :doc:`checkpoint`
     - Bounded-turn walks, closure and candidate selection
     - Continuously steered target-following layouts
   * - :doc:`repulsive`
     - Iterative self-repelling curve growth among obstacles
     - Folded loops when iterative generation fits the reset budget

:doc:`/tutorials/choosing-a-generator` connects these choices to gates, curricula and
simulator reset cadence. If you already have the geometry, use ``external``:
:ref:`ingesting-centerlines`.

.. _generator-batch-benchmarks:

Measured generation cost
-------------------------

Warm, complete road-generation calls at the Racing quickstart settings, measured on
26 October 2026 with TrackGen 0.3.0rc3 and Warp 1.18.0. Each cell shows **median
batch time**, attempted tracks per second, and the percentage passing final validity.
CPU batches contain **64 candidates**; GPU batches contain **4,096**.

.. list-table:: Finished-road generation at the measured batch sizes
   :header-rows: 1
   :widths: 16 28 28 28

   * - Generator
     - Mac M5 Pro CPU (64)
     - Spark GB10 GPU (4,096)
     - RTX 5090 GPU (4,096)
   * - :doc:`bezier`
     - | **55.1 ms**
       | 1,160 tracks/s
       | 100.00% valid
     - | **70.6 ms**
       | 58,000 tracks/s
       | 99.95% valid
     - | **26.7 ms**
       | 153,000 tracks/s
       | 99.95% valid
   * - :doc:`hull`
     - | **59.3 ms**
       | 1,080 tracks/s
       | 100.00% valid
     - | **71.3 ms**
       | 57,400 tracks/s
       | 99.92% valid
     - | **25.9 ms**
       | 158,000 tracks/s
       | 99.92% valid
   * - :doc:`polar`
     - | **41.7 ms**
       | 1,530 tracks/s
       | 100.00% valid
     - | **61.4 ms**
       | 66,700 tracks/s
       | 100.00% valid
     - | **20.9 ms**
       | 196,000 tracks/s
       | 100.00% valid
   * - :doc:`voronoi`
     - | **51.2 ms**
       | 1,250 tracks/s
       | 100.00% valid
     - | **63.4 ms**
       | 64,600 tracks/s
       | 100.00% valid
     - | **21.2 ms**
       | 193,000 tracks/s
       | 100.00% valid
   * - :doc:`checkpoint`
     - | **73.9 ms**
       | 866 tracks/s
       | 99.38% valid
     - | **66.1 ms**
       | 62,000 tracks/s
       | 99.45% valid
     - | **21.5 ms**
       | 191,000 tracks/s
       | 99.45% valid
   * - :doc:`repulsive`
     - | **145 s**
       | 0.44 tracks/s
       | 99.06% valid
     - | **38.4 s**
       | 107 tracks/s
       | 98.72% valid
     - | **11.6 s**
       | 352 tracks/s
       | 98.72% valid

CPU uses completed eager wall time; GPUs use CUDA events around a warmed Warp graph
replay. The five measured calls advance to fresh epochs. Failed candidates stay in
the denominator; batches are never refilled to reach a number of valid roads.
Valid percentages describe these measured samples.

Batch size affects throughput: these are different operating points, rather than
latencies for equal amounts of work. Initialization, compilation, construction and
initial capture are outside the steady times. Repulsive includes its iterative
growth solve and mandatory full-road disc/wall clearance; budget it for offline or
infrequent generation.

:doc:`benchmarks` gives the exact protocol and interpretation. Download the
:download:`configurations, individual samples and hardware metadata <../_data/generator-benchmarks.json>`
to inspect the measurements or reproduce the comparison.

Try a family
------------

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   gen = TrackGenerator(presets.track(
       generator="hull", num_envs=4, device="cpu"), seeds=7)
   track = gen.generate()
   print(track.count.numpy(), track.valid.numpy())

.. snippet: after

   assert track.valid.numpy().any()

Use ``device="cuda:0"`` for a CUDA batch. A seed/configuration is reproducible on a
given device; equality across CPU and CUDA is not promised. ``generate()`` reuses
result storage. Check validity and real counts before consuming a row, and clone
results you need to retain: :doc:`/user-guide/results`.

Roads and gate courses
----------------------

``bezier``, ``hull``, ``polar``, ``voronoi`` and ``checkpoint`` also have **native gate
anchors**. Those gates use sampled corners, knots, selected sites or radial targets;
they are not automatically stations on the track's spline or steered road. Native
track/gate controls differ, as each family guide explains.

Every track generator, including ``repulsive`` and ``external``, can supply
``source="stations"`` gates along its source centerline. Use
``source="relaxed_stations"`` when stations should follow the relaxed centerline.
See :doc:`/tutorials/gate-sequences` for spacing, elevation and ordering.

.. _generators-per-env:

Per-environment controls
------------------------

Dynamic float32 controls can be set per env or drawn by a uniform randomizer. Dynamic
integer controls use ``set``; counts and capacities that are static require rebuilding.

.. list-table:: Dynamic track-generator fields
   :header-rows: 1
   :widths: 18 62 20

   * - Family
     - Generator fields
     - Details
   * - Bezier
     - rad, handle_clamp_frac, size; min_num_points (int32)
     - :ref:`bezier-dynamic`
   * - Hull
     - hull_displacement, size; min_num_points (int32)
     - :ref:`hull-dynamic`
   * - Polar
     - polar_radial_jitter, polar_angular_jitter, size
     - :ref:`polar-dynamic`
   * - Voronoi
     - voronoi_radial_variation, voronoi_angular_jitter, size
     - :ref:`voronoi-dynamic`
   * - Checkpoint
     - checkpoint_radius_min_frac, checkpoint_angle_jitter, checkpoint_turn_rate, size
     - :ref:`checkpoint-dynamic`
   * - Repulsive
     - size (also affects raw width reservation and layout)
     - :ref:`repulsive-dynamic`
   * - External
     - none; upload geometry directly
     - :ref:`ingesting-centerlines`

``size`` scales the XY layout while width, spacing and altitude stay absolute.
Repulsive also uses it for width reservation, which can change the raw layout
and shape (:ref:`repulsive-dynamic`). A longer loop needs more output slots;
budget ``N_max`` for the largest size:
:ref:`tuning-size`. Native polar/voronoi/checkpoint gates instead use
``normalize.size``. Native Bezier/Hull gate count bounds are static.

For randomization, ownership and masked regeneration, follow
:doc:`/user-guide/curricula`. Every field's schema and defaults are in
:ref:`builtin-generator-params`; shared stages are in :doc:`/configuration/reference`.

Validity and generation cost
----------------------------

The sampling/steering families write a raw closed loop and ``gen.valid=1``.
Repulsive requires finite feasible raw geometry at its sampled target length;
its optional freeze state is separate from validity. Final road validity is
checked after the shared pipeline. Repulsive additionally certifies full-width
disc and wall clearance on the finished band. A generator-local fallback alone
is not a road-clearance certificate. Count, padding, poison defaults and
``fail_mask`` are part of the :doc:`/user-guide/results` contract.

Sampling/steering families use bounded sampling and geometry work. ``repulsive``
additionally runs an iterative growth solve with quadratic self-pair work at each
active resolution. Measure its construction, capture and steady regeneration costs
separately. Use :doc:`benchmarks` to measure current configurations on your target
machine; the selected pictures here make no numeric throughput or yield claims.

Extend the catalogue with :doc:`/contributing/writing-a-generator` or import an
existing centerline with :ref:`ingesting-centerlines`. Available registered names
come from ``track_gen.plugin.registry.available("generator")``.

.. toctree::
   :maxdepth: 1
   :hidden:

   bezier
   hull
   polar
   voronoi
   checkpoint
   repulsive
   benchmarks
