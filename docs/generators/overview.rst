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

Generator construction cost
---------------------------

Each generator produces **256 raw centerline points**, including its own curve
construction and internal resampling. Shared constant-spacing resampling, XPBD
relaxation and its smoothing/polish tail, elevation, road borders and final road
validity checks are outside the timed operation.

Measured 26 October 2026 with TrackGen 0.3.0rc3 and Warp 1.18.0. Each cell
shows **median batch time** and attempted centerlines per second. CPU batches
contain **64 candidates**; GPU batches contain **4,096**. Values are rounded to
three significant digits.

.. list-table:: Generator construction before shared road postprocessing
   :header-rows: 1
   :widths: 16 28 28 28

   * - Generator
     - Mac M5 Pro CPU (64)
     - Spark GB10 GPU (4,096)
     - RTX 5090 GPU (4,096)
   * - :doc:`Bézier <bezier>`
     - | **8.88 ms**
       | 7,210 centerlines/s
     - | **2.44 ms**
       | 1,680,000 centerlines/s
     - | **0.913 ms**
       | 4,490,000 centerlines/s
   * - :doc:`Hull <hull>`
     - | **10.2 ms**
       | 6,260 centerlines/s
     - | **3.65 ms**
       | 1,120,000 centerlines/s
     - | **1.42 ms**
       | 2,890,000 centerlines/s
   * - :doc:`Polar <polar>`
     - | **1.37 ms**
       | 46,600 centerlines/s
     - | **1.58 ms**
       | 2,590,000 centerlines/s
     - | **0.738 ms**
       | 5,550,000 centerlines/s
   * - :doc:`Voronoi <voronoi>`
     - | **15.9 ms**
       | 4,030 centerlines/s
     - | **8.21 ms**
       | 499,000 centerlines/s
     - | **4.62 ms**
       | 888,000 centerlines/s
   * - :doc:`Checkpoint <checkpoint>`
     - | **33.9 ms**
       | 1,890 centerlines/s
     - | **7.46 ms**
       | 549,000 centerlines/s
     - | **2.25 ms**
       | 1,820,000 centerlines/s
   * - :doc:`Repulsive <repulsive>`
     - | **144 s**
       | 0.444 centerlines/s
     - | **38.4 s**
       | 107 centerlines/s
     - | **11.6 s**
       | 354 centerlines/s

CPU uses completed eager wall time; GPUs use CUDA events around a warmed Warp graph
replay. Each operation advances the seed epoch, latches generator parameters and
executes the bound generator. The five measured calls use fresh epochs. Throughput
counts every attempted centerline; intrinsic generator status does not establish
finished-road validity.

Batch size affects throughput, so these timings describe different operating
points. Initialization, compilation, constructor warm-up and initial graph capture
are outside steady timing. Repulsive includes its own iterative growth and
obstacle/wall constraints. Measure the complete road pipeline separately when
budgeting a Racing reset.

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
