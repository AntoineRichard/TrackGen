Hull: midpoint lobes
====================

Use ``hull`` for loops with displaced midpoint lobes, pinches and varied straights.
The name refers to a hull-like base loop: the implementation sorts sampled points by
angle rather than computing a geometric convex hull.

.. figure:: ../assets/racing/generator-hull-examples.png
   :alt: Three hull raw centerlines above the corresponding finished roads.
   :align: center

   Three selected valid seeds at default geometry, with equal axes. Blue upper panels show the generator output; lower panels show the final road. Counts exclude padding. These examples do not measure yield.

Try it
------

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   gen = TrackGenerator(presets.track(
       generator="hull", num_envs=4, device="cpu"), seeds=7)
   track = gen.generate()
   print(track.count.numpy(), track.valid.numpy())

.. snippet: after

   assert track.valid.numpy().any()

Change the shape
----------------

.. figure:: ../assets/racing/generator-hull-controls.png
   :alt: Same-seed midpoint-displacement and corner-count comparisons.
   :align: center

   The displacement row changes midpoint positions; count rows change the anchor cycle. Blue raw centerlines and orange final centerlines show what the shared road pipeline changes.

Panels name the selected smooth curve or polygon fallback. Final road validity is
reported separately; a valid panel can still show a generator fallback.

``hull_displacement=0`` keeps inserted midpoints on the base edges. Increasing it
moves each midpoint inward or outward along its centroid-relative radial direction.
This is one displacement layer; it is not recursive terrain subdivision.

.. _hull-dynamic:

Controls and defaults
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Track generator controls
   :header-rows: 1
   :widths: 30 18 52

   * - Setting
     - Default / kind
     - Units, range and practical effect
   * - ``hull_displacement``
     - 0.15 / dynamic
     - Radial midpoint-displacement fraction in [0, 1]. Larger displacement permits deeper lobes and pinches; it can also produce harder road-clearance problems.
   * - ``min_num_points``
     - 9 / dynamic int32
     - Lower sampled anchor-count bound in [2, max_num_points]. Change with set(); uniform randomizers draw float32 fields only.
   * - ``max_num_points``
     - 13 / static
     - Upper anchor-count bound and capacity, at least min_num_points. There are twice as many vertices after midpoint insertion.
   * - ``min_point_distance``
     - 0.05 / static
     - Normalized grid spacing in (0, 0.5]; int(1/(2*d)) cells per side. Changes anchor placement, not final road clearance.
   * - ``num_points_per_segment``
     - 30 / static
     - At least 2 dense Catmull–Rom samples per augmented segment; trades sampling work for discretization accuracy.
   * - ``size``
     - 1 / dynamic
     - XY multiplier in [0.1, 10], with absolute road width and spacing. See :ref:`tuning-size` for capacity.

.. figure:: ../assets/racing/generator-hull-controls-more.png
   :alt: Same-seed changes to the anchor grid.
   :align: center

   Grid spacing changes the available anchor positions; the random seed is held fixed.

.. figure:: ../assets/racing/generator-hull-sampling.png
   :alt: Hull dense-curve sampling resolution comparison.
   :align: center

   Dots are actual dense samples. Final bead spacing and N_max belong to the shared pipeline, independently of this sampling count.

When it fails
-------------

If the smoothed raw loop has a proper self-crossing, the generator uses the augmented
polygon through the anchors and displaced midpoints. That polygon may still have
close approaches or sharp turns. Final validity is decided after resampling,
relaxation and inflation. Reduce displacement, allow more geometric space relative
to width, or inspect ``fail_mask`` and output capacity before changing solver budgets.
Failed panels are labeled and use explicit ``on_invalid="keep"``.

Native ``hull`` gates use the initial sampled anchors, before midpoint displacement
and smoothing. Their count bounds are static and they have no displacement control.
Use track stations when gates should follow the finished lobes.

How it is constructed
---------------------

The anchor cycle is angle-sorted about its centroid. One midpoint is inserted per
edge and displaced radially by a seed-dependent signed amount. A closed Catmull–Rom
curve passes through this augmented cycle; dense samples are arc-resampled. The
crossing check chooses the augmented polygon fallback when the smooth curve crosses.
The shared tail then determines whether the finite-width road is valid.

.. figure:: ../assets/racing/generator-hull-construction.png
   :alt: Hull construction buffers, generated centerline and finished road.
   :align: center

   Recorded construction geometry, followed by the selected generator centerline and the shared pipeline result. The left panel uses construction coordinates; the other panels use output coordinates.

.. raw:: html

   <video controls preload="none" style="width:100%;max-width:100%" poster="../_images/generator-hull-construction.png" aria-label="Three-stage reveal of actual hull construction buffers, generator centerline and finished road.">
     <source src="../generator-hull-construction.mp4" type="video/mp4">
     Three-stage reveal of actual hull construction buffers, generator centerline and finished road.
   </video>

:download:`Download the animation <../assets/racing/generator-hull-construction.mp4>`.

The animation reveals the three panels in order. It is an illustrative construction
sequence, not elapsed runtime or a trace of iterative solver states.

Continue with :doc:`/how-it-works/pipeline`, :doc:`/configuration/tuning`,
:ref:`builtin-generator-params`, or :doc:`/tutorials/gate-sequences`.
