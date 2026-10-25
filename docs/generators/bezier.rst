Bézier: corners and handles
===========================

Use ``bezier`` for circuits whose sampled corners are joined by cubic curves.
Corner count changes the layout; tangent direction and handle length change how the
curve enters and leaves each corner. The shared relaxation stage then makes room for
the road.

.. figure:: ../assets/racing/generator-bezier-examples.png
   :alt: Three bezier raw centerlines above the corresponding finished roads.
   :align: center

   Three selected valid seeds at default geometry, with equal axes. Blue upper panels show the generator output; lower panels show the final road. Counts exclude padding. These examples do not measure yield.

Try it
------

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   gen = TrackGenerator(presets.track(
       generator="bezier", num_envs=4, device="cpu"), seeds=7)
   track = gen.generate()
   print(track.count.numpy(), track.valid.numpy())

.. snippet: after

   assert track.valid.numpy().any()

Change the curve
----------------

.. figure:: ../assets/racing/generator-bezier-controls.png
   :alt: Same-seed changes to radius, handle clamp and tangent blend.
   :align: center

   Each row holds the seed and other controls fixed. Blue is the raw generator centerline; orange dashed is the final centerline inside the grey road. Default values are marked. Failed examples retain their geometry explicitly.

Panels name the selected smooth curve or polygon fallback independently of final
road validity.

``rad`` requests a handle length relative to its segment chord. The actual handle
also obeys ``handle_clamp_frac`` times the shorter incident edge. A larger ``rad``
can therefore stop changing the curve when the clamp is already active. ``edgy``
rotates the blended tangent toward the incoming or outgoing edge; it is not an
angle in radians.

.. _bezier-dynamic:

Controls and defaults
~~~~~~~~~~~~~~~~~~~~~

Dynamic fields can change per environment before the next generation; static fields
require rebuilding. All distances below are in generator coordinates unless stated.

.. list-table:: Track generator controls
   :header-rows: 1
   :widths: 30 18 52

   * - Setting
     - Default / kind
     - Units, range and practical effect
   * - ``rad``
     - 0.4 / dynamic
     - Chord fraction in [0, 1]. Longer requested handles round corners, subject to the clamp.
   * - ``handle_clamp_frac``
     - 0.4 / dynamic
     - Nonnegative fraction of the shorter incident edge. Lower values shorten handles near unevenly spaced corners.
   * - ``edgy``
     - 0 / static
     - Dimensionless tangent blend: p = atan(edgy)/π + 0.5. Zero balances incoming and outgoing directions.
   * - ``min_num_points``
     - 9 / dynamic int32
     - Lower bound of the sampled corner count, from 2 through max_num_points. Integer curricula use set(); uniform randomizers do not draw integers.
   * - ``max_num_points``
     - 13 / static
     - Corner capacity and sampled upper bound; at least min_num_points. More corners also increase dense-curve work and can lengthen the loop.
   * - ``min_point_distance``
     - 0.05 / static
     - Normalized grid spacing parameter in (0, 0.5]. Grid cells are int(1/(2*d)) per side. It changes the sampled corner grid, not a final clearance guarantee.
   * - ``num_points_per_segment``
     - 30 / static
     - At least 2 dense samples per cubic segment. Controls curve discretization before the fixed num_points arc-resample.
   * - ``size``
     - 1 / dynamic
     - XY multiplier in [0.1, 10]; width, spacing and altitude stay absolute. Budget output capacity for the largest size: :ref:`tuning-size`.

.. _bezier-corner-count:

Change the anchors
~~~~~~~~~~~~~~~~~~

.. figure:: ../assets/racing/generator-bezier-controls-more.png
   :alt: Same-seed corner-count bounds and grid-spacing comparisons.
   :align: center

   Changing a count bound can leave a particular random draw unchanged. The renderer selects a declared seed with a visible effect for each row; the manifest records the search and all settings.

.. figure:: ../assets/racing/generator-bezier-sampling.png
   :alt: Dense-curve sampling at 4, 30 and 60 samples per segment.
   :align: center

   Dots show the actual dense cubic samples. Increasing this count refines the curve representation; it does not increase final output capacity or set final bead spacing.

.. _bezier-style-vs-randomize:

For a curriculum, write ``gen.params["generator"].set("rad", 0.2)`` and regenerate,
or draw ``generator.rad`` with :doc:`/user-guide/curricula`. A randomizer owns the
fields it draws. Use ``min_num_points`` with ``set`` for integer corner-count changes.

When it fails
-------------

A smooth raw loop with a proper self-crossing is replaced by its angle-sorted corner
polygon before the shared tail. That fallback can introduce straights; it does not
certify that a road of the requested width fits. Final ``track.valid`` and
``track.fail_mask`` still decide which rows are usable. Narrow bends, close nonlocal
segments and insufficient ``N_max`` can fail after generation. Inspect failed rows
with explicit ``on_invalid="keep"``; the normal default poisons failed geometry.

Native ``bezier`` gates use sampled corner anchors directly. They have no ``rad``,
``edgy`` or handle-clamp controls, and their count bounds are static. To place gates
along the curved road, use station sampling instead.

How it is constructed
---------------------

The generator samples distinct grid cells, chooses a corner count, and sorts the
active corners by angle about their centroid. At each corner it blends the unit
incoming and outgoing edge directions. Each cubic joins two corners with handles
along those tangents, capped by local incident-edge lengths. Dense samples are
resampled uniformly in arclength to the raw centerline; a crossing check selects the
polygon fallback when needed. Angle sorting is a construction heuristic and does not
make this a convex-hull algorithm.

.. figure:: ../assets/racing/generator-bezier-construction.png
   :alt: Bezier construction buffers, generated centerline and finished road.
   :align: center

   Recorded construction geometry, followed by the selected generator centerline and the shared pipeline result. The left panel uses construction coordinates; the other panels use output coordinates.

.. raw:: html

   <video controls preload="none" style="width:100%;max-width:100%" poster="../_images/generator-bezier-construction.png" aria-label="Three-stage reveal of actual bezier construction buffers, generator centerline and finished road.">
     <source src="../generator-bezier-construction.mp4" type="video/mp4">
     Three-stage reveal of actual bezier construction buffers, generator centerline and finished road.
   </video>

:download:`Download the animation <../assets/racing/generator-bezier-construction.mp4>`.

The animation reveals the three panels in order. It is an illustrative construction
sequence, not elapsed runtime or a trace of iterative solver states.

Continue with :doc:`/how-it-works/pipeline`, :doc:`/configuration/tuning`,
:ref:`builtin-generator-params`, or :doc:`/tutorials/gate-sequences`.
