Polar: radial knots
===================

Use ``polar`` for a loop built around a ring of angularly ordered control knots.
Radial jitter changes how far each knot sits from the origin; angular jitter changes
its location inside its angular slot. A periodic cubic curve rounds the knot cycle.

.. figure:: ../assets/racing/generator-polar-examples.png
   :alt: Three polar raw centerlines above the corresponding finished roads.
   :align: center

   Three selected valid seeds at default geometry, with equal axes. Blue upper panels show the generator output; lower panels show the final road. Counts exclude padding. These examples do not measure yield.

Try it
------

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   gen = TrackGenerator(presets.track(
       generator="polar", num_envs=4, device="cpu",
       generator_params={"polar_radial_jitter": 0.6}), seeds=7)
   track = gen.generate()
   print(track.count.numpy(), track.valid.numpy())

.. snippet: after

   assert track.valid.numpy().any()

Change the knots
----------------

.. figure:: ../assets/racing/generator-polar-controls.png
   :alt: Same-seed radial jitter, angular jitter and knot-count comparisons.
   :align: center

   The default is the middle panel in each row. Blue is the generator curve after extent normalization; orange and grey show its final centerline and road. All panels in a row share axes.

Angular jitter is a fraction of ``2π / polar_num_knots``, not radians. Changing
knot count changes this angular unit as well as the number of random knots. The
raw curve is normalized by its longest bounding-box side, so a radial change is a
shape change rather than a simple change in overall course size.

.. _polar-dynamic:

Controls and defaults
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Track generator controls
   :header-rows: 1
   :widths: 30 18 52

   * - Setting
     - Default / kind
     - Units, range and practical effect
   * - ``polar_radial_jitter``
     - 0.60 / dynamic
     - Fraction of base radius, [0, 0.85]. Each knot radius is base_radius * (1 + U(-j, j)).
   * - ``polar_angular_jitter``
     - 0.30 / dynamic
     - Fraction of one angular slot, [0, 0.45]. Moves knots around the ring without changing their assigned index order.
   * - ``polar_num_knots``
     - 12 / static
     - Knot count, clamped to at least 4. More knots give finer angular structure and more dense segments.
   * - ``num_points_per_segment``
     - 30 / static
     - At least 2 dense samples per spline segment. A discretization control, independent of final bead spacing.
   * - ``size``
     - 1 / dynamic
     - XY extent multiplier, [0.1, 10]. Width, spacing and z are absolute; see :ref:`tuning-size`.

.. figure:: ../assets/racing/generator-polar-sampling.png
   :alt: Polar curve discretization at 4, 30 and 60 samples per knot segment.
   :align: center

   The same knot draw is resampled from different dense spline resolutions. Small changes can survive the shared pipeline; a large dense count is not itself a road-quality guarantee.

When it fails
-------------

``polar`` has no generator-local crossing fallback. Large radial excursions can make
a tight bend or bring separated parts of the curve close together. The shared
relaxation and validity stages handle the finite-width road; a visually smooth loop
still needs ``track.valid``. Reduce jitter or change the width-to-size ratio if
clearance is the problem. Increase ``N_max`` if a longer loop is truncated.

Native ``polar`` gates are the control knots, not stations on the cubic curve. Their
knot count equals the gate count; they share the two jitter controls. Native gate
size belongs to ``normalize.size`` rather than the generator stage.

How it is constructed
---------------------

For knot i, the angle is 2π(i + angular_delta)/K and the radius has an independent
bounded radial perturbation. The generator evaluates the periodic Catmull–Rom
segments, arc-resamples the dense loop to ``num_points``, recenters its bounding box,
and normalizes the longest side to the shared target extent times ``scale * size``.
The shared resampling and relaxation stages follow.

.. figure:: ../assets/racing/generator-polar-construction.png
   :alt: Polar construction buffers, generated centerline and finished road.
   :align: center

   Recorded construction geometry, followed by the selected generator centerline and the shared pipeline result. The left panel uses construction coordinates; the other panels use output coordinates.

.. raw:: html

   <video controls preload="none" style="width:100%;max-width:100%" poster="../_images/generator-polar-construction.png" aria-label="Three-stage reveal of actual polar construction buffers, generator centerline and finished road.">
     <source src="../generator-polar-construction.mp4" type="video/mp4">
     Three-stage reveal of actual polar construction buffers, generator centerline and finished road.
   </video>

:download:`Download the animation <../assets/racing/generator-polar-construction.mp4>`.

The animation reveals the three panels in order. It is an illustrative construction
sequence, not elapsed runtime or a trace of iterative solver states.

Continue with :doc:`/how-it-works/pipeline`, :doc:`/configuration/tuning`,
:ref:`builtin-generator-params`, or :doc:`/tutorials/gate-sequences`.
