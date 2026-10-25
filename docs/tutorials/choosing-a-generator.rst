.. _choosing-a-generator:

Choose a course family
======================

Start with ``bezier`` for corner and handle control, ``polar`` for radial/angle
control, or ``checkpoint`` for a steered target-following shape. Add ``hull`` and
``voronoi`` when their anchor construction gives the variation you need. Try
``repulsive`` when folded layouts justify iterative growth during resets.

.. figure:: ../assets/racing/generators-gallery.png
   :alt: Six current finished road examples at a common coordinate scale.
   :align: center

   One selected valid output per family. The individual :doc:`/generators/overview`
   guides show raw centerlines, several seeds, controls and failure cases.

Choose for the task
-------------------

.. list-table:: Practical starting points
   :header-rows: 1
   :widths: 34 28 38

   * - Need
     - Start with
     - Then inspect
   * - General-purpose closed road
     - :doc:`/generators/bezier`
     - Corner count, handle saturation and road width
   * - Midpoint lobes and pinches
     - :doc:`/generators/hull`
     - Displacement and raw-to-final changes
   * - A ring of explicitly jittered knots
     - :doc:`/generators/polar`
     - Radial versus angular jitter; final tight bends
   * - Site-field variation
     - :doc:`/generators/voronoi`
     - Site layout, anchor count and crossing fallback
   * - A target-following steering walk
     - :doc:`/generators/checkpoint`
     - Turn rate, lookahead, closure and selected candidate
   * - A densely folded growth process
     - :doc:`/generators/repulsive`
     - Growth states, capacity and regeneration cadence
   * - A dataset or hand-designed circuit
     - ``external``
     - :ref:`ingesting-centerlines` and input channel ownership

These are choices of construction and control. Whether a road is drivable for your
vehicle also depends on width, curvature, validity and your dynamics. Evaluate the
result with the actual vehicle footprint and speed controller, rather than treating
a family label as a difficulty score.

Try the choice
--------------

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   gen = TrackGenerator(presets.track(
       generator="polar", num_envs=4, device="cpu",
       generator_params={"polar_radial_jitter": 0.3}), seeds=7)
   track = gen.generate()
   print(track.valid.numpy(), track.count.numpy())

.. snippet: after

   assert track.valid.numpy().any()

Keep seeds and the rest of the configuration fixed when comparing a control. Include
the default and inspect both raw centerline and finished road: resampling, relaxation
and fallbacks can hide the generator's original change. All guide figures record
seeds/configurations and sample selection in
:download:`the renderer manifest <../assets/racing/generators-manifest.json>`.

Choose how to place gates
-------------------------

Use native anchors when the corners, polar knots, selected sites or radial targets
are themselves the intended course. Five families have this route; ``repulsive``
does not. Native anchors bypass the road's spline/steering construction and have a
smaller control set.

Use ``presets.gates(generator="polar", source="stations")`` when gate positions
should be sampled along the generated source centerline. Use ``relaxed_stations``
when they should follow the relaxed track. Gate bar width, collision radius,
elevation and ordering remain separate choices:
:doc:`/tutorials/gate-sequences`.

Choose regeneration and curricula
---------------------------------

A dynamic field can vary per env without rebuilding; a static field fixes host work,
capacity or algorithm structure. :ref:`generators-per-env` lists the current track
fields, and :doc:`/user-guide/curricula` shows draws and masked updates. ``size``
changes XY extent while spacing and width stay absolute, so check
:ref:`tuning-size` before increasing size bounds.

For frequent resets, measure steady generation at the actual batch size. For
``repulsive``, consider slower regeneration or smaller per-step environment slices
and measure the whole batch call: masking retains unselected results but does not
make every generator stage's work proportional to the selected count. Construction
also warms a generation; first CUDA graph recording has its own cost.
:doc:`/generators/benchmarks` explains comparable measurements.

To mix families, allocate disjoint env ranges into a shared result as described in
:ref:`results-mixed-batches`, or choose a registered seeded family per worker/run.
``external`` ingests geometry rather than generating variation from the seed.

When a choice fails
-------------------

First read ``track.fail_mask`` and distinguish truncation from clearance or
nonfinite geometry. More capacity can fix truncation; changing a validity threshold
cannot repair a curve. The default ``on_invalid="poison"`` prevents accidental use
of failed rows. Use explicit ``keep`` only for diagnostic geometry and continue
masking on validity. See :doc:`/user-guide/results` and
:doc:`/relaxation/constraints`.
