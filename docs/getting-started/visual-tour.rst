.. _visual-tour:

Watch how courses take shape
=============================

Play a short explanation, then open the matching guide to change the settings.
All players have controls and start paused. The still posters summarize the result
without playing a video; each guide also explains the process in text. Use a player's
fullscreen control to inspect the smaller diagnostic labels.

Relaxation: watch a problem settle
----------------------------------

These selected examples show actual output at the labeled solve budgets. The road
clip includes separate main-solve, smoothing and spacing-polish phases. Visual
settling alone does not establish validity: read the displayed checks too.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Road XPBD

      .. raw:: html

         <video controls preload="none" style="width:100%;max-width:100%" poster="../_static/relaxation-iterations-poster.png" aria-label="Road relaxation at labeled solve, smoothing and polish budgets.">
           <source src="../_static/relaxation-iterations.mp4" type="video/mp4">
           Road relaxation at labeled solve, smoothing and polish budgets.
         </video>

      :doc:`Compare iteration budgets and remaining failure cases. </relaxation/convergence>`

   .. grid-item-card:: Gate separation

      .. raw:: html

         <video controls preload="none" style="width:100%;max-width:100%" poster="../_static/relaxation-gates-iterations-poster.png" aria-label="Gate centers separating at increasing solve-round budgets.">
           <source src="../_static/relaxation-gates-iterations.mp4" type="video/mp4">
           Gate centers separating at increasing solve-round budgets.
         </video>

      :doc:`See why center separation and gate-bar crossing differ. </relaxation/gates>`

Generators: see the construction
--------------------------------

The construction drawings progressively expose recorded anchors, handles, sites
or path samples. They explain algorithm order, not elapsed runtime. Repulsive's
growth clip records actual solver iterations. A gallery sample is not a yield estimate.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Bézier

      .. raw:: html

         <video controls preload="none" style="width:100%;max-width:100%" poster="../_images/generator-bezier-construction.png" aria-label="Grid addresses, actual offsets and handles build the loop; a crossing can select its entire polygon.">
           <source src="../generator-bezier-construction.mp4" type="video/mp4">
           Grid addresses, actual offsets and handles build the loop; a crossing can select its entire polygon.
         </video>

      :doc:`Grid addresses, actual offsets and handles build the loop; a crossing can select its entire polygon. </generators/bezier>`

   .. grid-item-card:: Hull

      .. raw:: html

         <video controls preload="none" style="width:100%;max-width:100%" poster="../_images/generator-hull-construction.png" aria-label="Grid selection, midpoint lobes and whole-loop fallback shape the route.">
           <source src="../generator-hull-construction.mp4" type="video/mp4">
           Grid selection, midpoint lobes and whole-loop fallback shape the route.
         </video>

      :doc:`Grid selection, midpoint lobes and whole-loop fallback shape the route. </generators/hull>`

   .. grid-item-card:: Polar

      .. raw:: html

         <video controls preload="none" style="width:100%;max-width:100%" poster="../generator-polar-construction-poster.png" aria-label="Regular slots, random radii and independent angular jitter define the route.">
           <source src="../generator-polar-construction.mp4" type="video/mp4">
           Regular slots, random radii and independent angular jitter define the route.
         </video>

      :doc:`Regular slots, random radii and independent angular jitter define the route. </generators/polar>`

   .. grid-item-card:: Voronoi

      .. raw:: html

         <video controls preload="none" style="width:100%;max-width:100%" poster="../_images/generator-voronoi-construction.png" aria-label="Select sites, order the cycle and round its corners.">
           <source src="../generator-voronoi-construction.mp4" type="video/mp4">
           Select sites, order the cycle and round its corners.
         </video>

      :doc:`Select sites, order the cycle and round its corners. </generators/voronoi>`

   .. grid-item-card:: Checkpoint

      .. raw:: html

         <video controls preload="none" style="width:100%;max-width:100%" poster="../_images/generator-checkpoint-construction.png" aria-label="Steer a bounded-turn path and select a closed candidate.">
           <source src="../generator-checkpoint-construction.mp4" type="video/mp4">
           Steer a bounded-turn path and select a closed candidate.
         </video>

      :doc:`Steer a bounded-turn path and select a closed candidate. </generators/checkpoint>`

   .. grid-item-card:: Repulsive

      .. raw:: html

         <video controls preload="none" style="width:100%;max-width:100%" poster="../_images/generator-repulsive-growth.png" aria-label="Watch a growing ring separate and bend.">
           <source src="../generator-repulsive-growth.mp4" type="video/mp4">
           Watch a growing ring separate and bend.
         </video>

      :doc:`Watch a growing ring separate and bend. </generators/repulsive>`

Raw samples become road beads
-----------------------------

All track generators share this ending. ``num_points`` fixes the raw count;
``spacing`` sets an arc-distance target before relaxation. XPBD and final
same-count redistribution explain why final chord lengths need not equal it.

.. raw:: html

   <video controls preload="none" style="width:100%;max-width:100%;aspect-ratio:5/3;object-fit:contain" poster="../_images/pipeline-sampling.png" aria-label="Raw fixed-count samples become target-spaced beads, move in XPBD and are redistributed uniformly.">
     <source src="../pipeline-sampling.mp4" type="video/mp4">
     Raw fixed-count samples become target-spaced beads, move in XPBD and are redistributed uniformly.
   </video>

:ref:`See the four stages and measured distances <sampling-stages>`.

For side-by-side control effects, continue with :ref:`course-controls` and the
:ref:`generator gallery <generators-overview>`. For failure geometry, see
:ref:`geometry-validity` and :ref:`runtime collision <collision-backends>`.
