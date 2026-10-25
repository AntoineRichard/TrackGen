.. _track-relaxation:

Make room for the road
======================

Track relaxation moves sampled centerline beads apart at close approaches and rounds
corners that are tight relative to the road half-width. It runs after constant-spacing
resampling and width assignment, before final uniform resampling and border offsets.

.. raw:: html

   <video controls loop muted playsinline width="100%" poster="../_static/relaxation-iterations-poster.png" aria-label="Actual track relaxation prefixes, with iteration, validity and failure labels">
     <source src="../_static/relaxation-iterations.mp4" type="video/mp4">
     <p>Track relaxation opens close approaches and rounds tight bends. See the text below for phases and limits.</p>
   </video>

Three selected fixed-seed examples, shown after each solve prefix has passed through
uniform resampling and inflation. The raw outputs fail the sampled thickness check.
The main solve runs 50 sweeps, followed by five Taubin passes, then ten spacing-polish
sweeps. The camera stays fixed and each panel reports its actual validity and minimum
nonlocal bead distance. These examples demonstrate motion; they do not estimate yield.
The :doc:`convergence` page explains the frame construction and limitations.

Try a different budget
-----------------------

.. code-block:: python

   from track_gen import TrackGenConfig, TrackGenerator

   config = TrackGenConfig(num_envs=4, device="cuda:0", relax_iters=75)
   gen = TrackGenerator(config, seeds=42)
   track = gen.generate()
   accepted = track.valid.numpy() != 0
   failures = track.fail_mask.numpy()

.. snippet: after

   assert accepted.shape == (4,)
   assert failures.shape == (4,)

Use the fail bits to choose the next change. More sweeps may help unresolved sampled
thickness; a truncation failure needs more capacity or larger spacing. A rejected
imported centerline may need corrected input. Relaxation is not the only stage affecting
acceptance (:ref:`geometry-validity`).

The default ``relax.xpbd`` stage applies 50 Chebyshev-accelerated Jacobi sweeps and a
smoothing tail. It is a position-projection solver: it does not retain XPBD constraint
multipliers or solve a physical vehicle model. It can reshape the course and does not
guarantee that every input becomes valid. In a preset, ``relax=None`` selects
``relax.none``; the flat config equivalent is ``relax_enable=False``. External tracks
default to no relaxation so their supplied shape is preserved apart from resampling.

.. figure:: ../assets/racing/geometry-validity.svg
   :alt: Actual validity results on selected synthetic centerlines show close
         approaches, tight corners, crossings, overlapping bands and truncation.

   The sampled thickness check is sensitive to local curvature and nonlocal bead
   distances. Other failure bits describe different problems. Retained failed geometry
   here uses ``on_invalid="keep"`` explicitly.

Choose the detail you need
--------------------------

- :doc:`constraints`: bead neighbourhood, separation margin, spacing and bending.
- :doc:`solver`: Jacobi execution, candidate caches, acceleration and smoothing.
- :doc:`convergence`: how to read actual iteration states and tune a fixed budget.
- :doc:`gates`: the separate gate-sphere relaxation, which uses sequential pair updates.

.. toctree::
   :maxdepth: 1
   :hidden:

   constraints
   solver
   convergence
