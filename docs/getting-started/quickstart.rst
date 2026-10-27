.. _quickstart:

Create a race track
===================

Generate a batch, select a valid environment, and read its centerline and road borders.
Install TrackGen first (:doc:`installation`). Use ``"cpu"`` to explore without a GPU;
use ``"cuda:0"`` in your simulator.

.. code-block:: python

   import numpy as np
   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   gen = TrackGenerator(
       presets.track(num_envs=8, device="cuda:0", generator="bezier",
                     half_width=0.03, n_max=512),
       seeds=42,
   )
   track = gen.generate()
   valid = track.valid.numpy() != 0
   count = track.count.numpy()
   center = track.center.numpy().reshape(8, gen.n_max, 3)
   outer = track.outer.numpy().reshape(8, gen.n_max, 3)
   inner = track.inner.numpy().reshape(8, gen.n_max, 3)

   for e in np.flatnonzero(valid):
       xy = center[e, :count[e], :2]  # real centerline points, in course order
       outer_points = outer[e, :count[e]]  # corresponding road borders
       inner_points = inner[e, :count[e]]
       print(e, len(xy), float(track.length.numpy()[e]))

.. snippet: after

   assert valid.any()
   assert all(np.isfinite(center[e, :count[e]]).all() for e in np.flatnonzero(valid))

.. figure:: ../assets/racing/generator-bezier-examples.png
   :alt: Representative Bezier centerlines and their finished road bands.

   Bezier examples: the centerline describes the route; inflation adds the road band.
   The :doc:`generator guide </generators/bezier>` shows how its controls change both.

The few things to know first
----------------------------

- ``num_envs`` fixes the batch size. Every environment has its own seed and epoch.
- ``half_width`` is half the road width, in the same coordinate units as the track.
  :ref:`course-controls` shows width, size and spacing side by side.
- ``n_max`` is storage capacity per environment. Thin or large tracks need more points;
  insufficient capacity invalidates the result. It does not simplify the loop to fit.
- Read ``valid`` and ``count`` before geometry. Failed environments are poisoned by
  default. :ref:`invalid-envs` explains the failure bits and diagnostic ``keep`` mode.
- Results and their buffers are reused. ``track.clone()`` keeps a snapshot;
  :ref:`results` explains the layout and ownership.

Change the track
----------------

Choose any of the :ref:`six procedural generators <generators-overview>` with
``generator=``. Repeating ``generate()`` with unchanged seeds and epochs repeats the
geometry. Advance the epoch for new tracks:

.. code-block:: python

   snapshot = track.clone()
   track = gen.generate(advance=True)

For selected environments only, write an ``int32`` mask into
``gen.inputs["regen_mask"]`` and pass it as ``mask=``. The unselected rows retain their
tracks. :ref:`generating-seeds` covers seeds, epochs and masks; :ref:`curricula`
shows how to change dynamic settings without rebuilding.

Use it for racing
-----------------

A :ref:`Course <courses>` combines the generator with checkpoint progress, collision and
reset/regeneration. Start there when connecting your simulator, then follow
:ref:`cuda-graph-sim-loop` for capture and :ref:`rl-auto-reset` for device-side resets.

For drone racing, continue with :ref:`quickstart-gates`. For a dataset or a circuit you
already have, use :ref:`importing-a-track`.
