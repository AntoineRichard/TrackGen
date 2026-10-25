.. _quickstart:

Quickstart
==========

.. admonition:: Coming from 0.1?

   Read :ref:`upgrading-0.2` first: most 0.1 code keeps running, but some results
   change without an error.

This page builds a batch of tracks, a batch of gate sequences, and a :term:`course` an
RL loop can drive. Every example runs on the ``cpu`` device as well as on ``cuda``.
Install TrackGen first (:doc:`installation`); :doc:`concepts` explains the terms in one
page.

A batch of tracks
-----------------

A generator builds one track per :term:`env`, for a fixed batch of :term:`E` envs.
Describe the pipeline with a :term:`preset`, then build the generator:

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"
   spec = presets.track(num_envs=E, device=device, half_width=0.03, n_max=512)
   gen = TrackGenerator(spec, seeds=0)   # builds the pipeline and runs one generation

   track = gen.generate()                # the batch: a Track
   valid = track.valid.numpy()           # [E] int32: 1 when the env passed every check
   count = track.count.numpy()           # [E] int32: real points per env
   center = track.center.numpy().reshape(E, -1, 3)   # [E, N_max, 3] centerline points
   xy = center[0, : count[0], :2]        # env 0's centerline, evenly spaced points

- :func:`presets.track <track_gen.pipeline.presets.track>` returns a :term:`spec`: the
  list of :term:`stages <stage>` that turn a :term:`seed` into a road (generator,
  resampling, relaxation, the road band, the checks). Its keywords pick the generator and
  set its knobs; the defaults give the ``bezier`` generator with a 0.1 half-width.
- :class:`~track_gen.TrackGenerator` plans the pipeline, allocates every buffer once, and
  runs a first :term:`generation`. Build it outside any CUDA graph :term:`capture`.
- ``generate()`` returns the result, a ``Track``. It is the same object on every call,
  with its buffers overwritten in place: use ``track.clone()`` to keep a snapshot. Called
  again with the same seeds and epochs, it rebuilds the same batch.
- Each per-point field (``center``, ``outer``, ``inner``, ``tangent``, ``normal``) is a
  flat ``[E * N_max]`` array of 3D points: reshape it per env. Env ``e`` has
  ``count[e]`` real points; the rows after them are padding, with NaN in x and y.
- :term:`N_max` (``n_max=``) is the per-env point capacity. A thin road
  (``half_width=0.03``) gets a finer :term:`spacing`, so more points per track: the
  default 384 holds most of them but not all, and a track that does not fit is cut short
  and marked invalid. Hence 512 here (the sizing rule is in :ref:`tuning-size`).
- ``seeds=0`` gives env ``e`` the seed ``e``; ``seeds=k`` gives it ``k + e``.

New tracks come from advancing each env's :term:`epoch`, for the whole batch or for
selected envs only:

.. code-block:: python

   track = gen.generate(advance=True)    # every env gets a new track

   regen = gen.inputs["regen_mask"]      # [E] int32 on the device, stable address
   regen.assign((track.valid.numpy() == 0).astype("int32"))   # here: the invalid envs
   track = gen.generate(advance=True, mask=regen)   # the other envs keep their tracks

Always check how many envs are :term:`valid`: an :term:`invalid env` failed a check (a
self-crossing, a road band that overlaps itself, a track longer than ``N_max`` holds), and
the runtime utilities treat it as an env with no track.

.. code-block:: python

   invalid = (track.valid.numpy() == 0).mean()
   print(f"{invalid:.1%} of the batch is invalid")

:ref:`invalid-envs` lists the checks, their typical causes and the knob that lowers each.

A batch of gate sequences
-------------------------

For drone-style courses, a ``GateGenerator`` places gates (a center, a direction, two bar
ends) instead of building a road:

.. code-block:: python

   from track_gen import GateGenerator

   gate_gen = GateGenerator(presets.gates(num_envs=E, device=device, gate_width=0.1),
                            seeds=0)
   gates = gate_gen.generate()           # a GateSequence
   G = gate_gen.max_gates                # gate slots per env (32 by default)
   position = gates.position.numpy().reshape(E, G, 3)
   first = position[0, : gates.count.numpy()[0]]    # env 0's real gates, in order

.. list-table:: ``GateSequence`` fields
   :header-rows: 1
   :widths: 28 30 42

   * - Field
     - Shape and type
     - Meaning
   * - ``position``
     - ``[E * G]`` vec3f
     - gate centers, in passing order
   * - ``tangent``, ``forward``
     - ``[E * G]`` vec3f
     - the course direction at each gate; the gate plane's normal (the direction to
       pass it)
   * - ``left``, ``right``
     - ``[E * G]`` vec3f
     - the two bar ends, ``half_size`` from the center
   * - ``orientation``
     - ``[E * G]`` quatf
     - the gate pose (x forward, y left, z up)
   * - ``half_size``
     - ``[E * G]`` float32
     - half the bar width (``0.5 * gate_width``)
   * - ``count``
     - ``[E]`` int32
     - real gates per env; slots ``i >= count[e]`` are NaN
   * - ``valid``
     - ``[E]`` int32
     - 1 when the env passed every check

``gate_width`` is the full bar width; the default ``0`` makes point gates. Wider bars
cross each other more often, and an env whose bars cross is invalid: a small fraction
of a ``bezier`` batch at ``gate_width=0.1``, most of it at 0.4.
:ref:`gate-width-invalid-fraction` gives the measured fractions per generator and
``scale``.

A course for RL
---------------

A course adds the runtime side to a generator: checkpoints and progress, collision
of your agents with the road, per-env regeneration. ``Course.track`` builds the usual
:term:`components <component>` over a spec, and the course is ready as soon as it is
built:

.. code-block:: python

   import numpy as np
   import warp as wp
   from track_gen.course import Course

   course = Course.track(spec, checkpoint_spacing=0.6, collision="segments", seeds=42)

   # Stand-ins for your simulator: place valid envs on their start lines only.
   initial_valid = course.result.valid.numpy() != 0
   start = np.zeros((E, 3), dtype=np.float32)
   center = course.result.center.numpy().reshape(E, -1, 3)
   start[initial_valid] = center[initial_valid, 0]
   pos = wp.array(start, dtype=wp.vec3f, device=device)
   quat = wp.full(E, wp.quatf(0.0, 0.0, 0.0, 1.0), dtype=wp.quatf, device=device)
   half_extents = wp.full(E, wp.vec2f(0.02, 0.01), dtype=wp.vec2f, device=device)
   course.bind(position=pos, orientation=quat, half_extents=half_extents)

   regen = course.inputs["regen_mask"]
   for _ in range(10):
       # ... your simulator steps here and writes pos / quat in place ...
       res = course.step()                       # progress events and contacts
       valid = res.valid.numpy()                 # read it first: regenerate() updates it
       oob = res.contacts.oob.numpy()            # 1: the agent's box is off the road
       reward = np.where(valid == 1, 1.0 - oob, 0.0)    # gate every reward on valid
       done = (oob == 1) | (valid == 0)          # every invalid env is done too
       if done.any():                            # a host read: skip when nothing ended
           regen.assign(done.astype(np.int32))
           course.regenerate(mask=regen, advance=True)   # new courses for those envs only

- ``bind`` takes stable buffers once: write new poses into the same arrays every step,
  and ``step()`` reads them (also when recorded into a CUDA graph).
- ``res.events`` holds the progress events (checkpoints passed, laps, the distance to the
  next checkpoint) and ``res.contacts`` the collision answer, each as per-env device
  arrays.
- ``course.reset(mask)`` respawns envs on the course they already have;
  ``course.regenerate(mask=..., advance=True)`` gives them a new one. An invalid env needs
  a new one.
- On the ``cpu`` device (or with a generator that has a step that runs on the host),
  ``regenerate(mask=...)`` runs a full generation even when the mask is all zeros, hence
  the ``if``. On ``cuda`` it replays the course's own CUDA graph, which skips an all-zero
  mask on device when :term:`conditional nodes <conditional node>` are available (CUDA
  12.4 or newer), as does recording it into your own Warp capture (:ref:`rl-cost`).

:ref:`rl-auto-reset` covers the masks, invalid envs in a training loop and the in-graph
version of this loop; :ref:`courses` covers everything a course holds.

Coming from a config?
---------------------

A flat ``TrackGenConfig`` still works and builds the same pipeline as the preset:

.. code-block:: python

   from track_gen import TrackGenConfig

   config = TrackGenConfig(num_envs=E, half_width=0.03, N_max=512, device=device)
   gen_from_config = TrackGenerator(config, seeds=0)

.. snippet: after

   assert np.array_equal(gen_from_config.generate().center.numpy(),
                         TrackGenerator(spec, seeds=0).generate().center.numpy(),
                         equal_nan=True)

Presets are the recommended way: they name each stage's knobs and also accept plugin
stages. :ref:`generating` compares the ways to build a generator, and
:ref:`upgrading-flat-config` shows how to port a flat config.

Next steps
----------

Pick the path that matches what you are building:

- **Training an RL agent**: :ref:`courses`, then :ref:`cuda-graph-sim-loop`,
  :ref:`rl-auto-reset`, :ref:`collision-backends` and :ref:`curricula-course`.
- **Building datasets of tracks or gates**: :ref:`generating`, then :ref:`results`,
  :ref:`choosing-a-generator` and :ref:`curricula`.
- **Writing a plugin** (your own generator, stage or component):
  :ref:`writing-a-generator`, then :ref:`testing-a-plugin` and
  :ref:`packaging-a-plugin`.
- **Upgrading from 0.1**: :ref:`upgrading-0.2`.
