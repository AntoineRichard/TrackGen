.. _courses:

Courses
=======

A :term:`course` (:class:`track_gen.course.Course`) bundles a generator with the
:doc:`runtime utilities </utilities/overview>` an RL loop needs (checkpoints and progress,
collision, a localizer, ...) and keeps them coherent: every regeneration refreshes every
utility, and the runtime state of the :term:`envs <env>` it rebuilt is reset. Build one
from a :term:`preset`, bind your sim's buffers once, then call ``step()`` every sim step:

.. code-block:: python

   import numpy as np
   import warp as wp

   from track_gen.course import Course
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"                     # or "cpu"
   course = Course.track(presets.track(num_envs=E, device=device),
                         checkpoint_spacing=0.6, collision="segments", seeds=42)
   # Ready: course.result holds the first batch, every utility is built.

   pos = wp.zeros(E, dtype=wp.vec3f, device=device)          # your sim's stable buffers
   quat = wp.full(E, wp.quat_identity(), dtype=wp.quatf, device=device)
   he = wp.full(E, wp.vec2f(0.02, 0.01), dtype=wp.vec2f, device=device)
   course.bind(position=pos, orientation=quat, half_extents=he)

   res = course.step()                          # events + contacts + valid, no arguments
   valid = res.valid.numpy() != 0               # False: this env's course is invalid
   oob = res.contacts.oob.numpy() != 0
   passed = res.events.passed.numpy()
   reward = np.where(valid, 10.0 * passed - 5.0 * oob, 0.0)   # a select, never "* valid"

   course.regenerate(advance=True)              # new courses for every env

.. snippet: after

   assert course.checkpoint_sampler.max_checkpoints == 58
   assert np.isfinite(reward).all()

The course is ready at construction: there is no first ``generate()`` to call. Every
reward and observation is gated on ``res.valid``: an :term:`invalid env` (its course
failed a check) reads ``oob = 1`` and frozen progress at every step, and needs a new
course, not a respawn. :doc:`/user-guide/rl-auto-reset` has the full loop, per-env
auto-reset included.

.. contents:: On this page
   :local:
   :depth: 1

Three ways to build a course
----------------------------

.. snippet: group=ways requires=utilities/course#main

.. code-block:: python

   from track_gen import TrackGenerator, tools

   # 1. A preset: a config or a spec, plus the components of a track (or gate) course
   course = Course.track(presets.track(num_envs=E, device=device),
                         checkpoint_spacing=0.6, collision="segments", seeds=42)
   course = Course.gates(presets.gates(num_envs=E, device=device, gate_width=0.1),
                         collision=0.02, seeds=42)            # gate posts of radius 0.02

   # 2. Any generator and any components, your own included
   gen = TrackGenerator(presets.track(num_envs=E, device=device), seeds=42)
   course = Course(gen, [tools.Checkpoints(0.6), tools.Progress(),
                         tools.BandCollision("segments"), tools.Localizer()])

- ``Course.track`` / ``Course.gates`` take a ``TrackGenConfig`` / ``GateGenConfig``, a
  :term:`PipelineSpec` (``presets.track(...)``, ``presets.gates(...)``, or one you edited)
  or a built generator, and build the components a track or gate course needs.
  ``extra=[...]`` appends components of your own.
- ``Course(generator, components)`` takes a built generator and an ordered list of
  :term:`components <component>` (:mod:`track_gen.tools`, or your own).
- ``Course(CourseConfig(...))`` builds the course a flat ``CourseConfig`` describes (`From
  a CourseConfig`_).

:doc:`/utilities/composing-a-course` covers the presets' arguments, the components and
their order, ``bind`` routing, and the results of your own components.

Lifecycle
---------

**Construct.** Construction does all the cold work, outside any :term:`capture`: it
builds the generator, whose own construction runs one generation (so ``course.result``
holds the first batch), attaches every component (first checkpoint sample, bakes and
course-line refresh included), then runs one regeneration as a warm-up. Expect the cost of
about two generations, which matters for ``repulsive``. Construct the course before you
begin a capture: inside one, construction raises ``CaptureError`` before allocating
anything. ``prepare=False`` (on ``Course(...)`` and the presets) skips the warm-up;
``course.prepare()`` runs it later, and so does the first regeneration call outside a
capture (:ref:`composing-a-course`).

**Bind.** ``course.bind(position=..., orientation=..., half_extents=...)`` binds your sim's
stable buffers once; ``step()`` then takes no arguments and reads them in place. Each
array goes to every component that declares an input of that name
(:ref:`composing-a-course-binding-inputs`).

**Step and reset.** Every sim step, after your physics wrote the bound buffers:

- ``step(stream=None)`` returns the course's :class:`~track_gen.course.StepResult`, the
  same instance every call, overwritten in place (`Reading a step`_).
- ``reset(mask=None, stream=None)`` respawns the masked envs on the SAME course (their
  progress and the localizer's memory); ``mask=None`` resets every env.

**Regenerate.** New courses, and a coherent refresh of every component:

- ``regenerate(*, mask=None, advance=False, stream=None, sync=True)`` reruns the generator
  from the current :term:`seeds <seed>` and :term:`epochs <epoch>`, refreshes every
  component and resets the runtime state, then returns ``course.result``. Without
  ``advance`` it rebuilds the same courses; ``advance=True`` first advances every env's
  epoch, so the call builds new courses without a reseed. ``mask=`` limits the new
  courses (and the resets) to the masked envs (:ref:`rl-masked-regeneration`).
- ``generate(seeds=None, *, stream=None, sync=True)`` reseeds, then regenerates. Without
  ``seeds`` it is ``regenerate()``. ``seeds=`` takes an ``int`` (env ``e`` gets
  ``seeds + e``) or an ``[E]`` int32 ``wp.array``; it zeroes every epoch, so
  ``generate(seeds=s)`` reproduces the courses of a fresh course built with ``seeds=s``.
- ``enqueue_generate(*, stream=None, mask=None, advance=False)`` runs the same body without
  ever building the course's own CUDA graph and without blocking: recorded under a
  capture, eager outside one.

``regenerate`` and ``generate`` wait for their stream by default; ``sync=False`` returns at
once, the results being ordered on the stream like every other library call.

**Other members.**

- ``inputs`` is ``{"regen_mask": ..., "reset_mask": ...}``: two stable, zero-initialised
  ``[E]`` int32 arrays for your kernels to write the masks of ``regenerate(mask=...)`` and
  ``reset(mask)`` into.
- ``seeds`` is the generator's :class:`~track_gen.seeds.SeedState` (each env's seed and
  epoch); ``result.generation`` (``[E]`` int32, the :term:`generation`) is the epoch each
  env's current course was built with. ``rng`` is the ``PerEnvSeededRNG`` you passed as ``seeds=`` (a
  ``CourseConfig`` course builds its own), else ``None``.
- ``generator`` and ``result`` are the generator and its result (a ``Track`` or a
  ``GateSequence``); ``components`` maps each component's name to it, in execution order.
- The utilities stay reachable as ``course.progress``, ``course.collision``,
  ``course.checkpoints`` (the ``CheckpointSet``), ``course.checkpoint_sampler``,
  ``course.localizer``, ``course.course_line`` and ``course.heightfield``: each is
  ``None`` when the course has no component of that role.

**Determinism.** Each env's course depends only on its ``(seed, epoch)`` pair. With
unchanged seeds and epochs, ``regenerate()`` rebuilds the identical batch, byte for byte on
one device, for every generator, ``repulsive`` included; across devices the output differs
only statistically (fp32 rounding).

**One course per generator.** A course owns its generator: regenerate through the course,
never through ``course.generator`` (that would skip the refresh). Building a second course
over the same generator object takes it over, and the first one then raises
``RuntimeError`` at its next call. A ``CourseConfig`` course and a preset given a config
or a spec build their own generator.

Reading a step
--------------

``StepResult`` has four fields: ``events`` (the progress tracker's
:class:`~track_gen.progress.ProgressEvents`), ``contacts`` (the collision result, or
``None`` without collision), ``frame`` (the localizer's
:class:`~track_gen.localize.TrackFrame`: always there in gates mode, ``None`` on a track
course without a localizer) and ``valid``, the generator result's ``[E]`` int32
:term:`valid` (the same array as ``course.result.valid``: never write into it). It also
reads like a mapping from component name to that component's per-step result:

.. code-block:: python

   res = course.step()
   res["progress"] is res.events, res["collision"] is res.contacts
   res.get("localizer")                # the localizer's TrackFrame (None without one)
   list(res.keys())                    # the per-step components, in execution order

- ``valid`` is live: a regeneration rewrites it for the envs it rebuilds, so read it
  before regenerating, or ``clone()`` the result.
- ``res.clone()`` copies every entry and ``valid``; it allocates, so call it outside your
  graph.
- :ref:`composing-a-course` lists every mapping rule, and :doc:`/user-guide/results`
  describes the result types the course reads.

``step()`` and ``reset()`` never block the host, so they record into your own CUDA graph
as-is (:doc:`/tutorials/cuda-graph-in-a-sim`). Once ``step()`` is recorded, keep writing
into the SAME bound buffers: the graph reads the arrays that were bound when it was
recorded, and ``course.bind(...)`` with other arrays then emits
:class:`~track_gen.exec.RebindAfterCaptureWarning` (record the graph again).

Gates mode
----------

``Course.gates`` builds a course over a gate generator: checkpoints are the gates
themselves, a centerline through the gates (``course.course_line.centerline``, refreshed
on every regeneration) carries the localizer, and ``collision=`` is a post radius
(:class:`~track_gen.collision.DiscChecker` posts), a ``tools.FrameCollision`` (the gate
frames) or ``None``:

.. code-block:: python

   gates = Course.gates(presets.gates(num_envs=E, device=device, gate_width=0.1),
                        collision=0.02, seeds=7)
   gates.bind(position=pos, orientation=quat, half_extents=he)
   res = gates.step()
   hit = res.contacts.hit.numpy()      # a DiscContact here, not a BoxContact
   s = res.frame.s.numpy()             # arc length along the gates' centerline

- ``res.contacts`` is a :class:`~track_gen.collision.DiscContact` with posts (``hit``,
  ``disc``, ``depth``, ``nearest``), a frame contact with ``tools.FrameCollision``.
  The posts are rebuilt on device at every regeneration; each env's scan covers its
  ``2 * count[e]`` real posts, and an invalid env hits nothing.
- ``res.frame`` locates the agent on the gates' centerline: ``s`` (arc length), ``n``
  (signed offset to the right), ``n_up`` (signed vertical offset), ``segment``.
- **The gate width must stay above zero.** A width-0 gate has a degenerate crossing
  segment and can never be passed: ``Course.gates`` refuses a construction
  ``gate_width`` of ``0`` (``presets.gates`` defaults to ``0``: pass one), and refuses a
  later scalar ``course.generator.params["gate_frames"].set("gate_width", v)`` with
  ``v <= 0``. A per-env ``[E]`` array is not range-checked (that would read it back on
  the host): an entry ``<= 0`` is used as ``0`` and that env's gates can never be passed.
- The bar width also decides how many envs are valid: see
  :ref:`gate-width-invalid-fraction`, and :ref:`gates-3d-runtime` for 3D gate courses.

From a ``CourseConfig``
-----------------------

``Course(CourseConfig(...))`` (also ``Course(config=cfg)`` and
``Course.from_config(cfg)``) builds the course a flat config describes: ``mode="track"``
or ``"gates"``, a ``TrackGenConfig`` / ``GateGenConfig`` as ``gen``, and the utilities'
settings as flat fields. It builds the same components as the presets, with its own
``PerEnvSeededRNG`` as ``course.rng``, and stays supported:

.. snippet: group=config requires=utilities/course#main

.. code-block:: python

   from track_gen import TrackGenConfig
   from track_gen.course import CourseConfig

   course = Course(CourseConfig(mode="track", gen=TrackGenConfig(num_envs=E, device=device),
                                seeds=42, collision="segments", checkpoint_spacing=0.6))

Its SDF and height-field components declare no largest course size, so they never warn
about their resolution (:ref:`collision-resolution-check`). Prefer the explicit presets when declaring
maximum course sizes and component capacities.

.. _course-checkpoint-capacity:

Checkpoint capacity
-------------------

A track course samples a ring of checkpoints along each centerline. ``max_checkpoints`` (a
preset argument, ``tools.Checkpoints(..., max_checkpoints=)``, or
``CourseConfig.max_checkpoints``) is the number of slots per env: ``course.checkpoints``
arrays are ``[E * max_checkpoints]``, and a longer ring is clipped
(``course.checkpoint_sampler.truncated[e] == 1``). By default the course sizes it from the
longest track the generator can produce, read from the generator with no readback:

.. code-block:: text

   max_checkpoints = max(3, ceil(max_loop_length / checkpoint_spacing))
   max_loop_length = 1.5 * N_max * spacing + dZ

``max_loop_length`` bounds the 3D length of the centerline of every env that is not
truncated: ``N_max * spacing`` bounds its plan view, the ``1.5`` is measured headroom for
the relaxation's growth and a boundary's extra ``2 pi * half_width``, and ``dZ`` (``0`` at
flat elevation) bounds the climbs of the elevation from the declared ranges of its knobs.
It does not depend on the seeds, the batch or the course :term:`size`. At the defaults
(:term:`N_max` ``= 384``, :term:`spacing` ``= 0.06``, flat: ``max_loop_length = 34.56``):

.. list-table::
   :header-rows: 1

   * - ``checkpoint_spacing``
     - default ``max_checkpoints``
   * - 0.3
     - 116
   * - 0.5
     - 70
   * - 0.6
     - 58
   * - 1.0
     - 35

- **Memory.** Each slot holds ``position``, ``left``, ``right``, ``tangent`` and
  ``up_half`` (52 bytes per slot and env, plus 8 bytes per env for the set's ``valid`` /
  ``generation``): 28.5 MiB at ``E = 8192`` and 70 slots. The bound grows with
  ``N_max * spacing``: ``N_max = 2048``, ``spacing = 0.02`` and ``checkpoint_spacing =
  0.05`` give 1229 slots, 262 MB at ``E = 4096``; pass a smaller ``max_checkpoints``
  there.
- **Rows do not depend on the capacity.** An env's real rows, its count and its progress
  are the same at any capacity its ring fits.
- **When the generator declares no bound**, the course reads the construction batch back
  once and sizes the ring at ``ceil(1.5 * longest valid centerline / checkpoint_spacing)``:
  an elevation whose height knobs declare no range and no ``z_valid_grade`` (every
  elevated ``TrackGenConfig``), a generator that provides its own altitudes without a
  ``z_valid_grade`` or its own half-widths, a width stage other than ``width.constant``,
  a plugin geometry stage after the constant-spacing resample. A ``z_valid_grade = g > 0``
  declares one (``dZ = g * 1.5 * N_max * spacing``: 277 slots at ``g = 3``,
  ``checkpoint_spacing = 0.5``). A readback capacity is sized at the construction
  course size: check ``course.checkpoint_sampler.truncated`` after regenerating onto much
  longer tracks, or pass ``max_checkpoints``.
- **A warning.** When the default capacity clips a valid env of the construction batch,
  construction emits one ``UserWarning``. An explicit ``max_checkpoints`` never warns.
- ``course.checkpoint_sampler.max_checkpoints`` reads the capacity in use. Pass
  ``max_checkpoints`` explicitly to preserve array shapes across a change in the
  seed set, generator geometry or curriculum range.

A size curriculum grows the tracks after construction: :ref:`curricula-course` shows how
to size the checkpoint ring and the SDF and height-field grids for it.

One body, one graph
-------------------

``regenerate()`` runs ONE body: the generator, then every component's per-generation
refresh in execution order (checkpoint resample, SDF and height-field bakes, course-line
refresh, gate-post rebuild), then every component's reset in reverse order (progress and
localizer: every env, or the masked envs with ``mask=``). Where it runs depends on the
stream the call resolves:

- **Inside your capture** (``wp.ScopedCapture``, ``wp.capture_begin``,
  ``torch.cuda.graph``) the body is recorded into YOUR graph, so new courses and the
  per-step ``step()`` / ``reset()`` calls can share one graph. Nothing is synchronized.
- **Outside a capture on** ``cuda`` the body is recorded once into the course's own CUDA
  graph (``course.graph``, ``None`` until that first call) and replayed on every later
  call. Each ``(mask array, advance)`` combination gets a graph of its own, recorded on
  its first use; ``course.graph`` is the one of a plain ``regenerate()``.
- **On** ``cpu``, or with a generator that has a step running on the host (a plugin
  registered with ``capturable=False``), it runs eagerly. Recording such a generator into
  a capture raises ``NotCapturableError`` before launching anything.

.. snippet: setup

   @wp.kernel
   def _agent_k(center: wp.array(dtype=wp.vec3f), stride: int, k: int,
                pos: wp.array(dtype=wp.vec3f)):
       e = wp.tid()
       pos[e] = center[e * stride + k]

   def sim_step(pos, quat, k=0):
       r = course.result
       wp.launch(_agent_k, E, [r.center, r.layout.stride, k, pos], device=device)

.. code-block:: python

   K = 32                                       # sim steps per recorded episode
   crashed = course.inputs["reset_mask"]        # your kernel writes it: 1 = respawn
   with wp.ScopedCapture(device=device) as cap:
       course.regenerate(advance=True)          # new courses for the whole batch
       for k in range(K):
           sim_step(pos, quat, k)               # your physics: writes the bound buffers
           res = course.step()
           course.reset(crashed)                # valid envs that crashed: same course
   episode = cap.graph

   for _ in range(3):
       wp.capture_launch(episode)               # one episode per replay, new courses each

.. snippet: after

   assert int(course.result.generation.numpy()[0]) >= 3

Every replay reads the live seeds and epochs: a recorded ``regenerate(advance=True)``
advances every env's epoch on device, so each replay builds new courses with no host work.
To choose the seeds instead, record ``course.generate(seeds=seed_buf)`` with a stable
``[E]`` int32 ``seed_buf`` that you rewrite in place between replays. An ``int`` passed
to ``generate(seeds=...)`` inside a capture is baked into the graph: every replay reseeds
with that same value and rebuilds the same courses. :doc:`/tutorials/cuda-graph-in-a-sim`
has the full pattern, with the ``torch.cuda.graph`` and side-stream variants, and
:ref:`capture-rules` the rules for recording; a recorded graph keeps the course alive
(:ref:`capture-lifetime`).

Per-env auto-reset and invalid envs
-----------------------------------

The generator is fixed-batch, so every regeneration recomputes the whole batch. Per-env
control comes from the two masks:

- ``reset(mask)`` respawns the masked envs on the SAME course;
- ``regenerate(mask=m, advance=True)`` gives the masked envs NEW courses and resets them,
  leaving every other env bitwise unchanged; recorded next to ``step()``, it is a per-env
  auto-reset inside your step graph.

An invalid env (``res.valid[e] == 0``) reads ``oob = 1``, frozen progress with
``dist_to_next = 0`` and NaN geometry at every step, and a respawn keeps it so: mark it
done and regenerate it. :doc:`/user-guide/rl-auto-reset` has the recipe on the host and
inside a graph (:ref:`rl-valid-recipe`, :ref:`rl-in-graph`), what a masked regeneration
costs (:ref:`rl-cost`), and :ref:`utilities-invalid-envs` what every utility writes for
an invalid env.

API: :class:`track_gen.course.Course`, :class:`track_gen.course.StepResult`,
:class:`track_gen.course.CourseConfig` (:ref:`reference-runtime`).
