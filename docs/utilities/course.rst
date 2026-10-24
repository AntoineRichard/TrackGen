The Course facade
=================

The :doc:`runtime utilities </utilities/overview>` can be wired by hand — or bundled by
``track_gen.course.Course``, which owns the orchestration invariants
(rebake/resample/posts rebuild and a progress reset on every regeneration, full or
limited to the envs a mask selects; per-env respawns via a mask). A ``CourseConfig``
builds the 0.x course, whose loop is unchanged since 0.2.0.dev5:

.. code-block:: python

   import torch
   import warp as wp

   from track_gen import TrackGenConfig
   from track_gen.course import Course, CourseConfig

   course = Course(CourseConfig(
       mode="track",
       gen=TrackGenConfig(num_envs=E, device="cuda"),
       seeds=42,
       collision="segments",          # or "sdf" / None
       checkpoint_spacing=0.6,
   ))                                 # first batch + sub-tools are ready here
   course.bind(position=robot_pos, orientation=robot_quat, half_extents=robot_he)
   done = course.inputs["regen_mask"]                # stable [E] int32 regeneration mask
   crash = wp.zeros(E, dtype=wp.int32, device="cuda")              # stable respawn mask

   for _ in range(steps):
       sim.step()                     # writes the bound buffers in place
       res = course.step()            # events + contacts + valid, no args
       valid = wp.to_torch(res.valid).bool()         # False: this env's course is invalid
       oob = wp.to_torch(res.contacts.oob).bool()
       passed = wp.to_torch(res.events.passed).float()
       dist = wp.to_torch(res.events.dist_to_next)
       reward = torch.where(valid, 10.0 * passed - 5.0 * oob.float(), 0.0)  # a select, never "* valid"
       obs_dist = torch.where(valid, dist, 0.0)      # no NaN reaches the policy
       crashed = oob & valid                         # a valid env that crashed: same course
       finished = (wp.to_torch(res.events.laps) >= 3) | ~valid       # done |= 1 - valid
       if crashed.any():                             # host reads: see below
           wp.to_torch(crash).copy_(crashed.int())
           course.reset(crash)                       # respawn on the SAME course
       if finished.any():
           wp.to_torch(done).copy_(finished.int())
           course.regenerate(mask=done, advance=True)   # NEW courses (and resets) where done
       respawn(crashed | finished)    # your code: restart those envs' agents
   course.generate(seeds=next_seed)   # new courses for everyone
   course.regenerate()                # or: rebuild from the rng's current seeds
   course.regenerate(advance=True)    # or: next epoch for everyone, same seeds

Since 0.2.0.dev6 a ``Course`` is a composition root: a generator plus an ordered list
of components (:mod:`track_gen.tools`). The same course comes from a preset, and any
generator takes any components, your own included (:doc:`/utilities/composing-a-course`):

.. code-block:: python

   from track_gen import TrackGenConfig, TrackGenerator, tools

   course = Course.track(TrackGenConfig(num_envs=E, device="cuda"),
                         checkpoint_spacing=0.6, collision="segments", seeds=42)
   course = Course(TrackGenerator(TrackGenConfig(num_envs=E, device="cuda"), seeds=42),
                   [tools.Checkpoints(0.6, max_checkpoints="auto"), tools.Progress(),
                    tools.BandCollision("segments"), tools.Localizer()])

``Course(cfg)`` (also ``Course(config=cfg)``) is ``Course.from_config(cfg)``: it builds
the components of the :ref:`migration table <course-config-migration>` (with
``size_hi=None``, so no resolution warning, and its own ``PerEnvSeededRNG``), with the
outputs and launches it had before roadmap PR15 (0.2.0.dev6 at PR14; relative to
0.2.0.dev5, see the CHANGELOG migration items 1-8: invalid envs, the default checkpoint
capacity). A preset's ``course.rng`` is ``None`` unless you pass
``seeds=PerEnvSeededRNG(...)`` (``course.seeds`` holds the seeds either way); the
:ref:`migration table <course-config-migration>` maps every ``CourseConfig`` field.

An env whose course failed a check (``res.valid[e] == 0``: 0.4 to 1.6 % of the envs of
the default ``hull``, ``checkpoint`` and ``repulsive`` generators) reads ``oob = 1`` and
frozen progress at every step (`Invalid envs`_): gate everything on ``valid``, end the
episode, and give it a new course. ``course.reset(mask)`` respawns the masked envs on the
SAME course, which suits a valid env that crashed (this loop keeps a crashed episode's
course; put ``crashed`` into ``finished`` instead to give it a new one); an invalid env
respawned that way stays invalid. ``res.valid`` is live: read it before
``regenerate``, which rewrites it for the regenerated envs.

The ``if`` guards read the masks back on the host. They matter: run eagerly, a
``regenerate(mask=...)`` reruns the whole generator pipeline whenever it is called, even
with an all-zero mask. Call it unconditionally only in a CUDA graph recorded through a
capture Warp registered (``wp.ScopedCapture``, ``wp.capture_begin``, the course's
self-graph, or a ``torch.cuda.graph`` inside
``track_gen.interop.torch.registered(conditional=True)``) with conditional nodes
(``wp.is_conditional_graph_supported()``: CUDA driver and toolkit >= 12.4), where an
all-zero mask is skipped on device (`Masked regeneration`_,
:doc:`/tutorials/cuda-graph-in-a-sim`). Anywhere else (eager, a CPU capture, a plain
``torch.cuda.graph``, an older CUDA) it costs a full generation per call.

Construction does the cold work, outside any capture. It builds the rng and the
generator, whose own construction runs the pipeline once, so ``course.result`` holds the
first batch. It then attaches every component (first sample, bakes and course-line
refresh included) and runs the regenerate body once more as its warm-up. Expect a
construction cost of roughly two generations, which matters for ``repulsive``. Every
sub-tool exists right after construction, so ``step()`` and ``reset()`` only need a prior
``bind()``. Construct the course before you begin a capture: construction raises
``CaptureError``, before allocating anything, when its stream or Warp's current stream is
capturing. ``Course(generator, components, prepare=False)`` and the presets'
``prepare=False`` skip the warm-up; ``course.prepare()`` runs it later, and the first
regeneration call outside a capture runs it by itself (inside one it raises
``CaptureError``). It changes no env's course or state (:ref:`composing-a-course`).

The checkpoint ring is sized from the generator's plan, not from a batch: it holds
``max_checkpoints`` slots per env, by default the plan's size-independent bound (see
`Checkpoint capacity`_). Only when the plan declares no bound does the course read the
construction batch (``config.seeds``) back to size it, as every 0.2.0.dev5 course did.

**A size curriculum.** The construction batch is built at the configured ``size``
(``TrackGenConfig.size``, the only construction-time route, default ``1.0``); a
curriculum grows it later (``course.generator.params["generator"].set("size", arr)``
before ``regenerate``). At flat elevation the default ``max_checkpoints`` already holds
every track that is not truncated, at any size. With elevation and no declared z range,
the readback sizes the ring at the construction size: pass ``max_checkpoints``
(`Checkpoint capacity`_), or construct at the largest size and ``set`` it down. The SDF
and heightfield grids cover each env's bounding box at a fixed resolution, so their cells
grow with the size: raise ``sdf_resolution`` / ``heightfield_resolution`` with it, or use
``collision="segments"`` (:ref:`tuning-size-utilities`).

The generator pipelines are fixed-batch, so every regeneration recomputes the whole
batch. Per-env control is ``reset(mask)`` (respawn on the same course) and
``regenerate(mask=...)`` (new courses for the masked envs only, see `Masked
regeneration`_).

- ``regenerate(*, mask=None, advance=False, stream=None, sync=True)`` reruns the
  generator from the rng's CURRENT seeds and the envs' epochs, refreshes everything
  downstream, then returns ``course.result``. ``advance=True`` first advances the epoch of
  every env, so the call builds new courses without a reseed. Without ``mask`` the
  progress (and localizer) reset is full; with ``mask`` only the masked envs advance and
  are reset (see `Masked regeneration`_).
- ``generate(seeds=None, *, stream=None, sync=True)`` is the reseeding wrapper. Without
  ``seeds`` it is ``regenerate()``. ``seeds=`` takes an ``int`` (env ``e`` gets
  ``seeds + e``, as when constructing ``PerEnvSeededRNG``) or an ``[E]`` int32
  ``wp.array``. It reseeds the rng and zeroes every env's epoch on the same stream as the
  regeneration, without allocating, so ``generate(seeds=s)`` reproduces the courses of a
  fresh ``Course`` built with ``seeds=s``. The new seeds are written into
  ``course.rng.seeds_warp`` in place: if ``CourseConfig.seeds`` was a ``wp.array``, the
  rng holds that very array, so ``generate(seeds=...)`` rewrites it.
- ``enqueue_generate(*, stream=None, mask=None, advance=False)`` runs the same body as
  ``regenerate()``. Under a capture it is recorded; outside one it runs eagerly on the
  resolved stream, never building a self-graph and never blocking.
- ``inputs`` is ``{"regen_mask": ..., "reset_mask": ...}``: two persistent,
  zero-initialised ``[E]`` int32 arrays with stable addresses, for your kernels to write
  the masks of ``regenerate(mask=...)`` and ``reset(mask)`` into (``"reset_mask"`` since
  0.2.0.dev6).
- ``reset(mask=None, stream=None)`` respawns the masked envs on the SAME course (their
  progress and localizer state); ``mask=None`` (since 0.2.0.dev6) means every env.
- ``bind(position, orientation=None, half_extents=None, box_position=None, **arrays)``
  binds the sim buffers: each array goes to every component that declares its name
  (``position`` to progress, the localizer and your components; ``box_position`` defaults
  to ``position`` at one box). The four named arguments are accepted on every course and
  ignored when no component declares them; another unknown name raises ``TypeError``. A
  refused bind changes no binding (:ref:`composing-a-course`).
- ``seeds`` is the generator's :class:`~track_gen.seeds.SeedState` (the same object as
  ``course.generator.seeds``), holding each env's seed (bound to
  ``course.rng.seeds_warp``) and epoch; ``result.generation`` (``[E]`` int32) is the
  epoch each env's current course was built with.

The generators are deterministic under an unchanged RNG: ``regenerate()`` (or
``generate()`` WITHOUT ``seeds=``) reproduces the identical batch (plus a full progress
reset), while ``regenerate(advance=True)`` (next epoch, same seeds) and
``generate(seeds=...)`` (new seeds, epoch zero) give new courses. Each env's course
depends only on its ``(seed, epoch)`` pair. This reproduction is
byte-identical per device for all generators, ``repulsive`` included (its analytic-adjoint
gradient makes it bit-reproducible run-to-run on CUDA as well as CPU); only cross-*device*
output differs, and then only statistically (fp32 rounding). In gates mode the same
object wraps ``GateGenerator`` + gate progress + optional ``DiscChecker``
posts (``post_radius > 0``). The underlying tools stay reachable
(``course.collision``, ``course.progress``, ``course.checkpoints``,
``course.checkpoint_sampler``, ``course.localizer``, ``course.course_line``,
``course.heightfield``), with the same types as before: since 0.2.0.dev6 the course
builds them as the components of ``track_gen.tools`` (:ref:`utilities-components`) and
drives them through one lifecycle, in the same launch order. ``course.components`` maps
each component's name to it, in execution order; to add your own, see
:doc:`/utilities/composing-a-course`.
``step()`` and ``reset()`` never host-synchronize, so they record into your
own sim graph as-is (no flag; ``track_gen.set_capturing`` is a deprecated
no-op). Once ``step()`` is captured, keep writing into the SAME bound
buffers: rebinding after capture leaves the captured graph replaying
against the old pointers. ``course.bind(...)`` with other arrays after ``step()`` ran
under a capture emits ``track_gen.exec.RebindAfterCaptureWarning`` (a ``UserWarning``):
record the graph again after rebinding (the warning fires once on that recipe too; filter
it when you re-record on purpose). Binding the same arrays again is silent.

``StepResult`` has four fields: ``events`` (``ProgressEvents``), ``contacts`` (the
collision result or ``None``), ``frame`` (gates mode, else ``None``) and ``valid``, the
generator result's ``[E]`` int32 ``valid`` (the same array, ``course.result.valid``;
``clone()`` copies it). Since 0.2.0.dev6 it also reads like a mapping from component name
to that component's per-step result: ``res["progress"] is res.events``,
``res["collision"] is res.contacts``, ``res["localizer"] is res.frame``, plus
``res.get(name)``, ``keys()``, ``values()``, ``items()``, ``name in res`` and
``dict(res)``. On a ``CourseConfig`` course the keys are exactly those of the non-``None``
fields (``progress`` and ``collision`` in track mode; ``localizer`` too in gates mode).
It is not a ``collections.abc.Mapping``: ``bool(res)`` stays ``True``, ``iter(res)``
raises ``TypeError``, and the dataclass fields, ``==`` and the repr are unchanged
(:ref:`composing-a-course`).

In gates mode with ``post_radius > 0``, ``course.step().contacts`` is a
:class:`~track_gen.collision.DiscContact` instead of a
:class:`~track_gen.collision.BoxContact` — same ``StepResult`` field, a
different contact type depending on the course's collision checker:

.. code-block:: python

   res = course.step()
   hit = wp.to_torch(res.contacts.hit)      # DiscContact here, not BoxContact

**Gate width in gates mode.** A width-0 gate has a degenerate crossing segment and can
never be passed, so gates mode requires ``gen.gate_width > 0`` at construction. Since
0.2.0.dev5 the bar width is per env (``course.generator.params["gate_frames"]``, a
curriculum or a randomizer): a scalar ``set("gate_width", v)`` with ``v <= 0`` is refused,
but an ``[E]`` array is not range-checked (no host read), and a value ``<= 0`` is used as
``0``: keep every env's width ``> 0``, or those envs' gates can never be passed.

In gates mode ``StepResult`` also carries a third field, ``res.frame`` — the
:class:`~track_gen.localize.TrackFrame` localizing the agent on the gate
``CourseLine`` spline (``s`` arc length, ``n`` signed
right offset, ``n_up`` signed vertical offset, ``segment`` index). It is
``None`` in track mode. The spline is ``course.course_line.centerline``, a
:ref:`Centerline <results-protocols>` (``center``, ``tangent``, ``arclen``,
``length``, ``count``, ``valid`` and a ``layout``), refreshed in place on every
regeneration; a ``TrackLocalizer`` or ``curvature`` of your own can read it.
``course.course_line.track`` is a deprecated alias of it (a ``DeprecationWarning``;
removed at 1.0): it is no longer a ``Track``, so it has no ``outer``, ``inner``,
``normal`` or ``winding``, which used to be NaN or zero placeholders. Since 0.2.0.dev6 its
``generation`` is set (an ``[E]`` int32 written at every refresh from the gates'
``generation``), so a ``TrackLocalizer`` of your own on it resets itself after a
regeneration (:ref:`utilities-self-reset`).

The gate posts' ``DiscChecker`` is given the gates' count and ``valid``: it scans the
``2 * count[e]`` real posts of each env (it used to scan ``2 * max_gates``, skipping the
NaN padding) and reports no hit for an invalid env.

.. _course-checkpoint-capacity:

Checkpoint capacity
-------------------

``CourseConfig.max_checkpoints`` (track mode) is the number of checkpoint slots per env:
``course.checkpoints`` arrays are ``[E * max_checkpoints]``, and a ring longer than that
is clipped (``course.checkpoint_sampler.truncated[e] == 1``). ``None`` (the default) is,
since 0.2.0.dev6, the **plan's bound**, read from the generator's plan with no readback:

.. code-block:: text

   max_checkpoints = max(3, ceil(max_loop_length / checkpoint_spacing))
   max_loop_length = 1.5 * N_max * spacing + dZ

``max_loop_length`` bounds the 3D length of the centerline of every env that is not
truncated (``ComponentContext.for_producer``, :doc:`/contributing/writing-a-component`):
``N_max * spacing`` bounds its plan view, the ``1.5`` is measured headroom for the relax
growth and a boundary's extra ``2 pi * half_width``, and ``dZ`` (``0`` at flat elevation)
bounds the elevation's climbs from the declared ranges of its knobs. It does not depend on
the seeds, the batch, or the course ``size``. At the ``TrackGenConfig`` defaults
(``N_max = 384``, ``spacing = 0.06``, flat: ``max_loop_length = 34.56``):

.. list-table::
   :header-rows: 1

   * - ``checkpoint_spacing``
     - default since 0.2.0.dev6
     - 0.2.0.dev5 default (read back from the construction batch, seeds 0)
   * - 0.3
     - 116
     - 29 (``E = 6``)
   * - 0.5
     - 70
     - 13 (``E = 1``), 17 (``E = 6`` / ``8``), 19 (``E = 32``), 15 to 23 across the five
       sampling generators at ``E = 64``
   * - 0.6
     - 58
     -
   * - 1.0
     - 35
     -

- **What changes.** The shapes of ``course.checkpoints`` (``position``, ``left``,
  ``right``, ``tangent``, ``up_half``: 52 bytes per slot and env, plus 8 bytes per env for
  the set's ``valid`` / ``generation``): 28.4 MiB at ``E = 8192`` and 70 slots, against
  7.7 MiB at 19. The bound grows with ``N_max * spacing``: ``N_max = 2048``,
  ``spacing = 0.02`` and ``checkpoint_spacing = 0.05`` give 1229 slots, 262 MB at
  ``E = 4096``; pass a smaller ``max_checkpoints`` there.
- **What does not.** An env's real rows, its count and its progress are the same at any
  capacity its ring fits: every env that fit the 0.2.0.dev5 capacity is byte-identical.
  The envs the old capacity clipped (a size curriculum: 44 % / 94 % of the valid ``bezier``
  envs at size 2 / 2.5 with a size-1 construction) now get their full ring, so their
  progress and laps change.
- **When the plan declares no bound**, the course keeps the 0.2.0.dev5 readback
  (``ceil(1.5 * longest valid centerline / checkpoint_spacing)`` over the construction
  batch): an elevation whose z knobs declare no range and no ``z_valid_grade`` (every
  elevated ``TrackGenConfig``: ``z_min`` / ``z_max`` are unbounded dynamic knobs), a
  generator that provides its own altitudes (``provides`` ``z``) without a
  ``z_valid_grade`` or its own half-widths (``provides`` ``half_width``), a width other
  than ``width.constant``, a plugin geometry stage after the constant-spacing resample.
  ``TrackGenConfig.z_valid_grade = g > 0`` declares one
  (``dZ = g * 1.5 * N_max * spacing``: 277 slots at ``g = 3``,
  ``checkpoint_spacing = 0.5``).
- **A warning.** When the default (the bound or the readback) clips a valid env of the
  construction batch, the constructor emits one ``UserWarning`` (it reads two ``[E]``
  arrays back, once, after the warm-up). An explicit ``max_checkpoints`` never warns.
- ``course.checkpoint_sampler.max_checkpoints`` reads the capacity in use.

**To keep the 0.2.0.dev5 shapes** (a policy whose input size depends on them, code that
reshapes ``course.checkpoints.position`` with a hard-coded count), pass the old capacity
explicitly: run once under 0.2.0.dev5 and print
``course.checkpoints.position.shape[0] // E``, then pass that number as
``CourseConfig(max_checkpoints=...)``. The rows are the same; the rings it clips are
clipped as before.

One body, one graph
-------------------

``regenerate()`` runs ONE body: the generator pipeline, then the post-generation
refresh (checkpoint resample, SDF and height-field rebakes, course-line refresh, gate-post
rebuild, and the progress and localizer reset: full, or limited to the masked envs with
``mask=``). In component terms: the generator's ``enqueue``, every component's
``on_generate`` in execution order, then every component's ``reset`` in reverse order
(:ref:`composing-a-course`). Where it runs depends on the resolved stream:

- **Inside your capture** (``wp.ScopedCapture``, ``wp.capture_begin(external=True)`` or
  ``torch.cuda.graph``) the body is recorded into YOUR graph, so new courses and the
  per-step ``step()`` / ``reset()`` calls can share one graph. Nothing is synchronized.
- **Outside a capture on** ``cuda`` the body is recorded once into the course's own graph
  (``course.graph``, ``None`` until that first call) and replayed on every later call.
  Each ``(mask array, advance)`` combination has a self-graph of its own, recorded on its
  first use; ``course.graph`` is the one of a plain ``regenerate()``. With the default
  ``sync=True`` the call waits for its stream before returning.
- **On** ``cpu``, or with a generator that has a ``HOST`` stage (a ``capturable=False``
  plugin), it runs eagerly: a ``Course`` does not split its body into segments as a
  ``TrackGenerator`` does. Under a capture, such a generator raises
  ``NotCapturableError`` before launching anything.

That single graph replaces the two graphs of earlier releases (the generator's pipeline
graph plus a refresh graph replayed after it).

.. code-block:: python

   with wp.ScopedCapture(device="cuda") as cap:
       course.regenerate()            # new courses for the whole batch
       for _ in range(K):
           sim_step()                 # your physics: writes the bound buffers
           res = course.step()
           course.reset(done_mask)    # valid envs that crashed: same course
   episode = cap.graph

   course.rng.set_seeds_warp(new_seeds, None)   # in place: the replay reads it
   wp.capture_launch(episode)

Every replay of a recorded ``regenerate()`` reads the rng's live seeds and the live
epochs. To change the courses, record ``regenerate(advance=True)`` (each replay advances
every env's epoch on device), call ``course.rng.set_seeds_warp(new_seeds, None)`` between
replays, or record ``generate(seeds=seed_buf)`` with a stable ``[E]`` int32 ``seed_buf``
that you rewrite in place. An ``int`` passed to ``generate(seeds=...)`` inside a capture is
baked into the graph: every replay reseeds with that same value and rebuilds the same
courses. See :doc:`/tutorials/cuda-graph-in-a-sim` for the full pattern, including the
``torch.cuda.graph`` and side-stream variants.

Masked regeneration
-------------------

``regenerate(mask=m, advance=True)`` gives new courses to the envs with ``m[e] != 0`` and
leaves the others alone, so an RL step graph can auto-reset finished envs on device:

- **Masked envs** advance their epoch (with ``advance=True``), get the course of their
  new ``(seed, epoch)``, and have their runtime state reset as ``reset(m)`` would reset
  it: progress, and the localizer in gates mode. A masked env whose new course is
  invalid again reads ``valid = 0`` at the next step: the loop above marks it done
  again.
- **Unmasked envs** keep their course, the sub-tool data derived from it, and their
  progress and localizer state, bitwise.
- **An all-zero mask** makes the call an exact no-op.

The generator pipeline has fixed shapes, so the regeneration still recomputes EVERY env,
and the refresh (checkpoint resample, SDF and height-field bakes, course line, gate posts)
covers the whole batch. Unmasked envs keep the effective seed their current course was
built from (the seed prologue re-derives it for the masked envs only), so they are
recomputed to identical values, even if their seed or epoch was changed since; on top
of that, the generator's commit restores an unmasked env's ``Track`` rows bit for bit
after the recompute (see :ref:`pipeline-commit`). Such a change
(``course.rng.set_seeds_warp``, a write to ``course.rng.seeds_warp``,
``course.seeds.reset_epoch()``) takes effect for an env when a masked regeneration
selects it, or for every env at the next ``regenerate()`` without a mask. Every call
that is not skipped on device (below) costs a full-batch generation, even with no env
masked.

``advance=False`` rebuilds the masked envs from their current ``(seed, epoch)``: the same
course, unless your own kernel wrote new seeds for them into ``course.rng.seeds_warp``
first. Use it to choose seeds yourself; ``advance=True`` moves to the next epoch instead.

The mask is a contiguous ``[E]`` int32 array on the course device, checked on the host
before anything launches. ``course.inputs["regen_mask"]`` is a stable, zero-initialised one for
your kernels to write; the library reads it and never clears it, so write every entry on
every step. A mask array is recorded by pointer: keep using the same one (outside a
capture, each new array also records a new self-graph).

.. code-block:: python

   done = course.inputs["regen_mask"]
   with wp.ScopedCapture(device="cuda") as cap:
       sim_step()                     # your physics: writes the bound buffers
       res = course.step()
       update_done(res, done)         # your kernel: done[e] = finished or res.valid[e] == 0
       course.regenerate(mask=done, advance=True)   # new courses + resets where done[e]
   step = cap.graph                   # one replay = one RL step, auto-reset included

The body starts with ``track_gen.ops.any_set``, which reduces the mask to a flag on
device, and runs the rest under ``LaunchContext.skip_unless``. When no env is masked the
regeneration is then skipped on device, provided that:

- the call is recorded into a capture Warp registered (``wp.ScopedCapture``,
  ``wp.capture_begin`` with or without ``external=True``, the course's own self-graph, or
  a ``torch.cuda.graph`` inside ``track_gen.interop.torch.registered(conditional=True)``,
  which comes with a rule on later torch allocations), and
- ``wp.is_conditional_graph_supported()`` is true: CUDA driver and toolkit 12.4 or newer.

Otherwise the body runs unconditionally, which is correct (an all-zero mask is an exact
no-op) and costs a full generation per call. A graph that holds the conditional node
cannot be used as a CUDA child graph. See :doc:`/how-it-works/cuda-graph` for the
execution model and :doc:`/tutorials/cuda-graph-in-a-sim` for the full RL pattern,
including the ``torch.cuda.graph`` variant.

Lifetime
--------

Recording any course call (``step()``, ``reset()``, ``regenerate()``, ``generate()``,
``enqueue_generate()``) into a capture Warp registered (``wp.ScopedCapture``,
``wp.capture_begin``, ``track_gen.interop.torch.registered()``) anchors the course, and
with it the generator, every component and every bound array, to that graph, so its
buffers live as long as the graph (since 0.2.0.dev6 for a graph that records only
``step()`` / ``reset()``). A CPU graph (``wp.ScopedCapture("cpu")``) keeps the arrays its
launches touch by itself. Warp does not see a plain ``torch.cuda.graph``: keep ``course``
alive for as long as you replay one, or call ``track_gen.interop.torch.retain(g,
course)`` to tie it to the torch graph ``g``.

One course drives a generator: building another ``Course`` over ``course.generator``
takes it over, and the older course then raises ``RuntimeError`` at its next call
(:ref:`composing-a-course`). A ``CourseConfig`` course builds its own generator, so this
only concerns ``Course(old.generator, ...)``.

When ``max_checkpoints`` came from the readback (`Checkpoint capacity`_), check
``course.checkpoint_sampler.truncated`` after regenerating onto much longer tracks (a
larger ``size``, above).

.. _course-invalid-envs:

Invalid envs
------------

``TrackGenConfig`` / ``GateGenConfig`` expose no ``on_invalid``: a ``Course`` always runs
on ``"keep"``, so an invalid env keeps its failed geometry in ``course.result``. Since
0.2.0.dev6 every sub-tool answers for it as for an env with no course
(:ref:`utilities-invalid-envs`):

- ``contacts`` (track mode, ``BoxContact``): ``oob = 1``, NaN ``distance`` /
  ``nearest`` / ``normal``, ``boundary = -1``;
- ``contacts`` (gates mode, posts, ``DiscContact``): ``hit = 0``, ``disc = -1``,
  ``depth = 0``, NaN ``nearest``; (frames, ``FrameContact``): ``hit = 0``, as before;
- ``events``: frozen (``next_checkpoint`` / ``laps`` / ``progress`` keep their values,
  ``passed = 0``, ``checkpoint_passed = -1``, ``wrong_way = 0``,
  ``wrong_checkpoint = -1``) with ``dist_to_next = 0``;
- ``frame`` (gates mode): NaN ``s`` / ``n`` / ``n_up``, ``segment = -1``, as before (the
  localizer reads the course line);
- ``course.checkpoints`` (track mode): ``count = 0``, NaN slots, ``valid = 0``;
- ``course.heightfield.result`` (the ``HeightField``; also what ``bake()`` returns): NaN
  ``height``, ``lo = (-1, -1)``, ``hi = (1, 1)``;
- ``course.course_line.centerline``: ``count = 0``, NaN, as before.

``res.valid`` is the signal: gate rewards and observations on it with a select, set
``done |= 1 - valid``, and regenerate the done envs with ``advance=True`` (the loop at the
top of this page). Measured at the defaults (``E = 256``, seeds 0): no invalid env for
``bezier``, ``polar`` and ``voronoi``; 1.2 % for ``hull``, 0.4 % for ``checkpoint``,
1.6 % for ``repulsive``; ``bezier`` at size 4: 19 % (truncated at ``N_max``; none at size
3 or below). Truncation grows with the size: size ``N_max`` for the largest one
(:ref:`tuning-size`). In gates mode the bar width decides it (crossing bars, fail bit 4):
with the default ``bezier`` gate generator at ``scale = 1`` (``E = 256``, seeds 7), 8 % at
``gate_width = 0.1``, 50 % at 0.2 and 91 % at 0.4 (``polar``: 0 %, 0 %, 18 %); lower the
width or raise ``scale``, and check ``(course.result.valid.numpy() == 0).mean()`` after
construction.

API: :class:`track_gen.course.Course`,
:class:`track_gen.course.CourseConfig` — see the
:doc:`API reference </reference/api>`.
