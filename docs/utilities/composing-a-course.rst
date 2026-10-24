.. _composing-a-course:

Composing a course
==================

A :term:`course` (:class:`~track_gen.course.Course`) is a generator plus an ordered list
of :term:`components <component>` (:mod:`track_gen.tools`, or your own). The course owns
what ties them together: the order they run in, one ``bind`` for every input, the
regeneration recorded as ONE graph, the per-:term:`env` resets, and the lifetime of every
buffer a recorded graph reads. :doc:`/utilities/course` covers using a course; this page covers
building your own: the presets' arguments, the components and their order, input
binding, results, and the regeneration body.

.. contents:: On this page
   :local:
   :depth: 1

Building a course
-----------------

.. code-block:: python

   from track_gen import TrackGenConfig, TrackGenerator, tools
   from track_gen.course import Course, CourseConfig
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"                     # or "cpu"

   # 1. A preset, over a config, a PipelineSpec, or a built generator
   course = Course.track(presets.track(num_envs=E, device=device),
                         checkpoint_spacing=0.6, collision="segments", seeds=42)
   course = Course.gates(presets.gates(num_envs=E, device=device, gate_width=0.1),
                         collision=0.02, seeds=42)            # a post radius

   # 2. Any generator and any components
   gen = TrackGenerator(presets.track(num_envs=E, device=device), seeds=42)
   course = Course(gen, [tools.Checkpoints(0.6, max_checkpoints="auto"),
                         tools.Progress(),
                         tools.BandCollision("segments")])

   # 3. A flat CourseConfig (Course.from_config(config) and Course(config=...) are the same)
   course = Course(CourseConfig(mode="track", gen=TrackGenConfig(num_envs=E, device=device),
                                seeds=42, collision="segments", checkpoint_spacing=0.6))

``Course(generator, components=(), *, stream=None, prepare=True)`` takes a built
generator: a ``TrackGenerator``, a ``GateGenerator``, a ``Program`` (``pipeline.build(...)``,
including one built with ``prepare=False``, which the course prepares first), or
:ref:`an object of your own with the same surface <producer-protocol>`. A config or a
spec there is a ``TypeError`` naming the preset to use. ``stream=`` is the course's
default stream (the generator's own default stream when not given); ``prepare=`` is
`Preparing`_.

Every way runs ``Course.__init__`` and ends in the same setup, so a subclass's
``__init__`` runs for ``MyCourse.track(...)`` too: it must accept ``(generator,
components, *, stream=None, prepare=True)``.

The presets
~~~~~~~~~~~

.. snippet: skip=signature

.. code-block:: python

   Course.track(source, *, checkpoint_spacing, collision=None, max_checkpoints=None,
                extra=(), seeds=None, stream=None, prepare=True)
   Course.gates(source, *, collision=None, samples_per_gate=8, localize_window=None,
                extra=(), seeds=None, stream=None, prepare=True)

The :term:`presets <preset>` build the components of a track or gate course, then
append ``extra``:

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Preset
     - Components, in order (``name``: component)
   * - ``track``
     - ``checkpoints``: ``tools.Checkpoints(checkpoint_spacing,
       max_checkpoints=max_checkpoints or "auto")``; ``progress``: ``tools.Progress()``;
       ``collision`` when ``collision`` is given: ``tools.BandCollision(collision)`` for
       ``"segments"`` / ``"sdf"``, or the component you pass; then ``extra``.
   * - ``gates``
     - ``checkpoints``: ``tools.GateCheckpoints()``; ``centerline``:
       ``tools.CenterlineFromGates(samples_per_gate)``; ``progress``: ``tools.Progress()``;
       ``collision`` when ``collision`` is given: ``tools.DiscCollision(r)`` for a post
       radius ``r > 0``, or the component you pass (``tools.FrameCollision(...)``);
       ``localizer``: ``tools.Localizer(warm_window=localize_window,
       source="centerline")``; then ``extra``.

- **Sources.** A ``TrackGenConfig`` / ``GateGenConfig`` or a :term:`PipelineSpec` is built
  by the preset, with ``seeds=`` (anything ``TrackGenerator(seeds=)`` takes: an ``int``,
  an ``[E]`` int32 array bound by pointer, a ``PerEnvSeededRNG``, a ``SeedState``) and
  ``stream=``. A built ``TrackGenerator`` / ``GateGenerator`` / ``Program`` is used as is
  (``pipeline.build(spec, registry=...)``, ``out=`` views, generators you built
  yourself); passing ``seeds=`` or ``stream=`` with one is a ``TypeError`` (they
  configure the generator the preset builds), and a generator of the other family is a
  ``TypeError`` naming the other preset.
- **The collision slot.** ``track``: ``None`` (the default), ``"segments"``, ``"sdf"``, or
  a component. ``gates``: ``None``, a post radius, or a component. A component in this
  slot must be named ``"collision"``: it fills ``StepResult.contacts`` and
  ``course.collision``. A collision under another name goes in ``extra`` (``ValueError``
  otherwise). :doc:`/utilities/collision` compares the backends.
- **Checks before anything is built.** ``checkpoint_spacing`` a number ``> 0``, the
  ``collision`` value (a ``bool`` or another string is refused), a post radius ``> 0``,
  ``max_checkpoints`` an int ``>= 3`` (``"auto"`` is ``None``), ``samples_per_gate >=
  2``, ``localize_window >= 1``, ``extra`` a sequence (a single component is a
  ``TypeError``, as for ``Course(generator, components)``), the source's family, and
  ``extra`` sorted together with the preset's own names (duplicate names, an unknown
  ``after`` / ``source``, a cycle, an attached component) raise before any generator is
  built. ``gates`` also reads the construction ``gate_frames.gate_width`` from the
  config, the spec or the built generator and raises ``ValueError`` when it is ``0``
  (a width-0 gate can never be passed); when a randomizer draws the width
  (``with_randomize({"gate_frames.gate_width": (lo, hi)})``), its construction ``lo``
  must be ``> 0`` instead and the construction width is not used. After building, it
  refuses a scalar ``set`` of a width ``<= 0``, on a built generator too (a
  randomizer's live ``lo`` is not checked: keep it ``> 0``). ``presets.gates`` defaults
  ``gate_width=0``: pass one, or randomize it.
- **Checkpoint capacity** (``track``). ``max_checkpoints``, else
  ``Checkpoints(..., max_checkpoints="auto")``: the longest track the generator can
  produce, ``max(3, ceil(max_loop_length / checkpoint_spacing))``, else (an elevated
  config, which declares no bound) the readback of the construction batch
  (:ref:`course-checkpoint-capacity`). When a capacity the course chose clips a valid
  env of the first batch, construction warns once (``...; pass a larger
  max_checkpoints``).
- **Resolution checks.** A preset's ``BandCollision("sdf")`` and ``HeightFieldBake`` read
  the largest course :term:`size` the generator can draw (a size randomizer's
  construction ``hi``, else the construction size) and warn at construction when their
  grid does not cover it (:ref:`collision-resolution-check`). A curriculum that widens the
  range later is not checked again: pass its final ``hi`` as ``size_hi=``.
- **Seeds and the rng.** ``course.rng`` is the ``PerEnvSeededRNG`` you passed as
  ``seeds=`` (the :term:`seeds <seed>`), else ``None``; ``course.seeds`` is the generator's ``SeedState`` either
  way (``course.seeds.seeds`` holds the per-env seeds). ``generate(seeds=...)`` follows
  the rule of whoever owns the seeds: through the rng when there is one (it writes the
  rng's array), ``SeedState.set`` on seeds the generator owns, and a ``RuntimeError``
  for seeds bound to your array with no rng (the library never writes a bound seed
  array: write it, then ``regenerate()``). With ``seeds=PerEnvSeededRNG(...)``,
  ``Course.gates(cfg, collision=r, seeds=rng)`` is byte-identical to
  ``Course(CourseConfig(mode="gates", gen=cfg, seeds=s, post_radius=r))``.

Components and their order
--------------------------

The components run in a **stable topological order** of their ``after`` and ``source``
names over the order you give (an already-ordered list is kept as is):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Phase
     - Order
   * - ``attach`` (construction)
     - sorted
   * - ``on_generate`` (in the regeneration body)
     - sorted: each component refreshes after what it reads
   * - ``step``
     - sorted
   * - ``reset`` (in the regeneration body and in ``Course.reset``)
     - reversed: a component is reset before the components it reads

``course.components`` is a read-only ``name -> component`` mapping in execution order.
Construction refuses (before attaching anything): a name that is not a non-empty ``str``
(``TypeError``); two components with one name (``ValueError``: names are unique within a
course); an ``after`` / ``source`` naming no component of the course; a cycle; a component
already attached to another course or context (``ValueError``: build a new instance); a
component lacking any of ``attach``, ``bind``, ``on_generate``, ``step``, ``reset``
(``TypeError``: subclass ``tools.Component`` for the defaults). Then each component goes
through ``ComponentContext.add`` (its declared reads, ``requires`` / ``optional``,
``topology``, input conflicts, then ``attach``; :ref:`writing-a-component`).

Name a third-party component with a namespaced name (``"acme.lookahead"``), as plugins
name their :term:`stages <stage>`, so it never collides with a built-in one. The built-in
names back the course's attributes: ``checkpoints``, ``progress``, ``collision``,
``localizer``, ``centerline``, ``heightfield``. On any course: ``course.progress``,
``course.collision``, ``course.heightfield`` and ``course.localizer`` are the
``utility`` of the component of that name; ``course.checkpoint_sampler`` the
``utility`` of ``checkpoints`` (``course.checkpoints`` is that component's
``result``); ``course.course_line`` the ``utility`` of ``centerline``; each is ``None``
when the component is absent.

.. _composing-a-course-binding-inputs:

Binding inputs
--------------

``course.bind(**arrays)`` binds every component input at once, by name:

.. snippet: setup

   import warp as wp

   pos = wp.zeros(E, dtype=wp.vec3f, device=device)
   quat = wp.full(E, wp.quat_identity(), dtype=wp.quatf, device=device)
   he = wp.full(E, wp.vec2f(0.02, 0.01), dtype=wp.vec2f, device=device)

.. code-block:: python

   course = Course.track(presets.track(num_envs=E, device=device), checkpoint_spacing=0.6,
                         extra=[tools.Localizer()], seeds=42)     # no collision
   course.bind(position=pos)          # Progress and Localizer both read it

   boxed = Course.track(presets.track(num_envs=E, device=device), checkpoint_spacing=0.6,
                        collision="segments", seeds=42)
   boxed.bind(position=pos, orientation=quat, half_extents=he)    # + a box collision

- **One array, every declarer.** An array goes to every component that declares its name
  (``Progress``, ``Localizer`` and a component of yours can all take ``position``). An
  ``InputSpec.default`` routes an input that is not given from another one
  (``box_position`` from ``position`` at one box); a name given explicitly wins over a
  default route. Two declarations of one name with different ``(dtype, slots)`` are
  refused at construction.
- **Validated, then bound, atomically.** Every route is checked (unknown and missing
  names, shape, dtype, device) before anything is rebound; if a component's ``on_bind``
  raises part-way, the components already rebound by the call, and the one that raised,
  get their previous arrays back, their rebind-after-capture probes are restored, and
  the error propagates. A refused bind leaves every previous binding in place, and a
  later rebind after a capture still warns.
- **Names.** ``position``, ``orientation``, ``half_extents`` and ``box_position`` (also
  accepted positionally, in that order) are accepted on every course and ignored when no
  component declares them. Any other unknown name is a ``TypeError`` listing the
  component inputs.
- **Generator inputs are not bound by the course.** A name of
  ``course.generator.input_info`` (``"centerline.xy"``, a stage input) raises
  ``TypeError`` with the recipe: write ``course.generator.inputs[name]`` in place (a
  stable array that every regeneration and every replay reads), e.g. a ``wp.copy`` into
  it on the regeneration's stream.
- ``step()`` raises ``RuntimeError("call bind() before step()")`` while a component with
  a required input is unbound.

Calling ``bind`` again replaces every binding. The course's own graphs never read a bound
input (:ref:`The component contract in a course <component-contract-in-course>`), so a
rebind never replays a stale course graph; a graph of yours that recorded ``step()``
keeps the old arrays: record it again (the utilities emit ``RebindAfterCaptureWarning``
once, :ref:`component-rebind`).

Reading results
---------------

``course.step()`` returns the same :class:`~track_gen.course.StepResult` instance every
call. Besides its four fields it gives read-only mapping access to the result of every
**per-step** component, by name:

.. code-block:: python

   res = course.step()
   res["progress"]                    # ProgressEvents (the same object as res.events)
   res["localizer"].s                 # the Localizer's TrackFrame
   "collision" in res, list(res.keys()), dict(res)
   res.get("collision")               # None: this course has no collision
   res.valid                          # the generator result's [E] int32 valid
   course.components["checkpoints"].result   # a per-generation result: not an entry

- **Entries** are the components that override ``Component.step`` (their ``result``):
  progress, collisions, the localizer, your step components (``res["acme.lookahead"]``).
  Per-generation components (checkpoints, the course line, bakes, props) keep the base
  ``step`` and are read through ``course.components[name].result`` (or the course
  attributes). ``res[name]`` on another name raises ``KeyError`` listing the entries.
- **The four fields** are the entries by name: ``events`` is ``res["progress"]``,
  ``contacts`` ``res["collision"]``, ``frame`` ``res["localizer"]`` (``None`` when
  absent), and :term:`valid` is ``course.result.valid`` (generator state: never write it;
  it is live, so read it before a regeneration). On a ``CourseConfig`` course the
  entries are exactly the sources of these fields.
- **Not a** ``collections.abc.Mapping``: ``bool(res)`` is ``True``, ``iter(res)`` raises
  ``TypeError`` (use ``keys()`` / ``items()``), and it stays a dataclass
  (``dataclasses.fields``, ``==``, positional construction, its generated repr, which
  ``pprint`` lays out), so generic tree utilities (Lightning's ``apply_to_collection``,
  ``dm-tree``) treat it as a dataclass. ``dataclasses.replace(res, ...)`` keeps the
  entries of the four fields only. Assigning a field (``res.events = x``, as those
  utilities do on their copy) updates its entry on that instance only (``None`` removes
  it). Visible to duck typing: ``hasattr(res, "__iter__")`` (it is ``None``),
  ``isinstance(res, collections.abc.Container)`` (``__contains__``), and ``**res`` /
  ``dict.update(res)`` (``keys()`` + ``__getitem__``) succeed.
- ``res.clone()`` deep-copies every entry (a ``wp.array``, an object with ``clone()``, a
  dataclass's ``wp.array`` fields) and ``valid``; an entry it cannot copy raises
  ``TypeError`` naming the component. It allocates: call it outside your graph.

Regeneration: one body, one graph
---------------------------------

``regenerate()``, ``enqueue_generate()`` and ``generate()`` run ONE body:

.. code-block:: text

   regenerate()                    generator.enqueue(ctx, advance=a)
   regenerate(advance=True)          -> on_generate(ctx, None) for each component (sorted)
                                     -> reset(ctx, all envs)  for each component (reversed)

   regenerate(mask=m, advance=a)   any_set(m) -> flag
                                   skip_unless(flag):
                                     generator.enqueue(ctx, advance=a, mask=m)
                                       (advance(m) -> stages -> masked commit -> generation)
                                     -> on_generate(ctx, m) for each component (sorted)
                                     -> reset(ctx, m)       for each component (reversed)

With an all-zero mask the skippable part is an **exact identity**: the masked commit
restores every env's rows, ``on_generate(ctx, m)`` leaves every unmasked env's rows and
state unchanged (the component contract), and ``reset(ctx, m)`` writes only masked envs.
So recording it unconditionally is correct; skipping it on device only saves the cost.
:ref:`rl-cost` lists where the skip applies (a CUDA :term:`conditional node`) and where
the body runs in full.

Outside a capture on ``cuda``, ``regenerate`` records each variant once into a CUDA graph
of the course's own and replays it:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Call
     - The course's own graph
   * - ``regenerate()``, ``generate()``, ``generate(seeds=...)`` (its regeneration)
     - ``course.graph``
   * - ``regenerate(advance=True)``
     - one more
   * - ``regenerate(mask=m, advance=a)``
     - one per ``(a, m)`` array, so keep using the same mask array

Besides ``course.graph``, at most 16 variants get a graph of their own; calls with a
further new variant run eagerly, with one ``RuntimeWarning``. ``enqueue_generate`` never builds one. A generator with a step that
runs on the host (a plugin registered with ``capturable=False``) runs the body eagerly
and raises ``NotCapturableError`` under a capture.

Recorded into your own graph next to ``step()``, the masked regeneration is a per-env
auto-reset: :doc:`/user-guide/rl-auto-reset` has the recipe (:ref:`rl-in-graph`) and
:doc:`/tutorials/cuda-graph-in-a-sim` the recording patterns.

Preparing
---------

Construction (``prepare=True``, the default) runs the regeneration body once as a
warm-up, so every module is loaded before you record anything (a :term:`capture`). ``prepare=False`` skips
it: the first regeneration call outside a capture prepares, one inside a capture raises
``CaptureError`` (a CPU capture included), and ``course.prepare()`` prepares explicitly.
``prepare()`` is a no-op on a prepared course; on an unprepared one it runs the masked
body on an all-zero private mask, the exact identity above, so it changes no env's course
or state even after steps.

Ownership and lifetime
----------------------

**One course per generator.** Building a ``Course`` over a generator that a live course
drives (the same object) takes it over once its construction completes: the older
course raises ``RuntimeError`` at its next ``step``, ``reset``, ``regenerate``,
``enqueue_generate`` or ``generate`` (its components would read a source the new course
regenerates); a construction that raises (a warning turned into an error included)
takes nothing over. Rebuilding (``course = Course(gen, ...)`` run twice) works. Calling
``course.generator.generate()`` yourself regenerates the source without refreshing the
components: regenerate through the course.

**Lifetime.** The course holds the generator, every component and the ``StepResult``;
each component holds its buffers and its bound arrays. A graph that records a course call
keeps the course and the arrays that call reads alive in most captures, but not under a
plain ``torch.cuda.graph``: :ref:`capture-lifetime` has the rules.

Your own generator, a curriculum
--------------------------------

- A course drives any object with the generator surface (``result``, ``layout``,
  ``enqueue`` and a few more): :ref:`producer-protocol` lists it, and
  :ref:`component-worked-example` builds a component end to end.
- A curriculum changes the generator's :term:`dynamic fields <dynamic field>` between
  regenerations (``course.generator.params[...]``): :ref:`curricula-course` grows the
  course size of a course, with the checkpoint ring and the grids sized for it.

Troubleshooting
---------------

- ``cudaErrorStreamCaptureUnjoined`` (the capture is invalidated) at
  ``regenerate(mask=...)``: a side stream forked in your capture is not joined yet. Join
  it first, or record the call inside ``track_gen.exec.unconditional()``.
- A graph that holds the masked regeneration cannot be added as a CUDA **child graph**
  (it holds a conditional node): record it inside ``exec.unconditional()``.
- ``CaptureError`` naming the components at ``regenerate(mask=...)``: one of them uses an
  operation a conditional body cannot hold (a copy to or from pageable host memory, an
  event, another stream) in ``on_generate`` or ``reset``; ``check_component``
  (``capture.caller``, :ref:`kit-check-component`) finds which.
  ``exec.unconditional()`` records the body without the node.
- ``CaptureError`` "built with stream=": the course has a default stream and is called
  while another stream records; pass ``stream=``.
- ``CaptureError`` "built with prepare=False": call ``course.prepare()`` before
  recording.
- ``RuntimeError`` "driven by another Course": a newer course took the generator over;
  use it.
- ``TypeError`` "is an input of the generator": write ``course.generator.inputs[name]``
  in place.
- Torch ops failing to capture after the masked regeneration under
  ``registered(conditional=True)``: record the masked regeneration after every allocating
  torch op.

:ref:`capture-troubleshooting` covers the errors of recording in general.

API: :class:`track_gen.course.Course`, :class:`track_gen.course.StepResult`,
:mod:`track_gen.tools` (:ref:`reference-runtime`).
