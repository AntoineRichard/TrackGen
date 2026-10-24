.. _composing-a-course:

Composing a course
==================

Since 0.2.0.dev6 a :class:`~track_gen.course.Course` is a **composition root**: a
producer (a generator) plus an ordered list of components (:mod:`track_gen.tools`), with
the course owning what ties them together: the order they run in, one ``bind`` for every
input, the regeneration body recorded as ONE graph, the per-env resets, and the lifetime of
every buffer a recorded graph reads. ``Course(CourseConfig)`` builds exactly the course it
built before (the same outputs and launches); this page is about building your own.

.. contents:: On this page
   :local:
   :depth: 1

Three front doors
-----------------

.. code-block:: python

   from track_gen import TrackGenConfig, GateGenConfig, TrackGenerator, tools
   from track_gen.course import Course, CourseConfig

   cfg = TrackGenConfig(num_envs=E, device="cuda:0")

   # 1. The 0.x config, unchanged (Course.from_config(config) is the same call)
   course = Course(CourseConfig(mode="track", gen=cfg, seeds=42, collision="segments",
                                checkpoint_spacing=0.6))

   # 2. The presets: a config, a PipelineSpec, or a built generator / Program
   course = Course.track(cfg, checkpoint_spacing=0.6, collision="segments", seeds=42)
   course = Course.gates(GateGenConfig(num_envs=E, device="cuda:0", gate_width=0.1),
                         collision=0.02, seeds=42)            # a post radius

   # 3. Any producer and any components
   gen = TrackGenerator(cfg, seeds=42)
   course = Course(gen, [tools.Checkpoints(0.6, max_checkpoints="auto"),
                         tools.Progress(),
                         tools.BandCollision("segments")])

``Course(config=cfg)``, the 0.x keyword, is the same call as ``Course(cfg)``. Every front
door runs ``Course.__init__`` and ends in the same setup, so a subclass's ``__init__``
runs for ``MyCourse.track(...)`` too: it must accept ``(generator, components, *,
stream=None, prepare=True)``.

``Course(generator, components=(), *, stream=None, prepare=True)`` takes a built
producer: a ``TrackGenerator``, a ``GateGenerator``, a ``Program``
(``pipeline.build(...)``, including one built with ``prepare=False``, which the course
prepares first), or any object with the producer surface (`The Producer protocol`_). A
config or a spec there is a ``TypeError`` naming the preset to use. ``stream=`` is the
course's default stream (the generator's own default stream when not given);
``prepare=`` is `Preparing`_.

The presets
~~~~~~~~~~~

.. code-block:: python

   Course.track(source, *, checkpoint_spacing, collision=None, max_checkpoints=None,
                extra=(), seeds=None, stream=None, prepare=True)
   Course.gates(source, *, collision=None, samples_per_gate=8, localize_window=None,
                extra=(), seeds=None, stream=None, prepare=True)

They build the components a ``CourseConfig`` course of that mode has, then append
``extra``:

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

- **Sources.** A ``TrackGenConfig`` / ``GateGenConfig`` or a ``PipelineSpec`` is built by
  the preset, with ``seeds=`` (anything ``TrackGenerator(seeds=)`` takes: an ``int``, an
  ``[E]`` int32 array bound by pointer, a ``PerEnvSeededRNG``, a ``SeedState``) and
  ``stream=``. A built ``TrackGenerator`` / ``GateGenerator`` / ``Program`` is used as is
  (``pipeline.build(spec, registry=...)``, ``out=`` views, pre-built facades); passing
  ``seeds=`` or ``stream=`` with one is a ``TypeError`` (they configure the generator the
  preset builds), and a producer of the other family is a ``TypeError`` naming the other
  preset.
- **The collision slot.** ``track``: ``None`` (the default, as ``CourseConfig.collision``),
  ``"segments"``, ``"sdf"``, or a component. ``gates``: ``None``, a post radius, or a
  component. A component in this slot must be named ``"collision"``: it fills
  ``StepResult.contacts`` and ``course.collision``. A collision under another name goes in
  ``extra`` (``ValueError`` otherwise).
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
  narrows the field to ``> 0`` for a scalar ``set`` as ``Course(CourseConfig)`` does, on
  a built generator too (a randomizer's live ``lo`` is not narrowed: keep it ``> 0``).
  ``presets.gates`` defaults ``gate_width=0``: pass one, or randomize it.
- **Checkpoint capacity** (``track``). ``max_checkpoints``, else
  ``Checkpoints(..., max_checkpoints="auto")``: the plan's bound
  ``max(3, ceil(max_loop_length / checkpoint_spacing))``, else (an elevated config,
  whose plan declares no bound) the readback of the construction batch, as a
  ``CourseConfig`` course does (:ref:`course-checkpoint-capacity`). When a capacity the
  course chose clips a valid env of the first batch, construction warns once (``...;
  pass a larger max_checkpoints``).
- **Seeds and the rng.** ``course.rng`` is the ``PerEnvSeededRNG`` you passed as
  ``seeds=``, else ``None``; ``course.seeds`` is the generator's ``SeedState`` either
  way (``course.seeds.seeds`` holds the per-env seeds). ``generate(seeds=...)`` follows
  the rule of whoever owns the seeds: through the rng when there is one (as 0.x: it
  writes the rng's array), ``SeedState.set`` on seeds the generator owns, and a
  ``RuntimeError`` for seeds bound to your array with no rng (the library never writes
  a bound seed array: write it, then ``regenerate()``). With
  ``seeds=PerEnvSeededRNG(...)``, ``Course.gates(cfg, collision=r, seeds=rng)`` is
  byte-identical to ``Course(CourseConfig(mode="gates", gen=cfg, seeds=s,
  post_radius=r))``.

The Producer protocol
~~~~~~~~~~~~~~~~~~~~~

``track_gen.pipeline.Producer`` documents what a course reads from its generator. It is a
``typing.Protocol`` for documentation and type checkers; the course duck-types it and
never ``isinstance``-checks it (a facade's ``program``, when it is a ``Program``, is the
producer, so a facade reduces to its ``Program``; an unrelated ``program`` / ``rng``
attribute of a third-party producer is ignored). A missing required member is a
``TypeError`` listing it.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Member
     - Required
     - Used for
   * - ``result``
     - yes
     - The read target of every component with ``source=None``; ``StepResult.valid`` is
       its ``valid`` (when it has one).
   * - ``layout`` (a :class:`~track_gen.results.Layout`)
     - yes
     - ``num_envs``, the device, the ``ComponentContext``.
   * - ``enqueue(ctx, *, advance=False, mask=None)``
     - yes
     - The generation inside the regeneration body. With ``mask`` it leaves the unmasked
       envs' result rows bit for bit and is an exact no-op on an all-zero mask; it may run
       inside a CUDA conditional-node body (`The component contract in a course`_).
   * - ``seeds`` (``SeedState``)
     - no
     - ``course.seeds``, ``generate(seeds=)``.
   * - ``capture_class`` (``CaptureClass``)
     - no (``STATIC``)
     - ``HOST``: eager outside a capture, ``NotCapturableError`` inside one.
   * - ``prepared`` + ``prepare()``
     - no (prepared)
     - An unprepared producer is prepared first. Without them, the producer must have
       loaded every module ``enqueue`` uses before it reaches a course.
   * - ``default_stream``
     - no (``None``)
     - The course's default stream when ``stream=`` is not given.
   * - ``capacity``, ``bounds``
     - no
     - Read by ``ComponentContext.for_producer`` before the plan (``max_loop_length``;
       ``size_hi``).
   * - ``inputs`` + ``input_info``
     - no
     - The generator's own inputs (`Binding inputs`_).

``Program`` satisfies it (its read-only ``prepared`` and ``default_stream`` are the
runner's), and the facades' read-only ``rng`` gives ``course.rng``, so a course reads no
private member of a producer. A mixed batch (two generators writing views of one result)
is a producer too; ``pipeline.MixedBatch`` is not shipped (`Mixed batches`_).

Components and their order
--------------------------

The components run in a **stable topological order** of their ``after`` and ``source``
names over the order you give (an already-ordered list is kept as is, so every
``CourseConfig`` course keeps its 0.x order):

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
``topology``, input conflicts, then ``attach``; :doc:`/contributing/writing-a-component`).

Name a third-party component with a namespaced name (``"acme.lookahead"``), as plugins
name their stages, so it never collides with a built-in one. The built-in names are the
0.x attributes they back: ``checkpoints``, ``progress``, ``collision``, ``localizer``,
``centerline``, ``heightfield``. On any course: ``course.progress``,
``course.collision``, ``course.heightfield`` and ``course.localizer`` are the
``utility`` of the component of that name; ``course.checkpoint_sampler`` the
``utility`` of ``checkpoints`` (``course.checkpoints`` is that component's
``result``); ``course.course_line`` the ``utility`` of ``centerline``; each is ``None``
when the component is absent.

Binding inputs
--------------

``course.bind(**arrays)`` binds every component input at once, by name:

.. code-block:: python

   course.bind(position=pos)          # Progress, Localizer and your component all read it
   course.bind(position=pos, orientation=quat, half_extents=he)   # + a box collision

- **One array, every declarer.** An array goes to every component that declares its name
  (``Progress``, ``Localizer`` and ``CurvatureLookahead`` all take ``position``). An
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
- **Names.** ``position``, ``orientation``, ``half_extents`` and ``box_position`` (the
  0.x arguments, positional in that order) are accepted on every course and ignored when
  no component declares them. Any other unknown name is a ``TypeError`` listing the
  component inputs.
- **Generator inputs are not bound by the course.** A name of
  ``course.generator.input_info`` (``"centerline.xy"``, a stage input) raises
  ``TypeError`` with the recipe: write ``course.generator.inputs[name]`` in place (a
  stable array that every regeneration and every replay reads), e.g. a ``wp.copy`` into
  it on the regeneration's stream. Binding a generator input to your own array arrives
  with ``build(inputs=)``.
- ``step()`` raises ``RuntimeError("call bind() before step()")`` while a component with
  a required input is unbound.

Calling ``bind`` again replaces every binding. The course's own graphs never read a bound
input (`The component contract in a course`_), so a rebind never replays a stale course
graph; a graph of yours that recorded ``step()`` keeps the old arrays: record it again
(the utilities emit ``RebindAfterCaptureWarning`` once, :ref:`component-rebind`).

Reading results
---------------

``course.step()`` returns the same :class:`~track_gen.course.StepResult` instance every
call. Besides its 0.x fields it gives read-only mapping access to the result of every
**per-step** component, by name:

.. code-block:: python

   res = course.step()
   res["progress"]                    # ProgressEvents (the same object as res.events)
   res["acme.lookahead"].kappa        # your component's result
   "collision" in res, list(res.keys()), dict(res)
   res.get("localizer")               # None when the course has none
   res.valid                          # the generator result's [E] int32 valid
   course.components["checkpoints"].result   # a per-generation result: not an entry

- **Entries** are the components that override ``Component.step`` (their ``result``):
  progress, collisions, the localizer, your step components. Per-generation components
  (checkpoints, the course line, bakes, props) keep the base ``step`` and are read through
  ``course.components[name].result`` (or the 0.x attributes). ``res[name]`` on another
  name raises ``KeyError`` listing the entries.
- **The 0.x fields** are the entries by name: ``events`` is ``res["progress"]``,
  ``contacts`` ``res["collision"]``, ``frame`` ``res["localizer"]`` (``None`` when absent),
  and ``valid`` is ``course.result.valid`` (producer state: never write it; it is live,
  so read it before a regeneration). On a ``CourseConfig`` course the entries are exactly
  the sources of these fields.
- **Not a** ``collections.abc.Mapping``: ``bool(res)`` is ``True``, ``iter(res)`` raises
  ``TypeError`` (use ``keys()`` / ``items()``), and the dataclass is the 0.x one
  (``dataclasses.fields``, ``==``, positional construction, its generated repr, which
  ``pprint`` lays out), so generic tree utilities (Lightning's ``apply_to_collection``,
  ``dm-tree``) treat it as the dataclass it was. ``dataclasses.replace(res, ...)`` keeps
  the 0.x entries only. Assigning a 0.x field (``res.events = x``, as those utilities do
  on their copy) updates its entry on that instance only (``None`` removes it).
  Visible to duck typing: ``hasattr(res, "__iter__")`` (it is ``None``), ``isinstance(res,
  collections.abc.Container)`` (``__contains__``), and ``**res`` / ``dict.update(res)``
  (``keys()`` + ``__getitem__``) now succeed.
- ``res.clone()`` deep-copies every entry (a ``wp.array``, an object with ``clone()``, a
  dataclass's ``wp.array`` fields) and ``valid``; an entry it cannot copy raises
  ``TypeError`` naming the component. It allocates: call it outside your graph.

Regeneration: one body, one graph
---------------------------------

``regenerate()``, ``enqueue_generate()`` and ``generate()`` run ONE body:

.. code-block:: text

   regenerate()                    producer.enqueue(ctx, advance=a)
   regenerate(advance=True)          -> on_generate(ctx, None) for each component (sorted)
                                     -> reset(ctx, all envs)  for each component (reversed)

   regenerate(mask=m, advance=a)   any_set(m) -> flag
                                   skip_unless(flag):
                                     producer.enqueue(ctx, advance=a, mask=m)
                                       (advance(m) -> stages -> masked commit -> generation)
                                     -> on_generate(ctx, m) for each component (sorted)
                                     -> reset(ctx, m)       for each component (reversed)

With an all-zero mask the skippable part is an **exact identity**: the masked commit
restores every env's rows, ``on_generate(ctx, m)`` leaves every unmasked env's rows and
state unchanged (the component contract), and ``reset(ctx, m)`` writes only masked envs.
So recording it unconditionally is correct; skipping it on device only saves the cost.

.. list-table:: Where the masked body goes
   :header-rows: 1
   :widths: 55 45

   * - The call is recorded into (or runs in)
     - An all-zero mask
   * - ``wp.ScopedCapture`` / ``wp.capture_begin`` (``external=True`` included, except on
       torch's capturing stream), the course's own self-graph, a ``torch.cuda.graph`` inside
       ``track_gen.interop.torch.registered(conditional=True)``, with
       ``wp.is_conditional_graph_supported()`` (CUDA driver and toolkit >= 12.4)
     - skipped on device (a CUDA conditional node)
   * - a plain ``torch.cuda.graph``, ``registered()`` without ``conditional=True``, a CPU
       capture, an older CUDA, or inside ``track_gen.exec.unconditional()``
     - recorded unconditionally: an exact no-op that costs a full generation
   * - eager (no capture)
     - runs: an exact no-op that costs a full generation (guard it with a host check)

Outside a capture on ``cuda``, ``regenerate`` records each variant once into the course's
own graph and replays it:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Call
     - Self-graph
   * - ``regenerate()``, ``generate()``, ``generate(seeds=...)`` (its regeneration)
     - ``course.graph``
   * - ``regenerate(advance=True)``
     - one more
   * - ``regenerate(mask=m, advance=a)``
     - one per ``(a, m)`` array, so keep using the same mask array

At most 16 keyed graphs are kept, then the calls run eagerly with one ``RuntimeWarning``.
``enqueue_generate`` never self-graphs. A producer with a ``HOST`` stage runs eagerly (a
course does not split its body into segments) and raises ``NotCapturableError`` under a
capture.

.. _course-rl-loop:

The RL loop in one graph
------------------------

``course.inputs`` holds two stable, zero-initialised ``[E]`` int32 masks for your kernels
to write: ``"regen_mask"`` (for ``regenerate(mask=...)``) and ``"reset_mask"`` (for
``reset(mask)``). The library reads them and never clears them: write every entry on
every step.

.. code-block:: python

   import warp as wp

   from track_gen import TrackGenConfig, tools
   from track_gen.course import Course

   course = Course.track(TrackGenConfig(num_envs=E, device="cuda:0"),
                         checkpoint_spacing=0.6, collision="segments",
                         extra=[tools.Localizer()], seeds=5)
   course.bind(position=pos, orientation=quat, half_extents=he)   # your stable buffers
   done = course.inputs["regen_mask"]

   with wp.ScopedCapture(device="cuda:0") as cap:
       physics_step(pos, quat)            # your kernels, on the capture stream
       res = course.step()
       update_done(res, done)             # done[e] = finished or res.valid[e] == 0
       course.regenerate(mask=done, advance=True)   # new courses (and resets) where done
   step_graph = cap.graph

   for _ in range(num_steps):
       wp.capture_launch(step_graph)      # one RL step, auto-reset included

Read ``res.valid`` (and every reward input) in ``update_done`` and your reward kernels,
before the regeneration, which rewrites ``valid`` for the regenerated envs; gate rewards
and observations on it with a select (:ref:`course-invalid-envs`). Under torch, record
the same calls inside ``torch.cuda.graph(g)`` (library calls adopt torch's capture
stream; launch your own kernels with ``stream=wp.stream_from_torch()``), and wrap them in
``track_gen.interop.torch.registered(graph=g, conditional=True)`` to get the device skip:
record the masked regeneration after every torch op that allocates
(:doc:`/how-it-works/cuda-graph`).

- **Join every side stream before** ``course.regenerate(mask=...)``. ``wp.capture_if``
  ends and restarts the stream's capture around the conditional node, so a fork that is
  not joined yet fails with ``cudaErrorStreamCaptureUnjoined`` and invalidates the
  capture. Or record the call inside ``track_gen.exec.unconditional()``.
- **A course built with** ``stream=s`` (or over a generator built with ``stream=s``)
  runs on ``s`` by default. Record it with ``stream=`` (the capturing stream, or
  ``"torch"``): called without ``stream=`` while another stream is capturing (Warp's
  current stream, or torch's in this thread), it raises ``CaptureError`` instead of
  running eagerly on ``s``, outside your graph.
- ``generate(seeds=buf)`` records its reseed and the regeneration as one unit; an
  ``int`` seed is baked into the graph.

Preparing
---------

Construction (``prepare=True``, the default) runs the regeneration body once as a
warm-up, so every module is loaded before you record anything. ``prepare=False`` skips
it: the first regeneration call outside a capture prepares, one inside a capture raises
``CaptureError`` (a CPU capture included), and ``course.prepare()`` prepares explicitly.
``prepare()`` is a no-op on a prepared course; on an unprepared one it runs the masked
body on an all-zero private mask, the exact identity above, so it changes no env's course
or state even after steps.

Lifetime
--------

The course holds the generator, every component and the ``StepResult``; each component
holds its buffers and its bound arrays. Recording any course call (``step``, ``reset``,
``regenerate``, ``enqueue_generate``, ``generate``) into a capture Warp registered
(``wp.ScopedCapture``, ``wp.capture_begin``, ``registered()``) anchors the course on that
``wp.Graph``, with the arrays the recorded call reads that the course does not own: the
binding ``step()`` read (a later rebind drops the course's references to it, not the
graph's) and a ``mask`` / ``seeds`` array you pass. Drop every reference, or rebind, and
the graph still finds every buffer it reads. A CPU graph (``wp.ScopedCapture("cpu")``)
keeps the arrays its launches touch by itself (the course object is released, the arrays
live as long as the graph). A plain ``torch.cuda.graph`` is not seen by Warp: keep the
course and those arrays alive, or call ``track_gen.interop.torch.retain(g, course,
mask, ...)``.

**One course per generator.** Building a ``Course`` over a generator that a live course
drives (the same object) takes it over once its construction completes: the older
course raises ``RuntimeError`` at its next ``step``, ``reset``, ``regenerate``,
``enqueue_generate`` or ``generate`` (its components would read a source the new course
regenerates); a construction that raises (a warning turned into an error included)
takes nothing over. Rebuilding (``course = Course(gen, ...)`` run twice) works. Calling
``course.generator.generate()`` yourself regenerates the source without refreshing the
components: regenerate through the course.

A worked example: ``CurvatureLookahead``
----------------------------------------

A controller wants the centerline curvature a few points past the car. This component,
explained line by line in :doc:`/contributing/writing-a-component`, imports only public
modules (``track_gen.tools``, ``track_gen.results``, ``track_gen.localize``) and Warp:

.. code-block:: python

   # acme_tools/lookahead.py
   import dataclasses
   import types

   import warp as wp

   from track_gen.localize import curvature
   from track_gen.results import HasCenterline, HasValidity
   from track_gen.tools import Component, InputSpec


   @wp.kernel
   def _lookahead_k(kappa: wp.array(dtype=wp.float32), center: wp.array(dtype=wp.vec3f),
                    count: wp.array(dtype=wp.int32), S: int, ahead: int,
                    segment: wp.array(dtype=wp.int32), position: wp.array(dtype=wp.vec3f),
                    out_kappa: wp.array(dtype=wp.float32),
                    out_distance: wp.array(dtype=wp.float32)):
       e = wp.tid()
       m = count[e]
       if m > S:
           m = S
       seg = segment[e]
       if seg < 0 or m < 3:                 # no frame (invalid env, NaN position): NaN
           out_kappa[e] = wp.nan
           out_distance[e] = wp.nan
           return
       i = (seg + ahead) % m
       out_kappa[e] = kappa[e * S + i]
       out_distance[e] = wp.length(center[e * S + i] - position[e])


   @dataclasses.dataclass
   class Lookahead:
       kappa: wp.array                      # [E] float32, 1 / track units
       distance: wp.array                   # [E] float32, agent to that point


   class CurvatureLookahead(Component):
       """Centerline curvature ``ahead`` points past the agent, per env."""

       requires = (HasCenterline,)
       optional = (HasValidity,)
       inputs = types.MappingProxyType({"position": InputSpec(wp.vec3f, doc="the agent")})

       def __init__(self, ahead=6, *, window=2, localizer="localizer",
                    name="acme.lookahead"):
           self.ahead, self.window, self.name = int(ahead), int(window), name
           self.localizer = localizer
           self.after = (localizer,)        # reads that component's TrackFrame

       def attach(self, source, cc):
           lay = self.layout
           E, S, dev = lay.num_envs, lay.stride, lay.device
           self._source = source
           self._center, self._count = source.center, source.count
           self._frame = cc.component(self.localizer).result
           self._kappa = wp.zeros(E * S, dtype=wp.float32, device=dev)
           self._scratch = wp.zeros(E * S, dtype=wp.float32, device=dev)
           self.result = Lookahead(wp.zeros(E, dtype=wp.float32, device=dev),
                                   wp.zeros(E, dtype=wp.float32, device=dev))
           self._position = None
           self.on_generate(cc.ctx, None)   # the current batch's curvature (cold)
           wp.load_module(__name__, device=dev)

       def on_bind(self, position):
           self._position = position

       def on_generate(self, ctx, mask):    # per generation: no bound input read here
           curvature(self._source, self.window, ctx,
                     out=self._kappa, scratch=self._scratch)

       def step(self, ctx):                 # per step: reads position
           lay, r = self.layout, self.result
           ctx.launch(_lookahead_k, lay.num_envs,
                      [self._kappa, self._center, self._count, lay.stride,
                       self.ahead, self._frame.segment, self._position,
                       r.kappa, r.distance])
           return r

Plug it into a preset with a ``Localizer`` (the order of ``extra`` does not matter: the
course sorts it after the localizer it names in ``after``):

.. code-block:: python

   from track_gen import TrackGenConfig, tools
   from track_gen.course import Course

   from acme_tools.lookahead import CurvatureLookahead

   course = Course.track(TrackGenConfig(num_envs=E, device="cuda:0"),
                         checkpoint_spacing=0.6, collision="segments", seeds=5,
                         extra=[CurvatureLookahead(6), tools.Localizer()])
   list(course.components)            # ['checkpoints', 'progress', 'collision',
                                      #  'localizer', 'acme.lookahead']
   course.bind(position=pos, orientation=quat, half_extents=he)
   res = course.step()
   list(res.keys())                   # ['progress', 'collision', 'localizer', 'acme.lookahead']
   kappa = wp.to_torch(res["acme.lookahead"].kappa)   # NaN on an invalid env
   res.frame is res["localizer"]      # True: the 0.x field, on a track course too

``bind`` sends ``pos`` to ``Progress``, the ``Localizer`` and the lookahead (and, at one
box, to the collision's ``box_position``). The regeneration body refreshes the curvature
after the generator (``on_generate``) and resets the localizer and progress of the
regenerated envs; the lookahead keeps no per-env state, so its ``reset`` is the inherited
no-op. Check it with ``track_gen.testing.check_component(lambda cc:
CurvatureLookahead(6), upstream=[lambda cc: tools.Localizer()])`` (:ref:`kit-check-component`).

The component contract in a course
----------------------------------

Beyond the lifecycle contract of :doc:`/contributing/writing-a-component`, a component
in a course obeys these rows (``check_component`` enforces the first two):

.. list-table::
   :header-rows: 1
   :widths: 24 50 26

   * - Rule
     - Text
     - Enforced by
   * - ``on_generate`` / ``reset`` read no input
     - They read and write no bound input: the course's self-graph runs them keyed on
       ``(advance, mask)`` only, so an input read there would be baked at the recording,
       and the regenerated geometry would depend on the agent. Read inputs in ``step``.
       A respawn snapshot (``reset`` copying ``position``) belongs in ``step``: ``reset``
       arms a per-env flag that the next ``step`` consumes.
     - ``hot_path.buffers`` (CPU)
   * - ``on_generate`` / ``reset`` run in a conditional body
     - The masked regeneration runs them inside a CUDA conditional-node body (in the
       course's own graph and in your Warp-registered captures). They may emit only
       kernels, ``ctx.fill`` and device-to-device ``ctx.copy`` on ``ctx``'s stream: no
       copy to or from host or pinned memory, no event, no other stream. The course
       re-raises Warp's refusal as ``CaptureError`` naming its components.
     - ``capture.caller`` (CUDA, ``warp`` mode)
   * - ``reset`` reads only its own state
     - Only its own state and the mask, never another component's result or state:
       resets run in reverse order, before the components they read.
     - documented (the built-ins comply)
   * - ``StepResult`` entry
     - A component that overrides ``step`` is an entry (its ``result``); a
       per-generation one keeps the base ``step``.
     - ``result.stable`` (``step()`` returns ``result``)

A size curriculum
-----------------

The size curriculum that ``style_sampling``'s ``scale_range`` gave a ``CourseConfig``
course is a randomizer on the spec (``style_sampling`` is removed at 1.0; a config built
with ``scale_range=(a, b)`` maps to ``(a / scale, b / scale)``, with
``style_sampling=False``, the default):

.. code-block:: python

   from track_gen import TrackGenConfig, tools
   from track_gen.course import Course
   from track_gen.pipeline import legacy

   spec = legacy.to_pipeline_spec(TrackGenConfig(num_envs=E, device="cuda:0"))
   course = Course.track(spec.with_randomize({"generator.size": (1.0, 2.5)}),
                         checkpoint_spacing=0.5, seeds=0,
                         collision=tools.BandCollision("sdf", sdf_resolution=320))

At flat elevation the default checkpoint capacity does not depend on the size. The SDF
and height-field grids do: ``BandCollision`` and ``HeightFieldBake`` default to
``size_hi="auto"``, the largest size of the plan (``ComponentContext.bounds``: a
randomizer's construction ``hi`` on ``generator.size``, else the generator's construction
``size``), and warn at ``attach`` when the resolution does not cover it
(``ceil(128 * size_hi)``: 320 above; :ref:`collision-resolution-check`). The check reads
the construction range only: a curriculum that widens it later
(``course.generator.params["randomize.generator.size"].set("hi", 3.0)``) is not checked
again, so pass the curriculum's final ``hi`` explicitly
(``tools.BandCollision("sdf", sdf_resolution=384, size_hi=3.0)``). A ``CourseConfig``
course declares no ``size_hi`` and never warns.

.. _course-config-migration:

Migrating from ``CourseConfig``
-------------------------------

``Course(CourseConfig)`` keeps working through 0.x with no warning. To move to a preset:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``CourseConfig``
     - ``Course.track`` / ``Course.gates``
   * - ``mode="track"`` / ``"gates"``
     - the preset
   * - ``gen``
     - ``source`` (or a built generator)
   * - ``seeds`` (int / array)
     - ``seeds=`` (int / array). ``course.rng`` is ``None`` unless
       ``seeds=PerEnvSeededRNG(...)``: read ``course.seeds.seeds`` for logging. With a
       bound array and no rng, ``generate(seeds=)`` raises: pass
       ``seeds=PerEnvSeededRNG(seeds=arr, ...)`` to keep the 0.x write-in-place.
   * - ``checkpoint_spacing``, ``max_checkpoints``
     - the same names (``track``)
   * - ``collision`` (default ``None``)
     - ``collision=`` (default ``None``): ``"segments"``, ``"sdf"`` or a component
       named ``"collision"``
   * - ``sdf_resolution``, ``max_boxes`` (track)
     - ``collision=tools.BandCollision(method, max_boxes=k[, sdf_resolution=R])``, with
       ``method`` ``"segments"`` or ``"sdf"`` (the string form ``collision="segments"``
       means ``max_boxes=1``)
   * - ``heightfield_resolution``
     - ``extra=[tools.HeightFieldBake(R)]``
   * - ``post_radius``, ``max_boxes`` (gates)
     - ``collision=r``, or ``collision=tools.DiscCollision(r, max_boxes=k)``
   * - ``frame_collision``, ``agent_radius``, ``frame_thickness``, ``frame_depth``
     - ``collision=tools.FrameCollision(agent_radius, frame_thickness, frame_depth)``
   * - ``samples_per_gate``, ``localize_window``
     - the same names (``gates``)
   * - ``bind(position, orientation, half_extents, box_position)``
     - the same call; names no component declares are ignored. The errors are the
       components': a missing ``orientation`` / ``half_extents`` of a box or disc
       collision raises ``TypeError("collision: missing inputs [...]")`` (a config
       course: the 0.x ``RuntimeError``), and ``None`` means "not given"
       (``bind(None)`` raises ``TypeError("progress: missing inputs ['position']")``,
       a config course the 0.x ``RuntimeError`` / ``ValueError``)

Troubleshooting
---------------

- ``cudaErrorStreamCaptureUnjoined`` (the capture is invalidated) at
  ``regenerate(mask=...)``: a side stream forked in your capture is not joined yet. Join
  it first, or record the call inside ``track_gen.exec.unconditional()``.
- A graph that holds the masked regeneration cannot be added as a CUDA **child graph**
  (it holds a conditional node): record it inside ``exec.unconditional()``.
- ``CaptureError`` naming the components at ``regenerate(mask=...)``: one of them uses an
  operation a conditional body cannot hold (a host copy, an event, another stream) in
  ``on_generate`` or ``reset``; ``check_component`` (``capture.caller``) finds which.
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

Mixed batches
-------------

``pipeline.MixedBatch`` (several generators writing env slices of one result, driven as
one producer) is not shipped. The Producer protocol keeps it possible: two generators
writing ``view``\ s of one ``Track`` (``out=``), wrapped in an object with ``result``,
``layout`` and ``enqueue``, are driven by an unchanged ``Course``. A shipped version needs
a whole-batch ``SeedState``, merged ``fail_bits``, per-part capacities and input
namespacing.

API: :class:`track_gen.course.Course`, :class:`track_gen.course.StepResult`,
``track_gen.pipeline.Producer``, :mod:`track_gen.tools` (:doc:`/reference/api`).
