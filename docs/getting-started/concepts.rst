.. _concepts:

Concepts and terms
==================

TrackGen builds many tracks (or gate sequences) at once, one per simulated environment,
and then answers per-step questions about them inside your simulator. The whole library
follows one data flow:

.. code-block:: text

   seeds [E]
     │
     ▼
   generator ── a pipeline of stages: generate a loop, resample, relax, build the road
     │          band, check it (a spec, usually from a preset)
     ▼
   result ───── Track or GateSequence: flat device arrays, valid[E], fail_mask[E]
     │
     ▼
   course ───── components: checkpoints, progress, collision, localization, props, yours
     │
     ▼
   your sim ─── binds its buffers, calls step() / reset(mask) / regenerate(mask=...)

**The batch.** Everything is batched over a fixed number :term:`E` of :term:`envs <env>`.
Each env gets its own :term:`seed`, and its own :term:`epoch` that advances when it needs
a new track. The batch size, the device and the per-env capacity (:term:`N_max` points
per track, ``max_gates`` slots per gate sequence) are fixed when the generator is built,
so every buffer is allocated once and keeps its address: a CUDA graph recorded over it
stays valid.

**The generator.** A ``TrackGenerator`` or ``GateGenerator`` runs a pipeline of
:term:`stages <stage>` described by a :term:`spec`. A :term:`preset` builds the usual
spec from a few keywords; each stage takes typed :term:`Params`, and its
:term:`dynamic fields <dynamic field>` hold one value per env, which you can change
between generations. Stages pass data to each other through
:term:`channels <channel>`. One run of the pipeline over the batch is a
:term:`generation`; construction runs the first one.

**The result.** A generation writes a ``Track`` (a road: centerline, borders, frames)
or a ``GateSequence`` (gate poses) as flat device arrays, ``[E * N_max]`` per point
and ``[E]`` per env. Every env passes through the pipeline's checks: :term:`valid` says
whether it passed, and the :term:`fail bits` say which checks failed. An
:term:`invalid env` stays in the batch until you regenerate it.

**The course.** A :term:`course` drives one generator and its runtime
:term:`components <component>` for the whole batch: it binds your simulator's buffers
(agent poses), steps every component (progress events, collision contacts), respawns envs
on their course, and regenerates selected envs. All of it runs on the device, so you can
record a whole sim step into your own CUDA graph (a :term:`capture`); with
:term:`conditional nodes <conditional node>`, a masked regeneration with nothing to do is
skipped on the device.

The :doc:`quickstart` shows each step in code.

Glossary
--------

.. glossary::

   env
      One slot of the batch, indexed ``e = 0 .. E - 1``: one track or one gate sequence,
      with its own seed and epoch, and the course one agent of your simulator drives.
      Every per-env array has ``E`` entries.

   E
   num_envs
      The batch size: the number of envs a generator builds in every generation
      (``num_envs=`` in the presets and configs). It is fixed when the generator is built.

   N_max
      A ``Track``'s per-env point capacity (``n_max=`` in ``presets.track``, ``N_max`` in
      ``TrackGenConfig``; 384 by default): every per-point array has ``E * N_max`` rows,
      and env ``e`` uses the first ``count[e]``. A track that needs more points is cut
      short and marked invalid (fail bit ``truncation``): keep
      ``N_max >= max perimeter / spacing + 1``.

   spacing
      The distance between consecutive points of a track's centerline: every track is
      resampled to a constant spacing (by default ``0.6 * half_width``). With
      :term:`N_max`, it bounds the longest track a batch can hold.

   size
      A per-env multiplier of a course's extent in the plane (the generator's dynamic
      ``size`` field, 1 by default, from 0.1 to 10): at size 2 a course is twice as large,
      with the same road width. Larger sizes need a larger :term:`N_max`.

   seed
      A per-env ``int32`` that, with the env's epoch, determines its track or gate
      sequence. ``seeds=k`` gives env ``e`` the seed ``k + e``; a ``PerEnvSeededRNG`` or an
      ``[E]`` ``int32`` array sets them one by one.

   epoch
      A per-env counter (``gen.seeds.epoch``, 0 at construction). ``generate(advance=True)``
      and ``regenerate(advance=True)`` add one to every env, or to the envs of a mask, so
      they get a new track from the same seed. For a given generator, the same
      ``(seed, epoch)`` always rebuilds the same output.

   generation
      One run of a generator's pipeline over the whole batch: construction runs one, and
      so does every ``generate()`` / ``regenerate()`` call (a masked one changes only the
      masked envs). The result's ``generation`` field holds the epoch each env was built
      at.

   valid
      ``valid[e]`` (``[E]`` ``int32``, on every result and on a course's ``StepResult``)
      is 1 when env ``e`` passed every check of its pipeline, 0 when it did not (an
      invalid env).

   fail bits
      ``fail_mask[e]`` has one bit set per check env ``e`` failed (0 exactly when it is
      valid); ``result.fail_bits`` maps each check's name to its bit, for a track
      ``generator``, ``turning``, ``finite``, ``thickness``, ``w_floor``, ``border``,
      ``grade`` and ``truncation``.

   invalid env
      An env with ``valid[e] == 0``. By default its geometry stays in the result, but every
      runtime utility answers for it as for an env with no track (off the road, no
      checkpoints, frozen progress). Gate rewards, observations and episode ends on
      ``valid``, and give the env a new course with a masked regeneration
      (:ref:`invalid-envs`).

   preset
      A function that builds a standard configuration from keyword arguments:
      ``presets.track`` and ``presets.gates`` return a spec, ``Course.track`` and
      ``Course.gates`` build a course with the usual components.

   spec
   PipelineSpec
      The description, as data, of a generator's pipeline: the batch (``E``, device,
      ``N_max``, ...) and the ordered stages with their params. It is immutable; edit it
      into a new one (``replace``, ``insert_after``, ``remove``), save it with
      ``to_dict()``, and build it with ``TrackGenerator(spec)`` or ``GateGenerator(spec)``.

   stage
      One step of a pipeline, selected by its kind and name (for example the ``xpbd``
      stage of kind ``relax``) and configured by its Params. Plugins can provide their own
      generators and stages (:ref:`writing-a-generator`).

   Params
      A stage's typed parameters, a frozen dataclass per stage (for example the ``bezier``
      generator's ``BezierParams``), checked when the generator is built. Each field is
      static (fixed for the generator's lifetime) or dynamic.

   dynamic field
      A Params field the generator holds as a live ``[E]`` device array, one value per env
      (``gen.params[stage_id][field]``): set it from the host or from your own kernels, or
      draw it per env with ``randomize=``, and the next generation uses the new values
      without rebuilding anything (:ref:`curricula`). Changing a static field needs a new
      generator.

   channel
      A named device buffer that the stages of a pipeline pass to each other (for example
      ``cl.xy``, the centerline points while the track is being built). The pipeline
      allocates each channel once, when it is built.

   component
      A runtime piece that a course drives over the generator's result: checkpoints,
      progress, collision, localization, props, or your own
      (:ref:`writing-a-component`). It refreshes on every new course, steps with your
      simulator and resets on respawn.

   course
      A ``Course``: one generator plus its components, driven as one object for the whole
      batch (``bind``, ``step``, ``reset``, ``regenerate``); see :ref:`courses`. Also, in
      prose, the track or gate sequence one env's agent drives.

   capture
      Recording device work into a CUDA graph instead of running it (``wp.ScopedCapture``,
      ``torch.cuda.graph``), to replay it later with one launch. Generator and course calls
      made while their stream is capturing are recorded into your graph; building a
      generator or a course inside a capture raises an error. Warp also records and
      replays captures on the ``cpu`` device.

   conditional node
      A CUDA graph node that runs its body only when a value on the device says so. A
      masked regeneration recorded with one is skipped on the device when no env is
      masked; recorded without, it runs a full generation on every replay. It needs a
      capture Warp registered and CUDA 12.4 or newer in the driver and in Warp's build
      (``wp.is_conditional_graph_supported()``).
