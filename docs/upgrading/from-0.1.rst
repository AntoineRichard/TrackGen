.. _upgrading-0.2:

Upgrading from 0.1 to 0.2
=========================

.. note::

   This is the historical 0.1 to 0.2 guide. The current release has removed the
   compatibility APIs described below; continue with :doc:`from-0.2` after this guide.
   Historical examples of removed APIs either show the current error explicitly or
   are marked non-executed. The working replacement examples run against current code.

This page is for code written against TrackGen 0.1 (the ``v0.1.0`` release). It answers
three questions in order: will my code still run, will my results change without an
error, and do I have to retrain or rebaseline anything.

Most 0.1 code runs unchanged. ``TrackGenConfig``, ``GateGenConfig``, ``TrackGenerator``,
``GateGenerator`` and ``Course(CourseConfig(...))`` keep working in 0.2, and every
default generator except ``repulsive`` produces the same bytes as in 0.1. What needs your
attention is a short list of changes that alter results without raising, led by the way
:term:`invalid envs <invalid env>` now read. The 0.1 documentation is in the
`v0.1.0 documentation sources <https://github.com/AntoineRichard/TrackGen/tree/v0.1.0/docs>`_,
and the `changelog <https://github.com/AntoineRichard/TrackGen/blob/main/CHANGELOG.md>`_
has the full details per release. Each item below is self-contained.

The examples on this page share these imports; :term:`E` is the number of
:term:`envs <env>` in a batch:

.. snippet: group=common

.. code-block:: python

   import numpy as np
   import warp as wp

   from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG,
                          TrackGenConfig, TrackGenerator)
   from track_gen.course import Course, CourseConfig
   from track_gen.pipeline import legacy, presets

   E, device = 64, "cuda:0"  # or "cpu"

.. _upgrading-procedure:

How to upgrade
--------------

1. Raise your Warp pin to ``warp-lang>=1.18,<2``. On a GPU, Warp's PyPI wheels need an
   NVIDIA driver R580 or newer and an sm_75 or newer GPU; the ``cpu`` device runs
   anywhere. Skipping masked work on the device needs CUDA 12.4 or newer (driver and
   toolkit). :ref:`upgrading-warp` and :ref:`installation` have the details.
2. Run your tests and scripts with ``python -W error::DeprecationWarning`` (or at least
   ``-W default::DeprecationWarning``). The library attributes its deprecation warnings
   to the line of your code that triggers them, and Python hides a
   ``DeprecationWarning`` attributed to any module other than ``__main__``, so a gym
   environment or a training package never shows them by default. Fix each one with the
   items below.
3. Search your code for the patterns in the "Find it" column of the
   :ref:`triage table <upgrading-triage>`, starting with the rows marked **SILENT**.
4. Gate every reward, observation and termination on the :term:`valid` flag
   (:ref:`rl-valid-recipe`), and measure your invalid fraction right after
   construction: ``(course.result.valid.numpy() == 0).mean()``, or
   ``(track.valid.numpy() == 0).mean()`` on a generator's result. Expect 0 to 3 % for
   tracks at the settings of the 0.1 examples, and up to 94 % for gates at the 0.1
   quickstart's ``gate_width=0.4``.
5. Rebaseline stored datasets and evaluation sets built with ``repulsive``, with thin
   tracks that 0.1 truncated, or with wide gate bars, and retrain policies whose rewards
   or terminations read the ``sdf`` collision query.
6. Measure memory and throughput again: construction now runs a :term:`generation`, a
   :term:`course` holds more checkpoint slots by default, and every generator keeps a
   copy of its result (:ref:`upgrading-memory`).

.. _upgrading-triage:

Triage: what changed and who is affected
----------------------------------------

Rows are sorted by symptom: **SILENT** (results change and nothing raises or warns)
first, then errors, warnings, and changes that need no action.

.. list-table::
   :header-rows: 1
   :widths: 9 33 16 20 22

   * - Symptom
     - Change
     - Who is affected
     - Find it in your code
     - What to do
   * - **SILENT**
     - Gates at the 0.1 quickstart's ``gate_width=0.4``: about 94 % of a ``bezier``
       batch fails its bar-crossing check, on 0.1 too, but 0.2 freezes those envs instead
       of tracking them
     - gate courses and gate datasets
     - ``gate_width=0.4``, ``gate_width = 0.4``
     - narrower bars or a larger ``scale``: :ref:`upgrading-gates`
   * - **SILENT**
     - Every runtime utility reads an invalid env as empty: ``oob = 1``, NaN geometry,
       frozen progress, ``dist_to_next = 0``, no checkpoints, no props
     - every ``Course`` and utility user
     - ``.oob``, ``dist_to_next``, ``isnan(``, ``res.contacts``, ``res.events``
     - gate on ``valid``: :ref:`upgrading-invalid-envs`
   * - **SILENT**
     - Tracks truncated at ``N_max`` are invalid: the 0.1 README's ``half_width=0.03``
       with the default ``N_max`` goes from 0 to about 3 % invalid envs
     - track configs with ``half_width`` below 0.1
     - ``half_width=`` below 0.1 (for example ``half_width=0.03``), ``N_max``
     - raise ``N_max``: :ref:`upgrading-truncation`
   * - **SILENT**
     - The generator copies its config: changing the config object after construction
       is ignored
     - config-mutation curricula
     - ``cfg.<field> =`` after construction
     - set :term:`dynamic fields <dynamic field>`, rebuild for static ones:
       :ref:`upgrading-config-snapshot`
   * - **SILENT**
     - The ``sdf`` collision query is exact on the box outline: ``oob`` and
       ``distance`` change near the edges
     - ``sdf`` users
     - ``"sdf"``
     - retrain: :ref:`upgrading-sdf`
   * - **SILENT**
     - ``repulsive`` envs that converge early keep their shape
     - ``repulsive`` datasets
     - ``"repulsive"``
     - regenerate datasets: :ref:`upgrading-repulsive`
   * - **SILENT**
     - The default checkpoint capacity comes from the longest possible track (70 at the
       defaults instead of 15 to 23; the same or fewer for thin roads), so checkpoint
       arrays change shape
     - fixed-width observations
     - ``max_checkpoints``, ``checkpoints.``
     - pass ``max_checkpoints``: :ref:`upgrading-checkpoint-capacity`
   * - **SILENT**
     - No host sync after library calls
     - consumers on another stream, benchmarks
     - ``torch.cuda.Stream``, ``wp.Stream``, ``stream=``, timers
     - synchronize explicitly: :ref:`upgrading-host-syncs`
   * - **SILENT**
     - Library calls inside ``torch.cuda.graph(g)`` are recorded into ``g``
     - torch CUDA graph users
     - ``torch.cuda.graph``, ``CUDAGraph``
     - :ref:`upgrading-capture`
   * - **SILENT**
     - The list of available generators includes ``external``
     - per-run generator sampling
     - ``available(``, ``GENERATORS``
     - filter on ``seeded``: :ref:`upgrading-available-generators`
   * - **SILENT**
     - ``set_seeds_warp`` followed by a partial ``ids=`` sample draws differently
     - direct ``PerEnvSeededRNG`` users
     - ``set_seeds_warp``, ``ids=``
     - :ref:`upgrading-seeds`
   * - **SILENT**
     - More memory, and construction costs one generation (about two per ``Course``)
     - large ``E``
     - ``num_envs=``
     - :ref:`upgrading-memory`
   * - error
     - Warp 1.18 or newer
     - everyone
     - ``warp-lang`` pins
     - :ref:`upgrading-warp`
   * - error
     - Constructing a generator or a course inside a CUDA graph :term:`capture` raises;
       the rng must be on the generator's device
     - capture and multi-device users
     - ``ScopedCapture``, ``capture_begin``, ``PerEnvSeededRNG(``
     - construct first: :ref:`upgrading-generate-semantics`
   * - error
     - Newly rejected knob values (``rad > 1``, polar jitters above their clamp, bools)
     - tuned configs
     - ``polar_radial_jitter``, ``rad=``, ``hull_displacement``
     - :ref:`upgrading-bounds`
   * - error
     - New result fields (mappings, a ``Layout``, ``[E]`` arrays next to per-point ones)
       break generic field iteration
     - converters and loggers
     - ``dataclasses.fields(``, ``asdict(``
     - :ref:`upgrading-result-fields`
   * - error
     - Utilities: ``clone()`` refuses captures, construction errors are ``TypeError``,
       sources are bound at construction, ``CourseLine.track`` warns
     - direct utility users
     - ``.clone(``, ``except ValueError``, ``course_line.track``
     - :ref:`upgrading-utilities`
   * - error / **SILENT**
     - ``_build_subtools`` and ``_apply_bind`` overrides no longer run; ``_bind_args``
       holds only the given arrays; new read-only names
     - ``Course``, utility or generator subclasses and wrappers
     - ``_build_subtools``, ``_apply_bind``, ``_bind_args``, ``self.result =``,
       ``self.rng =``
     - :ref:`upgrading-subclassing`
   * - error
     - Removed private kernels and ``_pipe``; importing a generator module no longer
       registers it
     - 0.1 plugin authors
     - ``_src.``, ``_pipe``, ``_k(``
     - port to ``track_gen.ops``: :ref:`upgrading-plugin-private-names`
   * - error / warning
     - ``GENERATORS`` is read-only and ``registry.add`` refuses a duplicate name; the
       ``_src`` ``register()`` still works and warns when it replaces a name; a
       ``capturable=False`` generator called while your graph records raises
       ``NotCapturableError``
     - plugin authors
     - ``GENERATORS[``, ``_src.generator_registry``, ``register(``,
       ``capturable=False``
     - :ref:`upgrading-registry`
   * - warning
     - ``set_capturing`` is a no-op
     - capture users
     - ``set_capturing``
     - delete the calls: :ref:`upgrading-set-capturing`
   * - warning
     - ``style_sampling`` is deprecated
     - ``bezier`` configs
     - ``style_sampling``, ``scale_range``, ``rad_range``
     - ``randomize=``: :ref:`upgrading-style-sampling`
   * - warning
     - Built-in generators are :term:`stages <stage>` (``.params``),
       ``GateGeneratorSpec`` and the ``_src`` helpers are deprecated, plugin knobs move to
       ``Params``
     - plugin authors
     - ``.params``, ``GateGeneratorSpec``, ``.generate(``, ``TrackGenConfig)``
       subclasses
     - :ref:`upgrading-plugins`
   * - none
     - Flat config fields keep working; ``Params`` and :term:`presets <preset>` are the
       new forms
     - everyone
     - ``TrackGenConfig(``
     - optional: :ref:`upgrading-flat-config`
   * - none
     - ``Course`` presets; the first ``course.generate()`` is no longer needed
     - ``Course`` users
     - ``CourseConfig(``, ``course.generate()``
     - drop that call: :ref:`upgrading-course`

Silent changes
--------------

These change what your code computes without raising or warning. Read every one that
matches your code.

.. _upgrading-gates:

Gate courses at wide bar widths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: gate courses and gate datasets built with wide bars, such as the 0.1
   quickstart's ``gate_width=0.4``
:Find it: ``gate_width=0.4``, ``gate_width = 0.4``, any ``gate_width`` above 0.1

Every gate pipeline checks each env's bars for crossings, on 0.1 as on 0.2. With the 0.1
quickstart's settings (``bezier``, ``gate_width=0.4``, ``scale=1``) about 94 % of a batch
fails that check: 60 of 64 envs, measured on both versions. The fraction is not new; what
happens next is. 0.1 kept tracking progress and post hits on those failed courses. 0.2
treats each one as an invalid env (:ref:`next item <upgrading-invalid-envs>`): its
progress is frozen, its distance to the next gate reads 0 and it hits no post, so it is a
slot your agents cannot train on until it is regenerated.

Use bars of 0.1 or narrower, or a larger ``scale`` (the gates spread out, so wider bars
fit), and check the fraction after construction. :ref:`gate-width-invalid-fraction` lists
the measured fraction per generator, bar width and scale. That table uses the default
``ccw`` ordering; with the 0.1 quickstart's ``gate_ordering="random_pairs"`` the
fractions are higher (12.5 % at 0.1 for ``bezier``, 94 % at 0.4, as measured below), so
use 0.05 there (no invalid env in the same batch) or measure your own setting.

The 0.1 quickstart still runs, with no warning. The print comments below record
measurements from 0.2; the current :ref:`hashed seed mapping <upgrading-0.3-rng>`
changes the sampled courses and their exact invalid fraction.

.. snippet: group=gates requires=upgrading/from-0.1#common

.. code-block:: python

   config = GateGenConfig(generator="bezier", gate_ordering="random_pairs", num_envs=E,
                          max_gates=32, gate_width=0.4, gate_radius=0.025,
                          gate_solve_iters=8, device=device)
   rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
   gates = GateGenerator(config, rng).generate()
   print((gates.valid.numpy() == 0).mean())   # historical 0.2: 0.94 (60 of 64)

.. snippet: after

   wide_invalid = (gates.valid.numpy() == 0).sum()
   assert 0 < wide_invalid <= E

With narrower bars:

.. snippet: group=gates

.. code-block:: python

   config = GateGenConfig(generator="bezier", gate_ordering="random_pairs", num_envs=E,
                          max_gates=32, gate_width=0.1, gate_radius=0.025,
                          gate_solve_iters=8, device=device)
   gates = GateGenerator(config, rng).generate()
   print((gates.valid.numpy() == 0).mean())   # historical 0.2: 0.125 (8 of 64)

.. snippet: after

   assert (gates.valid.numpy() == 0).sum() < wide_invalid

Two clarifications, with the same bytes as 0.1: ``gate_width`` is the full bar width (the
0.1 docs called it a half-width; the code always used the full width), and the default
``gate_ordering`` is now ``None``, which resolves to ``"ccw"`` for every built-in
generator (a gate generator that accepts only ``"raw"`` now builds without naming it).

.. _upgrading-invalid-envs:

Invalid envs read as poisoned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: every ``Course`` user, and every direct use of the runtime utilities on a
   result that has invalid envs
:Find it: ``.oob``, ``dist_to_next``, ``isnan(``, ``res.contacts``, ``res.events``,
   ``course.reset(``

An env is invalid when its result has ``valid[e] == 0``: the track or the gates failed
one of the generator's checks (truncated, self-crossing, too thin, crossing bars). Its
:term:`fail bits` say which. On 0.1 every utility kept computing on the failed geometry.
On 0.2 every utility writes, for an invalid env, what it writes for an env with no track
at all. ``TrackGenConfig`` and ``GateGenConfig`` keep the failed geometry in the result
(as 0.1 did), so every config-built course, and every utility you run on a config-built
generator's result, sees this change.

.. list-table:: What an invalid env reads
   :header-rows: 1
   :widths: 30 30 40

   * - Field
     - 0.1
     - 0.2
   * - ``res.contacts`` in track mode (``BoxContact``; ``CollisionChecker``, both
       methods)
     - the answer on the failed geometry (usually ``oob = 0``)
     - ``oob = 1``, NaN ``distance`` / ``nearest`` / ``normal``, ``boundary = -1``
   * - The SDF bake (``collision="sdf"``)
     - the failed track's field and box
     - NaN texels, boundary id ``-1``, box ``(-1, -1)`` to ``(1, 1)``
   * - ``res.contacts`` in gates mode with posts (``DiscContact``)
     - hits on the invalid gates' posts
     - ``hit = 0``, ``disc = -1``, ``depth = 0``, NaN ``nearest``
   * - ``res.events`` (``ProgressEvents``)
     - progress tracked on the failed course, finite ``dist_to_next``
     - frozen: ``next_checkpoint``, ``laps`` and ``progress`` keep their values,
       ``passed = 0``, ``checkpoint_passed = -1``, ``wrong_way = 0``,
       ``wrong_checkpoint = -1``, ``dist_to_next = 0``
   * - A ``TrackLocalizer`` you build on a ``Track``
     - the projection on the failed centerline
     - NaN ``s`` / ``n`` / ``n_up``, ``segment = -1``
   * - ``course.checkpoints`` in track mode (``CheckpointSampler``)
     - the failed track's checkpoints
     - ``count = 0``, NaN slots, ``valid = 0``
   * - ``course.heightfield`` (``HeightFieldBaker.bake()``)
     - NaN heights, the failed track's box
     - NaN heights, box ``(-1, -1)`` to ``(1, 1)``
   * - ``PropSampler``
     - the failed track's props
     - ``count = 0``, NaN poses, NaN ``step``, ``truncated = 0``
   * - ``localize.curvature`` / ``speed_profile``
     - values
     - NaN rows

Unchanged from 0.1: frame contacts in gates mode (``FrameContact``: ``hit = 0``,
``depth = 0``), the course line (``count = 0``, NaN), so ``res.frame`` in gates mode
(NaN ``s`` / ``n`` / ``n_up``, ``segment = -1``), and ``CheckpointSet.from_gates`` (a
view of the gates). Each row was measured on both versions with the same
:term:`seeds <seed>`. When an env becomes valid again, its progress resumes without a
spurious checkpoint crossing.

``valid`` is the signal: ``StepResult.valid`` and ``CheckpointSet.valid`` are new.
``dist_to_next`` reads ``0`` for an invalid env (NaN only for a valid env of a
hand-built checkpoint set without checkpoints), so code that detected a broken env with
``isnan(dist_to_next)`` must read ``valid`` instead.

What breaks in an RL loop: a loop that penalises ``oob``, or ends episodes on it, without
gating on ``valid`` penalises or ends every invalid env at every step;
``course.reset(done)`` respawns them on the same invalid course forever; and a NaN in one
env's observation makes a batch mean NaN. Gate every reward, observation and termination
on ``valid`` with a select (never ``reward * valid``: ``NaN * 0`` is NaN), and give the
invalid envs a new course with a masked regeneration, new in 0.2
(:ref:`rl-valid-recipe`). A host version of the gating:

.. snippet: group=invalid requires=upgrading/from-0.1#common

.. code-block:: python

   config = TrackGenConfig(num_envs=E, half_width=0.03, device=device)   # the 0.1 README
   course = Course(CourseConfig(mode="track", gen=config, seeds=0, checkpoint_spacing=0.6,
                                collision="segments"))
   print((course.result.valid.numpy() == 0).mean())   # the invalid fraction

   # your sim's car boxes, one per env
   position = wp.zeros(E, dtype=wp.vec3f, device=device)
   orientation = wp.full(E, wp.quatf(0.0, 0.0, 0.0, 1.0), dtype=wp.quatf, device=device)
   half_extents = wp.full(E, wp.vec2f(0.02, 0.01), dtype=wp.vec2f, device=device)
   course.bind(position, orientation, half_extents)

   res = course.step()
   valid = res.valid.numpy() != 0             # read it before any regeneration
   oob = res.contacts.oob.numpy() != 0
   reward = np.where(valid, np.where(oob, -1.0, 0.1), 0.0)
   done = oob | ~valid

.. snippet: after

   # the invalid fraction above is data-dependent (one env of 64 here), so check the
   # reading where it certainly applies: at N_max=64 every track is truncated (the
   # shortest needs about 140 points at this spacing), so every env is invalid
   assert oob[~valid].all() and (reward[~valid] == 0.0).all() and done[~valid].all()
   config = TrackGenConfig(num_envs=E, half_width=0.03, N_max=64, device=device)
   course = Course(CourseConfig(mode="track", gen=config, seeds=0, checkpoint_spacing=0.6,
                                collision="segments"))
   course.bind(position, orientation, half_extents)
   res = course.step()
   valid = res.valid.numpy() != 0
   oob = res.contacts.oob.numpy() != 0
   reward = np.where(valid, np.where(oob, -1.0, 0.1), 0.0)
   done = oob | ~valid
   assert (~valid).any() and (course.result.truncated.numpy()[~valid] == 1).all()
   assert oob[~valid].all() and (reward[~valid] == 0.0).all() and done[~valid].all()

Combined with :ref:`truncation <upgrading-truncation>`, the settings of the 0.1 README
now leave about 3 % of the envs invalid, and each of them reads ``oob = 1`` with frozen
progress at every step. :ref:`invalid-envs` explains what ``valid == 0`` means and how to
lower the fraction; :ref:`utilities-invalid-envs` lists what every utility reads.

.. _upgrading-truncation:

Truncated tracks are invalid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: track configs whose tracks need more than ``N_max`` points, typically a
   ``half_width`` below the default 0.1 with the default ``N_max = 384``
:Find it: ``half_width=`` below 0.1 (for example ``half_width=0.03``), ``N_max``,
   ``n_max``

A track has one point every :term:`spacing` along its length (``spacing`` defaults to
``0.6 * half_width``), up to :term:`N_max` points. On 0.1 a track longer than that was cut
at ``N_max`` and still shipped with ``valid = 1``. On 0.2 it fails its check:
``track.truncated[e]`` is 1 and ``valid[e]`` is 0. The 0.1 README's
``TrackGenConfig(num_envs=E, half_width=0.03)`` gave 0 % invalid envs on 0.1 and gives
about 3 % on 0.2 (8 of 256), every one of them truncated. At the default
``half_width=0.1`` nothing changes.

Size ``N_max >= max(perimeter) / spacing + 1``; at ``half_width=0.03``, ``N_max=512``
is enough. :ref:`tuning-size` has the sizing rule for larger courses.

.. snippet: group=truncation requires=upgrading/from-0.1#common

.. code-block:: python

   rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
   config = TrackGenConfig(num_envs=E, half_width=0.03, device=device)   # the 0.1 README
   track = TrackGenerator(config, rng).generate()
   print(track.truncated.numpy().sum())       # 1 of 64 envs: invalid on 0.2

   config = TrackGenConfig(num_envs=E, half_width=0.03, N_max=512, device=device)
   track = TrackGenerator(config, rng).generate()
   print(track.truncated.numpy().sum())       # 0

.. snippet: after

   assert track.truncated.numpy().sum() == 0

With a preset, the same setting is ``presets.track(..., half_width=0.03,
n_max=512)``.

.. _upgrading-config-snapshot:

The generator copies its config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: curricula and sweeps that change a config object after building the generator
:Find it: ``cfg.<field> =`` or ``config.<field> =`` after ``TrackGenerator(`` /
   ``GateGenerator(`` / ``Course(``

``TrackGenerator`` and ``GateGenerator`` copy their config at construction. On 0.1, where
the pipeline ran eagerly (on ``cpu``), changing a field of the config object between two
``generate()`` calls changed the next batch; on 0.2 it changes nothing, and no warning
says so:

.. snippet: group=snapshot requires=upgrading/from-0.1#common

.. code-block:: python

   config = TrackGenConfig(num_envs=E, half_width=0.05, device=device)
   gen = TrackGenerator(config)
   before = gen.generate().outer.numpy().copy()
   config.half_width = 0.1                    # 0.1: the next batch was wider
   after = gen.generate().outer.numpy()       # 0.2: the same batch

.. snippet: after

   assert np.array_equal(np.nan_to_num(before), np.nan_to_num(after))

Change a dynamic field through ``gen.params`` (its value is read on the device at
every generation), draw it per env with ``randomize=`` (:ref:`curricula`), and build a
new generator to change a static field such as ``half_width`` (:ref:`generating-params`):

.. snippet: group=snapshot

.. code-block:: python

   gen.params["generator"].set("rad", 0.3)    # dynamic: applies from the next generation
   track = gen.generate()

   import dataclasses
   # static: build a new generator (spacing=None: derive it again from half_width)
   gen = TrackGenerator(dataclasses.replace(config, half_width=0.1, spacing=None))

.. snippet: after

   assert gen.config.half_width == 0.1 and np.isclose(gen.config.spacing, 0.06)

``gen.config`` returns a copy of the config the generator was built with. ``spacing``
defaults to ``0.6 * half_width`` when the config is built, so a ``dataclasses.replace``
keeps the old value unless you reset it to ``None``: without ``spacing=None`` above, the
new generator would place points every 0.03 instead of 0.06, twice as many per track,
and truncate long tracks sooner (:ref:`upgrading-truncation`).

.. _upgrading-sdf:

The ``sdf`` collision query
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: ``CollisionChecker(..., method="sdf")`` and ``CourseConfig(collision="sdf")``
   users, and policies trained on them
:Find it: ``"sdf"``, ``method="sdf"``, ``collision="sdf"``

The ``sdf`` box query now takes the exact minimum of the interpolated distance field over
the box outline. 0.1 took the minimum of five samples (the centre and the four corners),
so it missed an edge that crosses a box side while the four corners stay on the road,
typically the inner edge at a tight bend: a car box reaching 10 mm into an inner bend read
+16 mm. ``oob`` and ``distance`` therefore change near the edges (``distance`` never
increases), and ``nearest`` and ``normal`` come from the point where the minimum is
reached. On the racing near-edge test set (100 worlds of 1024 car boxes, resolution 256),
agreement with the exact ``segments`` backend goes from 99.728 % to 99.949 %: misses drop
from 259 to 32, and the 20 false positives are unchanged. When the change was made, a
query cost two to three times as much per CUDA-graph replay; :ref:`collision-cost` has
the current measurements. A box whose corners are not finite now reads ``oob = 1`` with
NaN ``distance`` (0.1 reported the grid's first texel).

Retrain, or at least re-evaluate, policies whose rewards or terminations read ``oob`` or
``distance`` from the ``sdf`` backend. The ``segments`` backend is unchanged.
:ref:`collision-backends` compares the two.

.. _upgrading-repulsive:

The ``repulsive`` generator's output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: datasets, evaluation sets and fingerprints built with ``generator="repulsive"``
:Find it: ``"repulsive"``

``repulsive`` can now be recorded in a CUDA graph. Each env stops changing once it
converges, independently of the rest of the batch (0.1 ran a host-driven loop with one
early exit for the whole batch). Envs that converge before the loop ends keep the shape
they had at that point and no longer get the later periodic resamples, so their tracks
change; envs that never converge are byte-identical, and each env now depends only on
its own seed, not on the rest of the batch. Every other default generator gives the same
bytes as 0.1.

Cost: a generation on ``cuda`` outside the generator's own CUDA graph runs the full
iteration budget; on ``cpu`` an eager generation stops at the first check that finds
every env converged (with the defaults, no earlier than iteration 144 of 269); the
generator's own graph, and a capture with :term:`conditional nodes <conditional node>`,
skip the finished windows on the device.

Regenerate stored ``repulsive`` datasets and evaluation sets.

.. _upgrading-checkpoint-capacity:

The default checkpoint capacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: track courses with the default ``CourseConfig.max_checkpoints=None``, and
   code that sizes observations from the checkpoint arrays
:Find it: ``max_checkpoints``, ``course.checkpoints.``, ``.shape[0] // E``

0.1 sized a track course's checkpoint arrays from the batch of the first ``generate()``:
``ceil(1.5 * longest valid centerline / checkpoint_spacing)`` slots per env, 15 to 23 at
``E = 64`` and ``checkpoint_spacing=0.5`` depending on the generator. 0.2 sizes them from
the longest track the generator can produce: ``max(3, ceil(1.5 * N_max * spacing /
checkpoint_spacing))`` (plus the altitude range of an elevated track), which no longer
depends on the seeds or the batch. At the ``TrackGenConfig`` defaults that is 70 slots at
``checkpoint_spacing=0.5``, 58 at 0.6, 116 at 0.3 and 35 at 1.0.

- Every ``course.checkpoints`` array changes to ``[E * M]``: usually larger (70 slots
  instead of 21 for the default ``bezier`` at ``checkpoint_spacing=0.5``), but the same
  or smaller for thin roads with the default ``N_max`` (18 slots, where 0.1 gave 18 or
  19 depending on the seeds, at the 0.1 README's ``half_width=0.03`` and
  ``checkpoint_spacing=0.6``). Each slot takes 52 bytes per env, plus 8 bytes per env
  for the new ``valid`` and ``generation``: 28.5 MiB at ``E = 8192``, ``M = 70``.
- An env whose checkpoints fit the old capacity gets the same rows, count, progress and
  laps. An env the old capacity cut short (``course.checkpoint_sampler.truncated == 1``)
  now gets every checkpoint, so its progress and laps change.
- When the generator declares no bound (an elevated ``TrackGenConfig``, whose ``z_min``
  / ``z_max`` declare no range), the course reads the construction batch back, as 0.1
  read the first batch.
- A defaulted capacity that cuts a valid env of the construction batch short emits one
  ``UserWarning`` at construction. A course whose construction batch has no valid env
  no longer raises at flat elevation.
- ``CheckpointSampler(track, spacing)`` and ``PropSampler(track, spacing)`` built
  directly keep 0.1's sizing (they read the bound batch once at construction).

To keep 0.1's shape, pass the old capacity explicitly (print
``course.checkpoints.position.shape[0] // E`` under 0.1):

.. snippet: group=capacity requires=upgrading/from-0.1#common

.. code-block:: python

   config = TrackGenConfig(num_envs=E, device=device)
   course = Course(CourseConfig(mode="track", gen=config, checkpoint_spacing=0.5))
   print(course.checkpoint_sampler.max_checkpoints)   # 70

   course = Course(CourseConfig(mode="track", gen=config, checkpoint_spacing=0.5,
                                max_checkpoints=21))   # the shape 0.1 gave this batch
   print(course.checkpoints.position.shape[0] // E)   # 21

.. snippet: after

   assert course.checkpoint_sampler.max_checkpoints == 21

.. _upgrading-host-syncs:

No host sync after library calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: code that reads library outputs from another CUDA stream, and benchmarks
:Find it: ``torch.cuda.Stream``, ``wp.Stream``, ``stream=``, ``time.perf_counter()``
   around library calls

On ``cuda``, 0.1 called a device-wide ``wp.synchronize()`` after every utility call
(``query()``, ``update()``, ``reset()``, ``sample()``, ``bake()``, ``refresh()``),
``Course.step()`` / ``reset()`` and every ``generate()``. 0.2 only enqueues the work on
a stream, and ``generate()`` waits only for the stream it ran on.

- Unchanged: work on the same stream, ``wp.array.numpy()`` (a blocking copy), and torch
  on its legacy default stream see finished results.
- A consumer on another stream (a torch side stream, a simulator's own stream) must
  order itself after the library's work: ``wp.synchronize_device()``, or a recorded
  ``wp.Event`` it waits on.
- A benchmark must synchronize before reading its timer.
- ``TRACKGEN_DEBUG_SYNC=1`` makes every library call block on its stream again, so an
  asynchronous CUDA error surfaces at the call that caused it (:ref:`api-environment`).

.. _upgrading-capture:

Library calls inside a torch CUDA graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: code that records TrackGen calls into a CUDA graph, with torch or with Warp
:Find it: ``torch.cuda.graph``, ``CUDAGraph``, ``ScopedCapture``, ``capture_begin``,
   ``set_capturing``

On 0.1 the utilities and ``Course.step()`` / ``reset()`` could be recorded into a
``wp.ScopedCapture`` only after ``track_gen.set_capturing(True)``, which turned off their
host syncs. Inside ``torch.cuda.graph(g)``, every library call ran eagerly on Warp's
stream at capture time and was missing from ``g``. On 0.2 every generator, course and
utility call records into your graph with no flag: delete the ``set_capturing`` calls
(:ref:`upgrading-set-capturing`). A 0.1 loop that recorded into a ``wp.ScopedCapture``
under ``set_capturing(True)`` keeps working (the call only warns). What else changes:

- Calls made inside ``torch.cuda.graph(g)`` are recorded into ``g``: a replay runs them,
  and nothing runs at capture time.
- Under torch's capturing stream, the device-side skip of a masked regeneration (new in
  0.2) is recorded unconditionally, so it costs a full generation at every replay, unless
  you opt in with ``track_gen.interop.torch.registered(conditional=True)`` or
  ``track_gen.exec.conditional()``.
- A graph that contains conditional nodes (``repulsive``, a masked regeneration) cannot
  be used as a CUDA child graph: record it inside ``track_gen.exec.unconditional()``.
- ``step()`` and ``reset()`` recorded through a Warp-registered capture keep the course
  alive as long as the graph; a plain ``torch.cuda.graph`` needs
  ``track_gen.interop.torch.retain(g, course)``.
- ``course.generate(seeds=k)`` with an ``int`` records that value into the graph: every
  replay reseeds with ``k``. Pass a ``wp.array`` and rewrite it in place to vary the
  seeds between replays.

:ref:`capture-rules` has the rules for recording into your graph, and
:ref:`capture-troubleshooting` the errors and warnings and what to do about each.

.. _upgrading-available-generators:

``available()`` lists ``external``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: code that samples a generator name per run (domain randomization, sweeps)
:Find it: ``available(``, ``GENERATORS``

``registry.available("generator")``, and the private
``track_gen._src.generator_registry.available()`` that 0.1 code used, include
``external``, a generator that ingests centerlines you provide and draws nothing from the
seed: sampled from that list, it gives the same placeholder circles for every
seed. Keep the seed-dependent generators only:

.. snippet: group=available requires=upgrading/from-0.1#common

.. code-block:: python

   from track_gen.plugin import registry

   names = [n for n in registry.available("generator")
            if registry.describe("generator", n).seeded]
   print(names)   # ['bezier', 'checkpoint', 'hull', 'polar', 'repulsive', 'voronoi']

.. snippet: after

   assert "external" not in names and "bezier" in names

.. _upgrading-seeds:

Seeds and epochs
~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: code that reseeds a ``PerEnvSeededRNG`` or samples from it directly
:Find it: ``set_seeds_warp``, ``ids=``, ``PerEnvSeededRNG(``

- ``rng.set_seeds_warp(seeds, None)`` followed by a sample with explicit ``ids``: on 0.1
  the envs the sample did not select went back to their states from before the reseed;
  on 0.2 they follow the reseed, so their later draws differ. Generator outputs do not
  depend on these states and are unaffected.
- ``TrackGenerator(config)`` and ``GateGenerator(config)`` work without an rng: env ``e``
  then gets seed ``e``, as with ``PerEnvSeededRNG(seeds=0, ...)`` (0.1 required one).
- The rng must be on the generator's device (:ref:`upgrading-generate-semantics`).
- New in 0.2: ``generate(advance=True)`` gives each env a new course by advancing its
  :term:`epoch`. A reseed through the rng keeps the epochs, so after an advance it does
  not reproduce a fresh generator: call ``gen.seeds.reset_epoch()`` as well.
  ``course.generate(seeds=...)`` does both. :ref:`generating-seeds` explains seeds and
  epochs.

.. snippet: group=seeds requires=upgrading/from-0.1#common

.. code-block:: python

   config = TrackGenConfig(num_envs=E, device=device)
   gen = TrackGenerator(config, PerEnvSeededRNG(seeds=0, num_envs=E, device=device))
   gen.generate(advance=True)                 # every env moves to its next course

   new_seeds = wp.array(np.arange(100, 100 + E, dtype=np.int32), device=device)
   gen.rng.set_seeds_warp(new_seeds, None)
   gen.seeds.reset_epoch()                    # without it: the seeds' epoch-1 courses
   track = gen.generate()                     # what a fresh generator with new_seeds gives

.. snippet: after

   fresh = TrackGenerator(config, PerEnvSeededRNG(seeds=new_seeds, num_envs=E, device=device)).generate()
   assert np.array_equal(np.nan_to_num(track.center.numpy()), np.nan_to_num(fresh.center.numpy()))

.. _upgrading-memory:

Memory and construction cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Silent
:Affects: large batches, memory budgets, start-up time
:Find it: ``num_envs=``, ``N_max=``

- Every generator holds a copy of its result, so it can restore the envs a masked
  regeneration leaves alone: 68 bytes per point plus 24 bytes per env, about 107 MB at
  ``E = 4096``, ``N_max = 384``.
- The pipeline keeps some working buffers apart: ``E * (2 * N_max + 1)`` floats for the
  final resample, ``E * (N_max + 1)`` more for an elevated track.
- Checkpoint arrays change shape with the default capacity, usually growing
  (:ref:`upgrading-checkpoint-capacity`).
- Construction runs one generation of each generator (so its result holds a batch right
  away), a full iteration budget for ``repulsive``; a ``Course`` runs about two at
  construction. 0.1 deferred that work to the first ``generate()``.

Breaking changes
----------------

These raise an error, or change the result of code that relied on an internal detail.

.. _upgrading-warp:

Warp 1.18 or newer
~~~~~~~~~~~~~~~~~~

:Type: Breaking
:Affects: everyone
:Find it: ``warp-lang`` in ``pyproject.toml``, ``requirements*.txt``, lock files and
   container images

0.2 requires ``warp-lang>=1.18,<2`` (0.1 accepted ``warp-lang>=1.14``), so a pin below
1.18 no longer resolves. Warp 1.18's PyPI wheels need an NVIDIA driver R580 or newer and
an sm_75 or newer GPU for the ``cuda`` device; the ``cpu`` device runs anywhere. The
device-side skip of masked work needs CUDA 12.4 or newer, driver and toolkit. To check an
environment:

.. snippet: group=warp

.. code-block:: python

   import warp as wp
   import track_gen

   print(track_gen.__version__, wp.__version__)        # 0.2.x and 1.18 or newer
   print(wp.is_cuda_available() and wp.is_conditional_graph_supported())

.. snippet: after

   assert tuple(int(p) for p in wp.__version__.split(".")[:2]) >= (1, 18)

.. _upgrading-generate-semantics:

Construction and ``generate()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Breaking
:Affects: code that builds generators or courses inside a CUDA graph capture, that
   times construction, or that gives a generator an rng on another device
:Find it: ``ScopedCapture``, ``capture_begin``, ``torch.cuda.graph``,
   ``PerEnvSeededRNG(``

On 0.1, construction was cheap and the first ``generate()`` ran three warm-ups, recorded
the generator's own CUDA graph and blocked with a device-wide ``wp.synchronize()``. On
0.2, construction runs one generation eagerly (it loads every kernel and leaves the batch
for the current seeds in the result) and blocks until that generation has finished on
the generator's stream; the first ``generate()`` on ``cuda`` then records the
generator's own graph and launches it. The same code still runs:

.. snippet: group=semantics requires=upgrading/from-0.1#common

.. code-block:: python

   config = TrackGenConfig(num_envs=E, device=device)
   rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
   gen = TrackGenerator(config, rng)   # 0.1: cheap; 0.2: one generation, then waits for it
   track = gen.generate()              # 0.1: warm-up, graph, device-wide sync

``generate()`` now waits only for its own stream, and two calls never wait:

.. snippet: group=semantics

.. code-block:: python

   track = gen.generate()              # waits for its own stream only
   track = gen.generate(sync=False)    # returns at once; results are stream-ordered
   gen.enqueue()                       # never waits

.. snippet: after

   assert track is gen.enqueue()

Constructing a generator or a course inside a capture raises ``CaptureError`` before it
allocates anything (construction allocates and runs a real generation). Construct it
first, then record your graph:

.. snippet: group=semantics expect-error=CaptureError

.. code-block:: python

   with wp.ScopedCapture(device=device):
       TrackGenerator(config, rng)     # raises CaptureError: construct before capturing

The rng must be on the generator's device: a ``PerEnvSeededRNG`` on another device
(a ``cuda`` rng for a ``cpu`` generator, which 0.1 accepted) raises ``ValueError`` at
construction. :ref:`generating-calls` describes the calls and their streams.

.. _upgrading-bounds:

Newly rejected values
~~~~~~~~~~~~~~~~~~~~~

:Type: Breaking
:Affects: tuned or swept configs with values outside the documented ranges
:Find it: ``rad=``, ``handle_clamp_frac=``, ``hull_displacement=``,
   ``polar_radial_jitter=``, ``polar_angular_jitter=``, ``voronoi_radial_variation=``,
   ``voronoi_angular_jitter=``, ``checkpoint_angle_jitter=``, ``checkpoint_turn_rate=``

Building a generator now raises ``ValueError`` for values 0.1 accepted:

- ``rad`` outside ``[0, 1]``, ``handle_clamp_frac`` below 0, ``hull_displacement``
  outside ``[0, 1]``;
- ``polar_radial_jitter`` outside ``[0, 0.85]`` and ``polar_angular_jitter`` outside
  ``[0, 0.45]``. 0.1 clamped them to these ranges, so set the bound itself for exactly
  the old output (``polar_radial_jitter=0.85`` for ``1.0``); the error says so;
- ``voronoi_radial_variation`` above 0.85 (no effect beyond it), ``voronoi_angular_jitter``
  above pi, ``checkpoint_angle_jitter`` outside ``[0, 1]``, ``checkpoint_turn_rate``
  outside ``[0, pi]``;
- a ``bool`` for any of these knobs (``rad=True`` was accepted).

.. snippet: group=bounds requires=upgrading/from-0.1#common expect-error=ValueError

.. code-block:: python

   TrackGenerator(TrackGenConfig(generator="polar", polar_radial_jitter=1.0,
                                 num_envs=E, device=device))
   # ValueError: polar_radial_jitter must be in [0, 0.85], got 1.0 (...: set
   # polar_radial_jitter=0.85 for the same output)

.. snippet: group=bounds

.. code-block:: python

   gen = TrackGenerator(TrackGenConfig(generator="polar", polar_radial_jitter=0.85,
                                       num_envs=E, device=device))   # 0.1's output for 1.0

A value that converts to a float is still accepted (a numeric string from YAML, a 0-d
array, a torch scalar). :ref:`builtin-generator-params` lists every knob's range.

.. _upgrading-result-fields:

New result fields
~~~~~~~~~~~~~~~~~

:Type: Breaking
:Affects: converters, loggers and serializers that iterate over the fields of a result
:Find it: ``dataclasses.fields(``, ``asdict(``, ``astuple(``, ``vars(res)``, ``**res``

The result dataclasses gain fields, after the 0.1 ones (positional construction is
unchanged):

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Class
     - New fields
   * - ``Track``
     - ``truncated``, ``generation``, ``fail_mask`` (``[E]`` ``int32`` arrays),
       ``half_width`` (``[E * N_max]`` ``float32``), ``fail_bits`` (a read-only mapping
       from check name to bit), ``attributes`` (a read-only mapping of extra per-point
       arrays), ``layout`` (a ``Layout``: ``num_envs``, ``stride``, ``device``,
       ``closed``)
   * - ``GateSequence``
     - ``generation``, ``fail_mask``, ``fail_bits``, ``attributes``, ``layout``
   * - ``StepResult``
     - ``valid`` (the course result's ``[E]`` ``valid`` array itself)
   * - ``CheckpointSet``
     - ``valid``, ``generation`` (``[E]``, next to the ``[E * M]`` fields)

Code that converts every field with ``.numpy()``, recurses into every ``StepResult``
field as a dataclass, or reshapes every ``CheckpointSet`` field to ``(E, M, ...)`` now
fails or mixes shapes. Keep the arrays, skip ``None``, and reshape by length:

.. snippet: group=fields requires=upgrading/from-0.1#common

.. code-block:: python

   import dataclasses

   track = TrackGenerator(TrackGenConfig(num_envs=E, device=device)).generate()
   arrays = {f.name: getattr(track, f.name) for f in dataclasses.fields(track)
             if isinstance(getattr(track, f.name), wp.array)}
   per_env = {k: a.numpy() for k, a in arrays.items() if a.shape[0] == E}
   per_point = {k: a.numpy().reshape(E, -1, *a.numpy().shape[1:])
                for k, a in arrays.items() if a.shape[0] != E}

.. snippet: after

   assert "fail_bits" not in arrays and "layout" not in arrays
   assert "valid" in per_env and per_point["center"].shape == (E, track.layout.stride, 3)

- ``StepResult.valid`` is the generator's state, not a step output: never write into it,
  and leave it out of code that zeroes or poisons step outputs. It is live: after a
  regeneration in the same iteration it describes the new courses, so read it before
  regenerating, or ``clone()`` the result.
- ``StepResult`` also gains read-only access by :term:`component` name
  (``res["progress"] is res.events``). Its dataclass behaviour is unchanged (fields,
  ``asdict``, ``==``, ``repr``, ``iter(res)`` still raises), but code that probes for a
  mapping now finds one: ``**res`` and ``dict.update(res)`` succeed,
  ``hasattr(res, "__iter__")`` is ``True`` and ``vars(res)`` gains ``_items``, so a
  serializer that sniffs these takes another path.
- ``track_gen.testing.capture.result_arrays(res)`` lists ``valid``.

:ref:`results` describes every field of the results.

.. _upgrading-utilities:

Runtime utilities
~~~~~~~~~~~~~~~~~

:Type: Breaking
:Affects: code that uses ``CollisionChecker``, ``TrackLocalizer``, ``CheckpointSampler``,
   ``PropSampler``, ``ProgressTracker``, ``HeightFieldBaker`` or the course line
   directly, and code that clones results
:Find it: ``.clone(``, ``except ValueError``, ``.track =``, ``.center =``,
   ``course_line.track``, ``bind_inputs(``

- ``clone()`` of a ``Track`` or ``GateSequence`` checks the result (a malformed
  hand-built one raises ``ValueError``) and blocks the host until the copy is complete.
  Inside a capture it raises ``CaptureError``: allocate a result once with
  ``Track.alloc(track.layout)`` and ``copy_from`` into it instead (below).
- A utility binds its source's arrays at construction. Assigning a new array to a member
  of the source later (``track.center = other``) is not seen, where 0.1 read the member
  again at every call: write into the arrays, or build a new utility.
- Construction errors changed: a missing member raises ``TypeError`` naming it
  (``CollisionChecker needs a HasBand source (...): Centerline has no 'outer', 'inner'``),
  a mismatched one a ``ValueError`` prefixed by the utility and the member, where 0.1
  inferred the layout from one field and raised ``ValueError("track batch layout
  invalid: ...")``. Code that catches construction errors should catch both types.
- A ``valid`` or ``generation`` member of the wrong shape, dtype or device on a
  hand-built source raises at construction (0.1 ignored it).
- ``course.course_line.track`` is a ``Centerline`` and emits a ``DeprecationWarning``:
  read ``course.course_line.centerline``. Its ``outer``, ``inner``, ``normal`` and
  ``winding`` (NaN or zero placeholders on 0.1) raise ``AttributeError``.
- Replacing a bound array after a call was recorded into a graph emits
  ``track_gen.exec.RebindAfterCaptureWarning`` (a ``UserWarning``): the graph keeps
  reading the old arrays, so record it again. A test suite run with ``-W error`` sees it.
- When you adopt ``generate(advance=True)`` (new in 0.2), ``ProgressTracker`` and
  ``TrackLocalizer`` reset an env's state by themselves once its course changes.

Copying a result inside a graph:

.. snippet: group=utilities requires=upgrading/from-0.1#common

.. code-block:: python

   from track_gen import Track

   gen = TrackGenerator(TrackGenConfig(num_envs=E, device=device))
   track = gen.generate()
   snapshot = Track.alloc(track.layout)       # allocate once, outside the capture
   with wp.ScopedCapture(device=device) as capture:
       gen.generate(advance=True)
       snapshot.copy_from(track)               # track.clone() would raise CaptureError here
   wp.capture_launch(capture.graph)

.. snippet: after

   assert np.array_equal(np.nan_to_num(snapshot.center.numpy()), np.nan_to_num(track.center.numpy()))

.. _upgrading-subclassing:

If you subclass or wrap Course, a utility or a generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Breaking, and Silent for overridden hooks
:Affects: subclasses, wrappers and monkeypatches of ``Course``, ``TrackGenerator``,
   ``GateGenerator`` or a utility, and code that reads their private attributes
:Find it: ``class .*(Course)``, ``_build_subtools``, ``_apply_bind``, ``_bind_args``,
   ``_step_result``, ``self.result =``, ``self.rng =``, ``self.components =``

- ``Course._build_subtools`` (which 0.1's first ``generate()`` called) is gone, and
  ``_apply_bind`` is no longer called by ``bind()``: an override or a monkeypatch of
  either silently stops running.
- ``course._bind_args`` holds only the arrays given to the last successful ``bind``. On
  0.1 it always had the four names, with ``None`` for an absent one, so
  ``course._bind_args["box_position"]`` (or ``["orientation"]`` on a course without
  collision) now raises ``KeyError``: use ``.get(name)``.
- ``course._step_result`` exists from construction (0.1 created it at the first
  ``step()``). The other 0.1 private names (``_cfg``, ``_posts``, ``_device``, ``_E``,
  ``_needs_boxes()``, ``_refresh``, ``_fill_posts``, ``_validate_seed_array``,
  ``_check_arr``) keep their meaning.
- New read-only properties raise ``AttributeError`` when assigned: ``.result`` on every
  utility, ``CollisionChecker.method``, ``CheckpointSampler.max_checkpoints``,
  ``PropSampler.max_props``, ``TrackGenerator.rng`` and ``GateGenerator.rng``,
  ``course.components``. A subclass that stores ``self.result``, ``self.rng`` or
  ``self.components`` must rename them.
- New public names on ``Course``, which a subclass attribute of the same name shadows:
  ``components``, ``prepare``, ``from_config``, ``track``, ``gates``, ``regenerate``,
  ``enqueue_generate``, ``inputs``, ``seeds``, ``graph``. New private names:
  ``_producer``, ``_needs_bind``, ``_superseded``, ``_reset_mask``, ``_gen_inputs``,
  ``_components_view``, ``_zero_mask``, ``_setup``, ``_init_from_config``,
  ``_set_legacy_attributes``, ``_ctx``, ``_ready``, ``_check_owner``, ``_check_mask``.
- The ``Course`` constructor is ``Course(generator=None, components=(), *, stream=None,
  prepare=True, config=None)``. A subclass whose ``__init__(self, config)`` keeps the 0.1
  signature still works for ``MyCourse(cfg)`` and ``MyCourse.from_config(cfg)``; the
  presets (``MyCourse.track(...)``) call ``cls(generator, components, stream=...,
  prepare=...)`` and need the new one.
- ``gen._graph`` is a read-only alias of ``gen.graph``; ``gen._generator_spec`` of a
  built-in is no longer a ``GeneratorSpec`` (:ref:`upgrading-plugins`).

**What to do.** The course is complete at construction, so move the body of a
``_build_subtools`` override to your ``__init__``, after ``super().__init__(...)``. For
work that must follow every regeneration, write a :term:`component` (its
``on_generate`` runs at each regeneration, :ref:`writing-a-component`) and pass it with
``extra=[...]`` to ``Course.track`` / ``Course.gates``, or in the ``components`` of
``Course(generator, components)``. ``Course(CourseConfig)`` takes no components, and
the presets build the course with ``cls(generator, components, stream=...,
prepare=...)``, so ``MyCourse.track(..., extra=[...])`` needs a subclass with the new
constructor signature (one that forwards ``*args, **kwargs`` to ``super().__init__``).
Replace an ``_apply_bind`` override with a ``bind()`` override that calls
``super().bind(...)`` first (it works with either signature), or with a component that
declares the inputs, passed the same way (so not on a course built from a
``CourseConfig``). A 0.1 subclass that added a sampler:

.. snippet: skip=legacy

.. code-block:: python

   from track_gen.props import PropSampler


   class MyCourse(Course):
       def _build_subtools(self):          # 0.1: run by the first generate(); 0.2: never
           super()._build_subtools()
           self.cones = PropSampler(self.result, spacing=0.3)

On 0.2:

.. snippet: group=subclass requires=upgrading/from-0.1#common

.. code-block:: python

   from track_gen.props import PropSampler


   class MyCourse(Course):
       def __init__(self, config):
           super().__init__(config)        # the course is complete here
           self.cones = PropSampler(self.result, spacing=0.3)


   course = MyCourse(CourseConfig(mode="track", gen=TrackGenConfig(num_envs=E, device=device),
                                  checkpoint_spacing=0.6))
   cones = course.cones.sample()

.. snippet: after

   assert cones.count.numpy().max() > 0

.. _upgrading-plugin-private-names:

Removed private names
~~~~~~~~~~~~~~~~~~~~~

:Type: Breaking
:Affects: 0.1 generator plugins that copied a built-in module and import its private
   kernels, and code that reads the generators' private scratch
:Find it: ``from track_gen._src``, ``_pipe``, ``_k(``, ``_k,``, ``_scratch``

The built-in generators are now built from the public device ops in ``track_gen.ops``,
and these private names are gone (their arguments also differ: explicit salts, strides
and masks):

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Removed
     - Use instead
   * - ``warp_generate._corner_count_sample_k``,
       ``warp_generate_hull._point_count_sample_k``
     - ``ops.sample_counts``
   * - ``warp_generate._corner_sample_k``, ``_corner_sample_style_k``,
       ``warp_generate_hull._point_sample_k``
     - ``ops.grid_cells`` and ``ops.sample_grid_points``
   * - ``warp_generate._ccw_sort_k``, ``warp_generate_hull._angle_sort_k``,
       ``warp_gate._order_ccw_k``
     - ``ops.angle_sort``
   * - ``warp_generate_hull._catmull_rom_k``, ``_aug_vertex``;
       ``warp_generate_voronoi._catmull_rom``, ``_aug_at``, ``_catmull_rom_dense_k``
     - ``ops.closed_catmull_rom`` (a whole spline) or ``ops.funcs.catmull_rom`` (one
       point, inside your kernel)
   * - ``warp_generate_hull._assemble_polygon_selected_k``,
       ``warp_generate_voronoi._assemble_polygon_selected_k``
     - ``ops.closed_polyline``
   * - ``_normalize_centerline_k`` (``polar``, ``voronoi``, ``checkpoint``),
       ``warp_gate._normalize_positions_k``
     - ``ops.normalize_extent`` (with ``count=`` for gates)
   * - every generator module's ``_pipe`` attribute
     - ``import track_gen.ops`` (or ``track_gen._src.warp_pipeline`` for the deprecated
       helpers of :ref:`upgrading-plugins`)
   * - ``warp_gate._run_gate_pipeline``, ``_gate_warp_alloc``; the gate modules'
       ``generate_*_gates``, ``*_gate_alloc_scratch``, ``PointGateScratch`` and
       ``_max_point_gates``
     - ``GateGenerator``, or the gate generator stages
       (``registry.get("gate_generator", name)``)

:ref:`generator-ops` describes each op. Also:

- Importing a built-in generator module no longer registers it. A plugin registers
  itself with a ``register(registry)`` function (:ref:`packaging-a-plugin`).
- The pipeline's working buffers no longer fall back to a generator scratch field of the
  same name, so a scratch field named ``w``, ``kappa``, ``z``, ``area_a`` or ``area_b``
  can no longer shadow them.
- ``gen._scratch`` is a view of the pipeline's buffers under the 0.1 field names (an
  absent stage's slot is ``None``), and ``GateGenerator._scratch`` (0.1's 6-tuple) is
  gone. Read intermediate results with ``gen.program.tap(...)`` (:ref:`pipeline-taps`).

Deprecations
------------

These still work. Most emit a ``DeprecationWarning`` (each item says which), and each is
removed in 0.3 (:ref:`upgrading-removals`).

.. _upgrading-set-capturing:

``set_capturing``
~~~~~~~~~~~~~~~~~

:Type: Deprecated
:Affects: code that switched the utilities into capture mode
:Find it: ``set_capturing``

``track_gen.set_capturing()`` (also ``track_gen.course.set_capturing`` and
``Course.set_capturing``) has no effect: the utilities no longer synchronize the host, so
they can be recorded into a graph without it (:ref:`upgrading-capture`).

.. snippet: group=set-capturing expect-error=AttributeError

.. code-block:: python

   import track_gen

   track_gen.set_capturing(True)   # AttributeError in the current release

Delete the calls.

.. _upgrading-style-sampling:

``style_sampling``
~~~~~~~~~~~~~~~~~~

:Type: Deprecated
:Affects: ``bezier`` configs with ``style_sampling=True`` and its ranges
:Find it: ``style_sampling``, ``scale_range``, ``rad_range``, ``handle_clamp_frac_range``

Building a ``bezier`` generator with ``style_sampling=True`` warns once per build.
Its per-env draws become randomizers, which draw each env's value from a uniform range
at every generation (:ref:`curricula`):

.. snippet: group=style requires=upgrading/from-0.1#common expect-error=TypeError

.. code-block:: python

   config = TrackGenConfig(num_envs=E, device=device, style_sampling=True,
                           scale_range=(0.8, 1.2), rad_range=(0.1, 0.35))

.. snippet: group=style

.. code-block:: python

   # a preset: the same uniform distributions
   gen = TrackGenerator(presets.track(num_envs=E, device=device,
                                      randomize={"generator.size": (0.8, 1.2),
                                                 "generator.rad": (0.1, 0.35)}))
   # from a cleaned config: randomize the spec the config builds
   config = TrackGenConfig(num_envs=E, device=device)
   spec = legacy.to_pipeline_spec(config).with_randomize({"generator.size": (0.8, 1.2)})
   gen = TrackGenerator(spec)
   # a Course
   course = Course.track(spec, checkpoint_spacing=0.5)

- ``scale_range=(a, b)`` maps to ``{"generator.size": (a / scale, b / scale)}`` (it
  must lie in ``[0.1, 10]``), ``rad_range`` to ``"generator.rad"``, and
  ``handle_clamp_frac_range`` to ``"generator.handle_clamp_frac"``.
- The distributions are the same, but the randomizers draw from another random stream,
  so each seed gives another track: rebaseline fixed evaluation sets built with
  ``style_sampling``.
- A course built from a randomized ``size`` checks its SDF and height-field resolution
  against the largest :term:`size` it can draw (``size_hi="auto"``,
  :ref:`collision-resolution-check`). At flat elevation the checkpoint capacity already
  holds every track that is not truncated, at any size (:ref:`curricula-course`).

.. _upgrading-registry:

The ``_src`` registries
~~~~~~~~~~~~~~~~~~~~~~~

:Type: Deprecated (no warning yet), and Breaking for direct writes
:Affects: generator plugins written for 0.1
:Find it: ``track_gen._src.generator_registry``, ``gate_generator_registry``,
   ``register(``, ``GENERATORS[``, ``GATE_GENERATORS[``, ``capturable=False``

A 0.1 plugin wrote ``generate(seeds_wp, config, out_centerline, out_valid_wp, scratch)``
and registered it in the private registry. Here is one, a wobbly circle:

.. snippet: group=wobbly

.. code-block:: python

   from types import SimpleNamespace

   import warp as wp


   @wp.kernel
   def _wobbly_k(seeds: wp.array(dtype=wp.int32), n: int, out: wp.array(dtype=wp.vec2f)):
       t = wp.tid()                               # one thread per point
       e = t // n
       theta = 6.2831853 * float(t - e * n) / float(n)
       r = 0.8 + 0.1 * wp.sin(3.0 * theta + float(seeds[e]))
       out[t] = wp.vec2f(r * wp.cos(theta), r * wp.sin(theta))


   @wp.kernel
   def _all_valid_k(valid: wp.array(dtype=wp.int32)):
       valid[wp.tid()] = 1


   def alloc_scratch(config):
       return SimpleNamespace()                   # this generator needs no buffers


   def generate(seeds_wp, config, out_centerline, out_valid_wp, scratch):
       E, N = int(config.num_envs), int(config.num_points)
       wp.launch(_wobbly_k, dim=E * N, inputs=[seeds_wp, N, out_centerline],
                 device=out_centerline.device)
       wp.launch(_all_valid_k, dim=E, inputs=[out_valid_wp], device=out_valid_wp.device)

Its 0.1 registration still works, with no warning:

.. snippet: group=reg-01 skip=legacy

.. code-block:: python

   from track_gen._src.generator_registry import GeneratorSpec, register

   register(GeneratorSpec(name="wobbly", alloc_scratch=alloc_scratch, generate=generate))

.. snippet: after

   assert TrackGenerator(TrackGenConfig(generator="wobbly", num_envs=E, device=device)).generate().valid.numpy().all()

``track_gen._src.generator_registry`` and ``gate_generator_registry`` are now views of
the public registry, ``track_gen.plugin.registry``, and they are removed in 0.3. Writing into
``GENERATORS`` or ``GATE_GENERATORS`` raises ``TypeError``: in tests, use
``registry.scoped()`` instead of resetting the dictionaries. The old ``register(spec)``
still replaces a track generator of the same name, now with a ``DeprecationWarning``,
while ``registry.add`` refuses a name that is already registered
(``DuplicateSpecError``) unless you pass ``replace=True``. Two ports:

.. snippet: group=reg-min skip=legacy

.. code-block:: python

   # the minimal port: the same generate(), the public registry
   from track_gen.plugin import GeneratorSpec, registry

   registry.add(GeneratorSpec(name="wobbly", alloc_scratch=alloc_scratch,
                              generate=generate))   # UserWarning: not a dotted name

.. snippet: after

   assert TrackGenerator(TrackGenConfig(generator="wobbly", num_envs=E, device=device)).generate().valid.numpy().all()

.. snippet: group=reg-full requires=upgrading/from-0.1#common,upgrading/from-0.1#wobbly

.. code-block:: python

   # the full port: every launch goes through ctx (or track_gen.ops), under a dotted name
   from track_gen import ops
   from track_gen.exec import resolve
   from track_gen.plugin import GeneratorSpec, registry


   def generate_ctx(seeds_wp, config, out_centerline, out_valid_wp, scratch, *, ctx):
       ctx = resolve(ctx, out_centerline.device)  # the facades always pass ctx
       E, N = int(config.num_envs), int(config.num_points)
       ctx.launch(_wobbly_k, E * N, [seeds_wp, N, out_centerline])
       ops.fill(ctx, out_valid_wp, 1)


   registry.add(GeneratorSpec(name="acme.wobbly", alloc_scratch=alloc_scratch,
                              generate=generate_ctx))

.. snippet: after

   assert TrackGenerator(TrackGenConfig(generator="acme.wobbly", num_envs=E, device=device)).generate().valid.numpy().all()

To ship it, declare the entry point in your package's ``pyproject.toml``
(``acme_tracks.register(registry)`` calls ``registry.add``):

.. code-block:: toml

   [project.entry-points."track_gen.plugins"]
   acme = "acme_tracks:register"

- The minimal port runs eagerly and in the generator's own CUDA graph. Recording it into
  your graph on another stream needs the full port: the minimal one raises
  ``LegacyPluginStreamError`` there (:ref:`capture-troubleshooting`).
- A generator registered with ``capturable=False`` raises ``NotCapturableError`` before
  any launch when you call it while your graph is recording (0.1 ran it eagerly at
  capture time). Call it outside the capture, or build from a ``PipelineSpec`` with
  ``require_caller_capturable=True``, so that construction refuses such a generator
  (``PlanError``) instead of the capture (:ref:`capture-troubleshooting`).
- A dotted name (``acme.wobbly``) changes the ``generator=`` string in your users'
  configs and saved :term:`specs <spec>`; a name without a dot only warns.
- ``GeneratorSpec`` gains two trailing fields, ``accepts_ctx`` (default ``False``) and
  ``params`` (default ``None``), so positional construction is unchanged.

:ref:`packaging-a-plugin` covers registration, entry points and names;
:ref:`plugin-abis` the ``GeneratorSpec`` contract and its status.

.. _upgrading-plugins:

Plugins: built-in generators are Stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Deprecated, and Breaking for code that calls the built-ins' specs
:Affects: plugin authors, and code that reads the built-in generators from the registry
:Find it: ``.params``, ``.generate(``, ``.alloc_scratch(``, ``.capturable``,
   ``GateGeneratorSpec``, ``warp_gate.``, ``GenScratch``, ``TrackGenConfig)``
   subclasses

The built-in generators are stages now (``Stage`` classes), not
``GeneratorSpec`` objects: ``registry.get("generator", "bezier")`` (and the 0.1
``generator_registry.get``, ``GENERATORS[...]``, the gate equivalents) returns a class.

- Its ``.generate`` and ``.alloc_scratch`` are gone (the module functions
  ``generate_*_warp`` / ``*_alloc_scratch`` remained in 0.2). Read capture support and
  the rest from ``registry.describe(kind, name)`` (``.capturable``, ``.accepts_ctx``,
  ``.seeded``), and a gate generator's orders from
  ``registry.describe("gate_generator", name).supported_orderings``.
- ``.params`` warns: use ``.Params``, the generator's :term:`Params` class.

.. snippet: group=plugins expect-error=AttributeError

.. code-block:: python

   from track_gen.plugin import registry

   params_cls = registry.get("generator", "bezier").params   # AttributeError in the current release

.. snippet: group=plugins

.. code-block:: python

   params_cls = registry.get("generator", "bezier").Params
   print(registry.describe("generator", "bezier").capturable)   # True

.. snippet: after

   assert params_cls.__name__ == "BezierParams"

For your own plugins:

- Write new generators as a ``Stage`` (:ref:`writing-a-generator`). A ``GeneratorSpec``
  keeps working (:ref:`upgrading-registry`); one with ``capturable=False`` now runs as a
  host step between recorded segments instead of making the whole pipeline eager.
- ``GateGeneratorSpec`` warns when a generator binds it (``GateGenerator``, ``Course``,
  the conformance kit), not when it is registered, and is removed in 0.3: write a
  ``Stage(kind="gate_generator", output="anchors")`` (:ref:`writing-a-gate-generator`).
- The ``warp_gate`` helpers of 0.1 gate plugins (``alloc_order_scratch``,
  ``order_points``, ``finish_ordered_gates``) warn; the ``warp_pipeline``,
  ``warp_generate`` and ``warp_generate_hull`` helpers (``resample_uniform``,
  ``self_intersections_inplace``, ``ccw_sort_inplace``, ...) are deprecated without a
  warning yet. :ref:`legacy-src-helpers` maps each one to its op in ``track_gen.ops``.
  ``warp_pipeline.GenScratch`` moved to ``warp_generate.GenScratch`` (the old name
  warns).
- Run ``track_gen.testing.check_generator`` in your CI (:ref:`testing-a-plugin`).

**Your plugin's knobs.** A 0.1 plugin read its knobs from a ``TrackGenConfig`` subclass.
That still works, but a frozen ``Params`` dataclass passed as ``params=`` supersedes
it: users set the knobs with ``generator_params``, the values are validated at
construction, and ``registry.describe`` reports them. Override ``Params.from_legacy`` so
the subclass configs your users already have keep their meaning:

.. snippet: group=knobs-01 skip=legacy

.. code-block:: python

   # 0.1: a config subclass, read with getattr
   @dataclasses.dataclass
   class AcmeConfig(TrackGenConfig):
       acme_lobes: int = 3

   def generate(seeds_wp, config, out_centerline, out_valid_wp, scratch):
       lobes = getattr(config, "acme_lobes", 3)
       ...

.. snippet: group=knobs requires=upgrading/from-0.1#common

.. code-block:: python

   import dataclasses

   from track_gen import ops
   from track_gen.exec import resolve
   from track_gen.plugin import GeneratorSpec, Params, registry, static


   @dataclasses.dataclass
   class AcmeConfig(TrackGenConfig):              # your users' existing configs
       acme_lobes: int = 3


   @dataclasses.dataclass(frozen=True, kw_only=True)
   class AcmeParams(Params):
       lobes: int = static(3, lo=1, hi=12, doc="lobes of the wobble")

       @classmethod
       def from_legacy(cls, cfg):                 # a config without generator_params
           return cls(lobes=getattr(cfg, "acme_lobes", 3))


   @wp.kernel
   def _lobes_k(seeds: wp.array(dtype=wp.int32), n: int, lobes: int,
                out: wp.array(dtype=wp.vec2f)):
       t = wp.tid()
       e = t // n
       theta = 6.2831853 * float(t - e * n) / float(n)
       r = 0.8 + 0.1 * wp.sin(float(lobes) * theta + float(seeds[e]))
       out[t] = wp.vec2f(r * wp.cos(theta), r * wp.sin(theta))


   def generate(seeds_wp, config, out_centerline, out_valid_wp, scratch, *, ctx):
       ctx = resolve(ctx, out_centerline.device)
       E, N = int(config.num_envs), int(config.num_points)
       lobes = config.generator_params.lobes     # the resolved, validated instance
       ctx.launch(_lobes_k, E * N, [seeds_wp, N, lobes, out_centerline])
       ops.fill(ctx, out_valid_wp, 1)


   registry.add(GeneratorSpec(name="acme.lobes", alloc_scratch=lambda config: None,
                              generate=generate, params=AcmeParams))

   old = TrackGenerator(AcmeConfig(generator="acme.lobes", acme_lobes=5,
                                   num_envs=E, device=device))
   new = TrackGenerator(TrackGenConfig(generator="acme.lobes",
                                       generator_params={"lobes": 5},
                                       num_envs=E, device=device))

.. snippet: after

   assert np.array_equal(np.nan_to_num(old.generate().center.numpy()), np.nan_to_num(new.generate().center.numpy()))

Optional: new ways, no action needed
------------------------------------

The 0.1 forms below keep working through 0.x with the same outputs. Move when it suits
you.

.. _upgrading-flat-config:

Flat config, ``Params`` and presets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Type: Optional: no action needed
:Affects: everyone
:Find it: ``TrackGenConfig(``, ``GateGenConfig(``

Flat ``TrackGenConfig`` and ``GateGenConfig`` fields kept working in 0.2, with
byte-identical outputs. 0.2 adds two forms that give the same bytes: each generator's
knobs as a typed ``Params`` class (``generator_params``), and presets, functions that
build a spec (a ``PipelineSpec``, the list of stages a generator runs).
:ref:`generating` describes them. Only these force a change:

- Fields that the generator never reads warn (``DeprecationWarning`` at config
  construction, removed in 0.3): ``energy_steps``, ``energy_lr``, ``energy_w_sep``,
  ``energy_w_len``, ``energy_w_bend``, ``energy_w_anchor``, ``tp_iters``, ``tp_tau``,
  ``tp_alpha``, ``tp_beta``, ``smooth_finish``, ``smooth_finish_iters``,
  ``smooth_finish_tau``, ``relax_chunk_size``, ``relax_use_warp``, ``max_regen_iters``,
  ``min_angle``, ``num_harmonics``, ``decay_p``, ``amplitude``,
  ``num_centerline_samples``. Setting one leaves every output unchanged: delete it.
- ``TrackGenConfig`` and ``GateGenConfig`` gain a public field named ``size`` (each
  track's size, a multiplier of ``scale``): a config subclass that declares its
  own ``size`` field must rename it.
- The values of :ref:`upgrading-bounds`.
- The checks of each generator's own knobs run when the generator is built, not when
  the config is: a bad knob of the selected generator raises from
  ``TrackGenerator(...)`` (or ``Course(...)``) instead of ``TrackGenConfig(...)``, with
  the same message, and a bad knob of a generator you did not select is ignored.

.. snippet: group=flat requires=upgrading/from-0.1#common expect-error=TypeError

.. code-block:: python

   TrackGenConfig(num_envs=E, device=device, energy_steps=200)   # TypeError in the current release

.. snippet: group=flat

.. code-block:: python

   flat = TrackGenConfig(num_envs=E, half_width=0.05, rad=0.3, edgy=0.1, device=device)
   typed = TrackGenConfig(num_envs=E, half_width=0.05, device=device,
                          generator_params={"rad": 0.3, "edgy": 0.1})
   spec = presets.track(num_envs=E, device=device, half_width=0.05,
                        generator_params={"rad": 0.3, "edgy": 0.1})
   tracks = [TrackGenerator(c).generate().center.numpy().copy()
             for c in (flat, typed, spec)]

.. snippet: after

   assert all(np.array_equal(np.nan_to_num(tracks[0]), np.nan_to_num(t)) for t in tracks[1:])

.. _upgrading-course:

``Course`` presets
~~~~~~~~~~~~~~~~~~

:Type: Optional: no action needed, except one call
:Affects: ``Course`` users
:Find it: ``CourseConfig(``, ``course.generate()``

``Course(CourseConfig(...))`` keeps working through 0.x with the same outputs. One call
changes meaning: on 0.1 a course was empty until its first ``generate()``
(``step()`` raised before it); on 0.2 it is ready at construction, so that first
``generate()`` only computes the same batch again. Drop it:

.. snippet: group=course requires=upgrading/from-0.1#common
.. snippet: setup

   position = wp.zeros(E, dtype=wp.vec3f, device=device)
   orientation = wp.full(E, wp.quatf(0.0, 0.0, 0.0, 1.0), dtype=wp.quatf, device=device)
   half_extents = wp.full(E, wp.vec2f(0.02, 0.01), dtype=wp.vec2f, device=device)

.. code-block:: python

   # 0.1, still runs
   config = TrackGenConfig(num_envs=E, half_width=0.05, device=device)
   course = Course(CourseConfig(mode="track", gen=config, seeds=42, checkpoint_spacing=0.6,
                                collision="segments"))
   course.bind(position, orientation, half_extents)
   course.generate()                   # 0.1: required; 0.2: the same batch again
   res = course.step()

With a preset (the same components and outputs):

.. snippet: group=course

.. code-block:: python

   course = Course.track(config, checkpoint_spacing=0.6, collision="segments", seeds=42)
   course.bind(position, orientation, half_extents)
   res = course.step()                 # ready at construction
   res["progress"] is res.events       # each component's result by name
   course.reset()                      # every env; reset(mask) as before

.. snippet: after

   assert res["collision"] is res.contacts and list(course.inputs) == ["regen_mask", "reset_mask"]

New on every course: ``StepResult`` access by component name; ``bind(**arrays)`` for
the inputs of your own components; ``reset(mask=None)`` resets every
env; ``course.inputs["reset_mask"]``, a stable ``[E]`` array your kernels can write the
reset mask into; masked regeneration (:ref:`rl-auto-reset`); ``Course(generator,
components)`` over a generator you built (one course per generator: a newer course built
on the same generator takes it over, and the older one raises at its next call). A
preset's ``collision="sdf"`` and ``tools.HeightFieldBake`` warn at construction when
their resolution does not cover the largest track size the generator can draw. Only a
``Course`` subclass may need changes (:ref:`upgrading-subclassing`). :ref:`courses` and
:ref:`composing-a-course` describe the presets.

.. _course-config-migration:

Migrating from ``CourseConfig``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Course(CourseConfig)`` keeps working through 0.x with no warning. To move to a preset:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - ``CourseConfig``
     - ``Course.track`` / ``Course.gates``
     - What changes
   * - ``mode="track"`` / ``"gates"``
     - the preset
     -
   * - ``gen``
     - ``source``: a config, a ``PipelineSpec``, or a built ``TrackGenerator`` /
       ``GateGenerator``
     -
   * - ``seeds`` (int / array)
     - ``seeds=`` (int / array)
     - ``course.rng`` is ``None`` unless ``seeds=PerEnvSeededRNG(...)``: read
       ``course.seeds.seeds`` for logging. With a bound array and no rng,
       ``generate(seeds=)`` raises ``RuntimeError`` (the library never writes into your
       seed array): pass ``seeds=PerEnvSeededRNG(seeds=arr, ...)`` to keep the
       write-in-place
   * - ``checkpoint_spacing``, ``max_checkpoints``
     - the same names (``track``)
     -
   * - ``collision`` (default ``None``)
     - ``collision=`` (default ``None``): ``"segments"``, ``"sdf"`` or a component
       named ``"collision"``
     -
   * - ``sdf_resolution``, ``max_boxes`` (track)
     - ``collision=tools.BandCollision(method, max_boxes=k[, sdf_resolution=R])``, with
       ``method`` ``"segments"`` or ``"sdf"`` (the string form ``collision="segments"``
       means ``max_boxes=1``)
     - an SDF resolution too coarse for the largest track size warns
   * - ``heightfield_resolution``
     - ``extra=[tools.HeightFieldBake(R)]``
     - as above
   * - ``post_radius``, ``max_boxes`` (gates)
     - ``collision=r``, or ``collision=tools.DiscCollision(r, max_boxes=k)``
     -
   * - ``frame_collision``, ``agent_radius``, ``frame_thickness``, ``frame_depth``
     - ``collision=tools.FrameCollision(agent_radius, frame_thickness, frame_depth)``
     -
   * - ``samples_per_gate``, ``localize_window``
     - the same names (``gates``)
     -
   * - ``bind(position, orientation, half_extents, box_position)``
     - the same call; names no component declares are ignored, and ``None`` means "not
       given"
     - the errors are the components': a missing ``orientation`` / ``half_extents`` of
       a box or disc collision raises ``TypeError("collision: missing inputs
       [...]")``, and ``bind(None)`` raises ``TypeError("progress: missing inputs
       ['position']")``, where a config course raises ``RuntimeError`` /
       ``ValueError``: catch ``TypeError`` after moving

``Course.gates(cfg, collision=r, seeds=PerEnvSeededRNG(seeds=s, num_envs=E,
device=d))`` gives the bytes of ``Course(CourseConfig(mode="gates", gen=cfg, seeds=s,
post_radius=r))``.

.. _upgrading-smaller:

Smaller changes
---------------

- Each generator and each ``Course`` is single-writer: never call the same one from two
  threads at once. Independent generators on different threads no longer serialize on a
  process-wide lock (0.1 plugins without ``accepts_ctx=True`` still take it).
- ``track_gen._experimental`` is not in the wheel any more (it stays in the source
  tree).
- ``TrackGenerator`` raises ``ValueError`` instead of ``AssertionError`` for an
  unsupported ``relax_solver`` or ``smooth_finish=True``, and checks that the rng has
  exactly ``num_envs`` seeds, as ``GateGenerator`` does.
- ``relax_solver``, ``z_profile`` and ``gate_ordering`` name any registered stage and
  are checked when the generator is built: ``TrackGenConfig(z_profile="bogus")`` and
  ``GateGenConfig(gate_ordering="spiral")`` construct, and the generator raises a
  ``ValueError`` that lists the available names.
- An unknown ``z_profile`` passed to the private ``apply_z_profile`` raises
  ``ValueError`` (0.1 fell back to ``"noise"``).
- In the legacy plugin helpers, a ``count[e]`` above the row length is clamped to it (0.1
  read past the env's row); valid counts give the same results.

.. _upgrading-preview:

If you used a 0.2.0.devN preview
--------------------------------

The ``v0.2.0.devN`` tags were previews of 0.2. Everything above applies to code written
for them as well; the changelog remains the exhaustive record of each release. This
table lists the changes between previews that alter an error, a message, a default, a
stream or an output, with what to do.

.. list-table::
   :header-rows: 1
   :widths: 12 50 38

   * - Since
     - Change
     - What to do
   * - 0.2.0.dev2
     - With ``TRACKGEN_DEBUG_SYNC=1``, the built-in generators synchronize once per call
       instead of after each internal step
     - nothing (a debugging aid: :ref:`upgrading-host-syncs`)
   * - 0.2.0.dev3, 0.2.0.dev5
     - ``gen.params`` lists every stage that has a dynamic field, after ``"generator"``
       (since 0.2.0.dev5 the gate pipelines list ``"gate_frames"``, and the ``polar`` /
       ``voronoi`` / ``checkpoint`` gate pipelines ``"normalize"``)
     - look stages up by id; never assume ``list(gen.params) == ["generator"]``
   * - 0.2.0.dev4
     - The width stage runs before ``relax``; ``XpbdParams``, ``OffsetParams`` and
       ``ValidityParams`` lose ``half_width``; spec format ``/1`` becomes ``/2``
     - :ref:`pipeline-dev3-specs` below
   * - 0.2.0.dev4
     - Plugin width stages run before ``relax`` on the constant-spacing points: XPBD
       separates with per-point widths, the band and the thickness check follow them,
       the floor check (fail bit 4) reads the width carried to the final points, and a
       constant plugin width no longer reproduces ``width.constant``'s bytes
     - rebaseline outputs of pipelines whose width is not ``width.constant``; a width
       that needs the final geometry rewrites ``cl.half_width`` after ``frames``
   * - 0.2.0.dev4
     - ``gate_frames`` / ``validity.gate_default`` must agree on ``gate_width``, and
       ``gate_relax.spheres`` / ``validity.gate_default`` on ``gate_radius``
       (``PlanError``)
     - pass one value to every stage (the presets do)
   * - 0.2.0.dev4
     - ``elevation.flat`` on tracks reads ``cl.count``
     - a hand-built spec that places it before any ``cl.count`` writer no longer plans:
       move it later
   * - 0.2.0.dev4
     - Built-in gate generators are ``Stage`` classes; ``GateGenerator._scratch`` is gone
     - :ref:`upgrading-plugins`; read intermediates with ``gen.program.tap``
   * - 0.2.0.dev4
     - Under torch's capturing stream, ``skip_unless`` records unconditionally unless
       opted in; graphs with conditional nodes cannot be CUDA child graphs
     - :ref:`upgrading-capture`
   * - 0.2.0.dev4
     - A ``HOST`` stage inside ``wp.ScopedCapture("cpu")`` raises
       ``NotCapturableError``, and constructing a producer there raises
       ``CaptureError``; ``NotCapturableError`` comes before the "never prepared"
       ``CaptureError``; its message says "graph capture" and can name the stages
     - construct before capturing; keep ``HOST`` stages out of CPU captures
   * - 0.2.0.dev4
     - ``LaunchContext.skip_unless`` returns a ``bool`` (``True`` when it recorded a
       conditional node)
     - code that tested ``is None`` reads the flag
   * - 0.2.0.dev5
     - ``out=`` must be a ``Track`` / ``GateSequence`` instance (``TypeError``);
       building into storage a live Program owns raises ``PlanError``, also for the same
       pipeline; on a view of a larger result, ``out.attributes`` keys the pipeline does
       not export are padded instead of refused
     - ``del`` the earlier generator before rebuilding into the same result
   * - 0.2.0.dev5
     - ``Layout`` validates its fields; ``BatchSpec.closed`` must be a ``bool``
       (``PlanError``)
     - pass Python ``int`` / ``bool`` values
   * - 0.2.0.dev5
     - The generator is the first stage that is not a randomizer (it was
       ``stages[0]``)
     - use ``spec.stages[spec.ids.index("generator")]``
   * - 0.2.0.dev5
     - ``presets.track(relax=)`` / ``presets.gates(relax=)`` default to ``"auto"``
     - nothing for the built-ins (the same stage); ``external`` gets ``relax.none``
   * - 0.2.0.dev5
     - The ``.params`` alias of the built-in generators warns; ``style_sampling`` warns,
       and its drawn knobs refuse ``set`` and ``randomize=``
     - :ref:`upgrading-plugins`, :ref:`upgrading-style-sampling`
   * - 0.2.0.dev5
     - ``validity.gate_default`` checks bar crossings per env; its ``gate_width`` and
       ``check_crossing`` are inert (removed in 0.3) and take part in the agreement
       check only when set
     - drop them from hand-built specs
   * - 0.2.0.dev5
     - ``min_num_points`` of ``bezier`` / ``hull`` is an ``int32``: ``set`` refuses a
       ``float``; ``StageParams.set`` names the merged range; a gates ``Course`` refuses
       ``set("gate_width", v)`` with ``v <= 0``
     - pass ints; keep bar widths positive
   * - 0.2.0.dev5
     - Spec dicts built from a built-in ``Params`` instance gain ``"size": 1.0``, so they
       no longer load on 0.2.0.dev4, and dict diffs change
     - regenerate stored dicts once
   * - 0.2.0.dev5
     - New ``ValueError`` / ``PlanError`` for a ``size`` no stage applies, for ``size=``
       together with ``generator_params={"size": ...}``, and for randomizing an ``int32``
       field; ``legacy.to_pipeline_spec`` given a ``GateGenConfig`` raises ``TypeError``
     - use ``legacy.to_gate_pipeline_spec`` for gate configs
   * - 0.2.0.dev5
     - ``registry.add`` accepts ``preferred={"relax": ...}`` on centerline generators and
       refuses a ``declare()`` override on other kinds; ``FrameChecker(seq, num_envs)``
       checks ``num_envs``; the conformance kit runs ``params.per_env``
     - a dynamic field's kernel reads ``param[e]``, never ``param[0]``
   * - 0.2.0.dev5
     - ``clone()`` validates, waits on the host and refuses captures; utilities bind
       their source's arrays at construction and raise ``TypeError`` for a missing
       member; ``CourseLine.track`` warns
     - :ref:`upgrading-utilities`
   * - 0.2.0.dev6
     - Invalid envs read as poisoned in every utility; ``StepResult.valid``,
       ``CheckpointSet.valid`` / ``.generation`` are new; the default checkpoint capacity
       comes from the longest track the generator can produce
     - :ref:`upgrading-invalid-envs`, :ref:`upgrading-result-fields`,
       :ref:`upgrading-checkpoint-capacity`
   * - 0.2.0.dev6
     - Rebinding after a capture warns; malformed optional members raise; gates-mode post
       collision scans each env's real posts; ``check_source``'s missing-member message
       lists the optional members apart
     - :ref:`upgrading-utilities`
   * - 0.2.0.dev6
     - ``ProgressTracker`` and ``TrackLocalizer`` reset an env by themselves when its
       ``generation`` (its epoch) changes; a regeneration without ``advance`` does not
       change it
     - stop resetting by hand after an advancing regeneration (``Course`` still resets)
   * - 0.2.0.dev6
     - ``course.inputs`` gains ``"reset_mask"``
     - code that expects one key sees two
   * - 0.2.0.dev6
     - ``Course._reset_order`` and ``_cp_default`` are gone; ``_bind(...)`` takes a
       ``{name: array}`` dict; the regeneration calls ``generator.program.enqueue``, so
       an override of the facade's ``_enqueue`` no longer runs
     - :ref:`upgrading-subclassing`
   * - 0.2.0.dev6
     - The ``style_sampling`` warning's ``Course`` clause names ``Course.track`` and no
       longer mentions checkpoint sizing
     - filters on the category or the first sentence are unaffected
   * - 0.2.0.dev6
     - A refused conditional body is re-raised as ``CaptureError`` naming the components
     - record inside ``track_gen.exec.unconditional()``
   * - 0.2.0.dev6
     - On CUDA, a ``Course(CourseConfig)`` built inside an ambient
       ``LaunchContext.scope(other)`` attaches its components on Warp's current stream,
       ordered with ``other``; a course on another stream than its generator's waits for
       the generator's pending work
     - nothing (same outputs; only the streams of construction launches change)

.. _pipeline-dev3-specs:

Specs written for 0.2.0.dev3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The width stage moved and three stages lost a field, so a spec built in Python against
0.2.0.dev3 may need an edit; a saved dict migrates on its own (:ref:`Saved specs: format /1
and /2 <pipeline-spec-format>`), with one ``DeprecationWarning``. ``TrackGenConfig``
users, ``presets.track(half_width=...)`` users and every ``Track`` consumer are
unaffected.

- A ``half_width`` key in the params of ``relax.xpbd``, ``offset.default`` or
  ``validity.default`` raises, before the params are coerced: ``stage 'relax'
  (relax.xpbd): half_width is no longer a relax.xpbd param (0.2.0.dev4): the half-width
  is cl.half_width, written by the width stage (presets.track(half_width=...) /
  width.constant)``.
- A spec in the old order (``relax``, ``uniform``, ``width``) has ``relax.xpbd`` read
  ``cl.half_width`` before any writer: the read error names the new order (the example
  under :ref:`build() and the planner <pipeline-stages-build-and-the-planner>`). A
  hand-built ``relaxed_stations`` gate spec written before the width stage moved, which
  had no width stage, gets the same message.
- An old-order spec without relaxation (``relax.none``, ``uniform``, ``width``, ...)
  still plans and gives the same bytes, with one launch fewer than in 0.2.0.dev3: its
  width stage now writes ``cl.half_width`` after the uniform resample, which has no
  version of it to fill.

.. _upgrading-removals:

Removed in 0.3, and what 0.2 kept
---------------------------------

What 0.2 deprecates is removed in 0.3:

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Removed in 0.3
     - Warns in 0.2
     - Replacement
   * - ``track_gen.set_capturing``, ``Course.set_capturing`` and the private
       ``_CAPTURING`` names
     - calls: yes; flags: no
     - nothing (:ref:`upgrading-set-capturing`)
   * - ``CourseLine.track``
     - yes
     - ``CourseLine.centerline``
   * - ``registry.get("generator", g).params``
     - yes
     - ``.Params``
   * - ``style_sampling``, ``scale_range``, ``rad_range``, ``handle_clamp_frac_range``
     - only when building a bezier generator with style enabled
     - ``randomize=`` (:ref:`upgrading-style-sampling`)
   * - ``GateGeneratorSpec`` and its adapter
     - yes, when bound
     - a gate generator ``Stage``
   * - the ``warp_gate`` helpers of 0.1 gate plugins
     - yes
     - the order stages and ``track_gen.ops``
   * - ``warp_pipeline.GenScratch``
     - yes
     - the generator facade or a Stage's private scratch
   * - the ``_src`` plugin helpers (``warp_pipeline``, ``warp_generate``,
       ``warp_generate_hull``) and the built-ins' module functions (``generate_*_warp``,
       ``*_alloc_scratch``, ``apply_z_profile``)
     - no
     - ``track_gen.ops`` (:ref:`legacy-src-helpers`)
   * - the ``_src`` registries (``generator_registry``, ``gate_generator_registry``)
     - only when a registration replaces another
     - ``track_gen.plugin.registry``
   * - the ``TrackGenConfig`` fields the generator never reads
     - yes, when set
     - nothing (:ref:`upgrading-flat-config`)
   * - ``validity.gate_default``'s ``gate_width`` and ``check_crossing``
     - no (inert)
     - nothing
   * - loading ``"track_gen.pipeline/1"`` spec dicts
     - yes
     - save them again as ``/2``

What did not change in 0.2:

- For valid envs, every default generator's output is byte-identical to 0.1, except
  ``repulsive`` (:ref:`upgrading-repulsive`); only the ``valid`` flag of tracks 0.1
  truncated changes (:ref:`upgrading-truncation`).
- ``TrackGenConfig``, ``GateGenConfig`` and ``Course(CourseConfig)`` keep working
  in 0.2, with no warning beyond the items above.
- The runtime utilities give 0.1's outputs for valid envs, apart from the ``sdf`` query
  (:ref:`upgrading-sdf`) and the checkpoint capacity
  (:ref:`upgrading-checkpoint-capacity`).
