.. _curricula:

Domain randomization and curricula
==================================

Many knobs of a generator are :term:`dynamic fields <dynamic field>`: per-:term:`env`
``[E]`` arrays the stages read at every generation, so each env can run its own value and
the values can change between generations with no rebuild and no new :term:`capture`.
Two tools write them:

- **randomize** (domain randomization): ``randomize=`` adds a :term:`stage` that draws a
  field per env, inside the :term:`generation`, from a range you set (`Per-env draws: randomize=`_);
- **a curriculum**: you write the live arrays yourself, from the host or from a kernel in
  your own graph (`Curricula: writing the live arrays`_).

This page lists the dynamic fields, then covers both tools, what each env was built
with, the difficulty axes, the course :term:`size` and the utilities it affects, and a
``Course`` curriculum. Static fields (counts, buffer sizes, the road width) need a new
generator: :ref:`generating-params`.

.. _tuning-per-env:

Dynamic fields
--------------

A dynamic field is a per-env ``[E]`` array (``float32``, or ``int32`` for a count) the
kernels read at every generation. You can draw a ``float32`` one per env with
``randomize=``, or write any of them yourself. The dynamic fields of the built-in stages
are:

- the built-in generators' shape knobs: ``rad`` and ``handle_clamp_frac`` (``bezier``),
  ``hull_displacement`` (``hull``), ``polar_radial_jitter`` and
  ``polar_angular_jitter`` (``polar``), ``voronoi_radial_variation`` and
  ``voronoi_angular_jitter`` (``voronoi``), ``checkpoint_radius_min_frac``,
  ``checkpoint_angle_jitter`` and ``checkpoint_turn_rate`` (``checkpoint``). The polar,
  voronoi and checkpoint **gate** generators have the same ring knobs (not
  ``checkpoint_turn_rate``: they do not steer). Each generator page gives their bounds,
  effects, visual examples and failure cases (:ref:`generators-per-env`);
- the course **size** (``size``, ``[0.1, 10]``) of every built-in generator but
  ``external``: ``generator.size`` on tracks and on the ``bezier`` / ``hull`` gate
  generators, ``normalize.size`` (the ``normalize.extent`` stage) for the ``polar``,
  ``voronoi`` and ``checkpoint`` gate generators (`Size: the sizing rule`_);
- the **corner count** of ``bezier`` and ``hull``: ``min_num_points``, an ``int32`` in
  ``[2, max_num_points]`` that you write (``set`` or your kernel); ``randomize=`` does
  not draw ``int32`` fields (:ref:`bezier-corner-count`);
- the gate **bar width**, ``gate_frames.default``'s ``gate_width`` (key
  ``gate_frames.gate_width``, ``>= 0``; :ref:`gate-width-per-env`);
- the elevation profiles' ``z_min`` and ``z_max`` (``uniform``, ``random_walk``,
  ``noise``), ``z_max_step`` (``random_walk``) and ``z_noise_amplitude`` (``noise``), on
  tracks, on gates and on the centerline before the stations alike (`Elevation knobs`_);
- ``width.from_channel``'s ``scale``, ``lo`` and ``hi``, and every field a plugin
  declares with ``plugin.dynamic`` (:ref:`generator-runtime-params`).

Everything else is static (``max_num_points``, knot, anchor and checkpoint counts,
``half_width``, ``spacing``, ``n_max``, ``edgy``, the core ``scale``, ``z_base``, the
relax and validity tolerances, the ``repulsive`` growth knobs, the gate generators'
``min_num_points``): build a new generator to change it. ``gen.params`` lists the stages
that have a dynamic field; ``registry.describe(kind, name).params_schema`` marks each
field (``"x-dynamic": true``).

.. _pipeline-randomize:

Per-env draws: ``randomize=``
-----------------------------

The :term:`preset` argument ``presets.track(randomize={"<stage id>.<field>": (lo, hi)})``
(and ``presets.gates``) draws each env's value uniformly in ``[lo, hi)`` at every generation. The generator's
stage id is ``"generator"``, the elevation's ``"elevation"``, in ``presets.track`` and
``presets.gates`` alike:

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   spec = presets.track(
       num_envs=E, device=device, generator="polar",
       elevation=("noise", {"z_min": -0.3, "z_max": 0.3, "z_noise_amplitude": 0.1}),
       randomize={"generator.polar_radial_jitter": (0.3, 0.85),
                  "elevation.z_noise_amplitude": (0.0, 0.3)},
   )
   spec.ids
   # ('randomize.generator.polar_radial_jitter', 'generator', 'resample', ...,
   #  'randomize.elevation.z_noise_amplitude', 'elevation', 'validity', 'lift')
   gen = TrackGenerator(spec)
   gen.params["generator"]["polar_radial_jitter"]   # [E]: what each env was drawn
   gen.generate(advance=True)                       # new tracks, new draws

.. snippet: after

   d = gen.params["generator"]["polar_radial_jitter"].numpy()
   assert d.min() >= 0.3 and d.max() < 0.85 and len(set(d.tolist())) > 1

- **The draw.** Env ``e`` gets ``lo[e] + u * (hi[e] - lo[e])``, with ``u`` uniform in
  ``[0, 1)`` from a stream keyed on the env's effective :term:`seed` (:ref:`generating-seeds`)
  and the randomizer, clamped to the field's bounds. The draws are deterministic per
  ``(seed, epoch)`` (:term:`epoch`), differ across envs and across randomizers, and change with
  ``advance=True`` (each replay of a graph that records ``generate(advance=True)``
  redraws) or a reseed. A masked generation redraws the masked envs only; the others
  keep their values and their rows.
- **Placement.** Each entry adds a ``randomize.uniform`` stage, with the id
  ``randomize.<stage>.<field>``, right before its target, so the target reads this
  generation's draw. In a spec you edit by hand, ``spec.with_randomize({...})`` inserts
  them the same way; a spec built from a config takes them the same way
  (``legacy.to_pipeline_spec(cfg).with_randomize({...})``; the configs have no
  ``randomize`` field).
- **What can be drawn.** A dynamic ``float32`` field. A static, ``int32`` or unknown
  field raises ``PlanError`` when the generator is built, listing the fields of the
  pipeline that can be drawn (with a hint where another key holds the axis:
  ``generator.size`` on a ``polar`` gate pipeline names ``normalize.size``). So do a
  randomizer listed after its target, two randomizers on one field, an initial range
  outside the field's bounds, and ``z_min`` / ``z_max`` randomizers that can cross
  (`Elevation knobs`_). :ref:`config-randomize` has every rule.
- **A drawn field belongs to its randomizer.** ``gen.params[stage][field]`` holds the
  current draws (read it to log what each env got), but ``set`` raises ``ValueError``
  and a direct write is overwritten at the next generation. To move the distribution,
  write the randomizer's own dynamic ``lo`` and ``hi``:
  ``gen.params["randomize.generator.polar_radial_jitter"].set("hi", 0.6)`` (a
  curriculum on a distribution). To change it in a spec, ``spec.replace(...)`` that
  randomizer; ``with_randomize`` refuses a field the spec already randomizes.
- **Cost.** Two small launches per randomizer per generation; none for a pipeline
  without one.

.. _pipeline-dynamic-builtins:

The built-in dynamic knobs
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - stage id: stage
     - dynamic fields (bounds)
   * - ``generator``: ``bezier``
     - ``min_num_points`` (``int32``, ``[2, max_num_points]``), ``rad`` ``[0, 1]``,
       ``handle_clamp_frac`` ``>= 0``, ``size`` ``[0.1, 10]``
   * - ``generator``: ``hull``
     - ``min_num_points``, ``hull_displacement`` ``[0, 1]``, ``size``
   * - ``generator``: ``bezier`` / ``hull`` gates
     - ``size``
   * - ``generator``: ``polar`` (tracks and gates)
     - ``polar_radial_jitter`` ``[0, 0.85]``, ``polar_angular_jitter`` ``[0, 0.45]``;
       ``size`` (tracks)
   * - ``generator``: ``voronoi`` (tracks and gates)
     - ``voronoi_radial_variation`` ``[0, 0.85]``, ``voronoi_angular_jitter``
       ``[0, pi]``; ``size`` (tracks)
   * - ``generator``: ``checkpoint`` (tracks and gates)
     - ``checkpoint_radius_min_frac`` ``[0, 1]``, ``checkpoint_angle_jitter``
       ``[0, 1]``; ``checkpoint_turn_rate`` ``[0, pi]`` and ``size`` (tracks)
   * - ``generator``: ``repulsive``
     - ``size``
   * - ``normalize``: ``extent`` (the ``polar`` / ``voronoi`` / ``checkpoint`` gate
       pipelines)
     - ``size``
   * - ``gate_frames``: ``default`` (every built-in gate pipeline)
     - ``gate_width`` ``>= 0`` (the full bar width)
   * - ``elevation``: ``uniform``
     - ``z_min``, ``z_max`` (unbounded)
   * - ``elevation``: ``random_walk``
     - ``z_min``, ``z_max``, ``z_max_step`` ``>= 0``
   * - ``elevation``: ``noise``
     - ``z_min``, ``z_max``, ``z_noise_amplitude`` ``>= 0``
   * - ``width``: ``from_channel``
     - ``scale``, ``lo``, ``hi``

The elevation stages are the same classes on a track, on gates
(:ref:`pipeline-gate-elevation`) and on the centerline before the stations
(:ref:`pipeline-stations-xyz`), so the same keys (``"elevation.z_max"``) work in all
three. ``external`` has no dynamic field: it ingests your geometry as given.

Curricula: writing the live arrays
----------------------------------

``gen.params[stage][field]`` is the live ``[E]`` array of a dynamic field, at a stable
address. Write it with ``gen.params[stage].set(field, value)`` (a scalar for every env,
or an ``[E]`` array of the field's dtype), or from a kernel recorded in your own graph
before the generation: the next generation reads it, for the envs it regenerates. A
curriculum kernel reads your training signal and writes the difficulty, inside the graph
that also regenerates the tracks:

.. code-block:: python

   import warp as wp
   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   @wp.kernel
   def difficulty_k(success: wp.array(dtype=wp.float32),
                    jitter: wp.array(dtype=wp.float32), z_max: wp.array(dtype=wp.float32)):
       e = wp.tid()
       level = wp.clamp(success[e], 0.0, 1.0)
       jitter[e] = 0.3 + 0.55 * level          # rounder tracks first, then eccentric ones
       z_max[e] = 0.1 + 0.4 * level            # then bigger hills

   spec = presets.track(
       num_envs=E, device=device, generator="polar",
       elevation=("noise", {"z_min": -0.3, "z_max": 0.3, "z_noise_amplitude": 0.1}),
       randomize={"elevation.z_noise_amplitude": (0.0, 0.3)},   # drawn: not written below
   )
   gen = TrackGenerator(spec)
   jitter = gen.params["generator"]["polar_radial_jitter"]     # [E] float32, stable
   z_max = gen.params["elevation"]["z_max"]
   success = wp.zeros(E, dtype=wp.float32, device=device)      # your training signal

   with wp.ScopedCapture(device=device) as capture:
       wp.launch(difficulty_k, dim=E, inputs=[success, jitter, z_max])
       gen.generate(advance=True)
   success.fill_(1.0)
   wp.capture_launch(capture.graph)        # each replay reads the new values

.. snippet: after

   assert abs(float(gen.params["generator"].in_effect("polar_radial_jitter").numpy()[0])
              - 0.85) < 1e-6
   assert abs(float(gen.params["elevation"].in_effect("z_max").numpy()[0]) - 0.5) < 1e-6

The same works from the host between generations
(``gen.params["generator"].set("polar_radial_jitter", 0.6)``), and on ``cuda`` in the
generator's own CUDA graph or in yours (:ref:`capture-rules`). A field you write must not
be drawn by a randomizer (above): this generator draws only the hill amplitude and leaves
``polar_radial_jitter`` and ``z_max`` to the curriculum kernel.

**One rule for bounds.** A dynamic built-in knob declares its bounds (`The built-in
dynamic knobs`_). A **scalar** is checked on the host: a value outside the bounds raises
``ValueError``, at construction, from ``set``, and for a randomizer's initial range
(``PlanError``). The comparison is made in ``float32``, the precision the array holds, so
a value read back from an array is always accepted. An **array** value (an ``[E]``
``set``, your own kernel, a randomizer's draw from bounds you wrote) is not checked: the
kernel clamps it to the bounds when it reads it (a NaN passes through, and an unbounded
side, such as ``z_min``'s or ``handle_clamp_frac``'s upper one, is not clamped). So a
curriculum may overshoot a bound without failing; the excess piles onto the bound.
``gen.params[stage].bounds(field)`` returns the ``(lo, hi)`` a value is held to (``None``
for an open side), narrowed per generator where the static configuration narrows it:
``min_num_points``' are ``(2, max_num_points)``, and ``set`` refuses a scalar outside
them (and a float: it is an ``int32``).

**Check the yield at both ends of every range.** Many knobs lose yield, or stop
mattering, well inside their bounds: ``rad`` has no effect above ``handle_clamp_frac``
(drawing both independently leaves ``rad`` inert in every env where it lands at or above
the clamp: keep ``hi(rad) <= lo(handle_clamp_frac)``; a ``generator.rad`` randomizer
whose ``hi`` exceeds every value the clamp can take emits a ``UserWarning`` when the
generator is built), ``voronoi_angular_jitter`` is legal up to ``pi`` but its yield is at
most 0.85 from ``0.5`` (0.71-0.85 by layout) and near zero from ``1.2``,
``checkpoint_radius_min_frac`` at ``0.99`` keeps two thirds of the tracks. Measure the
:term:`invalid env` fraction at each end (:ref:`invalid-envs`).

**With a stage that runs on the host** (a plugin generator or relax stage), the new
values are read at the start of the generation's first recordable part: a write must
come before it to apply to that generation (:ref:`pipeline-segments`).

.. _pipeline-in-effect:

.. _tuning-in-effect:

What each env was built with: ``in_effect`` and ``bounds``
----------------------------------------------------------

At the start of a generation the generator copies the live arrays into the working
copies its kernels read, for the envs it regenerates only (:ref:`how <pipeline-latch>`). Under
``generate(advance=True, mask=m)`` an env left out keeps the value its current track was
built with, so the live array may already hold a newer value than that env's track.
``gen.params[stage].in_effect(field)`` answers "which difficulty is each env running?":
the ``[E]`` values the stage **read** for each env's current rows (the drawn values, for
a field a randomizer draws):

.. code-block:: python

   import warp as wp
   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   gen = TrackGenerator(presets.track(num_envs=E, device=device))
   done = wp.array([1] + [0] * (E - 1), dtype=wp.int32, device=device)   # your done mask
   my_log = wp.zeros(E, dtype=wp.float32, device=device)

   p = gen.params["generator"]
   p.set("size", 1.5)                           # the curriculum's next level
   gen.generate(advance=True, mask=done)        # only the done envs pick it up
   built = p.in_effect("size")                  # [E]: what each env's current track read
   p.in_effect("size", out=my_log)              # or: one copy into your own [E] array
   p.bounds("size")                             # (0.1, 10.0): what a built-in clamps to

.. snippet: after

   b = built.numpy()
   assert b[0] == 1.5 and b[1] == 1.0 and my_log.numpy()[0] == 1.5

- **A view, or a copy.** Without ``out=`` it returns the internal array: no copy, a
  stable pointer, usable in your graph. Read it, never write it: a write (an in-place
  torch op on ``wp.to_torch(...)`` included) changes what the unmasked envs report, and
  what a randomizer redraws for them. With ``out=`` (a contiguous ``[E]`` array of the
  field's dtype on the generator's device; ``ValueError`` otherwise) it enqueues one copy
  into it, on the stream ``set`` would use (``stream=``, the ambient scope, the
  generator's stream), recordable in your graph, and returns ``out``. ``KeyError`` for a
  static or unknown field.
- **Values read, values used.** An array value is not range-checked, and the built-in
  generator, elevation, ``normalize`` and ``gate_frames`` stages clamp it to
  ``bounds(field)`` where they read it. So the value a built-in kernel **used** is
  ``clamp(in_effect(f), *bounds(f))``: equal to ``in_effect(f)`` for every in-range
  array, every scalar ``set`` (checked on the host) and every randomizer draw (clamped
  when drawn). Write in-range values, or book the clamp. Exceptions, used as written:
  ``width.from_channel``'s ``scale`` / ``lo`` / ``hi`` (it clamps its output), a
  randomizer's ``lo`` / ``hi`` (it clamps its draw), and, in
  a ``Course`` in gates mode, ``gate_frames.gate_width`` (its floor ``> 0`` is checked
  by ``set`` on a scalar only: an array value ``<= 0`` is used as ``0``).
- **When it is current.** The arrays are written on the generation's stream: read them
  on that stream (a later launch or node of the same graph, ``out=`` there) or after
  waiting for the generation. A drawn field behind a stage that runs on the host is
  current once that part of the generation has run (until then the array holds the
  previous generation's draws). Between a ``set`` and the next generation,
  ``in_effect`` returns the old values; before any generation, the initial value.

A drawn field is redrawn for every env, but an env left out of a masked generation keeps
its effective seed and its ``lo`` / ``hi``, so it redraws the same bits: ``in_effect``
of a drawn field is the draw each env's track was built with too.

**What stays static, and how to vary it anyway.** Mix generators or configurations in
one batch (:ref:`results-mixed-batches`), each writing its own env range of one
``Track``, or write a plugin generator whose knobs are ``plugin.dynamic`` (with declared
``lo`` / ``hi``, :ref:`generator-runtime-params`).

.. _tuning-difficulty-axes:

Difficulty axes and the knobs that reach them
---------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Axis
     - ``randomize=`` key, by generator (or how else to reach it, or why it cannot be
       reached)
   * - Curviness, corner sharpness
     - ``generator.rad`` (lower: sharper) and ``generator.handle_clamp_frac`` (bezier);
       ``generator.checkpoint_turn_rate`` (higher: sharper kinks; checkpoint)
   * - Irregularity
     - ``generator.hull_displacement`` (hull); ``generator.polar_angular_jitter``
       (polar); ``generator.voronoi_angular_jitter`` (voronoi, up to about ``0.5``);
       ``generator.checkpoint_angle_jitter`` (checkpoint)
   * - Eccentricity, inlets
     - ``generator.polar_radial_jitter`` (polar); ``generator.voronoi_radial_variation``
       (voronoi); ``generator.checkpoint_radius_min_frac`` (lower: deeper; checkpoint);
       ``generator.hull_displacement`` (pinches; hull)
   * - Hills
     - ``elevation.z_max`` with ``z_min`` fixed (``uniform``);
       ``elevation.z_noise_amplitude`` in a wide symmetric window (``noise``). Their
       grades fall as ``~1 / size``: scale the height with the size to keep them
       (`Size: the sizing rule`_)
   * - Grade
     - ``elevation.z_max_step`` in a wide window (``random_walk``; a grade cap, the same
       at every size)
   * - Course size
     - ``generator.size`` (``bezier``, ``hull``, ``polar``, ``voronoi``,
       ``checkpoint``, ``repulsive``; the ``bezier`` / ``hull`` gate generators and the
       track generator of ``source="stations"`` gates); ``normalize.size`` (the
       ``polar``, ``voronoi`` and ``checkpoint`` gate generators); none for
       ``external`` (scale your ``centerline.xy``). Higher: a longer course relative to
       the absolute road width; size ``n_max`` for its ``hi`` (`Size: the sizing
       rule`_)
   * - Corner count
     - ``set`` or your kernel only (``int32``): ``bezier`` / ``hull``
       ``min_num_points`` (higher: more corners). It narrows the count toward the
       static ``max_num_points`` only (:ref:`bezier-corner-count`): build with
       ``max_num_points`` at the hardest count, and size ``n_max`` for its longer loops
       (`Size: the sizing rule`_; the build-time warning scales its nominal loop by
       ``(max_num_points / 13) ** 0.6``); the knot, anchor and checkpoint counts size
       buffers
   * - Corner asymmetry
     - not reachable per env: ``bezier``'s ``edgy`` is static (transformed on the host)
   * - Gate bar width
     - ``gate_frames.gate_width`` (every built-in gate generator): the opening the agent
       passes through. Wider: easier to pass through, but fewer valid courses (the bars
       cross; a larger size restores the yield, :ref:`gate-width-invalid-fraction`);
       narrower: harder. Keep it ``> 0`` for an agent (a width-0 gate can never be
       passed). Its crossings are checked per env (:ref:`gate-width-per-env`)
   * - Gate count
     - not reachable per env: a gate generator's ``min_num_points`` / knot, anchor or
       checkpoint count is its gate capacity (``min_gates`` / ``max_gates``)

The gate pipelines take the same keys for the polar, voronoi and checkpoint ring knobs
and the elevation (``presets.gates(randomize=...)``).

Elevation knobs
~~~~~~~~~~~~~~~

The window ``[z_min, z_max]`` clips every profile, so the knob that sets the difficulty
depends on the profile (:ref:`tracks-25d-per-env`):

.. list-table::
   :header-rows: 1
   :widths: 18 36 46

   * - Profile
     - Difficulty knob
     - Window
   * - ``uniform``
     - ``z_max`` (higher: bigger hills)
     - keep ``z_min`` fixed (at the ground) and draw or raise ``z_max``
   * - ``random_walk``
     - ``z_max_step`` (higher: steeper grades)
     - a wide window: at ``z_min == z_max`` (the default ``[0, 0]``) the walk cannot
       move, and ``z_max_step`` has no effect
   * - ``noise``
     - ``z_noise_amplitude`` (higher: bigger hills)
     - a wide window **symmetric around** ``z_base`` (``z_min = -z_max`` at
       ``z_base = 0``): the default ``[0, 0]`` makes the amplitude inert, and ``[0,
       z_max]`` clips the negative half of every hill

When the generator is built, an inert window emits a ``UserWarning``: ``z_min == z_max``
with neither drawn, while ``z_max_step`` (``random_walk``) or ``z_noise_amplitude``
(``noise``) can be positive (its configured value, or the ``hi`` of its randomizer). A
window drawn by randomizers is open and does not warn; a window you open only at run time
(``set``) cannot be seen then, so a curriculum starting flat gets the warning.
Randomizers on ``z_min`` and ``z_max`` that could cross (``hi`` of ``z_min`` above ``lo``
of ``z_max``, or one drawn range crossing the other's fixed value) raise ``PlanError``
when the generator is built. A crossing you write yourself at run time is not refused:
``uniform`` then draws in ``[z_max, z_min]``, ``random_walk`` and ``noise`` return
``z_max`` everywhere.

.. _tuning-size:

Size: the sizing rule
---------------------

``size`` is a per-env factor on the core ``scale``, applied where each generator applies
``scale``: the corner grid of ``bezier`` and ``hull`` (and their gate generators), the
normalization target of ``polar``, ``voronoi`` and ``checkpoint`` (on gates, the
``normalize.extent`` stage), the output of ``repulsive`` (whose ``scale`` sets its
physics domain instead). At ``size = 1`` every output is byte-identical to a pipeline
without it; ``bezier`` / ``hull`` at ``scale = 1`` and ``size = s`` are bitwise ``scale =
s``, the normalized generators within one ``float32`` rounding of it. ``external`` has
no ``size``: it ingests your geometry as given (a size would move a circuit ingested in
world coordinates about the data's origin); scale ``centerline.xy`` in your own graph.

- **Initial value**: ``presets.track(size=)`` / ``presets.gates(size=)``,
  ``TrackGenConfig.size`` / ``GateGenConfig.size``; per env:
  ``randomize={"generator.size": (lo, hi)}`` (or ``normalize.size``),
  ``gen.params["generator"].set("size", ...)``, your kernel. It is not the batch size
  (``num_envs``).
- **A route that cannot apply it raises** ``ValueError`` instead of ignoring it: a flat
  ``size != 1`` with a generator that declares no ``size`` (``external``, a plugin whose
  Params lack it, a ``GeneratorSpec``), ``presets.track(generator="external",
  size=2.0)`` or ``size=`` together with ``generator_params={"size": ...}``.

**Plan view only.** ``size`` scales xy. The widths (``half_width``, ``spacing``, the relax
bands, ``gate_radius``, ``gate_width``) and z (``z_min`` / ``z_max``, the noise
amplitude, ``z_valid_grade``) are absolute: ``size`` is the "course larger relative to
the car" axis. So:

- under constant spacing a larger course needs more points, while smaller courses have
  less room for the same absolute road width;
- ``elevation.uniform`` places its knots at arc fractions and ``noise`` runs its harmonics
  in normalized arc length, so stretching the same altitude profile over a larger
  course reduces its grade. Scale altitude ranges/amplitude with size when you want
  similar slopes.
  ``random_walk``'s ``z_max_step`` is a grade cap per unit arc and does not depend on
  the size. On gates the spacing grows with the size, so the ``z_valid_grade`` check
  passes more envs. A joint size and hills curriculum is therefore not orthogonal.

**Truncation.** Size changes no buffer and no launch. The constant-spacing resample emits
``floor(size * L / spacing) + 1`` points (``L`` the raw perimeter at size 1) into
``n_max`` slots; a track that needs more is flagged in ``Track.truncated`` and fails
validity (:ref:`invalid-envs`). A gate pipeline with ``source="relaxed_stations"`` fails
the same way (its centerline is resampled into ``n_max``); anchor-sourced gates have a
fixed count and are not affected (their spacing grows). **The sizing rule**, for a size range up to ``size_hi`` (``L_max`` measured on your configuration with ``gen.program.tap("raw.xy")``):

1. ``n_max >= size_hi * L_max / spacing + 1`` (else truncation, an invalid env);
2. ``num_points >= size_hi * L_max / spacing``: the raw loop keeps its static
   ``num_points``, so a larger course can expose the raw polyline's resolution. Use
   enough raw points for the highest-curvature shape and largest size you need;
3. the utilities' capacities and resolutions (`Size and the utilities`_).

.. code-block:: python

   import warnings
   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   with warnings.catch_warnings(record=True) as caught:
       warnings.simplefilter("always")
       TrackGenerator(presets.track(num_envs=E, device=device,
                                    randomize={"generator.size": (1.0, 3.0)}))
   print(caught[0].message)   # generator.size=3.0 truncates at n_max=384 (...): raise n_max to >= 435 ...

   gen = TrackGenerator(presets.track(num_envs=E, device=device, n_max=448,
                                      randomize={"generator.size": (1.0, 3.0)}))

.. snippet: after

   assert any("truncates at n_max=384" in str(w.message) for w in caught)

**Build-time warning.** When the generator is built, a ``generator.size`` randomizer whose
``hi`` (or a construction value) would truncate the generator's nominal longest loop
emits one ``UserWarning`` naming the ``n_max`` that fits. The nominal loop is exact for
``repulsive`` (from its parameters); for the others it uses a nominal bound, scaled by the knobs that
lengthen it: ``bezier`` / ``hull``'s ``max_num_points`` (``(max_num_points / 13) **
0.6`` above 13), ``hull_displacement`` (its initial value or the ``hi`` of its
randomizer: ``1 + 0.6 * (d - 0.15)`` above 0.15). It is not checked when ``polar``'s ``polar_num_knots`` /
``polar_radial_jitter``, ``voronoi``'s ``voronoi_num_sites`` /
``voronoi_control_points`` / ``voronoi_angular_jitter``, or ``checkpoint``'s
``checkpoint_count`` / ``checkpoint_radius_min_frac`` / ``checkpoint_angle_jitter`` /
``checkpoint_steer_gain`` / ``checkpoint_lookahead_frac`` / ``checkpoint_best_of_k``
differ from their defaults or are drawn (the other knobs keep the check), nor for a
plugin generator. The longest loop is the maximum of 8192 envs: a larger batch can have
a longer one, so a few of its envs can truncate just below the threshold. It is a
heuristic, so a warning, never an error, and a curriculum written with ``set`` or from
your graph is not checked: size ``n_max`` for its largest value yourself (measure
``L_max`` with ``gen.program.tap("raw.xy")``).

.. _tuning-size-utilities:

Size and the utilities
~~~~~~~~~~~~~~~~~~~~~~

The runtime utilities size some buffers and grids once. A size curriculum that starts
small and grows must not outrun them while the tracks stay valid:

- **Checkpoint and prop rings.** ``Course`` and the components ``tools.Checkpoints`` /
  ``tools.Props`` size their rings from the longest track the generator can produce,
  with no readback (:ref:`course-checkpoint-capacity`):

  .. code-block:: text

     max_checkpoints = max(3, ceil(max_loop_length / checkpoint_spacing))
     max_props       = max(3, ceil(max_loop_length / prop_spacing))
     max_loop_length = 1.5 * n_max * spacing + dZ

  A track that fits ``n_max`` has a plan-view perimeter below ``n_max * spacing``; the
  ``1.5`` covers the relax growth and a boundary's extra ``~2 pi * half_width``
  (measured: at most 0.90 of ``n_max * spacing`` just below truncation), and ``dZ`` the
  elevation's climbs, from the declared ranges of the elevation knobs (``0`` when flat).
  At the defaults that is 70 checkpoints at spacing 0.5, at every size. A generator with
  no declared elevation range (``z_min`` / ``z_max`` declare none; ``z_valid_grade > 0``
  gives one), with a variable width, one that provides its own altitudes (no grade) or
  half-widths, or a plugin geometry stage declares no bound: there ``Course`` reads its
  construction batch back, and the components need an explicit capacity. The legacy
  ``CheckpointSampler(track, spacing)`` / ``PropSampler(track, spacing)`` with ``max_* =
  None`` also read the batch back (``ceil(1.5 * longest valid perimeter / spacing)`` over
  the bound batch). A readback taken at size 1 clips the grown tracks' rings
  (``truncated``), and progress and lap rewards read them (``bezier``, ``E = 32``, seeds
  ``7919 * e + 13``, checkpoint spacing 0.5, 20 checkpoints derived at size 1: truncated
  on 0 / 0 / 44 / 94 % of the valid envs at size 1 / 1.5 / 2 / 2.5): pass the capacity
  explicitly there, or construct at ``size = size_hi`` and ``set`` the size down.
- **SDF and heightfield grids.** The ``sdf`` collision backend and the heightfield baker
  cover each env's bounding box with a fixed ``R x R`` grid, so the cells grow with the
  size while the road stays absolute (``bezier``, ``R = 128``: 12.3 / 6.7 / 4.5 cells
  across the road at size 1 / 2 / 3; OOB flags agree with ``segments`` on 99.93 / 99.80
  / 99.68 % of the boxes). Collision accuracy then varies per env with the difficulty.
  Rule: ``sdf_resolution`` / ``heightfield_resolution >= R_ref * size_hi``, ``R_ref`` the
  resolution accurate enough at size 1; memory grows as ``size_hi^2``. Or use the exact
  ``segments`` backend (:ref:`collision-backends`). The components
  ``tools.BandCollision(method="sdf", size_hi=...)`` and ``tools.HeightFieldBake(...,
  size_hi=...)`` warn at ``attach`` when the rule is broken
  (:ref:`collision-resolution-check`). Their default ``size_hi="auto"`` reads the
  generator's declared range (a ``generator.size`` randomizer's construction ``hi``, else
  the construction ``size``), so a ``Course`` built with ``Course.track`` or
  ``Course(generator, components)`` warns (:doc:`/utilities/composing-a-course`);
  ``Course(CourseConfig)`` declares no ``size_hi`` and does not warn.

.. _curricula-course:

A course curriculum
-------------------

A :term:`course` drives its generator's dynamic fields like any generator:
``course.generator.params[...]``. Build the course from a spec with the randomizers or
the initial values you want, then move them between episodes, before
``course.regenerate(...)``:

.. code-block:: python

   from track_gen import tools
   from track_gen.course import Course
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   spec = presets.track(num_envs=E, device=device,
                        randomize={"generator.size": (1.0, 2.0)})
   course = Course.track(spec, checkpoint_spacing=0.5, seeds=0,
                         collision=tools.BandCollision("sdf", sdf_resolution=256))

   # later, as the agents improve: widen the size range for the next regenerations
   course.generator.params["randomize.generator.size"].set("hi", 2.5)

.. snippet: after

   assert course.generator.params["randomize.generator.size"]["hi"].numpy()[0] == 2.5

The course is ready at construction, at the spec's sizes. At flat elevation the default
checkpoint capacity already holds every track that is not truncated, at any size
(`Size and the utilities`_). The SDF and height-field grids do depend on the size:
``BandCollision`` and ``HeightFieldBake`` default to ``size_hi="auto"``, the largest
size of the spec (a randomizer's construction ``hi`` on ``generator.size``, else the
generator's construction ``size``), and warn at ``attach`` when the resolution does not
cover it (``ceil(128 * size_hi)``: 256 above; :ref:`collision-resolution-check`). The
check reads the construction range only: a curriculum that widens it later, as above, is
not checked again, so pass the curriculum's final ``hi`` explicitly
(``tools.BandCollision("sdf", sdf_resolution=320, size_hi=2.5)``), or use
``collision="segments"``. With elevation and no declared z range, the checkpoint ring is
sized from the construction batch: pass ``max_checkpoints``, or construct at the largest
size and ``set`` it down.

A ``Course(CourseConfig)`` course declares no ``size_hi`` and never warns; its
generator's size is the config's ``TrackGenConfig.size``. The bar width of a gates
course is ``course.generator.params["gate_frames"]`` (:ref:`gate-width-per-env`). To
end an episode on a new course at the new difficulty, regenerate the done envs
(:ref:`rl-auto-reset`); ``in_effect`` tells which difficulty each env is running.
