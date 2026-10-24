Tuning guide
============

This page collects practical tuning advice for ``TrackGenConfig``. For the
complete parameter descriptions, see the :doc:`Configuration reference <reference>`.

Yield vs. diversity
-------------------

*Yield* is the fraction of environments that pass the post-relaxation validity
gate. *Diversity* is how varied the resulting track shapes are across a batch.
These two goals pull in opposite directions: the settings that reliably produce
valid tracks tend to constrain the shape space.

A few principles:

- **Raise ``relax_iters`` first.** The default of 50 Chebyshev-accelerated
  sweeps is calibrated for approximately 99.9 % yield at the library defaults
  (matching what 150 plain-Jacobi sweeps used to need). If yield is low, the
  most conservative fix is to increase ``relax_iters``; cost is linear in
  sweeps and it does not restrict shapes. See
  :doc:`/relaxation/convergence` for the measured yield-vs-sweeps curves.

- **Widen the generator's shape range for diversity.** Draw the generator's dynamic
  knobs per env with ``randomize=`` (`Per-env parameters: randomization and
  curricula`_): ``rad`` / ``handle_clamp_frac`` for the Bezier generator,
  ``polar_radial_jitter`` for polar, ``checkpoint_radius_min_frac`` for checkpoint, and
  so on, and ``generator.size`` for a per-env course size (`Size: the sizing rule`_). For
  the Bezier and hull generators, also widen ``[min_num_points, max_num_points]``. For
  the checkpoint generator, increase ``checkpoint_count``; for polar, ``polar_num_knots``.
  Wider shape ranges can introduce harder-to-relax geometries, so watch yield when you
  push these.

- **Mix generators across workers** rather than pushing a single generator to
  its limits. Each centerline generator (``"bezier"``, ``"hull"``, ``"polar"``,
  ``"voronoi"``, ``"checkpoint"``) produces a visually distinct family; using
  different configs across parallel workers is the lowest-friction path to a
  broad distribution.

- **Or mix them inside one batch.** Several generators can write disjoint env ranges
  of one shared ``Track`` (``out=big.view(e0, e1)``), recorded in one graph and read by
  one set of utilities: see :ref:`results-mixed-batches`.

- **Use the parameter explorer to iterate quickly.** The interactive Gradio UI
  (``.venv/bin/python -m viz.param_explorer``) shows valid-yield and quality stats over a
  full batch in real time, so shape/yield trade-offs are immediately visible.

.. _tuning-per-env:

Per-env parameters: randomization and curricula
-----------------------------------------------

A **dynamic** parameter is a per-env ``[E]`` array (``float32``, or ``int32`` for a
count) the kernels read at every generation. You can draw a ``float32`` one per env with
``randomize=``, or write any of them yourself (a curriculum), with no rebuild and no
re-capture. The dynamic fields are:

- the built-in generators' shape knobs: ``rad`` and ``handle_clamp_frac`` (``bezier``),
  ``hull_displacement`` (``hull``), ``polar_radial_jitter`` and
  ``polar_angular_jitter`` (``polar``), ``voronoi_radial_variation`` and
  ``voronoi_angular_jitter`` (``voronoi``), ``checkpoint_radius_min_frac``,
  ``checkpoint_angle_jitter`` and ``checkpoint_turn_rate`` (``checkpoint``). The polar,
  voronoi and checkpoint **gate** generators have the same ring knobs (not
  ``checkpoint_turn_rate``: they do not steer). Each generator page gives their bounds,
  effective range and measured yield (:ref:`generators-per-env`);
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
  tracks, on gates and on the centerline before the stations alike;
- ``width.from_channel``'s ``scale``, ``lo`` and ``hi``, and every field a plugin
  declares with ``plugin.dynamic``.

Everything else is static (``max_num_points``, knot, anchor and checkpoint counts,
``half_width``, ``spacing``, ``n_max``, ``edgy``, the core ``scale``, ``z_base``, the
relax and validity tolerances, the ``repulsive`` growth knobs, the gate generators'
``min_num_points``): build a new generator to change it. Naming a static field in
``randomize=`` raises ``PlanError`` listing the fields of the pipeline that can be drawn
(with a hint where another key holds the axis: ``generator.size`` on a ``polar`` gate
pipeline names ``normalize.size``).

**Randomize.** ``presets.track(randomize={"<stage id>.<field>": (lo, hi)})`` draws each
env's value uniformly in ``[lo, hi)`` at every generation (:ref:`config-randomize`). The
generator's stage id is ``"generator"``, the elevation's ``"elevation"``, in
``presets.track`` and ``presets.gates`` alike:

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   spec = presets.track(
       num_envs=4096, device="cuda:0", generator="polar",
       elevation=("noise", {"z_min": -0.3, "z_max": 0.3, "z_noise_amplitude": 0.1}),
       randomize={"generator.polar_radial_jitter": (0.3, 0.85),
                  "elevation.z_noise_amplitude": (0.0, 0.3)},
   )
   gen = TrackGenerator(spec)
   gen.params["generator"]["polar_radial_jitter"]   # [E]: what each env was drawn

For a pipeline built from a ``TrackGenConfig``, add the randomizers to its spec:
``TrackGenerator(legacy.to_pipeline_spec(cfg).with_randomize({...}), rng)``; from a
``GateGenConfig``:
``GateGenerator(legacy.to_gate_pipeline_spec(gcfg).with_randomize({...}), rng)``
(``PipelineSpec.with_randomize`` is what the presets call; the configs have no
``randomize`` field). A field the spec already randomizes is refused: to change its
range for a curriculum phase, ``spec.replace(...)`` that randomizer or write its live
``lo`` / ``hi`` (below).

**Curricula.** ``gen.params[stage][field]`` is the live ``[E]`` array of a dynamic field,
at a stable address. Write it with ``gen.params[stage].set(field, value)`` (a scalar for
every env, or an ``[E]`` ``wp.float32`` array), or from a kernel recorded in your own
graph before ``generate()``: the next generation reads it. A field you write must not be
drawn by a randomizer (below), so this curriculum's generator, unlike the one above,
draws only the hill amplitude and leaves ``polar_radial_jitter`` and ``z_max`` to the
curriculum kernel:

.. code-block:: python

   import warp as wp

   spec = presets.track(
       num_envs=E, device="cuda:0", generator="polar",
       elevation=("noise", {"z_min": -0.3, "z_max": 0.3, "z_noise_amplitude": 0.1}),
       randomize={"elevation.z_noise_amplitude": (0.0, 0.3)},   # not the jitter
   )
   gen = TrackGenerator(spec)
   rjit = gen.params["generator"]["polar_radial_jitter"]     # [E] float32, stable
   z_max = gen.params["elevation"]["z_max"]

   with wp.ScopedCapture(stream=s) as capture:
       wp.launch(difficulty_k, dim=E, inputs=[success_rate, rjit, z_max], stream=s)
       gen.generate(advance=True, stream=s)
   wp.capture_launch(capture.graph, stream=s)   # each replay reads the new values

A field drawn by a randomizer belongs to it: ``set`` raises and a direct write is
overwritten at the next generation (with the generator of the randomize example above,
the kernel's ``polar_radial_jitter`` write would be lost). Move the draw instead, by
writing the randomizer's bounds,
``gen.params["randomize.generator.polar_radial_jitter"]["hi"]`` (a curriculum on a
distribution; keep it in ``[lo, 0.85]``).

**One rule for bounds.** A dynamic built-in knob declares its bounds (each generator
page lists them). A **scalar** is checked on the host: a value outside the bounds raises
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
them.

**Check the yield at both ends of every range.** Many knobs lose yield, or stop
mattering, well inside their bounds: ``rad`` has no effect above ``handle_clamp_frac``
(drawing both independently leaves ``rad`` inert in every env where it lands at or
above the clamp: keep ``hi(rad) <= lo(handle_clamp_frac)``; a ``generator.rad``
randomizer whose ``hi`` exceeds every value the clamp can take, its drawn ``hi`` or its
value, emits a ``UserWarning`` when the pipeline is built),
``voronoi_angular_jitter`` is legal up to ``pi`` but its yield is at most 0.85 from
``0.5`` (0.71-0.85 by layout) and near zero from ``1.2``, ``checkpoint_radius_min_frac`` at ``0.99`` keeps two
thirds of the tracks. The yield-versus-diversity advice above applies to the whole
range, not its middle.

**Elevation: the knob that sets the difficulty depends on the profile.** The window
``[z_min, z_max]`` clips every profile, so:

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

When the pipeline is built, an inert window emits a ``UserWarning``: ``z_min == z_max``
with neither drawn, while ``z_max_step`` (``random_walk``) or ``z_noise_amplitude``
(``noise``) can be positive (its configured value, or the ``hi`` of its randomizer). A
window drawn by randomizers is open and does not warn; a window you open only at run
time (``set``) cannot be seen then, so a curriculum starting flat gets the warning.
Randomizers on ``z_min`` and
``z_max`` that could cross (``hi`` of ``z_min`` above ``lo`` of ``z_max``, or one drawn
range crossing the other's fixed value) raise ``PlanError`` when the pipeline is built.
A crossing you write yourself at run time is not refused: ``uniform`` then draws in
``[z_max, z_min]``, ``random_walk`` and ``noise`` return ``z_max`` everywhere (exactly
what the scalar kernels did).

.. _tuning-in-effect:

**Masked regeneration and bookkeeping: in_effect.** At the start of a generation the
Program copies (*latches*) the live arrays into the working copies the kernels read, for
the envs it regenerates only. Under ``generate(advance=True, mask=m)`` an env left out
keeps the value its current track was built with, so the live array may already hold a
newer value than that env's track. ``gen.params[stage].in_effect(field)`` answers "which
difficulty is each env running?": the ``[E]`` values the stage **read** for each env's
current rows (the working copy of a latched field; the drawn array of a field a
randomizer, or ``style_sampling``, draws):

.. code-block:: python

   p = gen.params["generator"]
   p.set("size", next_sizes)                    # the curriculum's next level, [E] float32
   gen.generate(advance=True, mask=done)        # only the done envs pick it up
   built = p.in_effect("size")                  # [E]: what each env's current track read
   p.in_effect("size", out=my_log)              # or: one copy into your own [E] array
   p.bounds("size")                             # (0.1, 10.0): what a built-in clamps to

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
  randomizer's ``lo`` / ``hi`` (it clamps its draw), ``style_sampling``'s draws, and, in
  a ``Course`` in gates mode, ``gate_frames.gate_width`` (its floor ``> 0`` is checked
  by ``set`` on a scalar only: an array value ``<= 0`` is used as ``0``).
- **When it is current.** The arrays are written on the generation's stream: read them
  on that stream (a later launch or node of the same graph, ``out=`` there) or after
  waiting for the generation. A latched field is current once segment ``0`` has run
  (the latch is its prologue); a drawn field once the segment holding its randomizer has
  run (behind a ``HOST`` stage that is a later segment, and until then the array holds
  the previous generation's draws). Between a ``set`` and the next generation,
  ``in_effect`` returns the old values; before any generation, the initial value.

A drawn field is redrawn for every env, but an env left out of a masked generation keeps
its effective seed and its latched ``lo`` / ``hi``, so it redraws the same bits:
``in_effect`` of a drawn field is the draw each env's track was built with too.

**Segmented Programs.** In a Program with a ``HOST`` stage (a ``HOST`` plugin generator
or relax stage), the latch runs at the start of segment ``0``. A write must be enqueued
before segment ``0`` to apply to that generation; a write recorded in a later segment's
graph applies one generation later (:ref:`pipeline-segments`).

**What stays static, and how to vary it anyway.** Mix generators or configurations in
one batch (:ref:`results-mixed-batches`), each writing its own env range of one
``Track``, or write a plugin generator whose knobs are ``plugin.dynamic`` (with declared
``lo`` / ``hi``, :ref:`generator-runtime-params`).

.. _tuning-difficulty-axes:

Difficulty axes and the knobs that reach them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Axis
     - ``randomize=`` key, by generator (or how else to reach it, or why it cannot be
       reached yet)
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
     - not reachable in 0.2.0.dev5: ``bezier``'s ``edgy`` (roadmap PR16)
   * - Gate bar width
     - ``gate_frames.gate_width`` (every built-in gate generator): the opening the agent
       passes through. Wider: easier to pass through, but fewer valid courses (the bars
       cross; a larger size restores the yield, :ref:`gate-width-per-env`); narrower:
       harder. Keep it ``> 0`` for an agent (a width-0 gate can never be passed). Its
       crossings are checked per env
   * - Gate count
     - not reachable in 0.x: a gate generator's ``min_num_points`` / knot, anchor or
       checkpoint count is its gate capacity (``min_gates`` / ``max_gates``)

The gate pipelines take the same keys for the polar, voronoi and checkpoint ring knobs
and the elevation (``presets.gates(randomize=...)``).

.. _tuning-size:

Size: the sizing rule
~~~~~~~~~~~~~~~~~~~~~

``size`` is a per-env factor on the core ``scale``, applied where each generator applies
``scale``: the corner grid of ``bezier`` and ``hull`` (and their gate generators), the
normalization target of ``polar``, ``voronoi`` and ``checkpoint`` (on gates, the
``normalize.extent`` stage), the output of ``repulsive`` (whose ``scale`` sets its
physics domain instead). At ``size = 1`` every output is byte-identical to a pipeline
without it; ``bezier`` / ``hull`` at ``scale = 1`` and ``size = s`` are bitwise ``scale =
s``, the normalized generators within one ``float32`` rounding of it. ``external`` has
no ``size``: it ingests your geometry as given (a size would move a circuit ingested in
world coordinates about the data's origin); scale ``centerline.xy`` in your own graph.

- **Initial value**: ``TrackGenConfig.size`` / ``GateGenConfig.size`` (the only
  construction-time route for ``Course``), ``presets.track(size=)`` /
  ``presets.gates(size=)``; per env: ``randomize={"generator.size": (lo, hi)}`` (or
  ``normalize.size``), ``gen.params["generator"].set("size", ...)``, your kernel. It is
  not the batch size (``num_envs``).
- **A route that cannot apply it raises** ``ValueError`` instead of ignoring it: a flat
  ``size != 1`` with a generator that declares no ``size`` (``external``, a plugin whose
  Params lack it, a legacy ``GeneratorSpec``), ``presets.track(generator="external",
  size=2.0)``, ``size=`` together with ``generator_params={"size": ...}``, and the
  frozen legacy module functions (``generate_*_warp``) given a config with ``size !=
  1``.

**Plan view only.** ``size`` scales xy. The widths (``half_width``, ``spacing``, the relax
bands, ``gate_radius``, ``gate_width``) and z (``z_min`` / ``z_max``, the noise
amplitude, ``z_valid_grade``) are absolute: ``size`` is the "course larger relative to
the car" axis. So:

- under constant spacing a larger course has more points (the rule below), and small
  sizes lose yield to the absolute road width (self-overlap after the offset): for
  ``bezier``, 0.945 valid at ``size = 0.5``;
- ``elevation.uniform`` places its knots at arc fractions and ``noise`` runs its harmonics
  in normalized arc length, so their **grades fall as** ``~1 / size`` (``uniform`` with
  ``z_max = 0.3``: median per-track maximum grade 0.666 / 0.337 / 0.225 at size 1 / 2 /
  3): scale ``z_max`` / ``z_noise_amplitude`` with the size to keep the grade.
  ``random_walk``'s ``z_max_step`` is a grade cap per unit arc and does not depend on
  the size. On gates the spacing grows with the size, so the ``z_valid_grade`` check
  passes more envs. A joint size and hills curriculum is therefore not orthogonal.

**Truncation.** Size changes no buffer and no launch. The constant-spacing resample emits
``floor(size * L / spacing) + 1`` points (``L`` the raw perimeter at size 1) into
``n_max`` slots; a track that needs more is flagged in ``Track.truncated`` and fails
validity (rows kept or poisoned by ``on_invalid``). A gate pipeline with
``source="relaxed_stations"`` fails the same way (its centerline is resampled into
``n_max``); anchor-sourced gates have a fixed count and are not affected (their spacing
grows). At the defaults (``n_max = 384``, ``half_width = 0.1``, spacing ``0.06``; CPU),
the longest loop (rounded up) and its p99 over 8192 envs (seeds ``0..4095`` and
``7919 * e + 13`` for ``e < 4096``) and the yields over ``E = 256`` (seeds
``7919 * e + 13``); ``repulsive``: both over ``E = 32``, seeds ``7919 * e + 13``:

.. list-table::
   :header-rows: 1
   :widths: 16 26 24 34

   * - generator
     - longest raw loop at size 1 (p99)
     - ``n_max = 384`` truncates from
     - valid at size 0.5 / 1 / 2 / 3
   * - ``bezier``
     - 8.68 (7.17)
     - ~2.6
     - 0.945 / 1.000 / 0.996 / 0.996
   * - ``hull``
     - 8.30 (6.90)
     - ~2.8
     - 0.828 / 0.992 / 1.000 / 1.000
   * - ``polar``
     - 6.24 (5.67)
     - ~3.7
     - 0.961 / 1.000 / 1.000 / 1.000
   * - ``voronoi``
     - 5.01 (4.78)
     - ~4.6
     - 0.996 / 1.000 / 1.000 / 1.000
   * - ``checkpoint``
     - 7.26 (6.18)
     - ~3.2
     - 0.902 / 0.992 / 0.996 / 0.992
   * - ``repulsive``
     - 14.48 (bound 14.83)
     - ~1.6
     - — / 1.000 / 0.000 (every env truncated) / —

**The sizing rule**, for a size range up to ``size_hi`` (``L_max`` from the table, times
``scale``, or measured on your configuration with ``gen.program.tap("raw.xy")``):

1. ``n_max >= size_hi * L_max / spacing + 1`` (else truncation, a failed env);
2. ``num_points >= size_hi * L_max / spacing``: the raw loop keeps its static
   ``num_points``, so its point spacing is ``size * L / num_points`` (``bezier``, 256
   points: median per-env largest raw segment 0.019 / 0.038 / 0.057 at size 1 / 2 / 3,
   against the output spacing 0.06); a coarser raw loop makes the constant-spacing
   resample interpolate a polyline;
3. the utilities' capacities and resolutions (`Size and the utilities`_).

**Build-time warning.** When the pipeline is built, a ``generator.size`` randomizer whose
``hi`` (or a construction value) would truncate the generator's nominal longest loop
emits one ``UserWarning`` naming the ``n_max`` that fits (``... raise n_max to >= 496 or
lower the range; truncated envs fail validity``). The nominal loop is exact for
``repulsive`` (from its parameters); for the others it is the longest loop of the table
(so the warning fires from the "truncates from" column), scaled by the knobs that
lengthen it: ``bezier`` / ``hull``'s ``max_num_points`` (``(max_num_points / 13) **
0.6`` above 13), ``hull_displacement`` (its initial value or the ``hi`` of its
randomizer: ``1 + 0.6 * (d - 0.15)`` above 0.15) and ``style_sampling``'s
``scale_range``. It is not checked when ``polar``'s ``polar_num_knots`` /
``polar_radial_jitter``, ``voronoi``'s ``voronoi_num_sites`` /
``voronoi_control_points`` / ``voronoi_angular_jitter``, or ``checkpoint``'s
``checkpoint_count`` / ``checkpoint_radius_min_frac`` / ``checkpoint_angle_jitter`` /
``checkpoint_steer_gain`` / ``checkpoint_lookahead_frac`` / ``checkpoint_best_of_k``
differ from their defaults or are drawn (the other knobs keep the check), nor for a
plugin generator. The longest loop is the maximum of 8192 envs: a larger batch can have
a longer one, so a few of its envs can truncate just below the threshold (fail-closed).
It is a heuristic, so a warning, never
an error, and a curriculum written with ``set`` or from your graph is not checked: size
``n_max`` for its largest value yourself (measure ``L_max`` with
``gen.program.tap("raw.xy")``).

.. _tuning-size-utilities:

Size and the utilities
~~~~~~~~~~~~~~~~~~~~~~

The runtime utilities size some buffers and grids once. A size curriculum that starts
small and grows must not outrun them while the tracks stay valid:

- **Checkpoint and prop rings.** Since 0.2.0.dev6 ``Course`` and the components
  ``tools.Checkpoints`` / ``tools.Props`` size their rings from the generator's plan, with
  no readback (:ref:`course-checkpoint-capacity`):

  .. code-block:: text

     max_checkpoints = max(3, ceil(max_loop_length / checkpoint_spacing))
     max_props       = max(3, ceil(max_loop_length / prop_spacing))
     max_loop_length = 1.5 * n_max * spacing + dZ

  A track that fits ``n_max`` has a plan-view perimeter below ``n_max * spacing``; the
  ``1.5`` covers the relax growth and a boundary's extra ``~2 pi * half_width``
  (measured: at most 0.90 of ``n_max * spacing`` just below truncation), and ``dZ`` the
  elevation's climbs, from the declared ranges of the elevation knobs (``0`` when flat).
  At the defaults that is 70 checkpoints at spacing 0.5, at every size. A plan without a
  declared elevation range (``z_min`` / ``z_max`` declare none; ``z_valid_grade > 0``
  gives one), with a variable width, a generator that provides its own altitudes (no
  grade) or half-widths, or a plugin geometry stage declares no bound: there
  ``Course`` reads its construction batch back, as before, and the components need an
  explicit capacity. The legacy ``CheckpointSampler(track, spacing)`` /
  ``PropSampler(track, spacing)`` with ``max_* = None`` also keep the readback
  (``ceil(1.5 * longest valid perimeter / spacing)`` over the bound batch). A readback
  taken at size 1 clips the grown tracks' rings (``truncated``), and progress and lap
  rewards read them (``bezier``, ``E = 32``, seeds ``7919 * e + 13``, checkpoint spacing
  0.5, 20 checkpoints derived at size 1: truncated on 0 / 0 / 44 / 94 % of the valid envs
  at size 1 / 1.5 / 2 / 2.5): pass the bound explicitly there, or construct with
  ``TrackGenConfig.size = size_hi`` and ``set`` the size down.
- **SDF and heightfield grids.** The ``sdf`` collision backend and the heightfield baker
  cover each env's bounding box with a fixed ``R x R`` grid, so the cells grow with the
  size while the road stays absolute (``bezier``, ``R = 128``: 12.3 / 6.7 / 4.5 cells
  across the road at size 1 / 2 / 3; OOB flags agree with ``segments`` on 99.93 / 99.80
  / 99.68 % of the boxes). Collision accuracy then varies per env with the difficulty.
  Rule: ``sdf_resolution`` / ``heightfield_resolution >= R_ref * size_hi``, ``R_ref`` the
  resolution accurate enough at size 1; memory grows as ``size_hi^2`` (the ~670 MB of
  128² grids at ``E = 8192`` become ~6 GB at ``size_hi = 3``). Or use the exact
  ``segments`` backend. The components ``tools.BandCollision(method="sdf",
  size_hi=...)`` and ``tools.HeightFieldBake(..., size_hi=...)`` warn at ``attach`` when
  the rule is broken (:ref:`collision-resolution-check`). Their default
  ``size_hi="auto"`` reads the plan's bound (a ``generator.size`` randomizer's
  construction ``hi``, else the construction ``size``), so a ``Course`` built with
  ``Course.track`` or ``Course(generator, components)`` warns
  (:doc:`/utilities/composing-a-course`); ``Course(CourseConfig)`` declares no
  ``size_hi`` and does not warn.

``half_width`` and ``spacing`` coupling
----------------------------------------

``spacing`` controls the arc-length step between adjacent output points.
``half_width`` controls the track width and feeds directly into the XPBD
relaxation geometry.

When ``spacing`` is left at its default (``None``), it is auto-coupled to
``0.6 * half_width``. This is the *relax-friendly* rule of thumb: at that
spacing each bead is roughly 0.6 road-widths apart, which gives the Jacobi
XPBD solver enough room to converge within the default iteration budget.

.. warning::

   A fixed ``spacing`` value does **not** scale with ``half_width``. If you
   change ``half_width`` without adjusting ``spacing``, the per-bead geometry
   shifts and the solver may under-converge. Unless you have a specific reason
   to override, leave ``spacing=None`` and let the auto-coupling handle it.

The ``N_max`` buffer must be large enough to hold the longest track in the
batch (at its largest ``size``: :ref:`tuning-size`). The formula for the required size
is::

    N_max >= floor(max_perimeter / spacing) + 1

A track whose real point count exceeds ``N_max`` is truncated: its closing segment
would span the dropped arc, so the pipeline flags it (``Track.truncated[e] = 1``) and
fails it (``Track.valid[e] = 0``). At the library
defaults (``half_width=0.1``, auto ``spacing=0.06``) the default ``N_max=384``
gives ample headroom (max ~141 points per track; full coverage down to
``half_width≈0.05``, ~281 max). For a thin-track
regime such as ``half_width=0.03`` (auto ``spacing=0.018``, ~468 max points per
track), raise ``N_max`` to at least 512 to avoid truncation-driven invalids.

Summary of the coupling rules:

- ``spacing = None`` → auto ``0.6 * half_width``.
- ``N_max`` must satisfy ``N_max >= max_perimeter / spacing + 1``; raise it when
  you lower ``half_width`` or ``spacing``.
- ``scale`` controls the track coordinate range; doubling ``scale`` roughly
  doubles perimeters, so ``N_max`` scales proportionally. The per-env ``size`` does the
  same per env: size ``N_max`` (and ``num_points``) for its largest value.
- Ingested centerlines (``generator="external"``) are in your data's units, not
  ``scale``'s: set ``half_width``, ``spacing`` and ``N_max`` for them explicitly.
  ``load_centerlines`` checks ``N_max >= perimeter / spacing + 1`` per env before
  uploading (:ref:`ingesting-centerlines`).

Relaxation knobs
----------------

The XPBD solve has three classes of knobs.

**Iteration count and solver constraints**

``relax_iters`` (default 50) is the number of Jacobi sweeps. Each sweep
applies three corrections to every bead:

1. Non-neighbour separation — pushes beads that are too close apart.
2. Edge spacing — restores the constant bead-to-bead rest length.
3. Bending/radius guard — removes sharp kinks introduced by separation pushes.

**Chebyshev acceleration**

``relax_accel`` (default ``"chebyshev"``) accelerates the Jacobi sweeps with
the Chebyshev semi-iterative method: after ``relax_cheby_start`` (default 8)
plain warmup sweeps, each sweep blends the Jacobi update with the current and
previous iterates using a precomputed omega schedule driven by
``relax_cheby_rho`` (default 0.98) and damped by ``relax_cheby_gamma``
(default 0.9). This is why the default sweep count is 50 rather than 150.
``relax_accel="none"`` restores plain Jacobi. The defaults are a measured
operating point — ``relax_cheby_rho >= 0.99`` or ``relax_cheby_start <= 5``
were measured to *reduce* yield. For the full account (mechanics, measured
tables, and dead ends) see :doc:`/relaxation/solver` and
:doc:`/relaxation/convergence`.

``relax_margin`` (default 0.15) expands the separation target beyond the exact
road diameter:

.. code-block:: python

    target = 2 * half_width * (1 + relax_margin)

A 15 % margin leaves room for the downstream thickness validity gate (the gate
checks that minimum thickness is at least ``(1 - relax_tol) * half_width``).
Raising it buys yield but was measured to cost 20–40 % spacing-RMS quality —
prefer raising ``relax_iters`` instead. Decreasing it tightens the packing but
produces more near-threshold invalids; see the gate-coupling discussion in
:doc:`/relaxation/constraints`.

``relax_tol`` (default 0.02) is the validity thickness tolerance: the minimum
accepted per-point half-width is ``(1 - relax_tol) * half_width``.

**Smoothing tail**

After the main sweeps, a short post-solve smoothing tail runs:
``relax_smooth_passes`` (default 5) shrink-free Taubin smoothing passes followed
by ``relax_smooth_spacing_iters`` (default 10) spacing-only polish sweeps. This
strips the fine sub-``R_min`` curvature noise the bending guard's deadband cannot
see, cutting curvature-difference RMS by roughly a third with no validity loss.
Set both to 0 to disable the tail. For the mechanics (Taubin ``lambda``/``mu``,
why a plain Laplacian is *not* substituted, and the graph-capture note) see
:doc:`/relaxation/solver`.

**Per-constraint relaxation factors**

Three scale factors control how aggressively each correction is applied per
sweep:

- ``relax_sep_relax`` (default 1.0) — separation correction scale. Reduce
  toward 0 to soften separation enforcement (useful if the solver oscillates).
- ``relax_spc_relax`` (default 1.0) — edge-spacing correction scale.
- ``relax_bend_relax`` (default 1.5) — bending/radius-guard correction scale.
  The default slightly over-corrects to quickly remove sharp corners; reduce if
  the solve oscillates.

**Neighbour exclusion band**

``relax_band`` (default ``None``) sets the bead-index exclusion band for
separation: beads within band positions of each other along the chain are not
separated. ``None`` auto-computes ``round(2 * half_width / L0)`` per track,
where ``L0`` is the rest edge length. Override it only if you observe the
road trying to push adjacent beads apart — and never *widen* it to save
separation work: auto-band + 1 was measured to collapse defaults-regime yield
from 0.99 to 0.74 (see the gate coupling in :doc:`/relaxation/constraints`).

Separation-cache throughput recipe
-----------------------------------

The dominant cost in the XPBD solve is the non-neighbour separation scan, which
is ``O(count[e]^2)`` per track per sweep in the dense baseline. Three advanced
knobs expose a graph-capturable broadphase cache:

.. list-table:: Separation-cache knobs
   :header-rows: 1

   * - Setting
     - Meaning
     - Default
   * - ``relax_sep_every``
     - Broadphase refresh interval *K*. Without a cache this is a naive skip
       cadence; with cache enabled it rebuilds candidates every *K* sweeps.
       Keep it well below ``relax_iters`` (see the warning below).
     - ``20``
   * - ``relax_sep_cache_slots``
     - Fixed candidate capacity per bead. ``0`` disables caching. Larger values
       use more memory and narrowphase work but reduce candidate-overflow risk.
     - ``16``
   * - ``relax_sep_cache_skin``
     - Extra broadphase radius as a fraction of ``target``: cache radius is
       ``target * (1 + skin)``. The exact separation push is still applied only
       for ``dist < target``.
     - ``0.5``

Cached mode is active when ``relax_sep_cache_slots > 0`` **and**
``relax_sep_every > 1``. In cached mode, ``_build_sep_cache_kernel`` rebuilds a
fixed-size per-bead candidate list every *K* sweeps using the broadphase radius.
``_step_cached_kernel`` then runs every sweep, re-testing each cached candidate
with the exact narrowphase ``dist < target`` before applying the push. This is
fundamentally different from the naive cadence path (``relax_sep_cache_slots=0,
relax_sep_every > 1``), which simply skips separation on intermediate sweeps.

``skin=0.0`` stores only pairs already inside the exact separation target at
refresh time. Positive skin retains near-pairs that may enter contact during a
long refresh interval, which is safer for large *K* but slower.

.. warning::

   The candidate cache is built at sweep 0 and refreshed only at multiples of
   ``relax_sep_every``. A run shorter than ``relax_sep_every`` sweeps never
   refreshes it mid-solve and silently misses contacts that emerge during the
   run — under the old default of 40 this produced a measured yield cliff
   between 40 and 50 sweeps. That is why the default dropped to 20 alongside
   the ``relax_iters`` 150 → 50 change. If you shorten ``relax_iters``, keep
   ``relax_sep_every`` comfortably below it. See
   :doc:`/relaxation/convergence`.

A high-throughput config from the ``E=8192``, ``half_width=0.03``,
``relax_iters=150`` CUDA graph benchmark (note the long 150-sweep budget is
what makes ``relax_sep_every=40`` safe there):

.. code-block:: python

    TrackGenConfig(
        ...,
        relax_sep_every=40,
        relax_sep_cache_slots=16,
        relax_sep_cache_skin=0.0,
    )

On the benchmark machine this ran in approximately **0.066 s** per graph replay
versus **0.366 s** for the dense baseline (``relax_sep_cache_slots=0``), with
effectively unchanged validity in the checked runs.

.. note::

   Keep the dense baseline (``relax_sep_cache_slots=0``) for maximum
   conservatism. Use the cache knobs when throughput matters and validate yield
   in your target regime before committing to the cached config.

Per-generator tuning tips
--------------------------

Bezier (``generator="bezier"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``rad`` (default 0.4) is the primary curviness dial — the Bezier handle
  length as a fraction of each segment chord. Higher values produce rounder
  corners; very high values combined with a small ``handle_clamp_frac`` can
  introduce self-crossings that fall back to the polygon rescue path.
- ``handle_clamp_frac`` (default 0.4) prevents handles from overshooting nearby
  corners. Setting it equal to ``rad`` (the default) means the clamp only trims
  genuine overshoot; setting it below ``rad`` makes every handle respect the
  shorter adjacent edge.
- ``min_num_points`` / ``max_num_points`` (default 9–13) control the corner
  count per env. A wider range increases diversity; many corners with high
  ``rad`` can push more tracks into the polygon fallback. ``min_num_points`` is
  per env (``int32``): a curriculum raises it toward ``max_num_points``
  (:ref:`bezier-corner-count`).
- To let each env draw its own style, randomize ``rad``, ``handle_clamp_frac`` and
  ``size`` (``randomize={"generator.rad": (0.1, 0.4), "generator.size": (0.8, 1.2)}``;
  keep ``rad``'s range at or below the clamp, the lowest clamp draw if you draw it too).
  ``style_sampling`` and its ranges are deprecated: they map one to one onto these keys
  (:ref:`bezier-style-vs-randomize`).

Hull (``generator="hull"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``hull_displacement`` (default 0.15) is the main shape dial — the maximum
  radial midpoint displacement as a fraction of the centroid-to-midpoint
  distance. At 0.0 the method collapses toward a plain angle-sorted loop; larger
  values (typical range 0.1–0.4) create stronger lobes and straights, but can
  increase the downstream relaxation burden.

Polar (``generator="polar"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``polar_num_knots`` (default 12) controls how many frequency components the
  spline has. More knots add higher-frequency features.
- ``polar_radial_jitter`` (default 0.60) is the fractional radial jitter per
  knot. Larger values produce more eccentric shapes.
- ``polar_angular_jitter`` (default 0.30) adds angular irregularity. It is
  bounded to ``[0, 0.45]`` so knots stay in sorted angular order
  (``polar_radial_jitter`` to ``[0, 0.85]``): a value outside raises ``ValueError``.
- The polar generator has no local fallback: any rare bad geometry is handled
  by the shared post-relaxation validity gate, so unusually high jitter at low
  ``relax_iters`` may reduce yield.

Voronoi (``generator="voronoi"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``voronoi_site_layout`` (default ``"void_ring"``) selects the site distribution.
  ``"void_ring"`` biases sites into an annular band with a center void, which
  tends to produce usable closed loops. ``"ring"`` fills the bounding box
  uniformly (not an annular band despite the name). ``"clustered"`` and
  ``"mixed"`` create more complex layouts.
- ``voronoi_control_points`` (default 18) sets how many angular anchors are
  selected from the site field. More control points add local features and can
  raise the relaxation burden.
- ``voronoi_radial_variation`` (default 0.62) controls how eccentric the anchor
  ring is. At 0 all anchors are placed at a uniform annulus radius.
- ``voronoi_angular_jitter`` (default 0.08, radians) breaks the rotational regularity.
  Keep it below about ``0.3`` for full yield: it drops to 0.71-0.85 at ``0.5`` and near
  zero from ``1.2``.
- Higher ``voronoi_num_sites`` enriches the anchor snap targets but increases
  the all-sites scan cost per bead.

Checkpoint (``generator="checkpoint"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``checkpoint_count`` (default 12) controls the number of radial checkpoints.
  More checkpoints produce wavier, chicane-rich tracks; fewer produce calmer,
  rounder shapes.
- ``checkpoint_best_of_k`` (default 4) generates *K* decorrelated candidates
  per env and keeps the one with the fewest self-intersections. Increasing *K*
  reduces pre-relaxation self-crossings at the cost of extra generation work.
- ``checkpoint_turn_rate`` (default 0.42 rad/step) bounds the path curvature.
  Too large allows kinks; too small prevents the path from tracking tight inlets.
- ``checkpoint_clip_fallback`` (default ``False``) opts into a single-crossing
  clip rescue. Leave it off unless you are seeing a meaningful fraction of
  crossing survivors that the validity gate is discarding.
