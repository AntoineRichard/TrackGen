3D gate courses
===============

Gate sequences (``GateGenerator``, tutorial at :doc:`/tutorials/gate-sequences`) are
**2.5D**: the plan-view layout comes from the proven 2D centerline generators, and
altitude (Z) is layered on afterwards as a first-class profiler stage. The result is a
genuine 3D course — gates carry a ``vec3f`` position, a roll-free orientation
quaternion, and can be flown, localized, and collided against in three dimensions —
without giving up any of the 2D generators' layout quality or determinism. This page
covers the 2.5D design, the four ``z_profile`` families and their knobs, the two
``gate_align`` modes, the runtime parity with track mode, the validity rules, and the
breaking changes from the pre-3D API.

Why 2.5D
--------

The 2D gate pipeline is well understood: each native generator (``bezier``, ``hull``,
``polar``, ``voronoi``, ``checkpoint``) samples corner anchors, they are ordered and
bounding-box normalized, the :doc:`gate self-collision solve </relaxation/gates>`
separates overlapping gates, and tangents follow. All of that is a plan-view (XY)
computation and stays **bit-identical** to the 2D era.

Altitude is added as a separate stage that runs on the final ordered, relaxed 2D
anchors — *before* the lift to ``vec3f``. The gate pipeline is a list of stages
(:ref:`pipeline-gate-stages`), and its stage ids, in order, are:

.. code-block:: text

   generator (anchors) → order → normalize → gate_relax (SPHERE RELAXATION)
       → gate_arclen → elevation (Z PROFILE, writes per-gate z) → lift (to vec3f)
       → tangents → gate_frames → validity

``normalize`` runs for the generators that ask for it (``polar``, ``voronoi``,
``checkpoint``), and ``gate_arclen`` (the plan-view arc tables the profiles read) for a
non-flat profile. Keeping Z orthogonal to the XY layout is what makes the feature cheap
and safe: the plan-view generators are untouched, the Z stage is a single
data-independent kernel launch (graph-capturable, zero per-call allocation), and turning
altitude off (``z_profile="flat"`` with ``z_base=0``) reproduces the old planar course
exactly.

The ``elevation`` stage is the one tracks use: the pipeline rewires its ports from the
centerline channels to the gate ones (:ref:`pipeline-gate-elevation`), so a plugin
elevation runs on gates too. A gate generator that knows its own altitudes and tangents
*provides* them instead: the pipeline then drops ``gate_arclen``, ``elevation`` and
``tangents`` and keeps the generator's values (:ref:`pipeline-provides`).

Altitude profiles
-----------------

``GateGenConfig.z_profile`` selects the altitude family: one of the four built-ins below,
or any elevation stage a plugin registers (:ref:`gate-stage-shortcuts`). All four
built-ins write every real gate slot and NaN the padding; the ``z_*`` knobs below are
inert for the profiles they do not apply to.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - ``z_profile``
     - Altitude model
     - Knobs
   * - ``"flat"`` (default)
     - Every real gate sits at ``z_base``. With the default ``z_base=0`` this is the
       old planar course.
     - ``z_base``
   * - ``"uniform"``
     - Each gate's altitude is drawn i.i.d. uniform in ``[z_min, z_max]``.
     - ``z_min``, ``z_max``
   * - ``"random_walk"``
     - A Brownian-bridge walk clamped to ``[z_min, z_max]``, closed back to its start
       so the loop is continuous. Each step is capped by a **grade** — the maximum
       ``|dz|`` per unit of plan-view arc length — so altitude changes stay proportional
       to horizontal travel.
     - ``z_base`` (walk origin), ``z_min``, ``z_max``, ``z_max_step`` (grade cap)
   * - ``"noise"``
     - Periodic harmonic noise oscillating around ``z_base``: a sum of
       ``z_noise_harmonics`` sinusoids of total amplitude ``z_noise_amplitude``, clamped
       to ``[z_min, z_max]``.
     - ``z_base``, ``z_noise_amplitude``, ``z_noise_harmonics`` (default 3),
       ``z_min``, ``z_max``

.. note::

   ``z_max_step`` is a **grade**, not an absolute step: it bounds ``|dz|`` per unit
   plan-view arc length between adjacent gates, so widely spaced gates may still differ
   in altitude by more than ``z_max_step``.

``z_min``, ``z_max``, ``z_max_step`` and ``z_noise_amplitude`` are per-env (dynamic)
parameters, as on tracks: ``presets.gates(randomize={"elevation.z_max": (lo, hi)})``
draws them per env at every generation, and a curriculum writes
``gen.params["elevation"][...]`` from its own graph. Which knob sets the difficulty for
each profile, the window it needs and the crossing rules are on the track page
(:ref:`tracks-25d-per-env`); the gate generators' own dynamic knobs are listed in
:ref:`generators-per-env`, and the per-env bar width and course size in
:ref:`gate-width-per-env`. The course size (``generator.size`` or ``normalize.size``)
scales the plan view, not the altitude: the gates spread out while ``z`` keeps its range,
so the grades between adjacent gates fall and the ``z_valid_grade`` check passes more envs
at a larger size (:ref:`tuning-size`).

.. _gates-3d-stations:

Gates on a track centerline: 3D spacing
---------------------------------------

Gates made from a track generator's centerline (``presets.gates(..., source="stations"
or "relaxed_stations")``, :ref:`pipeline-stations`) are placed at equal arc length along
the loop. By default that is plan-view arc length, and the altitude is drawn afterwards,
over the gates, exactly as for an anchors generator. On a course that climbs, equal
plan-view spacing is not equal flight distance: ``spacing_metric="xyz"`` places the
stations at equal **3D** arc length instead. That needs the altitude on the loop before
the stations are placed, from one of two sources:

- a generator that computes its own altitude (``provides={"z"}``,
  :ref:`generator-provides`);
- with ``source="relaxed_stations"``, the elevation stage run on the relaxed centerline:
  ``elevation_on="centerline"``.

The stations then carry the altitude to the gates (``carry.z``, with the stations'
``order.raw``), and no gate-level elevation runs.

.. code-block:: python

   from track_gen import GateGenerator
   from track_gen.pipeline import presets

   spec = presets.gates(num_envs=64, device="cuda:0", generator="bezier",
                        source="relaxed_stations", gates=12,
                        elevation=("noise", {"z_noise_amplitude": 0.3,
                                             "z_min": -0.3, "z_max": 0.3}),
                        spacing_metric="xyz")       # elevation_on defaults to "centerline"
   gates = GateGenerator(spec).generate()

.. warning::

   ``spacing_metric`` and ``elevation_on`` are separate controls, and ``elevation_on=None``
   (the default) follows the metric: ``"centerline"`` for ``"xyz"``, ``"gates"`` for
   ``"xy"``. **Switching the metric with the default** ``elevation_on`` **therefore also
   moves the elevation** from the gates to the centerline, which changes the course, not
   only the spacing: a profile drawn along the loop and sampled at the stations instead
   of one drawn over the gates. For the knot profiles (``uniform``, ``random_walk``) the
   knots then lie along the loop (``z_control_points`` of them) rather than at the gates.
   To compare the two metrics on the same altitude model, pass ``elevation_on``
   explicitly (``elevation_on="centerline"`` with both metrics). Moving a plugin
   elevation to the centerline also drops its exported channels from
   ``GateSequence.attributes`` (the stations carry generator attributes only).

``elevation_on="gates"`` with ``"xyz"``, ``elevation_on="centerline"`` with
``source="stations"`` (a raw loop has no sub-pipeline to elevate), ``"xyz"`` with no
altitude on the loop (a flat elevation and no ``z`` provider), an explicit
``elevation_on`` with a ``z`` provider (no elevation runs), ``"xyz"`` with an
anchors generator, and an explicit order other than ``raw`` with an altitude on the loop
raise ``ValueError``; :ref:`pipeline-stations-xyz` has the full table. The gate arc
tables (``gate_arclen``) stay plan-view in every case: they feed the gate elevation, which
produces the altitude a 3D metric would need.

.. figure:: _static/gate-courses.png
   :alt: Three 3D gate courses, each as a plan view and an elevation profile
   :align: center

   Three gate courses rendered by ``viz/plot_gate_courses.py`` (``z_profile="random_walk"``,
   ``gate_align="full_tangent"``). Each env is drawn as a plan view (XY centerline plus
   per-gate left/right bars) and an elevation profile (arc length vs. z), with the closed
   :ref:`CourseLine <gates-3d-runtime>` spline threading the gates. Regenerate with
   ``python viz/plot_gate_courses.py --out docs/_static/gate-courses.png --envs 3 --seed 5``.

Gate alignment: ``gate_align``
------------------------------

Once each gate has an altitude, its **frame** (the roll-free orientation quaternion, with
local x = gate forward, y = left, z = up) is built from the tangent. ``gate_align``
chooses which tangent:

- ``"yaw_only"`` (default) uses the **plan-view** tangent — the 3D tangent projected onto
  the XY plane. The gate posts stay horizontal, so a gate's ``left`` and ``right``
  endpoints share the center's altitude. This is the natural choice for drone gates that
  hang level regardless of the course slope.
- ``"full_tangent"`` aligns the frame forward axis with the **full 3D** tangent, tilting
  the posts to follow the course's climb or descent. Use it when the gate plane should be
  perpendicular to the actual flight direction.

Either way the frame is roll-free (the up axis is kept as close to world-up as the
forward axis allows), and ``orientation`` is what the runtime, the collision frames, and
downstream consumers read — the removed ``normal`` field is recovered as
``wp.quat_rotate(orientation, wp.vec3f(0, 1, 0))`` when a left-axis vector is needed.

.. _gates-3d-runtime:

Runtime parity
--------------

A gates-mode :doc:`Course </utilities/course>` drives these 3D courses exactly like track
mode — **construct → bind → generate → step / reset** — with three additions that make
the third dimension first-class:

Pass detection
~~~~~~~~~~~~~~~

Progress is a **swept-segment plane-crossing** test: between two steps the agent's motion
segment is tested against the target gate's plane, and a pass is counted only when the
crossing falls inside the opening — within the left-right extent **and** vertically within
the gate's half-opening (``CheckpointSet.up_half``, aliased from ``GateSequence.half_size``).
Because track cross-sections use an effectively unbounded vertical half-opening, planar
motion behaves exactly as before; a gate course, by contrast, will not credit a pass for a
drone that flies over or under the gate. ``res.events`` still exposes ``passed``,
``dist_to_next``, and ``progress`` for reward shaping.

Pass detection uses the gate's *physical* plane — the pose forward — in both alignment
modes, so a ``yaw_only`` gate on a sloped course is crossed exactly where its upright frame
stands.

The track frame
~~~~~~~~~~~~~~~~

In gates mode ``course.step()`` returns a ``StepResult`` whose third field, ``res.frame``,
is a :class:`~track_gen.localize.TrackFrame` localizing the agent on a closed 3D
``CourseLine`` spline (a periodic Catmull-Rom curve through the gate centers). It carries
``s`` (arc length along the spline), ``n`` (signed lateral offset, positive to the right
of travel), ``n_up`` (signed vertical offset in the roll-free frame at the foot point), and
``segment`` (nearest spline segment). In track mode ``res.frame`` is ``None``.

The centerline is sampled at ``samples_per_gate`` uniform *spline-parameter* steps per
gate segment — not constant 3D arc-length (samples bunch slightly where gates are close
together). ``arclen`` is nevertheless the true cumulative arc length of the sampled
polyline, so localization and progress are unaffected.

Frame collision
~~~~~~~~~~~~~~~~

Beyond the ``post_radius`` gate-post discs, gates mode offers ``frame_collision=True``:
each square gate becomes four thin oriented frame members and a ``FrameChecker`` tests the
agent **sphere** against them in the gate's own 3D frame. It requires ``frame_thickness``,
``frame_depth``, and ``agent_radius`` all ``> 0`` and is **mutually exclusive** with
``post_radius > 0`` (choose discs *or* frames). The contact is a ``FrameContact``
(``hit``, ``depth``).

Validity
--------

The 2D layout validity checks are unchanged: a sequence is rejected for the same planar
reasons as before (degenerate counts, crossing gate bars in an env whose bars have a
positive width, and so on). ``GateSequence.fail_mask`` says which check rejected an env,
one bit per
check (:ref:`pipeline-gate-fail-bits`). The 3D pipeline adds one altitude check (fail
bit ``grade``):

- ``z_valid_grade`` — the maximum allowed ``|dz|/ds`` grade between adjacent gates
  (``ds`` = plan-view distance). A sequence with any adjacent pair steeper than this is
  marked invalid. ``z_valid_grade = 0`` (the default) **disables** the check.

Two v1 limitations are explicit by design:

1. **Plan-view self-crossings.** The track pipeline still rejects plan-view crossings, but
   a *gate course* may self-cross in XY on purpose — a stacked over/under section is a
   legitimate 3D course whose plan view crosses itself. Gate validity therefore does not
   reject XY crossings.
2. **Spline XY drift is unchecked.** Validity is evaluated on the gate anchors, not on the
   ``CourseLine`` spline that interpolates them. The Catmull-Rom curve can bow away from
   the straight anchor-to-anchor chords, and that drift is not checked against the layout —
   keep gate spacing sane if you rely on the spline staying near the anchors.

.. _gates-3d-migration:

Migration from the 2D API
-------------------------

The 3D data model is a breaking change relative to the pre-3D gate/track API. Verbatim
from the commit that introduced it:

.. admonition:: Breaking changes
   :class: warning

   ``GateSequence``/``Track`` geometry is ``vec3f``; ``GateSequence.normal`` removed,
   ``orientation`` (``quatf``) + ``half_size`` added; ``CheckpointSet`` gains ``up_half``;
   ``TrackFrame`` gains ``n_up``; bound position buffers are ``[E]`` ``vec3f``;
   ``Course.bind`` takes orientation quaternions instead of yaw. XY behavior is
   golden-verified bit-identical.

In practice:

- Reshape geometry arrays with a trailing ``3`` instead of ``2``:
  ``wp.to_torch(gates.position).view(E, G, 3)`` (likewise ``Track.center/outer/inner``).
  The ``z`` component is 0 for the planar track pipeline and carries real altitude for
  gate courses.
- Replace ``GateSequence.normal`` reads with ``orientation`` (a ``quatf`` pose) plus
  ``half_size``; recover the old left-normal as
  ``wp.quat_rotate(orientation, wp.vec3f(0, 1, 0))``.
- Bind ``vec3f`` position buffers and pass ``Course.bind(position=..., orientation=...,
  half_extents=...)`` with an ``[E * max_boxes]`` ``quatf`` orientation buffer (identity
  quaternion ``(0, 0, 0, 1)`` if you have no rotation) instead of a ``yaw`` float array.

See also
--------

- :doc:`/tutorials/gate-sequences` — the end-to-end gate-generation recipe and output
  fields, including the RL runtime loop.
- :doc:`/relaxation/gates` — the plan-view gate self-collision solve that runs before the
  Z profile.
- :doc:`/utilities/course` — the ``Course`` facade that turns gates into checkpoints, the
  3D localizer, and optional post/frame collision.
