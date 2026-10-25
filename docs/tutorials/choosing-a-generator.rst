.. _choosing-a-generator:

Choosing a Centerline Generator
===============================

.. note::

   Figures and empirical yield, range and timing measurements on this page predate the
   0.3 hashed-RNG switch. They are historical reference measurements; current generated
   bytes and seed-specific outcomes differ (:ref:`upgrading-0.3-rng`).

This is the decision guide: a few questions, and the generator they point to. The
catalogue, :ref:`generators-overview`, has the facts per family (how each one builds its
loops, its per-env knobs, the shared contract and the cost table); each generator has a
deep-dive page.

TrackGen ships six procedural centerline generators and ``external``, which ingests
centerlines you load. Select one by name in the :term:`preset`:
``presets.track(generator="hull")``
(``presets.gates(generator=...)`` for gates, ``TrackGenConfig.generator`` in a config).
Every generator is deterministic in its :term:`seed` and configuration on each device, works on
``cpu`` and ``cuda``, and can be recorded into a CUDA graph; the choice is about the
*shape* of the tracks and, for ``repulsive``, the generation cost.

.. figure:: ../assets/readme-generator-strip.png
   :alt: One sample centerline from each of the six procedural generators: bezier,
         checkpoint, hull, polar, voronoi and repulsive.
   :align: center

   One sample per generator, at the defaults.

What shape do you want?
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Generator
     - What it looks like
     - Pick it when
   * - ``"bezier"`` (default)
     - Star-shaped corner families joined by smooth cubic curves: sharp hairpins and
       long sweepers.
     - You want general-purpose racing tracks and the widest set of knobs (corner
       count, handle radius, edginess). The starting point for most workflows.
   * - ``"hull"``
     - Radial lobes, pinches and straights.
     - You want more variety of straights and pinches than ``bezier``'s corner families.
   * - ``"polar"``
     - Smooth, centered, nearly convex loops.
     - You want consistent, roundish, easy tracks (a low relaxation burden; the best
       yield with wide gate bars).
   * - ``"voronoi"``
     - Compact, irregular loops shaped by a field of sites.
     - You want layout diversity distinct from the corner families (the most compact
       loops).
   * - ``"checkpoint"``
     - Organic, continuously undulating "flowing" loops, in the style of Gymnasium
       CarRacing.
     - You want curvy tracks with no straight-and-hairpin character.
   * - ``"repulsive"``
     - Dense, foldy serpentines grown around random obstacles.
     - The shape is worth a generation that costs orders of magnitude more than the
       others: an iterative optimizer (the cost table of :ref:`generators-overview`).
       Regenerate rarely, or a slice of envs at a time.
   * - ``"external"``
     - Your own centerlines (a dataset, a learned model, a hand-drawn circuit).
     - You have the geometry already: :ref:`ingesting-centerlines`. It ignores the seed.

:doc:`/generators/benchmarks` gives the measured shape metrics (compactness, length,
chicanes, curvature) behind these descriptions.

Tracks or gates?
----------------

Five generators have a gate generator (``bezier``, ``hull``, ``polar``, ``voronoi``,
``checkpoint``): ``presets.gates(generator=...)`` places the gates at their anchors.
Every track generator, ``repulsive`` and ``external`` included, can also place gates at
equal arc length along its centerline (``presets.gates(..., source="stations")``,
:doc:`/tutorials/gate-sequences`). With bars (``gate_width > 0``), ``polar`` keeps the
most envs valid and ``bezier`` / ``hull`` the fewest: :ref:`gate-width-invalid-fraction`.

How many envs come out invalid?
-------------------------------

At the defaults almost no :term:`env` is :term:`invalid <invalid env>`: 0 % for ``bezier``, ``polar`` and ``voronoi``, 0.4 % for
``checkpoint``, 1.2 % for ``hull``, 1.6 % for ``repulsive`` (``E = 256``, CPU). Settings
that thin the course relative to the road or truncate the loop (a large ``size``, a small
``half_width`` with the default ``n_max``) raise it for every generator:
:ref:`invalid-envs`.

Do you need per-env knobs or a curriculum?
------------------------------------------

Every procedural generator has a per-env ``size``, and each of the five sampling
generators has shape knobs you can draw per env (``randomize=``) or move with a
curriculum without rebuilding the generator: ``rad`` and ``handle_clamp_frac``
(``bezier``), ``hull_displacement`` (``hull``), the radial and angular jitters
(``polar``, ``voronoi``), the radius, angle and turn-rate knobs (``checkpoint``), plus
``bezier`` / ``hull``'s corner count. ``repulsive`` has only ``size``; ``external`` has
none (:ref:`generators-per-env`, :doc:`/user-guide/curricula`).

Do you want several generators?
-------------------------------

- **In one batch:** several generators write disjoint env ranges of one shared result,
  read by one set of utilities and recorded in one graph (:ref:`results-mixed-batches`).
- **Across runs or workers:** sample a name per run. ``registry.available("generator")``
  lists every registered generator, installed plugins and ``external`` included; keep the
  seed-dependent ones:

.. code-block:: python

   from track_gen.plugin import registry

   registry.available("generator")
   # ['bezier', 'checkpoint', 'external', 'hull', 'polar', 'repulsive', 'voronoi', ...]
   seeded = [n for n in registry.available("generator")
             if registry.describe("generator", n).seeded]

.. snippet: after

   assert "external" not in seeded and {"bezier", "repulsive"} <= set(seeded)

``external`` ignores the seed: every env gets the same placeholder circle until you load
data, so sampling it by accident gives a batch of circles.

How to Set the Generator
------------------------

Pass the name to the preset (or to the config); it is looked up in the registry when the
generator is built, and an unknown name raises ``ValueError`` listing the registered
ones:

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"  # or "cpu"

   for name in ("bezier", "hull", "polar", "voronoi", "checkpoint"):
       gen = TrackGenerator(presets.track(num_envs=E, device=device, generator=name),
                            seeds=0)
       track = gen.generate()
       print(f"{name}: {(track.valid.numpy() == 0).mean():.1%} invalid")

   # repulsive: an iterative optimizer, far slower; this example keeps the batch small
   slow = TrackGenerator(presets.track(num_envs=8, device=device, generator="repulsive"),
                         seeds=0)

.. note::

   The Fourier generator lives in ``track_gen._experimental`` and is **unsupported**: it
   is not on the Warp pipeline and receives no compatibility guarantees. It is not
   shipped in the wheel; import it from a source checkout.

Generator-Specific Parameters
-----------------------------

Each generator has its own Params class, its knobs (:ref:`generating-params`). Pass them
as ``generator_params``:

.. code-block:: python

   spec = presets.track(num_envs=E, device=device, generator="polar",
                        generator_params={"polar_num_knots": 9, "polar_radial_jitter": 0.5})

A few examples:

- ``"bezier"``: ``min_num_points``, ``max_num_points``, ``rad``, ``edgy``,
  ``handle_clamp_frac``.
- ``"hull"``: ``hull_displacement`` (the strength of the midpoint lobes and pinches).
- ``"polar"``: ``polar_num_knots``, ``polar_radial_jitter``, ``polar_angular_jitter``.
- ``"voronoi"``: ``voronoi_num_sites``, ``voronoi_site_layout``,
  ``voronoi_control_points``, ``voronoi_radial_variation``.
- ``"checkpoint"``: ``checkpoint_count``, ``checkpoint_turn_rate``,
  ``checkpoint_steer_gain``, ``checkpoint_best_of_k``.
- ``"repulsive"``: ``repulsive_grow_mult_min`` / ``_max``, ``repulsive_domain_frac``,
  ``repulsive_ratchet_rate``, ``repulsive_obstacle_count_min`` / ``_max`` (and more, see
  :doc:`/generators/repulsive`).

Every field, with its default and rule, is in :ref:`builtin-generator-params`. In a
``TrackGenConfig`` the same names are flat fields, and the fields of the generators you
did not select are ignored, so one config can carry every knob and switch only the
``generator`` string. :doc:`/configuration/tuning` has per-generator tuning tips.

Deep-Dive References
--------------------

Detailed descriptions of each generator's algorithm, knobs, and tradeoffs:

- :doc:`/generators/bezier` — Bézier corner-sort deep dive
- :doc:`/generators/hull` — Hull midpoint-displacement deep dive
- :doc:`/generators/polar` — Polar spline deep dive
- :doc:`/generators/voronoi` — Voronoi site-field deep dive
- :doc:`/generators/checkpoint` — Checkpoint-steering deep dive
- :doc:`/generators/repulsive` — Repulsive-growth deep dive (the slow, iterative one)
- :doc:`/generators/benchmarks` — Side-by-side quality, diversity, and speed metrics
- :doc:`/contributing/writing-a-generator` — write your own generator
