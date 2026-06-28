Choosing a First-Stage Generator
==================================

``track_gen`` ships five registered first-stage centerline generators. The
generator is selected by passing ``generator=...`` to ``TrackGenConfig`` (or
``GateGenConfig``):

.. code-block:: python

   from track_gen import TrackGenConfig

   config = TrackGenConfig(generator="bezier", num_envs=64, device="cuda")

All five generators are fully supported, graph-capturable, and deterministic in
``(seed, config)``. The choice is about track *shape variety* and *relaxation
cost*, not correctness.

The Five Generators
--------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Best for
   * - ``"bezier"`` (default)
     - General-purpose racing tracks; the original generator with the widest
       parameter surface (corner count, Bezier-handle radius, edginess, style
       sampling). Good starting point for most workflows.
   * - ``"hull"``
     - Tracks with stronger lobes, pinches, and straights; the midpoint-displacement
       layer adds more variety than plain angle-sort Bezier, and the Catmull-Rom
       smoothing is lighter than a full Bezier assembly.
   * - ``"polar"``
     - Smooth, centered, nearly-convex tracks with low relaxation burden; starts
       from a closed radial representation so it is smooth and centered by
       construction. Good when you want consistent roundish shapes.
   * - ``"voronoi"``
     - Tracks built from a fixed site cloud with an angular-anchor cycle; tends
       to produce irregular, site-density-driven shapes distinct from the
       corner-polygon families. Useful for layout diversity.
   * - ``"checkpoint"``
     - Organic, continuously-undulating "flowing" loops in the style of
       Gymnasium CarRacing; the bounded-turn steering gives it a different
       character from the star-shaped families.

How to Set the Generator
--------------------------

Pass the generator name as a string to the config constructor. The name is
looked up through the production generator registry at ``TrackGenerator``
(or ``GateGenerator``) construction time:

.. code-block:: python

   from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG
   import warp as wp
   wp.init()

   E, device = 64, "cuda"

   # Switch generators by changing this one field:
   for name in ("bezier", "hull", "polar", "voronoi", "checkpoint"):
       config = TrackGenConfig(generator=name, num_envs=E, device=device)
       rng    = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
       track  = TrackGenerator(config, rng).generate()
       print(f"{name}: done")

To list all registered generators at runtime:

.. code-block:: python

   from track_gen._src.generator_registry import available
   print(available())

.. note::

   The Fourier generator lives in ``track_gen._experimental`` and is
   **unsupported** — it is not on the Warp pipeline and receives no
   compatibility guarantees.

Generator-Specific Parameters
--------------------------------

Each generator has its own set of shape knobs. A few examples:

- ``"bezier"`` — ``min_num_points``, ``max_num_points``, ``rad``, ``edgy``,
  ``handle_clamp_frac``, ``style_sampling`` (and the ``*_range`` fields that
  go with it).
- ``"hull"`` — ``hull_displacement`` (controls the strength of midpoint lobes
  and pinches).
- ``"polar"`` — ``polar_num_knots``, ``polar_radial_jitter``,
  ``polar_angular_jitter``.
- ``"voronoi"`` — ``voronoi_num_sites``, ``voronoi_site_layout``,
  ``voronoi_control_points``, ``voronoi_radial_variation``.
- ``"checkpoint"`` — ``checkpoint_count``, ``checkpoint_turn_rate``,
  ``checkpoint_steer_gain``, ``checkpoint_best_of_k``.

All generator-specific fields live on ``TrackGenConfig`` (and ``GateGenConfig``)
alongside the shared pipeline fields. Unrelated generator fields are silently
ignored, so you can prepare a single config with all knobs set and switch only
the ``generator`` string.

Deep-Dive References
----------------------

Detailed descriptions of each generator's algorithm, knobs, and tradeoffs:

- :doc:`/generators/bezier` — Bézier corner-sort deep dive
- :doc:`/generators/hull` — Hull midpoint-displacement deep dive
- :doc:`/generators/polar` — Polar spline deep dive
- :doc:`/generators/voronoi` — Voronoi site-field deep dive
- :doc:`/generators/checkpoint` — Checkpoint-steering deep dive
- :doc:`/generators/benchmarks` — Side-by-side quality, diversity, and speed metrics
- :doc:`/contributing/writing-a-generator` — contract every registered generator must satisfy
