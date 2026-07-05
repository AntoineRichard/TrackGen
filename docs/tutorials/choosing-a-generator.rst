Choosing a First-Stage Generator
==================================

``track_gen`` ships six registered first-stage centerline generators. The
generator is selected by passing ``generator=...`` to ``TrackGenConfig`` (or
``GateGenConfig``):

.. code-block:: python

   from track_gen import TrackGenConfig

   config = TrackGenConfig(generator="bezier", num_envs=64, device="cuda")

All six generators are fully supported and deterministic in ``(seed, config)`` — byte-identical
run-to-run per device (both CPU and CUDA; ``repulsive`` included, via its analytic-adjoint
gradient). All six are graph-capturable; ``repulsive`` is an iterative ``O(N²)`` optimizer whose
final-stage early exit runs device-side under capture (``wp.capture_while``), and it is the
slowest generator by far — hundreds of times slower than ``bezier`` (see
:doc:`/generators/repulsive`). The choice is about track *shape variety* and
*relaxation cost* (and, for ``repulsive``, generation cost), not correctness.

The Six Generators
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
   * - ``"repulsive"``
     - Dense, foldy serpentine circuits grown by self-repulsion around random
       obstacles; the foldiest family by far, but the slowest generator (an iterative
       ``O(N²)`` optimizer, hundreds of times slower than ``bezier``) — use only when the
       shape is worth the generation cost.

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

   # Switch generators by changing this one field. "repulsive" is the slowest by far
   # (an iterative O(N^2) optimizer) — see docs/generators/repulsive.rst.
   for name in ("bezier", "hull", "polar", "voronoi", "checkpoint", "repulsive"):
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
- ``"repulsive"`` — ``repulsive_grow_mult_min``/``_max``, ``repulsive_domain_frac``,
  ``repulsive_ratchet_rate``, ``repulsive_obstacle_count_min``/``_max`` (and more — see
  :doc:`/generators/repulsive`).

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
- :doc:`/generators/repulsive` — Repulsive-growth deep dive (the slowest, iterative one)
- :doc:`/generators/benchmarks` — Side-by-side quality, diversity, and speed metrics
- :doc:`/contributing/writing-a-generator` — contract every registered generator must satisfy
