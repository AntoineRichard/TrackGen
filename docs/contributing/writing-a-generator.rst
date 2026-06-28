Writing a Generator
===================

A generator produces the initial closed centerline that the pipeline then
resamples, relaxes (XPBD), and inflates.  To add one, implement two callables
and register a ``GeneratorSpec`` (see
``track_gen/_src/generator_registry.py``).

What you implement
------------------

Two callables are required.

``alloc_scratch(config) -> scratch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allocate your generator's private working buffers **once**.  Use fixed shapes
derived from ``config`` (e.g. ``num_envs``, ``max_num_points``,
``num_points_per_segment``, ``num_points``), all on ``str(config.device)``.
Return any object exposing the buffers your ``generate`` function uses.

Do **not** allocate the output centerline or the validity array here — the
orchestrator owns those.

``generate(seeds_wp, config, out_centerline, out_valid_wp, scratch) -> None``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write a closed centerline of ``config.num_points`` points per environment into
the output arrays in place.  Parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``seeds_wp``
     - ``[E]`` int32 ``wp.array`` — one base seed per environment.
   * - ``out_centerline``
     - ``[E*num_points]`` ``wp.vec2f`` — write the closed centerline in place.
   * - ``out_valid_wp``
     - ``[E]`` int32 — current runtime generators fill this stage flag with
       ``1`` for every env.  Final geometric validity is decided later by the
       shared post-relax inflation validity gate.
   * - ``scratch``
     - The object your ``alloc_scratch`` returned.

Hard rules
----------

Five rules must hold for every generator:

1. **Pure Warp kernels only.**  Use ``wp.launch``; one env per row.  No torch
   code inside ``track_gen/_src``.
2. **Zero dynamic allocation inside** ``generate``.  All buffers come from
   ``alloc_scratch``.
3. **CUDA-graph capturable.**  No host sync, no host-side retry loop
   conditioned on generated data, no per-env Python branching inside
   ``generate``.
4. **Fixed bounds for every loop and buffer.**  Graph capture requires static
   shapes.
5. **Deterministic in** ``(per-env seed, config)``.  Use the Warp RNG
   (``track_gen._src.rng_*``).  CPU vs CUDA RNG results may differ, as
   elsewhere in the codebase.

What you do NOT have to guarantee
----------------------------------

A simple (non-self-intersecting) loop is preferred but not required.  Local
fallback is generator-specific: ``bezier``, ``hull``, and ``voronoi`` use
selected polygon-style rescues; ``checkpoint`` uses best-of-K plus an optional
clip fallback; ``polar`` has no local fallback.  Residual bad geometry is
handled by the common post-relax validity gate.

Output must be finite (no NaN) for generated environments.

Current standard generators
----------------------------

The standard runtime generators are ``bezier``, ``checkpoint``, ``hull``,
``polar``, and ``voronoi``.  Call
``track_gen._src.generator_registry.available()`` for the authoritative list.

The Voronoi method is implemented as a fixed-budget site-field / graph-cycle
generator rather than exact Voronoi ridge walking; exact Delaunay/Voronoi
construction remains an offline diagnostic until it can satisfy the same
fixed-shape Warp contract.  The checkpoint method is the bounded,
graph-capturable version of the Gymnasium CarRacing checkpoint-steering family:
fixed-N steering, additive heading-ramp closure, best-of-K selection, and
optional clip fallback.

How a generator is judged
--------------------------

Run ``benchmarks/compare_generators.py`` to characterise a new generator
against the existing ones (see ``docs/generator-baseline.md`` for baseline
numbers and the comparison methodology).

Generators are characterised, never gated: a method that scores worse on yield
but better on speed or style stays selectable.
