Pipeline — data-flow overview
==============================

The ``track_gen`` pipeline turns a batch of per-environment seeds into a batch of
``Track`` objects. The whole pipeline lives in
``track_gen/_src/warp_pipeline.py`` (plus the relaxation solve in
``track_gen/_src/warp_relax.py``) and is written entirely in
`NVIDIA Warp <https://github.com/NVIDIA/warp>`_ kernels.

Goals
-----

1. **No torch/Warp mixing.** One implementation. Every pipeline stage is a Warp kernel.
   Runtime package imports do not depend on torch; tests and diagnostics may bridge via
   ``wp.from_torch`` / ``wp.to_torch`` at the boundary.
2. **One codebase, two devices.** Warp kernels run on the Warp ``cpu`` device (GPU-free,
   for tests/CI) and on ``cuda`` (production). The same code path serves both.
3. **A single replayable CUDA graph.** The entire pipeline is static (fixed shapes, fixed
   iteration counts, no host-side branching on tensor data), so it captures into one CUDA
   graph that can be replayed with new seeds.

Data flow
---------

.. code-block:: text

   seeds[E]
     │  FIRST-STAGE GENERATION  (registered config.generator:
     │    "bezier", "hull", "polar", "voronoi", "checkpoint", or "repulsive";
     │    single pass, generator-private scratch, no regen loop, no generation gate)
     ▼
   centerline[E, num_points, 2]     (every env real) + valid[E] (all True —
     │                                 final geometric validity is decided post-relax by INFLATE)
     │  RESAMPLE   resample_constant_spacing → per-track count[e] = ⌊perimeter/spacing⌋+1, capped at N_max
     ▼
   spaced[E, N_max, 2]              (NaN-padded past each track's count[e])
     │  RELAX      (XPBD: separation + spacing + bending, fixed iters, double-buffered, count-aware)
     ▼
   relaxed[E, N_max, 2]
     │  INFLATE    resample_uniform (re-uniformize) ─► frame+curvature ─► constant-width offset
     │             ─► validity ─► arclength
     ▼
   Track(outer, center, inner, tangent, normal, arclen, length, valid, count)

.. figure:: ../assets/readme-pipeline-stages.png

   The four pipeline stages: generation, resample, relax, and inflate.

Registered first-stage generators
----------------------------------

The first stage is pluggable. ``TrackGenerator.__init__`` resolves
``config.generator`` through ``track_gen._src.generator_registry``, allocates that
generator's private scratch once, and ``_run_pipeline`` calls the resolved
``GeneratorSpec.generate`` with orchestrator-owned ``out_centerline`` and
``out_valid_wp`` buffers. The production runtime registry currently exposes:

.. list-table::
   :header-rows: 1

   * - generator
     - module
     - representation
     - repair path
   * - ``"bezier"``
     - ``warp_generate.py``
     - sampled grid corners → angle-sorted closed cubic Bezier
     - selected corner-polygon fallback for Bezier self-crossers
   * - ``"hull"``
     - ``warp_generate_hull.py``
     - angle-sorted point loop → displaced midpoints → closed Catmull-Rom
     - selected augmented-polygon fallback for Catmull-Rom self-crossers
   * - ``"polar"``
     - ``warp_generate_polar.py``
     - sorted polar control knots → periodic Catmull-Rom spline
     - no generator-local fallback; downstream relaxation/inflation gates final validity
   * - ``"voronoi"``
     - ``warp_generate_voronoi.py``
     - fixed site field → angular anchor cycle → smoothed graph-cycle loop
     - selected anchor-polygon fallback for smooth-loop self-crossers
   * - ``"checkpoint"``
     - ``warp_generate_checkpoint.py``
     - radial checkpoints → bounded-turn steering → additive heading-ramp closure
     - best-of-K candidate selection (default K=4); optional single-crossing clip fallback (off by default)

Every registered runtime generator follows the same hard contract:

- ``alloc_scratch(config)`` allocates fixed-shape, generator-private Warp buffers once.
- ``generate(seeds_wp, config, out_centerline, out_valid_wp, scratch)`` writes an
  ``[E*num_points]`` closed centerline into the supplied output buffer and writes ``[E]``
  generation flags.
- The hot path is pure Warp, zero allocation, deterministic in ``(seed, config)``, and
  graph-capturable: no host-side retry loop, no host branch on generated tensor data, and
  no per-env Python branching. **Exception:** ``repulsive`` declares
  ``GeneratorSpec(capturable=False)`` and is exempt from the allocation and
  graph-capturable clauses — see :doc:`/generators/repulsive` and the note below.
- ``out_valid_wp`` is filled with ``1`` by the current runtime generators. It is a stage
  flag, not the final quality decision. Turning, thickness, NaNs, width floor, and optional
  border intersections are judged later by ``inflate_warp`` after constant-spacing and XPBD
  relaxation.

Per-generator algorithm details are covered in the individual generator deep-dive pages
(bezier, hull, polar, voronoi, checkpoint, repulsive).

Determinism, yield, and FP tolerance
--------------------------------------

**Determinism.** Warp's per-env RNG is deterministic, so a given seed buffer reproduces the
same tracks run-to-run on a device. The ``cpu`` and ``cuda`` RNG streams may differ — each
device is internally reproducible; cross-device yields are compared statistically, not
per-env. **Exception:** ``repulsive`` (the one ``capturable=False`` generator) records a
fresh ``wp.Tape`` per growth iteration; its CUDA autodiff gradients accumulate via atomics
whose float summation order varies run-to-run, so on ``cuda`` it is only *statistically*
reproducible (same distribution, yield, and compactness band), not bit-identical, per seed
— CPU stays byte-identical. See ``track_gen/_src/warp_generate_repulsive.py``'s module
docstring for the full account.

**Yield.** Relaxed-valid yield is approximately **0.999** end-to-end (E ≥ 2048): ≈ 0.9991
at the fat-band default (``half_width=0.5``, ``scale=10``, ``spacing=0.30``,
``N_max=384``), ≈ 0.9955 at the library default config, ≈ 0.9998 in the thin
(``half_width=0.03``) regime — all measured at E=8192. Two changes got it there:

- **Constant spacing made relaxation lossless.** The old ``fixed``-256 ceiling (≈ 0.68 in
  the fat-band regime) was slow-Jacobi under-convergence from over-resolution, not
  un-relaxable geometry: at 256 points the centerline is over-resolved relative to its
  half-width, so the fixed iteration count cannot drive the Jacobi solve to convergence.
  Relaxing at ~0.6×half_width spacing (≈ 145–160 nodes/track, not 256) lifted that same
  regime **0.684 → 0.999** — and runs faster (fewer nodes), while staying
  graph-capturable.
- **Single-pass first-stage generation replaced the regen loop.** With relaxation lossless,
  the default Bezier residual was a small fraction of smooth-centerline self-crossers.
  Rather than a fixed ``max_regen_iters`` accept-first-valid loop, the Bezier generator now
  takes one corner draw per env and routes any track whose smooth Bezier centerline
  self-crosses to its provably simple corner polygon, which XPBD re-rounds. The ``hull``
  generator follows the same selected-polygon rescue pattern for its Catmull-Rom
  self-crossers, ``voronoi`` falls back to its selected anchor polygon for smooth-loop
  crossings, ``checkpoint`` uses bounded best-of-K selection with an optional
  single-crossing clip fallback, and ``polar`` emits a closed radial spline with no
  generator-local fallback. ``max_regen_iters`` is therefore vestigial on the Warp path:
  it remains a ``TrackGenConfig`` field for the torch oracle but is ignored by
  ``_run_pipeline``.

**FP tolerance and hard thresholds.** Validity gates (``th_ok``, ``turn_ok``) are hard
comparisons; near a decision boundary the accepted ~1e-4 Warp-vs-torch drift can flip a
single env's bool. Tests keep their inputs away from those boundaries; the end-to-end yield
comparison uses an aggregate tolerance, not per-env equality.
