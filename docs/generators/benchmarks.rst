Generator Benchmarks
====================

Reference metrics for the six registered centerline generators, produced by
``benchmarks/compare_generators.py``.  These numbers characterize tradeoffs among
quality, diversity, and speed.  They never gate which generators ship — every
registered generator stays selectable via ``config.generator``.

.. note::

   Suite: seed base 0, E=512, default ``TrackGenConfig``. The ``bezier``, ``checkpoint``,
   ``hull``, ``polar``, and ``voronoi`` rows run on the Warp ``cpu`` device; CPU timings are
   machine-dependent and intended for relative comparison only, use a larger ``E`` for
   release-grade timing. The ``repulsive`` row is measured on ``cuda`` (an RTX 4090)
   instead — see the warning below.

.. warning::

   ``repulsive`` is a host-driven, non-graph-capturable optimizer: it does not benefit from
   CUDA-graph replay the way the other five generators do, so its ``gen_ms_per_call`` is
   measured on ``cuda`` directly rather than ``cpu`` (a CPU eager run would be even slower
   and less representative of production use) and is **not directly comparable** to the
   other rows. Indicative RTX-4090 wall-clock: ~0.2 s at E=64 and ~5 s at E=8192
   (~3 ms/track amortized), versus ~2 ms per batch for ``bezier`` — roughly **1000×**
   slower. The number is also throttle-sensitive (±1.5× from GPU clock/pool state). See
   :doc:`repulsive` for the full cost discussion and the recommended regeneration-cadence /
   staggered-slice usage.

Reproduce
---------

.. code-block:: bash

   .venv/bin/python -m benchmarks.compare_generators --E 512 --seed 0

Metrics Table
-------------

.. list-table::
   :header-rows: 1
   :widths: 14 8 14 12 12 13 13 14 14 12 12 12 10 14

   * - generator
     - yield
     - pre_relax_self_intersection_rate
     - xpbd_displacement
     - mean_length
     - mean_compactness
     - compactness_p50
     - compactness_degenerate_rate
     - shape_variety_pass
     - mean_chicanes
     - straight_frac
     - peak_curvature
     - lap_time
     - gen_ms_per_call
   * - bezier
     - 0.9922
     - 0.005859
     - 0.05008
     - 5.056
     - 0.4387
     - 0.4238
     - 0
     - 1
     - 13.21
     - 0.2295
     - 8.762
     - 10.16
     - 794
   * - checkpoint
     - 0.9766
     - 0.001953
     - 0.03541
     - 4.224
     - 0.6142
     - 0.6225
     - 0
     - 1
     - 14.8
     - 0.1991
     - 8.577
     - 8.549
     - 1125
   * - hull
     - 0.9941
     - 0.007812
     - 0.06167
     - 4.958
     - 0.4254
     - 0.4111
     - 0
     - 1
     - 14.13
     - 0.1919
     - 8.908
     - 10.52
     - 839.5
   * - polar
     - 1
     - 0
     - 0.0271
     - 4.736
     - 0.5586
     - 0.5524
     - 0
     - 1
     - 12.72
     - 0.1978
     - 8.44
     - 9.488
     - 568
   * - voronoi
     - 1
     - 0
     - 0.01315
     - 4.27
     - 0.7335
     - 0.7318
     - 0
     - 1
     - 13.88
     - 0.2185
     - 8.205
     - 7.849
     - 736.9
   * - repulsive
     - 0.9902
     - 0.001953
     - 0.02211
     - 12.99
     - 0.1541
     - 0.1526
     - 0
     - 1
     - 34.71
     - 0.1903
     - 8.723
     - 25.23
     - 242.5 (cuda; see warning — not comparable to the cpu rows; now stable run-to-run
       within roughly 242–245 across repeated measurements, since the analytic-adjoint
       gradient makes the flow byte-deterministic — the old tape path spread 525–1060)

Metric Definitions
------------------

**yield**
   Fraction of generated environments that pass the full post-relax validity gate
   (turning number, thickness floor, no NaN); higher is better.

**pre_relax_self_intersection_rate**
   Fraction of environments whose raw generated centerline contains at least one
   proper crossing before any relaxation; lower indicates a cleaner generator-level
   output.

**xpbd_displacement**
   Mean displacement (in track-coordinate units) that XPBD relaxation applies to the
   centerline; lower means the generator already produces geometry close to the
   relaxed equilibrium.

**mean_length**
   Mean centerline perimeter (track-coordinate units) across valid environments;
   reflects the typical physical size of generated tracks.

**mean_compactness**
   Mean isoperimetric compactness (``4π·area / perimeter²``) across valid
   environments; higher values indicate rounder, more compact track shapes.

**compactness_p50**
   Median compactness; a robust central-tendency measure less sensitive to extreme
   shape outliers than the mean.

**compactness_degenerate_rate**
   Fraction of valid environments with compactness below a degenerate threshold;
   zero means no degenerate (near-zero-area) shapes were produced in this suite run.

**shape_variety_pass**
   Binary flag (0 or 1) indicating whether the generator produces sufficient
   shape diversity across the batch; all six generators pass at default config.

**mean_chicanes**
   Mean count of chicane-like direction-reversals per valid track; reflects how
   much local curvature variation the generator introduces.

**straight_frac**
   Mean fraction of the centerline classified as a straight segment; higher values
   indicate more open, high-speed layout character.

**peak_curvature**
   Mean maximum absolute curvature over the relaxed centerline; reflects how tight
   the sharpest corner of a typical track is.

**lap_time**
   Estimated lap time (seconds) from a simple kinematic model; lower values
   correspond to faster, more open tracks.

**gen_ms_per_call**
   Wall-clock generation time in milliseconds per ``TrackGenerator.generate()`` call
   on CPU at E=512; machine-dependent — use for relative comparison only. The
   ``repulsive`` row is the exception: it is measured on ``cuda`` (see the warning
   above) and is not comparable to the cpu-measured rows.
