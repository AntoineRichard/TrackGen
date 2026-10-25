.. _generator-benchmarks:

Measure a generator
===================

Benchmark the course settings and batch size you intend to use. Compare families on
the same machine, device, width, spacing, capacity and solver settings. The generator
pictures are selected explanations; they do not establish yield, speed or a universal
shape ranking.

The repository's ``benchmarks.compare_generators`` command computes current metrics
from actual generated batches. It reports the complete ``generate()`` call, including
the shared road pipeline. Start with a small CPU sampling-family check:

.. code-block:: bash

   python -m benchmarks.compare_generators --E 32 --seed 0 \
       --generators bezier checkpoint hull polar voronoi

For production CUDA comparison, run the same suite on the same GPU and separate
``repulsive`` if its iterative work dominates the run:

.. code-block:: bash

   python -m benchmarks.compare_generators --E 512 --seed 0 --cuda \
       --generators bezier checkpoint hull polar voronoi
   python -m benchmarks.compare_generators --E 32 --seed 0 --cuda \
       --generators repulsive

These commands are measurement recipes, not published results. Record the source
commit, Warp version, platform/GPU, exact config and seeds with the output. CPU and
CUDA timing rows are not comparable, and changing batch size changes amortization.
Use several seed batches before drawing conclusions about rare failures or diversity.

What to compare
---------------

.. list-table:: Metrics and their limits
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Interpretation
   * - ``yield``
     - Fraction passing final validity in the whole generated batch. Keep failed rows in the denominator; a gallery filtered to valid examples cannot estimate this.
   * - ``pre_relax_self_intersection_rate``
     - Number of retained finite rows whose post-resample, pre-relax XY slice has proper crossings, divided by the whole batch E. The tap, final-count slicing and filtering are described below.
   * - ``xpbd_displacement``
     - Point-weighted mean XY displacement between the post-resample, pre-relax slice and final Track.center, in coordinate units. Uses the retained finite points described below; it is not a road-quality proof.
   * - ``mean_length``
     - Mean closed centerline perimeter over valid rows, in coordinate units. Compare with size/scale held fixed and report validity filtering.
   * - ``mean_compactness`` / ``compactness_p50``
     - Mean/median of 4π * enclosed area / perimeter² over valid rows. Round loops score higher; a racing task may prefer a different structure.
   * - ``compactness_degenerate_rate``
     - Fraction of retained valid rows with compactness above 0.90, the tool's near-circular diagnostic. This threshold is a tool heuristic, not a vehicle constraint.
   * - ``shape_variety_pass``
     - Heuristic based on mean compactness and the fraction above 0.90. It does not compare pairs of tracks or measure coverage of a task's whole distribution.
   * - ``mean_chicanes`` / ``straight_frac``
     - Geometric heuristics for direction changes and straight portions. Their thresholds are not a driving policy or difficulty specification.
   * - ``peak_curvature``
     - Mean of each valid track's maximum absolute curvature. Sensitive to sample resolution; compare at the same spacing and pipeline settings.
   * - ``lap_time``
     - Seconds estimated by the benchmark's kinematic model. It is not a measured simulator or real-vehicle lap.
   * - ``gen_ms_per_call``
     - Mean wall-clock milliseconds of five calls after two warm calls. Includes the shared pipeline; excludes initial generator construction and its warm-up.

The crossing and displacement diagnostics read ``cl.xy`` after constant-spacing
``resample``, rather than ``raw.xy``. Both that tap and final ``Track.center`` are
sliced with the **final** ``track.count[e]``. Rows with count below 4 or nonfinite
final XY geometry are skipped; a nonfinite pre-relax slice also contributes to
neither crossing count nor displacement. Default poison output therefore excludes
failed rows from these numerators. With explicit ``on_invalid="keep"``, finite
failed rows can contribute: validity itself is not this filter.

The crossing denominator remains the entire batch E, including skipped rows. Its
value is a filtered post-resample diagnostic; a low value does not establish that
raw generator output, or the excluded failed rows, had no crossings. Endpoint and
collinear contact and finite-width clearance are separate questions. Displacement
instead divides summed point distances by the retained point count (NaN if none
remain), so it weights longer retained rows more heavily.

Include latency distribution and first-use cost in a simulator integration benchmark.
Warp kernel compilation, pipeline construction/warm-up, initial CUDA graph recording
and steady graph replay are distinct costs. A warm mean alone does not show the
worst reset pause or the memory budget.

Read the implementation in ``benchmarks/compare_generators.py`` and its metric
helpers when changing thresholds or adding a task-specific score. Choose a family
with :doc:`/tutorials/choosing-a-generator`; inspect the raw/final overlays in its
guide before interpreting a metric difference as a pure generator effect.
