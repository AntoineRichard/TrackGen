.. _generator-benchmarks:

Measure a generator
===================

Benchmark the course settings and batch size you intend to use. Compare families on
the same machine, device, width, spacing, capacity and solver settings. The generator
pictures are selected explanations; they do not establish yield, speed or a universal
shape ranking.

The :ref:`batch comparison <generator-batch-benchmarks>` measures the
complete finished-road pipeline consumed by racing integrations: generation,
resampling, relaxation, road borders and final validity. All CPU families use batches
of 64 **candidates**; all GPU families use 4,096. Failed rows remain in the validity
denominator. The benchmark never refills a batch to reach a requested number of
valid roads. ``external`` ingests supplied geometry and is outside this six-family
procedural comparison.

Batch latency protocol
----------------------

``benchmarks.benchmark_generators`` uses the Racing quickstart road settings for
every family and machine: 256 raw points, per-env capacity 512, half-width 0.03, spacing
0.018, size and scale 1, 50 XPBD iterations, flat elevation, default validity checks
and ``on_invalid="poison"``. Each family retains its own default generator
parameters. These settings are shared; the resulting lengths and point counts can
differ between families.

CPU samples measure a completed eager ``generate(advance=True)`` call
with ``time.perf_counter_ns()``. GPU samples replay an explicit Warp
``ScopedCapture`` of ``generate(advance=True, sync=False)``, bracketed by
timing-enabled Warp events on the same stream. GPU event time covers device execution;
separately recorded wall time
also includes host submission and waiting for completion. The comparison table
uses CPU wall time and GPU event time, in milliseconds per batch. It compares the
measured systems, execution modes and batch sizes; it does not isolate an architecture
speedup. Attempted tracks per second is ``1000 * batch_size / median_ms`` and counts
failed candidates too. Throughput depends on batch size: this normalization does
not predict a CPU batch of 4,096 or a GPU batch of 64, and the batch latencies cannot
be compared as equal amounts of work.

Initialization, construction (which runs a generation), CUDA graph recording and
warm-up are excluded from steady latency. Warm-up runs at least three complete
calls (on CUDA, the first replay and two more) and lasts at least one second.
After warm-up, epochs reset to zero. Every family receives explicitly materialized
``int32`` seeds ``0..63`` on CPU and ``0..4095`` on GPU; the five measured calls
advance to epochs ``1..5``. The table reports the median of those five latencies.
Validity is read
after each completed timing interval and includes every candidate per call: 64 on
CPU or 4,096 on GPU.
It is a final geometric validity measurement for this workload, not a vehicle's
drivability score or a universal yield guarantee.

Before measurement, the runner verifies that recording does not execute the graph,
replay advances epochs once, generation metadata matches, and replay overwrites
poisoned result storage. It also checks that finite valid geometry changes between
epochs; all-invalid poisoned output alone cannot establish that regeneration ran.

Reproduce a run from a checkout, recording its source revision:

.. code-block:: bash

   python -m benchmarks.benchmark_generators --device cpu --envs 64 \
       --generators bezier checkpoint hull polar repulsive voronoi --repeats 5 \
       --warmup-seconds 1 --timeout-seconds 3600 \
       --output-dir /tmp/trackgen-mac-benchmarks --label "Mac CPU" \
       --source-revision <commit>
   python -m benchmarks.benchmark_generators --device cuda:0 --envs 4096 \
       --generators bezier checkpoint hull polar repulsive voronoi --repeats 5 \
       --warmup-seconds 1 --timeout-seconds 1800 \
       --output-dir /tmp/trackgen-gpu-benchmarks --label "GPU" \
       --source-revision <commit>

Both commands include all six families, including an actual 64-candidate CPU
Repulsive run. The timeout applies to each generator's whole subprocess, including
setup: 3,600 seconds for CPU and 1,800 for GPU here; zero disables it.
Generators run sequentially, and a requested CUDA device never falls back to CPU.
Avoid other workloads on the measured device.

An out-of-memory error, unsupported configuration, unmeasured case or timeout is
reported as that outcome. A timeout names its phase: timing out during construction
does not establish a lower bound on warm batch latency. Keep the same batch size,
resolution and iteration budget when a case is expensive. A smaller workload belongs
in a separately labelled comparison.

Preserve the per-call latency and validity samples, timing ranges, setup/capture
costs, resolved configuration, seeds and epochs with each result. Record the source
revision, protocol/source hashes, date, CPU/GPU model, OS, driver and Python, Warp
and NumPy versions. The download separates the installed NVIDIA driver version from
CUDA driver API compatibility, and records CUDA-visible total memory separately from
NVIDIA-SMI memory reporting.

Quality and shape diagnostics
-----------------------------

The separate ``benchmarks.compare_generators`` command computes geometric metrics
from actual generated batches, including the shared road pipeline. Its default
configuration differs from the quickstart workload above. Start with a small CPU
sampling-family check:

.. code-block:: bash

   python -m benchmarks.compare_generators --E 32 --seed 0 \
       --generators bezier checkpoint hull polar voronoi

For CUDA diagnostics, run the same suite on the same GPU and separate ``repulsive``
if its iterative work dominates the run:

.. code-block:: bash

   python -m benchmarks.compare_generators --E 512 --seed 0 --cuda \
       --generators bezier checkpoint hull polar voronoi
   python -m benchmarks.compare_generators --E 32 --seed 0 --cuda \
       --generators repulsive

These commands are diagnostic recipes, not the protocol for the latency table.
Record the source commit, Warp version, platform/GPU, exact config and seeds with
the output. ``compare_generators`` repeats the same seeds and epochs: its
``gen_ms_per_call`` is the mean wall time of five calls after two warm calls, and
its yield describes the initial batch. It does not produce the advancing-epoch
median or GPU device-event time above. Changing batch size changes amortization.
Use several seed batches before drawing conclusions about rare failures or diversity.

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
and steady graph replay are distinct costs. A warm median or mean alone does not
show the worst reset pause or the memory budget.

Read the implementation in ``benchmarks/compare_generators.py`` and its metric
helpers when changing thresholds or adding a task-specific score. Choose a family
with :doc:`/tutorials/choosing-a-generator`; inspect the raw/final overlays in its
guide before interpreting a metric difference as a pure generator effect.
