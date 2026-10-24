.. _generator-benchmarks:

Generator Benchmarks
====================

Reference metrics for the six registered centerline generators, produced by
``benchmarks/compare_generators.py``. They characterize trade-offs among quality,
diversity and speed, and never gate which generators ship: every registered generator
stays selectable by name. :ref:`choosing-a-generator` turns them into a recommendation.

Suite: seed base 0, ``E = 512``, the default ``TrackGenConfig`` with ``generator=`` set.
Every number on this page is written by the commands under `Reproduce`_ into
``docs/_data/`` and shown from there, never typed by hand.

Sampling generators (CPU)
-------------------------

The ``bezier``, ``checkpoint``, ``hull``, ``polar`` and ``voronoi`` rows run on Warp's
``cpu`` device. Every column but the last is deterministic for the suite (the same
values on every run of the same Warp version and platform); ``gen_ms_per_call`` depends
on the machine and its load, so compare it across rows of one table only.

.. csv-table:: Generator metrics, ``cpu``, ``E = 512``, seed base 0
   :file: ../_data/metrics-cpu.csv
   :header-rows: 1

Machine:

.. include:: ../_data/machine-cpu.txt

``repulsive`` (CUDA)
--------------------

``repulsive`` is an iterative optimizer (:doc:`repulsive`), so its suite runs on
``cuda``: on ``cpu`` the same suite takes far longer and tells little about production
use. Its ``gen_ms_per_call`` is therefore **not comparable** to the CPU table above, and
depends on how soon the batch converges (its own CUDA graph skips the iterations left
once every env has frozen). Its metric columns are comparable.

.. csv-table:: ``repulsive`` metrics, ``cuda``, ``E = 512``, seed base 0
   :file: ../_data/metrics-repulsive-cuda.csv
   :header-rows: 1

Machine:

.. include:: ../_data/machine-cuda.txt

The generation cost of every generator at large batches on CUDA is in
:ref:`the catalogue <generators-overview>` (its Cost section).

Reproduce
---------

From the repository root (``docs/_data/README.md`` lists every data file and its
command):

.. code-block:: bash

   python -m benchmarks.compare_generators --E 512 --seed 0 \
       --generators bezier checkpoint hull polar voronoi \
       | python docs/_tools/md_table_to_csv.py > docs/_data/metrics-cpu.csv
   python -m benchmarks.compare_generators --E 512 --seed 0 --cuda --generators repulsive \
       | python docs/_tools/md_table_to_csv.py > docs/_data/metrics-repulsive-cuda.csv

``compare_generators`` prints a Markdown table (after Warp's start-up lines, which go to
stdout too); ``md_table_to_csv.py`` keeps only the table rows.

Metric Definitions
------------------

**yield**
   Fraction of generated environments that pass the full post-relax validity gate
   (``Track.valid``: turning number, thickness floor, no NaN, not truncated at
   :term:`N_max`, see :ref:`pipeline-fail-bits`); higher is better.

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
   shape diversity across the batch.

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
   Wall-clock time in milliseconds per ``TrackGenerator.generate()`` call: the mean of
   five calls after two warm ones, at ``E = 512``. Machine-dependent: use it for
   relative comparison within one table. The ``repulsive`` row is measured on ``cuda``
   and is not comparable to the CPU rows.
