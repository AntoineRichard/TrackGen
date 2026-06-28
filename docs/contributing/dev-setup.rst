:orphan:

Development Setup
=================

Prerequisites
-------------

Python 3.10 or later is required.  ``warp-lang`` and ``numpy`` are the only
required runtime dependencies.  Tests, benchmarks, and oracle comparisons need
the ``dev`` extra (``torch``, ``scipy``, ``matplotlib``, ``pytest``).

Editable install with the dev extra
------------------------------------

Using ``uv`` (recommended):

.. code-block:: bash

   # Install uv if not already present
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create a project venv — uv fetches Python 3.12 if it isn't present
   uv venv --python 3.12

   # Install track_gen (editable) with all dev dependencies
   uv pip install -e ".[dev]"

Using plain ``venv`` and ``pip``:

.. code-block:: bash

   python -m venv .venv
   .venv/bin/pip install -e ".[dev]"

Both approaches create a ``.venv/`` directory.  Run anything inside it with
``.venv/bin/python …`` (or ``source .venv/bin/activate``, or ``uv run …``).

Running the test suite
-----------------------

.. warning::

   **Always run pytest with the** ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``
   **environment variable set.**

   The system-wide ROS 2 Humble installation at ``/opt/ros/humble`` ships
   broken pytest plugins that are picked up automatically by pytest's plugin
   discovery.  Without this variable, test collection fails before a single
   test runs.

   Example fast-lane command (copy-paste ready):

   .. code-block:: bash

      PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -m "not slow and not benchmark and not cuda"

Fast lane
~~~~~~~~~

Skips slower quality gates, benchmark checks, and CUDA-only graph tests.
Suitable for rapid iteration during development:

.. code-block:: bash

   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -m "not slow and not benchmark and not cuda"

Full suite
~~~~~~~~~~

Runs all tests.  CUDA-only assertions are skipped automatically when no CUDA
device is available:

.. code-block:: bash

   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q

Pytest markers
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Marker
     - Meaning
   * - ``slow``
     - Quality gates that are useful but heavier than smoke tests.
   * - ``benchmark``
     - Benchmark harness checks.
   * - ``cuda``
     - Tests that require a CUDA device; skipped automatically when CUDA is
       unavailable.

Additional development commands
---------------------------------

Compare every registered first-stage generator on quality, diversity, and
speed:

.. code-block:: bash

   .venv/bin/python -m benchmarks.compare_generators --E 512 --seed 0

End-to-end pipeline benchmark (auto device, E=8192):

.. code-block:: bash

   # Capture and time the CUDA graph
   .venv/bin/python -m benchmarks.benchmark_pipeline --graph

   # CPU run with a smaller batch
   .venv/bin/python -m benchmarks.benchmark_pipeline --E 2048 --cpu

Render sample tracks without launching the interactive UI:

.. code-block:: bash

   .venv/bin/python -m viz.plot_tracks --images 1 --rows 4 --cols 4 --cpu
