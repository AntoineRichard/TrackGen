Installation
============

Python ≥ 3.10. This is a Warp-first library: ``warp-lang`` and ``numpy`` are the only
required runtime dependencies. ``torch``, ``scipy``, ``matplotlib``, and ``pytest`` live in the
``dev`` extra for tests, benchmarks, and oracle comparisons; ``gradio`` lives in the ``ui``
extra. The runtime path runs on the Warp ``cpu`` device (GPU-free, for tests/CI) and on
``cuda``.

From scratch with uv (recommended)
-----------------------------------

`uv <https://docs.astral.sh/uv/>`_ is the recommended way to set up the project.

.. code-block:: bash

   # 1. install uv (skip if you already have it)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # 2. create the project venv — uv fetches Python 3.12 if it isn't present
   uv venv --python 3.12

   # 3. install track_gen (editable) with the dev extras (warp-lang comes in as a core dep)
   uv pip install -e ".[dev]"

   # 4. verify the fast lane
   .venv/bin/python -m pytest -q -m "not slow and not benchmark and not cuda"

   # Optional: add the Gradio UI and open the parameter explorer
   uv pip install -e ".[ui]"
   uv run python -m viz.param_explorer   # opens a local URL (default http://127.0.0.1:7860)

With venv + pip
---------------

.. code-block:: bash

   python -m venv .venv
   .venv/bin/pip install -e ".[dev]"

Both approaches create a ``.venv/``; run anything in it with ``.venv/bin/python …``
(or ``source .venv/bin/activate``, or ``uv run …``).

Extras
------

.. list-table::
   :header-rows: 1

   * - Extra
     - Packages added
   * - ``dev``
     - ``pytest``, ``matplotlib``, ``scipy``, ``torch``
   * - ``ui``
     - ``gradio``

CPU and CUDA
------------

Core deps are ``numpy`` and ``warp-lang``. The runtime path runs on the Warp ``cpu`` device
(GPU-free, suitable for tests and CI) and on ``cuda`` (production). No CUDA installation is
required to install or run the CPU path.
