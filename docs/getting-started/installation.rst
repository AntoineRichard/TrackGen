.. _installation:

Installation
============

TrackGen is a Warp-first library: ``warp-lang`` and ``numpy`` are its only runtime
dependencies, and every pipeline runs on the Warp ``cpu`` device as well as on ``cuda``.

This guide and its examples target ``v0.3.0rc1``, a release candidate for downstream
racing-repository validation.

Requirements
------------

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Requirement
     - Version
     - Notes
   * - Python
     - 3.10 or newer
     -
   * - ``warp-lang``
     - ``>=1.18,<2``
     - Installed with TrackGen.
   * - ``numpy``
     - any
     - Installed with TrackGen.
   * - NVIDIA driver
     - R580 or newer
     - For the ``cuda`` device with Warp's PyPI wheels. The ``cpu`` device needs no GPU,
       driver or CUDA installation.
   * - NVIDIA GPU
     - compute capability 7.5 (Turing) or newer (``sm_75+``)
     - For the ``cuda`` device with Warp's PyPI wheels.
   * - CUDA 12.4 or newer, in the driver and in Warp's build
     - optional
     - Lets a masked regeneration recorded into a CUDA graph be skipped on device when no
       :term:`env` is selected (a :term:`conditional node`). Check it with
       ``wp.is_conditional_graph_supported()``. Without it the recorded call always runs.
   * - PyTorch
     - optional
     - Only for :mod:`track_gen.interop.torch` (recording into ``torch.cuda.graph``) and
       for zero-copy ``wp.to_torch`` views of the results. TrackGen never imports it on its
       own.

Install into a project
----------------------

With `uv <https://docs.astral.sh/uv/>`_, add TrackGen to an existing project from its Git
repository. ``uv add`` records the dependency in your ``pyproject.toml``, resolves it into
``uv.lock`` and syncs the environment:

.. code-block:: bash

   # pin the release candidate used by this guide
   uv add "git+https://github.com/AntoineRichard/TrackGen.git@v0.3.0rc1"

   # with an extra (here the Gradio parameter explorer's)
   uv add "track_gen[ui] @ git+https://github.com/AntoineRichard/TrackGen.git@v0.3.0rc1"

   # or a local v0.3.0rc1 checkout (see below), as an editable path dependency
   uv add --editable /path/to/TrackGen

``uv sync`` later rebuilds the environment from the lockfile (on a fresh clone, or in CI).

With pip:

.. code-block:: bash

   pip install "track_gen @ git+https://github.com/AntoineRichard/TrackGen.git@v0.3.0rc1"

From a checkout
---------------

To work on TrackGen itself, or to run the examples, tests and tools of the repository,
install the checkout in editable mode with the ``dev`` extra:

.. code-block:: bash

   git clone --branch v0.3.0rc1 https://github.com/AntoineRichard/TrackGen.git && cd TrackGen

   # uv (fetches Python 3.12 if it is missing)
   uv venv --python 3.12
   uv pip install -e ".[dev]"

   # or venv + pip
   python -m venv .venv
   .venv/bin/pip install -e ".[dev]"

   # the fast test lane: no GPU needed
   .venv/bin/python -m pytest -q -m "not slow and not benchmark and not cuda"

Both create a ``.venv/``: run anything in it with ``.venv/bin/python``, ``uv run``, or
after ``source .venv/bin/activate``.

Extras
------

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Extra
     - Packages added
     - For
   * - ``dev``
     - ``pytest``, ``matplotlib``, ``scipy``, ``torch``
     - the test suite, the benchmarks and the figure scripts
   * - ``ui``
     - ``gradio``
     - the :doc:`parameter explorer <parameter-explorer>` (which also needs ``dev``)
   * - ``docs``
     - ``sphinx``, ``furo``, ``sphinx-copybutton``, ``sphinx-design``
     - building this documentation (see :ref:`dev-setup`)

Check the install
-----------------

.. code-block:: bash

   python -c "import track_gen, warp as wp; print(track_gen.__version__, wp.__version__, wp.get_cuda_device_count(), wp.is_conditional_graph_supported())"

Warp initializes at the first device query in this command and prints its banner first:
the CUDA toolkit and driver versions, and each device with its architecture (``sm_75``
or higher is needed).
The last line prints the TrackGen version, the Warp version (1.18 or newer), the number of
CUDA devices (0 on a machine without a usable GPU: use ``device="cpu"``) and whether
conditional nodes are available.

CPU and CUDA
------------

Every generator, course and utility takes a ``device``: ``"cuda:0"`` (or ``"cuda"``) on a
GPU, ``"cpu"`` anywhere. The two run the same kernels, so the ``cpu`` device suits tests
and CI; the ``cuda`` device is the one to use for large batches and in a simulator. The
examples in this documentation write ``device = "cuda:0"  # or "cpu"``.

Next: the :doc:`quickstart`.
