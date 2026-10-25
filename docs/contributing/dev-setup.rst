.. _dev-setup:

Development Setup
=================

Prerequisites
-------------

Python 3.10 or later is required. ``warp-lang`` (``>=1.18,<2``) and ``numpy`` are the
only required runtime dependencies. Tests, benchmarks and oracle comparisons need the
``dev`` extra (``torch``, ``scipy``, ``matplotlib``, ``pytest``); building the docs needs
the ``docs`` extra (``sphinx``, ``furo``, ``sphinx-copybutton``, ``sphinx-design``).

Editable install with the dev extra
------------------------------------

Using ``uv`` (recommended):

.. code-block:: bash

   # Install uv if not already present
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create a project venv — uv fetches Python 3.12 if it isn't present
   uv venv --python 3.12

   # Install track_gen (editable) with the dev and docs dependencies
   uv pip install -e ".[dev,docs]"

Using plain ``venv`` and ``pip``:

.. code-block:: bash

   python -m venv .venv
   .venv/bin/pip install -e ".[dev,docs]"

Both approaches create a ``.venv/`` directory. Run anything inside it with
``.venv/bin/python …`` (or ``source .venv/bin/activate``, or ``uv run …``).

Running the test suite
-----------------------

.. warning::

   **If pytest fails during collection, set** ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``.

   Some system-wide installations (a ROS 2 install under ``/opt/ros``, for example) ship
   pytest plugins that pytest's plugin discovery picks up automatically, and that break
   collection before a single test runs. The variable turns that discovery off; the
   commands below set it.

Fast lane
~~~~~~~~~

Skips slower quality gates, benchmark checks and CUDA-only tests: what the CPU CI lane
runs. Suitable for rapid iteration during development:

.. code-block:: bash

   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -m "not slow and not benchmark and not cuda"

Full suite
~~~~~~~~~~

Runs all tests. CUDA-only tests are skipped automatically when Warp sees no CUDA
device:

.. code-block:: bash

   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q

GPU lane
~~~~~~~~

On a CUDA machine, run the ``cuda`` tests with ``TRACKGEN_REQUIRE_CUDA=1``. It is a
test-suite variable (``tests/conftest.py``): the session errors out at start-up when
Warp sees no CUDA device, and fails at the end when a ``cuda`` test was skipped or when
none ran, so a missing GPU can never pass silently.

.. code-block:: bash

   TRACKGEN_REQUIRE_CUDA=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -m cuda -ra

Byte goldens and the host
~~~~~~~~~~~~~~~~~~~~~~~~~

The byte-identity goldens (``tests/goldens/``) are keyed by platform (OS, architecture and
Warp version), but a few fields also depend on the machine that captured them: the
``checkpoint`` track generator's path tracer is chaotic, so last-ulp differences of the C
library's transcendental functions or of the CPU's code generation grow into visible
differences between two ``x86_64`` hosts. ``TRACKGEN_SKIP_HOST_SENSITIVE_GOLDENS=1``
leaves those fields out and compares every other one; CI sets it, because its runners are
not the capture machines. Leave it unset on the machine a golden was captured on.

Capturing a new byte baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After an intentional geometry change, capture one fragment per platform against
identical source, then merge every fragment together:

.. code-block:: bash

   python -m tests.tools.capture_03_goldens capture golden-fragment.json
   python -m tests.tools.capture_03_goldens merge darwin-fragment.json linux-fragment.json cuda-fragment.json

Capture runs all five geometry and utility reference families. Merge checks source
hashes and complete family coverage before writing any golden; it replaces the stored
platform set with the supplied fragments. The manual test workflow's
``capture_goldens`` input captures the Linux runner fragment as an artifact. The
reference captures explicitly use ``on_invalid="keep"`` so the 0.3 poison default
does not replace failed geometry in the byte baseline. Public-default poison behavior
has separate contract tests.

A changed seed mapping also needs the empirical distribution check described in
:ref:`upgrading-0.3-rng`; replacing byte hashes alone does not establish that the
shape distributions stayed close.

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
     - Tests that require a CUDA device; skipped automatically when Warp sees none
       (and a failure under ``TRACKGEN_REQUIRE_CUDA=1``).

.. _dev-setup-docs:

Building and checking the docs
------------------------------

The docs are a Sphinx project under ``docs/`` (``docs/conf.py``), built with warnings as
errors. Nitpicky mode is on: every cross-reference (``:ref:``, ``:doc:``, ``:term:``,
``:class:``, ...) must resolve. A local extension, ``docs/_ext/trackgen_docscheck.py``,
also checks that every public name is documented, that no page outside
``docs/upgrading/`` carries release or planning jargon, that every data file a page
includes exists, and that every entry of ``docs/_tools/nitpick_allowlist.txt`` still
silences an unresolved reference (remove the entries it reports as stale).

**Build** (from the repository root). ``docs/_tools/build_ci_env.py`` runs Sphinx as the
docs CI does, on this checkout (it prints where ``track_gen`` is imported from), with
``pytest``, ``torch``, ``scipy`` and ``matplotlib`` blocked (the docs workflow installs
only ``.[docs]``), so an import that only CI would miss fails locally too. Always pass ``-E`` and a private output and doctree directory: an incremental build
does not re-read unchanged pages and can pass falsely.

.. code-block:: bash

   .venv/bin/python docs/_tools/build_ci_env.py -W --keep-going -E -b html \
       -d /tmp/trackgen-doctrees docs /tmp/trackgen-html

The extension's findings are printed during the build and written to
``<doctrees>/docscheck.txt``. Then check the built site:

.. code-block:: bash

   # every label and glossary term the site promises still exists, on its page
   .venv/bin/python docs/_tools/check_labels.py /tmp/trackgen-html
   # every link from README.md into the published docs, anchors included
   .venv/bin/python docs/_tools/check_readme_anchors.py /tmp/trackgen-html README.md

**Executed examples.** Every ``.. code-block:: python`` of every page (and the Python
blocks of ``README.md`` and of the public docstrings) runs in the CPU test lane, through
``tests/test_doc_snippets.py``, on the ``cpu`` device (a ``"cuda"`` / ``"cuda:0"``
string becomes ``"cpu"``). A block can carry metadata in an RST comment right before it
(``.. snippet: group=<name>``, ``requires=``, ``register=``, ``setup``, ``after``,
``cuda``, ``slow``, ``skip=<reason>``, ``expect-warning=``, ``expect-error=``); the
grammar is in the docstring of ``tests/_doc_snippets.py``. Run one page, or print the
per-page report (blocks run, skipped, and why):

.. code-block:: bash

   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_doc_snippets.py -k "how-it-works/pipeline-stages.rst"
   .venv/bin/python tests/_doc_snippets.py

Test ids are ``<page>::<group>``; a ``cuda`` group only runs in the GPU lane
(``-m cuda``). ``tests/test_generator_docs.py`` (the generator guides and the
configuration reference) and ``tests/test_docs_links.py`` (README links, excluded
directories) also read the docs, without building them.

**External links.** The link check runs in its own workflow, not in the build (external
sites are slow and some refuse automated clients); run it before changing many links:

.. code-block:: bash

   .venv/bin/python -m sphinx -b linkcheck -q docs /tmp/trackgen-linkcheck

**Numbers and figures.** Benchmark tables come from the files in ``docs/_data/``, each
written by the command its ``README.md`` lists; figures come from the scripts in
:doc:`rendering-assets`. Never type a measured number into a page.

Additional development commands
---------------------------------

Compare every registered centerline generator on quality, diversity, and
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
