.. _rendering-assets:

Render figures and animations
=============================

Figures are generated from current TrackGen output or explicitly labeled geometric
fixtures, then committed as static assets. The site build does not run a simulation or
require plotting/video libraries. Reproduction information lives beside the assets in
``docs/assets/racing/*manifest.json``.

Render to scratch first
------------------------

Use a development environment with Matplotlib. Commands below run on CPU and write
selected figure families to a scratch directory; inspect the output before copying it.

.. code-block:: bash

   MPLCONFIGDIR=/tmp/trackgen-mpl python -m viz.render_generator_guide --out-dir /tmp/tg-generators
   MPLCONFIGDIR=/tmp/trackgen-mpl python -m viz.render_course_guide --out-dir /tmp/tg-controls
   MPLCONFIGDIR=/tmp/trackgen-mpl python -m viz.render_collision_guide --out-dir /tmp/tg-geometry
   MPLCONFIGDIR=/tmp/trackgen-mpl python -m viz.render_data_guide --out-dir /tmp/tg-data

.. list-table:: Asset families
   :header-rows: 1
   :widths: 25 42 33

   * - Renderer / prefix
     - What it explains
     - Data and interpretation
   * - ``render_generator_guide`` / ``generator-*``, ``generators-*``
     - Six families, centerlines/final roads, knob comparisons and construction clips
     - Fixed seeds and real buffers; construction phases are labeled; repulsive growth records actual iterations
   * - ``render_course_guide`` / ``controls-*``
     - Size/width, sampling/capacity, elevation, gate sources, spacing and frames
     - Same input/seed where comparing a control; numerical checks prevent vacuous comparisons
   * - ``render_collision_guide`` / ``geometry-*``
     - Validity, crossing predicates, footprints, SDF resolution and obstacles
     - Synthetic fixtures are explicit; labels report real operation/query results
   * - ``render_data_guide`` / ``data-*``
     - Flat layout, borrowed storage and imported routes
     - Layout/lifecycle diagrams are schematic; the imported road and gates are real public-API output

The manifest records source version, settings, seeds or fixture coordinates, sample
selection and checks. A selected gallery sample is not a yield estimate. To show a
failed candidate, use ``on_invalid="keep"`` explicitly and retain its validity/failure
labels. Otherwise the default poison policy correctly removes its geometry.

Solver convergence clips
------------------------

``viz.render_relaxation_video`` renders the track and gate convergence movies and their
posters. It needs ffmpeg; ``imageio-ffmpeg`` is an optional way to supply a local binary:

.. code-block:: bash

   pip install imageio-ffmpeg
   python -m viz.render_relaxation_video --help

Use the CLI's output-directory option to render in scratch. The track sequence compares
actual pipeline output at successive solve budgets, with smoothing off, then separate
smoothing and spacing-polish phases. Each frame still passes through final resampling
and inflation: it is not a raw bead-buffer capture. The gate clip shows actual round
budgets. Captions and overlays state the chosen seeds, counts, validity and phase.
A visually settled shape can still fail checks.

Videos use a fixed view and a short final hold. Keep a poster and a nearby written
explanation, use player controls, and avoid autoplay. The generator MP4 files are copied
by ``html_extra_path``; solver videos/posters live in ``docs/_static``. Both routes leave
the Sphinx build free of rendering dependencies.

Inspect and publish selected output
-----------------------------------

Check every still at the size it will appear on the page (about 736 CSS pixels on a
wide-screen Furo layout). Check a video's beginning, transition frames and end. Look for
clipped labels, misleading axes, poisoned rows, non-changing comparisons and unstated
fallback selection. Confirm the poster and the text alternative explain the main point.

Copy only the selected current assets. Update the corresponding manifest and page
caption together. CPU/GPU floating-point differences and font rasterization can change
pixels across machines; image-byte equality is not the correctness criterion. Verify
data/counts/checks and inspect the render.

For simulator utility figures, ``viz.render_utility_assets`` provides additional
checkpoint/progress/props illustrations. Prefer the focused racing renderers for the
reader guides. To add another explanation, use a static comparison for a setting's
result, and an animation when order or iteration is the point. Manim can be an optional
tool for original mathematical diagrams; solver motion must continue to come from real
TrackGen states.
