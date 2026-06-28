# Docs Landing Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bare `docs/index.rst` with a rich landing page — hero + CTA buttons, a six-card feature summary, two gallery figures, and six section-navigation cards — using `sphinx-design` and existing committed figures.

**Architecture:** Single file. The nine section toctrees become `:hidden:` (Furo sidebar nav stays identical) and the page body above them is hand-authored with `sphinx-design` directives (`grid`, `grid-item`, `grid-item-card`, `button-ref`, `button-link`) plus `figure`. No new assets, no CSS, no other file changes.

**Tech Stack:** Sphinx, Furo, `sphinx-design` (already enabled in `conf.py`), reStructuredText.

## Global Constraints

- Only `docs/index.rst` changes. No new assets, no CSS/theme overrides, no edits to other pages or `conf.py`.
- Pure reStructuredText with `sphinx-design` directives. All figure paths are `assets/<name>.png` (relative to `docs/`, NOT `../assets/` — index.rst is at the docs root).
- The strict build must stay clean: `.venv/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html` → zero warnings (every `button-ref`/`:link:` doc target must resolve).
- All nine existing toctree blocks are preserved verbatim except for an added `:hidden:` option — captions and entries unchanged.
- GitHub URL is exactly `https://github.com/AntoineRichard/TrackGen`.

---

### Task 1: Rich landing page in `docs/index.rst`

**Files:**
- Modify (full rewrite): `docs/index.rst`

**Interfaces:**
- Consumes: existing committed figures `docs/assets/readme-pipeline-stages.png`, `readme-generator-grid.png`, `readme-gate-strip.png`; existing doc targets `getting-started/installation`, `getting-started/quickstart`, `getting-started/parameter-explorer`, `tutorials/batch-of-tracks`, `generators/overview`, `how-it-works/pipeline`, `configuration/tuning`, `reference/api` (all confirmed present).
- Produces: the site homepage; sidebar nav unchanged.

- [ ] **Step 1: Replace the entire contents of `docs/index.rst` with:**

```rst
track_gen
=========

**GPU-batched generation of closed-loop race tracks** — thousands of smooth,
validated tracks per ``generate()`` call, expressed as NVIDIA Warp kernels and
ready to drop into a batched RL simulator.

.. grid:: 3
   :gutter: 2

   .. grid-item::
      :columns: auto

      .. button-ref:: getting-started/installation
         :ref-type: doc
         :color: primary

         Get started

   .. grid-item::
      :columns: auto

      .. button-ref:: generators/overview
         :ref-type: doc
         :color: primary
         :outline:

         Browse generators

   .. grid-item::
      :columns: auto

      .. button-link:: https://github.com/AntoineRichard/TrackGen
         :color: secondary
         :outline:

         GitHub

.. figure:: assets/readme-pipeline-stages.png
   :alt: TrackGen pipeline stages
   :align: center

   The runtime pipeline turns a raw Phase-1 centerline into a constant-spacing path,
   relaxes it with XPBD, then inflates it into a constant-width road band.

Features
--------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: GPU-batched

      Generate ``E`` tracks in parallel per ``generate()`` call — the Warp ``cpu``
      device for tests/CI, ``cuda`` for production.

   .. grid-item-card:: Pure Warp pipeline

      Generation → constant-spacing resample → XPBD relaxation → inflation, every
      stage a Warp kernel over flat ``[E*N]`` arrays.

   .. grid-item-card:: Five generators

      Bezier, Hull, Polar, Voronoi, and Checkpoint — pluggable first-stage
      generators, each with a distinct shape family.

   .. grid-item-card:: CUDA-graph capture

      The whole pipeline is captured once into a replayable CUDA graph and replayed
      on every later call for high throughput.

   .. grid-item-card:: Gate sequences

      Drone-style gate courses — gate centres and orientations — straight from the
      first-stage anchors via ``GateGenerator``.

   .. grid-item-card:: RL-ready output

      Constant-width outer / center / inner borders, per-point tangent and normal
      frames, per-track validity, and real-point counts.

Gallery
-------

.. figure:: assets/readme-generator-grid.png
   :alt: Five generators, one batch
   :align: center

   Five generators, one batch — representative raw Phase-1 centerlines from each
   standard generator.

.. figure:: assets/readme-gate-strip.png
   :alt: Gate sequences with collision relaxation
   :align: center

   Gate sequences with the phase-2 collision solve — raw anchors (top) versus
   separated gates (bottom).

Explore the docs
----------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting-started/installation
      :link-type: doc

      Install the library and generate your first batch.

   .. grid-item-card:: Tutorials
      :link: tutorials/batch-of-tracks
      :link-type: doc

      End-to-end recipes for tracks, gates, and CUDA-graph sim loops.

   .. grid-item-card:: Generators
      :link: generators/overview
      :link-type: doc

      How each first-stage generator works and when to use it.

   .. grid-item-card:: How it works
      :link: how-it-works/pipeline
      :link-type: doc

      The pipeline, XPBD relaxation, inflation, and CUDA-graph capture.

   .. grid-item-card:: Configuration & tuning
      :link: configuration/tuning
      :link-type: doc

      Every knob, plus a guide to trading yield, diversity, and throughput.

   .. grid-item-card:: API reference
      :link: reference/api
      :link-type: doc

      ``TrackGenerator``, ``GateGenerator``, configs, and result types.

.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   getting-started/installation
   getting-started/quickstart
   getting-started/parameter-explorer

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/batch-of-tracks
   tutorials/gate-sequences
   tutorials/choosing-a-generator
   tutorials/cuda-graph-in-a-sim

.. toctree::
   :maxdepth: 1
   :caption: Generators
   :hidden:

   generators/overview
   generators/bezier
   generators/hull
   generators/polar
   generators/voronoi
   generators/checkpoint
   generators/benchmarks

.. toctree::
   :maxdepth: 1
   :caption: How it works
   :hidden:

   how-it-works/pipeline
   how-it-works/resample
   how-it-works/relaxation
   how-it-works/inflation
   how-it-works/cuda-graph
   how-it-works/conventions

.. toctree::
   :maxdepth: 1
   :caption: Configuration & tuning
   :hidden:

   configuration/reference
   configuration/tuning

.. toctree::
   :maxdepth: 1
   :caption: API reference
   :hidden:

   reference/api

.. toctree::
   :maxdepth: 1
   :caption: Contributing
   :hidden:

   contributing/writing-a-generator
   contributing/dev-setup
   contributing/rendering-assets

.. toctree::
   :maxdepth: 1
   :caption: Related work
   :hidden:

   related-work/prior-art
   related-work/state-of-the-art

.. toctree::
   :maxdepth: 1
   :caption: Appendix
   :hidden:

   appendix/future-generators
```

- [ ] **Step 2: Strict build (the gate)**

Run: `.venv/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html`
Expected: `build succeeded`, ZERO warnings. If a `button-ref`/`:link:` target warns ("unknown document" / "undefined label"), the doc path is wrong — fix the path; do not drop `-W`. If a `sphinx-design` option is rejected (e.g. an unknown option on a directive), correct the directive option rather than removing the directive.

- [ ] **Step 3: Linkcheck**

Run: `.venv/bin/python -m sphinx -b linkcheck docs docs/_build/linkcheck 2>&1 | tail -20`
Expected: internal links OK; the new `https://github.com/AntoineRichard/TrackGen` resolves (200). If it flakily fails (rate-limited), it is already covered or may be added to `linkcheck_ignore` in `conf.py` — but the repo URL should resolve.

- [ ] **Step 4: Eyeball the rendered homepage**

Open `docs/_build/html/index.html` with the Read tool. Confirm: the hero tagline + three buttons (Get started, Browse generators, GitHub), the pipeline hero figure, six feature cards (GPU-batched, Pure Warp pipeline, Five generators, CUDA-graph capture, Gate sequences, RL-ready output), two gallery figures (generator grid, gate strip), six nav cards, and that the left sidebar still lists all nine caption sections (Getting started … Appendix). If a card grid or figure is obviously broken, fix the directive and rebuild.

- [ ] **Step 5: Commit**

```bash
git add docs/index.rst
git commit -m "docs: rich landing page with hero, feature cards, and gallery"
```

---

## Self-Review

**Spec coverage:**
- Hero (tagline + pipeline figure + Get started / Browse generators / GitHub buttons) → Step 1 hero block. ✔
- Six feature cards (exact text) → Step 1 Features grid. ✔
- Two gallery figures (generator grid, gate strip) → Step 1 Gallery. ✔
- Six section-nav cards → Step 1 Explore-the-docs grid. ✔
- Hidden toctrees (nav unchanged) → Step 1 toctree blocks with `:hidden:`. ✔
- Verification: clean `-W` build + linkcheck + eyeball → Steps 2–4. ✔
- Scope (only index.rst) → Files + Global Constraints. ✔

**Placeholder scan:** No TBD/TODO; the full rST file content is literal and complete. ✔

**Type/name consistency:** All `button-ref`/`:link:` doc targets (`getting-started/installation`, `generators/overview`, `tutorials/batch-of-tracks`, `how-it-works/pipeline`, `configuration/tuning`, `reference/api`) and figure paths (`assets/readme-pipeline-stages.png`, `assets/readme-generator-grid.png`, `assets/readme-gate-strip.png`) match files confirmed present in the repo. The nine `:hidden:` toctrees reproduce the current entries exactly. ✔
