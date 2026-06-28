# Docs landing page — design

**Date:** 2026-06-28
**Status:** Approved (brainstorming), pending spec review

## Problem

The Sphinx docs site (live at https://antoinerichard.github.io/TrackGen/) has a bare
landing page: `docs/index.rst` is a title, a one-line description, and the section
toctrees (which Furo also renders as the left sidebar). The main content area is empty
above those lists, so the homepage does not communicate what `track_gen` is or showcase it.

## Goal

A rich, modern landing page in `docs/index.rst` — a hero with a figure and call-to-action
buttons, a feature-summary card grid, a small figure showcase, and section-navigation cards
— built entirely with the already-enabled `sphinx-design` extension and the figures already
committed under `docs/assets/`. The strict `-W` build must stay clean.

## Decisions (locked in brainstorming)

- **Mechanism:** convert the existing section toctrees in `index.rst` to `:hidden:` (the
  left sidebar nav stays identical), and hand-author the body above them.
- **Tooling:** `sphinx-design` directives (`.. grid::`, `.. card::`, `.. button-ref::`,
  `.. button-link::`) + `.. figure::`. Both extensions (`sphinx_design`, `sphinx_copybutton`)
  are already in `conf.py`.
- **Figures:** reuse three existing assets — no new rendering.
- **Scope:** only `docs/index.rst` changes. No theme/CSS files, no new assets, no edits to
  other pages.

## Non-goals

- No new rendered figures or changes to `viz/render_readme_assets.py`.
- No custom CSS / theme overrides / Furo option changes.
- No content changes to any page other than `index.rst`.

## Page structure (main content area of `index.rst`)

1. **Hero**
   - H1 `track_gen` + a tagline: "GPU-batched generation of closed-loop race tracks —
     thousands of smooth, validated tracks per call, expressed as NVIDIA Warp kernels."
   - CTA buttons: **Get started** → `getting-started/installation` (`button-ref`),
     **Browse generators** → `generators/overview` (`button-ref`),
     **GitHub** → https://github.com/AntoineRichard/TrackGen (`button-link`).
   - Hero figure: `.. figure:: assets/readme-pipeline-stages.png` (the Phase-1 → constant
     spacing → XPBD relax → inflated-road pipeline) with a one-line caption.

2. **Features** — a `.. grid::` of six `.. card::`s (title + one line each), all accurate to
   the code:
   - **GPU-batched** — generate `E` tracks in parallel per `generate()` call; CPU for
     tests/CI, CUDA for production.
   - **Pure Warp pipeline** — generation → constant-spacing resample → XPBD relaxation →
     inflation, all as Warp kernels.
   - **Five generators** — Bezier, Hull, Polar, Voronoi, Checkpoint — pluggable, each a
     distinct shape family.
   - **CUDA-graph capture** — the whole pipeline captured once and replayed for high
     throughput.
   - **Gate sequences** — drone-style gate courses (centres + orientations) via
     `GateGenerator`.
   - **RL-ready output** — constant-width outer/center/inner borders, per-point frames
     (tangent/normal), validity, and counts.

3. **Figure showcase** — two more figures with captions:
   - `assets/readme-generator-grid.png` — "Five generators, one batch."
   - `assets/readme-gate-strip.png` — "Gate sequences with phase-2 collision relaxation."

4. **Explore the docs** — a `.. grid::` of six navigation `.. card::`s, each linking (via
   `:link:` / `:link-type: doc`) to a section landing page: Getting started
   (`getting-started/installation`), Tutorials (`tutorials/batch-of-tracks`), Generators
   (`generators/overview`), How it works (`how-it-works/pipeline`), Configuration & tuning
   (`configuration/tuning`), API reference (`reference/api`).

5. **Hidden toctrees** — the nine existing toctree blocks gain `:hidden:`; their captions and
   entries are otherwise unchanged so the Furo sidebar nav is identical to today.

## Verification

- Strict build clean: `.venv/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html`
  → "build succeeded", zero warnings (every `button-ref`/`:link:` doc target must resolve; a
  bad target warns under `-W`).
- Linkcheck: `.venv/bin/python -m sphinx -b linkcheck docs docs/_build/linkcheck` — the new
  GitHub link resolves (internal links clean).
- Eyeball the rendered `docs/_build/html/index.html` (hero, six feature cards, two showcase
  figures, six nav cards) before finishing.

## Files touched

- `docs/index.rst` — the only change.
