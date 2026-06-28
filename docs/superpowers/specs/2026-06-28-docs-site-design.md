# track_gen documentation site — design

**Date:** 2026-06-28
**Status:** Approved (brainstorming), pending spec review

## Problem

`track_gen` has substantial, high-quality documentation scattered across the README and
several `docs/*.md` files, but no navigable, published documentation site. We want a
high-quality docs site with: a full public-API reference, tuning guides, tutorials/examples,
and in-depth per-generator deep-dives that highlight how the five phase-1 generators differ
and how their internals work. A GitHub Pages site will be enabled later.

## Goal

A Sphinx documentation site, authored in reStructuredText, that builds cleanly today and is
ready to publish to GitHub Pages with a one-time repo setting. It is the single source of
truth for `track_gen` documentation: existing prose docs migrate into it, and the public-API
reference is generated from enriched docstrings.

## Decisions (locked in brainstorming)

- **Engine:** Sphinx. **Authoring format:** pure reStructuredText (existing Markdown is
  converted to rST; the originals are deleted — single source of truth).
- **API reference:** `autodoc` + `napoleon`, driven by **enriched docstrings** on all public
  symbols (structured `Attributes:`/`Parameters:`/`Returns:`/`Raises:`). No `myst-parser`.
- **Theme:** Furo.
- **Extensions:** `sphinx.ext.autodoc`, `sphinx.ext.napoleon`, `sphinx.ext.autosummary`,
  `sphinx.ext.intersphinx`, `sphinx.ext.viewcode`, `sphinx.ext.mathjax`, `sphinx_copybutton`,
  `sphinx_design`.
- **Scope:** core pages + all four optional sections (per-generator figures + math,
  benchmarks page, related-work pages, future/experimental appendix).
- **Deploy:** staged — the site builds locally now; a `.github/workflows/docs.yml` builds on
  PRs and is ready to publish, but enabling GitHub Pages is a deferred manual repo setting.

## Non-goals

- Enabling GitHub Pages (done later by the maintainer).
- Versioned/multi-version docs, i18n, a custom domain.
- Rewriting the survey prose beyond Markdown→rST conversion and light editorial cleanup.
- Changing any runtime behavior of `track_gen` (docstring text only; no signature changes).

## Toolchain & layout

- Sphinx project rooted at `docs/`: `docs/conf.py`, `docs/index.rst`, content in subfolders
  (see IA), `docs/_static/` for theme assets, `docs/_build/` git-ignored.
- `exclude_patterns` covers `superpowers/**`, `_build/**` — the internal process docs never
  enter the build.
- Existing rendered figures stay in **`docs/assets/`** (so README image links keep working);
  rST pages reference them with relative paths (Sphinx copies referenced images into
  `_images/` at build).
- `track_gen` is installed editable, so autodoc imports it directly — no `sys.path` hacks.
  Autodoc imports run on the Warp **CPU** device (no GPU, no `torch`); the public surface
  imports `warp`, not `torch`, so no `autodoc_mock_imports` are needed.
- New `docs` extra in `pyproject.toml`:
  `docs = ["sphinx", "furo", "sphinx-copybutton", "sphinx-design"]`
  (`autodoc`/`napoleon`/`autosummary`/`intersphinx`/`viewcode`/`mathjax` ship with Sphinx).

## Information architecture (toctree)

1. **Overview** (`index.rst`) — what track_gen is, the pipeline one-liner, hero figure
   (`readme-pipeline-stages.png`), quick links. Source: README intro.
2. **Getting started**
   - Installation — Source: README "Install".
   - Quickstart — minimal `TrackGenerator` + `GateGenerator` examples. Source: README.
   - Interactive parameter explorer — the Gradio app. Source: README "Parameter explorer".
3. **Tutorials**
   - Generating a batch of tracks end-to-end (reading `Track`, `wp.to_torch`, `count`/`valid`).
   - Gate sequences for drone courses.
   - Choosing & configuring a generator.
   - Using the CUDA graph in a batched sim (the runtime facade contract).
4. **Generators**
   - Overview — the five, the contract summary, the grid figure
     (`readme-generator-grid.png`), "when to use which" table.
   - **Bezier**, **Hull**, **Polar**, **Voronoi**, **Checkpoint** — one deep-dive page each:
     per-generator figure, algorithm steps, math (LaTeX), parameters, what makes it distinct,
     fallback behavior. Source: `ARCHITECTURE.md` lines 100-265 + the
     `warp_generate_*.py` sources.
   - Benchmarks / comparison — Source: `generator-baseline.md` + reproduce command.
5. **How it works (architecture)**
   - Pipeline overview · Constant-spacing resample · XPBD relaxation (+ separation cache) ·
     Inflation & the `Track` result · CUDA-graph capture & runtime facade · Kernel
     conventions. Source: `ARCHITECTURE.md`.
6. **Configuration & tuning**
   - `TrackGenConfig` & `GateGenConfig` references (autodoc of enriched dataclasses).
   - Tuning guide: yield vs diversity, `half_width`/`spacing` coupling, relaxation knobs,
     separation-cache throughput, per-generator tuning tips. Source: README "Advanced XPBD
     separation cache" + config docstrings + ARCHITECTURE.
7. **API reference** (autodoc) — `TrackGenerator`, `GateGenerator`, `PerEnvSeededRNG`,
   `Track`, `GateSequence`, `TrackGenConfig`, `GateGenConfig`.
8. **Contributing**
   - Writing a generator plug-in. Source: `generator-contract.md`.
   - Dev setup & tests — includes the `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` gotcha (ROS pytest
     plugins leak from `/opt/ros/humble` and break collection otherwise).
   - Rendering documentation assets (`viz.render_readme_assets`).
9. **Related work**
   - Prior art — Source: `racetrack-generation-prior-art.md`.
   - State of the art — Source: `track-generation-state-of-the-art.md`.
10. **Appendix: future / experimental generators** — Source:
    `pre-relaxation-generator-methods.md` (deferred ideas as a research backlog).

## Migration map (delete source after conversion)

| Existing file | Becomes | Notes |
|---|---|---|
| `docs/ARCHITECTURE.md` | "How it works" pages + generator deep-dive internals | Split by section into the IA. |
| `docs/generator-contract.md` | Contributing → "Writing a generator plug-in" | Near-verbatim. |
| `docs/generator-baseline.md` | Generators → "Benchmarks" | Table + context prose. |
| `docs/racetrack-generation-prior-art.md` | Related work → "Prior art" | Convert; keep prose. |
| `docs/track-generation-state-of-the-art.md` | Related work → "State of the art" | Convert; keep prose. |
| `docs/pre-relaxation-generator-methods.md` | Appendix → "Future / experimental" | Convert; mark deferred items. |

README reconciliation: the README currently links to `docs/ARCHITECTURE.md`,
`docs/generator-contract.md`, `docs/generator-baseline.md`. After deletion these links break,
so update them to the new rST source paths (and leave a placeholder for the published site
URL, added when Pages is enabled). The README stays as the repo landing page; it is not
gutted. `docs/assets/*.png` are NOT moved, so README image links are unaffected.

## Docstring enrichment

Enrich docstrings on the 7 public symbols so autodoc renders a rich API + configuration
reference (this is the single source of truth for parameters):

- `TrackGenConfig`, `GateGenConfig`: NumPy/Google-style `Attributes:` sections covering the
  tunable knobs, grouped by concern (generator selection, per-generator families, width,
  relaxation core, separation cache, output, validity), moving the existing inline-comment
  guidance into the structured docstring. Vestigial/oracle-only fields are documented as such
  or grouped separately.
- `Track`, `GateSequence`: `Attributes:` for the result arrays (shapes, NaN-padding,
  aliasing/`clone()` note).
- `TrackGenerator`, `GateGenerator`, `PerEnvSeededRNG`: class docstrings + `Parameters:`/
  `Returns:`/`Raises:` on `__init__`, `generate()`, and sampling methods.

Constraint: docstring **text only** — no signature, default, or behavior changes. Existing
warning text (aliasing, CUDA graph, fixed-batch) is preserved.

## New rendering work

Extend `viz/render_readme_assets.py` with a deterministic, fixed-seed **per-generator figure**
— a small-multiples panel of 3–5 representative outputs per generator — committed under
`docs/assets/`, reusing the existing self-contained, public-API-only, CPU style. Generator
deep-dives also carry the math (Catmull-Rom, Bézier handle construction, the checkpoint
steering walk) in MathJax.

## Build & verification (quality gate)

- Clean HTML build with warnings-as-errors:
  `python -m sphinx -W --keep-going -b html docs docs/_build/html`. Fix every warning
  (use a scoped `nitpick_ignore` only for genuinely external/un-resolvable references).
- External-link check: `python -m sphinx -b linkcheck docs docs/_build/linkcheck`
  (allowed to surface known-flaky external URLs; internal links must pass).
- Autodoc import verified on CPU (no GPU required).
- A README/CI note documents the `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` requirement for the
  separate test suite; the docs build itself does not run pytest.

## Execution model (for the plan)

- **Phase 1 — Foundation (sequential):** `docs` extra, `docs/conf.py` (theme, extensions,
  intersphinx, autodoc config), `docs/index.rst` + empty toctree skeleton, `.gitignore` for
  `docs/_build/`, `exclude_patterns`. Deliverable: an empty site builds clean with `-W`.
- **Phase 2 — Docstring enrichment + autodoc pages (sequential-ish):** enrich the 7 symbols;
  add the API reference + configuration autodoc pages. Deliverable: API/config pages build.
- **Phase 3 — Independent content pages (parallel fan-out):** each is a self-contained page
  from existing sources with no shared state, suitable for parallel subagents:
  prior-art, state-of-the-art, future/experimental appendix, the five generator deep-dives,
  the architecture/how-it-works pages, the benchmarks page, the tutorials, getting-started.
  The per-generator figure rendering is one shared task that lands before the deep-dive pages
  reference its outputs.
- **Phase 4 — Integration (sequential):** wire the full toctree/nav, add cross-references,
  update README links, add `.github/workflows/docs.yml`, and run the `-W` clean build +
  linkcheck as the final gate.

## Risks

- **`-W` strictness vs autodoc:** autodoc/napoleon can emit warnings (e.g., unresolved
  cross-refs, duplicate object descriptions). Mitigation: scoped `nitpick_ignore`, careful
  `autosummary` config, and iterating the build to zero warnings.
- **Dataclass `Attributes` rendering:** large grouped `Attributes:` blocks must render
  legibly; if a single block is unwieldy, split guidance into the prose tuning page and keep
  `Attributes:` factual.
- **Autodoc importing Warp on CI:** the GitHub Actions runner must `pip install -e .[docs]`
  so `warp-lang` is present; the CPU device works without a GPU.

## Files touched (summary)

- New: `docs/conf.py`, `docs/index.rst`, the rST page tree, `docs/_static/`,
  `.github/workflows/docs.yml`, per-generator figure(s) in `docs/assets/`.
- Modified: `pyproject.toml` (`docs` extra), `.gitignore` (`docs/_build/`),
  `viz/render_readme_assets.py` (per-generator figure), `README.md` (link reconciliation),
  the 7 public symbols' docstrings in `track_gen/_src/`.
- Deleted (after migration): the six `docs/*.md` prose files listed in the migration map.
