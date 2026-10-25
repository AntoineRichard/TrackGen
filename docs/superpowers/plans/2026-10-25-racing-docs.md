# Racing-first visual documentation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Give racing users a short route to usable courses, then accurate visual control, geometry, simulator and extension guides.

**Architecture:** Retain Sphinx/Furo, current public APIs, executable snippets and strict reference checks. Render original figures and short motion clips explicitly from current library output; commit selected assets and their reproducible source, with no new Sphinx-build dependency. Preserve internal project records outside the site.

**Tech stack:** reStructuredText, Sphinx, Python, NumPy, Matplotlib, Warp; ffmpeg for solver clips, optionally Manim for explanatory geometry.

**Spec:** `docs/superpowers/specs/2026-10-25-racing-docs-design.md` (approved 2026-10-25, with subsequent direction to retain generator centerlines and use useful animations).

## Global constraints

- Finish the RC handoff first. Final release still requires the user's racing-repository confirmation and explicit go.
- Work on `codex/racing-docs` from the immutable RC commit; leave `v3` available for release fixes.
- Historical content leaves published reader documentation only. Retain design/handoff/roadmap and archived release/migration material outside the site.
- No runtime generation, public API or collision algorithm changes. Documentation rendering tools may be fixed.
- Keep `import track_gen` light and the lean docs environment free of rendering dependencies.
- Keep centerlines prominent in generator explanations; lead overview pages with final roads and gate courses.
- Use actual fixed-seed output, equal axes, explicit units/selection/failure labels; never imply selected examples establish yield.
- Motion must explain a process. Solver convergence clips use actual intermediate TrackGen states, fixed viewpoint, iteration labels, useful poster and text alternative.
- Every setting has current defaults, units/ranges, effect/tradeoff and static/dynamic behavior; geometric settings have meaningful visual examples, storage/execution settings have appropriate lifecycle diagrams.
- Commit as Antoine Richard <antoiner@nvidia.com>; only push after required validation.

## Review focus

1. Poisoned invalid output: every plotted example respects count/mask; deliberate failed geometry uses explicit keep, and the UI never implies validity.
2. Ambiguous collision geometry: distinguish sampled validity, proper segment crossings, endpoint/collinear cases, planar road queries, interpolated SDF and 3D gate bars.
3. Static/dynamic lookalikes: track and gate generator Params differ; changing size is not always changing generator scale.
4. Missing reader contracts: reshaped arrays, padding, borrowed storage, masking, streams and capture rules survive page splitting.
5. Historical material leaking through search/copies: exclusions cover sources/assets/tests, links and labels resolve, examples exercise only current APIs.

## Files and ownership

- Generator unit: `viz/render_generator_guide.py`, `docs/assets/racing/generators-*`, `docs/generators/*.rst`, `docs/tutorials/choosing-a-generator.rst`.
- Course-control unit: `viz/render_course_guide.py`, `docs/assets/racing/controls-*`, `docs/tracks-25d.rst`, `docs/gates-3d.rst`, `docs/relaxation/gates.rst`, `docs/tutorials/gate-sequences.rst`, `docs/configuration/tuning.rst`.
- Geometry/collision unit: `viz/render_collision_guide.py`, `viz/render_relaxation_video.py`, `docs/assets/racing/geometry-*`, related convergence videos/posters in `docs/_static/`, `docs/utilities/collision.rst`, `docs/how-it-works/inflation.rst`, `docs/relaxation/{overview,constraints,solver,convergence}.rst`.
- Reader/extension unit (root): `docs/index.rst`, getting-started pages, `docs/user-guide/*`, focused extension recipes under `docs/contributing/`, `docs/reference/*`, `docs/configuration/reference.rst`, imports/data/lifecycle figures and external-centerline recipe.
- Integration (root): `docs/conf.py`, historical source locations, label/snippet/link checks, remaining public pages/README links, rendering instructions and provenance.

Existing tests enforce generator Params section headings and example block markers. Keep those contracts or deliberately update their extraction when examples move; do not weaken assertions to make prose pass.

### Task 1: Catalog and six generator guides

**Consumes:** current generator Params, generate/seed contracts, existing renderer helpers. **Produces:** `python -m viz.render_generator_guide --out-dir DIR`, deterministic assets and a JSON manifest of seed/config/sample selection; six consistent generator pages and overview/chooser.

- [ ] Read generator implementations and `/private/tmp/trackgen-docs-settings-audit.md`; map every active shape control to a figure or precise reference.
- [ ] Add focused renderer checks for six families, finite real-count geometry, non-vacuous comparisons and manifest provenance; run once before/after implementation. Avoid pixel-byte assertions.
- [ ] Render finished roads plus prominent centerlines, a small diversity strip per family, same-seed knob comparisons, and fallback/failure explanations. Add short generator-construction animations where the process is informative (anchors/handles, hull or site selection, steering/growth); use current geometry and clearly label schematics versus captured iterative states. Keep repulsive batches small.
- [ ] Rewrite the pages result/example/controls/failure/depth; remove RNG history and old cost/yield claims. Preserve accurate algorithms, clearly distinguishing native gate anchor behavior.
- [ ] Inspect all outputs; run generator docs tests and CPU snippets for touched pages. Commit the coherent unit after integration validation.

### Task 2: Course controls, elevation and gates

**Consumes:** current TrackGenConfig/GateGenConfig, track/gate params and current gate relaxation solver. **Produces:** `python -m viz.render_course_guide --out-dir DIR`, manifest, labeled setting guides, gate-convergence clip/poster if the existing clip needs regeneration.

- [ ] Check size/scale/width/spacing/capacity and the static/dynamic routing, including centerline stations and native anchors.
- [ ] Render comparable controls, sample beads and truncation, altitude versus arclength for all profiles, and native/track-sampled/3D gate geometry. Choose representative controls without inventing effects for storage knobs.
- [ ] Add focused coverage/non-vacuity checks for the generated data and use real count/validity. Explain unsupported gate combinations and poison/keep deliberately.
- [ ] Rewrite the assigned course pages with minimal runnable examples, short controls first, frame/ordering/solve/radius details linked below. Keep tested entry points and valid labels.
- [ ] Inspect assets and any gate animation at beginning/middle/end; run snippets and relevant gate/config docs checks.

### Task 3: Validity, collision and XPBD

**Consumes:** actual ops/self-intersection/thickness/exact/SDF/disc/gate-query behavior and existing solver renderer. **Produces:** `python -m viz.render_collision_guide --out-dir DIR`, geometry figures/manifest and current track convergence video/poster.

- [ ] Reproduce a valid band, raw crossing, close nonlocal beads, tight bend, overlapping bands, endpoint touch/collinearity and truncation; tie pictures to actual diagnostics without claiming an exact proof.
- [ ] Render vehicle footprint/edge-only crossing/touching/clearance and SDF grid/narrow-gap effects using public queries, plus disc and gate-frame diagrams. Check assertions against actual returned values.
- [ ] Repair video tooling for poison defaults and current RNG; avoid torch where NumPy transfer suffices, use temporary output and expose configurable paths. Verify solver-prefix reasoning against source; do not fake interpolation between iterations.
- [ ] Re-render meaningful convergence samples, label chosen seeds/selection and smoothing phase, retain a useful failure case. Explain sampled constraints/limits, planar elevated roads and distance semantics accurately.
- [ ] Rewrite assigned explanation pages to lead with pictures/clips and interpretation, followed by contracts/equations; remove old solver-history and stale benchmark conclusions.
- [ ] Inspect every figure and video samples; run relevant collision, validity, renderer and snippet checks.

### Task 4: Reader path, public data and API

**Consumes:** assets/labels from Tasks 1–3; existing public API and tests. **Produces:** task-based homepage/nav, two short quickstarts, focused result contracts and annotated lifecycle/array-layout figures.

- [ ] Lead the homepage with output gallery and create-track/create-gates actions; group navigation as Start racing / Generators / Shape courses / Simulator / Extend / API / Explanations.
- [ ] Rewrite concepts and quickstarts for current racing use, remove upgrade-first narrative and add clear links into the six generator centerline guides.
- [ ] Split the long result narrative into a short usage page and precise linked storage/ownership/masking references; show env rows, stride/count/padding/valid/fail/generation and borrowed versus cloned snapshots.
- [ ] Keep full public autodoc coverage and setting reference; fix stale `/1` support and misleading aliases, validate defaults against dataclasses/Params.
- [ ] Execute examples, run metadata/API/reference/link tests and strict Sphinx. Inspect the built homepage, one generator page, one collision page and mobile-width navigation if a browser renderer is available.

### Task 5: Build and extend recipes

**Consumes:** current external generator ingestion, GeneratorSpec (`ctx=`), plugin.Stage and Course APIs. **Produces:** short entry recipes for importing a track, assembling gates, simple generator, stage and component, with detailed contracts linked separately.

- [ ] Read and preserve the executable Wobbly/Star/Wobble/Scatter examples and their conformance assertions.
- [ ] Put one complete useful example first on each extension path. Split long contract discussions into focused linked sections/pages while preserving tested example markers or updating extraction intentionally.
- [ ] Add an imported-centerline visual and track-to-gate/station recipe; explain generator versus Stage choice, capacity/count/seed/ctx contracts, testing and entry-point packaging.
- [ ] Keep deep lifecycle/stream/capture/ownership details discoverable. Execute every new snippet and existing plugin guide tests, including valid-output checks.

### Task 6: Remove public history, integrate and verify

**Consumes:** all units. **Produces:** history-free current site, retained internal archive, reproducible asset maintenance notes, validated reviewable docs branch.

- [ ] Move old migration/background/future-design pages and solely historical tables/measurements into excluded `docs/library-readiness/archive/`; remove their public nav and references. Preserve tagged GitHub migration links in the release records.
- [ ] Audit every published page for release comparisons, roadmap labels, obsolete examples/measurements and contradictions. Update remaining simulator/how-it-works/reference prose to current behavior. Preserve legitimate algorithm citations and current `legacy.to_pipeline_spec()` public naming.
- [ ] Update labels/owner contracts, snippet collectors and excluded-directory checks only as required by intentional moves; strengthen detection of history leaking into the built site. Verify old labels still used by current pages resolve or replace their links.
- [ ] Document rendering commands, optional tool setup, source version/config/seeds, asset sizes and posters. Rebuild in a lean environment and check actual HTML/assets/search exclude internal records.
- [ ] Run `PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WARP_CACHE_PATH=/private/tmp/trackgen-warp-cache /private/tmp/trackgen-pr16-venv/bin/python -m pytest -q -p no:cacheprovider -m 'not cuda and not slow and not benchmark'`.
- [ ] Run `/private/tmp/trackgen-pr16-venv/bin/python docs/_tools/build_ci_env.py -W --keep-going -E -b html -d /private/tmp/trackgen-racing-docs-dt docs /private/tmp/trackgen-racing-docs-html` and linkcheck; inspect raw linkcheck output.
- [ ] Export a candidate commit, run changed/new CUDA doc examples on spark-home after the RC GPU jobs finish; run broader GPU lane only if a failure or changed capture path warrants it. Executable runtime code must match the RC; correcting public source docstrings is permitted and must leave behavior unchanged.
- [ ] Request one independent whole-branch technical/content review, resolve findings and rerun affected checks. Commit with required identity; push validated branch and provide review/preview paths. Do not merge or release 0.3.0 without the user's explicit go.

## Self-review

The six tasks cover all approved requirements, including current public setting coverage, centerlines, motion, extension paths, data/API and historical exclusions. Independent visual/content units may be delegated concurrently after the RC handoff, with disjoint files; root integrates navigation/reference/extension work. Existing meaningful checks remain; renderer tests target non-vacuous data/coverage rather than pixels. Final branch review covers sampled-geometry and collision wording that tests cannot prove. User approval to proceed already covers routine execution choices; no further design approval is needed unless scope/runtime behavior changes.
