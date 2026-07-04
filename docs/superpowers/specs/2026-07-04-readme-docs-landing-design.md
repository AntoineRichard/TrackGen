# README Rework & Utilities-as-First-Class Docs — Design

**Date:** 2026-07-04
**Status:** Approved
**Scope:** README.md, docs/utilities/* (new section), docs/index.rst,
docs/reference/api.rst, docs/tutorials/runtime-utilities.rst (retired),
GitHub repo About metadata.

## Goal

Make the repo landing attractive and lean — the README pitches, shows, and
links; the docs site carries the depth — and promote the runtime utilities
to first-class docs citizens with their own section, mirroring how the
generators are presented.

## Part A — README rework (~140 lines, from 392)

Structure, in order:

1. **Hero**: `# TrackGen` (display name; package stays `track_gen`),
   one-sentence pitch covering BOTH halves of the library (GPU-batched
   track/gate generation AND the runtime collision/progress utilities, pure
   NVIDIA Warp, CUDA-graph native). Badge row:
   - `📖 Documentation` → https://antoinerichard.github.io/TrackGen/
   - docs workflow status badge (`.github/workflows/docs.yml` on main)
   - `Python ≥ 3.10` static badge
   - `warp-lang ≥ 1.14` static badge
   Then the existing pipeline one-liner ASCII diagram and the
   `readme-generator-strip.png` figure with its caption.
2. **What's in the box** — a table, one row per capability, each ending in
   a deep link into the docs site:
   | capability | one-liner | docs link |
   Rows: five phase-1 generators; gate sequences; out-of-bounds collision
   (exact segments / baked SDF backends + disc obstacles); boundary prop
   instancing (cones/walls); checkpoints & progress (gates or tracks,
   delta-distance rewards); the Course facade (one object: generate → bind
   → step/reset). Below the table: `utilities-overview.png` figure.
3. **Install** (~12 lines): uv path (3 commands) and pip path (2 commands),
   one sentence on extras (`dev`, `ui`, `docs`); link to the installation
   page for details.
4. **Quickstart** — two snippets, verbatim-runnable:
   a. Batch generation with `TrackGenerator` (existing snippet, trimmed:
      config → rng → generate → shapes comment).
   b. The Course facade RL loop: config → bind → generate → step/reset
      (mirrors the tutorial's snippet, ~12 lines), with a closing line
      pointing at the runtime-utilities docs section.
5. **Documentation** — nav table linking: Getting started · Tutorials ·
   Generators · Runtime utilities · How it works · Configuration · API
   reference (each to its docs URL).
6. **Development** — three commands only (fast test lane, full suite, docs
   build) + a line pointing at the contributing pages.

Deleted from the README (docs own them; no content is lost from the
project): XPBD separation-cache section, architecture section, project
layout, gate-sequence deep dive, generator-choice discussion, output-mode
discussion, parameter-explorer section (one row in the nav table instead),
long install prose.

Constraints: relative image paths (render on GitHub); absolute
`https://antoinerichard.github.io/TrackGen/...` links for all docs links
(render anywhere, including PyPI later); keep the fixed-seed figure files
unchanged (no asset regeneration needed); tests/test_readme_assets.py
untouched.

## Part B — docs: utilities become a first-class section

New `docs/utilities/` section, toctree placed directly after Generators,
section caption "Runtime utilities":

- `utilities/overview.rst` — the family in one page: shared conventions
  (flat `[E*stride]` NaN-padded batches, preallocated in-place results +
  `clone()`, eager NaN-proof validation, input binding, undefined for
  invalid envs), the CUDA-capture story (`track_gen.set_capturing`), the
  after-regeneration checklist table (moves here from the tutorial), the
  `utilities-overview.png` figure, and links to the four tool pages.
- `utilities/collision.rst` — out-of-bounds checking: segments vs sdf
  backend narrative with the trade-off guidance, **the Performance
  subsection moves here from docs/reference/api.rst** (4090 table,
  accuracy numbers, break-even rule, reproduce command — api.rst keeps
  pure reference plus a "see performance" link), `DiscChecker` with the
  gate-post recipe and the `disc-collision.png` figure.
- `utilities/props.rst` — points/segments modes, snap-spacing semantics,
  truncation/derivation, rendering-only caveat (+ pointer to DiscChecker
  for making props physical); references the props panels of the overview
  figure.
- `utilities/progress.rst` — CheckpointSet contract; the two sources
  (from_gates zero-copy, CheckpointSampler virtual gates) with
  `checkpoints-overview.png`; ProgressTracker events, reset semantics
  (NaN sentinel, teleport safety), the delta-distance reward pattern with
  the reset caveat, `progress-tracking.png`.
- `utilities/course.rst` — the facade: lifecycle (construct → bind →
  generate → step/reset), whole-batch generate vs per-env reset rationale,
  the two-graph story (pipeline Graph A, refresh Graph B), determinism
  contract, capture how-to, gates-mode example, truncation surfacing.

Content sourcing: `docs/tutorials/runtime-utilities.rst` is RETIRED — its
sections redistribute into the five pages above (edited to fit, not
duplicated). The file is deleted and removed from the tutorials toctree.
Inbound references are repointed: `docs/reference/api.rst`'s
`:doc:` link → `/utilities/overview`; any other `runtime-utilities`
references (grep) likewise. The tutorials section keeps its other four
pages.

Landing page (`docs/index.rst`):
- Add a "Runtime utilities" nav card (next to Generators) linking
  `utilities/overview`.
- Update the feature-card grid: repurpose/extend so the utilities are
  visible (a "Collision & progress" card and a "Course facade" mention;
  keep the grid balanced at 6 cards).
- Add one utilities figure to the landing gallery.
- Add the new toctree block.

`docs/contributing/rendering-assets.rst`: update the figure-location
references if they name the tutorial page (grep; figures themselves are
unchanged).

## Part C — repo About metadata

- Keep the homepage (already set; GitHub always renders it as a bare URL —
  labeled links are not supported in About, which is why the README hero
  carries the labeled `📖 Documentation` badge).
- Set the repo description via `gh repo edit --description`:
  "GPU-batched race-track & gate generation with runtime collision,
  progress and instancing utilities — pure NVIDIA Warp, CUDA-graph native."

## Testing / acceptance

- Docs build clean under `-W` semantics (CI builds with `-W
  --keep-going`): no broken refs after the tutorial retirement, all five
  new pages in the toctree, no orphan warnings.
- Full test suite green (docs-only change; the readme-assets tests must
  still pass untouched).
- README: all image paths resolve in the repo (relative), all docs links
  absolute and correct (spot-check against the deployed site's URL
  scheme: /tutorials/…, /generators/…, /utilities/…, /reference/api.html).
- Line count target ~140 (soft; attractiveness beats the number).

## Out of scope

- New figures or asset regeneration.
- PyPI packaging/metadata beyond the README itself.
- Restructuring the generators or how-it-works sections.
- Any runtime code changes.
