# Circuit branding implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the user-approved Circuit logo in documentation and README.

**Architecture:** One optional deterministic pixel renderer produces static SVG and favicon assets. Furo's native theme selection displays the matching wordmark; no runtime or new browser logic is involved.

**Tech Stack:** Python standard library/Pillow for optional asset exports, SVG, Sphinx/Furo, existing browser tooling.

**Spec:** ../specs/2026-10-27-circuit-branding-design.md (user-selected visual concept).

## Global Constraints

- Preserve the approved stepped loop and 5×7 wordmark; original art has no font dependency.
- Transparent light/dark wordmarks have identical geometry; monochrome derives from the same source.
- Retain every prior media byte and provenance entry; add the branding pair only.
- Leave navigation, public runtime, dependencies, package versions, goldens and immutable tags unchanged.
- Follow the updated user preference: graphical identity only; no ASCII banner export or download.
- Keep optional rendering out of the docs build. Preserve user files and review worktree.

## Review Focus

- A small icon must retain a clear loop at 16/24/32 pixels.
- Native automatic/manual theme selection must show exactly one matching logo.
- The brand home link needs a meaningful accessible name; avoid duplicate visible text.
- README uses real light/dark assets and preserves its textual heading/links.
- Reproduction must use only task-owned output files and keep all existing 82 media pairs intact.

### Task 1: Circuit assets and integration

**Files:** Create viz/render_brand_assets.py; static brand assets; docs/_templates/sidebar/brand.html and branding CSS. Modify docs/conf.py, README.md and the media registry only for the new pair; document optional regeneration in existing maintainer guide.

**Interfaces:** Consumes the approved pixel grid/letterforms. Produces static brand/ and media-dark/ assets, matching paths in Furo and README, and a source/output hash manifest.

- [x] Create the deterministic asset renderer and verify scratch output, SVG/XML validity, favicon sizes; use existing source/runtime rather than new dependencies. Asset-only work needs visual/integrity checks, not mirrored implementation tests.
- [x] Commit the renderer/design source before generating final assets; record that revision in the asset manifest.
- [x] Add native Furo/README integration and the new registry row, preserving all old rows.
- [x] Run strict docs and the required CPU lane; inspect desktop/mobile light/dark, accessible home name, images, favicon and unchanged video playback; self-review and independent scoped review.
- [ ] Commit/push validated assets/integration, update the hosted preview, and report the result. Keep PR4 draft for Racing and final-release approval.
