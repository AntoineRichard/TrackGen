# Generator movie layout and branding implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent source and reader tasks, then integrate captures and review.

**Goal:** Generator videos explain their generator and all nine clips carry the graphical Circuit identity.
**Architecture:** Modify native Matplotlib scenes; share a figure-level Circuit image; recapture both themes from committed source and selectively promote media.
**Tech stack:** Matplotlib, NumPy, Pillow, FFmpeg, Sphinx.
**Spec:** docs/superpowers/specs/2026-10-27-generator-movies-design.md

## Global constraints

No runtime, RNG, defaults, goldens, package, navigation, release/tag or deployment changes. Preserve real captures, selections, calm illustrations, native16:9 and unchanged media provenance.

## Review focus

Logo/footer overlap across movie phases; real raw geometry after pane removal; immutable detailed static figures; theme geometry equality; selective provenance without relabeling old captures.

## Tasks

- [x] Finish the already-started graphics-only export cleanup and record its validation.
- [x] Add focused failing tests for reusable branding bounds/theme/data digest; implement add_video_branding(fig, theme=None) in viz/racing_theme.py. Integrate it in sampling and XPBD figures, retaining their result views and reserving the footer.
- [x] Delegate generator scene changes and tests in storyboard_bezier_hull.py, storyboard_polar.py and render_generator_guide.py. Remove finished-road axes, keep raw references, fix visible captions and reserve the same footer corner.
- [x] Delegate matching reader caption/maintainer guide edits, preserving useful downstream explanations.
- [x] Review source, run focused media tests; commit source before final capture.
- [x] Render nine clips/posters in both themes, inspect decoded phases and compare held geometry. Promote intended media only and merge accurate source/hash registry entries.
- [x] Run required CPU and strict docs/snippets; review independently and self-review; update preview, commit/push validated work, report remaining Racing approval.
