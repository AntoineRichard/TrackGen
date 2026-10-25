# Racing-first visual documentation design

Status: approved by the user on 2026-10-25; subsequent user direction includes motion where it explains behavior. The RC handoff finishes first; this work does not authorize the final 0.3.0 release.

## Intent and scope

Help racing users see what TrackGen produces, choose a generator, build a usable road or gate course, and understand the controls and failure cases. A reader should get a useful result quickly and follow links into exact contracts and deeper explanations when needed.

The user confirmed that historical content should leave the **published reader documentation**. Internal design, roadmap and handoff records remain outside the published site. The repository changelog and Git history remain the record of releases.

This is a content, navigation and visual-guide overhaul. Runtime generation, collision algorithms and public APIs retain their RC behavior. Fixes to documentation rendering tools are in scope where needed to produce current examples.

## Recommended approach and alternatives

Recommendation: keep Sphinx/Furo and its executable-example, public-API and reference checks. Rewrite the learning path and render figures from current TrackGen data with NumPy/Matplotlib. Publish SVG or high-resolution PNG assets; rendering is an explicit maintenance step, not a docs-build dependency.

A new documentation framework would spend effort replacing working publishing/testing machinery. An interactive-first site would add browser and simulation infrastructure before the basic explanations are clear. Start with reproducible visual comparisons; use small interactive elements only when they materially improve understanding and fit the current stack. This recommendation is the implementer's proposal, not a previously approved user choice.

## Reader navigation

1. **Start racing:** final-output gallery, track/gates quickstarts, concepts and data format.
2. **Choose a generator:** six procedural generator cards, imported geometry, native gates and gates sampled along tracks.
3. **Shape your course:** visual controls for shape, size, width, sampling/capacity, relaxation, elevation and gates; reproducible seeds and curricula.
4. **Use it in a simulator:** collision, progress/checkpoints, localization, resets/regeneration, CUDA graphs and data ownership.
5. **Build and extend:** import a centerline, assemble a gate course, write a simple generator, write a Stage, test and package a plugin, add a runtime component.
6. **API reference:** complete current public contracts and settings, grouped by task/module, linked from the guides.
7. **Deeper explanations:** geometry, validity, solver behavior and collision accuracy. Reach these from the corresponding visual guides rather than making them the first reading path.

The user confirmed that centerlines remain important for understanding the generators. Generator pages keep prominent centerline examples and raw-to-final overlays; the overview gallery leads with finished outputs and links directly to that generator detail.

The homepage opens with final roads and gate courses, a short statement of what users get, and two actions: create a track or create gates. Upgrade notices and development history leave the primary site. Preserve existing useful page URLs/anchors where practical; obsolete historical routes can point readers to the current guide without retaining historical content.

## Page pattern: short first, depth on demand

Every generator/control guide follows the same order:

- **What you get:** one sentence, representative final outputs, racing characteristics and limits.
- **Try it:** one complete minimal runnable example with the current public API.
- **Change it:** compact setting summaries and labeled visual comparisons.
- **When it fails:** actual failure cases, their diagnostics, and actionable settings.
- **Go deeper:** algorithm, equations, detailed behavior and API links.

Full reference coverage remains available. Long tutorials and extension pages become focused recipes plus linked contract/reference pages; shortening the first reading path must not erase important ownership, masking, padding, stream or capture rules.

## Visual coverage

| Area | Required figures and explanations |
|---|---|
| Generator overview | Final inflated roads from Bezier, Hull, Polar, Voronoi, Checkpoint and Repulsive; several examples to show diversity; imported centerline to road; native gates and stations along a track |
| Individual generators | Same-seed comparisons of meaningful shape controls; raw/final overlays where relaxation or fallback changes the result; limits and failure examples |
| Shared geometry controls | Size versus width; sample spacing with point markers; adequate versus truncated capacity; relaxation iteration/margin/smoothing effects |
| Elevation | Plan view plus altitude against arclength for flat, uniform, random-walk and noise profiles; bounds, step/amplitude and control-point effects |
| Gates | Ordering, radius, solver iterations, width, frames/openings; native-anchor versus track-station construction; 3D orientation and collision geometry |
| Self-intersection/validity | Centerline crossings, nonlocal close approaches, tight curvature, overlapping road bands and a valid case; show the offending geometry and corresponding checks/failure bits |
| Runtime collision | Vehicle footprints, touching/crossing/clearance cases, exact segments versus SDF, grid-resolution effects and narrow features; separate gate-frame and disc queries |
| Public data | Annotated flat env rows showing layout/stride, real count, padding, validity, failure masks and generation; borrowed result versus retained snapshot; simulator lifecycle diagram |

Each public setting receives its default, units/range, effect, tradeoff, static/dynamic status and a relevant example/reference. Shape, validity and accuracy controls receive visual comparisons. Execution/storage controls receive appropriate lifecycle, data-layout or performance explanations instead of invented geometry effects. Related settings can share a figure with individually labeled effects.

## Figure and technical accuracy rules

Use actual current library output, fixed documented seeds, consistent axes/scales and one-setting-at-a-time comparisons. Use small grids only for important interactions. Label units, parameters, validity and any fallback/filtered sample selection; show failed geometry deliberately with explicit `on_invalid="keep"`. Do not silently turn a selected attractive sample into a yield claim.

Use short videos or animations where the process matters, especially track and gate XPBD convergence. The user also requested generator construction as an animation candidate: use short labeled construction sequences where they explain anchors, handles, site selection, steering or growth more clearly than a still. Distinguish algorithm steps from later relaxation and indicate illustrative schematics versus measured solver states. Show iteration counts and relevant validity/clearance information; keep the viewpoint fixed. Include a useful still poster, caption and text alternative so the main result remains accessible and scannable. Motion should explain progression rather than decorate a page. The user suggested 3Blue1Brown’s Manim as an optional animation tool: consider it for geometric/constraint explanations, while solver convergence clips use actual TrackGen states. Keep Manim optional and outside library/docs-build dependencies.

Render in scratch output, inspect every new figure/video and then copy selected assets. Record the reproduction command, source version, parameters and seeds. Replace or remove old measurements and images when they no longer describe current behavior; measure any retained numerical claims on current source with the machine/method stated.

Correctly distinguish generation validity from runtime agent collision. Thickness uses sampled bead distances and local curvature; it is not a universal proof of segment separation. Proper segment crossings, endpoint touches and collinear overlaps are different cases. Explain planar road collision for elevated tracks, gate-bar collision versus route crossings, SDF interpolation/normal accuracy, and why signed clearance is not always physical penetration depth. Verify every claim against the implementation and small reproductions before publishing it.

## Maintenance and acceptance

Reuse the existing CPU/CUDA snippet execution and strict Sphinx/API checks. Keep the docs-only environment free of rendering/dev dependencies. Add focused checks for the new rendering/setting manifest where they catch real missing coverage or vacuous comparisons; avoid brittle image-byte tests.

Acceptance: a new racing user can select and generate a course from the first pages; all current generators and supported import/extension paths have examples; meaningful controls have understandable visual effects; collision/self-intersection limits are accurate; the published site contains no release/development narrative or stale measurement claims; every new visual and example has been inspected/executed; strict docs and relevant tests pass.

## Inspiration

- scikit-learn parameter plots: labeled effect grids, then quantitative/deeper explanation and runnable source: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
- MuJoCo collision documentation: define the collision model and inputs/outputs before algorithms and thresholds: https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection
- Shapely manual: minimal geometric counterexamples distinguish simplicity, validity and crossing behavior: https://shapely.readthedocs.io/en/stable/manual.html

Adapt the explanatory patterns; create original TrackGen figures and prose.
