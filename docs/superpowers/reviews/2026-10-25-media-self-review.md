# Racing media review

Scope: the user's Bezier/Hull/Polar construction feedback, shared sampling-tail
explanation, and real light/dark variants of all published media. Library behavior
and the separately deferred PR4 spacing decision are unchanged.

Two agents independently cross-reviewed work they did not author, against actual
generator code, recorded arrays, Sphinx output and browser behavior.

- Generator explanations: grid-address selection and actual offsets, all-point
  angular ordering, Bezier handles, Hull midpoint displacement, sampled proper
  crossing detection and whole-loop polygon fallback, Polar slot/radius/angular
  jitter, and fixed-count/raw versus target-spacing/final redistribution. No
  algorithmic explanation or replay parity defect remained after review.
- Theme audit: found 3D collections absent from the plotted-coordinate digest
  before canvas drawing. Added actual 3D arrow segments and scatter coordinates;
  both public-API regressions failed first and then passed. All 26 adjacent theme,
  renderer and Sphinx tests passed. Real paired Course frames retained identical
  coordinate hashes. Browser checks passed for explicit/system selection, URLs,
  one player, playing/paused position and rate, and rapid theme toggles.
- Selective captures: found stale per-file provenance/digests and an unregistered
  Polar video poster. Refreshes now register only files actually rewritten, with
  current source/theme/command and coordinate hashes, preserving untouched
  metadata. The three maintenance modes failed the regression first, then passed.

The integrated source suite passed 70 tests before these final two audit repairs;
subsequent focused checks cover those repairs. Final assets must be captured from
committed sources. Full-site strict build, full-media browser inspection, final
combined CPU/CI and GPU validation are separate release gates; this review does
not claim those pending checks have passed.


Final capture source A is `0da98417b18b9064250c6014f9fb5254ca825fdb`.
All 80 published originals have real dark counterparts. The 43 generator/common
assets have 36 matching plotted hashes and 144 matching nested data fingerprints;
all 14 generator/common movies and four XPBD movies fully decode. Eight full-budget
non-generator jobs passed paired geometry checks. Native UI controls, statistics
and the original plot payload are identical under the two real Gradio themes.

Final-image QA additionally caught a clipped projected axis label and dim dark
Repulsive ticks. A dedicated bottom margin preserves Course geometry/camera, and
dark ticks use the palette's readable muted color while light stays unchanged.
Twelve Course/theme tests and 14 construction/provenance/Repulsive replay tests
passed. Both affected producers receive fresh committed-source captures.

The inventory also encountered an unused historical image with the same basename
as a current illustration. It now identifies the current source from actual built
file bytes while preserving historical assets and rejecting genuinely ambiguous
publication. RED/GREEN checks cover the reproduced duplicate and HTML-root escape;
22 focused theme/inventory tests passed.

The full CPU lane passed 3,723 tests with four optional UI skips and 836 deselections
(601.45 s), including executable documentation examples. The complete strict bootstrap
site also passed. Final registry enforcement and browser checks follow the fresh
captures; no final release approval is inferred from these results.


Corrective capture source B is `fa0d2fa96560f9d3a5eaddb54162db371224c9ec`.
The full Course producer reran both palettes: 13 figures, 48 samples and all 30
numerical checks match A; only the gate-frame image bytes change. All projected
labels fit while the camera and world coordinates remain unchanged.
Repulsive's growth movie/poster and two barrier figures were selectively recaptured
with matching-theme manifests. All light bytes remain identical; dark ticks are
readable. Raw/sample/force records and all 36 plotted hashes still match A. Only
actually recaptured files receive B provenance. The registry records each asset's
actual capture commit rather than relabeling the complete original capture.

The README now reuses the current paired six-family gallery instead of an older
illustration and selects both gallery and utility artwork by the native color
preference. Historical upgrade details remain in excluded project records and
release notes; the reader introduction focuses on current result handling.
