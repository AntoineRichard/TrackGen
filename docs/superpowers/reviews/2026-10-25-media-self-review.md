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


Final browser review at prepared candidate `e8acd90414b4caee7285fee818ac201ebfb4fdff`
passed all 71 reader pages in explicit light/dark, nine mobile pages in both themes,
all 160 delivered media hashes, all nine movies' themed playback/posters, native
system preference, full-media/download links, playing/paused/rate/volume retention
and rapid switches. Both README picture elements select and decode the actual
native variants. No console/network errors or horizontal clipping remained.
Thirty production movie frames were inspected for offsets, spline handles, Hull
midpoints and whole-loop fallback, Polar neighbors/radii, shared sampling and gate
convergence. The final strict Sphinx build passed with the registry enabled and
243 generated labels. Version/public API/fingerprint/reader checks passed 39 tests;
runtime, installed package metadata and pyproject all agree on `0.3.0rc3`.

Relative to the GPU runtime-test tree `23fea40`, the only library file changed is
`track_gen/_version.py`. Subsequent implementation changes are maintenance renderers,
reader documentation and their tests. Final Linux platform CI and exact-candidate
spark-home validation remain release gates; no final release approval is implied.

## Calmer sample and common movie aspect

The user requested a calmer Bezier construction example and the same height/width
ratio for every video. Bezier now uses real seed112 with four corners and
min_point_distance=.25. It is the first valid, four-distinct-cell example with all
sampled points inside their displayed cells in the declared0..2047 pool. The
sampler/distribution and default galleries are unchanged. Settings are visible;
the default97 fallback remains a separately labelled real example.

All nine movies and their video posters use native16:9 canvases. The common helper
rounds pixel width to a multiple of32 and derives an even9/16 height before placing
artists, with no raster cropping/padding. Previews encoded/decoded successfully at
1696x954 (Bezier/Hull),1440x810 (other generator/shared clips) and1216x684 (XPBD).
Generic Voronoi/Checkpoint players gain actual-frame posters; their detailed static
diagrams stay separate. Checkpoint inset titles/captions were adjusted for clarity.
The resulting reader inventory has82 original media files plus82 dark counterparts.

A final audit mutation exposed missing child-axis coordinates. Geometry hashing
now walks child/grandchild axes in stable order with identity deduplication and no
forced draw. Both mutations failed first, then passed. Animation refresh also
replays its own explicit seed/config rather than borrowing the gallery pool;
the seed112-vs-gallery regression failed before the fix. Per-file provenance
allows honest selective capture sources while requiring identical runtime hashes.
The integrated media/source checks passed101 tests in57.03s. Final clean-source
captures, registry hashes, browser checks and combined release gates follow.

## Portable finished-band regression

The full spark-home run at23fea40 passed4517 tests and exposed two host-sensitive
seeded preconditions (four optional UI skips). On Mac, seed7/growth3..3.5 reaches
its raw target then the tail meets a disc. On Linux ARM its raw length6.2434 misses
target8.27986; finalfail257 correctly includes generator and disc rejection.
Independent double-precision geometry confirms both outcomes are correctly rejected.

Commit6413222 replaces optimizer branch assumptions with captured raw input/discs
and target, then calls the real raw certificate and unchanged shared tail. It
independently proves raw target/width clearance, final disc penetration, exact
fail256 and keep/poison behavior. Disabling final barrier validation makes both
cases fail, and independent review found no issues. All70 related CPU/CUDA tests
passed on spark-home without skips. Numerical runtime and golden values are unchanged.

The user then refined the construction view to an8x8 grid. Its actual selected
seed is11560502 with four corners and min_point_distance=.0625 (pitch.125).
Sampler-only canonical epoch0 replay matched a real64-env generator byte for byte.
A compact ascending search stopped after twelve1,048,576-seed chunks, through
12582911, within a declared67108864-seed bound. Only matching candidates entered
the full pipeline; no large TrackGenerator was allocated. All four recorded points
lie inside their selected lower-left-origin cells, and the real finished road is
valid. Settings and selection provenance are visible; the unchanged distribution
is not described as a within-cell guarantee. Both preview movies fully decode
144frames at1696x954 and preserve matching light/dark geometry. The23 focused
sampler/selection/construction checks passed. Only actual newly captured files
will receive this source revision; earlier captures retain their true sources.

Final widescreen capture sources arefa1ee66 (all nine producers) andbeb6e3c
(the final8x8 Bezier replacement only). Per-file metadata retains earlier captures
for the untouched illustrations. All14 generator/common movies fromfa1 fully
decoded, with matching16:9 movie posters and unchanged physical data except the
explicitly selected Bezier illustration. The final replacement is the actual
11560502 input and preserves every other animation/family record. XPBD's four
movies also fully decode and preserve captured states, cameras, selections and
holds. The assembled82-pair/9-movie registry passed the strict default Sphinx
build; all243 labels checked without problems. Actual assembly/provenance/reader
checks passed80 tests in37.32s. Final whole CPU/GPU lanes and final-site browser
acceptance are the remaining release gates.
