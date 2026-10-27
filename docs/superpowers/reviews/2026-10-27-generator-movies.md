# Generator movie layout and Circuit branding review

Scope: the graphical Circuit identity and native movie layouts on `v3`, with rendering source committed as `a25ef61c6a1d52ae4a87b42f970c326a0730bb8f`. The user withdrew the ASCII export, requested branding inside the videos, and asked to remove reconstructed-road panels from generator explanations.

## Result and independent review

Source review and a separate final encoded-media/integration review both passed with no actionable findings. The reviewer inspected all 18 decoded light/dark contact sheets and full-size scenes covering fallback, midpoint displacement, normalization, candidate selection, growth and solver tails. Executor self-review inspected the source diff, representative full-size posters and the desktop/mobile players.

All six generator movies now stop at construction/raw centerlines. Five retain a raw-output reference; Repulsive uses one fixed-domain growth scene. Shared sampling and the two XPBD movies retain their pipeline views. Captions and text alternatives match the scenes. The real ten-anchor, 8x8 teaching regimes, recorded draws and whole-loop fallbacks are retained.

The Circuit wordmark reuses the original pixel geometry, with theme-matched colors and transparent surroundings. A figure-level image reserves the lower-right footer without changing scientific axes or coordinate hashes. The ASCII export/download is removed; the fourteen graphical branding exports are byte-identical to the preceding approved artwork. Navigation is unchanged.

## Capture and integrity evidence

- Source was committed before final capture, which ran from a clean local checkout of that exact revision. Matplotlib and FFmpeg remain the rendering tools. No geometric interpolation or new numerical tuning was introduced.
- All nine MP4s and posters have both native light/dark variants. All eighteen movies retain their previous frame counts and durations, native 16:9 and same-size posters. Decoded logo regions match the posters across sampled construction/solver phases.
- Exactly eighteen media-registry pairs changed (nine movies plus nine posters). All 166 registered file hashes match. The other 65 pairs retain all 130 original file bytes and their provenance. Separate detailed Polar/Voronoi/Checkpoint diagrams are retained.
- Generator captures retain numerical signatures, counts, selected examples, force states and fallback parameters. The bounded examples now record automatic spacing as `None`; its resolved value remains 0.06. The solver capture hashes, selected environments, validity, distance observations and plotted-coordinate hashes match their previous captures exactly.
- Per-asset provenance points to the actual committed source and exact Python dependency hashes. Old static captures are not relabeled. Solver manifests retain their hardware-independent capture configuration and software versions.
- Runtime, package metadata and goldens match the fully validated automatic-spacing head `79f1cebf2ddbc1395865c364877fe05bfdecb4bd`. The immutable rc3 tag remains unchanged. No navigation, merge, tag or deployment action is included.

## Validation

- Generator scene regression: 12 cases, all six families in both themes. Real rendering is inspected at every frame; raw samples, stage coverage, pane removal, frame totals, logo/footer clearance and actual poster/frame equality are checked.
- Existing focused generator suite: 49 passed. Theme/helper checks: 20 passed. Sampling/solver checks: 9 passed. Independent source probes also checked all three pipeline clips in both themes.
- Spark focused optional-renderer/media suite: 119 passed, no skips, failures or errors, in 149.92 s. All 887 exported files match before and after. These are CPU media captures performed on Spark, not a new numerical CUDA/full-suite claim; the library source is unchanged.
- Strict Sphinx `-W --keep-going -E` passed with the enforced 83-pair registry. A final build includes the finalized downloadable manifests.
- Browser acceptance: all nine players in both themes on desktop and mobile (36 cases) played, paused and sought correctly, used 16:9 frames, and returned HTTP206 for byte ranges. No page overflow or browser/network errors. The loopback preview is refreshed from the strict build.
- Full CPU lane: 3,792 passed, four optional UI skips, 839 deselected, in 605.03 s. Executable documentation examples and retained numerical byte guards are included.

Racing validation and explicit final-release approval remain separate. No candidate tag was recreated and the PR remains draft.
