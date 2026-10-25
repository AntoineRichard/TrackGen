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
