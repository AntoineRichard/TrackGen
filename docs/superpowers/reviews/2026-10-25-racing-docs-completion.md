# Racing documentation completion on rc2

The reader overhaul is reconciled with immutable `v0.3.0rc2` (`a42759a`). It remains
on `codex/racing-docs` in PR #5, targeting `v3`; this work does not authorize the
final release or public deployment. Finding #4's spacing contract is unchanged.

## Scope check

All requested content families are present: useful racing outputs and short track/gate
quickstarts; six detailed procedural generators with prominent centerlines, final roads,
settings and construction; imports and native/station gate courses; public data and full
API/storage/stream/capture contracts; generator/Stage/component extension recipes;
visual generation validity, self-intersections, collision models and numerical limits.
Historical reader pages remain in the excluded archive; internal project records remain.

A fresh coverage audit found two substantive weaknesses in the parked version. Both
are addressed: five simple panel reveals are replaced by progressive construction
explanations, and the solver guide now isolates iterations, margin and smoothing in
three actual same-seed comparisons. The eight animations are collected in the Visual
tour, linked from the homepage and navigation. The public data figure now includes
the per-environment generation epoch. Stale measured-yield promises, upgrade-path
wording and legacy-host-generator descriptions were removed.

## Reconciliation

- Current installation pins rc2. Split extension/reference examples use the shared
  `rng.init`, remove the sampler `keying` selector and describe current hashed behavior.
- The archived migration guide retains rc2's removals; it is excluded from the site.
- All 43 runtime files differ from rc2 only in docstrings. Executable AST comparisons
  pass, existing byte goldens are unchanged, and no public behavior is changed.
- The merge initially carried an old randomizer body in one conflict resolution.
  AST verification caught it; the exact rc2 implementation was restored before testing.
- Existing figure provenance is retained honestly; rerendered data/animation/solver
  assets record the actual source and settings used. Illustrative construction is
  distinguished from actual solver states; no elapsed-time or yield claim is inferred.

## Fresh verification before candidate commit

- Complete CPU lane: **3,525 passed, four optional UI skips, 763 deselected**, 215.37 s.
- Strict lean Sphinx `-W --keep-going -E` build: passed with rendering/test packages blocked.
- **237** contract labels and every README docs anchor resolve.
- Seven representative pages checked at desktop and 390px mobile widths: all figures
  load, no horizontal overflow. All eight video players load and play; five revised
  construction clips last 16 seconds with 96 frames. Worker reviews inspected their
  beginning, transitions and ending, and all three new solver figures.
- Focused checks include 23 updated contract snippet groups, 23 construction checks
  and 32 relaxation/geometry/link checks. The full CPU run includes all current CPU
  examples and the new renderer checks.
- CUDA snippet collection is 97 groups versus rc2's 109: 23 historical migration
  groups leave the collector when archived, 11 current groups are added, and all
  15 extension groups moved into reference pages retain execution coverage.

Exact committed-tree CUDA/full-suite results, independent review and final CI are
recorded in PR #5 after they complete. No work is pushed before required local/GPU
validation. The original-base evidence is retained in the adjacent validation record.

## Decisions

Use the approved Sphinx stack and render actual library data, keeping rendering tools
out of the docs build. Preserve full contracts behind the shorter reader path. Use
rc2 as the immutable install target while racing validation gates the final release.
A future re-render must inspect labels/failures as well as numerical checks; matching
pixels is not a correctness criterion. Retain internal history and implementation
comments outside the published narrative.
