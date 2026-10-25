# Racing documentation validation

The implementation is based on immutable `v0.3.0rc1` (`4664986`), isolated on
`codex/racing-docs`. No merge, final release or site deployment is authorized.

## Delivered

Current racing-first navigation, track/gate quickstarts, all six generator guides,
56 figures and eight videos, setting comparisons, collision/self-intersection
explanations, data/storage diagrams and extension recipes. Complete detailed
contracts and executable examples survive the split. Historical reader pages are
retained in the excluded `docs/library-readiness/archive/`; internal records remain.

## Validation and review

- Full CPU lane: 3,497 passed, four optional UI skips, 763 deselected in 164.02 s.
- Subsequent affected docs/tool/generator checks: 64 passed. Final installation fix:
  32 focused checks passed; seven shell examples passed syntax checks.
- Exact committed-tree CUDA documentation lane on spark-home at `63fb284`:
  97 passed, 122 deselected, zero skips in 28.93 s. Later edits change only installation
  text, README and excluded project records, so executable snippets are unchanged.
- Final strict Sphinx build with pytest, torch, scipy and matplotlib blocked passed.
  All 237 contract labels and README anchors resolve. External linkcheck reported
  34 working, one ignored, 305 unchecked and no broken links.
- All eight built videos loaded metadata with finite duration/dimensions. Desktop
  and mobile homepage inspected; mobile has no horizontal overflow. Unit reviewers
  inspected every new figure and sampled beginning/middle/end video frames.
- All 43 edited runtime source files have identical executable AST after stripping
  docstrings. Generation, collision algorithms and public API behavior match the RC.
- Three visual units and reader/extension work passed independent scoped reviews.
  Fresh whole-branch review found one material installation mismatch. One combined
  fix wave (`bd77ccc`) pins installation to the validated RC; scoped review confirms
  it addressed, with no material findings remaining.

A final build attempted without network could not load Python/NumPy intersphinx
inventories, producing dependent unresolved-reference warnings. Retrying with
approved network access passed without a source change.

## Decisions and remaining polish

- Independent visual units ran in parallel with exclusive files. Cost if wrong:
  integration conflicts; shared files stayed under the root implementer's ownership.
- Focused checks validate each unit; the full CPU suite runs at integration. Cost if
  wrong: an integration defect is detected later; final CI supplies another platform.
- Existing executable examples and strict references validate prose; renderer tests
  check meaningful data rather than copied prose or pixel bytes. Cost if wrong:
  a wording/visual mistake still needs review; those reviews were performed.
- Public source docstrings may be corrected only with identical executable AST.
  Cost if wrong: inaccurate explanations; behavior equivalence is separately checked.
- Installation uses existing `v0.3.0rc1` so commands match current guide semantics.
  Cost if wrong: readers select a candidate rather than an old stable version; this
  is explicitly labeled. The final `v0.3.0` release remains gated by user approval.
- Linked implementation-source comments retain their history; reader narrative,
  navigation and search exclude project history. Cost if wrong: source-view readers
  still encounter roadmap identifiers. Removing code comments exceeds reader scope.

Optional polish remains: old external subsection fragments could forward to moved
result-contract sections; active internal labels/links resolve. Additional fixture
finiteness assertions are optional hardening, with no observed poisoned rendering.

The docs branch is reviewable and can remain parked while PR #4 receives its
separately authorized review fixes. The user's racing validation gates the release.
