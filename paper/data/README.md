# Survey corpus data contract

The files in this directory are the auditable data layer for the course-generation
survey. CSV headers and their order are fixed by `paper/scripts/validate_corpus.py`.
Use UTF-8 text, preserve every discovered candidate, and record only facts supported
by the reviewed source or an allowed official source.

## Coding conventions

Use semicolons to separate multiple values in one cell, with no space before and one
space after each semicolon. Strip surrounding whitespace from each item. A semicolon
is always a list separator and has no escape form inside a list item; rephrase an item
that would otherwise contain one. Empty elements, including double or trailing
semicolons, are invalid. Use standard CSV quoting for cells containing a comma, quote,
or newline, and double an embedded quote. CSV quoting does not turn a semicolon into
a literal character.

The scalar controlled fields are `candidates.csv` `screening_status` and
`metadata_status`, `evidence.csv` `survey_evidence_tier` and `code_status`, and
`claims.csv` `evidence_status`. Each requires exactly one value. The other controlled fields in
`evidence.csv` (`domain`, `course_object`, `representation_family`,
`generator_family`, `generation_role`, and `validity_strategy`) may contain one or
more semicolon-separated values. The validator canonicalizes controlled values by
trimming each token and joining lists with a semicolon followed by one space.

`survey_evidence_tier=core` directly supports generated or parameterized-course
technical claims. `supporting` covers transferred fixed-course, benchmark, interface,
metric, simulator, dataset, or reporting requirements, but not a generation method.
`contextual` covers field structure, terminology, or survey-gap evidence, not
implementation or performance evidence. Retained-source count is not
generation-method count.

`NR` is permitted as a sole sentinel in controlled fields only in `evidence.csv`; it
must not be combined with another value and is invalid for candidate or claim statuses.
In noncontrolled factual cells, use `NR` only when the reviewed source does not report
the fact. Never infer a negative from silence. Use
`not_applicable` only when the controlled field provides that value and the concept does
not apply. Record a documented negative only when direct or allowed official evidence
supports it.

For `survey_evidence_tier`, `NR` is an uncoded draft/transition sentinel, including
for migrated pre-v7 rows. It is valid while contribution extraction is incomplete,
but it must be resolved to `core`, `supporting`, or `contextual` before the source
enters final tier reliability estimates, method counts, or claim synthesis. `NR` is
not a taxonomy value.

Use `code_status=not_found` only for an explicit search outcome across documented
official project and author surfaces. Record the searched locations and search date in
`coding_notes`. Use `code_status=NR` when availability was not assessed or not reported.
Code and asset status otherwise require an official repository or author page;
third-party search results and silence are not evidence of availability or absence.

Evidence locators must identify the supporting material as a page, section, figure,
table, appendix, or official URL, for example `p. 7`, `Section 3.2`, `Figure 4`,
`Table 2`, `Appendix A`, or a complete official URL.

The controlled vocabulary is stored in `taxonomy.json`. Values may be split or merged
only through a recorded codebook decision in Task 7.

Do not mechanically redesign `claims.csv` for these tiers: claims currently have no
claim-type field, so the tier is a source-role guardrail rather than a claim type.

## Screening rules

Include a source when it satisfies at least one of these rules:

1. It synthesizes, selects, mutates, repairs, validates, or serializes course geometry
   or a course distribution for robot racing or an adjacent transferable domain.
2. It defines a fixed benchmark, competition course set, or simulator interface that
   materially constrains course representation or evaluation.
3. It defines a metric or dynamics model used to characterize generated courses.
4. It is an adjacent survey needed to establish the survey gap.

Mark a source `boundary` when it studies racing or control on fixed courses but
contributes a requirement, metric, dataset, or reporting practice used by this survey.

Exclude a source when it only optimizes a racing line inside a fixed corridor, only
randomizes appearance or dynamics, only generates traffic participants on fixed roads,
or mentions a course without enough detail to support a survey claim. Preserve every
excluded row in `candidates.csv` and record its specific exclusion reason.

The concise rules above are orientation only. The complete normative eligibility,
evidence, independence, calibration, adjudication, automation, and accountable-author
requirements are in `screening_protocol.md`; that frozen file governs every rating.

## Immutable duplicate-screening workflow

Screening artifacts are versioned snapshots, never mutable `current` directories.
The canonical stage order is:

1. freeze the 202-candidate coordinator under `screening_inputs/vN`;
2. publish only the 30-candidate calibration release under
   `screening_releases/calibration/vN`;
3. collect six isolated reviewer result files under `screening_runs/calibration/vN`;
4. seal all 60 ratings under `screening_results/calibration/vN`;
5. seal the calibration decision under `screening_decisions/vN`;
6. only after a passing decision, publish the 172-candidate main release under
   `screening_releases/main/vN`;
7. collect and seal all 344 main ratings under the corresponding run and result
   versions;
8. seal adjudications and their execution register under
   `screening_adjudication/vN`; and
9. after exact accountable-author verification of every final candidate decision,
   seal the publication projection under `screening_projection/vN`.

The coordinator binds candidates, conflicts, bibliography, citation-key ledger,
taxonomy, protocol, the frozen execution profile and reviewer-prompt template,
deterministic assignments, and the stable 30-candidate calibration selection. Each
reviewer release contains those frozen execution inputs and the six coordinator-held
packets for one phase. Before a reviewer starts, `--stage-role` derives a random
role-private snapshot containing exactly that role's one packet, rendered prompt, and
hash preimages. Each phase-result snapshot binds the exact reviewer release from which
its assignments were visible. Main publication and main-result sealing additionally
bind the calibration release, the 60-rating calibration result, and the passing
calibration decision as one coherent authorization tuple.

Reviewer roles are `screening-01` through `screening-06`. Each role writes only its
assigned canonical result file without another reviewer's ratings or traces. Automated
reviewer rows MUST use `human_role=NR`. Each automated reviewer MUST start with
`fork_context=false` in a fresh context. On a shared host, procedural isolation uses a
separately generated random, role-private working and output path for each execution;
this reduces accidental cross-role access but is not a claim of ACL, container, mount,
or same-user process isolation. If role staging fails, its private parent is intentionally
retained for inspection and MUST NOT be treated as a valid reviewer input. Provider
retrieval-cache isolation is explicitly
recorded as not exposed and reported as a residual limitation. Human and hybrid
execution records use a
stable `human_role` identity binding; distinct role labels alone do not establish
distinct human reviewers. Raw ratings are immutable after sealing. Discussion and
adjudication append records and never rewrite them. Accountable-author verification of
all 202 final decisions and every deciding evidence locator remains mandatory before
publication.

The immutable-input and publication tools reject symlinks, aliases, path overlap,
noncanonical modes, changed inodes, unexpected entries, checksum drift, and
self-consistently rehashed substitutions. Cleanup uses a capture-then-classify
operation: after a best-effort identity precheck, `renameat2(RENAME_NOREPLACE)`
atomically captures whatever occupies the source name into a fresh
`.trackgen-retired-*` quarantine name, and only then classifies the captured
`(dev, ino)`. Expected transaction entries remain quarantined; snapshot cleanup
captures the complete root once and never recursively deletes children. A captured
foreign entry is restored with one no-replace rename when the source is still empty.
If another writer has refilled the source, cleanup overwrites neither name: both
entries remain, and the error reports the foreign inode's exact quarantine path and
identity for recovery. Thus cleanup never deletes or overwrites an entry, but a
non-cooperating same-privilege writer can cause a foreign entry to be moved into
quarantine. Preventing even that movement requires exclusive control of the parent
namespace.

The directory names above define roles, not committed production versions. No
calibration or main snapshot exists until the infrastructure tests and independent
protocol/code reviews pass. Accountable-author verification is scientific inspection,
not an automated formality, and MUST NOT be generated from the structural validator.

## DOI normalization

DOIs are compared after trimming whitespace, converting to lowercase, removing one
leading `https://doi.org/`, `http://doi.org/`, or `doi:` prefix, and removing trailing
slashes. Store the bare canonical DOI when possible. Two nonempty values that normalize
to the same DOI are duplicates.

## `search_queries.csv`

This file freezes the discovery query matrix. Each row has a unique `query_id`, one of
the matrix's allowed discovery streams, a domain from `taxonomy.json`, and nonempty
query and rationale text. The validator checks the generic matrix contract; it does not
hardcode the current query strings.

## `search_log.csv`

One row records one reproducible search action. Assign a unique `search_id`; record the
date, discovery stream, agent, exact query, search surface, counts screened and added,
and any notes needed to replay or interpret the search.

Count fields are nonnegative decimal integers by default. The literal sentinel `NR` is
allowed only for non-bootstrap rows whose notes explicitly state that the count or
counts were not captured. Blank, negative, and malformed values remain invalid.

For `stream=bootstrap` and `search_surface=local-corpus`, `results_screened` counts
inventoried visible named-mention occurrences in the named local file. It includes
repeated mentions under different headings and repeated summary occurrences. This unit
must equal the number of `seed_coverage.csv` rows for that source path. Bootstrap count
fields are always nonnegative integers; `NR` is not permitted.

Task 4 exact-query rows reproduce every executed query recorded in an immutable agent
report, in report order and without deduplicating queries repeated across rounds. They
use `search_surface=mixed-primary-web`; both count fields are `NR` because per-query hit
and addition counts were not recorded. Their notes identify the report and round or
section exactly as recorded and explicitly document the missing counts.

Each run has exactly one `RUN-SUMMARY:<report-path>` row using
`search_surface=documented-agent-run`. Its `results_screened` may be `NR`, while
`candidates_added` is the integer number of rows in that run's output CSV. Summary notes
distinguish retained and excluded rows, preserve the final saturation arithmetic, and
state that the total screened-hit count was not captured. Under the append-only search
ledger, corrective query rows may follow a run's summary; the summary need not be
physically last.

Each paired report contains exactly one `final-saturation` fenced block under
`## Canonical final-round record`. Its two ordered lines use
`round=R# added=<integer> denominator=<positive integer> cumulative_retained=<integer> percent=<decimal>%`.
Round identities must be different and consecutive, both recomputed percentages must be
below 5%, and the summary note repeats both records exactly. A displayed percentage is
valid when its absolute difference from `100 * added / denominator` is no more than
half one unit in its final displayed decimal place.

The Task 4 rows document search actions and output accounting. They must not be pooled
with bootstrap named-mention counts, and `NR` must not be converted into zero or an
invented screened-hit total.

## `agent_runs/`

This directory contains exactly four independent discovery outputs, each as one CSV and
one Markdown report. The CSVs share a fixed 35-column header. Validate their filenames,
row shape, run-specific IDs and provenance, status scalars, list encoding, evidence
surfaces, deduplication, and report coverage with
`python3 -m paper.scripts.validate_agent_runs` from the repository root.

Blind-run technical fields preserve the agents' descriptive terminology and are not
checked against `taxonomy.json`. Integration validates structure and provenance without
rewriting the scientific content of the run artifacts.

Runtime validation also reads the sibling `taxonomy.json` and `search_log.csv`. For each
run it compares the report's exact-query multiset with exact-query log rows for the
declared stream and agent, including the recorded report section, preserves duplicate
executions and append-batch report order, and requires exactly one summary. Structured
summary notes must match the report path, CSV row count, retained/excluded status counts,
and two final saturation statements present in the paired report.

Blind-run `discovery_query` values are exact report/log literals. Aware-run values use
exactly one explicit mode: `query::<literal>`, `seed::<C####>`, or
`citation::<source identifier>`. A `query::` literal must resolve through both exact-query
ledgers. A `seed::` ID must name an existing bootstrap row in `candidates.csv`, and the
paired aware report must record that row-to-seed relationship. A `citation::` value must
use a stable nonempty identifier and match the report's citation provenance ledger.

All cells other than a non-excluded row's `exclusion_reason` must use `NR` instead of a
blank. `cite_key=NR` is permitted. Excluded rows require a specific reason. Aware-run
controlled fields use `taxonomy.json` values or sole `NR`, and every aware-run
`coding_notes` cell must state bootstrap/seed lineage or newly discovered provenance.

An `evidence_locator` must contain a complete HTTP(S) URL or a page, section, table,
figure, appendix, source-path, or line marker. This is only a structural locator check;
it does not prove that the cited material semantically supports the coded claim.

## `metadata_manifest.csv` and `metadata_runs/`

The versioned manifest freezes every candidate exactly once into one of six
metadata-verification batches. `input_sha256` binds the complete candidate row and all
sorted conflict rows; `snapshot_sha256` binds the complete assignment. The committed
manifest is semantically bound to the explicit immutable version
`metadata_inputs/v1/`, whose `candidates.csv` and `conflicts.csv` are the exact
pre-integration bytes from commit `c96c2d6`. Their raw SHA-256 digests are
`62b7fc3a2716f923422b77d538e9cfb4c95cefb1687bf979af4cb953656e90a3` and
`4495d57179822dd099299825015bc27a6ddf91e397ecbba8a4ac63ec1363ca52`,
respectively. There is deliberately no mutable `current` path. The repository
`.gitattributes` marks every file below `metadata_inputs/` as binary (`-text`), so Git
does not apply checkout newline conversion. Raw-byte integrity is intentionally checked
separately from the semantic manifest contract.

Verify the exact checked-out `v1` bytes with:

~~~bash
set -euo pipefail
snapshot_hashes=(
  "62b7fc3a2716f923422b77d538e9cfb4c95cefb1687bf979af4cb953656e90a3  paper/data/metadata_inputs/v1/candidates.csv"
  "4495d57179822dd099299825015bc27a6ddf91e397ecbba8a4ac63ec1363ca52  paper/data/metadata_inputs/v1/conflicts.csv"
)
printf '%s\n' "${snapshot_hashes[@]}" | sha256sum --check --strict
~~~

Validate the committed manifest and frozen inputs after canonical integration with this
non-mutating command:

~~~bash
set -euo pipefail
python3 paper/scripts/prepare_metadata_batches.py \
  --snapshot-dir paper/data/metadata_inputs/v1 \
  --output paper/data/metadata_manifest.csv
~~~

Each metadata agent writes only its assigned `metadata-0N.csv` and
`metadata-0N-conflicts.csv` under `metadata_runs/`. Result rows must match the
manifest candidate ID, batch ID, and input hash. Agents never edit canonical candidates,
conflicts, bibliography, or BibTeX files; the central integrator validates all six result
pairs before producing those artifacts.

## `citation_keys.csv`

This two-column ledger is the append-only authority for issued citation identities.
It contains `candidate_id,cite_key` only: mutable title, author, year, screening, and
metadata fields never participate in the stored assignment. Existing rows may not be
renamed, reassigned, reordered, or deleted. Key uniqueness is case-insensitive across
all rows, including dormant assignments.

The current ledger preserves the 184 keys issued by the first canonical metadata
integration as an exact prefix and appends 14 newly verified candidates in numeric ID
order. Its full-file SHA-256 is
`48d891587257f79b9c7cf97f90dd3ebd36bd0378e9ed8c100628afcfd6540e5f`.
A verified, non-excluded candidate must have its ledger key in `candidates.csv`,
`bibliography.csv`, and `references.bib`. An excluded or otherwise inactive candidate
may retain a dormant ledger reservation while generated citation artifacts omit it;
re-inclusion restores the same key.

Routine metadata replay is strict and fails when an active candidate lacks a ledger
assignment. Adding candidates requires the explicit `--extend-citation-keys` mode and a
distinct `--output-citation-keys` path. Review the append-only output before publishing
it; strict replay never rewrites the ledger or seeds keys from generated candidates.

The following Bash block replays all six result pairs from `v1` into a temporary
directory and compares every generated artifact byte-for-byte with the canonical tree.
A failed `cmp` means reviewed run rows and canonical integration are out of sync; update
canonical outputs through the integrator rather than weakening the comparison.

~~~bash
set -euo pipefail
metadata_replay_args=(
  --candidates paper/data/metadata_inputs/v1/candidates.csv
  --conflicts paper/data/metadata_inputs/v1/conflicts.csv
  --manifest paper/data/metadata_manifest.csv
  --citation-keys paper/data/citation_keys.csv
  --metadata-result paper/data/metadata_runs/metadata-01.csv
  --metadata-result paper/data/metadata_runs/metadata-02.csv
  --metadata-result paper/data/metadata_runs/metadata-03.csv
  --metadata-result paper/data/metadata_runs/metadata-04.csv
  --metadata-result paper/data/metadata_runs/metadata-05.csv
  --metadata-result paper/data/metadata_runs/metadata-06.csv
  --conflict-result paper/data/metadata_runs/metadata-01-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-02-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-03-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-04-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-05-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-06-conflicts.csv
)
replay_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "$replay_dir"
}
trap cleanup EXIT
python3 paper/scripts/integrate_metadata.py "${metadata_replay_args[@]}" \
  --output-candidates "$replay_dir/candidates.csv" \
  --output-conflicts "$replay_dir/conflicts.csv" \
  --output-bibliography "$replay_dir/bibliography.csv" \
  --output-bibtex "$replay_dir/references.bib"
cmp -- "$replay_dir/candidates.csv" paper/data/candidates.csv
cmp -- "$replay_dir/conflicts.csv" paper/data/conflicts.csv
cmp -- "$replay_dir/bibliography.csv" paper/data/bibliography.csv
cmp -- "$replay_dir/references.bib" paper/references.bib
~~~

After review, publish the same replay through the integrator's staged four-output writer:

~~~bash
set -euo pipefail
metadata_replay_args=(
  --candidates paper/data/metadata_inputs/v1/candidates.csv
  --conflicts paper/data/metadata_inputs/v1/conflicts.csv
  --manifest paper/data/metadata_manifest.csv
  --citation-keys paper/data/citation_keys.csv
  --metadata-result paper/data/metadata_runs/metadata-01.csv
  --metadata-result paper/data/metadata_runs/metadata-02.csv
  --metadata-result paper/data/metadata_runs/metadata-03.csv
  --metadata-result paper/data/metadata_runs/metadata-04.csv
  --metadata-result paper/data/metadata_runs/metadata-05.csv
  --metadata-result paper/data/metadata_runs/metadata-06.csv
  --conflict-result paper/data/metadata_runs/metadata-01-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-02-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-03-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-04-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-05-conflicts.csv
  --conflict-result paper/data/metadata_runs/metadata-06-conflicts.csv
)
python3 paper/scripts/integrate_metadata.py "${metadata_replay_args[@]}" \
  --output-candidates paper/data/candidates.csv \
  --output-conflicts paper/data/conflicts.csv \
  --output-bibliography paper/data/bibliography.csv \
  --output-bibtex paper/references.bib
~~~

A future refreeze must name a new, non-existing version explicitly. For example, after a
documented corpus change, create `v2` and replace the manifest together with:

~~~bash
set -euo pipefail
snapshot_dir="paper/data/metadata_inputs/v2"
if [[ -e "$snapshot_dir" || -L "$snapshot_dir" ]]; then
  printf 'snapshot version already exists: %s\n' "$snapshot_dir" >&2
  exit 1
fi
python3 paper/scripts/prepare_metadata_batches.py \
  --candidates paper/data/candidates.csv \
  --conflicts paper/data/conflicts.csv \
  --snapshot-dir "$snapshot_dir" \
  --output paper/data/metadata_manifest.csv \
  --refreeze
~~~

Initial freeze uses direct `--candidates` and `--conflicts` arguments without
`--refreeze`; both the manifest and named version must be absent. Once the manifest
exists, ordinary validation omits the direct inputs, names the immutable version
explicitly, and performs no writes.

The tool copies the source bytes into a hidden staging directory, builds and stages the
manifest from those copies, and publishes the immutable version with Linux
`renameat2(RENAME_NOREPLACE)`. Any existing or racing entry, including a symlink or
empty directory, is rejected without replacement. Legacy direct manifest creation uses
the same atomic no-clobber operation.

Before refreeze publication, the tool creates an adjacent recovery journal matching
`.metadata_manifest.csv.recovery.*`. It contains `old-manifest.csv`, an independent
byte copy of the pre-refreeze manifest, and `old-manifest.inode`, a hard link used for
exact-inode checks. After publishing the snapshot, the tool atomically exchanges the
staged and current manifests with `renameat2(RENAME_EXCHANGE)`, verifies the swapped-out
inode and SHA-256 content, and rechecks ownership of the published inode.

The bounded guarantee is: while the canonical path still contains the transaction-owned
new inode, exchange-back restores the exact manifest displaced by the final swap. If
ownership changes again or restoration cannot be proven complete, the tool does not
overwrite the unknown canonical manifest. It retains the old byte copy and inode link,
moves the published snapshot to `snapshot/`, preserves any swapped path as
`swapped-manifest.csv`, writes `RECOVERY.txt`, and annotates the original exception
with the exact recovery directory. A complete success or rollback removes the journal.
Changes made by a non-cooperating writer after the final ownership check are outside
this bounded guarantee.

Generated snapshot version directories use mode `0755` and their two CSV files use
`0644`, independent of the caller's umask. `--refreeze` requires an existing regular
non-symlink manifest and an explicit new `--snapshot-dir`. Legacy direct-input
validation remains available, but the explicit versioned workflow above is canonical
for this corpus.

## `candidate_aliases.csv`

This ledger records explicit candidate-ID retirements when multiple assigned rows are proven to describe one citable work or one project whose paper is its citable system description. Each row names one retired ID, one surviving ID, a specific reason, and primary or official evidence. Migrations are direct and acyclic: unaffected IDs never change, retired IDs remain permanent gaps, and neither retired nor surviving IDs may be reassigned. A shared repository, a similar title, or `seed::` discovery provenance alone is not sufficient evidence for an alias. Distinct versions, standards, competitions, and related projects remain separate unless the ledger states otherwise.

## `candidate_corrections.csv`

This ledger records reviewed canonical-field corrections that must replay identically during a full corpus rebuild. Each row identifies the candidate and bibliographic field, preserves both old and new values, and supplies a reason, authoritative evidence, and resolver. The merge creates or updates the corresponding conflict rather than deleting the original observation. Correction and candidate-source origins carry complete-row SHA-256 digests so their meaning remains stable after later edits.

## `candidates.csv`

One row records one discovered source, including provenance, screening disposition, and
metadata verification. `candidate_id` is required and unique. Included and boundary
sources require a unique `cite_key`, verified metadata, and exactly one matching row in
`evidence.csv`. Excluded sources require a specific `exclusion_reason`.
Candidate IDs are stable after assignment and surviving records are never renumbered.
Retired IDs remain as documented gaps, such as C0072, so later records keep durable
identities across reviews and merges.


This table records discovery and screening, not verified technical coding. Conflicting
metadata remains visible with `metadata_status=conflict` until it is resolved through
`conflicts.csv`.

## `seed_coverage.csv`

One row maps one pre-existing seed mention to a candidate. `coverage_status` is
`unreviewed`, `linked`, or `excluded`. Linked and excluded rows must identify an existing
`candidate_id`; notes explain non-obvious mappings or exclusions.

## `evidence.csv`

One row contains the verified technical coding for one included or boundary source.
Its `cite_key` must match that source exactly. Code the representation, generation role,
validity, metrics, distributions, simulator/export details, reproducibility, and
availability only from inspected direct evidence. Put ambiguity, multiple roles, and
domain-transfer assumptions in `coding_notes` and provide precise `evidence_locator`
values.

## `claims.csv`

One row records one manuscript claim and its support state. `claim_id` is unique.
Separate multiple supporting `cite_keys` with semicolons; every key must identify an
included or boundary source. Use `reviewer_notes` for support limitations or follow-up
review findings.

## `metrics.csv`

One row defines one survey metric, its layer, procedure, units, preferred direction,
domain, dynamics dependency, minimum reporting requirement, sources, and limitations.
`metric_id` is unique. Separate multiple `cite_keys` with semicolons, and retain the
source's reported metric name rather than silently translating terminology.

## `simulators.csv`

One row records one simulator or system interface and the details needed to reproduce
course loading and evaluation. A nonempty `cite_key` must identify an included or
boundary source. Record coordinate frames, units, collision geometry, reset behavior,
RL interfaces, and open-source status only when explicitly documented.

## `conflicts.csv`

One row preserves one unresolved or resolved disagreement about a record field.
`conflict_id` is unique. This phase supports only `record_type=candidate` and
`record_type=evidence`. A candidate conflict uses `candidate_id` as `record_key`, and
its `field` must be a `candidates.csv` column. An evidence conflict uses `cite_key` as
`record_key`, and its `field` must be an `evidence.csv` column. Unsupported record
types, orphaned keys, and unknown fields are invalid. Keep both observed values and
identify their source context in `resolution_evidence`. A nonempty resolution requires
both a resolver and resolution evidence; never erase the original disagreement.
