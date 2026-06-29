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
`metadata_status`, `evidence.csv` `code_status`, and `claims.csv`
`evidence_status`. Each requires exactly one value. The other controlled fields in
`evidence.csv` (`domain`, `course_object`, `representation_family`,
`generator_family`, `generation_role`, and `validity_strategy`) may contain one or
more semicolon-separated values. The validator canonicalizes controlled values by
trimming each token and joining lists with a semicolon followed by one space.

`NR` is permitted as a sole sentinel in controlled fields only in `evidence.csv`; it
must not be combined with another value and is invalid for candidate or claim statuses.
In noncontrolled factual cells, use `NR` only when the reviewed source does not report
the fact. Never infer a negative from silence. Use
`not_applicable` only when the controlled field provides that value and the concept does
not apply. Record a documented negative only when direct or allowed official evidence
supports it.

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
section and explicitly document the missing counts.

Each run ends with one `RUN-SUMMARY:<report-path>` row using
`search_surface=documented-agent-run`. Its `results_screened` may be `NR`, while
`candidates_added` is the integer number of rows in that run's output CSV. Summary notes
distinguish retained and excluded rows, preserve the final saturation arithmetic, and
state that the total screened-hit count was not captured.

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
