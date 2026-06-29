# Survey corpus data contract

The files in this directory are the auditable data layer for the course-generation
survey. CSV headers and their order are fixed by `paper/scripts/validate_corpus.py`.
Use UTF-8 text, preserve every discovered candidate, and record only facts supported
by the reviewed source or an allowed official source.

## Coding conventions

Use semicolons to separate multiple values in one cell, with no space before and one
space after each semicolon. Strip surrounding whitespace from each item. A semicolon
is always a list separator and has no escape form inside a list item; rephrase an item
that would otherwise contain one. Use standard CSV quoting for cells containing a
comma, quote, or newline, and double an embedded quote. CSV quoting does not turn a
semicolon into a literal character.

Use the literal `NR` only when the reviewed source does not report a fact. Do not
infer a negative from silence. Use `not_applicable` only when the controlled field
explicitly provides that value and the concept does not apply. Record a documented
negative only when the paper, official supplement, official repository, or author
page explicitly supports it, and provide the corresponding evidence locator.

Evidence locators must identify the supporting material as a page, section, figure,
table, appendix, or official URL, for example `p. 7`, `Section 3.2`, `Figure 4`,
`Table 2`, `Appendix A`, or a complete official URL. Code and asset status require an
official repository or author page; third-party search results and silence are not
evidence of availability or absence.

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

## `search_log.csv`

One row records one reproducible search action. Assign a unique `search_id`; record the
date, discovery stream, agent, exact query, search surface, counts screened and added,
and any notes needed to replay or interpret the search.

## `candidates.csv`

One row records one discovered source, including provenance, screening disposition, and
metadata verification. `candidate_id` is required and unique. Included and boundary
sources require a unique `cite_key`, verified metadata, and exactly one matching row in
`evidence.csv`. Excluded sources require a specific `exclusion_reason`.

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
`conflict_id` is unique. Keep both observed values and identify their source context in
`resolution_evidence`. A nonempty resolution requires both a resolver and resolution
evidence; never erase the original disagreement.
