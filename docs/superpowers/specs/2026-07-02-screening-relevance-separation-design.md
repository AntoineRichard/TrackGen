# Screening Relevance And Contribution Separation Design

**Status:** Approved design

**Decision:** Screening v6 uses one controlled inclusion criterion,
`include-relevant`. The source-native eligibility rules remain disjunctive and retain
their current scope, but contribution types are characterized during evidence
extraction rather than forced into mutually exclusive screening labels.

## Context

The sealed v5 calibration reached the required exact status agreement of 24/30, but
criterion agreement was 18/30. Three records, C0025, C0172, and C0175, repeated the
same `include-1` versus `include-2` disagreement after an explicit precedence rule had
been added. The reports were consistently regarded as relevant; reviewers disagreed
about whether a course operation or an interface, standard, or benchmark representation
was primary.

That distinction is scientifically useful but is not an eligibility boundary. Making
it a single-choice screening criterion asks reviewers to resolve an artificial
taxonomy problem before inclusion. V6 separates the two decisions: screening answers
whether a source is eligible, and evidence extraction records all supported
contribution types.

This revision does not broaden the survey scope, change the definition of a course,
weaken source-native evidence requirements, alter access sufficiency, or reinterpret
v5 ratings. The v5 coordinator, ratings, and revise decision remain immutable history.

## Alternatives Considered

1. **One inclusion criterion with downstream characterization (selected).** Every
   eligible source records `included,include-relevant`. Existing evidence fields then
   capture geometry operations, representations, interfaces, benchmarks, metrics, and
   survey context without forcing one to displace another. This aligns the screening
   variable with the actual gate question.
2. **Merge only `include-1` and `include-2`.** This would fix the observed pair while
   retaining primary labels for metric and survey sources. It leaves the same
   single-primary-label problem whenever a generator also introduces a metric or an
   adjacent review contributes a reusable taxonomy.
3. **Keep v5 labels and gate on status agreement only.** This would preserve the
   ambiguous data model while declaring its disagreements non-blocking. It weakens
   auditability because a field used in adjudication would remain knowingly
   under-specified.

## Normative Eligibility Model

The three screening statuses remain `included`, `boundary`, and `excluded`.

An `included` result MUST use criterion `include-relevant`. A source is eligible when
material evidence establishes at least one of the existing source-native inclusion
rules:

- it operates on explicit course geometry or a course distribution for racing robots
  or a named transferable adjacent domain;
- it defines a representation, design or simulator interface, dataset, benchmark,
  competition course set, or interchange artifact specifically for generated or
  parameterized courses;
- it defines or validates a metric, feasibility test, or dynamics-grounded model by
  applying it to variation across generated or parameterized courses; or
- it is a survey or systematic review directly needed to establish, organize, or
  substantiate the course-generation survey gap.

These rules are alternatives for establishing relevance, not competing criterion
values. A source satisfying several rules remains one `included,include-relevant`
screening result. Reviewers MUST NOT choose or rank a primary contribution subtype in
the screening row.

Boundary and exclusion behavior is unchanged. A boundary result uses `boundary`. An
excluded result uses one existing controlled exclusion criterion and a substantive
source-specific reason. Inclusion retains precedence over boundary and exclusion.
Adjacent-domain inclusion still requires a source-native mapping in `notes`.

## Contribution Characterization

No new evidence column is introduced for v6 screening. After inclusion and
adjudication, the existing multi-valued evidence schema records supported contribution
types without a single-primary constraint:

- `generation_role` captures geometry synthesis, task selection, mutation, repair,
  serialization, and benchmark-only roles;
- `representation_family`, `course_object`, `simulator`, and `export_format` capture
  representations and interfaces;
- `validity_strategy` and the geometry, difficulty, and diversity metric fields
  capture characterization and feasibility contributions; and
- candidate source type, claims, evidence locators, and `coding_notes` retain survey
  context and source-specific nuances.

Evidence values remain source-supported and may be multi-valued where the current data
contract permits. The screening `notes` field is not a substitute for evidence
extraction and need not enumerate contribution subtypes.

## Version-Aware Validation

Historical v1-v5 snapshots use `include-1` through `include-4` and MUST continue to
validate byte-for-byte. V6 therefore MUST NOT replace a process-global constant in a
way that invalidates old snapshots.

The frozen v6 coordinator taxonomy declares
`"screening_inclusion_criterion": ["include-relevant"]`. Coordinator capture carries
that vocabulary into phase validation, and role staging binds it as
`"allowed_inclusion_criteria": ["include-relevant"]` in immutable execution
configuration version 2 for the role-local validator. Old coordinator snapshots and
stage configuration version 1 predate these fields and use the legacy four-value
vocabulary. The freeze path for every new coordinator requires the taxonomy field;
omission is not a valid way to regain legacy behavior. Historical snapshot validation
remains the only fallback path.

The result CSV header remains unchanged. For v6, structural validation rejects
`include-1`, `include-2`, `include-3`, and `include-4`; for historical snapshots it
continues to accept the vocabulary frozen with those snapshots. Public unbound helper
behavior remains legacy-compatible and MUST NOT be used to validate a v6 sealed or
staged result without its coordinator or stage binding.

## Protocol And Coordinator Changes

Create protocol and coordinator version v6 from the byte-identical v5 bibliographic
inputs and stable calibration selection. The substantive changes are limited to:

- replacing the four single-choice inclusion criteria with the one
  `include-relevant` value and the disjunctive eligibility rules above;
- removing include-1/include-2 precedence and primary-subtype instructions;
- updating decision, pairing, calibration, agreement, and adjudication text to the new
  inclusion vocabulary;
- adding the frozen inclusion vocabulary to v6 taxonomy and role execution
  configuration; and
- updating validators and focused tests for version-aware behavior.

The screening result schema, assignments, candidate corpus, conflicts, bibliography,
citation-key ledger, exclusion vocabulary, access rules, evidence provenance rules,
execution profile, and reviewer-prompt delivery mechanism remain unchanged.

## Calibration And Blinding

V6 requires another fresh blind calibration because criterion vocabulary is a
substantive protocol change. Use the same stable 30 candidates and six new Terra/high
reviewer contexts. Do not supply v3-v5 ratings, disagreement discussion, or prior
decisions to reviewers. Each context receives only its exact rendered v6 prompt and
role stage, and every result passes the role-local validator before sealing.

Report exact status and criterion agreement. With one inclusion criterion, criterion
disagreement remains meaningful for boundary and exclusion coding but no longer acts
as a proxy for contribution taxonomy. Systematic ambiguity remains true when the same
eligibility, boundary, exclusion, access, or evidence-sufficiency rule causes
disagreement in two or more records, or when any record cannot map to one unique
status and permitted criterion without new substantive guidance.

Main screening is released only when all 60 ratings and bindings are valid, exact
status agreement is at least 0.80, and systematic ambiguity is false. A failed v6 gate
is sealed as `revise`; it does not authorize a main release.

## Validation And Success Criteria

Before v6 calibration execution:

- legacy v5 coordinator, result, and decision snapshots still validate unchanged;
- tests prove v6 bound rows accept only `include-relevant` for `included` status;
- tests prove historical bound rows retain the legacy four-value vocabulary;
- role-local validation derives the same vocabulary as authoritative phase sealing;
- the v6 coordinator validates and its calibration selection is byte-identical to v5;
- protocol and prompt bytes are frozen and checksum-complete; and
- focused screening tests pass without expanding the existing security model.

After execution, the calibration result and decision snapshots must pass their public
validators and checksum gates. Procedural limitations remain documented in the v6
README, and local staging artifacts remain outside the committed scientific record.
