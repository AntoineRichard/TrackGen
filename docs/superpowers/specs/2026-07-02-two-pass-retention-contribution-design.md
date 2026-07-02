# Two-Pass Retention And Contribution Coding Design

**Status:** Approved concept; written specification pending user review

**Decision:** Separate eligibility retention from contribution characterization. The
first pass decides only whether a source is retained for the survey. The second pass
codes how a retained source contributes, using multiple labels and an explicit evidence
tier. Duplicate reviewers inspect the same frozen source artifacts while rating
independently.

## Context

V6 removed competing inclusion-subtype labels, but calibration still achieved only
21/30 exact status agreement. Five records, C0140, C0168, C0175, C0180, and C0198,
split between `included` and `boundary`. In each case, reviewers substantially agreed
about the source facts but disagreed about whether a fixed obstacle course, route set,
gate specification, or competition course was an inclusion contribution or boundary
material.

Three additional disagreements were affected by unequal source access. One reviewer
inspected a full report, rules document, or repository artifact while the other reached
only an abstract, blocked page, or incomplete file. Duplicate screening should compare
independent judgments of the same evidence, not independent success at document
retrieval.

The current workflow therefore conflates three questions:

1. Should this source remain in the survey corpus?
2. Is its relationship to course generation core, supporting, or contextual?
3. What technical contributions does it make?

V7 assigns these questions to separate stages and fields.

## Alternatives Considered

1. **Binary retention, downstream contribution coding, and shared evidence packets
   (selected).** This aligns screening with one decision, preserves fixed-course work
   without calling it a generation method, and ensures both reviewers inspect the same
   source versions.
2. **Binary retention with independent public retrieval.** This fixes the status model
   but preserves the observed 403, abstract-only, and version-asymmetry failures.
3. **Keep `boundary` as a secondary screening label.** This preserves familiar
   terminology but continues asking eligibility reviewers to decide contribution
   directness before evidence extraction.
4. **Proceed from failed v6 ratings through adjudication.** This violates the frozen
   calibration gate and would make the methods section indefensible.

## Pass 1: Retention Screening

V7 screening has two final statuses:

| Status | Meaning |
| --- | --- |
| `included` | Retained because material evidence supports at least one core, supporting, or contextual role in the survey. |
| `excluded` | Not retained because no qualifying role is established or the frozen evidence is insufficient to support a survey claim. |

`included` is the machine-readable status for what the paper describes as retained.
Every included row uses criterion `include-relevant`. Existing controlled exclusion
criteria remain available for excluded rows.

`boundary` is retired as a v7 screening status. Historical v1-v6 snapshots remain
immutable and continue to validate with their frozen vocabulary. V7 coordinator and
role-stage bindings structurally reject new `boundary` results.

A source is retained when it establishes at least one of these relationships:

- **Core:** directly contributes a method, representation, interface, dataset,
  benchmark, validity test, or metric for generated or parameterized courses.
- **Supporting:** provides a fixed-course requirement, interface, benchmark property,
  dataset property, metric, simulator constraint, or reporting practice explicitly
  transferred into this survey or its benchmark design.
- **Contextual:** is a survey or systematic review required to establish the field,
  terminology, or literature gap.

These relationships are not mutually exclusive contribution buckets during screening.
If any relationship is materially supported, the source is retained. Screening notes
identify an adjacent-domain mapping when required but MUST NOT perform full
contribution coding.

### Fixed-Route Example

A paper or standard that publishes fixed CARLA routes is retained when those routes
provide a citable course representation, benchmark format, simulator interface, or
evaluation requirement. It is not treated as a course-generation method merely because
it is retained. In Pass 2 it receives a supporting evidence tier and fixed-course or
benchmark contribution labels.

## Pass 2: Contribution Coding

Contribution coding starts only after screening and adjudication. It is multi-label:
a retained source may contribute a generator, representation, simulator interface,
benchmark, metric, and serialization format simultaneously.

The existing evidence fields continue to record technical contribution dimensions:

- `course_object`;
- `representation_family`;
- `generator_family`;
- `generation_role`;
- `validity_strategy`;
- geometry, difficulty, and diversity metrics;
- training and evaluation distributions;
- simulator and export format; and
- code, asset, and reproducibility status.

Add one scalar `survey_evidence_tier` field to `evidence.csv` and the taxonomy:

| Value | Meaning |
| --- | --- |
| `core` | The source directly supports at least one generated- or parameterized-course technical claim. |
| `supporting` | The source supports transferred fixed-course, benchmark, interface, metric, simulator, dataset, or reporting requirements but no core generation claim. |
| `contextual` | The source supports field structure, terminology, or survey-gap claims but no core or supporting technical claim. |

The tier records the source's strongest relationship to the survey. It does not imply
that every claim extracted from a core source is itself a core generation claim.
Individual claims remain constrained by their exact locators and evidence status.

Existing multi-valued evidence fields capture where the source contributes. For a
supporting fixed-route source, `generation_role=boundary_case` or `benchmark_only`,
together with representation, simulator, evaluation, and coding notes, preserves the
technical contribution without promoting it to a generation method.

## Claim-Synthesis Guardrails

Evidence tier constrains how a source may be used in the paper:

- Core evidence may support claims about generation, representation, validation, or
  generated-course evaluation, limited to the located facts.
- Supporting evidence may motivate requirements, interfaces, benchmark formats,
  simulator feasibility, metrics, or evaluation practice. It MUST NOT be counted as a
  course-generation method.
- Contextual evidence may support field organization, terminology, and gap claims. It
  MUST NOT support implementation or performance claims.

Tables and quantitative summaries MUST distinguish core methods from supporting and
contextual sources. A retained count is not a method count.

## Shared Frozen Evidence Packets

Both reviewers assigned to a candidate inspect the same frozen evidence packet. Their
judgments remain independent, but retrieval success is removed as an uncontrolled
variable.

Each candidate evidence manifest records one or more artifacts with these fields:

```text
candidate_id,artifact_id,artifact_role,source_url,evidence_version,
evidence_retrieved_on,access_status,evidence_archive_url,evidence_sha256,
local_filename,redistribution_status,retrieval_notes
```

The packet may contain a primary report and authoritative companion artifacts such as
official documentation, a repository revision, a standard, rules, or dataset metadata.
Every artifact has a stable ID, version, retrieval date, and SHA-256 when local bytes
exist.

Copyrighted or otherwise nonredistributable bytes remain local and untracked. Their
manifest rows and hashes are committed. Publicly redistributable artifacts may be
committed only when licensing permits. Role staging copies the exact locally verified
artifacts needed by that role into its untracked stage and binds their hashes in the
stage manifest.

Before a phase release:

- every assigned candidate has a complete evidence manifest;
- every referenced local artifact exists and matches its hash;
- both assignments for one candidate resolve to the same artifact IDs and hashes; and
- missing full text is represented by the same frozen best-available evidence and the
  same documented access limitation for both reviewers.

During rating, reviewers use packet evidence for eligibility facts. Public retrieval
may verify metadata or report a packet defect, but it MUST NOT silently replace or add
eligibility evidence for one reviewer. A materially stronger source discovered after
freeze invalidates the affected phase packet and requires a new version rather than an
in-place rating change.

## Version-Aware Validation

V7 taxonomy adds a result-only status vocabulary alongside the existing corpus status
and inclusion criterion:

```json
{
  "screening_result_status": ["included", "excluded"],
  "screening_inclusion_criterion": ["include-relevant"]
}
```

The existing `screening_status` taxonomy remains unchanged because it also governs the
pre-screening `candidate` value in the corpus. Role-stage execution configuration
version 3 derives and binds:

```json
{
  "allowed_screening_statuses": ["included", "excluded"],
  "allowed_inclusion_criteria": ["include-relevant"],
  "evidence_packet_manifest_sha256": "<lowercase SHA-256>"
}
```

New freezes require both taxonomy values. V7 role-stage configuration binds them
together with the exact candidate evidence-packet manifest hash. Role-local and
authoritative phase validation reject `boundary` under v7.

Historical snapshots and stage configurations without the v7 status contract retain
their historical allowed statuses. Compatibility is determined by frozen fields and
configuration version, never by directory names.

## Workflow And Data Flow

1. Assemble and audit candidate evidence manifests and local artifacts.
2. Freeze a v7 coordinator with binary result statuses and the stable 30 calibration
   selection.
3. Publish calibration packets containing the same candidate evidence for both
   assignments.
4. Collect two independent retention decisions per candidate.
5. Seal calibration, compute agreement, and inspect every disagreement.
6. Release main screening only after exact status agreement is at least 0.80,
   systematic ambiguity is false, and all evidence bindings validate.
7. Adjudicate retained/excluded disagreements from sealed ratings.
8. Perform separate multi-label contribution extraction for retained sources.
9. Calibrate and report reliability for `survey_evidence_tier` and the existing
   controlled evidence fields independently of screening agreement.

## Calibration And Historical Data

V7 is a substantive protocol and evidence-input revision. It requires six fresh blind
reviewer contexts and fresh ratings of the unchanged stable 30. V3-v6 ratings and
disagreement discussions are not supplied to reviewers.

V6 ratings are not recoded into v7. They remain immutable evidence that motivated the
design change. No v7 main packet is released from the failed v6 gate.

## Implementation Decomposition

Implementation is split into two reviewed plans:

1. **Retention and contribution contract:** version-aware binary statuses, protocol,
   `survey_evidence_tier`, validators, tests, and v7 coordinator inputs.
2. **Shared evidence packets and execution:** evidence manifest schema, local artifact
   verification, role-stage binding, calibration packet assembly, and fresh execution.

The first plan can be implemented and tested without source retrieval. The second plan
must complete the stable-30 evidence packets before any calibration reviewer is
launched.

## Success Criteria

Before v7 calibration:

- v1-v6 coordinators, results, and decisions still validate unchanged;
- v7 bound results accept only `included` and `excluded`;
- included rows accept only `include-relevant`;
- every stable-30 candidate has one validated shared evidence packet;
- duplicate assignments bind identical candidate artifact IDs and hashes;
- protocol tests prove that fixed-course supporting sources are retained but cannot
  support generation-method claims; and
- focused producer, result, protocol, and evidence-packet tests pass.

The calibration gate remains unchanged: main screening requires at least 0.80 exact
status agreement, no systematic ambiguity, 60 valid ratings, and complete bindings.
