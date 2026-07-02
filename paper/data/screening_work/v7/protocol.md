# Binary retention protocol v7

This protocol governs prospective Pass 1 retention for the course-generation survey.
It is a working, unfrozen v7 contract. The terms MUST, MUST NOT, REQUIRED, MAY, and
SHOULD are normative.

## Scope and final result vocabulary

Final v7 screening statuses are exactly `included` and `excluded`. `boundary` is
historical terminology only and MUST NOT be assigned as a v7 result, criterion, or
CSV value. Historical corpus records may still contain `boundary`; this protocol does
not rewrite them.

screening_inclusion_criterion is exactly [`include-relevant`].
The taxonomy field `screening_result_status` is exactly [`included`, `excluded`]. The
corpus field `screening_status` remains unchanged, including `candidate` and historical
`boundary`.

### Status and criterion pairing

| screening_status | Allowed criterion | exclusion_reason |
| --- | --- | --- |
| `included` | `include-relevant` | `NR` |
| `excluded` | Exactly one controlled exclusion criterion | A substantive, source-specific reason |

### Exclusion criteria

| Value | Normative meaning |
| --- | --- |
| `exclude-fixed-racing-line` | The source only plans, optimizes, predicts, or controls a racing line on a fixed course and provides no retained survey evidence. |
| `exclude-appearance-dynamics` | The source only varies appearance, sensing, vehicle parameters, disturbances, or dynamics without retained survey evidence. |
| `exclude-traffic-only` | The source only generates traffic participants, behavior, scenarios, or interactions on fixed roads without retained survey evidence. |
| `exclude-insufficient-detail` | The frozen evidence packet is insufficient to establish a retention condition. |
| `exclude-out-of-scope` | The source establishes none of the retention conditions. |

## Pass 1 retention decision

Pass 1 retains a source as `included`,`include-relevant` when the frozen evidence
packet establishes at least one of the following conditions:

1. **Core.** A direct generated- or parameterized-course method, representation,
   interface, dataset, benchmark, validity test, or metric.
2. **Supporting.** A fixed-course requirement, interface, benchmark property, dataset
   property, metric, simulator constraint, or reporting practice explicitly transferred
   into survey or benchmark design.
3. **Contextual.** A survey or systematic review establishing the field, terminology,
   or literature gap.

Otherwise, Pass 1 assigns `excluded` and exactly one controlled exclusion criterion.
The reviewer records the deciding packet evidence and locator, not a speculative future
use of the source.

### Fixed-route guardrail

Fixed CARLA routes or an equivalent fixed-route source are retained when they provide a
citable representation, benchmark format, simulator interface, or evaluation
requirement. They are supporting evidence and MUST NOT be called a generation method.
A fixed route without one of those citable contributions is excluded.

### Pass 1 limits

Pass 1 MUST NOT choose or rank a primary contribution. It MUST NOT perform full Pass 2
coding. Retained-source count is not a method count.

## Frozen evidence packets and duplicate review

Both duplicate reviewers MUST rate the same frozen evidence packet. The packet is the
sole eligibility-evidence basis for the two ratings and binds the source excerpts,
locators, versions, and provenance used for Pass 1.

Public retrieval during rating MAY verify metadata or report a packet defect but MUST
NOT silently replace or add eligibility evidence. A reviewer records a defect for the
coordinator rather than changing the eligibility basis. Stronger evidence discovered after freeze requires a new packet version. Any required rerating follows that version.

Reviewers MUST NOT receive v3-v6 ratings or disagreements. They work in six blind
reviewer contexts with no access to the other reviewer's rating or rationale.

## Pass 2 evidence coding and claim limits

Pass 2 is multi-label and separately assigns `survey_evidence_tier` as `core`,
`supporting`, or `contextual`. A source may receive more than one supported Pass 2
label; this does not convert the source count into a method count.

Core evidence may support direct generation-method, representation, interface,
dataset, benchmark, validity-test, or metric claims. Supporting evidence may support
survey or benchmark design claims only. Supporting evidence MUST NOT substantiate
generation-method claims. Contextual evidence may support field, terminology, or gap
claims only. Contextual evidence MUST NOT support implementation or performance claims.

## Reviewer result schema

The result CSV field order remains unchanged:

```csv
assignment_id,phase,candidate_id,input_sha256,snapshot_sha256,batch_id,coder_id,screened_on,screening_status,criterion,access_status,source_urls,evidence_version,evidence_retrieved_on,evidence_archive_url,evidence_sha256,screening_locator,exclusion_reason,notes
```

Every field MUST contain a value. The `screening_status`, `criterion`, and
`exclusion_reason` fields MUST follow the status and criterion pairing table. No v7
result may contain `boundary`.

## Calibration gate

Before any reviewer launch, the evidence inventory MUST be complete and the protocol
and evidence packet version MUST be frozen. V7 requires a fresh stable-30 calibration
with six blind reviewer contexts, agreement >= 0.80, no systematic ambiguity, and 60
valid ratings. Calibration is a gate: no main screening release may be created until it
passes under this v7 contract. Earlier ratings and disagreements are not calibration
inputs for v7.
