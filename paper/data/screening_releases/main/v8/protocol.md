# Duplicate full-text screening codebook v8

This codebook prospectively governs binary Pass 1 retention for the course-generation
survey. Once frozen, it applies to the complete coordinator-bound population of 202
candidate reports. Each report receives two independent, blind ratings of the same
frozen evidence packet under one final protocol and coordinator snapshot version.
Calibration, main screening, adjudication, work-level deduplication, Pass 2 coding, and
author verification are separate controlled stages.

The terms MUST, MUST NOT, REQUIRED, MAY, and SHOULD are normative. A result that
violates a MUST, MUST NOT, or REQUIRED rule is invalid and cannot be integrated.

## Operational definitions

| Term | Operational definition |
| --- | --- |
| `course` | A course MUST be an ordered or topologically connected spatial navigation constraint that defines where a racing robot may or must travel, including a road or track corridor, centerline with width, closed loop, open route, waypoint or gate sequence, buoy course, channel, or waterway. Geometry, topology, boundaries, control points, gates, buoys, or equivalent constraints MUST materially determine traversal; appearance-only variation is not a course. |
| `racing robot` | A racing robot MUST be a physical robot or a physics- or kinematics-based simulated vehicle whose objective includes fast, valid traversal of a course under control, safety, collision, or rule constraints. This includes aerial drones, ground cars, and surface or marine craft; a purely visual game object without vehicle state or control is insufficient. |
| `transferable adjacent domain` | A transferable adjacent domain MUST address spatial route, road, map, network, level, or test-input generation outside robot racing and expose a representation, generator, constraint, metric, feasibility test, interface, or benchmark that maps to robot courses without changing the contribution's essential technical claim. The reviewer MUST name that mapping in `notes`. |
| `source-native contribution` | A source-native contribution MUST be claimed, defined, implemented, evaluated, or released by the source itself for generated or parameterized courses or for a transferable adjacent domain. Intended reuse invented only by this survey is not source-native. |
| `supporting transfer` | A supporting transfer MUST be made by this survey protocol or its accountable authors from a fixed-course property directly established by frozen packet evidence. The property MUST belong to the closed supporting list and the reviewer MUST record its concrete protocol-level mapping in `notes`. The source need not claim the transfer, and the transfer MUST NOT imply that the source contributes a course-generation method. |
| `material evidence` | Material evidence MUST be content captured in the frozen evidence packet from the report being screened or its authoritative companion artifact, specific enough to decide retention and support the resulting survey claim. Primary reports or authoritative artifacts are required for technical contributions; a directly inspected survey or systematic review is permitted as secondary evidence only for field, terminology, or literature-gap context. Evidence MUST identify the source-specific fact, a precise locator, access mode, version, retrieval date, and required provenance. Topic similarity, a title, or an unsupported abstract inference is not material evidence. |
| `report` | A report MUST be one identifiable dissemination object, such as an article, preprint version, thesis, technical report, standard release, dataset release, software release, repository revision, or official competition document. A candidate record represents one report. |
| `work` | A work MUST be the underlying intellectual or technical contribution that one or more reports communicate. Companion papers, preprints, final articles, documentation, repositories, datasets, and releases may belong to one work and MUST be linked before synthesis. |

## Eligibility dimensions and synthesis unit

No language restriction applies. Sources in any language are eligible, but a rating
MUST rely on text the reviewer can verify directly or through a documented translation.
The original text, translated passage or locator, translation method, and verifier MUST
be retained in the audit trail. Machine translation alone does not remove the authors'
verification obligation.

No publication-date restriction applies. The search and screening dates bound the
review operationally, but reports from any year are eligible. Living artifacts MUST be
bound to a release, commit, document revision, archived capture, or retrieval date.

No publication-type restriction applies. Peer-reviewed articles, conference and
workshop papers, preprints, theses, books or chapters, technical reports, standards,
datasets, benchmarks, competition materials, software releases, repositories, and
official documentation may qualify. Publication type never substitutes for material
evidence. Informal pages, promotional material, videos, or social posts qualify only
when they are the authoritative primary report for an artifact and meet all evidence
and provenance rules.

The screening unit is the report; the synthesis unit is the work. Exact duplicate
reports MUST be collapsed before assignment. Exact duplicates are byte-identical
artifacts or records proven to identify the same immutable report; the coordinator
retains all identifiers and URLs in an alias record while assigning one candidate.
Versioned, companion, or overlapping reports MUST remain linked until full-text
screening establishes whether each contains unique material evidence. They are not
silently discarded as duplicates.

Each assigned report receives its own eligibility ratings. After screening, reports
about the same work MUST be grouped by stable identifiers, authorship, title,
artifact identity, and contribution overlap. The most complete citable report becomes
the canonical synthesis report; companion reports remain linked when they supply
unique material evidence. Conflicting versions MUST be described rather than merged
silently. A work MUST be counted at most once in quantitative synthesis. Every report used as
evidence remains citable and auditable.

## Independence and blinding

The two ratings for a candidate MUST be produced by distinct reviewers in distinct
assigned batches. Both duplicate reviewers MUST rate the same frozen evidence packet.
The packet bytes and digest, including every eligibility excerpt, locator, version, and
provenance field, MUST be identical for the pair. Before both ratings are locked,
neither reviewer may see the other reviewer's status, criterion, evidence
interpretation, exclusion reason, notes, work in progress, or execution trace.
Reviewer-facing packets MUST omit prior screening decisions, exclusion reasons,
unresolved-conflict flags, citation keys, and any other field that reveals a prior
eligibility decision.

Public retrieval during rating MAY verify metadata or report a packet defect but MUST
NOT silently replace or add eligibility evidence. The reviewer records the defect for
the coordinator and completes no affected rating under altered evidence. Stronger
evidence after freeze requires a new packet version. Any affected duplicate ratings
require rerating under that version before integration.

A locked rating is an immutable result file whose exact bytes and SHA-256 digest have
been recorded in a sealed primary-result snapshot. Discussion, reconciliation, and
adjudication occur only after both ratings for the relevant phase are locked. Raw
ratings remain immutable; later decisions are append-only records.

Independence is procedural, not merely nominal. Distinct `coder_id` values are
necessary but insufficient. Human reviewers MUST work without access to one another's
results. Automated-review independence is governed by the separate automation section
and requires isolated execution contexts using the same frozen evidence packet.

## Reviewer result schema

Each reviewer result file MUST use this exact header and field order:

```csv
assignment_id,phase,candidate_id,input_sha256,snapshot_sha256,batch_id,coder_id,screened_on,screening_status,criterion,access_status,source_urls,evidence_version,evidence_retrieved_on,evidence_archive_url,evidence_sha256,screening_locator,exclusion_reason,notes
```

Every field MUST contain a value; blank fields are invalid.

| Field | Normative rule |
| --- | --- |
| `assignment_id` | Exact assignment identifier from the frozen manifest. |
| `phase` | Exactly `calibration` or `main`, as assigned by the frozen manifest. |
| `candidate_id` | Exact candidate identifier from the assigned packet. |
| `input_sha256` | Exact lowercase 64-hex candidate-input digest from the frozen manifest. |
| `snapshot_sha256` | Exact lowercase 64-hex digest of the immutable coordinator screening snapshot. |
| `batch_id` | Exact assigned batch identifier binding the assignment to one reviewer packet. |
| `coder_id` | Stable reviewer-role identifier from the assignment and execution registers; it MUST NOT impersonate a person. |
| `screened_on` | ISO 8601 calendar date in `YYYY-MM-DD` form. |
| `screening_status` | Exactly one controlled screening status. |
| `criterion` | Exactly one criterion permitted for the chosen status. |
| `access_status` | Exactly one controlled access status. |
| `source_urls` | Canonical semicolon-separated absolute HTTP(S) source URLs for all deciding evidence. |
| `evidence_version` | Non-`NR` identifier for the inspected edition, version of record, release, commit, revision, document date, or explicit `unversioned@YYYY-MM-DD` state. |
| `evidence_retrieved_on` | ISO 8601 date in `YYYY-MM-DD` form on which the deciding evidence was retrieved. |
| `evidence_archive_url` | Canonical absolute HTTP(S) URL pinned to the inspected version or exactly `NR` under the evidence rules. |
| `evidence_sha256` | Lowercase 64-hex SHA-256 of the exact inspected artifact bytes or exactly `NR` under the evidence rules. |
| `screening_locator` | Precise page, section, table, figure, algorithm, appendix, repository path and lines, or stable official-documentation anchor. |
| `exclusion_reason` | `NR` for `included`; a substantive source-specific reason for `excluded`. |
| `notes` | Concise supplementary information, including an adjacent-domain mapping and any required concrete supporting protocol-level mapping, or exactly `NR`. |

`screened_on` MUST be an ISO 8601 calendar date in `YYYY-MM-DD` form.
`source_urls` MUST contain canonical, semicolon-separated absolute HTTP(S) URLs.
`screening_locator` MUST identify a page, section, table, figure, algorithm,
appendix, or stable official-documentation anchor. `evidence_version` and
`evidence_retrieved_on` are REQUIRED for every rating. `notes` MAY be
`NR`; all uses of `NR` are closed by the access and evidence rules.

## Controlled vocabularies

These tables are closed vocabularies. Reviewers and integration tools MUST reject any
unlisted value, spelling variant, capitalization variant, or legacy result code.

### Screening statuses

Final prospective v8 screening statuses are exactly `included` and `excluded`.
`boundary` is historical terminology only and MUST NOT be assigned as a v8 result,
criterion, or CSV value. Historical corpus `screening_status` values remain unchanged
for provenance and may include `candidate` and `boundary`; they are not valid reviewer
results under this protocol.

| Value | Normative meaning |
| --- | --- |
| `included` | Frozen packet evidence establishes at least one core, supporting, or contextual retention condition. |
| `excluded` | Frozen packet evidence establishes no retention condition, or is insufficient to support a survey claim. |

### Inclusion criteria

`screening_inclusion_criterion` is exactly [`include-relevant`]. The three retention
conditions below are disjunctive evidence-presence checks, not competing criterion
values.

| Value | Normative meaning |
| --- | --- |
| `include-relevant` | Frozen packet evidence establishes at least one core, supporting, or contextual retention condition. |

Pass 1 retains a source when frozen packet evidence establishes at least one:

1. **Core.** A direct generated- or parameterized-course method, representation,
   interface, dataset, benchmark, validity test, or metric.
2. **Supporting.** A fixed-course requirement, interface, benchmark property, dataset
   property, metric, simulator constraint, or reporting practice explicitly transferred
   by this survey protocol or its accountable authors into survey or benchmark design.
3. **Contextual.** A survey or systematic review establishing the field, terminology,
   or literature gap.

A source satisfying several conditions remains one `included`,`include-relevant`
result. Pass 1 MUST NOT choose or rank a primary contribution and MUST NOT perform full
Pass 2 coding. Retained-source count is not method count. The screening `notes` field
MUST NOT substitute for downstream evidence extraction.

### Eligibility and supporting-transfer clarification

A directly inspected source-native script, implementation, algorithm, or specification
captured in the frozen packet that emits, samples, places, connects, or serializes
explicit course coordinates, corridors, routes, waypoints, gates, or buoys satisfies
the core retention condition, even when a downstream benchmark later uses one generated
realization as a fixed course. A source that normatively defines a reusable course
representation, interchange format, design interface, simulator course interface,
dataset, or competition course set for generated or parameterized courses also
satisfies the core condition.

The survey protocol or accountable authors make the supporting transfer; the source
need not claim that transfer. Frozen packet evidence MUST directly establish at least
one property in the closed supporting list: fixed-course requirement, interface,
benchmark property, dataset property, metric, simulator constraint, or reporting
practice. For an included source retained on supporting evidence, the reviewer MUST
record the concrete protocol-level mapping in `notes`. Speculative future reuse is
insufficient. This is a binary evidence-presence check, not Pass 2 ranking or coding.

Fixed CARLA routes or equivalent fixed-route sources are retained when frozen packet
evidence provides a citable representation, benchmark format, simulator interface, or
evaluation requirement that the survey maps into its design. They are supporting
evidence and MUST NOT be called generation methods. Merely loading, controlling on,
evaluating on, or describing a fixed course does not satisfy retention.

A generic laboratory, project, promotional, or topic page does not become eligible
from broad relevance or possible future reuse. Absent material evidence for a core,
supporting, or contextual retention condition, the source is excluded.

### Exclusion criteria

| Value | Normative meaning |
| --- | --- |
| `exclude-fixed-racing-line` | Only plans, optimizes, predicts, or controls a racing line in a fixed corridor and supplies no retained supporting property. |
| `exclude-appearance-dynamics` | Only randomizes appearance, sensing, vehicle parameters, disturbances, or dynamics without a core, supporting, or contextual retention condition. |
| `exclude-traffic-only` | Only generates traffic participants, behavior, scenarios, or interactions on fixed roads without a retained course property. |
| `exclude-insufficient-detail` | Mentions a course, road, track, gate sequence, generator, benchmark, or metric but frozen packet evidence is too incomplete to support a survey claim. |
| `exclude-out-of-scope` | Has no material evidence for a core, supporting, or contextual retention condition. |

### Access statuses

| Value | Normative meaning |
| --- | --- |
| `full_text` | The complete primary report was inspected during packet assembly. |
| `full_text_and_supplement` | The complete primary report and relevant supplementary material or repository documentation were inspected during packet assembly. |
| `official_documentation` | Official, version-identifiable documentation was inspected during packet assembly for a software, dataset, benchmark, standard, or competition artifact. |
| `abstract_only` | Exhaustive packet-assembly retrieval failed and only an abstract or similarly incomplete bibliographic record was available. |

## Pass 2 evidence coding and claim limits

Pass 2 is multi-label and separately assigns `survey_evidence_tier` as `core`,
`supporting`, or `contextual`. A retained source may receive more than one supported
contribution label. Retained-source count is not method count.

Core evidence may support direct method, representation, interface, dataset, benchmark,
validity-test, or metric claims. Supporting evidence may support survey or benchmark
design claims only. Supporting evidence MUST NOT substantiate generation-method claims.
Contextual evidence may support field, terminology, or literature-gap claims only.
Contextual evidence MUST NOT support implementation or performance claims.
## Normative decision procedure

Retention has precedence over exclusion. Frozen packet evidence MUST establish at least
one core, supporting, or contextual retention condition. Core evidence is source-native;
supporting transfer is made by the survey protocol or accountable authors from a
directly established property in the closed supporting list; contextual evidence is a
survey or systematic review establishing field, terminology, or literature gap.

`boundary` MUST NOT be assigned as a v8 result or criterion. Historical boundary
ratings and disagreements are not reviewer inputs. Intended reuse without a directly
established packet property is insufficient.

1. **Establish access.** Verify the frozen packet binding and use its recorded access
   status, canonical source URLs, version provenance, evidence digest or archive, and
   precise locators. Do not add or replace eligibility evidence during rating.
2. **Apply retention.** Determine whether frozen packet evidence establishes at least
   one core, supporting, or contextual retention condition. If it does, assign
   `included`,`include-relevant`. Record every required supporting protocol-level
   mapping in `notes`. Do not choose or rank a contribution or perform Pass 2 coding.
3. **Apply exclusion.** If no retention condition applies, assign `excluded` with
   exactly one controlled primary exclusion criterion and a source-specific reason.
   Choose the criterion that most directly explains why the frozen packet cannot
   support this survey; do not use a generic reason when a specific rule applies.
4. **Validate the result.** Confirm binary status-criterion pairing, access sufficiency,
   source and provenance fields, locator precision, supporting mapping or exclusion
   reason requirements, assignment binding, and absence of blank fields before locking
   the rating.

### Status and criterion pairing

| screening_status | Allowed criterion | exclusion_reason |
| --- | --- | --- |
| `included` | `include-relevant` | `NR` |
| `excluded` | Exactly one controlled exclusion criterion | A substantive, source-specific reason |
## Access and evidence rules

Packet assembly and audit occur before reviewer launch. Before assigning
`abstract_only` in the evidence inventory, the coordinator MUST make and document an
exhaustive retrieval attempt. At minimum, that attempt covers the DOI or publisher
record, an exact title-and-author search, a recognized scholarly index or repository,
and an official project, dataset, software, competition, or author page when one
exists. Attempted locations and outcomes belong in the packet inventory;
inaccessible or unsafe URLs MUST NOT be invented.

During rating, the frozen packet is the sole eligibility-evidence basis. Public
retrieval may verify metadata or report a packet defect but cannot change the rating
basis. A defect or stronger evidence requires a new packet version and rerating as
specified by the independence rules.

`included` MUST NOT use `abstract_only`; it requires `full_text`,
`full_text_and_supplement`, or `official_documentation`. If only an abstract
remains available and it cannot support a survey claim, the result MUST be `excluded`
with criterion `exclude-insufficient-detail`. An abstract may justify another
exclusion code only when it directly and unambiguously establishes that code.

Each `source_urls` value MUST contain one or more absolute `http://` or
`https://` URLs. Canonicalization removes surrounding whitespace, lowercases the
URI scheme and host, removes fragments that are not evidence locators, removes
duplicates, sorts URLs by UTF-8 byte representation, and joins them with one semicolon
and no surrounding spaces. A documentation anchor used as evidence is retained and
repeated in `screening_locator`.

Each locator MUST identify where deciding material evidence appears: a page or page
range, named or numbered section, table, figure, algorithm, appendix, repository path
and lines at a pinned revision, or stable official-documentation anchor. A title, bare
URL, search query, or the word "abstract" alone is invalid. Multiple locators are
separated by semicolons.

`evidence_version` and `evidence_retrieved_on` are always required and MUST
NOT be `NR`. For an immutable scholarly report, a version-of-record designation,
DOI version, repository version, or dated report revision is valid. For a source that
publishes no version identifier, record `unversioned@YYYY-MM-DD` using the packet
retrieval date; this declares the limitation rather than inventing a version.

An `included` rating based on `official_documentation` MUST record either a
version-pinned `evidence_archive_url` or a lowercase 64-hex `evidence_sha256`.
Recording both is preferred. A branch URL, mutable latest-documentation URL, or unpinned
repository page is not version-pinned. `evidence_archive_url` MAY be `NR` only when no
version-pinned archive was inspected. `evidence_sha256` MAY be `NR` only when the exact
inspected bytes were not lawfully or technically obtainable. If both are `NR`, official
documentation cannot support `included`.

For `excluded`, `exclusion_reason` MUST state the source-specific fact that activates
the selected exclusion criterion; merely repeating the criterion label is invalid. For
`included`, `exclusion_reason` MUST be exactly `NR`. `notes` MAY be `NR` except that a
supporting retention condition requires the concrete protocol-level mapping. The
mapping requirement does not change the controlled inclusion criterion.
`resolved_conflict_ids` MAY be `NR` only in an adjudication with no conflict to resolve.
Only `evidence_archive_url`, `evidence_sha256`, `exclusion_reason`, `notes`, and
`resolved_conflict_ids` may contain `NR` under their field-specific rules. No field may
be blank.
## Calibration and release gate

Calibration is a mandatory prospective workflow phase and release gate. It is not a
post hoc subgroup or an analysis label.

Before any v8 coordinator freeze or reviewer launch, the evidence inventory MUST be
complete and each duplicate pair's common evidence packet MUST be versioned and
digest-bound. V8 requires a fresh stable-30 calibration conducted in six blind reviewer
contexts and producing 60 valid ratings. Release requires agreement >= 0.80 and no
systematic ambiguity. V8 reviewers MUST NOT receive v3-v6 ratings or disagreements.
No v8 main release may be created until this gate passes.
### Calibration gate requirements

| Property | Required value |
| --- | --- |
| Calibration records | 30 |
| Main records | 172 |
| Final-version records | 202 |
| Locked blind ratings per record | 2 |
| Minimum calibration exact status agreement | >= 0.80 |
| Systematic-ambiguity tolerance | None |

### Deterministic calibration selection

The calibration sample is selected from frozen bibliographic and discovery metadata
before reviewer assignment. Only `candidate_id`, `source_type`,
`discovery_stream`, and `discovery_query` may influence selection. Citation
key, prior screening status, exclusion reason, conflict state, and metadata evidence
MUST NOT influence selection. Reviewer-facing packets still omit discovery fields.

The literal salt is `trackgen-screening-calibration-v1`. It is a stable
candidate-ID salt and MUST NOT depend on the protocol hash or protocol version. For
ranking, compute `SHA-256(salt + NUL + candidate_id)` over UTF-8 bytes and order by
digest bytes, then by UTF-8 `candidate_id` bytes. A protocol revision alone
therefore cannot change the sample.

Normalize `source_type` with Unicode NFKC, casefold it, replace each maximal
nonalphanumeric run with one ASCII space, collapse whitespace, and trim it. Assign
each candidate to the first matching coarse stratum in this priority table;
`official-other` is the mandatory default.

| Priority | Coarse source-type stratum | Matching tokens |
| --- | --- | --- |
| 1 | `standard-specification` | `standard`, `specification`, or `file format` |
| 2 | `competition` | `competition` |
| 3 | `benchmark-dataset` | `benchmark` or `dataset` |
| 4 | `software` | `software`, `repository`, `simulator`, `platform`, `package`, `game`, `engine`, or `tool` |
| 5 | `scholarly` | `article`, `paper`, `preprint`, `chapter`, `thesis`, `report`, or `survey` |
| 6 | `official-other` | No earlier stratum matches |

For every populated stratum, allocate an initial quota of
`min(2, stratum size)`. Allocate the remaining places in proportion to each
stratum's residual capacity. Use integer quotients first, then assign any unallocated
places by descending remainder; ties follow priority-table order. The quotas MUST sum
to 30 and MUST NOT exceed any stratum size.

Split both `discovery_stream` and `discovery_query` on semicolons. Normalize every
token independently with Unicode NFKC, casefold it, replace each maximal
nonalphanumeric run with one ASCII space, collapse whitespace, trim it, discard empty
tokens, and deduplicate exact normalized tokens. Sort normalized tokens by UTF-8 bytes,
prefix them with `stream:` or `query:`, and take their set as the candidate's
discovery labels.

Within each stratum, fill its quota greedily. At each step choose the remaining
candidate that adds the largest number of discovery labels not yet represented in that
stratum. Break ties by the stable digest ordering and then UTF-8 `candidate_id` bytes.
The union of stratum selections MUST contain exactly 30 candidates; their canonical
release order is the same stable digest ordering. Any corpus other than exactly 202
unique candidate IDs is a hard error. Inability to select exactly 30 is also a hard
error.

The selected candidate-ID list and its order are frozen separately from the protocol.
`calibration_selection.csv` MUST be a coordinator-root CSV with exactly the
one-column header `candidate_id` and exactly 30 data rows.

```csv
candidate_id
```

Its row order MUST be SHA-256(salt + NUL + candidate_id), then UTF-8 candidate_id; the
file is covered by `SHA256SUMS`. The 30 candidate IDs in
`calibration_selection.csv` MUST equal exactly the candidate IDs whose two manifest
rows have phase `calibration`. Validation MUST rederive the selection from the frozen
metadata and reject any ID or order difference.

A substantive protocol revision MUST reuse the same stable 30 unless bibliographic or
discovery metadata itself is formally corrected; such a correction requires a new
selection-version record explaining the changed IDs and cannot be disguised as a
protocol revision.

### Calibration decision schema

Each calibration gate decision MUST use this exact header and field order:

```csv
decision_id,protocol_sha256,coordinator_snapshot_sha256,calibration_result_snapshot_sha256,candidate_ids_sha256,assignment_ids_sha256,status_agreement_numerator,status_agreement_denominator,status_agreement,systematic_ambiguity,decision,decided_on,decision_makers,resolution_evidence
```

| Field | Normative rule |
| --- | --- |
| `decision_id` | Unique immutable gate-decision identifier. |
| `protocol_sha256` | Lowercase 64-hex SHA-256 of the exact protocol bytes used for all 60 ratings. |
| `coordinator_snapshot_sha256` | Lowercase 64-hex digest of the coordinator snapshot that assigned the stable 30. |
| `calibration_result_snapshot_sha256` | Lowercase 64-hex digest of the sealed snapshot containing exactly 60 calibration ratings. |
| `candidate_ids_sha256` | SHA-256 of the 30 ordered candidate IDs, one UTF-8 ID per LF-terminated line. |
| `assignment_ids_sha256` | SHA-256 of the 60 assignment IDs sorted by UTF-8 bytes, one per LF-terminated line. |
| `status_agreement_numerator` | Integer count of calibration candidates with identical raw statuses. |
| `status_agreement_denominator` | Exactly `30`. |
| `status_agreement` | Numerator divided by 30, rendered as a six-decimal value from `0.000000` through `1.000000`. |
| `systematic_ambiguity` | Exactly `true` or `false` under the ambiguity rule. |
| `decision` | Exactly `release` or `revise` under the release rule. |
| `decided_on` | ISO 8601 calendar date in `YYYY-MM-DD` form. |
| `decision_makers` | Semicolon-separated distinct stable IDs sorted by UTF-8 bytes; the exact stable role `accountable-author` is required. |
| `resolution_evidence` | Substantive append-only account of disagreements, ambiguity assessment, changes required, and release rationale. |

The candidate and assignment digest preimages MUST be retained with the decision. The
decision is invalid if any hash, count, identity, protocol version, or phase binding
does not match the sealed snapshots.

### Calibration decision snapshot artifacts

A sealed calibration decision snapshot MUST use this exact one-row manifest header and
field order:

```csv
manifest_version,calibration_decision_snapshot_sha256,protocol_sha256,coordinator_snapshot_sha256,calibration_result_snapshot_sha256,decision_id,decision_file_sha256,candidate_ids_file_sha256,assignment_ids_file_sha256,row_count
```

| Field | Normative rule |
| --- | --- |
| `manifest_version` | Exactly `1`. |
| `calibration_decision_snapshot_sha256` | Lowercase 64-hex SHA-256 binding the decision, both identifier preimages, protocol, coordinator, calibration result, decision identity, manifest version, and row count. |
| `protocol_sha256` | Lowercase 64-hex SHA-256 of the exact frozen protocol bytes. |
| `coordinator_snapshot_sha256` | Lowercase 64-hex digest of the authoritative coordinator snapshot. |
| `calibration_result_snapshot_sha256` | Lowercase 64-hex digest of the authoritative 60-rating calibration result snapshot. |
| `decision_id` | Exact `decision_id` from `decision.csv`. |
| `decision_file_sha256` | Lowercase 64-hex SHA-256 of the exact `decision.csv` bytes. |
| `candidate_ids_file_sha256` | Lowercase 64-hex SHA-256 of the exact `candidate_ids.txt` bytes. |
| `assignment_ids_file_sha256` | Lowercase 64-hex SHA-256 of the exact `assignment_ids.txt` bytes. |
| `row_count` | Exactly `1`. |

The exact snapshot file set is `decision.csv`, `candidate_ids.txt`,
`assignment_ids.txt`, `manifest.csv`, and `SHA256SUMS`; no other entry
is allowed. `candidate_ids.txt` MUST be exactly the 30 frozen coordinator
calibration IDs in frozen sequence, one UTF-8 ID per LF-terminated line.
`assignment_ids.txt` MUST be exactly the 60 calibration assignment IDs sorted
by UTF-8 bytes, one ID per LF-terminated line. Validation MUST compare both
preimage files byte-for-byte with the authoritative coordinator and calibration
result snapshots.

The decision snapshot hash MUST be SHA-256 of the canonical UTF-8 JSON object
containing exactly `assignment_ids_file_sha256`,
`calibration_result_snapshot_sha256`, `candidate_ids_file_sha256`,
`coordinator_snapshot_sha256`, `decision_file_sha256`, `decision_id`,
`manifest_version`, `protocol_sha256`, and numeric `row_count`. JSON keys
MUST be sorted, non-ASCII text MUST remain unescaped, and separators MUST be
comma and colon with no added whitespace. `SHA256SUMS` MUST cover the other
four files in lexical path order.

### Reviewer release manifest

Every calibration and main reviewer release MUST contain canonical
`release_manifest.csv` covered by `SHA256SUMS`. It MUST contain exactly one
row under this exact header and field order:

```csv
manifest_version,phase,coordinator_snapshot_sha256,protocol_sha256,assignment_count,calibration_result_snapshot_sha256,calibration_decision_snapshot_sha256
```

| Field | Normative rule |
| --- | --- |
| `manifest_version` | Exactly `1`. |
| `phase` | Exactly `calibration` or `main`, matching every released packet row. |
| `coordinator_snapshot_sha256` | Lowercase 64-hex digest of the captured coordinator snapshot, matching every released packet row. |
| `protocol_sha256` | Lowercase 64-hex SHA-256 of the exact released `protocol.md` bytes. |
| `assignment_count` | Exactly `60` for calibration or `344` for main. |
| `calibration_result_snapshot_sha256` | Exactly `NR` for calibration; for main, the lowercase 64-hex digest of the captured calibration result snapshot. |
| `calibration_decision_snapshot_sha256` | Exactly `NR` for calibration; for main, the lowercase 64-hex digest of the captured calibration decision snapshot. |

The exact release-root entry set MUST be `protocol.md`,
`release_manifest.csv`, `SHA256SUMS`, and `packets`. The `packets`
directory MUST contain exactly the six canonical reviewer packet filenames.
All files MUST be mode `0644`, both directories MUST be mode `0755`, and
`SHA256SUMS` MUST cover every other file in lexical path order.

For phase `calibration`, `assignment_count` MUST be 60 and both gate hash
fields MUST be exactly `NR`. For phase `main`, `assignment_count` MUST be
344 and the calibration result and calibration decision snapshot hashes MUST
equal the coherent tuple that authorized publication. The published release
manifest MUST be validated after publication against its captured authorization
inputs. Missing, extra, reordered, noncanonical, tampered, aliased, or
self-consistently rehashed artifacts MUST be rejected.

### Phase-result manifest and release binding

Every sealed calibration or main phase-result snapshot MUST contain exactly the six
canonical batch result files, `manifest.csv`, and `SHA256SUMS`; no other entry is
allowed. `manifest.csv` MUST contain exactly one row per batch under this exact
header and field order:

```csv
manifest_version,phase_result_snapshot_sha256,coordinator_snapshot_sha256,protocol_sha256,reviewer_release_sha256,phase,batch_id,coder_id,result_filename,result_file_sha256,row_count
```

| Field | Normative rule |
| --- | --- |
| `manifest_version` | Exactly `1`. |
| `phase_result_snapshot_sha256` | One lowercase 64-hex digest shared by all six rows and computed from the canonical phase-result binding below. |
| `coordinator_snapshot_sha256` | Exact authoritative coordinator digest shared by all released assignments and results. |
| `protocol_sha256` | Exact digest of the frozen protocol in the authoritative coordinator and reviewer release. |
| `reviewer_release_sha256` | Exact lowercase 64-hex digest of the complete authoritative reviewer release, including its protocol, release manifest, checksums, and all six packets. |
| `phase` | Exactly `calibration` or `main`, shared by all six rows. |
| `batch_id` | One canonical batch ID, appearing exactly once. |
| `coder_id` | Exactly the same canonical reviewer-role ID as `batch_id`. |
| `result_filename` | Exactly `<batch_id>.csv`. |
| `result_file_sha256` | Lowercase 64-hex SHA-256 of the exact canonical result-file bytes. |
| `row_count` | Exact positive decimal row count for that batch. |

The phase-result snapshot digest MUST be SHA-256 of the canonical UTF-8 JSON object
containing exactly `coordinator_snapshot_sha256`, `manifest_version`, `phase`,
`protocol_sha256`, `reviewer_release_sha256`, and `results`. `results` MUST
contain six objects in canonical batch order, each with exactly `batch_id`,
`coder_id`, `filename`, numeric-string `row_count`, and `sha256`. JSON keys
MUST be sorted, non-ASCII text MUST remain unescaped, and separators MUST be comma and
colon with no added whitespace.

Calibration results MUST bind the exact calibration reviewer release. Main results
MUST bind the exact gated main reviewer release and MUST additionally be validated
against the same calibration reviewer release, calibration result, and passing
calibration decision that authorized that main release. A result snapshot created
before, outside, or against a different release is invalid even when its assignments
could be reconstructed from the coordinator. Validation MUST compare each batch's
result assignment set exactly with its released packet assignment set and MUST reject
missing, extra, substituted, or cross-batch assignments.

### Authoritative phase-result validation

Public phase-result validation MUST always receive exactly one authoritative
coordinator anchor: an immutable coordinator snapshot path or an already captured
coordinator snapshot. The validator MUST replay every assignment, candidate,
`input_sha256`, batch, phase, protocol, snapshot, and reviewer-release binding against
that anchor.
Self-declared coordinator and protocol hashes in a phase-result manifest are integrity
metadata only and MUST NOT be accepted as provenance. A call that omits the anchor,
supplies more than one anchor, or relies only on hashes declared by the result snapshot
is invalid.

### Gate procedure

1. **Release calibration.** The stable 30-record calibration set MUST be released
   separately as phase `calibration`. Main-phase packets remain inaccessible.
2. **Lock blind ratings.** Each calibration record MUST receive two independent ratings
   from distinct reviewers. All 60 ratings MUST be locked in one immutable result
   snapshot before any is disclosed or discussed.
3. **Analyze agreement.** Compute exact status agreement, exact criterion agreement,
   the status contingency table, criterion disagreements, and recurring reasons from
   the 60 locked raw ratings. Exact status agreement is the number of identical status
   pairs divided by 30.
4. **Discuss disagreements.** Only after locking, reviewers and the coordinator examine
   disagreements for inconsistent rule application or ambiguous wording. Discussion
   MUST NOT alter raw ratings. Systematic ambiguity exists when the same operational
   definition or rule boundary causes disagreement in two or more records, or when any
   record cannot be mapped to one unique status and criterion without new substantive
   guidance.
5. **Record the gate decision.** Seal one decision using the exact calibration schema.
   If agreement is below `0.80`, systematic ambiguity is `true`, any rating
   is invalid, or any binding is incomplete, `decision` MUST be `revise`.
   Only when agreement is at least `0.80`, systematic ambiguity is `false`,
   all 60 ratings are valid, and every binding verifies, `decision` MUST be
   `release`.
6. **Revise when required.** Any substantive protocol revision MUST invalidate the
   calibration run, increment both the protocol version and snapshot version, and
   require fresh isolated reviewers to rerate the same stable 30 calibration records
   blindly. A revision is substantive if it changes scope, eligibility, precedence,
   vocabulary, evidence sufficiency, field validation, or criterion interpretation.
   No reviewer exposed to failed-calibration ratings or discussion may rerate the
   stable 30 after revision. A `revise` decision MUST NOT release any main-phase
   packet and MUST require a new protocol and coordinator snapshot.
7. **Release the main phase.** The main 172 records MUST NOT be released until an
   immutable calibration decision records both exact status agreement >= 0.80 and no
   systematic ambiguity. Reviewers from a passing calibration MAY continue only with
   previously unseen main-phase records. The release MUST bind the passing decision,
   protocol, coordinator snapshot, and reviewer packets.
8. **Complete final-version duplication.** Every one of the 202 records MUST ultimately
   have two locked ratings made under the same final protocol and snapshot version. If
   calibration caused a substantive revision, only ratings produced by fresh isolated
   reviewers under the final version count for the stable 30.

The calibration decision is append-only and content-addressed. Coordinator discretion,
a mutable pointer, an overwritten report, or verbal approval cannot waive or replace
the gate.

## Adjudication

Adjudication occurs only after the two final-version ratings are locked. The
adjudication file MUST use this exact header and field order:

```csv
candidate_id,input_sha256,snapshot_sha256,primary_snapshot_sha256,assignment_ids,adjudicator_id,reviewer_ids,decided_on,screening_status,criterion,access_status,source_urls,evidence_version,evidence_retrieved_on,evidence_archive_url,evidence_sha256,screening_locator,exclusion_reason,resolution_evidence,resolved_conflict_ids,notes
```

### Adjudication triggers

The triggers are mandatory and cumulative.

| Trigger | Required condition |
| --- | --- |
| A1 | Any screening-status disagreement. |
| A2 | Any criterion disagreement, even when statuses agree. |
| A3 | Both ratings are excluded with the same criterion but have unequal normalized exclusion reasons under the deterministic A3 rule. |
| A4 | The frozen candidate has a known unresolved screening conflict: one or more authoritative unresolved `screening_status` conflicts in the frozen coordinator snapshot, even when both ratings agree; all such conflicts trigger one complete-set resolution. |

For A3, normalize each complete `exclusion_reason` independently: apply Unicode
NFKC, then casefold, replace every maximal run of nonalphanumeric characters with one
ASCII space, collapse whitespace to one ASCII space, and trim leading and trailing
spaces. A3 is triggered if and only if the two normalized strings are unequal. Empty
normalized reasons are invalid before trigger evaluation. This deterministic rule
replaces subjective tests such as "materially different." It intentionally favors
sensitivity over adjudication economy: independently worded exclusions may trigger A3
even when their criteria agree, and the review report MUST disclose the resulting
adjudication count and workload.

A4 applies to the complete authoritative set of unresolved candidate
`screening_status` conflicts in the frozen coordinator snapshot. For every
adjudication, `resolved_conflict_ids` MUST exactly equal that complete set in UTF-8 byte
order; it MUST be `NR` if and only if the set is empty. When A4 applies the set is
nonempty, so `NR`, a proper subset, a superset, a conflict for another candidate, a
non-`screening_status` conflict, or an already resolved conflict is invalid. One
adjudication resolves the full authoritative set atomically; conflicts MUST NOT remain
partially resolved.

### Adjudication field contract

| Field | Normative rule |
| --- | --- |
| `candidate_id` | Exact candidate ID shared by both primary assignments. |
| `input_sha256` | Exact lowercase 64-hex candidate-input digest shared by both ratings. |
| `snapshot_sha256` | Exact coordinator screening snapshot digest shared by both ratings. |
| `primary_snapshot_sha256` | Lowercase 64-hex digest of the sealed primary-result snapshot containing both ratings. |
| `assignment_ids` | The two exact assignment IDs sorted by UTF-8 bytes and joined by one semicolon. |
| `adjudicator_id` | Stable ID of a third reviewer not present in `reviewer_ids`. |
| `reviewer_ids` | The two exact `coder_id` values in `assignment_ids` order, joined by one semicolon. |
| `decided_on` | ISO 8601 calendar date in `YYYY-MM-DD` form. |
| `screening_status` | One final controlled status. |
| `criterion` | Exactly one criterion permitted for the final status. |
| `access_status` | Controlled access status for the evidence inspected by the adjudicator. |
| `source_urls` | Canonical deciding-evidence URLs under the reviewer evidence rules. |
| `evidence_version` | Required non-`NR` version identifier under the reviewer evidence rules. |
| `evidence_retrieved_on` | Required ISO 8601 retrieval date for the adjudicator's evidence. |
| `evidence_archive_url` | Version-pinned archive URL or `NR` under the evidence rules. |
| `evidence_sha256` | Lowercase 64-hex artifact digest or `NR` under the evidence rules. |
| `screening_locator` | Precise locator for every fact controlling the final decision. |
| `exclusion_reason` | `NR` for included; a substantive source-specific reason for excluded. |
| `resolution_evidence` | Canonical one-line JSON object conforming exactly to the structured adjudication rationale contract below. |
| `resolved_conflict_ids` | Semicolon-separated IDs sorted by UTF-8 bytes that exactly equal the complete authoritative set of unresolved candidate `screening_status` conflicts, or `NR` if and only if that set is empty. |
| `notes` | Supplementary information or exactly `NR`. |

### Structured adjudication rationale

`resolution_evidence` MUST be a source-specific canonical one-line JSON object
rather than marker-delimited prose or a keyword list. Its complete serialized value
MUST be at least 120 characters. The object MUST contain exactly these ten top-level
keys and no others: `comparison_analysis`, `controlling_rules`,
`deciding_fact`, `deciding_locator`, `final_decision`,
`raw_exclusion_reasons`, `raw_ratings`, `resolved_conflicts`,
`schema_version`, and `source_url`. `schema_version` MUST be the JSON
string `1`; the numeric value `1` is invalid.

The following single line is the normative object shape. Angle-bracketed strings are
placeholders and MUST be replaced by the values required below.

```json
{"comparison_analysis":"<candidate-specific comparative analysis>","controlling_rules":["A1","A2","A3","A4"],"deciding_fact":{"kind":"retention_source_fact","text":"<source-specific fact>"},"deciding_locator":"<complete screening_locator>","final_decision":{"criterion":"<criterion>","screening_status":"<screening_status>"},"raw_exclusion_reasons":[{"assignment_id":"<assignment_id_0>","reason":"<complete reason 0>"},{"assignment_id":"<assignment_id_1>","reason":"<complete reason 1>"}],"raw_ratings":[{"assignment_id":"<assignment_id_0>","criterion":"<criterion_0>","screening_status":"<screening_status_0>"},{"assignment_id":"<assignment_id_1>","criterion":"<criterion_1>","screening_status":"<screening_status_1>"}],"resolved_conflicts":[{"conflict_id":"<conflict_id>","field":"<field>","value_a":"<value_a>","value_b":"<value_b>"}],"schema_version":"1","source_url":"<one complete canonical source URL>"}
```

| Key | Required JSON type and exact shape |
| --- | --- |
| `comparison_analysis` | String containing the candidate-specific comparative rationale required below. |
| `controlling_rules` | Array of one or more strings drawn from `A1`, `A2`, `A3`, and `A4`. |
| `deciding_fact` | Object with exactly the string keys `kind` and `text`. |
| `deciding_locator` | String containing the complete adjudication `screening_locator`. |
| `final_decision` | Object with exactly the string keys `criterion` and `screening_status`. |
| `raw_exclusion_reasons` | Array whose entries are objects with exactly the string keys `assignment_id` and `reason`. |
| `raw_ratings` | Array of exactly two objects, each with exactly the string keys `assignment_id`, `criterion`, and `screening_status`. |
| `resolved_conflicts` | Array whose entries are objects with exactly the string keys `conflict_id`, `field`, `value_a`, and `value_b`. |
| `schema_version` | JSON string `1`. |
| `source_url` | String containing one complete canonical deciding URL. |

All listed nested objects MUST contain exactly the named keys and no additional keys.
The complete JSON value MUST be serialized as UTF-8 with
`ensure_ascii=False`, `sort_keys=True`, `separators=(",", ":")`, and
`allow_nan=False`. Sorting applies recursively to every object; array order is
preserved because it carries authoritative meaning. The serialized field MUST contain
no indentation, byte-order mark, surrounding whitespace, or literal line break.
Only standard JSON values are permitted; `NaN`, `Infinity`, and
`-Infinity` are invalid.

The following fields bind exactly to immutable screening data:

- `raw_ratings` MUST exactly equal the two locked ratings in ascending UTF-8
  `assignment_id` order. Each object carries that row's complete
  `assignment_id`, `criterion`, and `screening_status`.
- `controlling_rules` MUST exactly equal the triggered `A1` through `A4` IDs
  in that order, with no untriggered, duplicate, or omitted ID.
- `raw_exclusion_reasons` MUST be an empty array unless `A3` applies. When
  `A3` applies, it MUST contain exactly two objects in the same assignment order,
  binding each complete raw `exclusion_reason` to its `assignment_id`.
- `resolved_conflicts` MUST exactly equal the conflicts named by
  `resolved_conflict_ids`. `resolved_conflicts` MUST reproduce every and only those
  authoritative conflicts in that field's UTF-8 order. Each object MUST reproduce
  the authoritative `conflict_id`, `field`, `value_a`, and `value_b`;
  the array MUST be empty when `resolved_conflict_ids` is `NR`. When A4 applies,
  both representations MUST cover the complete authoritative unresolved set.
- `final_decision` MUST exactly equal the adjudication row's
  `screening_status` and `criterion`.
- `deciding_locator` MUST exactly equal `screening_locator`.
- `source_url` MUST equal one complete canonical URL in `source_urls`.
  Prefix, suffix, or substring URL matches are invalid.
- For an excluded decision, `deciding_fact` MUST have
  `kind=exclusion_reason` and `text` exactly equal to the final
  `exclusion_reason`. For an included decision, it MUST have
  `kind=retention_source_fact` and a substantive non-`NR` `text` of at least
  48 characters and eight words that states the deciding source retention fact.

The `comparison_analysis` string, after Unicode NFKC normalization and whitespace
collapse, MUST contain at least 120 characters, 18 alphabetic words, 12 distinct
casefolded words, the candidate ID, and the comparative word `whereas`. For
`A1`, it MUST contain both differing raw statuses; for `A2`, both differing
raw criteria; for `A3`, a distinctive alphabetic word of at least five letters
from each complete raw exclusion reason; and for `A4`, every resolved conflict
ID. After authoritative identifiers, labels, reasons, URLs, locators, conflict
values, and decision values are removed, at least ten alphabetic words and eight
distinct words MUST remain. Token inventories, generic rationales, or concatenated
required values are invalid.

Automated sealing checks exact JSON structure, canonical serialization, provenance,
identifiers, trigger-specific bindings, and minimum comparison form. It can reject
missing fields and known boilerplate patterns, but it MUST NOT be represented as
proving semantic adequacy: arbitrary prose can satisfy any finite lexical validator.
A sealed adjudication or projection therefore remains structurally validated but
semantically pending. The accountable-author verification of all 202 decisions and
locators is the mandatory publication gate that rejects token inventories and
unsupported interpretations.

The adjudicator MUST be a third reviewer distinct from both original reviewers. The
adjudicator identity and execution record MUST differ from the two reviewer identities
actually bound to the immutable assignments; a self-asserted or free-form reviewer list
is insufficient. The set of `reviewer_ids` MUST exactly equal those two locked
`coder_id` values, and `assignment_ids` MUST identify the corresponding rows
in `primary_snapshot_sha256`.

The adjudicator inspects the strongest accessible source, both locked ratings, their
evidence and provenance, and any coordinator-only unresolved-conflict record.
Included adjudications remain subject to the full-text and official-
documentation provenance gates. `resolution_evidence` MUST be source-specific;
a label, status restatement, or unexplained preference is invalid.

Both locked raw ratings MUST be preserved byte-for-byte. Adjudication MUST append a
separate decision and MUST NOT rewrite either rating. Prior conflict information
remains hidden from original reviewers and is disclosed only to the adjudicator after
both ratings lock.

## Adjudication snapshot artifacts

A sealed adjudication snapshot MUST contain one canonical manifest row under this
exact header and field order:

```csv
manifest_version,adjudication_snapshot_sha256,coordinator_snapshot_sha256,protocol_sha256,calibration_result_snapshot_sha256,calibration_decision_snapshot_sha256,main_result_snapshot_sha256,primary_snapshot_sha256,adjudication_file_sha256,execution_registry_sha256,row_count,execution_row_count
```

| Field | Normative rule |
| --- | --- |
| `manifest_version` | Exactly `1`. |
| `adjudication_snapshot_sha256` | Lowercase 64-hex SHA-256 of the canonical adjudication binding defined below. |
| `coordinator_snapshot_sha256` | Exact lowercase 64-hex digest of the authoritative coordinator snapshot. |
| `protocol_sha256` | Exact lowercase 64-hex digest of the frozen protocol bytes. |
| `calibration_result_snapshot_sha256` | Exact lowercase 64-hex digest of the authoritative 60-rating calibration result snapshot. |
| `calibration_decision_snapshot_sha256` | `calibration_decision_snapshot_sha256` MUST identify the immutable calibration decision whose `decision` is `release` and whose coordinator, protocol, calibration result, identifier preimages, threshold, and ambiguity bindings all validate. |
| `main_result_snapshot_sha256` | Exact lowercase 64-hex digest of the authoritative 344-rating main result snapshot. |
| `primary_snapshot_sha256` | Exact lowercase 64-hex combined-primary digest of the calibration and main result snapshots. |
| `adjudication_file_sha256` | Lowercase 64-hex SHA-256 of canonical `adjudications.csv`. |
| `execution_registry_sha256` | Lowercase 64-hex SHA-256 of canonical `execution_registry.csv`. |
| `row_count` | Base-10 number of required adjudication rows. |
| `execution_row_count` | Base-10 number of execution rows, covering all 404 ratings and every required adjudication. |

The exact snapshot file set is `adjudications.csv`, `execution_registry.csv`,
`manifest.csv`, and `SHA256SUMS`; no other entry is allowed. `SHA256SUMS` MUST cover
the other three files in lexical path order. The execution register copied into the
snapshot MUST be byte-identical to the independently captured canonical input.

The adjudication snapshot hash MUST be SHA-256 of a canonical UTF-8 JSON object
containing exactly `manifest_version`, `coordinator_snapshot_sha256`,
`protocol_sha256`, `calibration_result_snapshot_sha256`,
`calibration_decision_snapshot_sha256`, `main_result_snapshot_sha256`,
`primary_snapshot_sha256`, `adjudication_file_sha256`,
`execution_registry_sha256`, and numeric `row_count` and
`execution_row_count`. Keys MUST be sorted, non-ASCII text MUST remain unescaped, and
separators MUST be comma and colon with no added whitespace.

Sealing and validation MUST capture and coherently re-attest the coordinator,
calibration result, calibration decision, main result, adjudication input, and
execution register. Adjudication MUST NOT be sealed or validated from a `revise`
decision, a mutable decision pointer, a self-declared decision hash, or a decision
bound to different inputs. The exact passing decision digest MUST remain in every
subsequent projection.

## Citation-key activation ledger

The audited citation-key activation ledger MUST be canonical UTF-8 CSV under this
exact header and field order:

```csv
candidate_id,cite_key
```

| Field | Normative rule |
| --- | --- |
| `candidate_id` | Exact candidate identifier from the authoritative coordinator snapshot; each candidate may appear at most once. |
| `cite_key` | Nonempty citation key matching `[A-Za-z0-9][A-Za-z0-9:._/+\-]*`; keys MUST be unique under Unicode casefold comparison. |

A previously keyless included candidate MAY be activated only through this
audited full `citation_keys.csv` ledger. A delta, replacement row, mutable side table,
or direct edit to `candidates.csv` is insufficient. The ledger MUST preserve the
coordinator `citation_keys.csv` rows as an exact append-only prefix. No baseline row may
be deleted, reordered, or changed. New assignments MUST apply only to previously
keyless candidates and MUST be appended in UTF-8 `candidate_id` byte order. Unknown
candidate IDs, duplicate candidate IDs, duplicate casefolded keys, reissued existing
keys, and blank cells are invalid.

Every final `included` candidate MUST resolve to one active key in this
ledger. The final projected `candidates.csv` exposes that key for included
records and leaves the active candidate key blank for excluded records; this output
rule does not alter the immutable ledger. The projection MUST copy the complete
canonical ledger to `citation_keys.csv`. Its manifest MUST bind both
`citation_key_ledger_sha256`, the digest of the independently captured full ledger
input, and `citation_keys_sha256`, the digest of the copied canonical projection file.

## Reliability reporting

All reliability statistics MUST use the two locked, pre-adjudication, final-version
ratings for all 202 records. Adjudicated decisions MUST NOT replace raw ratings in
reliability calculations. Order the two ratings per candidate by `assignment_id`
before forming directional tables. For statuses `k` and `l`, `n_kl` is the number
of candidates whose first assignment-ordered rating is `k` and whose second
assignment-ordered rating is `l`.

Duplicate agreement measures consistency of interpretation of coordinator-curated
frozen packets. It does not independently estimate retrieval reliability, packet
completeness, source authenticity, or evidence-selection bias. Packet assembly and
audit are separate processes. Packet defects require versioning and rerating.

| Metric | Required calculation |
| --- | --- |
| Overall exact agreement | Number of candidates with identical statuses divided by 202, with exact numerator and denominator. |
| Overall exact criterion agreement | Number of candidates with identical controlled criterion values divided by 202, with exact numerator and denominator. |
| Category-specific agreement | For each category `k`, define directional `a_k=n_kk`, `b_k=sum_(l!=k) n_kl`, `c_k=sum_(l!=k) n_lk`, and `d_k=N-a_k-b_k-c_k`; report `2a_k/(2a_k+b_k+c_k)` and `2d_k/(2d_k+b_k+c_k)`. A zero denominator is `not_estimable`. |
| Nominal Krippendorff alpha | Let `P_o=sum_k n_kk/N`, `D_o=1-P_o`, and pooled `n_k=sum_l(n_kl+n_lk)` over `2N` ratings. Use `D_e=sum_k n_k(2N-n_k)/(2N(2N-1))` and `alpha=1-D_o/D_e`; `D_e=0` is `not_estimable`. |
| Nominal Gwet AC1 | With `K=2` and `p_k=n_k/(2N)`, use `P_e=sum_k p_k(1-p_k)/(K-1)` and `AC1=(P_o-P_e)/(1-P_e)`; `1-P_e=0` is `not_estimable`. |

Overall exact criterion agreement MUST be reported as an exact numerator and
denominator over all 202 candidates. Also report a criterion-by-criterion disagreement
table so that equal statuses with unequal controlled criteria remain visible. Missing, invalid, or duplicate ratings are hard errors rather than omissions
from denominators.

Status agreement, exact criterion agreement, nominal Krippendorff alpha, and nominal
Gwet AC1 MUST each have a deterministic 95% candidate-bootstrap interval for the
`calibration` and `full_corpus` report scopes. Use exactly 10000 replicates per scope.
The resampling unit is the candidate: every replicate samples `N` paired candidate
units with replacement, where `N` is 30 or 202 for the named scope, and preserves both
raw ratings within each selected unit.

Define the lowercase combined-primary digest as SHA-256 of canonical UTF-8 JSON
`{"calibration_result_snapshot_sha256":C,"main_result_snapshot_sha256":M}` with
lexicographically sorted keys and separators `,` and `:` without spaces. For each
scope `S` (`calibration` or `full_corpus`), replicate `r` from 0 through 9999, and
draw `j` from 0 through `N-1`, compute SHA-256 over UTF-8
`screening-bootstrap-v1`, one NUL byte, the ASCII combined-primary digest, one NUL
byte, ASCII `S`, one NUL byte, base-10 `r`, one NUL byte, and base-10 `j`.
Interpret the first eight digest bytes as an unsigned big-endian integer and reduce
modulo `N` to select the candidate in global UTF-8 `candidate_id` order for that scope.
Each scope starts at `(r=0,j=0)` and has an independent hash stream. This is the
normative pseudo-random generator; no stateful generator or prior-scope draws may
influence it.

For each metric, sort finite replicate estimates numerically. With `m` valid
replicates, use the one-indexed order statistics at
`ceil(0.025 * m)` and `ceil(0.975 * m)` as the percentile limits. A replicate
whose alpha or AC1 denominator is zero is non-estimable for that metric and MUST NOT be
coerced to zero or NaN. The valid-replicate count MUST be reported separately for each
metric, including `10000` for status agreement. Zero valid replicates is a hard
error. Report point estimates, interval endpoints, implementation version, complete
status contingency table, and bootstrap algorithm identifier.

Calibration agreement and its criterion disagreements are reported separately as
release-control statistics. They do not substitute for final reliability on all 202
reports.

## Evidence packet phase releases

Every evidence packet manifest is canonical UTF-8 CSV with LF endings, this exact
header and field order:

```csv
candidate_id,artifact_id,artifact_role,source_url,evidence_version,evidence_retrieved_on,access_status,evidence_archive_url,evidence_sha256,local_filename,redistribution_status,retrieval_notes
```

Each row names one candidate artifact. `candidate_id` and `artifact_id` are sorted by
UTF-8 bytes and the pair is unique. URLs are canonical HTTPS URLs; the version and
retrieval date identify the inspected artifact; `access_status` is an allowed access
classification; and `redistribution_status` is exactly `public-redistributable`,
`local-restricted`, or `metadata-only`. `metadata-only` rows use `NR` for both
`evidence_sha256` and `local_filename`; all other locally retained bytes use a
lowercase SHA-256 digest paired with one normalized relative POSIX `local_filename`.
The named local bytes MUST hash to `evidence_sha256` during packet assembly.

For limited access, `retrieval_notes` has the exact grammar `attempted:
doi_or_publisher=<outcome> | title_author=<outcome> |
scholarly_index_or_repository=<outcome> | official_page=<outcome>; outcome:
<final outcome>`. These four attempt labels are required in that order. The final
outcome records the access limitation; full-text rows may instead use `NR` or a
substantive retrieval note.

Evidence binds to immutable reviewer phase releases, never to the coordinator. A
calibration release contains exactly the 30 calibration candidates and a main release,
created only after a passing calibration gate, contains exactly the 172 main
candidates. Each assigned candidate has at least one manifest artifact, and no
unassigned candidate appears. Both assignments for a candidate use the same ordered
candidate binding of `(artifact_id,evidence_sha256,local_filename)`.

For binary-result coordinators, the release root copies the canonical manifest as
`evidence_packet_manifest.csv`. Its SHA-256, artifact count, and canonical
candidate-binding digest are bound in the versioned release manifest, release snapshot
digest, and `SHA256SUMS`. Compatibility is selected from authenticated manifest
version and fields, never from a path name. Historical release manifests and trees
remain v1 artifacts.

Local evidence bytes remain untracked; canonical manifests and their hashes are
committed. Release creation verifies the local bytes. Later committed-release
validation trusts the committed hashes and canonical manifest rather than reopening
the archive; role staging re-verifies the local bytes before a reviewer uses a packet.
There is no transaction-wide defense against a concurrent hostile local archive writer:
ordinary controlled packet assembly is assumed, and the authoritative control is the
per-file SHA-256 digest.

For an evidence-bound release v2, each role stage uses execution
`configuration_version` `3`. Its controlled additions are exactly
`allowed_screening_statuses` = [`included`,`excluded`],
`allowed_inclusion_criteria` = [`include-relevant`], and
`evidence_packet_manifest_sha256`, the SHA-256 of that role's filtered staged
`evidence_packet_manifest.csv`. Historical release v1 stages retain their v1/v2
configuration derivation and do not gain this evidence tree.

Role staging filters the phase manifest to the candidates in its one role packet while
preserving canonical candidate/artifact order, and requires exact candidate coverage.
For every row with local bytes, staging reopens and verifies the archive source before
copying the exact bytes to
`evidence/<candidate_id>/<artifact_id>/<basename>`. `metadata-only` rows create no
byte file. Duplicate reviewers for the same candidate receive identical filtered rows
and identical staged bytes.

The staged manifest and every staged evidence path and hash are included in the role
stage digest and `SHA256SUMS`. Before use, validation checks the configuration binding,
manifest bytes, exact evidence tree and paths, and each stage-local SHA-256; missing,
extra, swapped, or mutated bytes are invalid. These stage paths and copied bytes are
procedural untracked artifacts, not committed evidence inventory artifacts.

## Automation, AI assistance, and accountability

The paper MUST disclose whether automation or AI assistance was used and, if it was,
identify its role, tool or model and version, instructions, affected stages, and
verification procedure. The disclosure distinguishes bibliographic retrieval,
full-text access, eligibility rating, evidence extraction, validation, adjudication
support, statistical computation, and accountable-author verification. It reports
limitations and deviations using immutable run records.

The protocol MUST NOT represent an automated agent as a human reviewer. Every
`coder_id` and `adjudicator_id` MUST be typed in an immutable execution
register as `human`, `automated`, or `hybrid`. An automated process is
an auditable reviewer role, not a person. Distinct IDs do not by themselves establish
independence.

Independent automated ratings MUST use fresh contexts. No conversation history,
memory, ratings, results, or reviewer-produced retrieval state may be supplied across
roles. Duplicate roles MAY share only the same frozen protocol and the same frozen
evidence packet bound to their candidate. Public retrieval is limited to metadata
verification or packet-defect reporting and MUST NOT add eligibility evidence. When a
provider does not expose retrieval-cache isolation, the frozen profile and execution
register MUST record that limitation and the paper MUST disclose it; they MUST NOT claim
technical cache isolation. No automated reviewer may be resumed from another
reviewer's context or receive a summary of another review before both results lock.

Every automated or hybrid reviewer MUST receive a rendered copy of
`screening_reviewer_prompt.md` that binds its role, exact protocol and packet paths and
hashes, and output path. The immutable execution register MUST record the exact model
identifier, model version, configuration hash, and prompt hash for every automated
role. It MUST also bind the provider or runtime, tool configuration, decoding
parameters, canonical system/developer/user instruction hashes, context or run ID,
retrieval configuration, cache-isolation statement, start date, and result-file
digest. Changes to any of these create a new execution identity. Human and hybrid
roles MUST record the human role, training or calibration exposure, and which actions
were automated.

Limited-provider mode records only provider metadata that was not exposed and does not relax
the required visible prompt, execution configuration, user instruction, or result
bindings.

Before coordinator freeze, canonical `execution_profile.json` and
`screening_reviewer_prompt.md` bytes MUST pass the public validators and become
digest-bound coordinator and reviewer-release inputs. Before each execution, the public
staging command MUST derive one random role-private snapshot containing exactly one
packet and the rendered absolute-path prompt. Its manifest is the authoritative
preimage for `configuration_sha256`, `prompt_sha256`, and
`user_instruction_sha256`; the latter two are equal because the rendered prompt is the
sole user message. Mode `0700` and random paths reduce accidental disclosure on the
shared host but are procedural controls, not an ACL, container, or mount boundary.

### Execution register schema

The immutable execution register MUST use this exact header and field order:

Limited-provider mode MUST be configured and validated before any screening freeze.
Its limitations are fields within `tool_configuration` under the exact rules below.

```csv
execution_id,role_id,role_type,context_id,task,work_item_id,model_identifier,model_version,configuration_sha256,prompt_sha256,provider,runtime,tool_configuration,retrieval_configuration,decoding_parameters,system_instruction_sha256,developer_instruction_sha256,user_instruction_sha256,cache_isolation_statement,started_on,completed_on,result_file_sha256,human_role,training_calibration_exposure,automated_actions
```

| Field | Normative rule |
| --- | --- |
| `execution_id` | Stable execution identifier. All rows for one `role_id` and `task` MUST use one execution identifier, and it MUST NOT be shared by another role-task owner. |
| `role_id` | Exact `coder_id` for screening or exact `adjudicator_id` for adjudication. |
| `role_type` | Exactly `human`, `automated`, or `hybrid`; one role MUST NOT change type across tasks. |
| `context_id` | Stable context or run identifier. It MUST be unique to one role-task owner. |
| `task` | Exactly `calibration-screening`, `main-screening`, or `adjudication`. |
| `work_item_id` | Exact assignment ID for screening tasks or candidate ID for adjudication. |
| `model_identifier` | Required stable identifier for automated and hybrid roles; exactly `NR` for a human role. |
| `model_version` | Required backend model version for automated and hybrid roles. When the backend version is not exposed, it MUST use exact form `requested:<alias-or-date>` under the limitation rule below. It is exactly `NR` for a human role. |
| `configuration_sha256` | Lowercase 64-hex digest of the complete execution configuration for automated and hybrid roles; `NR` for human. |
| `prompt_sha256` | Lowercase 64-hex digest of the exact UTF-8 bytes of the rendered visible reviewer prompt for automated and hybrid roles; hidden instructions are excluded; `NR` for human. |
| `provider` | Required stable provider identifier for automated and hybrid roles; `NR` for human. |
| `runtime` | Required stable runtime identifier for automated and hybrid roles; `NR` for human. |
| `tool_configuration` | Canonical JSON object with sorted keys and no insignificant spaces for automated and hybrid roles. Its optional `provider_metadata_limitations` member is governed below; `NR` for human. |
| `retrieval_configuration` | Canonical JSON object with sorted keys and no insignificant spaces for automated and hybrid roles; `NR` for human. |
| `decoding_parameters` | Canonical JSON object for automated and hybrid roles, or exactly `NR` only under the matching provider limitation below; it MUST be `NR` for human. |
| `system_instruction_sha256` | Lowercase 64-hex digest of exact system-instruction bytes for automated and hybrid roles, or exactly `NR` only under the matching provider limitation below; `NR` for human. |
| `developer_instruction_sha256` | Lowercase 64-hex digest of exact developer-instruction bytes for automated and hybrid roles, or exactly `NR` only under the matching provider limitation below; `NR` for human. |
| `user_instruction_sha256` | Lowercase 64-hex digest of canonical user instructions for automated and hybrid roles; `NR` for human. |
| `cache_isolation_statement` | Exactly `Fresh context; no shared conversation history, memory, ratings, results, or retrieval cache.` in full-provider mode for automated and hybrid roles, or exactly `Fresh context; no shared conversation history, memory, ratings, or results were supplied; provider retrieval-cache isolation was not exposed.` only under the matching provider limitation below, after Unicode NFKC, whitespace collapse, and casefold comparison; exactly `NR` for human. |
| `started_on` | ISO 8601 date in `YYYY-MM-DD`; it MUST NOT follow `completed_on`. |
| `completed_on` | ISO 8601 date in `YYYY-MM-DD`. |
| `result_file_sha256` | Lowercase 64-hex digest of the exact sealed reviewer batch file, or of canonical `adjudications.csv` for adjudication. |
| `human_role` | Required stable human identity/role identifier that identifies the accountable individual for human and hybrid roles, not merely a generic role label; exactly `NR` for automated. |
| `training_calibration_exposure` | Substantive record of prior training or calibration exposure for human and hybrid roles; exactly `NR` for automated. |
| `automated_actions` | Non-`NR` account of automated actions for human and hybrid roles; a hybrid role MUST NOT use `none`; exactly `NR` for automated. |

For automated and hybrid roles, `configuration_sha256`, `prompt_sha256`, and
`user_instruction_sha256` remain REQUIRED lowercase 64-hex digests in both
full-provider and limited-provider modes. `prompt_sha256` is the SHA-256 of the exact
UTF-8 bytes of the rendered visible reviewer prompt; hidden system and developer
instructions are excluded and have their own fields.

The optional `tool_configuration.provider_metadata_limitations` member has this closed
key vocabulary, and every present key MUST have the exact string value
`provider-not-exposed`:

`{"backend_model_version":"provider-not-exposed","decoding_parameters":"provider-not-exposed","developer_instruction_bytes":"provider-not-exposed","retrieval_cache_isolation":"provider-not-exposed","system_instruction_bytes":"provider-not-exposed"}`

The displayed object shows all permitted keys; an execution records only the applicable
nonempty subset. A `system_instruction_sha256` value of `NR` is permitted only when
`provider_metadata_limitations` contains `system_instruction_bytes` with exact value
`provider-not-exposed`. A `developer_instruction_sha256` value of `NR` is permitted
only when `provider_metadata_limitations` contains `developer_instruction_bytes` with
exact value `provider-not-exposed`. `decoding_parameters` MAY be `NR` only when
`provider_metadata_limitations` contains `decoding_parameters` with exact value
`provider-not-exposed`. An exposed configuration with no tunable parameters MUST use
the canonical empty object `{}` rather than another sentinel.

`model_version` MUST use the exact `requested:<alias-or-date>` form if and only if
`provider_metadata_limitations` contains `backend_model_version` with exact value
`provider-not-exposed`. The `<alias-or-date>` suffix MUST satisfy the stable-identifier
grammar. Thus a missing key for a requested value, a key paired with a backend-reported
value, and a malformed requested value are all invalid; `model_version` MUST NOT be
`NR` for any automated or hybrid role. The limited cache-isolation statement is
permitted only when
`provider_metadata_limitations` contains `retrieval_cache_isolation` with exact value
`provider-not-exposed`; the full statement requires that key to be absent.

The limitations object MUST be a nonempty JSON object containing only the five
documented keys and exact `provider-not-exposed` values. Unknown keys, missing
justifications, mismatched fields, unjustified limitations, and `NR` values without
their corresponding limitation MUST be rejected.

The register contains exactly one row for each of the 404 screening assignments and one
row for each required adjudication candidate, with no extra work item. Rows are
canonical UTF-8 CSV ordered by task, work item, and role byte order. All non-work-item
fields MUST be identical within one role-task execution. Each rating's role and
`result_file_sha256` MUST match its sealed phase-result batch; each adjudication's
role and digest MUST match canonical `adjudications.csv`.

For every candidate, the two reviewer execution contexts MUST be distinct. For every
adjudicated candidate, the adjudicator context MUST differ from both reviewer contexts.
An identifier inequality without this registered context evidence is insufficient.

`human_role` is a stable human identity/role identifier that identifies the
accountable individual across all human or hybrid execution rows. It MUST NOT be a
generic role-type label that could be reused by different people. For every candidate,
the paired human or hybrid reviewer `human_role` values MUST be distinct. For every
adjudicated candidate, a human or hybrid adjudicator's `human_role` MUST differ from
both paired human or hybrid reviewer `human_role` values. These identity constraints
apply in addition to distinct `role_id`, `execution_id`, and `context_id` bindings.

The authors remain accountable for the final eligibility decisions, extracted
evidence, analyses, and claims. Before publication, accountable authors MUST verify all
202 final eligibility decisions and every deciding evidence locator against the cited
source. Verification includes status, criterion, access sufficiency, version and
retrieval provenance, retention or exclusion rationale, adjudication outcome, and
work-level duplicate grouping. Each check MUST be bound to an author identifier, date,
candidate ID, evidence version, and final-decision digest in an immutable sign-off
record. Sampling is insufficient for this publication gate.

This codebook does not name any person or tool as having completed screening and does
not assert that any planned action has occurred. Completed actions, identities, model
versions, oversight, and verification steps may be reported only from immutable run
records; they MUST NOT be inferred from this prospective protocol.

## Accountable-author verification

Before projection, author verification MUST be supplied as canonical UTF-8 CSV under
this exact header and field order:

```csv
candidate_id,primary_snapshot_sha256,adjudication_snapshot_sha256,decision_sha256,evidence_versions_sha256,deciding_locators_sha256,verified_by,verified_role,verified_on,verification_status,verification_evidence
```

| Field | Normative rule |
| --- | --- |
| `candidate_id` | Exact candidate ID; the file contains exactly one row for every frozen candidate. |
| `primary_snapshot_sha256` | Exact lowercase 64-hex digest of the combined calibration and main result snapshots used for the final decision. |
| `adjudication_snapshot_sha256` | Exact lowercase 64-hex digest of the authoritative adjudication snapshot. |
| `decision_sha256` | Lowercase 64-hex digest of the candidate-specific final decision binding defined below. |
| `evidence_versions_sha256` | Lowercase 64-hex digest of the canonical evidence-version array used by the final decision. |
| `deciding_locators_sha256` | Lowercase 64-hex digest of the canonical deciding-locator array used by the final decision. |
| `verified_by` | Stable author identifier matching `[A-Za-z0-9][A-Za-z0-9._:-]{2,127}`. |
| `verified_role` | `verified_role` MUST be exactly `accountable-author`. |
| `verified_on` | ISO 8601 calendar date in `YYYY-MM-DD` form. |
| `verification_status` | `verification_status` MUST be exactly `verified`. |
| `verification_evidence` | `verification_evidence` MUST be a substantive candidate-specific sign-off: after Unicode NFKC normalization and trimming it is not `NR`, contains the candidate ID case-insensitively, contains at least 48 characters, and contains at least eight alphabetic words. |

The file MUST contain exactly one row for each of the 202 candidates in UTF-8
`candidate_id` byte order, with no omission, duplicate, or extra row. Every row MUST
bind the same authoritative primary and adjudication snapshots.

For each candidate, `decision_sha256` is SHA-256 of canonical UTF-8 JSON containing the
complete projected candidate row, all candidate-level conflict rows in UTF-8
`conflict_id` order, and the two final screening-decision rows in UTF-8
`assignment_id` order. For an adjudicated candidate, the evidence-version and
deciding-locator digest preimages are one-element JSON arrays containing the
adjudicator's `evidence_version` and `screening_locator`. For a direct final decision,
each preimage is the unique values from the two locked ratings sorted by UTF-8 bytes.
Each digest uses canonical JSON with sorted object keys, unescaped non-ASCII text, and
comma and colon separators without added whitespace.

The accountable author MUST inspect the cited source and verify status, criterion,
access sufficiency, evidence version, retrieval provenance, every deciding locator,
rationale, adjudication outcome, conflict resolution, and work-level grouping before
signing. `verification_evidence` records that substantive candidate-specific check; a
generic attestation, sampling, delegated automated assertion, or lexical token
inventory is invalid. These rows are an external immutable input and MUST be
re-attested before, during, and after projection publication.

## Screening projection snapshot

The publication projection MUST contain one canonical manifest row under this exact
header and field order:

```csv
manifest_version,projection_snapshot_sha256,coordinator_snapshot_sha256,protocol_sha256,calibration_result_snapshot_sha256,calibration_decision_snapshot_sha256,main_result_snapshot_sha256,primary_snapshot_sha256,adjudication_snapshot_sha256,execution_registry_sha256,citation_key_ledger_sha256,author_verification_sha256,candidates_sha256,citation_keys_sha256,conflicts_sha256,screening_decisions_sha256,screening_agreement_sha256,candidate_count,decision_row_count,agreement_row_count
```

| Field | Normative rule |
| --- | --- |
| `manifest_version` | Exactly `1`. |
| `projection_snapshot_sha256` | Lowercase 64-hex SHA-256 of the canonical projection binding defined below. |
| `coordinator_snapshot_sha256` | Exact lowercase 64-hex digest of the authoritative coordinator snapshot. |
| `protocol_sha256` | Exact lowercase 64-hex digest of the frozen protocol bytes. |
| `calibration_result_snapshot_sha256` | Exact lowercase 64-hex digest of the authoritative calibration result snapshot. |
| `calibration_decision_snapshot_sha256` | `calibration_decision_snapshot_sha256` MUST identify the same immutable passing `release` decision bound by adjudication and replayed against the coordinator and calibration result. |
| `main_result_snapshot_sha256` | Exact lowercase 64-hex digest of the authoritative main result snapshot. |
| `primary_snapshot_sha256` | Exact lowercase 64-hex combined-primary digest. |
| `adjudication_snapshot_sha256` | Exact lowercase 64-hex digest of the authoritative adjudication snapshot. |
| `execution_registry_sha256` | Exact lowercase 64-hex digest of canonical `execution_registry.csv` in the adjudication snapshot. |
| `citation_key_ledger_sha256` | Lowercase 64-hex digest of the independently captured audited full citation-key ledger. |
| `author_verification_sha256` | Lowercase 64-hex digest of the independently captured canonical author-verification file; the copied `author_verification.csv` MUST have the same digest. |
| `candidates_sha256` | Lowercase 64-hex SHA-256 of projected `candidates.csv`. |
| `citation_keys_sha256` | Lowercase 64-hex SHA-256 of the complete canonical projected `citation_keys.csv`. |
| `conflicts_sha256` | Lowercase 64-hex SHA-256 of projected `conflicts.csv`. |
| `screening_decisions_sha256` | Lowercase 64-hex SHA-256 of projected `screening_decisions.csv`. |
| `screening_agreement_sha256` | Lowercase 64-hex SHA-256 of projected `screening_agreement.csv`. |
| `candidate_count` | Exactly `202`. |
| `decision_row_count` | Exactly `404`, preserving both locked ratings for every candidate. |
| `agreement_row_count` | Base-10 row count of canonical `screening_agreement.csv`. |

The exact projection file set is `candidates.csv`, `citation_keys.csv`,
`conflicts.csv`, `screening_decisions.csv`, `screening_agreement.csv`,
`author_verification.csv`, `manifest.csv`, and `SHA256SUMS`; no other entry is
allowed. `SHA256SUMS` MUST cover every other file in lexical path order. All CSV files
MUST be canonical replay outputs from the captured immutable inputs.

The projection snapshot hash MUST be SHA-256 of canonical UTF-8 JSON containing
`manifest_version`; every input digest through `author_verification_sha256`; an
`outputs` object mapping each of the six projected data filenames to its lowercase
SHA-256 digest; and numeric `candidate_count`, `decision_row_count`, and
`agreement_row_count`. Object keys MUST be sorted recursively, non-ASCII text MUST
remain unescaped, and separators MUST be comma and colon with no added whitespace.
The manifest's individual output hashes and counts MUST equal that binding.

Projection sealing is forbidden until all 202 accountable-author verification rows
validate. Sealing and validation MUST coherently replay and re-attest the authoritative
coordinator, calibration result, passing calibration decision, main result,
adjudication snapshot, execution register, citation-key ledger, and author-verification
input. A self-consistently rehashed projection, a changed input, a missing sign-off, a
nonpassing calibration decision, or any cross-snapshot binding mismatch is invalid.
