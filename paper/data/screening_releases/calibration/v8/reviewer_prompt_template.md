# Screening reviewer execution prompt

ROLE_ID: {{ROLE_ID}}
STAGE_PATH: {{STAGE_PATH}}
PROTOCOL_PATH: {{PROTOCOL_PATH}}
PROTOCOL_SHA256: {{PROTOCOL_SHA256}}
PACKET_PATH: {{PACKET_PATH}}
PACKET_SHA256: {{PACKET_SHA256}}
OUTPUT_PATH: {{OUTPUT_PATH}}
OUTPUT_SHA256: compute from the exact closed output bytes and report at completion

Act only as the screening reviewer identified by `ROLE_ID`. The absolute `STAGE_PATH`, `PROTOCOL_PATH`, `PACKET_PATH`, and `OUTPUT_PATH` above are the complete path instructions for this execution. Follow the exact protocol bytes at `PROTOCOL_PATH` and process every assignment in the exact packet bytes at `PACKET_PATH`. Before rating, verify both supplied SHA-256 digests. Stop without writing ratings if any path binding or digest fails.

## Sole inputs and context

The protocol, assigned packet, `{{STAGE_PATH}}/evidence_packet_manifest.csv`, and the evidence files rooted at `{{STAGE_PATH}}/evidence/` are the sole supplied screening inputs. For every assignment, use the manifest to identify and inspect that candidate's staged artifact before deciding eligibility.
No other conversation history, memory, ratings, results, summaries, or context may be supplied.
Both duplicate reviewers MUST rate the same frozen evidence packet.
Do not inspect another role's output, execution trace, or working path. Do not use v3-v6 ratings or disagreements.

## Evidence procedure

Inspect only the direct material evidence frozen into the staged evidence packet when deciding eligibility. Do not substitute the bibliographic packet row for its staged artifact. For PDF evidence, record a locator in the form `PDF p. <number>; Section <label>` (adding Figure, Table, Algorithm, Appendix, or a stable official anchor where applicable).
Public retrieval during rating MAY verify metadata or report a packet defect but MUST NOT silently replace or add eligibility evidence.
Stronger evidence after freeze requires a new packet version. Report the defect and do not alter the current packet's eligibility basis.
Apply the protocol's binary retention procedure: return `included`,`include-relevant` when packet evidence establishes a core, supporting, or contextual condition; otherwise return `excluded` with one controlled exclusion criterion. `boundary` is historical terminology only and is forbidden as a v7 result. Record source-specific evidence, precise locators, access status, version, retrieval date, archive URL, and artifact SHA-256 exactly as required.
Do not choose or rank a primary contribution, perform Pass 2 coding, call fixed routes generation methods, infer eligibility from titles or snippets, or use another reviewer's work.

## Result contract

RESULT_HEADER:

```csv
assignment_id,phase,candidate_id,input_sha256,snapshot_sha256,batch_id,coder_id,screened_on,screening_status,criterion,access_status,source_urls,evidence_version,evidence_retrieved_on,evidence_archive_url,evidence_sha256,screening_locator,exclusion_reason,notes
```

Write only canonical UTF-8 CSV to `OUTPUT_PATH`.
Use the exact `RESULT_HEADER`, LF line endings, RFC 4180 quoting, and one row for every packet assignment in canonical assignment order. Populate every field; use `NR` only where the protocol permits it. Set every `coder_id` to `ROLE_ID`. Write no temporary rating file outside the role-private working path. Before returning the completion record, run `python3 -B -m paper.scripts.screening_results --validate-role-result --reviewer-stage {{STAGE_PATH}} --result {{OUTPUT_PATH}}`; return a completion record only if that command exits with status 0.

Do not emit a prose rating summary.
Do not include decision, status, criterion, access, or evidence counts or summaries in the completion response. The required total `ROWS_WRITTEN` is the only count permitted. After closing the canonical CSV, compute its SHA-256 from the exact file bytes and return only the completion record below.

```text
ROLE_ID={{ROLE_ID}}
ROWS_WRITTEN={{ROWS_WRITTEN}}
OUTPUT_PATH={{OUTPUT_PATH}}
OUTPUT_SHA256={{OUTPUT_SHA256}}
```

## Prompt provenance

`prompt_sha256` is the SHA-256 of the exact UTF-8 bytes of this rendered visible reviewer prompt.
Hidden system or developer instructions are not part of `prompt_sha256`; their exact bytes use the separate execution-register fields or the protocol's explicit provider limitation declarations.
