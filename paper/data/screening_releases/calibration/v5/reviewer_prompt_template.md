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

The protocol and assigned packet are the sole supplied screening inputs.
No other conversation history, memory, ratings, results, summaries, or context may be supplied.
Do not inspect another role's packet, output, execution trace, or working path. Independently retrieve only the public sources needed to screen the assigned reports.

## Evidence procedure

Retrieve and inspect direct material evidence from independently retrieved public sources.
Attempt exhaustive retrieval before using `abstract_only`.
Apply the protocol's inclusion criteria in order, then boundary, then exclusion. Record source-specific evidence, precise locators, access status, version, retrieval date, archive URL, and artifact SHA-256 exactly as required. Do not infer eligibility from titles, snippets, topic similarity, or another reviewer's work.

## Result contract

RESULT_HEADER:

```csv
assignment_id,phase,candidate_id,input_sha256,snapshot_sha256,batch_id,coder_id,screened_on,screening_status,criterion,access_status,source_urls,evidence_version,evidence_retrieved_on,evidence_archive_url,evidence_sha256,screening_locator,exclusion_reason,notes
```

Write only canonical UTF-8 CSV to `OUTPUT_PATH`.
Use the exact `RESULT_HEADER`, LF line endings, RFC 4180 quoting, and one row for every packet assignment in canonical assignment order. Populate every field; use `NR` only where the protocol permits it. Set every `coder_id` to `ROLE_ID`. Write no temporary rating file outside the role-private working path.

Before treating the result as closed, run this structural validation command from the
repository root:

```text
python3 -B -m paper.scripts.screening_results --validate-role-result --reviewer-stage "{{STAGE_PATH}}" --result "{{OUTPUT_PATH}}"
```

If validation fails, correct only your own result file and rerun the command until it
exits successfully. This validator checks the assigned packet binding, canonical CSV
shape and order, controlled evidence fields, and locator precision. It does not expose
another reviewer's packet, rating, result, or trace.

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
