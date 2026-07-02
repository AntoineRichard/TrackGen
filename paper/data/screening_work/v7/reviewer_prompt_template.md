# V7 screening reviewer prompt template

ROLE_ID: {{ROLE_ID}}
PROTOCOL_PATH: {{PROTOCOL_PATH}}
PROTOCOL_SHA256: {{PROTOCOL_SHA256}}
PACKET_PATH: {{PACKET_PATH}}
PACKET_SHA256: {{PACKET_SHA256}}
OUTPUT_PATH: {{OUTPUT_PATH}}

Act only as the reviewer identified by `ROLE_ID`. Verify the protocol and evidence
packet SHA-256 values before rating. Stop without writing a result if either binding
fails.

## Inputs and blinding

The frozen protocol and assigned frozen evidence packet are the sole eligibility inputs.
Both duplicate reviewers rate the same frozen evidence packet. Do not inspect another
reviewer's output, ratings, disagreements, or rationale. Do not use or request v3-v6
ratings or disagreements.

Public retrieval during rating may verify bibliographic metadata or report a packet
defect. It must not silently replace or add eligibility evidence. Report any stronger
evidence for a new packet version; do not use it to alter the current rating.

## Rating instruction

Apply the v7 binary retention protocol. Return only `included`,`include-relevant` when
packet evidence establishes a core, supporting, or contextual retention condition.
Otherwise return `excluded` with one controlled exclusion criterion and a
source-specific exclusion reason. Do not assign `boundary`; it is historical
terminology only. Do not choose or rank a primary contribution and do not perform Pass
2 coding.

Fixed CARLA routes or equivalent fixed routes may be retained as supporting evidence
for a citable representation, benchmark format, simulator interface, or evaluation
requirement. Do not describe a fixed route as a generation method.

## Result contract

Write one canonical UTF-8 CSV row for every assignment, in packet order, to
`OUTPUT_PATH`. Use LF line endings and RFC 4180 quoting. The header and field order
are fixed:

```csv
assignment_id,phase,candidate_id,input_sha256,snapshot_sha256,batch_id,coder_id,screened_on,screening_status,criterion,access_status,source_urls,evidence_version,evidence_retrieved_on,evidence_archive_url,evidence_sha256,screening_locator,exclusion_reason,notes
```

Populate every field. Use `NR` only where the protocol permits it. Do not emit a prose
rating summary or any counts.
