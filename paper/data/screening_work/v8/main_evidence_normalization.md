# Main Evidence Normalization

## Coverage

- Frozen candidate coverage: 202/202.
- Trusted byte-backed rows: 28.
- Provisional metadata-only rows: 174.

## Acquisition Queue

Exact CSV header:

```text
candidate_id,title,source_url,raw_access_status,action,priority,limitation_note
```

| Action | Count |
| --- | ---: |
| replace-mismatched-local | 1 |
| replace-corrupt-local | 1 |
| archive-public-full-text | 50 |
| archive-official-source | 48 |
| user-fetch-or-document-limitation | 74 |

Queue order is deterministic: high, medium, then low priority; ties use UTF-8 candidate_id order.
High priority applies when title, source type, or source URL contains a generation, course, scenario, or environment term (including track, road, route, world, map, or terrain). Medium priority applies to replacement work or simulation/driving/vehicle/racing/benchmark/navigation evidence. All remaining candidates are low priority.

## Limitations

This normalization did not download public links or copy any source-archive bytes. A row is byte-backed only when its v7 manifest row declares a local file present beneath the supplied source archive root and its SHA-256 matches that declaration. All other rows are deliberately metadata-only and require the separately listed acquisition action.
The provisional evidence_retrieved_on value (2026-07-06) records the deterministic normalization date, not a public-link download claim.
