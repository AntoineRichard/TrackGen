# Main Evidence Packet

## Coverage

- Final manifest rows: 202/202.
- Byte-backed rows: 82.
- Provisional metadata-only rows: 120.
- Remaining acquisition queue rows: 120.

## Access Status

| Value | Count |
| --- | ---: |
| abstract_only | 120 |
| full_text | 57 |
| official_documentation | 25 |

## Redistribution Status

| Value | Count |
| --- | ---: |
| local-restricted | 74 |
| metadata-only | 120 |
| public-redistributable | 8 |

## Acquisition Source

| Value | Count |
| --- | ---: |
| high-difficult | 14 |
| high-official | 15 |
| high-public | 24 |
| provisional-metadata | 120 |
| trusted-v7 | 28 |
| user-supplied | 1 |

## Remaining Action/Priority

| Action | Priority | Count |
| --- | --- | ---: |
| replace-mismatched-local | high | 1 |
| archive-public-full-text | medium | 22 |
| archive-public-full-text | low | 17 |
| archive-official-source | medium | 10 |
| archive-official-source | low | 23 |
| user-fetch-or-document-limitation | high | 21 |
| user-fetch-or-document-limitation | medium | 14 |
| user-fetch-or-document-limitation | low | 12 |

## Limitations

- Evidence bytes under `paper/data/source_archive/v8/` are deliberately untracked local artifacts; the manifest hashes bind the reviewed bytes but do not distribute them.
- Public and official upstream endpoints can change or disappear. Their recorded URLs and versions are provenance, not a guarantee of future availability.
- C0122 was supplied by the user and is stored locally as restricted evidence; that supply does not grant redistribution rights.
