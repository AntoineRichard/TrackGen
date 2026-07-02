# Screening Codebook v5

Version 5 is a substantive revision following sealed v4 decision
`bf3e8b5c444c0cf1ddb1007bb608bfb6d71cf37c29243e782ef18ab90e2a9097`.
Only `include-1`/`include-2` precedence changed. The stable 30 and bibliography are
unchanged. v4 ratings are not supplied.

## Calibration executions

| Role | Agent context | Selected stage |
| --- | --- | --- |
| `screening-01` | `019f21de-a06d-7883-817d-060e09c084a4` | `screening_staging/v5/calibration/screening-01-d1134376bdc126d95e27b842c28d9599/v1` |
| `screening-02` | `019f21de-bf8f-79e1-a158-9b2e8e11c9d5` | `screening_staging/v5/calibration/screening-02-abb5db6e779651e717b960726ed41ada/v1` |
| `screening-03` | `019f21de-d853-7212-becb-4780f117ad51` | `screening_staging/v5/calibration/screening-03-44c4c8c7e989c65894d3e568bc94f76f/v1` |
| `screening-04` | `019f21e9-b7cb-7f70-bbe1-280250b40e51` | `screening_staging/v5/calibration/screening-04-b5a0a90b3799098bd998b153d7b204e5/v1` |
| `screening-05` | `019f21df-39dc-7b60-b19d-7f7a2893f853` | `screening_staging/v5/calibration/screening-05-f74d17ae1f27b6ab9ed643745a20d696/v1` |
| `screening-06` | `019f21df-71fa-7602-b79a-fcef34496c0b` | `screening_staging/v5/calibration/screening-06-e5e9ec40c6020a4a7df37a47823b01d0/v1` |


| Role | Started | Completed | Selected result | Completion SHA-256 |
| --- | --- | --- | --- | --- |
| `screening-01` | 2026-07-02 | 2026-07-02 | `screening_staging/v5/calibration/screening-01-d1134376bdc126d95e27b842c28d9599/screening-01-result.csv` | `0cc173362042de5be78f6fd809bac1a243d82593734b774b8240e067bdae3bbd` |
| `screening-02` | 2026-07-02 | 2026-07-02 | `screening_staging/v5/calibration/screening-02-abb5db6e779651e717b960726ed41ada/screening-02-result.csv` | `89da36f4e17f2756c07f050bdaceb821d72e1b52632c6cbb2c9bd8d860992997` |
| `screening-03` | 2026-07-02 | 2026-07-02 | `screening_staging/v5/calibration/screening-03-44c4c8c7e989c65894d3e568bc94f76f/screening-03-result.csv` | `2b1d0db9080937df298b8b59b3faef5d06e1773a7e8b298108e39b7d14bebb84` |
| `screening-04` | 2026-07-02 | 2026-07-02 | `screening_staging/v5/calibration/screening-04-b5a0a90b3799098bd998b153d7b204e5/screening-04-result.csv` | `372d5ca0219ab3ce8e55725eb7de6e55d3fe2a9260e0f2d18de388b8d4ede5c4` |
| `screening-05` | 2026-07-02 | 2026-07-02 | `screening_staging/v5/calibration/screening-05-f74d17ae1f27b6ab9ed643745a20d696/screening-05-result.csv` | `78297fef270a23245847254cbeb45b4c84e32ca8a3e075757dc2d327a38736f5` |
| `screening-06` | 2026-07-02 | 2026-07-02 | `screening_staging/v5/calibration/screening-06-e5e9ec40c6020a4a7df37a47823b01d0/screening-06-result.csv` | `a3e5d74248b1a83be19d5f27dc64fa66d9f28c0f235c85834c7090d6f6e1e8fc` |

Each worker received the exact rendered `reviewer_prompt.md` bytes as its sole user
message in a fresh context. These bindings identify automated contexts, not people.

The first v5 `screening-04` execution (`019f21de-f0eb-7412-8566-7e1e97427b54`)
produced a structurally valid 11-row file but reported `ROWS_WRITTEN=12` in its
completion record. That execution and stage are superseded and are not sealing inputs.
The selected replacement reported `ROWS_WRITTEN=10`, while its immutable stage,
canonical CSV parse, and role-local validator all establish exactly 11 assigned and
written rows; its reported output hash matches the closed file. Completion prose is
therefore retained as a non-authoritative orchestration-log defect. The stage manifest,
validated CSV, and sealed phase manifest are authoritative for row count and content.

## Calibration outcome

The authoritative calibration result snapshot is
`eb5e3cd18b8cedf21eaba5aae1450ffa96d3091f6b411b18a042d8a9854c656c`.
All 60 assigned ratings were sealed. Status agreement was 24/30 (`0.800000`), and
criterion agreement was 18/30 (`0.600000`).

The same `include-1` versus `include-2` operational boundary recurred for C0025,
C0172, and C0175. Under the frozen v5 ambiguity rule, this is systematic ambiguity.
The calibration decision is therefore `revise`, and no v5 main-phase release is
permitted. A substantive protocol revision and fresh blind calibration are required
before main screening.
