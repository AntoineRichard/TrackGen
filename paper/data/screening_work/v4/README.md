# Screening Rerun v4

Version 4 is the corrected Terra/high screening rerun. Its frozen coordinator digest is
`d3a8cf24f8f8f646724f321cfbd943e35bdb11dcb1da9e1259348598892142ce`.
Relative to v3, the reviewer prompt adds a role-local structural validation command.
Each worker receives the exact rendered `reviewer_prompt.md` bytes as its sole user
message; no wrapper or prior conversation context is supplied.

## Calibration executions

| Role | Agent context | Selected stage |
| --- | --- | --- |
| `screening-01` | `019f1e83-e054-79e2-bed8-154ad471c98a` | `screening_staging/v4/calibration/screening-01-506410130a08fb5035bcbf091a979cc9/v1` |
| `screening-02` | `019f1e84-07a9-77c2-ad67-75c063eee40c` | `screening_staging/v4/calibration/screening-02-a18a3ed97098e2eccba8a1d059704e0d/v1` |
| `screening-03` | `019f1e84-24cf-71d3-98cf-5ba8314ba416` | `screening_staging/v4/calibration/screening-03-c14c492927ec241586d04776d487e7a6/v1` |
| `screening-04` | `019f1e84-46db-72c3-a372-4e264766811f` | `screening_staging/v4/calibration/screening-04-b50ed802386df91583e8a3c6f3df8aac/v1` |
| `screening-05` | `019f1e84-7aef-7e40-b236-e33697a00c51` | `screening_staging/v4/calibration/screening-05-142650fcb5c34f91e65ab8c7dcb6aaf7/v1` |
| `screening-06` | `019f1e84-9244-73d3-a916-5eb9367829c9` | `screening_staging/v4/calibration/screening-06-2531d4f82d458691424c372e2587de8b/v1` |

These bindings identify automated contexts, not people. The provider does not expose
backend model-version bytes, hidden instruction bytes, decoding parameters, or
retrieval-cache isolation; the frozen execution profile records those limitations.

## Calibration outcome

All 60 ratings passed role-local validation and were sealed under calibration result
snapshot `ee01c6e34192aa0f2afd99ec2176e61b02c35d2e9cf63c0718a580edeeb304ae`.
Exact status agreement was 24/30 (`0.800000`) and exact criterion agreement was
20/30 (`0.666667`). C0147, C0168, and C0172 independently repeated the same
`include-1` versus `include-2` precedence disagreement. Under the frozen protocol this
is systematic ambiguity, so decision snapshot
`bf3e8b5c444c0cf1ddb1007bb608bfb6d71cf37c29243e782ef18ab90e2a9097`
records `revise`. No v4 main release is authorized.
