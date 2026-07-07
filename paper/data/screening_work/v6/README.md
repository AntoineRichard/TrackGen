# Screening Codebook v6

Version 6 follows the sealed v5 decision
`38b466c04fc5e8a2938b7e8c7f84a251c636042c8525460cda62d4cbfad58c69`.

This is an inclusion-vocabulary-only change: every eligible source is recorded as
`included`,`include-relevant`, while the four source-native eligibility rules remain
disjunctive and contribution characterization moves to downstream evidence extraction.
Bibliographic inputs are unchanged, and the stable calibration set remains 30 records.
No prior ratings were supplied to v6 reviewers.

## Frozen coordinator

The v6 coordinator snapshot is
`d91b44f278376601725a8290d24c75326d02808513f3e6c40f77b89493d6f14f`.
Its internal `SHA256SUMS` ledger is checksum-valid.

| Bound input | SHA-256 |
| --- | --- |
| Protocol | `30ac1cbcec55e99d9fa32843c435427c2c72efb01772e956a03d8f13613da32e` |
| Taxonomy | `9405d26139253a6d61cf489eec515bbede721c3229c893936ac964fa41ff2bed` |
| Reviewer prompt template | `847af75b839557d4006e83dea5d096ef36f4acc3f9745c2f2a5178ec608af35b` |

The manifest contains 404 assignments: 60 calibration and 344 main. It covers 202
candidates, each assigned exactly twice, across six batches. The 30-record stable
calibration selection is byte-identical to v5. The v5 candidates, conflicts,
bibliography, citation keys, execution profile, reviewer prompt template, and
calibration selection are also byte-identical to v6; only the frozen protocol and
taxonomy differ among the coordinator source inputs.

The focused cross-component gate recorded `42 passed, 502 deselected`. After the v6
release-fixture correction, the full producer suite (`tests/test_screening_batches.py`)
passed 133 tests. The full result suite passed 189 tests in Task 2, and the protocol
suite passed 30 tests. V5 coordinator/result validation and corpus validation also
passed. The pre-existing execution-register integration contract mismatch remains
outside v6 scope; the full integration module is not claimed passing.

## Calibration executions

| Role | Agent context | Selected stage | Selected result | Rows | Completion SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `screening-01` | `019f226a-8d01-7221-81da-cf3f7c923145` | `screening_staging/v6/calibration/screening-01-ce9002d8f4a75081684c56d3e2312f30/v1` | `screening_staging/v6/calibration/screening-01-ce9002d8f4a75081684c56d3e2312f30/screening-01-result.csv` | 8 | `03ef549c841797b160ddfb42c67abe14e490cfa61cd2589796d8c9556b7c4cfd` |
| `screening-02` | `019f226a-8d32-7f60-8e3b-951f0ec81dd2` | `screening_staging/v6/calibration/screening-02-027038924382445d5232abadba758527/v1` | `screening_staging/v6/calibration/screening-02-027038924382445d5232abadba758527/screening-02-result.csv` | 9 | `27c2e612711e14febc3a9fd3fd3e36efebcf38a3a4578c2b98450b76b7e6cd94` |
| `screening-03` | `019f226a-8d88-7a53-8964-4a7c63fdb061` | `screening_staging/v6/calibration/screening-03-894fdcab536201d33c92dbf11a4dd715/v1` | `screening_staging/v6/calibration/screening-03-894fdcab536201d33c92dbf11a4dd715/screening-03-result.csv` | 7 | `6f616afbcb189e24be1e1e8cf229eec726b1ab2846aa42b3459e67f385088dfb` |
| `screening-04` | `019f226a-8dd6-7c12-b1c1-408cb9af1c8e` | `screening_staging/v6/calibration/screening-04-d725e979eea86fd73386e2dc55c7160d/v1` | `screening_staging/v6/calibration/screening-04-d725e979eea86fd73386e2dc55c7160d/screening-04-result.csv` | 11 | `5112ae2bb2aa0e75e7761895c8cceee1cde68d9ce5213b8647df37392d8e16b0` |
| `screening-05` | `019f226a-8e36-7de2-b333-ad0868d35640` | `screening_staging/v6/calibration/screening-05-205db1cbffd6df7147d5d44c2c39d920/v1` | `screening_staging/v6/calibration/screening-05-205db1cbffd6df7147d5d44c2c39d920/screening-05-result.csv` | 13 | `7401a5d7439c4a8fc69cb861017200cd7e96c12f3fc6f423bbf64f48d978ebd6` |
| `screening-06` | `019f226a-8e82-72e3-9b97-6ad75d5c393f` | `screening_staging/v6/calibration/screening-06-b0df3abeb6dc98fd1362d5b378810da3/v1` | `screening_staging/v6/calibration/screening-06-b0df3abeb6dc98fd1362d5b378810da3/screening-06-result.csv` | 12 | `1500f450a45c69e09cc02549d1e808c4dd062023e32c6157e358b5da62cb1dc6` |

All six contexts started and completed on 2026-07-02. Each received its complete
rendered `reviewer_prompt.md` as the sole user message in a fresh Terra/high context.
Every closed result passed the role-local v2 validator, and each reported hash matched
the selected result bytes. These bindings identify automated contexts, not people.

## Calibration outcome

The authoritative calibration result snapshot is
`78113e4b2bae7e28c8a21f63e15f834978ee2334cf0658df4ac26a7a8fb98453`.
All 60 assigned ratings were sealed. Status agreement was 21/30 (`0.700000`), and
criterion agreement was 19/30 (`0.633333`). Every included rating used
`include-relevant`.

Five records, C0140, C0168, C0175, C0180, and C0198, repeatedly split between
`included` and `boundary` at the source-native course operation or artifact versus
fixed-course boundary-transfer rule. C0088 and C0147 also repeated the boundary versus
insufficient-evidence rule boundary. Under the frozen v6 ambiguity rule, systematic
ambiguity is true. The required calibration decision is `revise`; no v6 main-phase
release is permitted. The sealed calibration decision snapshot is
`1a8741e9ee36c3c23d8464c454bc559d7aa67d5c1106e468f67799c44d83dba7`.
