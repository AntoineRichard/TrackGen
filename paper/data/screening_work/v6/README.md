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

The v6 coordinator validator passes, and the preserved v5 historical coordinator and
result snapshots still validate. The parent focused cross-component gate recorded `42 passed,
502 deselected`; v5 coordinator/result validation and corpus validation also passed.
The known unrelated full integration contract test was not rerun for this freeze.
