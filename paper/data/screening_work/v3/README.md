# Screening Rerun v3: Failed Calibration Attempt

Version 3 is retained as an unsealed procedural failure record. It MUST NOT be used as
a screening decision source, calibration gate, or input to a publication projection.
No `screening_results/calibration/v3` snapshot was published.

The v3 coordinator changed only the execution profile from the v2 requested Sol/high
configuration to the user-approved Terra/high configuration. Its coordinator snapshot
digest is `a57cb89fd689d156d04c0b7ec8c95a90edeb0d360589ea01cb39ce04c82e4d4e`.

The first launch partially succeeded after the orchestration service reported its
thread limit. Two `screening-01` contexts then wrote the same intended output path and
reported different hashes. That role output is contaminated and was never selected.
Fresh stages were created for affected roles; all superseded stages and outputs remain
local for audit rather than being rewritten or treated as ratings.

The sealing validator rejected the selected role-01 output because one locator lacked
a page, section, table, figure, algorithm, appendix, or stable anchor. Pre-validation
then found equivalent structural failures in roles 05 and 06, including insufficient
`abstract_only` limitation notes. Replacement outputs were not promoted.

The controller also identified that workers received a wrapper directing them to read
`reviewer_prompt.md`, rather than receiving the exact rendered prompt bytes as their
visible user instruction. Consequently the frozen `user_instruction_sha256` contract
did not describe actual delivery. The attempt was stopped before calibration sealing.
A successor version must deliver the exact rendered prompt bytes and provide a
role-local structural result validator before completion.
