# V7 screening work

v7 has a frozen coordinator at `paper/data/screening_inputs/v7`. It binds the binary
retention contract to 404 assignments while retaining the unchanged v6 bibliographic
corpus. A calibration release exists at `paper/data/screening_releases/calibration/v7`,
but no sealed calibration result, decision, or main release exists.

The initial v7 reviewer attempt is unsealed and supplies no survey result. One role
(`screening-01`) passed the role-result validator. The other original role outputs were
rejected for malformed CSV, duplicate dispatch, or inadequate evidence locators. These
private staging outputs remain untracked procedural traces and must not be repaired,
merged, or used for screening decisions.

The calibration evidence inventory is complete. Evidence is bound to immutable phase
releases, so duplicate reviewers receive the same frozen packet. Any stronger
eligibility evidence found after release creation requires a new packet version.

A fresh stable-30 calibration is required before a main release. It uses six blind
reviewer contexts and requires agreement >= 0.80, no systematic ambiguity, and 60 valid
ratings. Reviewers do not receive v3-v6 ratings or disagreements.

No v7 main release exists. A next-version calibration procedure is required before
main screening may proceed.
