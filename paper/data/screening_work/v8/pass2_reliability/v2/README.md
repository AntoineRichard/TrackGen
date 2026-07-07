# Pass-2 Reliability v2

This no-clobber snapshot freezes the supplied fresh 18-source reliability round. The seven files in `inputs/` are byte-for-byte coordinator inputs. `disagreements.csv` is deterministically derived from the locked primary and reliability rows, and `bindings.csv` freezes primary/v2, the draft release, and pilot-v1 `CODEBOOK-v2.md`; no bound artifact is modified.

## Outcome

The round passed `survey_evidence_tier` (0.888889) and `asset_status` (0.888889), but failed `course_object` (0.666667), `representation_family` (0.500000), `generator_family` (0.611111), `generation_role` (0.666667), `validity_strategy` (0.500000), and `code_status` (0.500000).

This second failed reliability round triggers stopping label-by-label codebook exception iteration. The record must not adjudicate to manufacture agreement. `PROCEDURAL-LIMITATIONS.md` preserves the resulting claim limits and future structural redesign recommendation.

## Integrity

`manifest/checksums.csv` records SHA-256 hashes and row counts for every copied input and generated record. `SHA256SUMS` checks the manifest itself. Run `python paper/scripts/integrate_pass2_reliability.py --repository-root . --input-root /tmp --version v2 --validate --snapshot paper/data/screening_work/v8/pass2_reliability/v2` from the repository root to validate this record.
