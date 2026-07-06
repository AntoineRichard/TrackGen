# Pass-2 Reliability Pilot v1

This no-clobber snapshot preserves the completed 18-source Pass-2 reliability pilot and its follow-up adjudication as supplied. All files in `inputs/` are byte-for-byte copies of the coordinator inputs. `execution_registry.csv` records the supplied reliability-coder, source-adjudicator, and methods-reviewer metadata. `bindings.csv` binds this record to `pass2_coding/primary/v1` and the non-final `pass2_drafts/v1` release; neither bound release is modified.

## Pilot outcome

The pilot failed all four required exact-set gates:

- `representation_family: 8/18 (0.444) - FAIL`
- `generator_family: 12/18 (0.667) - FAIL`
- `generation_role: 8/18 (0.444) - FAIL`
- `validity_strategy: 9/18 (0.500) - FAIL`

The pilot therefore cannot support final prevalence/taxonomy claims. The passed diagnostic fields do not change this conclusion. `PROCEDURAL-LIMITATIONS.md` records the restrictions; `CODEBOOK-v2.md` is prospective and contains no source-specific answer keys.

## Integrity

`manifest/checksums.csv` records SHA-256 hashes and row counts for each copied input and generated artifact. `SHA256SUMS` additionally checks the manifest itself. Run `python paper/scripts/integrate_pass2_reliability.py --repository-root . --validate --snapshot paper/data/screening_work/v8/pass2_reliability/pilot-v1 --input-root /tmp` from the repository root to validate this record.
