# Pass-2 Primary Coding Snapshot v2

This no-clobber snapshot integrates the six fixed v2 primary batches against the immutable `pass2_drafts/v1` release. `batches/` retains each supplied CSV byte-for-byte; `coding/evidence.csv` is their deterministic 75-row merge sorted by `cite_key`.

The same six primary coders normalized the v1 coding under the prospective `pass2_reliability/pilot-v1/CODEBOOK-v2.md`. This is not independent or blind reliability. `bindings.csv` freezes the exact draft-release, primary-v1, and codebook artifacts used; `normalization_summary.csv` deterministically compares the v2 and primary-v1 evidence rows and cells.

`execution_registry.csv` records coordinator-supplied coder metadata and source/output digests. `manifest/checksums.csv` records every integrated batch, generated artifact, documentation record, and immutable release binding. No final counts, prevalence, or taxonomy claims may be made until fresh blind reliability is completed.
