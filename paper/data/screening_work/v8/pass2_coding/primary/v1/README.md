# Pass-2 Primary Coding Snapshot

This no-clobber snapshot integrates the six completed primary Pass-2 coding batches against the immutable `pass2_drafts/v1` release. `batches/` retains the supplied CSV files byte-for-byte; `coding/evidence.csv` is their deterministic 75-row merge sorted by `cite_key`.

`execution_registry.csv` records the coordinator-supplied roles, agent identifiers, model, reasoning effort, `human_role=NR`, row counts, and source/output digests. `manifest/checksums.csv` binds those outputs, documentation, and the immutable release manifest and `SHA256SUMS`.

See `PROCEDURAL-LIMITATIONS.md` for the provenance and non-final-use limits.
