# Corrected Reratings v1

This append-only corrected-rerating sidecar records corrected packet version 1 for C0046 and C0173. It preserves the frozen v8 ratings byte-for-byte and records four fresh isolated duplicate reratings against corrected evidence bytes.

The copied JSON files in `inputs/` are the coordinator-supplied raw ratings. `ratings.csv` preserves every scientific field while normalizing the auxiliary `retention_role` to `core`. Both duplicate pairs agree on `included`, `include-relevant`, `full_text`, and the corrected evidence digest.

This is not a sealed primary, adjudication, or projection snapshot. It does not alter frozen v8 releases, results, or adjudication drafts, and it is not a final integration record.
