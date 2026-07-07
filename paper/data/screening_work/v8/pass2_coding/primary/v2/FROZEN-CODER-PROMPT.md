# Frozen Coder Prompt

This records the common instructions supplied to all six primary coding batches. No prelaunch timestamp is asserted.

Code only with `DRAFT_C####` keys. The evidence template uses the exact `evidence.csv` header and intentionally has no `coder_id` column.

- `survey_evidence_tier` is scalar. When evidence could support multiple tiers, use `core`, then `supporting`, then `contextual` precedence.
- Controlled multi-label fields use semicolon-separated labels in the order listed by `paper/data/taxonomy.json`; the first `domain` label is the primary domain because reliability sampling stratifies on it.
- Every non-`NR` analytical field requires a source-native, field-addressable locator in `evidence_locator`, written as `field_name=locator` entries separated by semicolons. Use page, section, table, figure, algorithm, appendix, repository path and lines, or stable documentation anchors.
- `supporting` rows may only state fixed-course properties that are directly established by the source and mapped by the protocol. `contextual` rows may only support field, terminology, or literature-gap context. Neither tier establishes a source-native course-generation method.
- `asset_status` is prospectively controlled with the same scalar vocabulary as `code_status`: `official_open`, `unofficial_open`, `closed`, `not_found`, or `not_applicable`.

Mutable coding output uses `evidence.csv` with exactly these 75 draft keys. `claims.csv`, `metrics.csv`, and `simulators.csv` are optional until populated, but each must use its exact release template header and may reference only draft keys. Validate an output directory with `validate_pass2_draft.py --coding-output`; this checks output only and never rewrites the immutable release or its checksums.

Leave all analytical fields blank while a row remains a template row.

This v2 normalization applies the prospective `pilot-v1/CODEBOOK-v2.md` without changing the immutable draft release.
