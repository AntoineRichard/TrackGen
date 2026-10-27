# PR4 finding #4 — independent final review

Verdict: **approve the scoped implementation; no actionable findings**. Release/push gates remain conditional on the executor's exact-commit validation; this review does not claim those gates have completed.

Reviewed the uncommitted finding-only changes against `8b1e9c3a60da8e544e1b66aacad4a5ee363db2a8`, the approved automatic-spacing design/plan, AGENTS authorization, and the private progress ledger. The inspection covered all changed source/tests/docs and the unchanged config snapshot, generator binding, stage conversion, parameter mapping, and numeric-reader paths. No source files were edited and no expensive suite was run.

## Contract and implementation

- `TrackGenConfig.spacing` retains the supplied optional field. The read-only property returns a Python float, while frozen dataclass replacement naturally preserves automatic versus explicit intent. No hidden state or post-init materialization remains.
- The specialized `ConstantSpacingParams.from_legacy` resolves spacing at the existing boundary. Specs, planner consumers and stage launch scalars therefore remain concrete. Other flat Params retain their existing mapping behavior.
- Generator binding deep-copies both user and runtime configs; the public config snapshot retains `None`. No built-in generator Params overlays `half_width` or `spacing`, so delayed resolution does not change existing built-in binding arithmetic.
- Frozen subclasses retain their class/additional fields under replacement. `asdict` and JSON reconstruction retain automatic intent because the property is not a dataclass field. Serialized pipeline specs intentionally capture a numeric step, as documented.
- Explicit spacing retains the prior positive-value validation. Existing half-width validation still applies.

## Numerical compatibility judgment

`float(0.6 * self.half_width)` is the correct compatibility-preserving choice. The baseline materialized `0.6 * self.half_width`, and the production stage later converted that value to `float`. Moving conversion ahead of multiplication can change NumPy float32 rounding, particularly with NumPy 2 scalar-promotion semantics. The implemented order retains the original operation and final scalar conversion. The regression compares exact scalar values rather than an approximation and checks the stage boundary too. The ledger explicitly records why this differs from the design's literal formula spelling; it is consistent with the stronger unchanged-output requirement.

Reference/oracle readers change only the numeric accessor. The storyboard uses the same resolved scalar for replay and metadata. An audit found no remaining active config numeric reader using the optional `.spacing` field; remaining `.spacing` accesses belong to concrete Params, unrelated utility instances, CLI arguments, or tests of caller intent.

## Tests, scope, and remaining evidence

New regressions cover automatic subclass replacement, explicit replacement/reset, JSON reconstruction, concrete Params/spec serialization, NumPy scalar rounding, and byte comparisons of all public result arrays across initial and masked advanced generation. The generation regression preserves config intent and includes CPU/CUDA device parametrization. Existing config/type, oracle and explorer assertions were adapted appropriately.

Current guidance, migration guidance, and changelog describe the changed contract accurately. Old candidate notes remain historical. The reviewed diff does not change solver/kernel arithmetic, RNG, defaults, goldens, package version, media, or tags. `git diff --check 8b1e9c3` completed successfully.

The root executor still owns full CPU/docs, fresh same-machine immutable-rc3 full/gate snapshots, CUDA/full Spark, and exact-SHA CI evidence. Those are necessary to substantiate global byte compatibility beyond the reviewed narrow regression, and should not be inferred from this static review.

## Executor evidence at source commit

- Baseline config/types: 87 passed in 3.32 s. New contract regressions failed on the baseline (5 failed, one explicit control passed). Real automatic-width geometry regression failed while its explicit control passed. The NumPy scalar-rounding regression failed before preserving multiplication order.
- Targeted config/types/planner/presets/inflation/storyboard checks: 200 passed, four CUDA cases deselected, 6.89 s.
- Strict Sphinx build with the default 82-pair media registry: passed, 16.16 s. No media was regenerated or relabeled.
- Self-review: checked all active config numeric readers, the frozen reference accessor-only adaptation, config snapshot intent, stage/spec concrete values, scalar rounding and untouched numerical code/assets/goldens. No additional issue found.
- Full local CPU is running. Fresh immutable-rc3 track/gate baselines have been rebuilt successfully on Spark. The source is committed locally before those long gates finish so exact committed source can be exported; it will only be pushed after CPU, same-machine byte comparisons and CUDA/full Spark pass. Final validation receipts and the validated SHA belong in the PR review thread, keeping this source commit immutable.
