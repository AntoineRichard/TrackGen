# V6 Screening Relevance Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace competing inclusion-subtype labels with one coordinator-bound
`include-relevant` criterion, preserve historical snapshot validation, and execute v6
calibration followed by main screening only after a passing sealed gate.

**Architecture:** New freezes bind the inclusion vocabulary in `taxonomy.json`; v6
role stages carry the same vocabulary in execution configuration version 2. Result
validation consumes the coordinator or stage binding, while legacy coordinators and
v1 stage configurations retain the historical four-value vocabulary. The result CSV
schema and all bibliographic inputs remain unchanged.

**Tech Stack:** Python 3, pytest, canonical JSON/CSV, SHA-256 snapshot manifests,
Markdown protocol, Codex Terra/high isolated reviewers.

---

### Task 1: Bind Inclusion Vocabulary In New Coordinators And Role Stages

**Files:**
- Modify: `paper/scripts/prepare_screening_batches.py`
- Modify: `tests/test_screening_batches.py`
- Modify: `tests/test_screening_integration.py`

- [ ] **Step 1: Add failing freeze and stage-configuration tests**

Update fresh-input test fixtures to add:

```python
taxonomy["screening_inclusion_criterion"] = ["include-relevant"]
```

Add `test_new_freeze_requires_screening_inclusion_criterion`, which removes the key,
calls `freeze()`, and asserts the missing-key `SnapshotError`. Add the parameterized
`test_new_freeze_rejects_invalid_screening_inclusion_criterion` with these exact invalid
values:

```python
[
    ["include-1"],
    ["include-relevant", "include-1"],
    [],
]
```

Add `test_v6_stage_binds_allowed_inclusion_criteria` with these exact assertions:

```python
assert configuration["configuration_version"] == "2"
assert configuration["allowed_inclusion_criteria"] == ["include-relevant"]
```

Add `test_historical_v5_coordinator_and_v1_stage_still_validate`, which validates the
committed v5 coordinator, stages `screening-01` from its committed calibration release
under `tmp_path`, validates the stage, and asserts configuration version `1` with no
`allowed_inclusion_criteria` key.

The historical test MUST use committed `paper/data/screening_inputs/v5` and
`paper/data/screening_releases/calibration/v5`, stage one role under `tmp_path`, and
validate the resulting v1 configuration. It MUST NOT depend on untracked staging.

- [ ] **Step 2: Run the focused tests and confirm the intended failures**

```bash
pytest -q tests/test_screening_batches.py tests/test_screening_integration.py \
  -k 'inclusion_criterion or allowed_inclusion or historical_v5'
```

Expected: new-freeze and v2-stage assertions fail because the producer neither
requires nor emits the vocabulary yet; the historical validation control passes.

- [ ] **Step 3: Implement strict-new and compatible-historical taxonomy parsing**

Add constants and a single resolver near the manifest constants:

```python
SCREENING_INCLUSION_CRITERION_KEY = "screening_inclusion_criterion"
CURRENT_INCLUSION_CRITERIA = ("include-relevant",)

def _inclusion_criteria_for_taxonomy(
    taxonomy: dict[str, list[str]], *, required: bool
) -> tuple[str, ...]:
    values = taxonomy.get(SCREENING_INCLUSION_CRITERION_KEY)
    if values is None:
        if required:
            raise SnapshotError(
                "taxonomy.json: taxonomy is missing "
                "'screening_inclusion_criterion'"
            )
        return ()
    if tuple(values) != CURRENT_INCLUSION_CRITERIA:
        raise SnapshotError(
            "taxonomy.json: screening_inclusion_criterion must be exactly "
            "['include-relevant']"
        )
    return tuple(values)
```

Thread `require_screening_inclusion_criterion` through `_parse_taxonomy()`,
`_load_source_data()`, and `build_snapshot_artifacts()`. `freeze_snapshot()` MUST call
the strict path. `validate_snapshot()` and canonical re-derivation of existing frozen
snapshots MUST use the compatible path.

- [ ] **Step 4: Emit and validate role execution configuration v2**

Extend `_execution_configuration()` with
`allowed_inclusion_criteria: tuple[str, ...] | None = None`:

```python
if allowed_inclusion_criteria is None:
    configuration["configuration_version"] = "1"
else:
    configuration["configuration_version"] = "2"
    configuration["allowed_inclusion_criteria"] = list(
        allowed_inclusion_criteria
    )
```

`stage_reviewer_execution()` MUST derive the optional tuple from the captured
coordinator taxonomy and pass it through `build_reviewer_stage_artifacts()`.
`validate_reviewer_stage_snapshot()` MUST recreate and compare v1 configurations
without the field and v2 configurations with the exact singleton field. Reject every
other configuration version or v2 value.

- [ ] **Step 5: Run producer, integration, and historical regression tests**

```bash
pytest -q tests/test_screening_batches.py tests/test_screening_integration.py
python3 -B -m paper.scripts.prepare_screening_batches \
  --snapshot-dir paper/data/screening_inputs/v5
```

Expected: all tests pass and the frozen v5 coordinator validates byte-for-byte.

- [ ] **Step 6: Commit Task 1**

```bash
git add paper/scripts/prepare_screening_batches.py \
        tests/test_screening_batches.py tests/test_screening_integration.py
git diff --cached --check
git -c commit.gpgsign=false commit \
  -m "feat: bind screening inclusion vocabulary"
```

### Task 2: Enforce Bound Inclusion Vocabulary In Result Validation

**Files:**
- Modify: `paper/scripts/screening_results.py`
- Modify: `tests/test_screening_results.py`

- [ ] **Step 1: Add failing result-validation tests**

Keep the public unbound helper legacy-compatible with
`test_unbound_decision_validation_retains_legacy_inclusion_values`, parameterized over
`include-1` through `include-4`, and
`test_unbound_decision_validation_rejects_include_relevant`.

Add these bound tests with the exact stated behavior:

- `test_v6_phase_accepts_include_relevant` seals a fresh included row using
  `include-relevant`.
- `test_v6_phase_rejects_legacy_inclusion_values` is parameterized over `include-1`
  through `include-4` and expects `ScreeningResultError`.
- `test_v2_role_result_rejects_legacy_inclusion_value` mutates one staged v2 result
  from `include-relevant` to `include-1` and expects `ScreeningResultError`.
- `test_committed_v5_calibration_snapshot_still_validates` validates the committed v5
  coordinator, release, and result tuple.

The final regression MUST validate the committed v5 coordinator, reviewer release,
and calibration results through `validate_phase_result_snapshot()`.

- [ ] **Step 2: Run the focused tests and confirm bound-v6 failures**

```bash
pytest -q tests/test_screening_results.py \
  -k 'inclusion_value or include_relevant or committed_v5 or role_result'
```

Expected: v6 bound checks fail because the validator still uses the global legacy
tuple; the public-helper and v5 controls pass.

- [ ] **Step 3: Carry inclusion vocabulary in the coordinator capture**

Retain the existing global `INCLUSION_CRITERIA` as the legacy tuple. Extend
`CoordinatorSnapshot` with:

```python
allowed_inclusion_criteria: tuple[str, ...]
```

During `_capture_coordinator()`, parse `payloads["taxonomy.json"]`; use the singleton
taxonomy value when present and `INCLUSION_CRITERIA` when absent. Include the new tuple
in `_coordinator_state_signature()` validation and equality so caller-forged state is
rejected.

- [ ] **Step 4: Pass bound values through phase and role validation**

Change the private validator signature without changing public behavior:

```diff
-def _validate_result_decision(row: Row, *, context: str) -> None:
+def _validate_result_decision(
+    row: Row,
+    *,
+    context: str,
+    allowed_inclusion_criteria: tuple[str, ...] = INCLUSION_CRITERIA,
+) -> None:
```

`_validate_phase_payloads()` MUST pass
`coordinator.allowed_inclusion_criteria` for every bound row. The public
`validate_result_decision()` wrapper omits the argument and remains legacy-compatible.

In `_validate_role_result()`, parse the already authenticated
`execution_configuration.json`. V1 uses `INCLUSION_CRITERIA`; v2 uses the exact
`allowed_inclusion_criteria` list. Pass that tuple to `_validate_result_decision()`.

- [ ] **Step 5: Run focused and complete screening-result tests**

```bash
pytest -q tests/test_screening_results.py
python3 -B -c "from pathlib import Path; from paper.scripts.screening_results import validate_phase_result_snapshot as v; v(Path('paper/data/screening_results/calibration/v5'), coordinator_snapshot_dir=Path('paper/data/screening_inputs/v5'), reviewer_release_snapshot_dir=Path('paper/data/screening_releases/calibration/v5')); print('v5 valid')"
```

Expected: all tests pass and the real v5 snapshot prints `v5 valid`.

- [ ] **Step 6: Commit Task 2**

```bash
git add paper/scripts/screening_results.py tests/test_screening_results.py
git diff --cached --check
git -c commit.gpgsign=false commit \
  -m "feat: enforce coordinator inclusion criteria"
```

### Task 3: Create The V6 Protocol And Taxonomy

**Files:**
- Modify: `paper/data/screening_protocol.md`
- Modify: `paper/data/taxonomy.json`
- Create: `paper/data/screening_work/v6/protocol.md`
- Create: `paper/data/screening_work/v6/README.md`
- Modify: `tests/test_screening_protocol.py`

- [ ] **Step 1: Write failing protocol-contract tests**

Set the expected included criterion to `{"include-relevant"}` and assert:

```python
assert criteria["include-relevant"].startswith(
    "Material evidence establishes at least one source-native eligibility rule"
)
assert "include-1" not in protocol
assert "include-2" not in protocol
assert "include-3" not in protocol
assert "include-4" not in protocol
assert pairing["included"]["Allowed criterion"] == "`include-relevant`"
assert "MUST NOT choose or rank a primary contribution subtype" in protocol
```

Also assert the root taxonomy contains exactly:

```python
["include-relevant"]
```

for `screening_inclusion_criterion`.

- [ ] **Step 2: Run protocol tests and confirm failure against v5 semantics**

```bash
pytest -q tests/test_screening_protocol.py
```

Expected: failures identify the legacy four-value inclusion table and missing taxonomy
key.

- [ ] **Step 3: Derive and edit the v6 protocol**

Copy the latest frozen protocol, not the older canonical protocol:

```bash
mkdir -p paper/data/screening_work/v6
cp paper/data/screening_inputs/v5/protocol.md \
   paper/data/screening_work/v6/protocol.md
```

Apply the approved design from
`docs/superpowers/specs/2026-07-02-screening-relevance-separation-design.md`:

- title: `# Duplicate full-text screening codebook v4`;
- replace the four-row inclusion table with one `include-relevant` row;
- express the four unchanged eligibility rules as disjunctive bullets;
- rename the precedence clarification to eligibility/boundary clarification;
- remove primary-subtype ranking and `include-1`/`include-2` precedence;
- change the decision procedure to assign `include-relevant` after any eligibility
  rule is established;
- change included status/criterion pairing to `include-relevant`;
- update glossary, adjudication, agreement, and accountable-verification prose so no
  legacy inclusion token remains; and
- preserve boundary, exclusion, access, provenance, blinding, gate, and security text.

The normative inclusion row begins exactly:

```markdown
| `include-relevant` | Material evidence establishes at least one source-native eligibility rule for course operations, generated-course artifacts or interfaces, generated-course characterization, or survey-gap synthesis. |
```

Copy the finished bytes to the canonical working protocol:

```bash
cp paper/data/screening_work/v6/protocol.md \
   paper/data/screening_protocol.md
```

- [ ] **Step 4: Add the root taxonomy key and v6 provenance README**

Add to `paper/data/taxonomy.json` using canonical JSON formatting:

```json
"screening_inclusion_criterion": [
  "include-relevant"
]
```

Create `paper/data/screening_work/v6/README.md` recording that v6 follows sealed v5
decision `38b466c04fc5e8a2938b7e8c7f84a251c636042c8525460cda62d4cbfad58c69`,
changes inclusion vocabulary only, preserves bibliographic inputs and the stable 30,
and does not expose prior ratings to v6 reviewers.

- [ ] **Step 5: Verify protocol scope and tests**

```bash
cmp paper/data/screening_protocol.md \
    paper/data/screening_work/v6/protocol.md
rg -n 'include-[1234]' paper/data/screening_protocol.md \
    paper/data/screening_work/v6/protocol.md
pytest -q tests/test_screening_protocol.py
python3 -B -m paper.scripts.validate_corpus
```

Expected: `cmp` and tests succeed; `rg` returns no matches; corpus validation succeeds.

- [ ] **Step 6: Commit Task 3**

```bash
git add paper/data/screening_protocol.md paper/data/taxonomy.json \
        paper/data/screening_work/v6 tests/test_screening_protocol.py
git diff --cached --check
git -c commit.gpgsign=false commit \
  -m "docs: separate screening relevance from contribution type"
```

### Task 4: Freeze And Review The V6 Coordinator

**Files:**
- Create: `paper/data/screening_inputs/v6/`
- Modify: `paper/data/screening_work/v6/README.md`

- [ ] **Step 1: Run the complete focused screening suite**

```bash
pytest -q tests/test_screening_batches.py tests/test_screening_results.py \
  tests/test_screening_integration.py tests/test_screening_protocol.py
```

Expected: all tests pass before freezing scientific artifacts.

- [ ] **Step 2: Freeze v6 from unchanged v5 bibliographic inputs**

```bash
python3 -B -m paper.scripts.prepare_screening_batches --freeze \
  --candidates paper/data/screening_inputs/v5/candidates.csv \
  --conflicts paper/data/screening_inputs/v5/conflicts.csv \
  --bibliography paper/data/screening_inputs/v5/bibliography.csv \
  --citation-keys paper/data/screening_inputs/v5/citation_keys.csv \
  --taxonomy paper/data/taxonomy.json \
  --protocol paper/data/screening_work/v6/protocol.md \
  --execution-profile paper/data/screening_inputs/v5/execution_profile.json \
  --reviewer-prompt-template paper/data/screening_inputs/v5/reviewer_prompt_template.md \
  --output-dir paper/data/screening_inputs/v6
```

- [ ] **Step 3: Validate stable selection and changed-only inputs**

```bash
python3 -B -m paper.scripts.prepare_screening_batches \
  --snapshot-dir paper/data/screening_inputs/v6
cmp paper/data/screening_inputs/v5/calibration_selection.csv \
    paper/data/screening_inputs/v6/calibration_selection.csv
cmp paper/data/screening_inputs/v5/candidates.csv \
    paper/data/screening_inputs/v6/candidates.csv
cmp paper/data/screening_inputs/v5/bibliography.csv \
    paper/data/screening_inputs/v6/bibliography.csv
```

Expected: 404 assignments; stable 30 and bibliographic inputs are byte-identical.

- [ ] **Step 4: Run independent spec and quality reviews**

Dispatch a fresh spec reviewer to compare v6 coordinator/protocol against the approved
design and this plan. After approval, dispatch a separate quality reviewer to inspect
historical compatibility, canonical hashes, test evidence, and scope. Resolve every
Critical or Important issue and repeat the corresponding review.

- [ ] **Step 5: Commit Task 4**

```bash
git add paper/data/screening_inputs/v6 \
        paper/data/screening_work/v6/README.md
git diff --cached --check
git -c commit.gpgsign=false commit -m "docs: freeze v6 screening coordinator"
```

### Task 5: Execute And Seal Fresh V6 Calibration

**Files:**
- Create: `paper/data/screening_releases/calibration/v6/`
- Create locally: `paper/data/screening_staging/v6/calibration/`
- Create: `paper/data/screening_results/calibration/v6/`
- Create: `paper/data/screening_decisions/v6/`
- Modify: `paper/data/screening_work/v6/README.md`

- [ ] **Step 1: Publish release and six role-private stages**

```bash
mkdir -p -m 750 paper/data/screening_staging/v6/calibration
python3 -B -m paper.scripts.prepare_screening_batches --release \
  --snapshot-dir paper/data/screening_inputs/v6 --phase calibration \
  --output-dir paper/data/screening_releases/calibration/v6
```

Run `--stage-role` once for each `screening-01` through `screening-06`, binding the v6
coordinator and calibration release. Verify each execution configuration is v2 and
contains only `include-relevant`.

- [ ] **Step 2: Dispatch six fresh blind Terra/high reviewers**

Start six new contexts with `fork_context=false`. Deliver each complete rendered
`reviewer_prompt.md` as the exact user message with no wrapper. Do not supply v3-v5
ratings, discussions, or decisions. Record role, context ID, stage, start/completion
date, result path, and SHA-256 contemporaneously in the v6 README. Require every
result to pass `--validate-role-result`.

- [ ] **Step 3: Seal all 60 calibration ratings**

```bash
python3 -B -m paper.scripts.screening_results --seal-phase \
  --coordinator-snapshot paper/data/screening_inputs/v6 \
  --reviewer-release-snapshot paper/data/screening_releases/calibration/v6 \
  --phase calibration \
  --result "$RESULT_01" --result "$RESULT_02" --result "$RESULT_03" \
  --result "$RESULT_04" --result "$RESULT_05" --result "$RESULT_06" \
  --output-dir paper/data/screening_results/calibration/v6
```

- [ ] **Step 4: Derive and seal the gate**

Compute exact status and criterion agreement from locked rows. Inspect every
disagreement without editing ratings. Seal `release` only when status agreement is at
least 0.80, systematic ambiguity is false, all 60 ratings validate, and every binding
is complete. Otherwise seal `revise` and stop before Task 6.

```bash
python3 -B -m paper.scripts.screening_results --seal-calibration-decision \
  --coordinator-snapshot paper/data/screening_inputs/v6 \
  --reviewer-release-snapshot paper/data/screening_releases/calibration/v6 \
  --calibration-result-snapshot paper/data/screening_results/calibration/v6 \
  --decision-input paper/data/screening_work/v6/calibration_decision.csv \
  --output-dir paper/data/screening_decisions/v6
```

- [ ] **Step 5: Review, verify, and commit calibration**

Run public snapshot validators, all three `SHA256SUMS` files, role-result tests, and
independent spec then quality review. Commit only release, sealed results, sealed
decision, decision input, and README. Keep staging untracked.

```bash
git add paper/data/screening_releases/calibration/v6 \
        paper/data/screening_results/calibration/v6 \
        paper/data/screening_decisions/v6 \
        paper/data/screening_work/v6
git diff --cached --check
git -c commit.gpgsign=false commit -m "docs: seal v6 calibration gate"
```

### Task 6: Execute Main Screening Only After A Passing V6 Gate

**Files:**
- Create: `paper/data/screening_releases/main/v6/`
- Create locally: `paper/data/screening_staging/v6/main/`
- Create: `paper/data/screening_results/main/v6/`
- Modify: `paper/data/screening_work/v6/README.md`

- [ ] **Step 1: Publish a gate-bound main release**

```bash
python3 -B -m paper.scripts.prepare_screening_batches --release \
  --snapshot-dir paper/data/screening_inputs/v6 --phase main \
  --calibration-reviewer-release-snapshot paper/data/screening_releases/calibration/v6 \
  --calibration-result-snapshot paper/data/screening_results/calibration/v6 \
  --calibration-decision-snapshot paper/data/screening_decisions/v6 \
  --output-dir paper/data/screening_releases/main/v6
```

Expected: the command succeeds only for a coherent sealed `release` decision.

- [ ] **Step 2: Stage and execute six fresh main contexts**

Create six v2 role stages. Start six fresh Terra/high contexts with
`fork_context=false`, exact rendered-prompt delivery, and no calibration ratings or
discussion. Validate each result locally and record all execution bindings in the v6
README. The six canonical files MUST total 344 rows.

- [ ] **Step 3: Seal and validate main results**

```bash
python3 -B -m paper.scripts.screening_results --seal-phase \
  --coordinator-snapshot paper/data/screening_inputs/v6 \
  --reviewer-release-snapshot paper/data/screening_releases/main/v6 \
  --phase main \
  --calibration-reviewer-release-snapshot paper/data/screening_releases/calibration/v6 \
  --calibration-result-snapshot paper/data/screening_results/calibration/v6 \
  --calibration-decision-snapshot paper/data/screening_decisions/v6 \
  --result "$RESULT_01" --result "$RESULT_02" --result "$RESULT_03" \
  --result "$RESULT_04" --result "$RESULT_05" --result "$RESULT_06" \
  --output-dir paper/data/screening_results/main/v6
```

Verify 344 rows, all checksums, public snapshot validation, and absence of calibration
records in main outputs.

- [ ] **Step 4: Independently review and commit main screening**

After spec and quality approval:

```bash
git add paper/data/screening_releases/main/v6 \
        paper/data/screening_results/main/v6 \
        paper/data/screening_work/v6/README.md
git diff --cached --check
git -c commit.gpgsign=false commit -m "docs: seal v6 main screening ratings"
```

- [ ] **Step 5: Hand off to a separate adjudication plan**

Create a new design/plan for v6 disagreement adjudication and evidence extraction.
Use only sealed v6 snapshots. Do not import failed v3-v5 ratings into final decisions.
