# V7 Retention And Contribution Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement binary v7 retention screening and a separately coded source-evidence tier while preserving byte-valid v1-v6 screening history.

**Architecture:** New coordinators bind a result-only status vocabulary from taxonomy; captured coordinators, role stages, agreement reports, and adjudication consume that binding. The corpus keeps its broader lifecycle status vocabulary. Contribution coding adds one scalar evidence tier and reliability field, with `NR` permitted only as the existing uncoded sentinel for pre-v7 drafts.

**Tech Stack:** Python 3, pytest, canonical JSON/CSV, SHA-256 snapshot manifests, Markdown protocols.

---

### Task 1: Bind Binary Result Statuses In New Coordinators

**Files:**
- Modify: `paper/scripts/prepare_screening_batches.py`
- Modify: `tests/test_screening_batches.py`
- Modify: `tests/test_screening_integration.py`

- [ ] **Step 1: Write failing status-vocabulary tests**

Add fresh-freeze tests that delete or mutate `screening_result_status`, plus a committed-history test validating coordinators v1-v6:

```python
@pytest.mark.parametrize(
    "statuses",
    [None, [], ["included"], ["included", "boundary", "excluded"]],
)
def test_new_freeze_requires_exact_binary_result_statuses(tmp_path, statuses):
    inputs = build_inputs(tmp_path / "inputs")
    taxonomy = json.loads(inputs.taxonomy.read_text())
    if statuses is None:
        taxonomy.pop("screening_result_status", None)
    else:
        taxonomy["screening_result_status"] = statuses
    inputs.taxonomy.write_bytes(_canonical_json_bytes(taxonomy))
    with pytest.raises(SnapshotError, match="screening_result_status"):
        freeze(inputs, tmp_path / "v1")
```

- [ ] **Step 2: Confirm the new tests fail**

```bash
pytest -q tests/test_screening_batches.py tests/test_screening_integration.py \
  -k 'binary_result_statuses or historical_coordinators or allowed_screening'
```

Expected: fresh-v7 requirements fail; historical validation passes.

- [ ] **Step 3: Implement strict-new and compatible-historical parsing**

Add beside the inclusion resolver:

```python
SCREENING_RESULT_STATUS_KEY = "screening_result_status"
CURRENT_SCREENING_RESULT_STATUSES = ("included", "excluded")

def _resolve_screening_result_statuses(
    taxonomy: dict[str, list[str]], *, strict_new: bool
) -> tuple[str, ...] | None:
    values = taxonomy.get(SCREENING_RESULT_STATUS_KEY)
    if values is None:
        if strict_new:
            raise SnapshotError(
                "taxonomy.json: taxonomy is missing 'screening_result_status'"
            )
        return None
    if tuple(values) != CURRENT_SCREENING_RESULT_STATUSES:
        raise SnapshotError(
            "taxonomy.json: screening_result_status must equal "
            '["included", "excluded"]'
        )
    return tuple(values)
```

New source freezes use `strict_new=True`; validation and rederivation of existing frozen coordinators use `False`. Compatibility must never depend on directory names. Thread the optional tuple through stage construction, but do not serialize an incomplete configuration v3 before Plan 2 binds evidence.

- [ ] **Step 4: Run producer and historical tests**

```bash
pytest -q tests/test_screening_batches.py -k 'result_status or historical or stage'
for version in 1 2 3 4 5 6; do
  python3 -B -m paper.scripts.prepare_screening_batches \
    --snapshot-dir "paper/data/screening_inputs/v${version}"
done
```

Expected: focused tests and every committed coordinator validation pass.

- [ ] **Step 5: Commit**

```bash
git add paper/scripts/prepare_screening_batches.py \
        tests/test_screening_batches.py tests/test_screening_integration.py
git diff --cached --check
git -c commit.gpgsign=false commit -m "feat: bind binary screening statuses"
```

### Task 2: Enforce Binary Statuses In Bound Results

**Files:**
- Modify: `paper/scripts/screening_results.py`
- Modify: `tests/test_screening_results.py`

- [ ] **Step 1: Add failing coordinator-bound tests**

Build a fresh-v7 fixture and prove `included` and `excluded` pass while `boundary` fails. Preserve an unbound legacy control where `validate_result_decision()` accepts `boundary`. Add a loop validating every committed result snapshot that exists for v1-v6 against its coordinator and release.

- [ ] **Step 2: Confirm v7 boundary is still accepted**

```bash
pytest -q tests/test_screening_results.py \
  -k 'binary_status or boundary or committed_history'
```

- [ ] **Step 3: Carry statuses in coordinator capture**

Extend `CoordinatorSnapshot` with:

```python
allowed_screening_statuses: tuple[str, ...]
```

Resolve `screening_result_status` when present and otherwise use legacy `SCREENING_STATUSES`. Include the tuple in state signatures, reattestation, and forged-state rejection.

- [ ] **Step 4: Parameterize private decision validation**

```python
def _validate_result_decision(
    row: Row,
    *,
    context: str,
    allowed_screening_statuses: tuple[str, ...] = tuple(sorted(SCREENING_STATUSES)),
    allowed_inclusion_criteria: tuple[str, ...] = INCLUSION_CRITERIA,
) -> None:
```

Bound phase validation passes both coordinator tuples. The public helper omits both and stays legacy-compatible. Role validation reads statuses only from configuration v3; v1/v2 retain legacy semantics.

- [ ] **Step 5: Run and commit**

```bash
pytest -q tests/test_screening_results.py
git add paper/scripts/screening_results.py tests/test_screening_results.py
git diff --cached --check
git -c commit.gpgsign=false commit -m "feat: enforce bound screening statuses"
```

### Task 3: Propagate The Contract To Agreement And Adjudication

**Files:**
- Modify: `paper/scripts/screening_agreement.py`
- Modify: `paper/scripts/integrate_screening.py`
- Modify: `tests/test_screening_agreement.py`
- Modify: `tests/test_screening_integration.py`

- [ ] **Step 1: Write failing downstream tests**

Add a v7 agreement fixture with an injected `boundary` row and a v7 adjudication fixture whose final decision uses `boundary`; both must fail. Keep committed v5/v6 agreement and decision controls.

- [ ] **Step 2: Confirm failures**

```bash
pytest -q tests/test_screening_agreement.py tests/test_screening_integration.py \
  -k 'v7_binary or historical_agreement or historical_decision'
```

- [ ] **Step 3: Consume captured coordinator values**

Replace local hard-coded validation with `coordinator.allowed_screening_statuses` and `coordinator.allowed_inclusion_criteria`. Adjudication validates final decisions against the bound values instead of the public legacy helper. Do not change historical snapshot bytes or resolution triggers.

- [ ] **Step 4: Run and commit**

```bash
pytest -q tests/test_screening_agreement.py tests/test_screening_integration.py
git add paper/scripts/screening_agreement.py paper/scripts/integrate_screening.py \
        tests/test_screening_agreement.py tests/test_screening_integration.py
git diff --cached --check
git -c commit.gpgsign=false commit -m "feat: propagate binary screening contract"
```

The known unrelated execution-register header failure may remain documented; do not broaden this task to change that contract.

### Task 4: Add The Survey Evidence Tier

**Files:**
- Modify: `paper/scripts/validate_corpus.py`
- Modify: `paper/scripts/prepare_screening_batches.py`
- Modify: `paper/data/taxonomy.json`
- Modify: `paper/data/evidence.csv`
- Modify: `paper/data/README.md`
- Modify: `tests/test_survey_corpus.py`

- [ ] **Step 1: Add failing schema and scalar tests**

Update fixtures with `survey_evidence_tier="core"`. Accept `core`, `supporting`, `contextual`, and sole `NR`; reject blank, `core;supporting`, lowercase `nr`, and unknown values. Assert exact taxonomy order.

- [ ] **Step 2: Confirm failures**

```bash
pytest -q tests/test_survey_corpus.py -k 'survey_evidence_tier or exact_headers'
```

- [ ] **Step 3: Implement the scalar field**

Insert `survey_evidence_tier` after `cite_key` in both exact evidence headers. Add it to `DEFAULT_TAXONOMY`, `CONTROLLED_FIELDS`, `SCALAR_CONTROLLED_FIELDS`, and `REQUIRED_FIELDS`:

```python
"survey_evidence_tier": ["core", "supporting", "contextual"]
```

Update the header-only canonical CSV and taxonomy. Keep `NR` as the existing evidence sentinel, not a taxonomy value.

- [ ] **Step 4: Document semantics and run validation**

Document all tiers, pre-v7 `NR`, and that retained-source count is not generation-method count. State supporting/contextual evidence cannot substantiate generation-method claims.

```bash
pytest -q tests/test_survey_corpus.py
python3 -B -m paper.scripts.validate_corpus paper/data
```

- [ ] **Step 5: Commit**

```bash
git add paper/scripts/validate_corpus.py paper/scripts/prepare_screening_batches.py \
        paper/data/taxonomy.json paper/data/evidence.csv paper/data/README.md \
        tests/test_survey_corpus.py
git diff --cached --check
git -c commit.gpgsign=false commit -m "feat: classify survey evidence tiers"
```

### Task 5: Add Tier Reliability Coding

**Files:**
- Modify: `paper/scripts/coding_reliability.py`
- Modify: `tests/test_coding_reliability.py`
- Modify: committed draft CSVs under `paper/data/screening_work/v2/evidence_drafts/`

- [ ] **Step 1: Add failing field-order and disagreement tests**

Assert `survey_evidence_tier` is the first reliability field and appears in blind templates, paired comparisons, and summaries. A `core` versus `supporting` fixture must count as an independent disagreement.

- [ ] **Step 2: Confirm failures**

```bash
pytest -q tests/test_coding_reliability.py -k 'field_order or tier or disagreement'
```

- [ ] **Step 3: Extend reliability generation**

```python
CORE_FIELDS = (
    "survey_evidence_tier",
    "course_object",
    "representation_family",
    "generator_family",
    "generation_role",
    "validity_strategy",
    "code_status",
    "asset_status",
)
```

Mechanically add `survey_evidence_tier=NR` to pre-v7 draft rows/templates. Never infer a tier from historical screening status or rewrite frozen artifacts.

- [ ] **Step 4: Run and commit**

```bash
pytest -q tests/test_coding_reliability.py tests/test_survey_corpus.py
git add paper/scripts/coding_reliability.py tests/test_coding_reliability.py \
        paper/data/screening_work/v2/evidence_drafts
git diff --cached --check
git -c commit.gpgsign=false commit -m "feat: measure evidence tier reliability"
```

### Task 6: Publish The V7 Working Protocol

**Files:**
- Modify: `paper/data/screening_protocol.md`
- Modify: `paper/data/screening_reviewer_prompt.md`
- Modify: `paper/data/taxonomy.json`
- Create: `paper/data/screening_work/v7/protocol.md`
- Create: `paper/data/screening_work/v7/reviewer_prompt_template.md`
- Create: `paper/data/screening_work/v7/README.md`
- Modify: `tests/test_screening_protocol.py`

- [ ] **Step 1: Add failing protocol assertions**

Assert only `included`/`excluded` are final v7 statuses; `boundary` is historical only; fixed-course sources with transferred requirements are retained; Pass 1 does not perform contribution coding; and packet-only eligibility evidence is mandatory.

- [ ] **Step 2: Update protocol and prompt**

Specify the three retention relationships, exact status/criterion pairing, fixed-route example, evidence-tier definitions, claim guardrails, and shared-frozen-evidence rule. Preserve result CSV field order.

- [ ] **Step 3: Run focused gates**

```bash
pytest -q tests/test_screening_protocol.py
pytest -q tests/test_screening_batches.py tests/test_screening_results.py \
  tests/test_screening_agreement.py tests/test_coding_reliability.py \
  tests/test_survey_corpus.py
python3 -B -m paper.scripts.validate_corpus paper/data
```

- [ ] **Step 4: Commit**

```bash
git add paper/data/screening_protocol.md paper/data/screening_reviewer_prompt.md \
        paper/data/taxonomy.json paper/data/screening_work/v7 \
        tests/test_screening_protocol.py
git diff --cached --check
git -c commit.gpgsign=false commit -m "docs: publish v7 retention protocol"
```

### Task 7: Final Contract Review

- [ ] **Step 1: Run historical and focused validation**

```bash
pytest -q tests/test_screening_batches.py tests/test_screening_results.py \
  tests/test_screening_agreement.py tests/test_screening_protocol.py \
  tests/test_coding_reliability.py tests/test_survey_corpus.py
git diff --check HEAD~6..HEAD
```

- [ ] **Step 2: Confirm invariants**

No v1-v6 snapshot bytes changed; no v7 main release exists; corpus `screening_status` still includes `candidate`; only bound v7 result paths reject `boundary`.

- [ ] **Step 3: Request independent spec and quality reviews**

Resolve all findings before starting the shared-evidence plan.
