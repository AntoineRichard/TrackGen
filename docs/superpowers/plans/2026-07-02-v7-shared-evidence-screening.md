# V7 Shared Evidence Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give duplicate v7 reviewers identical, hash-bound evidence artifacts, then run fresh stable-30 calibration and release main screening only after the sealed gate passes.

**Architecture:** A canonical committed manifest describes candidate artifacts while local nonredistributable bytes remain untracked. Coordinator freeze validates coverage and hashes; releases filter rows by assignment; role stages copy verified bytes and bind a configuration-v3 manifest hash. Calibration is a separate immutable execution with six fresh Terra/high reviewer contexts.

**Tech Stack:** Python 3, pytest, canonical CSV/JSON, SHA-256, local source archive, Codex Terra/high workers.

---

### Task 1: Validate Canonical Evidence Packet Manifests

**Files:**
- Modify: `paper/scripts/prepare_screening_batches.py`
- Modify: `tests/test_screening_batches.py`

- [ ] **Step 1: Write failing manifest-shape tests**

Define this exact header and test canonical ordering, unique `(candidate_id, artifact_id)`, nonempty artifact sets, ISO dates, canonical HTTP(S) URLs, lowercase SHA-256, allowed access/redistribution values, repository-relative local filenames, path containment, no symlinks, missing files, and hash mismatches.

```python
EVIDENCE_PACKET_HEADER = (
    "candidate_id", "artifact_id", "artifact_role", "source_url",
    "evidence_version", "evidence_retrieved_on", "access_status",
    "evidence_archive_url", "evidence_sha256", "local_filename",
    "redistribution_status", "retrieval_notes",
)
```

- [ ] **Step 2: Confirm failures**

```bash
pytest -q tests/test_screening_batches.py -k 'evidence_packet_manifest'
```

- [ ] **Step 3: Implement the parser and local-byte verifier**

Add `parse_evidence_packet_manifest(payload, *, candidates, source_archive)` returning canonical rows. Add a mutually exclusive `--validate-evidence-manifest` CLI mode requiring `--candidates`, `--evidence-manifest`, and `--source-archive`; success prints canonical candidate and artifact counts. Require `evidence_sha256` and `local_filename` together or both `NR`; when present, open a regular file beneath `source_archive`, hash bytes, and reject traversal, symlinks, or changes during capture. Reuse existing canonical CSV and attested-file helpers.

- [ ] **Step 4: Run and commit**

```bash
pytest -q tests/test_screening_batches.py -k 'evidence_packet_manifest'
git add paper/scripts/prepare_screening_batches.py tests/test_screening_batches.py
git diff --cached --check
git -c commit.gpgsign=false commit -m "feat: validate screening evidence manifests"
```

### Task 2: Bind Evidence Into Coordinators And Releases

**Files:**
- Modify: `paper/scripts/prepare_screening_batches.py`
- Modify: `paper/scripts/screening_results.py`
- Modify: `tests/test_screening_batches.py`
- Modify: `tests/test_screening_results.py`
- Modify: `tests/test_screening_integration.py`

- [ ] **Step 1: Write failing freeze and release tests**

Fresh v7 freezes require `evidence_packet_manifest.csv`; historical snapshots without it validate. Every assigned candidate must have at least one row, and both assignments must resolve to identical ordered `(artifact_id, evidence_sha256)` tuples. Releases contain only rows for their phase.

- [ ] **Step 2: Confirm failures**

```bash
pytest -q tests/test_screening_batches.py tests/test_screening_results.py \
  -k 'evidence_manifest or identical_artifact or historical_without_evidence'
```

- [ ] **Step 3: Extend frozen coordinator artifacts**

Add the manifest to fresh snapshot artifacts, `manifest.csv`, snapshot digest, and `SHA256SUMS`. Capture its digest and per-candidate bindings in `CoordinatorSnapshot`; historical captures use `None` and retain old exact file sets.

- [ ] **Step 4: Extend phase releases**

For v7, include a phase-filtered evidence manifest and bind it into release snapshot hashes. Validation reconstructs the candidate set from assignments and rejects missing, extra, or altered rows. Historical releases retain old exact trees.

- [ ] **Step 5: Run and commit**

```bash
pytest -q tests/test_screening_batches.py tests/test_screening_results.py \
  tests/test_screening_integration.py \
  -k 'evidence_manifest or committed_v5 or historical'
git add paper/scripts/prepare_screening_batches.py paper/scripts/screening_results.py \
        tests/test_screening_batches.py tests/test_screening_results.py \
        tests/test_screening_integration.py
git diff --cached --check
git -c commit.gpgsign=false commit -m "feat: bind evidence manifests to screening"
```

### Task 3: Create Configuration V3 And Stage Verified Bytes

**Files:**
- Modify: `paper/scripts/prepare_screening_batches.py`
- Modify: `paper/scripts/screening_results.py`
- Modify: `tests/test_screening_batches.py`
- Modify: `tests/test_screening_results.py`

- [ ] **Step 1: Write failing v3 tests**

```python
assert configuration["configuration_version"] == "3"
assert configuration["allowed_screening_statuses"] == ["included", "excluded"]
assert configuration["allowed_inclusion_criteria"] == ["include-relevant"]
assert configuration["evidence_packet_manifest_sha256"] == sha256(
    stage["evidence_packet_manifest.csv"]
)
```

Test role filtering, byte-exact artifact copies, no unassigned artifacts, swapped bytes, missing copies, and stage-manifest mutation. Keep v1/v2 committed-history controls.

- [ ] **Step 2: Confirm failures**

```bash
pytest -q tests/test_screening_batches.py tests/test_screening_results.py \
  -k 'configuration_v3 or staged_evidence or v1_stage or v2_stage'
```

- [ ] **Step 3: Build and validate v3 stages**

Serialize v3 only when statuses, criterion, and evidence manifest are all bound. Add the role-filtered manifest to stage artifacts. Copy each verified file to `evidence/<candidate_id>/<artifact_id>/<basename>` with exclusive creation, fsync, and post-copy hash verification. Include all paths and hashes in the stage digest. `_validate_role_result()` consumes both v3 vocabularies and revalidates the evidence manifest before reading results.

- [ ] **Step 4: Run and commit**

```bash
pytest -q tests/test_screening_batches.py tests/test_screening_results.py
git add paper/scripts/prepare_screening_batches.py paper/scripts/screening_results.py \
        tests/test_screening_batches.py tests/test_screening_results.py
git diff --cached --check
git -c commit.gpgsign=false commit -m "feat: stage shared screening evidence"
```

### Task 4: Assemble The Stable-30 Evidence Inventory

**Files:**
- Create: `paper/data/screening_work/v7/evidence_packet_manifest.csv`
- Create: `paper/data/screening_work/v7/evidence_inventory.md`
- Add untracked bytes under: `paper/data/source_archive/v7/`

- [ ] **Step 1: Record best-available artifacts**

For each unchanged stable-30 candidate, identify a primary report and authoritative companion artifacts required for eligibility. Record exact version, retrieval date, access limitation, archive URL, SHA-256/local filename when bytes exist, redistribution status, and factual retrieval notes.

- [ ] **Step 2: Audit completeness mechanically**

```bash
python3 -B -m paper.scripts.prepare_screening_batches \
  --validate-evidence-manifest \
  --candidates paper/data/candidates.csv \
  --evidence-manifest paper/data/screening_work/v7/evidence_packet_manifest.csv \
  --source-archive paper/data/source_archive/v7
```

Expected: 30/30 candidate coverage and zero hash/path errors. If full text is unavailable, both reviewers receive the same documented best-available artifact.

- [ ] **Step 3: Report and independently audit inventory**

Report artifact, access-status, and redistribution counts plus procedural limitations. List unresolved files by paper title for user retrieval; do not use illicit access services. Dispatch one content/metadata auditor and one hash/path/coverage auditor, then resolve all discrepancies.

- [ ] **Step 4: Commit manifests only**

```bash
git add paper/data/screening_work/v7/evidence_packet_manifest.csv \
        paper/data/screening_work/v7/evidence_inventory.md
git diff --cached --check
git -c commit.gpgsign=false commit -m "data: audit v7 calibration evidence"
```

Never stage restricted source bytes.

### Task 5: Freeze And Verify The V7 Coordinator

**Files:**
- Create: `paper/data/screening_inputs/v7/`
- Modify: `paper/data/screening_work/v7/README.md`

- [ ] **Step 1: Freeze reviewed inputs**

Use the approved protocol, prompt, taxonomy, unchanged bibliographic inputs, stable-30 selection, and audited evidence manifest. Freeze once; never edit the directory afterward.

- [ ] **Step 2: Validate v7 and history**

```bash
python3 -B -m paper.scripts.prepare_screening_batches \
  --snapshot-dir paper/data/screening_inputs/v7
for version in 1 2 3 4 5 6; do
  python3 -B -m paper.scripts.prepare_screening_batches \
    --snapshot-dir "paper/data/screening_inputs/v${version}"
done
```

- [ ] **Step 3: Audit frozen invariants**

Verify 404 assignments, 60 calibration assignments, 30 stable candidates, two assignments per candidate, binary statuses, one inclusion criterion, unchanged stable-30 bytes, and identical evidence bindings for duplicate assignments.

- [ ] **Step 4: Commit**

```bash
git add paper/data/screening_inputs/v7 paper/data/screening_work/v7/README.md
git diff --cached --check
git -c commit.gpgsign=false commit -m "data: freeze v7 screening coordinator"
```

### Task 6: Run Fresh Stable-30 Calibration

**Files:**
- Create: `paper/data/screening_releases/calibration/v7/`
- Create: `paper/data/screening_results/calibration/v7/`
- Create: `paper/data/screening_decisions/v7/`
- Modify: `paper/data/screening_work/v7/README.md`
- Keep stages/results untracked under: `paper/data/screening_staging/v7/`

- [ ] **Step 1: Release and stage six calibration roles**

Create the immutable calibration release and six isolated v3 stages. Validate each stage and record stage, prompt, packet, evidence-manifest, and result-path hashes.

- [ ] **Step 2: Dispatch six fresh Terra/high workers**

Every implementation/reviewer worker uses model `gpt-5.6-terra` and reasoning effort `high`. Each receives only its rendered prompt and staged packet/evidence paths. Do not supply v3-v6 ratings, disagreements, or other reviewers' outputs.

- [ ] **Step 3: Validate and seal results**

Validate every role result, canonical bytes, and assignment coverage. Record worker context IDs, dates, result hashes, and procedural limits. Seal exactly 60 ratings, compute exact status agreement, inspect all disagreements, and seal one calibration decision.

- [ ] **Step 4: Enforce the gate**

Pass only with complete bindings, 60 ratings, agreement at least `0.80`, and `systematic_ambiguity=false`. On failure, commit the sealed revise decision and stop; do not release main.

- [ ] **Step 5: Commit immutable calibration evidence**

```bash
git add paper/data/screening_releases/calibration/v7 \
        paper/data/screening_results/calibration/v7 \
        paper/data/screening_decisions/v7 paper/data/screening_work/v7/README.md
git diff --cached --check
git -c commit.gpgsign=false commit -m "data: seal v7 calibration gate"
```

### Task 7: Release Main Screening Only After A Pass

**Files:**
- Create only after pass: `paper/data/screening_releases/main/v7/`
- Modify: `paper/data/screening_work/v7/README.md`

- [ ] **Step 1: Reattest the complete passing tuple**

Revalidate coordinator, calibration release, results, and decision immediately before main release. Reject any changed digest or nonpassing decision.

- [ ] **Step 2: Publish and validate main release**

Create the main-only release, validate candidate/assignment coverage and filtered evidence manifests, and stage six fresh Terra/high role contexts under the same isolation rules.

- [ ] **Step 3: Commit before rating**

```bash
git add paper/data/screening_releases/main/v7 \
        paper/data/screening_work/v7/README.md
git diff --cached --check
git -c commit.gpgsign=false commit -m "data: release v7 main screening"
```

- [ ] **Step 4: Stop at the released checkpoint**

Report coordinator/release hashes, role-stage validation, calibration statistics, evidence coverage, and remaining procedural limitations. Main ratings are a subsequent auditable execution.
