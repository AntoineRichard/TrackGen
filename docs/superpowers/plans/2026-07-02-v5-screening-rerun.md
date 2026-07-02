# V5 Screening Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a valid v5 screening calibration under the approved `include-1` precedence rule and, only after a passing sealed gate, complete and seal all 344 main-phase ratings.

**Architecture:** Preserve v4 as immutable history. Build a v5 coordinator from byte-identical bibliographic inputs, Terra/high execution profile, and reviewer prompt, changing only the protocol precedence text. Execute each reviewer from an isolated role stage with exact rendered-prompt delivery and role-local validation, then use the existing authoritative sealing tools.

**Tech Stack:** Python 3, canonical CSV/JSON, SHA-256 manifests, pytest, Codex isolated subagents, existing `prepare_screening_batches.py` and `screening_results.py` CLIs.

---

### Task 1: Create The V5 Protocol Draft

**Files:**
- Create: `paper/data/screening_work/v5/protocol.md`
- Create: `paper/data/screening_work/v5/README.md`
- Reference: `docs/superpowers/specs/2026-07-01-screening-inclusion-precedence-design.md`
- Reference: `paper/data/screening_inputs/v4/protocol.md`

- [ ] **Step 1: Copy the immutable v4 protocol into the v5 working area**

```bash
mkdir -p paper/data/screening_work/v5
cp paper/data/screening_inputs/v4/protocol.md paper/data/screening_work/v5/protocol.md
```

Expected: the draft initially has the same SHA-256 as v4 protocol
`d3177ec60cfb8f0c229aa2c471dd1b0c4259a1c45a05a286baf8d19d778aeee6`.

- [ ] **Step 2: Apply the approved normative wording**

Change the title to `# Duplicate full-text screening codebook v3`. Replace the
`include-1` and `include-2` table definitions with:

```markdown
| `include-1` | Directly synthesizes, samples, selects, places, connects, mutates, repairs, validates, optimizes, or serializes explicit course geometry or a course distribution for racing robots or a transferable adjacent domain. When any such source-native operation is established, `include-1` takes precedence over `include-2`. |
| `include-2` | Makes a source-native contribution that defines a representation, design interface, simulator interface, dataset, benchmark, competition course set, or interchange artifact specifically for generated or parameterized courses, and no qualifying `include-1` operation is established. |
```

Add this paragraph to `### Inclusion-boundary precedence clarification`:

```markdown
When one source both performs a qualifying `include-1` operation and defines an `include-2` representation, interface, dataset, benchmark, course set, simulator contract, or interchange artifact, the reviewer MUST record `include-1`. Additional `include-2` applicability MAY be recorded in `notes` but MUST NOT replace the primary criterion. `include-2` is selected only when no qualifying `include-1` operation is established. Merely loading, referencing, displaying, or controlling on supplied fixed geometry is not an `include-1` operation.
```

- [ ] **Step 3: Record the revision provenance**

Create `paper/data/screening_work/v5/README.md` recording:

```markdown
# Screening Rerun v5

Version 5 is a substantive protocol revision following sealed v4 calibration decision
`bf3e8b5c444c0cf1ddb1007bb608bfb6d71cf37c29243e782ef18ab90e2a9097`.
It changes only `include-1` versus `include-2` precedence according to the approved
design. Bibliographic inputs and the stable 30-candidate calibration selection remain
unchanged. V4 ratings are not supplied to v5 reviewers.
```

- [ ] **Step 4: Verify the draft scope**

```bash
diff -u paper/data/screening_inputs/v4/protocol.md paper/data/screening_work/v5/protocol.md
rg -n "include-1|include-2|codebook v3" paper/data/screening_work/v5/protocol.md
```

Expected: only the title, two criterion definitions, and one precedence paragraph differ.

### Task 2: Freeze And Validate The V5 Coordinator

**Files:**
- Create: `paper/data/screening_inputs/v5/`
- Reuse unchanged: `paper/data/screening_inputs/v4/{candidates.csv,conflicts.csv,bibliography.csv,citation_keys.csv,taxonomy.json,execution_profile.json,reviewer_prompt_template.md}`

- [ ] **Step 1: Freeze v5**

```bash
python3 -B -m paper.scripts.prepare_screening_batches --freeze \
  --candidates paper/data/screening_inputs/v4/candidates.csv \
  --conflicts paper/data/screening_inputs/v4/conflicts.csv \
  --bibliography paper/data/screening_inputs/v4/bibliography.csv \
  --citation-keys paper/data/screening_inputs/v4/citation_keys.csv \
  --taxonomy paper/data/screening_inputs/v4/taxonomy.json \
  --protocol paper/data/screening_work/v5/protocol.md \
  --execution-profile paper/data/screening_inputs/v4/execution_profile.json \
  --reviewer-prompt-template paper/data/screening_inputs/v4/reviewer_prompt_template.md \
  --output-dir paper/data/screening_inputs/v5
```

Expected: a new immutable v5 coordinator with 404 assignments.

- [ ] **Step 2: Validate and verify stable calibration selection**

```bash
python3 -B -m paper.scripts.prepare_screening_batches \
  --snapshot-dir paper/data/screening_inputs/v5
cmp paper/data/screening_inputs/v4/calibration_selection.csv \
    paper/data/screening_inputs/v5/calibration_selection.csv
python3 -B -m paper.scripts.validate_corpus \
  --input-dir paper/data/screening_inputs/v5
```

Expected: all commands exit 0; the stable 30 are byte-identical.

- [ ] **Step 3: Commit the protocol and coordinator**

```bash
git add paper/data/screening_work/v5 paper/data/screening_inputs/v5
git diff --cached --check
git -c commit.gpgsign=false commit -m "docs: freeze v5 screening precedence protocol"
```

### Task 3: Execute And Seal Fresh V5 Calibration

**Files:**
- Create: `paper/data/screening_releases/calibration/v5/`
- Create locally: `paper/data/screening_staging/v5/calibration/`
- Create: `paper/data/screening_results/calibration/v5/`
- Create: `paper/data/screening_decisions/v5/`
- Modify: `paper/data/screening_work/v5/README.md`

- [ ] **Step 1: Publish the calibration release and six role stages**

```bash
mkdir -p -m 750 paper/data/screening_staging/v5/calibration
python3 -B -m paper.scripts.prepare_screening_batches --release \
  --snapshot-dir paper/data/screening_inputs/v5 --phase calibration \
  --output-dir paper/data/screening_releases/calibration/v5
```

For each `screening-01` through `screening-06`, run:

```bash
python3 -B -m paper.scripts.prepare_screening_batches --stage-role \
  --snapshot-dir paper/data/screening_inputs/v5 \
  --reviewer-release-snapshot paper/data/screening_releases/calibration/v5 \
  --role-id "$ROLE_ID" \
  --staging-root paper/data/screening_staging/v5/calibration
```

Expected: six random role-private stage paths ending in `/v1`.

- [ ] **Step 2: Dispatch fresh reviewers with exact prompt bytes**

For each role, start a new Terra/high agent with `fork_context=false`. Pass the complete
contents of that stage's `reviewer_prompt.md` as the user message, with no wrapper.
Do not reuse any v4 reviewer context. Record role ID, returned agent/context ID, stage
path, start date, completion date, result path, and completion hash in the v5 README.

Expected: six completion records and six role-local validator successes totaling 60 rows.

- [ ] **Step 3: Authoritatively seal calibration**

```bash
python3 -B -m paper.scripts.screening_results --seal-phase \
  --coordinator-snapshot paper/data/screening_inputs/v5 \
  --reviewer-release-snapshot paper/data/screening_releases/calibration/v5 \
  --phase calibration \
  --result "$RESULT_01" --result "$RESULT_02" --result "$RESULT_03" \
  --result "$RESULT_04" --result "$RESULT_05" --result "$RESULT_06" \
  --output-dir paper/data/screening_results/calibration/v5
```

Expected: immutable six-file calibration result snapshot plus manifest and checksums.

- [ ] **Step 4: Derive and seal the calibration gate**

Compute exact status/criterion agreement from the two locked ratings per candidate.
Inspect every disagreement under the v5 ambiguity rule without altering raw ratings.
Create the exact one-row decision CSV and seal it with:

```bash
python3 -B -m paper.scripts.screening_results --seal-calibration-decision \
  --coordinator-snapshot paper/data/screening_inputs/v5 \
  --reviewer-release-snapshot paper/data/screening_releases/calibration/v5 \
  --calibration-result-snapshot paper/data/screening_results/calibration/v5 \
  --decision-input paper/data/screening_work/v5/calibration_decision.csv \
  --output-dir paper/data/screening_decisions/v5
```

Expected: `release` only when status agreement is at least 0.80 and systematic ambiguity is false. Otherwise seal `revise` and stop before Task 4.

- [ ] **Step 5: Commit the sealed calibration artifacts**

```bash
git add paper/data/screening_releases/calibration/v5 \
        paper/data/screening_results/calibration/v5 \
        paper/data/screening_decisions/v5 \
        paper/data/screening_work/v5
git diff --cached --check
git -c commit.gpgsign=false commit -m "docs: seal v5 calibration gate"
```

### Task 4: Execute And Seal Main Screening After A Passing Gate

**Files:**
- Create: `paper/data/screening_releases/main/v5/`
- Create locally: `paper/data/screening_staging/v5/main/`
- Create: `paper/data/screening_results/main/v5/`
- Modify: `paper/data/screening_work/v5/README.md`

- [ ] **Step 1: Publish the gate-bound main release**

```bash
python3 -B -m paper.scripts.prepare_screening_batches --release \
  --snapshot-dir paper/data/screening_inputs/v5 --phase main \
  --calibration-reviewer-release-snapshot paper/data/screening_releases/calibration/v5 \
  --calibration-result-snapshot paper/data/screening_results/calibration/v5 \
  --calibration-decision-snapshot paper/data/screening_decisions/v5 \
  --output-dir paper/data/screening_releases/main/v5
```

Expected: release succeeds only for a sealed passing v5 decision.

- [ ] **Step 2: Stage and execute six fresh main contexts**

Create six role-private main stages using the v5 main release. Start six new Terra/high
agents with `fork_context=false`, delivering each complete rendered prompt as the exact
user message. Record every role/context/stage binding contemporaneously. Do not supply
calibration ratings or discussion. Each output must pass its role-local validator.

Expected: six canonical result files totaling 344 rows.

- [ ] **Step 3: Seal main results**

```bash
python3 -B -m paper.scripts.screening_results --seal-phase \
  --coordinator-snapshot paper/data/screening_inputs/v5 \
  --reviewer-release-snapshot paper/data/screening_releases/main/v5 \
  --phase main \
  --calibration-reviewer-release-snapshot paper/data/screening_releases/calibration/v5 \
  --calibration-result-snapshot paper/data/screening_results/calibration/v5 \
  --calibration-decision-snapshot paper/data/screening_decisions/v5 \
  --result "$RESULT_01" --result "$RESULT_02" --result "$RESULT_03" \
  --result "$RESULT_04" --result "$RESULT_05" --result "$RESULT_06" \
  --output-dir paper/data/screening_results/main/v5
```

Expected: immutable 344-rating main snapshot bound to the passing calibration tuple.

- [ ] **Step 4: Commit and hand off to adjudication planning**

```bash
git add paper/data/screening_releases/main/v5 \
        paper/data/screening_results/main/v5 \
        paper/data/screening_work/v5/README.md
git diff --cached --check
git -c commit.gpgsign=false commit -m "docs: seal v5 main screening ratings"
```

Run final validation:

```bash
python3 -B -m paper.scripts.prepare_screening_batches \
  --snapshot-dir paper/data/screening_inputs/v5
pytest -q tests/test_screening_results.py -k 'role_result'
```

Expected: coordinator validation and focused tests pass. Begin a separate adjudication/execution-registry plan using only the sealed v5 snapshots.
