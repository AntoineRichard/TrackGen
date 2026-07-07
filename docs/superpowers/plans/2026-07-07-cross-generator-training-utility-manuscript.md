# Cross-Generator Training Utility Manuscript Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a controlled cross-generator RL transfer matrix and model-based reference-controller calibration protocol to the survey without implying that the benchmark has been implemented.

**Architecture:** Section 8 defines estimands and reporting units; Section 9 defines the experiment and confound controls; Section 12 makes the proposal falsifiable; the introduction and conclusion surface the objective without duplicating technical detail. MPPI is an initial ground-racing implementation of a generic reference-controller contract, not an oracle or generator score.

**Tech Stack:** LaTeX, `latexmk`, BibTeX, existing `cleveref`/`tabularx` conventions, Git.

**Design:** `docs/superpowers/specs/2026-07-07-cross-generator-training-utility-design.md`

---

### Task 1: Define Training-Utility And Reference-Controller Metrics

**Files:**
- Modify: `paper/sections/08-metrics.tex:174-240`

- [ ] **Step 1: Confirm the metric subsection is absent**

Run:

```bash
rg -n "Cross-Generator Training Utility|cross-generator-transfer-gap|reference-controller-gap" \
  paper/sections/08-metrics.tex
```

Expected: no matches and exit status 1.

- [ ] **Step 2: Insert the metric subsection**

Immediately before the metric-map table following `Simulation Feasibility`, add:

```latex
\subsection{Cross-Generator Training Utility}

Controller outcomes can evaluate a generator's utility as a training distribution,
but they do not define an intrinsic generator quality. Let $G_1,\ldots,G_m$ be
generators, let $E_1,\ldots,E_m$ be their immutable generated evaluation suites, and
let $E_{\mathrm{real}}$ be the frozen evaluation-only real-track suite. Write
$\mathcal{J}=\{1,\ldots,m,\mathrm{real}\}$ and let $C_j$ be the course set for
$E_j$, including $C_{\mathrm{real}}$. Let $P$ be the common global policy-seed set
and $\mathcal{R}_{pcj}$ the common rollout-seed schedule across training-generator
rows. Retain rollout outcomes and define
\begin{equation}
  \overline{Y}_{ijcp}
  =\operatorname{Agg}_{r\in\mathcal{R}_{pcj}}Y_{ijcpr},
  \qquad
  M^Y_{ij}
  =\frac{1}{|P|\,|C_j|}
  \sum_{p\in P}\sum_{c\in C_j}\overline{Y}_{ijcp},
  \quad j\in\mathcal{J}.
\end{equation}

Within generated evaluation column $j$, compare training distributions on common
courses:
\begin{equation}
  \Delta^Y_{ij}=M^Y_{ij}-M^Y_{jj},
  \label{eq:cross-generator-transfer-gap}
\end{equation}
with a declared outcome direction. For the common-stratum analysis, freeze represented
strata $\mathcal{H}^{\cap}$, positive weights $w_h$ summing to one, and per-suite
sets $C^{\cap}_{jh}$, then define
\begin{equation}
  M^{Y,\cap}_{ij}
  =\sum_{h\in\mathcal{H}^{\cap}}w_h
  \left(
    \frac{1}{|P|\,|C^{\cap}_{jh}|}
    \sum_{p\in P}\sum_{c\in C^{\cap}_{jh}}\overline{Y}_{ijcp}
  \right),
  \qquad
  \Delta^{Y,\cap}_{ij}=M^{Y,\cap}_{ij}-M^{Y,\cap}_{jj}.
  \label{eq:common-stratum-transfer-gap}
\end{equation}
For each preregistered training-row pair $i<k$, compare policies on the same real
courses and seed schedules:
\begin{equation}
  \Delta^Y_{ik,\mathrm{real}}
  =M^Y_{i,\mathrm{real}}-M^Y_{k,\mathrm{real}}.
  \label{eq:real-track-transfer-gap}
\end{equation}
Preregister outcome directions, primary contrast families, and multiplicity control.
Report the complete matrix and separate suite-specific, common-stratum, and real-track
contrasts. These quantities form a training-utility profile, not a scalar generator
ranking.

Every evaluation course should also be attempted by a declared model-based reference
controller. The contract is controller-generic; MPPI is the proposed initial
ground-racing implementation. Freeze the controller model, objective, constraints,
horizon, sampling budget, stochastic seed set, compute budget, and termination rules.
Tune it once on a declared calibration set disjoint from all training and evaluation
suites; generator-specific tuning is prohibited.

Report outcome calibration---completion, collision, violation, timeout, or
inconclusive budget exhaustion---before conditional performance. On explicitly stated
subsets, particularly courses completed by both methods, report:
\begin{equation}
  R^{\mathrm{time}}_{ijcp}
  =\frac{\overline{T}^{\mathrm{RL}}_{ijcp}}
  {\widetilde{T}^{\mathrm{ref}}_{jc}},
  \qquad
  \Delta v_{ijcp}
  =\overline{v}^{\mathrm{RL}}_{ijcp}
  -\widetilde{v}^{\mathrm{ref}}_{jc},
  \label{eq:reference-controller-gap}
\end{equation}
where the overbar is the declared RL rollout aggregation and the tilde is the
preregistered aggregation over reference-controller seeds. Reference-controller
success with consistent RL failure is diagnostic of a training or policy limitation
conditional on the declared model and controller configuration. Failure by both
methods does not establish geometric or dynamic infeasibility. Reference-controller
outcomes calibrate policy results and are not part of a generator ranking.
```

- [ ] **Step 3: Add the metric-map row**

Insert before `Can another system reproduce and use it?`:

```latex
    Does a training generator support transfer? & Crossed generator-to-suite policy
    outcomes, within-column transfer gaps, real-track outcomes, and
    reference-controller gaps & Poor transfer, training sensitivity, or a
    controller-conditional limitation; not intrinsic generator quality \\
```

- [ ] **Step 4: Build and scan**

Run:

```bash
latexmk -g -r paper/latexmkrc -cd -pdf paper/main.tex
rg -n "undefined|Overfull|Float too large" paper/build/main.log
```

Expected: build exit 0; no undefined references/citations, overfull boxes, or
float-too-large warnings caused by Section 8. Existing compact-table underfull
warnings are acceptable.

- [ ] **Step 5: Verify equations and non-ranking boundaries**

```bash
rg -n "eq:cross-generator-transfer-gap|eq:common-stratum-transfer-gap|eq:real-track-transfer-gap|eq:reference-controller-gap|not a scalar generator ranking|not part of a generator ranking" \
  paper/sections/08-metrics.tex
```

Expected: all four labels and both non-ranking qualifications are present.

- [ ] **Step 6: Commit**

```bash
git add paper/sections/08-metrics.tex
git -c commit.gpgsign=false commit -m "docs: define cross-generator training utility metrics"
```

---

### Task 2: Specify The Crossed Training And Evaluation Protocol

**Files:**
- Modify: `paper/sections/09-benchmark-protocol.tex:89-170`

- [ ] **Step 1: Confirm the crossed protocol is absent**

```bash
rg -n "Suite-specific/full-suite analysis|Common-stratum analysis|evaluation-only real-track|Cross-Generator Training-Utility Protocol" \
  paper/sections/09-benchmark-protocol.tex
```

Expected: no matches and exit status 1.

- [ ] **Step 2: Add the benchmark subsection**

Insert after the existing `Policy Evaluation` text and before
`tab:benchmark-requirements`:

```latex
\subsection{Cross-Generator Training-Utility Protocol}

For each generator $G_i$, train the same RL algorithm under the same vehicle,
dynamics, simulator, observation/action, reward, termination, policy architecture,
optimization, environment-interaction, policy-update, and policy-seed contracts. Only
the training-course distribution changes. Freeze and report unique training-course
IDs and hashes, generator seeds, sampling order or probabilities, reuse counts, and
any curriculum or scheduling rule. Equal interaction budgets are the primary
comparison; separately report generation attempts and failures, feasible courses
observed, simulator steps, updates, wall-clock time, and compute so that generator
cost and training exposure are not hidden.

Evaluate every trained policy on immutable suites $E_1,\ldots,E_m$ and on a frozen
real-track suite $E_{\mathrm{real}}$. Real tracks are evaluation-only in the core
protocol because training on a finite real corpus would confound generator utility
with memorization and augmentation. Before evaluation, run manifest-backed duplicate
and near-duplicate checks between every training population and every generated suite
and the real-track suite. Use common course identifiers and rollout random streams
across policies where meaningful, and report the matrix and within-column contrasts
from \cref{eq:cross-generator-transfer-gap}.

Construct every generated suite after raw feasibility accounting with the same frozen
descriptor transform, absolute admissible bounds, spectrum strata, diversity rules,
and minimum suite-size contract. The suite-specific/full-suite analysis uses every
course in each source's released suite and makes no claim to enumerate the generator's
full support. For the common-stratum analysis, preregister a minimum occupancy
$n_{\min}\geq2$ admitted feasible, non-duplicate courses per stratum. A stratum is
represented for suite $E_j$ only when its frozen manifest meets that threshold. Freeze
the intersection $\mathcal{H}^{\cap}$, positive weights $w_h$ summing to one, and
per-suite stratum course sets $C^{\cap}_{jh}$. Use the weighted
$M^{Y,\cap}_{ij}$ and $\Delta^{Y,\cap}_{ij}$ rather than an unweighted pooled mean,
and report missing-stratum and occupancy shortfalls without relaxing the contract.
Freeze $C_{\mathrm{real}}$, its conditioning variables, and
$\mathcal{R}_{pc,\mathrm{real}}$ before evaluation. Preregister primary outcomes,
directions, bounds, and one multiplicity family containing selected suite-specific,
common-stratum, and paired real-track contrasts.

Run the declared reference controller on the same evaluation courses using one
generator-independent configuration and calibration set. Preserve its stochastic
seeds and compute budget. Report joint RL/reference outcome categories before the
conditional lap-time, progress, speed, tracking, and control-effort differences in
\cref{eq:reference-controller-gap}. State the exact subset and denominator for every
conditional metric; the default subset contains only policy-seed/course cells for
which both methods complete. Reference-controller failure is inconclusive with respect
to course infeasibility.

For each outcome, aggregate rollouts within policy-seed/course cells. Resample common
policy-seed indices jointly across rows and columns and course identifiers jointly
across rows within each evaluation column. For common-stratum estimates, resample
within strata and recombine with the frozen weights. Resample nested RL rollout seeds
within cells; for reference gaps, resample controller seeds once per course and share
the draw across rows. Report paired intervals, full failure taxonomies, and
outcome-specific denominators. Generator yield, coverage, diversity, cost,
interoperability, training-utility transfer, and reference-controller gaps remain
separate results.
```

- [ ] **Step 3: Update the benchmark table**

Replace the `Policy results` row with:

```latex
    Policy results & Common course IDs, multiple policy seeds, crossed
    generator-to-suite outcomes, real-track evaluation, intervals, and failure
    taxonomy, plus a frozen model-based reference controller & Additional policy and
    reference-controller families \\
```

- [ ] **Step 4: Build and scan**

```bash
latexmk -g -r paper/latexmkrc -cd -pdf paper/main.tex
rg -n "undefined|Overfull|Float too large" paper/build/main.log
```

Expected: build exit 0 and no new undefined, overfull, or float-too-large warning. If
the table becomes too tall, shorten only the modified `Policy results` cells.

- [ ] **Step 5: Verify confound controls**

```bash
rg -n "Only the training-course distribution changes|evaluation-only|suite-specific/full-suite analysis|common-stratum analysis|generator-independent|inconclusive" \
  paper/sections/09-benchmark-protocol.tex
```

Expected: all six controls are present.

- [ ] **Step 6: Commit**

```bash
git add paper/sections/09-benchmark-protocol.tex
git -c commit.gpgsign=false commit -m "docs: specify crossed generator transfer benchmark"
```

---

### Task 3: Make The Claim Falsifiable And Surface It In The Framing

**Files:**
- Modify: `paper/sections/12-open-problems.tex:41-47`
- Modify: `paper/sections/01-introduction.tex:100-125`
- Modify: `paper/sections/13-conclusion.tex:20-45`

- [ ] **Step 1: Replace H6**

Replace the current H6 paragraph with:

```latex
\paragraph{H6: Cross-generator training utility and real-track transfer.}
Train matched policy seeds separately on each generator and evaluate the resulting
policies on the full crossed generator-to-suite matrix and an evaluation-only
real-track suite. Compare preregistered suite-specific gaps $\Delta^Y_{ij}$,
stratum-weighted gaps $\Delta^{Y,\cap}_{ij}$, and paired real-track contrasts
$\Delta^Y_{ik,\mathrm{real}}$. Predefine primary outcomes, directions, bounds, and
their multiplicity family. Use a frozen model-based reference controller to calibrate
policy failure without treating it as an oracle. Reject a generator training-utility
claim if any designated primary contrast breaches its bound under the preregistered
multiplicity rule, or if the suite-specific and common-stratum analyses yield opposite
bound decisions.
Required artifacts are training distributions and exposure logs, immutable generated
and real suites, descriptor strata, fixed stratum weights
and shortfall records, policy, rollout, and reference-controller seeds,
reference-controller configuration, paired outcomes, and failure labels.
```

- [ ] **Step 2: Add one sentence to introduction contribution 4**

After the sentence ending `an explicit failure taxonomy.` add:

```latex
  The protocol also evaluates each generator as a training distribution through a
  crossed generator-to-suite policy matrix, an evaluation-only real-track column, and
  a separately reported model-based reference-controller calibration.
```

Keep contribution item 5 intact so the proposal remains explicitly unreleased.

- [ ] **Step 3: Add the objective to the conclusion**

After the sentence ending `without discarding failure modes or inventing false
pairings.` add:

```latex
Crossed evaluation of policies trained separately on each generator can then measure
training-distribution utility, while an evaluation-only real-track suite tests transfer
beyond generated suites. A frozen model-based reference controller can provide
conditional diagnostic evidence when it succeeds consistently where RL fails; shared
failure remains inconclusive. It is not an optimal oracle and does not convert these
outcomes into a scalar generator rank.
```

- [ ] **Step 4: Build and scan**

```bash
latexmk -g -r paper/latexmkrc -cd -pdf paper/main.tex
rg -n "undefined|Overfull|Float too large" paper/build/main.log
```

Expected: build exit 0 and no new undefined, overfull, or float-too-large warning.

- [ ] **Step 5: Verify terminology and proposal boundaries**

```bash
rg -n "training distribution|evaluation-only real-track|reference controller|not an optimal oracle|scalar generator rank" \
  paper/sections/01-introduction.tex \
  paper/sections/12-open-problems.tex \
  paper/sections/13-conclusion.tex
```

Expected: the objective appears in all three locations with non-oracle and non-ranking
qualifications.

- [ ] **Step 6: Commit**

```bash
git add paper/sections/01-introduction.tex \
  paper/sections/12-open-problems.tex \
  paper/sections/13-conclusion.tex
git -c commit.gpgsign=false commit -m "docs: add generator training utility hypothesis"
```

---

### Task 4: Cross-Section Scientific And Build Verification

**Files:**
- Verify: `paper/sections/01-introduction.tex`
- Verify: `paper/sections/08-metrics.tex`
- Verify: `paper/sections/09-benchmark-protocol.tex`
- Verify: `paper/sections/12-open-problems.tex`
- Verify: `paper/sections/13-conclusion.tex`

- [ ] **Step 1: Verify required concepts**

```bash
rg -n "Cross-Generator Training Utility|Cross-Generator Training-Utility Protocol|Cross-generator training utility and real-track transfer|eq:cross-generator-transfer-gap|eq:common-stratum-transfer-gap|eq:real-track-transfer-gap|eq:reference-controller-gap" \
  paper/sections
```

Expected: one metric subsection, one benchmark subsection, one H6 paragraph, and all four
equation labels.

- [ ] **Step 2: Inspect proposal boundaries**

```bash
git diff 70d1771..HEAD -- \
  paper/sections/01-introduction.tex \
  paper/sections/08-metrics.tex \
  paper/sections/09-benchmark-protocol.tex \
  paper/sections/12-open-problems.tex \
  paper/sections/13-conclusion.tex
```

Confirm the diff does not claim that the matrix was executed, the real suite or MPPI
adapter was released, MPPI proves feasibility/optimality, real tracks are used for
core training, or generators receive a universal scalar rank.

- [ ] **Step 3: Run the full build**

```bash
latexmk -g -r paper/latexmkrc -cd -pdf paper/main.tex
```

Expected: exit 0 and updated `paper/build/main.pdf`.

- [ ] **Step 4: Scan the final log**

```bash
rg -n "undefined citations|undefined references|Citation .* undefined|Reference .* undefined|Overfull|Float too large" \
  paper/build/main.log
```

Expected: no matches. The two pre-existing empty-year BibTeX warnings for
`KlimovNodateCarRacing` and `EclipseSUMONodateNetgenerate`, plus compact-table
underfull warnings, may remain.

- [ ] **Step 5: Run hygiene checks**

```bash
git diff --check
git status --short
```

Expected: diff check exits 0; status contains only known unrelated untracked
staging/source/archive files.

- [ ] **Step 6: Run two-stage review**

Dispatch fresh reviewers in sequence:

1. Specification review against
   `docs/superpowers/specs/2026-07-07-cross-generator-training-utility-design.md`.
2. Scientific-quality review of estimands, confound control, MPPI interpretation,
   proposal/result boundaries, notation, and LaTeX presentation.

Resolve every Critical or Important finding and rerun the relevant reviewer.

- [ ] **Step 7: Commit review fixes only when needed**

```bash
git add paper/sections/01-introduction.tex \
  paper/sections/08-metrics.tex \
  paper/sections/09-benchmark-protocol.tex \
  paper/sections/12-open-problems.tex \
  paper/sections/13-conclusion.tex
git -c commit.gpgsign=false commit -m "docs: refine training utility benchmark"
```

If reviewers require no fixes, do not create an empty commit. Report relevant commit
SHAs, build result, warning inventory, and remaining empirical implementation work.
