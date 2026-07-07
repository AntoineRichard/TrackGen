# Cross-Generator Training Utility Design

**Date:** 2026-07-07  
**Status:** Approved design for manuscript planning  
**Scope:** Proposed benchmark and metric additions; no empirical result or released implementation claim

## Objective

Extend the survey's proposed benchmark so that course generators are evaluated not
only by artifact-level properties, but also by their utility as training
distributions. The primary experiment trains the same reinforcement-learning method
separately on each generator and evaluates every resulting policy on immutable suites
from every generator plus an evaluation-only real-track reference suite.

The output is a multidimensional training-utility profile. It is not an intrinsic
generator score, a universal generator ranking, or evidence that the proposed
benchmark has already been implemented.

## Non-Goals

- Do not combine generator yield, diversity, transfer, and controller performance into
  one weighted score.
- Do not train on the real-track reference corpus in the core protocol.
- Do not treat MPPI or another model-based controller as an optimal oracle.
- Do not infer geometric or dynamic infeasibility solely from controller failure.
- Do not tune the RL method or reference controller independently for each generator.

## Experimental Units

Let \(G_1,\ldots,G_m\) denote course generators. For each \(G_i\), train the same RL
algorithm using:

- the same vehicle and dynamics contract;
- the same simulator version and integration step;
- the same observation and action spaces;
- the same reward and termination definitions;
- the same policy architecture and optimization hyperparameters;
- the same environment-interaction and policy-update budgets; and
- a common schedule of independent policy seeds.

Only the training-course distribution changes across rows of the experiment. Record
generation attempts, feasible courses observed, simulator interactions, updates,
wall-clock time, and compute separately so that equal interaction budgets do not hide
generator cost.

Each trained policy is evaluated on immutable generated-course suites
\(E_1,\ldots,E_m\) and a real-track suite \(E_{\mathrm{real}}\). The real-track corpus
is evaluation-only because training on a finite real corpus would introduce
memorization and augmentation-policy confounds.

## Transfer Matrix

For outcome \(Y\), define \(M^Y_{ij}\) as the aggregate performance of policies trained
on \(G_i\) and evaluated on \(E_j\). The real-track column is
\(M^Y_{i,\mathrm{real}}\). Every cell retains policy-seed, course, and rollout-seed
outcomes before aggregation.

Report:

- diagonal, in-source performance \(M^Y_{ii}\);
- all off-diagonal cells;
- leave-source-out and worst-source summaries;
- the real-track column;
- course and policy-seed sensitivity; and
- training exposure and cost.

Diagonal-to-off-diagonal differences across unlike suites are not the primary transfer
estimand. Within evaluation column \(j\), compare training distributions on the same
courses:

\[
  \Delta^Y_{ij}=M^Y_{ij}-M^Y_{jj}
\]

for higher-is-better outcomes, with the sign reversed or stated explicitly for
lower-is-better outcomes. The real-track column has no privileged diagonal; policies
are compared directly through paired course outcomes.

## Evaluation Suites

Construct generated evaluation suites after raw feasibility accounting, using the same
frozen descriptor transform, absolute admissible bounds, spectrum strata, diversity
rules, and minimum suite-size contract already proposed by the survey. Preserve raw
generation and repair outcomes separately from suite selection.

Report two complementary analyses:

1. **Suite-specific/full-suite analysis:** every course in each released \(E_j\),
   showing practical transfer over the selected suite without claiming to enumerate
   the generator's full support.
2. **Common-stratum analysis:** before evaluation, freeze the intersection of
   descriptor strata represented across all compared generators, fixed stratum
   weights, and restricted course sets \(E_j^{\cap}\). Report missing strata and
   insufficient-course shortfalls separately rather than silently shrinking the
   comparison.

The real-track suite uses the same vehicle, domain, units, frames, descriptor
extraction, and feasibility contract. Its composition and conditioning variables are
frozen before policy evaluation.

## Reference-Controller Calibration

Every evaluation course is also attempted by a declared model-based reference
controller. The benchmark contract is controller-generic; MPPI is the initial proposed
ground-racing implementation.

Freeze the controller model, objective, constraints, horizon, sampling budget,
controller-seed set, compute budget, and termination rules before evaluation. Tune the
reference controller once on a declared calibration set that is disjoint from all
training and evaluation suites; generator-specific tuning is prohibited. MPPI
aggregates must state how its stochastic controller seeds are combined.

Reference-controller reporting has two parts:

1. **Outcome calibration:** completion, collision, rule violation, timeout, or
   inconclusive budget exhaustion for RL and the reference controller.
2. **Conditional performance:** lap time, progress, mean speed, tracking error, and
   control effort on explicitly stated subsets, particularly courses completed by
   both methods.

Example paired quantities are:

\[
  R^{\mathrm{time}}_{icp}
  = \frac{\overline{T}^{\mathrm{RL}}_{icp}}
  {\widetilde{T}^{\mathrm{ref}}_c},
  \qquad
  \Delta v_{icp}
  = \overline{v}^{\mathrm{RL}}_{icp}
  -\widetilde{v}^{\mathrm{ref}}_c,
\]

where the overbar denotes the declared RL rollout aggregation and the tilde denotes
the preregistered aggregation over reference-controller seeds. Consistent MPPI success
with RL failure is diagnostic evidence of a training or policy limitation conditional
on the declared reference model and controller configuration. Failure by both methods
identifies a difficult or potentially infeasible controller-envelope case, but does
not establish geometric or dynamic infeasibility. MPPI outcomes are calibration
evidence and are not part of a generator ranking.

## Statistical Analysis

Evaluate policies on common immutable course identifiers and use common rollout random
streams where meaningful. For comparisons between training generators:

- aggregate rollouts within each policy-seed/course cell;
- preserve policy-seed clustering across all courses;
- resample policy seeds, courses, and nested rollout seeds hierarchically;
- report paired intervals for within-column contrasts;
- report full failure taxonomies rather than conditioning every metric on success; and
- state outcome-specific denominators and censoring rules.

Success, violation, time, speed, tracking, control effort, generator yield, descriptor
coverage, feasible diversity, and interoperability remain separate outcomes.

## Manuscript Changes

### Section 8: Metrics

Add a subsection defining the generator-to-suite transfer matrix, within-column
training-distribution contrasts, the evaluation-only real-track column, and the
two-part reference-controller comparison.

### Section 9: Benchmark Protocol

Add a subsection specifying the crossed training/evaluation design, fixed training
budgets, common policy seeds, immutable generated and real suites, full-suite and
common-stratum analyses, reference-controller execution, and hierarchical paired
uncertainty.

### Section 12: Research Hypotheses

Strengthen H6 so that cross-generator training utility is falsifiable through the
transfer matrix, paired real-track contrasts, preregistered within-column
$\Delta^Y_{ij}$ bounds, outcome directions, and multiplicity rules.

### Introduction And Conclusion

Add one sentence to each identifying cross-generator training utility as a proposed
benchmark objective. Preserve the existing statement that the distributions, suites,
CourseSpec implementation, simulator/RL adapters, and reference controllers are
planned rather than released.

## Acceptance Criteria

The manuscript change is acceptable when:

- generator training utility is the primary controller-based evaluation claim;
- MPPI is presented only as an initial ground-domain reference implementation;
- real tracks are evaluation-only in the core protocol;
- comparisons use the same evaluation courses within each column;
- full-suite and common-stratum results are both required;
- training interactions and generator cost are reported separately;
- reference-controller tuning data, stochastic seeds, and compute budget are frozen
  independently of generator identity;
- failure-aware outcomes precede conditional speed or lap-time comparisons;
- no scalar generator ranking is proposed; and
- no text implies that the empirical matrix, MPPI adapter, or real-track suite already
  exists.
