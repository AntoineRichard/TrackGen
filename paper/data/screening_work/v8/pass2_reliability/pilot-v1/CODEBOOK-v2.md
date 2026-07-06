# Pass-2 Codebook v2 (Prospective Draft)

## Status and scope

This is a prospective, frozen-for-next-round decision codebook distilled from pilot disagreement patterns and independent review. It does not contain source-specific answer keys and does not revise the existing v1 release. Every non-missing label needs a source-addressable locator for the particular claim.

## Representation family

1. Code a representation only when it is source-native and course-defining: the source explicitly defines, emits, consumes, serializes, directly inspects, or releases it as course state, parameterization, or a reusable course artifact.
2. Do not infer a representation from a renderer mesh, simulator internals, physics shape, occupancy map, coordinate array, visualization, cache, import, export, or downstream conversion unless the source establishes that object as a course-defining representation.
3. Use multiple labels only when separate course-defining representations are each directly evidenced; each multi-label assignment requires a distinct locator. Do not add labels merely because a pipeline has stages. Use `hybrid` only for an explicitly composite representation whose definition requires the components together.
4. A fixed benchmark or simulator may receive a directly documented fixed-course representation. Otherwise use `NR`; do not infer undocumented substrate details.

## Generator family

1. `constructive` requires explicit rules, assembly, grammar, geometry construction, or parameter-to-course computation. Random initialization or random parameter values alone do not add `stochastic_procedural`.
2. `stochastic_procedural` requires a named random sampling or stochastic assembly step that determines alternative topology or geometry. It may combine with `constructive` only when both the constructor and the geometry-determining stochastic operation are directly evidenced.
3. `search_evolutionary` requires explicit iterative candidate search in which an objective, fitness value, comparison, or selection rule affects which course candidate persists, is varied, or is returned. Parameter fitting, one-shot optimization, and random sampling without candidate selection are insufficient. It may combine with `constructive` or `stochastic_procedural` only when a separately evidenced constructor or stochastic geometry operator supplies candidates to the search.
4. `learned_generative` requires a trained model to output course state, parameters, or geometry. A learned controller is not a learned generator solely because it is evaluated on courses.
5. `environment_design` requires selection, adaptation, or optimization of an environment/course distribution using learner, agent, or task-performance feedback. Combine it with `learned_generative` only when both mechanisms are separately evidenced.
6. `human_designed` requires human course-defining layout decisions, not merely choosing a seed or inspecting output. It can combine with `constructive` for an evidenced authoring-plus-construction workflow.
7. `repair_projection` is a generator family only when repair or projection is the course-producing mechanism that transforms an incomplete, invalid, or proposed course into the final course. A downstream validity repair does not establish this generator family by itself. It may combine with `search_evolutionary`, `constructive`, or `learned_generative` only when the source separately establishes both candidate production and the repair/projection mechanism.
8. `selection_replay` is retrieval, replay, permutation, or selection of already complete courses. Assembling new geometry from primitives is `constructive`.
9. Each compatible multi-label assignment requires separately located evidence for every mechanism; a shared pipeline description or one operation described with several synonyms is insufficient.

## Generation role

1. `geometry_synthesis` creates new course geometry or course-defining spatial structure.
2. `task_selection` chooses, weights, schedules, or adapts among already defined courses/tasks without changing their geometry.
3. `mutation` requires an explicit operation that transforms an existing complete course into another candidate. It may combine with `geometry_synthesis` only if both the initial construction and whole-course mutation are directly evidenced.
4. `repair` requires an explicit operation on an existing course candidate that changes course-defining state to remove a stated violation or restore feasibility. Penalty, rejection, or a validity label alone does not establish the role. It may combine with `geometry_synthesis` or `mutation` only when the source separately evidences the initial or mutating operation and the subsequent repair.
5. `serialization` requires an explicit source contribution that converts or emits an existing course definition into a persistent, exchange, or simulator-consumable representation without thereby changing course-defining geometry. An incidental save/export call is insufficient. It may combine with a course-changing role only when both contributions are separately evidenced.
6. `benchmark_only` means the source contributes a fixed course/benchmark for use or evaluation and establishes no source-native course-changing or task-selection operation. It is mutually exclusive with `geometry_synthesis`, `task_selection`, `mutation`, and `repair`. It may combine with `serialization` when the source explicitly contributes the reusable benchmark encoding.
7. `boundary_case` requires explicit generation, selection, or curation for rare, adversarial, failure-inducing, or limit-testing cases. Reporting a worst result or ordinary distribution tail is insufficient. It may combine with the separately evidenced operation that creates or selects those cases, or with `benchmark_only` when the fixed benchmark is explicitly a boundary-case set.
8. `NR` means no source-native course-operation role is established. It is sole-valued and is not shorthand for reviewer uncertainty.

## Validity and missingness

1. `by_construction` requires an explicit rule, parameterization, or invariant that ensures the stated validity property before candidate testing.
2. `rejection` requires generation followed by an explicit test that discards failing candidates. Bounded sampling is `by_construction` when bounds guarantee validity without candidate-level discard.
3. `penalty`, `repair_projection`, and `constraint_solver` each require the corresponding mechanism. Use multiple validity labels only for separate evidenced stages or conditions.
4. `simulation_validation` requires a simulated run whose outcome assesses, accepts, rejects, or reports course feasibility or validity. Simulation used only to train or evaluate an agent is not enough.
5. `not_reported` applies when a source-native generation or selection contribution makes a field applicable but the frozen source does not state the mechanism. Its locator identifies the inspected generation/selection material.
6. `NR` is structural non-applicability: no source-native contribution exists to which the field applies. It is sole-valued for analytical fields.

## Core and supporting evidence

1. A source is `core` for these fields when it defines, implements, or releases a parameterized or stochastic mechanism that changes course geometry or course-defining spatial constraints.
2. A source is `supporting` when it establishes a fixed-course interface, benchmark, simulator constraint, or reporting practice only. Supporting evidence may code directly documented fixed-course representation or `benchmark_only`, but uses `NR` for an unestablished generator, course-changing role, and associated validity strategy.
3. The label `simulator` alone does not decide the evidence tier.

## Availability evidence

1. `official_open` requires a frozen, inspectable official release/repository/documentation locator connecting the artifact to the authors, project, publisher, or official organization, including revision or path and public access terms.
2. `unofficial_open` requires an inspectable third-party release explicitly linked to the work; it must not be attributed as an author release.
3. `closed` requires affirmative evidence of restricted, proprietary, request-only, or otherwise unavailable material. `not_found` requires the documented source-first availability check to find no qualifying release. `not_applicable` means no code or course asset could reasonably be released.
4. Preserve the repository rule: code_status/asset_status may be sole NR when availability was not assessed or not reported. This draft does not adopt the review suggestion that completed rows cannot use `NR`.

## Next-round protocol

1. Freeze this codebook, a concise decision table, and synthetic boundary examples. Discuss only the synthetic examples during calibration; do not use pilot sources or adjudications as answer keys.
2. Recode all 75 sources under frozen v2. Validate completed rows with the existing release validator before reliability comparison.
3. Draw a fresh blind holdout and require exact-set >=0.80 for each of the eight fields. The draft gate also requires no repeated ambiguity class after locked ratings; revise the codebook rather than adjudicating individual sources to manufacture agreement.
4. This draft does not require two consecutive 30-source rounds. Stronger pre-submission replication with an additional fresh blind holdout is recommended.
