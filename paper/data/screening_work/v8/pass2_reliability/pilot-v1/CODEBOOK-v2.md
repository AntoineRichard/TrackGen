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
3. `learned_generative` requires a trained model to output course state, parameters, or geometry. A learned controller is not a learned generator solely because it is evaluated on courses.
4. `environment_design` requires selection, adaptation, or optimization of an environment/course distribution using learner, agent, or task-performance feedback. Combine it with `learned_generative` only when both mechanisms are separately evidenced.
5. `human_designed` requires human course-defining layout decisions, not merely choosing a seed or inspecting output. It can combine with `constructive` for an evidenced authoring-plus-construction workflow.
6. `selection_replay` is retrieval, replay, permutation, or selection of already complete courses. Assembling new geometry from primitives is `constructive`.

## Generation role

1. `geometry_synthesis` creates new course geometry or course-defining spatial structure.
2. `mutation` requires an explicit operation that transforms an existing complete course into another candidate. It may combine with `geometry_synthesis` only if both the initial construction and whole-course mutation are directly evidenced.
3. `task_selection` chooses, weights, schedules, or adapts among already defined courses/tasks without changing their geometry.
4. `benchmark_only` means the source contributes a fixed course/benchmark for use or evaluation and establishes no source-native course-changing operation. It is mutually exclusive with `geometry_synthesis`, `mutation`, `repair`, and `task_selection` for that contribution.
5. `NR` means no source-native course-operation role is established. For analytical fields, `NR` is sole-valued and is not shorthand for reviewer uncertainty.

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
