# Pass-2 Codebook Reliability Review

## Scope and finding

This is a prospective codebook review, not an adjudication. It does not recommend a
value for any individual source.

The 18-source reliability sample misses the 0.80 exact-set gate for all four target
fields: `representation_family` 8/18 (0.444), `generator_family` 12/18 (0.667),
`generation_role` 8/18 (0.444), and `validity_strategy` 9/18 (0.500). The 42
disagreement rows are patterned rather than random: 10 representation, 6 generator,
10 role, and 9 validity disagreements account for 35 of them. Most are an added,
omitted, or differently scoped secondary label, rather than mutually incompatible
readings of a source.

The current Pass-2 instructions specify label ordering and locator requirements, but
not the decision boundaries that produced these patterns. The taxonomy is a label
list, not a codebook. The result is that coders reasonably switch among an author
described course representation, a derived implementation substrate, a downstream
simulator representation, and a survey-inferred abstraction.

There is also a measurement effect. `paper/scripts/coding_reliability.py` compares
the whole semicolon-separated set as one nominal category. Thus `a` versus `a;b`
is a total disagreement, just like `a` versus `c`; its kappa is a sparse powerset
category kappa. This is appropriate as a stringent release gate only after the
codebook states exactly when a second label is warranted. It is not evidence that
multi-label coding itself is unreliable. The comparison function validates only the
evidence tier, whereas the release validator validates the other controlled fields.
Run the release validator before a reliability comparison so invalid labels, duplicate
labels, and noncanonical order cannot enter the measurement.

## Smallest prospective rules

Add the following rules to the Pass-2 codebook. They preserve genuine multi-label
claims, but make every added label require a distinct, source-addressable fact.

### 1. Scope and representation

1. A representation label is source-native only when the source explicitly defines,
   emits, consumes, serializes, or directly inspects that structure as the course
   state, course parameterization, or reusable course artifact. The locator must name
   the relevant data object, schema, algorithm variable, or artifact path.
2. Do not code an inferred implementation substrate. A renderer mesh, occupancy map,
   simulator internals, physics shape, sampled visualization, or coordinate array is
   not a representation label merely because the course passes through it. It becomes
   codeable only when the source establishes it as a course-defining object or a
   released course artifact.
3. Code only course-defining representations. Do not code every supported import,
   export, cache, rendering, or simulator format. A derived export receives a label
   only when the source presents it as an independently reusable course definition,
   not just as a conversion target.
4. Use multiple representation labels only when two distinct, course-defining
   representations are each directly established. Do not add `hybrid` merely because
   a pipeline has stages or a course is rendered in several forms. Use `hybrid` only
   when the source describes one composite course representation whose definition
   requires the named components together.
5. A fixed benchmark or simulator source may code a directly documented fixed-course
   representation, but it must not acquire a representation from the simulator's
   undocumented substrate. If no directly documented course representation exists,
   use `NR` rather than infer one.

### 2. Generator family

1. `constructive` means explicit rules, assembly, grammar, geometry construction, or
   parameter-to-course computation that creates the course. Random initialization or
   random parameter values alone do not add `stochastic_procedural`.
2. `stochastic_procedural` means a named random sampling or stochastic assembly step
   itself determines alternative course topology or geometry. It may co-occur with
   `constructive` only when the source establishes both an explicit constructor and a
   geometry-determining stochastic operator. A seed, noise term, randomized order, or
   stochastic simulator execution without that role is insufficient.
3. `learned_generative` means a trained model directly outputs course state,
   course parameters, or course geometry. A learned driving/controller policy is not
   a learned generator merely because it is evaluated on courses.
4. `environment_design` means a procedure selects, adapts, or optimizes an
   environment/course distribution using learner, agent, or task-performance feedback.
   It may co-occur with `learned_generative` only when a learned model generates the
   course and the source separately establishes that feedback-driven environment
   design loop. A learned policy that places objects once is not, by itself, both.
5. `human_designed` requires a human to make course-defining layout choices, not just
   choose a seed, configure an experiment, or inspect outputs. It may co-occur with
   `constructive` when an authoring interface combines human layout decisions with
   explicit automated construction.
6. `selection_replay` is selection, replay, permutation, or retrieval of already
   complete courses/scenarios without construction of new geometry. Assembling new
   geometry from primitives is `constructive`, even when the primitives came from a
   library.

### 3. Generation role

1. `geometry_synthesis` means the contribution creates a new course geometry or
   course-defining spatial structure from parameters, rules, a learned output, or
   components.
2. `mutation` requires an explicit operation that transforms an existing complete
   course into another course candidate. Changing a parameter while constructing from
   scratch, generic optimizer updates, or policy mutations are not course mutation.
   It may co-occur with `geometry_synthesis` only if both the initial construction and
   the whole-course mutation operation are directly established.
3. `task_selection` requires choosing, weighting, scheduling, or adapting among
   already defined courses/tasks without changing their geometry. Do not use it for
   choosing parameters that immediately synthesize a new course.
4. `benchmark_only` means the source contributes a fixed course/benchmark for use or
   evaluation and establishes no source-native course-changing operation. It is
   mutually exclusive with `geometry_synthesis`, `mutation`, `repair`, and
   `task_selection` for the same source-level contribution.
5. `NR` means the retained source supplies none of these course-operation roles. It
   is mutually exclusive with every generation-role label. It is not a synonym for
   "the reviewer did not find a method."

### 4. Validity strategy and missingness

1. `by_construction` requires an explicit parameterization, invariant, or generation
   rule that guarantees the stated validity property before a candidate is tested.
   Constraints that are merely listed as desired properties do not qualify.
2. `rejection` requires generation followed by an explicit validity test that discards
   failing candidates. Bounded sampling is `by_construction` when bounds guarantee
   validity and no candidate-level discard is performed.
3. `penalty`, `repair_projection`, and `constraint_solver` require the correspondingly
   named or unmistakably described mechanism. A penalty is not rejection unless
   candidates are actually discarded. Multiple validity labels are allowed only for
   separate, evidenced stages or validity conditions; a loose description of one
   mechanism must receive one label.
4. `simulation_validation` requires a simulated run of generated or parameterized
   course instances whose outcome is used to assess, accept, reject, or report the
   course's validity/feasibility. Simulation used only to train or evaluate an agent,
   demonstrate a benchmark, or render an environment is not simulation validation.
5. `not_reported` applies only when a source-native course-generation or
   course-selection contribution makes a validity strategy applicable, but the frozen
   source states none. Its locator is the inspected generation/selection section and
   the coding note records that no validity mechanism is stated there.
6. `NR` is structural non-applicability: the source has no source-native generator or
   selection contribution to which a validity strategy could apply. It is not an
   absence-of-reporting code. For the four target fields, `NR` must be the sole value.

### 5. Evidence tier and release status

1. A simulator/environment source is `core` for these fields when the frozen source
   itself defines, implements, or releases a parameterized or stochastic mechanism
   that changes course geometry or course-defining spatial constraints. Stochastic
   weather, traffic, sensors, dynamics, or rendering alone does not cross this
   boundary.
2. It is `supporting` when it establishes only a fixed-course interface, benchmark,
   simulator constraint, or reporting practice mapped by the protocol. Supporting
   evidence may code that direct fixed-course representation or `benchmark_only`, but
   must use `NR` for an unestablished generator, course-changing role, and associated
   validity strategy. The word "simulator" never determines the tier by itself.
3. `code_status` and `asset_status` are status fields, not inferred reproducibility
   judgments. Do not use `NR` in a completed coding row. Make `asset_status` an
   explicit taxonomy entry with the same five scalar values as `code_status`.
4. `official_open` requires a frozen, inspectable release/repository/documentation
   locator that connects the code or asset to the authors, project, publisher, or
   official organization, plus a revision/path and public access terms. `unofficial_open`
   requires an inspectable third-party release explicitly linked to the work or artifact;
   it must not be attributed as an author release.
5. `closed` requires affirmative evidence that the relevant material exists but is
   restricted, proprietary, request-only, or otherwise unavailable. `not_found`
   requires the documented, source-first availability check to find no qualifying
   release evidence. `not_applicable` means this report has no code or course asset
   that could reasonably be released; it is not a fallback for an unsearched source.
6. For every non-`NR` analytical label and every status, retain evidence of the
   specific claim: source/page or pinned repository path, revision or archive,
   artifact type, and for status the ownership/linkage and access or license statement.
   A bare project URL, a citation to a simulator dependency, or a search result is
   insufficient.

## Reliability plan and stopping rule

1. Freeze these rules, a one-page label decision table, and six synthetic boundary
   examples. Do not use the current source-specific disagreements as answer keys.
   Validate each completed row with `validate_pass2_draft.py --coding-output` before
   comparison.
2. Conduct a short, non-scored calibration using the synthetic examples. Coders may
   discuss the rules, not individual survey sources. Revise wording once, freeze a
   versioned codebook, and prohibit further rule changes during the test round.
3. Draw a fresh frozen, stratified 30-source holdout. Ensure it contains apparent
   multi-representation pipelines, constructive/stochastic procedures, learned or
   environment-design methods, fixed benchmarks, simulator environments, and sources
   with and without release evidence. Two independent blind coders use identical
   packets and do not see the current ratings, disagreements, or one another's work.
4. Use exact canonical label-set agreement as the release gate: each target field
   must achieve at least 24/30 (0.80). Report label-level precision/recall or Jaccard
   only as a diagnostic for multi-label behavior, never as a substitute for the
   exact-set gate. Report kappa descriptively, not as the sole decision statistic.
5. After locking ratings, classify disagreements by the rule number above. Stop the
   round and revise the codebook if any rule class causes two or more disagreements,
   even if a numerical gate happens to pass. Do not resolve sources one at a time to
   manufacture agreement.
6. Stop codebook iteration and unlock production coding only after two consecutive
   fresh blind holdouts pass all four exact-set gates with no recurring rule class,
   using an unchanged codebook. If the same field fails two rounds for the same rule
   class, stop adding exceptions and split or redefine that taxonomy dimension before
   another test.

## Implementation note

No files other than this review were modified. The prospective rules would require a
later codebook/protocol change and corresponding validator/reliability tests, but no
such changes are made here.
