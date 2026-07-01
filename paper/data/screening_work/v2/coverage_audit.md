# v2 Coverage Audit

## Status and boundaries

This is a corpus-informed, provisional coverage audit of the canonical unsealed
final draft at
`.worktrees/survey-foundation/paper/data/screening_work/v2/decision_drafts/adjudications.csv`.
It uses only local records: that draft, the frozen v2 candidate metadata,
the v2 taxonomy and protocol, the v2 adjudication-work README, the release
decision, and the local scope/planning documents. No external search or research
fact was used.

The source draft is explicitly unsealed and requires accountable-author inspection;
therefore every count and observation below is a provisional corpus property, not a
final scientific conclusion. `included` denotes a source-native inclusion under the
screening codebook and `boundary` denotes a fixed-course transfer, not a source-native
course-generation contribution.

## Confirmed Local Facts

### Population and provisional decisions

The adjudication draft contains 102 candidates. Its 40 current final-draft
`included` or `boundary` candidates are the audit cohort.

| Provisional decision status | All 102 candidates | Audit cohort |
| --- | ---: | ---: |
| included | 35 | 35 |
| boundary | 5 | 5 |
| excluded | 62 | 0 |
| **Total** | **102** | **40** |

| Criterion | All 102 candidates | Audit cohort |
| --- | ---: | ---: |
| include-1 | 29 | 29 |
| include-2 | 5 | 5 |
| include-4 | 1 | 1 |
| boundary | 5 | 5 |
| exclude-out-of-scope | 31 | 0 |
| exclude-insufficient-detail | 19 | 0 |
| exclude-fixed-racing-line | 6 | 0 |
| exclude-appearance-dynamics | 4 | 0 |
| exclude-traffic-only | 2 | 0 |
| **Total** | **102** | **40** |

The cohort's recorded access statuses are 22 `full_text`, 3
`full_text_and_supplement`, and 15 `official_documentation`. None is
`abstract_only`.

### Scope anchors recorded locally

The local protocol defines eligible racing robots to include aerial drones, ground
cars, and surface or marine craft. The local planning material keeps domain,
representation, and generation mechanism orthogonal, and calls separately for
validity, difficulty/diversity, metrics, benchmarks, simulator/export portability,
and RL-facing evaluation. These are scope axes for this audit, not evidence that
every candidate covers every axis.

The unsealed deciding facts record examples of direct material evidence, including:

- `C0026`: random drone-racing gate primitives, relative poses, counts, and varying
  complexity.
- `C0038`: centerline-plus-width track inputs, smooth-curvature expectations, lap
  time, and energy as a boundary transfer.
- `C0042`: fixed-track minimum traversal time under friction-circle, actuator, and
  track-boundary constraints as a boundary transfer.
- `C0126`, `C0128`, `C0170`, `C0178`, `C0192`, `C0193`, and `C0210`: named map,
  scenario, geometry, CSV, API, mission, or YAML serialization/interface facts.
- `C0137` and `C0143`: connected-road validation, nonintersection, or turn
  constraints.
- `C0150`, `C0198`, `C0200`, and `C0209`: ordered maritime waypoint, gate, buoy, or
  route facts.

These are source-specific eligibility facts already recorded in the local draft. They
are not a completed, normalized coding of every representation, generator family,
validity strategy, difficulty level, or interface property.

## Metadata-Only Distribution

The following rough-domain labels are intentionally conservative. They use only
explicit frozen candidate metadata (`title`, `venue`, and, where present, the
controlled `discovery_stream` label), not an inferred vehicle type or a claim about
the candidate's method. `domain not explicit` is an information state, not an
adjacent-domain classification.

| Rough domain from explicit metadata | Candidates | Count | Exact source-type distribution |
| --- | --- | ---: | --- |
| ground/road explicit | C0001, C0008, C0038, C0087, C0122, C0126, C0127, C0128, C0137 | 9 | conference paper (5); simulator; official repository (1); journal article (1); book chapter; conference paper (1); workshop paper (1) |
| aerial explicit | C0026, C0073, C0074, C0104, C0165 | 5 | conference paper (2); journal article (3) |
| maritime explicit | C0150, C0198, C0200, C0209, C0210 | 5 | journal article (1); official competition specification (1); competition specification (2); competition evaluation repository (1) |
| adjacent/other explicit | C0013, C0053, C0054, C0118 | 4 | conference paper (2); software (1); journal article (1) |
| domain not explicit in frozen metadata | C0002, C0042, C0057, C0069, C0079, C0088, C0119, C0135, C0143, C0153, C0170, C0178, C0180, C0182, C0185, C0192, C0193 | 17 | conference paper (4); journal article (2); simulator; official repository (1); competition (1); conference paper; benchmark (1); preprint (1); standard (1); track dataset repository (1); conference paper; official software (1); dataset repository (1); simulator software (1); software documentation (1); official documentation; file-format specification (1) |
| **Total** |  | **40** |  |

Candidate-level labels and source types:

| Candidate | Rough domain label | Exact frozen `source_type` |
| --- | --- | --- |
| C0001 | ground/road explicit | simulator; official repository |
| C0002 | domain not explicit | simulator; official repository |
| C0008 | ground/road explicit | conference paper |
| C0013 | adjacent/other explicit | conference paper |
| C0026 | aerial explicit | conference paper |
| C0038 | ground/road explicit | journal article |
| C0042 | domain not explicit | conference paper |
| C0053 | adjacent/other explicit | software |
| C0054 | adjacent/other explicit | journal article |
| C0057 | domain not explicit | journal article |
| C0069 | domain not explicit | journal article |
| C0073 | aerial explicit | conference paper |
| C0074 | aerial explicit | journal article |
| C0079 | domain not explicit | conference paper |
| C0087 | ground/road explicit | book chapter; conference paper |
| C0088 | domain not explicit | competition |
| C0104 | aerial explicit | journal article |
| C0118 | adjacent/other explicit | conference paper |
| C0119 | domain not explicit | conference paper; benchmark |
| C0122 | ground/road explicit | conference paper |
| C0126 | ground/road explicit | conference paper |
| C0127 | ground/road explicit | conference paper |
| C0128 | ground/road explicit | conference paper |
| C0135 | domain not explicit | conference paper |
| C0137 | ground/road explicit | workshop paper |
| C0143 | domain not explicit | conference paper |
| C0150 | maritime explicit | journal article |
| C0153 | domain not explicit | preprint |
| C0165 | aerial explicit | journal article |
| C0170 | domain not explicit | standard |
| C0178 | domain not explicit | track dataset repository |
| C0180 | domain not explicit | conference paper; official software |
| C0182 | domain not explicit | dataset repository |
| C0185 | domain not explicit | simulator software |
| C0192 | domain not explicit | software documentation |
| C0193 | domain not explicit | official documentation; file-format specification |
| C0198 | maritime explicit | official competition specification |
| C0200 | maritime explicit | competition specification |
| C0209 | maritime explicit | competition specification |
| C0210 | maritime explicit | competition evaluation repository |

## Research Questions: Unassessed Coverage Cells

### Why these are unassessed

The frozen candidate schema has no `domain`, `representation_family`,
`generator_family`, `generation_role`, `validity_strategy`, difficulty-spectrum, or
simulator/export/RL-interface field. The adjudication draft records status, criterion,
access, locator, deciding fact, and notes, but likewise has no normalized coverage
codes. The taxonomy enumerates allowed labels; it does not assign them to candidates.

Consequently, an unassessed cell below does **not** mean zero evidence or zero
literature. It means that the current local records do not provide directly inspected,
candidate-level evidence coded to that cell. Direct source inspection and a controlled
coding pass are needed before a coverage claim can be made.

### Domain x representation family

The taxonomy's representation families are `segment_grammar`, `tile_grid`,
`parametric_curve`, `sampled_centerline`, `centerline_plus_width`, `boundary_pair`,
`gate_poses`, `waypoint_graph`, `occupancy_heightfield_mesh`, `simulator_native`, and
`hybrid`.

- Ground/road: `C0038` directly names centerline-plus-width inputs, but no normalized
  row-level code exists. The remaining plausible ground/road representation cells are
  unassessed: segment grammar, tile grid, parametric curve, sampled centerline,
  boundary pair, gate poses, waypoint graph, occupancy/heightfield/mesh,
  simulator-native, and hybrid.
- Aerial: `C0026` directly records gate primitives and relative poses, but no
  normalized row-level code exists. The remaining plausible aerial representation
  cells are unassessed: segment grammar, tile grid, parametric curve, sampled
  centerline, centerline-plus-width, boundary pair, waypoint graph,
  occupancy/heightfield/mesh, simulator-native, and hybrid.
- Maritime: all listed representation-family cells remain unassessed at the normalized
  coverage level. The local draft records ordered waypoints and buoy/gate constraints,
  but it does not assign a taxonomy representation-family label.

### Domain x generator family

The taxonomy's generator families are `constructive`, `stochastic_procedural`,
`search_evolutionary`, `learned_generative`, `environment_design`, `human_designed`,
`repair_projection`, and `selection_replay`.

All 24 ground/road, aerial, and maritime domain x generator-family cells are
unassessed as normalized coverage cells. Several deciding facts contain potentially
relevant verbs (for example, generating gate layouts, selecting road networks,
validating roads, or randomizing buoy routes), but none supplies a controlled
candidate-to-`generator_family` assignment. Those descriptions are leads for direct
coding, not a defensible absence/presence matrix.

### Other scope axes that require direct coding

- **Generator roles:** no controlled candidate-level assignment distinguishes geometry
  synthesis, task selection, mutation, repair, serialization, benchmark-only, or
  boundary-case roles. The `include-*` criterion cannot serve as a substitute.
- **Validity:** no candidate-level `validity_strategy` code distinguishes construction,
  rejection, penalty, repair/projection, solver, simulation validation, or not
  reported. The recorded C0038, C0042, C0137, and C0143 facts warrant direct
  extraction, not a comparative conclusion.
- **Difficulty spectrum:** isolated local facts mention varying complexity (`C0026`),
  minimum feasible time (`C0042`), and task evaluation (`C0180`), but no common
  difficulty scale, distribution, or reporting denominator is recorded.
- **Simulation, export, and RL interface:** specific interface evidence exists, but
  there is no structured indicator for import/export format, loader validation,
  coordinate convention, reset semantics, training-distribution use, or RL API.
  The listed serialization/API records are therefore not sufficient to measure
  cross-domain portability or RL-interface coverage.

## Priority Prospective Queries

These are targeted future search questions only. They were not executed for this
audit, and their wording is not evidence.

1. `"drone racing" ("track generator" OR "course generation") ("gate poses" OR "centerline" OR spline) validity`
   - Obtain direct aerial evidence that distinguishes gate-pose, curve, centerline,
     and simulator-native representations and documents feasibility checks.
2. `("autonomous race car" OR "ground racing") ("segment grammar" OR "tile grid" OR "boundary pair") "track generation"`
   - Test unassessed ground/road representation families with primary technical
     artifacts, rather than treating road-generation titles as coverage.
3. `("autonomous surface vessel" OR maritime OR RoboBoat OR RobotX) ("course generator" OR randomized) (waypoint OR buoy OR gate)`
   - Seek directly inspectable maritime generator-family, representation, and
     validity evidence beyond competition-course specifications.
4. `("drone racing" OR "autonomous racing" OR maritime) (evolutionary OR generative OR "reinforcement learning") "course generation"`
   - Establish, by domain, whether constructive, stochastic, search/evolutionary,
     learned-generative, environment-design, repair/projection, and selection/replay
     families are actually supported by source evidence.
5. `"racing course generation" (validity OR feasibility OR "self-intersection" OR "curvature constraint" OR "simulation validation")`
   - Extract comparable validity strategies, their input representation, and the
     condition each check establishes.
6. `("drone racing" OR "race car" OR "surface vessel") course (difficulty OR diversity OR curriculum OR generalization) generator`
   - Look for explicit difficulty-spectrum definitions, sampling distributions, and
     train/test partition or curriculum evidence.
7. `("OpenDRIVE" OR Lanelet2 OR CommonRoad OR "mission plan") (import OR export OR serialization OR validation) racing simulator`
   - Audit format conversion, coordinate frames, geometry/asset loss, loader
     validation, and post-import behavior rather than format-name mentions alone.
8. `("racing simulator" OR "drone racing gym" OR maritime) (RL OR Gym OR environment) (reset OR distribution OR benchmark)`
   - Establish direct evidence for the RL interface, reset semantics, exposed course
     parameters, and separation of training distributions from fixed evaluation suites.

## Local Inputs Consulted

- `.worktrees/survey-foundation/paper/data/screening_work/v2/decision_drafts/adjudications.csv`
- `.worktrees/survey-foundation/paper/data/screening_inputs/v2/candidates.csv`
- `.worktrees/survey-foundation/paper/data/screening_inputs/v2/taxonomy.json`
- `.worktrees/survey-foundation/paper/data/screening_inputs/v2/protocol.md`
- `.worktrees/survey-foundation/paper/data/screening_work/v2/README.md`
- `.worktrees/survey-foundation/paper/data/screening_decisions/v2/decision.csv`
- `.worktrees/survey-foundation/paper/sections/03-scope-definitions.tex`
- `.worktrees/survey-foundation/paper/notes/survey-structure.md`
