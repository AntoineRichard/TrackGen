# Generator Implementation Gap Roadmap

## Status

- **Baseline repository revision:** `1903402`.
- **Status:** provisional engineering audit, not a finalized literature synthesis.
- **Screening state:** `paper/data/evidence.csv` and `paper/data/claims.csv` contain zero data rows.
- **Candidate ledger:** 202 verified candidates: 198 `candidate` and 4 `excluded`.
- **Decision rule:** revisit every literature-derived implementation decision after final screening; this document does not convert candidate status into finalized evidence.

## Purpose and Evidence Labels

This roadmap translates currently visible implementation gaps into bounded engineering work. It is an ordering proposal, not a literature frequency count or a scalar ranking of methods.

| Label | Meaning in this document |
|---|---|
| **manuscript lead** | The current manuscript discusses the cited mechanism or contract as a relevant method or requirement. It is provisional and not direct evidence while screening is incomplete. |
| **candidate-only lead** | A verified candidate points to a potentially relevant method, but the current audit has not established a finalized method claim. |
| **boundary inspiration** | A cited body of work motivates a representation or downstream check, not evidence that it is a track generator. |
| **implementation inference** | A proposed code contract inferred from the current repository and manuscript; it is not a literature result. |

## Current Envelope

At revision `1903402`, the registered runtime contains six closed, planar track generators: `bezier`, `hull`, `polar`, `voronoi`, `checkpoint`, and `repulsive`. `fourier` is experimental and unregistered. The Phase 2 track path is generator-local selection or fallback, constant-spacing resampling, XPBD geometric projection, inflation, and final geometric validity.

There is a separate planar gate-generation path with five generators: `bezier`, `hull`, `polar`, `voronoi`, and `checkpoint`. Gate generation does not reuse the track path as a gate adapter, and `repulsive` is not a gate generator. Current `Track` and `GateSequence` buffers are fixed-batch mutable runtime outputs. Utilities for collision, progress, localization, speed-profile proxies, and rendering props are not generators.

The current implementation has no public raw-to-derived lineage, `CourseSpec` serialization, immutable course corpus, Gymnasium/mainstream simulator adapter, or dynamics/simulator feasibility certificate. CUDA capture is available for the fixed-shape track generators; `repulsive` has a host-controlled eager path. The existing documents under `docs/appendix/future-generators.rst` are historical/detail notes, not canonical ordering.

## Missing at a Glance

G1--G4 are the missing closed-track mechanisms closest to implementation: chain-code/direction sequences, curvature-profile/clothoid loops, segment/road-block grammars, and bounded search over an explicit genotype. G2 is boundary-inspired by `Theodosis2011GeneratingRacing` and `Heilmeier2020MinimumCurvature`, not direct track-generation evidence.

G5 is constructive graph/network generation and depends on a `CourseSpec` graph. G6 and G7 require data/model or learner-feedback subsystems; C1 is request compilation and constraint extraction, not a generator, and composes with G6 or another synthesizer. R1--R4 are missing course representations and semantics, not interchangeable generator plugins. A1--A2, V1--V2, and I1--I2 are enabling management, validation, and integration gaps and must not inflate generator counts. B1 racing-line optimization is downstream calibration and excluded as a generator.

## Engineering Gap Classes

| Class | Scope |
|---|---|
| Runtime generator mechanism | A bounded, seeded mechanism emitting course-defining geometry in the runtime pipeline. |
| Offline/search generator | A budgeted candidate producer, mutation operator, or learned model that may not fit CUDA capture. |
| Course representation/schema | A public typed representation for tracks, gates, routes, and their semantics. |
| Adaptive selection/management | Archive, replay, or curriculum logic acting on generated artifacts and traces. |
| Request compilation/constraint extraction | A typed compiler from constrained requests into auditable synthesizer inputs; it is not a generator. |
| Validation/repair | Declared predicates, repair stages, diagnostics, and raw-to-final provenance. |
| Simulator/export | Conversion to an external simulator or serialization format while recording preserved and lost semantics. |

## Main Gap Matrix

Effort bands are based on contract impact, not calendar time: **S** local implementation within an existing contract; **M** a new mechanism or bounded adapter; **L** a public representation or cross-stage change; **XL** a new subsystem with data, evaluation, or multiple external contracts.

| ID | Gap class | Literature basis | Nearest current capability | Exact missing behavior | Proposed output / adapter contract | Effort | Evidence status |
|---|---|---|---|---|---|---|---|
| G1 | Runtime generator mechanism | `doNascimento2021ProceduralGeneration`, *Procedural Generation of Isometric Racetracks Using Chain Code for Racing Games* | Fixed-shape `TrackGenerator`; checkpoint steering | Sample an explicit direction sequence, decode a closed centerline candidate within fixed bounds, and report construction/failure through P0 telemetry. | `TrackGenerator` plugin: seed + static `ChainCodeConfig` -> serializable `ChainCodeGenotype` -> bounded decoder -> current `Track` output plus P0 telemetry. | M | manuscript lead + implementation inference |
| G2 | Runtime generator mechanism | `Theodosis2011GeneratingRacing`; `Heilmeier2020MinimumCurvature`; **boundary inspiration only**, not track-generation evidence | Polar/Fourier-style smooth closed centerline ideas; XPBD | Sample a periodic curvature profile or clothoid/arc vocabulary, enforce bounded closure, and retain profile parameters only as raw diagnostics. | `TrackGenerator` plugin: seed + fixed-basis `CurvatureProfileConfig` -> current `Track` output plus P0 telemetry with raw closure residuals and parameters. | M | boundary inspiration (`Theodosis2011GeneratingRacing`, `Heilmeier2020MinimumCurvature`) + implementation inference |
| G3 | Runtime generator mechanism | `Li2020ImprovingGeneralization`, *Improving the Generalization of End-to-End Driving through Procedural Generation*; `Li2023MetaDriveComposing`, *MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning*; `Muktadir2023ProceduralGeneration`, *Procedural Generation of High-Definition Road Networks for Autonomous Vehicle Testing and Traffic Simulations* | Checkpoint path; no reusable segment vocabulary | Compose straights, turns, hairpins, chicanes, and connectors from explicit parameters with bounded closure/backtracking. | `SegmentProgram` genotype -> bounded compiler -> current raw centerline for closed single loops; open or branching compilation waits for `CourseSpec`. | L | manuscript lead (`Li2020ImprovingGeneralization`, `Muktadir2023ProceduralGeneration`); candidate-only lead (`Li2023MetaDriveComposing`) + implementation inference |
| G4 | Offline/search generator | `Togelius2006MakingRacing`, *Making Racing Fun Through Player Modeling and Track Evolution*; `Prasetya2016SearchBased`, *Search-based Procedural Content Generation for Race Tracks in Video Games*; `Alyaseri2024ComparativeAnalysis`, *Comparative Analysis of Metaheuristic Algorithms for Procedural Race Track Generation in Games*; `Kluck2021AutomaticGeneration`, *Automatic Generation of Challenging Road Networks for ALKS Testing based on Bezier Curves and Search*; `Nylnder2025SearchBased` | Generator-local best-of-K/fallback; no public genotype or budget trace | Budgeted mutation, evaluation, retention, and returned candidate selection: search `ChainCodeGenotype` first, then add `SegmentProgram` after G3; waypoint mutation is available only after R1. | Offline `SearchGenerator`: genotype, seed schedule, budget, objective vector, predicates -> `CandidateSet` + immutable attempt log; runtime export only through an adapter. | L | manuscript lead (`Togelius2006MakingRacing`, `Kluck2021AutomaticGeneration`); candidate-only lead (`Prasetya2016SearchBased`, `Alyaseri2024ComparativeAnalysis`, `Nylnder2025SearchBased`) + implementation inference |
| G5 | Offline/search generator | `Muktadir2023ProceduralGeneration`, *Procedural Generation of High-Definition Road Networks for Autonomous Vehicle Testing and Traffic Simulations*; `Mi2021HDMapGenHierarchical`, *HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps*; `EclipseSUMONodateNetgenerate` | Closed loop `Track`; ordered planar `GateSequence` | Construct a graph/network by topology sampling, then assign nodes, directed edges, lane/corridor attributes, route choice, and graph-aware validity. | `ConstructiveGraphGenerator`: seed + topology sampling -> `CourseSpec.route_graph`; requires the minimal immutable `CourseSpec` v0 graph dependency and emits an explicit route when an adapter needs one. | XL | manuscript lead + implementation inference |
| G6 | Offline/search generator | `Mi2021HDMapGenHierarchical`, *HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps*; `Rowe2025ScenarioDreamer`, *Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments*; `Sun2024DriveSceneGenGenerating`, *DriveSceneGen: Generating Diverse and Realistic Driving Scenarios from Scratch* | No learned generator or training-data contract | Train and sample a graph/latent/scenario model while validating every emitted structure against immutable `CourseSpec`. | Offline `LearnedCourseGenerator`: model/data manifest + conditioning + seed -> immutable `CourseSpec` + inference trace; no direct mutable `Track` write. | XL | manuscript lead (`Mi2021HDMapGenHierarchical`, `Rowe2025ScenarioDreamer`); candidate-only lead (`Sun2024DriveSceneGenGenerating`) + implementation inference |
| C1 | Request compilation/constraint extraction | `Steininger2025AutomaticallyGenerating`, *Automatically Generating Content for Testing Autonomous Vehicles from User Descriptions* | No text interface or semantic compiler | Extract constrained user text into a typed, auditable course request and reject unsupported semantics; do not synthesize geometry. | `TextCourseRequest` -> parser/compiler -> `CourseRequest` with source prompt, model version, parsed constraints, and validation results; compose with G6 or another synthesizer. | XL | candidate-only lead + implementation inference |
| G7 | Offline/search generator | `Jiang2021ReplayGuided`, *Replay-Guided Adversarial Environment Design*; `Dennis2020EmergentComplexity`, *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design*; `Wang2019POETOpen`, *POET: Open-Ended Coevolution of Environments and their Optimized Solutions* | Seeded static generation; no learner feedback boundary | Mutate spatial genotypes using declared learner/task feedback while separating generation from replay. | `UEDMutationPolicy`: feedback snapshot + parent genotype + seed -> child candidates; frozen evaluation adapter required. | XL | manuscript lead (`Jiang2021ReplayGuided`, `Wang2019POETOpen`); candidate-only lead (`Dennis2020EmergentComplexity`) + implementation inference |
| R1 | Course representation/schema | Current scope includes waypoints, buoys, and ordered passage relations | Closed `Track`; planar `GateSequence` | Represent open, ordered waypoint/buoy courses with direction, tolerance, pass-side, and terminal semantics. | `CourseSpec.open_route`: ordered markers or successor graph, explicit completion rule, frame/units, and optional corridor geometry. | L | manuscript lead + implementation inference |
| R2 | Course representation/schema | Current scope requires 3D gate pose and crossing semantics | Planar `GateSequence` | Represent a distinct 3D pose/aperture/traversal object with position, orientation, aperture, direction, precedence, and explicit open or cyclic order. | `CourseSpec.gates3d`: ordered `GateTraversal3D` objects with open/cyclic traversal; no 2D `GateSequence` or pipeline reuse. | XL | manuscript lead + implementation inference |
| R3 | Course representation/schema | Current representation discussion identifies width, elevation, and surface loss | Constant-width planar `Track` | Preserve variable width, elevation, banking, and surface/friction semantics through generation and conversion. | `CourseSpec.corridor_profile` keyed by arc length: left/right width, elevation, bank, surface; adapters declare unsupported fields. | XL | manuscript lead + implementation inference |
| R4 | Course representation/schema | Current scope distinguishes task-defining obstacles/world geometry | Rendering-only props; collision helpers | Generate semantic obstacles and world structure with collision/task roles rather than decorative poses. | `CourseSpec.world`: typed obstacles, keep-out volumes, semantic roles, geometry, and rule links; realization adapter creates simulator objects. | XL | manuscript lead + implementation inference |
| A1 | Adaptive selection/management | Search and benchmark protocol emphasize descriptor coverage; no archive implementation | Generator-local selection/fallback | Retain a descriptor-indexed archive without treating retention as synthesis. | `CourseArchive`: content hash, descriptor vector, predicates, lineage, and selection rationale; supports Pareto/coverage views, not a scalar winner. | L | manuscript lead + implementation inference |
| A2 | Adaptive selection/management | `Jiang2021PrioritizedLevel`, *Prioritized Level Replay*; `ParkerHolder2022EvolvingCurricula`, *Evolving Curricula with Regret-Based Environment Design* | No replay/curriculum manager | Schedule existing courses from replay/curriculum signals while preserving the generator-versus-management boundary. | `CurriculumManager`: frozen course IDs + feedback trace -> exposure schedule; it cannot modify geometry unless composed with G7. | L | candidate-only lead + implementation inference |
| V1 | Validation/repair | Current manuscript requires paired raw/pre/post lineage | Private Phase 1/2 buffers; final geometric validity | Expose fixed-shape device telemetry arrays for stage, fallback, selected candidate, bounded residuals, and counters, then persist lineage outside capture. | Eagerly materialize immutable content-addressed `GenerationRecord` values outside capture from P0 telemetry, with stage artifacts or references, predicate outcomes, timings, and repair deltas. | L | manuscript lead + implementation inference |
| V2 | Validation/repair | Current manuscript distinguishes geometric from dynamics/control/visibility/rule feasibility | Geometric validity and utilities only | Validate declared vehicle/control, gate visibility, ordered passage, and domain rules without claiming a universal feasibility certificate. | Predicate registry: `CourseSpec` + declared envelope -> structured pass/fail/indeterminate diagnostics and versioned solver/controller inputs. | XL | manuscript lead + implementation inference |
| I1 | Simulator/export | Current representation and benchmark sections require partial typed conversion | Runtime buffers only; no serialization | Define versioning, units/frames, draft lifecycle, content hash, and runtime conversion before serialization/export records field-level loss or approximation. | Minimal immutable `CourseSpec` v0 plus versioned serialization and import/export adapters returning `ConversionReport` with hashes, unit/frame transforms, runtime conversion, and unsupported semantics. | L | manuscript lead + implementation inference |
| I2 | Simulator/export | Current audit identifies missing Gymnasium/mainstream simulator adapter | `Course` is explicitly not a Gymnasium environment | Instantiate a generated course in a simulator with declared observation/action/reward/termination ownership. | Simulator adapter consumes immutable `CourseSpec`; a Gymnasium wrapper is separate and exposes environment semantics, not generator logic. | XL | manuscript lead + implementation inference |
| B1 | Boundary / downstream calibration | `Theodosis2011GeneratingRacing`; `Heilmeier2020MinimumCurvature`; `Cardamone2010SearchingOptimal`; `Jain2020ComputingRacing` | Localization and speed-profile proxies | Calibrate or evaluate final generated geometry with a racing-line optimizer; do not present optimization as course synthesis. | `RacingLineCalibration` consumes final `CourseSpec.corridor`, writes an optional derived artifact and diagnostics, and reports raw-to-final correspondence whenever raw diagnostics are compared; never a generator result. | M | boundary inspiration + implementation inference |

## Recommended Implementation Order

### P0: Instrumentation and Lineage Prerequisite

Implement V1 first. P0 defines fixed-shape device telemetry arrays for stage, fallback, selected candidate, bounded residuals, and counters, then eagerly materializes immutable content-addressed `GenerationRecord` values outside capture. The current `GeneratorSpec` centerline-plus-valid contract and mutable `Track` output cannot satisfy this yet. Capture raw output, generator-local fallback/selection, Phase 2 stages, predicate outcomes, repair displacement, timings, seed/config/version, and content hashes so yield, diversity, and repair burden are attributable to a generator.

### P1: Two Closed-Track Runtime Generators

Implement G1 and G2 against the existing `Track` output and XPBD Phase 2, with P0 telemetry; neither returns `GenerationTrace` directly. They are bounded representations that fit the current closed 2D envelope, exercise complementary controls, and do not require a public course schema first. Depend on P0; do not fold the separate gate path into this work.

### P2: Explicit Search Before Grammar

Implement G4 as bounded offline search over G1's `ChainCodeGenotype` first, then implement G3 and add its segment genotype to search. Depend on P0 and P1 benchmark descriptors. G3 compiles closed programs to the current raw centerline; open or branching programs wait for `CourseSpec`. Waypoint mutation is available only after R1.

### P3: Broaden Course Semantics

Establish minimal immutable `CourseSpec` v0 and I1 first: versioning, units/frames, draft lifecycle, content hash, and runtime conversion. Then implement R1, R2, R3, and G5 in that order. R2 remains a distinct 3D pose/aperture/traversal representation with its own open/cyclic order, not a reuse of the 2D gate path.

### P4: Learned/Text Methods and UED

Implement G6, C1, and G7 only after the P3 schema/export foundation and frozen benchmarks exist. These mechanisms have high contract and evaluation cost: they need training or prompt provenance where relevant, repair diagnostics, and held-out evaluation. R4, A1, A2, V2, I2, and B1 are coupled work in this phase; B1 remains downstream calibration, not a generator.

Scientific priority and engineering effort are distinct. P1 is scientifically useful because it tests two interpretable representations under the existing envelope, while its M effort is modest. G5, G6, C1, and G7 may be scientifically important but are XL because they widen the public object model, provenance, validation, and evaluation obligations.

## P1/P2 Implementation Sketches

### G1: Chain-Code / Direction Sequence

- **Representation:** serializable `ChainCodeGenotype` with fixed-length discrete directions, optional run lengths, scale, and deterministic smoothing/resampling; its decoder has fixed bounds.
- **Seeded generation path:** derive a `ChainCodeGenotype` and bounded local edits from the existing per-environment seed; the bounded decoder integrates vertices, normalizes scale and winding, and emits a raw centerline.
- **Closure and failure:** reserve a bounded tail for displacement correction; reject non-simple/over-residual candidates and select from a fixed number of seeded alternatives. P0 telemetry records tail correction, rejection reason, selected alternative, residuals, and counters; never use an unbounded retry loop.
- **Phase 2 reuse:** pass the chosen centerline into the existing constant-spacing, XPBD, inflation, and final-validity stages; emit the current `Track` output plus P0 telemetry, not a direct `GenerationTrace`.
- **Capture expectation:** CUDA-capturable only if sequence length, alternative count, and closure iterations are static; otherwise eager CPU/CUDA with the same fixed-shape telemetry contract.
- **Primary risks:** lattice artifacts, self-intersection, anisotropy, and an apparent yield improvement caused by fallback rather than the sequence mechanism.

### G2: Curvature-Profile / Clothoid Closed Loop

- **Representation:** fixed low-order periodic curvature basis or a bounded sequence of straight, clothoid-in, constant-curvature, and clothoid-out elements.
- **Seeded generation path:** sample coefficients/segment parameters from a seed, enforce net-heading closure, integrate at fixed samples, then apply a bounded low-dimensional displacement-closure projection.
- **Closure and failure:** report heading and displacement residuals before/after projection in bounded P0 telemetry; reject samples outside fixed residual/predicate bounds and select only from a fixed candidate set.
- **Phase 2 reuse:** emit the integrated centerline to the same Phase 2 pipeline and current `Track` output plus P0 telemetry; profile parameters are raw diagnostics only. B1 consumes final geometry and reports raw-to-final correspondence whenever it compares them.
- **Capture expectation:** capture is plausible for fixed-basis evaluation and fixed iteration count; a host nonlinear solver is an eager prototype, not a runtime contract.
- **Primary risks:** fragile displacement closure, overly smooth or star-shaped outputs, hidden curvature spikes after correction, and incorrectly treating racing-line sources as generator evidence.

### G4: Bounded Search / Evolutionary Synthesis

- **Representation:** begin with G1's explicit, serializable `ChainCodeGenotype`; add G3's `SegmentProgram` genotype only after G3 exists. Objectives remain a vector plus named constraints; waypoint mutation is available only after R1.
- **Seeded generation path:** initialize a fixed population of `ChainCodeGenotype` values from a seed schedule, mutate/crossover for a declared fixed budget, evaluate generation and validity predicates, and retain a documented candidate set. Add segment mutation only after G3.
- **Closure and failure:** each genotype owns a bounded decoder; invalid candidates receive P0-backed structured failure records rather than silent repair. The returned output identifies whether it was raw, locally repaired, or selected.
- **Phase 2 reuse:** decode selected candidates through the common Phase 2 adapter and current `Track` output plus P0 telemetry so comparison preserves identical downstream processing.
- **Capture expectation:** offline/eager by default; CUDA can accelerate evaluation but does not change the budget, seed, lineage, or selection contract.
- **Primary risks:** objective gaming, cost explosion, opaque best-of selection, policy/simulator dependence, and collapsing multi-objective evidence into one score.

### G3: Segment / Road-Block Grammar

- **Representation:** typed `SegmentProgram` genotype with socket frames, geometry parameters, semantic tags, and a closing connector policy; it becomes a G4 search genotype only after this compiler exists.
- **Seeded generation path:** sample a bounded program from seed and grammar constraints; compose transforms in order; compile closed single-loop programs to the current raw centerline; perform bounded connector or backtracking attempts.
- **Closure and failure:** require socket position/heading/tangent residual bounds at closure; reject failed programs with exact conflicting socket/constraint diagnostics instead of deforming an arbitrary final segment.
- **Phase 2 reuse:** closed single-loop compiler outputs use existing Phase 2 and P0 telemetry; open or branching programs wait for `CourseSpec`.
- **Capture expectation:** a fixed maximum program length and fixed connector attempts can be capture-compatible; dynamic graph construction remains an offline/eager route.
- **Primary risks:** closure bias, grammar-limited diversity, confusing road-network composition with a closed racing track, and losing segment semantics after flattening to a polyline.

## Acceptance Gates

Apply the following gate groups to the capability being released; they are not a universal generator checklist.

### Common

- **Registered contract:** name, version, supported object type, static configuration, declared capacity, output schema, and failure modes are registered. Track and gate registries remain separate.
- **Determinism scope:** document the supported device/backend scope and seed schedule. Do not claim cross-device bit identity unless tested and guaranteed.
- **Bounded failure and lineage:** every applicable generation, search, repair, and closure path has a fixed bound and structured failure reason; retain raw, fallback/selection, Phase 2, validation, and final identifiers/hashes.
- **Report a vector, not a winner:** report yield, fallback rate, repair displacement, descriptor diversity, runtime/cost, and declared predicate outcomes without a scalar generator ranking.

### Runtime Generators

- **Pilot before scale:** run at least 10,000 attempted courses per runtime generator before a release-scale study, with the full attempted population retained or reconstructible.
- **CPU/CUDA/capture:** test CPU and CUDA paths, capture/eager behavior, fixed-batch constraints, and the documented determinism scope.

### Offline Search/Learned

- **Budget and candidate record:** retain the fixed budget, seed schedule, genotype/model version, objectives, predicates, selected candidate, and structured failures.
- **Frozen evaluation boundary:** learned or search/UED methods evaluate on a separately frozen or reproducibly sampled suite; no replay-only scheduling is presented as synthesis.

### Representation/Conversion

- **Course lifecycle:** versioned `CourseSpec` records units, frames, draft lifecycle, content hash, and runtime conversion.
- **Adapter conformance:** exports and simulator adapters report field preservation, loss, approximation, import outcome, units, frames, and external version.

### Archive/Curriculum

- **Archive identity:** retained artifacts include content hash, descriptor vector, predicates, lineage, and selection rationale.
- **Management boundary:** curriculum exposure uses frozen course IDs and feedback traces; geometry mutation is attributed to its composed generator.

## What Not to Implement as a Generator

The following can be useful adjacent components, but must not be counted or registered as course generators by themselves:

- Collision checking, signed-distance fields, and contact queries.
- Progress, laps, wrong-way logic, and checkpoint crossing signals.
- Rendering props or decorative world placement without task-defining semantics.
- Localization and speed-profile proxies, including racing-line-like diagnostics.
- `Course` facade behavior, reset logic, and batch-buffer ownership.
- Racing-line optimization: retain B1 as downstream calibration/evaluation on a completed course.
- Replay-only curricula: retain A2 as artifact scheduling unless coupled to an explicit spatial mutation mechanism such as G7.
- Serialization, exporters, importers, and simulator realization adapters.

## Concrete Next Action

Design and implement P0 first: fixed-shape device telemetry arrays and eager immutable content-addressed `GenerationRecord` materialization outside capture. Then write separate G1 and G2 implementation specifications against the current `Track` output, XPBD pipeline, and P0 telemetry before beginning G4 search or G3 segment work.
