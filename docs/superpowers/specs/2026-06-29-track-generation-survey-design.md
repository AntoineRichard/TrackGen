# Track and Gate Generation Survey Design

**Date:** 2026-06-29
**Status:** Design approved in conversation, spec for review
**Target level:** IJRR / Science Robotics survey quality
**Primary thesis:** Robot racing and agile-control research lacks a shared, open,
vehicle-agnostic theory and benchmark practice for generating tracks, gate courses, and
race corridors. This weakens reproducibility, generalization claims, and algorithm
comparisons across cars, drones, boats, and related robots.

## Scope Guardrail

The survey scope is literature-driven, not repository-driven. The current TrackGen code,
`docs/related-work/state-of-the-art.rst`, `docs/related-work/prior-art.rst`, and
`docs/generators/benchmarks.rst` are seed materials and reference implementation
evidence. They must not define the boundary of the paper.

The paper should search beyond the current SOTA notes into at least:

- robot racing and autonomous-racing track generation;
- drone racing gate and waypoint course generation;
- autonomous-driving road/scenario generation when the geometry is relevant;
- maritime, boat, USV, and buoy-course task generation;
- procedural content generation, search-based PCG, and environment design;
- simulator map and asset formats used by RL/control researchers;
- benchmark suites, competitions, and fixed-course datasets;
- open-source generators and practical simulator integrations.

TrackGen should appear as a set of reference implementations and benchmark tools, not as
the paper's conceptual boundary or sole contribution.

## Paper Positioning

The article should be framed as a field-defining survey. Its contribution is not simply
"a survey of TrackGen-like methods." It should establish what a robot-racing course
generator is, what should be reported, how methods should be compared, and how generated
courses should be packaged for training and evaluation.

Proposed contribution claims:

1. A cross-domain taxonomy of course representations for cars, drones, boats, and
   adjacent racing/agile-control robots.
2. A taxonomy of generation methods, including constructive grammars, random procedural
   generators, search/evolutionary methods, learned generators, curriculum/environment
   design, human-designed benchmark sets, and geometric repair/projection methods.
3. A reporting and evaluation protocol covering feasibility, difficulty, diversity,
   dynamics relevance, simulator portability, reproducibility, and generation throughput.
4. OSS reference implementations for representative generator families.
5. Large training distributions, such as 10,000-100,000 courses per generator.
6. Curated easy/medium/hard evaluation suites selected by explicit metrics and diversity
   criteria.

## Survey Exemplar Lessons

The structure should follow strong robotics surveys rather than a chronological literature
review. Initial exemplars include IJRR surveys such as "Reinforcement learning in
robotics: A survey" and "Human motion trajectory prediction: a survey", plus Science
Robotics reviews such as "Social robots for education: A review" and "A review of
collective robotic construction".

Reusable structure from these exemplars:

- define the problem boundary early;
- state why existing surveys do not cover the gap;
- present a durable taxonomy;
- compare papers in tables instead of prose only;
- separate representations, methods, metrics, and benchmarks;
- expose reproducibility and evaluation gaps;
- close with open problems and a research agenda.

## Proposed Paper Structure

### 1. Introduction

Argue that track and gate generation is usually treated as incidental infrastructure in
RL/control papers, even though it controls distribution shift, curriculum, safety margins,
sim-to-real relevance, and benchmark comparability. The introduction should make the
central claim that course generation deserves the same level of standardization that
robotics has given to datasets, simulators, and benchmark tasks.

### 2. Scope, Definitions, and Boundaries

Define the objects of study:

- closed tracks and road corridors;
- open courses, waypoint sequences, and gate chains;
- waterway/buoy courses;
- centerlines, boundaries, obstacles, gates, checkpoints, and racing lines;
- task distributions versus fixed benchmark courses.

Separate course generation from downstream racing-line optimization, trajectory planning,
control, perception, and policy learning. Those downstream areas remain relevant when
they define requirements or metrics for generated courses.

### 3. Why Previous Surveys Are Not Enough

Include a comparison table against adjacent surveys: RL in robotics, procedural content
generation, autonomous driving scenario generation, drone racing/control, autonomous
racing, and robot benchmarks. The table should identify whether each covers:

- geometric course representation;
- generator algorithms;
- feasibility constraints;
- vehicle-specific difficulty;
- open-source generators;
- benchmark course distributions;
- simulator export and reproducibility.

### 4. Representation Taxonomy

Organize representations before algorithms:

- tile, segment, and road-block sequences;
- spline, Bezier, clothoid, and control-point curves;
- sampled closed curves and centerline-plus-width tracks;
- gate and waypoint pose sequences;
- waypoint graphs and route networks;
- occupancy, heightfield, mesh, and asset-level worlds;
- OpenDRIVE, simulator-native maps, and game-engine assets;
- buoy fields, water corridors, and maritime race marks.

Each representation should be evaluated for expressivity, validity by construction,
vehicle compatibility, simulator portability, and suitability for batched RL training.

### 5. Generator Taxonomy

Classify methods by how they create and validate courses:

- constructive grammars and component libraries;
- random procedural generators;
- search, evolutionary, and metaheuristic generators;
- learned generative models;
- adversarial and curriculum environment design;
- human-designed benchmark sets and competition courses;
- repair, projection, relaxation, and constraint-solving methods.

The taxonomy should make clear when "generation" means geometry synthesis versus task
selection, mutation, or replay.

### 6. Domain Constraints

Use separate subsections for vehicles:

- Cars: closed-loop consistency, road width, curvature, friction, racing line, overtaking
  space, map formats, lane/boundary definitions, and dynamics-aware speed profiles.
- Drones: 3D gate pose, visibility, gate spacing, obstacle clearance, vertical motion,
  yaw/roll difficulty, field-of-view limits, and sim-to-real asset placement.
- Boats/USVs: buoy gates, water corridors, turn radius under hydrodynamics, currents,
  wind, shoreline/obstacle clearance, waypoint ambiguity, and maritime simulator support.
- Other agile robots: legged racing, hovercraft, or mixed-terrain robots when course
  generation affects policy generalization.

### 7. Metrics and Reporting Protocol

Define metrics in layers:

- Feasibility: finite geometry, no invalid intersections, minimum width/clearance,
  gate/buoy separation, simulator loadability, and spawn/reset validity.
- Geometry: length, area, compactness, curvature, torsion for 3D, elevation change,
  straight fraction, chicane count, gate density, obstacle density.
- Difficulty: dynamics-aware lap-time proxy, minimum-turn-radius violations, speed
  profile, control effort proxy, visibility/line-of-sight, recovery margins.
- Diversity: coverage in metric space, novelty, clustering, entropy over representation
  choices, train/test distribution distance.
- Reproducibility: seed determinism, versioned configs, documented rejection rate,
  generation throughput, and stable serialized outputs.
- Sim feasibility: export support, validation after import, consistent coordinate
  frames, asset units, collision geometry, reset/spawn semantics, and framework adapters.

The paper should recommend a minimum reporting checklist for future papers that use
generated racing courses.

### 8. Benchmark Protocol

Propose two benchmark products:

1. Training distributions: 10,000-100,000 generated courses per generator family, with
   fixed seeds, configs, metrics, and train/validation splits.
2. Evaluation suites: curated easy/medium/hard sets selected by metric quantiles and
   diversity clustering. The default paper target is 100 courses per difficulty tier
   for each main domain. A smaller 100-total smoke suite can be published as an
   additional convenience set for expensive simulator evaluations.

The selection process should avoid cherry-picking: first filter for feasibility, then
stratify by domain-specific difficulty metrics, then choose diverse representatives
within each tier.

### 9. OSS Reference Implementations

Position OSS code as reference implementations for the taxonomy:

- checkpoint/radial generators;
- Bezier/control-point generators;
- convex-hull and polar generators;
- Voronoi or graph-derived generators;
- segment grammar or road-block generators;
- gate primitive chains;
- water/buoy course generators.

The implementation should expose stable serialization and simulator adapters. Simulator
feasibility is important enough to be a first-class benchmark dimension, even if full
adapter implementation belongs to follow-on engineering work rather than the initial
paper structure.

Target export surfaces to investigate:

- Gymnasium / CarRacing-style arrays and reset hooks;
- F1TENTH / RoboRacer maps;
- CARLA and OpenDRIVE-style road descriptions;
- Isaac Lab / Isaac Sim assets or task configs;
- AirSim-style gate pose lists;
- Gazebo / VRX-style maritime worlds;
- generic JSON/NPZ course bundles for RL pipelines.

### 10. Lessons from Existing RL/Control Papers

Analyze how current papers report course generation. The survey should look for missing
details: seed policy, rejection rate, train/test split, geometry metrics, simulator
format, fixed evaluation tracks, difficulty controls, and whether the generator is
released.

This section should connect the survey to practical scientific claims: without a clear
course generator, generalization and curriculum results are hard to reproduce.

### 11. Open Problems and Research Agenda

End with forward-looking questions:

- dynamics-aware course generation;
- sim-to-real course distributions;
- benchmark transfer across simulators;
- co-design between generator and curriculum;
- safety and recoverability metrics;
- multi-agent and overtaking-aware tracks;
- water-domain benchmarks;
- standardized course cards or datasheets;
- reproducible generator-policy evaluation.

## Literature Search Plan

The next research pass should deliberately go beyond the current related-work files.
Suggested query families:

- "procedural track generation reinforcement learning car racing";
- "autonomous racing benchmark generated tracks";
- "drone racing random gate course generation reinforcement learning";
- "AirSim drone racing gate randomization";
- "F1TENTH generated tracks benchmark";
- "OpenDRIVE procedural road generation autonomous driving simulation";
- "search based procedural content generation autonomous driving roads";
- "boat racing reinforcement learning buoy course";
- "autonomous surface vehicle waypoint benchmark VRX RobotX";
- "USV simulator reinforcement learning course generation";
- "simulator export generated tracks OpenDRIVE CARLA Isaac Gazebo";

For each paper or system, extract:

- domain and vehicle;
- course representation;
- generation method;
- validity checks and rejection/repair behavior;
- metrics reported;
- benchmark or training distribution size;
- simulator and export format;
- whether code/assets are public;
- relevance to the proposed taxonomy.

## LaTeX Artifact Plan

The paper should be written as a LaTeX project once the outline and search matrix are
accepted. The repository should eventually contain:

- `paper/main.tex`;
- `paper/sections/*.tex`;
- `paper/figures/`;
- `paper/tables/`;
- `paper/references.bib`;
- `paper/Makefile` or equivalent build command;
- a checked TeX engine setup, likely `latexmk` with `pdflatex` or `lualatex`.

Installation of LaTeX tooling should happen during implementation planning, not in this
design spec. The build should be reproducible from a clean environment.

## Initial Figures and Tables

High-value survey artifacts:

- taxonomy diagram crossing representation, generator method, vehicle domain, and output
  format;
- comparison table of adjacent surveys and the gap this paper fills;
- method table mapping papers to representation, generator, validity, metrics, and code;
- metric hierarchy table;
- benchmark selection pipeline figure;
- simulator export compatibility matrix;
- easy/medium/hard metric distribution plots from reference generators.

## Success Criteria

The design is successful if it leads to a survey that:

- reads as a reference article for course generation in robot racing;
- is not limited by TrackGen's current code or docs;
- uses TrackGen-style OSS as reproducible reference implementations;
- proposes concrete metrics and benchmark suites;
- treats simulation feasibility as a core evaluation dimension;
- gives future RL/control papers a checklist for reporting generated courses.

