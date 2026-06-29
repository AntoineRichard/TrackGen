# Blind Ground Literature Discovery

Date: 2026-06-29  
Agent: `blind-ground`

## Blindness and Evidence Protocol

The dispatch supplied the thesis/domain brief and only the destination paths for the two output files under a repository whose name, `TrackGen`, was therefore visible. It did not supply known paper names, corpus content, taxonomy content, source or documentation paths, other agent outputs, or TrackGen implementation details. This worker did not read any such material and did not query for TrackGen.

The visible repository name and destination hierarchy are a methodological limitation: the worker was content-blind, but not path-blind. Repository access was limited to reading, revising, and validating `blind-ground.csv` and `blind-ground.md`. This review response was post-search evidence cleanup only; it did not reopen discovery or add seeds/candidates.

Search indexes and general web search were used for discovery only. The cleanup rechecked claims against primary papers, publisher/proceedings records, institutional repositories, official project documentation, official package registries, author-owned repositories, and official artifacts. Unsupported facts are `NR`. Every CSV record remains `metadata_status=unverified`.

## Outcome and Coverage

- Initial notebook: 47 provisional source manifestations.
- DOI-first, normalized-title-second deduplication: 43 unique publication/system leads.
- Boundary screening before formal refinement: 40 unique in-scope candidates.
- Formal refinements: 4 additions in round 1, 1 in round 2, and 0 in round 3.
- Final CSV: 45 unique candidates, 45 unique candidate IDs, 45 unique normalized titles, and no repeated non-`NR` DOI.
- Domain coverage: 14 autonomous-racing/racing-game-transfer candidates; 22 autonomous-vehicle testing or learned-map candidates; 9 legged/open-ended terrain candidates.
- Publication years: 2006-2026.
- Metadata coverage after cleanup: 34 DOI values and 11 `NR`; code status is 21 `official_open`, 1 `not_found`, and 23 `NR`; asset status is 8 `official_open`, 2 `not_found`, and 35 `NR`.
- All 45 records have a nonempty `evidence_locator` pointing to a complete primary paper/record, an official report, and/or an official artifact path. This does not mean every full text was available or verified line by line.
- All records retain `screening_status=candidate`, `metadata_status=unverified`, and `discovery_agent=blind-ground`.

Representations covered include polar and Cartesian control points, Bezier/Catmull-Rom splines, segment sequences, Frenet encodings, cone boundaries, road blocks, lane/topology graphs, OpenDRIVE interchanges, raster-to-vector learned maps, heightfields, triangle meshes, CPPNs, executable terrain programs, and parametric 3D tunnel blocks. Generator families include constructive PCG, evolutionary and illumination search, combinatorial testing, GAN/diffusion/autoregressive models, open-ended paired evolution, curriculum generation, and LLM/VLM program or function generation.

Status semantics used in the CSV are deliberately narrow. `official_open` requires a currently accessible official package, project, author repository, or artifact. `not_found` records a paper-linked or promised artifact that was unavailable at audit time. `NR` means the audit did not establish a supported status. No row uses `unofficial_open`, `closed`, or `not_applicable`. A paper page, project video, or figure alone is not treated as a code or asset release.

## Deduplication

The DOI key was lowercased and normalized after removing DOI URL prefixes. Records without a DOI were compared by a title key formed by lowercasing and removing punctuation and whitespace. DOI identity took precedence over title variation.

Four duplicate source manifestations were collapsed:

1. OpenAI Gym paper, official repository, and maintained CarRacing documentation became one candidate.
2. F1TENTH paper and official simulator documentation became one candidate.
3. MetaDrive paper, preprint, and official repository became one candidate.
4. POET preprint and official implementation/project manifestations became one candidate.

Three provisional boundary-only leads were removed before refinement:

1. Learn-to-Race and similar systems that distribute precise fixed real-world track maps but do not synthesize or mutate course geometry.
2. SDC-Scissor/ITS4SDC-style selection, prioritization, and replay over existing tests without geometry synthesis.
3. Racing-line, minimum-curvature trajectory, and control optimization on a fixed course.

## Search Surfaces

- General scholarly/web search for discovery and citation chasing.
- Official publisher/proceedings surfaces: ACM Digital Library, IEEE Xplore/DOI records, Springer, Wiley, Elsevier/ScienceDirect, SAE Mobilus, PMLR, CVF Open Access, ECVA, SBGames, IJEEI, and institutional publication repositories.
- Primary preprints: arXiv.
- Official system/project surfaces: BeamNG.tech research, F1TENTH Gym documentation, MetaDrive/PGDrive projects, official author project pages, GitHub repositories, Zenodo artifacts, and PyPI.
- Crossref/DBLP-like indexes were used only to locate or triangulate records; they were not used as CSV evidence locators.

## Discovery Queries

### Autonomous Racing and Simulators

```text
autonomous racing procedural track generation paper robot
F1TENTH procedural track generator paper
autonomous race track generation reinforcement learning random tracks paper
autonomous racing simulator procedural tracks RaceCarGym paper
OpenAI Gym CarRacing random track generation official repository paper
procedural road maps driving reinforcement learning simulator random tracks benchmark
site:f1tenth-gym.readthedocs.io random track generator OpenAI CarRacing
site:github.com/f1tenth f1tenth_gym random track generator
F1TENTH gym paper simulation environment official
"random track generator" F1TENTH
Formula Student Driverless random track generator official repository FSSIM paper
FSSIM open source autonomous racing simulator track generator paper
FSDS simulator procedural track generation Formula Student official
Formula Student cone track procedural generation research paper
"procedural pipeline" "unique race" tracks autonomous racing
autonomous racing training procedurally generated race tracks sim-to-real paper
robot race procedural track generator unseen tracks paper
"procedural race track generation for domain randomization" Behrens PDF
"Technical Reports in Computing Science" "CS-01-2020"
```

### Racing-Game Geometry Transfer

```text
site:dl.acm.org racing track procedural generation evolutionary TORCS
site:ieeexplore.ieee.org racing track generation evolutionary computation game
"procedural generation" racetracks racing games paper DOI
"Making Racing Fun Through Player Modeling and Track Evolution"
race track generation entropy curvature speed profiles evolutionary paper
"Interactive evolution for the procedural generation of tracks" ACM
"Making Racing Fun Through Player Modeling and Track Evolution" Springer
"Search-based procedural content generation for race tracks" official journal
"Automatic Track Generation for High-End Racing Games Using Evolutionary Computation" DOI
"Racing Tracks Improvisation" IEEE official
site:sbgames.org/proceedings2021 "Procedural Generation of Isometric Racetracks Using Chain Code"
"From Generation to Gameplay: Authoring Race Tracks With Repulsive Curves"
```

### Autonomous-Vehicle Road and Map Generation

```text
autonomous vehicle testing road generation procedural paper AsFault
site:dl.acm.org autonomous driving road generation testing DeepHyperion DeepJanus
site:ieeexplore.ieee.org procedural road generation autonomous vehicle testing OpenDRIVE
JunctionArt procedural generation high-definition road networks official paper
RoadGen generating road scenarios autonomous vehicle testing official repository
"DeepHyperion" autonomous driving road official paper
"DeepJanus" autonomous driving roads paper official
"Automatically testing self-driving cars" AsFault official ACM
search based road generation autonomous driving BeamNG test generation paper
"A search-based framework for automatic generation of testing environments" DOI
"Using genetic algorithms for automating automated lane-keeping system testing" DOI
"Frenetic-lib" road structures ADS testing paper
CRAG combinatorial testing based generator road geometries ADS testing paper
"Wasserstein generative adversarial networks for online test generation"
"Diversity-guided Search Exploration for Self-driving Cars Test Generation through Frenet Space Encoding"
"Procedural Generation of High-Definition Road Networks for Autonomous Vehicle Testing and Traffic Simulations"
"DiffRoad: Realistic and Diverse Road Scenario Generation for Autonomous Vehicle Testing"
"FLYOVER: A Model-Driven Method to Generate Diverse Highway Interchanges for Autonomous Vehicle Testing"
"HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps" official
"DriveSceneGen: Generating Diverse and Realistic Driving Scenarios from Scratch" official paper
SLEDGE generative simulator road map generation autonomous driving official paper
CVPR 2025 driving simulation environment generation vectorized map geometry Scenario Dreamer
generative driving simulation environments static map geometry diffusion official paper
HD map generation autonomous driving simulation novel road topology paper
"Automatically Generating Content for Testing Autonomous Vehicles from User Descriptions"
```

### Legged, Parkour, and Open-Ended Terrain

```text
procedural obstacle course generation quadruped robot reinforcement learning paper
legged robot parkour procedurally generated obstacle courses curriculum paper
terrain generation curriculum agile quadruped locomotion official paper
open ended environment generation BipedalWalker obstacle courses POET PAIRED
"Learning to Walk in Minutes" terrain curriculum procedural terrain official
"Learning robust perceptive locomotion" procedural terrains official paper
"Eurekaverse" terrain generation quadruped official paper
"Robot Parkour Learning" terrain generation obstacle curriculum
site:proceedings.mlr.press POET open ended environments BipedalWalker
site:proceedings.neurips.cc PAIRED environment generation BipedalWalker
site:proceedings.mlr.press ACCEL environment design BipedalWalker
"Adversarially Compounding Complexity by Editing Levels" paper
"Paired Open-Ended Trailblazer" GECCO official DOI
site:arxiv.org/abs/1901.01753 POET
```

## Formal Refinement and Saturation

Yield is `new evidence-backed in-scope candidates / post-round cumulative unique in-scope candidates`. For comparison, entry-relative yield is also shown. A lead without retrievable primary/official evidence was retained under retrieval needs and did not enter the evidence-backed denominator.

### Round 1: Named Testing-Tool Refinement

```text
"RIGAA" road topology generation autonomous vehicle
"RoadSign" failure-revealing roads self-driving cars
"Spirale" lane keeping road generation SBFT 2023
"GenRL" road topology generation SBST 2022
```

Added RIGAA, RoadSign, Spirale, and GenRL. Cumulative set: 44. Post-round yield: `4/44 = 9.09%`; entry-relative yield: `4/40 = 10.00%`. This round was not saturated.

### Round 2: Ground-Course Representation Refinement

```text
"procedural tunnel geometry" quadruped robot course generation
"cone course generator" Formula Student Driverless simulation official
"Frenet road generator" autonomous driving testing 2024 2025
"LLM terrain curriculum" quadruped obstacle course generation
```

Added GenTe. Robot Squid Game was already in the set. Skill-Nav was excluded as waypoint-conditioned navigation/control on supplied terrain rather than course synthesis. An accepted IROS 2026 rover-terrain curriculum title lacked a retrievable paper and was moved to retrieval needs. Cumulative set: 45. Post-round yield: `1/45 = 2.22%`; entry-relative yield: `1/44 = 2.27%`.

### Round 3: Topology and Serialization Refinement

```text
"lane graph generation" autonomous driving simulator novel road topology
"racetrack generator" robot racing benchmark procedural geometry
"obstacle course distribution" legged robot reinforcement learning generator
"OpenDRIVE generator" autonomous vehicle testing procedural geometry
```

Added no evidence-backed candidate. HDMapGen and JunctionArt were already present. SeqGrowGraph and digital-twin results reconstruct fixed observed roads rather than synthesize new course distributions. A Kempten procedural-track paper was relevant but lacked a primary/official locator and was moved to retrieval needs. Cumulative set: 45. Yield: `0/45 = 0.00%`.

Rounds 2 and 3 are consecutive rounds below 5%, so the stopping rule is satisfied.

## Boundary Judgments

- Included racing-game PCG only when the generated object is transferable geometry: centerline, segment sequence, spline, tile path, intersection, or serialized map. Game appearance generation alone was excluded.
- Included failure-directed road generation for ADS testing because the generator synthesizes or mutates geometry, even when lane departure is the optimization target.
- Included learned HD-map/lane-graph models only when they sample novel topology or geometry from scratch. Perception, lane extraction, map reconstruction, and digital-twin conversion of a fixed observed road were excluded.
- Included official software without a paper when it directly generates and exports robot-racing courses, as with the Formula Student Random Track Generator.
- Included open-ended and curriculum systems when they mutate or synthesize terrain/course geometry. Prioritized replay, curriculum scheduling, reward generation, and policy transfer without geometry generation were excluded.
- Excluded appearance-only and dynamics-only domain randomization. When a paper contains geometry plus appearance/dynamics randomization, only geometry fields were coded.
- Excluded actor/traffic trajectory generation on fixed maps unless the same system also generates static lane/road geometry; only the geometry stage was coded for mixed systems.
- Excluded racing-line optimization, waypoint planning, trajectory optimization, and control on a fixed course. Skill-Nav was screened this way despite random waypoint experiments.
- Adaptive user-model papers that output track-alteration decisions but do not implement geometry generation were treated as task adaptation, not generators.
- PAIRED/ACCEL/PLR-style UED papers were not included merely for generating abstract levels; a ground-robot course or locomotion-terrain case was required.

## Terminology Encountered Beyond the Brief

- **Road spine / control line:** a polyline or curve from which lane boundaries and surfaces are derived.
- **Frenet encoding:** road geometry represented by incremental heading/curvature relative to path progress.
- **Illumination search / MAP-Elites:** generation that fills feature-map cells rather than optimizing one scalar objective.
- **Expressive range:** coverage and distribution of generated content over selected geometric descriptors.
- **Search-based software testing (SBST):** optimization of generated tests, often roads, toward system failures and diversity.
- **Minimal criterion:** a learnability filter that rejects open-ended environments that are too easy or too hard.
- **Paired environment-policy evolution:** an archive in which generated courses and policies coevolve and policies transfer across courses.
- **Terrain lattice:** parallel simulator tiles arranged by terrain family and curriculum difficulty.
- **Executable terrain program:** generated code that constructs heightfields or meshes, rather than a directly generated mesh.
- **Lane graph / lanelet / HD map:** lane-level topology and geometry used as simulator-ready static environment structure.
- **Feature-map coverage:** competition or illumination metric counting occupied behavior/geometry bins.

## Sparse and Contradictory Areas

- Few autonomous-racing papers make novel track generation the primary contribution; random generators are often incidental simulator utilities with weakly reported metrics.
- Formula Student cone-course generation is represented mainly by official software. Peer-reviewed descriptions, formal validity proofs, and distribution evaluations are sparse.
- Competition tool papers for RoadSign, Spirale, and GenRL are short; representation, validity, and hyperparameter details remain partly `NR`.
- Racing-game papers often report playability or human fun but not robot-relevant curvature, clearance, traversability, or export metrics; public code is uncommon for older work.
- Learned lane-map generators report realism/diversity but frequently omit simulator serialization and hard geometric validity rates.
- Legged-terrain work commonly entangles geometry curriculum with dynamics and sensor randomization. The CSV separates these where the primary paper permits it.
- Publication-year conventions conflict for online-first versus issue dates, notably MetaDrive and the automated lane-keeping GA paper; the coded year follows the primary record used and remains unverified.
- The Kempten technical-report trail has contradictory author/title manifestations for the 2020 domain-randomization report. No claim was promoted without an official report record.
- RoadSign's DOI is now resolved as `10.1109/SBFT59156.2023.00006`; its exact road representation, validity predicates, export, code, and released assets remain `NR`.
- RoadGPT's DOI is resolved as `10.1109/ICSE-NIER66352.2025.00021`; no paper-specific full text or author copy was located in the final audit, so all abstract/index-only technical fields are `NR` and the generic BeamNG index is not used as evidence.
- The GenTe paper reports 100 terrains (50 text and 50 figure), while the pinned official repository README reports 200 samples (100 text and 100 image). The CSV preserves the paper's evaluation count and records the repository discrepancy in `coding_notes`.
- Exact seeds, generated-suite sizes, export formats, artifact licenses, and historical availability are the most frequent remaining reproducibility gaps.

## High-Priority Retrieval Needs (Updated)

1. **Procedural Race Track Generation for Domain Randomization**, Kempten Technical Reports in Computing Science CS-01-2020. Retrieve the official report to resolve Fabian Behrens versus Ulrich Goehner metadata, algorithm parameters, and license.
2. **Construction of an Autonomous Driving Model Vehicle: Procedural Track Generation for Driving Simulations** (Callum Munro, Roman Wecker, Bunyamin Orumcek, 2021). Discovery material indicates graph/A* generation, noise-weighted variety, VDI/Carolo constraints, and OpenDRIVE export, but no primary/official locator was found.
3. **Automatic Terrain Curriculum Generation via Optimal Transport for Multi-DoF Rover Robot Locomotion** (accepted IROS 2026). An official lab publication list exists, but the paper and artifact were not yet retrievable.
4. **RoadSign**: DOI and IEEE document are resolved. Retrieve full text or an official artifact to resolve road encoding, hard validity predicates, export format, code, and released tests.
5. **Random Track Generator**: PyPI 1.1.0 establishes package availability. The owner-matched repository snapshot at commit `4ad014853493d5500aec9dd7121d6dc8f8c240c0` establishes current provenance, not an exact source mapping for that PyPI release. Resolve the release-to-commit mapping; PyPI omits `project_urls`.
6. **RoadGPT**: DOI and IEEE document are resolved, but no paper-specific full text or author copy was located. All technical coding is `NR`; retrieve full text or an official artifact before restoring representation, generator, validity, distribution, evaluation, simulator/export, reproducibility, code, or asset claims.
7. **GenTe**: official code and benchmark are resolved at commit `60ceed680802fb30db86e1be8f0e62a321e7d782`. Resolve the missing license and the paper/repository benchmark-count contradiction.
8. **Learning to Race Full-Scale Autonomous Racecars** technical report. Retrieve the official report/artifact section describing the procedural pipeline reported to produce more than 100 tracks.

## Post-Review Field Changes

The candidate set, IDs, discovery queries, and saturation rounds are unchanged. Field changes are:

- `BG-001`, `BG-004`, `BG-007`, `BG-010`, `BG-028`, `BG-029`, `BG-030`, and `BG-031`: removed mixed supported-plus-`NR` values from `representation_family` and/or `validity_strategy`; moved unresolved detail to `coding_notes`.
- `BG-011`: changed `representation_family`, `generator_family`, `generation_role`, `validity_strategy`, `geometry_metrics`, `difficulty_metrics`, `diversity_metrics`, `training_distribution`, `evaluation_suite`, `simulator`, `reproducibility_fields`, `metadata_evidence`, `evidence_locator`, `code_status`, `asset_status`, and `coding_notes`. It is now explicitly a control paper using an inherited generator; Appendix B reports four tracks.
- `BG-012`: changed `authors`, `metadata_evidence`, `evidence_locator`, `reproducibility_fields`, `code_status`, `asset_status`, and `coding_notes` to record PyPI 1.1.0, owner-matched current source provenance, pinned repository paths, dependencies, and MIT license without equating the audited commit to the PyPI release source.
- `BG-015`, `BG-016`, `BG-017`, `BG-020`, `BG-023`, `BG-025`, `BG-029`, `BG-030`, `BG-033`, `BG-034`, `BG-035`, `BG-038`, and `BG-041`: replaced narrative status claims with audited scalar statuses and added direct official repository/artifact locators; unsupported asset claims were set to `NR` or `not_found`.
- `BG-018`: changed both statuses to `not_found` and recorded that the paper-cited official replication URL returned 404.
- `BG-021`, `BG-022`, and `BG-024`: changed unsupported positive code/asset claims to `NR` and documented the missing official artifact locator.
- `BG-032`: resolved bibliographic metadata from the IEEE DOI/document record. The final evidence audit superseded abstract/index-based technical coding: all technical fields are now `NR`, and the generic BeamNG index is no longer a row-level evidence locator.
- `BG-042`: changed `doi`, `url`, `metadata_evidence`, `representation_family`, `generator_family`, `generation_role`, `validity_strategy`, all three metric fields, `training_distribution`, `evaluation_suite`, `simulator`, `export_format`, `reproducibility_fields`, `evidence_locator`, both statuses, and `coding_notes`; IEEE document `10190387` and the official competition report now support the retained claims.
- `BG-043`: changed `representation_family`, `validity_strategy`, all three metric fields, `training_distribution`, `export_format`, `evidence_locator`, both statuses, and `coding_notes`.
- `BG-044`: changed `representation_family`, `validity_strategy`, `training_distribution`, `export_format`, `evidence_locator`, both statuses, and `coding_notes`.
- `BG-045`: changed `metadata_evidence`, `simulator`, `reproducibility_fields`, `evidence_locator`, both statuses, and `coding_notes` after locating the paper-linked official repository; recorded missing license and benchmark-count contradiction.
- `code_status` changed to controlled scalars for `BG-008`-`BG-018`, `BG-020`-`BG-025`, `BG-029`, `BG-030`, `BG-033`-`BG-035`, `BG-037`, `BG-038`, and `BG-041`-`BG-045`.
- `asset_status` changed to controlled scalars for `BG-008`-`BG-018`, `BG-020`-`BG-025`, `BG-028`-`BG-045`.
- `evidence_locator` was replaced for all `BG-001`-`BG-045` with a complete primary paper/record and, where applicable, precise section/page or commit-pinned official artifact path.
- All semicolon-delimited cells were normalized to `; `.

## Final Evidence-Audit Corrections

The candidate set, IDs, discovery queries, row counts, and saturation calculations remain unchanged. This final pass changed only these CSV fields:

- `BG-012` `reproducibility_fields` and `coding_notes`: labeled commit `4ad014853493d5500aec9dd7121d6dc8f8c240c0` as an owner-matched current provenance snapshot and made the PyPI 1.1.0 release-to-commit mapping explicitly unresolved.
- `BG-030` `coding_notes`: retained `asset_status=not_found` and dated the checkpoint-TODO observation to the 2026-06-29 audit.
- `BG-032` `metadata_evidence`, `vehicle`, `course_object`, `representation_family`, `generator_family`, `generation_role`, `validity_strategy`, `training_distribution`, `evaluation_suite`, `simulator`, `reproducibility_fields`, `evidence_locator`, and `coding_notes`: removed the generic BeamNG index and downgraded every abstract/index-only technical claim to `NR` after no paper-specific full text or author copy was located. `geometry_metrics`, `difficulty_metrics`, `diversity_metrics`, `export_format`, `code_status`, and `asset_status` were already `NR` and remain unchanged.
- `BG-018` received no field change; its dated 2026-06-29 `not_found` evidence remains intact.

## Limitations After Cleanup

- The destination path and repository name were visible even though corpus contents and implementation details were not; this is path-blindness leakage, explicitly disclosed above.
- This was a targeted evidence audit, not a new discovery round. It did not test whether omitted literature exists and did not alter saturation.
- `official_open` records current accessibility and official provenance, not buildability, completeness, archival permanence, or license sufficiency.
- Closed/paywalled publisher records were not necessarily available as full text. Retained claims for those rows are bounded by accessible abstracts/official records; detailed fields remain `NR` where support was insufficient.
- Repository commit pins capture the audited state on 2026-06-29. Historical release state may differ.
- BG-012 provenance is owner-matched and current but indirect because PyPI omits `project_urls`; the audited repository commit is not established as the exact source for PyPI 1.1.0, and the release-to-commit mapping remains unresolved.
- BG-045 has a live official release but no located license and conflicting benchmark counts between paper and repository.

## Validation Record

The repository validator `python3 -m paper.scripts.validate_agent_runs` was run from the survey worktree after the corrections and returned `agent discovery validation passed`. A focused two-file check also confirmed the unchanged discovery-query digest, the BG-012 release-mapping disclaimer, the BG-032 technical-field downgrade, and the dated BG-018/BG-030 `not_found` evidence.

The CSV was parsed with Python's standard `csv` module. Validation checks the literal required header; 35 fields in every row; 45 sequential candidate IDs; DOI-first and normalized-title deduplication; fixed candidate/unverified/agent values; canonical `; ` separators; absence of mixed `NR` values in `course_object`, `representation_family`, `generator_family`, `generation_role`, and `validity_strategy`; scalar membership for both status columns; a direct official package/repository/artifact locator for every positive code/asset status; and at least one complete primary/official URL in every `evidence_locator`. Saturation values are rechecked as unchanged at `4/44 = 9.09%`, `1/45 = 2.22%`, and `0/45 = 0.00%`.

These are schema, consistency, deduplication, status, and locator-presence checks. They do not assert that every paper was available in full text, that every technical claim was independently reproduced, or that every repository builds.

Files written by this run:

- `paper/data/agent_runs/blind-ground.csv`
- `paper/data/agent_runs/blind-ground.md`
