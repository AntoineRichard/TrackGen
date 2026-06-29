# Corpus-aware geometry/RL discovery report

Date: 2026-06-29  
Worker: `aware-geometry-rl`  
Owned data file: `paper/data/agent_runs/aware-geometry-rl.csv`

## Scope and decision rule

This search focused on primary work that synthesizes, mutates, selects, repairs, validates, serializes, or benchmarks ground-racing and road/course geometry. It also retained methodological boundary work when it contributes a transferable representation, validity method, difficulty metric, task-distribution method, or evaluation protocol.

The bootstrap files `paper/data/candidates.csv` and `paper/data/seed_coverage.csv` were used only as seed maps. They were not treated as scope boundaries or as evidence. Every retained row points to a primary paper/proceedings record, an authoritative DOI/arXiv record, an official standard, or an official project repository.

The key screening distinction is:

- Course contribution: creates or changes the road, circuit, path, lane graph, world geometry, or distribution of those objects.
- Selection contribution: chooses or replays already parameterized tasks or generated tests.
- Repair contribution: projects geometry toward constraints or removes invalid/self-intersecting configurations.
- Benchmark/serialization contribution: defines an exchange format, fixed corpus, competition, or simulator protocol.
- Fixed-track control contribution: optimizes a trajectory or controller inside supplied boundaries. These are retained only as explicitly labeled boundary cases.
- Excluded scenario-only contribution: changes traffic actors, behavior, appearance, or surroundings while course geometry remains fixed.

## Search surfaces

### Local corpus

The following files were read completely before searching:

- `docs/superpowers/specs/2026-06-29-track-generation-survey-design.md`
- `docs/related-work/state-of-the-art.rst`
- `docs/related-work/prior-art.rst`
- `docs/generators/benchmarks.rst`
- `paper/data/README.md`
- `paper/data/taxonomy.json`
- `paper/data/candidates.csv`
- `paper/data/seed_coverage.csv`

### External surfaces

- Crossref REST API for DOI, author, year, venue, and title verification.
- DOI content negotiation (`Accept: application/x-bibtex`) when Crossref JSON omitted an otherwise registered record.
- arXiv export API and primary PDFs for preprints and paper sections.
- Official ACM, IEEE, Springer, Elsevier, PMLR, CVF, PNAS, and ASAM records.
- Official author/project repositories for code, schemas, generated data, and simulator/export behavior.
- OpenAlex only for lead discovery where a registry query was incomplete. It is not used as final evidence in the CSV.
- Search snippets and secondary pages were used only to identify leads, never as `metadata_evidence` or `evidence_locator`.
- Crossref and DOI-issued BibTeX were used for metadata only. Technical coding requires a primary full-text section/page, an official source-code path, or an official specification section. Abstract and DOI landing pages are not field-level technical evidence.

## Evidence audit revision

The 55-row file was revalidated after review. For the nine rows in this final correction, every retained non-`NR` technical field now has an `evidence_locator` containing a complete primary/official URL plus a precise section, page, table, algorithm, documentation subsection, or source-code path. Vague repository-level and “primary paper” locators in those rows were replaced.

For these nine rows, when no primary full text or official source path was retrievable, unsupported representation, generator/role, validity, metric, simulator/export, code/asset, distribution/evaluation, and reproducibility fields were set to `NR`; abstracts were not retained as field-level support. The remaining retrieval need was added to `coding_notes`. The audit also corrected AGRL0003 to its closed B-spline/Bezier representation, AGRL0042 from BeamNG to the fixed-track Udacity simulator study, and several values that described internal representations rather than serialized export formats.

### Final nine-row evidence correction

The final review targeted AGRL0011, AGRL0021, AGRL0027, AGRL0028, AGRL0031, AGRL0032, AGRL0041, AGRL0043, and AGRL0044. Screening statuses and saturation membership did not change.

| Row | Direct primary/official support added | Fields downgraded or left `NR` |
|---|---|---|
| AGRL0011 | None; DOI/Crossref metadata only | Downgraded `representation_family`, `generator_family`, `generation_role`, `validity_strategy`, `geometry_metrics`, `training_distribution`, `evaluation_suite`, `simulator`, and `reproducibility_fields`; all other audited technical fields remain `NR` |
| AGRL0021 | SciOpen accepted manuscript Secs. 2.1-4.3 and ETS package support representation, synthesis/serialization, penalties, metrics, CARLA, OpenDRIVE, code, assets, distribution, evaluation, and reproducibility | `diversity_metrics` downgraded to `NR`; the 15 complexity levels are a distribution design, not a diversity metric |
| AGRL0027 | Authors' RoadSearch repository paths support all retained representation, search/mutation, validity, OOB/diversity metric, BeamNG.tech, CSV, code, asset, distribution, evaluation, and reproducibility fields | None |
| AGRL0028 | None; DOI metadata only | Downgraded `representation_family`, `generator_family`, `generation_role`, `validity_strategy`, `geometry_metrics`, `difficulty_metrics`, `training_distribution`, `evaluation_suite`, `simulator`, and `reproducibility_fields`; remaining audited fields stay `NR` |
| AGRL0031 | None; DOI/Crossref metadata only | Downgraded `representation_family`, `generator_family`, `generation_role`, `validity_strategy`, all three metric fields, `training_distribution`, `evaluation_suite`, and `reproducibility_fields`; simulator/export/code/assets remain `NR` |
| AGRL0032 | ACM full text Secs. 3-5.5 plus Zenodo/GitHub support representation, evolutionary/RL generation, mutation, validity, metrics, BeamNG.tech/MuJoCo, code, assets, distribution, evaluation, and reproducibility | No field was downgraded to `NR`; `export_format` remains `NR`, and role/validity labels were recoded to the directly reported mutation, rejection, and penalty behavior |
| AGRL0041 | Springer full text Secs. 2.2-4.2.3 plus GitHub/Zenodo support road features, selection role, metrics, BeamNG.tech, code, assets, distributions, evaluations, and reproducibility | No field was downgraded; `export_format` remains `NR` because no course serialization schema is defined |
| AGRL0043 | White Rose manuscript Secs. 3-4.1 and the author source archive support world/mission representation, PCG, validity, metrics, Player/Stage files/logs, code, assets, distribution, evaluation, and reproducibility | `diversity_metrics` downgraded to `NR`; 500 situations is a distribution size, not a diversity metric |
| AGRL0044 | HAL manuscript Secs. III-VI supports world representation/generation, constraints, metrics, MORSE/Blender/Bullet, distributions, evaluations, and reproducibility | No newly unsupported field; export/code/assets remain `NR` because the full text reports no official package or serialized world format |

The three unresolved technical retrievals are therefore AGRL0011 (IEEE Transactions on Games), AGRL0028 (SBST 2021), and AGRL0031 (ISSRE 2023). Their scope-level course labels are preserved, but no detailed technical field relies on an abstract.

## Exact queries

The API wrapper differed by surface, but the literal search strings were as follows.

### Seed verification and racing-game expansion

- `procedural race track generation games`
- `racing game track design generation evolutionary`
- `racing game track generation interactive evolution`
- `TrackGen interactive track generator TORCS Speed-Dreams`
- `personalised track design car racing games generator difficulty`
- `procedural generation road paths driving simulation`
- `Learn to Race autonomous racing environment`
- `CommonRoad composable benchmarks road serialization lanelet official`
- `Making Racing Fun Through Player Modeling and Track Evolution`
- `Search-Based Procedural Content Generation for Race Tracks in Video Games`
- `Repulsive Curves Keenan Crane Yu Schumacher`
- `Position Based Dynamics Muller Heidelberger Hennix Ratcliff`

### Autonomous-driving road generation and testing

- `autonomous driving road geometry test generation search procedural`
- `DeepJanus DeepHyperion road generation autonomous driving`
- `AmbieGen automatic generation testing environments road topology`
- `road representations search-based testing autonomous driving systems six representations`
- `path-aware crossover autonomous driving road generation AsFault`
- `Frenetic-lib road structures ADS testing Bezier Cartesian Kappa Theta`
- `CRAG combinatorial testing generator road geometries ADS`
- `critical scenario generation integrating road structures EvoScenario`
- `reinforcement learning informed evolutionary search road topology generation autonomous testing`
- `SBST tool competition BeamNG road test generation benchmark`
- `forward citations ASFault machine learning test selection generated road tests`
- `autonomous vehicle testing quality metrics oracles road diversity`
- `complexity controllable road network generation virtual testing autonomous driving`

### UED, learned generation, robot-world transfer, and difficulty

- `REPAIRED CarRacing Bezier F1 tracks`
- `CLUTR CarRacing task manifold F1`
- `Prioritized Level Replay task selection`
- `PAIRED unsupervised environment design regret`
- `ACCEL edit levels regret environment design`
- `WOGAN learned test generator cyber physical systems road generation RIGAA baseline`
- `procedural content generation autonomous robot control testing generated worlds`
- `generated virtual worlds robot navigation difficulty level trajectory curves`
- `evolving diverse collection robot path planning problems generated environments`

### Final residual saturation round

These four Crossref `query.bibliographic` strings were run as the final round:

- `procedural race course synthesis ground vehicle geometry repair projection`
- `autonomous driving road geometry generation curriculum environment design`
- `racetrack generator serialization validity TORCS BeamNG CarRacing`
- `road topology generation reinforcement learning autonomous systems testing`

They returned generic geometry, trajectory-planning, road-design, and unrelated robotics records, but no additional in-scope primary course source after deduplication and screening.

Exact-title and DOI lookups were also run for each candidate. Examples include `query.title=Global Curvature Thickness and the Ideal Shapes of Knots`, `query.title=Learn to Race autonomous racing environment`, and direct `/works/{doi}` requests. These were metadata verification operations, not separate expansion rounds.

## Citation-chain expansion

### Racing-game PCG chain

The 2011 Loiacono, Cardamone, and Lanzi paper was the main backward/forward seed.

- Backward links recovered Togelius et al. 2006 and 2007 as segment-string evolution and player-model conditioning.
- Same-author backward expansion found Cardamone et al. 2011 on interactive evolution.
- Forward expansion found the 2015 TrackGen system paper and Georgiou and Demiris 2016 on personalized track design.
- Alternate terms `road paths`, `chain code`, `interactive evolution`, and `authoring` added work that does not consistently use the phrase `track generation`.
- Henrich and Koetter 2025 closes the geometry chain from repulsive curves to playable 3D race-track authoring.

### Autonomous-driving test chain

ASFault, DeepJanus, and the seeded Bezier ALKS work anchored this branch.

- ASFault/DeepJanus forward links found DeepHyperion, the six-representation analysis, path-aware crossover, SBST competition protocols, quality metrics/oracles, and machine-learning test selection.
- Representation terminology exposed Frenetic-lib and its Bezier, Cartesian, kappa, and theta encodings.
- Combinatorial-testing terminology exposed CRAG.
- Integrating road structures with dynamic scenarios exposed EvoScenario.
- AmbieGen backward/forward links exposed its 2022 and 2023 versions, then RIGAA and WOGAN.
- Current-year expansion found Zhu et al. 2026, which links prescribed road-network complexity to driving outcomes.

### UED and curriculum chain

PLR and PAIRED were followed through REPAIRED, ACCEL, and CLUTR.

- REPAIRED and CLUTR actually instantiate bounded Bezier CarRacing tracks and evaluate zero-shot transfer on 20 human-designed Formula 1 tracks.
- PLR contributes replay/selection.
- PAIRED contributes regret-based environment design.
- ACCEL contributes level editing and replay.
- The latter three are useful methodological boundaries but do not report a new ground-racing geometry representation in their original experiments.
- WOGAN was reached through the RIGAA comparison chain but is excluded after review: its original paper generates generic CPS test vectors and contributes no course geometry, course mutation, repair, serialization, or course distribution.

### Geometry validity and projection chain

- Gonzalez and Maddocks supplies global curvature and thickness.
- Yu, Schumacher, and Crane supplies tangent-point repulsion and self-avoidance optimization.
- Henrich and Koetter applies repulsive curves to race-track authoring.
- PBD and XPBD supply generic constraint projection and compliant projection machinery.
- These mathematical/solver papers are labeled as repair or metric boundaries; they are not counted as racing generators.

### Dynamics and fixed-track chain

Heilmeier et al. and Christ et al. were verified but coded as fixed-track boundaries. Their curvature, friction, feasibility, and lap-time quantities are useful difficulty signals, yet neither paper creates or mutates the circuit. Learn-to-Race is similarly a fixed-track benchmark, not a generator.

## Inclusion and boundary judgments

| Class | Treatment | Representative rows |
|---|---|---|
| Closed-course synthesis and/or export | In scope | Loiacono 2011; TrackGen 2015; Gymnasium CarRacing; Henrich 2025 |
| Road-network/corridor synthesis | In scope | PGDrive; MetaDrive; ASFault; AmbieGen; Frenetic-lib; CRAG; EvoScenario |
| Adaptive geometry/task distributions | In scope with exact role | REPAIRED; CLUTR; RIGAA; DeepHyperion |
| Serialization and benchmark distributions | Retained boundary | OpenDRIVE; CommonRoad; SBST competition; Learn-to-Race |
| Geometry repair/projection | Retained boundary | Repulsive Curves; PBD; XPBD; global thickness |
| Fixed-track trajectory/control | Retained boundary | Heilmeier 2019; Christ 2019 |
| Robot-world generation and difficulty | Retained transfer boundary | Arnold 2013; Sotiropoulos 2016; Ashlock 2006 |
| Non-course test, traffic, or surroundings generation | Excluded with reason | WOGAN; AV-FUZZER; Scenario Factory; Simunek et al. 2025 |

The CSV keeps `screening_status=candidate` for retained rows, including boundaries, and uses controlled `generation_role=boundary_case` where applicable. Four out-of-scope rows use `screening_status=excluded` and a nonempty `exclusion_reason`: WOGAN, AV-FUZZER, Scenario Factory, and the race-track-surroundings paper.

## Terminology absent from the supplied brief

The following terms materially expanded recall:

- `illumination search`, `MAP-Elites`, and `feature-space exploration`
- `road representation`, `kappa`, `theta`, `Cartesian`, and `Frenet encoding`
- `path-aware crossover` and phenotype-preserving crossover
- `combinatorial road geometry` and covering-array generation
- `road structure integration` in critical-scenario generation
- `potential collision risk` as a road-network complexity measure
- `interactive evolution`, `track authoring`, and `road path generation`
- `dual curriculum design`, level replay, level editing, and task-manifold learning
- `global radius of curvature`, thickness, ropelength, and tangent-point energy
- `virtual-world difficulty`, mission duration, and trajectory curvature
- `test selection` over corpora of already generated road cases

These terms identify materially different roles. In particular, `scenario generation` alone has low precision because much of that literature fixes the road and varies only actors.

## Sparse and contradictory areas

- Joint geometry generation plus RL remains sparse. REPAIRED and CLUTR adapt task distributions over a bounded CarRacing generator; RIGAA uses RL to seed evolutionary search. Few sources learn a validity-preserving road representation end to end.
- Closed-track papers frequently report preference, entropy, or playability but omit explicit minimum-clearance, curvature-bound, or self-intersection statistics.
- AV testing papers report many failure oracles but often use open corridors rather than closed racing tracks.
- Export is concentrated in TORCS/Speed-Dreams, OpenDRIVE, CommonRoad, MetaDrive-native scenarios, and BeamNG tooling. Cross-tool round trips are rarely evaluated.
- Dynamics-aware difficulty is usually downstream: speed-profile entropy, lane departure, regret, lap time, friction, mission duration, or safety violations. Geometry-to-dynamics calibration is uncommon; Zhu et al. 2026 is a notable direct link.
- Official code was verified only where an author/project repository or paper explicitly supplied it. Silence remains `NR`; it was never converted to `not_found`.
- Asset status is especially sparse and remains `NR` unless an official repository or paper exposes data/assets.
- Frenetic-lib has contradictory license metadata: the repository API reports GPL-3.0 while README text/badges mention MIT. The CSV records code openness but not a resolved license.
- Publication-year labels differ between preprint and venue records. The CSV uses the venue year where a venue publication exists: CLUTR 2023, MetaDrive 2023, and the two Vehicle System Dynamics DOI records as 2019. The coding does not silently inherit the later years shown in bootstrap prose.
- No official author code repository was verified for HDMapGen. An unofficial reimplementation was not used to infer official code status.
- The evidence audit corrected AGRL0042 to the Udacity self-driving car simulator and its three fixed tracks; BeamNG appears only in its related-work discussion, not as the empirical platform.

## Integration-conflict note for Task 5

Per the reviewer-supplied cross-stream summary, three independent observations disagree with the simulator stream and must be preserved as integration conflicts: CarRacing (AGRL0013), OpenDRIVE (AGRL0016), and CommonRoad (AGRL0019). This worker did not read or alter the other stream. Task 5 should retain these rows and reconcile the conflicting generator, serialization, and benchmark interpretations explicitly rather than silently preferring either stream.

## Saturation arithmetic

Only unique retained in-scope/boundary candidates count toward the denominator. The four explicit exclusions do not count. Deduplication used normalized DOI first and normalized title otherwise.

| Round | Expansion focus | Prior unique retained | New retained | Rate | New total |
|---|---|---:|---:|---:|---:|
| Baseline | Verified bootstrap-represented sources in this worker's scope | 0 | 29 | n/a | 29 |
| 1 | Racing-game backward/forward links, road-path terminology, CommonRoad, Learn-to-Race | 29 | 6 | 20.69% | 35 |
| 2 | ASFault/DeepJanus/AmbieGen chains, representations, crossover, Frenetic, CRAG, EvoScenario, RIGAA, SBST | 35 | 10 | 28.57% | 45 |
| 3 | Learned test distributions, AV quality metrics, robot-world generation, difficulty, diverse planning problems | 45 | 4 | 8.89% | 49 |
| 4 | Current complexity-controlled road generation and generated-test selection | 49 | 2 | 4.08% | 51 |
| 5 | Four residual cross-domain/alternate-terminology queries listed above | 51 | 0 | 0.00% | 51 |

AGRL0040 (WOGAN) was removed from round 3 after the evidence audit, so round 3 contributes four rather than five retained sources. Rounds 4 and 5 remain consecutive sub-5% rounds: `2 / 49 = 4.08%`, then `0 / 51 = 0.00%`. The stopping criterion therefore remains met without an additional refinement round.

WOGAN is an excluded round-3 lead. The three round-5 leads retained as exclusion records also do not alter the retained total:

- AV-FUZZER: traffic behavior on fixed maps.
- Scenario Factory: actor/scenario synthesis on existing roads.
- Controllable Procedural Generation of Race Track Surroundings: scenery around a supplied track.

## High-priority manual retrieval

The final audit retrieved primary full text or official source paths for Zhu et al. 2026, Castellano et al. 2021, Humeniuk et al. 2024, Birchler et al. 2023, Arnold and Alexander 2013, and Sotiropoulos et al. 2016. The following sources still need primary full-text retrieval for fields left `NR`:

1. Georgiou and Demiris 2016, `10.1109/cig.2016.7860435`: exact track encoding, simulator, and validity procedure.
2. Campos, Leitao, and Coelho 2015, `10.4018/ijcicg.2015070103`: representation, generator family, construction constraints, simulator, and evaluation.
3. Alyaseri and Conner 2024, `10.4018/ijamc.350330`: representation, validity checks, metric formulas, and implementation status.
4. Henrich and Koetter 2025 (AGRL0011), `10.1109/tg.2025.3561107`: all detailed representation, generator/role, validity, metric, simulator/export, distribution/evaluation, code/asset, and reproducibility fields.
5. Arcaini and Cetinkaya 2024, `10.1016/j.scico.2024.103171`: CRAG encoding, constraints, simulator, export, and generated assets.
6. Han et al. 2021 (AGRL0028), `10.1109/sbst52555.2021.00020`: crossover representation/operator, validity, metrics, simulator, distribution, and evaluation.
7. Tang et al. 2023 (AGRL0031), `10.1109/issre59848.2023.00054`: road/maneuver representation, generator/role, validity, metrics, distributions, evaluation, named industrial simulator, and export schema.
8. Ashlock, Manikas, and Ashenayi 2006, `10.1109/cec.2006.1688530`: exact problem representation and validity treatment.
9. Heilmeier et al. and Christ et al. 2019: named simulator/tooling, reproducibility package, and any output interchange format.

## Counts and validation

- CSV rows: 55.
- Retained candidates/boundaries: 51.
- Explicit exclusions: 4.
- Bootstrap-represented retained rows: 29.
- Newly discovered retained rows: 22.
- Newly discovered excluded rows: 4.
- Columns: 35 per header and per row.
- Rows missing `title`: 0.
- Rows missing `metadata_evidence`: 0.
- Rows missing `evidence_locator`: 0.
- Reported normalized DOIs: 47; DOI values set to `NR`: 8; duplicate DOI groups: 0.
- Normalized titles: 55; duplicate title groups: 0.
- Candidate IDs unique: yes.
- Cite keys unique: yes.

Validation used Python's `csv` module with `newline=""` and UTF-8, compared the parsed header to the required literal header, checked every parsed row width, validated controlled labels, rejected mixed `NR` list values and noncanonical semicolon spacing, checked direct locator URLs/precision markers for the nine final-audit rows, and normalized DOI/title keys independently.

## Verification limitations

- DOI and venue metadata are authoritative, but Crossref supports metadata only; it is never used as technical evidence. AGRL0011, AGRL0028, and AGRL0031 still require manual primary-full-text access, and their detailed technical fields are `NR`.
- A few early papers are verified from author-hosted primary PDFs rather than DOI registries.
- `NR` means unreported or not verified in the inspected primary surface. It does not mean absent.
- Code and asset statuses were not inferred from paper silence or from unofficial mirrors.
- Simulator/export fields are conservative. A simulator or export value is `NR` when the inspected primary source did not expose a named platform or serialized schema; internal parameter vectors and geometric representations are not treated as export formats.
- This worker did not broaden into aerial-only gate-course papers except where a method directly informed a ground/road citation chain.

