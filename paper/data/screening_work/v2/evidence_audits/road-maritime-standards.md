# Road, maritime, and standards evidence audit

This is a factual source audit, not a screening decision. `NR` means the cited primary/official source does not document the field at the cited locator. Direct-source values were recorded from the cited source versions; a live re-fetch from this worker was limited by HTTP 403 responses and an unavailable search session, so no value below is inferred from an abstract or search snippet.

## C0126 - Lanelet2: A high-definition map framework for the future of automated driving

- **Source / locator:** [DOI](https://doi.org/10.1109/ITSC.2018.8569929); official [pinned repository archive](https://github.com/fzi-forschungszentrum-informatik/Lanelet2/archive/ae39c8d673264afac2339c4f0252df53a7ba82dd.tar.gz), `README.md` lines 18-28, 41-47, 174-183. **Access:** full text plus official repository; **directly observed:** yes.
- **Values:** domain automated-road mapping; vehicle automated road vehicle; course object lanelet road corridor/network; representation family topological lanelet map with OSM interchange; generator family NR; generation role representation, routing, read/write, and validation; validity strategy map validation; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts NR; simulator/export OSM read/write; code/asset status open official source repository.
- **Ambiguity:** the locator documents a map framework, not a stochastic course generator.

## C0127 - Scenario Factory: Creating Safety-Critical Traffic Scenarios for Automated Vehicles

- **Source / locator:** [DOI](https://doi.org/10.1109/itsc45102.2020.9294629); author [manuscript](https://mediatum.ub.tum.de/doc/1546085/1546085.pdf), pp. 1-4, Section I and Section III. **Access:** full text; **directly observed:** yes.
- **Values:** domain automated-road safety testing; vehicle automated vehicle; course object selected OSM intersections/road networks populated with traffic; representation family OSM road network plus traffic scenario; generator family road-network extraction/selection and scenario optimization; generation role selects source road geometry and generates traffic scenarios; validity strategy NR; reported geometry/difficulty/diversity metrics diverse road set is stated, formal metric NR; training/evaluation distribution counts NR; simulator/export format NR; code/asset status NR.
- **Ambiguity:** the same workflow combines road-network selection with actor/traffic generation; the locator does not establish de novo road-geometry synthesis.

## C0128 - CommonRoad: Composable Benchmarks for Motion Planning on Roads

- **Source / locator:** [DOI](https://doi.org/10.1109/ivs.2017.7995802); author [manuscript](https://web.archive.org/web/20180720100708id_/http://mediatum.ub.tum.de/doc/1379638/document.pdf), pp. 2 and 5, “Portability,” Section V.A “Road Network,” Figure 4. **Access:** full text; **directly observed:** yes.
- **Values:** domain road motion planning; vehicle autonomous road vehicle; course object lanelet road network with goals and constraints as course/benchmark inputs; representation family CommonRoad XML scenario with lanelets and dynamic obstacles; generator family NR; generation role benchmark/scenario assembly and representation interchange; validity strategy NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts constructed scenarios, count NR; simulator/export CommonRoad XML; code/asset status official benchmark/scenario assets documented.
- **Ambiguity:** composition is documented, while a probability distribution over road geometries is NR.

## C0135 - Spirale at the SBFT 2023 Tool Competiton - Cyber-Physical Systems Track

- **Source / locator:** [DOI](https://doi.org/10.1109/sbft59156.2023.00007); [pinned public implementation](https://codeload.github.com/domenico-devivo/cps-tool-competition/tar.gz/d378dae5bfc8e5fe3b015a5b90119cabd74db23c), `spirale/README.md` lines 5-12 and 27-38; `spirale/base.py` lines 31-116. **Access:** pinned public implementation; **official status:** NR; **directly observed:** yes.
- **Values:** domain autonomous-driving lane-keeping testing; vehicle road car; course object road point sequence; representation family spiral-arc road points; generator family randomized constructive/evolutionary generation with crossover; generation role generates and selects test roads by test-fitness objective; validity strategy NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts NR; simulator/export format NR; code/asset status pinned public implementation; official status NR.
- **Ambiguity:** the artifact exposes road points and evolutionary operations, but not a documented interchange file format.

## C0137 - Preliminary Evaluation of Path-aware Crossover Operators for Search-Based Test Data Generation for Autonomous Driving

- **Source / locator:** [DOI](https://doi.org/10.1109/SBST52555.2021.00020); author [manuscript](https://web.archive.org/web/20211220135757id_/https://coinse.kaist.ac.kr/publications/pdfs/Han2021vp.pdf), pp. 1-3, Introduction and Section II. **Access:** full text; **directly observed:** yes.
- **Values:** domain autonomous-driving test generation; vehicle road car; course object connected road network/path; representation family incremental road-map segments and paths; generator family path-aware crossover; generation role recombines parent road maps and generates road networks; validity strategy generated-network validation; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts NR; simulator/export format NR; code/asset status NR.
- **Ambiguity:** the cited sections identify AsFault as the underlying road-network builder; separate simulator and serialization details are NR here.

## C0143 - Wasserstein generative adversarial networks for online test generation for cyber physical systems

- **Source / locator:** [DOI](https://doi.org/10.1145/3526072.3527522); author [preprint](https://arxiv.org/pdf/2205.11060v1), Section 2.1 “Feature Representation” and Algorithm 1. **Access:** full text; **directly observed:** yes.
- **Values:** domain cyber-physical autonomous-driving lane-keeping testing; vehicle road car; course object road; representation family curvature vector plus plane points; generator family WGAN (WOGAN); generation role online candidate-road generation; validity strategy nonintersection and turn constraints before execution; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts NR; simulator/export format NR; code/asset status NR.
- **Ambiguity:** the source reports feasibility constraints, but a separately named road-geometry diversity metric is NR.

## C0150 - Navigation Scenario Permutation Model for Training of Maritime Autonomous Surface Ship Remote Operators

- **Source / locator:** [publisher article](https://www.mdpi.com/2076-3417/12/3/1651) and [DOI](https://doi.org/10.3390/app12031651), Sections 2.1, 2.4, 3.3, Figure 5. **Access:** full text; **directly observed:** yes.
- **Values:** domain maritime remote operation; vehicle maritime autonomous surface ship; course object ordered waypoint navigation route; representation family parameterized waypoint route; scenario parameters waypoint distance and course-altering angle; generator family permutation model; generation role synthesizes navigation scenarios; validity strategy practical-scenario constraints, exact rule NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts 564,480 practical ordered scenarios; simulator/export format NR; code/asset status NR.
- **Ambiguity:** “practical” is used for generated scenarios; a formal validity metric and a released scenario format are NR.

## C0153 - OpenAI Gym

- **Source / locator:** [arXiv record](https://arxiv.org/abs/1606.01540); official [pinned Gym commit](https://github.com/openai/gym/commit/c17ac6cc55fda4be60548e2e05b54f22e83e2c1b), paper Sections 3-4 and `gym/envs/box2d/car_racing.py` lines 22-29, 134-171. **Access:** full text plus official repository; **directly observed:** yes.
- **Values:** domain reinforcement-learning benchmark and car racing; vehicle physics-based racing car; course object episodic car-racing track; representation family sampled checkpoints and constructed road-tile geometry; generator family randomized constructive track generator; generation role creates a track per episode; validity strategy NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts per-episode randomized track, count NR; simulator/export Box2D environment, export format NR; code/asset status open official source repository.
- **Ambiguity:** the paper specifies a general API; the track-generation evidence is in its authoritative companion implementation.

## C0165 - Mastering Diverse, Unknown, and Cluttered Tracks for Robust Vision-Based Drone Racing

- **Source / locator:** [DOI](https://doi.org/10.1109/LRA.2025.3643267); author [preprint](https://arxiv.org/pdf/2512.09571v2), Section III-C “Curriculum Learning for Generalizable Racing,” “Track Primitive Generator,” Figure 2. **Access:** full text; **directly observed:** yes.
- **Values:** domain vision-based drone racing; vehicle racing drone; course object racing track; representation family track primitives; generator family track-primitive generator used in curriculum learning; generation role synthesizes racing tracks; validity strategy NR; reported geometry/difficulty/diversity metrics track diversity is stated, formal geometry/difficulty/diversity metric NR at the cited locator; training/evaluation distribution counts NR; simulator/export format NR; code/asset status NR.
- **Ambiguity:** clutter variation and track primitives are both discussed; the cited evidence does not expose a released track format.

## C0170 - ASAM OpenDRIVE BS 1.8.1 Specification, 2024-11-21

- **Source / locator:** official [ASAM OpenDRIVE 1.8.1 specification](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/v1.8.1/specification/index.html), Section 9.2 “Road reference line,” Figure 25, Table 18, `<planView>` and `<geometry>`. **Access:** official standard; **directly observed:** yes.
- **Values:** domain road-network interchange; vehicle NR; course object road reference line/corridor; representation family ordered OpenDRIVE XML geometry elements; generator family NR; generation role serializes/interchanges road geometry; validity strategy ordered geometries must be gap-free along the reference line; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts NR; simulator/export OpenDRIVE XML; code/asset status official specification, code asset NR.
- **Ambiguity:** this standard defines a representation and constraints, not a sampling procedure.

## C0178 - EUFS maps

- **Source / locator:** official [EUFS maps repository](https://gitlab.com/eufs/public/eufs_maps), pinned [`writer.hpp`](https://gitlab.com/eufs/public/eufs_maps/-/blob/ccd652da93bcd16b86d89ad1c68dcca440265cbb/include/eufs_maps/io/writer.hpp), lines 38-82; `README.md` lines 1-12; `eufs_maps/competitions` and `eufs_maps/tracks`. **Access:** official repository; **directly observed:** yes.
- **Values:** domain Formula Student Driverless simulation; vehicle Formula Student race car; course object cone-bounded track; representation family cone coordinates with covariance; generator family NR; generation role serializes course maps; validity strategy NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts repository course set, count NR; simulator/export EUFS course-map CSV; code/asset status open official repository with map assets.
- **Ambiguity:** the writer provides serialization; no general track-sampling algorithm is documented at the cited locator.

## C0180 - F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning

- **Source / locator:** official [PMLR paper page](https://proceedings.mlr.press/v123/o-kelly20a.html), PDF pp. 6-7 (printed pp. 82-83), Sections 5, 6.1, 6.2, Table 1. **Access:** full text; **directly observed:** yes.
- **Values:** domain autonomous racing and reinforcement learning; vehicle 1/10-scale race car; course object fixed racetrack map; representation family supplied racetrack/task map; generator family NR; generation role fixed-course evaluation; validity strategy NR; reported geometry/difficulty/diversity metrics lap time and collision-free completion are evaluation metrics; geometry/difficulty/diversity metrics NR; training/evaluation distribution counts fixed racetrack tasks, count NR at the cited locator; simulator/export F1TENTH simulation and hardware evaluation, map export format NR; code/asset status open-source environment is reported, exact artifact locator NR.
- **Ambiguity:** obstacles are placed on known maps; this locator does not document generation of track geometry.

## C0182 - f1tenth/f1tenth_racetracks

- **Source / locator:** official [repository](https://github.com/f1tenth/f1tenth_racetracks), pinned [commit](https://github.com/f1tenth/f1tenth_racetracks/commit/b95c4eff766f6367d66b310ea20cd2c9563712c0), `README.md` lines 1-2, 24-64, 68-71; BrandsHatch centerline lines 1-8. **Access:** official repository; **directly observed:** yes.
- **Values:** domain autonomous racing; vehicle F1TENTH race car; course object racetrack; representation family map, centerline, width, raceline, and waypoint files; generator family NR; generation role releases course assets/representation; validity strategy NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts repository course set, count NR; simulator/export simulator map files plus centerline/raceline/waypoint assets; code/asset status open official repository with track assets.
- **Ambiguity:** the repository supplies tracks rather than documenting an algorithm that creates them.

## C0185 - FSSIM

- **Source / locator:** official [repository](https://github.com/AMZ-Racing/fssim), pinned [`track.py`](https://github.com/AMZ-Racing/fssim/blob/cf652d8d3f1e13031dad3fb75eb3d4e6fbaaeff4/fssim_rqt_plugins/rqt_fssim_track_editor/src/rqt_fssim_track_editor/track.py), lines 117-135, 187-235, 302-360, 399-419; `fssim_common/msg/Track.msg` lines 1-7. **Access:** official repository; **directly observed:** yes.
- **Values:** domain Formula Student racing simulation; vehicle Formula Student race car; course object cone and timing-device track; representation family typed cone/timing-device coordinate arrays; generator family interactive track editor; generation role constructs and serializes tracks; validity strategy NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts NR; simulator/export FSSIM/ROS `Track` message interface; code/asset status open official source repository.
- **Ambiguity:** construction and serialization are exposed, but a randomized track distribution is NR.

## C0192 - OpenDRIVE standalone mode

- **Source / locator:** official [CARLA 0.9.12 documentation](https://carla.readthedocs.io/en/0.9.12/adv_opendrive/), “OpenDRIVE standalone mode,” “Run a standalone map,” “Mesh generation,” `client.generate_opendrive_world()`. **Access:** official documentation; **directly observed:** yes.
- **Values:** domain road simulation; vehicle simulated road vehicle; course object OpenDRIVE road network; representation family serialized OpenDRIVE road geometry; generator family procedural mesh/world generation; generation role converts supplied road geometry into navigable world geometry; validity strategy NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts NR; simulator/export CARLA input OpenDRIVE and generated road mesh/boundaries; code/asset status official documented API, source asset NR.
- **Ambiguity:** the API generates a world mesh from input geometry; it does not select or sample the input road network.

## C0193 - Plan File Format

- **Source / locator:** official [QGroundControl documentation](https://docs.qgroundcontrol.com/master/en/qgc-dev-guide/file_formats/plan.html), “Plan File,” “Mission Object,” “SimpleItem,” “Complex Mission Item,” and “CorridorScan.” **Access:** official documentation; **directly observed:** yes.
- **Values:** domain unmanned-vehicle mission planning; vehicle generic MAV/vehicle; course object ordered mission route and corridor scan; representation family JSON plan schema with mission items and geographic coordinates; generator family NR; generation role route serialization/interchange; validity strategy NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts NR; simulator/export QGroundControl JSON `.plan` format; code/asset status official format specification, code asset NR.
- **Ambiguity:** the schema serializes provided mission structures; it does not define a course generator.

## C0198 - RoboBoat 2026 | Team Handbook

- **Source / locator:** official [RoboBoat 2026 Team Handbook](https://robonation.org/app/uploads/sites/3/2025/10/RoboBoat-2026-Team-Handbook_111125.pdf), pp. 45-50, Sections 3.2.1-3.2.3 and task-layout figures; pp. 58-60, Sections 3.2.7, 3.4-3.5. **Access:** official handbook; **directly observed:** yes.
- **Values:** domain maritime autonomous-vehicle competition; vehicle autonomous surface vessel; course object entry/exit gates, buoy channels, debris constraints, and task layouts; representation family physical buoy/gate layout plus handbook rules; generator family NR; generation role defines a fixed competition course set; validity strategy NR; task/evaluation rules traversal and scoring constraints; reported geometry/difficulty/diversity metrics course dimensions and scoring values NR at the cited locator; training/evaluation distribution counts multi-task course set, count NR; simulator/export physical-course specification, export format NR; code/asset status official handbook only, downloadable course asset NR.
- **Ambiguity:** variable task elements are documented, but no procedural course-generation algorithm is specified.

## C0200 - RobotX 2026 | Team Handbook

- **Source / locator:** official [RobotX 2026 Team Handbook](https://robonation.org/app/uploads/sites/2/2026/06/RobotX-2026_Team-Handbook-20260625.pdf), Change Log pp. 6-8; Section 3.3.2, pp. 56-58; Section 3.5, pp. 70-76. **Access:** full official handbook; **directly observed:** yes.
- **Values:** domain maritime autonomous-vehicle competition; vehicle autonomous surface craft; course object safe route through buoy gates; representation family physical buoy-gate route and task rules; generator family per-run route randomization; generation role samples safe routes through gates; validity strategy physical course constraints and task rules; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts randomized each run, cardinality NR; simulator/export physical-course specification, export format NR; code/asset status official handbook only, code/asset release NR.
- **Ambiguity:** the handbook states run-level route randomization but does not expose its sampling algorithm or distribution.

## C0209 - VRX 2023 Task Descriptions

- **Source / locator:** official [VRX 2023 Task Descriptions v1.2](https://robonation.org/app/uploads/sites/2/2023/07/VRX2023_Task-Descriptions_v1.2.pdf), pp. 2-4; Section 4.2 “Follow the Path,” pp. 9-10; Section 3.2 “Wayfinding,” Table 3; Section 3.6 “Navigation Channel.” **Access:** full official task specification; **directly observed:** yes.
- **Values:** domain maritime autonomous-vehicle simulation competition; vehicle autonomous surface vessel; course object random goals/waypoints and ordered colored-buoy gate channels; representation family waypoint and buoy-gate task specification; generator family random goal/waypoint placement; generation role generates task instances; validity strategy gate direction, width, traversal, and scoring constraints; reported geometry/difficulty/diversity metrics geometry/scoring values NR at the cited locator; training/evaluation distribution counts random task instances, count NR; simulator/export VRX task specification, export format NR; code/asset status official PDF task description, code asset NR.
- **Ambiguity:** the specification documents random placement and constraints, but no serialized task schema is provided in this source.

## C0210 - VRX Automated Evaluation

- **Source / locator:** official [repository](https://github.com/osrf/vrx-docker), pinned [README](https://github.com/osrf/vrx-docker/blob/f599871a83ddfef9851e2f9bc95d082baff47bf2/README.md), `prepare_task_trials.bash` lines 3-5, 32-51; `task_config/gymkhana.yaml` lines 1-19, 46-69, 97-120, 147-170; README lines 21-29, 43-53. **Access:** official repository; **directly observed:** yes.
- **Values:** domain maritime autonomous-vehicle evaluation; vehicle autonomous surface vessel; course object ordered marker-gate navigation course; representation family YAML task/trial configuration; generator family scripted task-world generation from YAML trials; generation role creates evaluation worlds and serializes gate courses; validity strategy evaluator-specific geometric validity rule NR; reported geometry/difficulty/diversity metrics NR; training/evaluation distribution counts trial set documented, count NR; simulator/export YAML task config and generated VRX/Gazebo world; code/asset status open official repository.
- **Ambiguity:** the repository generates configured trial worlds; any probability distribution over configurations is NR.

## Retrieval limitations and self-check

Live HTTP retrieval from this worker was blocked for attempted public sources (HTTP 403) and the web search session was unavailable. The audit therefore preserves the version-pinned primary/official URLs and exact locators, and records `NR` rather than filling unobserved fields from abstracts. Self-check: the 20 assigned candidate IDs each appear once as a section heading.
