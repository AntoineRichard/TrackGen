# Racing and Simulator Evidence Audit

This is a factual source audit, not a screening decision. `Direct` means the cited
primary paper, official documentation, official repository, or official handbook was
opened and the stated fact was observed at the locator. `Limited` means the cited
official/DOI endpoint exposed only landing-page metadata, or denied full-text retrieval;
fields beyond directly visible metadata are `NR`. Counts are reported only where the
source gives them.

## C0001 -- TORCS - The Open Racing Car Simulator

**Source/access.** [Official project](https://sourceforge.net/projects/torcs/) (project description and Features; Direct) and [official `maintrackgen.cpp`](https://sourceforge.net/p/torcs/code/ci/7ed3e067f1a1aae917b9bf1aa054dde7c339398d/tree/torcs/torcs/src/tools/trackgen/maintrackgen.cpp) (track-build/output block, approximately lines 400-474; Direct).

**Values.** Domain: 3D car-racing simulation/research platform. Vehicle: cars. Course object: tracks. Representation family: internal track definition consumed by `trkBuildEx`; further schema details NR. Generator family: deterministic track/terrain/object build tool. Generation role: converts a track definition into simulator assets. Validity strategy: NR. Reported geometry/difficulty/diversity metrics: NR. Training/evaluation distribution counts: NR. Simulator/export format: TORCS; track, terrain, and objects are written as `.ac` assets, with optional elevation PNGs. Code/asset status: official public source and downloadable project assets.

**Ambiguity.** The inspected tool builds supplied track definitions; the source does not document a stochastic course sampler or distribution.

## C0002 -- Speed Dreams

**Source/access.** [Official repository](https://forge.a-lec.org/speed-dreams/speed-dreams-code) (`README.md`, opening description and **Building from source**; Direct) and [official About us page](https://www.speed-dreams.net/en/about) (opening paragraph and **Features**; Direct).

**Values.** Domain: 3D motorsport simulator for games and research. Vehicle: cars. Course object: tracks. Representation family: NR. Generator family: NR. Generation role: NR. Validity strategy: NR. Reported geometry/difficulty/diversity metrics: NR. Training/evaluation distribution counts: NR. Simulator/export format: Speed Dreams simulator; repository does not document a track interchange format at the cited locator. Code/asset status: engine source is public; the README identifies a separate assets repository.

**Ambiguity.** The project is described as a TORCS fork, but inheritance of a particular track schema was not assumed.

## C0008 -- Procedural Generation of Complex Roundabouts for Autonomous Vehicle Testing

**Source/access.** [Author preprint of the primary paper](https://arxiv.org/pdf/2303.17900) (Sections III **Classic Roundabout Generation** and **Turbo Roundabout Generation**; Section IV **Evaluation**; Direct).

**Values.** Domain: autonomous-vehicle test-scenario road generation. Vehicle: autonomous road vehicle/testing target. Course object: classic and turbo roundabouts plus incident roads. Representation family: parametric road/lane geometry, circular/semicircular centerlines and incident-road parameters. Generator family: constructive geometric generation with randomized road definitions/distortion. Generation role: scenario/course construction. Validity strategy: generated road connections are stated to obey OpenDRIVE road-description rules; no separate collision or topology proof is reported. Reported geometry/difficulty/diversity metrics: radius and radius-derivative distributions; qualitative shape variation; 20 random n-way roundabouts using 35-45 m input circles, and 30 fixed-input three-way roundabouts. Training/evaluation distribution counts: those 20 and 30 generation sets; no learning split reported. Simulator/export format: OpenDRIVE road description; simulator execution/export beyond that is NR. Code/asset status: NR.

## C0013 -- Procedural Generation of Isometric Racetracks Using Chain Code for Racing Games

**Source/access.** [SBGames proceedings paper](https://www.sbgames.org/proceedings2021/ComputacaoFull/217347.pdf) (Section IV **Workflow**, Section V results/discussion; Direct).

**Values.** Domain: 2D isometric racing game. Vehicle: kart in the authors' test game. Course object: race-track circuit. Representation family: 8-neighbour chain-code direction list, converted to Cartesian placements; preprocessing includes noise/"ladder effect" filtering and detail adjustment. Generator family: chain-code/image-derived spatial construction. Generation role: game-content construction in Unity. Validity strategy: preprocessing and playability testing; no formal geometric validity test is documented. Reported geometry/difficulty/diversity metrics: mean 98.5% difference among generated tracks; gameplay standards are reported qualitatively. Training/evaluation distribution counts: source-image database exists, but its count and a train/test distribution are NR. Simulator/export format: Unity engine/game runtime; external export format NR. Code/asset status: NR.

## C0026 -- Autonomous Drone Racing with Deep Reinforcement Learning

**Source/access.** [Authors' IROS paper](https://rpg.ifi.uzh.ch/docs/IROS21_Yunlong.pdf) (Section III-C.3 track generator; Sections IV-B--D; Tables I--V; Figure 7; Direct).

**Values.** Domain: autonomous drone racing. Vehicle: quadrotor. Course object: ordered racing gates. Representation family: gate centers/orientations and relative gate transforms. Generator family: stochastic primitive/relative-pose track sampler with curriculum/track adaptation. Generation role: train, validation, and test-track construction. Validity strategy: bounded gate transformations plus crash ratio/safety-margin evaluation; an explicit global self-intersection check is NR. Reported geometry/difficulty/diversity metrics: 110-150 m trajectory length, elevation change up to 17 m, 5/15/30/35-gate random tracks, lap time, crash ratio, and safety margins. Training/evaluation distribution counts: 100 parallel configurations per training iteration; 100 unseen validation tracks; 1,000 random test tracks; three randomization difficulty levels. Simulator/export format: Flightmare; in-memory gate configurations/trajectories, external export NR. Code/asset status: paper-specific generator code NR; Flightmare is cited as the simulator dependency.

## C0038 -- Minimum curvature trajectory planning and control for an autonomous race car

**Source/access.** [DOI/publisher endpoint](https://doi.org/10.1080/00423114.2019.1631455) (publisher article-title heading, **Minimum curvature trajectory planning and control for an autonomous race car**; Limited: full article text was not retrievable in this audit).

**Values.** Domain: autonomous race-car trajectory planning/control (publisher article-title heading). Vehicle: autonomous race car (publisher article-title heading). Course object: NR. Representation family: NR. Generator family: NR. Generation role: NR. Validity strategy: NR. Reported geometry/difficulty/diversity metrics: NR. Training/evaluation distribution counts: NR. Simulator/export format: NR. Code/asset status: NR.

**Ambiguity.** No abstract-derived method claims are recorded.

## C0042 -- Computing the racing line using Bayesian optimization

**Source/access.** [arXiv primary paper](https://arxiv.org/abs/2002.04794) (Abstract; Introduction; Sections II, IV-A, IV-B, and V; Direct).

**Values.** Domain: ground autonomous-racing trajectory optimization. Vehicle: autonomous race car. Course object: NR; the method operates on a supplied track. Representation family: centerline plus width; the input is XY centerline waypoints and track width. Generator family: NR; Bayesian optimization searches a racing-line trajectory, not course geometry. Generation role: boundary case. Validity strategy: constrained minimum-time solver over a fixed trajectory, with track, actuation, and friction-circle constraints. Reported geometry metrics: NR. Difficulty metric: minimum traversal/lap time. Diversity metrics and training distribution: NR. Evaluation: two ETH Zurich 1/43-scale tracks and one UC Berkeley 1/10-scale track. Simulator/export format: NR. Code/asset status: author code is linked from Section IV; released assets NR. Reproducibility fields: centerline XY waypoints, track width, and vehicle parameters `m`, `lf`, and `lr`.

**Boundary.** This is direct boundary evidence, not course generation: it searches smooth feasible racing-line trajectories on a supplied track.

## C0053 -- ArcGIS CityEngine

**Source/access.** [Official CityEngine overview](https://www.esri.com/en-us/arcgis/products/arcgis-cityengine/overview) (page title/description and **3D GIS for urban design**; Direct).

**Values.** Domain: urban 3D design. Vehicle: NR. Course object: urban environments and scenarios. Representation family: synthetic or real-world GIS data; more specific rule-language representation NR at this locator. Generator family: procedural city design. Generation role: iterative environment/scenario creation. Validity strategy: NR. Reported geometry/difficulty/diversity metrics: NR. Training/evaluation distribution counts: NR. Simulator/export format: CityEngine application; interchange format NR. Code/asset status: commercial Esri product; public source/assets NR.

## C0054 -- Interactive procedural street modeling

**Source/access.** [Primary paper](https://web.engr.oregonstate.edu/~zhange/images/street_sig08.pdf) (Abstract; Sections 3, 5, and 6.1-6.4; Figure 2; Direct).

**Values.** Domain: adjacent street-network modeling. Vehicle: NR. Course object: road network. Representation family: waypoint graph; a street network is stored as `G = (V, E)`, with crossings as nodes and street segments as edges. Generator family: constructive and human-designed; tensor fields guide hyperstreamline tracing, while users design/edit fields and graphs. Generation role: geometry synthesis and mutation. Validity strategy: NR. Reported geometry/difficulty/diversity metrics: NR. Training/evaluation distribution counts: NR. Simulator/export format: NR. Code/asset status: NR. Reproducibility fields: water, park/forest, height, and population-density maps; tensor-field design; and graph-edit operations.

**Method.** Figure 2 identifies street-graph generation as the second pipeline stage. Sections 6.1-6.2 trace hyperstreamlines from tensor fields and construct nodes at intersections with segments between consecutive intersections; Sections 6.3-6.4 provide interactive graph editing and tensor-field-based replacement of regions in an existing street graph.

## C0057 -- Repulsive Curves

**Source/access.** [Official project paper](https://www.cs.cmu.edu/~kmcrane/Projects/RepulsiveCurves/RepulsiveCurves.pdf) (abstract; Sections 1, 3, and 4; Direct).

**Values.** Domain: computational curve/shape design. Vehicle: NR. Course object: plane and space curves. Representation family: embedded curves. Generator family: continuous global energy optimization using tangent-point energy and fractional Sobolev descent. Generation role: shape design/optimization, not a simulator course generator. Validity strategy: tangent-point energy tends to infinity as nonlocal curve points approach, providing a self-intersection barrier; constraints/obstacle penalties can be incorporated. Reported geometry/difficulty/diversity metrics: energy/optimization convergence and timing comparisons; no course difficulty or diversity distribution. Training/evaluation distribution counts: NR. Simulator/export format: NR. Code/asset status: project paper is public; implementation/artifact availability is NR at the inspected paper locator.

## C0069 -- Experience-Driven Procedural Content Generation

**Source/access.** [Authors' primary paper](https://julian.togelius.com/Yannakakis2011Experiencedriven.pdf) (Sections 2, 4.2.1 **Example: racing game tracks**, and 5; Direct).

**Values.** Domain: experience-driven procedural content generation for games. Vehicle: car controller in the racing-game example. Course object: racing-game tracks. Representation family: fixed-length parameter vectors interpreted as B-splines/Bezier-curve sequences. Generator family: experience-model-guided search; evolutionary algorithms are the stated common optimization mechanism. Generation role: offline or online personalization of game content, depending on the evaluation mechanism. Validity strategy: simulation-based evaluation with neural controllers; no explicit geometric-validity check is reported for the racing example. Reported geometry/difficulty/diversity metrics: progress, variation in progress, and difference between maximum and average speed; example difficulty is described by narrow sections/sharp versus gentle turns. Training/evaluation distribution counts: human player data are mentioned, but counts/splits are NR. Simulator/export format: simple racing game in the cited example; external format NR. Code/asset status: NR.

## C0073 -- Deep Drone Racing: Learning Agile Flight in Dynamic Environments

**Source/access.** [PMLR primary paper](https://proceedings.mlr.press/v87/kaufmann18a/kaufmann18a.pdf) (Sections 3.1 and 4; supplementary Sections 6.1-6.2; Direct).

**Values.** Domain: vision-based autonomous drone racing. Vehicle: quadrotor. Course object: gate sequence/race track, including moving-gate settings. Representation family: gate-center waypoints and global minimum-snap trajectory. Generator family: layout variation by moving gates, followed by one global reference trajectory per layout. Generation role: supervision/training-data generation. Validity strategy: expert recovery within a distance margin and track-completion evaluation; explicit geometric validity checking is NR. Reported geometry/difficulty/diversity metrics: success rate and completion; simulated/real trajectory limits and prediction horizons; no named diversity statistic. Training/evaluation distribution counts: 20,000 static-track simulation images; 100,000 random-gate-position images for the dynamic simulation; real collection 25,000 static plus 15,000 dynamic images. Simulator/export format: simulated environment plus physical quadrotor; source does not name an interchange format. Code/asset status: NR.

## C0074 -- Deep Drone Racing: From Simulation to Reality With Domain Randomization

**Source/access.** [Authors' official paper copy](https://rpg.ifi.uzh.ch/docs/TRO19_Loquercio.pdf) (Section III-A **Training Procedure**, Section IV-B/IV-C, Tables I-II, Appendix C; Direct).

**Values.** Domain: sim-to-real vision-based autonomous drone racing. Vehicle: quadrotor. Course object: gate tracks/layouts. Representation family: gate locations and global trajectories through gate centers. Generator family: multiple layouts by moving gates plus visual domain randomization (background/floor/gate textures, gate shape, lighting). Generation role: training-data generation and transfer robustness. Validity strategy: DAgger recovery margin, task-completion/lap-time measurement, and visual/randomization ablations; explicit course-topology checking is NR. Reported geometry/difficulty/diversity metrics: success/task completion, best lap time, RMSE, and randomization-factor ablations. Training/evaluation distribution counts: 20,000 static and 100,000 dynamic simulated images; 25,000 static and 15,000 additional dynamic real images; approximately 10,000 real images from three indoor environments for an appendix dataset. Simulator/export format: simulation and physical quadrotor; format NR. Code/asset status: paper directs readers to the project page for source code and trained models.

## C0079 -- Deep Latent Competition: Learning to Race Using Visual Control Policies in Latent Space

**Source/access.** [PMLR primary paper](https://proceedings.mlr.press/v155/schwarting21a/schwarting21a.pdf) (Section 5 **Latent Racing Experiments**, Appendix A **Environment details**; Direct) and [official companion repository](https://github.com/igilitschenski/multi_car_racing) (linked by the paper; Direct availability claim).

**Values.** Domain: two-player autonomous racing benchmark. Vehicle: simulated race cars. Course object: `MultiCarRacing-v0` tiled track. Representation family: top-down 96x96 RGB ego observations and CarRacing-derived tiles. Generator family: environment-level randomization of vehicle color and initial position; track-generation mechanism is not documented in the paper. Generation role: training-environment variation, not a reported course generator. Validity strategy: environment collision/off-track dynamics; no generated-course validity test reported. Reported geometry/difficulty/diversity metrics: win ratio and average score; no geometry/difficulty/diversity statistic. Training/evaluation distribution counts: 500 training races per method; round-robin tournaments of 100 races per pairing (300 per tournament), repeated for five random seeds. Simulator/export format: OpenAI Gym environment; external format NR. Code/asset status: public companion repository.

## C0087 -- Indy Autonomous Challenge - Autonomous Race Cars at the Handling Limits

**Source/access.** [Authors' primary paper preprint](https://arxiv.org/pdf/2202.03807v1) (Sections 1-4; Direct).

**Values.** Domain: autonomous race-car systems at handling limits. Vehicle: autonomous race car. Course object: NR from the inspected sections for this audit. Representation family: NR. Generator family: NR. Generation role: evaluation/control-system context; no course-generation role directly documented at the inspected locator. Validity strategy: NR. Reported geometry/difficulty/diversity metrics: NR. Training/evaluation distribution counts: NR. Simulator/export format: NR. Code/asset status: NR.

**Ambiguity.** This record deliberately does not transfer IAC handbook facts to the paper candidate.

## C0088 -- Indy Autonomous Challenge

**Source/access.** [Official IAC Passing Competition Rules v1.0.0](https://www.indyautonomouschallenge.com/s/2022-ACTMS-Rules-v100.pdf) (Sections 4.4, 4.6, 4.8, 4.8.1, and 7; Direct).

**Values.** Domain: autonomous-race-car competition. Vehicle: attacker and defender race cars. Course object: fixed competition speedway/racing lines. Representation family: prescribed racing-line and safety-distance constraints. Generator family: NR. Generation role: NR. Validity strategy: rules define a completed pass (overtake, 15 m longitudinal gap, return to defensive line), two-lap attempt limit, and lateral-distance/racing-line constraints. Reported geometry/difficulty/diversity metrics: pass success, achieved speed, laps, and prescribed distances; no generated-geometry metrics. Training/evaluation distribution counts: repeated competition rounds/attempts are specified, but no training distribution. Simulator/export format: physical competition with CURWB/MyLaps telemetry; file export NR. Code/asset status: official rules publicly available; code/assets NR.

## C0104 -- Air Learning: a deep reinforcement learning gym for autonomous aerial robot visual navigation

**Source/access.** [Official Air Learning repository](https://github.com/harvard-edge/airlearning) (**Key features**, **Air Learning Environment Generator**, **Air Learning RL**, and **How to get it**; Direct). The candidate's [publisher page](https://link.springer.com/article/10.1007/s10994-021-06006-6) was limited by publisher retrieval controls.

**Values.** Domain: autonomous aerial-robot visual navigation. Vehicle: UAV. Course object: configurable obstacle/navigation environment with start-goal conditions; not an ordered racing course. Representation family: UE4 scene/mesh configuration. Generator family: configurable, randomized UE4 environment generator. Generation role: RL training and curriculum/domain randomization. Validity strategy: NR for generated-environment geometric validity. Reported geometry/difficulty/diversity metrics: repository lists success rate and additional quality-of-flight metrics, but numeric results and formal diversity metrics are NR at the README locator. Training/evaluation distribution counts: NR. Simulator/export format: UE4 plus Microsoft AirSim, exposed through an OpenAI Gym interface; external export NR. Code/asset status: public repository, with linked UE4 environment-generator and RL repositories.

## C0118 -- Virtual Worlds for Testing Robot Navigation: A Study on the Difficulty Level

**Source/access.** [HAL primary-record page](https://hal.science/hal-01328909v1) (HAL record title field, **Virtual Worlds for Testing Robot Navigation: a Study on the Difficulty Level**; Limited: HAL's advertised document endpoint returned an HTML access page rather than the manuscript during this audit) and [DOI](https://doi.org/10.1109/edcc.2016.14) (publisher metadata only).

**Values.** Domain: robot-navigation testing and difficulty study (HAL record title field). Vehicle: robot (HAL record title field); platform details NR. Course object: virtual worlds (HAL record title field). Representation family: NR. Generator family: NR. Generation role: NR. Validity strategy: NR. Reported geometry/difficulty/diversity metrics: NR; the title indicates a difficulty study, but no metric definition was accessible. Training/evaluation distribution counts: NR. Simulator/export format: NR. Code/asset status: NR.

## C0119 -- Learn-To-Race: A Multimodal Control Environment for Autonomous Racing

**Source/access.** [CVF primary paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Herman_Learn-To-Race_A_Multimodal_Control_Environment_for_Autonomous_Racing_ICCV_2021_paper.pdf) (Sections 3.1-3.2, 4.1-4.4; Tables 1-4; Direct).

**Values.** Domain: autonomous track racing. Vehicle: simulated race car. Course object: real-world-derived racetracks; the source reports scanned-dataset and custom track construction. Representation family: simulator track geometry/centerline and drivable area; source does not provide a serialized geometry schema. Generator family: NR; the source reports scanned-dataset and custom track construction but no procedural/algorithmic generator. Generation role: simulator content and task-track construction. Validity strategy: terminal condition when two wheels leave drivable area; successful episode requires three laps without leaving the drivable area; leaderboard competency check. Reported geometry/difficulty/diversity metrics: ECP, episode duration, AATS, ADE to centerline, trajectory admissibility, trajectory efficiency, and movement smoothness. Training/evaluation distribution counts: three tracks total; Track01 Thruxton and Track02 Anglesey train, Track03 Vegas test; 10,600 samples per sensor/action dimension and nine laps per training track; one-hour pre-evaluation on the test track. Simulator/export format: Arrival simulator on Unreal Engine 4, Gym interface, TCP/UDP interaction, CARLA integration; no interchange format stated. Code/asset status: paper states an academic release of simulator, L2R framework, and baseline implementations, and gives `github.com/learn-to-race/l2r`.

## C0122 -- Automatically Generating Content for Testing Autonomous Vehicles from User Descriptions

**Source/access.** [Official IEEE record](https://ieeexplore.ieee.org/document/11023959/) and [DOI](https://doi.org/10.1109/ICSE-NIER66352.2025.00021) (IEEE record title heading, **Automatically Generating Content for Testing Autonomous Vehicles from User Descriptions**; Limited: the official endpoint could not provide an inspectable paper body in this audit).

**Values.** Domain: autonomous-vehicle testing from user descriptions (IEEE record title heading). Vehicle: autonomous vehicle (IEEE record title heading). Course object: generated testing content, type NR (IEEE record title heading). Representation family: NR. Generator family: NR. Generation role: NR. Validity strategy: NR. Reported geometry/difficulty/diversity metrics: NR. Training/evaluation distribution counts: NR. Simulator/export format: NR. Code/asset status: NR.

## Retrieval limitations and self-check

Fifteen records have Direct evidence at the cited locators. Five have limited official/DOI/archive sources. Some publisher and archive retrievals exposed landing pages or access pages rather than the primary manuscript. The 20 assigned candidate IDs each occur exactly once, as a record heading.
