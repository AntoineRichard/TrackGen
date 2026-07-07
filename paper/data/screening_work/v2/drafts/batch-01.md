# Batch 01 adjudication dossier

Prepared 2026-07-01 against the frozen v2 protocol and sealed main-result snapshot `09f48afb2f33b6d613aafa4340a9597ae031bcd421418c52ed8ccd2994904f44`. `Snapshot rating IDs` below are the two immutable raw assignments. `Ledger trigger IDs` are the relevant `conflicts.csv` IDs; `none` means no candidate row was found there. Recommendations are draft adjudications, not a decision CSV.

### C0001 - TORCSNodateTORCSOpen: TORCS - The Open Racing Car Simulator
- Raw ratings / trigger IDs: `A-C0001-03` included / include-1 / official_documentation; `A-C0001-05` included / include-2 / official_documentation. Ledger trigger IDs: none.
- Recommendation: **included / include-1**. Exclusion reason: `NR`.
- Access and evidence: official documentation; TORCS commit `7ed3e067f1a1aae917b9bf1aa054dde7c339398d`, snapshot retrieval `2026-06-30`, pinned raw-file archive and SHA-256 `312baf44a68a4a06a76baa9ed25bee2460df3316fc89af708bb7545f470a737a`.
- Canonical source URL(s): <https://sourceforge.net/p/torcs/code/ci/7ed3e067f1a1aae917b9bf1aa054dde7c339398d/tree/torcs/torcs/src/tools/trackgen/maintrackgen.cpp>; <https://sourceforge.net/p/torcs/code/ci/7ed3e067f1a1aae917b9bf1aa054dde7c339398d/tree/torcs/torcs/doc/man/trackgen.6?format=raw>.
- Deciding locator and fact: `maintrackgen.cpp` lines 400-474 and `track.cpp` lines 2818-2845 generate a track from track data; the manual and XML identify the generator and serialized track representation.
- Protocol comparison: an inspected source-native generator emits racing-course geometry, satisfying include-1 before the representation alternative in include-2.

### C0002 - SpeedDreamsNodateSpeedDreams: Speed Dreams
- Raw ratings / trigger IDs: `A-C0002-01` included / include-1 / official_documentation; `A-C0002-05` included / include-2 / official_documentation. Ledger trigger IDs: none.
- Recommendation: **included / include-1**. Exclusion reason: `NR`.
- Access and evidence: official documentation; code commit `6080e2bca084443be7fd0f3ab4f75430633ee978`, data commit `d6754a1c0b74f607eab8abd67900b9bed0f423a6`, snapshot retrieval `2026-06-30`; sealed SHA-256 `b3c803b9a1a5c524bc12c75ecf3a829b28f640aa3b1ffc3e8c455b6d1234158f`.
- Canonical source URL(s): <https://forge.a-lec.org/speed-dreams/speed-dreams-code/commit/6080e2bca084443be7fd0f3ab4f75430633ee978>; <https://forge.a-lec.org/speed-dreams/speed-dreams-code/raw/commit/6080e2bca084443be7fd0f3ab4f75430633ee978/doc/man/sd2-trackgen.6>.
- Deciding locator and fact: `src/tools/trackgen/main.cpp` lines 99-105 and 366-398 emit a track mesh and terrain; `sd2-trackgen.6` documents the track generator.
- Protocol comparison: explicit source-native emission of racing-track geometry is include-1; include-2 is secondary under the first-applicable rule.

### C0003 - Cardamone2015TrackGenInteractive: TrackGen: An interactive track generator for TORCS and Speed-Dreams
- Raw ratings / trigger IDs: `A-C0003-05` included / include-1 / full_text; `A-C0003-06` excluded / exclude-insufficient-detail / abstract_only. Ledger trigger IDs: `X74541EDFB3BB`, `XDBA5F65BEA06` (metadata only).
- **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**: the sealed full-text rating cites the Politecnico author manuscript, but the authoritative bitstream could not be re-retrieved for this adjudication; the other sealed rating also lacked inspectable full text.
- Recommendation: **excluded / exclude-insufficient-detail**. Exclusion reason: current direct access did not provide inspectable report bytes establishing the claimed TrackGen backend.
- Access and evidence: abstract_only for this adjudication; DOI version of record `10.1016/j.asoc.2014.11.010`, retrieval `2026-07-01`. The locked included rating records author manuscript dated `2014-08-04`, retrieval `2026-06-30`, SHA-256 `0b1da243416eaf96e81121a1418be1ea006b2bebdb89ef229630020f12cc72f5`.
- Canonical source URL(s): <https://doi.org/10.1016/j.asoc.2014.11.010>; <https://re.public.polimi.it/handle/11311/984831>.
- Deciding locator and fact: locked locator is Abstract, Section 3 `TrackGen`, and Section 4 `Track-generation backend`; no current accessible copy could be inspected at those locators.
- Protocol comparison: include-1 would follow from a source-native backend, but eligibility cannot be inferred from title or a prior rating when the primary text is inaccessible; exclusion is the conservative rule pending author verification.

### C0005 - Barthet2023SolidRally: Solid-Rally-Plus
- Raw ratings / trigger IDs: `A-C0005-03` excluded / exclude-insufficient-detail / official_documentation; `A-C0005-05` excluded / exclude-out-of-scope / official_documentation. Ledger trigger IDs: none.
- **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**: the pinned README contains only the project name and the inspected tree exposes assets without technical generator documentation; that is not enough to prove or disprove a source-native generator.
- Recommendation: **excluded / exclude-insufficient-detail**. Exclusion reason: accessible repository documentation does not establish a course generator, representation, or qualifying transfer.
- Access and evidence: official documentation; commit `2d3e3af16d7c2845cbe63b620d8b4a8fd709e542`, retrieval `2026-07-01`; sealed archive SHA-256 `8828dd9d1d8b385538e119fa29b455ab558320c64df7ca82c13a20a4172a07a3`.
- Canonical source URL(s): <https://github.com/DiogoAABranco/Solid-Rally-Plus/tree/2d3e3af16d7c2845cbe63b620d8b4a8fd709e542>.
- Deciding locator and fact: `README.md` line 1 and the pinned tree identify a rally project and asset paths, not an inspectable course-generation implementation.
- Protocol comparison: the protocol forbids inference from topic or asset names; unavailable technical detail requires exclude-insufficient-detail rather than a categorical out-of-scope conclusion.

### C0008 - Ikram2023ProceduralGeneration: Procedural Generation of Complex Roundabouts for Autonomous Vehicle Testing
- Raw ratings / trigger IDs: `A-C0008-01` excluded / exclude-insufficient-detail / abstract_only; `A-C0008-05` included / include-1 / full_text. Ledger trigger IDs: `X64D080A5D5B9`, `X8BE6D9BDA923` (metadata only).
- Recommendation: **included / include-1**. Exclusion reason: `NR`.
- Access and evidence: full_text; arXiv `2303.17900v2`, retrieved `2026-07-01`, pinned PDF archive <https://arxiv.org/pdf/2303.17900v2>, sealed SHA-256 `a16724999de0aa16fbdd1ee5d178a77a1dbd7cdbf31ed7efeaa30b6c1aa518f8`.
- Canonical source URL(s): <https://arxiv.org/abs/2303.17900>; <https://doi.org/10.1109/iv55152.2023.10186533>.
- Deciding locator and fact: Abstract and Section III, `Methods for Roundabout Generation`, define a three-phase procedure that produces incident and circular roads and exports road geometry in OpenDRIVE.
- Protocol comparison: source-native generation of connected road corridors is an explicit transferable-adjacent mapping to robot courses, so include-1 overrides the abstract-only exclusion.

### C0012 - Alyaseri2024ComparativeAnalysis: Comparative Analysis of Metaheuristic Algorithms for Procedural Race Track Generation in Games
- Raw ratings / trigger IDs: `A-C0012-01` excluded / exclude-insufficient-detail / abstract_only; `A-C0012-04` excluded / exclude-insufficient-detail / abstract_only. Ledger trigger IDs: `X87967FA77E43`, `XF8DBBBF6F664`, `X9D78AE994FCE`, `X1ECEB618C254`, `X49B0671427FE` (metadata only).
- **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**: DOI and publisher sources did not yield an inspectable complete report.
- Recommendation: **excluded / exclude-insufficient-detail**. Exclusion reason: the accessible abstract describes metaheuristic racetrack generation, but no primary method, representation, or evaluation could be inspected.
- Access and evidence: abstract_only; version of record DOI `10.4018/ijamc.350330`, retrieval `2026-07-01`; no archive or digest obtained.
- Canonical source URL(s): <https://doi.org/10.4018/ijamc.350330>; <https://www.igi-global.com/article/comparative-analysis-of-metaheuristic-algorithms-for-procedural-race-track-generation-in-games/350330>.
- Deciding locator and fact: publisher/Crossref abstract record only; full text remained unavailable.
- Protocol comparison: abstract-only evidence cannot support inclusion, even where the abstract is topically promising.

### C0013 - doNascimento2021ProceduralGeneration: Procedural Generation of Isometric Racetracks Using Chain Code for Racing Games
- Raw ratings / trigger IDs: `A-C0013-04` excluded / exclude-insufficient-detail / abstract_only; `A-C0013-06` included / include-1 / full_text. Ledger trigger IDs: `XB9B490D43F0D`, `X2B5E9EC4EECC`, `X5CBF6494EC6D`, `X9C13F0A6D22E`, `X9B9BB96A34A1`, `XB7CB28F913D4` (metadata only).
- Recommendation: **included / include-1**. Exclusion reason: `NR`.
- Access and evidence: full_text; SBGames 2021 proceedings PDF, retrieved `2026-07-01`, pinned archive <https://www.sbgames.org/proceedings2021/ComputacaoFull/217347.pdf>, sealed SHA-256 `d168baea43c3fc567c71cf0036eeecb85c0fc90f4fee3678f4adc8125b5d00ed`.
- Canonical source URL(s): <https://www.sbgames.org/proceedings2021/ComputacaoFull/217347.pdf>; <https://doi.org/10.1109/sbgames54170.2021.00025>.
- Deciding locator and fact: pp. 136-139, especially Sections I and D-E, define a chain-code procedure that produces new racetrack circuits and instantiates them in Unity.
- Protocol comparison: generated racing-game centerlines and tiles map directly to simulated racing-robot course geometry; direct synthesis satisfies include-1.

### C0026 - Song2021AutonomousDrone: Autonomous Drone Racing with Deep Reinforcement Learning
- Raw ratings / trigger IDs: `A-C0026-01` excluded / exclude-fixed-racing-line / abstract_only; `A-C0026-06` included / include-1 / full_text. Ledger trigger IDs: `X226F3AF07C12`, `XE608AE4C5220`, `XD2665AB6934D`, `X67679029475A` (metadata only).
- Recommendation: **included / include-1**. Exclusion reason: `NR`.
- Access and evidence: full_text; IROS 2021 author manuscript, retrieved `2026-07-01`, source URL <https://rpg.ifi.uzh.ch/docs/IROS21_Yunlong.pdf>, sealed SHA-256 `cf39bc90b01921b689ba911a665636972366fc4cf55da7dbfb6d1b995132faa4`.
- Canonical source URL(s): <https://rpg.ifi.uzh.ch/docs/IROS21_Yunlong.pdf>; <https://doi.org/10.1109/iros51168.2021.9636053>.
- Deciding locator and fact: Section III-B, `Random Track Curriculum`, pp. 3-4 defines a generator that concatenates randomly generated gate primitives with bounded relative poses and gate counts, producing tracks of arbitrary complexity and length.
- Protocol comparison: a gate sequence is a course under the protocol; source-native randomized gate-course synthesis is include-1, not fixed-course racing-line work.

### C0028 - NVIDIA2020IsaacGym: Isaac Gym
- Raw ratings / trigger IDs: `A-C0028-01` and `A-C0028-02` excluded / exclude-out-of-scope / official_documentation. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-out-of-scope**. Exclusion reason: official Isaac Gym documentation identifies a general GPU robot-learning simulator, not a course generator, course representation, course benchmark, or boundary transfer.
- Access and evidence: official documentation; `unversioned@2026-07-01`; sealed documentation SHA-256 `4ceada825e4e5cfd4c9f65ef8dfbdd41c9beaaa56a9dfd46d2868e384fdcf3bf`.
- Canonical source URL(s): <https://developer.nvidia.com/isaac-gym>.
- Deciding locator and fact: `About Isaac Gym` and `Getting Started` describe simulation and learning infrastructure rather than spatial-course generation.
- Protocol comparison: a general simulator is not include-2 without a source-native generated-course interface.

### C0030 - PolyphonyDigital1997GranTurismo: Gran Turismo
- Raw ratings / trigger IDs: `A-C0030-01` excluded / exclude-out-of-scope / official_documentation; `A-C0030-06` excluded / exclude-insufficient-detail / official_documentation. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-out-of-scope**. Exclusion reason: the official report is a product catalogue and supplies no technical course-generation or survey-transfer contribution.
- Access and evidence: official documentation; `unversioned@2026-07-01`; sealed SHA-256 `1fa617291589e34f2ecf94904a0ec0f7ca11343fbfd0cf2c5a376b5e4b9eb407`.
- Canonical source URL(s): <https://www.polyphony.co.jp/products/>.
- Deciding locator and fact: product-series and Gran Turismo (1997) entries identify a game release, not a technical artifact for course generation.
- Protocol comparison: product existence and topical relevance do not establish a source-native contribution or boundary transfer.

### C0034 - PolyphonyDigital2017GranTurismo: Gran Turismo Sport
- Raw ratings / trigger IDs: `A-C0034-01` and `A-C0034-05` excluded / exclude-out-of-scope / official_documentation. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-out-of-scope**. Exclusion reason: official feature material lists fixed locations/layouts and no generator or parameterized-course artifact.
- Access and evidence: official documentation; product page state `2017`, retrieved `2026-07-01`; sealed SHA-256 `a23daee34ffbd4ee4ae9795a1f40e8eb8ad6098b4b13d804e0d7ace6e042afcf`.
- Canonical source URL(s): <https://www.gran-turismo.com/us/products/gtsport/>.
- Deciding locator and fact: `Features` / `Tracks` lists 18 locations and 54 layouts as supplied content.
- Protocol comparison: documented fixed game content is neither source-native course generation nor a named transferable boundary.

### C0035 - PolyphonyDigital2022GranTurismo: Gran Turismo 7
- Raw ratings / trigger IDs: `A-C0035-02` and `A-C0035-06` excluded / exclude-out-of-scope / official_documentation. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-out-of-scope**. Exclusion reason: official product material describes supplied fixed tracks and gameplay, with no qualifying technical contribution.
- Access and evidence: official documentation; `Gran Turismo 7 Spec III product state (2025-12-04)`, retrieved `2026-07-01`; sealed SHA-256 `62e74e27f7d9316a0d8142b2f8a09f950bae58515535c2d7f62dc1177bf9410a`.
- Canonical source URL(s): <https://www.gran-turismo.com/us/products/gt7/>.
- Deciding locator and fact: `Tracks from 41 locations with 121 layouts` / `World Circuits` describes fixed layouts.
- Protocol comparison: fixed-course product material does not satisfy any inclusion criterion or name a boundary transfer.

### C0036 - Theodosis2011GeneratingRacing: Generating a Racing Line for an Autonomous Racecar Using Professional Driving Techniques
- Raw ratings / trigger IDs: `A-C0036-03` excluded / exclude-insufficient-detail / abstract_only; `A-C0036-04` excluded / exclude-fixed-racing-line / abstract_only. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-fixed-racing-line**. Exclusion reason: the accessible abstract compares clothoid/arc racing-line maneuvers on an existing track, not course geometry.
- Access and evidence: abstract_only; version of record DOI `10.1115/dscc2011-6097`, retrieval `2026-07-01`; no archive or digest obtained.
- Canonical source URL(s): <https://doi.org/10.1115/dscc2011-6097>; <https://asmedigitalcollection.asme.org/DSCC/proceedings/DSCC2011/54761/853/353667>.
- Deciding locator and fact: Crossref abstract field; its stated scope is racing-line maneuver generation for a given track.
- Protocol comparison: the abstract directly establishes the fixed-racing-line exclusion, and identifies no named boundary transfer.

### C0037 - Braghin2008RaceDriver: Race driver model
- Raw ratings / trigger IDs: `A-C0037-01` and `A-C0037-02` excluded / exclude-insufficient-detail / abstract_only. Ledger trigger IDs: none.
- **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**: only bibliographic records were accessible; no abstract or technical text could establish either scope or a boundary transfer.
- Recommendation: **excluded / exclude-insufficient-detail**. Exclusion reason: the available DOI/publisher metadata has no material technical evidence for a qualifying course contribution.
- Access and evidence: abstract_only; version of record DOI `10.1016/j.compstruc.2007.04.028`, retrieval `2026-07-01`; no archive or digest obtained.
- Canonical source URL(s): <https://doi.org/10.1016/j.compstruc.2007.04.028>; <https://www.sciencedirect.com/science/article/pii/S0045794908000163>.
- Deciding locator and fact: DOI and publisher bibliographic fields only; substantive report content was inaccessible.
- Protocol comparison: title and metadata cannot establish inclusion, boundary, or a more specific exclusion.

### C0038 - Heilmeier2020MinimumCurvature: Minimum curvature trajectory planning and control for an autonomous race car
- Raw ratings / trigger IDs: `A-C0038-04` boundary / boundary / official_documentation; `A-C0038-05` excluded / exclude-fixed-racing-line / abstract_only. Ledger trigger IDs: `X31B07748CBE8`, `X7395DB32A4A7` (metadata only).
- Recommendation: **boundary / boundary**. Exclusion reason: `NR`.
- Access and evidence: official documentation; companion repository commit `afa0aa710adb9fdd23fa3d6fbdd6a13664253616`, retrieved `2026-07-01`, pinned archive <https://github.com/TUMFTM/laptime-simulation/blob/afa0aa710adb9fdd23fa3d6fbdd6a13664253616/README.md>, sealed SHA-256 `ec5c43853c2c9d58401a7cdc3146cbc4ba8e691065515b13faaf1fcc0dcf0872`.
- Canonical source URL(s): <https://github.com/TUMFTM/laptime-simulation/blob/afa0aa710adb9fdd23fa3d6fbdd6a13664253616/README.md>; <https://doi.org/10.1080/00423114.2019.1631455>.
- Deciding locator and fact: README lines 42-64 specifies a CSV track input `[x, y, w_tr_right, w_tr_left]`, smooth curvature, and reports lap time and energy; lines 71-75 identify the assigned paper.
- Protocol comparison: it remains fixed-course work, but directly supports the named survey boundary transfer of centerline-plus-width representation and dynamics-grounded reporting.

### C0039 - Muhlmeier2002OptimizationDriving: Optimization of the Driving Line on a Race Track
- Raw ratings / trigger IDs: `A-C0039-01` and `A-C0039-04` excluded / exclude-fixed-racing-line / abstract_only. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-fixed-racing-line**. Exclusion reason: the official abstract optimizes a driving line among candidate trajectories on a given race track and names no survey boundary transfer.
- Access and evidence: abstract_only; SAE paper `2002-01-3339`, retrieved `2026-07-01`; no archive or digest obtained.
- Canonical source URL(s): <https://doi.org/10.4271/2002-01-3339>; <https://saemobilus.sae.org/papers/optimization-driving-line-a-race-track-2002-01-3339>.
- Deciding locator and fact: SAE landing-page abstract, sentences 1-6.
- Protocol comparison: optimizing traversal inside an input track is the protocol's fixed-racing-line exclusion.

### C0040 - Cardamone2010SearchingOptimal: Searching for the optimal racing line using genetic algorithms
- Raw ratings / trigger IDs: `A-C0040-02` and `A-C0040-06` excluded / exclude-fixed-racing-line / abstract_only. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-fixed-racing-line**. Exclusion reason: the accessible IEEE/OpenAlex record describes genetic optimization of line length and curvature within 11 fixed TORCS tracks.
- Access and evidence: abstract_only; version of record DOI `10.1109/itw.2010.5593330`, retrieved `2026-07-01`; sealed digest `ae1e1d14292beb73d2a3a71927e9081557ac7f2c40b0d9cb417fa85ec585357d` for the inspected IEEE record.
- Canonical source URL(s): <https://doi.org/10.1109/itw.2010.5593330>; <https://ieeexplore.ieee.org/document/5593330/>.
- Deciding locator and fact: IEEE abstract paragraphs 1-2 and partial Section I; all reported tracks are supplied TORCS tracks.
- Protocol comparison: source-native trajectory optimization does not generate or parameterize a course.

### C0041 - Christ2021TimeOptimal: Time-optimal trajectory planning for a race car considering variable tyre-road friction coefficients
- Raw ratings / trigger IDs: `A-C0041-03` and `A-C0041-05` excluded / exclude-fixed-racing-line / abstract_only. Ledger trigger IDs: `XB11C4373451B`, `XA7B4ABC09DB0` (metadata only).
- Recommendation: **excluded / exclude-fixed-racing-line**. Exclusion reason: the accessible abstract limits the contribution to time-optimal trajectories on an input racetrack and friction map.
- Access and evidence: abstract_only; version of record DOI `10.1080/00423114.2019.1704804`, retrieved `2026-07-01`; no archive or digest obtained.
- Canonical source URL(s): <https://doi.org/10.1080/00423114.2019.1704804>; <https://www.tandfonline.com/doi/full/10.1080/00423114.2019.1704804>.
- Deciding locator and fact: accessible abstract paragraph beginning `This paper shows the planning of time-optimal trajectories`.
- Protocol comparison: the abstract directly establishes planning in a supplied corridor, with no named boundary transfer.

### C0042 - Jain2020ComputingRacing: Computing the racing line using Bayesian optimization
- Raw ratings / trigger IDs: `A-C0042-02` excluded / exclude-fixed-racing-line / full_text; `A-C0042-05` boundary / boundary / full_text. Ledger trigger IDs: none.
- Recommendation: **boundary / boundary**. Exclusion reason: `NR`.
- Access and evidence: full_text; arXiv `2002.04794v1`, retrieved `2026-07-01`, pinned archive <https://arxiv.org/pdf/2002.04794v1>, sealed SHA-256 `53fad9b164cd7dc06b71cf24d9ef6987b5f108dbdcb0bae6aeda8c4c9452e355`.
- Canonical source URL(s): <https://arxiv.org/abs/2002.04794>; <https://doi.org/10.1109/cdc42340.2020.9304147>.
- Deciding locator and fact: Section IV-B defines minimum traversal time on a fixed trajectory under friction-circle and actuator constraints; the track parameterization uses centerline and width.
- Protocol comparison: no inclusion applies, but the fixed-course feasibility/lap-time calculation is a named dynamics-grounded boundary metric and representation transfer.

### C0043 - Kapania2020LearningRacetrack: Learning at the Racetrack: Data-Driven Methods to Improve Racing Performance Over Multiple Laps
- Raw ratings / trigger IDs: `A-C0043-02` excluded / exclude-insufficient-detail / abstract_only; `A-C0043-06` excluded / exclude-fixed-racing-line / abstract_only. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-fixed-racing-line**. Exclusion reason: the accessible IEEE abstract states that the method improves tracking and modifies a desired racing trajectory over repeated laps on a fixed racetrack.
- Access and evidence: abstract_only; version of record DOI `10.1109/tvt.2020.2998065`, retrieved `2026-07-01`; sealed digest `a3eb3a1a9a7228bf9800b0e27ca9f31ca36e76f8f78f46982c48f8631f3a5a2a` for the inspected IEEE record.
- Canonical source URL(s): <https://doi.org/10.1109/tvt.2020.2998065>; <https://ieeexplore.ieee.org/document/9102440/>.
- Deciding locator and fact: IEEE abstract and partial Section I; the object changed is the desired trajectory, not course geometry.
- Protocol comparison: the available source directly supports the fixed-racing-line exclusion and no specific boundary transfer is proposed.

### C0044 - Bonab2019OptimizationBased: Optimization-based Path Planning for an Autonomous Vehicle in a Racing Track
- Raw ratings / trigger IDs: `A-C0044-01` and `A-C0044-06` excluded / exclude-fixed-racing-line / abstract_only. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-fixed-racing-line**. Exclusion reason: the official IEEE record describes path planning in a supplied route on the fixed Suzuka circuit.
- Access and evidence: abstract_only; version of record DOI `10.1109/iecon.2019.8926856`, retrieved `2026-07-01`; sealed digest `04e7142a33544ae608c825b585cc69493141f1e7cbd8d45a62330ee2e8691e44`.
- Canonical source URL(s): <https://doi.org/10.1109/iecon.2019.8926856>; <https://ieeexplore.ieee.org/document/8926856/>.
- Deciding locator and fact: IEEE abstract and partial Section I identify a fixed Suzuka-circuit horizon.
- Protocol comparison: a trajectory planner within a supplied circuit does not synthesize a course or provide a named boundary transfer.

### C0045 - CattoNodateBox2D: Box2D
- Raw ratings / trigger IDs: `A-C0045-02` and `A-C0045-05` excluded / exclude-out-of-scope / official_documentation. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-out-of-scope**. Exclusion reason: the pinned README defines Box2D as a general 2D physics engine and lists generic collision and rigid-body features only.
- Access and evidence: official documentation; commit `56edae79f2949d86142b03450d5d60f63bcf5a6f`, retrieved `2026-07-01`, pinned archive <https://github.com/erincatto/box2d/archive/56edae79f2949d86142b03450d5d60f63bcf5a6f.tar.gz>, sealed SHA-256 `fa412f9d70d34a8a94618db71406817da23cd77ada45ce5d748a02db6363073d`.
- Canonical source URL(s): <https://github.com/erincatto/box2d/blob/56edae79f2949d86142b03450d5d60f63bcf5a6f/README.md>.
- Deciding locator and fact: README lines 1-46, including `Box2D is a 2D physics engine for games` and generic feature lists.
- Protocol comparison: physics capability alone is not a course contribution or a qualifying transfer.

### C0048 - UnityTechnologiesNodateUnity: Unity
- Raw ratings / trigger IDs: `A-C0048-02` and `A-C0048-05` excluded / exclude-out-of-scope / official_documentation. Ledger trigger IDs: none.
- Recommendation: **excluded / exclude-out-of-scope**. Exclusion reason: the Unity User Manual is general engine documentation, not a source-native course-generation, interchange, dataset, or metric artifact.
- Access and evidence: official documentation; Unity 6.5 (`6000.5`) User Manual, retrieved `2026-07-01`; sealed SHA-256 `8d8aa30b87abbef0e799ae1dbccc762ffff18401ec4bff0290d8d58246ba56ed`.
- Canonical source URL(s): <https://docs.unity3d.com/Manual/UnityManual.html>.
- Deciding locator and fact: manual introduction and `Highlights of Unity 6` identify a general game engine and tools.
- Protocol comparison: generic infrastructure does not meet include-2 absent a source-native generated-course interface.

### C0053 - EsriNodateArcGISCityEngine: ArcGIS CityEngine
- Raw ratings / trigger IDs: `A-C0053-05` excluded / exclude-insufficient-detail / official_documentation; `A-C0053-06` included / include-1 / official_documentation. Ledger trigger IDs: none.
- Recommendation: **included / include-1**. Exclusion reason: `NR`.
- Access and evidence: official documentation `ArcGIS CityEngine 2023.0`, last modified `2023-06-12`, retrieved `2026-07-01`, pinned archive <https://doc.arcgis.com/en/cityengine/2023.0/help/help-grow-a-street.htm>, sealed SHA-256 `a463c31114b32d34ab04e62b038d9b05630d40d54a1c1802c885e5cb58106fbc`.
- Canonical source URL(s): <https://doc.arcgis.com/en/cityengine/2023.0/help/help-grow-a-street.htm>; <https://www.esri.com/en-us/arcgis/products/arcgis-cityengine/overview>.
- Deciding locator and fact: `Generate street networks` states that Grow Streets creates and extends networks with organic/raster/radial patterns; it exposes street count, length, bend, topology, terrain/obstacle, and width constraints.
- Protocol comparison: this is a source-native adjacent-domain generator of connected, width-bearing road geometry. Those structures map directly to robot-course topology and corridors, satisfying include-1.

### C0054 - Chen2008InteractiveProcedural: Interactive procedural street modeling
- Raw ratings / trigger IDs: `A-C0054-02` included / include-1 / full_text; `A-C0054-05` excluded / exclude-insufficient-detail / abstract_only. Ledger trigger IDs: none.
- **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**: the author-manuscript URL did not yield a parseable complete PDF in this adjudication, so the sealed inclusion locator could not be independently re-inspected.
- Recommendation: **excluded / exclude-insufficient-detail**. Exclusion reason: current accessible material did not provide inspectable primary text establishing street-graph synthesis and the transferable mapping.
- Access and evidence: abstract_only for this adjudication; DOI version of record `10.1145/1360612.1360702`, retrieval `2026-07-01`. The locked included rating records author manuscript `(2008)`, retrieval `2026-06-30`, SHA-256 `25438a9fef50b49a15f17de40405f68bfbb25f354517d8c70f5c97ab2ac3bd26`.
- Canonical source URL(s): <https://doi.org/10.1145/1360612.1360702>; <https://dl.acm.org/doi/10.1145/1360612.1360702>; <https://web.engr.oregonstate.edu/~zhange/images/street_sig08.pdf>.
- Deciding locator and fact: locked inclusion locator is Abstract, Section 3 `Tensor-Field Design`, and Section 4 `Street-Graph Editing`; the current PDF transfer was malformed and could not be inspected.
- Protocol comparison: the reported street-graph mapping would satisfy include-1, but current inaccessible evidence requires the conservative insufficient-detail recommendation pending accountable-author verification.

### C0055 - Gonzalez1999GlobalCurvature: Global Curvature, Thickness, and the Ideal Shapes of Knots
- Raw ratings / trigger IDs: `A-C0055-01` and `A-C0055-02` excluded / exclude-out-of-scope / full_text. Ledger trigger IDs: `X489BA96ED744`, `X75433D41B59F` (metadata only).
- Recommendation: **excluded / exclude-out-of-scope**. Exclusion reason: the full primary article develops global curvature, thickness, and ideal knot geometry, not navigation-course generation or representation.
- Access and evidence: full_text; PNAS version of record DOI `10.1073/pnas.96.9.4769`, retrieved `2026-07-01`, public archive <https://pmc.ncbi.nlm.nih.gov/articles/PMC21766/>, sealed SHA-256 `3397de1d1d9ed460af1c5e0c94783d05a25d72a24ac25e280080b8a306af1ea6`.
- Canonical source URL(s): <https://doi.org/10.1073/pnas.96.9.4769>; <https://pmc.ncbi.nlm.nih.gov/articles/PMC21766/>.
- Deciding locator and fact: Abstract and sections `Global Curvature and Thickness` and `Ideal Knots` define geometric properties of curves and knots in a DNA context.
- Protocol comparison: the report supplies neither a navigable route/level artifact nor a source-native course metric, feasibility test, or survey-gap contribution.
