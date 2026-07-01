# Batch 03 adjudication dossier

Scope: `C0106;C0110;C0113;C0114;C0115;C0116;C0118;C0119;C0122;C0123;C0124;C0125;C0126;C0127;C0128;C0129;C0130;C0132;C0134;C0135;C0137;C0139;C0141;C0143;C0144;C0150`.

This is an adjudication draft, not a result CSV or a projection. Raw sealed ratings are transcribed as `assignment_id: status / criterion / access`. “NEEDS_ACCOUNTABLE_AUTHOR_REVIEW” means that the public primary evidence needed for a fully accountable finding could not be retrieved; the stated recommendation is the conservative protocol outcome.

## C0106

- **candidate_id:** `C0106`
- **cite_key/title:** `Zhang2025VoyagePlanning` — *A voyage planning framework for energy performance analysis of autonomous inland waterway vessels*.
- **Raw sealed ratings / trigger IDs:** `A-C0106-02: excluded / exclude-insufficient-detail / abstract_only`; `A-C0106-06: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** The accessible publisher record describes path following, energy prediction, river hydraulics, and speed optimization on supplied waterways; no inspected report establishes source-native generation or selection of route geometry.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1016/j.energy.2025.137906 ; https://www.sciencedirect.com/science/article/pii/S0360544225035480
- **Evidence version / retrieval / pin or digest:** version of record DOI; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not yield complete text; no archive pin.
- **Deciding locator and fact:** ScienceDirect Highlights and Abstract: the reported framework analyzes a supplied inland-waterway voyage rather than documenting emitted route coordinates or a route generator.
- **Comparison rationale:** Both snapshots agree. The protocol prohibits `included` or `boundary` on `abstract_only`; absent direct material evidence for `include-1`, the conservative outcome is `exclude-insufficient-detail`.

## C0110

- **candidate_id:** `C0110`
- **cite_key/title:** `Arcaini2024CRAGCombinatorial` — *CRAG – a combinatorial testing-based generator of road geometries for ADS testing*.
- **Raw sealed ratings / trigger IDs:** `A-C0110-02: excluded / exclude-insufficient-detail / abstract_only`; `A-C0110-05: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** The title indicates road-geometry generation, but no public full report or authoritative artifact was retrievable to verify representation, constraints, outputs, or source-native contribution.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1016/j.scico.2024.103171 ; https://www.sciencedirect.com/science/article/pii/S0167642324000947
- **Evidence version / retrieval / pin or digest:** DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` found metadata only; no archive pin.
- **Deciding locator and fact:** Elsevier/DOI metadata record: no inspectable abstract, methods, or companion implementation was available.
- **Comparison rationale:** Both snapshots agree. Title-only relevance is expressly not material evidence, so the protocol requires the conservative insufficient-detail exclusion.

## C0113

- **candidate_id:** `C0113`
- **cite_key/title:** `Ashlock2006EvolvingDiverse` — *Evolving A Diverse Collection of Robot Path Planning Problems*.
- **Raw sealed ratings / trigger IDs:** `A-C0113-04: excluded / exclude-insufficient-detail / abstract_only`; `A-C0113-05: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** Indexed material says grid path-planning problems are evolved, but no full report was available to verify a transferable route representation or source-native generator.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1109/cec.2006.1688530 ; https://ieeexplore.ieee.org/document/1688530/
- **Evidence version / retrieval / pin or digest:** IEEE DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not recover complete text; no archive pin.
- **Deciding locator and fact:** IEEE/OpenAlex abstract record: generated grid path-planning instances are mentioned without inspectable representation, generation process, or evaluation.
- **Comparison rationale:** Both snapshots agree. A possible adjacent-domain mapping cannot replace the source-native material evidence required by `include-1`.

## C0114

- **candidate_id:** `C0114`
- **cite_key/title:** `Wang2014RacingTracks` — *Racing tracks improvisation*.
- **Raw sealed ratings / trigger IDs:** `A-C0114-03: excluded / exclude-insufficient-detail / abstract_only`; `A-C0114-06: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** Available bibliographic/partial material says music-improvisation methods generate tracks, but not enough of the representation or algorithm was inspectable to support a survey claim.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1109/cig.2014.6932899 ; https://ieeexplore.ieee.org/document/6932899/
- **Evidence version / retrieval / pin or digest:** IEEE DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not recover full text; no archive pin.
- **Deciding locator and fact:** IEEE record Abstract and partial Section I: the accessible text does not expose a verifiable track encoding, generator, or released artifact.
- **Comparison rationale:** Both snapshots agree. The protocol rejects title, topic, and incomplete abstract inference as material evidence.

## C0115

- **candidate_id:** `C0115`
- **cite_key/title:** `Georgiou2016PersonalisedTrack` — *Personalised track design in car racing games*.
- **Raw sealed ratings / trigger IDs:** `A-C0115-03: excluded / exclude-insufficient-detail / abstract_only`; `A-C0115-04: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** The abstract reports personalized generated tracks, but the primary methods, evidence, and generated-track artifact were not publicly inspectable.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1109/cig.2016.7860435 ; http://hdl.handle.net/10044/1/39560
- **Evidence version / retrieval / pin or digest:** IEEE DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not obtain the report; no archive pin.
- **Deciding locator and fact:** IEEE/OpenAlex abstract record and Imperial handle: neither supplied a complete inspectable primary report.
- **Comparison rationale:** Both snapshots agree. The likely relevance of personalized tracks cannot overcome the full-text evidence requirement.

## C0116

- **candidate_id:** `C0116`
- **cite_key/title:** `NR` — *Controllable Procedural Generation of Race Track Surroundings for Iterative Level Design*.
- **Raw sealed ratings / trigger IDs:** `A-C0116-02: excluded / exclude-appearance-dynamics / abstract_only`; `A-C0116-05: excluded / exclude-appearance-dynamics / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-appearance-dynamics`.
- **Exclusion reason:** The accessible abstract explicitly identifies generated forests, fields, cities, mountains, assets, and vegetation as track surroundings, not generated track geometry.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1109/cog64752.2025.11114175 ; https://ieeexplore.ieee.org/document/11114175/
- **Evidence version / retrieval / pin or digest:** DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not obtain full text; no archive pin.
- **Deciding locator and fact:** IEEE/OpenAlex abstract: generation is expressly for surrounding environmental content.
- **Comparison rationale:** Both snapshots agree. The protocol makes appearance-only variation insufficient; although the abstract supports this specific exclusion, the unavailable primary report merits accountable-author confirmation.

## C0118

- **candidate_id:** `C0118`
- **cite_key/title:** `Sotiropoulos2016VirtualWorlds` — *Virtual Worlds for Testing Robot Navigation: A Study on the Difficulty Level*.
- **Raw sealed ratings / trigger IDs:** `A-C0118-01: excluded / exclude-insufficient-detail / abstract_only`; `A-C0118-03: included / include-1 / full_text`.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://doi.org/10.1109/edcc.2016.14 ; https://hal.science/hal-01328909v1/file/EDCC.pdf
- **Evidence version / retrieval / pin or digest:** `HAL:hal-01328909v1`; retrieved `2026-07-01`; archive https://hal.science/hal-01328909v1 ; SHA-256 `7ae4753fdd53047b4c572a60ab213a9d9805da5c4ec1ab72fa9e16f76a3eba69`.
- **Deciding locator and fact:** PDF Abstract; Section IV, “World and Mission Generation”; Section V: the framework procedurally generates 3D worlds and a navigation mission with starting and destination positions, parameterized to control generated-map difficulty.
- **Comparison rationale:** The full public report resolves the abstract-only rating. The generated map plus source-defined start-goal mission is a transferable connected navigation constraint, so direct generation satisfies `include-1` and has precedence.

## C0119

- **candidate_id:** `C0119`
- **cite_key/title:** `Herman2021LearnRace` — *Learn-To-Race: A Multimodal Control Environment for Autonomous Racing*.
- **Raw sealed ratings / trigger IDs:** `A-C0119-01: included / include-1 / full_text`; `A-C0119-02: boundary / boundary / full_text`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text` in both sealed ratings; direct CVF re-retrieval timed out on `2026-07-01`.
- **Canonical source URL(s):** https://doi.org/10.1109/iccv48922.2021.00965 ; https://openaccess.thecvf.com/content/ICCV2021/html/Herman_Learn-To-Race_A_Multimodal_Control_Environment_for_Autonomous_Racing_ICCV_2021_paper.html ; https://openaccess.thecvf.com/content/ICCV2021/papers/Herman_Learn-To-Race_A_Multimodal_Control_Environment_for_Autonomous_Racing_ICCV_2021_paper.pdf
- **Evidence version / retrieval / pin or digest:** ICCV 2021 proceedings version; sealed retrieval `2026-06-30`; sealed PDF SHA-256 `fe6540c38f19aedc95db9f8992cdc750b3a017e2dcb3796db3ed6c641e765fde`.
- **Deciding locator and fact:** sealed full-text locator: pp. 3-4, Section 3.2 “Track Generation and Custom Track Construction,” Figure 2; it describes construction and export of custom racing-track geometry.
- **Comparison rationale:** The `include-1` rating identifies source-native custom-track construction, which takes precedence over boundary. Accountable-author confirmation is needed solely because the official primary endpoint could not be re-fetched in this adjudication session.

## C0122

- **candidate_id:** `C0122`
- **cite_key/title:** `Steininger2025AutomaticallyGenerating` — *Automatically Generating Content for Testing Autonomous Vehicles from User Descriptions*.
- **Raw sealed ratings / trigger IDs:** `A-C0122-03: excluded / exclude-insufficient-detail / abstract_only`; `A-C0122-05: included / include-1 / official_documentation`.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://doi.org/10.1109/icse-nier66352.2025.00021 ; https://github.com/Stoneymon/RoadGPT/archive/86d34da2a2ce1b368639c728b59628d0babd42a2.tar.gz
- **Evidence version / retrieval / pin or digest:** commit `86d34da2a2ce1b368639c728b59628d0babd42a2`; retrieved `2026-07-01`; pinned archive URL above; SHA-256 `b10e30f323e11b0a8f350678eb0653833d321c22f57b79646fbd78cf3a412c18`.
- **Deciding locator and fact:** `run_roadgpt.py` lines 136-187 and `self_driving/beamng_road_imagery.py` lines 7-40: a natural-language road description is passed to `RoadGenerator`, which produces middle, left, and right road points for BeamNG roads.
- **Comparison rationale:** The pinned official artifact supplies the missing material evidence: it emits explicit road coordinates and corridor boundaries. That directly satisfies `include-1`, superseding the abstract-only exclusion.

## C0123

- **candidate_id:** `C0123`
- **cite_key/title:** `Jahangirova2021QualityMetrics` — *Quality Metrics and Oracles for Autonomous Vehicles Testing*.
- **Raw sealed ratings / trigger IDs:** `A-C0123-03: excluded / exclude-insufficient-detail / abstract_only`; `A-C0123-04: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** No accessible full report establishes that the reported metrics/oracles are applied across generated or parameterized courses, as `include-3` requires.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1109/icst49551.2021.00030 ; https://ieeexplore.ieee.org/document/9438556/
- **Evidence version / retrieval / pin or digest:** IEEE DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not obtain full text; no archive pin.
- **Deciding locator and fact:** IEEE/OpenAlex record: no report body was available to verify metric application over a generated-course distribution.
- **Comparison rationale:** Both snapshots agree. A metric title and general AV-testing context do not establish `include-3` under the protocol.

## C0124

- **candidate_id:** `C0124`
- **cite_key/title:** `NR` — *AV-FUZZER: Finding Safety Violations in Autonomous Driving Systems*.
- **Raw sealed ratings / trigger IDs:** `A-C0124-01: excluded / exclude-traffic-only / abstract_only`; `A-C0124-03: excluded / exclude-traffic-only / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-traffic-only`.
- **Exclusion reason:** The accessible abstract states that AV-FUZZER perturbs traffic-participant maneuvers and trajectory parameters in an existing road environment, rather than road/course geometry.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1109/issre5003.2020.00012 ; https://ieeexplore.ieee.org/document/9251068/
- **Evidence version / retrieval / pin or digest:** IEEE DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not obtain full text; no archive pin.
- **Deciding locator and fact:** IEEE/Semantic Scholar abstract locator in the sealed ratings: maneuvers and participant trajectories are mutated, with no road modification.
- **Comparison rationale:** Both snapshots agree. The abstract directly activates the traffic-only exclusion, but the inaccessible primary report remains an accountable-author review item.

## C0125

- **candidate_id:** `C0125`
- **cite_key/title:** `Tang2023EvoScenarioIntegrating` — *EvoScenario: Integrating Road Structures into Critical Scenario Generation for Autonomous Driving System Testing*.
- **Raw sealed ratings / trigger IDs:** `A-C0125-03: excluded / exclude-insufficient-detail / abstract_only`; `A-C0125-06: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** The abstract mentions sequential road-segment generation, but no public report or official implementation exposes representation, operators, constraints, or evaluation.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1109/issre59848.2023.00054 ; https://ieeexplore.ieee.org/document/10301222
- **Evidence version / retrieval / pin or digest:** IEEE DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not obtain complete evidence; no archive pin.
- **Deciding locator and fact:** IEEE abstract and document record: road segments are named but their source-native technical treatment cannot be inspected.
- **Comparison rationale:** Both snapshots agree. Under the full-text rule, an abstract-only allusion to road generation cannot support `include-1`.

## C0126

- **candidate_id:** `C0126`
- **cite_key/title:** `Poggenhans2018Lanelet2High` — *Lanelet2: A high-definition map framework for the future of automated driving*.
- **Raw sealed ratings / trigger IDs:** `A-C0126-01: included / include-2 / official_documentation`; `A-C0126-03: included / include-1 / full_text_and_supplement`.
- **Recommendation:** `included` / `include-2`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text_and_supplement`.
- **Canonical source URL(s):** https://doi.org/10.1109/itsc.2018.8569929 ; https://github.com/fzi-forschungszentrum-informatik/Lanelet2/archive/ae39c8d673264afac2339c4f0252df53a7ba82dd.tar.gz
- **Evidence version / retrieval / pin or digest:** version of record plus commit `ae39c8d673264afac2339c4f0252df53a7ba82dd`; retrieved `2026-07-01`; pinned archive URL above; SHA-256 `3f36d52f216609156189c06c2ce3967eea08153162905dd9221110559fd4f8d9`.
- **Deciding locator and fact:** `README.md` lines 18-28, 41-47, and 174-183: Lanelet2 defines map primitives, routing, OSM read/write, and map validation for automated-driving corridor data.
- **Comparison rationale:** Both ratings support inclusion. The artifact’s essential contribution is a reusable representation, interface, validation, and interchange framework, so `include-2` is more precise than `include-1`.

## C0127

- **candidate_id:** `C0127`
- **cite_key/title:** `NR` — *Scenario Factory: Creating Safety-Critical Traffic Scenarios for Automated Vehicles*.
- **Raw sealed ratings / trigger IDs:** `A-C0127-01: excluded / exclude-traffic-only / full_text`; `A-C0127-06: included / include-1 / full_text`.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://doi.org/10.1109/itsc45102.2020.9294629 ; https://mediatum.ub.tum.de/doc/1546085/1546085.pdf
- **Evidence version / retrieval / pin or digest:** TUM author manuscript for DOI `10.1109/ITSC45102.2020.9294629`; retrieved `2026-07-01`; SHA-256 `1c9121cd3b1e980fe2c925e9aa2bc5f0552a5f1a5e1fea2813e960095d1dc87d`.
- **Deciding locator and fact:** pp. 1-4, Abstract, Section I contributions, and Section III: the workflow extracts a large, diverse set of OSM road intersections/networks before populating them with traffic and optimizing scenarios.
- **Comparison rationale:** The traffic-only rating captures later actor optimization, but `include-1` explicitly includes *selecting* transferable course geometry. The source-native extraction/selection of road networks is material and has inclusion precedence.

## C0128

- **candidate_id:** `C0128`
- **cite_key/title:** `Althoff2017CommonRoadComposable` — *CommonRoad: Composable Benchmarks for Motion Planning on Roads*.
- **Raw sealed ratings / trigger IDs:** `A-C0128-05: boundary / boundary / official_documentation`; `A-C0128-06: included / include-1 / full_text`.
- **Recommendation:** `included` / `include-2`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://doi.org/10.1109/ivs.2017.7995802 ; https://web.archive.org/web/20180720100708id_/http://mediatum.ub.tum.de/doc/1379638/document.pdf
- **Evidence version / retrieval / pin or digest:** author manuscript for DOI `10.1109/IVS.2017.7995802`; retrieved `2026-07-01`; archived PDF URL above; SHA-256 `fa78a36bb0b499551b245bf93ae44c6ea6338b49f8ab075fc6e79922fe1fab86`.
- **Deciding locator and fact:** pp. 2 and 5, “Portability,” Section V.A “Road Network,” and Figure 4: the source defines XML scenarios containing lanelet road networks, goals/constraints, and composable benchmark components; it includes constructed scenarios.
- **Comparison rationale:** This is more than a fixed-course reporting transfer. The source natively defines a reusable course/scenario representation and interchange benchmark, satisfying `include-2`; that resolves the boundary versus inclusion disagreement.

## C0129

- **candidate_id:** `C0129`
- **cite_key/title:** `Xu2024OmniDronesEfficient` — *OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control*.
- **Raw sealed ratings / trigger IDs:** `A-C0129-01: excluded / exclude-appearance-dynamics / full_text`; `A-C0129-04: included / include-2 / full_text`.
- **Recommendation:** `excluded` / `exclude-appearance-dynamics`.
- **Exclusion reason:** OmniDrones supplies control benchmark tasks: `Track` follows a supplied reference-state trajectory and `FlyThrough` places obstacles; the paper does not define a source-native generated or parameterized course representation, generator, or course benchmark.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://doi.org/10.1109/lra.2024.3356168 ; https://arxiv.org/pdf/2309.12825
- **Evidence version / retrieval / pin or digest:** `arXiv:2309.12825v1`; retrieved `2026-07-01`; SHA-256 `437eb8f35be7741e8ab9ac40da97a2a84c45d696e59b6e0b4024e8c3375b4cd6`.
- **Deciding locator and fact:** Section III.D “Benchmarking Tasks”: Track is reference-trajectory tracking; FlyThrough uses placed obstacles requiring coherent control actions.
- **Comparison rationale:** The included rating treats task specifications as a course interface, but the inspected report characterizes them as control tasks and does not establish a generated-course contribution. The more specific appearance/dynamics exclusion is conservative.

## C0130

- **candidate_id:** `C0130`
- **cite_key/title:** `Yu2025MAVRLLearn` — *MAVRL: Learn to Fly in Cluttered Environments With Varying Speed*.
- **Raw sealed ratings / trigger IDs:** `A-C0130-01: excluded / exclude-out-of-scope / abstract_only`; `A-C0130-05: included / include-1 / full_text`.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** The accessible report trains an adaptive-speed collision-avoidance policy in randomly generated clutter; it does not define an ordered route, corridor, gate sequence, or source-native course generator.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://doi.org/10.1109/lra.2024.3522778 ; https://arxiv.org/pdf/2402.08381v1
- **Evidence version / retrieval / pin or digest:** `arXiv:2402.08381v1`; retrieved `2026-07-01`; SHA-256 `dcf4f37059ec3adea927fcec51ad3680a0a4f5a8160d6624c5036ef79808e17b`.
- **Deciding locator and fact:** Section III and Figure 3: obstacle complexity and target-directed trajectories train a navigation policy; generated content is clutter, not a course representation or course-distribution contribution.
- **Comparison rationale:** The full text removes the abstract-only limitation but does not support the broad obstacle-course mapping used by the inclusion rating. Under the protocol’s course definition and source-native requirement, exclusion is warranted.

## C0132

- **candidate_id:** `C0132`
- **cite_key/title:** `LeBars2015WorldRobotic` — *The World Robotic Sailing Championship, a Competition to Stimulate the Development of Autonomous Sailboats*.
- **Raw sealed ratings / trigger IDs:** `A-C0132-03: excluded / exclude-insufficient-detail / abstract_only`; `A-C0132-05: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** Available material discusses the championship generally but does not expose a technically inspectable course definition, parameterization, competition course set, or citable boundary transfer.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1109/oceans-genova.2015.7271767 ; https://ieeexplore.ieee.org/document/7271767/
- **Evidence version / retrieval / pin or digest:** IEEE DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not obtain full text; no archive pin.
- **Deciding locator and fact:** IEEE abstract/introductory record: championship description does not supply inspectable course technical detail.
- **Comparison rationale:** Both snapshots agree. Neither `include-2` nor `boundary` can be based on a generic competition mention.

## C0134

- **candidate_id:** `C0134`
- **cite_key/title:** `Ayerdi2023RoadSignSBFT` — *RoadSign at the SBFT 2023 Tool Competition Cyber-Physical Systems Track*.
- **Raw sealed ratings / trigger IDs:** `A-C0134-01: excluded / exclude-insufficient-detail / abstract_only`; `A-C0134-03: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** The abstract identifies generated failure-revealing roads, but no public technical report or official artifact was available to verify the implementation, road representation, constraints, or precise contribution.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1109/sbft59156.2023.00006 ; https://ieeexplore.ieee.org/document/10190387/
- **Evidence version / retrieval / pin or digest:** IEEE DOI version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not locate a full report/artifact; no archive pin.
- **Deciding locator and fact:** IEEE Computer Society abstract: road generation, seeding, and multi-objective optimization are named without material technical detail.
- **Comparison rationale:** Both snapshots agree. This is a strong retrieval candidate, but the protocol bars inclusion on abstract-only evidence.

## C0135

- **candidate_id:** `C0135`
- **cite_key/title:** `DeVivo2023SpiraleSBFT` — *Spirale at the SBFT 2023 Tool Competiton - Cyber-Physical Systems Track*.
- **Raw sealed ratings / trigger IDs:** `A-C0135-01: excluded / exclude-insufficient-detail / abstract_only`; `A-C0135-06: included / include-1 / official_documentation`.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://doi.org/10.1109/sbft59156.2023.00007 ; https://codeload.github.com/domenico-devivo/cps-tool-competition/tar.gz/d378dae5bfc8e5fe3b015a5b90119cabd74db23c
- **Evidence version / retrieval / pin or digest:** commit `d378dae5bfc8e5fe3b015a5b90119cabd74db23c`; retrieved `2026-07-01`; pinned archive URL above; SHA-256 `10664ab86971732aaf1fae984fcdf57b0dbb5b80526d0f53f7a89e763d55cb7f`.
- **Deciding locator and fact:** `spirale/README.md` lines 5-12 and 27-38; `spirale/base.py` lines 31-116: the tool creates random spiral-arc road points, crosses roads over generations, and selects candidates by test fitness.
- **Comparison rationale:** The pinned official implementation supplies direct source-native geometry generation that the abstract-only rating lacked. `include-1` therefore applies.

## C0137

- **candidate_id:** `C0137`
- **cite_key/title:** `Han2021PreliminaryEvaluation` — *Preliminary Evaluation of Path-aware Crossover Operators for Search-Based Test Data Generation for Autonomous Driving*.
- **Raw sealed ratings / trigger IDs:** `A-C0137-04: included / include-1 / full_text`; `A-C0137-06: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://doi.org/10.1109/sbst52555.2021.00020 ; https://web.archive.org/web/20211220135757id_/https://coinse.kaist.ac.kr/publications/pdfs/Han2021vp.pdf
- **Evidence version / retrieval / pin or digest:** SBST 2021 archived author manuscript; retrieved `2026-07-01`; archived PDF URL above; SHA-256 `4a64ca9f77c3e07e413705dc57adf5e74149657adb42a2243d78ee711e1c5cba`.
- **Deciding locator and fact:** pp. 1-3, Introduction and Section II: AsFault incrementally builds road networks from road segments; the paper’s path-aware crossover combines parent road-map segments and validates generated networks.
- **Comparison rationale:** The public author manuscript resolves the evidence gap. It directly mutates and generates connected road geometry, fulfilling `include-1`.

## C0139

- **candidate_id:** `C0139`
- **cite_key/title:** `Loquercio2021LearningHigh` — *Learning high-speed flight in the wild*.
- **Raw sealed ratings / trigger IDs:** `A-C0139-03: excluded / exclude-appearance-dynamics / full_text`; `A-C0139-05: included / include-1 / full_text`.
- **Recommendation:** `excluded` / `exclude-appearance-dynamics`.
- **Exclusion reason:** The source randomizes trees and generic convex obstacles to train a collision-avoidance policy; it separately computes a reference trajectory and does not contribute a course generator, ordered course representation, or course benchmark.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://doi.org/10.1126/scirobotics.abg5810 ; https://arxiv.org/pdf/2110.05113v1
- **Evidence version / retrieval / pin or digest:** `arXiv:2110.05113v1`; retrieved `2026-07-01`; SHA-256 `9f6c4e319a95ffcca6ec85fd7e2446f3a4f22ddc8b6921b033a16e5876a424a9`.
- **Deciding locator and fact:** Supplementary Materials, Section C “Training environments for the task”: trees/shapes are spawned from randomized distributions and a global collision-free trajectory is then computed for policy training.
- **Comparison rationale:** The inclusion rating treats random obstacle fields as generated courses. The protocol excludes appearance/environment variation without a source-native ordered or topologically connected course contribution; the direct full text supports the more specific exclusion.

## C0141

- **candidate_id:** `C0141`
- **cite_key/title:** `Cardamone2011InteractiveEvolution` — *Interactive Evolution for the Procedural Generation of Tracks in a High-End Racing Game*.
- **Raw sealed ratings / trigger IDs:** `A-C0141-02: excluded / exclude-insufficient-detail / abstract_only`; `A-C0141-03: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** Indexed material reports interactive evolution of TORCS tracks, but no full paper or authoritative implementation was available to inspect the representation, operators, or evaluation.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1145/2001576.2001631 ; http://hdl.handle.net/11311/609170
- **Evidence version / retrieval / pin or digest:** ACM version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` did not recover complete evidence; no archive pin.
- **Deciding locator and fact:** ACM DOI/OpenAlex and institutional-handle records: no inspectable report file or official companion artifact.
- **Comparison rationale:** Both snapshots agree. The protocol does not permit a source-native course-generation claim from indexed abstract content alone.

## C0143

- **candidate_id:** `C0143`
- **cite_key/title:** `NR` — *Wasserstein generative adversarial networks for online test generation for cyber physical systems*.
- **Raw sealed ratings / trigger IDs:** `A-C0143-02: included / include-1 / full_text`; `A-C0143-06: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://doi.org/10.1145/3526072.3527522 ; https://arxiv.org/pdf/2205.11060v1
- **Evidence version / retrieval / pin or digest:** `arXiv:2205.11060v1`; retrieved `2026-07-01`; SHA-256 `d7965979d29824c35dd84cb6a784092e404839d2776b8e349ddc4236f55aecd8`.
- **Deciding locator and fact:** Abstract; Section 2.1 “Feature Representation”; Algorithm 1: WOGAN represents roads with curvature vectors and plane points, generates candidates, and validates nonintersection and turn constraints before lane-keeping execution.
- **Comparison rationale:** The public preprint resolves the abstract-only rating. The direct road representation and generation/validation process satisfy `include-1`.

## C0144

- **candidate_id:** `C0144`
- **cite_key/title:** `Starace2022GenRLSBST` — *GenRL at the SBST 2022 Tool Competition*.
- **Raw sealed ratings / trigger IDs:** `A-C0144-02: excluded / exclude-insufficient-detail / abstract_only`; `A-C0144-06: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** The available abstract identifies a lane-keeping test generator, but no public paper or official implementation exposes its road representation, generated geometry, or validation process.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.1145/3526072.3527533 ; https://dl.acm.org/doi/10.1145/3526072.3527533
- **Evidence version / retrieval / pin or digest:** ACM version of record; sealed retrieval `2026-06-30`; public-primary recheck `2026-07-01` found no complete report or official artifact; no archive pin.
- **Deciding locator and fact:** ACM/OpenAlex record abstract: lane-keeping test generation is named but not technically inspectable.
- **Comparison rationale:** Both snapshots agree. The protocol’s full-text/evidence rules require exclusion until a source-native artifact is recovered.

## C0150

- **candidate_id:** `C0150`
- **cite_key/title:** `Hwang2022NavigationScenario` — *Navigation Scenario Permutation Model for Training of Maritime Autonomous Surface Ship Remote Operators*.
- **Raw sealed ratings / trigger IDs:** `A-C0150-02: included / include-1 / full_text`; `A-C0150-04: excluded / exclude-insufficient-detail / abstract_only`.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://doi.org/10.3390/app12031651 ; https://mdpi-res.com/d_attachment/applsci/applsci-12-01651/article_deploy/applsci-12-01651-v2.pdf
- **Evidence version / retrieval / pin or digest:** Applied Sciences 12(3):1651, v2 PDF; retrieved `2026-07-01`; SHA-256 `95ab84d67915610dbf600b9e097092f59ff27d2d8871146d48b4742f0e8e41b3`.
- **Deciding locator and fact:** Sections 2.1 and 2.4, Figure 5, and Section 3.3: permutations of course-altering angles at waypoints and distances between waypoints generate 564,480 practical ordered navigation scenarios.
- **Comparison rationale:** The full public article resolves the abstract-only rating. It directly synthesizes ordered marine routes from waypoint geometry and therefore meets `include-1`.
