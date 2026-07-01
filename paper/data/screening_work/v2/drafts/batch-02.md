# Batch 02 adjudication dossier

Prepared 2026-07-01 from the frozen v2 protocol, reviewer prompt, source metadata,
candidate-conflict ledger, and sealed calibration/main result snapshots. `Raw ratings`
are transcribed, not altered. `Trigger IDs` means the candidate-conflict IDs present in
the frozen v2 ledger; a rating disagreement has no separate trigger ID in the sealed
result schema and is identified by its two assignment rows. Evidence digests below are
the sealed reviewers' artifact digests where available, not newly computed digests.

## C0056 - Cantarella2002MinimumRopelength, *On the minimum ropelength of knots and links*

- **Raw ratings / trigger IDs:** `A-C0056-02` excluded / `exclude-out-of-scope`; `A-C0056-04` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: the report proves bounds and regularity for mathematical knots and links, not a navigable course, course generator, representation, benchmark, or survey-gap contribution.
- **Access / canonical URLs:** full text; https://arxiv.org/pdf/math/0103224v3 ; https://doi.org/10.1007/s00222-002-0234-y
- **Evidence:** `arXiv:math/0103224v3`; retrieved 2026-07-01; archive https://arxiv.org/pdf/math/0103224v3; sealed digest `edec2d3d3912d913d9e95ec911cd485cf57e9749ff4dee001febe7c6ac6f9c10`.
- **Deciding locator / fact:** Abstract, Section 2 "Thickness and Ropelength," and Theorem 1. The stated contribution is mathematical knot/link ropelength theory.
- **Protocol comparison:** unlike a source-native adjacent route generator or repair method, no source claim connects the mathematical results to generated or parameterized courses; `include-1` through `include-4` and boundary therefore do not apply.

## C0057 - Yu2021RepulsiveCurves, *Repulsive Curves*

- **Raw ratings / trigger IDs:** `A-C0057-01` excluded / `exclude-out-of-scope`; `A-C0057-02` included / `include-1`. Trigger IDs: `X42D1680B2D7D`, `X426E79EE6B31`, `XBAA739D6E473`, `X2A6237B1576F` (metadata only).
- **Recommendation:** **included - `include-1`**. Exclusion reason: `NR`.
- **Access / canonical URLs:** full text; https://arxiv.org/pdf/2006.07859v1 ; https://www.cs.cmu.edu/~kmcrane/Projects/RepulsiveCurves/ ; https://doi.org/10.1145/3439429
- **Evidence:** `arXiv:2006.07859v1`; retrieved 2026-07-01; project page exposes the primary report; sealed digest `5e6afce7166f1ed1f5c3864b8b7d69e0ed1afdaa422ff1abc89c5332c69b84c6`.
- **Deciding locator / fact:** primary-project overview; report Sections 3-4, "Repulsive Curves" and "Optimization." The source develops an implemented global curve-optimization method with self-intersection avoidance and identifies robotic path planning among its applications.
- **Protocol comparison:** this is source-native optimization of a spatial route in a transferable adjacent domain. A racing centerline is such a route and can be repaired/optimized without changing the method's essential claim; this differs from C0056/C0059, which only prove geometric results.

## C0058 - Henrich2025GenerationGameplay, *From Generation to Gameplay: Authoring Race Tracks With Repulsive Curves*

- **Raw ratings / trigger IDs:** `A-C0058-03` excluded / `exclude-insufficient-detail`; `A-C0058-04` excluded / `exclude-insufficient-detail`. Trigger IDs: `XDDC1F2D3E974`, `X54DAC6C175D0`, `XABFE93E09ADB`, `X6D1581F3ABA3` (metadata only).
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW; excluded - `exclude-insufficient-detail`**. Exclusion reason: the accessible DOI/metadata abstract says the work generates race tracks, but no complete primary report or authoritative implementation was retrievable for inspection.
- **Access / canonical URLs:** abstract only; https://doi.org/10.1109/tg.2025.3561107 ; https://ieeexplore.ieee.org/document/10965488/
- **Evidence:** version of record `DOI:10.1109/TG.2025.3561107`; retrieved 2026-07-01; no lawful inspectable full-text archive or digest obtained.
- **Deciding locator / fact:** IEEE DOI record and deposited abstract. The abstract alone describes a repulsive-curve race-track generator but does not expose its representation, constraints, or generated artifact.
- **Protocol comparison:** the apparent `include-1` signal cannot override the protocol's material-evidence rule. An accountable author must obtain and inspect the report or authoritative artifact before inclusion.

## C0059 - Lagemann2023TangentPoint, *Tangent-point energies and ropelength as Gamma-limits of discrete tangent-point energies on biarc curves*

- **Raw ratings / trigger IDs:** `A-C0059-01` excluded / `exclude-out-of-scope`; `A-C0059-03` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: the report proves Gamma-convergence for tangent-point energies on biarc curves and makes no course-generation, course-representation, racing, or adjacent-route artifact claim.
- **Access / canonical URLs:** full text; https://arxiv.org/abs/2203.16383 ; https://arxiv.org/pdf/2203.16383v1 ; https://doi.org/10.1186/s13662-022-03750-4
- **Evidence:** `arXiv:2203.16383v1`; retrieved 2026-07-01; archive https://arxiv.org/pdf/2203.16383v1; sealed digest `120134d2c977055331f834d5a4027b54e44d5f7ee87c50f2c77647b11443435d`.
- **Deciding locator / fact:** Abstract and Sections 1-4. The report is a convergence analysis of continuous and discrete curve energies.
- **Protocol comparison:** mathematical curve theory alone is not a source-native generated/parameterized course contribution and supplies no named fixed-course boundary transfer.

## C0060 - Muller2007PositionBased, *Position Based Dynamics*

- **Raw ratings / trigger IDs:** `A-C0060-01` excluded / `exclude-out-of-scope`; `A-C0060-04` excluded / `exclude-out-of-scope`. Trigger IDs: `X720BB2214BD3`, `X4BE257385A54` (metadata only).
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: the author manuscript presents a general position-based solver for dynamic objects and cloth, with no course geometry, interchange, generation, or benchmark claim.
- **Access / canonical URLs:** full text; https://matthias-research.github.io/pages/publications/posBasedDyn.pdf ; https://doi.org/10.1016/j.jvcir.2007.01.005
- **Evidence:** author-hosted VRIPHYS manuscript for the DOI contribution; retrieved 2026-07-01; sealed digest `8823d1b0549e84c56e0c3cbbf5db3cd00dbd59858835f4d012b64604aac2f4b0`.
- **Deciding locator / fact:** Abstract; Sections 1-4, "Position Based Simulation" and "Cloth Simulation." The method handles physical simulation constraints, not navigation constraints.
- **Protocol comparison:** a general simulator is not an `include-2` course interface merely because later course tools may use it.

## C0061 - Macklin2016XPBDPosition, *XPBD: Position-Based Simulation of Compliant Constrained Dynamics*

- **Raw ratings / trigger IDs:** `A-C0061-04` (calibration) excluded / `exclude-out-of-scope`; `A-C0061-05` (calibration) excluded / `exclude-out-of-scope`. Trigger IDs: `X9CAF616BE34E`, `X43966B70C759` (metadata only).
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: XPBD is a compliant-constraint solver for deformable-body simulation, not a generator, representation, characterization, or benchmark of spatial courses.
- **Access / canonical URLs:** full text; https://matthias-research.github.io/pages/publications/XPBD.pdf ; https://doi.org/10.1145/2994258.2994272
- **Evidence:** proceedings contribution `DOI:10.1145/2994258.2994272`; retrieved 2026-07-01; sealed digest `cd96241fe2cce7816c3fa1cb2b9e862e5a610263fd38cc8560bb51b7f6ecd502`.
- **Deciding locator / fact:** Abstract, Section 1, Algorithm 1, and conclusion. The source simulates elastic/dissipative energy potentials and evaluates deformable bodies.
- **Protocol comparison:** as with C0060, reusable physics infrastructure does not itself claim a generated or parameterized course contribution.

## C0062 - Liu2023RoboticManipulation, *Robotic Manipulation of Deformable Rope-Like Objects Using Differentiable Compliant Position-Based Dynamics*

- **Raw ratings / trigger IDs:** `A-C0062-01` excluded / `exclude-out-of-scope`; `A-C0062-06` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: the accessible primary bibliographic abstract unambiguously describes differentiable rope physics and rope manipulation, not navigation courses, racing, or an adjacent route artifact.
- **Access / canonical URLs:** abstract only; https://doi.org/10.1109/lra.2023.3264766 ; https://ieeexplore.ieee.org/document/10093017/
- **Evidence:** version of record `DOI:10.1109/LRA.2023.3264766`; retrieved 2026-07-01; no complete public primary report recovered; sealed abstract-record digest `692333388b0ce524412d147dd2e100769adf46cb336ecc150bbfd59dda96bbf8`.
- **Deciding locator / fact:** deposited abstract, describing manipulation of deformable rope-like objects using compliant position-based dynamics.
- **Protocol comparison:** the abstract directly establishes out-of-scope subject matter, so the protocol permits this specific exclusion despite incomplete access; it cannot support inclusion or boundary.

## C0064 - Macklin2022WarpHigh, *Warp: A High-performance Python Framework for GPU Simulation and Graphics*

- **Raw ratings / trigger IDs:** `A-C0064-03` excluded / `exclude-out-of-scope`; `A-C0064-06` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: Warp is documented as a general GPU simulation, robotics, and geometry-processing framework, without a source-native course contribution.
- **Access / canonical URLs:** official documentation; https://github.com/NVIDIA/warp/tree/ebcce32577cca35930c03191ad6c0ba34c749461
- **Evidence:** commit `ebcce32577cca35930c03191ad6c0ba34c749461`; retrieved 2026-07-01; archive https://codeload.github.com/NVIDIA/warp/tar.gz/ebcce32577cca35930c03191ad6c0ba34c749461; sealed digest `009c06f76412ddfa672b95196f13e6b8c3383f8da05c51efd78cfd523feecb5e`.
- **Deciding locator / fact:** `README.md` lines 8-17 and 80-100. The official description names general simulation and geometry primitives, not course generation/interchange/benchmarking.
- **Protocol comparison:** general-purpose capability or downstream use cannot satisfy `include-1` or `include-2` without a source-native course claim.

## C0065 - NVIDIANodateCUDAGraphs, *CUDA Graphs*

- **Raw ratings / trigger IDs:** `A-C0065-01` excluded / `exclude-out-of-scope`; `A-C0065-02` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: CUDA Graphs capture and replay GPU workloads; they are not spatial road, route, gate, or course graphs.
- **Access / canonical URLs:** official documentation; https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- **Evidence:** CUDA Programming Guide v13.3; retrieved 2026-07-01; official document page is version-labelled; sealed digest `bab118cdfac30dff05b6c9f70dbd18e5e6708fcbc033bc28a9393a9ad5acdfd2`.
- **Deciding locator / fact:** CUDA Programming Guide Section 4.2, "CUDA Graphs." The page defines a GPU execution/work-submission graph.
- **Protocol comparison:** terminology overlap does not create a transferable spatial-route contribution; no inclusion or boundary rule applies.

## C0066 - Ecoffet2021FirstReturn, *First return, then explore*

- **Raw ratings / trigger IDs:** `A-C0066-01` excluded / `exclude-out-of-scope`; `A-C0066-03` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: Go-Explore is a general exploration and robustification algorithm for supplied task environments, not a course-geometry generator or course evaluation method.
- **Access / canonical URLs:** full text; https://arxiv.org/abs/2004.12919 ; https://arxiv.org/pdf/2004.12919v6 ; https://doi.org/10.1038/s41586-020-03157-9
- **Evidence:** `arXiv:2004.12919v6`; retrieved 2026-07-01; archive https://arxiv.org/abs/2004.12919; sealed digest `e293b9f72b0ed95277eb356ff543db8011ede976ba07e628cf2fa81e16a027f3`.
- **Deciding locator / fact:** Sections 2-3, "Go-Explore algorithm," and methods. It archives and returns to promising states while exploring a fixed environment.
- **Protocol comparison:** producing trajectories through an input environment is not synthesizing, selecting, repairing, or serializing the environment's course geometry.

## C0069 - Yannakakis2011ExperienceDriven, *Experience-Driven Procedural Content Generation*

- **Raw ratings / trigger IDs:** `A-C0069-04` included / `include-4`; `A-C0069-06` excluded / `exclude-insufficient-detail`. Trigger IDs: none.
- **Recommendation:** **included - `include-4`**. Exclusion reason: `NR`.
- **Access / canonical URLs:** full text; http://julian.togelius.com/Yannakakis2011Experiencedriven.pdf ; https://doi.org/10.1109/T-AFFC.2011.6
- **Evidence:** author manuscript for `DOI:10.1109/T-AFFC.2011.6`; retrieved 2026-07-01; sealed digest `e4c61ed23a93214fec56bf78322d95b4af8b8ece592af5f7e6f5ca710b0eccde`.
- **Deciding locator / fact:** Sections 2-5, especially Section 5.1-Section 5.2 and the generated racing-track example. The review/framework organizes representation, evaluation, search, and generation of game content, including racing-game work.
- **Protocol comparison:** unlike C0070, a directly inspected survey supplies the needed taxonomy and examples for the course-generation survey gap; `include-4` is the first applicable criterion.

## C0070 - Hendrikx2013ProceduralContent, *Procedural Content Generation for Games: A Survey*

- **Raw ratings / trigger IDs:** `A-C0070-01` excluded / `exclude-insufficient-detail`; `A-C0070-03` excluded / `exclude-insufficient-detail`. Trigger IDs: none.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW; excluded - `exclude-insufficient-detail`**. Exclusion reason: the accessible DOI and abstract-level records establish a broad game-PCG survey but no complete primary survey text was available to verify a course-generation treatment or direct survey-gap role.
- **Access / canonical URLs:** abstract only; https://doi.org/10.1145/2422956.2422957 ; https://dl.acm.org/doi/10.1145/2422956.2422957
- **Evidence:** version of record `DOI:10.1145/2422956.2422957`; retrieved 2026-07-01; no inspectable full-text archive or digest obtained.
- **Deciding locator / fact:** ACM DOI record and deposited abstract-level bibliographic records. They describe a six-layer PCG taxonomy but do not provide inspectable course/road evidence.
- **Protocol comparison:** `include-4` requires a directly inspected survey needed to establish the gap. Broad title/abstract similarity is expressly insufficient; accountable review needs an accessible report.

## C0073 - Kaufmann2018DeepDrone, *Deep Drone Racing: Learning Agile Flight in Dynamic Environments*

- **Raw ratings / trigger IDs:** `A-C0073-01` boundary / `boundary`; `A-C0073-04` included / `include-1`. Trigger IDs: none.
- **Recommendation:** **included - `include-1`**. Exclusion reason: `NR`.
- **Access / canonical URLs:** full text; https://proceedings.mlr.press/v87/kaufmann18a/kaufmann18a.pdf ; https://proceedings.mlr.press/v87/kaufmann18a.html
- **Evidence:** PMLR volume 87 (CoRL 2018); retrieved 2026-07-01; archive https://proceedings.mlr.press/v87/kaufmann18a/kaufmann18a.pdf; sealed digest `9be82e11d0c3d2ee4c9d1cb745a62c27131f4e804fd73a12f192d2d2a203d48d`.
- **Deciding locator / fact:** Section 3.1, "Training procedure," pp. 3-4; Section 4 and Appendix, pp. 11-12. The source varies gate positions to create layouts, computes a global gate-passing trajectory for each, and uses them to generate training data.
- **Protocol comparison:** generated gate layouts are ordered aerial-course constraints and are source-native, so inclusion precedes the otherwise useful fixed/moving-gate reporting boundary.

## C0074 - Loquercio2020DeepDrone, *Deep Drone Racing: From Simulation to Reality With Domain Randomization*

- **Raw ratings / trigger IDs:** `A-C0074-01` excluded / `exclude-appearance-dynamics`; `A-C0074-03` included / `include-1`. Trigger IDs: none.
- **Recommendation:** **included - `include-1`**. Exclusion reason: `NR`.
- **Access / canonical URLs:** full text; https://arxiv.org/pdf/1905.09727v2 ; https://doi.org/10.1109/TRO.2019.2942989
- **Evidence:** `arXiv:1905.09727v2`; retrieved 2026-07-01; archive https://arxiv.org/pdf/1905.09727v2; sealed digest `7091b4eb199684aaec9097b811c563b399074bf50955e5390b4b9b3d07036b48`.
- **Deciding locator / fact:** Section III-A, "Training Procedure," especially the text stating that multiple layouts are generated by moving gates; the same section records 100,000 images of random gate positions and a global trajectory through gate centers. Section III-C separately fixes approximate layout for visual randomization.
- **Protocol comparison:** the visual/appearance-only randomization is excluded by itself, but the independently stated layout and random-gate-position generation changes the ordered course geometry. That material `include-1` evidence takes precedence.

## C0079 - Schwarting2021DeepLatent, *Deep Latent Competition: Learning to Race Using Visual Control Policies in Latent Space*

- **Raw ratings / trigger IDs:** `A-C0079-02` boundary / `boundary`; `A-C0079-05` included / `include-1`. Trigger IDs: none.
- **Recommendation:** **included - `include-1`**. Exclusion reason: `NR`.
- **Access / canonical URLs:** full text and supplement; https://proceedings.mlr.press/v155/schwarting21a/schwarting21a.pdf ; https://github.com/igilitschenski/multi_car_racing/archive/cb593194ed85e00b014ea654db3d19e11dcff5de.tar.gz
- **Evidence:** PMLR 155 (2021) and companion commit `cb593194ed85e00b014ea654db3d19e11dcff5de`; retrieved 2026-07-01; archive URL above; sealed digest `b69b7328fdecff124a589146796862b9dbc920a56dff8f8bebfdcd291d75a1ea`.
- **Deciding locator / fact:** paper Section 5, "Racing environment"; `gym_multi_car_racing/multi_car_racing.py` lines 26 and 183-360 at the pinned companion commit. The released benchmark extends CarRacing with its course construction and serializes the multi-agent racing environment used by the report.
- **Protocol comparison:** the pinned companion is source-native implementation evidence for generated course geometry, so inclusion precedes the alternative boundary interpretation based only on progress rewards and repeated races.

## C0081 - Amini2022VISTA2, *VISTA 2.0: An Open, Data-driven Simulator for Multimodal Sensing and Policy Learning for Autonomous Vehicles*

- **Raw ratings / trigger IDs:** `A-C0081-02` excluded / `exclude-appearance-dynamics`; `A-C0081-03` excluded / `exclude-appearance-dynamics`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-appearance-dynamics`**. Exclusion reason: VISTA synthesizes sensor viewpoints and agents from recorded driving data while retaining the supplied road geometry.
- **Access / canonical URLs:** full text; https://arxiv.org/abs/2111.12083 ; https://arxiv.org/pdf/2111.12083v1 ; https://doi.org/10.1109/icra46639.2022.9812276
- **Evidence:** `arXiv:2111.12083v1`; retrieved 2026-07-01; archive https://arxiv.org/abs/2111.12083; sealed digest `4a626ee857b87c77118af2eb6ae650786524a5efe34c150d1a7e9e6e76d39f55`.
- **Deciding locator / fact:** Section III-A-Section III-C and Section IV. The source generates novel sensor observations and agent/viewpoint variation around recorded trajectories, not roads or course geometry.
- **Protocol comparison:** it is exactly the protocol's appearance/sensing/agent variation on fixed geometry, not `include-1` spatial-course generation.

## C0082 - MIT2017MITRACECAR, *MIT RACECAR*

- **Raw ratings / trigger IDs:** `A-C0082-05` excluded / `exclude-out-of-scope`; `A-C0082-06` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: the official project page documents a 1/10-scale research and teaching platform, not a course generator, representation, benchmark, or named boundary transfer.
- **Access / canonical URLs:** official documentation; https://mit-racecar.github.io/
- **Evidence:** MIT RACECAR project site `unversioned@2026-07-01`; retrieved 2026-07-01; no version-pinned archive/digest independently obtained.
- **Deciding locator / fact:** Home, "A Powerful Platform for Robotics Research and Teaching." The page describes sensors, compute hardware, and open-source educational material.
- **Protocol comparison:** a racing platform does not become a source-native generated-course artifact merely because it can be used on courses.

## C0087 - Wischnewski2022IndyAutonomous, *Indy Autonomous Challenge - Autonomous Race Cars at the Handling Limits*

- **Raw ratings / trigger IDs:** `A-C0087-04` excluded / `exclude-out-of-scope`; `A-C0087-06` boundary / `boundary`. Trigger IDs: none.
- **Recommendation:** **boundary - `boundary`**. Exclusion reason: `NR`.
- **Access / canonical URLs:** full text; https://arxiv.org/pdf/2202.03807v1 ; https://doi.org/10.1007/978-3-662-64550-5_10
- **Evidence:** `arXiv:2202.03807v1`; retrieved 2026-07-01; archive https://arxiv.org/pdf/2202.03807v1; sealed digest `4303396f6524bf83ec396ee057c8ca19190f3a029e2cac90769eee92e9fe119b`.
- **Deciding locator / fact:** pp. 1-9, especially Section 2 on racing scenarios and Section 3 on performance aspects. The report specifies fixed-oval track-boundary, surface-condition, safety, and multi-vehicle scenario requirements.
- **Protocol comparison:** it does not generate courses, but these are material, named fixed-course requirements that the survey can transfer for evaluating generated course realizations; boundary is therefore available after inclusion fails.

## C0088 - IndyAutonomousChallengeNodateIndyAutonomous, *Indy Autonomous Challenge*

- **Raw ratings / trigger IDs:** `A-C0088-04` (calibration) boundary / `boundary`; `A-C0088-05` (calibration) excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **boundary - `boundary`**. Exclusion reason: `NR`.
- **Access / canonical URLs:** official documentation; https://www.indyautonomouschallenge.com/s/2022-ACTMS-Rules-v100.pdf ; https://www.indyautonomouschallenge.com/
- **Evidence:** *IAC Passing Competition Rules* v1.0.0 (2022-11-08); retrieved 2026-07-01; version-pinned official PDF https://www.indyautonomouschallenge.com/s/2022-ACTMS-Rules-v100.pdf; sealed digest `57488a57951b046b43e41de56ef5a6fdaf1bc4d4199895a387d4cc96a75bff0c`.
- **Deciding locator / fact:** Section 4.4, pp. 5-6, and Section 7, p. 12. The rules define successful-pass/safety conditions, completed-lap speed, repeated rounds, and time-trial seeding on a fixed speedway.
- **Protocol comparison:** the official rule set supplies specific reporting and safety conditions transferable as a boundary, while making no course-generation contribution; it is stronger evidence than the general competition home page.

## C0090 - CMURI2024DriverlessIntelligent, *Driverless Intelligent Vehicles Lab*

- **Raw ratings / trigger IDs:** `A-C0090-02` (calibration) excluded / `exclude-out-of-scope`; `A-C0090-04` (calibration) excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: the lab profile lists broad autonomous-driving themes and racing participation but no technical course artifact or precise boundary-transfer evidence.
- **Access / canonical URLs:** official documentation; https://www.ri.cmu.edu/robotics-groups/driverless-intelligent-vehicles-lab/
- **Evidence:** page revision 2026-01-06; retrieved 2026-07-01; no version-pinned archive/digest independently obtained.
- **Deciding locator / fact:** "Statement" tab and "Adversarial Multi-Agent Systems" section. The material describes research topics, not a course generator or representation.
- **Protocol comparison:** a lab/project description is expressly insufficient for inclusion or boundary without source-specific technical evidence.

## C0091 - AIRacingTech2020AIRacing, *AI Racing Tech*

- **Raw ratings / trigger IDs:** `A-C0091-01` excluded / `exclude-out-of-scope`; `A-C0091-04` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: the official team biography and competition-history page defines no generator, course representation, metric, dataset property, or transferable technical practice.
- **Access / canonical URLs:** official documentation; https://www.airacingtech.com/about-us
- **Evidence:** `unversioned@2026-07-01`; retrieved 2026-07-01; no version-pinned archive/digest independently obtained.
- **Deciding locator / fact:** "Our Mission" and "The Team" sections. The page is a team profile rather than an authoritative course artifact.
- **Protocol comparison:** topical relevance to racing cannot establish either a source-native contribution or a named boundary transfer.

## C0098 - OpenStreetMapNodateOpenStreetMap, *OpenStreetMap*

- **Raw ratings / trigger IDs:** `A-C0098-03` included / `include-1`; `A-C0098-04` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: OpenStreetMap documentation describes a collaboratively surveyed and edited real-world map dataset, not generation or parameterization of a course distribution.
- **Access / canonical URLs:** official documentation; https://wiki.openstreetmap.org/wiki/About_OpenStreetMap ; https://wiki.openstreetmap.org/wiki/Elements
- **Evidence:** OpenStreetMap Wiki revision `3045185`; retrieved 2026-07-01; official document revision is visible on the page; sealed digest `d2f658b8c7fe83204f40818db809ca85ff7bc4d55e2bd7674db00b5e496b671e`.
- **Deciding locator / fact:** "About OpenStreetMap," "The Map," "Mapping," and "Using OpenStreetMap data"; Elements anchors "Node," "Way," and "Relation." Ways serialize surveyed map features as ordered nodes, but the source does not claim generated/parameterized courses.
- **Protocol comparison:** serializing observed road geometry is not enough: the required source-native contribution must concern generated/parameterized courses or a transferable generator/representation without changing its essential claim. No such claim is present.

## C0099 - Weiss2017PositionBased, *Position-Based Multi-Agent Dynamics for Real-Time Crowd Simulation*

- **Raw ratings / trigger IDs:** `A-C0099-03` excluded / `exclude-traffic-only`; `A-C0099-05` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-traffic-only`**. Exclusion reason: the method updates crowd-agent positions, collision constraints, and behavior inside supplied spaces; it creates no road, route, or course geometry.
- **Access / canonical URLs:** full text; https://arxiv.org/abs/1802.02673 ; https://arxiv.org/pdf/1802.02673 ; https://doi.org/10.1145/3136457.3136462
- **Evidence:** `arXiv:1802.02673`; retrieved 2026-07-01; archive https://arxiv.org/abs/1802.02673; sealed digest `8f949e12b8c987caec322e07d0139ff10cd01f93941c77a5b08c8fa5324df179`.
- **Deciding locator / fact:** Sections 3-5, on position-based crowd dynamics, collision constraints, and experiments. The generated/updated entities are traffic participants, not the spatial environment.
- **Protocol comparison:** this squarely matches the traffic-only exclusion, which is more precise than the generic out-of-scope alternative.

## C0100 - Marsh2024Uv, *uv*

- **Raw ratings / trigger IDs:** `A-C0100-03` excluded / `exclude-out-of-scope`; `A-C0100-05` excluded / `exclude-out-of-scope`. Trigger IDs: none.
- **Recommendation:** **excluded - `exclude-out-of-scope`**. Exclusion reason: uv is a Python package/project manager with no spatial course, route, road, simulator-interface, or benchmark contribution.
- **Access / canonical URLs:** official documentation; https://docs.astral.sh/uv/
- **Evidence:** documentation dated 2026-03-13; retrieved 2026-07-01; no version-pinned archive/digest independently obtained.
- **Deciding locator / fact:** documentation home, "Highlights," "Projects," and "Python versions." The source defines package and environment-management functions.
- **Protocol comparison:** build tooling is unrelated to every inclusion and boundary criterion.

## C0101 - Klampfl2022UsingGenetic, *Using Genetic Algorithms for Automating Automated Lane-Keeping System Testing*

- **Raw ratings / trigger IDs:** `A-C0101-04` excluded / `exclude-insufficient-detail`; `A-C0101-06` excluded / `exclude-insufficient-detail`. Trigger IDs: none.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW; excluded - `exclude-insufficient-detail`**. Exclusion reason: the publisher-deposited abstract describes automated generation of challenging parametric road networks from control points, but the complete report/authoritative companion could not be inspected.
- **Access / canonical URLs:** abstract only; https://doi.org/10.1002/smr.2520 ; https://onlinelibrary.wiley.com/doi/10.1002/smr.2520
- **Evidence:** version of record `DOI:10.1002/smr.2520`; retrieved 2026-07-01; Crossref exposes the publisher-deposited abstract, but no inspectable full-text archive or digest was obtained.
- **Deciding locator / fact:** Crossref work `10.1002/smr.2520`, `message.abstract`. It states that control points construct a parametric road network and a genetic search changes road geometry to make lane keeping fail.
- **Protocol comparison:** the abstract strongly signals transferable-adjacent `include-1`, but the protocol forbids inclusion based on incomplete evidence. Accountable-author retrieval of the paper or authoritative code is required.

## C0104 - Krishnan2021AirLearning, *Air Learning: a deep reinforcement learning gym for autonomous aerial robot visual navigation*

- **Raw ratings / trigger IDs:** `A-C0104-03` excluded / `exclude-appearance-dynamics`; `A-C0104-04` included / `include-1`. Trigger IDs: none.
- **Recommendation:** **included - `include-1`**. Exclusion reason: `NR`.
- **Access / canonical URLs:** full text; https://link.springer.com/content/pdf/10.1007/s10994-021-06006-6.pdf ; https://doi.org/10.1007/s10994-021-06006-6
- **Evidence:** version of record `DOI:10.1007/s10994-021-06006-6`; retrieved 2026-07-01; archive https://doi.org/10.1007/s10994-021-06006-6; sealed digest `2567e3a3d1face863c5389b196b0246a12ebccf4233ab1699fe815db605966fe`.
- **Deciding locator / fact:** Section 4.1, "Environment generator," pp. 10-12; Table 2 and Figures 2-3; generated-environment discussion pp. 17-20. The source exposes a configurable random environment generator that changes obstacle number/placement and speed, arena size, and start/goal conditions for point-to-point aerial navigation.
- **Protocol comparison:** the appearance/dynamics-only reading omits material spatial variation. Obstacle positions and arena bounds determine the traversable navigation geometry; that transferable adjacent navigation level maps to an aerial robot course without changing the source's essential generator claim.

