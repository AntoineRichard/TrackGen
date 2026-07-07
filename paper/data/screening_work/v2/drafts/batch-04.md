# Batch 04 adjudication dossier

Scope: `C0151;C0153;C0155;C0157;C0159;C0160;C0164;C0165;C0170;C0178;C0180;C0182;C0185;C0187;C0188;C0190;C0192;C0193;C0198;C0200;C0203;C0204;C0209;C0210`.

This is an adjudication draft, not a result CSV or a projection. Raw sealed ratings are transcribed as `assignment_id: status / criterion / access`; no trigger-ID column exists in the sealed result schema, so every entry records `trigger IDs: none recorded`. Evidence versions, locators, and digests below are from the locked primary-source inspections dated `2026-06-30`. A direct public-primary recheck was attempted on `2026-07-01`, but the retrieval service did not return source bodies; recommendations therefore do not infer facts beyond the source-specific sealed evidence.

## C0151

- **candidate_id:** `C0151`
- **cite_key/title:** `Campos2015ProceduralGeneration` — *Procedural Generation of Road Paths for Driving Simulation*.
- **Raw sealed ratings / trigger IDs:** `A-C0151-01: excluded / exclude-insufficient-detail / abstract_only`; `A-C0151-05: excluded / exclude-insufficient-detail / abstract_only`; trigger IDs: none recorded.
- **Recommendation:** **NEEDS_ACCOUNTABLE_AUTHOR_REVIEW**; `excluded` / `exclude-insufficient-detail`.
- **Exclusion reason:** Available abstract/excerpt material claims procedural road-path generation but does not expose a complete method, representation, constraints, outputs, or authoritative implementation.
- **Access status:** `abstract_only`.
- **Canonical source URL(s):** https://doi.org/10.4018/ijcicg.2015070103 ; https://www.igi-global.com/viewtitle.aspx?TitleId=147171
- **Evidence version / retrieval / pin or digest:** DOI version of record; sealed retrieval `2026-06-30`; no inspectable primary artifact pin. Sealed record digests: `b5b0b3505b6bc9f20b96c9722b48e74b5692db44af8c4a9f4c91f465b4c1dc58`, `c4560e53490655a014b321ec7438217fe22bfe4f3d309ac05f24f7a548f650d5`.
- **Deciding locator and fact:** OpenAlex abstract sentences 1-7; IGI Global Abstract and Introduction excerpt paragraphs 1-6. They describe the topic but do not provide inspectable technical material.
- **Comparison rationale:** Both ratings agree. The protocol forbids an inclusion or boundary decision from abstract-only evidence; title and abstract relevance cannot establish `include-1`.

## C0153

- **candidate_id:** `C0153`
- **cite_key/title:** `Brockman2016OpenAIGym` — *OpenAI Gym*.
- **Raw sealed ratings / trigger IDs:** `A-C0153-05: excluded / exclude-out-of-scope / full_text`; `A-C0153-06: included / include-1 / full_text_and_supplement`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text_and_supplement`.
- **Canonical source URL(s):** https://arxiv.org/abs/1606.01540 ; https://arxiv.org/pdf/1606.01540v1 ; https://github.com/openai/gym/commit/c17ac6cc55fda4be60548e2e05b54f22e83e2c1b
- **Evidence version / retrieval / pin or digest:** `arXiv:1606.01540v1`; Gym commit `c17ac6cc55fda4be60548e2e05b54f22e83e2c1b`; sealed retrieval `2026-06-30`; pinned archive https://codeload.github.com/openai/gym/tar.gz/c17ac6cc55fda4be60548e2e05b54f22e83e2c1b ; SHA-256 `d8c4e9e7aa5774fa32eb42d1ce2c6e93a182dc2226512eb6cd18973c97383af3`.
- **Deciding locator and fact:** Paper Sections 3-4; `gym/envs/box2d/car_racing.py` lines 22-29 and 134-171 at the pinned commit. The companion code samples checkpoints and constructs a random physics-based car-racing track for an episode.
- **Comparison rationale:** The paper alone is a generic API, but the authoritative companion supplies source-native generated course geometry. Under the inclusion-boundary clarification, emitted explicit racing-track geometry satisfies `include-1`, which precedes exclusion.

## C0155

- **candidate_id:** `C0155`
- **cite_key/title:** `Zhou2022OpenEnded` — *Open-Ended Learning Strategies for Learning Complex Locomotion Skills*.
- **Raw sealed ratings / trigger IDs:** `A-C0155-04: excluded / exclude-out-of-scope / full_text`; `A-C0155-06: included / include-1 / full_text`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** The source evolves height-map terrains for hexapod locomotion; it does not define an ordered or connected route, corridor, gate/waypoint sequence, road network, or course interface.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://arxiv.org/abs/2206.06796 ; https://arxiv.org/pdf/2206.06796v1
- **Evidence version / retrieval / pin or digest:** `arXiv:2206.06796v1`; sealed retrieval `2026-06-30`; archive https://arxiv.org/pdf/2206.06796v1 ; SHA-256 `e7020f14e54ead43742f328c18caec44ef5e7798209db9c9e5a769f542e8e359`.
- **Deciding locator and fact:** pp. 4-7, Environment Evolution and terrain generation; Appendix pp. 11-16. Generated artifacts are terrain surfaces and physical properties, not course constraints.
- **Comparison rationale:** The proposed “sequential terrain” mapping changes the source’s essential terrain contribution into a course claim. The protocol’s course definition requires a traversal-defining connected or ordered spatial constraint, which the source-specific evidence does not establish.

## C0157

- **candidate_id:** `C0157`
- **cite_key/title:** `Joshi2023SimReal` — *Sim-to-Real Deep Reinforcement Learning based Obstacle Avoidance for UAVs under Measurement Uncertainty*.
- **Raw sealed ratings / trigger IDs:** `A-C0157-02: included / include-1 / full_text`; `A-C0157-03: excluded / exclude-appearance-dynamics / full_text`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** The training environment randomizes obstacle count/placement, start and target positions, and measurement noise for point-to-point avoidance, but specifies no ordered route, connected corridor, gate sequence, or source-native course representation.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://arxiv.org/abs/2303.07243 ; https://arxiv.org/pdf/2303.07243v1
- **Evidence version / retrieval / pin or digest:** `arXiv:2303.07243v1`; sealed retrieval `2026-06-30`; archive https://arxiv.org/abs/2303.07243 ; SHA-256 `c3714747db66bff01409fbe2f5c7de7af68186d16935973bff636f9f4803643f`.
- **Deciding locator and fact:** Section III-A, Environment; Section IV-A, Training; Table I; equations 8-10 and the obstacle-generation paragraph. The varied items are an avoidance scenario, not an encoded course.
- **Comparison rationale:** Spatial obstacles plus a start/goal do not, without direct source evidence of an ordered or connected navigation constraint, meet the protocol’s course definition. The inclusion mapping would infer a corridor from free-space navigation, which the protocol forbids.

## C0159

- **candidate_id:** `C0159`
- **cite_key/title:** `Kulkarni2024ReinforcementLearning` — *Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding*.
- **Raw sealed ratings / trigger IDs:** `A-C0159-02: included / include-1 / full_text`; `A-C0159-03: excluded / exclude-appearance-dynamics / full_text`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** The source trains a target-directed collision-avoidance policy using obstacle poses, room bounds, and start/goal samples; it neither generates nor serializes an ordered route, gate sequence, corridor, or course distribution.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://arxiv.org/abs/2402.03947 ; https://arxiv.org/pdf/2402.03947v1 ; https://doi.org/10.1109/icra57147.2024.10610287
- **Evidence version / retrieval / pin or digest:** `arXiv:2402.03947v1`; sealed retrieval `2026-06-30`; archive https://arxiv.org/abs/2402.03947 ; SHA-256 `0d3cbdccb79b3a118d8ee5d71cd0dde775c23600be49e2b5ea1f4e3578a2e5d6`.
- **Deciding locator and fact:** Sections III-B--III-D and IV, especially Section IV-D Training Environment, Figure 4, and Section V-A. The contribution is collision encoding/navigation policy evaluation in obstacle rooms.
- **Comparison rationale:** As with C0157, the source-native contribution is not a course generator. Treating bounded obstacle scenes as open courses would add an unclaimed technical mapping and fail the source-native rule.

## C0160

- **candidate_id:** `C0160`
- **cite_key/title:** `Zhao2024LearningSpeed` — *Learning Speed Adaptation for Flight in Clutter*.
- **Raw sealed ratings / trigger IDs:** `A-C0160-03: excluded / exclude-appearance-dynamics / full_text`; `A-C0160-06: excluded / exclude-out-of-scope / abstract_only`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** The full report varies obstacle distributions and learns speed constraints for a trajectory planner in clutter; it contributes no generated, selected, validated, or serialized course geometry.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://arxiv.org/abs/2403.04586 ; https://arxiv.org/pdf/2403.04586 ; https://doi.org/10.1109/lra.2024.3421789
- **Evidence version / retrieval / pin or digest:** `arXiv:2403.04586`; sealed retrieval `2026-06-30`; archive https://arxiv.org/abs/2403.04586 ; SHA-256 `65545c4541f103f2747149318b2f1519e75bfb2b109e175b86cd0be708beaeaa`.
- **Deciding locator and fact:** Section III-A, Model-based Trajectory Planner; Section IV-B, Training Environment; Section V, Evaluation. The technical claim is speed adaptation for a supplied clutter-navigation task.
- **Comparison rationale:** The full text resolves the abstract-only limitation. Neither rating identifies an ordered/connected course contribution; `exclude-out-of-scope` is the more direct code.

## C0164

- **candidate_id:** `C0164`
- **cite_key/title:** `Wan2025GenTeGenerative` — *GenTe: Generative Real-world Terrains for General Legged Robot Locomotion Control*.
- **Raw sealed ratings / trigger IDs:** `A-C0164-01: excluded / exclude-out-of-scope / full_text`; `A-C0164-06: excluded / exclude-out-of-scope / full_text`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** GenTe generates locomotion surfaces, height maps, and terrain physical properties, not an ordered route, connected corridor, course topology, or course benchmark instance.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://arxiv.org/abs/2504.09997 ; https://arxiv.org/pdf/2504.09997v1
- **Evidence version / retrieval / pin or digest:** `arXiv:2504.09997v1`; sealed retrieval `2026-06-30`; archive https://arxiv.org/pdf/2504.09997v1 ; SHA-256 `8d09aa9a1d0fd97ebbfb90268cb9a924455fb44304b3c4c8bd7e40e20ecf3bcb`.
- **Deciding locator and fact:** pp. 1-4, Abstract and Sections I--III; Section III-A Geometry Terrains; Figure 1. The generated object is terrain, not a traversal course.
- **Comparison rationale:** Both snapshots agree. Terrain generation alone is not a transferable route or course under the protocol’s operational definition.

## C0165

- **candidate_id:** `C0165`
- **cite_key/title:** `Yu2025MasteringDiverse` — *Mastering Diverse, Unknown, and Cluttered Tracks for Robust Vision-Based Drone Racing*.
- **Raw sealed ratings / trigger IDs:** `A-C0165-05: included / include-1 / full_text`; `A-C0165-06: excluded / exclude-insufficient-detail / abstract_only`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://arxiv.org/abs/2512.09571 ; https://arxiv.org/pdf/2512.09571v2 ; https://doi.org/10.1109/lra.2025.3643267
- **Evidence version / retrieval / pin or digest:** `arXiv:2512.09571v2`; sealed retrieval `2026-06-30`; archive https://arxiv.org/pdf/2512.09571v2 ; SHA-256 `274d76d31c82035c13b1ed4e9ec9e83a6c7ebf9604ecad7b2d9ca19e9194b63d`.
- **Deciding locator and fact:** Section III-C, Curriculum Learning for Generalizable Racing, “Track Primitive Generator”; Figure 2. The source documents a generator of racing-track primitives.
- **Comparison rationale:** The accessible full report supplies the missing material evidence. A source-native track primitive generator directly satisfies `include-1`, which supersedes the abstract-only exclusion.

## C0170

- **candidate_id:** `C0170`
- **cite_key/title:** `ASAM2024ASAMOpenDRIVE` — *ASAM OpenDRIVE BS 1.8.1 Specification, 2024-11-21*.
- **Raw sealed ratings / trigger IDs:** `A-C0170-01: included / include-2 / official_documentation`; `A-C0170-03: included / include-1 / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/v1.8.1/specification/index.html ; https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/v1.8.1/specification/09_geometries/09_02_road_reference_line.html
- **Evidence version / retrieval / pin or digest:** ASAM OpenDRIVE 1.8.1, `2024-11-21`; sealed retrieval `2026-06-30`; version-pinned documentation URL above; SHA-256 `b21168096602ee9cfd0e9285335e88c35b058c96ee492bb121b3c10847a00126`.
- **Deciding locator and fact:** Section 9.2, Road reference line, Figure 25 and Table 18; `<planView>` and ordered `<geometry>` elements. The standard serializes no-gap ordered road reference-line geometry.
- **Comparison rationale:** The standard also provides a reusable interchange representation (`include-2`), but the protocol clarification assigns `include-1` first where source-native documentation serializes explicit transferable road/course centerline geometry.

## C0178

- **candidate_id:** `C0178`
- **cite_key/title:** `EUFS2023EUFSMaps` — *EUFS maps*.
- **Raw sealed ratings / trigger IDs:** `A-C0178-01: included / include-1 / official_documentation`; `A-C0178-04: included / include-2 / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://gitlab.com/eufs/public/eufs_maps/-/commit/ccd652da93bcd16b86d89ad1c68dcca440265cbb ; https://gitlab.com/eufs/public/eufs_maps/-/blob/ccd652da93bcd16b86d89ad1c68dcca440265cbb/README.md
- **Evidence version / retrieval / pin or digest:** commit `ccd652da93bcd16b86d89ad1c68dcca440265cbb`; sealed retrieval `2026-06-30`; version-pinned commit URL above; SHA-256 `30c2dae59c81ebadd4bcd90f28fe65367f04b4fb0d1cefcb2e4ec4d7d2bebfc6`.
- **Deciding locator and fact:** `include/eufs_maps/io/writer.hpp` lines 38-82; README lines 1-12; `eufs_maps/competitions` and `eufs_maps/tracks`. The writer serializes Formula Student cone coordinates/covariance into the EUFS course-map CSV form.
- **Comparison rationale:** The released course set and CSV interface also meet `include-2`; however, source-native serialization of explicit course-boundary coordinates triggers the earlier `include-1` criterion.

## C0180

- **candidate_id:** `C0180`
- **cite_key/title:** `OKelly2020F1TENTHOpen` — *F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning*.
- **Raw sealed ratings / trigger IDs:** `A-C0180-01: boundary / boundary / full_text`; `A-C0180-02: included / include-1 / full_text`; trigger IDs: none recorded.
- **Recommendation:** `boundary` / `boundary`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://proceedings.mlr.press/v123/o-kelly20a.html ; https://proceedings.mlr.press/v123/o-kelly20a/o-kelly20a.pdf
- **Evidence version / retrieval / pin or digest:** PMLR volume 123 (2020), pp. 77-89; sealed retrieval `2026-06-30`; archive https://proceedings.mlr.press/v123/o-kelly20a/o-kelly20a.pdf ; SHA-256 `45e171299565b5c99026c1e589b0c74dd73bebc69c72449ef86009446d77699d`.
- **Deciding locator and fact:** PDF pp. 6-7 (printed pp. 82-83), Sections 5, 6.1, 6.2, and Table 1. It evaluates fixed racetrack tasks with collision-free completion, lap time, and paired simulation/hardware reporting.
- **Comparison rationale:** Obstacle placement on known track maps does not establish a source-native course generator. The documented performance and transfer measures are a named fixed-course reporting transfer, so boundary applies only after inclusion fails.

## C0182

- **candidate_id:** `C0182`
- **cite_key/title:** `F1TENTH2021F1tenthF1tenth` — *f1tenth/f1tenth_racetracks*.
- **Raw sealed ratings / trigger IDs:** `A-C0182-03: included / include-1 / official_documentation`; `A-C0182-06: included / include-2 / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-2`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://github.com/f1tenth/f1tenth_racetracks/tree/b95c4eff766f6367d66b310ea20cd2c9563712c0 ; https://github.com/f1tenth/f1tenth_racetracks/commit/b95c4eff766f6367d66b310ea20cd2c9563712c0
- **Evidence version / retrieval / pin or digest:** commit `b95c4eff766f6367d66b310ea20cd2c9563712c0`; sealed retrieval `2026-06-30`; pinned archive https://codeload.github.com/f1tenth/f1tenth_racetracks/tar.gz/b95c4eff766f6367d66b310ea20cd2c9563712c0 ; SHA-256 `a38dcc6852a718efbd97b73d927a963ea1a9e1d9c869767112b481db9122a0e3`.
- **Deciding locator and fact:** README lines 1-2, 24-64, and 68-71; BrandsHatch centerline lines 1-8. The repository releases reusable maps, centerlines, widths, racelines, waypoints, and simulator files.
- **Comparison rationale:** This is a source-native competition course set and course representation, expressly covered by `include-2`. The evidence documents supplied course artifacts rather than a generator or serialization operation, so `include-2` is more precise than `include-1`.

## C0185

- **candidate_id:** `C0185`
- **cite_key/title:** `Kabzan2019FSSIM` — *FSSIM*.
- **Raw sealed ratings / trigger IDs:** `A-C0185-01: included / include-2 / official_documentation`; `A-C0185-03: included / include-1 / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://github.com/AMZ-Racing/fssim/tree/cf652d8d3f1e13031dad3fb75eb3d4e6fbaaeff4 ; https://github.com/AMZ-Racing/fssim/blob/cf652d8d3f1e13031dad3fb75eb3d4e6fbaaeff4/fssim_rqt_plugins/rqt_fssim_track_editor/src/rqt_fssim_track_editor/track.py
- **Evidence version / retrieval / pin or digest:** commit `cf652d8d3f1e13031dad3fb75eb3d4e6fbaaeff4`; sealed retrieval `2026-06-30`; version-pinned tree URL above; SHA-256 `afb05ea9efefd21b44a09cb0b2b7095c84629d7eb1b12e89d75683e4b31f606e`.
- **Deciding locator and fact:** `track.py` lines 117-135, 187-235, 302-360, and 399-419; `fssim_common/msg/Track.msg` lines 1-7. The track editor constructs and serializes typed cone/timing-device coordinate arrays for Formula Student tracks.
- **Comparison rationale:** FSSIM also defines a simulator course interface (`include-2`), but the source-native editor’s explicit construction/serialization of track coordinates satisfies the earlier `include-1` criterion.

## C0187

- **candidate_id:** `C0187`
- **cite_key/title:** `BYUFRoStLabNodateHoloOcean1` — *HoloOcean 1.0.0: Scenarios*.
- **Raw sealed ratings / trigger IDs:** `A-C0187-02: excluded / exclude-out-of-scope / official_documentation`; `A-C0187-04: excluded / exclude-appearance-dynamics / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** The scenario schema retains a selected fixed world/map and only places agents/sensors and optionally randomizes an agent’s starting location/rotation; it exposes no course geometry, generator, or qualifying boundary transfer.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://byu-holoocean.github.io/holoocean-docs/v1.0.0/usage/scenarios.html#what-is-a-scenario ; https://byu-holoocean.github.io/holoocean-docs/v1.0.0/usage/scenarios.html#location-randomization
- **Evidence version / retrieval / pin or digest:** HoloOcean 1.0.0; sealed retrieval `2026-06-30`; versioned documentation URL above; SHA-256 `d1c51cac66c5dd9ebb83402605a2815dae24eeb5642eea950e598ac356284961`.
- **Deciding locator and fact:** “What is a scenario?” and “Location Randomization.” The world is loaded, not generated or parameterized as a course.
- **Comparison rationale:** The appearance/dynamics code captures the optional pose randomization, but the more direct finding is that this generic scenario loader has no source-native course contribution at all.

## C0188

- **candidate_id:** `C0188`
- **cite_key/title:** `IsaacLabNodateIsaacLab` — *isaaclab.terrains.terrain_generator*.
- **Raw sealed ratings / trigger IDs:** `A-C0188-02: excluded / exclude-out-of-scope / official_documentation`; `A-C0188-05: excluded / exclude-out-of-scope / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** `TerrainGenerator` composes generic terrain meshes, patches, and curriculum difficulty grids; it does not generate a route, corridor, centerline, gates, or course topology.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/terrains/terrain_generator.html
- **Evidence version / retrieval / pin or digest:** Isaac Lab documentation 2.3.2, `main@2026-06-30`; sealed retrieval `2026-06-30`; no immutable archive URL supplied; SHA-256 `bfe4e086b9d04638377234a975ea6b259fb00b47fbc4c39c9afa8428e9674b71`.
- **Deciding locator and fact:** `TerrainGenerator` class and terrain import/generation methods; curriculum and random terrain-generation paragraphs. The output is terrain geometry, not a course.
- **Comparison rationale:** Both snapshots agree. Generic terrain variation cannot be made a course contribution by an inferred downstream use.

## C0190

- **candidate_id:** `C0190`
- **cite_key/title:** `Rudin2022LearningWalk` — *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*.
- **Raw sealed ratings / trigger IDs:** `A-C0190-04: excluded / exclude-out-of-scope / full_text`; `A-C0190-05: excluded / exclude-out-of-scope / full_text`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** The report’s source-native contribution is quadruped policy training over tiled terrain types and randomized obstacles, not an ordered course, route, corridor, or course generator.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://proceedings.mlr.press/v164/rudin22a.html ; https://proceedings.mlr.press/v164/rudin22a/rudin22a.pdf
- **Evidence version / retrieval / pin or digest:** PMLR volume 164 (2022); sealed retrieval `2026-06-30`; archive https://proceedings.mlr.press/v164/rudin22a/rudin22a.pdf ; SHA-256 `420fc2ac08a7b3d04894e4ad36266e3c7cfc66f6fdc5662e5fcc2931715fdb10`.
- **Deciding locator and fact:** Section 3, Task Description; Figure 2 terrain types; pp. 3-7. The report treats locomotion terrain curriculum, not course structure.
- **Comparison rationale:** Both snapshots agree. Terrain patches do not meet the protocol’s required course topology or connected traversal constraint.

## C0192

- **candidate_id:** `C0192`
- **cite_key/title:** `CARLA2021OpenDRIVEStandalone` — *OpenDRIVE standalone mode*.
- **Raw sealed ratings / trigger IDs:** `A-C0192-02: included / include-1 / official_documentation`; `A-C0192-04: included / include-2 / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://carla.readthedocs.io/en/0.9.12/adv_opendrive/
- **Evidence version / retrieval / pin or digest:** CARLA 0.9.12 documentation; sealed retrieval `2026-06-30`; version-pinned documentation URL above; SHA-256 `2bd8dad73b459955ea79e620918737709031848069d767166a81435ac7a70aa9`.
- **Deciding locator and fact:** `#opendrive-standalone-mode`, `#run-a-standalone-map`, and `#mesh-generation`; `client.generate_opendrive_world()`. The API takes serialized OpenDRIVE road geometry and procedurally creates a navigable road mesh/boundaries.
- **Comparison rationale:** The documentation also defines a simulator interface (`include-2`), but procedural creation from explicit transferable course geometry is direct `include-1` evidence and has criterion precedence.

## C0193

- **candidate_id:** `C0193`
- **cite_key/title:** `QGroundControlNodatePlanFile` — *Plan File Format*.
- **Raw sealed ratings / trigger IDs:** `A-C0193-01: included / include-2 / official_documentation`; `A-C0193-02: included / include-1 / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-2`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://docs.qgroundcontrol.com/master/en/qgc-dev-guide/file_formats/plan.html ; https://raw.githubusercontent.com/mavlink/qgroundcontrol/master/docs/en/qgc-dev-guide/file_formats/plan.md
- **Evidence version / retrieval / pin or digest:** QGroundControl Plan schema v1 and Mission Object v2, `unversioned@2026-06-30`; sealed retrieval `2026-06-30`; no immutable archive URL supplied; SHA-256 `f4b20a832bf8bf1adc0362ec4b65e59eafaafca74e9156bbaa9aa0fe03b3d342`.
- **Deciding locator and fact:** Plan File, Mission Object, SimpleItem, Complex Mission Item, and CorridorScan; `plan.md` lines 1-44, 83-100, and 190-217. The schema serializes ordered mission items, geographic coordinates, and corridor-scan structures.
- **Comparison rationale:** It is a reusable aerial-route representation and design/interchange interface, exactly `include-2`. Mere serialization of supplied mission data does not show that the source generates courses, so `include-2` is more precise than `include-1`.

## C0198

- **candidate_id:** `C0198`
- **cite_key/title:** `RoboNation2026RoboBoat2026` — *RoboBoat 2026 | Team Handbook*.
- **Raw sealed ratings / trigger IDs:** `A-C0198-01: included / include-2 / official_documentation`; `A-C0198-02: included / include-1 / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-2`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://robonation.org/app/uploads/sites/3/2025/10/RoboBoat-2026-Team-Handbook_111125.pdf
- **Evidence version / retrieval / pin or digest:** RoboBoat 2026 Team Handbook initial release `2025-11-11`; sealed retrieval `2026-06-30`; version-pinned official PDF URL above; SHA-256 `ee93f4b9775a646657cde91b442f4ed2daedc6f68e3db54e04316bea8e3b2d69`.
- **Deciding locator and fact:** PDF pp. 45-50, Sections 3.2.1-3.2.3 and task-layout figures; pp. 58-60, Sections 3.2.7 and 3.4-3.5. The handbook defines an official multi-course set with entry/exit gates, buoy channels, debris constraints, and task layouts.
- **Comparison rationale:** The artifact defines a competition course set, expressly an `include-2` contribution. Variable elements do not themselves establish a source-native course-generation procedure needed for `include-1`.

## C0200

- **candidate_id:** `C0200`
- **cite_key/title:** `RoboNation2026RobotX2026` — *RobotX 2026 | Team Handbook*.
- **Raw sealed ratings / trigger IDs:** `A-C0200-01: included / include-2 / official_documentation`; `A-C0200-04: included / include-1 / full_text`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://robonation.org/app/uploads/sites/2/2026/06/RobotX-2026_Team-Handbook-20260625.pdf
- **Evidence version / retrieval / pin or digest:** RobotX 2026 Team Handbook revision `2026-06-25`; sealed retrieval `2026-06-30`; version-pinned official PDF URL above; SHA-256 `bbf6564f8b97ff1b8fa00c9f4306a3bc62e74ef5ff11ec8ec938a0441526f634`.
- **Deciding locator and fact:** Change Log pp. 6-8; Section 3.3.2 Mission Task 1 pp. 56-58; Section 3.5 pp. 70-76. The specification randomizes each run’s safe route through buoy gates and defines physical course constraints for autonomous craft.
- **Comparison rationale:** It also defines a competition course set (`include-2`), but source-native route randomization and ordered gate constraints directly satisfy the earlier `include-1` criterion.

## C0203

- **candidate_id:** `C0203`
- **cite_key/title:** `OpenRoboticsNodateSDFWorlds` — *SDF worlds*.
- **Raw sealed ratings / trigger IDs:** `A-C0203-01: excluded / exclude-out-of-scope / official_documentation`; `A-C0203-04: excluded / exclude-out-of-scope / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** The tutorial defines a generic simulation world and model placement, with no course-specific representation, generator, benchmark, metric, or simulator course interface.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://gazebosim.org/docs/harmonic/sdf_worlds/
- **Evidence version / retrieval / pin or digest:** Gazebo Harmonic documentation, `unversioned@2026-06-30`; sealed retrieval `2026-06-30`; no immutable archive URL supplied; SHA-256 `74e40502aa45901ec1066dd551632190ce7643ed2bb23113c27590d182c92da1`.
- **Deciding locator and fact:** SDF worlds; Defining a world; Adding models. Generic world composition is not a course artifact.
- **Comparison rationale:** Both snapshots agree. The protocol excludes generic simulator capability without a course-specific technical contribution.

## C0204

- **candidate_id:** `C0204`
- **cite_key/title:** `OpenSourceRoboticsFoundation2020PoseFrame` — *Pose Frame Semantics Tutorial*.
- **Raw sealed ratings / trigger IDs:** `A-C0204-02: excluded / exclude-out-of-scope / official_documentation`; `A-C0204-04: excluded / exclude-out-of-scope / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `excluded` / `exclude-out-of-scope`.
- **Exclusion reason:** The standard tutorial specifies generic relative-pose and coordinate-frame semantics, not course geometry, route serialization, a course generator, or a course-specific interchange artifact.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://sdformat.org/tutorials/specification/pose_frame_semantics/1.7/
- **Evidence version / retrieval / pin or digest:** SDFormat 1.7; sealed retrieval `2026-06-30`; versioned documentation URL above; SHA-256 `a05485cb9327fe7ee72020820a42d39dc93c36db6ce18a36525b366f013845ee`.
- **Deciding locator and fact:** What’s New in SDFormat 1.7; Pose and Frame Semantics; Frame semantics in nested models. The defined semantics are generic simulation primitives.
- **Comparison rationale:** Both snapshots agree. General coordinate-system semantics cannot be treated as a course interface merely because a course could be represented with poses.

## C0209

- **candidate_id:** `C0209`
- **cite_key/title:** `RoboNation2023VRX2023` — *VRX 2023 Task Descriptions*.
- **Raw sealed ratings / trigger IDs:** `A-C0209-02: included / include-2 / official_documentation`; `A-C0209-04: included / include-1 / full_text`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `full_text`.
- **Canonical source URL(s):** https://robonation.org/app/uploads/sites/2/2023/07/VRX2023_Task-Descriptions_v1.2.pdf
- **Evidence version / retrieval / pin or digest:** VRX 2023 Task Descriptions version 1.2, `2023-07-03`; sealed retrieval `2026-06-30`; version-pinned official PDF URL above; SHA-256 `f093ef26bb20f2e077eddd77f9ad153bfdd0c43fb41c92bcba174e6e125526fe`.
- **Deciding locator and fact:** pp. 2-4, random goal and waypoint tasks; pp. 9-10, Section 4.2 Follow the Path; Section 3.2 Wayfinding/Table 3; Section 3.6 Navigation Channel. The specification places random goals/waypoints and defines ordered colored-buoy gate channels with direction, width, traversal, and scoring constraints.
- **Comparison rationale:** The document also defines reusable competition interfaces (`include-2`), but random placement and explicit ordered gate-course constraints meet the earlier `include-1` criterion.

## C0210

- **candidate_id:** `C0210`
- **cite_key/title:** `OpenRoboticsNodateVRXAutomated` — *VRX Automated Evaluation*.
- **Raw sealed ratings / trigger IDs:** `A-C0210-01: included / include-1 / official_documentation`; `A-C0210-04: included / include-2 / official_documentation`; trigger IDs: none recorded.
- **Recommendation:** `included` / `include-1`.
- **Exclusion reason:** `NR`.
- **Access status:** `official_documentation`.
- **Canonical source URL(s):** https://github.com/osrf/vrx-docker/tree/f599871a83ddfef9851e2f9bc95d082baff47bf2 ; https://github.com/osrf/vrx-docker/blob/f599871a83ddfef9851e2f9bc95d082baff47bf2/README.md
- **Evidence version / retrieval / pin or digest:** commit `f599871a83ddfef9851e2f9bc95d082baff47bf2`; sealed retrieval `2026-06-30`; version-pinned repository URL above; SHA-256 `98a163bf8460da6c1a3222ad30035d0577d747ff9db7a5cc1b51728ac34c070c`.
- **Deciding locator and fact:** `prepare_task_trials.bash` lines 3-5 and 32-51; `task_config/gymkhana.yaml` lines 1-19, 46-69, 97-120, and 147-170; README lines 21-29 and 43-53. The repository generates task worlds from YAML trials and serializes ordered marker-gate navigation courses.
- **Comparison rationale:** The artifact also supplies competition evaluation interfaces (`include-2`), but generated task worlds and serialized gate courses are direct source-native `include-1` evidence.
