# Batch 01 Adjudication Draft

## C0073
Locked ratings: `A-C0073-01` = **excluded / `exclude-fixed-racing-line`**; `A-C0073-04` = **excluded / `exclude-out-of-scope`**.
Triggers: A2.
Evidence/locator analysis: C0073 received exclude-fixed-racing-line and exclude-out-of-scope, whereas both locked reviewers assessed excluded. The paper's gate-layout variation merely supports training a flight policy: it does not define how layouts are parameterized, serialized, validated, or made reusable. Its central method predicts waypoints and commands during traversal. Because the packet mentions layout variation yet lacks sufficient details for a survey claim, exclude-insufficient-detail is more precise than either raw criterion.
Final rationale: The report says training layouts are made by slightly moving gates, but it does not define a reusable course-generation procedure, representation, or retained interface beyond control of supplied race tracks.
- Recommendation: **excluded / `exclude-insufficient-detail`**

## C0098
Locked ratings: `A-C0098-03` = **excluded / `exclude-insufficient-detail`**; `A-C0098-04` = **excluded / `exclude-out-of-scope`**.
Triggers: A2.
Evidence/locator analysis: C0098 received exclude-insufficient-detail and exclude-out-of-scope, whereas both locked reviewers assessed excluded. The frozen About page supplies only licensing language declaring OpenStreetMap open data. It contains neither a data schema nor any documented route, road, map, level, or test-input construction. Since the packet establishes no eligibility-bearing technical property rather than an incomplete description of one, exclude-out-of-scope is the direct result.
Final rationale: The frozen About page states only that OpenStreetMap is open data under licensing terms and provides no map schema, route or road representation, generator, benchmark, or supporting transfer fact.
- Recommendation: **excluded / `exclude-out-of-scope`**

## C0131
Locked ratings: `A-C0131-02` = **excluded / `exclude-out-of-scope`**; `A-C0131-04` = **included / `include-relevant`**.
Triggers: A1, A2.
Evidence/locator analysis: C0131 received excluded with exclude-out-of-scope and included with include-relevant, whereas the frozen documentation directly specifies random asset selection, position ratios, and spawning with randomized obstacle positions within environment bounds. Those configurable spatial obstacles are a source-native adjacent level/test-input generator, not merely appearance randomization. Mapping the bounded obstacle layouts to aerial course geometry preserves the environment-configuration claim, so the documented generator satisfies retention.
Final rationale: The documentation specifies random asset selection, asset-position ratios, and an environment strategy that spawns and randomizes obstacle positions within configured bounds. Transferable mapping: source-native obstacle selection and bounded position randomization form adjacent-domain level/test-input generation that maps to parameterized aerial course geometry and obstacle constraints.
- Recommendation: **included / `include-relevant`**

## C0140
Locked ratings: `A-C0140-01` = **included / `include-relevant`**; `A-C0140-05` = **excluded / `exclude-insufficient-detail`**.
Triggers: A1, A2.
Evidence/locator analysis: C0140 received included with include-relevant and excluded with exclude-insufficient-detail, whereas the bound full report explicitly creates several obstacle terrains in increasing difficulty and evaluates ordered waypoint courses using displacement and edge-violation measures. This establishes an adjacent-domain course progression and reusable traversal metrics, rather than a bare mention of terrain. The complete frozen PDF contains the cited technical sections, so retained supporting evidence is sufficient.
Final rationale: The report creates ramps, gaps, hurdles, and high-step terrains in increasing difficulty, and evaluates waypoint obstacle courses with mean displacement and mean edge-violation metrics. Supporting mapping: its ordered obstacle-course difficulty progression plus mean displacement and mean edge violation map to course-generator difficulty and traversal-validity evaluation.
- Recommendation: **included / `include-relevant`**

## C0143
Locked ratings: `A-C0143-02` = **included / `include-relevant`**; `A-C0143-06` = **included / `include-relevant`**.
Triggers: A4.
Evidence/locator analysis: C0143's two locked reviewers both returned included and include-relevant, whereas unresolved conflict X57B57E64E501 records historical candidate versus excluded values that must be resolved atomically. The bound paper describes WGAN-based road-test generation and a feature representation of road geometry and curvature. That source-native adjacent road generator maps directly to parameterized robot-course centerlines, so the locked inclusion is retained and the coordinator conflict is resolved.
Final rationale: The report uses a Wasserstein generative adversarial network to generate road test cases from feature representations that encode road geometry and curvature. Adjacent-domain mapping: generated road point sequences and curvature representations map to parameterized robot-course centerlines.
- Recommendation: **included / `include-relevant`**

## C0187
Locked ratings: `A-C0187-02` = **excluded / `exclude-out-of-scope`**; `A-C0187-04` = **included / `include-relevant`**.
Triggers: A1, A2.
Evidence/locator analysis: C0187 received excluded with exclude-out-of-scope and included with include-relevant, whereas the scenario documentation says that a scenario loads a world, places agents, and may randomize only their start locations or rotations. It expressly leaves the underlying world or map unchanged. These agent and sensor settings do not supply an ordered marine course, a course representation, or a retained simulator-course property, so exclusion governs.
Final rationale: The scenario schema loads a pre-existing world and randomizes only agents' start location or rotation; it neither defines nor changes a route, course, gate, buoy, or course-generation interface.
- Recommendation: **excluded / `exclude-out-of-scope`**

## C0188
Locked ratings: `A-C0188-02` = **included / `include-relevant`**; `A-C0188-05` = **excluded / `exclude-out-of-scope`**.
Triggers: A1, A2.
Evidence/locator analysis: C0188 received included with include-relevant and excluded with exclude-out-of-scope, whereas the versioned source code defines a terrain generator that constructs mesh terrains, selects random or curriculum difficulty, and lays out subterrain by grid position. This is source-native adjacent level generation with explicit spatial representation and parameters. Mapping those meshes, origins, and difficulty values into a ground-racing course level preserves the technical claim, so inclusion is warranted.
Final rationale: TerrainGenerator creates terrain meshes from height fields or generator functions, samples or curricula their difficulty, and places subterrain meshes by row and column. Transferable mapping: generated terrain meshes, per-subterrain origins, and difficulty parameters map to a parameterized ground-racing course level without changing the terrain-generation claim.
- Recommendation: **included / `include-relevant`**

## C0203
Locked ratings: `A-C0203-01` = **included / `include-relevant`**; `A-C0203-04` = **excluded / `exclude-out-of-scope`**.
Triggers: A1, A2.
Evidence/locator analysis: C0203 received included with include-relevant and excluded with exclude-out-of-scope, whereas the Harmonic documentation defines an SDF world container and the procedure for adding models to that world. The resulting serializable simulator-world interface can represent placed corridor, gate, and obstacle elements without recasting the source as a generator. This is a concrete transferable representation property, so inclusion is supported.
Final rationale: The SDF tutorial defines a versioned world container and documents adding models to it, furnishing a serializable simulator-world interface for positioned course elements. Supporting mapping: SDF world and model definitions serialize a simulator course layout by placing corridor, gate, and obstacle models; it is a representation interface, not a generation method.
- Recommendation: **included / `include-relevant`**
