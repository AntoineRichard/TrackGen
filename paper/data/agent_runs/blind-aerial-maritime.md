# Blind aerial and maritime literature-discovery report

## Protocol and scope

This run was conducted from the supplied thesis statement and domain brief only. The destination file path and the repository name `TrackGen` were visible as operational context. No repository corpus, taxonomy, known-reference file or list, documentation or source paths, other agent outputs, or implementation details were supplied to this run or read, and `TrackGen` was not queried. Destination-path and repository-name visibility is a limitation even though repository content remained unread.

Primary papers, publisher/proceedings records, official project pages, official competition specifications, and official repositories were used as evidence. Search-result snippets were used only to discover sources. Every CSV `metadata_evidence` and `evidence_locator` resolves to primary or official material; unresolved facts are `NR`.

The strict in-scope unit was a source that generates, instantiates, or defines at least one of: ordered gates/waypoints, static obstacle-course geometry, buoy courses, port/waterway geometry, a course distribution, or a portable executable course/world representation used for aerial or maritime robot training/evaluation. Fixed-course planning/control, racing-line optimization, appearance-only randomization, dynamics-only randomization, and task replay/selection were not counted as geometry/course candidates.

## Result summary

- Unique candidates after DOI-then-title deduplication: **32**.
- Aerial: **17**. Maritime: **15**.
- Source mix: **24 research/system papers**, **2 official repositories**, and **6 official competition specifications**.
- DOI present: **23/32**. The nine `NR` DOI records are repositories/specifications or the Flightmare proceedings record.
- Code status after final availability re-audit: **17/32 `official_open`**, **9/32 `NR`**, and **6/32 `not_applicable`**.
- Reusable-asset status after final availability re-audit: **19/32 `official_open`** and **13/32 `NR`**.
- Every row has `screening_status=candidate` and `metadata_status=unverified`, as requested. `NR` is used rather than inference where a primary source did not report a field.

`official_open` is reserved for reusable code, configurations, models, environments, or data exposed by an official author or organization source. A paper, handbook, rules page, project video, illustration, or data-on-request statement is not counted as a reusable asset. Public simulator dependencies are identified separately from paper-specific generator availability. Undated absence judgments without row-specific official availability surfaces are coded `NR`, not `not_found`.

Coverage is intentionally non-exclusive:

| Coverage family | Candidate examples | Count |
|---|---|---:|
| Ordered gate/waypoint generation | random gate primitives; environment-as-policy; waypoint permutation; WPgen | 7 |
| Static obstacle/arena distributions | Air Learning; Aerial Gym; AvoidBench; obstacle-aware racing; differentiable physics | 13 |
| Portable simulator/world/task representation | AirSim; Flightmare; VRX; VRX automated evaluation; ASVSim | 10 |
| Official physical/virtual course families | VRX; RobotX; RoboBoat; WRSC; SailBot; Njord | 7 |
| Port or waterway geometry | randomized ports; ASVSim PCG; hydraulic meandering waterways | 3 |

The strongest direct generators are BAAM-A001, A003-A005, A007, M003-M004, M012-M015. Competition specifications are retained because they define bounded, repeatable course families, but they are coded as human-authored course instantiation rather than procedural generation.

## Deduplication

Deduplication was applied in this order:

1. Canonical DOI: lowercase, with `https://doi.org/` and `doi:` removed.
2. If DOI was absent, normalized title: lowercase Unicode-folded text with punctuation and whitespace removed.

No duplicate DOI or normalized title remains. The 2023 Aerial Gym precursor was collapsed into the 2025 journal/system record. Paper, implementation repository, automated-evaluation repository, and competition specification were kept as separate VRX records because they have different evidence roles and titles. Preprint DOI values are DataCite arXiv DOIs only where no final DOI was located.

## Search surfaces

- arXiv abstract/full-text pages and DataCite DOI links.
- IEEE Xplore and official IEEE conference metadata.
- Springer Nature, Nature, MDPI, ScienceDirect, PMLR, ACM/official conference program pages, and ASME records.
- Institutional primary copies from UZH RPG, NTNU Open, TU Delft/MAVLab, Chalmers, Simula, Microsoft Research, and official author project pages.
- Official repositories and documentation for AirSim Drone Racing Lab, Flightmare, Aerial Gym, Air Learning, AvoidBench, OmniDrones, VRX, VRX automated evaluation, ASVSim, and WPgen.
- RoboNation/RobotX, RoboBoat, WRSC/IRSC, SailBot, and Njord official rules/handbooks.
- General web search was used for discovery only; metadata/evidence was then resolved to one of the primary/official surfaces above.

## Exact query log

URL opens and DOI resolution are omitted; the strings below are the search queries used. Quoting and capitalization are preserved.

### Round 1: broad thesis terms

```text
"drone racing" randomized tracks gate sequence reinforcement learning
quadrotor "procedurally generated" obstacle course simulator
"Virtual RobotX" course generation buoy
autonomous surface vessel reinforcement learning random waypoint course buoys simulator
```

### Round 2: system and official competition surfaces

```text
site:microsoft.com AirSim Drone Racing Lab generation racing tracks paper
site:github.com/ntnu-arl aerial_gym procedurally generated environments paper
site:uzh.ch Flightmare random racing tracks gates simulator paper
site:arxiv.org quadrotor procedural obstacle environments agile flight random obstacles
site:openrobotics.org VRX Virtual RobotX paper course buoy world generation
site:github.com/osrf/vrx world generation buoy course task documentation
site:robotx.org rules course buoys RobotX Maritime Challenge PDF
site:robotx.org Virtual RobotX competition task descriptions buoy course
```

### Round 3: simulators, obstacle distributions, and competition families

```text
site:arxiv.org Flightmare flexible quadrotor simulator gates random environments
site:arxiv.org OmniDrones track task gates procedural environment
site:arxiv.org VisFly simulator drone navigation random obstacles waypoint
site:arxiv.org Air Learning deep reinforcement learning gym aerial robot random obstacle environments
OmniDrones efficient flexible platform reinforcement learning drone control track task official
VisFly efficient versatile simulator learning to fly official paper
AvoidBench benchmark aerial obstacle avoidance environment generation official paper
"AvoidBench" aerial robot benchmark obstacle generation
Learning Minimum-Time Flight in Cluttered Environments random obstacle environments official paper
site:rpg.ifi.uzh.ch "Learning High-Speed Flight in the Wild" random environment obstacles
site:robonation.org/app/uploads/sites/3 2026 RoboBoat team handbook task ideas
site:njordchallenge.com rules buoy course autonomous ship challenge PDF
SailBot competition rules autonomous sailboat buoy navigation race official PDF
World Robotic Sailing Championship rules buoy race autonomous sailboat official
```

### Round 4: direct-title and generator-family closure

```text
"Environment as Policy: Learning to Race in Unseen Tracks" ICRA 2025 paper
"Learning generalizable policy for obstacle-aware autonomous drone racing"
site:arxiv.org "obstacle-aware autonomous drone racing" generalizable policy
site:github.com "Learning generalizable policy" drone racing
"Learning generalizable policy for obstacle-aware" DOI
"MAVRL: Learn to Fly in Cluttered Environments with Varying Speed"
"Reinforcement Learning for Collision-Free Flight Exploiting Deep Collision Encoding"
"Learning vision-based agile flight via differentiable physics" random obstacles
"Learning speed adaptation for flight in clutter"
site:arxiv.org "Mastering Diverse, Unknown, and Cluttered Tracks"
"buoy course" generator autonomous surface vessel simulation
"buoy field" random generation USV simulator
"procedurally generated" waterways autonomous vessel
"scenario generation" autonomous surface vessel benchmark waypoints buoys
site:arxiv.org autonomous surface vessel random obstacle scenarios reinforcement learning waypoint distribution
autonomous surface vessel simulator procedurally generated environments reinforcement learning official paper
"ASVSim" AirSim Surface Vehicles GitHub official
"Search-based Generation of Waypoints for Triggering Self-Adaptations" official
```

### Formal saturation refinement 1

```text
site:arxiv.org quadrotor "randomly generated tracks" gates racing
site:ieeexplore.ieee.org drone racing "track generation" gates
site:github.com autonomous surface vessel simulator "generated_worlds" buoy waypoints
site:proceedings.mlr.press aerial robot "procedurally generated" obstacles navigation
```

### Formal saturation refinement 2

```text
"race track generator" quadrotor gates paper
"random gate tracks" drone racing
"autonomous sailboat simulator" buoy course official
"waterway generation" autonomous ship simulation paper
```

### Boundary audit queries

These queries tested whether maritime encounter generation or newer generic simulators concealed static course geometry. They produced no additional strict geometry/course candidate after screening.

```text
"autonomous vessel" "test scenario generation" AIS sampling paper
"dynamic test scenario generation" autonomous ship simulation
"randomly generated virtual test scenarios" maritime autonomous vessel
maritime "combinatorial test scenario generation" ship collision avoidance
"Ship Encounter Scenario Generation for Collision Avoidance Algorithm Testing Based on AIS Data"
"Generation of naturalistic and adversarial sailing environment" autonomous ships
"Interactive testing for evaluating the performance of collision avoidance algorithms in multi-ship encounter scenarios"
"Towards Simulation-based Verification of Autonomous Navigation Systems" DOI
"Automatic simulation-based testing of autonomous ships using Gaussian processes and temporal logic" authors
"Testbed Scenario Design Exploiting Traffic Big Data" DOI
"Assessing Scene Generation Techniques for Testing COLREGS-Compliance"
"Towards Automated Test Scenario Generation for Assuring COLREGs Compliance"
```

## Round accounting and saturation

The round ledger was rebuilt from each retained row's exact `discovery_query`, then screened and deduplicated by DOI and normalized title. A candidate is credited to the round containing that retained query. Search hits that lacked primary evidence, randomized only appearance/dynamics, or optimized control on a fixed course remain unsupported leads and are outside both the numerator and denominator.

| Round | Focus | New strict candidates | Cumulative unique | Yield calculation |
|---|---|---:|---:|---:|
| 1 | broad aerial/maritime thesis terms | 2 | 2 | 2/2 = 100.000% |
| 2 | simulator and official competition surfaces | 7 | 9 | 7/9 = 77.778% |
| 3 | obstacle distributions and course/task formats | 10 | 19 | 10/19 = 52.632% |
| 4 | direct-title, citation, and generator-family closure | 12 | 31 | 12/31 = 38.710% |
| 5 | formal refinement 1 | 0 | 31 | **0/31 = 0.000%** |
| 6 | formal refinement 2 | 1 | 32 | **1/32 = 3.125%** |

Candidate additions are reproducible against the query blocks above:

- Round 1: BAAM-A001; BAAM-A006.
- Round 2: BAAM-A002; BAAM-A008; BAAM-M001; BAAM-M002; BAAM-M003; BAAM-M004; BAAM-M005.
- Round 3: BAAM-A007; BAAM-A009; BAAM-A010; BAAM-A011; BAAM-A017; BAAM-M006; BAAM-M007; BAAM-M008; BAAM-M009; BAAM-M010.
- Round 4: BAAM-A003; BAAM-A004; BAAM-A005; BAAM-A012; BAAM-A013; BAAM-A014; BAAM-A015; BAAM-A016; BAAM-M011; BAAM-M012; BAAM-M013; BAAM-M014.
- Round 5: none.
- Round 6: BAAM-M015.

**BAAM-A012 provenance audit.** The literal discovery query remains `site:arxiv.org autonomous surface vessel random obstacle scenarios reinforcement learning waypoint distribution`. The original run context records the UAV paper as cross-domain search-result spillover from that Round 4 maritime query. It was screened into the aerial candidate set during Round 4 after title/abstract review; no alternate executed aerial query was retained for this candidate, so the query was not replaced. The separate `search_log` was not edited and may need an integrator update to mirror this clarification.

The stopping calculation is `new strict candidates / cumulative unique strict candidates after the round`. The final two formal refinements are consecutive and below 5%: Round 5 produced `0/31`, then Round 6 produced `1/32 = 3.125%`. A later boundary audit found dynamic ship-encounter generators but no additional static geometry, waypoint, buoy, gate, port, or waterway candidate; those unsupported or out-of-scope leads remain outside the saturation denominator.

## Boundary judgments

- **Geometry versus appearance:** Air Learning and AvoidBench expose both obstacle geometry and texture/material controls. They are included for geometry; appearance knobs are not coded as course generation.
- **Geometry versus dynamics:** wave, wind, sensor noise, vessel dynamics, motor variation, and moving-ship behavior are not geometry. They appear only where coupled to an independently generated course.
- **Dynamic maritime encounter generation:** AIS sampling, COLREG scene generation, adversarial target ships, Gaussian-process falsification, and multi-vessel initial-state generation were audited but excluded from the strict set when they did not generate static course/waterway geometry or waypoint routes. Examples include Pedersen et al. (2020), Bolbot et al. (2022), Torben et al. (2023), Kargen and Varro (2024), Frey et al. (2025), and Chen et al. (2025).
- **Fixed-course planning/control:** trajectory optimization, racing-line optimization, collision avoidance on a hand-fixed map, and control-policy work without a reported course distribution were excluded. The generated geometry, not the downstream solver, determines inclusion.
- **Fixed track plus randomized obstacles:** BAAM-A006 is included but explicitly coded as obstacle placement and bounded gate perturbation, not full track synthesis.
- **Simulator host versus generator:** Flightmare is retained as a course/scene host because downstream racing generators operate through its object interface; no native stochastic track-generator claim is made. AirSim Drone Racing Lab is stronger because its paper explicitly states track generation.
- **Competition specifications:** VRX, RobotX, RoboBoat, WRSC, SailBot, and Njord are included as bounded course-family definitions. They are human-authored and sometimes weather-conditioned, not probabilistic generators.
- **Operator training transfer:** BAAM-M013 directly generates waypoint chains from empirical distributions but trains remote operators, not robot policies. It is retained as a transferable generator and marked accordingly.
- **Task selection/replay:** choosing, prioritizing, or replaying an existing level without changing course geometry was excluded.

## Terminology found outside the brief

- **Random track curriculum:** gradually increasing gate-chain length/complexity.
- **Environment as policy / adaptive environment shaping:** a learned secondary policy changes the task geometry.
- **Track primitive generator:** compositions of recurring gate-layout motifs such as circular, U-shaped, or zigzag tracks.
- **Gymkhana / Follow the Path:** VRX term for a colored buoy-pair gate chain.
- **Red, Right, Returning:** direction convention for buoy-gate traversal.
- **Wayfinding PoseArray:** unordered WGS84 goal poses with headings exposed over ROS 2.
- **World xacro:** a parameter-expanded Gazebo world representation generated from task YAML.
- **Poisson radius:** a clutter/traversability proxy used in aerial obstacle fields.
- **Soft-collision / hard-collision phase:** curriculum phases that first preserve exploration and later enforce physical collision termination.
- **Safe-margin obstacle generator:** obstacle placement constrained away from gates or traversable corridors.
- **Functional, logical, and concrete scenario:** levels used in maritime test generation; screened out when they only specified moving-vessel encounters.

## Sparse and contradictory areas

- **Maritime stochastic buoy geometry is sparse.** VRX has executable world generation, but physical RobotX/RoboBoat/WRSC/SailBot/Njord courses are mostly bounded hand-authored layouts. Published random buoy-chain algorithms were not located.
- **Validity is underreported.** Relative-pose bounds, safe margins, simulator collisions, and environment bounds are common; explicit self-intersection, global connectivity, and guaranteed traversability checks are rare.
- **Distribution size is often absent.** A few aerial papers report hundreds of thousands of environments or hundreds/thousands of test tracks. Most simulator and competition sources report tasks, not the cardinality or entropy of the course distribution.
- **Diversity metrics are weak.** Success on named unseen layouts is common, while geometric novelty, coverage, dispersion, and topology diversity are seldom quantified.
- **Difficulty mixes causes.** Gate count, obstacle density, clearance, sensor noise, dynamics uncertainty, and policy performance are frequently collapsed into one curriculum level. The CSV separates geometry and difficulty fields where sources allow it.
- **Export formats are not standardized.** VRX has the clearest path from YAML to `.world`/`.world.xacro`. Aerial work commonly leaves tracks as in-memory pose arrays or simulator-specific Python configuration.
- **Accepted-preprint metadata can conflict with final publication metadata.** Mastering Diverse Tracks remains an accepted-preprint record with final publisher metadata unverified. ASVSim is reconciled to its 2026 IEEE Access record and DOI `10.1109/ACCESS.2026.3687084` using the official repository and DOI record.
- **Living competition documents change.** RobotX 2026 and Njord 2026 were current on 29 June 2026 but may change before their events.

## High-priority retrievals

1. Final IEEE Robotics and Automation Letters record, DOI, supplement, and generator code for **Mastering Diverse, Unknown, and Cluttered Tracks**.
2. Version-pinned Unreal PCG parameter schema and reproducible release artifact for **ASVSim**; final IEEE Access metadata and DOI are now resolved.
3. Full supplement or released code for **Environment as Policy**, especially environment-action bounds and any global gate-layout validity checks.
4. Paper-specific track-generator release for **Autonomous Drone Racing with Deep Reinforcement Learning**, distinct from the public Flightmare simulator.
5. Exact object/course serialization and portability guarantees in **Flightmare** and **AirSim Drone Racing Lab**.
6. Full official proceedings text for the 2015 **World Robotic Sailing Championship** paper to recover edition-specific dimensions and evaluation counts.
7. Version-pinned export of the living **Njord 2026** GitBook and final RobotX 2026 task/course appendix after organizer freeze.
8. Full OMAE text and supplements for **Autonomous Port Navigation With Ranging Sensors Using Model-Based Reinforcement Learning**; only the primary abstract was retrieved, so simulator, export, validity, cadence, and route-randomization details remain `NR`.
9. Full methods and supplements for **Sim-to-Real Deep Reinforcement Learning based Obstacle Avoidance** and **Learning Speed Adaptation for Flight in Clutter**.
10. Detailed generated-map counts, seeds, and feasibility checks for **MAVRL** and **Deep Collision Encoding**.
11. Final venue/code status for the 2026 **Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles** preprint.
12. The June 2026 **MuJoCo-Drones-Gym** gate-racing task implementation, to determine whether it supports random gate geometry or only a fixed demonstration course; it was not promoted without that evidence.

## Files and validation

- `blind-aerial-maritime.csv`: 32 candidate rows plus the exact 35-column header.
- `blind-aerial-maritime.md`: this protocol, query, screening, saturation, and retrieval report.

Post-write validation uses a standards-compliant CSV parser and checks the literal header, 35 columns on every row, exact `; ` list separators, scalar status vocabularies, sole-`NR` technical fields, candidate/domain counts, DOI and normalized-title uniqueness, and complete primary/official URLs in every evidence locator. The manual evidence audit removed unsupported technical details from BAAM-A012, A016, M007, and M011 where full text was not retrieved. The final availability audit adds official repository evidence for BAAM-A002, BAAM-A007, and BAAM-M012 and replaces unsupported undated absence judgments with `NR`. Validation reads only these two destination files.
