# Aware Simulation and Benchmark Discovery Report

Date: 2026-06-29

## Scope and evidence policy

This agent run expands the survey corpus around aerial, maritime, and ground course representations; simulation and competition benchmarks; simulator map/course formats; and mainstream RL interfaces. The seed files were used as maps to known material, not as scope boundaries.

Rows were admitted only when a primary paper/proceedings record, standard, official project documentation, official competition document, or official repository directly supported the system and the coded claim. Search snippets, aggregators, blogs, and Wikipedia were lead-generation surfaces only and do not appear as `metadata_evidence` or `evidence_locator`.

The output contains 30 unique candidate rows (`ASIM0001` through `ASIM0030`). All rows remain `screening_status=candidate`; boundary status is carried in `generation_role` and `coding_notes` rather than replacing the requested screening status. Three rows are direct bootstrap-seed systems or standards. F1TENTH Gym and Isaac Lab are newly discovered from bootstrap lineages rather than exact seed records. The remaining rows are newly discovered or newly verified supplied targets. Each row states its seed/new status in `coding_notes`.

## Local files read

The following local files were read completely before discovery and coding:

- `docs/superpowers/specs/2026-06-29-track-generation-survey-design.md`
- `docs/tutorials/gate-sequences.rst`
- `docs/superpowers/specs/2026-06-23-gate-sequence-generation-design.md`
- `docs/related-work/state-of-the-art.rst`
- `docs/related-work/prior-art.rst`
- `docs/generators/benchmarks.rst`
- `paper/data/README.md`
- `paper/data/taxonomy.json`
- `paper/data/candidates.csv`
- `paper/data/seed_coverage.csv`

The candidate and seed-coverage CSVs were used as seed maps, never as scope boundaries.

## Search surfaces

The following surfaces were searched or followed to authoritative records:

- Local design/specification corpus, related-work pages, generator benchmark notes, `paper/data/candidates.csv`, and `paper/data/seed_coverage.csv`.
- Official project documentation: Gymnasium/Farama, F1TENTH, CARLA, Isaac Lab, AirSim, HoloOcean, Gazebo, SDFormat, SUMO, CommonRoad, Lanelet2/ROS, FSDS, PyFlyt, QGroundControl, MAVLink, PX4, and ArduPilot.
- Official source and asset repositories on GitHub or GitLab: Farama Foundation, F1TENTH, CARLA, ASAM publication mirrors, CommonRoad, Lanelet2, Eclipse SUMO, Isaac Lab, Microsoft AirSim, NTNU ARL, PyFlyt, gym-auv, BYU HoloOcean, OSRF VRX, Gazebo, AMZ Racing, EUFS, and TUMFTM.
- Standards: ASAM OpenDRIVE, OpenSCENARIO XML, and OpenCRG; SDFormat specification and frame-semantics documentation.
- Primary publication surfaces: PMLR, Frontiers, arXiv, official author/project PDFs, and DOI/IEEE metadata linked by official projects.
- Official competition documentation: RoboNation VRX and Maritime RobotX handbooks/technical guides.
- Official package metadata: PyPI project page and release files for `random-track-generator`.

Search-engine results were used to reach these sources, but no result snippet was treated as row evidence.

## Exact queries by round

These are the recorded discovery/refinement query strings. Direct URL opening, repository navigation, and in-document searching followed each query and are not additional search queries.

### Round 1: named-system verification

```text
site:gymnasium.farama.org CarRacing-v3 random track reset domain_randomize official
site:f1tenth-gym.readthedocs.io map yaml reset poses Gymnasium official
site:carla.readthedocs.io OpenDRIVE standalone mode coordinate system spawn points official
site:asam.net OpenDRIVE coordinate systems units validation official standard
site:isaac-sim.github.io/IsaacLab terrain generator curriculum difficulty USD Gymnasium reset
site:microsoft.github.io/AirSim drone racing gates coordinate system NED simSpawnObject simLoadLevel reset
site:github.com/osrf/vrx world generation yaml SDF competition official
site:robotx.org Maritime RobotX Challenge rules buoy course official PDF
```

### Round 2: official citation and format chain

```text
site:github.com/f1tenth f1tenth_gym maps yaml centerline raceline official repository
site:github.com/f1tenth f1tenth_racetracks centerline.csv raceline.csv
site:github.com TUMFTM racetrack-database centerline width raceline official
site:github.com TUMFTM laptime-simulation track CSV GeoJSON official
site:leaderboard.carla.org routes xml scenarios json benchmark official
site:asam.net OpenSCENARIO route trajectory coordinate system file format official
site:asam.net OpenCRG road surface curved regular grid format official
site:commonroad.in.tum.de scenario XML lanelet format scenario designer OpenDRIVE official
site:docs.ros.org lanelet2 map format OSM coordinates units official
```

### Round 3: aerial and maritime terminology expansion

```text
USV reinforcement learning environment random waypoint course generator official GitHub paper
gym-auv official GitHub randomized paths obstacles autonomous underwater vehicle environment
quadrotor waypoint Gymnasium environment random targets official repository
Aerial Gym Simulator procedural environments obstacle generation official GitHub paper
marine simulator scenario JSON waypoint RL interface official HoloOcean
site:robonation.org VRX 2022 Technical Guide world course randomization Gazebo
```

### Round 4: serialization, road-network, and cone-course expansion

```text
site:sumo.dlr.de/docs netgenerate random road network XML official
site:commonroad.in.tum.de tutorial map conversion OpenDRIVE Lanelet SUMO Scenario Designer
site:gazebosim.org SDF world pose collision spawn reset ROS 2 Simulation Interfaces official
site:sdformat.org spec world pose frame units collision geometry SDF official
Formula Student driverless simulator track CSV cones official GitHub EUFS simulator
site:github.com/QUT-Motorsport/eufs_sim track csv cone map generator official
site:fs-driverless.github.io Formula Student Driverless Simulator track format cones AirSim official
site:github.com/AMZ-Racing/fssim track YAML cone format map official
Formula Student random track generator Voronoi FSSIM YAML FSDS CSV GPX official
```

### Round 5: focused FSSIM/FSDS refinement

```text
site:github.com/AMZ-Racing/fssim track YAML cone format map official
site:github.com/AMZ-Driverless/fssim "track" ".yaml" cones
site:fs-driverless.github.io "track spline" "CSV" FSDS
site:fs-driverless.github.io track selection cones reset collision FSDS
```

### Round 6: portable waypoint/course-format expansion

```text
site:docs.qgroundcontrol.com plan file format mission items geofence rally points JSON
site:mavlink.io mission protocol waypoint coordinate frames official
site:ardupilot.org planner waypoint file format WPL 110 official
site:docs.px4.io mission file format QGroundControl plan official
site:docs.ros.org nav_msgs Path message header poses frame_id official
site:ogc.org KML standard waypoint route course official
site:mavlink.io/en/file_formats mission plain text QGC WPL 110 official
site:ardupilot.org/copter/docs common-planning-a-mission-with-waypoints-and-events save load mission official
```

Six candidate families or terminology branches were screened; admission required primary/official evidence:

| Candidate family | Evidence/result screened | Decision |
|---|---|---|
| QGroundControl Plan File Format | QGC developer file-format specification, Plan View documentation, official repository | Added as `ASIM0030`; versioned JSON bundle directly stores mission items plus optional geofence and rally points. |
| MAVLink Mission Protocol | MAVLink Mission (Plan) Protocol | Not added separately; this is a wire protocol used by QGC plan items, not another serialized course bundle. |
| MAVLink QGC WPL 110 plain text | MAVLink official File Formats page | Not added; the official page calls it an older de facto format outside the MAVLink standard and it omits geofence/rally bundles. |
| ArduPilot/PX4 mission planning | ArduPilot and PX4 official mission documentation | Not added; these are consumers/execution semantics for MAVLink missions, not a distinct portable format supported by the retrieved pages. |
| ROS `nav_msgs/Path` | Official ROS message definition | Not added; it is a runtime header plus pose array with no standalone serialization, load validation, reset, collision, or course-bundle contract. |
| OGC KML route/course terminology | OGC-focused query | Not added; no official course-specific execution or validation contract was located. |

Round 6 therefore added one supported unique candidate from a prior total of 29: `1 / 29 = 3.45%`.

The earlier Gymnasium/PettingZoo API checks remain useful interface evidence for existing rows, but they are explicitly not counted as an expansion or saturation round. Their exact non-expansion queries were:

```text
site:gymnasium.farama.org api env reset seed options terminated truncated official
site:gymnasium.farama.org vectorize custom environment official
site:pettingzoo.farama.org api parallel reset seed multi agent official
```

## Seed and citation-chain expansion

The three direct bootstrap matches are:

- `ASIM0001` Gymnasium Car Racing, exact seed C0017.
- `ASIM0008` ASAM OpenDRIVE, exact seed C0007.
- `ASIM0016` AirSim Drone Racing Lab, exact seed C0025.

`ASIM0002` F1TENTH Gym is newly discovered from bootstrap lineage: C0083 and C0084 are F1TENTH ecosystem surveys, not the exact F1TENTH Gym record. `ASIM0015` Isaac Lab Terrain Generator is likewise not the same source as Isaac Gym seed C0028. CARLA, Gazebo/VRX, RobotX, and other explicitly requested systems were supplied targets but were not marked as bootstrap sources when the seed CSV did not contain the exact record.

The main citation/terminology chains were:

- F1TENTH Gym -> official racetrack assets -> TUM track/raceline database -> fixed-track lap-time optimization boundary.
- CARLA OpenDRIVE ingestion -> ASAM OpenDRIVE -> OpenSCENARIO XML and OpenCRG -> CARLA Leaderboard route/scenario serialization.
- OpenDRIVE/map interchange -> CommonRoad -> Scenario Designer -> Lanelet2 -> SUMO `netgenerate`.
- AirSim drone racing -> Aerial Gym and PyFlyt waypoint-task interfaces.
- VRX/RobotX -> gym-auv procedural underwater paths and HoloOcean fixed-world scenario serialization.
- Gazebo/VRX -> SDFormat world and pose-frame semantics.
- Formula Student simulators -> FSDS, FSSIM, EUFS fixed cone maps, and the supported `random-track-generator` package.
- Portable mission/course terminology -> QGroundControl JSON Plan files -> MAVLink mission items, geofences, rally points, ArduPilot/PX4 mission consumers, and ROS path-message boundary screening.

## Inclusion and boundary judgments

Actual generated course/path geometry is directly supported for:

- `ASIM0001`: reset-time randomized closed-track geometry in Car Racing.
- `ASIM0014`: parameterized and seeded grid, spider, and random SUMO road-network generation.
- `ASIM0019`: randomized 3-D waypoint/reference paths in gym-auv, separate from obstacle/current variation.
- `ASIM0029`: seeded Formula Student cone-track geometry with FSSIM/FSDS/GPX exporters.

Generated geometry adjacent to, but not itself an ordered course, is coded as a boundary case:

- `ASIM0015` generates heightfield/triangle-mesh terrain patches.
- `ASIM0017` generates/randomizes obstacle worlds and simulation conditions.

Fixed maps, authored routes, or benchmark selection are not coded as geometry generation:

- F1TENTH Gym and Racetracks, CARLA Leaderboard, CommonRoad, VRX/RobotX, FSDS, FSSIM, and EUFS Maps select/replay authored geometry or task layouts.
- CARLA standalone mode constructs a simulator mesh from an externally supplied XODR network; this is serialization/realization, not road-network synthesis.
- AirSim permits caller-driven gate object placement and provides fixed competition levels; the retrieved official sources do not document a random gate-chain generator.
- HoloOcean scenarios place agents/sensors/tasks into a packaged world and explicitly do not alter world geometry.
- `ASIM0030` QGroundControl Plan files serialize ordered waypoint/mission items with optional geofence and rally points; they do not define a course-geometry sampler or simulator collision world.

Other important boundaries:

- Car Racing color `domain_randomize`, VRX wind/waves/fog/light, FSSIM cone-sensor noise, and Aerial Gym sensor/dynamics variation are not counted as geometric course generation.
- TUM Lap Time Simulation optimizes/evaluates on an external fixed track. Racing-line optimization does not generate the course envelope.
- OpenDRIVE, OpenSCENARIO XML, OpenCRG, Lanelet2, SDFormat, and CommonRoad XML are representations/interchange formats unless paired with a separately evidenced generator.
- Generic Gymnasium and PettingZoo APIs define episode/multi-agent contracts, not course distributions.

## Terminology not present in the supplied brief

Discovery required terms that are easy to miss when searching only for "track generator" or "course generator":

- `lanelet`, `lanelet2`, `lane boundary`, `regulatory element`, and `map projector`.
- `xodr`, `xosc`, `OpenCRG`, `curved regular grid`, and `reference line`.
- `netgenerate`, `net.xml`, `spider network`, and `random abstract network`.
- `SDF world`, `SDFormat`, `relative_to`, `frame semantics`, `world asset`, and `entity spawn`.
- `scenario designer`, `scenario database`, `route XML`, `trajectory`, and `goal region`.
- `occupancy map`, `map YAML`, `PNG/PGM`, `origin`, and `resolution`.
- `cone map`, `track spline`, `cone covariance`, `FSSIM`, `FSDS`, and `EUFS`.
- `reference path`, `flight dome`, `waypoint environment`, `buoy course`, `wayfinding`, and `task region`.
- `autoreset`, `terminated`, `truncated`, and `Parallel API` for RL interface compatibility rather than geometry.
- `Plan file`, `.plan`, `mission item`, `MAV_FRAME`, `geofence`, `rally point`, `QGC WPL 110`, and `nav_msgs/Path` for portable or runtime waypoint representations.

## Sparse and contradictory areas

- "Procedural" is overloaded. CARLA procedurally realizes an existing OpenDRIVE network; Isaac Lab procedurally creates terrain; Car Racing and SUMO actually sample course/network geometry. Those claims are not interchangeable.
- Coordinate conventions are heterogeneous: CARLA is left-handed x-forward/y-right/z-up; AirSim racing APIs use NED world coordinates; Gazebo/SDFormat uses scoped frame semantics; F1TENTH depends on map metadata; physical RobotX documents dimensions but not a machine-readable global course frame.
- Reset semantics differ across current Gymnasium, legacy OpenAI Gym integrations, simulator world resets, competition task resets, and physical course redeployment. No common reset contract can be inferred.
- Aerial and maritime systems commonly serialize an entire simulator level/world or runtime object placements. QGroundControl supplies a portable waypoint/mission bundle, but no canonical cross-simulator gate/buoy world bundle was supported.
- VRX has legacy evidence of YAML/Xacro world generation, but the current official branch/documentation did not provide a stable versioned public schema during this run. The row therefore records the current exact schema as `NR`.
- Isaac Lab terrain documentation and Isaac Sim/OpenUSD stage documentation change by release. Stage units, up-axis, persistence, collision filtering, and reset behavior were not merged across unmatched versions.
- F1TENTH and Formula Student map repositories expose useful assets, but ancillary centerline/raceline/track schemas are not uniformly documented for every file/version.
- Source availability is not inferred from a paper's silence. `official_open` is used only where an official repository/package was directly located; unresolved asset or library status is `NR`.
- Several official documentation projects do not publish an archival DOI/year for the documentation itself. Those metadata fields remain `NR` rather than being guessed from repository history.
- ASAM rows are now version matched: `ASIM0008` uses only OpenDRIVE 1.8.1 technical sections and its 2024-11-21 release page; `ASIM0009` uses only OpenSCENARIO XML 1.3.0 release/trajectory evidence. No inheritance from 1.9.0 or 1.4.0 is asserted.
- Public standards/specification documents are not treated as reusable course assets. Asset status for OpenDRIVE, OpenSCENARIO XML, OpenCRG, SDFormat, `random-track-generator`, and the QGC Plan format is `NR` unless a reusable course asset was directly located.

## Saturation arithmetic

Unique counts were deduplicated first by normalized non-`NR` DOI and otherwise by normalized title.

| Round | Focus | New supported unique candidates | Prior unique total | Yield | Cumulative total |
|---|---|---:|---:|---:|---:|
| 1 | Named-system verification | 8 | 0 | N/A (bootstrap denominator is zero) | 8 |
| 2 | Official citation/format chain | 8 | 8 | 8 / 8 = 100.00% | 16 |
| 3 | Aerial/maritime terminology expansion | 5 | 16 | 5 / 16 = 31.25% | 21 |
| 4 | Serialization, network, and cone-course expansion | 8 | 21 | 8 / 21 = 38.10% | 29 |
| 5 | Focused FSSIM/FSDS refinement | 0 | 29 | 0 / 29 = 0.00% | 29 |
| 6 | Portable waypoint/course-format expansion | 1 | 29 | 1 / 29 = 3.45% | 30 |

Round membership was recorded as follows:

- Round 1: `ASIM0001`, `ASIM0002`, `ASIM0006`, `ASIM0008`, `ASIM0015`, `ASIM0016`, `ASIM0021`, `ASIM0023`.
- Round 2: `ASIM0003`, `ASIM0004`, `ASIM0005`, `ASIM0007`, `ASIM0009`, `ASIM0010`, `ASIM0011`, `ASIM0013`.
- Round 3: `ASIM0017`, `ASIM0018`, `ASIM0019`, `ASIM0020`, `ASIM0022`.
- Round 4: `ASIM0012`, `ASIM0014`, `ASIM0024`, `ASIM0025`, `ASIM0026`, `ASIM0027`, `ASIM0028`, `ASIM0029`.
- Round 5 refined already supported records and added no row.
- Round 6 added `ASIM0030` after screening six portable/runtime waypoint-format candidate families.

At the Round 6 prior total of 29, the 5% threshold is `29 * 0.05 = 1.45`; therefore either zero or one new candidate is below 5%. Round 5 added zero (`0 / 29 = 0.00%`) and genuine expansion Round 6 added one (`1 / 29 = 3.45%`). These are two consecutive documented expansion/refinement rounds below 5%, so the required stopping criterion is satisfied and discovery stopped after Round 6 with 30 candidates.

## Cross-stream conflict handling

Three rows are independent observations of systems also observed by other discovery streams:

- `ASIM0001` Car Racing.
- `ASIM0008` ASAM OpenDRIVE 1.8.1.
- `ASIM0011` CommonRoad.

They remain in this agent-local output. Task 5 must reconcile metadata, evidence, and coding conflicts across streams; this agent did not alter any other stream's files.

## Review-driven row changes

All rows received canonical semicolon-plus-space and direct-locator normalization. Substantive claim, sentinel, version, asset, or retrieval-note changes were made to `ASIM0001`-`ASIM0006`, `ASIM0008`-`ASIM0012`, `ASIM0015`-`ASIM0023`, and `ASIM0025`-`ASIM0029`. `ASIM0030` is the sole new row. Unsupported subfacts were removed or represented by a whole-cell `NR`; no cell combines `NR` with another value.

## High-priority manual retrieval

The six pending leads were not added because no primary/official locator directly supported a complete row during this run:

1. The working-note Song paper lead: retrieve the authoritative proceedings/DOI record and full text that explicitly states course representation, generation role, export, and simulator integration.
2. The working-note Liu paper lead: retrieve the authoritative proceedings/DOI record and full text, especially the distinction between generated geometry and fixed-track control.
3. Isaac Sim/OpenUSD as a standalone format/interface row: retrieve one version-matched official source covering stage `metersPerUnit`, up-axis, collision schemas, asset persistence, world reset, and RL handoff.
4. TORCS: retrieve an official maintained repository or project manual that directly documents the track XML schema, coordinate/units convention, collision mesh construction, start grid/reset, and loader validation.
5. Speed Dreams: retrieve official versioned source/documentation for its track format and generation/editor behavior rather than relying on community descriptions.
6. Generic simulator course bundles beyond the supported QGroundControl mission plan: identify an official versioned JSON/NPZ schema and loader with frame, unit, collision, spawn/reset, and validation contracts. No generic simulator-format row is justified by an ad hoc file example.

Supported rows that still need targeted source retrieval rather than a new candidate include:

- PyFlyt's version-matched waypoint sampler and seed path.
- The current VRX randomized-world/configuration schema and validation path.
- A per-track inventory of F1TENTH Racetracks ancillary centerline/raceline files.
- OpenCRG C/MATLAB API source/license status.
- HoloOcean world-package asset license and coordinate/load-validation details.
- FSSIM's stable versioned track-schema documentation.

These are coded `NR` and do not block inclusion of the directly supported system-level records.

## Verification status and limitations

- CSV rows: 30.
- Candidate IDs: continuous from `ASIM0001` to `ASIM0030`.
- Metadata status: all 30 rows are `verified` against at least one primary/official locator for the title/system identity; unreported author/year/DOI details remain `NR`.
- Evidence policy: every row has non-empty `metadata_evidence` and `evidence_locator`; no search snippet is cited as evidence.
- Serialization/interface detail is uneven. Where an official source did not directly state frames, units, collision geometry, spawn/reset, validation, RL API, code, or asset status, the cell or subfield is `NR` and the missing retrieval target is named.
- Compatibility claims are limited to directly documented consumers. No cross-simulator compatibility was inferred from file extensions alone.
- The report does not assert that every simulator version behaves identically to the cited documentation version.

Final validation checks the exact 35-column header, uniform row width, required fields, continuous/unique IDs, controlled labels, whole-cell `NR` semantics, semicolon-plus-space list formatting, direct evidence locators, and normalized DOI/title duplicates. The fresh command result is reported in the task completion response.
