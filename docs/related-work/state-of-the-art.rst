Track Generation State of the Art
==================================

This note focuses only on track and road-like geometry generation. It excludes most
downstream autonomous-racing papers unless their course representation or generator is
directly relevant. The organizing question is: how have prior systems created valid
race tracks, road centerlines, or 3D gate courses, and where do curve-thickness,
tangent-point/Sobolev methods, and PBD/XPBD-style constraint projection fit?

Constructive Segment, Tile, and Road-Block Generators
------------------------------------------------------

The most common family builds tracks from prevalidated parts. A generator chooses
straights, turns, roundabout pieces, road blocks, gates, or grid tiles and connects them
under local compatibility rules. The advantage is robustness: validity is mostly baked
into the grammar or component library. The limitation is that the generated geometry is
strongly shaped by the available parts, and invalid layouts are usually rejected or
rerouted rather than repaired as continuous curves.

- **TORCS / Speed Dreams track tooling and the Interactive Track Generator for TORCS and
  Speed Dreams.** Speed Dreams and TORCS represent tracks as segment lists: straights and
  left/right turns, plus main track, sides, borders, and barriers. The automated
  generator described in Speed Dreams documentation applies evolutionary computing /
  genetic programming to generate TORCS/Speed Dreams track outlines. This is a direct
  racing-game precedent, but the geometry model remains segment-based and editor-like.
  Link: `<https://en.wikipedia.org/wiki/Speed_Dreams>`__

- **Barthet et al., "Closing the Affective Loop via Experience-Driven Reinforcement
  Learning Designers" (2024).** This is a recent racetrack PCG paper for the Solid Rally
  game. Tracks are strings of track-component IDs placed on a 2D tile grid. A genetic
  algorithm or RL designer searches for layouts whose predicted arousal trace matches a
  target. Feasibility is handled through grid collisions and shortest-path closure with
  Dijkstra when needed. It is a strong example of modern racing PCG, but the geometry is
  discrete and affect-driven. Link: `<https://arxiv.org/abs/2408.06346>`__

- **Li et al., "Improving the Generalization of End-to-End Driving through Procedural
  Generation" / PGDrive (2020).** PGDrive generates driving scenes by sampling and
  connecting elementary road blocks, then adds traffic and simulation state. Its purpose
  is RL generalization rather than racetrack geometry, but it is a key example of road
  generation by construction. Link: `<https://arxiv.org/abs/2012.13681>`__

- **Ikram, Muktadir, Whitehead, "Procedural Generation of Complex Roundabouts for
  Autonomous Vehicle Testing" (2023).** This work procedurally constructs roundabouts
  and connecting roads for HD-map / OpenDRIVE-style autonomous-vehicle testing. It uses
  geometric construction around a maximal circle, connects incident roads, and injects
  irregularity. The core idea is again constructive validity: generate a road primitive
  from parameterized components rather than relax an arbitrary centerline.
  Link: `<https://arxiv.org/abs/2303.17900>`__

Evolutionary and Metaheuristic Race-Track Generators
-----------------------------------------------------

This family is closer to game PCG than to robotics. The generator exposes a track
representation, then uses evolutionary or swarm-style search to optimize playability,
style, challenge, or designer objectives. The important distinction is that invalid or
low-quality tracks are selected against; the generator does not usually contain an
explicit geometric repair operator for fixed-width offset validity.

- **Loiacono, Cardamone, Lanzi, "Automatic Track Generation for High-End Racing Games
  Using Evolutionary Computation" (2011).** This is a central missing reference for
  racing-game track generation. It uses evolutionary computation to synthesize tracks
  for high-end racing games, and is tied to the TORCS / Speed Dreams style of segment-
  and simulator-oriented racing content. It is valuable prior art because it treats race
  tracks as the object being optimized, not just as RL levels. The method remains a
  search/generation pipeline, not a curve-thickness repair solve.
  Links: `<https://doi.org/10.1109/TCIAIG.2011.2163692>`__ and
  `<https://www.semanticscholar.org/paper/e2633f542e8e9b673e98c68d0e6e6c6d13fe5ed0>`__

- **Togelius, De Nardi, Lucas, "Making Racing Fun Through Player Modeling and Track
  Evolution" (2006).** This early work connects racing-track evolution to player models
  and fun, and is part of the lineage leading to personalized/experience-driven racing
  PCG. It matters historically because the track is optimized for an experience model;
  geometric validity is handled through the representation and search process rather than
  a continuous repair solve.

- **Prasetya and Maulidevi, "Search-Based Procedural Content Generation for Race Tracks
  in Video Games" (2016).** This paper is another direct race-track PCG reference from
  the search-based family. It reinforces that the natural framing in game-PCG work is to
  search over a representation with a fitness/objective, not to solve fixed-width curve
  feasibility as an explicit geometric projection.
  Link: `<https://doi.org/10.15676/ijeei.2016.8.4.6>`__

- **Alyaseri and Conner, "Comparative Analysis of Metaheuristic Algorithms for
  Procedural Race Track Generation in Games" (2024).** This paper compares genetic
  algorithms, artificial bee colony, and particle swarm optimization for a racetrack game
  level-design task. It is useful as recent evidence that race-track PCG is often framed
  as a metaheuristic content-search problem.
  Link: `<https://doi.org/10.4018/ijamc.350330>`__

- **Nascimento et al., "Procedural Generation of Isometric Racetracks Using Chain Code
  for Racing Games" (2021).** This work represents isometric racetracks using chain-code
  style directional encodings, which makes it part of the discrete path/grammar family:
  tracks are generated as symbol sequences over local movement directions and then
  interpreted as game geometry.
  Link: `<https://www.semanticscholar.org/paper/c0f0a3de8bc973ce258554153f2c4846e9765287>`__

Search over Splines and Control Points
---------------------------------------

A second family represents a road or track as a spline or control-point sequence and then
searches the parameter space. This is closer to TrackGen because the object is a
continuous centerline, but prior work usually treats validity as a constraint, fitness
term, or rejection filter.

- **Klueck, Klampfl, Wotawa, "Automatic Generation of Challenging Road Networks for ALKS
  Testing based on Bezier Curves and Search" (2021).** Roads are Bezier curves generated
  from control points. A genetic algorithm searches for roads that challenge an automated
  lane-keeping system, while a validity check rejects overlaps and too-sharp curves
  before execution. This is one of the closest continuous-geometry neighbors, but it
  avoids invalid candidates rather than projecting them into a valid constant-width road.
  Link: `<https://arxiv.org/abs/2103.01288>`__

- **Jiang et al., "Replay-Guided Adversarial Environment Design" / REPAIRED (2021).**
  This RL/UED work uses a modified OpenAI Gym CarRacing environment with closed-loop
  Bezier race tracks as procedurally generated levels. The paper's contribution is not
  the track generator itself; it is the curriculum mechanism that combines adversarial
  environment design with level replay. Track geometry is a parameterized level space
  that gets sampled and curated for training.
  Link: `<https://arxiv.org/abs/2110.02439>`__

- **Azad et al., "CLUTR: Curriculum Learning via Unsupervised Task Representation
  Learning" (2022).** CLUTR also uses CarRacing-like Bezier tracks, but learns a latent
  task manifold and samples curricula from it. This is important if TrackGen becomes an
  RL level source: CLUTR, PLR, PAIRED, and ACCEL decide which tasks to train on, while
  the underlying geometry generator remains a separate component.
  Link: `<https://arxiv.org/abs/2210.10243>`__

- **Gambi, Mueller, Fraser, "Automatically Testing Self-Driving Cars with
  Search-Based Procedural Content Generation" / ASFault (2019).** This is a core
  autonomous-driving testing reference: roads are generated procedurally and search is
  used to find scenarios that make a self-driving system fail. It is road generation for
  testing rather than racetrack design, but it belongs in the same umbrella because it
  searches a road-geometry space with validity and failure objectives.
  Link: `<https://doi.org/10.1145/3293882.3330566>`__

- **DeepJanus and Frenet-space SDC testing.** Search-based autonomous-driving test
  generation often encodes roads as curves and optimizes for frontier behaviours or
  failures. DeepJanus searches road inputs that reveal behavioural boundaries in a
  lane-keeping model; Frenet-space encodings generate smooth self-driving tests with
  higher validity rates. These systems are useful comparisons because they expose the
  same pattern: search in road-curve space plus validity filtering, not geometric repair.
  Links: `<https://arxiv.org/abs/2007.02787>`__ and
  `<https://arxiv.org/abs/2401.14682>`__

3D Gate and Waypoint Course Generators
---------------------------------------

Drone racing is the relevant 3D analogue. Most systems define a course as an ordered set
of gates or waypoints with pose, size, and sometimes obstacles. Generation means sampling
relative gate poses, selecting a benchmark track, or exposing APIs to place gate assets.
The output is a task for control/RL, not a repaired 3D tube or curve.

- **Madaan et al., "AirSim Drone Racing Lab" (2020).** AirSim Drone Racing Lab is a
  simulation framework that supports generation of drone-racing tracks in photorealistic
  environments. The course is represented by gate assets, gate poses, and direction
  vectors; the framework also supplies sensors, benchmarking APIs, and domain
  randomization. It includes useful 3D difficulty metrics such as curvature of a spline
  through gate centers. Link: `<https://arxiv.org/abs/2003.05654>`__

- **Song, Steinweg, Kaufmann, Scaramuzza, "Autonomous Drone Racing with Deep
  Reinforcement Learning" (2021).** This Scaramuzza/RPG paper is the closest 3D
  generation neighbor found. A track is defined by gates in 3D; the generator
  concatenates random gate primitives parameterized by relative position and orientation,
  then uses a curriculum that increases gate-pose diversity and track complexity. The
  method randomizes tasks for RL, but does not solve a global 3D feasibility projection
  problem over clearance, curvature, obstacle distance, or gate visibility.
  Link: `<https://arxiv.org/abs/2103.08624>`__

- **Liu et al., "Learning Generalizable Policy for Obstacle-Aware Autonomous Drone
  Racing" (2024).** This work uses a waypoint generator and obstacle manager in Isaac
  Gym. It samples relative waypoint poses and obstacles to train policies that generalize
  to unseen 3D tracks. It is highly relevant for a future 3D TrackGen interface because
  it suggests useful output fields: waypoint pose, obstacle layout, spacing, and
  difficulty. Link: `<https://arxiv.org/abs/2411.04246>`__

Boundary Case: Racing Agents on Fixed Tracks
---------------------------------------------

Some high-profile racing AI papers are tempting to include because they are about
superhuman racing, but they are not track-generation methods. They generally assume a
fixed simulator and a fixed catalogue of human-designed tracks, then learn control,
strategy, perception, or reward shaping.

- **Wurman et al., "Outracing champion Gran Turismo drivers with deep reinforcement
  learning" / GT Sophy (Nature 2022).** Sony AI and Polyphony Digital train a
  championship-level Gran Turismo agent with model-free deep RL, mixed-scenario training,
  and a reward that balances speed, tactics, and racing etiquette. The paper is useful
  as evidence that high-quality racing simulators and curricula matter, but the tracks
  are Gran Turismo circuits and training scenarios; the contribution is not procedural
  track geometry. Link: `<https://www.nature.com/articles/s41586-021-04357-7>`__

- **Fuchs et al., "Super-Human Performance in Gran Turismo Sport Using Deep
  Reinforcement Learning" (2020/2021), and later Sony AI vision-based GT agents.**
  These papers train policies for Gran Turismo Sport / Gran Turismo 7 with course
  progress rewards, privileged state or vision, and evaluation across cars and tracks.
  They are downstream consumers of tracks: the track appears as a fixed environment,
  progress coordinate, or visual context, not as a generated object.
  Links: `<https://arxiv.org/abs/2008.07971>`__,
  `<https://arxiv.org/abs/2406.12563>`__, and
  `<https://arxiv.org/abs/2504.09021>`__

Boundary Case: Racing-Line Optimization on Fixed Tracks
--------------------------------------------------------

Autonomous-racing papers often use "trajectory generation" or even "track generation" to
mean computing the best racing line around an existing circuit. This is not procedural
track geometry generation. The input track already exists as boundaries, centerline, or
a map; the output is a path inside that corridor, often with a velocity profile, that
minimizes curvature, lap time, or tracking error subject to vehicle dynamics. This
literature is still useful for TrackGen as downstream metadata: after TrackGen emits a
valid circuit, these methods could compute racing lines, speed profiles, difficulty
labels, or reference trajectories.

- **Theodosis and Gerdes, "Generating a Racing Line for an Autonomous Racecar Using
  Professional Driving Techniques" (2011).** This geometric approach decomposes turns
  into professional-driving primitives such as straights, clothoids, and constant-radius
  arcs. It is interpretable and vehicle-aware, but operates inside an existing track.
  Link: `<https://doi.org/10.1115/dscc2011-6097>`__

- **Braghin et al., "Race driver model" (2008), and Heilmeier et al., "Minimum curvature
  trajectory planning and control for an autonomous race car" (2020).** These are the
  canonical minimum-curvature / QP-style racing-line references. The path is usually
  parameterized by lateral offsets from a reference line, then optimized to reduce
  curvature subject to track and vehicle constraints. Note: the correct Braghin DOI is
  ``10.1016/j.compstruc.2007.04.028``; the ``...05.028`` DOI resolves to an unrelated
  sails paper. Links: `<https://doi.org/10.1016/j.compstruc.2007.04.028>`__ and
  `<https://doi.org/10.1080/00423114.2019.1631455>`__

- **Muehlmeier and Mueller, "Optimization of the Driving Line on a Race Track" (2002),
  and Cardamone et al., "Searching for the optimal racing line using genetic algorithms"
  (2010).** These use evolutionary search over candidate driving lines, often represented
  as splines through offsets from the track centerline. They are optimization over an
  existing corridor, not synthesis of the corridor itself.
  Links: `<https://doi.org/10.4271/2002-01-3339>`__ and
  `<https://doi.org/10.1109/itw.2010.5593330>`__

- **Christ et al., "Time-optimal trajectory planning for a race car considering variable
  tyre-road friction coefficients" (2021).** This is the more rigorous minimum-lap-time
  optimal-control family: the track is fixed, and the optimizer solves for a dynamically
  feasible path/speed profile, including friction variation.
  Link: `<https://doi.org/10.1080/00423114.2019.1704804>`__

- **Jain and Morari, "Computing the racing line using Bayesian optimization" (2020).**
  This treats racing-line search as a data-efficient black-box optimization problem over
  lateral waypoint perturbations, using a Gaussian-process model of lap time. The track
  geometry is an input. Link: `<https://doi.org/10.1109/cdc42340.2020.9304147>`__

- **Kapania and Gerdes, "Learning at the Racetrack" (2020), and Bonab and Emadi,
  "Optimization-based Path Planning for an Autonomous Vehicle in a Racing Track" (2019).**
  These represent lap-to-lap learning and receding-horizon optimization lines of work.
  They are important autonomous-racing references, but they optimize control/path
  behavior on a known track rather than generate new tracks.
  Links: `<https://doi.org/10.1109/tvt.2020.2998065>`__ and
  `<https://doi.org/10.1109/iecon.2019.8926856>`__

Open-Source Implementations and Practical Baselines
-----------------------------------------------------

Open-source implementations are weaker evidence than papers, but they matter because
they show what practitioners actually build when they need a quick track generator. The
common pattern is random control points, convex hulls, Catmull-Rom or cubic splines,
tile/segment composition, or road-block grammars. These are useful baselines to compare
against, especially because most do not include robust self-intersection or fixed-width
offset repair.

- **Gymnasium / OpenAI Gym CarRacing.** The ``CarRacing-v3`` environment generates a
  random track on reset. The implementation samples 12 radial checkpoints around a
  heavily morphed circle, steers a path toward successive checkpoints, trims a closed
  loop, rejects failures when the closure is not glued well enough, and rasterizes the
  resulting centerline into fixed-width Box2D road tiles. It is a canonical RL baseline,
  but its generation is a procedural path heuristic plus rejection, not a track-thickness
  solver. Link:
  `<https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/car_racing.py>`__

- **MetaDrive / PGDrive.** The open-source MetaDrive simulator implements the PGDrive
  road-block approach: road networks are generated compositionally from elementary road
  blocks, then used for driving and RL benchmarks. This is the strongest OSS baseline
  for scalable procedural driving scenes, but not for continuous racetrack repair.
  Links: `<https://github.com/metadriverse/metadrive>`__ and
  `<https://arxiv.org/abs/2109.12674>`__

- **AirSim Drone Racing Lab.** AirSim is open-source and Drone Racing Lab adds gate
  assets, race orchestration, APIs for gate placement, and benchmark environments. It is
  the practical 3D counterpart to 2D track generators: courses are ordered gate poses and
  simulator assets, not repaired tubes or splines.
  Link: `<https://github.com/microsoft/AirSim>`__

- **TORCS / Speed Dreams.** These open-source simulators define tracks with explicit
  segment/file formats and have a long history in racing AI and PCG. They are important
  practical references for exporting playable simulator content, but their track tooling
  is segment/editor based.
  Links: `<https://sourceforge.net/projects/torcs/>`__ and
  `<https://gitlab.com/speed-dreams/speed-dreams-code>`__

- **ChickenKorma/Track-Generator.** A small Unity implementation that generates random
  points, takes their convex hull to form a loop, inserts/displaces midpoints for more
  corners, smooths with Catmull-Rom splines, and builds a 3D track mesh and terrain. The
  README explicitly notes that only Unity scripts are included.
  Link: `<https://github.com/ChickenKorma/Track-Generator>`__

- **Drallig/ProceduralTrack.** A small Unity project combining Perlin-noise terrain with
  a procedural race track using A* pathfinding. Useful as an example of terrain/path
  construction rather than continuous-width validity.
  Link: `<https://github.com/Drallig/ProceduralTrack>`__

- **meraccos/random_car_racing_track_generator.** A compact Python generator that samples
  sorted polar control points, fits periodic cubic splines, and rasterizes a fixed-width
  track map. This is a good minimal spline baseline; it does not repair self-overlap or
  offset-band validity.
  Link: `<https://github.com/meraccos/random_car_racing_track_generator>`__

- **h3h3h0h0/Racetrackgen2.** A small C++ generator that combines noise maps,
  randomized points, path construction, and smoothing. Like the Unity examples, it is a
  practical procedural shape generator rather than a geometry-validity solver.
  Link: `<https://github.com/h3h3h0h0/Racetrackgen2>`__

Procedural Road and Street Networks
------------------------------------

Procedural road-network work is not racetrack generation, but it supplies a broader
geometric vocabulary: graphs embedded in terrain, local/global constraints, user control,
and tensor-field or L-system growth.

- **Parish and Mueller, "Procedural Modeling of Cities" (SIGGRAPH 2001).** This is the
  classic CityEngine-style L-system approach to street networks. Roads grow by rewrite
  rules under global goals and local constraints. It is graph generation, not
  constant-width circuit repair.

- **Chen et al., "Interactive Procedural Street Modeling" (SIGGRAPH 2008).** This paper
  generates street graphs from user-edited tensor fields. It is a major example of
  continuous field-guided road layout, but it optimizes urban pattern control and
  editing, not a closed race circuit with fixed-width borders.
  Link: `<https://www.peterwonka.net/Publications/pdfs/2008.SG.Chen.InteractiveProceduralStreetModeling.pdf>`__

Curve Thickness, Ropelength, and Tangent-Point/Sobolev Methods
---------------------------------------------------------------

This is the pure-geometry family most closely aligned with the "track must have
thickness" idea. A constant-width road can be viewed as a tube or band around a
centerline. Validity then depends on local curvature and non-local self-distance: the
offset band should not self-intersect.

- **Gonzalez and Maddocks, "Global Curvature, Thickness and the Ideal Shapes of Knots"
  (1999), and knot-thickness / ropelength work.** These papers formalize thickness as
  the radius of the largest embedded tube around a curve. Ropelength is length divided by
  thickness; ideal knots minimize length under tube self-avoidance. The direct fit to
  track generation is conceptual: a track centerline must support an embedded tube or
  strip of radius equal to the road half-width.
  Links: `<https://en.wikipedia.org/wiki/Knot_thickness>`__ and
  `<https://arxiv.org/abs/math/0103224>`__

- **Yu, Schumacher, Crane, "Repulsive Curves" (SIGGRAPH 2021 / arXiv 2020).** This is
  the key tangent-point / TP-Sobolev reference. The method minimizes tangent-point energy,
  which creates an infinite barrier to self-intersection by considering all point pairs
  on a curve. A Sobolev-Slobodeckij inner product preconditions gradient descent so that
  optimization progresses in a resolution-independent way. It supports constraints such
  as inextensibility and obstacle avoidance and is demonstrated on curve packing, knot
  untangling, graph embedding, non-crossing spline interpolation, and robotic path
  planning. It is a high-quality global curve-repulsion method, but it is not specialized
  to real-time batched racetrack generation.
  Link: `<https://arxiv.org/abs/2006.07859>`__

- **Henrich and Koetter, "From Generation to Gameplay: Authoring Race Tracks With
  Repulsive Curves" (2025).** This is the closest reference found to the TP-Sobolev /
  repulsive-curve branch being used directly for race tracks. The algorithm grows an
  initial closed curve inside a constrained space under self-repulsion to avoid
  self-intersections and achieve tight packing, then fits the result to splines,
  optionally introduces intersections, and builds a Unity editor / 3D model pipeline
  with crossings and bridges. It is very relevant prior art, but it uses global
  self-repulsive curve growth/authoring rather than TrackGen's local PBD/XPBD constraint
  projection over a batched fixed-width centerline.
  Links: `<https://doi.org/10.1109/TG.2025.3561107>`__ and
  `<https://www.semanticscholar.org/paper/3f14835119333af2313791ddd4e4265e661a6353>`__

- **Lagemann and von der Mosel, "Tangent-point energies and ropelength as Gamma-limit of
  discrete tangent-point energies on biarc curves" (2022).** This paper connects
  discrete tangent-point energies and ropelength, giving mathematical support for the
  idea that discrete curve energies can converge to continuous thickness-aware
  functionals. It is useful background for any TrackGen discussion involving
  discretized centerlines and TP-Sobolev alternatives.
  Link: `<https://arxiv.org/abs/2203.16383>`__

Position-Based Constraint Projection
--------------------------------------

PBD/XPBD is not a track-generation literature by itself; it is the solver family that
makes TrackGen's repair stage natural. PBD represents geometry as particles and
iteratively projects them to satisfy constraints. XPBD adds compliance so constraint
strength is less dependent on timestep and iteration count. This style is widely used in
graphics and real-time simulation because it is robust, simple, and GPU-friendly.

- **Mueller et al., "Position Based Dynamics" (2007), and Macklin et al., "XPBD:
  Position-Based Simulation of Compliant Constrained Dynamics" (2016).** These works
  provide the core projection paradigm: encode desired geometric properties as local
  constraints and iteratively move positions until constraints are satisfied.
  Links: `<https://matthias-research.github.io/pages/publications/posBasedDyn.pdf>`__ and
  `<https://matthias-research.github.io/pages/publications/XPBD.pdf>`__

- **Liu et al., "Differentiable Robotic Manipulation of Deformable Rope-like Objects
  Using Compliant Position-based Dynamics" (2022).** This is a useful rope-like-object
  analogue: XPBD constraints model stretching, bending, twisting, and compliance in a
  deformable rope. It is not about track generation, but it shows that rope/curve-like
  geometry can be handled with XPBD-style constraint systems.
  Link: `<https://arxiv.org/abs/2202.09714>`__

Positioning TrackGen
---------------------

TrackGen builds on several of these lines but combines them differently: like spline and
control-point generators, it starts from a compact continuous centerline representation;
like ropelength, knot-thickness, and TP-Sobolev / repulsive-curve work, it treats the
essential validity property as the ability to fit a non-self-intersecting thick band
around that centerline; and like PBD/XPBD graphics solvers, it enforces geometry through
iterative constraint projection on a discretized bead chain. The difference is that prior
track generators generally construct validity from prevalidated pieces, search over
parameters while rejecting invalid candidates, or sample/curate tasks for RL, whereas
TrackGen makes constant-width feasibility an explicit batched repair stage: it
arc-length resamples a generated loop, projects spacing/separation/bending constraints,
inflates the repaired centerline to inner and outer borders, and validates those borders.
Compared with TP-Sobolev or global repulsive-curve optimization, including recent
race-track authoring with repulsive curves, this is a more local, engineering-oriented
solve, but it is deterministic, GPU-batched, Warp-native, and suited to large-scale
RL/data generation where throughput and fixed tensor shapes matter. The novel
contribution therefore appears to be not PBD itself, not spline generation, and not curve
thickness as a concept, but the use of PBD/XPBD-style curve repair as the central
racetrack-generation feasibility operator for producing constant-width valid tracks at
scale.
