# Track Generation State of the Art

This note focuses only on track and road-like geometry generation. It excludes most
downstream autonomous-racing papers unless their course representation or generator is
directly relevant. The organizing question is: how have prior systems created valid
race tracks, road centerlines, or 3D gate courses, and where do curve-thickness,
tangent-point/Sobolev methods, and PBD/XPBD-style constraint projection fit?

## Constructive Segment, Tile, and Road-Block Generators

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
  Link: https://en.wikipedia.org/wiki/Speed_Dreams

- **Barthet et al., "Closing the Affective Loop via Experience-Driven Reinforcement
  Learning Designers" (2024).** This is a recent racetrack PCG paper for the Solid Rally
  game. Tracks are strings of track-component IDs placed on a 2D tile grid. A genetic
  algorithm or RL designer searches for layouts whose predicted arousal trace matches a
  target. Feasibility is handled through grid collisions and shortest-path closure with
  Dijkstra when needed. It is a strong example of modern racing PCG, but the geometry is
  discrete and affect-driven. Link: https://arxiv.org/abs/2408.06346

- **Li et al., "Improving the Generalization of End-to-End Driving through Procedural
  Generation" / PGDrive (2020).** PGDrive generates driving scenes by sampling and
  connecting elementary road blocks, then adds traffic and simulation state. Its purpose
  is RL generalization rather than racetrack geometry, but it is a key example of road
  generation by construction. Link: https://arxiv.org/abs/2012.13681

- **Ikram, Muktadir, Whitehead, "Procedural Generation of Complex Roundabouts for
  Autonomous Vehicle Testing" (2023).** This work procedurally constructs roundabouts
  and connecting roads for HD-map / OpenDRIVE-style autonomous-vehicle testing. It uses
  geometric construction around a maximal circle, connects incident roads, and injects
  irregularity. The core idea is again constructive validity: generate a road primitive
  from parameterized components rather than relax an arbitrary centerline. Link:
  https://arxiv.org/abs/2303.17900

## Search over Splines and Control Points

A second family represents a road or track as a spline or control-point sequence and then
searches the parameter space. This is closer to TrackGen because the object is a
continuous centerline, but prior work usually treats validity as a constraint, fitness
term, or rejection filter.

- **Klueck, Klampfl, Wotawa, "Automatic Generation of Challenging Road Networks for ALKS
  Testing based on Bezier Curves and Search" (2021).** Roads are Bezier curves generated
  from control points. A genetic algorithm searches for roads that challenge an automated
  lane-keeping system, while a validity check rejects overlaps and too-sharp curves
  before execution. This is one of the closest continuous-geometry neighbors, but it
  avoids invalid candidates rather than projecting them into a valid constant-width
  road. Link: https://arxiv.org/abs/2103.01288

- **Jiang et al., "Replay-Guided Adversarial Environment Design" / REPAIRED (2021).**
  This RL/UED work uses a modified OpenAI Gym CarRacing environment with closed-loop
  Bezier race tracks as procedurally generated levels. The paper's contribution is not
  the track generator itself; it is the curriculum mechanism that combines adversarial
  environment design with level replay. Track geometry is a parameterized level space
  that gets sampled and curated for training. Link: https://arxiv.org/abs/2110.02439

- **Azad et al., "CLUTR: Curriculum Learning via Unsupervised Task Representation
  Learning" (2022).** CLUTR also uses CarRacing-like Bezier tracks, but learns a latent
  task manifold and samples curricula from it. This is important if TrackGen becomes an
  RL level source: CLUTR, PLR, PAIRED, and ACCEL decide which tasks to train on, while
  the underlying geometry generator remains a separate component. Link:
  https://arxiv.org/abs/2210.10243

- **DeepJanus and Frenet-space SDC testing.** Search-based autonomous-driving test
  generation often encodes roads as curves and optimizes for frontier behaviours or
  failures. DeepJanus searches road inputs that reveal behavioural boundaries in a
  lane-keeping model; Frenet-space encodings generate smooth self-driving tests with
  higher validity rates. These systems are useful comparisons because they expose the
  same pattern: search in road-curve space plus validity filtering, not geometric repair.
  Links: https://arxiv.org/abs/2007.02787 and https://arxiv.org/abs/2401.14682

## 3D Gate and Waypoint Course Generators

Drone racing is the relevant 3D analogue. Most systems define a course as an ordered set
of gates or waypoints with pose, size, and sometimes obstacles. Generation means sampling
relative gate poses, selecting a benchmark track, or exposing APIs to place gate assets.
The output is a task for control/RL, not a repaired 3D tube or curve.

- **Madaan et al., "AirSim Drone Racing Lab" (2020).** AirSim Drone Racing Lab is a
  simulation framework that supports generation of drone-racing tracks in photorealistic
  environments. The course is represented by gate assets, gate poses, and direction
  vectors; the framework also supplies sensors, benchmarking APIs, and domain
  randomization. It includes useful 3D difficulty metrics such as curvature of a spline
  through gate centers. Link: https://arxiv.org/abs/2003.05654

- **Song, Steinweg, Kaufmann, Scaramuzza, "Autonomous Drone Racing with Deep
  Reinforcement Learning" (2021).** This Scaramuzza/RPG paper is the closest 3D
  generation neighbor found. A track is defined by gates in 3D; the generator
  concatenates random gate primitives parameterized by relative position and orientation,
  then uses a curriculum that increases gate-pose diversity and track complexity. The
  method randomizes tasks for RL, but does not solve a global 3D feasibility projection
  problem over clearance, curvature, obstacle distance, or gate visibility. Link:
  https://arxiv.org/abs/2103.08624

- **Liu et al., "Learning Generalizable Policy for Obstacle-Aware Autonomous Drone
  Racing" (2024).** This work uses a waypoint generator and obstacle manager in Isaac
  Gym. It samples relative waypoint poses and obstacles to train policies that generalize
  to unseen 3D tracks. It is highly relevant for a future 3D TrackGen interface because
  it suggests useful output fields: waypoint pose, obstacle layout, spacing, and
  difficulty. Link: https://arxiv.org/abs/2411.04246

## Procedural Road and Street Networks

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
  editing, not a closed race circuit with fixed-width borders. Link:
  https://www.peterwonka.net/Publications/pdfs/2008.SG.Chen.InteractiveProceduralStreetModeling.pdf

## Curve Thickness, Ropelength, and Tangent-Point/Sobolev Methods

This is the pure-geometry family most closely aligned with the "track must have
thickness" idea. A constant-width road can be viewed as a tube or band around a
centerline. Validity then depends on local curvature and non-local self-distance: the
offset band should not self-intersect.

- **Gonzalez and Maddocks, "Global Curvature, Thickness and the Ideal Shapes of Knots"
  (1999), and knot-thickness / ropelength work.** These papers formalize thickness as
  the radius of the largest embedded tube around a curve. Ropelength is length divided by
  thickness; ideal knots minimize length under tube self-avoidance. The direct fit to
  track generation is conceptual: a track centerline must support an embedded tube or
  strip of radius equal to the road half-width. Links:
  https://en.wikipedia.org/wiki/Knot_thickness and https://arxiv.org/abs/math/0103224

- **Yu, Schumacher, Crane, "Repulsive Curves" (SIGGRAPH 2021 / arXiv 2020).** This is
  the key tangent-point / TP-Sobolev reference. The method minimizes tangent-point energy,
  which creates an infinite barrier to self-intersection by considering all point pairs
  on a curve. A Sobolev-Slobodeckij inner product preconditions gradient descent so that
  optimization progresses in a resolution-independent way. It supports constraints such
  as inextensibility and obstacle avoidance and is demonstrated on curve packing, knot
  untangling, graph embedding, non-crossing spline interpolation, and robotic path
  planning. It is a high-quality global curve-repulsion method, but it is not specialized
  to real-time batched racetrack generation. Link: https://arxiv.org/abs/2006.07859

- **Lagemann and von der Mosel, "Tangent-point energies and ropelength as Gamma-limit of
  discrete tangent-point energies on biarc curves" (2022).** This paper connects
  discrete tangent-point energies and ropelength, giving mathematical support for the
  idea that discrete curve energies can converge to continuous thickness-aware
  functionals. It is useful background for any TrackGen discussion involving
  discretized centerlines and TP-Sobolev alternatives. Link:
  https://arxiv.org/abs/2203.16383

## Position-Based Constraint Projection

PBD/XPBD is not a track-generation literature by itself; it is the solver family that
makes TrackGen's repair stage natural. PBD represents geometry as particles and
iteratively projects them to satisfy constraints. XPBD adds compliance so constraint
strength is less dependent on timestep and iteration count. This style is widely used in
graphics and real-time simulation because it is robust, simple, and GPU-friendly.

- **Mueller et al., "Position Based Dynamics" (2007), and Macklin et al., "XPBD:
  Position-Based Simulation of Compliant Constrained Dynamics" (2016).** These works
  provide the core projection paradigm: encode desired geometric properties as local
  constraints and iteratively move positions until constraints are satisfied. Links:
  https://matthias-research.github.io/pages/publications/posBasedDyn.pdf and
  https://matthias-research.github.io/pages/publications/XPBD.pdf

- **Liu et al., "Differentiable Robotic Manipulation of Deformable Rope-like Objects
  Using Compliant Position-based Dynamics" (2022).** This is a useful rope-like-object
  analogue: XPBD constraints model stretching, bending, twisting, and compliance in a
  deformable rope. It is not about track generation, but it shows that rope/curve-like
  geometry can be handled with XPBD-style constraint systems. Link:
  https://arxiv.org/abs/2202.09714

## Positioning TrackGen

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
Compared with TP-Sobolev or global repulsive-curve optimization, this is a more local,
engineering-oriented solve, but it is deterministic, GPU-batched, Warp-native, and suited
to large-scale RL/data generation where throughput and fixed tensor shapes matter. The
novel contribution therefore appears to be not PBD itself, not spline generation, and not
curve thickness as a concept, but the use of PBD/XPBD-style curve repair as the central
racetrack-generation feasibility operator for producing constant-width valid tracks at
scale.
