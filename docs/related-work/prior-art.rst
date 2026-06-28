Racetrack Generation Prior Art
==============================

This note maps prior approaches to racetrack and road-like geometry generation, then
positions TrackGen's current pipeline:

``seeded corners -> sort -> Bezier centerline -> gates -> arc-length resample -> XPBD relax/resolve -> constant-width inflate -> validity``.

The short conclusion: I found many uses of procedural construction, tile grammars,
splines/control points, rejection/validity checks, evolutionary search, reinforcement
learning curricula, drone-racing gate randomization, road-network synthesis, and curve
self-repulsion. The RL literature is closer than it first looked: several UED papers use
Bezier closed-loop car-racing tracks as procedurally generated "levels." Drone racing is
also relevant for the later 3D version: courses are usually represented as ordered 3D
gates/waypoints plus obstacles, then solved by planning, control, RL, or domain
randomization. Autonomous-racing labs and platforms such as MIT/Daniela Rus,
F1TENTH/RoboRacer, TUM Autonomous Motorsport, ETH/MPCC, AutoRally, and CMU DRIVE
mostly assume track maps already exist and focus on racing lines, planning, control,
perception, learning, and safety. Still, I did not find a racetrack generator that uses a
PBD/XPBD bead-chain resolve as the central feasibility step for turning an arbitrary
simple centerline into a constant-width valid road. The closest neighbors are general
PBD/XPBD physics, repulsive-curve/ropelength mathematics, RL/UED track curricula,
drone-racing waypoint/gate generators, and autonomous-driving road generators that
generate, curate, or reject tasks rather than geometrically repair them.


What Prior Work Usually Does
----------------------------

1. Constructive segment/tile generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These build a track from discrete pieces: straights, turns, loops, bridges, ramps,
roundabout components, or road blocks. They typically keep validity simple because
the pieces have known connection rules.

- **Barthet, Branco, Gallotta, Khalifa, Yannakakis, "Closing the Affective Loop via
  Experience-Driven Reinforcement Learning Designers" (2024).**

  - Link: https://arxiv.org/abs/2408.06346
  - Domain: Solid Rally racing game.
  - Representation: string of track component IDs on a 2D tile grid.
  - Generation methods: genetic algorithm and Go-Explore-style RL designer.
  - Feasibility: grid collision checks; Dijkstra connects start/end to make a playable
    circuit if needed.
  - Objective: match target arousal traces for player/annotator clusters.
  - Relevance to TrackGen: strong example of modern racetrack PCG, but the geometry is
    discrete/tile-based and feasibility is penalty/rejection plus shortest-path closure,
    not continuous curve relaxation.

- **Li, Peng, Zhang, Liu, Zhou, "Improving the Generalization of End-to-End Driving
  through Procedural Generation" / PGDrive (2020).**

  - Link: https://arxiv.org/abs/2012.13681
  - Representation: configurable road blocks connected into diverse road networks.
  - Objective: improve driving-policy generalization across procedurally generated
    scenarios.
  - Relevance: similar target domain if TrackGen is used for RL, but PGDrive is a
    driving-environment generator, not a closed-loop race-track curve-thickness resolver.

- **Ikram, Muktadir, Whitehead, "Procedural Generation of Complex Roundabouts for
  Autonomous Vehicle Testing" (2023).**

  - Link: https://arxiv.org/abs/2303.17900
  - Representation: parameterized roundabout/incident-road components exported to
    OpenDRIVE.
  - Geometry method: fit a maximal circle, segment the circular road, connect incident
    roads, add noise/irregularity, handle turbo roundabouts.
  - Relevance: careful geometry construction for a specific road primitive; validity is
    handled by construction and local overlap avoidance.

- **TORCS / Speed Dreams track tooling and "Interactive Track Generator for TORCS and
  Speed Dreams".**

  - Link: https://en.wikipedia.org/wiki/Speed_Dreams
  - Representation: segment list of straights and left/right turns; main track, sides,
    borders, barriers.
  - The Speed Dreams page describes an evolutionary/genetic-programming track generator
    from Politecnico di Milano producing TORCS/Speed Dreams track outlines.
  - Relevance: historically close to racing, but segment-oriented and interactive /
    evolutionary; not a continuous curve-thickness relaxation approach.


2. Search over spline/control-point road geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These generate a parametric road by choosing control points, then optimize or search for
cases that are challenging, valid, diverse, or personalized.

- **Klueck, Klampfl, Wotawa, "Automatic Generation of Challenging Road Networks for
  ALKS Testing based on Bezier Curves and Search" (2021).**

  - Link: https://arxiv.org/abs/2103.01288
  - Representation: Bezier curve defined by control points.
  - Generation method: genetic algorithm searches control-point arrangements.
  - Fitness: force an automated lane-keeping system to cross the lane center or leave
    the road.
  - Validity: pre-execution validity check rejects overlap and too-sharp curves.
  - Relevance: closest to TrackGen's continuous-curve side, but the invalid geometry is
    rejected or avoided during search, not resolved by physically inspired projection.

- **Togelius, De Nardi, Lucas, "Towards Automatic Personalised Content Creation for
  Racing Games" (IEEE CIG 2007).**

  - Link via 2024 paper bibliography: https://ar5iv.org/html/2408.06346v1
  - Representation/objective: racetracks generated according to a player/fun model
    inspired by Koster's theory of fun.
  - Method family: early search-based / experience-driven PCG for racing games.
  - Relevance: important historical racetrack PCG reference; the contribution is
    personalization/fitness modeling, not geometric feasibility repair.


3. Experience-driven and search-based PCG frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are not racetrack-specific algorithms, but they explain how much of the track
literature thinks about generation: genotype, mutation/crossover/search, simulation-based
fitness, player model, novelty/diversity, and expressive range.

- **Togelius, Yannakakis, Stanley, Browne, "Search-Based Procedural Content Generation:
  A Taxonomy and Survey" (2011).**

  - Link via bibliography: https://ar5iv.org/html/2408.06346v1
  - Relevance: frames track generation as search in content space with a fitness
    function.

- **Yannakakis and Togelius, "Experience-Driven Procedural Content Generation" (2011).**

  - Link via bibliography: https://ar5iv.org/html/2408.06346v1
  - Relevance: player-experience model becomes the objective; used later in racing
    affect/arousal work.

- **Hendrikx, Meijer, van der Velden, Iosup, "Procedural Content Generation for Games:
  A Survey" (2013).**

  - Link via bibliography: https://ar5iv.org/html/2408.06346v1
  - Relevance: broad PCG taxonomy; useful context, not a direct geometric solution.


4. RL, UED, and procedural racing levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These papers are important because they use procedural race tracks as training tasks or
curricula. The generator is usually not the contribution; the research question is how to
sample, curate, mutate, replay, or adversarially select environments so an RL agent
generalizes.

- **Jiang et al., "Replay-Guided Adversarial Environment Design" / REPAIRED (NeurIPS
  2021).**

  - Link: https://arxiv.org/abs/2110.02439
  - Domain: modified OpenAI Gym CarRacing.
  - Representation: closed-loop tracks generated as Bezier curves with up to 12 control
    points under curvature constraints.
  - Objective: combine UED with Prioritized Level Replay so the adversary produces
    challenging but learnable levels.
  - Evaluation: zero-shot transfer to 20 human-designed F1 tracks that are out of
    distribution for the generator.
  - Relevance to TrackGen: this is one of the closest RL references. It still treats
    track geometry as a parameterized level distribution; it does not repair
    constant-width feasibility with a PBD/XPBD geometric resolve.

- **Dennis et al., "Emergent Complexity and Zero-shot Transfer via Unsupervised
  Environment Design" / PAIRED (2020).**

  - Link: https://arxiv.org/abs/2012.02096
  - Method: an adversary designs environments by maximizing regret between protagonist
    and antagonist agents.
  - Relevance: conceptual ancestor of REPAIRED. Useful for the "environment design"
    framing, but not a racetrack geometry method.

- **Jiang et al., "Prioritized Level Replay" (2020).**

  - Link: https://arxiv.org/abs/2010.03934
  - Method: replay levels that have high learning potential, instead of uniformly
    sampling procedural environments.
  - Relevance: a strong baseline if TrackGen is later used as an RL level source; it
    decides which tracks to revisit, not how to make track geometry valid.

- **Parker-Holder et al., "Evolving Curricula with Regret-Based Environment Design" /
  ACCEL (2022).**

  - Link: https://arxiv.org/abs/2203.01302
  - Method: mutates previously seen environments and uses regret to select useful
    curriculum items.
  - Relevance: closest to "search in level space" among the UED papers, but the edits are
    environment-level mutations rather than continuous curve-thickness projection.

- **Azad et al., "CLUTR: Curriculum Learning via Unsupervised Task Representation
  Learning" (2022).**

  - Link: https://arxiv.org/abs/2210.10243
  - Domain: includes CarRacing with Bezier tracks and the same F1-style transfer
    benchmark family.
  - Method: learns a latent task manifold with a recurrent VAE and samples curricula in
    latent space.
  - Relevance: very useful for the RL positioning: CLUTR changes how tasks are sampled,
    not the underlying geometry generator.

- **Cobbe et al., "Leveraging Procedural Generation to Benchmark Reinforcement
  Learning" / Procgen (2019).**

  - Link: https://arxiv.org/abs/1912.01588
  - Method: benchmark suite built around procedurally generated levels to measure RL
    generalization.
  - Relevance: general RL context for why a fast, deterministic, batched generator like
    TrackGen matters.


5. Drone racing and 3D gate-course generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drone racing is the best preview of the 3D version of TrackGen. The dominant
representation is an ordered sequence of gates or waypoints in 3D, often with gate
orientation, gate size, obstacles, and a reward based on progress through gate centers.
The literature focuses on perception, time-optimal trajectory generation, RL control,
sim-to-real transfer, and domain randomization. I found gate/waypoint randomizers, but
not a PBD/XPBD geometric feasibility resolver over a 3D curve or gate chain.

- **Madaan et al., "AirSim Drone Racing Lab" (2020).**

  - Link: https://arxiv.org/abs/2003.05654
  - Representation: racing tracks as gate poses and direction vectors in photorealistic
    simulation environments.
  - System support: gate assets, APIs for spawning/destructing gates, object pose/scale
    changes, multiple sensors, and domain randomization.
  - Geometry metrics: fits a 3D third-order spline through gate centers and measures
    curvature per unit length as a track-complexity signal.
  - Relevance to TrackGen: very useful for 3D output format and difficulty metrics, but
    the simulator supplies courses; it does not relax a proposed 3D course into a valid
    tube/gate layout.

- **Kaufmann, Loquercio, Ranftl, Dosovitskiy, Koltun, Scaramuzza, "Deep Drone Racing:
  Learning Agile Flight in Dynamic Environments" (CoRL 2018).**

  - Link: https://arxiv.org/abs/1806.08548
  - Representation: tracks with static or moving gates; the controller predicts a local
    waypoint and desired speed from images, then a planner generates short minimum-jerk
    trajectory segments.
  - Training data: global trajectories through gates are computed at training time; gate
    positions are varied to improve robustness.
  - Relevance: good Scaramuzza/RPG reference for the waypoint abstraction and moving-gate
    robustness; not a course generator in the TrackGen sense.

- **Loquercio, Kaufmann, Ranftl, Dosovitskiy, Koltun, Scaramuzza, "Deep Drone Racing:
  From Simulation to Reality with Domain Randomization" (2019).**

  - Link: https://arxiv.org/abs/1905.09727
  - Method: trains a perception/planning stack in simulation with domain randomization
    and deploys zero-shot on a physical quadrotor.
  - Relevance: strong sim-to-real reference if TrackGen is used to produce large
    training distributions. The randomization is visual/gate/layout robustness, not a
    geometry solve.

- **Foehn et al., "AlphaPilot: Autonomous Drone Racing" (2020/2021).**

  - Link: https://arxiv.org/abs/2005.12813
  - Representation: race course as a sequence of known gates with uncertain detected
    poses.
  - Method: learned gate detection, nonlinear filtering, global gate map, and near
    time-optimal trajectory planning.
  - Relevance: important Scaramuzza/RPG competition system. Course geometry is an input
    map to perception/planning, not the generated object.

- **Song, Steinweg, Kaufmann, Scaramuzza, "Autonomous Drone Racing with Deep
  Reinforcement Learning" (IROS 2021).**

  - Link: https://arxiv.org/abs/2103.08624
  - Representation: a race track is completely defined by gates in 3D; progress reward is
    computed by projecting position onto straight segments between adjacent gate centers.
  - Random generator: concatenates random gate primitives parameterized by relative
    position and orientation; adjusts pose ranges and gate count to change complexity.
  - Curriculum: starts near straight-line tracks and increases relative-pose diversity as
    crash rate improves.
  - Relevance: probably the closest Scaramuzza/RPG prior for procedural 3D racing tasks.
    It randomizes gate chains for RL, but does not project/repair a 3D track to satisfy
    clearance, curvature, visibility, or obstacle constraints.

- **Kaufmann et al., "Champion-level Drone Racing using Deep Reinforcement Learning"
  (Nature 2023).**

  - Link: https://www.nature.com/articles/s41586-023-06419-4
  - System: Swift, an onboard vision/RL system that beat human champions on a fixed
    physical track.
  - Representation: a seven-gate 3D circuit; reward includes progress toward the next
    gate and a perception objective to keep gates visible.
  - Relevance: strongest "autonomous racing at human/champion level" citation, but the
    track is a fixed human-designed benchmark.

- **Song, Romero, Mueller, Koltun, Scaramuzza, "Reaching the Limit in Autonomous Racing:
  Optimal Control versus Reinforcement Learning" (Science Robotics 2023).**

  - Link: https://arxiv.org/abs/2310.10943
  - Method: compares RL and optimal control on drone-racing gate courses; RL directly
    optimizes a gate-progress objective with domain randomization.
  - Relevance: key Scaramuzza/RPG analysis of RL versus trajectory optimization, but
    again assumes gate-course tasks rather than generating or repairing geometry.

- **Krinner, Romero, Bauersfeld, Zeilinger, Carron, Scaramuzza, "MPCC++: Model
  Predictive Contouring Control for Time-Optimal Flight with Safety Constraints" (RSS
  2024).**

  - Link: https://arxiv.org/abs/2403.17551
  - Method: model predictive contouring control for drone racing with safety constraints,
    learned residual dynamics, and controller tuning.
  - Relevance: valuable for 3D TrackGen because it makes gate clearance and track-safety
    constraints explicit. It is still a controller over a given gate course, not a
    generator/repair stage.

- **Liu et al., "Learning Generalizable Policy for Obstacle-Aware Autonomous Drone
  Racing" (2024).**

  - Link: https://arxiv.org/abs/2411.04246
  - Representation: waypoint generator plus obstacle manager, with relative waypoint
    poses and randomized obstacles in Isaac Gym.
  - Relevance: directly useful for 3D TrackGen. It suggests outputting waypoints/gates,
    obstacles, and difficulty ranges, but the generator is randomization for policy
    generalization, not a constraint-projection geometry repair stage.

For a 3D TrackGen extension, these papers suggest representing tasks as ordered
gates/waypoints with pose, pass-through rectangle/tube, direction constraints, obstacle
SDFs, and difficulty metrics such as curvature per length, gate visibility, waypoint
spacing, heading/pitch/roll changes, obstacle clearance, and required acceleration. The
unique opening would be a 3D XPBD/PBD resolve over a space curve or gate chain that
satisfies curvature, clearance, spacing, slope/bank, obstacle, and visibility constraints
before emitting trainable tasks.


6. Autonomous-racing labs, platforms, and downstream consumers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mistake to avoid in related work is treating these as track-generation papers. They
are better framed as downstream consumers and benchmark ecosystems: fixed or benchmark
maps, racing lines, lane boundaries, track widths, progress coordinates, opponents,
friction limits, and safety constraints.

- **MIT CSAIL / Daniela Rus, MIT RACECAR, VISTA, and racing world models.**

  - Links: https://arxiv.org/abs/2102.09812, https://arxiv.org/abs/2103.04909,
    https://arxiv.org/abs/2111.12083
  - Examples: Deep Latent Competition learns multi-agent racing policies in a latent
    world model; Latent Imagination studies F1TENTH-style racing from
    LiDAR observations; VISTA/VISTA 2.0 is a data-driven simulator for autonomous driving.
  - Relevance to TrackGen: strong MIT/Daniela Rus-adjacent racing and simulation context,
    but the track geometry is a map, benchmark, or data-derived scene rather than a
    generated-and-repaired continuous road band.

- **Evans et al., "Unifying F1TENTH Autonomous Racing: Survey, Methods and
  Benchmarks" (2024), and RoboRacer survey work.**

  - Links: https://arxiv.org/abs/2402.18558, https://arxiv.org/abs/2506.15899
  - Scope: F1TENTH/RoboRacer benchmark ecosystems for perception, planning, control,
    learning, simulation, and sim-to-real transfer.
  - Track representation: maps, centerlines, track boundaries/widths, LiDAR observations,
    progress, and standard benchmark circuits.
  - Relevance to TrackGen: this is probably the most important small-scale ground-robot
    racing ecosystem. It needs valid maps and benchmarks; it does not provide a PBD/XPBD
    track generator.

- **Liniger, Domahidi, Morari, "Optimization-Based Autonomous Racing of 1:43 Scale RC
  Cars" (2015/2017).**

  - Link: https://arxiv.org/abs/1711.07300
  - Method: model predictive contouring control for autonomous racing, maximizing
    progress while satisfying track and opponent constraints.
  - Relevance: foundational ETH/MPCC-style racing-control reference. It defines useful
    downstream quantities, especially progress, contouring/lag errors, and track-boundary
    constraints, but assumes the track is already given.

- **TUM Autonomous Motorsport / Indy Autonomous Challenge work.**

  - Links: https://arxiv.org/abs/2205.15979, https://arxiv.org/abs/2202.03807
  - Scope: autonomous full-scale racecar stack design, localization, perception,
    planning, control, and system architecture for the Indy Autonomous Challenge.
  - Relevance: strong racing lab/platform reference. Track maps, corridors, and racelines
    are inputs to the stack; the work is not procedural geometry generation.

- **Goldfain et al., "AutoRally: An Open Platform for Aggressive Autonomous Driving"
  (2018).**

  - Link: https://arxiv.org/abs/1806.00678
  - Platform: 1:5-scale autonomous racing/research vehicle for aggressive off-road
    driving, learning, and model-predictive control.
  - Relevance: another mature racing testbed. It is useful for downstream dynamics and
    dataset expectations, but not track synthesis.

- **CMU Driverless Intelligent Vehicles Lab / DRIVE Lab.**

  - Link: https://www.ri.cmu.edu/robotics-groups/driverless-intelligent-vehicles-lab/
  - Research scope: behaviors, planning, control, and perception for autonomous vehicles,
    with emphases on adversarial multi-agent systems, safe control under uncertainty,
    high-performance driving, and adaptation to dynamic environments.
  - Racing context: participates in the Indy Autonomous Challenge through AI Racing Tech.
  - Relevance to TrackGen: useful downstream consumer perspective, especially for
    racing-line, friction, opponent, and safety constraints. I did not find DRIVE work
    where the primary contribution is procedural track geometry generation.

- **Xue, Zhu, Dolan, Borrelli, "Learning Model Predictive Control with Error Dynamics
  Regression for Autonomous Racing" (2023).**

  - Link: https://arxiv.org/abs/2309.10716
  - Method: LMPC for autonomous racing that learns local error dynamics and explores
    safely toward the handling limit, including full-scale Indy Autonomous Challenge
    experiments.
  - Relevance: assumes tracks/courses exist; the contribution is vehicle dynamics
    learning and control robustness.

- **Kalaria, Lin, Dolan, "Towards Safety Assured End-to-End Vision-Based Control for
  Autonomous Racing" (2023).**

  - Link: https://arxiv.org/abs/2303.02267
  - Method: imitation-learned end-to-end racing controller guarded by a control barrier
    function to stay within lane boundaries.
  - Track representation: optimal racing line from center coordinates and widths; safety
    functions use lane-boundary constraints.
  - Relevance: gives strong terminology for downstream constraints: lane boundaries,
    racing line, progress, and safety sets. It is not a generator.

- **Kalaria, Lin, Dolan, "Adaptive Planning and Control with Time-Varying Tire Models
  for Autonomous Racing Using Extreme Learning Machine" (2023).**

  - Link: https://arxiv.org/abs/2303.08235
  - Method: online tire-model adaptation and planning/control for racing at friction
    limits.
  - Relevance: useful for difficulty labels or downstream dynamics-aware costs, but not
    track synthesis.

- **AL-Sunni, Almubarak, Horng, Dolan, "LLA-MPC: Fast Adaptive Control for Autonomous
  Racing" (2025).**

  - Link: https://arxiv.org/abs/2505.19512
  - Method: learning-free adaptive MPC with a model bank for rapidly changing
    tire-surface interactions.
  - Track representation: minimum-curvature racing line over a sequence of track points
    and widths; velocity profiles adapt to estimated friction.
  - Relevance: useful if TrackGen emits not just geometry but also per-track/per-segment
    friction, width, curvature, and expected speed metadata.

- **Betz et al., "Autonomous Vehicles on the Edge: A Survey on Autonomous Vehicle
  Racing" (2022).**

  - Link: https://arxiv.org/abs/2202.07008
  - Scope: survey of autonomous racecars covering perception, planning, control,
    end-to-end learning, and racing platforms.
  - Relevance: a good broad citation for autonomous racing as a downstream field. The
    survey is not about procedural generation, which helps keep TrackGen's contribution
    separated from vehicle-control work.


7. Procedural road/city network generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These solve "roads" rather than "racetracks", usually as graphs embedded in terrain or
urban constraints.

- **Parish and Mueller, "Procedural Modeling of Cities" (SIGGRAPH 2001).**

  - Link via CityEngine publications: https://en.wikipedia.org/wiki/CityEngine
  - Representation: L-system growth of street networks.
  - Relevance: foundational procedural street-network work; graph growth rather than
    closed-loop continuous track repair.

- **Chen, Esch, Wonka, Mueller, Zhang, "Interactive Procedural Street Modeling"
  (SIGGRAPH 2008).**

  - Link: https://www.peterwonka.net/Publications/pdfs/2008.SG.Chen.InteractiveProceduralStreetModeling.pdf
  - Representation: street graph generated from user-designed tensor fields.
  - Relevance: major example of continuous field-guided road graph generation; it
    prioritizes interactive control and urban patterns, not constant-width circuit
    validity.

- **Road/HD-map generators for AV testing.**

  - Examples cited by the roundabout paper include PGDrive, ASFault, HDMapGen,
    OSM/GIS extraction pipelines, OpenDRIVE tooling, and GAN/data-driven road synthesis.
  - Relevance: they create varied driving environments and often include validity checks,
    but the common move is construction/rejection, not PBD-style shape projection.


8. Curve self-avoidance, thickness, and repulsive geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the nearest mathematical family to TrackGen's "make this curve feasible for a
road band" idea.

- **Gonzalez and Maddocks, "Global Curvature, Thickness and the Ideal Shapes of Knots"
  (PNAS 1999).**

  - Link: https://en.wikipedia.org/wiki/Knot_thickness
  - Core idea: curve thickness can be characterized through local curvature radius and
    global self-distance; a thick tube around a curve should not self-intersect.
  - Relevance: directly supports TrackGen's use of thickness as the unified validity
    concept for constant-width inflation.

- **Yu, Schumacher, Crane, "Repulsive Curves" (SIGGRAPH 2021 / arXiv 2020).**

  - Link: https://arxiv.org/abs/2006.07859
  - Method: tangent-point energy with Sobolev-Slobodeckij preconditioning for curve
    self-repulsion and non-crossing design.
  - Relevance: closest high-quality continuous alternative to XPBD. It is a global energy
    optimization method, useful for curve design, packing, knot untangling, spline
    interpolation, and paths. It is not specifically a racetrack generator and does not
    use the simple local PBD/XPBD projection recipe TrackGen uses as its default.


9. PBD / XPBD and constraint projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the solver family TrackGen borrows from.

- **Mueller et al., "Position Based Dynamics" (2007).**

  - Link via PBD survey/reference discussion: https://arxiv.org/abs/1802.02673
  - Core idea: iteratively project particle positions to satisfy constraints; popular in
    real-time physics because it is robust and simple.

- **Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
  (2016).**

  - Link via PBD survey/reference discussion: https://arxiv.org/abs/1802.02673
  - Core idea: extend PBD to better handle compliance and iteration/time-step
    dependence.

- **Weiss et al., "Position-Based Multi-Agent Dynamics for Real-Time Crowd Simulation"
  (2017/2018).**

  - Link: https://arxiv.org/abs/1802.02673
  - Method: PBD constraints for multi-agent collision avoidance.
  - Relevance: shows PBD escaping cloth/soft-body simulation into 2D spatial constraint
    systems and real-time games. Still not racetrack generation.


Where TrackGen Lands
--------------------

TrackGen is not just another random/spline generator. The distinctive pipeline is:

#. Generate simple closed centerline candidates in batch.
#. Arc-length resample them.
#. Treat the centerline as a bead chain.
#. Project/relax the chain with separation, spacing, and bending constraints until a
   constant-width road band can fit.
#. Inflate by a fixed half-width.
#. Validate the actual road borders.
#. Run the full process GPU-batched in Warp, with static loops and CUDA graph capture.

The important conceptual difference is that prior generators usually avoid invalid
geometry by construction, reject it, score it poorly, or sample a different task.
TrackGen takes a simple generated curve and repairs the geometric feasibility of the
constant-width road itself.


What Seems Unique
-----------------

Strong uniqueness claim:

- **Using PBD/XPBD-style bead-chain constraint projection as the main racetrack
  feasibility resolver** appears new in the searched literature.
- The constraints are not merely "smooth the path"; they target the physical/geometric
  property needed by a road: enough local radius and non-local clearance for a
  constant-width offset.
- The output is a constant-width inner/outer/center track with a real border
  self-intersection validity check, not a variable-width rescue or a rejected candidate.

Medium-strength uniqueness claim:

- **GPU-batched, deterministic, replayable track generation for RL** is also unusual.
  PGDrive, UED CarRacing, drone racing simulators, and AV-testing generators produce
  procedural driving/flying scenes, but TrackGen's pure-Warp, count-aware,
  CUDA-graph-capturable pipeline is a different systems contribution.
- **TrackGen is complementary to UED/PLR/CLUTR and drone-racing RL.** Those methods
  decide which levels to train on or how to adapt policies/controllers; TrackGen
  provides a higher-quality continuous geometry generator/repair stage that could feed
  such curricula.
- **TrackGen is upstream of autonomous-racing stacks.** MIT/Rus, F1TENTH/RoboRacer,
  TUM/IAC, ETH/MPCC, AutoRally, and CMU DRIVE mostly consume maps, centerlines, widths,
  boundaries, progress coordinates, racelines, and friction/safety metadata. TrackGen can
  be positioned as a way to produce those inputs at scale.

Caveat:

- PBD/XPBD itself is not new.
- Curve thickness, ropelength, tangent-point energy, and repulsive curves are not new.
- Gate/waypoint randomization for drone racing is not new.
- What looks new is the combination: closed-loop racetrack generation + constant-width
  validity as curve thickness + XPBD/PBD projection resolve + GPU-batched RL-scale
  execution.


Suggested Positioning Sentence
-------------------------------

"Unlike prior racetrack PCG systems that construct, search, curate, or reject candidate
layouts, TrackGen treats a generated centerline as a constrained bead chain and uses a
PBD/XPBD-style geometric resolve to reshape it until a constant-width road band is valid;
this turns validity from a post-hoc filter into a batched, deterministic repair stage."


Suggested Related-Work Buckets
-------------------------------

For a paper/report, I would structure related work as:

#. **Racing-game PCG and experience-driven track generation**:
   Togelius et al. 2007; Barthet et al. 2024; broader EDPCG/SBPCG surveys.
#. **RL environment design and procedural racing curricula**:
   PAIRED/UED, PLR, REPAIRED, ACCEL, CLUTR, Procgen.
#. **Drone racing and 3D gate courses**:
   AirSim Drone Racing Lab, Scaramuzza/RPG drone racing papers, AlphaPilot, Swift,
   optimal-control-versus-RL drone racing, random waypoint/obstacle generators.
#. **Autonomous-racing labs, platforms, and downstream consumers**:
   MIT/Daniela Rus/RACECAR/VISTA, F1TENTH/RoboRacer, TUM Autonomous Motorsport/IAC,
   ETH/MPCC, AutoRally, CMU DRIVE Lab, IAC/AI Racing Tech, LMPC/adaptive MPC/CBF
   safety, autonomous-racing surveys.
#. **Procedural roads and driving-simulation environments**:
   PGDrive, ASFault/AV testing, Bezier road search, roundabout/OpenDRIVE generators,
   street-network methods.
#. **Continuous curve validity and self-avoidance**:
   curve thickness/ropelength, tangent-point/repulsive curves.
#. **Position-based constraint projection**:
   PBD, XPBD, PBD for crowds/collision avoidance.
#. **This work**:
   XPBD track resolve for constant-width, GPU-batched, graph-capturable RL track
   generation.
