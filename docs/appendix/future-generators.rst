Future and Experimental Generators
===================================

This page lists candidate methods for the first stage of TrackGen: the stage that
produces an initial closed centerline before constant-spacing resampling and XPBD
relaxation. It is partly historical: ``hull``, ``polar``, ``checkpoint``, and ``voronoi``
have since shipped as standard runtime generators, while several later sections remain
investigation notes. The canonical runtime contract is documented in
:doc:`/contributing/writing-a-generator`.

The goal is not to replace relaxation. The goal is to feed XPBD better initial curves:
more diverse, more controllable, less degenerate, and easier to repair into
constant-width valid tracks.


Contract for Any Centerline Generator
---------------------------------------

The generator should be judged by the interface it gives to the downstream pipeline:

- **Input:** per-environment seed and static config.
- **Output:** a closed 2D centerline or a small set of control points that can be
  deterministically expanded into a closed centerline.
- **Shape:** fixed upper bounds for CUDA graph capture: fixed max control count, fixed
  dense sample count, bounded loops, no host-side retry loop based on generated data.
- **Validity target:** simple closed loop is preferred, but not necessarily
  constant-width valid. XPBD owns thickness repair.
- **Diversity target:** support length, turn density, hairpin count, straight length,
  compactness, symmetry/asymmetry, and "racing character" as tunable axes.
- **Performance target:** batch-friendly Warp kernels, one env per row, no dynamic
  allocation, no per-env Python branching.

The evaluation should be post-relaxation, not only pre-relaxation. A generator that
looks rough but relaxes into high-quality tracks is more useful than a prettier generator
that produces rare unrecoverable cases.


Metrics for Comparing Candidates
----------------------------------

For each method, run a fixed seed suite and log:

- **Pre-relax simplicity:** centerline self-intersection rate before fallback.
- **Post-relax yield:** final ``Track.valid`` rate after XPBD and inflation.
- **Fallback rate:** how often generation must route to polygon fallback or another
  rescue path.
- **Relaxation burden:** XPBD displacement, iterations-to-valid if measured, min
  thickness before/after, curvature-radius deficit before relaxation.
- **Diversity:** distributions of length, area, compactness, curvature, straight-section
  length, turn-angle histogram, and self-approach distances.
- **Control:** how predictably config knobs affect difficulty and style.
- **Racing-line feasibility:** cheap post-relax proxy for whether the circuit supports a
  meaningful racing line, such as minimum-curvature cost, peak curvature, straight-to-turn
  structure, and a simple friction-circle velocity profile.
- **Systems fit:** Warp implementation complexity, static-shape fit, CUDA graph
  compatibility, memory footprint.


Recommended Investigation Order
---------------------------------

1. Strengthen the Current Corner-Sort-Bezier Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Idea.** Keep the current pipeline:
``corner_sample -> count -> prune-then-ccw-sort -> closed Bezier -> de-cross fallback``,
but treat it as a family rather than one generator. Sample style parameters per env:
corner count, radial spread, grid/noise strength, Bezier handle length, handle clamp, and
possibly an anisotropic scale/rotation.

**Why investigate.** This is the lowest-risk path. It already fits Warp, fixed shapes,
CUDA graph capture, and XPBD. Per-env style randomization may produce much richer tracks
without adding a new generator.

**Risks.** It stays star-shaped around a centroid, so it may underproduce long
straights, nested turns, chicanes, and strongly non-convex layouts. Too much handle
freedom increases fallback rate.

**Experiment.** Add a style sampler outside the core geometry first, or emulate it by
sweeping config batches. Report valid yield, fallback rate, and diversity metrics across
``min/max_num_points``, ``rad``, ``handle_clamp_frac``, radial jitter, and aspect ratio.

**Priority:** immediate baseline and likely first implementation.


2. Convex Hull plus Midpoint Displacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented: see the :doc:`Hull Generator deep dive </generators/hull>`.

**Idea.** Sample random points, compute or approximate their convex hull, insert
midpoints along hull edges, displace those midpoints inward/outward, then smooth with
Catmull-Rom or Bezier segments. This is the common OSS pattern used in small Unity
generators such as ``ChickenKorma/Track-Generator``.

**Why investigate.** It is simple, controllable, and gives a natural route to straights
plus corners. The hull makes the base loop simple, and midpoint displacement adds racing
shape without immediately creating a tangled ordering problem.

**Risks.** Exact convex hull is awkward in fixed-shape Warp. A cheap version can sort
points by angle, which collapses back toward the current method. Pure hull loops can be
too convex unless displacement is strong.

**Warp fit.** Good if we avoid a general hull algorithm: use angle-sorted random points
as a hull-like base, then insert one or two deterministic midpoint layers. Full dynamic
hull is less attractive.

**Experiment.** Implement a torch/oracle prototype with ``P`` base points and one midpoint
subdivision layer. Compare against current generator on diversity and fallback rate. If
promising, port the deterministic midpoint expansion to Warp.

**Priority:** high. Good cheap competitor to the current method.


3. Periodic Polar Spline / Radial Function Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented: see the :doc:`Polar Generator deep dive </generators/polar>`.

**Idea.** Represent the loop in polar form around a center:
``r(theta) = base + random low-frequency signal``, then sample sorted angles and fit a
periodic cubic spline or directly evaluate a Fourier-like radial function.

**Why investigate.** This produces smooth simple loops by construction if ``r(theta)``
stays positive and angular sampling is monotone. It is close to the unsupported Fourier
generator but can be made Warp-native and simpler.

**Risks.** It is inherently star-shaped. It may be too smooth and "blob-like" unless
augmented with straightening/chicane controls. It can hide local tight curvature in
high-frequency radial modes.

**Warp fit.** Strong. Fixed number of modes or control radii, static evaluation loop,
deterministic from seed. No sorting if angles are fixed.

**Experiment.** Prototype fixed-angle radial control points with periodic interpolation.
Try low-frequency Fourier, random radial knots, and optional angular jitter. Measure
curvature-radius deficit and post-XPBD displacement.

**Priority:** high. It should give a smooth lower-burden generator and a useful contrast
to corner polygons.


4. Curvature-Profile / Clothoid-Arc Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Idea.** Sample a smooth periodic curvature profile ``kappa(s)`` and integrate it into a
closed centerline: ``theta'(s) = kappa(s)``, ``x'(s) = cos(theta)``,
``y'(s) = sin(theta)``. The profile can be represented by fixed basis functions,
fixed-length piecewise-constant curvature arcs, or a small vocabulary of straights,
clothoid-in ramps, constant-radius apex segments, and clothoid-out ramps.

**Why investigate.** This is the most useful lesson from the racing-line literature for
the centerline-generation stage. Theodosis/Gerdes-style clothoid-plus-arc corner models and minimum-curvature
racing-line optimizers are solving a different problem, but they point at a strong track
representation: curvature is the natural variable for controlling radius, sweepers,
hairpins, chicanes, and straight exits. Used before relaxation, this generator could
produce initial centerlines with lower curvature noise and more deliberate racing
character than point sorting.

**Risks.** Closure is the hard part. The integrated curve must satisfy net heading and
net displacement closure, and a naive solve can become iterative or CPU-heavy. The method
may also be too smooth unless the sampled curvature profile includes asymmetric sectors,
short straights, and paired left-right features.

**Warp fit.** Medium-high if closure is handled with fixed basis coefficients or a
bounded projection. Fixed samples, fixed basis loops, and static integration are
CUDA-graph friendly. A host-side nonlinear closure solve is not.

**Experiment.** Start with a CPU prototype: sample a low-frequency ``kappa(s)`` profile or
piecewise clothoid/arc sequence, project heading closure, integrate, then apply an affine
or low-dimensional displacement-closure correction. Compare against polar splines on
post-XPBD yield, curvature-radius deficit, straight-section distribution, and the cheap
racing-line feasibility score.

**Priority:** high. This should join the first-pass shortlist because it is a direct way
to translate racing-line prior art into a pre-relaxation centerline generator.


5. Checkpoint-Steering Generator Like Gym CarRacing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented: see the :doc:`Checkpoint Generator deep dive </generators/checkpoint>`.

**Idea.** Sample radial checkpoints around a rough circle, then integrate a heading that
steers toward the next checkpoint under a bounded turn rate. Gymnasium CarRacing uses this
kind of procedural path heuristic.

**Why investigate.** It can create natural flowing tracks with long arcs and a stronger
sense of vehicle path than point sorting. Turn-rate limits directly control curvature
before XPBD.

**Risks.** Loop trimming and failure detection are data-dependent in the classic
implementation. We need a bounded, branchless version for CUDA graphs. It may produce
rare poor closures or require a fallback path.

**Warp fit.** Medium. A fixed-step integrator is easy; robust closure selection without
dynamic search is harder. Could always integrate for ``K`` steps and resample the final
cycle implied by phase.

**Experiment.** Implement a CPU prototype that integrates a fixed number of steps around
monotone checkpoint phase. Measure how often closure is acceptable without retries.
Investigate a deterministic closure repair before considering Warp.

**Priority:** medium-high. Promising for track feel, but more systems risk.


6. Segment Grammar / Road-Block Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Idea.** Build tracks from a finite vocabulary: straight, constant-radius turn,
hairpin, chicane, S-bend, kink, sweeper. Each segment has length, angle, and curvature
parameters. Connect segments sequentially and close the loop with a final connector or a
global normalization step.

**Why investigate.** This is how TORCS/Speed Dreams-style tools and PGDrive-like systems
think about roads. It gives direct control over racing vocabulary: long straight,
technical sector, hairpin frequency, chicane count.

**Risks.** Closure is the hard part. Naively concatenating segments rarely closes. A
closure solve may become iterative, host-side, or fragile. Track shapes can become
formulaic.

**Warp fit.** Medium if the grammar emits a fixed number of segments and uses a bounded
closure recipe. Low if it needs search/retry.

**Experiment.** Start offline in Python: sample ``S`` segments, then solve a simple affine
closure over segment lengths/turns. Feed to XPBD and check whether the relaxation hides
closure artifacts.

**Priority:** medium. Valuable if we want explicit racing-style knobs.


7. Chain-Code / Direction-Sequence Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Idea.** Generate a closed path as a sequence of discrete directions or turn commands,
similar to chain-code isometric racetrack generation. Interpret the sequence as a
polyline, simplify/smooth, then pass to XPBD.

**Why investigate.** It is a compact fixed-length representation and maps well to
evolutionary or RL-based level design. It can create isometric/arcade-like tracks and
explicitly encode turn patterns.

**Risks.** Grid artifacts. Closure and self-intersection handling are nontrivial. It may
produce too many orthogonal or diagonal corners unless smoothing is strong.

**Warp fit.** Good for generation, medium for cleanup. Direction sequences are static
arrays; self-crossing rescue still needed.

**Experiment.** Generate balanced turn sequences with net displacement near zero, then
smooth with Catmull-Rom/Bezier and XPBD. Compare against segment grammar for style
control.

**Priority:** medium. Useful for discrete style and future curriculum methods.


8. Graph-Cycle Extraction from Random Geometric Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented: see the :doc:`Voronoi Generator deep dive </generators/voronoi>`.

**Idea.** Sample random points, build a local graph such as Delaunay, k-nearest-neighbor,
or grid-neighbor graph, then extract a simple cycle. Smooth the cycle and relax it.

**Why investigate.** Graph cycles can produce non-star-shaped loops, long straights,
bypasses, and more varied topology while still being simple before smoothing.

**Risks.** Delaunay and graph search are not a natural fit for fixed-shape Warp kernels.
Cycle extraction is algorithmic and branch-heavy. Exact Voronoi ridge walking remains
better as an offline diagnostic unless it is reduced to bounded primitives.

**Warp fit.** Medium after simplification. The production ``generator="voronoi"`` uses a
fixed site field, angular anchor targets, nearest-unused-site snapping, smoothing, and
polygon fallback. That keeps the useful cell-count/layout controls while avoiding dynamic
Delaunay/Voronoi construction.

**Experiment.** The remaining research question is whether exact face/ring extraction is
worth distilling into another bounded runtime primitive.

**Priority:** implemented, with exact Voronoi traversal deferred.


9. Repulsive-Curve Growth as Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Idea.** Grow a closed curve in a bounded box under self-repulsion, as in recent
race-track authoring with repulsive curves. The generator itself avoids self-intersection
and packs the curve into space before spline fitting.

**Why investigate.** It is the closest literature neighbor to our geometry view. It may
produce very good pre-relaxation centerlines with low self-crossing and high packing
quality.

**Risks.** It overlaps conceptually with relaxation and may be too expensive for the
runtime first stage. Global pairwise repulsion is ``O(N^2)`` and iterative. It may blur
the paper's distinction between generation and XPBD repair unless framed carefully.

**Warp fit.** Medium for a bounded fixed-iteration version, but likely not as cheap as
current generation. Could be an offline generator or optional high-quality mode.

**Experiment.** Implement a small torch prototype: initialize a noisy circle, run a few
self-repulsive growth steps with length/box constraints, fit spline, then XPBD. Compare
XPBD displacement and diversity against the current method.

**Priority:** medium. Important scientifically, but not the cheapest production path.


10. Scalar-Field Contour Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Idea.** Generate a random scalar field from low-frequency noise or Gaussian bumps, then
extract a closed contour as the centerline. Smooth and relax the contour.

**Why investigate.** Contours naturally form closed loops and can produce organic shapes.
They are useful if terrain or elevation becomes part of the generator.

**Risks.** Marching squares / contour tracing is dynamic and awkward on GPU. Extracted
contours can fragment, self-touch, or produce tiny loops. Contour selection is a search
problem.

**Warp fit.** Low for runtime. Better as an offline source of shape priors or templates.

**Experiment.** Offline only. Generate contours, filter by length/area, relax, and see
whether the style is worth distilling.

**Priority:** low-medium. Useful for future terrain-aware work.


11. Data-Driven Track Prior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Idea.** Build a small corpus of real and game tracks, normalize them, resample to
control points, and sample from a PCA/VAE/diffusion-like latent prior. The first stage
emits control points, then XPBD repairs exact width validity.

**Why investigate.** If the target is "tracks that feel real," a learned prior may be
better than hand-tuned random geometry. It can also provide templates for curriculum
difficulty.

**Risks.** Dataset curation cost. Harder to claim procedural novelty. Learned outputs may
need clipping, rejection, or projection. Runtime sampling may not be Warp-native unless
the learned model is very small or precomputed.

**Warp fit.** Low for neural runtime, medium for a precomputed codebook/template sampler.

**Experiment.** Start with a codebook: store normalized control-point templates and
sample affine/style perturbations. Avoid neural models until template sampling proves
useful.

**Priority:** low for current TrackGen, higher if visual realism becomes important.


Cross-Cutting Variants Worth Testing
--------------------------------------

These can apply to several generators:

- **Per-env style randomization:** sample generator knobs per track instead of one config
  per batch.
- **Template plus noise:** start from a known class such as oval, figure-eight-like but
  non-crossing, road-course, kart track, or test-track, then perturb.
- **Anisotropic scaling:** stretch/compress before XPBD to create long straights and
  fast circuits.
- **Straight-section preservation:** mark some control spans as low-curvature targets so
  XPBD does not round every sector equally.
- **Racing vocabulary templates:** reuse a small set of track-building motifs from
  racing-line work, such as straight, clothoid-in, constant-radius apex, clothoid-out,
  sweeper, chicane, and hairpin. These can be segment templates or priors over a
  curvature-profile generator.
- **Difficulty labels:** emit approximate pre-relax metrics such as expected curvature,
  length, turn count, and compactness for curriculum use.
- **Downstream racing-line proxy:** after relaxation, run a cheap minimum-curvature or
  friction-circle speed-profile approximation to rank whether the generated track is
  merely valid or actually produces a meaningful racing problem.
- **Fallback families:** if Bezier smoothing self-crosses, choose between polygon
  fallback, reduced-handle fallback, or local handle clamp instead of one global fallback.


Proposed Shortlist
-------------------

The first implementation pass should investigate five methods:

#. **Current generator with per-env style sampling.** Minimal code risk; likely immediate
   diversity gain.
#. **Convex hull plus midpoint displacement.** Strong practical baseline from OSS
   implementations; easy to compare.
#. **Periodic polar spline / radial function.** Smooth-by-construction baseline and a
   Warp-friendly replacement candidate for the unsupported Fourier path.
#. **Curvature-profile / clothoid-arc generator.** Racing-line-inspired representation
   that directly controls radius, straights, sweepers, and corner vocabulary before XPBD.
#. **Checkpoint-steering generator.** Potentially better racing feel; worth prototyping
   before committing to a Warp port.

The second pass should investigate:

6. **Segment grammar** for explicit racing-style control.
7. **Repulsive-curve growth** as the closest literature method and high-quality optional
   generator.
8. **Chain-code direction sequences** if we want an evolutionary/RL-friendly discrete
   representation.

Graph-cycle extraction, scalar-field contours, and data-driven priors should stay
offline until they show a clear quality advantage.


Implementation Notes
---------------------

Keep the first production versions simple:

- Prototype in torch/Python first with the same output contract as
  ``generate_centerline_warp``.
- Reuse the same downstream ``resample_constant_spacing -> xpbd -> inflate`` benchmark
  for every candidate.
- Do not optimize for pre-relax visual smoothness alone. Optimize for post-relax valid
  yield, diversity, and low XPBD displacement.
- Do not add unbounded retry loops to the Warp path. If a method needs search, make it
  an offline generator or give it a fixed-budget fallback.
- Keep one common comparison script so each generator can be ranked by the same metrics.


Suggested Experiment Table
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - Method
     - Prototype first?
     - Warp candidate?
     - Main metric to watch
   * - Current + style sampling
     - no, already exists
     - yes
     - diversity vs fallback rate
   * - Convex hull + midpoint displacement
     - yes
     - yes, simplified
     - post-XPBD yield and style diversity
   * - Periodic polar spline
     - yes
     - yes
     - smoothness vs blob-like shapes
   * - Curvature-profile / clothoid-arc
     - yes
     - yes, if closure is bounded
     - closure quality and racing-line feasibility
   * - Checkpoint steering
     - yes
     - maybe
     - closure failures and racing feel
   * - Segment grammar
     - yes
     - maybe
     - closure artifacts and style control
   * - Chain-code directions
     - yes
     - maybe
     - self-cross rate and grid artifacts
   * - Graph-cycle / Voronoi site snapping
     - yes, done
     - yes, implemented
     - site-count/layout diversity
   * - Repulsive-curve growth
     - yes
     - optional
     - quality vs generation cost
   * - Scalar-field contours
     - offline
     - unlikely
     - contour selection failures
   * - Data-driven templates
     - offline
     - maybe as codebook
     - realism and controllability
