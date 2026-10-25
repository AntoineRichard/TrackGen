Repulsive Generator
====================

The ``repulsive`` generator grows each environment's closed centerline from a small
seed circle toward a sampled target perimeter. A tangent-point energy and a
Sobolev-preconditioned flow shape the curve inside a circular domain containing
random disc obstacles. Growth runs coarse-to-fine (``N = 64 → 128 → 256`` by default)
before the shared resample → relax → inflate tail.

A valid finished road must clear every sampled filled disc and remain inside the
outer circular wall, including its full width. The sampled force rings shape the
curve; they are not the clearance guarantee. Analytic checks guard raw updates,
and a mandatory check certifies the actual finished road band after the tail.
A disc inside the closed loop's central hole is allowed.

The construction draws on Henrich and Kötter's *Generating Race Tracks With
Repulsive Curves* and *From Generation to Gameplay: Authoring Race Tracks With
Repulsive Curves*, and Yu, Schumacher, and Crane's *Repulsive Curves*. See
:doc:`/related-work/state-of-the-art` for the references.

.. note::

   The clearance and solver repair changes Repulsive outputs, defaults and size
   behavior. Earlier figures, yield measurements and timings do not characterize
   this implementation. See :ref:`upgrading-0.3-repulsive` before reusing saved
   configurations or generated datasets.


How It Works
------------

1. **Sample the obstacles and target length.**
   The ``generator.repulsive/obstacles`` salt (:ref:`ops-rng-keying`) draws inner
   discs in angular wedges, with per-env radius and position variation. Their
   analytic centers and radii are retained alongside the sampled force rings.
   The ``generator.repulsive/growth`` salt draws ``grow_mult`` from
   ``[repulsive_grow_mult_min, repulsive_grow_mult_max]`` and sets
   ``L_final = grow_mult · 2π · r_init``. The RNG salts and draw order are unchanged
   by the repair; width reservation can change the sampled obstacle positions.

2. **Take bounded, feasible shape and growth steps.**
   Analytic endpoint derivatives of the edge-tangent energy and obstacle
   potentials are gathered without atomics. A circulant Sobolev inverse metric
   preconditions the gradient; a metric projection removes its first-order
   length change. Bounded Armijo backtracking accepts shape descent at the
   current perimeter, then a separate guarded expansion approaches the ratcheted
   target. Each accepted transition must clear the analytic discs and wall over
   its entire swept edges. An endpoint-only check would permit tunneling.
   Rejected proposals preserve the last feasible curve. Energy reductions,
   perimeter/retraction arithmetic and metric operations use double precision;
   stored curve coordinates remain ``float32``.

3. **Refine and reparameterize conservatively.**
   Resolution transitions insert vertices along existing edges, preserving the
   polygonal loop. Periodic and final same-resolution arc-length resampling are
   guarded proposals too: they must retain feasibility, and cannot silently cut
   through a disc or wall. The periodic proposal also checks its shape energy.

4. **Optionally freeze an env.**
   At ``repulsive_stall_window`` intervals in the final resolution, an env may
   freeze after reaching its actual target length with small area change, small
   movement and small projected stationarity. The stationarity measure does not
   shrink just because ``repulsive_tau`` is small. Rejected searches and
   quantized no-op proposals do not count as progress. Freeze is absorbing.
   A finite, feasible target-reaching raw curve may also finish at the iteration
   budget while its energy still admits improvement: raw validity does not
   promise optimizer convergence.

5. **Check the finished road.**
   The generator publishes its raw validity and internal barrier metadata. The
   shared tail resamples, relaxes and inflates the curve; ``validity.default``
   then checks the actual band against those barriers as well as its ordinary
   road-validity rules. The planner requires this final check and refuses
   geometry-changing stages after it. No barrier metadata is added to the public
   Track result.


Early exit and capture
----------------------

``repulsive`` is a Stage generator with ``capture = CaptureClass.BOUNDED``
(``registry.describe("generator", "repulsive").capture``). Its
``ctx.bounded_loop`` has a fixed host schedule computed from the static parameters
(:ref:`cuda-graph-bounded`):

.. math::

   n_{\mathrm{ratchet}} = \left\lceil
       \frac{\log(\mathtt{repulsive\_grow\_mult\_max})}
            {\log(1 + \mathtt{repulsive\_ratchet\_rate})}
       \right\rceil,
   \qquad
   n_{\mathrm{iters}} = \lceil 1.6\,n_{\mathrm{ratchet}}\rceil
       + \mathtt{repulsive\_settle\_iters}.

Refinement, periodic resampling and stall-check cadences depend on this schedule,
not device readback. After the final resolution begins, each stall check writes
``active = any env not frozen`` on device. Frozen envs retain their geometry for
all remaining iterations, so the following lowerings produce the same bytes on
a fixed device:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Call
     - Growth loop
   * - Eager on ``cpu``
     - Polls at window boundaries and stops when all envs are frozen.
   * - Eager on ``cuda``
     - Runs the full bound without a host sync; frozen envs skip geometry work.
   * - The generator's own graph, a ``wp.ScopedCapture`` or a capture registered
       with ``registered(conditional=True)``, with supported CUDA conditional nodes
     - Skips remaining windows on device when every env is frozen.
   * - A capture without conditional nodes, including a plain ``torch.cuda.graph``
       or a recording inside ``track_gen.exec.unconditional()``
     - Records and runs the full bound; frozen envs skip geometry work.

A conditional-node graph cannot be used as a CUDA child graph, and a side stream
that joined a Warp capture must rejoin before a conditional window is recorded.
Use ``track_gen.exec.unconditional()`` when an unrolled graph is required; see
:doc:`/how-it-works/cuda-graph`. Early exit is optional and does not determine
whether a completed road is valid.


Cost
----

Repulsive runs an iterative dense-pair optimizer with bounded proposal searches.
Its work depends on the resolution schedule and iteration budget, and conditional
early exit depends on the whole batch freezing. Measure generation cost for your
configuration and regeneration cadence. The earlier timing table is not a
measurement of the repaired solver.

Construction runs an eager warm-up pass. On CUDA the first ``generate()`` also
records the generator's graph before replay; recording into a caller's graph has
its own one-time cost.


Math
----

**Edge-tangent energy.** For ordered disjoint edges ``I = (a, b)`` and
``J = (c, d)``, let ``T_I = (b - a) / |I|``. The endpoint trapezoidal quadrature is

.. math::

   E_{\mathrm{TP}} = \sum_{I,J\,\mathrm{disjoint}}
       \frac{|I|\,|J|}{4}
       \sum_{u\in\{a,b\}}\sum_{v\in\{c,d\}}
       \frac{|(v-u)\times T_I|^{\alpha}}
            {(\lVert v-u\rVert^2 + \varepsilon^2)^{\beta/2}}.

Edges sharing an endpoint are excluded, including across the loop closure.
Derivatives include both edge lengths and the owner edge's tangent, rather than
an averaged vertex tangent. The supported exponents satisfy
``alpha > 1`` and ``alpha + 2 <= beta < 2 * alpha + 1``.

**Obstacle potential.** The sampled force-ring potential uses inverse power
``p = beta - alpha`` and curve-length quadrature weights:

.. math::

   E_{\mathrm{rings}} = \sum_i w_i\sum_m
       m_w(\lVert x_i-o_m\rVert^2+\varepsilon^2)^{-p/2}.

Analytic log-barrier potentials for vertex and segment clearances steer the curve
before the hard constraints bind. These potentials stay active throughout growth
and settling. Feasibility is enforced independently of their energy values.

**Sobolev metric and length constraint.** The circulant inverse metric has
nonzero Fourier response ``1 / (lambda_k**s + eps_reg)``, with
``s = (beta - 1) / alpha`` and the ring-Laplacian eigenvalues normalized by their
first nonzero mode. Its translation mode is zero. Metric rows and accumulation
use double precision; a requested resolution/order whose metric cannot retain a
positive accurate spectrum is rejected at construction. The projected shape step
retracts to the current perimeter during each backtracking trial. The separate
length ratchet does not guarantee that an infeasible target can be reached.


Parameters
----------

These ``TrackGenConfig`` fields are also the fields of ``RepulsiveParams``. Set
them flat or through ``generator_params`` (:ref:`builtin-generator-params`).
Generator-specific rules are checked when constructing a Repulsive generator.
Core fields such as ``scale``, ``num_points`` and ``half_width`` remain shared
pipeline parameters.

``repulsive_grow_mult_min`` / ``repulsive_grow_mult_max`` (default 2.0 / 3.0)
   Bounds of the target-perimeter multiplier draw. The minimum must be at least
   1; the maximum must be greater than 1 and at least the minimum. Larger targets
   can be infeasible under permanent full-width clearance.

``repulsive_domain_frac`` (default 0.35)
   Positive domain-radius fraction: ``r_dom = repulsive_domain_frac · scale_ref
   · scale``, where ``scale_ref = 4.9029``.

``repulsive_domain_init_ratio`` (default 4.0)
   Ratio ``r_dom / r_init``; must exceed 1.

``repulsive_obstacle_count_min`` / ``repulsive_obstacle_count_max`` (default 8 / 12)
   Inclusive bounds of the inner-disc count draw; the minimum is at least 1 and
   the maximum is at least the minimum. A disc whose placement band is empty is
   inactive, while the RNG draw sequence is preserved.

``repulsive_obstacle_radius_min_frac`` / ``repulsive_obstacle_radius_max_frac``
   (default 0.02 / 0.045)
   Disc-radius fractions of ``r_dom``; positive minimum, maximum at least minimum.

``repulsive_ratchet_rate`` (default 0.012)
   Positive per-iteration target-length growth factor. Geometry advances only
   through accepted feasible expansions.

``repulsive_alpha`` / ``repulsive_beta`` (default 3.0 / 6.0)
   Energy exponents: ``alpha > 1`` and
   ``alpha + 2 <= beta < 2 * alpha + 1``.

``repulsive_tau`` (default 0.4)
   Positive flow-step parameter. Backtracking and a trust cap bound the actual
   displacement; reducing tau does not establish stationarity.

``repulsive_stages`` (default ``(64, 128, 256)``)
   Non-empty, strictly increasing positive multiples of 4. The last entry must
   equal ``num_points``.

``repulsive_settle_iters`` (default 40)
   Additional iteration budget; at least 1.

``repulsive_resample_every`` (default 25)
   Periodic arc-length proposal interval; at least 1.

``repulsive_stall_window`` (default 16)
   Interval between optional per-env freeze checks; at least 1.

``repulsive_stall_area_tol`` (default 0.05)
   Relative area-change tolerance, also used for normalized movement and
   projected stationarity in the freeze check. Area change alone cannot freeze
   an env.

``size`` (default 1.0)
   Dynamic per-env size in ``[0.1, 10]``; also affects the unscaled width
   reservation (`Per-env size`_).

``repulsive_w_len`` and ``repulsive_deactivate_obstacles`` are removed. They are
not accepted by configs, Params or saved generator parameter mappings.

.. _repulsive-dynamic:

Per-env size
------------

``size`` is Repulsive's one dynamic ``float32`` field. Draw it with
``presets.track(generator="repulsive", randomize={"generator.size": (lo, hi)})``,
or update ``gen.params["generator"].set("size", ...)`` or write it in your own
graph (:ref:`tuning-per-env`). Other Repulsive knobs are static.

The final raw centerline, analytic discs and wall are multiplied by ``size[e]``;
road width and spacing stay absolute. When constant road width is known, the
planner supplies a width hint and growth reserves ``half_width / size[e]`` in
unscaled coordinates. This affects obstacle placement, barrier forces and feasible
motion. Changing size can therefore change the raw layout and curve shape, not
just multiply a fixed curve's XY coordinates. The actual finished band remains
authoritative, including with variable width or a custom tail.

Size can also exceed the resampled point capacity. A target-perimeter capacity
bound is ``size_hi * repulsive_grow_mult_max * 2π * r_init / spacing + 1``;
round up and choose ``n_max`` for the largest size (:ref:`tuning-size`). A
construction value or randomizer range that exceeds the planner's capacity bound
emits a ``UserWarning``. Dynamic writes are not checked by that warning; truncated
results fail validity.


What Makes It Distinct
-----------------------

Repulsive constructs its raw curve through iterative constrained growth rather
than a single sampling pass. Each env uses its own seed, target and feasibility
checks; another env freezing does not change its curve. For fixed parameters,
seeds, device and Warp version, the deterministic gathers and fixed reduction
order make the result reproducible. Cross-device byte equality is not promised.

Repulsive's repair changes its seeded outputs. The other generators' outputs and
RNG streams are unchanged by this repair (:ref:`upgrading-0.3-repulsive`).


Fallback and Validity
----------------------

There is no generator-local retry or fallback. Raw ``gen.valid`` requires finite,
feasible geometry that reaches the sampled target within the numerical length
tolerance. A raw curve may satisfy this at the iteration budget without being
frozen or stationary. A short or infeasible raw curve fails the generator check.

Final ``Track.valid`` additionally requires the ordinary road checks and full-band
clearance after resampling, relaxation and inflation. Each four-vertex road cell
is checked through its convex hull. This certifies the occupied ruled strip,
including folded cells, but can conservatively reject a disc in an unused part of
a concave cell's hull. It does not fill the loop's global central hole. Circular
wall containment of all band vertices contains each entire cell because the wall
bounds a convex domain.

Repulsive programs add ``repulsive_disc`` (bit 8, mask ``1 << 8``) and
``repulsive_wall`` (bit 9, mask ``1 << 9``) to the result's ``fail_bits`` mapping.
These bits are absent from other generators' programs. Inspect ``valid`` and
``fail_mask`` before using a result. Explicit aggressive tuning can fail after the
shared tail even if raw growth succeeded; no configuration or seed range has a
universal success guarantee. Invalid results obey the existing ``on_invalid``
policy: ``"poison"`` by default, or explicit ``"keep"`` for diagnosis
(:ref:`upgrading-0.3-poison`).
