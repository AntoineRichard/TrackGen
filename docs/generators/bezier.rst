Bezier Generator
================

The ``bezier`` generator is the default and most configurable of the centerline generators.
It samples a set of random grid-jittered corner anchors, sorts them by angle around
their centroid to form a simple star-shaped polygon, and assembles a closed cubic
Bézier spline through those corners.  Handle lengths are proportional to
``rad * chord`` and adaptively clamped so that no handle overshoots a nearby corner.
Its two handle knobs, ``rad`` and ``handle_clamp_frac``, are per-env (dynamic)
parameters: ``randomize=`` draws them and a curriculum can write them from its own graph
(`Per-env (dynamic) parameters`_). The older per-environment style draw
(``style_sampling``) is still supported, and remains the way to vary ``scale`` per env.
Any smooth curve that still self-crosses after clamping is rescued by its provably
simple corner polygon, which the downstream XPBD solver re-rounds.

.. figure:: ../assets/generator-bezier.png
   :alt: Sample tracks produced by the Bezier generator
   :align: center

   Sample tracks produced by the Bezier generator at the default configuration.

How It Works
------------

The generator runs a fixed sequence of Warp kernels in a single pass — no
host-side retry loop, no per-environment Python branching:

1. **Optional style draw.**  When ``style_sampling=True``, a dedicated kernel
   (``_style_sample_k``) draws per-environment ``rad``, ``scale``, and
   ``handle_clamp_frac`` values from their respective ``*_range`` fields using RNG salt
   ``2741``.  A ranged ``rad`` / ``handle_clamp_frac`` is drawn straight into that
   knob's per-env parameter array, which then reports the draw
   (``gen.params["generator"]["rad"]``); a knob whose range is ``None`` is not drawn and
   reads its own per-env parameter array, like a generation without style sampling. A
   ``scale_range`` of ``None`` collapses to the scalar ``scale``.  This Python branch is
   resolved at CUDA-graph capture time; the per-environment values are device arrays
   and the captured graph remains static.

2. **Corner count.**  ``ops.sample_counts`` (salt ``6151``, offset ``0``) draws
   ``count[e]`` uniformly from ``[min_num_points, max_num_points]`` for each environment.

3. **Corner positions.**  ``ops.sample_grid_points`` (salt ``9781``, offset ``0``)
   samples ``max_num_points`` grid cells from the ``ops.grid_cells(min_point_distance)``
   grid of ``num_cells × num_cells`` cells, where
   ``num_cells = int(1 / (min_point_distance * 2))``.  Each cell selection uses
   bounded duplicate rejection (up to eight retries per corner) to spread corners
   across distinct cells, then adds per-axis jitter in ``[-0.5, 0.5)``.  Coordinates are
   multiplied by ``scale`` (or by the per-environment ``scale[e]`` array in style mode).

4. **Prune-then-sort.**  The ``ops.angle_sort`` kernel (with the legacy count floor of 1:
   a count below 1 sorts the first corner) sorts only the first ``count[e]`` corners by
   ``atan2`` angle around the centroid of those same kept corners (not all
   ``max_num_points`` slots), writing NaN past the live count.  Sorting *after*
   pruning matters: sorting all ``max_num_points`` points and keeping a prefix would
   compute the centroid over unpruned slots, producing angular wedges around the wrong
   centroid that can close as figure-eight loops.

5. **Vertex tangents.**  ``_vertex_tangents_k`` computes each corner's tangent as a
   normalized blend of the outgoing unit edge direction ``u_out`` and the incoming unit
   edge direction ``u_in``, weighted by ``p = atan(edgy) / π + 0.5``.  At the default
   ``edgy=0``, ``p=0.5`` gives a symmetric bisector tangent.  It also records each
   corner's shorter incident edge length for the adaptive clamp (step 6).

6. **Cubic Bézier assembly.**  ``_assemble_dyn_k`` emits ``num_points_per_segment`` dense
   samples for each closed edge (corner ``i`` to corner ``(i+1) % count[e]``),
   including the wrap-around closing edge.  For each segment the handle lengths are
   computed as:

   .. math::

      h_0 = \min\!\bigl(\texttt{rad} \cdot \text{chord},\;
                         \texttt{handle\_clamp\_frac} \cdot \min_{\text{adj}}(e_0)\bigr)

   .. math::

      h_1 = \min\!\bigl(\texttt{rad} \cdot \text{chord},\;
                         \texttt{handle\_clamp\_frac} \cdot \min_{\text{adj}}(e_1)\bigr)

   where :math:`\min_{\text{adj}}(e)` is the length of the shorter incident edge at
   that corner.  The cubic control points are then:

   .. math::

      P_1 = c_0 + \hat{t}_0 \cdot h_0, \qquad
      P_2 = c_1 - \hat{t}_1 \cdot h_1

   and the segment is evaluated via the standard Bernstein basis (see Math section).
   ``rad`` and ``handle_clamp_frac`` are env ``e``'s values, read from the per-env
   arrays (step 1's draws for a ranged knob in style mode) and clamped to the declared
   bounds (`Per-env (dynamic) parameters`_; a style draw is used as drawn).  The frozen
   legacy entry point ``generate_centerline_warp`` keeps the scalar kernels
   ``_assemble_k`` and, in style mode, ``_assemble_style_k``; the arithmetic is the
   same, so both give the same bytes.

7. **Dense-to-N arc-resample.**  ``ops.arc_resample`` converts the dense
   ``max_num_points × num_points_per_segment`` loop to exactly ``num_points``
   arc-uniformly spaced samples, building cumulative arc length in ``float64`` and
   interpolating with a searchsorted lookup.

8. **Self-intersection check and fallback.**  ``ops.self_intersections`` counts
   proper crossings on the ``num_points``-point resampled loop.  For environments
   where the Bézier curve self-crosses, ``_assemble_polygon_selected_k`` emits the
   straight corner polygon in Bernstein form (equivalent to ``handle_clamp_frac=0``;
   it rounds differently from ``ops.closed_polyline``'s lerp, so bezier keeps its own
   kernel) and ``ops.arc_resample(..., active=crossers)`` overwrites only those rows.
   ``_select_vec2_k`` then writes the Bézier result for non-crossers and the polygon
   result for crossers into the output buffer.

9. **Generation flag.**  All environments are marked ``out_valid_wp=1`` unconditionally.
   The generation gate is always True; geometric validity is decided later by the
   shared post-relax inflation gate.

Math
----

**Cubic Bézier segment.**  Each corner-to-corner segment is a standard degree-3
Bézier curve parameterised by :math:`t \in [0, 1]`:

.. math::

   B(t) = (1-t)^3 P_0 + 3(1-t)^2 t\, P_1 + 3(1-t)t^2 P_2 + t^3 P_3

where :math:`P_0 = c_0` and :math:`P_3 = c_1` are the adjacent corner anchors, and
:math:`P_1`, :math:`P_2` are the interior control (handle) points constructed below.

**Vertex tangent blend.**  The tangent direction at corner :math:`i` is:

.. math::

   p = \frac{\arctan(\texttt{edgy})}{\pi} + 0.5,
   \qquad
   \hat{t}_i = \operatorname{normalize}\!\bigl(p\,\hat{u}^{\text{out}}_i
               + (1-p)\,\hat{u}^{\text{in}}_i\bigr)

where :math:`\hat{u}^{\text{out}}_i` is the unit vector from corner :math:`i` to
corner :math:`i{+}1` and :math:`\hat{u}^{\text{in}}_i` is the unit vector from corner
:math:`i{-}1` to corner :math:`i`.  At ``edgy=0``, :math:`p = 0.5` and the tangent
bisects the two edge directions symmetrically.  As ``edgy`` grows, :math:`p \to 1`
and the tangent aligns with the outgoing edge, sharpening the exit from each corner.

**Handle construction.**  For each segment from :math:`c_0` to :math:`c_1` with
chord :math:`\|c_1 - c_0\|`:

.. math::

   h_i = \min\!\bigl(\texttt{rad} \cdot \|c_1 - c_0\|,\;
                      \texttt{handle\_clamp\_frac} \cdot l_{\min,i}\bigr)

where :math:`l_{\min,i}` is the shorter of the two edges incident at corner :math:`i`.
The interior control points follow:

.. math::

   P_1 = c_0 + \hat{t}_0 \cdot h_0, \qquad
   P_2 = c_1 - \hat{t}_1 \cdot h_1

The clamp prevents :math:`h_i` from exceeding ``handle_clamp_frac`` times the
shorter adjacent edge, so a long handle cannot overshoot a nearby corner and cause
a self-crossing.

Parameters
----------

All fields are part of ``TrackGenConfig`` (see the Configuration reference for the
full list, defaults, and validation rules).

The generator-specific fields below are also the fields of ``BezierParams``, the
generator's Params class (core fields such as ``scale`` are not): set them flat or
through ``generator_params`` (see :ref:`builtin-generator-params`). Their rules are
checked when the ``TrackGenerator`` is constructed, and only for ``generator="bezier"``.

``min_num_points``
    Minimum corner-anchor count per environment.  Must be >= 2.

``max_num_points``
    Maximum corner-anchor count per environment.  A wider range adds more
    per-environment shape diversity.  Also sets the buffer stride ``P``.

``num_points_per_segment``
    Dense Bézier samples emitted per corner-to-corner segment before arc-resampling.
    Must be >= 2; higher values give the arc-resampler a finer cumulative-arc-length
    grid, reducing interpolation error.

``min_point_distance``
    Minimum grid-cell spacing for corner sampling.  Controls the corner grid as
    ``num_cells = int(1 / (min_point_distance * 2))``.  Must be in ``(0, 0.5]``.

``min_angle``
    Deprecated, and not part of ``BezierParams``: ``TrackGenerator`` never reads it (only
    a corner-angle kernel the tests launch does). A non-default value emits a
    ``DeprecationWarning``; the field will be removed in 1.0.

``rad``
    Handle length as a fraction of the segment chord — the primary curviness dial.
    Only live when ``rad <= handle_clamp_frac``; if ``rad`` exceeds the clamp,
    ``handle_clamp_frac`` binds every segment and ``rad`` has no independent effect.
    Dynamic, in ``[0, 1]`` (see below).

``edgy``
    Corner-tangent blend weight.  Converted as ``p = atan(edgy) / π + 0.5``.
    At the default ``0.0``, ``p = 0.5`` gives a symmetric bisector tangent; larger
    positive values push ``p`` toward 1, making turns asymmetrically sharper
    (outgoing-edge-dominant).

``scale``
    Isotropic scale multiplier applied to sampled corner positions.  Scales all
    track coordinates; interacts with ``spacing`` and ``half_width`` through the
    coordinate range.

``handle_clamp_frac``
    Adaptive handle clamp.  Each corner's handle is capped at
    ``handle_clamp_frac × (its shorter incident edge)`` to prevent overshoot past
    a nearby corner.  Set equal to ``rad`` (the default ``0.4``) so the clamp only
    trims genuine overshoot; set below ``rad`` to bind every handle regardless of
    ``rad`` (near-polygonal tracks); set very large to disable. Dynamic, ``>= 0``
    (see below).

``style_sampling``
    Opt-in per-environment Bézier style randomisation (default ``False``).  When
    ``True``, each environment draws its own ``rad``, ``scale``, and
    ``handle_clamp_frac`` from the ``*_range`` fields, so a single batch spans a
    family of styles. Still supported; for ``rad`` and ``handle_clamp_frac`` it is
    superseded by ``randomize=`` (`Style sampling and randomize`_).

``rad_range``
    ``(lo, hi)`` range for per-environment ``rad`` when ``style_sampling=True``.
    ``None`` leaves ``rad`` to its per-env parameter array (every env at the configured
    ``rad`` unless you ``set`` or randomize it). Superseded by
    ``randomize={"generator.rad": (lo, hi)}``.

``scale_range``
    ``(lo, hi)`` range for per-environment ``scale`` when ``style_sampling=True``.
    ``None`` collapses to the scalar ``scale`` for all environments. Fully supported:
    it is the only per-env size control until roadmap PR13d.

``handle_clamp_frac_range``
    ``(lo, hi)`` range for per-environment ``handle_clamp_frac`` when
    ``style_sampling=True``.  ``None`` leaves the clamp to its per-env parameter array.
    Superseded by ``randomize={"generator.handle_clamp_frac": (lo, hi)}``.

.. _bezier-dynamic:

Per-env (dynamic) parameters
----------------------------

``rad`` and ``handle_clamp_frac`` are **dynamic** fields of ``BezierParams``: each is a
per-env ``[E]`` ``float32`` array, initialised from the configured value. Draw them per
env with ``presets.track(randomize={"generator.rad": (lo, hi)})``, write them with
``gen.params["generator"].set("rad", value_or_array)``, or from a kernel in your own CUDA
graph (a curriculum): see :ref:`tuning-per-env`. A scalar outside the bounds raises
``ValueError`` (at construction, or from ``set``); an array value outside them (a
``set`` with an ``[E]`` array, your own kernel) is clamped to them when the kernel reads
it.

.. list-table::
   :header-rows: 1
   :widths: 22 9 12 22 22 13

   * - Field (``randomize=`` key)
     - Default
     - Bounds
     - Harder / more varied when
     - Effective range
     - Yield at lo / default / hi
   * - ``rad`` (``generator.rad``)
     - ``0.4``
     - ``[0, 1]``
     - lower: sharper, more polygonal corners
     - ``[0, handle_clamp_frac]``: above the clamp ``rad`` has no effect
     - 1.00 / 1.00 / 1.00
   * - ``handle_clamp_frac`` (``generator.handle_clamp_frac``)
     - ``0.4``
     - ``>= 0``
     - lower: near-polygonal (every handle bound by its shorter edge)
     - ``[0, ~rad]``; a very large value disables the clamp
     - 1.00 at ``0``, ``0.4`` and ``1e9``

*Yield* is the fraction of valid tracks with the default pipeline, ``E = 128``, seeds
``0..127`` (CPU), with the knob set to the value for every env.

**Mind the clamp.** Each handle is ``min(rad * chord, handle_clamp_frac *
shorter_edge)``, and the shorter incident edge is never longer than the chord, so
``rad >= handle_clamp_frac`` leaves ``rad`` inert: the defaults (``rad =
handle_clamp_frac = 0.4``) sit exactly at saturation. To vary curviness, keep the
``rad`` range at or below the clamp (``randomize={"generator.rad": (0.1, 0.4)}``); when
you draw ``handle_clamp_frac`` too, keep ``hi(rad) <= lo(handle_clamp_frac)``, or accept
that ``rad`` is inert in the envs where its draw lands at or above the clamp's. The
build warns about a ``rad`` range above the clamp (:ref:`tuning-per-env`).

**Static knobs, and why.** ``min_num_points`` / ``max_num_points`` (``max`` sizes the
corner buffers; a dynamic corner count is planned for roadmap PR13d),
``num_points_per_segment`` (sizes the dense buffer and its loops),
``min_point_distance`` (sets the corner grid on the host), ``edgy`` (turned into
``atan(edgy) / pi + 0.5`` on the host in double precision, which a kernel cannot
reproduce bit for bit: planned for PR16), the ``style_sampling`` fields (a Python branch
and host ranges) and the core ``scale`` (per-env size: PR13d; until then
``scale_range``).

.. _bezier-style-vs-randomize:

Style sampling and randomize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``style_sampling`` is kept, and not deprecated in this release. Its ``rad_range`` and
``handle_clamp_frac_range`` are superseded by ``randomize=`` on the two dynamic fields
(the same uniform distributions, drawn from different random streams, so the tracks
differ); ``scale_range`` has no replacement until per-env size ships (roadmap PR13d,
which will deprecate ``style_sampling``):

.. code-block:: python

   # style sampling (still supported)
   TrackGenConfig(style_sampling=True, rad_range=(0.1, 0.35),
                  handle_clamp_frac_range=(0.3, 0.9))

   # randomize: the same distributions, other draws (the style ranges, for comparison:
   # rad is inert in the envs where its draw lands at or above the clamp's, see above)
   spec = presets.track(num_envs=E, device=device,
                        randomize={"generator.rad": (0.1, 0.35),
                                   "generator.handle_clamp_frac": (0.3, 0.9)})

   # randomize on a pipeline built from a config
   gen = TrackGenerator(
       legacy.to_pipeline_spec(cfg).with_randomize(
           {"generator.rad": (0.1, 0.35), "generator.handle_clamp_frac": (0.3, 0.9)}),
       rng)

Under ``style_sampling=True`` a knob with a range is **drawn by the style kernel**: its
parameter array holds each env's draw after a generation (read it to log what each env
was built with), ``gen.params["generator"].drawn`` maps it to the stage's own id
(``"generator"``), ``set`` on it raises ``ValueError`` (``generator.rad is drawn every
generation by the generator's style_sampling (rad_range); set rad_range=None to set
it``), and a randomizer on it is a ``PlanError``. A knob whose range is ``None`` stays an
ordinary dynamic field:
``set`` and ``randomize=`` work on it. The bytes of every ``style_sampling``
configuration are unchanged.

What Makes It Distinct
----------------------

The Bezier generator is the **default** and most knob-rich of the six generators:

- **Widest style control.** Eight primary knobs (``min_num_points``, ``max_num_points``,
  ``num_points_per_segment``, ``min_point_distance``, ``rad``, ``edgy``, ``scale``,
  ``handle_clamp_frac``) plus the opt-in ``style_sampling`` family are all
  exclusive to the corner-Bézier representation.  No other generator exposes handle
  geometry, and only ``style_sampling`` varies ``scale`` per env (every built-in
  sampling generator, ``bezier``, ``hull``, ``polar``, ``voronoi`` and ``checkpoint``,
  has per-env shape knobs for ``randomize=``).

- **Cubic Bézier representation.** Compared to the **hull** generator (which uses a
  closed uniform Catmull-Rom spline) and the **polar** generator (also Catmull-Rom),
  the Bézier formulation gives explicit, per-corner handle control through ``rad`` and
  ``edgy``.  The hull generator adds radial midpoint displacement for pronounced lobes
  and pinches; the Bézier generator keeps the shape entirely corner-determined.

- **Star-shaped corner families.** Like the hull generator the Bézier generator samples
  Cartesian corners and angle-sorts them.  Unlike the **polar** generator, which starts
  from polar control knots and is smooth and centered by construction, and unlike the
  **voronoi** generator, which snaps anchors to a pre-sampled site field, the Bézier
  generator's shapes are anchored to sampled grid corners and are roughly star-shaped
  around the corner centroid.  Unlike the **checkpoint** generator — which steers a
  bounded-turn path through radial checkpoints and produces organically curving loops —
  the Bézier generator is purely geometric and not trajectory-based.

- **Lowest relaxation burden at the default config.** At the default
  ``handle_clamp_frac=0.4`` (matching ``rad=0.4``) and fat-band settings the valid
  yield is approximately 99.9 % end-to-end, and the generator produces consistently
  smooth, well-spaced centerlines before relaxation begins.

Fallback and Validity
---------------------

**Local polygon fallback.**  After arc-resampling to ``num_points``, any environment
whose Bézier centerline contains at least one proper self-crossing is redirected to its
provably simple corner polygon (straight line segments between the sorted corners).
Only the affected environments are reprocessed; non-crossing environments are
unaffected.  The XPBD bead-chain solver then re-rounds the polygon fallback through its
bending and separation corrections.

**Generation-stage validity.**  The generator writes ``out_valid_wp=1`` for every
environment unconditionally.  The generation stage does not gate on geometry.  Turning
number, track thickness, NaN presence, and width-floor checks are all evaluated later
by the shared post-relax inflation validity gate (``_validity_k`` inside
``inflate_warp``).  The final ``Track.valid`` field therefore reflects post-relax
geometric quality, not generation success.
