Track relaxation — setup and constraints
========================================

Relaxation is a PBD/XPBD-style position projection over the constant-spacing centerline
beads. It does not allocate dense ``[E,N,N]`` tensors, does not run an optimizer, and
does not store a per-constraint Lagrange-multiplier history. Instead, every fixed sweep
reads one complete position buffer, computes local position corrections, and writes a
second buffer (see :doc:`solver`). This page covers the per-env **setup** and the three
**corrections** each sweep applies, then the **gate coupling** that ties the separation
target to the downstream validity gate.

Setup
-----

Setup happens once, before the sweeps, in ``band_l0_inplace`` (kernel ``_band_l0_k``):

- :math:`L_0[e] = \text{perimeter}(\text{center}[e]) / \text{count}[e]`, the rest edge
  length for the constant-spacing bead chain.
- :math:`\text{band}[e] = \text{round}(2 \cdot \text{half\_width} / L_0[e])`, clamped to
  a minimum of 1, unless ``config.relax_band`` overrides it. The band excludes immediate
  geometric neighbours from the separation scan, so the road does not try to push apart
  points that are adjacent along the same local segment. The identical
  ``_separation_band`` ``@wp.func`` is shared with the pipeline's ``_validity_k``, so the
  band definition lives in exactly one place.
- :math:`\text{target} = 2 \cdot \text{half\_width} \cdot (1 + \text{relax\_margin})` is
  the non-local separation distance — a slightly inflated road diameter (default
  ``relax_margin`` = 0.15).
- :math:`R_{\min} = \text{half\_width} \cdot (1 + \text{relax\_margin})` is the local
  curvature-radius target used by the bending correction. Note :math:`\text{target} =
  2\,R_{\min}` exactly.

Per-sweep corrections
---------------------

Each sweep applies three corrections per real bead ``i`` (kernels ``_step_kernel`` /
``_step_cached_kernel``):

Non-local separation
~~~~~~~~~~~~~~~~~~~~~~

For every bead ``j`` whose circular index distance from ``i`` exceeds ``band[e]``, if
the current Euclidean distance is below ``target``, bead ``i`` receives a push along
:math:`x_i - x_j` of magnitude :math:`0.5\,(\text{target} - \text{dist})/\text{dist}`.
The pushes are **averaged** over all colliding candidates for that bead (Jacobi
averaging), then scaled by ``relax_sep_relax``. This is the term that opens
self-approaches and makes enough room for a constant-width road.

Edge spacing
~~~~~~~~~~~~~

The two incident edges ``(i-1, i)`` and ``(i, i+1)`` are corrected toward the rest
length :math:`L_0[e]` with the local formula
:math:`0.25\big(\tfrac{\ell_n - L_0}{\ell_n}d_n - \tfrac{\ell_p - L_0}{\ell_p}d_p\big)`,
scaled by ``relax_spc_relax``, so the relaxed loop keeps near-constant bead spacing
instead of stretching into a few long chords.

Bending / radius guard
~~~~~~~~~~~~~~~~~~~~~~~~

The local Menger curvature through ``(i-1, i, i+1)`` gives a radius estimate. If
:math:`\text{radius} < R_{\min}`, the bead is pushed toward the midpoint of its
neighbours by a scale of
:math:`\text{relax\_bend\_relax}\cdot (R_{\min} - \text{radius})/R_{\min}`, **clamped to
1** so the bead never passes the chord midpoint in a single sweep (the flip clamp). This
removes jagged under-radius corners introduced by generation or by separation pushes.

The three corrections sum into a single per-bead step
:math:`\Delta x = \text{sr}\cdot\text{sep} + \text{pr}\cdot\text{spc} +
\text{bscale}\cdot\text{toward}` written to the output buffer.

Count-aware kernels
-------------------

The solver is count-aware. Kernels launch over the static stride ``N_max``, but threads
with ``i >= count[e]`` copy the NaN-padded tail through unchanged. When every
``count[e] == N_max``, the same kernels reduce to the old fixed-N parity path.

.. _relax-knobs:

The ``relax_*`` knobs
---------------------

.. list-table::
   :header-rows: 1

   * - Knob
     - Effect
   * - ``relax_sep_relax``
     - Scale factor applied to each separation push (damping). Default 1.0.
   * - ``relax_spc_relax``
     - Scale factor applied to edge-spacing corrections. Default 1.0.
   * - ``relax_bend_relax``
     - Scale factor applied to bending/radius-guard corrections. Default 1.5
       (over-corrects slightly to remove sharp corners quickly).
   * - ``relax_margin``
     - Inflation margin: ``target = 2*half_width*(1+relax_margin)``. Default 0.15.
   * - ``relax_band``
     - Override the auto-computed separation exclusion band. Default ``None`` (auto).
   * - ``relax_iters``
     - Number of Jacobi sweeps. Default 50 (see :doc:`convergence`).

The separation execution knobs (``relax_sep_every``, ``relax_sep_cache_slots``,
``relax_sep_cache_skin``) and the Chebyshev knobs (``relax_accel``, ``relax_cheby_rho``,
``relax_cheby_gamma``, ``relax_cheby_start``) are covered in :doc:`solver`.

.. _relax-gate-coupling:

Gate coupling — a load-bearing overshoot
----------------------------------------

The separation target is **not** the validity gate's floor; it deliberately overshoots
it, and that overshoot is load-bearing. Recent experiments established the relationship
and, critically, that weakening it collapses yield.

At the library defaults (``half_width`` = ``hw``, ``relax_margin`` = 0.15,
``relax_tol`` = 0.02):

- the separation **target** is :math:`2\,\text{hw}\,(1.15) = 2.3\,\text{hw}`, and
  :math:`\text{target} = 2\,R_{\min}` exactly;
- the validity **gate** requires :math:`\tfrac{1}{2}\,\text{sep}_{\min} \ge (1 -
  \text{relax\_tol})\,\text{hw}`, i.e. :math:`\text{sep}_{\min} \ge 1.96\,\text{hw}`.

So the solver aims a bead pair across a legal minimum-radius hairpin at
:math:`2.3\,\text{hw}` while the gate would accept it at :math:`1.96\,\text{hw}`.
Because :math:`\text{target} = 2\,R_{\min}`, a pair straddling a legal
minimum-radius hairpin sits at **zero separation slack** relative to the target — the
constraint pushes it exactly as hard as the geometry allows. At converged equilibrium
the solver therefore holds tight hairpins a comfortable **~4–8% above the gate line**,
not on it.

That headroom is what carries yield. Two changes that *look* like harmless efficiency
tweaks were both measured to be dead ends because they let near-band pairs settle onto
the gate floor instead of above it:

- **Widening the exclusion band** (``relax_band`` = auto + 1) removes the near-hairpin
  pairs from the separation scan entirely. Defaults-regime yield collapsed
  **0.99 → 0.74**.
- **Capping near-band pair targets at the chord-feasible distance** (only pushing pairs
  apart to the distance a minimum-radius arc geometrically permits, rather than to the
  full ``target``) removes exactly the overshoot. Near-band pairs then settle onto the
  gate floor and float across it under float noise, again failing the gate.

.. warning::

   These are recorded as **measured dead ends**, not options. Do not widen the exclusion
   band and do not cap near-band pair targets at the chord-feasible distance: both let
   near-band pairs settle onto the gate floor, and yield falls off a cliff. The overshoot
   (``target`` = ``2*R_min`` = ``2.3*hw`` at defaults) is deliberate margin — keep it.

Disabling relaxation
--------------------

If ``relax_enable=False``, ``_run_pipeline`` bypasses ``band_l0_inplace`` and
``xpbd_solve_inplace`` and inflates the constant-spacing centerline directly (an
identity pass-through, useful for ablations). Otherwise the relaxed output is
re-uniformized by the inflation stage before frame, offset, validity, and arclength are
computed.
