Relaxation — PBD/XPBD bead-chain solve
========================================

Relaxation lives in ``track_gen/_src/warp_relax.py`` and is the only production relax
backend used by ``TrackGenerator`` (``relax_solver="xpbd"``, ``smooth_finish=False``). The
implementation is a PBD/XPBD-style position projection over the constant-spacing centerline
beads: it does not allocate dense ``[E,N,N]`` tensors, does not run an optimizer, and does
not store a per-constraint Lagrange multiplier history. Instead, every fixed sweep reads one
complete position buffer, computes local position corrections, writes a second buffer, and
swaps the two buffers. That gives Jacobi semantics and keeps the solve graph-capturable.

Setup
-----

Setup happens before the sweep in ``band_l0_inplace``:

- :math:`L_0[e] = \text{perimeter}(\text{center}[e]) / \text{count}[e]`, the rest edge
  length for the constant-spacing bead chain.
- :math:`\text{band}[e] = \text{round}(2 \cdot \text{half\_width} / L_0[e]).\text{clamp\_min}(1)`
  unless ``config.relax_band`` overrides it. The band excludes immediate geometric
  neighbours from the separation scan, so the road does not try to push apart points that
  are adjacent along the same local segment.
- :math:`\text{target} = 2 \cdot \text{half\_width} \cdot (1 + \text{relax\_margin})` is
  the non-local separation distance. It is a slightly inflated road diameter, so relaxation
  leaves margin for the later thickness gate.
- :math:`R_{\min} = \text{half\_width} \cdot (1 + \text{relax\_margin})` is the local
  curvature-radius target used by the bending correction.

Per-sweep corrections
---------------------

Each sweep applies three corrections per real bead ``i``:

Non-local separation
~~~~~~~~~~~~~~~~~~~~

For every bead ``j`` with circular index distance greater than ``band[e]``, if the current
Euclidean distance is below ``target``, bead ``i`` receives a push along ``xi - xj`` of
``0.5 * (target - dist) / dist``. The pushes are averaged over all colliding candidates
for that bead, then scaled by ``relax_sep_relax``. This is the term that opens
self-approaches and makes enough room for a constant-width road.

Edge spacing
~~~~~~~~~~~~

The two incident edges ``(i-1,i)`` and ``(i,i+1)`` are corrected toward rest length
:math:`L_0[e]`. The implementation uses the local formula from ``_step_kernel``, scaled by
``relax_spc_relax``, so the relaxed loop keeps near-constant bead spacing instead of
stretching into a few long chords.

Bending / radius guard
~~~~~~~~~~~~~~~~~~~~~~

The local Menger curvature through ``(i-1,i,i+1)`` gives a radius estimate. If
``radius < R_min``, the bead is pushed toward the midpoint of its neighbours. The scale is
``relax_bend_relax * (R_min - radius) / R_min``, clamped to ``1``, so the bead never
passes the chord midpoint in a single sweep. This removes jagged under-radius corners
introduced by generation or by separation pushes.

Count-aware kernels
-------------------

The solver is count-aware. Kernels launch over the static stride ``N_max``, but threads
with ``i >= count[e]`` copy through the NaN-padded tail. When every ``count[e] == N_max``,
the same kernels reduce to the old fixed-N parity path.

Separation execution modes
--------------------------

Dense / cadenced mode
~~~~~~~~~~~~~~~~~~~~~

``_step_kernel`` scans all non-band neighbours in :math:`O(\text{count}[e]^2)` whenever
``step_i % relax_sep_every == 0``; spacing and bending still run every sweep. If
``relax_sep_cache_slots == 0`` and ``relax_sep_every > 1``, this is a naive skip cadence:
there is no separation force between dense scans.

Broadphase-cached mode
~~~~~~~~~~~~~~~~~~~~~~

When ``relax_sep_cache_slots > 0`` and ``relax_sep_every > 1``,
``_build_sep_cache_kernel`` refreshes a fixed-slot directed candidate list every
``relax_sep_every`` sweeps using radius ``target*(1+relax_sep_cache_skin)``. Then
``_step_cached_kernel`` runs every sweep, re-testing each cached candidate with the exact
current ``dist < target`` narrowphase before applying the separation push. Cache arrays live
in ``RelaxScratch``, including an overflow counter for beads whose candidate list exceeded
the configured slot count, so the mode remains allocation-free and CUDA-graph-capturable.

The ``relax_sep_*`` knobs
--------------------------

.. list-table::
   :header-rows: 1

   * - Knob
     - Effect
   * - ``relax_sep_relax``
     - Scale factor applied to each separation push (damping).
   * - ``relax_spc_relax``
     - Scale factor applied to edge-spacing corrections.
   * - ``relax_bend_relax``
     - Scale factor applied to bending/radius-guard corrections.
   * - ``relax_sep_every``
     - Run the dense or cached separation scan every this many sweeps.
   * - ``relax_sep_cache_slots``
     - Number of candidate slots per bead for cached broadphase (0 = dense mode).
   * - ``relax_sep_cache_skin``
     - Skin factor: cache radius = ``target * (1 + relax_sep_cache_skin)``.
   * - ``relax_band``
     - Override the auto-computed band (minimum circular index distance for non-local separation).
   * - ``relax_margin``
     - Inflation margin: ``target = 2*half_width*(1+relax_margin)``.

Disabling relaxation
--------------------

If ``relax_enable=False``, ``_run_pipeline`` bypasses ``band_l0_inplace`` and
``xpbd_solve_inplace`` and inflates the constant-spacing centerline directly. Otherwise the
relaxed output is re-uniformized by ``inflate_warp`` before frame, offset, validity, and
arclength are computed.
