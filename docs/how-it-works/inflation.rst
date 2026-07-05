Inflation — borders, validity, and the ``Track`` result
=========================================================

``inflate_warp(center, config, valid=None, count=None)`` is the final pipeline stage. It
composes several sub-stages to produce the complete ``Track`` named-tuple from the relaxed
centerline.

Composition
-----------

The function composes in order:

1. ``resample_uniform`` — re-uniformize the relaxed bead chain so that frame and curvature
   estimates are not biased by residual spacing irregularity.
2. ``frame_curvature`` (``_frame_k``) — central-difference unit tangent, left-normal, and
   Menger curvature at each bead.
3. **Constant half-width offset** — ``offset`` (``_offset_build_k`` + ``_offset_assign_k``)
   shifts every centerline point by ``±w`` along the normal. The outer border is chosen as
   the larger-signed-area candidate, so the convention is consistent even for
   counter-clockwise layouts.
4. ``validity`` (``_validity_k``) — a single per-env kernel that combines all geometric
   acceptance criteria (see below).
5. ``_arclength_k`` — computes cumulative arc length along the centerline for each valid
   env.
6. Returns a ``Track`` named-tuple.

The offset, validity, and arclength stages are count-aware — they operate over each env's
``count[e]`` real points. The fixed-N parity path is ``count[e] == N``.

Validity gate — ``_validity_k``
--------------------------------

``_validity_k`` is a single per-env kernel that combines the following checks:

- **Generation flag** — all True now that generation no longer gates; validity is purely
  geometric.
- **Closed-loop turning ≈ 2π** — the net turning number must be approximately one full
  winding.
- **Width floor** — minimum track width must exceed the configured floor.
- **No NaN** — all points in the env's real-point range must be finite.
- **Thickness ≥** ``(1 − relax_tol) · half_width`` — the inflated road must not collapse
  below the required half-width (within tolerance).
- **Border self-intersections** — only when ``validity_border_check`` is set (**default
  off**). This check is redundant with the thickness/separation gate: a crossing or
  fat-band overlap drives ``separation_min → 0 → thickness < half_width → invalid`` anyway.

Index-aligned borders
---------------------

The inner and outer borders returned in ``Track`` are index-aligned with the centerline:
``outer[e, i]``, ``center[e, i]``, and ``inner[e, i]`` are all offset from the same
centerline bead ``i``, so downstream consumers can index them identically.

Half-width recovery
-------------------

The half-width at each bead can be recovered as
``‖outer[e, i] − center[e, i]‖ = ‖center[e, i] − inner[e, i]‖ = half_width``
(constant by construction, modulo floating-point rounding). The ``thickness`` validity
metric checks the minimum over all beads to ensure no inflation collapsed.

``Track`` fields
----------------

.. list-table::
   :header-rows: 1

   * - Field
     - Shape
     - Description
   * - ``outer``
     - ``[E, N_max, 2]``
     - Outer border points (NaN-padded past ``count[e]``).
   * - ``center``
     - ``[E, N_max, 2]``
     - Re-uniformized centerline points.
   * - ``inner``
     - ``[E, N_max, 2]``
     - Inner border points (NaN-padded past ``count[e]``).
   * - ``tangent``
     - ``[E, N_max, 2]``
     - Unit tangent at each centerline bead.
   * - ``normal``
     - ``[E, N_max, 2]``
     - Left-normal ``(-tangent.y, tangent.x)`` at each centerline bead; the boundary
       it faces is winding-dependent (see ``winding``), so read ``outer``/``inner``
       to identify the borders.
   * - ``arclen``
     - ``[E, N_max]``
     - Cumulative arc length at each bead (0 at bead 0).
   * - ``length``
     - ``[E]``
     - Total track perimeter per env.
   * - ``valid``
     - ``[E]``
     - Boolean validity mask (all criteria above must pass).
   * - ``count``
     - ``[E]``
     - Number of real beads per env (padding is NaN past this index).
   * - ``winding``
     - ``[E]``
     - Signed loop winding: ``+1.0`` CCW, ``-1.0`` CW, ``0.0`` degenerate
       (sign of the centerline's signed area).

``generate()`` always returns the **same** ``Track`` instance with stable ``wp.array``
pointers. Callers that need a snapshot should use ``Track.clone()``.
