Inflation — borders, validity, and the ``Track`` result
=========================================================

*Inflation* is the tail of the track pipeline: the :term:`stages <stage>` after
relaxation that turn the relaxed bead chain into the road band, judge it, and lift it to
3D. In the default pipeline (:ref:`pipeline-stages-the-default-track-stages`) they are
``uniform``, ``frames``, ``offset``, ``elevation``, ``validity`` and ``lift``.

Composition
-----------

In order, for every :term:`env`:

1. ``uniform`` (``resample.uniform``: ``_resample_scan_k`` + ``_resample_lookup_k``)
   re-uniformises the relaxed bead chain, so that frame and curvature estimates are not
   biased by residual spacing irregularity (:doc:`resample`).
2. ``frames`` (``_frame_k``, ``_arclength_k``, ``_winding_k``): central-difference unit
   tangent, left normal and Menger curvature at each bead, the plan-view cumulative arc
   length and loop length, and the signed winding.
3. ``offset`` (``_offset_build_k`` + ``_offset_assign_k``) shifts every centerline point
   by ``±w`` along the normal, ``w`` the point's half-width (``cl.half_width``). The outer
   border is chosen as the larger-signed-area candidate, so the convention is consistent
   for clockwise and counter-clockwise layouts.
4. ``elevation`` writes the per-point altitude (nothing at the default ``flat``).
5. ``validity`` (``_validity_k``): one per-env kernel that combines every geometric
   acceptance criterion into the env's :term:`fail bits` (below).
6. ``lift`` (``_lift_track_k``) builds the 3D ``Track`` arrays from the plan-view band
   and the altitude.

These stages are count-aware: they operate over each env's ``count[e]`` real points.
``inflate_warp(center, config, ...)`` in
``track_gen/_src/warp_pipeline.py`` is the same tail as one scalar composition; the
pipeline does not call it, and the tests keep it as a byte-identity reference.

Validity gate — ``_validity_k``
--------------------------------

``_validity_k`` is a single per-env kernel that combines the following checks, one fail
bit each (the bit table is in :ref:`pipeline-fail-bits`):

- **Generation flag**: the generator's ``gen.valid`` (``1`` for every env of the
  built-ins: validity is purely geometric).
- **Closed-loop turning ≈ 2π**: the net turning number must be approximately one full
  winding.
- **No NaN**: all points in the env's real-point range must be finite.
- **Thickness ≥** ``(1 − relax_tol) · half_width``: the inflated road must not collapse
  below the required half-width (within tolerance).
- **Width floor**: every real point's half-width must exceed ``w_floor``.
- **Border self-intersections**: only when ``validity_border_check`` is set (**default
  off**). This check is redundant with the thickness gate: a crossing or fat-band overlap
  drives ``separation_min → 0 → thickness < half_width → invalid`` anyway.
- **Grade**: a segment's ``|dz| / ds`` must not exceed ``z_valid_grade`` (only when it is
  ``> 0``).
- **Not truncated at** ``N_max`` (``Track.truncated == 0``): the constant-spacing resample
  flags any env whose loop needed more than ``N_max`` points; its clamped loop closes with
  a chord across the dropped arc, so the env fails closed.

An env with any bit set is an :term:`invalid env` (``Track.valid == 0``); what the result
then holds is set by the commit (:ref:`pipeline-commit`), and :ref:`invalid-envs`
covers it from the user's side.

Index-aligned borders
---------------------

The inner and outer borders are index-aligned with the centerline: ``outer[e * N_max +
i]``, ``center[e * N_max + i]`` and ``inner[e * N_max + i]`` are all offset from the same
centerline bead ``i``, so downstream consumers can index them identically.

Half-width recovery
-------------------

At constant width the half-width at each bead can be recovered as
``‖outer − center‖ = ‖center − inner‖ = half_width`` (modulo floating-point rounding);
``Track.half_width`` stores it per point (NaN padding). The ``thickness`` validity metric
checks the minimum over all beads to ensure no inflation collapsed. In a pipeline whose
width stage writes a per-point ``cl.half_width`` (anything but ``width.constant``), the
offset, the thickness check and the width floor use each point's own half-width instead;
see :ref:`pipeline-width`.

``Track`` geometry
------------------

Every per-point array is flat, ``[E * N_max]``, NaN-padded past ``count[e]``
(:doc:`conventions`); :ref:`results` documents every field of the ``Track``.

.. list-table::
   :header-rows: 1

   * - Field
     - Shape, dtype
     - Description
   * - ``outer``, ``center``, ``inner``
     - ``[E * N_max]`` ``vec3f``
     - Outer border, re-uniformised centerline, inner border (``z`` the altitude).
   * - ``tangent``
     - ``[E * N_max]`` ``vec3f``
     - Unit tangent at each centerline bead.
   * - ``normal``
     - ``[E * N_max]`` ``vec3f``
     - Left normal at each centerline bead; the border it faces depends on the winding
       (see ``winding``), so read ``outer`` / ``inner`` to identify the borders.
   * - ``arclen``, ``half_width``
     - ``[E * N_max]`` ``float32``
     - Cumulative arc length (0 at bead 0) and the band's half-width per bead.
   * - ``length``, ``winding``
     - ``[E]`` ``float32``
     - Loop length; signed winding (``+1.0`` CCW, ``-1.0`` CW, ``0.0`` degenerate: the
       sign of the centerline's signed area).
   * - ``count``, ``valid``, ``truncated``, ``fail_mask``, ``generation``
     - ``[E]`` ``int32``
     - Real beads per env; the validity flag; the truncation flag; the fail bits; the
       :term:`epoch` the env's current output was built with
       (:ref:`cuda-graph-masked-regeneration`).

``generate()`` always returns the **same** ``Track`` instance with stable ``wp.array``
pointers. Callers that need a snapshot use ``Track.clone()``.
