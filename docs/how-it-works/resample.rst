:orphan:

Resample — constant-spacing arc-length resampling
===================================================

The pipeline provides three arc-length resamplers, but only one output mode.

Resampler overview
------------------

``arc_length_resample_warp(points[E,M,2], num)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general, **NaN-aware** resampler (kernels ``_arc_scan_k`` + ``_arc_lookup_k``). It
compacts the finite points per env (drops NaN, in order), builds the closed-loop cumulative
arc length in ``float64``, and looks up ``num`` arc-uniform targets (searchsorted + lerp).
Envs with fewer than 2 real points yield an all-NaN row and ``count 0``. Fused into the
generator output as the dense-to-``num_points`` resample and reused by the selected
polygon-fallback de-cross path; also used by the standalone ``gates`` parity wrapper
(dense-to-``num_points`` and dense-to-``num_points_per_segment``), which the torch oracle
tests exercise but the single-pass generator no longer calls.

``resample_uniform(center[E,N,2], n, count=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The count-aware N→N re-uniformizer (``_resample_scan_k`` + ``_resample_lookup_k``), used
after relax (and inside ``inflate_warp``). With ``count=None`` all ``E*N`` points are real;
with ``count`` it re-uniformizes each env's ``count[e]`` real points (NaN-padded past
``count[e]``).

``resample_constant_spacing(center[E,N,2], spacing, N_max)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The count-aware resampler selected by ``output_mode="constant_spacing"``. From a fixed
source it picks a per-track ``count = floor(perimeter/spacing)+1`` (decremented while
``(count-1)·spacing ≥ perimeter``, capped at ``N_max``), lays the arc-uniform points into
an ``[E, N_max, 2]`` buffer NaN-padded past ``count[e]``, and matches the
``geometry.arc_length_resample(spacing=)`` oracle.

Why ``constant_spacing`` is the only output mode
-------------------------------------------------

``constant_spacing`` is the **only** output mode (the dataclass enforces it; any other
``output_mode`` raises in ``__post_init__``). Each track is relaxed and emitted at a
constant arc spacing: a per-track ``count[e] = floor(perimeter/spacing)+1`` (decremented
while ``(count-1)·spacing ≥ perimeter``, capped at ``N_max``, NaN-padded past
``count[e]``).

The legacy ``fixed`` mode — every track padded to a constant point count (``num_points``)
— was **dropped**. A fixed 256 points **over-resolves** the centerline relative to its
half-width (segment ≈ 0.2 m ≪ a 0.5 m half-width), so the slow Jacobi XPBD solve
**under-converges** under the fixed iteration count, producing jagged tracks whose road
self-overlaps. Relaxing at a width-appropriate spacing instead lets the same solve converge,
yielding smooth, valid tracks on fewer nodes per track (so it is also faster).
``num_points`` survives only as the intermediate dense-resample resolution before the
constant-spacing step.

``count[e]`` derivation
-----------------------

For each environment ``e``:

.. math::

   \text{count}[e] = \left\lfloor \frac{\text{perimeter}[e]}{\text{spacing}} \right\rfloor + 1

then decremented while :math:`(\text{count}[e] - 1) \cdot \text{spacing} \geq \text{perimeter}[e]`,
and finally capped at ``N_max``.

This derivation matches the ``geometry.arc_length_resample(spacing=)`` torch oracle. It
ensures that the ``count[e] - 1`` inter-bead gaps each measure no more than ``spacing``
along the arc, so the bead chain is never over-packed.

Spacing defaults and sizing
---------------------------

``spacing`` defaults to ``None``, which auto-couples to ``0.6·half_width`` (the
relax-friendly rule of thumb) — a fixed spacing default would be wrong as ``half_width``
varies. Set it explicitly to override.

**Size** ``N_max ≥ max(perimeter)/spacing + 1``: a track whose true count exceeds
``N_max`` is silently truncated (its closing segment then spans the gap) and fails validity.
The fat-band default (``half_width=0.5``, ``spacing=0.30``, ``N_max=384``) leaves ample
headroom (mean ≈ 160, max ≈ 270 points/track).
