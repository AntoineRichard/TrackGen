Resample — constant-spacing arc-length resampling
===================================================

The pipeline resamples a centerline by arc length at three points: the generators'
dense-to-``num_points`` resample, the constant-spacing resample that turns a raw loop into
:term:`N_max` beads (the ``resample`` :term:`stage`), and the re-uniformising resample
after relaxation (the ``uniform`` stage). All three are public ops in ``track_gen.ops``,
launched on a :class:`~track_gen.exec.LaunchContext`, allocation-free and without host
syncs; each is two kernels, a per-env scan (``E`` threads) and a per-point lookup.

Resampler overview
------------------

``ops.arc_resample``
~~~~~~~~~~~~~~~~~~~~

``ops.arc_resample(ctx, src, stride, num, scratch, out, count_out, *, active=None)``

The **NaN-aware** resampler (kernels ``_arc_scan_k`` + ``_arc_lookup_k``). Per env it
compacts the points whose x and y are finite, in order, builds the closed-loop
cumulative arc length, and looks up ``num`` arc-uniform targets (searchsorted + lerp).
Envs with fewer than 2 real points get an all-NaN row and ``count_out[e] = 0``. With
``active``, only the envs with ``active[e] > 0`` are resampled. The generators use it for
their dense-to-``num_points`` resample and for the selected polygon-fallback path.

``ops.resample_uniform``
~~~~~~~~~~~~~~~~~~~~~~~~

``ops.resample_uniform(ctx, src, stride, scratch, out, *, count, tables=None)``

The count-aware N→N re-uniformiser (``_resample_scan_k`` + ``_resample_lookup_k``), run
after relaxation: each env's first ``count[e]`` points (all finite) are resampled to
``count[e]`` arc-uniform points, NaN-padded past ``count[e]``.

``ops.resample_constant_spacing``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ops.resample_constant_spacing(ctx, src, stride, spacing, out_stride, scratch, out, count_out, truncated_out, *, tables=None)``

The constant-spacing resampler. From a fully real loop it picks a per-track
``count = floor(perimeter / spacing) + 1`` (`count[e] derivation`_), places point ``k``
at arc length ``k * spacing`` from point 0 into an ``[E * out_stride]`` buffer NaN-padded
past ``count[e]``, and sets ``truncated_out[e] = 1`` when the cap applied. A loop whose
perimeter is not positive gets ``count = 0``. It matches the torch oracle
``arc_length_resample(points, spacing=...)`` of the test suite (``tests/_oracle``).

``resample_uniform`` and ``resample_constant_spacing`` can also write *resample tables*
(``ops.ResampleTables``: each output point's source segment ``i0``, ``i1`` and fraction
``frac``), through which ``ops.resample_channels`` carries per-point channels with the
points' own interpolation (:ref:`pipeline-carried-channels`). ``ops.ArcScratch`` holds
the scan buffers, allocated once.

Why constant spacing
--------------------

Each track is relaxed and emitted at a constant arc spacing, with a per-track point
count, rather than at a fixed point count. A fixed 256 points **over-resolves** the
centerline relative to its half-width (a segment of about 0.2 m against a 0.5 m
half-width), so the Jacobi XPBD solve **under-converges** within its fixed iteration
count, and the road of the jagged result overlaps itself. Relaxing at a width-appropriate
spacing lets the same solve converge, giving smooth, valid tracks on fewer nodes per
track (so it is also faster). ``num_points`` is only the dense resolution of the
generators' output, before the constant-spacing step. The config path always uses
constant-spacing output.

``count[e]`` derivation
-----------------------

For each :term:`env` ``e``:

.. math::

   \text{count}[e] = \left\lfloor \frac{\text{perimeter}[e]}{\text{spacing}} \right\rfloor + 1

then decremented while :math:`(\text{count}[e] - 1) \cdot \text{spacing} \geq \text{perimeter}[e]`,
and finally capped at ``N_max``.

This derivation matches the torch oracle. It ensures that the ``count[e] - 1``
inter-bead gaps each measure no more than ``spacing`` along the arc, so the bead chain is
never over-packed.

Spacing defaults and sizing
---------------------------

:term:`spacing` defaults to ``None``, which couples it to ``0.6 * half_width`` (the
relax-friendly rule of thumb; ``0.06`` at the default ``half_width = 0.1``): a fixed
spacing default would be wrong as ``half_width`` varies. Set it explicitly to override.
A frozen config resolves this at construction; use ``dataclasses.replace(cfg,
half_width=..., spacing=None)`` to derive it again for a replacement.

**Size** ``N_max >= max(perimeter) / spacing + 1``: a track whose true count exceeds
``N_max`` is truncated (its closing segment would span the gap); the scan flags it in
``Track.truncated`` and it fails validity (:ref:`pipeline-fail-bits`,
:ref:`tuning-size`). Measure the largest perimeter at the upper end of your intended size/shape range,
allow headroom for relaxation and check ``truncated`` after generation. Capacity is a
storage budget, not a shape simplification target.
