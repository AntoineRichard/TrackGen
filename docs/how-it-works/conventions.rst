Conventions — arrays, kernels, and testing
===========================================

This page documents the low-level conventions shared across all pipeline kernels.

Flat ``[E*N]`` arrays
----------------------

Points are flat ``wp.array`` of length ``E*N`` (or ``E*M``, ``E*P``). Internal
generation and relaxation kernels stage them as ``wp.vec2f``; the public boundary
arrays (``Track.center/outer/inner``, ``GateSequence.position`` …) surface as
``wp.vec3f`` after the 3D lift (``z=0`` for the planar track pipeline). Per-env scalars
are ``[E]`` arrays. There are no nested lists or ragged tensors; every buffer is a
dense, fixed-shape Warp array.

Thread-per-element and ``e = tid // N``
----------------------------------------

**One thread per output element.** A kernel launched with ``dim=E*N`` decodes its
environment as ``e = tid // N`` and its within-env index as ``i = tid % N``. Reductions
(per-env min/sum/count) use ``dim=E``, one thread looping that env's ``N`` points.

Count-aware buffers
--------------------

Post-generation, ``n_max`` is the buffer stride and ``count[e]`` is env ``e``'s
real-point count; padding slots ``i ≥ count[e]`` hold ``wp.nan``. Count-aware kernels:

- Loop ``range(count[e])``.
- Base their reads at ``e*n_max``.
- Wrap neighbours with ``% count[e]``.
- Guard ``i ≥ count[e]``.

**Parity invariant:** when ``count[e] == N_max`` for all envs, every count-aware kernel
is bit-identical to the fixed-``N`` kernel. This is what protects the fixed-N parity path
that the per-kernel oracle tests exercise.

Test and diagnostic wrappers
-----------------------------

Test and diagnostic wrappers follow the pattern:

1. ``_init()`` — idempotent ``wp.init``.
2. Optional ``wp.from_torch(t.reshape(...).contiguous(), dtype=...)`` — bridge from torch
   inputs at the test boundary.
3. ``wp.launch(kernel, dim=..., device=str(device))`` — launch the Warp kernel.
4. ``_sync(device)`` — synchronize.
5. Optional torch views via ``wp.to_torch`` — bridge outputs at the test boundary.

In-kernel idioms
-----------------

The following idioms appear throughout to avoid breaking graph capture:

- **Boolean reductions** use ``int`` 0/1 flags (Warp cannot fold Python ``bool`` values in
  dynamic loops).
- **NaN** is written as ``wp.nan``.
- **Conditional selects** use ``wp.where``.
- **Floating accumulations** that must track a torch ``cumsum`` are done in ``wp.float64``
  then cast to ``float32``.

Shared ``@wp.func`` helpers
----------------------------

Shared ``@wp.func`` helpers keep the heavy geometry DRY across kernels:

.. list-table::
   :header-rows: 1

   * - Helper
     - Description
   * - ``_safe_normalize2``
     - ``v / max(‖v‖, 1e-8)`` — normalize without division by zero.
   * - ``_nan0``
     - NaN/inf → 0.
   * - ``_pruned_corner``
     - Returns NaN for ``i ≥ count``.
   * - ``_thickness_func``
     - Per-bead thickness computation.
   * - ``_self_intersections_func``
     - Per-bead self-intersection count.
   * - ``_turning_func``
     - Per-bead signed turning contribution.

The standalone kernels (``_thickness_k``, ``_self_intersections_by_i_k``, ``_turning_k``)
and the fused ``_validity_k`` / ``gates`` all call the same helpers.

Torch as the test oracle
-------------------------

The original torch implementation is **retained, but only as the verification oracle** and
lives under ``tests/_oracle/`` (importable by tests as ``tests._oracle.*``); it is **not**
shipped as part of the ``track_gen`` package. The modules ``tests._oracle.geometry``,
``tests._oracle.inflation``, ``tests._oracle.generators``, and
``tests._oracle.relaxation`` are warp-free and are **not** imported by the runtime
pipeline.

Every Warp kernel has a test (``tests/test_warp_*.py``) asserting it matches its torch
counterpart on both ``cpu`` and ``cuda``:

- ``torch.equal`` for integer/boolean results.
- ``allclose`` at ~1e-4 for float results — Warp's float32 ``sqrt``/``length`` differs
  from torch by ~ULP, which is geometrically negligible and an accepted tolerance.
- The corner-sampling RNG is validated by structural properties, not bit-equality, since
  it is a deliberate redesign.

.. note::

   The Fourier generator lives in ``track_gen._experimental.fourier`` and is
   **unsupported** — it is self-contained, not on the Warp pipeline, and receives no
   compatibility guarantees.
