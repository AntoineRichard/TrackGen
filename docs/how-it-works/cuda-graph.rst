CUDA Graph — capture, replay, and buffer reuse
================================================

``TrackGenerator`` owns the production graph-captured path. Construction resolves the
selected generator, pre-allocates the persistent ``Track``, all per-stage scratch groups,
and the ``[E]`` seed buffer. ``generate()`` always returns that same ``Track`` instance
with stable ``wp.array`` pointers; callers that need a snapshot use ``Track.clone()``.

Runtime facade
--------------

``TrackGenerator`` acts as a runtime facade over the pipeline. Its public interface is:

- ``__init__(config)`` — resolves generator, allocates all persistent buffers.
- ``generate(seeds)`` — runs or replays the pipeline, returns the stable ``Track``.
- ``Track.clone()`` — snapshot the current output arrays.

On the Warp ``cpu`` device, every ``generate()`` call runs ``_run_pipeline`` eagerly. On
``cuda``, the first call compiles and warms the kernels, captures ``_run_pipeline`` with
``wp.ScopedCapture``, stores the resulting ``wp.Graph``, and then launches it. Subsequent
calls copy the current ``rng.seeds_warp`` values into the pre-allocated seed buffer and
replay the stored graph with ``wp.capture_launch``.

This capture-then-replay path applies to generators whose ``GeneratorSpec.capturable`` is
``True`` — which is now **all six** (``bezier``, ``hull``, ``polar``, ``voronoi``,
``checkpoint``, and ``repulsive``). ``repulsive`` was the one exception until 2026-07-05: its
growth loop transitions coarse-to-fine stages (a host-deterministic schedule that simply unrolls
into the graph) and drove a final-stage area-stall early exit from a host readback (a host branch
on device data, illegal inside a capture). That early exit now runs **device-side** under capture
via a CUDA-graph conditional-node while-loop (``wp.capture_while`` on a device flag, with the
periodic resample and stall freeze inside ``wp.capture_if`` branches), so the captured replay stops
at the identical iteration as the eager run and is byte-for-byte identical to it; the ``cpu`` path
keeps the plain host-driven loop. Capture is roughly wall-clock-neutral for ``repulsive`` (it is
GPU-compute/latency-bound, not host-launch-bound), so the benefit is architectural uniformity — no
eager special case and no per-window host syncs. See :doc:`/generators/repulsive`.

Capture requirements
--------------------

The capture works because every stage is pure Warp and fixed-shape:

- A module global ``_CAPTURING`` makes every wrapper's ``_sync`` and ``warp_relax``'s
  final ``wp.synchronize`` a no-op during capture. Host-blocking syncs are illegal inside
  capture, and the graph records stream ordering.
- The seed buffer address is stable; replay reuses the same buffer and reads the new seed
  contents on device.
- ``output_mode="constant_spacing"`` captures too: per-track ``count[e]`` is device-side
  data, and count-aware kernels keep static launch dimensions via ``N_max``.
- Generator selection and ``relax_enable`` are Python branches resolved before capture. The
  captured graph is fixed for that ``TrackGenerator``'s config.

Buffer reuse
------------

All persistent buffers — the output ``Track`` arrays, per-stage scratch (generator scratch,
resample buffers, relax scratch including broadphase cache arrays), and the seed buffer —
are allocated once at construction and reused across every ``generate()`` call. There is no
per-call allocation on the hot path.

``RelaxScratch`` holds the double buffer pair (positions A and B), the ``band`` and ``L0``
arrays, and — in cached broadphase mode — the fixed-slot candidate list and overflow
counter. All addresses are stable across replays.

Performance characteristics
----------------------------

At large batches the pipeline is compute-bound (relaxation dominates), so graph replay is
approximately the same wall-clock as the eager call. The graph's value is a single,
GPU-resident, deployable replayable unit — not a raw speedup — that eliminates kernel
launch overhead and driver overhead for the full pipeline in a single ``capture_launch``
call.

.. note::

   ``cpu``-device runs are always eager. The graph capture path is ``cuda``-only and applies to
   every ``capturable=True`` generator — which is now all six, ``repulsive`` included (its
   final-stage stall loop uses ``wp.capture_while`` conditional graph nodes; the ``cpu`` path stays
   host-driven).
