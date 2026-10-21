Integrating the CUDA Graph into a Batched RL Sim Loop
=======================================================

For generators whose ``GeneratorSpec`` has ``capturable=True``, ``track_gen``
captures the entire generation pipeline as a single replayable CUDA graph on the
first ``generate()`` call. All six shipped generators are capturable, including
``repulsive``; a plugin generator registered with ``capturable=False`` instead
runs eagerly on CUDA every call. This page explains the fixed-batch contract,
capture and replay, reused output buffers, and how to slot the generator into a
batched reinforcement-learning step loop.

The Fixed-Batch Contract
--------------------------

``TrackGenerator`` is **fixed-batch**: the batch size ``E = config.num_envs`` is
set at construction time and cannot change between calls. For a capturable generator,
the CUDA graph records one fixed execution with fixed-shape tensors. Passing a different
batch size at replay time is not possible — construct a new ``TrackGenerator`` instead.

All pre-allocated buffers (generator scratch, pipeline scratch, seed buffer,
the persistent ``Track``) are created once in ``__init__``. Their shapes and
device addresses are stable for the lifetime of the ``TrackGenerator`` object.

First-Call Capture vs Replay
------------------------------

On the ``cpu`` device, every ``generate()`` call runs the pipeline eagerly —
there is no graph.

On ``cuda``, the following applies when the selected generator has
``capturable=True`` (every shipped generator):

1. **First call** — warms the Warp kernels with three eager runs (compiles
   JIT-specialized CUDA code), captures the full pipeline with
   ``wp.ScopedCapture``, stores the resulting ``wp.Graph``, and immediately
   replays it to produce the first batch of tracks.
2. **Subsequent calls** — copies the current ``rng.seeds_warp`` values into
   the pre-allocated seed buffer on device, then replays the stored graph with
   ``wp.capture_launch``.

The graph belongs to the ``TrackGenerator`` instance: a new instance warms up
and captures again on its own first call.

Capture works because a capturable generator's pipeline is **entirely static**:

- All buffer shapes and loop counts are fixed at construction time.
- There is no host-side branching on device tensor data (branching on config
  fields is resolved before capture).
- Per-track ``count[e]`` values are device-side data; count-aware kernels keep
  static launch dimensions via ``N_max``.
- The pipeline never host-synchronizes or reads device data back (both are
  illegal inside a CUDA graph scope); ordering comes from the stream.

``repulsive`` is the outlier on cost, not on capture: its growth loop runs a
fixed, host-precomputed iteration schedule with no global early exit, so every
call pays the full iteration budget (~1000× slower than the other generators),
and its first call also pays the three eager warm-ups plus capturing a ~4k-node
graph. Converged envs are frozen on device, so each env still depends only on
its own seed. See :doc:`/generators/repulsive`.

.. code-block:: python

   import numpy as np
   import warp as wp
   wp.init()

   from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

   E = 512
   # Thin tracks need N_max >= 512: the default 384 covers only ~98% of them, and
   # truncated tracks fail validity (see Track.truncated and the tuning guide).
   config = TrackGenConfig(num_envs=E, half_width=0.03, N_max=512, device="cuda")
   rng    = PerEnvSeededRNG(seeds=0, num_envs=E, device="cuda")

   gen = TrackGenerator(config, rng)

   # First call: warms up, captures the pipeline, replays immediately.
   track = gen.generate()

   # Subsequent calls on the SAME generator: replay only (the RNG's current seeds
   # are written to the seed buffer before each replay).
   rng.set_seeds_warp(wp.array(42 + np.arange(E), dtype=wp.int32, device="cuda"), None)
   track = gen.generate()                  # fast replay

Buffer Reuse and ``clone()``
-----------------------------

``generate()`` always returns the **same** ``Track`` instance with the **same**
underlying Warp array pointers. The arrays are overwritten in-place on every
call. If you read the arrays after a second ``generate()`` call without cloning,
you will see the second batch's data, not the first.

For a sim loop that needs to keep the previous episode's tracks while generating
a new batch, use ``track.clone()`` before the next call:

.. code-block:: python

   track = gen.generate()
   saved = track.clone()   # independent copy; safe to store across generate()

   track = gen.generate()  # overwrites gen's internal buffers
   # saved still holds the first batch's data

The same pattern applies to ``GateGenerator`` and ``GateSequence.clone()``.

Integrating into a Batched RL Step Loop
-----------------------------------------

A typical batched RL loop resets environments in groups. With ``track_gen`` the
full batch is always regenerated together; the fixed-batch contract maps cleanly
onto episode resets.

.. code-block:: python

   import warp as wp
   import numpy as np
   wp.init()
   import torch

   from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

   E, device = 256, "cuda"
   config = TrackGenConfig(num_envs=E, half_width=0.03, N_max=512, device=device)
   rng    = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
   gen    = TrackGenerator(config, rng)

   # First reset — first call captures the CUDA graph.
   track = gen.generate()

   center = wp.to_torch(track.center).view(E, config.N_max, 3)
   outer  = wp.to_torch(track.outer).view(E, config.N_max, 3)
   inner  = wp.to_torch(track.inner).view(E, config.N_max, 3)
   valid  = wp.to_torch(track.valid).bool()
   count  = wp.to_torch(track.count)

   # --- Sim step loop ---
   for episode in range(1000):

       # Work with the current tracks.  All views above stay valid until the
       # next generate() call.  Clone if you need them to survive past that.
       obs = center[:, :int(count.max()), :]   # example: max-length slice

       # ... run RL steps ...

       # On episode boundary: generate a fresh batch (fast CUDA graph replay).
       new_seeds = wp.array(episode + 1 + np.arange(E), dtype=wp.int32, device=device)
       rng.set_seeds_warp(new_seeds, None)
       track = gen.generate()

.. note::

   Keep one ``TrackGenerator`` and its existing ``PerEnvSeededRNG`` alive across
   episodes. Build a per-environment ``wp.int32`` seed array, call
   ``rng.set_seeds_warp(new_seeds, None)``, then call ``gen.generate()``. The
   generator copies the RNG's current seeds into its pre-allocated seed buffer and
   replays the captured graph. Because the persistent ``Track`` buffers retain stable
   addresses, the Torch views above remain bound and do not need to be recreated.

Recording the Per-Step Utilities
----------------------------------

The runtime utilities (collision, progress, localizer, props, checkpoints,
height field) and ``Course.step()``/``reset()`` never host-synchronize, so
their per-step calls record into your own sim graph directly. No flag is
needed: ``track_gen.set_capturing`` is a deprecated no-op.

.. code-block:: python

   tracker = ProgressTracker(checkpoints, position=sim_pos)  # bind once
   tracker.update()                                         # warm up outside capture
   with wp.ScopedCapture(device="cuda") as cap:
       physics_step()
       tracker.update()
   # every sim step:
   wp.capture_launch(cap.graph)

The same calls outside a capture return without blocking the host. Results are
ordered by the Warp stream they were launched on: ``.numpy()``, later Warp
launches and torch on its default stream all see finished data. A consumer on a
different stream (for example a torch side stream) must order itself explicitly,
e.g. with ``wp.synchronize_device()`` or a recorded ``wp.Event``. Benchmarks must
synchronize before reading a timer. For debugging asynchronous CUDA errors, set
``TRACKGEN_DEBUG_SYNC=1`` to make every library call block on its stream. It is
skipped while that stream (or a Warp capture on the device) is capturing; do not
combine it with a foreign capture on another stream (e.g. a default
``torch.cuda.graph``), which it cannot detect and whose capture the sync would
invalidate.

Performance Notes
------------------

At large batch sizes the pipeline is compute-bound (XPBD relaxation dominates).
The CUDA graph's benefit is a single, GPU-resident, deployable unit — not a
speedup over eager execution. Representative replay times at ``E=8192``:

.. list-table::
   :header-rows: 1

   * - Generator
     - Replay time (CUDA)
   * - ``"bezier"``
     - 93.2 ms
   * - ``"checkpoint"``
     - 92.2 ms
   * - ``"hull"``
     - 98.9 ms
   * - ``"polar"``
     - 80.8 ms
   * - ``"voronoi"``
     - 81.3 ms

For high-throughput scenarios where relaxation is the bottleneck, the advanced
XPBD separation cache (``relax_sep_every``, ``relax_sep_cache_slots``,
``relax_sep_cache_skin``) can reduce per-replay time significantly. The
recommended starting point from the benchmark is:

.. code-block:: python

   config = TrackGenConfig(
       num_envs=E,
       half_width=0.03,
       N_max=512,
       relax_iters=150,      # the long budget is what makes relax_sep_every=40 safe
       relax_sep_every=40,
       relax_sep_cache_slots=16,
       relax_sep_cache_skin=0.0,
       device="cuda",
   )

On the benchmark machine this setting achieved roughly ``0.066 s`` per replay
versus ``0.366 s`` for the dense baseline at ``E=8192`` (``relax_iters=150``). Keep
``relax_sep_every`` well below ``relax_iters``; see :doc:`/configuration/tuning`.
Validate yield in your target regime before relying on the cache.

CPU Fallback
-------------

On the ``cpu`` device the pipeline runs eagerly on every ``generate()`` call —
there is no CUDA graph capture. The same ``TrackGenerator`` API works on both
devices; only the device string in ``TrackGenConfig`` needs to change.

.. code-block:: python

   config_cpu = TrackGenConfig(num_envs=E, half_width=0.03, N_max=512, device="cpu")
   rng_cpu    = PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")
   gen_cpu    = TrackGenerator(config_cpu, rng_cpu)
   track_cpu  = gen_cpu.generate()   # eager, no graph
