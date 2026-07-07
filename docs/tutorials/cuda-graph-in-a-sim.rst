Integrating the CUDA Graph into a Batched RL Sim Loop
=======================================================

For generators whose ``GeneratorSpec`` has ``capturable=True``, ``track_gen``
captures the entire generation pipeline as a single replayable CUDA graph on the
first ``generate()`` call. The ``repulsive`` generator is
``capturable=False`` and runs eagerly with host-controlled execution. This page
explains the fixed-batch contract, capture and replay, reused output buffers, and
how to slot the generator into a batched reinforcement-learning step loop.

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
``capturable=True``. With ``repulsive``, every call instead runs eagerly under
host control and no CUDA graph is created.


1. **First call** — warms the Warp kernels (compiles JIT-specialized CUDA
   code), captures the full pipeline with ``wp.ScopedCapture``, stores the
   resulting ``wp.Graph``, and immediately replays it to produce the first
   batch of tracks.
2. **Subsequent calls** — copies the current ``rng.seeds_warp`` values into
   the pre-allocated seed buffer on device, then replays the stored graph with
   ``wp.capture_launch``.

Capture works for those generators because their pipeline is **entirely static**:

- All buffer shapes and loop counts are fixed at construction time.
- There is no host-side branching on device tensor data (branching on config
  fields is resolved before capture).
- Per-track ``count[e]`` values are device-side data; count-aware kernels keep
  static launch dimensions via ``N_max``.
- Host-blocking synchronization calls (``wp.synchronize``) are suppressed
  during capture — they are illegal inside a CUDA graph scope.

.. code-block:: python

   import warp as wp
   wp.init()

   from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

   E = 512
   config = TrackGenConfig(num_envs=E, half_width=0.03, device="cuda")
   rng    = PerEnvSeededRNG(seeds=0, num_envs=E, device="cuda")

   gen = TrackGenerator(config, rng)

   # First call: captures the pipeline, replays immediately.
   track = gen.generate()

   # Subsequent calls: replay only (new seeds are written to the seed buffer
   # before each replay).
   rng2   = PerEnvSeededRNG(seeds=42, num_envs=E, device="cuda")
   gen2   = TrackGenerator(config, rng2)   # fresh RNG, same config
   track2 = gen2.generate()                # fast replay

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
   config = TrackGenConfig(num_envs=E, half_width=0.03, device=device)
   rng    = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
   gen    = TrackGenerator(config, rng)

   # First reset — first call captures the CUDA graph.
   track = gen.generate()

   center = wp.to_torch(track.center).view(E, config.N_max, 2)
   outer  = wp.to_torch(track.outer).view(E, config.N_max, 2)
   inner  = wp.to_torch(track.inner).view(E, config.N_max, 2)
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
       relax_sep_every=40,
       relax_sep_cache_slots=16,
       relax_sep_cache_skin=0.0,
       device="cuda",
   )

On the benchmark machine this setting achieved roughly ``0.066 s`` per replay
versus ``0.366 s`` for the dense baseline at ``E=8192``. Validate yield in your
target regime before relying on the cache.

CPU Fallback
-------------

On the ``cpu`` device the pipeline runs eagerly on every ``generate()`` call —
there is no CUDA graph capture. The same ``TrackGenerator`` API works on both
devices; only the device string in ``TrackGenConfig`` needs to change.

.. code-block:: python

   config_cpu = TrackGenConfig(num_envs=E, half_width=0.03, device="cpu")
   rng_cpu    = PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")
   gen_cpu    = TrackGenerator(config_cpu, rng_cpu)
   track_cpu  = gen_cpu.generate()   # eager, no graph
