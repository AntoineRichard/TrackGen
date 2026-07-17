Quickstart
==========

Track generation
----------------

The following minimal example generates a batch of 64 closed-loop race tracks on CUDA.

.. code-block:: python

   import warp as wp
   wp.init()

   from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

   E, device = 64, "cuda"  # or "cpu"
   config = TrackGenConfig(num_envs=E, half_width=0.03, device=device)
   rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)

   generator = TrackGenerator(config, rng)
   track = generator.generate()  # fixed batch: config.num_envs tracks

   center = wp.to_torch(track.center).view(E, config.N_max, 3)
   outer = wp.to_torch(track.outer).view(E, config.N_max, 3)
   inner = wp.to_torch(track.inner).view(E, config.N_max, 3)
   valid = wp.to_torch(track.valid).bool()
   count = wp.to_torch(track.count)

   center[0, : int(count[0])]  # real arc-uniform centerline points for env 0

``TrackGenerator`` is fixed-batch. Omit ``generate()``'s argument, or pass the integer
``E == config.num_envs``; explicit environment-id sequences are rejected because the CUDA
graph captures one fixed batch shape. Results are read via ``wp.to_torch``, which provides
a zero-copy view of the underlying Warp array as a PyTorch tensor. The same ``Track``
instance and Warp buffers are reused on every call, so use ``track.clone()`` when you need
an independent snapshot.

Gate sequence generation
------------------------

For drone-style courses where track width is irrelevant, use ``GateGenerator``.
It emits gate centres and orientations directly from native centerline-generator anchors
and skips constant-spacing, XPBD relaxation, and inflation.

.. code-block:: python

   import warp as wp
   wp.init()

   from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG

   E, device = 64, "cuda"
   config = GateGenConfig(
       generator="bezier",
       gate_ordering="random_pairs",
       num_envs=E,
       max_gates=32,
       gate_width=0.4,
       gate_radius=0.025,
       gate_solve_iters=8,
       device=device,
   )
   rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)

   gates = GateGenerator(config, rng).generate()
   position = wp.to_torch(gates.position).view(E, config.max_gates, 3)
   tangent = wp.to_torch(gates.tangent).view(E, config.max_gates, 3)
   valid = wp.to_torch(gates.valid).bool()

``GateGenerator`` follows the same fixed-batch contract as ``TrackGenerator``: the output
batch size is always ``config.num_envs``. Results are read via ``wp.to_torch`` for
zero-copy access as PyTorch tensors. The same ``GateSequence`` instance and its Warp
buffers are reused on every ``generate()`` call; use ``gates.clone()`` when you need an
independent snapshot.

The ``GateSequence`` fields are:

.. list-table::
   :header-rows: 1

   * - Field
     - Shape
     - Meaning
   * - ``position``
     - ``[E, G, 2]``
     - gate centers (``G = max_gates``)
   * - ``tangent``, ``normal``
     - ``[E, G, 2]``
     - unit tangent and left-normal at each gate
   * - ``left``, ``right``
     - ``[E, G, 2]``
     - gate endpoints (``center ± 0.5 * gate_width * normal``)
   * - ``valid``
     - ``[E]`` bool
     - per-sequence validity
   * - ``count``
     - ``[E]`` int
     - real gates per env; slots ``i >= count[e]`` are NaN padding
