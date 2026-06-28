Generating Gate Sequences for Drone Courses
============================================

For drone-style courses where road width is irrelevant, ``track_gen`` provides
``GateGenerator``. It emits gate centers and orientations directly from the
first-stage generator anchors and skips constant-spacing resampling, XPBD
relaxation, and inflation — making it faster than the full track pipeline.

Prerequisites
-------------

Install ``track_gen`` with the ``dev`` extra:

.. code-block:: bash

   uv pip install -e ".[dev]"

Building the Gate Config
--------------------------

``GateGenConfig`` controls every gate-generation parameter. The key fields are
``generator`` (which first-stage method to use), ``gate_ordering``,
``num_envs``, ``max_gates``, ``gate_width``, ``gate_radius``, and
``gate_solve_iters``.

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

Generating the Gate Sequence
------------------------------

.. code-block:: python

   gates = GateGenerator(config, rng).generate()

   position = wp.to_torch(gates.position).view(E, config.max_gates, 2)
   tangent  = wp.to_torch(gates.tangent).view(E, config.max_gates, 2)
   valid    = wp.to_torch(gates.valid).bool()

``GateGenerator`` pre-allocates all buffers in ``__init__``, exactly like
``TrackGenerator``. On ``cuda``, the first ``generate()`` captures the pipeline
into a ``wp.Graph``; subsequent calls replay it with updated seeds.

Reading the Output Fields
--------------------------

.. list-table::
   :header-rows: 1

   * - Field
     - Shape
     - Meaning
   * - ``position``
     - ``[E, G, 2]``
     - Gate centers (``G = max_gates``). Slots at index ``i >= count[e]`` hold
       NaN padding.
   * - ``tangent``
     - ``[E, G, 2]``
     - Unit tangent at each gate (direction of traversal).
   * - ``normal``
     - ``[E, G, 2]``
     - Unit left-normal at each gate (perpendicular to ``tangent``).
   * - ``left``, ``right``
     - ``[E, G, 2]``
     - Gate endpoints: ``center ± 0.5 * gate_width * normal``. With the default
       ``gate_width=0.0`` these collapse onto the gate center (point gates).
   * - ``valid``
     - ``[E]`` bool
     - Per-sequence validity. A positive ``gate_width`` additionally invalidates
       any sequence whose gate bars cross.
   * - ``count``
     - ``[E]`` int
     - Real gates per environment. Slots at index ``i >= count[e]`` are NaN-padded.

Slicing Real Gates
-------------------

Like ``Track``, the ``GateSequence`` arrays are NaN-padded past each
environment's real gate count. Use ``count`` to slice valid gates:

.. code-block:: python

   count = wp.to_torch(gates.count)

   n0 = int(count[0])
   pos_env0  = position[0, :n0]   # real gate centers for env 0
   tan_env0  = tangent[0,  :n0]   # real gate tangents for env 0

Gate Figure
-----------

.. figure:: ../assets/readme-gate-strip.png

   Phase-2 gate collision solve. Top: raw anchors (``gate_solve_iters=0``);
   bottom: after the solve, with gate tangents and ``gate_width`` bars.

Gate Ordering per Generator
-----------------------------

The ordering of gates in the output sequence is controlled by
``GateGenConfig(gate_ordering=...)``. Not all orderings are supported by every
generator:

.. list-table::
   :header-rows: 1

   * - Generator
     - Supported orderings
   * - ``"bezier"``
     - ``"ccw"``, ``"random_pairs"``
   * - ``"hull"``
     - ``"ccw"``, ``"random_pairs"``
   * - ``"polar"``
     - ``"ccw"``, ``"raw"``
   * - ``"voronoi"``
     - ``"ccw"``, ``"raw"``
   * - ``"checkpoint"``
     - ``"ccw"``, ``"raw"``

.. note::

   The ``"ccw"`` ordering name is kept for API compatibility. Its current
   centroid-angle convention is **clockwise** in standard xy coordinates.

The ``gate_solve_iters`` Collision Behavior
--------------------------------------------

Gate centers are treated as disks with radius ``gate_radius``. The center
spacing target between adjacent gates is ``2 * gate_radius``. Setting
``gate_solve_iters > 0`` runs up to that many iterations of a deterministic
pairwise push that separates overlapping gate disks before recomputing tangents.

Set ``gate_solve_iters=0`` to inspect the raw anchors before the collision
solve. The anchors are still ordered and bounding-box normalized — they are just
not spread apart yet, so close gates may overlap.

When ``gate_solve_iters > 0``, the collision solve may expand the final bounding
box when necessary to satisfy the requested ``gate_radius``. A positive
``gate_width`` causes any sequence whose gate bars still cross after the solve to
be marked invalid.

.. note::

   ``max_gates`` must be at least the chosen generator's reachable gate count.
   ``min_gates`` rejects a config that the generator cannot satisfy. Both bounds
   are checked at ``GateGenerator`` construction time.

Buffer Reuse and Snapshots
---------------------------

The same ``GateSequence`` instance and its underlying Warp buffers are reused on
every ``generate()`` call. For an independent snapshot use ``gates.clone()``:

.. code-block:: python

   snapshot = gates.clone()   # independent copy; safe to keep across generate()

Full Example
------------

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
   rng   = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
   gates = GateGenerator(config, rng).generate()

   position = wp.to_torch(gates.position).view(E, config.max_gates, 2)
   tangent  = wp.to_torch(gates.tangent).view(E, config.max_gates, 2)
   left     = wp.to_torch(gates.left).view(E, config.max_gates, 2)
   right    = wp.to_torch(gates.right).view(E, config.max_gates, 2)
   valid    = wp.to_torch(gates.valid).bool()
   count    = wp.to_torch(gates.count)

   n0 = int(count[0])
   print(f"env 0: {n0} gates, valid={bool(valid[0])}")
   print("gate centers:", position[0, :n0])
