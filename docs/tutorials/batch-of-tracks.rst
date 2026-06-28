:orphan:

Generating a Batch of Tracks
=============================

This tutorial walks through the end-to-end workflow: build a config and RNG,
call ``generate()``, read the output arrays, and work with the real points for
each environment.

Prerequisites
-------------

Install ``track_gen`` with the ``dev`` extra (pulls in ``warp-lang``, ``torch``,
and the rest of the development dependencies):

.. code-block:: bash

   uv pip install -e ".[dev]"

Building the Config and RNG
-----------------------------

``TrackGenConfig`` holds every generation parameter. The two required fields are
``num_envs`` (batch size) and ``device`` (``"cpu"`` or ``"cuda"``).
``half_width`` is the road half-width in world units and defaults to ``0.5``.

.. code-block:: python

   import warp as wp
   wp.init()

   from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

   E, device = 64, "cuda"  # or "cpu"
   config = TrackGenConfig(num_envs=E, half_width=0.03, device=device)
   rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)

``PerEnvSeededRNG`` derives one independent seed per environment from the master
seed. Passing an integer broadcasts the same master seed across all environments;
passing a 1-D array of length ``E`` sets per-environment seeds directly.

Running the Generator
----------------------

.. code-block:: python

   generator = TrackGenerator(config, rng)
   track = generator.generate()

``TrackGenerator.__init__`` pre-allocates all GPU buffers (generator scratch,
pipeline scratch, seed buffers, and the persistent ``Track``) exactly once. On
``cuda``, the first ``generate()`` call warms the kernels, captures the entire
pipeline into a single replayable ``wp.Graph``, and immediately replays it.
Subsequent calls copy the updated seeds into the fixed seed buffer and replay the
same graph.

.. note::

   ``TrackGenerator`` is **fixed-batch**: ``generate()`` always produces exactly
   ``config.num_envs`` tracks. Passing an explicit environment-id sequence is not
   supported because the CUDA graph captures one fixed batch shape.

Reading the Output Arrays
--------------------------

``track`` is a ``Track`` dataclass whose fields are flat Warp arrays. Use
``wp.to_torch`` to get PyTorch views:

.. code-block:: python

   center = wp.to_torch(track.center).view(E, config.N_max, 2)
   outer  = wp.to_torch(track.outer).view(E, config.N_max, 2)
   inner  = wp.to_torch(track.inner).view(E, config.N_max, 2)
   valid  = wp.to_torch(track.valid).bool()
   count  = wp.to_torch(track.count)

The full field table:

.. list-table::
   :header-rows: 1

   * - Field
     - Shape
     - Meaning
   * - ``outer``, ``center``, ``inner``
     - ``[E, N_max, 2]``
     - Border / centerline / border points. All three are index-aligned: the same
       cross-section normal passes through ``outer[e, i]``, ``center[e, i]``, and
       ``inner[e, i]``.
   * - ``tangent``, ``normal``
     - ``[E, N_max, 2]``
     - Unit tangent and left-normal along the centerline.
   * - ``arclen``
     - ``[E, N_max]``
     - Cumulative arc length (0 at index 0).
   * - ``length``
     - ``[E]``
     - Closed-loop perimeter for each environment.
   * - ``valid``
     - ``[E]`` bool
     - Per-track validity flag. Invalid tracks failed a geometric check
       (turning, thickness, NaNs, or road self-overlap).
   * - ``count``
     - ``[E]`` int
     - Real point count per track. Slots at index ``i >= count[e]`` are
       NaN-padded.

NaN Padding and Slicing Real Points
-------------------------------------

``track_gen`` uses **constant-spacing** output: each track is emitted at a
constant arc spacing rather than a fixed point count. The per-track point count
is ``count[e] = floor(perimeter / spacing) + 1``, capped at ``N_max``. Slots
beyond ``count[e]`` are filled with ``NaN``.

To work only with the real arc-uniform points for a given environment:

.. code-block:: python

   # Real centerline points for environment 0
   n0 = int(count[0])
   center_env0 = center[0, :n0]   # shape [n0, 2]
   outer_env0  = outer[0,  :n0]   # shape [n0, 2]
   inner_env0  = inner[0,  :n0]   # shape [n0, 2]

The default ``spacing`` is ``None``, which auto-sets to ``0.6 * half_width``.
Size ``N_max >= max(perimeter) / spacing + 1`` so no track is silently
truncated; the default fat-band configuration leaves ample headroom.

Buffer Reuse and Snapshots
---------------------------

The same ``Track`` instance and its underlying Warp arrays are **reused** on
every ``generate()`` call. If you need an independent snapshot — for example, to
keep the tracks from one episode while generating a fresh batch — call
``track.clone()``:

.. code-block:: python

   snapshot = track.clone()   # independent copy; safe to keep across generate()

Converting with ``wp.to_torch``
---------------------------------

``wp.to_torch`` returns a **zero-copy** PyTorch view of the underlying Warp
buffer. The view is only valid as long as the backing ``Track`` (or clone) is
alive. For a persistent tensor, call ``.clone()`` on the PyTorch side as well:

.. code-block:: python

   center_torch = wp.to_torch(track.center).view(E, config.N_max, 2).clone()

Putting It All Together
------------------------

.. code-block:: python

   import warp as wp
   wp.init()

   from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

   E, device = 64, "cuda"
   config = TrackGenConfig(num_envs=E, half_width=0.03, device=device)
   rng    = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)

   generator = TrackGenerator(config, rng)
   track     = generator.generate()

   center = wp.to_torch(track.center).view(E, config.N_max, 2)
   outer  = wp.to_torch(track.outer).view(E, config.N_max, 2)
   inner  = wp.to_torch(track.inner).view(E, config.N_max, 2)
   valid  = wp.to_torch(track.valid).bool()
   count  = wp.to_torch(track.count)

   # Slice real points for env 0
   n0 = int(count[0])
   print(f"env 0: {n0} real points, valid={bool(valid[0])}")
   print("centerline:", center[0, :n0])
