Track-frame localization
=========================

Racing controllers and reward shapers want positions in the TRACK's frame,
not the world's: how far along the lap (``s``), how far off the racing line
(``n``), and how sharp the road ahead is. ``track_gen.localize`` answers all
three against the same batched ``Track`` buffers as the sibling utilities.

The (s, n) frame
----------------

``TrackLocalizer`` projects one point per :term:`env` onto the centerline and
returns a ``TrackFrame`` — arc length ``s`` (same units as the track,
consistent with ``Track.arclen``/``Track.length``), signed lateral offset
``n``, and the nearest centerline segment index:

.. code-block:: python

   import warp as wp

   from track_gen import TrackGenerator
   from track_gen.localize import TrackLocalizer
   from track_gen.pipeline import presets

   E, device = 64, "cuda:0"                     # or "cpu"
   track = TrackGenerator(presets.track(num_envs=E, device=device)).generate()

``robot_pos`` is your sim's stable ``[E]`` position buffer, which ``sim_step`` (your
physics) writes in place. Bind it once, then query with no arguments:

.. snippet: setup

   @wp.kernel
   def _walk_k(center: wp.array(dtype=wp.vec3f), count: wp.array(dtype=wp.int32),
               valid: wp.array(dtype=wp.int32), stride: int, t: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3f)):
       e = wp.tid()
       n = count[e]
       if valid[e] != 0 and n > 0:
           pos[e] = center[e * stride + (t[0] * 2) % n] + wp.vec3f(0.0, 0.01, 0.0)

   @wp.kernel
   def _tick_k(t: wp.array(dtype=wp.int32)):
       t[0] = t[0] + 1

   _t = wp.zeros(1, dtype=wp.int32, device=device)
   robot_pos = wp.zeros(E, dtype=wp.vec3f, device=device)

   def sim_step():
       wp.launch(_walk_k, E, [track.center, track.count, track.valid, track.layout.stride, _t, robot_pos],
                 device=device)
       wp.launch(_tick_k, 1, [_t], device=device)

.. code-block:: python

   loc = TrackLocalizer(track, position=robot_pos)   # bind your sim's buffer once
   for _ in range(10):
       sim_step()                    # your physics: writes robot_pos in place
       frame = loc.query()           # no args: bound mode
       # frame.s [E] float32, frame.n [E] float32, frame.segment [E] int32

``TrackLocalizer`` needs a centerline, not a band: it takes any ``HasCenterline``
source (a ``Track``, a ``Centerline`` such as a gates-mode
``course.course_line.centerline``, an env-slice ``view()``, or your own object with
``center``, ``arclen``, ``length`` and ``count``) and checks those members at
construction (:ref:`utilities-sources`). ``curvature`` reads ``center`` and ``count``,
``speed_profile`` also ``arclen`` and ``length``. All three read :term:`valid` when the
source has it, and the localizer :term:`generation` too (checked the same way; a malformed one raises
at construction).

``n`` is positive to the RIGHT of the centerline's direction of travel
(the direction of increasing ``s``), negative to the left. Whether the
right side is the outer or the inner boundary depends on the loop's
winding, which is generator-dependent (e.g. polar loops wind CCW — outer
on the right; bezier loops wind CW — outer on the left); either way
``|n| <= half_width`` means the point is on the road. NaN positions yield
NaN frames (``segment == -1``) — the same pause semantics as the sibling
utilities. So do :term:`invalid envs <invalid env>` (``valid[e] == 0``): NaN ``s`` /
``n`` / ``n_up``, ``segment = -1``, and the warm memory is dropped
(:ref:`utilities-invalid-envs`).

Warm starts
-----------

Consumers query every sim step with small motions, so scanning the whole
centerline each time is wasted work. ``TrackLocalizer(track,
warm_window=W)`` scans only the ``2*W + 1`` segments centered on each env's
previous result — exact (identical to the full scan) whenever the nearest
segment stays inside the window, the usual case when per-step motion is
small against ``W * spacing``. Small motion alone does not guarantee it:
where the loop pinches close to itself, the nearest segment can jump half
a lap discontinuously, and the warm result then stays on the traveled
branch (often preferable for racing consumers — ``s`` stays continuous —
but not the cold answer). Both modes are fixed-bound kernels with no host
sync, so ``query()``/``reset()`` are CUDA-graph capturable as-is.

The warm memory refers to segment INDICES, so it goes stale when the
geometry changes: reset it for the regenerated envs, and call ``reset(mask)`` for any env
that teleports beyond the window. Cold-scan localizers (``warm_window=None``) need
neither.

After a regeneration the localizer resets itself: it keeps a ``seen_generation`` per env
and, at the top of every ``query()``, drops the warm memory of every env whose source's
``generation`` changed, on device, exactly as ``reset(mask)`` would (so an explicit reset
changes nothing). A source without ``generation`` never self-resets. ``generation`` is the
:term:`epoch`: a regeneration without ``advance``, a bare reseed, or ``reset_epoch()`` followed by
as many advances leaves it equal, so call ``reset(mask)`` after those
(:ref:`utilities-self-reset`).

Curvature & speed profiles
--------------------------

Two per-generation helpers share the module. By default they ALLOCATE their result (and
``curvature`` a scratch when ``window > 0``): call them after ``generate()``, outside any
:term:`capture`. With ``out=`` they write into your arrays instead, allocate nothing,
never synchronize, and record into a graph like ``query()``:

.. code-block:: python

   from track_gen.localize import curvature, speed_profile

   kappa = curvature(track)                          # [E*N_max] signed, 1/units
   v_ref = speed_profile(track, a_lat_max=6.0,       # [E*N_max] target speeds
                         a_accel=3.0, a_brake=5.0,
                         v_cap=4.0, kappa=kappa)     # kappa= optional reuse

   # allocation-free, recordable: preallocate once, refresh after every generation
   E, N_max = track.layout.num_envs, track.layout.stride    # [E * stride] per array
   k = wp.empty(E * N_max, dtype=wp.float32, device=track.layout.device)
   tmp = wp.empty(E * N_max, dtype=wp.float32, device=track.layout.device)
   v = wp.empty(E * N_max, dtype=wp.float32, device=track.layout.device)
   curvature(track, out=k, scratch=tmp)              # returns k
   speed_profile(track, 6.0, 3.0, 5.0, 4.0, kappa=k, out=v)

- ``out`` is a contiguous ``[E * stride]`` ``float32`` array on the source's device
  (``track.layout``; the stride is :term:`N_max` for a ``Track``), returned as is. The
  same kernels run on the same inputs, so the values are byte-identical to the
  allocating form.
- ``curvature(out=)`` with ``window > 0`` needs ``scratch=`` (the same shape, not aliasing
  ``out``; ``ValueError`` otherwise); with ``window == 0`` it is ignored.
- ``speed_profile(out=)`` needs ``kappa=`` (``ValueError``: compute it with
  ``curvature(track, out=k, scratch=s)``).
- Both take ``ctx=`` (a :class:`~track_gen.exec.LaunchContext`) like the other hot
  methods.

An invalid env (``valid[e] == 0``) gets a NaN row in both, like an env with no track.

``curvature`` is the discrete turn angle over the mean incident edge length
with a wrap-aware moving average (``window=2`` by default); positive where
the loop turns counter-clockwise — a CCW circle of radius ``R`` gives
``+1/R``. ``speed_profile`` is the classic three-stage racing profile:
steady-state ``min(sqrt(a_lat_max / |kappa|), v_cap)``, a forward pass
capping acceleration out of corners, and a backward pass capping braking
into them, each run twice around the closed loop so the wrap converges.
Pair ``frame.segment`` with these arrays to read the local target speed or
curvature at the vehicle every step.
