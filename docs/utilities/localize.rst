Track-frame localization
=========================

Racing controllers and reward shapers want positions in the TRACK's frame,
not the world's: how far along the lap (``s``), how far off the racing line
(``n``), and how sharp the road ahead is. ``track_gen.localize`` answers all
three against the same batched ``Track`` buffers as the sibling utilities.

The (s, n) frame
----------------

``TrackLocalizer`` projects one point per env onto the centerline and
returns a ``TrackFrame`` — arc length ``s`` (same units as the track,
consistent with ``Track.arclen``/``Track.length``), signed lateral offset
``n``, and the nearest centerline segment index:

.. code-block:: python

   from track_gen.localize import TrackLocalizer

   loc = TrackLocalizer(track, position=robot_pos)   # latch onto sim buffer
   for _ in range(steps):
       sim.step()                    # writes robot_pos in place
       frame = loc.query()           # no args: bound mode
       # frame.s [E] float32, frame.n [E] float32, frame.segment [E] int32

``n`` is positive to the RIGHT of the centerline's direction of travel
(the direction of increasing ``s``), negative to the left. Whether the
right side is the outer or the inner boundary depends on the loop's
winding, which is generator-dependent (e.g. polar loops wind CCW — outer
on the right; bezier loops wind CW — outer on the left); either way
``|n| <= half_width`` means the point is on the road. NaN positions yield
NaN frames (``segment == -1``) — the same pause semantics as the sibling
utilities.

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
but not the cold answer). Both modes are fixed-bound kernels: no host
sync, CUDA-graph capturable via ``track_gen.set_capturing``.

The warm memory refers to segment INDICES, so it goes stale when the
geometry changes: call ``reset(mask)`` for all envs after regenerating the
bound track, and for any env that teleports beyond the window. Cold-scan
localizers (``warm_window=None``) need neither.

Curvature & speed profiles
--------------------------

Two per-generation helpers share the module (they ALLOCATE their results —
call them after ``generate()``, outside capture regions, unlike the
allocation-free ``query()``):

.. code-block:: python

   from track_gen.localize import curvature, speed_profile

   kappa = curvature(track)                          # [E*N_max] signed, 1/units
   v_ref = speed_profile(track, a_lat_max=6.0,       # [E*N_max] target speeds
                         a_accel=3.0, a_brake=5.0,
                         v_cap=4.0, kappa=kappa)     # kappa= optional reuse

``curvature`` is the discrete turn angle over the mean incident edge length
with a wrap-aware moving average (``window=2`` by default); positive where
the loop turns counter-clockwise — a CCW circle of radius ``R`` gives
``+1/R``. ``speed_profile`` is the classic three-stage racing profile:
steady-state ``min(sqrt(a_lat_max / |kappa|), v_cap)``, a forward pass
capping acceleration out of corners, and a backward pass capping braking
into them, each run twice around the closed loop so the wrap converges.
Pair ``frame.segment`` with these arrays to read the local target speed or
curvature at the vehicle every step.
