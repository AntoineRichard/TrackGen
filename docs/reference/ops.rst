.. _reference-ops:

Device ops
==========

The building blocks the built-in generators are made of, public for plugin generators;
:ref:`writing-a-generator` shows how they fit together. The ops modules import only Warp
and each other, so importing ``track_gen.ops`` loads no generator module and no torch.
A device op takes a :class:`~track_gen.exec.LaunchContext` first and launches on its
stream; it never allocates, never synchronizes and never reads device data back (the
conventions are listed in the module docstring below). ``grid_cells``, ``hash32_host``,
``check_array`` and the scratch classes' ``alloc`` run on the host.

``sample_grid_points`` and ``normalize_extent`` take a keyword-only ``size=``: an ``[E]``
``float32`` per-:term:`env` :term:`size` multiplier (the built-in generators' ``size``,
:ref:`tuning-size`), each value clamped at use to
:data:`~track_gen.ops.constants.SIZE_BOUNDS`. ``size=None`` (the default) runs the
unchanged kernels, and a size-1 array gives the same bytes; ``normalize_extent`` refuses
``size=`` with ``count=`` (``ValueError``). A plugin generator declares its own ``size``
with ``SIZE_BOUNDS`` and passes ``b.param("size")`` to the op that applies its ``scale``
(:ref:`generator-size-convention`).

.. automodule:: track_gen.ops
   :no-members:

Resampling
----------

.. autoclass:: track_gen.ops.ArcScratch
   :members: alloc

.. autofunction:: track_gen.ops.arc_resample

.. autofunction:: track_gen.ops.resample_uniform

.. autofunction:: track_gen.ops.resample_constant_spacing

Carrying per-point :term:`channels <channel>` through a resample (``tables=`` above): the
tables record where each output point lies on its source loop, and
``resample_channels`` interpolates any channel of the source through them; see
:ref:`pipeline-carried-channels`.

.. autoclass:: track_gen.ops.ResampleTables
   :members: alloc

.. autofunction:: track_gen.ops.resample_channels

Polygon geometry
----------------

.. autofunction:: track_gen.ops.self_intersections

.. autofunction:: track_gen.ops.turning_number

.. autofunction:: track_gen.ops.thickness

.. autofunction:: track_gen.ops.normalize_extent

.. autofunction:: track_gen.ops.angle_sort

Random sampling
---------------

Each env's stream is ``wp.rand_init(seeds[e] * salt + offset)`` in int32 wrap-around
arithmetic, the keying the built-in generators use. The ``(salt, offset)`` pairs the
built-ins already use are listed in :ref:`the generator guide <ops-rng-keying>`.

.. autofunction:: track_gen.ops.grid_cells

.. autofunction:: track_gen.ops.sample_counts

.. autofunction:: track_gen.ops.sample_grid_points

Curves
------

.. autofunction:: track_gen.ops.closed_catmull_rom

.. autofunction:: track_gen.ops.closed_polyline

Fills, flags and validation
---------------------------

.. autofunction:: track_gen.ops.fill

.. autofunction:: track_gen.ops.any_set

.. autofunction:: track_gen.ops.all_set

.. autofunction:: track_gen.ops.check_array

Hashing
-------

.. py:currentmodule:: track_gen.ops

.. py:function:: hash32(x: uint32) -> uint32

   ``@wp.func``: the MurmurHash3 ``fmix32`` finaliser, a ``uint32`` bijection with full
   avalanche (wrap-around arithmetic). Call it inside a kernel; it is also exported as
   ``track_gen.ops.funcs.hash32``.

.. autofunction:: track_gen.ops.hash32_host

.. py:currentmodule:: None

Device functions: ``track_gen.ops.funcs``
-----------------------------------------

.. automodule:: track_gen.ops.funcs
   :no-members:

.. py:currentmodule:: track_gen.ops.funcs

.. py:function:: ccw(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float

   Orientation of ``(o, p, q)``: ``(qy - oy)*(px - ox) - (py - oy)*(qx - ox)``, twice the
   signed triangle area; positive when ``o -> p -> q`` turns counter-clockwise.

.. py:function:: cross2(a: vec2f, b: vec2f) -> float

   2-D cross product ``a.x*b.y - a.y*b.x``.

.. py:function:: nan0(x: float) -> float

   ``x`` if finite, else ``0.0`` (a no-op on finite inputs).

.. py:function:: safe_normalize2(v: vec2f) -> vec2f

   ``v / max(|v|, 1e-8)``: a unit vector with a length floor. An all-NaN vector (a
   pruned corner) stays all-NaN. A NaN length is floored to ``1e-8`` like a short one,
   so a vector with one NaN component keeps the other component, scaled by ``1e8``.

.. py:function:: catmull_rom(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, u: float) -> vec2f

   Uniform Catmull-Rom (tension 0.5) segment from ``p1`` to ``p2`` at ``u`` in
   ``[0, 1]``: ``u = 0`` gives ``p1``, ``u = 1`` gives ``p2``. The function behind
   :func:`track_gen.ops.closed_catmull_rom`.

.. py:function:: segments_cross(a: vec2f, b: vec2f, c: vec2f, d: vec2f) -> int

   ``1`` if segments ``ab`` and ``cd`` cross properly, else ``0``. A strict sign test:
   touching endpoints and collinear overlap are not crossings. It is the gate validity
   predicate, not the one :func:`track_gen.ops.self_intersections` uses.

.. py:function:: hash32(x: uint32) -> uint32
   :no-index:

   The same function as :func:`track_gen.ops.hash32`.

.. py:currentmodule:: None

These are the operations the built-in kernels perform, in the same order. A plugin
kernel built from them matches the built-ins bit for bit on CPU. On CUDA, Warp compiles
with FMA contraction on by default (the module option ``fuse_fp``, default ``True``;
``wp.set_module_options({"fuse_fp": ...})``), and which multiply-add pairs become FMAs
depends on the code around the inlined call, so a differently structured kernel can
differ in the last bit.

Constants
---------

.. automodule:: track_gen.ops.constants
   :no-members:

.. autodata:: track_gen.ops.constants.TARGET_EXTENT

.. autodata:: track_gen.ops.constants.SIZE_BOUNDS
