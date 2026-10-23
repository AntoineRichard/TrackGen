.. _results:

Results: Track, GateSequence, Centerline
========================================

A generator's output is a *result object*: a ``Track`` (``TrackGenerator``), a
``GateSequence`` (``GateGenerator``) or a ``Centerline`` (a centerline with no road band,
such as the spline a gates-mode ``Course`` threads through its gates). This page describes
their data model, the ``Layout`` that gives their shape, the methods that check, slice,
copy and allocate them, caller-owned results (``out=``) and mixed batches, and the
capability protocols the runtime utilities accept.

``track_gen.results`` is the module of all of it: ``Layout``, ``Track``,
``GateSequence``, ``Centerline``, the protocols ``HasCenterline``, ``HasBand``,
``HasGates``, ``HasGateFrames`` and ``HasGeneration``, and ``check_source``.
``track_gen`` re-exports ``Track``, ``GateSequence``, ``Layout`` and the five protocols
(the same objects), and ``track_gen.plugin.Layout`` (the shape a stage is built for) is
the same ``Layout``.

.. code-block:: python

   from track_gen import TrackGenerator
   from track_gen.pipeline import presets
   from track_gen.results import Layout, Track

   gen = TrackGenerator(presets.track(num_envs=8, device="cpu"))
   track = gen.generate()           # the generator's result (the same object every call)
   track.layout                     # Layout(num_envs=8, stride=384, device='cpu', closed=True)
   track.validate()                 # the same Layout: every field checked
   half = track.view(0, 4)          # envs 0..3, zero-copy
   snap = Track.alloc(track.layout) # preallocated, cold
   snap.copy_from(track)            # one copy per field, no allocation: the hot-path snapshot

The data model
--------------

Every array of a result is a flat, contiguous, 1-D Warp array on one device, in one of
two forms:

- **per point**, ``[E * stride]``: env ``e`` owns the rows ``e * stride`` to
  ``e * stride + stride - 1``, and its first ``count[e]`` rows are real. The rows past
  ``count[e]`` hold the padding: NaN in the ``float32`` fields and in every component of
  a float attribute, ``-1`` in an ``int32`` attribute, and NaN in the x and y of the
  ``vec3f`` point fields. A generated ``Track``'s z component holds what the elevation
  stage wrote there (with ``elevation="flat"``: ``z_base`` in ``outer`` / ``center`` /
  ``inner``, 0 in ``tangent`` / ``normal``), so test ``i >= count[e]`` or ``isnan(x)``,
  not every component. ``GateSequence`` fields, a ``Centerline`` and an ``alloc()``
  result are NaN in every component. ``stride`` is the per-env row capacity: ``N_max``
  for a ``Track``, ``max_gates`` for a ``GateSequence``, and for a ``Centerline`` its own
  capacity (``max_gates * samples_per_gate`` for a gates-mode ``Course``'s
  ``CourseLine.centerline``).
- **per env**, ``[E]``: ``count``, ``valid``, ``length``, ``winding``, ``generation``,
  ``fail_mask``, ``truncated``.

A generator allocates its result once, at construction, and returns the same instance
with the same array addresses on every call: a CUDA graph recorded against it stays
valid for the generator's lifetime. ``fail_bits`` (the names of ``fail_mask``'s bits),
``attributes`` (the per-point plugin attributes the pipeline exports, see
:ref:`pipeline-attributes`) and ``layout`` are metadata, not device arrays.

.. _results-layout:

``Layout``
----------

.. code-block:: python

   Layout(num_envs, stride, device, closed=True)      # frozen

- ``num_envs`` (``E >= 1``) and ``stride`` (``>= 1``) are integers; a NumPy integer is
  accepted and stored as an ``int``, a ``bool`` raises ``TypeError``, a value below 1
  ``ValueError``.
- ``device`` is a string, stored **as given**: ``Layout(8, 384, "cuda")`` is legal, and
  constructing it touches no device (so a CPU-only host can build CUDA layouts).
- ``closed`` is a ``bool`` (``TypeError`` otherwise): whether the loops are closed. Only
  ``True`` is accepted by the pipeline and the utilities in 0.x (see `Topology`_).

``Layout`` equality is the dataclass equality of the stored values. Every check the
library makes (``validate`` against a stored layout, ``out=``, ``copy_from``,
``check_source``, the conformance kit) compares layouts by their *canonical* form
instead, with the device resolved through ``wp.get_device`` at check time, so a hand-set
``Layout(E, S, "cuda")`` matches a pipeline's ``"cuda:0"``. Every layout the library
makes (the planner's, ``alloc``'s, ``view``'s, an inferred one) stores the canonical
device string, so ``==`` holds between them.

``result.layout`` is set on every result a generator, ``view``, ``alloc`` or ``clone``
returns. It is ``None`` on a result you build by hand: ``validate()`` then infers it
(and never stores it), and a build with ``out=`` sets it on the object you passed when
it was ``None``. ``program.layout`` is the pipeline's layout: the ``layout`` of the
result it allocates, and the canonical equal of an ``out=`` result's.

Schemas
-------

Each class has a fixed schema: the arrays ``validate`` checks, with their dtype and form.
An *optional* field may be ``None`` (a hand-built result); the others are required.

.. list-table::
   :header-rows: 1
   :widths: 16 40 30 14

   * - Class
     - Per point, ``[E * stride]``
     - Per env, ``[E]``
     - Optional
   * - ``Track``
     - ``outer``, ``center``, ``inner``, ``tangent``, ``normal`` (``vec3f``);
       ``arclen``, ``half_width`` (``float32``)
     - ``length``, ``winding`` (``float32``); ``valid``, ``count``, ``truncated``,
       ``generation``, ``fail_mask`` (``int32``)
     - ``truncated``, ``generation``, ``fail_mask``, ``half_width``
   * - ``GateSequence``
     - ``position``, ``tangent``, ``forward``, ``left``, ``right`` (``vec3f``);
       ``orientation`` (``quatf``); ``half_size`` (``float32``)
     - ``valid``, ``count``, ``generation``, ``fail_mask`` (``int32``)
     - ``generation``, ``fail_mask``
   * - ``Centerline``
     - ``center``, ``tangent`` (``vec3f``); ``arclen`` (``float32``)
     - ``length`` (``float32``); ``count``, ``valid``, ``generation`` (``int32``)
     - ``generation``

An attribute (an entry of ``attributes``) is a per-point ``[E * stride]`` array of
``float32``, ``int32``, ``vec2f`` or ``vec3f``.

.. _results-validate:

``validate(need=())``
---------------------

``result.validate()`` checks every array of the result and returns its ``Layout``:

1. ``count`` must be a ``wp.array``: it gives ``E``.
2. The layout is ``result.layout`` when it is set (it must be a ``Layout``), else
   inferred: ``E = count.shape[0]``, ``stride`` from the first per-point field
   (``center``; ``position`` for gates) divided by ``E`` (a length that ``E`` does not
   divide raises), the device of ``count``, ``closed=True``.
3. Every schema field that is not ``None`` must be a 1-D contiguous array of its dtype,
   ``[E * stride]`` or ``[E]``, on the layout's device (a 2-D ``[E, stride]`` array is
   named as such: ``expected a flat 1-D [E*stride] array, got a 2-D array of shape
   (4, 384) (flatten it, e.g. arr.reshape((-1,)))``). A field named in ``need`` (a
   name or a tuple of names) must not be ``None``.
4. Every attribute must be a 1-D contiguous ``[E * stride]`` array of an attribute
   dtype on that device.

It raises ``ValueError`` with the field, what was expected and what it got:

.. code-block:: text

   Track.inner: expected a contiguous [3072] vec3f ([E*stride] = [8*384]) array on cpu,
     got shape (3064,) vec3f on cpu
   Track.winding: expected a contiguous [8] float32 ([E] = [8]) array on cpu,
     got shape (7,) float32 on cpu
   Track.half_width is None; this consumer needs it

``validate`` runs on the host only (a few microseconds), launches nothing, never syncs
and never mutates the result, so it is safe anywhere, a capture included. It is
**explicit**: constructing a ``Track(...)`` by hand checks nothing (a partial ``Track``
is a legal value), and the library calls ``validate`` where every field is about to be
used: ``out=``, ``view``, ``copy_from`` and ``clone``. The runtime utilities check only
the members they read (`Capability protocols`_).

.. _results-views:

Views, copies and snapshots
---------------------------

``view(e0, e1)``
   An env-slice view of envs ``e0`` to ``e1 - 1``: the same class (a subclass is kept),
   whose per-point arrays are ``arr[e0 * S : e1 * S]``, per-env arrays ``arr[e0 : e1]``
   and attributes sliced alike; ``None`` fields stay ``None``, ``fail_bits`` is shared,
   and ``layout`` is ``Layout(e1 - e0, S, device, closed)``. A Warp slice of a 1-D
   array is a pointer offset with the same element stride, so a view is contiguous,
   allocates nothing, launches nothing and keeps its parent's memory alive. It validates
   the parent first and raises ``IndexError`` unless ``0 <= e0 < e1 <= E``
   (``Track.view(5, 9): env range outside [0, 8)``; ``Track.view(2, 2): empty env range:
   need e0 < e1``). Views of views work, and a view
   remembers its env offset in its root result (the seeds rule of `Mixed batches`_).
   Callable anywhere, inside a capture too.

``copy_from(src, *, stream=None, ctx=None)``
   Copies ``src`` into this result: one ``ctx.copy`` per array this result holds. ``src``
   must be an instance of the same class (``TypeError``) with the same canonical layout
   (``ValueError``: ``Track.copy_from: layouts differ: this Track is Layout(4, 384,
   'cpu', True), src is Layout(8, 384, 'cpu', True)``). A field ``None`` here is
   skipped; a field held here but ``None`` in ``src`` raises; every attribute key here
   must be in ``src`` with the same dtype (extra keys in ``src`` are ignored). The same
   array on both sides is skipped (``t.copy_from(t)`` is a no-op), and a partial overlap
   raises ``ValueError`` naming the field. It allocates nothing and never syncs: under a
   capture it records one copy node per array. This is the hot-path snapshot: copy into a
   result preallocated with ``alloc``.

``clone(*, stream=None)``
   A new result that owns its memory: one new array per held field and attribute, ``None``
   fields kept, the class kept, the same ``layout``, and ``fail_bits`` shared (the same
   object). It validates first, so a malformed hand-built result raises. It **allocates
   and blocks on every call**: keep it off the hot path, and use ``copy_from`` into an
   ``alloc`` result instead. It raises ``CaptureError`` before allocating under any
   capture (a Warp CPU or CUDA capture of Warp's current stream or of the producer's
   stream, or a torch capture): inside a graph, ``copy_from`` into a preallocated
   ``alloc`` result.

``alloc(layout, *, attributes=None)`` (a classmethod)
   A new result of ``layout`` with every field allocated (the optional ones included)
   in the "no track yet" state: per-point fields and attributes at their padding,
   ``count``, ``valid``, ``truncated``, ``generation`` and ``fail_mask`` zero,
   ``length`` and ``winding`` zero, ``fail_bits`` empty. ``attributes`` maps plugin keys
   (``"<package>.<name>"``) to dtypes (``{"acme.grip": wp.float32}``; the names
   ``"float32"``, ``"int32"``, ``"vec2f"``, ``"vec3f"`` work too). It allocates, blocks
   and raises ``CaptureError`` under a capture, like ``clone``.

**Streams.** ``validate`` and ``view`` do no device work. The others order themselves as
follows:

- ``copy_from`` runs on ``ctx`` when given (``stream=`` and ``ctx=`` together raise
  ``TypeError``), else on the stream resolved from ``stream=``, the ambient
  ``LaunchContext.scope``, the stream that produced ``src``, torch's capture stream and
  Warp's current stream, in that order. "The stream that produced ``src``" is the
  default stream of the generator whose result it is (``TrackGenerator(...,
  stream=s)``), and a view inherits it; a hand-built or ``alloc`` result has none. When
  you generated on another stream (``gen.generate(stream=t)``), pass ``stream=t`` or
  order the streams with events: ``copy_from`` only knows the producer's default.
- ``clone`` allocates on Warp's current stream (where ``wp.empty`` allocates), makes it
  wait on the producer's stream (``stream=`` names the producer; else the ambient scope,
  the generator's default, torch, current), copies on Warp's current stream, then waits
  for those copies on the host. The clone is therefore **usable on any stream** as soon
  as it returns: read it (``.numpy()``, a utility), or ``copy_from`` into it on a
  generator's stream. Outside a capture and with no producer stream involved, it copies
  the same bytes on the same stream as the 0.2.0.dev4 ``clone`` (and then waits for
  them).
- ``alloc`` allocates and fills on Warp's current stream (no ambient or torch stream is
  adopted), like ``wp.full``, then waits for the fills on the host: the result is
  **usable on any stream** on return. ``snap = Track.alloc(track.layout)`` followed by
  ``snap.copy_from(track)`` on the producer's stream needs no sync of yours (the fills
  cannot land after the copies, and the producer's stream never writes memory Warp's
  current stream is still using), nor does a generator later built with ``out=`` a view
  of it on another stream.

.. list-table::
   :header-rows: 1
   :widths: 16 22 38 24

   * - Call
     - Allocates
     - Device work
     - Under a capture
   * - ``validate``
     - nothing
     - none
     - fine (host only)
   * - ``view``
     - nothing
     - none
     - fine
   * - ``copy_from``
     - nothing
     - one copy per held array (resolved stream)
     - recorded
   * - ``clone``
     - every held array
     - a stream wait, one copy per array (current stream), a host wait
     - ``CaptureError`` before allocating
   * - ``alloc``
     - every array
     - one fill per array (current stream), a host wait
     - ``CaptureError``

.. _results-view-safety:

View safety
~~~~~~~~~~~

A kernel that takes a view sees an ordinary array whose element ``0`` is the view's
first row and whose length is the view's. It is view-safe when it touches only indices
in ``[0, shape[0])`` derived from ``wp.tid()`` and the per-env layout (``e * S + i``),
reaches every buffer through its arguments and assumes nothing about two arguments being
distinct allocations: the bounds-correctness every kernel already needs on an owned array
of the same shape. Every TrackGen kernel that takes a result array (the producers: the
arc-length, winding, offset, validity, lift, frames, resample-scan and attribute-carry
kernels, the commit's hold / restore / poison kernels, the gate frame, validity, relax
and elevation kernels, the seed-generation copy; the consumers: every runtime utility)
indexes this way; none uses an absolute pointer or the parent's size. Plugin kernels
follow the same rule, and the conformance kit's ``output.view`` check runs every
generator and stage with ``out=`` a view of a larger poisoned result
(:doc:`/contributing/testing-a-plugin`).

.. _results-out:

Caller-owned results: ``out=``
------------------------------

``TrackGenerator(config_or_spec, rng=None, *, seeds=None, stream=None, out=None)``,
``GateGenerator(..., out=None)`` and ``pipeline.build(spec, *, ..., out=None)`` take a
result you own. The Program's result **is** ``out``: its arrays are the ones the stages
and the commit write, from the warm-up on, and their addresses never change, so a graph
of yours that reads them keeps working. The checks run on the host, before anything is
allocated:

1. ``out`` is a ``Track`` (a ``GateSequence`` for a gate pipeline) instance, else
   ``TypeError``: an object that merely has the fields is refused (0.2.0.dev4 accepted
   one).
2. ``out.validate()`` passes, with every required field present, and its canonical
   layout equals the pipeline's (``ValueError``: ``out: a Track for Layout(4, 384,
   'cpu', True); this pipeline writes Layout(8, 384, 'cpu', True)``). Messages name the
   field as ``out.<field>`` (``out.truncated is None; this pipeline needs it``).
3. An optional field ``out`` leaves ``None`` (``generation``, ``fail_mask``,
   ``half_width``; ``truncated`` is required) is not added to ``out``: the Program keeps
   a private buffer for it, which ``tap`` reaches.
4. ``out.attributes`` follows the rule of `Attributes of a caller-owned result`_.
5. ``out`` may not overlap the result of a live Program (`Disjoint outputs`_).

After the build, ``out.layout`` is set when it was ``None``, ``out.fail_bits`` names the
pipeline's bits, and the generator's default stream is recorded as ``out``'s producer
stream (for ``copy_from`` and ``clone``). A view keeps its parent's memory alive, so
``out=big.view(...)`` needs nothing else to stay valid.

.. _results-mixed-batches:

Mixed batches
~~~~~~~~~~~~~

Several generators can write disjoint env ranges of one result, each through a view of
it: a batch whose envs come from different generators or configurations, read by one set
of utilities and recorded into one graph.

.. code-block:: python

   import warp as wp
   from track_gen import TrackGenerator
   from track_gen.pipeline import presets
   from track_gen.results import Layout, Track

   E, dev = 8, "cuda:0"
   seeds = wp.array(list(range(100, 100 + E)), dtype=wp.int32, device=dev)
   done = wp.zeros(E, dtype=wp.int32, device=dev)       # your done mask

   big = Track.alloc(Layout(E, 384, dev))
   a = TrackGenerator(presets.track(num_envs=4, device=dev, generator="bezier"),
                      seeds=seeds[0:4], out=big.view(0, 4))
   b = TrackGenerator(presets.track(num_envs=4, device=dev, generator="polar"),
                      seeds=seeds[4:8], out=big.view(4, 8))

   s = wp.Stream(dev)
   with wp.ScopedCapture(stream=s) as cap:              # one caller graph
       a.generate(stream=s, advance=True, mask=done[0:4])
       b.generate(stream=s, advance=True, mask=done[4:8])
   wp.capture_launch(cap.graph, stream=s)               # big holds both halves

- **One stride.** Every generator of the batch uses the same ``n_max`` (the same
  ``max_gates`` for gates), the stride of ``big``.
- **Seeds.** Pass each generator its ``[E_i]`` slice of one seed array, as above (a slice
  is bound by pointer, like any seed array). With ``seeds=None`` or an ``int``, env ``e``
  of a view starting at env ``k`` of its root result gets seed ``base + k + e`` (``base``
  is the ``int``, or ``0``): a mixed batch with default seeds is seeded like one batch,
  and two halves running the same generator never get the same seeds. A seed array, a
  ``PerEnvSeededRNG`` or a ``SeedState`` is used as given. A view rebuilt by hand
  (``dataclasses.replace``) has no offset.
- **Masks.** Each generator takes its own ``[E_i]`` slice of your done mask. A masked
  generation holds and restores only that generator's rows; the other generators' rows
  are untouched.
- **Streams.** The generators may share a stream (ordered) or use different ones (their
  memory is disjoint). Order your readers of ``big`` after both.
- **Fail bits.** Each generator's ``fail_bits`` is set on its own view object (and on
  ``gen.program.fail_bits``); ``big.fail_bits`` stays empty. Generators with different
  plugin checks may give one bit different names: read ``fail_mask`` per range with that
  range's ``fail_bits``.
- **Utilities.** A utility bound to ``big`` reads every env; one bound to a view reads
  its range only. A ``Course`` builds its own generator and takes no ``out=``.

.. _results-disjoint:

Disjoint outputs
~~~~~~~~~~~~~~~~

Two live Programs may not write the same memory. At every build, the planner looks up the
allocation that holds ``out.count`` and the byte range ``out.count`` covers in it; when the
range overlaps that of a Program still alive, the build raises ``PlanError``:

.. code-block:: text

   out: envs [0, 4) of this storage are already written by a live Program (envs [2, 6)
     of the same allocation); views given to different generators must be disjoint; to
     rebuild into the same storage, release the earlier generator / Program first
     (del gen)

Every successful build registers its result, an owned one included, once the Program is
usable (after its warm-up; a facade after its whole construction), so
``build(spec, out=gen.program.result)`` raises while ``gen`` is alive, and a build that
raised registers nothing (a retry with the same ``out=`` is not refused). Once a Program
is garbage (``del gen``), its range is free again: to rebuild into the same storage
(new static params, keeping the pointers a graph of yours holds), drop the earlier
generator first. 0.2.0.dev4 allowed two live Programs over one ``out``. The check is
keyed on ``count``: two hand-built results that share per-point storage but have
separate ``count`` arrays are not detected.

.. _results-out-attributes:

Attributes of a caller-owned result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a view of a larger result, ``out.attributes`` may hold any keys, so a mixed batch can
share one attribute array between generators that export it and generators that do not:

- A key **this pipeline exports** (an exported per-point plugin channel, see
  :ref:`pipeline-attributes`) is bound: its final version is ``out``'s array (the dtype
  must be the family's, ``ValueError`` otherwise), it is listed in
  ``Program.attributes``, and the commit holds, restores and poisons it.
- **Every other key** (a family this pipeline does not write, or one it carries
  without exporting it, whose dtype it must then have) is checked (``[E * stride]``, an
  attribute dtype, the device, contiguous) and **filled once with its padding** (NaN, or
  ``-1``) at build, ordered
  before the warm-up, and never written again: the pipeline keeps its own private buffer
  for such a channel. The rows of this generator hold the padding through every plain,
  masked and poisoned generation, and ``Program.attributes`` does not list the key.
- An exported key that ``out`` leaves out is not added to it (the Program keeps a
  private buffer, ``tap`` reaches it).

On the **whole** of its allocation (``out`` is not a view of a larger result, or a view
of all its envs), no other generator can write a key, so a key this pipeline does not
export is refused, as in 0.2.0.dev4 (``ValueError: out.attributes['acme.gripp']: this
pipeline does not export 'acme.gripp'; exported: ['acme.grip'] ...``): a misspelled key
is reported instead of NaN-filling your buffer. An exported key of the wrong dtype
raises ``ValueError`` (``out.attributes['acme.tag'] is float32; this pipeline exports
'acme.tag' as int32``).

So in a batch of a generator that exports ``acme.grip`` and one that does not,
``big.attributes["acme.grip"]`` holds grip values on the first generator's envs and NaN
on the other's.

.. _results-protocols:

Capability protocols
--------------------

A runtime utility needs a few arrays of its source, not a whole ``Track``. Each
capability is a ``typing.Protocol`` (``runtime_checkable``) in ``track_gen.results``: the
members a family of consumers shares. Each consumer checks the subset it actually reads
(``check_source(..., members=...)``, listed in :doc:`/utilities/overview`), so a source
needs only those:

.. list-table::
   :header-rows: 1
   :widths: 18 36 46

   * - Protocol
     - Members
     - Read by
   * - ``HasCenterline``
     - ``center``, ``arclen``, ``length``, ``count``
     - ``TrackLocalizer``, ``localize.curvature``, ``localize.speed_profile``
   * - ``HasBand``
     - ``center``, ``outer``, ``inner``, ``count``, ``valid``
     - ``CollisionChecker``, ``PropSampler``, ``CheckpointSampler``,
       ``HeightFieldBaker``
   * - ``HasGates``
     - ``position``, ``count``, ``valid``
     - a gates-mode ``Course``'s course line (``course.course_line``)
   * - ``HasGateFrames``
     - ``HasGates`` plus ``forward``, ``left``, ``right``, ``orientation``,
       ``half_size``
     - a gates-mode ``Course``'s frame checker, ``CheckpointSet.from_gates``
   * - ``HasGeneration``
     - ``generation``
     - generation-aware consumers (roadmap PR14), checked together with another
       protocol: ``check_source(obj, (HasCenterline, HasGeneration), ...)``

The course line (``CourseLine``) and the frame checker (``FrameChecker``) are internal
to ``Course`` in 0.x (no public import path); the other consumers are public.

A ``Track`` satisfies ``HasCenterline`` and ``HasBand``, a ``GateSequence``
``HasGates`` and ``HasGateFrames``, a ``Centerline`` ``HasCenterline``; each satisfies
``HasGeneration`` when its ``generation`` is set. Any object with the members works: your
own dataclass, a ``SimpleNamespace``, a proxy. No protocol has a ``layout`` member:
a source without one gets its layout inferred.

**Shipped protocols are frozen.** Their members never change; a new capability is a
new protocol (a later ``HasWidth`` or ``HasAttributes``), and a consumer that needs two
capabilities names both. The library itself never calls ``isinstance`` against a
protocol (``runtime_checkable`` member checks differ between Python 3.10 / 3.11 and
3.12 for objects that compute attributes in ``__getattr__``): it checks sources with
``check_source``. Use the protocols for typing, for documentation, and in your own
``isinstance`` checks.

``Centerline``
~~~~~~~~~~~~~~

A centerline without a road band: ``center``, ``tangent`` (``[E * S]`` ``vec3f``),
``arclen`` (``[E * S]`` ``float32``), all NaN past ``count[e]``; ``length``
(``[E]`` ``float32``), ``count``, ``valid`` (``[E]`` ``int32``), ``generation``
(optional) and ``layout``, with ``validate``, ``view``, ``copy_from``, ``clone`` and
``alloc`` as above. A gates-mode ``Course`` localizes its agent on
``course.course_line.centerline``, a ``Centerline`` sampled from a closed Catmull-Rom
spline through the gate centers (:doc:`/gates-3d`). ``CourseLine.track``, which used to
return a ``Track`` with NaN ``outer`` / ``inner`` / ``normal`` and a zero ``winding``,
is a deprecated alias of ``centerline`` (it emits a ``DeprecationWarning``; removed at
1.0).

.. _results-check-source:

``check_source``
~~~~~~~~~~~~~~~~

.. code-block:: python

   check_source(obj, protocol, *, consumer, topology="closed", members=None) -> Layout

The check every utility runs at construction, public so that your own utilities (and
roadmap PR14's components) validate their sources with the same rules and messages:

1. ``protocol`` is a capability protocol of ``track_gen.results``, or a tuple of them
   (``(HasCenterline, HasGeneration)``: their members are checked together, against one
   layout); anything else raises ``TypeError``. ``members``, when given, must be members
   of it (``ValueError: check_source: members ['centre'] are not members of HasBand
   (center, outer, inner, count, valid)``): a consumer's typo is reported as such.
2. Each member of ``protocol`` (or of ``members``, the subset a consumer reads) is read
   with ``getattr``, once; missing (or ``None``) ones raise
   ``TypeError`` (``CollisionChecker needs a HasBand source (center, outer, inner,
   count, valid): Centerline has no 'outer', 'inner'``).
3. The layout is ``obj.layout`` when it is a ``Layout``. On a ``Track``,
   ``GateSequence`` or ``Centerline`` it must be a ``Layout`` or ``None``
   (``TypeError`` otherwise); another object's ``layout`` attribute that is not a
   ``Layout`` (a string, an enum of another library) is ignored. Otherwise the layout is
   inferred from the checked members only: ``num_envs`` from ``count`` (else the first
   per-env member), ``stride`` from the first per-point member (``1``, i.e. undefined,
   when every checked member is per env, e.g. ``HasGeneration`` alone).
4. Only those members are checked, with the schema's dtype and form and the messages of
   `validate(need=())`_, prefixed by the consumer (``CollisionChecker: Track.inner:
   expected a contiguous [3072] vec3f ...``). Fields outside the protocol are never
   looked at, and the object's own ``validate`` method, if any, is never called.
5. With ``topology="closed"``, a source whose layout says ``closed=False`` raises
   ``ValueError`` (`Topology`_).

It returns the layout (with the canonical device). The utilities pass ``members=`` when
they read fewer members than the protocol lists (``CollisionChecker``, ``PropSampler``,
``HeightFieldBaker``, ``FrameChecker``, ``CheckpointSet.from_gates``, ``curvature``); see
:doc:`/utilities/overview`. A utility **binds** the arrays it checked at construction and
uses those on every call: reassigning a member of the source afterwards has no effect
(rebuild the utility), and a proxy computing its members is read once.

.. _results-topology:

Topology
--------

``Layout.closed`` says whether the loops are closed. Every built-in stage, generator and
utility assumes a closed loop, and 0.x accepts nothing else; the flag is a host-side
contract, checked before anything runs (no kernel takes it):

- ``BatchSpec.closed`` must be a ``bool`` (``PlanError: BatchSpec.closed must be a bool,
  got 1``). ``closed=False`` is refused right after the spec's shape checks, before any
  generator is bound or any ``Params.validate`` runs, naming the stages that support
  closed loops only (``PlanError: BatchSpec.closed=False: open tracks are not supported
  in 0.x: stages 'generator' (bezier), 'resample' (resample.constant_spacing), ...
  support closed loops only (Stage.topology='closed'); open topology arrives after
  1.0``). A pipeline whose stages all declare ``topology="any"`` is refused too.
- Stages declare ``Stage.topology`` (``"closed"`` or ``"any"``); every built-in is
  ``"closed"``, and a legacy ``GeneratorSpec`` counts as ``"closed"``.
- The utilities that read a result declare ``topology = "closed"`` as a class
  attribute (``DiscChecker`` reads raw disc arrays and has none) and refuse a source
  whose layout has ``closed=False`` (``ValueError: CollisionChecker supports closed loops only
  (topology='closed'); its source has layout.closed=False``). ``copy_from`` refuses to
  copy across topologies (the layouts differ).

Open tracks are scheduled after 1.0.
