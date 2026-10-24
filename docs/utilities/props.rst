Boundary props
==============

``track_gen.props.PropSampler`` resamples a boundary at a set spacing into
instancing poses — cones (``mode="points"``) or wall pieces
(``mode="segments"``, chord midpoint + yaw + length).

.. figure:: ../assets/boundary-props.png
   :alt: One track with cone poses along both boundaries (points mode) and
         wall chords along both boundaries (segments mode).

   The two prop modes on one track: cone poses (left) and wall chords
   (right), sampled from the inner and outer boundary independently.

Both modes snap the requested spacing per environment —
``n = clamp(round(perimeter / spacing), 3, max_props)`` at effective step
``perimeter / n`` — so every closed ring places props with no seam gap or
doubled prop. ``PropSet.count``/``truncated``/``step`` report what was
actually placed; see the left panels of the overview figure on
:doc:`the utilities overview </utilities/overview>`.

``PropSampler(track, spacing)`` with ``max_props=None`` sizes the ring once, at
construction, from the bound batch (a readback; a supported legacy convenience), and
``sampler.max_props`` reads it. Tracks that grow later (a per-env ``size`` curriculum)
then get clipped rings (``truncated``, at a coarser step). Size it from the plan instead,
with no readback, through the component:

.. code-block:: python

   from track_gen.tools import ComponentContext, Props

   cc = ComponentContext.for_producer(gen)        # the generator's layout and plan
   props = cc.add(Props(0.2, boundary="outer"), gen.generate())
   ring = props.result                            # the PropSet; props.utility is the sampler
   props.on_generate(ctx, None)                   # after every generation (ctx: a LaunchContext)

``Props(spacing)`` takes ``max_props = max(3, ceil(max_loop_length / spacing))`` from
the plan's bound (``max_loop_length = 1.5 * n_max * spacing + dZ``,
:doc:`/contributing/writing-a-component`): 173 props at spacing 0.2 for the
``TrackGenConfig`` defaults, which holds the boundary of every track that is not
truncated, at any size (:ref:`tuning-size-utilities`). A plan that declares no bound (an
elevation without declared z ranges, a variable width, a generator that provides its own
altitudes or half-widths) makes ``Props`` raise ``ValueError``: pass ``max_props``.

``PropSampler(track, ...)`` takes any ``HasBand`` source (a ``Track``, a
``track.view(e0, e1)``, or an object with the members) and checks what it reads,
``outer``, ``inner``, ``count`` and ``valid``, at construction (see
:ref:`utilities-sources`). An invalid env (``valid[e] == 0``) gets no props since
0.2.0.dev6: ``count = 0``, NaN poses, ``truncated = 0``, ``step`` NaN, as an env with no
track (:ref:`utilities-invalid-envs`).

Props are rendering-only by design. To make point props physical (cones a
vehicle can clip), feed their positions to
:class:`track_gen.collision.DiscChecker` — see
:doc:`collision </utilities/collision>`.

API: :class:`track_gen.props.PropSampler`,
:class:`track_gen.props.PropSet` — see the
:doc:`API reference </reference/api>`.
