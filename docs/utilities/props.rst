Boundary props
==============

``track_gen.props.PropSampler`` resamples a boundary at a set spacing into
instancing poses — cones (``mode="points"``) or wall pieces
(``mode="segments"``, chord midpoint + yaw + length).

Both modes snap the requested spacing per environment —
``n = clamp(round(perimeter / spacing), 3, max_props)`` at effective step
``perimeter / n`` — so every closed ring places props with no seam gap or
doubled prop. ``PropSet.count``/``truncated``/``step`` report what was
actually placed; see the left panels of the overview figure on
:doc:`the utilities overview </utilities/overview>`.

Props are rendering-only by design. To make point props physical (cones a
vehicle can clip), feed their positions to
:class:`track_gen.collision.DiscChecker` — see
:doc:`collision </utilities/collision>`.

API: :class:`track_gen.props.PropSampler`,
:class:`track_gen.props.PropSet` — see the
:doc:`API reference </reference/api>`.
