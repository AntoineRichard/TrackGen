# Gate Sequence Generation — Design Spec

**Date:** 2026-06-23
**Status:** Design approved in conversation, pre-implementation
**Primary constraint:** existing track generation must behave exactly the same.

## Goal

Add a public gate-generation surface for use cases where a closed centerline may
self-intersect and track width is irrelevant, but physical gates must remain usable. This
targets drone-style literature setups: a sequence of gate poses defines the course, and
the only strict geometric protection is that gates do not overlap/intersect and their
centres are not too close.

The new surface must be additive. `TrackGenerator`, `TrackGenConfig`, the existing
first-stage generator registry, XPBD relaxation, inflation, and track validity semantics
stay unchanged.

## Public API

Expose these names from `track_gen.__all__`:

- `GateGenConfig`: gate-specific configuration.
- `GateSequence`: batched gate result.
- `GateGenerator`: fixed-batch facade mirroring `TrackGenerator`'s ownership model.

`GateGenConfig` should reuse the existing generator/style knobs where practical, but it is
not a subclass of `TrackGenConfig`. It should carry only gate-relevant fields:
`generator`, `device`, `num_envs`, `max_gates`, `min_gates`, `min_gate_distance`,
`gate_width`, `gate_ordering`, and the existing per-generator shape knobs needed by the
native extractors.

`GateSequence` stores flat Warp arrays with `[E * max_gates]` stride:

- `position`: gate centre points.
- `tangent`: unit forward direction for the path through the gate.
- `normal`: gate crossbar direction, the left normal of `tangent`.
- `left` / `right`: gate segment endpoints, written when `gate_width > 0`.
- `valid`: `[E]` int32 validity flags.
- `count`: `[E]` int32 real gate counts.

Like `Track`, `GateSequence` should support `clone()` and the generator should return the
same instance on every call, overwriting pre-allocated buffers in place.

## Gate-Native Registry

Create a parallel registry, separate from the existing track generator registry:

```text
GateGeneratorSpec(
    name,
    alloc_scratch(config),
    generate(seeds_wp, config, out_gate_sequence, scratch),
)
```

The registry is public only through `GateGenConfig(generator=...)` and
`GateGenerator`; individual modules remain private. This keeps track generation isolated:
no existing `GeneratorSpec` signature changes, and no track module is required to know
about gate-specific outputs.

## Generator Semantics

Every standard generator gets a gate-native extractor rather than a generic
centerline-wrapper:

- **Bezier:** sample raw corner/gate points, optionally order them, and compute orientation
  from the Bezier spline handles/tangent construction. With `gate_ordering="ccw"` this is
  close to the current first-stage interpretation. With unordered/random-pair modes, the
  gates are allowed to define a self-crossing path as long as gate segments pass validity.
- **Hull:** use sampled points or augmented midpoint-displacement vertices as the gate
  anchors. Sorting is optional and controlled by `gate_ordering`.
- **Polar:** use polar knots or sampled spline anchors as gates; tangent comes from the
  local periodic spline derivative or adjacent anchors.
- **Voronoi:** use selected anchors as gates; tangent follows anchor order.
- **Checkpoint:** use the sampled checkpoints as gates in the first implementation.
  Bounded steering samples can be added later if checkpoints prove too sparse or uneven
  for a specific benchmark.

Track-specific repair paths are intentionally not inherited by default. Gate generation
does not need to rescue self-intersecting centerlines, relax width, or inflate borders.
Generator-local fallback should exist only if it is needed to produce finite gate poses or
to satisfy the gate-specific minimum-distance contract.

## Ordering Modes

`gate_ordering` starts with:

- `"ccw"`: angular sort around the centroid, useful for continuity with existing track
  families.
- `"raw"`: preserve sampled generator order where the generator has a natural order.
- `"random_pairs"`: build a path by pairing/ordering sampled gates randomly, allowing
  centreline self-intersection by construction.

Initial support is explicit: `"ccw"` for every native extractor, `"raw"` for generators
with a natural sampled order (`polar`, `voronoi`, `checkpoint`), and `"random_pairs"` for
Bezier and Hull. Unsupported combinations should raise a clear `ValueError` at
construction, not silently fall back.

## Validity

Validity is gate-specific and intentionally smaller than track validity:

- all real gate pose fields are finite;
- `count[e] >= min_gates`;
- pairwise gate-centre distance is at least `min_gate_distance` for non-identical gates;
- if `gate_width > 0`, gate line segments do not intersect except for an exact same-gate
  segment.

There is no thickness gate, no turning-number requirement, no relaxation check, and no
border self-intersection check. A course whose centreline crosses itself can still be
valid.

## Implementation Boundaries

The first implementation should add new gate modules and shared gate kernels instead of
editing track behavior. Reuse small geometry helpers where that does not change existing
semantics, but do not alter current track kernels to serve gates unless the change is a
strict refactor covered by track regression tests.

CUDA graph capture should follow the `TrackGenerator` model: fixed batch, pre-allocated
scratch, no hot-path allocation, and deterministic output for `(seed, config)`.

## Testing

Required tests:

- public API surface includes the new gate names and preserves existing track names;
- `GateGenerator` returns a `GateSequence` with stable buffers across calls;
- fixed seeds produce deterministic gates on the same device;
- `min_gate_distance` invalidates too-close gates;
- `gate_width > 0` invalidates intersecting gate segments;
- existing `TrackGenerator` outputs are unchanged for fixed seeds and representative
  configs;
- unsupported `gate_ordering` / generator combinations raise clear errors.

The initial implementation should keep tests focused on CPU Warp first, with CUDA tests
following existing optional-device patterns.
