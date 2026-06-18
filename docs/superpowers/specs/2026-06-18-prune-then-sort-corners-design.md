# Prune-then-sort corner ordering — design

**Date:** 2026-06-18
**Status:** approved (brainstorm), pending implementation plan
**Goal:** Eliminate the figure-eight (winding-0) generation failures at their source by sorting the corners we actually use about *their own* centroid, instead of sorting all candidate corners and then truncating to a mis-centered angular wedge.

## Background

At the fat-band regime (1 m track width, ≤20 m box: `half_width=0.5`, `scale=10`, `constant_spacing`), a single generation attempt yields only ~0.53 valid tracks. The failures partition into three generation-stage buckets (relaxation is lossless — 0 post-relax folds): ANGLE (sharp corners, ~23%), SELF-X (winding-±1 self-crossers, ~20%), and TURN / figure-eight (winding-0, ~4%). This design targets the figure-eight bucket; ANGLE is handled separately (the `min_angle<=0` gate-skip), and deep SELF-X is a separate follow-up.

We chose generation-side **prevention** over post-hoc untangling after two spikes:
- A Warp PBD repulsive "untangle" resolves only ~8% of SELF-X and cannot make a winding-0 loop simple (a local winding term drives the turning integral to ±2π but produces a non-simple sliver). Negative — dropped.
- A deterministic Warp corner-ordering fix eliminates figure-eights, trims SELF-X ~21%, and lifts per-attempt yield ~+10 pts, at ~0.05 ms and fully graph-capturable. Adopted — this design.

## Root cause

The corner pipeline is **sort-then-prune** in both the Warp pipeline and the torch oracle:

- Warp `generate_centerline_warp` (warp_pipeline.py:1100–1102): `corners = ccw_sort(corner_sample(...))` sorts **all** `P = max_num_points` (e.g. 13) corners by angle about the centroid of **all P**; `count = corner_count_sample(...)` is drawn independently; `assemble` then uses only corners `0..count-1` (via `_pruned_corner`, real iff `i < count`).
- Oracle `generators._prune_corners` (generators.py:106–133): `points = ccw_sort(points)` (all P), then `pruned = where(arange < count, points, nan)`. The comment "disjoint angular wedges → simple polygon" records the (incorrect) original assumption.

Consequence: when `count < P`, the kept corners are the `count` smallest-angle corners of the full set — a contiguous angular **wedge** ordered about the **wrong** (all-P) centroid, closed by a long chord across the missing angular gap. The Bézier rounding of that long closing chord can overshoot into a loop, giving a self-intersecting, winding-0 figure-eight. Evidence (Spike 1): figure-eights have 2.4× the sort/use centroid offset of valid tracks and concentrate at low corner count (count=9 → 12.9% fig-8; count=13 → 0%); minimum angular gap does *not* separate them.

## The fix: prune-then-sort

Sort the corners we will actually use, about their own centroid:

1. Sample `count` first.
2. Take the first `count` corners (by original index — a uniform random subset, matching the intent of "vary the number of corners").
3. Angle-sort **those `count`** corners about **their own** centroid → an angularly-monotone (star-shaped, winding-±1) polygon. Corners `count..P-1` are left untouched (they are NaN-pruned downstream by `assemble`/`_pruned_corner` exactly as today).

This is the variety-preserving variant (it keeps the sampled corner radii/positions, only corrects the ordering and centroid). It is **not** the equispace hard-guarantee variant — see Scope.

### Components

- **Warp — count-aware sort.** Make `ccw_sort` / `_ccw_sort_k` accept `count`: compute the centroid over the first `count` corners (float64 accumulation, as today) and run the existing insertion sort over only those `count` (sort key `atan2(dx, dy)`, X-first quirk preserved). Rows `>= count` are passed through unchanged. Reorder `generate_centerline_warp` to sample `count` before sorting. `assemble` is unchanged.
- **Oracle — same reorder.** In `generators._prune_corners`, select the first `count` corners, sort that subset about its centroid (padding sorted to the tail, e.g. via an `+inf` key), NaN rows `>= count`. The dense centerline (`_assemble_centerline`) is unchanged.
- Both sides sort the same `count` real corners about the same (count) centroid, so `generate_centerline_warp` stays bit-for-bit equivalent to the oracle on the dense output (modulo the established ~5e-4 resample drift), and the `gates()` parity holds.

### Data flow (unchanged except the sort)

`seeds → corner_sample [E,P,2] → (count = corner_count_sample [E]) → ccw_sort(corners, count) → assemble → resample → gates → relax → inflate`.

## Expected outcome

Per Spike 1's re-sort-own-centroid variant (E=24,576, fat-band): figure-eight rate 3.5% → ~0.1%, SELF-X 30.9% → ~26%, per-attempt valid 65.6% → ~73.8%. Cost: one extra O(P) pass, ~0.05 ms at E=8192, no atomics, graph-capturable, deterministic. The residual ~0.1% figure-eights and the remaining SELF-X share the Bézier-overshoot mechanism (`rad`), addressed in the follow-up.

## Testing

- **Parity preserved:** the existing Warp-vs-oracle equivalence (count-aware `ccw_sort` == oracle reordered prune; `generate_centerline_warp` == oracle dense; `gates()` == oracle accept) must hold on `cpu` and `cuda`. Update any fixtures whose accept mask shifts because of the reorder.
- **Behaviour:** a new test asserting that on a fat-band batch the winding-±1 (turn_ok) rate rises and the figure-eight rate falls vs the old ordering (e.g. figure-8 rate < 0.5% after the fix), and that the count-aware sort produces an angularly-monotone polygon about the kept subset's centroid for representative counts.
- **Yield:** measure per-attempt valid-yield delta (regen=1) at the fat-band regime; assert it does not regress and record the improvement.
- All existing tests green on `cpu` (+`cuda` where available).

## Scope boundaries (YAGNI)

- **Equispace hard-guarantee variant — out.** Prune-then-sort preserves shape variety; we accept the residual ~0.1% figure-eights rather than redistribute corner angles uniformly.
- **Deep SELF-X (radius / `rad` Bézier overshoot) + coarser `spacing` — separate follow-up.** Needs its own diagnosis; spec after this lands and we measure the real residual.
- **PBD repulsion / winding-number term — dropped** (spike negatives).
- **Angle-gate skip (`min_angle<=0`) — already implemented** on the `angle-gate-skip` branch; same prevention strategy, merged separately.

## Risks / open questions

- Reordering changes the *shape distribution* of generated tracks (different corner orderings), so any test asserting specific end-to-end generation output or exact yield numbers will shift and must be re-baselined — not a correctness regression.
- Count-aware sort must keep the float64 centroid and X-first `atan2` key to preserve Warp↔oracle parity at the ULP level.
- Confirm `corner_count_sample` is reproducible/independent of the sort reorder (it is seed-driven, not corner-driven), so moving it earlier does not change the sampled counts.
