# Phase-1 spline self-intersection cleanup — design

**Date:** 2026-06-19
**Status:** draft (brainstorm) — pending approval
**Branch:** `feat/phase1-closed-assemble` (builds on F1+F2, commit `dd46b24`)
**Goal:** Close the last ~0.6% of Phase-1 splines that self-intersect on a *clean* corner
polygon, via two targeted fixes — without pruning/distorting the 99.4% of good tracks.

## Background (what we know)

- The corner **polygon** from angle-sort is always simple (theorem; 0/4096). All centerline
  self-crossings are **spline-level** (Bézier), not ordering.
- F1 (closed assemble) + F2 (adaptive handle clamp) took single-attempt crossing-free
  94.8% → 99.4%. The residual ~0.6% (26/4096) are splines that self-cross over a clean
  polygon. Two corner-geometry mechanisms (measured):
  - **(A) close pair** — two sampled corners very close (min pair ≤ ~0.2). Minority (≈2/5
    deep failures). Root cause: the sampler's jitter `±0.5` is ~5× the grid cell `0.1`, so
    `min_point_distance` does **not** actually enforce a minimum spacing.
  - **(B) corner near a non-adjacent edge** — a corner within ~0.06–0.12 of a far edge
    (the unifying signature of all 5 deep failures). This is also the **hairpin** mechanism.
- A distance **threshold cannot separate** crossers from good tracks: they sit inside the
  valid cloud (`viz/out/corner_geom_scatter.png`); e.g. `corner-edge < 0.3` catches 24/26
  crossers but flags ~15% of valid tracks (the hairpins). So we must **not** prune by radius.

## Fix A — make `min_point_distance` enforce real corner spacing (the pair problem)

The sampler currently adds jitter `±0.5` on a `0.1` grid → distinct cells routinely collide.
**Fix:** scale the jitter to the cell so distinct cells stay separated, i.e. enforce a real
minimum corner spacing `≈ min_point_distance`. Two candidate mechanisms — pick on measurement:

1. **Jitter-scale (1-line, static):** jitter `← uniform(-j, j) * (min_point_distance*2)` with
   `j ≤ 0.25` so adjacent-cell corners stay ≥ `min_point_distance` apart. Simplest; but it
   shrinks the jitter that currently drives much of the shape diversity — **must measure
   diversity** (roundness/area/lobedness spreads) and yield; if they regress, use (2).
2. **Fixed-K corner separation pass (static, diversity-preserving):** keep the current random
   draw, then run `K` (≈3) parallel rounds pushing apart any corner pair closer than
   `r_min` (a mini repulsion on the ≤13 corners). Preserves the random distribution; only
   removes collisions.

**Recommendation:** try (1) first (cheapest); fall back to (2) if diversity regresses.
Either is static / branchless / CUDA-graph-capturable. Applies in `corner_sample` (oracle +
Warp). Expected: removes the close-pair crossers; modest overall crossing benefit (A is the
minority mechanism — Fix B does the heavy lifting).

## Fix B — straighten self-intersecting spline pieces to their chord (corner-to-edge)

The polygon is clean, so **reverting an offending Bézier piece to its straight chord (the
polygon edge) cannot introduce a polygon-level crossing** and removes that piece's overshoot.

**Algorithm (fixed-K, static):**
1. Assemble the dense centerline (F1+F2) as today — `count` cubic pieces.
2. Detect which **pieces** participate in a self-intersection (map the dense self-crossing
   edge pairs back to their owning piece `= dense_edge // npseg`).
3. Re-assemble those pieces as **straight lines** (lerp between the two corners) instead of
   the cubic. A straightened piece is just the polygon edge.
4. Repeat for fixed `K` rounds (≈2–3). **Convergence is guaranteed:** straightening only
   removes bulges and never creates a crossing between two straight edges (polygon simple);
   in the limit (all pieces straight) the centerline *is* the simple polygon. Most converge
   in 1 round.

Per-piece, not per-track: only the few offending pieces lose their rounding (a locally
slightly sharper corner), the rest of the track is untouched → **diversity preserved**.

**Interaction with F2:** Fix B surgically removes the residual overshoot crossings, so F2's
global clamp can be **relaxed** (larger `handle_clamp_frac`, restoring corner roundness) —
worth re-measuring once B lands. (Optional, not required for this spec.)

**Implication:** with B, pre-relax crossing-free → ~100%, so the self-intersection-gated
regen discussed earlier becomes largely unnecessary.

## Scope / non-goals

- Both fixes land in the **assemble/sampler** (oracle + Warp), preserving oracle↔Warp parity
  and CUDA-graph capture — same pattern as F1/F2.
- **Not** in scope here: removing the generation gates / regen loop (separate decision),
  thin-band regime sweep, the `handle_clamp_frac` re-tune (noted as a follow-up).

## Testing

- **Crossing-free:** single-attempt fat-band crossing-free ≥ 0.999 (was 0.994 with F1+F2)
  — Fix B should make the assembled centerline essentially always simple.
- **Spacing (Fix A):** min corner-pair distance ≥ ~`min_point_distance` across the batch.
- **Convergence (Fix B):** after K rounds, `self_intersections(dense) == 0` for ≥ 0.999.
- **Diversity guard:** roundness/area/lobedness spreads within ~10–15% of pre-fix.
- **Parity:** oracle == Warp (`test_warp_assemble` extended); CUDA-graph capture test green.
- Full suite stays green.

## Integration ("get it to main")

- Implement on `feat/phase1-closed-assemble` (TDD), atop F1+F2.
- **Recommended staging:** merge **F1+F2 to `main` now** (done, tested, high-value, low-risk),
  then land A+B as a second PR once measured — so the proven win isn't blocked on the
  longer-tail fixes. (Alternative: one combined merge if you'd rather ship together.)
