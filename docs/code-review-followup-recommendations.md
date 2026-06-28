# Code-Review Follow-Up — Recommended Changes (for review)

**Date:** 2026-06-27
**Status:** Draft for your review — nothing here is implemented yet.
**Context:** The 2026-06-27 full review's Critical + Important findings and the cheap/safe
suggestions are already fixed and merged to `main`. This doc is the remaining backlog —
the items intentionally deferred because they need a design decision, are larger refactors,
or are low value. Each is tagged with effort, risk, and my recommendation. Tick the ones
you want and I'll plan/execute them.

Source detail for every item: `docs/code-review-2026-06-27.md`.

---

## Tier 1 — Recommended (clear wins, low risk)

- [ ] **R1. Surface or remove `sep_cache_overflow`** — `warp_relax.py:168` (alloc `warp_pipeline.py:1437`)
  The counter is incremented when a bead has more broadphase candidates than `relax_sep_cache_slots`, but **nothing ever reads it**, so an undersized cache silently drops real collision candidates with zero diagnostic. *Recommendation:* expose it (warn on the non-capture path when `overflow > 0`, or surface it on the scratch so callers can detect an undersized `relax_sep_cache_slots`), or delete the dead counter so it doesn't imply a safety net that isn't there. *Effort: S–M · Risk: low.*

- [ ] **R2. Broaden RNG parameter validation** — `rng_utils.py:25`, dispatchers in `rng_kernels.py`
  No checks that a `wp.array` `seeds` length equals `num_envs`, or that `low <= high` / `std >= 0` / `lam > 0`. A mismatch causes silent partial init or undefined kernel behavior. *Recommendation:* add cheap host-side guards (`seeds.shape[0] == num_envs`, `low <= high`, `std >= 0`, `lam > 0`) with clear messages. *Effort: S · Risk: low (only rejects already-broken calls).*

- [ ] **R3. Validate `min_gates` reachability for point-family gate generators** — `gate_generator.py:38`
  Construction only checks the generator's *max* count against `min_gates`; bezier/hull draw per-env counts in `[min_num_points, max_num_points]`, so `min_gates=11, min_num_points=9` constructs fine but silently yields `valid=0` envs. *Recommendation:* also validate the *minimum* producible count (or document `min_gates` as a validity floor, not a generation guarantee). *Effort: S · Risk: low.*

- [ ] **R4. Validate documented config bounds that currently aren't enforced** — `types.py`
  `relax_solver` (doc: `{xpbd,energy,tp_sobolev}`), the checkpoint steering knobs (`checkpoint_angle_jitter < 1`, `checkpoint_steer_gain ∈ (0,1]`, etc.), and the style `*_range` tuples (`rad_range`/`scale_range`/`handle_clamp_frac_range` — inverted or wrong-arity tuples flow straight into per-env RNG). Their neighbors are already validated. *Recommendation:* add range/enum/`lo<=hi` checks to match the surrounding fields. *Effort: S · Risk: low.*

- [ ] **R5. Fix the over-broad `shape[0] == 1` RNG collapse** — `rng_kernels.py:132` (every dispatcher)
  `if shape[0] == 1` collapses to the 1D kernel even for shapes like `(1, 5)`: it then draws 1 value per env while advancing state by `prod((1,5)) = 5`, desyncing the state advance from values drawn. *Recommendation:* special-case only the exact scalar-per-env shape (`len(shape)==1 and shape[0]==1`). *Effort: S · Risk: low–med (touches the shared dispatch path — test 2D/3D thoroughly).*

---

## Tier 2 — Consider (design decision or medium refactor)

- [ ] **C1. Quaternion uniform-over-SO(3) distribution** — `rng_kernels.py:1441+`
  We already decorrelated axis/angle, but axis-angle with a *uniform* angle is **not** uniform over SO(3). *Recommendation:* if uniform rotations are intended, switch to "sample 4 normals and normalize." **Note:** `quaternion` is unused in the runtime today, so this is low urgency — only worth doing if a consumer needs correct uniformity. *Effort: S · Risk: behavior change for any future consumer.*

- [ ] **C2. Polar generator self-intersection fallback** — `warp_generate_polar.py:178`
  Polar marks every env valid with no self-intersection check or polygon fallback (bezier/hull both have one), so self-crossing polar centerlines lower downstream yield. *Recommendation:* add the shared `self_intersections_inplace` + straight-chord polygon fallback (the polar control polygon is angle-monotone/star-shaped, so a chord fallback is simple), **or** explicitly document that polar relies on the inflate gate and accepts lower yield. *Effort: M · Risk: med (changes polar output for crossing cases — re-baseline any yield assertions).*

- [ ] **C3. Hull polygon fallback may itself self-intersect** — `warp_generate_hull.py:250`
  The straight-chord fallback connects inward-displaced midpoints, so it isn't guaranteed simple and isn't re-tested. *Recommendation:* bound the inward displacement, or re-test the fallback and drop back to the un-augmented angle-sorted polygon if it still crosses. *Effort: M · Risk: med.*

- [ ] **C4. Hoist the duplicated `_normalize_centerline_k`** — voronoi/checkpoint/polar
  Three verbatim copies of the bbox-center-and-scale kernel invite drift. *Recommendation:* move one copy into `warp_pipeline` and call it from all three generators. *Effort: M · Risk: low–med (pure refactor; verify all three still produce identical output).*

- [ ] **C5. `gate_width` collision misses collinear/touching bars** — `warp_gate.py:434`
  The width-overlap check uses strict proper-intersection only, so two nearly-collinear wide gates whose bars overlap end-to-end aren't flagged. *Recommendation:* add a collinear-overlap / min-segment-distance test for wide gates, or document that the gate detects only proper crossings. *Effort: S–M · Risk: low.*

- [ ] **C6. Investigate `_corner_angles_gate_k` neighbor wrap (`mod P` vs `mod cnt`)** — `warp_generate.py:425`
  Closing-seam corners are skipped by the angle gate because neighbors wrap `mod P` (buffer size) not `mod cnt` (real count). The sibling tangent/assemble kernels wrap `mod cnt`. *Recommendation:* confirm whether the `mod P` is matching a legacy oracle (then comment it) or a real bug (then fix to `mod cnt`). *Effort: S + investigation · Risk: low–med (consumed by the gate path/tests).*

---

## Tier 3 — Optional (low value / cosmetic)

- [ ] **O1. Spurious `N_max` truncation warning** — `warp_pipeline.py:1088`
  The `RuntimeWarning` fires whenever `count == n_max`, including the legitimate exactly-fits case. *Recommendation:* record a real pre-cap overflow flag and warn only on genuine overflow. *Effort: S–M · Risk: low.* (This is the persistent "1 warning" in the test run.)

- [ ] **O2. `count=None` divisibility assertion** — `warp_pipeline.py:1517`
  The convenience path infers `E = center.shape[0] // n_pts` with no exactness check; a mismatched length yields a silently wrong layout. *Recommendation:* `assert center.shape[0] % n_pts == 0`. *Effort: S · Risk: low.*

- [ ] **O3. Atomic-area outer/inner label determinism** — `warp_pipeline.py:88`
  Order-nondeterministic atomic-float area sums can flip the outer/inner label for tracks with near-equal candidate areas (validity usually rejects these). *Recommendation:* a clarifying comment is probably enough; a deterministic per-env reduction only if bitwise reproducibility matters. *Effort: M (if changed) · Risk: med.*

- [ ] **O4. Registry symmetry / dead-guard cleanups**
  (a) `generator_registry.register` lacks the cross-module dup detection the gate registry has — **note:** adding it broke a legitimate test idiom (`test_generator_registry` re-registers `bezier` from the test module), so it needs a test redesign first. (b) The dead `if cluster > 5` guard in `warp_generate_voronoi.py:82` (skipped earlier to avoid kernel recompile churn). *Effort: S · Risk: low, but low value.*

---

## Suggested order if you want a single follow-up pass
1. **Tier 1** as one batch (all small guards/diagnostics, low risk) — clear "good place" wins.
2. **C4** (hoist the normalize kernel) for maintainability, and **C1** only if quaternions will actually be used.
3. **C2 / C3** (generator fallbacks) as a focused "raise polar/hull yield" effort with re-baselined assertions.
4. Tier 3 as cleanup whenever convenient.
