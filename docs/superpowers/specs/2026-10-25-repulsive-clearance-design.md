# Repulsive clearance and stable growth

## Binding requirements

The user authorized changing Repulsive output before release: every finished road band must clear every sampled inner disc and remain within the circular domain. The road is the ruled strip joining corresponding inner and outer edge samples; a disc in the closed loop's hole is allowed. A generation that cannot satisfy this bounded computation must report invalid and use the existing poison/keep commit policy. The rc2 tag, other generators, seed salts/draw order, public result schemas, allocation-free launch contexts, and CPU/CUDA capture remain unchanged.

No accepted optimizer, length-ratchet, or resampling update may silently cross the analytic circular barriers. Final validity must examine the actual offset road after relaxation and resampling. This closes the known seed-7 disc penetration and the wall crossing caused by hard length rescaling. The sampled force rings remain a shaping energy, never the clearance guarantee.

## Approaches and tradeoffs

1. Reject invalid finished roads only: small and exact, but permits unstable/infeasible solver motion and poor yield. Insufficient alone.
2. Project every vertex independently onto circle exteriors/interior: cheap, but segments can still cross discs and projections can create kinks. Rejected.
3. **Chosen implementation proposal:** analytic segment-clearance checks for proposed solver/resampling updates, bounded backtracking on the complete optimizer-plus-length update, and an exact road-cell/disc and wall check after the tail. Preserve the last feasible state when no proposal succeeds; geometrically infeasible or short-target runs fail closed. This costs fixed extra kernels and scratch but preserves graph capture and exposes bounded failure.

The normalized fixed-size step and area-only convergence are independently suspect. Use the roughness audit to select a bounded step rule that damps reversal rather than merely smoothing the final output. Convergence must require geometric movement/stationarity as well as area change; reaching a length target alone is insufficient. Deactivation may affect soft force only; hard barriers always remain active. Reject beta <= alpha because it makes obstacle force inert or attractive.

## Geometry and stage contract

Store the analytic sampled disc center/radius alongside the existing force rings without changing RNG consumption. Raw acceptance uses segment-to-disc distance and circular-wall containment, with conservative road-width reservation where the actual width is known. The final check uses the actual band vertices and a conservative convex hull of each four-vertex cell. Wall containment follows from convexity. Disc clearance checks all chords and triangle coverage of that hull; this covers folded bilinear or triangulated strips whose occupied lobes a polygon winding test can cancel. It is exact for ordinary convex cells and may reject a disc in an unused part of a concave cell hull. It must not mistake the whole loop hole for occupied road.

Final Repulsive barrier metadata must travel through declared stage ports with no public result fields. Planner validation must ensure a custom Repulsive pipeline cannot omit its final check or run geometry-changing stages after it. All scratch is allocated at construction; every launch/copy goes through ctx. Freeze is absorbing, and ordinary/unrolled/conditional lowerings remain byte-equivalent per device.

## Verification

Write red tests first for segment-through-disc with clear vertices, finite-width overlap, disc entirely in the loop hole, a disc enclosed in a single road cell, full-band wall containment, and seed 7. Then verify accepted raw updates, stage/periodic/final resampling, infeasible bounded failure, poison/keep, masks/frozen/zero-selected batches, and final band metadata ports. Representative seeds must have meaningful bounded movement/curvature behavior and no false area-only convergence. Verify eager/captured byte parity on CPU; root schedules CUDA and graph/conditional validation. Run targeted tests before the required full CPU lane and docs build. Rebaseline only Repulsive fingerprints/statistics after correctness evidence.

## Validity and bounded stopping ruling

The user required smooth growth and hard full-road clearance. The initial internal
plan additionally required every valid run to reach optimizer stationarity; that
was an implementation assumption, not a separate user decision. Controlled tests
show feasible, target-reaching smooth curves rejected solely because they still
admit objective descent at the fixed budget. Geometric construction success and
optional early-stop convergence are therefore separate. A finite raw curve that
meets its target-length tolerance and analytic constraints may reach the normal
budget, while the complete finished-road checks still decide Track.valid. Short
or geometrically infeasible runs fail. Frozen remains an absorbing early-stop
state requiring actual small movement and projected stationarity; rejected or
quantized no-op steps cannot establish it. This preserves bounded approximate
generation without claiming an energy minimum. The cost if this ruling is wrong
is accepting a geometrically valid curve that could become a better shape with
more iterations; assess actual growth curvature and motion and expose the budget
truthfully in the visual guide.

The controlled per-edge constraint experiment was not adopted: it stalled earlier
and produced sharper bends than the corrected total-length solver. Preserve the
independent proof and source as excluded project records, not unused runtime code.
