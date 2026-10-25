# Repulsive edge-constraint projection: isolated implementation and evidence

This rejected, unwired experiment is archived beside this report in `repulsive_constraints.py` and `verify_repulsive_constraints.py`. It is outside the published `track_gen` package and default test collection. Root rejected adoption after the complete controlled probe described in the [README](README.md); the mathematical helper tests do not establish generator convergence or usable yield.

## Operator and cost

For unit edge directions `u_i`, `J v = u_i dot (v[i+1]-v[i])`. Let `P = I - J.T (J J.T)^-1 J`. The proposed operator is `d = P H P grad`, with the existing positive Sobolev inverse H. In exact arithmetic, `J d = 0` and `grad dot d = (P grad) dot H (P grad) >= 0`. This is a valid constrained SPD search operator, not the H-orthogonal projector used by a full constrained Gram solve.

`J J.T` is cyclic tridiagonal. Its actual diagonal is `2 dot(u_i,u_i)` and its off-diagonal is `-dot(u_i,u_{i+1})`, including the wrap. Unit directions are formed and normalized from vec2d differences, so coefficients, RHS, factors and corrections describe the same double Jacobian. A rank-one cyclic correction reduces each solve to two Thomas RHS in O(N). Two projection calls around the existing H convolution need two O(N) per-environment kernels, rather than an iterative O(N²) Gram solve. The construction-only workspace has five fixed arrays per E*Nmax: unit/delta vec2d, upper/rhs/rank float64.

A retraction solves `J J.T lambda = actual_edge_lengths - scale*reference_edge_lengths`, then applies `-J.T lambda`. Each bounded Newton update must reduce the squared constraint residual evaluated after vec2f storage. Final success requires the configured maximum relative residual and a nonsingular final Jacobian. No success is inferred from exhaustion or rejected updates. A failed candidate may be dirty; the reference stays unchanged. Targets preserve each reference edge ratio; they do not guarantee globally equal edges, especially after arbitrary legal refinement ratios.

## API

- `alloc_edge_constraint_workspace(E, Nmax, device, alloc=b.scratch)` is construction-only.
- Inline `project_edge_tangent_field(reference, field, e, n, workspace, out) -> bool` supports field==out after caching RHS; reference/output must be distinct.
- Inline `retract_edge_lengths(reference, candidate, e, n, target_scale: wp.float64, workspace, relative_tol, max_iters, max_backtracks) -> bool` mutates candidate scratch only.
- Host wrappers `project_edges(ctx, ...)` and `retract_edges(ctx, ...)` launch through ctx and use preallocated status/output arrays. Retraction wrapper scales are float64.

Callers must propagate projection/retraction false to blocked/invalid status. The helpers certify constraints only; energy acceptance and full swept obstacle/domain geometry remain caller responsibilities. Their tolerance is not a stationarity condition and does not remove all finite coordinate-quantization work. In particular, the existing 2e-5 edge tolerance can admit a 1e-7-relative perimeter drift, so the lead's independent f64 objective/perimeter work remains necessary. The projector validates a double residual before vec2f field storage; the stored field has ordinary O(float32 epsilon) tangent error, so final retraction and actual objective acceptance are still required.

## Validation

TDD red: five selected cases failed for the absent helper. Initial green: 21 CPU passed, 12 CUDA skipped. A requested sub-float32-scale regression then exposed the old private scale channel rounding `1+2^-24` to 1: it falsely returned success. With float64 inline/wrapper scales, the request is retained and the strict local retraction honestly returns false when its coordinate correction rounds away. Final focused helper suite: 22 CPU passed, 12 CUDA skipped. Combined constraint, barrier, tangent-energy and metric suites: **80 passed, 32 skipped**, all CUDA skips local. No full project suite was claimed; it remains with the lead after controlled integration.

Dense independent NumPy proof: `repulsive_cyclic_projection_proof.py` and `repulsive-cyclic-projection-proof.json`. N = 4…1024 and three actual N256 curves gave cyclic relative solve residual 1.7e-16..1.0e-13, tangent projection error around 1e-15 and idempotence around 4e-15. `grad dot PHPgrad` matched `Pgrad dot H Pgrad` and was positive. Tangent proposals retracted with O(step²) correction. Undamped growth Newton could initially increase residual sharply; `repulsive_damped_retraction_proof.py` demonstrates why damping and final validation are required.

## Controlled evidence and limits

Final-only edge ratios initially appeared nearly uniform, so a density-collapse claim was rejected. Internal instrumentation through the normal LaunchContext boundary, with host reads only in the scratch probe, then measured pre-growth max/min peaks for seeds 7/23/41 of [3.30,5.06,2.19] at N64, [2.87,1.48,3.40] at N128 and [2.736,2.575,3.501] at N256. Periodic/final resampling can hide these excursions. Two warm/generate passes were identical. Reproducer and JSON: `repulsive_internal_edge_ratios.py`, `repulsive-internal-edge-ratios.json` (loaded Warp module 346bf61).

Fresh-gradient single-state comparison used the actual current centerline and recomputed full TP+barrier derivative, rather than stale final scratch gradients. Reproducer `repulsive_projector_direction_comparison.py`; JSON `repulsive-projector-direction-comparison.json`; exact field/curve/metric/obstacle arrays `repulsive-projector-current-state.npz` (loaded Warp module c5e5fcb). P H P removed first-order per-edge length rates to ~1e-14, but did not remove all normal velocity: unbacktracked normal RMS in mean-edge units changed 4.26 → 4.68 for seed 23 and 2.59 → 8.66 for seed 41. Thus projection is a coherent invariant change requiring a complete controlled growth/retraction probe, not a demonstrated stand-alone jitter cure.


## Disposition

The controlled complete-pipeline experiment did not justify adoption. Its small scalar outputs are preserved as [baseline](controlled-baseline.json) and [candidate](controlled-candidate.json). Both used source SHA-256 `d405b7c216b8ab7dc34499c1f4ef03f7cebce0fdba86dc93801bbf791efd23ce`, seeds 7/23/41 and 210 iterations. Although the candidate final edge ratios stayed near one, all three rows blocked before reaching target length and none was valid. The helper was moved out of production and remains documentary only. No captured coordinate arrays are included. The artifact names above identify original local probes and are historical references, not portable archive dependencies.
