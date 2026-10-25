# Geometric certificate arithmetic

Root self-review found an additional precision gap in the new hard-clearance
helper. At larger world coordinates, reconstructing a closest point in float32
can erase the much smaller offset to a disc. This is separate from the folded
cell coverage gap and from energy/projection precision.

A deterministic fixture uses already-stored float32 values:

- Segment a = (423.90789794921875, 856.0101928710938).
- Segment b = (435.1707763671875, 856.9830322265625).
- Disc center = (428.0730895996094, 856.3699951171875).
- Radius = 0.00005065876757726073.

Independent float64 geometry gives distance 0.00003065876708829704, so the segment
penetrates the disc by approximately 0.00002. The previous float32 predicate
reported clear. The same fixture embedded in a 0.1-wide cell and its swept hull
also reported clear. A separate near-wall fixture lost the conservative wall
margin through float32 arithmetic. Both regressions were observed failing before
the repair.

Stored coordinates, radii and metadata transforms keep their existing float32
representation. The certificate now casts before subtraction and retains the
projected offset relative to the disc center. Segment distance, triangle
orientation/containment and wall squared-distance calculations use float64. The
clearance tolerance is still positive and conservative; no penetration allowance
was introduced. The reviewed helper source SHA-256 is
`3cd7a53633f3d0a6741275d22d2a3c00165627f6abe552609e39787020b8b00b`.

A deterministic 20,000-case CPU probe at each coordinate scale recorded:

| Coordinate scale | False-clear cases before | After |
| --- | ---: | ---: |
| 100 | 0 | 0 |
| 1,000 | 8 | 0 |
| 10,000 | 286 | 0 |
| 1,000,000 | 259 | 0 |

These are empirical cases, not a proof over all floating-point inputs. The root
rerun overwrote its scratch summary with the after results; the before counts
and exact fixture above are retained from the original tool output and RED run.
No scratch dataset is committed. The portable regressions are
`test_large_coordinate_small_disc_predicates_preserve_sub_ulp_clearance` and
`test_wall_certificate_keeps_its_margin_at_large_coordinates` in
[the final-clearance tests](../../../../tests/test_repulsive_final_clearance.py).
Together with the existing hole, fold, size and boundary fixtures, the focused
file passed eight CPU cases with eight CUDA cases skipped locally. Five targeted
solver integration cases passed after the helper change. Exact committed-source
GPU and release verification remained pending at this record.

## Raw-domain wall integration

The final certificate repair exposed a separate raw-helper integration issue.
At domain radius 1,000,000 and zero reserved width, the finite float32 point
(471861.03125, -881672.9375) has true float64 norm
1,000,000.0007660962, outside the wall. Float32 length instead reports
999999.9375, and the original `_proposal_clear` returned clear. The actual core
helper falsely cleared all 1,584 selected candidates in a one-million-point
sample at that radius. No candidates were found by that screen at radii 100,
1,000 or 10,000; this is not a proof over those radii.

The raw helper now uses the same radius-relative conservative tolerance, a
finite positive float64 effective limit and a float64 squared norm. The
`test_core_wall_certificate_uses_double_geometry_and_scaled_tolerance` cases
in the solver-clearance tests cover the outside point, tolerance band, valid
interior, negative limit and nonfinite limits. Four CPU cases failed before
the fix; six passed afterward, with six CUDA cases skipped locally. Root's
independent rerun retained the same 1,584 selected points and found zero actual
core false clears. The original before summary was retained separately.

The solver source SHA-256 after this narrow integration repair is
`499908359030b1e02abe10e2d81244d19f01e12e792f9c46b452750eaac603ad`.
The held solver and geometry helper passed the combined 120-test CPU focused
suite. Versioned three-seed and sixteen-seed proofs were refreshed with all six
mathematical source hashes checked before and after execution. The coordinated
GPU and full-suite gates remain separate from this CPU evidence.
