# Finished road-band certification

An adversarial CPU review found a folded-cell gap in the **new** final-band
certifier during the post-rc2 repair. This was a constructed built-in offset
fixture, not a complete Repulsive seed reproduction or a claim that the old rc2
certifier emitted it. No source hash was recorded by this geometry review.

## Physical occupancy counterexample

The initial certifier used even/odd interior of a cell's four ordered boundary
edges. That is sound for a simple polygon interior, including concave polygons.
It does not cover every lobe of a folded bilinear ruled strip or an ordinary
triangle strip: winding can cancel an occupied lobe.

The scratch-only fixture had an 18-point simple centerline and constant
half-width 3.47804546. Built-in frame/offset kernels created the cell; generic
validity passed (`valid=1`, `fail=0`). Its minimum curvature radius 3.42664623
passed the normal 2% thickness threshold `(1 - .02) * width = 3.40848455`.
Border checking was off, as at the default.

In a radius-30 domain, place a radius-0.00412013894 disc at
`(-6.5819940567, 8.0151453018)`, outside the global centerline loop. The actual
Warp cell vertices were:

```text
a = (-6.5786151886, 7.9747247696)
b = (-6.7078304291, 8.2097263336)
c = (-4.8660326004, 1.5018973351)
d = (-0.5882952213, 4.4387283325)
```

All ordered boundary edges clear the disc, and the initial certifier also
passed it. Yet with `t = .4125`, `s = .0116666667`, the ruled-strip point

```text
(1-t) * ((1-s)*a + s*d) + t * ((1-s)*b + s*c)
```

lies within 1.7e−7 of its center, well inside the disc. The usual BD triangle
strip also contains the center. This is not the permitted central-loop hole.
Raw width reservation cannot exclude every tail-generated fold, and small
positive disc radii are supported; the fixture did not use default radius bounds.
Boundary-winding plots can hide the same lobe and are not occupancy proofs.

## Conservative repair and recorded checks

The final-cell predicate now reuses the complete four-point swept/convex-hull
certificate: all six chords and four triangles cover the convex hull, including
degenerate cases. The ruled strip and either triangulation lie inside that hull.
A disc in an unused concave-hull pocket may therefore be conservatively rejected;
a disc outside the hull remains allowed. Checking cells separately preserves the
global central loop hole. For the outer wall, a circular domain is convex, so
containing every band vertex contains every cell hull.

The new regression returned `[1,1,1]` before the fix versus expected `[0,0,1]`
for the real folded cell, a concave-hull pocket and an outside-hull no-hit control.
After replacing the ray rule and removing its unused helper, focused geometry and
pipeline checks recorded **17 CPU passed, 6 CUDA skipped**. Replaying the built-in
fixture left generic validity at `(1,0)` but final validity became `(0,256)`.
No full generator-seed or GPU proof was claimed.

The original full coordinate fixture, search scripts and before/after records
were scratch-only and are not committed. The portable cell and ruled-point
counterexample is now in
[tests/test_repulsive_final_clearance.py](../../../../tests/test_repulsive_final_clearance.py),
alongside crossing/enclosure/tangency, loop-hole, width, wall and size checks.
[Pipeline tests](../../../../tests/test_repulsive_clearance_pipeline.py) enforce
private barrier metadata, the mandatory final check, no later geometry writes,
checked-band lifting and poison/keep/mask behavior.
[Solver tests](../../../../tests/test_repulsive_solver_clearance.py) also verify
that raw success under explicit high growth cannot bypass a finished-band failure.
These repository fixtures replace machine-specific scratch dependencies; their
CUDA-marked counterparts still require the coordinated GPU lane.
