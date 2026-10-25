# Upstream provenance and port differences

Source audit dated 2026-10-25; read-only inspection, with no upstream execution or
source copied into TrackGen. The primary repository is
[LasseHenrich/racetrack-generation](https://github.com/LasseHenrich/racetrack-generation),
whose README identifies a prototype and the paper *Generating Race Tracks With
Repulsive Curves* ([IEEE 10645670](https://ieeexplore.ieee.org/document/10645670)).
The audited upstream tree is pinned to
[`2f0b796c1550e365893ac73104be94696493b3df`](https://github.com/LasseHenrich/racetrack-generation/tree/2f0b796c1550e365893ac73104be94696493b3df).
At that snapshot GitHub metadata reported no license and the tree contained no
LICENSE-named file. This records provenance, not permission to copy the source.

The local [July spike README](../../../superpowers/spikes/2026-07-05-repulsive-growth-phase1/README.md)
records comparison with `EnergyCurve.cs` and `MapGenToolBase.cs`, and Torch → Warp
porting. The [July design](../../../superpowers/specs/2026-07-05-repulsive-generator-design.md)
names `grow_warp.py` as its evidence base. The port chain is Torch spike
`283751c1d73d4c1884bc6d25693c895df9f102fa`, Warp port `1f0cfdf`, coarse-to-fine
stall pass `74d0925`, production `6b4835d`, then analytic adjoints `1d8604f`.
It does not establish which upstream commit was consulted in July.

The audit inspected documentation checkout HEAD
`1aefa4ea3e8b5d29ddb4b9db5e84b2f41f5d9db2`: production Repulsive, both growth
spikes and the Torch oracle relaxation source had no diff from `v0.3.0rc2`.
Thus the following port comparison applies to rc2, not only to an older spike.

| Aspect | Audited Unity prototype | Torch/Warp path through rc2 |
| --- | --- | --- |
| TP quadrature | Vertex pairs and dual masses; tangent averages two unit edge tangents; adjacent nonself pairs included. | Vertex pairs/dual masses, chord tangent, circular-distance ≤ 2 exclusion, numerator/distance regularization. |
| Metric | Geometry-dependent finite-element Gram and constrained saddle solve; `s = (beta - 1) / alpha = 5/3`. | Fixed ring-Laplacian inverse filter with half order `s = (beta - 1) / (2 alpha) = 5/6`. This originated in the Torch spike, before analytic adjoints. |
| Acceptance | Adaptive Armijo-style energy search and halving. | Fixed normalized motion capped by `tau * mean_edge_length`; no energy acceptance search. |
| Growth | Fixed increments to each edge target, constraint projection and Newton backprojection with halving. | Multiplicative total-perimeter ratchet, metric length-direction projection and uniform rescaling. |
| Refinement | Actual mean edge length triggers subdivision. | Shared host schedule plus periodic arc-length resampling. |
| Stop | Three consecutive bad steps after target; the apparent norm-zero stop is effectively disabled. | Target plus small enclosed-area change in a capped budget. |
| Obstacles | Sampled inverse-power rings; target attainment may deactivate the whole list, including wall. | Weighted rings; optional inner-disc deactivation, live wall. Neither gives hard full-road clearance. |

Pinned source anchors: [vertex tangents/masses](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/CurveGen/Curve.cs#L96-L120),
[default vertex BVH](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/CurveGen/TPEBVH.cs#L11-L39),
[Sobolev Gram/order](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/CurveGen/SoboSlobo.cs#L39-L138),
[flow/deactivation](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/CurveGen/EnergyCurve.cs#L336-L429),
[line search](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/CurveGen/EnergyCurve.cs#L479-L538),
[backprojection/growth](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/CurveGen/EnergyCurve.cs#L653-L740),
[edge constraints](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/CurveGen/Constraints.cs#L192-L249),
and [stopping/refinement](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/MapGenToolBase.cs#L695-L739).

Two apparent source inconsistencies preclude using the prototype as a numerical
oracle: [TPE.cs assigns both pair lengths from the owner edge](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/CurveGen/TPE.cs#L296-L301),
while its differentiated pair uses both edges; [Obstacle.cs weights force by node
mass but omits that mass from energy](https://github.com/LasseHenrich/racetrack-generation/blob/2f0b796c1550e365893ac73104be94696493b3df/Assets/Scripts/CurveGen/Obstacle.cs#L92-L140).
These are source findings, not experimentally confirmed prototype failures.

The repair uses independently checked ordered disjoint-edge endpoint quadrature:

```text
T_I = (b-a)/|I|,  I=(a,b), J=(c,d)
E_IJ = |I|*|J|/4 * sum_(u in I, v in J)
       |(v-u) cross T_I|^alpha / (|v-u|^2 + eps^2)^(beta/2)
```

Full derivatives include both edge lengths and the owner tangent. Supported
exponents satisfy `alpha > 1`, `alpha + 2 <= beta < 2*alpha + 1`. The repair uses
order `s = (beta - 1) / alpha`, bounded accepted steps and permanent analytic barriers. It retains a fixed surrogate metric and total
length constraint; it does not claim to reproduce Unity's finite-element solver.
The portable checks are [tangent energy](../../../../tests/test_repulsive_tangent_energy.py),
[metric precision](../../../../tests/test_repulsive_metric_precision.py) and
[solver clearance/stopping](../../../../tests/test_repulsive_solver_clearance.py).
Source comparison alone cannot attribute observed roughness to one divergence.
