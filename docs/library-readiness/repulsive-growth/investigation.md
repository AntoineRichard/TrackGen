# Repulsive growth investigation

Investigated 2026-10-25 on macOS arm64 / CPU, Warp 1.18.0, docs head `4aec44a8f30d7ef998bb37ce6cb86e6ad2d12bd2`. Runtime source is the RC2 implementation. The original investigation changed no repository files or runtime behavior and used no GPU. This internal record, portable diagnostic scripts, two PNGs and a small hash manifest were added afterward under `docs/library-readiness/repulsive-growth/`; no runtime changes are included. Sphinx excludes `library-readiness/**` from the published site.

## Diagnosis

The user's observations are correct. Jaggedness exists in the actual solver coordinates. The displayed default loop also crosses an inner disc and the finished road overlaps four discs. The findings have distinct causes:

1. **Default optimizer/stopping weakness:** a fixed normalized step produces high-frequency zigzags and reversals; the area-only stall detector freezes a visibly rough curve. It is a real quality issue, not video compression or interpolation. The shared road tail substantially smooths the curve.
2. **Intended inner-disc deactivation plus misleading presentation:** the default explicitly disables all inner-disc forces on reaching the target; the animation keeps drawing those rings in the same green as active forces. Seed 7 penetrates only after deactivation. The discs are shaping scaffolding, not exported physical colliders.
3. **A separate real constraint limitation:** even force-active discs and the always-active wall can be crossed for accepted numerical settings. Substep measurements prove the hard perimeter rescale can undo a repulsion step. Lowering tau alone produces a visually smoother but escaped track; it is not a safe fix.
4. **Validity does not check these obstacles or the domain:** `track.valid=True` cannot establish obstacle clearance, wall containment, or optimizer convergence.

There is no evidence here of a wrong analytic-gradient sign or a rendering reimplementation bug. The existing analytic-gradient reference test passes. Default numerical stability, weak termination criteria, and soft force constraints deserve a separately reviewed runtime correction; none has been silently changed.

## Exact reproduction and observation

Run the commands below from the checkout root. The historical runtime revision is RC2 `a42759a3f999a8668a200cddc3f04e0508478dd1`; the report originally inspected docs revision `4aec44a8f30d7ef998bb37ce6cb86e6ad2d12bd2`. Source line numbers describe that original revision; later docs-only presentation edits can move renderer lines.

The tools require Python, NumPy, Warp, and Matplotlib for figures. The original observations used Warp 1.18.0 on Darwin arm64; floating-point trajectories and array hashes are platform/device specific. Use a normal project environment with its dependencies installed. No GPU is needed.

```sh
python docs/library-readiness/repulsive-growth/reproduce.py --list-cases
MPLCONFIGDIR=/tmp/trackgen-repulsive-mpl WARP_CACHE_PATH=/tmp/trackgen-repulsive-warp \
  python docs/library-readiness/repulsive-growth/reproduce.py \
  --case baseline --case tau02 --case tau01 --case keep_obstacles \
  --out-dir /tmp/trackgen-repulsive-growth
MPLCONFIGDIR=/tmp/trackgen-repulsive-mpl \
  python docs/library-readiness/repulsive-growth/figures.py \
  --input-dir /tmp/trackgen-repulsive-growth --out-dir /tmp/trackgen-repulsive-growth
```

`--case` is repeatable; additional case names reproduce the limited comparisons in the tables. `--out-dir` defaults to a task-specific folder under the platform temporary directory. Each run writes its full frozen config, runtime and docs reference revisions, current checkout revision, source-file SHA-256 hashes, real-point geometry metrics, every-iteration states and post-step/post-rescale substeps to a local JSON/NPZ pair. Generated NPZ datasets are intentionally not committed. `figures.py --verify-only` checks the source bytes and recomputes the raw/returned-road metrics and band intersections without plotting.

`array_record` hashes the original contiguous array bytes **before** float64 metric conversion and records dtype and shape. Hashes of ordinary `raw.xy`, separately observed `raw.xy`, and returned road `center`/`inner`/`outer` (only real points) distinguish generator geometry from road output. Every frame/substep also records its coordinate-array key/hash and effective force activation. The observer calls the original kernels, not a reimplemented solver, and asserts byte equality between observed and ordinary raw outputs. `reference-array-hashes.json` preserves the original source-array hashes; compare these only on the matching CPU/platform/Warp stack. An array hash changing across devices is not by itself a defect.

The actual MP4 and poster were inspected. Video is H.264, 864×744, 6 fps, 10.67 s: 52 captured states plus 12 held final frames. Manifest `animations.repulsive` specifies seed 7, final equality true, iterations 0–192. The raw output matches the selected example's count 204, valid true, fail 0. The selected gallery batch is four explicit seeds; the capture regenerates seed 7 alone.

Displayed configuration is default `TrackGenConfig(generator='repulsive', device='cpu', num_envs=1, on_invalid='keep')`, explicit Warp seed array `[7]`, epoch zero. Main defaults: scale=size=1, raw N=256, stages=(64,128,256), grow bounds 4.5–5.5, tau=.4, ratchet=.012, alpha=3, beta=6, w_len=30, settle=40, resample every25, stall every16 with area tolerance .05, deactivate inner obstacles=True; half-width=.1, road spacing=.06, N_max=384, standard XPBD tail.

Domain radius = 1.716015; initial radius = .42900375. Nine inner discs were sampled for this seed. Host iteration budget269; stage starts are zero-based iterations `[0,59,117]` (post-step snapshots at iterations60 and118 show new resolutions). Target becomes 12.51645088 at iteration129; the env freezes at192 and CPU stops. Capture code is `viz/render_generator_guide.py:678–726` at the original docs revision, with actual coordinate snapshots every four iterations and transition frames. `draw` at744 connects vertices with straight segments, closes the loop, and does no spatial or temporal spline interpolation. Sparse playback is choppy by design, but does not create the measured spatial zigzags.

## Quantitative smoothness

Vertex turn is the absolute angle between consecutive polyline edges. Curvature here means `abs(turn radians) / mean(adjacent edge lengths)` in inverse coordinate units; it is a reproducible polyline roughness diagnostic, not a claim that the polyline is a differentiable curve. All real points are used, with padding excluded.

| Seed7 case | Raw max / p95 turn | Finished-road max / p95 turn | Raw max / p95 curvature | Finished-road max / p95 curvature |
|---|---:|---:|---:|---:|
| Current default | 114.97° / 47.76° | 20.78° / 18.77° | 50.59 / 19.33 | 6.11 / 5.56 |
| Only tau=.2 | 45.66° / 21.22° | 22.96° / 19.95° | 16.30 / 7.70 | 6.77 / 5.88 |
| Only tau=.1 (**wall escape**) | 21.36° / 12.11° | 16.60° / 10.52° | 7.74 / 4.34 | 4.86 / 3.08 |
| Only retain inner forces | 85.62° / 44.54° | 30.27° / 22.11° | 30.56 / 17.30 | 9.08 / 6.55 |
| Area tol=.001, settle=200 | 168.89° / 64.49° | 22.71° / 19.65° | 103.59 / 24.13 | 6.70 / 5.85 |
| tau=.1, ratchet=.003, retain forces | 77.52° / 25.24° | 28.24° / 20.44° | 27.70 / 9.22 | 8.30 / 6.04 |

Intermediate default iteration47 reaches max turn175.43°; iteration48, which is actually displayed, is151.09° max and109.21° p95. Immediately before final raw resampling, iteration192 is139.19° max and72.00° p95. The final arc resample reduces these to114.97°/47.76° but does not smooth them away. Road tail resampling/relaxation reduces roughness further. Thus the movie exposes a rough construction state, not the smoother delivered centerline.

`_gmax_msl_k` and `_step_k` at `track_gen/_src/warp_generate_repulsive.py:546–571` normalize the maximum per-vertex displacement to tau×mean-edge-length: approximately .0196 units per iteration at the final target and N256. There is no decay or acceptance search. Over iterations177–192 (no resample or resolution transition), consecutive full displacement vectors have mean cosine −.763: substantial alternating motion persists. This is stronger evidence than visual roughness alone. The total TP+wall energy still falls over this window, so the result is not proof of increasing total energy; local zigzags can coexist with global energy decrease.

`_freeze_update_k` at602–621 checks **only** reaching target and relative area change over16 iterations. Seed7 freezes at relative area change .0447389 < .05, despite the roughness above. It checks neither gradient/shape displacement, curvature, collisions, nor road validity. Tightening area tolerance and extending the bound to429 iterations leaves the curve unfrozen and *rougher* (max169°); more iterations alone is not a cure. Regardless of freeze, generator output validity is unconditionally set1 at968.

The final raw perimeter is12.22229 versus target12.51645 (2.35% shorter), and the road center perimeter is12.08025. Arc resampling replaces jagged chains with chords; `reached=True` records the target ratchet state, not a guarantee that final resampled perimeter equals the target. This is another reason to distinguish target, freeze, and final geometry.

## Clearance and what the rings mean

Gaps are `min distance(circle center, all closed-polyline segments) - circle radius`; negative means actual circle penetration. Rings are reconstructed from their12 sampled points. They are genuine sampled disc radii, **not** a road-width-inflated exclusion radius. Road-band intersection was also checked using the exact inner/outer quadrilaterals drawn by the guide; four disc centers lie in or intersect that band at default. For a center inside a quad, band gap is recorded as −radius, an overlap indicator rather than a signed penetration-depth calculation.

| Seed7 case | Final raw min disc gap | Raw min wall clearance | Finished-road min centerline disc gap | Finished road overlaps discs | valid / fail |
|---|---:|---:|---:|---:|---|
| Current default | −.04275 | +.61537 | −.04801 | 4 / 9 | True /0 |
| Only tau=.2 | −.01114 | +.20606 | −.04441 | 3 /9 | True /0 |
| Only tau=.1 (**escaped**) | +.14330 | **−.72871** | +.14741 | 0 /9 | True /0 |
| Only retain inner forces | +.20220 | +.44591 | +.16507 | 0 /9 | True /0 |
| tau=.1, ratchet=.003, retain forces | +.19808 | +.43520 | +.16055 | 0 /9 | True /0 |

Wall clearance means domain radius minus maximum centerline vertex radius; since the disc domain is convex, this also tests all centerline segments. It does not include road half-width. For the default and retain-only cases the full road band also stays inside the domain by a substantial margin. The tau=.1 road center is .72990 outside the domain after the tail; this is a persistent escaped output, not a transient movie artifact.

Default seed7 remains clear of all discs before reaching target. `_ratchet_k` is executed before `_grad_gather_k` at897–913; when `reached` flips true at iteration129, inner forces are already omitted in that same iteration. First penetration occurs147, persists through freeze192 and final resampling. The hit disc has center(−.222180,.611365), radius .0761104; it is the ring intersected by the upper internal fold in the poster. At iterations158–160 all vertices are outside the disc but connecting segments still penetrate: node-only checks would miss those collisions. Those iterations are after deactivation, so this does not establish a failure of active repulsion in the baseline.

Two additional default seeds confirm sensitivity without a broad sweep: seed23 raw min disc gap+.03284 but finished band overlaps4 discs; seed41 raw gap−.03608 and finished band overlaps3 discs. Their raw max turns100.26° and91.02°, versus finished-road22.37° and24.93°. All three selected seeds pass road validity; this small selected set is not a yield estimate.

## Proven active-force leak, attributed to a substep

A single tau change to.1 (still positive and accepted) is sufficient to expose the limitation. It produces smoother geometry, but the smaller normalized repulsion movement cannot keep up with the fixed ratcheted length expansion:

| First event, seed7 tau=.1 | Gap after original gradient step | Gap after original hard rescale | reached |
|---|---:|---:|---|
| Active inner disc, iteration71 | +.00422852 | **−.00526623** | False |
| Active wall, iteration99 | +.01764391 | **−.00224552** | False |

The original `_rescale_k` at590–598 multiplies all offsets from the barycenter by target/current perimeter with no obstacle or wall feasibility check. It undoes the avoidance gained in `_step_k`. The wall gap then worsens; by129 it is−.73757. Retaining inner-disc forces does not prevent this escape. Slowing the ratchet to.003 while using tau=.1 and retaining forces avoids these penetrations for this one seed, but still has raw77.52° max turn and costs528 iterations. This is a targeted hypothesis check of the growth/step balance, **not** a recommended safe preset or evidence of a general fix.

The obstacle energy at253 and438–455 repels each curve **vertex** from weighted fixed **sample points**:96 wall samples,12 points per inner ring, inverse power beta−alpha (=3 at defaults). It has no exact signed distance to a filled circle, continuous curve/disc barrier, swept collision test, finite road width, or post-rescale projection. A finite set of singular point potentials does not establish a continuous circular barrier. There is also no obstacle-aware handling in final arc resampling or road relaxation. These are genuine algorithmic limitations, even though inner discs are documented as scaffolding.

## Final validity contract

`Repulsive` publishes only raw.xy and gen.valid (`warp_generate_repulsive.py:986–987`); it unconditionally writes gen.valid=1 at968. It does not report failure to reach target, failure to freeze, disc penetration, or wall escape.

`ValidityDefault.reads` at `track_gen/stages/track.py:630–634` has no obstacle/domain channels. `_validity_k` in `track_gen/_src/warp_pipeline.py:174–285` checks generator flag, turning number, finite output, width floor, self-thickness, optional border self-intersection, grade and truncation. Its “border” flag is road-border **self-intersection**, not the growth-domain wall. Fail-bit names at `track_gen/stages/channels.py:69–72` likewise have no external-obstacle or domain check. This explains true/0 for both a disc-overlapping default track and an escaped tau=.1 track. Existing docs at `docs/generators/repulsive.rst:104–106` correctly call obstacles scaffolding/not exported colliders, but the animation's equally green rings encourage the opposite interpretation.

## Exact animation overlay fields for a docs-only correction

Do not mutate runtime or scratch weights to change presentation.

- Private source constants: `_N_WALL=96`, `_N_DISC=12`; `_RepulsiveScalars` also exposes n_wall, n_disc, M, r_dom, r_init at96–117. Default M=96+12×12=240 (unused disc slots included).
- For env e in E envs, scratch `obs_pts` and `obs_mw` have stride M. Wall is `[e*M:e*M+96]`. Inner disc j is `[e*M+96+j*12:e*M+96+(j+1)*12]`.
- A sampled inner disc exists when its block's weights are nonzero; unused blocks have zero weights and NaN points. The first96 wall samples always exist and remain force-active.
- Inner force is effectively active on a snapshot iff it exists AND `not(config.repulsive_deactivate_obstacles and frame.reached)`. **obs_mw is not zeroed on deactivation**: the gradient condition at450 skips the term. Weight alone cannot detect effective force activation.
- `capture_repulsive` already captures reached/frozen at711–713. Derive the display state from reached+config, or record an explicit `inner_forces_active` field for clarity.
- Renderer at `viz/render_generator_guide.py:732–733` draws obs_pts once in one green color; `draw` at743–747 never updates those colors. Split the scatter into wall and inner sets; retain green wall, fade/grey the inner points and explicitly label “inner shaping forces disabled” after reached. Explain that force-active still means a soft sampled potential, not a collision guarantee.
- Keep the default seed7 curve and its roughness/crossing visible. Show the delivered road separately, and label any tuning comparison with measured wall/disc outcomes. Do not switch the default clip to tau=.1 or select a nicer seed.

A truthful footer could say: “Recorded optimizer states. Inner rings are shaping scaffolding; grey rings have inactive forces. Wall and disc forces are soft. Final road validity does not test these rings or the growth domain.” The larger explanation belongs beside the animation rather than all in its footer.

## Parameter scaling and additional contract hazards

Source-based analysis, not a claim that these caused the default-seed symptom:

- Step magnitude is tied to tau×perimeter/N, while target growth is a separate per-iteration factor. Therefore tau and ratchet rate cannot be tuned independently without checking feasibility. More iterations with a fixed normalized step need not reduce roughness.
- Coarse-to-fine resamples and every25-step reparameterizations reduce jaggedness abruptly; they are actual discrete geometry operations. The schedule itself explains some jumps in the movie. Four-iteration snapshots at6fps add temporal discontinuities; interpolating them would conceal solver behavior and cannot cure it.
- E_TP uses curve dual lengths w_i w_j; obstacle energy sums vertices without a curve dual-length factor. Refining N changes the relative obstacle weighting as well as resolution. With coordinate scaling s and ignoring eps, TP energy scales as s^(alpha−beta+2), obstacle ring energy as s^(alpha−beta+1). Relative obstacle strength therefore changes as1/s; `scale` changes the physical system, while dynamic `size` is applied only after growth. The one scale=2 check had normalized pointwise displacement .674 from baseline, and default N_max then truncated the road (fail128), consistent with this being more than a finished-curve multiplier.
- Validation at `track_gen/_src/builtin_params.py:657–660` only requires positive alpha and beta. beta=alpha makes p_exp=0 and obstacle forces vanish; beta<alpha reverses the obstacle-force sign. These accepted regimes should not be described unconditionally as obstacle repulsion. This is an independent static-contract concern requiring review, not a tested default regression.

## Recommended next actions and output impact

1. **Immediate docs-only correction:** expose active vs disabled rings, label them as soft shaping forces, state explicitly that final validity excludes discs/domain, explain area freeze versus smooth convergence, and retain the default example. This preserves runtime outputs and API.
2. **Separately reviewed solver-quality change:** investigate an accepted-step/step-decay rule and a convergence condition that observes shape movement or gradients, so a nearly unchanged area cannot freeze a zigzagging state. Address hard perimeter rescale feasibility and growth/repulsion balance together; merely lowering tau or adding settle iterations is not a fix. These changes alter seeded outputs, convergence and possibly runtime/capture cost, and require new CPU/CUDA evidence and downstream review.
3. **Clarify desired obstacle contract:** if discs remain scaffolding, no physical-clearance guarantee should be implied. If users need colliders/containment, add explicit circle/segment and finite-width clearance, preserve/check it after resampling and road tail, and report failures. Passing/exporting those primitives into the pipeline is an API/behavior design change. Retain-only cleared the displayed seed, but is not a proof or general safeguard.
4. Review exponent admissibility and resolution/scale weighting as part of that runtime work. Do not silently change RC2 defaults or golden outputs while completing the existing docs review.

## Artifacts and validation

All committed artifacts are in this directory:

- `default-diagnostics.png`: actual default coordinates at displayed iterations48/192, measured roughness and disc-gap histories.
- `tuning-comparison.png`: default plus same-seed one-control changes, including the entire smoother-but-escaped tau=.1 loop; common uncropped viewport; active/disabled disc styling; finished roads.
- `reproduce.py`: portable CLI capturing unchanged CPU solver launches and returned road arrays; configs and source-array hashes accompany measurements.
- `metrics.py`: segment/circle clearance, node clearance, wall containment, real-point turning/curvature, exact drawn-quad overlap indicators, and source TP+wall energy evaluation.
- `figures.py`: validates hashes and recomputed metrics, then draws the two diagnostic comparisons.
- `reference-array-hashes.json`: small historical byte-hash manifest for the source arrays behind this report, including exact revisions and source-file hashes. No trajectory datasets are committed.

Original focused checks: `test_analytic_gradient_matches_reference` and `test_the_schedule_at_the_defaults`: **2 passed, 1 skipped** (CUDA unavailable locally),4.41s. The original investigation left `git status --short` and `git diff --stat` empty. For preserving these records, the scripts were syntax-checked and their argument parsing, geometry calculations, hashing, data loading and plotting pipeline were smoke-checked without a generator sweep or GPU run.

A synthetic geometry smoke verifies that a segment crossing a circle is detected even when all nodes clear it, and that road quads exclude their central hole. The portable metric/figure pipeline was also exercised against the original local source arrays, with source hashes checked. The portable observer itself was subsequently checked once on the exact seed7 baseline; its raw/returned-road byte hashes and all four historical source-file hashes matched the original manifest. No broad generator experiments or GPU validation were repeated during preservation.

These are CPU findings for exact displayed output plus three-seed spot checks, not a device-wide yield study or a universal parameter recommendation. GPU validation was left undisturbed.
