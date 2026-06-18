# Relaxation Quality — TP-Sobolev Untangle Hybrid + Anti-Creep Anchor — Design

**Date:** 2026-06-18
**Status:** Proposed (pre-implementation)
**Builds on:** the merged pure-Warp pipeline (`track_gen/warp_pipeline.py`, `warp_relax.py`) and the
existing torch relaxation backends (`track_gen/relaxation.py`: XPBD / energy / **tp_sobolev**, plus the
`smooth_finish` finisher).
**Evidence base:** the E=8192 yield study (`benchmarks/benchmark_yield_sweep.py`, report
`viz/out/track_gen_report.pdf`) + feedback from Miles Macklin on the spike figures.

---

## 1. Goal & context

The default relaxation is a quasistatic XPBD repulsive solve (fused Warp `warp_relax.xpbd_solve`:
separation + spacing + bending). The E=8192 study established two limitations of the current default,
both confirmed by experiment:

1. **Topological failures it cannot fix.** The residual invalid tracks (≈2% thin-band, ≈32% in the
   1 m/20 m fat-band regime) are *self-crossed / looped* centerlines. Pure point-based repulsion is
   local — in a crossed configuration the push direction can drive overlapping strands *deeper* into
   each other rather than untangling them. The study showed this is a hard ceiling: more `relax_iters`
   plateaus (0.52→0.83 over 50→600) and larger XPBD steps *diverge* (Jacobi is unstable above
   relaxation factor 1: yield 0.68→0.00 over step ×1→×2.5). Miles' suggestion: resolve loops with a
   topology-aware energy. The standard tool — already implemented here as the `tp_sobolev` backend
   (tangent-point / fractional-Sobolev "Repulsive Curves") — is globally self-avoiding and untangles.

2. **Over-deformation of the easy tracks.** With `relax_margin=0.15` the solve over-inflates 15% past
   the validity target *everywhere*, and over many iters beads drift far from the generated shape
   (the wiggly inner borders visible at 600 iters). There is no regularization keeping beads near
   where they started; the energy backend has an anchor term (`energy_w_anchor`), XPBD does not.

This spec adds two **opt-in** relaxation enhancements. Both default off, so the verified pure-Warp
pipeline and its single-CUDA-graph path are unchanged unless explicitly enabled.

## 2. Decisions (locked)

| Decision | Choice |
|---|---|
| TP untangle solver | Reuse the existing **full** `relaxation._tp_flow(early_stop=True)` (tangent-point/Sobolev self-avoiding solve), NOT the light `smooth_finish` smoother. |
| TP scope | A toggle `relax_tp_finish ∈ {"off","all","failures"}`. `"failures"` runs the solver only on envs failing post-XPBD validity; `"all"` on the whole batch. Cost is ~equal in batched mode; under `early_stop`, `"all"` auto-freezes already-valid envs, so the toggle's main effect is *guaranteeing* valid tracks are untouched in `"failures"`. |
| TP wiring | `generate_tracks_warp` orchestrates: XPBD(Warp) → detect failure mask → `_tp_flow` finisher → re-resample → inflate. NOT folded into `relaxation.relax` (that module has no notion of post-relax validity). |
| Purity trade-off | The TP finisher is torch (autograd + FFT preconditioning). When `relax_tp_finish != "off"` the pipeline is no longer pure-Warp / graph-capturable — an explicit, documented quality opt-in. Default `"off"` keeps the pipeline pure-Warp. |
| Anchor | Add `relax_anchor` (float, default 0.0). Per XPBD sweep, pull each bead toward its pre-relax position `x0` by `relax_anchor*(x0 − x)`. Mirror in BOTH the Warp kernel and the torch `_relax_xpbd` to preserve the verified Warp==torch parity. |
| Anchor default | Ship default 0.0 (current behavior); sweep `relax_anchor` at E=8192 and *recommend* a value (do not silently change the default in this work). |
| Scope | One combined spec; the implementation plan groups the work (anchor / TP-finish / study-update). |

## 3. Architecture

### 3.1 Config additions (`track_gen/types.py`)
- `relax_tp_finish: str = "off"`  — one of `{"off", "all", "failures"}`.
- `relax_anchor: float = 0.0`     — XPBD position-anchor weight (per-sweep pull toward `x0`).
- Reuse existing `tp_iters` (100), `tp_tau` (0.7), `tp_alpha` (2.0), `tp_beta` (4.5) for the finisher.

### 3.2 Feature A — anti-creep anchor (pure-Warp, no torch reintroduction)
- **`warp_relax._disp_kernel`**: add args `x0: wp.array(dtype=wp.vec2f)` and `anchor: wp.float32`.
  After the sep/spc/bend terms: `out[t] = sr*sep + pr*spc + bscale*toward + anchor*(x0[t] - center[t])`.
- **`warp_relax.xpbd_solve`**: read `anchor = float(config.relax_anchor)`; wrap the *original* `center0`
  (a frozen read-only copy, NOT the mutated working buffer `cb`) as `x0` and pass it + `anchor` to every
  `_disp_kernel` launch. (`_sep_kernel` is unaffected.)
- **`relaxation._relax_xpbd`** (torch oracle path): add the same term to the per-sweep displacement
  (`disp = disp + anchor*(x0 - center)`, `x0 = center0`) so the torch and Warp solves remain numerically
  equivalent (guards the `test_warp_relax` parity assertion).
- Default `relax_anchor=0.0` ⇒ both paths bit-identical to today.

### 3.3 Feature B — TP-Sobolev untangle hybrid (opt-in, torch)
- **`relaxation._tp_flow`**: add `initial_active: torch.Tensor | None = None`. When given, the per-env
  `active` mask starts as `initial_active` (instead of all-True); the existing `early_stop`
  deactivation (`active &= th < target`) still applies. Behavior for `initial_active=None` is unchanged.
- **`generate_tracks_warp`** (`track_gen/warp_pipeline.py`): replace the current
  `assert not config.smooth_finish` region with:
  1. XPBD(Warp) `xpbd_solve` → `resample_uniform` → `relaxed` (as today, now with the anchor).
  2. If `config.relax_tp_finish != "off"` (lazy `from . import relaxation`):
     - `band = round(2*hw / mean_seg_len(relaxed)).clamp_min(1)` (reuse `_mean_seg_len_torch`).
     - failure mask: `"all"` → all-True; `"failures"` → `thickness(relaxed, band) < (1-relax_tol)*hw`
       OR `self_intersections(offset borders) > 0` (use the existing Warp validity primitives).
     - `relaxed = relaxation._tp_flow(relaxed, band, config, n_steps=tp_iters, tau=tp_tau,`
       `early_stop=True, initial_active=mask)` then `resample_uniform(relaxed, N)`.
  3. `inflate_warp(relaxed, config, valid=gen_valid)`.
- **`generate_tracks_warp_graph`**: assert `config.relax_tp_finish == "off"` (graph capture requires the
  pure-Warp path; the torch TP finisher cannot be captured). Keep the existing `smooth_finish`/solver
  guards.

### 3.4 Module boundaries
The anchor stays inside the pure-Warp `warp_relax` + its torch oracle. The TP hybrid is the *only* place
the runtime touches `relaxation.py` (torch), and only when opted in. `relaxation._tp_flow` is the
validated self-avoiding solver; we extend it minimally (one optional arg).

## 4. Verification strategy
- **Parity:** `test_warp_relax` (Warp XPBD == torch XPBD) must stay green with `relax_anchor` both 0 and
  >0 (add an anchored case asserting Warp==torch to ~the existing tolerance).
- **Anchor effect:** new test — for a batch, `relax_anchor>0` yields strictly smaller mean
  `‖relaxed − x0‖` than `relax_anchor=0`, while validity is not catastrophically reduced.
- **TP untangle:** new test — a fixed *looped* fixture (a centerline XPBD leaves self-intersecting /
  invalid) becomes valid (or its self-intersection count drops to 0) with `relax_tp_finish="failures"`;
  `"off"` leaves it invalid. Run cpu (the tp path is torch, CPU-testable).
- **Study (E=8192):** extend `benchmark_yield_sweep.py` with `relax_tp_finish` and `relax_anchor`
  configs; re-run; confirm the TP hybrid lifts the looped-failure yield ceiling (vs the iters plateau)
  and sweep `relax_anchor` to recommend a default (deformation ↓ at ≈equal yield). Add TP-hybrid and
  anchor pages to `viz/make_report.py`.
- **Regression:** full suite green; default-config behavior unchanged.

## 5. Risks & open questions
- **TP finisher cost.** Tangent-point energy is O(N²) with autograd + FFT preconditioning — far slower
  than XPBD. Report its E=8192 wall-clock; it is an opt-in quality mode, not the default.
- **`"all"` ≈ `"failures"` under early_stop.** Valid envs deactivate immediately, so the two scopes are
  near-equivalent for the full untangler; the toggle mainly guarantees valid tracks are left untouched.
  Documented, not a defect.
- **Anchor vs yield.** Too-large `relax_anchor` resists the separation push and can lower yield; the
  sweep finds the balance. The recommended default must not degrade baseline yield meaningfully.
- **Does TP actually untangle these loops?** Expected from theory + the existing `tp_sobolev` backend,
  but must be demonstrated on real failures in the study (primary success metric).
- **`_tp_flow` band / resample.** The finisher needs a per-track band and a re-resample to N afterward;
  ensure consistency with the XPBD path.

## 6. Out of scope
- Porting the TP-Sobolev solver to Warp (it stays torch; graph capture is disabled when it is on).
- Changing the default pipeline behavior (both features default off).
- A bespoke winding-number energy term (the tangent-point solver is the chosen untangler; a winding-
  number force is noted as a future alternative, not built here).
- Selective *sub-batch gathering* of failures for speed (batched masking is sufficient; the user noted
  it makes no wall-clock difference in batched mode).

## 7. Definition of done
- `relax_tp_finish` and `relax_anchor` config knobs exist; defaults (`"off"`, `0.0`) leave the pipeline
  bit-identical to today and still graph-capturable.
- The anchor reduces mean deformation (`‖relaxed − x0‖`) and Warp==torch parity holds (tests).
- The TP hybrid untangles looped failures XPBD leaves invalid, demonstrably lifting the yield ceiling on
  the failing set (test + study).
- `generate_tracks_warp_graph` rejects `relax_tp_finish != "off"`.
- Full suite green; `benchmark_yield_sweep.py` + `make_report.py` updated; a recommended `relax_anchor`
  default is reported (default unchanged in code).
