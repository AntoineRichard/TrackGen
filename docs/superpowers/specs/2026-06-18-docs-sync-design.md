# Docs ↔ Code Sync (constant-spacing) — Design

**Goal:** Bring the documentation in line with the code after the constant-spacing work, which (a) added `output_mode="constant_spacing"` and per-track count-awareness across the pure-Warp pipeline, (b) corrected the cause of the fat-band yield ceiling (slow-Jacobi under-convergence, not un-relaxable geometry), and (c) removed the `relax_anchor` / `relax_tp_finish` experiment.

**Non-goals:** No code behavior changes. No rewrite of the historical specs/plans beyond a supersede banner (they are point-in-time records). No new docs files beyond this spec.

---

## Background (what changed in the code)

- `generate_tracks_warp` / `inflate_warp` now support `output_mode="constant_spacing"`: each track is relaxed and inflated at a per-track point count `count[e] = round(perimeter[e] / spacing)` (constant arc spacing, `spacing` ≈ 0.6×half_width), stored in `[E, N_max, 2]` NaN-padded buffers. Every post-generation Warp stage was made **count-aware** (loop `range(count[e])`, base `e*n_max`, wrap `%count[e]`, guard `i≥count[e]`), with the invariant that `count[e]==N_max` reproduces the fixed path bit-identically. It is still CUDA-graph-capturable.
- Measured win (E=8192, 1 m-track / 20 m-box): yield **0.684 → 0.999**, smoother tracks, and faster (~0.56 s vs ~0.80 s).
- The `relax_anchor` + `relax_tp_finish` features were implemented then **removed** (anchor lowered yield with no drift benefit; the TP-Sobolev finisher circularized tangled tracks). They are preserved on git branch `anchor-tp-finisher`.

The living docs (`README.md`, `docs/ARCHITECTURE.md`) describe only fixed-256 mode and still assert the old "un-relaxable" yield story; the removed knobs survive only in their historical spec+plan; one in-code comment (`inflation.py`) is now misleading.

---

## Scope — concrete changes

### Bucket 1 — Living docs

**`README.md`**
1. Add a short **"Output modes"** note (near Quickstart / the `Track` table): `fixed` (default — every track `num_points` points) vs **`constant_spacing`** (per-track `count = round(perimeter/spacing)`, `spacing` ≈ 0.6×half_width; converges to smoother, higher *honest*-yield tracks in tight-width regimes; configured via `output_mode`, `spacing`, `N_max`).
2. `count` table row → "real point count (`== N` in fixed mode; per-track `round(perimeter/spacing)` in constant_spacing)".
3. Conventions blurb (the "env index = tid // N" lines) → add that post-generation stages are **count-aware**: flat `[E, N_max, 2]` buffers with a per-track `count[e]`; fixed mode is the `count==N_max` special case.

**`docs/ARCHITECTURE.md`**
4. **Conventions §** → add the count-aware convention: `n_max` is the buffer stride, `count[e]` the per-track real-point count; padding slots `i≥count[e]` hold `wp.nan`; **`count[e]==N_max` ⇒ bit-identical to the fixed-N kernel** (the parity invariant that protects fixed mode and the existing tests).
5. **Resample §** → add `resample_constant_spacing` (fixed source → per-track `count=round(perimeter/spacing)`, NaN-padded to `N_max`, oracle-matched). **Relax/Inflate/validity §** → note these stages operate over `count[e]` real points.
6. Add a short **"Output modes / constant spacing"** subsection: the convergence rationale — a fixed 256 over-resolves the centerline relative to its half-width, so the slow Jacobi XPBD solve under-converges → jagged tracks whose 1 m road self-overlaps; relaxing at constant ~0.6×half_width spacing lets it converge → smooth, valid tracks.
7. **Correct the "Determinism, yield, FP tolerance" yield bullet** (the key factual fix): the fixed-mode residual loss — and the ~0.68 fat-band yield — is largely **slow-Jacobi under-convergence from over-resolution**, *not* genuinely un-relaxable geometry; `output_mode="constant_spacing"` lifts E=8192 yield **0.684 → 0.999**, produces smoother tracks, runs faster, and remains graph-capturable. Keep the FP-tolerance / determinism paragraphs as-is.
8. **Graph-capture §** → note constant_spacing also captures (the per-track `count[e]` is device-side data; all launch dims stay static via `N_max`).

### Bucket 2 — Supersede banners

9. Prepend a blockquote banner to BOTH `docs/superpowers/specs/2026-06-18-relaxation-quality-design.md` and `docs/superpowers/plans/2026-06-18-relaxation-quality.md`:

> **⚠️ Superseded — not in the live code.** The `relax_anchor` and `relax_tp_finish` features described below were implemented then **removed**: the anchor lowered yield with no deformation benefit, and the TP-Sobolev finisher circularized tangled tracks (it chases the validity gate on jagged input). The real root cause — fixed-resolution / slow-Jacobi **under-convergence** — is addressed by [`plans/2026-06-18-constant-spacing-warp.md`](2026-06-18-constant-spacing-warp.md). The removed implementation is preserved on git branch `anchor-tp-finisher`. This document is kept as a point-in-time record.

(Adjust the relative link path per file location.)

### Bucket 3 — Code-comment sweep

10. `track_gen/inflation.py` (`_validity_stage` docstring, ~lines 112-115): the note "constant_spacing is not supported by the relaxed-track validity gate" is misleading. Clarify it applies to **this torch oracle** gate (which does not count-mask the NaN padding); the runtime **Warp** pipeline's `_validity_k` **is** count-masked, so `output_mode="constant_spacing"` is fully supported on the Warp path. (Greps found no other stale fixed-only claims in `track_gen/`.)

---

## Success criteria

- README + ARCHITECTURE describe `output_mode="constant_spacing"`, the count-aware convention (incl. the `count==N_max` parity invariant), and the corrected yield story (0.684→0.999, under-convergence not un-relaxable); no living doc implies fixed-256 is the only mode.
- A reader of the anchor/TP spec/plan immediately sees they are superseded/removed and where the real fix lives.
- No in-code comment implies constant_spacing is unsupported on the runtime path.
- No code behavior change; the test suite still passes unchanged (docs/comments only).

## Verification

- `grep` confirms `relax_anchor`/`relax_tp_finish` appear in the live docs only under a "superseded/removed" banner.
- README/ARCHITECTURE mention `constant_spacing`, `count[e]`/`N_max`, and the 0.684→0.999 result.
- `.venv/bin/python -m pytest -q` unchanged (no code logic touched; the only `.py` edit is a comment).
- Spot-read: the edited sections read coherently in each doc's existing voice.
