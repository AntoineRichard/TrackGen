# Checkpoint-steering generator — Warp port report

Method #5 "checkpoint-steering" (generator name `"checkpoint"`) productionized from the
validated host prototype `track_gen/_experimental/checkpoint_proto.py` into a registered,
pure-Warp, CUDA-graph-capturable first-stage centerline generator. Mirrors the
`polar`/`hull` module shape.

## Status: DONE

## What was added (additive only)

1. **Config fields** (`track_gen/_src/types.py`): a `checkpoint_*` block with tuned
   prototype defaults + per-field WHY comments, plus `__post_init__` validation:
   - `checkpoint_count: int = 12`
   - `checkpoint_radius_min_frac: float = 0.33`
   - `checkpoint_angle_jitter: float = 0.55`
   - `checkpoint_turn_rate: float = 0.42`
   - `checkpoint_steer_gain: float = 0.65`
   - `checkpoint_lookahead_frac: float = 0.16`
   - `checkpoint_best_of_k: int = 4` (ship 4 — proven low SI; the prototype's 8 was only
     its render-driver K)
   - `checkpoint_clip_fallback: bool = False` (opt-in single-crossing clip; default off,
     like the bezier/hull polygon fallback)
   - Validation: `checkpoint_count >= 3`, `checkpoint_best_of_k >= 1`,
     `0 <= checkpoint_radius_min_frac < 1`.

2. **Generator module** (`track_gen/_src/warp_generate_checkpoint.py`): pure-Warp.
   Kernels:
   - `_sample_checkpoints_k` (dim=E*K): C checkpoints per (env, candidate),
     angle `2*pi*c/C + jitter`, radius `U(rmin*R, R)`, R=1.
   - `_steer_k` (dim=E*K): fixed-N bounded-turn steering, `dl = ring_perimeter/N`
     (ring perimeter is a reduction over the C checkpoints — pins ~one lap); per-step
     int-counter target advance (no host branch).
   - `_close_heading_ramp_k` (dim=E*K): additive heading-drift closure to turning
     number 1 (preserves local curvature), gap-distribution displacement close,
     recenter. (Only `heading_ramp` ported; the `pos`/`kappa` variants were
     comparison-only and ignored.)
   - best-of-K selection over E*K: `self_intersections_inplace` over E*K candidates,
     then `_select_best_k` (deterministic argmin per env, ties -> lowest k) copies the
     min-crossing candidate into `out_centerline`. K==1 uses `_copy_single_k`
     (capture-time Python branch — skips the crossing pass entirely).
   - `_normalize_centerline_k` (dim=E): center + bbox rescale (identical to polar).
   - `_clip_assemble_k` + `_arc_resample_inplace`: opt-in single-crossing clip
     (capture-time Python branch on `config.checkpoint_clip_fallback`). Finds the first
     crossing + intersection point P, keeps the longer sub-loop arc, NaN-pads into a
     dense buffer, arc-resamples back to N. Reuses the shared NaN-aware arc-resampler
     and the `_pipe._ccw` proper-crossing predicate.
   - `CheckpointScratch` + `checkpoint_alloc_scratch` (one alloc, sized by K; clip
     buffers allocated only when `checkpoint_clip_fallback=True`). Distinct RNG salt
     `_CHECKPOINT_SALT = 4099` + per-candidate offset salt `_CAND_SALT = 2741`.

3. **Registration**: one `GeneratorSpec(name="checkpoint", ...)` at module bottom + one
   import line in `generator_registry._ensure_loaded`. `track_gen.__all__` unchanged.

## HARD constraints — verified
- Pure Warp, torch-free: `grep -rn "import torch" track_gen/_src/` is empty.
- Zero per-call allocation: all buffers from `checkpoint_alloc_scratch`; outputs
  orchestrator-owned.
- CUDA-graph-capturable: verified capture + replay on cuda:0 for K=1, K=4, and
  K=4+clip — all replay-stable (no host sync, fixed loops over C/N/K, K-branch and
  clip-branch are capture-time Python branches).
- Deterministic in (seed, config): cuda:0 SI/compactness numbers match cpu exactly.

## Verify — results

`available()` -> `['bezier', 'checkpoint', 'hull', 'polar']` (includes `checkpoint`).

Pre-relax self-intersection rate (`relax_enable=False`, via
`benchmarks.track_metrics.self_intersects`), E=128, hw=0.5, scale=10, spacing=0.30 —
**identical on cpu and cuda:0**:

| K | clip  | pre-relax SI | compactness median |
|---|-------|--------------|--------------------|
| 1 | False | 0.2031       | 0.571              |
| 1 | True  | 0.0469       | 0.592              |
| 4 | False | 0.0000       | 0.598              |
| 4 | True  | 0.0000       | 0.599              |

K=4 drives pre-relax SI to 0.0% (beats the expected ~0.4%); K=1 clip drops SI from
20% to 4.7% (the opt-in fallback works). Compactness median ~0.59 is well below the
shape-variety gate (< 0.65 / < 0.85) — not a circle.

Post-relax (default: K=4, hw=0.5, scale=10, spacing=0.30, relax_iters=150):
**valid yield = 1.0000** (25/25 in the render batch, 128/128 in the metrics batch),
compactness median ~0.59.

Tests:
- `pytest tests/test_shape_variety.py tests/test_warp_graph.py -q` -> 6 passed
  (shape-variety auto-includes `checkpoint`; the CUDA graph tripwire green).
- Full suite `pytest -q` -> **286 passed, 1 skipped**. The single skip is
  `tests/test_param_explorer.py` (missing `gradio`), pre-existing and unrelated to
  checkpoint.

Render: a 5x5 post-relax centerline grid of the default config can be regenerated as
`viz/out/checkpoint_warp_grid.png` (all 25 valid in the original run, non-circular
CarRacing-style loops). `viz/out/` is gitignored render scratch, so the PNG is not part
of the committed report evidence and may not be present in a fresh checkout.

## Concerns
- None blocking. The single test skip is pre-existing (`gradio` not installed) and
  unrelated to this work.
- Render PNGs are intentionally not committed because `viz/out/` is gitignored
  (matches the existing repo convention for local renders); regenerate them when needed.
