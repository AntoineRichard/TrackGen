# Handoff — Pure-Warp End-to-End Track-Gen Pipeline

**Date:** 2026-06-17
**Branch:** `feat/pure-warp-pipeline` (off `main`; `main` already has the merged + pushed relaxation feature + fused-Warp XPBD acceleration).
**Status:** 5 of 15 plan tasks done; foundation + all "easy" geometry/inflation kernels ported and verified. ~10 tasks remain (one validity kernel, the inflate assembly, the whole generation phase, and end-to-end assembly + CUDA-graph + integration + torch removal).

---

## 1. The goal (read these two first)
- **Spec:** `docs/superpowers/specs/2026-06-17-pure-warp-pipeline-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-17-pure-warp-pipeline.md` (15 tasks; this handoff tracks which are done)

Re-express the **entire** pipeline (generation → resample → relax → inflate) in **pure NVIDIA Warp kernels**, captured as **one end-to-end CUDA graph**, running on Warp's **CPU device** (tests/CI, GPU-free) and **CUDA** (prod), with **torch only as the I/O container**. The torch implementation is the **verification oracle** for each kernel and is **deleted at the end** (Task 15). No torch/Warp mixing is the explicit owner requirement.

## 2. How to run
- Python: `/home/antoiner/Documents/TrackGen/.venv/bin/python` (CUDA torch 2.6.0+cu124 + warp-lang 1.14 installed). There IS a GPU (RTX 5000 Ada, 16 GB).
- Tests: `.venv/bin/python -m pytest -q` (130 passing now). Warp kernels run on the Warp **cpu** device, so most tests run GPU-free; CUDA-only assertions are guarded by `torch.cuda.is_available()`.
- Warp init prints banner/compile noise to stderr — filter with `grep -v -i "module load\|kernel cache\|compiling\|Toolkit\|Devices\|x86_64\|sm_89\|mempool\|cache/warp\|cuda device\|initialized"`.
- Running standalone scripts that import `benchmarks.*`: prefix `PYTHONPATH=/home/antoiner/Documents/TrackGen`.
- **The GPU is shared and only 16 GB — run GPU work SERIALLY** (no parallel GPU agents; they OOM/contend). Kill stray python procs holding GPU memory between heavy runs (`nvidia-smi --query-compute-apps=pid --format=csv,noheader` then `kill -9`).

## 3. What's DONE (committed on this branch, all oracle-verified on cpu AND cuda)
All new pipeline kernels live in **`track_gen/warp_pipeline.py`**; tests in `tests/test_warp_*.py`.

| Plan task | Kernel(s) / wrapper | Oracle matched | Test |
|---|---|---|---|
| 1 | scaffolding, `_smoke_double`, `_init`/`_sync` | — (warp cpu+cuda smoke) | `test_warp_pipeline_smoke.py` |
| 3 | `_frame_k` → `frame_curvature(center)` | `geometry.tangents_normals` + `menger_curvature` | `test_warp_frame.py` |
| 4 | `_offset_build_k`/`_offset_assign_k` → `offset(center, Nrm, half_width)` | `inflation._offset_stage` (constant w) | `test_warp_offset.py` |
| 5 | `_ccw`(func), `_self_intersections_k`, `_sep_min_k`, `_curvrad_min_k`, `_thickness_k` → `self_intersections`/`separation_min`/`curvature_radius_min`/`thickness` | `geometry.*` (exact int for crossings; allclose for thickness) | `test_warp_geom_gates.py` |
| 2 | `_resample_scan_k`/`_resample_lookup_k` → `resample_uniform(center, n)` | `relaxation._resample_uniform` (FP-tol 5e-4) | `test_warp_resample.py` |

**Relax is already pure-Warp** (`track_gen/warp_relax.py::xpbd_solve` — fused separation+spacing+bending, validated 295s→0.32s @ E=8192, 0.98 valid). Phase-1 "relax" reuses it; no port needed.

## 4. What's LEFT (in order)
**Phase 1 finish:**
- **Task 6 — validity kernel.** Oracle: `inflation._validity_stage(center, w, count, gen_valid, config, outer, inner)`. Combine (per env): gen_valid AND |turning|≈2π AND w>w_floor AND no-NaN AND `thickness ≥ (1-relax_tol)*half_width` AND border self-intersections==0. Reuse the Task-5 `thickness`/`self_intersections` kernels + a turning-number kernel. Verify equal to the oracle on circle (valid), figure-eight (invalid), folded-border ellipse (invalid).
- **Task 7 — `inflate_warp(center, config) -> types.Track`.** Compose resample(Task2) → frame_curvature(Task3) → constant width → offset(Task4) → validity(Task6) → an arclength/length kernel. Verify all `Track` fields allclose to `inflation.inflate`.

**Phase 2 — generation (the design-heavy part):**
- **Task 8 — corner sampling** with **Warp built-in RNG** (`wp.rand_init(seed)` seeded per (env, attempt); `wp.randf`). Redesign of the grid-top-k is acceptable (RNG break already accepted) — validate by yield/diversity, NOT bit-equality.
- **Task 9 — `ccw_sort` kernel.** Oracle `geometry.ccw_sort` (orders by `atan2(dx,dy)` around centroid). Per-env insertion sort of ≤max_num_points corners; **ordering must match the oracle** so downstream geometry matches. Verify on fixed scrambled inputs.
- **Task 10 — assemble (vertex tangents + cubic Bézier).** Oracle `generators.vertex_tangents` + `_segment`/`_cubic_bezier`. Per dense sample: blend unit tangents `normalize(p*u_out+(1-p)*u_in)`, handles `rad*chord`, eval cubic Bernstein; NaN-propagate pruned corners. Verify allclose on fixed corners.
- **Task 11 — gates kernel.** Oracle: `_corner_angles`/min-angle, `geometry.turning_number`, simplicity via `self_intersections` on a **256-resample** (NOT full-res dense — that has sub-resolution corner cusps; the 256-resample is the validated criterion). Verify the per-env accept mask equals the torch conjunction.
- **Task 12 — static regen → `generate_centerline_warp(seeds, config)`.** Fixed `K=max_regen_iters` attempts; each env keeps its **first valid** candidate (masked); already-valid envs ignore later attempts. NO data-dependent control flow (so it's graph-capturable). Verify yield ≥ ~95% and every valid centerline is simple.

**Phase 3 — assembly + graph + integration:**
- **Task 13 — `generate_tracks_warp(config, seeds)`** = gen → resample → relax → inflate; verify validity/shape vs the torch `TrackGenerator.generate` (≥0.9 valid @ E=64, hw=0.03).
- **Task 14 — end-to-end CUDA graph capture** with fixed pre-allocated buffers + `wp.capture_begin/end` over the static region; verify replay==non-graph; record E=8192 timing.
- **Task 15 — facade integration + test migration + REMOVE torch compute** (the `inflation._*_stage`, torch gen internals in `generators.py`, unused torch geometry). Torch stays only as the array container. Full suite green; update the benchmark to drive the Warp pipeline.

## 5. How to port a kernel (the established pattern — copy it)
1. Kernel inside `if _HAVE_WARP:` in `warp_pipeline.py`; one thread per output element; env index `e = tid // N`.
2. Wrapper: `_init()`, wrap torch via `wp.from_torch(t.reshape(...).contiguous(), dtype=wp.vec2f|wp.float32|wp.int32)`, `wp.launch(kernel, dim=..., inputs=[...], device=str(tensor.device))`, `_sync(device)`, return torch views.
3. Test in `tests/test_warp_<name>.py`: `DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])`, parametrize, assert `allclose`/`equal` to the **torch oracle**. TDD: write the failing test first.
4. Run the targeted test (cpu+cuda) then the full suite; commit per task: `git commit -m "warp_pipeline: <kernel> == torch oracle"`.
Look at `_frame_k`/`frame_curvature` + `test_warp_frame.py` as the canonical template. More Warp idioms in `track_gen/warp_relax.py` and `docs/superpowers/spikes/2026-06-17-warp-xpbd/`.

## 6. Hard-won gotchas (these bit us — don't relearn them)
1. **RNG device mismatch = the "generation crash."** Passing **CPU** seed tensors to `PerEnvSeededRNG(..., device="cuda")` left seeds on cpu while states were on cuda → CUDA *illegal memory access*. It is **not** a Warp interop wall (a trivial Warp kernel mutating a torch CUDA tensor works fine — see the spike). For pure-Warp gen, use **Warp built-in RNG** and sidestep `PerEnvSeededRNG` entirely.
2. **Pure-Warp vs bit-tight oracle match.** Warp's float32 `sqrt`/`length` differs from torch by ~ULP; over N segments it accumulates to ~1e-4. **Do NOT reintroduce torch compute to force a bit-match** (a subagent did this in resample; it was reverted). Keep the kernel pure-Warp, accumulate sums in `wp.float64`, and **relax the test `atol` to the FP delta with a comment.** Geometrically negligible (~1e-4 on scale-2 coords).
3. **Warp runs on the CPU device** — that's the GPU-free test path; always test cpu+cuda.
4. **Fused Warp kernels are O(E·N) memory** (no `[E,N,N]` materialization) → **no chunking needed** (chunking was only for the torch dense path).
5. **CUDA-graph capture buys ~nothing at N=256** (kernels are compute-bound; launch overhead is negligible). The fused kernel (no torch ops, no per-iter sync) is the actual win. Capture is still wanted for end-to-end tidiness/launch-overhead at smaller batch — capture the gen→inflate region with fixed buffers + the static regen.
6. **Self-intersection metric scale.** The generator's simplicity gate uses a **256-point resample** (the pipeline resolution); full-res dense has sub-resolution corner cusps that falsely flag (this caused a yield collapse earlier — fixed). Keep simplicity at 256.
7. **`_quality`/`self_intersections` on full E=8192 can OOM** (builds `[8192,256,256]`). Compute validity metrics on a subset or chunk them.

## 7. Key files
- `track_gen/warp_pipeline.py` — pure-Warp kernels (this port). `track_gen/warp_relax.py` — fused relax (done).
- Oracles (the source of truth until Task 15 removes them): `track_gen/geometry.py`, `track_gen/generators.py`, `track_gen/inflation.py`, `track_gen/relaxation.py`.
- `track_gen/types.py` — `TrackGenConfig`, `Track`. `track_gen/track_generator.py` — facade (Task 15 wires it to `generate_tracks_warp`).
- Project memory: `~/.claude/projects/-home-antoiner-Documents-TrackGen/memory/` (`track-relaxation-redesign`, `trackgen-packaging-and-env`).

## 8. Definition of done (from the spec)
`generate_tracks_warp(config, seeds)` produces a `Track` on cuda + cpu (Warp devices), end-to-end CUDA-graph-captured on CUDA; validity yield + shape stats match the torch oracle (≥98% relaxed-valid) on a fixed seed set; per-kernel allclose tests pass cpu+cuda; torch compute removed (torch only the I/O container); full suite green; benchmark drives the Warp pipeline.

**Immediate next step:** Task 6 (validity kernel) — it only needs the Task-5 kernels + a turning-number kernel, so it's a clean continuation of the established pattern.
