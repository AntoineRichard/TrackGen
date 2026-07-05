# Repulsive-Growth Centerline Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the validated repulsive-growth spike into a first-class registered generator `"repulsive"` — the first non-CUDA-graph-capturable generator — with config, param-explorer, docs, README, and benchmark surfaces, and the speed limitation surfaced prominently.
**Architecture:** New `warp_generate_repulsive.py` (seed-driven obstacle sampling + pure-Warp TP-Sobolev growth, ported from the spike's `grow_warp.py`); a `capturable: bool` field on `GeneratorSpec` that routes non-capturable generators to an eager CUDA path; `repulsive_*` config fields; the untouched standard resample→relax→inflate tail.
**Tech Stack:** Python 3.10+, NVIDIA Warp (`warp-lang`), numpy (runtime); torch + matplotlib (dev-only, harness/assets). `.venv/bin/python` is the interpreter. **All pytest runs are prefixed `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** (a ROS pytest plugin autoloads otherwise); the machine has an RTX 4090 — **never** filter with `-m "not cuda"`.

## Global Constraints

- **Production module is torch-free.** `warp_generate_repulsive.py` imports only `warp` + `numpy`. Guard: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -c "import sys, track_gen; from track_gen._src import warp_generate_repulsive; assert 'torch' not in sys.modules"` → exit 0.
- **The four existing generators stay graph-capturable and unchanged.** `capturable` defaults to `True`; their registrations are not edited. `tests/test_warp_graph.py` must stay green for bezier/hull/polar/voronoi/checkpoint.
- **Repulsive is NOT captured.** The CUDA facade runs it eagerly; no `wp.Graph` is built for it. It is excluded from the capture parity suite.
- **Scratch allocated once.** `repulsive_alloc_scratch` sizes buffers at the max stage `N=256`; coarse stages slice the `[0:E·N_stage]` prefix. The per-iteration `wp.Tape` is the one deliberate exception (the reason for `capturable=False`).
- **Public API frozen.** No changes to `Track`, `TrackGenerator.generate()` signatures, or existing config defaults.
- **Baseline pytest count.** Record the current `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q` pass count before Task 1; assert the arithmetic at the end.
- Commits use `--no-gpg-sign` (no TTY for GPG in this env). Run all commands from the repo root.

---

## File Structure

- **Create** `track_gen/_src/warp_generate_repulsive.py` — the generator (obstacle kernel + growth kernels + scratch + `generate` + registration).
- **Create** `tests/test_generate_repulsive.py` — per-generator contract test (cpu+cuda).
- **Create** `docs/generators/repulsive.rst` — per-generator docs page (with the speed `.. warning::`).
- **Modify** `track_gen/_src/generator_registry.py` — `capturable` field + import line.
- **Modify** `track_gen/_src/track_generator.py` — eager CUDA branch for non-capturable generators.
- **Modify** `track_gen/_src/types.py` — `repulsive_*` fields + `__post_init__` validation + docstrings.
- **Modify** `tests/test_warp_graph.py` — filter `_ALL_GENERATORS` to capturable specs.
- **Modify** `tests/test_generator_registry.py` — `capturable` contract assertions.
- **Modify** `viz/param_explorer.py` — `repulsive_*` controls wiring.
- **Modify** `tests/test_param_explorer.py` — visibility/section-count updates.
- **Modify** `viz/render_readme_assets.py` — extend `GENERATORS` with `("repulsive", "Repulsive")`.
- **Modify** `tests/test_readme_assets.py` — add `generator-repulsive.png` to the expected set.
- **Modify** `docs/generators/overview.rst`, `docs/generators/benchmarks.rst`, `docs/index.rst`, `docs/contributing/writing-a-generator.rst`, `README.md` — count/wording/rows.

---

## Task 1: Capture contract — `capturable` flag + eager CUDA fallback

Add the `capturable` seam so a non-capturable generator can register and run without breaking the graph path. Behavior-preserving for the four existing generators.

**Files:**
- Modify: `track_gen/_src/generator_registry.py` (`GeneratorSpec` field + docstring)
- Modify: `track_gen/_src/track_generator.py` (`generate()` CUDA branch)
- Modify: `tests/test_warp_graph.py` (`_ALL_GENERATORS` filter)
- Modify: `tests/test_generator_registry.py` (new assertions)

**Interfaces:**
- Produces: `GeneratorSpec(name, alloc_scratch, generate, capturable=True)`.
- Consumes: `TrackGenerator._generator_spec.capturable`.

- [ ] **Step 1: Write failing registry test.** In `tests/test_generator_registry.py` add `test_generatorspec_has_capturable_default_true` (a `GeneratorSpec(...)` with three args has `.capturable is True`) and `test_existing_generators_are_capturable` (`reg.get("bezier").capturable is True`). Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_generator_registry.py`. Expected: FAIL (`GeneratorSpec` has no `capturable`).
- [ ] **Step 2: Add the field.** In `generator_registry.py` add `capturable: bool = True` to the frozen dataclass; update the docstring to state that `capturable=False` generators may use host control / per-call allocation and are run eagerly on CUDA. Run the Step-1 test. Expected: PASS.
- [ ] **Step 3: Route the facade.** In `track_generator.py::__init__` cache `self._capturable = self._generator_spec.capturable`. In `generate()` change the CUDA guard from `if _is_cuda:` to `if _is_cuda and self._capturable:`; add an `else:` that calls `self._run()` eagerly (covers cpu + non-capturable cuda). Leave `_CAPTURING` handling as-is.
- [ ] **Step 4: Filter the capture suite.** In `tests/test_warp_graph.py` change `_ALL_GENERATORS = generator_registry.available()` to keep only capturable specs: `_ALL_GENERATORS = [n for n in generator_registry.available() if generator_registry.get(n).capturable]`. Add `test_noncapturable_generators_excluded` asserting any `not capturable` name is absent from `_ALL_GENERATORS`.
- [ ] **Step 5: Run the FULL suite — existing behavior unchanged.** Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`. Expected: **baseline count, all passing** (the seam is behavior-preserving; `test_warp_graph.py` still covers the four capturable generators on the 4090). If the graph test count changes, a filter or branch is wrong — fix before continuing.
- [ ] **Step 6: Commit.**
```bash
git add track_gen/_src/generator_registry.py track_gen/_src/track_generator.py \
        tests/test_warp_graph.py tests/test_generator_registry.py
git commit --no-gpg-sign -m "feat(gen): GeneratorSpec.capturable + eager CUDA path for non-capturable generators"
```

---

## Task 2: Config surface — `repulsive_*` fields + validation

Add the tuned config knobs and their validation + docstrings.

**Files:**
- Modify: `track_gen/_src/types.py` (`TrackGenConfig` fields, docstrings, `__post_init__`)
- Modify: `tests/test_types.py` (validation tests)

- [ ] **Step 1: Write failing validation tests.** In `tests/test_types.py` add cases: defaults construct clean; `repulsive_grow_mult_max < repulsive_grow_mult_min` raises `ValueError`; `repulsive_obstacle_count_min < 1` raises; `repulsive_domain_init_ratio <= 1` raises; `repulsive_ratchet_rate <= 0` raises; `repulsive_stages=(64,128,200)` with `generator="repulsive"` (last ≠ `num_points=256`) raises; non-increasing `stages` raises. Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_types.py`. Expected: FAIL (fields absent).
- [ ] **Step 2: Add the fields.** Add the `repulsive_*` block (Spec §3.1 defaults) to `TrackGenConfig`, grouped under a `# --- Repulsive-growth generator params (generator="repulsive") ---` comment. Add the docstring entries in the existing house format, including an explicit **cost warning** line: repulsive is a host-driven, non-graph-capturable optimizer (~1000× slower than bezier; regenerate on a slow cadence / staggered slices, not every frame).
- [ ] **Step 3: Add validation.** In `__post_init__` add the ordered/positive checks (Spec §3.1), each raising `ValueError` with the offending value; gate the `repulsive_stages[-1] == num_points` check on `self.generator == "repulsive"`. Run the Step-1 test. Expected: PASS.
- [ ] **Step 4: Run the FULL suite.** Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`. Expected: baseline + new `test_types` cases, all passing.
- [ ] **Step 5: Commit.**
```bash
git add track_gen/_src/types.py tests/test_types.py
git commit --no-gpg-sign -m "feat(gen): repulsive_* config fields, defaults, and validation"
```

---

## Task 3: Seed-driven obstacle sampling kernel

Port the spike's host torch `sample_obstacles` to a pure-Warp, seed-driven, rejection-free kernel. Landed first so its determinism is proven before the growth loop consumes it.

**Files:**
- Create: `track_gen/_src/warp_generate_repulsive.py` (obstacle kernel + a thin test entry `_sample_obstacles_inplace`)
- Create: `tests/test_generate_repulsive.py` (obstacle determinism test only, for now)

**Interfaces:**
- Produces: `_sample_obstacles_k(seeds, config-derived scalars, obs_pts[E·M], obs_mw[E·M])`, deterministic per `(seed, env)`.

- [ ] **Step 1: Write failing determinism test.** In `tests/test_generate_repulsive.py`: `DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])`; `@pytest.mark.parametrize("dev", DEVS)` `test_obstacle_layout_deterministic` — sample obstacles twice with the same seeds → `np.array_equal`; distinct seeds → not equal; wall columns have weight 1.0, active disc columns 0.25, unused columns 0.0; every finite obstacle point lies within `r_dom`. Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_generate_repulsive.py`. Expected: FAIL (`ModuleNotFoundError` / attribute absent).
- [ ] **Step 2: Implement the kernel.** In `warp_generate_repulsive.py` write `_sample_obstacles_k` (one thread per env): `state = wp.rand_init(seeds[e]*_OBSTACLE_SALT + 23)`; write the `n_wall`-point wall ring at `r_wall = r_dom` (weight 1.0, mass `2π·r_wall/n_wall`); draw `k = wp.randi(state, k_min, k_max+1)`, `phase = wp.randf(state)*2π`; for `j in range(K_max)`: if `j < k`, draw `r_disc ∈ [r_frac_lo,r_frac_hi]·r_dom`, analytic band `lo = r_init + r_disc + 0.05·r_dom`, `hi = c_frac·r_dom`, `rad ∈ [lo,hi]` (skip → weight 0 if `lo ≥ hi`), `ang = phase + j·2π/k`, write the `n_disc`-point ring (weight 0.25); else zero the column block. Add `_sample_obstacles_inplace(...)` wrapper + module constants (`_OBSTACLE_SALT`, `n_wall=96`, `n_disc=12`, `c_frac=0.9`). Run the Step-1 test. Expected: PASS on cpu+cuda.
- [ ] **Step 3: Commit.**
```bash
git add track_gen/_src/warp_generate_repulsive.py tests/test_generate_repulsive.py
git commit --no-gpg-sign -m "feat(gen): seed-driven device-side obstacle layout for repulsive"
```

---

## Task 4: Growth loop + scratch + registration

Port the spike's `grow_warp.grow_warp` growth kernels and loop into the production module, allocate scratch once, wire `generate` + registration, and pass the full per-generator contract through the standard tail.

**Files:**
- Modify: `track_gen/_src/warp_generate_repulsive.py` (growth kernels, `RepulsiveScratch`, `repulsive_alloc_scratch`, `generate_repulsive_warp`, registration)
- Modify: `track_gen/_src/generator_registry.py` (import line in `_ensure_loaded`)
- Modify: `tests/test_generate_repulsive.py` (full contract)

**Interfaces:**
- Produces: `GeneratorSpec("repulsive", repulsive_alloc_scratch, generate_repulsive_warp, capturable=False)`.
- Consumes: `warp_pipeline.resample_uniform`, `warp_pipeline.arc_length_resample_inplace`, `warp_pipeline._fill_i32_k`.

- [ ] **Step 1: Write failing contract tests.** Extend `tests/test_generate_repulsive.py` (mirror `tests/test_generate_polar.py` / `test_generate_voronoi.py`), parametrized over `DEVS`, driving `TrackGenerator`: (a) `test_repulsive_is_registered` + `spec.capturable is False`; (b) `gen_centerline` shape `[E·num_points,2]`, all finite; (c) closed loop (gap ≤ 3×step); (d) NaN-tail contract on the post-tail `Track`; (e) determinism (same seed → `array_equal`) + diversity (distinct seeds → not `allclose`); (f) post-tail yield `> 0.5`; (g) `_compactness(...).mean() < 0.85`; (h) CUDA facade builds no graph for repulsive (`gen._graph is None` after two `generate()` calls on cuda). Use a small `E` (e.g. 16–32) for test speed. Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_generate_repulsive.py`. Expected: FAIL (generate/registration absent).
- [ ] **Step 2: Port the growth + step kernels.** Copy the differentiable energy kernels (`_tp_energy_k`, `_obstacle_energy_k`, `_length_penalty_k`), the Sobolev `_conv_k`, and the optimizer-step kernels (`_ratchet_k`, `_length_grad_k`, `_numden_k`, `_project_k`, `_gmean_k`, `_gmax_msl_k`, `_step_k`, `_perim_bc_k`, `_rescale_k`, `_freeze_update_k`) verbatim from `grow_warp.py`. Add `_init_circle_k` (writes the `r_init` circle into the coarsest-stage prefix). Add host helpers `_sobolev_circulant_row` (numpy irfft) and `_stage_schedule` (numpy `log1p`), unchanged from the spike.
- [ ] **Step 3: Scratch (alloc once).** Add `RepulsiveScratch` (`__slots__`) + `repulsive_alloc_scratch(config)`: per-env state arrays, `obs_pts/obs_mw` `[E·M]` (`M = n_wall + K_max·n_disc`), N-dependent buffers sized at `N=256`, the per-stage circulant rows uploaded as device arrays, and the host-computed stage schedule + `n_ratchet`/`n_iters` (from `L_final.max()` — precomputed on host from the `grow_mult` bounds so no device readback is needed to size the loop). `_BEZIER_PERIMETER_REF = 5.05` module constant; `P_ref = _BEZIER_PERIMETER_REF·config.scale`, `r_dom = domain_frac·P_ref`, `r_init = r_dom/domain_init_ratio`.
- [ ] **Step 4: `generate_repulsive_warp`.** Orchestrate: sample obstacles (Task 3 kernel) → seed `L_final` per env from a `grow_mult` draw off `seeds` (a tiny `wp.rand_init` kernel; `L_final = grow_mult·2π·r_init`) → `_init_circle_k` → the host-driven coarse-to-fine + stall-stop loop (stage transitions via `arc_length_resample_inplace` on prefix slices; periodic in-stage `resample_uniform`; per-window `frozen.numpy().sum()` readback + global early exit) → final `resample_uniform` at `N=256` → `wp.copy(out_centerline, final_N256_buffer)` → `wp.launch(_fill_i32_k, out_valid_wp, 1)`. Register the spec with `capturable=False` and add the import line to `generator_registry._ensure_loaded`. Run the Step-1 test. Expected: PASS on cpu+cuda.
- [ ] **Step 5: torch-free guard.** Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -c "import sys, track_gen; from track_gen._src import warp_generate_repulsive; assert 'torch' not in sys.modules"`. Expected: exit 0. If torch leaks, a spike import (oracle/torch) sneaked in — replace with the Warp/numpy equivalent.
- [ ] **Step 6: Run the FULL suite.** Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`. Expected: baseline + repulsive contract + the auto-enrolled `test_shape_variety.py`/`test_generator_simplicity_gate.py`/`test_compare_generators.py` (now iterating repulsive) all passing. `test_warp_graph.py` unchanged (repulsive excluded).
- [ ] **Step 7: Commit.**
```bash
git add track_gen/_src/warp_generate_repulsive.py track_gen/_src/generator_registry.py \
        tests/test_generate_repulsive.py
git commit --no-gpg-sign -m "feat(gen): repulsive-growth generator (pure-Warp TP-Sobolev, non-capturable)"
```

---

## Task 5: Parameter explorer wiring

Expose the `repulsive_*` controls in the Tracks tab (the dropdown already lists it via the registry).

**Files:**
- Modify: `viz/param_explorer.py` (9 wiring points — Spec/agent survey)
- Modify: `tests/test_param_explorer.py` (visibility + section-count updates)

- [ ] **Step 1: Write failing explorer tests.** In `tests/test_param_explorer.py`: add `"repulsive": False` to each expected dict in `test_track_visible_sections_are_generator_specific` and a `px.track_visible_sections("repulsive") == {...True for repulsive...}` case; update the `len(px.track_mode_visibility(gen)) == 31` assertion to `31 + N` and include `"repulsive"` in the loop; add `test_build_config_maps_repulsive_shape_knobs` mirroring the voronoi case. Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_param_explorer.py`. Expected: FAIL.
- [ ] **Step 2: Wire the controls.** Edit the 9 points (mirroring the voronoi block): `default_params()`, `build_config()`, `track_visible_sections()`, `TRACK_MODE_SECTION_SIZES` (`("repulsive", N)`), `_collect()` keys, `build_app()` control creation (a `### Repulsive growth` markdown + one slider/checkbox per field, `visible=track_mode_visible["repulsive"]`), the `controls` list, and `track_mode_outputs` — keeping positional order consistent with `_collect` and the section-size guard. Run the Step-1 test. Expected: PASS.
- [ ] **Step 3: Run the FULL suite.** Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`. Expected: all passing (the `build_app` smoke test exercises the new controls).
- [ ] **Step 4: Commit.**
```bash
git add viz/param_explorer.py tests/test_param_explorer.py
git commit --no-gpg-sign -m "feat(explorer): repulsive generator controls"
```

---

## Task 6: Docs — generator page, overview, benchmarks, index, contract note

**Files:**
- Create: `docs/generators/repulsive.rst`
- Modify: `docs/generators/overview.rst`, `docs/generators/benchmarks.rst`, `docs/index.rst`, `docs/contributing/writing-a-generator.rst`

- [ ] **Step 1: Write the generator page.** Create `docs/generators/repulsive.rst` mirroring `voronoi.rst`'s structure (title, intro, `.. figure:: ../assets/generator-repulsive.png`, How It Works, Math, Parameters — one entry per `repulsive_*` field, What Makes It Distinct, Fallback and Validity). Include a prominent **`.. warning::`** callout: repulsive is a host-driven, non-graph-capturable optimizer — ~0.18 s @ E=64 / ~5.1 s @ E=8192 on a 4090 vs ~2 ms for bezier; regenerate on a slow cadence or in staggered per-env slices, not every frame; graph capture is future work.
- [ ] **Step 2: Update overview / index / contract.** In `overview.rst`: "five" → "six", add the `repulsive` subsection + a "When to Use Which" row (foldy serpentine, compactness ≈ 0.15, best for maze-like circuits; note the speed caveat). In `benchmarks.rst`: "five" → "six" and add a placeholder `* - repulsive` row (filled in Task 8 from the harness). In `docs/index.rst`: add `generators/repulsive` to the toctree, update the "Five generators" card wording, and add a "(all generators except `repulsive`)" caveat to the "CUDA-graph capture" card. In `writing-a-generator.rst`: note the new `capturable` flag on `GeneratorSpec` (rule 3 — a generator may declare `capturable=False` to opt out of the graph path and run eagerly).
- [ ] **Step 3: Build the docs.** Run: `.venv/bin/python -m sphinx -b html docs docs/_build/html -q` (or the repo's documented docs-build command). Expected: clean build, no toctree/reference warnings for `generators/repulsive`.
- [ ] **Step 4: Commit.**
```bash
git add docs/generators/repulsive.rst docs/generators/overview.rst \
        docs/generators/benchmarks.rst docs/index.rst docs/contributing/writing-a-generator.rst
git commit --no-gpg-sign -m "docs: repulsive generator page + overview/index/contract updates"
```

---

## Task 7: README + gallery assets

**Files:**
- Modify: `README.md` (pipeline strip line, feature-table wording)
- Modify: `viz/render_readme_assets.py` (`GENERATORS` list)
- Modify: `tests/test_readme_assets.py` (expected PNG set)

- [ ] **Step 1: Write failing assets test update.** In `tests/test_readme_assets.py` add `"generator-repulsive.png"` to the expected `names` set in `test_render_generator_panels_writes_pngs`. Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_readme_assets.py`. Expected: FAIL (repulsive not yet in `GENERATORS`).
- [ ] **Step 2: Extend the asset list + README text.** In `render_readme_assets.py` append `("repulsive", "Repulsive")` to `GENERATORS`. In `README.md`: add `repulsive` to the pipeline ASCII strip line and update the feature-table row ("Five track generators" → "Six …, add repulsive") with a short parenthetical noting repulsive is the slow, non-graph-captured serpentine option. Run the Step-1 test. Expected: PASS.
- [ ] **Step 3: Render the assets.** Run: `.venv/bin/python -m viz.render_readme_assets`. Expected: writes `docs/assets/generator-repulsive.png` + refreshed `readme-generator-strip.png` / `readme-generator-grid.png` (now with the repulsive column).
- [ ] **Step 4: Commit.**
```bash
git add README.md viz/render_readme_assets.py tests/test_readme_assets.py \
        docs/assets/generator-repulsive.png docs/assets/readme-generator-strip.png \
        docs/assets/readme-generator-grid.png
git commit --no-gpg-sign -m "docs(readme): repulsive in feature table + gallery assets"
```

---

## Task 8: Benchmarks entry

`benchmarks/compare_generators.py` iterates `generator_registry.available()`, so it auto-includes repulsive. Fill the docs benchmarks row from a real run and flag the wall-clock.

**Files:**
- Modify: `docs/generators/benchmarks.rst` (fill the `repulsive` row)
- (No code change to `compare_generators.py` unless it hard-codes a generator list.)

- [ ] **Step 1: Confirm auto-inclusion.** Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_compare_generators.py`. Expected: PASS (repulsive iterated; if the test hard-codes a name set or count, update it here).
- [ ] **Step 2: Generate the metrics + wall-clock.** Run `benchmarks/compare_generators.py` (per its docstring) on the 4090; capture repulsive's row (compactness ≈ 0.15, yield, and the ~0.18 s @ E=64 / ~5.1 s @ E=8192 wall-clock). Paste into the `* - repulsive` row in `benchmarks.rst`, with a note that its wall-clock is ~1000× the captured generators and throttle-sensitive (±1.5×).
- [ ] **Step 3: Commit.**
```bash
git add docs/generators/benchmarks.rst tests/test_compare_generators.py
git commit --no-gpg-sign -m "docs(bench): repulsive metrics row + wall-clock caveat"
```

---

## Task 9: Final verification

- [ ] **Step 1: Full suite on the 4090.** Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`. Expected: **baseline + (registry capturable 2, types validation ~6, repulsive contract ~8, param-explorer additions, readme-assets 1) = new total, all passing.** `test_warp_graph.py` green with repulsive excluded; the auto-enrolled `test_shape_variety` / `test_generator_simplicity_gate` / `test_compare_generators` green with repulsive included. **Do not** add `-m "not cuda"`.
- [ ] **Step 2: torch-free guard.** Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -c "import sys, track_gen; from track_gen._src import warp_generate_repulsive; assert 'torch' not in sys.modules"`. Expected: exit 0.
- [ ] **Step 3: Docs build + asset render.** Run the docs build and `.venv/bin/python -m viz.render_readme_assets`. Expected: clean build; assets regenerate without error.
- [ ] **Step 4: Eager-CUDA sanity.** Run a tiny script: construct a `TrackGenConfig(generator="repulsive", device="cuda", num_envs=32)`, `TrackGenerator(...).generate()` twice, assert `gen._graph is None` (no capture) and `Track.valid.numpy().mean() > 0.5`. Expected: PASS. (Serves as the manual end-to-end verification the spec's DoD requires.)
- [ ] **Step 5: Final commit (if any residual).**
```bash
git add -A
git commit --no-gpg-sign -m "chore(gen): finalize repulsive generator (suite green, docs + assets)"
```

---

## Self-Review (plan author)

**Spec coverage:** capture contract → Task 1 ✓; module + scratch-once + torch-free → Tasks 3–4 ✓; seed-driven device-side obstacles → Task 3 ✓; config fields + validation + cost docstring → Task 2 ✓; output at `num_points` + `valid=1` + untouched tail → Task 4 ✓; obstacles-as-content out of scope → not implemented (resume note in spec) ✓; speed surfacing (config, docs warning, README, benchmarks) → Tasks 2/6/7/8 ✓; docs/param-explorer/README/assets → Tasks 5–8 ✓; final suite + docs build + asset render → Task 9 ✓.

**Placeholder scan:** the only intentionally deferred value is the `benchmarks.rst` metrics row (Task 8 Step 2, filled from a real harness run) and the param-explorer section-count `N` (resolved when the control count is fixed in Task 5). No `TODO`/`...` left in shipped code.

**Type consistency:** `capturable: bool = True` keeps the four existing 3-arg registrations valid; `repulsive_*` fields follow the existing `voronoi_*`/`checkpoint_*` typing; the generator writes the same `out_centerline [E·num_points] vec2f` / `out_valid [E] int32` buffers as every other generator; scratch is a `__slots__` class like `PolarScratch`.
