# Pluggable First-Stage Generator Framework — Design Spec

**Date:** 2026-06-21
**Status:** Design (pre-implementation)
**Depends on:** the merged gen/relax split (`track_gen/_src/warp_generate.py` isolates
generation; `config.generator` selector exists; `track_generator` currently asserts
`"bezier"`).
**Strategy input:** `docs/pre-relaxation-generator-methods.md` (generator contract,
comparison metrics, 5-method shortlist).

## Goal

Turn the first stage of TrackGen (initial closed-centerline generation, before
constant-spacing resampling + XPBD relaxation) into a **pluggable catalog of
user-selectable generators**, plus an **evaluation harness that characterizes each
method's tradeoffs** (quality, diversity, speed, racing character).

The framework is the product. The goal is **not** to crown a single best generator: we
want to add methods from the state of the art even when they are not the strongest on
yield, because users may favor them for speed, style, controllability, or research value.
Evaluation exists to *inform* the user's choice, never to gate which methods ship. Every
registered generator stays selectable.

A secondary requirement shapes the architecture: adding a method must be an **additive,
parallelizable** operation. We intend to dispatch independent agents — each in its own git
worktree — to implement different methods concurrently. The design must let N such agents
add N generators while touching almost no shared code.

## Background / current state

- `track_gen/_src/warp_generate.py` holds the current generator
  `generate_centerline_warp(seeds_wp, config, out_centerline, out_valid, scratch)`
  (sample corners → ccw-sort → closed Bézier → arc-resample → polygon fallback).
- `TrackGenConfig.generator: str = "bezier"  # one of {"bezier","fourier"}`; `"fourier"`
  is an unsupported torch-only experimental path.
- `track_generator.py` asserts `config.generator == "bezier"`.
- Scratch is split into `GenScratch` / `RelaxScratch` / `InflateScratch` + bridge buffers
  (from the recent split refactor), pre-allocated once in `_inflate_warp_alloc`.
- The runtime is strictly Warp-native and torch-free; `TrackGenerator` pre-allocates all
  buffers and the whole pipeline is captured as one CUDA graph on cuda.

## Non-goals (this spec)

- **No new generator method** is implemented here. This spec delivers the *framework* and
  validates it with the existing bezier generator as the baseline.
- **No decision on the torch-prototype-first vs Warp-direct methodology** for building new
  methods — deferred to the first-generator sub-project.
- The harness does **not** gate, prune, or auto-select generators. It characterizes.

## Architecture overview

Two deliverables:

1. **Dispatch seam** (runtime, `track_gen/_src`): a registry of generators keyed by name;
   `TrackGenerator` resolves `config.generator` once at construction, pre-allocates that
   generator's scratch, and the orchestrator calls it. Static dispatch — only the chosen
   generator's kernels enter the captured graph. Switching generators = a new
   `TrackGenerator`. Zero-allocation and CUDA-graph capture are preserved.

2. **Comparison harness** (dev tool, `benchmarks/`): runs any registered generator over a
   fixed seed suite through the full pipeline, reads results back once, computes quality +
   speed metrics host-side in numpy/torch, and prints a **tradeoff table** (+ optional
   track-grid render). It is not part of the shipped runtime, so torch/numpy/matplotlib
   are fine.

**Rejected alternatives.** (a) `if/elif config.generator` branching in `_run_pipeline` —
couples the orchestrator to every generator and bloats the shared scratch union. (b) An OO
`Generator` ABC with instances — heavier than the functional Warp-kernel style and buys
nothing over a module + a spec record.

## Component 1 — Generator dispatch seam

- A `GeneratorSpec` record bundles a generator's two callables:
  - `alloc_scratch(config) -> <gen scratch group>` — allocate this generator's per-env
    scratch buffers ONCE (fixed shapes from config). Called at `TrackGenerator.__init__`.
  - `generate(seeds_wp, config, out_centerline_wp, out_valid_wp, gen_scratch) -> None` —
    write the centerline + validity in place (see the contract below).
- A registry `GENERATORS: dict[str, GeneratorSpec]` maps the `config.generator` string to a
  spec. Assembled in one small module that imports each generator module and registers its
  spec (one import + one dict entry per generator — the only shared touch-point).
- `TrackGenerator.__init__`: resolve `spec = GENERATORS[config.generator]` (raise a clear
  `ValueError` listing available names if absent), call `spec.alloc_scratch(config)` to get
  the gen-scratch group, store `self._generate = spec.generate`.
- `_run_pipeline` / `_inflate_warp_alloc`: call the resolved spec instead of hardcoding
  bezier. `RelaxScratch` / `InflateScratch` / bridge buffers remain shared and generator-
  agnostic.
- `track_generator`'s `assert config.generator == "bezier"` becomes
  `if config.generator not in GENERATORS: raise ValueError(...)`.
- The current `warp_generate.py` becomes the registered `"bezier"` generator; its existing
  entry already matches the contract — formalize it as a `GeneratorSpec`.

This keeps the captured graph allocation-free (the chosen generator's scratch is
pre-allocated; only its kernels are recorded) and makes adding a generator additive.

## Component 2 — The generator contract (the brief for worktree agents)

This is the durable interface every method implements. It is written to be sufficient for a
fresh agent to implement a method against it with no other context.

- **Identity:** registered under a unique name (the `config.generator` value).
- **Inputs:** `seeds_wp` (`[E]` int32 per-env seed), static `config` (`TrackGenConfig`), and
  the pre-allocated gen-scratch group from its own `alloc_scratch`.
- **Outputs (in place):** `out_centerline_wp` (`[E*N]` `wp.vec2f`, `N = config.num_points`,
  a CLOSED 2D centerline per env) and `out_valid_wp` (`[E]` int32, 1 if the env produced a
  usable centerline). Downstream resample/relax/inflate consume `out_centerline_wp`.
- **Validity target:** a simple (non-self-intersecting) closed loop is preferred but NOT
  required — XPBD repairs thickness and the polygon fallback handles self-crossings. Output
  must be finite (no NaN) for valid envs.
- **Shape / graph constraints:** all buffers fixed-size from config (fixed max control
  count, fixed dense sample count), bounded loops, no dynamic allocation, no host-side retry
  loop conditioned on generated data, no per-env Python branching. The whole `generate` must
  be CUDA-graph capturable (pure Warp launches; no host sync inside).
- **Determinism:** output is a deterministic function of `(per-env seed, config)`; per-env
  RNG via the existing Warp samplers. cpu vs cuda may differ (Warp RNG), as elsewhere.
- **Diversity knobs:** read style parameters from `config`; per-env style sampling is
  encouraged (sample style per row, not per batch).
- **Performance:** batch-friendly, one env per row.

The contract lives as a documented section (in `warp_generate`'s package docstring or a
short `docs/generator-contract.md`) so it can be handed verbatim to a worktree agent.

## Component 3 — Comparison harness

`benchmarks/compare_generators.py` — a dev tool (NOT imported by the runtime; may use
torch/numpy/matplotlib).

- Inputs: a list of generator names (default: all registered), a fixed seed suite
  (seed base + `E`), and a base `TrackGenConfig`.
- Per generator:
  - **Full-pipeline pass:** build `TrackGenerator(config_with_that_generator, rng)`, run
    over the seed suite (batched), read back the resulting `Track` once.
  - **Pre-relax capture (single pass):** XPBD writes a *separate* `relaxed` buffer, so the
    bridge buffer `cs_center` (the constant-spacing centerline XPBD receives) is not
    overwritten during a run. The harness reads `cs_center` (pre-relax) and `Track.center`
    (post-relax) from one full-pipeline run — no extra generation-only pass needed. (The
    harness is a dev tool and may reach `TrackGenerator`'s scratch directly.)
  - Compute all metrics (Component 4) host-side in numpy/torch.
- Output: a **tradeoff table** (rows = generators, columns = metrics, including speed),
  printed and saved (CSV/markdown), plus an optional track-grid render. The harness ranks
  and characterizes; it never filters generators out.

## Component 4 — Metrics

All computed host-side in numpy/torch on read-back arrays (the harness is offline).

- **post_relax_yield** = mean(`Track.valid`).
- **pre_relax_self_intersection_rate** = fraction of envs whose pre-relax centerline
  self-intersects (using the existing self-intersection logic / a numpy equivalent).
- **fallback_rate** (optional, generator-specific) = fraction routed to a rescue path,
  if the generator exposes a per-env fallback counter; else reported N/A.
- **xpbd_displacement** = mean over envs of mean-point `‖c_post − c_pre‖` (relaxation
  burden proxy).
- **diversity:** centerline length (perimeter), shoelace area, compactness
  (`4π·area / perimeter²`), and a turn-angle (exterior-angle) histogram.
- **racing_line_proxy:** peak curvature, integrated `∫κ² ds`, and a friction-circle
  speed/lap-time estimate (`v_i = sqrt(a_lat_max / max(κ_i, ε))`, lap-time
  `∝ Σ ds_i / v_i`).
- **generation_cost:** warm per-call wall-clock and throughput (tracks/s) for `generate`
  alone, plus full-pipeline wall-clock. This is the speed axis users may optimize for.

## Component 5 — Seed suite + bezier baseline

- A fixed, reproducible suite: fixed seed base, fixed `E` (e.g. 4096), and a representative
  default `TrackGenConfig`. Documented so every method is judged identically.
- Run the harness on `"bezier"` and **commit a baseline metrics table** as the reference all
  new methods are reported against.

## Constraints / invariants (carried from the runtime)

- **Runtime (`track_gen/_src`) stays Warp-native and torch-free.** The seam, registry, and
  every generator obey this. The single gated `count.numpy()` truncation warning is the only
  host readback and keeps its `not _CAPTURING` gate.
- **Zero per-call allocation** on the `generate()` path; the chosen generator's scratch is
  pre-allocated at construction. The CUDA-graph capture region stays allocation-free (the
  graph parity test is the tripwire).
- **The harness is a dev tool** in `benchmarks/`; torch/numpy/matplotlib are fine there and
  it is never imported by the package.
- **Public API:** `config.generator` is the documented user selector; `track_gen.__all__`
  is unchanged. Generators are internal modules selected by name. README/ARCHITECTURE list
  the available generators.
- Commits use `--no-gpg-sign` (no TTY for GPG in this env).

## Testing

- **Per-metric unit tests** on analytic shapes: a circle (known area, compactness ≈ 1,
  constant curvature, zero self-intersections), a figure-eight (self-intersection detected),
  a known polygon (length/area). Pins each metric's correctness independent of any
  generator.
- **Seam/dispatch test:** the registry resolves names; an unknown name raises a clear error;
  the existing bezier path is byte-for-byte unchanged through the seam (the full suite stays
  green, including the CUDA-graph parity test, proving zero-alloc-during-capture survives).
- **Harness smoke test:** runs on a tiny `E` and produces a table without error.

## Parallel-worktree development model (enabled, not exercised here)

Once this framework lands, each new method is its own follow-on sub-project. The intended
workflow: dispatch one agent per method, each in its own git worktree, each handed the
generator contract (Component 2) as its brief and the harness (Component 3) as its
acceptance check. A method is "done" when it registers cleanly, the full suite stays green,
and it produces a harness row. Because a method = one new module + one `GeneratorSpec` + one
registry line + its own tests, N agents collide only on the single registry line (a trivial
merge). This spec exists partly to make that collision surface as small as possible.

## Scope boundary & follow-on sub-projects

- **This spec:** dispatch seam + registry, the documented generator contract, the comparison
  harness (with the speed axis), and the committed bezier baseline. No new method.
- **Follow-on specs (one per method, parallelizable):** the shortlist from
  `docs/pre-relaxation-generator-methods.md` — per-env style sampling on bezier, convex-hull
  + midpoint displacement, periodic polar spline, curvature-profile/clothoid,
  checkpoint-steering, etc. Each follow-on also settles the torch-prototype-vs-Warp-direct
  methodology for itself.
