# AGENTS.md — handoff for the 0.3 line (branch `v3`)

TrackGen is a pure NVIDIA Warp, CUDA-first batched race-track and gate-course generator, with
runtime utilities (collision, progress, localization, ...) and `Course`. Everything must stay
CUDA-graph capturable: no host syncs or allocations on hot paths, and every launch goes
through a `LaunchContext` (`ctx`). Read `docs/library-readiness/target-architecture.md` for
the architecture and the roadmap, and `CHANGELOG.md` for what shipped.

## State (2026-10-25)
- `main` = **v0.2.0** (released; GitHub release v0.2.0). It is the v2 refactor, roadmap
  PR0–PR15b; see the CHANGELOG and `docs/upgrading/from-0.1.rst`.
- `v3` = the 0.3 line, version `0.3.0.dev4`. Roadmap PR16 cleanup is implemented in
  four dev milestones. The rc and downstream racing-repository validation are still
  required before 0.3.0. 1.0 is a later milestone.

## User decisions for 0.3 (binding; they override any older doc that says otherwise)
Not production software: break things cleanly, with no compatibility shims or deprecation
cycles.
1. **Aggressive removals:** remove ALL deprecated and legacy surface, whether or not it
   warned in 0.2.0. That covers:
   - the `_src` helper re-exports and module hooks, and the legacy registries;
   - the frozen `generate_*_warp` / `*_alloc_scratch` / `apply_z_profile*` functions, the
     float sampling helpers, and their scalar and style kernels;
   - the 0.1 plugin ABIs (`GateGeneratorSpec`; decide whether `GeneratorSpec` stays as the
     simple plugin path or everything is `Stage`, and say why in the CHANGELOG);
   - `style_sampling` and its ranges, `output_mode`, the dead config fields,
     `set_capturing` / `_CAPTURING`, the `.params` alias, `CourseLine.track`, the `/1`
     spec format;
   - inert params, e.g. `validity.gate_default`'s `gate_width` / `check_crossing`, plus
     `tools.Progress(checkpoints=)` and the ignored `capturing=` args.
2. **RNG:** the built-ins switch to `ops.rng.init`-style hashed salts, which fixes the
   seed-0 salt collapse and the ad-hoc salts. There is NO byte-identical reproduction mode.
   Every track changes for a given seed, so re-baseline the goldens once.
3. **`on_invalid="poison"`** becomes the default everywhere, configs included.
4. **Frozen configs** (`frozen=True`), and the newer `Track` / `GateSequence` fields become
   required.
5. Dev releases are `0.3.0.devN` tags on `v3`. 0.3.0 ships later, after an rc that the
   user's racing repos test. List every break in the CHANGELOG and in a new
   `docs/upgrading/from-0.2.rst`.

Inputs: `docs/library-readiness/handoff-0.3/`:
- `pr16-spec.md`;
- `pr16-design-0.2-conservative.md`, the earlier design written for the *conservative*
  path. Reuse its removal inventory, its evidence (§1, §12 critic review) and its pitfalls.
  Its §13 records the decisions above; ignore its tombstones, `rng="0.2"` and staged
  deprecations. Historical pitfall: live stages called legacy helpers; PR16 moved every
  internal caller to private or `ops` code before deleting them.

Suggested split (each one its own commit and dev tag):
- **A, removals:** must change NO output; the byte guards below must stay green.
- **B, RNG switch + golden re-baseline:** add a statistics check replacing the byte check.
  Yield, compactness, curvature and length distributions must stay close to 0.2's.
- **C, defaults:** poison, frozen configs, required fields.
- **D, docs:** `from-0.2.rst`, every page, and the CHANGELOG.

## Validation (every part, before pushing)
CPU (macOS arm64 works; Python venv with `pip install -e ".[dev,docs]"`):
```bash
PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  -m "not cuda and not slow and not benchmark"
python docs/_tools/build_ci_env.py -W --keep-going -E -b html -d /tmp/dt docs /tmp/html   # CI-equivalent docs build
```
- Byte guards (must not change in part A): `tests/test_generator_fingerprints.py`,
  `tests/test_pipeline_parity.py`, `tests/test_gate_pipeline_parity.py`, and the goldens in
  `tests/goldens/`. They are keyed by platform (`darwin-arm64`, `linux-aarch64`, `linux-x86_64`, plus the Warp version). The
  checkpoint track generator's fields are host-sensitive:
  `TRACKGEN_SKIP_HOST_SENSITIVE_GOLDENS=1` (CI sets it) excludes them.
- Doc examples are executed: `tests/test_doc_snippets.py`.

GPU: `ssh spark-home` (NVIDIA GB10, sm_121, aarch64; venv `~/venvs/trackgen` with
warp-lang 1.18.0 and torch cu130; test trees in `~/code/TrackGen-test/<name>/`; tools in
`~/code/TrackGen-test/tools/`, copies in `docs/library-readiness/handoff-0.3/tools/`). The
old `workstation` host is dead.
- Byte snapshots are per machine. Rebuild the baseline ON spark-home from the last
  validated tree (v0.2.0, or the previous dev tag), then compare:
  - `full_snapshot.py` (all generators);
  - `gate_snapshot.py` (160 gate configs x 3 calls, including a masked regeneration);
  - see `spark-validate.sh` for the sequence.
- Run the CUDA lane on the GPU with
  `TRACKGEN_REQUIRE_CUDA=1 python -m pytest -q -m cuda tests -ra`, and also the full suite.
  CUDA-only test helpers often need updating when result fields or behaviour change; tests
  that build arrays must pass `device=` explicitly, because the default is `cuda:0` on a
  GPU host.
- Use `ssh -o ConnectTimeout=10 ...`: a dead host otherwise hangs.

## Conventions
- Commits: author "Antoine Richard <antoiner@nvidia.com>". Use
  `git -c user.name=... -c user.email=...` and never change the global git config. Push
  only validated work; tag `v0.3.0.devN` at milestones.
- Match the surrounding code's style and comment density. Keep `import track_gen` cheap: it
  must not pull `pipeline`, `stages`, `plugin`, `testing` or `torch`.
- Every launch goes through `ctx`. No per-call allocation or host sync on hot paths. Keep
  the conformance kit (`track_gen.testing`) passing for every built-in.
- Ask the user before changes that alter what downstream racing code sees at runtime beyond
  the decisions above.
