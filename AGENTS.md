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

### Reaching the GPU machine
- `spark-home` is a `Host` alias in the `~/.ssh/config` of the user's Mac (key-based auth,
  default user). It works only from that Mac. This repository is public: never commit host
  addresses, keys or user names.
- The Codex sandbox blocks network access by default. `ssh`, `scp` and `rsync` need network
  access approved for the command, or running outside the sandbox.
- To send a tree, export a commit (never the working tree with stray files) and sync it:
  ```bash
  rm -rf /tmp/tg-<name> && mkdir -p /tmp/tg-<name> && git archive <ref> | tar -x -C /tmp/tg-<name>
  rsync -a --delete /tmp/tg-<name>/ spark-home:code/TrackGen-test/<name>/
  ssh -o ConnectTimeout=10 spark-home 'cd ~/code/TrackGen-test/<name> && PYTHONPATH=$PWD \
    ~/venvs/trackgen/bin/python -m pytest -q -p no:cacheprovider -m cuda tests -ra'
  ```
  Run long jobs (the full suite takes about 6 min; snapshots several minutes) with their
  output redirected to a log, and read the log afterwards.

## Next: release 0.3.0 (do the steps in order)
1. **CI on `v3`.** The test workflow runs only on pull requests and on pushes to `main`, so
   `v3` has never run on GitHub CI. Open a DRAFT pull request `v3` -> `main` with
   `gh pr create --draft --base main --head v3`. The body summarises dev1-dev4, links
   `docs/upgrading/from-0.2.rst`, and lists the test plan. The PR runs:
   - the CPU lanes on Linux x86_64 (Warp 1.18.0 and latest);
   - the docs build and the linkcheck.

   Fix every failure on `v3`, with the validation rules above. When a golden fails ONLY on
   CI, decide whether it is host sensitivity or a real regression before touching it. For
   0.2.0 the checkpoint path tracer's last-ulp differences were host sensitivity. Never
   re-baseline blindly.
2. **Cut `v0.3.0rc1`** once CI is green:
   - set `0.3.0rc1` in `pyproject.toml` and `track_gen/_version.py`, plus a CHANGELOG header;
   - commit "release: 0.3.0rc1", make an annotated tag `v0.3.0rc1`, push the branch and the tag;
   - validate the rc tree on spark-home: the CUDA lane and the full suite. Its outputs must
     equal dev4's, since only the version changed.
3. **Hand the rc to the user.** The racing repositories pin
   `git+https://github.com/AntoineRichard/TrackGen.git@v0.3.0rc1`. STOP here until the user
   confirms the racing validation passed. Fix anything they report on `v3`, then cut `rc2`
   the same way.
4. **Release, only after the user's explicit go:**
   - set `0.3.0` and date the CHANGELOG section `## 0.3.0 (<date>, tag v0.3.0)`, with a short
     summary that links the 0.2 -> 0.3 guide;
   - pin `@v0.3.0` in `README.md` and `docs/getting-started/installation.rst`;
   - commit "release: 0.3.0", push, and mark the PR ready;
   - merge it with a MERGE COMMIT (`gh pr merge --merge`), so the dev and rc tags stay
     reachable from `main`;
   - tag `v0.3.0` (annotated) on the merge commit and push it;
   - `gh release create v0.3.0` with notes that link the migration guide;
   - check that the docs workflow on `main` deployed.

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
