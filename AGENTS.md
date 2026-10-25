# AGENTS.md — handoff for the 0.3 line (branch `v3`)

TrackGen is a pure NVIDIA Warp, CUDA-first batched race-track and gate-course generator, with
runtime utilities (collision, progress, localization, ...) and `Course`. Everything must stay
CUDA-graph capturable: no host syncs or allocations on hot paths, and every launch goes
through a `LaunchContext` (`ctx`). Read `docs/library-readiness/target-architecture.md` for
the architecture and the roadmap, and `CHANGELOG.md` for what shipped.

## State (2026-10-25)
- `main` = **v0.2.0** (released). The 0.1/0.2 migration records are retained under
  `docs/library-readiness/archive/`, outside the published site.
- The combined working candidate is **0.3.0rc3**. Immutable `v0.3.0rc2` remains
  `a42759a3f999a8668a200cddc3f04e0508478dd1`; never recreate an existing candidate tag.
  Draft PR #4 (`v3` to `main`) prepares 0.3.0. PR4 review findings #1–#3 and #5–#14
  were addressed in rc2; #4 (auto-spacing after `dataclasses.replace`) remains deferred
  until the user answers its review thread. Do not change that public contract.
- The release-blocking Repulsive repair and the racing documentation are combined.
  Finish the candidate validation/tag steps below, then STOP for the user's racing
  validation. Final release, merge to `main` and docs deployment need explicit approval.

## Release-blocking Repulsive repair (user decision, 2026-10-25)

The user has explicitly authorized fixing Repulsive before release and changing its
seeded outputs where necessary. The entire finished road, including width, must
clear every sampled disc and stay inside the outer wall. This is a hard validity
contract after resampling, relaxation and inflation, not merely a soft growth force.
No accepted optimizer/rescale/resample step may tunnel through the barriers. A run
with infeasible geometry or a missed target length must report invalid and respect
poison/keep. Geometric validity is separate from optional early-stop convergence:
a feasible smooth road may use the full iteration budget while its energy still
admits improvement. Only genuine small movement/stationarity may mark it frozen;
a rejected or quantized no-op step must not fake convergence.

Correct the evidenced numerical/roughness defects rather than only smoothing the
visualization. The implementation design and plan are under docs/superpowers/.
Preserve the other generators' outputs, RNG salts/draw order, lightweight imports,
ctx launches, fixed buffers and CPU/CUDA capture. The immutable rc2 tag stays unchanged;
new Repulsive fingerprints/statistics require correctness evidence and explicit review.
No final release is authorized; a validated next candidate needs racing validation.
The separate auto-spacing review finding #4 remains deferred.

## Documentation work and release progress
- The repaired solver and its parallel implementation have independent code reviews,
  full-road clearance evidence and same-machine comparisons. The numerical source is
  held; do not change it without a concrete reproduced failure. Only Repulsive reference
  fields/statistics may change. The fresh rc2 comparison passed 190 non-Repulsive full
  arrays and all 4,800 gate arrays byte for byte on spark-home.
- The required repaired CUDA lane passed 817 tests with no CUDA skips. The full
  spark-home suite passed 4,586 tests (four optional UI skips) in 2,196.36 s on
  `d33cf898ecdd47da6d804a088946d2ae733ace6f`. The final ten-point media revision
  changes no library, package metadata or golden files relative to that validated
  numerical source. Its 48 media tests also passed on spark-home with no skips.
- The final local CPU lane passed 3,750 tests (four optional UI skips), including
  executable doc examples. The strict docs build and full-site/scoped media browser
  reviews passed. Linux x86 reference capture, normal platform CI and the tag remain
  required gates; inspect actual GitHub/tag/log state when resuming.
- The approved reader overhaul is integrated. Historical reader pages live under the
  excluded `docs/library-readiness/archive/`; keep internal design/handoff/roadmap
  records. The public site explains current racing usage, all six generators, Course,
  collision/intersections, API/data and extension workflows. Centerlines remain prominent.
- Published media files have true light/dark counterparts. The Bezier/Hull/Polar
  storyboards explain actual selection/jitter, splines, whole-loop crossing fallback
  and sampling; a shared clip distinguishes raw count, target spacing, XPBD movement and
  final same-count arc redistribution. All nine movies and their video posters use native 16:9 canvases; solver clips show
  actual convergence. Bezier illustrates ten points on an 8×8 grid with explicitly
  labelled cell-bounded inputs; the production sampler remains unchanged.
  Committed per-asset provenance distinguishes the full capture from selective corrections.
  The Sphinx media registry verifies both files and hashes; never relabel old captures.
- Render sources before generating final media; see `contributing/rendering-assets.rst`.
  Source-only bootstrap uses `-D themed_media_registry=`. Final builds must use the
  default enforced registry. Do not deploy the site before final release approval.

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
   `docs/library-readiness/archive/upgrading/from-0.2.rst`.

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
  Run long jobs with their output redirected to a log, and read the log afterwards.
  The corrected solver's full suite takes about 37 minutes on this host. Its slow CPU
  E8/E64 Repulsive batch comparison can pause visible progress for many minutes.

## Next: release 0.3.0 (resume from actual Git/CI/tag state)
1. **Validate the combined candidate.** The required CPU lane, strict docs build and
   executable snippets are listed above. Finish both GitHub CPU lanes, strict docs and
   linkcheck on draft PR #4. For a Linux x86_64 Repulsive capture, the existing tests
   workflow accepts `capture_goldens=true`; extract only the 11 `track/repulsive/`
   fields from its fingerprint fragment. Never merge whole fragments or update other
   families/platforms. Follow capture with normal CI against all retained byte guards.
2. **Validate on spark-home.** Run the CUDA lane and full suite. Rebuild the previous
   validated baseline on that machine before byte comparison. Other generators and gate
   snapshots must stay byte-identical; Repulsive changes are authorized only with the
   independent full-road clearance, target/smoothness and statistics evidence. Keep
   machine/log details out of public reader pages and never commit host addresses.
3. **Cut `v0.3.0rc3` only after the required gates pass.** Package metadata and install
   pins are prepared for rc3. Commit any final validated changes, create an annotated
   tag, and push `v3` and the tag. Do not recreate immutable rc1/rc2 tags. The user's racing
   repos then pin `git+https://github.com/AntoineRichard/TrackGen.git@v0.3.0rc3`.
   Report the CI results and spark-home summary, then STOP until racing validation.
4. **Final release, only after the user's explicit go.** Set version `0.3.0`, date the
   CHANGELOG and update install pins. Push validated work, mark PR #4 ready, and merge
   with a MERGE COMMIT so development/candidate tags remain reachable from `main`.
   Annotate/push `v0.3.0` on that merge commit, create the GitHub release, and verify
   docs deployment. No final release is authorized by the present task.

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
