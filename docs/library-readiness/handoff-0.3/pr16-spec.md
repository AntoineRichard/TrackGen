# PR16 spec — 1.0 cleanup (roadmap PR16)

Worktree: /Users/antoiner/code/TrackGen/.claude/worktrees/v2 (branch v2). Base: fe43a28
(release 0.2.0; PR #3 merges v2 into main as v0.2.0, the warning release for the 1.0
removals). Version after 0.2.0: the 1.0 line (the design proposes 1.0.0.devN / 1.0.0rcN).

The spec is the roadmap entry "### PR16" in docs/library-readiness/target-architecture.md
(scope incl. the "From PR13c / PR13d" removals, fixes, test gate, risk) plus everything the
0.2.0 CHANGELOG / migration guide (docs/upgrading/from-0.1.rst, its "1.0 removals" table)
promised for 1.0, the deferred items recorded in the roadmap (e.g. the nitpick allowlist
entries, planner messages citing dev releases, the legacy readback sizing — not planned),
and decision records D1-D14. Read them fully.

Context:
- Downstream racing repositories consume Course / the utilities and will pin v0.2.0 while
  1.0 lands; 1.0 is a breaking release, but every removal must have had a warning in 0.2.0
  (verify each one: if something slated for removal never warned in 0.2.0, it cannot be
  removed in 1.0 — flag it).
- Deliberate output changes in 1.0 (each needs the user's explicit sign-off): the built-ins'
  RNG keying switches to `ops.rng.init` with hashed salts (every track changes for a given
  seed; goldens re-baselined ONCE in one commit; `legacy_rng=True` kept for one minor
  release), `on_invalid="poison"` as the default, `edgy` dynamic (its in-kernel transform
  differs from the host float64 one), and anything else the design finds.
- GPU validation runs on spark-home (NVIDIA GB10): byte snapshots must show ONLY the
  declared output changes (e.g. with legacy_rng=True every output equal to 0.2.0).

## Deliverables (from the roadmap)
1. Built-ins switch to `ops.rng.init` with hashed salts; `legacy_rng=True` for one minor
   release; goldens re-baselined once.
2. `on_invalid="poison"` default.
3. Configs `frozen=True`; Track / GateSequence new fields required.
4. Remove the shims and deprecated surface: `set_capturing`, the `_CAPTURING` names, the
   `ctx=None` shims, the PEP 562 re-export, the legacy registries and the 5-arg adapter, the
   dead config fields, the built-in generators' scalar kernels and frozen legacy module
   functions (`generate_*_warp`, `*_alloc_scratch`, `apply_z_profile*`, the float sampling
   helpers; the `*_per_env` helpers stay), the deprecated `.params` alias, `style_sampling`
   and its ranges, `validity.gate_default`'s inert `gate_width` / `check_crossing`, the
   deprecated GateGeneratorSpec / warp_gate helpers, anything else the 0.2 migration guide
   lists for 1.0.
5. `edgy` dynamic.
6. Warp floor: already 1.18 (the roadmap's 1.16 is superseded); conditional lowerings on by
   default where supported (verify); CPU APIC self-graph opt-in.
7. Docs: migration guide gains "0.2 -> 1.0"; every page updated; CHANGELOG.

## Test gate
The full suite at the floor and at latest on both lanes; goldens re-baselined in one commit;
legacy_rng=True reproduces 0.2.0 byte for byte (GPU snapshots vs v0.2.0 on the GB10).

## Constraints
Do not commit or push. Prefer splitting PR16 into reviewable parts (the design proposes the
split, e.g. PR16a removals, PR16b RNG + goldens, PR16c defaults + frozen configs).
