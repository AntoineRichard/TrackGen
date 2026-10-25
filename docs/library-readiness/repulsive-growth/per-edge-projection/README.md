# Rejected per-edge projection experiment

This internal archive records an isolated Repulsive constraint experiment. It is **not adopted**, is outside the published `track_gen` package, and is excluded from the public documentation build and default test collection. No runtime module imports it. The rest of the Repulsive repair continues independently.

The proposed direction was `P H P grad`, where `P` projects onto the tangent space of all reference edge-length constraints. A bounded damped Newton retraction restored those constraints after an update. The helper preserves each reference edge ratio rather than enforcing globally equal spacing. Independent dense-reference and finite-precision tests establish the helper's local mathematical behavior, not generator success.

The complete controlled probe used the same solver source SHA-256 `d405b7c216b8ab7dc34499c1f4ef03f7cebce0fdba86dc93801bbf791efd23ce`, seeds 7/23/41, target multipliers 3.0–3.5 and 210 iterations. The [baseline](controlled-baseline.json) reached its target lengths but did not converge; its final edge ratios were 1.027/1.004/1.003. The [candidate](controlled-candidate.json) kept edge ratios 1.00012/1.00044/1.00041 yet all three rows blocked at lengths 4.332/5.625/6.370, below targets 8.280/8.392/9.144. The recorded comparison used the then-current assumption that a row must freeze to be valid. The later contract separates geometric validity from optional convergence, so the recorded `gen_valid=0` for baseline seeds 23/41 is not evidence of a geometry failure. The per-edge rejection rests on the candidate’s actual short target lengths and sharper bends, rather than those historical validity bits. This narrow result does not prove per-edge projection can never be useful.

[Design, proof references and limitations](design-and-evidence.md) explain the cyclic solve, precision policy and earlier probes. The original process-local integration probe was `repulsive_per_edge_probe.py` (a historical local artifact); it depended on that historical solver source. Only small scalar result records are retained here, with no captured arrays.

The verification file uses a `verify_` name so default pytest discovery does not collect this rejected experiment. Run it explicitly from the repository root. They load the sibling helper by file path and use the current repository's execution/test utilities. Activate the project test environment first. A local CPU run is:

```sh
PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider -p tests.conftest \
  -m 'not cuda' \
  docs/library-readiness/repulsive-growth/per-edge-projection/verify_repulsive_constraints.py
```

The explicit `-p tests.conftest` loads the shared device gating because this excluded archive lies outside `tests/`. Omitting it is unsupported for this archived test. On the coordinated CUDA test host, use that host's Python environment and the same explicit test path/plugin with `TRACKGEN_REQUIRE_CUDA=1` and `-m cuda`. CUDA-marked cases include projection, retraction, frozen-row behavior and captured replay. This archival disposition was verified on CPU only; CUDA execution remains with the root's coordinated validation.

Archive verification after the move: **22 passed, 12 CUDA skipped** with the shared configuration loaded. No production or default-suite import references the archived helper.
