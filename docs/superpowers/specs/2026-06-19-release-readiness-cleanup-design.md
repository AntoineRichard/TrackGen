# Release-readiness cleanup: Warp-only package + explicit public API

**Date:** 2026-06-19
**Status:** Design — approved, pending spec review
**Branch:** `chore/release-readiness-cleanup`

## Goal

`track_gen` has been through many incremental changes on the way to a pure-Warp
pipeline. The runtime is already Warp-only at the import level, but the package
still *advertises* and *ships* a large torch "oracle" surface, a dead Fourier
generator, a deprecated compatibility shim, and accumulated repo cruft.

Make the repository release-ready by:

1. **Shipping only the Warp main path.** The installed wheel contains the Warp
   pipeline and nothing else. "The main path is the path."
2. **Exposing one explicit public API**, Newton-style: a curated, flat top-level
   namespace re-exported from a private `_src/` core; everything else private.
3. **Relocating the torch oracle into the test tree**, where it belongs — it is a
   verification fixture, not library code.
4. **Tidying the repo** so a fresh clone is clean.

Backward compatibility is explicitly **not** a goal: old import paths, the
deprecated shim, and re-exported internals may all break. No compatibility shims.

## Current state

The hot path is already cleanly separated *at the import level*:

- **Warp main path** (imports only `math`, `torch`, and each other):
  `warp_pipeline.py` (generation → resample → relax → inflate + all kernels),
  `warp_relax.py` (XPBD), `track_generator.py` (facade), `rng_utils.py` /
  `rng_kernels.py` (Warp RNG), `types.py` (`Track`, `TrackGenConfig`).
- **Torch oracle** (imported *only* by tests and by `__init__.py`):
  `geometry.py`, `inflation.py`, `relaxation.py`, `generators.py`. Every Warp
  kernel has a test asserting it matches its torch counterpart. The oracle is not
  on the runtime path.

Problems for release:

- `__init__.py` re-exports the entire oracle (14 geometry primitives, `relax`,
  generator classes) as if it were the library API.
- The **Fourier** generator was never ported to Warp; its only consumer
  (`viz/plot_ablations.py::figure3_bezier_vs_fourier`) already crashes because the
  facade rejects `generator != "bezier"`.
- `generate_tracks()` is a deprecated centerline-only shim.
- Repo cruft: 24 untracked `viz/out/*.png`, four `.worktrees/`, `.superpowers/`,
  and three uncommitted docs; `.gitignore` does not cover them.

## Target design

### Module layout

```
track_gen/                 # the shipped wheel = Warp main path only
  __init__.py              # curated public API + __all__ (re-exports from _src)
  _version.py              # __version__
  _src/                    # ALL Warp hot-path implementation (private core)
    __init__.py
    track_generator.py
    warp_pipeline.py
    warp_relax.py
    rng_utils.py
    rng_kernels.py
    types.py
  _experimental/           # kept & tracked, private, unsupported
    __init__.py
    fourier.py             # FourierCenterlineGenerator, self-contained

tests/
  _oracle/                 # torch reference impl — test fixtures, NOT shipped
    __init__.py
    geometry.py
    inflation.py
    relaxation.py
    generators.py          # Bézier oracle + Centerline / CenterlineGenerator base
  test_*.py                # imports repointed (see mapping below)
  conftest.py

benchmarks/                # left alone (import lines only)
viz/                       # left alone (import lines only)
docs/
```

Rationale for the split: the **wheel** should contain only what production needs
(the Warp path) plus a clearly-private, opt-in `_experimental`. The oracle is a
*test* artifact, so it moves under `tests/`. This is the Newton convention applied
faithfully: real code under a private `_src`, a thin curated public surface on top,
underscore = private.

### Public API

`track_gen/__init__.py` re-exports from `track_gen._src` and defines exactly:

```python
__all__ = [
    "TrackGenerator",
    "generate_tracks_warp",
    "generate_tracks_warp_graph",
    "TrackGenConfig",
    "Track",
    "PerEnvSeededRNG",
    "__version__",
]
```

- `PerEnvSeededRNG` stays **lazily** imported (via module `__getattr__`) so
  `import track_gen` does not require Warp until the RNG is actually touched.
- `generate_tracks_warp` / `generate_tracks_warp_graph` are re-exported as
  top-level functions (the README's "direct entry points"). The `warp_pipeline`
  module itself is **not** public — it lives at `track_gen._src.warp_pipeline`.

**Removed from the public surface entirely** (no shim, no alias):

- the deprecated `generate_tracks()` function — deleted;
- the 14 geometry primitives, `relax`, the `geometry`/`relaxation` modules, and the
  `*CenterlineGenerator` / `Centerline` classes — no longer re-exported (they move
  to `tests/_oracle` and `track_gen/_experimental`).

### Oracle → `tests/_oracle/`

The four torch modules move verbatim (logic unchanged) into `tests/_oracle/`:
`geometry.py`, `inflation.py`, `relaxation.py`, `generators.py` (Bézier generator +
the `Centerline` dataclass and `CenterlineGenerator` ABC). They import the shared
public dataclasses from `track_gen._src.types` (or the public `track_gen` names).

`tests/_oracle/` must be importable by the test suite as a package
(e.g. `from tests._oracle import geometry`). The exact mechanism — making `tests/`
a package (`tests/__init__.py`) and/or a root `conftest.py` ensuring the repo root
is on `sys.path` — is finalized in the implementation plan and **verified** by a
green `pytest -q` run; the package currently has no `tests/__init__.py`, no root
`conftest.py`, and no pytest config in `pyproject.toml`.

### Fourier → `track_gen/_experimental/`

`FourierCenterlineGenerator` moves to `track_gen/_experimental/fourier.py`, made
**self-contained** (it carries whatever minimal generator base it needs rather than
reaching into `tests/_oracle`, since `_experimental` ships and `tests/` does not).
The Fourier-only `TrackGenConfig` fields (`num_harmonics`, `decay_p`,
`num_centerline_samples`) stay in the config but are commented as experimental-only.
It is private (underscore) and absent from `__all__`.

### Tests

**Strategy: keep, repoint, don't rewrite — for now.** The oracle-backed equivalence
tests remain the regression net; only their import paths change. No test is
converted to a self-contained Warp-only check in this pass.

Import rewrites (deterministic mapping):

| Old import | New import |
|---|---|
| `track_gen.geometry` | `tests._oracle.geometry` |
| `track_gen.inflation` | `tests._oracle.inflation` |
| `track_gen.relaxation` | `tests._oracle.relaxation` |
| `track_gen.relax` | `tests._oracle.relaxation.relax` |
| `track_gen.generators` (Bézier / base) | `tests._oracle.generators` |
| `FourierCenterlineGenerator` | `track_gen._experimental.fourier` |
| `track_gen.warp_pipeline` | `track_gen._src.warp_pipeline` |
| `track_gen.warp_relax` | `track_gen._src.warp_relax` |
| `track_gen.track_generator` | `track_gen._src.track_generator` |
| `track_gen.rng_utils` | `track_gen._src.rng_utils` |
| `track_gen.rng_kernels` | `track_gen._src.rng_kernels` |
| `track_gen.types` | `track_gen._src.types` |
| `TrackGenerator`, `TrackGenConfig`, `Track`, `PerEnvSeededRNG`, `generate_tracks_warp(_graph)` | `from track_gen import …` (public) |

Test-file consequences:

- `test_public_api.py` / `test_public_api_full.py` — **rewritten** to assert the new
  curated surface (the public `__all__`, and that the oracle/Fourier are *not*
  importable from `track_gen`).
- `test_generate_tracks_compat.py` — **deleted** (its subject, the deprecated shim,
  is gone).
- `test_types.py` — its "leaf must not import siblings" assertion updated for the
  `_src` location.
- All other test files — import-line rewrites only.

### benchmarks/ + viz/

Logic untouched; **import lines only**:

- Warp scripts (`benchmark_pipeline.py`, `benchmark_yield_sweep.py`,
  `make_report.py`, `param_explorer.py`): `from track_gen._src import warp_pipeline
  as wpl` (preserves their `wpl.generate_tracks_warp` / `wpl.thickness` calls) and
  public types via `from track_gen import TrackGenConfig`.
- `benchmark_relaxation.py`: imports the torch oracle, now
  `from tests._oracle import relaxation, geometry, inflation` +
  `from tests._oracle.generators import BezierCenterlineGenerator`.

### Repo hygiene

- `.gitignore` += `viz/out/`, `.worktrees/`, `.superpowers/`.
- **Commit** the three untracked docs under `docs/`
  (`racetrack-generation-prior-art.md`, `track-generation-state-of-the-art.md`,
  `superpowers/plans/2026-06-17-warp-geometry-kernels.md`).
- Update `README.md` (Project layout, the "Direct pure-Warp entry points" snippet →
  top-level imports, Development/oracle wording) and `docs/ARCHITECTURE.md` (module
  layout, "torch as test oracle" now lives under `tests/_oracle`).

## Non-goals

- No rewrite of the test suite to self-contained Warp-only assertions (future work).
- No changes to `benchmarks/` or `viz/` beyond import lines.
- No behavioral change to the Warp pipeline, kernels, or numerics.
- No backward-compatibility shims or deprecation aliases.

## Risks & known smells (flagged, not hidden)

- **`benchmark_relaxation.py` will import test-only code** (`tests._oracle`). It
  benchmarks the torch relaxation backends, which are no longer library code. Left
  working per "leave benchmarks alone"; a candidate for future removal/rework.
- **`viz/plot_ablations.py::figure3_bezier_vs_fourier` stays broken** — it already
  crashes via the facade (`generator="fourier"` rejected). Untouched per "leave viz
  alone"; no new breakage introduced.
- **`tests/_oracle` import wiring** is the one mechanical risk; it must be proven by
  a green collection + run, not assumed.

## Verification

- `git mv` used for relocations so history is preserved where possible.
- `.venv/bin/python -m pytest -q` is **green before and after** on the Warp `cpu`
  device (the baseline must be captured first).
- `python -c "import track_gen; print(track_gen.__all__)"` shows exactly the curated
  surface, and importing the oracle/Fourier from `track_gen` fails.
- `benchmarks/` and `viz/` modules import without error (smoke import; full runs are
  out of scope but imports must resolve).

## Future work (captured, not done here)

The torch oracle was invaluable during build-up but pins the Warp path to a second
implementation. Once moved under `tests/`, scope a follow-up that replaces the
per-kernel equivalence tests with **self-contained, Warp-only** analytic/property
tests (e.g. closed-loop turning number ≈ 2π, arc-uniform spacing invariants,
thickness ≥ half-width, idempotent resample), letting the oracle be retired.
