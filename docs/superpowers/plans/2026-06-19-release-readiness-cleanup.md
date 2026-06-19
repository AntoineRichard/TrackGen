# Release-readiness cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship only the Warp main path as `track_gen` behind one curated public API (Newton-style `_src/` private core), relocate the torch oracle into `tests/_oracle/`, move the Fourier generator into `track_gen/_experimental/`, and tidy the repo for release.

**Architecture:** A global, mechanical package reorg. Files move with `git mv`; every importer is repointed in the same task so the full test suite stays green at each task boundary. The torch oracle becomes test-only fixtures (not shipped); Fourier becomes a private, self-contained experimental module. No backward-compatibility shims, aliases, or deprecation paths.

**Tech Stack:** Python ≥ 3.10, PyTorch, NVIDIA Warp (`warp-lang`), pytest, setuptools.

## Global Constraints

- **No backward compatibility.** Old import paths, the deprecated `generate_tracks()` shim, and re-exported oracle internals may all break. No shims/aliases.
- **`import track_gen` must stay Warp-free.** Only `PerEnvSeededRNG` (lives in `rng_utils`, which does a top-level `import warp`) may require Warp; it is exposed lazily via module `__getattr__`. `warp_pipeline`/`warp_relax` import Warp only inside functions, so re-exporting `generate_tracks_warp(_graph)` at top level is safe.
- **Curated public API — exactly:** `TrackGenerator`, `generate_tracks_warp`, `generate_tracks_warp_graph`, `TrackGenConfig`, `Track`, `PerEnvSeededRNG`, `__version__`.
- **Verification gate every task:** `.venv/bin/python -m pytest -q` is green (run on the Warp `cpu` device; `cuda`-only asserts self-skip). Commit only on green.
- **Leave `benchmarks/` and `viz/` logic untouched** — edit import lines only.
- **License header** on every new `.py` file (copy verbatim):
  ```python
  # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  ```

---

### Task 0: Capture the green baseline

**Files:** none (verification only).

**Interfaces:**
- Produces: a recorded passing test count used as the safety net for every later task.

- [ ] **Step 1: Confirm branch**

Run: `git -C /home/antoiner/Documents/TrackGen branch --show-current`
Expected: `chore/release-readiness-cleanup`

- [ ] **Step 2: Run the full suite and record the result**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all green). Record the summary line (e.g. `N passed, M skipped`) — this is the number every later task must match (minus the deleted `test_generate_tracks_compat.py` cases, called out in Task 1).

- [ ] **Step 3: No commit** (read-only baseline).

---

### Task 1: Relocate the torch oracle to `tests/_oracle/` + curate the public API

Moves `geometry.py`, `inflation.py`, `relaxation.py`, `generators.py` out of the shipped package into `tests/_oracle/`, rewrites `track_gen/__init__.py` to the curated surface (warp modules still at their current top-level paths — they move in Task 2), deletes the deprecated `generate_tracks()` shim, and repoints every oracle importer. Suite green at the end.

**Files:**
- Create: `tests/__init__.py`, `tests/_oracle/__init__.py`
- Move (`git mv`): `track_gen/{geometry,inflation,relaxation,generators}.py` → `tests/_oracle/`
- Modify: `tests/_oracle/inflation.py`, `tests/_oracle/relaxation.py` (internal imports)
- Modify: `track_gen/__init__.py` (rewrite), `track_gen/track_generator.py` (drop shim)
- Rewrite: `tests/test_public_api.py`, `tests/test_public_api_full.py`
- Delete: `tests/test_generate_tracks_compat.py`
- Modify (mechanical repoint): all `tests/*.py`, `benchmarks/benchmark_relaxation.py`

**Interfaces:**
- Produces: `track_gen.__all__` == the curated 7 names; oracle importable as `tests._oracle.{geometry,inflation,relaxation,generators}`.

- [ ] **Step 1: Make `tests/` a package and create the oracle package dir**

```bash
cd /home/antoiner/Documents/TrackGen
mkdir -p tests/_oracle
printf '' > tests/__init__.py
printf '' > tests/_oracle/__init__.py
```

- [ ] **Step 2: Move the four oracle modules**

```bash
cd /home/antoiner/Documents/TrackGen
git mv track_gen/geometry.py    tests/_oracle/geometry.py
git mv track_gen/inflation.py   tests/_oracle/inflation.py
git mv track_gen/relaxation.py  tests/_oracle/relaxation.py
git mv track_gen/generators.py  tests/_oracle/generators.py
```

- [ ] **Step 3: Fix oracle-internal cross-imports that pointed at now-moved siblings**

`tests/_oracle/inflation.py` line 19 — change the leaf-types import (types is still at `track_gen/types.py` until Task 2):

```python
# OLD: from .types import Track, TrackGenConfig  # noqa: F401  (TrackGenConfig used for typing)
from track_gen.types import Track, TrackGenConfig  # noqa: F401  (TrackGenConfig used for typing)
```

`tests/_oracle/relaxation.py` line 19 — the torch relax oracle calls into the Warp solve (still top-level until Task 2):

```python
# OLD: from . import warp_relax
from track_gen import warp_relax
```

(`from . import geometry` in both, and `from .geometry import …` in `generators.py`, stay as-is — geometry now lives alongside them in `tests/_oracle/`.)

- [ ] **Step 4: Repoint every oracle importer (mechanical, indentation-preserving)**

Run this script (substring replacements are ordered so the combined warp+oracle line is split before the bare-name rules run):

```bash
cd /home/antoiner/Documents/TrackGen
.venv/bin/python - <<'PY'
import pathlib
REPL = [
    ("from track_gen.geometry import",   "from tests._oracle.geometry import"),
    ("from track_gen.generators import", "from tests._oracle.generators import"),
    ("from track_gen.relaxation import", "from tests._oracle.relaxation import"),
    ("from track_gen.inflation import",  "from tests._oracle.inflation import"),
    # combined warp+oracle import -> split into two lines (must precede bare rules)
    ("from track_gen import warp_pipeline as wpl, warp_relax, geometry",
     "from track_gen import warp_pipeline as wpl, warp_relax\nfrom tests._oracle import geometry"),
    # package-attribute oracle imports; substring also covers the ", x" combos
    ("from track_gen import geometry",   "from tests._oracle import geometry"),
    ("from track_gen import inflation",  "from tests._oracle import inflation"),
    ("from track_gen import relaxation", "from tests._oracle import relaxation"),
]
files = sorted(pathlib.Path("tests").glob("*.py")) + [pathlib.Path("benchmarks/benchmark_relaxation.py")]
for f in files:
    s = f.read_text(); o = s
    for a, b in REPL:
        s = s.replace(a, b)
    if s != o:
        f.write_text(s); print("repointed", f)
PY
```

- [ ] **Step 5: Drop the deprecated `generate_tracks()` shim**

In `track_gen/track_generator.py`: delete the `"generate_tracks",` entry from `__all__` (line 26) and delete the entire `def generate_tracks(...)` function (lines 93–122, from `def generate_tracks` through its final `return track.center`). The `import warnings` at line 15 becomes unused — delete it too.

- [ ] **Step 6: Rewrite `track_gen/__init__.py` to the curated surface (interim: warp at current top-level paths)**

Replace the whole file with:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""track_gen — GPU-batched race-track generator.

The public API is the Warp pipeline plus its result types. NVIDIA Warp is
imported lazily (only ``PerEnvSeededRNG`` needs it at import time), so
``import track_gen`` works in a Warp-free environment.
"""

__version__ = "0.1.0"

from .types import Track, TrackGenConfig
from .track_generator import TrackGenerator
from .warp_pipeline import generate_tracks_warp, generate_tracks_warp_graph


def __getattr__(name):
    # rng_utils imports NVIDIA Warp at module load; defer it so `import track_gen`
    # stays Warp-free.
    if name == "PerEnvSeededRNG":
        from .rng_utils import PerEnvSeededRNG
        return PerEnvSeededRNG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

- [ ] **Step 7: Rewrite `tests/test_public_api.py` to lock the new surface**

Replace the whole file with:

```python
import track_gen


def test_public_api_surface_is_exactly_curated():
    assert set(track_gen.__all__) == {
        "TrackGenerator",
        "generate_tracks_warp",
        "generate_tracks_warp_graph",
        "TrackGenConfig",
        "Track",
        "PerEnvSeededRNG",
        "__version__",
    }


def test_public_names_are_accessible():
    for name in track_gen.__all__:
        assert hasattr(track_gen, name), f"track_gen.{name} missing"


def test_oracle_internals_are_not_public():
    # The torch oracle moved to tests/_oracle and is no longer part of the package.
    for gone in ("geometry", "relaxation", "inflation", "generators", "relax",
                 "safe_normalize", "polygon_area", "Centerline", "warp_pipeline"):
        assert not hasattr(track_gen, gone), f"track_gen.{gone} should not be public"
```

- [ ] **Step 8: Rewrite `tests/test_public_api_full.py` (drop `Centerline`, add the entry points)**

Replace the whole file with:

```python
def test_full_public_api_is_reexported():
    import track_gen

    for name in ("PerEnvSeededRNG", "TrackGenerator", "generate_tracks_warp",
                 "generate_tracks_warp_graph", "TrackGenConfig", "Track"):
        assert hasattr(track_gen, name), f"track_gen.{name} is not exported"
```

- [ ] **Step 9: Delete the deprecated-shim test**

```bash
cd /home/antoiner/Documents/TrackGen
git rm tests/test_generate_tracks_compat.py
```

- [ ] **Step 10: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS. Pass count == baseline minus the `test_generate_tracks_compat.py` cases (it had 2 test functions). If collection fails with `ModuleNotFoundError: tests`, confirm `tests/__init__.py` exists; if it still fails, add a repo-root `conftest.py` containing `import os, sys; sys.path.insert(0, os.path.dirname(__file__))` and re-run.

- [ ] **Step 11: Sanity-check the Warp-free import and the curated surface**

Run: `.venv/bin/python -c "import track_gen; print(sorted(track_gen.__all__)); assert not hasattr(track_gen, 'geometry')"`
Expected: prints the 7 curated names; no error.

- [ ] **Step 12: Commit**

```bash
cd /home/antoiner/Documents/TrackGen
git add -A
git commit -m "refactor: move torch oracle to tests/_oracle, curate public API, drop generate_tracks shim"
```

---

### Task 2: Move the Warp hot path into `track_gen/_src/`

Relocates the six hot-path modules into a private `_src/` core, adds `_version.py`, repoints `__init__.py` and every importer (tests, benchmarks, viz, and the two `tests/_oracle/` files that reference the moved `types`/`warp_relax`), and fixes `pyproject.toml` so the subpackages ship in the wheel.

**Files:**
- Create: `track_gen/_src/__init__.py`, `track_gen/_version.py`
- Move (`git mv`): `track_gen/{types,track_generator,warp_pipeline,warp_relax,rng_utils,rng_kernels}.py` → `track_gen/_src/`
- Modify: `track_gen/__init__.py`, `pyproject.toml`, `tests/test_types.py`
- Modify (mechanical repoint): `tests/*.py`, `tests/_oracle/*.py`, `benchmarks/*.py`, `viz/*.py`

**Interfaces:**
- Consumes: curated `__init__` from Task 1.
- Produces: hot path importable as `track_gen._src.{types,track_generator,warp_pipeline,warp_relax,rng_utils,rng_kernels}`; `track_gen._version.__version__`.

- [ ] **Step 1: Create the private core and move the six modules**

Their cross-imports are relative (`from .types import …`, `from . import warp_relax`, `from .rng_kernels import …`) and move together, so they need no internal edits.

```bash
cd /home/antoiner/Documents/TrackGen
mkdir -p track_gen/_src
printf '' > track_gen/_src/__init__.py
git mv track_gen/types.py           track_gen/_src/types.py
git mv track_gen/track_generator.py track_gen/_src/track_generator.py
git mv track_gen/warp_pipeline.py   track_gen/_src/warp_pipeline.py
git mv track_gen/warp_relax.py      track_gen/_src/warp_relax.py
git mv track_gen/rng_utils.py       track_gen/_src/rng_utils.py
git mv track_gen/rng_kernels.py     track_gen/_src/rng_kernels.py
```

- [ ] **Step 2: Add `track_gen/_version.py`**

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__version__ = "0.1.0"
```

- [ ] **Step 3: Repoint `track_gen/__init__.py` at `_src` and `_version`**

Replace the import lines and drop the inline `__version__` so the file reads:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""track_gen — GPU-batched race-track generator.

The public API is the Warp pipeline plus its result types. The heavy Warp
implementation lives in the private ``track_gen._src`` subpackage; this module
re-exports the supported surface. NVIDIA Warp is imported lazily (only
``PerEnvSeededRNG`` needs it at import time), so ``import track_gen`` works in a
Warp-free environment.
"""

from ._version import __version__
from ._src.types import Track, TrackGenConfig
from ._src.track_generator import TrackGenerator
from ._src.warp_pipeline import generate_tracks_warp, generate_tracks_warp_graph


def __getattr__(name):
    # _src.rng_utils imports NVIDIA Warp at module load; defer it so
    # `import track_gen` stays Warp-free.
    if name == "PerEnvSeededRNG":
        from ._src.rng_utils import PerEnvSeededRNG
        return PerEnvSeededRNG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

- [ ] **Step 4: Repoint every other importer (tests, oracle, benchmarks, viz)**

```bash
cd /home/antoiner/Documents/TrackGen
.venv/bin/python - <<'PY'
import pathlib
REPL = [
    ("from track_gen import warp_pipeline as", "from track_gen._src import warp_pipeline as"),
    ("from track_gen import warp_relax",       "from track_gen._src import warp_relax"),
    ("from track_gen.types import",            "from track_gen._src.types import"),
    ("import track_gen.types as",              "import track_gen._src.types as"),
    ("from track_gen.track_generator import",  "from track_gen._src.track_generator import"),
    ("from track_gen.rng_utils import",        "from track_gen._src.rng_utils import"),
]
dirs = ["tests", "tests/_oracle", "benchmarks", "viz"]
files = [p for d in dirs for p in pathlib.Path(d).glob("*.py")]
for f in files:
    s = f.read_text(); o = s
    for a, b in REPL:
        s = s.replace(a, b)
    if s != o:
        f.write_text(s); print("repointed", f)
PY
```

- [ ] **Step 5: Update the leaf-purity test in `tests/test_types.py`**

The mechanical script (Step 4) already changed line 5 and the inline imports to `from track_gen._src.types import …` and line 112 to `import track_gen._src.types as t`. Now replace the body of `test_types_module_has_no_intra_package_imports` (lines 110–116) with the new sibling list:

```python
def test_types_module_has_no_intra_package_imports():
    # The leaf must not import the Warp-touching siblings (keeps the leaf
    # dataclasses and `import track_gen` Warp-free).
    import track_gen._src.types as t

    src = open(t.__file__).read()
    for forbidden in ("from .track_generator", "from .warp_pipeline",
                      "from .warp_relax", "from .rng_utils", "from .rng_kernels",
                      "import warp"):
        assert forbidden not in src, f"types.py must not contain '{forbidden}'"
```

- [ ] **Step 6: Make the subpackages ship in the wheel**

In `pyproject.toml`, replace the `[tool.setuptools]` block:

```toml
# OLD:
# [tool.setuptools]
# packages = ["track_gen"]

[tool.setuptools.packages.find]
include = ["track_gen*"]
```

This auto-includes `track_gen`, `track_gen._src`, and `track_gen._experimental` while excluding `tests`, `benchmarks`, and `viz`.

- [ ] **Step 7: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, same count as end of Task 1.

- [ ] **Step 8: Verify the package still imports and the wheel-packaging is sane**

```bash
cd /home/antoiner/Documents/TrackGen
.venv/bin/python -c "import track_gen; track_gen.generate_tracks_warp; track_gen.__version__; print('ok', track_gen.__version__)"
.venv/bin/pip install -e . -q && .venv/bin/python -c "import track_gen._src.warp_pipeline, track_gen; print('reinstall ok')"
```
Expected: `ok 0.1.0` then `reinstall ok`.

- [ ] **Step 9: Commit**

```bash
cd /home/antoiner/Documents/TrackGen
git add -A
git commit -m "refactor: move Warp hot path into track_gen/_src, add _version, fix wheel packaging"
```

---

### Task 3: Extract Fourier into `track_gen/_experimental/`

Pulls `FourierCenterlineGenerator` out of `tests/_oracle/generators.py` into a private, self-contained shipped module (it carries its own `Centerline`, `CenterlineGenerator` base, and the small torch helpers it needs), repoints its one test consumer, and marks the Fourier config fields experimental.

**Files:**
- Create: `track_gen/_experimental/__init__.py`, `track_gen/_experimental/fourier.py`
- Modify: `tests/_oracle/generators.py` (remove the Fourier class), `tests/test_generators.py` (repoint Fourier imports), `track_gen/_src/types.py` (comment Fourier fields)

**Interfaces:**
- Consumes: `TrackGenConfig` Fourier fields (`num_harmonics`, `decay_p`, `amplitude`, `num_centerline_samples`, `scale`, `turning_tol`) and a `PerEnvSeededRNG`-like `rng` with `sample_normal_torch`.
- Produces: `track_gen._experimental.fourier.FourierCenterlineGenerator`, `.Centerline`, `.CenterlineGenerator`.

- [ ] **Step 1: Create the experimental package**

```bash
cd /home/antoiner/Documents/TrackGen
mkdir -p track_gen/_experimental
printf '' > track_gen/_experimental/__init__.py
```

- [ ] **Step 2: Write `track_gen/_experimental/fourier.py` (self-contained)**

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental truncated-Fourier centerline generator.

NOT part of the supported Warp pipeline: it was never ported to Warp, and the
TrackGenerator facade rejects ``generator != "bezier"``. Kept — private and
self-contained (torch only) — for experimentation. Vendors the two tiny torch
geometry helpers it needs so it has no dependency on the test oracle.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass

import torch


@dataclass
class Centerline:
    """A closed, ordered, dense centerline batch (points [E, M, 2], valid [E])."""

    points: torch.Tensor
    valid: torch.Tensor


class CenterlineGenerator(abc.ABC):
    """Interface a centerline generator implements: one Centerline per env id."""

    @abc.abstractmethod
    def generate(self, ids: torch.Tensor) -> Centerline:
        raise NotImplementedError


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / norm.clamp_min(eps)


def _turning_number(points: torch.Tensor) -> torch.Tensor:
    """Signed total turning of a closed polygon (±2π for a simple loop)."""
    nxt = torch.roll(points, shifts=-1, dims=1)
    dirs = _safe_normalize(nxt - points)
    theta = torch.atan2(dirs[..., 1], dirs[..., 0])
    dtheta = theta - torch.roll(theta, shifts=1, dims=1)
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))  # wrap into (-pi, pi]
    return dtheta.sum(dim=1)


class FourierCenterlineGenerator(CenterlineGenerator):
    """Truncated-Fourier centerline generator: smooth-by-construction closed curves."""

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.device = config.device
        self.K = config.num_harmonics
        self.M = config.num_centerline_samples

        # Dense parameter grid over [0, 2*pi) (endpoint excluded so the loop closes cleanly).
        t = torch.linspace(0.0, 2.0 * math.pi, self.M + 1, device=self.device)[:-1]  # [M]
        self.t = t
        k = torch.arange(1, self.K + 1, device=self.device, dtype=torch.float32)  # [K]
        self.cos_kt = torch.cos(k.unsqueeze(1) * t.unsqueeze(0))  # [K, M]
        self.sin_kt = torch.sin(k.unsqueeze(1) * t.unsqueeze(0))  # [K, M]
        # Per-harmonic std: amplitude / k**decay_p.
        self.std_k = config.amplitude / (k**config.decay_p)  # [K]

    def generate(self, ids: torch.Tensor) -> Centerline:
        # Sample standard normals (float args), then scale by the per-harmonic decay in torch.
        # NOTE: do NOT pass a tensor std into sample_normal_torch (warp dispatch rejects
        # float-mean / tensor-std, and only honors a per-env scalar std).
        a = self.rng.sample_normal_torch(0.0, 1.0, (self.K, 2), ids=ids)  # [E, K, 2]
        b = self.rng.sample_normal_torch(0.0, 1.0, (self.K, 2), ids=ids)  # [E, K, 2]
        a = a * self.std_k.view(1, self.K, 1)
        b = b * self.std_k.view(1, self.K, 1)

        # c(t) = sum_k a_k cos(k t) + b_k sin(k t); c0 omitted (cancels under mean-centering).
        curve = torch.einsum("ekd,km->emd", a, self.cos_kt) + torch.einsum("ekd,km->emd", b, self.sin_kt)

        curve = curve - curve.mean(dim=1, keepdim=True)
        bbox = curve.amax(dim=1) - curve.amin(dim=1)  # [E, 2]
        longest = bbox.amax(dim=1, keepdim=True).clamp_min(1e-8)  # [E, 1]
        curve = curve * (self.config.scale / longest).unsqueeze(1)

        # valid via the turning-number safety net for rare low-K crossings.
        turn = _turning_number(curve)  # [E]
        valid = (turn.abs() - 2.0 * math.pi).abs() <= self.config.turning_tol

        return Centerline(points=curve, valid=valid)
```

- [ ] **Step 3: Remove the Fourier class from the oracle**

In `tests/_oracle/generators.py`, delete the entire `class FourierCenterlineGenerator(CenterlineGenerator):` block (originally lines 297–337, from that `class` line through its final `return Centerline(points=curve, valid=valid)`). Leave `Centerline`, `CenterlineGenerator`, `bernstein`, and `BezierCenterlineGenerator` intact.

- [ ] **Step 4: Repoint the Fourier import in `tests/test_generators.py`**

After Task 1's repoint, line 309 reads `from tests._oracle.generators import FourierCenterlineGenerator`. Change it to:

```python
from track_gen._experimental.fourier import FourierCenterlineGenerator
```

- [ ] **Step 5: Fix `test_module_exposes_both_generators` (its "both in one module" premise is gone)**

Replace that test (originally lines 385–394) with one that checks each generator against the base in its new home:

```python
def test_generators_live_in_their_split_homes():
    from tests._oracle.generators import (
        BezierCenterlineGenerator,
        CenterlineGenerator as OracleBase,
    )
    from track_gen._experimental.fourier import (
        FourierCenterlineGenerator,
        CenterlineGenerator as FourierBase,
    )

    assert issubclass(BezierCenterlineGenerator, OracleBase)
    assert issubclass(FourierCenterlineGenerator, FourierBase)
```

- [ ] **Step 6: Mark the Fourier config fields experimental**

In `track_gen/_src/types.py`, the Fourier params block (around the current `# --- Fourier params ---` comment and `num_centerline_samples`) — change the section comment to make their status explicit (field defaults unchanged):

```python
    # --- Fourier params (EXPERIMENTAL: consumed only by track_gen._experimental.fourier;
    #     the supported Warp pipeline ignores them) ---
```

- [ ] **Step 7: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, same count as end of Task 2.

- [ ] **Step 8: Verify the experimental module imports standalone (no oracle dependency)**

Run: `.venv/bin/python -c "from track_gen._experimental.fourier import FourierCenterlineGenerator, Centerline, CenterlineGenerator; print('fourier ok')"`
Expected: `fourier ok`.

- [ ] **Step 9: Commit**

```bash
cd /home/antoiner/Documents/TrackGen
git add -A
git commit -m "refactor: extract Fourier generator into private track_gen/_experimental"
```

---

### Task 4: Repo hygiene + docs

`.gitignore` strays, commit the three reference docs, and update `README.md` + `docs/ARCHITECTURE.md` to the new layout.

**Files:**
- Modify: `.gitignore`, `README.md`, `docs/ARCHITECTURE.md`
- Add to git: `docs/racetrack-generation-prior-art.md`, `docs/track-generation-state-of-the-art.md`, `docs/superpowers/plans/2026-06-17-warp-geometry-kernels.md`

**Interfaces:** none (docs/hygiene only).

- [ ] **Step 1: Ignore the strays**

Append to `.gitignore`:

```
viz/out/
.worktrees/
.superpowers/
```

(`__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.venv/` are already ignored. The git worktrees under `.worktrees/` are left on disk; pruning them with `git worktree remove` is a manual, out-of-scope step.)

- [ ] **Step 2: Commit the three reference docs**

```bash
cd /home/antoiner/Documents/TrackGen
git add docs/racetrack-generation-prior-art.md \
        docs/track-generation-state-of-the-art.md \
        docs/superpowers/plans/2026-06-17-warp-geometry-kernels.md
```

- [ ] **Step 3: Update `README.md`**

Make these specific edits:
1. **Direct entry points** snippet (currently `from track_gen import warp_pipeline as wpl` / `wpl.generate_tracks_warp(...)`): change to the public top-level imports —
   ```python
   from track_gen import generate_tracks_warp, generate_tracks_warp_graph
   track = generate_tracks_warp(config, seeds)
   captured = generate_tracks_warp_graph(config, seeds_template)
   ```
2. **Project layout** block: replace the file list with the new tree —
   ```
   track_gen/
     __init__.py        # curated public API (TrackGenerator, generate_tracks_warp[_graph],
                        #   TrackGenConfig, Track, PerEnvSeededRNG, __version__)
     _version.py
     _src/              # the Warp pipeline (private core)
       warp_pipeline.py warp_relax.py track_generator.py types.py rng_utils.py rng_kernels.py
     _experimental/     # Fourier generator (unsupported, not on the Warp path)
   tests/
     _oracle/           # torch reference impl used to validate the Warp kernels
     test_*.py
   benchmarks/  viz/  docs/
   ```
3. **Architecture** paragraph: change "The existing torch implementation (`geometry`/`inflation`/`generators`/`relaxation`) is retained as the verification oracle" to note it now lives under `tests/_oracle/` and is not part of the shipped package.
4. Remove the line "Only the **Bézier** generator is supported … the legacy Fourier generator is not part of the Warp pipeline" → restate that Fourier lives in `track_gen._experimental` and is unsupported.

- [ ] **Step 4: Update `docs/ARCHITECTURE.md`**

Update the module-layout section and the "torch as test oracle" section to reflect: hot path under `track_gen/_src/`; oracle under `tests/_oracle/` (imported by tests as `tests._oracle.*`, not shipped); Fourier under `track_gen/_experimental/`. Update any `track_gen.<module>` import paths mentioned in prose to their new `track_gen._src.<module>` / `tests._oracle.<module>` homes.

- [ ] **Step 5: Confirm the suite is still green (docs/ignore changes are inert)**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, same count as end of Task 3.

- [ ] **Step 6: Commit**

```bash
cd /home/antoiner/Documents/TrackGen
git add -A
git commit -m "docs: update README + ARCHITECTURE for _src/_experimental/_oracle layout; gitignore strays"
```

---

## Self-Review

**1. Spec coverage** (against `docs/superpowers/specs/2026-06-19-release-readiness-cleanup-design.md`):
- Wheel = Warp main path only → Task 2 (`_src` move) + Task 6-pyproject fix. ✓
- Newton `_src` private core → Task 2. ✓
- Curated public API + `__all__`, lazy `PerEnvSeededRNG`, drop `generate_tracks()` → Task 1 (Steps 5–8). ✓
- Oracle → `tests/_oracle/`, importable, green pytest → Task 1 (Steps 1–4, 10). ✓
- Fourier → `track_gen/_experimental/`, self-contained → Task 3. ✓
- Repoint all tests + benchmarks + viz; rewrite `test_public_api*`; delete `test_generate_tracks_compat` → Tasks 1–2 (mechanical scripts) + Task 1 Steps 7–9. ✓
- Hygiene (`.gitignore`, 3 docs, README + ARCHITECTURE) → Task 4. ✓
- Tests kept, only repointed; "future work" note → in the spec; no test rewrites here except the API/leaf tests whose subject changed. ✓
- Verification gate green before/after → Task 0 baseline + per-task pytest. ✓

**2. Placeholder scan:** No `TBD`/`TODO`/"handle edge cases". The one conditional (Step 10 fallback conftest) is an explicit, complete remedy, not a placeholder.

**3. Type/name consistency:** `__all__` is identical across Task 1 Step 6, Task 2 Step 3, and `test_public_api` (Task 1 Step 7). The repoint mapping in Task 1/Task 2 scripts matches the spec's table. `FourierCenterlineGenerator` / `Centerline` / `CenterlineGenerator` names match between `_experimental/fourier.py` (Task 3 Step 2) and its test (Task 3 Steps 4–5).

**Known smells (carried from the spec, intentional):** `benchmark_relaxation.py` now imports `tests._oracle` (a benchmark leaning on test code); `viz/plot_ablations.py::figure3_bezier_vs_fourier` stays broken (pre-existing). Both are left as-is per "leave benchmarks/viz alone."
