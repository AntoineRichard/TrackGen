# Gate Sequence Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public, fixed-batch gate sequence generator that uses native first-stage generator anchors without changing existing track generation behavior.

**Architecture:** Add public gate dataclasses in the existing leaf `types.py`, a separate gate registry/facade, common Warp kernels for gate pose finalization and validity, and native gate generator modules for the five standard generator families. Existing `TrackGenerator`, `TrackGenConfig`, `generator_registry`, relaxation, and inflation stay untouched except for public re-exports and tests.

**Tech Stack:** Python dataclasses, NVIDIA Warp kernels, existing `PerEnvSeededRNG`, pytest, torch only in tests.

---

## File Structure

- Modify `track_gen/_src/types.py`: add `GateGenConfig` and `GateSequence` next to the existing leaf dataclasses.
- Create `track_gen/_src/gate_generator_registry.py`: gate-only registry and `GateGeneratorSpec`.
- Create `track_gen/_src/warp_gate.py`: common ordering, normalization, tangent, endpoint, and validity kernels.
- Create `track_gen/_src/warp_generate_gates.py`: Bezier and Hull gate-native extractors using existing sampled point kernels.
- Create `track_gen/_src/warp_generate_polar_gates.py`: Polar gate-native extractor using polar controls.
- Create `track_gen/_src/warp_generate_voronoi_gates.py`: Voronoi gate-native extractor using selected anchors.
- Create `track_gen/_src/warp_generate_checkpoint_gates.py`: Checkpoint gate-native extractor using sampled checkpoints.
- Create `track_gen/_src/gate_generator.py`: public fixed-batch facade, CUDA graph handling, output reuse.
- Modify `track_gen/__init__.py`: re-export new public names.
- Modify `tests/test_public_api.py` and `tests/test_public_api_full.py`: include new public names.
- Modify `tests/test_types.py`: cover gate config/result defaults and validation.
- Create `tests/test_gate_generator_registry.py`: registry resolution and unsupported ordering checks.
- Create `tests/test_warp_gate.py`: common gate finalization/validity tests.
- Create `tests/test_gate_generator.py`: facade behavior and standard generator smoke tests.
- Create `tests/test_gate_track_compat.py`: fixed-seed track behavior stays unchanged when gate generation is imported and run.

## Task 1: Public Gate Types

**Files:**
- Modify: `track_gen/_src/types.py`
- Modify: `track_gen/__init__.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_public_api_full.py`
- Modify: `tests/test_types.py`

- [ ] **Step 1: Write failing public API and type tests**

Add these assertions to `tests/test_public_api.py`:

```python
def test_public_api_surface_is_exactly_curated():
    assert set(track_gen.__all__) == {
        "TrackGenerator",
        "TrackGenConfig",
        "Track",
        "GateGenerator",
        "GateGenConfig",
        "GateSequence",
        "PerEnvSeededRNG",
        "__version__",
    }
```

Add gate names to `tests/test_public_api_full.py`:

```python
def test_full_public_api_is_reexported():
    import track_gen

    for name in (
        "PerEnvSeededRNG",
        "TrackGenerator",
        "TrackGenConfig",
        "Track",
        "GateGenerator",
        "GateGenConfig",
        "GateSequence",
    ):
        assert hasattr(track_gen, name), f"track_gen.{name} is not exported"
```

Append these tests to `tests/test_types.py`:

```python
def test_gate_config_defaults_instantiate():
    from track_gen._src.types import GateGenConfig

    cfg = GateGenConfig()
    assert cfg.generator == "bezier"
    assert cfg.device == "cpu"
    assert cfg.num_envs == 1
    assert cfg.min_gates == 4
    assert cfg.max_gates == 32
    assert cfg.gate_radius == 0.025
    assert cfg.gate_width == 0.0
    assert cfg.gate_ordering == "ccw"
    assert cfg.max_num_points == 13
    assert cfg.polar_num_knots == 12
    assert cfg.voronoi_control_points == 18
    assert cfg.checkpoint_count == 12


def test_gate_config_validates_basic_bounds():
    from track_gen._src.types import GateGenConfig

    with pytest.raises(ValueError, match="min_gates"):
        GateGenConfig(min_gates=1)
    with pytest.raises(ValueError, match="max_gates"):
        GateGenConfig(min_gates=6, max_gates=5)
    with pytest.raises(ValueError, match="gate_radius"):
        GateGenConfig(gate_radius=-1.0)
    with pytest.raises(ValueError, match="gate_width"):
        GateGenConfig(gate_width=-1.0)
    with pytest.raises(ValueError, match="gate_ordering"):
        GateGenConfig(gate_ordering="spiral")


def test_gate_sequence_construct_from_warp_arrays_and_clone():
    import warp as wp
    from track_gen._src.types import GateSequence

    wp.init()
    E, G = 2, 8
    gates = GateSequence(
        position=wp.zeros(E * G, dtype=wp.vec2f),
        tangent=wp.zeros(E * G, dtype=wp.vec2f),
        normal=wp.zeros(E * G, dtype=wp.vec2f),
        left=wp.zeros(E * G, dtype=wp.vec2f),
        right=wp.zeros(E * G, dtype=wp.vec2f),
        valid=wp.zeros(E, dtype=wp.int32),
        count=wp.zeros(E, dtype=wp.int32),
    )
    assert gates.position.shape == (E * G,)
    assert gates.tangent.dtype == wp.vec2f
    clone = gates.clone()
    assert clone is not gates
    assert clone.position.ptr != gates.position.ptr
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_public_api.py tests/test_public_api_full.py tests/test_types.py
```

Expected: failures mentioning missing `GateGenConfig`, `GateSequence`, and `GateGenerator`.

- [ ] **Step 3: Add public gate dataclasses**

In `track_gen/_src/types.py`, import remains leaf-only and add:

```python
@dataclass
class GateGenConfig:
    """Configuration for fixed-batch native gate sequence generation."""

    generator: str = "bezier"
    device: str = "cpu"
    num_envs: int = 1

    min_gates: int = 4
    max_gates: int = 32
    gate_radius: float = 0.025
    gate_solve_iters: int = 8
    gate_width: float = 0.0
    gate_ordering: str = "ccw"

    min_num_points: int = 9
    max_num_points: int = 13
    num_points_per_segment: int = 30
    min_point_distance: float = 0.05
    rad: float = 0.4
    edgy: float = 0.0
    scale: float = 1.0
    handle_clamp_frac: float = 0.4
    hull_displacement: float = 0.15

    polar_num_knots: int = 12
    polar_radial_jitter: float = 0.60
    polar_angular_jitter: float = 0.30

    voronoi_num_sites: int = 256
    voronoi_site_layout: str = "void_ring"
    voronoi_control_points: int = 18
    voronoi_radial_variation: float = 0.62
    voronoi_angular_jitter: float = 0.08

    checkpoint_count: int = 12
    checkpoint_radius_min_frac: float = 0.33
    checkpoint_angle_jitter: float = 0.55

    def __post_init__(self):
        if int(self.min_gates) < 2:
            raise ValueError(f"min_gates must be >= 2, got {self.min_gates!r}")
        if int(self.max_gates) < int(self.min_gates):
            raise ValueError(
                f"max_gates must be >= min_gates, got "
                f"{self.max_gates!r} < {self.min_gates!r}"
            )
        if float(self.gate_radius) < 0.0:
            raise ValueError(f"gate_radius must be >= 0, got {self.gate_radius!r}")
        if float(self.gate_width) < 0.0:
            raise ValueError(f"gate_width must be >= 0, got {self.gate_width!r}")
        if self.gate_ordering not in {"ccw", "raw", "random_pairs"}:
            raise ValueError(
                "gate_ordering must be one of {'ccw', 'raw', 'random_pairs'}, "
                f"got {self.gate_ordering!r}"
            )
        if int(self.voronoi_control_points) < 3:
            raise ValueError(
                f"voronoi_control_points must be >= 3, got {self.voronoi_control_points!r}"
            )
        if int(self.voronoi_num_sites) < int(self.voronoi_control_points):
            raise ValueError(
                "voronoi_num_sites must be >= voronoi_control_points, got "
                f"{self.voronoi_num_sites!r} < {self.voronoi_control_points!r}"
            )
        if self.voronoi_site_layout not in {"ring", "void_ring", "clustered", "mixed"}:
            raise ValueError(
                "voronoi_site_layout must be one of "
                "{'ring', 'void_ring', 'clustered', 'mixed'}, got "
                f"{self.voronoi_site_layout!r}"
            )
        if int(self.checkpoint_count) < 3:
            raise ValueError(
                f"checkpoint_count must be >= 3, got {self.checkpoint_count!r}"
            )
        if not (0.0 <= float(self.checkpoint_radius_min_frac) < 1.0):
            raise ValueError(
                "checkpoint_radius_min_frac must be in [0, 1), got "
                f"{self.checkpoint_radius_min_frac!r}"
            )


@dataclass
class GateSequence:
    """Batched fixed-stride gate result returned by GateGenerator."""

    position: wp.array
    tangent: wp.array
    normal: wp.array
    left: wp.array
    right: wp.array
    valid: wp.array
    count: wp.array

    def clone(self) -> "GateSequence":
        return GateSequence(
            position=wp.clone(self.position),
            tangent=wp.clone(self.tangent),
            normal=wp.clone(self.normal),
            left=wp.clone(self.left),
            right=wp.clone(self.right),
            valid=wp.clone(self.valid),
            count=wp.clone(self.count),
        )
```

In `track_gen/__init__.py`, update imports and `__all__`:

```python
from ._src.types import GateGenConfig, GateSequence, Track, TrackGenConfig
from ._src.track_generator import TrackGenerator
from ._src.gate_generator import GateGenerator

__all__ = [
    "TrackGenerator",
    "TrackGenConfig",
    "Track",
    "GateGenerator",
    "GateGenConfig",
    "GateSequence",
    "PerEnvSeededRNG",
    "__version__",
]
```

- [ ] **Step 4: Run tests to verify the type layer passes or only facade import fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_types.py
```

Expected: PASS.

Run:

```bash
.venv/bin/python -m pytest -q tests/test_public_api.py tests/test_public_api_full.py
```

Expected: failures only from `track_gen._src.gate_generator` not existing yet.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/types.py track_gen/__init__.py tests/test_public_api.py tests/test_public_api_full.py tests/test_types.py
git commit --no-gpg-sign -m "feat: add public gate sequence types"
```

## Task 2: Gate Registry and Facade Skeleton

**Files:**
- Create: `track_gen/_src/gate_generator_registry.py`
- Create: `track_gen/_src/gate_generator.py`
- Create: `tests/test_gate_generator_registry.py`
- Create: `tests/test_gate_generator.py`

- [ ] **Step 1: Write failing registry and facade tests**

Create `tests/test_gate_generator_registry.py`:

```python
import pytest

from track_gen._src import gate_generator_registry as reg
from track_gen._src.types import GateGenConfig


def test_gate_registry_available_returns_a_list():
    names = reg.available()
    assert isinstance(names, list)


def test_unknown_gate_generator_raises_with_available_list():
    with pytest.raises(ValueError) as e:
        reg.get("does-not-exist")
    assert "bezier" in str(e.value)


def test_unsupported_ordering_raises_at_facade_construction():
    from track_gen import GateGenerator, PerEnvSeededRNG

    cfg = GateGenConfig(generator="polar", gate_ordering="random_pairs")
    rng = PerEnvSeededRNG(seeds=0, num_envs=cfg.num_envs, device=cfg.device)
    with pytest.raises(ValueError):
        GateGenerator(cfg, rng)
```

Create the first facade tests in `tests/test_gate_generator.py`:

```python
import pytest
import torch

pytest.importorskip("warp")

from track_gen import GateGenConfig, GateGenerator, GateSequence, PerEnvSeededRNG
from tests._warp_compare import to_t


def _make_rng(num_envs, seed=0, device="cpu"):
    import warp as wp

    wp.init()
    return PerEnvSeededRNG(seeds=seed, num_envs=num_envs, device=device)


def test_gate_generator_returns_reused_sequence_buffers():
    E, G = 4, 32
    cfg = GateGenConfig(num_envs=E, max_gates=G, device="cpu", gate_radius=0.0)
    gen = GateGenerator(cfg, _make_rng(E))

    gates1 = gen.generate(E)
    assert isinstance(gates1, GateSequence)
    position_ptr = gates1.position.ptr
    view = to_t(gates1.position)
    view.fill_(-999.0)

    gates2 = gen.generate(E)

    assert gates2 is gates1
    assert gates2.position.ptr == position_ptr
    assert not torch.all(view == -999.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_gate_generator_registry.py tests/test_gate_generator.py
```

Expected: failures for missing registry modules, missing facade, or missing `GateGeneratorSpec`.

- [ ] **Step 3: Add the registry**

Create `track_gen/_src/gate_generator_registry.py`:

```python
"""Registry for native gate sequence generators."""
from __future__ import annotations

import dataclasses
from typing import Callable, FrozenSet


@dataclasses.dataclass(frozen=True)
class GateGeneratorSpec:
    name: str
    alloc_scratch: Callable
    generate: Callable
    max_gates: Callable
    supported_orderings: FrozenSet[str]


GATE_GENERATORS: dict[str, GateGeneratorSpec] = {}
_LOADED = False


def register(spec: GateGeneratorSpec) -> None:
    GATE_GENERATORS[spec.name] = spec


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    from . import warp_generate_gates  # noqa: F401
    from . import warp_generate_polar_gates  # noqa: F401
    from . import warp_generate_voronoi_gates  # noqa: F401
    from . import warp_generate_checkpoint_gates  # noqa: F401

    _LOADED = True


def get(name: str) -> GateGeneratorSpec:
    _ensure_loaded()
    if name not in GATE_GENERATORS:
        raise ValueError(
            f"unknown gate generator {name!r}; available: {sorted(GATE_GENERATORS)}"
        )
    return GATE_GENERATORS[name]


def available() -> list[str]:
    _ensure_loaded()
    return sorted(GATE_GENERATORS)
```

- [ ] **Step 4: Add the facade skeleton**

Create `track_gen/_src/gate_generator.py`:

```python
"""Public fixed-batch facade for native gate sequence generation."""
from __future__ import annotations

import warp as wp

from .types import GateGenConfig, GateSequence

__all__ = ["GateGenConfig", "GateGenerator", "GateSequence"]


class GateGenerator:
    """Generate batched gate sequences into persistent Warp buffers."""

    def __init__(self, config: GateGenConfig, rng) -> None:
        if rng is None:
            raise ValueError("A random number generator must be provided.")

        from . import gate_generator_registry
        from . import warp_gate

        self._generator_spec = gate_generator_registry.get(config.generator)
        if config.gate_ordering not in self._generator_spec.supported_orderings:
            raise ValueError(
                f"gate generator {config.generator!r} does not support "
                f"gate_ordering={config.gate_ordering!r}; supported: "
                f"{sorted(self._generator_spec.supported_orderings)}"
            )
        needed = int(self._generator_spec.max_gates(config))
        if needed > int(config.max_gates):
            raise ValueError(
                f"GateGenConfig.max_gates={config.max_gates} is too small for "
                f"{config.generator!r}, which needs {needed}"
            )

        self._config = config
        self._rng = rng
        self._gates = warp_gate.alloc_gate_sequence(config)
        self._scratch = self._generator_spec.alloc_scratch(config)
        self._seed_buf = wp.empty(int(config.num_envs), dtype=wp.int32, device=str(config.device))
        self._graph: "wp.Graph | None" = None

    def _run(self) -> None:
        from . import warp_gate

        self._generator_spec.generate(
            self._seed_buf,
            self._config,
            self._gates,
            self._scratch,
        )
        warp_gate.finalize_gate_sequence(self._gates, self._config)

    def generate(self, num_or_ids=None) -> GateSequence:
        if num_or_ids is not None:
            if not isinstance(num_or_ids, int):
                raise TypeError(
                    "GateGenerator.generate() does not accept explicit environment ids; "
                    "construct a generator with the desired fixed num_envs instead."
                )
            if num_or_ids != self._config.num_envs:
                raise ValueError(
                    f"GateGenerator is fixed-batch for {self._config.num_envs} envs; "
                    f"got num_or_ids={num_or_ids}."
                )

        from . import warp_gate

        wp.copy(self._seed_buf, self._rng.seeds_warp)
        dev = str(self._config.device)
        if "cuda" in dev:
            if self._graph is None:
                warp_gate._CAPTURING = True
                try:
                    for _ in range(3):
                        self._run()
                    wp.synchronize()
                    with wp.ScopedCapture(device=dev) as cap:
                        self._run()
                    self._graph = cap.graph
                finally:
                    warp_gate._CAPTURING = False
            wp.capture_launch(self._graph)
            wp.synchronize()
        else:
            self._run()
        return self._gates
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_gate_generator_registry.py tests/test_gate_generator.py
```

Expected: registry tests pass far enough to prove import and unknown-name behavior; facade construction may still fail until `warp_gate` and native modules exist.

- [ ] **Step 6: Commit**

```bash
git add track_gen/_src/gate_generator_registry.py track_gen/_src/gate_generator.py tests/test_gate_generator_registry.py tests/test_gate_generator.py
git commit --no-gpg-sign -m "feat: add gate generator facade skeleton"
```

## Task 3: Common Gate Warp Kernels

**Files:**
- Create: `track_gen/_src/warp_gate.py`
- Create: `tests/test_warp_gate.py`

- [ ] **Step 1: Write failing common kernel tests**

Create `tests/test_warp_gate.py`:

```python
import pytest
import torch

pytest.importorskip("warp")

import warp as wp

from track_gen._src.types import GateGenConfig
from track_gen._src import warp_gate
from tests._warp_compare import to_t


def _manual_sequence(E=1, G=4):
    return warp_gate.alloc_gate_sequence(GateGenConfig(num_envs=E, max_gates=G))


def test_finalize_computes_normals_and_endpoints():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    tan = to_t(gates.tangent).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([2.0, 0.0])
    tan[0, 0] = torch.tensor([1.0, 0.0])
    tan[0, 1] = torch.tensor([1.0, 0.0])
    count[0] = 2

    cfg = GateGenConfig(max_gates=4, gate_width=2.0, gate_radius=0.0)
    warp_gate.finalize_gate_sequence(gates, cfg)

    normal = to_t(gates.normal).view(1, 4, 2)
    left = to_t(gates.left).view(1, 4, 2)
    right = to_t(gates.right).view(1, 4, 2)
    valid = to_t(gates.valid).bool()
    assert torch.allclose(normal[0, 0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(left[0, 0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(right[0, 0], torch.tensor([0.0, -1.0]), atol=1e-6)
    assert valid.tolist() == [True]
    assert torch.isnan(left[0, 2:]).all()


def test_finalize_invalidates_too_close_gate_centres():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    tan = to_t(gates.tangent).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([0.01, 0.0])
    tan[0, :2] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    count[0] = 2

    cfg = GateGenConfig(max_gates=4, min_gates=2, gate_radius=0.05)
    warp_gate.finalize_gate_sequence(gates, cfg)

    assert to_t(gates.valid).bool().tolist() == [False]


def test_finalize_invalidates_crossing_gate_segments():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    tan = to_t(gates.tangent).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([0.0, 0.5])
    tan[0, 0] = torch.tensor([1.0, 0.0])
    tan[0, 1] = torch.tensor([0.0, 1.0])
    count[0] = 2

    cfg = GateGenConfig(max_gates=4, min_gates=2, gate_width=2.0, gate_radius=0.0)
    warp_gate.finalize_gate_sequence(gates, cfg)

    assert to_t(gates.valid).bool().tolist() == [False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_warp_gate.py
```

Expected: import failure for `warp_gate`.

- [ ] **Step 3: Implement common kernels**

Create `track_gen/_src/warp_gate.py` with these responsibilities:

```python
"""Common Warp kernels for native gate sequence generation."""
from __future__ import annotations

import warp as wp

from .types import GateSequence

_INITED = False
_CAPTURING = False
_ORDER_SALT = 6421


def _init() -> None:
    global _INITED
    if not _INITED:
        wp.init()
        _INITED = True


def _sync(device) -> None:
    if _CAPTURING:
        return
    if "cuda" in str(device):
        wp.synchronize()
```

Add `alloc_gate_sequence(config)`:

```python
def alloc_gate_sequence(config) -> GateSequence:
    _init()
    E = int(config.num_envs)
    G = int(config.max_gates)
    dev = str(config.device)
    return GateSequence(
        position=wp.empty(E * G, dtype=wp.vec2f, device=dev),
        tangent=wp.empty(E * G, dtype=wp.vec2f, device=dev),
        normal=wp.empty(E * G, dtype=wp.vec2f, device=dev),
        left=wp.empty(E * G, dtype=wp.vec2f, device=dev),
        right=wp.empty(E * G, dtype=wp.vec2f, device=dev),
        valid=wp.empty(E, dtype=wp.int32, device=dev),
        count=wp.empty(E, dtype=wp.int32, device=dev),
    )
```

Add kernels for NaN padding, counted copy, ccw order, random order, normalization, tangent, endpoints, and validity. Use the same segment predicate style as `warp_pipeline._self_intersections_by_i_k`, but check gate segments `left[i] -> right[i]` rather than path edges. Exact kernel names:

```python
@wp.func
def _safe_normalize2(v: wp.vec2f) -> wp.vec2f:
    return v / wp.max(wp.length(v), 1.0e-8)


@wp.func
def _ccw(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
    return (qy - oy) * (px - ox) - (py - oy) * (qx - ox)


@wp.kernel
def _fill_padding_k(position, tangent, normal, left, right, max_gates: int, count):
    t = wp.tid()
    e = t // max_gates
    i = t % max_gates
    if i >= count[e]:
        nan = wp.vec2f(wp.nan, wp.nan)
        position[t] = nan
        tangent[t] = nan
        normal[t] = nan
        left[t] = nan
        right[t] = nan


@wp.kernel
def _copy_counted_k(src, src_stride: int, count, max_gates: int, dst):
    t = wp.tid()
    e = t // max_gates
    i = t % max_gates
    if i < count[e]:
        dst[t] = src[e * src_stride + i]
    else:
        dst[t] = wp.vec2f(wp.nan, wp.nan)


@wp.kernel
def _ccw_order_k(src, src_stride: int, count, max_gates: int, keys, dst):
    e = wp.tid()
    m = count[e]
    sbase = e * src_stride
    dbase = e * max_gates
    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    for i in range(max_gates):
        if i < m:
            p = src[sbase + i]
            sx += wp.float64(p[0])
            sy += wp.float64(p[1])
    cx = wp.float32(sx / wp.float64(wp.max(m, 1)))
    cy = wp.float32(sy / wp.float64(wp.max(m, 1)))
    for i in range(max_gates):
        if i < m:
            p = src[sbase + i]
            key = wp.atan2(p[0] - cx, p[1] - cy)
            j = i - 1
            while j >= 0 and keys[dbase + j] > key:
                keys[dbase + j + 1] = keys[dbase + j]
                dst[dbase + j + 1] = dst[dbase + j]
                j = j - 1
            keys[dbase + j + 1] = key
            dst[dbase + j + 1] = p
        else:
            dst[dbase + i] = wp.vec2f(wp.nan, wp.nan)


@wp.kernel
def _random_order_k(seeds, src, src_stride: int, count, max_gates: int, keys, dst):
    e = wp.tid()
    m = count[e]
    sbase = e * src_stride
    dbase = e * max_gates
    state = wp.rand_init(seeds[e] * _ORDER_SALT + 1)
    for i in range(max_gates):
        if i < m:
            p = src[sbase + i]
            key = wp.randf(state)
            j = i - 1
            while j >= 0 and keys[dbase + j] > key:
                keys[dbase + j + 1] = keys[dbase + j]
                dst[dbase + j + 1] = dst[dbase + j]
                j = j - 1
            keys[dbase + j + 1] = key
            dst[dbase + j + 1] = p
        else:
            dst[dbase + i] = wp.vec2f(wp.nan, wp.nan)
```

Add wrapper functions:

```python
def order_points(seeds_wp, src, src_stride: int, count, max_gates: int, ordering: str, keys, dst) -> None:
    _init()
    E = count.shape[0]
    dev = str(dst.device)
    if ordering == "ccw":
        wp.launch(_ccw_order_k, dim=E, inputs=[src, src_stride, count, max_gates, keys, dst], device=dev)
    elif ordering == "random_pairs":
        wp.launch(_random_order_k, dim=E, inputs=[seeds_wp, src, src_stride, count, max_gates, keys, dst], device=dev)
    elif ordering == "raw":
        wp.launch(_copy_counted_k, dim=E * max_gates, inputs=[src, src_stride, count, max_gates, dst], device=dev)
    else:
        raise ValueError(f"unsupported gate ordering {ordering!r}")
    _sync(dev)
```

Add these remaining kernels and public wrappers:

```python
@wp.kernel
def _normalize_positions_k(position, max_gates: int, count, target_extent: float):
    e = wp.tid()
    base = e * max_gates
    cnt = count[e]
    min_x = float(1.0e30)
    max_x = float(-1.0e30)
    min_y = float(1.0e30)
    max_y = float(-1.0e30)
    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            min_x = wp.min(min_x, p[0])
            max_x = wp.max(max_x, p[0])
            min_y = wp.min(min_y, p[1])
            max_y = wp.max(max_y, p[1])
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    extent = wp.max(max_x - min_x, max_y - min_y)
    scale = target_extent / wp.max(extent, 1.0e-8)
    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            position[base + i] = wp.vec2f((p[0] - cx) * scale, (p[1] - cy) * scale)


@wp.kernel
def _tangents_from_positions_k(position, tangent, max_gates: int, count):
    t = wp.tid()
    e = t // max_gates
    i = t % max_gates
    cnt = count[e]
    base = e * max_gates
    if i >= cnt or cnt < 2:
        tangent[t] = wp.vec2f(wp.nan, wp.nan)
        return
    prev = position[base + (i + cnt - 1) % cnt]
    nxt = position[base + (i + 1) % cnt]
    tangent[t] = _safe_normalize2(nxt - prev)


@wp.kernel
def _endpoint_k(position, tangent, normal, left, right, max_gates: int, count, gate_width: float):
    t = wp.tid()
    e = t // max_gates
    i = t % max_gates
    if i >= count[e]:
        nan = wp.vec2f(wp.nan, wp.nan)
        normal[t] = nan
        left[t] = nan
        right[t] = nan
        return
    tan = _safe_normalize2(tangent[t])
    tangent[t] = tan
    n = wp.vec2f(-tan[1], tan[0])
    normal[t] = n
    half = 0.5 * gate_width
    left[t] = position[t] + half * n
    right[t] = position[t] - half * n


@wp.kernel
def _gate_validity_k(position, tangent, left, right, max_gates: int, count, min_gates: int, min_dist: float, has_width: int, valid):
    e = wp.tid()
    base = e * max_gates
    cnt = count[e]
    ok = int(1)
    if cnt < min_gates:
        ok = int(0)
    for i in range(max_gates):
        if i < cnt:
            p = position[base + i]
            t = tangent[base + i]
            if not (wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(t[0]) and wp.isfinite(t[1])):
                ok = int(0)
            for j in range(i + 1, max_gates):
                if j < cnt:
                    q = position[base + j]
                    if wp.length(q - p) < min_dist:
                        ok = int(0)
                    if has_width == 1:
                        a0 = left[base + i]
                        a1 = right[base + i]
                        b0 = left[base + j]
                        b1 = right[base + j]
                        d1 = _ccw(b0[0], b0[1], b1[0], b1[1], a0[0], a0[1])
                        d2 = _ccw(b0[0], b0[1], b1[0], b1[1], a1[0], a1[1])
                        d3 = _ccw(a0[0], a0[1], a1[0], a1[1], b0[0], b0[1])
                        d4 = _ccw(a0[0], a0[1], a1[0], a1[1], b1[0], b1[1])
                        cross_ab = (d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)
                        cross_ba = (d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)
                        if cross_ab and cross_ba:
                            ok = int(0)
    valid[e] = ok


def normalize_positions(position, max_gates: int, count, target_extent: float) -> None:
    _init()
    E = count.shape[0]
    wp.launch(_normalize_positions_k, dim=E, inputs=[position, max_gates, count, float(target_extent)], device=str(position.device))
    _sync(position.device)


def tangents_from_positions(position, tangent, max_gates: int, count) -> None:
    _init()
    E = count.shape[0]
    wp.launch(_tangents_from_positions_k, dim=E * max_gates, inputs=[position, tangent, max_gates, count], device=str(position.device))
    _sync(position.device)


def finalize_gate_sequence(gates: GateSequence, config) -> None:
    _init()
    E = int(config.num_envs)
    G = int(config.max_gates)
    dev = str(gates.position.device)
    wp.launch(_endpoint_k, dim=E * G, inputs=[gates.position, gates.tangent, gates.normal, gates.left, gates.right, G, gates.count, float(config.gate_width)], device=dev)
    wp.launch(_fill_padding_k, dim=E * G, inputs=[gates.position, gates.tangent, gates.normal, gates.left, gates.right, G, gates.count], device=dev)
    wp.launch(_gate_validity_k, dim=E, inputs=[gates.position, gates.tangent, gates.left, gates.right, G, gates.count, int(config.min_gates), 2.0 * float(config.gate_radius), int(float(config.gate_width) > 0.0), gates.valid], device=dev)
    _sync(dev)
```

Acceptance details:
- `normalize_positions` centers each env by bbox and scales longest bbox extent to `target_extent`.
- `tangents_from_positions` uses central difference `position[i+1] - position[i-1]` with wrap over `count[e]`.
- `finalize_gate_sequence` normalizes tangent again, writes `normal=(-ty, tx)`, `left=position + 0.5*gate_width*normal`, `right=position - 0.5*gate_width*normal`, NaN-pads slots `i >= count[e]`, and writes `valid[e]`.
- `valid[e]` is 1 only when count, finite pose fields, centre distances, and optional segment intersections pass.

- [ ] **Step 4: Run common kernel tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_warp_gate.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/warp_gate.py tests/test_warp_gate.py
git commit --no-gpg-sign -m "feat: add common gate geometry kernels"
```

## Task 4: Bezier and Hull Gate-Native Generators

**Files:**
- Create: `track_gen/_src/warp_generate_gates.py`
- Modify: `tests/test_gate_generator.py`
- Modify: `tests/test_gate_generator_registry.py`

- [ ] **Step 1: Add failing Bezier/Hull generator tests**

Append to `tests/test_gate_generator.py`:

```python
@pytest.mark.parametrize("generator", ["bezier", "hull"])
@pytest.mark.parametrize("ordering", ["ccw", "random_pairs"])
def test_point_family_gate_generators_emit_finite_native_gates(generator, ordering):
    E, G = 8, 32
    cfg = GateGenConfig(
        generator=generator,
        gate_ordering=ordering,
        num_envs=E,
        max_gates=G,
        device="cpu",
        gate_radius=0.0,
    )
    gates = GateGenerator(cfg, _make_rng(E, seed=31)).generate(E)
    position = to_t(gates.position).view(E, G, 2)
    tangent = to_t(gates.tangent).view(E, G, 2)
    count = to_t(gates.count)
    valid = to_t(gates.valid).bool()

    assert valid.all()
    assert torch.all(count >= cfg.min_gates)
    for e in range(E):
        c = int(count[e])
        assert torch.isfinite(position[e, :c]).all()
        assert torch.isfinite(tangent[e, :c]).all()
        assert torch.isnan(position[e, c:]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_gate_generator.py::test_point_family_gate_generators_emit_finite_native_gates
```

Expected: missing registrations for Bezier/Hull gate specs.

- [ ] **Step 3: Implement point-family gate generators**

Create `track_gen/_src/warp_generate_gates.py`.

Use existing sampling functions:
- Bezier: `warp_generate.corner_count_sample_inplace`, `warp_generate.corner_sample_inplace`.
- Hull: `warp_generate_hull.point_count_sample_inplace`, `warp_generate_hull.point_sample_inplace`.

Add scratch:

```python
class PointGateScratch:
    __slots__ = ("count", "points", "used", "ordered", "keys")

    def __init__(self, count, points, used, ordered, keys):
        self.count = count
        self.points = points
        self.used = used
        self.ordered = ordered
        self.keys = keys
```

Allocation:

```python
def _point_gate_alloc_scratch(config):
    from . import warp_gate

    warp_gate._init()
    E = int(config.num_envs)
    G = int(config.max_gates)
    dev = str(config.device)
    return PointGateScratch(
        count=wp.empty(E, dtype=wp.int32, device=dev),
        points=wp.empty(E * G, dtype=wp.vec2f, device=dev),
        used=wp.empty(E * G, dtype=wp.int32, device=dev),
        ordered=wp.empty(E * G, dtype=wp.vec2f, device=dev),
        keys=wp.empty(E * G, dtype=wp.float32, device=dev),
    )
```

Bezier generate:

```python
def generate_bezier_gates(seeds_wp, config, out, scratch) -> None:
    from . import warp_gate, warp_generate

    warp_generate.corner_count_sample_inplace(seeds_wp, 0, config, scratch.count)
    warp_generate.corner_sample_inplace(
        seeds_wp, 0, config, scratch.points, scratch.used
    )
    warp_gate.order_points(
        seeds_wp, scratch.points, int(config.max_gates), scratch.count,
        int(config.max_gates), str(config.gate_ordering), scratch.keys, out.position,
    )
    warp_gate.tangents_from_positions(
        out.position, out.tangent, int(config.max_gates), scratch.count
    )
    wp.copy(out.count, scratch.count)
```

Hull generate:

```python
def generate_hull_gates(seeds_wp, config, out, scratch) -> None:
    from . import warp_gate, warp_generate_hull

    warp_generate_hull.point_count_sample_inplace(seeds_wp, config, scratch.count)
    warp_generate_hull.point_sample_inplace(
        seeds_wp, config, scratch.points, scratch.used
    )
    warp_gate.order_points(
        seeds_wp, scratch.points, int(config.max_gates), scratch.count,
        int(config.max_gates), str(config.gate_ordering), scratch.keys, out.position,
    )
    warp_gate.tangents_from_positions(
        out.position, out.tangent, int(config.max_gates), scratch.count
    )
    wp.copy(out.count, scratch.count)
```

Register:

```python
from . import gate_generator_registry as _registry

_registry.register(_registry.GateGeneratorSpec(
    name="bezier",
    alloc_scratch=_point_gate_alloc_scratch,
    generate=generate_bezier_gates,
    max_gates=lambda config: int(config.max_num_points),
    supported_orderings=frozenset({"ccw", "random_pairs"}),
))

_registry.register(_registry.GateGeneratorSpec(
    name="hull",
    alloc_scratch=_point_gate_alloc_scratch,
    generate=generate_hull_gates,
    max_gates=lambda config: int(config.max_num_points),
    supported_orderings=frozenset({"ccw", "random_pairs"}),
))
```

- [ ] **Step 4: Run Bezier/Hull gate tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_gate_generator.py::test_point_family_gate_generators_emit_finite_native_gates tests/test_gate_generator_registry.py
```

Expected: Bezier/Hull tests pass; registry test may still fail until all five specs are registered.

- [ ] **Step 5: Commit**

```bash
git add track_gen/_src/warp_generate_gates.py tests/test_gate_generator.py tests/test_gate_generator_registry.py
git commit --no-gpg-sign -m "feat: add bezier and hull gate generators"
```

## Task 5: Polar, Voronoi, and Checkpoint Gate-Native Generators

**Files:**
- Create: `track_gen/_src/warp_generate_polar_gates.py`
- Create: `track_gen/_src/warp_generate_voronoi_gates.py`
- Create: `track_gen/_src/warp_generate_checkpoint_gates.py`
- Modify: `tests/test_gate_generator.py`
- Modify: `tests/test_gate_generator_registry.py`

- [ ] **Step 1: Add failing native generator tests**

Append to `tests/test_gate_generator.py`:

```python
@pytest.mark.parametrize(
    ("generator", "orderings"),
    [
        ("polar", ["ccw", "raw"]),
        ("voronoi", ["ccw", "raw"]),
        ("checkpoint", ["ccw", "raw"]),
    ],
)
def test_structured_gate_generators_emit_finite_native_gates(generator, orderings):
    E, G = 6, 32
    for ordering in orderings:
        cfg = GateGenConfig(
            generator=generator,
            gate_ordering=ordering,
            num_envs=E,
            max_gates=G,
            device="cpu",
            gate_radius=0.0,
        )
        gates = GateGenerator(cfg, _make_rng(E, seed=71)).generate(E)
        position = to_t(gates.position).view(E, G, 2)
        count = to_t(gates.count)
        valid = to_t(gates.valid).bool()
        assert valid.all()
        for e in range(E):
            c = int(count[e])
            assert c >= cfg.min_gates
            assert torch.isfinite(position[e, :c]).all()
            assert torch.isnan(position[e, c:]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_gate_generator.py::test_structured_gate_generators_emit_finite_native_gates tests/test_gate_generator_registry.py
```

Expected: missing registrations for Polar/Voronoi/Checkpoint.

- [ ] **Step 3: Implement Polar gates**

Create `track_gen/_src/warp_generate_polar_gates.py`.

Use `warp_generate_polar._polar_controls_k` to fill controls, then order/copy controls into `out.position`, normalize, compute tangents, and copy count.

The module must register:

```python
_registry.register(_registry.GateGeneratorSpec(
    name="polar",
    alloc_scratch=polar_gate_alloc_scratch,
    generate=generate_polar_gates,
    max_gates=lambda config: int(config.polar_num_knots),
    supported_orderings=frozenset({"ccw", "raw"}),
))
```

Implementation details:
- Scratch fields: `controls`, `count`, `ordered`, `keys`.
- `count[e] = polar_num_knots` via `wp.launch(_fill_i32_k, dim=E, inputs=[scratch.count, K])`.
- Use `_polar_num_knots(config)`, `_BASE_RADIUS`, and `_BEZIER_EXTENT` from `warp_generate_polar`.
- Use `warp_gate.normalize_positions(out.position, max_gates, out.count, config.scale * _BEZIER_EXTENT)`.

- [ ] **Step 4: Implement Voronoi gates**

Create `track_gen/_src/warp_generate_voronoi_gates.py`.

Use existing Voronoi kernels:
- `_sample_sites_k`
- `_select_anchor_sites_k`
- `_voronoi_layout_mode`
- `_TARGET_EXTENT`

The module must register:

```python
_registry.register(_registry.GateGeneratorSpec(
    name="voronoi",
    alloc_scratch=voronoi_gate_alloc_scratch,
    generate=generate_voronoi_gates,
    max_gates=lambda config: int(config.voronoi_control_points),
    supported_orderings=frozenset({"ccw", "raw"}),
))
```

Implementation details:
- Scratch fields: `sites`, `used`, `selected`, `count`, `ordered`, `keys`.
- `count[e] = voronoi_control_points`.
- `raw` uses selected anchors in angular anchor order.
- `ccw` reorders selected anchors by centroid angle.
- Normalize selected positions to `config.scale * _TARGET_EXTENT`.

- [ ] **Step 5: Implement Checkpoint gates**

Create `track_gen/_src/warp_generate_checkpoint_gates.py`.

Use `warp_generate_checkpoint._sample_checkpoints_k` with `K=1`. Register:

```python
_registry.register(_registry.GateGeneratorSpec(
    name="checkpoint",
    alloc_scratch=checkpoint_gate_alloc_scratch,
    generate=generate_checkpoint_gates,
    max_gates=lambda config: int(config.checkpoint_count),
    supported_orderings=frozenset({"ccw", "raw"}),
))
```

Implementation details:
- Scratch fields: `checkpoints`, `count`, `ordered`, `keys`.
- `count[e] = checkpoint_count`.
- `raw` preserves checkpoint index order.
- `ccw` reorders checkpoints by centroid angle.
- Normalize positions to `config.scale * warp_generate_checkpoint._BEZIER_EXTENT`.

- [ ] **Step 6: Run structured generator tests and registry tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_gate_generator.py tests/test_gate_generator_registry.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add track_gen/_src/warp_generate_polar_gates.py track_gen/_src/warp_generate_voronoi_gates.py track_gen/_src/warp_generate_checkpoint_gates.py tests/test_gate_generator.py tests/test_gate_generator_registry.py
git commit --no-gpg-sign -m "feat: add structured gate generators"
```

## Task 6: Validity Behavior and Track Compatibility

**Files:**
- Modify: `tests/test_gate_generator.py`
- Create: `tests/test_gate_track_compat.py`
- Modify: `README.md`

- [ ] **Step 1: Add behavior and compatibility tests**

Append to `tests/test_gate_generator.py`:

```python
def test_gate_generator_invalidates_large_gate_radius():
    E = 4
    cfg = GateGenConfig(
        generator="checkpoint",
        gate_ordering="raw",
        num_envs=E,
        max_gates=32,
        gate_radius=50.0,
        device="cpu",
    )
    gates = GateGenerator(cfg, _make_rng(E, seed=5)).generate(E)
    assert not to_t(gates.valid).bool().any()


def test_generate_wrong_batch_raises():
    cfg = GateGenConfig(num_envs=4, device="cpu")
    gen = GateGenerator(cfg, _make_rng(4))
    with pytest.raises(ValueError):
        gen.generate(5)


def test_generate_rejects_sequence_ids():
    cfg = GateGenConfig(num_envs=4, device="cpu")
    gen = GateGenerator(cfg, _make_rng(4))
    with pytest.raises(TypeError, match="does not accept explicit environment ids"):
        gen.generate([0, 1, 2, 3])
```

Create `tests/test_gate_track_compat.py`:

```python
import pytest
import torch

pytest.importorskip("warp")

from track_gen import (
    GateGenConfig,
    GateGenerator,
    PerEnvSeededRNG,
    TrackGenConfig,
    TrackGenerator,
)
from tests._warp_compare import to_t


def _track_snapshot(seed=123):
    E = 4
    cfg = TrackGenConfig(
        generator="bezier",
        num_envs=E,
        num_points=64,
        N_max=128,
        device="cpu",
    )
    rng = PerEnvSeededRNG(seeds=seed, num_envs=E, device="cpu")
    track = TrackGenerator(cfg, rng).generate(E).clone()
    return (
        to_t(track.center).clone(),
        to_t(track.outer).clone(),
        to_t(track.inner).clone(),
        to_t(track.valid).clone(),
        to_t(track.count).clone(),
    )


def test_gate_generation_does_not_change_track_generation_outputs():
    before = _track_snapshot()

    gate_cfg = GateGenConfig(num_envs=4, max_gates=32, device="cpu", gate_radius=0.0)
    gate_rng = PerEnvSeededRNG(seeds=77, num_envs=4, device="cpu")
    GateGenerator(gate_cfg, gate_rng).generate(4)

    after = _track_snapshot()

    for lhs, rhs in zip(before, after):
        assert torch.equal(lhs, rhs)
```

- [ ] **Step 2: Run tests to verify behavior**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_gate_generator.py tests/test_gate_track_compat.py
```

Expected: PASS.

- [ ] **Step 3: Add README quickstart**

Add a short section after the existing `TrackGenerator` quickstart:

```markdown
### Gate sequence generation

For drone-style courses where track width is irrelevant, use `GateGenerator`.
It emits gate centres and orientations directly from native first-stage generator anchors
and skips constant-spacing, XPBD relaxation, and inflation.

```python
import warp as wp
wp.init()

from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG

E, device = 64, "cuda"
config = GateGenConfig(
    generator="bezier",
    gate_ordering="random_pairs",
    num_envs=E,
    max_gates=32,
    gate_width=0.4,
    gate_radius=0.05,
    device=device,
)
rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)

gates = GateGenerator(config, rng).generate()
position = wp.to_torch(gates.position).view(E, config.max_gates, 2)
tangent = wp.to_torch(gates.tangent).view(E, config.max_gates, 2)
valid = wp.to_torch(gates.valid).bool()
```
```

- [ ] **Step 4: Run public and focused tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_public_api.py tests/test_public_api_full.py tests/test_types.py tests/test_gate_generator_registry.py tests/test_warp_gate.py tests/test_gate_generator.py tests/test_gate_track_compat.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_gate_generator.py tests/test_gate_track_compat.py README.md
git commit --no-gpg-sign -m "test: verify gate behavior and track compatibility"
```

## Task 7: Final Verification

**Files:**
- No source edits expected.

- [ ] **Step 1: Run the fast test lane**

Run:

```bash
.venv/bin/python -m pytest -q -m "not slow and not benchmark and not cuda"
```

Expected: PASS.

- [ ] **Step 2: Check runtime import remains torch-free**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_public_api.py::test_import_track_gen_pulls_no_torch
```

Expected: PASS.

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only planned gate source, tests, and README files are changed.

- [ ] **Step 4: Commit final cleanup if any verification-only edits were needed**

If a verification command exposes a small fix, apply the fix, rerun the focused failing test, and commit:

```bash
git add <fixed-files>
git commit --no-gpg-sign -m "fix: polish gate sequence generation"
```

If no fixes were needed, do not create an empty commit.
