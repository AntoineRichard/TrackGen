# Warp-native Phase A — Milestone 2 (runtime de-torch + strict zero-allocation)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Turn the shipped pipeline fully Warp-native with **strict zero per-call allocation**:
`Track` holds `wp.array`; `TrackGenerator` pre-allocates **all** buffers (output + every stage's
scratch + seed) once; every stage writes **in place** into caller-owned buffers; `generate()`
returns the **same** `Track` (stable `.ptr`); the free `generate_tracks_warp`/`_graph` +
`CapturedTracks` are removed; `__all__` slims to `{TrackGenerator, TrackGenConfig, Track,
PerEnvSeededRNG, __version__}`. (Builds on M1, which already made `rng_utils` torch-free.)

**Scale & approach (read this first).** Strict zero-allocation means **every** stage function
(`assemble`, `offset`, `frame_curvature`, `resample_*`, `self_intersections`, `thickness`,
`separation_min`, `curvature_radius_min`, `turning_number`, `gates`, `validity`,
`arc_length_resample_warp`, `corner_sample`, `corner_count_sample`, `ccw_sort`, `inflate_warp`,
the `_band_l0` launch, and `warp_relax.xpbd_solve`/`resample`) changes from "allocate-and-return"
to "write into a caller-provided pre-allocated buffer", and the **20 per-kernel test files** that
call these directly change their call pattern. This is a ~20-stage rewrite + `Track` flip +
`~40`-file test migration — too large for one diff. It is executed as an **ordered sequence of
small green increments**, each its own task cycle, detailed just-in-time. Do **not** attempt it
as one change.

**Tech Stack:** Python ≥ 3.10, NVIDIA Warp, numpy, pytest.

## Global Constraints (bind every increment)

- **Strict zero per-call allocation:** no `wp.zeros`/`wp.empty`/`torch.*` allocation inside
  `generate()` or any stage it calls on the hot path. All buffers are pre-allocated once in
  `TrackGenerator.__init__` (sized `E=num_envs`, `N_max`) and written in place. Stages take their
  output buffer(s) as explicit args so they stay independently callable (per-kernel tests allocate
  the out buffer once and pass it).
- **Zero torch in `track_gen/_src`** and **zero readback** (`wp.to_numpy`/`.numpy()`); numpy only
  in `__init__` one-time setup. The boundary serves `wp.array`.
- **`Track` fields are `wp.array`**; `generate()` returns the same `Track` instance (stable `.ptr`).
- **Public `__all__`** (end of M2) = `TrackGenerator`, `TrackGenConfig`, `Track`,
  `PerEnvSeededRNG`, `__version__`.
- **Oracle stays torch**; per-kernel & e2e tests `wp.to_torch(...)` the pipeline output before the
  existing `torch.allclose`/`torch.equal` vs the oracle. A small shared test helper
  (`tests/_warp_compare.py`: `to_t(wp_arr) -> torch.Tensor`) keeps each migration one-line.
- **Gate:** `.venv/bin/python -m pytest -q` green on the Warp `cpu` device after **every** increment
  (baseline before M2: **232 passed**). GPG fails → commit `--no-gpg-sign`.

## Increment sequence (detail each just-in-time before executing)

1. **Inc 1 — Output spine + `Track`→`wp.array` + buffer ownership (DETAILED BELOW).** Flip `Track`
   to `wp.array`; `TrackGenerator` pre-allocates the output `Track` buffers and `inflate_warp`
   writes the final results into them in place; `generate()` returns the persistent `Track`. The
   internal stages stay as-is (torch, allocate-return) for now — `inflate_warp` copies their final
   torch results into the pre-allocated `wp.array` Track buffers (a temporary boundary copy removed
   as stages convert in Inc 2+). Migrate only the **Track-consuming** tests (`test_warp_inflate`,
   `test_warp_pipeline_e2e`, `test_warp_generate`, facade tests, the 14 field-accessors) to the
   `wp.to_torch` boundary + add the stable-`.ptr` test. Keep `generate_tracks_warp` as a private
   helper for now (free-function removal is the final API increment). Establishes the buffer-
   ownership pattern + test helper every later increment reuses.
2. **Inc 2..N — per-stage in-place conversion (one stage or tight group per increment).** For each
   stage, in dependency order (leaf stages first: `offset`, `frame_curvature`, `self_intersections`,
   `thickness`, `separation_min`, `curvature_radius_min`, `turning_number`, `arc_length_resample`,
   `ccw_sort`, `assemble`, `corner_sample`, `corner_count_sample`, `resample_uniform`,
   `resample_constant_spacing`, `gates`, `validity`, `generate_centerline_warp`, the `_band_l0`
   launch, `warp_relax.xpbd_solve`): change it to take a pre-allocated out buffer + write in place
   (torch→`wp` internally; `torch.where`→a `wp` select kernel where it appears), pre-allocate that
   buffer in `TrackGenerator.__init__`, update its per-kernel test to allocate+pass+`wp.to_torch`,
   and drop the now-removed boundary copy. Green per increment. Delete the dead `_mean_seg_len_torch`
   when its stage is reached.
3. **Inc final-A — API slim:** remove `generate_tracks_warp`/`_graph` + `CapturedTracks`; fold the
   orchestration into `TrackGenerator` private methods; set `__all__`. Migrate the ~7 tests that
   still call the free functions to the `TrackGenerator` API. Green.
4. **Inc final-B — torch-free + zero-alloc guards:** assert no `import torch` anywhere in
   `track_gen/_src`; assert no `wp.to_numpy`/`.numpy()` in `track_gen/`; assert numpy only in
   `__init__`. (The `pyproject` torch→dev move + the torch-free-import subprocess test are **M3**.)

---

## Increment 1 — Output spine: `Track`→`wp.array` + buffer ownership

**Files:**
- Modify: `track_gen/_src/types.py` (Track fields → `wp.array`; import warp), `tests/test_types.py`
  (leaf-purity test now allows `import warp`), `track_gen/_src/warp_pipeline.py` (`inflate_warp`
  writes into caller buffers), `track_gen/_src/track_generator.py` (own + pre-allocate the Track
  buffers, return persistent Track), the Track-consuming tests.
- Create: `tests/_warp_compare.py` (shared `to_t` helper).

**Interfaces:**
- Produces: `Track` with `wp.array` fields; `TrackGenerator.generate()` returns the same `Track`
  instance each call (stable `.ptr`), buffers written in place; `inflate_warp(center, config, out:
  Track, valid=None, count=None) -> None` writes into `out`'s buffers.

- [ ] **Step 1: Baseline** — `.venv/bin/python -m pytest -q` (record count; must be **232**).

- [ ] **Step 2: `Track` → `wp.array`** in `track_gen/_src/types.py`

Replace `from torch import Tensor` with `import warp as wp`; retype the 9 fields:
```python
import warp as wp
# ...
@dataclass
class Track:
    outer: wp.array    # [E, N] vec2f
    center: wp.array   # [E, N] vec2f
    inner: wp.array    # [E, N] vec2f
    tangent: wp.array  # [E, N] vec2f
    normal: wp.array   # [E, N] vec2f
    arclen: wp.array   # [E, N] float32
    length: wp.array   # [E] float32
    valid: wp.array    # [E] bool/int32
    count: wp.array    # [E] int32
```
Update the module docstring line that claims it imports no warp (it now does — warp is a core dep).

- [ ] **Step 3: Update the leaf-purity test** in `tests/test_types.py`

`test_types_module_has_no_intra_package_imports` forbids `import warp`. `types.py` now legitimately
imports warp (for the `wp.array` field types). Remove `"import warp"` from the forbidden tuple,
keep the sibling-import bans (`from .track_generator`, `from .warp_pipeline`, etc.). Also fix
`test_track_construct_from_tensors_field_shapes`: build the fields as `wp.array`/`wp.zeros` and
assert on `.shape`/`.dtype` (or wrap with `wp.to_torch`) instead of torch tensors.

- [ ] **Step 4: Add the shared compare helper** `tests/_warp_compare.py`

```python
import warp as wp

def to_t(a):
    """wp.array -> torch.Tensor (zero-copy, same device) for oracle comparisons."""
    import torch  # tests are dev-side; torch is available
    return a if a.__class__.__module__.startswith("torch") else wp.to_torch(a)
```

- [ ] **Step 5: `inflate_warp` writes into caller-owned `Track` buffers**

Change the signature to `def inflate_warp(center, config, out, valid=None, count=None) -> None`
where `out` is a `Track` whose `wp.array` fields are pre-allocated `[E, N_max]`. Keep the existing
per-stage torch computation, but at the end, instead of `return Track(outer=..., ...)`, **copy each
final result into `out`'s buffer**: `wp.copy(out.center, wp.from_torch(rs.reshape(E*n_max,2),
dtype=wp.vec2f))` (and likewise outer/inner/tangent/normal/arclen/length/valid/count). This is the
temporary boundary copy; Inc 2+ remove it as stages write `out`'s buffers directly. Add an
internal `_inflate_warp_alloc(config) -> Track` that allocates the `wp.array` Track buffers (used
by `TrackGenerator.__init__`).

- [ ] **Step 6: `TrackGenerator` owns the Track + returns it persistently**

In `track_gen/_src/track_generator.py`, in `__init__` allocate the output `Track` once via
`_inflate_warp_alloc(config)` and store `self._track`. `generate()` runs the pipeline (still via the
now-private `generate_tracks_warp` helper, but passing `out=self._track` to `inflate_warp`) and
returns `self._track` — the **same instance every call**. (The pipeline's internal stages are
unchanged this increment.)

- [ ] **Step 7: Migrate the Track-consuming tests to the `wp.to_torch` boundary**

In `test_warp_inflate.py`, `test_warp_pipeline_e2e.py`, `test_warp_generate.py`,
`test_track_generator_facade.py`, and the other field-accessor tests, wrap every `track.<field>`
read used in a torch op with `to_t(...)` from `tests/_warp_compare.py` (e.g.
`to_t(track.center)`), and where they called `inflate_warp` directly, pre-allocate an `out` Track
(`from track_gen._src.warp_pipeline import _inflate_warp_alloc`) and pass it. Oracle comparisons
stay torch.

- [ ] **Step 8: Stable-`.ptr` test** — add to `tests/test_warp_generate.py` (or a new
`tests/test_buffer_reuse.py`):
```python
def test_generate_reuses_output_buffers():
    import warp as wp, torch
    gen = _make_generator(E=8)            # construct TrackGenerator (mirror existing helpers)
    t1 = gen.generate(); p1 = t1.center.ptr
    view = wp.to_torch(t1.center)         # take a torch view once
    t2 = gen.generate(); 
    assert t2 is t1 and t2.center.ptr == p1, "Track buffers must be reused in place"
    assert torch.equal(view, wp.to_torch(t2.center)), "torch view must reflect the 2nd run"
```

- [ ] **Step 9: Full suite** — `.venv/bin/python -m pytest -q` → **green** (232 + the new
buffer-reuse test = 233; the per-kernel stage tests are untouched this increment and still pass).

- [ ] **Step 10: Commit** (`--no-gpg-sign`):
`refactor(warp): Track->wp.array, TrackGenerator owns persistent output buffers (M2 inc1)`

---

## Self-Review

- **Spec coverage:** Inc 1 covers spec §1 (Track→wp.array) + the output half of §5 (buffer
  ownership, stable pointers, same-instance return) + §7 test boundary + the stable-`.ptr` guard.
  The per-stage in-place rewrite (§2 strict, the bulk of §5) and §6 deps/§ free-function removal are
  the later increments, explicitly sequenced above and detailed just-in-time.
- **Placeholders:** Inc 1 is concrete; Inc 2+ are an ordered list to be detailed per-increment
  (each is a small, uniform stage conversion — not a placeholder but a deliberate just-in-time plan
  for a multi-session rewrite).
- **Consistency:** `inflate_warp(out)` + `_inflate_warp_alloc` + `self._track` + `to_t` helper are
  used consistently across Steps 5–8.
- **Risk note:** the temporary boundary copy in Step 5 (`wp.copy(out.*, wp.from_torch(torch_result))`)
  keeps internals torch this increment; it is removed stage-by-stage in Inc 2+. This is the
  intended scaffold, not the end state.
