"""The gate Program is byte-identical to 0.2.0.dev3 at every stage boundary (roadmap PR11).

``tests/goldens/pr11_gate_stages.json`` holds, per case, one digest of the arrays the
LIVE dd181f0 gate pipeline produced at every stage boundary
(``tests/tools/capture_pr11_gate_stages.py``), and the launch sequence of one
``generate()`` for a subset. ``tests/_legacy_gate_pr11.py`` is the frozen reference (the
dd181f0 code PR11 edited or deleted; the unchanged kernels imported live):

- ``test_reference_is_frozen``: the reference reproduces the dd181f0 digests (so it can
  stand in for the deleted path, and an unchanged kernel imported live cannot drift);
- ``test_every_stage_boundary_matches_head``: every Program tap (``program.tap(channel,
  after=stage_id)``) equals the reference array, raw bytes (NaN padding included); a
  masked regeneration then rebuilds the masked envs at their new effective seeds and
  leaves the others byte-unchanged;
- ``test_launch_sequence_matches_head``: the same kernels, dims and scalars, in the same
  order (polar / checkpoint / voronoi: the normalization moved after the count copy and
  runs out of place, ``_normalize_extent_to_k``: the one sequence delta of PR11; the
  launch count is unchanged). Roadmap PR13c re-pin (deliberate, the rule of
  ``tests/_pr13c_delta.py`` applied to the recorded sequence; the golden is not
  re-captured): the polar / checkpoint / voronoi gate generators latch their two dynamic
  knobs and launch the anchor twin (+2), the noise elevation latches its three (+3) and
  launches ``_z_noise_dyn_k``.

Float results depend on the platform and the Warp version: the golden is keyed by the PR7
platform key and the digest tests skip on a key it does not hold (the tap-vs-reference
comparison runs everywhere).
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests import _legacy_gate_pr11 as ref_mod
from tests._pr13c_delta import ELEVATION_LATCHES, GATE_GENERATOR_LATCHES, pr13c_sequence
from tests.tools import capture_pr11_gate_stages as tool
from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
from track_gen.seeds import effective_seed
from track_gen.testing import trace_launches

E = tool.E
MASK = (1, 0, 1, 0)


def _golden():
    return tool.load().get(tool.platform_key())


def _cfg(kw) -> GateGenConfig:
    return GateGenConfig(num_envs=E, device="cpu", **kw)


def _rng():
    return PerEnvSeededRNG(seeds=wp.array(np.asarray(tool.SEEDS, np.int32), dtype=wp.int32,
                                          device="cpu"), num_envs=E, device="cpu")


def _taps(gen) -> dict:
    """The Program's arrays under the reference's tap names."""
    program, seq = gen.program, gen._gate_sequence

    def tap(key):
        if key in ref_mod.FINAL_FIELDS:
            return getattr(seq, key)
        if key == "fallbacks":
            return gen.frame_fallbacks
        if "@" in key:
            ch, after = key.split("@")
            return program.tap(ch, after=after)
        return program.tap(key)
    return tap


def _compare(gen, want: dict) -> list:
    tap = _taps(gen)
    bad = []
    for key, arr in sorted(want.items()):
        got = tap(key).numpy()
        if got.dtype != arr.dtype or got.shape != arr.shape or \
                got.tobytes() != arr.tobytes():
            bad.append(key)
    return bad


def test_reference_is_frozen():
    golden = _golden()
    if golden is None:
        pytest.skip(f"no dd181f0 gate-stage digests for {tool.platform_key()}")
    from track_gen.plugin import registry

    bad = [case for case, kw in tool.CASES.items()
           if tool.digest(ref_mod.run_reference(_cfg(kw), tool.SEEDS))
           != golden["digests"][case]]
    with registry.scoped():
        tool.register_legacy(registry)
        for case, kw in tool.LEGACY_CASES.items():
            spec = registry.get("gate_generator", kw["generator"])
            taps = ref_mod.run_reference_legacy(_cfg(kw), spec, tool.SEEDS)
            if tool.digest(taps) != golden["digests"][case]:
                bad.append(case)
    assert not bad, f"{len(bad)} cases differ from dd181f0: {bad[:10]}"


def _rows(arr: np.ndarray, e: int, per_env: int) -> bytes:
    return arr[e * per_env:(e + 1) * per_env].tobytes()


@pytest.mark.parametrize("case", sorted(tool.CASES))
def test_every_stage_boundary_matches_head(case):
    kw = tool.CASES[case]
    cfg = _cfg(kw)
    gen = GateGenerator(cfg, _rng())
    seq = gen.generate()
    want = ref_mod.run_reference(cfg, tool.SEEDS)
    bad = _compare(gen, want)
    assert not bad, f"{case}: taps differ from dd181f0: {bad}"
    fm, valid = seq.fail_mask.numpy(), seq.valid.numpy()
    assert np.array_equal(valid, (fm == 0).astype(valid.dtype))

    # Masked regeneration: masked envs at their new effective seed, others untouched.
    before = {f: getattr(seq, f).numpy().copy() for f in ref_mod.FINAL_FIELDS}
    mask = wp.array(np.asarray(MASK, np.int32), dtype=wp.int32, device="cpu")
    gen.generate(advance=True, mask=mask)
    new_seeds = [effective_seed(s, 1) if m else s for s, m in zip(tool.SEEDS, MASK)]
    want2 = ref_mod.run_reference(cfg, new_seeds)
    G = int(cfg.max_gates)
    for f in ref_mod.FINAL_FIELDS:
        got = getattr(seq, f).numpy()
        n = G if got.shape[0] == E * G else 1
        for e in range(E):
            expect = want2[f] if MASK[e] else before[f]
            assert _rows(got, e, n) == _rows(expect, e, n), (case, f, e)


@pytest.mark.parametrize("case", sorted(tool.LEGACY_CASES))
def test_legacy_adapter_matches_head(case, plugin_registry):
    kw = tool.LEGACY_CASES[case]
    tool.register_legacy(plugin_registry)
    cfg = _cfg(kw)
    with pytest.warns(DeprecationWarning, match="GateGeneratorSpec"):
        gen = GateGenerator(cfg, _rng())
    gen.generate()
    spec = plugin_registry.get("gate_generator", kw["generator"])
    want = ref_mod.run_reference_legacy(cfg, spec, tool.SEEDS)
    bad = _compare(gen, want)
    assert not bad, f"{case}: taps differ from dd181f0: {bad}"


def _expected_launches(case: str, rows: list) -> list:
    """The dd181f0 sequence; a normalizing generator's ``[normalize, copy(count)]``
    becomes ``[copy(count), normalize]`` (the order stage copies the count, then the
    normalize stage runs the same arithmetic out of place, ``_normalize_extent_to_k``:
    same dims and scalars, one more array argument), so PR11 left the launch count
    unchanged. Roadmap PR13c: the latches of the generator's and the elevation's dynamic
    knobs follow the effective seed, and their kernels are the array-reading twins."""
    gen = case.split("/")[0]
    n_latch = GATE_GENERATOR_LATCHES[gen] + ELEVATION_LATCHES[case.split("/")[2]]
    rows = pr13c_sequence(rows, n_latch, E)
    if gen not in tool.NORMALIZING:
        return rows
    i = next(k for k, r in enumerate(rows) if r[1].endswith("_normalize_extent_count_k"))
    normalize, count_copy = rows[i], rows[i + 1]
    assert count_copy[0] == "copy"
    normalize = [normalize[0], "track_gen.stages.gates._normalize_extent_to_k",
                 normalize[2], normalize[3]]
    return rows[:i] + [count_copy, normalize] + rows[i + 2:]


@pytest.mark.parametrize("case", tool.LAUNCH_CASES)
def test_launch_sequence_matches_head(case):
    golden = _golden()
    if golden is None:
        pytest.skip(f"no dd181f0 gate-stage digests for {tool.platform_key()}")
    gen = GateGenerator(_cfg(tool.CASES[case]), _rng())
    with trace_launches() as trace:
        gen.generate()
    assert tool.launch_rows(trace) == _expected_launches(case, golden["launches"][case])


def test_default_bezier_ccw_emits_14_records():
    gen = GateGenerator(GateGenConfig(num_envs=E, device="cpu"), _rng())
    with trace_launches() as trace:
        gen.generate()
    assert len(trace) == 14


def test_relax_early_return_keeps_positions():
    """A ``_relax_gate_spheres_k`` launch whose float32 distance or iteration count makes
    it return early still copies its input row (the output version is never left at its
    zero initialisation). Params validation refuses such a stage in a spec, so this
    drives the kernel directly."""
    from track_gen._src import warp_gate

    G = 4
    src_np = np.array([[0.0, 0.0], [0.01, 0.0], [np.nan, np.nan], [np.nan, np.nan]] * 2,
                      dtype=np.float32)
    src = wp.array(src_np, dtype=wp.vec2f, device="cpu")
    count = wp.array([2, 2], dtype=wp.int32, device="cpu")
    for distance, iters in ((1.0e-46, 8), (0.05, 0)):
        dst = wp.zeros(2 * G, dtype=wp.vec2f, device="cpu")
        wp.launch(warp_gate._relax_gate_spheres_k, 2,
                  [dst, G, count, float(distance), int(iters), src], device="cpu")
        assert dst.numpy().tobytes() == src.numpy().tobytes()
