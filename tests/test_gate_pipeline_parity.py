"""Independent scalar references and current Program taps agree after the 0.3 RNG switch.

The platform goldens retain every gate boundary and selected launch sequences.
Scalar references use the same frozen stream formulas with independent composition;
masked regeneration must leave unselected rows byte-identical.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests import _legacy_gate_pr11 as ref_mod
from tests.tools import capture_pr11_gate_stages as tool
from tests.tools._golden_config import keep_config
from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
from track_gen.seeds import effective_seed
from track_gen.testing import trace_launches

E = tool.E
MASK = (1, 0, 1, 0)


def _golden():
    return tool.load().get(tool.platform_key())


def _cfg(kw) -> GateGenConfig:
    return keep_config(GateGenConfig, num_envs=E, device="cpu", **kw)


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
        if key == "gates.xy@generator":
            return program.tap("gates.xy", after="order")
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
        pytest.skip(f"no 0.3 gate-stage digests for {tool.platform_key()}")
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
    assert not bad, f"{len(bad)} cases differ from the 0.3 gate baseline: {bad[:10]}"


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
    assert not bad, f"{case}: taps differ from the independent reference: {bad}"
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
    gen = GateGenerator(cfg, _rng())
    gen.generate()
    spec = plugin_registry.get("gate_generator", kw["generator"])
    want = ref_mod.run_reference_legacy(cfg, spec, tool.SEEDS)
    bad = _compare(gen, want)
    assert not bad, f"{case}: taps differ from the independent reference: {bad}"


@pytest.mark.parametrize("case", tool.LAUNCH_CASES)
def test_launch_sequence_matches_head(case):
    golden = _golden()
    if golden is None:
        pytest.skip(f"no 0.3 gate-stage digests for {tool.platform_key()}")
    gen = GateGenerator(_cfg(tool.CASES[case]), _rng())
    with trace_launches() as trace:
        gen.generate()
    assert tool.launch_rows(trace) == golden["launches"][case]
    assert len(trace) == len(golden["launches"][case])


def test_default_bezier_ccw_emits_16_records():
    """0.2.0.dev3's 14 records + the latches of the bezier gate generator's ``size`` and
    ``gate_frames.gate_width`` (roadmap PR13d re-pin)."""
    gen = GateGenerator(keep_config(GateGenConfig, num_envs=E, device="cpu"), _rng())
    with trace_launches() as trace:
        gen.generate()
    assert len(trace) == 16


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
