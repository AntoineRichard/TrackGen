"""Validity as fail bits (roadmap PR10b, CPU).

``validity.default`` keeps its fused ``_validity_k`` (one launch, ``Track.valid``
byte-identical) and also writes one ``valid.fail`` bit per failed check (bits 0-7,
``BUILTIN_FAIL_BITS``); plugin validity stages get bits 16-30 from
``StageBuilder.fail_bit`` in pipeline order. ``Track.fail_mask`` is the final
``valid.fail`` and ``Track.valid == (fail_mask == 0)``. Covered: the table, the per-check
attribution of the kernel on hand-built loops, every parity case, the bit allocator and
its errors.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import warp as wp

from tests.test_pipeline_parity import CASES, _make
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import warp_pipeline
from track_gen.pipeline import BUILTIN_FAIL_BITS, PlanError, StageRef, build, presets
from track_gen.plugin import Stage, registry

DEV = "cpu"
E = 2

# ---------------------------------------------------------------------------------------
# The table, the default pipeline
# ---------------------------------------------------------------------------------------


def test_the_builtin_fail_bit_table():
    assert dict(BUILTIN_FAIL_BITS) == {
        "generator": 0, "turning": 1, "finite": 2, "thickness": 3, "w_floor": 4,
        "border": 5, "grade": 6, "truncation": 7}
    with pytest.raises(TypeError):
        BUILTIN_FAIL_BITS["x"] = 9                                   # read-only


def test_the_default_pipeline_names_the_builtin_bits():
    gen = TrackGenerator(TrackGenConfig(device=DEV, num_envs=E),
                         PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    assert dict(gen.program.fail_bits) == dict(BUILTIN_FAIL_BITS)
    assert dict(gen._track.fail_bits) == dict(BUILTIN_FAIL_BITS)
    assert gen._track.fail_mask.dtype is wp.int32 and gen._track.fail_mask.shape == (E,)
    assert gen.program.tap("valid.fail") is gen._track.fail_mask
    assert gen.program.channels["valid.fail"] == ("validity",)
    assert gen._track.clone().fail_bits is gen._track.fail_bits


@pytest.mark.parametrize("case", sorted(CASES))
def test_valid_is_fail_mask_zero_for_every_parity_case(case):
    gen = _make(case)
    for _ in range(2):
        t = gen._track
        fm, valid = t.fail_mask.numpy(), t.valid.numpy()
        np.testing.assert_array_equal(valid, (fm == 0).astype(np.int32), err_msg=case)
        assert not (fm & ~0xFF).any(), f"{case}: a bit outside 0-7: {fm.tolist()}"
        if gen._config.validity_border_check is False:
            assert not (fm & 32).any(), case                       # border off
        if float(gen._config.z_valid_grade) == 0.0:
            assert not (fm & 64).any(), case                       # grade off
        np.testing.assert_array_equal((fm >> 7) & 1, t.truncated.numpy(), err_msg=case)
        gen.generate(advance=True)


# ---------------------------------------------------------------------------------------
# The kernel's attribution, on hand-built loops
# ---------------------------------------------------------------------------------------

N = 64          # stride
M = 48          # real points


def _circle(rx=1.0, ry=1.0, n=M):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([rx * np.cos(t), ry * np.sin(t)], 1).astype(np.float32)


def _figure_eight(n=M):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.sin(t), np.sin(t) * np.cos(t)], 1).astype(np.float32)


def _gate(center, *, count=M, gen_valid=1, truncated=0, w=None, outer=None, z=None,
          has_border=0, z_valid_grade=0.0, half_width=0.1) -> int:
    """Run validity_inplace on one env; returns its fail bits (asserting valid ==
    (fail == 0))."""
    def pad(a, dim):
        out = np.full((N, dim) if dim > 1 else (N,), np.nan, np.float32)
        out[:len(a)] = a
        return out

    c = pad(center, 2)
    wv = np.full(N, 0.1, np.float32) if w is None else w
    o = pad(outer if outer is not None else center * 1.1, 2)
    i = pad(center * 0.9, 2)
    zv = np.zeros(N, np.float32) if z is None else z
    cfg = types.SimpleNamespace(half_width=half_width, turning_tol=0.1, w_floor=1e-3,
                                relax_tol=0.02, z_valid_grade=z_valid_grade)

    def arr(a, dt):
        return wp.array(a, dtype=dt, device=DEV)

    valid = wp.zeros(1, dtype=wp.int32, device=DEV)
    fail = wp.full(1, -1, dtype=wp.int32, device=DEV)
    warp_pipeline.validity_inplace(
        arr(c, wp.vec2f), arr(wv, wp.float32), arr(np.array([count], np.int32), wp.int32),
        arr(np.array([gen_valid], np.int32), wp.int32), arr(o, wp.vec2f), arr(i, wp.vec2f),
        has_border, N, arr(zv, wp.float32), valid, cfg,
        truncated_wp=arr(np.array([truncated], np.int32), wp.int32), fail_wp=fail)
    v, f = int(valid.numpy()[0]), int(fail.numpy()[0])
    assert v == int(f == 0), (v, f)
    return f


def test_a_good_loop_sets_no_bit():
    assert _gate(_circle()) == 0


def test_each_check_sets_its_own_bit():
    assert _gate(_circle(), truncated=1) == 1 << 7                  # truncation only
    assert _gate(_circle(), gen_valid=0) == 1 << 0                  # generator only
    w = np.full(N, 0.1, np.float32)
    w[3] = 0.0
    assert _gate(_circle(), w=w) == 1 << 4                          # w_floor only
    nan = _circle()
    nan[5] = np.nan
    assert _gate(nan) & (1 << 2)                                    # finite
    assert _gate(_circle(1.0, 0.05)) & (1 << 3)                     # thin: thickness
    assert not _gate(_circle(1.0, 0.05)) & (1 << 1)                 # ... still one turn
    assert _gate(_figure_eight()) & (1 << 1)                        # turning
    crossed = _circle() * 1.1
    crossed[[10, 11]] = crossed[[11, 10]]                           # a bow-tie in outer
    assert _gate(_circle(), outer=crossed, has_border=1) == 1 << 5   # border only
    assert _gate(_circle(), outer=crossed, has_border=0) == 0        # ... when checked
    z = np.zeros(N, np.float32)
    z[:M:2] = 1.0                                                   # a 1.0 step per point
    assert _gate(_circle(), z=z, z_valid_grade=0.5) == 1 << 6        # grade only
    assert _gate(_circle(), z=z, z_valid_grade=0.0) == 0             # ... when capped


def test_an_empty_loop_fails_turning():
    assert _gate(_circle(), count=0) == 1 << 1
    assert _gate(_circle(), count=0, gen_valid=0, truncated=1) == (1 << 1) | 1 | (1 << 7)


def test_validity_inplace_without_fail_wp_still_runs():
    c = np.full((N, 2), np.nan, np.float32)
    c[:M] = _circle()
    valid = wp.zeros(1, dtype=wp.int32, device=DEV)
    cfg = types.SimpleNamespace(half_width=0.1, turning_tol=0.1, w_floor=1e-3,
                                relax_tol=0.02, z_valid_grade=0.0)
    one = wp.array([1], dtype=wp.int32, device=DEV)
    warp_pipeline.validity_inplace(
        wp.array(c, dtype=wp.vec2f, device=DEV), wp.full(N, 0.1, dtype=wp.float32,
                                                        device=DEV),
        wp.array([M], dtype=wp.int32, device=DEV), one,
        wp.array(c, dtype=wp.vec2f, device=DEV), wp.array(c, dtype=wp.vec2f, device=DEV),
        0, N, wp.zeros(N, dtype=wp.float32, device=DEV), valid, cfg)
    assert valid.numpy().tolist() == [1]


# ---------------------------------------------------------------------------------------
# The allocator (plugin checks get 16..30, in pipeline order)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _pass_k(fail_in: wp.array(dtype=wp.int32), fail_out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    fail_out[e] = fail_in[e]


def _check(name="test.check", body=None, kind="validity"):
    """A validity check: ``body(b)`` runs inside build (default: one fail_bit())."""
    def build(self, b, p):
        if body is not None:
            body(b)
        else:
            b.fail_bit()
        fin, fout = b.read("fail"), b.write("fail", wp.int32)
        return lambda ctx: ctx.launch(_pass_k, b.E, [fin, fout])

    ns = dict(kind=kind, name=name, reads={"fail": "valid.fail"},
              writes={"fail": "valid.fail"}, build=build)
    return type("Check", (Stage,), ns)


def _spec(*checks):
    return presets.track(num_envs=E, device=DEV, checks=("default", *checks))


def test_plugin_bits_are_assigned_in_pipeline_order():
    registry.add(_check())
    registry.add(_check("test.named", body=lambda b: (b.fail_bit("a"), b.fail_bit("b"))))
    p = build(_spec(StageRef("validity", "test.check", id="c1"), "test.named",
                    StageRef("validity", "test.check", id="c2")), seeds=1)
    plugin = [(n, b) for n, b in p.fail_bits.items() if b >= 16]
    assert plugin == [("c1", 16), ("a", 17), ("b", 18), ("c2", 19)]
    assert list(p.fail_bits)[:8] == list(BUILTIN_FAIL_BITS)
    assert p._fail_owner["a"] == "test.named" and p._fail_owner["c2"] == "c2"
    assert dict(p.result.fail_bits) == dict(p.fail_bits)


def test_fail_bit_errors():
    registry.add(_check("test.clash", body=lambda b: b.fail_bit("turning")))
    with pytest.raises(PlanError, match=r"stage 'test.clash' \(validity.test.clash\): fail "
                                        r"bit name 'turning' is already taken by stage "
                                        r"'validity'"):
        build(_spec("test.clash"))
    registry.add(_check("test.greedy",
                        body=lambda b: [b.fail_bit(f"x{i}") for i in range(16)]))
    with pytest.raises(PlanError, match=r"no plugin fail bit left: bits 16-30 are taken "
                                        r"by \['x0', .*'x14'\]; a pipeline holds at most "
                                        r"15 plugin fail bits"):
        build(_spec("test.greedy"))
    registry.add(_check("test.lazy", body=lambda b: None))
    with pytest.raises(PlanError, match=r"stage 'test.lazy' \(validity.test.lazy\) never "
                                        r"called b.fail_bit\(\): a validity stage owns at "
                                        r"least one fail bit"):
        build(_spec("test.lazy"))
    registry.add(_check("test.thief", body=lambda b: b._reserve_fail_bits({"mine": 9})))
    with pytest.raises(PlanError, match="bits 0-15 are reserved for TrackGen's built-in "
                                        "checks"):
        build(_spec("test.thief"))
    # Built-in-ness is the registry's record, not the class's module name.
    registry.add(_check("test.disguised", body=lambda b: b._reserve_fail_bits({"m": 9})))
    registry.get("validity", "test.disguised").__module__ = "track_gen.stages.fake"
    with pytest.raises(PlanError, match="bits 0-15 are reserved for TrackGen's built-in "
                                        "checks"):
        build(_spec("test.disguised"))


def test_valid_fail_starts_at_validity_default():
    """A check that writes valid.fail without reading it would replace validity.default
    and drop the built-in checks (truncation would fail open): refused at plan time."""
    def build_(self, b, p):
        b.fail_bit()
        b.write("fail", wp.int32)
        return lambda ctx: None

    Gate = type("Gate", (Stage,), dict(kind="validity", name="test.gate",
                                       writes={"fail": "valid.fail"}, build=build_))
    registry.add(Gate)
    with pytest.raises(PlanError, match=r"stage 'validity' \(validity.test.gate\) starts "
                                        r"valid.fail .* in place of validity.default, which "
                                        r"would drop the built-in generator, turning, "
                                        r"thickness and truncation checks"):
        build(_spec().replace("validity", StageRef("validity", "test.gate")))
    with pytest.raises(ValueError, match="checks must start with 'default'"):
        presets.track(num_envs=E, device=DEV, checks=("test.gate",))


def test_fail_bit_from_another_kind_raises():
    def build_(self, b, p):
        b.fail_bit()

    Relax = type("Relax", (Stage,), dict(kind="relax", name="test.relax",
                                         reads={"xy": "cl.xy"}, writes={"xy": "cl.xy"},
                                         build=build_))
    registry.add(Relax)
    with pytest.raises(ValueError, match=r"stage 'relax' \(Relax\): fail_bit\(\) is for "
                                         r"validity stages; this stage is kind 'relax'"):
        build(presets.track(num_envs=E, device=DEV, relax="test.relax"))
