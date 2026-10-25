import pytest

from benchmarks import compare_generators as cg
from track_gen._src.types import TrackGenConfig

pytestmark = pytest.mark.benchmark

_EXPECTED_KEYS = {
    "generator", "yield", "pre_relax_self_intersection_rate", "xpbd_displacement",
    "mean_length", "mean_compactness", "peak_curvature", "lap_time",
    "gen_ms_per_call", "compactness_degenerate_rate", "shape_variety_pass",
}


def test_run_generator_bezier_smoke():
    cfg = TrackGenConfig(device="cpu", num_envs=16, half_width=0.1)
    row = cg.run_generator("bezier", seed_base=0, E=16, base_config=cfg)
    assert _EXPECTED_KEYS.issubset(row.keys())
    assert 0.0 <= row["yield"] <= 1.0
    assert row["mean_length"] > 0.0


def test_compare_and_format_table():
    cfg = TrackGenConfig(device="cpu", num_envs=16, half_width=0.1)
    rows = cg.compare(["bezier"], seed_base=0, E=16, base_config=cfg)
    assert len(rows) == 1 and rows[0]["generator"] == "bezier"
    table = cg.format_table(rows)
    assert "bezier" in table and "yield" in table


def test_run_generator_polar_passes_roundness_gate():
    cfg = TrackGenConfig(device="cpu", num_envs=32, half_width=0.1)
    row = cg.run_generator("polar", seed_base=0, E=32, base_config=cfg)
    assert _EXPECTED_KEYS.issubset(row.keys())
    assert row["compactness_degenerate_rate"] < 0.25
    assert row["shape_variety_pass"] == 1.0


def test_compare_runs_a_plugin_stage_with_its_generator_params():
    # A plugin generator with Params runs from a plain TrackGenConfig, with its own
    # generator_params (roadmap PR8).
    from tests._plugin_toys import ToyWobble
    from track_gen.plugin import registry

    registry.add(ToyWobble)
    cfg = TrackGenConfig(device="cpu", num_envs=8, half_width=0.1)
    rows = cg.compare(["toy.wobble"], seed_base=0, E=8, base_config=cfg,
                      generator_params={"toy.wobble": {"lobes": 4}})
    assert len(rows) == 1 and rows[0]["generator"] == "toy.wobble"
    assert _EXPECTED_KEYS.issubset(rows[0].keys())
    assert 0.0 <= rows[0]["yield"] <= 1.0


def test_run_generator_never_carries_base_generator_params_over():
    # base_config's generator_params belong to ITS generator; another generator runs
    # with its own (here None) instead of failing on the wrong Params class.
    base = TrackGenConfig(device="cpu", num_envs=8, half_width=0.1, generator="polar",
                          generator_params={"polar_num_knots": 9})
    row = cg.run_generator("bezier", seed_base=0, E=8, base_config=base)
    assert row["generator"] == "bezier"


def test_compare_rejects_generator_params_for_uncompared_names():
    cfg = TrackGenConfig(device="cpu", num_envs=8, half_width=0.1)
    with pytest.raises(ValueError, match="not compared"):
        cg.compare(["bezier"], E=8, base_config=cfg,
                   generator_params={"polra": {"polar_num_knots": 9}})


def test_parse_generator_params():
    assert cg.parse_generator_params('{"repulsive": {"repulsive_stages": [16, 32, 64]}}') == {
        "repulsive": {"repulsive_stages": [16, 32, 64]}}     # the generator tuples it
    with pytest.raises(ValueError, match="JSON object"):
        cg.parse_generator_params('{"polar": 9}')
    with pytest.raises(ValueError):
        cg.parse_generator_params("not json")
