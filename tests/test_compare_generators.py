import dataclasses
from benchmarks import compare_generators as cg
from track_gen._src.types import TrackGenConfig

_EXPECTED_KEYS = {
    "generator", "yield", "pre_relax_self_intersection_rate", "xpbd_displacement",
    "mean_length", "mean_compactness", "peak_curvature", "lap_time",
    "gen_ms_per_call",
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
