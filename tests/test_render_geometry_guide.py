"""Numerical contracts behind the racing geometry figures, without pixel comparisons."""
import numpy as np


def test_geometry_cases_are_measured_and_nonvacuous():
    from viz.render_collision_guide import collect_cases

    cases = collect_cases()
    validity = cases["validity"]
    assert validity[0]["valid"] == 1
    assert all(row["count"] > 3 for row in validity)
    assert any(row["center_crossings"] > 0 for row in validity)
    assert validity[-1]["truncated"] == 1 and validity[-1]["fail_mask"] & 128
    assert [row["crossings"] for row in cases["predicates"]] == [1, 0, 0]
    assert cases["predicates"][0]["thickness"] > 0.1  # crossing between distant samples
    footprints = cases["footprints"]
    assert [r["contact"]["oob"] for r in footprints] == [0, 1, 1, 1]
    assert footprints[1]["contact"]["distance"] == 0.0  # side crossing, all corners on road
    assert footprints[3]["contact"]["distance"] == 0.0  # inclusive touch
    sdf = cases["sdf"]
    assert [row["contact"]["oob"] for row in sdf] == [1, 0, 1]
    assert sdf[1]["cell_width"] > sdf[2]["cell_width"]
    assert [row["hit"] for row in cases["discs"]] == [0, 1, 1]
    assert cases["discs"][1]["depth"] == 0.0
    assert [row["hit"] for row in cases["frames"]] == [0, 1, 1]
    assert all(np.isfinite(row["depth"]) for row in cases["frames"])


def test_video_prefix_capture_is_real_and_has_separate_tail_phases():
    from viz.render_relaxation_video import capture_track_states, verify_prefix_schedule

    verify_prefix_schedule(12)
    states, metadata = capture_track_states(batch=2, sweeps=3, smooth_passes=2,
                                             polish=2, seed=17, device="cpu")
    assert [s["phase"] for s in states] == ["raw", "sweep", "sweep", "sweep",
                                           "Taubin", "Taubin", "polish", "polish"]
    assert metadata["on_invalid"] == "keep"
    assert metadata["prefix_verified"] is True
    assert states[4]["config"]["relax_smooth_spacing_iters"] == 0
    assert states[5]["config"]["relax_smooth_passes"] == 2
    assert states[6]["config"]["relax_smooth_passes"] == 2
    assert states[6]["config"]["relax_smooth_spacing_iters"] == 1
    assert not np.array_equal(states[0]["center"], states[3]["center"], equal_nan=True)
    assert all(np.isfinite(s["center"][e, :s["count"][e]]).all()
               for s in states for e in range(2))
