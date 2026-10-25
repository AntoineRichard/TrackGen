"""The finished Repulsive band must always pass its analytic barrier check."""
from __future__ import annotations

import dataclasses

import pytest

from track_gen.pipeline import StageRef, build, presets
from track_gen.pipeline.planner import PlanError


_KW = dict(repulsive_stages=(16,), repulsive_settle_iters=1,
           repulsive_grow_mult_min=1.0, repulsive_grow_mult_max=1.01)


def _spec(generator="repulsive"):
    return presets.track(generator=generator, num_envs=1, device="cpu", n_max=512, num_points=16,
                         generator_params=_KW if generator == "repulsive" else None,
                         half_width=0.1)


def test_repulsive_barrier_channel_is_declared_and_private():
    program = build(_spec(), prepare=False)
    assert "gen.repulsive.barriers" in program.channels
    assert program.fail_bits["repulsive_disc"] == 8
    assert program.fail_bits["repulsive_wall"] == 9
    assert not program.result.attributes


def test_other_generators_have_no_repulsive_barrier_channel_or_fail_bits():
    program = build(_spec("bezier"), prepare=False)
    assert "gen.repulsive.barriers" not in program.channels
    assert "repulsive_disc" not in program.fail_bits
    assert "repulsive_wall" not in program.fail_bits


def test_repulsive_rejects_geometry_changes_after_final_barrier_check():
    spec = _spec()
    spec = dataclasses.replace(spec, stages=(*spec.stages, StageRef("relax", "xpbd", id="late_relax")))
    with pytest.raises(PlanError, match="Repulsive.*after.*validity"):
        build(spec, prepare=False)


def test_repulsive_cannot_omit_final_barrier_check():
    spec = _spec()
    spec = dataclasses.replace(spec, stages=tuple(s for s in spec.stages if s.kind != "validity"))
    with pytest.raises(PlanError, match="Repulsive requires validity.default"):
        build(spec, prepare=False)


@pytest.mark.parametrize("policy", ["keep", "poison"])
def test_finished_band_wall_failure_uses_existing_commit_policy(policy):
    import numpy as np

    spec = _spec()
    spec = spec.replace("width", StageRef("width", "constant", params={"half_width": 10.0}))
    spec = dataclasses.replace(spec, batch=dataclasses.replace(spec.batch, on_invalid=policy))
    program = build(spec)
    out = program.run()
    assert out.fail_mask.numpy()[0] & (1 << program.fail_bits["repulsive_wall"])
    assert out.valid.numpy()[0] == 0
    if policy == "poison":
        assert out.count.numpy()[0] == 0
        assert np.isnan(out.center.numpy()).all()
    else:
        assert out.count.numpy()[0] >= 3
        assert np.isfinite(out.center.numpy()[:out.count.numpy()[0]]).all()


def test_repulsive_barrier_check_is_capture_safe_and_zero_mask_preserves_output():
    import numpy as np
    import warp as wp

    from track_gen.testing import hot_path_guard

    program = build(_spec())
    with hot_path_guard():
        program.run()
    fields = ("outer", "center", "inner", "count", "valid", "fail_mask", "generation")
    ref = {name: getattr(program.result, name).numpy().tobytes() for name in fields}
    with wp.ScopedCapture(device="cpu") as capture:
        program.run()
    wp.capture_launch(capture.graph)
    assert {name: getattr(program.result, name).numpy().tobytes() for name in fields} == ref
    mask = wp.array(np.zeros(1, np.int32), dtype=wp.int32, device="cpu")
    program.run(advance=True, mask=mask)
    assert {name: getattr(program.result, name).numpy().tobytes() for name in fields} == ref


def test_repulsive_solver_receives_constant_finished_width_reservation():
    program = build(_spec(), prepare=False)
    builder = program._binding.pipeline_spec._built.builder
    assert getattr(builder, "_repulsive_half_width", 0.0) == 0.1


def test_repulsive_validity_binds_caller_owned_view_without_touching_neighbors():
    from track_gen.results import Layout, Track
    from tests._results_helpers import outside_is_poison, poisoned
    from track_gen.testing.capture import result_arrays

    spec = _spec()
    storage = poisoned(Track.alloc(Layout(3, spec.batch.n_max, "cpu")))
    out = storage.view(1, 2)
    program = build(spec, out=out)
    assert program.result is out
    program.run()
    for name, arr in result_arrays(storage).items():
        assert outside_is_poison(arr, 1, 2, spec.batch.n_max,
                                 per_point=name in ("outer", "center", "inner", "tangent", "normal", "arclen", "half_width"))


def test_repulsive_barrier_failure_preserves_unselected_rows():
    import numpy as np
    import warp as wp
    from track_gen.testing.capture import result_arrays

    spec = _spec()
    spec = dataclasses.replace(spec, batch=dataclasses.replace(spec.batch, num_envs=2))
    program = build(spec)
    before = {name: arr.numpy().reshape(2, -1)[0].tobytes()
              for name, arr in result_arrays(program.result).items()}
    mask = wp.array(np.array([0, 1], np.int32), dtype=wp.int32, device="cpu")
    program.run(advance=True, mask=mask)
    after = {name: arr.numpy().reshape(2, -1)[0].tobytes()
             for name, arr in result_arrays(program.result).items()}
    assert after == before
    assert program.result.generation.numpy().tolist() == [0, 1]


def test_repulsive_lift_cannot_export_geometry_other_than_the_checked_band():
    spec = _spec()
    spec = spec.replace("lift", StageRef("lift", "flat", ports={"outer": "cl.normal2"}))
    with pytest.raises(PlanError, match="Repulsive.*lift.*band.outer2"):
        build(spec, prepare=False)
