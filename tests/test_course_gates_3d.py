"""Gates-mode Course facade: 3D CourseLine + TrackLocalizer integration."""
import numpy as np
import pytest
import warp as wp

from track_gen._src.course import Course, CourseConfig
from track_gen._src.types import GateGenConfig

E = 4


def _course(**kw):
    # scale=3.0: at the default scale=1.0, 0.2-wide gates overlap/cross and
    # the finalizer rejects every env (seeds=11 yields valid == [0,0,0,0]),
    # making the localization test vacuous. A larger loop keeps all 4 valid.
    gcfg = GateGenConfig(device="cpu", num_envs=E, gate_width=0.2,
                         scale=3.0,
                         z_profile="uniform", z_min=0.5, z_max=1.5)
    c = Course(CourseConfig(mode="gates", gen=gcfg, seeds=11, **kw))
    pos = wp.zeros(E, dtype=wp.vec3f, device="cpu")
    c.bind(pos)
    c.generate()
    return c, pos


def test_gates_course_has_localizer_and_line():
    c, _ = _course()
    assert c.course_line is not None
    assert c.localizer is not None
    assert int(c.course_line.track.valid.numpy().sum()) \
        == int(c.result.valid.numpy().sum())


def test_step_localizes_near_first_gate():
    c, pos = _course()
    e = int(np.flatnonzero(c.result.valid.numpy())[0])
    G = c.result.position.shape[0] // E
    gate0 = c.result.position.numpy()[e * G]
    p = pos.numpy()
    p[e] = gate0 + np.array([0.0, 0.0, 0.1], np.float32)
    wp.copy(pos, wp.array(p, dtype=wp.vec3f, device="cpu"))
    res = c.step()
    frame = res.frame
    assert frame is not None
    # foot point at gate 0 => s near 0 (mod length), n_up near +0.1
    L = float(c.course_line.track.length.numpy()[e])
    s = float(frame.s.numpy()[e])
    assert min(s, L - s) < 0.2 * L
    assert abs(float(frame.n_up.numpy()[e]) - 0.1) < 0.05


def test_track_mode_rejects_gate_options():
    from track_gen._src.types import TrackGenConfig
    tcfg = TrackGenConfig(device="cpu", num_envs=E)
    with pytest.raises(ValueError):
        CourseConfig(mode="track", gen=tcfg, checkpoint_spacing=0.1,
                     samples_per_gate=4)
    with pytest.raises(ValueError):
        CourseConfig(mode="track", gen=tcfg, checkpoint_spacing=0.1,
                     localize_window=4)


@pytest.mark.cuda
def test_cuda_gates_graph_replay_refreshes_line_and_frame():
    """Graph B (captured refresh) must contain the CourseLine resample and
    the localizer reset: poison both, replay via generate(seeds=...), and
    check they are recomputed; then step() localizes on the new line."""
    if wp.get_cuda_device_count() == 0:
        pytest.skip("cuda")
    dev = "cuda:0"
    from track_gen._src.course_line import CourseLine
    gcfg = GateGenConfig(device=dev, num_envs=E, gate_width=0.2, scale=3.0,
                         z_profile="uniform", z_min=0.5, z_max=1.5)
    c = Course(CourseConfig(mode="gates", gen=gcfg, seeds=11,
                            localize_window=4))
    pos = wp.zeros(E, dtype=wp.vec3f, device=dev)
    c.bind(pos)
    c.generate()
    assert c._refresh_graph is not None          # Graph B captured
    c.course_line.track.center.fill_(12345.0)    # poison the line ...
    c.localizer._last.fill_(5)                    # ... and the warm memory
    c.generate(seeds=77)                          # replay path
    ref = CourseLine(c.result, 8)
    ref.refresh()
    np.testing.assert_allclose(c.course_line.track.center.numpy(),
                               ref.track.center.numpy(), rtol=1e-5,
                               equal_nan=True)
    assert (c.localizer._last.numpy() == -1).all()

    valid = np.flatnonzero(c.result.valid.numpy())
    assert valid.size > 0
    e = int(valid[0])
    G = c.result.position.shape[0] // E
    gate0 = c.result.position.numpy()[e * G]
    p = pos.numpy()
    p[e] = gate0 + np.array([0.0, 0.0, 0.1], np.float32)
    wp.copy(pos, wp.array(p, dtype=wp.vec3f, device=dev))
    frame = c.step().frame
    assert frame is not None
    L = float(c.course_line.track.length.numpy()[e])
    s = float(frame.s.numpy()[e])
    assert min(s, L - s) < 0.2 * L
    assert abs(float(frame.n_up.numpy()[e]) - 0.1) < 0.05
