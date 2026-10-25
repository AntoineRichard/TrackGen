"""The private barrier frame must match the size used by the actual road."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests.conftest import DEVS
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src.repulsive_clearance import RepulsiveBarriers, publish_barriers_k
from track_gen.exec import LaunchContext
from track_gen.testing.capture import result_arrays
from track_gen.testing.contract import no_allocation, no_host_sync


@pytest.mark.parametrize("dev", DEVS)
def test_barrier_metadata_uses_effective_size_and_preserves_nan_through_capture(dev):
    values = np.array([-np.inf, -1., .01, .1, 1., 10., 100., np.inf, np.nan], np.float32)
    expected = np.clip(values, .1, 10.)
    centers = wp.zeros(len(values), dtype=wp.vec2f, device=dev)
    radii = wp.zeros(len(values), dtype=wp.float32, device=dev)
    sizes = wp.array(values, dtype=wp.float32, device=dev)
    metadata = wp.empty(len(values), dtype=RepulsiveBarriers, device=dev)
    ctx = LaunchContext.current(dev)
    args = [centers, radii, 1, 2., sizes, metadata]
    ctx.launch(publish_barriers_k, len(values), args)
    np.testing.assert_array_equal(metadata.numpy()["size"], expected)
    with wp.ScopedCapture(device=dev) as capture:
        with no_allocation(), no_host_sync():
            ctx.launch(publish_barriers_k, len(values), args)
    metadata.zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(metadata.numpy()["size"], expected)


def _small(bound, dev, num_envs=2):
    width, spacing = (.002, .005) if bound == .1 else (1., .6)
    cfg = TrackGenConfig(generator="repulsive", device=dev, num_envs=num_envs,
                         num_points=16, N_max=512, half_width=width, spacing=spacing,
                         size=bound, repulsive_stages=(16,),
                         repulsive_grow_mult_min=1., repulsive_grow_mult_max=1.01,
                         repulsive_settle_iters=1, on_invalid="keep")
    seeds = wp.array(np.full(num_envs, 7, np.int32), dtype=wp.int32, device=dev)
    return TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds, num_envs=num_envs, device=dev))


def _bytes(result):
    return {name: value.numpy().tobytes() for name, value in result_arrays(result).items()}


@pytest.mark.parametrize("bound,values", [(.1, [.01, -1., -np.inf]),
                                           (10., [100., np.inf])])
@pytest.mark.parametrize("dev", DEVS)
def test_public_size_arrays_match_bound_with_mask_and_captured_replay(dev, bound, values):
    gen, reference = _small(bound, dev), _small(bound, dev)
    assert _bytes(gen.generate()) == _bytes(reference.generate())
    def first_row(result):
        return {name: value.numpy().reshape(2, -1)[0].tobytes()
                for name, value in result_arrays(result).items()}
    untouched = first_row(gen.program.result)
    mask = wp.array(np.array([0, 1], np.int32), dtype=wp.int32, device=dev)
    for value in values:
        incoming = wp.array(np.array([bound, value], np.float32), dtype=wp.float32, device=dev)
        gen.params["generator"].set("size", incoming)
        expected = _bytes(reference.generate(advance=True, mask=mask))
        result = gen.generate(advance=True, mask=mask)
        assert gen.program.tap("raw.xy").numpy().tobytes() == reference.program.tap("raw.xy").numpy().tobytes()
        assert _bytes(result) == expected
        assert first_row(result) == untouched
        np.testing.assert_array_equal(gen.program.tap("gen.repulsive.barriers").numpy()["size"],
                                      np.full(2, bound, np.float32))
        with wp.ScopedCapture(device=dev) as capture:
            with no_allocation(), no_host_sync():
                gen.generate(advance=True, mask=mask)
        wp.capture_launch(capture.graph)
        expected = _bytes(reference.generate(advance=True, mask=mask))
        assert _bytes(gen.program.result) == expected
        assert first_row(gen.program.result) == untouched


def test_above_bound_size_cannot_change_validity_of_the_same_finished_road():
    cfg = TrackGenConfig(generator="repulsive", device="cpu", num_envs=1,
                         half_width=1., spacing=.6, size=10.,
                         repulsive_grow_mult_min=3., repulsive_grow_mult_max=3.5,
                         on_invalid="keep")
    seeds = wp.array(np.array([7], np.int32), dtype=wp.int32, device="cpu")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds, num_envs=1, device="cpu"))
    before = _bytes(gen.generate())
    raw = gen.program.tap("raw.xy").numpy().tobytes()
    gen.params["generator"].set("size", wp.array([100.], dtype=wp.float32, device="cpu"))
    result = gen.generate()
    assert gen.program.tap("raw.xy").numpy().tobytes() == raw
    assert _bytes(result) == before


@pytest.mark.parametrize("dev", DEVS)
def test_nan_size_remains_invalid_instead_of_becoming_a_finite_barrier_scale(dev):
    gen = _small(.1, dev, num_envs=1)
    gen.params["generator"].set("size", wp.array([np.nan], dtype=wp.float32, device=dev))
    result = gen.generate()
    assert result.valid.numpy().tolist() == [0]
    assert result.fail_mask.numpy()[0] != 0
    assert np.isnan(gen.program.tap("gen.repulsive.barriers").numpy()["size"][0])
