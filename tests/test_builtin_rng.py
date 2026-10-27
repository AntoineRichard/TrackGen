"""Built-in streams use stable hashed namespaces, including the seed-zero batch."""
import dataclasses
import importlib

import numpy as np
import pytest
import warp as wp

from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenerator, ops
from track_gen.exec import LaunchContext
from track_gen.pipeline import presets
from track_gen.plugin.stage import _fnv1a_32
from track_gen.ops.hash import hash32_host
from track_gen.testing.capture import result_arrays


STREAMS = (
    ("warp_generate", "CORNER_COUNT_SALT", "generator.bezier/corner_count", 0xB739AFFB),
    ("warp_generate", "CORNER_SALT", "generator.bezier/corners", 0xC2C74E45),
    ("warp_generate_hull", "POINT_COUNT_SALT", "generator.hull/point_count", 0x55E558EB),
    ("warp_generate_hull", "POINT_SALT", "generator.hull/points", 0xC60040A4),
    ("warp_generate_hull", "_DISPLACE_SALT", "generator.hull/displacement", 0x8C818DE1),
    ("warp_generate_polar", "_CONTROL_SALT", "generator.polar/controls", 0x23DAB0D9),
    ("warp_generate_voronoi", "_SITE_SALT", "generator.voronoi/sites", 0x877BE6DF),
    ("warp_generate_voronoi", "_ANCHOR_SALT", "generator.voronoi/anchors", 0xC9F3730A),
    ("warp_generate_checkpoint", "_CHECKPOINT_SALT", "generator.checkpoint/rings", 0x7357F137),
    ("warp_generate_repulsive", "_OBSTACLE_SALT", "generator.repulsive/obstacles", 0x15710913),
    ("warp_generate_repulsive", "_GROW_SALT", "generator.repulsive/growth", 0xCD2AA706),
    ("warp_gate", "_ORDER_SALT", "order.random_pairs/pairs", 0x977A7D68),
    ("warp_zprofile", "_P_UNIFORM", "elevation.uniform/z", 0x7D59D787),
    ("warp_zprofile", "_P_WALK", "elevation.random_walk/z", 0x9717C554),
    ("warp_zprofile", "_P_NOISE", "elevation.noise/z", 0xD8AEA614),
)
SEEDS = np.array([0, 1, 7, 2147483647], dtype=np.int32)


def _rng(seeds=SEEDS):
    return PerEnvSeededRNG(num_envs=len(seeds), device="cpu",
                          seeds=wp.array(seeds, dtype=wp.int32, device="cpu"))


def _bytes(result):
    return {name: array.numpy().tobytes() for name, array in result_arrays(result).items()}


@pytest.mark.parametrize("module,field,namespace,want", STREAMS)
def test_builtin_stream_salts_are_fixed(module, field, namespace, want):
    assert hash32_host(_fnv1a_32(namespace.encode())) == want
    assert getattr(importlib.import_module("track_gen._src." + module), field, None) == want


def test_seed_zero_stream_states_are_distinct():
    states = [ops.rng.init_host(0, salt) for _, _, _, salt in STREAMS]
    assert len(set(states)) == len(STREAMS)
    assert all(ops.rng.init_at_host(0, salt, 0) != ops.rng.init_at_host(0, salt, 1)
               for _, _, _, salt in STREAMS)


@pytest.mark.parametrize("name,salts", [("bezier", (0xB739AFFB, 0xC2C74E45)),
                                         ("hull", (0x55E558EB, 0xC60040A4))])
def test_builtin_grid_streams_match_hashed_ops(name, salts):
    cfg = GateGenConfig(generator=name, num_envs=len(SEEDS), device="cpu")
    gen = GateGenerator(cfg, _rng())
    gen.generate()
    count = wp.empty(len(SEEDS), dtype=wp.int32, device="cpu")
    used = wp.empty(len(SEEDS) * cfg.max_num_points, dtype=wp.int32, device="cpu")
    points = wp.empty(len(SEEDS) * cfg.max_num_points, dtype=wp.vec2f, device="cpu")
    seed = wp.array(SEEDS, dtype=wp.int32, device="cpu")
    ctx = LaunchContext.for_call("cpu")
    cells, cell_size = ops.grid_cells(cfg.min_point_distance)
    ops.sample_counts(ctx, seed, salts[0], 0, cfg.min_num_points, cfg.max_num_points,
                      count)
    ops.sample_grid_points(ctx, seed, salts[1], 0, cfg.max_num_points, cells, cell_size,
                           cfg.scale, used, points)
    assert gen.program.tap("anchors.count").numpy().tobytes() == count.numpy().tobytes()
    assert gen.program.tap("anchors.xy").numpy().tobytes() == points.numpy().tobytes()


@pytest.mark.parametrize("kind,name", [("track", name) for name in
    ("bezier", "hull", "polar", "voronoi", "checkpoint", "repulsive")] +
    [("gates", name) for name in ("bezier", "hull", "polar", "voronoi", "checkpoint")])
def test_builtin_streams_ignore_stage_ids_and_preserve_masked_rows(kind, name):
    kw = dict(num_envs=len(SEEDS), device="cpu", generator=name,
              elevation=("noise", {"z_min": -0.3, "z_max": 0.3, "z_noise_amplitude": 0.1}))
    if name == "repulsive":
        kw.update(num_points=32, generator_params={"repulsive_stages": (16, 32)})
    pairs = kind == "gates" and name in ("bezier", "hull")
    spec = presets.track(**kw) if kind == "track" else presets.gates(
        **kw, order="random_pairs" if pairs else None)
    facade = TrackGenerator if kind == "track" else GateGenerator
    original = facade(spec, _rng())
    # Generator IDs are reserved by the planner; order/elevation IDs are user-selected.
    sid = "order" if pairs else "elevation"
    ref = spec.stages[spec.ids.index(sid)]
    renamed = spec.replace(sid, dataclasses.replace(ref, id="renamed"))
    twin = facade(renamed, _rng())
    assert _bytes(original.generate()) == _bytes(twin.generate())
    # A stream is keyed by the env's seed, independently of its row or batch size.
    channel = "raw.xy" if kind == "track" else "anchors.xy"
    raw = original.program.tap(channel).numpy().reshape(len(SEEDS), -1).copy()
    scalar_spec = dataclasses.replace(spec, batch=dataclasses.replace(spec.batch, num_envs=1))
    for env, seed in enumerate(SEEDS):
        scalar = facade(scalar_spec, _rng(np.array([seed], dtype=np.int32)))
        scalar.generate()
        assert scalar.program.tap(channel).numpy().tobytes() == raw[env].tobytes()
    before = {name: array.numpy().copy() for name, array in result_arrays(original.program.result).items()}
    mask = wp.array([1, 0, 1, 0], dtype=wp.int32, device="cpu")
    got = original.generate(advance=True, mask=mask)
    expected = twin.generate(advance=True, mask=wp.clone(mask))
    assert _bytes(got) == _bytes(expected)
    for field, array in result_arrays(got).items():
        rows = array.numpy().reshape(len(SEEDS), -1)
        previous = before[field].reshape(len(SEEDS), -1)
        assert rows[[1, 3]].tobytes() == previous[[1, 3]].tobytes(), field
