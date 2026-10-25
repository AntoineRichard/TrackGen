"""Immutable construction configs and the default invalid-output policy."""
from __future__ import annotations

import dataclasses

import pytest

from track_gen import GateGenConfig, TrackGenConfig


@pytest.mark.parametrize("config_type", [TrackGenConfig, GateGenConfig])
def test_config_fields_are_frozen(config_type):
    config = config_type()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.num_envs = 2


@pytest.mark.parametrize("config_type", [TrackGenConfig, GateGenConfig])
def test_config_defaults_to_poison_and_allows_explicit_keep(config_type):
    assert config_type().on_invalid == "poison"
    assert config_type(on_invalid="keep").on_invalid == "keep"


@pytest.mark.parametrize("config_type", [TrackGenConfig, GateGenConfig])
@pytest.mark.parametrize("policy", ["drop", "", None, False])
def test_config_rejects_an_unknown_invalid_output_policy(config_type, policy):
    with pytest.raises(ValueError, match="on_invalid must be 'keep' or 'poison'"):
        config_type(on_invalid=policy)


def test_frozen_config_subclasses_support_replace_and_spacing():
    @dataclasses.dataclass(frozen=True)
    class PluginConfig(TrackGenConfig):
        plugin_radius: float = 0.5

    config = PluginConfig(half_width=0.2)
    assert config.spacing == pytest.approx(0.12)
    changed = dataclasses.replace(config, half_width=0.3, plugin_radius=0.7)
    assert type(changed) is PluginConfig and changed.plugin_radius == 0.7
    assert changed.spacing == config.spacing
    assert dataclasses.replace(changed, spacing=None).spacing == pytest.approx(0.18)
    assert config.plugin_radius == 0.5 and config.half_width == 0.2


@pytest.mark.parametrize("kind", ["track", "gates"])
@pytest.mark.parametrize("policy", ["poison", "keep"])
def test_config_policy_survives_spec_and_serialization_round_trips(kind, policy):
    from track_gen.pipeline import PipelineSpec, legacy

    if kind == "track":
        config = TrackGenConfig(device="cpu", num_envs=2, on_invalid=policy)
        spec = legacy.to_pipeline_spec(config)
        restored = legacy.generator_config(spec)
    else:
        config = GateGenConfig(device="cpu", num_envs=2, on_invalid=policy)
        spec = legacy.to_gate_pipeline_spec(config)
        restored = legacy.gate_generator_config(spec)
    assert spec.batch.on_invalid == policy and restored.on_invalid == policy
    assert PipelineSpec.from_dict(spec.to_dict()).batch.on_invalid == policy


@pytest.mark.parametrize("kind", ["track", "gates"])
def test_default_policy_poisons_invalid_rows_and_explicit_keep_preserves_geometry(kind):
    import numpy as np

    from track_gen import GateGenerator, TrackGenerator

    if kind == "track":
        config_type, facade = TrackGenConfig, TrackGenerator
        options = {"N_max": 8}
        geometry = "center"
    else:
        config_type, facade = GateGenConfig, GateGenerator
        options = {"gate_radius": 100.0, "gate_solve_iters": 0}
        geometry = "position"
    config = config_type(device="cpu", num_envs=2, **options)
    poisoned = facade(config, seeds=5).generate()
    kept = facade(dataclasses.replace(config, on_invalid="keep"), seeds=5).generate()
    assert poisoned.valid.numpy().tolist() == kept.valid.numpy().tolist() == [0, 0]
    assert poisoned.count.numpy().tolist() == [0, 0]
    assert (kept.count.numpy() > 0).all()
    assert np.isnan(getattr(poisoned, geometry).numpy()).all()
    assert np.isfinite(getattr(kept, geometry).numpy()).any()
    np.testing.assert_array_equal(poisoned.fail_mask.numpy(), kept.fail_mask.numpy())


def test_missing_serialized_policy_uses_the_poison_default():
    from track_gen.pipeline import BatchSpec, PipelineSpec, presets

    assert BatchSpec(2, "cpu", 384).on_invalid == "poison"
    for preset in (presets.track, presets.gates):
        spec = preset(num_envs=2, device="cpu")
        assert spec.batch.on_invalid == "poison"
        data = spec.to_dict()
        del data["batch"]["on_invalid"]
        assert PipelineSpec.from_dict(data).batch.on_invalid == "poison"
