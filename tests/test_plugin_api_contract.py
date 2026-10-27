"""The 0.3 plugin boundary accepts context-aware track specs and gate Stages."""
from __future__ import annotations

import importlib
import types

import pytest

from track_gen.pipeline import PipelineSpec
from track_gen.plugin import GeneratorSpec


def test_generator_spec_rejects_context_free_opt_in_flag():
    with pytest.raises(TypeError, match="accepts_ctx"):
        GeneratorSpec(name="test.old", alloc_scratch=lambda config: None,
                      generate=lambda *args: None, accepts_ctx=False)


def test_gate_registry_requires_a_stage(plugin_registry):
    old_gate = types.SimpleNamespace(name="test.old_gate", generate=lambda *a: None,
                                     alloc_scratch=lambda config: None,
                                     max_gates=lambda config: 8,
                                     supported_orderings=frozenset({"raw"}))
    with pytest.raises(TypeError, match="gate_generator.*Stage subclass"):
        plugin_registry.add(old_gate, kind="gate_generator")


@pytest.mark.parametrize("module", ["generator_registry", "gate_generator_registry"])
def test_legacy_registry_module_is_removed(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"track_gen._src.{module}")


def test_pipeline_v1_spec_is_rejected():
    with pytest.raises(ValueError, match="format"):
        PipelineSpec.from_dict({"format": "track_gen.pipeline/1", "batch": {},
                                "stages": []})


def test_runner_rejects_context_free_stream_adapter():
    from track_gen.exec import GraphRunner

    with pytest.raises(TypeError, match="legacy_stream"):
        GraphRunner("cpu", capturable=True, legacy_stream=True)
