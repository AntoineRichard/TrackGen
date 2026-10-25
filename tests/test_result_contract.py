"""Required result storage remains complete through construction and pipeline binding."""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
import warp as wp

from track_gen.results import Centerline, GateSequence, HasCenterline, Layout, Track, check_source

_REQUIRED = {
    Track: ("truncated", "generation", "fail_mask", "half_width", "layout"),
    GateSequence: ("generation", "fail_mask", "layout"),
}


def _fields(cls):
    result = cls.alloc(Layout(2, 8, "cpu"))
    return {f.name: getattr(result, f.name) for f in dataclasses.fields(cls)}


@pytest.mark.parametrize("cls,field", [(cls, field) for cls, names in _REQUIRED.items()
                                      for field in names])
def test_omitting_result_storage_is_an_error(cls, field):
    fields = _fields(cls)
    fields.pop(field)
    with pytest.raises(TypeError, match=field):
        cls(**fields)


@pytest.mark.parametrize("cls,field", [(cls, field) for cls in _REQUIRED
                                      for field in (*[n for n, _, _ in cls._SCHEMA], "layout")])
def test_none_result_storage_is_rejected_at_construction(cls, field):
    fields = _fields(cls)
    fields[field] = None
    with pytest.raises(ValueError, match=rf"{cls.__name__}\.{field}.*None"):
        cls(**fields)


@pytest.mark.parametrize("cls,field", [(cls, field) for cls in _REQUIRED
                                      for field in (*[n for n, _, _ in cls._SCHEMA], "layout")])
def test_validate_rejects_storage_cleared_after_construction(cls, field):
    result = cls.alloc(Layout(2, 8, "cpu"))
    setattr(result, field, None)
    with pytest.raises(ValueError, match=rf"{cls.__name__}\.{field}.*None"):
        result.validate()


def test_metadata_defaults_remain_independent_empty_mappings():
    fields = _fields(Track)
    fields.pop("fail_bits")
    fields.pop("attributes")
    a, b = Track(**fields), Track(**fields)
    a.fail_bits["custom"] = 16
    assert b.fail_bits == {} and a.attributes == b.attributes == {}
    assert a.attributes is not b.attributes


def test_centerline_and_capability_sources_still_infer_layout():
    fields = _fields(Track)
    centerline = Centerline(**{name: fields[name] for name in
                              ("center", "tangent", "arclen", "length", "count", "valid")})
    assert centerline.layout is None and centerline.generation is None
    assert centerline.validate() == Layout(2, 8, "cpu")
    source = types.SimpleNamespace(**{name: fields[name] for name in
                                      ("center", "arclen", "length", "count")})
    assert check_source(source, HasCenterline, consumer="test") == Layout(2, 8, "cpu")


@pytest.mark.parametrize("cls,field", [(cls, field) for cls, names in _REQUIRED.items()
                                      for field in names])
def test_out_rejects_required_storage_cleared_after_construction(cls, field):
    from track_gen.pipeline import build, presets

    result = cls.alloc(Layout(2, 384 if cls is Track else 32, "cpu"))
    setattr(result, field, None)
    preset = presets.track if cls is Track else presets.gates
    spec = preset(num_envs=2, device="cpu", on_invalid="keep")
    with pytest.raises(ValueError, match=rf"out\.{field}.*None"):
        build(spec, out=result)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("with_out", [False, True])
def test_pipeline_without_offset_keeps_nan_width_storage(with_out, device):
    from track_gen.pipeline import build, presets

    spec = presets.track(num_envs=2, device=device, on_invalid="keep").remove("offset")
    for stage_id in ("validity", "lift"):
        old = spec.stages[spec.ids.index(stage_id)]
        spec = spec.replace(stage_id, dataclasses.replace(
            old, ports={"outer": "cl.xy", "inner": "cl.xy"}))
    out = Track.alloc(Layout(2, 384, device)) if with_out else None
    if out is not None:
        out.half_width.fill_(7.0)
    program = build(spec, seeds=1, out=out)
    result = program.result
    assert isinstance(result.half_width, wp.array)
    assert result.half_width.shape == (2 * 384,)
    assert np.isnan(result.half_width.numpy()).all()
    assert result.validate() == program.layout
    copied = result.view(0, 1).clone()
    assert copied.half_width.shape == (384,)
    assert np.isnan(copied.half_width.numpy()).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_private_inflation_populates_required_diagnostics_and_width(device):
    from track_gen import TrackGenConfig
    from track_gen._src import warp_pipeline
    from track_gen.exec import LaunchContext

    n = 32
    theta = np.arange(n, dtype=np.float32) * (2 * np.pi / n)
    xy = np.stack((2 * np.cos(theta), 2 * np.sin(theta)), axis=1).astype(np.float32)
    points = wp.array(xy, dtype=wp.vec2f, device=device)
    cfg = TrackGenConfig(num_envs=1, num_points=n, N_max=n, device=device,
                         half_width=0.05, relax_enable=False, on_invalid="keep")
    ctx = LaunchContext.for_call(device)
    standalone = warp_pipeline._inflate_warp(points, cfg, ctx=ctx)
    assert standalone.validate() == Layout(1, n, device)
    np.testing.assert_array_equal(standalone.truncated.numpy(), [0])
    np.testing.assert_array_equal(standalone.generation.numpy(), [0])
    np.testing.assert_allclose(standalone.half_width.numpy(), 0.05)
    np.testing.assert_array_equal(standalone.valid.numpy(),
                                  standalone.fail_mask.numpy() == 0)

    out, scratch = warp_pipeline._inflate_warp_alloc(cfg)
    assert scratch.inflate.fail is out.fail_mask
    assert scratch.inflate.hw_out is out.half_width
    flag = wp.ones(1, dtype=wp.int32, device=device)
    warp_pipeline._inflate_warp(points, cfg, out=out, scratch=scratch,
                                truncated=flag, ctx=ctx)
    np.testing.assert_array_equal(out.truncated.numpy(), [1])
    assert out.valid.numpy()[0] == 0 and out.fail_mask.numpy()[0] != 0
    warp_pipeline._inflate_warp(points, cfg, out=out, scratch=scratch, ctx=ctx)
    np.testing.assert_array_equal(out.truncated.numpy(), [0])
    assert out.validate() == Layout(1, n, device)
