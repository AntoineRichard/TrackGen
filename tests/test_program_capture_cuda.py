"""A standalone :class:`track_gen.pipeline.Program` on CUDA (roadmap PR10a).

- recorded into a caller's capture (Warp ``ScopedCapture``, a registered
  ``torch.cuda.graph`` and a plain one): poisoned replays equal an eager twin, before
  and after a reseed (:func:`track_gen.testing.capture.assert_caller_capturable`);
- its own self-graph replays the eager batch, also into poisoned outputs and for the
  masked-advance key;
- with ``on_invalid="keep"``, the Program is the PR9 composition byte for byte on
  CUDA (every Track field, the same launch sequence), as on CPU
  (``test_pipeline_parity``).

``TrackGenerator`` itself runs a Program: the existing caller-capture, stream-capture
and in-place-regeneration CUDA tests cover it unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import warp_pipeline
from tests._pr13c_delta import ELEVATION_LATCHES, pr13c_sequence
from track_gen._src.generator_binding import _StageGenerator, bind_generator
from track_gen.exec import LaunchContext
from track_gen.pipeline import build, legacy
from track_gen.plugin import registry
from track_gen.testing import capture
from track_gen.testing.capture import assert_caller_capturable, poison_all
from track_gen.testing.trace import trace_launches

pytestmark = pytest.mark.cuda

DEV = "cuda"
E = 8
SPECS = {
    "default": {},
    "noise_no_relax": dict(relax_enable=False, z_profile="noise", z_max=0.3,
                           z_noise_amplitude=0.2),
    "random_walk": dict(z_profile="random_walk", z_max=0.3, z_max_step=0.5),
}


def _spec(name, *, on_invalid=None):
    options = dict(SPECS[name])
    if on_invalid is not None:
        options["on_invalid"] = on_invalid
    return legacy.to_pipeline_spec(TrackGenConfig(device=DEV, num_envs=E, **options))


def _twin(spec):
    def twin(seeds):
        p = build(spec, seeds=wp.array(np.asarray(seeds, np.int32), dtype=wp.int32,
                                       device=DEV))
        wp.synchronize_device(DEV)
        return capture._snapshot(p.result)
    return twin


@pytest.mark.parametrize("name", sorted(SPECS))
def test_a_standalone_program_is_caller_capturable(name):
    # Without relaxation both reseed batches fail validity; retain distinct raw
    # geometry so the helper's replay-after-reseed check cannot be vacuous.
    spec = _spec(name, on_invalid="keep") if name == "noise_no_relax" else _spec(name)
    program = build(spec, seeds=3)
    status = assert_caller_capturable(program, record=lambda s: program.run(stream=s),
                                      twin=_twin(spec))
    assert status["warp"] == "ok", status
    assert program.graph is None                     # recordings never self-capture


@pytest.mark.parametrize("name", sorted(SPECS))
def test_the_program_self_graph_replays_the_eager_batch(name):
    spec = _spec(name)
    program = build(spec, seeds=3)
    want = capture._snapshot(program.result)         # prepare(): one eager generation
    program.run(sync=True)
    assert program.graph is not None
    poison_all(program.result)
    program.run(sync=True)
    capture._assert_same(capture._snapshot(program.result), want, "self-graph replay")
    mask = program.inputs["regen_mask"]
    wp.copy(mask, wp.array(np.arange(E, dtype=np.int32) % 2, dtype=wp.int32, device=DEV))
    program.run(advance=True, mask=mask, sync=True)
    assert set(program.runner.graph_keys) == {None, (True, int(mask.ptr))}
    gen = program.result.generation.numpy()
    np.testing.assert_array_equal(gen, np.arange(E) % 2)
    twin = build(spec, seeds=3)
    twin.run(advance=True, mask=wp.clone(mask), policy="eager", sync=True)
    capture._assert_same(capture._snapshot(program.result), capture._snapshot(twin.result),
                         "masked advance")


@pytest.mark.parametrize("name", sorted(SPECS))
def test_the_program_is_the_pr9_composition_on_cuda(name):
    # The raw composition has no invalid-output commit.
    gen = TrackGenerator(TrackGenConfig(device=DEV, num_envs=E, on_invalid="keep",
                                        **SPECS[name]),
                         PerEnvSeededRNG(seeds=3, num_envs=E, device=DEV))
    cfg = gen._config
    spec = bind_generator("generator", cfg, registry=registry).pipeline_spec
    seed = wp.empty(E, dtype=wp.int32, device=DEV)
    # A Stage generator binds its seed buffer at build (no seed copy in the trace).
    kw = {"seed": seed} if isinstance(spec, _StageGenerator) else {}
    ref, scratch = warp_pipeline._inflate_warp_alloc(cfg, generator_spec=spec, **kw)
    ctx = LaunchContext.for_call(DEV)
    with trace_launches() as program:
        gen._program.enqueue(ctx)
    # Roadmap PR13c: the built-in generator is a Stage (both sides run it; its dynamic
    # knobs' latches by adapter.latch); the elevation's dynamic z knobs are latched by
    # the Program only and its kernel is the array-reading twin (tests/_pr13c_delta.py).
    adapter = spec if isinstance(spec, _StageGenerator) else None
    with trace_launches() as pr9:
        gen.seeds.effective(seed, None, ctx=ctx)
        if adapter is not None and adapter._pairs:
            adapter.latch(ctx, None)
        warp_pipeline._run_pipeline(cfg, seed, out=ref, scratch=scratch,
                                    generator_spec=spec, ctx=ctx)
        gen.seeds._write_generation(ref.generation, None, ctx)
    wp.synchronize_device(DEV)
    topo = [(r.op, r.kernel, r.dim, r.scalars) for r in program]
    want = [(r.op, r.kernel, r.dim, r.scalars) for r in pr9]
    n_z = ELEVATION_LATCHES[str(cfg.z_profile)]
    if n_z:
        n_gen = len(adapter._pairs) if adapter is not None else 0
        want = pr13c_sequence(want, n_z, E, insert_after=n_gen)
    assert topo == want
    for f in ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
              "valid", "count", "winding", "truncated", "generation"):
        assert getattr(gen._track, f).numpy().tobytes() == \
            getattr(ref, f).numpy().tobytes(), f
