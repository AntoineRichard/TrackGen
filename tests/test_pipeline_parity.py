"""The Program is the PR9 composition, byte for byte, at every stage boundary (PR10a).

``TrackGenerator`` runs a :class:`track_gen.pipeline.Program` of the default stage list.
The PR9 composition (``warp_pipeline._run_pipeline`` -> ``_inflate_warp``) is kept
unchanged as the reference for the generators retaining that algorithm. Repulsive
intentionally changed its solver and final road checks; its geometry, bounded-loop
and capture regressions live in the Repulsive-specific tests. Each case here builds
a generator, generates (plainly, then with
a masked advance), runs ``_run_pipeline`` on fresh PR9 buffers from the same effective
seeds, and compares every channel version the Program wrote with the PR9 buffer it
reproduces (raw bytes: NaN padding compares too). The launch sequences (op, kernel,
dims, scalar arguments) of one generation are equal as well, so no launch moved.

Roadmap PR13c re-pin (deliberate, ``tests/_pr13c_delta.py``): the built-in generators are
Stages whose dynamic knobs are latched; the reference binds the generator through
``bind_generator(...).pipeline_spec`` (the same Stage, its latches run by
``adapter.latch``), so its generator half is Stage vs Stage. The elevation stages'
``z_*`` knobs are dynamic too, which the PR9 composition (``apply_z_profile*`` with
host scalars) does not latch: for the ``z_*`` cases the expected sequence is the
reference's with the elevation's latch records (uniform 2, random_walk 3, noise 3)
inserted after the generator's and its ``_z_*_k`` record renamed to the array-reading
twin (knob scalars dropped). Output bytes are unchanged at every boundary.

Roadmap PR13d re-pin (deliberate, ``tests/_pr13d_delta.py``): the bezier / hull
``min_num_points`` (int32) and every generator's ``size`` are dynamic too (both sides run
the Stage and its latches: the sequence test is unchanged); the default bezier count
moves 154 -> 156 (``test_default_bezier_launch_count``).
"""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from tests._plugin_toys import (TOY_CIRCLE, register_stage_toys,
                                toy_circle_alloc_scratch, toy_circle_generate)
from tests._pr13c_delta import ELEVATION_LATCHES, pr13c_sequence
from tests._pr13d_delta import LATCH_ALL
from tests.tools._golden_config import keep_config
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from tests._reference_generators import warp_pipeline
from track_gen._src.generator_binding import _StageGenerator, bind_generator
from track_gen.exec import LaunchContext
from track_gen.plugin import GeneratorSpec, registry
from track_gen.testing.trace import trace_launches

DEV = "cpu"
SEEDS = [0, 1, 7, 99991, 2147483647, 1000003]
TRACK_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "length",
                "valid", "count", "winding", "truncated")

_ZU = dict(z_profile="uniform", z_min=-0.2, z_max=0.3, z_valid_grade=0.5)
_ZW = dict(z_profile="random_walk", z_base=0.05, z_min=-0.2, z_max=0.3, z_max_step=0.4,
           z_valid_grade=0.8, z_control_points=7)
_ZN = dict(z_profile="noise", z_base=0.1, z_min=-0.3, z_max=0.4, z_noise_amplitude=0.2,
           z_noise_harmonics=4, z_valid_grade=0.6)

CASES = {
    "bezier": dict(generator="bezier"),
    "hull": dict(generator="hull"),
    "polar": dict(generator="polar"),
    "voronoi": dict(generator="voronoi"),
    "checkpoint": dict(generator="checkpoint"),
    "bezier_fallback": dict(generator="bezier", rad=0.49, _E=32),
    "voronoi_fallback": dict(generator="voronoi", voronoi_radial_variation=0.85,
                             voronoi_angular_jitter=0.6, voronoi_num_sites=64, _E=32),
    "hull_fallback": dict(generator="hull", hull_displacement=0.9, _E=32),
    "no_relax": dict(generator="bezier", relax_enable=False),
    "flat_z_base": dict(generator="bezier", z_base=0.25),
    "z_uniform": dict(generator="bezier", **_ZU),
    "z_random_walk": dict(generator="bezier", **_ZW),
    "z_noise": dict(generator="bezier", **_ZN),
    "z_noise_no_relax": dict(generator="bezier", relax_enable=False, **_ZN),
    "border_check": dict(generator="bezier", validity_border_check=True),
    "accel_none": dict(generator="bezier", relax_accel="none"),
    "no_sep_cache": dict(generator="bezier", relax_sep_cache_slots=0),
    "relax_band": dict(generator="bezier", relax_band=2),
    "truncation": dict(generator="bezier", N_max=64, spacing=0.06),
    "toy.circle": dict(generator="toy.circle"),
    "toy.wobble": dict(generator="toy.wobble"),
    "toy.wobble_input": dict(generator="toy.wobble_input"),
    "toy.lobed": dict(generator="toy.lobed"),
}


def _register_toys():
    registry.add(TOY_CIRCLE)
    register_stage_toys(registry)


def _make(case):
    kw = dict(CASES[case])
    E = kw.pop("_E", 6)
    seeds = (np.arange(E, dtype=np.int32) * 7919 + 3) if E > len(SEEDS) \
        else np.asarray(SEEDS[:E], np.int32)
    if kw["generator"].startswith("toy."):
        _register_toys()
    rng = PerEnvSeededRNG(seeds=wp.array(seeds, dtype=wp.int32, device=DEV), num_envs=E,
                          device=DEV)
    return TrackGenerator(keep_config(TrackGenConfig, device=DEV, num_envs=E, **kw), rng)


def _bytes(arr) -> bytes:
    return np.ascontiguousarray(arr.numpy()).tobytes()


class _Reference:
    """The PR9 composition on fresh buffers, from ``gen``'s current effective seeds,
    with the plan-view arc tables snapshot right after ``_arclength`` (PR9 overwrites
    them in place on an elevated profile)."""

    def __init__(self, gen, monkeypatch) -> None:
        cfg = gen._config
        self.spec = bind_generator("generator", cfg, registry=registry).pipeline_spec
        E = int(cfg.num_envs)
        self.seed = wp.empty(E, dtype=wp.int32, device=DEV)
        kw = {"seed": self.seed} if isinstance(self.spec, _StageGenerator) else {}
        self.track, self.scratch = warp_pipeline._inflate_warp_alloc(
            cfg, generator_spec=self.spec, **kw)
        self.cfg = cfg
        self.arc2d: dict = {}
        self._recording = False
        real = warp_pipeline._arclength

        def arclength(center, out_arclen, out_length, count, ctx=None):
            real(center, out_arclen, out_length, count, ctx=ctx)
            if self._recording:                     # the reference run only
                self.arc2d = {"cl.arclen2": _bytes(out_arclen),
                              "cl.length2": _bytes(out_length)}

        monkeypatch.setattr(warp_pipeline, "_arclength", arclength)

    def run(self, gen) -> None:
        wp.copy(self.seed, gen._seed_buf)
        self._recording = True
        try:
            warp_pipeline._run_pipeline(self.cfg, self.seed, out=self.track,
                                        scratch=self.scratch, generator_spec=self.spec,
                                        ctx=LaunchContext.for_call(DEV))
        finally:
            self._recording = False


def _assert_boundaries(gen, ref: _Reference, label: str) -> None:
    p, rs, rt = gen._program, ref.scratch, ref.track
    relax_on = bool(gen._config.relax_enable)
    pairs = [
        ("raw.xy", "generator", rs.gen_centerline),
        ("gen.valid", "generator", rs.gen_valid),
        ("cl.xy", "resample", rs.cs_center),
        ("cl.count", "resample", rs.count),
        ("cl.truncated", "resample", rt.truncated),
        ("cl.xy", "uniform", rs.inflate.ctr2),
        ("cl.half_width", "width", rs.inflate.w),
        ("track.count", "frames", rt.count),
        ("cl.tangent2", "frames", rs.inflate.tan2),
        ("cl.normal2", "frames", rs.inflate.nrm2),
        ("cl.kappa", "frames", rs.inflate.kappa),
        ("cl.winding", "frames", rt.winding),
        ("band.outer2", "offset", rs.inflate.out2),
        ("band.inner2", "offset", rs.inflate.inn2),
        ("cl.z", "elevation", rs.inflate.z),
        ("track.valid", "validity", rt.valid),
    ]
    if relax_on:
        pairs.append(("cl.xy", "relax", rs.relax.relaxed))
    bad = [f"{ch}@{after}" for ch, after, want in pairs
           if _bytes(p._channel(ch, after=after)) != _bytes(want)]
    for ch in ("cl.arclen2", "cl.length2"):                 # the plan-view tables
        if _bytes(p._channel(ch, after="frames")) != ref.arc2d[ch]:
            bad.append(f"{ch}@frames")
    scratch = p._scratch
    if relax_on:
        for name, want in (("band", rs.relax.band), ("L0", rs.relax.L0)):
            if _bytes(scratch["relax"][name]) != _bytes(want):
                bad.append(f"scratch:relax/{name}")
    for name, want in (("area_a", rs.inflate.area_a), ("area_b", rs.inflate.area_b)):
        if _bytes(scratch["offset"][name]) != _bytes(want):
            bad.append(f"scratch:offset/{name}")
    for f in TRACK_FIELDS:                                  # the lift and the exports
        if _bytes(getattr(gen._track, f)) != _bytes(getattr(rt, f)):
            bad.append(f"Track.{f}")
    assert not bad, f"{label}: the Program differs from the PR9 composition at {bad}"


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_stage_boundary_matches_the_pr9_composition(case, monkeypatch):
    gen = _make(case)
    ref = _Reference(gen, monkeypatch)
    gen.generate()
    ref.run(gen)
    _assert_boundaries(gen, ref, f"{case} generate()")
    E = int(gen._config.num_envs)
    mask = wp.array((np.arange(E) % 2 == 0).astype(np.int32), dtype=wp.int32, device=DEV)
    gen.generate(advance=True, mask=mask)
    ref.run(gen)
    _assert_boundaries(gen, ref, f"{case} generate(advance=True, mask=...)")


def _topology(trace) -> list:
    return [(r.op, r.kernel.replace("tests._reference_generators.", "track_gen._src."),
             r.dim, r.scalars) for r in trace]


def _pr13c_elevation_delta(case: str, rows: list, n_gen_latches: int, E: int) -> list:
    """The PR9 reference sequence under the PR13c rule for the elevation (the generator
    half needs nothing: both sides run the Stage): the elevation's latch records after
    the generator's, its ``_z_*_k`` renamed to the twin."""
    n = ELEVATION_LATCHES[CASES[case].get("z_profile", "flat")]
    if not n:
        return rows
    first = next(i for i, r in enumerate(rows) if r[1].endswith("_effective_seed_k"))
    return pr13c_sequence(rows, n, E, insert_after=first + n_gen_latches)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_program_emits_the_pr9_launch_sequence(case, monkeypatch):
    gen = _make(case)
    ref = _Reference(gen, monkeypatch)
    ctx = LaunchContext.for_call(DEV)
    with trace_launches() as program:
        gen._program.enqueue(ctx)
    adapter = ref.spec if isinstance(ref.spec, _StageGenerator) else None
    with trace_launches() as pr9:
        gen.seeds.effective(ref.seed, None, ctx=ctx)
        if adapter is not None and adapter._pairs:
            adapter.latch(ctx, None)                      # PR9's TrackGenerator._latch
        warp_pipeline._run_pipeline(ref.cfg, ref.seed, out=ref.track, scratch=ref.scratch,
                                    generator_spec=ref.spec, ctx=ctx)
        gen.seeds._write_generation(ref.track.generation, None, ctx)
    got = _topology(program)
    want = _pr13c_elevation_delta(case, _topology(pr9),
                                  len(adapter._pairs) if adapter is not None else 0,
                                  int(gen._config.num_envs))
    assert len(got) == len(want), f"{case}: {len(got)} launches vs PR9's {len(want)}"
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    assert first is None, f"{case}: launch {first} differs: {got[first]} vs {want[first]}"


def test_default_bezier_launch_count():
    """The default bezier pipeline: the effective-seed launch + the latches of its
    dynamic fields + 150 pipeline launches / copies + the generation copy (independent of
    E). Roadmap PR13c re-pin: 152 -> 154 (``rad`` / ``handle_clamp_frac``); roadmap
    PR13d re-pin: 154 -> 156 (``min_num_points`` int32 first, ``size`` last: the block
    ``[i32, f32, f32, f32]``; the corner count is the array-reading twin)."""
    gen = TrackGenerator(keep_config(TrackGenConfig, device=DEV, num_envs=2),
                         PerEnvSeededRNG(seeds=3, num_envs=2, device=DEV))
    with trace_launches() as t:
        gen.generate()
    assert len(t) == 156
    assert t[0].kernel.endswith("_effective_seed_k") and t[-1].op == "copy"
    assert [r.kernel for r in t[1:5]] == [LATCH_ALL["i32"]] + [LATCH_ALL["f32"]] * 3
    assert t[5].kernel.endswith("_sample_counts_dyn_h_k")
