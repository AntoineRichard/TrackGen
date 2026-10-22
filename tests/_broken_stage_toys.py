"""Deliberately broken pipeline-stage plugins: each one breaks exactly one rule of the
stage contract, so ``check_stage`` must fail exactly the named check
(``tests/test_stage_plugins.py``).

- relax ``broken.writes_read``: writes its result into the version it READ as well
  (``stage.isolation``: the previous version is corrupted);
- elevation ``broken.stateful``: reads its own previous output (``stage.isolation``:
  re-running it on the same inputs changes ``cl.z``; ``statelessness`` too);
- validity ``broken.builtin_bit``: ORs a built-in bit (3, ``thickness``) besides its own
  (``validity.bits``);
- validity ``broken.drops_bits``: writes ``own`` instead of ``fail_in | own``, clearing
  the bits of earlier checks (``validity.bits``: on the kit's default batch no env fails
  upstream, so only its synthetic ``valid.fail`` input shows it);
- validity ``broken.writes_failures_only``: writes only the envs it fails, so the others
  keep whatever the buffer held (``validity.bits`` and ``stage.isolation``;
  ``statelessness`` too once a reseed turns a failing env into a passing one);
- elevation ``broken.skips_padding``: writes ``cl.z`` for ``i < count[e]`` only, so the
  padding keeps whatever an earlier generation left there (``stage.isolation``;
  ``statelessness`` too once a reseed shortens a loop);
- elevation ``broken.alloc``: allocates a buffer in ``enqueue`` (``hot_path.allocation``).
"""
from __future__ import annotations

import warp as wp

from track_gen import plugin

BROKEN = {"relax": "broken.writes_read", "stateful": "broken.stateful",
          "validity": "broken.builtin_bit", "drops_bits": "broken.drops_bits",
          "failures_only": "broken.writes_failures_only",
          "padding": "broken.skips_padding", "alloc": "broken.alloc"}


@wp.kernel
def _copy_both_k(src: wp.array(dtype=wp.vec2f), dst: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    v = src[t]
    if not wp.isnan(v[0]):
        v = v * 1.001
    dst[t] = v
    src[t] = v                          # the bug: writes the version it read


class WritesRead(plugin.Stage):
    kind = "relax"
    name = BROKEN["relax"]
    reads = {"xy": "cl.xy"}
    writes = {"xy": "cl.xy"}

    def build(self, b, p):
        x, out = b.read("xy"), b.write("xy", wp.vec2f, per="cl")
        n = b.E * int(b.layout.stride)
        return lambda ctx: ctx.launch(_copy_both_k, n, [x, out])


@wp.kernel
def _accumulate_k(count: wp.array(dtype=wp.int32), S: int, z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // S
    if t - e * S < count[e]:
        z[t] = z[t] + 0.01              # the bug: reads its own previous output
    else:
        z[t] = 0.0


class Stateful(plugin.Stage):
    kind = "elevation"
    name = BROKEN["stateful"]
    reads = {"count": "cl.count"}
    writes = {"z": "cl.z"}

    def build(self, b, p):
        count, z = b.read("count"), b.write("z", wp.float32, per="cl")
        S = int(b.layout.stride)
        return lambda ctx: ctx.launch(_accumulate_k, b.E * S, [count, S, z])


@wp.kernel
def _real_points_only_k(count: wp.array(dtype=wp.int32), S: int,
                        z: wp.array(dtype=wp.float32)):
    t = wp.tid()
    e = t // S
    if t - e * S < count[e]:             # the bug: padding slots are never written
        z[t] = 0.01 * float(count[e])


class SkipsPadding(plugin.Stage):
    kind = "elevation"
    name = BROKEN["padding"]
    reads = {"count": "cl.count"}
    writes = {"z": "cl.z"}

    def build(self, b, p):
        count, z = b.read("count"), b.write("z", wp.float32, per="cl")
        S = int(b.layout.stride)
        return lambda ctx: ctx.launch(_real_points_only_k, b.E * S, [count, S, z])


@wp.kernel
def _or_bits_k(bit: int, fail_in: wp.array(dtype=wp.int32),
               fail_out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    fail_out[e] = fail_in[e] | (1 << bit) | 8        # the bug: bit 3 is built-in


class BuiltinBit(plugin.Stage):
    kind = "validity"
    name = BROKEN["validity"]
    reads = {"fail": "valid.fail"}
    writes = {"fail": "valid.fail"}

    def build(self, b, p):
        fail_in, fail_out = b.read("fail"), b.write("fail", wp.int32)
        bit = b.fail_bit()
        return lambda ctx: ctx.launch(_or_bits_k, b.E, [bit, fail_in, fail_out])


@wp.kernel
def _drop_bits_k(count: wp.array(dtype=wp.int32), bit: int,
                 fail_in: wp.array(dtype=wp.int32), fail_out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    fail_out[e] = (count[e] % 2) << bit              # the bug: fail_in[e] is dropped


class DropsBits(plugin.Stage):
    kind = "validity"
    name = BROKEN["drops_bits"]
    reads = {"count": "cl.count", "fail": "valid.fail"}
    writes = {"fail": "valid.fail"}

    def build(self, b, p):
        count, fail_in = b.read("count"), b.read("fail")
        fail_out, bit = b.write("fail", wp.int32), b.fail_bit()
        return lambda ctx: ctx.launch(_drop_bits_k, b.E, [count, bit, fail_in, fail_out])


@wp.kernel
def _failures_only_k(count: wp.array(dtype=wp.int32), bit: int,
                     fail_in: wp.array(dtype=wp.int32), fail_out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if count[e] % 2 == 1:                            # the bug: passing envs not written
        fail_out[e] = fail_in[e] | (1 << bit)


class WritesFailuresOnly(plugin.Stage):
    kind = "validity"
    name = BROKEN["failures_only"]
    reads = {"count": "cl.count", "fail": "valid.fail"}
    writes = {"fail": "valid.fail"}

    def build(self, b, p):
        count, fail_in = b.read("count"), b.read("fail")
        fail_out, bit = b.write("fail", wp.int32), b.fail_bit()
        return lambda ctx: ctx.launch(_failures_only_k, b.E,
                                      [count, bit, fail_in, fail_out])


@wp.kernel
def _zero_k(z: wp.array(dtype=wp.float32)):
    z[wp.tid()] = 0.0


class Allocates(plugin.Stage):
    kind = "elevation"
    name = BROKEN["alloc"]
    writes = {"z": "cl.z"}

    def build(self, b, p):
        z = b.write("z", wp.float32, per="cl")
        n, dev = b.E * int(b.layout.stride), b.device

        def enqueue(ctx):
            tmp = wp.empty(n, dtype=wp.float32, device=dev)   # the bug
            ctx.launch(_zero_k, n, [tmp])
            ctx.launch(_zero_k, n, [z])
        return enqueue


def register_broken_stages(reg) -> None:
    for cls in (WritesRead, Stateful, SkipsPadding, BuiltinBit, DropsBits, WritesFailuresOnly,
                Allocates):
        reg.add(cls)
