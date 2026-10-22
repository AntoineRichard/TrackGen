"""The commit: the Program's epilogue that decides what each env's ``Track`` holds.

The stages compute EVERY env on every generation (fixed shape, capturable). The final
version of each exported channel is the ``Track``'s own array (no copy: an unmasked
generation of the default pipeline emits exactly the launches it emitted before the
commit existed). The commit then fixes, per env, what the generation leaves in the
``Track``:

- a masked regeneration (``generate(advance=True, mask=m)``): before the stages,
  :func:`_hold_k` copies the ``Track`` rows of every env with ``m[e] == 0`` into a held
  copy; after them, :func:`_commit_k` copies them back, bit for bit. An unmasked env
  ends the generation with exactly the bytes it had, whatever the stages recomputed for
  it (no reliance on bitwise determinism); an all-zero mask is an exact no-op. This is a
  hold / restore, not a sole-writer commit: WITHIN the generation the stages overwrite
  the unmasked rows with their recomputation, and the commit then restores them, all on
  the generation's stream. Work ordered after the generation (later on that stream, or
  after an event recorded on it) never observes the recomputed rows. A reader on ANOTHER
  stream with no dependency on the generation can: for a while an unmasked env may show
  its recomputed track (after a reseed, a different one; under
  ``on_invalid="poison"``, a poisoned env with ``count > 0`` and finite geometry);
- ``valid``: ``fail_mask == 0`` when the pipeline's last ``valid.fail`` writer is not
  ``validity.default`` (a plugin check ran after it), else ``validity.default``'s own
  flag (equal by construction);
- ``on_invalid="poison"``: a committed env with ``fail_mask != 0`` leaves with
  ``count = 0``, NaN geometry (``outer`` / ``center`` / ``inner`` / ``tangent`` /
  ``normal`` / ``arclen``), ``length = winding = 0`` and ``valid = 0`` (its ``fail_mask``
  and ``truncated`` are kept as diagnostics), so every utility's ``count < 3`` path
  fires.

Cost: no launch for an unmasked generation of a pipeline whose ``valid`` is
``validity.default``'s and ``on_invalid="keep"`` (the default); one launch for an
unmasked generation otherwise (:func:`_derive_valid_k`, one thread per env, when only
``valid`` is derived; :func:`_commit_k` with ``on_invalid="poison"``); two (:func:`_hold_k` +
:func:`_commit_k`, each over every ``(env, point)``, whatever the number of masked envs)
for a masked one. Memory: one held copy of the ``Track`` (64 B per point + 24 B per env,
about 100 MB at ``E = 4096``, ``N_max = 384``), allocated once at build for every
Program, whether or not it is ever run with a mask.
"""
from __future__ import annotations

import warp as wp

__all__ = ["ENV_FIELDS", "POINT_FIELDS"]

#: The per-point ``Track`` fields the commit holds / restores / poisons.
POINT_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen")
#: The per-env ``Track`` fields the commit holds / restores (``fail_mask`` is the final
#: ``valid.fail``).
ENV_FIELDS = ("length", "winding", "count", "valid", "fail_mask", "truncated")
_DTYPES = {"outer": wp.vec3f, "center": wp.vec3f, "inner": wp.vec3f, "tangent": wp.vec3f,
           "normal": wp.vec3f, "arclen": wp.float32, "length": wp.float32,
           "winding": wp.float32, "count": wp.int32, "valid": wp.int32,
           "fail_mask": wp.int32, "truncated": wp.int32}


@wp.kernel
def _hold_k(mask: wp.array(dtype=wp.int32), S: int,
            outer: wp.array(dtype=wp.vec3f), center: wp.array(dtype=wp.vec3f),
            inner: wp.array(dtype=wp.vec3f), tangent: wp.array(dtype=wp.vec3f),
            normal: wp.array(dtype=wp.vec3f), arclen: wp.array(dtype=wp.float32),
            length: wp.array(dtype=wp.float32), winding: wp.array(dtype=wp.float32),
            count: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
            fail_mask: wp.array(dtype=wp.int32), truncated: wp.array(dtype=wp.int32),
            h_outer: wp.array(dtype=wp.vec3f), h_center: wp.array(dtype=wp.vec3f),
            h_inner: wp.array(dtype=wp.vec3f), h_tangent: wp.array(dtype=wp.vec3f),
            h_normal: wp.array(dtype=wp.vec3f), h_arclen: wp.array(dtype=wp.float32),
            h_length: wp.array(dtype=wp.float32), h_winding: wp.array(dtype=wp.float32),
            h_count: wp.array(dtype=wp.int32), h_valid: wp.array(dtype=wp.int32),
            h_fail_mask: wp.array(dtype=wp.int32), h_truncated: wp.array(dtype=wp.int32)):
    # One thread per (env, point): hold the Track rows of every UNMASKED env.
    t = wp.tid()
    e = t // S
    i = t - e * S
    if mask[e] != 0:
        return
    h_outer[t] = outer[t]
    h_center[t] = center[t]
    h_inner[t] = inner[t]
    h_tangent[t] = tangent[t]
    h_normal[t] = normal[t]
    h_arclen[t] = arclen[t]
    if i == 0:
        h_length[e] = length[e]
        h_winding[e] = winding[e]
        h_count[e] = count[e]
        h_valid[e] = valid[e]
        h_fail_mask[e] = fail_mask[e]
        h_truncated[e] = truncated[e]


@wp.kernel
def _commit_k(mask: wp.array(dtype=wp.int32), use_mask: int, poison: int, S: int,
              h_outer: wp.array(dtype=wp.vec3f), h_center: wp.array(dtype=wp.vec3f),
              h_inner: wp.array(dtype=wp.vec3f), h_tangent: wp.array(dtype=wp.vec3f),
              h_normal: wp.array(dtype=wp.vec3f), h_arclen: wp.array(dtype=wp.float32),
              h_length: wp.array(dtype=wp.float32), h_winding: wp.array(dtype=wp.float32),
              h_count: wp.array(dtype=wp.int32), h_valid: wp.array(dtype=wp.int32),
              h_fail_mask: wp.array(dtype=wp.int32), h_truncated: wp.array(dtype=wp.int32),
              outer: wp.array(dtype=wp.vec3f), center: wp.array(dtype=wp.vec3f),
              inner: wp.array(dtype=wp.vec3f), tangent: wp.array(dtype=wp.vec3f),
              normal: wp.array(dtype=wp.vec3f), arclen: wp.array(dtype=wp.float32),
              length: wp.array(dtype=wp.float32), winding: wp.array(dtype=wp.float32),
              count: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
              fail_mask: wp.array(dtype=wp.int32), truncated: wp.array(dtype=wp.int32)):
    # One thread per (env, point). fail_mask is the final valid.fail version.
    t = wp.tid()
    e = t // S
    i = t - e * S
    if use_mask != 0:
        if mask[e] == 0:
            # An unmasked env: restore every byte held before the stages ran.
            outer[t] = h_outer[t]
            center[t] = h_center[t]
            inner[t] = h_inner[t]
            tangent[t] = h_tangent[t]
            normal[t] = h_normal[t]
            arclen[t] = h_arclen[t]
            if i == 0:
                length[e] = h_length[e]
                winding[e] = h_winding[e]
                count[e] = h_count[e]
                valid[e] = h_valid[e]
                fail_mask[e] = h_fail_mask[e]
                truncated[e] = h_truncated[e]
            return
    f = fail_mask[e]
    bad = int(0)
    if poison != 0:
        if f != 0:
            bad = int(1)
    if bad != 0:
        nan3 = wp.vec3f(wp.nan, wp.nan, wp.nan)
        outer[t] = nan3
        center[t] = nan3
        inner[t] = nan3
        tangent[t] = nan3
        normal[t] = nan3
        arclen[t] = wp.nan
    if i == 0:
        if f == 0:
            valid[e] = 1
        else:
            valid[e] = 0
        if bad != 0:
            count[e] = 0
            length[e] = 0.0
            winding[e] = 0.0


@wp.kernel
def _derive_valid_k(fail_mask: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32)):
    # One thread per env: an unmasked generation that only derives valid (no poison).
    e = wp.tid()
    if fail_mask[e] == 0:
        valid[e] = 1
    else:
        valid[e] = 0


class _Commit:
    """The bound epilogue of one Program (built once by the planner).

    ``targets``: field -> the ``Track`` array the generation leaves (``fail_mask``: the
    final ``valid.fail`` version); ``always``: launch :func:`_commit_k` on unmasked
    generations too (``valid`` derived from ``valid.fail``, or ``on_invalid="poison"``).
    """

    def __init__(self, *, targets, E: int, S: int, device, dummy_mask, poison: bool,
                 derive_valid: bool) -> None:
        self.targets = dict(targets)
        self.E, self.S = int(E), int(S)
        self.poison = bool(poison)
        self.derive_valid = bool(derive_valid)
        self.always = self.poison or self.derive_valid
        self._dummy_mask = dummy_mask
        n = self.E * self.S
        self.hold = {f: wp.empty(n if f in POINT_FIELDS else self.E, dtype=_DTYPES[f],
                                 device=device)
                     for f in (*POINT_FIELDS, *ENV_FIELDS)}
        fields = (*POINT_FIELDS, *ENV_FIELDS)
        self._track = [self.targets[f] for f in fields]
        self._held = [self.hold[f] for f in fields]

    def enqueue_hold(self, ctx, mask) -> None:
        """Before the stages of a masked generation: hold the unmasked envs' rows."""
        ctx.launch(_hold_k, self.E * self.S, [mask, self.S, *self._track, *self._held])

    def enqueue_commit(self, ctx, mask) -> None:
        """After the stages: restore the unmasked envs (masked generation), derive
        ``valid`` and poison the invalid envs (when configured). Nothing is launched on
        an unmasked generation that needs neither."""
        if mask is None and not self.always:
            return
        if mask is None and not self.poison:
            # Only valid to derive: one thread per env.
            ctx.launch(_derive_valid_k, self.E,
                       [self.targets["fail_mask"], self.targets["valid"]])
            return
        use_mask = 0 if mask is None else 1
        ctx.launch(_commit_k, self.E * self.S,
                   [self._dummy_mask if mask is None else mask, use_mask,
                    1 if self.poison else 0, self.S, *self._held, *self._track])
