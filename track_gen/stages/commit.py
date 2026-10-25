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
  ``normal`` / ``arclen`` / ``half_width``), ``length = winding = 0`` and ``valid = 0``
  (its ``fail_mask`` and ``truncated`` are kept as diagnostics), so every utility's
  ``count < 3`` path fires; every exported attribute row gets its padding (NaN /
  ``-1``).

Cost: no launch for an unmasked generation of a pipeline whose ``valid`` is
``validity.default``'s and ``on_invalid="keep"`` (the default); one launch for an
unmasked generation otherwise (:func:`_derive_valid_k`, one thread per env, when only
``valid`` is derived; :func:`_commit_k` with ``on_invalid="poison"``); two (:func:`_hold_k` +
:func:`_commit_k`, each over every ``(env, point)``, whatever the number of masked envs)
for a masked one. Memory: one held copy of the ``Track`` (68 B per point + 24 B per env,
about 107 MB at ``E = 4096``, ``N_max = 384``), allocated once at build for every
Program, whether or not it is ever run with a mask.

Exported attributes (``Track.attributes`` / ``GateSequence.attributes``, roadmap PR13a)
are committed by one :class:`_AttrCommit` each, launched right after the result's own
hold / commit and only when attributes exist: a masked generation adds two launches
per attribute (hold + restore), an unmasked one under ``on_invalid="poison"`` one per
attribute, an unmasked ``keep`` (derived ``valid`` included) none; each holds a copy of
its attribute.

A gate Program (roadmap PR11) commits its ``GateSequence`` the same way
(:class:`_GateCommit`: :func:`_gate_hold_k` / :func:`_gate_commit_k` over
:data:`GATE_POINT_FIELDS` and :data:`GATE_ENV_FIELDS`); a poisoned gate env leaves with
``count = 0``, every per-gate field NaN and ``valid = 0`` (``fail_mask`` kept). Its held
copy is 80 B per gate slot + 12 B per env.
"""
from __future__ import annotations

import warp as wp

__all__ = ["ENV_FIELDS", "GATE_ENV_FIELDS", "GATE_POINT_FIELDS", "POINT_FIELDS"]

#: The per-point ``Track`` fields the commit holds / restores / poisons.
POINT_FIELDS = ("outer", "center", "inner", "tangent", "normal", "arclen", "half_width")
#: The per-env ``Track`` fields the commit holds / restores (``fail_mask`` is the final
#: ``valid.fail``).
ENV_FIELDS = ("length", "winding", "count", "valid", "fail_mask", "truncated")
_DTYPES = {"outer": wp.vec3f, "center": wp.vec3f, "inner": wp.vec3f, "tangent": wp.vec3f,
           "normal": wp.vec3f, "arclen": wp.float32, "half_width": wp.float32,
           "length": wp.float32,
           "winding": wp.float32, "count": wp.int32, "valid": wp.int32,
           "fail_mask": wp.int32, "truncated": wp.int32}
#: The per-gate ``GateSequence`` fields the gate commit holds / restores / poisons.
GATE_POINT_FIELDS = ("position", "tangent", "forward", "left", "right", "orientation",
                     "half_size")
#: The per-env ``GateSequence`` fields the gate commit holds / restores.
GATE_ENV_FIELDS = ("count", "valid", "fail_mask")
_GATE_DTYPES = {"position": wp.vec3f, "tangent": wp.vec3f, "forward": wp.vec3f,
                "left": wp.vec3f, "right": wp.vec3f, "orientation": wp.quatf,
                "half_size": wp.float32, "count": wp.int32, "valid": wp.int32,
                "fail_mask": wp.int32}


@wp.kernel
def _hold_k(mask: wp.array(dtype=wp.int32), S: int,
            outer: wp.array(dtype=wp.vec3f), center: wp.array(dtype=wp.vec3f),
            inner: wp.array(dtype=wp.vec3f), tangent: wp.array(dtype=wp.vec3f),
            normal: wp.array(dtype=wp.vec3f), arclen: wp.array(dtype=wp.float32),
            half_width: wp.array(dtype=wp.float32),
            length: wp.array(dtype=wp.float32), winding: wp.array(dtype=wp.float32),
            count: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
            fail_mask: wp.array(dtype=wp.int32), truncated: wp.array(dtype=wp.int32),
            h_outer: wp.array(dtype=wp.vec3f), h_center: wp.array(dtype=wp.vec3f),
            h_inner: wp.array(dtype=wp.vec3f), h_tangent: wp.array(dtype=wp.vec3f),
            h_normal: wp.array(dtype=wp.vec3f), h_arclen: wp.array(dtype=wp.float32),
            h_half_width: wp.array(dtype=wp.float32),
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
    h_half_width[t] = half_width[t]
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
              h_half_width: wp.array(dtype=wp.float32),
              h_length: wp.array(dtype=wp.float32), h_winding: wp.array(dtype=wp.float32),
              h_count: wp.array(dtype=wp.int32), h_valid: wp.array(dtype=wp.int32),
              h_fail_mask: wp.array(dtype=wp.int32), h_truncated: wp.array(dtype=wp.int32),
              outer: wp.array(dtype=wp.vec3f), center: wp.array(dtype=wp.vec3f),
              inner: wp.array(dtype=wp.vec3f), tangent: wp.array(dtype=wp.vec3f),
              normal: wp.array(dtype=wp.vec3f), arclen: wp.array(dtype=wp.float32),
              half_width: wp.array(dtype=wp.float32),
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
            half_width[t] = h_half_width[t]
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
        half_width[t] = wp.nan
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


@wp.kernel
def _gate_hold_k(mask: wp.array(dtype=wp.int32), S: int,
                 position: wp.array(dtype=wp.vec3f), tangent: wp.array(dtype=wp.vec3f),
                 forward: wp.array(dtype=wp.vec3f), left: wp.array(dtype=wp.vec3f),
                 right: wp.array(dtype=wp.vec3f), orientation: wp.array(dtype=wp.quatf),
                 half_size: wp.array(dtype=wp.float32), count: wp.array(dtype=wp.int32),
                 valid: wp.array(dtype=wp.int32), fail_mask: wp.array(dtype=wp.int32),
                 h_position: wp.array(dtype=wp.vec3f), h_tangent: wp.array(dtype=wp.vec3f),
                 h_forward: wp.array(dtype=wp.vec3f), h_left: wp.array(dtype=wp.vec3f),
                 h_right: wp.array(dtype=wp.vec3f), h_orientation: wp.array(dtype=wp.quatf),
                 h_half_size: wp.array(dtype=wp.float32), h_count: wp.array(dtype=wp.int32),
                 h_valid: wp.array(dtype=wp.int32), h_fail_mask: wp.array(dtype=wp.int32)):
    # One thread per (env, gate slot): hold the GateSequence rows of every UNMASKED env.
    t = wp.tid()
    e = t // S
    i = t - e * S
    if mask[e] != 0:
        return
    h_position[t] = position[t]
    h_tangent[t] = tangent[t]
    h_forward[t] = forward[t]
    h_left[t] = left[t]
    h_right[t] = right[t]
    h_orientation[t] = orientation[t]
    h_half_size[t] = half_size[t]
    if i == 0:
        h_count[e] = count[e]
        h_valid[e] = valid[e]
        h_fail_mask[e] = fail_mask[e]


@wp.kernel
def _gate_commit_k(mask: wp.array(dtype=wp.int32), use_mask: int, poison: int, S: int,
                   h_position: wp.array(dtype=wp.vec3f),
                   h_tangent: wp.array(dtype=wp.vec3f),
                   h_forward: wp.array(dtype=wp.vec3f), h_left: wp.array(dtype=wp.vec3f),
                   h_right: wp.array(dtype=wp.vec3f),
                   h_orientation: wp.array(dtype=wp.quatf),
                   h_half_size: wp.array(dtype=wp.float32),
                   h_count: wp.array(dtype=wp.int32), h_valid: wp.array(dtype=wp.int32),
                   h_fail_mask: wp.array(dtype=wp.int32),
                   position: wp.array(dtype=wp.vec3f), tangent: wp.array(dtype=wp.vec3f),
                   forward: wp.array(dtype=wp.vec3f), left: wp.array(dtype=wp.vec3f),
                   right: wp.array(dtype=wp.vec3f), orientation: wp.array(dtype=wp.quatf),
                   half_size: wp.array(dtype=wp.float32), count: wp.array(dtype=wp.int32),
                   valid: wp.array(dtype=wp.int32), fail_mask: wp.array(dtype=wp.int32)):
    # One thread per (env, gate slot). fail_mask is the final valid.fail version.
    t = wp.tid()
    e = t // S
    i = t - e * S
    if use_mask != 0:
        if mask[e] == 0:
            position[t] = h_position[t]
            tangent[t] = h_tangent[t]
            forward[t] = h_forward[t]
            left[t] = h_left[t]
            right[t] = h_right[t]
            orientation[t] = h_orientation[t]
            half_size[t] = h_half_size[t]
            if i == 0:
                count[e] = h_count[e]
                valid[e] = h_valid[e]
                fail_mask[e] = h_fail_mask[e]
            return
    f = fail_mask[e]
    bad = int(0)
    if poison != 0:
        if f != 0:
            bad = int(1)
    if bad != 0:
        nan3 = wp.vec3f(wp.nan, wp.nan, wp.nan)
        position[t] = nan3
        tangent[t] = nan3
        forward[t] = nan3
        left[t] = nan3
        right[t] = nan3
        orientation[t] = wp.quatf(wp.nan, wp.nan, wp.nan, wp.nan)
        half_size[t] = wp.nan
    if i == 0:
        if f == 0:
            valid[e] = 1
        else:
            valid[e] = 0
        if bad != 0:
            count[e] = 0


# ---------------------------------------------------------------------------------------
# Exported attributes (roadmap PR13a): one hold / commit per attribute, for float32 /
# int32 / vec2f / vec3f (padding NaN / -1).
# ---------------------------------------------------------------------------------------

@wp.func
def _attr_pad(x: wp.float32) -> wp.float32:
    return wp.nan


@wp.func
def _attr_pad(x: wp.int32) -> wp.int32:  # noqa: F811 - an overload
    return -1


@wp.func
def _attr_pad(x: wp.vec2f) -> wp.vec2f:  # noqa: F811
    return wp.vec2f(wp.nan, wp.nan)


@wp.func
def _attr_pad(x: wp.vec3f) -> wp.vec3f:  # noqa: F811
    return wp.vec3f(wp.nan, wp.nan, wp.nan)


# One CONCRETE hold / commit kernel per dtype (no generic ``Any`` + ``wp.overload``
# kernels): Warp 1.18's CPU graph recorder keys a kernel by ``kernel.key``, shared by every
# overload, so a CPU capture launching several dtypes back to back could replay another
# overload's code.
@wp.kernel
def _attr_hold_f32_k(mask: wp.array(dtype=wp.int32), S: int,
                     arr: wp.array(dtype=wp.float32), held: wp.array(dtype=wp.float32)):
    # One thread per (env, slot): hold the attribute rows of every UNMASKED env.
    t = wp.tid()
    e = t // S
    if mask[e] != 0:
        return
    held[t] = arr[t]


@wp.kernel
def _attr_hold_i32_k(mask: wp.array(dtype=wp.int32), S: int,
                     arr: wp.array(dtype=wp.int32), held: wp.array(dtype=wp.int32)):
    # One thread per (env, slot): hold the attribute rows of every UNMASKED env.
    t = wp.tid()
    e = t // S
    if mask[e] != 0:
        return
    held[t] = arr[t]


@wp.kernel
def _attr_hold_vec2f_k(mask: wp.array(dtype=wp.int32), S: int,
                       arr: wp.array(dtype=wp.vec2f), held: wp.array(dtype=wp.vec2f)):
    # One thread per (env, slot): hold the attribute rows of every UNMASKED env.
    t = wp.tid()
    e = t // S
    if mask[e] != 0:
        return
    held[t] = arr[t]


@wp.kernel
def _attr_hold_vec3f_k(mask: wp.array(dtype=wp.int32), S: int,
                       arr: wp.array(dtype=wp.vec3f), held: wp.array(dtype=wp.vec3f)):
    # One thread per (env, slot): hold the attribute rows of every UNMASKED env.
    t = wp.tid()
    e = t // S
    if mask[e] != 0:
        return
    held[t] = arr[t]


@wp.kernel
def _attr_commit_f32_k(mask: wp.array(dtype=wp.int32), use_mask: int, poison: int,
                       S: int, fail_mask: wp.array(dtype=wp.int32),
                       held: wp.array(dtype=wp.float32), arr: wp.array(dtype=wp.float32)):
    # One thread per (env, slot): restore an unmasked env (masked generation), pad a
    # committed env whose fail_mask != 0 under poison.
    t = wp.tid()
    e = t // S
    if use_mask != 0:
        if mask[e] == 0:
            arr[t] = held[t]
            return
    if poison != 0:
        if fail_mask[e] != 0:
            arr[t] = _attr_pad(arr[t])


@wp.kernel
def _attr_commit_i32_k(mask: wp.array(dtype=wp.int32), use_mask: int, poison: int,
                       S: int, fail_mask: wp.array(dtype=wp.int32),
                       held: wp.array(dtype=wp.int32), arr: wp.array(dtype=wp.int32)):
    # One thread per (env, slot): restore an unmasked env (masked generation), pad a
    # committed env whose fail_mask != 0 under poison.
    t = wp.tid()
    e = t // S
    if use_mask != 0:
        if mask[e] == 0:
            arr[t] = held[t]
            return
    if poison != 0:
        if fail_mask[e] != 0:
            arr[t] = _attr_pad(arr[t])


@wp.kernel
def _attr_commit_vec2f_k(mask: wp.array(dtype=wp.int32), use_mask: int, poison: int,
                         S: int, fail_mask: wp.array(dtype=wp.int32),
                         held: wp.array(dtype=wp.vec2f), arr: wp.array(dtype=wp.vec2f)):
    # One thread per (env, slot): restore an unmasked env (masked generation), pad a
    # committed env whose fail_mask != 0 under poison.
    t = wp.tid()
    e = t // S
    if use_mask != 0:
        if mask[e] == 0:
            arr[t] = held[t]
            return
    if poison != 0:
        if fail_mask[e] != 0:
            arr[t] = _attr_pad(arr[t])


@wp.kernel
def _attr_commit_vec3f_k(mask: wp.array(dtype=wp.int32), use_mask: int, poison: int,
                         S: int, fail_mask: wp.array(dtype=wp.int32),
                         held: wp.array(dtype=wp.vec3f), arr: wp.array(dtype=wp.vec3f)):
    # One thread per (env, slot): restore an unmasked env (masked generation), pad a
    # committed env whose fail_mask != 0 under poison.
    t = wp.tid()
    e = t // S
    if use_mask != 0:
        if mask[e] == 0:
            arr[t] = held[t]
            return
    if poison != 0:
        if fail_mask[e] != 0:
            arr[t] = _attr_pad(arr[t])


#: dtype -> (hold kernel, commit kernel) of an exported attribute.
_ATTR_KERNELS = {
    wp.float32: (_attr_hold_f32_k, _attr_commit_f32_k),
    wp.int32: (_attr_hold_i32_k, _attr_commit_i32_k),
    wp.vec2f: (_attr_hold_vec2f_k, _attr_commit_vec2f_k),
    wp.vec3f: (_attr_hold_vec3f_k, _attr_commit_vec3f_k),
}


class _AttrCommit:
    """The hold / commit of one exported attribute (``[E * S]``)."""

    def __init__(self, key: str, arr, E: int, S: int, device) -> None:
        self.key = str(key)
        self.arr = arr
        self.E, self.S = int(E), int(S)
        self._hold_k, self._commit_k = _ATTR_KERNELS[arr.dtype]
        self.held = wp.empty(self.E * self.S, dtype=arr.dtype, device=device)

    def enqueue_hold(self, ctx, mask) -> None:
        ctx.launch(self._hold_k, self.E * self.S, [mask, self.S, self.arr, self.held])

    def enqueue_commit(self, ctx, mask, dummy_mask, poison: bool, fail_mask) -> None:
        ctx.launch(self._commit_k, self.E * self.S,
                   [dummy_mask if mask is None else mask, 0 if mask is None else 1,
                    1 if poison else 0, self.S, fail_mask, self.held, self.arr])


class _Commit:
    """The bound epilogue of one Program (built once by the planner).

    ``targets``: field -> the ``Track`` array the generation leaves (``fail_mask``: the
    final ``valid.fail`` version); ``always``: launch :func:`_commit_k` on unmasked
    generations too (``valid`` derived from ``valid.fail``, or ``on_invalid="poison"``).
    """

    _point_fields = POINT_FIELDS
    _env_fields = ENV_FIELDS
    _dtypes = _DTYPES

    def __init__(self, *, targets, E: int, S: int, device, dummy_mask, poison: bool,
                 derive_valid: bool, attributes=None) -> None:
        self.targets = dict(targets)
        self.E, self.S = int(E), int(S)
        self.poison = bool(poison)
        self.derive_valid = bool(derive_valid)
        self.always = self.poison or self.derive_valid
        self._dummy_mask = dummy_mask
        n = self.E * self.S
        fields = (*self._point_fields, *self._env_fields)
        self.hold = {f: wp.empty(n if f in self._point_fields else self.E,
                                 dtype=self._dtypes[f], device=device)
                     for f in fields}
        self._track = [self.targets[f] for f in fields]
        self._held = [self.hold[f] for f in fields]
        # key -> the exported attribute's final version (the result's array).
        self.attributes = [_AttrCommit(k, a, self.E, self.S, device)
                           for k, a in dict(attributes or {}).items()]

    _hold_kernel = _hold_k
    _commit_kernel = _commit_k

    def enqueue_hold(self, ctx, mask) -> None:
        """Before the stages of a masked generation: hold the unmasked envs' rows."""
        ctx.launch(self._hold_kernel, self.E * self.S,
                   [mask, self.S, *self._track, *self._held])
        for a in self.attributes:
            a.enqueue_hold(ctx, mask)

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
        ctx.launch(self._commit_kernel, self.E * self.S,
                   [self._dummy_mask if mask is None else mask, use_mask,
                    1 if self.poison else 0, self.S, *self._held, *self._track])
        for a in self.attributes:
            a.enqueue_commit(ctx, mask, self._dummy_mask, self.poison,
                             self.targets["fail_mask"])


class _GateCommit(_Commit):
    """The epilogue of a gate Program: :class:`_Commit` over the ``GateSequence``
    (:data:`GATE_POINT_FIELDS`, :data:`GATE_ENV_FIELDS`; ``S = max_gates``)."""

    _point_fields = GATE_POINT_FIELDS
    _env_fields = GATE_ENV_FIELDS
    _dtypes = _GATE_DTYPES
    _hold_kernel = _gate_hold_k
    _commit_kernel = _gate_commit_k


# ---------------------------------------------------------------------------------------
# Failing envs after the fact (roadmap PR14; private: the component kit's
# invalid.sentinel check and the library's tests)
# ---------------------------------------------------------------------------------------

@wp.kernel
def _fail_envs_k(fail: wp.array(dtype=wp.int32), fail_mask: wp.array(dtype=wp.int32)):
    # One thread per env: OR bit 0 into the fail mask of every env with fail[e] != 0.
    e = wp.tid()
    if fail[e] != 0:
        fail_mask[e] = fail_mask[e] | 1


def _commit_envs(result, fail, *, poison: bool, ctx) -> None:
    """Fail the envs with ``fail[e] != 0`` of a committed ``Track`` / ``GateSequence``
    (``fail_mask |= 1``) and run the commit's own kernel over every env (``use_mask=0``),
    then the attribute commits: ``valid`` becomes ``fail_mask == 0`` everywhere; with
    ``poison`` every env whose ``fail_mask != 0`` gets the poisoned rows (``count = 0``,
    NaN geometry, attribute padding), without it the geometry is kept (what
    ``on_invalid="keep"`` leaves). Private (``check_component``'s ``invalid.sentinel``
    builds K and P with it); host checks, then launches on ``ctx`` only.

    Args:
        result: A ``Track`` or ``GateSequence`` (a view works) with ``fail_mask`` and
            every committed field set.
        fail: ``[E]`` int32 on the result's device.
        poison: ``on_invalid="poison"`` semantics.
        ctx: The :class:`~track_gen.exec.LaunchContext` to launch on.
    """
    from .._src.types import GateSequence  # noqa: PLC0415 - avoid an import cycle

    lay = result.validate()
    E, S = int(lay.num_envs), int(lay.stride)
    gates = isinstance(result, GateSequence)
    fields = (GATE_POINT_FIELDS + GATE_ENV_FIELDS) if gates else (POINT_FIELDS + ENV_FIELDS)
    arrays = [getattr(result, f, None) for f in fields]
    missing = [f for f, a in zip(fields, arrays) if a is None]
    if missing:
        raise ValueError(f"_commit_envs: {type(result).__name__} has no {missing}")
    if not isinstance(fail, wp.array) or fail.dtype is not wp.int32 or \
            tuple(fail.shape) != (E,) or str(fail.device) != str(result.count.device):
        raise ValueError(f"_commit_envs: fail must be an [{E}] int32 array on "
                         f"{result.count.device}")
    fm = result.fail_mask
    ctx.launch(_fail_envs_k, E, [fail, fm])
    kernel = _gate_commit_k if gates else _commit_k
    p = 1 if poison else 0
    # use_mask=0: the held arrays are never read (the targets stand in for them).
    ctx.launch(kernel, E * S, [fail, 0, p, S, *arrays, *arrays])
    for arr in dict(getattr(result, "attributes", None) or {}).values():
        _, commit = _ATTR_KERNELS[arr.dtype]
        ctx.launch(commit, E * S, [fail, 0, p, S, fm, arr, arr])
