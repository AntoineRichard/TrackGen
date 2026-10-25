"""Private independent scalar numeric references retained for migration parity."""
import importlib


def spec(name):
    from track_gen.plugin import GeneratorSpec
    module = importlib.import_module(f"tests._reference_generators.warp_generate{'' if name == 'bezier' else '_' + name}")
    return GeneratorSpec(name=name, alloc_scratch=module._alloc_scratch, generate=module._generate)


def apply_profile(config, seeds, count, stride, cum, perim, z, ctx=None):
    from track_gen._src import warp_zprofile
    from track_gen.exec.context import resolve
    import warp as wp
    ctx = resolve(ctx, str(config.device))
    dyn = {name: wp.full(config.num_envs, float(getattr(config, name)), dtype=wp.float32,
                         device=config.device)
           for name in warp_zprofile._DYNAMIC.get(config.z_profile, ())}
    return warp_zprofile._apply_z_profile(config, seeds, count, stride, cum, perim, z, ctx, dyn)


def apply_knots(config, seeds, count, stride, arclen, perim, knot_cum, knot_count, knot_z, z, ctx=None):
    from track_gen._src import warp_zprofile
    from track_gen.exec.context import resolve
    import warp as wp
    ctx = resolve(ctx, str(config.device))
    dyn = {name: wp.full(config.num_envs, float(getattr(config, name)), dtype=wp.float32,
                         device=config.device)
           for name in warp_zprofile._DYNAMIC.get(config.z_profile, ())}
    return warp_zprofile._apply_z_profile_knots(config, seeds, count, stride, arclen, perim,
                                               knot_cum, knot_count, knot_z, z, ctx, dyn)


def generate_raw(name, seeds, config, out_centerline, out_valid_wp, scratch, ctx=None):
    """Convenient test adapter to a current built-in's private Stage implementation."""
    import importlib
    import warp as wp
    from track_gen.exec.context import resolve
    out, valid = out_centerline, out_valid_wp
    mod = importlib.import_module("track_gen._src.warp_generate" +
                                  ("" if name == "bezier" else "_" + name))
    ctx = resolve(ctx, str(config.device))
    if name == "repulsive":
        size = wp.full(config.num_envs, float(config.size), dtype=wp.float32, device=config.device)
        return mod._generate_repulsive(seeds, config, out, valid, scratch, ctx, size=size)
    dyn = {field: wp.full(config.num_envs, getattr(config, field),
                          dtype=wp.int32 if field == "min_num_points" else wp.float32,
                          device=config.device)
           for field in mod._DYNAMIC}
    return getattr(mod, "_enqueue_" + name)(seeds, config, out, valid, scratch, ctx, dyn)
