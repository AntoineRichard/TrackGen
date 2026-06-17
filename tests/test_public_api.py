import track_gen


def test_geometry_primitives_are_reexported():
    names = [
        "safe_normalize",
        "polygon_area",
        "ccw_sort",
        "segment_directions",
        "vertex_tangents",
        "turning_number",
        "menger_curvature",
        "tangents_normals",
        "arc_length_resample",
        "nearest_nonadjacent_distance",
    ]
    for name in names:
        assert hasattr(track_gen, name), f"track_gen.{name} is not exported"
        assert callable(getattr(track_gen, name))


def test_relaxation_surface_exported():
    import track_gen
    assert hasattr(track_gen, "relax")
    from track_gen import relaxation  # module importable
    from track_gen.geometry import thickness, self_intersections, separation_min
    assert callable(track_gen.relax)
