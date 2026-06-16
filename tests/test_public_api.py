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
