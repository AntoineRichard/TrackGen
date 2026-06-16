"""Scaffolding tests: prove the package and its test dir import cleanly."""


def test_package_imports():
    import track_gen

    assert hasattr(track_gen, "__version__")
    assert isinstance(track_gen.__version__, str)


def test_geometry_module_imports():
    from track_gen import geometry

    assert geometry is not None
