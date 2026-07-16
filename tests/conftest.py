# track_gen is installed editable (`pip install -e .`), so `import track_gen` works
# from anywhere — no sys.path manipulation needed, and the package's types module does
# not shadow the stdlib `types` module.

# Warp >= 1.14 requires wp.init() before interop entry points like wp.from_torch;
# tests that call them directly (e.g. test_warp_relax) would otherwise fail with
# 'NoneType' has no attribute 'cuda_devices'. Library code paths init themselves.
try:
    import warp as _wp
    _wp.init()
except ImportError:
    pass
