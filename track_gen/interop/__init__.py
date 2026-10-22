"""Interoperability helpers for other frameworks.

- :mod:`track_gen.interop.torch`: ``registered()`` (make a ``torch.cuda.graph``
  capture visible to Warp, enabling conditional skips and lifetime anchoring) and
  ``retain()`` (keep library objects alive with a torch graph).

Importing this package never imports torch; the torch helpers import it lazily
when called.
"""

__all__ = ["torch"]


def __getattr__(name):
    if name == "torch":
        import importlib  # noqa: PLC0415
        return importlib.import_module(".torch", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
