"""A fake CUDA device for exercising capture/replay POLICY on a CPU-only machine.

``FakeCuda(monkeypatch)`` patches the few Warp entry points the execution layer
(``track_gen.exec``) calls on a CUDA device so the record / eager / self-graph
decisions can be checked without a GPU:

- ``wp.get_device`` resolves ``"cuda..."`` (and the fake device itself) to
  :class:`FakeDevice`; ``wp.get_stream`` returns ``fake.current``;
- ``wp.Stream`` becomes :class:`FakeStream` (creations are recorded in
  ``fake.created``);
- ``wp.ScopedCapture`` records the capture (``fake.captures``) and makes its
  stream report ``is_capturing`` while open; the resulting graph is a
  :class:`FakeGraph`;
- ``wp.capture_launch`` records ``(graph, stream)`` in ``fake.launched``;
- ``track_gen.exec._compat.stream_synchronize`` records the stream in
  ``fake.synced``.

A caller's capture is simulated by setting ``stream.is_capturing = True`` on the
stream a call resolves (e.g. ``fake.current``). No kernel ever runs: the enqueue
bodies under test must be fakes (or patched pipelines).
"""
from __future__ import annotations

import itertools

import warp as wp

from track_gen.exec import _compat

_handles = itertools.count(0x1000)


class FakeDevice:
    is_cuda = True
    is_cpu = False
    ordinal = 0

    def __init__(self, alias: str = "cuda:0") -> None:
        self.alias = alias

    def __str__(self) -> str:
        return self.alias

    def __repr__(self) -> str:
        return f"FakeDevice({self.alias!r})"


class FakeStream:
    """Stands in for ``wp.Stream`` (the class is patched in, so isinstance works)."""

    created: list = []

    def __init__(self, device=None, priority: int = 0, capturing: bool = False) -> None:
        self.device = device if device is not None else FakeDevice()
        self.cuda_stream = next(_handles)
        self.is_capturing = capturing
        self.waited_on: list = []
        FakeStream.created.append(self)

    def wait_stream(self, other, event=None) -> None:
        self.waited_on.append(other)


class FakeGraph:
    def __init__(self, stream) -> None:
        self.stream = stream


class FakeCuda:
    def __init__(self, monkeypatch, alias: str = "cuda:0") -> None:
        self.device = FakeDevice(alias)
        FakeStream.created = []
        self.current = FakeStream(self.device)
        FakeStream.created = []          # only count streams created after setup
        self.captures: list = []
        self.launched: list = []
        self.synced: list = []
        fake = self
        real_get_device = wp.get_device

        def get_device(d=None):
            if isinstance(d, FakeDevice):
                return d
            if isinstance(d, str) and d.startswith("cuda"):
                return fake.device
            return real_get_device(d)

        class ScopedCapture:
            def __init__(self, device=None, stream=None, force_module_load=None,
                         external=False, **kwargs) -> None:
                self.device = device
                self.stream = stream if stream is not None else fake.current
                self.force_module_load = force_module_load
                self.external = external
                self.graph = None

            def __enter__(self):
                self.stream.is_capturing = True
                fake.captures.append(self)
                return self

            def __exit__(self, exc_type, exc, tb):
                self.stream.is_capturing = False
                self.graph = FakeGraph(self.stream)
                return False

        monkeypatch.setattr(wp, "get_device", get_device)
        monkeypatch.setattr(wp, "get_stream", lambda d=None: fake.current)
        monkeypatch.setattr(wp, "Stream", FakeStream)
        monkeypatch.setattr(wp, "ScopedCapture", ScopedCapture)
        monkeypatch.setattr(wp, "capture_launch",
                            lambda graph, stream=None: fake.launched.append((graph, stream)))
        monkeypatch.setattr(_compat, "stream_synchronize", fake.synced.append)

    @property
    def created(self) -> list:
        """Streams created through ``wp.Stream`` since setup (private capture streams)."""
        return FakeStream.created


class FakeSeedState:
    """Stands in for :class:`track_gen.seeds.SeedState` in facades built on the fake
    device (the real one allocates). Records the stream of every effective-seed prologue
    (the FIRST launch of every generator body) in ``sink`` and every advance in
    ``advanced`` as ``(mask, stream)``."""

    def __init__(self, sink: list, advanced: list) -> None:
        self.sink = sink
        self.advanced = advanced
        self.epoch = object()

    def effective(self, out, mask=None, *, stream=None, ctx=None) -> None:
        self.sink.append(ctx.stream)

    def advance(self, mask=None, *, stream=None, ctx=None) -> None:
        self.advanced.append((mask, ctx.stream))

    def check_mask(self, mask, name: str = "mask") -> None:
        pass


def patch_facade_seeds(monkeypatch) -> "tuple[list, list]":
    """Make both facades build a :class:`FakeSeedState`; returns ``(prologues,
    advances)``, the lists every fake records into."""
    from track_gen._src import gate_generator, track_generator

    prologues: list = []
    advances: list = []
    for mod in (track_generator, gate_generator):
        monkeypatch.setattr(mod, "SeedState",
                            lambda *a, **k: FakeSeedState(prologues, advances))
    return prologues, advances
