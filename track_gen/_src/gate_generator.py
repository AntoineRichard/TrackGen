from .types import GateGenConfig, GateSequence

__all__ = ["GateGenConfig", "GateGenerator", "GateSequence"]


class GateGenerator:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GateGenerator is implemented in the gate facade task")
