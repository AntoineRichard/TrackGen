"""Private knobs for torch oracle experiments, outside the runtime API."""
import math
from dataclasses import dataclass
from track_gen._src.types import TrackGenConfig


@dataclass
class OracleConfig(TrackGenConfig):
    min_angle: float = (12.5 / 180) * math.pi
    num_harmonics: int = 5  # K
    decay_p: int = 2  # decay exponent: amplitude ~ amp / k**decay_p
    amplitude: float = 1.0
    num_centerline_samples: int = 256  # Fourier dense sample count (M_max)
    relax_chunk_size: int | None = None
    relax_use_warp: bool | None = None
    energy_steps: int = 800
    energy_lr: float = 3e-3
    energy_w_sep: float = 80.0
    energy_w_len: float = 8.0
    energy_w_bend: float = 1.0
    energy_w_anchor: float = 0.01
    tp_iters: int = 100
    tp_tau: float = 0.7
    tp_alpha: float = 2.0
    tp_beta: float = 4.5
    smooth_finish: bool = False
    smooth_finish_iters: int = 8
    smooth_finish_tau: float = 0.2
    max_regen_iters: int = 10
