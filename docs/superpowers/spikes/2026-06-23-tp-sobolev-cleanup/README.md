# TP-Sobolev Cleanup Spike

This directory records the June 2026 TP-Sobolev relaxation spike. The goal was to
revisit TP-Sobolev after the self-intersection fixes and evaluate whether it is
useful as a second-stage cleanup or smoothing finisher.

## Artifacts

- `plot_tp_finisher/`: early pre/post TP-Sobolev finisher overlays and metric
  plots.
- `gentle_smoothing/`: original centerline, XPBD, local smoother, and gentle
  TP-Sobolev comparisons.
- `anchor_experiment/`: anchored TP-Sobolev variants, including 8-step vs
  16-step comparisons.
- `raw_tp_checkpoint/`: raw TP-Sobolev from original centerlines with checkpoint
  selection, including final, first-valid, and balanced selections.
- `xpbd256_tp_highres/`: XPBD at 256 points followed by higher-resolution
  TP-Sobolev cleanup at 384/512 points.
- `plot_tp_finisher.py`: plotting scaffold used for the initial finisher plots.

## Main Findings

- Standalone TP-Sobolev from original centerlines is better after the
  self-intersection fixes, but still tends to round tracks toward circles when
  allowed to run too long.
- Lower TP step counts or checkpoint selection preserve shape better than taking
  the final TP iterate unconditionally.
- A checkpointed "first valid" or "balanced" selection is the most promising
  way to expose TP-Sobolev as a real phase-2 relaxation method.
- Running XPBD at 256 points and then doing TP-Sobolev at a higher resolution
  improved segment-length smoothness with small displacement, especially at
  512 points.
- Anchored TP-Sobolev can reduce drift, but the anchor policy is the hard part:
  fixed points need to come from a meaningful geometric criterion, not arbitrary
  indices.

## Resume Points

- Production integration direction: keep the existing XPBD path graph-captured
  end to end. For `relax_solver="tp_sobolev_checkpointed"`, capture only phase 1
  and run the torch TP-Sobolev phase 2 eagerly before Warp inflation.
- Smoothing finisher thread: resume from `gentle_smoothing/` and
  `xpbd256_tp_highres/`. The most useful follow-up is a controlled finisher
  mode after XPBD with:
  - high-resolution TP phase 2, likely 512 points;
  - modest `tp_tau`;
  - early-stop/checkpoint selection instead of fixed final iterate;
  - weak or optional repulsion if we want smoothing more than clearance.
- Anchor thread: revisit only if we can define anchors from stable features such
  as high-curvature extrema, start/finish constraints, or low-confidence
  segments from the generator.
