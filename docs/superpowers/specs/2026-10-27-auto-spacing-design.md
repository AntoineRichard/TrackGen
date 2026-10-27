# Automatic spacing after config replacement

The user approved PR4 review finding #4 on 2026-10-27 (“Do 2.”), including its proposed public contract. The review is the design authority: https://github.com/AntoineRichard/TrackGen/pull/4#discussion_r4370409300.

Keep `TrackGenConfig.spacing` exactly as requested: `None` means automatic. A read-only `resolved_spacing: float` returns `0.6 * float(half_width)` for automatic spacing, otherwise `float(spacing)`. Frozen-config replacements therefore follow changed widths without losing an explicit spacing. Validate explicit spacing with the existing positive-value rule. Resolve spacing at the config-to-stage boundary; Params/specs always hold a concrete number. Update numeric readers and current guidance, plus the retained 0.2 migration record.

Materializing spacing in `__post_init__` was rejected because `dataclasses.replace` cannot distinguish automatic from explicit. A hidden tracking field or compatibility sentinel adds unnecessary state. The approved property preserves caller intent directly.

All existing generation inputs must retain their rc3 output bytes; only newly corrected width replacements intentionally select the newly derived spacing. Keep RNG/defaults, solver arithmetic, goldens, media assets and immutable tags unchanged. Use one commit for this finding, reply in its review thread with the SHA, then resolve it. No merge, final release or candidate retag is authorized. Racing validation is being arranged independently on immutable rc3.
