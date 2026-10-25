# Generator explanations and light/dark media

This refines the approved racing documentation design in response to the user's
2026-10-25 feedback. Keep Sphinx/Furo and the existing scientific rendering tools.
No library output, RNG, sampling or fallback behavior changes are authorized by
this media request. Repulsive release validation continues independently.

## Requested result

Explain Bezier/Hull point selection, offsets, ordering, splines and the actual
whole-loop crossing fallback. Explain Polar's regular angular slots, sampled
radii, slot-relative angular jitter, Catmull-Rom and raw arc samples. Distinguish
fixed-count raw sampling from configured target spacing, XPBD changes and the
final same-count uniform redistribution. Use readable paced visual stages.

Produce true light/dark variants of every published documentation illustration,
poster and movie. The initial inventory is 67 illustrations, 2 additional XPBD
posters and 8 movies; the new common sampling-tail clip adds a movie and poster, and Polar gains a
landscape movie poster. The resulting inventory is 80 original files plus 80 dark
counterparts.
Internal historical records and Sphinx's own UI icons are outside this scope.

## Approach

Use real palette-specific renders and native picture/source selection, plus one
video element whose source/poster follows Furo's selected theme. Preserve playback
position and paused intent when switching. CSS inversion would distort scientific
colors; hidden duplicate players would complicate controls and accessibility.
Keep existing light URLs and store dark counterparts under `_static/media-dark/`.
The screenshot of the parameter explorer needs current real UI captures, using a
separate optional-dependency environment rather than recoloring an old raster.

Use recorded buffers/public stage taps and canonical replay primitives. Reference
views with jitter suppressed are explanatory counterfactuals, clearly labelled;
they are not literal intermediate runtime states. Observe Hull before its rejected
smooth buffer is overwritten. Keep existing primary seeds; use declared default
side cases Bezier97 and Hull23 to show actual whole-loop fallbacks.

## Work split

1. Bezier/Hull agent: new storyboard module/tests, actual grid IDs and offsets,
   ordered polygon, handles/displacement, smooth candidate and fallback examples.
2. Polar agent: new storyboard module/tests, correct base ring and independent
   radius/angular changes, actual raw samples and normalization.
3. Theme agent: palette/context helpers, Sphinx media transform, theme-selection
   JS/CSS, focused tests and non-generator renderer integration.
4. Root: generator dispatcher, common sampling-tail clip, RST/captions, config
   integration and final generation/validation. Agents do not modify library code.

Commit completed renderer sources before final captures. Cache actual numerical
capture data across palettes where safe; both themes must use identical samples,
counts, failures and stage data. Record actual capture source and renderer hashes.
Do not relabel an earlier capture with a later commit.

## Validation

Verify observations/replays match ordinary raw, validity and public output bytes.
Check radius/angular units, crossing fallback choice, and both sampling stages
against canonical buffers. Show a genuinely crossing example rather than imply a
local repair. Check measured spacing along the source curve and distinguish chord
lengths on bends. Run focused renderer/theme tests and doc examples, strict Sphinx,
and browser checks in light/dark/auto, on desktop and mobile. Verify one visible
image/player per figure, theme toggles during playback and while paused, and all
published media pairs exist. Inspect representative stills and decoded movie
frames for contrast and readable labels in each theme.

The full CPU pass already recorded at combined984 and Repulsive CUDA gates remain
valid evidence for their exact scopes; later documentation changes receive their
own focused checks and the final combined CI run. Final release/main deployment
still requires the user's explicit approval after racing validation.
