# TangoPosture Settings: practical examples for beginners

This guide follows the current desktop Settings screen: **72 editable settings
and eight Calibration defaults rows**. Each parameter is explained once, with a
worked example. Examples illustrate software behaviour, not medical limits or
universal tango technique.

The guide covers **Library video import, reference calibration, Review and
optional Ollama commentary**. Unavailable live-camera, microphone and automatic
recording controls, removed baseline controls and their adjustment recipes are
not part of this guide or current Settings exports.

Older configuration files remain loadable for backend compatibility. New
Settings exports contain the current editable controls and Calibration defaults
only, not a complete backup of nondefault legacy configuration.

## Contents

- [Before adjusting settings](#before-adjusting-settings)
- [1. Camera & view](#1-camera--view)
- [2. Files](#2-files)
- [3. Quality & calibration](#3-quality--calibration)
- [4. Tracking](#4-tracking)
- [5. Head & shoulders](#5-head--shoulders)
- [6. Embrace & knees](#6-embrace--knees)
- [7. Torso & hip offset](#7-torso--hip-offset)
- [8. Slow tango](#8-slow-tango)
- [9. Music](#9-music)
- [10. Replay movement](#10-replay-movement)
- [11. Review & Ollama](#11-review--ollama)
- [12. Calibration defaults](#12-calibration-defaults)
- [Reference-video knees and shoulders: the actual calculations](#reference-video-knees-and-shoulders-the-actual-calculations)
- [Applying, saving and reanalyzing](#applying-saving-and-reanalyzing)
- [Practical video-analysis recipes](#practical-video-analysis-recipes)
- [Further reading and implementation](#further-reading-and-implementation)

## Before adjusting settings

### Which posture limits are active?

| Situation | Posture limits |
| --- | --- |
| A calibration profile is selected | Its approved minimum/maximum ranges. Missing metrics stay unavailable. |
| Review without a selected profile | No posture alerts, even if Settings defaults are enabled. Tracking, pivot rings and footprints remain available. |
| Live Practice without a profile, with Calibration defaults enabled | The enabled default ranges. Disabled metrics stay unavailable. |
| Live Practice without a profile or defaults | The applicable fallback descriptive-warning controls. |

Fallback controls remain in use in Live Practice and supported legacy analysis.
They do **not** override selected/default approved ranges.
Role/embrace recognition, tracking quality and the slow-tango lift cue have
their own purposes and must not be confused with choosing posture limits.

### How to use the examples

1. Inspect the dots and the correct dancer first.
2. Change one parameter and click **Apply & Save settings.toml**.
3. Reanalyze when changing the model or cached processing results.
4. Compare the same video passage before and after.
5. Confirm the saved setting is suitable.

An unavailable measurement means insufficient usable evidence, not bad
posture. Wider ranges cannot fix a misplaced joint. Interpolated drawing is
not recovered measurement evidence.

**Units:** a ratio of `0.06` means six percent; a percentage field uses `6`
for six percent. Degrees, seconds, pixels and language-model tokens are
different units. None of these settings is a height/weight/body-build preset.

## 1. Camera & view

Despite the retained tab name, this tab now contains video measurement/view
controls, not live-camera device, resolution or mirroring controls.

### Measurement side - `side`

**Default:** `left`; choices `left`, `right`.

The anatomical side preferred by side-dependent geometry. This is the dancer's
left/right, not the screen's. Some helpers can use another usable side.

**Example:** if the dancer's right ear/shoulder/hip chain is the reliably visible
one, choose `right` before producing reference evidence and matching analysis.
An existing reference editor restores its cached capture settings; changing this
dropdown does not relabel an already cached reference.

### Facing direction - `facing`

**Default:** `right`; choices `left`, `right`.

Direction faced in the original image, used to interpret signed forward/
posterior measurements. It is not a request to flip the video.

**Example:** a dancer facing screen-left should use `left`. Leaving `right`
can reverse the interpretation of a signed offset. Reanalyze matching evidence
after correcting the setup rather than compensating with wider posture limits.

### Capture setup name - `capture_profile`

**Default:** `default`.

An identifier stored with capture metadata. It describes the setup; it does not
change the camera or landmarks.

**Example:** use `studio-side-a` for new evidence from one arrangement and
`studio-front-b` for another. These names help distinguish saved setups.
Renaming the setting does not rotate existing footage or update cached metadata.

### Allow all angles - `allow_all_angles`

**Default:** disabled.

Selects the more permissive core measurement path, which can use shoulder/hip
midpoints or usable alternative sides. It does not invent hidden joints.
Approved metrics still enforce their own `side`, `front` or `all` view.

**Example:** enabling it can let core geometry survive a usable front-facing
passage that the strict side-view path rejects. A metric approved only for
`side` will still be unavailable outside its approved view.

## 2. Files

### Lite model file - `model_path`

**Base default:** `models/pose_landmarker_lite.task`; packaged paths can differ.

Location of the pinned Lite model. The selected file must pass checksum
verification. This does not turn a Lite model into Full or Heavy.

**Example:** with Offline analysis model set to `lite`, select an existing,
verified Lite `.task` file with **Browse...** to reuse it. Renaming a Heavy file
to the Lite filename fails verification. Missing offline models are downloaded
when no local candidate file exists. A corrupt cached candidate raises a
verification error rather than silently triggering another download.

### Offline analysis model - `analysis_model`

**Default:** `lite`; choices `lite`, `full`, `heavy`.

Selects the MediaPipe variant used by new Library/Review analysis. Existing
local assets are verified and reused; missing assets download and are cached.

**Example:** switch from `lite` to `full`, then explicitly reanalyze a recording.
Compare its dots and processing time on the same passage. Merely changing the
selector leaves saved results unchanged. Calibration and comparison evidence
must have compatible model identities.

### Data folder - `data_dir`

**Base default:** `data`.

Base location for app data and practice artifacts. Packaged startup can supply
a different path. This is not the destination of every Save/Export action.

**Example:** changing from `data` to `sessions-b` can show an empty Library
because it now searches another location. It does not delete the old archives.
The **Apply & Save settings.toml** destination remains the working directory.

## 3. Quality & calibration

These settings determine whether evidence is usable, not whether posture is
correct. The approved posture limits are configured separately.

### Minimum landmark confidence - `confidence`

**Default:** `0.70`; allowed `0` to `1`.

Strict app-side checks require finite coordinates, in-image positions, and
sufficient visibility and presence. Some descriptive/movement helpers have
their own thresholds.

**Example:** a point with visibility `0.65` and presence `0.95` fails a `0.70`
strict gate but passes a `0.60` gate if the other checks pass. Lowering the
setting admits weaker evidence; it does not make the dot more accurate.

### Min torso - `min_torso`

**Default:** `0.12`; positive fraction of image height.

Minimum projected shoulder-to-hip length for core geometry. This does not
control every helper independently.

**Example:** in a 480-pixel-high video, the minimum is `57.6` pixels. A
50-pixel torso fails this check. Reducing the setting to `0.10` changes the
minimum to 48 pixels, but improving framing is preferable to admitting tiny,
uncertain body segments.

### Min head - `min_head`

**Default:** `0.04`; positive fraction of image height.

Minimum core shoulder-to-ear segment, **not the whole head**. It qualifies
head-forward/head-inclination evidence; the ear-to-nose pitch helper has
separate guards.

**Example:** at 480 pixels high, the minimum is `19.2` pixels. A 17-pixel
segment is unavailable for those core head metrics. A value of `0.03` lowers
the minimum to `14.4` pixels. Failure does not automatically invalidate every
other metric or the whole calibration.

### Max view width - `max_view_width`

**Default:** `0.45`; positive ratio.

Separates the projected side/front view policies. Side requires a usable ratio
at or below the boundary; front requires one above it. `all` skips this split,
not landmark-quality checks.

**Example:** a usable ratio of `0.48` fails a side-approved metric at `0.45`
but can pass that view gate at `0.50`. The same change can make it fail a
front-approved metric. Check the actual viewpoint before changing this boundary.

### Suggested lower percentile (%) - `calibration_suggest_lower_pct`

**Default:** `10%`; from `0` up to below the upper suggested percentile.

Lower percentile proposed when clicking the suggestion button in the Calibration
editor. It does not restrict manually entered bounds.

**Example:** across 100 sorted samples from a balanced standing passage, `10%`
proposes the 10th sample as the minimum limit instead of the single lowest
outlier. Setting `5%` widens the suggested lower bound to cover 95% at the bottom.

### Suggested upper percentile (%) - `calibration_suggest_upper_pct`

**Default:** `90%`; above the lower suggested percentile up to `100%`.

Upper percentile proposed when clicking the suggestion button in the Calibration
editor. Pair it with `calibration_suggest_lower_pct` (default 10th to 90th =
middle 80%).

**Example:** with `10%` lower and `90%` upper on a steady walking clip, the
editor proposes the middle 80% of eligible values; setting `20%` and `80%`
narrows the suggested proposal to the middle 60%.

### Minimum usable calibration samples - `calibration_min_usable_samples`

**Default:** `2`; integer `2` or greater.

Minimum finite, reliable samples required inside selected passages before the
Calibration editor summarizes a metric for approval.

**Example:** in a short 30 FPS clip where only `4` frames pass quality checks,
a minimum of `2` allows summarizing and approving the passage, whereas raising
it to `10` rejects that sparse clip and asks for more reliable frames.

## 4. Tracking

These controls smooth numbers and time events. Set acceptable posture ranges
in the selected calibration or Calibration defaults, not in this tab.

### Measurement smoothing (seconds) - `smoothing`

**Default:** `0.25`; positive time constant.

Makes evaluated values respond gradually. Higher values smooth more but follow
changes more slowly.

**Example:** after a sudden change, over `0.1` seconds a `0.25`-second filter
moves about 33% toward the new value; a `0.50`-second filter moves about 18%.
This is not a fixed video delay and does not change raw cached landmarks.

### Overlay smoothing - `overlay_smoothing`

**Default:** `0.20`; zero or greater.

Current offline processing uses this for some descriptive head/shoulder/knee
values. It is not the control for Review's same-frame skeleton or interpolation.
Some descriptive filters use Measurement smoothing when this is zero.

**Example:** changing `0.20` to `0.10` makes applicable descriptive values
follow a change faster, with more possible jitter. Setting zero does not remove
every filter. Reanalyze to regenerate stored descriptive results.

### Minimum filter time constant (seconds) - `min_filter_time_constant`

**Default:** `0.0001` seconds; positive.

Safety floor clamped onto the exponential smoothing time constant for
descriptive knee and shoulder filters (`max(min_filter_time_constant, overlay_smoothing or smoothing)`),
preventing division by zero when smoothing is very small or zero.

**Example:** with default `0.0001`, if both Overlay smoothing and Measurement
smoothing were set near `0.00005`, the descriptive knee/shoulder filter uses
`0.0001` seconds. Raising it to `0.05` guarantees at least `50 ms` smoothing.

### Entry duration - `entry_duration`

**Default:** `1.5 seconds`; positive.

Time a continuous evaluated excursion must persist before confirming an event.
It does not change the acceptable range.

**Example:** a `0.4`-second excursion outside approved bounds does not confirm
at `1.5`, but can at `0.3` if valid observations remain continuous. Raising
the duration to `2.0` ignores more brief excursions and confirms later.

### Clear duration - `clear_duration`

**Default:** `0.75 seconds`; positive.

Time recovery must persist before an active event clears.

**Example:** returning inside an approved interval for `0.4` seconds is not
enough at `0.75`; returning for `0.8` seconds can clear it when observations
qualify. At `0.25`, the same event can clear sooner.

### Max gap - `max_gap`

**Default:** `0.5 seconds`; positive.

Continuity/reset limit for applicable tracking and evaluation paths. Missing
raw measurements still become UNKNOWN.

**Example:** a `0.7`-second observation gap exceeds `0.5`, so the applicable
filter/event continuity resets. Increasing to `0.8` permits that timestamp gap
but does not fill missing joints. The calibration editor uses the gap policy
saved with its reference; changing current Settings alone does not replace it.

## 5. Head & shoulders

These are **active fallback descriptive controls** for live coaching when
neither a selected profile nor enabled default ranges supplies posture limits.
They do not define approved native ranges or trigger uncalibrated Review alerts.

### Head neutral offset - `head_neutral_offset`

**Default:** `-12 degrees`.

Subtracted from raw ear-to-nose pitch for the corrected fallback cue.

**Example:** raw pitch `-12` minus offset `-12` gives corrected zero.
Changing the offset to `-8` gives `-4`; changing it to `-16` gives `+4`.
Approved `head_pitch` instead compares the raw angle without this subtraction.

### Head down duration - `head_down_duration`

**Default:** `3 seconds`; positive.

Persistence required for the fallback downward-head warning.

**Example:** corrected pitch below its limit for two seconds does not confirm
this warning at `3`. Setting the duration to `1` makes a qualifying two-second
passage long enough. Approved-range events use Tracking's Entry duration.

### Head down angle limit (degrees) - `head_down_angle`

**Default:** `-15 degrees`; must remain negative.

Boundary for corrected fallback ear-to-nose pitch.

**Example:** corrected pitch `-12` does not cross a `-15` limit, but does cross
`-10`. Changing to `-20` tolerates more downward pitch. The duration must also
qualify before a warning; this setting does not move a selected profile's bounds.

### Shoulder tension duration - `shoulder_tension_duration`

**Default:** `1 second`; positive.

Persistence before the fallback shoulder-elevation banner. The name does not
mean that muscle tension is measured.

**Example:** a `0.6`-second qualifying elevation does not confirm at `1`,
but can at `0.5`. Increasing to `2` ignores more brief changes. Raw clearance
and approved clearance bounds are unchanged.

### Shoulder tension limit (%) - `shoulder_raise_threshold`

**Default:** `10%`; positive.

Boundary for computed fallback elevation relative to effective neutral
clearance. This is a percentage, not the approved clearance ratio.

**Example:** computed elevation `12%` exceeds a `10%` limit, but not `15%`.
Lowering the limit to `6%` admits more warning candidates. Persistence and
recovery rules still apply.

### Shoulder neutral clearance - `shoulder_neutral_clearance`

**Default:** `1.05`; positive.

Starting fallback baseline for `100 * (baseline - raw clearance) / baseline`.
The helper clamps its effective baseline to at least `0.20`; offline processing
can also adapt it from qualifying observations.

**Example:** at raw clearance `0.95`, a baseline of `1.00` gives `5%` elevation,
while `1.10` gives about `13.6%`. Raising the baseline can therefore make warning
more likely without moving any dot. It does not change raw approved clearance.

### Shoulder helper visibility floor - `shoulder_visibility_floor`

**Default:** `0.50`; from `0` to `1`.

Minimum landmark visibility checked inside the shoulder clearance geometry helper
for ear, shoulder, hip, and nose points, alongside any stricter approval mask.

**Example:** if an occluded ear has visibility `0.45`, it is excluded at `0.50`;
lowering the helper floor to `0.40` lets fallback/descriptive shoulder estimates
read that point (unless the stricter approval `confidence` mask rejects it).

### Shoulder ear-to-hip scale factor - `shoulder_ear_hip_scale_factor`

**Default:** `0.26`; positive multiplier.

Primary normalization factor multiplying vertical ear-to-hip pixel height to
compute the reference scale `S = factor * ear_hip_dy` for shoulder clearance.

**Example:** when vertical ear-to-hip distance is `200` pixels, `0.26` produces a
`52`-pixel reference scale `S`; a `52`-pixel shoulder/ear gap gives raw clearance
`1.0`. Setting `0.25` gives `S = 50` px and ratio `1.04` for the same pixel gap.

### Shoulder min ear-to-hip height (px) - `shoulder_min_ear_hip_px`

**Default:** `20.0` pixels; positive.

Minimum vertical ear-to-hip pixel separation required for a side's primary torso
scale candidate to qualify before falling back to head/shoulder-width scales.

**Example:** in a very small distant crop where vertical ear-to-hip distance is
only `18` pixels, `20.0` skips the primary torso scale and uses head/shoulder
fallbacks instead; setting `15.0` keeps the primary torso scale active there.

### Shoulder fallback nose-ear scale factor - `shoulder_nose_ear_scale_factor`

**Default:** `0.68`; positive multiplier.

Fallback reference scale multiplier applied to the 2D pixel distance between
nose and ear when no primary ear-to-hip candidate qualifies.

**Example:** with hips out of frame and nose-to-ear distance `50` pixels, `0.68`
yields fallback reference scale `S = 34` pixels. Adjusting to `0.70` gives `35`
pixels to calibrate head-scale continuity for that camera angle.

### Shoulder fallback ear-ear scale factor - `shoulder_ear_ear_scale_factor`

**Default:** `0.65`; positive multiplier.

Fallback reference scale multiplier applied to horizontal pixel distance between
left and right ears when both ears are usable and no primary torso scale qualifies.

**Example:** if horizontal ear-to-ear width is `60` pixels in a tight upper-body
shot, `0.65` gives candidate scale `39` pixels (competing with the other fallback
candidates, largest wins). Changing to `0.60` yields `36` pixels.

### Shoulder fallback width scale factor - `shoulder_width_scale_factor`

**Default:** `0.28`; positive multiplier.

Fallback reference scale multiplier applied to horizontal pixel distance between
left and right shoulders when no primary torso scale qualifies.

**Example:** if shoulder-to-shoulder width is `140` pixels with hips cropped out,
`0.28` yields `39.2` pixels as a fallback scale candidate; `0.30` yields `42.0` px.

### Shoulder min reference scale (px) - `shoulder_min_ref_scale_px`

**Default:** `8.0` pixels; positive.

Minimum selected reference scale `S` in pixels required to compute normalized
shoulder clearance (`ear_to_shoulder_dy / S`); smaller scales return unavailable.

**Example:** if the subject is far away and `S` drops to `6.5` pixels, clearance
is reported unavailable at `8.0` rather than dividing by near-zero noise. Lowering
to `5.0` permits small distant shots at higher sensitivity to pixel jitter.

### Shoulder baseline clamp floor - `shoulder_baseline_clamp_floor`

**Default:** `0.20`; positive ratio.

Minimum floor applied to the effective neutral clearance `B = max(floor, neutral_clearance)`
when computing fallback shoulder elevation percentage `100 * (B - C) / B`.

**Example:** even if neutral clearance is set very low (say `0.15`), `0.20`
prevents division by near-zero by clamping `B = 0.20`. Keeping defaults near `1.05`
never hits this floor unless configured lower.

### Adaptive baseline max samples - `shoulder_adaptive_max_samples`

**Default:** `600`; integer `1` or greater.

Maximum number of qualifying relaxed-clearance samples stored for the adaptive
neutral shoulder baseline window during fallback analysis/coaching.

**Example:** at 30 FPS, `600` samples equals about 20 seconds of qualifying frames
before the window stops appending new samples. Raising to `900` collects over 30s.

### Adaptive baseline min samples - `shoulder_adaptive_min_samples`

**Default:** `15`; integer from `1` up to `shoulder_adaptive_max_samples`.

Minimum qualifying relaxed-clearance samples required before the adaptive
percentile replaces/raises the starting neutral shoulder clearance baseline.

**Example:** for the first `14` qualifying frames at default `15`, fallback
elevation uses `shoulder_neutral_clearance` (`1.05`). On frame `15`, the `75th`
percentile enters (`max(start, percentile)`). Setting `30` waits twice as long.

### Adaptive baseline min clearance - `shoulder_adaptive_min_clearance`

**Default:** `0.65`; positive ratio below `shoulder_adaptive_max_clearance`.

Lowest raw clearance ratio `C` admitted into the adaptive relaxed-clearance
sample window, filtering out already-tense/shrugged frames from the baseline.

**Example:** a shrugged frame with raw clearance `0.55` is rejected from the
baseline window at `0.65`, avoiding teaching the baseline that tense shoulders
are "neutral". Lowering to `0.55` would admit tighter anatomy frames.

### Adaptive baseline max clearance - `shoulder_adaptive_max_clearance`

**Default:** `1.50`; positive ratio above `shoulder_adaptive_min_clearance`.

Highest raw clearance ratio `C` admitted into the adaptive baseline window,
rejecting outlier/elongated landmark spikes from raising the neutral baseline too high.

**Example:** an anatomical outlier spike of `1.65` is skipped at `1.50` so it
does not inflate the baseline. Raising to `1.80` accommodates dancers with unusually
long measured ear-to-shoulder clearance.

### Adaptive baseline percentile (%) - `shoulder_adaptive_percentile`

**Default:** `75%`; above `0` up to `100%`.

Percentile of collected qualifying relaxed-clearance samples compared against
the starting neutral clearance (`max(start, percentile)`).

**Example:** given 20 qualifying samples between `0.90` and `1.30`, `75%` takes
the 75th percentile (~`1.20`) so a naturally relaxed baseline wins over `1.05`.
Lowering to `50%` uses the median instead of the upper quartile.

### Shoulder recovery min floor (%) - `shoulder_recovery_min_pct`

**Default:** `4.0%`; zero or greater.

Minimum percentage floor used in the fallback shoulder tension hysteresis
recovery boundary `max(floor, shoulder_recovery_ratio * shoulder_raise_threshold)`.

**Example:** if `shoulder_raise_threshold` is reduced to `5%` with `0.65` ratio
(`3.25%`), the `4.0%` floor keeps the clear boundary at `4.0%` unless lowered.

### Shoulder recovery threshold ratio - `shoulder_recovery_ratio`

**Default:** `0.65`; above `0` up to `1.0`.

Fraction of `shoulder_raise_threshold` used as the fallback shoulder recovery
boundary (hysteresis band) once a shoulder warning has started.

**Example:** at default `10%` raise threshold and `0.65` ratio, the warning starts
at `10%` and clears once elevation drops below `max(4.0, 6.5) = 6.5%`. Setting
`0.80` narrows the hysteresis band so recovery occurs at `8.0%`.

## 6. Embrace & knees

Role/embrace recognition runs independently of the source of posture limits.
The last two controls below are fallback live knee warnings, not approved bounds.

### Role assign ms - `role_assign_ms`

**Default:** `300 milliseconds`; zero or greater.

Required persistence of a matching arm pattern before assigning its role label.
It is not person-identity tracking.

**Example:** a pattern visible for `0.2` seconds is too brief at `300 ms`,
but can qualify at `100 ms` once the update timestamps permit assignment.
Increasing to `600 ms` makes assignment slower and more resistant to brief noise.

### Role hold ms - `role_hold_ms`

**Default:** `5000 milliseconds`; zero or greater.

How long a confirmed role label can remain after the matching pattern disappears.

**Example:** after a turn hides the pattern for two seconds, a `5000 ms` hold
can retain the previous label; a `1000 ms` hold releases it sooner. A retained
label is history, not fresh evidence that the arms remain in that position.

### Embrace closed threshold - `embrace_closed_threshold`

**Default:** `0.72`; greater than zero and at most one.

Boundary on a heuristic openness score, not physical partner distance.
The recognizer also applies smoothing, a switching band and persistence.

**Example:** an initial smoothed score of `0.70` falls on the CLOSED side of
`0.72`, but the OPEN side of `0.65`. Raising the threshold makes CLOSED more
likely for a fixed score. State changes still depend on the recognizer's timing.

### Embrace hysteresis seconds - `embrace_hysteresis_seconds`

**Default:** `1 second`; zero or greater.

Persistence before switching between OPEN and CLOSED once the switching
conditions qualify.

**Example:** a candidate lasting `0.4` seconds is too brief at `1.0`, but can
switch at `0.3`. A longer duration gives steadier but later labels; it does not
change the measured knee angle or approve a calibration range.

### Closed embrace min knee bend - `closed_embrace_min_knee_bend`

**Default:** `8 degrees`; zero or greater.

Minimum projected knee flexion for the fallback straight-knee cue. Despite
its historical name, older offline fallback scoring was not CLOSED-gated.

**Example:** a measured/smoothed bend of `6` is below `8`, so it can start a
warning candidate; at a minimum of `5`, it does not. Raising the minimum to `12`
requires more bend. Do not bend to satisfy an arbitrary software example.

### Knee straight duration - `knee_straight_duration`

**Default:** `1 second`; positive.

Persistence required for the fallback straight-knee banner.

**Example:** a qualifying straight-looking interval of `0.7` seconds does not
confirm at `1.0`, but can at `0.5`. A selected/default approved knee range uses
Tracking's event timing instead.

### Knee recovery margin (degrees) - `knee_recovery_margin`

**Default:** `2.5` degrees; zero or greater.

Hysteresis margin added above `closed_embrace_min_knee_bend` to form the
fallback straight-knee clear boundary (`min_knee_bend + margin`).

**Example:** with `closed_embrace_min_knee_bend = 8.0` and margin `2.5`, a
straight-knee warning starts below `8.0°` and only clears once knee bend exceeds
`10.5°`. Raising margin to `4.0` requires `12.0°` bend before clearing.

### Knee helper visibility floor - `knee_visibility_floor`

**Default:** `0.35`; from `0` to `1`.

Minimum landmark visibility checked inside the knee flexion geometry helper for
hip, knee, and ankle joints, alongside any stricter approval `confidence` mask.

**Example:** if a dancer's trailing ankle dips to visibility `0.40`, it qualifies
at `0.35` in descriptive knee overlays (when approval mask isn't stricter);
raising this floor to `0.55` drops noisy low-visibility legs sooner.

### Knee min segment length (px) - `knee_min_segment_px`

**Default:** `10.0` pixels; positive.

Minimum pixel length required for both the hip-to-knee (`u`) and ankle-to-knee (`v`)
segments before computing the knee interior angle.

**Example:** if foreshortening or a tiny subject shrinks the shin segment to `8`
pixels, `10.0` rejects that leg rather than outputting unstable `atan/acos` angles.
Lowering to `6.0` accepts smaller distant subjects at higher angular noise risk.

### Knee tie-break other-leg max visibility - `knee_tiebreak_other_max_vis`

**Default:** `0.60`; from `0` to `1`.

Maximum visibility of the occluded/far leg below which the preferred-side knee
tie-break rule can override `min(near_bend, far_bend)`.

**Example:** when both legs qualify (`left` `0.90` vis at `14°` bend; `right`
`0.50` vis at `3°`), `right < 0.60` permits selecting `left` (`14°`) instead of
being misled by the noisy occluded leg. Lowering to `0.45` keeps `min(...)` active.

### Knee tie-break min visibility advantage - `knee_tiebreak_min_advantage`

**Default:** `0.15`; from `0` to `1`.

Minimum visibility advantage (`preferred_vis - other_vis >= advantage`) required
for the preferred camera-facing side to win the knee tie-break.

**Example:** with preferred leg vis `0.72` and other leg `0.55` (difference `0.17`),
`0.15` triggers preferring the camera-facing leg (provided other < `0.60`), whereas
`0.25` keeps reporting the smaller bend across both legs.

## 7. Torso & hip offset

These active fallback controls concern hip displacement from the ankle-to-head
line. **They are not pink torso-line tilt settings.**

### Axis curve entry - `axis_curve_entry`

**Default:** `0.06`; positive ratio.

Posterior corrected hip-offset boundary for starting a fallback event candidate.

**Example:** corrected deviation `0.07` exceeds `0.06`, but not `0.08`.
Raising entry tolerates more displacement; lowering it detects smaller changes.
The ratio is not degrees. Approved `hip_offset` uses its native bounds instead.

### Axis curve clear - `axis_curve_clear`

**Default:** `0.035`; nonnegative and below entry.

Recovery boundary for the active fallback hip-offset event.

**Example:** smoothed corrected deviation `0.03` is below `0.035`, allowing
clearing to begin. At a clear threshold of `0.02`, it must recover farther.
Keep clear below entry; the two thresholds have different jobs.

### Axis curve duration - `axis_curve_duration`

**Default:** `1.5 seconds`; positive.

Persistence above the fallback hip-offset entry threshold before confirmation.

**Example:** a one-second qualifying excursion does not confirm at `1.5`,
but can at `0.5`. It does not alter the hip-offset ratio or the timing of
approved-range metrics.

### Axis curve clear duration - `axis_curve_clear_duration`

**Default:** `0.75 seconds`; positive.

Persistence of recovery before the fallback hip-offset event clears.

**Example:** `0.5` seconds below the recovery threshold is insufficient at
`0.75`, but can clear at `0.3`. Increasing this setting keeps the event active
through more brief recoveries without moving either threshold.

### Axis neutral offset - `axis_neutral_offset`

**Default:** `0.0`.

Neutral value subtracted from raw hip offset for fallback corrected deviation.

**Example:** raw `0.05` minus neutral `0.02` gives corrected `0.03`.
Neutral `-0.01` instead gives `0.06`. This shifts the fallback cue, not the hip,
and does not change the approved raw `hip_offset` measurement.

## 8. Slow tango

This is a live/legacy ankle-height/consecutive-step cue, separate from offline
footprint placement and pivot detection. It does not create uncalibrated Review alerts.

### Slow tango max BPM - `slow_tango_max_bpm`

**Default:** `115 BPM`; positive and at most `240`.

This setting exists **once** in the app. It is the maximum tempo at which the
slow-tango lift cue can operate, not the current/fallback tempo itself.

**Example:** with a supplied tempo of `110 BPM`, a `115` cutoff allows the cue
to operate if ankle evidence and high-step conditions qualify. At `120 BPM`,
that cutoff disables the cue. Raising the cutoff to `125` admits `120 BPM`.
This does not change playback speed or the separate footprint detector.

### Ankle high lift ratio - `ankle_high_lift_ratio`

**Default:** `0.06`; greater than zero and at most one.

Vertical ankle separation considered high relative to estimated leg length.
Internal step-start/end boundaries are also derived from this value.

**Example:** with a 300-pixel estimated leg length, `0.06` corresponds to an
18-pixel threshold, while `0.08` corresponds to 24 pixels. A 21-pixel separation
is above the first but below the second. This is projected separation, not
certified floor clearance in centimetres.

### Max consecutive high ankle steps - `max_consecutive_high_ankle_steps`

**Default:** `2`; integer at least one.

Consecutive qualifying high steps needed for the lift alert.

**Example:** at `2`, the alert can appear during the second qualifying high
step; at `3`, it waits for another. A qualifying low step can reset the streak.
The BPM gate must also permit the cue, and hidden feet can make it unavailable.

## 9. Music

This tab now contains only the fallback tempo. It does not expose microphone,
beat-analysis or automatic-recording controls.

### Fallback tempo (BPM) - `tango_bpm`

**Default:** `120 BPM`; greater than `20` and at most `240`.

Tempo supplied when analysis has no soundtrack BPM. Current Library/Review
analysis disables soundtrack analysis, but does not show fallback lift alerts.
It is **not another copy of the slow-tango cutoff**.

**Example:** the defaults supply `120 BPM` against a `115 BPM` slow-tango
cutoff, so that lift cue is inactive. Setting the fallback to `105 BPM` makes
the BPM gate pass; an alert still needs qualifying high steps. Enter a tempo
appropriate to the clip rather than lowering it just to force an alert.

## 10. Replay movement

### Replay step history seconds - `replay_step_history_seconds`

**Default:** `5 seconds`; allowed `0.5` to `30`.

Media-time window over which observed step footprints fade out.

**Example:** increasing to `10` shows a longer trail during playback or
scrubbing; reducing to `2` leaves fewer recent footprints visible. It does
not create/delete steps or change their stored timestamps.

### Review joint interpolation - `replay_joint_interpolation`

**Default:** `linear`; choices `linear`, `quadratic`.

Display-only filling of bracketed gaps between reliable joint observations.
Quadratic attempts a curve using another sample and falls back to linear
when unsuitable.

**Example:** for a joint seen before and after a short gap, `linear` moves it
along the direct path between those observations. `quadratic` can curve that
path. Neither guarantees the actual hidden motion or supplies scoring evidence.

### Maximum joint gap to fill (seconds; 0 disables) - `replay_joint_max_gap`

**Default:** `3 seconds`; zero or greater.

Maximum separation of bracketing usable observations for drawing interpolation.
It is different from Tracking's continuity limit.

**Example:** endpoints separated by `0.4` seconds can be filled at `0.5`,
but not at `0.2`. Setting zero disables filling. Increasing to `3` bridges
longer, more uncertain intervals; there is no indefinite edge extrapolation.

## 11. Review & Ollama

Ollama is optional report commentary, not the pose tracker or calibration
evaluator. Report text is sent to the configured server. A nonlocal server
has a different privacy boundary from a local one.

### Ollama server address - `ollama_base_url`

**Default:** `http://127.0.0.1:11434`.

Address of the Ollama service, not a pose-model download location.

**Example:** use the default for a server on the same machine at port 11434.
If you deliberately run it locally at port 11435, use
`http://127.0.0.1:11435`. Changing the address does not start the server.

### Ollama model - `ollama_model`

**Default:** `llama3.2`; nonempty text.

Language model requested from Ollama. This does not select MediaPipe.

**Example:** choose an installed model name such as `llama3.2`. A name that
is unavailable on that server produces a request error; changing it to a
larger language model does not improve joint coordinates.

### Ollama timeout - `ollama_timeout`

**Default:** `120 seconds`; positive and at most `1800`.

Maximum wait for a model request.

**Example:** a request taking 150 seconds can time out at `120`; a `180`
timeout permits waiting longer. This does not make generation faster.
Reducing to `30` fails sooner when the service is slow.

### Response variation - `ollama_temperature`

**Default:** `0.2`; allowed `0` to `2`.

Requested variation in generated language.

**Example:** changing from `0.2` to `0.8` can produce more variable wording
for the same report. Setting zero requests less variation, not guaranteed
accuracy or deterministic behaviour on every server.

### Model context size (tokens) - `ollama_num_ctx`

**Default:** `16384`; integer `1024` to `131072`.

Requested language-model context capacity. Tokens are text units, not frames.

**Example:** reducing to `8192` requests less context capacity and can reduce
resource demand, but leaves less room for the prompt/report. Increasing to
`32768` requests more capacity if the server/model supports it.

### Maximum response length (tokens) - `ollama_num_predict`

**Default:** `2048`; integer `128` to `16384`.

Maximum requested generated response length, not an exact output length.

**Example:** setting `512` asks for a shorter generation budget than `2048`.
A model may still stop before either maximum. Increasing it permits a longer
answer and potentially more generation time.

### Maximum report length (characters) - `ollama_max_report_chars`

**Default:** `40000`; at least `1000`.

Budget for report text included for commentary, not the amount of raw video
analyzed.

**Example:** a 30000-character report fits a `40000` budget. At `10000`,
the supplied report content is limited and can omit useful context. Increasing
context tokens is not the same as increasing this character budget.

### Analysis focus - `ollama_instruction`

**Default:** explain supported observations and suggest three focused practice
exercises.

Additional instruction accompanying the report.

**Example:** enter "Summarize observable changes and identify unavailable
measurements." This requests different commentary; it does not change
numeric limits, run new video inference or make medical assessments valid.

## 12. Calibration defaults

Each row has **Use**, minimum, maximum and applicable view. Units are fixed.
Initially all rows are disabled. In Live Practice, once any row is enabled,
with no selected profile only the enabled metrics are available; fallback
posture limits do not fill the disabled rows. Review requires a `.tpc` profile
and does not use these defaults for alerts.

Minimum must not exceed maximum. Endpoints are included. `side` and `front`
apply their view gate; `all` does not bypass joint quality. Equal bounds are
valid configuration but can be impractically sensitive to small variation.

### Head pitch (ear-to-nose) - `head_pitch`

**Unit:** degrees; raw ear-to-nose image angle.

**Example:** enable a deliberately chosen interval `-20` to `-5`: raw `-12`
is inside, raw `-25` is outside. Lowering the minimum to `-30` admits `-25`.
The fallback neutral offset is not subtracted from these values.

### Head-forward offset - `head_forward`

**Unit:** torso lengths; signed torso-relative forward ear displacement.

**Example:** with a range `-0.02` to `0.06`, a value of `0.08` is outside.
Raising the maximum to `0.10` admits it. This changes an approved native
interval, not a change-from-baseline threshold.

### Head inclination - `head_inclination`

**Unit:** degrees; shoulder-to-ear orientation relative to the torso.

**Example:** a chosen interval `-5` to `10` accepts `7` but rejects `12`.
Raising the maximum to `15` admits `12`. This is not horizontal ear-to-nose
pitch or lateral head roll.

### Ear-to-shoulder clearance - `shoulder_clearance`

**Unit:** normalized clearance; smaller generally means shoulder nearer ear.

**Example:** a range `0.85` to `1.15` rejects `0.80`. Lowering the minimum
to `0.75` admits that smaller clearance. These values are not elevation
percentages or a left/right shoulder-height comparison.

### Hip offset from ankle-to-head axis - `hip_offset`

**Unit:** axis lengths; signed raw projected hip offset.

**Example:** a range `-0.02` to `0.04` rejects `0.05`. Raising the maximum
to `0.06` admits it. The fallback neutral offset is not subtracted here.
This is not pelvic rotation or an actual balance measurement.

### Torso axis tilt (pink line) - `torso_axis_tilt`

**Unit:** degrees; absolute midpoint-torso tilt from image vertical.

**Example:** a range `0` to `8` accepts a tilt magnitude of `6` and rejects
`10`. Raising the maximum to `12` admits `10`; raising the minimum to `2`
would also reject upright zero. Both shoulders and hips must be reliable.

### Trunk inclination - `trunk_inclination`

**Unit:** degrees; signed core torso orientation.

**Example:** a range `-4` to `6` rejects `-7`. Lowering the minimum to `-8`
admits it. The landmark/sign rules differ from the pink-axis magnitude;
do not copy one metric's limits into the other.

### Least-bent reliably visible knee - `knee_flexion`

**Unit:** degrees; zero is straight in the projected geometry.

**Example:** with an interval `5` to `25`, a bend of `12` is inside and `3`
is outside. Raising the minimum to `15` also rejects `12`; raising the maximum
permits more bend. These examples are not instructions to force a knee angle.

Use **Use as Settings defaults** in Library or the calibration editor to copy
a profile's approved bounds/units/views. It replaces the default set and asks
before saving. It does not copy full evidence or grant camera-scoring approval.

## Reference-video knees and shoulders: the actual calculations

### Importing a reference is not approving its posture

1. **Library > Import video**, importing as a reference, creates a replayable
   `.tpr` containing video and cached model evidence.
2. **Create calibration** opens that reference's approval editor.
3. Select the correct dancer, metric, applicable view and example passages.
4. Inspect the evidence, edit acceptable bounds, **Approve metric** and save
   the `.tpc`.

The editor uses raw cached `measurement_poses`, not smoothed descriptive
percentages or interpolated display dots. A display-only cache must be
reanalyzed before it can supply verified raw calibration evidence.

The editor restores the reference's cached capture settings. Changing today's
measurement side does not rewrite the saved reference. The selected inference
model also does not replace existing cached landmarks until reanalysis.

### Knee geometry

For each usable leg, form pixel-space vectors from knee to hip and knee to ankle:

```text
u = hip - knee
v = ankle - knee
interior angle = degrees(acos(clamp(dot(u, v) / (length(u) * length(v)), -1, 1)))
knee flexion = max(0, 180 - interior angle)
```

Coordinates use the actual video's width and height, not removed camera-size
controls. An interior angle of 170 degrees gives 10 degrees of projected flexion;
180 gives zero.

Normally the least-bent usable knee wins: 20 degrees on one side and 6 on the
other gives 6, not their average. This does not identify a weight-bearing leg.
If both legs qualify but the other leg's minimum visibility is below
`knee_tiebreak_other_max_vis` (default `0.60`) and the preferred side is at least
`knee_tiebreak_min_advantage` (default `0.15`) more visible, the preferred side is
used. The configurable strict confidence mask normally removes such a weak leg first.

Required hip/knee/ankle points must pass the approval confidence/presence mask.
The helper also checks `knee_visibility_floor` (default `0.35`) and requires both
leg segments to be at least `knee_min_segment_px` (default `10` pixels). If neither
leg qualifies, the sample is missing, not zero degrees. The unsigned image angle
is not a 3D diagnosis of joint locking or hyperextension.

### Shoulder geometry

For each usable ear/shoulder pair:

```text
vertical gap = (shoulder.y - ear.y) * image height
raw clearance C = smallest usable vertical gap / reference scale S
```

Image y increases downward. A shoulder rising toward its ear reduces the gap.
The smallest usable gap selects the more elevated-looking side, not an average
and not the difference between the two shoulder heights.

The reference scale is chosen as follows (all multipliers and pixel floors are
configurable under **Head & shoulders** in Settings):

1. For each usable ear/hip pair with vertical separation greater than
   `shoulder_min_ear_hip_px` (default `20` px), calculate
   `shoulder_ear_hip_scale_factor * vertical ear-to-hip separation` (default `0.26`).
   Prefer the largest candidate.
2. Without such a candidate, take the largest available fallback:
   - `shoulder_nose_ear_scale_factor * two-dimensional nose-to-paired-ear pixel distance` (default `0.68`);
   - `shoulder_ear_ear_scale_factor * horizontal distance between ears` (default `0.65`);
   - `shoulder_width_scale_factor * horizontal distance between shoulders` (default `0.28`).
3. If the selected scale is smaller than `shoulder_min_ref_scale_px` (default `8` px),
   the result is unavailable.

The helper checks `shoulder_visibility_floor` (default `0.50`); approval first
applies the configurable confidence/presence mask and image-boundary checks.

At ear-to-hip separation 200 pixels, S is `0.26 * 200 = 52` pixels.
A selected shoulder/ear gap of 52 pixels gives C = 1.0; a gap of 41.6 gives 0.8.
Normalization reduces simple camera-distance effects, but anatomy, head
motion, perspective and model errors still matter.

### Why fallback analysis can show a percentage instead

Fallback shoulder coaching converts clearance C to:

```text
B = max(shoulder_baseline_clamp_floor, effective neutral clearance)   # default floor 0.20
elevation percent = 100 * (B - C) / B
```

B = 1.05 and C = 0.90 gives approximately 14.3%. C = B gives zero;
C greater than B can give a negative percentage. This does not measure muscle
contraction.

Current Library movement import starts from configured neutral clearance.
It collects up to `shoulder_adaptive_max_samples` (default `600`) raw clearances
between `shoulder_adaptive_min_clearance` (`0.65`) and `shoulder_adaptive_max_clearance`
(`1.50`). After at least `shoulder_adaptive_min_samples` (`15`) qualifying samples,
the baseline is the greater of its starting value and those samples'
`shoulder_adaptive_percentile` (`75th`) percentile. A fixed raw gap can therefore
acquire a different descriptive percentage as that baseline develops.

For the fallback shoulder warning, recovery uses
`max(shoulder_recovery_min_pct, shoulder_recovery_ratio * shoulder_raise_threshold)`
percent: `max(4.0, 0.65 * 10.0) = 6.5%` at defaults.
The fallback knee recovery boundary is `closed_embrace_min_knee_bend + knee_recovery_margin`:
`8.0 + 2.5 = 10.5` degrees at defaults.

Applicable descriptive knee/shoulder filters use Overlay smoothing when nonzero,
otherwise Measurement smoothing, clamped to at least `min_filter_time_constant`
(default `0.0001` seconds). A gap above Max gap resets those filters. These raw
descriptive helpers use their configured visibility floors; a displayed descriptive
number does not prove that the stricter approval checks passed.

**Approved knee/shoulder ranges do not use that adaptive percentage or
descriptive smoothing as their raw calibration samples.** After approval,
range evaluation applies general measurement smoothing and event timing.

### From raw examples to approved bounds

The editor summarizes the selected dancer's eligible samples within the chosen
passages. Lost identity, missing landmarks and wrong views do not become zeros.
Side/front eligibility uses the per-metric view and Max view width; `all` skips
that split, not missing evidence.

**Suggest middle 80% of examples** proposes the configured
`calibration_suggest_lower_pct` (`10th`) and `calibration_suggest_upper_pct`
(`90th`) percentiles, not the observed extremes or a universal technique standard.
At least `calibration_min_usable_samples` (default `2`) usable samples and a positive
reliable time interval are needed. Reliable duration excludes invalid intervals
and uses the gap policy saved with the reference.

Inspect low/high examples and explicitly approve the acceptable minimum and
maximum. Standing and walking distributions can differ. These measurements
remain two-dimensional projections, not force, partner pressure or medical
assessments.

## Applying, saving and reanalyzing

| Action | What it does | Practical example |
| --- | --- | --- |
| Apply & Save settings.toml | Confirms overwrite, saves valid edits and then applies them. | Read the displayed destination; it is not automatically Data folder. |
| Export TOML... | Applies valid edits and writes current editable settings to your chosen file. | Export a named video-analysis preset. Removed legacy-only values are omitted. |
| Import TOML... | Validates and loads a chosen configuration. | Load that preset, inspect visible values, then reanalyze where required. |
| Reset Defaults | Restores built-in configuration and disables default ranges. | Reset experimental edits without deleting Library archives or saved files. |

If saving fails, the current configuration remains unchanged. Read errors rather
than assuming success. Startup automatically loads `settings.toml` from the
process working directory; an explicit `--config` and CLI overrides take
precedence. The factory `configs/default.toml` is not overwritten.

Changing the inference model requires new analysis for new landmarks.
Changing cached descriptive processing also requires regeneration.
Approved-range Review comparisons can be rebuilt from raw cached evidence
without downloading/running a different pose model.

## Practical video-analysis recipes

### Brief excursions should not confirm events

```toml
entry_duration = 2.0
clear_duration = 0.75
```

A one-second qualifying outside-range interval will not confirm at this entry
duration. Bounds and raw joints are unchanged.

### Fill only short drawing gaps

```toml
replay_joint_interpolation = "linear"
replay_joint_max_gap = 0.5
```

A bracketed 0.4-second gap can be drawn; a 0.8-second gap is not filled.
Neither supplies valid missing scoring samples.

### Use a pink-axis default in Live Practice

```toml
[calibration_defaults.torso_axis_tilt]
lower = 0.0
upper = 8.0
unit = "degrees"
view = "all"
```

This example enables only that default metric for Live Practice. With no selected
profile, other disabled default metrics stay unavailable. Review requires a
selected `.tpc` instead. Both shoulders and hips must remain reliable.

## Further reading and implementation

- [Calibration guide](CALIBRATION_BEGINNERS_GUIDE.md).
- [Tracking-engine guide](TRACKING_ENGINE_BEGINNERS_GUIDE.md).
- [General manual](TANGO_POSTURE_GUIDE.md).
- [Architecture](ARCHITECTURE.md).
- [Editable groups, defaults and export](../src/tango_posture/config.py).
- [Qt Settings controls](../src/tango_posture/desktop_qt/screens/settings.py).
- [Model selection/cache resolution](../src/tango_posture/models.py).
- [Geometry and ankle-lift tempo gate](../src/tango_posture/geometry.py).
- [Approval measurements](../src/tango_posture/calibration/measurements.py).
- [Raw reference evidence](../src/tango_posture/calibration/evidence.py).
- [Passage statistics](../src/tango_posture/calibration/statistics.py).
- [Approval editor](../src/tango_posture/desktop_qt/calibration/editor.py).
- [Offline pipeline and fallback tempo](../src/tango_posture/references/pipeline.py).
- [Fallback coaching](../src/tango_posture/references/coach.py).
