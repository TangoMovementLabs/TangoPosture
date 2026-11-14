# TangoPosture Settings: every parameter explained for beginners

**A tab-by-tab guide to what each setting does, how to adjust it, and what changes.**

Open **Settings** in the desktop navigation. This guide follows the current
tab order and covers all 71 registered settings fields plus the eight
**Calibration defaults** rows.

The default values below are the application's built-in configuration defaults.
An imported settings file, packaged startup configuration, or selected reference
can supply different effective values. Read your visible settings and status
messages instead of assuming that an older screenshot shows today's defaults.

Examples demonstrate software behaviour. They are not medical limits,
validated tango standards, or prescriptions for different body types.

## Contents

- [Before changing anything](#before-changing-anything)
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
- [Apply, save, export, import and reset](#apply-save-export-import-and-reset)
- [Practical adjustment recipes](#practical-adjustment-recipes)
- [Invalid values and misleading adjustments](#invalid-values-and-misleading-adjustments)
- [Quick checklist and further reading](#quick-checklist-and-further-reading)

## Before changing anything

### The most important rule: which limits are active?

| Current situation | Limits used for posture comparison |
| --- | --- |
| A calibration profile is selected | Its approved ranges. Missing/unapproved metrics do not fall back to older posture settings. |
| No profile is selected, but at least one Calibration defaults row is enabled | The enabled default ranges. Disabled metrics remain unavailable, and older posture thresholds are replaced. |
| No profile is selected and no default ranges are enabled | Older Settings-based descriptive/warning controls where observations are available. Formal reference scoring still requires a reference. |

For example, changing **Head down angle limit** does not override a selected
approved `head_pitch` range. Change the approved/default range instead.

Camera, quality, smoothing and general event timing are separate from the source
of the posture limits. They still matter when using approved ranges.

### How to make a controlled change

1. Export your current Settings if you want a backup.
2. Identify the actual problem: incorrect dots, missing evidence, noisy numbers,
   delayed alerts, wrong limits, or missing recordings are different problems.
3. Change one setting.
4. Click **Apply Settings** and read the result.
5. Restart a live preview when changing its camera/model/processing setup:
   a running worker has its own configuration.
6. Reanalyze video when changing the analysis model or settings whose results
   are already stored in an analysis cache.
7. Compare the same passage before and after.
8. Save the successful configuration explicitly.

Widening limits cannot repair a hip dot on clothing. Increasing interpolation
cannot produce reliable measurements of a hidden hip.

### Understand the units

- **Pixels:** image dimensions or distances, not centimetres.
- **Seconds:** durations and smoothing time constants.
- **Milliseconds:** 1000 milliseconds equal one second.
- **Degrees:** angles. Negative values can be meaningful.
- **Ratios:** distances divided by a body/image reference length.
- **Percentages:** use `10` for ten percent only in a percentage field; a ratio
  of ten percent is `0.10`.
- **BPM:** beats per minute.
- **Tokens:** language-model text units, not camera frames.

There is no built-in height, weight or body-build preset that derives validated
posture limits. Improve framing and use individual reference evidence.

## 1. Camera & view

### Camera device index - `camera`

**Default:** `0`.

Selects which camera the computer opens. It is a device number, not quality.

- Try another available index if the wrong camera opens.
- Increasing from `0` to `1` does not improve tracking: it selects another device.
- A nonexistent or unavailable device produces a camera error.
- Check operating-system camera permission if none opens.

Close the preview before changing cameras. A checked reference may specify its
own camera source; the practice worker loads that reference's capture settings.

### Image width (pixels) - `width`

**Default:** `640`; must be positive.

Requests the live-camera image width.

- Increase, for example to `1280`, for more potential detail if supported.
- Decrease for less image data and potentially lower processing load.
- More pixels do not guarantee better landmark estimates or faster inference.
- The camera can negotiate a different actual size.

This does not resize an imported video's original resolution. Changing the
actual capture geometry can require a new/checkable live reference.

### Image height (pixels) - `height`

**Default:** `480`; must be positive.

Requests live-camera height, with the same detail/load trade-off as width.

Use a resolution the device supports, such as `640 x 480` or `1280 x 720`.
Increasing height does not replace moving the camera when the feet are cropped.
Check the resulting image rather than assuming the request was honoured.

### Measurement side - `side`

**Default:** `left`; choices: `left`, `right`.

The dancer's anatomical side preferred by side-dependent measurements.

- Choose the side that is appropriate and reliably visible.
- This is not the screen's left/right side.
- Changing it can change which ear/shoulder/hip chain is used.
- Some helpers can use the other usable side; the pink axis still requires
  both shoulders and both hips.

Keep side consistent between reference evidence and comparison footage.

### Facing direction - `facing`

**Default:** `right`; choices: `left`, `right`.

Direction faced in the original, unmirrored camera image. It defines forward/
posterior directions for some signed measurements.

Wrong facing can reverse interpretation of head-forward or hip-offset values.
Correct this by inspecting the original view, not by toggling Mirror.
Existing evidence may need consistent reanalysis/recalibration.

### Camera setup name - `capture_profile`

**Default:** `default`.

A descriptive identifier for the arrangement, such as `room-a-left-side`.
It is not a camera adjustment or an automatically detected pose.

Use distinct names for genuinely different setups. Changing the name affects
reference compatibility; it does not fix perspective, roll or bad landmarks.
Keep the physical setup unchanged after its camera check.

### Mirror - `mirror`

**Default:** disabled (`false`).

Mirrors the live displayed image when that display path applies it.
Saved measurement coordinates retain the original orientation.

- Enable if a mirror-style preview is easier to follow.
- Disable to view the camera's original orientation.
- Do not change facing/side merely to compensate for screen mirroring.

Do not assume every replay/export path uses this live-display switch.
Mirroring is a drawing transformation, not recalibration.

### Allow all angles - `allow_all_angles`

**Default:** disabled (`false`).

Enables the core all-angle measurement path. It can use shoulder/hip midpoints
or usable alternative sides during additional views.

- Enable deliberately for the corresponding measurement workflow.
- Disable for the stricter selected side-view path.
- Enabling does not invent hidden joints or validate all camera perspectives.
- Approved/default ranges still apply their own `side`, `front` or `all` view.

It is not an automatic switch that makes every 360-degree movement measurable.

## 2. Files

### Live Practice model file (Lite) - `model_path`

**Base default:** `models/pose_landmarker_lite.task`.
Packaged startup normally supplies its bundled Lite location.

Use **Browse...** to select the exact supported Lite model if necessary.
The app verifies the file's checksum.

- Changing the path changes which file is loaded, not a sensitivity value.
- An arbitrary `.task` file or renamed Full/Heavy file is not accepted as Lite.
- Live Practice uses Lite and does not automatically download a missing model.

For supported offline variants, use the next selector.

### Offline analysis model - `analysis_model`

**Default:** `lite`; choices: `lite`, `full`, `heavy`.

Controls offline analysis through **Library** and
**Review > Analyze posture & movement**.

- Lite is the default and matches the live model.
- Full/Heavy are alternatives with greater processing cost.
- None guarantees visibility through occlusion or better results for every dancer.
- Existing local supported weights are verified and reused before downloading.
- Missing offline weights download with progress and are cached.

Changing this selector does not rewrite old results: reanalyze the recording.
Calibration/model identities must match. **Live Practice > Source: Video file**
still uses the Lite practice worker, not this selectable offline pipeline.

### Data folder - `data_dir`

**Base default:** `data`; packaged app locations can differ.

Use **Browse...** to select where local data and archives are organised.

Changing folders can make the Library appear empty because it is looking in a
different place. It does not mean the old files were deleted.

This does not control the destination of the **settings.toml** save button,
which uses the process's current working directory. Read the displayed path.

## 3. Quality & calibration

Some controls here belong to the older baseline capture system. The current
guided wizard uses fixed holds and its own repeat checks; see the notices below.

### Minimum landmark confidence - `confidence`

**Default:** `0.70`; allowed `0 to 1`.

Sets acceptance for strict landmark quality checks, including visibility and
presence.

- Increase to accept fewer uncertain observations: more missing dots/metrics
  can result.
- Decrease to accept more observations: more noise or wrong positions can result.
- Try framing/lighting improvements before lowering it.
- A value of `0.70` does not mean "70% accurate".

This is an app-side filter, not a model-training control or the MediaPipe
detector's configurable minimum-detection-confidence option. Some older helpers
and the separate movement detector use other thresholds.

### Min torso - `min_torso`

**Default:** `0.12`; positive fraction of image height.

Core shoulder-to-hip geometry must be large enough.
At image height `480`, `0.12` is about `57.6 pixels`.

- Increase to require a larger core torso segment.
- Decrease to permit smaller segments, with less image detail.
- Improve framing first; reducing this does not restore pixels.

It does not govern every descriptive helper or offline movement scale check.

### Min head - `min_head`

**Default:** `0.04`; positive fraction of image height.

Minimum core shoulder-to-ear segment size.
At height `480`, `0.04` is about `19.2 pixels`.

- Increase to demand more visible segment size.
- Decrease to accept smaller segments.
- This is not a minimum nose angle or a head-height-in-centimetres control.

The ear-to-nose helper has separate internal geometry guards.

### Max view width - `max_view_width`

**Default:** `0.45`; must be positive.

Maximum projected width-to-torso ratio treated as an acceptable side view.
It also separates side/front gates for approved ranges.

- Increase to accept a broader projected ratio as side-on.
- Decrease to demand a narrower ratio.
- This is not shoulder width in centimetres or a body-weight setting.

If trustworthy side-on evidence repeatedly measures `0.48`, testing `0.50`
may change its classification. First verify the view and dot placement.
The same change also affects front-view range checks: retest both.

### View change tolerance - `view_change_tolerance`

**Default:** `0.15`; positive ratio allowance.

Used by older baseline view-width comparisons.

- Increase to permit more variation from the captured view-width range.
- Decrease to insist on closer view consistency.
- It is not an allowable head/trunk tilt in degrees.

It does not replace a new profile's per-metric applicable-view requirement.

### Calibration duration - `calibration_duration`

**Default:** `12 seconds`; positive.

Older baseline capture window.

- Increase for a longer old capture window.
- Decrease for a shorter old capture window.
- Must be at least the required valid duration.

This does **not** extend the new guided wizard's two-second settle and
three-second capture phases.

### Calibration valid duration - `calibration_valid_duration`

**Default:** `9 seconds`; positive and no greater than calibration duration.

Minimum continuous usable time required by older baseline acceptance.

- Increase to demand more reliable old baseline evidence.
- Decrease to accept less evidence.
- Not the new guided repeat check's coverage percentage.

Setting `15` while calibration duration remains `12` is invalid.

### Calibration samples - `calibration_samples`

**Default:** `60`; integer, at least `2`.

Minimum valid observations for older baseline acceptance.

- Increase to demand more samples.
- Decrease to demand fewer.
- More samples do not guarantee that the chosen posture is desirable.

The guided wizard's repeatability acceptance is not controlled by this count.

### Offset mad - `offset_mad`

**Default:** `0.03`; positive.

Maximum allowed spread around the older head-forward baseline median.
MAD means median absolute deviation.

- Increase to accept a noisier/more variable old baseline.
- Decrease to demand steadier evidence.
- Not an acceptable maximum head-forward posture value.

Do not raise it simply to accept a badly detected ear dot.

### Angle mad - `angle_mad`

**Default:** `4 degrees`; positive.

Maximum spread for older angular baseline measurements.

- Increase to accept more angular variation during old capture.
- Decrease to require steadier angular evidence.
- Not a pink-axis or trunk-alignment target angle.

The new approval editor uses chosen passages and explicit acceptable bounds.

## 4. Tracking

There are three distinct questions:

1. How much should a measured number be smoothed?
2. How long should an excursion last before confirmation?
3. What limits should it be compared with?

Changing one does not automatically change the others.

### Measurement smoothing (seconds) - `smoothing`

**Default:** `0.25 seconds`; must be positive.

Time constant for evaluated measurements.

- Increase, e.g. `0.40`, for calmer numbers but more gradual response.
- Decrease, e.g. `0.15`, for faster response with more visible noise.
- It is not a fixed video delay.
- It cannot be set to zero through valid configuration.

Approved-range events use this smoothing. It does not change raw cached joint
evidence or teach the pose model new anatomy.

### Overlay smoothing - `overlay_smoothing`

**Default:** `0.20 seconds`; zero or greater.

Used by live pose drawing and some older derived/display-related measurements.

- Increase for smoother but more trailing live lines.
- Decrease for less trailing and more jitter.
- Zero disables the pose-coordinate smoother.

Zero does not remove every other filter. Some older knee/shoulder paths use
`smoothing` as a fallback when this is zero.
Review uses same-frame raw skeleton evidence plus optional display interpolation;
do not assume this slider changes every Review skeleton.

### Entry duration - `entry_duration`

**Default:** `1.5 seconds`; positive.

How long an evaluated excursion must persist to confirm a range/baseline event.

- Increase to ignore more brief excursions and confirm later.
- Decrease to confirm sooner, potentially counting more brief noise.
- Does not move the video or skeleton sooner.

A `0.4-second` outside-range excursion does not confirm at `1.5`, but can at
`0.3`, if continuity and measurements qualify.

### Clear duration - `clear_duration`

**Default:** `0.75 seconds`; positive.

How long recovery must persist before the evaluator clears an event.

- Increase for more stable but later clearing.
- Decrease for faster clearing, potentially more on/off changes.

Approved ranges require recovery inside the interval.
Older baseline scoring instead uses its lower clear threshold.

### Max gap - `max_gap`

**Default:** `0.5 seconds`; positive.

Observation-gap/continuity guard for several measurement/tracking paths.

- Increase to tolerate larger intervals before some resets.
- Decrease to reset continuity sooner.
- Missing raw values still become UNKNOWN; a bigger gap limit does not invent them.

It is **not** the Review joint-interpolation maximum gap.

### Head forward / entry - `head_forward.entry`

**Default:** `0.08 torso lengths`; positive.

Older forward-change threshold relative to a stored baseline.

- Increase to tolerate a larger forward change before an old event.
- Decrease for more sensitivity.
- Baseline `0.03` plus entry `0.08` means a forward value above roughly `0.11`
  before timing confirmation.

Not an absolute approved `head_forward` maximum.

### Head forward / clear - `head_forward.clear`

**Default:** `0.05 torso lengths`; nonnegative and below entry.

Older remaining forward change at which clearing can begin.

- Increase to clear while farther from the baseline.
- Decrease to require closer recovery.

Do not set it equal to or greater than entry.

### Head inclination / entry - `head_inclination.entry`

**Default:** `10 degrees`; positive.

Older magnitude of angular change from the head-inclination baseline.

- Increase to tolerate a larger change.
- Decrease to detect smaller sustained changes.

Head inclination is torso-relative shoulder-to-ear direction, not the
ear-to-nose horizontal head-pitch metric.

### Head inclination / clear - `head_inclination.clear`

**Default:** `6 degrees`; nonnegative and below entry.

- Increase to allow earlier clearing farther from the old baseline.
- Decrease to require a closer return.

Does not override an approved head-inclination minimum/maximum interval.

### Trunk inclination / entry - `trunk_inclination.entry`

**Default:** `8 degrees`; positive.

Older magnitude of trunk-angle change from its baseline.

- Increase to tolerate larger angular changes.
- Decrease to detect smaller changes.

This is not the pink-axis magnitude limit.

### Trunk inclination / clear - `trunk_inclination.clear`

**Default:** `5 degrees`; nonnegative and below entry.

- Increase to allow clearing at a larger remaining old trunk change.
- Decrease to require closer recovery.

The three older entry/clear pairs do not override new approved/default ranges.

## 5. Head & shoulders

These older warning controls are used only in the older Settings-based mode,
not as substitutes for a selected approved range/default-range set.

### Head neutral offset - `head_neutral_offset`

**Default:** `-12 degrees`.

Subtracted from raw ear-to-nose pitch in the older horizontal head cue.

At raw pitch `-12`:

- Offset `-12` gives corrected `0`.
- Increase the offset to `-8`: corrected angle becomes `-4`.
- Decrease the offset to `-16`: corrected angle becomes `+4`.

This shifts a cue; it does not level the camera.
Approved `head_pitch` uses the raw angle, without this subtraction.

### Head down duration - `head_down_duration`

**Default:** `3 seconds`; positive.

Duration required for the older downward-head warning.

- Increase to require longer persistence.
- Decrease to warn sooner.
- Does not change the warning angle or general approved-range entry duration.

### Head down angle limit (degrees) - `head_down_angle`

**Default:** `-15 degrees`; must stay negative.

Boundary for the older corrected ear-to-nose cue.

- Change to `-10`: warn after a smaller downward excursion.
- Change to `-20`: allow more downward excursion before this boundary.
- More negative is not "higher sensitivity": watch the sign.

With an approved head-pitch profile, change its native limits instead.

### Shoulder tension duration - `shoulder_tension_duration`

**Default:** `1 second`; positive.

Duration before the older sustained shoulder banner.

- Increase to avoid more brief banners.
- Decrease to confirm sooner.

It does not change measured ear-to-shoulder clearance.

### Shoulder tension limit (%) - `shoulder_raise_threshold`

**Default:** `10%`; positive.

Older computed shoulder-elevation percentage above its effective clearance
baseline.

- Increase to `15`: require a larger computed elevation percentage.
- Decrease to `6`: warn at a smaller computed elevation.

The approved shoulder metric is a raw clearance ratio, not this percentage.
This is also not a left/right shoulder-level comparison or muscle-tension sensor.

### Shoulder neutral clearance - `shoulder_neutral_clearance`

**Default:** `1.05`; positive.

Fallback baseline for the older elevation percentage:

`100 * (neutral clearance - raw clearance) / neutral clearance`.

At raw clearance `0.95`:

- Baseline `1.00` gives `5%`.
- Baseline `1.10` gives approximately `13.6%`.

Increasing the baseline increases computed elevation at fixed raw clearance;
decreasing reduces it. The effective old baseline can also adapt from observed
clearances or come from a saved older calibration.

Do not adjust this to compensate for an incorrectly placed shoulder dot.

## 6. Embrace & knees

Role/embrace labels are heuristics based on observed arm geometry.
They are not identity recognition or measurement of partner pressure.

### Role assign ms - `role_assign_ms`

**Default:** `300 milliseconds`; zero or greater.

Time an observed arm pattern must persist before a role is assigned.

- Increase for slower, more stable assignment.
- Decrease for faster assignment with more possible label changes.
- `300 ms` equals `0.3 seconds`.

This does not control person identity tracking.

### Role hold ms - `role_hold_ms`

**Default:** `5000 milliseconds`; zero or greater.

How long a confirmed role can remain after the expected pattern stops appearing.

- Increase to preserve the label through longer turns/occlusions.
- Decrease to release it sooner.

The retained label is not new evidence that the arm pattern stayed visible.

### Embrace closed threshold - `embrace_closed_threshold`

**Default:** `0.72`; greater than zero and no greater than one.

Boundary on a heuristic openness score.

- Increase: a fixed score is more likely to be classified CLOSED.
- Decrease: a fixed score is more likely to be OPEN.

It is not physical distance in metres. Wrong arm dots can change the score;
fix the evidence before changing the boundary.

### Embrace hysteresis seconds - `embrace_hysteresis_seconds`

**Default:** `1 second`; zero or greater.

Required persistence for a candidate OPEN/CLOSED switch, alongside internal
smoothing and a small hysteresis band.

- Increase to switch more slowly/stably.
- Decrease to switch sooner.

There is no independent GUI slider for the internal score band.

### Closed embrace min knee bend - `closed_embrace_min_knee_bend`

**Default:** `8 degrees`; zero or greater.

Minimum projected knee flexion for the older straight-knee cue.

- Increase to `12`: more flexion is required to be above the old boundary.
- Decrease to `5`: a straighter-looking leg is allowed.

The older banner is gated by CLOSED embrace. An approved `knee_flexion` range
is a separate minimum/maximum comparison, not automatically that embrace-only
rule. Do not force bending to satisfy an arbitrary example.

### Knee straight duration - `knee_straight_duration`

**Default:** `1 second`; positive.

Required persistence for the older straight-knee banner.

- Increase to delay/avoid more brief banners.
- Decrease to confirm sooner.

Does not change the projected angle or an approved range's general timing.

## 7. Torso & hip offset

These settings describe the older ankle-to-head-line **hip-offset** warning.
They are not the pink shoulder-midpoint-to-hip-midpoint angle controls.

### Axis curve entry - `axis_curve_entry`

**Default:** `0.06`; positive ratio.

Older positive posterior hip-deviation entry threshold.

- Increase to require more deviation.
- Decrease to detect smaller deviation.
- `0.06` is six percent of the estimated ankle-to-head axis length, not six degrees.

Approved `hip_offset` instead uses raw signed minimum/maximum limits.

### Axis curve clear - `axis_curve_clear`

**Default:** `0.035`; zero or greater, and below entry.

Smoothed old hip deviation must fall below this before clearing can begin.

- Increase to clear at a larger remaining deviation.
- Decrease to require a smaller remaining deviation.

Keep it below entry to preserve hysteresis.

### Axis curve duration - `axis_curve_duration`

**Default:** `1.5 seconds`; positive.

- Increase to require longer sustained old hip deviation.
- Decrease to confirm sooner.

It is not the duration for every approved metric; those use general Tracking
entry duration.

### Axis curve clear duration - `axis_curve_clear_duration`

**Default:** `0.75 seconds`; positive.

- Increase for longer recovery before the old warning clears.
- Decrease for faster clearing.

Does not alter the numeric entry/clear boundaries.

### Axis neutral offset - `axis_neutral_offset`

**Default:** `0.0`.

Subtracted from raw offset for the older corrected deviation.
At raw `0.05`:

- Neutral `0.00`: corrected `0.05`.
- Increase neutral to `0.02`: corrected `0.03`.
- Decrease neutral to `-0.01`: corrected `0.06`.

Approved raw `hip_offset` does not subtract this field.
It is not a pelvic-rotation or camera-levelling control.

The older TOML `axis_curve` threshold object is a compatibility alias, not a
second independent GUI metric. Prefer the visible entry/clear controls.

## 8. Slow tango

These controls govern an older ankle-height/consecutive-step heuristic.
They do not configure the separate offline footprint placement detector.

### Slow tango max bpm - `slow_tango_max_bpm`

**Default:** `115 BPM`; positive and no greater than `240`.

Maximum known tempo for the slow-tango lift heuristic.

- Increase to let it operate at faster known tempos.
- Decrease to restrict it to slower known tempos.

The tracker can also treat unknown tempo as active. Its caller may supply a
fallback tempo. Verify beat tracking rather than assuming music is perfectly
classified.

### Ankle high lift ratio - `ankle_high_lift_ratio`

**Default:** `0.06`; greater than zero and no greater than one.

Normalized vertical ankle separation considered high.

- Increase to tolerate larger separation.
- Decrease to classify smaller separation as high.

For estimated leg length `300 pixels`, `0.06` corresponds to `18 pixels` in this
projected ratio. This is not certified floor clearance in centimetres.

Internal start/end separation boundaries are derived from this setting, so it
can influence segmentation as well as the high threshold.

### Max consecutive high ankle steps - `max_consecutive_high_ankle_steps`

**Default:** `2`; integer at least one.

- Increase to allow more consecutive high steps before alerting.
- Decrease to alert after fewer.

A qualifying low step can reset the consecutive count.
This is not a count of all offline footprints, and hidden feet can make the
measurement unavailable.

## 9. Music

These affect music tracking and automatic recording, not landmark placement.

### Beat tracking - `beat_tracking`

**Default:** enabled (`true`).

- Enable music/beat tracking where the input workflow supports it.
- Disable if you do not need it.
- Automatic song recording depends on usable music detection, not just this
  checkbox alone.

Microphone access is relevant to live-camera music detection.

### Starting tempo (BPM) - `tango_bpm`

**Default:** `120`; greater than `20` and no greater than `240`.

Starting/fallback tempo.

- Increase for a faster starting tempo.
- Decrease for slower.
- Actual detected tempo can replace the initial/fallback value.

It can affect tempo-gated heuristics when used as their active tempo. It does
not make the camera run at more frames per second.

### Beat analysis seconds - `beat_analysis_seconds`

**Default:** `30 seconds`; positive.

Beat/phrase analysis window.

- Increase for a longer window and more time before an analysis is complete.
- Decrease for a shorter window with less evidence.

Not the camera-calibration window or an alert-duration parameter.

### Auto record start delay - `auto_record_start_delay`

**Default:** `1 second`; allowed `0 to 60`.

Continuous detected music required before an automatic recording begins.

- Increase to avoid reacting to short music-like bursts.
- Decrease to start sooner.

Requires the automatic song-recording option and appropriate source/music
conditions. It does not start every manual recording.

### Auto record stop delay - `auto_record_stop_delay`

**Default:** `3 seconds`; allowed `0 to 120`.

Continuous detected silence required before stopping an automatic recording.

- Increase to keep recording across longer pauses.
- Decrease to stop sooner, possibly splitting a song at short quiet passages.

Not a delay before closing an arbitrary live preview.

### Max recordings - `max_recordings`

**Default:** `10`; integer `1 to 1000`.

Maximum recording starts in one coach run.

- Increase to permit more starts.
- Decrease to hit the cap sooner.

It does not delete old recordings or set the total number of archives retained
in Library.

## 10. Replay movement

### Replay auto analyse movement - `replay_auto_analyse_movement`

**Default:** enabled (`true`).

An exposed configuration flag that the current runtime analysis workflows do
not consume.

Changing it currently does not turn automatic analysis on or off.
Use the explicit **Analyze posture & movement** action. Do not troubleshoot
missing results by repeatedly toggling an inactive flag.

### Replay step history seconds - `replay_step_history_seconds`

**Default:** `5 seconds`; allowed `0.5 to 30`.

How long observed step footprints remain visible as they fade in replay.

- Increase for a longer visible trail.
- Decrease for less clutter.

Does not create, delete or re-detect step events. Scrubbing changes which stored
events are visible at that media time.

### Review joint interpolation - `replay_joint_interpolation`

**Default:** `linear`; choices: `linear`, `quadratic`.

How Review fills qualifying gaps between reliable joint positions for drawing.

- Linear draws a straight path between the surrounding observations.
- Quadratic attempts a curved path using a third nearby sample.
- Quadratic falls back to linear when unsuitable.

Neither recovers the actual hidden movement with certainty.
Both are display-only, not calibration/scoring evidence.

### Maximum joint gap to fill (seconds; 0 disables) - `replay_joint_max_gap`

**Default:** `3 seconds`; zero or greater.

Maximum separation between qualifying surrounding samples for Review gap filling.

- Increase to draw through longer bracketed gaps, with more uncertainty.
- Decrease to fill fewer/shorter gaps.
- Set `0` to disable filling.

No indefinite extrapolation at clip edges. This is not the general `max_gap`
tracking/measurement control, and it does not predict live hidden joints.

## 11. Review & Ollama

Ollama is optional text interpretation of reports. It is not the pose tracker.
Changing its settings does not move skeleton dots or change numeric approvals.

### Replay audio - `replay_audio`

**Default:** disabled (`false`).

Retained configuration field, currently not used by Qt Review's playback path.
Changing it does not enable in-app audio playback.
Export the video stream for playback in an external audio-capable player.

### Ollama server address - `ollama_base_url`

**Default:** `http://127.0.0.1:11434`.

The report-interpretation server address.

- Keep a local address to use your local server.
- A different host means a different server, potentially remote.
- This is not a pose-model download location.

Report text is sent to the configured server for interpretation. Consider that
privacy boundary before using a nonlocal server.

### Ollama model - `ollama_model`

**Default:** `llama3.2`; nonempty text.

Language model used on that server.

Choose an installed/available model supported by the server.
A larger language model can need more memory and time; it is not necessarily
better commentary for your report.
This does not select MediaPipe Lite/Full/Heavy.

### Ollama timeout - `ollama_timeout`

**Default:** `120 seconds`; positive, maximum `1800`.

- Increase to wait longer for a slow server/model.
- Decrease to fail sooner.

A larger timeout does not make the server faster or restore a disconnected
server.

### Response variation - `ollama_temperature`

**Default:** `0.2`; allowed `0 to 2`.

- Increase for potentially more variable generated wording.
- Decrease for less variation.

Not a landmark confidence score. A low temperature does not guarantee factual
or medically valid interpretation.

### Model context size (tokens) - `ollama_num_ctx`

**Default:** `16384`; integer `1024 to 131072`.

Requested language-model context capacity.

- Increase to allow more possible text context, with greater resource cost.
- Decrease to reduce capacity/resource demand.

Actual server/model capabilities still matter. This is not video duration or
the number of tracked frames.

### Maximum response length (tokens) - `ollama_num_predict`

**Default:** `2048`; integer `128 to 16384`.

- Increase for potentially longer responses.
- Decrease for shorter responses and less generation work.

This is a maximum request, not a guarantee that a response will contain exactly
that many tokens.

### Maximum report length (characters) - `ollama_max_report_chars`

**Default:** `40000`; at least `1000`.

Text budget from the report supplied for interpretation.

- Increase to supply more report text.
- Decrease to supply less, potentially omitting useful detail.

It does not analyze more raw video frames. A small report budget and a very
large context window are different controls.

### Analysis focus - `ollama_instruction`

**Default:** explain supported observations and suggest three focused practice
exercises.

Additional instruction for commentary.

An example is: "Summarize observable changes and clearly identify unavailable
measurements."

Changing it affects requested text, not numeric scores, camera quality or which
joints are implemented. Do not use commentary as a medical assessment.

## 12. Calibration defaults

This tab follows the other Settings tabs. Its configuration container is
`calibration_defaults`.

Initially all eight ranges are disabled. Each row has **Use**, minimum, maximum
and view. Units are fixed by the metric.

| Control | How to adjust | Impact |
| --- | --- | --- |
| Use | Enable only a deliberately reviewed range | If at least one is enabled and no profile is selected, enabled ranges replace older posture thresholds. Missing/disabled metrics stay unavailable. |
| Minimum | Lower to permit smaller native values; raise to exclude more small values | Narrows/widens one end of the interval |
| Maximum | Raise to permit larger native values; lower to exclude more large values | Narrows/widens the other end |
| View | Choose side, front or all deliberately | Applies its comparability gate; all does not invent hidden joints or validate different cameras |

Minimum must not exceed maximum. Equal bounds can be valid configuration but
extremely sensitive to small variation.

### Head pitch (ear-to-nose) - `head_pitch`

**Unit:** degrees; raw ear-to-nose image angle.

An invented interval `-20 to -5` includes raw `-12`.

- Lower minimum to `-25`: permit more negative raw pitch.
- Raise maximum to `0`: permit higher raw pitch.

The old head neutral offset is not subtracted. Do not copy zeroed overlay
angles as if they were raw reference values.

### Head-forward offset - `head_forward`

**Unit:** torso lengths; forward component of ear displacement relative to torso.

Invented interval `-0.02 to 0.06`.

- Raise maximum to `0.10`: allow more forward displacement.
- Lower minimum: allow more displacement in the opposite direction.

Not centimetres and not the older change-from-baseline entry threshold.

### Head inclination - `head_inclination`

**Unit:** degrees; shoulder-to-ear direction relative to torso.

- Widen the signed interval to include more native angular variation.
- Narrow it to reject more variation.

Not the same as horizontal ear-to-nose pitch or head lateral roll.

### Ear-to-shoulder clearance - `shoulder_clearance`

**Unit:** normalized clearance.

Invented interval `0.95 to 1.15`.

- Lower minimum: allow smaller clearance.
- Raise maximum: allow larger clearance.

Smaller clearance generally corresponds to shoulder nearer ear.
This is neither the old elevation percentage nor equal left/right shoulder level.

### Hip offset from ankle-to-head axis - `hip_offset`

**Unit:** axis lengths; signed raw offset from the projected ankle-to-head line.

Invented interval `-0.02 to 0.04`.

- Raise maximum: allow more posterior-side raw offset.
- Lower minimum: allow more offset in the other direction.

The old axis neutral subtraction is not applied. This is not pelvic rotation.

### Torso axis tilt (pink line) - `torso_axis_tilt`

**Unit:** degrees; absolute shoulder-midpoint-to-hip-midpoint tilt from image
vertical. Requires both shoulders and both hips.

Invented interval `0 to 8`.

- Raise maximum to `10`: allow greater tilt magnitude.
- Lower maximum to `4`: a `4.5` observation is outside.
- Raise minimum to `2`: a vertical `0` observation is outside too.

Either tilt direction has the same magnitude. Keep the camera level.
The `axis_curve_*` controls do not set this range.

### Trunk inclination - `trunk_inclination`

**Unit:** degrees; signed core shoulder-to-hip orientation.

- Widen the native interval to accept more orientation variation.
- Narrow it to accept less.

Different landmark/quality/sign rules from the pink axis. Do not swap their
limits just because both use degrees.

### Least-bent reliably visible knee - `knee_flexion`

**Unit:** degrees; `180 - projected hip-knee-ankle interior angle`.

Invented interval `5 to 25`.

- Lower minimum: allow a straighter projected leg.
- Raise minimum: require more projected flexion.
- Raise maximum: allow more flexion.

The helper generally reports the least-bent usable leg, not independent
weight-bearing-leg limits. The projected angle is not a medical diagnosis.

### Copying from a profile

Use **Use as Settings defaults** in Library or the approval editor.
The operation replaces the default-range set with that profile's approved
metrics and saves all current Settings after overwrite confirmation.

It preserves limits, units and views. It does not copy the full evidence or
grant live-camera scoring approval.
Selected profiles still take precedence over these defaults.

## Apply, save, export, import and reset

### Apply Settings

Validates edits and updates the current app configuration. It does not write a
settings file. Read any error message.

### Apply & Save settings.toml

Confirms overwrite, applies valid edits, and attempts a file save.
The file path is the current working directory's `settings.toml`, not necessarily
Data folder. Inspect the confirmation destination.

If applying succeeds but the file write fails, the app can have new values that
are not saved. Check the status message.

### Export TOML...

Applies valid edits and writes a configuration at your chosen destination.
Use it for a backup or a clearly located settings file.

### Import TOML...

Loads a chosen file into the app.
It does not automatically rewrite another file.
The ordinary loader reads a file only when supplied/configured; naming a file
`settings.toml` does not guarantee automatic loading after restart.

### Reset Defaults

Restores built-in current configuration, including disabled default ranges.
It does not automatically overwrite saved files or delete Library archives.

Stop an active preview before relying on a reset for that running worker.

## Practical adjustment recipes

These are invented experiments, not universal presets.

### A. Live skeleton trails too much

First check camera/processing latency and the actual dots.
Then try a smaller overlay time constant:

```toml
overlay_smoothing = 0.08
smoothing = 0.25
```

This reduces live coordinate smoothing while leaving the general evaluator
time constant unchanged. It may make drawing noisier. Restart the preview.
It does not select a faster model or alter camera hardware latency.

### B. Measured numbers are noisy but drawing looks acceptable

```toml
smoothing = 0.40
overlay_smoothing = 0.20
```

This makes evaluated numbers respond more gradually.
It does not repair incorrectly located joints. Inspect whether genuine brief
changes you care about are now hidden by smoothing.

### C. Brief changes produce too many confirmed events

```toml
entry_duration = 2.0
clear_duration = 0.75
```

This requires longer persistence before range events confirm.
It leaves acceptable limits unchanged and does not fix tracking quality.

### D. Review drawing should fill only short gaps

```toml
replay_joint_interpolation = "linear"
replay_joint_max_gap = 0.5
```

This reduces display inference compared with the three-second default.
It does not supply missing scoring evidence. Use zero instead to disable it.

### E. A default pink-axis comparison without selecting a profile

```toml
[calibration_defaults.torso_axis_tilt]
lower = 0.0
upper = 8.0
unit = "degrees"
view = "all"
```

This invented example enables only the pink-axis default range.
With no selected profile, the other default metrics remain unavailable and old
posture thresholds do not fill them in. It is not camera-scoring permission.
Both shoulder/hip pairs must still be reliably observed.

## Invalid values and misleading adjustments

| Problem | What to do |
| --- | --- |
| Confidence outside `0 to 1` | Use an allowed value; do not interpret a number above one as extra accuracy. |
| Clear equal to/greater than entry | Lower clear or raise entry while preserving the intended hysteresis. |
| Minimum greater than maximum | Correct the order. Negative signed values are allowed where meaningful. |
| Required old calibration time exceeds capture duration | Reduce required time or increase the old window. |
| Head-down limit is zero/positive | Use a negative old limit; approved raw ranges are a separate system. |
| Entering a percent as a ratio | Convert carefully: six percent is `0.06`, not `6`. |
| Changing an inactive replay flag | Use the actual analysis/export action, not repeated toggling. |
| Adding internal pivot/step detector keys to settings TOML | They are not exposed Settings fields; unknown keys are rejected. |
| Different pose model in calibration and recording | Recreate matching evidence; do not bypass compatibility by copying limits. |
| Saving but not loading after restart | Import/configure the saved file and verify current values. |

## Quick checklist and further reading

- [ ] I know whether a profile, default ranges or older thresholds are active.
- [ ] I understand the unit and sign of the value I am editing.
- [ ] I fixed framing/dot placement before changing limits.
- [ ] I changed one parameter at a time.
- [ ] I applied the edit and read the result.
- [ ] I restarted/reanalyzed where necessary.
- [ ] I saved to a known location and know how to reload it.
- [ ] I did not interpret UNKNOWN or a smooth skeleton as reliable evidence.

Companion guides:

- [Detailed calibration guide](CALIBRATION_BEGINNERS_GUIDE.md).
- [Tracking-engine guide](TRACKING_ENGINE_BEGINNERS_GUIDE.md).
- [General manual](TANGO_POSTURE_GUIDE.md).

Implementation references:

- [Settings defaults, constraints and groups](../src/tango_posture/config.py).
- [Qt tabs, labels and Apply/Save behaviour](../src/tango_posture/desktop_qt/screens/settings.py).
- [Default-range copying and save destination](../src/tango_posture/desktop_qt/service_bridge.py).
- [Metric names and native units](../src/tango_posture/calibration/models.py).
