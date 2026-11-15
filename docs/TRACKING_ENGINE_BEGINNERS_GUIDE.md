# TangoPosture tracking engine: a detailed guide for beginners

**How a camera image becomes dots, lines, posture observations, steps and pivots.**

This guide explains the current implementation, not an ideal tracking system.
You do not need to know Python, geometry or machine learning to follow it.
Technical names are included so that you can match an explanation to Settings,
a report, or the source code.

Read the [Settings guide](SETTINGS_BEGINNERS_GUIDE.md) for every editable field,
organized by the actual desktop tabs. Read the
[calibration guide](CALIBRATION_BEGINNERS_GUIDE.md) for detailed camera/video
calibration instructions and the meaning of the approved measurements.

Examples below are invented explanations of software behaviour. They are not
medical limits, validated tango standards, or promises of tracking accuracy.

## Contents

- [1. The engine in plain English](#1-the-engine-in-plain-english)
- [2. Live practice and video analysis are different](#2-live-practice-and-video-analysis-are-different)
- [3. Models, local files and downloads](#3-models-local-files-and-downloads)
- [4. Capture, queues and timestamps](#4-capture-queues-and-timestamps)
- [5. Landmarks and coordinate systems](#5-landmarks-and-coordinate-systems)
- [6. Confidence, visibility and missing evidence](#6-confidence-visibility-and-missing-evidence)
- [7. Following a dancer rather than a pose number](#7-following-a-dancer-rather-than-a-pose-number)
- [8. Raw evidence versus a nice-looking skeleton](#8-raw-evidence-versus-a-nice-looking-skeleton)
- [9. Smoothing and its unavoidable trade-off](#9-smoothing-and-its-unavoidable-trade-off)
- [10. Review interpolation explained](#10-review-interpolation-explained)
- [11. Posture measurements and calibration precedence](#11-posture-measurements-and-calibration-precedence)
- [12. How an observation becomes a confirmed event](#12-how-an-observation-becomes-a-confirmed-event)
- [13. Role, embrace and older live trackers](#13-role-embrace-and-older-live-trackers)
- [14. The offline movement engine](#14-the-offline-movement-engine)
- [15. How a footprint becomes a step](#15-how-a-footprint-becomes-a-step)
- [16. How a turn becomes a pivot](#16-how-a-turn-becomes-a-pivot)
- [17. All internal movement parameters](#17-all-internal-movement-parameters)
- [18. Floor guides, rings and fading footprints](#18-floor-guides-rings-and-fading-footprints)
- [19. Caches, limits and background work](#19-caches-limits-and-background-work)
- [20. Performance and apparent skeleton delay](#20-performance-and-apparent-skeleton-delay)
- [21. Practical adjustment recipes](#21-practical-adjustment-recipes)
- [22. Troubleshooting by symptom](#22-troubleshooting-by-symptom)
- [23. What this engine cannot establish](#23-what-this-engine-cannot-establish)
- [24. Glossary and implementation map](#24-glossary-and-implementation-map)

## 1. The engine in plain English

Imagine several people working on the same picture:

1. A **camera operator** supplies the next image.
2. A **dot finder** estimates where body landmarks are.
3. An **evidence checker** decides whether the needed landmarks are usable.
4. A **measurer** calculates an angle or distance ratio.
5. A **timekeeper** checks whether a change lasts long enough.
6. A **movement observer** looks for settled feet and supported turns in video.
7. An **illustrator** draws skeletons, annotations, rings and footprints.
8. A **record keeper** saves observations and results.

These jobs are related but not interchangeable.

A smoother illustrator can make a skeleton look better without improving the
dot finder's evidence. A longer timekeeper delay can reduce brief alerts without
moving a single joint. A wider calibration range changes what is accepted, not
what the camera can see.

### The most useful question

Before changing a parameter, ask:

> Is my problem in the picture, the dots, the measurement, the event timing,
> the selected limits, or only the drawing?

For example:

| Symptom | Start investigating here |
| --- | --- |
| A hip dot is on a loose coat | Image/framing and pose estimation |
| Correct dots but missing pink-axis measurement | Required joints and strict quality checks |
| Correct numbers but late warning | Measurement smoothing and persistence |
| A continuous skeleton but UNKNOWN scoring | Display interpolation is not raw evidence |
| Feet move but no steps are counted | Heel/toe visibility and movement confirmation rules |
| A pivot event exists but its ring disappears | Per-frame drawing geometry |
| A changed model seems to have no effect | Existing cached analysis |

There is no single "tracking sensitivity" slider governing all these jobs.

## 2. Live practice and video analysis are different

### Live Practice

Use **Live Practice** and choose the camera source.

The live worker uses MediaPipe **Lite**. It processes current images, evaluates
available posture evidence, runs live role/embrace/lift helpers, and submits a
rendered image to the desktop.

The camera capture queue keeps the latest available frame rather than allowing
old frames to build up indefinitely. A separate latest-image display mailbox
also prevents an unlimited backlog of rendered previews.

Live drawing can be smoothed. Measurements and approved-range evidence must
remain based on the raw observations, not on the illustrator's estimates.

**Important:** choosing **Video file** as a Live Practice source still uses the
practice worker and Lite. It is not the same route as offline Review analysis.

### Offline Library and Review analysis

Use **Library > Import & Analyze Video...**, or open a replayable recording in
**Review** and use **Analyze posture & movement**.

This shared analysis route uses the model selected in **Settings > Files >
Analysis pose model**: `lite`, `full` or `heavy`.

It decodes the selected material, preserves raw observations, and reuses those
observations for the posture and movement outputs. In the shared route,
movement receives the decoded observations with `sample_stride = 1`.

Offline analysis can wait for work that would feel too slow in a camera preview.
That does not guarantee that Full or Heavy will correctly identify every joint.
Occlusion, framing and camera geometry still matter.

### Other lower-level routes

The code also contains a standalone sampled movement scanner and a movement
analysis coordinator. Their timing, sampling and model defaults are not a
description of every visible Qt analysis action.

In particular, do not assume that pressing **Analyze posture & movement**
always means "scan at 15 FPS". Section 14 explains the distinction.

### Camera calibration compatibility

Guided live camera calibration/checking uses Lite too. A calibration derived
from Full/Heavy evidence does not automatically satisfy the Lite camera-model
compatibility check.

Use matching evidence for a live camera approval. Copying numerical ranges into
Settings does not bypass camera scoring approval.

## 3. Models, local files and downloads

### What Lite, Full and Heavy mean here

They are three pose-landmarker model variants, not three unrelated tracking
programs and not three posture standards.

The app uses pinned float16, revision-1 model assets. The variants offer
different inference-cost/quality trade-offs; the application is not training
them on your recordings.

| Variant | Typical reason to choose it | Important limitation |
| --- | --- | --- |
| Lite | Lower inference cost; required live route | Can still miss or misplace landmarks |
| Full | Try a more substantial model in offline analysis | More processing work; accuracy is not guaranteed |
| Heavy | Try the largest supported variant offline | Greater resource/time demand; cannot see through occlusion |

The estimator explicitly selects the **CPU** delegate. There is no GPU
selection setting in this implementation. "Float16" describes the model asset;
it is not a promise of GPU inference or a user-visible precision slider.

### How offline model lookup works

Before downloading the selected analysis variant, the app checks known local
locations, including its primary destination, data cache, the configured model
directory, and the working directory's models folder.

This is a targeted search, not a scan of your entire disk.

An existing candidate must match the pinned checksum. A corrupt or incompatible
candidate causes an explicit verification error; the app does not silently use
an arbitrary file just because its name looks right.

If the selected offline variant is missing, the app downloads it with progress,
verifies it, caches it locally, and then uses it.

Once a valid cached asset is present, another analysis does not need to download
the same asset again.

### Live model path is a separate concern

**Settings > Files > Pose model file** (`model_path`) defaults to
`models/pose_landmarker_lite.task`.

Live Practice expects the required verified Lite asset. Its missing-model
behaviour is not the offline selected-variant auto-download workflow.

Changing **Analysis pose model** does not switch the live worker to Heavy and
does not rewrite old analysis caches. Reanalyze the recording when you want new
model evidence.

### What MediaPipe is asked to do

The estimator:

- Converts OpenCV's BGR image to RGB.
- Uses MediaPipe's VIDEO running mode.
- Calls synchronous `detect_for_video`.
- Allows up to two detected poses.
- Preserves image-space landmarks and separate world-space estimates.

The allowed pose count is not a stable dancer-ID service. Two detected poses do
not imply that the app can safely score two dancers at once.

The app's `confidence` parameter is an application-side evidence threshold.
It is not a GUI override for all MediaPipe detection/tracking options.

## 4. Capture, queues and timestamps

### Camera capture: why an old frame can be dropped

The webcam producer has a queue with room for **one** frame.

Suppose inference is processing image A while the camera produces B and C.
If B has not been consumed when C arrives, the queue can replace B with C.
This favours a more recent picture rather than making you watch a growing
backlog of old motion.

It does not make processing instantaneous. The camera, conversion, inference,
measurement, drawing and desktop still take time.

Requested `width` and `height` are not a guarantee of the actual negotiated
camera resolution. Inspect the real capture setup when judging image detail.

Camera-open, permission, disconnection and stall failures are surfaced rather
than treated as successful zero-pose observations. Capture also has stale-frame
guards, including a 0.5-second freshness guard and additional stall/permission
handling. These are not user-adjustable posture limits.

### Inference does not create an unlimited work queue

The live application uses one inference executor worker. It waits for the
current result in short timeout intervals so quit/abort can remain responsive.
It does not enqueue every camera exposure into an unbounded inference backlog.

Most other controls still have to wait for the relevant processing work to
complete. A responsive Quit button does not prove that the model itself is
fast.

### A second queue serves the display

The GUI display mailbox also has room for one latest image. An old unconsumed
preview can be replaced by a newer one.

Control messages and saved-result notifications use separate lossless queues.
Dropping an obsolete preview image is not the same as discarding a save result.

### Three clocks you should not confuse

| Clock/value | Purpose | Not the same as |
| --- | --- | --- |
| Observation timestamp | When the source observation belongs in a sequence | Physical display presentation time |
| Absolute monotonic capture time | Measuring elapsed pipeline latency | Wall-clock calendar time |
| Media time/frame index | Position in a recorded clip | Time spent analyzing it |

For a webcam, observation time is elapsed monotonic time from the capture
origin. A monotonic clock avoids jumps caused by changing the computer's
calendar clock.

For a video file, capture prefers valid presentation timestamps. If they are
missing, nonfinite or nonincreasing, it can use frame-index/FPS timing with a
continuity offset. If neither usable timing nor FPS exists, it reports an error
instead of inventing a successful timeline.

Premature termination before a declared end can also produce an explicit
capture error.

### Why detector milliseconds can differ from stored observation time

MediaPipe VIDEO mode needs increasing integer-millisecond timestamps.
The backend supplies:

```text
detector_ms = max(previous_detector_ms + 1, round(observation_seconds * 1000))
```

If two very close observations both round to 1000 ms, the second detector call
uses 1001 ms. The original observation timestamp remains in the app's
`FrameObservation`; it is not replaced by this detector adjustment.

This protects the detector API's ordering requirement. It is not a mechanism
for reconstructing missing source frames.

### Recorded raw video is not every physical exposure

The recording path writes processed captured frames. A one-frame capture queue
can skip camera exposures while inference is busy.

Do not infer "every camera exposure was analyzed" from the existence of a raw
recording. Raw means unannotated source imagery in this context, not unlimited
capture throughput.

## 5. Landmarks and coordinate systems

### What a landmark is

A landmark is a model-estimated point associated with a named body feature.
The app does not attach physical sensors to your joints.

MediaPipe returns 33 pose landmarks. TangoPosture retains these **19** names:

| Landmark names | What they represent | Main relevance |
| --- | --- | --- |
| `nose` | Estimated nose position | Ear-to-nose pitch and head geometry |
| `left_ear`, `right_ear` | Estimated ear positions | Head pitch, forward offset, head inclination, shoulder clearance |
| `left_shoulder`, `right_shoulder` | Estimated shoulder positions | Torso, pink axis, head reference, arm/embrace geometry |
| `left_elbow`, `right_elbow` | Estimated elbows | Arm patterns and embrace/role helpers |
| `left_wrist`, `right_wrist` | Estimated wrists | Arm patterns and embrace/role helpers |
| `left_hip`, `right_hip` | Estimated hip positions | Torso, pink axis, knee angles, dancer association and pivot heading |
| `left_knee`, `right_knee` | Estimated knees | Projected knee flexion and leg geometry |
| `left_ankle`, `right_ankle` | Estimated ankles | Knee geometry, axis/lift helpers and foot support geometry |
| `left_heel`, `right_heel` | Estimated heels | Full footprint position and orientation |
| `left_foot_index`, `right_foot_index` | Estimated front-of-foot points | Footprint direction, travel and settling |

Left/right refer to the model's anatomical labels, not "the left half of your
screen". Facing the camera and mirroring can make these look reversed to a
viewer.

Eyes, mouth points and finger landmarks are not retained in this pose mapping.
You cannot enable them by increasing confidence or selecting Heavy.

### Image-space coordinates

The image layer uses normalized coordinates:

- `x = 0` is the image's left edge; `x = 1` is its right edge.
- `y = 0` is the top; `y = 1` is the bottom.
- Convert to pixels by multiplying by image width/height.

At 640 by 480, `x = 0.5`, `y = 0.25` corresponds to approximately pixel
`(320, 120)`.

Normalized coordinates do not make an observation immune to camera rotation,
perspective or cropping. They simply express image position relative to image
size.

### Predicted depth and world landmarks

The backend also preserves model-estimated world poses. Some movement geometry
uses these predicted coordinates, especially the horizontal hip direction for
pivots.

World coordinates are estimates produced from an image. They are not a
calibrated room map, measured distance from the camera, a floor survey, or
motion-capture ground truth.

Image-landmark `z` is also a predicted depth-like value, not a pixel coordinate.
Do not combine image x/y and model z into a physical ruler.

### Derived points are not additional detected joints

The shoulder midpoint averages two shoulder positions. The hip midpoint
averages two hip positions. The pink torso line joins those midpoints.

Those midpoints are calculations, not separately detected spinal vertebrae.
The pink line is not a reconstruction of the curvature of your actual spine.

When its measurement needs both shoulders and both hips, a hidden hip cannot
be replaced by guessing where a midpoint ought to be.

## 6. Confidence, visibility and missing evidence

### Visibility and presence

Landmarks can include:

- **Visibility:** the model's estimate associated with whether the landmark is
  visible/observable.
- **Presence:** the model's estimate associated with the landmark being present.

They are not independently validated probabilities that a point is correctly
placed. A confidently misplaced dot is possible.

### Strict posture/calibration reliability

The strict quality helper checks:

1. x, y, visibility and presence exist and are finite.
2. x and y are within the image interval `0 to 1`.
3. visibility and presence are each at least `confidence`.
4. visibility and presence are not above 1.

The default **Settings > Quality & calibration > Confidence** is `0.70`.
Missing confidence metadata fails this strict check.

At 0.70, a point with visibility 0.85 but presence 0.60 fails. Raising the
threshold to 0.85 can reject more evidence. Lowering it to 0.55 may retain more
points but also admits less-supported ones.

Lowering confidence does not repair wrong coordinates. First inspect the
actual dots, framing, lighting and occlusion.

### Not every engine uses the same helper

Older live geometry, role helpers and offline movement have their own
policies. Some older role geometry uses visibility around 0.30; other older
knee/ankle, axis, head and shoulder helpers use different thresholds.

The offline movement detector has internal confidence defaults of 0.55 for
general movement landmarks and 0.45 for step landmarks. Its landmark helper can
accept absent confidence metadata under its own checks.

Therefore:

- A visible role label does not prove strict calibration evidence is valid.
- A foot used by movement does not automatically pass a 0.70 posture gate.
- A drawing can be available while a stricter measurement is unavailable.
- Increasing the app's `confidence` is not a universal change to every internal
  movement threshold.

### Other quality and view gates

**Settings > Quality & calibration** includes:

- `min_torso` and `min_head`: minimum usable torso and shoulder-to-ear segment sizes.
- `max_view_width`: a side-view width gate.

The older view-change tolerance and baseline duration/sample/MAD controls are
no longer exposed in Settings or new exports. Their internal configuration
fields remain loadable for older baseline workflows, not reference-range approval.

**Settings > Camera & view** includes `side`, `facing` and `allow_all_angles`.

These determine geometry/comparability in applicable routes. Allowing all
angles is not permission to invent missing landmarks or accept a different
camera without checking it.

For the pink axis, both shoulder and hip pairs remain required even when
`allow_all_angles` is true.

### UNKNOWN is useful information

UNKNOWN means the system lacks usable evidence for that measurement now.
It does not mean "good", "bad", or "the dancer definitely stopped moving".

Do not hide UNKNOWN by widening posture ranges. A wider range still needs an
observable value to compare.

## 7. Following a dancer rather than a pose number

### Pose 0 is not a permanent person

A detector can return pose 0 for one person in one frame and a different person
later. Camera position, occlusion and detection confidence can affect ordering.

Neither a pose number nor a retained role label is a biometric identity.

### Reference-editor spatial association

The calibration editor associates observations with a selected numbered dancer
starting at an anchor. It compares reliable hip-centre positions in normalized
image coordinates.

The current association rules require:

- The nearest candidate centre is strictly less than `0.15` away.
- If there is another candidate, its distance exceeds the best distance by
  strictly more than `0.03`.
- The elapsed interval does not exceed the cached evidence's `max_gap`.

Examples:

| Best candidate distance | Second candidate distance | Result |
| --- | --- | --- |
| 0.04 | 0.10 | Sufficiently close and separated |
| 0.04 | 0.06 | Ambiguous: difference 0.02 is too small |
| 0.16 | No second candidate | Too far |

The distance is an image-coordinate distance, not 15 centimetres.

If the track is lost, ambiguous, or separated by an excessive gap, it ends.
There is no silent reseeding to whichever person next appears.

An explicit editor restart can anchor a new segment; the user confirms that
the evidence still belongs to the intended dancer.

Changing current Settings `max_gap` does not retroactively rewrite the gap
stored with cached evidence.

### One visible person is not proof of continuity

If dancer A leaves and dancer B enters, the fact that only one pose is now
visible is not proof that it is still dancer A.

The live/range core and offline movement require exactly one pose for their
relevant single-person processing. The backend's capacity for two poses is
helpful for detecting ambiguity, not a guarantee of two-person scoring.

Offline movement also uses image-jump and scale-change guards. Those are
continuity heuristics, not identity recognition.

## 8. Raw evidence versus a nice-looking skeleton

### Three pose layers

| Layer | Purpose | Can it establish approved comparison evidence? |
| --- | --- | --- |
| Raw `measurement_poses` | What the detector supplied for that frame | Yes, if required joints, quality and comparability pass |
| Historical display `poses` | Smoothed display layer stored by applicable analysis paths | No substitute for raw evidence |
| Derived `display_poses` | Review interpolation/display layer | No |

This separation is essential. Suppose an ankle is hidden for one second:

- The raw evidence can say "ankle unavailable".
- The illustrator can draw a plausible path between earlier and later positions.
- The measurement must still say "unavailable" wherever raw evidence is missing.

A plausible path is not an observation of the ankle.

### Live drawing and scoring

Live overlay points can use display smoothing. Approved measurement/scoring
uses the raw observations and applicable measurement/event filtering.

Consequently, a live pink line and an instantaneous numeric value can differ
slightly in their apparent response: the picture may be coordinate-smoothed
while the raw measurement comes from the current frame.

### Review drawing and scoring

Review uses same-frame raw skeleton evidence plus qualifying display-only joint
interpolation, rather than letting an older display-smoothed skeleton trail
behind the recorded body.

Raw annotations, comparisons and scores remain separate from gap-filled dots.
A cache that has only display poses is not sufficient approved comparison
evidence.

### An overlay video has already-painted pixels

A recorded overlay stream contains annotations baked into its video pixels.
Changing Settings cannot erase those old pixels.

Use the raw stream and current dynamic overlays when you want to inspect new
drawing/settings behaviour. Reanalysis is needed for changes to stored
analysis results.

## 9. Smoothing and its unavoidable trade-off

### Think of a spring, not a delay timer

Smoothing gradually moves a displayed/evaluated value toward a new observation.
It is not simply "wait 0.25 seconds and then show the exact old image".

The general exponential rule is:

```text
alpha = 1 - exp(-elapsed_seconds / time_constant)
new_smoothed = old_smoothed + alpha * (raw - old_smoothed)
```

A larger time constant means less response to each new sample.
A smaller time constant means quicker response and less noise suppression.

### A numeric example

Start at 0; receive raw 10 after 0.1 seconds:

| Time constant | First smoothed result, approximately | Interpretation |
| --- | --- | --- |
| 0.10 seconds | 6.321 | Fast response |
| 0.25 seconds | 3.297 | Slower response |
| 0.50 seconds | 1.813 | Stronger smoothing |

The next observations continue the movement toward the raw value.
These are mathematical examples, not measured camera latency.

### Measurement smoothing - `smoothing`

Find it in **Settings > Tracking**. Default: `0.25 seconds`, positive.

It filters applicable numeric evaluations and is also used by the older
hip-offset tracker.

- Increase it to reduce rapid numerical variation.
- Decrease it to respond faster.
- Too much can obscure brief real changes.
- Too little can expose jitter.

You cannot set this field to zero. A very small positive value is a valid
near-unsmoothed experiment, but usually is not the first fix for noisy dots.

### Overlay smoothing - `overlay_smoothing`

Also in **Settings > Tracking**. Default: `0.20 seconds`.

This is the live display-coordinate smoother, not the approved-range limit.

- Increase for a calmer skeleton, with more apparent trailing.
- Decrease for a more immediate, potentially shakier drawing.
- Set zero to disable this smoothing.

Changing it does not change the model variant, camera buffering or inference
time. Review's same-frame/raw-plus-interpolation drawing is a separate route.

### Exactly what the live PoseSmoother can hold

When enabled, its short hold duration is:

```text
hold_seconds = min(0.15, max_gap * 0.5)
```

With the default `max_gap = 0.5`, that is 0.15 seconds.

However, the smoother iterates the landmark names supplied in the current
pose. It does **not** insert an entirely absent landmark name from memory.
It can briefly hold a recent previous point when a currently supplied point
has invalid coordinates.

Confidence smoothing also softens dips; confidence decays using a slightly
longer time constant than it rises. This is display stabilisation, not recovery
of an occluded joint's true position.

Long gaps or backwards timing reset the applicable smoothing history.
Multiperson input is returned unsmoothed.

The smoother's output does not provide reconstructed z. It must not be reused
as 3D measurement evidence.

### Several internal smoothers also exist

Role/embrace/lift/movement paths do not all use the Settings time constant.
Examples include embrace score smoothing at 0.15 seconds, legacy ankle-lift
smoothing at 0.08 seconds, and offline foot/heading smoothing at 0.10 seconds.

Changing `overlay_smoothing` does not change those independent algorithms.

## 10. Review interpolation explained

### Where to configure it

Open **Settings > Replay movement**:

- `replay_joint_interpolation`: `linear` by default; also `quadratic`.
- `replay_joint_max_gap`: `3 seconds` by default; zero disables filling.

These settings affect qualifying Review drawing gaps. They do not make live
inference predict unseen joints and do not supply posture-scoring evidence.

### Linear: connect two known positions

Suppose a wrist is reliable at time 1.0 at x = 0.30 and at time 1.4 at x = 0.50.
The missing midpoint at time 1.2 can be drawn at x = 0.40.

The same proportion is applied to y. This assumes a straight path at constant
interpolated speed between the observations.

Advantages: simple, predictable and unlikely to overshoot a curved fit.
Limitation: the real wrist could have curved, paused, or moved out and back.

### Quadratic: try a curve

Quadratic interpolation uses a third nearby reliable sample when available.
This can draw a curved trajectory through the surrounding observations.

It needs more useful context and can be misleading when the true motion is
not well described by that curve.

The implementation falls back to linear if the third sample is unavailable or
the quadratic result is unsuitable, including an out-of-frame estimate.

"Quadratic selected" does not mean every filled point uses a quadratic fit.

### What makes a gap eligible

The gap needs reliable bracketing samples for the joint, compatible timing and
the permitted maximum separation.

Segments break on multiperson evidence or invalid/nonincreasing timing.
The code can fill qualifying empty single-person gaps, but it does not
extrapolate indefinitely before the first or after the last observation.

The gap limit measures separation of the surrounding usable samples, not the
number of missing pixels or a posture event's duration.

### Synthetic confidence is not detector confidence

Filled display landmarks have visibility/presence set to 1.0 so they can be
drawn. That is an implementation signal for illustration, not proof that the
detector observed them with perfect confidence.

They have no reconstructed z. Raw `measurement_poses` are untouched.

### Three seconds can be visually helpful and very uncertain

A three-second interval is a long time for a tango wrist, knee or foot.
Interpolation may hide disappearance nicely while inventing a path that the
body never followed.

For cautious visual review, try 0.3 to 0.5 seconds, or zero when inspecting
exact detector failures. Use the raw evidence to decide what was observed.

This gap control is separate from:

- General evaluator `max_gap`, default 0.5 seconds.
- Internal offline movement tracking gap, default 0.35 seconds.
- Internal pivot-display interpolation gap, default 0.22 seconds.

Changing one does not automatically change the others.

## 11. Posture measurements and calibration precedence

### The eight approved metric types

| Metric | Native measurement | Important evidence requirement/distinction |
| --- | --- | --- |
| `head_pitch` - Head pitch (ear-to-nose) | Raw projected ear-to-nose angle, degrees | Not the old zeroed overlay angle |
| `head_forward` - Head-forward offset | Ear displacement relative to torso, torso lengths | Not centimetres |
| `head_inclination` - Head inclination | Head/torso angular relationship, degrees | Different from ear-to-nose horizontal pitch |
| `shoulder_clearance` - Ear-to-shoulder clearance | Normalized clearance | Not measured muscle tension or equal shoulder level |
| `hip_offset` - Hip offset from ankle-to-head axis | Signed projected offset, axis lengths | Not pelvic rotation or true balance |
| `torso_axis_tilt` - Torso axis tilt (pink line) | Absolute midpoint-torso tilt from image vertical, degrees | Both shoulders and both hips must be usable |
| `trunk_inclination` - Trunk inclination | Core projected shoulder-to-hip orientation, degrees | Not interchangeable with pink-axis tilt |
| `knee_flexion` - Least-bent reliably visible knee | 180 minus projected hip-knee-ankle angle, degrees | Reports the least-bent usable knee, not force/support |

The pink metric measures tilt **magnitude**. Equal leftward and rightward tilts
can both be 7 degrees. It is not a signed cyclic heading.

The app retains separate legacy angular/baseline behaviour for applicable older
measurements. Do not assume that all angle columns share the same sign, zero,
wrapping or landmarks.

### Which limits win?

1. **Selected approved calibration profile:** use its approved ranges.
   Missing/unapproved metrics stay unavailable; old Settings limits do not fill
   them in.
2. **No selected profile, but enabled Calibration defaults:** use those enabled
   ranges. Disabled metrics remain unavailable.
3. **Neither:** applicable older Settings-based controls can provide descriptive
   cues where their required observations exist.

Thus, changing `head_down_angle`, `shoulder_raise_threshold`,
`closed_embrace_min_knee_bend`, or `axis_curve_entry` does not override a
selected approved profile.

For pink-axis limits, edit the approved/default `torso_axis_tilt` range, not the
older hip-offset `axis_curve_*` controls.

### Formal live scoring has another gate

A reference is required for formal live reference scoring, and only metrics
with `camera_check_passed` approval enter that range-scoring session.

Video-derived profiles are normally comparison-only until explicitly checked
and approved for the camera setup using **Library > Check camera**.

Copying a profile using **Use as Settings defaults** preserves range bounds,
units and views. It does not copy complete evidence or grant camera approval.

### "Shoulder height" is easy to misread

The implemented approved metric is ear-to-shoulder clearance.
It is not a direct independent comparison of left and right shoulder heights
and not a measurement of shoulder muscle tension.

Similarly, the pink line is not a plumb-line sensor for weight-bearing balance.
These distinctions matter when explaining an alert to a beginner.

## 12. How an observation becomes a confirmed event

### The generic evaluator has four states

| State | Plain-English meaning |
| --- | --- |
| WITHIN_REFERENCE | Available filtered evidence is not currently a confirmed deviation |
| PENDING | Outside evidence has begun, but has not persisted long enough |
| DEVIATION | Outside evidence persisted long enough to confirm an event |
| UNKNOWN | Usable measurement evidence is unavailable |

A clearing event can remain DEVIATION temporarily after its value returns
inside, because the clear duration has not finished.

### Approved range versus older entry/clear thresholds

For an approved range, the evaluator compares the smoothed measurement to an
inclusive lower/upper interval.

With limits 0 to 8:

- 5 is inside; delta 0.
- 10 is outside above; delta +2.
- -1 is outside below; delta -1.
- Exactly 8 is inside.

An outside value starts pending. Persistence uses `entry_duration`.
An inside value starts clearing. Persistence uses `clear_duration`.

Set the native bounds in the selected profile or Calibration defaults.
The old per-metric baseline entry/clear pairs are no longer Settings controls;
do not confuse them with the current persistence durations.

### Timing settings

In **Settings > Tracking**:

- `entry_duration = 1.5`: how long the applicable outside condition must persist
  before confirmation.
- `clear_duration = 0.75`: how long the applicable clear condition must persist
  before closing an event.
- `max_gap = 0.5`: excessive gaps reset applicable tracking/filter continuity.

Increase entry duration for fewer brief confirmed events but later feedback.
Decrease it for quicker confirmation but more sensitivity to brief changes.

Increase clear duration to require a longer return before clearing.
Decrease it for quicker clearing but potentially more event fragmentation.

These are not the separate older head/shoulder/knee warning durations.

### A complete example timeline

For illustration only, use:

```toml
smoothing = 0.0001
entry_duration = 0.6
clear_duration = 0.3
max_gap = 0.5
```

Give this evaluator approved bounds 0 to 8:

| Time, seconds | Raw value | State after update | Why |
| --- | --- | --- | --- |
| 0.0 | 5 | WITHIN_REFERENCE | Inside |
| 0.2 | 10 | PENDING | Outside starts |
| 0.4 | 10 | PENDING | Only 0.2 seconds outside |
| 0.6 | 10 | PENDING | Only 0.4 seconds outside |
| 0.8 | 10 | DEVIATION | 0.6 seconds outside |
| 1.0 | 7 | DEVIATION | Inside, clearing begins |
| 1.2 | 7 | DEVIATION | Only 0.2 seconds clearing |
| 1.4 | 7 | WITHIN_REFERENCE | Clear duration has passed |

The event's start is the beginning of pending, 0.2, not the later confirmation
time 0.8. Reports can therefore identify the beginning of a sustained change
even though confirmation arrived later.

With a larger smoothing constant, the filtered values would cross the interval
at different times. Timing and smoothing are independent contributors.

### Missing evidence is not a grace-period score

A missing/nonfinite raw value makes this evaluator UNKNOWN immediately and
closes pending/active event continuity as tracking lost.

`max_gap` does not mean "keep scoring a guessed value for half a second".
A sufficiently long gap also resets the filter history. A brief missing value
does not necessarily erase the numeric filter history immediately, but it
still does not produce a valid score for that missing observation.

Timestamps must be finite and strictly increasing; invalid ordering raises an
error. Repeated timestamps are not harmless extra evidence.

## 13. Role, embrace and older live trackers

These helpers have different state machines. Do not assume they all have the
generic evaluator's immediate UNKNOWN behaviour.

### Role: a held arm-pattern label

The older role helper recognizes arm patterns associated with Leader/Follower.
It is not proof of the dancers' actual roles or intentions.

In **Settings > Embrace & knees**:

- `role_assign_ms = 300`: a candidate pattern must persist before assignment
  from NONE.
- `role_hold_ms = 5000`: a confirmed label is retained after the last matching
  pattern for the configured hold interval.

A different pattern does not immediately replace a label while its hold is
active.

Shorter assignment reacts faster and can fluctuate more.
Longer assignment needs more stable evidence.
Shorter hold loses labels sooner during occlusion/turns.
Longer hold preserves labels longer, including periods when the pattern is no
longer observed.

A retained label is history, not proof of current visible joint evidence.

### Embrace: a smoothed score with hysteresis

The older embrace helper smooths its score with an internal 0.15-second time
constant and uses:

- `embrace_closed_threshold = 0.72`.
- An internal band of approximately plus/minus 0.025 around that threshold.
- `embrace_hysteresis_seconds = 1.0` for state-switch persistence.

The score measures openness: compact/wrapped geometry has a lower score.
The band helps avoid switching OPEN/CLOSED on tiny score fluctuations.
A larger closed threshold makes a fixed score more likely to be classified
CLOSED; a smaller threshold makes OPEN more likely. A longer hysteresis
requires a more sustained switch.

Missing score can retain a state or use the initial CLOSED fallback.
Therefore a CLOSED label is not a guarantee of freshly measured embrace
geometry at every frame.
Missing pose or an inactive role instead resets this tracker to NONE.

### Older head/shoulder/knee cues

Their limits and durations are described individually in the Settings guide:

| Group | Relevant controls | What to remember |
| --- | --- | --- |
| Head & shoulders | `head_neutral_offset`, `head_down_angle`, `head_down_duration` | Older neutral-adjusted head cue is not approved raw head pitch |
| Head & shoulders | `shoulder_neutral_clearance`, `shoulder_raise_threshold`, `shoulder_tension_duration` | Clearance/elevation heuristic is not muscle tension |
| Embrace & knees | `closed_embrace_min_knee_bend`, `knee_straight_duration` | Older closed-embrace cue is not an independent clinical knee assessment |

They do not replace missing metrics when approved/default ranges are active.
Camera/quality and relevant independent role settings are separate concerns.

### Older hip offset: the misleading name `axis_curve`

`AxisCurveTracker` is the legacy hip-offset tracker, not the approved pink
torso-axis-tilt metric.

In **Settings > Torso & hip offset**:

| Field | Default | Effect |
| --- | --- | --- |
| `axis_curve_entry` | 0.06 | Older positive posterior offset deviation needed to enter |
| `axis_curve_clear` | 0.035 | Lower clear threshold; must remain below entry |
| `axis_curve_duration` | 1.5 seconds | Entry persistence |
| `axis_curve_clear_duration` | 0.75 seconds | Clear persistence |
| `axis_neutral_offset` | 0 | Older baseline subtraction |

It uses applicable numeric smoothing. Missing raw geometry yields UNKNOWN/no
alert; a long gap resets tracking.

The compatibility `axis_curve` object in configuration is not another
independent Settings tab or another pink-axis control.

### Slow-tango ankle lift is not offline step detection

The live `AnkleStepTracker` evaluates projected ankle separation/lift and a
consecutive-high-lift count.

In **Settings > Slow tango**:

- `slow_tango_max_bpm = 115`.
- `ankle_high_lift_ratio = 0.06`.
- `max_consecutive_high_ankle_steps = 2`.

This does not generate the full heel-to-toe offline footprints.
Projected separation is not physical centimetres above the floor.

Its internal smoothing is 0.08 seconds. The high ratio is clamped to at least
0.005 internally. Derived start/end thresholds are:

```text
start = max(0.012, min(0.025, high_ratio * 0.35))
end   = max(0.008, start * 0.65)
```

At high ratio 0.06, start is 0.021 and end is 0.01365.
This lower exit threshold avoids treating tiny fluctuations as many steps.

A high peak is counted once for the in-flight step, and the count/alert can
change before that step has finished. A completed low step lasting at least
0.12 seconds can reset the consecutive-high count.

Outside known slow tempo, the consecutive count resets. Unknown tempo can be
treated as active, so an absent tempo lock is not proof the tracker is inactive.

Missing information produces no new raw lift, but some retained
smoothed/count/alert state can remain. Long gaps reset in-flight/filter state
without necessarily resetting every cumulative counter.

Do not use the generic evaluator's rules to infer this helper's behaviour.

### Music and Ollama are separate engines

**Settings > Music** contains the fallback tempo used when analysis has no
soundtrack BPM. The slow-tango ceiling is a separate cutoff, not a duplicate.
The removed microphone/automatic-recording controls remain internal to retained
services. None of these moves the skeleton or changes the pose model.

**Settings > Review & Ollama** controls optional report text interpretation.
Ollama is not MediaPipe and does not supply pose landmarks, approve ranges or
recalculate step geometry.

## 14. The offline movement engine

### Its inputs

Movement analysis receives frame observations with named image/world poses,
dimensions and timing information.

It looks for a single usable pose, estimates body scale, checks continuity,
tracks each foot, estimates hip heading, and constructs step/pivot events.

The shared Library/Review analysis route reuses all decoded raw observations.
It does not need to rerun the model separately just to obtain foot/pivot input.

### The standalone sampled scanner

The standalone `scan_video_movement` route can sample a sequential decode.
Its internal target is 15 sampled FPS, using approximately:

```text
if source_fps > target_fps * 1.35:
    stride = max(1, round(source_fps / target_fps))
else:
    stride = 1
```

Examples with target 15:

| Source FPS | Stride | Approximate sampled FPS |
| --- | --- | --- |
| 30 | 2 | 15 |
| 60 | 4 | 15 |
| 24 | 2 | 12 |
| 20 | 1 | 20 |

This is not an exact resampler guaranteeing precisely 15 FPS.
It is also not the shared Qt route's `sample_stride = 1`.

### Frame-index timing

The pure movement detector uses source `frame_index / fps` timing, rather than
blindly using each observation's timestamp.

For variable-frame-rate material, this can be approximate. The standalone scan
records a timing note when presentation-delta variability crosses its test.
Shared analysis also carries source/clip timing information, and packaging
rebases source-frame keys and events into the archived clip's indices.

When comparing clips, distinguish source time from clip time.
Frame 300 of a source is not necessarily frame 300 of the selected excerpt.

### What "body scale" means

The detector forms useful projected body-length candidates, including torso
midpoint/same-side lengths and a scaled hip-to-ankle length. It uses their
median when available.

The usable scale must be at least the greater of 20 pixels or 8 percent of image
height. It smooths scale with a recent median over nine values.

This scale lets a rule like "travel 0.075 body scales" adapt to a person appearing
larger or smaller in the image.

It does not estimate your height in centimetres. Two people with the same real
height can have different pixel scales because they stand at different
distances from the camera.

### Cut and jump guards

A sudden raw-scale change greater than 0.65 relative to the recent scale
reference can be treated as a cut.

A sufficiently large hip-centre jump within the tracking interval can also
trigger a cut: more than the greater of 55 percent of image width or 3.2 body
scales.

Cuts reset foot-tracker context and separate applicable movement evidence.
Footprints before the latest cut are excluded from the current display history.

These guards help avoid connecting unrelated shots/positions. They are not
camera-motion compensation, scene understanding, or proof of person identity.

## 15. How a footprint becomes a step

### An ankle dot alone is insufficient

A full footprint needs ankle, heel and foot-index geometry.
The heel-to-front-of-foot length must be usable, approximately 0.03 to 0.70
body scales.

The foot centre is:

```text
centre = 0.45 * heel + 0.45 * foot_index + 0.10 * ankle
```

This emphasizes the foot's heel/front geometry rather than treating the ankle
as the entire foot.

An ankle-only observation can support some pivot-anchor geometry but cannot
create a complete heel-to-toe footprint.

### Each foot has its own state machine

```text
INITIAL -> ANCHORED -> MOVING -> SETTLING -> ANCHORED
```

- **INITIAL:** gather enough usable samples to establish an initial anchor.
- **ANCHORED:** the foot has a current stable image position.
- **MOVING:** evidence suggests sufficient motion away from the anchor.
- **SETTLING:** the foot has slowed and clustered around a possible new place.
- **ANCHORED again:** accept a new stable placement if confirmation rules pass.

The initial standing anchor is not automatically counted as a step.
It commonly takes two or three usable samples to establish it.
If the first samples already show movement, INITIAL can instead establish an
origin from the first sample and go directly to MOVING.

A missing foot sample produces no new placement event. A later usable sample
can continue a brief interruption, but a gap longer than the internal tracking
limit resets that foot's current tracking state.

### Motion is not the same as placement

A moving foot must settle sufficiently to be accepted.
The default settling cluster radius is 0.045 body scales, and low-speed
settling must persist for at least 0.10 seconds.

Then the confirmation rules include:

1. At least 0.25 seconds of debounce relative to relevant prior confirmation.
2. Centre travel at least 0.075 body scales.
3. Both heel and toe travel at least 0.065 scales, unless the larger
   centre-travel swivel override applies at 0.20 scales.
4. Apparent leg extension at least 0.78 of the tracked standing span.
5. Relative airborne-lift geometry no greater than 0.14 scales when the other
   foot provides usable comparison.

These rules try to reject jitter, small rotations in place and suspended-foot
positions that look superficially like landings.

They are image heuristics, not ground-contact sensors.

### A 200-pixel body-scale example

| Rule | Ratio | Equivalent image distance |
| --- | --- | --- |
| Minimum centre travel | 0.075 | 15 pixels |
| Minimum heel/toe travel | 0.065 | 13 pixels |
| Settling radius | 0.045 | 9 pixels |
| Large-travel swivel override | 0.20 | 40 pixels |
| Relative airborne-lift limit | 0.14 | 28 pixels |

These pixel examples explain scale multiplication, not recommended personal
thresholds. The same ratios yield different pixel values for different framing.

### Why a step can be confirmed later than its stored time

The event is assigned to the first settled sample of the accepted placement.
Later samples can be needed to confirm that settling lasted long enough.

Therefore "the frame where the event is stored" and "the frame where the
algorithm knew enough to accept it" are not necessarily identical.

### Common missing-step causes

- Heel/front-of-foot landmarks hidden by shoes, clothing, a partner or cropping.
- Travel too small to distinguish from jitter.
- Foot still moving rather than settled.
- Leg apparently shortened or airborne.
- A tracking gap/cut breaks continuity.
- Multiple poses make single-person evidence unavailable.

Increasing Review joint interpolation does not repair these raw detector
conditions or increase the offline step count.

## 16. How a turn becomes a pivot

### Heading comes from predicted 3D hip direction

The engine uses the directed left-to-right hip vector in the model's horizontal
X/Z plane:

```text
heading_degrees = degrees(atan2(right_hip.z - left_hip.z,
                               right_hip.x - left_hip.x))
```

It prefers the separate world pose. If no world-pose dictionary is available,
it can use predicted z from image landmarks under its own checks.

It does not invent yaw simply from the screen angle of a 2D shoulder/hip line.

### Why some visible turns have no usable heading

Heading can be rejected when:

- Required hip coordinates or depth are missing/nonfinite.
- The horizontal X/Z hip span is too small.
- The usable 2D hip span is inadequate.
- Substantial 2D/3D X projections disagree in sign.
- Single-person continuity is unavailable.

A narrow side view can be difficult because the hips project close together.
Heavy does not guarantee adequate depth/hip geometry.

### Unwrapping avoids a false giant turn

Angles near the wrap boundary need special handling.
Moving from +179 to -179 degrees can mean a small 2-degree continuation, not a
358-degree reversal.

The engine unwraps heading and smooths it with a default internal 0.10-second
time constant before evaluating turn windows.

### A pivot needs turn evidence and an anchor

Default entry evidence includes:

- A window between 0.35 and 0.80 seconds.
- At least 20 degrees net turn.
- Mean entry turn rate at least 30 degrees per second.
- Limited motion in the opposite direction.
- Sufficient stable foot-anchor evidence.

The window maximum is an **entry-evidence window**, not an 0.80-second maximum
duration for an entire pivot.

The foot anchor needs at least the greater of two samples or 70 percent of the
window's samples, and maximum displacement around its median no greater than
0.18 body scales.

An image-stable foot can be labelled left, right or both for support.
This is not measurement of which foot bears your weight.

### Exit, trimming and reversal

The detector has separate default exit and trim rates of 14 and 12 degrees per
second. It can split reversals and trim low-motion boundaries.

Final events must retain sufficient duration/rotation. The minimum final
duration is related to the entry minimum, approximately 0.8 times that minimum,
not a universal independently exposed Settings field.

The final rotation can use appropriate raw/smoothed endpoint evidence so
smoothing does not needlessly shrink a supported turn's reported magnitude.

### Direction and mirrors

Positive heading movement is labelled clockwise in the app's documented
camera X-to-Z convention; negative is anticlockwise.

This is a coordinate convention, not a compass heading or a universal
observer-independent description. Facing and viewpoint matter.

Where the movement renderer is explicitly given a mirror option, it reflects
the arrow once without rewriting the stored event direction.

Current Qt Review does not pass a general mirror setting into this renderer.
The live-only mirror control was removed from Settings; it was not a replay-
direction switch.

## 17. All internal movement parameters

**This section is explanatory, not a supported Settings tuning menu.**

These 28 entries belong to the movement module's `DETECTOR_CONFIG`.
They are not fields in the desktop Settings tabs and must not be pasted as
unknown keys into a settings TOML file.

The Config loader rejects unknown keys. Programmatic detector overrides also
change the detector fingerprint; production validation expects the current
default fingerprint. Editing source constants is not a supported user preset.

"Increase/decrease" below describes the local rule, not a guarantee that total
counts change monotonically. Multiple gates interact.

### Sampling, quality, continuity and heading

| Internal key | Default | Purpose and adjustment effect |
| --- | --- | --- |
| `target_sample_fps` | 15.0 | Target for the standalone sampled scan. Higher usually samples more often; lower less often, subject to stride rounding. It does not force the shared Qt pipeline to sample at this rate. |
| `min_landmark_confidence` | 0.55 | General movement landmark quality threshold. Higher rejects more scored points; lower admits more, including weaker evidence. It is not the GUI posture `confidence`. |
| `step_min_landmark_confidence` | 0.45 | Step/foot landmark quality threshold. Higher can reduce usable foot coverage; lower may retain poorly supported foot geometry. |
| `frame_margin` | 0.01 | Image-edge exclusion margin. Larger rejects more near-edge points; smaller admits points closer to cropping boundaries. |
| `max_tracking_gap_s` | 0.35 | Movement continuity gap limit. Larger tolerates longer interruptions; smaller breaks tracking sooner. It is not Review joint gap filling. |
| `max_interp_gap_s` | 0.22 | Short display hip-geometry interpolation limit within pivot context. Larger can bridge more drawing gaps; smaller requires denser geometry. It does not synthesize posture evidence. |
| `heading_smooth_tau_s` | 0.10 | Heading smoothing time constant. Larger calms heading but responds more slowly; smaller exposes more immediate rotation/noise. |
| `min_3d_hip_span` | 0.035 | Minimum useful predicted horizontal 3D hip span. Larger rejects more narrow/unstable vectors; smaller admits directions more vulnerable to depth noise. |
| `min_2d_hip_span_ratio` | 0.08 | Minimum projected hip span relative to body scale. Larger needs clearer separation; smaller admits narrower projections. |

### Pivot confirmation and ending

| Internal key | Default | Purpose and adjustment effect |
| --- | --- | --- |
| `pivot_window_min_s` | 0.35 | Shortest entry evidence window. Larger requires longer context; smaller can accept briefer evidence. It interacts with speed, angle and final duration checks. |
| `pivot_window_max_s` | 0.80 | Longest entry evidence window. Larger permits a longer entry search; smaller restricts it. Not the maximum duration of a completed pivot. |
| `pivot_min_net_deg` | 20.0 | Minimum net supported rotation. Larger rejects more small turns; smaller can admit more minor motion/noise. |
| `pivot_entry_rate_deg_s` | 30.0 | Required entry mean turn rate. Larger needs faster evidence; smaller permits slower entry if the other gates pass. |
| `pivot_exit_rate_deg_s` | 14.0 | Low-rate ending criterion. Larger can end a turn at a higher remaining rate; smaller can retain slower rotation longer. Other ending/reversal rules still apply. |
| `pivot_trim_rate_deg_s` | 12.0 | Rate used to trim low-motion boundaries. Larger generally removes more low-rate boundary motion; smaller retains more. |
| `pivot_max_opposite_fraction` | 0.22 | Allowed fraction of opposite-direction motion in applicable windows. Larger tolerates more reversals/noise; smaller demands more consistently directed rotation. |
| `pivot_foot_anchor_radius_scale` | 0.18 | Maximum anchor displacement around its median, in body scales. Larger accepts more foot drift; smaller requires steadier projected support. |

### Foot motion and settling

| Internal key | Default | Purpose and adjustment effect |
| --- | --- | --- |
| `step_smooth_tau_s` | 0.10 | Foot-position smoothing time constant. Larger reduces jitter but can delay motion/settling; smaller responds faster and exposes noise. |
| `step_move_speed_scale_s` | 0.30 | Body-scale-per-second speed threshold used by one movement-entry branch, with an internal 0.75 multiplier. Larger makes that speed branch harder; smaller easier. An alternate centre-travel branch can still start motion independently. |
| `step_settle_speed_scale_s` | 0.35 | Low-speed settling criterion, in body scales per second. Larger permits faster apparent motion as settling; smaller needs slower motion. |
| `step_settle_radius_scale` | 0.045 | Allowed settled-position cluster radius. Larger accepts a looser cluster; smaller requires a tighter one. |
| `step_settle_min_duration_s` | 0.10 | Required settled duration. Larger needs a longer stable placement; smaller confirms sooner with less persistence. |
| `step_min_center_travel_scale` | 0.075 | Minimum centre travel from the anchor for applicable step rules. Larger rejects more small placements; smaller can admit jitter-sized travel. |
| `step_min_both_ends_travel_scale` | 0.065 | Minimum travel of both heel and toe unless the swivel override applies. Larger rejects more small foot rotations; smaller admits more. |
| `step_swivel_override_travel_scale` | 0.20 | Large centre travel allowing the both-ends rule to be bypassed. Larger makes the override harder; smaller makes it easier. |
| `step_min_leg_extension_ratio` | 0.78 | Minimum apparent leg extension relative to tracked standing span. Larger requires a more extended-looking leg; smaller admits more shortened/flexed geometry. |
| `step_max_airborne_lift_scale` | 0.14 | Maximum relative projected lift for the applicable placement check. Larger is more permissive of apparent raised-foot geometry; smaller rejects more. Not measured floor clearance. |
| `step_debounce_s` | 0.25 | Separation/debounce between relevant confirmations. Larger suppresses nearby repeat events; smaller permits closer confirmations. |

### What a normal user should change instead

Use supported controls and evidence:

1. Improve foot/hip framing.
2. Choose the appropriate offline model.
3. Reanalyze the same raw clip.
4. Inspect confidence/coverage and actual event frames.
5. Change footprint display history only if the problem is visual clutter.

If an internal movement heuristic consistently misclassifies a clearly visible
movement, retain a reproducible clip/result for debugging. Do not disguise it
with unrelated posture limits.

## 18. Floor guides, rings and fading footprints

### The floor grid is decorative

The neutral floor guide uses an image-space stage with a horizon at about
66 percent of image height and a central vanishing point.

It does not estimate a calibrated floor plane, homography, physical distance,
room coordinates or true shoe contact.

A perspective-looking drawing is not proof of perspective calibration.

### Pivot rings need geometry at the displayed frame

The renderer needs an active pivot event **and** usable per-frame hip geometry
to place the ring around the current hip midpoint.

If the ring disappears, the event may still exist. Its drawing geometry may
temporarily be unavailable.

Short hip-display interpolation can bridge qualifying gaps up to the internal
0.22-second default within pivot context. This is independent of the
three-second Review skeleton interpolation setting.

The ring is centred on projected body geometry, not a measured floor support
axis.

### Footprints preserve detected placement geometry

Footprints use the stored heel/toe position and orientation at accepted steps.
Their history fades using **media time**, not how long analysis took.

In **Settings > Replay movement**, `replay_step_history_seconds` defaults to 5
and permits 0.5 to 30 seconds.

- Increase it to see more old placements, with more clutter.
- Decrease it to focus on recent placements.
- It changes visibility/history, not how many steps were detected.

The fade has a short initial ramp and subsequent decay. A step's exact landing
frame receives a nonzero initial ramp rather than needing to wait a full
history interval.

### Review checkboxes are presentation controls

In the Review inspector, **Overlay layers** include:

- **Skeleton**.
- **Torso axis** (pink line).
- **Pivot rings**.
- **Step footprints**.
- **Floor lines only**.

Hiding a layer does not erase its analysis events or alter event counts.
Choosing floor-only presentation does not calibrate a floor.

## 19. Caches, limits and background work

### Why results are cached

Pose inference and movement analysis can be expensive. Caches let Review open
stored evidence/events without rerunning the entire video every time.

This also means a Settings change may not affect an already-stored analysis.
Use the analysis action again when you want new model/processing evidence.

### What a movement payload records

Its metadata includes:

- Schema version, currently 1.
- Detector version, currently `1.0`.
- Detector configuration fingerprint and model hash.
- Segment/source member and source hash.
- Width, height, FPS, frame count and duration.
- Timing basis (`frame_index_over_fps`), timing note and sample stride.
- Sampled/valid-pose/pivot/left-foot/right-foot coverage counts.
- Cut frames, pivot events, hip display frames and step events.

Coverage tells you how much usable input the engine had. It is not an accuracy
percentage. Quality summaries based on visibility are not validated
probabilities of correct artistic or physical interpretation.

### Validation is explicit, but caller-dependent

The movement validator checks supported schema/detector versions, current
default fingerprint, timing basis, dimensions/counts, event ordering and
geometry. It can also check supplied expected identities such as the source
or model.

Different loaders supply different expected arguments. Do not assume every
cached UI path checks every optional identity against today's Settings.

Changing the selected model is not equivalent to recomputing stored results.

### Resource limits are route-specific

| Limit | Scope |
| --- | --- |
| 120-second selected clip | Current Library video import UI |
| 2 GiB video size | Standalone movement scanner limit |
| 45-minute duration | Standalone movement scanner limit |
| 200,000 frames | Movement limits/checks in applicable paths |
| 5,000 pivots | Movement event-list limit |
| 20,000 steps | Movement event-list limit |
| 200,000 hip frames | Movement display-data limit |

The module also defines a 10 MiB analysis-JSON size constant.
Its definition alone is not proof that every reader enforces that exact cap;
do not treat it as a universal guarantee for every archive/cache path.

The 120-second import UI limit and 45-minute standalone scanner limit do not
describe the same operation.

### The lower-level movement coordinator

The module's coordinator offers:

- One background worker.
- A bounded queue of 32 pending tasks.
- Deduplication based on task kind, resolved path and segment.
- Cancellation and explicit error results.
- A computed result plus a warning when applicable persistence cannot succeed
  in a read-only location.

The current visible Qt shared-analysis route is not simply this coordinator
under a different button label. Do not assume its exact queue governs all
Library work.

### Worker errors are not successful empty analyses

Qt practice workers emit errors, close storage and finish explicitly.
An error notification should be investigated rather than interpreted as
"there were zero deviations".

A result that could not be saved needs its warning addressed before you assume
it will be available after reopening the app.

## 20. Performance and apparent skeleton delay

### Several delays add together

```text
camera exposure/buffering
  + frame conversion and transfer
  + model inference
  + measurement/event processing
  + overlay drawing
  + GUI/display scheduling
  + visible smoothing response
```

No single Settings value removes all of them.

The latest-frame queues reduce accumulating backlog, but cannot eliminate
camera or inference latency.

### What the performance numbers mean

| Report value | Meaning | Important limitation |
| --- | --- | --- |
| `processed_fps` | Reciprocal of mean inference duration | Not a guarantee of end-to-end camera/display FPS |
| `capture_to_overlay_ms_median` | Median measured capture-to-overlay-submission latency | Stops before physical monitor presentation |
| `capture_to_overlay_ms_p95` | 95th-percentile latency for that measured scope | Not the worst imaginable delay |
| `dropped_frames` | Capture-queue drops | Does not count every possible display/recording/source loss |
| `latency_scope` | Whether timing is capture-to-overlay or headless decode/processing | Required context for interpreting latency |

If mean inference time is 0.04 seconds, inference-derived throughput is
approximately 25 FPS. Drawing, audio, decoding and desktop overhead can still
make total throughput lower.

Offline inference throughput is not the same as "a ten-minute video will always
finish in this exact number of seconds".

### Practical order for reducing live trailing

1. Confirm you are looking at live/dynamic drawing, not baked overlay pixels.
2. Check whether the raw dots are correct.
3. Keep Live Practice on its supported Lite route.
4. Try lower `overlay_smoothing`, for example 0.08.
5. Restart the preview after processing/setup changes.
6. Try a supported lower camera resolution if inference/capture load is high.
7. Avoid assuming more interpolation will accelerate inference.

Lower smoothing can trade delay for jitter. Lower resolution can trade detail
for processing cost. Measure the same movement rather than choosing the
smallest numbers blindly.

### Hardware/body-size examples without invented presets

| Situation | Useful adjustment | Why |
| --- | --- | --- |
| Tall dancer loses head or shoes at image edges | Move camera back or frame more vertically; verify all required landmarks | Missing pixels cannot be restored by changing posture limits |
| Short dancer occupies little of the image | Move closer/reframe while keeping full required body parts | More usable projected detail can help quality gates |
| Loose clothing obscures hips/knees | Improve lighting/view and inspect raw dots | Body silhouette is not reliable joint evidence by itself |
| Larger build or partner overlap hides landmarks | Use a view/solo reference that exposes required points | Widening angles does not make hidden joints observable |
| Thin limbs at a distance produce shaky foot points | Improve foot detail and compare offline models on the same clip | Do not assume one universal body-type confidence value |

There is no supported tall/short/thin/large preset that derives accurate
posture limits from body build.

## 21. Practical adjustment recipes

Make one experiment at a time and keep the same raw clip/movement for comparison.
The following TOML fragments are examples of valid configuration, not complete
universal recommended settings files.

### Recipe A: a more immediate live skeleton

```toml
overlay_smoothing = 0.08
smoothing = 0.25
```

Expected effect: less live coordinate smoothing without changing the general
numeric smoothing constant.

Possible cost: more visible jitter. It does not change CPU delegate, camera
buffering or the live model.

### Recipe B: quieter numerical observations

```toml
smoothing = 0.40
overlay_smoothing = 0.20
```

Expected effect: applicable numeric values respond more gradually.
Possible cost: brief real changes become less prominent/later.

If a dot is incorrectly positioned, repair the input before using this recipe.

### Recipe C: require a longer sustained deviation

```toml
entry_duration = 2.0
clear_duration = 0.75
```

Expected effect: slower confirmation of applicable range events.
It does not widen the approved range or change separate older warning timers.

### Recipe D: cautious Review gap filling

```toml
replay_joint_interpolation = "linear"
replay_joint_max_gap = 0.4
```

Expected effect: fill only shorter qualifying display gaps.
Missing raw measurement evidence remains missing.

To inspect detector failures without synthetic joints:

```toml
replay_joint_max_gap = 0.0
```

### Recipe E: offline model experiment

```toml
analysis_model = "full"
```

Apply the setting and reanalyze the same raw video.
Compare hip/foot landmark coverage and representative event frames.
Do not assume that more events means more correct events.

Lite remains the live practice/check model. A Full-derived calibration needs
compatible handling before live camera scoring.

### Recipe F: reduce footprint clutter only

```toml
replay_step_history_seconds = 2.0
```

Expected effect: shorter visible history.
It does not reduce the actual stored step count or change foot confirmation.

### Save and reload deliberately

**Apply & Save settings.toml** confirms overwrite and saves to the process
working directory, not necessarily `data_dir`. The new configuration is applied
only after the atomic save succeeds.

Use **Export TOML...** for a chosen backup location.
The app loads `settings.toml` from its working directory on startup.
Use **Import TOML...** or `--config` to load a different file.

A running preview worker has its own configuration. Restart it when changing
processing/setup settings, and reanalyze stored evidence when appropriate.

## 22. Troubleshooting by symptom

### "Lines disappear and then return"

Possible causes include missing endpoints, out-of-frame points, quality failure
or a single-person continuity break.

For Review, try short linear interpolation. If a line joins two endpoints,
both endpoints need drawable positions.

For live practice, remember that its short smoother hold is not the same as
Review's bracketed interpolation and does not insert completely absent joint
names.

Do not report a filled line as recovered measurement evidence.

### "The skeleton is smooth but the coach says UNKNOWN"

This can be correct. The display may contain smoothed/interpolated points while
the strict raw measurement lacks required evidence.

Inspect raw landmarks and the metric's joint requirements. For pink tilt,
check both shoulders and both hips.

### "The skeleton is half a second behind me"

Separate smoothing from measured pipeline latency.
Try a smaller overlay time constant and inspect performance scope.
Keep the current supported live model/capture setup and restart the preview.

Review should use same-frame raw evidence plus display gap filling. Old baked
overlay pixels do not become current just because Settings changed.

### "I raised a limit but the warning did not change"

Check the active limit source. In Review, posture alerts require a selected
approved profile. In Live Practice, enabled default ranges also replace older
Settings thresholds when no profile is selected.

Change the relevant approved/default range rather than unrelated baseline
controls. Inspect whether a cached result or running worker still uses older
configuration.

### "I see two dancers; why is the measurement unavailable?"

Detector capacity for two poses is not safe two-person scoring.
Single-person routes intentionally reject ambiguous evidence.

Use the calibration editor's explicit dancer selection/association for
supported reference evidence, and inspect where the track ends.

### "A slow pivot is missing"

Check raw hip depth/direction, span, anchor evidence, turn rate, angle and
continuity. A visible torso rotation does not guarantee those gates pass.

There is no Settings pivot-rate slider. Compare models/visibility on the same
clip and retain a reproducible example of a supported turn that is missed.

### "A pivot exists but its ring is gone"

An event is temporal data; a ring additionally needs current hip drawing
geometry. Inspect that frame's geometry rather than assuming the event was
deleted.

### "Moving my foot did not create a footprint"

Motion alone is insufficient. Full heel/toe geometry, sufficient travel,
settling, extension and applicable lift/debounce checks must pass.

Initial standing anchors and small swivels are not automatically steps.

### "The model keeps downloading"

Check the selected variant, known destination, working directory, data path
and verification error. A changed working directory can change relative paths.

A candidate that fails its checksum is not a valid installed asset.
Do not rename arbitrary files or bypass verification.

### "I enabled automatic movement analysis but nothing happens"

`replay_auto_analyse_movement` has no runtime consumer and was removed from
Settings and new TOML exports. Older TOML files containing it still load.

Use the explicit **Analyze posture & movement** or Library import/analysis
action. Repeated toggling will not queue work.

### "Replay audio remains silent"

`replay_audio` was also removed from Settings and new TOML exports; older files
still load. Qt Review playback is video-only. Export/play the video externally
when you need an audio-capable player.

Neither replay flag is a tracking-quality parameter.

### "Changing Ollama did not improve the dots"

Ollama interprets report text. MediaPipe supplies the dots.
Text commentary is not a replacement for pose evidence, calibration approval
or movement reanalysis.

## 23. What this engine cannot establish

Do not interpret it as providing:

- Medical diagnosis or validated joint-health limits.
- Muscle tension, pain or injury risk.
- Actual force, centre of pressure or which foot carries body weight.
- Guaranteed true ground contact.
- Physical floor height or calibrated distances in centimetres.
- An anatomically exact spine or pelvic rotation from the pink line.
- Stable biometric person identification.
- Perfect visibility through clothing, occlusion or a partner.
- Recovered true hidden motion from linear/quadratic interpolation.
- Universal accurate posture thresholds for a body size/type.
- Guaranteed better results because a larger model produced more events.
- Artistic judgement of good tango.

The useful output is a reproducible description of supported projected
observations, with explicit uncertainty and missing evidence.

## 24. Glossary and implementation map

### Glossary

| Term | Beginner meaning |
| --- | --- |
| Landmark | Estimated named body point |
| Raw evidence | Detector output before display-only invention/smoothing |
| Normalized coordinate | Image coordinate expressed relative to width/height |
| World pose | Model-estimated 3D coordinates, not a room survey |
| Confidence gate | Rule deciding whether evidence is usable |
| Time constant | How gradually a smoother responds |
| Hysteresis | Different entry/exit rules or persistence to avoid rapid switching |
| Pending | A possible event not yet sustained long enough |
| Debounce | Suppression of closely repeated confirmations |
| Interpolation | Estimate between observed points |
| Extrapolation | Estimate beyond observed endpoints; not indefinite Review filling |
| Anchor | A reference position/starting observation for continuity |
| Body scale | Estimated projected body length for relative movement rules |
| Cut | Continuity break, such as a large position/scale change |
| Coverage | Amount of usable input, not validated accuracy |
| Fingerprint | Hash identifying the detector configuration |
| Baked overlay | Annotation already painted into saved video pixels |

### Source map

These links are for readers who want to verify the implementation:

| Responsibility | Source |
| --- | --- |
| Settings fields, defaults, constraints and tabs | [Configuration](../src/tango_posture/config.py), [Qt Settings](../src/tango_posture/desktop_qt/screens/settings.py) |
| Camera/file capture and timestamps | [Capture](../src/tango_posture/capture.py) |
| MediaPipe model inference | [MediaPipe backend](../src/tango_posture/pose/mediapipe_backend.py) |
| Local model verification/download support | [Models](../src/tango_posture/models.py) |
| Strict evidence reliability | [Quality](../src/tango_posture/quality.py) |
| Named observation data | [Domain](../src/tango_posture/domain.py) |
| Geometry, live smoothing, role/embrace/lift/hip helpers | [Geometry](../src/tango_posture/geometry.py) |
| Event persistence and numeric filtering | [Temporal evaluator](../src/tango_posture/temporal.py) |
| Approved metric registration | [Calibration models](../src/tango_posture/calibration/models.py) |
| Raw reference evidence and dancer association | [Calibration evidence](../src/tango_posture/calibration/evidence.py) |
| Joint gap filling | [Pose interpolation](../src/tango_posture/pose/interpolation.py) |
| Shared offline analysis | [Reference pipeline](../src/tango_posture/references/pipeline.py) |
| Raw/display comparison layers | [Reference coaching](../src/tango_posture/references/coach.py) |
| Feet, pivots, display geometry, caches and coordinator | [Movement](../src/tango_posture/movement.py) |
| Live orchestration, recording and performance | [Application](../src/tango_posture/app.py) |
| Latest rendered-image mailbox | [Practice display](../src/tango_posture/practice.py) |
| Qt live worker lifecycle | [Practice worker](../src/tango_posture/desktop_qt/practice/worker.py) |
| Review controls and dynamic drawing | [Review](../src/tango_posture/desktop_qt/screens/review.py) |
| Clip packaging/rebased frame indices | [Library archives](../src/tango_posture/desktop_qt/library/archives.py) |

### Final checklist

- [ ] I know whether I am using Live Practice or offline analysis.
- [ ] I know which model created the evidence, not just today's selected model.
- [ ] I inspected raw dots before changing thresholds.
- [ ] I distinguish a held/interpolated drawing from observed evidence.
- [ ] I know which approved/default/older limits are active.
- [ ] I understand event confirmation and clearing delays.
- [ ] I do not confuse ankle-lift alerts with offline footprints.
- [ ] I treat pivot support and floor graphics as image heuristics.
- [ ] I applied, restarted/reanalyzed and saved when needed.
- [ ] I checked warnings and missing evidence rather than hiding them.

Continue with the [tab-by-tab Settings guide](SETTINGS_BEGINNERS_GUIDE.md),
the [detailed calibration guide](CALIBRATION_BEGINNERS_GUIDE.md), or the
[general manual](TANGO_POSTURE_GUIDE.md).
