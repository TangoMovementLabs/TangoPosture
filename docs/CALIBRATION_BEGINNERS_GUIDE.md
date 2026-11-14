# TangoPosture calibration: a very detailed beginner's guide

**For live camera coaching and offline video analysis.**

This guide explains what to click, what the dots and lines mean, what each
relevant setting changes, and how to investigate an unexpected warning.
You do not need to understand programming or mathematics to use it.
Configuration keys are included so you can match this guide to Settings or a
TOML settings file.

The descriptions below follow the current Qt desktop application. Examples are
invented demonstrations of the controls, **not recommended tango technique,
medical limits, or validated settings for a particular body type**.

> The app observes estimated landmarks in a camera image. It does not measure
> muscle tension, diagnose posture, measure your weight, or establish whether
> your balance or dance technique is correct. A trustworthy teacher and your
> own comfort matter more than making a line turn green.

## Contents

1. [The five rules to read first](#1-the-five-rules-to-read-first)
2. [Where everything is in the app](#2-where-everything-is-in-the-app)
3. [The words and numbers, in plain English](#3-the-words-and-numbers-in-plain-english)
4. [Every stored landmark and what uses it](#4-every-stored-landmark-and-what-uses-it)
5. [Prepare the camera before changing thresholds](#5-prepare-the-camera-before-changing-thresholds)
6. [Live camera calibration, click by click](#6-live-camera-calibration-click-by-click)
7. [Calibration from a video, click by click](#7-calibration-from-a-video-click-by-click)
8. [Approve a metric without getting lost](#8-approve-a-metric-without-getting-lost)
9. [Head alignment: three different measurements](#9-head-alignment-three-different-measurements)
10. [Shoulder height and clearance](#10-shoulder-height-and-clearance)
11. [Axis alignment: the pink line](#11-axis-alignment-the-pink-line)
12. [Hip alignment: a different axis and a different number](#12-hip-alignment-a-different-axis-and-a-different-number)
13. [Torso and trunk inclination](#13-torso-and-trunk-inclination)
14. [Knees, arms, embrace, ankles and feet](#14-knees-arms-embrace-ankles-and-feet)
15. [Every calibration-related Settings tab](#15-every-calibration-related-settings-tab)
16. [Examples for tall, short, slender and larger builds](#16-examples-for-tall-short-slender-and-larger-builds)
17. [Worked example: eight approved default ranges](#17-worked-example-eight-approved-default-ranges)
18. [Save, reload and replace Settings safely](#18-save-reload-and-replace-settings-safely)
19. [Missing joints, interpolation and apparent delay](#19-missing-joints-interpolation-and-apparent-delay)
20. [Troubleshooting: symptom, cause, next action](#20-troubleshooting-symptom-cause-next-action)
21. [Other Settings that affect a session or video review](#21-other-settings-that-affect-a-session-or-video-review)
22. [Printable checklists and final summary](#22-printable-checklists-and-final-summary)
23. [Implementation references](#23-implementation-references)

## 1. The five rules to read first

### Rule 1: a calibration is an approved comparison, not a perfect-pose detector

Think of calibration as telling the app:

> "For this dancer, this activity and this camera view, compare the observed
> numbers with these ranges that I have reviewed and accepted."

The app cannot know that an example is desirable just because you selected it.
If you calibrate from an unsuitable example, it can faithfully compare against
an unsuitable example.

Create different profiles for different purposes. For example:

- Your own comfortable standing reference.
- Your own solo walking reference.
- A teacher's selected demonstration, used as a video comparison.
- A checked live-camera version of a reference.

Do not automatically copy someone else's limits because that person is taller,
shorter, thinner, heavier or more experienced.

### Rule 2: there are three sources of posture limits

| Situation | Where the posture limits come from | What happens to missing metrics |
| --- | --- | --- |
| A calibration profile is selected | That profile's approved ranges | Missing or unapproved metrics stay unavailable. There is no fallback to the old posture thresholds. |
| No calibration is selected, but Settings has enabled **Calibration defaults** | Those enabled minimum/maximum ranges | Disabled or missing default metrics stay unavailable. These defaults replace the old posture-threshold system. |
| No calibration is selected and no default ranges are enabled | The older Settings thresholds, such as the head-down, shoulder and knee controls | The older descriptive/warning logic is used where observations are available. Formal reference scoring still requires a reference. |

**Example:** your selected profile approves only head pitch and the pink torso
axis. Changing **Settings > Embrace & knees > Closed embrace min knee bend**
will not add a knee warning to that profile.

Camera selection, landmark quality, smoothing and timing are separate controls.
They still matter; the table above describes the source of the posture limits.

### Rule 3: video approval and live-camera approval are different

A profile made from a reference video is normally **comparison only**.
You can compare it with suitable cached footage in Review.

For live scoring, perform **Library > Check camera**, review the repeated
movement evidence, and explicitly approve each eligible metric for that camera.
The app does not grant live approval merely because a video profile exists.

Saving ranges as **Settings defaults** also does **not** transfer camera-scoring
approval.

### Rule 4: a missing measurement is not a good measurement

**UNKNOWN**, **Unavailable**, **Not observed**, or
**No approved / valid measurement** means the app cannot make that comparison.
It does not mean that the joint is correctly aligned.

You may see a smooth-looking skeleton even when the underlying measurement is
unavailable. Review interpolation improves drawing, not measurement evidence.

### Rule 5: geometry first, thresholds second

Before widening an acceptable range:

1. Check that the correct dancer is selected.
2. Check that the dots are actually on the intended body parts.
3. Check that the camera is level and the important joints are visible.
4. Check the selected side, facing direction and applicable view.
5. Check whether the reference and recording use the same model.
6. Only then decide whether the acceptable range needs changing.

Wider limits do not repair an incorrectly located shoulder or hip.

## 2. Where everything is in the app

The primary routes for this guide are **Library**, **Live Practice**, **Review**
and **Settings**. You do not need a References button in the navigation rail.

| You want to... | Where to go | What to click or inspect |
| --- | --- | --- |
| Start a new camera calibration | **Library** | **Guided camera calibration...** |
| Start it from the practice screen | **Live Practice** | Choose **Source: Camera**, enter **New profile name**, then **New Calibration**. Close an existing preview first. |
| Import a reference video | **Library** | Enable **Import as reference (.tpr)**, then **Import & Analyze Video...** |
| Make a reference from a recorded practice | **Library**, on a `.tp` row | **Promote to Reference**; choose recording and excerpt. The practice original is retained. |
| Calibrate from a reference recording | **Library**, on a `.tpr` row | **Create calibration** |
| Check a video profile against your camera | **Library**, on a profile-only `.tpc` row | **Check camera** |
| Read saved limits and approval status | **Library**, on a profile-only `.tpc` row | **Profile details** |
| Find only one archive type | **Library** | **File type:** References (`.tpr`), Calibrations (`.tpc`) or Practices (`.tp`); optionally combine with name search. |
| Use a profile during live practice | **Live Practice** | **Calibration profile:** selector; for scoring-ready profiles, confirm **Camera placement unchanged**. |
| Compare analyzed video with a profile | **Review**, right-hand frame-insights panel | **Calibration profile:** selector |
| Analyze a recording's raw video | **Review**, frame-insights panel | **Analyze posture & movement** |
| Draw the skeleton during practice | **Live Practice** | **Skeleton (V)** |
| Show the pink torso axis during practice | **Live Practice** | **Vertical axis (Y)** |
| Show synchronized skeleton/posture in Review | **Review**, frame-insights panel | **Skeleton / posture overlay** |
| Copy approved ranges into default Settings | Calibration editor | **Use approved ranges as Settings defaults...** |
| Copy an already saved profile into defaults | **Library**, profile-only `.tpc` row | **Use as Settings defaults** |
| Edit copied ranges | **Settings > Calibration defaults** | **Use**, minimum, maximum and view for each metric |
| Change video measurement side or facing | **Settings > Camera & view** | Measurement side and Facing direction |
| Choose Lite, Full or Heavy for offline analysis | **Settings > Files** | **Offline analysis model** |
| Change evidence quality checks | **Settings > Quality & calibration** | Confidence, segment size and view gates |
| Change measurement/event timing | **Settings > Tracking** | Smoothing, entry/clear durations and gap limits |
| Change older head/shoulder warnings | **Settings > Head & shoulders** | Head offset, warning angle/duration and shoulder controls |
| Change older embrace/knee warnings | **Settings > Embrace & knees** | Role, embrace and knee controls |
| Change older hip-offset warnings | **Settings > Torso & hip offset** | Axis-curve thresholds, durations and neutral offset |
| Fill short gaps in the Review skeleton | **Settings > Replay movement** | **Review joint interpolation**, **Maximum joint gap to fill** |
| Save changed Settings to disk | **Settings** | **Apply & Save settings.toml**, or **Export TOML...** |
| Load a settings file | **Settings** | **Import TOML...** |

Some buttons end with an ellipsis (`...`): that means another dialog opens.
The actual Qt display may use the single ellipsis character instead.

### The three file types

| Extension | Plain-English meaning | Contains video? |
| --- | --- | --- |
| `.tp` | A practice archive, possibly with several recordings | Yes, for recorded segments |
| `.tpr` | A replayable reference recording with evidence/provenance | Yes |
| `.tpc` | A new approved-range calibration profile | No; the approved ranges are independent of the video |

Older `.tpc` baseline archives can still exist. Do not assume an old baseline
has the new per-metric ranges or pink-axis approval. Prefer creating a new
approved profile from trustworthy reference evidence when you need those features.

### Two different ways to process a video

| Route | Model used | Intended use |
| --- | --- | --- |
| **Live Practice > Source: Camera** | Lite | Live preview/coaching and recording |
| **Live Practice > Source: Video file** | Lite, through the practice worker | Practice-style playback, not the selectable offline-analysis pipeline |
| **Library import analysis** or **Review > Analyze posture & movement** | The Lite/Full/Heavy choice in **Settings > Files** | Offline analysis and reusable video evidence |

If you specifically want Full or Heavy, use the Library/Review analysis route.

## 3. The words and numbers, in plain English

### Landmark

A landmark is an estimated dot, such as the left shoulder or right ankle.
It is not a marker attached to your actual bone. Clothes, lighting, occlusion,
camera angle and the pose model can move that estimate.

### Joint

Here, a joint usually means a body location such as shoulder, elbow, hip, knee
or ankle. The app also uses non-joint landmarks, such as nose, ear and heel.

You approve a **measurement made from several landmarks**. You do not manually
calibrate every dot's coordinates, drag a knee into place, or teach the model
where your shoulder is.

### Angle in degrees

An angle describes the direction or bend of a line.

- `0 degrees` on the pink axis means the torso line is vertical in the image.
- `0 degrees` for knee flexion means the projected leg looks straight.
- A head-pitch value can be negative.
- An angle is not a distance in centimetres.

Zero has a different meaning for different metrics. It is not a universal
"correct posture" score.

### Ratio

A ratio divides one image distance by another image distance.

If a forward head displacement is `12 pixels` and torso length is `200 pixels`:

`12 / 200 = 0.06 torso lengths`.

This helps measurements resist simple camera-distance changes. It does not
remove every effect of body proportions, perspective, clothing or rotation.

### Percentage

`0.06` as a ratio is `6%`, but not every number in the app is a percentage.

- Shoulder warning elevation uses a percentage relative to a clearance baseline.
- Approved shoulder clearance uses a raw normalized ratio.
- Hip-offset ratios can be displayed as percentages in older overlays.

Never paste a displayed `10%` as the raw number `10` into a ratio range.
In a ratio field, ten percent is `0.10`.

### Lower/minimum and upper/maximum

These describe the acceptable interval:

`minimum <= observed value <= maximum`.

For an approved pink-axis range `0 to 8 degrees`:

- `4.5` is inside.
- `9` is outside.
- `0` is inside.

For a range `2 to 8 degrees`, `0` is outside too. Setting a positive minimum
does not tell the app "ignore values closer to vertical"; it excludes them.

**Increasing the maximum** generally allows larger values.
**Decreasing the minimum** generally allows smaller values.
These changes widen the interval and usually produce fewer outside-range events.

Increasing the minimum or decreasing the maximum narrows the interval.
Changing both limits in the same direction shifts the target rather than simply
making it easier or harder.

### Seconds, milliseconds and time constants

- `1.5 seconds` is one and a half seconds.
- `300 milliseconds` is `0.3 seconds`.
- A smoothing value is a **time constant**, not a literal video delay setting.
  A larger smoothing value makes a number respond more gradually.
- An entry duration delays confirmation until an excursion persists.
  It does not delay the camera image itself.

### Confidence, visibility and presence

The pose model provides confidence-like values. The strict quality check uses
both **visibility** and **presence**, along with valid coordinates.

At **Minimum landmark confidence = 0.70**, a strict check will reject a point
whose visibility or presence is below `0.70`.

This is not a guarantee of "70% accurate", and it does not retrain the model.
It is an acceptance threshold used by the app. Some older descriptive/role
helpers have their own internal visibility thresholds, so this one setting does
not govern every old label in exactly the same way.

### Median, p10, p90 and the middle 80%

Suppose selected examples contain many pink-axis values:

- **Median:** the middle value, not an average of the highest and lowest.
- **p10:** about ten percent of reliable sample values lie below it.
- **p90:** about ninety percent lie below it.
- **Middle 80%:** the interval between p10 and p90.

**Suggest middle 80% of examples** offers a starting interval. It does not
certify the movement and does not promise that 80% of your future practice
will be acceptable.

### Samples, reliable duration and inside fraction

- **Samples:** how many reliable observations were used.
- **Selected duration:** the length of your chosen passages.
- **Reliable/valid duration:** continuous usable time within those passages.
- **Inside fraction:** the fraction of reliable sample values inside the edited
  minimum/maximum.

Ten seconds of selected video can contain only four seconds of usable evidence.
Thirty unusable frames are not equivalent to thirty measured frames.

The percentiles and inside fraction describe samples, not a formal statistical
confidence interval or necessarily the exact fraction of elapsed time.

### Entry, clear and hysteresis

The older threshold system can use a higher boundary to start a warning and a
lower boundary to clear it. This separation is called **hysteresis**: it avoids
rapid on/off flickering near a boundary.

With approved minimum/maximum ranges, the app instead checks whether the
smoothed value is outside or back inside that interval. The general
**entry_duration** and **clear_duration** still control persistence.

### Left/right and facing

- **Measurement side** is the dancer's anatomical left or right.
- **Facing direction** is the direction in the original, unmirrored image.
- **Mirror** flips the display only. It does not flip saved measurements.

Imagine somebody standing opposite you. Their left arm can appear on your
screen's right. Do not confuse those two meanings of "left".

## 4. Every stored landmark and what uses it

MediaPipe's model outputs 33 landmarks. The app currently names and retains
the following **19** for these workflows. It does not expose 33 independent
calibration controls.

The individual left/right entries below are important: hiding one side can
remove a measurement even when the other side looks fine.

| Stored landmark | What the dot approximately represents | What uses it | Relevant controls |
| --- | --- | --- | --- |
| `nose` | Nose location | Ear-to-nose head pitch; core head/view checks; back-of-head estimate for hip offset; some role/embrace orientation cues | **Camera & view**: facing, framing; **Quality & calibration**: confidence; **Head & shoulders**: older head offset/limit; approved `head_pitch` and `hip_offset` ranges |
| `left_ear` | Left ear region | Head pitch, left-side head-forward/inclination, shoulder clearance, hip-offset head endpoint | Measurement side; confidence; minimum head segment for core measurements; head/shoulder/hip approved ranges |
| `right_ear` | Right ear region | Same functions on the right or as an alternative/head-scale reference | Same controls; hiding the right ear can affect front-view scaling even when the left ear remains visible |
| `left_shoulder` | Left shoulder joint region | Torso axis midpoint, left trunk/head measurements, clearance, left arm chain, embrace | Confidence; side/view; minimum torso; approved shoulder/trunk/pink-axis ranges; role/embrace settings |
| `right_shoulder` | Right shoulder joint region | Other half of the pink-axis midpoint; right trunk/head measurements; right arm and shoulder functions | Same controls; the pink axis requires this shoulder as well as the left one |
| `left_elbow` | Left elbow | Skeleton arm line; role/embrace heuristics | Confidence for drawing; **Embrace & knees** role/embrace timing and threshold; no approved elbow-angle metric |
| `right_elbow` | Right elbow | Right arm chain; role/embrace heuristics | Same controls; no independent right-elbow range |
| `left_wrist` | Left wrist | Arm posture/role, embrace openness and skeleton | Confidence for drawing; role/embrace controls; no wrist-flexion calibration |
| `right_wrist` | Right wrist | Same on the right | Same controls; no independent wrist-angle range |
| `left_hip` | Left hip joint region | Left torso line; pink hip midpoint; hip-offset endpoint; left leg angle/length | Confidence; side/facing/view; minimum torso; approved torso, hip and knee ranges; ankle ratio controls |
| `right_hip` | Right hip joint region | Right torso line; other half of the pink midpoint; right leg | Same controls; the pink axis requires both hips |
| `left_knee` | Left knee region | Left hip-knee-ankle flexion and leg-length estimate | Confidence in strict ranges; measurement side; approved `knee_flexion`; older knee-bend limit/duration; ankle lift ratio |
| `right_knee` | Right knee region | Right leg flexion and length | Same controls; the least-bent reliable leg may determine the reported knee value |
| `left_ankle` | Left ankle region | Knee flexion, hip-offset axis endpoint, core framing check, ankle separation and steps | Confidence; full-body framing; approved hip/knee ranges; **Slow tango** controls |
| `right_ankle` | Right ankle region | Right-leg equivalent and the other ankle in lift comparison | Same controls; the lift helper needs both ankles |
| `left_heel` | Left heel estimate | Foot-related stored evidence and movement rendering/analysis where used | Confidence, shoe/floor visibility, **Replay movement** display controls; no approved heel-alignment range |
| `right_heel` | Right heel estimate | Same on the right | Same controls; not independently calibrated |
| `left_foot_index` | Left forefoot/toe-region estimate | Foot-related stored evidence and movement rendering/analysis where used | Confidence and unobstructed feet; no independent toe-angle calibration |
| `right_foot_index` | Right forefoot/toe-region estimate | Same on the right | Same controls; not independently calibrated |

### What you can approve

There are eight approved-range metrics, grouped into five steps:

| Editor step | Exact metric label | Configuration/profile key | Unit |
| --- | --- | --- | --- |
| Head | Head pitch (ear-to-nose) | `head_pitch` | `degrees` |
| Head | Head-forward offset | `head_forward` | `torso_lengths` |
| Head | Head inclination | `head_inclination` | `degrees` |
| Shoulders | Ear-to-shoulder clearance | `shoulder_clearance` | `normalized_clearance` |
| Hips | Hip offset from ankle-to-head axis | `hip_offset` | `axis_lengths` |
| Torso | Torso axis tilt (pink line) | `torso_axis_tilt` | `degrees` |
| Torso | Trunk inclination | `trunk_inclination` | `degrees` |
| Knees | Least-bent reliably visible knee | `knee_flexion` | `degrees` |

Eyes, mouth corners and finger landmarks are not named in this app's retained
mapping. There are also no approved per-elbow, wrist, heel or toe limits.
Left/right shoulder-level comparison, head yaw, head lateral roll, pelvic
rotation and support-leg-specific ranges are not part of these eight approvals.

### Skeleton colours and missing lines

In the ordinary detailed skeleton, reliable dots/segments are green and
low-confidence dots can be orange. A connecting skeleton line needs reliable
endpoints. A missing line often means an endpoint failed a check.

Green means the point passed a drawing quality check. It does not mean the
person's technique passed a medical or artistic assessment.

The pink line and its cyan dashed vertical guide are separate from the green
skeleton. Approval does not necessarily recolour that pink line. Read the
numeric comparison and event status as well.

## 5. Prepare the camera before changing thresholds

### A practical setup procedure

1. Put the camera on a stable support.
2. Keep the image level: a truly vertical room edge should look vertical.
3. Start with a view that contains your head, both hips and both feet.
4. Leave space above the head and below the feet for movement.
5. Use enough even lighting that the app can locate ears, shoulders, hips and
   knees. Avoid a bright window directly behind you.
6. Prefer clothing whose outline is easy to distinguish from the background.
7. Check the actual dots rather than assuming that a visible body part was
   correctly detected.
8. For side-dependent measurements, turn into the intended side view and set
   **Measurement side** and **Facing direction** accordingly.
9. Save a descriptive **Camera setup name**, such as `room-a-left-side`.
10. Do not move or tilt the camera between calibration and the checked practice.

No single distance in metres works for every lens, room, height and movement.
Moving farther away makes the body smaller in the image; moving closer makes it
larger but can crop the head or feet. Find a position that gives both framing
margin and usable joint detail.

### Camera height versus camera tilt

For a tall dancer, you may need to raise the support and/or move farther back.
For a short dancer, you may need to lower the support and/or move closer.

Move the support if possible rather than using a steep upward/downward tilt to
fit somebody into the frame. Perspective changes the projected geometry.
Recheck the calibration after any placement change.

### Side, front and all views

In an approval:

- **side:** use the app's side-view comparability checks.
- **front:** use the app's front-view comparability checks.
- **all:** do not apply that side/front width gate to the range.

**All** does not mean all camera positions are scientifically interchangeable.
It does not invent hidden joints. The pink axis still needs both shoulders and
both hips.

The side/front distinction uses a projected width-to-torso ratio, not an exact
measurement of your body's rotation angle. Broader shoulders and perspective
can affect it.

### A simple size example

At image height `480`:

- `min_torso = 0.12` means a core shoulder-to-hip segment needs at least about
  `57.6 pixels`.
- `min_head = 0.04` means a core shoulder-to-ear segment needs at least about
  `19.2 pixels`.

If your torso is only `40 pixels` long, changing a warning angle will not fix
that core quality failure. Improve framing or resolution first.

These are core measurement checks. Other helpers have their own geometry
requirements; they are not all controlled by `min_torso` or `min_head`.

## 6. Live camera calibration, click by click

This section describes retained camera implementations, not a current desktop
navigation workflow. Camera device/resolution/mirroring controls are no longer
in Settings; existing legacy configurations remain loadable. For the current
desktop workflow, import a `.tpr` and use **Create calibration** instead.

### Part A: get ready

The retained capture entry point must be supplied with its camera configuration
and verified Lite weights; current Settings exports are not a complete legacy
camera configuration. Device access and a one-person image are prerequisites.
The retained wizard's subsequent steps are described below for implementation
reference, not as instructions to find removed Library or Live Practice actions.

### Part B: preview and choose the activity

1. Select **Neutral posture**, **Solo walking** or **Close embrace** in the
   wizard's context selector.
2. Click **Start camera preview**.
3. Look at the estimated dots and check framing.
4. Click **Begin guided checks**.
5. The wizard gives a three-second countdown.

The context name describes the purpose of your profile. Choosing **Close
embrace** is not a measurement of partner pressure and does not make a
two-person calibration supported.

### Part C: the four holds in each step

Each of **Head, Shoulders, Hips, Torso and Knees** records four phases:

1. Neutral hold.
2. A small changed posture.
3. Return to the starting posture.
4. A fresh repetition of the changed posture.

Each phase allows **two seconds to settle, then three seconds of capture**.
That is about twenty seconds per step, excluding retries and review time.
The older `calibration_duration` setting does not change these fixed hold times.

| Step | Neutral prompt | Changed prompt | Important landmarks |
| --- | --- | --- | --- |
| Head | Look comfortably ahead | Lower your gaze slightly, without forcing your neck | Nose and visible ear; shoulder/hip for the additional head metrics |
| Shoulders | Relax your shoulders | Gently raise your shoulders | Ear and corresponding shoulder; hip/head landmarks help establish scale |
| Hips | Stand comfortably with your hips balanced | Gently shift your hips a little forward | Ear, hip and ankle on a usable side; nose helps the head endpoint |
| Torso | Stand comfortably upright | Incline your torso a little, without forcing | Both shoulders and both hips for pink-axis tilt; appropriate side view for trunk inclination |
| Knees | Keep your knees comfortably straighter, not locked | Gently soften and bend your knees | Hip, knee and ankle on a reliable leg |

These movements test whether the observed number responds repeatably. They are
not instructions to exaggerate a movement, force a joint, or seek an extreme
posture. Retry or skip if an action is uncomfortable.

### Part D: understand the repeat check

The check looks for:

- Four completed holds.
- At least `0.5 seconds` of reliable evidence in each hold.
- At least `60%` reliable coverage in each hold.
- Changes distinguishable from variation within the holds.
- Both repetitions changing in the same direction.
- A return close enough to the first neutral hold.

A passing result means **repeatable response in these observations**, not
validated dance technique or universal cross-camera accuracy.

Several measurements belong to Head and Torso. A step can allow continuation
when at least one measurement is repeatable. That does not approve every
measurement in the group.

Use:

- **Retry step** when framing or performance can be improved.
- **Skip step** when reliable evidence is not available.
- **Continue** after an eligible result.

For **Solo walking**, the wizard adds a walking passage: two seconds to settle
and about eight seconds to capture natural walking.

### Part E: approve the ranges

After capture, the app releases the camera and opens
**Review evidence & approve calibration ranges**.

For every measurement you want:

1. Select the step and metric.
2. Select the applicable view.
3. Select appropriate captured examples.
4. Inspect the chart and seek to the low/high examples.
5. Edit **Acceptable minimum** and **Acceptable maximum**.
6. For eligible camera evidence, explicitly check
   **Approve this metric for this checked camera**.
7. Click **Approve metric**.

If you cannot justify a measurement, use **Skip / remove metric**.
You must approve at least one metric to save a new profile.

Give the profile a clear name and save with **Save approved .tpc profile**.
Use a distinct name rather than expecting the app to silently replace an
existing profile with the same name.

### Part F: actually use the result

1. Open **Live Practice**.
2. Choose the new **Calibration profile**.
3. If it is scoring-ready, confirm **Camera placement unchanged** only if that
   statement is actually true.
4. Click **Start Preview**.
5. Use **Start Scoring (S)** for the formal session comparison.
6. Use **Record (R)** if you want video evidence.
7. Finish using **Stop & Save (E)**.

Only camera-approved metrics are included in the live range-scoring session.
An approved but comparison-only metric is not silently promoted to live scoring.

## 7. Calibration from a video, click by click

### Part A: choose the model before analyzing the evidence

Go to **Settings > Files > Offline analysis model**:

- **Lite:** the default; also the model used by live practice.
- **Full:** an alternative with greater processing cost.
- **Heavy:** another alternative with greater processing cost.

Full and Heavy are not guaranteed to be more accurate for a particular dancer,
and none can guarantee recovery of a joint hidden behind somebody else.
The variants use the same underlying landmark identities.

The current pinned weights are the Lite, Full and Heavy **float16 revision 1**
models listed in [the model manifest](../models/manifest.json).
Do not edit the manifest to adjust your posture. It identifies supported model
files and checksums; it is not a calibration settings file.

Before downloading, offline analysis checks its expected local locations for the
selected variant. Existing weights are verified and reused. If missing, that
variant downloads with progress and is cached. A bad checksum produces an error;
it does not silently substitute another model.

**Important:** a Lite calibration and Heavy recording are not interchangeable
just because both report angles in degrees.

### Part B: import the reference

1. Open **Library**.
2. Enable **Import as reference (.tpr)**.
3. Click **Import & Analyze Video...**.
4. Choose a local video.
5. Enter a useful **Title**.
6. Choose **Clip start (seconds)** and **Clip end (seconds)**.
7. Click **Analyze**.

The current local import dialog permits up to **120 seconds per selected clip**.
Choose a shorter, relevant excerpt rather than a long mixture of standing,
walking, turns, camera movement and occlusion.

The source video is not modified. The imported reference is a separate archive.
Local import starts in movement-review mode; this is not yet an approved
posture calibration.

If you already have a `.tp` recording, use **Promote to Reference** instead of
recording it again. Choose the appropriate recording index and excerpt.

### Part C: make sure the evidence is usable

1. Find the `.tpr` row in Library.
2. Use **Open in Review** to inspect the source/raw video.
3. If the app reports missing or display-only landmark evidence, use
   **Analyze posture & movement** on the raw/source recording.
4. If you changed the offline model, reanalyze to regenerate evidence with that
   model. Merely changing Settings does not rewrite cached results.

A baked overlay is a video with text and skeleton pixels already painted into
it. The app cannot remove old painted warnings by selecting a new calibration.
Use the raw/source stream whenever possible.

### Part D: create the calibration

1. Return to Library and click **Create calibration** on the `.tpr` row.
2. At the playhead, select the numbered dancer you intend to use.
3. Inspect enough of the passage to make sure tracking remains on that dancer.
4. Use **Select dancer / restart track here** after an excluded gap or ambiguity
   if necessary.
5. Confirm **I confirm all manually linked passages follow the same dancer**
   only after inspecting them.
6. Optionally save the selection using
   **Save dancer selection in reference (.tpr)**.
7. Review and approve the individual ranges as described in the next section.
8. Give the profile a distinct name and save the `.tpc`.

Tracking is conservative spatial association, not identity recognition.
An anchor does not prove two appearances belong to the same person.
Changing the selected dancer evidence clears earlier approvals, so select the
right evidence before doing all the range work.

### Part E: compare another video

1. Analyze the comparison recording with a model matching the calibration.
2. Open it in Review.
3. In the right-hand frame-insights panel, choose the **Calibration profile**.
4. Scrub/play the source video and read measurements, limits and states.

This selector performs a numeric comparison. **Compare reference** /
**Open Comparison...** is the separate side-by-side visual comparison.
Opening a reference beside a practice does not by itself select approved limits.

Current cached range comparison checks model identity and recorded side/facing,
then applies each metric's view/quality checks. It does not prove that different
physical camera angles or lenses are equivalent.
Live scoring has stricter capture-profile matching.

### Part F: if you want that video profile to work live

Use **Library > Check camera** on the saved `.tpc`.
This records fresh guided evidence and opens the approval editor with the
reference limits available for review.

You must explicitly approve the eligible metrics for the checked camera and
save a distinctly named checked profile.

**Model warning:** camera evidence uses Lite. A Full/Heavy reference fails the
camera/reference model-matching check. For a live-compatible profile, create
Lite reference evidence and a corresponding Lite calibration, or start a new
guided Lite camera calibration. Do not use copied Settings defaults as a way
to pretend that a model mismatch has been validated.

## 8. Approve a metric without getting lost

The editor contains a video on the left and controls on the right.
Use the following routine for each metric.

### 8.1 Choose the measurement, not just the body group

Choosing **Head** does not approve all three head measurements.
Choosing **Torso** does not approve both torso measurements.

Choose the metric in the second selector, inspect its unit, and approve it
individually.

### 8.2 Choose the view

Set **Applicable view** to the view you intend to compare later.
For example, a side-dependent hip-offset reference should not casually be used
for unrelated frontal footage.

Choose **all** only when you intentionally accept the lack of the side/front
width gate and the required landmarks are still trustworthy.

### 8.3 Select passages in the displayed timeline

In **Example passages**, enter intervals such as:

`0-3; 5-9`

This selects the first three seconds and the passage from five to nine seconds.
Use the timeline displayed by the imported reference, not an assumed position
in the original untrimmed source. Avoid overlapping passages.

Pick passages appropriate to the intended target:

- Comfortable neutral holds for a neutral comparison.
- Selected walking passages for a walking comparison.
- Demonstrated changed holds to inspect response, but not automatically as
  desirable target limits.

You can choose different passages for different metrics.

### 8.4 Inspect the evidence, including the extremes

Use **Show low example** and **Show high example**.
Ask:

- Is this really the same dancer?
- Are the dots on the body parts?
- Is the value an actual movement or a detection glitch?
- Does the camera/view match the intended comparison?

If an extreme is an incorrect hip dot, including it by widening a range does
not make the measurement trustworthy.

### 8.5 Decide the limits separately from the examples

Suppose selected pink-axis examples have:

- Median: `4.5 degrees`.
- p10: `3 degrees`.
- p90: `6 degrees`.

**Suggest middle 80% of examples** suggests `3 to 6`.

You might deliberately approve `0 to 8` after inspecting the movement and
deciding that this comparison should include a more vertical position.
That decision is yours; it is not an app-certified technique standard.

Increasing the upper limit from `6` to `8` allows values between six and eight.
Lowering the lower limit from `3` to `0` also allows values closer to vertical.

The **inside fraction** updates to describe the edited limits. It does not
make the reference more accurate.

### 8.6 Approve, skip, then save

- **Approve metric** stores the current examples, limits, view, unit and
  applicable camera approval.
- **Skip / remove metric** removes that metric from the approvals.
- **Save approved .tpc profile** saves the approved set.

After changing an already approved metric's limits, examples or view, click
**Approve metric** again so the saved approval reflects those edits.

Skipping a metric is better than approving a number you cannot observe reliably.

## 9. Head alignment: three different measurements

Do not treat every "head angle" as the same number.

### 9.1 Ear-to-nose head pitch: the horizontal head cue

**Editor:** Head > **Head pitch (ear-to-nose)**  
**Key/unit:** `head_pitch`, `degrees`.

The app estimates the direction of the ear-to-nose line relative to image
horizontal. It uses a usable ear and nose; front-view geometry can use
inter-ear scaling to reduce some rotation effects.

Negative values mean the nose is lower relative to the ear/horizontal reference.
This is not a direct neck-joint angle or a reliable measurement of head yaw or
left/right head roll.

**Approved ranges use the raw ear-to-nose angle.**
The older `head_neutral_offset` is not subtracted from `head_pitch` during
approved-range measurement.

#### A clear example of raw versus zeroed

Suppose your selected comfortable examples have raw pitch around `-12 degrees`.

- An illustrative approved range could be `-20 to -5 degrees`.
- Raw `-12` is inside.
- Raw `-25` is below the minimum.
- Raw `0` is above the maximum.

For the older warning display, with `head_neutral_offset = -12`:

`displayed corrected angle = raw angle - (-12)`.

Thus raw `-12` displays as `0`, while raw `-30` displays as `-18`.
With old `head_down_angle = -15`, that `-18` can trigger the older warning after
the old head warning duration.

These are two different comparison systems. Do not approve `-15 to 15` merely
because the old zeroed display looks near zero.

#### Older Settings: Head & shoulders

| Control | Built-in default | Increasing/decreasing: concrete effect |
| --- | --- | --- |
| `head_neutral_offset` | `-12 degrees` | At raw `-12`, changing offset to `-8` makes corrected angle `-4`; changing it to `-16` makes corrected angle `+4`. It shifts the older cue, not the approved raw range. |
| **Head down angle limit (degrees)** (`head_down_angle`) | `-15` | Change to `-10`: warning can begin at a smaller downward excursion. Change to `-20`: permits a larger downward excursion before that boundary. The value must remain negative. |
| `head_down_duration` | `3 seconds` | `1.5` confirms persistent downward pitch sooner; `5` requires it to persist longer. Does not change the angle or move the skeleton. |

If a selected approved profile governs head pitch, edit its range and general
Tracking timing instead of expecting these older controls to override it.

### 9.2 Head forward: ear displacement relative to the torso

**Editor:** Head > **Head-forward offset**  
**Key/unit:** `head_forward`, `torso_lengths`.

This measures the forward component of the shoulder-to-ear vector relative to
the torso axis, divided by torso length.
The configured facing direction defines "forward".

It uses ear, shoulder and hip geometry. The core side-view path also requires
other framing/view evidence, including nose and an ankle.

Illustration:

- Torso length: `200 pixels`.
- Ear's forward component: `10 pixels`.
- Head-forward value: `0.05 torso lengths`.

Doubling both distances through a simple uniform image enlargement still gives
`0.05`. Changing body proportions or camera perspective can change the ratio.

An illustrative approved interval `-0.02 to 0.06` includes slight displacement
on either side of that target. Raising the maximum to `0.10` allows more
forward displacement. Raising the minimum can reject values farther backward.

### 9.3 Head inclination: shoulder-to-ear direction relative to torso

**Editor:** Head > **Head inclination**  
**Key/unit:** `head_inclination`, `degrees`.

This uses the angle of the shoulder-to-ear vector relative to the torso axis.
It is not the ear-to-nose horizontal pitch.

You can therefore have:

- A near-neutral ear-to-nose pitch.
- A forward-shifted ear position relative to the shoulder.
- A changed shoulder-to-ear inclination.

Those can be different observations, not a software contradiction.

For approved profiles, use the signed native interval shown by the editor.

### Head visibility checklist

- Is the nose dot stable?
- Is at least one ear genuinely visible and correctly placed?
- Is the selected measurement side appropriate?
- Are shoulder and hip dots usable for torso-relative measurements?
- Is the head segment large enough in the image?
- Does the chosen view match the intended range?

Hair, a partner's head, hats, strong backlighting or an almost edge-on face can
interfere. Improve the view before lowering confidence.

## 10. Shoulder height and clearance

**Editor:** Shoulders > **Ear-to-shoulder clearance**  
**Key/unit:** `shoulder_clearance`, `normalized_clearance`.

### What "shoulder height" means here

The app measures vertical clearance between an ear and its corresponding
shoulder, normalized by an image-derived reference scale.
When a shoulder moves toward the ear, that clearance generally decreases.

The helper considers usable left/right pairs and selects the **smallest
clearance**, so a more elevated usable shoulder can dominate the number.
Ear/hip geometry is preferred for scaling; other head/shoulder cues can be
used when that scale is unavailable.

This is **not**:

- Comparing the two shoulder dots to see whether they have equal height.
- Measuring muscle tension.
- Measuring shoulder width in centimetres.
- Diagnosing asymmetry or pain.

### Raw clearance versus the older percentage

An approved shoulder range stores a raw normalized clearance.
The older shoulder warning derives a percentage:

`elevation % = 100 * (neutral clearance - raw clearance) / neutral clearance`.

Invented example:

- Neutral clearance: `1.05`.
- Raw clearance: `0.945`.
- Elevation: `10%`.

For that example, `shoulder_clearance` is **0.945**, not **10**.

An illustrative approved range `0.95 to 1.15`:

- Raw `1.05`: inside.
- Raw `0.90`: below the range.
- Raw `1.20`: above the range.

Both ends matter in an approved interval. A high value is simply outside your
chosen examples/limits, not automatically proof of another posture problem.

### Relevant Settings: Head & shoulders

| Control | Built-in default | Effect |
| --- | --- | --- |
| **Shoulder tension limit (%)** (`shoulder_raise_threshold`) | `10%` | Increasing to `15` requires a larger computed elevation percentage; decreasing to `6` makes the older warning more sensitive. |
| `shoulder_tension_duration` | `1 second` | Increasing to `2` requires a longer sustained excursion; decreasing to `0.5` confirms sooner. |
| `shoulder_neutral_clearance` | `1.05` | At fixed raw clearance, increasing this baseline increases the computed elevation percentage; decreasing it reduces that percentage. It is not a camera zoom control. |

**Baseline example:** with raw clearance `0.95`:

- Neutral `1.00`: elevation `5%`.
- Neutral `1.10`: elevation about `13.6%`.

The older live/offline cue can also adapt its effective neutral clearance from
observed clearances. A saved older baseline can provide another neutral value.
Do not expect every old percentage to remain a simple calculation from the
fallback field alone.

For approved ranges, use the raw clearance limits. The older percentage
thresholds do not override them.

### Why body proportions matter

A shorter apparent neck clearance or different ear/hip proportions can produce
a different raw ratio even without a change you intend to flag.
Do not assign a special threshold purely from "thin" or "big". Select comfortable,
visually trustworthy evidence for that individual and inspect the extremes.

## 11. Axis alignment: the pink line

**Editor:** Torso > **Torso axis tilt (pink line)**  
**Key/unit:** `torso_axis_tilt`, `degrees`.

### Exactly which line?

The pink line joins:

1. The midpoint between left and right shoulders.
2. The midpoint between left and right hips.

It is extended slightly beyond the two midpoints for drawing.
The cyan dashed line through the hip midpoint is **image vertical**.

The displayed/calibrated pink-axis number is the **magnitude of tilt** from that
vertical:

- `0`: vertical in the image.
- `4.5`: tilted by 4.5 degrees.
- The same leftward and rightward tilt produce the same displayed magnitude.

The underlying geometry has a signed camera-right direction, but this approved
metric deliberately uses the absolute magnitude shown on the video.

### All four landmarks are required

The app needs:

- Left shoulder.
- Right shoulder.
- Left hip.
- Right hip.

If one is unreliable or missing, the pink-axis measurement is unavailable.
It does not replace a hidden shoulder with the head or copy the visible hip
to invent the other hip.

An interpolated skeleton can look complete while this raw measurement remains
UNKNOWN.

### Approving the pink axis

1. Choose **Torso**.
2. Choose **Torso axis tilt (pink line)**.
3. Set an appropriate view.
4. Select passages with trustworthy shoulder and hip dots.
5. Inspect the suggested distribution.
6. Set and approve minimum/maximum.

Illustrative range: `0 to 7 degrees`.

- Increase maximum to `10`: values between seven and ten become allowed.
- Decrease maximum to `4`: a 4.5-degree observation becomes outside.
- Raise minimum to `2`: an exactly vertical observation becomes outside too.

There is no separate old **pink-axis limit** slider in **Torso & hip offset**.
Those `axis_curve_*` settings belong to a different hip-offset measurement.
For pink-axis limits, use the profile or **Settings > Calibration defaults**.

### Camera tilt is not body tilt

If the camera is rolled sideways, a physically vertical torso can appear tilted.
Keep the camera level before calibrating.

Changing **Head neutral offset** may shift the older head display but does not
level the camera or correct the pink torso axis.

This projected line is not a centre-of-mass, gravity sensor, spine reconstruction
or balance diagnosis.

## 12. Hip alignment: a different axis and a different number

**Editor:** Hips > **Hip offset from ankle-to-head axis**  
**Key/unit:** `hip_offset`, `axis_lengths`.

### What is measured?

The app makes an image line from a usable ankle to an estimated back-of-head
location. It then measures how far the hip lies to the posterior side of that
line, divided by the line's length.

It uses a usable ear, hip and ankle on a side. Nose information helps estimate
the back-of-head point. If that helper has no usable nose, the ear can act as
the head endpoint. Other core view gates can still require nose evidence.

The configured **Facing direction** defines the posterior/back direction.
The helper prefers the selected side and can use the alternative usable side.

### Signed raw offset and old neutral-corrected offset

- Positive raw offset: hip lies toward the defined posterior side.
- Negative raw offset: hip lies on the other side.
- Old corrected deviation: `raw offset - axis_neutral_offset`.
- Approved `hip_offset` ranges use the raw offset, without that old neutral
  subtraction.

Invented image example:

- Ankle-to-head line length: `400 pixels`.
- Posterior hip displacement: `20 pixels`.
- Raw offset: `20 / 400 = 0.05 axis lengths`, or `5%`.

An illustrative approved range `-0.02 to 0.04` flags `0.05` as above the range.
Raising the upper limit to `0.06` includes it.
Lowering the minimum to `-0.04` permits more displacement in the other direction.

### Relevant Settings: Torso & hip offset

| Control | Built-in default | Effect |
| --- | --- | --- |
| `axis_curve_entry` | `0.06` | Higher means the older posterior-offset warning needs more deviation; lower means it reacts to smaller positive deviation. |
| `axis_curve_clear` | `0.035` | After a warning, the old smoothed offset must fall below this to clear. Increasing makes clearing possible sooner; decreasing requires a smaller offset. Must be below entry. |
| `axis_curve_duration` | `1.5 seconds` | Higher confirms sustained old hip-offset warnings later; lower confirms sooner. |
| `axis_curve_clear_duration` | `0.75 seconds` | Higher requires longer recovery below clear; lower clears sooner. |
| `axis_neutral_offset` | `0.0` | Higher reduces the old corrected posterior deviation at a fixed raw value; lower increases it. Does not change the raw approved metric. |

For raw `0.05`:

- Neutral `0.00`: corrected deviation `0.05`.
- Neutral `0.02`: corrected deviation `0.03`.
- Neutral `-0.01`: corrected deviation `0.06`.

The older positive-offset warning is not the same as an approved two-sided
minimum/maximum interval.

### What it cannot tell you

This is not direct pelvic rotation, pelvic-level alignment, actual anatomical
pelvic tilt, or a 3-D spinal curve measurement.
The line is built from projected estimated points. A changed ankle endpoint,
incorrect ear dot or camera rotation can change the hip number.

## 13. Torso and trunk inclination

**Editor:** Torso > **Trunk inclination**  
**Key/unit:** `trunk_inclination`, `degrees`.

This older/core measurement describes shoulder-to-hip orientation relative to
image-up and configured facing.

In the side-view path it uses the selected side's shoulder-to-hip segment.
The all-angle path can use shoulder/hip midpoints when both sides are usable,
or usable alternative sides when not.

### Why it is not the pink-axis metric

| Property | Pink-axis tilt | Trunk inclination |
| --- | --- | --- |
| Value used for approval | Absolute magnitude | Signed native angle |
| Landmarks | Always both shoulders and both hips | Selected side or the core all-angle path |
| Facing dependence | Magnitude does not use configured facing | Facing affects the sign/convention |
| Missing-side substitution | No | Core all-angle path can use usable alternatives |
| Quality path | Four-point pink-axis helper, plus target view checks | Core torso/framing/view checks |

Do not rename an old trunk range as a pink-axis range and assume it means the
same thing.

### Relevant Settings: Tracking

Use the native signed minimum/maximum in the approved profile or Calibration
defaults. Tracking's measurement smoothing and entry/clear durations affect
evaluation timing, not those acceptable bounds.
Head inclination and trunk angles can be handled continuously across an angle
wrap, so an editor distribution may use an unwrapped interval. Do not force
every angular range into `0 to 180`; the pink-axis magnitude is a different case.

## 14. Knees, arms, embrace, ankles and feet

### 14.1 Knee flexion

**Editor:** Knees > **Least-bent reliably visible knee**  
**Key/unit:** `knee_flexion`, `degrees`.

The app builds a hip-knee-ankle angle and reports:

`flexion = 180 - interior hip-knee-ankle angle`.

Illustrations:

- Interior angle `180`: flexion `0`.
- Interior angle `170`: flexion `10`.
- Interior angle `160`: flexion `20`.

This is a projected image angle. Zero means the projected leg looks straight;
it does not medically establish a locked knee or hyperextension.

When both legs are usable, the helper generally chooses the least-bent leg.
A less reliable far leg can be excluded in favour of the preferred visible side.
It is not a pair of separately approved left/right knee ranges and does not
identify the weight-bearing leg with certainty.

Illustrative approved range `5 to 25`:

- `3`: below the range.
- `12`: inside.
- `30`: above the range.

Lower minimum from `5` to `2`: permits a straighter projected leg.
Raise minimum to `8`: rejects more straight-looking observations.
Raise maximum from `25` to `35`: permits more flexion.

Do not force extra bending merely to satisfy an example number.

### 14.2 Older knee warning

**Settings > Embrace & knees**:

| Control | Default | Effect |
| --- | --- | --- |
| `closed_embrace_min_knee_bend` | `8 degrees` | Increase to `12`: the older cue asks for more projected flexion. Decrease to `5`: permits a straighter-looking leg before its boundary. |
| `knee_straight_duration` | `1 second` | Increase to `2`: require a longer sustained straight-knee condition; decrease to `0.5`: confirm sooner. |

The older banner is tied to the heuristic **CLOSED** embrace state.
An approved `knee_flexion` interval is instead evaluated as a selected valid
range; choosing a practice context does not add the same old embrace-only gate.

If knees are hidden behind a skirt, partner or crossing leg, improve the view
or skip the metric rather than treating an invented dot as evidence.

### 14.3 Arms, role and embrace

Elbows and wrists are used for estimated arm patterns and openness cues.
There is no approved elbow bend, wrist angle or embrace-pressure range.

**Settings > Embrace & knees**:

| Control | Default | Increasing / decreasing |
| --- | --- | --- |
| `role_assign_ms` | `300 ms` | Higher requires the arm pattern to persist longer before a role is assigned; lower assigns sooner but can flicker more. |
| `role_hold_ms` | `5000 ms` | Higher retains a role longer through lost/changed arm observations; lower releases it sooner. |
| `embrace_closed_threshold` | `0.72` | At fixed openness score, higher makes CLOSED classification easier; lower makes OPEN classification easier. This is not a distance in metres. |
| `embrace_hysteresis_seconds` | `1 second` | Higher requires a candidate open/closed change to persist longer; lower switches sooner. |

The names LEADER/FOLLOWER describe heuristic observed arm patterns. They are
not a reliable declaration of a person's identity, skill or actual role.

The embrace tracker uses a smoothed openness score and a small internal
hysteresis band around the threshold. There is no separate Settings slider for
that internal band.

### 14.4 Ankle height and feet

The older lift helper compares vertical separation between the two ankle dots,
normalized by a leg-length estimate.
That is not a calibrated distance from the true floor in centimetres.

**Settings > Slow tango**:

| Control | Default | Increasing / decreasing |
| --- | --- | --- |
| `slow_tango_max_bpm` | `115 BPM` | Higher allows the slow-tango lift heuristic to operate at faster known tempos; lower restricts it to slower known tempos. |
| `ankle_high_lift_ratio` | `0.06` | Higher allows more normalized ankle separation before a high-lift classification; lower flags smaller separation. |
| `max_consecutive_high_ankle_steps` | `2` | Higher tolerates more consecutive high steps before alerting; lower alerts after fewer. |

For an estimated leg length of `300 pixels`, `0.06` corresponds to `18 pixels`
of vertical ankle separation in that projected measurement.
It is not automatically an actual 18-pixel floor clearance or a centimetre limit.

Heels and forefoot/toe-region landmarks can contribute to movement evidence.
Keep both shoes and the floor region visible. There are no independent heel or
toe calibration ranges in the approval editor.

## 15. Every calibration-related Settings tab

The tables in Sections 9-14 covered the body-specific controls.
This section covers the shared controls and how they interact with those metrics.
The listed defaults come from the current `Config`, not from older screenshots.

### 15.1 Camera & view

| Label/key | Default | Meaning and what changing it does |
| --- | --- | --- |
| **Measurement side** (`side`) | `left` | Anatomical side preferred by side-dependent measurements. Choose the visible/intended side, not whichever side looks easier in a mirrored screen. |
| **Facing direction** (`facing`) | `right` | Direction in the original image. Changes signed forward/posterior/trunk conventions. Wrong facing can reverse interpretation. |
| **Capture setup name** (`capture_profile`) | `default` | Identifies the capture arrangement in metadata; renaming does not physically correct the camera or rewrite cached evidence. |
| `allow_all_angles` | `false` | Enable the core all-angle measurement path. It may use midpoints/usable sides during additional views, but does not remove per-profile target-view checks or the pink four-point requirement. |

Changing source, resolution, side, facing or setup after a live check can make a
saved live reference incompatible. Do not check **Camera placement unchanged**
as a shortcut around a real setup change.

### 15.2 Files and pose models

| Label/key | Default | Meaning |
| --- | --- | --- |
| **Lite model file** (`model_path`) | `models/pose_landmarker_lite.task` in the base configuration | Path to the pinned Lite model for Lite offline analysis. Packaged startup can override it. An arbitrary `.task` file is not accepted just because the filename looks right. |
| **Offline analysis model** (`analysis_model`) | `lite` | Lite, Full or Heavy for Library/Review offline analysis. Changing it requires reanalysis for new cached results. |
| **Data folder** (`data_dir`) | `data` in the base configuration | Where local app data/artifacts are organised. Packaged app paths can differ. Changing it may make profiles appear missing because you are looking in a different data folder. |

A larger model does not modify acceptable joint limits. It can change detected
landmarks and processing time. Compare visually trustworthy results rather than
assuming "Heavy = correct".

### 15.3 Quality & calibration

| Key/label | Default | Increase | Decrease | Important limitation |
| --- | --- | --- | --- | --- |
| **Minimum landmark confidence** (`confidence`) | `0.70` | Accept fewer strict observations; potentially more missing measurements | Accept more uncertain observations; potentially more noisy/misplaced dots | Does not guarantee accuracy or change every older helper's internal threshold |
| `min_torso` | `0.12` of image height | Require a larger core torso segment | Permit a smaller core segment | Smaller is not automatically reliable; not a universal setting for every line/helper |
| `min_head` | `0.04` of image height | Require a larger core shoulder-to-ear segment | Permit a smaller segment | Not an ear-to-nose minimum-angle setting |
| `max_view_width` | `0.45` | Accept a broader projected ratio as side view | Require a narrower projected ratio for side view | Also changes the side/front dividing gate; not actual shoulder width in centimetres |
The six older baseline controls (view-change tolerance, capture duration,
required valid duration, sample count, offset MAD and angle MAD) were removed
from Settings and new TOML exports. Older configuration files still load for
compatibility with retained baseline code; the current reference-range workflow
does not use those controls.

The shared approval editor uses selected evidence, distributions and explicit
ranges. The guided repeat check uses the fixed criteria explained in Section 6.

Some helpers also have internal pixel guards, such as minimum useful leg-segment
length or head-line separation. There is no separate GUI setting for every such
guard. Good framing remains necessary even with relaxed quality settings.

### 15.4 Tracking

| Key/label | Default | Increasing / decreasing |
| --- | --- | --- |
| **Measurement smoothing (seconds)** (`smoothing`) | `0.25` | Higher stabilises evaluated numbers but responds more gradually; lower follows changes more promptly but exposes more noise. |
| `overlay_smoothing` | `0.20 seconds` | Higher smooths some live/analysis display-related paths and older derived cues more; lower reduces that smoothing. Review's same-frame raw skeleton is not made faster by blindly changing this. |
| `entry_duration` | `1.5 seconds` | Higher requires a longer continuous outside-range/change condition; lower confirms sooner. Used by approved-range evaluators. |
| `clear_duration` | `0.75 seconds` | Higher requires longer recovery inside approved limits or below an old clear boundary; lower clears sooner. |
| `max_gap` | `0.5 seconds` | Higher permits larger observation intervals before continuity/filter reset checks; lower rejects continuity sooner. Missing raw values still become UNKNOWN. |

**Timing example:** a smoothed pink-axis value goes above an approved maximum
for `0.4 seconds`.

- With `entry_duration = 1.5`, that brief excursion does not confirm an event.
- With `entry_duration = 0.3`, it may confirm.
- Increasing the maximum so that the value is inside means there is no
  outside-range condition to time in the first place.

### 15.5 Calibration defaults

Initially no default ranges are enabled.
Each of the eight metrics has:

| Control | Meaning | Effect of editing |
| --- | --- | --- |
| **Use** | Enable this default range | With at least one enabled range, defaults replace the older posture-threshold system when no profile is selected |
| Minimum | Lowest accepted native value | Lowering generally permits smaller values; raising excludes more small values |
| Maximum | Highest accepted native value | Raising generally permits larger values; lowering excludes more large values |
| View | `side`, `front` or `all` | Controls the range's view gate; does not change missing landmarks into measured ones |
| Unit in the row label | Fixed native unit for that metric | Not an editable conversion choice; use the unit shown |

Minimum must not exceed maximum. Equal limits are allowed but can be extremely
sensitive to small measurement variation.

When copying a profile into defaults, the app replaces the default-range set
with that profile's approved metrics. It does not merge skipped metrics from an
older default set. Review the resulting **Use** checkboxes.

The defaults keep limits, units and views. They do not retain the full evidence,
profile identity, repeat checks or camera-scoring permission.

### 15.6 Replay movement

| Label/key | Default | Meaning and effect |
| --- | --- | --- |
| `replay_step_history_seconds` | `5 seconds` | Increase to retain a longer step-footprint history; decrease for less visual clutter. Allowed range is `0.5 to 30`. |
| **Review joint interpolation** (`replay_joint_interpolation`) | `linear` | Choose linear or quadratic drawing interpolation. See Section 19. |
| **Maximum joint gap to fill (seconds; 0 disables)** (`replay_joint_max_gap`) | `3 seconds` | Increase to draw through longer bracketed joint gaps; decrease to limit inferred drawing. `0` disables it. Does not supply scoring evidence. |

The inactive `replay_auto_analyse_movement` control was removed from Settings.
Use **Analyze posture & movement** explicitly. Older TOML files remain importable.

### 15.7 A short explanation of invalid combinations

Examples that the configuration rejects:

- Minimum greater than maximum.
- Confidence outside `0 to 1`.
- A head-down angle that is not negative.
- Clear threshold equal to or above entry in a threshold pair.
- Older required valid calibration time greater than its capture window.
- Negative interpolation gap.
- Unknown metric or wrong unit in default ranges.

If applying Settings fails, read the explicit error. Do not assume an invalid
value was applied just because you typed it into a numeric widget.

## 16. Examples for tall, short, slender and larger builds

### First, the honest answer about body-type presets

The app has **no height, weight, BMI or body-build preset** that computes
validated posture thresholds.

Angles resist simple image scaling; ratios normalise some distances. This is
why a taller person does not automatically need a larger knee angle or pink-axis
limit. Body proportions and visibility can still affect measurements.

The useful adjustments are:

1. Camera position and framing.
2. Requested resolution if needed and supported.
3. Correct side/facing/view.
4. Individual reference evidence and approved ranges.
5. Quality/timing changes only when observations justify them.

The examples below deliberately keep most numerical settings the same. That is
not missing advice: **size alone is not evidence for changing the thresholds**.

### 16.1 Tall adult: keep head and feet in the frame

Possible situation: your head nearly touches the top edge and the ankles leave
the image during a step.

Try:

1. Move the camera farther back and adjust support height.
2. Keep it level rather than pointing steeply upward.
3. Leave movement margin.
4. If the body becomes too small/noisy, try supported `1280 x 720` capture.
5. Create a fresh calibration for the resulting setup.

Illustrative settings fragment:

```toml
width = 1280
height = 720
side = "left"
facing = "right"
capture_profile = "tall-framing-room-a"
confidence = 0.70
min_torso = 0.12
min_head = 0.04
max_view_width = 0.45
analysis_model = "lite"
```

**What this changes:** potential image detail and setup identification.
It does not add a height measurement or relax knee/head limits.
At height `720`, the core minimum torso length is `86.4 pixels`.
Confirm the actual camera resolution and check whether the body is large enough.

If higher resolution makes live inference sluggish, return to a supported
lower resolution and improve framing. The extra pixels are not free performance.

### 16.2 Short adult: make the useful segments big enough

Possible situation: the full body is visible but occupies a small part of the
image, and ears or knees flicker.

Try:

1. Lower the camera support and move closer.
2. Avoid cropping the feet while making the torso/head larger in the image.
3. Keep the same confidence threshold initially.
4. Use an individual reference rather than the tall person's raw shoulder
   clearance interval.

Illustrative settings fragment:

```toml
width = 1280
height = 720
side = "left"
facing = "right"
capture_profile = "short-framing-room-a"
confidence = 0.70
min_torso = 0.12
min_head = 0.04
max_view_width = 0.45
analysis_model = "lite"
```

**What this changes:** a separately identified setup with enough potential
detail. The same ratio thresholds can be used because they refer to image
proportions, not centimetres of person height.

Only if framing cannot be improved and the dots are visibly trustworthy might
you test a smaller core size guard, such as `min_torso = 0.10`.
That permits a smaller segment; it does not restore detail that the camera
never captured. Test the repeatability again.

### 16.3 Slender adult: check outline and landmark stability

Possible situation: the hip dot jumps between your body and loose dark clothing
against a dark background.

Try:

1. Improve contrast and lighting.
2. Make sure clothing outlines do not obscure the hip/knee region.
3. Inspect the actual dot during stillness and gentle movement.
4. Keep quality defaults initially.
5. Select reliable passages from that person's own reference.

Illustrative settings fragment:

```toml
width = 640
height = 480
capture_profile = "slender-side-room-a"
side = "left"
facing = "right"
confidence = 0.70
smoothing = 0.25
overlay_smoothing = 0.20
max_view_width = 0.45
```

**What this means:** no special slender-body angle limits have been invented.
If the dot is stable but slightly noisy, testing `smoothing = 0.35` can calm the
evaluated number while making it respond more slowly. It will not repair a dot
on the wrong location.

### 16.4 Larger or broader adult: inspect the view gate, not the person's size label

Possible situation: you look side-on, but the app reports a view problem.
The projected shoulder/hip span relative to torso length is broader than the
side-view limit.

Try:

1. Verify side/facing and genuinely turn to the intended view.
2. Check that both shoulder dots and hips are correctly located.
3. Keep `max_view_width = 0.45` initially.
4. If a trustworthy side view repeatedly produces, for example, ratio `0.48`,
   you could test a limit such as `0.50`, then repeat the camera check.
5. Do not increase the limit merely to classify frontal footage as side-on.

Illustrative starting fragment:

```toml
width = 1280
height = 720
capture_profile = "broader-side-room-a"
side = "left"
facing = "right"
confidence = 0.70
min_torso = 0.12
min_head = 0.04
max_view_width = 0.45
```

**What this means:** the starting quality rules are unchanged.
The optional `0.50` adjustment depends on observed projected geometry, not
weight or a build category. Changing this field also changes the front/side
gate for profiles and defaults, so retest the intended comparisons.

For shoulder clearance, record an individual comfortable reference. Different
proportions can change the ratio; they do not justify automatically accepting
every low-clearance observation.

### 16.5 Combining cases

A person can be tall and slender, short and broad, or neither.
Combine the framing advice that solves the actual visibility problem.
Do not combine hypothetical "body-type corrections" to invent a target.

For all four cases, a reasonable beginner sequence is:

1. Start with built-in limits/quality controls.
2. Improve the camera evidence.
3. Make an individual reference.
4. Review native measurements.
5. Approve only the metrics you trust.
6. Change one setting at a time and compare the same passage.

## 17. Worked example: eight approved default ranges

The following is an **invented configuration exercise**, not a prescription
for tall, short, slender or larger people.

Suppose you have reviewed individual evidence and want to demonstrate the
native units of all eight metrics. These illustrative default ranges could be
written as:

```toml
[calibration_defaults.head_pitch]
lower = -20.0
upper = -5.0
unit = "degrees"
view = "side"

[calibration_defaults.head_forward]
lower = -0.02
upper = 0.06
unit = "torso_lengths"
view = "side"

[calibration_defaults.head_inclination]
lower = -8.0
upper = 8.0
unit = "degrees"
view = "side"

[calibration_defaults.shoulder_clearance]
lower = 0.95
upper = 1.15
unit = "normalized_clearance"
view = "side"

[calibration_defaults.hip_offset]
lower = -0.02
upper = 0.04
unit = "axis_lengths"
view = "side"

[calibration_defaults.torso_axis_tilt]
lower = 0.0
upper = 8.0
unit = "degrees"
view = "all"

[calibration_defaults.trunk_inclination]
lower = -6.0
upper = 6.0
unit = "degrees"
view = "side"

[calibration_defaults.knee_flexion]
lower = 5.0
upper = 25.0
unit = "degrees"
view = "side"
```

**What this example means:**

- Head pitch is raw and can be negative.
- Shoulder clearance is a normalized ratio, not an elevation percent.
- Hip offset is signed and measured in axis lengths.
- Pink-axis tilt is a nonnegative magnitude, with both tilt directions treated
  equally.
- Trunk inclination remains a separate signed range.
- Knee flexion has both a minimum and a maximum.
- Each metric has its own view.
- These are **default ranges**, not a `.tpc` profile and not live-scoring approval.

If a profile is selected, that profile still takes precedence.
If no profile is selected, the presence of these default ranges replaces the
older posture thresholds.

### Tuning a single example rather than changing everything

Imagine the pink-axis examples are trustworthy, but natural selected movement
occasionally reaches `9 degrees` and you deliberately want to include it.

Change only the upper limit from `8` to `10`, inspect the same passage, and
review the outside-range events. Do not simultaneously lower confidence,
increase smoothing and widen every other range: you will not know which change
affected the result.

If the `9-degree` observation comes from a hip dot jumping onto clothing,
do not use that example to justify a wider target. Improve or exclude the
unreliable evidence instead.

## 18. Save, reload and replace Settings safely

### Apply is not the same as save

| Action | Updates the current app configuration? | Writes a settings file? |
| --- | --- | --- |
| **Apply Settings** | Yes, if validation succeeds | No |
| **Apply & Save settings.toml** | Applies the Settings and attempts an atomic file save | Yes, after overwrite confirmation |
| **Export TOML...** | Applies valid current edits before exporting | Yes, at your chosen destination |
| **Import TOML...** | Loads the chosen file into the app | Does not automatically rewrite another file |
| **Reset Defaults** | Replaces current configuration with built-in defaults | Not automatically |
| **Use as Settings defaults** from a calibration | Replaces the default-range set after successful saving | Yes, overwrites `settings.toml` after confirmation |

The calibration-copy save publishes its new configuration after a successful
write. The Settings page's Apply-and-Save action first applies the edits; if the
file write then fails, the current app can have applied changes that are not
persisted. Always read the result/status message.

### Where is settings.toml?

These save actions resolve `settings.toml` in the app process's **current working
directory**, not automatically in **Data folder**.
Read the exact destination displayed by the confirmation dialog.

Do not assume that changing `data_dir` changes the TOML destination.
If you want a known location, use **Export TOML...** and choose it explicitly.

The app's ordinary configuration loader reads an explicitly supplied settings
path; it does not automatically read an arbitrary file merely because it is
named `settings.toml`.

After restarting, use **Import TOML...** to load your saved file, or launch the
source CLI with its supported configuration-file option. Verify the visible
Settings before relying on the saved defaults.

### Copying a calibration into defaults

1. Export your current Settings as a backup if you want to retain them.
2. In the editor or Library, choose the applicable **Use ... as Settings defaults**
   action.
3. Read the profile name, metrics and destination path.
4. Confirm overwrite only if that is what you intend.
5. Open **Settings > Calibration defaults** and inspect the enabled rows.
6. Confirm the save succeeded.

The operation saves all current configuration along with the replacement range
set. It does not overwrite the original reference video or convert defaults into
camera approval.

## 19. Missing joints, interpolation and apparent delay

### Why a dot or line disappears

Typical causes:

- A body part leaves the frame.
- A partner or another limb hides it.
- Clothing or lighting makes the estimate uncertain.
- Presence/visibility falls below the drawing/measurement threshold.
- Dancer association becomes ambiguous.

A drawn line depends on its endpoints. One disappearing endpoint can remove
several connected segments.

### Review interpolation: what it actually does

**Settings > Replay movement**:

- **Review joint interpolation:** linear by default, or quadratic.
- **Maximum joint gap to fill:** three seconds by default; zero disables.

If a joint is valid before and after a short gap, Review can draw estimated
positions between those endpoints:

- **Linear:** follow a straight path between the known positions.
- **Quadratic:** use an additional nearby reliable sample to attempt a curved
  path; fall back to linear when unsuitable.

The app does not extrapolate indefinitely before the first observation or after
the last one. It also avoids bridging unsuitable timing or ambiguous person
segments.

**Example:** wrist visible at `2.0s`, hidden for `0.4s`, visible again at `2.5s`.
Review can fill the drawing if the surrounding samples and timing qualify.

This does not prove where the wrist actually moved during the hidden period.
Fast circular movement can be poorly represented by a straight interpolation.

### Increasing/decreasing the gap

- Increase from `3` to `5`: more long gaps may be drawn through, with more
  uncertainty about the actual hidden path.
- Decrease from `3` to `0.5`: fewer/shorter gaps are filled.
- Set `0`: no interpolation.

This is a Review display setting, not a live camera occlusion predictor.
The separate `max_gap = 0.5` tracking/measurement setting is not the same control.

### Can interpolation repair calibration?

No. It does not replace the raw cached poses used for calibration distributions,
quality checks, feedback or scoring.

If the right hip is hidden, a displayed estimated skeleton hip does not supply
the pink axis's required measured right hip.

### Why the skeleton or number seems behind the movement

There are several different sources of apparent delay:

1. Camera capture/processing latency.
2. Model inference time.
3. Some live/display smoothing.
4. Measurement smoothing.
5. Warning confirmation duration.

Review draws the skeleton from same-frame raw evidence rather than the lagged
causally smoothed skeleton, then optionally fills display gaps.
That improves frame alignment without removing measurement smoothing.

Reducing **entry_duration** makes an alert confirm sooner, not the skeleton
move sooner. Reducing **smoothing** changes evaluated numbers, not camera capture
latency. Larger offline models cost more processing and do not guarantee a
better-looking or more responsive live experience.

## 20. Troubleshooting: symptom, cause, next action

| Symptom | Likely thing to check | Next action |
| --- | --- | --- |
| No pink-axis metric in an older profile | It was never approved | Create/review a reference calibration; choose Torso > Torso axis tilt (pink line), approve and save a new profile. |
| Pink line disappears while green torso segments remain | One of four required shoulders/hips is unreliable | Inspect all four raw dots. Improve view or accept UNKNOWN; do not assume the visible side can replace the hidden side. |
| Pink angle looks wrong for a vertical person | Camera roll, midpoint estimates or image perspective | Level the camera and inspect both shoulder/hip pairs before widening the range. |
| Head is near zero on the old overlay but outside approved limits | Corrected display angle versus raw `head_pitch` | Inspect raw reference values; do not apply `head_neutral_offset` twice. |
| Changing head-down/shoulder/knee sliders does nothing | A profile/default range is taking precedence | Edit the relevant approved/default range, or intentionally select no profile and disable all default ranges to use the older mode. |
| Knees have no approved measurement | Not approved, wrong view, or hidden hip/knee/ankle | Check the profile's metric set and raw landmarks. Missing metrics do not fall back to old Settings. |
| Shoulder value is around `1.0`, not `10%` | Raw clearance versus old elevation percentage | Use normalized clearance limits, not percent values, in the approved row. |
| Hip direction seems reversed | Wrong original facing direction | Verify facing in unmirrored video; mirror is not the correction. Recreate/reanalyze evidence consistently as needed. |
| Side-view UNKNOWN for a broader outline | Framing, incorrectly located dots or width-to-torso gate | Check actual geometry before adjusting `max_view_width`; repeat the intended camera check after changes. |
| Wizard repeatedly fails the repeat check | Too little usable evidence, noisy holds, changed view or inconsistent return | Improve framing/lighting and perform only a comfortable repeat. Retry or skip rather than forcing movement. |
| Increasing calibration duration does not extend wizard holds | That field belongs to the older baseline capture | The guided phases are fixed at two seconds settle plus three seconds capture. |
| Saved video calibration does not enable live scoring | Comparison-only status | Use Library > Check camera and explicitly approve eligible metrics for that camera. |
| Check camera reports different models | Full/Heavy reference versus Lite camera | Create matching Lite reference/calibration evidence for live use; do not bypass the mismatch. |
| Selecting another analysis model does not change existing video results | Old cache remains | Run Analyze posture & movement again with the intended model. |
| Old warnings remain painted on the video | Baked overlay stream | Switch to raw/source video. A calibration cannot erase existing pixels. |
| Joint appears during a gap but the metric stays UNKNOWN | Display-only interpolation | Expected: measurement evidence remains raw. |
| Profile disappeared after changing Data folder | Library is looking in another location | Restore/select the intended data folder or locate the original archive. Do not assume it was deleted. |
| Saved Settings are absent after restart | File not imported or startup not configured to load it | Import the saved TOML and verify current values. |
| Apply/Save fails | Invalid combination or file access problem | Read the status/error; correct values or choose a writable export destination. |

### A controlled tuning experiment

Use one short clip with both a stable passage and an actual change you want to
observe.

1. Record the current range, confidence and timing settings.
2. Watch the same passage at the same playback position.
3. Change exactly one parameter.
4. Check whether fewer warnings came from a justified wider target or merely
   from accepting poor observations.
5. Restore the old value if the intended movement is now missed.
6. Repeat with a new sample, not just the passage you tuned against.

This is a practical consistency check, not a validated model-accuracy study.

## 21. Other Settings that affect a session or video review

These are not per-joint calibration limits, but they are present in Settings
and can explain recording or review behaviour.

### Music

| Key/label | Default | Effect |
| --- | --- | --- |
| **Fallback tempo (BPM)** (`tango_bpm`) | `120` | Tempo supplied when analysis has no soundtrack BPM, not the slow-tango ceiling. At the defaults, 120 exceeds the 115 BPM ceiling and the lift cue is inactive. |

The live microphone/automatic-recording controls are absent from current Settings.
They do not affect raw shoulder or knee landmarks.

### Review & Ollama

Ollama interpretation is optional report commentary. It does not relocate
landmarks or replace the numeric calibration evaluator.

| Key/label | Default | Effect |
| --- | --- | --- |
| **Ollama server address** (`ollama_base_url`) | `http://127.0.0.1:11434` | Address of the configured report-interpretation server. A nonlocal server is not the same privacy boundary as a local server. |
| `ollama_model` | `llama3.2` | The language model for interpretation, not Lite/Full/Heavy pose tracking. |
| `ollama_timeout` | `120 seconds` | Increase to wait longer for commentary; decrease to fail sooner on slow responses. Maximum `1800`. |
| **Response variation** (`ollama_temperature`) | `0.2` | Higher allows more variable language-model responses; lower tends toward less variation. Not a measurement confidence control. Allowed `0 to 2`. |
| **Model context size (tokens)** (`ollama_num_ctx`) | `16384` | Increase for more possible language-model context with more resource cost; decrease for less. Allowed `1024 to 131072`. |
| **Maximum response length (tokens)** (`ollama_num_predict`) | `2048` | Increase for potentially longer commentary; decrease for shorter output. Allowed `128 to 16384`. |
| **Maximum report length (characters)** (`ollama_max_report_chars`) | `40000` | Increase the text budget supplied for interpretation; decrease it and more report text may be omitted. Minimum `1000`. |
| **Analysis focus** (`ollama_instruction`) | Explain supported observations and suggest three focused practice exercises | Changes what commentary is requested. Does not add a new calibrated joint or establish medical validity. |

Do not confuse the pose model selector in **Files** with the language model
field here.

The older TOML `axis_curve` threshold object is a compatibility alias for the
axis entry/clear pair, not another independent GUI metric. Prefer the visible
`axis_curve_entry` / `axis_curve_clear` controls when editing current Settings.

## 22. Printable checklists and final summary

### Before a camera calibration

- [ ] One dancer in the evidence.
- [ ] Camera stable and level.
- [ ] Head and feet have movement margin.
- [ ] Needed joints are correctly located, not just visible.
- [ ] Side and unmirrored facing are correct.
- [ ] Lite model is available.
- [ ] Setup name describes this camera arrangement.
- [ ] Activity/context is chosen deliberately.

### Before saving a profile

- [ ] Correct numbered dancer and passage selection.
- [ ] Low/high examples inspected.
- [ ] Native units understood.
- [ ] Minimum and maximum edited independently of the examples.
- [ ] Required view set for each metric.
- [ ] Each intended metric explicitly approved.
- [ ] Edited approved metrics re-approved after changes.
- [ ] Untrustworthy metrics skipped.
- [ ] Camera approval checked only for eligible, inspected camera evidence.
- [ ] Distinct profile name chosen.

### Before comparing a video

- [ ] Raw/source recording available.
- [ ] Analysis evidence exists and is not only display-smoothed poses.
- [ ] Model matches the calibration.
- [ ] Recorded side/facing matches.
- [ ] Appropriate Calibration profile selected in Review.
- [ ] Visual side-by-side comparison not confused with numeric profile selection.
- [ ] UNKNOWN treated as missing evidence, not success.

### Before using saved default ranges

- [ ] No selected profile is unexpectedly overriding them.
- [ ] Intended Use checkboxes enabled.
- [ ] Skipped metrics understood to remain unavailable.
- [ ] Save/export destination checked.
- [ ] Overwrite decision intentional.
- [ ] Success status confirmed.
- [ ] File imported or explicitly configured for startup after a restart.

### The shortest useful summary

1. **Set up the camera well.**
2. **Record or import trustworthy evidence.**
3. **Choose the correct dancer, context, view and model.**
4. **Review each measurement in its own native unit.**
5. **Approve ranges deliberately, not merely because a suggestion exists.**
6. **Check the camera before live scoring.**
7. **Remember which source of limits is active.**
8. **Use body size to guide framing, not to invent posture standards.**
9. **Treat interpolation as drawing and UNKNOWN as unavailable evidence.**
10. **Save and reload Settings explicitly.**

## 23. Implementation references

For readers who want to verify an exact parameter or measurement definition:

- [Current Settings fields and defaults](../src/tango_posture/config.py).
- [Qt Settings tabs, labels and save actions](../src/tango_posture/desktop_qt/screens/settings.py).
- [Retained landmark names](../src/tango_posture/pose/constants.py).
- [Geometry helpers for head, shoulders, knees and both axes](../src/tango_posture/geometry.py).
- [Core head and trunk quality/view checks](../src/tango_posture/metrics.py).
- [Approved metric registry and profile definitions](../src/tango_posture/calibration/models.py).
- [Raw approved-range measurements and view gates](../src/tango_posture/calibration/measurements.py).
- [Evidence approval editor](../src/tango_posture/desktop_qt/calibration/editor.py).
- [Guided camera capture workflow](../src/tango_posture/desktop_qt/calibration/guided.py).
- [Repeatability and passage statistics](../src/tango_posture/calibration/statistics.py).
- [Range/event timing](../src/tango_posture/temporal.py).
- [Live session camera-approved metric selection](../src/tango_posture/session.py).
- [Settings-default copying and file destination](../src/tango_posture/desktop_qt/service_bridge.py).
- [Cached Review comparisons](../src/tango_posture/references/coach.py).
- [Pinned Lite/Full/Heavy weights](../models/manifest.json).
- [General TangoPosture manual](TANGO_POSTURE_GUIDE.md).
