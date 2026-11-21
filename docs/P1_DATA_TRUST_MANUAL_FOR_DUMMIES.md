# TangoPosture Data-Trust Manual (For Dummies): The Four P1 Fixes

> **The 10-Second Summary:**  
> Before these fixes, TangoPosture could occasionally **mistake a copy-pasted guess for a real camera measurement**, **turn a single noisy frame at the end of a clip into a fake mistake**, **glue Partner A's turn start to Partner B's turn end**, or **stamp a clean `"0"` or `"user_confirmed"` badge onto data that was partly guessed or never even checked**.  
> These four **P1 fixes** make the numbers honest, keep visual helpers separate from real measurements, and tell you plainly when evidence is `mixed`, `insufficient_evidence`, or `unavailable`.

---

## 1. The 60-Second Version: Before vs. After

Think of TangoPosture like a dance teacher taking notes from video:

| Area | What went wrong before ("Before") | How it behaves now ("After") |
| :--- | :--- | :--- |
| **1. Where poses came from (`PoseOrigin`)** | Copy-pasted ("propagated") poses in close embrace looked identical to real fresh detections and inflated measured coverage. | Every pose carries its birth certificate (`detected`, `crop_detected`, `propagated`, `display_interpolated`). By default (`measured_only`), propagated guesses still draw on video but do **not** pad measured stats. |
| **2. Transition checks (rises & reversals)** | A single-frame speed spike right at the end of a clip counted as a fault; overlapping windows could double-count one moment; reversals used one clip-wide sign even when the couple danced in two directions. | A transition cue must last for the full required duration (`persistence_s`) inside one unbroken tracking segment, gets one stable episode ID so overlaps never double-count, and checks reversals against that phase's own travel direction. |
| **3. Pairing turn markers** | Turn starts, ends, and exits matched purely by timestamp order — ignoring whether the marker belonged to Partner A, Partner B, or the pair — and incomplete markers vanished silently. | Turns link by explicit `turn_id` and matching `subject` (`A`, `B`, or `pair`). Ordering/subject conflicts are caught, legacy overlapping A/B turns stay separated, and unmatched markers stay visible so you can fix them. |
| **4. Honest summary badges & denominators** | Mixing 1 confirmed event with 4 automatic guesses still said `user_confirmed`; 0 checked transition windows reported a clean `0` faults; edited coordination lags always claimed `measured`; coverage hid what total it divided by. | Summaries aggregate status counts separately (`measured`, `estimated`, `user_confirmed`, `mixed`, `insufficient_evidence`, `unavailable`), show `eligible/total frames` beside coverage, and report `Unavailable` whenever 0 transition windows were evaluated. |

---

## 2. Deep-Dive on Each Issue (Zero Tech Background Needed)

### Issue 1: "A Sketch on a Sticky Note Looked Like a Photograph" (Pose Origins)

#### What was broken (Everyday Analogy)
Imagine you are tracking two runners in a race and your stopwatch camera blinks for two frames. To keep the smooth overlay line visible on screen, you photocopy the previous frame and slide it forward a little bit (`_translate_pose`).

Previously, once those copied poses entered the tracking pipeline (`TrackedDancerSample`), TangoPosture threw away the sticky note saying *"I slid this forward from an earlier frame."* So downstream calculations counted the copied sticky notes as 100% genuine new observations! Meanwhile, the app's documentation promised that "interpolation is display-only," which wasn't true for that earlier propagation step.

#### Why it could mislead you
* **Inflated coverage:** If 1 frame was genuinely split in close embrace and 2 later frames were translated copies, coverage could look like **100% (3/3)** instead of **33% (1/3)**.
* **Phantom precision:** Smoothed velocity and torso angles could look `"measured"` even during frames where the pose detector never actually saw both bodies separately.

#### How we solved it
1. **Every landmark and pose now wears an ID badge (`PoseOrigin`)** defined in `src/tango_posture/domain.py`:
   * **`detected`** — Freshly detected directly by MediaPipe on that frame (`mediapipe_backend.py`).
   * **`crop_detected`** — Re-detected on that frame by splitting a tight left/right crop around a merged close-embrace couple (`resolve_close_embrace_frame_poses` in `couple/tracking.py`).
   * **`propagated`** — Translated forward from an earlier split pair without running fresh pose detection (`references/pipeline.py`, `engines/couple.py::_resolve_close_embrace_fallback`).
   * **`display_interpolated`** — Smoothly filled in **only for drawing overlays** (`pose/interpolation.py`, `couple/overlay.py`).
2. **Origins survive saving & reloading:** Archive round-trips (`.tp`/`.tpr`, `coach_data` sibling metadata `measurement_pose_origins` / `measurement_landmark_origins`, `couple_result["series"]["frame_pose_origins"]`, and `frame_overlays`) preserve every origin without breaking legacy 5-number landmark caches (`calibration/evidence.py`, `calibration/normalization.py`, `references/coach.py`).
3. **Drawing is separate from measuring:** In `couple/tracking.py` and `couple/measurements.py`, an explicit policy (`dancer_analysis.pose_origin_policy`) controls metric eligibility:
   * Default **`"measured_only"`**: Keeps `display_pose` so the skeleton still draws cleanly (with dashed `(est)` styling in `couple/overlay.py`), while excluding `propagated` samples from measured coverage, velocities, angles, Pearson `r`, fixed torso scale, and step/pivot extraction.
   * Opt-in **`"allow_propagated_estimates"`**: Lets you include `propagated` frames in calculations **explicitly tagged as `"estimated"`** (never `"measured"`), turning mixed runs into `"mixed"` summaries while still recording `measured_only_coverage` separately.
   * **`display_interpolated`** samples are **never** admitted into measurement math under either policy.

#### Before vs. After Example
On a 3-frame close-embrace clip with **1 `crop_detected`** keyframe and **2 `propagated`** translated frames:
* **Before:** `pair_torso_coverage = 1.0 (100%)`, every frame says `measured`.
* **After (`measured_only` default):** `pair_torso_coverage = 0.3333 (33%)`, frames 1 & 2 report `status: "unavailable"`, `reason: "propagated_pose_excluded"`, `pose_origin: "propagated"`, and overlays still draw smoothly.
* **After (`allow_propagated_estimates`):** `pair_torso_coverage = 1.0`, `measured_only_coverage = 0.3333`, frames 1 & 2 report `status: "estimated"`, and the posture summary honestly reports `status: "mixed"`.

---

### Issue 2: "One Hiccup Before the Buzzer Counted as Falling Over" (Transition Cues)

#### What was broken (Everyday Analogy)
During transitions (when a couple slows down, pauses, or reverses direction), TangoPosture checks for unexpected secondary speed surges (`secondary_speed_peaks`) and unintended reversals (`unintended_reversals`), requiring them to hold for at least `persistence_s` (e.g. `0.15 s` or `0.20 s`).

Previously there were three sneaky traps inside `_evaluate_transitions` (`couple/measurements.py`):
1. **Last-frame loophole:** The check asked *"were all existing samples from `tk` up to `tk + persistence_s` above the threshold?"* — it forgot to check whether any sample actually reached `tk + persistence_s`! On the very last frame of a window (or right before a camera cut), a 1-frame spike had 0 future points to disagree with it, so Python's `all(...)` happily said `True`.
2. **Crossing cuts & double-counting:** Adjacent `slow` events created overlapping windows that evaluated independently and could count the exact same speed rise twice; breaks/gaps weren't partitioned cleanly everywhere.
3. **One clip-wide direction sign:** The reversal detector checked what direction you travelled **most of the clip overall**. If you walked forward (+) for 1 second, paused intentionally, and walked back (−) for 3 seconds, the whole-clip majority was `−`, so your entire 1-second forward walk (+) could look like a reversal!

#### How we solved it
1. **Strict contiguous interval partitioning (`_partition_contiguous_eligible_intervals`):** Any camera cut (`break_mask`), ineligible frame (`not pair_hip_eligible`), missing velocity, or gap larger than `max_observation_gap_s` splits the timeline into completely separate islands. A persistence check can never jump across a cut or dropout.
2. **Real elapsed time required (`_has_elapsed_support`):** A speed rise or reversal only counts if we actually observe unbroken eligible samples spanning all the way from onset `tk` to at least `tk + persistence_s`. A final-frame glitch now emits **zero** cues.
3. **Canonical Episode IDs & overlap deduplication:** Confirmed cues receive stable episode IDs (`ep_speed_peak_<ms>` and `ep_reversal_<ms>`) and recorded support duration. Overlapping inspection windows never emit the same physical episode twice.
4. **Per-phase reversal reference (`_phase_reference_sign`):** Reversals are now compared against the couple's actual pre-slowing travel direction inside that **same** unbroken motion phase, not the clip-wide majority. Also, `_detect_motion_events_for_track` now recognizes when a clip starts already mid-stride so later deceleration (`slow`) windows are still captured cleanly without inventing fake `start` events at `t = 0`.

#### Before vs. After Example
* **Last-frame spike:** 0.05 s before clip end (`persistence_s = 0.20 s`): **Before = 1 fault**, **After = 0 faults**.
* **Overlapping slow windows over one genuine rise:** **Before = 2 faults**, **After = 1 deduplicated episode**.
* **Two-direction practice clip (short `+` phrase, pause, longer `−` phrase):** **Before = false reversal during Phase 1**, **After = 0 false reversals** (and catches any real reversal inside Phase 2).

---

### Issue 3: "Partner A Borrowed Partner B's Shoes" (Turn Marker Matching)

#### What was broken (Everyday Analogy)
Suppose Dancer A starts a giro at 1.0 s and finishes at 2.0 s, while Dancer B starts their own movement at 1.2 s and finishes at 1.8 s.

Previously, `_turn_events_from_annotations_and_candidates` threw all `turn_start` markers into one bucket and all `turn_end` markers into another, pairing each start with the chronologically next end regardless of who the marker was for! So A's start at 1.0 s could grab B's end at 1.8 s, and any orphan start missing an end was quietly ignored instead of shown for cleanup.

#### How we solved it
1. **Persistent `turn_id` (`CoupleAnnotationItem.turn_id`, `CoupleEvent.turn_id`):** Turn markers now carry an explicit ID tying `turn_start`, `turn_end`, and `exit` into one turn object. When you add a `turn_start` followed by `turn_end` / `exit` for the same subject in the **Review** screen (`desktop_qt/screens/review.py`), TangoPosture automatically links them with a shared `turn-...` ID.
2. **Strict validation & subject isolation (`couple/measurements.py`):**
   * Explicit `turn_id` groups reject subject conflicts (`turn_id_subject_mismatch`) and reversed timestamps (`invalid_turn_marker_order`) instead of silently pairing garbage.
   * Legacy annotations without a `turn_id` are paired **strictly within the same subject** (`A` only matches `A`, `B` only matches `B`, `pair` only matches `pair`) — overlapping A/B turns never cross-contaminate, and shared `pair` turns carry an explicit `shared_pair_turn = True` flag. Nested ambiguous same-subject legacy turns emit a clear diagnostic warning.
   * Exits pair strictly by matching `turn_id` or same `subject` within `turn_exit_match_window_s`.
3. **Unmatched markers are never lost:** Any incomplete `turn_start` (`missing_turn_end`), orphan `turn_end` (`missing_turn_start`), orphan `exit` (`unmatched_exit_no_matching_turn`), or invalid marker stays visible in both `turn_summary["unmatched_markers"]` and `events` so you can see and fix it in Review/Analytics.

---

### Issue 4: "One Rubber Stamp Didn't Mean Everything Was Checked" (Honest Summary Statuses)

#### What was broken (Everyday Analogy)
Imagine a teacher grading a 5-question quiz where you answered 1 question yourself and an AI guessed the other 4 — and putting a giant green **"100% MANUALLY CONFIRMED"** sticker at the top. Or turning in a blank page where no transition windows existed at all and seeing **"0 MISTAKES — MEASURED!"**

Previously in `couple/measurements.py`:
* **Turns & Steps:** If **any** single turn (or step) was `user_confirmed`, the entire summary badge claimed `"user_confirmed"`, hiding 4 unreviewed `estimated` auto-turns!
* **Transitions:** When `0` transition windows were actually evaluated, the summary claimed `status = "measured"` with `secondary_speed_peaks = 0` and `unintended_reversals = 0`!
* **Coordination Lags:** Whenever paired lags existed, the summary hardcoded `status = "measured"`, even if the reviewer had manually shifted Partner A's start timestamp (`user_confirmed` + `measured` = mixed).
* **Coverage Denominators:** Labels said things like `(120 samples)` without clearly showing `120/240 frames`, and failed to explain why frames were excluded.

#### How we solved it
1. **Uniform evidence-status aggregator (`_summarize_evidence_status` & `SummaryEvidenceStatus`):**
   * Added explicit summary states **`"mixed"`** and **`"insufficient_evidence"`** alongside `"measured"`, `"estimated"`, `"user_confirmed"`, and `"unavailable"`.
   * **Turns:** 1 `user_confirmed` + 4 `estimated` auto-turns now clearly report `status: "mixed"` with `status_counts: {"user_confirmed": 1, "estimated": 4, ...}`.
   * **Coordination:** Each lag (`start_lag_ms`, `slow_lag_ms`, `settle_lag_ms`) inspects **both** paired Partner A and Partner B events! Untouched auto events → `"measured"`; editing one partner → `"mixed"`; confirming both → `"user_confirmed"`.
   * **Step Timing:** Aggregates both contact statuses and matched beat statuses — mixing confirmed contacts with unconfirmed soundtrack beats or mixed contacts reports `"mixed"`.
   * **Transitions:** Reports `evaluated_window_count`, `candidate_window_count`, and `unavailable_window_count`. When `evaluated_window_count == 0`, `status` is **`"unavailable"`** and `secondary_speed_peaks` / `unintended_reversals` are **`None`**, rendered in UI tables as **`Unavailable`** (and blank `""` in CSV exports) rather than a deceptive `0`.
   * **Posture & Pearson `r`:** Expose `eligible_count`, `total_frame_count`, `coverage_denominator`, `status_counts`, and `origin_counts`; `torso_tilt_pearson_r` distinguishes `"insufficient_evidence"` (1–2 paired frames, not enough for correlation) from `"unavailable"` (0 paired frames or flat variance).

---

## 3. What You Will Actually See Differently in TangoPosture Now

Here is exactly what changes on screen, in settings, and in exports:

### Analytics Screen
* **Axis Chart Coverage Captions:** Instead of only seeing `(120 samples)`, captions now show the denominator and evidence status right next to coverage:  
  `Median: +6.8° · IQR: 1.4° · Ref: 0.0° · Coverage: 50% (120/240 frames, measured)`  
  and Pearson details show `Paired samples: 120/240 frames · ... · Status: measured`.
* **Transitions Summary Table:** Includes a new **`evaluated window count`** row. If no transition windows met coverage/duration rules, **`secondary speed peaks`** and **`unintended reversals`** show **`Unavailable`** with status **`unavailable`** instead of a misleading `0.000`.
* **Turns & Exits Summary Table:** Shows **`mixed`** when auto suggestions and manual turns coexist, and appends an **`unmatched count`** row whenever orphan turn markers exist.
* **Turns Events Table:** Displays `[turn_id]` and any `unmatched: <reason>` note in the detail column.

### Review Screen
* **Adding Turn Markers:** Picking `turn_start` -> `turn_end` -> `exit` for `A`, `B`, or `pair` automatically links them under one `[turn-...]` ID. Overlapping A and B turns stay completely separate.
* **Unmatched Marker Visibility:** Orphan starts, ends, or exits appear in `coupleEventsTable` with `[turn-...]` and `unmatched: ...`, plus `(N unmatched)` in the timeline markers bar so you never wonder why a turn didn't calculate.
* **Overlay Rendering & Frame Details:** Close-embrace `propagated` poses still draw smoothly with dashed `(est)` styling so visual feedback stays intact, while frame details display `(eligible/total frames)` and `[status]`.

### Analysis Screen & Markdown Reports
* The post-analysis summary table displays **`Unavailable (0 evaluated windows)`** when no transition windows were evaluated, shows frame denominators on posture rows, and notes unmatched turn counts on the turns row.
* Saved Markdown reports (`render_couple_review_report`) include `{eligible}/{total} frames` denominators, `[status]` tags, `evaluated windows`, and `unmatched markers`.

### Settings (`Couple · Tracking / Interpolation`)
* New setting **`dancer_analysis.pose_origin_policy`**:
  * **`"measured_only"`** *(default)*: Strict scientific accounting — translated propagated poses never enter measured metrics or measured coverage.
  * **`"allow_propagated_estimates"`**: Explicit opt-in for continuous close-embrace clips where you want propagated estimates folded into series/summaries (clearly labeled `estimated` / `mixed` alongside `measured_only_coverage`).

### CSV Exports
* New columns appended to Couple CSV exports (`couple/csv_export.py`): `turn_id`, `pose_origin`, `eligible_count`, `coverage_denominator`, `evaluated_window_count`, `status_counts`.
* Unavailable transition counts export as empty cells (`""`) rather than `"0"`, so spreadsheet pivots never confuse "unmeasured" with "zero cues."

---

## 4. How Much Has This Improved the Experience? (Honest Impact Assessment)

In short: **this is a major leap in scientific honesty and multi-dancer editing reliability**, even though the desktop layout itself didn't get a flashy visual redesign.

| Dimension | Improvement Level | Why it matters in real practice |
| :--- | :--- | :--- |
| **Trust in Numbers** | **Very High** | You will no longer chase phantom "final-frame transition spikes," false "reversals" in multi-direction clips, inflated close-embrace coverage percentages, or misleading `0` counts when nothing could be measured. |
| **Editing Turns & Exits (A / B / Pair)** | **High** | Overlapping leader/follower turns no longer cross-wire in Review, shared pair turns are explicit, and half-finished markers stay visible with exact fix reasons instead of silently vanishing. |
| **Debugging Missing Data** | **High** | Seeing `120/240 frames`, `evaluated window count = 0`, `(1 unmatched)`, or `reason: propagated_pose_excluded` immediately tells you *why* a metric is missing or `mixed`. |
| **Close-Embrace Viewing Experience** | **Better Clarity** | Smooth video overlays are preserved during brief close-embrace splits, while dashed styling + `pose_origin_policy` prevent visual smoothing from masquerading as fresh detector evidence. |
| **Raw MediaPipe Detection Accuracy** | *Unchanged* | These fixes make accounting, pairing, transition math, and status reporting honest — they do not replace the underlying MediaPipe pose detector itself, nor do they implement the remaining P2/P3 backlog items in `TECHNICAL_DEBT_AND_REDESIGN.md` (such as bounding musical-target match distance or separating `couple/measurements.py` into smaller modules). |

---

## 5. Quick Cheat Sheet: What Every Origin & Status Label Means

### Pose Origins (`pose_origin`)
| Label | Plain-English Meaning | Counted in Default `measured_only` Metrics? |
| :--- | :--- | :--- |
| `detected` | Fresh MediaPipe skeleton directly detected on this frame. | **Yes** (as `measured`). |
| `crop_detected` | Fresh MediaPipe skeleton detected by cropping/splitting a merged close-embrace frame. | **Yes** (as `measured`). |
| `propagated` | Translated forward from a nearby split frame without re-running detection. | **No** (`unavailable` by default; `estimated` only under `allow_propagated_estimates`). |
| `display_interpolated` | Filled linearly strictly to draw smoother lines on video. | **Never** (strictly visual only). |

### Summary Evidence Statuses (`status`)
| Label | Plain-English Meaning | What to Do |
| :--- | :--- | :--- |
| `measured` | Built from untouched, eligible detections (`detected` / `crop_detected`). | Safe to treat as direct detector observation (still check coverage!). |
| `estimated` | Built from algorithmic suggestions or permitted `propagated` estimates. | Replay in Review and confirm key events before coaching from it. |
| `user_confirmed` | **All** contributing events/markers in this summary were confirmed or authored by a human reviewer. | Highest reviewer confidence. |
| `mixed` | Combines two or more evidence types (e.g., 1 confirmed turn + 4 automatic turns, or edited A + auto B lag, or `measured` + `propagated` estimates). | Expand the events/frames table or `status_counts` to see which parts were reviewed vs. estimated. |
| `insufficient_evidence` | Some valid points existed (e.g., 1–2 paired frames for Pearson `r`), but too few to compute a reliable statistic. | Treat as not yet calculable; inspect longer/cleaner passages. |
| `unavailable` | Zero usable windows or pairs could be evaluated (e.g., `evaluated_window_count == 0`, occluded landmarks, or excluded propagated frames). | Do **not** read this as "zero problems" — check camera framing, occlusion, or coverage denominators. |
