"""Headless Qt tests for Review measurement normalization UI, routing, and notices."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QTabWidget

from tango_posture.archive import (
    append_segment,
    calibration_directory,
    practice_directory,
    store_analytics,
)
from tango_posture.calibration.archives import save_range_profile
from tango_posture.calibration.baseline import profile as build_capture_profile
from tango_posture.calibration.models import ApprovedRange, Passage, RangeProfile
from tango_posture.calibration.normalization import (
    MeasurementNormalizationConfig,
    MetricNormalizationRule,
    NormalizationPairRule,
    build_source_identity,
    extract_coach_observations,
    make_pair_rule_id,
    profile_content_digest,
    resolve_raw_media_sha256,
)
from tango_posture.config import Config, export_toml
from tango_posture.desktop_qt.calibration.normalization_panel import NullableDoubleEdit
from tango_posture.desktop_qt.router import Route
from tango_posture.desktop_qt.shell import MainWindow
from tango_posture.domain import FrameObservation, Landmark
from tango_posture.models import MANIFEST
from tango_posture.references.coach import _serialise_pose


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    return app


@pytest.fixture(autouse=True)
def isolated_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def _lm(x: float, y: float) -> Landmark:
    return Landmark(x=x, y=y, visibility=0.95, presence=0.95, z=0.0)


def _pose_frame(
    timestamp: float,
    *,
    head_pitch_deg: float = 3.0,
    knee_bend_deg: float = 15.0,
    width: int = 1000,
    height: int = 1000,
) -> FrameObservation:
    ls = _lm(0.50, 0.50)
    rs = _lm(0.50, 0.50)
    lh = _lm(0.50, 0.75)
    rh = _lm(0.50, 0.75)
    le = _lm(0.50, 0.42)
    pitch_rad = math.radians(head_pitch_deg)
    nose = _lm(0.60, 0.42 - 0.10 * math.tan(pitch_rad))

    bend_rad = math.radians(max(0.0, knee_bend_deg))
    lk = _lm(0.50, 0.85)
    rk = _lm(0.50, 0.85)
    la = _lm(0.50 - 0.10 * math.sin(bend_rad), 0.85 + 0.10 * math.cos(bend_rad))
    ra = la
    return FrameObservation(
        timestamp,
        width,
        height,
        [{
            "nose": nose,
            "left_ear": le,
            "left_shoulder": ls,
            "right_shoulder": rs,
            "left_hip": lh,
            "right_hip": rh,
            "left_knee": lk,
            "right_knee": rk,
            "left_ankle": la,
            "right_ankle": ra,
        }],
    )


def _coach_data(cfg: Config, frames: list[FrameObservation]) -> dict:
    entries = {}
    for idx, frame in enumerate(frames):
        entries[str(idx)] = {
            "timestamp": frame.timestamp,
            "poses": [_serialise_pose(p) for p in frame.poses],
            "measurement_poses": [_serialise_pose(p) for p in frame.poses],
        }
    return {
        "model_hash": MANIFEST["sha256"],
        "capture_profile": build_capture_profile(cfg, 1000, 1000, "video"),
        "width": 1000,
        "height": 1000,
        "frames": entries,
    }


def _make_video(path: Path, frame_count: int = 6, frame_seed: int = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for i in range(frame_count):
            writer.write(np.full((48, 64, 3), (frame_seed + i) % 250, np.uint8))
    finally:
        writer.release()
    return path


def _make_calibration(cfg: Config, cal_id: str = "cal-head-pitch", name: str = "Teacher Pitch") -> tuple[Path, RangeProfile]:
    profile = RangeProfile(
        id=cal_id,
        name=name,
        created_at_utc="2026-11-15T12:00:00Z",
        context="Open embrace",
        model_hash=MANIFEST["sha256"],
        capture_profile=build_capture_profile(cfg, 1000, 1000, "video"),
        source={"archive": "teacher_ref.tpr"},
        metrics={
            "head_pitch": ApprovedRange(
                lower=1.0,
                upper=5.0,
                unit="degrees",
                view="side",
                passages=[Passage(start=0.0, end=2.0)],
                median=3.0,
                p10=1.5,
                p90=4.5,
                valid_duration=2.0,
                selected_duration=2.0,
                samples=20,
                inside_fraction=1.0,
                approved=True,
                camera_check_passed=True,
            ),
            "knee_flexion": ApprovedRange(
                lower=10.0,
                upper=25.0,
                unit="degrees",
                view="side",
                passages=[Passage(start=0.0, end=2.0)],
                median=15.0,
                p10=12.0,
                p90=20.0,
                valid_duration=2.0,
                selected_duration=2.0,
                samples=20,
                inside_fraction=1.0,
                approved=True,
                camera_check_passed=True,
            ),
        },
        camera_checks={},
    )
    cal_dir = calibration_directory(cfg.data_dir)
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cal_dir / f"{cal_id}.tpc"
    save_range_profile(cal_path, profile, overwrite=True)
    return cal_path, profile


def _make_practice_archive(
    cfg: Config,
    tmp_path: Path,
    filename: str,
    frames: list[FrameObservation],
    *,
    frame_seed: int = 10,
) -> tuple[Path, dict]:
    prac_dir = practice_directory(cfg.data_dir)
    prac_dir.mkdir(parents=True, exist_ok=True)
    tp_path = prac_dir / filename
    video_path = _make_video(tmp_path / f"{filename}.avi", max(2, len(frames)), frame_seed=frame_seed)

    report_path = tmp_path / f"{filename}.md"
    report_path.write_text("# Practice Report\nObserved posture during practice.\n", encoding="utf-8")
    timestamps = [f.timestamp for f in frames]
    append_segment(tp_path, report_path, video_path, video_path, timestamps)
    data = _coach_data(cfg, frames)
    store_analytics(tp_path, 0, {"coach_data": data})
    return tp_path, data


def _process_until(qapp: QApplication, predicate, timeout_sec: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert predicate()


def test_nullable_double_edit_blank_and_finite_values(qapp) -> None:
    """NullableDoubleEdit represents blank text as None and validates finite floats."""
    edit = NullableDoubleEdit(placeholder="Default")
    try:
        assert edit.value() is None
        edit.setText("   ")
        assert edit.value() is None
        edit.setValue(0.75)
        assert edit.value() == pytest.approx(0.75)
        edit.setValue(None)
        assert edit.value() is None
        assert edit.text() == ""
        edit.setText("2.5")
        assert edit.value() == pytest.approx(2.5)
        edit.setValue(float("nan"))
        assert edit.value() is None
    finally:
        edit.close()


def test_review_to_settings_routing_playhead_preview_async_holds_and_snapshot_cancel(
    qapp, tmp_path, monkeypatch,
) -> None:
    """Routing from Review preselects pair, supports playhead/preview/async holds, and discards stale jobs."""
    cfg = Config(data_dir=str(tmp_path), smoothing=0.01, entry_duration=0.01, clear_duration=0.01)
    window = MainWindow(cfg)
    cal_path, profile = _make_calibration(cfg)

    ref_frames = [
        *[_pose_frame(0.1 * i, head_pitch_deg=3.0) for i in range(10)],
        *[_pose_frame(2.0 + 0.1 * i, head_pitch_deg=3.2) for i in range(10)],
    ]
    prac_frames_a = [
        *[_pose_frame(0.1 * i, head_pitch_deg=7.0) for i in range(10)],
        *[_pose_frame(2.0 + 0.1 * i, head_pitch_deg=7.4) for i in range(10)],
        _pose_frame(4.0, head_pitch_deg=8.0, knee_bend_deg=5.0),
    ]
    prac_frames_b = [_pose_frame(0.1 * i, head_pitch_deg=6.0) for i in range(10)]

    tp_a, data_a = _make_practice_archive(cfg, tmp_path, "session_a.tp", prac_frames_a, frame_seed=15)
    tp_b, _data_b = _make_practice_archive(cfg, tmp_path, "session_b.tp", prac_frames_b, frame_seed=95)

    window.show()
    try:
        review = window.screens[Route.REVIEW]
        settings = window.screens[Route.SETTINGS]
        panel = settings.normalization_panel
        review.refresh()
        settings.refresh()

        # Navigate to Review, load session_a, select profile, mark hold anchors, and inspect frame 20 (t=4.0, 8.0° head pitch)
        window.router.navigate(Route.REVIEW)
        review.load_archive(tp_a)
        review._coach_data = data_a
        cal_idx = review.reference_combo.findData(str(cal_path.resolve()))
        assert cal_idx >= 0
        review.reference_combo.setCurrentIndex(cal_idx)
        review._show_frame(5)  # t = 0.5
        review.mark_hold_start_btn.click()
        assert window.bridge.review_context["anchor_start"] == pytest.approx(0.5, abs=1e-3)
        review._show_frame(12)  # t = 2.2
        review.mark_hold_end_btn.click()
        assert window.bridge.review_context["anchor_end"] == pytest.approx(2.2, abs=1e-3)
        review._show_frame(20)  # t = 4.0, head_pitch = 8.0°
        qapp.processEvents()

        # Click "Measurement normalization settings…" from Review -> lands on Settings "Review & Ollama" tab
        review.normalization_settings_btn.click()
        qapp.processEvents()
        assert window.router.current == Route.SETTINGS
        assert settings.tabs.tabText(settings.tabs.currentIndex()) == "Review & Ollama"
        assert panel.profile_combo.currentData() == profile.id
        assert tuple(panel.practice_combo.currentData()) == (str(tp_a.resolve()), 0)

        # Playhead markers and manual offset / live preview
        head_row = panel.metric_rows["head_pitch"]
        head_row.adjust_radio.setChecked(True)
        head_row.method_combo.setCurrentIndex(head_row.method_combo.findData("matched_holds"))

        head_row.fit_prac_set_start_btn.click()
        assert head_row.fit_prac_start.value() == pytest.approx(4.0, abs=1e-3)
        window.bridge.review_context["playhead_seconds"] = 0.5
        head_row.fit_prac_set_start_btn.click()
        assert head_row.fit_prac_start.value() == pytest.approx(0.5, abs=1e-3)
        window.bridge.review_context["playhead_seconds"] = 2.2
        head_row.check_prac_set_end_btn.click()
        assert head_row.check_prac_end.value() == pytest.approx(2.2, abs=1e-3)

        # Configure full worked-example hold windows on head_pitch & compute asynchronously
        head_row.fit_ref_start.setValue(0.0)
        head_row.fit_ref_end.setValue(1.0)
        head_row.fit_prac_start.setValue(0.0)
        head_row.fit_prac_end.setValue(1.0)
        head_row.check_ref_start.setValue(2.0)
        head_row.check_ref_end.setValue(3.0)
        head_row.check_prac_start.setValue(2.0)
        head_row.check_prac_end.setValue(3.0)

        def _fake_load_frames(
            _cal_id, _cal_path, _profile, prac_path, _seg_idx, _comp_cfg, _override,
        ):
            frames_for_prac = prac_frames_a if Path(prac_path).resolve() == tp_a.resolve() else prac_frames_b
            return ref_frames, {"path": tmp_path / "teacher_ref.tpr", "capture_profile": profile.capture_profile, "raw_media_sha256": "ref-sha"}, frames_for_prac, {"path": prac_path, "capture_profile": data_a["capture_profile"], "raw_media_sha256": resolve_raw_media_sha256(Path(prac_path), 0)}

        monkeypatch.setattr(panel, "_load_frames_for_compute", _fake_load_frames)
        panel.compute_matched_holds_async("head_pitch")
        _process_until(qapp, lambda: "verified" in head_row.status_badge.text().lower())

        assert head_row.offset_spin.value() == pytest.approx(-4.0, abs=1e-4)
        assert "error 0.20" in head_row.status_badge.text()

        # Live preview: Use current Review frame (frame 20: ~8.0° -> ~4.0° inside 1..5), 8.0 -> 4.00°, and 11.0° -> 7.00° outside
        head_row.use_review_frame_btn.click()
        assert head_row.preview_input.value() == pytest.approx(8.0, abs=0.05)
        assert "inside 1..5" in head_row.preview_output_lbl.text()
        assert "(Verified)" in head_row.preview_output_lbl.text()

        head_row.preview_input.setValue(8.0)
        assert "4.00°" in head_row.preview_output_lbl.text()
        assert "inside 1..5" in head_row.preview_output_lbl.text()

        head_row.preview_input.setValue(11.0)
        assert "7.00°" in head_row.preview_output_lbl.text()
        assert "outside 1..5" in head_row.preview_output_lbl.text()

        # Out-of-domain physical domain error on knee_flexion (5.0 + (-10.0) = -5.0°)
        knee_row = panel.metric_rows["knee_flexion"]
        knee_row.adjust_radio.setChecked(True)
        knee_row.offset_spin.setValue(-10.0)
        knee_row.preview_input.setValue(5.0)
        assert "Adjustment unavailable" in knee_row.preview_output_lbl.text()

        # Snapshot cancellation / stale-result discard when switching practice_combo mid-flight
        entered = threading.Event()
        release = threading.Event()

        def _slow_load(*args, **kwargs):
            entered.set()
            release.wait(timeout=2.0)
            return _fake_load_frames(*args, **kwargs)

        monkeypatch.setattr(panel, "_load_frames_for_compute", _slow_load)
        panel.compute_matched_holds_async("head_pitch")
        assert entered.wait(timeout=2.0)
        # Switch recording to session_b before the worker emits finished
        b_idx = next(
            (i for i in range(panel.practice_combo.count()) if tuple(panel.practice_combo.itemData(i) or ()) == (str(tp_b.resolve()), 0)),
            -1,
        )
        assert b_idx >= 0
        panel.practice_combo.setCurrentIndex(b_idx)
        release.set()
        if panel._compute_thread is not None:
            panel._compute_thread.join(timeout=2.0)
        qapp.processEvents()

        # Newly selected session_b must stay untouched on Direct comparison / 0.0 offset
        b_head = panel.metric_rows["head_pitch"]
        assert b_head.direct_radio.isChecked()
        assert b_head.offset_spin.value() == pytest.approx(0.0, abs=1e-6)

        # Reset adjustment & Remove comparison on session_a
        a_idx = next(
            (i for i in range(panel.practice_combo.count()) if tuple(panel.practice_combo.itemData(i) or ()) == (str(tp_a.resolve()), 0)),
            -1,
        )
        panel.practice_combo.setCurrentIndex(a_idx)
        head_row.reset_btn.click()
        assert head_row.direct_radio.isChecked()
        assert head_row.offset_spin.value() == pytest.approx(0.0, abs=1e-6)

        head_row.adjust_radio.setChecked(True)
        head_row.offset_spin.setValue(-4.0)
        panel.remove_pair_btn.click()
        collected = panel.measurement_normalization_config()
        assert collected.pairs == []
    finally:
        window.close()


def test_settings_toml_import_valid_and_invalid_normalization(qapp, tmp_path, monkeypatch) -> None:
    """Invalid normalization TOML fails cleanly without mutating config; valid TOML updates config."""
    cfg = Config(data_dir=str(tmp_path))
    window = MainWindow(cfg)
    _make_calibration(cfg)
    frames = [_pose_frame(0.0, head_pitch_deg=8.0)]
    tp_path, data = _make_practice_archive(cfg, tmp_path, "import_prac.tp", frames)
    media_sha = resolve_raw_media_sha256(tp_path, 0)

    window.show()
    try:
        settings = window.screens[Route.SETTINGS]
        settings.refresh()
        original_dump = window.bridge.config.model_dump()

        # 1. Invalid TOML (missing window end / wrong field) leaves config untouched
        bad_toml = tmp_path / "bad_norm.toml"
        bad_toml.write_text(
            "[measurement_normalization]\nenabled = true\nmin_hold_samples = 0\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "tango_posture.desktop_qt.screens.settings.QFileDialog.getOpenFileName",
            lambda *args: (str(bad_toml), "TOML (*.toml)"),
        )
        settings._import_toml()
        assert window.bridge.config.model_dump() == original_dump
        assert window.status_strip._status_label.text().startswith("Could not import settings:")

        # 2. Valid TOML import updates bridge.config.measurement_normalization and panel
        valid_cfg = cfg.model_copy(update={
            "measurement_normalization": MeasurementNormalizationConfig(
                enabled=True,
                pairs=[
                    NormalizationPairRule(
                        id=make_pair_rule_id("cal-head-pitch", media_sha or "x"),
                        name="Valid Import Pair",
                        enabled=True,
                        calibration_id="cal-head-pitch",
                        profile_content_digest="abc",
                        practice_source=build_source_identity(
                            archive_path=tp_path,
                            segment_index=0,
                            cfg=cfg,
                            observations=frames,
                            capture_profile=data["capture_profile"],
                            raw_media_sha256=media_sha,
                        ),
                        metrics={
                            "head_pitch": MetricNormalizationRule(enabled=True, method="manual", offset=-4.0),
                        },
                    )
                ],
            )
        })
        good_toml = tmp_path / "good_norm.toml"
        good_toml.write_text(export_toml(valid_cfg), encoding="utf-8")
        monkeypatch.setattr(
            "tango_posture.desktop_qt.screens.settings.QFileDialog.getOpenFileName",
            lambda *args: (str(good_toml), "TOML (*.toml)"),
        )
        settings._import_toml()
        assert window.bridge.config.measurement_normalization.enabled is True
        assert len(window.bridge.config.measurement_normalization.pairs) == 1
        assert window.bridge.config.measurement_normalization.pairs[0].metrics["head_pitch"].offset == pytest.approx(-4.0)
        assert settings.normalization_panel.enable_cb.isChecked()
    finally:
        window.close()


def test_review_and_coach_normalization_notices_and_stale_recheck_fallback(
    qapp, tmp_path,
) -> None:
    """Verify unadjusted text, normalized Expanded Insights + Report notice, stale recheck fallback, and Coach banner."""
    cfg = Config(data_dir=str(tmp_path), smoothing=0.01, entry_duration=0.01, clear_duration=0.01)
    window = MainWindow(cfg)
    cal_path, profile = _make_calibration(cfg)
    prac_frames = [
        _pose_frame(0.0, head_pitch_deg=8.0),
        _pose_frame(0.05, head_pitch_deg=8.0),
    ]
    tp_path, data = _make_practice_archive(cfg, tmp_path, "notice_prac.tp", prac_frames, frame_seed=45)
    media_sha = resolve_raw_media_sha256(tp_path, 0)
    prac_id = build_source_identity(
        archive_path=tp_path,
        segment_index=0,
        cfg=cfg,
        observations=extract_coach_observations(data),
        capture_profile=data["capture_profile"],
        raw_media_sha256=media_sha,
    )
    pair = NormalizationPairRule(
        id=make_pair_rule_id(profile.id, media_sha or "x"),
        name="Review Pair",
        enabled=True,
        calibration_id=profile.id,
        calibration_name=profile.name,
        profile_content_digest=profile_content_digest(profile),
        practice_source=prac_id,
        metrics={
            "head_pitch": MetricNormalizationRule(enabled=True, method="manual", offset=-4.0),
        },
    )

    window.show()
    try:
        review = window.screens[Route.REVIEW]
        review.refresh()
        window.router.navigate(Route.REVIEW)
        review.load_archive(tp_path)
        review._coach_data = data
        cal_idx = review.reference_combo.findData(str(cal_path.resolve()))
        review.reference_combo.setCurrentIndex(cal_idx)
        review._show_frame(1)
        qapp.processEvents()

        # 1. Unadjusted comparison -> original scale notice in Expanded Insights, no report banner
        unadjusted_text = review.insights_text.text()
        assert "Original calibration scale · no adjustment applied" in unadjusted_text
        assert review.report_normalization_notice.isHidden()
        assert review.normalization_recheck_bar.isHidden()

        # 2. Enable matching normalization pair -> Insights shows Raw/Adjustment/Reference scale/Used for assessment
        norm_cfg = cfg.model_copy(update={
            "measurement_normalization": MeasurementNormalizationConfig(enabled=True, pairs=[pair])
        })
        window.bridge.update_config(norm_cfg)
        review._update_calibration_comparison()
        review._show_frame(1)
        qapp.processEvents()

        adjusted_text = review.insights_text.text()
        assert "Raw 8.0°" in adjusted_text
        assert "Adjustment −4.0°" in adjusted_text
        assert "Reference scale 4.0°" in adjusted_text
        assert "Used for assessment (smoothed)" in adjusted_text
        assert not review.report_normalization_notice.isHidden()
        for tab_widget in review.findChildren(QTabWidget):
            report_tab = review.report_normalization_notice.parentWidget()
            if report_tab is not None and tab_widget.indexOf(report_tab) >= 0:
                tab_widget.setCurrentWidget(report_tab)
        qapp.processEvents()
        assert review.report_normalization_notice.isVisible()
        assert review.normalization_recheck_bar.isHidden()

        # 3. Open in AI Coach while normalized -> Coach notice visible
        review._open_report_in_coach()
        qapp.processEvents()
        coach = window.screens[Route.COACH]
        assert window.router.current == Route.COACH
        assert coach.normalization_notice_lbl.isVisible()
        assert "original unadjusted analysis scale" in coach.normalization_notice_lbl.text().lower()

        # 4. Stale pair (profile digest mismatch) -> shows recheck bar; "Use original scale" reverts cleanly
        window.router.navigate(Route.REVIEW)
        review._coach_data = data
        cal_idx = review.reference_combo.findData(str(cal_path.resolve()))
        review.reference_combo.setCurrentIndex(cal_idx)
        stale_pair = pair.model_copy(update={"profile_content_digest": "outdated-digest"})
        stale_cfg = cfg.model_copy(update={
            "measurement_normalization": MeasurementNormalizationConfig(enabled=True, pairs=[stale_pair])
        })
        window.bridge.update_config(stale_cfg)
        review._update_calibration_comparison()
        review._show_frame(1)
        review.inspector_tabs.setCurrentIndex(0)
        qapp.processEvents()

        assert review.normalization_recheck_bar.isVisible()
        assert "Needs recheck" in review.insights_text.text()
        assert "Not applied (showing original calibration scale)" in review.insights_text.text()

        review.recheck_use_original_btn.click()
        review._show_frame(1)
        qapp.processEvents()
        assert "Original calibration scale · no adjustment applied" in review.insights_text.text()
        assert review.normalization_recheck_bar.isHidden()
        assert review.report_normalization_notice.isHidden()
    finally:
        window.close()
