"""Qt integration tests for Auto/Single/Couple workflows across Analysis, Library, Review, and Analytics."""

from __future__ import annotations

import csv
import io
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from tango_posture.archive import append_segment, file_sha256, read_analytics, store_analytics
from tango_posture.config import Config
from tango_posture.couple import (
    CoupleAnalysisRequest,
    CoupleAnnotationItem,
    CoupleAnnotations,
    run_couple_analysis,
)
from tango_posture.couple.models import AutoDecisionResult, CandidatePreview
from tango_posture.desktop_qt.components.analysis_options import (
    AnalysisOptionsWidget,
    NeedsSelectionDialog,
)
from tango_posture.desktop_qt.router import Route
from tango_posture.desktop_qt.screens.analysis import AnalysisScreen
from tango_posture.desktop_qt.shell import MainWindow
from tango_posture.domain import FrameObservation, Landmark
from tango_posture.references.results import (
    CalibrationReadiness,
    ClipAnalysisResult,
    RunManifest,
    SoundtrackAnalysisSummary,
)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(["-platform", "offscreen"])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("Qt Couple tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def _pose(hip_x: float, shoulder_dx: float = 0.0, torso_dy: float = 0.24) -> dict[str, Landmark]:
    sh_x = hip_x + shoulder_dx
    sh_y = 0.62 - torso_dy
    return {
        "nose": Landmark(sh_x, sh_y - 0.06, 0.95, 0.95, 0.0),
        "left_ear": Landmark(sh_x - 0.015, sh_y - 0.06, 0.95, 0.95, 0.0),
        "right_ear": Landmark(sh_x + 0.015, sh_y - 0.06, 0.95, 0.95, 0.0),
        "left_shoulder": Landmark(sh_x - 0.04, sh_y, 0.95, 0.95, 0.0),
        "right_shoulder": Landmark(sh_x + 0.04, sh_y, 0.95, 0.95, 0.0),
        "left_hip": Landmark(hip_x - 0.03, 0.62, 0.95, 0.95, 0.0),
        "right_hip": Landmark(hip_x + 0.03, 0.62, 0.95, 0.95, 0.0),
        "left_knee": Landmark(hip_x - 0.03, 0.76, 0.95, 0.95, 0.0),
        "right_knee": Landmark(hip_x + 0.03, 0.76, 0.95, 0.95, 0.0),
        "left_ankle": Landmark(hip_x - 0.03, 0.90, 0.95, 0.95, 0.0),
        "right_ankle": Landmark(hip_x + 0.03, 0.90, 0.95, 0.95, 0.0),
    }


def _write_video(path: Path, *, fps: float = 20.0, duration_s: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 160, 120
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (w, h))
    assert writer.isOpened()
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    for _ in range(round(fps * duration_s)):
        writer.write(frame)
    writer.release()
    return path


def _build_couple_result(video_sha256: str, cfg: Config, *, run_id: str = "couple-qt-run-1") -> dict:
    obs_list = []
    fps = 20.0
    for i in range(40):
        t = round(i / fps, 4)
        xa = 0.25 if t < 0.5 else min(0.60, 0.25 + 0.16 * (t - 0.5))
        xb = 0.35 if t < 0.6 else min(0.70, 0.35 + 0.16 * (t - 0.6))
        obs_list.append(
            (i, FrameObservation(t, 160, 120, [_pose(xa, shoulder_dx=0.025), _pose(xb, shoulder_dx=-0.025)]))
        )
    anns = CoupleAnnotations(
        dancer_a_name="Alice",
        dancer_a_role="Lead",
        dancer_b_name="Bruno",
        dancer_b_role="Follow",
        items=[
            CoupleAnnotationItem("b1", "musical_target", 1.00, subject="beat", status="user_confirmed"),
            CoupleAnnotationItem("c1", "contact", 1.03, subject="A", status="user_confirmed"),
            CoupleAnnotationItem("ts1", "turn_start", 1.20, subject="pair", status="user_confirmed"),
            CoupleAnnotationItem("te1", "turn_end", 1.70, subject="pair", status="user_confirmed"),
            CoupleAnnotationItem("ex1", "exit", 1.75, subject="pair", status="user_confirmed"),
        ],
    )
    return run_couple_analysis(
        obs_list,
        cfg,
        clip_in=0.0,
        clip_out=2.0,
        request=CoupleAnalysisRequest(requested_mode="couple", forced_override=True, annotations=anns),
        video_sha256=video_sha256,
        run_id=run_id,
    ).to_dict()


def _make_archive_with_runs(
    archive_path: Path,
    cfg: Config,
    *,
    include_single: bool = True,
    include_couple: bool = True,
    couple_run_id: str = "couple-qt-run-1",
) -> tuple[Path, str, list[float]]:
    raw = _write_video(archive_path.parent / f"{archive_path.stem}_raw.avi", fps=20.0, duration_s=2.0)
    report = archive_path.parent / f"{archive_path.stem}_report.md"
    report.write_text("# Session\n", encoding="utf-8")
    timestamps = [round(i / 20.0, 4) for i in range(40)]
    append_segment(archive_path, report, raw, None, timestamps)
    vsha = file_sha256(raw)

    base_analytics = {
        "model_hash": "qt-couple-test",
        "capture_profile": "default",
        "max_gap": cfg.max_gap,
        "frame_timestamps": timestamps,
        "frames": [
            {"timestamp": t, "values": {"head_forward": 0.1}, "alerts": []}
            for t in timestamps
        ],
        "coach_data": {"width": 160, "height": 120, "frames": {}},
        "movement_data": {"steps": [], "pivots": []},
    }
    if include_single:
        store_analytics(
            archive_path,
            0,
            base_analytics,
            expected_video_sha256=vsha,
            dancer_analysis_run={
                "run_id": f"single-{archive_path.stem}",
                "requested_mode": "single",
                "resolved_mode": "single",
                "coach_data": base_analytics["coach_data"],
                "movement_data": base_analytics["movement_data"],
                "session": {"frames": len(timestamps)},
                "frames": base_analytics["frames"],
            },
        )
    if include_couple:
        couple_res = _build_couple_result(vsha, cfg, run_id=couple_run_id)
        store_analytics(
            archive_path,
            0,
            base_analytics,
            expected_video_sha256=vsha,
            dancer_analysis_run={
                "run_id": couple_run_id,
                "requested_mode": "auto",
                "resolved_mode": "couple",
                "auto_decision": couple_res.get("auto_decision"),
                "couple_result": couple_res,
            },
        )
    elif not include_single:
        store_analytics(archive_path, 0, base_analytics, expected_video_sha256=vsha)
    return archive_path, vsha, timestamps


def test_analysis_options_widget_and_needs_selection_dialog(qapp):
    cfg = Config()
    widget = AnalysisOptionsWidget(cfg)
    idx_couple = widget.mode_combo.findData("couple")
    widget.mode_combo.setCurrentIndex(idx_couple)
    widget.use_region_cb.setChecked(True)
    widget.x_min_spin.setValue(0.15)
    widget.soundtrack_cb.setChecked(True)
    widget.auto_turns_cb.setChecked(True)

    req = widget.build_request()
    assert req.requested_mode == "couple"
    assert req.analysis_region is not None and req.analysis_region["x_min"] == pytest.approx(0.15)
    assert req.analyze_soundtrack is True
    assert req.auto_turn_suggestions is True

    dec = AutoDecisionResult(
        detector_version="couple_auto_v1",
        requested_mode="auto",
        resolved_mode="needs_selection",
        forced_override=False,
        configured_window_s=10.0,
        effective_window_in=0.0,
        effective_window_out=10.0,
        effective_duration_s=10.0,
        primary_eligible_duration_s=8.0,
        primary_coverage=0.8,
        primary_moving_duration_s=4.0,
        second_coexistence_duration_s=5.0,
        second_joint_coverage=0.5,
        max_continuous_paired_span_s=1.0,
        max_co_movement_span_s=0.5,
        reason="ambiguous_coexistence",
        explanation="Opening window shows a second person without clear pair confirmation.",
        candidate_previews=[
            CandidatePreview(
                candidate_id="A",
                pose_index=0,
                timestamp=1.0,
                frame_index=10,
                hip_xy=(0.3, 0.6),
                shoulder_xy=(0.3, 0.36),
                torso_length_px=24.0,
                bbox=(0.2, 0.25, 0.4, 0.85),
                eligible_duration_s=8.0,
                coverage=0.8,
            )
        ],
    ).to_dict()
    dlg = NeedsSelectionDialog(dec)
    dlg._pin_candidate("A", dec["candidate_previews"][0])
    dlg._accept_mode("couple")
    applied = dlg.apply_to_request(CoupleAnalysisRequest(requested_mode="auto"))
    assert applied is not None
    assert applied.requested_mode == "couple"
    assert applied.forced_override is True
    assert len(applied.annotations.anchors) == 1


def test_analysis_screen_needs_selection_override_and_cancel_leaves_archive_untouched(
    qapp, tmp_path, monkeypatch
):
    cfg = Config(data_dir=str(tmp_path / "data"))
    window = MainWindow(cfg)
    arch, vsha, timestamps = _make_archive_with_runs(
        tmp_path / "data" / "practice" / "Studio.tp",
        cfg,
        include_single=True,
        include_couple=False,
    )
    before_bytes = arch.read_bytes()

    window.router.navigate(Route.ANALYSIS, arch)
    qapp.processEvents()
    assert window.nav_buttons[Route.ANALYSIS].isChecked()
    screen: AnalysisScreen = window.screens[Route.ANALYSIS]  # type: ignore[assignment]

    def _fake_result(resolved: str, *, selection_req: bool, couple_dict: dict | None = None) -> ClipAnalysisResult:
        return ClipAnalysisResult(
            manifest=RunManifest(
                model_variant="lite",
                model_hash="qt-couple-test",
                metric_version="v1",
                capture_profile={},
                source_sha256=vsha,
                source_filename="raw.avi",
                width=160,
                height=120,
                fps=10.0,
                selected_in=0.0,
                selected_out=2.0,
                clip_duration=2.0,
                analysis_mode="movement",
                requested_dancer_mode="auto" if selection_req else "couple",
                resolved_dancer_mode="needs_selection" if selection_req else resolved,
            ),
            video_path=arch,
            title="Studio",
            source_sha256=vsha,
            selected_in=0.0,
            selected_out=2.0,
            clip_duration=2.0,
            analysis_mode="movement",
            readiness=CalibrationReadiness(True, "movement", "Ready"),
            reference=None,
            calibration_window=None,
            coach_data={"width": 160, "height": 120, "frames": {}},
            movement_data={"steps": [], "pivots": []},
            soundtrack=SoundtrackAnalysisSummary(False, "none", 0.0, 0, []),
            coverage={},
            frame_log=[{"timestamp": t} for t in timestamps],
            scored_frame_log=[{"timestamp": t, "values": {}, "alerts": []} for t in timestamps],
            frame_observations=[],
            performance={},
            requested_dancer_mode="auto" if selection_req else "couple",
            resolved_dancer_mode="needs_selection" if selection_req else resolved,
            auto_decision={"requested_mode": "auto", "resolved_mode": resolved},
            couple_result=couple_dict,
            selection_required=selection_req,
        )

    calls: list[dict] = []
    couple_dict = _build_couple_result(vsha, cfg, run_id="couple-override-run")

    def fake_analyze_clip(**kwargs):
        req = CoupleAnalysisRequest.from_dict(kwargs.get("dancer_request"))
        calls.append(req.to_dict())
        if not req.forced_override:
            return _fake_result("needs_selection", selection_req=True)
        return _fake_result("couple", selection_req=False, couple_dict=couple_dict)

    monkeypatch.setattr("tango_posture.desktop_qt.screens.analysis.analyze_clip", fake_analyze_clip)

    # Case A: Cancel NeedsSelectionDialog leaves archive untouched
    class _RejectDialog(NeedsSelectionDialog):
        def exec(self) -> int:
            return NeedsSelectionDialog.DialogCode.Rejected

    monkeypatch.setattr("tango_posture.desktop_qt.screens.analysis.NeedsSelectionDialog", _RejectDialog)
    screen._on_run_clicked()
    deadline = time.monotonic() + 5.0
    while screen.has_active_jobs() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    assert arch.read_bytes() == before_bytes
    assert "untouched" in screen.status_label.text()

    # Case B: Accept NeedsSelectionDialog as Couple -> forced override saves Couple run
    class _AcceptCoupleDialog(NeedsSelectionDialog):
        def exec(self) -> int:
            self.selected_mode = "couple"
            return NeedsSelectionDialog.DialogCode.Accepted

    monkeypatch.setattr("tango_posture.desktop_qt.screens.analysis.NeedsSelectionDialog", _AcceptCoupleDialog)
    screen._on_run_clicked()
    deadline = time.monotonic() + 5.0
    while (
        screen.has_active_jobs()
        or len(calls) < 2
        or screen._last_result is None
        or screen._last_result.resolved_dancer_mode != "couple"
    ) and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()

    assert len(calls) >= 2 and calls[-1].get("forced_override") is True
    assert "Saved Couple" in screen.status_label.text()
    saved = read_analytics(arch, 0)
    assert saved is not None and "couple" in saved["dancer_analysis"]["runs"]
    assert "single" in saved["dancer_analysis"]["runs"]
    window.close()


def test_library_badges_and_analyze_routing(qapp, tmp_path):
    cfg = Config(data_dir=str(tmp_path / "data"))
    practice_dir = Path(cfg.data_dir) / "practice"
    dual_arch, _, _ = _make_archive_with_runs(
        practice_dir / "Dual.tp", cfg, include_single=True, include_couple=True
    )
    _couple_only, _, _ = _make_archive_with_runs(
        practice_dir / "DuoOnly.tp", cfg, include_single=False, include_couple=True, couple_run_id="duo-only-1"
    )

    window = MainWindow(cfg)
    window.router.navigate(Route.LIBRARY)
    qapp.processEvents()
    lib = window.screens[Route.LIBRARY]

    types_by_name = {
        lib.table.item(r, 0).text(): lib.table.item(r, 4).text()
        for r in range(lib.table.rowCount())
        if lib.table.item(r, 0) and lib.table.item(r, 4)
    }
    assert "Single + Couple" in types_by_name["Dual"]
    assert "Couple" in types_by_name["DuoOnly"]

    # Click "Analyze" on Dual row -> routes to Route.ANALYSIS
    dual_row = next(r for r in range(lib.table.rowCount()) if lib.table.item(r, 0).text() == "Dual")
    actions = lib.table.cellWidget(dual_row, 3)
    assert actions is not None
    analyze_btn = next(b for b in actions.findChildren(QPushButton) if b.text() == "Analyze")
    analyze_btn.click()
    qapp.processEvents()
    assert window.router.current is Route.ANALYSIS
    assert window.screens[Route.ANALYSIS].archive_combo.currentData() == dual_arch
    window.close()


def test_review_couple_evidence_edit_undo_save_and_seek(qapp, tmp_path):
    cfg = Config(data_dir=str(tmp_path / "data"))
    practice_dir = Path(cfg.data_dir) / "practice"
    arch, _, _ = _make_archive_with_runs(
        practice_dir / "ReviewCouple.tp",
        cfg,
        include_single=True,
        include_couple=True,
        couple_run_id="rev-run-777",
    )

    window = MainWindow(cfg)
    window.router.navigate(
        Route.REVIEW,
        {"archive_path": arch, "segment_index": 0, "run_id": "rev-run-777", "timestamp": 1.20},
    )
    qapp.processEvents()
    review = window.screens[Route.REVIEW]

    assert review.current_time_sec == pytest.approx(1.20, abs=0.06)
    assert review.view_run_combo.isEnabled()
    assert review.view_run_combo.currentData() == "couple"
    assert review.dancer_a_name_edit.text() == "Alice"
    assert review.couple_events_table.rowCount() >= 1

    # Edit dancer label, undo back to "Alice", then re-edit and mark planned transition + save atomically
    orig_candidates_len = len(review._couple_result["automated_candidates"])
    review.dancer_a_name_edit.setText("Camila")
    review._apply_couple_names()
    assert review._couple_result["annotations"]["dancer_a_name"] == "Camila"
    assert review.couple_undo_btn.isEnabled()

    review._undo_couple_annotations()
    assert review._couple_result["annotations"]["dancer_a_name"] == "Alice"

    review.dancer_a_name_edit.setText("Valentina")
    review._apply_couple_names()
    review._mark_planned_transition("planned_reversal")
    review._save_couple_annotations()
    qapp.processEvents()

    stored = read_analytics(arch, 0)
    assert stored is not None
    saved_couple = stored["dancer_analysis"]["runs"]["couple"]["couple_result"]
    assert saved_couple["revision_id"] >= 2
    assert saved_couple["annotations"]["dancer_a_name"] == "Valentina"
    assert len(saved_couple["annotations"]["planned_transitions"]) == 1
    assert len(saved_couple["automated_candidates"]) == orig_candidates_len
    window.close()


def test_analytics_mode_filter_five_panes_seek_and_csv(qapp, tmp_path):
    cfg = Config(data_dir=str(tmp_path / "data"))
    practice_dir = Path(cfg.data_dir) / "practice"
    dual_arch, _, _ = _make_archive_with_runs(
        practice_dir / "MixDual.tp",
        cfg,
        include_single=True,
        include_couple=True,
        couple_run_id="analytics-couple-99",
    )
    _make_archive_with_runs(
        practice_dir / "SoloOnly.tp",
        cfg,
        include_single=True,
        include_couple=False,
    )

    window = MainWindow(cfg)
    window.router.navigate(Route.ANALYTICS)
    qapp.processEvents()
    analytics = window.screens[Route.ANALYTICS]

    # Mode filter: "All" shows both, "Couple" shows only MixDual
    all_labels = [
        analytics.session_combo.itemText(i)
        for i in range(analytics.session_combo.count())
        if analytics.session_combo.itemData(i) is not None
    ]
    assert len(all_labels) == 2

    idx_couple = analytics.mode_filter_combo.findData("couple")
    analytics.mode_filter_combo.setCurrentIndex(idx_couple)
    qapp.processEvents()
    couple_labels = [
        analytics.session_combo.itemText(i)
        for i in range(analytics.session_combo.count())
        if analytics.session_combo.itemData(i) is not None
    ]
    assert len(couple_labels) == 1
    assert "MixDual" in couple_labels[0]

    mix_idx = next(
        i
        for i in range(analytics.session_combo.count())
        if "MixDual" in analytics.session_combo.itemText(i)
    )
    if analytics.session_combo.currentIndex() != mix_idx:
        analytics.session_combo.setCurrentIndex(mix_idx)
        qapp.processEvents()

    # Verify all 5 panes exist and hold Couple summary / event rows
    assert analytics.couple_panes.count() == 5
    pane_titles = [analytics.couple_panes.tabText(i) for i in range(5)]
    assert pane_titles == [
        "Coordination",
        "Step and beat timing",
        "Transitions",
        "Posture and axes",
        "Turns and exits",
    ]
    assert analytics._couple_summary_tables["coordination"].rowCount() >= 1
    assert analytics._couple_summary_tables["step_beat"].rowCount() >= 1
    assert analytics._couple_summary_tables["transitions"].rowCount() >= 1
    assert analytics._couple_summary_tables["posture_axes"].rowCount() >= 1
    assert analytics._couple_summary_tables["turns_exits"].rowCount() >= 1
    assert analytics._couple_event_tables["turns_exits"].rowCount() >= 1

    # Click-to-seek from turns_exits event table routes to Review with run_id + timestamp
    seen_nav: list[tuple[Route, object]] = []
    orig_nav = window.router.navigate

    def capture_nav(route: Route, payload: object = None):
        seen_nav.append((route, payload))
        return orig_nav(route, payload)

    window.router.navigate = capture_nav  # type: ignore[method-assign]
    turns_tbl = analytics._couple_event_tables["turns_exits"]
    analytics._on_couple_event_row_clicked(turns_tbl, 0)
    qapp.processEvents()

    assert seen_nav and seen_nav[-1][0] is Route.REVIEW
    nav_payload = seen_nav[-1][1]
    assert isinstance(nav_payload, dict)
    assert nav_payload["archive_path"] == dual_arch
    assert nav_payload["run_id"] == "analytics-couple-99"
    assert isinstance(nav_payload["timestamp"], float)

    # CSV export produces RFC-4180 CSV with summary + event rows
    csv_out = tmp_path / "exported_couple.csv"
    exported = analytics.export_couple_csv_file(csv_out)
    assert exported == csv_out and csv_out.is_file()
    rows = list(csv.DictReader(io.StringIO(csv_out.read_text(encoding="utf-8"))))
    assert any(r["row_type"] == "summary" and r["run_id"] == "analytics-couple-99" for r in rows)
    assert any(r["row_type"] == "event" and r["run_id"] == "analytics-couple-99" for r in rows)
    window.close()


def test_video_analysis_local_couple_saves_tpr_archive(qapp, tmp_path, monkeypatch):
    from tango_posture.archive import read_manifest
    from tango_posture.references.archives import reference_directory

    cfg = Config(data_dir=str(tmp_path / "data"))
    local_vid = _write_video(tmp_path / "LocalCouple.avi", fps=20.0, duration_s=2.0)
    vsha = file_sha256(local_vid)
    timestamps = [round(i / 20.0, 4) for i in range(40)]
    couple_dict = _build_couple_result(vsha, cfg, run_id="local-couple-run-1")

    res = ClipAnalysisResult(
        manifest=RunManifest(
            model_variant="lite",
            model_hash="qt-local-couple",
            metric_version="v1",
            capture_profile={"width": 160, "height": 120, "side": "left", "facing": "right"},
            source_sha256=vsha,
            source_filename=local_vid.name,
            width=160,
            height=120,
            fps=20.0,
            selected_in=0.0,
            selected_out=2.0,
            clip_duration=2.0,
            analysis_mode="movement",
            requested_dancer_mode="couple",
            resolved_dancer_mode="couple",
        ),
        video_path=local_vid,
        title="LocalCouple",
        source_sha256=vsha,
        selected_in=0.0,
        selected_out=2.0,
        clip_duration=2.0,
        analysis_mode="movement",
        readiness=CalibrationReadiness(True, "movement", "Ready"),
        reference=None,
        calibration_window=None,
        coach_data={
            "width": 160,
            "height": 120,
            "frames": {
                str(i): {"timestamp": timestamps[i], "poses": [], "measurement_poses": []}
                for i in range(40)
            },
        },
        movement_data={"steps": [], "pivots": []},
        soundtrack=SoundtrackAnalysisSummary(enabled=False, available=False, status="disabled"),
        coverage={},
        frame_log=[{"frame_index": i, "timestamp": timestamps[i]} for i in range(40)],
        scored_frame_log=[{"frame_index": i, "timestamp": timestamps[i]} for i in range(40)],
        frame_observations=[],
        performance={},
        requested_dancer_mode="couple",
        resolved_dancer_mode="couple",
        auto_decision={"requested_mode": "couple", "resolved_mode": "couple", "explanation": "Couple"},
        couple_result=couple_dict,
        selection_required=False,
    )
    monkeypatch.setattr("tango_posture.desktop_qt.screens.analysis.analyze_clip", lambda **_kw: res)

    window = MainWindow(cfg)
    try:
        window.router.navigate(Route.ANALYSIS)
        qapp.processEvents()
        assert window.nav_buttons[Route.ANALYSIS].text() == "Video Analysis"
        assert window.nav_buttons[Route.ANALYTICS].text() == "Analytics"

        screen: AnalysisScreen = window.screens[Route.ANALYSIS]  # type: ignore[assignment]
        screen.local_radio.setChecked(True)
        screen._set_local_video(local_vid)
        idx_couple = screen.options_widget.mode_combo.findData("couple")
        screen.options_widget.mode_combo.setCurrentIndex(idx_couple)
        screen._on_run_clicked()

        deadline = time.monotonic() + 10.0
        while (screen.has_active_jobs() or screen._last_archive_path is None) and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.02)
        qapp.processEvents()

        tprs = list(reference_directory(cfg.data_dir).glob("*.tpr"))
        assert len(tprs) == 1
        saved_tpr = tprs[0]
        assert screen._last_archive_path == saved_tpr
        assert screen.open_review_btn.isEnabled()
        assert screen.open_analytics_btn.isEnabled()
        assert ".tpr" in screen.status_label.text()
        manifest = read_manifest(saved_tpr)
        assert manifest["kind"] == "reference"
        saved_analytics = read_analytics(saved_tpr, 0)
        assert saved_analytics is not None
        assert saved_analytics["dancer_analysis"]["active_mode"] == "couple"
        assert "couple" in saved_analytics["dancer_analysis"]["runs"]
        labels_in_results = [
            screen.results_table.item(r, 0).text()
            for r in range(screen.results_table.rowCount())
        ]
        assert "Torso tilt Pearson r (A vs B)" in labels_in_results
    finally:
        window.close()


def test_review_couple_local_video_saves_tpr_even_when_hidden(qapp, tmp_path, monkeypatch):
    import threading

    from tango_posture.archive import read_manifest
    from tango_posture.desktop_qt.review.analysis import ReviewAnalysis
    from tango_posture.references.archives import reference_directory

    cfg = Config(data_dir=str(tmp_path / "data"))
    local_vid = _write_video(tmp_path / "ReviewCoupleLocal.avi", fps=20.0, duration_s=2.0)
    vsha = file_sha256(local_vid)
    timestamps = [round(i / 20.0, 4) for i in range(40)]
    couple_dict = _build_couple_result(vsha, cfg, run_id="rev-local-couple-1")
    started = threading.Event()
    proceed = threading.Event()

    res = ClipAnalysisResult(
        manifest=RunManifest(
            model_variant="lite",
            model_hash="qt-rev-couple",
            metric_version="v1",
            capture_profile={"width": 160, "height": 120, "side": "left", "facing": "right"},
            source_sha256=vsha,
            source_filename=local_vid.name,
            width=160,
            height=120,
            fps=20.0,
            selected_in=0.0,
            selected_out=2.0,
            clip_duration=2.0,
            analysis_mode="movement",
            requested_dancer_mode="couple",
            resolved_dancer_mode="couple",
        ),
        video_path=local_vid,
        title="ReviewCoupleLocal",
        source_sha256=vsha,
        selected_in=0.0,
        selected_out=2.0,
        clip_duration=2.0,
        analysis_mode="movement",
        readiness=CalibrationReadiness(True, "movement", "Ready"),
        reference=None,
        calibration_window=None,
        coach_data={
            "width": 160,
            "height": 120,
            "frames": {
                str(i): {"timestamp": timestamps[i], "poses": [], "measurement_poses": []}
                for i in range(40)
            },
        },
        movement_data={"steps": [], "pivots": []},
        soundtrack=SoundtrackAnalysisSummary(enabled=False, available=False, status="disabled"),
        coverage={},
        frame_log=[{"frame_index": i, "timestamp": timestamps[i]} for i in range(40)],
        scored_frame_log=[{"frame_index": i, "timestamp": timestamps[i]} for i in range(40)],
        frame_observations=[],
        performance={},
        requested_dancer_mode="couple",
        resolved_dancer_mode="couple",
        auto_decision={"requested_mode": "couple", "resolved_mode": "couple", "explanation": "Couple"},
        couple_result=couple_dict,
        selection_required=False,
    )

    def fake_review_clip(_path, _cfg, **_kwargs):
        started.set()
        assert proceed.wait(5.0)
        return ReviewAnalysis(
            res.coach_data,
            res.movement_data,
            res.manifest,
            res,
            requested_dancer_mode="couple",
            resolved_dancer_mode="couple",
            auto_decision=res.auto_decision,
            couple_result=couple_dict,
        )

    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.review.analyse_review_clip", fake_review_clip
    )

    window = MainWindow(cfg)
    try:
        window.router.navigate(Route.REVIEW, local_vid)
        qapp.processEvents()
        review = window.screens[Route.REVIEW]
        idx_couple = review.dancer_mode_combo.findData("couple")
        review.dancer_mode_combo.setCurrentIndex(idx_couple)

        review._start_posture_analysis()
        assert started.wait(5.0)

        # Switching screens while analysis runs must not cancel it
        window.router.navigate(Route.HOME)
        qapp.processEvents()
        assert review.has_active_jobs()

        proceed.set()
        deadline = time.monotonic() + 10.0
        while review.has_active_jobs() and time.monotonic() < deadline:
            qapp.processEvents()
            review._poll_posture_analysis()
            time.sleep(0.02)
        qapp.processEvents()
        review._poll_posture_analysis()

        tprs = list(reference_directory(cfg.data_dir).glob("*.tpr"))
        assert len(tprs) == 1
        saved_tpr = tprs[0]
        assert review.current_archive_path == saved_tpr
        manifest = read_manifest(saved_tpr)
        assert manifest["kind"] == "reference"
        assert manifest["segments"][0]["overlay"] == "overlay_00.mp4"
        assert manifest["segments"][0]["lines"] == "lines_00.mp4"
        saved_analytics = read_analytics(saved_tpr, 0)
        assert saved_analytics is not None
        assert saved_analytics["dancer_analysis"]["active_mode"] == "couple"
        assert "couple" in saved_analytics["dancer_analysis"]["runs"]
    finally:
        window.close()


def test_library_import_and_analyze_couple_video_creates_tpr_archive(qapp, tmp_path, monkeypatch):
    from tango_posture.archive import read_manifest
    from tango_posture.references.archives import reference_directory
    from tango_posture.references.pipeline import analyse_imported_clip

    cfg = Config(data_dir=str(tmp_path / "data"))
    local_vid = _write_video(tmp_path / "LibraryCouple.avi", fps=15.0, duration_s=1.2)

    class _FakeCoupleEstimator:
        def estimate(self, image, ts):
            return FrameObservation(
                ts,
                image.shape[1],
                image.shape[0],
                [_pose(0.32, shoulder_dx=0.025), _pose(0.56, shoulder_dx=-0.020)],
            )

        def close(self) -> None:
            return None

    orig_import = analyse_imported_clip
    monkeypatch.setattr(
        "tango_posture.references.worker.analyse_imported_clip",
        lambda **kw: orig_import(**{**kw, "estimator_factory": _FakeCoupleEstimator}),
    )

    window = MainWindow(cfg)
    try:
        window.router.navigate(Route.LIBRARY)
        qapp.processEvents()
        lib = window.screens[Route.LIBRARY]
        lib.import_reference_cb.setChecked(True)
        lib._start_import(
            local_vid,
            0.0,
            1.2,
            "Library Couple Import",
            dancer_request={"requested_mode": "couple", "forced_override": True},
            analyse_soundtrack=False,
        )

        deadline = time.monotonic() + 10.0
        while lib.has_active_jobs() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.02)
        qapp.processEvents()

        tprs = list(reference_directory(cfg.data_dir).glob("*.tpr"))
        assert len(tprs) == 1
        saved_tpr = tprs[0]
        manifest = read_manifest(saved_tpr)
        assert manifest["kind"] == "reference"
        saved_analytics = read_analytics(saved_tpr, 0)
        assert saved_analytics is not None
        assert saved_analytics["dancer_analysis"]["active_mode"] == "couple"
        assert "couple" in saved_analytics["dancer_analysis"]["runs"]
    finally:
        window.close()


def test_analytics_couple_axis_charts_populate_seek_and_export(qapp, tmp_path):
    cfg = Config(data_dir=str(tmp_path / "data"))
    practice_dir = Path(cfg.data_dir) / "practice"
    dual_arch, _, _ = _make_archive_with_runs(
        practice_dir / "AxesCouple.tp",
        cfg,
        include_single=False,
        include_couple=True,
        couple_run_id="axes-couple-42",
    )

    window = MainWindow(cfg)
    try:
        window.router.navigate(Route.ANALYTICS)
        qapp.processEvents()
        analytics = window.screens[Route.ANALYTICS]
        axes_idx = next(
            i
            for i in range(analytics.session_combo.count())
            if "AxesCouple" in analytics.session_combo.itemText(i)
        )
        analytics.session_combo.setCurrentIndex(axes_idx)
        qapp.processEvents()

        expected_keys = (
            "torso_tilt_a",
            "torso_tilt_b",
            "couple_midpoint_tilt",
            "torso_separation",
        )
        assert set(analytics.couple_axis_charts) >= set(expected_keys)
        for key in expected_keys:
            chart = analytics.couple_axis_charts[key]
            assert chart.isEnabled(), f"Expected couple axis chart {key} to be enabled"
            assert chart.reference_line == pytest.approx(0.0)
            assert analytics.couple_axis_export_buttons[key].isEnabled()
            assert "Median:" in analytics.couple_axis_coverage_labels[key].text()

        assert "Alice" in analytics.couple_axis_title_labels["torso_tilt_a"].text()
        assert "Bruno" in analytics.couple_axis_title_labels["torso_tilt_b"].text()

        # Clicking / seeking from a couple axis chart routes to Review with run_id + timestamp
        seen_nav: list[tuple[Route, object]] = []
        orig_nav = window.router.navigate

        def capture_nav(route: Route, payload: object = None):
            seen_nav.append((route, payload))
            return orig_nav(route, payload)

        window.router.navigate = capture_nav  # type: ignore[method-assign]
        analytics.couple_axis_charts["torso_tilt_a"].seek_requested.emit(0.75)
        qapp.processEvents()

        assert seen_nav and seen_nav[-1][0] is Route.REVIEW
        nav_payload = seen_nav[-1][1]
        assert isinstance(nav_payload, dict)
        assert nav_payload["archive_path"] == dual_arch
        assert nav_payload["run_id"] == "axes-couple-42"
        assert nav_payload["timestamp"] == pytest.approx(0.75)

        out_png = tmp_path / "torso_tilt_a.png"
        out_svg = tmp_path / "couple_midpoint_tilt.svg"
        assert analytics.export_couple_axis_chart("torso_tilt_a", out_png) == out_png
        assert out_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert analytics.export_couple_axis_chart("couple_midpoint_tilt", out_svg) == out_svg
        assert "<svg" in out_svg.read_text(encoding="utf-8")
    finally:
        window.close()


def test_couple_axis_smoothing_reduces_jitter_and_updates_when_settings_saved(
    qapp, tmp_path, monkeypatch
):
    import tomllib

    from tango_posture.config import startup_config_path
    from tango_posture.couple.overlay import smooth_couple_frame_overlays

    monkeypatch.chdir(tmp_path)
    cfg = Config(data_dir=str(tmp_path / "data"), smoothing=0.05, overlay_smoothing=0.05)

    # 1. Overlay smoothing removes high-frequency alternating jitter in torso tilts & midpoints
    jitter_overlays = [
        {
            "frame_index": i,
            "timestamp": round(i * 0.05, 4),
            "tilt_a_deg": 8.0 if i % 2 == 0 else -8.0,
            "tilt_b_deg": -6.0 if i % 2 == 0 else 6.0,
            "couple_midpoint_tilt_deg": 5.0 if i % 2 == 0 else -5.0,
            "torso_separation_deg": 14.0 if i % 2 == 0 else 2.0,
            "pair_hip_norm": [0.40 + (0.04 if i % 2 == 0 else -0.04), 0.60],
            "pair_shoulder_norm": [0.40 + (0.04 if i % 2 == 0 else -0.04), 0.36],
        }
        for i in range(16)
    ]
    smoothed_overlays = smooth_couple_frame_overlays(jitter_overlays, tau_s=0.25, max_gap_s=0.20)
    raw_deltas = [
        abs(jitter_overlays[i + 1]["tilt_a_deg"] - jitter_overlays[i]["tilt_a_deg"])
        for i in range(len(jitter_overlays) - 1)
    ]
    sm_deltas = [
        abs(smoothed_overlays[i + 1]["tilt_a_deg"] - smoothed_overlays[i]["tilt_a_deg"])
        for i in range(len(smoothed_overlays) - 1)
    ]
    assert max(sm_deltas) < max(raw_deltas) * 0.25

    # 2. Analytics Couple Axis charts smooth alternating jitter and refresh when Settings saves config.toml
    raw_vid = _write_video(tmp_path / "JitterRaw.avi", fps=20.0, duration_s=1.0)
    arch = Path(cfg.data_dir) / "practice" / "JitterCouple.tp"
    timestamps = [round(i * 0.05, 4) for i in range(20)]
    report = tmp_path / "jitter_rep.md"
    report.write_text("# Jitter\n", encoding="utf-8")
    append_segment(arch, report, raw_vid, None, timestamps)
    vsha = file_sha256(raw_vid)

    couple_res = _build_couple_result(vsha, cfg, run_id="jitter-couple-1")
    couple_res["series"]["torso_tilt_a"] = [
        {
            "timestamp": ts,
            "value": 10.0 if idx % 2 == 0 else -10.0,
            "status": "measured",
            "reason": "",
        }
        for idx, ts in enumerate(timestamps)
    ]
    store_analytics(
        arch,
        0,
        {
            "model_hash": "qt-jitter-test",
            "capture_profile": "default",
            "max_gap": cfg.max_gap,
            "frame_timestamps": timestamps,
            "frames": [],
            "coach_data": {"width": 160, "height": 120, "frames": {}},
            "movement_data": {"steps": [], "pivots": []},
        },
        expected_video_sha256=vsha,
        dancer_analysis_run={
            "run_id": "jitter-couple-1",
            "requested_mode": "couple",
            "resolved_mode": "couple",
            "auto_decision": couple_res.get("auto_decision"),
            "couple_result": couple_res,
        },
    )

    window = MainWindow(cfg)
    try:
        window.router.navigate(Route.ANALYTICS)
        qapp.processEvents()
        analytics = window.screens[Route.ANALYTICS]
        idx = next(
            i for i in range(analytics.session_combo.count())
            if "JitterCouple" in analytics.session_combo.itemText(i)
        )
        analytics.session_combo.setCurrentIndex(idx)
        qapp.processEvents()

        chart_a = analytics.couple_axis_charts["torso_tilt_a"]
        vals_initial = [v for _, v in chart_a.data.points if v is not None]
        spread_initial = max(vals_initial) - min(vals_initial)
        # Already much smoother than raw 20 deg alternating peak-to-peak swing
        assert spread_initial < 14.0

        # Save stronger smoothing in Settings -> writes config.toml and refreshes Analytics live
        window.router.navigate(Route.SETTINGS)
        qapp.processEvents()
        settings = window.screens[Route.SETTINGS]
        settings.smooth_spin.setValue(0.65)
        settings.save_config_btn.click()
        qapp.processEvents()

        cfg_file = tmp_path / "config.toml"
        assert cfg_file.is_file()
        assert tomllib.loads(cfg_file.read_text(encoding="utf-8"))["smoothing"] == pytest.approx(0.65)
        assert startup_config_path(None) is not None

        vals_after = [v for _, v in analytics.couple_axis_charts["torso_tilt_a"].data.points if v is not None]
        spread_after = max(vals_after) - min(vals_after)
        assert spread_after < spread_initial
    finally:
        window.close()


def _pixmap_rgb_array(pixmap) -> np.ndarray:
    from PySide6.QtGui import QImage

    img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    raw = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes())
    return raw.reshape(img.height(), img.bytesPerLine())[:, : img.width() * 3].reshape(
        img.height(), img.width(), 3
    ).copy()


def test_analysis_screen_draws_xy_region_lines_on_preview_update(qapp, tmp_path):
    cfg = Config(data_dir=str(tmp_path / "data"))
    local_vid = _write_video(tmp_path / "RegionPreview.avi", fps=20.0, duration_s=0.5)

    window = MainWindow(cfg)
    try:
        window.router.navigate(Route.ANALYSIS)
        qapp.processEvents()
        screen: AnalysisScreen = window.screens[Route.ANALYSIS]  # type: ignore[assignment]

        # 1. Even before a video frame is loaded, toggling region draws X/Y lines on a synthetic preview
        assert screen._preview_frame_bgr is None
        screen.options_widget.use_region_cb.setChecked(True)
        qapp.processEvents()
        pm_synth = screen.preview_label.pixmap()
        assert pm_synth is not None and not pm_synth.isNull()
        rgb_synth = _pixmap_rgb_array(pm_synth)
        # Active box / cyan X&Y lines create brighter non-uniform pixels over dark (28,28,28) background
        assert int(rgb_synth.max()) > 150

        screen.options_widget.use_region_cb.setChecked(False)
        qapp.processEvents()
        pm_cleared = screen.preview_label.pixmap()
        assert pm_cleared is None or pm_cleared.isNull()

        # 2. With a real local video frame loaded, toggling use_region_cb and changing X/Y updates live
        screen.local_radio.setChecked(True)
        screen._set_local_video(local_vid)
        qapp.processEvents()

        pm_base = screen.preview_label.pixmap()
        assert pm_base is not None and not pm_base.isNull()
        rgb_base = _pixmap_rgb_array(pm_base)

        screen.options_widget.use_region_cb.setChecked(True)
        screen.options_widget.x_min_spin.setValue(0.20)
        screen.options_widget.y_min_spin.setValue(0.25)
        screen.options_widget.x_max_spin.setValue(0.80)
        screen.options_widget.y_max_spin.setValue(0.75)
        qapp.processEvents()

        rgb_first = _pixmap_rgb_array(screen.preview_label.pixmap())
        assert not np.array_equal(rgb_base, rgb_first)

        # Exterior top-left pixel (outside [0.20..0.80, 0.25..0.75]) is dimmed compared to interior
        assert int(rgb_first[2, 2].sum()) < int(rgb_base[2, 2].sum())

        # Full-span vertical X line near x=0.20 and horizontal Y line near y=0.25 show drawn cyan lines
        h, w = rgb_first.shape[:2]
        x_col_first = round(0.20 * (w - 1))
        y_row_first = round(0.25 * (h - 1))
        band_first = rgb_first[4, max(0, x_col_first - 1) : min(w, x_col_first + 2)]
        y_band_first = rgb_first[max(0, y_row_first - 1) : min(h, y_row_first + 2), 4]
        assert int(band_first.max()) > int(rgb_first[4, 2].max()) + 80
        assert int(y_band_first.max()) > int(rgb_first[2, 4].max()) + 80

        # Moving X min / Y max shifts the drawn lines immediately
        screen.options_widget.x_min_spin.setValue(0.35)
        screen.options_widget.y_max_spin.setValue(0.60)
        qapp.processEvents()

        rgb_second = _pixmap_rgb_array(screen.preview_label.pixmap())
        assert not np.array_equal(rgb_first, rgb_second)
        x_col_second = round(0.35 * (w - 1))
        band_second = rgb_second[4, max(0, x_col_second - 1) : min(w, x_col_second + 2)]
        assert int(band_second.max()) > int(rgb_second[4, 2].max()) + 80
        # Previously drawn line at x=0.20 is no longer lit at top margin (now dimmed exterior)
        old_band_after = rgb_second[4, max(0, x_col_first - 1) : min(w, x_col_first + 2)]
        assert int(old_band_after.max()) < int(band_first.max()) - 80
    finally:
        window.close()


def test_couple_analytics_pearson_r_and_couple_settings_parameters(qapp, tmp_path, monkeypatch):
    import tomllib

    monkeypatch.chdir(tmp_path)
    cfg = Config(data_dir=str(tmp_path / "data"))
    raw_vid = _write_video(tmp_path / "PearsonRaw.avi", fps=20.0, duration_s=1.0)
    arch = Path(cfg.data_dir) / "practice" / "PearsonCouple.tp"
    timestamps = [round(i * 0.05, 4) for i in range(20)]
    report = tmp_path / "pearson_rep.md"
    report.write_text("# Pearson\n", encoding="utf-8")
    append_segment(arch, report, raw_vid, None, timestamps)
    vsha = file_sha256(raw_vid)

    couple_res = _build_couple_result(vsha, cfg, run_id="pearson-couple-1")
    couple_res["series"]["torso_tilt_a"] = [
        {"timestamp": ts, "value": float(idx - 10) + (2.0 if idx % 2 == 0 else -2.0), "status": "measured", "reason": ""}
        for idx, ts in enumerate(timestamps)
    ]
    couple_res["series"]["torso_tilt_b"] = [
        {"timestamp": ts, "value": float(idx - 10) * 1.5, "status": "measured", "reason": ""}
        for idx, ts in enumerate(timestamps)
    ]
    store_analytics(
        arch,
        0,
        {
            "model_hash": "qt-pearson-test",
            "capture_profile": "default",
            "max_gap": cfg.max_gap,
            "frame_timestamps": timestamps,
            "frames": [],
            "coach_data": {"width": 160, "height": 120, "frames": {}},
            "movement_data": {"steps": [], "pivots": []},
        },
        expected_video_sha256=vsha,
        dancer_analysis_run={
            "run_id": "pearson-couple-1",
            "requested_mode": "couple",
            "resolved_mode": "couple",
            "auto_decision": couple_res.get("auto_decision"),
            "couple_result": couple_res,
        },
    )

    window = MainWindow(cfg)
    try:
        window.router.navigate(Route.ANALYTICS)
        qapp.processEvents()
        analytics = window.screens[Route.ANALYTICS]
        idx = next(
            i for i in range(analytics.session_combo.count())
            if "PearsonCouple" in analytics.session_combo.itemText(i)
        )
        analytics.session_combo.setCurrentIndex(idx)
        qapp.processEvents()

        # Pearson correlation shown in dedicated prominent card, banner, coverage label, both tables, and CSV
        assert analytics.couple_pearson_value_label.text().startswith("Pearson r = +")
        assert "Paired samples:" in analytics.couple_pearson_detail_label.text()
        assert "In-phase parallel sway" in analytics.couple_pearson_detail_label.text()
        assert "Torso tilt Pearson r:" in analytics.couple_run_info_label.text()
        assert "Pearson r(A,B):" in analytics.couple_axis_coverage_labels["torso_tilt_a"].text()
        coord_tbl = analytics._couple_summary_tables["coordination"]
        assert any(
            coord_tbl.item(row, 0).text() == "torso tilt pearson r"
            for row in range(coord_tbl.rowCount())
        )
        posture_tbl = analytics._couple_summary_tables["posture_axes"]
        r_row = next(
            row for row in range(posture_tbl.rowCount())
            if posture_tbl.item(row, 0).text() == "torso tilt pearson r"
        )
        initial_r = float(posture_tbl.item(r_row, 1).text())
        assert initial_r > 0.85

        csv_out = tmp_path / "pearson.csv"
        analytics.export_couple_csv_file(csv_out)
        csv_rows = list(csv.DictReader(io.StringIO(csv_out.read_text(encoding="utf-8"))))
        assert any(
            row["row_type"] == "summary" and row["measure"] == "torso_tilt_pearson_r" and row["unit"] == "r"
            for row in csv_rows
        )

        # Legacy archive with coach_data.couple_analysis (without dancer_analysis) also loads Pearson r
        legacy_arch = Path(cfg.data_dir) / "practice" / "LegacyCoachCouple.tp"
        append_segment(legacy_arch, report, raw_vid, None, timestamps)
        store_analytics(
            legacy_arch,
            0,
            {
                "model_hash": "qt-legacy-couple",
                "capture_profile": "default",
                "max_gap": cfg.max_gap,
                "frame_timestamps": timestamps,
                "frames": [],
                "coach_data": {"width": 160, "height": 120, "frames": {}, "couple_analysis": couple_res},
                "movement_data": {"steps": [], "pivots": []},
            },
            expected_video_sha256=vsha,
        )
        analytics.refresh()
        legacy_idx = next(
            i for i in range(analytics.session_combo.count())
            if "LegacyCoachCouple" in analytics.session_combo.itemText(i)
        )
        analytics.session_combo.setCurrentIndex(legacy_idx)
        qapp.processEvents()
        assert analytics.couple_pearson_value_label.text().startswith("Pearson r = +")
        analytics.session_combo.setCurrentIndex(
            next(i for i in range(analytics.session_combo.count()) if "PearsonCouple" in analytics.session_combo.itemText(i))
        )
        qapp.processEvents()

        # Updating Couple parameters in Settings persists to config.toml and refreshes live Pearson r
        window.router.navigate(Route.SETTINGS)
        qapp.processEvents()
        settings = window.screens[Route.SETTINGS]
        settings.couple_axis_smooth_spin.setValue(0.80)
        settings.field_widgets["dancer_analysis.shot_cut_jump_torso_lengths"].setValue(2.10)
        settings.save_config_btn.click()
        qapp.processEvents()

        saved_toml = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
        assert saved_toml["dancer_analysis"]["axis_smoothing_s"] == pytest.approx(0.80)
        assert saved_toml["dancer_analysis"]["shot_cut_jump_torso_lengths"] == pytest.approx(2.10)

        r_row_after = next(
            row for row in range(posture_tbl.rowCount())
            if posture_tbl.item(row, 0).text() == "torso tilt pearson r"
        )
        after_r = float(posture_tbl.item(r_row_after, 1).text())
        assert after_r >= initial_r
    finally:
        window.close()
