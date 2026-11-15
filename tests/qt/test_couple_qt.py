"""Qt integration tests for Auto/Single/Couple workflows across Analysis, Library, Review, and Analytics."""

from __future__ import annotations

import csv
import io
import socket
import time
from pathlib import Path
from zipfile import ZipFile

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
                title="Studio",
                source_filename="raw.avi",
                source_sha256=vsha,
                clip_in=0.0,
                clip_out=2.0,
                analysis_mode="movement",
                model_variant="lite",
                model_hash="qt-couple-test",
                capture_profile="default",
                requested_dancer_mode="auto",
                resolved_dancer_mode=resolved,
            ),
            title="Studio",
            video_path=arch,
            source_sha256=vsha,
            selected_in=0.0,
            selected_out=2.0,
            clip_duration=2.0,
            analysis_mode="movement",
            readiness=CalibrationReadiness(True, "movement", "Ready"),
            reference=None,
            movement_data={"steps": [], "pivots": []},
            coach_data={"width": 160, "height": 120, "frames": {}},
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
        req_dict = kwargs.get("dancer_request") or {}
        calls.append(req_dict)
        if not req_dict.get("forced_override"):
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
    while (screen.has_active_jobs() or not screen.open_analytics_btn.isEnabled()) and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()

    assert len(calls) >= 2 and calls[-1].get("forced_override") is True
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
    couple_only, _, _ = _make_archive_with_runs(
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
    all_labels = [analytics.session_combo.itemText(i) for i in range(analytics.session_combo.count())]
    assert len(all_labels) == 2

    idx_couple = analytics.mode_filter_combo.findData("couple")
    analytics.mode_filter_combo.setCurrentIndex(idx_couple)
    qapp.processEvents()
    couple_labels = [analytics.session_combo.itemText(i) for i in range(analytics.session_combo.count())]
    assert len(couple_labels) == 1
    assert "MixDual" in couple_labels[0]

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
