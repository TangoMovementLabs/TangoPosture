"""Qt Library discovers persisted reports and opens source files without copying them."""

import re
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton

from tango_posture.archive import append_segment, read_manifest, save_calibration_archive
from tango_posture.calibration.archives import save_range_profile
from tango_posture.calibration.baseline import profile as build_capture_profile
from tango_posture.calibration.models import ApprovedRange, Passage, RangeProfile
from tango_posture.config import Config
from tango_posture.desktop_qt.router import Route, Router
from tango_posture.desktop_qt.screens.library import LibraryScreen, ProfileDetailsDialog
from tango_posture.desktop_qt.service_bridge import ServiceBridge
from tango_posture.models import MANIFEST
from tango_posture.references.archives import promote_reference
from tango_posture.references.errors import ImportCancelledError
from tango_posture.references.results import ImportedClipResult


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(["-platform", "offscreen"])


def test_library_lists_archives_and_opens_them_in_review(qapp, tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    report = reports / "report_20261112_120000.md"
    report.write_text("# Saved report\n", encoding="utf-8")
    archive = tmp_path / "practice" / "practice.tp"
    append_segment(archive, report, None, None, [])
    router = Router()
    navigations = []
    router.route_changed.connect(lambda route, payload: navigations.append((route, payload)))
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), router)

    assert screen.table.rowCount() == 1
    assert screen.table.item(0, 0).text() == "practice"
    assert screen.table.item(0, 1).text() == "1"
    open_button = screen.table.cellWidget(0, 3).findChildren(QPushButton)[0]
    open_button.click()
    assert navigations[-1] == (Route.REVIEW, archive.resolve())
    screen.close()


def test_library_has_no_legacy_migration_control(qapp, tmp_path):
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    assert all(
        button.text() != "Migrate Legacy Data"
        for button in screen.findChildren(QPushButton)
    )
    screen.close()


def test_cancelled_video_selection_does_not_navigate(qapp, tmp_path, monkeypatch):
    router = Router()
    navigations = []
    router.route_changed.connect(lambda route, payload: navigations.append((route, payload)))
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), router)
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.library.QFileDialog.getOpenFileName",
        lambda *args: ("", ""),
    )

    screen._import_video()

    assert navigations == []
    screen.close()


def test_import_success_navigates_to_generated_archive(qapp, tmp_path):
    router = Router()
    navigations = []
    router.route_changed.connect(lambda route, payload: navigations.append((route, payload)))
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), router)
    archive_path = tmp_path / "practice" / "imported.tp"
    screen._import_job_id = "job-1"
    screen._set_import_busy(True)

    screen._handle_import_event(("archive_done", "job-1", archive_path))

    assert navigations[-1] == (Route.REVIEW, archive_path)
    assert "indexed analytics" in screen.import_status.text()
    assert screen.import_btn.isEnabled()
    assert screen.cancel_import_btn.isEnabled() is False
    screen.close()


def test_rename_persists_manifest_name_without_breaking_path(qapp, tmp_path, monkeypatch):
    report = tmp_path / "report.md"
    report.write_text("# Report\n")
    path = tmp_path / "practice" / "original.tp"
    append_segment(path, report, None, None, [])
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.library.QInputDialog.getText",
        lambda *args, **kwargs: ("My renamed practice", True),
    )
    screen._rename(path, "original")
    assert path.is_file()
    assert read_manifest(path)["name"] == "My renamed practice"
    second = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    assert second.table.item(0, 0).text() == "My renamed practice"
    screen.close()
    second.close()


@pytest.mark.parametrize("suffix", [".tp", ".tpr", ".tpc"])
def test_delete_requires_confirmation_and_uses_trash(
    qapp, tmp_path, monkeypatch, reference, suffix
):
    report = tmp_path / "report.md"
    report.write_text("# Report\n")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture video")
    path = tmp_path / "practice" / "practice.tp"
    append_segment(path, report, video, None, [])
    if suffix == ".tpr":
        destination = tmp_path / "references" / "reference.tpr"
        promote_reference(path, destination, "Reference")
        path = destination
    elif suffix == ".tpc":
        path = tmp_path / "calibrations" / "calibration.tpc"
        save_calibration_archive(path, reference, video, video, [])
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    trash = tmp_path / "trash"
    trash.mkdir()
    moved = []

    def move_to_trash(filename):
        source = Path(filename)
        moved.append(str(source))
        source.replace(trash / source.name)
        return True

    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.library.QFile",
        lambda filename: SimpleNamespace(
            moveToTrash=lambda: move_to_trash(filename),
            errorString=lambda: "Permission denied",
        ),
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No,
    )
    screen._delete(path)
    assert not moved
    assert path.is_file()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes,
    )
    row = next(i for i, item in enumerate(screen.practices) if item["path"] == path.resolve())
    delete = next(
        button for button in screen.table.cellWidget(row, 3).findChildren(QPushButton)
        if button.text() == "Delete"
    )
    delete.click()
    assert moved == [str(path)]
    assert not path.exists()
    assert (trash / path.name).is_file()
    assert all(item["path"] != path.resolve() for item in screen.practices)
    assert "moved to Trash" in screen.import_status.text()
    assert report.is_file()
    (trash / path.name).replace(path)
    screen.refresh()
    monkeypatch.setattr(
        "tango_posture.desktop_qt.screens.library.QFile",
        lambda filename: SimpleNamespace(
            moveToTrash=lambda: False, errorString=lambda: "Permission denied",
        ),
    )
    screen._delete(path)
    assert "Could not move" in screen.import_status.text()
    assert "Permission denied" in screen.import_status.text()
    assert path.is_file()
    screen.close()


def test_library_type_filter_combines_with_search_and_survives_refresh(qapp, tmp_path, monkeypatch):
    bridge = ServiceBridge(Config(data_dir=str(tmp_path)))
    entries = [
        {"name": name, "path": tmp_path / f"{name}{suffix}",
         "segments_count": 0 if suffix == ".tpc" else 1, "created_at": ""}
        for suffix in (".tpr", ".tpc", ".tp")
        for name in ("Walking", "Turn")
    ]
    monkeypatch.setattr(bridge, "list_library_archives", lambda: entries)
    screen = LibraryScreen(bridge, Router())
    assert screen.table.rowCount() == 6
    for suffix in (".tpr", ".tpc", ".tp"):
        screen.type_filter.setCurrentIndex(screen.type_filter.findData(suffix))
        assert screen.table.rowCount() == 2
        assert all(screen.table.item(row, 4).text() == suffix for row in range(2))
        screen.search_input.setText(" WALK ")
        assert screen.table.rowCount() == 1
        assert screen.table.item(0, 0).text() == "Walking"
        screen.refresh()
        assert screen.type_filter.currentData() == suffix
        assert screen.table.rowCount() == 1
        screen.search_input.setText("missing")
        assert screen.table.rowCount() == 0
        screen.search_input.clear()
    screen.type_filter.setCurrentIndex(0)
    assert screen.table.rowCount() == 6
    screen.close()


@pytest.mark.parametrize("accepted", [False, True])
def test_library_picker_only_offers_practice_and_reference_archives(
    qapp, tmp_path, reference, monkeypatch, accepted
):
    report = tmp_path / "report.md"
    report.write_text("# Report\n")
    practice = tmp_path / "practice" / "practice.tp"
    append_segment(practice, report, None, None, [])
    video = tmp_path / "loose-video.mp4"
    video.write_bytes(b"fixture")
    calibration = tmp_path / "calibrations" / "reference.tpc"
    save_calibration_archive(calibration, reference, video, video, [0.0])
    choices = []
    def choose(parent, title, label, items, *args):
        choices.extend(items)
        return items[-1], accepted
    monkeypatch.setattr("PySide6.QtWidgets.QInputDialog.getItem", choose)
    bridge = ServiceBridge(Config(data_dir=str(tmp_path)))
    selected = bridge.choose_library_archive(None, "Open archive")
    assert selected == (calibration.resolve() if accepted else None)
    assert len(choices) == 2
    assert all(".mp4" not in label and ".md" not in label for label in choices)


def test_closing_library_cancels_owned_packaging_thread(qapp, tmp_path, monkeypatch):
    started = threading.Event()
    cancelled = threading.Event()
    def package(path, cfg, cancel_check, *, as_reference=False):
        started.set()
        while not cancel_check():
            cancelled.wait(0.01)
        cancelled.set()
        raise ImportCancelledError("Cancelled")
    monkeypatch.setattr("tango_posture.desktop_qt.screens.library.archive_report", package)
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    result = object.__new__(ImportedClipResult)
    object.__setattr__(result, "report_path", tmp_path / "report.md")
    screen._import_job_id = "test-package"
    screen._handle_import_event(("import_done", "test-package", result))
    assert started.wait(1)
    screen.close()
    assert cancelled.is_set()
    assert not screen._package_thread.is_alive()


def test_started_import_can_be_cancelled_and_recovers_controls(qapp, tmp_path, monkeypatch):
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    started = []
    cancelled = []
    monkeypatch.setattr(
        screen._import_worker,
        "start",
        lambda **kwargs: started.append(kwargs) or "job-1",
    )
    monkeypatch.setattr(
        screen._import_worker, "cancel", lambda: cancelled.append(True) or True
    )

    screen._start_import(Path("dance.mp4"), 2.0, 12.0, "Practice clip")

    assert started[0]["video_path"] == Path("dance.mp4")
    assert started[0]["selected_in"] == 2.0
    assert started[0]["selected_out"] == 12.0
    assert started[0]["analysis_mode"] == "movement"
    assert started[0]["analyse_soundtrack"] is False
    assert screen.import_btn.isEnabled() is False
    screen._cancel_import()
    assert cancelled == [True]

    screen._handle_import_event(("import_cancelled", "job-1"))

    assert screen.import_status.text() == "Import cancelled. You can select another video."
    assert screen.import_btn.isEnabled()
    assert screen.cancel_import_btn.isEnabled() is False
    screen.close()


def test_unexpected_import_event_recovers_controls(qapp, tmp_path):
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    screen._import_job_id = "job-1"
    screen._set_import_busy(True)

    screen._handle_import_event(("unexpected", "job-1"))

    assert "unexpected event" in screen.import_status.text()
    assert screen.import_btn.isEnabled()
    screen.close()


def test_library_has_no_online_download_or_live_calibration_buttons(qapp, tmp_path):
    screen = LibraryScreen(ServiceBridge(Config(data_dir=str(tmp_path))), Router())
    labels = {button.text() for button in screen.findChildren(QPushButton)}
    assert "Import YouTube / Facebook…" not in labels
    assert "Guided camera calibration…" not in labels
    assert not hasattr(screen, "url_import_btn")
    assert not hasattr(screen, "calibration_btn")
    screen.close()


def test_library_formats_dates_and_sorts_by_name_and_date(qapp, tmp_path, monkeypatch):
    bridge = ServiceBridge(Config(data_dir=str(tmp_path)))
    entries = [
        {
            "name": "Zebra",
            "path": tmp_path / "zebra.tp",
            "segments_count": 1,
            "created_at": "2026-11-10T09:15:00",
        },
        {
            "name": "Alpha",
            "path": tmp_path / "alpha.tp",
            "segments_count": 2,
            "created_at": "2026-11-14T18:45:00",
        },
        {
            "name": "Medio",
            "path": tmp_path / "medio.tp",
            "segments_count": 1,
            "created_at": "2026-11-12T12:05:00Z",
        },
    ]
    monkeypatch.setattr(bridge, "list_library_archives", lambda: entries)
    screen = LibraryScreen(bridge, Router())
    try:
        # Default sort is date_desc (newest first)
        assert [screen.table.item(r, 0).text() for r in range(3)] == ["Alpha", "Medio", "Zebra"]
        assert screen.table.item(0, 2).text() == "14/11/2026 18:45"
        assert screen.table.item(2, 2).text() == "10/11/2026 09:15"
        expected_medio = (
            datetime.fromisoformat("2026-11-12T12:05:00+00:00")
            .astimezone()
            .strftime("%d/%m/%Y %H:%M")
        )
        assert screen.table.item(1, 2).text() == expected_medio
        for r in range(3):
            assert re.fullmatch(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", screen.table.item(r, 2).text())

        # Sort by Name A-Z then toggle header click to Name Z-A
        screen.sort_combo.setCurrentIndex(screen.sort_combo.findData("name_asc"))
        assert [screen.table.item(r, 0).text() for r in range(3)] == ["Alpha", "Medio", "Zebra"]
        assert screen.table.cellWidget(0, 3) is not None

        screen._on_header_clicked(0)
        assert screen.sort_combo.currentData() == "name_desc"
        assert [screen.table.item(r, 0).text() for r in range(3)] == ["Zebra", "Medio", "Alpha"]

        # Sort by Date oldest via header click on Date/Created column
        screen._on_header_clicked(2)
        assert screen.sort_combo.currentData() == "date_desc"
        screen._on_header_clicked(2)
        assert screen.sort_combo.currentData() == "date_asc"
        assert [screen.table.item(r, 0).text() for r in range(3)] == ["Zebra", "Medio", "Alpha"]
    finally:
        screen.close()


def test_profile_details_dialog_renders_structured_layout_and_metrics(
    qapp, tmp_path, monkeypatch
):
    cfg = Config(data_dir=str(tmp_path))
    calibration = tmp_path / "calibrations" / "PivotBaseline.tpc"
    profile = RangeProfile(
        name="Pivot Baseline",
        created_at_utc="2026-11-14T08:30:00",
        context="Solo walking",
        model_hash=MANIFEST["sha256"],
        capture_profile=build_capture_profile(cfg, 1280, 720, "video"),
        source={"archive": "teacher_walk.tpr"},
        metrics={
            "head_pitch": ApprovedRange(
                lower=-12.0,
                upper=8.0,
                unit="degrees",
                view="side",
                passages=[Passage(start=0.0, end=5.0)],
                median=-2.0,
                p10=-10.0,
                p90=6.0,
                valid_duration=4.5,
                selected_duration=5.0,
                samples=45,
                inside_fraction=0.92,
            ),
            "torso_axis_tilt": ApprovedRange(
                lower=0.0,
                upper=5.5,
                unit="degrees",
                view="all",
                passages=[Passage(start=0.0, end=5.0)],
                median=2.1,
                p10=0.5,
                p90=4.8,
                valid_duration=5.0,
                selected_duration=5.0,
                samples=50,
                inside_fraction=1.0,
            ),
        },
    )
    save_range_profile(calibration, profile)
    screen = LibraryScreen(ServiceBridge(cfg), Router())
    monkeypatch.setattr(ProfileDetailsDialog, "exec", lambda self: int(self.DialogCode.Accepted))
    try:
        screen._profile_details(calibration)
        dialog = screen._profile_dialog
        assert isinstance(dialog, ProfileDetailsDialog)
        label_texts = [lbl.text() for lbl in dialog.findChildren(QLabel)]
        assert "Pivot Baseline" in label_texts
        assert "Solo walking" in label_texts
        assert "14/11/2026 08:30" in label_texts
        assert "Left side · Right facing" in label_texts
        assert "1280×720" in label_texts
        assert "teacher_walk.tpr" in label_texts
        assert dialog.metrics_table.rowCount() == 2
        assert dialog.metrics_table.item(0, 0).text() == "Head"
        assert dialog.metrics_table.item(0, 1).text() == "Head pitch (ear-to-nose)"
        assert "-12° … 8°" in dialog.metrics_table.item(0, 2).text()
        assert dialog.metrics_table.item(1, 0).text() == "Torso"
        assert dialog.metrics_table.item(1, 4).text() == "All"
        dialog.close()
    finally:
        screen.close()


