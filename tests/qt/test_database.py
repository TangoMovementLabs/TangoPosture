import sqlite3
from threading import Event, Thread

import pytest
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QMessageBox

from tango_posture.config import Config
from tango_posture.desktop_qt.router import Route
from tango_posture.desktop_qt.screens.database import CONFIRMATION_PHRASE, DatabaseScreen
from tango_posture.desktop_qt.service_bridge import ServiceBridge
from tango_posture.desktop_qt.shell import MainWindow
from tango_posture.storage import DATABASE_TABLES, Storage, database_counts


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(["-platform", "offscreen"])


@pytest.fixture
def populated_database(tmp_path, reference):
    storage = Storage(tmp_path)
    try:
        storage.save_calibration(reference)
        storage.start("fixture-session", reference.id, "fixture", "synthetic", {}, {})
    finally:
        storage.close()
    return tmp_path / "tango.sqlite3"


def prepare_confirmation(screen):
    screen.acknowledge_cb.setChecked(True)
    screen.confirmation_input.setText(CONFIRMATION_PHRASE)


def test_missing_database_is_not_created_or_clearable(qapp, tmp_path):
    screen = DatabaseScreen(ServiceBridge(Config(data_dir=str(tmp_path))), lambda: None)
    try:
        prepare_confirmation(screen)
        assert not screen.clear_btn.isEnabled()
        assert "No database exists" in screen.status_label.text()
        assert not (tmp_path / "tango.sqlite3").exists()
    finally:
        screen.close()


def test_typed_confirmation_and_final_no_preserve_data(
    qapp, tmp_path, populated_database, monkeypatch,
):
    screen = DatabaseScreen(ServiceBridge(Config(data_dir=str(tmp_path))), lambda: None)
    before = database_counts(populated_database)
    try:
        screen.confirmation_input.setText(CONFIRMATION_PHRASE)
        assert not screen.clear_btn.isEnabled()
        screen.acknowledge_cb.setChecked(True)
        screen.confirmation_input.setText("delete all data")
        assert not screen.clear_btn.isEnabled()
        screen.confirmation_input.setText(CONFIRMATION_PHRASE)
        assert screen.clear_btn.isEnabled()

        def decline(*args):
            assert args[-1] == QMessageBox.StandardButton.No
            assert str(populated_database) in args[2]
            assert "No backup" in args[2]
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "warning", decline)
        screen.clear_btn.click()

        assert screen._thread is None
        assert not screen.clear_btn.isEnabled()
        assert database_counts(populated_database) == before
    finally:
        screen.close()


def test_confirmed_cleanup_refreshes_counts_and_resets_confirmation(
    qapp, tmp_path, populated_database, monkeypatch,
):
    screen = DatabaseScreen(ServiceBridge(Config(data_dir=str(tmp_path))), lambda: None)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args: QMessageBox.StandardButton.Yes,
    )
    try:
        prepare_confirmation(screen)
        screen.clear_btn.click()
        assert not screen.clear_btn.isEnabled()
        assert not screen.refresh_btn.isEnabled()
        assert screen._progress is not None
        screen._progress.reject()
        assert screen._progress.isVisible()
        screen._wait_for_cleanup()
        completion = QSignalSpy(screen.bridge.status_message)
        assert completion.wait(2000)
        assert screen._thread is None

        assert database_counts(populated_database) == dict.fromkeys(DATABASE_TABLES, 0)
        assert "Deleted 2 rows" in screen.status_label.text()
        assert "No backup was created" in screen.status_label.text()
        assert not screen.clear_btn.isEnabled()
        assert screen.refresh_btn.isEnabled()
        assert not screen.confirmation_input.text()
        assert not screen.acknowledge_cb.isChecked()
        assert screen._progress is None
        assert all(screen.table.item(row, 1).text() == "0" for row in range(4))
    finally:
        screen._wait_for_cleanup()
        screen._poll_cleanup()
        screen.close()


def test_cleanup_failure_is_reported_without_success_message(
    qapp, tmp_path, populated_database, monkeypatch,
):
    screen = DatabaseScreen(ServiceBridge(Config(data_dir=str(tmp_path))), lambda: None)
    before = database_counts(populated_database)
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args: QMessageBox.StandardButton.Yes,
    )

    def fail(_path):
        raise sqlite3.OperationalError("fixture database locked")

    monkeypatch.setattr("tango_posture.desktop_qt.screens.database.clear_database", fail)
    try:
        prepare_confirmation(screen)
        screen.clear_btn.click()
        screen._wait_for_cleanup()
        screen._poll_cleanup()
        assert "Database cleanup failed" in screen.status_label.text()
        assert "fixture database locked" in screen.status_label.text()
        assert "Deleted" not in screen.status_label.text()
        assert database_counts(populated_database) == before
        assert screen._progress is None
    finally:
        screen._wait_for_cleanup()
        screen._poll_cleanup()
        screen.close()


def test_active_jobs_block_cleanup(qapp, tmp_path, populated_database, monkeypatch):
    screen = DatabaseScreen(
        ServiceBridge(Config(data_dir=str(tmp_path))), lambda: "Wait for fixture analysis.",
    )
    before = database_counts(populated_database)

    def unexpected_dialog(*args):
        pytest.fail("Busy cleanup must not reach the confirmation dialog")

    monkeypatch.setattr(QMessageBox, "warning", unexpected_dialog)
    try:
        prepare_confirmation(screen)
        screen.clear_btn.click()
        assert "Wait for fixture analysis" in screen.status_label.text()
        assert screen._thread is None
        assert database_counts(populated_database) == before
    finally:
        screen.close()


def test_config_change_during_confirmation_cannot_clear_another_database(
    qapp, tmp_path, populated_database, monkeypatch,
):
    bridge = ServiceBridge(Config(data_dir=str(tmp_path)))
    screen = DatabaseScreen(bridge, lambda: None)
    before = database_counts(populated_database)

    def change_target(*args):
        bridge.update_config(bridge.config.model_copy(update={"data_dir": str(tmp_path / "other")}))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "warning", change_target)
    try:
        prepare_confirmation(screen)
        screen.clear_btn.click()
        assert screen._thread is None
        assert "selection changed" in screen.status_label.text()
        assert not screen.clear_btn.isEnabled()
        assert database_counts(populated_database) == before
        assert not (tmp_path / "other").exists()
    finally:
        screen.close()


@pytest.mark.parametrize("job", ["import", "packaging", "analysis"])
def test_shell_blocks_cleanup_during_active_jobs(
    qapp, tmp_path, populated_database, monkeypatch, job,
):
    window = MainWindow(Config(data_dir=str(tmp_path)))
    screen = window.screens[Route.DATABASE]
    assert isinstance(screen, DatabaseScreen)
    before = database_counts(populated_database)
    stop = Event()
    thread = Thread(target=stop.wait)
    if job == "import":
        monkeypatch.setattr(window.screens[Route.LIBRARY], "_import_job_id", "fixture-import")
    elif job == "packaging":
        monkeypatch.setattr(window.screens[Route.LIBRARY], "_package_thread", thread)
        thread.start()
    else:
        monkeypatch.setattr(window.screens[Route.REVIEW], "_analysis_thread", thread)
        thread.start()

    def unexpected_dialog(*args):
        pytest.fail("Active jobs must prevent destructive confirmation")

    monkeypatch.setattr(QMessageBox, "warning", unexpected_dialog)
    try:
        prepare_confirmation(screen)
        screen.clear_btn.click()
        assert "Wait for Library" in screen.status_label.text()
        assert screen._thread is None
        assert database_counts(populated_database) == before
    finally:
        stop.set()
        if thread.ident is not None:
            thread.join()
        window.close()
