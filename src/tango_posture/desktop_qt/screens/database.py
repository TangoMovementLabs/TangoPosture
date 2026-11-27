"""Explicit, database-only maintenance with transactional cleanup."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...storage import DATABASE_TABLES, clear_database, database_counts
from ..service_bridge import ServiceBridge

CONFIRMATION_PHRASE = "DELETE ALL DATA"


class _CleanupProgress(QDialog):
    def reject(self) -> None:
        return

    def closeEvent(self, event: QCloseEvent) -> None:
        event.ignore()


class DatabaseScreen(QWidget):
    def __init__(
        self, bridge: ServiceBridge, busy_reason: Callable[[], str | None],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.bridge = bridge
        self._busy_reason = busy_reason
        self._counts: dict[str, int] | None = None
        self._thread: Thread | None = None
        self._results: Queue[dict[str, int] | str] = Queue()
        self._progress: _CleanupProgress | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll_cleanup)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        title = QLabel("Database maintenance")
        title.setProperty("class", "title")
        layout.addWidget(title)
        description = QLabel(
            "Empty calibrations, sessions, events and observations in the configured SQLite "
            "database. Tables, indexes, .tp/.tpr/.tpc archives, videos and reports are preserved. "
            "Archived history and file-based profiles may therefore still appear in Library."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)

        self.table = QTableWidget(len(DATABASE_TABLES), 2)
        self.table.setHorizontalHeaderLabels(["Table", "Rows"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, name in enumerate(DATABASE_TABLES):
            self.table.setItem(row, 0, QTableWidgetItem(name))
        layout.addWidget(self.table)
        self.refresh_btn = QPushButton("Refresh row counts")
        self.refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(self.refresh_btn)

        warning = QLabel(
            "Permanent deletion: no backup will be created and there is no Undo. "
            "Stop other TangoPosture processes before continuing. Cleanup is blocked while "
            "this app imports, packages or analyzes video, and cannot be cancelled once started."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self.acknowledge_cb = QCheckBox(
            "I understand that all rows in these four tables will be deleted without a backup."
        )
        self.acknowledge_cb.toggled.connect(self._update_clear_button)
        layout.addWidget(self.acknowledge_cb)
        self.confirmation_input = QLineEdit()
        self.confirmation_input.setPlaceholderText(f"Type {CONFIRMATION_PHRASE} to continue")
        self.confirmation_input.setAccessibleName("Database cleanup confirmation phrase")
        self.confirmation_input.textChanged.connect(self._update_clear_button)
        layout.addWidget(self.confirmation_input)
        self.clear_btn = QPushButton("Clear all four tables...")
        self.clear_btn.clicked.connect(self._start_cleanup)
        layout.addWidget(self.clear_btn)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()

        bridge.config_changed.connect(self.refresh)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._wait_for_cleanup)
        self.refresh()

    def _database_path(self) -> Path:
        return (Path(self.bridge.config.data_dir) / "tango.sqlite3").resolve()

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.bridge.status_message.emit(message)

    def refresh(self, *_args: object) -> None:
        if self._thread is not None:
            return
        path = self._database_path()
        self.path_label.setText(f"Database: {path}")
        self.confirmation_input.clear()
        self.acknowledge_cb.setChecked(False)
        self._counts = None
        for row in range(len(DATABASE_TABLES)):
            self.table.setItem(row, 1, QTableWidgetItem("-"))
        if not path.is_file():
            self._set_status("No database exists at this path. Nothing will be created or cleared.")
        else:
            try:
                self._counts = database_counts(path)
            except (OSError, sqlite3.Error, ValueError) as error:
                self._set_status(f"Could not read database: {error}")
            else:
                for row, name in enumerate(DATABASE_TABLES):
                    self.table.setItem(row, 1, QTableWidgetItem(str(self._counts[name])))
                self._set_status(
                    f"{sum(self._counts.values())} rows across four tables."
                    if any(self._counts.values()) else "All four tables are already empty."
                )
        self._update_clear_button()

    def _update_clear_button(self, *_args: object) -> None:
        self.clear_btn.setEnabled(
            self._thread is None
            and self._counts is not None and any(self._counts.values())
            and self.acknowledge_cb.isChecked()
            and self.confirmation_input.text() == CONFIRMATION_PHRASE
        )

    def _start_cleanup(self) -> None:
        if not self.clear_btn.isEnabled():
            self._set_status("Acknowledge the warning and type the confirmation phrase first.")
            return
        reason = self._busy_reason()
        if reason is not None:
            self._set_status(reason)
            return
        path = self._database_path()
        if QMessageBox.warning(
            self, "Permanently clear database?",
            f"Delete ALL rows from calibrations, sessions, events and observations in:\n"
            f"{path}\n\nNo backup will be created. There is no Undo.\n"
            "Archive, profile, video and report files will not be deleted.\n\n"
            "Stop other TangoPosture processes before continuing. Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            self.refresh()
            return
        reason = self._busy_reason()
        if reason is not None or path != self._database_path():
            self.refresh()
            self._set_status(reason or "Database selection changed. Review it and confirm again.")
            return

        results = self._results

        def work() -> None:
            try:
                results.put(clear_database(path))
            except (OSError, sqlite3.Error, ValueError, RuntimeError) as error:
                results.put(f"{type(error).__name__}: {error}")

        self._progress = _CleanupProgress(self)
        self._progress.setWindowTitle("Clearing database")
        self._progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._progress.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        progress_layout = QVBoxLayout(self._progress)
        progress_layout.addWidget(QLabel("Clearing all four tables. Please wait; do not close the app."))
        progress = QProgressBar()
        progress.setRange(0, 0)
        progress_layout.addWidget(progress)
        self._progress.show()
        self._thread = Thread(target=work, name="database-cleanup")
        self.refresh_btn.setEnabled(False)
        self._update_clear_button()
        self._thread.start()
        self._timer.start()

    def _poll_cleanup(self) -> None:
        if self._thread is None or self._thread.is_alive():
            return
        self._thread.join()
        self._thread = None
        self._timer.stop()
        if self._progress is not None:
            self._progress.accept()
            self._progress.deleteLater()
            self._progress = None
        self.refresh_btn.setEnabled(True)
        try:
            result = self._results.get_nowait()
        except Empty:
            result = "Cleanup worker finished without a result."
        self.refresh()
        if isinstance(result, str):
            self._set_status(f"Database cleanup failed: {result}")
        else:
            self._set_status(
                f"Deleted {sum(result.values())} rows. No backup was created; schema, indexes "
                f"and archive/media files were preserved. {self.status_label.text()}"
            )

    def _wait_for_cleanup(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None:
            event.ignore()
            return
        super().closeEvent(event)
