"""Main window shell with left navigation rail and stacked content screens."""

from __future__ import annotations

import sys
from importlib.resources import files as package_files
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QCloseEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import Config
from .components.status_strip import StatusStrip
from .navigation_icons import navigation_icon
from .router import Route, Router
from .screens.about import AboutDialog
from .screens.analysis import AnalysisScreen
from .screens.analytics import AnalyticsScreen
from .screens.coach import CoachScreen
from .screens.database import DatabaseScreen
from .screens.help import HelpScreen
from .screens.home import HomeScreen
from .screens.library import LibraryScreen
from .screens.practice import PracticeScreen
from .screens.references import ReferencesScreen
from .screens.review import ReviewScreen
from .screens.settings import SettingsScreen
from .service_bridge import ServiceBridge
from .theme import (
    STYLESHEET,
)


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, parent: QWidget | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self.bridge = ServiceBridge(cfg, self)
        self.router = Router(self)
        self.about_dialog: AboutDialog | None = None

        self.setWindowTitle("TangoPosture Studio")
        self.resize(1200, 800)
        self.setMinimumSize(1024, 700)
        self.setStyleSheet(STYLESHEET)

        # Main Root Widget
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Body: Left Nav + Central Screen Stack
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # 1. Left Navigation Rail
        self.nav_rail = QFrame()
        self.nav_rail.setObjectName("navRail")
        self.nav_rail.setFixedWidth(210)
        nav_layout = QVBoxLayout(self.nav_rail)
        nav_layout.setContentsMargins(12, 18, 12, 14)
        nav_layout.setSpacing(4)

        # Brand Header
        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        logo_lbl = QLabel()
        logo_data = self._load_logo()
        self.brand_pixmap = QPixmap()
        if logo_data:
            self.brand_pixmap.loadFromData(logo_data)
            logo_lbl.setPixmap(
                self.brand_pixmap.scaled(
                    28,
                    28,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        brand_title = QLabel("TangoPosture")
        brand_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        brand_row.addWidget(logo_lbl)
        brand_row.addWidget(brand_title)
        brand_row.addStretch()
        nav_layout.addLayout(brand_row)
        nav_layout.addSpacing(18)

        # Nav Buttons Group
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        self.nav_buttons: dict[Route, QPushButton] = {}

        nav_items = [
            (Route.HOME, "Home", "home"),
            (Route.PRACTICE, "Self Practice", "practice"),
            (Route.LIBRARY, "Library", "library"),
            (Route.REFERENCES, "Self Calibration", "references"),
            (Route.ANALYSIS, "Video Analysis", "analysis"),
            (Route.REVIEW, "Review", "review"),
            (Route.ANALYTICS, "Analytics", "analytics"),
            (Route.COACH, "AI Coach", "coach"),
            (Route.SETTINGS, "Settings", "settings"),
            (Route.HELP, "Help & Guide", "help"),
            (Route.DATABASE, "Database maintenance", "database"),
        ]

        for route, label, icon_name in nav_items:
            btn = QPushButton(label)
            btn.setProperty("class", "nav-btn")
            btn.setIcon(navigation_icon(icon_name))
            btn.setIconSize(QSize(20, 20))
            btn.setCheckable(True)
            btn.setAccessibleName(label)
            btn.setToolTip(label)
            btn.clicked.connect(lambda _, r=route: self.router.navigate(r))
            self.btn_group.addButton(btn)
            self.nav_buttons[route] = btn
            nav_layout.addWidget(btn)

        nav_layout.addStretch()
        self.about_btn = QPushButton("About")
        self.about_btn.setProperty("class", "nav-btn")
        self.about_btn.setIcon(navigation_icon("about"))
        self.about_btn.setIconSize(QSize(20, 20))
        self.about_btn.setAccessibleName("About TangoPosture")
        self.about_btn.clicked.connect(self._show_about)
        nav_layout.addWidget(self.about_btn)
        body_layout.addWidget(self.nav_rail)

        # 2. Central Stacked Content Screens
        self.stack = QStackedWidget()
        self.screens: dict[Route, QWidget] = {
            Route.HOME: HomeScreen(self.bridge, self.router),
            Route.PRACTICE: PracticeScreen(self.bridge, self.router),
            Route.REFERENCES: ReferencesScreen(self.bridge, self.router),
            Route.LIBRARY: LibraryScreen(self.bridge, self.router),
            Route.ANALYSIS: AnalysisScreen(self.bridge, self.router),
            Route.REVIEW: ReviewScreen(self.bridge, self.router),
            Route.ANALYTICS: AnalyticsScreen(self.bridge, self.router),
            Route.COACH: CoachScreen(self.bridge, self.router),
            Route.SETTINGS: SettingsScreen(self.bridge, self.router),
            Route.HELP: HelpScreen(self.bridge, self.router),
            Route.DATABASE: DatabaseScreen(self.bridge, self._database_cleanup_blocker),
        }

        for route, screen in self.screens.items():
            self.stack.addWidget(screen)

        body_layout.addWidget(self.stack, 1)
        root_layout.addWidget(body, 1)

        # 3. Bottom Status Strip
        self.status_strip = StatusStrip(self)
        root_layout.addWidget(self.status_strip)

        # Wire Signals
        self.router.route_changed.connect(self._on_route_changed)
        self.bridge.status_message.connect(self.status_strip.set_status)

        # Setup Shortcuts (Ctrl+1 to Ctrl+8 / Cmd+1 to Cmd+8)
        for idx, (route, _, _) in enumerate(nav_items, start=1):
            modifier = "Meta" if sys.platform == "darwin" else "Ctrl"
            shortcut = QShortcut(QKeySequence(f"{modifier}+{idx}"), self)
            shortcut.activated.connect(lambda r=route: self._navigate_shortcut(r))

        # Initial Route
        self.nav_buttons[Route.HOME].setChecked(True)
        self.stack.setCurrentWidget(self.screens[Route.HOME])

    def _database_cleanup_blocker(self) -> str | None:
        library = self.screens[Route.LIBRARY]
        analysis = self.screens[Route.ANALYSIS]
        review = self.screens[Route.REVIEW]
        if (
            isinstance(library, LibraryScreen) and library.has_active_jobs()
            or isinstance(analysis, AnalysisScreen) and analysis.has_active_jobs()
            or isinstance(review, ReviewScreen) and review.has_active_jobs()
        ):
            return "Wait for Library imports/packaging, Analysis, and Review jobs to finish before cleanup."
        return None

    def _navigate_shortcut(self, route: Route) -> None:
        focused_widget = QApplication.focusWidget()
        while focused_widget is not None:
            if isinstance(focused_widget, (QLineEdit, QPlainTextEdit, QTextEdit)):
                return
            focused_widget = focused_widget.parentWidget()
        self.router.navigate(route)

    def _show_about(self) -> None:
        if self.about_dialog is None:
            self.about_dialog = AboutDialog(self.brand_pixmap, self)
        self.about_dialog.show()
        self.about_dialog.raise_()
        self.about_dialog.activateWindow()

    def _load_logo(self) -> bytes | None:
        try:
            res = package_files("tango_posture").joinpath("TangoPosture_logo_under_1MB.png")
            return res.read_bytes()
        except (OSError, ValueError, FileNotFoundError):
            path = Path(__file__).resolve().parents[3] / "logos" / "TangoPosture_logo_under_1MB.png"
            if path.is_file():
                return path.read_bytes()
        return None

    def _on_route_changed(self, route: Route, payload: object) -> None:
        practice = self.screens.get(Route.PRACTICE)
        if route is not Route.PRACTICE and isinstance(practice, PracticeScreen) and practice.worker_running:
            practice.stop_for_navigation()
        refs = self.screens.get(Route.REFERENCES)
        if route is not Route.REFERENCES and isinstance(refs, ReferencesScreen):
            refs.stop_for_navigation()
        if route in self.nav_buttons:
            self.nav_buttons[route].setChecked(True)

        screen = self.screens.get(route)
        if screen:
            self.stack.setCurrentWidget(screen)
            if hasattr(screen, "refresh"):
                screen.refresh()
            if route is Route.ANALYSIS and isinstance(screen, AnalysisScreen) and payload is not None:
                screen.load_source(payload)
            if route is Route.REVIEW and isinstance(screen, ReviewScreen):
                if isinstance(payload, Path):
                    screen.load_archive(payload)
                elif isinstance(payload, dict):
                    archive_path = payload.get("archive_path")
                    segment_index = payload.get("segment_index", 0)
                    timestamp = payload.get("timestamp")
                    run_id = payload.get("run_id")
                    if (
                        isinstance(archive_path, Path)
                        and isinstance(segment_index, int)
                        and (timestamp is None or isinstance(timestamp, (int, float)))
                    ):
                        screen.load_archive(
                            archive_path,
                            segment_index=segment_index,
                            timestamp=timestamp,
                            run_id=str(run_id) if isinstance(run_id, str) and run_id else None,
                        )
            if route is Route.ANALYTICS and isinstance(screen, AnalyticsScreen):
                if isinstance(payload, Path):
                    screen.load_archive(payload)
                elif isinstance(payload, dict):
                    archive_path = payload.get("archive_path")
                    segment_index = payload.get("segment_index", 0)
                    calibration_path = payload.get("calibration_path")
                    if (
                        isinstance(archive_path, Path)
                        and isinstance(segment_index, int)
                        and (calibration_path is None or isinstance(calibration_path, Path))
                    ):
                        screen.load_archive(
                            archive_path,
                            segment_index=segment_index,
                            calibration_path=calibration_path,
                        )
            if (
                route is Route.COACH
                and isinstance(screen, CoachScreen)
                and isinstance(payload, (Path, dict))
            ):
                screen.load_report(payload)
            if (
                route is Route.SETTINGS
                and isinstance(screen, SettingsScreen)
                and isinstance(payload, dict)
            ):
                screen.open_measurement_normalization(payload)

    def closeEvent(self, event: QCloseEvent) -> None:
        super().closeEvent(event)
