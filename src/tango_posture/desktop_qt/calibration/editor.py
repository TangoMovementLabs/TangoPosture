"""Shared evidence and explicit range approval for video and camera calibrations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...archive import (
    PracticeArchive,
    calibration_directory,
    calibration_path,
    file_sha256,
    read_analytics,
    read_movement_analysis,
)
from ...calibration.archives import read_range_profile, save_range_profile
from ...calibration.evidence import (
    cached_evidence,
    contiguous_pose_run_end,
    dancer_frames,
    has_linkable_pose,
    is_static_pose_run,
    single_dancer_anchors,
)
from ...calibration.measurements import metric_samples
from ...calibration.models import (
    METRIC_DETAILS,
    PRACTICE_CONTEXTS,
    STEPS,
    ApprovedRange,
    Passage,
    RangeProfile,
)
from ...calibration.statistics import repeat_check, summarize_examples
from ...config import Config
from ...domain import FrameObservation
from ...movement import render_movement_overlay, validate_movement_analysis
from ...quality import reliable
from ...references.archives import reference_directory
from ...references.coach import (
    compare_approved_coach_data,
    interpolate_coach_data,
    render_coach_frame_overlay,
)
from ...replay import VideoReader
from ..analytics.data import MetricData
from ..components.metric_chart import MetricChart
from ..components.video_stage import VideoStage
from ..service_bridge import ServiceBridge, numpy_to_qimage
from .preview import render_approved_feedback

_METRIC_HINTS: dict[str, str] = {
    "head_pitch": (
        "Ear-to-nose face/gaze pitch angle relative to horizontal, in degrees "
        "(independent of neck-to-torso inclination)."
    ),
    "head_forward": (
        "Linear forward offset of the ear–shoulder neck vector perpendicular to the "
        "torso axis, in torso lengths. Shares the same ear–shoulder–hip geometry as "
        "Head inclination, so their curves have nearly identical shapes with different Y-axis units."
    ),
    "head_inclination": (
        "Angle of the ear–shoulder neck vector relative to the torso axis, in degrees. "
        "Computed from the same ear–shoulder–hip vector as Head-forward offset, producing "
        "a nearly identical curve shape scaled in degrees."
    ),
    "torso_axis_tilt": (
        "Pink line: shoulder-midpoint to hip-midpoint tilt from image vertical, "
        "in degrees (0 = upright). The angle is a magnitude, as on the video overlay; "
        "both shoulders and hips must be reliably visible. Keep the camera level."
    ),
}


def _resolve_reference_for_profile(profile: RangeProfile, profile_path: Path, cfg: Config) -> Path:
    archive_name = str(profile.source.get("archive") or "").strip()
    archive_hash = str(profile.source.get("archive_sha256") or "").strip()
    search_dirs: list[Path] = []
    for directory in (reference_directory(cfg.data_dir), profile_path.parent.parent / "references"):
        if directory not in search_dirs:
            search_dirs.append(directory)
    if archive_name:
        for directory in search_dirs:
            candidate = directory / archive_name
            if candidate.is_file() and (not archive_hash or file_sha256(candidate) == archive_hash):
                return candidate
    if archive_hash:
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob("*.tpr")):
                if file_sha256(candidate) == archive_hash:
                    return candidate
    # Fallback when the .tpr archive was updated in place (e.g. Library rename or dancer selection).
    if archive_name:
        for directory in search_dirs:
            candidate = directory / archive_name
            if candidate.is_file():
                return candidate
    prov = profile.source.get("provenance")
    prov_sha = str(prov.get("sha256") or "").strip() if isinstance(prov, dict) else ""
    if prov_sha:
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob("*.tpr")):
                try:
                    analytics = read_analytics(candidate, 0) or {}
                    src_prov = analytics.get("source_provenance")
                    if isinstance(src_prov, dict) and str(src_prov.get("sha256") or "").strip() == prov_sha:
                        return candidate
                except (OSError, ValueError, KeyError, TypeError):
                    continue
    raise FileNotFoundError(
        f"Source reference '{archive_name or 'unknown'}' was not found in "
        f"{reference_directory(cfg.data_dir)}. Keep the source .tpr reference in Library "
        "to edit this calibration."
    )


def selected_tpc_calibration_path(selected: object) -> Path | None:
    """Return the existing `.tpc` file path when a combo selection points to one."""
    if not isinstance(selected, str) or not selected or selected == "__defaults__":
        return None
    candidate = Path(selected)
    if candidate.suffix.lower() != ".tpc" or not candidate.is_file():
        return None
    return candidate


def open_profile_calibration_editor(
    path: Path,
    cfg: Config,
    parent: QWidget,
    *,
    bridge: ServiceBridge | None = None,
) -> Path | None:
    """Open CalibrationEditor for an existing .tpc profile, prompting if its .tpr moved."""
    source_reference: Path | None = None
    while True:
        try:
            editor = CalibrationEditor(
                path,
                cfg,
                parent,
                bridge=bridge,
                source_reference=source_reference,
            )
            if editor.exec() == QDialog.DialogCode.Accepted:
                return editor.saved_path
            return None
        except FileNotFoundError:
            chosen, _ = QFileDialog.getOpenFileName(
                parent,
                "Locate Source Reference (.tpr)",
                str(reference_directory(cfg.data_dir)),
                "Reference archives (*.tpr);;All files (*)",
            )
            if not chosen:
                raise
            source_reference = Path(chosen)


class CalibrationEditor(QDialog):
    def __init__(
        self, path: Path, cfg: Config, parent: QWidget | None = None, *,
        holds: dict[str, list[Passage]] | None = None,
        base_profile: RangeProfile | None = None,
        bridge: ServiceBridge | None = None,
        source_reference: Path | None = None,
    ) -> None:
        super().__init__(parent)
        suffix = path.suffix.lower()
        if suffix == ".tpc":
            self._editing_profile_path: Path | None = path
            base_profile = base_profile or read_range_profile(path)
            tpr_path = source_reference or _resolve_reference_for_profile(base_profile, path, cfg)
            self.setWindowTitle("Edit approved range calibration")
        elif suffix == ".tpr":
            self._editing_profile_path = None
            tpr_path = path
            self.setWindowTitle("Review evidence & approve calibration ranges")
        else:
            raise ValueError("Create a .tpr reference before approving calibration ranges")
        self.resize(1180, 800)
        self.cfg = cfg.model_copy(deep=True)
        self.bridge = bridge
        self.evidence = cached_evidence(tpr_path)
        self._source_hash = self.evidence.source["archive_sha256"]
        self.holds = holds or {}
        self.base_profile = base_profile
        if base_profile is not None and base_profile.model_hash != self.evidence.model_hash:
            raise ValueError("Reference and camera measurement models differ; reanalyze the reference")
        self.saved_path: Path | None = None
        self.approved: dict[str, ApprovedRange] = (
            dict(base_profile.metrics)
            if self._editing_profile_path is not None and base_profile is not None
            else {}
        )
        self._approved_coach_data: dict[str, Any] | None = None
        self._draft_keys: set[str] = set()
        self._draft_specs: dict[str, dict[str, Any]] = {}
        self._selected_frames: list[FrameObservation] = []
        self._samples_by_view: dict[str, dict[str, list[tuple[float, float | None]]]] = {}
        self._stats: dict[str, Any] | None = None
        self._track_start = 0
        self._anchors: dict[int, int] = {0: 0}
        capture = self.evidence.capture_profile
        if not capture:
            raise ValueError("Reference capture settings are missing; reanalyze it in Review")
        self.cfg = Config.model_validate({
            **self.cfg.model_dump(),
            **{key: capture[key] for key in ("side", "facing", "width", "height", "capture_profile")},
        })
        self._archive = PracticeArchive(tpr_path)
        try:
            if file_sha256(tpr_path) != self._source_hash:
                raise ValueError("Reference changed while loading evidence; reopen it")
            segment = self._archive.manifest["segments"][0]
            descriptor = json.loads((self._archive.folder / "reference.json").read_text())
            clip_starts_static = is_static_pose_run(self.evidence.frames, 0)

            def _is_stale_static_anchor(frame_index: int) -> bool:
                return (
                    frame_index > 0
                    and not clip_starts_static
                    and is_static_pose_run(self.evidence.frames, frame_index)
                )

            anchors = descriptor.get("dancer_anchors")
            if anchors is not None:
                if not isinstance(anchors, list) or not anchors:
                    raise ValueError("Invalid saved reference dancer selection")
                filtered: dict[int, int] = {}
                for anchor in anchors:
                    if (
                        not isinstance(anchor, dict) or type(anchor.get("frame")) is not int
                        or type(anchor.get("dancer")) is not int
                        or not 0 <= anchor["frame"] < len(self.evidence.frames)
                        or not 0 <= anchor["dancer"] < len(self.evidence.frames[anchor["frame"]].poses)
                    ):
                        raise ValueError("Saved dancer selection does not match the cached poses")
                    if not _is_stale_static_anchor(anchor["frame"]):
                        filtered[anchor["frame"]] = anchor["dancer"]
                self._anchors = filtered or {0: 0}
            if self._editing_profile_path is not None and base_profile is not None:
                profile_anchors = base_profile.source.get("dancer_anchors")
                if isinstance(profile_anchors, list) and profile_anchors:
                    restored: dict[int, int] = {}
                    valid = True
                    for anchor in profile_anchors:
                        if (
                            not isinstance(anchor, dict) or type(anchor.get("frame")) is not int
                            or type(anchor.get("dancer")) is not int
                            or not 0 <= anchor["frame"] < len(self.evidence.frames)
                            or not 0 <= anchor["dancer"] < len(
                                self.evidence.frames[anchor["frame"]].poses
                            )
                        ):
                            valid = False
                            break
                        if not _is_stale_static_anchor(anchor["frame"]):
                            restored[anchor["frame"]] = anchor["dancer"]
                    if valid and restored:
                        self._anchors = restored
            self._reader = VideoReader(self._archive.folder / segment["video"])
            if self._reader.count != len(self.evidence.frames):
                self._reader.close()
                raise ValueError("Landmark cache and reference video frame counts differ; reanalyze")
            analytics = read_analytics(self._archive.folder, 0)
            if analytics is None:
                raise ValueError("Reference analysis is missing; analyze it in Review first")
            raw_coach = analytics.get("coach_data")
            has_coach_frames = isinstance(raw_coach, dict) and bool(raw_coach.get("frames"))
            coach: dict[str, Any]
            if isinstance(raw_coach, dict) and raw_coach.get("frames"):
                coach = raw_coach
            else:
                coach = {
                    "width": self.evidence.frames[0].width,
                    "height": self.evidence.frames[0].height,
                    "frames": {
                        str(index): {
                            "timestamp": frame.timestamp,
                            "measurement_poses": [
                                {name: [p.x, p.y, p.z, p.visibility, p.presence]
                                 for name, p in pose.items()}
                                for pose in frame.poses
                            ],
                        }
                        for index, frame in enumerate(self.evidence.frames)
                    },
                }
            auto_linked_single = False
            if (
                anchors is None
                and self._editing_profile_path is None
                and has_coach_frames
                and max((len(frame.poses) for frame in self.evidence.frames), default=0) == 1
            ):
                self._anchors = single_dancer_anchors(self.evidence, self.cfg.confidence)
                self._track_start = min(self._anchors)
                auto_linked_single = True
            self._movement_data = analytics.get("movement_data")
            if self._movement_data is None:
                movement = read_movement_analysis(self._archive.folder, 0)
                if movement is not None:
                    self._movement_data = validate_movement_analysis(
                        movement, expected_segment_index=0
                    )
            self._coach_data = interpolate_coach_data(
                coach,
                self.cfg,
                pose_recovery=coach.get("pose_recovery"),
                movement_data=self._movement_data,
            )
        except Exception:
            if hasattr(self, "_reader"):
                self._reader.close()
            self._archive.close()
            raise

        layout = QVBoxLayout(self)
        form = QHBoxLayout()
        default_name = (
            base_profile.name
            if self._editing_profile_path is not None and base_profile is not None
            else f"{base_profile.name} camera check"
            if base_profile is not None
            else f"{tpr_path.stem} calibration"
        )
        self.name_input = QLineEdit(default_name)
        self.context_combo = QComboBox()
        self.context_combo.addItems(PRACTICE_CONTEXTS)
        self.context_combo.setToolTip(
            "Describes the activity this profile was created for; it does not change "
            "measurement rules or automatically detect the activity."
        )
        if base_profile is not None:
            self.context_combo.setEditable(True)
            self.context_combo.setCurrentText(base_profile.context)
        form.addWidget(QLabel("Profile name:"))
        form.addWidget(self.name_input, 1)
        form.addWidget(QLabel("Practice context:"))
        form.addWidget(self.context_combo)
        layout.addLayout(form)
        warning = QLabel(
            "A reference describes this dancer and passage, not a universal technique standard. "
            "Separate standing/walking passages when their distributions differ. "
            "Examples and acceptable limits are independent; only explicitly approved metrics are saved."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        video = QWidget()
        video_layout = QVBoxLayout(video)
        self.stage = VideoStage()
        video_layout.addWidget(self.stage, 1)
        if self._movement_data is None:
            movement_info = QLabel(
                "No cached movement analysis. Analyze this reference in Review to show "
                "pivot rings and footprints."
            )
            movement_info.setWordWrap(True)
            video_layout.addWidget(movement_info)
        playback = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        playback.addWidget(self.play_button)
        self.time_label = QLabel()
        playback.addWidget(self.time_label)
        video_layout.addLayout(playback)
        self.play_timer = QTimer(self)
        self.play_timer.setSingleShot(True)
        self.play_timer.timeout.connect(self._advance_frame)
        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, self._reader.count - 1)
        self.scrubber.valueChanged.connect(self._seek_frame)
        video_layout.addWidget(self.scrubber)
        selection = QHBoxLayout()
        self.dancer_spin = QSpinBox()
        self.dancer_spin.setRange(1, max(len(frame.poses) for frame in self.evidence.frames))
        self.dancer_spin.setValue(min(self._anchors.items())[1] + 1)
        self.dancer_spin.valueChanged.connect(self._restart_track)
        restart = QPushButton("Select dancer / restart track here")
        restart.clicked.connect(self._restart_track)
        self.auto_link_button = QPushButton("Auto-link single dancer")
        self.auto_link_button.setEnabled(
            max((len(frame.poses) for frame in self.evidence.frames), default=0) == 1
        )
        self.auto_link_button.setToolTip(
            "Automatically link single-dancer segments across brief pose dropouts"
        )
        self.auto_link_button.clicked.connect(self._auto_link_single_dancer)
        selection.addWidget(QLabel("Dancer at playhead:"))
        selection.addWidget(self.dancer_spin)
        selection.addWidget(restart)
        selection.addWidget(self.auto_link_button)
        video_layout.addLayout(selection)
        self.confirm_identity = QCheckBox("I confirm all manually linked passages follow the same dancer")
        self.confirm_identity.setChecked(len(self._anchors) == 1 or auto_linked_single)
        video_layout.addWidget(self.confirm_identity)
        save_selection = QPushButton("Save dancer selection in reference (.tpr)")
        save_selection.clicked.connect(self._save_selection)
        video_layout.addWidget(save_selection)
        chart_header = QHBoxLayout()
        chart_header.addWidget(QLabel(
            "Metric chart · pink = playhead · dashed = limits · shaded = examples"
        ), 1)
        chart_header.addWidget(QLabel("X-axis zoom:"))
        self.chart_zoom_combo = QComboBox()
        for label, factor in (("1× (Fit)", 1.0), ("2×", 2.0), ("4×", 4.0), ("8×", 8.0)):
            self.chart_zoom_combo.addItem(label, factor)
        self.chart_zoom_combo.currentIndexChanged.connect(self._change_chart_zoom)
        chart_header.addWidget(self.chart_zoom_combo)
        video_layout.addLayout(chart_header)
        video_layout.addWidget(QLabel(
            "Approved state: green = within · orange = pending · red = alert"
        ))
        self.chart = MetricChart()
        self.chart.setMinimumHeight(140)
        self.chart.seek_requested.connect(self._seek_timestamp)
        video_layout.addWidget(self.chart)
        splitter.addWidget(video)
        inspector = QWidget()
        inspector.setMaximumWidth(360)
        controls = QVBoxLayout(inspector)
        self.step_combo = QComboBox()
        self.step_combo.addItems(STEPS)
        self.metric_combo = QComboBox()
        self.step_combo.currentTextChanged.connect(self._change_step)
        self.metric_combo.currentIndexChanged.connect(self._change_metric)
        controls.addWidget(self.step_combo)
        controls.addWidget(self.metric_combo)
        self.metric_hint = QLabel()
        self.metric_hint.setWordWrap(True)
        controls.addWidget(self.metric_hint)
        options = QFormLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems(["side", "front", "all"])
        options.addRow("Applicable view:", self.view_combo)
        self.passages_input = QLineEdit()
        self.passages_input.setPlaceholderText("Example seconds: 0-3; 5-9")
        options.addRow("Example passages:", self.passages_input)
        self.pose_combo = QComboBox()
        self.pose_combo.addItems(["Neutral hold", "Changed hold", "Return hold", "Fresh repeat", "Walking drill"])
        self.pose_combo.setVisible(bool(self.holds))
        self.pose_combo.currentIndexChanged.connect(self._select_hold)
        options.addRow("Use captured examples:", self.pose_combo)
        self.lower_spin, self.upper_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for field in (self.lower_spin, self.upper_spin):
            field.setRange(-10000, 10000)
            field.setDecimals(5)
        options.addRow("Acceptable minimum:", self.lower_spin)
        options.addRow("Acceptable maximum:", self.upper_spin)
        controls.addLayout(options)
        suggest = QPushButton("Suggest middle 80% of examples")
        suggest.clicked.connect(self._on_suggest_clicked)
        controls.addWidget(suggest)
        controls.addWidget(QLabel("Distribution · tinted area = acceptable range"))
        self.distribution = QLabel()
        controls.addWidget(self.distribution)
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        controls.addWidget(self.stats_label)
        extreme_row = QHBoxLayout()
        for text, key in (("Show low example", "minimum_time"), ("Show high example", "maximum_time")):
            button = QPushButton(text)
            button.clicked.connect(
                lambda _, key=key: self._seek_timestamp(self._stats[key]) if self._stats else None
            )
            extreme_row.addWidget(button)
        controls.addLayout(extreme_row)
        self.check_label = QLabel()
        self.check_label.setWordWrap(True)
        controls.addWidget(self.check_label)
        self.live_check = QCheckBox("Approve this metric for this checked camera")
        self.live_check.setToolTip(
            "I confirm that the camera view and movement make this metric comparable to the reference. "
            "A repeat check demonstrates response, not universal cross-camera accuracy."
        )
        controls.addWidget(self.live_check)
        approve = QPushButton("Approve metric")
        approve.clicked.connect(self._approve)
        skip = QPushButton("Skip / remove metric")
        skip.clicked.connect(self._skip)
        self.approve_all_button = QPushButton("Approve all metrics")
        self.approve_all_button.clicked.connect(self._approve_all_metrics)
        controls.addWidget(approve)
        controls.addWidget(skip)
        controls.addWidget(self.approve_all_button)
        next_step = QPushButton("Next: head → shoulders → hips → torso → knees")
        next_step.clicked.connect(
            lambda: self.step_combo.setCurrentIndex((self.step_combo.currentIndex() + 1) % len(STEPS))
        )
        controls.addWidget(next_step)
        self.approved_label = QLabel("No metrics approved")
        self.approved_label.setWordWrap(True)
        controls.addWidget(self.approved_label)
        controls.addStretch()
        save = QPushButton("Save approved .tpc profile")
        save.clicked.connect(self._save)
        controls.addWidget(save)
        self.defaults_button = QPushButton("Use approved ranges as Settings defaults...")
        self.defaults_button.setEnabled(bridge is not None)
        self.defaults_button.clicked.connect(self._use_defaults)
        controls.addWidget(self.defaults_button)
        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setMinimumWidth(340)
        inspector_scroll.setMaximumWidth(380)
        inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inspector_scroll.setWidget(inspector)
        splitter.addWidget(inspector_scroll)
        splitter.setSizes([820, 340])
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.metric_combo.activated.connect(self._mark_current_metric_draft)
        self.passages_input.editingFinished.connect(self._on_passages_edited)
        self.lower_spin.valueChanged.connect(self._on_limits_edited)
        self.upper_spin.valueChanged.connect(self._on_limits_edited)
        self.view_combo.activated.connect(lambda: self._mark_current_metric_draft())
        self.live_check.clicked.connect(lambda: self._mark_current_metric_draft())
        self.view_combo.currentIndexChanged.connect(lambda: self._recompute())
        self._samples: dict[str, list[tuple[float, float | None]]] = {}
        if (
            self._editing_profile_path is not None
            and self.approved
            and all(target.view == "all" for target in self.approved.values())
        ):
            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentText("all")
            self.view_combo.blockSignals(False)
        self._recompute()
        if self.approved:
            self._approved_coach_data = self._evaluate_approved(self.approved)
            self._update_approved_label()
        self._change_step(STEPS[0])
        self._seek_frame(0)
        self.finished.connect(self._release)

    def _release(self, *_args: object) -> None:
        self.play_timer.stop()
        self._reader.close()
        self._archive.close()

    def _mark_current_metric_draft(self, *_args: object) -> None:
        key = self.metric_combo.currentData()
        if key is None:
            return
        self._draft_keys.add(key)
        self._draft_specs[key] = {
            "view": self.view_combo.currentText(),
            "passages_text": self.passages_input.text(),
            "lower": self.lower_spin.value(),
            "upper": self.upper_spin.value(),
            "camera_check_passed": self.live_check.isChecked(),
        }

    def _on_passages_edited(self) -> None:
        self._update_stats()
        self._mark_current_metric_draft()

    def _on_limits_edited(self, *_args: object) -> None:
        self._update_stats()
        self._mark_current_metric_draft()

    def _on_suggest_clicked(self) -> None:
        self._update_stats(suggest=True)
        self._mark_current_metric_draft()

    def _update_approved_label(self) -> None:
        if not self.approved:
            self.approved_label.setText("No metrics approved")
        else:
            self.approved_label.setText(
                "Approved: " + ", ".join(METRIC_DETAILS[k][1] for k in self.approved)
            )

    def _refresh_metric_combo_labels(self) -> None:
        self.metric_combo.blockSignals(True)
        try:
            for index in range(self.metric_combo.count()):
                key = self.metric_combo.itemData(index)
                if key in METRIC_DETAILS:
                    label = METRIC_DETAILS[key][1]
                    self.metric_combo.setItemText(
                        index, f"{label} ✓" if key in self.approved else label
                    )
        finally:
            self.metric_combo.blockSignals(False)

    def _restart_track(self, *_args: object) -> None:
        self._track_start = self.scrubber.value()
        self._anchors[self._track_start] = self.dancer_spin.value() - 1
        single_dancer = max((len(frame.poses) for frame in self.evidence.frames), default=0) <= 1
        if self._editing_profile_path is not None:
            self.confirm_identity.setChecked(single_dancer or self.confirm_identity.isChecked())
            current_key = self.metric_combo.currentData()
            self._recompute(suggest=current_key not in self.approved)
            if self.approved:
                self._approved_coach_data = self._evaluate_approved(self.approved)
            self.status_label.setText(
                f"Linked track at frame {self._track_start}; kept {len(self.approved)} approved metric(s)."
            )
        else:
            self.confirm_identity.setChecked(False)
            self.approved.clear()
            self._draft_keys.clear()
            self._draft_specs.clear()
            self._approved_coach_data = None
            self.approved_label.setText("Dancer selection changed; approve the new evidence again")
            self._refresh_metric_combo_labels()
            self._recompute()
        self._seek_frame(self.scrubber.value())

    def _auto_link_single_dancer(self) -> None:
        try:
            anchors = single_dancer_anchors(self.evidence, self.cfg.confidence)
        except ValueError as error:
            self.status_label.setText(str(error))
            return
        self._anchors = anchors
        self._track_start = min(anchors)
        self.dancer_spin.blockSignals(True)
        self.dancer_spin.setValue(anchors[self._track_start] + 1)
        self.dancer_spin.blockSignals(False)
        self.confirm_identity.setChecked(True)
        current_key = self.metric_combo.currentData()
        self._recompute(suggest=current_key not in self.approved)
        if self.approved:
            self._approved_coach_data = self._evaluate_approved(self.approved)
        self._seek_frame(self.scrubber.value())
        self.status_label.setText(
            f"Auto-linked single dancer across {len(anchors)} segment(s) over the full clip."
        )

    def _samples_for_view(self, view_mode: str) -> dict[str, list[tuple[float, float | None]]]:
        cached = self._samples_by_view.get(view_mode)
        if cached is not None:
            return cached
        cfg = self.cfg.model_copy(update={"allow_all_angles": view_mode != "side"})
        computed = metric_samples(self._selected_frames, cfg, view=view_mode)  # type: ignore[arg-type]
        self._samples_by_view[view_mode] = computed
        return computed

    def _recompute(self, *_args: object, suggest: bool = True) -> None:
        from ...calibration.evidence import Evidence

        anchors = sorted(self._anchors.items())
        frames: list[FrameObservation] = []
        if anchors[0][0] > 0:
            frames.extend(FrameObservation(frame.timestamp, frame.width, frame.height, [])
                          for frame in self.evidence.frames[:anchors[0][0]])
        for position, (start, dancer) in enumerate(anchors):
            end = anchors[position + 1][0] if position + 1 < len(anchors) else len(self.evidence.frames)
            evidence = Evidence(
                self.evidence.frames[start:end], self.evidence.model_hash,
                self.evidence.capture_profile, self.evidence.source, self.evidence.max_gap,
            )
            frames.extend(dancer_frames(evidence, dancer, self.cfg.confidence))
        self._selected_frames = frames
        self._samples_by_view.clear()
        selected_view = self.view_combo.currentText()
        view_mode = (
            selected_view if selected_view in ("side", "front", "all") else "side"
        )
        self._samples = self._samples_for_view(view_mode)
        if self.metric_combo.currentData():
            self._update_stats(suggest=suggest)

    def _change_step(self, step: str) -> None:
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for key, (group, label, _unit) in METRIC_DETAILS.items():
            if group == step:
                self.metric_combo.addItem(label, key)
        self.metric_combo.blockSignals(False)
        self._refresh_metric_combo_labels()
        self._change_metric()

    def _change_metric(self, *_args: object) -> None:
        key = self.metric_combo.currentData()
        if key is None:
            return
        hint = _METRIC_HINTS.get(key, "")
        self.metric_hint.setText(hint)
        self.metric_hint.setVisible(bool(hint))
        first, last = self.evidence.frames[0].timestamp, self.evidence.frames[-1].timestamp
        holds = self.holds.get(self.step_combo.currentText())
        passages = [holds[0]] if holds else [Passage(start=first, end=last + 1 / self._reader.fps)]
        draft_spec = self._draft_specs.get(key) if key in self._draft_keys else None
        approved = self.approved.get(key)
        target = approved or (
            self.base_profile.metrics.get(key) if self.base_profile is not None else None
        )
        if draft_spec is not None:
            self.passages_input.setText(draft_spec["passages_text"])
            view_changed = self.view_combo.currentText() != draft_spec["view"]
            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentText(draft_spec["view"])
            self.view_combo.blockSignals(False)
            if view_changed:
                self._recompute(suggest=False)
            for field, val in ((self.lower_spin, draft_spec["lower"]), (self.upper_spin, draft_spec["upper"])):
                field.blockSignals(True)
                field.setValue(val)
                field.blockSignals(False)
            self._update_stats(suggest=False)
            self.live_check.setChecked(bool(draft_spec["camera_check_passed"]))
            return
        if approved is not None:
            passages = approved.passages
        self.passages_input.setText("; ".join(f"{p.start:.6f}-{p.end:.6f}" for p in passages))
        if target is not None:
            view_changed = self.view_combo.currentText() != target.view
            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentText(target.view)
            self.view_combo.blockSignals(False)
            if view_changed:
                self._recompute(suggest=False)
            for field, val in ((self.lower_spin, target.lower), (self.upper_spin, target.upper)):
                field.blockSignals(True)
                field.setValue(val)
                field.blockSignals(False)
            self._update_stats(suggest=False)
        else:
            if self.view_combo.currentText() == "side":
                min_samples = max(2, int(self.cfg.calibration_min_usable_samples))
                side_valid = sum(
                    1 for _, value in self._samples.get(key, [])
                    if value is not None and math.isfinite(value)
                )
                all_valid = sum(
                    1 for _, value in self._samples_for_view("all").get(key, [])
                    if value is not None and math.isfinite(value)
                )
                profile_uses_all = (
                    self._editing_profile_path is not None
                    and bool(self.approved)
                    and all(existing.view == "all" for existing in self.approved.values())
                )
                if profile_uses_all or (
                    all_valid >= min_samples
                    and (side_valid < min_samples or side_valid < 0.5 * all_valid)
                ):
                    self.view_combo.blockSignals(True)
                    self.view_combo.setCurrentText("all")
                    self.view_combo.blockSignals(False)
                    self._samples = self._samples_for_view("all")
            self._update_stats(suggest=True)
        self.live_check.setChecked(bool(target.camera_check_passed) if target is not None else False)

    def _select_hold(self, index: int) -> None:
        holds = (
            self.holds.get("Walking", []) if index == 4 else
            self.holds.get(self.step_combo.currentText(), [])[index:index + 1]
        )
        if not holds:
            self.status_label.setText("That hold / drill was skipped or is unavailable.")
            return
        self.passages_input.setText("; ".join(f"{p.start:.6f}-{p.end:.6f}" for p in holds))
        self._update_stats()
        self._mark_current_metric_draft()

    def _parse_passages_text(self, text: str) -> list[Passage]:
        passages = []
        origin = self.evidence.frames[0].timestamp
        end = self.evidence.frames[-1].timestamp + 1 / self._reader.fps
        for value in text.split(";"):
            start_text, end_text = value.strip().split("-")
            passage = Passage(start=float(start_text), end=float(end_text))
            if passage.start < origin or passage.end > end + 1e-6:
                raise ValueError("Example passage is outside the reference timeline")
            passages.append(passage)
        return passages

    def _passages(self) -> list[Passage]:
        return self._parse_passages_text(self.passages_input.text())

    def _change_chart_zoom(self) -> None:
        factor = self.chart_zoom_combo.currentData()
        if isinstance(factor, int | float):
            self.chart.set_x_zoom(float(factor))

    def _tracked_end_timestamp(self) -> float:
        for selected_frame in reversed(self._selected_frames):
            if selected_frame.poses:
                return selected_frame.timestamp
        return self.evidence.frames[-1].timestamp

    def _build_chart_series(
        self, key: str
    ) -> tuple[list[tuple[float, float | None]], list[list[tuple[float, float]]], float]:
        origin = self.evidence.frames[0].timestamp
        points = [(t - origin, value) for t, value in self._samples[key]]
        segments: list[list[tuple[float, float]]] = []
        segment: list[tuple[float, float]] = []
        for timestamp, value in points:
            if value is None or (segment and timestamp - segment[-1][0] > self.evidence.max_gap):
                if segment:
                    segments.append(segment)
                segment = []
            if value is not None:
                segment.append((timestamp, value))
        if segment:
            segments.append(segment)
        tracked_end = self._tracked_end_timestamp() - origin
        last_valid = segments[-1][-1][0] if segments else 0.0
        effective_end = max(tracked_end, last_valid)
        if effective_end > 0.0:
            trimmed = [item for item in points if item[0] <= effective_end + 1e-6]
            if trimmed:
                points = trimmed
        return points, segments, origin

    def _first_dropout_time(self) -> float | None:
        seen_linked = False
        skip_until = -1
        origin = self.evidence.frames[0].timestamp
        for index, (selected_frame, raw_frame) in enumerate(
            zip(self._selected_frames, self.evidence.frames)
        ):
            if selected_frame.poses:
                seen_linked = True
                continue
            if index < skip_until or not seen_linked:
                continue
            if not has_linkable_pose(raw_frame, self.cfg.confidence):
                continue
            if is_static_pose_run(self.evidence.frames, index):
                skip_until = contiguous_pose_run_end(self.evidence.frames, index)
                continue
            return raw_frame.timestamp - origin
        return None

    def _update_stats(self, *, suggest: bool = False) -> None:
        key = self.metric_combo.currentData()
        if key is None or not self._samples:
            return
        self._stats = None
        dropout_time = self._first_dropout_time()
        try:
            stats = summarize_examples(
                self._samples[key], self._passages(), self.evidence.max_gap,
                None if suggest else self.lower_spin.value(),
                None if suggest else self.upper_spin.value(),
                suggest_lower_pct=self.cfg.calibration_suggest_lower_pct,
                suggest_upper_pct=self.cfg.calibration_suggest_upper_pct,
                min_usable_samples=self.cfg.calibration_min_usable_samples,
            )
            self._stats = stats
            if suggest:
                for field, value in ((self.lower_spin, stats["lower"]), (self.upper_spin, stats["upper"])):
                    field.blockSignals(True)
                    field.setValue(value)
                    field.blockSignals(False)
            lo_pct = self.cfg.calibration_suggest_lower_pct
            hi_pct = self.cfg.calibration_suggest_upper_pct
            pct_label = (
                "Observed middle 80%"
                if (lo_pct, hi_pct) == (10.0, 90.0)
                else f"Observed {lo_pct:g}th–{hi_pct:g}th%"
            )
            self.stats_label.setText(
                f"Median: {stats['median']:.4g} {METRIC_DETAILS[key][2]}\n"
                f"{pct_label}: {stats['p10']:.4g} … {stats['p90']:.4g}\n"
                f"Usable: {stats['valid_duration']:.2f} / {stats['selected_duration']:.2f}s "
                f"({stats['samples']} samples)\nInside edited range: {stats['inside_fraction']:.1%}"
            )
            self._draw_distribution(key)
            points, segments, origin = self._build_chart_series(key)
            states: dict[float, str] = {}
            target = self.approved.get(key)
            if (
                target is not None and self._approved_coach_data is not None
                and target.view == self.view_combo.currentText()
                and target.lower == self.lower_spin.value()
                and target.upper == self.upper_spin.value()
            ):
                for index, frame in enumerate(self._selected_frames):
                    entry = self._approved_coach_data["frames"][str(index)]
                    row = next(row for row in entry["calibration_rows"] if row["metric"] == key)
                    states[frame.timestamp - origin] = row["state"]
            self.chart.set_data(
                MetricData(points, segments, {}, 0, 0), points[-1][0], origin,
                bounds=(self.lower_spin.value(), self.upper_spin.value()),
                intervals=[(p.start - origin, p.end - origin) for p in self._passages()],
                states=states,
            )
            holds = self.holds.get(self.step_combo.currentText(), [])
            passed, reason = repeat_check(self._samples[key], holds, self.evidence.max_gap)
            is_camera = str(self.evidence.capture_profile.get("source", "")).startswith("camera:")
            self.live_check.setEnabled(passed and is_camera)
            if not passed or not is_camera:
                self.live_check.setChecked(False)
            self.check_label.setText(
                reason if is_camera else
                "Comparison only. A guided camera repeat check is required before live scoring."
            )
            if dropout_time is not None:
                self.status_label.setText(
                    f"Track stopped at {dropout_time:.2f}s — click 'Auto-link single dancer' or "
                    "'Select dancer / restart track here' to plot the rest of the clip."
                )
            else:
                self.status_label.setText("Edit examples independently from target limits, then approve this metric.")
        except (ValueError, KeyError) as error:
            self.stats_label.setText(f"Unavailable: {error}")
            self.distribution.clear()
            self.live_check.setEnabled(False)
            if key in self._samples and any(
                value is not None and math.isfinite(value) for _, value in self._samples[key]
            ):
                points, segments, origin = self._build_chart_series(key)
                try:
                    intervals = [(p.start - origin, p.end - origin) for p in self._passages()]
                except ValueError:
                    intervals = []
                self.chart.set_data(
                    MetricData(points, segments, {}, 0, 0), points[-1][0], origin,
                    bounds=(self.lower_spin.value(), self.upper_spin.value()),
                    intervals=intervals,
                    states={},
                )
            else:
                self.chart.set_data(None)
            hints = [str(error)]
            if self.view_combo.currentText() == "side":
                hints.append("Try switching 'Applicable view' to 'all'.")
            if dropout_time is not None:
                hints.append(f"Track stopped at {dropout_time:.2f}s; try 'Auto-link single dancer'.")
            self.status_label.setText(" ".join(hints))

    def _draw_distribution(self, key: str) -> None:
        passages = self._passages()
        values = [value for timestamp, value in self._samples[key]
                  if value is not None and math.isfinite(value)
                  and any(p.start <= timestamp < p.end for p in passages)]
        if len(values) >= 20:
            p2, p10, p90, p98 = np.percentile(values, [2, 10, 90, 98])
            lower_val = self.lower_spin.value()
            upper_val = self.upper_spin.value()
            core_span = max(float(p90 - p10), abs(upper_val - lower_val), 1e-3)
            display_left = max(float(min(values)), min(float(p2), lower_val - 0.5 * core_span))
            display_right = min(float(max(values)), max(float(p98), upper_val + 0.5 * core_span))
            if display_right > display_left:
                clipped = np.clip(values, display_left, display_right)
                counts, edges = np.histogram(clipped, bins=14, range=(display_left, display_right))
            else:
                counts, edges = np.histogram(values, bins=14)
        else:
            counts, edges = np.histogram(values, bins=14)
        pixmap = QPixmap(320, 100)
        pixmap.fill(QColor("#19151b"))
        painter = QPainter(pixmap)
        left, right = float(edges[0]), float(edges[-1])
        span = right - left or 1.0
        project = lambda value: max(8.0, min(312.0, 8 + (value - left) / span * 304))
        low, high = project(self.lower_spin.value()), project(self.upper_spin.value())
        painter.fillRect(QRectF(low, 3, max(0, high - low), 73), QColor("#315748"))
        peak = int(max(counts)) or 1
        for index, count in enumerate(counts):
            height = float(count) / peak * 66
            painter.fillRect(QRectF(8 + index * 304 / 14, 76 - height, 304 / 14 - 2, height),
                             QColor("#bb9974"))
        painter.setPen(QColor("#ddd5d0"))
        painter.drawText(8, 94, f"{left:.3g}")
        painter.drawText(
            QRectF(200, 80, 112, 18),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{right:.3g}",
        )
        painter.end()
        self.distribution.setPixmap(pixmap)

    def _build_approved_range(
        self,
        key: str,
        *,
        view: str,
        passages: list[Passage],
        lower: float | None,
        upper: float | None,
        camera_check_passed: bool,
    ) -> ApprovedRange:
        view_mode = view if view in ("side", "front", "all") else "side"
        samples = self._samples_for_view(view_mode).get(key, [])
        stats = summarize_examples(
            samples,
            passages,
            self.evidence.max_gap,
            lower,
            upper,
            suggest_lower_pct=self.cfg.calibration_suggest_lower_pct,
            suggest_upper_pct=self.cfg.calibration_suggest_upper_pct,
            min_usable_samples=self.cfg.calibration_min_usable_samples,
        )
        clean = {name: val for name, val in stats.items() if name not in ("minimum_time", "maximum_time")}
        return ApprovedRange.model_validate({
            **clean,
            "unit": METRIC_DETAILS[key][2],
            "view": view_mode,
            "passages": passages,
            "camera_check_passed": camera_check_passed,
        })

    def _commit_pending_drafts(self) -> None:
        if not (self._editing_profile_path is not None or self.approved):
            return
        if not self._draft_keys:
            return
        updated = dict(self.approved)
        changed = False
        for key in list(self._draft_keys):
            spec = self._draft_specs.get(key)
            if not spec:
                continue
            try:
                passages = self._parse_passages_text(spec["passages_text"])
                view = spec["view"] if spec["view"] in ("side", "front", "all") else "side"
                existing = updated.get(key)
                if (
                    existing is not None
                    and existing.view == view
                    and existing.lower == spec["lower"]
                    and existing.upper == spec["upper"]
                    and existing.camera_check_passed == bool(spec["camera_check_passed"])
                    and existing.passages == passages
                ):
                    continue
                updated[key] = self._build_approved_range(
                    key,
                    view=view,
                    passages=passages,
                    lower=spec["lower"],
                    upper=spec["upper"],
                    camera_check_passed=bool(spec["camera_check_passed"]),
                )
                changed = True
            except (ValueError, TypeError, KeyError):
                continue
        if changed:
            self._approved_coach_data = self._evaluate_approved(updated)
            self.approved = updated
            self._draft_keys.difference_update(updated.keys())
            self._update_approved_label()
            self._refresh_metric_combo_labels()

    def _approve(self) -> None:
        self._update_stats()
        if self._stats is None:
            return
        key = self.metric_combo.currentData()
        try:
            target = self._build_approved_range(
                key,
                view=self.view_combo.currentText(),
                passages=self._passages(),
                lower=self.lower_spin.value(),
                upper=self.upper_spin.value(),
                camera_check_passed=self.live_check.isChecked(),
            )
            approved = {**self.approved, key: target}
            compared = self._evaluate_approved(approved)
        except (ValueError, TypeError, KeyError) as error:
            self.status_label.setText(f"Could not evaluate approved metrics: {error}")
            return
        self.approved = approved
        self._draft_keys.discard(key)
        self._draft_specs.pop(key, None)
        self._approved_coach_data = compared
        self._update_approved_label()
        self._refresh_metric_combo_labels()
        self._update_stats()
        self._seek_frame(self.scrubber.value())

    def _approve_all_metrics(self) -> None:
        current_key = self.metric_combo.currentData()
        updated = dict(self.approved)
        skipped: list[str] = []
        first, last = self.evidence.frames[0].timestamp, self.evidence.frames[-1].timestamp
        fallback_passage = [Passage(start=first, end=last + 1 / self._reader.fps)]
        if updated:
            existing_first = next(iter(updated.values()))
            if existing_first.passages:
                fallback_passage = list(existing_first.passages)

        for key, (step_name, label, _unit) in METRIC_DETAILS.items():
            if key == current_key and self._stats is not None:
                try:
                    updated[key] = self._build_approved_range(
                        key,
                        view=self.view_combo.currentText(),
                        passages=self._passages(),
                        lower=self.lower_spin.value(),
                        upper=self.upper_spin.value(),
                        camera_check_passed=self.live_check.isChecked(),
                    )
                    continue
                except (ValueError, TypeError, KeyError):
                    pass
            if key in updated and key not in self._draft_keys:
                continue
            spec = self._draft_specs.get(key) if key in self._draft_keys else None
            holds = self.holds.get(step_name)
            if spec is not None:
                try:
                    passages = self._parse_passages_text(spec["passages_text"])
                    preferred_view = spec["view"] if spec["view"] in ("side", "front", "all") else "side"
                    lower, upper = spec["lower"], spec["upper"]
                    cam_ok = bool(spec["camera_check_passed"])
                except (ValueError, TypeError, KeyError):
                    passages = [holds[0]] if holds else fallback_passage
                    preferred_view = self.view_combo.currentText()
                    lower, upper = None, None
                    cam_ok = False
            else:
                passages = [holds[0]] if holds else fallback_passage
                preferred_view = self.view_combo.currentText()
                lower, upper = None, None
                cam_ok = False

            candidates = [preferred_view] if preferred_view == "all" else [preferred_view, "all"]
            approved_target: ApprovedRange | None = None
            for view_mode in candidates:
                try:
                    approved_target = self._build_approved_range(
                        key,
                        view=view_mode,
                        passages=passages,
                        lower=lower,
                        upper=upper,
                        camera_check_passed=cam_ok,
                    )
                    break
                except (ValueError, TypeError, KeyError):
                    continue
            if approved_target is not None:
                updated[key] = approved_target
            else:
                skipped.append(label)

        if not updated:
            self.status_label.setText("No metrics had enough reliable samples to approve.")
            return
        try:
            compared = self._evaluate_approved(updated)
        except (ValueError, TypeError, KeyError) as error:
            self.status_label.setText(f"Could not evaluate approved metrics: {error}")
            return
        self.approved = updated
        self._draft_keys.difference_update(updated.keys())
        for key in updated:
            self._draft_specs.pop(key, None)
        self._approved_coach_data = compared
        self._update_approved_label()
        self._refresh_metric_combo_labels()
        self._update_stats(suggest=False)
        self._seek_frame(self.scrubber.value())
        if skipped:
            self.status_label.setText(
                f"Approved {len(updated)}/{len(METRIC_DETAILS)} metrics. "
                f"Skipped (insufficient samples): {', '.join(skipped)}"
            )
        else:
            self.status_label.setText(
                f"Approved all {len(updated)} metrics. Click 'Save approved .tpc profile' to save."
            )

    def _skip(self) -> None:
        key = self.metric_combo.currentData()
        approved = dict(self.approved)
        approved.pop(key, None)
        self._draft_keys.discard(key)
        self._draft_specs.pop(key, None)
        try:
            compared = self._evaluate_approved(approved) if approved else None
        except (ValueError, TypeError, KeyError) as error:
            self.status_label.setText(f"Could not evaluate approved metrics: {error}")
            return
        self.approved = approved
        self._approved_coach_data = compared
        self._update_approved_label()
        self._refresh_metric_combo_labels()
        self._update_stats()
        self._seek_frame(self.scrubber.value())

    def _evaluate_approved(self, approved: dict[str, ApprovedRange]) -> dict[str, Any]:
        data = {
            "width": self.evidence.frames[0].width,
            "height": self.evidence.frames[0].height,
            "frames": {
                str(index): {
                    "timestamp": frame.timestamp,
                    "measurement_poses": [
                        {name: [lm.x, lm.y, lm.z, lm.visibility, lm.presence]
                         for name, lm in pose.items()}
                        for pose in frame.poses
                    ],
                }
                for index, frame in enumerate(self._selected_frames)
            },
        }
        return compare_approved_coach_data(data, self.cfg, approved)

    def _approved_profile(self) -> RangeProfile:
        self._commit_pending_drafts()
        if file_sha256(self._archive.path) != self._source_hash:
            raise ValueError("Reference changed while reviewing; reopen it before saving a profile")
        if not self.confirm_identity.isChecked():
            raise ValueError("Confirm that manually linked passages follow the same dancer")
        if not self.approved:
            raise ValueError("Approve at least one metric; skipped metrics will remain unavailable")
        name = self.name_input.text().strip()
        source: dict[str, Any] = {
            **self.evidence.source, "dancer_at_track_start": self.dancer_spin.value() - 1,
            "track_start_frame": self._track_start,
            "dancer_anchors": [
                {"frame": frame, "dancer": dancer} for frame, dancer in sorted(self._anchors.items())
            ],
        }
        extra: dict[str, Any] = {}
        if self._editing_profile_path is not None and self.base_profile is not None:
            if "target_reference" in self.base_profile.source:
                source["target_reference"] = self.base_profile.source["target_reference"]
            if name == self.base_profile.name:
                extra["id"] = self.base_profile.id
                extra["created_at_utc"] = self.base_profile.created_at_utc
        elif self.base_profile is not None:
            source["target_reference"] = self.base_profile.source
        return RangeProfile(
            **extra,
            name=name, context=self.context_combo.currentText(),
            model_hash=self.evidence.model_hash, capture_profile=self.evidence.capture_profile,
            source=source,
            metrics=self.approved,
            camera_checks={"holds": {
                key: [hold.model_dump() for hold in holds] for key, holds in self.holds.items()
            }},
        )

    def _use_defaults(self) -> None:
        if self.bridge is None:
            self.status_label.setText("Settings bridge is unavailable.")
            return
        try:
            if self.bridge.use_calibration_defaults(self._approved_profile(), self):
                self.status_label.setText("Default ranges saved to settings.toml.")
        except (OSError, ValueError, TypeError) as error:
            self.status_label.setText(f"Could not use calibration defaults: {error}")

    def _save(self) -> None:
        try:
            result = self._approved_profile()
            if (
                self._editing_profile_path is not None
                and self.base_profile is not None
                and result.name == self.base_profile.name
            ):
                path = self._editing_profile_path
                overwrite = True
            else:
                path = calibration_path(calibration_directory(self.cfg.data_dir), result.name)
                overwrite = (
                    self._editing_profile_path is not None
                    and path.resolve() == self._editing_profile_path.resolve()
                )
            save_range_profile(path, result, overwrite=overwrite)
            self.saved_path = path
            self.accept()
        except (OSError, ValueError, TypeError) as error:
            self.status_label.setText(f"Calibration not saved: {error}")

    def _save_selection(self) -> None:
        from ...archive import _archive_lock, _rewrite_members

        try:
            if not self.confirm_identity.isChecked():
                raise ValueError("Confirm that the linked passages follow the same dancer first")
            path = self._archive.path
            with _archive_lock(path):
                if file_sha256(path) != self._source_hash:
                    raise ValueError("Reference changed while reviewing; reopen before saving selection")
                from zipfile import ZipFile

                with ZipFile(path) as archive:
                    descriptor = json.loads(archive.read("reference.json"))
                descriptor["selected_dancer"] = min(self._anchors.items())[1]
                descriptor["dancer_anchors"] = [
                    {"frame": frame, "dancer": dancer} for frame, dancer in sorted(self._anchors.items())
                ]
                _rewrite_members(path, {"reference.json": json.dumps(descriptor).encode()})
                self._source_hash = file_sha256(path)
                self.evidence.source["archive_sha256"] = self._source_hash
            self.status_label.setText("Dancer selection saved. Measurements and raw video are unchanged.")
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.status_label.setText(f"Reference selection not saved: {error}")
    def _seek_timestamp(self, timestamp: float) -> None:
        index = min(
            range(len(self.evidence.frames)),
            key=lambda index: abs(self.evidence.frames[index].timestamp - timestamp),
        )
        self.scrubber.setValue(index)

    def _seek_frame(self, index: int) -> None:
        try:
            frame = self.evidence.frames[index]
            image = render_coach_frame_overlay(
                self._reader.frame(index), self._coach_data, index, self.cfg,
                verbose_skeleton=True, show_vertical_axis=True, show_footer=False,
                lines_only=True,
            )
            image = render_movement_overlay(
                image, self._movement_data, index, self._reader.fps,
                frame_timestamp=frame.timestamp - self.evidence.frames[0].timestamp,
                show_pivots=True, show_steps=True,
                step_history_seconds=self.cfg.replay_step_history_seconds,
            )
            if self._approved_coach_data is not None:
                entry = self._approved_coach_data["frames"][str(index)]
                image = render_approved_feedback(
                    image, self._selected_frames[index], self.cfg, self.approved,
                    entry["calibration_rows"],
                )
            height, width = image.shape[:2]
            for number, pose in enumerate(frame.poses, start=1):
                hip = pose.get("left_hip") or pose.get("right_hip")
                if hip is not None and reliable(hip, self.cfg.confidence):
                    position = (round(hip.x * width), round(hip.y * height))
                    color = (90, 230, 150) if number == self.dancer_spin.value() else (230, 200, 150)
                    cv2.putText(image, str(number), position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            self.stage.set_frame(numpy_to_qimage(image))
            self.chart.set_playhead(frame.timestamp)
            self.time_label.setText(f"{frame.timestamp:.3f}s · frame {index}")
            if self.play_button.text() == "Pause":
                if index + 1 == len(self.evidence.frames):
                    self.play_timer.stop()
                    self.play_button.setText("Play")
                else:
                    gap = self.evidence.frames[index + 1].timestamp - frame.timestamp
                    self.play_timer.start(max(1, round(gap * 1000)))
        except (OSError, ValueError, RuntimeError) as error:
            self.status_label.setText(f"Cannot display reference frame: {error}")

    def _toggle_play(self) -> None:
        if self.play_button.text() == "Pause":
            self.play_timer.stop()
            self.play_button.setText("Play")
        else:
            self.play_button.setText("Pause")
            self._seek_frame(self.scrubber.value())

    def _advance_frame(self) -> None:
        self.scrubber.setValue(min(self.scrubber.maximum(), self.scrubber.value() + 1))
