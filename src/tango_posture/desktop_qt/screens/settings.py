"""Settings management screen with grouped configuration and TOML import/export."""

from __future__ import annotations

import re
import tomllib
from importlib.resources import files as package_files
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...calibration.models import METRIC_DETAILS
from ...config import (
    SETTING_DESCRIPTIONS,
    SETTING_ENGINE_SCOPES,
    SETTING_GROUPS,
    Config,
    RangeDefault,
    set_setting,
    setting_value,
)
from ..calibration.normalization_panel import MeasurementNormalizationPanel
from ..router import Router
from ..service_bridge import ServiceBridge
from ..theme import COLOR_COPPER, COLOR_TEXT_MUTED

_FLOAT_RANGES = {
    "confidence": (0.0, 1.0),
    "calibration_suggest_lower_pct": (0.0, 99.9999),
    "calibration_suggest_upper_pct": (0.0001, 100.0),
    "min_filter_time_constant": (0.000001, 10.0),
    "shoulder_visibility_floor": (0.0, 1.0),
    "shoulder_ear_hip_scale_factor": (0.0001, 10.0),
    "shoulder_min_ear_hip_px": (0.0001, 10000.0),
    "shoulder_nose_ear_scale_factor": (0.0001, 10.0),
    "shoulder_ear_ear_scale_factor": (0.0001, 10.0),
    "shoulder_width_scale_factor": (0.0001, 10.0),
    "shoulder_min_ref_scale_px": (0.0001, 10000.0),
    "shoulder_baseline_clamp_floor": (0.0001, 10.0),
    "shoulder_adaptive_min_clearance": (0.0001, 100.0),
    "shoulder_adaptive_max_clearance": (0.0002, 100.0),
    "shoulder_adaptive_percentile": (0.0001, 100.0),
    "shoulder_recovery_min_pct": (0.0, 1000.0),
    "shoulder_recovery_ratio": (0.0001, 1.0),
    "knee_recovery_margin": (0.0, 180.0),
    "knee_visibility_floor": (0.0, 1.0),
    "knee_min_segment_px": (0.0001, 10000.0),
    "knee_tiebreak_other_max_vis": (0.0, 1.0),
    "knee_tiebreak_min_advantage": (0.0, 1.0),
    "replay_recovery_max_gap_s": (0.05, 2.00),
    "replay_recovery_context_s": (0.10, 2.00),
    "replay_pivot_interpolation_max_gap_s": (0.0, 1.00),
    "replay_recovery_budget_s": (1.0, 60.0),
    "tango_bpm": (20.0, 240.0),
    "dancer_analysis.auto_detection_window_s": (0.5, 600.0),
    "dancer_analysis.primary_coverage_min": (0.01, 1.0),
    "dancer_analysis.primary_duration_min_s": (0.1, 600.0),
    "dancer_analysis.second_joint_coverage_min": (0.01, 1.0),
    "dancer_analysis.second_sustained_min_s": (0.1, 600.0),
    "dancer_analysis.brief_second_max_s": (0.0, 60.0),
    "dancer_analysis.together_min_s": (0.1, 60.0),
    "dancer_analysis.together_max_hip_separation_torso_lengths": (0.1, 10.0),
    "dancer_analysis.together_direction_cosine_min": (-1.0, 1.0),
    "dancer_analysis.identity_max_innovation_torso_lengths": (0.05, 5.0),
    "dancer_analysis.identity_margin_torso_lengths": (0.0, 2.0),
    "dancer_analysis.max_observation_gap_s": (0.02, 5.0),
    "dancer_analysis.display_interpolation_max_gap_s": (0.0, 3.0),
    "dancer_analysis.velocity_filter_window_s": (0.03, 2.0),
    "dancer_analysis.persistence_s": (0.02, 2.0),
    "dancer_analysis.pairing_window_s": (0.05, 5.0),
    "dancer_analysis.start_speed": (0.02, 5.0),
    "dancer_analysis.stop_speed": (0.0, 4.9),
    "dancer_analysis.slowing_fraction": (0.01, 0.99),
    "dancer_analysis.transition_min_coverage": (0.05, 1.0),
    "dancer_analysis.secondary_speed_rise": (0.01, 5.0),
    "dancer_analysis.axis_smoothing_s": (0.0, 5.0),
    "dancer_analysis.overlay_smoothing_s": (0.0, 5.0),
    "dancer_analysis.axis_bridge_max_gap_s": (0.0, 3.0),
    "dancer_analysis.overlay_min_landmark_visibility": (0.0, 1.0),
    "dancer_analysis.duplicate_pose_max_distance_torso_lengths": (0.01, 2.0),
    "dancer_analysis.fixed_scale_min_ratio": (0.1, 0.99),
    "dancer_analysis.fixed_scale_max_ratio": (1.05, 10.0),
    "dancer_analysis.shot_cut_max_frame_gap_s": (0.1, 30.0),
    "dancer_analysis.shot_cut_jump_torso_lengths": (0.2, 10.0),
    "dancer_analysis.turn_exit_match_window_s": (0.1, 30.0),
    "dancer_analysis.close_embrace_max_tilt_deg": (5.0, 90.0),
}

_INT_RANGES = {
    "calibration_min_usable_samples": (2, 1_000_000),
    "shoulder_adaptive_max_samples": (1, 1_000_000),
    "shoulder_adaptive_min_samples": (1, 1_000_000),
    "max_consecutive_high_ankle_steps": (1, 1_000),
}


def _guide_markdown_path() -> Path:
    return Path(__file__).resolve().parents[4] / "docs" / "TANGO_POSTURE_GUIDE.md"


def _load_guide_markdown() -> str:
    guide_path = _guide_markdown_path()
    if guide_path.is_file():
        return guide_path.read_text(encoding="utf-8")
    try:
        packaged = package_files("tango_posture").joinpath("TANGO_POSTURE_MANUAL.md")
        return packaged.read_text(encoding="utf-8")
    except (OSError, ValueError, FileNotFoundError):
        return ""


def load_parameter_guides() -> dict[str, dict[str, str]]:
    """Parse per-parameter explanations and worked examples from the unified manual."""
    guides: dict[str, dict[str, str]] = {}
    text = _load_guide_markdown()
    if text:
        sections = re.findall(
            r"^### ([^\n]+) - `([^`]+)`\n(.*?)(?=^### |^## |\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        for title, key, raw_body in sections:
            body = raw_body.strip()
            # Drop trailing overview notes after the final calibration-default item.
            if key == "knee_flexion" and "\n\nUse **Use as Settings defaults**" in body:
                body = body.split("\n\nUse **Use as Settings defaults**", 1)[0].strip()
            meta_line = ""
            example_text = ""
            body_for_explanation = body
            meta_match = re.match(r"^\*\*(Default|Unit):\*\*\s*([^\n]+)", body_for_explanation)
            if meta_match:
                meta_line = f"{meta_match.group(1)}: {meta_match.group(2).strip()}"
                body_for_explanation = body_for_explanation[meta_match.end():].strip()
            example_idx = body_for_explanation.find("**Example:**")
            if example_idx >= 0:
                explanation_text = body_for_explanation[:example_idx].strip()
                example_raw = body_for_explanation[example_idx + len("**Example:**"):].strip()
                example_text = " ".join(example_raw.splitlines()).strip()
            else:
                explanation_text = " ".join(body_for_explanation.splitlines()).strip()
            explanation_text = " ".join(explanation_text.splitlines()).strip()
            if not explanation_text:
                if key in SETTING_DESCRIPTIONS:
                    explanation_text = SETTING_DESCRIPTIONS[key]
                elif key in METRIC_DETAILS:
                    group, label, unit = METRIC_DETAILS[key]
                    explanation_text = f"{group} metric '{label}' measured in {unit}."
            guides[key] = {
                "title": title.strip(),
                "meta": meta_line.replace("**", ""),
                "explanation": explanation_text,
                "example": example_text,
            }
    return guides

_SCOPE_SUMMARIES = {
    "All": (
        "Showing all 15 configuration tabs across Common (Both Engines), Single Dancer Engine, "
        "and Couple Dancer Engine."
    ),
    "Common (Both Engines)": (
        "Shared settings used by both engines: pose models, Auto scene routing, detection confidence, "
        "music defaults, and Ollama Review coaching."
    ),
    "Single Dancer Engine": (
        "Settings exclusive to the Solo Dancer Engine: side/view calibration, 33-landmark posture alerts, "
        "embrace/knee heuristics, footwork, gap recovery, normalization, and calibration defaults."
    ),
    "Couple Dancer Engine": (
        "Settings exclusive to the Couple Dancer Engine: two-dancer identity assignment, pair torso axes, "
        "velocity/coordination lag thresholds, transitions, and turn suggestions."
    ),
}

_GROUP_HINTS = {
    "Common · Scene Routing": (
        "Common engine router settings governing Auto detection over the opening clip window and "
        "co-movement qualification between Single and Couple engines."
    ),
    "Common · Detection / Music": (
        "Shared MediaPipe landmark confidence, continuous observation gap ceiling, overlay line "
        "smoothing, fallback tempo, and soundtrack beat extraction."
    ),
    "Single · Quality / Calibration": (
        "Single-dancer quality controls determine whether evidence is usable, not whether posture is "
        "correct. Approve reference ranges using example passages and acceptable limits in Calibration."
    ),
    "Single · Head / Shoulders": (
        "Fallback descriptive warning controls for Single Dancer Engine. A selected calibration or enabled "
        "Calibration defaults supplies approved ranges instead; Review alerts require a selected calibration file."
    ),
    "Single · Embrace / Knees": (
        "Single-dancer role/embrace recognition runs independently of calibration. Knee-warning thresholds "
        "are fallbacks, replaced by selected calibration ranges or enabled Calibration defaults."
    ),
    "Single · Torso / Footwork": (
        "Single-dancer fallback hip-offset and slow-tango ankle-lift controls. Selected calibration ranges "
        "or enabled Calibration defaults replace posture limits; tempo ceiling is compared against Common · Detection / Music."
    ),
    "Single · Replay / Recovery": (
        "Single-dancer Review footprint trail, backward lower-limb tracking recovery, and display-only "
        "skeleton gap interpolation settings."
    ),
    "Couple · Tracking / Interpolation": (
        "Couple-only identity matching innovation/margin gates, continuous tracking span gap limits, "
        "and pair display-only 2D interpolation."
    ),
    "Couple · Axes / Coordination": (
        "Couple-only center-of-mass velocity regression, start/slow/settle persistence and pairing windows, "
        "transition smoothness coverage, and optional turn marker suggestions."
    ),
}


class SettingsScreen(QScrollArea):
    def __init__(self, bridge: ServiceBridge, router: Router, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.router = router
        self.field_widgets: dict[str, QWidget] = {}
        self.range_widgets: dict[str, tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox, QComboBox]] = {}
        self.help_toggles: dict[str, QPushButton] = {}
        self.help_panels: dict[str, QFrame] = {}
        self.scope_buttons: dict[str, QPushButton] = {}
        self._current_engine_scope = "All"
        self._parameter_guides = load_parameter_guides()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        self.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Application Settings")
        title.setProperty("class", "title")
        subtitle = QLabel(
            "Configure shared engine routing & models, Solo Dancer posture/footwork, and Couple Axis/Coordination parameters."
        )
        subtitle.setProperty("class", "subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.save_config_btn = QPushButton("Save Settings")
        self.save_config_btn.setProperty("class", "primary")
        self.save_config_btn.clicked.connect(self._save_settings)
        self.toggle_all_guides_btn = QPushButton("Expand All Guides")
        self.toggle_all_guides_btn.setCheckable(True)
        self.toggle_all_guides_btn.clicked.connect(self._toggle_all_guides)
        self.export_btn = QPushButton("Export TOML…")
        self.export_btn.clicked.connect(self._export_toml)
        self.import_btn = QPushButton("Import TOML…")
        self.import_btn.clicked.connect(self._import_toml)
        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)
        header.addWidget(self.save_config_btn)
        header.addWidget(self.toggle_all_guides_btn)
        header.addWidget(self.export_btn)
        header.addWidget(self.import_btn)
        header.addWidget(self.reset_btn)
        layout.addLayout(header)

        scope_bar = QHBoxLayout()
        scope_bar.setContentsMargins(0, 0, 0, 0)
        scope_bar.setSpacing(8)
        scope_lbl = QLabel("Engine scope:")
        scope_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-weight: 600;")
        scope_bar.addWidget(scope_lbl)
        for scope_name in ("All", *SETTING_ENGINE_SCOPES):
            btn = QPushButton(scope_name)
            btn.setCheckable(True)
            btn.setChecked(scope_name == "All")
            btn.clicked.connect(lambda _checked=False, s=scope_name: self.set_engine_scope(s))
            self.scope_buttons[scope_name] = btn
            scope_bar.addWidget(btn)
        scope_bar.addStretch()
        layout.addLayout(scope_bar)

        self.scope_summary_label = QLabel(_SCOPE_SUMMARIES["All"])
        self.scope_summary_label.setWordWrap(True)
        self.scope_summary_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(self.scope_summary_label)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        cfg = self.bridge.config
        for group, fields in SETTING_GROUPS.items():
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            if group in _GROUP_HINTS:
                hint = QLabel(_GROUP_HINTS[group])
                hint.setWordWrap(True)
                form.addRow(hint)
            if group == "Common · Files / Models":
                model_hint = QLabel(
                    "Offline analysis uses the model selected below. The Lite model file supplies "
                    "the Lite path; Full and Heavy use their own files. Missing models are downloaded "
                    "and cached locally; Full and Heavy take more processing time."
                )
                model_hint.setWordWrap(True)
                form.addRow(model_hint)
            for field in fields:
                editor, row_widget = self._create_editor(field, setting_value(cfg, field))
                self.field_widgets[field] = editor
                wrapped = self._wrap_with_guide(
                    field,
                    row_widget,
                    fallback_summary=SETTING_DESCRIPTIONS.get(field, ""),
                )
                form.addRow(self._field_label(field), wrapped)
            self.tabs.addTab(page, group)
        norm_page = QWidget()
        norm_layout = QVBoxLayout(norm_page)
        norm_layout.setContentsMargins(16, 16, 16, 16)
        norm_layout.setSpacing(12)
        norm_hint = QLabel(
            "Re-express Review metric observations on an approved calibration profile's scale "
            "using matched holds or an explicit offset, without altering the saved .tpc bounds."
        )
        norm_hint.setWordWrap(True)
        norm_layout.addWidget(norm_hint)
        self.normalization_panel = MeasurementNormalizationPanel(self.bridge, norm_page)
        self._parameter_guides.setdefault("measurement_normalization", {
            "meta": "Per calibration profile & recording pair · Additive Review offset",
            "explanation": (
                "Re-expresses Review metric observations on the approved calibration profile's scale "
                "before temporal evaluation, while leaving .tpc bounds unchanged. Use ONLY to remove "
                "a verified camera, costume, or teacher-vs-student anatomical neutral difference — never "
                "to hide genuine student posture habits."
            ),
            "example": (
                "Worked example (head pitch): reference fit hold reads 3.0° and practice fit hold "
                "reads 7.0° -> signed offset -4.0°. On separate Check holds (3.2° vs 7.4°), "
                "7.4° + (-4.0°) = 3.4° (error 0.2° <= 1.0° -> Verified). In Review, raw 8.0° maps to "
                "8.0° + (-4.0°) = 4.0°, inside 1.0°..5.0° (WITHIN_REFERENCE)."
            ),
        })
        wrapped_norm = self._wrap_with_guide("measurement_normalization", self.normalization_panel)
        norm_layout.addWidget(wrapped_norm, 1)
        self.tabs.addTab(norm_page, "Measurement normalization")
        ranges_page = QWidget()
        ranges_form = QFormLayout(ranges_page)
        ranges_hint = QLabel(
            "Exact ranges copied from calibration, used when no calibration is selected. "
            "Enabled ranges replace the legacy posture thresholds; disabled metrics stay unavailable. "
            "They do not grant camera-scoring approval."
        )
        ranges_hint.setWordWrap(True)
        ranges_form.addRow(ranges_hint)
        for key, (metric_group, label, unit) in METRIC_DETAILS.items():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            enabled = QCheckBox("Use")
            lower, upper = QDoubleSpinBox(), QDoubleSpinBox()
            view = QComboBox()
            view.addItems(["side", "front", "all"])
            for spin, description in ((lower, "minimum"), (upper, "maximum")):
                spin.setDecimals(12)
                spin.setRange(-1e9, 1e9)
                spin.setAccessibleName(f"{label} {description}")
                enabled.toggled.connect(spin.setEnabled)
            enabled.toggled.connect(view.setEnabled)
            for widget in (enabled, lower, upper, view):
                row_layout.addWidget(widget)
            wrapped_range = self._wrap_with_guide(
                key,
                row,
                fallback_summary=f"{metric_group} metric ({unit}).",
            )
            ranges_form.addRow(f"{label} ({unit})", wrapped_range)
            self.range_widgets[key] = (enabled, lower, upper, view)
        self.tabs.addTab(ranges_page, "Calibration defaults")
        layout.addWidget(self.tabs, 1)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(8)
        self.apply_btn = QPushButton("Save Settings")
        self.apply_btn.setFixedHeight(38)
        self.apply_btn.clicked.connect(self._save_settings)
        actions_row.addWidget(self.apply_btn)
        self.save_btn = QPushButton("Apply && Save config.toml / settings.toml")
        self.save_btn.setProperty("class", "primary")
        self.save_btn.setFixedHeight(38)
        self.save_btn.clicked.connect(self._save_toml)
        actions_row.addWidget(self.save_btn, 1)
        layout.addLayout(actions_row)
        self.save_toml_btn = self.save_btn

        self._set_legacy_widget_aliases()
        self._apply_config(cfg)
        self.bridge.config_changed.connect(self._apply_config)

    def _create_editor(self, field: str, value: object) -> tuple[QWidget, QWidget]:
        editor: QWidget
        if isinstance(value, bool):
            checkbox = QCheckBox("Enabled")
            checkbox.setChecked(value)
            editor = checkbox
        elif field in {
            "side",
            "facing",
            "replay_joint_interpolation",
            "analysis_model",
            "dancer_analysis.default_dancer_mode",
        }:
            combo = QComboBox()
            combo.addItems(
                ["auto", "single", "couple"]
                if field == "dancer_analysis.default_dancer_mode"
                else ["lite", "full", "heavy"]
                if field == "analysis_model"
                else ["linear", "quadratic"]
                if field == "replay_joint_interpolation"
                else ["left", "right"]
            )
            combo.setCurrentText(str(value))
            editor = combo
        elif isinstance(value, int):
            int_spin = QSpinBox()
            int_spin.setRange(*_INT_RANGES.get(field, (0, 2_147_483_647)))
            int_spin.setValue(value)
            editor = int_spin
        elif isinstance(value, float):
            float_spin = QDoubleSpinBox()
            float_spin.setDecimals(12)
            float_spin.setSingleStep(
                0.0001
                if field == "min_filter_time_constant"
                else 1.0
                if field == "replay_recovery_budget_s"
                else 0.05
                if _FLOAT_RANGES.get(field) == (0.0, 1.0)
                or field in {
                    "replay_recovery_max_gap_s",
                    "replay_recovery_context_s",
                    "replay_pivot_interpolation_max_gap_s",
                }
                else 0.1
            )
            float_spin.setRange(
                *_FLOAT_RANGES.get(field, (-1_000_000_000.0, 1_000_000_000.0))
            )
            float_spin.setValue(value)
            editor = float_spin
        else:
            line_edit = QLineEdit(str(value))
            editor = line_edit

        editor.setAccessibleName(field.replace(".", " "))
        editor.setToolTip(f"Configuration key: {field}")
        if field not in {"data_dir", "model_path"} or not isinstance(editor, QLineEdit):
            return editor, editor

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(editor, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(lambda: self._browse_path(field, editor))
        row_layout.addWidget(browse_btn)
        return editor, row

    @staticmethod
    def _field_label(field: str) -> str:
        labels = {
            "side": "Measurement side",
            "facing": "Facing direction",
            "capture_profile": "Capture setup name",
            "model_path": "Lite model file",
            "analysis_model": "Offline analysis model",
            "data_dir": "Data folder",
            "confidence": "Minimum landmark confidence",
            "calibration_suggest_lower_pct": "Suggested lower percentile (%)",
            "calibration_suggest_upper_pct": "Suggested upper percentile (%)",
            "calibration_min_usable_samples": "Min usable calibration samples",
            "smoothing": "Measurement smoothing (seconds)",
            "overlay_smoothing": "Overlay smoothing (seconds)",
            "min_filter_time_constant": "Minimum filter time constant (s)",
            "replay_joint_interpolation": "Review joint interpolation",
            "replay_joint_max_gap": "Maximum joint gap to fill (seconds; 0 disables)",
            "replay_recovery_auto": "Recover short tracking gaps during analysis",
            "replay_recovery_max_gap_s": "Maximum recoverable gap",
            "replay_recovery_context_s": "Context on each side",
            "replay_pivot_interpolation_max_gap_s": "Maximum interpolated leg gap during a pivot",
            "replay_show_recovered_joints": "Show estimated joints",
            "replay_recovery_budget_s": "Automatic recovery budget",
            "head_down_angle": "Head down angle limit (degrees)",
            "shoulder_raise_threshold": "Shoulder tension limit (%)",
            "shoulder_visibility_floor": "Shoulder helper visibility floor",
            "shoulder_ear_hip_scale_factor": "Shoulder ear-hip scale factor",
            "shoulder_min_ear_hip_px": "Shoulder min ear-hip height (px)",
            "shoulder_nose_ear_scale_factor": "Shoulder nose-ear scale factor",
            "shoulder_ear_ear_scale_factor": "Shoulder ear-ear scale factor",
            "shoulder_width_scale_factor": "Shoulder width scale factor",
            "shoulder_min_ref_scale_px": "Shoulder min reference scale (px)",
            "shoulder_baseline_clamp_floor": "Shoulder baseline clamp floor",
            "shoulder_adaptive_max_samples": "Adaptive baseline max samples",
            "shoulder_adaptive_min_samples": "Adaptive baseline min samples",
            "shoulder_adaptive_min_clearance": "Adaptive baseline min clearance",
            "shoulder_adaptive_max_clearance": "Adaptive baseline max clearance",
            "shoulder_adaptive_percentile": "Adaptive baseline percentile (%)",
            "shoulder_recovery_min_pct": "Shoulder recovery min floor (%)",
            "shoulder_recovery_ratio": "Shoulder recovery threshold ratio",
            "knee_recovery_margin": "Knee recovery margin (degrees)",
            "knee_visibility_floor": "Knee helper visibility floor",
            "knee_min_segment_px": "Knee min segment length (px)",
            "knee_tiebreak_other_max_vis": "Knee tie-break other max visibility",
            "knee_tiebreak_min_advantage": "Knee tie-break min vis advantage",
            "slow_tango_max_bpm": "Slow tango max BPM",
            "tango_bpm": "Fallback tempo (BPM)",
            "dancer_analysis.default_dancer_mode": "Default dancer mode",
            "dancer_analysis.auto_detection_window_s": "Auto detection window (seconds)",
            "dancer_analysis.primary_coverage_min": "Primary coverage minimum",
            "dancer_analysis.primary_duration_min_s": "Primary minimum duration (seconds)",
            "dancer_analysis.second_joint_coverage_min": "Second dancer joint coverage minimum",
            "dancer_analysis.second_sustained_min_s": "Second dancer sustained span (seconds)",
            "dancer_analysis.brief_second_max_s": "Brief second coexistence max (seconds)",
            "dancer_analysis.together_min_s": "Co-movement minimum span (seconds)",
            "dancer_analysis.together_max_hip_separation_torso_lengths": (
                "Co-movement max hip separation (torso lengths)"
            ),
            "dancer_analysis.together_direction_cosine_min": "Co-movement direction cosine min",
            "dancer_analysis.identity_max_innovation_torso_lengths": (
                "Identity max innovation (torso lengths)"
            ),
            "dancer_analysis.identity_margin_torso_lengths": "Identity assignment margin (torso lengths)",
            "dancer_analysis.max_observation_gap_s": "Max observation gap (seconds)",
            "dancer_analysis.display_interpolation_max_gap_s": "Display interpolation max gap (seconds)",
            "dancer_analysis.velocity_filter_window_s": "Velocity regression half-window (seconds)",
            "dancer_analysis.persistence_s": "State persistence (seconds)",
            "dancer_analysis.pairing_window_s": "Event pairing window (seconds)",
            "dancer_analysis.start_speed": "Travel start speed (torso lengths/s)",
            "dancer_analysis.stop_speed": "Travel stop speed (torso lengths/s)",
            "dancer_analysis.slowing_fraction": "Slowing drop fraction",
            "dancer_analysis.transition_min_coverage": "Transition minimum coverage",
            "dancer_analysis.secondary_speed_rise": "Secondary speed peak rise (pair torso lengths/s)",
            "dancer_analysis.axis_smoothing_s": "Couple axis chart smoothing (seconds)",
            "dancer_analysis.overlay_smoothing_s": "Couple overlay smoothing (seconds)",
            "dancer_analysis.axis_bridge_max_gap_s": "Couple axis display bridge max gap (seconds)",
            "dancer_analysis.overlay_min_landmark_visibility": "Couple overlay min landmark visibility",
            "dancer_analysis.duplicate_pose_max_distance_torso_lengths": (
                "Duplicate pose suppression threshold (torso lengths)"
            ),
            "dancer_analysis.fixed_scale_min_ratio": "Fixed scale min ratio",
            "dancer_analysis.fixed_scale_max_ratio": "Fixed scale max ratio",
            "dancer_analysis.shot_cut_max_frame_gap_s": "Shot-cut max frame gap (seconds)",
            "dancer_analysis.shot_cut_jump_torso_lengths": "Shot-cut hip jump threshold (torso lengths)",
            "dancer_analysis.turn_exit_match_window_s": "Turn exit match window (seconds)",
            "dancer_analysis.close_embrace_max_tilt_deg": "Close-embrace max torso tilt (degrees)",
            "dancer_analysis.analyze_soundtrack": "Analyze soundtrack beats by default",
            "dancer_analysis.auto_turn_suggestions": "Suggest turn markers automatically",
            "ollama_base_url": "Ollama server address",
            "ollama_num_ctx": "Model context size (tokens)",
            "ollama_num_predict": "Maximum response length (tokens)",
            "ollama_max_report_chars": "Maximum report length (characters)",
            "ollama_instruction": "Analysis focus",
            "ollama_temperature": "Response variation",
        }
        return labels.get(field, field.replace(".", " / ").replace("_", " ").capitalize())

    def _wrap_with_guide(
        self,
        key: str,
        control_widget: QWidget,
        *,
        fallback_summary: str = "",
    ) -> QWidget:
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 4)
        wrapper_layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        top_row.addWidget(control_widget, 1)

        toggle_btn = QPushButton("▸ Explanation && example")
        toggle_btn.setCheckable(True)
        toggle_btn.setStyleSheet("padding: 4px 10px; font-size: 12px;")
        top_row.addWidget(toggle_btn)
        wrapper_layout.addLayout(top_row)

        guide = self._parameter_guides.get(key, {})
        meta_text = guide.get("meta", "").strip()
        explanation_text = (guide.get("explanation") or fallback_summary).strip()
        example_text = guide.get("example", "").strip()

        panel = QFrame()
        panel.setProperty("class", "card")
        panel.setVisible(False)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 8, 12, 8)
        panel_layout.setSpacing(4)

        if meta_text:
            meta_lbl = QLabel(meta_text)
            meta_lbl.setWordWrap(True)
            meta_lbl.setStyleSheet(f"color: {COLOR_COPPER}; font-size: 12px; font-weight: 600;")
            panel_layout.addWidget(meta_lbl)
        if explanation_text:
            exp_lbl = QLabel(explanation_text)
            exp_lbl.setWordWrap(True)
            exp_lbl.setStyleSheet("font-size: 12px;")
            panel_layout.addWidget(exp_lbl)
        if example_text:
            ex_lbl = QLabel(f"Example: {example_text}")
            ex_lbl.setWordWrap(True)
            ex_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
            panel_layout.addWidget(ex_lbl)
        key_lbl = QLabel(f"Setting key: {key}")
        key_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        panel_layout.addWidget(key_lbl)

        wrapper_layout.addWidget(panel)

        def _on_toggle(checked: bool) -> None:
            panel.setVisible(checked)
            toggle_btn.setText("▾ Hide explanation" if checked else "▸ Explanation && example")
            self._sync_toggle_all_button()

        toggle_btn.toggled.connect(_on_toggle)
        self.help_toggles[key] = toggle_btn
        self.help_panels[key] = panel
        return wrapper

    def _toggle_all_guides(self, checked: bool) -> None:
        for btn in self.help_toggles.values():
            btn.setChecked(checked)
        self.toggle_all_guides_btn.setText("Collapse All Guides" if checked else "Expand All Guides")

    def _sync_toggle_all_button(self) -> None:
        if not self.help_toggles:
            return
        all_open = all(btn.isChecked() for btn in self.help_toggles.values())
        self.toggle_all_guides_btn.blockSignals(True)
        self.toggle_all_guides_btn.setChecked(all_open)
        self.toggle_all_guides_btn.setText("Collapse All Guides" if all_open else "Expand All Guides")
        self.toggle_all_guides_btn.blockSignals(False)

    def _browse_path(self, field: str, editor: QLineEdit) -> None:
        if field == "data_dir":
            path = QFileDialog.getExistingDirectory(self, "Choose data folder", editor.text())
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose pose model", editor.text(), "Model files (*.task);;All files (*)"
            )
        if path:
            editor.setText(path)

    def _set_legacy_widget_aliases(self) -> None:
        side = self.field_widgets["side"]
        facing = self.field_widgets["facing"]
        confidence = self.field_widgets["confidence"]
        smoothing = self.field_widgets["smoothing"]
        head_angle = self.field_widgets["head_down_angle"]
        shoulder = self.field_widgets["shoulder_raise_threshold"]
        bpm = self.field_widgets["tango_bpm"]
        ollama_url = self.field_widgets["ollama_base_url"]
        ollama_model = self.field_widgets["ollama_model"]
        couple_axis_smooth = self.field_widgets["dancer_analysis.axis_smoothing_s"]
        couple_overlay_smooth = self.field_widgets["dancer_analysis.overlay_smoothing_s"]
        assert isinstance(side, QComboBox)
        assert isinstance(facing, QComboBox)
        assert isinstance(confidence, QDoubleSpinBox)
        assert isinstance(smoothing, QDoubleSpinBox)
        assert isinstance(head_angle, QDoubleSpinBox)
        assert isinstance(shoulder, QDoubleSpinBox)
        assert isinstance(bpm, QDoubleSpinBox)
        assert isinstance(ollama_url, QLineEdit)
        assert isinstance(ollama_model, QLineEdit)
        assert isinstance(couple_axis_smooth, QDoubleSpinBox)
        assert isinstance(couple_overlay_smooth, QDoubleSpinBox)
        self.side_combo = side
        self.facing_combo = facing
        self.conf_spin = confidence
        self.smooth_spin = smoothing
        self.head_angle_spin = head_angle
        self.shoulder_spin = shoulder
        self.bpm_spin = bpm
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.couple_axis_smooth_spin = couple_axis_smooth
        self.couple_overlay_smooth_spin = couple_overlay_smooth

    def _collect_settings(self) -> Config:
        data = self.bridge.config.model_dump(exclude={"axis_curve"})
        for field, editor in self.field_widgets.items():
            if isinstance(editor, QCheckBox):
                value = str(editor.isChecked())
            elif isinstance(editor, QComboBox):
                value = editor.currentText()
            elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
                value = str(editor.value())
            elif isinstance(editor, QLineEdit):
                value = editor.text().strip()
            else:
                raise TypeError(f"Unsupported settings editor for {field}")
            set_setting(data, field, value)
        data["calibration_defaults"] = {
            key: RangeDefault.model_validate({
                "lower": lower.value(), "upper": upper.value(), "unit": METRIC_DETAILS[key][2],
                "view": view.currentText(),
            }).model_dump()
            for key, (enabled, lower, upper, view) in self.range_widgets.items()
            if enabled.isChecked()
        }
        if hasattr(self, "normalization_panel"):
            data["measurement_normalization"] = (
                self.normalization_panel.measurement_normalization_config().model_dump()
            )
        return Config.model_validate(data)

    def _save_settings(self) -> bool:
        try:
            cfg = self._collect_settings()
        except (TypeError, ValueError, ValidationError) as error:
            self.bridge.status_message.emit(f"Could not apply settings: {error}")
            return False
        destination = Path("config.toml").resolve()
        try:
            self.bridge.save_settings(cfg, destination)
        except OSError as error:
            self.bridge.update_config(cfg)
            self._apply_config(cfg)
            self.bridge.status_message.emit(
                f"Settings applied in memory (could not write {destination.name}: {error})."
            )
            return True
        self._apply_config(cfg)
        self.bridge.status_message.emit(f"Settings saved to {destination}")
        return True

    def _reset_defaults(self) -> None:
        default_cfg = Config()
        self.bridge.update_config(default_cfg)
        self._apply_config(default_cfg)

    def _apply_config(self, cfg: Config) -> None:
        for field, editor in self.field_widgets.items():
            value = setting_value(cfg, field)
            if isinstance(editor, QCheckBox):
                editor.setChecked(value)
            elif isinstance(editor, QComboBox):
                editor.setCurrentText(str(value))
            elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
                editor.setValue(value)
            elif isinstance(editor, QLineEdit):
                editor.setText(str(value))
        for key, (enabled, lower, upper, view) in self.range_widgets.items():
            target = cfg.calibration_defaults.get(key)
            enabled.setChecked(target is not None)
            lower.setValue(target.lower if target else 0)
            upper.setValue(target.upper if target else 0)
            view.setCurrentText(target.view if target else "side")
            for widget in (lower, upper, view):
                widget.setEnabled(target is not None)
        if hasattr(self, "normalization_panel"):
            self.normalization_panel.set_config(cfg)

    def refresh(self) -> None:
        if hasattr(self, "normalization_panel"):
            self.normalization_panel.refresh_library_choices()

    def set_engine_scope(self, scope: str) -> None:
        resolved = scope if scope in ("All", *SETTING_ENGINE_SCOPES) else "All"
        self._current_engine_scope = resolved
        for name, btn in self.scope_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(name == resolved)
            btn.blockSignals(False)
        self.scope_summary_label.setText(_SCOPE_SUMMARIES[resolved])
        allowed = None if resolved == "All" else set(SETTING_ENGINE_SCOPES[resolved])
        first_visible: int | None = None
        for index in range(self.tabs.count()):
            visible = allowed is None or self.tabs.tabText(index) in allowed
            self.tabs.setTabVisible(index, visible)
            if visible and first_visible is None:
                first_visible = index
        cur_idx = self.tabs.currentIndex()
        if (
            cur_idx < 0
            or not self.tabs.isTabVisible(cur_idx)
        ) and first_visible is not None:
            self.tabs.setCurrentIndex(first_visible)

    def open_measurement_normalization(self, payload: dict[str, Any]) -> None:
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == "Measurement normalization":
                if not self.tabs.isTabVisible(index):
                    self.set_engine_scope("Single Dancer Engine")
                self.tabs.setCurrentIndex(index)
                break
        if hasattr(self, "normalization_panel"):
            self.normalization_panel.select_context(payload)

    def _save_toml(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        destination = Path("settings.toml").resolve()
        if QMessageBox.question(
            self, "Save Settings", f"Apply these Settings and overwrite {destination}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            cfg = self._collect_settings()
            self.bridge.save_settings(cfg, destination)
        except (OSError, TypeError, ValueError, ValidationError) as error:
            self.bridge.status_message.emit(f"Could not save settings.toml: {error}")
            return
        self.bridge.status_message.emit(f"Settings saved to {destination}")

    def _export_toml(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export TOML Configuration", "settings.toml", "TOML (*.toml)"
        )
        if not path:
            return
        if not self._save_settings():
            return
        try:
            self.bridge.export_settings_toml(Path(path))
        except OSError as error:
            self.bridge.status_message.emit(f"Could not export settings: {error}")

    def _import_toml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import TOML Configuration", "", "TOML (*.toml)")
        if path:
            try:
                self.bridge.import_settings_toml(Path(path))
            except (
                OSError,
                UnicodeError,
                tomllib.TOMLDecodeError,
                ValidationError,
                ValueError,
                TypeError,
            ) as error:
                self.bridge.status_message.emit(f"Could not import settings: {error}")
                return
            self._apply_config(self.bridge.config)
