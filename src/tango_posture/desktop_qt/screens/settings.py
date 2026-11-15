"""Settings management screen with grouped configuration and TOML import/export."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

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
    SETTING_GROUPS,
    Config,
    RangeDefault,
    set_setting,
    setting_value,
)
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
    "tango_bpm": (20.0, 240.0),
}

_INT_RANGES = {
    "calibration_min_usable_samples": (2, 1_000_000),
    "shoulder_adaptive_max_samples": (1, 1_000_000),
    "shoulder_adaptive_min_samples": (1, 1_000_000),
    "max_consecutive_high_ankle_steps": (1, 1_000),
}


def _guide_markdown_path() -> Path:
    return Path(__file__).resolve().parents[4] / "docs" / "SETTINGS_BEGINNERS_GUIDE.md"


def load_parameter_guides() -> dict[str, dict[str, str]]:
    """Parse per-parameter explanations and worked examples from SETTINGS_BEGINNERS_GUIDE.md."""
    guides: dict[str, dict[str, str]] = {}
    guide_path = _guide_markdown_path()
    if guide_path.is_file():
        text = guide_path.read_text(encoding="utf-8")
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

_GROUP_HINTS = {
    "Quality & calibration": (
        "Quality controls determine whether evidence is usable, not whether posture is correct. "
        "Approve reference ranges using example passages, acceptable limits and per-metric views "
        "in the calibration editor."
    ),
    "Head & shoulders": (
        "Fallback descriptive warning controls. A selected calibration or enabled Calibration "
        "defaults supplies approved ranges instead; missing metrics do not fall back here. "
        "Review alerts require a selected calibration file."
    ),
    "Embrace & knees": (
        "Role/embrace recognition runs independently of calibration. Knee-warning thresholds "
        "are fallbacks, replaced by selected calibration ranges or enabled Calibration defaults."
    ),
    "Torso & hip offset": (
        "Fallback hip-offset controls, not pink torso-line tilt. Selected calibration ranges "
        "or enabled Calibration defaults replace these posture limits."
    ),
    "Music": (
        "This fallback tempo is the input, not the Slow tango max BPM cutoff. Library/Review "
        "analysis does not detect soundtrack BPM or show fallback lift alerts."
    ),
    "Slow tango": (
        "The BPM ceiling gates the ankle-lift cue, not video playback or footprint detection. "
        "It is compared with the fallback tempo in Music when no soundtrack BPM is available."
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
            "Configure video analysis, algorithmic geometry parameters, reference comparison, review, and Ollama options."
        )
        subtitle.setProperty("class", "subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.toggle_all_guides_btn = QPushButton("Expand All Guides")
        self.toggle_all_guides_btn.setCheckable(True)
        self.toggle_all_guides_btn.clicked.connect(self._toggle_all_guides)
        self.export_btn = QPushButton("Export TOML…")
        self.export_btn.clicked.connect(self._export_toml)
        self.import_btn = QPushButton("Import TOML…")
        self.import_btn.clicked.connect(self._import_toml)
        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)
        header.addWidget(self.toggle_all_guides_btn)
        header.addWidget(self.export_btn)
        header.addWidget(self.import_btn)
        header.addWidget(self.reset_btn)
        layout.addLayout(header)

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
            if group == "Files":
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

        self.save_btn = QPushButton("Apply && Save settings.toml")
        self.save_btn.setProperty("class", "primary")
        self.save_btn.setFixedHeight(38)
        self.save_btn.clicked.connect(self._save_toml)
        layout.addWidget(self.save_btn)
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
        elif field in {"side", "facing", "replay_joint_interpolation", "analysis_model"}:
            combo = QComboBox()
            combo.addItems(
                ["lite", "full", "heavy"] if field == "analysis_model" else
                ["linear", "quadratic"] if field == "replay_joint_interpolation" else ["left", "right"]
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
                else 0.05
                if _FLOAT_RANGES.get(field) == (0.0, 1.0)
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
        assert isinstance(side, QComboBox)
        assert isinstance(facing, QComboBox)
        assert isinstance(confidence, QDoubleSpinBox)
        assert isinstance(smoothing, QDoubleSpinBox)
        assert isinstance(head_angle, QDoubleSpinBox)
        assert isinstance(shoulder, QDoubleSpinBox)
        assert isinstance(bpm, QDoubleSpinBox)
        assert isinstance(ollama_url, QLineEdit)
        assert isinstance(ollama_model, QLineEdit)
        self.side_combo = side
        self.facing_combo = facing
        self.conf_spin = confidence
        self.smooth_spin = smoothing
        self.head_angle_spin = head_angle
        self.shoulder_spin = shoulder
        self.bpm_spin = bpm
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

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
        return Config.model_validate(data)

    def _save_settings(self) -> bool:
        try:
            cfg = self._collect_settings()
        except (TypeError, ValueError, ValidationError) as error:
            self.bridge.status_message.emit(f"Could not apply settings: {error}")
            return False
        self.bridge.update_config(cfg)
        self._apply_config(cfg)
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
            except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValidationError) as error:
                self.bridge.status_message.emit(f"Could not import settings: {error}")
                return
            self._apply_config(self.bridge.config)
