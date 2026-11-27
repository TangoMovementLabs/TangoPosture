"""Shared dancer analysis options panel and Auto manual-selection dialog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...config import Config
from ...couple.models import CoupleAnalysisRequest, CoupleAnnotations, DancerAnchor
from ...replay import VideoReader


class AnalysisOptionsWidget(QWidget):
    """Reusable widget for Auto / Single / Couple options shared across Analysis, Library & Review."""

    options_changed = Signal()

    def __init__(
        self,
        cfg: Config,
        *,
        show_recover_gaps: bool = True,
        compact: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6 if compact else 10)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6 if compact else 8)

        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("analysisDancerModeCombo")
        self.mode_combo.addItem("Auto (recommended)", "auto")
        self.mode_combo.addItem("Single dancer", "single")
        self.mode_combo.addItem("Couple (two dancers)", "couple")
        form.addRow("Dancers:", self.mode_combo)

        self.mode_hint_label = QLabel()
        self.mode_hint_label.setProperty("class", "muted")
        self.mode_hint_label.setWordWrap(True)
        form.addRow(self.mode_hint_label)

        self.use_region_cb = QCheckBox("Limit analysis region (ignore background dancers)")
        self.use_region_cb.setObjectName("useRegionCheckBox")
        form.addRow(self.use_region_cb)

        region_row = QWidget()
        region_layout = QHBoxLayout(region_row)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.setSpacing(6)
        self.x_min_spin = self._coord_spin(0.0)
        self.y_min_spin = self._coord_spin(0.0)
        self.x_max_spin = self._coord_spin(1.0)
        self.y_max_spin = self._coord_spin(1.0)
        for label_text, spin in (
            ("X min:", self.x_min_spin),
            ("Y min:", self.y_min_spin),
            ("X max:", self.x_max_spin),
            ("Y max:", self.y_max_spin),
        ):
            region_layout.addWidget(QLabel(label_text))
            region_layout.addWidget(spin)
        self._region_row = region_row
        form.addRow("Region (0–1):", region_row)

        self.soundtrack_cb = QCheckBox("Analyze video soundtrack for step-to-beat timing")
        self.soundtrack_cb.setObjectName("analyzeSoundtrackCheckBox")
        form.addRow(self.soundtrack_cb)

        self.auto_turns_cb = QCheckBox("Suggest candidate turns & exits automatically")
        self.auto_turns_cb.setObjectName("autoTurnSuggestionsCheckBox")
        form.addRow(self.auto_turns_cb)

        self.recover_gaps_cb = QCheckBox("Recover short tracking gaps")
        self.recover_gaps_cb.setObjectName("recoverGapsCheckBox")
        self.recover_gaps_cb.setToolTip(
            "Run display-only backward pose recovery on short lower-limb dropouts in Single mode."
        )
        form.addRow(self.recover_gaps_cb)
        if not show_recover_gaps:
            self.recover_gaps_cb.hide()

        layout.addLayout(form)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_or_option_changed)
        self.use_region_cb.toggled.connect(self._on_region_toggled)
        for spin in (self.x_min_spin, self.y_min_spin, self.x_max_spin, self.y_max_spin):
            spin.valueChanged.connect(lambda _: self.options_changed.emit())
        self.soundtrack_cb.toggled.connect(lambda _: self.options_changed.emit())
        self.auto_turns_cb.toggled.connect(lambda _: self.options_changed.emit())
        self.recover_gaps_cb.toggled.connect(lambda _: self.options_changed.emit())

        self.set_config(cfg)

    def _coord_spin(self, initial: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(initial)
        return spin

    def set_config(self, cfg: Config) -> None:
        self._cfg = cfg
        self.set_dancer_mode(cfg.dancer_analysis.default_dancer_mode)
        self.auto_turns_cb.setChecked(bool(cfg.dancer_analysis.auto_turn_suggestions))
        self.recover_gaps_cb.setChecked(bool(cfg.replay_recovery_auto))
        self._on_region_toggled(self.use_region_cb.isChecked())
        self._update_hint()

    def _on_region_toggled(self, checked: bool) -> None:
        self._region_row.setEnabled(bool(checked))
        self.options_changed.emit()

    def _on_mode_or_option_changed(self) -> None:
        self._update_hint()
        self.options_changed.emit()

    def _update_hint(self) -> None:
        mode = self.dancer_mode()
        win_s = float(self._cfg.dancer_analysis.auto_detection_window_s)
        if mode == "auto":
            self.mode_hint_label.setText(
                f"Auto classifies the opening {win_s:.0f} s window "
                "(precedence: Couple > Single > Needs selection)."
            )
        elif mode == "couple":
            self.mode_hint_label.setText(
                "Runs paired Dancer A & B tracking, lead/follow coordination, "
                "transitions, posture tilts, and turns/exits."
            )
        else:
            self.mode_hint_label.setText(
                "Runs solo posture and movement analysis on one dancer."
            )

    def dancer_mode(self) -> str:
        return str(self.mode_combo.currentData() or "auto")

    def set_dancer_mode(self, mode: str) -> None:
        idx = self.mode_combo.findData(str(mode).lower())
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self._update_hint()

    def analysis_region(self) -> dict[str, float] | None:
        if not self.use_region_cb.isChecked():
            return None
        x_min = float(self.x_min_spin.value())
        y_min = float(self.y_min_spin.value())
        x_max = float(self.x_max_spin.value())
        y_max = float(self.y_max_spin.value())
        if x_max <= x_min or y_max <= y_min:
            return None
        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

    def set_analysis_region(self, region: dict[str, float] | None) -> None:
        if not region:
            self.use_region_cb.setChecked(False)
            return
        self.x_min_spin.setValue(float(region.get("x_min", 0.0)))
        self.y_min_spin.setValue(float(region.get("y_min", 0.0)))
        self.x_max_spin.setValue(float(region.get("x_max", 1.0)))
        self.y_max_spin.setValue(float(region.get("y_max", 1.0)))
        self.use_region_cb.setChecked(True)

    def build_request(
        self,
        *,
        forced_override: bool = False,
        annotations: CoupleAnnotations | dict[str, Any] | None = None,
    ) -> CoupleAnalysisRequest:
        ann = (
            annotations
            if isinstance(annotations, CoupleAnnotations)
            else CoupleAnnotations.from_dict(annotations)
        )
        raw_mode = self.dancer_mode()
        mode = raw_mode if raw_mode in ("auto", "single", "couple") else "auto"
        return CoupleAnalysisRequest(
            requested_mode=mode,  # type: ignore[arg-type]
            forced_override=bool(forced_override),
            analysis_region=self.analysis_region(),
            analyze_soundtrack=bool(self.soundtrack_cb.isChecked()),
            auto_turn_suggestions=bool(self.auto_turns_cb.isChecked()),
            annotations=ann,
        )


def _draw_candidate_thumbnail(
    frame_bgr: np.ndarray | None,
    candidate: dict[str, Any],
    pin_label: str | None = None,
) -> QPixmap:
    canvas: np.ndarray
    if frame_bgr is None or frame_bgr.size == 0:
        canvas = np.full((180, 260, 3), 26, dtype=np.uint8)
        w, h = 260, 180
    else:
        fh, fw = frame_bgr.shape[:2]
        scale = min(260.0 / max(1, fw), 180.0 / max(1, fh))
        w = max(1, round(fw * scale))
        h = max(1, round(fh * scale))
        canvas = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)

    bbox = candidate.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1 = max(0, min(w - 1, round(float(bbox[0]) * w)))
        y1 = max(0, min(h - 1, round(float(bbox[1]) * h)))
        x2 = max(x1 + 2, min(w - 1, round(float(bbox[2]) * w)))
        y2 = max(y1 + 2, min(h - 1, round(float(bbox[3]) * h)))
        color = (255, 160, 60) if pin_label == "A" else (90, 110, 255) if pin_label == "B" else (200, 190, 175)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    hip = candidate.get("hip_xy")
    if isinstance(hip, (list, tuple)) and len(hip) == 2:
        hx = max(0, min(w - 1, round(float(hip[0]) * w)))
        hy = max(0, min(h - 1, round(float(hip[1]) * h)))
        cv2.circle(canvas, (hx, hy), 5, (72, 224, 240), -1, cv2.LINE_AA)

    qimg = QImage(canvas.data, w, h, w * 3, QImage.Format.Format_BGR888).copy()
    return QPixmap.fromImage(qimg)


class NeedsSelectionDialog(QDialog):
    """Prompt user when Auto cannot confidently classify Single vs Couple in the opening window."""

    def __init__(
        self,
        decision: dict[str, Any],
        *,
        video_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Dancer Mode — Needs Selection")
        self.setMinimumWidth(680)
        self.decision = dict(decision)
        self.selected_mode: str | None = None
        self._pinned_anchors: dict[str, dict[str, Any]] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Auto detection needs your choice before continuing")
        title.setProperty("class", "section")
        layout.addWidget(title)

        explanation = QLabel(
            str(
                decision.get("explanation")
                or "The opening detection window could not confidently distinguish Single from Couple."
            )
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        stats_text = (
            f"Opening window: {float(decision.get('effective_window_in', 0.0)):.1f} s – "
            f"{float(decision.get('effective_window_out', 0.0)):.1f} s "
            f"({float(decision.get('effective_duration_s', 0.0)):.1f} s) · "
            f"Primary coverage: {float(decision.get('primary_coverage', 0.0)) * 100:.0f}% · "
            f"Joint coverage: {float(decision.get('second_joint_coverage', 0.0)) * 100:.0f}% · "
            f"Max paired span: {float(decision.get('max_continuous_paired_span_s', 0.0)):.2f} s · "
            f"Co-movement span: {float(decision.get('max_co_movement_span_s', 0.0)):.2f} s"
        )
        stats_lbl = QLabel(stats_text)
        stats_lbl.setProperty("class", "muted")
        stats_lbl.setWordWrap(True)
        layout.addWidget(stats_lbl)

        previews = list(decision.get("candidate_previews") or [])[:3]
        self._preview_frames: dict[int, np.ndarray | None] = {}
        if previews and video_path is not None and video_path.is_file():
            try:
                reader = VideoReader(video_path)
                try:
                    for cand in previews:
                        f_idx = int(cand.get("frame_index", 0))
                        if 0 <= f_idx < reader.count:
                            try:
                                frame = reader.frame(f_idx)
                            except ValueError:
                                frame = None
                            self._preview_frames[f_idx] = frame
                finally:
                    reader.close()
            except (OSError, ValueError, RuntimeError):
                pass

        self._card_labels: list[tuple[dict[str, Any], QLabel, QLabel]] = []
        if previews:
            thumbs_title = QLabel(
                "Detected subjects in opening window (optionally pin Dancer A & Dancer B anchors):"
            )
            layout.addWidget(thumbs_title)
            row = QHBoxLayout()
            row.setSpacing(10)
            for cand in previews:
                card = QFrame()
                card.setProperty("class", "card")
                clayout = QVBoxLayout(card)
                clayout.setContentsMargins(8, 8, 8, 8)
                clayout.setSpacing(6)

                f_idx = int(cand.get("frame_index", 0))
                img_lbl = QLabel()
                img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                img_lbl.setPixmap(_draw_candidate_thumbnail(self._preview_frames.get(f_idx), cand))
                clayout.addWidget(img_lbl)

                meta_lbl = QLabel(
                    f"Subject @{float(cand.get('timestamp', 0.0)):.2f} s · "
                    f"cov {float(cand.get('coverage', 0.0)) * 100:.0f}%"
                )
                meta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                clayout.addWidget(meta_lbl)

                btn_row = QHBoxLayout()
                pin_a = QPushButton("Pin Dancer A")
                pin_b = QPushButton("Pin Dancer B")
                pin_a.clicked.connect(lambda _, c=cand: self._pin_candidate("A", c))
                pin_b.clicked.connect(lambda _, c=cand: self._pin_candidate("B", c))
                btn_row.addWidget(pin_a)
                btn_row.addWidget(pin_b)
                clayout.addLayout(btn_row)

                self._card_labels.append((cand, img_lbl, meta_lbl))
                row.addWidget(card)
            layout.addLayout(row)

        self._anchor_status = QLabel("No manual anchors pinned (tracker will initialize automatically).")
        self._anchor_status.setProperty("class", "muted")
        layout.addWidget(self._anchor_status)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.single_btn = QPushButton("Analyze as Single")
        self.couple_btn = QPushButton("Analyze as Couple")
        self.couple_btn.setProperty("class", "primary")
        self.cancel_btn.clicked.connect(self.reject)
        self.single_btn.clicked.connect(lambda: self._accept_mode("single"))
        self.couple_btn.clicked.connect(lambda: self._accept_mode("couple"))
        btn_box.addWidget(self.cancel_btn)
        btn_box.addWidget(self.single_btn)
        btn_box.addWidget(self.couple_btn)
        layout.addLayout(btn_box)

    def _pin_candidate(self, dancer_id: str, candidate: dict[str, Any]) -> None:
        self._pinned_anchors[dancer_id] = {
            "dancer_id": dancer_id,
            "timestamp": float(candidate.get("timestamp", 0.0)),
            "frame_index": candidate.get("frame_index"),
            "hip_xy": candidate.get("hip_xy"),
            "pose_index": candidate.get("pose_index"),
        }
        for cand, img_lbl, _ in self._card_labels:
            pin = next(
                (did for did, a in self._pinned_anchors.items() if a.get("pose_index") == cand.get("pose_index") and a.get("frame_index") == cand.get("frame_index")),
                None,
            )
            f_idx = int(cand.get("frame_index", 0))
            img_lbl.setPixmap(
                _draw_candidate_thumbnail(self._preview_frames.get(f_idx), cand, pin_label=pin)
            )
        pinned = ", ".join(
            f"{did} @ {a['timestamp']:.2f}s" for did, a in sorted(self._pinned_anchors.items())
        )
        self._anchor_status.setText(f"Pinned anchors: {pinned}")

    def _accept_mode(self, mode: str) -> None:
        self.selected_mode = mode
        self.accept()

    @property
    def selected_anchors(self) -> list[dict[str, Any]]:
        return [dict(v) for v in self._pinned_anchors.values()]

    def apply_to_request(self, base: CoupleAnalysisRequest | None = None) -> CoupleAnalysisRequest | None:
        if self.selected_mode not in ("single", "couple"):
            return None
        req = CoupleAnalysisRequest.from_dict(base.to_dict() if base is not None else None)
        anchors = [DancerAnchor.from_dict(a) for a in self.selected_anchors]
        if anchors:
            req.annotations.anchors = anchors
        req.requested_mode = self.selected_mode  # type: ignore[assignment]
        req.forced_override = True
        return req
