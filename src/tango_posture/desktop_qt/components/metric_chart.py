"""Reusable chart of timestamped evidence without gap interpolation."""

from __future__ import annotations

import csv
import math
from itertools import pairwise
from pathlib import Path
from xml.sax.saxutils import escape

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QToolTip,
    QWidget,
)

from ..analytics.data import (
    MetricData,
)
from ..theme import COLOR_BASE, COLOR_BORDER, COLOR_TEXT_MUTED, COLOR_TEXT_PRIMARY

STATE_COLORS = {
    "WITHIN_REFERENCE": "#50DC50",
    "PENDING": "#FFBE00",
    "DEVIATION": "#FF0000",
    "UNKNOWN": "#B4B4B4",
}

IMAGE_EXPORT_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class MetricChart(QWidget):
    """Draw stored samples without interpolating across missing coverage."""

    seek_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.data: MetricData | None = None
        self.time_end = 0.0
        self.origin_timestamp = 0.0
        self.bounds: tuple[float, float] | None = None
        self.reference_line: float | None = None
        self.intervals: list[tuple[float, float]] = []
        self.playhead_timestamp: float | None = None
        self.states: dict[float, str] = {}
        self.unit: str = ""
        self._hover_point: tuple[float, float] | None = None
        self.hover_hint_text: str = ""
        self.x_zoom = 1.0
        self.x_pan = 0.0
        self.setMinimumHeight(155)
        self.setEnabled(False)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Persisted smoothed delta chart")

    def set_data(
        self,
        data: MetricData | None,
        time_end: float = 0.0,
        origin_timestamp: float = 0.0,
        *,
        bounds: tuple[float, float] | None = None,
        reference_line: float | None = None,
        intervals: list[tuple[float, float]] | None = None,
        states: dict[float, str] | None = None,
        unit: str = "",
    ) -> None:
        self.data, self.time_end, self.origin_timestamp = data, time_end, origin_timestamp
        self.bounds = bounds
        self.reference_line = (
            float(reference_line)
            if reference_line is not None and math.isfinite(reference_line)
            else None
        )
        self.intervals = intervals or []
        self.states = states or {}
        self.unit = unit.strip()
        self._hover_point = None
        self.hover_hint_text = ""
        self.setToolTip("")
        if data is None or not data.segments or self.time_end <= 0:
            self.x_zoom = 1.0
            self.x_pan = 0.0
        elif self.x_zoom > 1.001:
            span = self.time_end / self.x_zoom
            self.x_pan = max(0.0, min(max(0.0, self.time_end - span), self.x_pan))
        else:
            self.x_pan = 0.0
        self.setEnabled(data is not None and bool(data.segments))
        self.update()

    def _view_window(self) -> tuple[float, float]:
        total = self.time_end or 1.0
        if self.x_zoom <= 1.001 or total <= 0.0:
            return 0.0, total
        span = total / self.x_zoom
        start = max(0.0, min(max(0.0, total - span), self.x_pan))
        return start, min(total, start + span)

    def set_x_zoom(self, zoom: float) -> None:
        new_zoom = max(1.0, float(zoom))
        total = self.time_end or 1.0
        if new_zoom <= 1.001 or total <= 0.0:
            self.x_zoom = 1.0
            self.x_pan = 0.0
        else:
            span = total / new_zoom
            if self.playhead_timestamp is not None:
                center = max(0.0, min(total, self.playhead_timestamp - self.origin_timestamp))
            else:
                old_start, old_end = self._view_window()
                center = 0.5 * (old_start + old_end)
            self.x_zoom = new_zoom
            self.x_pan = max(0.0, min(max(0.0, total - span), center - 0.5 * span))
        self.update()

    def set_playhead(self, timestamp: float | None) -> None:
        self.playhead_timestamp = timestamp
        if timestamp is not None and self.x_zoom > 1.001 and self.time_end > 0.0:
            span = self.time_end / self.x_zoom
            elapsed = max(0.0, min(self.time_end, timestamp - self.origin_timestamp))
            if elapsed < self.x_pan + 0.1 * span or elapsed > self.x_pan + 0.9 * span:
                self.x_pan = max(0.0, min(max(0.0, self.time_end - span), elapsed - 0.5 * span))
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        left, right = 62.0, float(max(63, self.width() - 20))
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.isEnabled()
            and self.time_end > 0
            and left <= event.position().x() <= right
        ):
            view_start, view_end = self._view_window()
            fraction = (event.position().x() - left) / (right - left)
            self.seek_requested.emit(
                self.origin_timestamp + view_start + fraction * (view_end - view_start)
            )
        super().mousePressEvent(event)

    def _nearest_sample_for_pos(self, pos: QPointF) -> tuple[float, float] | None:
        if self.data is None or not self.data.segments or self.time_end <= 0.0 or not self.isEnabled():
            return None
        left = 62.0
        right = float(max(63, self.width() - 20))
        top = 12.0
        bottom = float(max(13, self.height() - 38))
        if not (left <= pos.x() <= right and top <= pos.y() <= bottom):
            return None
        view_start, view_end = self._view_window()
        span = max(view_end - view_start, 1e-9)
        target_t = view_start + ((pos.x() - left) / max(1.0, right - left)) * span
        best: tuple[float, float] | None = None
        best_dt = math.inf
        for segment in self.data.segments:
            for ts, val in segment:
                if not (view_start <= ts <= view_end):
                    continue
                dt = abs(ts - target_t)
                if dt < best_dt:
                    best_dt = dt
                    best = (ts, val)
        return best

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        nearest = self._nearest_sample_for_pos(event.position())
        if nearest is None:
            if self._hover_point is not None or self.hover_hint_text:
                self._hover_point = None
                self.hover_hint_text = ""
                self.setToolTip("")
                QToolTip.hideText()
                self.update()
        else:
            ts_rel, val = nearest
            abs_t = self.origin_timestamp + ts_rel
            unit_suffix = f" {self.unit}" if self.unit else ""
            hint = f"t = {abs_t:.2f} s · {val:.3g}{unit_suffix}"
            changed = self._hover_point != nearest or self.hover_hint_text != hint
            self._hover_point = nearest
            self.hover_hint_text = hint
            self.setToolTip(hint)
            QToolTip.showText(event.globalPosition().toPoint(), hint, self)
            if changed:
                self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if self._hover_point is not None or self.hover_hint_text:
            self._hover_point = None
            self.hover_hint_text = ""
            self.setToolTip("")
            QToolTip.hideText()
            self.update()
        super().leaveEvent(event)

    def _y_range(self) -> tuple[float, float] | None:
        if self.data is None or not self.data.segments:
            return None
        segment_values = [value for segment in self.data.segments for _, value in segment]
        if len(segment_values) >= 30:
            sorted_vals = sorted(segment_values)
            count = len(sorted_vals)

            def _pct(frac: float) -> float:
                pos = frac * (count - 1)
                lo_i = math.floor(pos)
                hi_i = math.ceil(pos)
                if lo_i == hi_i:
                    return sorted_vals[lo_i]
                weight = pos - lo_i
                return sorted_vals[lo_i] * (1.0 - weight) + sorted_vals[hi_i] * weight

            p2, p10, p90, p98 = _pct(0.02), _pct(0.10), _pct(0.90), _pct(0.98)
            spread = p90 - p10
            if spread > 1e-6 and (sorted_vals[-1] - sorted_vals[0]) > 3.0 * spread:
                clip_lo = min(p2, p10 - 1.5 * spread)
                clip_hi = max(p98, p90 + 1.5 * spread)
                values = [min(clip_hi, max(clip_lo, val)) for val in segment_values]
            else:
                values = list(segment_values)
        else:
            values = list(segment_values)
        if self.bounds is not None:
            values.extend(value for value in self.bounds if math.isfinite(value))
        if self.reference_line is not None and math.isfinite(self.reference_line):
            values.append(self.reference_line)
        if not values:
            return None
        low, high = min(values), max(values)
        padding = max((high - low) * 0.1, 0.001)
        return low - padding, high + padding

    def _point_state(self, timestamp: float, value: float) -> str:
        state = self.states.get(timestamp, "")
        if self.bounds is not None and state in ("DEVIATION", "PENDING"):
            low_b, high_b = min(self.bounds[0], self.bounds[1]), max(self.bounds[0], self.bounds[1])
            if low_b <= value <= high_b:
                return "WITHIN_REFERENCE"
        return state

    def _segment_pieces(
        self,
        t0: float,
        v0: float,
        t1: float,
        v1: float,
    ) -> list[tuple[float, float, float, float, str]]:
        """Split a segment at calibration bounds so in-range portions draw green."""
        if self.bounds is None or not self.states:
            color = STATE_COLORS.get(self._point_state(t1, v1), "#69B9CF")
            return [(t0, v0, t1, v1, color)]
        low_b, high_b = min(self.bounds[0], self.bounds[1]), max(self.bounds[0], self.bounds[1])
        dv = v1 - v0
        cuts: list[float] = [0.0, 1.0]
        if abs(dv) > 1e-12:
            for bound in (low_b, high_b):
                u = (bound - v0) / dv
                if 1e-9 < u < 1.0 - 1e-9:
                    cuts.append(u)
        cuts = sorted(set(cuts))
        raw_s0 = self.states.get(t0, "")
        raw_s1 = self.states.get(t1, "")
        if "DEVIATION" in (raw_s0, raw_s1):
            out_state = "DEVIATION"
        elif "PENDING" in (raw_s0, raw_s1):
            out_state = "PENDING"
        else:
            out_state = self._point_state(t1, v1) or "WITHIN_REFERENCE"
        out_color = STATE_COLORS.get(out_state, "#69B9CF")
        in_color = STATE_COLORS["WITHIN_REFERENCE"]
        pieces: list[tuple[float, float, float, float, str]] = []
        for u0, u1 in pairwise(cuts):
            if u1 - u0 <= 1e-9:
                continue
            ta = t0 + u0 * (t1 - t0)
            va = v0 + u0 * dv
            tb = t0 + u1 * (t1 - t0)
            vb = v0 + u1 * dv
            vm = 0.5 * (va + vb)
            color = in_color if (low_b - 1e-9) <= vm <= (high_b + 1e-9) else out_color
            pieces.append((ta, va, tb, vb, color))
        return pieces or [
            (t0, v0, t1, v1, in_color if (low_b - 1e-9) <= v1 <= (high_b + 1e-9) else out_color)
        ]

    def _paint_chart(
        self,
        painter: QPainter,
        canvas_rect: QRectF,
        *,
        include_playhead: bool = True,
    ) -> None:
        painter.fillRect(canvas_rect, QColor(COLOR_BASE))
        painter.setPen(QColor(COLOR_TEXT_MUTED))
        y_limits = self._y_range()
        if self.data is None or not self.data.segments or y_limits is None:
            painter.drawText(
                canvas_rect, Qt.AlignmentFlag.AlignCenter, "No valid persisted deltas — unknown"
            )
            return
        low, high = y_limits
        plot = QRectF(
            canvas_rect.left() + 62,
            canvas_rect.top() + 12,
            max(1.0, canvas_rect.width() - 82),
            max(1.0, canvas_rect.height() - 50),
        )
        total_duration = self.time_end or 1.0
        view_start, view_end = (
            self._view_window() if include_playhead else (0.0, total_duration)
        )
        span = max(view_end - view_start, 1e-9)

        def point(timestamp: float, value: float) -> QPointF:
            return QPointF(
                plot.left() + (timestamp - view_start) / span * plot.width(),
                plot.bottom() - (value - low) / (high - low) * plot.height(),
            )

        painter.save()
        painter.setClipRect(plot)
        for start, end in self.intervals:
            clipped_start = max(view_start, start)
            clipped_end = min(view_end, end)
            left = plot.left() + (clipped_start - view_start) / span * plot.width()
            right = plot.left() + (clipped_end - view_start) / span * plot.width()
            if right > left:
                painter.fillRect(
                    QRectF(left, plot.top(), right - left, plot.height()),
                    QColor(49, 87, 72, 90),
                )
        if self.bounds is not None:
            y_low = point(view_start, self.bounds[0]).y()
            y_high = point(view_start, self.bounds[1]).y()
            band_top = min(y_low, y_high)
            band_height = max(1.0, abs(y_high - y_low))
            painter.fillRect(
                QRectF(plot.left(), band_top, plot.width(), band_height),
                QColor(80, 220, 80, 28),
            )
            painter.setPen(QPen(QColor("#E3B66B"), 1.5, Qt.PenStyle.DashLine))
            for label, value in zip(("Min", "Max"), self.bounds, strict=True):
                y = point(view_start, value).y()
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                text_y = max(plot.top() + 12.0, min(plot.bottom() - 3.0, y - 3.0))
                painter.drawText(
                    QPointF(max(plot.left(), plot.right() - 95.0), text_y),
                    f"{label} {value:.3g}",
                )
        if self.reference_line is not None:
            y_ref = point(view_start, self.reference_line).y()
            painter.setPen(QPen(QColor("#A3E635"), 1.5, Qt.PenStyle.DashDotLine))
            painter.drawLine(QPointF(plot.left(), y_ref), QPointF(plot.right(), y_ref))
            ref_y = max(plot.top() + 12.0, min(plot.bottom() - 3.0, y_ref - 4.0))
            painter.drawText(
                QPointF(plot.left() + 6.0, ref_y),
                f"Ref {self.reference_line:.3g}",
            )
        painter.restore()
        painter.setPen(QColor(COLOR_BORDER))
        painter.drawRect(plot)
        painter.setPen(QColor(COLOR_TEXT_MUTED))
        painter.drawText(int(canvas_rect.left()) + 2, int(plot.top()) + 8, f"{high:.3g}")
        painter.drawText(int(canvas_rect.left()) + 2, int(plot.bottom()), f"{low:.3g}")
        label_y = plot.bottom() + 6.0
        label_h = max(16.0, canvas_rect.bottom() - label_y - 2.0)
        start_label = "0 s" if abs(view_start) < 1e-6 else f"{view_start:.2f} s"
        painter.drawText(
            QRectF(plot.left(), label_y, 80.0, label_h),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            start_label,
        )
        painter.drawText(
            QRectF(plot.right() - 90.0, label_y, 90.0, label_h),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{view_end:.2f} s",
        )
        if plot.width() >= 320.0:
            for frac in (0.25, 0.5, 0.75):
                tick_x = plot.left() + frac * plot.width()
                tick_t = view_start + frac * (view_end - view_start)
                painter.setPen(QColor(COLOR_BORDER))
                painter.drawLine(QPointF(tick_x, plot.bottom()), QPointF(tick_x, plot.bottom() + 4.0))
                painter.setPen(QColor(COLOR_TEXT_MUTED))
                painter.drawText(
                    QRectF(tick_x - 32.0, label_y, 64.0, label_h),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                    f"{tick_t:.2f} s",
                )

        visible_points = sum(
            1 for seg in self.data.segments for ts, _ in seg if view_start <= ts <= view_end
        )
        px_per_point = plot.width() / max(1, visible_points)
        dot_radius = 2.0 if px_per_point >= 4.0 else 1.2 if px_per_point >= 2.0 else 0.0

        painter.save()
        painter.setClipRect(plot)
        for segment in self.data.segments:
            prev_t: float | None = None
            prev_v: float | None = None
            single_point = len(segment) == 1
            for timestamp, value in segment:
                current = point(timestamp, value)
                if (
                    prev_t is not None
                    and prev_v is not None
                    and not (
                        (prev_t < view_start and timestamp < view_start)
                        or (prev_t > view_end and timestamp > view_end)
                    )
                ):
                    for ta, va, tb, vb, piece_color in self._segment_pieces(
                        prev_t, prev_v, timestamp, value
                    ):
                        painter.setPen(QPen(QColor(piece_color), 1.5))
                        painter.drawLine(point(ta, va), point(tb, vb))
                if view_start <= timestamp <= view_end and (dot_radius > 0.0 or single_point):
                    dot_color = STATE_COLORS.get(self._point_state(timestamp, value), "#69B9CF")
                    painter.setPen(QPen(QColor(dot_color), 1.5))
                    radius = max(dot_radius, 1.5) if single_point else dot_radius
                    painter.drawEllipse(current, radius, radius)
                prev_t = timestamp
                prev_v = value
        painter.restore()
        if include_playhead and self._hover_point is not None and self.time_end > 0:
            hover_t, hover_val = self._hover_point
            if view_start <= hover_t <= view_end:
                hp = point(hover_t, hover_val)
                painter.save()
                painter.setClipRect(plot)
                painter.setPen(QPen(QColor(220, 226, 232, 130), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(hp.x(), plot.top()), QPointF(hp.x(), plot.bottom()))
                dot_color = STATE_COLORS.get(self._point_state(hover_t, hover_val), "#69B9CF")
                painter.setBrush(QColor(dot_color))
                painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
                painter.drawEllipse(hp, 4.0, 4.0)
                if self.hover_hint_text:
                    metrics = painter.fontMetrics()
                    text_w = metrics.horizontalAdvance(self.hover_hint_text) + 12.0
                    text_h = float(metrics.height()) + 6.0
                    bx = min(max(plot.left() + 4.0, hp.x() + 8.0), max(plot.left() + 4.0, plot.right() - text_w - 4.0))
                    by = max(plot.top() + 4.0, min(hp.y() - text_h - 6.0, plot.bottom() - text_h - 4.0))
                    badge = QRectF(bx, by, text_w, text_h)
                    painter.setBrush(QColor(20, 24, 31, 225))
                    painter.setPen(QPen(QColor(COLOR_BORDER), 1.0))
                    painter.drawRoundedRect(badge, 4.0, 4.0)
                    painter.setPen(QColor(COLOR_TEXT_PRIMARY))
                    painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, self.hover_hint_text)
                painter.restore()
        if include_playhead and self.playhead_timestamp is not None and self.time_end > 0:
            elapsed = max(0.0, min(self.time_end, self.playhead_timestamp - self.origin_timestamp))
            if view_start <= elapsed <= view_end:
                x = plot.left() + (elapsed - view_start) / span * plot.width()
                painter.setPen(QPen(QColor("#F06FD2"), 1.5))
                painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_chart(painter, QRectF(self.rect()))

    def render_to_image(
        self,
        width: int = 920,
        height: int = 240,
        *,
        title: str = "",
        subtitle: str = "",
    ) -> QImage:
        """Render this chart (with optional title and subtitle header) to an RGB QImage."""
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(QColor(COLOR_BASE))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        header_height = 0.0
        if title or subtitle:
            header_height = 46.0 if title and subtitle else 28.0
            painter.fillRect(QRectF(0, 0, width, header_height), QColor(COLOR_BASE))
            if title:
                title_font = QFont(painter.font())
                title_font.setPointSize(11)
                title_font.setBold(True)
                painter.setFont(title_font)
                painter.setPen(QColor(COLOR_TEXT_PRIMARY))
                painter.drawText(QRectF(14, 6, width - 28, 22), Qt.AlignmentFlag.AlignLeft, title)
            if subtitle:
                sub_font = QFont(painter.font())
                sub_font.setPointSize(9)
                sub_font.setBold(False)
                painter.setFont(sub_font)
                painter.setPen(QColor(COLOR_TEXT_MUTED))
                sub_top = 26.0 if title else 6.0
                painter.drawText(
                    QRectF(14, sub_top, width - 28, 18), Qt.AlignmentFlag.AlignLeft, subtitle
                )
        chart_font = QFont(painter.font())
        chart_font.setPointSize(9)
        chart_font.setBold(False)
        painter.setFont(chart_font)
        self._paint_chart(
            painter,
            QRectF(0, header_height, float(width), max(80.0, float(height) - header_height)),
            include_playhead=False,
        )
        painter.end()
        return image

    def svg_fragment(
        self,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        width: int = 920,
        height: int = 240,
        *,
        title: str = "",
        subtitle: str = "",
    ) -> str:
        """Build an SVG <g> fragment describing this chart and its calibration lines."""
        header_height = (46.0 if title and subtitle else 28.0) if (title or subtitle) else 0.0
        chart_height = max(80.0, float(height) - header_height)
        plot_left = 62.0
        plot_top = header_height + 12.0
        plot_width = max(1.0, float(width) - 82.0)
        plot_height = max(1.0, chart_height - 50.0)
        plot_right = plot_left + plot_width
        plot_bottom = plot_top + plot_height

        parts = [
            f'<g transform="translate({x_offset:.1f},{y_offset:.1f})">',
            (
                f'<rect x="0" y="0" width="{width}" height="{height}" fill="{COLOR_BASE}" '
                f'stroke="{COLOR_BORDER}" stroke-width="1"/>'
            ),
        ]
        if title:
            parts.append(
                f'<text x="14" y="20" fill="{COLOR_TEXT_PRIMARY}" font-family="sans-serif" '
                f'font-size="13" font-weight="bold">{escape(title)}</text>'
            )
        if subtitle:
            sub_y = 38 if title else 20
            parts.append(
                f'<text x="14" y="{sub_y}" fill="{COLOR_TEXT_MUTED}" font-family="sans-serif" '
                f'font-size="11">{escape(subtitle)}</text>'
            )

        y_limits = self._y_range()
        if self.data is None or not self.data.segments or y_limits is None:
            parts.append(
                f'<text x="{width / 2:.1f}" y="{header_height + chart_height / 2:.1f}" '
                f'text-anchor="middle" fill="{COLOR_TEXT_MUTED}" font-family="sans-serif" '
                'font-size="12">No valid persisted deltas — unknown</text>'
            )
            parts.append("</g>")
            return "\n".join(parts)

        low, high = y_limits
        duration = self.time_end or 1.0

        def px(timestamp: float) -> float:
            return plot_left + timestamp / duration * plot_width

        def py(value: float) -> float:
            return plot_bottom - (value - low) / (high - low) * plot_height

        for start, end in self.intervals:
            left = plot_left + max(0.0, start) / duration * plot_width
            right = plot_left + min(duration, end) / duration * plot_width
            if right > left:
                parts.append(
                    f'<rect x="{left:.2f}" y="{plot_top:.2f}" width="{right - left:.2f}" '
                    f'height="{plot_height:.2f}" fill="rgba(49,87,72,0.35)"/>'
                )

        if self.bounds is not None:
            y_low = py(self.bounds[0])
            y_high = py(self.bounds[1])
            band_top = min(y_low, y_high)
            band_height = max(1.0, abs(y_high - y_low))
            parts.append(
                f'<rect x="{plot_left:.2f}" y="{band_top:.2f}" width="{plot_width:.2f}" '
                f'height="{band_height:.2f}" fill="rgba(80,220,80,0.11)"/>'
            )
            for label, value in zip(("Min", "Max"), self.bounds, strict=True):
                y_val = py(value)
                text_y = max(plot_top + 12.0, min(plot_bottom - 3.0, y_val - 3.0))
                parts.append(
                    f'<line x1="{plot_left:.2f}" y1="{y_val:.2f}" '
                    f'x2="{plot_right:.2f}" y2="{y_val:.2f}" '
                    'stroke="#E3B66B" stroke-width="1.5" stroke-dasharray="6,4"/>'
                )
                parts.append(
                    f'<text x="{max(plot_left, plot_right - 95.0):.2f}" y="{text_y:.2f}" '
                    f'fill="#E3B66B" font-family="sans-serif" font-size="10">'
                    f'{label} {value:.3g}</text>'
                )

        if self.reference_line is not None:
            y_ref = py(self.reference_line)
            ref_y = max(plot_top + 12.0, min(plot_bottom - 3.0, y_ref - 4.0))
            parts.append(
                f'<line x1="{plot_left:.2f}" y1="{y_ref:.2f}" '
                f'x2="{plot_right:.2f}" y2="{y_ref:.2f}" '
                'stroke="#A3E635" stroke-width="1.5" stroke-dasharray="8,4,2,4"/>'
            )
            parts.append(
                f'<text x="{plot_left + 6.0:.2f}" y="{ref_y:.2f}" '
                f'fill="#A3E635" font-family="sans-serif" font-size="10">'
                f'Ref {self.reference_line:.3g}</text>'
            )

        parts.append(
            f'<rect x="{plot_left:.2f}" y="{plot_top:.2f}" width="{plot_width:.2f}" '
            f'height="{plot_height:.2f}" fill="none" stroke="{COLOR_BORDER}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="4" y="{plot_top + 10:.2f}" fill="{COLOR_TEXT_MUTED}" '
            f'font-family="sans-serif" font-size="10">{high:.3g}</text>'
        )
        parts.append(
            f'<text x="4" y="{plot_bottom:.2f}" fill="{COLOR_TEXT_MUTED}" '
            f'font-family="sans-serif" font-size="10">{low:.3g}</text>'
        )
        parts.append(
            f'<text x="{plot_left:.2f}" y="{header_height + chart_height - 8:.2f}" '
            f'fill="{COLOR_TEXT_MUTED}" font-family="sans-serif" font-size="10">0 s</text>'
        )
        parts.append(
            f'<text x="{plot_right - 4.0:.2f}" y="{header_height + chart_height - 8:.2f}" '
            f'text-anchor="end" fill="{COLOR_TEXT_MUTED}" font-family="sans-serif" '
            f'font-size="10">{self.time_end:.2f} s</text>'
        )

        total_points = sum(len(seg) for seg in self.data.segments)
        px_per_point = plot_width / max(1, total_points)
        dot_radius = 2.0 if px_per_point >= 4.0 else 1.2 if px_per_point >= 2.0 else 0.0

        for segment in self.data.segments:
            prev_t: float | None = None
            prev_v: float | None = None
            single_point = len(segment) == 1
            for timestamp, value in segment:
                dot_color = STATE_COLORS.get(self._point_state(timestamp, value), "#69B9CF")
                cx, cy = px(timestamp), py(value)
                if prev_t is not None and prev_v is not None:
                    for ta, va, tb, vb, piece_color in self._segment_pieces(
                        prev_t, prev_v, timestamp, value
                    ):
                        parts.append(
                            f'<line x1="{px(ta):.2f}" y1="{py(va):.2f}" '
                            f'x2="{px(tb):.2f}" y2="{py(vb):.2f}" '
                            f'stroke="{piece_color}" stroke-width="1.5"/>'
                        )
                if dot_radius > 0.0 or single_point:
                    r = max(dot_radius, 1.5) if single_point else dot_radius
                    parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:g}" fill="{dot_color}"/>')
                prev_t = timestamp
                prev_v = value

        parts.append("</g>")
        return "\n".join(parts)

    def to_svg_document(
        self,
        width: int = 920,
        height: int = 240,
        *,
        title: str = "",
        subtitle: str = "",
    ) -> str:
        fragment = self.svg_fragment(0.0, 0.0, width, height, title=title, subtitle=subtitle)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
            f"{fragment}\n"
            "</svg>\n"
        )

    def export_to_file(
        self,
        destination: Path,
        *,
        title: str = "",
        subtitle: str = "",
        width: int = 920,
        height: int = 240,
    ) -> Path:
        """Export this chart as PNG/JPG/BMP image, SVG vector graphic, or CSV series."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix.lower()
        if suffix in IMAGE_EXPORT_SUFFIXES:
            image = self.render_to_image(width=width, height=height, title=title, subtitle=subtitle)
            if not image.save(str(destination)):
                raise OSError(f"Could not save chart image to {destination}")
            return destination
        if suffix == ".svg":
            destination.write_text(
                self.to_svg_document(width=width, height=height, title=title, subtitle=subtitle),
                encoding="utf-8",
            )
            return destination
        if suffix == ".csv":
            min_bound = f"{self.bounds[0]:.6g}" if self.bounds is not None else ""
            max_bound = f"{self.bounds[1]:.6g}" if self.bounds is not None else ""
            ref_val = f"{self.reference_line:.6g}" if self.reference_line is not None else ""
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["timestamp_s", "value", "state", "min_bound", "max_bound", "reference_line"]
                )
                if self.data is not None:
                    for timestamp, value in self.data.points:
                        state = self.states.get(timestamp, "UNKNOWN" if value is None else "")
                        writer.writerow([
                            f"{timestamp:.6f}",
                            "" if value is None else f"{value:.6g}",
                            state,
                            min_bound,
                            max_bound,
                            ref_val,
                        ])
            return destination
        raise ValueError(f"Unsupported chart export format: {destination.suffix or '(none)'}")
