"""Reusable chart of timestamped evidence without gap interpolation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
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
        self.setMinimumHeight(155)
        self.setEnabled(False)
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
        self.setEnabled(data is not None and bool(data.segments))
        self.update()

    def set_playhead(self, timestamp: float | None) -> None:
        self.playhead_timestamp = timestamp
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        left, right = 62.0, float(max(63, self.width() - 20))
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.isEnabled()
            and self.time_end > 0
            and left <= event.position().x() <= right
        ):
            fraction = (event.position().x() - left) / (right - left)
            self.seek_requested.emit(self.origin_timestamp + fraction * self.time_end)
        super().mousePressEvent(event)

    def _y_range(self) -> tuple[float, float] | None:
        if self.data is None or not self.data.segments:
            return None
        values = [value for segment in self.data.segments for _, value in segment]
        if self.bounds is not None:
            values.extend(value for value in self.bounds if math.isfinite(value))
        if self.reference_line is not None and math.isfinite(self.reference_line):
            values.append(self.reference_line)
        if not values:
            return None
        low, high = min(values), max(values)
        padding = max((high - low) * 0.1, 0.001)
        return low - padding, high + padding

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
        duration = self.time_end or 1.0

        def point(timestamp: float, value: float) -> QPointF:
            return QPointF(
                plot.left() + timestamp / duration * plot.width(),
                plot.bottom() - (value - low) / (high - low) * plot.height(),
            )

        painter.save()
        painter.setClipRect(plot)
        for start, end in self.intervals:
            left = plot.left() + max(0.0, start) / duration * plot.width()
            right = plot.left() + min(duration, end) / duration * plot.width()
            if right > left:
                painter.fillRect(
                    QRectF(left, plot.top(), right - left, plot.height()),
                    QColor(49, 87, 72, 90),
                )
        if self.bounds is not None:
            y_low = point(0, self.bounds[0]).y()
            y_high = point(0, self.bounds[1]).y()
            band_top = min(y_low, y_high)
            band_height = max(1.0, abs(y_high - y_low))
            painter.fillRect(
                QRectF(plot.left(), band_top, plot.width(), band_height),
                QColor(80, 220, 80, 28),
            )
            painter.setPen(QPen(QColor("#E3B66B"), 1.5, Qt.PenStyle.DashLine))
            for label, value in zip(("Min", "Max"), self.bounds, strict=True):
                y = point(0, value).y()
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                text_y = max(plot.top() + 12.0, min(plot.bottom() - 3.0, y - 3.0))
                painter.drawText(
                    QPointF(max(plot.left(), plot.right() - 95.0), text_y),
                    f"{label} {value:.3g}",
                )
        if self.reference_line is not None:
            y_ref = point(0, self.reference_line).y()
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
        painter.drawText(int(plot.left()), int(canvas_rect.bottom()) - 8, "0 s")
        painter.drawText(
            int(plot.right()) - 90, int(canvas_rect.bottom()) - 8, f"{self.time_end:.2f} s"
        )
        for segment in self.data.segments:
            previous: QPointF | None = None
            for timestamp, value in segment:
                painter.setPen(QPen(QColor(STATE_COLORS.get(
                    self.states.get(timestamp, ""), "#69B9CF",
                )), 1.5))
                current = point(timestamp, value)
                if previous is not None:
                    painter.drawLine(previous, current)
                painter.drawEllipse(current, 2, 2)
                previous = current
        if include_playhead and self.playhead_timestamp is not None and self.time_end > 0:
            elapsed = max(0.0, min(self.time_end, self.playhead_timestamp - self.origin_timestamp))
            x = plot.left() + elapsed / duration * plot.width()
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

        for segment in self.data.segments:
            prev_xy: tuple[float, float] | None = None
            for timestamp, value in segment:
                color = STATE_COLORS.get(self.states.get(timestamp, ""), "#69B9CF")
                cx, cy = px(timestamp), py(value)
                if prev_xy is not None:
                    parts.append(
                        f'<line x1="{prev_xy[0]:.2f}" y1="{prev_xy[1]:.2f}" '
                        f'x2="{cx:.2f}" y2="{cy:.2f}" stroke="{color}" stroke-width="1.5"/>'
                    )
                parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2" fill="{color}"/>')
                prev_xy = (cx, cy)

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
