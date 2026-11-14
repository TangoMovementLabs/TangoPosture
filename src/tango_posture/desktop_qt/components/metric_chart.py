"""Reusable chart of timestamped evidence without gap interpolation."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QWidget,
)

from ..analytics.data import (
    MetricData,
)
from ..theme import COLOR_BASE, COLOR_BORDER, COLOR_TEXT_MUTED

STATE_COLORS = {
    "WITHIN_REFERENCE": "#50DC50",
    "PENDING": "#FFBE00",
    "DEVIATION": "#FF0000",
    "UNKNOWN": "#B4B4B4",
}


class MetricChart(QWidget):
    """Draw stored samples without interpolating across missing coverage."""

    seek_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.data: MetricData | None = None
        self.time_end = 0.0
        self.origin_timestamp = 0.0
        self.bounds: tuple[float, float] | None = None
        self.intervals: list[tuple[float, float]] = []
        self.playhead_timestamp: float | None = None
        self.states: dict[float, str] = {}
        self.setMinimumHeight(155)
        self.setEnabled(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Persisted smoothed delta chart")

    def set_data(
        self, data: MetricData | None, time_end: float = 0.0, origin_timestamp: float = 0.0,
        *, bounds: tuple[float, float] | None = None,
        intervals: list[tuple[float, float]] | None = None,
        states: dict[float, str] | None = None,
    ) -> None:
        self.data, self.time_end, self.origin_timestamp = data, time_end, origin_timestamp
        self.bounds = bounds
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

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLOR_BASE))
        painter.setPen(QColor(COLOR_TEXT_MUTED))
        if self.data is None or not self.data.segments:
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No valid persisted deltas — unknown"
            )
            return
        plot = QRectF(62, 12, max(1, self.width() - 82), max(1, self.height() - 50))
        values = [value for segment in self.data.segments for _, value in segment]
        if self.bounds is not None:
            values.extend(self.bounds)
        low, high = min(values), max(values)
        padding = max((high - low) * 0.1, 0.001)
        low, high = low - padding, high + padding
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
                painter.fillRect(QRectF(left, plot.top(), right - left, plot.height()),
                                 QColor(49, 87, 72, 90))
        if self.bounds is not None:
            painter.setPen(QPen(QColor("#E3B66B"), 1.5, Qt.PenStyle.DashLine))
            for label, value in zip(("Min", "Max"), self.bounds, strict=True):
                y = point(0, value).y()
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                painter.drawText(QPointF(max(plot.left(), plot.right() - 90), y - 3),
                                 f"{label} {value:.3g}")
        painter.restore()
        painter.setPen(QColor(COLOR_BORDER))
        painter.drawRect(plot)
        painter.setPen(QColor(COLOR_TEXT_MUTED))
        painter.drawText(2, int(plot.top()) + 8, f"{high:.3g}")
        painter.drawText(2, int(plot.bottom()), f"{low:.3g}")
        painter.drawText(int(plot.left()), self.height() - 8, "0 s")
        painter.drawText(int(plot.right()) - 90, self.height() - 8, f"{self.time_end:.2f} s")
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
        if self.playhead_timestamp is not None and self.time_end > 0:
            elapsed = max(0.0, min(self.time_end, self.playhead_timestamp - self.origin_timestamp))
            x = plot.left() + elapsed / duration * plot.width()
            painter.setPen(QPen(QColor("#F06FD2"), 1.5))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
