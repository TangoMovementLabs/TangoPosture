"""Screen routes and navigation state for the desktop application."""

from __future__ import annotations

from enum import StrEnum

from PySide6.QtCore import QObject, Signal

from ..troubleshoot import is_troubleshoot_enabled, log_event


class Route(StrEnum):
    HOME = "home"
    PRACTICE = "practice"
    REFERENCES = "references"
    LIBRARY = "library"
    ANALYSIS = "analysis"
    REVIEW = "review"
    ANALYTICS = "analytics"
    COACH = "coach"
    SETTINGS = "settings"
    DATABASE = "database"
    HELP = "help"


def _summarize_payload(payload: object) -> str:
    if payload is None:
        return "None"
    if isinstance(payload, dict):
        return f"dict(keys={sorted(payload.keys())})"
    return repr(payload)[:120]


class Router(QObject):
    route_changed = Signal(Route, object)  # route, optional payload

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._current_route = Route.HOME
        self._history: list[Route] = [Route.HOME]
        if is_troubleshoot_enabled():
            log_event("ui.route", f"Initial menu entered: {self._current_route.value!r}")

    @property
    def current(self) -> Route:
        return self._current_route

    def navigate(self, target: Route | str, payload: object = None) -> None:
        route = Route(target)
        if route != self._current_route or payload is not None:
            prev = self._current_route
            self._current_route = route
            self._history.append(route)
            if is_troubleshoot_enabled():
                log_event(
                    "ui.route",
                    f"Menu entered: {route.value!r} from={prev.value!r} "
                    f"payload={_summarize_payload(payload)} depth={len(self._history)}",
                )
            self.route_changed.emit(route, payload)

    def go_back(self) -> bool:
        if len(self._history) > 1:
            curr = self._history.pop()  # pop current
            prev = self._history[-1]
            self._current_route = prev
            if is_troubleshoot_enabled():
                log_event(
                    "ui.route",
                    f"Menu back: {prev.value!r} from={curr.value!r} depth={len(self._history)}",
                )
            self.route_changed.emit(prev, None)
            return True
        return False
