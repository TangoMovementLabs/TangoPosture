"""Bridge connecting Qt screens to in-process services and background workers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QIODevice, QObject, QSaveFile, Signal
from PySide6.QtGui import QImage

from ..config import Config, RangeDefault, export_toml
from ..core.entities import ReferenceSummary
from ..core.session_state import SessionState
from ..core.types import ArchiveKind

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

    from ..calibration.models import RangeProfile


def numpy_to_qimage(image_bgr: Any) -> QImage:
    """Convert an OpenCV BGR numpy array to a QImage without external file writes."""
    import cv2

    height, width, channels = image_bgr.shape
    bytes_per_line = channels * width
    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Note: store buffer reference so memory isn't collected prematurely
    qimg = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return qimg.copy()


def parse_created_at(created_at: str | None, path: Path | None = None) -> tuple[str, float]:
    """Return (dd/mm/yyyy HH:mm label, unix_timestamp) for an ISO timestamp or file path."""
    if isinstance(created_at, str) and created_at.strip():
        raw = created_at.strip()
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is not None:
                dt = dt.astimezone()
            return dt.strftime("%d/%m/%Y %H:%M"), dt.timestamp()
        except ValueError:
            pass
    if path is not None:
        try:
            if path.exists():
                mtime = path.stat().st_mtime
                dt = datetime.fromtimestamp(mtime)
                return dt.strftime("%d/%m/%Y %H:%M"), float(mtime)
        except OSError:
            pass
    return "Recent", 0.0


def format_created_at(created_at: str | None, path: Path | None = None) -> str:
    """Format an ISO timestamp or file modification time as dd/mm/yyyy HH:mm."""
    return parse_created_at(created_at, path)[0]


def created_at_timestamp(created_at: str | None, path: Path | None = None) -> float:
    """Return a sortable local/UTC unix timestamp for an ISO timestamp or file path."""
    return parse_created_at(created_at, path)[1]


def sort_named_dated_items(
    items: list[dict[str, Any]], sort_by: str | None
) -> list[dict[str, Any]]:
    """Sort archive or item dictionaries by name or creation timestamp."""
    ordered = list(items)
    if sort_by == "name_asc":
        ordered.sort(
            key=lambda item: (str(item.get("name", "")).casefold(), str(item.get("path", "")))
        )
    elif sort_by == "name_desc":
        ordered.sort(
            key=lambda item: (str(item.get("name", "")).casefold(), str(item.get("path", ""))),
            reverse=True,
        )
    elif sort_by == "date_asc":
        ordered.sort(
            key=lambda item: created_at_timestamp(item.get("created_at", ""), item.get("path"))
        )
    elif sort_by == "date_desc":
        ordered.sort(
            key=lambda item: -created_at_timestamp(item.get("created_at", ""), item.get("path"))
        )
    return ordered


class ServiceBridge(QObject):
    """Coordinates local storage, config, and background tasks."""

    status_message = Signal(str)
    config_changed = Signal(object)
    session_state_changed = Signal(object)  # SessionState
    frame_emitted = Signal(QImage, str)     # video frame, frame note summary
    beat_emitted = Signal(float, int, bool) # bpm, beat_num, is_active
    practice_finished = Signal(int, str, object)  # code, message, Path | None

    def __init__(self, cfg: Config, parent: QObject | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self._active_session: SessionState = SessionState()
        self._running_worker = None
        self._reference_cache: dict[tuple[Path, int, int], ReferenceSummary] = {}
        self._manifest_cache: dict[tuple[Path, int, int], dict[str, Any]] = {}

    @property
    def config(self) -> Config:
        return self.cfg

    def update_config(self, new_cfg: Config) -> None:
        self.cfg = new_cfg
        self.config_changed.emit(new_cfg)
        self.status_message.emit("Settings updated.")

    def use_calibration_defaults(self, profile: RangeProfile, parent: QWidget) -> bool:
        from PySide6.QtWidgets import QMessageBox

        destination = Path("settings.toml").resolve()
        names = ", ".join(profile.metrics)
        answer = QMessageBox.question(
            parent, "Overwrite Settings defaults?",
            f"Replace the default ranges with {profile.name} ({names}) and save all current "
            f"Settings to:\n{destination}\n\nAny existing settings.toml will be overwritten. "
            "Skipped metrics stay unavailable; camera scoring approval is not transferred.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        try:
            data = self.cfg.model_dump(exclude={"axis_curve"})
            data["calibration_defaults"] = {
                key: RangeDefault(
                    lower=target.lower, upper=target.upper, unit=target.unit, view=target.view,
                ).model_dump()
                for key, target in profile.metrics.items()
            }
            cfg = Config.model_validate(data)
            self.save_settings(cfg, destination)
        except (OSError, ValueError) as error:
            self.status_message.emit(f"Could not save calibration defaults: {error}")
            return False
        self.status_message.emit(f"Calibration defaults saved to {destination}")
        return True

    def save_settings(self, cfg: Config, destination: Path) -> None:
        payload = export_toml(cfg).encode("utf-8")
        output = QSaveFile(str(destination))
        if not output.open(QIODevice.OpenModeFlag.WriteOnly):
            raise OSError(output.errorString())
        if output.write(payload) != len(payload):
            error = output.errorString()
            output.cancelWriting()
            raise OSError(error)
        if not output.commit():
            raise OSError(output.errorString())
        self.update_config(cfg)

    def list_references(self) -> list[ReferenceSummary]:
        """Discover existing .tpc and legacy JSON calibrations."""
        from ..calibration.baseline import load_calibration_file
        from ..calibration.models import profile_from_calibration
        from ..storage import Storage

        data_dir = Path(self.cfg.data_dir)
        folder = Path("calibrations") if self.cfg.data_dir == "data" else data_dir / "calibrations"
        results: dict[str, ReferenceSummary] = {}

        if (data_dir / "tango.sqlite3").is_file():
            storage = Storage(data_dir)
            try:
                for row in storage.list("calibrations"):
                    results[row["id"]] = ReferenceSummary(
                        id=row["id"],
                        name=row.get("name", "Reference"),
                        path=folder / f"{row['id']}.tpc",
                        side=row.get("side", "left"),
                        facing=row.get("facing", "right"),
                        camera=str(row.get("camera", 0)),
                        created_at=row.get("created_at_utc", ""),
                        is_archive=True,
                    )
            except (OSError, ValueError, KeyError):
                pass
            finally:
                storage.close()

        if folder.is_dir():
            for path in sorted([*folder.glob("calibration_*.json"), *folder.glob("*.tpc")], reverse=True):
                try:
                    resolved = path.resolve()
                    stat_result = resolved.stat()
                    cache_key = (resolved, stat_result.st_mtime_ns, stat_result.st_size)
                    cached = self._reference_cache.get(cache_key)
                    if cached is not None:
                        results[cached.id] = cached
                        continue
                    ref = load_calibration_file(str(path), folder)
                    if ref:
                        prof = getattr(ref, "profile", {})
                        ranges = profile_from_calibration(ref)
                        summary = ReferenceSummary(
                            id=ref.id,
                            name=getattr(ref, "name", path.stem),
                            path=resolved,
                            side=prof.get("side", "left"),
                            facing=prof.get("facing", "right"),
                            camera=str(prof.get("camera", 0)),
                            created_at=getattr(ref, "created_at_utc", ""),
                            is_archive=path.suffix.lower() == ".tpc",
                            scoring_ready=ranges is None or any(
                                value.camera_check_passed for value in ranges.metrics.values()
                            ),
                        )
                        self._reference_cache[cache_key] = summary
                        results[ref.id] = summary
                except (OSError, ValueError, KeyError):
                    continue

        return list(results.values())

    def _read_cached_manifest(self, archive_path: Path) -> dict[str, Any]:
        from ..archive import read_manifest

        resolved = archive_path.resolve()
        stat_result = resolved.stat()
        cache_key = (resolved, stat_result.st_mtime_ns, stat_result.st_size)
        cached = self._manifest_cache.get(cache_key)
        if cached is not None:
            return cached
        manifest = read_manifest(archive_path)
        self._manifest_cache[cache_key] = manifest
        return manifest

    def list_practices(self) -> list[dict[str, Any]]:
        """List practice archives (.tp) using their persisted display names."""
        from ..archive import practice_directory

        folder = practice_directory(self.cfg.data_dir)
        items = []
        if folder.is_dir():
            for archive_path in sorted(folder.glob("*.tp"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
                try:
                    manifest = self._read_cached_manifest(archive_path)
                    segments = manifest.get("segments", [])
                    items.append({
                        "name": manifest.get("name", archive_path.stem),
                        "path": archive_path.resolve(),
                        "kind": ArchiveKind.PRACTICE,
                        "created_at": manifest.get("created_at_utc", ""),
                        "segments_count": len(segments),
                        "segments": segments,
                    })
                except (OSError, ValueError, KeyError) as error:
                    self.status_message.emit(f"Library archive unavailable: {archive_path.name}: {error}")
        return items

    def list_library_archives(self) -> list[dict[str, Any]]:
        from ..archive import calibration_directory
        from ..references.archives import reference_directory

        items = self.list_practices()
        for folder, suffix, kind in (
            (calibration_directory(self.cfg.data_dir), "*.tpc", ArchiveKind.CALIBRATION),
            (reference_directory(self.cfg.data_dir), "*.tpr", ArchiveKind.REFERENCE),
        ):
            for path in sorted(folder.glob(suffix)):
                try:
                    manifest = self._read_cached_manifest(path)
                    items.append({
                        "name": manifest.get("name", path.stem),
                        "path": path.resolve(),
                        "kind": kind,
                        "version": manifest["version"],
                        "created_at": manifest.get("created_at_utc", ""),
                        "segments_count": len(manifest["segments"]),
                    })
                except (OSError, ValueError) as error:
                    self.status_message.emit(f"Library archive unavailable: {path.name}: {error}")
        return items

    def choose_library_archive(
        self, parent: Any, title: str, sort_by: str | None = None
    ) -> Path | None:
        from PySide6.QtWidgets import QInputDialog

        items = [item for item in self.list_library_archives() if item["segments_count"]]
        if not items:
            self.status_message.emit("Library is empty. Import a video through Library first.")
            return None
        if sort_by is not None:
            items = sort_named_dated_items(items, sort_by)
        labels = []
        for item in items:
            date_label = format_created_at(item.get("created_at", ""), item.get("path"))
            if date_label != "Recent":
                labels.append(f"{item['name']} ({date_label}) — {item['path'].name}")
            else:
                labels.append(f"{item['name']} — {item['path'].name}")
        selected, accepted = QInputDialog.getItem(parent, title, "Library archive:", labels, 0, False)
        return items[labels.index(selected)]["path"] if accepted else None

    def list_reports(self) -> list[Path]:
        from ..review import report_directory

        folder = report_directory(self.cfg.data_dir)
        if folder.is_dir():
            return sorted(folder.glob("report_*.md"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        return []

    def export_settings_toml(self, destination: Path) -> None:
        destination.write_text(export_toml(self.cfg), encoding="utf-8")
        self.status_message.emit(f"Settings saved to {destination.name}")

    def import_settings_toml(self, source: Path) -> None:
        import tomllib

        self.update_config(Config.model_validate(tomllib.loads(source.read_text(encoding="utf-8"))))
        self.status_message.emit(f"Loaded settings from {source.name}")
