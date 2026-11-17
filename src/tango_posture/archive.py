"""Portable practice archives and temporary, validated replay extraction."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import stat
import sys
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

if sys.platform != "win32":
    import fcntl
else:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from .domain import Calibration

logger = logging.getLogger(__name__)

_locks: dict[Path, threading.RLock] = {}
_locks_guard = threading.Lock()
_lock_state = threading.local()

def _replace_archive(temporary: Path, destination: Path) -> None:
    # Windows readers and file scanners may briefly deny an otherwise valid replacement.
    for delay in (0.025, 0.05, 0.1, 0.2, 0.4):
        try:
            temporary.replace(destination)
            return
        except PermissionError as error:
            if sys.platform != "win32" or getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            time.sleep(delay)
    temporary.replace(destination)


@contextmanager
def _archive_lock(path: Path):
    resolved = path.resolve()
    with _locks_guard:
        thread_lock = _locks.setdefault(resolved, threading.RLock())
    with thread_lock:
        held = getattr(_lock_state, "paths", None)
        if held is None:
            held = _lock_state.paths = set()
        if resolved in held:
            yield
            return
        lock_file = None
        if fcntl is not None:
            with suppress(OSError):
                resolved.parent.mkdir(parents=True, exist_ok=True)
                lock_path = resolved.with_name(f".{resolved.name}.lock")
                lock_file = lock_path.open("a+", encoding="utf-8")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            held.add(resolved)
            yield
        finally:
            held.discard(resolved)
            if lock_file is not None and fcntl is not None:
                with suppress(OSError):
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                with suppress(OSError):
                    lock_file.close()


def movement_member_name(index: int) -> str:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("Invalid practice segment index")
    return f"movement_{index:02d}.v1.json"


def analytics_member_name(index: int) -> str:
    movement_member_name(index)
    return f"analytics_{index:02d}.json"


def _rewrite_members(path: Path, replacements: dict[str, bytes | Path]) -> None:
    with NamedTemporaryFile(dir=path.parent, suffix=".archive.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(path) as previous, ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output:
            for member in previous.infolist():
                if member.filename not in replacements:
                    with previous.open(member) as source, output.open(member, "w") as target:
                        shutil.copyfileobj(source, target)
            for name, content in replacements.items():
                if isinstance(content, Path):
                    output.write(content, name, compress_type=ZIP_STORED)
                else:
                    output.writestr(name, content)
        _replace_archive(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rename_archive(path: Path, name: str) -> None:
    name = name.strip()
    if not name or any(ord(char) < 32 for char in name):
        raise ValueError("Enter a nonempty library name without control characters")
    with _archive_lock(path):
        manifest = read_manifest(path)
        manifest["name"] = name
        _rewrite_members(path, {"manifest.json": json.dumps(manifest).encode("utf-8")})


def read_analytics(path: Path, index: int) -> dict[str, Any] | None:
    name = analytics_member_name(index)
    if path.is_dir():
        member = path / name
        if not member.is_file():
            return None
        raw = member.read_bytes()
    else:
        with _archive_lock(path), ZipFile(path) as archive:
            manifest = read_manifest(path)
            if not 0 <= index < len(manifest["segments"]):
                raise ValueError("Recording not found in archive")
            if name not in archive.namelist():
                return None
            raw = archive.read(name)
    return _validate_analytics(json.loads(raw), index)


def _validate_analytics(data: Any, index: int) -> dict[str, Any]:
    name = analytics_member_name(index)
    if (
        not isinstance(data, dict) or data.get("schema_version") != 1
        or data.get("segment_index") != index
    ):
        raise ValueError(f"Invalid indexed analytics: {name}")
    for key in ("session", "coach_data", "movement_data"):
        if data.get(key) is not None and not isinstance(data[key], dict):
            raise ValueError(f"Invalid archived analytics object: {key}")
    timestamps = data.get("frame_timestamps", [])
    if (
        not isinstance(timestamps, list)
        or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0 for value in timestamps
        )
        or any(left >= right for left, right in pairwise(timestamps))
    ):
        raise ValueError("Invalid archived analytics frame timestamps")
    frames = data.get("frames", [])
    if not isinstance(frames, list) or any(not isinstance(frame, dict) for frame in frames):
        raise ValueError("Invalid archived analytics frame measurements")
    coach_frames = (data.get("coach_data") or {}).get("frames", {})
    if not isinstance(coach_frames, dict):
        raise TypeError("Invalid archived coach frames")
    for key, entry in coach_frames.items():
        if (
            not isinstance(key, str) or not key.isdecimal()
            or not isinstance(entry, dict)
            or (timestamps and int(key) >= len(timestamps))
        ):
            raise ValueError("Invalid archived coach frame index")
    from .pose.recovery import validate_pose_recovery_payload

    coach = data.get("coach_data")
    valid_indices = {int(k) for k in coach_frames} if coach_frames else None
    ts_list = [float(t) for t in timestamps] if timestamps else None
    frame_count = len(timestamps) if timestamps else None
    if isinstance(coach, dict) and coach.get("pose_recovery") is not None:
        try:
            dancer_sel = (
                coach.get("dancer_selection")
                if isinstance(coach.get("dancer_selection"), dict)
                else None
            )
            validate_pose_recovery_payload(
                coach["pose_recovery"],
                frame_count=frame_count,
                valid_frame_indices=valid_indices,
                timestamps=ts_list,
                source_sha256=coach.get("source_sha256"),
                model_sha256=coach.get("model_hash"),
                source_width=coach.get("width"),
                source_height=coach.get("height"),
                dancer_selection=dancer_sel,
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Ignoring invalid or stale pose_recovery in archived analytics %s: %s", name, exc)
            coach = dict(coach)
            coach.pop("pose_recovery", None)
            data = {**data, "coach_data": coach}
    if data.get("pose_recovery") is not None:
        try:
            coach_meta = data.get("coach_data") if isinstance(data.get("coach_data"), dict) else {}
            validate_pose_recovery_payload(
                data["pose_recovery"],
                frame_count=frame_count,
                valid_frame_indices=valid_indices,
                timestamps=ts_list,
                source_sha256=coach_meta.get("source_sha256"),
                model_sha256=coach_meta.get("model_hash") or data.get("model_hash"),
                source_width=coach_meta.get("width"),
                source_height=coach_meta.get("height"),
                dancer_selection=(
                    coach_meta.get("dancer_selection")
                    if isinstance(coach_meta.get("dancer_selection"), dict)
                    else None
                ),
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Ignoring invalid or stale top-level pose_recovery in %s: %s", name, exc)
            data = {k: v for k, v in data.items() if k != "pose_recovery"}
    dancer_analysis = data.get("dancer_analysis")
    if dancer_analysis is not None:
        if not isinstance(dancer_analysis, dict) or dancer_analysis.get("schema_version") != 1:
            raise ValueError(f"Invalid archived dancer_analysis schema in {name}")
        active_mode = dancer_analysis.get("active_mode")
        if active_mode not in ("single", "couple"):
            raise ValueError(f"Invalid archived dancer_analysis active_mode in {name}")
        runs = dancer_analysis.get("runs")
        if not isinstance(runs, dict):
            raise ValueError(f"Invalid archived dancer_analysis runs in {name}")
        for mode_key, entry in runs.items():
            if mode_key not in ("single", "couple") or not isinstance(entry, dict):
                raise ValueError(f"Invalid archived dancer_analysis run entry {mode_key!r} in {name}")
    return data


def store_analytics(
    path: Path, index: int, payload: dict[str, Any], *, expected_video_sha256: str | None = None,
    report_section: str | None = None,
    report_text: str | None = None,
    overlay_path: Path | None = None,
    lines_path: Path | None = None,
    metadata_updates: dict[str, Any] | None = None,
    frame_log: list[dict[str, Any]] | None = None,
    dancer_analysis_run: dict[str, Any] | None = None,
) -> None:
    with _archive_lock(path):
        manifest = read_manifest(path)
        if not 0 <= index < len(manifest["segments"]):
            raise ValueError("Recording not found in archive")
        if expected_video_sha256 is not None:
            video = manifest["segments"][index].get("video")
            with ZipFile(path) as archive:
                if not video or _stream_member_sha256(archive, video) != expected_video_sha256:
                    raise ValueError("Recorded video changed before analytics could be saved")
        data = {**payload, "schema_version": 1, "segment_index": index}
        previous = read_analytics(path, index)
        if previous is not None:
            data = {**previous, **data}
        if dancer_analysis_run is not None or "dancer_analysis" in data or (
            isinstance(previous, dict) and "dancer_analysis" in previous
        ):
            prev_da = (
                previous.get("dancer_analysis")
                if isinstance(previous, dict) and isinstance(previous.get("dancer_analysis"), dict)
                else None
            )
            payload_da = (
                payload.get("dancer_analysis")
                if isinstance(payload.get("dancer_analysis"), dict)
                else None
            )
            runs: dict[str, Any] = {}
            if prev_da and isinstance(prev_da.get("runs"), dict):
                runs.update({k: dict(v) for k, v in prev_da["runs"].items() if isinstance(v, dict)})
            elif isinstance(previous, dict) and (
                previous.get("coach_data") or previous.get("movement_data") or previous.get("session")
            ):
                runs["single"] = {
                    "run_id": str(previous.get("run_id") or f"legacy-single-{index}"),
                    "requested_mode": "single",
                    "resolved_mode": "single",
                    "coach_data": previous.get("coach_data"),
                    "movement_data": previous.get("movement_data"),
                    "session": previous.get("session"),
                    "frames": previous.get("frames", []),
                    "frame_timestamps": previous.get("frame_timestamps", []),
                }
            if payload_da and isinstance(payload_da.get("runs"), dict):
                runs.update({k: dict(v) for k, v in payload_da["runs"].items() if isinstance(v, dict)})
            active_mode = str(
                (payload_da or {}).get("active_mode")
                or (prev_da or {}).get("active_mode")
                or "single"
            )
            active_run_id = str(
                (payload_da or {}).get("active_run_id")
                or (prev_da or {}).get("active_run_id")
                or ""
            )
            if dancer_analysis_run is not None:
                run_mode = str(
                    dancer_analysis_run.get("resolved_mode")
                    or dancer_analysis_run.get("mode")
                    or "single"
                ).lower()
                if run_mode not in ("single", "couple"):
                    raise ValueError(f"Unsupported dancer analysis run mode: {run_mode!r}")
                couple_res = dancer_analysis_run.get("couple_result")
                run_id = str(
                    dancer_analysis_run.get("run_id")
                    or (couple_res.get("run_id") if isinstance(couple_res, dict) else "")
                    or f"{run_mode}-{index}-{int(time.time() * 1000)}"
                )
                runs[run_mode] = {
                    **dancer_analysis_run,
                    "resolved_mode": run_mode,
                    "run_id": run_id,
                }
                active_mode = run_mode
                active_run_id = run_id
            if active_mode not in runs and runs:
                active_mode = "couple" if "couple" in runs else "single"
                active_run_id = str(runs[active_mode].get("run_id") or "")
            data["dancer_analysis"] = {
                "schema_version": 1,
                "active_mode": active_mode,
                "active_run_id": active_run_id,
                "runs": runs,
            }
        _validate_analytics(data, index)
        manifest["segments"][index]["analytics"] = analytics_member_name(index)
        replacements: dict[str, bytes | Path] = {
            analytics_member_name(index): json.dumps(data, allow_nan=False).encode("utf-8"),
        }
        if report_text is not None:
            replacements[manifest["segments"][index]["report"]] = report_text.encode("utf-8")
        for key, source in (("overlay", overlay_path), ("lines", lines_path)):
            if source is not None:
                member = f"{key}_{index:02d}.mp4"
                manifest["segments"][index][key] = member
                replacements[member] = source
        if lines_path is not None or frame_log is not None:
            with ZipFile(path) as archive:
                names = set(archive.namelist())
            legacy_lines = f"overlay_{index:02d}_lines.mp4"
            legacy_notes = f"overlay_{index:02d}.notes.jsonl"
            if lines_path is not None and legacy_lines in names:
                replacements[legacy_lines] = lines_path
            if frame_log is not None and legacy_notes in names:
                replacements[legacy_notes] = "".join(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                    for row in frame_log
                ).encode("utf-8")
        if metadata_updates is not None:
            metadata_name = manifest["segments"][index]["metadata"]
            with ZipFile(path) as archive:
                metadata = json.loads(archive.read(metadata_name))
            if not isinstance(metadata, dict):
                raise ValueError("Invalid archived report metadata")
            metadata.update(metadata_updates)
            metadata["overlay_video"] = manifest["segments"][index].get("overlay")
            metadata["lines_video"] = manifest["segments"][index].get("lines")
            replacements[metadata_name] = json.dumps(metadata, allow_nan=False).encode("utf-8")
        if report_text is not None and data.get("movement_data") is not None:
            movement = json.dumps(data["movement_data"], allow_nan=False).encode("utf-8")
            if len(movement) > 10 * 1024 * 1024:
                raise ValueError("Movement analysis exceeds the supported archive size")
            replacements[movement_member_name(index)] = movement
        replacements["manifest.json"] = json.dumps(manifest, allow_nan=False).encode("utf-8")
        if report_section is not None:
            report_name = manifest["segments"][index]["report"]
            with ZipFile(path) as archive:
                report = archive.read(report_name).decode("utf-8")
            start = "<!-- TangoPosture latest Review analysis -->\n"
            end = "<!-- End TangoPosture latest Review analysis -->\n\n"
            if report.startswith(start):
                _, marker, report = report.partition(end)
                if not marker:
                    raise ValueError("Invalid latest analysis section in archived report")
            replacements[report_name] = (
                f"{start}{report_section.rstrip()}\n{end}{report}"
            ).encode()
        _rewrite_members(path, replacements)


def update_couple_annotations(
    path: Path,
    index: int,
    *,
    run_id: str,
    annotations: dict[str, Any],
    cfg: Any,
    expected_video_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically rebuild and store edited Couple annotations for a matching run_id."""
    from .couple.measurements import rebuild_couple_result_with_annotations

    with _archive_lock(path):
        manifest = read_manifest(path)
        if not 0 <= index < len(manifest["segments"]):
            raise ValueError("Recording not found in archive")
        if expected_video_sha256 is not None:
            video = manifest["segments"][index].get("video")
            with ZipFile(path) as archive:
                if not video or _stream_member_sha256(archive, video) != expected_video_sha256:
                    raise ValueError("Recorded video changed before Couple annotations could be saved")
        previous = read_analytics(path, index)
        if previous is None:
            raise ValueError("No saved analytics found for recording")
        da = previous.get("dancer_analysis")
        if not isinstance(da, dict) or not isinstance(da.get("runs"), dict):
            raise ValueError("No saved Couple run found in archive analytics")  # noqa: TRY004
        couple_entry = da["runs"].get("couple")
        if not isinstance(couple_entry, dict) or not isinstance(couple_entry.get("couple_result"), dict):
            raise ValueError("No saved Couple analysis result found in archive analytics")  # noqa: TRY004
        stored_run_id = str(
            couple_entry.get("run_id") or couple_entry["couple_result"].get("run_id") or ""
        )
        if not run_id or stored_run_id != str(run_id):
            raise ValueError(
                f"Couple annotation run_id mismatch: expected {run_id!r}, stored {stored_run_id!r}"
            )
        couple_video_sha = str(couple_entry["couple_result"].get("video_sha256") or "")
        if (
            expected_video_sha256 is not None
            and couple_video_sha
            and couple_video_sha != expected_video_sha256
        ):
            raise ValueError("Couple run video SHA-256 mismatch")

        rebuilt = rebuild_couple_result_with_annotations(
            couple_entry["couple_result"],
            annotations,
            cfg,
        )
        new_couple_entry = {**couple_entry, "couple_result": rebuilt, "run_id": stored_run_id}
        new_runs = {**da["runs"], "couple": new_couple_entry}
        new_da = {**da, "runs": new_runs}
        updated = {**previous, "dancer_analysis": new_da}
        _validate_analytics(updated, index)
        _rewrite_members(
            path,
            {analytics_member_name(index): json.dumps(updated, allow_nan=False).encode("utf-8")},
        )
        return rebuilt


def set_active_dancer_run(path: Path, index: int, mode: str) -> dict[str, Any]:
    """Switch the active saved run mode ('single' or 'couple') for an archived segment."""
    mode_clean = str(mode).lower()
    if mode_clean not in ("single", "couple"):
        raise ValueError(f"Invalid dancer mode: {mode!r}")
    with _archive_lock(path):
        manifest = read_manifest(path)
        if not 0 <= index < len(manifest["segments"]):
            raise ValueError("Recording not found in archive")
        previous = read_analytics(path, index)
        if previous is None:
            raise ValueError("No saved analytics found for recording")
        da = previous.get("dancer_analysis")
        if not isinstance(da, dict) or not isinstance(da.get("runs"), dict):
            raise ValueError("No saved dancer analysis runs found in archive analytics")  # noqa: TRY004
        if mode_clean not in da["runs"] or not isinstance(da["runs"][mode_clean], dict):
            raise ValueError(f"No saved {mode_clean!r} run found for segment {index}")
        active_run_id = str(da["runs"][mode_clean].get("run_id") or "")
        updated = {
            **previous,
            "dancer_analysis": {
                **da,
                "active_mode": mode_clean,
                "active_run_id": active_run_id,
            },
        }
        _validate_analytics(updated, index)
        _rewrite_members(
            path,
            {analytics_member_name(index): json.dumps(updated, allow_nan=False).encode("utf-8")},
        )
        return updated


def practice_directory(data_dir: str) -> Path:
    return Path("practice") if data_dir == "data" else Path(data_dir) / "practice"


def calibration_directory(data_dir: str) -> Path:
    return Path("calibrations") if data_dir == "data" else Path(data_dir) / "calibrations"


def calibration_path(folder: Path, name: str) -> Path:
    return practice_path(folder, name.removesuffix(".tpc")).with_suffix(".tpc")


def practice_path(folder: Path, name: str) -> Path:
    name = name.strip()
    if not name or name in (".", "..") or any(char in name for char in '/\\\x00'):
        raise ValueError("Give the practice a name without path separators")
    return folder / f"{name.removesuffix('.tp')}.tp"


def read_manifest(path: Path) -> dict:
    try:
        with ZipFile(path) as archive:
            data = json.loads(archive.read("manifest.json"))
            if isinstance(data, dict) and data.get("version") == 2:
                from .calibration.archives import read_range_profile

                if path.suffix.lower() != ".tpc" or data.get("segments") != []:
                    raise ValueError("Invalid range calibration archive")
                read_range_profile(path)
                return data
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("Unsupported practice archive version")
            segments = data.get("segments")
            if not isinstance(segments, list):
                raise TypeError("Invalid practice segment list")
            if path.suffix.lower() == ".tpc" and (
                data.get("kind") != "calibration" or data.get("reference") != "reference.json"
                or "reference.json" not in archive.namelist() or len(segments) != 1
            ):
                raise ValueError("Invalid calibration archive")
            if path.suffix.lower() == ".tpr" and (
                data.get("kind") != "reference" or data.get("reference") != "reference.json"
                or "reference.json" not in archive.namelist() or len(segments) != 1
            ):
                raise ValueError("Invalid reference recording archive")
            for index, segment in enumerate(segments):
                if not isinstance(segment, dict) or segment.get("index") != index:
                    raise ValueError("Invalid practice segment index")
                for key, prefix, suffix in (
                    ("video", "video", ".mp4"), ("overlay", "overlay", ".mp4"),
                    ("report", "report", ".md"), ("metadata", "report", ".metadata.json"),
                    ("lines", "lines", ".mp4"), ("analytics", "analytics", ".json"),
                ):
                    expected = f"{prefix}_{index:02d}{suffix}"
                    if key in ("video", "overlay", "lines", "analytics") and segment.get(key) is None:
                        continue
                    if segment.get(key) != expected or expected not in archive.namelist():
                        raise ValueError(f"Missing practice segment file: {expected}")
            return data
    except (BadZipFile, KeyError, UnicodeError, TypeError) as error:
        raise ValueError(f"Invalid practice archive: {path.name}") from error


def write_review_companions(output, overlay, name, *, include_lines=True):
    for suffix in ("_lines.mp4", ".notes.jsonl"):
        if suffix == "_lines.mp4" and not include_lines:
            continue
        source = overlay.with_name(overlay.stem + suffix)
        if source.is_file():
            output.write(source, Path(name).stem + suffix, compress_type=ZIP_STORED)


def append_segment(path: Path, report: Path, video: Path | None, overlay: Path | None,
                   timestamps: list[float], *, analytics: dict[str, Any] | None = None,
                   lines: Path | None = None, metadata_extra: dict[str, Any] | None = None) -> int:
    with _archive_lock(path):
        return _append_segment(
            path, report, video, overlay, timestamps,
            analytics=analytics, lines=lines, metadata_extra=metadata_extra,
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_member_sha256(archive: ZipFile, member_name: str, destination: Path | None = None) -> str:
    digest = hashlib.sha256()
    with archive.open(member_name) as source:
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
        else:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def extract_segment_video(path: Path, index: int, destination: Path) -> str:
    with _archive_lock(path):
        manifest = read_manifest(path)
        if manifest.get("kind") == "calibration":
            raise ValueError("Calibration archives do not support movement analysis")
        segments = manifest["segments"]
        if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < len(segments)):
            raise ValueError(f"Recording {index:02d} not found in archive")
        video_member = segments[index].get("video")
        if not video_member:
            raise ValueError("Raw video required")
        with ZipFile(path) as archive:
            if video_member not in archive.namelist():
                raise ValueError("Raw video required")
            return _stream_member_sha256(archive, str(video_member), destination)


def store_movement_analysis(
    path: Path, index: int, expected_video_sha256: str, payload: dict[str, Any]
) -> str:
    member_name = movement_member_name(index)
    content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with _archive_lock(path):
        manifest = read_manifest(path)
        if manifest.get("kind") == "calibration":
            raise ValueError("Calibration archives do not support movement analysis")
        segments = manifest["segments"]
        if not (0 <= index < len(segments)):
            raise ValueError(f"Recording {index:02d} not found in archive")
        video_member = segments[index].get("video")
        if not video_member:
            raise ValueError("Raw video required")
        with ZipFile(path) as current_archive:
            if video_member not in current_archive.namelist():
                raise ValueError("Raw video required")
            actual_sha256 = _stream_member_sha256(current_archive, str(video_member))
        if actual_sha256 != expected_video_sha256:
            raise ValueError("Archived raw video changed before movement analysis could be committed")
        with NamedTemporaryFile(dir=path.parent, suffix=".tp.tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with (
                ZipFile(path) as previous,
                ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output,
            ):
                for member in previous.infolist():
                    if member.filename == member_name:
                        continue
                    with previous.open(member) as source, output.open(member, "w") as target:
                        shutil.copyfileobj(source, target)
                output.writestr(member_name, content)
            _replace_archive(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return member_name


def read_movement_analysis(path: Path, index: int) -> dict[str, Any] | None:
    member_name = movement_member_name(index)
    if path.is_dir():
        target = path / member_name
        if not target.is_file():
            return None
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"Corrupt movement analysis {member_name}: {error}") from error
        if not isinstance(data, dict):
            raise ValueError(f"Invalid movement analysis {member_name}")
        return data
    with _archive_lock(path):
        read_manifest(path)
        try:
            with ZipFile(path) as archive:
                if member_name not in archive.namelist():
                    return None
                raw_bytes = archive.read(member_name)
        except (BadZipFile, KeyError, OSError) as error:
            raise ValueError(f"Cannot read movement analysis {member_name}: {error}") from error
    if len(raw_bytes) > 10 * 1024 * 1024:
        raise ValueError(f"Movement analysis {member_name} exceeds maximum size")
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as error:
        raise ValueError(f"Corrupt movement analysis {member_name}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Invalid movement analysis {member_name}")  # noqa: TRY004
    return data


def _append_segment(path: Path, report: Path, video: Path | None, overlay: Path | None,
                    timestamps: list[float], *, analytics: dict[str, Any] | None = None,
                    lines: Path | None = None, metadata_extra: dict[str, Any] | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lines is None and overlay is not None:
        companion = overlay.with_name(overlay.stem + "_lines.mp4")
        if companion.is_file():
            lines = companion
    manifest: dict[str, Any] = read_manifest(path) if path.exists() else {
        "version": 1, "name": path.stem, "segments": [],
    }
    segments_list: list[dict[str, Any]] = manifest["segments"]
    index = len(segments_list)
    video_name = f"video_{index:02d}.mp4" if video else None
    overlay_name = f"overlay_{index:02d}.mp4" if overlay else None
    report_name = f"report_{index:02d}.md"
    metadata_name = f"report_{index:02d}.metadata.json"
    segment: dict[str, Any] = {
        "index": index, "video": video_name,
        "overlay": overlay_name,
        "report": report_name,
        "metadata": metadata_name,
    }
    if lines is not None:
        segment["lines"] = f"lines_{index:02d}.mp4"
    if analytics is not None:
        _validate_analytics({**analytics, "schema_version": 1, "segment_index": index}, index)
        segment["analytics"] = analytics_member_name(index)
    metadata = {
        **(metadata_extra or {}),
        "version": 1, "report_file": report_name,
        "raw_video": video_name, "overlay_video": overlay_name,
        "lines_video": segment.get("lines"),
        "source_video": None, "frame_timestamps": timestamps,
    }
    markdown = report.read_text(encoding="utf-8")
    for src_path, replacement_name in ((video, video_name), (overlay, overlay_name)):
        if src_path is None or replacement_name is None:
            continue
        markdown = markdown.replace(src_path.resolve().as_uri(), replacement_name)
        markdown = markdown.replace(str(src_path.resolve()), replacement_name)
        markdown = markdown.replace(str(src_path), replacement_name)
    segments_list.append(segment)
    with NamedTemporaryFile(dir=path.parent, suffix=".tp.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output:
            if path.exists():
                with ZipFile(path) as previous:
                    for member in previous.infolist():
                        if member.filename == "manifest.json":
                            continue
                        with previous.open(member) as member_src, output.open(member, "w") as target:
                            shutil.copyfileobj(member_src, target)
            if video and video_name:
                output.write(video, video_name, compress_type=ZIP_STORED)
            if overlay and overlay_name:
                output.write(overlay, overlay_name, compress_type=ZIP_STORED)
                write_review_companions(output, overlay, overlay_name, include_lines=False)
            if lines is not None:
                output.write(lines, segment["lines"], compress_type=ZIP_STORED)
            if analytics is not None:
                output.writestr(
                    analytics_member_name(index),
                    json.dumps(
                        {**analytics, "schema_version": 1, "segment_index": index},
                        allow_nan=False,
                    ),
                )
            output.writestr(report_name, markdown)
            output.writestr(metadata_name, json.dumps(metadata, allow_nan=False))
            output.writestr("manifest.json", json.dumps(manifest, allow_nan=False))
        _replace_archive(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return index


def save_calibration_archive(path: Path, reference: Calibration, video: Path,
                             overlay: Path, timestamps: list[float]) -> None:
    video_name = "video_00.mp4"
    overlay_name = "overlay_00.mp4"
    report_name = "report_00.md"
    metadata_name = "report_00.metadata.json"
    segment: dict[str, Any] = {
        "index": 0, "video": video_name, "overlay": overlay_name,
        "report": report_name, "metadata": metadata_name,
    }
    manifest = {
        "version": 1, "kind": "calibration", "name": reference.name,
        "reference": "reference.json", "segments": [segment],
    }
    metadata = {
        "version": 1, "report_file": report_name, "raw_video": video_name,
        "overlay_video": overlay_name, "source_video": None,
        "frame_timestamps": timestamps,
    }
    report = (
        f"# Calibration: {reference.name}\n\n"
        f"Created: {reference.created_at_utc}\n\n"
        "## Reference baseline\n\n| Metric | Value |\n| --- | --- |\n"
        + "".join(f"| {metric} | {value} |\n" for metric, value in reference.baseline.items())
        + "\n## Calibration reference JSON\n\n```json\n"
        + json.dumps(asdict(reference), indent=2, allow_nan=False) + "\n```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with _archive_lock(path):
        with NamedTemporaryFile(dir=path.parent, suffix=".tpc.tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output:
                output.write(video, video_name, compress_type=ZIP_STORED)
                output.write(overlay, overlay_name, compress_type=ZIP_STORED)
                write_review_companions(output, overlay, overlay_name)
                output.writestr("reference.json", json.dumps(asdict(reference), allow_nan=False))
                output.writestr(report_name, report)
                output.writestr(metadata_name, json.dumps(metadata, allow_nan=False))
                output.writestr("manifest.json", json.dumps(manifest, allow_nan=False))
            _replace_archive(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_calibration_archive(path: Path) -> Calibration:
    manifest = read_manifest(path)
    if path.suffix.lower() != ".tpc":
        raise ValueError("Practice scoring requires a .tpc calibration profile, not a reference video")
    if manifest["version"] == 2:
        from .calibration.archives import read_range_profile

        return read_range_profile(path).calibration()
    try:
        with ZipFile(path) as archive:
            return Calibration(**json.loads(archive.read("reference.json")))
    except (BadZipFile, KeyError, TypeError) as error:
        raise ValueError("Invalid calibration reference JSON") from error


def store_analysis(path: Path, analysis: Path) -> None:
    with _archive_lock(path):
        _store_analysis(path, analysis)


def _store_analysis(path: Path, analysis: Path) -> None:
    read_manifest(path)
    with NamedTemporaryFile(dir=path.parent, suffix=".tp.tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(path) as previous, ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output:
            for member in previous.infolist():
                if member.filename == analysis.name:
                    continue
                with previous.open(member) as source, output.open(member, "w") as target:
                    shutil.copyfileobj(source, target)
            output.write(analysis, analysis.name)
        _replace_archive(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PracticeArchive:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.temporary = TemporaryDirectory(prefix="tango-replay-", ignore_cleanup_errors=True)
        self.folder = Path(self.temporary.name)
        self._leases = 1
        self._lease_lock = threading.Lock()
        try:
            with _archive_lock(self.path), ZipFile(self.path) as archive:
                self.manifest = read_manifest(self.path)
                names = set()
                for member in archive.infolist():
                    name = PurePosixPath(member.filename)
                    mode = member.external_attr >> 16
                    if (
                        name.is_absolute() or ".." in name.parts or "\\" in member.filename
                        or stat.S_ISLNK(mode) or member.filename in names
                    ):
                        raise ValueError("Unsafe path in practice archive")
                    names.add(member.filename)
                    target = self.folder / member.filename
                    if not target.resolve().is_relative_to(self.folder.resolve()):
                        raise ValueError("Unsafe path in practice archive")
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, target.open("wb") as output:
                            shutil.copyfileobj(source, output)
        except Exception:
            self.close()
            raise

    def acquire(self) -> PracticeArchive:
        with self._lease_lock:
            if self._leases <= 0:
                raise RuntimeError("PracticeArchive is already closed")
            self._leases += 1
        return self

    def report(self, index: int) -> Path:
        return self.folder / self.manifest["segments"][index]["report"]

    def close(self) -> None:
        with self._lease_lock:
            if self._leases <= 0:
                return
            self._leases -= 1
            should_cleanup = self._leases == 0
        if should_cleanup:
            self.temporary.cleanup()