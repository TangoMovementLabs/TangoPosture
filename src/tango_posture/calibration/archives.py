"""Atomic video-independent range profile archives."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

from .models import RangeProfile

EMBEDDED_EVIDENCE_MEMBER = "evidence.tpr"


def save_range_profile(
    path: Path,
    profile: RangeProfile,
    *,
    overwrite: bool = False,
    embedded_evidence: Path | None = None,
) -> None:
    from ..archive import _archive_lock, _replace_archive

    if path.suffix.lower() != ".tpc":
        raise ValueError("Calibration profiles must use the .tpc extension")
    if embedded_evidence is not None and not embedded_evidence.is_file():
        raise FileNotFoundError(f"Embedded evidence archive not found: {embedded_evidence}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _archive_lock(path):
        if path.exists() and not overwrite:
            raise FileExistsError("Choose a new profile name; existing profiles are not overwritten")
        with NamedTemporaryFile(dir=path.parent, suffix=".tpc.tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("profile.json", profile.model_dump_json(indent=2))
                archive.writestr("manifest.json", json.dumps({
                    "version": 2, "kind": "calibration", "name": profile.name,
                    "profile": "profile.json", "segments": [],
                    "created_at_utc": profile.created_at_utc,
                }))
                if embedded_evidence is not None:
                    archive.write(
                        embedded_evidence,
                        arcname=EMBEDDED_EVIDENCE_MEMBER,
                        compress_type=ZIP_STORED,
                    )
            _replace_archive(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def has_embedded_evidence(path: Path) -> bool:
    """Return True if a .tpc calibration archive embeds a raw evidence.tpr archive."""
    if path.suffix.lower() != ".tpc" or not path.is_file():
        return False
    try:
        with ZipFile(path) as archive:
            return EMBEDDED_EVIDENCE_MEMBER in archive.namelist()
    except (OSError, BadZipFile):
        return False


def extract_embedded_evidence(path: Path, dest_dir: Path) -> Path | None:
    """Extract embedded evidence.tpr from a .tpc archive into dest_dir if present."""
    if not has_embedded_evidence(path):
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / EMBEDDED_EVIDENCE_MEMBER
    with ZipFile(path) as archive:
        if EMBEDDED_EVIDENCE_MEMBER not in archive.namelist():
            return None
        with archive.open(EMBEDDED_EVIDENCE_MEMBER) as src, dest_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return dest_path


def read_range_profile(path: Path) -> RangeProfile:
    if path.suffix.lower() != ".tpc":
        raise ValueError("Calibration profiles must use the .tpc extension")
    try:
        return _read_range_profile(path)
    except (BadZipFile, KeyError) as error:
        raise ValueError("Invalid range calibration archive") from error


def _read_range_profile(path: Path) -> RangeProfile:
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != 2 or manifest.get("kind") != "calibration"
            or manifest.get("profile") != "profile.json"
            or manifest.get("segments") != []
        ):
            raise ValueError("Not a version-2 range calibration profile")
        result = RangeProfile.model_validate_json(archive.read("profile.json"))
        name = manifest.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Calibration profile name is missing")
        return result.model_copy(update={"name": name})
