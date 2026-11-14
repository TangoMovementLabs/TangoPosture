import csv
import json
import socket
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from conftest import frame

from tango_posture.metrics import measure
from tango_posture.session import Session
from tango_posture.storage import DATABASE_TABLES, Storage, clear_database, database_counts


def run_session(root, reference, cfg):
    db = Storage(root)
    db.save_calibration(reference)
    session = Session(
        reference, cfg, db, 0, "synthetic", {"fixture": "synthetic, not real inference"}
    )
    for i in range(51):
        f = frame(i / 10, forward=0.2 if i >= 10 else 0)
        if 40 <= i < 44:
            f.poses.clear()
        session.update(measure(f, cfg))
    session.pause(5.1)
    session.resume(7)
    for t in [7.1, 7.2, 9, 9.1]:
        session.update(measure(frame(t), cfg))
    summary = session.finish(9.5)
    out = db.export(session.id, root / "exports")
    return db, session, summary, out


def test_roundtrip_accounting_determinism(tmp_path, reference, cfg, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network attempted during analysis")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    db, session, summary, out = run_session(tmp_path / "one", reference, cfg)
    other, _, again, _ = run_session(tmp_path / "two", reference, cfg)
    try:
        assert summary == again
        assert summary["active_duration"] == pytest.approx(7.6)
        assert summary["pause_duration"] == pytest.approx(1.9)
        for data in summary["metrics"].values():
            assert data["valid_duration"] + data["unknown_duration"] == pytest.approx(7.6)
            assert data["unknown_duration"] >= 2.1
        exported = json.loads((out / "summary.json").read_text())
        assert exported["summary"] == summary
        assert exported["units"]["head_forward"] == "torso_lengths"
        assert exported["reference"]["id"] == reference.id
        with (out / "observations.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(session.rows)
        assert any(r["delta"] == "" and r["state"] == "UNKNOWN" for r in rows)
        with (out / "events.csv").open() as f:
            events = list(csv.DictReader(f))
        assert len(events) == sum(m["event_count"] for m in summary["metrics"].values())
        assert all(e["end_reason"] == "tracking_lost" for e in events)
    finally:
        db.close()
        other.close()


def test_zero_valid_and_endpoints(tmp_path, reference, cfg):
    db = Storage(tmp_path)
    try:
        db.save_calibration(reference)
        s = Session(reference, cfg, db, 0, "synthetic", {})
        for t in [1, 2]:
            f = frame(t)
            f.poses.clear()
            s.update(measure(f, cfg))
        summary = s.finish(3)
        for m in summary["metrics"].values():
            assert m["unknown_duration"] == 3
            assert m["confirmed_percent_of_valid"] is None
            assert m["median_delta"] is None
    finally:
        db.close()


def test_sample_export(tmp_path, reference, cfg):
    db, _, _, _ = run_session(tmp_path, reference, cfg)
    # Re-saving an existing calibration (e.g., via --calibration latest) must succeed
    # even when referenced by existing sessions.
    db.save_calibration(reference)
    assert db.calibration(reference.id).id == reference.id
    db.close()


def test_clear_database_preserves_schema_files_and_future_sessions(tmp_path, reference, cfg):
    root = tmp_path / "data with # and ?"
    db, _, _, out = run_session(root, reference, cfg)
    try:
        schema = [tuple(row) for row in db.db.execute("SELECT * FROM sqlite_master")]
    finally:
        db.close()
    path = root / "tango.sqlite3"
    for suffix in (".tp", ".tpr", ".tpc", ".mp4"):
        (root / f"retained{suffix}").write_bytes(b"untouched fixture")
    preserved = {
        file: file.read_bytes() for file in root.rglob("*")
        if file.is_file() and file != path
    }
    counts = database_counts(path)
    assert set(counts) == set(DATABASE_TABLES)
    assert all(value > 0 for value in counts.values())

    assert clear_database(path) == counts

    assert database_counts(path) == dict.fromkeys(DATABASE_TABLES, 0)
    assert {file: file.read_bytes() for file in preserved} == preserved
    assert set(root.rglob("*")) == set(preserved) | {path, out}
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT * FROM sqlite_master").fetchall() == schema
    db, _, summary, _ = run_session(root, reference, cfg)
    db.close()
    assert summary["active_duration"] > 0
    assert database_counts(path)["sessions"] == 1


def test_clear_database_rolls_back_children_when_parent_delete_fails(tmp_path, reference, cfg):
    db, _, _, _ = run_session(tmp_path, reference, cfg)
    try:
        db.db.execute(
            "CREATE TRIGGER reject_cleanup BEFORE DELETE ON sessions "
            "BEGIN SELECT RAISE(ABORT, 'fixture prevents deletion'); END"
        )
        db.db.commit()
    finally:
        db.close()
    path = tmp_path / "tango.sqlite3"
    before = database_counts(path)

    with pytest.raises(sqlite3.IntegrityError, match="fixture prevents deletion"):
        clear_database(path)

    assert database_counts(path) == before


@pytest.mark.parametrize("operation", [database_counts, clear_database])
def test_database_maintenance_never_creates_missing_database(tmp_path, operation):
    path = tmp_path / "missing-directory" / "tango.sqlite3"
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        operation(path)
    assert not path.parent.exists()


@pytest.mark.parametrize("operation", [database_counts, clear_database])
def test_database_maintenance_rejects_unsupported_schema(tmp_path, operation):
    db = Storage(tmp_path)
    db.db.execute("PRAGMA user_version=2")
    db.close()
    path = tmp_path / "tango.sqlite3"
    before = path.read_bytes()

    with pytest.raises(ValueError, match="Unsupported database schema version 2"):
        operation(path)

    assert path.read_bytes() == before


def test_clear_database_reports_lock_without_deleting_rows(tmp_path, reference, cfg):
    db, _, _, _ = run_session(tmp_path, reference, cfg)
    path = tmp_path / "tango.sqlite3"
    before = database_counts(path)
    try:
        db.db.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            clear_database(path)
    finally:
        db.db.rollback()
        db.close()
    assert database_counts(path) == before


if __name__ == "__main__":
    from tango_posture.calibration.baseline import calibrate, profile
    from tango_posture.config import Config
    from tango_posture.models import MANIFEST

    cfg = Config()
    ref = calibrate(
        [measure(frame(i / 10), cfg) for i in range(121)],
        0,
        cfg,
        profile(cfg, 640, 480, "video"),
        MANIFEST["sha256"],
        "SYNTHETIC TEST ONLY",
    )
    db, _, _, out = run_session(Path("/tmp/tango-sample-export"), ref, cfg)
    import shutil

    shutil.copytree(out, Path("samples/synthetic"), dirs_exist_ok=True)
    db.close()
