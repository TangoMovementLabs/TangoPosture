"""Low-level troubleshooting telemetry logger activated via ``--troubleshoot``."""

from __future__ import annotations

import atexit
import builtins
import ctypes
import ctypes.util
import faulthandler
import gc
import importlib.metadata
import json
import logging
import os
import platform
import queue
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from .pose.constants import NAMES

_INDEX_TO_NAME = {idx: name for name, idx in NAMES.items()}
_KEY_LANDMARKS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
)
_SECRET_ENV_MARKERS = ("KEY", "SECRET", "TOKEN", "PASS", "AUTH", "CREDENTIAL", "COOKIE")

_SESSION_LOCK = threading.RLock()
_ACTIVE_SESSION: TroubleshootSession | None = None


def is_troubleshoot_enabled() -> bool:
    """Return True when a troubleshoot logging session is currently active."""
    with _SESSION_LOCK:
        return _ACTIVE_SESSION is not None and not _ACTIVE_SESSION.closed


def current_log_path() -> Path | None:
    """Return the active troubleshoot log file path, if enabled."""
    with _SESSION_LOCK:
        if _ACTIVE_SESSION is None or _ACTIVE_SESSION.closed:
            return None
        return _ACTIVE_SESSION.log_path


def log_event(subsystem: str, message: str, *, level: str = "INFO") -> None:
    """Write a structured timestamped line to the active troubleshoot log, if enabled."""
    with _SESSION_LOCK:
        session = _ACTIVE_SESSION
    if session is not None and not session.closed:
        session.write_line(level, subsystem, message)


def log_config(cfg: Any, *, source: str = "startup") -> None:
    """Record a full JSON snapshot of the active Pydantic Config object."""
    session = _get_session()
    if session is None:
        return
    try:
        payload = cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else str(cfg)
        rendered = json.dumps(payload, indent=2, sort_keys=True, default=str)
    except Exception as exc:  # noqa: BLE001
        rendered = f"<failed to serialize config: {exc!r}>"
    session.write_block("INFO", f"config.{source}", f"Effective configuration:\n{rendered}")


def log_qt_environment(app: Any) -> None:
    """Record Qt runtime, screen geometry, and QtMultimedia device telemetry."""
    session = _get_session()
    if session is None:
        return
    try:
        from PySide6 import QtCore

        screens_info = []
        for idx, scr in enumerate(app.screens()):
            geo = scr.geometry()
            screens_info.append(
                f"screen[{idx}] name={scr.name()!r} size={geo.width()}x{geo.height()} "
                f"dpr={scr.devicePixelRatio():.2f} dpi={scr.logicalDotsPerInch():.1f} "
                f"refresh={scr.refreshRate():.1f}Hz"
            )
        session.write_line(
            "INFO",
            "qt.runtime",
            f"qVersion={QtCore.qVersion()} platform={app.platformName()!r} "
            f"appName={app.applicationName()!r} screens={len(screens_info)} "
            f"[{'; '.join(screens_info)}]",
        )
    except Exception as exc:  # noqa: BLE001
        session.write_line("WARNING", "qt.runtime", f"Failed reading Qt screens: {exc!r}")

    try:
        from PySide6.QtMultimedia import QMediaDevices

        video_inputs = [
            f"id={bytes(cam.id().data()).decode('utf-8', 'replace')!r} "
            f"desc={cam.description()!r} default={cam.isDefault()}"
            for cam in QMediaDevices.videoInputs()
        ]
        audio_inputs = [
            f"id={bytes(dev.id().data()).decode('utf-8', 'replace')!r} "
            f"desc={dev.description()!r} default={dev.isDefault()}"
            for dev in QMediaDevices.audioInputs()
        ]
        session.write_line(
            "INFO",
            "qt.multimedia",
            f"videoInputs({len(video_inputs)})=[{'; '.join(video_inputs)}] "
            f"audioInputs({len(audio_inputs)})=[{'; '.join(audio_inputs)}]",
        )
    except Exception as exc:  # noqa: BLE001
        session.write_line("DEBUG", "qt.multimedia", f"QMediaDevices unavailable: {exc!r}")


def _get_session() -> TroubleshootSession | None:
    with _SESSION_LOCK:
        if _ACTIVE_SESSION is None or _ACTIVE_SESSION.closed:
            return None
        return _ACTIVE_SESSION


def _safe_version(mod: Any) -> str | None:
    mod_dict = getattr(mod, "__dict__", None)
    if not isinstance(mod_dict, dict):
        return None
    for attr in ("__version__", "VERSION", "version"):
        val = mod_dict.get(attr)
        if isinstance(val, (str, tuple)):
            return str(val)
    return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct / 100.0)))
    return ordered[idx]


def _latency_stats(samples: list[float]) -> str:
    if not samples:
        return "count=0"
    return (
        f"count={len(samples)} min={min(samples):.2f}ms "
        f"p50={statistics.median(samples):.2f}ms "
        f"p95={_percentile(samples, 95):.2f}ms max={max(samples):.2f}ms"
    )


def _mac_sysctl(*keys: str) -> dict[str, str]:
    if sys.platform != "darwin":
        return {}
    out: dict[str, str] = {}
    for key in keys:
        try:
            proc = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True,
                text=True,
                timeout=0.5,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                out[key] = proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def _mac_system_profiler() -> dict[str, Any]:
    if sys.platform != "darwin":
        return {}
    try:
        proc = subprocess.run(
            [
                "system_profiler",
                "SPCameraDataType",
                "SPAudioDataType",
                "SPDisplaysDataType",
                "-json",
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            if isinstance(data, dict):
                return data
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return {}


def _linux_proc_summary() -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        return {}
    info: dict[str, Any] = {}
    try:
        os_rel = Path("/etc/os-release")
        if os_rel.is_file():
            info["os_release"] = os_rel.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        pass
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            lines = meminfo.read_text(encoding="utf-8", errors="replace").splitlines()[:6]
            info["meminfo"] = lines
    except OSError:
        pass
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            models = [
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.lower().startswith("model name")
            ]
            if models:
                info["cpu_model"] = models[0]
    except OSError:
        pass
    return info


def _windows_ram_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:  # noqa: BLE001
        return None
    return None


def _mac_tcc_permissions() -> dict[str, Any]:
    if sys.platform != "darwin":
        return {}
    out: dict[str, Any] = {}
    labels = {0: "NotDetermined", 1: "Restricted", 2: "Denied", 3: "Authorized"}
    try:
        from .capture import _mac_camera_status

        cam = _mac_camera_status()
        out["camera_status_code"] = cam
        out["camera_status"] = labels.get(cam, "Unknown") if cam is not None else "Unavailable"
    except Exception as exc:  # noqa: BLE001
        out["camera_status_error"] = repr(exc)
    try:
        from .beat import _mac_mic_status

        mic = _mac_mic_status(request_if_needed=False)
        out["microphone_status_code"] = mic
        out["microphone_status"] = labels.get(mic, "Unknown") if mic is not None else "Unavailable"
    except Exception as exc:  # noqa: BLE001
        out["microphone_status_error"] = repr(exc)
    return out


def collect_system_specs(*, include_profiler: bool = True) -> dict[str, Any]:
    """Gather detailed hardware, OS, Python runtime, permission, and disk specs."""
    cwd = Path.cwd()
    disk = None
    with suppress(OSError):
        du = shutil.disk_usage(cwd)
        disk = {
            "total_gb": round(du.total / (1024**3), 2),
            "used_gb": round(du.used / (1024**3), 2),
            "free_gb": round(du.free / (1024**3), 2),
        }

    specs: dict[str, Any] = {
        "utc_time": datetime.now().astimezone().astimezone().isoformat(),
        "local_time": datetime.now().astimezone().isoformat(),
        "timezone": time.tzname,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "frozen": bool(getattr(sys, "frozen", False)),
        "meipass": getattr(sys, "_MEIPASS", None),
        "pid": os.getpid(),
        "ppid": os.getppid() if hasattr(os, "getppid") else None,
        "cwd": str(cwd),
        "argv": list(sys.argv),
        "sys_path": list(sys.path),
        "disk_cwd": disk,
    }

    if sys.platform == "darwin":
        sysctl_vals = _mac_sysctl(
            "hw.model",
            "machdep.cpu.brand_string",
            "hw.memsize",
            "hw.physicalcpu",
            "hw.logicalcpu",
            "kern.osproductversion",
            "kern.osversion",
        )
        mem_bytes = sysctl_vals.get("hw.memsize")
        if mem_bytes and mem_bytes.isdigit():
            sysctl_vals["ram_gb"] = f"{int(mem_bytes) / (1024**3):.2f}"
        specs["mac_sysctl"] = sysctl_vals
        specs["mac_permissions"] = _mac_tcc_permissions()
        if include_profiler:
            profiler = _mac_system_profiler()
            if profiler:
                specs["mac_hardware_devices"] = profiler
    elif sys.platform.startswith("linux"):
        specs["linux_proc"] = _linux_proc_summary()
    elif sys.platform == "win32":
        ram_bytes = _windows_ram_bytes()
        if ram_bytes is not None:
            specs["windows_ram_gb"] = round(ram_bytes / (1024**3), 2)

    sanitized_env = {
        k: v
        for k, v in sorted(os.environ.items())
        if not any(secret in k.upper() for secret in _SECRET_ENV_MARKERS)
    }
    specs["environment"] = sanitized_env
    return specs


def collect_python_packages() -> list[dict[str, str]]:
    """Return sorted metadata for every installed Python distribution."""
    items: list[dict[str, str]] = []
    for dist in importlib.metadata.distributions():
        meta: Any = dist.metadata
        raw_name = meta.get("Name", "<unknown>")
        name = str(raw_name or "<unknown>").strip()
        ver = (dist.version or "<unknown>").strip()
        location = str(getattr(dist, "_path", ""))
        items.append({"name": name, "version": ver, "location": location})
    items.sort(key=lambda item: item["name"].casefold())
    return items


def _cap_prop_lookup() -> dict[int, str]:
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return {}
    mapping: dict[int, str] = {}
    for attr in dir(cv2):
        if attr.startswith("CAP_PROP_"):
            val = getattr(cv2, attr, None)
            if isinstance(val, int) and val not in mapping:
                mapping[val] = attr
    return mapping


def _decode_fourcc(val: float) -> str:
    ival = int(val)
    if ival <= 0:
        return str(ival)
    chars = [chr((ival >> (8 * shift)) & 0xFF) for shift in range(4)]
    printable = "".join(c if 32 <= ord(c) <= 126 else "?" for c in chars)
    return f"{printable} (0x{ival:08X})"


def _frame_sample_mean(image: Any) -> float | None:
    try:
        if hasattr(image, "shape") and len(image.shape) >= 2:
            return float(image[::16, ::16].mean())
    except Exception:  # noqa: BLE001
        return None
    return None


class _TroubleshootLogHandler(logging.Handler):
    def __init__(self, session: TroubleshootSession) -> None:
        super().__init__(level=logging.DEBUG)
        self._session = session
        self._in_emit = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._in_emit, "active", False):
            return
        self._in_emit.active = True
        try:
            with suppress(Exception):
                msg = self.format(record)
                source = f"py.{record.name}:{record.funcName}:{record.lineno}"
                self._session.write_line(record.levelname, source, msg)
        finally:
            self._in_emit.active = False


class _StderrTee:
    def __init__(self, original: Any, session: TroubleshootSession) -> None:
        self._original = original
        self._session = session
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        written = self._original.write(data)
        with self._lock:
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line:
                    self._session.write_line("STDERR", "stderr", line)
        return written if isinstance(written, int) else len(data)

    def flush(self) -> None:
        with self._lock:
            if self._buf:
                self._session.write_line("STDERR", "stderr", self._buf)
                self._buf = ""
        self._original.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return bool(self._original.isatty())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _InstrumentedVideoCapture:
    """Transparent proxy logging all OpenCV VideoCapture interactions with timestamps."""

    def __init__(
        self,
        factory: Any,
        session: TroubleshootSession,
        prop_names: dict[int, str],
        cap_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._session = session
        self._prop_names = prop_names
        self._cap_id = cap_id
        self._source = args[0] if args else kwargs.get("filename", kwargs.get("index"))
        self._read_times: list[float] = []
        self._frames_ok = 0
        self._frames_fail = 0
        self._released = False
        self._t_created = time.perf_counter()

        t0 = time.perf_counter()
        self._cap = factory(*args, **kwargs)
        open_ms = (time.perf_counter() - t0) * 1000.0
        self._log_open_state(open_ms, args, kwargs)

    def _prop_label(self, prop_id: int) -> str:
        return self._prop_names.get(prop_id, f"PROP_{prop_id}")

    def _snapshot_props(self) -> str:
        import cv2

        props = (
            ("CAP_PROP_FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH),
            ("CAP_PROP_FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT),
            ("CAP_PROP_FPS", cv2.CAP_PROP_FPS),
            ("CAP_PROP_FOURCC", cv2.CAP_PROP_FOURCC),
            ("CAP_PROP_FRAME_COUNT", cv2.CAP_PROP_FRAME_COUNT),
            ("CAP_PROP_FORMAT", cv2.CAP_PROP_FORMAT),
            ("CAP_PROP_BUFFERSIZE", cv2.CAP_PROP_BUFFERSIZE),
            ("CAP_PROP_CONVERT_RGB", cv2.CAP_PROP_CONVERT_RGB),
            ("CAP_PROP_AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
            ("CAP_PROP_EXPOSURE", cv2.CAP_PROP_EXPOSURE),
        )
        parts: list[str] = []
        for label, pid in props:
            with suppress(Exception):
                val = float(self._cap.get(pid))
                if pid == cv2.CAP_PROP_FOURCC:
                    parts.append(f"{label}={_decode_fourcc(val)}")
                else:
                    parts.append(f"{label}={val:.3f}".rstrip("0").rstrip("."))
        return " ".join(parts)

    def _log_open_state(self, open_ms: float, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        opened = False
        backend = "unknown"
        with suppress(Exception):
            opened = bool(self._cap.isOpened())
        if opened:
            with suppress(Exception):
                backend = str(self._cap.getBackendName())
        props = self._snapshot_props() if opened else ""
        self._session.write_line(
            "INFO" if opened else "WARNING",
            f"camera.cv2.cap#{self._cap_id}",
            f"VideoCapture.open source={self._source!r} args={args[1:]!r} kwargs={kwargs!r} "
            f"opened={opened} backend={backend!r} open_ms={open_ms:.2f} {props}".strip(),
        )

    def open(self, *args: Any, **kwargs: Any) -> bool:
        self._source = args[0] if args else self._source
        t0 = time.perf_counter()
        ok = bool(self._cap.open(*args, **kwargs))
        open_ms = (time.perf_counter() - t0) * 1000.0
        self._log_open_state(open_ms, args, kwargs)
        return ok

    def isOpened(self) -> bool:
        return bool(self._cap.isOpened())

    def set(self, prop_id: int, value: float) -> bool:
        t0 = time.perf_counter()
        ok = bool(self._cap.set(prop_id, value))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        readback = None
        with suppress(Exception):
            readback = self._cap.get(prop_id)
        self._session.write_line(
            "DEBUG",
            f"camera.cv2.cap#{self._cap_id}",
            f"set({self._prop_label(prop_id)}, {value!r}) -> ok={ok} "
            f"readback={readback!r} elapsed_ms={dt_ms:.2f}",
        )
        return ok

    def get(self, prop_id: int) -> float:
        import cv2

        val = float(self._cap.get(prop_id))
        if prop_id not in (cv2.CAP_PROP_POS_MSEC, cv2.CAP_PROP_POS_FRAMES):
            self._session.write_line(
                "DEBUG",
                f"camera.cv2.cap#{self._cap_id}",
                f"get({self._prop_label(prop_id)}) -> {val!r}",
            )
        return val

    def read(self, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        import cv2

        t0 = time.perf_counter()
        ok, image = self._cap.read(*args, **kwargs)
        read_ms = (time.perf_counter() - t0) * 1000.0
        self._read_times.append(read_ms)
        if ok:
            self._frames_ok += 1
        else:
            self._frames_fail += 1

        shape = getattr(image, "shape", None)
        dtype = getattr(image, "dtype", None)
        mean_px = _frame_sample_mean(image) if ok else None
        pos_msec = None
        pos_frames = None
        with suppress(Exception):
            pos_msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        with suppress(Exception):
            pos_frames = float(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

        mean_txt = f" sample_mean={mean_px:.2f}" if mean_px is not None else ""
        self._session.write_line(
            "DEBUG" if ok else "WARNING",
            f"camera.cv2.cap#{self._cap_id}",
            f"read seq={self._frames_ok + self._frames_fail} ok={ok} read_ms={read_ms:.2f} "
            f"shape={shape} dtype={dtype} pos_msec={pos_msec} pos_frames={pos_frames}{mean_txt}",
        )
        return ok, image

    def grab(self) -> bool:
        t0 = time.perf_counter()
        ok = bool(self._cap.grab())
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._session.write_line(
            "DEBUG" if ok else "WARNING",
            f"camera.cv2.cap#{self._cap_id}",
            f"grab() -> ok={ok} elapsed_ms={dt_ms:.2f}",
        )
        return ok

    def retrieve(self, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        t0 = time.perf_counter()
        ok, image = self._cap.retrieve(*args, **kwargs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._session.write_line(
            "DEBUG" if ok else "WARNING",
            f"camera.cv2.cap#{self._cap_id}",
            f"retrieve() -> ok={ok} shape={getattr(image, 'shape', None)} elapsed_ms={dt_ms:.2f}",
        )
        return ok, image

    def release(self) -> None:
        if not self._released:
            self._released = True
            lifetime = time.perf_counter() - self._t_created
            self._session.write_line(
                "INFO",
                f"camera.cv2.cap#{self._cap_id}",
                f"release source={self._source!r} lifetime_s={lifetime:.3f} "
                f"frames_ok={self._frames_ok} frames_fail={self._frames_fail} "
                f"read_stats=({_latency_stats(self._read_times)})",
            )
        self._cap.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cap, name)


class _InstrumentedVideoWriter:
    """Transparent proxy logging OpenCV VideoWriter creation, writes, and close stats."""

    def __init__(
        self,
        factory: Any,
        session: TroubleshootSession,
        writer_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._session = session
        self._writer_id = writer_id
        self._filename = args[0] if args else kwargs.get("filename")
        self._write_times: list[float] = []
        self._frames = 0
        self._released = False
        self._t_created = time.perf_counter()

        t0 = time.perf_counter()
        self._writer = factory(*args, **kwargs)
        open_ms = (time.perf_counter() - t0) * 1000.0
        opened = False
        backend = "unknown"
        with suppress(Exception):
            opened = bool(self._writer.isOpened())
        if opened:
            with suppress(Exception):
                backend = str(self._writer.getBackendName())
        self._session.write_line(
            "INFO" if opened else "ERROR",
            f"video.cv2.writer#{self._writer_id}",
            f"VideoWriter.open args={args!r} kwargs={kwargs!r} opened={opened} "
            f"backend={backend!r} open_ms={open_ms:.2f}",
        )

    def isOpened(self) -> bool:
        return bool(self._writer.isOpened())

    def write(self, image: Any) -> None:
        t0 = time.perf_counter()
        self._writer.write(image)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._frames += 1
        self._write_times.append(dt_ms)
        if self._frames == 1 or self._frames % 60 == 0 or dt_ms > 50.0:
            self._session.write_line(
                "DEBUG",
                f"video.cv2.writer#{self._writer_id}",
                f"write frame={self._frames} shape={getattr(image, 'shape', None)} "
                f"write_ms={dt_ms:.2f}",
            )

    def release(self) -> None:
        self._writer.release()
        if not self._released:
            self._released = True
            size_bytes = None
            if isinstance(self._filename, (str, Path)):
                with suppress(OSError):
                    p = Path(self._filename)
                    if p.is_file():
                        size_bytes = p.stat().st_size
            self._session.write_line(
                "INFO",
                f"video.cv2.writer#{self._writer_id}",
                f"release file={self._filename!r} frames={self._frames} "
                f"size_bytes={size_bytes} lifetime_s={time.perf_counter() - self._t_created:.3f} "
                f"write_stats=({_latency_stats(self._write_times)})",
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)


def _summarize_mp_result(result: Any) -> str:
    pose_list = getattr(result, "pose_landmarks", None) or []
    world_list = getattr(result, "pose_world_landmarks", None) or []
    if not pose_list:
        return f"poses=0 world_poses={len(world_list)} NO_POSES_DETECTED"
    summaries: list[str] = []
    for idx, landmarks in enumerate(pose_list):
        vis_vals = [
            float(lm.visibility)
            for lm in landmarks
            if getattr(lm, "visibility", None) is not None
        ]
        mean_vis = sum(vis_vals) / len(vis_vals) if vis_vals else 0.0
        min_vis = min(vis_vals) if vis_vals else 0.0
        key_parts: list[str] = []
        for lm_idx, name in _INDEX_TO_NAME.items():
            if name in _KEY_LANDMARKS and lm_idx < len(landmarks):
                lm = landmarks[lm_idx]
                vis = getattr(lm, "visibility", None)
                vis_s = f"{float(vis):.2f}" if vis is not None else "na"
                key_parts.append(f"{name}=({lm.x:.3f},{lm.y:.3f},v={vis_s})")
        summaries.append(
            f"pose[{idx}](mean_v={mean_vis:.2f},min_v={min_vis:.2f},{' '.join(key_parts)})"
        )
    return f"poses={len(pose_list)} world_poses={len(world_list)} " + " ; ".join(summaries)


class _InstrumentedPoseLandmarker:
    """Transparent proxy logging every MediaPipe PoseLandmarker inference and close."""

    def __init__(
        self,
        detector: Any,
        session: TroubleshootSession,
        model_id: int,
        options_summary: str,
    ) -> None:
        self._detector = detector
        self._session = session
        self._model_id = model_id
        self._options_summary = options_summary
        self._infer_times: list[float] = []
        self._calls = 0
        self._empty_calls = 0
        self._poses_total = 0
        self._closed = False
        self._t_created = time.perf_counter()

    def detect_for_video(self, mp_image: Any, timestamp_ms: int) -> Any:
        t0 = time.perf_counter()
        try:
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._session.write_line(
                "ERROR",
                f"mediapipe.model#{self._model_id}",
                f"detect_for_video FAILED ts_ms={timestamp_ms} elapsed_ms={dt_ms:.2f} err={exc!r}",
            )
            raise
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._record_result("detect_for_video", mp_image, timestamp_ms, dt_ms, result)
        return result

    def detect(self, mp_image: Any) -> Any:
        t0 = time.perf_counter()
        try:
            result = self._detector.detect(mp_image)
        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._session.write_line(
                "ERROR",
                f"mediapipe.model#{self._model_id}",
                f"detect FAILED elapsed_ms={dt_ms:.2f} err={exc!r}",
            )
            raise
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._record_result("detect", mp_image, None, dt_ms, result)
        return result

    def _record_result(
        self,
        op: str,
        mp_image: Any,
        timestamp_ms: int | None,
        dt_ms: float,
        result: Any,
    ) -> None:
        self._calls += 1
        self._infer_times.append(dt_ms)
        pose_list = getattr(result, "pose_landmarks", None) or []
        self._poses_total += len(pose_list)
        if not pose_list:
            self._empty_calls += 1
        w = getattr(mp_image, "width", None)
        h = getattr(mp_image, "height", None)
        fmt = getattr(mp_image, "image_format", None)
        ts_part = f" ts_ms={timestamp_ms}" if timestamp_ms is not None else ""
        self._session.write_line(
            "DEBUG",
            f"mediapipe.model#{self._model_id}",
            f"{op} seq={self._calls}{ts_part} image={w}x{h} fmt={fmt} "
            f"infer_ms={dt_ms:.2f} {_summarize_mp_result(result)}",
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._session.write_line(
                "INFO",
                f"mediapipe.model#{self._model_id}",
                f"close {self._options_summary} lifetime_s={time.perf_counter() - self._t_created:.3f} "
                f"calls={self._calls} poses_total={self._poses_total} "
                f"empty_calls={self._empty_calls} infer_stats=({_latency_stats(self._infer_times)})",
            )
        self._detector.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._detector, name)


class _InstrumentedInputStream:
    """Transparent proxy logging sounddevice.InputStream without blocking audio callbacks."""

    def __init__(
        self,
        factory: Any,
        session: TroubleshootSession,
        stream_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._session = session
        self._stream_id = stream_id
        self._callbacks = 0
        self._status_errors = 0
        user_cb = kwargs.get("callback")
        if user_cb is None and len(args) >= 6:
            user_cb = args[5]

        if callable(user_cb):
            def wrapped_cb(indata: Any, frames: int, time_info: Any, status: Any) -> None:
                self._callbacks += 1
                if status:
                    self._status_errors += 1
                    self._session.enqueue_async(
                        "WARNING",
                        f"audio.sd.stream#{self._stream_id}",
                        f"callback status={status!r} cb_seq={self._callbacks} frames={frames}",
                    )
                elif self._callbacks == 1 or self._callbacks % 100 == 0:
                    rms = 0.0
                    with suppress(Exception):
                        rms = float((indata[:, 0] ** 2).mean() ** 0.5)
                    self._session.enqueue_async(
                        "DEBUG",
                        f"audio.sd.stream#{self._stream_id}",
                        f"callback seq={self._callbacks} frames={frames} rms={rms:.6f}",
                    )
                user_cb(indata, frames, time_info, status)

            if "callback" in kwargs:
                kwargs["callback"] = wrapped_cb
            elif len(args) >= 6:
                args_list = list(args)
                args_list[5] = wrapped_cb
                args = tuple(args_list)

        t0 = time.perf_counter()
        self._stream = factory(*args, **kwargs)
        open_ms = (time.perf_counter() - t0) * 1000.0
        safe_kwargs = {k: v for k, v in kwargs.items() if k != "callback"}
        self._session.write_line(
            "INFO",
            f"audio.sd.stream#{self._stream_id}",
            f"InputStream.open kwargs={safe_kwargs!r} open_ms={open_ms:.2f} "
            f"samplerate={getattr(self._stream, 'samplerate', None)} "
            f"blocksize={getattr(self._stream, 'blocksize', None)} "
            f"channels={getattr(self._stream, 'channels', None)} "
            f"latency={getattr(self._stream, 'latency', None)}",
        )

    def start(self) -> None:
        self._session.write_line("INFO", f"audio.sd.stream#{self._stream_id}", "InputStream.start()")
        self._stream.start()

    def stop(self) -> None:
        self._session.write_line(
            "INFO",
            f"audio.sd.stream#{self._stream_id}",
            f"InputStream.stop() callbacks={self._callbacks} status_errors={self._status_errors}",
        )
        self._stream.stop()

    def close(self) -> None:
        self._session.write_line(
            "INFO",
            f"audio.sd.stream#{self._stream_id}",
            f"InputStream.close() callbacks={self._callbacks} status_errors={self._status_errors}",
        )
        self._stream.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class TroubleshootSession:
    """Owns the troubleshoot log file, hooks, instrumentation proxies, and heartbeat."""

    def __init__(
        self,
        log_dir: Path,
        *,
        source: str = "cli",
        heartbeat_interval_s: float = 5.0,
        include_profiler: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pid = os.getpid()
        candidate = self.log_dir / f"troubleshoot_{stamp}_pid{pid}.log"
        suffix = 1
        while candidate.exists():
            candidate = self.log_dir / f"troubleshoot_{stamp}_pid{pid}_{suffix}.log"
            suffix += 1
        self.log_path = candidate.resolve()
        self._origin_mono = time.perf_counter()
        self._lock = threading.RLock()
        self._closed = False
        self._line_count = 0
        self._async_queue: queue.SimpleQueue[tuple[str, str, str]] = queue.SimpleQueue()

        # Counters for instrumented wrappers
        self._cap_seq = 0
        self._writer_seq = 0
        self._mp_seq = 0
        self._sd_seq = 0

        # Original hooks saved for clean teardown
        self._orig_import: Any = None
        self._import_guard = threading.local()
        self._orig_excepthook: Any = None
        self._orig_thread_excepthook: Any = None
        self._orig_stderr: Any = None
        self._log_handler: _TroubleshootLogHandler | None = None
        self._orig_root_level: int | None = None
        self._patched_libs: set[str] = set()
        self._orig_cv2_vc: Any = None
        self._orig_cv2_vw: Any = None
        self._orig_mp_create: Any = None
        self._orig_sd_input: Any = None
        self._qt_handler_installed = False

        self._fp = self.log_path.open("w", encoding="utf-8", buffering=1)
        self._heartbeat_interval_s = heartbeat_interval_s
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

        self._configure_native_env_vars()
        self._install_logging_and_crash_hooks()
        self._write_startup_banner(source=source, include_profiler=include_profiler)
        self._install_import_hook()
        self._patch_already_loaded_libraries()
        self._start_heartbeat()

    @property
    def closed(self) -> bool:
        return self._closed

    def _configure_native_env_vars(self) -> None:
        defaults = {
            "OPENCV_LOG_LEVEL": "DEBUG",
            "OPENCV_VIDEOIO_DEBUG": "1",
            "GLOG_minloglevel": "0",
            "GLOG_v": "1",
            "TF_CPP_MIN_LOG_LEVEL": "0",
            "ABSL_MIN_LOG_LEVEL": "0",
            "QT_LOGGING_RULES": "qt.multimedia*=true;qt.qpa*=true",
        }
        for key, value in defaults.items():
            os.environ.setdefault(key, value)

    def write_line(self, level: str, subsystem: str, message: str) -> None:
        if self._closed:
            return
        now_dt = datetime.now().astimezone().isoformat(timespec="microseconds")
        elapsed = time.perf_counter() - self._origin_mono
        thread = threading.current_thread()
        tid = getattr(thread, "native_id", None) or thread.ident or 0
        line = (
            f"{now_dt} | +{elapsed:10.6f}s | {level:<8s} | "
            f"{thread.name}({tid}) | {subsystem} | {message}\n"
        )
        with self._lock:
            if self._closed:
                return
            self._fp.write(line)
            self._fp.flush()
            self._line_count += 1

    def write_block(self, level: str, subsystem: str, text: str) -> None:
        for idx, part in enumerate(text.splitlines()):
            prefix = "" if idx == 0 else "  "
            self.write_line(level, subsystem, f"{prefix}{part}")

    def enqueue_async(self, level: str, subsystem: str, message: str) -> None:
        """Non-blocking telemetry enqueue for real-time audio callbacks."""
        if not self._closed:
            self._async_queue.put((level, subsystem, message))

    def _drain_async_queue(self) -> None:
        while True:
            try:
                level, subsystem, msg = self._async_queue.get_nowait()
            except queue.Empty:
                break
            self.write_line(level, subsystem, msg)

    def _install_logging_and_crash_hooks(self) -> None:
        with suppress(Exception):
            faulthandler.enable(file=self._fp, all_threads=True)

        root = logging.getLogger()
        self._orig_root_level = root.level
        root.setLevel(logging.DEBUG)
        handler = _TroubleshootLogHandler(self)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        self._log_handler = handler
        logging.captureWarnings(True)

        self._orig_excepthook = sys.excepthook

        def _excepthook(exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            tb_txt = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self.write_block("CRITICAL", "uncaught.main", tb_txt.rstrip())
            if self._orig_excepthook is not None:
                self._orig_excepthook(exc_type, exc_val, exc_tb)

        sys.excepthook = _excepthook

        self._orig_thread_excepthook = threading.excepthook

        def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
            tb_txt = "".join(
                traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
            )
            tname = args.thread.name if args.thread is not None else "unknown"
            self.write_block("CRITICAL", f"uncaught.thread.{tname}", tb_txt.rstrip())
            if self._orig_thread_excepthook is not None:
                self._orig_thread_excepthook(args)

        threading.excepthook = _thread_excepthook

        self._orig_stderr = sys.stderr
        sys.stderr = _StderrTee(self._orig_stderr, self)

    def _write_startup_banner(self, *, source: str, include_profiler: bool) -> None:
        self.write_line("INFO", "troubleshoot", "=" * 80)
        self.write_line(
            "INFO",
            "troubleshoot",
            f"TangoPosture Troubleshoot Session Started | source={source} | log={self.log_path}",
        )
        self.write_line("INFO", "troubleshoot", "=" * 80)

        specs = collect_system_specs(include_profiler=include_profiler)
        self.write_block(
            "INFO",
            "system.specs",
            json.dumps(specs, indent=2, default=str),
        )

        packages = collect_python_packages()
        self.write_line("INFO", "python.packages", f"Installed distributions count={len(packages)}")
        for pkg in packages:
            self.write_line(
                "INFO",
                "python.packages",
                f"{pkg['name']}=={pkg['version']} ({pkg['location']})",
            )

        loaded = sorted(sys.modules.keys())
        self.write_line("INFO", "python.modules.initial", f"Pre-loaded modules count={len(loaded)}")
        for mod_name in loaded:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            ver = _safe_version(mod)
            origin = getattr(mod, "__file__", None)
            if ver or origin:
                self.write_line(
                    "DEBUG",
                    "python.modules.initial",
                    f"{mod_name} version={ver} file={origin}",
                )

    def _install_import_hook(self) -> None:
        self._orig_import = builtins.__import__
        orig_import = self._orig_import
        session = self

        def _traced_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if getattr(session._import_guard, "depth", 0) > 0:
                return orig_import(name, globals, locals, fromlist, level)

            session._import_guard.depth = getattr(session._import_guard, "depth", 0) + 1
            try:
                was_present = name in sys.modules
                top_name = name.split(".", 1)[0]
                top_was_present = top_name in sys.modules
                t0 = time.perf_counter()
                mod = orig_import(name, globals, locals, fromlist, level)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if not top_was_present and top_name in sys.modules:
                    top_mod = sys.modules[top_name]
                    session.write_line(
                        "DEBUG",
                        "python.import",
                        f"loaded module={top_name} elapsed_ms={elapsed_ms:.3f} "
                        f"version={_safe_version(top_mod)} "
                        f"file={getattr(top_mod, '__file__', None)}",
                    )
                    session._maybe_instrument_module(top_name)
                elif not was_present and name in sys.modules:
                    loaded_mod = sys.modules[name]
                    if elapsed_ms >= 0.5 or name.startswith(("tango_posture", "cv2", "mediapipe", "PySide6", "sounddevice")):
                        session.write_line(
                            "DEBUG",
                            "python.import",
                            f"loaded submodule={name} elapsed_ms={elapsed_ms:.3f} "
                            f"version={_safe_version(loaded_mod)} "
                            f"file={getattr(loaded_mod, '__file__', None)}",
                        )
                    session._maybe_instrument_module(name)
                return mod
            finally:
                session._import_guard.depth -= 1

        builtins.__import__ = _traced_import

    def _patch_already_loaded_libraries(self) -> None:
        for mod_name in ("cv2", "mediapipe", "sounddevice", "PySide6.QtCore"):
            if mod_name in sys.modules:
                self._maybe_instrument_module(mod_name)

    def _maybe_instrument_module(self, mod_name: str) -> None:
        if mod_name == "cv2" or mod_name.startswith("cv2."):
            self._instrument_cv2()
        elif mod_name == "mediapipe" or mod_name.startswith("mediapipe."):
            self._instrument_mediapipe()
        elif mod_name == "sounddevice" or mod_name.startswith("sounddevice."):
            self._instrument_sounddevice()
        elif mod_name in ("PySide6", "PySide6.QtCore") or mod_name.startswith("PySide6.Qt"):
            self._instrument_qt()

    def _instrument_cv2(self) -> None:
        if "cv2" in self._patched_libs:
            return
        cv2_mod: Any = sys.modules.get("cv2")
        if cv2_mod is None:
            return
        self._patched_libs.add("cv2")

        with suppress(Exception):
            utils_logging = getattr(getattr(cv2_mod, "utils", None), "logging", None)
            if utils_logging is not None and hasattr(utils_logging, "setLogLevel"):
                verbose_lvl = getattr(utils_logging, "LOG_LEVEL_VERBOSE", 5)
                utils_logging.setLogLevel(verbose_lvl)

        self.write_line(
            "INFO",
            "cv2.init",
            f"OpenCV version={getattr(cv2_mod, '__version__', None)} file={getattr(cv2_mod, '__file__', None)}",
        )
        with suppress(Exception):
            self.write_block("DEBUG", "cv2.build_info", str(cv2_mod.getBuildInformation()))

        prop_names = _cap_prop_lookup()
        session = self

        orig_vc: Any = getattr(cv2_mod, "VideoCapture", None)
        if callable(orig_vc):
            self._orig_cv2_vc = orig_vc

            def _wrapped_vc(*args: Any, **kwargs: Any) -> _InstrumentedVideoCapture:
                with session._lock:
                    session._cap_seq += 1
                    cid = session._cap_seq
                return _InstrumentedVideoCapture(orig_vc, session, prop_names, cid, *args, **kwargs)

            cv2_mod.VideoCapture = _wrapped_vc

        orig_vw: Any = getattr(cv2_mod, "VideoWriter", None)
        if callable(orig_vw):
            self._orig_cv2_vw = orig_vw

            class _WrappedVideoWriter:
                def __new__(cls, *args: Any, **kwargs: Any) -> _InstrumentedVideoWriter:  # type: ignore[misc]
                    with session._lock:
                        session._writer_seq += 1
                        wid = session._writer_seq
                    return _InstrumentedVideoWriter(orig_vw, session, wid, *args, **kwargs)

                @staticmethod
                def fourcc(*args: Any, **kwargs: Any) -> int:
                    return int(orig_vw.fourcc(*args, **kwargs))

            cv2_mod.VideoWriter = _WrappedVideoWriter

    def _instrument_mediapipe(self) -> None:
        if "mediapipe" in self._patched_libs:
            return
        mp_mod = sys.modules.get("mediapipe")
        if mp_mod is None:
            return
        vision = getattr(getattr(mp_mod, "tasks", None), "vision", None)
        landmarker_cls = getattr(vision, "PoseLandmarker", None)
        orig_create = getattr(landmarker_cls, "create_from_options", None)
        if landmarker_cls is None or not callable(orig_create):
            return
        self._patched_libs.add("mediapipe")
        self._orig_mp_create = orig_create
        self.write_line(
            "INFO",
            "mediapipe.init",
            f"MediaPipe version={getattr(mp_mod, '__version__', None)} "
            f"file={getattr(mp_mod, '__file__', None)}",
        )
        session = self

        def _wrapped_create(options: Any) -> _InstrumentedPoseLandmarker:
            with session._lock:
                session._mp_seq += 1
                mid = session._mp_seq

            base_opts = getattr(options, "base_options", None)
            model_path_str = getattr(base_opts, "model_asset_path", None)
            model_size = None
            if model_path_str:
                with suppress(OSError):
                    p = Path(str(model_path_str))
                    if p.is_file():
                        model_size = p.stat().st_size

            options_summary = (
                f"model={model_path_str!r} size_bytes={model_size} "
                f"delegate={getattr(base_opts, 'delegate', None)!r} "
                f"running_mode={getattr(options, 'running_mode', None)!r} "
                f"num_poses={getattr(options, 'num_poses', None)} "
                f"min_det_conf={getattr(options, 'min_pose_detection_confidence', None)} "
                f"min_pres_conf={getattr(options, 'min_pose_presence_confidence', None)} "
                f"min_track_conf={getattr(options, 'min_tracking_confidence', None)} "
                f"seg_masks={getattr(options, 'output_segmentation_masks', None)}"
            )
            t0 = time.perf_counter()
            detector = orig_create(options)
            create_ms = (time.perf_counter() - t0) * 1000.0
            session.write_line(
                "INFO",
                f"mediapipe.model#{mid}",
                f"PoseLandmarker.create_from_options {options_summary} create_ms={create_ms:.2f}",
            )
            return _InstrumentedPoseLandmarker(detector, session, mid, options_summary)

        landmarker_cls.create_from_options = staticmethod(_wrapped_create)

    def _instrument_sounddevice(self) -> None:
        if "sounddevice" in self._patched_libs:
            return
        sd_mod: Any = sys.modules.get("sounddevice")
        if sd_mod is None:
            return
        self._patched_libs.add("sounddevice")

        pa_ver = None
        with suppress(Exception):
            pa_ver = sd_mod.get_portaudio_version()
        self.write_line(
            "INFO",
            "audio.sd.init",
            f"sounddevice version={getattr(sd_mod, '__version__', None)} portaudio={pa_ver!r} "
            f"default_device={getattr(getattr(sd_mod, 'default', None), 'device', None)!r}",
        )
        with suppress(Exception):
            devices = sd_mod.query_devices()
            self.write_block("DEBUG", "audio.sd.devices", str(devices))

        orig_input: Any = getattr(sd_mod, "InputStream", None)
        if callable(orig_input):
            self._orig_sd_input = orig_input
            session = self

            def _wrapped_input(*args: Any, **kwargs: Any) -> _InstrumentedInputStream:
                with session._lock:
                    session._sd_seq += 1
                    sid = session._sd_seq
                return _InstrumentedInputStream(orig_input, session, sid, *args, **kwargs)

            sd_mod.InputStream = _wrapped_input

    def _instrument_qt(self) -> None:
        if self._qt_handler_installed:
            return
        qt_core = sys.modules.get("PySide6.QtCore")
        if qt_core is None or not hasattr(qt_core, "qInstallMessageHandler"):
            return
        self._qt_handler_installed = True
        session = self

        def _qt_msg_handler(mode: Any, context: Any, message: str) -> None:
            level_map = {
                qt_core.QtMsgType.QtDebugMsg: "DEBUG",
                qt_core.QtMsgType.QtInfoMsg: "INFO",
                qt_core.QtMsgType.QtWarningMsg: "WARNING",
                qt_core.QtMsgType.QtCriticalMsg: "ERROR",
                qt_core.QtMsgType.QtFatalMsg: "CRITICAL",
            }
            lvl = level_map.get(mode, "INFO")
            cat = getattr(context, "category", "qt") or "qt"
            func = getattr(context, "function", None)
            line = getattr(context, "line", None)
            loc = f" ({func}:{line})" if func else ""
            session.write_line(lvl, f"qt.{cat}", f"{message}{loc}")

        qt_core.qInstallMessageHandler(_qt_msg_handler)
        self.write_line("INFO", "qt.init", f"Installed Qt message handler qVersion={qt_core.qVersion()}")

    def _start_heartbeat(self) -> None:
        if self._heartbeat_interval_s <= 0:
            return

        def _loop() -> None:
            while not self._stop_event.wait(self._heartbeat_interval_s):
                self._drain_async_queue()
                self._emit_heartbeat()

        self._heartbeat_thread = threading.Thread(
            target=_loop, daemon=True, name="troubleshoot-heartbeat"
        )
        self._heartbeat_thread.start()

    def _emit_heartbeat(self) -> None:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss_raw = float(ru.ru_maxrss)
        # macOS reports ru_maxrss in bytes, Linux reports KiB
        rss_mb = rss_raw / (1024 * 1024) if sys.platform == "darwin" else rss_raw / 1024.0
        times = os.times()
        threads = [t.name for t in threading.enumerate()]
        self.write_line(
            "DEBUG",
            "runtime.heartbeat",
            f"max_rss_mb={rss_mb:.2f} cpu_user_s={times.user:.2f} cpu_sys_s={times.system:.2f} "
            f"threads={len(threads)} names={threads} gc_counts={gc.get_count()}",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        self._drain_async_queue()
        total_s = time.perf_counter() - self._origin_mono
        self.write_line(
            "INFO",
            "troubleshoot",
            f"Session closing | total_duration_s={total_s:.3f} | lines={self._line_count + 1}",
        )
        with self._lock:
            self._closed = True
            self._restore_hooks()
            with suppress(Exception):
                faulthandler.disable()
            with suppress(Exception):
                self._fp.flush()
                self._fp.close()

    def _restore_hooks(self) -> None:
        if self._orig_import is not None:
            builtins.__import__ = self._orig_import
            self._orig_import = None
        if self._orig_excepthook is not None:
            sys.excepthook = self._orig_excepthook
            self._orig_excepthook = None
        if self._orig_thread_excepthook is not None:
            threading.excepthook = self._orig_thread_excepthook
            self._orig_thread_excepthook = None
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
            self._orig_stderr = None
        if self._log_handler is not None:
            root = logging.getLogger()
            root.removeHandler(self._log_handler)
            if self._orig_root_level is not None:
                root.setLevel(self._orig_root_level)
            self._log_handler = None
        cv2_mod: Any = sys.modules.get("cv2")
        if self._orig_cv2_vc is not None and cv2_mod is not None:
            cv2_mod.VideoCapture = self._orig_cv2_vc
        if self._orig_cv2_vw is not None and cv2_mod is not None:
            cv2_mod.VideoWriter = self._orig_cv2_vw
        if self._orig_mp_create is not None and "mediapipe" in sys.modules:
            mp_mod = sys.modules["mediapipe"]
            vision = getattr(getattr(mp_mod, "tasks", None), "vision", None)
            landmarker_cls = getattr(vision, "PoseLandmarker", None)
            if landmarker_cls is not None:
                landmarker_cls.create_from_options = self._orig_mp_create
        sd_mod: Any = sys.modules.get("sounddevice")
        if self._orig_sd_input is not None and sd_mod is not None:
            sd_mod.InputStream = self._orig_sd_input
        if self._qt_handler_installed and "PySide6.QtCore" in sys.modules:
            with suppress(Exception):
                sys.modules["PySide6.QtCore"].qInstallMessageHandler(None)


def enable_troubleshoot(
    log_dir: Path | str | None = None,
    *,
    source: str = "cli",
    heartbeat_interval_s: float = 5.0,
    include_profiler: bool = True,
) -> Path:
    """Activate detailed low-level troubleshoot logging into a per-run file in ``logs/``."""
    global _ACTIVE_SESSION
    with _SESSION_LOCK:
        if _ACTIVE_SESSION is not None and not _ACTIVE_SESSION.closed:
            return _ACTIVE_SESSION.log_path
        resolved_dir = Path(
            log_dir
            if log_dir is not None
            else os.environ.get("TANGOPOSTURE_LOG_DIR", "").strip() or "logs"
        )
        session = TroubleshootSession(
            resolved_dir,
            source=source,
            heartbeat_interval_s=heartbeat_interval_s,
            include_profiler=include_profiler,
        )
        _ACTIVE_SESSION = session
        return session.log_path


def shutdown_troubleshoot() -> None:
    """Flush, restore hooks, and close any active troubleshoot session."""
    global _ACTIVE_SESSION
    with _SESSION_LOCK:
        session = _ACTIVE_SESSION
        _ACTIVE_SESSION = None
    if session is not None:
        session.close()


atexit.register(shutdown_troubleshoot)
