# Changelog

All notable changes to **TangoPostureOpenCommunity** (and its upstream `TangoPosture` releases) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initialized `TangoPostureOpenCommunity` repository structure (`src/`, `tests/`, `examples/`, `assets/`, `scripts/`, `.github/workflows/`).
- Added community and governance documentation: `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `CHANGELOG.md`.
- Updated internal asset, configuration, documentation, packaging, CI workflow, and repository links for `TangoMovementLabs/TangoPostureOpenCommunity`.

## [0.5.3] - 2026-11-24

### Added
- Practice session audio asset (`Regreso a la Práctica.mp3`).
- Enhanced camera discovery and UI integration in startup splash and practice screens.

### Changed
- Refactored code structure for improved readability and maintainability.
- Updated guide paths across packaging, workflows, screens, and tests.

## [0.5.2] - 2026-11-24

### Added
- MediaPipe landmark index map in the technical guide.
- Qt UI telemetry in `--troubleshoot` diagnostic logging and data-trust improvements.

### Fixed
- Added missing `PoseOrigin` import and type hint for `new_origins` in `interpolate_poses`.

## [0.5.1] - 2026-11-22

### Added
- Embedded evidence in calibration archives (`.tpc`).
- Installation, system requirements, and camera compatibility guide (`INSTALLATION_GUIDE.md`), plus improved Linux desktop icon and bundle support.
- Enhanced dancer analysis configuration and couple measurements.

### Changed
- Refactored normalization panel to handle missing metric details gracefully and improved camera selection in the practice screen.

### Fixed
- Cross-platform path checks in speech manager tests and Windows path handling in sample export/database tests.

## [0.5.0] - 2026-11-20

### Added
- Dual-engine architecture separating solo and couple analysis (`engines/solo.py`, `engines/couple.py`).
- Couple tracking, classification, CSV export, measurements, and calibration workflows.
- Solo guided camera calibration, measurement normalization, and calibration editor UI.
- Video analysis screen, dancer mode support, soundtrack beat extraction, and pose interpolation/recovery.

## [0.2.3] - 2026-11-16

### Added
- Desktop installers, packaged smoke testing, and cross-platform CI release workflow.

## [0.2.0] - 2026-11-14

### Added
- PySide6 Qt desktop interface replacing the legacy Tk UI.
- Portable `.tp`, `.tpr`, and `.tpc` ZIP archive workflows, Review playback, Analytics, AI Coach (Ollama), and offline manual viewer.

## [0.1.0] - 2026-11-10

### Added
- Initial CLI, MediaPipe pose pipeline, 8-beat tango phrase tracker, calibration baselines, and SQLite session storage.

## [0.0.1] - 2026-11-01

### Added
- Initial project scaffold.
