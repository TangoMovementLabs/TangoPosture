# Contributing to TangoPostureOpenCommunity

Thank you for your interest in contributing to **TangoPostureOpenCommunity**! Whether you are fixing a bug, improving posture metrics, refining the Qt desktop UI, expanding documentation, or sharing Argentine Tango domain insights, your help is welcome.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

---

## Ways to Contribute

- **Report bugs & hardware/camera issues:** Open an issue with OS, Python/installer version, steps to reproduce, and (when relevant) `--troubleshoot` logs with any sensitive local paths redacted.
- **Propose features or metric improvements:** Describe the use case for dancers, teachers, or researchers, expected behaviour, and any biomechanics or tracking assumptions.
- **Submit code, tests, or documentation improvements:** Open a pull request against the default branch.

---

## Development Setup

TangoPostureOpenCommunity uses **Python 3.11 (`>=3.11,<3.12`)** and [`uv`](https://github.com/astral-sh/uv) with a pinned `uv.lock`.

```bash
git clone https://github.com/TangoMovementLabs/TangoPostureOpenCommunity.git
cd TangoPostureOpenCommunity

uv python install 3.11
uv sync --frozen
uv run tango-posture models fetch
```

On Linux, install the system desktop libraries listed in [assets/docs/INSTALLATION_GUIDE.md](assets/docs/INSTALLATION_GUIDE.md) (OpenGL, XCB, and PortAudio) before running Qt or audio tests.

---

## Quality Gates

All pull requests are checked by GitHub Actions across macOS, Windows, and Linux. Run the full suite locally before opening a PR:

```bash
uv run --frozen ruff check src tests
uv run --frozen mypy src
QT_QPA_PLATFORM=offscreen uv run --frozen pytest -q
uv build
```

### Project Conventions

- **Formatting & linting:** Ruff with `line-length = 100` (configured in `pyproject.toml`).
- **Typing:** Strict `mypy src` (`python_version = "3.11"`, `check_untyped_defs = true`).
- **Settings & manual sync:** `tests/unit/test_settings.py` and `tests/unit/test_tutorial.py` enforce that every setting in `SETTING_MAP` and every metric in `METRIC_DETAILS` is documented with a worked example in [assets/docs/TANGO_POSTURE_GUIDE.md](assets/docs/TANGO_POSTURE_GUIDE.md) and included in [examples/default.toml](examples/default.toml).
- **Model manifest sync:** `tests/unit/test_cli.py` verifies that [assets/model_manifest.json](assets/model_manifest.json) matches `src/tango_posture/model_manifest.json`. Update both if changing pinned model weights.
- **No silent approximations:** Keep unknown/untracked frames explicit rather than fabricating scores, and preserve existing disclaimers that TangoPosture is experimental movement observation, not medical assessment.

---

## Pull Request Process

1. Fork the repository and create a topic branch from `main`.
2. Keep changes focused and well-tested (add unit, integration, or headless Qt tests under `tests/`).
3. Ensure `ruff`, `mypy`, and `pytest` pass cleanly.
4. Update [CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]` for user-visible changes when appropriate.
5. Open a pull request describing *what* changed, *why*, and *how* it was verified.

---

## Licensing

By contributing to TangoPostureOpenCommunity, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
