# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| `0.5.x` | Yes |
| `< 0.5.0` | No |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Instead, report vulnerabilities privately through [GitHub Security Advisories](https://github.com/TangoMovementLabs/TangoPostureOpenCommunity/security/advisories/new).

Please include as much of the following information as possible to help maintainers reproduce and resolve the issue quickly:

- A clear description of the vulnerability and affected component (e.g., archive handling `.tp`/`.tpr`/`.tpc`, TOML config loading, model verification, downloaders, or desktop packaging).
- Steps to reproduce or a minimal proof of concept.
- Affected version(s) and operating system.
- Potential impact and any suggested remediation.

### Response Timeline

- **Acknowledgment:** Within 3 business days of receiving the report.
- **Initial assessment:** Within 7 business days, including severity triage and next steps.
- **Disclosure:** We coordinate a fix and public advisory once a patched release or mitigation is ready.

---

## Security & Privacy Practices in TangoPostureOpenCommunity

- **Local execution:** Live camera practice, calibration, and offline video analysis run locally without accounts or cloud telemetry.
- **Model integrity:** MediaPipe `.task` weights are verified against pinned SHA-256 checksums recorded in [assets/model_manifest.json](assets/model_manifest.json) and `src/tango_posture/model_manifest.json` before use.
- **Release verification:** Pre-built desktop installers are currently unsigned; always verify downloaded artifacts against the release's `SHA256SUMS.txt` before running them.
- **Optional external tools:** Optional Ollama analysis sends only the selected report text to the user-configured `ollama_base_url` (default `http://127.0.0.1:11434`) on explicit user action.
