# Examples & Sample Data

This directory contains example configuration files and synthetic session fixtures for **TangoPostureOpenCommunity**.

## Contents

- [`default.toml`](default.toml): Complete reference configuration file containing every editable application setting and its default value.
- [`synthetic/`](synthetic/README.md): Hand-authored synthetic session export (`events.csv`, `observations.csv`, `summary.json`) demonstrating the export schema, tracking gaps, pauses, and metric denominators.

## Using `default.toml`

Run the Qt desktop studio or CLI using an explicit TOML configuration file:

```bash
uv run tango-posture --config examples/default.toml qt
```

You can also copy `examples/default.toml`, adjust parameters for your studio setup, and load it in the desktop app via **Settings → Import settings TOML**.
