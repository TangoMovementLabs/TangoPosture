# Synthetic sample only

`synthetic/` was generated from hand-authored landmarks by `tests/integration/test_storage_session.py`. It is not video inference, a person, a webcam benchmark, or teacher validation. It demonstrates an excursion, tracking loss, a pause, missing CSV fields, and duration denominators.

Regenerate from the repository root with (after updating the output directory in `tests/integration/test_storage_session.py` or copying from `/tmp/tango-sample-export`):

```bash
PYTHONPATH=tests uv run python tests/integration/test_storage_session.py
```

The `__main__` block writes to `examples/synthetic`.

UUIDs and UTC analysis times change; metric values and event timings reproduce within `1e-9`. The reference's model hash identifies the configured model, but no model inference produced this fixture; the session runtime metadata explicitly labels it synthetic.
