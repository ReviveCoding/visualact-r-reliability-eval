# Windows Filename Portability Fix Validation

## Root cause

`src/visualact_r/agents/sft.py` sanitized `image_id` but concatenated `crop_id` into the output filename without sanitization. SFT crop IDs include colon-delimited provenance keys. Windows rejects `:` in file-name components, so `tempfile.NamedTemporaryFile` failed before it could write the crop.

## Code change

- Sanitizes image IDs, crop IDs, and candidate IDs with a deterministic portable filename-component function.
- Replaces Windows-reserved characters and control characters, removes trailing dots/spaces, protects device names, and bounds each component with a stable digest suffix.
- Restricts derived crop suffixes to `.bmp`, `.jpeg`, `.jpg`, `.png`, or `.webp`; falls back to `.png`.
- Adds regression tests using a colon-delimited crop ID matching the failing Windows pattern.

## Validation completed

- `python -m compileall -q src`: PASS
- `python -m ruff check .`: PASS
- `python -m ruff format --check .`: PASS
- `python -m mypy --python-version 3.11 --no-site-packages src`: PASS (35 source files)
- `pytest -q tests/test_sft.py tests/test_versioning.py`: PASS (4 tests)
- Wheel and sdist build: PASS

The current Windows-local Phase A run is the final host-specific confirmation because it executes the full suite with the actual Windows filesystem and actual RTX 4090 environment.
