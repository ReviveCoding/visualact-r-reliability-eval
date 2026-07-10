# Windows Filename Portability Patch (v0.5.3)

## Issue

The SFT crop materialization path sanitized `image_id` but embedded `crop_id` directly in the filename. Crop identifiers contain provenance separators such as `:`, which are not valid filename characters on Windows. This caused `tempfile.NamedTemporaryFile` to fail during SFT preparation.

## Fix

Both `image_id` and `crop_id` are now passed through a deterministic portable-component sanitizer. The sanitizer replaces Windows-reserved characters and control characters, removes trailing periods/spaces, protects device names, bounds component length with a stable digest suffix, and allows only a known image suffix set for rendered crops.

## Validation

A regression test constructs a colon-delimited crop identifier and verifies that the output path contains no Windows-reserved filename characters, remains deterministic, and that materialization succeeds.
