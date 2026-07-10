# v0.5.2 Pre-Execution Validation

## Patch purpose

v0.5.2 hardens the local BDD PowerShell wrapper. The prior wrapper invoked the Python `preflight` command but did not explicitly stop when the native Python process returned a nonzero exit code. PowerShell does not automatically turn every native-process failure into a terminating exception. The wrapper now checks `$LASTEXITCODE`, throws on failure, and prints `BDD_PREFLIGHT_WRAPPER=PASS` only after a successful CLI return.

## Executed checks

- Python source compilation, Ruff lint/format, and mypy targeted to Python 3.11
- 41 tests in two bounded batches
- Synthetic `fixture-run -> qualify -> report` contract
- Actual supplied BDD archive preflight: 10,000 annotations; 10,000 archived images; 100% join rate; no invalid rows or duplicate image names
- wheel and sdist build and resource-content checks
- installed wheel and installed sdist provenance, bundled-config writing, fixture, and qualification smoke checks
- static PowerShell grammar parse and source test asserting native exit propagation

## Remaining gates

This record is not a Windows or GPU run. The user must execute Phase A locally to validate Windows PowerShell engine behavior, Python 3.11 dependency resolution, NVIDIA driver/CUDA runtime, torchvision GPU operators, bitsandbytes NF4, and any later Qwen model loading.
