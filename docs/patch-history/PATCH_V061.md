# VisualAct-R v0.6.1: Direct BDD Annotation-Filename Fix

## Fixed behavior

The direct BDD adapter now supports an already-extracted, one-file-per-image label layout where an annotation file has a stem such as:

```text
Labels/train/0000f77c-6257be58.json
```

and the paired image has one supported image suffix such as:

```text
Images/train/0000f77c-6257be58.jpg
```

When an annotation JSON omits an image `name`, the adapter resolves the annotation-file stem against the actual image index only if it maps to **exactly one** image. It stops with `HOLD` if the stem maps to zero or multiple images.

## Additional hardening

`bdd-direct-audit` now writes a structured JSON report with `status: HOLD` when it encounters a BDD adapter input error. The command returns exit code 2, but the report is not silently lost.

## Safety boundary

This patch does not create task records, modify BDD source data, download a model, run QLoRA, or start training. It only changes the direct BDD schema reader and audit error reporting.

## Local validation after extraction

Run the installer block supplied in the chat. It verifies the source provenance, runs the direct-adapter test set, then runs the read-only BDD direct audit.
