# Artifact Policy

## Versioned evidence

The repository keeps only the evidence needed to reconstruct the current
experiment conclusions:

- `reports/source_logs/EgoPER/`: selected action-segmentation, error-detection,
  and error-recognition logs.
- `reports/experiments/egoper_runs.json`: fixed run IDs, config references, and
  source-log paths.
- `reports/experiments/canonical/`: generated Markdown, CSV, and JSON tables.
- `reports/compare_runs/`: older comparison snapshots retained for provenance.

## Local-only artifacts

The following are generated or machine-specific and should not be versioned:

- `ckpts/` and `exp_update/ckpts/`
- `runs/` and TensorBoard event files
- debug JSON, visualizations, and per-video prediction outputs
- dataset features and labels under `data/`

Removing these paths from Git tracking does not delete the local files. Relevant
metrics must first be copied into `reports/source_logs/` and registered in the
experiment manifest.

## Checkpoint retention

Model checkpoints are useful for local evaluation but are too large for normal
Git history. A future public release should publish selected checkpoints through
Git LFS, a GitHub release, or external artifact storage, with hashes recorded in
the manifest.
