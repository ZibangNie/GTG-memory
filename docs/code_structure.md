# GTG-memory Code Structure

This document describes the cleaned logical structure of the repository. It is meant as a reading guide, not a design claim that every module is finished.

## Runtime Entry Points

- `main.py`
  - Parses command-line flags.
  - Supports portable data/checkpoint root overrides.
  - Delays heavy imports so `--help` works before the runtime environment is loaded.
  - Creates `Runner`.
  - Calls `train()` or `evaluate()`.

- `runner.py`
  - Owns the end-to-end training/evaluation loop.
  - Loads config, datasets, task graph, prototypes, and checkpoints.
  - Calls the model, GTG2Vid dynamic programming, ERM, metric writers, and optional debug dumps.

`runner.py` is still the largest file and should be the next major refactor target. The cleanup branch fixed debug-config wiring and moved debug JSON serialization out of the core train/eval loop without changing the modeling path.

- `utils/runtime_config.py`
  - Loads JSON configs with UTF-8 handling.
  - Applies CLI or environment path overrides.
  - Resolves relative checkpoint references against the selected checkpoint root.

- `scripts/check_runtime.py`
  - Checks the active interpreter and runtime dependencies.
  - Validates config, dataset, split, feature, prototype, and checkpoint paths.
  - Supports `--code-only` when processed datasets are unavailable.

## Model Layer

- `models/models.py`
  - Defines `ASDiffusionBackbone`.
  - Keeps the upstream GTG feature trunk.
  - Selects baseline, visual-memory, or visual-semantic-memory scorer based on config flags.

- `models/visual_memory.py`
  - Visual memory scorer.
  - Implements base projection, short memory, slow long memory, fusion, and final logits.

- `models/semantic_memory.py`
  - Semantic prototype bank.
  - Semantic observation builder.
  - Semantic memory core with coverage and uncertainty traces.

- `models/fusion_heads.py`
  - Visual-semantic fusion scorer.
  - Combines visual branch, semantic branch, asymmetric gate, and prototype boost head.

- `models/memory_utils.py`
  - Small tensor utilities shared by visual/semantic memory modules.

## ERM Layer

- `src/erm/soft_erm.py`
  - Experimental soft multi-candidate ERM.
  - Uses anchor step, semantic candidates, graph neighbors, coverage, and prototype similarity.

Current status: implemented but not successful in existing logs. Treat it as an unfinished experiment.

## Data And Graph Layer

- `datasets/gtg_dataset_loader.py`
  - Loads per-video features and frame labels.

- `datasets/loader_graph.py`
  - Hard-coded EgoPER and CaptainCook4D task graphs.

- `dp/graph_utils.py`
  - Generalized metagraph dynamic programming and cost construction.

- `utils/semantic_prototype_loader.py`
  - Loads normal and error semantic prototypes from task folders.

- `utils/debug_dump.py`
  - Serializes per-video model and ERM debug traces during evaluation.
  - Keeps bulky JSON payload assembly out of `runner.py`.

- `utils/metrics.py`
  - TAS, ED, ER, and omission metrics.

## Experiment Scripts

- `scripts/egoper_utils.py`
  - Shared script utilities introduced by the cleanup branch.
  - Centralizes task list, JSON IO, base-config selection, dataset naming metadata, and baseline/visual-memory config generation.

- `scripts/probe_available_egoper_tasks.py`
  - Checks which original EgoPER task splits are fully loadable.

- `scripts/build_available_only_egoper_splits_and_configs.py`
  - Rebuilds split files using only videos available in the local data installation.
  - Generates available-only baseline and visual-memory configs.

- `scripts/gen_all_egoper_task_configs.py`
  - Generates baseline and visual-memory configs from a ready-task JSON.

- `scripts/compare_egoper_runs.py`
  - Parses existing run logs and writes Markdown/CSV/JSON comparisons.

- `scripts/experiment_metrics.py`
  - Shared parser for TAS, ED, omission, and ER log files.

- `scripts/build_experiment_report.py`
  - Reads fixed run provenance from `reports/experiments/egoper_runs.json`.
  - Regenerates the canonical Markdown, CSV, and JSON result tables.

- `scripts/run_available_egoper_pipeline.sh`
  - Linux batch pipeline for the available-only experiment flow.
  - Uses environment or CLI path overrides instead of fixed AutoDL paths.

## Known Structural Debt

- `runner.py` mixes configuration, IO, model construction, DP, ERM, metrics, TensorBoard logging, and debug dumping.
- Historical configs still contain AutoDL paths, but runtime and generation scripts
  now accept path overrides.
- Historical large artifacts remain in Git history; selected logs are retained
  under `reports/source_logs/` for reproducible metric extraction.
- The canonical report has one run per task and no seed variance.
- The archived `archive/legacy/dp/soft_dp.py` is research history, not a supported runtime module.

## Suggested Next Refactor

1. Extract runner configuration into a small config object.
2. Move model/prototype construction into a factory module.
3. Move metric parsing/reporting into a `reports` utility module.
4. Split evaluation-time output writers from metric computation.
5. Only after that, touch the train/evaluate loops.
