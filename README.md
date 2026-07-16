# GTG-memory

GTG-memory is a research prototype for procedural-video error recognition. It
extends the GTG/GTG2Vid pipeline with visual memory, semantic memory, and an
experimental soft-candidate ERM.

This repository is a research fork, not the unchanged upstream implementation.
The upstream paper and implementation are:

- [Error Recognition in Procedural Videos using Generalized Task Graph (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_Error_Recognition_in_Procedural_Videos_using_Generalized_Task_Graph_ICCV_2025_paper.pdf)

## Current Status

- The baseline, visual-memory, and semantic-memory model paths are implemented.
- The synthetic semantic-memory forward/backward smoke test passes.
- Runtime paths can be overridden without editing historical JSON configs.
- The canonical experiment table is generated from fixed, preserved source logs.
- A full local train/eval run still requires the processed EgoPER or
  CaptainCook4D data and the corresponding checkpoints.
- Visual memory is the strongest completed prototype. Semantic memory is
  unstable overall, and soft ERM v1 is an unsuccessful partial experiment.

Project orientation:

- [Project recap](docs/project_recap.md)
- [Code structure](docs/code_structure.md)
- [Experiment interpretation](docs/experiments_summary.md)
- [Canonical experiment report](reports/experiments/canonical/egoper_results.md)
- [Artifact policy](docs/artifact_policy.md)

## Environment

Use one Python interpreter consistently for installation and execution. PyTorch
and torchvision must match the CUDA/runtime available on the target machine.

```bash
python -m pip install -r requirements.txt
python scripts/check_runtime.py --code-only
python scripts/smoke_semantic_memory.py
```

On the current Windows development machine, the verified interpreter is:

```powershell
& "C:\Program Files\Python311\python.exe" scripts\check_runtime.py --code-only
```

The original `environment.yml` and the files under
`references/environment_snapshot/` are historical Linux/AutoDL environment
snapshots. They are retained for provenance, not as the preferred portable setup.

## Data Layout

Processed datasets and pre-extracted visual/semantic features are not included.
The runtime expects one dataset root per dataset family.

- EgoPER: [official repository](https://github.com/robert80203/EgoPER_official)
- CaptainCook4D: [project website](https://captaincook4d.github.io/captain-cook/)
- The original GTG2Vid README asks researchers to request its processed
  features and pretrained weights from `lee.shih@northeastern.edu`.

```text
data/
  EgoPER/
    action2idx.json
    idx2action.json
    actiontype2idx.json
    idx2actiontype.json
    tea/
      vc_v_features_10fps/
      refined_label_v3/
      vc_normal_action_features/
      vc_chatgpt4omini_error_features/
      training.txt
      validation.txt
      test.txt
      normal_actions.txt
      chatgpt4omini_error.txt
    oatmeal/
    pinwheels/
    quesadilla/
    coffee/
```

CaptainCook4D follows the same top-level mapping-file convention, with its own
task folders and label paths.

Run the full readiness check before training:

```powershell
& "C:\Program Files\Python311\python.exe" scripts\check_runtime.py `
  --config configs\EgoPER\tea\vc_4omini_post_db0.6.json `
  --data-root D:\path\to\data\EgoPER `
  --ckpt-root D:\path\to\ckpts
```

## Training And Evaluation

Historical configs still record the original AutoDL paths. Prefer CLI or
environment overrides instead of editing every config.

```bash
python main.py \
  --config configs/EgoPER/tea/vc_4omini_post_db0.6.visual_memory.train.json \
  --data-root /path/to/data/EgoPER \
  --ckpt-root /path/to/ckpts \
  --dir vm_experiment
```

```bash
python main.py \
  --config configs/EgoPER/tea/vc_4omini_post_db0.6.visual_memory.train.json \
  --data-root /path/to/data/EgoPER \
  --ckpt-root /path/to/ckpts \
  --dir vm_experiment_01_01_00_00_00 \
  --eval
```

Equivalent environment variables are `GTG_DATA_ROOT` and `GTG_CKPT_ROOT`.
Linux batch scripts also support `GTG_EGOPER_DATA_ROOT`,
`GTG_CAPTAINCOOK_DATA_ROOT`, and optional `GTG_CONDA_ENV`.

## Experiments

The fixed experiment manifest is:

```text
reports/experiments/egoper_runs.json
```

Regenerate the canonical Markdown, CSV, and JSON outputs with:

```bash
python scripts/build_experiment_report.py
```

For exploratory baseline-vs-visual-memory runs, the legacy-compatible dynamic
report remains available:

```bash
python scripts/compare_egoper_runs.py \
  --repo_root . \
  --ckpt_root ckpts \
  --baseline_tag baseline_retrain \
  --vm_tag vm_warmstart
```

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py runner.py models dp src utils scripts datasets tests
python scripts/smoke_semantic_memory.py
python scripts/build_experiment_report.py
```

## Upstream Citation

```bibtex
@InProceedings{Lee_2025_ICCV,
    author    = {Lee, Shih-Po and Elhamifar, Ehsan},
    title     = {Error Recognition in Procedural Videos using Generalized Task Graph},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {10009-10021}
}
```
