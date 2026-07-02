# GTG-memory Experiment Summary

## Sources

The current reliable experiment evidence comes from generated logs and reports already present in the repository:

- `reports/compare_runs/20260403_023208/comparison_report.md`
- `reports/compare_runs/20260403_023208/comparison_summary.csv`
- `ckpts/EgoPER/*/sem_v1_avail_train5_*/log/*.txt`
- `ckpts/EgoPER/tea/tea_ermv1_0405_011412/log/*.txt`
- `ckpts/EgoPER/tea/tea_availonly_ermv1_eval_0405_015142/log/*.txt`
- `exp_update/ckpts/EgoPER/coffee/*/log/*.txt`

## Metric meanings

- TAS F1@0.500: segment-level task/action segmentation F1 at 0.5 IoU.
- TAS Edit: normalized edit score over the predicted and ground-truth step sequences.
- TAS Acc: foreground frame accuracy for action segmentation.
- ED F1@0.500: binary error-detection segment F1 at 0.5 IoU.
- Omission IoU: intersection-over-union between predicted and ground-truth omitted step sets.
- Omission Acc: recall-like accuracy over ground-truth omitted steps.
- ER w-F1: total error-recognition F1 multiplied by EAcc.
- ER EAcc: error-type coverage ratio, not ordinary frame accuracy.

The metric implementation is in `utils/metrics.py`; ER table aggregation is in `runner.py`.

## Five-task average comparison

Tasks: EgoPER `tea`, `oatmeal`, `pinwheels`, `quesadilla`, `coffee`.

| Run | TAS F1@0.5 | ED F1@0.5 | Omission IoU | Omission Acc | ER w-F1@0 | ER w-F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 68.7 | 31.4 | 47.1 | 74.9 | 21.0 | 2.1 |
| Visual memory | 69.5 | 31.5 | 49.3 | 67.6 | 17.8 | 1.2 |
| Semantic memory available-only | 68.7 | 30.2 | 44.9 | 60.4 | 19.4 | 2.1 |

## Interpretation

Visual memory gives a small but real-looking improvement on action segmentation and omission IoU. It does not solve error-type recognition.

Semantic memory is not yet stable. It helps some local ER numbers, especially on individual tasks, but the five-task average is not better than baseline.

Soft ERM v1 is not successful in the current artifacts. The tea ERM evaluations keep TAS near visual/semantic levels but reduce ED F1@0.5 to 9.7 and ER w-F1@0 to 8.9, which is much worse than the baseline.

## Per-task highlights

| Task | Baseline TAS | VM TAS | Semantic TAS | Baseline ER w-F1@0 | VM ER w-F1@0 | Semantic ER w-F1@0 |
|---|---:|---:|---:|---:|---:|---:|
| tea | 70.1 | 71.8 | 72.0 | 19.9 | 23.1 | 24.1 |
| oatmeal | 82.9 | 83.4 | 82.3 | 29.0 | 24.9 | 33.6 |
| pinwheels | 61.1 | 61.8 | 61.1 | 18.3 | 10.3 | 15.5 |
| quesadilla | 68.2 | 65.4 | 64.1 | 32.4 | 24.4 | 20.3 |
| coffee | 61.3 | 65.3 | 64.2 | 5.4 | 6.4 | 3.7 |

## Current experiment conclusion

The cleanest defensible result is:

> Visual memory slightly improves GTG2Vid action segmentation and omission IoU, but the current semantic-memory and ERM variants do not yet provide a stable overall gain for error recognition.

The project should not claim that the full memory + ERM stack is solved. It should present visual memory as the strongest completed prototype and semantic/ERM as follow-up work or failed/partial ablations.
