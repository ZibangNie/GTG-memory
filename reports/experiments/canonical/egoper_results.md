# Canonical EgoPER Experiment Report

- Dataset: `EgoPER`
- Source snapshot: `2026-04-05`
- Manifest: `reports/experiments/egoper_runs.json`
- Values are parsed from the preserved source logs listed below.

## Five-task averages

| Variant | TAS F1@0.500 | ED F1@0.500 | Omission IoU | Omission Acc | ER w-F1@0.000 | ER w-F1@0.500 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 68.7 | 31.4 | 47.1 | 74.9 | 21.0 | 2.1 |
| Visual memory | 69.5 | 31.5 | 49.3 | 67.6 | 17.8 | 1.2 |
| Semantic memory | 68.7 | 30.2 | 44.9 | 60.4 | 19.4 | 2.1 |

## Per-task results

| Task | Variant | TAS F1@0.500 | ED F1@0.500 | Omission IoU | Omission Acc | ER w-F1@0.000 | ER w-F1@0.500 |
|---|---|---:|---:|---:|---:|---:|---:|
| tea | Baseline | 70.1 | 44.2 | 38.6 | 83.0 | 19.9 | 1.1 |
| oatmeal | Baseline | 82.9 | 46.9 | 70.1 | 96.8 | 29.0 | 6.4 |
| pinwheels | Baseline | 61.1 | 23.8 | 50.8 | 79.5 | 18.3 | 0.5 |
| quesadilla | Baseline | 68.2 | 30.4 | 49.2 | 62.5 | 32.4 | 0.8 |
| coffee | Baseline | 61.3 | 11.6 | 27.0 | 52.6 | 5.4 | 1.8 |
| tea | Visual memory | 71.8 | 39.0 | 40.9 | 76.6 | 23.1 | 1.3 |
| oatmeal | Visual memory | 83.4 | 49.6 | 83.8 | 90.5 | 24.9 | 3.9 |
| pinwheels | Visual memory | 61.8 | 25.7 | 50.9 | 74.4 | 10.3 | 0.6 |
| quesadilla | Visual memory | 65.4 | 21.7 | 48.1 | 54.2 | 24.4 | 0.3 |
| coffee | Visual memory | 65.3 | 21.5 | 22.9 | 42.1 | 6.4 | 0.0 |
| tea | Semantic memory | 72.0 | 40.8 | 41.0 | 72.3 | 24.1 | 1.9 |
| oatmeal | Semantic memory | 82.3 | 47.5 | 85.3 | 92.1 | 33.6 | 7.1 |
| pinwheels | Semantic memory | 61.1 | 26.5 | 42.8 | 63.2 | 15.5 | 0.6 |
| quesadilla | Semantic memory | 64.1 | 18.5 | 34.6 | 37.5 | 20.3 | 0.6 |
| coffee | Semantic memory | 64.2 | 17.6 | 20.6 | 36.8 | 3.7 | 0.2 |

## Partial experiments

| Task | Variant | TAS F1@0.500 | ED F1@0.500 | Omission IoU | Omission Acc | ER w-F1@0.000 | ER w-F1@0.500 |
|---|---|---:|---:|---:|---:|---:|---:|
| tea | Soft ERM v1 | 71.7 | 9.7 | 39.3 | 70.2 | 8.9 | 0.6 |

## Provenance

| Variant | Task | Run ID | Config | Source logs |
|---|---|---|---|---|
| Baseline | tea | `baseline_retrain_04_01_17_04_37` | `configs/EgoPER/tea/generated_available_only/vc_4omini_post_db0.6.available_only.baseline.train.json` | `reports/source_logs/EgoPER/baseline/tea` |
| Baseline | oatmeal | `baseline_retrain_04_01_17_16_36` | `configs/EgoPER/oatmeal/generated_available_only/vc_4omini_post_db0.6.available_only.baseline.train.json` | `reports/source_logs/EgoPER/baseline/oatmeal` |
| Baseline | pinwheels | `baseline_retrain_04_01_17_39_09` | `configs/EgoPER/pinwheels/generated_available_only/vc_4omini_post_db0.6.available_only.baseline.train.json` | `reports/source_logs/EgoPER/baseline/pinwheels` |
| Baseline | quesadilla | `baseline_retrain_04_01_17_55_56` | `configs/EgoPER/quesadilla/generated_available_only/vc_4omini_post_db0.6.available_only.baseline.train.json` | `reports/source_logs/EgoPER/baseline/quesadilla` |
| Baseline | coffee | `baseline_retrain_04_01_18_01_54` | `configs/EgoPER/coffee/generated_available_only/vc_4omini_post_db0.6.available_only.baseline.train.json` | `reports/source_logs/EgoPER/baseline/coffee` |
| Visual memory | tea | `vm_warmstart_04_02_07_33_44` | `configs/EgoPER/tea/generated_available_only/vc_4omini_post_db0.6.available_only.visual_memory.train.json` | `reports/source_logs/EgoPER/visual_memory/tea` |
| Visual memory | oatmeal | `vm_warmstart_04_02_08_14_50` | `configs/EgoPER/oatmeal/generated_available_only/vc_4omini_post_db0.6.available_only.visual_memory.train.json` | `reports/source_logs/EgoPER/visual_memory/oatmeal` |
| Visual memory | pinwheels | `vm_warmstart_04_02_09_28_25` | `configs/EgoPER/pinwheels/generated_available_only/vc_4omini_post_db0.6.available_only.visual_memory.train.json` | `reports/source_logs/EgoPER/visual_memory/pinwheels` |
| Visual memory | quesadilla | `vm_warmstart_04_02_10_38_26` | `configs/EgoPER/quesadilla/generated_available_only/vc_4omini_post_db0.6.available_only.visual_memory.train.json` | `reports/source_logs/EgoPER/visual_memory/quesadilla` |
| Visual memory | coffee | `vm_warmstart_04_02_11_02_30` | `configs/EgoPER/coffee/generated_available_only/vc_4omini_post_db0.6.available_only.visual_memory.train.json` | `reports/source_logs/EgoPER/visual_memory/coffee` |
| Semantic memory | tea | `sem_v1_avail_train5_tea_04_04_03_53_35` | `configs/EgoPER/tea/generated_available_only/vc_4omini_post_db0.6.available_only.semantic_memory.train5.json` | `reports/source_logs/EgoPER/semantic_memory/tea` |
| Semantic memory | oatmeal | `sem_v1_avail_train5_oatmeal_04_04_05_27_31` | `configs/EgoPER/oatmeal/generated_available_only/vc_4omini_post_db0.6.available_only.semantic_memory.train5.json` | `reports/source_logs/EgoPER/semantic_memory/oatmeal` |
| Semantic memory | pinwheels | `sem_v1_avail_train5_pinwheels_04_04_08_20_35` | `configs/EgoPER/pinwheels/generated_available_only/vc_4omini_post_db0.6.available_only.semantic_memory.train5.json` | `reports/source_logs/EgoPER/semantic_memory/pinwheels` |
| Semantic memory | quesadilla | `sem_v1_avail_train5_quesadilla_04_04_11_36_33` | `configs/EgoPER/quesadilla/generated_available_only/vc_4omini_post_db0.6.available_only.semantic_memory.train5.json` | `reports/source_logs/EgoPER/semantic_memory/quesadilla` |
| Semantic memory | coffee | `sem_v1_avail_train5_coffee_04_04_12_21_08` | `configs/EgoPER/coffee/generated_available_only/vc_4omini_post_db0.6.available_only.semantic_memory.train5.json` | `reports/source_logs/EgoPER/semantic_memory/coffee` |
| Soft ERM v1 | tea | `tea_availonly_ermv1_eval_0405_015142` | `configs/EgoPER/tea/generated_available_only/vc_4omini_post_db0.6.available_only.visual_semantic_memory.erm_v1.eval.json` | `reports/source_logs/EgoPER/soft_erm_v1/tea` |

## Limitations

- The preserved run directories did not contain embedded config snapshots; config paths were reconstructed from experiment naming and scripts.
- The table contains one selected run per task and variant, so it does not provide multi-seed variance or significance testing.
- Soft ERM v1 has only a tea evaluation and must not be averaged or compared as a five-task result.
