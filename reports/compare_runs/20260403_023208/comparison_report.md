# EgoPER Baseline vs Visual-Memory Comparison Report

- Generated at: 2026-04-03T02:32:08
- Repo root: `/root/autodl-tmp/GTG-memory`
- Baseline tag: `baseline_retrain`
- Visual-memory tag: `vm_warmstart`
- Tasks: `['tea', 'oatmeal', 'pinwheels', 'quesadilla', 'coffee']`

## Run directories

| Task | Baseline dir | Visual-memory dir |
|---|---|---|
| tea | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/tea/baseline_retrain_04_01_17_04_37` | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/tea/vm_warmstart_04_02_07_33_44` |
| oatmeal | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/oatmeal/baseline_retrain_04_01_17_16_36` | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/oatmeal/vm_warmstart_04_02_08_14_50` |
| pinwheels | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/pinwheels/baseline_retrain_04_01_17_39_09` | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/pinwheels/vm_warmstart_04_02_09_28_25` |
| quesadilla | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/quesadilla/baseline_retrain_04_01_17_55_56` | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/quesadilla/vm_warmstart_04_02_10_38_26` |
| coffee | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/coffee/baseline_retrain_04_01_18_01_54` | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/coffee/vm_warmstart_04_02_11_02_30` |

## Overall average deltas (VM - Baseline)

| Metric | Baseline Avg | VM Avg | Delta |
|---|---:|---:|---:|
| TAS F1@0.500 | 68.7 | 69.5 | +0.8 |
| TAS Edit | 73.5 | 73.9 | +0.4 |
| TAS Acc | 64.6 | 65.7 | +1.1 |
| ED F1@0.500 | 31.4 | 31.5 | +0.1 |
| Omission IoU | 47.1 | 49.3 | +2.2 |
| Omission Acc | 74.9 | 67.6 | -7.3 |
| ER w-F1@0.000 | 21.0 | 17.8 | -3.2 |
| ER w-F1@0.500 | 2.1 | 1.2 | -0.9 |
| ER EAcc@0.000 | 100.0 | 90.0 | -10.0 |
| ER EAcc@0.500 | 70.0 | 40.0 | -30.0 |

## Per-task comparison

| Task | TAS F1@0.500 (B) | TAS F1@0.500 (VM) | Δ | TAS Edit (B) | TAS Edit (VM) | Δ | TAS Acc (B) | TAS Acc (VM) | Δ | ED F1@0.500 (B) | ED F1@0.500 (VM) | Δ | Omission IoU (B) | Omission IoU (VM) | Δ | Omission Acc (B) | Omission Acc (VM) | Δ | ER w-F1@0.000 (B) | ER w-F1@0.000 (VM) | Δ | ER w-F1@0.500 (B) | ER w-F1@0.500 (VM) | Δ | ER EAcc@0.000 (B) | ER EAcc@0.000 (VM) | Δ | ER EAcc@0.500 (B) | ER EAcc@0.500 (VM) | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tea | 70.1 | 71.8 | +1.7 | 75.4 | 77.7 | +2.3 | 61.0 | 63.7 | +2.7 | 44.2 | 39.0 | -5.2 | 38.6 | 40.9 | +2.3 | 83.0 | 76.6 | -6.4 | 19.9 | 23.1 | +3.2 | 1.1 | 1.3 | +0.2 | 100.0 | 100.0 | +0.0 | 75.0 | 75.0 | +0.0 |
| oatmeal | 82.9 | 83.4 | +0.5 | 88.2 | 89.5 | +1.3 | 78.3 | 80.2 | +1.9 | 46.9 | 49.6 | +2.7 | 70.1 | 83.8 | +13.7 | 96.8 | 90.5 | -6.3 | 29.0 | 24.9 | -4.1 | 6.4 | 3.9 | -2.5 | 100.0 | 75.0 | -25.0 | 75.0 | 50.0 | -25.0 |
| pinwheels | 61.1 | 61.8 | +0.7 | 62.3 | 62.3 | +0.0 | 58.5 | 59.3 | +0.8 | 23.8 | 25.7 | +1.9 | 50.8 | 50.9 | +0.1 | 79.5 | 74.4 | -5.1 | 18.3 | 10.3 | -8.0 | 0.5 | 0.6 | +0.1 | 100.0 | 75.0 | -25.0 | 50.0 | 50.0 | +0.0 |
| quesadilla | 68.2 | 65.4 | -2.8 | 69.5 | 66.3 | -3.2 | 65.2 | 60.4 | -4.8 | 30.4 | 21.7 | -8.7 | 49.2 | 48.1 | -1.1 | 62.5 | 54.2 | -8.3 | 32.4 | 24.4 | -8.0 | 0.8 | 0.3 | -0.5 | 100.0 | 100.0 | +0.0 | 50.0 | 25.0 | -25.0 |
| coffee | 61.3 | 65.3 | +4.0 | 72.1 | 73.7 | +1.6 | 59.9 | 64.9 | +5.0 | 11.6 | 21.5 | +9.9 | 27.0 | 22.9 | -4.1 | 52.6 | 42.1 | -10.5 | 5.4 | 6.4 | +1.0 | 1.8 | 0.0 | -1.8 | 100.0 | 100.0 | +0.0 | 100.0 | 0.0 | -100.0 |
