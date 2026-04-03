# EgoPER Baseline vs Visual-Memory Comparison Report

- Generated at: 2026-04-01T10:20:33
- Repo root: `/root/autodl-tmp/GTG-memory`
- Baseline tag: `baseline_retrain`
- Visual-memory tag: `vm_warmstart`

## Run directories

| Task | Baseline dir | Visual-memory dir |
|---|---|---|
| tea | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/tea/baseline_retrain_04_01_01_56_33` | `/root/autodl-tmp/GTG-memory/ckpts/EgoPER/tea/vm_warmstart_04_01_02_07_52` |
| oatmeal | `-` | `-` |
| pinwheels | `-` | `-` |
| quesadilla | `-` | `-` |
| coffee | `-` | `-` |

## Overall average deltas (VM - Baseline)

| Metric | Baseline Avg | VM Avg | Delta |
|---|---:|---:|---:|
| TAS F1@0.500 | 70.1 | 71.8 | +1.7 |
| TAS Edit | 75.4 | 77.7 | +2.3 |
| TAS Acc | 61.2 | 63.7 | +2.5 |
| ED F1@0.500 | 43.3 | 39.0 | -4.3 |
| Omission IoU | 38.6 | 40.9 | +2.3 |
| Omission Acc | 83.0 | 76.6 | -6.4 |
| ER w-F1@0.000 | 19.9 | 23.1 | +3.2 |
| ER w-F1@0.500 | 1.1 | 1.3 | +0.2 |
| ER EAcc@0.000 | 100.0 | 100.0 | +0.0 |
| ER EAcc@0.500 | 75.0 | 75.0 | +0.0 |

## Per-task comparison

| Task | TAS F1@0.500 (B) | TAS F1@0.500 (VM) | Δ | TAS Edit (B) | TAS Edit (VM) | Δ | TAS Acc (B) | TAS Acc (VM) | Δ | ED F1@0.500 (B) | ED F1@0.500 (VM) | Δ | Omission IoU (B) | Omission IoU (VM) | Δ | Omission Acc (B) | Omission Acc (VM) | Δ | ER w-F1@0.000 (B) | ER w-F1@0.000 (VM) | Δ | ER w-F1@0.500 (B) | ER w-F1@0.500 (VM) | Δ | ER EAcc@0.000 (B) | ER EAcc@0.000 (VM) | Δ | ER EAcc@0.500 (B) | ER EAcc@0.500 (VM) | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tea | 70.1 | 71.8 | +1.7 | 75.4 | 77.7 | +2.3 | 61.2 | 63.7 | +2.5 | 43.3 | 39.0 | -4.3 | 38.6 | 40.9 | +2.3 | 83.0 | 76.6 | -6.4 | 19.9 | 23.1 | +3.2 | 1.1 | 1.3 | +0.2 | 100.0 | 100.0 | +0.0 | 75.0 | 75.0 | +0.0 |
