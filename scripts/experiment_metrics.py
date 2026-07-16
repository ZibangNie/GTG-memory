from __future__ import annotations

import re
from pathlib import Path


METRIC_SPECS = [
    ("tas_f1_050", "TAS F1@0.500"),
    ("tas_edit", "TAS Edit"),
    ("tas_acc", "TAS Acc"),
    ("ed_f1_050", "ED F1@0.500"),
    ("omit_oiou", "Omission IoU"),
    ("omit_oacc", "Omission Acc"),
    ("er_wf1_000", "ER w-F1@0.000"),
    ("er_wf1_050", "ER w-F1@0.500"),
    ("er_eacc_000", "ER EAcc@0.000"),
    ("er_eacc_050", "ER EAcc@0.500"),
]

PRIMARY_METRICS = [
    "tas_f1_050",
    "ed_f1_050",
    "omit_oiou",
    "omit_oacc",
    "er_wf1_000",
    "er_wf1_050",
]

METRIC_LABELS = dict(METRIC_SPECS)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def parse_first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_table_value_pair(text: str, header_key: str) -> tuple[float | None, float | None]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if header_key not in line:
            continue
        for candidate in lines[i + 1 : i + 5]:
            candidate = candidate.strip()
            if not candidate.startswith("|"):
                continue
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", candidate)
            if len(nums) >= 5:
                return float(nums[3]), float(nums[4])
    return None, None


def parse_action_seg(path: Path) -> dict[str, float | None]:
    text = read_text(path)
    return {
        "tas_f1_050": parse_first_float(r"Avg F1@0\.500:\s*([0-9.]+)", text),
        "tas_edit": parse_first_float(r"\|Edit:([0-9.]+)\|Acc:[0-9.]+\|", text),
        "tas_acc": parse_first_float(r"\|Edit:[0-9.]+\|Acc:([0-9.]+)\|", text),
    }


def parse_error_detection(path: Path) -> dict[str, float | None]:
    text = read_text(path)
    return {
        "ed_f1_050": parse_first_float(r"Avg F1@0\.500:\s*([0-9.]+)", text),
        "omit_oiou": parse_first_float(r"\|oIoU:([0-9.]+)\|oAcc:[0-9.]+\|", text),
        "omit_oacc": parse_first_float(r"\|oIoU:[0-9.]+\|oAcc:([0-9.]+)\|", text),
    }


def parse_error_recognition(path: Path) -> dict[str, float | None]:
    text = read_text(path)
    wf1_000, eacc_000 = parse_table_value_pair(text, "All w-F1@0.000")
    wf1_050, eacc_050 = parse_table_value_pair(text, "All w-F1@0.500")
    return {
        "er_wf1_000": wf1_000,
        "er_wf1_050": wf1_050,
        "er_eacc_000": eacc_000,
        "er_eacc_050": eacc_050,
    }


def parse_log_dir(log_dir: Path) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    metrics.update(parse_action_seg(log_dir / "action_segmentation.txt"))
    metrics.update(parse_error_detection(log_dir / "error_detection.txt"))
    metrics.update(parse_error_recognition(log_dir / "error_recognition.txt"))
    return metrics


def parse_run_dir(run_dir: Path) -> dict[str, float | None]:
    return parse_log_dir(run_dir / "log")


def validate_metrics(metrics: dict[str, float | None], source: Path) -> None:
    missing = [key for key, _ in METRIC_SPECS if metrics.get(key) is None]
    if missing:
        raise ValueError(f"Missing parsed metrics {missing} in {source}")
