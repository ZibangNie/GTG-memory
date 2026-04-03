<<<<<<< Updated upstream
# 文件：scripts/compare_egoper_runs.py
# 作用：
# 只比较 ready_tasks 里的任务，并生成 Markdown / CSV / JSON 报告

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

DEFAULT_TASKS = ["tea", "oatmeal", "pinwheels", "quesadilla", "coffee"]

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

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")

def parse_first_float(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None

def parse_table_value_pair(text: str, header_key: str):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if header_key in line:
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j].strip()
                if candidate.startswith("|"):
                    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", candidate)
                    if len(nums) >= 5:
                        return float(nums[3]), float(nums[4])
    return None, None

def parse_action_seg(path: Path):
    text = read_text(path)
    return {
        "tas_f1_050": parse_first_float(r"Avg F1@0\.500:\s*([0-9.]+)", text),
        "tas_edit": parse_first_float(r"\|Edit:([0-9.]+)\|Acc:[0-9.]+\|", text),
        "tas_acc": parse_first_float(r"\|Edit:[0-9.]+\|Acc:([0-9.]+)\|", text),
    }

def parse_error_detection(path: Path):
    text = read_text(path)
    return {
        "ed_f1_050": parse_first_float(r"Avg F1@0\.500:\s*([0-9.]+)", text),
        "omit_oiou": parse_first_float(r"\|oIoU:([0-9.]+)\|oAcc:[0-9.]+\|", text),
        "omit_oacc": parse_first_float(r"\|oIoU:[0-9.]+\|oAcc:([0-9.]+)\|", text),
    }

def parse_error_recognition(path: Path):
    text = read_text(path)
    wf1_000, eacc_000 = parse_table_value_pair(text, "All w-F1@0.000")
    wf1_050, eacc_050 = parse_table_value_pair(text, "All w-F1@0.500")
    return {
        "er_wf1_000": wf1_000,
        "er_wf1_050": wf1_050,
        "er_eacc_000": eacc_000,
        "er_eacc_050": eacc_050,
    }

def parse_run_dir(run_dir: Path):
    log_dir = run_dir / "log"
    out = {}
    out.update(parse_action_seg(log_dir / "action_segmentation.txt"))
    out.update(parse_error_detection(log_dir / "error_detection.txt"))
    out.update(parse_error_recognition(log_dir / "error_recognition.txt"))
    return out

def latest_run_dir(repo_root: Path, task: str, tag: str):
    task_root = repo_root / "ckpts" / "EgoPER" / task
    candidates = [p for p in task_root.glob(f"{tag}_*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def fmt(v):
    return "-" if v is None else f"{v:.1f}"

def delta(v1, v0):
    if v1 is None or v0 is None:
        return None
    return v1 - v0

def fmt_delta(v):
    if v is None:
        return "-"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}"

def build_task_row(task, base_metrics, vm_metrics):
    row = {"task": task}
    for k, _ in METRIC_SPECS:
        row[f"baseline_{k}"] = base_metrics.get(k)
        row[f"vm_{k}"] = vm_metrics.get(k)
        row[f"delta_{k}"] = delta(vm_metrics.get(k), base_metrics.get(k))
    return row

def avg_metric(rows, prefix_key):
    vals = [r[prefix_key] for r in rows if r[prefix_key] is not None]
    return mean(vals) if vals else None

def write_csv(path: Path, rows):
    fieldnames = ["task"]
    for k, _ in METRIC_SPECS:
        fieldnames += [f"baseline_{k}", f"vm_{k}", f"delta_{k}"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def write_markdown(path: Path, rows, base_dirs, vm_dirs, summary, tasks):
    lines = []
    lines.append("# EgoPER Baseline vs Visual-Memory Comparison Report\n\n")
    lines.append(f"- Generated at: {summary['generated_at']}\n")
    lines.append(f"- Repo root: `{summary['repo_root']}`\n")
    lines.append(f"- Baseline tag: `{summary['baseline_tag']}`\n")
    lines.append(f"- Visual-memory tag: `{summary['vm_tag']}`\n")
    lines.append(f"- Tasks: `{tasks}`\n\n")

    lines.append("## Run directories\n\n")
    lines.append("| Task | Baseline dir | Visual-memory dir |\n")
    lines.append("|---|---|---|\n")
    for task in tasks:
        lines.append(f"| {task} | `{base_dirs.get(task, '-')}` | `{vm_dirs.get(task, '-')}` |\n")
    lines.append("\n")

    lines.append("## Overall average deltas (VM - Baseline)\n\n")
    lines.append("| Metric | Baseline Avg | VM Avg | Delta |\n")
    lines.append("|---|---:|---:|---:|\n")
    for k, label in METRIC_SPECS:
        base_avg = summary["avg"].get(f"baseline_{k}")
        vm_avg = summary["avg"].get(f"vm_{k}")
        d_avg = summary["avg"].get(f"delta_{k}")
        lines.append(f"| {label} | {fmt(base_avg)} | {fmt(vm_avg)} | {fmt_delta(d_avg)} |\n")
    lines.append("\n")

    lines.append("## Per-task comparison\n\n")
    header = ["Task"]
    for _, label in METRIC_SPECS:
        header.extend([f"{label} (B)", f"{label} (VM)", "Δ"])
    lines.append("| " + " | ".join(header) + " |\n")
    lines.append("|" + "|".join(["---"] * len(header)) + "|\n")

    for row in rows:
        cells = [row["task"]]
        for k, _ in METRIC_SPECS:
            cells.extend([
                fmt(row[f"baseline_{k}"]),
                fmt(row[f"vm_{k}"]),
                fmt_delta(row[f"delta_{k}"]),
            ])
        lines.append("| " + " | ".join(cells) + " |\n")

    path.write_text("".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default="/root/autodl-tmp/GTG-memory")
    parser.add_argument("--baseline_tag", type=str, default="baseline_retrain")
    parser.add_argument("--vm_tag", type=str, default="vm_warmstart")
    parser.add_argument("--task_list_json", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.task_list_json:
        tasks = load_json(Path(args.task_list_json))["ready_tasks"]
    else:
        tasks = DEFAULT_TASKS

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = repo_root / "reports" / "compare_runs" / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    base_dirs = {}
    vm_dirs = {}

    for task in tasks:
        base_dir = latest_run_dir(repo_root, task, args.baseline_tag)
        vm_dir = latest_run_dir(repo_root, task, args.vm_tag)

        base_dirs[task] = str(base_dir) if base_dir else "-"
        vm_dirs[task] = str(vm_dir) if vm_dir else "-"

        if base_dir is None or vm_dir is None:
            print(f"[SKIP] task={task}, missing baseline or vm run")
            continue

        base_metrics = parse_run_dir(base_dir)
        vm_metrics = parse_run_dir(vm_dir)
        rows.append(build_task_row(task, base_metrics, vm_metrics))

    avg_summary = {}
    for k, _ in METRIC_SPECS:
        avg_summary[f"baseline_{k}"] = avg_metric(rows, f"baseline_{k}")
        avg_summary[f"vm_{k}"] = avg_metric(rows, f"vm_{k}")
        avg_summary[f"delta_{k}"] = avg_metric(rows, f"delta_{k}")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "baseline_tag": args.baseline_tag,
        "vm_tag": args.vm_tag,
        "tasks": tasks,
        "avg": avg_summary,
        "rows": rows,
        "baseline_dirs": base_dirs,
        "vm_dirs": vm_dirs,
    }

    md_path = output_root / "comparison_report.md"
    csv_path = output_root / "comparison_summary.csv"
    json_path = output_root / "comparison_summary.json"

    write_markdown(md_path, rows, base_dirs, vm_dirs, summary, tasks)
    write_csv(csv_path, rows)
    write_json(json_path, summary)

    print(f"[WRITE] {md_path}")
    print(f"[WRITE] {csv_path}")
    print(f"[WRITE] {json_path}")

if __name__ == "__main__":
=======
# 文件：scripts/compare_egoper_runs.py
# 作用：
# 只比较 ready_tasks 里的任务，并生成 Markdown / CSV / JSON 报告

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

DEFAULT_TASKS = ["tea", "oatmeal", "pinwheels", "quesadilla", "coffee"]

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

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")

def parse_first_float(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.MULTILINE)
    return float(m.group(1)) if m else None

def parse_table_value_pair(text: str, header_key: str):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if header_key in line:
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j].strip()
                if candidate.startswith("|"):
                    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", candidate)
                    if len(nums) >= 5:
                        return float(nums[3]), float(nums[4])
    return None, None

def parse_action_seg(path: Path):
    text = read_text(path)
    return {
        "tas_f1_050": parse_first_float(r"Avg F1@0\.500:\s*([0-9.]+)", text),
        "tas_edit": parse_first_float(r"\|Edit:([0-9.]+)\|Acc:[0-9.]+\|", text),
        "tas_acc": parse_first_float(r"\|Edit:[0-9.]+\|Acc:([0-9.]+)\|", text),
    }

def parse_error_detection(path: Path):
    text = read_text(path)
    return {
        "ed_f1_050": parse_first_float(r"Avg F1@0\.500:\s*([0-9.]+)", text),
        "omit_oiou": parse_first_float(r"\|oIoU:([0-9.]+)\|oAcc:[0-9.]+\|", text),
        "omit_oacc": parse_first_float(r"\|oIoU:[0-9.]+\|oAcc:([0-9.]+)\|", text),
    }

def parse_error_recognition(path: Path):
    text = read_text(path)
    wf1_000, eacc_000 = parse_table_value_pair(text, "All w-F1@0.000")
    wf1_050, eacc_050 = parse_table_value_pair(text, "All w-F1@0.500")
    return {
        "er_wf1_000": wf1_000,
        "er_wf1_050": wf1_050,
        "er_eacc_000": eacc_000,
        "er_eacc_050": eacc_050,
    }

def parse_run_dir(run_dir: Path):
    log_dir = run_dir / "log"
    out = {}
    out.update(parse_action_seg(log_dir / "action_segmentation.txt"))
    out.update(parse_error_detection(log_dir / "error_detection.txt"))
    out.update(parse_error_recognition(log_dir / "error_recognition.txt"))
    return out

def latest_run_dir(repo_root: Path, task: str, tag: str):
    task_root = repo_root / "ckpts" / "EgoPER" / task
    candidates = [p for p in task_root.glob(f"{tag}_*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def fmt(v):
    return "-" if v is None else f"{v:.1f}"

def delta(v1, v0):
    if v1 is None or v0 is None:
        return None
    return v1 - v0

def fmt_delta(v):
    if v is None:
        return "-"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}"

def build_task_row(task, base_metrics, vm_metrics):
    row = {"task": task}
    for k, _ in METRIC_SPECS:
        row[f"baseline_{k}"] = base_metrics.get(k)
        row[f"vm_{k}"] = vm_metrics.get(k)
        row[f"delta_{k}"] = delta(vm_metrics.get(k), base_metrics.get(k))
    return row

def avg_metric(rows, prefix_key):
    vals = [r[prefix_key] for r in rows if r[prefix_key] is not None]
    return mean(vals) if vals else None

def write_csv(path: Path, rows):
    fieldnames = ["task"]
    for k, _ in METRIC_SPECS:
        fieldnames += [f"baseline_{k}", f"vm_{k}", f"delta_{k}"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def write_markdown(path: Path, rows, base_dirs, vm_dirs, summary, tasks):
    lines = []
    lines.append("# EgoPER Baseline vs Visual-Memory Comparison Report\n\n")
    lines.append(f"- Generated at: {summary['generated_at']}\n")
    lines.append(f"- Repo root: `{summary['repo_root']}`\n")
    lines.append(f"- Baseline tag: `{summary['baseline_tag']}`\n")
    lines.append(f"- Visual-memory tag: `{summary['vm_tag']}`\n")
    lines.append(f"- Tasks: `{tasks}`\n\n")

    lines.append("## Run directories\n\n")
    lines.append("| Task | Baseline dir | Visual-memory dir |\n")
    lines.append("|---|---|---|\n")
    for task in tasks:
        lines.append(f"| {task} | `{base_dirs.get(task, '-')}` | `{vm_dirs.get(task, '-')}` |\n")
    lines.append("\n")

    lines.append("## Overall average deltas (VM - Baseline)\n\n")
    lines.append("| Metric | Baseline Avg | VM Avg | Delta |\n")
    lines.append("|---|---:|---:|---:|\n")
    for k, label in METRIC_SPECS:
        base_avg = summary["avg"].get(f"baseline_{k}")
        vm_avg = summary["avg"].get(f"vm_{k}")
        d_avg = summary["avg"].get(f"delta_{k}")
        lines.append(f"| {label} | {fmt(base_avg)} | {fmt(vm_avg)} | {fmt_delta(d_avg)} |\n")
    lines.append("\n")

    lines.append("## Per-task comparison\n\n")
    header = ["Task"]
    for _, label in METRIC_SPECS:
        header.extend([f"{label} (B)", f"{label} (VM)", "Δ"])
    lines.append("| " + " | ".join(header) + " |\n")
    lines.append("|" + "|".join(["---"] * len(header)) + "|\n")

    for row in rows:
        cells = [row["task"]]
        for k, _ in METRIC_SPECS:
            cells.extend([
                fmt(row[f"baseline_{k}"]),
                fmt(row[f"vm_{k}"]),
                fmt_delta(row[f"delta_{k}"]),
            ])
        lines.append("| " + " | ".join(cells) + " |\n")

    path.write_text("".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default="/root/autodl-tmp/GTG-memory")
    parser.add_argument("--baseline_tag", type=str, default="baseline_retrain")
    parser.add_argument("--vm_tag", type=str, default="vm_warmstart")
    parser.add_argument("--task_list_json", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.task_list_json:
        tasks = load_json(Path(args.task_list_json))["ready_tasks"]
    else:
        tasks = DEFAULT_TASKS

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = repo_root / "reports" / "compare_runs" / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    base_dirs = {}
    vm_dirs = {}

    for task in tasks:
        base_dir = latest_run_dir(repo_root, task, args.baseline_tag)
        vm_dir = latest_run_dir(repo_root, task, args.vm_tag)

        base_dirs[task] = str(base_dir) if base_dir else "-"
        vm_dirs[task] = str(vm_dir) if vm_dir else "-"

        if base_dir is None or vm_dir is None:
            print(f"[SKIP] task={task}, missing baseline or vm run")
            continue

        base_metrics = parse_run_dir(base_dir)
        vm_metrics = parse_run_dir(vm_dir)
        rows.append(build_task_row(task, base_metrics, vm_metrics))

    avg_summary = {}
    for k, _ in METRIC_SPECS:
        avg_summary[f"baseline_{k}"] = avg_metric(rows, f"baseline_{k}")
        avg_summary[f"vm_{k}"] = avg_metric(rows, f"vm_{k}")
        avg_summary[f"delta_{k}"] = avg_metric(rows, f"delta_{k}")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "baseline_tag": args.baseline_tag,
        "vm_tag": args.vm_tag,
        "tasks": tasks,
        "avg": avg_summary,
        "rows": rows,
        "baseline_dirs": base_dirs,
        "vm_dirs": vm_dirs,
    }

    md_path = output_root / "comparison_report.md"
    csv_path = output_root / "comparison_summary.csv"
    json_path = output_root / "comparison_summary.json"

    write_markdown(md_path, rows, base_dirs, vm_dirs, summary, tasks)
    write_csv(csv_path, rows)
    write_json(json_path, summary)

    print(f"[WRITE] {md_path}")
    print(f"[WRITE] {csv_path}")
    print(f"[WRITE] {json_path}")

if __name__ == "__main__":
>>>>>>> Stashed changes
    main()