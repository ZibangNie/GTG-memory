#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from egoper_utils import PROJECT_ROOT, load_json
from experiment_metrics import (
    METRIC_LABELS,
    METRIC_SPECS,
    PRIMARY_METRICS,
    parse_log_dir,
    validate_metrics,
)


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def average_rows(rows: list[dict]) -> dict[str, float | None]:
    averages: dict[str, float | None] = {}
    for key, _ in METRIC_SPECS:
        values = [row["metrics"][key] for row in rows if row["metrics"].get(key) is not None]
        averages[key] = mean(values) if values else None
    return averages


def collect_results(
    repo_root: Path,
    manifest: dict,
    *,
    manifest_ref: str = "reports/experiments/egoper_runs.json",
) -> dict:
    expected_tasks = set(manifest["tasks"])
    variants = {}

    for variant_id, variant in manifest["variants"].items():
        rows = []
        run_tasks = set(variant["runs"])
        if variant["scope"] == "five_task" and run_tasks != expected_tasks:
            raise ValueError(
                f"Variant {variant_id} must define exactly {sorted(expected_tasks)}, got {sorted(run_tasks)}"
            )

        for task, run in variant["runs"].items():
            log_dir = repo_root / run["log_dir"]
            config_path = repo_root / run["config"]
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Missing config for variant={variant_id}, task={task}: {config_path}"
                )
            metrics = parse_log_dir(log_dir)
            validate_metrics(metrics, log_dir)
            rows.append(
                {
                    "task": task,
                    "run_id": run["run_id"],
                    "config": run["config"],
                    "log_dir": run["log_dir"],
                    "metrics": metrics,
                }
            )

        variants[variant_id] = {
            "label": variant["label"],
            "scope": variant["scope"],
            "task_count": len(rows),
            "average": average_rows(rows),
            "rows": rows,
        }

    return {
        "schema_version": manifest["schema_version"],
        "dataset": manifest["dataset"],
        "snapshot_date": manifest["snapshot_date"],
        "manifest": manifest_ref,
        "variants": variants,
        "limitations": manifest.get("limitations", []),
    }


def write_csv(path: Path, payload: dict) -> None:
    fieldnames = ["variant", "variant_label", "scope", "task", "run_id", "config", "log_dir"]
    fieldnames.extend(key for key, _ in METRIC_SPECS)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant_id, variant in payload["variants"].items():
            for row in variant["rows"]:
                output = {
                    "variant": variant_id,
                    "variant_label": variant["label"],
                    "scope": variant["scope"],
                    "task": row["task"],
                    "run_id": row["run_id"],
                    "config": row["config"],
                    "log_dir": row["log_dir"],
                }
                output.update(row["metrics"])
                writer.writerow(output)


def markdown_table_header(columns: list[str], numeric_from: int = 1) -> list[str]:
    alignments = ["---" if i < numeric_from else "---:" for i in range(len(columns))]
    return ["| " + " | ".join(columns) + " |", "|" + "|".join(alignments) + "|"]


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Canonical EgoPER Experiment Report",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Source snapshot: `{payload['snapshot_date']}`",
        f"- Manifest: `{payload['manifest']}`",
        "- Values are parsed from the preserved source logs listed below.",
        "",
        "## Five-task averages",
        "",
    ]

    columns = ["Variant"] + [METRIC_LABELS[key] for key in PRIMARY_METRICS]
    lines.extend(markdown_table_header(columns))
    for variant in payload["variants"].values():
        if variant["scope"] != "five_task":
            continue
        cells = [variant["label"]] + [fmt(variant["average"][key]) for key in PRIMARY_METRICS]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Per-task results", ""])
    columns = ["Task", "Variant"] + [METRIC_LABELS[key] for key in PRIMARY_METRICS]
    lines.extend(markdown_table_header(columns, numeric_from=2))
    for variant in payload["variants"].values():
        if variant["scope"] != "five_task":
            continue
        for row in variant["rows"]:
            cells = [row["task"], variant["label"]]
            cells.extend(fmt(row["metrics"][key]) for key in PRIMARY_METRICS)
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Partial experiments", ""])
    partial_columns = ["Task", "Variant"] + [METRIC_LABELS[key] for key in PRIMARY_METRICS]
    lines.extend(markdown_table_header(partial_columns, numeric_from=2))
    for variant in payload["variants"].values():
        if variant["scope"] == "five_task":
            continue
        for row in variant["rows"]:
            cells = [row["task"], variant["label"]]
            cells.extend(fmt(row["metrics"][key]) for key in PRIMARY_METRICS)
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Provenance", ""])
    lines.extend(markdown_table_header(["Variant", "Task", "Run ID", "Config", "Source logs"], numeric_from=5))
    for variant in payload["variants"].values():
        for row in variant["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant["label"],
                        row["task"],
                        f"`{row['run_id']}`",
                        f"`{row['config']}`",
                        f"`{row['log_dir']}`",
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Limitations", ""])
    for limitation in payload["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed-provenance EgoPER report.")
    parser.add_argument(
        "--manifest",
        default="reports/experiments/egoper_runs.json",
        help="Run manifest path, relative to the repository root by default.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/experiments/canonical",
        help="Output directory, relative to the repository root by default.",
    )
    args = parser.parse_args()

    repo_root = PROJECT_ROOT
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest_ref = manifest_path.relative_to(repo_root).as_posix()
    except ValueError:
        manifest_ref = str(manifest_path)

    payload = collect_results(
        repo_root,
        load_json(manifest_path),
        manifest_ref=manifest_ref,
    )
    json_path = output_dir / "egoper_results.json"
    csv_path = output_dir / "egoper_results.csv"
    markdown_path = output_dir / "egoper_results.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, payload)
    write_markdown(markdown_path, payload)

    print(f"[WRITE] {markdown_path}")
    print(f"[WRITE] {csv_path}")
    print(f"[WRITE] {json_path}")


if __name__ == "__main__":
    main()
