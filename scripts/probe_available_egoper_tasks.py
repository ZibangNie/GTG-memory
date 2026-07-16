# Probe which EgoPER tasks can be loaded by the current data installation.
#
# Outputs:
# - reports/task_probe/egoper_ready_tasks_<timestamp>.json
# - reports/task_probe/egoper_ready_tasks_latest.json
import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.gtg_dataset_loader import get_data_dict
from egoper_utils import (
    ALL_EGOPER_TASKS,
    PROJECT_ROOT,
    add_egoper_data_root_arg,
    apply_data_root_override,
    dump_json,
    find_base_config,
    load_json,
    naming_meta,
)


def check_split(cfg, split_key: str):
    root_data_dir = cfg["root_data_dir"]
    dataset_name = cfg["dataset_name"]
    naming = cfg["naming"]

    action2idx = load_json(Path(root_data_dir) / "action2idx.json")[dataset_name]
    actiontype2idx = load_json(Path(root_data_dir) / "actiontype2idx.json")

    meta = naming_meta(naming)

    v_feature_dir = Path(root_data_dir) / dataset_name / cfg["v_feat_path"]
    label_dir = Path(root_data_dir) / dataset_name / cfg["label_path"]
    split_file = Path(root_data_dir) / dataset_name / f"{cfg[split_key]}.txt"

    if not v_feature_dir.is_dir():
        return False, f"missing feature dir: {v_feature_dir}"
    if not label_dir.is_dir():
        return False, f"missing label dir: {label_dir}"
    if not split_file.is_file():
        return False, f"missing split file: {split_file}"

    with split_file.open("r", encoding="utf-8") as f:
        video_list = [line.strip() for line in f.readlines() if line.strip()]

    try:
        _ = get_data_dict(
            v_feature_dir=str(v_feature_dir),
            label_dir=str(label_dir),
            video_list=video_list,
            action2idx=action2idx,
            actiontype2idx=actiontype2idx,
            addition_name=meta["addition_name"],
            suffix=meta["suffix"],
        )
        return True, f"{split_key} ok"
    except FileNotFoundError as e:
        return False, f"{split_key} missing file: {e}"
    except Exception as e:
        return False, f"{split_key} error: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=str(PROJECT_ROOT))
    add_egoper_data_root_arg(parser)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "generated_at": ts,
        "repo_root": str(repo_root),
        "dataset": "EgoPER",
        "ready_tasks": [],
        "skipped_tasks": [],
        "records": {},
    }

    for task in ALL_EGOPER_TASKS:
        task_dir = repo_root / "configs" / "EgoPER" / task
        base_cfg = find_base_config(task_dir)

        if base_cfg is None:
            report["skipped_tasks"].append(task)
            report["records"][task] = {
                "status": "skip",
                "reason": f"no base config found under {task_dir}",
                "base_config": None,
            }
            print(f"[SKIP] {task}: no base config")
            continue

        cfg = apply_data_root_override(load_json(base_cfg), args.data_root)
        split_checks = {}
        ok_all = True

        for split_key in ["train_split", "val_split", "test_split"]:
            ok, msg = check_split(cfg, split_key)
            split_checks[split_key] = {"ok": ok, "message": msg}
            if not ok:
                ok_all = False

        if ok_all:
            report["ready_tasks"].append(task)
            report["records"][task] = {
                "status": "ready",
                "base_config": str(base_cfg),
                "split_checks": split_checks,
            }
            print(f"[READY] {task} -> {base_cfg.name}")
        else:
            report["skipped_tasks"].append(task)
            report["records"][task] = {
                "status": "skip",
                "base_config": str(base_cfg),
                "split_checks": split_checks,
            }
            print(f"[SKIP] {task} -> {base_cfg.name}")

    out_dir = repo_root / "reports" / "task_probe"
    out_path = out_dir / f"egoper_ready_tasks_{ts}.json"
    latest_path = out_dir / "egoper_ready_tasks_latest.json"

    dump_json(out_path, report)
    dump_json(latest_path, report)

    print(f"[WRITE] {out_path}")
    print(f"[WRITE] {latest_path}")
    print(f"[READY TASKS] {report['ready_tasks']}")


if __name__ == "__main__":
    main()
