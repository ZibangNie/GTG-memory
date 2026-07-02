# Build EgoPER available-only splits and matching configs.
#
# Outputs:
# - /root/autodl-tmp/data/EgoPER/<task>/{training,validation,test}_available_only.txt
# - configs/EgoPER/<task>/generated_available_only/
# - reports/task_probe/egoper_available_only_latest.json
import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.gtg_dataset_loader import get_data_dict
from egoper_utils import (
    ALL_EGOPER_TASKS,
    add_training_config_args,
    baseline_config,
    dump_json,
    dump_lines,
    find_base_config,
    load_json,
    naming_meta,
    visual_memory_config,
)


def try_one_video(cfg, split_key: str, video_id: str):
    root_data_dir = cfg["root_data_dir"]
    dataset_name = cfg["dataset_name"]
    naming = cfg["naming"]

    action2idx = load_json(Path(root_data_dir) / "action2idx.json")[dataset_name]
    actiontype2idx = load_json(Path(root_data_dir) / "actiontype2idx.json")

    meta = naming_meta(naming)

    v_feature_dir = Path(root_data_dir) / dataset_name / cfg["v_feat_path"]
    label_dir = Path(root_data_dir) / dataset_name / cfg["label_path"]

    try:
        _ = get_data_dict(
            v_feature_dir=str(v_feature_dir),
            label_dir=str(label_dir),
            video_list=[video_id],
            action2idx=action2idx,
            actiontype2idx=actiontype2idx,
            addition_name=meta["addition_name"],
            suffix=meta["suffix"],
        )
        return True, "ok"
    except FileNotFoundError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def build_available_split(cfg, split_key: str):
    root_data_dir = Path(cfg["root_data_dir"])
    dataset_name = cfg["dataset_name"]
    split_name = cfg[split_key]
    split_file = root_data_dir / dataset_name / f"{split_name}.txt"

    if not split_file.is_file():
        return {
            "ok": False,
            "reason": f"missing split file: {split_file}",
            "kept": [],
            "skipped": {},
        }

    with split_file.open("r", encoding="utf-8") as f:
        raw_videos = [line.strip() for line in f.readlines() if line.strip()]

    kept = []
    skipped = {}
    reason_counter = Counter()

    for vid in raw_videos:
        ok, msg = try_one_video(cfg, split_key, vid)
        if ok:
            kept.append(vid)
        else:
            skipped[vid] = msg
            # Keep the report compact by grouping failures coarsely.
            if "No such file or directory" in msg:
                reason_counter["missing_file"] += 1
            else:
                reason_counter["other_error"] += 1

    return {
        "ok": True,
        "original_count": len(raw_videos),
        "kept_count": len(kept),
        "skipped_count": len(skipped),
        "reason_counter": dict(reason_counter),
        "kept": kept,
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default="/root/autodl-tmp/GTG-memory")
    add_training_config_args(parser)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "generated_at": ts,
        "repo_root": str(repo_root),
        "dataset": "EgoPER",
        "mode": "available_only",
        "ready_tasks": [],
        "skipped_tasks": [],
        "records": {},
    }

    for task in ALL_EGOPER_TASKS:
        task_dir = repo_root / "configs" / "EgoPER" / task
        src_cfg = find_base_config(task_dir)

        if src_cfg is None:
            summary["skipped_tasks"].append(task)
            summary["records"][task] = {
                "status": "skip",
                "reason": f"no base config found under {task_dir}",
            }
            print(f"[SKIP] {task}: no base config")
            continue

        cfg = load_json(src_cfg)
        train_info = build_available_split(cfg, "train_split")
        val_info = build_available_split(cfg, "val_split")
        test_info = build_available_split(cfg, "test_split")

        record = {
            "base_config": str(src_cfg),
            "train_split": train_info,
            "val_split": val_info,
            "test_split": test_info,
        }

        # A task is ready only when all three available-only splits are non-empty.
        all_nonempty = (
            train_info["ok"] and val_info["ok"] and test_info["ok"]
            and len(train_info["kept"]) > 0
            and len(val_info["kept"]) > 0
            and len(test_info["kept"]) > 0
        )

        if not all_nonempty:
            summary["skipped_tasks"].append(task)
            record["status"] = "skip"
            summary["records"][task] = record
            print(f"[SKIP] {task}: no usable full split set")
            continue

        # Write available-only split files beside the dataset split files.
        data_root = Path(cfg["root_data_dir"]) / cfg["dataset_name"]
        train_name = "training_available_only"
        val_name = "validation_available_only"
        test_name = "test_available_only"

        dump_lines(data_root / f"{train_name}.txt", train_info["kept"])
        dump_lines(data_root / f"{val_name}.txt", val_info["kept"])
        dump_lines(data_root / f"{test_name}.txt", test_info["kept"])

        generated_dir = task_dir / "generated_available_only"

        baseline_cfg = baseline_config(cfg, args)
        baseline_cfg["train_split"] = train_name
        baseline_cfg["val_split"] = val_name
        baseline_cfg["test_split"] = test_name
        baseline_out = generated_dir / "vc_4omini_post_db0.6.available_only.baseline.train.json"
        dump_json(baseline_out, baseline_cfg)

        vm_cfg = visual_memory_config(cfg, args, task)
        vm_cfg["train_split"] = train_name
        vm_cfg["val_split"] = val_name
        vm_cfg["test_split"] = test_name
        vm_out = generated_dir / "vc_4omini_post_db0.6.available_only.visual_memory.train.json"
        dump_json(vm_out, vm_cfg)

        record["status"] = "ready"
        record["generated_baseline_config"] = str(baseline_out)
        record["generated_vm_config"] = str(vm_out)
        summary["records"][task] = record
        summary["ready_tasks"].append(task)

        print(
            f"[READY] {task} | "
            f"train={len(train_info['kept'])}/{train_info['original_count']} "
            f"val={len(val_info['kept'])}/{val_info['original_count']} "
            f"test={len(test_info['kept'])}/{test_info['original_count']}"
        )

    out_dir = repo_root / "reports" / "task_probe"
    out_path = out_dir / f"egoper_available_only_{ts}.json"
    latest_path = out_dir / "egoper_available_only_latest.json"
    dump_json(out_path, summary)
    dump_json(latest_path, summary)

    print(f"[WRITE] {out_path}")
    print(f"[WRITE] {latest_path}")
    print(f"[READY TASKS] {summary['ready_tasks']}")


if __name__ == "__main__":
    main()
