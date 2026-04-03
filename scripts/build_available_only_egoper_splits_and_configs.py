<<<<<<< Updated upstream
# 文件：scripts/build_available_only_egoper_splits_and_configs.py
# 作用：
# 1) 对 EgoPER 每个任务，按当前本地实际可读的视频重建 train/val/test split
# 2) 写回到 data/EgoPER/<task>/*.txt
# 3) 生成 available-only 的 baseline / visual-memory config
# 4) 输出 summary json，后续训练脚本只跑 ready_tasks
#
# 输出：
# - /root/autodl-tmp/data/EgoPER/<task>/{training,validation,test}_available_only.txt
# - configs/EgoPER/<task>/generated_available_only/
# - reports/task_probe/egoper_available_only_latest.json

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.gtg_dataset_loader import get_data_dict

ALL_TASKS = ["tea", "oatmeal", "pinwheels", "quesadilla", "coffee"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def dump_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in lines:
            f.write(str(x).strip() + "\n")


def find_base_config(task_dir: Path):
    candidates = []
    for p in task_dir.glob("*.json"):
        name = p.name
        if ".visual_memory." in name:
            continue
        if ".baseline.debug" in name:
            continue
        if ".vm_tmp" in name:
            continue
        if name.endswith(".visual_memory.train.json"):
            continue
        if name.endswith(".baseline.train.json"):
            continue
        candidates.append(p)
    candidates = sorted(candidates)
    return candidates[0] if candidates else None


def naming_meta(naming: str):
    if naming == "CaptainCook4D":
        return {"addition_name": "Other", "suffix": "_360p"}
    elif naming == "EgoPER":
        return {"addition_name": "Error_Addition", "suffix": ""}
    else:
        raise ValueError(f"Unsupported naming: {naming}")


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
            # 只统计粗粒度原因
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
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--background_weight", type=float, default=2.0)
    parser.add_argument("--short_dim", type=int, default=256)
    parser.add_argument("--long_dim", type=int, default=384)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--long_write_cap", type=float, default=0.2)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--backbone_learning_rate", type=float, default=5e-5)
    parser.add_argument("--vm_learning_rate", type=float, default=1e-4)
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

    for task in ALL_TASKS:
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

        # 三个 split 都至少要非空才能进入 ready
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

        # 写 available-only split 文件到 data 目录
        data_root = Path(cfg["root_data_dir"]) / cfg["dataset_name"]
        train_name = "training_available_only"
        val_name = "validation_available_only"
        test_name = "test_available_only"

        dump_lines(data_root / f"{train_name}.txt", train_info["kept"])
        dump_lines(data_root / f"{val_name}.txt", val_info["kept"])
        dump_lines(data_root / f"{test_name}.txt", test_info["kept"])

        generated_dir = task_dir / "generated_available_only"

        # baseline config
        baseline_cfg = dict(cfg)
        baseline_cfg["train_split"] = train_name
        baseline_cfg["val_split"] = val_name
        baseline_cfg["test_split"] = test_name
        baseline_cfg["batch_size"] = args.batch_size
        baseline_cfg["learning_rate"] = args.learning_rate
        baseline_cfg["weight_decay"] = args.weight_decay
        baseline_cfg["num_epochs"] = args.num_epochs
        baseline_cfg["log_freq"] = args.log_freq
        baseline_cfg["num_iterations"] = args.num_iterations
        baseline_cfg["background_weight"] = args.background_weight
        baseline_cfg["use_visual_memory"] = False
        baseline_cfg.pop("short_dim", None)
        baseline_cfg.pop("long_dim", None)
        baseline_cfg.pop("fusion_dim", None)
        baseline_cfg.pop("long_write_cap", None)
        baseline_cfg.pop("fusion_dropout", None)
        baseline_cfg.pop("backbone_learning_rate", None)
        baseline_cfg.pop("vm_learning_rate", None)
        baseline_cfg.pop("pretrained_backbone_ckpt", None)

        baseline_out = generated_dir / "vc_4omini_post_db0.6.available_only.baseline.train.json"
        dump_json(baseline_out, baseline_cfg)

        # vm config
        vm_cfg = dict(cfg)
        vm_cfg["train_split"] = train_name
        vm_cfg["val_split"] = val_name
        vm_cfg["test_split"] = test_name
        vm_cfg["batch_size"] = args.batch_size
        vm_cfg["learning_rate"] = args.learning_rate
        vm_cfg["weight_decay"] = args.weight_decay
        vm_cfg["num_epochs"] = args.num_epochs
        vm_cfg["log_freq"] = args.log_freq
        vm_cfg["num_iterations"] = args.num_iterations
        vm_cfg["background_weight"] = args.background_weight
        vm_cfg["use_visual_memory"] = True
        vm_cfg["short_dim"] = args.short_dim
        vm_cfg["long_dim"] = args.long_dim
        vm_cfg["fusion_dim"] = args.fusion_dim
        vm_cfg["long_write_cap"] = args.long_write_cap
        vm_cfg["fusion_dropout"] = args.fusion_dropout
        vm_cfg["backbone_learning_rate"] = args.backbone_learning_rate
        vm_cfg["vm_learning_rate"] = args.vm_learning_rate
        vm_cfg["pretrained_backbone_ckpt"] = f"ckpts/EgoPER/{task}/best/best_checkpoint.pth"

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
=======
# 文件：scripts/build_available_only_egoper_splits_and_configs.py
# 作用：
# 1) 对 EgoPER 每个任务，按当前本地实际可读的视频重建 train/val/test split
# 2) 写回到 data/EgoPER/<task>/*.txt
# 3) 生成 available-only 的 baseline / visual-memory config
# 4) 输出 summary json，后续训练脚本只跑 ready_tasks
#
# 输出：
# - /root/autodl-tmp/data/EgoPER/<task>/{training,validation,test}_available_only.txt
# - configs/EgoPER/<task>/generated_available_only/
# - reports/task_probe/egoper_available_only_latest.json

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.gtg_dataset_loader import get_data_dict

ALL_TASKS = ["tea", "oatmeal", "pinwheels", "quesadilla", "coffee"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def dump_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in lines:
            f.write(str(x).strip() + "\n")


def find_base_config(task_dir: Path):
    candidates = []
    for p in task_dir.glob("*.json"):
        name = p.name
        if ".visual_memory." in name:
            continue
        if ".baseline.debug" in name:
            continue
        if ".vm_tmp" in name:
            continue
        if name.endswith(".visual_memory.train.json"):
            continue
        if name.endswith(".baseline.train.json"):
            continue
        candidates.append(p)
    candidates = sorted(candidates)
    return candidates[0] if candidates else None


def naming_meta(naming: str):
    if naming == "CaptainCook4D":
        return {"addition_name": "Other", "suffix": "_360p"}
    elif naming == "EgoPER":
        return {"addition_name": "Error_Addition", "suffix": ""}
    else:
        raise ValueError(f"Unsupported naming: {naming}")


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
            # 只统计粗粒度原因
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
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--background_weight", type=float, default=2.0)
    parser.add_argument("--short_dim", type=int, default=256)
    parser.add_argument("--long_dim", type=int, default=384)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--long_write_cap", type=float, default=0.2)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--backbone_learning_rate", type=float, default=5e-5)
    parser.add_argument("--vm_learning_rate", type=float, default=1e-4)
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

    for task in ALL_TASKS:
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

        # 三个 split 都至少要非空才能进入 ready
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

        # 写 available-only split 文件到 data 目录
        data_root = Path(cfg["root_data_dir"]) / cfg["dataset_name"]
        train_name = "training_available_only"
        val_name = "validation_available_only"
        test_name = "test_available_only"

        dump_lines(data_root / f"{train_name}.txt", train_info["kept"])
        dump_lines(data_root / f"{val_name}.txt", val_info["kept"])
        dump_lines(data_root / f"{test_name}.txt", test_info["kept"])

        generated_dir = task_dir / "generated_available_only"

        # baseline config
        baseline_cfg = dict(cfg)
        baseline_cfg["train_split"] = train_name
        baseline_cfg["val_split"] = val_name
        baseline_cfg["test_split"] = test_name
        baseline_cfg["batch_size"] = args.batch_size
        baseline_cfg["learning_rate"] = args.learning_rate
        baseline_cfg["weight_decay"] = args.weight_decay
        baseline_cfg["num_epochs"] = args.num_epochs
        baseline_cfg["log_freq"] = args.log_freq
        baseline_cfg["num_iterations"] = args.num_iterations
        baseline_cfg["background_weight"] = args.background_weight
        baseline_cfg["use_visual_memory"] = False
        baseline_cfg.pop("short_dim", None)
        baseline_cfg.pop("long_dim", None)
        baseline_cfg.pop("fusion_dim", None)
        baseline_cfg.pop("long_write_cap", None)
        baseline_cfg.pop("fusion_dropout", None)
        baseline_cfg.pop("backbone_learning_rate", None)
        baseline_cfg.pop("vm_learning_rate", None)
        baseline_cfg.pop("pretrained_backbone_ckpt", None)

        baseline_out = generated_dir / "vc_4omini_post_db0.6.available_only.baseline.train.json"
        dump_json(baseline_out, baseline_cfg)

        # vm config
        vm_cfg = dict(cfg)
        vm_cfg["train_split"] = train_name
        vm_cfg["val_split"] = val_name
        vm_cfg["test_split"] = test_name
        vm_cfg["batch_size"] = args.batch_size
        vm_cfg["learning_rate"] = args.learning_rate
        vm_cfg["weight_decay"] = args.weight_decay
        vm_cfg["num_epochs"] = args.num_epochs
        vm_cfg["log_freq"] = args.log_freq
        vm_cfg["num_iterations"] = args.num_iterations
        vm_cfg["background_weight"] = args.background_weight
        vm_cfg["use_visual_memory"] = True
        vm_cfg["short_dim"] = args.short_dim
        vm_cfg["long_dim"] = args.long_dim
        vm_cfg["fusion_dim"] = args.fusion_dim
        vm_cfg["long_write_cap"] = args.long_write_cap
        vm_cfg["fusion_dropout"] = args.fusion_dropout
        vm_cfg["backbone_learning_rate"] = args.backbone_learning_rate
        vm_cfg["vm_learning_rate"] = args.vm_learning_rate
        vm_cfg["pretrained_backbone_ckpt"] = f"ckpts/EgoPER/{task}/best/best_checkpoint.pth"

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
>>>>>>> Stashed changes
    main()