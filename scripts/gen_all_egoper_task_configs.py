# 文件：scripts/gen_all_egoper_task_configs.py
# 作用：
# 只为“探测为 ready 的任务”生成 generated config
#
# 输入：
# --task_list_json reports/task_probe/egoper_ready_tasks_latest.json

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default="/root/autodl-tmp/GTG-memory")
    parser.add_argument("--task_list_json", type=str, required=True)
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
    task_list_json = Path(args.task_list_json)
    task_payload = load_json(task_list_json)
    tasks = task_payload["ready_tasks"]

    for task in tasks:
        task_dir = repo_root / "configs" / "EgoPER" / task
        src_cfg_path = find_base_config(task_dir)

        if src_cfg_path is None:
            print(f"[SKIP] no base config found for task={task}")
            continue

        print(f"[FOUND] task={task} -> {src_cfg_path.name}")

        cfg = load_json(src_cfg_path)

        cfg["batch_size"] = args.batch_size
        cfg["learning_rate"] = args.learning_rate
        cfg["weight_decay"] = args.weight_decay
        cfg["num_epochs"] = args.num_epochs
        cfg["log_freq"] = args.log_freq
        cfg["num_iterations"] = args.num_iterations
        cfg["background_weight"] = args.background_weight

        generated_dir = task_dir / "generated"

        baseline_cfg = dict(cfg)
        baseline_cfg["use_visual_memory"] = False
        baseline_cfg.pop("short_dim", None)
        baseline_cfg.pop("long_dim", None)
        baseline_cfg.pop("fusion_dim", None)
        baseline_cfg.pop("long_write_cap", None)
        baseline_cfg.pop("fusion_dropout", None)
        baseline_cfg.pop("backbone_learning_rate", None)
        baseline_cfg.pop("vm_learning_rate", None)
        baseline_cfg.pop("pretrained_backbone_ckpt", None)

        baseline_out = generated_dir / "vc_4omini_post_db0.6.baseline.train.json"
        dump_json(baseline_out, baseline_cfg)
        print(f"[WRITE] {baseline_out}")

        vm_cfg = dict(cfg)
        vm_cfg["use_visual_memory"] = True
        vm_cfg["short_dim"] = args.short_dim
        vm_cfg["long_dim"] = args.long_dim
        vm_cfg["fusion_dim"] = args.fusion_dim
        vm_cfg["long_write_cap"] = args.long_write_cap
        vm_cfg["fusion_dropout"] = args.fusion_dropout
        vm_cfg["backbone_learning_rate"] = args.backbone_learning_rate
        vm_cfg["vm_learning_rate"] = args.vm_learning_rate
        vm_cfg["pretrained_backbone_ckpt"] = f"ckpts/EgoPER/{task}/best/best_checkpoint.pth"

        vm_out = generated_dir / "vc_4omini_post_db0.6.visual_memory.train.json"
        dump_json(vm_out, vm_cfg)
        print(f"[WRITE] {vm_out}")


if __name__ == "__main__":
    main()