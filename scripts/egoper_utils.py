from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ALL_EGOPER_TASKS = ["tea", "oatmeal", "pinwheels", "quesadilla", "coffee"]


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path | str, obj: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def dump_lines(path: Path | str, lines: Iterable[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).strip() + "\n")


def find_base_config(task_dir: Path | str) -> Path | None:
    task_dir = Path(task_dir)
    candidates = []
    for path in task_dir.glob("*.json"):
        name = path.name
        if ".visual_memory." in name:
            continue
        if ".baseline.debug" in name:
            continue
        if ".semantic_memory." in name:
            continue
        if ".visual_semantic_memory." in name:
            continue
        if ".vm_tmp" in name:
            continue
        if name.endswith(".visual_memory.train.json"):
            continue
        if name.endswith(".baseline.train.json"):
            continue
        candidates.append(path)
    candidates = sorted(candidates)
    return candidates[0] if candidates else None


def naming_meta(naming: str) -> dict[str, str]:
    if naming == "CaptainCook4D":
        return {"addition_name": "Other", "suffix": "_360p"}
    if naming == "EgoPER":
        return {"addition_name": "Error_Addition", "suffix": ""}
    raise ValueError(f"Unsupported naming: {naming}")


def add_training_config_args(parser) -> None:
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


def apply_common_training_overrides(cfg: dict[str, Any], args) -> dict[str, Any]:
    out = dict(cfg)
    out["batch_size"] = args.batch_size
    out["learning_rate"] = args.learning_rate
    out["weight_decay"] = args.weight_decay
    out["num_epochs"] = args.num_epochs
    out["log_freq"] = args.log_freq
    out["num_iterations"] = args.num_iterations
    out["background_weight"] = args.background_weight
    return out


def baseline_config(cfg: dict[str, Any], args) -> dict[str, Any]:
    out = apply_common_training_overrides(cfg, args)
    out["use_visual_memory"] = False
    for key in [
        "short_dim",
        "long_dim",
        "fusion_dim",
        "long_write_cap",
        "fusion_dropout",
        "backbone_learning_rate",
        "vm_learning_rate",
        "pretrained_backbone_ckpt",
        "use_semantic_memory",
        "use_new_erm",
    ]:
        out.pop(key, None)
    return out


def visual_memory_config(cfg: dict[str, Any], args, task: str) -> dict[str, Any]:
    out = apply_common_training_overrides(cfg, args)
    out["use_visual_memory"] = True
    out["short_dim"] = args.short_dim
    out["long_dim"] = args.long_dim
    out["fusion_dim"] = args.fusion_dim
    out["long_write_cap"] = args.long_write_cap
    out["fusion_dropout"] = args.fusion_dropout
    out["backbone_learning_rate"] = args.backbone_learning_rate
    out["vm_learning_rate"] = args.vm_learning_rate
    out["pretrained_backbone_ckpt"] = f"ckpts/EgoPER/{task}/best/best_checkpoint.pth"
    return out
