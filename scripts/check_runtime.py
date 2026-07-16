#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.runtime_config import load_runtime_config


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "networkx",
    "matplotlib",
    "tensorboard",
    "tqdm",
]

REQUIRED_CONFIG_KEYS = [
    "naming",
    "root_data_dir",
    "dataset_name",
    "v_feat_path",
    "label_path",
    "train_split",
    "val_split",
    "test_split",
    "ckpt_dir",
    "input_dim",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "num_epochs",
    "log_freq",
    "ignore_idx",
    "num_iterations",
    "background_weight",
    "drop_base",
    "simple_error_path",
    "simple_error_filename",
]


class CheckResult:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")


def check_file(result: CheckResult, path: Path, label: str) -> bool:
    if path.is_file():
        result.ok(f"{label}: {path}")
        return True
    else:
        result.fail(f"{label} missing: {path}")
        return False


def check_dir(result: CheckResult, path: Path, label: str) -> bool:
    if path.is_dir():
        result.ok(f"{label}: {path}")
        return True
    else:
        result.fail(f"{label} missing: {path}")
        return False


def read_nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def check_split_assets(
    result: CheckResult,
    split_path: Path,
    feature_dir: Path,
    label_dir: Path,
    *,
    feature_suffix: str,
    label: str,
) -> None:
    if not split_path.is_file() or not feature_dir.is_dir() or not label_dir.is_dir():
        return

    missing = []
    videos = read_nonempty_lines(split_path)
    for video in videos:
        feature_path = feature_dir / f"{video}{feature_suffix}.npy"
        label_path = label_dir / f"{video}.txt"
        if not feature_path.is_file():
            missing.append(str(feature_path))
        if not label_path.is_file():
            missing.append(str(label_path))

    if missing:
        examples = ", ".join(missing[:3])
        result.fail(f"{label} assets missing: count={len(missing)}, examples={examples}")
    else:
        result.ok(f"{label} assets: {len(videos)} videos")


def check_named_features(
    result: CheckResult,
    list_path: Path,
    feature_dir: Path,
    *,
    label: str,
) -> None:
    if not list_path.is_file() or not feature_dir.is_dir():
        return

    names = [line.split(maxsplit=1)[0] for line in read_nonempty_lines(list_path)]
    missing = [str(feature_dir / f"{name}.npy") for name in names if not (feature_dir / f"{name}.npy").is_file()]
    if missing:
        examples = ", ".join(missing[:3])
        result.fail(f"{label} missing: count={len(missing)}, examples={examples}")
    else:
        result.ok(f"{label}: {len(names)} feature files")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check GTG-memory runtime readiness.")
    parser.add_argument(
        "--config",
        default="configs/EgoPER/tea/vc_4omini_post_db0.6.json",
        help="Experiment config to validate.",
    )
    parser.add_argument("--data-root", default=None, help="Override config.root_data_dir.")
    parser.add_argument("--ckpt-root", default=None, help="Override config.ckpt_dir.")
    parser.add_argument("--eval", action="store_true", help="Also require an evaluation checkpoint.")
    parser.add_argument("--load-dir", default="best", help="Evaluation checkpoint directory name.")
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Check imports and config loading without requiring datasets or checkpoints.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = CheckResult()

    result.ok(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            result.ok(f"dependency {module_name}: {version}")
        except Exception as exc:
            result.fail(f"dependency {module_name}: {type(exc).__name__}: {exc}")

    try:
        config = load_runtime_config(
            args.config,
            data_root=args.data_root,
            ckpt_root=args.ckpt_root,
        )
        result.ok(f"config: {config['_config_path']}")
    except Exception as exc:
        result.fail(f"config load: {type(exc).__name__}: {exc}")
        config = None

    if config is not None:
        missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
        if missing_keys:
            result.fail(f"config keys missing: {missing_keys}")
        else:
            result.ok("required config keys")

    if args.code_only or config is None:
        print(f"[SUMMARY] failures={result.failures} warnings={result.warnings} mode=code-only")
        return 1 if result.failures else 0

    data_root = Path(config["root_data_dir"])
    task_root = data_root / config["dataset_name"]
    check_dir(result, data_root, "data root")
    for name in ["action2idx.json", "idx2action.json", "actiontype2idx.json", "idx2actiontype.json"]:
        check_file(result, data_root / name, f"mapping {name}")

    check_dir(result, task_root, "task root")
    feature_dir = task_root / config["v_feat_path"]
    label_dir = task_root / config["label_path"]
    check_dir(result, feature_dir, "visual features")
    check_dir(result, label_dir, "labels")
    feature_suffix = "_360p" if config["naming"] == "CaptainCook4D" else ""
    for split_key in ["train_split", "val_split", "test_split"]:
        split_path = task_root / f"{config[split_key]}.txt"
        check_file(result, split_path, split_key)
        check_split_assets(
            result,
            split_path,
            feature_dir,
            label_dir,
            feature_suffix=feature_suffix,
            label=split_key,
        )

    error_list_path = task_root / f"{config['simple_error_filename']}.txt"
    normal_list_path = task_root / "normal_actions.txt"
    error_feature_dir = task_root / config["simple_error_path"]
    normal_feature_dir = task_root / "vc_normal_action_features"
    check_file(result, error_list_path, "error prototype list")
    check_file(result, normal_list_path, "normal action list")
    check_dir(result, error_feature_dir, "error prototype features")
    check_dir(result, normal_feature_dir, "normal action features")
    check_named_features(
        result,
        error_list_path,
        error_feature_dir,
        label="error prototype features",
    )
    check_named_features(
        result,
        normal_list_path,
        normal_feature_dir,
        label="normal action features",
    )

    if config.get("use_semantic_memory"):
        normal_dir = config.get("semantic_feature_dir", "vc_normal_action_features")
        error_dir = config.get(
            "semantic_error_feature_dir",
            config.get("simple_error_path", "vc_chatgpt4omini_error_features"),
        )
        if task_root / normal_dir != normal_feature_dir:
            check_dir(result, task_root / normal_dir, "normal semantic prototypes")
        if task_root / error_dir != error_feature_dir:
            check_dir(result, task_root / error_dir, "error semantic prototypes")

    pretrained = config.get("pretrained_backbone_ckpt")
    if pretrained and not args.eval:
        check_file(result, Path(pretrained), "pretrained backbone checkpoint")

    if args.eval:
        eval_ckpt = (
            Path(config["ckpt_dir"])
            / config["naming"]
            / config["dataset_name"]
            / args.load_dir
            / "best_checkpoint.pth"
        )
        check_file(result, eval_ckpt, "evaluation checkpoint")

    print(f"[SUMMARY] failures={result.failures} warnings={result.warnings} mode=full")
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
