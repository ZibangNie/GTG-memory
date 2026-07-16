from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_absolute_on_any_platform(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def _resolve_project_path(value: str | os.PathLike[str], project_root: Path) -> Path:
    raw = str(value)
    path = Path(raw).expanduser()
    if _is_absolute_on_any_platform(raw):
        return path
    return (project_root / path).resolve()


def _resolve_checkpoint_reference(
    value: str,
    original_ckpt_dir: str,
    resolved_ckpt_dir: Path,
    project_root: Path,
) -> Path:
    if _is_absolute_on_any_platform(value):
        return Path(value).expanduser()

    reference = Path(value)
    original_root = Path(original_ckpt_dir)
    try:
        relative = reference.relative_to(original_root)
    except ValueError:
        return (project_root / reference).resolve()
    return (resolved_ckpt_dir / relative).resolve()


def load_runtime_config(
    config_path: str | os.PathLike[str],
    *,
    data_root: str | os.PathLike[str] | None = None,
    ckpt_root: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
) -> dict[str, Any]:
    """Load a JSON config and apply portable runtime path overrides."""

    root = Path(project_root).resolve()
    resolved_config_path = _resolve_project_path(config_path, root)
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    original_ckpt_dir = str(config.get("ckpt_dir", "ckpts"))
    selected_data_root = data_root or os.environ.get("GTG_DATA_ROOT") or config.get("root_data_dir")
    selected_ckpt_root = ckpt_root or os.environ.get("GTG_CKPT_ROOT") or original_ckpt_dir

    if not selected_data_root:
        raise ValueError("root_data_dir is missing; set it in the config or pass --data-root.")

    resolved_data_root = _resolve_project_path(selected_data_root, root)
    resolved_ckpt_root = _resolve_project_path(selected_ckpt_root, root)
    resolved_runs_root = _resolve_project_path(config.get("runs_dir", "runs"), root)

    config["root_data_dir"] = str(resolved_data_root)
    config["ckpt_dir"] = str(resolved_ckpt_root)
    config["runs_dir"] = str(resolved_runs_root)
    config["_config_path"] = str(resolved_config_path)
    config["_project_root"] = str(root)

    pretrained = config.get("pretrained_backbone_ckpt")
    if pretrained:
        config["pretrained_backbone_ckpt"] = str(
            _resolve_checkpoint_reference(
                str(pretrained),
                original_ckpt_dir,
                resolved_ckpt_root,
                root,
            )
        )

    return config
