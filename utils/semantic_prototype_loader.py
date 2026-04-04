from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from datasets.loader_graph import GraphLoader


ACTION_RE = re.compile(r"[Aa]ction_(-?\d+)")
ERROR_RE = re.compile(r"[Aa]ction_(-?\d+)_type_(\d+)")


def _load_mean_feature(paths: List[Path], feature_dim: int) -> np.ndarray:
    if len(paths) == 0:
        return np.zeros((feature_dim,), dtype=np.float32)

    feats = []
    for p in paths:
        arr = np.load(p)
        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 0:
            arr = np.full((feature_dim,), float(arr), dtype=np.float32)
        elif arr.ndim == 1:
            pass
        else:
            arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)

        if arr.shape[-1] != feature_dim:
            raise ValueError(f"Feature dim mismatch in {p}: got {arr.shape[-1]}, expected {feature_dim}")
        feats.append(arr)

    return np.stack(feats, axis=0).mean(axis=0).astype(np.float32)


def _infer_naming(root_data_dir: str) -> str:
    name = Path(root_data_dir).name
    if name in ("EgoPER", "CaptainCook4D"):
        return name
    raise ValueError(f"Cannot infer dataset naming from root_data_dir={root_data_dir}")


def _load_num_classes(root_data_dir: str, dataset_name: str) -> int:
    action2idx_path = Path(root_data_dir) / "action2idx.json"
    if not action2idx_path.exists():
        raise FileNotFoundError(f"action2idx.json not found: {action2idx_path}")

    with open(action2idx_path, "r") as fp:
        action2idx_all = json.load(fp)

    if dataset_name not in action2idx_all:
        raise KeyError(f"{dataset_name} not found in {action2idx_path}")

    return len(action2idx_all[dataset_name])


def load_task_semantic_prototypes(
    root_data_dir: str,
    dataset_name: str,
    feature_dim: int = 256,
    normal_dir_name: str = "vc_normal_action_features",
    error_dir_name: str = "vc_chatgpt4omini_error_features",
) -> Dict[str, object]:
    root_data_dir = str(root_data_dir)
    task_dir = Path(root_data_dir) / dataset_name
    if not task_dir.exists():
        raise FileNotFoundError(f"Task dir not found: {task_dir}")

    normal_dir = task_dir / normal_dir_name
    error_dir = task_dir / error_dir_name

    if not normal_dir.exists():
        raise FileNotFoundError(f"Normal prototype dir not found: {normal_dir}")
    if not error_dir.exists():
        raise FileNotFoundError(f"Error prototype dir not found: {error_dir}")

    naming = _infer_naming(root_data_dir)
    num_classes = _load_num_classes(root_data_dir, dataset_name)

    graph_info = GraphLoader(naming, dataset_name, num_classes).graph_info
    step_node_ids = sorted(int(x) for x in graph_info["nodes"])
    step_node_id_set = set(step_node_ids)
    predecessor_edges = [tuple(map(int, e)) for e in graph_info["edges"]]

    # -------- load normal step prototypes --------
    step_file_map: Dict[int, List[Path]] = {}
    for p in sorted(normal_dir.rglob("*.npy")):
        m = ACTION_RE.search(p.stem)
        if m is None:
            continue
        action_id = int(m.group(1))
        if action_id in step_node_id_set:
            step_file_map.setdefault(action_id, []).append(p)

    step_prototypes = []
    missing_steps = []
    for sid in step_node_ids:
        paths = step_file_map.get(sid, [])
        if len(paths) == 0:
            missing_steps.append(sid)
        step_prototypes.append(_load_mean_feature(paths, feature_dim))

    if len(missing_steps) > 0:
        raise FileNotFoundError(
            f"Missing normal prototypes for steps {missing_steps} under {normal_dir}"
        )

    step_prototypes = torch.from_numpy(np.stack(step_prototypes, axis=0))

    # -------- load error prototypes --------
    err_file_map: Dict[Tuple[int, int], List[Path]] = {}
    valid_type_ids = set()

    for p in sorted(error_dir.rglob("*.npy")):
        m = ERROR_RE.search(p.stem)
        if m is None:
            continue
        action_id = int(m.group(1))
        type_id = int(m.group(2))

        # ignore background / extra class prototypes like action_-1_type_4_*
        if action_id not in step_node_id_set:
            continue

        valid_type_ids.add(type_id)
        err_file_map.setdefault((action_id, type_id), []).append(p)

    if len(valid_type_ids) == 0:
        raise FileNotFoundError(f"No valid step-specific error prototype files matched under {error_dir}")

    sorted_type_ids = sorted(valid_type_ids)
    typeid_to_col = {tid: idx for idx, tid in enumerate(sorted_type_ids)}
    max_type_id = len(sorted_type_ids)

    error_proto_arr = np.zeros((len(step_node_ids), max_type_id, feature_dim), dtype=np.float32)
    missing_pairs = []

    for i, sid in enumerate(step_node_ids):
        for tid in sorted_type_ids:
            paths = err_file_map.get((sid, tid), [])
            col = typeid_to_col[tid]
            if len(paths) == 0:
                missing_pairs.append((sid, tid))
                continue
            error_proto_arr[i, col] = _load_mean_feature(paths, feature_dim)

    error_prototypes = torch.from_numpy(error_proto_arr)

    return {
        "naming": naming,
        "num_classes": num_classes,
        "step_prototypes": step_prototypes,
        "error_prototypes": error_prototypes,
        "step_node_ids": step_node_ids,
        "predecessor_edges": predecessor_edges,
        "num_error_types": max_type_id,
        "type_ids": sorted_type_ids,
        "missing_error_pairs": missing_pairs,
        "task_dir": str(task_dir),
        "normal_dir": str(normal_dir),
        "error_dir": str(error_dir),
    }
