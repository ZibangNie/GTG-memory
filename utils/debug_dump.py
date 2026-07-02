"""Debug JSON dump helpers used by runner evaluation."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def dump_model_debug_json(
    save_dir,
    video_id,
    aux,
    pred,
    no_drop_pred_raw,
    label=None,
    type_label=None,
    drop_costs=None,
    node_drop_costs=None,
    no_drop_costs=None,
    no_node_drop_costs=None,
    dump_full_debug_tensors=False,
):
    if aux is None:
        return

    pred_np = np.asarray(pred)
    no_drop_np = np.asarray(no_drop_pred_raw)

    payload = {
        "video_id": str(video_id),
        "pred": pred_np.tolist(),
        "pred_error_mask": (pred_np == -1).astype(int).tolist(),
        "no_drop_pred_raw": no_drop_np.tolist(),
        "gt_label": to_serializable(label),
        "gt_type_label": to_serializable(type_label),
        "first_pass_error_ratio": float((pred_np == -1).mean()),
        "first_pass_unique_labels": sorted(np.unique(pred_np).tolist()),
        "no_drop_unique_labels": sorted(np.unique(no_drop_np).tolist()),
    }

    if drop_costs is not None:
        drop_costs_np = _as_numpy(drop_costs)
        payload["dp_drop_cost_t"] = drop_costs_np.tolist()
        payload["dp_drop_cost_mean"] = float(np.mean(drop_costs_np))
        payload["dp_drop_cost_min"] = float(np.min(drop_costs_np))
        payload["dp_drop_cost_max"] = float(np.max(drop_costs_np))

    if node_drop_costs is not None:
        node_drop_costs_np = _as_numpy(node_drop_costs)
        payload["dp_node_drop_cost"] = node_drop_costs_np.tolist()
        payload["dp_node_drop_cost_mean"] = float(np.mean(node_drop_costs_np))
        payload["dp_node_drop_cost_min"] = float(np.min(node_drop_costs_np))
        payload["dp_node_drop_cost_max"] = float(np.max(node_drop_costs_np))

    if no_drop_costs is not None:
        payload["dp_no_drop_cost_t"] = _as_numpy(no_drop_costs).tolist()

    if no_node_drop_costs is not None:
        payload["dp_no_node_drop_cost"] = _as_numpy(no_node_drop_costs).tolist()

    if "step_posteriors" in aux and aux["step_posteriors"] is not None:
        alpha = aux["step_posteriors"].detach().cpu().squeeze(0)
        alpha_entropy_t = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1)
        alpha_top1_t = alpha.max(dim=-1).values
        payload["alpha_entropy_t"] = alpha_entropy_t.tolist()
        payload["alpha_top1_t"] = alpha_top1_t.tolist()

    if "error_posteriors" in aux and aux["error_posteriors"] is not None:
        gamma = aux["error_posteriors"].detach().cpu().squeeze(0)
        payload["error_mass_t"] = gamma.sum(dim=(-1, -2)).tolist()

    if "candidate_mask" in aux and aux["candidate_mask"] is not None:
        cand = aux["candidate_mask"].detach().cpu().squeeze(0)
        payload["candidate_count_t"] = cand.float().sum(dim=-1).tolist()

    if "sem_long_gate_seq" in aux and aux["sem_long_gate_seq"] is not None:
        sem_long_gate = aux["sem_long_gate_seq"].detach().cpu().squeeze(0)
        payload["sem_long_gate_mean_t"] = sem_long_gate.mean(dim=-1).tolist()

    if "coverage_trace_seq" in aux and aux["coverage_trace_seq"] is not None:
        coverage = aux["coverage_trace_seq"].detach().cpu().squeeze(0)
        payload["coverage_mean_t"] = coverage.mean(dim=-1).tolist()
        payload["coverage_peak_t"] = coverage.max(dim=-1).values.tolist()

    if "uncertainty_trace_seq" in aux and aux["uncertainty_trace_seq"] is not None:
        unc = aux["uncertainty_trace_seq"].detach().cpu().squeeze(0)
        payload["uncertainty_norm_t"] = unc.norm(dim=-1).tolist()

    if "semantic_fuse_gate_seq" in aux and aux["semantic_fuse_gate_seq"] is not None:
        sfg = aux["semantic_fuse_gate_seq"].detach().cpu().squeeze(0)
        payload["semantic_fuse_gate_mean_t"] = sfg.mean(dim=-1).tolist()

    if "proto_gate" in aux and aux["proto_gate"] is not None:
        pg = aux["proto_gate"].detach().cpu().squeeze(0)
        payload["proto_gate_mean_t"] = pg.mean(dim=-1).tolist()

    if dump_full_debug_tensors:
        keep_keys = [
            "raw_step_logits",
            "step_posteriors",
            "error_posteriors",
            "candidate_mask",
            "aux_stats_seq",
            "sem_short_seq",
            "sem_long_seq",
            "sem_long_gate_seq",
            "coverage_trace_seq",
            "coverage_summary_seq",
            "uncertainty_trace_seq",
            "semantic_obs_seq",
            "semantic_fuse_gate_seq",
            "proto_gate",
            "main_logits",
            "final_logits",
            "step_node_ids",
            "extra_node_ids",
        ]
        for key in keep_keys:
            if key in aux:
                payload[key] = to_serializable(aux[key])
    else:
        if "step_node_ids" in aux:
            payload["step_node_ids"] = to_serializable(aux["step_node_ids"])
        if "extra_node_ids" in aux:
            payload["extra_node_ids"] = to_serializable(aux["extra_node_ids"])

    if "main_logits" in aux and "final_logits" in aux:
        _add_logit_debug(payload, aux)

    _write_json(_debug_output_path(save_dir, "model_debug", video_id), payload)


def dump_erm_debug_json(save_dir, video_id, erm_aux):
    """Save per-video ERM-v2 debug tensors for offline inspection."""
    if erm_aux is None:
        return

    payload = {key: to_serializable(value) for key, value in erm_aux.items()}
    _write_json(_debug_output_path(save_dir, "erm_debug", video_id), payload)


def _add_logit_debug(payload, aux):
    main_logits = aux["main_logits"].detach().cpu().squeeze(0)
    final_logits = aux["final_logits"].detach().cpu().squeeze(0)

    main_probs = torch.softmax(main_logits, dim=0)
    final_probs = torch.softmax(final_logits, dim=0)

    main_entropy_t = -(
        main_probs.clamp_min(1e-8) * main_probs.clamp_min(1e-8).log()
    ).sum(dim=0)
    final_entropy_t = -(
        final_probs.clamp_min(1e-8) * final_probs.clamp_min(1e-8).log()
    ).sum(dim=0)

    payload["main_entropy_t"] = main_entropy_t.tolist()
    payload["final_entropy_t"] = final_entropy_t.tolist()
    payload["main_top1_prob_t"] = main_probs.max(dim=0).values.tolist()
    payload["final_top1_prob_t"] = final_probs.max(dim=0).values.tolist()
    payload["main_vs_final_logit_abs_mean_t"] = (
        (final_logits - main_logits).abs().mean(dim=0).tolist()
    )

    step_node_ids = _long_index_tensor(aux.get("step_node_ids", []))
    extra_node_ids = _long_index_tensor(aux.get("extra_node_ids", []))

    if step_node_ids.numel() > 0:
        payload["main_step_mean_t"] = main_logits.index_select(0, step_node_ids).mean(dim=0).tolist()
        payload["final_step_mean_t"] = final_logits.index_select(0, step_node_ids).mean(dim=0).tolist()

    if extra_node_ids.numel() > 0:
        main_extra_mean_t = main_logits.index_select(0, extra_node_ids).mean(dim=0)
        final_extra_mean_t = final_logits.index_select(0, extra_node_ids).mean(dim=0)

        payload["main_extra_mean_t"] = main_extra_mean_t.tolist()
        payload["final_extra_mean_t"] = final_extra_mean_t.tolist()

        if step_node_ids.numel() > 0:
            main_step_mean_t = main_logits.index_select(0, step_node_ids).mean(dim=0)
            final_step_mean_t = final_logits.index_select(0, step_node_ids).mean(dim=0)
            payload["main_step_vs_extra_gap_t"] = (main_step_mean_t - main_extra_mean_t).tolist()
            payload["final_step_vs_extra_gap_t"] = (final_step_mean_t - final_extra_mean_t).tolist()


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _long_index_tensor(value):
    if value is None:
        return torch.as_tensor([], dtype=torch.long)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().long()
    return torch.as_tensor(value, dtype=torch.long)


def _debug_output_path(save_dir, subdir, video_id):
    out_dir = Path(save_dir) / "output" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / _safe_debug_filename(video_id)


def _safe_debug_filename(video_id):
    filename = str(video_id)
    for char in '/\\:*?"<>|':
        filename = filename.replace(char, "__")
    filename = filename.strip()
    if filename in {"", ".", ".."}:
        filename = "video"
    return f"{filename}.json"


def _write_json(path, payload):
    with path.open("w") as fp:
        json.dump(payload, fp)
