from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.memory_utils import (
    build_predecessor_index_lists,
    cosine_normalize,
    cumulative_mass_topk,
    ema_update,
    entropy_from_probs,
    safe_log,
)
from models.visual_memory import SlowUpdateLongMemory


class SemanticPrototypeBank(nn.Module):
    def __init__(self, feature_dim: int, real_num_classes: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.real_num_classes = real_num_classes

        self.register_buffer("step_prototypes", torch.empty(0, feature_dim), persistent=False)
        self.register_buffer("error_prototypes", torch.empty(0, 0, feature_dim), persistent=False)
        self.register_buffer("step_node_ids", torch.empty(0, dtype=torch.long), persistent=False)

        self.predecessor_index_lists: List[List[int]] = []
        self.num_error_types: int = 0

    @property
    def is_ready(self) -> bool:
        return self.step_prototypes.numel() > 0 and self.error_prototypes.numel() > 0

    @property
    def num_real_steps(self) -> int:
        return int(self.step_prototypes.shape[0])

    def configure(
        self,
        step_prototypes: torch.Tensor,
        error_prototypes: torch.Tensor,
        step_node_ids: Optional[Sequence[int]] = None,
        predecessor_edges: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> None:
        if step_prototypes.ndim != 2:
            raise ValueError(f"step_prototypes must be [S, D], got {tuple(step_prototypes.shape)}")
        if error_prototypes.ndim != 3:
            raise ValueError(f"error_prototypes must be [S, M, D], got {tuple(error_prototypes.shape)}")
        if step_prototypes.shape[0] != error_prototypes.shape[0]:
            raise ValueError("step/error prototype step dimension mismatch")
        if step_prototypes.shape[1] != self.feature_dim or error_prototypes.shape[2] != self.feature_dim:
            raise ValueError("prototype feature dim mismatch")

        num_steps = int(step_prototypes.shape[0])
        if step_node_ids is None:
            step_node_ids = list(range(1, num_steps + 1))

        step_node_ids = torch.as_tensor(step_node_ids, dtype=torch.long)
        if step_node_ids.ndim != 1 or step_node_ids.numel() != num_steps:
            raise ValueError("step_node_ids must have length S")

        self.step_prototypes = step_prototypes.detach().clone()
        self.error_prototypes = error_prototypes.detach().clone()
        self.step_node_ids = step_node_ids
        self.num_error_types = int(error_prototypes.shape[1])
        self.predecessor_index_lists = build_predecessor_index_lists(num_steps, predecessor_edges)

    def extra_node_ids(self, num_classes: int) -> torch.Tensor:
        if self.step_node_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        all_ids = torch.arange(num_classes, dtype=torch.long, device=self.step_node_ids.device)
        mask = torch.ones_like(all_ids, dtype=torch.bool)
        valid_ids = self.step_node_ids[(self.step_node_ids >= 0) & (self.step_node_ids < num_classes)]
        mask[valid_ids] = False
        return all_ids[mask]


class SemanticObservationBuilder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_error_types: int,
        tau_step: float = 0.07,
        tau_err: float = 0.07,
        rho_err: float = 0.85,
        k_max: int = 5,
        lambda_self: float = 1.0,
        lambda_succ: float = 0.8,
        lambda_pred: float = 0.4,
        lambda_topo: float = 0.5,
        aux_dim: int = 8,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_error_types = num_error_types
        self.tau_step = tau_step
        self.tau_err = tau_err
        self.rho_err = rho_err
        self.k_max = k_max
        self.lambda_self = lambda_self
        self.lambda_succ = lambda_succ
        self.lambda_pred = lambda_pred
        self.lambda_topo = lambda_topo
        self.aux_dim = aux_dim

        self.step_query = nn.Linear(feature_dim, feature_dim)
        self.err_query = nn.Linear(feature_dim, feature_dim)
        self.step_adapter = nn.Linear(feature_dim, feature_dim)
        self.err_adapter = nn.Linear(feature_dim, feature_dim)

    def _build_topology_bias(
        self,
        prev_alpha: torch.Tensor,
        predecessor_index_lists: Sequence[Sequence[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = prev_alpha.device
        _, nsteps = prev_alpha.shape

        topo_bias = self.lambda_self * prev_alpha
        topo_mass = self.lambda_self * prev_alpha

        succ_from_pred = torch.zeros_like(prev_alpha)
        pred_mass = torch.zeros_like(prev_alpha)

        for step_idx in range(nsteps):
            preds = list(predecessor_index_lists[step_idx]) if step_idx < len(predecessor_index_lists) else []
            if len(preds) == 0:
                continue

            pred_tensor = torch.as_tensor(preds, dtype=torch.long, device=device)
            pred_vals = prev_alpha.index_select(dim=1, index=pred_tensor)

            pred_mean = pred_vals.mean(dim=1)
            pred_mass[:, step_idx] = pred_mean
            topo_bias[:, step_idx] = topo_bias[:, step_idx] + self.lambda_pred * pred_mean

            pred_max = pred_vals.max(dim=1).values
            succ_from_pred[:, step_idx] = pred_max
            topo_bias[:, step_idx] = topo_bias[:, step_idx] + self.lambda_succ * pred_max

        topo_mass = topo_mass + self.lambda_succ * succ_from_pred + self.lambda_pred * pred_mass
        topo_mass = torch.clamp(topo_mass, min=1e-8)
        topo_log_bias = self.lambda_topo * safe_log(topo_mass)
        return topo_log_bias, topo_mass

    def forward(
        self,
        frame_seq: torch.Tensor,
        bank: SemanticPrototypeBank,
    ) -> Dict[str, torch.Tensor]:
        if not bank.is_ready:
            raise RuntimeError("SemanticPrototypeBank has not been configured")

        step_proto = cosine_normalize(self.step_adapter(bank.step_prototypes), dim=-1)   # [S, D]
        err_proto = cosine_normalize(self.err_adapter(bank.error_prototypes), dim=-1)     # [S, M, D]

        q_step = cosine_normalize(self.step_query(frame_seq), dim=-1)                     # [B, T, D]
        q_err = cosine_normalize(self.err_query(frame_seq), dim=-1)                       # [B, T, D]

        raw_step_logits = torch.einsum("btd,sd->bts", q_step, step_proto) / self.tau_step # [B, T, S]

        alpha_list = []
        topo_mass_list = []
        prev_alpha = torch.full(
            (frame_seq.shape[0], bank.num_real_steps),
            fill_value=1.0 / max(bank.num_real_steps, 1),
            device=frame_seq.device,
            dtype=frame_seq.dtype,
        )

        for t in range(frame_seq.shape[1]):
            topo_log_bias, topo_mass = self._build_topology_bias(prev_alpha, bank.predecessor_index_lists)
            logits_t = raw_step_logits[:, t, :] + topo_log_bias
            alpha_t = torch.softmax(logits_t, dim=-1)

            alpha_list.append(alpha_t)
            topo_mass_list.append(topo_mass)
            prev_alpha = alpha_t

        alpha = torch.stack(alpha_list, dim=1)            # [B, T, S]
        topo_mass = torch.stack(topo_mass_list, dim=1)    # [B, T, S]

        err_logits = torch.einsum("btd,smd->btsm", q_err, err_proto) / self.tau_err  # [B, T, S, M]
        candidate_mask = cumulative_mass_topk(alpha, self.rho_err, self.k_max)       # [B, T, S]

        masked_err_logits = err_logits.masked_fill(~candidate_mask.unsqueeze(-1), -1e4)
        beta = torch.softmax(masked_err_logits, dim=-1)                               # [B, T, S, M]
        gamma = alpha.unsqueeze(-1) * beta * candidate_mask.unsqueeze(-1).float()     # [B, T, S, M]

        step_sem_obs = torch.einsum("bts,sd->btd", alpha, step_proto)                 # [B, T, D]
        err_sem_obs = torch.einsum("btsm,smd->btd", gamma, err_proto)                 # [B, T, D]

        alpha_entropy = entropy_from_probs(alpha, dim=-1)                             # [B, T]
        alpha_top2 = torch.topk(alpha, k=min(2, alpha.shape[-1]), dim=-1).values
        alpha_top1 = alpha_top2[..., 0]
        alpha_gap = alpha_top2[..., 0] - alpha_top2[..., 1] if alpha_top2.shape[-1] > 1 else alpha_top1

        error_mass = gamma.sum(dim=(-1, -2))                                          # [B, T]
        beta_entropy = entropy_from_probs(beta, dim=-1)                               # [B, T, S]
        error_entropy = (beta_entropy * alpha).sum(dim=-1)                            # [B, T]

        topo_mass_mean = topo_mass.mean(dim=-1)                                       # [B, T]
        topo_mass_peak = topo_mass.max(dim=-1).values                                 # [B, T]
        candidate_count = candidate_mask.float().sum(dim=-1)                          # [B, T]

        aux_stats = torch.stack(
            [
                alpha_entropy,
                alpha_top1,
                alpha_gap,
                error_mass,
                error_entropy,
                topo_mass_mean,
                topo_mass_peak,
                candidate_count / max(float(self.k_max), 1.0),
            ],
            dim=-1,
        )                                                                              # [B, T, 8]

        return {
            "step_posteriors": alpha,
            "error_posteriors": gamma,
            "error_type_posteriors": beta,
            "step_sem_obs": step_sem_obs,
            "error_sem_obs": err_sem_obs,
            "aux_stats_seq": aux_stats,
            "candidate_mask": candidate_mask,
            "raw_step_logits": raw_step_logits,
        }


class SemanticMemoryCore(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        short_dim: int = 256,
        long_dim: int = 384,
        uncertainty_dim: int = 32,
        coverage_rate: float = 0.05,
        uncertainty_rate: float = 0.10,
        long_write_cap: float = 0.2,
        aux_dim: int = 8,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.short_dim = short_dim
        self.long_dim = long_dim
        self.uncertainty_dim = uncertainty_dim
        self.coverage_rate = coverage_rate
        self.uncertainty_rate = uncertainty_rate
        self.aux_dim = aux_dim

        # 新增：用固定维度摘要 coverage，避免完全丢掉 coverage 信息
        self.coverage_summary_dim = 3

        self.obs_fuser = nn.Sequential(
            nn.Linear(feature_dim * 2 + aux_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self.short_input_proj = nn.Sequential(
            nn.Linear(feature_dim + aux_dim + uncertainty_dim + self.coverage_summary_dim, short_dim),
            nn.LayerNorm(short_dim),
            nn.GELU(),
        )
        self.short_cell = nn.GRUCell(short_dim, short_dim)

        self.unc_proj = nn.Sequential(
            nn.Linear(aux_dim, uncertainty_dim),
            nn.LayerNorm(uncertainty_dim),
            nn.GELU(),
        )
        self.conf_gate = nn.Linear(aux_dim, 1)

        self.long_summary = nn.Sequential(
            nn.Linear(short_dim + uncertainty_dim + aux_dim + self.coverage_summary_dim, long_dim),
            nn.LayerNorm(long_dim),
            nn.GELU(),
            nn.Linear(long_dim, long_dim),
        )
        self.long_updater = SlowUpdateLongMemory(long_dim, long_dim, write_cap=long_write_cap)

    def _summarize_coverage(self, coverage: torch.Tensor) -> torch.Tensor:
        """
        coverage: [B, S]
        returns:  [B, 3] = [max, mean, normalized_entropy]
        """
        cov_sum = coverage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        cov_prob = coverage / cov_sum

        cov_max = coverage.max(dim=-1).values
        cov_mean = coverage.mean(dim=-1)

        num_steps = max(int(coverage.shape[-1]), 2)
        cov_entropy = entropy_from_probs(cov_prob.clamp_min(1e-8), dim=-1)
        cov_entropy = cov_entropy / math.log(num_steps)

        return torch.stack([cov_max, cov_mean, cov_entropy], dim=-1)

    def forward(
        self,
        step_sem_obs: torch.Tensor,
        error_sem_obs: torch.Tensor,
        aux_stats_seq: torch.Tensor,
        step_posteriors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz, tlen, _ = step_sem_obs.shape
        nsteps = step_posteriors.shape[-1]
        device = step_sem_obs.device
        dtype = step_sem_obs.dtype

        sem_short = torch.zeros(bsz, self.short_dim, device=device, dtype=dtype)
        sem_long = torch.zeros(bsz, self.long_dim, device=device, dtype=dtype)
        coverage = torch.zeros(bsz, nsteps, device=device, dtype=dtype)
        uncertainty = torch.zeros(bsz, self.uncertainty_dim, device=device, dtype=dtype)

        sem_short_list, sem_long_list, sem_long_gate_list = [], [], []
        coverage_list, uncertainty_list, fused_obs_list = [], [], []
        coverage_summary_list = []

        for t in range(tlen):
            alpha_t = step_posteriors[:, t, :]
            step_obs_t = step_sem_obs[:, t, :]
            err_obs_t = error_sem_obs[:, t, :]
            aux_t = aux_stats_seq[:, t, :]

            fused_obs_t = self.obs_fuser(torch.cat([step_obs_t, err_obs_t, aux_t], dim=-1))

            unc_raw_t = self.unc_proj(aux_t)
            uncertainty = ema_update(uncertainty, unc_raw_t, self.uncertainty_rate)
            coverage = ema_update(coverage, alpha_t, self.coverage_rate)

            coverage_summary_t = self._summarize_coverage(coverage)
            conf_t = torch.sigmoid(self.conf_gate(aux_t))  # [B, 1]

            # short：真的吃到 coverage 摘要
            short_in_t = self.short_input_proj(
                torch.cat([fused_obs_t, aux_t, uncertainty, coverage_summary_t], dim=-1)
            )
            short_in_t = conf_t * short_in_t
            sem_short = self.short_cell(short_in_t, sem_short)

            # long：真的用 conf_t 控实际写入，不只是记录个假的 gate
            long_summary_t = self.long_summary(
                torch.cat([sem_short, uncertainty, aux_t, coverage_summary_t], dim=-1)
            )
            sem_long, sem_long_gate = self.long_updater(
                long_summary_t,
                sem_long,
                gate_scale=conf_t,
            )

            sem_short_list.append(sem_short)
            sem_long_list.append(sem_long)
            sem_long_gate_list.append(sem_long_gate)
            coverage_list.append(coverage)
            uncertainty_list.append(uncertainty)
            fused_obs_list.append(fused_obs_t)
            coverage_summary_list.append(coverage_summary_t)

        return {
            "sem_short_seq": torch.stack(sem_short_list, dim=1),
            "sem_long_seq": torch.stack(sem_long_list, dim=1),
            "sem_long_gate_seq": torch.stack(sem_long_gate_list, dim=1),
            "coverage_trace_seq": torch.stack(coverage_list, dim=1),
            "coverage_summary_seq": torch.stack(coverage_summary_list, dim=1),
            "uncertainty_trace_seq": torch.stack(uncertainty_list, dim=1),
            "semantic_obs_seq": torch.stack(fused_obs_list, dim=1),
        }
