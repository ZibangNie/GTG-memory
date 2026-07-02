from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.semantic_memory import SemanticMemoryCore, SemanticObservationBuilder, SemanticPrototypeBank
from models.visual_memory import LinearNormGELU, SlowUpdateLongMemory


class VisualSemanticMemoryScorer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        real_num_classes: int,
        short_dim: int = 256,
        long_dim: int = 384,
        fusion_dim: int = 256,
        uncertainty_dim: int = 32,
        long_write_cap: float = 0.2,
        fusion_dropout: float = 0.1,
        tau_step: float = 0.07,
        tau_err: float = 0.07,
        rho_err: float = 0.85,
        k_max: int = 5,
        lambda_self: float = 1.0,
        lambda_succ: float = 0.8,
        lambda_pred: float = 0.4,
        lambda_topo: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.real_num_classes = real_num_classes
        self.short_dim = short_dim
        self.long_dim = long_dim
        self.fusion_dim = fusion_dim
        self.uncertainty_dim = uncertainty_dim

        # ---------- visual branch ----------
        self.base_projector = LinearNormGELU(feature_dim, fusion_dim)
        self.vis_short_cell = nn.GRUCell(fusion_dim, short_dim)
        self.vis_summary_mlp = nn.Sequential(
            nn.Linear(fusion_dim + short_dim, long_dim),
            nn.LayerNorm(long_dim),
            nn.GELU(),
            nn.Linear(long_dim, long_dim),
        )
        self.vis_long_updater = SlowUpdateLongMemory(long_dim, long_dim, write_cap=long_write_cap)
        self.vis_ctx_proj = nn.Sequential(
            nn.Linear(short_dim + long_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # ---------- semantic branch ----------
        self.prototype_bank = SemanticPrototypeBank(
            feature_dim=feature_dim,
            real_num_classes=real_num_classes,
        )
        self.semantic_obs = SemanticObservationBuilder(
            feature_dim=feature_dim,
            num_error_types=1,  # overwritten after configure_prototypes
            tau_step=tau_step,
            tau_err=tau_err,
            rho_err=rho_err,
            k_max=k_max,
            lambda_self=lambda_self,
            lambda_succ=lambda_succ,
            lambda_pred=lambda_pred,
            lambda_topo=lambda_topo,
            aux_dim=8,
        )
        self.semantic_core = SemanticMemoryCore(
            feature_dim=feature_dim,
            short_dim=short_dim,
            long_dim=long_dim,
            uncertainty_dim=uncertainty_dim,
            coverage_rate=0.05,
            uncertainty_rate=0.10,
            long_write_cap=long_write_cap,
            aux_dim=8,
        )
        self.semantic_ctx_proj = nn.Sequential(
            nn.Linear(short_dim + long_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.aux_proj = nn.Sequential(
            nn.Linear(8, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # ---------- asymmetric gated fusion ----------
        self.base_fuse_proj = LinearNormGELU(fusion_dim, fusion_dim)
        self.semantic_fuse_gate = nn.Linear(fusion_dim * 4, fusion_dim)  # channel-wise
        self.joint_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(512, fusion_dim),
        )
        self.joint_norm = nn.LayerNorm(fusion_dim)

        # ---------- heads ----------
        self.main_head = nn.Conv1d(fusion_dim, num_classes, kernel_size=1)

        self.proto_query = nn.Linear(fusion_dim, feature_dim)
        self.proto_adapter = nn.Linear(feature_dim, feature_dim)

        proto_steps = max(real_num_classes - 1, 1)
        self.proto_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2 + 8, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, proto_steps),
        )
        nn.init.constant_(self.proto_gate[-1].bias, -2.0)

    def configure_prototypes(
        self,
        step_prototypes: torch.Tensor,
        error_prototypes: torch.Tensor,
        step_node_ids: Optional[Sequence[int]] = None,
        predecessor_edges: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> None:
        self.prototype_bank.configure(
            step_prototypes=step_prototypes,
            error_prototypes=error_prototypes,
            step_node_ids=step_node_ids,
            predecessor_edges=predecessor_edges,
        )
        self.semantic_obs.num_error_types = self.prototype_bank.num_error_types

    def _scan_visual(self, base_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, tlen, _ = base_seq.shape
        device = base_seq.device
        dtype = base_seq.dtype

        vis_short = torch.zeros(bsz, self.short_dim, device=device, dtype=dtype)
        vis_long = torch.zeros(bsz, self.long_dim, device=device, dtype=dtype)

        vis_short_list, vis_summary_list, vis_long_list, vis_gate_list = [], [], [], []

        for t in range(tlen):
            base_t = base_seq[:, t, :]
            vis_short = self.vis_short_cell(base_t, vis_short)
            summary_t = self.vis_summary_mlp(torch.cat([base_t, vis_short], dim=-1))
            vis_long, vis_gate = self.vis_long_updater(summary_t, vis_long)

            vis_short_list.append(vis_short)
            vis_summary_list.append(summary_t)
            vis_long_list.append(vis_long)
            vis_gate_list.append(vis_gate)

        vis_short_seq = torch.stack(vis_short_list, dim=1)
        vis_summary_seq = torch.stack(vis_summary_list, dim=1)
        vis_long_seq = torch.stack(vis_long_list, dim=1)
        vis_long_gate_seq = torch.stack(vis_gate_list, dim=1)

        visual_ctx_seq = self.vis_ctx_proj(torch.cat([vis_short_seq, vis_long_seq], dim=-1))

        return {
            "base_seq": base_seq,
            "short_memory_seq": vis_short_seq,
            "summary_seq": vis_summary_seq,
            "long_memory_seq": vis_long_seq,
            "long_write_gate_seq": vis_long_gate_seq,
            "visual_ctx_seq": visual_ctx_seq,
        }

    def _forward_impl(self, features_bdt: torch.Tensor, return_aux: bool = False):
        if features_bdt.ndim != 3:
            raise ValueError(f"Expected [B, D, T], got {tuple(features_bdt.shape)}")
        if not self.prototype_bank.is_ready:
            raise RuntimeError("VisualSemanticMemoryScorer prototypes are not configured")

        # [B, D, T] -> [B, T, D]
        frame_seq = features_bdt.transpose(1, 2)

        # base + visual branch
        base_seq = self.base_projector(frame_seq)
        visual_dict = self._scan_visual(base_seq)

        # semantic observation + semantic memory
        sem_obs = self.semantic_obs(frame_seq=frame_seq, bank=self.prototype_bank)
        sem_mem = self.semantic_core(
            step_sem_obs=sem_obs["step_sem_obs"],
            error_sem_obs=sem_obs["error_sem_obs"],
            aux_stats_seq=sem_obs["aux_stats_seq"],
            step_posteriors=sem_obs["step_posteriors"],
        )

        semantic_ctx_seq = self.semantic_ctx_proj(
            torch.cat([sem_mem["sem_short_seq"], sem_mem["sem_long_seq"]], dim=-1)
        )
        aux_emb = self.aux_proj(sem_obs["aux_stats_seq"])

        # asymmetric gated fusion
        base_fused = self.base_fuse_proj(base_seq)
        visual_ctx = visual_dict["visual_ctx_seq"]

        fuse_gate = torch.sigmoid(
            self.semantic_fuse_gate(
                torch.cat([base_fused, visual_ctx, semantic_ctx_seq, aux_emb], dim=-1)
            )
        )  # [B, T, fusion_dim]

        gated_semantic_ctx = fuse_gate * semantic_ctx_seq
        fusion_delta = self.joint_fusion(torch.cat([base_fused, visual_ctx, gated_semantic_ctx], dim=-1))
        joint_fused = self.joint_norm(base_fused + fusion_delta)  # [B, T, fusion_dim]

        # main head
        frame_features = joint_fused.transpose(1, 2)  # [B, fusion_dim, T]
        main_logits = self.main_head(frame_features)  # [B, C_gtg, T]

        # prototype head for real step nodes only
        step_proto = self.prototype_bank.step_prototypes.to(frame_seq.device, dtype=frame_seq.dtype)
        step_proto = self.proto_adapter(step_proto)
        step_proto = step_proto / torch.clamp(torch.norm(step_proto, p=2, dim=-1, keepdim=True), min=1e-8)

        q_proto = self.proto_query(joint_fused)
        q_proto = q_proto / torch.clamp(torch.norm(q_proto, p=2, dim=-1, keepdim=True), min=1e-8)

        proto_logits_steps = torch.einsum("btd,sd->bts", q_proto, step_proto) / 0.07  # [B, T, S]
        proto_gate = torch.sigmoid(
            self.proto_gate(torch.cat([joint_fused, semantic_ctx_seq, sem_obs["aux_stats_seq"]], dim=-1))
        )  # [B, T, S]

        final_logits = main_logits.clone()
        step_boost = proto_gate * proto_logits_steps  # [B, T, S]
        step_node_ids = self.prototype_bank.step_node_ids.to(frame_seq.device)

        for step_idx, node_id in enumerate(step_node_ids.tolist()):
            if 0 <= node_id < self.num_classes:
                final_logits[:, node_id, :] = final_logits[:, node_id, :] + step_boost[:, :, step_idx]

        if not return_aux:
            return final_logits, frame_features

        aux = {
            **visual_dict,
            **sem_obs,
            **sem_mem,
            "step_node_ids": step_node_ids.detach().cpu(),
            "extra_node_ids": self.prototype_bank.extra_node_ids(self.num_classes).detach().cpu(),
            "semantic_ctx_seq": semantic_ctx_seq,
            "semantic_fuse_gate_seq": fuse_gate,
            "joint_fused_seq": joint_fused,
            "main_logits": main_logits,
            "proto_logits": proto_logits_steps,
            "proto_gate": proto_gate,
            "final_logits": final_logits,
            "fused_seq": joint_fused,
        }
        return final_logits, frame_features, aux

    def forward(self, features_bdt: torch.Tensor):
        return self._forward_impl(features_bdt, return_aux=False)

    def forward_with_aux(self, features_bdt: torch.Tensor):
        return self._forward_impl(features_bdt, return_aux=True)
