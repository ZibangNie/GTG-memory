from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCandidateERM(nn.Module):
    """
    Inference-only ERM v2:
      - multi-candidate soft step conditioning
      - memory-enhanced query
      - prototype-based joint similarity aggregation

    Inputs are packed in a dict by Runner.
    """

    def __init__(
        self,
        bg_idx: int,
        addition_idx: int,
        num_types: int,
        step_prototypes: torch.Tensor,      # [S, D]
        error_prototypes: torch.Tensor,     # [S, M, D]
        step_node_ids: Sequence[int],       # len S
        type_ids: Sequence[int],            # len M, actual error type ids
        rho: float = 0.85,
        kmax_sem: int = 5,
        kmax_final: int = 6,
        lambda_anchor: float = 0.8,
        lambda_nb: float = 0.3,
        lambda_cov: float = 0.2,
        lambda_vis: float = 0.5,
        lambda_sem: float = 0.7,
        lambda_obs: float = 0.3,
        similarity_scale: float = 20.0,
        smooth_window: int = 5,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.bg_idx = int(bg_idx)
        self.addition_idx = int(addition_idx)
        self.num_types = int(num_types)

        self.rho = float(rho)
        self.kmax_sem = int(kmax_sem)
        self.kmax_final = int(kmax_final)

        self.lambda_anchor = float(lambda_anchor)
        self.lambda_nb = float(lambda_nb)
        self.lambda_cov = float(lambda_cov)

        self.lambda_vis = float(lambda_vis)
        self.lambda_sem = float(lambda_sem)
        self.lambda_obs = float(lambda_obs)

        self.similarity_scale = float(similarity_scale)
        self.smooth_window = int(smooth_window)
        self.eps = float(eps)

        step_node_ids_t = torch.as_tensor(step_node_ids, dtype=torch.long)
        type_ids_t = torch.as_tensor(type_ids, dtype=torch.long)

        self.register_buffer(
            "step_prototypes",
            self._l2_normalize(step_prototypes.float(), dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "error_prototypes",
            self._l2_normalize(error_prototypes.float(), dim=-1),
            persistent=False,
        )
        self.register_buffer("step_node_ids", step_node_ids_t, persistent=False)
        self.register_buffer("type_ids", type_ids_t, persistent=False)

        self.step_node_id_to_local = {
            int(node_id): idx for idx, node_id in enumerate(step_node_ids_t.tolist())
        }

    @staticmethod
    def _l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / torch.clamp(torch.norm(x, p=2, dim=dim, keepdim=True), min=1e-8)

    @staticmethod
    def _to_long_tensor(x, device: torch.device) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.long)
        return torch.as_tensor(x, dtype=torch.long, device=device)

    @staticmethod
    def _to_float_tensor(x, device: torch.device) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _max_filter_1d(self, scores_ct: torch.Tensor) -> torch.Tensor:
        """
        scores_ct: [C, T]
        """
        if self.smooth_window <= 1:
            return scores_ct

        left = self.smooth_window // 2
        right = self.smooth_window - 1 - left

        x = scores_ct.unsqueeze(0)  # [1, C, T]
        x = F.pad(x, (left, right), mode="replicate")
        x = F.max_pool1d(x, kernel_size=self.smooth_window, stride=1)
        return x.squeeze(0)

    def _resolve_anchor_node(
        self,
        raw_anchor_node: int,
        alpha_t: torch.Tensor,
    ) -> Tuple[int, bool]:
        """
        Returns:
            anchor_node_id, is_fallback
        """
        raw_anchor_node = int(raw_anchor_node)
        if raw_anchor_node in self.step_node_id_to_local:
            return raw_anchor_node, False

        best_local = int(torch.argmax(alpha_t).item())
        best_node_id = int(self.step_node_ids[best_local].item())
        return best_node_id, True

    def _semantic_candidate_locals(self, alpha_t: torch.Tensor) -> List[int]:
        sorted_vals, sorted_idx = torch.sort(alpha_t, descending=True)

        out: List[int] = []
        mass = 0.0
        for val, idx in zip(sorted_vals.tolist(), sorted_idx.tolist()):
            out.append(int(idx))
            mass += float(val)
            if mass >= self.rho or len(out) >= self.kmax_sem:
                break

        if len(out) == 0:
            out = [int(torch.argmax(alpha_t).item())]
        return out

    def _graph_neighbor_nodes(self, graph, anchor_node_id: int) -> Set[int]:
        if graph is None:
            return set()

        if anchor_node_id not in graph.nodes:
            return set()

        preds = {int(x) for x in graph.predecessors(anchor_node_id)}
        succs = {int(x) for x in graph.successors(anchor_node_id)}
        return preds.union(succs)

    def _build_query(
        self,
        t: int,
        frame_features: torch.Tensor,              # [T, D]
        vis_short_seq: Optional[torch.Tensor],     # [T, D] or None
        sem_short_seq: Optional[torch.Tensor],     # [T, D] or None
        semantic_obs_seq: Optional[torch.Tensor],  # [T, D] or None
        uncertainty_trace_seq: Optional[torch.Tensor],  # [T, U] or None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        q_frame = self._l2_normalize(frame_features[t], dim=-1)

        if vis_short_seq is not None:
            q_vis = self._l2_normalize(vis_short_seq[t], dim=-1)
        else:
            q_vis = torch.zeros_like(q_frame)

        if sem_short_seq is not None:
            q_sem = self._l2_normalize(sem_short_seq[t], dim=-1)
        else:
            q_sem = torch.zeros_like(q_frame)

        if semantic_obs_seq is not None:
            q_obs = self._l2_normalize(semantic_obs_seq[t], dim=-1)
        else:
            q_obs = torch.zeros_like(q_frame)

        if uncertainty_trace_seq is not None:
            unc_t = uncertainty_trace_seq[t]
            unc_norm = unc_t.norm(p=2) / max(float(unc_t.numel()) ** 0.5, 1.0)
            sem_conf = torch.clamp(torch.exp(-unc_norm), min=0.25, max=1.0)
        else:
            sem_conf = torch.tensor(1.0, device=q_frame.device, dtype=q_frame.dtype)

        q = q_frame + self.lambda_vis * q_vis + sem_conf * (
            self.lambda_sem * q_sem + self.lambda_obs * q_obs
        )
        q = self._l2_normalize(q, dim=-1)

        info = {
            "q_frame_norm": float(q_frame.norm().item()),
            "q_vis_norm": float(q_vis.norm().item()),
            "q_sem_norm": float(q_sem.norm().item()),
            "q_obs_norm": float(q_obs.norm().item()),
            "q_final_norm": float(q.norm().item()),
            "sem_conf": float(sem_conf.item()),
        }
        return q, info

    def _candidate_scores(
        self,
        candidate_locals: List[int],
        anchor_node_id: int,
        sem_local_set: Set[int],
        topo_local_set: Set[int],
        alpha_t: torch.Tensor,
        coverage_t: Optional[torch.Tensor],
    ) -> Tuple[List[int], torch.Tensor, List[int]]:
        """
        Returns:
            kept_candidate_locals
            candidate_weights [K]
            candidate_flags list[int], bitmask:
                1 = anchor
                2 = semantic
                4 = topology
        """
        raw_scores = []
        flags = []

        for local_idx in candidate_locals:
            node_id = int(self.step_node_ids[local_idx].item())

            flag = 0
            if node_id == anchor_node_id:
                flag |= 1
            if local_idx in sem_local_set:
                flag |= 2
            if local_idx in topo_local_set:
                flag |= 4

            score = torch.log(alpha_t[local_idx].clamp_min(self.eps))
            if flag & 1:
                score = score + self.lambda_anchor
            if flag & 4:
                score = score + self.lambda_nb
            if coverage_t is not None:
                score = score + self.lambda_cov * (1.0 - coverage_t[local_idx].clamp(0.0, 1.0))

            raw_scores.append(score)
            flags.append(flag)

        raw_scores_t = torch.stack(raw_scores, dim=0)

        if len(candidate_locals) > self.kmax_final:
            top_idx = torch.topk(raw_scores_t, k=self.kmax_final, dim=0).indices.tolist()
            candidate_locals = [candidate_locals[i] for i in top_idx]
            flags = [flags[i] for i in top_idx]
            raw_scores_t = raw_scores_t[top_idx]

        weights = torch.softmax(raw_scores_t, dim=0)
        return candidate_locals, weights, flags

    def forward(self, erm_inputs: Dict[str, object]):
        pred, type_pred, error_pred, _ = self.forward_with_aux(erm_inputs)
        return pred, type_pred, error_pred

    def forward_with_aux(self, erm_inputs: Dict[str, object]):
        device = self.step_prototypes.device

        pred_t = self._to_long_tensor(erm_inputs["pred"], device=device)                     # [T]
        no_drop_pred_t = self._to_long_tensor(erm_inputs["no_drop_pred"], device=device)     # [T]

        frame_features = self._to_float_tensor(erm_inputs["frame_features"], device=device)   # [T, D]
        step_posteriors = self._to_float_tensor(erm_inputs["step_posteriors"], device=device) # [T, S]

        vis_short_seq = erm_inputs.get("vis_short_seq", None)
        if vis_short_seq is not None:
            vis_short_seq = self._to_float_tensor(vis_short_seq, device=device)

        sem_short_seq = erm_inputs.get("sem_short_seq", None)
        if sem_short_seq is not None:
            sem_short_seq = self._to_float_tensor(sem_short_seq, device=device)

        semantic_obs_seq = erm_inputs.get("semantic_obs_seq", None)
        if semantic_obs_seq is not None:
            semantic_obs_seq = self._to_float_tensor(semantic_obs_seq, device=device)

        coverage_trace_seq = erm_inputs.get("coverage_trace_seq", None)
        if coverage_trace_seq is not None:
            coverage_trace_seq = self._to_float_tensor(coverage_trace_seq, device=device)

        uncertainty_trace_seq = erm_inputs.get("uncertainty_trace_seq", None)
        if uncertainty_trace_seq is not None:
            uncertainty_trace_seq = self._to_float_tensor(uncertainty_trace_seq, device=device)

        graph = erm_inputs.get("graph", None)

        tlen = int(pred_t.shape[0])
        num_err_proto = int(self.type_ids.numel())

        error_mask = pred_t == -1

        type_scores = torch.zeros(self.num_types + 1, tlen, device=device, dtype=torch.float32)
        type_scores[0, ~error_mask] = 1.0

        candidate_ids_seq = torch.full(
            (tlen, self.kmax_final), -1, dtype=torch.long, device=device
        )
        candidate_weights_seq = torch.zeros(
            (tlen, self.kmax_final), dtype=torch.float32, device=device
        )
        candidate_flags_seq = torch.zeros(
            (tlen, self.kmax_final), dtype=torch.long, device=device
        )
        candidate_count_seq = torch.zeros(
            (tlen,), dtype=torch.long, device=device
        )

        anchor_step_seq = torch.full((tlen,), -1, dtype=torch.long, device=device)
        anchor_fallback_seq = torch.zeros((tlen,), dtype=torch.float32, device=device)

        q_component_norms = torch.zeros((tlen, 6), dtype=torch.float32, device=device)
        joint_scores_seq = torch.zeros(
            (tlen, self.kmax_final, num_err_proto), dtype=torch.float32, device=device
        )
        aggregated_scores_seq = torch.zeros(
            (tlen, self.num_types + 1), dtype=torch.float32, device=device
        )

        for t in range(tlen):
            if not bool(error_mask[t].item()):
                continue

            alpha_t = step_posteriors[t]  # [S]
            coverage_t = coverage_trace_seq[t] if coverage_trace_seq is not None else None

            raw_anchor_node = int(no_drop_pred_t[t].item())
            anchor_node_id, is_fallback = self._resolve_anchor_node(raw_anchor_node, alpha_t)
            anchor_local = self.step_node_id_to_local[anchor_node_id]

            sem_locals = self._semantic_candidate_locals(alpha_t)
            sem_local_set = set(sem_locals)

            topo_nodes = self._graph_neighbor_nodes(graph, anchor_node_id)
            topo_locals = [
                self.step_node_id_to_local[n]
                for n in topo_nodes
                if n in self.step_node_id_to_local
            ]
            topo_local_set = set(topo_locals)

            candidate_local_set = set(sem_locals)
            candidate_local_set.update(topo_locals)
            candidate_local_set.add(anchor_local)

            candidate_locals = sorted(candidate_local_set)

            candidate_locals, candidate_weights, candidate_flags = self._candidate_scores(
                candidate_locals=candidate_locals,
                anchor_node_id=anchor_node_id,
                sem_local_set=sem_local_set,
                topo_local_set=topo_local_set,
                alpha_t=alpha_t,
                coverage_t=coverage_t,
            )

            q_t, q_info = self._build_query(
                t=t,
                frame_features=frame_features,
                vis_short_seq=vis_short_seq,
                sem_short_seq=sem_short_seq,
                semantic_obs_seq=semantic_obs_seq,
                uncertainty_trace_seq=uncertainty_trace_seq,
            )

            agg_scores = torch.zeros(num_err_proto, dtype=torch.float32, device=device)

            for rank, (local_idx, weight, flag) in enumerate(
                zip(candidate_locals, candidate_weights, candidate_flags)
            ):
                normal_proto = self.step_prototypes[local_idx]            # [D]
                err_proto = self.error_prototypes[local_idx]              # [M, D]

                joint_proto = normal_proto.unsqueeze(0) * err_proto       # [M, D]
                score_vec = torch.sigmoid(
                    self.similarity_scale * torch.matmul(joint_proto, q_t)
                )                                                         # [M]

                agg_scores = agg_scores + weight * score_vec

                candidate_ids_seq[t, rank] = int(self.step_node_ids[local_idx].item())
                candidate_weights_seq[t, rank] = float(weight.item())
                candidate_flags_seq[t, rank] = int(flag)
                joint_scores_seq[t, rank, :] = score_vec

            if torch.max(agg_scores) <= self.eps:
                if 0 < self.addition_idx <= self.num_types:
                    type_scores[self.addition_idx, t] = 1.0
            else:
                for proto_col, type_id in enumerate(self.type_ids.tolist()):
                    if 0 < int(type_id) <= self.num_types:
                        type_scores[int(type_id), t] = agg_scores[proto_col]

            candidate_count_seq[t] = len(candidate_locals)
            anchor_step_seq[t] = anchor_node_id
            anchor_fallback_seq[t] = 1.0 if is_fallback else 0.0
            q_component_norms[t, 0] = q_info["q_frame_norm"]
            q_component_norms[t, 1] = q_info["q_vis_norm"]
            q_component_norms[t, 2] = q_info["q_sem_norm"]
            q_component_norms[t, 3] = q_info["q_obs_norm"]
            q_component_norms[t, 4] = q_info["q_final_norm"]
            q_component_norms[t, 5] = q_info["sem_conf"]

            aggregated_scores_seq[t, :] = type_scores[:, t]

        smoothed_scores = type_scores.clone()
        if self.smooth_window > 1:
            smoothed_scores[1:, :] = self._max_filter_1d(type_scores[1:, :])

        type_pred = torch.argmax(smoothed_scores, dim=0)
        type_pred[~error_mask] = 0

        # force every predicted error frame to keep a non-normal type
        fallback_normal_mask = error_mask & (type_pred == 0)
        if torch.any(fallback_normal_mask):
            type_pred[fallback_normal_mask] = self.addition_idx

        # ED stays owned by GTG2Vid in v1 ERM
        error_pred = error_mask.long()

        aux = {
            "candidate_ids_seq": candidate_ids_seq.detach().cpu(),
            "candidate_weights_seq": candidate_weights_seq.detach().cpu(),
            "candidate_flags_seq": candidate_flags_seq.detach().cpu(),
            "candidate_count_seq": candidate_count_seq.detach().cpu(),
            "anchor_step_seq": anchor_step_seq.detach().cpu(),
            "anchor_fallback_seq": anchor_fallback_seq.detach().cpu(),
            "q_component_norms": q_component_norms.detach().cpu(),
            "joint_scores_seq": joint_scores_seq.detach().cpu(),
            "aggregated_scores_seq": aggregated_scores_seq.detach().cpu(),
            "smoothed_scores_seq": smoothed_scores.detach().cpu(),
            "type_ids": self.type_ids.detach().cpu(),
            "step_node_ids": self.step_node_ids.detach().cpu(),
        }

        return (
            pred_t.detach().cpu().numpy(),
            type_pred.detach().cpu().numpy(),
            error_pred.detach().cpu().numpy(),
            aux,
        )