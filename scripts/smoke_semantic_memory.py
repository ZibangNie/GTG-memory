import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running:
#   python scripts/smoke_semantic_memory.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from models.models import ASDiffusionBackbone


def main():
    torch.manual_seed(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    time_steps = 24
    feature_dim = 256
    real_num_classes = 6   # 0 background, 1..5 real steps
    num_classes = 11       # includes GTG extra/addition-related nodes
    num_types = 5

    model = ASDiffusionBackbone(
        input_dim=feature_dim,
        num_classes=num_classes,
        real_num_classes=real_num_classes,
        num_types=num_types,
        addition_idx=0,
        device=device,
        bg_w=2.0,
        use_visual_memory=True,
        use_semantic_memory=True,
        short_dim=256,
        long_dim=384,
        fusion_dim=256,
        long_write_cap=0.2,
        fusion_dropout=0.1,
        uncertainty_dim=32,
        tau_step=0.07,
        tau_err=0.07,
        rho_err=0.85,
        error_candidate_max_k=5,
        topo_lambda_self=1.0,
        topo_lambda_succ=0.8,
        topo_lambda_pred=0.4,
        topo_lambda_total=0.5,
    ).to(device)

    num_real_steps = real_num_classes - 1
    step_prototypes = torch.randn(num_real_steps, feature_dim, device=device)
    error_prototypes = torch.randn(num_real_steps, num_types, feature_dim, device=device)
    step_node_ids = list(range(1, real_num_classes))
    predecessor_edges = [(1, 2), (2, 3), (2, 4), (4, 5)]

    model.configure_semantic_prototypes(
        step_prototypes=step_prototypes,
        error_prototypes=error_prototypes,
        step_node_ids=step_node_ids,
        predecessor_edges=predecessor_edges,
    )

    x = torch.randn(batch_size, time_steps, feature_dim, device=device)

    action_logits, frame_features, aux = model.forward_with_aux(x)

    print("action_logits:", tuple(action_logits.shape))
    print("frame_features:", tuple(frame_features.shape))

    expected_aux = [
        "base_seq",
        "short_memory_seq",
        "summary_seq",
        "long_memory_seq",
        "long_write_gate_seq",
        "visual_ctx_seq",
        "step_posteriors",
        "error_posteriors",
        "step_sem_obs",
        "error_sem_obs",
        "aux_stats_seq",
        "sem_short_seq",
        "sem_long_seq",
        "sem_long_gate_seq",
        "coverage_trace_seq",
        "uncertainty_trace_seq",
        "semantic_ctx_seq",
        "semantic_fuse_gate_seq",
        "joint_fused_seq",
        "main_logits",
        "proto_logits",
        "proto_gate",
        "final_logits",
    ]
    for k in expected_aux:
        assert k in aux, f"Missing aux key: {k}"

    labels = torch.randint(low=0, high=real_num_classes, size=(time_steps,), device=device)
    sample = {
        "action_logits": action_logits[0],
        "framewise_labels": labels,
    }

    loss_dict = model.gtg2vid_loss([sample])
    loss = loss_dict["action_ce_loss"]
    print("loss:", float(loss.detach().cpu()))

    loss.backward()

    grad_count = 0
    grad_norm_sum = 0.0
    for _, p in model.named_parameters():
        if p.grad is not None:
            grad_count += 1
            grad_norm_sum += float(p.grad.norm().detach().cpu())

    print("params_with_grad:", grad_count)
    print("sum_grad_norm:", grad_norm_sum)
    print("step_posteriors:", tuple(aux["step_posteriors"].shape))
    print("error_posteriors:", tuple(aux["error_posteriors"].shape))
    print("coverage_trace_seq:", tuple(aux["coverage_trace_seq"].shape))
    print("uncertainty_trace_seq:", tuple(aux["uncertainty_trace_seq"].shape))
    print("proto_gate:", tuple(aux["proto_gate"].shape))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
