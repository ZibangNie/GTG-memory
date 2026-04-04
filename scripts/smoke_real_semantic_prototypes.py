import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from models.models import ASDiffusionBackbone
from utils.semantic_prototype_loader import load_task_semantic_prototypes


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_data_dir = "/root/autodl-tmp/data/EgoPER"
    dataset_name = "tea"
    feature_dim = 256

    payload = load_task_semantic_prototypes(
        root_data_dir=root_data_dir,
        dataset_name=dataset_name,
        feature_dim=feature_dim,
        normal_dir_name="vc_normal_action_features",
        error_dir_name="vc_chatgpt4omini_error_features",
    )

    step_prototypes = payload["step_prototypes"].to(device)
    error_prototypes = payload["error_prototypes"].to(device)
    step_node_ids = payload["step_node_ids"]
    predecessor_edges = payload["predecessor_edges"]

    real_num_classes = len(step_node_ids) + 1
    num_classes = int(payload["num_classes"])
    num_types = int(payload["num_error_types"])

    print("task_dir:", payload["task_dir"])
    print("normal_dir:", payload["normal_dir"])
    print("error_dir:", payload["error_dir"])
    print("step_prototypes:", tuple(step_prototypes.shape))
    print("error_prototypes:", tuple(error_prototypes.shape))
    print("step_node_ids:", step_node_ids[:10], "..." if len(step_node_ids) > 10 else "")
    print("num_error_types:", num_types)
    print("missing_error_pairs:", len(payload["missing_error_pairs"]))

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

    model.configure_semantic_prototypes(
        step_prototypes=step_prototypes,
        error_prototypes=error_prototypes,
        step_node_ids=step_node_ids,
        predecessor_edges=predecessor_edges,
    )

    x = torch.randn(2, 24, feature_dim, device=device)

    action_logits, frame_features, aux = model.forward_with_aux(x)

    print("action_logits:", tuple(action_logits.shape))
    print("frame_features:", tuple(frame_features.shape))
    print("step_posteriors:", tuple(aux["step_posteriors"].shape))
    print("error_posteriors:", tuple(aux["error_posteriors"].shape))
    print("coverage_trace_seq:", tuple(aux["coverage_trace_seq"].shape))
    print("uncertainty_trace_seq:", tuple(aux["uncertainty_trace_seq"].shape))
    print("proto_gate:", tuple(aux["proto_gate"].shape))

    labels = torch.randint(low=0, high=real_num_classes, size=(24,), device=device)
    sample = {
        "action_logits": action_logits[0],
        "framewise_labels": labels,
    }
    loss = model.gtg2vid_loss([sample])["action_ce_loss"]
    print("loss:", float(loss.detach().cpu()))
    loss.backward()

    grad_count = 0
    for _, p in model.named_parameters():
        if p.grad is not None:
            grad_count += 1
    print("params_with_grad:", grad_count)
    print("REAL PROTOTYPE SMOKE PASSED")


if __name__ == "__main__":
    main()
