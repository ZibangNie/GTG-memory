import torch
from typing import Iterable, List, Tuple


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def entropy_from_probs(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    probs = torch.clamp(probs, min=eps)
    return -(probs * probs.log()).sum(dim=dim)


def cosine_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, p=2, dim=dim, keepdim=True), min=eps)


def ema_update(prev: torch.Tensor, cur: torch.Tensor, rate: float) -> torch.Tensor:
    return (1.0 - rate) * prev + rate * cur


def build_predecessor_index_lists(
    num_real_steps: int,
    edges: Iterable[Tuple[int, int]] | None,
) -> List[List[int]]:
    preds: List[List[int]] = [[] for _ in range(num_real_steps)]
    if edges is None:
        return preds

    for src, dst in edges:
        if 1 <= dst <= num_real_steps and 1 <= src <= num_real_steps:
            preds[dst - 1].append(src - 1)
    return preds


def cumulative_mass_topk(alpha_bt_s: torch.Tensor, mass_threshold: float, max_k: int) -> torch.Tensor:
    """
    Args:
        alpha_bt_s: [B, T, S]
    Returns:
        mask: [B, T, S] bool
    """
    _, _, nsteps = alpha_bt_s.shape
    sorted_vals, sorted_idx = torch.sort(alpha_bt_s, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=-1)

    keep_sorted = cumsum <= mass_threshold
    keep_sorted[..., 0] = True

    first_cross = torch.argmax((cumsum >= mass_threshold).to(torch.int64), dim=-1, keepdim=True)
    keep_sorted.scatter_(-1, first_cross, True)

    if max_k < nsteps:
        rank = torch.arange(nsteps, device=alpha_bt_s.device).view(1, 1, nsteps)
        keep_sorted = keep_sorted & (rank < max_k)
        keep_sorted[..., 0] = True

    mask = torch.zeros_like(alpha_bt_s, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, keep_sorted)
    return mask
