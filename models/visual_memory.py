import torch
import torch.nn as nn
from typing import Dict, Tuple


class LinearNormGELU(nn.Module):
    """
    Simple Linear + LayerNorm + GELU block for sequence features [B, T, D].
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlowUpdateLongMemory(nn.Module):
    """
    Long-memory updater with capped write gate:
        g_t = cap * sigmoid(W_g u_t + U_g m_{t-1} + b_g)
        m~_t = tanh(W_u u_t + U_u m_{t-1} + b_u)
        m_t = (1 - g_t) * m_{t-1} + g_t * m~_t
    """
    def __init__(self, input_dim: int, long_dim: int, write_cap: float = 0.2):
        super().__init__()
        self.write_cap = write_cap

        self.gate_from_input = nn.Linear(input_dim, long_dim)
        self.gate_from_state = nn.Linear(long_dim, long_dim, bias=False)

        self.update_from_input = nn.Linear(input_dim, long_dim)
        self.update_from_state = nn.Linear(long_dim, long_dim, bias=False)

    def forward(
        self,
        summary_t: torch.Tensor,
        prev_long: torch.Tensor,
        gate_scale=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            summary_t: [B, input_dim]
            prev_long: [B, long_dim]
            gate_scale: [B, 1] or [B, long_dim] or None
        Returns:
            new_long: [B, long_dim]
            gate_t:   [B, long_dim]
        """
        gate_t = self.write_cap * torch.sigmoid(
            self.gate_from_input(summary_t) + self.gate_from_state(prev_long)
        )

        if gate_scale is not None:
            if gate_scale.ndim == 1:
                gate_scale = gate_scale.unsqueeze(-1)
            gate_t = gate_t * gate_scale

        proposal_t = torch.tanh(
            self.update_from_input(summary_t) + self.update_from_state(prev_long)
        )
        new_long = (1.0 - gate_t) * prev_long + gate_t * proposal_t
        return new_long, gate_t


class VisualMemoryScorer(nn.Module):
    """
    Visual-memory scorer for GTG-memory visual backbone.

    Input:
        backbone features in [B, D, T]

    Output:
        action_logits: [B, C_gtg, T]
        frame_features: [B, fusion_dim, T]

    Additional:
        forward_with_aux() returns intermediate sequences for debugging/statistics.
    """
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        short_dim: int = 256,
        long_dim: int = 384,
        fusion_dim: int = 256,
        long_write_cap: float = 0.2,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.short_dim = short_dim
        self.long_dim = long_dim
        self.fusion_dim = fusion_dim
        self.long_write_cap = long_write_cap

        # 1) base projector: features -> h_base_t
        self.base_projector = LinearNormGELU(feature_dim, fusion_dim)

        # 2) short memory: GRUCell
        self.short_cell = nn.GRUCell(fusion_dim, short_dim)

        # 3) summary MLP: [h_base_t ; m_short_t] -> u_t
        self.summary_mlp = nn.Sequential(
            nn.Linear(fusion_dim + short_dim, long_dim),
            nn.LayerNorm(long_dim),
            nn.GELU(),
            nn.Linear(long_dim, long_dim),
        )

        # 4) slow long-memory updater
        self.long_updater = SlowUpdateLongMemory(
            input_dim=long_dim,
            long_dim=long_dim,
            write_cap=long_write_cap,
        )

        # 5) project to shared fusion space
        self.base_fuse_proj = LinearNormGELU(fusion_dim, fusion_dim)
        self.short_fuse_proj = LinearNormGELU(short_dim, fusion_dim)
        self.long_fuse_proj = LinearNormGELU(long_dim, fusion_dim)

        # 6) fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(512, fusion_dim),
        )

        # 7) residual keep-safe norm
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # 8) new final head (official output head in visual-memory mode)
        self.final_head = nn.Conv1d(fusion_dim, num_classes, kernel_size=1)

    def _scan_memory(
        self,
        base_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run causal memory scan over time.

        Args:
            base_seq: [B, T, fusion_dim]

        Returns:
            short_seq: [B, T, short_dim]
            summary_seq: [B, T, long_dim]
            long_seq: [B, T, long_dim]
            gate_seq: [B, T, long_dim]
        """
        batch_size, seq_len, _ = base_seq.shape
        device = base_seq.device
        dtype = base_seq.dtype

        short_state = torch.zeros(batch_size, self.short_dim, device=device, dtype=dtype)
        long_state = torch.zeros(batch_size, self.long_dim, device=device, dtype=dtype)

        short_states = []
        summary_states = []
        long_states = []
        gate_states = []

        for t in range(seq_len):
            h_base_t = base_seq[:, t, :]                          # [B, fusion_dim]
            short_state = self.short_cell(h_base_t, short_state) # [B, short_dim]

            summary_t = self.summary_mlp(
                torch.cat([h_base_t, short_state], dim=-1)
            )                                                    # [B, long_dim]

            long_state, gate_t = self.long_updater(summary_t, long_state)

            short_states.append(short_state)
            summary_states.append(summary_t)
            long_states.append(long_state)
            gate_states.append(gate_t)

        short_seq = torch.stack(short_states, dim=1)
        summary_seq = torch.stack(summary_states, dim=1)
        long_seq = torch.stack(long_states, dim=1)
        gate_seq = torch.stack(gate_states, dim=1)

        return short_seq, summary_seq, long_seq, gate_seq

    def _forward_impl(
        self,
        features_bdt: torch.Tensor,
        return_aux: bool = False,
    ):
        """
        Args:
            features_bdt: [B, D, T]
        """
        if features_bdt.ndim != 3:
            raise ValueError(
                f"Expected features_bdt with shape [B, D, T], got {tuple(features_bdt.shape)}"
            )

        # [B, D, T] -> [B, T, D]
        features_btd = features_bdt.transpose(1, 2)

        # base sequence
        base_seq = self.base_projector(features_btd)  # [B, T, fusion_dim]

        # causal memory scan
        short_seq, summary_seq, long_seq, gate_seq = self._scan_memory(base_seq)

        # project three branches into shared fusion space
        base_fused = self.base_fuse_proj(base_seq)       # [B, T, fusion_dim]
        short_fused = self.short_fuse_proj(short_seq)    # [B, T, fusion_dim]
        long_fused = self.long_fuse_proj(long_seq)       # [B, T, fusion_dim]

        # feature-level fusion
        fusion_input = torch.cat([base_fused, short_fused, long_fused], dim=-1)
        fusion_delta = self.fusion_mlp(fusion_input)

        # residual keep-safe
        fused_seq = self.fusion_norm(base_fused + fusion_delta)  # [B, T, fusion_dim]

        # [B, T, fusion_dim] -> [B, fusion_dim, T]
        frame_features = fused_seq.transpose(1, 2)

        # official output logits in visual-memory mode
        action_logits = self.final_head(frame_features)

        if not return_aux:
            return action_logits, frame_features

        aux: Dict[str, torch.Tensor] = {
            "base_seq": base_seq,
            "short_memory_seq": short_seq,
            "summary_seq": summary_seq,
            "long_memory_seq": long_seq,
            "long_write_gate_seq": gate_seq,
            "fused_seq": fused_seq,
        }
        return action_logits, frame_features, aux

    def forward(
        self,
        features_bdt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._forward_impl(features_bdt, return_aux=False)

    def forward_with_aux(
        self,
        features_bdt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return self._forward_impl(features_bdt, return_aux=True)
