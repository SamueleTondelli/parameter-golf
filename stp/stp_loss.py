"""
Semantic Tube Prediction (STP) loss.

A lightweight JEPA-style regularizer that enforces local linearity of
hidden-state trajectories.  See stp/STP.md for theory and references.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_stp_loss(hidden_states: Tensor) -> Tensor:
    """
    Compute the Semantic Tube Prediction loss for a batch of last-layer
    hidden states.

    Args:
        hidden_states: shape [batch_size, seq_len, hidden_dim]

    Returns:
        Scalar STP loss (0 when seq_len < 3).
    """
    seq_len = hidden_states.size(1)
    if seq_len < 3:
        return torch.tensor(0.0, device=hidden_states.device)

    # Pick 3 random sorted positions (shared across the batch for speed).
    indices = torch.sort(torch.randperm(seq_len, device=hidden_states.device)[:3])[0]
    s, r, t = indices[0], indices[1], indices[2]

    h_s = hidden_states[:, s, :]
    h_r = hidden_states[:, r, :]
    h_t = hidden_states[:, t, :]

    vec_sr = h_r - h_s  # s -> r
    vec_rt = h_t - h_r  # r -> t

    cos_sim = F.cosine_similarity(vec_rt, vec_sr, dim=-1)
    return (1.0 - cos_sim).mean()
