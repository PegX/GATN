# -*- coding: utf-8 -*-
"""Minimal GATN-style model for AViC.

This file is **inspired by** the upstream GATN implementation:
- Attention / AttentionBlock structure follows `transformer.py` in a791702141/GATN.
- Adjacency generation + 2-layer GCN follows `models.py` in a791702141/GATN.

We intentionally keep only what AViC needs:
- A topology transformer block producing an adjacency matrix.
- Two-layer graph convolution over class nodes.
- A final bilinear scoring between an image feature vector and class node representations.

This minimal model runs on a toy task (no COCO/VOC files required).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Circuit:
    """Circuit identifier for AViC@GATN.

    attn_id: which attention module inside AttentionBlock (0 or 1)
    head_id: which head inside that attention module
    """

    attn_id: int
    head_id: int


class MultiHeadSelfAttention(nn.Module):
    """A small multi-head self-attention layer with instrumentation.

    Differences vs upstream:
    - supports `set_ablation(attn_head=...)`
    - caches `last_attn` (softmax probabilities) for tracing / circuit-aware loss
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self._ablate_head: Optional[int] = None
        self.last_attn: Optional[torch.Tensor] = None  # [B, H, N, N]

    def set_ablation(self, head_id: Optional[int]):
        self._ablate_head = head_id

    def _transpose(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, C] -> [B, H, N, D]
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        q = self._transpose(self.query(x))
        k = self._transpose(self.key(x))
        v = self._transpose(self.value(x))

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B,H,N,N]
        attn = F.softmax(scores, dim=-1)

        # head ablation: zero-out one head's attention distribution
        if self._ablate_head is not None:
            h = int(self._ablate_head)
            if 0 <= h < self.num_heads:
                attn = attn.clone()
                attn[:, h, :, :] = 0.0

        self.last_attn = attn

        ctx = attn @ v  # [B,H,N,D]
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), self.hidden_size)
        out = self.out(ctx)
        return out


class AttentionBlock(nn.Module):
    """A minimal AttentionBlock with two attention modules (attn1, attn2).

    Upstream `transformer.py` does:
        x1 = attn1(norm(x))
        x2 = attn2(norm(x_save))
        x = bmm(x1, x2)

    We keep this structure because it produces an adjacency-like matrix.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn1 = MultiHeadSelfAttention(hidden_size, num_heads)
        self.attn2 = MultiHeadSelfAttention(hidden_size, num_heads)

    def set_ablation(self, circuit: Optional[Circuit]):
        # circuit.attn_id: 0->attn1, 1->attn2
        self.attn1.set_ablation(None)
        self.attn2.set_ablation(None)
        if circuit is None:
            return
        if circuit.attn_id == 0:
            self.attn1.set_ablation(circuit.head_id)
        elif circuit.attn_id == 1:
            self.attn2.set_ablation(circuit.head_id)

    def forward(self, A1: torch.Tensor, A2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute an adjacency-like matrix.

        Args:
            A1, A2: [B, N, N] float tensors (two topology priors)

        Returns:
            adj: [B, N, N]
            cache: attention matrices for tracing
        """
        x = A1
        x_save = x
        x1 = self.attn1(self.norm(x))  # [B,N,N]
        x2 = self.attn2(self.norm(x_save))
        adj = torch.bmm(x1, x2)  # [B,N,N]

        cache = {
            "attn1": self.attn1.last_attn if self.attn1.last_attn is not None else torch.empty(0),
            "attn2": self.attn2.last_attn if self.attn2.last_attn is not None else torch.empty(0),
        }
        return adj, cache


class GraphConvolution(nn.Module):
    """Simple GCN layer: X' = A X W"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, Fin], adj: [B, N, N]
        support = x @ self.weight  # [B,N,Fout]
        out = adj @ support
        return out


def normalize_adj(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """GCN normalization: D^{-1/2} A D^{-1/2}."""
    deg = adj.sum(dim=-1)  # [B,N]
    deg_inv_sqrt = torch.rsqrt(deg + eps)
    D = torch.diag_embed(deg_inv_sqrt)
    return D @ adj @ D


class GATNMinimal(nn.Module):
    """A minimal end-to-end model.

    Inputs:
      - img_feat: [B, F] image features (toy)
      - node_feat: [B, N, Cin] class-node embeddings (attack surface)
      - A1, A2: [B, N, N] topology priors

    Output:
      - logits: [B, N]
      - cache: intermediate attention tensors
    """

    def __init__(
        self,
        num_classes: int,
        in_channel: int,
        hidden_gcn: int,
        feat_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.topo_block = AttentionBlock(hidden_size=num_classes, num_heads=num_heads)
        self.gc1 = GraphConvolution(in_channel, hidden_gcn)
        self.gc2 = GraphConvolution(hidden_gcn, feat_dim)
        self.act = nn.LeakyReLU(0.2)

    def set_ablation(self, circuit: Optional[Circuit]):
        self.topo_block.set_ablation(circuit)

    def forward(
        self,
        img_feat: torch.Tensor,
        node_feat: torch.Tensor,
        A1: torch.Tensor,
        A2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, _ = node_feat.shape
        assert N == self.num_classes

        adj_raw, cache = self.topo_block(A1, A2)  # [B,N,N]
        # add self loops
        adj = adj_raw + torch.eye(N, device=adj_raw.device).unsqueeze(0)
        adj = normalize_adj(adj)

        x = self.act(self.gc1(node_feat, adj))
        x = self.gc2(x, adj)  # [B,N,F]
        # bilinear scoring: feature dot class vectors
        logits = torch.einsum("bf,bnf->bn", img_feat, x)
        return logits, {**cache, "adj": adj_raw}
