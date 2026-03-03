from __future__ import annotations

from typing import Any, Dict, Optional

from avic_gatn.core.circuits import CircuitID


class CircuitController:
    """
    - ablation: optional CircuitID to ablate (single-circuit ablation for Alg1 baseline)
    - focus_set: optional set of circuit keys for "circuit-aware" (Alg2 regularizer)
    - cache: stores last forward's attention weights for trace/export
    """
    def __init__(self):
        self.ablation: Optional[CircuitID] = None
        self.focus_set = None  # Optional[set[str]]
        self._cache: Dict[str, Any] = {}

    def clear(self):
        self.ablation = None
        self.focus_set = None
        self._cache = {}

    def clear_ablation(self):
        self.ablation = None

    def set_ablation(self, circuit: CircuitID):
        self.ablation = circuit

    def set_focus(self, circuits):
        if circuits is None:
            self.focus_set = None
        else:
            self.focus_set = set([c.key() if hasattr(c, "key") else str(c) for c in circuits])

    def should_ablate(self, layer_idx: int, attn_name: str, head_idx: int) -> bool:
        c = self.ablation
        return (c is not None) and (c.layer == layer_idx) and (c.attn == attn_name) and (c.head == head_idx)

    def should_focus(self, layer_idx: int, attn_name: str, head_idx: int) -> bool:
        if self.focus_set is None:
            return True
        key = f"layer{layer_idx}.{attn_name}.head{head_idx}"
        return key in self.focus_set

    def put_cache(self, layer_idx: int, attn_name: str, attn_weights):
        # store last weights
        self._cache[f"layer{layer_idx}.{attn_name}.attn"] = attn_weights.detach().cpu()

    def get_cache(self) -> Dict[str, Any]:
        return self._cache
    


    # ----------------------------
    # Methods expected by transformer.py hooks
    # ----------------------------
    def apply_on_attn(self, attn_weights, layer_idx: int, attn_name: str):
        """
        Called right after softmax attention weights are computed.
        Stores cache and optionally masks heads (ablation) in the weights domain.
        Expected attn_weights shapes:
          - [B, H, N, N]  (common)
          - [H, N, N]
        """
        if attn_weights is None:
            return attn_weights

        try:
            self.put_cache(layer_idx, attn_name, attn_weights)
        except Exception:
            pass

        c = self.ablation
        if c is None:
            return attn_weights
        if not (c.layer == layer_idx and c.attn == attn_name):
            return attn_weights

        h = int(c.head)
        # mask selected head
        if hasattr(attn_weights, "dim"):
            if attn_weights.dim() == 4 and attn_weights.shape[1] > h:
                attn_weights[:, h, :, :] = 0
            elif attn_weights.dim() == 3 and attn_weights.shape[0] > h:
                attn_weights[h, :, :] = 0
        return attn_weights

    def apply_on_head_out(self, context_layer, layer_idx: int, attn_name: str):
        """
        Called before heads are merged (head output tensor).
        We implement single-head ablation by zeroing that head.
        Expected context_layer shapes:
          - [B, H, N, D]  OR  [B, N, H, D]
          - (rare) [H, N, D] / [N, H, D]
        """
        c = self.ablation
        if c is None:
            return context_layer
        if not (c.layer == layer_idx and c.attn == attn_name):
            return context_layer

        h = int(c.head)
        if not hasattr(context_layer, "dim"):
            return context_layer

        if context_layer.dim() == 4:
            # [B,H,N,D]
            if context_layer.shape[1] > h:
                context_layer[:, h, :, :] = 0
            # [B,N,H,D]
            elif context_layer.shape[2] > h:
                context_layer[:, :, h, :] = 0
        elif context_layer.dim() == 3:
            # [H,N,D]
            if context_layer.shape[0] > h:
                context_layer[h, :, :] = 0
            # [N,H,D]
            elif context_layer.shape[1] > h:
                context_layer[:, h, :] = 0
        return context_layer