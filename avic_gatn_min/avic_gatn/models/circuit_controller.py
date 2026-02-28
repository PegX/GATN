# avic_gatn/models/circuit_controller.py
from dataclasses import dataclass
from typing import Dict, Optional
import torch

@dataclass(frozen=True)
class CircuitID:
    layer: int
    attn: str   # e.g., "attn" / "attn1" / "attn2"
    head: int

@dataclass
class CircuitPatch:
    mode: str = "scale_head"     # scale_head | mix_uniform | temp_attn_logits
    scale: float = 0.9
    mix_beta: float = 0.03
    temperature: float = 1.2

class CircuitController:
    def __init__(self):
        self.ablate: Dict[CircuitID, bool] = {}
        self.patches: Dict[CircuitID, CircuitPatch] = {}
        self.cache: Dict[str, torch.Tensor] = {}

    def reset_cache(self):
        self.cache = {}

    def set_ablation(self, cid: Optional[CircuitID]):
        self.ablate.clear()
        if cid is not None:
            self.ablate[cid] = True

    def clear_ablation(self):
        self.ablate.clear()

    def set_patch(self, cid: CircuitID, patch: CircuitPatch):
        self.patches[cid] = patch

    def clear_patches(self):
        self.patches.clear()

    # --- hook helpers ---
    def cache_tensor(self, key: str, t: torch.Tensor):
        # 只缓存轻量信息或 cpu copy（避免爆显存）
        self.cache[key] = t.detach().float().cpu()

    def apply_on_attn(self, attn: torch.Tensor, layer: int, attn_name: str) -> torch.Tensor:
        """
        attn: [B,H,N,N] (softmax后) 或者你自己决定缓存 logits
        """
        self.cache_tensor(f"L{layer}.{attn_name}.attn", attn)

        B, H, N, _ = attn.shape
        out = attn

        for h in range(H):
            cid = CircuitID(layer, attn_name, h)
            if cid in self.ablate:
                out = out.clone()
                out[:, h] = 0.0
                continue
            if cid in self.patches:
                p = self.patches[cid]
                if p.mode == "mix_uniform":
                    out = out.clone()
                    w = out[:, h]
                    uni = torch.full_like(w, 1.0 / w.size(-1))
                    out[:, h] = (1 - p.mix_beta) * w + p.mix_beta * uni
        return out

    def apply_on_head_out(self, head_out: torch.Tensor, layer: int, attn_name: str) -> torch.Tensor:
        """
        head_out: [B,H,N,D] attention聚合后合并前
        """
        # 缓存 head norm 作为 trace
        hn = torch.linalg.vector_norm(head_out, dim=-1).mean(dim=(0, 2))  # [H]
        self.cache_tensor(f"L{layer}.{attn_name}.head_norm", hn)

        B, H, N, D = head_out.shape
        out = head_out

        for h in range(H):
            cid = CircuitID(layer, attn_name, h)
            if cid in self.ablate:
                out = out.clone()
                out[:, h] = 0.0
                continue
            if cid in self.patches:
                p = self.patches[cid]
                if p.mode == "scale_head":
                    out = out.clone()
                    out[:, h] = out[:, h] * p.scale
        return out
    
    