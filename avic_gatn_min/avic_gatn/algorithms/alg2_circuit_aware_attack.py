from __future__ import annotations
from avic_gatn.models.gatn_min import Circuit
#from avic_gatn.tasks.toy_task import ToyGATNTaskAdapter
from avic_gatn.tasks.adapter_api import AdapterAPI


from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

# circuits are CircuitID (or similar) with fields: layer, attn, head, and .key()
 
# existing imports...
@dataclass
class AttackResult:
    clean_primary: float
    adv_primary: float
    success_delta: float


def _get_primary(x):
    # supports: float, dict {"primary": ...}, or EvalResult(primary=...)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict) and "primary" in x:
        return float(x["primary"])
    if hasattr(x, "primary"):
        return float(x.primary)
    raise TypeError(f"Unsupported evaluate_clean() return type: {type(x)}")


def _extract_attn(cache: Dict[str, Any], circuit) -> torch.Tensor:
    """
    Expect cache key like: 'layer0.attn1.attn' -> Tensor [B,H,N,N] or [H,N,N] or [B,N,N]
    We try to normalize to [B,N,N] for a given head.
    """
    layer = getattr(circuit, "layer", 0)
    attn_name = getattr(circuit, "attn", "attn1")
    head = int(getattr(circuit, "head", 0))
    k = f"layer{layer}.{attn_name}.attn"
    if k not in cache:
        raise KeyError(f"Missing attention cache key: {k}. Available: {list(cache.keys())[:10]}")
    w = cache[k]
    if not torch.is_tensor(w):
        w = torch.as_tensor(w)

    # common shapes:
    # [B,H,N,N] -> select head => [B,N,N]
    # [H,N,N]   -> select head => [N,N] -> add batch => [1,N,N]
    # [B,N,N]   -> already head-collapsed (no head dimension)
    if w.dim() == 4:
        w = w[:, head, :, :]
    elif w.dim() == 3:
        if w.shape[0] == head or w.shape[0] > head:
            # assume [H,N,N]
            if w.shape[0] != 1:
                w = w[head, :, :].unsqueeze(0)
    elif w.dim() == 2:
        w = w.unsqueeze(0)
    return w


def _circuit_attn_kl(clean_cache: Dict[str, Any], adv_cache: Dict[str, Any], circuits: List[Any]) -> torch.Tensor:
    """
    KL(clean || adv) on selected circuits, averaged.
    Both caches may be CPU tensors; caller moves result to device if needed.
    """
    regs = []
    for c in circuits:
        wc = _extract_attn(clean_cache, c).float()
        wa = _extract_attn(adv_cache, c).float()
        # normalize to probability along last dim
        wc = wc / (wc.sum(dim=-1, keepdim=True) + 1e-8)
        wa = wa / (wa.sum(dim=-1, keepdim=True) + 1e-8)
        # KL(clean || adv) = sum clean * (log clean - log adv)
        regs.append((wc * ((wc + 1e-8).log() - (wa + 1e-8).log())).mean())
    if len(regs) == 0:
        return torch.tensor(0.0)
    return torch.stack(regs).mean()

 
def pgd_attack_node_features(
    adapter,
    circuits: List[Any],
    eps: float,
    step_size: float,
    steps: int,
    lam: float,
    steps_eval: int,
    mode: str = "attn_kl",
) -> AttackResult:

    device = adapter.device
    assert hasattr(adapter, "model") and adapter.model is not None

    # clean baseline
    adapter.set_ablation(None)
    clean_primary = adapter.evaluate_clean(steps_eval)

    bsz = int(adapter.cfg["gatn"]["batch_size"])
    clean_primary = _get_primary(adapter.evaluate_clean(steps_eval))

    adv_losses = []

    for _ in range(steps_eval):
        img_feat, node_feat, A1, A2, y = adapter.sample_batch()
 
        node_feat = node_feat.detach()
        delta = torch.empty_like(node_feat).uniform_(-eps, eps)
        delta = delta.clamp(-eps, eps)
        delta.requires_grad_(True)
 
        with torch.no_grad():
            _ = adapter.forward_with_cache(img_feat, node_feat, A1, A2)
            clean_cache = adapter.get_last_cache()
 
        for _k in range(steps):
            adv_node = node_feat + delta
            logits = adapter.forward_with_cache(img_feat, adv_node, A1, A2)
            loss_main = F.binary_cross_entropy_with_logits(logits, y)
            loss = loss_main
 
            if lam > 0 and len(circuits) > 0 and mode == "attn_kl":
               adv_cache = adapter.get_last_cache()
               reg = _circuit_attn_kl(clean_cache, adv_cache, circuits).to(device)
               loss = loss_main - lam * reg
 
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta.data = (delta.data + step_size * torch.sign(grad)).clamp(-eps, eps)
            delta.grad = None
 
        with torch.no_grad():
            adv_node = node_feat + delta.detach()
            logits_adv = adapter.forward_with_cache(img_feat, adv_node, A1, A2)
            loss_adv = F.binary_cross_entropy_with_logits(logits_adv, y)
            adv_losses.append(float(loss_adv.item()))
 
     # primary = -bce
    adv_primary = - (sum(adv_losses) / max(1, len(adv_losses)))
    return AttackResult(
        clean_primary=float(clean_primary),
        adv_primary=float(adv_primary),
        success_delta=float(clean_primary - adv_primary),
    )