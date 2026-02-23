from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from avic_gatn.models.gatn_min import Circuit
from avic_gatn.tasks.toy_task import ToyGATNTaskAdapter


@dataclass
class AttackResult:
    clean_primary: float
    adv_primary: float
    success_delta: float


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p,q: [..., N]
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def _circuit_attn_kl(clean_cache: dict, adv_cache: dict, circuits: List[Circuit]) -> torch.Tensor:
    reg = 0.0
    cnt = 0
    for c in circuits:
        key = "attn1" if c.attn_id == 0 else "attn2"
        if key not in clean_cache or key not in adv_cache:
            continue
        A0 = clean_cache[key]  # [B,H,N,N]
        A1 = adv_cache[key]
        if A0.numel() == 0 or A1.numel() == 0:
            continue
        h = c.head_id
        # KL over last dim (dest nodes), averaged over src nodes and batch
        kl = _kl(A0[:, h, :, :], A1[:, h, :, :]).mean()
        reg = reg + kl
        cnt += 1
    if cnt == 0:
        return torch.tensor(0.0)
    return reg / cnt


def pgd_attack_node_features(
    adapter: ToyGATNTaskAdapter,
    circuits: List[Circuit],
    eps: float,
    step_size: float,
    steps: int,
    lam: float,
    steps_eval: int,
    mode: str = "attn_kl",
) -> AttackResult:
    """PGD on node features (class-node embeddings).

    Attack surface: `node_feat` in adapter.sample_batch().

    Loss:
      L = BCE(logits_adv, y) - lam * KL(attn_clean || attn_adv) on selected circuits

    We maximize loss (untargeted), i.e. gradient ascent on node_feat perturbation.
    """

    device = adapter.device
    model = adapter.model
    assert model is not None

    # clean baseline
    adapter.set_ablation(None)
    clean_primary = adapter.evaluate(steps_eval).primary

    bsz = int(adapter.cfg["data"]["batch_size"])

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
