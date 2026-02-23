from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict

import torch

from avic_gatn.algorithms.alg1_circuit_discovery import discover_vulnerability_circuits
from avic_gatn.algorithms.alg2_circuit_aware_attack import pgd_attack_node_features
from avic_gatn.tasks.toy_task import ToyGATNTaskAdapter


def run_end2end(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device(cfg.get("device", "cpu"))

    adapter = ToyGATNTaskAdapter(cfg=cfg, device=device)
    adapter.setup()

    steps_eval = int(cfg["data"]["steps_eval"])

    # === Baseline eval ===
    base = adapter.evaluate(steps_eval)

    # === Alg1: circuit discovery (head ablation) ===
    topk = int(cfg["alg1"]["topk_circuits"])
    circuit_scores = discover_vulnerability_circuits(adapter, topk=topk, steps_eval=steps_eval)

    # === Alg2: circuit-aware PGD attack on node features ===
    a2 = cfg["alg2"]
    attack = pgd_attack_node_features(
        adapter,
        circuits=[cs.circuit for cs in circuit_scores],
        eps=float(a2["eps"]),
        step_size=float(a2["step_size"]),
        steps=int(a2["steps"]),
        lam=float(a2["circuit_aware_lambda"]),
        steps_eval=steps_eval,
        mode=str(a2.get("circuit_aware_mode", "attn_kl")),
    )

    # === Minimal trace export ===
    # Run one clean+adv step to export attention snapshots
    img_feat, node_feat, A1, A2, y = adapter.sample_batch()
    _ = adapter.forward_with_cache(img_feat, node_feat, A1, A2)
    clean_cache = adapter.get_last_cache()

    # create a single-step adversarial sample using the same settings
    # (reuse the attack helper but with steps_eval=1)
    attack_one = pgd_attack_node_features(
        adapter,
        circuits=[cs.circuit for cs in circuit_scores],
        eps=float(a2["eps"]),
        step_size=float(a2["step_size"]),
        steps=int(a2["steps"]),
        lam=float(a2["circuit_aware_lambda"]),
        steps_eval=1,
        mode=str(a2.get("circuit_aware_mode", "attn_kl")),
    )
    # The above call runs its own sampled batches; to keep this minimal and deterministic,
    # we just store the last cache currently in adapter as the "adv" snapshot.
    adv_cache = adapter.get_last_cache()

    # serialize only small tensors
    def _tensor_stats(t: torch.Tensor) -> Dict[str, float]:
        t = t.detach().float().cpu()
        return {
            "shape": list(t.shape),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    trace = {
        "clean": {k: _tensor_stats(v) for k, v in clean_cache.items() if isinstance(v, torch.Tensor)},
        "adv": {k: _tensor_stats(v) for k, v in adv_cache.items() if isinstance(v, torch.Tensor)},
    }

    report = {
        "seed": seed,
        "baseline": {"primary": base.primary, **base.metrics},
        "alg1_top_circuits": [
            {
                "attn_id": cs.circuit.attn_id,
                "head_id": cs.circuit.head_id,
                "primary_drop": cs.primary_drop,
            }
            for cs in circuit_scores
        ],
        "alg2_attack": asdict(attack),
        "trace": trace,
    }

    out_dir = cfg["report"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    tag = cfg["report"].get("tag", "run")
    out_path = os.path.join(out_dir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {"report_path": out_path, **report}
