from __future__ import annotations

from dataclasses import dataclass
from typing import List

from avic_gatn.models.gatn_min import Circuit
from avic_gatn.tasks.toy_task import ToyGATNTaskAdapter


@dataclass
class CircuitScore:
    circuit: Circuit
    primary_drop: float


def discover_vulnerability_circuits(adapter: ToyGATNTaskAdapter, topk: int, steps_eval: int) -> List[CircuitScore]:
    # baseline
    adapter.set_ablation(None)
    base = adapter.evaluate(steps_eval).primary

    scores: List[CircuitScore] = []
    circuit = adapter.list_circuits()
    for c in circuit:
        adapter.set_ablation(c)
        val = adapter.evaluate(steps_eval).primary
        drop = base - val
        scores.append(CircuitScore(circuit=c, primary_drop=float(drop)))
    scores.sort(key=lambda x: x.primary_drop, reverse=True)
    adapter.set_ablation(None)
    return scores[:topk]
