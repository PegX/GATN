from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class CircuitID:
    layer: int
    attn: str
    head: int

    def key(self) -> str:
        return f"layer{self.layer}.{self.attn}.head{self.head}"