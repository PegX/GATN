from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, Tuple
import torch

class EvalResult(Protocol):
    primary: float
    metrics: Dict[str, Any]


class AdapterAPI(Protocol):
    device: torch.device

    def setup(self) -> None: ...

    def evaluate_clean(self, steps_eval: int = -1) -> EvalResult: ...

    def list_circuits(self) -> List[Any]: ...

    def set_ablation(self, circuit: Optional[Any]) -> None: ...

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, torch.Tensor]: ...

    def forward_with_cache(self, img: torch.Tensor, node_feat: torch.Tensor, A1: Any = None, A2: Any = None) -> torch.Tensor: ...

    def get_last_cache(self) -> Dict[str, Any]: ...