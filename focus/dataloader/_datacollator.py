"""
    data collator for focus, 
    will be used in focus/dataloader/_dataloader.py
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import torch 
from torch_geometric import data
from torch_geometric.transforms import Constant
from torch_geometric.loader import DataLoader


@dataclass
class DataCollatorForNodeClassification:
    """Data collator used for node classification."""

    def __call__(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""Merge a list of samples to form a mini-batch."""
        batch = torch.stack(batch, dim=0)
        return {'batch': batch}
    
@dataclass
class DataCollatorForGraphClassification:
    """Data collator used for graph classification."""

    def __call__(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""Merge a list of samples to form a mini-batch."""
        batch = torch.stack(batch, dim=0)
        return {'batch': batch}
    
def collator_for_graph_classification(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    r"""Merge a list of samples to form a mini-batch."""
    pass
    
