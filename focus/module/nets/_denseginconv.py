from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import Dropout, LayerNorm, Linear, Module, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv

class DenseGINConv(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 bias: bool=True):
        super(DenseGINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            Linear(out_channels, out_channels),
            torch.nn.ReLU(),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.eps)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = self.mlp((1+self.eps)*x + torch.sparse.mm(adj, x))
        if self.bias is not None:
            out = out + self.bias
        out = F.relu(out)
        return out