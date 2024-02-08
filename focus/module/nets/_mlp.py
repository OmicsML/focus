from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import Dropout, LayerNorm, Linear, Module, BatchNorm1d, ModuleList
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv

class MLP(nn.Module):
    """
    MLP
    """
    def __init__(self, 
                 num_layers: int, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        # define model layers
        self.linears = ModuleList([Linear(input_dim, hidden_dim)])
        
        # handle single layer perceptron case
        if num_layers > 1:
            for _ in range(num_layers - 2):
                self.linears.append(Linear(hidden_dim, hidden_dim))
            self.linears.append(Linear(hidden_dim, output_dim))

        # define activation function
        self.activation = F.relu

        # define batch norms
        self.batch_norms = ModuleList([BatchNorm1d((hidden_dim)) for _ in range(num_layers - 1)])

    def forward(self, x):
        for l in range(self.num_layers - 1):
            # print(l, self.linears[l])
            x = self.linears[l](x)
            x = self.batch_norms[l](x)
            x = self.activation(x)

        return self.linears[-1](x)