from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import Dropout, LayerNorm, Linear, Module, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv


class GAT(nn.Module):
    def __init__(self, 
                 dataset: List[Any],
                 hidden: int,
                 num_feat_layers: int = 1,
                 num_conv_layers: int = 3,
                 num_fc_layers: int = 2,
                 collapse: bool = False,
                 residual: bool = False,
                 res_branch: str = "BNConvReLU",
                 global_pool: str = "sum",
                 dropout: int = 0,
                 ):
        super().__init__()
        
        assert num_feat_layers == 1
        self.conv_residual = residual
        self.fc_residual = False
        self.res_branch = res_branch
        self.collapse = collapse
        
        # TODO: add 
    
    # TODO: add more on GAT class
        
    def reset_parameters(self):
        pass
    
    def forward(self, data):
        pass
    
    
        
        
        