from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import Dropout, LayerNorm, Linear, Module, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv

from .._encoder_module import Encoder 

class simclr(nn.Module):
    def __init__(self, 
                 num_features, 
                 hidden_dim, 
                 num_gc_layers, 
                 prior, 
                 alpha=0.5, 
                 beta=1., 
                 gamma=.1):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

        self.embedding_dim = hidden_dim * num_gc_layers

        self.encoder = Encoder(num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, device):
        # batch_size = data.num_graphs
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # if x is None:
        #     x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(data, device)
        y = self.proj_head(y)
        return y