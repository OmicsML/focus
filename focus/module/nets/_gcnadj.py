from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .layers import GraphConvolution
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_scipy_sparse_matrix

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv

from ._graphconvolution import GraphConvolution

class GCNadj(nn.Module):
    def __init__(self, 
                 nfeat, 
                 hidden, 
                 dropout,
                 label_num: List[int] = None,
                 ):
        super(GCNadj, self).__init__()
        
        self.gc1 = GraphConvolution(nfeat, hidden)
        self.gc2 = GraphConvolution(hidden, hidden)
        self.dropout = dropout
        self.lin_class = Linear(hidden, dataset.num_classes)
        self.global_pool = global_mean_pool

    def forward(self, data):
        x, batch = data.x, data.batch
        # print('-'*20, 'data.adj', data.adj, type(data.adj))
        # adj = to_scipy_sparse_matrix(data.adj, num_nodes=data.num_nodes).to_dense()
        # adj = data.adj.to_dense()
        if 'adj' not in data:
            adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)).to(data.x.device), (data.num_nodes, data.num_nodes), requires_grad=True).to_dense()
            print('-=-=-=adj not in data')
        else:
            adj = data.adj
            print('*******adj in data')
        # print('-'*20, 'data.adj', data.adj, type(data.adj))
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.global_pool(x, batch)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=1)