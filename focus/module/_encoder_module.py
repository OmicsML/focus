from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import Dropout, LayerNorm, Linear, Module, BatchNorm1d, ModuleList
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv

from .nets._denseginconv import DenseGINConv
from .nets._mlp import MLP


class Encoder(nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        
        self.eps = Parameter(torch.zeros(self.num_gc_layers))
        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = DenseGINConv(dim, dim)
            else:
                conv = DenseGINConv(num_features, dim)
            bn = BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)
            
    def forward(self, data, device):
        x, batch = data.x, data.batch
        if 'adj' not in data:
            adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)).to(device), (data.num_nodes, data.num_nodes), requires_grad=True)#.to_dense()
        else:
            adj = data.adj
        
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, adj))
            x = self.bns[i](x)
            # x = self.propagate(x, i, adj)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)
    
    def get_embeddings(self, data_loader, device):
        ret = []
        y = []
        for data in data_loader:
            with torch.no_grad():
                if isinstance(data, list):
                    data = data[0].to(device)
                else:
                    data = data.to(device)

                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(data)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


    def get_embeddings_v(self, loader, device):
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y
    
class Encoder2(nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        # super(Encoder, self).__init__()
        super(Encoder2, self).__init__() # changed on 1/6/2024
        

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers

        self.eps = Parameter(torch.zeros(self.num_gc_layers))
        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = MLP(2, dim, dim, dim)
            else:
                conv = MLP(2, num_features, dim, dim) # i = 0
            # conv = GINConv(nn)
            bn = BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)
            
    
    def propagate(self, x, layer, adj):
        # pooled = adj @ x
        pooled = torch.sparse.mm(adj, x)
        pooled = pooled + (1 + self.eps[layer]) * x
        # print('propagate', layer, self.convs[layer])
        x = F.relu(self.convs[layer](pooled))
        h = self.bns[layer](x)

        return h
        

    def forward(self, data, device):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if 'adj' not in data:
            adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(device), (data.num_nodes, data.num_nodes), requires_grad=True)#.to_dense()
            # print('-=-=-=adj not in data')
        else:
            adj = data.adj
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):
            x = self.propagate(x, i, adj)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, data_loader, device):
        ret = []
        y = []
        for data in data_loader:
            with torch.no_grad():
                if isinstance(data, list):
                    data = data[0].to(device)
                else:
                    data = data.to(device)
                x, _ = self.forward(data)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def get_embeddings_v(self, loader, device):
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y