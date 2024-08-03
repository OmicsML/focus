import os 
import numpy as np 
import networkx as nx
import sys 
import time
import re
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch import optim
from torch_geometric import data
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from .ResGCN import ResGCN
from .GCN_adj import ResGCNadj, GCNadj
from .losses import local_global_loss_

class simclr(nn.Module):
    def __init__(self, num_features, hidden_dim, num_gc_layers, prior, alpha=0.5, beta=1., gamma=.1):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

        self.embedding_dim = hidden_dim * num_gc_layers

        self.encoder = Encoder(num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, device):
        y, M = self.encoder(data, device)
        y = self.proj_head(y)
        return y


class DenseGINConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DenseGINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
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

class Encoder(nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        
        self.eps = torch.nn.Parameter(torch.zeros(self.num_gc_layers))
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = DenseGINConv(dim, dim)
            else:
                conv = DenseGINConv(num_features, dim)
            # conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

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
    
class Encoder2(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.eps = torch.nn.Parameter(torch.zeros(self.num_gc_layers))
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = MLP(2, dim, dim, dim)
            else:
                conv = MLP(2, num_features, dim, dim) # i = 0
            # conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

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

class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        # define model layers
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_dim, hidden_dim)])
        
        # handle single layer perceptron case
        if num_layers > 1:
            for _ in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

        # define activation function
        self.activation = F.relu

        # define batch norms
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d((hidden_dim)) for _ in range(num_layers - 1)])

    def forward(self, x):
        for l in range(self.num_layers - 1):
            # print(l, self.linears[l])
            x = self.linears[l](x)
            x = self.batch_norms[l](x)
            x = self.activation(x)

        return self.linears[-1](x)


class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    """
    Residual block MLP
    """
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class GCNInfomax(nn.Module):
    def __init__(self, args, hidden_dim, num_gc_layers,num_features, alpha=0.5, beta=1., gamma=.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior
        
        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(num_features, self.embedding_dim, num_gc_layers)
        
        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        
        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)
        
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x, edge_index, batch, device):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
            
        y, M = self.encoder(x, edge_index, batch)
        
        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        measure='JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
    
        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0
        
        return local_global_loss + PRIOR
    
    
def get_model_with_default_configs(model_name, num_features,  num_feat_layers,
                                   num_conv_layers, num_fc_layers,
                                   residual,
                                   res_branch,
                                   global_pool,
                                   dropout,
                                   edge_norm,
                                   hidden):
    
    num_feat_layers = 1
    if model_name.startswith('ResGFN'):
        collapse = True if 'flat' in model_name else False
        def foo(dataset):
            return ResGCN(dataset, num_features, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=True, collapse=collapse,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    elif model_name.startswith('ResGCN'):
        def foo(dataset):
            return ResGCN(dataset,num_features, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=False, collapse=False,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    elif model_name.startswith('adjResGCN'):
        def foo(dataset):
            return ResGCNadj(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=False, collapse=False,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    elif model_name == 'GCNadj':
        def foo(dataset):
            return GCNadj(dataset, hidden, dropout=dropout)
    else:
        raise NotImplementedError
    return foo  

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                