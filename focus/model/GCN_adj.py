import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_scipy_sparse_matrix

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv
# from gcn_conv import GCNConv

from IPython import embed

class GCNadj(nn.Module):
    def __init__(self, dataset, hidden, dropout):
        super(GCNadj, self).__init__()
        
        nfeat = dataset.num_features
        self.gc1 = GraphConvolution(nfeat, hidden)
        self.gc2 = GraphConvolution(hidden, hidden)
        self.dropout = dropout
        self.lin_class = nn.Linear(hidden, dataset.num_classes)
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





class ResGCNadj(nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0,
                 edge_norm=True):
        super().__init__()
        
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch
        self.collapse = collapse
        self.use_xg = False
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout


        if self.use_xg:  # Utilize graph level features.
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)
            

        hidden_in = dataset.num_features

        self.bn_feat = BatchNorm1d(hidden_in)
        
        self.conv_feat = GraphConvolution(hidden_in, hidden)
        if "gating" in global_pool:
            self.gating = nn.Sequential(
                Linear(hidden, hidden),
                nn.ReLU(),
                Linear(hidden, 1),
                nn.Sigmoid())
        else:
            self.gating = None
        self.bns_conv = nn.ModuleList()
        self.convs = nn.ModuleList()
        if self.res_branch == "resnet":
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GraphConvolution(hidden, hidden))
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GraphConvolution(hidden, hidden))
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GraphConvolution(hidden, hidden))
        else:
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GraphConvolution(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = nn.ModuleList()
        self.lins = nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, dataset.num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)
        
        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))


    def forward(self, data):
        x, batch = data.x, data.batch
        if 'adj' not in data:
            adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)).to(data.x.device), (data.num_nodes, data.num_nodes), requires_grad=True)#.to_dense()
            # print('-=-=-=adj not in data')
        else:
            adj = data.adj
            # print('*******adj in data')
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.res_branch == "BNConvReLU":
            return self.forward_BNConvReLU(x, adj, batch, xg)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_BNConvReLU(self, x, adj, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, adj))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, adj))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_last_layers(self, data):
        x, adj, batch = data.x, data.adj, data.batch
        if self.use_xg:
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, adj))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, adj))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        
        out1 = self.bn_hidden(x)
        if self.dropout > 0:
            out1 = F.dropout(out1, p=self.dropout, training=self.training)
        
        out2 = self.lin_class(out1)
        out3 = F.log_softmax(out2, dim=-1)
        return out1, out2, out3

    def forward_cl(self, data):
        x, adj, batch = data.x, data.adj, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, adj))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, adj))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        return x
    
    def forward_graph_cl(self, data):
        x, adj, batch = data.x, data.adj, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, adj))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, adj))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x