from functools import partial
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv


class ResGCN(nn.Module):
    def __init__(self, dataset, num_features, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 edge_norm=True):
        super().__init__()
        self.num_features = num_features
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.use_xg = False
        self.res_branch = res_branch
        self.collapse = collapse
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout

        if self.use_xg:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)

        hidden_in = num_features

        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden)
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
                self.convs.append(GCNConv(hidden, hidden))
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
        else:
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = nn.ModuleList()
        self.lins = nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        label_num = [dataset[i].y for i in range(len(dataset))]
        num_clasess = max(label_num) + 1
        self.lin_class = Linear(hidden, num_clasess)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)
        
        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x.ndim == 1:
            x = x.to(torch.int64)
            x =  F.one_hot(x, num_classes=self.num_features)
            x = x.float()
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.res_branch == "BNConvReLU":
            return self.forward_BNConvReLU(x, edge_index, batch, xg)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_BNConvReLU(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
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