from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import Dropout, LayerNorm, Linear, Module, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv


class ResGCN(nn.Module):
    def __init__(self, 
                 nfeat: int, 
                 hidden: int, 
                 num_feat_layers: int = 1, 
                 num_conv_layers: int = 3,
                 num_fc_layers: int = 2, 
                #  gfn: bool = False, 
                 collapse: bool = False, 
                 residual: bool = False,
                 res_branch: str = "BNConvReLU", 
                 global_pool: str = "sum", 
                 dropout: int = 0,
                #  edge_norm: bool = True
                 label_num: List[int] = None,
                 ):
        super().__init__()
        
        assert num_feat_layers == 1
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch
        self.collapse = collapse
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout

        # GCNConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        # if "xg" in dataset[0]:  # Utilize graph level features.
        #     self.use_xg = True
        #     self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
        #     self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
        #     self.bn2_xg = BatchNorm1d(hidden)
        #     self.lin2_xg = Linear(hidden, hidden)
        # else:
        #     self.use_xg = False

        hidden_in = nfeat
        self.bn_feat = BatchNorm1d(hidden_in)
        feat_gfn = True  # set true so GCNConv is feat transform
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
            for _ in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GCNConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = nn.ModuleList()
        self.lins = nn.ModuleList()
        for _ in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        # label_num = [dataset[i].y for i in range(len(dataset))]
        num_clasess = max(label_num) + 1
        self.lin_class = Linear(hidden, num_clasess)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)
        
        self.proj_head = nn.Sequential(Linear(hidden, hidden), nn.ReLU(inplace=True), Linear(hidden, hidden))

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
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