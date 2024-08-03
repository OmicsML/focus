import gc
import logging
import math
import os 
from typing import (
    Any, Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Union
    )
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import torch.distributed as dist
import torch.nn.functional as F
import umap

from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GINConv,GCNConv, global_add_pool, \
    global_mean_pool, global_max_pool, EdgeConv, MLP, MessagePassing, GAE, VGAE
    
from .utils import edge_to_intra_inter, process_subbatch
from torch_geometric.utils import batched_negative_sampling

    
class GIN_MLP_Encoder(nn.Module):
    def __init__(
            self, 
            num_features: int,
            dim: int,
            setting = None,
            *args
            ):
        super(GIN_MLP_Encoder, self).__init__()
        
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm1d(dim)
        
        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm1d(dim)
        
        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = BatchNorm1d(dim)
        
        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = BatchNorm1d(dim)
        
        if setting == 'node':
            nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 2))
            self.conv5 = GINConv(nn5)
            self.bn5 = torch.nn.BatchNorm1d(2)
        else:
            nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            self.conv5 = GINConv(nn5)
            self.bn5 = torch.nn.BatchNorm1d(dim)
        
        if setting == 'node':
            pass
        elif setting == 'edge':
            self.fc = Linear(dim*2, 2)
        elif setting == 'subgraph':
            self.fc = Linear(dim, 2)
        elif setting == 'intra_edge':
            self.fc = Linear(dim*2, 2)
        elif setting == 'inter_edge':
            self.pool = global_mean_pool
            self.fc = Linear(dim*4, 2)
        elif setting == 'subgraph_based':
            self.fc = Linear(dim, 6) # subgraph_shuffle/inter_edge/intra_edge/node/none
        elif setting == 'subgraph_based_one':
            self.fc = Linear(dim, 6) # subgraph_shuffle/inter_edge/intra_edge/node/none
            self.pool = global_mean_pool
            self.node_fc = Linear(dim, 2)
            self.intra_edge_fc = Linear(dim*2, 2)
            self.inter_edge_fc = Linear(dim*4, 2)
        else:
            raise ValueError('GIN_Encoder wrong setting: {}'.format(setting))
        
        self.setting = setting
        self.percent = args.percent 
        self.args = args
    
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        
     
        if self.setting == 'node':
            return x
        elif self.setting == 'edge':
            node1 = x[edge_index[0], :]
            node2 = x[edge_index[1], :]
            x = torch.cat([node1, node2], dim=1)
            x = self.fc(x) # edge_num * 2
            return x
        elif self.setting == 'subgraph':
            subsplit = process_subbatch(data.subsplit, data.subsplit_cnt, data.ptr)
            x = global_add_pool(x, batch=subsplit)
            x = self.fc(x) # 6
            return x, subsplit
        elif self.setting == 'intra_edge':
            keep_edge_idx = torch.nonzero(data.intra_edge_mask, as_tuple=False).view(-1,) 
            edge_index = torch.index_select(edge_index, 1, keep_edge_idx) 
            
            # sample edges to drop
            edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) <= self.percent
            other_edges = edge_index[:, ~edge_mask]
            edge_index = edge_index[:, edge_mask]
        
            # negtive sampling
            subsplit = process_subbatch(data.subsplit, data.subsplit_cnt, data.ptr)
            # num_neg_samples=int((data.edge_index.size(1)*self.percent)//(len(data.ptr)-1))
            negative_edge = batched_negative_sampling(data.edge_index, data.batch)#, num_neg_samples=num_neg_samples)
            neg_intra_edge, _ = edge_to_intra_inter(negative_edge, subsplit)
            edge_mask = torch.rand(neg_intra_edge.size(1), device=neg_intra_edge.device) <= self.percent
            neg_intra_edge = neg_intra_edge[:, edge_mask]
            edge_index = torch.cat([edge_index, neg_intra_edge], dim=1)
            
            # edge embedding
            node1 = x[edge_index[0], :]
            node2 = x[edge_index[1], :]
            x = torch.cat([node1, node2-node1], dim=1)
            x = self.fc(x)
            return x, edge_index, other_edges
        elif self.setting == 'inter_edge':
            keep_edge_idx = torch.nonzero(~data.intra_edge_mask, as_tuple=False).view(-1,) 
            edge_index = torch.index_select(edge_index, 1, keep_edge_idx) 
            subsplit = process_subbatch(data.subsplit, data.subsplit_cnt, data.ptr)
            
            # sample edges
            edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) <= self.percent
            other_edges = edge_index[:, ~edge_mask]
            edge_index = edge_index[:, edge_mask]
            
            # negative sampling
            negative_edge = batched_negative_sampling(data.edge_index, data.batch)
            _, neg_inter_edge = edge_to_intra_inter(negative_edge, subsplit)
            edge_mask = torch.rand(neg_inter_edge.size(1), device=neg_inter_edge.device) <= self.percent
            neg_inter_edge = neg_inter_edge[:, edge_mask]
            edge_index = torch.cat([edge_index, neg_inter_edge], dim=1)
            
            # edge embedding
            sub_x = self.pool(x, batch=subsplit)
            node1 = x[edge_index[0], :]
            node1_sub = sub_x[subsplit[edge_index[0]], :]
            node2 = x[edge_index[1], :]
            node2_sub = sub_x[subsplit[edge_index[1]], :]
            
            x = torch.cat([node1, node1_sub, node2-node1, node2_sub-node1_sub], dim=1)
            x = self.fc(x)
            return x, edge_index, other_edges
        elif self.setting == 'subgraph_based':
            subsplit = process_subbatch(data.subsplit, data.subsplit_cnt, data.ptr)
            sub_x = global_add_pool(x, batch=subsplit)
            x = self.fc(sub_x)
            return x, subsplit
        elif self.setting == 'subgraph_based_one':
            # subgraph_select
            subsplit = process_subbatch(data.subsplit, data.subsplit_cnt, data.ptr)
            sub_x = global_add_pool(x, batch=subsplit)
            sub_select_x = self.fc(sub_x)
            
            # node
            node_x = self.node_fc(x)

            # drop edge
            keep_edge_idx = torch.nonzero(data.intra_edge_mask, as_tuple=False).view(-1,) 
            intra_edge_index = torch.index_select(edge_index, 1, keep_edge_idx) 
            keep_edge_idx = torch.nonzero(~data.intra_edge_mask, as_tuple=False).view(-1,) 
            inter_edge_index = torch.index_select(edge_index, 1, keep_edge_idx) 
            subsplit = process_subbatch(data.subsplit, data.subsplit_cnt, data.ptr)

            # sample edges to drop
            edge_mask = torch.rand(intra_edge_index.size(1), device=intra_edge_index.device) <= self.percent
            other_intra_edges = intra_edge_index[:, ~edge_mask]
            intra_edge_index = intra_edge_index[:, edge_mask]
            edge_mask = torch.rand(inter_edge_index.size(1), device=inter_edge_index.device) <= self.percent
            other_inter_edges = inter_edge_index[:, ~edge_mask]
            inter_edge_index = inter_edge_index[:, edge_mask]

            # negtive edge sampling
            subsplit = process_subbatch(data.subsplit, data.subsplit_cnt, data.ptr)
            negtive_edge = batched_negative_sampling(data.edge_index, data.batch)
            neg_intra_edge, neg_inter_edge = edge_to_intra_inter(negtive_edge, subsplit)
            edge_mask = torch.rand(neg_intra_edge.size(1), device=neg_intra_edge.device) <= self.percent
            neg_intra_edge = neg_intra_edge[:, edge_mask]
            edge_mask = torch.rand(neg_inter_edge.size(1), device=neg_inter_edge.device) <= self.percent
            neg_inter_edge = neg_inter_edge[:, edge_mask]
            intra_edge_index = torch.cat([intra_edge_index, neg_intra_edge], dim=1)
            inter_edge_index = torch.cat([inter_edge_index, neg_inter_edge], dim=1)

            # drop intra edge
            node1 = x[intra_edge_index[0], :]
            node2 = x[intra_edge_index[1], :]
            intra_edge = torch.cat([node1, node2-node1], dim=1)
            intra_edge = self.intra_edge_fc(intra_edge)

            # drop inter edge
            sub_x = self.pool(x, batch=subsplit)
            node1 = x[inter_edge_index[0], :]
            node1_sub = sub_x[subsplit[inter_edge_index[0]], :]
            node2 = x[inter_edge_index[1], :]
            node2_sub = sub_x[subsplit[inter_edge_index[1]], :]
            inter_edge = torch.cat([node1, node1_sub, node2-node1, node2_sub-node1_sub], dim=1)
            inter_edge = self.inter_edge_fc(inter_edge)

            return sub_select_x, subsplit,node_x, intra_edge, intra_edge_index, other_intra_edges, \
                inter_edge, inter_edge_index, other_inter_edges
