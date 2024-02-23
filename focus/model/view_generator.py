import gc
import logging
import math
import os 
from typing import (
    Any, Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Union,
)
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import umap

from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GINConv,GCNConv, global_add_pool, global_mean_pool, global_max_pool, \
    EdgeConv, MLP, MessagePassing, GAE, VGAE
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops, negative_sampling, \
    subgraph, to_dense_adj, to_scipy_sparse_matrix, batched_negative_sampling, index_to_mask
    
from ..module.utils import element_wise_multiplication, subgraph2inter_edge_tensor2, subgraph2node_tensor, save_file_with_unique_name, \
    subgraph2intra_edge_tensor, subgraph2inter_edge_tensor, shuffle_sub_edge

from ..model.graph_encoder import GIN_MLP_Encoder

class ViewGenerator_subgraph_based_pipeline(VGAE):
    def __init__(self, 
                 view_graph_num_features,
                 view_graph_dim,
                 view_graph_encoder_s,
                 add_mask=False,
                 args=None):
        self.add_mask = add_mask
        subgraph_encoder = view_graph_encoder_s(view_graph_num_features, view_graph_dim, self.add_mask, 'subgraph_based_one', args)
        super().__init__(encoder=subgraph_encoder)
        
        self.subgraph_encoder = subgraph_encoder
        
        self.percent  = args.percent
        
    def subgraph_shuffle(self, data, subsplit, sub_shuffle_sample):
        edge_index = data.edge_index
        
        # keep edge index
        if self.args.sparse:
            sub2inter_edge = subgraph2inter_edge_tensor2(subsplit, edge_index).to(edge_index.device) 
            # result element-wise multiplication
            result = element_wise_multiplication(sub2inter_edge.t(), sub_shuffle_sample.view(-1, 1)).t()
            drop_inter_edge_sample = torch.sparse.sum(result, dim=0).to_dense()
            drop_inter_edge_sample = torch.where(drop_inter_edge_sample>=1, torch.ones_like(drop_inter_edge_sample), drop_inter_edge_sample)
            drop_inter_edge_adj = torch.sparse_coo_tensor(edge_index, drop_inter_edge_sample, (data.num_nodes, data.num_nodes), requires_grad=True)
        else:
            sub2inter_edge = subgraph2inter_edge_tensor(subsplit, edge_index).to(edge_index.device)
            drop_inter_edge_sample = torch.sum(sub2inter_edge*sub_shuffle_sample.view(-1,1), dim=0)
            drop_inter_edge_sample = torch.where(drop_inter_edge_sample>=1, torch.ones_like(drop_inter_edge_sample), drop_inter_edge_sample)
            drop_inter_edge_adj = torch.sparse_coo_tensor(edge_index, drop_inter_edge_sample, (data.num_nodes, data.num_nodes), requires_grad=True)
        
        # inter_edge_index
        sub2node = subgraph2node_tensor(subsplit).to(edge_index.device)
        selected_nodes = torch.sum(sub2node*sub_shuffle_sample.view(-1,1), dim=0)
        replace_node_idx = torch.nonzero(selected_nodes.bool(), as_tuple=False).view(-1,)
        inter_edge_index = edge_index[:, drop_inter_edge_sample.bool()] # get inter-edge from selected subgraph
        
        # subgraph replace: shuffle inter edge index
        shuffle_inter_ei = shuffle_sub_edge(inter_edge_index, subsplit, data.batch, replace_node_idx)
        shuffle_inter_adj = torch.sparse_coo_tensor(shuffle_inter_ei, torch.ones(shuffle_inter_ei.size(1)).to(shuffle_inter_ei.device), (data.num_nodes, data.num_nodes), requires_grad=True)
        
        perm_adj = (-drop_inter_edge_adj + shuffle_inter_adj).coalesce()
        
        return perm_adj # changes to ori_adj
    
    def edge_permutation(self, data, subsplit, sub_intra_edge_sample, sub_inter_edge_sample, encoder_result):
        edge_index = data.edge_index
        ori_adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), (data.num_nodes, data.num_nodes), requires_grad=True)
        
        intra_p, intra_edge_index, intra_stable_edge_index, inter_p, inter_edge_index, inter_stable_edge_index = encoder_result
        stable_edge_index = torch.cat((intra_stable_edge_index, inter_stable_edge_index), dim=1)
        stable_adj = torch.sparse_coo_tensor(stable_edge_index, torch.ones(stable_edge_index.size(1)).to(edge_index.device), (data.num_nodes, data.num_nodes), requires_grad=True)
        
        # drop_intra_edge_sample in selected subgraph(sub_intra_edge_sample)
        sub2intra_edge = subgraph2intra_edge_tensor(subsplit, intra_edge_index).to(edge_index.device)
        drop_intra_edge_sample = torch.sum(sub2intra_edge*sub_intra_edge_sample.view(-1,1), dim=0)
        
        # drop_inter_edge_sample in selected subgraph(sub_inter_edge_sample)
        sub2inter_edge = subgraph2inter_edge_tensor(subsplit, inter_edge_index).to(edge_index.device)
        drop_inter_edge_sample = torch.sum(sub2inter_edge*sub_inter_edge_sample.view(-1,1), dim=0)
        drop_inter_edge_sample = torch.where(drop_inter_edge_sample==1, torch.zeros_like(drop_inter_edge_sample), drop_inter_edge_sample)
        drop_inter_edge_sample = drop_inter_edge_sample/2
        
        # concaten intra & inter edges in subgraph
        sub_drop_sample = torch.cat((drop_intra_edge_sample, drop_inter_edge_sample), dim=0)

        perm_edge_index = torch.cat((intra_edge_index, inter_edge_index), dim=1)
        p = torch.cat((intra_p, inter_p), dim=0)
        
        # gumbel softmax
        sample = F.gumbel_softmax(p, hard=True)
        edge_sample = sample[:,0]
        
        final_sample = torch.matmul(edge_sample, torch.diag_embed(sub_drop_sample))
        shuffle_adj = torch.sparse_coo_tensor(perm_edge_index, final_sample, (data.num_nodes, data.num_nodes), requires_grad=True)#.to_dense()
        
        perm_adj = (shuffle_adj + stable_adj - ori_adj).coalesce()
        
        return perm_adj# changes to ori_adj
    
    def node_drop(self, data, subsplit, sub_node_sample, requires_grad, node_p):
        x = data.x
        x = x.float()
        x.requires_grad = requires_grad
        
        sample = F.gumbel_softmax(node_p, hard=True)
        
        drop_data = copy.deepcopy(data)
        drop_data.importance_score = node_p[:,0]
        
        drop_sample = sample[:,0]
        drop_node_mask = torch.rand(drop_sample.size(0), device=drop_sample.device) <= self.percent 
        
        # nodes(drop_sub_node_sample) in selected subgraph(sub_node_sample)
        sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
        selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
        
        # record drop nodes
        drop_data.mask = mask 
        if drop_data.epoch[0].item() >= self.save_epoch:
            drop_data_list = self.batch_split(drop_data, "node_level")
            self.save_node(drop_data_list, 'node_drop')
            
        # choose ratio
        mask = drop_sample * drop_node_mask.float() * selected_nodes
        
        keep_mask = 1 - mask
        x = x * keep_mask.view(-1, 1)
        
        # subgraph
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1,)
        edge_index, edge_attr, edge_mask = subgraph(keep_idx, data.edge_index, num_nodes=data.num_nodes, return_edge_mask=True)
        edge_index_tmp = data.edge_index[:, ~edge_mask]
        node_adj = torch.sparse_coo_tensor(edge_index_tmp, torch.ones(edge_index_tmp.size(1)).to(edge_index_tmp.device), (data.num_nodes, data.num_nodes), requires_grad=True) # drop edge
        
        return x, drop_sample, -node_adj
    
    def node_mask(self, data, subsplit, sub_node_sample, requires_grad, node_p):
        x = data.x
        x = x.float()
        x.requires_grad = requires_grad
        
        sample = F.gumbel_softmax(node_p, hard=True)
        mask_data = copy.deepcopy(data)
        mask_data.importance_score = node_p[:,0]
        drop_sample = sample[:,0]
        drop_node_mask = torch.rand(drop_sample.size(0), device=drop_sample.device) <= self.percent 
        
        # nodes(drop_sub_node_sample) in selected subgraph(sub_node_sample)
        sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
        selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
        
        # choose ratio
        mask = drop_sample * drop_node_mask.float() * selected_nodes
        
        # record mask nodes
        mask_data.mask = mask 
        if mask_data.epoch[0].item() >= self.save_epoch:
            mask_data_list = self.batch_split(mask_data, "node_level")
            self.save_node(mask_data_list, 'node_mask')
        keep_mask = 1 - mask
        
        return keep_mask.clone().detach().to(dtype=torch.long), drop_sample#, torch.sparse_coo_tensor((data.num_nodes, data.num_nodes)).to(x.device)

    def save_node(self, batch_splits, aug_type):
        for graph in batch_splits:
            # (edge_mask, '{}/one_graph_mask/{}_{}.mat'.format(save_path, dataset_name, idx))
            node_path = os.path.join(self.args.data_path, 'draw')
            if not os.path.exists(node_path):
                os.makedirs(node_path)
            graph_path = "./{}/draw/{}_{}.pt".format(self.args.data_path, graph.name, aug_type)
            graph_path = save_file_with_unique_name(graph_path)
            torch.save(graph, graph_path)
    def batch_split(self, data, aug_level):
        # assert self.args.batch_size == len(data.name)
        num_batches = len(data.name)
        batch_split_list = []
        for i in range(num_batches):
            if aug_level == "node_level":
                test = Data(x = torch.argmax(data.x[data.ptr[i]:data.ptr[i+1]],dim=1),
                            importance_score = data.importance_score[data.ptr[i]:data.ptr[i+1]],
                            mask = data.mask[data.ptr[i]:data.ptr[i+1]],
                            # y = data.y[data.ptr[i]:data.ptr[i+1]],
                            y = data.y[i],
                            pos = data.pos[data.ptr[i]:data.ptr[i+1]],
                            name = data.name[i])
                test = test.cpu().detach()
                batch_split_list.append(test)
            elif aug_level == "subgraph_level":
                if i == 0:
                    test = Data(subgraph_score = data.subgraph_score[:data.subsplit_cnt[i]],
                                y = data.y[i],
                                name = data.name[i])
                else:
                    test = Data(subgraph_score = data.subgraph_score[data.subsplit_cnt[i-1]:(data.subsplit_cnt[i-1] + data.subsplit_cnt[i])],
                                y = data.y[i],
                                name = data.name[i])
                test = test.cpu().detach()
                batch_split_list.append(test)
        return batch_split_list

    def save_subgraph(self, batch_splits):
        for graph in batch_splits:
            subgraph_path = os.path.join(self.args.data_path, 'draw_subgraph')
            if not os.path.exists(subgraph_path):
                os.makedirs(subgraph_path)
            graph_path = "./{}/draw_subgraph/{}_subgraph.pt".format(self.args.data_path, graph.name)
            graph_path = save_file_with_unique_name(graph_path)
            torch.save(graph, graph_path)
            
    def forward(self, data_in, requires_grad):
        data = copy.deepcopy(data_in)
        
        x, edge_index = data.x, data.edge_index
        
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = data.edge_attr

        data.x = data.x.float()
        x = x.float()
        x.requires_grad = requires_grad
        
        subgraph_p, subsplit, node_p, intra_p, intra_edge_index, intra_stable_edge_index, inter_p, inter_edge_index, inter_stable_edge_index = self.subgraph_encoder(data)
        sample = F.gumbel_softmax(subgraph_p, hard=True) # subgraph_shuffle/inter_edge/intra_edge/node/none
        sub_shuffle_sample = sample[:,0] 
        intra_edge_sample = sample[:,1] 
        inter_edge_sample = sample[:,2] 
        node_sample = sample[:,3] 
        node_mask_sample = sample[:,4] 
        none = sample[:,5]
        
        # subgraph_shuffle
        # if sub_shuffle_sample.any()>0:
        if (sub_shuffle_sample>0).any():
            sub_adj = self.subgraph_shuffle(data, subsplit, sub_shuffle_sample)
            # save subgraph
            subgraph_data = copy.deepcopy(data)
            subgraph_data.subgraph_score = subgraph_p[:,0]
            if subgraph_data.epoch[0].item() >= self.save_epoch:
                subgraph_data_list = self.batch_split(subgraph_data, "subgraph_level")
                self.save_subgraph(subgraph_data_list)
        else:
            sub_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
        
        # intra/inter_edge
        if (intra_edge_sample>0).any() or (inter_edge_sample>0).any():
            edge_encoder_result = (intra_p, intra_edge_index, intra_stable_edge_index, inter_p, inter_edge_index, inter_stable_edge_index)
            edge_adj = self.edge_permutation(data, subsplit, intra_edge_sample, inter_edge_sample, edge_encoder_result)
        else:
            edge_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
        
        # node drop
        if (node_sample>0).any():
            x, drop_node_sample, node_adj = self.node_drop(data, subsplit, node_sample, requires_grad, node_p)
        else:
            x = x
            drop_node_sample = torch.zeros_like(node_sample).to(data.edge_index.device)
            node_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
            
        
        # node mask
        if (node_sample>0).any():
            keep_x_mask, mask_node_sample = self.node_mask(data, subsplit, node_mask_sample, requires_grad, node_p)
            x[keep_x_mask] = data.x.detach().mean()
        else:
            x = x
            mask_node_sample = torch.zeros_like(node_mask_sample).to(data.edge_index.device)
    
        ori_adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), (data.num_nodes, data.num_nodes), requires_grad=True)
        adj = (ori_adj + sub_adj + edge_adj + node_adj).coalesce()
        
        values = adj.values()
        values = torch.where(values>1, torch.ones_like(values), values)
        values = torch.where(values<0, torch.zeros_like(values), values)
        
        norm_adj = torch.sparse_coo_tensor(adj.indices(), values, (data.num_nodes, data.num_nodes), requires_grad=True)

        data.adj = norm_adj
        data.x = x
        return data, (adj, sample, drop_node_sample, mask_node_sample)


class ViewGenerator_subgraph_based_one(nn.Module):
    def __init__(self, 
                 view_graph_num_features,
                 view_graph_dim,
                 encoder_s: nn.Module = GIN_MLP_Encoder,
                 setting: Literal['node','edge','subgraph_based_one'] = 'subgraph_based_one',
                 args=None):
        view_graph_encoder = encoder_s(num_features=view_graph_num_features, 
                                       dim = view_graph_dim, 
                                       setting = setting,
                                       args = args)
        super().__init__(encoder=view_graph_encoder)
        self.view_graph_encoder = view_graph_encoder
        self.args = args
        self.setting = setting
    
    
    @staticmethod
    def node_setting(x, edge_index): # mask 
        def node_mask(self, x, edge_index):
            sample = F.gumbel_softmax(x, hard=True)
            sample_score = sample[:,0]
            
            drop_node_mask = torch.rand(sample_score.size(0), device=sample_score.device) <= 0.1
            mask = sample_score * drop_node_mask.float() * x[:, 0]
            
            keep_mask = 1 - mask
            x = x * keep_mask.view(-1, 1)
            return x, edge_index
        
        return node_mask(x, edge_index)
    
    @staticmethod
    def edge_setting(data):
        x, edge_index = data.x, data.edge_index
        # edge_dropping or edge perturbation
        # sample = F.gumbel_softmax(x, hard=True)
        pass
        
        
        
            
        
    def forward(self, data_in, requires_grad):
        data = copy.deepcopy(data_in)
        x, edge_index = data.x, data.edge_index

        data.x = data.x.float()
        x = x.float()
        x.requires_grad = requires_grad
        
        if self.setting == "node":
            return self.node_setting(x, edge_index)
        elif self.setting == "edge":
            return self.edge_setting(data)
        else:
            raise ValueError("setting should be node or edge")
