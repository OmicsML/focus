import copy 
import os  
import sys 
sys.append(os.getcwd())
import numpy as np
import pandas as pd 
import networkx as nx

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, EdgeConv, MLP, MessagePassing
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import subgraph, batched_negative_sampling

from utils import *


class GIN_MLP_Encoder(nn.Module):
    def __init__(self, num_features, dim, add_mask=False, setting=None, args=None):
        super().__init__()
        self.num_features = num_features
        nn1 = Sequential(Linear(self.num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        
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
        x = x.to(torch.int64)
        x =  F.one_hot(x, num_classes=self.num_features)
        x = x.float()
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
            x = self.fc(x)
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
            negtive_edge = batched_negative_sampling(data.edge_index, data.batch)#, num_neg_samples=num_neg_samples)
            neg_intra_edge, _ = edge_to_intra_inter(negtive_edge, subsplit)
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
            # negtive sampling
            negtive_edge = batched_negative_sampling(data.edge_index, data.batch)
            _, neg_inter_edge = edge_to_intra_inter(negtive_edge, subsplit)
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

            return sub_select_x, subsplit,node_x, intra_edge, intra_edge_index, other_intra_edges, inter_edge, inter_edge_index, other_inter_edges


class ViewGenerator_subgraph_based_pipeline(VGAE):
    def __init__(self, num_features, dim, encoder, add_mask=False, args=None):
        self.add_mask = add_mask
        subgraph_encoder = encoder(num_features, dim, add_mask, 'subgraph_based', args)
        super().__init__(encoder=subgraph_encoder)
        
        self.subgraph_encoder = subgraph_encoder
        self.intra_encoder = encoder(num_features, dim, add_mask, 'intra_edge', args)
        self.inter_encoder = encoder(num_features, dim, add_mask, 'inter_edge', args)
        self.node_encoder = encoder(num_features, dim, add_mask, 'node', args)
        # Save the encoders as attributes
        self.percent  = args.percent
        self.save_epoch = args.save_epoch
        self.args = args
        
        
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
        shuffle_node_sample = torch.sum(sub2node*sub_shuffle_sample.view(-1,1), dim=0)
        replace_node_idx = torch.nonzero(shuffle_node_sample.bool(), as_tuple=False).view(-1,)
        keep_node_index = torch.nonzero(~shuffle_node_sample.bool(), as_tuple=False).view(-1,)
        inter_edge_index = edge_index[:, drop_inter_edge_sample.bool()] # get inter-edge from selected subgraph
        # subgraph replace: shuffle inter edge index
        shuffle_inter_ei = perm_sub_edge(inter_edge_index, subsplit, data.batch, sub2node, sub_shuffle_sample, replace_node_idx, keep_node_index)
        shuffle_inter_adj = torch.sparse_coo_tensor(shuffle_inter_ei, drop_inter_edge_sample[torch.nonzero(drop_inter_edge_sample).view(-1,)], (data.num_nodes, data.num_nodes), requires_grad=True)
        perm_adj = (-drop_inter_edge_adj + shuffle_inter_adj).coalesce()                
        return perm_adj # changes to ori_adj
    
    def edge_permutation(self, data, subsplit, sub_intra_edge_sample, sub_inter_edge_sample):
        edge_index = data.edge_index
        ori_adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), (data.num_nodes, data.num_nodes), requires_grad=True)
        
        intra_p, intra_edge_index, intra_stable_edge_index = self.intra_encoder(data) # edge drop probability
        inter_p, inter_edge_index, inter_stable_edge_index = self.inter_encoder(data) # edge drop probability
        stable_edge_index = torch.cat((intra_stable_edge_index, inter_stable_edge_index), dim=1)
        stable_adj = torch.sparse_coo_tensor(stable_edge_index, torch.ones(stable_edge_index.size(1)).to(edge_index.device), (data.num_nodes, data.num_nodes), requires_grad=True)
        
        # drop_intra_edge_sample in selected subgraph(sub_intra_edge_sample)
        "sparse tensor"
        # sub2intra_edge = subgraph2intra_edge_tensor2(subsplit, intra_edge_index).to(edge_index.device)
        # sub_intra_edge_sample_new = sub_intra_edge_sample.view(-1,1)
        # result = element_wise_multiplication(sub2intra_edge.t(), sub_intra_edge_sample_new).t()
        # drop_intra_edge_sample = torch.sparse.sum(result, dim=0).to_dense()
        "dense tensor"        
        sub2intra_edge = subgraph2intra_edge_tensor(subsplit, intra_edge_index).to(edge_index.device)
        drop_intra_edge_sample = torch.sum(sub2intra_edge*sub_intra_edge_sample.view(-1,1), dim=0)
        "dense tensor"
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
        # final_sample = torch.matmul(edge_sample, torch.diag_embed(sub_drop_sample))
        final_sample = edge_sample * sub_drop_sample
        shuffle_adj = torch.sparse_coo_tensor(perm_edge_index, final_sample, (data.num_nodes, data.num_nodes), requires_grad=True)#.to_dense()
        perm_adj = (shuffle_adj + stable_adj - ori_adj).coalesce()
        return perm_adj # changes to ori_adj
    
    def node_drop(self, data, subsplit, sub_node_sample, requires_grad):
        x = data.x
        x = x.float()
        x.requires_grad = requires_grad
        
        p = self.node_encoder(data)
        sample = F.gumbel_softmax(p, hard=True)
        # save nodes importance score
        drop_data = copy.deepcopy(data)
        drop_data.importance_score = p[:,0]
        drop_sample = sample[:,0]
        drop_node_mask = torch.rand(drop_sample.size(0), device=drop_sample.device) <= self.percent 
        # nodes(drop_sub_node_sample) in selected subgraph(sub_node_sample)
        sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
        selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
        # choose ratio
        mask = drop_sample * drop_node_mask.float() * selected_nodes
        # record drop nodes
        drop_data.mask = mask 
        if drop_data.epoch[0].item() >= self.save_epoch:
            drop_data_list = self.batch_split(drop_data, "node_level")
            self.save_node(drop_data_list, 'node_drop')
        keep_mask = 1 - mask
        x = x * keep_mask.view(-1, 1)
        # subgraph
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1,)
        edge_index, edge_attr, edge_mask = subgraph(keep_idx, data.edge_index, num_nodes=data.num_nodes, return_edge_mask=True)
        edge_index_tmp = data.edge_index[:, ~edge_mask]
        node_adj = torch.sparse_coo_tensor(edge_index_tmp, torch.ones(edge_index_tmp.size(1)).to(edge_index_tmp.device), \
            (data.num_nodes, data.num_nodes), requires_grad=True) # drop edge
        return x, drop_sample, -node_adj
    
    def node_mask(self, data, subsplit, sub_node_sample, requires_grad):
        x = data.x
        x = x.float()
        x.requires_grad = requires_grad
        p = self.node_encoder(data)
        sample = F.gumbel_softmax(p, hard=True)
        # save nodes importance score
        mask_data = copy.deepcopy(data)
        mask_data.importance_score = p[:,0]
        drop_sample = sample[:,0]
        
        drop_node_mask = torch.rand(drop_sample.size(0), device=drop_sample.device) <= self.percent 
        # nodes(drop_sub_node_sample) in selected subgraph(sub_node_sample)
        sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
        selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
        mask = drop_sample * drop_node_mask.float() * selected_nodes
        mask_data.mask = mask 

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
                    test = Data(subgraph_score = data.subgraph_score[:data.subsplit_cnt[i]], y = data.y[i],
                                name = data.name[i])
                else:
                    test = Data(subgraph_score = data.subgraph_score[data.subsplit_cnt[i-1]:(data.subsplit_cnt[i-1] + data.subsplit_cnt[i])], y = data.y[i],
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
        data.x = data.x.float()
        x = x.float()
        x.requires_grad = requires_grad
        # subgraph-based encoder
        subgraph_p, subsplit = self.subgraph_encoder(data)
        sample = F.gumbel_softmax(subgraph_p, hard=True) # subgraph_shuffle/inter_edge/intra_edge/node/none
        sub_shuffle_sample = sample[:,0] 
        intra_edge_sample = sample[:,1] 
        inter_edge_sample = sample[:,2] 
        node_sample = sample[:,3]
        node_mask_sample = sample[:,4]
        none = sample[:,5]
        
        with torch.no_grad():
            sample_score = F.gumbel_softmax(subgraph_p, hard=False)
            # print(sample_score)
        # subgraph data
        subgraph_data = copy.deepcopy(data)
        # with torch.no_grad():
        #     subgraph_data.subgraph_score = F.gumbel_softmax(subgraph_p, hard=False)[:,0]
        # save subgraph
        # subgraph_data.subgraph_score = subgraph_p[:,0]
        # if subgraph_data.epoch[0].item() >= self.save_epoch:
        #     subgraph_data_list = self.batch_split(subgraph_data, "subgraph_level")
        #     self.save_subgraph(subgraph_data_list)

        # subgraph_shuffle
        sub_adj = self.subgraph_shuffle(data, subsplit, sub_shuffle_sample)
        # intra/inter_edge
        if (intra_edge_sample>0).any() or (inter_edge_sample>0).any():
            edge_adj = self.edge_permutation(data, subsplit, intra_edge_sample, inter_edge_sample)
        else:
            edge_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
        # node drop
        x, drop_node_sample, node_adj = self.node_drop(data, subsplit, node_sample, requires_grad)        
        # node mask
        keep_x_mask, mask_node_sample = self.node_mask(data, subsplit, node_mask_sample, requires_grad)
        keep_mask_tmp = torch.ones(data.x.size(0), dtype=torch.long).to(data.edge_index.device)
        x = x * keep_mask_tmp.view(-1, 1)
        x[torch.nonzero(keep_x_mask == 0).squeeze()] = data.x.detach().mean()
        ori_adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), (data.num_nodes, data.num_nodes), requires_grad=True)
        adj = (ori_adj + sub_adj + edge_adj + node_adj).coalesce()
        values = adj.values()
        values = torch.where(values>1, torch.ones_like(values), values)
        values = torch.where(values<0, torch.zeros_like(values), values)
        norm_adj = torch.sparse_coo_tensor(adj.indices(), values, (data.num_nodes, data.num_nodes), requires_grad=True)
        data.adj = norm_adj
        data.x = x
        return data, (adj, sample, drop_node_sample, mask_node_sample)


class ViewGenerator_subgraph_based_one(VGAE):
    def __init__(self, 
                 num_features, dim, encoder, add_mask=False, args=None):
        self.add_mask = add_mask
        subgraph_encoder = encoder(num_features, dim, self.add_mask, 'subgraph_based_one', args)
        super().__init__(encoder=subgraph_encoder)
        self.subgraph_encoder = subgraph_encoder
        self.percent  = args.percent
        self.args = args
        
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
    
    # NOTE: edge_permutation 和 pipeline的区别
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
        # concatenate intra & inter edges in subgraph
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
        drop_sample = sample[:,0]
        drop_node_mask = torch.tensor([1] * int(drop_sample.size(0) * self.percent) + \
            [0] * (drop_sample.size(0) - int(drop_sample.size(0) * self.percent)), device = drop_sample.device)
        drop_node_mask = drop_node_mask[torch.randperm(drop_node_mask.size(0))]

        sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
        selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
        mask = drop_sample * drop_node_mask.float() * selected_nodes
        # choose ratio
        mask = drop_sample * drop_node_mask.float() * selected_nodes
        keep_mask = 1 - mask
        x = x * keep_mask.view(-1, 1)
        # subgraph
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1,)
        _, _, edge_mask = subgraph(keep_idx, data.edge_index, num_nodes=data.num_nodes, return_edge_mask=True)
        edge_index_tmp = data.edge_index[:, ~edge_mask]
        node_adj = torch.sparse_coo_tensor(edge_index_tmp, torch.ones(edge_index_tmp.size(1)).to(edge_index_tmp.device), (data.num_nodes, data.num_nodes), requires_grad=True) # drop edge
        
        return x, drop_sample, -node_adj
    
    def node_mask(self, data, subsplit, sub_node_sample, requires_grad, node_p):
        x = data.x
        x = x.float()
        x.requires_grad = requires_grad
        
        sample = F.gumbel_softmax(node_p, hard=True)
        mask_sample = sample[:,0]
        node_mask = torch.tensor([1] * int(mask_sample.size(0) * self.percent) + \
            [0] * (mask_sample.size(0) - int(mask_sample.size(0) * self.percent)), device = mask_sample.device)
        node_mask = node_mask[torch.randperm(node_mask.size(0))]
        # nodes(drop_sub_node_sample) in selected subgraph(sub_node_sample)
        sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
        selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
        # choose ratio
        mask = mask_sample * node_mask.float() * selected_nodes
        keep_mask = 1 - mask
        return keep_mask.clone().detach().to(dtype=torch.long), mask_sample#, torch.sparse_coo_tensor((data.num_nodes, data.num_nodes)).to(x.device)    
            
    def forward(self, data_in, requires_grad):
        data = copy.deepcopy(data_in)
        x, edge_index = data.x, data.edge_index
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
        # determine which augmentation to use
        is_subgraph_shuffle = self.args.is_subgraph_shuffle
        is_edge_permutation = self.args.is_edge_permutation
        is_node_drop = self.args.is_node_drop
        is_node_mask = self.args.is_node_mask
        # subgraph_shuffle
        # if sub_shuffle_sample.any()>0:
        if (sub_shuffle_sample>0).any() and is_subgraph_shuffle:
            sub_adj = self.subgraph_shuffle(data, subsplit, sub_shuffle_sample)
        else:
            sub_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
        # intra/inter_edge
        if ((intra_edge_sample>0).any() or (inter_edge_sample>0).any()) and is_edge_permutation:
            edge_encoder_result = (intra_p, intra_edge_index, intra_stable_edge_index, inter_p, inter_edge_index, inter_stable_edge_index)
            edge_adj = self.edge_permutation(data, subsplit, intra_edge_sample, inter_edge_sample, edge_encoder_result)
        else:
            edge_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
        # node drop
        if (node_sample>0).any() and is_node_drop:
            x, drop_node_sample, node_adj = self.node_drop(data, subsplit, node_sample, requires_grad, node_p)
        else:
            x = x
            drop_node_sample = torch.zeros_like(node_sample).to(data.edge_index.device)
            node_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
        # node mask
        if (node_sample>0).any() and is_node_mask:
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


class ViewGenerator_based_one(VGAE):
    def __init__(self, num_features,dim,encoder,setting = "node",add_mask = False,args = None):
        super().__init__(encoder=encoder(num_features, dim, add_mask, setting, args))
        if setting == "node":
            if args.aug_type == "all":
                self.node_drop_encoder = encoder(num_features, dim, add_mask, setting, args)
                self.node_mask_encoder = encoder(num_features, dim, add_mask, setting, args)
            elif args.aug_type == "node_drop":
                self.node_drop_encoder = encoder(num_features, dim, add_mask, setting, args)
            elif args.aug_type == "node_mask":
                self.node_mask_encoder = encoder(num_features, dim, add_mask, setting, args)
            else:
                raise ValueError("Invalid aug_type")
        if setting == "edge":
            self.edge_permutation_encoder = encoder(num_features, dim, add_mask, setting, args)
        self.num_features = num_features
        self.aug_type = args.aug_type
        self.save_epoch = args.save_epoch
        self.setting = setting
        self.percent = args.percent
        self.args = args
    
    def node_drop(self, data, node_sample, requires_grad):
        x = data.x
        x = x.to(torch.int64)
        x =  F.one_hot(x, num_classes=self.num_features)
        x = x.float()
        
        x.requires_grad = requires_grad
        
        drop_node_mask = torch.tensor([1] * int(node_sample.size(0) * self.percent) + [0] * (node_sample.size(0) - int(node_sample.size(0) * self.percent)), device = node_sample.device)
        drop_node_mask = drop_node_mask[torch.randperm(drop_node_mask.size(0))]
        mask = node_sample * drop_node_mask.float()
        keep_mask = 1 - mask
        x = x * keep_mask.view(-1, 1)
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1,)
        _, _, edge_mask = subgraph(keep_idx, data.edge_index, num_nodes=data.num_nodes, return_edge_mask=True)
        edge_index_tmp = data.edge_index[:, ~edge_mask]
        node_adj = torch.sparse_coo_tensor(edge_index_tmp, torch.ones(edge_index_tmp.size(1)).to(edge_index_tmp.device), \
            (data.num_nodes, data.num_nodes), requires_grad=True) # drop edge
        return x, -node_adj
    
    def node_mask(self, node_sample):
        node_mask = torch.tensor([1] * int(node_sample.size(0) * self.percent) + [0] * (node_sample.size(0) - int(node_sample.size(0) * self.percent)), device = node_sample.device)
        node_mask = node_mask[torch.randperm(node_mask.size(0))]
        mask = node_sample * node_mask.float()
        keep_mask = 1 - mask
        return keep_mask.clone().detach().to(dtype=torch.long)
    
    def edge_permutation(self, data, edge_sample, requires_grad):
        edge_index = data.edge_index
        ori_adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), \
            (data.num_nodes, data.num_nodes), requires_grad=requires_grad)
        edge_sample = edge_sample.view(-1, 1)
        edge_mask = torch.tensor([1] * int(edge_index.size(1) * self.percent) + [0] * (edge_index.size(1) - int(edge_index.size(1) * self.percent)), \
            device = edge_index.device)
        edge_mask = edge_mask[torch.randperm(edge_mask.size(0))]
        keep_mask = 1 - edge_mask
        new_edge_index = edge_index.t()[keep_mask.bool()].t()
        return new_edge_index
    
    def record_scores(self, data, sample_p, subsplit_id):
        with torch.no_grad():
            sample_score = F.gumbel_softmax(sample_p, hard=False)
            sample_score = sample_score[:, 1]
            aggregated_scores = torch.zeros(sum(data.subsplit_cnt), device=sample_score.device)
            weights = torch.full((sample_score.size(0),), 1, device=sample_score.device)
            aggregated_scores.scatter_add_(0, subsplit_id, sample_score*weights)
            aggregated_scores = aggregated_scores / torch.bincount(subsplit_id)
            data_list = []
            for i in range(len(data.ptr)-1):
                if i == 0:
                    subgraph_score = aggregated_scores[:data.subsplit_cnt[i]]
                    split_single_id = data.subsplit[data.ptr[i]:data.ptr[i+1]]
                    single_pos = data.pos[data.ptr[i]:data.ptr[i+1]]
                    gene_score = sample_score[data.ptr[i]:data.ptr[i+1]]
                    # single_gene_id = torch.argmax(data.x[data.ptr[i]:data.ptr[i+1]], axis=1)
                    single_gene_id = data.x[data.ptr[i]:data.ptr[i+1]].int()
                    single_cell_id =data.y[i]
                else:
                    subgraph_score = aggregated_scores[data.subsplit_cnt[i-1]:data.subsplit_cnt[i-1]+data.subsplit_cnt[i]]
                    single_cell_id =data.y[i]
                    
                    if i != len(data.ptr) -1:
                        split_single_id = data.subsplit[data.ptr[i]:data.ptr[i+1]]
                        single_pos = data.pos[data.ptr[i]:data.ptr[i+1]]
                        gene_score = sample_score[data.ptr[i]:data.ptr[i+1]]
                        # single_gene_id = torch.argmax(data.x[data.ptr[i]:data.ptr[i+1]], axis=1)
                        single_gene_id = data.x[data.ptr[i]:data.ptr[i+1]].int()
                    else:
                        split_single_id = data.subsplit[data.ptr[i]:]
                        single_pos = data.pos[data.ptr[i]:]
                        gene_score = sample_score[data.ptr[i]:]
                        # single_gene_id = torch.argmax(data.x[data.ptr[i]:], axis=1)
                        single_gene_id = data.x[data.ptr[i]:].int()
                
                start_node = data.ptr[i]
                end_node = data.ptr[i+1]
                mask = (data.edge_index[0] >= start_node) & (data.edge_index[0] < end_node)
                current_edge_index = data.edge_index[:, mask] - start_node

                single_data = Data(gene_id = single_gene_id, pos = single_pos, edge_index = current_edge_index, gene_score = gene_score, subgraph_score = subgraph_score, \
                    gene_subgraph_id = split_single_id, cell_id = single_cell_id, name = data.name[i], epoch = data.epoch[0].item())
                data_list.append(single_data.cpu().detach())
                
        return data_list
    
    def forward(self, data_in, requires_grad):
        data = copy.deepcopy(data_in)
        
        x, edge_index = data.x, data.edge_index
        x = x.to(torch.int64)
        x =  F.one_hot(x, num_classes=self.num_features)
        x = x.float()

        if self.setting == "node" and self.aug_type == "all":
            # node drop
            node_drop_sample_p = self.node_drop_encoder(data)
            node_drop_sample = F.gumbel_softmax(node_drop_sample_p, hard=True)
            node_drop_sample = node_drop_sample[:, 0]

            # node mask
            node_mask_sample_p = self.node_mask_encoder(data)
            node_mask_sample = F.gumbel_softmax(node_mask_sample_p, hard=True)
            node_mask_sample = node_mask_sample[:, 0]
            
            x, drop_adj = self.node_drop(data, node_drop_sample, requires_grad)
            keep_mask = self.node_mask(node_mask_sample)
            # x[keep_mask] = data.x.detach().mean()
            x[keep_mask] = x.mean()
            
            ori_adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), (data.num_nodes, data.num_nodes), requires_grad=True)
            adj = (ori_adj + drop_adj).coalesce()
            values = adj.values()
            values = torch.where(values>1, torch.ones_like(values), values)
            values = torch.where(values<0, torch.zeros_like(values), values)
            norm_adj = torch.sparse_coo_tensor(adj.indices(), values, (data.num_nodes, data.num_nodes), requires_grad=True)
            data.adj = norm_adj
            data.x = x
            return data, (adj, node_drop_sample, node_mask_sample)
        
        elif self.setting == "node" and self.aug_type == "node_drop":
            # node drop
            node_drop_sample_p = self.node_drop_encoder(data)
            node_drop_sample = F.gumbel_softmax(node_drop_sample_p, hard=True)
            node_drop_sample = node_drop_sample[:, 0]
            
            if data.epoch[0].item() >= self.save_epoch:
                subsplit_id = torch.zeros(data.subsplit.size(0),dtype=torch.int64, device=node_drop_sample.device)
                for i in range(len(data.ptr)-1):
                    subsplit_id[data.ptr[i]:data.ptr[i+1]] = data.subsplit[data.ptr[i]: data.ptr[i+1]] + sum(data.subsplit_cnt[:i])
                    if i == len(data.ptr)-1:
                        subsplit_id[data.ptr[i]:] = data.subsplit[data.ptr[i]: ] + sum(data.subsplit_cnt[:i])
                data_list = self.record_scores(data, node_drop_sample_p, subsplit_id)
                
            x, drop_adj = self.node_drop(data, node_drop_sample, requires_grad)
            ori_adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), (data.num_nodes, data.num_nodes), requires_grad=True)
            adj = (ori_adj + drop_adj).coalesce()
            values = adj.values()
            values = torch.where(values>1, torch.ones_like(values), values)
            values = torch.where(values<0, torch.zeros_like(values), values)
            norm_adj = torch.sparse_coo_tensor(adj.indices(), values, (data.num_nodes, data.num_nodes), requires_grad=True)
            data.adj = norm_adj
            data.x = x
            if data.epoch[0].item() < self.save_epoch:
                return data, (adj, node_drop_sample)
            else:
                return data, data_list

        elif self.setting == "node" and self.aug_type == "node_mask":
            # node mask
            node_mask_sample_p = self.node_mask_encoder(data)
            node_mask_sample = F.gumbel_softmax(node_mask_sample_p, hard=True)
            node_mask_sample = node_mask_sample[:, 0]
            
            keep_mask = self.node_mask(node_mask_sample)
            # x[keep_mask] = data.x.detach().mean()
            x[keep_mask] = x.mean()
            data.x = x
            return data, (node_mask_sample)
        
        elif self.setting == "edge":
            edge_sample_p = self.edge_permutation_encoder(data)
            edge_sample = F.gumbel_softmax(edge_sample_p, hard=True)
            drop_edge_sample = edge_sample[:,0]
            new_edge_index = self.edge_permutation(data, drop_edge_sample, requires_grad)
            new_adj = torch.sparse_coo_tensor(new_edge_index, torch.ones(new_edge_index.size(1)).to(new_edge_index.device), (data.num_nodes, data.num_nodes), requires_grad=True)
            data.adj = new_adj
            return data, (new_adj, edge_sample)
            

        
        
            