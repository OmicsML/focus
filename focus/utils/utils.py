from collections import defaultdict
import os 
import copy 
import random
import numpy as np
import pandas as pd 
import networkx as nx
import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.utils import subgraph

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def subgraph2inter_edge_tensor2(subsplit, edge_index):
    num_rows = max(subsplit)+1 # num: number of subgraphs
    original_matrix = torch.cat([subsplit[edge_index[0]].unsqueeze(0), subsplit[edge_index[1]].unsqueeze(0)])
    num_cols = original_matrix.shape[1]
    row_diff = torch.abs(original_matrix[0] - original_matrix[1])
    # num_rows = torch.max(original_matrix) + 1  # Assuming zero-based indexing
    # Initialize a list to store the non-zero indices
    non_zero_indices = torch.nonzero(row_diff).squeeze()
    indices = original_matrix[:, non_zero_indices]  
    values = torch.ones(indices.size(1), dtype=torch.float32).to(indices.device)
    sparse_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([num_rows, num_cols]))
    # return torch.Tensor(inter_edge_counts)
    return sparse_matrix

def element_wise_multiplication(sparse_matrix, dense_matrix):
    data = sparse_matrix.coalesce().indices() 
    index = torch.nonzero(dense_matrix[data.t()[:, 1]].flatten())[:, 0]
    new_matrix_indices = data.t()[index] 
    return torch.sparse_coo_tensor(new_matrix_indices.t(), \
        torch.ones(new_matrix_indices.shape[0]).to(sparse_matrix.device), sparse_matrix.size())
    
def subgraph2inter_edge_tensor(subsplit, edge_index):
    """
    may not use in focus graph
    """
    num = max(subsplit)+1 # num: number of subgraphs
    result = torch.arange(0,num).view(-1,1).repeat(1, edge_index.size(1)).to(subsplit.device)
    tmp1 = subsplit[edge_index[0]].repeat(num, 1)
    tmp2 = subsplit[edge_index[1]].repeat(num, 1)
    result = (tmp1==result) ^ (tmp2==result)
    return result


def subgraph2node_tensor(subsplit):
    """
    may not use in focus graph
    """
    num = max(subsplit)+1
    tmp = subsplit.repeat(num, 1)
    result = torch.arange(0,num).view(-1,1).repeat(1, subsplit.size(0)).to(subsplit.device)
    result = result == tmp
    return result

def subgraph2inter_edge_tensor(subsplit, edge_index):
    """
    may not use in focus graph
    """
    num = max(subsplit)+1 # num: number of subgraphs
    result = torch.arange(0,num).view(-1,1).repeat(1, edge_index.size(1)).to(subsplit.device)
    tmp1 = subsplit[edge_index[0]].repeat(num, 1)
    tmp2 = subsplit[edge_index[1]].repeat(num, 1)
    result = (tmp1==result) ^ (tmp2==result)
    return result

def shuffle_sub_edge(edge_index, subsplit, batch, replace_node_idx):
    '''
    Given selected subgraph
    shuffle inter-edge in these subgraphs within same graph
    '''
    # print(edge_index[:,:100])
    device = edge_index.device
    edge_index, subsplit, batch, replace_node_idx = edge_index.cpu().numpy(), subsplit.cpu().numpy(), batch.cpu().numpy(), replace_node_idx.cpu().numpy()
    
    sub2node = defaultdict(list)
    sub_batch = defaultdict(list)
    for node in replace_node_idx:
        sub2node[subsplit[node]].append(node)
        if subsplit[node] not in sub_batch[batch[node]]:
            sub_batch[batch[node]].append(subsplit[node])

    node_map = defaultdict(int)
    for sub in sub_batch.values(): # batch
        sub_map = copy.deepcopy(sub)
        random.shuffle(sub_map)
        for i in range(len(sub)): # subgraph
            nodes = sub2node[sub[i]]
            tmp = sub2node[sub_map[i]]
            for j in range(len(nodes)):
                node_map[nodes[j]] = random.choice(tmp)
    
    edge_index = edge_index.reshape(-1)
    result = copy.deepcopy(edge_index)
    for old, new in node_map.items():
        result[edge_index == old] = new
    result = torch.tensor(result).view(2, -1)
    return result.to(device)


def subgraph2intra_edge_tensor(subsplit, edge_index):

    # result is num_node_groups by edge_num matrix

    num = max(subsplit)+1
    result = torch.arange(0,num).view(-1,1).repeat(1, edge_index.size(1)).to(subsplit.device)
    tmp1 = subsplit[edge_index[0]].repeat(num, 1)
    tmp2 = subsplit[edge_index[1]].repeat(num, 1)
    result = (tmp1==result) & (tmp2==result)
    return result
    
# def subgraph_shuffle
def subgraph_shuffle(data, subsplit, sub_shuffle_sample, args):
    edge_index = data.edge_index
    
    # keep edge index
    if args.sparse:
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


# def edge_permutation
def edge_permutation(data, subsplit, sub_intra_edge_sample, sub_inter_edge_sample, encoder_result):
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


def node_drop(data, subsplit, sub_node_sample, requires_grad, node_p, args):
    x = data.x
    x = x.float()
    x.requires_grad = requires_grad
    
    sample = F.gumbel_softmax(node_p, hard=True)
    drop_sample = sample[:,0]
    drop_data = copy.deepcopy(data)
    drop_data.importance_score = node_p[:,0]
    drop_node_mask = torch.rand(drop_sample.size(0), device=drop_sample.device) <= args.percent 
    
    # nodes(drop_sub_node_sample) in selected subgraph(sub_node_sample)
    sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
    selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
    
    # record drop nodes
    drop_data.mask = mask 
        
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

def node_mask(data, subsplit, sub_node_sample, requires_grad, node_p, args):
    x = data.x
    x = x.float()
    x.requires_grad = requires_grad
    
    sample = F.gumbel_softmax(node_p, hard=True)
    mask_data = copy.deepcopy(data)
    mask_data.importance_score = node_p[:,0]
    drop_sample = sample[:,0]
    drop_node_mask = torch.rand(drop_sample.size(0), device=drop_sample.device) <= args.percent 
    
    # nodes(drop_sub_node_sample) in selected subgraph(sub_node_sample)
    sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
    selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
    
    # choose ratio
    mask = drop_sample * drop_node_mask.float() * selected_nodes
    
    keep_mask = 1 - mask
    
    return keep_mask.clone().detach().to(dtype=torch.long), drop_sample#, torch.sparse_coo_tensor((data.num_nodes, data.num_nodes)).to(x.device)
