"""utils for SubcellularGCL"""
import os
import pandas as pd 
import numpy as np
import copy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from communities.algorithms import louvain_method
import networkx as nx
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, to_dense_adj, remove_self_loops, subgraph
import random
from collections import defaultdict, Counter

def unique_list_mapping_to_one_hot(unique_list, target_list):
    """\
        Convert a list of Unique list to one hot vector.
    """
    unique_elements = sorted(set(unique_list))
    element_to_index = {element: index for index, element in enumerate(unique_elements)}
    
    one_hot_encodings = []
    for target_element in target_list:
        if target_element not in element_to_index:
            raise ValueError("Target element not found in unique list.")
        
        one_hot_vector = [0] * len(unique_elements)
        target_index = element_to_index[target_element]
        one_hot_vector[target_index] = 1
        one_hot_encodings.append(one_hot_vector)
    
    return np.array(one_hot_encodings)


def one_graph_splits(data, idx: int = 0):
    """\
        Return: bool, whether edge is intra subgraph
    """
    edge_index = data.edge_index
    undirected_dege_index = to_undirected(edge_index)
    try:
        adj = to_dense_adj(undirected_dege_index).cpu().numpy()[0]
    except:
        adj = np.zeros((data.num_nodes, data.num_nodes), dtype=int)
    
    subgraphs = louvain_method(adj)[0]
    node_group = torch.zeros(data.num_nodes, dtype=torch.long)

    for i in range(len(subgraphs)):
        for node in subgraphs[i]:
            node_group[node] = i
            
    intra_edge = torch.tensor([node_group[edge_index[0][i]]==node_group[edge_index[1][i]] for i in range(data.num_edges)], dtype=torch.bool)
    
    return intra_edge, node_group

def one_graph_splits_nx_save(args):
    graph, idx, dataset_name, save_path = args 
    edge_mask, node_group, idx = one_graph_splits_nx(graph, idx)

    with open('{}/one_graph_mask/{}_{}.mat'.format(save_path, dataset_name, idx), 'wb') as edge_mask_file:
        torch.save(edge_mask, edge_mask_file)
    with open('{}/one_graph_split/{}_{}.mat'.format(save_path, dataset_name, idx), 'wb') as node_group_file:
        torch.save(node_group, node_group_file)
    
    
def one_graph_splits_nx(graph,  idx: int = 0, seed: int = 42):
    '''
    output: bool, whether edge is intra subgraph
    may not use in focus graph
    '''
    
    edge_index = graph.edge_index
    x = graph.x
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    try:
        nx_partitions = nx.algorithms.community.louvain_communities(g, seed=seed)
    except:
        return one_graph_splits(graph, idx)
    
    node_group = torch.zeros(data.num_nodes, dtype=torch.long)

    for i in range(len(nx_partitions)):
        for node in nx_partitions[i]:
            node_group[node] = i
            
    intra_edge = torch.tensor([node_group[edge_index[0][i]]==node_group[edge_index[1][i]] for i in range(data.num_edges)], dtype=torch.bool)
    
    return intra_edge, node_group, idx


def one_graph_splits_nx_girvan_newman(graph,  seed: int = 42):
    '''
    output: bool, whther edge is intra subgraph
    may not use in focus graph
    '''
    
    edge_index = graph.edge_index
    x = graph.x
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    try:
        nx_partitions = nx.algorithms.community.girvan_newman(g)
        nx_partitions = list(set(sorted(c)) for c in next(nx_partitions))
    except:
        return one_graph_splits(graph)
    
    node_group = torch.zeros(data.num_nodes, dtype=torch.long)
    for i in range(len(nx_partitions)):
        for node in nx_partitions[i]:
            node_group[node] = i
    intra_edge = torch.tensor([node_group[edge_index[0][i]]==node_group[edge_index[1][i]] for i in range(data.num_edges)], dtype=torch.bool)
    
    return intra_edge, node_group


def one_graph_splits_feature(graph,  seed: int = 42):
    '''
    output: bool, whther edge is intra subgraph
    may not use in focus graph
    '''
    edge_index = graph.edge_index
    x = graph.x
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    try:
        partition = (set(torch.nonzero((x>0).view(-1)).view(-1).numpy()), set(torch.nonzero((x==0).view(-1)).view(-1).numpy()))
        nx_partitions = partition
    except:
        return one_graph_splits(graph)
    
    node_group = torch.zeros(data.num_nodes, dtype=torch.long)

    for i in range(len(nx_partitions)):
        for node in nx_partitions[i]:
            node_group[node] = i
    intra_edge = torch.tensor([node_group[edge_index[0][i]]==node_group[edge_index[1][i]] for i in range(data.num_edges)], dtype=torch.bool)
    return intra_edge, node_group

def multi_graph_split_nx(data_list):
    mask_list = []
    split_list = []
    for data in data_list:
        intra_edge, node_group = one_graph_splits_nx(data)
        mask_list.append(intra_edge)
        split_list.append(node_group)
    return mask_list, split_list

def multi_graph_split(data_list):
    result = []
    for data in data_list:
        intra_edge, _ = one_graph_splits(data)
        result.append(intra_edge)
    return result

def process_subbatch(subsplit, subsplit_cnt, ptr):
    '''
    process subgraph in batch
    output: 1*node_num: [0,0,1,1,0,2,3]
    '''
    subsplit, subsplit_cnt = copy.deepcopy(subsplit), subsplit_cnt
    cnt = 0
    # print(subsplit, subsplit_cnt, ptr)
    for i in range(0, len(ptr)-1):
        subsplit[ptr[i]: ptr[i+1]] += cnt
        cnt += subsplit_cnt[i]
    return subsplit


def batch2node_tensor(batch):
    num = max(batch)+1
    tmp = batch.repeat(num, 1)
    result = torch.arange(0,num).view(-1,1).repeat(1, batch.size(0)).to(batch.device)
    result = result == tmp
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

def subgraph2intra_edge_tensor2(subsplit, edge_index):
    num_rows = max(subsplit)+1 # num: number of subgraphs
    original_matrix = torch.cat([subsplit[edge_index[0]].unsqueeze(0), subsplit[edge_index[1]].unsqueeze(0)])
    num_cols = original_matrix.shape[1]
    # num_rows = torch.max(original_matrix) + 1  # Assuming zero-based indexing
    # Initialize a list to store the non-zero indices
    indices = []
    for col_idx in range(num_cols):
        row_indices = original_matrix[:, col_idx]
        if row_indices[0] != row_indices[1]: 
            continue
        indices.extend([(row, col_idx) for row in row_indices])
    indices = torch.LongTensor(indices).t()
    # inter_edge_counts = list(Counter(list(indices[0].cpu().numpy())).values())    
    values = torch.ones(indices.size(1), dtype=torch.float32)
    sparse_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([num_rows, num_cols]))
    # return torch.Tensor(inter_edge_counts)
    return sparse_matrix


def subgraph2intra_edge_tensor(subsplit, edge_index):
    num = max(subsplit)+1
    result = torch.arange(0,num).view(-1,1).repeat(1, edge_index.size(1)).to(subsplit.device)
    tmp1 = subsplit[edge_index[0]].repeat(num, 1)
    tmp2 = subsplit[edge_index[1]].repeat(num, 1)
    result = (tmp1==result) & (tmp2==result)
    return result

def subgraph2inter_intra_edge_tensor2(subsplit, edge_index):
    num = max(subsplit)+1 # num: number of subgraphs
    test = torch.cat([subsplit[edge_index[0]].unsqueeze(0), subsplit[edge_index[1]].unsqueeze(0)])
    return create_sub2inter_intra_sparse_tensor(test, num)

def create_sub2inter_intra_sparse_tensor(original_matrix, num_rows):
    num_cols = original_matrix.shape[1]
    # num_rows = torch.max(original_matrix) + 1  # Assuming zero-based indexing
    # Initialize a list to store the non-zero indices
    intra_indices = []
    inter_indices = []
    for col_idx in range(num_cols):
        row_indices = original_matrix[:, col_idx]
        if row_indices[0] != row_indices[1]:
            inter_indices.extend([(row, col_idx) for row in row_indices])
        intra_indices.extend([(row, col_idx) for row in row_indices])
    # Convert the list of indices to a Torch LongTensor
    # indices is 
    inter_indices = torch.LongTensor(inter_indices).t()
    intra_indices = torch.LongTensor(intra_indices).t()
    # inter_edge_counts = list(Counter(list(indices[0].cpu().numpy())).values())    
    inter_values = torch.ones(inter_indices.size(1), dtype=torch.float32)
    intra_values = torch.ones(intra_indices.size(1), dtype=torch.float32)
    
    inter_sparse_matrix = torch.sparse.FloatTensor(inter_indices, inter_values, torch.Size([num_rows, num_cols]))
    inter_sparse_matrix = torch.sparse.FloatTensor(intra_indices, intra_values, torch.Size([num_rows, num_cols]))
    # return torch.Tensor(inter_edge_counts)
    return inter_sparse_matrix, inter_sparse_matrix

def edge_to_intra_inter(edge_index, subsplit):
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    mask = subsplit[edge_index[0]]==subsplit[edge_index[1]]
        
    keep_edge_idx = torch.nonzero(mask, as_tuple=False).view(-1,) 
    intra_edge_index = torch.index_select(edge_index, 1, keep_edge_idx) 
    keep_edge_idx = torch.nonzero(~mask, as_tuple=False).view(-1,) 
    inter_edge_index = torch.index_select(edge_index, 1, keep_edge_idx) 
    
    return intra_edge_index, inter_edge_index

def remap_values(remapping, x):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)

def perm_sub_edge(edge_index, subsplit, batch, sub2node, sub_permutation_sample, replace_node_idx, keep_node_index):
    # node2sub, sub2sub, sub2node
    device = sub2node.device
    batch2node = batch2node_tensor(batch)
    batch2sub = torch.mm(batch2node.float(), sub2node.float().T) # ele==0 or ele>=1
    batch2sub_sample = batch2sub.bool() & sub_permutation_sample.bool()
    
    node2sub = torch.nonzero(sub2node.T)[:,1]
    # sub2sub
    ori_sub = torch.tensor([], device=device)
    sub_mapping = torch.tensor([], device=device)
    for i in range(batch2sub_sample.size(0)):
        sub_list = torch.nonzero(batch2sub[i]).view(-1,)
        sub_perm_list = torch.nonzero(batch2sub_sample[i]).view(-1,)
        if sub_perm_list.size(0) != 0:
            mapping = sub_perm_list[torch.randperm(sub_list.size(0)) % sub_perm_list.size(0)] # shuffle
        else:
            mapping = sub_list
        sub_mapping = torch.cat((sub_mapping, mapping))
        ori_sub = torch.cat((ori_sub, sub_list))
    _, idx = ori_sub.sort()
    sub2sub = sub_mapping[idx].type(torch.long)
    
    # node2node
    node_mapping = []
    nodes = sub2node[sub2sub[node2sub[replace_node_idx]]]
    for i in range(nodes.size(0)): # for each node in replace_node_idx, pick up one new node for mapping
        tmp = random.choice(nodes[i].nonzero().view(-1,))
        node_mapping.append(tmp)
    node_mapping = torch.tensor(node_mapping, device=device)
        
    node_index = torch.cat((replace_node_idx, keep_node_index))
    node_mapping = torch.cat((node_mapping, keep_node_index))
    node_index, idx = node_index.sort()
    node_mapping = node_mapping[idx]
    
    remapping = node_index, node_mapping
    result = remap_values(remapping, edge_index)
    
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
             
    
def dense_to_coo(a:torch.Tensor):
    idx = torch.nonzero(a).T
    data = a[idx[0],idx[1]]
    coo = torch.sparse_coo_tensor(idx, data, a.shape)
    return coo


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def dense2sparse(dense_matrix):
    nonzero_coords = dense_matrix.nonzero()
    values = dense_matrix[nonzero_coords[:, 0], nonzero_coords[:, 1]]
    sparse_tensor = torch.sparse_coo_tensor(nonzero_coords.t(), values, dense_matrix.size())
    return sparse_tensor

def element_wise_multiplication(sparse_matrix, dense_matrix):
    data = sparse_matrix.coalesce().indices() 
    index = torch.nonzero(dense_matrix[data.t()[:, 1]].flatten())[:, 0]
    new_matrix_indices = data.t()[index] 
    return torch.sparse_coo_tensor(new_matrix_indices.t(), \
        torch.ones(new_matrix_indices.shape[0]).to(sparse_matrix.device), sparse_matrix.size())

def save_file_with_unique_name(base_filename):
    file_extension = os.path.splitext(base_filename)[1]
    filename = base_filename
    index = 1
    while os.path.exists(filename):
        filename = f"{base_filename[:-len(file_extension)]}_{index}{file_extension}"
        index += 1
    return filename


def subgraph_shuffle(data, subsplit, sub_shuffle_sample, is_sparse=True):
    edge_index = data.edge_index
    # keep edge index
    if is_sparse:
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
    

def node_drop_in_subgraph(data, subsplit, sub_node_sample, requires_grad, node_p, percent):
    x = data.x
    x = x.float()
    x.requires_grad = requires_grad
    
    sample = F.gumbel_softmax(node_p, hard=True)
    drop_sample = sample[:,0]
    drop_data = copy.deepcopy(data)
    drop_node_mask = torch.rand(drop_sample.size(0), device=drop_sample.device) <= percent 
    
    # nodes(drop_sub_node_sample) in selected subgraph(sub_node_sample)
    sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
    selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
    
    mask = drop_sample * drop_node_mask.float() * selected_nodes
        
    keep_mask = 1 - mask
    x = x * keep_mask.view(-1, 1)
    
    # subgraph
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1,)
    _, _, edge_mask = subgraph(keep_idx, data.edge_index, num_nodes=data.num_nodes, return_edge_mask=True)
    edge_index_tmp = data.edge_index[:, ~edge_mask]
    node_adj = torch.sparse_coo_tensor(edge_index_tmp, torch.ones(edge_index_tmp.size(1)).to(edge_index_tmp.device), (data.num_nodes, data.num_nodes), requires_grad=True) # drop edge
    
    return x, drop_sample, -node_adj

def node_mask_in_subgraph(data, subsplit, sub_node_sample, requires_grad, node_p, percent):
    x = data.x
    x = x.float()
    x.requires_grad = requires_grad
    
    sample = F.gumbel_softmax(node_p, hard=True)
    drop_sample = sample[:,0]
    drop_node_mask = torch.rand(drop_sample.size(0), device=drop_sample.device) <= percent 
    
    # nodes(drop_sub_node_sample) in selected subgraph(sub_node_sample)
    sub2node = subgraph2node_tensor(subsplit).to(data.edge_index.device)
    selected_nodes = torch.sum(sub2node*sub_node_sample.view(-1,1), dim=0)
    # choose ratio
    mask = drop_sample * drop_node_mask.float() * selected_nodes
    keep_mask = 1 - mask
    
    return keep_mask.clone().detach().to(dtype=torch.long), drop_sample#, torch.sparse_coo_tensor((data.num_nodes, data.num_nodes)).to(x.device)    

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