import numpy as np 
import copy 
import torch
from torch_geometric.utils import to_undirected, to_dense_adj, remove_self_loops
import random
import os 
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    # Convert the list of indices to a Torch LongTensor
    # indices is 
    indices = torch.LongTensor(indices).t()
    # inter_edge_counts = list(Counter(list(indices[0].cpu().numpy())).values())    
    values = torch.ones(indices.size(1), dtype=torch.float32)
    sparse_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([num_rows, num_cols]))
    # return torch.Tensor(inter_edge_counts)
    return sparse_matrix


def subgraph2inter_edge_tensor(subsplit, edge_index):
    num = max(subsplit)+1 # num: number of subgraphs
    result = torch.arange(0,num).view(-1,1).repeat(1, edge_index.size(1)).to(subsplit.device)
    tmp1 = subsplit[edge_index[0]].repeat(num, 1)
    tmp2 = subsplit[edge_index[1]].repeat(num, 1)
    result = (tmp1==result) ^ (tmp2==result)
    return result

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
    # Convert the list of indices to a Torch LongTensor
    indices = torch.LongTensor(indices).t()
    # inter_edge_counts = list(Counter(list(indices[0].cpu().numpy())).values())    
    values = torch.ones(indices.size(1), dtype=torch.float32)
    sparse_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([num_rows, num_cols]))
    # return torch.Tensor(inter_edge_counts)
    return sparse_matrix

def subgraph2node_tensor(subsplit):
    num = max(subsplit)+1
    tmp = subsplit.repeat(num, 1)
    result = torch.arange(0,num).view(-1,1).repeat(1, subsplit.size(0)).to(subsplit.device)
    result = result == tmp
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
    result = torch.tensor(result).view(2, -1).to(device)
    return result


def unique_list_mapping_to_one_hot(unique_list: List, target_list: List)-> np.array:
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

def element_wise_multiplication(sparse_matrix, dense_matrix):
    data = sparse_matrix.coalesce().indices() 
    index = torch.nonzero(dense_matrix[data.t()[:, 1]].flatten())[:, 0]
    new_matrix_indices = data.t()[index] 
    return torch.sparse_coo_tensor(new_matrix_indices.t(), \
        torch.ones(new_matrix_indices.shape[0]).to(sparse_matrix.device), sparse_matrix.size())


def process_subbatch(subsplit, subsplit_cnt, ptr):
    '''
    process subgraph in batch
    output: 1*node_num: [0,0,1,1,0,2,3]
    '''
    # will not use in focus module
    subsplit, subsplit_cnt = copy.deepcopy(subsplit), subsplit_cnt
    cnt = 0
    # print(subsplit, subsplit_cnt, ptr)
    for i in range(0, len(ptr)-1):
        subsplit[ptr[i]: ptr[i+1]] += cnt
        cnt += subsplit_cnt[i]
    return subsplit

def save_file_with_unique_name(base_filename):
    file_extension = os.path.splitext(base_filename)[1]
    filename = base_filename
    index = 1
    
    while os.path.exists(filename):
        filename = f"{base_filename[:-len(file_extension)]}_{index}{file_extension}"
        index += 1
    return filename