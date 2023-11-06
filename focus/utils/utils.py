"""
utils.py for focus
"""
import os
import pandas as pd 
import numpy as np
from communities.algorithms import louvain_method
import networkx as nx
import torch
import torch_geometric
from torch_geometric.utils import to_undirected, to_dense_adj, remove_self_loops
import random
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


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


def find_subcellular_domains(cell_data: pd.DataFrame,
                             transcript_data: pd.DataFrame) -> pd.DataFrame:
    """\
    Find the subcellular domains of a cell.
    
    Args:
        cell_data: pd.DataFrame
            columns: "cell_boundaries", "nucleus_boundaries"
        transcript_data: pd.DataFrame
            columns: "x", "y", "gene", 
        
    Returns:
        subcellular_domains: the subcellular domains of a cell
    """
    pass

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



def one_graph_splits_feature(graph,  seed: int = 42):
    '''
    output: bool, whther edge is intra subgraph
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

def multi_graph_split_nx(data_list: List) -> List:
    mask_list = []
    split_list = []
    for data in data_list:
        intra_edge, node_group = one_graph_splits_nx(data)
        mask_list.append(intra_edge)
        split_list.append(node_group)
    return mask_list, split_list

def multi_graph_split(data_list: List) -> List:
    result = []
    for data in data_list:
        intra_edge, _ = one_graph_splits(data)
        result.append(intra_edge)
    return result






    





             
    


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
