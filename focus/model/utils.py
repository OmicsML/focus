
import os 
import pandas as pd 
import numpy as np
import copy
import torch 
import torch_geometric


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

def edge_to_intra_inter(edge_index, subsplit):
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    mask = subsplit[edge_index[0]]==subsplit[edge_index[1]]
        
    keep_edge_idx = torch.nonzero(mask, as_tuple=False).view(-1,) 
    intra_edge_index = torch.index_select(edge_index, 1, keep_edge_idx) 
    keep_edge_idx = torch.nonzero(~mask, as_tuple=False).view(-1,) 
    inter_edge_index = torch.index_select(edge_index, 1, keep_edge_idx) 
    
    return intra_edge_index, inter_edge_index