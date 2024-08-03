"""transform npy file into torch_geometric.data.Data"""

import glob
import json
import os
import numpy as np 
import torch
import threading
from tqdm import tqdm
from typing import Callable, List, Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from ..model.utils import unique_list_mapping_to_one_hot, one_graph_splits_nx
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

class MyThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
    
    def run(self):
        self.result = self.func(*self.args)
        
    def get_result(self):
        try:
            return self.result
        except Exception as e:
            return None


class NPY2TorchG(InMemoryDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        """transform npy file into torch_geometric.data.Data"""
        self.name = name
        self.data_list = []
        super(NPY2TorchG, self).__init__(root, transform, pre_transform, pre_filter)
        
    def process(self):    
        print(os.path.join(self.root, self.name))
        if self.name.split("/")[-1].split(".")[0] == "u2os_merfish":
            data = np.load(os.path.join(self.root, self.name), allow_pickle=True)
            gene = []
            with open(os.path.join(self.root, "./raw/gene.txt"), 'r') as f:
                for line in f:
                    gene.append(line.strip())
            for graph_idx in tqdm(range(len(data))):
                graph = data[graph_idx][0]
                # graph name 
                name = graph[0]
                # transform into node features
                one_hot_encodings = unique_list_mapping_to_one_hot(gene, graph[2])
                x = torch.Tensor(one_hot_encodings, dtype=torch.float)
                # transform into edge index
                edge_index = torch.Tensor(graph[1], dtype=torch.long)
                # transform into y 
                mapping = {
                    "nucleus": 0,
                    "nucleus edge": 1,
                    "cytoplasm": 2,
                    "cell edge": 3,
                    "none": 4
                }
                for i in range(len(graph[3])):
                    if isinstance( graph[3][i], float) and np.isnan( graph[3][i]):
                        graph[3][i] = 'none'
                node_label = [mapping[item] for item in graph[3]]
                y = torch.Tensor(node_label, dtype=torch.long)
                num_nodes = len(graph[2])
                # node position
                pos = torch.Tensor(graph[4].T)
                graph_data = Data(x=x, edge_index=edge_index, y=y, num_nodes = num_nodes, pos=pos, name=name)
                self.data_list.append(graph_data) 
            self.data, self.slices = self.collate(self.data_list)
            self.save()

        elif self.name.split("/")[-1].split(".")[0] in ['Xenium_r1', "Xenium_r2"] \
            or self.name.split("/")[-1].split(".")[0].startswith("mouse"):
            data = np.load(os.path.join(self.root, self.name), allow_pickle=True)
            gene = []
            with open(os.path.join(self.root, "./raw/gene.txt"), 'r') as f:
                for line in f:
                    gene.append(line.strip())
            for graph_idx in tqdm(range(len(data))):
                graph = data[graph_idx][0]
                # graph name 
                name = graph[0]
                # transform into node features
                int2str = [str(i) for i in graph[2]]
                one_hot_encodings = unique_list_mapping_to_one_hot(gene, int2str)
                num_nodes = len(graph[2])
                x = torch.tensor(one_hot_encodings, dtype=torch.float)
                # transform into edge index
                edge_index = torch.tensor(graph[1], dtype=torch.long)
                # node position
                pos = torch.Tensor(graph[3].T)
                if self.name.split("/")[-1].split(".")[0] == 'Xenium_r1'\
                    or self.name.split("/")[-1].split(".")[0].split("_")[0] == "Xenium" \
                    or self.name.split("/")[-1].split(".")[0].startswith("mouse"):
                    y = graph[4]
                else: 
                    raise ValueError("Unknown value: {}".format(self.name))
                graph_data = Data(x=x, edge_index=edge_index,y=y, num_nodes = num_nodes, pos=pos, name=name)
                self.data_list.append(graph_data) 
            self.data, self.slices = self.collate(self.data_list)
            self.save()
        elif self.name.split("/")[-1].split(".")[0].split("_")[0] == 'CosMx' \
            or self.name.split("/")[-1].split(".")[0].startswith("Run"):
            data = np.load(os.path.join(self.root, self.name), allow_pickle=True)
            gene = []
            with open(os.path.join(self.root, "./raw/gene.txt"), 'r') as f:
                for line in f:
                    gene.append(line.strip())
            for graph_idx in tqdm(range(len(data))):
                graph = data[graph_idx][0]
                # graph name 
                name = graph[0]
                # transform into node features
                int2str = [str(i) for i in graph[2]]
                one_hot_encodings = unique_list_mapping_to_one_hot(gene, int2str)
                num_nodes = len(graph[2])
                x = torch.tensor(one_hot_encodings, dtype=torch.float)
                x = torch.argmax(x, dim=1)
                # transform into edge index
                edge_index = torch.tensor(graph[1], dtype=torch.long)
                # node position
                pos = torch.Tensor(graph[3].T)
                # graph label
                y = graph[4]
                # node label 
                mapping = {
                    "Nuclear": 0,
                    "Cytoplasm": 1,
                    "Membrane": 2,
                }
                node_label = [mapping[item] for item in graph[5]]
                node_label = torch.Tensor(node_label)
                graph_data = Data(x=x, edge_index=edge_index,y=y, num_nodes = num_nodes, pos=pos, name=name, node_label = node_label)
                self.data_list.append(graph_data) 
            self.data, self.slices = self.collate(self.data_list)
            self.save()
        else:
            raise ValueError("Unknown value: {}".format(self.name))
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return len(self.data_list[idx])
    
    def save(self):
        processed_path = os.path.join(self.root, "processed")
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        torch.save(self.data_list, os.path.join(processed_path, "data.pt"))

    def load(self):
        self.data_list = torch.load(os.path.join(self.root, "processed", "data.pt"))
        return self.data_list

    @property
    def raw_file_names(self):
        raw_npy = self.name.split("/")[-1].split(".")[0] + ".npy"
        return [raw_npy, 'gene.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'




def subgraph_splits(data_list: List, dataset_name: str,save_path: str, num_threads: int) -> None:
    """\
        Split the graph into subgraphs.
        
        Args:
            data_list: list of torch_geometric.data.Data
            save_path: save path of the subgraphs    
            num_threads: number of threads
    """        

    max_workers = num_threads
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        one_graph_mask_path = os.path.join(save_path, "one_graph_mask")
        one_graph_split_path = os.path.join(save_path, "one_graph_split")
        if not os.path.exists(one_graph_mask_path):
            os.makedirs(one_graph_mask_path)
        if not os.path.exists(one_graph_split_path):
            os.makedirs(one_graph_split_path)
        for single_graph in tqdm(data_list, desc="Subgraph Splits"):
            futures[executor.submit(one_graph_splits_nx, single_graph, data_list.index(single_graph))] = single_graph
        for f in tqdm(as_completed(futures), total=len(futures), desc="Subgraph Saving"):
            try:
                edge_mask, node_group, idx = f.result()
                torch.save(edge_mask, '{}/one_graph_mask/{}_{}.mat'.format(save_path, dataset_name, idx))
                torch.save(node_group, '{}/one_graph_split/{}_{}.mat'.format(save_path, dataset_name, idx))
            
            except Exception as e:
                print(e)
                raise ValueError("Unknown value: {}".format(e))


 
    
    
    # merge all graphs into one file    
    all_graph_mask_path = os.path.join(save_path, "all_graph_mask")
    all_graph_split_path = os.path.join(save_path, "all_graph_split")
    if not os.path.exists(all_graph_mask_path):
        os.makedirs(all_graph_mask_path)
    if not os.path.exists(all_graph_split_path):
        os.makedirs(all_graph_split_path)     
    all_graph_mask = []
    all_graph_split = []
    for i in tqdm(range(len(data_list)), desc="One Graph Saving"):
        # graph_mask = edge_mask_dict[name]
        # graph_split = node_group_dict[name]
        graph_mask = torch.load('{}/one_graph_mask/{}_{}.mat'.format(save_path, dataset_name, i))
        graph_split = torch.load('{}/one_graph_split/{}_{}.mat'.format(save_path, dataset_name, i))
        all_graph_mask.append(graph_mask)
        all_graph_split.append(graph_split)
    torch.save(all_graph_mask, '{}/all_graph_mask/{}.mat'.format(save_path, dataset_name))
    torch.save(all_graph_split, '{}/all_graph_split/{}.mat'.format(save_path, dataset_name))    
    

def return_aug_data(data_path, dataset_name, n_view):
    # for cosmx data
    dataset = NPY2TorchG(data_path, dataset_name)
    dataset = dataset.load()

    node_path = os.path.join(data_path, "draw")
    subgraph_path = os.path.join(data_path, "draw_subgraph")
    celltype_mapping = os.path.join(data_path, "processed/mapping.json")
    node_files = glob.glob(os.path.join(node_path, "*node_drop_{}.pt".format(n_view)))
    subgraph_files = glob.glob(os.path.join(subgraph_path, "*subgraph_{}.pt".format(n_view)))
    
    celltype_mapping = json.load(open(celltype_mapping))
    num2celltype = {value: key for key, value in celltype_mapping.items()}
    
    graph_dict = {}
    for file in node_files:
        temporal_data = torch.load(file)
        temporal_data.y = num2celltype[temporal_data.y.item()]
        if temporal_data.y not in graph_dict:
            graph_dict[temporal_data.y] = []
        graph_dict[temporal_data.y].append(temporal_data) 

    graph_dataset = []
    for label in graph_dict.keys():
        graph_list = graph_dict[label]
        temp_list = []
        for graph in graph_list:
            # if np.count_nonzero(graph.importance_score.cpu().detach().numpy()>0) / len(graph.importance_score) > 0.1:
                temp_list.append(graph)
        graph_dataset.extend(temp_list)
    for graph in graph_dataset:
        varname = graph.y + '_' + graph.name
        globals()[varname] = graph

    subgraph_dict = {}
    for file in tqdm(subgraph_files):
        temporal_data = torch.load(file)
        if num2celltype[temporal_data.y.item()] not in subgraph_dict:
            subgraph_dict[num2celltype[temporal_data.y.item()]] = []
        subgraph_dict[num2celltype[temporal_data.y.item()]].append(temporal_data)    
    subgraph_dataset = []
    for label in subgraph_dict.keys():
        subgraph_list = subgraph_dict[label]
        temp_list = []
        for graph in subgraph_list:
            # if np.count_nonzero(graph.subgraph_score.cpu().detach().numpy()>0) / len(graph.subgraph_score) > 0.1:
            temp_list.append(graph)
        subgraph_dataset.extend(temp_list)
    for graph in subgraph_dataset:
        varname = num2celltype[graph.y.item()] + '_' + graph.name + '_sub'
        globals()[varname] = graph
        
    subcellular_graph = []
    for subcellular_cell in tqdm(dataset):
        varname = subcellular_cell.y + '_' + subcellular_cell.name
        varname_sub = subcellular_cell.y + '_' + subcellular_cell.name + '_sub'
        if varname in globals() and varname_sub in globals():
            subcellular_cell.importance_score = globals()[varname].importance_score
            subcellular_cell.subgraph_score = globals()[varname_sub].subgraph_score
            subcellular_graph.append(subcellular_cell)
    subcelluar_data_dict = {}
    for data in tqdm(subcellular_graph):
        if data.y not in subcelluar_data_dict:
            subcelluar_data_dict[data.y] = []
        subcelluar_data_dict[data.y].append(data)
        
    return subcelluar_data_dict
