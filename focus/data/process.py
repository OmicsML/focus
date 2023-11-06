"""transform npy file into torch_geometric.data.Data"""

import glob
import json
import os
import numpy as np 
import torch
import threading
from tqdm import tqdm
import pandas as pd 
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import multiprocessing as mp 
from multiprocessing import Process
from scipy.spatial import distance_matrix 
from scipy.spatial import KDTree 
import networkx as nx 
from concurrent.futures import ProcessPoolExecutor, as_completed
from ..utils.utils import unique_list_mapping_to_one_hot, one_graph_splits_nx
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


class KNN_Radius_Graph(object):
    def __init__(self, 
                 radius: float, 
                 dataset: pd.DataFrame, 
                 is_3D: bool = False,
                 cell_ID: str = 'cell_ID',
                 gene_column: str = 'gene',
                 transcript_label: str = 'subcellular_domains',
                 ) -> None:
        """\
            KNN radius Graph to find the nearest neighbors of each node in the graph.
        
        Args: 
            radius: the radius of spatial transcript neighbor threshold
            dataset_name: the name of the dataset
            is_3D: check the dimension of transcript coordinates
            cell_ID: cell ID, "31-0"
            gene_column: "gene" : column name of transcript dataset
            transcript_label: the label of the transcripts : subcellular domains
            
        """    
        self.radius = radius 
        self.dataset = dataset
        self.is_3D = is_3D
        self.cell_ID = cell_ID
        self.gene_column = gene_column
        self.transcript_label = transcript_label
        self.gene_list = self.gene_list()
        self.selected_cell_data = self._data_process()
        
    def _data_process(self) -> pd.DataFrame:
        """
            Find the data of a target cell
        Args: 
            dataset: dataset
            cell_ID: the ID of the target cell
            
        Returns:
            pd.DataFrame: the data of the target cell
            
        """
        return self.dataset[self.dataset['cell_ID'] == self.cell_ID]
    
    def gene_list(self) -> List[str]:
        """
           Find the gene list of the dataset.
        
        Args: 
            dataset: dataset
        Returns:
            gene_list: the list of genes in the dataset
        """
        return sorted(self.dataset['gene'].unique().tolist())
        
    def edge_index(self) -> np.array:
        """
           Find the edge index of the graph of a selected cell.
        
        Args:
            dataset: dataset
            cell_ID: the ID of the target cell
            radius: the radius of spatial transcript neighbor threshold  
        Returns:
            edge_index: the edge index of the graph of a selected cell
        """
        
        selected_data = self.selected_cell_data
        if self.is_3D is True:
            x = np.array(selected_data['x'].to_list())
            y = np.array(selected_data['y'].to_list())
            z = np.array(selected_data['z'].to_list())
            r_c = np.column_stack((x, y, z))
        else:
            x = np.array(selected_data['x'].to_list())
            y = np.array(selected_data['y'].to_list())
            r_c = np.column_stack((x, y))
        kdtree = KDTree(r_c)
        G = nx.Graph()
        for i, x in enumerate(r_c):
            idx = kdtree.query_ball_point(x, self.radius)
            for j in idx:
                if i < j:
                    G.add_edge(i, j)
        adj_matrix = nx.to_numpy_array(G)
        rows, cols = np.where(adj_matrix == 1)
        edge_index = np.array([rows, cols])
        return edge_index
    
    def node_label(self):
        """
            Node label: transcript label: subcellular domains
        """
        return np.array(self.selected_cell_data[self.transcript_label])
    
    def node_type(self):
        """
            Node type: transcript type: Gene name
        """
        return np.array(self.selected_cell_data[self.gene_column])
    
    def node_spatial(self):
        if self.is_3D is True:
            return np.array(self.selected_cell_data[['x', 'y', 'z']]).T
        else:
            return np.array(self.selected_cell_data[['x', 'y']]).T
    
    def graph_label(self):
        return self.selected_cell_data[self.selected_cell_data['cell_ID'] == self.cell_ID]['cell_type'].unique()[0]


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

    # merfe all graphs into one file    
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