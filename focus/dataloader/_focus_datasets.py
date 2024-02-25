import json
from math import ceil, floor
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch_geometric.data import Data
from ..data.preprocess._knn_radius_graph import KNN_Radius_Graph
from .utils import unique_list_mapping_to_one_hot, read_gene_list

"""
    1. Load the data from the disk
    2. Create a torch_geometric.data.Data object
    3. Create a torch_geometric.data.DataLoader object
    4. Return the torch_geometric.loader.DataLoader object
    
"""

class FocusDataset(Dataset):
    """Focus dataset generate from csv file"""
    def __init__(self, 
                 subcellular_pd: pd.DataFrame,
                 gene_list: List[str], 
                 knn_graph_radius: float,
                 gene_tx_threshold: int, 
                 celltype_threshold: float,
                 cell_ID_key: str = 'cell_ID',
                 cell_type_key: str = 'celltype',
                 gene_key: str = 'gene',
                 transcript_key: str = 'subcellular_domains',
                 embedding_type: str = 'one_hot',
                 subcellular_mapping: Union[str, Dict[str, int]]= {"nucleus": 0, "nucleus edge": 1, \
                     "cytoplasm": 2, "cell edge": 3, "none": 4},
                 celltype_mapping: Union[str, Dict[str, int]] = None,
                 ):
        self.subcellular_pd = subcellular_pd
        self.knn_graph_radius = knn_graph_radius
        self.gene_tx_threshold = gene_tx_threshold
        self.celltype_threshold = celltype_threshold
        self.gene_list = gene_list
        
        self.cell_ID_key = cell_ID_key
        self.cell_type_key = cell_type_key
        self.gene_key = gene_key
        self.transcript_key = transcript_key
        
        self.embedding_type = embedding_type # may not be used in the future
        self.subcellular_mapping = subcellular_mapping
        if type(celltype_mapping) == str:
            # celltype_mapping is a path to a json file
            self.celltype_mapping = json.load(open(celltype_mapping))
        elif type(celltype_mapping) == dict:
            self.celltype_mapping = celltype_mapping
        else:
            raise ValueError("celltype_mapping must be a path to a json file or a dictionary.")
        
        self.subcellular_pd_filtered = self.subcellular_filter()
        self.cell_id_list = list(self.subcellular_pd_filtered[self.cell_ID_key].unique())
    
    
    def subcellular_filter(self):
        """
            Filter the subcellular pd.Data based on the cell type threshold
        """
        # subcellular_pd = pd.read_csv(self.subcellular_csv_path, index_col=0)
        gene_per_cell_value_counts = self.subcellular_pd[self.cell_ID_key].value_counts()
        subcellular_pd_filtered = self.subcellular_pd[self.subcellular_pd[self.cell_ID_key].\
            isin(gene_per_cell_value_counts[gene_per_cell_value_counts > self.gene_tx_threshold])]
        subcellular_pd_filtered = subcellular_pd_filtered[subcellular_pd_filtered\
            [self.gene_key].isin(self.gene_list)]
        cell_type_value_counts = subcellular_pd_filtered[self.cell_type_key].value_counts()
        cell_type_proportions = cell_type_value_counts / len(subcellular_pd_filtered)
        subcellular_pd_filtered = subcellular_pd_filtered[subcellular_pd_filtered\
            [self.cell_type_key].isin(cell_type_value_counts[cell_type_proportions >= self.celltype_threshold].index)]
        return subcellular_pd_filtered
        
            
    def __len__(self):
        return len(self.subcellular_pd_filtered[self.cell_ID_key].unique())
    
    def __getitem__(self, idx) -> Dict[str, Union[str, torch.Tensor, np.array]]:
        # cell_id
        cell_idx = self.cell_id_list[idx]
        cell_graph = KNN_Radius_Graph(
            radius=self.knn_graph_radius,
            dataset=self.subcellular_pd_filtered,
            is_3D=True,
            cell_ID=cell_idx,
            transcript_label=self.transcript_key,
        )
        # edge_index
        edge_index = torch.Tensor(cell_graph.edge_index, dtype=torch.long)
        # node_type
        node_type = cell_graph.node_type
        if self.embedding_key == 'one_hot':
            node_type_str = [str(i) for i in node_type]
            node_embedding = unique_list_mapping_to_one_hot(self.gene_list, node_type_str)
            node_embedding = torch.Tensor(node_embedding, dtype=torch.float)
        else:
            raise ValueError("Embedding key not supported yet.")
        # node_spatial
        node_spatial = torch.Tensor(cell_graph.node_spatial)
        # node_label
        node_label = cell_graph.node_label
        for node_idx in range(len(node_label)):
            if isinstance(node_label[node_idx], float) and np.isnan(node_label[node_idx]):
                node_label[node_idx] = "none"
        node_label = [self.subcellular_mapping[label] for label in node_label]
        node_label = torch.Tensor(node_label, dtype=torch.long)
        # num_nodes
        num_nodes = len(node_type)
        # graph_label: cell type, etc.
        graph_label = cell_graph.graph_label
        graph_label = torch.Tensor(self.celltype_mapping[graph_label], dtype=torch.long)
        return Data(
            cell_name=cell_idx,
            edge_index=edge_index,
            x = node_embedding,
            num_nodes = num_nodes,
            node_spatial=node_spatial,
            graph_label=graph_label,
            node_label=node_label,
        )
        
        