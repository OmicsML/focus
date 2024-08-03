import pandas as pd 
import numpy as np 
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import multiprocessing as mp 
from multiprocessing import Process
from scipy.spatial import distance_matrix 
from scipy.spatial import KDTree 
import networkx as nx 

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
        """\
        Find the data of a target cell
        
        Args: 
            dataset: dataset
            cell_ID: the ID of the target cell
            
        Returns:
            pd.DataFrame: the data of the target cell
            
        """
        return self.dataset[self.dataset['cell'] == self.cell_ID]
    
    def gene_list(self) -> List[str]:
        """\
        Find the gene list of the dataset.
        
        Args: 
            dataset: dataset
            
        Returns:
            gene_list: the list of genes in the dataset
        """
        return sorted(self.dataset['gene'].unique().tolist())
        
    def edge_index(self) -> np.array:
        """\
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
        """\
            Node label: transcript label: subcellular domains
        """
        return np.array(self.selected_cell_data[self.transcript_label])
    
    def node_type(self):
        """\
            Node type: transcript type: Gene name
        """
        return np.array(self.selected_cell_data[self.gene_column])
    