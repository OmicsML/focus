import os
from abc import abstractmethod
from math import ceil, floor
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from anndata import AnnData
from torch_geometric.loader import DataLoader 
from tqdm import tqdm

from focus.dataloader.utils import read_gene_list

from focus.dataloader._focus_datasets import FocusDataset
from focus._constants import CELL_ID_KEY, CELL_TYPE_KEY, GENE_KEY, TRANSCRIPT_KEY

class FocusDataLoader(pl.LightningDataModule):
    """\
        Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    Args:
        reference_data_kwargs (Dict[str, Union[str, float, int, bool, None]]): reference dataset kwargs
        query_data_kwargs (Dict[str, Union[str, float, int, bool, None]]): query dataset kwargs
        train_size (float, optional): The proportion of training set. Defaults to 0.8.
        batch_size (int, optional): The batch size. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        use_cuda (bool, optional): Whether to use cuda. Defaults to False.
        num_workers (int, optional): The number of workers. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory. Defaults to True.
        
    """
    def __init__(self,
                 # reference dataset kwargs
                 reference_data_kwargs: Dict[str, Union[str, float, int, bool, None]],
                 query_data_kwargs: Dict[str, Union[str, float, int, bool, None]],
                 train_size: float = 0.8,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 use_cuda: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 ):
        super().__init__()
        self.reference_kwargs = reference_data_kwargs
        self.query_kwargs = query_data_kwargs
        self.train_size = train_size
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.data_collator = None # DataCollator in _datacollator.py
        
        self.reference_data_pd = pd.read_csv(self.reference_kwargs['reference_data_path'], index_col=0)
        if self.train_size >= 0 and self.train_size <= 1:
            cell_id_list = list(self.reference_data_pd[CELL_ID_KEY].unique())
            train_size = floor(len(cell_id_list) * self.train_size)
            train_cell_id_list = np.random.choice(cell_id_list, train_size, replace=False)
            val_cell_id_list = list(set(cell_id_list) - set(train_cell_id_list))
            self.reference_train_pd = self.reference_data_pd[self.reference_data_pd[CELL_ID_KEY].isin(train_cell_id_list)]
            self.reference_val_pd = self.reference_data_pd[self.reference_data_pd[CELL_ID_KEY].isin(val_cell_id_list)]
        else:
            raise ValueError("train_size should be a float between 0 and 1")

        self.query_data_pd = pd.read_csv(self.query_kwargs['query_data_path'], index_col=0)
        
        self.gene_list = read_gene_list(self.query_kwargs["gene_list_txt_path"])
        
        self.setup()
    
    
    def setup(self, stage: str | None = None):
        """\
        Load the dataset and split indices in train/test/val sets and load the prepared dataset.
        """
        if stage == "fit" or stage is None:
            self.train_set = FocusDataset(
                subcellular_pd=self.reference_train_pd,
                gene_list=self.gene_list,
                knn_graph_radius=self.reference_kwargs["knn_graph_radius"],
                gene_tx_threshold=self.reference_kwargs["gene_tx_threshold"],
                celltype_threshold=self.reference_kwargs["celltype_threshold"],
                cell_ID_key=CELL_ID_KEY,
                cell_type_key=CELL_TYPE_KEY,
                gene_key=GENE_KEY,
                transcript_key=TRANSCRIPT_KEY,
                embedding_type= self.reference_kwargs["embedding_type"],
                subcellular_mapping=self.reference_kwargs["subcellular_mapping"],
                celltype_mapping=self.reference_kwargs["celltype_mapping"],
            )
            self.validation_set = FocusDataset(
                subcellular_pd=self.reference_val_pd,
                gene_list=self.gene_list,
                knn_graph_radius=self.reference_kwargs["knn_graph_radius"],
                gene_tx_threshold=self.reference_kwargs["gene_tx_threshold"],
                celltype_threshold=self.reference_kwargs["celltype_threshold"],
                cell_ID_key=CELL_ID_KEY,
                cell_type_key=CELL_TYPE_KEY,
                gene_key=GENE_KEY,
                transcript_key=TRANSCRIPT_KEY,
                embedding_key=self.reference_kwargs["embedding_type"],
                subcellular_mapping=self.reference_kwargs["subcellular_mapping"],
                celltype_mapping=self.reference_kwargs["celltype_mapping"],
            )
        if stage == "test" or stage is None:
            self.test_set = FocusDataset(
                subcellular_pd=self.query_data_pd,
                gene_list=self.gene_list,
                knn_graph_radius=self.query_kwargs["knn_graph_radius"],
                gene_tx_threshold=self.query_kwargs["gene_tx_threshold"],
                celltype_threshold=self.query_kwargs["celltype_threshold"],
                cell_ID_key=CELL_ID_KEY,
                cell_type_key=CELL_TYPE_KEY,
                gene_key=GENE_KEY,
                transcript_key=TRANSCRIPT_KEY,
                embedding_key=self.query_kwargs["embedding_type"],
                subcellular_mapping=self.query_kwargs["subcellular_mapping"],
                celltype_mapping=self.query_kwargs["celltype_mapping"],
            )
        
    
    def train_dataloader(self):
        """Create train data loader."""
        return DataLoader(self.train_set,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=self.data_collator)
        
    
    def val_dataloader(self):
        """Create validation data loader."""
        return DataLoader(self.validation_set,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=self.data_collator)
    
    def test_dataloader(self):
        """Create test data loader."""
        return DataLoader(self.test_set,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=self.data_collator)
    
    