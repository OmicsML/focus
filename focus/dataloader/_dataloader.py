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

from ._focus_datasets import FocusDataset

CELL_ID_KEY = "cell_ID"
CELL_TYPE_KEY = "celltype"
GENE_KEY = "gene"
TRANSCRIPT_KEY = "subcellular_domains"
EMBEDDING_KEY = "one_hot"
SUBCELLULAR_MAPPING = {"nucleus": 0, "nucleus edge": 1, "cytoplasm": 2, "cell edge": 3, "none": 4}
CELLTYPE_MAPPING = None

class FocusDataLoader(pl.LightningDataModule):
    """\
        Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    Args:
        pl (_type_): _description_
    """
    def __init__(self,
                #  adata: Union[Sequence[AnnData], Dict[str, AnnData]] | None = None,
                 reference_data_path: Optional[str] = None,
                 query_data_path: Optional[str] = None,
                 train_size: float = 0.8,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 use_cuda: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 sampler: Optional[torch.utils.data.Sampler] = None,
                 **kwargs,
                 ):
        super().__init__()
        self.reference_data_path = reference_data_path
        self.query_data_path = query_data_path
        self.train_size = train_size
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sampler = sampler
        self.data_collator = None
        self.kwargs = kwargs
        
        self.data_collator = None # DataCollator in _datacollator.py
        
        self.reference_data_pd = pd.read_csv(self.reference_data_path, index_col=0)
        if self.train_size >= 0 and self.train_size <= 1:
            cell_id_list = list(self.reference_data_pd[CELL_ID_KEY].unique())
            train_size = floor(len(cell_id_list) * self.train_size)
            train_cell_id_list = np.random.choice(cell_id_list, train_size, replace=False)
            val_cell_id_list = list(set(cell_id_list) - set(train_cell_id_list))
            self.reference_train_pd = self.reference_data_pd[self.reference_data_pd[CELL_ID_KEY].isin(train_cell_id_list)]
            self.reference_val_pd = self.reference_data_pd[self.reference_data_pd[CELL_ID_KEY].isin(val_cell_id_list)]
        else:
            raise ValueError("train_size should be a float between 0 and 1")

        self.query_data_pd = pd.read_csv(self.query_data_path, index_col=0)
        self.setup()
    
    
    def setup(self, stage: str | None = None):
        """\
        Load the dataset and split indices in train/test/val sets and load the prepared dataset.
        """
        if stage == "fit" or stage is None:
            self.train_set = FocusDataset(
                subcellular_pd=self.reference_train_pd,
                gene_list_txt_path=self.kwargs["gene_list_txt_path"],
                knn_graph_radius=self.kwargs["knn_graph_radius"],
                gene_tx_threshold=self.kwargs["gene_tx_threshold"],
                celltype_threshold=self.kwargs["celltype_threshold"],
                cell_ID_key=CELL_ID_KEY,
                cell_type_key=CELL_TYPE_KEY,
                gene_key=GENE_KEY,
                transcript_key=TRANSCRIPT_KEY,
                embedding_key=EMBEDDING_KEY,
                subcellular_mapping=SUBCELLULAR_MAPPING,
                celltype_mapping=CELLTYPE_MAPPING,
            )
            self.validation_set = FocusDataset(
                subcellular_pd=self.reference_val_pd,
                gene_list_txt_path=self.kwargs["gene_list_txt_path"],
                knn_graph_radius=self.kwargs["knn_graph_radius"],
                gene_tx_threshold=self.kwargs["gene_tx_threshold"],
                celltype_threshold=self.kwargs["celltype_threshold"],
                cell_ID_key=CELL_ID_KEY,
                cell_type_key=CELL_TYPE_KEY,
                gene_key=GENE_KEY,
                transcript_key=TRANSCRIPT_KEY,
                embedding_key=EMBEDDING_KEY,
                subcellular_mapping=SUBCELLULAR_MAPPING,
                celltype_mapping=CELLTYPE_MAPPING,
            )
        if stage == "test" or stage is None:
            self.test_set = FocusDataset(
                subcellular_pd=self.query_data_pd,
                gene_list_txt_path=self.kwargs["gene_list_txt_path"],
                knn_graph_radius=self.kwargs["knn_graph_radius"],
                gene_tx_threshold=self.kwargs["gene_tx_threshold"],
                celltype_threshold=self.kwargs["celltype_threshold"],
                cell_ID_key=CELL_ID_KEY,
                cell_type_key=CELL_TYPE_KEY,
                gene_key=GENE_KEY,
                transcript_key=TRANSCRIPT_KEY,
                embedding_key=EMBEDDING_KEY,
                subcellular_mapping=SUBCELLULAR_MAPPING,
                celltype_mapping=CELLTYPE_MAPPING,
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
    
    