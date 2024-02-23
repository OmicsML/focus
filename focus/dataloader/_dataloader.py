import os
from abc import abstractmethod
from math import ceil, floor
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from anndata import AnnData
from torch_geometric.loader import DataLoader 
from tqdm import tqdm

from ._focus_datasets import FocusDataset


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
        
        train_dataset = FocusDataset(reference_data_path, **kwargs)
        test_dataset = FocusDataset(query_data_path, **kwargs)
        
        
        
    def prepare_data(self):
        """\
        Prepare (tokenize) dataset and save it to the disk.
        This function is called on 1 GPU or CPU.
        """
        pass
    
    def setup(self, stage: str | None = None):
        """\
        Load the dataset and split indices in train/test/val sets and load the prepared dataset.
        """
        pass
    
    def train_dataloader(self):
        """Create train data loader."""
        return DataLoader(self.train_set,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            sampler=self.sampler,
                            collate_fn=self.data_collator)
        
    
    def val_dataloader(self):
        """Create validation data loader."""
        return DataLoader(self.validation_set,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            sampler=self.sampler,
                            collate_fn=self.data_collator)
    
    def test_dataloader(self):
        """Create test data loader."""
        return DataLoader(self.test_set,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=self.data_collator)
    
    