import os
from abc import abstractmethod
from math import ceil, floor
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from anndata import AnnData
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class FocusDataLoader(pl.LightningDataModule):
    """\
        Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    Args:
        pl (_type_): _description_
    """
    def __init__(self,
                #  adata: Union[Sequence[AnnData], Dict[str, AnnData]] | None = None,
                 vocab: Optional[GeneVocab] = None,
                 train_size: float = 0.9,
                 validation_size: Optional[float] = None,
                 shuffle_set_split: bool = True,
                 seed: Optional[int] = 369,
                 batch_size: int = 1,
                 prep_batch_size: int = 128,
                 shuffle: bool = False,
                 use_default_converter: bool = True,
                 use_cuda: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 sampler: Optional[torch.utils.data.Sampler] = None,
                 filter_out_zero: bool = True,
                 dataset_path: Optional[str] = "prepared_dataset.dataset",
                 replace: bool = False,
                 use_cell: bool = False,
                 cell_seq_path: Optional[str] = None,
                 dataset_path_val: Optional[str] = None,
                 cell_seq_path_val: Optional[str] = None,
                 adata_val: Optional[AnnData] = None,
                 **kwargs,
                 ):
        super().__init__()
        self.vocab = vocab
        self.train_size = train_size
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.seed = seed
        self.batch_size = batch_size
        self.prep_batch_size = prep_batch_size
        self.shuffle = shuffle
        self.use_default_converter = use_default_converter
        self.use_cuda = use_cuda
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sampler = sampler
        self.filter_out_zero = filter_out_zero
        self.dataset_path = dataset_path
        self.replace = replace
        self.use_cell = use_cell
        self.cell_seq_path = cell_seq_path
        self.dataset_path_val = dataset_path_val
        self.cell_seq_path_val = cell_seq_path_val
        self.adata_val = adata_val
        self.kwargs = kwargs
        
        self.data_collator = None # DataCollator in _datacollator.py
        
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
        pass 
    
    def val_dataloader(self):
        """Create validation data loader."""
        pass
    
    def test_dataloader(self):
        """Create test data loader."""
        pass
    
    