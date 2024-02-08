# /*
#  * This code is based on or includes code from the scvi-tools repository by scverse.
#  * Repository: https://github.com/scverse/scvi-tools/
#  * Copyright (c) 2023, Adam Gayoso, Romain Lopez, Martin Kim, Pierre Boyeau, Nir Yosef
#  */

import inspect
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Type, Union

import dynamo as dyn
import matplotlib.pyplot as plt
import numpy as np
import rich
import torch
import umap
from anndata import AnnData
from mudata import MuData
from sklearn.decomposition import PCA
from tqdm import tqdm

from ... import main_info, settings
from ...preprocess import GeneVocab

_UNTRAINED_WARNING_MESSAGE = "Trying to query inferred values from an untrained model. Please train the model first."


class BaseModelClass(ABC):
    """Abstract class for focus models."""

    def __init__(self, 
                 adata: Optional[AnnOrMuData] = None, 
                 vocab: Optional[GeneVocab] = None):
        self._adata = adata
        if vocab is not None:
            self._vocab = vocab

        self.is_trained_ = False
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self.module_list = []


    @property
    def vocab(self) -> GeneVocab:
        """Vocab attached to model instance."""
        return self._vocab

    @vocab.setter
    def vocab(self, vocab: GeneVocab):
        if vocab is None:
            raise ValueError("vocab cannot be None.")
        self._vocab = vocab

    def to_device(self, device: Union[str, int]):
        """\
        Move model to device.

        Args:
            device: Device to move model to. Options: 'cpu' for CPU, integer GPU index (eg. 0),
                or 'cuda:X' where X is the GPU index (eg. 'cuda:0'). See torch.device for more info.

        Examples
        --------
        >>> counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
        >>> adata = ad.AnnData(counts)
        >>> model = pillar.model.OldGeneFormerBert(adata)
        >>> model.to_device('cpu')      # moves model to CPU
        >>> model.to_device('cuda:0')   # moves model to GPU 0
        >>> model.to_device(0)          # also moves model to GPU 0
        """
        my_device = torch.device(device)
        for module in self.module_list:
            module.to(my_device)

    @property
    def device(self) -> str:
        """The current device that the module's params are on."""
        return self.module_list[0].device

    def _make_data_loader(
        self,
        adata: AnnOrMuData,
        indices: Sequence[int] | None = None,
        batch_size: int = 1,
        prep_batch_size: int = 128,
        shuffle: bool = False,
        keep_value: bool = True,
        rank_gene: bool = False,
        split: bool = False,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        inference: bool = False,
        **dataprep_kwargs,
    ):
        """\
        Create a AnnDataLoader object for data iteration.
        copy from pillar, but may not use this function, 

        Args:
            adata: AnnData object with equivalent structure to initial AnnData.
            indices: Indices of cells in adata to use. If `None`, all cells are used.
            batch_size: Minibatch size for data loading into model. Defaults to `pillar.settings.batch_size`.
            shuffle: Whether observations are shuffled each iteration though the data loader.
            dataprep_kwargs: Kwargs to the class-specific data loader class
        """

        if batch_size is None:
            batch_size = settings.batch_size

        if "num_workers" not in dataprep_kwargs:
            dataprep_kwargs.update({"num_workers": settings.dl_num_workers})

        if inference:
            return AnnLoader(
                adata,
                shuffle=shuffle,
                indices=indices,
                batch_size=batch_size,
                **dataprep_kwargs,
            )

        dl = self.data_loader_class(
            adata,
            shuffle=shuffle,
            keep_value=keep_value,
            rank_gene=rank_gene,
            indices=indices,
            batch_size=batch_size,
            prep_batch_size=prep_batch_size,
            train_size=train_size,
            validation_size=validation_size,
            vocab=self.vocab,
            **dataprep_kwargs,
        )

        return dl

    def _check_if_trained(self, warn: bool = True, message: str = _UNTRAINED_WARNING_MESSAGE):
        """Check if the model is trained.

        If not trained and `warn` is True, raise a warning, else raise a RuntimeError.
        """
        if not self.is_trained_:
            if warn:
                warnings.warn(message, UserWarning, stacklevel=settings.warnings_stacklevel)
            else:
                raise RuntimeError(message)

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self.is_trained_

    @property
    def test_indices(self) -> np.ndarray:
        """Observations that are in test set."""
        return self.test_indices_

    @property
    def train_indices(self) -> np.ndarray:
        """Observations that are in train set."""
        return self.train_indices_

    @property
    def validation_indices(self) -> np.ndarray:
        """Observations that are in validation set."""
        return self.validation_indices_

    @property
    def n_test(self) -> int:
        """Number of observations in test set."""
        return len(self.test_indices)

    @property
    def n_train(self) -> int:
        """Number of observations in train set."""
        return len(self.train_indices)

    @property
    def n_validation(self) -> int:
        """Number of observations in validation set."""
        return len(self.validation_indices)

    @train_indices.setter
    def train_indices(self, value):
        self.train_indices_ = value

    @test_indices.setter
    def test_indices(self, value):
        self.test_indices_ = value

    @validation_indices.setter
    def validation_indices(self, value):
        self.validation_indices_ = value

    @is_trained.setter
    def is_trained(self, value):
        self.is_trained_ = value

    def _get_user_attributes(self):
        """Returns all the self attributes defined in a model class, e.g., `self.is_trained_`."""
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        return attributes

    @abstractmethod
    def train(self):
        """Trains the model."""
