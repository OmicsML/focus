from functools import partial
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import os.path as osp
import torch 



def load_dataset_from_disk(
    path: str,
    name: str,
    transform: Optional[Callable] = None,
    pre_transform: Optional[Callable] = None,
    pre_filter: Optional[Callable] = None,
    **kwargs: Any,
    ):
    """Loads a dataset from disk.

    Args:
        path (str): Path to the directory containing the dataset.
        name (str): Name of the dataset.
        transform (Optional[Callable], optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (Optional[Callable], optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (Optional[Callable], optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean value,
            indicating whether the data object should be included in the final dataset.
            (default: :obj:`None`)
        verbose (bool, optional): If set to :obj:`True`, will print out the size of the
            dataset. (default: :obj:`False`)
        **kwargs (optional): Additional arguments for loading the dataset.

    Returns:
        An instance of a subclass of :class:`torch_geometric.data.Dataset`.
    """
    path = osp.join(path, name)
    dataset = torch.load(path, **kwargs) 
    if transform is not None:
        dataset.transform = transform
    if pre_transform is not None:
        dataset.pre_transform = pre_transform
    if pre_filter is not None:
        dataset.pre_filter