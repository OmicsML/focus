from functools import partial
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import os.path as osp
import numpy as np
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
        
def unique_list_mapping_to_one_hot(unique_list: List, target_list: List)-> np.array:
    """\
        Convert a list of Unique list to one hot vector.
    """
    unique_elements = sorted(set(unique_list))
    element_to_index = {element: index for index, element in enumerate(unique_elements)}
    
    one_hot_encodings = []
    for target_element in target_list:
        if target_element not in element_to_index:
            raise ValueError("Target element not found in unique list.")
        
        one_hot_vector = [0] * len(unique_elements)
        target_index = element_to_index[target_element]
        one_hot_vector[target_index] = 1
        one_hot_encodings.append(one_hot_vector)
    
    return np.array(one_hot_encodings)

def read_gene_list(gene_list_txt_path: str) -> List[str]:
    gene_list = []
    f = open(gene_list_txt_path, 'r')
    for line in f:
        gene_list.append(line.strip())
    f.close()
    return gene_list