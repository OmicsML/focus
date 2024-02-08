import math
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union
import copy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall

TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]

from ..module.base import BaseModuleClass
from ..utils.utils import *

class TunableMeta(type):
    """Metaclass for Tunable class."""

    def __getitem__(cls, values):
        if not isinstance(values, tuple):
            values = (values,)
        return type("Tunable_", (Tunable,), {"__args__": values})


class Tunable(metaclass=TunableMeta):
    """Typing class for tagging keyword arguments as tunable."""


class ViewGraphGeneratorTrainingPlan(pl.LightningModule):
    """LightningModule for training a view graph generator model.

    Args:   
        pl (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    def __init__(
        self,
        optimizer: Tunable[Literal["Adam", "AdamW", "Custom"]] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: Tunable[float] = 1e-3,
        lr_scheduler: Literal[None, "step", "cosine", "plateau"] = None,
        num_features: Tunable[int] = 64,
        hidden_dim: Tunable[int] = 64,
        add_mask : Tunable[bool] = False,
        save_epoch: Tunable[int] = 1,
        encoder: Tunable[type[nn.Module]] = None,
        mode: Tunable[str] = "node", 
        
        
        
    ):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_creator = optimizer_creator
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.add_mask = add_mask
        self.save_epoch = save_epoch
        self.encoder = encoder
        
        
        
        
        
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        outputs = self.forward(batch)
        
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        pass
    
    def test_step(self, batch):
        """Test step for the model."""
        pass
    
    def configure_optimizers(self):
        """Configure optimizers for the model."""
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == "Custom":
            optimizer = self.optimizer_creator(self.parameters())
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        return optimizer
    
    def forward(self, batch, **kwargs):
        """Forward pass for the model."""
        data = copy.deepcopy(batch)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
        
        data.x = data.x.float()
        x = x.float()
        x.requires_grad = requires_grad # TODO: check if this is necessary
        
        subgraph_p, subsplit, node_p, intra_p, intra_edge_index, intra_stable_edge_index, inter_p, \
            inter_edge_index, inter_stable_edge_index = self.encoder(x, edge_index, edge_attr)
        sample = F.gumble_softmax(subgraph_p, hard=True)
        sub_shuffle_sample = sample[:, 0]
        intra_edge_sample = sample[:,1] 
        inter_edge_sample = sample[:,2] 
        node_sample = sample[:,3] 
        node_mask_sample = sample[:,4] 
        none = sample[:,5]
        
        # subgraph_shuffle
        if (sub_shuffle_sample>0).any():
            sub_adj = subgraph_shuffle(subgraph_p, subsplit, sub_shuffle_sample)
            
            # TODO: implement save function
        
        else:
            sub_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
            
        # intra/inter_edge 
        if (intra_edge_sample>0).any() or (inter_edge_sample>0).any():
            edge_encoder_result = (intra_p, intra_edge_index, intra_stable_edge_index, inter_p, inter_edge_index, inter_stable_edge_index)
            edge_adj = edge_permutation(data, subsplit, intra_edge_sample, inter_edge_sample, edge_encoder_result)
        else:
            edge_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
        
        
        # node drop
        if (node_sample>0).any():
            x, drop_node_sample, node_adj = node_drop(data, subsplit, node_sample, requires_grad, node_p)
        else:
            x = x
            drop_node_sample = torch.zeros_like(node_sample).to(data.edge_index.device)
            node_adj = torch.sparse_coo_tensor([[], []], [], (data.num_nodes, data.num_nodes)).to(data.edge_index.device)
            
        # node mask
        if (node_sample>0).any():
            keep_x_mask, mask_node_sample = node_mask(data, subsplit, node_mask_sample, requires_grad, node_p)
            x[keep_x_mask] = data.x.detach().mean()
        else:
            x = x
            mask_node_sample = torch.zeros_like(node_mask_sample).to(data.edge_index.device)
    
        ori_adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(edge_index.device), \
            (data.num_nodes, data.num_nodes), requires_grad=True)
        adj = (ori_adj + sub_adj + edge_adj + node_adj).coalesce()
        
        values = adj.values()
        values = torch.where(values>1, torch.ones_like(values), values)
        values = torch.where(values<0, torch.zeros_like(values), values)
        
        norm_adj = torch.sparse_coo_tensor(adj.indices(), values, (data.num_nodes, data.num_nodes), requires_grad=True)

        data.adj = norm_adj
        data.x = x
        return data
        

        
class GNNTrainingPlan(pl.LightningModule):
    """LightningModule for training a GNN model.

    Args:
        pl (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    def __init__(
        self,
        optimizer: Tunable[Literal["Adam", "AdamW", "Custom"]] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: Tunable[float] = 1e-3,
        gnn_net: Tunable[Literal["GAT", "GIN", "ResGCN"]] = "ResGCN",
        n_layers_fc: Tunable[int] = 2,
        n_layers_feat: Tunable[int] = 2,
        n_layers_conv: Tunable[int] = 2,
        skip_connection: Tunable[bool] = True,
        res_branch: Tunable[bool] = True,
        global_pooling: Tunable[bool] = True,
        dropout: Tunable[float] = 0.5,
        edge_norm: Tunable[bool] = True,
        hidden: Tunable[int] = 64,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_creator = optimizer_creator
        self.lr = lr
        self.model_name = gnn_net,
        self.n_layers_fc = n_layers_fc
        self.n_layers_feat = n_layers_feat
        self.n_layers_conv = n_layers_conv
        self.skip_connection = skip_connection
        self.res_branch = res_branch
        self.global_pooling = global_pooling
        self.dropout = dropout
        self.edge_norm = edge_norm
        self.hidden = hidden
        
        
        
        
        
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        outputs = self.forward(batch)
        
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        pass
    
    def test_step(self, batch):
        """Test step for the model."""
        pass
    
    def configure_optimizers(self):
        """Configure optimizers for the model."""
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == "Custom":
            optimizer = self.optimizer_creator(self.parameters())
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        return optimizer
    
    def forward(self, batch, **kwargs):
        """Forward pass for the model."""
        pass




class FocusTrainingPlan(pl.LightningModule):
    """LightningModule for training a focus model.





#     Args:
#         pl (_type_): _description_
#     """
#     def __init__(self, 
#                  module_list:  List[BaseModuleClass],
#                  optimizer: Tunable[Literal["Adam", "AdamW", "Custom"]] = "Adam",
#                  optimizer_creator: Optional[TorchOptimizerCreator] = None,
#                  lr: Tunable[float] = 1e-3,
#                  weight_decay: Tunable[float] = 1e-6,
#                  eps: Tunable[float] = 0.01,
#                  lr_scheduler: Literal[None, "step", "cosine", "plateau"] = None,
#                  lr_factor: Tunable[float] = 0.6,
#                  lr_patience: Tunable[int] = 30,
#                  lr_threshold: Tunable[float] = 0.0,
#                  lr_scheduler_metric: Literal["reconstruction_loss_validation"] = "reconstruction_loss_validation",
#                  lr_min: Tunable[float] = 0,
#                  lr_decay_steps: Tunable[int] = 0,
#                  lr_decay_rate: Tunable[float] = 0.1,
#                  lr_decay_min_lr: Tunable[float] = 0,
#                  lr_decay_iters: Tunable[int] = 100000,
#                  custom_decay_lr: bool = False,
#                  warmup_iters: int = 10000,
#                  loss_kwargs: dict = {},
#                  **kwargs):
#         super().__init__()
#         self.module_list = module_list
#         self.add_module("module_list", module_list)
#         self.optimizer = optimizer
#         self.optimizer_creator = optimizer_creator
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.eps = eps
#         self.lr_scheduler = lr_scheduler
#         self.lr_factor = lr_factor
#         self.lr_patience = lr_patience
#         self.lr_threshold = lr_threshold
#         self.lr_scheduler_metric = lr_scheduler_metric
#         self.lr_min = lr_min
#         self.lr_decay_steps = lr_decay_steps
#         self.lr_decay_rate = lr_decay_rate
#         self.lr_decay_min_lr = lr_decay_min_lr
#         self.lr_decay_iters = lr_decay_iters
#         self.custom_decay_lr = custom_decay_lr
#         self.warmup_iters = warmup_iters
#         self.loss_kwargs = loss_kwargs
#         # self.save_hyperparameters()
#         self.ViewGraphGenerator = module_list[0]
#         self.GNN = module_list[1]
        
        
        
        
        
#     @property
#     def use_sync_dist(self):
#         return isinstance(self.trainer.strategy, DDPStrategy)
    
#     def training_step(self, batch, batch_idx):
#         """Training step for the model."""
#         outputs = self.forward(batch, compute_loss=True)
        
#         #TODO: implement loss computation
#         # return {'loss': loss}
#         pass 
        
    
#     def validation_step(self, batch, batch_idx):
#         """Validation step for the model."""
        
    
#     def test_step(self, batch):
#         """Test step for the model."""
#         pass
    
#     def configure_optimizers(self):
#         """Configure optimizers for the model."""
#         # view graph generator optimizer
#         if self.optimizer == "Adam":
#             optimizer = torch.optim.Adam(self.ViewGraphGenerator.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=self.eps)
#         elif self.optimizer == "AdamW":
#             optimizer = torch.optim.AdamW(self.ViewGraphGenerator.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=self.eps)
#         elif self.optimizer == "Custom":
#             optimizer = self.optimizer_creator(self.ViewGraphGenerator.parameters())
#         else:
#             raise ValueError(f"Unknown optimizer {self.optimizer}")
    
#     def configure_schedulers(self):
#         """Configure schedulers for the model."""
#         pass
    
#     def forward(self, batch, **kwargs):
#         """Forward pass for the model."""
#         # TODO: implement forward pass and loss computation
#         augmented_data1 = self.ViewGraphGenerator(batch)
#         augmented_data2 = self.ViewGraphGenerator(batch)

        
        
    
    
        