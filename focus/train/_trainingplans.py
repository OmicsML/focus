import math
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union
import copy
import numpy as np
import re
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall

from model.losses import loss_cl

TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]

from ..module.base import BaseModuleClass
from ..utils.utils import *
from ..module.nets._resgcn import ResGCN
from ..module.nets._gat import GAT
from ..module.nets._denseginconv import DenseGINConv

from ..model.view_generator import ViewGenerator_subgraph_based_one
from ..model.graph_encoder import GIN_MLP_Encoder

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
        out = self.forward(batch)
    
    def test_step(self, batch):
        """Test step for the model."""
        out = self.forward(batch)
    
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
        # TODO: check 
        requires_grad = True
        
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
        

        
class FocusTrainingPlan(pl.LightningModule):
    """LightningModule for training a Focus model. Subcellular Graph

    Args:
        pl (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    def __init__(
        self,
        # parameters for view graph generator
        view_graph_num_features: Tunable[int] = 64,
        view_graph_dim: Tunable[int] = 64,
        view_graph_encoder_s: Tunable[nn.Module] = GIN_MLP_Encoder,
        add_mask: Tunable[bool] = False,
        # parameters for gnn model
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
        
        # loss parameters
        lamb: Tunable[float] = 0.5,
        # others 
        args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # parameters for view graph generator
        self.view_graph_num_features = view_graph_num_features
        self.view_graph_dim =view_graph_dim
        self.view_graph_encoder_s = view_graph_encoder_s
        # parameters for gnn model
        
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
        
        # loss parameters 
        # \lambda to balance the contrastive loss and classification loss
        self.lamb = lamb
        
        
        self.ViewGraphGenerator = ViewGenerator_subgraph_based_one(
            view_graph_num_features= self.view_graph_num_features,
            view_graph_dim = self.hidden_dim,
            view_graph_encoder_s = self.encoder_s,
            add_mask = self.add_mask,
            args=args #TODO: check if this is necessary
        )
        
        
    
    # GNN config
    def get_model_with_configs(self, dataset):
        if self.model_name == "ResGCN":
            return ResGCN(dataset, 
                          hidden_dim=self.hidden, 
                          n_layers_feat=self.n_layers_feat, 
                          n_layers_conv=self.n_layers_conv, 
                          n_layers_fc=self.n_layers_fc, 
                          skip_connection=self.skip_connection, 
                          res_branch=self.res_branch, 
                          global_pooling=self.global_pooling, 
                          dropout=self.dropout, 
                          edge_norm=self.edge_norm)
        elif self.model_name == "DenseGIN":
            return DenseGINConv(dataset, 
                       hidden_dim=self.hidden, 
                       n_layers_feat=self.n_layers_feat, 
                       n_layers_conv=self.n_layers_conv, 
                       n_layers_fc=self.n_layers_fc, 
                       skip_connection=self.skip_connection, 
                       res_branch=self.res_branch, 
                       global_pooling=self.global_pooling, 
                       dropout=self.dropout, 
                       edge_norm=self.edge_norm)
        elif self.model_name == "GAT":
            return GAT(dataset, 
                       hidden_dim=self.hidden, 
                       n_layers_feat=self.n_layers_feat, 
                       n_layers_conv=self.n_layers_conv, 
                       n_layers_fc=self.n_layers_fc, 
                       skip_connection=self.skip_connection, 
                       res_branch=self.res_branch, 
                       global_pooling=self.global_pooling, 
                       dropout=self.dropout, 
                       edge_norm=self.edge_norm)
        else:
            raise ValueError(f"Unknown model {self.model_name}")
            
        
        
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        re = self.forward(batch)
        self.log("train_loss",re["loss"],on_epoch=True,prog_bar=True)
        return re["loss"]
        
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        re = self.forward(batch)
        return re
    
    def test_step(self, batch):
        """Test step for the model."""
        re = self.forward(batch)
        return re
        
    def optimizer_creator(self, parameters):
        # TODO: check if this is necessary
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
        
        # batch data is from the view graph generator
        view_graph_1 = self.ViewGraphGenerator(batch)
        view_graph_2 = self.ViewGraphGenerator(batch)
        
        GNN = self.get_model_with_configs(batch)
        GNN_view_graph_1 = GNN(view_graph_1)
        GNN_view_graph_2 = GNN(view_graph_2)
        GNN_raw_graph = GNN(batch)
        
        # for contrastive loss
        contrastive_loss = loss_cl(GNN_view_graph_1, GNN_view_graph_2)
        
        # for classification loss
        loss_raw_graph = F.nll_loss(GNN_raw_graph, batch.y)
        loss_view_1 = F.nll_loss(GNN_view_graph_1, batch.y)
        loss_view_2 = F.nll_loss(GNN_view_graph_2, batch.y)
        
        classification_loss = (loss_raw_graph + loss_view_1 + loss_view_2)/3
        
        # total loss for the model
        loss = self.lamb * contrastive_loss + (1 - self.lamb) * classification_loss
        
        return {"total_loss": loss, "contrastive_loss": contrastive_loss, "classification_loss": classification_loss}
        
        
        
        
        
        
        
        
        
        
        
        
        



