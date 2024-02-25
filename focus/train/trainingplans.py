import math
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union
import copy
import numpy as np
import re
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.optim.optimizer import Optimizer
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall

from model.losses import loss_cl

TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]

from focus.utils.utils import *
from focus.module.nets._resgcn import ResGCN
from focus.module.nets._gat import GAT
from focus.module.nets._denseginconv import DenseGINConv

from focus.model.view_generator import ViewGenerator_subgraph_based_one
from focus.model.graph_encoder import GIN_MLP_Encoder



        
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
        n_feat: Union[int] = 64,
        view_graph_dim: Union[int] = 64,
        view_graph_encoder: Union[Literal['GIN_MLP_Encoder']] = "GIN_MLP_Encoder",
        add_mask: Union[bool] = False,
        # parameters for gnn model
        optimizer: Union[Literal["Adam", "AdamW", "Custom"]] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: Union[float] = 1e-3,
        gnn_net: Union[Literal["GAT", "GIN", "ResGCN"]] = "ResGCN",
        n_layers_fc: Union[int] = 2,
        n_layers_feat: Union[int] = 2,
        n_layers_conv: Union[int] = 2,
        skip_connection: Union[bool] = True,
        res_branch: Union[bool] = True,
        global_pooling: Union[bool] = True,
        dropout: Union[float] = 0.5,
        edge_norm: Union[bool] = True,
        hidden: Union[int] = 64,
        label_num: Union[int] = None,
        
        # loss parameters
        lamb: Union[float] = 0.5,
    ):
        super().__init__()
        
        # parameters for view graph generator
        self.n_feat = n_feat
        self.view_graph_dim =view_graph_dim
        if view_graph_encoder == "GIN_MLP_Encoder":
            self.view_graph_encoder = GIN_MLP_Encoder
        self.add_mask = add_mask
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
        self.label_num = label_num
        
        # loss parameters 
        # \lambda to balance the contrastive loss and classification loss
        self.lamb = lamb
        
        self.ViewGraphGenerator = ViewGenerator_subgraph_based_one(
            view_graph_num_features= self.n_feat,
            view_graph_dim = self.view_graph_dim,
            view_graph_encoder = self.view_graph_encoder,
            add_mask = self.add_mask,
        )
        
        self.GNN = self.get_model_with_configs()
        
        
    
    # GNN config
    def get_model_with_configs(self):
        if self.model_name == "ResGCN":
            return ResGCN(nfeat=self.n_feat, 
                          hidden_dim=self.hidden, 
                          n_layers_feat=self.n_layers_feat, 
                          n_layers_conv=self.n_layers_conv, 
                          n_layers_fc=self.n_layers_fc, 
                          skip_connection=self.skip_connection, 
                          res_branch=self.res_branch, 
                          global_pooling=self.global_pooling, 
                          dropout=self.dropout, 
                          edge_norm=self.edge_norm,
                          label_num=self.label_num)
        elif self.model_name == "DenseGIN": # rewrite the class 
            return DenseGINConv(nfeat = self.n_feat, 
                       hidden_dim=self.hidden, 
                       n_layers_feat=self.n_layers_feat, 
                       n_layers_conv=self.n_layers_conv, 
                       n_layers_fc=self.n_layers_fc, 
                       skip_connection=self.skip_connection, 
                       res_branch=self.res_branch, 
                       global_pooling=self.global_pooling, 
                       dropout=self.dropout, 
                       edge_norm=self.edge_norm,
                       label_num=self.label_num)
        elif self.model_name == "GAT":
            return GAT(nfeat=self.n_feat, 
                       hidden_dim=self.hidden, 
                       n_layers_feat=self.n_layers_feat, 
                       n_layers_conv=self.n_layers_conv, 
                       n_layers_fc=self.n_layers_fc, 
                       skip_connection=self.skip_connection, 
                       res_branch=self.res_branch, 
                       global_pooling=self.global_pooling, 
                       dropout=self.dropout, 
                       edge_norm=self.edge_norm,
                       label_num=self.label_num)
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
    
    def optimizer_step(self, 
                       epoch: int, 
                       batch_idx: int, 
                       optimizer: Optimizer | LightningOptimizer, 
                       optimizer_closure: Callable[[], Any] | None = None) -> None:
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
    
    def configure_optimizers(self):
        """Configure optimizers for the model."""
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == "Custom":
            optimizer = self.optimizer_creator(self.parameters()) # TODO: check
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        return optimizer
    
    def forward(self, batch):
        """Forward pass for the model."""
        
        # batch data is from the view graph generator
        view_graph_1 = self.ViewGraphGenerator(batch)
        view_graph_2 = self.ViewGraphGenerator(batch)
        
        GNN_view_graph_1 = self.GNN(view_graph_1)
        GNN_view_graph_2 = self.GNN(view_graph_2)
        GNN_raw_graph = self.GNN(batch)
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        



