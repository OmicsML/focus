



import copy
from ctypes import Union
from typing import Callable, Iterable, Literal, Optional

import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl
from focus.utils.utils import edge_permutation, node_drop, node_mask, subgraph_shuffle
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]


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
        optimizer: Union[Literal["Adam", "AdamW", "Custom"]] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: Union[float] = 1e-3,
        lr_scheduler: Literal[None, "step", "cosine", "plateau"] = None,
        num_features: Union[int] = 64,
        hidden_dim: Union[int] = 64,
        add_mask : Union[bool] = False,
        save_epoch: Union[int] = 1,
        encoder: Union[type[nn.Module]] = None,
        mode: Union[str] = "node", 
        
        
        
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
        
