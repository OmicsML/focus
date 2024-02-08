from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from ._decorators import auto_move_data

class BaseModuleClass(nn.Module):
    """Abstract base for focus modules."""
    
    def __init__(
        self,
    ):
        super().__init__()
    
    @property
    def device(self):
        device = list({p.device for p in self.parameters()})
        if len(device) > 1:
            raise RuntimeError("Module tensors on multiple devices.")
        return device[0]
    
    def on_load(self, model):
        """Called after model is loaded."""
        pass
        
    @auto_move_data
    def forward(
        self,
        tensors,
        get_inference_input_kwargs: dict | None = None,
        inference_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
        compute_loss=True,
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Forward pass through the network.

        Args:
            tensors: tensors to pass through
            get_inference_input_kwargs: Keyword args for ``_get_inference_input()``
            inference_kwargs: Keyword args for ``inference()``
            loss_kwargs: Keyword args for ``loss()``
            compute_loss: Whether to compute loss on forward pass. This adds
                another return value.
        """
        return _generic_forward(
            self,
            tensors,
            inference_kwargs,
            loss_kwargs,
            get_inference_input_kwargs,
            compute_loss,
        )

    @abstractmethod
    def _get_inference_input(self, tensors: dict[str, Tensor], **kwargs):
        """Parse tensors dictionary for inference related values."""

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> dict[str, Tensor | torch.distributions.Distribution]:
        """Run the recognition model.

        This function should return a dictionary with str keys and :class:`~torch.Tensor` values.
        """

    @abstractmethod
    def loss(self, *args, **kwargs) -> Tensor:
        """Compute the loss for a minibatch of data.

        This function uses the outputs of the inference to compute a loss.
        This many optionally include other penalty terms, which should be computed here.
        """


def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param


def _generic_forward(
    module,
    tensors,
    inference_kwargs,
    loss_kwargs,
    get_inference_input_kwargs,
    compute_loss,
) -> Union[Dict[str, Tensor], Tensor]:
    """Core of the forward call."""
    inference_kwargs = _get_dict_if_none(inference_kwargs)
    loss_kwargs = _get_dict_if_none(loss_kwargs)
    get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)

    inference_inputs = module._get_inference_input(tensors, **get_inference_input_kwargs)
    inference_outputs = module.inference(**inference_inputs, **inference_kwargs)
    if compute_loss:
        losses = module.loss(tensors, inference_outputs, **loss_kwargs)
        return {"loss": losses, "inference_outputs": inference_outputs}
    else:
        return inference_outputs
    
    
    
    