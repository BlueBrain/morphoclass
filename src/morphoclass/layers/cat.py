"""Implementation of the Cat layer."""
from __future__ import annotations

import torch
from torch.nn import Module


class Cat(Module):
    """Concatenate the outputs of multiple parallel layers."""

    def __init__(self):
        super().__init__()

    def forward(self, x_list):
        """Perform the forward pass.

        Parameters
        ----------
        x_list : iterable of torch.Tensor
            An iterable of outputs of a number of torch layers.

        Returns
        -------
        torch.Tensor
            The concatenated input tensors.
        """
        return torch.cat(x_list, dim=1)
