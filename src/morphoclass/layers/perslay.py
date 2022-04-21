# Copyright © 2022 Blue Brain Project/EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the PersLay layer."""
from __future__ import annotations

import abc
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_scatter import scatter_max
from torch_scatter import scatter_mean
from torch_scatter import scatter_softmax
from torch_scatter import scatter_sum


class PointTransformer(nn.Module, abc.ABC):
    """A point transformation for persistence diagram embedding."""

    @abc.abstractmethod
    def forward(self, input: torch.Tensor, point_index: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        input : torch.Tensor
            A batch of input data.
        point_index : torch.Tensor
            A segmentation map for the samples in the batch.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """

    @abc.abstractmethod
    def extra_repr(self) -> str:
        """Get a string representation of layer parameters."""


class GaussianPointTransformer(PointTransformer):
    """Applies Gaussian point transformation for persistence diagram embedding.

    This transformation can be applied to persistence diagrams, and is
    implemented as described in [1]. Note that points are assumed to lie in a
    normalized diagram, i.e. coordinates should lie in the interval (0, 1).

    Shapes:

    - input: [N, 2].
      N is the total number of points in the batch (belonging to different
      persistence diagrams as specified by point_index), and 2 refers to the
      2 coordinates of each point of each persistence diagram i.e.
      (birth_date, death_date).
    - point_index: [N].
      N is the total number of points in the batch.
    - output: [N, Q].
      N is the total number of points in the batch, and Q is the desired
      number of learnable sample points used in the transformation.

    Parameters
    ----------
    out_features : int
        Size of each output sample, corresponding to the desired number of
        sample points.

    Attributes
    ----------
    sample_points : tensor
        Sample points of the transformation, with shape [2, Q].
    sample_inverse_sigmas : tensor
        Inverse standard deviations of the transformation, with shape [2, Q]
        since each of the 2 dimensions may have different sigma.

    Examples
    --------
    >>> inp = torch.tensor([[0.2, 0.3], [0.1, 0.4], [0.6, 0.3], [0.2, 0.1],\
        [0.5, 0.2], [0.1, 0.3], [0.6, 0.2]])
    >>> point_index = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    >>> m = GaussianPointTransformer(out_features=32)
    >>> output = m(inp)
    >>> print(inp.size())
    torch.Size([7, 2])
    >>> print(output.size())
    torch.Size([7, 32])

    References
    ----------
    [1] Carriere, Mathieu, et al. "PersLay: A Neural Network Layer for
    Persistence Diagrams and New Graph Topological Signatures." stat 1050
    (2019): 17.
    """

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.sample_points = Parameter(torch.Tensor(2, self.out_features))
        self.sample_inverse_sigmas = Parameter(torch.Tensor(2, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Randomly initialize the trainable parameters."""
        # points must lie in (0, 1) x (0, 1)
        nn.init.uniform_(self.sample_points, a=0.0, b=1.0)
        # reasonable values for the standard deviations seem to be in (0.8, 1.2)
        nn.init.uniform_(self.sample_inverse_sigmas, a=0.8, b=1.2)

    def forward(self, input, point_index):
        """Perform the forward pass.

        Parameters
        ----------
        input : torch.Tensor
            A batch of input data.
        point_index : torch.Tensor
            A segmentation map for the samples in the batch.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        x = input.unsqueeze(-1)
        z = (x - self.sample_points) * self.sample_inverse_sigmas
        return torch.exp(-torch.sum(z**2, dim=-2))

    def extra_repr(self):
        """Get a string representation of layer parameters."""
        return f"out_features={self.out_features}"


class PointwisePointTransformer(PointTransformer):
    """Applies point-wise point transformation for persistence diagram embedding.

    Parameters
    ----------
    out_features : int
        Size of each output sample, corresponding to the desired number of
        sample points.
    hidden_features : int
        The size of the hidden layer.
    """

    def __init__(self, out_features, hidden_features=32):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.layer1 = nn.Linear(in_features=2, out_features=hidden_features)
        self.layer2 = nn.Linear(in_features=hidden_features, out_features=out_features)

    def forward(self, input, point_index):
        """Perform the forward pass.

        Parameters
        ----------
        input : torch.Tensor
            A batch of input data.
        point_index : torch.Tensor
            A segmentation map for the samples in the batch.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        x = self.layer1(input)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x

    def extra_repr(self):
        """Get a string representation of layer parameters."""
        return (
            f"hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}"
        )


class PersLay(nn.Module):
    """Applies PersLay embedding layer.

    This transformation can be applied to persistence diagrams, and is
    implemented as described in [1]. Note that points are assumed to lie in a
    normalized diagram, i.e. coordinates should lie in the interval (0, 1).

    Shapes:

    - input : [N, 2].
      N is the total number of points in the batch (belonging to different
      persistence diagrams as specified by point_index), and 2 refers to the
      2 coordinates of each point of each persistence diagram i.e.
      (birth_date, death_date).
    - point_index : [N].
      N is the total number of points in the batch.
    - output : [D, Q].
      D is the number of different persistence diagrams in the batch, and Q is
      the desired number of learnable sample points used in the
      transformation.

    Parameters
    ----------
    out_features : int
        Output size of the produced embedding.
    transformation : str or nn.Module
        A point transformation, mapping each point of a persistence diagram
        to a vector. One of 'gaussian', 'pointwise', or a nn.Module.
    operation : str
        A permutation invariant operation. One of 'sum', 'mean', 'max'.
    weights : str
        Approach to be used for the weights. One of 'attention' (learnable
        pointwise weights), 'uniform' (all weights set to 1), 'grid' (learnable
        weights on a 10x10 grid).

    References
    ----------
    [1] Carriere, Mathieu, et al. "PersLay: A Neural Network Layer for
    Persistence Diagrams and New Graph Topological Signatures." stat 1050
    (2019): 17.
    """

    def __init__(
        self,
        out_features,
        transformation="gaussian",
        operation="sum",
        weights="uniform",
    ):
        super().__init__()
        self.out_features = out_features
        self.transformation = transformation
        self.operation = operation
        self.weights = weights

        self.point_transformer: PointTransformer
        if self.transformation == "gaussian":
            self.point_transformer = GaussianPointTransformer(self.out_features)
        elif self.transformation == "pointwise":
            self.point_transformer = PointwisePointTransformer(self.out_features)
        elif isinstance(self.transformation, PointTransformer):
            self.point_transformer = self.transformation
        else:
            raise ValueError(
                f"Point transformation {self.transformation} is not" f"available!"
            )

        self.reduction: Callable
        if self.operation == "mean":
            self.reduction = partial(scatter_mean, dim=0)
        elif self.operation == "max":
            self.reduction = lambda *args, **kwargs: partial(scatter_max, dim=0)(
                *args, **kwargs
            )[0]
        elif self.operation == "sum":
            self.reduction = partial(scatter_sum, dim=0)
        else:
            raise ValueError(
                f"Permutation invariant operation {self.operation}"
                f" is not available!"
            )

        if self.weights == "attention":
            self.a_linear = nn.Linear(in_features=2, out_features=1)
        elif self.weights == "uniform":
            pass
        elif self.weights == "grid":
            self.n_grid_points = 10
            self.grid_min, self.grid_max = -0.001, 1.001
            self.w_grid = Parameter(
                torch.Tensor(self.n_grid_points, self.n_grid_points, 1)
            )
        else:
            raise ValueError(f"Attention weights {self.weights} are " f"not available!")

        self.reset_parameters()

    def reset_parameters(self):
        """Randomly initialize trainable parameters."""
        if hasattr(self, "w_grid"):
            nn.init.uniform_(self.w_grid, a=0.0, b=1.0)

    def forward(self, input, point_index):
        """Perform the forward pass.

        Parameters
        ----------
        input : torch.Tensor
            A batch of input data.
        point_index : torch.Tensor
            A segmentation map for the samples in the batch.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        phi_x = self.point_transformer(input, point_index)

        if self.weights == "uniform":
            return self.reduction(phi_x, point_index)
        else:
            if self.weights == "attention":
                w_x = self.a_linear(input)
                w_x = torch.tanh(w_x)
                w_x = scatter_softmax(w_x, point_index, dim=0)
            elif self.weights == "grid":
                idxs = (
                    self.n_grid_points
                    * (input - self.grid_min)
                    / (self.grid_max - self.grid_min)
                ).long()
                w_x = self.w_grid[idxs[..., 0], idxs[..., 1]]
            return self.reduction(phi_x * w_x, point_index)
