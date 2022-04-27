# Copyright © 2022-2022 Blue Brain Project/EPFL
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
"""Implementation of the ChebConv layer."""
from __future__ import annotations

import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add


class ChebConv(MessagePassing):
    """ChebConv layer based on `torch_geometric.nn.conv.ChebConv`.

    This implementation corrects a few things that were not quite correct
    in the original implementation:

    1. Initialization set to glorot rather then uniform
    2. The default flow is set to `target_to_source`. This way the call
       `self.propagate(...)` computes `M @ x` rather than `M.T @ x`.
    3. The computation of the normalised Laplacian

    These points might be implemented in future versions of PyG so that this
    implementation will become redundant.

    Parameters
    ----------
    in_channels : int
        Size of each input sample.
    out_channels : int
        Size of each output sample.
    K : int
        Chebyshev filter size, *i.e.* number of hops :math:`K`.
    bias : bool, default True
        If set to :obj:`False`, the layer will not learn an additive bias.
    **kwargs
        Additional arguments of `torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self, in_channels, out_channels, K, bias=True, flow="target_to_source", **kwargs
    ):
        kwargs["flow"] = flow
        super().__init__(**kwargs)

        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all weights with Glorot and set all biases to zero."""
        # Note we're using glorot rather than uniform here
        glorot(self.weight)
        zeros(self.bias)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None, lambda_max=3.0):
        """Given the adjacency matrix calculate the normalised Laplacian.

        Parameters
        ----------
        edge_index
            The edge indices of the adjacency matrix.
        num_nodes
            The number of nodes in the graph, the adjacency matrix is of
            dimension `(num_nodes, num_nodes)`.
        edge_weight
            The entries in the adjacency matrix index by `edge_index`
        dtype
            The dtype of the adjacency matrix.
        lambda_max
            The pre-computed maximal eigenvalue of the normalised Laplacian.
            Note that in the original PyG implementation this was implicitly
            assumed to have value `2.0`.

        Returns
        -------
        edge_index, norm
            The sparse form of the normalised graph Laplacian.
        """
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # Compute A_norm = D^{-1/2}.A.D^{-1/2}
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Compute the laplacian L_norm = 1 - A_norm
        edge_index, norm = add_self_loops(
            edge_index=edge_index, edge_weight=-norm, fill_value=1, num_nodes=num_nodes
        )

        # Compute (tilde L)_norm = 2 / lambda_max L_norm - 1
        edge_index, norm = add_self_loops(
            edge_index=edge_index,
            edge_weight=2.0 / lambda_max * norm,
            fill_value=-1,
            num_nodes=num_nodes,
        )

        return edge_index, norm

    def forward(self, x, edge_index, edge_weight=None, batch=None, lambda_max=3.0):
        """Compute the forward pass.

        Parameters
        ----------
        x
            The batched input node features.
        edge_index
            The batched input adjacency matrices.
        edge_weight
            The edge weights.
        batch
            Not used?
        lambda_max : float
            The lambda_max value to use for the normalization of the graph
            Laplacian.

        Returns
        -------
        out : torch.Tensor
            The output feature maps.
        """
        edge_index, norm = self.norm(
            edge_index, x.size(0), edge_weight, x.dtype, lambda_max=lambda_max
        )

        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])

        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        """Perform a message passing step.

        Parameters
        ----------
        x_j
            Node features.
        norm
            The edge weights (?). See documentation of `pytorch-geometric`
            for more details.

        Returns
        -------
        torch.Tensor
            The result of message passing.
        """
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        """Compute the repr of the object."""
        return (
            f"{self.__class__.__qualname__}("
            f"{self.in_channels}, "
            f"{self.out_channels}, "
            f"K={self.weight.size(0)})"
        )
