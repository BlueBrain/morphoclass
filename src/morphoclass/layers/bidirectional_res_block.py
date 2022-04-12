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
"""Implementation of the BidirectionalResBlock layer."""
from __future__ import annotations

from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU

from morphoclass.layers.bidirectional_block import BidirectionalBlock


class BidirectionalResBlock(Module):
    """A BidirectionalBlock layer with a skip connection.

    Parameters
    ----------
    c_in : int
        The number of input channels.
    c_out : int
        The number of output channels.
    """

    def __init__(self, c_in, c_out):
        super().__init__()

        if c_out % 2 != 0:
            raise ValueError("The output channel size must be even")

        self.bidirectional_1 = BidirectionalBlock(c_in, c_out)
        self.fc = Linear(c_out, c_out // 2)
        self.bidirectional_2 = BidirectionalBlock(c_out // 2, c_out)
        self.proj = Linear(c_in, c_out)
        self.relu = ReLU()

    def forward(self, x, edge_index):
        """Perform the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            A batch of node features.
        edge_index : torch.Tensor
            A batch of adjacency matrices.

        Returns
        -------
        x : torch.Tensor
            The output node features.
        """
        x_orig = x
        x = self.bidirectional_1(x, edge_index)
        x = self.relu(x)
        x = self.fc(x)
        x = self.bidirectional_2(x, edge_index)
        x = x + self.proj(x_orig)

        return x
