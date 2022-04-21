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
"""Implementation of the AttentionGlobalPool layer."""
from __future__ import annotations

import torch
from torch.nn import Module
from torch.nn import Parameter
from torch.nn.init import calculate_gain
from torch.nn.init import xavier_uniform_
from torch_scatter import scatter_add


class AttentionGlobalPool(Module):
    """A graph global pooling layer with attention.

    Parameters
    ----------
    n_features : int
        The number of input features.
    attention_per_feature : bool, default False
        If true then separate attention weights are learned for each feature.
    save_attention : bool, default False.
        If true then the attention values generated upon the forward pass will
        be cached in the layer instance. Might be useful for debugging and
        explain-AI applications.
    """

    def __init__(self, n_features, attention_per_feature=False, save_attention=False):
        super().__init__()
        self.n_features = n_features
        self.per_feature = attention_per_feature
        self.save_attention = save_attention

        # Initialize weights and bias
        n_out = n_features if attention_per_feature else 1
        w = torch.empty([n_features, n_out])
        b = torch.zeros([n_out])

        gain = calculate_gain("tanh")
        xavier_uniform_(w, gain)

        self.weight = Parameter(w)
        self.bias = Parameter(b)

        self.last_a_j = None

    def forward(self, x, batch_segmentation):
        """Compute the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            A batch of node features.
        batch_segmentation : torch.Tensor
            A segmentation map for the node features. It's a one-dimensional
            tensor with integer entries. Nodes with the same value in the
            segmentation map are considered to be from the same graph.

        Returns
        -------
        torch.Tensor
            The pooled node features.
        """
        e_j = torch.tanh(x @ self.weight + self.bias)
        a_j = torch.exp(e_j)
        a_norm = scatter_add(a_j, batch_segmentation, dim=0)
        a_j = a_j / a_norm[batch_segmentation]

        if self.save_attention:
            self.last_a_j = a_j

        return scatter_add(a_j * x, batch_segmentation, dim=0)
