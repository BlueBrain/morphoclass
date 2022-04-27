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
"""Implementation of the MultiAdjNet classifier."""
from __future__ import annotations

import torch
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn.functional import log_softmax
from torch.nn.functional import nll_loss
from torch_geometric.nn import global_mean_pool

from morphoclass import layers


class MultiAdjNet(torch.nn.Module):
    """Model for classifying morphologies of pyramidal neurons.

    This is the architecture that performed best in the
    TensorFlow implementation. It consists of two graph
    convolutions layers with each computing two convolutions:
    one on the directed adjacency matrix, and one with the
    adjacency matrix with the reversed direction. The results
    of both convolutions are concatenated and passed to the next
    layer. After the two parallel graph convolutions follows
    a global average pooling layer and a fully-connected layer.
    Finally, a softmax layer is used for prediction.

    Parameters
    ----------
    n_features : int, default 1
        The number of input features.
    n_classes : int, default 4
        The number of output classes.
    attention : bool, default False
        If true, then an attention-based global pooling layer will be used,
        if false a global mean pooling layer.
    attention_per_feature : bool, default False
        If true then the attention will be optimized for each feature separately.
        This will increase the number of trainable parameters. Only has effect
        if the parameter `attention` is true.
    save_attention : bool, default False
        If true then the attention weights will be cached within the attention
        layer instance. See the `AttentionGlobalPool` class for more details.
    """

    def __init__(
        self,
        n_features=1,
        n_classes=4,
        attention=False,
        attention_per_feature=False,
        save_attention=False,
    ):
        super().__init__()
        # Note that K=4 in GCN corresponds to K=5 in PyG
        self.conv11 = layers.ChebConv(n_features, 64, K=5)
        self.conv12 = layers.ChebConv(n_features, 64, K=5)
        self.cat1 = layers.Cat()

        self.conv21 = layers.ChebConv(128, 256, K=5)
        self.conv22 = layers.ChebConv(128, 256, K=5)
        self.cat2 = layers.Cat()

        self.relu = ReLU()

        if attention:
            self.pool = layers.AttentionGlobalPool(
                n_features=512,
                attention_per_feature=attention_per_feature,
                save_attention=save_attention,
            )
        else:
            self.pool = global_mean_pool
        self.fc = Linear(512, n_classes)

    def forward(self, data):
        """Compute the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.data.Data
            A batch of input data.

        Returns
        -------
        log_softmax
            The log softmax of the predictions.
        """
        x, edge_index = data.x, data.edge_index
        edge_index_rev = edge_index[[1, 0]]

        x1 = self.relu(self.conv11(x, edge_index))
        x2 = self.relu(self.conv12(x, edge_index_rev))
        x = self.cat1([x1, x2])

        x1 = self.relu(self.conv21(x, edge_index))
        x2 = self.relu(self.conv22(x, edge_index_rev))
        x = self.cat2([x1, x2])

        # Note: this only works if data is of type Batch
        # this would fail for full-batch training where
        # data is of type Data
        x = self.pool(x, data.batch)
        x = self.fc(x)

        return log_softmax(x, dim=1)

    def accuracy(self, data):
        """Run the forward pass and compute the accuracy.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        acc : float
            The accuracy on the current data batch.
        """
        pred = self(data).argmax(dim=1)
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / len(data.y)
        return acc

    def loss_acc(self, data):
        """Run the forward pass and compute the loss and accuracy.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        loss : float
            The loss on the given data batch.
        acc : float
            The accuracy on the current data batch.
        """
        # Predictions
        out = self(data)

        # Loss
        loss = nll_loss(out, data.y).item()

        # Accuracy
        pred = out.argmax(dim=1)
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / len(data.y)

        return loss, acc
