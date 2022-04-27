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
"""Implementation of the BidirectionalNet classifier."""
from __future__ import annotations

import torch.nn
import torch.nn.functional as nnf
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_mean_pool


class BidirectionalNet(torch.nn.Module):
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
    num_classes : int
        The number of output classes.
    num_nodes_features : int
        The number of input node features.
    """

    def __init__(self, num_classes, num_nodes_features):
        super().__init__()
        self.conv11 = ChebConv(num_nodes_features, 64, K=4)
        self.conv12 = ChebConv(num_nodes_features, 64, K=4)

        self.conv21 = ChebConv(128, 256, K=4)
        self.conv22 = ChebConv(128, 256, K=4)

        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(512, num_classes)

        self.num_nodes_features = num_nodes_features
        self.abs_max = torch.empty(num_nodes_features, dtype=torch.float32)

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

        # histoos([1, self.num_nodes_features], dtype=torch.float32)

        edge_index_rev = edge_index[[1, 0]]

        x1 = nnf.relu(self.conv11(x, edge_index))
        x2 = nnf.relu(self.conv12(x, edge_index_rev))
        x = torch.cat([x1, x2], dim=1)

        x1 = nnf.relu(self.conv21(x, edge_index))
        x2 = nnf.relu(self.conv22(x, edge_index_rev))
        x = torch.cat([x1, x2], dim=1)

        # Note: this only works if data is of type Batch
        # this would fail for full-batch training where
        # data is of type Data
        x = self.pool(x, data.batch)
        x = self.fc(x)

        return nnf.log_softmax(x, dim=1)

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
        loss = nnf.nll_loss(out, data.y).item()

        # Accuracy
        pred = out.argmax(dim=1)
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / len(data.y)

        return loss, acc
