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
"""Implementation of the PersLay-based CorianderNet network."""
from __future__ import annotations

from torch import nn
from torch.nn import functional as nnf

from morphoclass import layers


class CorianderNet(nn.Module):
    """A PersLay-based neural network for neuron m-type classification.

    Parameters
    ----------
    n_classes : int
        The number of m-type classes to predict.
    n_features : int
        The number of output feature maps for the PersLay layer.
    dropout : bool, default False
        If true a dropout layer is inserted between the two fully-connected
        layers of the classifier part of the network.
    """

    def __init__(self, n_classes=4, n_features=64, dropout=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.dropout = dropout

        # network backbone - feature extractor
        self.feature_extractor = layers.PersLay(out_features=self.n_features)

        # network head - classifier
        if dropout:
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features // 2),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.n_features // 2, n_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.n_features, self.n_features // 2),
                nn.ReLU(),
                nn.Linear(self.n_features // 2, n_classes),
            )

    def forward(self, data):
        """Compute the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Batch
            A batch of MorphologyEmbeddingDataset dataset.

        Returns
        -------
        log_softmax
            The log softmax of the predictions.
        """
        # Note: case in else section can be removed as soon as they implement
        # the feature, see https://github.com/pytorch/captum/issues/494
        if hasattr(data, "diagram"):
            x = data.diagram
        else:
            # captum tensors for XAI
            x = data[:, 0:2]

        # Note: case in else section can be removed as soon as they implement
        # the feature, see https://github.com/pytorch/captum/issues/494
        if hasattr(data, "diagram_batch"):
            point_index = data.diagram_batch
        else:
            # captum tensors for XAI
            point_index = data[:, 2].long()
            point_index.new()

        x = self.feature_extractor(x, point_index)
        x = self.classifier(x)

        return nnf.log_softmax(x, dim=1)

    def loss_acc(self, data):
        """Get loss and accuracy.

        Parameters
        ----------
        data : torch_geometric.data.Batch
            A batch of MorphologyEmbeddingDataset dataset.

        Returns
        -------
        loss : float
            The loss value.
        acc : float
            The accuracy value.
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
