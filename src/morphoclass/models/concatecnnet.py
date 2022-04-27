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
"""Implementation of the ConcateCNNet classifier."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nnf

from morphoclass.models.cnnet import CNNEmbedder
from morphoclass.models.man_net import ManEmbedder


class ConcateCNNet(nn.Module):
    """A neuron m-type classifier based on graph and image convolutions.

    In the feature extraction part of the network graph convolution layers
    are applied to the graph node features of the apical dendrites, while
    the CNN layers are applied to the persistence image representation
    of the same data. The resulting features are concatenated and passed
    through a fully-connected layer for classification.

    Parameters
    ----------
    n_node_features : int
        The number of input node features for the GNN layers.
    n_classes : int
        The number of output classes.
    image_size : int
        The width (or height) of the input persistence images. It is assumed
        that the images are square so that the width and height are equal.
    bn : bool, default False
        Whether or not to include a batch normalization layer between the
        feature extractor and the fully-connected classification layer.
    """

    def __init__(self, n_node_features, n_classes, image_size, bn=False):
        super().__init__()
        self.n_node_features = n_node_features
        self.n_classes = n_classes
        self.image_size = image_size
        self.bn = bn

        self.gnn_embedder = ManEmbedder(n_features=self.n_node_features)
        self.cnn_embedder = CNNEmbedder()

        self.weight_gnn = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.weight_cnn = nn.Parameter(torch.ones([1]), requires_grad=True)

        n_out_features_gnn = 512
        n_out_features_cnn = 3 * (self.image_size // 4) ** 2
        n_out_features = n_out_features_gnn + n_out_features_cnn
        if self.bn:
            self.bn = nn.BatchNorm1d(num_features=n_out_features)

        self.fc = nn.Linear(in_features=n_out_features, out_features=self.n_classes)

    def forward(self, data, images):
        """Compute the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.data.Data
            A batch of input graph data for the GNN layers.
        images
            A batch of input persistence images for the CNN layers.

        Returns
        -------
        log_softmax
            The log softmax of the predictions.
        """
        x_gnn = self.weight_gnn * self.gnn_embedder(data)
        x_cnn = self.weight_cnn * self.cnn_embedder(images)

        x = torch.cat([x_gnn, x_cnn], dim=1)
        if self.bn:
            x = self.bn(x)
            x = nnf.relu(x)
        x = self.fc(x)

        return nnf.log_softmax(x, dim=1)
