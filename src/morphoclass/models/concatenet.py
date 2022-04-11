"""Implementation of the ConcateNet classifier."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nnf

from morphoclass.layers import PersLay
from morphoclass.models.man_net import ManEmbedder


class ConcateNet(nn.Module):
    """A neuron m-type classifier based on graph convolutions and PersLay.

    In the feature extraction part of the network graph convolution layers
    are applied to the graph node features of the apical dendrites, while
    the PersLay layer is applied to the persistence diagram representation
    of the same data. The resulting features are concatenated and passed
    through a fully-connected layer for classification.

    Parameters
    ----------
    n_node_features : int
        The number of input node features for the GNN layers.
    n_classes : int
        The number of output classes.
    n_features_perslay : int
        The number of features for the PersLay layer.
    bn : bool, default False
        Whether or not to include a batch normalization layer between the
        feature extractor and the fully-connected classification layer.
    """

    def __init__(self, n_node_features, n_classes, n_features_perslay, bn=False):

        super().__init__()
        self.n_node_features = n_node_features
        self.n_classes = n_classes
        self.n_features_perslay = n_features_perslay
        self.bn = bn

        self.gnn_embedder = ManEmbedder(n_features=self.n_node_features)
        self.perslay_embedder = PersLay(out_features=self.n_features_perslay)

        self.weight_gnn = nn.Parameter(torch.ones([1]), requires_grad=True)
        self.weight_perslay = nn.Parameter(torch.ones([1]), requires_grad=True)

        n_out_features_gnn = 512
        n_out_features = n_out_features_gnn + self.n_features_perslay
        if self.bn:
            self.bn = nn.BatchNorm1d(num_features=n_out_features)
        self.fc = nn.Linear(in_features=n_out_features, out_features=self.n_classes)

    def forward(self, data, diagrams, point_index):
        """Compute the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.data.Data
            A batch of input graph data for the GNN layers.
        diagrams
            A batch of input persistence diagrams for the PersLay layer.
        point_index : torch.Tensor
            A one-dimensional integer tensor holding the segmentation map
            for samples in the batched data, e.g. tensor([0, 0, 1, 1, 1, 2, ...]).

        Returns
        -------
        log_softmax
            The log softmax of the predictions.
        """
        x_gnn = self.weight_gnn * self.gnn_embedder(data)
        x_perslay = self.weight_perslay * self.perslay_embedder(diagrams, point_index)

        x = torch.cat([x_gnn, x_perslay], dim=1)
        if self.bn:
            x = self.bn(x)
            x = nnf.relu(x)
        x = self.fc(x)

        return nnf.log_softmax(x, dim=1)
