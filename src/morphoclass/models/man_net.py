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
"""The `ManNet` model and trainer classes."""
from __future__ import annotations

import torch
import torch_geometric as pyg
from torch import nn
from torch.nn import functional as nnf

from morphoclass import layers


class ManEmbedder(nn.Module):
    """The embedder for the `ManNet` network.

    The embedder consists of two bidirectional ChebConv blocks followed
    by a global pooling layer.

    Parameters
    ----------
    n_features : int
        The number of input features.
    pool_name : {"avg", "sum", "att"}
        The type of pooling layer to use:
        - "avg": global average pooling
        - "sum": global sum pooling
        - "att": global attention pooling (trainable)
    lambda_max : float or list of float or None
        Originally the highest eigenvalue(s) of the adjacency matrix. In
        ChebConvs this value is usually computed from the adjacency matrix
        directly and used for normalization. This however doesn't work for
        non-symmetric matrices and we fix a constant value instead of computing
        it. Experiments show that there is no impact on performance.
    normalization : {None, "sym", "rw"}
        The normalization type of the graph Laplacian to use in the ChebConvs.
        Possible values:
        - None: no normalization
        - "sym": symmetric normalization
        - "rw": random walk normalization
    flow : {"target_to_source", "source_to_target"}
        The message passing flow direction in ChebConvs for directed graphs.
    edge_weight_idx : int or None
        The index of the edge feature tensor (`data.edge_attr`) to use as
        edge weights.
    """

    def __init__(
        self,
        n_features=1,
        pool_name="avg",
        lambda_max=3.0,
        normalization="sym",
        flow="target_to_source",
        edge_weight_idx=None,
    ):
        super().__init__()

        self.n_features = n_features

        self.pool_name = pool_name
        self.lambda_max = lambda_max
        self.normalization = normalization
        self.flow = flow
        self.edge_weight_idx = edge_weight_idx

        conv_kwargs = {
            "K": 5,
            "flow": self.flow,
            "normalization": self.normalization,
            "lambda_max": self.lambda_max,
        }

        self.bi1 = layers.BidirectionalBlock(n_features, 128, **conv_kwargs)
        self.bi2 = layers.BidirectionalBlock(128, 512, **conv_kwargs)
        self.relu = nn.ReLU()

        if pool_name == "avg":
            self.pool = pyg.nn.global_mean_pool
        elif pool_name == "sum":
            self.pool = pyg.nn.global_add_pool
        elif pool_name == "att":
            self.pool = layers.AttentionGlobalPool(512)
        else:
            raise ValueError(f"Unknown pooling method ({pool_name})")

    def forward(self, data):
        """Run the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        torch.Tensor
            The computed graph embeddings of the input morphologies. The
            shape is (n_samples, 512).
        """
        # Note: case in else section can be removed as soon as they implement
        # the feature, see https://github.com/pytorch/captum/issues/494
        if hasattr(data, "x"):
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch = data.batch
        else:
            # captum tensors for XAI
            if data.shape[1] == 4:
                x = data[:, 0].reshape((-1, 1))
                edge_index = data[:, 1:3].long()
                edge_index = edge_index[edge_index != -2].reshape((-1, 2)).T.long()
                edge_index[edge_index == -1] = 0  # cannot handle
                edge_index.new()
                batch = data[:, 3].reshape(-1).long()
                batch.new()
                edge_attr = None

            else:
                raise NotImplementedError(
                    "Case with different number of tensors not supported yet"
                )
        edge_weight = None
        if edge_attr is not None and self.edge_weight_idx is not None:
            edge_weight = edge_attr[:, self.edge_weight_idx]

        x = self.bi1(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.bi2(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.pool(x, batch)

        return x


class ManNetR(nn.Module):
    """The regression version of the `ManNet` classifier (no softmax).

    Also it's possible to train on a number of global features and have
    an optional batch normalization.

    Parameters
    ----------
    n_features : int
        The number of input features.
    n_global_features : int
        The number of global features.
    n_classes : int
        The number of classes. For each sample the output of the model will
        be an array of real values of length `n_classes`.
    pool_name : {"avg", "sum", "att"}
        The type of pooling layer to use:
        - "avg": global average pooling
        - "sum": global sum pooling
        - "att": global attention pooling (trainable)
    lambda_max : float or list of float or None
        Originally the highest eigenvalue(s) of the adjacency matrix. In
        ChebConvs this value is usually computed from the adjacency matrix
        directly and used for normalization. This however doesn't work for
        non-symmetric matrices and we fix a constant value instead of computing
        it. Experiments show that there is no impact on performance.
    normalization : {None, "sym", "rw"}
        The normalization type of the graph Laplacian to use in the ChebConvs.
        Possible values:
        - None: no normalization
        - "sym": symmetric normalization
        - "rw": random walk normalization
    flow : {"target_to_source", "source_to_target"}
        The message passing flow direction in ChebConvs for directed graphs.
    edge_weight_idx : int or None
        The index of the edge feature tensor (`data.edge_attr`) to use as
        edge weights.
    bn : bool
        Whether or not to apply batch normalization between the embedder and
        the classifier.
    """

    def __init__(
        self,
        n_features=1,
        n_global_features=0,
        n_classes=4,
        pool_name="avg",
        lambda_max=3.0,
        normalization="sym",
        flow="target_to_source",
        edge_weight_idx=None,
        bn=False,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_global_features = n_global_features
        self.n_classes = n_classes
        self.bn = bn

        # network backbone - feature extractor
        self.feature_extractor = ManEmbedder(
            n_features, pool_name, lambda_max, normalization, flow, edge_weight_idx
        )
        # network head - classifier
        n_out_features_gnn = 512
        n_out_features = n_out_features_gnn + n_global_features
        if bn:
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features=n_out_features),
                nn.ReLU(),
                nn.Linear(n_out_features, n_classes),
            )
        else:
            self.classifier = nn.Sequential(nn.Linear(n_out_features, n_classes))

    def forward(self, data):
        """Run the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        torch.Tensor
            The log of the prediction probabilities. The tensor has shape
            (n_samples, n_classes).
        """
        x = self.feature_extractor(data)

        if self.n_global_features > 0:
            # Note: case in else section can be removed as soon as they implement
            # the feature, see https://github.com/pytorch/captum/issues/494
            if hasattr(data, "u"):
                u = data.u
            else:
                # captum tensors for XAI
                raise NotImplementedError("Test this case")

            x = torch.cat([x, u], dim=1)

        x = self.classifier(x)

        return x


class ManNet(ManNetR):
    """The update version of the `MultiAdjNet` classifier.

    Changes:
    - custom pooling
    - edge_weights
    - ChebConvs from pytorch-geometric
    - customizable normalization
    - customizable lambda_max

    Parameters
    ----------
    n_features : int
        The number of input features.
    n_global_features : int
        The number of global features.
    n_classes : int
        The number of classes. For each sample the output of the model will
        be an array of real values of length `n_classes`.
    pool_name : {"avg", "sum", "att"}
        The type of pooling layer to use:
        - "avg": global average pooling
        - "sum": global sum pooling
        - "att": global attention pooling (trainable)
    lambda_max : float or list of float or None
        Originally the highest eigenvalue(s) of the adjacency matrix. In
        ChebConvs this value is usually computed from the adjacency matrix
        directly and used for normalization. This however doesn't work for
        non-symmetric matrices and we fix a constant value instead of computing
        it. Experiments show that there is no impact on performance.
    normalization : {None, "sym", "rw"}
        The normalization type of the graph Laplacian to use in the ChebConvs.
        Possible values:
        - None: no normalization
        - "sym": symmetric normalization
        - "rw": random walk normalization
    flow : {"target_to_source", "source_to_target"}
        The message passing flow direction in ChebConvs for directed graphs.
    edge_weight_idx : int or None
        The index of the edge feature tensor (`data.edge_attr`) to use as
        edge weights.
    bn : bool
        Whether or not to apply batch normalization between the embedder and
        the classifier.
    """

    def __init__(
        self,
        n_features=1,
        n_global_features=0,
        n_classes=4,
        pool_name="avg",
        lambda_max=3.0,
        normalization="sym",
        flow="target_to_source",
        edge_weight_idx=None,
        bn=False,
    ):
        super().__init__(
            n_features,
            n_global_features,
            n_classes,
            pool_name,
            lambda_max,
            normalization,
            flow,
            edge_weight_idx,
            bn,
        )

    def forward(self, data):
        """Run the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        torch.Tensor
            The log of the prediction probabilities. The tensor has shape
            (n_samples, n_classes).
        """
        x = super().forward(data)

        return nnf.log_softmax(x, dim=1)

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
