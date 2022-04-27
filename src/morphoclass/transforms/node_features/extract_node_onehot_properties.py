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
"""Various one-hot property node-feature extractors.

This name is misleading as what the transforms in this module extract are rather
boolean features and not one-hot matrices.
"""
from __future__ import annotations

import abc

import torch

from morphoclass.transforms.helper import require_field


class ExtractOneHotProperty(abc.ABC):
    """Base class for boolean node feature extractors.

    The name of this class is misleading and should be changed to reflect that
    the extracted features are not one-hot encoded but are boolean values
    (encoded as 0.0 and 1.0 float values).
    """

    @abc.abstractmethod
    def extract_property(self, apical):
        """Extract boolean node features for a given apical tree.

        Parameters
        ----------
        apical : tmd.Tree.Tree
            An apical tree.

        Returns
        -------
        features
            Torch tensor of shape (n_nodes,) and dtype float32 containing
            values 0.0 and 1.0.
        """
        pass

    @require_field("tmd_neurites")
    def __call__(self, data):
        """Apply the transform to a given data sample to extract node features.

        Parameters
        ----------
        data
            A morphology data sample.

        Returns
        -------
        data
            The processed data sample with the extracted node features
            attached to it.
        """
        features_list = [self.extract_property(apical) for apical in data.tmd_neurites]

        if len(features_list) == 0:
            features_list = [torch.empty(size=(0,), dtype=torch.float32)]
        features = torch.cat(features_list).unsqueeze(dim=1)

        if hasattr(data, "x") and data.x is not None:
            data.x = torch.cat([data.x, features], dim=1)
        else:
            data.x = features

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}()"


class ExtractIsRoot(ExtractOneHotProperty):
    """For each node determine if it's the root node."""

    def extract_property(self, apical):
        """Extract boolean node features for a given apical tree.

        Parameters
        ----------
        apical : tmd.Tree.Tree
            An apical tree.

        Returns
        -------
        features
            Torch tensor of shape (n_nodes,) and dtype float32 containing
            the value 1.0 if the node is a root node, otherwise 0.0.
        """
        result = (apical.dA.sum(axis=1) == 0).A1
        return torch.tensor(result, dtype=torch.float32)


class ExtractIsLeaf(ExtractOneHotProperty):
    """For each node determine if it's a leaf node."""

    def extract_property(self, apical):
        """Extract boolean node features for a given apical tree.

        Parameters
        ----------
        apical : tmd.Tree.Tree
            An apical tree.

        Returns
        -------
        features
            Torch tensor of shape (n_nodes,) and dtype float32 containing
            the value 1.0 if the node is a leaf node, otherwise 0.0.
        """
        result = (apical.dA.sum(axis=0) == 0).A1
        return torch.tensor(result, dtype=torch.float32)


class ExtractIsIntermediate(ExtractOneHotProperty):
    """For each node determine if it's an intermediate node.

    An intermediate node is one that has exactly one parent and one child,
    i.e. a node that looks like that: --o--.
    """

    def extract_property(self, apical):
        """Extract boolean node features for a given apical tree.

        Parameters
        ----------
        apical : tmd.Tree.Tree
            An apical tree.

        Returns
        -------
        features
            Torch tensor of shape (n_nodes,) and dtype float32 containing
            the value 1.0 if the node is an intermediate node, otherwise 0.0.
        """
        one_child = (apical.dA.sum(axis=0) == 1).A1
        one_parent = (apical.dA.sum(axis=1) == 1).A1
        return torch.tensor(one_child & one_parent, dtype=torch.float32)


class ExtractIsBranching(ExtractOneHotProperty):
    """For each node determine if it's a branching node."""

    def extract_property(self, apical):
        """Extract boolean node features for a given apical tree.

        Parameters
        ----------
        apical : tmd.Tree.Tree
            An apical tree.

        Returns
        -------
        features
            Torch tensor of shape (n_nodes,) and dtype float32 containing
            the value 1.0 if the node is a branching node, otherwise 0.0.
        """
        result = (apical.dA.sum(axis=0) > 1).A1
        return torch.tensor(result, dtype=torch.float32)
