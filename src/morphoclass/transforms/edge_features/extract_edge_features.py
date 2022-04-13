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
"""Implementation of the base class for edge feature extractors."""
from __future__ import annotations

import abc

import torch


class ExtractEdgeFeatures(abc.ABC):
    """Base class for edge feature extractors."""

    @abc.abstractmethod
    def extract_edge_features(self, data):
        """Extract some edge features from given data sample.

        Parameters
        ----------
        data : torch_geometric.data.Data
            A data sample.

        Returns
        -------
        edge_attr : torch.tensor
            The extracted edge attributes, shape (n_edges, n_features).
        """

    def __call__(self, data):
        """Apply the transform to given data sample.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The data sample to transform.

        Returns
        -------
        data : torch_geometric.data.Data
            The transformed data sample.
        """
        edge_attr = self.extract_edge_features(data)

        if edge_attr is not None:
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr.unsqueeze(dim=1)
            edge_attr = edge_attr.to(dtype=torch.float32)

            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                data.edge_attr = torch.cat([data.edge_attr, edge_attr], dim=1)
            else:
                data.edge_attr = edge_attr

        return data

    def __repr__(self):
        """Get representation of class instance."""
        return f"{self.__class__.__name__}()"
