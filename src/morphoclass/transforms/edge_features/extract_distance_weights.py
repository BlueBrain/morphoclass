"""Implementation of the distance weights edge feature extractor."""
from __future__ import annotations

import numpy as np
import torch

from morphoclass.transforms.edge_features.extract_edge_features import (
    ExtractEdgeFeatures,
)
from morphoclass.transforms.helper import require_field


class ExtractDistanceWeights(ExtractEdgeFeatures):
    """The distance weights edge feature extractor.

    The feature is computed as exp(-len(edge)^2 / scale^2).

    Parameters
    ----------
    scale : float
        The scale factor for the formula above.
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    @require_field("tmd_neurites")
    @require_field("edge_index")
    def extract_edge_features(self, data):
        """Extract the distance weights edge features from given data sample.

        The feature is computed as exp(-len(edge)^2 / scale^2).

        Parameters
        ----------
        data : torch_geometric.data.Data
            A data sample.

        Returns
        -------
        edge_attr : torch.tensor
            The extracted distance weights edge features.
        """
        coords = [
            np.transpose([apical.x, apical.y, apical.z]) for apical in data.tmd_neurites
        ]
        coords = np.concatenate(coords, axis=0)
        distances = coords[data.edge_index[0]] - coords[data.edge_index[1]]
        features = np.exp(-np.sum(distances**2, axis=1) / self.scale**2)

        return torch.tensor(features, dtype=torch.float32)
