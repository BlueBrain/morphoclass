"""Implementation of the node feature extractor for constant node features."""
from __future__ import annotations

import torch

from morphoclass.transforms.helper import require_field
from morphoclass.transforms.node_features.extract_node_features import (
    ExtractNodeFeatures,
)


class ExtractConstFeature(ExtractNodeFeatures):
    """Extractor that assigns the constant feature value 1.0 to each node."""

    @require_field("tmd_neurites")
    def extract_node_features(self, data):
        """Extract the constant node feature value 1.0 for each node.

        Parameters
        ----------
        data
            A data sample

        Returns
        -------
        features
            Torch tensor of shape (n_nodes,)
        """
        ones = [torch.ones(apical.d.shape) for apical in data.tmd_neurites]
        return torch.cat(ones)
