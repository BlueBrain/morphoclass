"""Implementation of the branch diameter node feature extractor."""
from __future__ import annotations

import torch

from morphoclass.transforms.helper import require_field
from morphoclass.transforms.node_features.extract_node_features import (
    ExtractNodeFeatures,
)


class ExtractDiameters(ExtractNodeFeatures):
    """Extract the diameter value attached to each node."""

    @require_field("tmd_neurites")
    def extract_node_features(self, data):
        """Extract the diameter node features from data.

        Parameters
        ----------
        data
            A data sample

        Returns
        -------
        features
            Torch tensor of shape (n_nodes,)
        """
        diameters = [torch.from_numpy(apical.d) for apical in data.tmd_neurites]
        return torch.cat(diameters)
