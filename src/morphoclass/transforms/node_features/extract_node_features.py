"""Abstract base class for node feature extractors."""
from __future__ import annotations

import abc

import torch


class ExtractNodeFeatures(abc.ABC):
    """Abstract base class for arbitrary node feature extraction."""

    @abc.abstractmethod
    def extract_node_features(self, data):
        """Extract some node features from data.

        Parameters
        ----------
        data
            A data sample

        Returns
        -------
        features
            Torch tensor of shape (n_nodes, n_features)
        """

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
        features = self.extract_node_features(data)

        if features is not None:
            if len(features.size()) == 1:
                features = features.unsqueeze(dim=1)
            features = features.to(dtype=torch.float32)

            if hasattr(data, "x") and data.x is not None:
                data.x = torch.cat([data.x, features], dim=1)
            else:
                data.x = features

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}()"
