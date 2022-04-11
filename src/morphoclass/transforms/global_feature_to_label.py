"""Implementation of the `GlobalFeatureToLabel` transform."""
from __future__ import annotations


class GlobalFeatureToLabel:
    """Convert a global feature to label.

    Parameters
    ----------
    global_feature_index : int
        The index of the global feature that shall be used.
    """

    def __init__(self, global_feature_index):
        self.global_feature_index = global_feature_index

    def __call__(self, data):
        """Apply the morphology transformation.

        Parameters
        ----------
        data
            The input morphology data sample.

        Returns
        -------
        data
            The modified morphology data sample.
        """
        data.y = data.u[0, self.global_feature_index].item()

        return data
