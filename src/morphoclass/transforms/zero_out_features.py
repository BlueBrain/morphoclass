"""Implementation of the `ZeroOutFeatures` transform."""
from __future__ import annotations


class ZeroOutFeatures:
    """Transform that sets all node features to zero."""

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
        if hasattr(data, "x"):
            data.x.zero_()
        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}()"
