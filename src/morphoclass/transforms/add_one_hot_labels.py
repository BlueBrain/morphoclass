"""Implementation of the `AddOneHotLabels` transform."""
from __future__ import annotations

import torch

from morphoclass.transforms.helper import require_field


class AddOneHotLabels:
    """Add one-hot labels to data.

    The prerequisite is that data already contains numerical sparse labels.
    These sparse labels are transformed into one-hot labels using the function
    provided in the constructor.

    Parameters
    ----------
    fn_get_oh_label : callable
        Function that maps sparse labels to one-hot labels.
    """

    def __init__(self, fn_get_oh_label):
        self.fn_get_oh_label = fn_get_oh_label

    @require_field("y")
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
        oh_label = self.fn_get_oh_label(data.y)
        data.y_oh = torch.tensor(oh_label).unsqueeze(0)

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}(fn_get_oh_label={self.fn_get_oh_label})"
