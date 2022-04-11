"""Implementation of the `AddRandomPointsToReductionMask` transform."""
from __future__ import annotations

import random

import numpy as np

from morphoclass.transforms.helper import require_field
from morphoclass.utils import print_warning


class AddRandomPointsToReductionMask:
    """A Transform that adds a fixed number of random points to the apical mask.

    Parameters
    ----------
    n_points : int
        The number of random points to add to the apical mask.
    """

    def __init__(self, n_points):

        self.n_points = n_points

    @require_field("tmd_neurites_masks")
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
        for _ in range(self.n_points):
            pass
            # If all points in all apicals have already been added
            # then warn user and break
            if all(False not in mask for mask in data.tmd_neurites_masks):
                print_warning(
                    "All possible points have already been added. "
                    "Cannot add any more points"
                )
                break

            mask = random.choice(data.tmd_neurites_masks)
            while False not in mask:
                mask = random.choice(data.tmd_neurites_masks)
            mask[np.random.choice(np.where(~mask)[0])] = True
        return data
