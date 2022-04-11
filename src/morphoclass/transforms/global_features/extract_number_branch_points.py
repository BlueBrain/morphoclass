"""Implementation of the `ExtractNumberBranchPoints` global feature extractor."""
from __future__ import annotations

import numpy as np

from morphoclass.transforms.global_features.abstract_global_feature_extractor import (
    AbstractGlobalFeatureExtractor,
)
from morphoclass.transforms.helper import require_field


class ExtractNumberBranchPoints(AbstractGlobalFeatureExtractor):
    """Extract number of branch points.

    The data should contain the field `tmd_neurites`, see the
    `ExtractTMDNeurites` class.
    """

    @require_field("tmd_neurites")
    def extract_global_feature(self, data):
        """Extract the number of branch points global feature from data.

        Parameters
        ----------
        data
            The input morphology.

        Returns
        -------
        The number of branch points in the morphology.
        """
        return sum(
            np.sum(apical.dA.sum(axis=0).A1 >= 2) for apical in data.tmd_neurites
        )
